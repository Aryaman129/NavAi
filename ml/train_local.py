#!/usr/bin/env python3
"""
Local training script optimized for RTX 4050 (6GB VRAM)
Trains IMU speed estimation models with CUDA acceleration
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.data_loader import DataLoader as NavAIDataLoader
from models.speed_estimator import SpeedCNN, SpeedLSTM, WindowGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalTrainer:
    """Optimized trainer for RTX 4050 with 6GB VRAM"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Memory optimization for 6GB VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.8)
        
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    def load_data(self):
        """Load and preprocess data"""
        logger.info("Loading data...")
        
        data_loader = NavAIDataLoader(target_sample_rate=self.config.sample_rate)
        
        # Check for existing data
        data_paths = {
            'navai': self.config.data_dir / 'navai_logs'
        }
        
        # Load data
        df = data_loader.load_combined_dataset({k: str(v) for k, v in data_paths.items()})
        
        if df.empty:
            logger.warning("No data found. Creating synthetic data for testing...")
            df = self._create_synthetic_data()
        
        logger.info(f"Loaded {len(df)} samples")
        
        # Filter valid GPS data
        valid_data = df[
            (df['gps_speed_mps'] >= 0) & 
            (df['gps_speed_mps'] <= 50) &  # Reasonable speed range
            (df['gps_speed_mps'].notna())
        ].copy()
        
        logger.info(f"Valid samples: {len(valid_data)} ({len(valid_data)/len(df)*100:.1f}%)")
        
        return valid_data
    
    def _create_synthetic_data(self):
        """Create synthetic data for testing when no real data is available"""
        logger.info("Creating synthetic IMU data for testing...")
        
        n_samples = 10000
        time_ns = np.arange(n_samples) * int(1e9 / self.config.sample_rate)
        
        # Simulate realistic IMU patterns
        t = np.arange(n_samples) / self.config.sample_rate
        
        # Simulate vehicle motion with varying speed
        speed_profile = 5 + 10 * np.sin(0.1 * t) + 5 * np.sin(0.05 * t)
        speed_profile = np.maximum(speed_profile, 0)  # No negative speeds
        
        # Simulate accelerometer (with gravity and motion)
        accel_x = 0.2 * np.sin(0.5 * t) + 0.1 * np.random.randn(n_samples)
        accel_y = 0.3 * np.cos(0.3 * t) + 0.1 * np.random.randn(n_samples)
        accel_z = -9.81 + 0.5 * np.sin(0.2 * t) + 0.2 * np.random.randn(n_samples)
        
        # Simulate gyroscope (angular velocities)
        gyro_x = 0.1 * np.sin(0.4 * t) + 0.05 * np.random.randn(n_samples)
        gyro_y = 0.1 * np.cos(0.6 * t) + 0.05 * np.random.randn(n_samples)
        gyro_z = 0.2 * np.sin(0.3 * t) + 0.05 * np.random.randn(n_samples)
        
        # Create DataFrame
        synthetic_df = pd.DataFrame({
            'timestamp_ns': time_ns,
            'accel_x': accel_x,
            'accel_y': accel_y,
            'accel_z': accel_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z,
            'mag_x': np.zeros(n_samples),
            'mag_y': np.zeros(n_samples),
            'mag_z': np.zeros(n_samples),
            'qw': np.ones(n_samples),
            'qx': np.zeros(n_samples),
            'qy': np.zeros(n_samples),
            'qz': np.zeros(n_samples),
            'gps_lat': np.zeros(n_samples),
            'gps_lon': np.zeros(n_samples),
            'gps_speed_mps': speed_profile,
            'device': 'synthetic',
            'source': 'synthetic'
        })
        
        return synthetic_df
    
    def prepare_windows(self, df):
        """Create training windows"""
        logger.info("Creating training windows...")
        
        window_generator = WindowGenerator(
            window_size_sec=self.config.window_size_sec,
            stride_sec=self.config.stride_sec,
            sample_rate=self.config.sample_rate,
            feature_cols=['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'],
            target_col='gps_speed_mps'
        )
        
        X, y = window_generator.create_windows(df)
        
        if len(X) == 0:
            raise ValueError("No valid windows created. Check your data.")
        
        logger.info(f"Created {len(X)} windows of shape {X.shape[1:]}")
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_val, y_train, y_val
    
    def create_model(self, input_shape):
        """Create model optimized for RTX 4050"""
        if self.config.model_type == 'cnn':
            model = SpeedCNN(
                input_channels=input_shape[-1],
                hidden_dim=self.config.hidden_dim,
                dropout=self.config.dropout
            )
        elif self.config.model_type == 'lstm':
            model = SpeedLSTM(
                input_size=input_shape[-1],
                hidden_size=self.config.hidden_dim,
                dropout=self.config.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        return model.to(self.device)
    
    def train(self, X_train, X_val, y_train, y_val):
        """Train the model"""
        logger.info("Starting training...")
        
        # Convert to tensors
        X_train_torch = torch.FloatTensor(X_train).to(self.device)
        y_train_torch = torch.FloatTensor(y_train).to(self.device)
        X_val_torch = torch.FloatTensor(X_val).to(self.device)
        y_val_torch = torch.FloatTensor(y_val).to(self.device)
        
        # Create data loaders (smaller batch size for 6GB VRAM)
        train_dataset = TensorDataset(X_train_torch, y_train_torch)
        val_dataset = TensorDataset(X_val_torch, y_val_torch)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Create model
        model = self.create_model(X_train.shape[1:])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), self.config.output_dir / 'best_model.pth')
            
            if (epoch + 1) % 5 == 0:
                logger.info(f'Epoch [{epoch+1}/{self.config.num_epochs}], '
                          f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Load best model
        model.load_state_dict(torch.load(self.config.output_dir / 'best_model.pth', weights_only=False))
        
        return model, train_losses, val_losses
    
    def evaluate(self, model, X_val, y_val):
        """Evaluate the model"""
        logger.info("Evaluating model...")
        
        model.eval()
        X_val_torch = torch.FloatTensor(X_val).to(self.device)
        
        with torch.no_grad():
            predictions = model(X_val_torch).squeeze().cpu().numpy()
        
        # Calculate metrics
        mse = mean_squared_error(y_val, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, predictions)
        
        mean_speed = np.mean(y_val)
        rmse_percent = (rmse / mean_speed) * 100
        mae_percent = (mae / mean_speed) * 100
        
        logger.info(f"Evaluation Results:")
        logger.info(f"RMSE: {rmse:.3f} m/s ({rmse_percent:.1f}%)")
        logger.info(f"MAE: {mae:.3f} m/s ({mae_percent:.1f}%)")
        logger.info(f"Mean speed: {mean_speed:.3f} m/s")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'rmse_percent': rmse_percent,
            'mae_percent': mae_percent,
            'predictions': predictions,
            'actuals': y_val
        }

def main():
    parser = argparse.ArgumentParser(description='Train NavAI speed estimation model locally')
    parser.add_argument('--data-dir', type=Path, default=Path('data'), help='Data directory')
    parser.add_argument('--output-dir', type=Path, default=Path('models'), help='Output directory')
    parser.add_argument('--model-type', choices=['cnn', 'lstm'], default='cnn', help='Model type')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (optimized for 6GB VRAM)')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--window-size-sec', type=float, default=1.5, help='Window size in seconds')
    parser.add_argument('--stride-sec', type=float, default=0.25, help='Stride in seconds')
    parser.add_argument('--sample-rate', type=int, default=100, help='Sample rate')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = LocalTrainer(args)
    
    try:
        # Load data
        df = trainer.load_data()
        
        # Prepare windows
        X_train, X_val, y_train, y_val = trainer.prepare_windows(df)
        
        # Train model
        model, train_losses, val_losses = trainer.train(X_train, X_val, y_train, y_val)
        
        # Evaluate
        results = trainer.evaluate(model, X_val, y_val)
        
        # Save results
        results_path = args.output_dir / 'training_results.txt'
        with open(results_path, 'w') as f:
            f.write(f"Training Results\n")
            f.write(f"================\n")
            f.write(f"Model Type: {args.model_type}\n")
            f.write(f"RMSE: {results['rmse']:.3f} m/s ({results['rmse_percent']:.1f}%)\n")
            f.write(f"MAE: {results['mae']:.3f} m/s ({results['mae_percent']:.1f}%)\n")
            f.write(f"Training completed successfully!\n")
        
        logger.info(f"Training completed! Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
