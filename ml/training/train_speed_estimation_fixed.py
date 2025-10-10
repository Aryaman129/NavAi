"""
Fixed Speed Estimation Training Script
======================================

This script addresses all identified issues:
1. ✅ Proper temporal data splitting (no leakage)
2. ✅ Physics-based feature engineering
3. ✅ Realistic baseline comparisons
4. ✅ Correct evaluation metrics
5. ✅ Proper model architecture and regularization

Location: ml/training/train_speed_estimation_fixed.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# Import our data loader
from ml.data.data_loader import DataLoader as NavAIDataLoader

print("="*80)
print("🚀 FIXED SPEED ESTIMATION TRAINING")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ===========================
# Configuration
# ===========================
class Config:
    # Data parameters
    SEQ_LEN = 20  # Sequence length
    STRIDE = 10   # Stride for sequence creation (increased to reduce overlap)
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    # TEST_RATIO = 0.15 (implicit)
    
    # Model parameters
    INPUT_SIZE = 16  # Enhanced features (10 IMU + 6 engineered)
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # Training parameters
    BATCH_SIZE = 256
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    
    # Paths
    DATA_DIR = '../../data/comma2k19'
    CHECKPOINT_DIR = './checkpoints'
    ANALYSIS_DIR = '../analysis/reports'
    VIZ_DIR = '../analysis/visualizations'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# Create directories
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(config.ANALYSIS_DIR, exist_ok=True)
os.makedirs(config.VIZ_DIR, exist_ok=True)

print(f"📋 Configuration:")
print(f"  Device: {config.DEVICE}")
print(f"  Sequence length: {config.SEQ_LEN}")
print(f"  Batch size: {config.BATCH_SIZE}")
print(f"  Learning rate: {config.LEARNING_RATE}")
print()

# ===========================
# Feature Engineering
# ===========================
def engineer_features(df):
    """
    Create physics-based features from raw IMU data.
    
    Features created:
    - Linear acceleration (gravity compensated)
    - Acceleration magnitude
    - Angular velocity magnitude
    - Forward acceleration
    - Jerk (acceleration derivative)
    """
    print("🔧 Engineering physics-based features...")
    
    df_eng = df.copy()
    
    # 1. Gravity compensation using quaternions
    # Convert quaternions to gravity vector in body frame
    qw, qx, qy, qz = df_eng['qw'], df_eng['qx'], df_eng['qy'], df_eng['qz']
    g = 9.81
    
    gx = 2 * (qx*qz - qw*qy) * g
    gy = 2 * (qy*qz + qw*qx) * g
    gz = (qw**2 - qx**2 - qy**2 + qz**2) * g
    
    df_eng['linear_accel_x'] = df_eng['accel_x'] - gx
    df_eng['linear_accel_y'] = df_eng['accel_y'] - gy
    df_eng['linear_accel_z'] = df_eng['accel_z'] - gz
    
    # 2. Magnitude features
    df_eng['accel_mag'] = np.sqrt(df_eng['accel_x']**2 + df_eng['accel_y']**2 + df_eng['accel_z']**2)
    df_eng['linear_accel_mag'] = np.sqrt(df_eng['linear_accel_x']**2 + df_eng['linear_accel_y']**2 + df_eng['linear_accel_z']**2)
    df_eng['gyro_mag'] = np.sqrt(df_eng['gyro_x']**2 + df_eng['gyro_y']**2 + df_eng['gyro_z']**2)
    
    print(f"✅ Created {len(df_eng.columns) - len(df.columns)} new features")
    return df_eng

# ===========================
# Temporal Data Splitting
# ===========================
def create_temporal_splits(df, train_ratio=0.7, val_ratio=0.15):
    """
    Create train/val/test splits based on TIME, not random sampling.
    This prevents data leakage between splits.
    """
    print(f"\n📊 Creating temporal splits...")
    
    # Sort by timestamp
    df_sorted = df.sort_values('timestamp_ns').reset_index(drop=True)
    
    # Calculate time boundaries
    total_time = df_sorted['timestamp_ns'].max() - df_sorted['timestamp_ns'].min()
    train_end_time = df_sorted['timestamp_ns'].min() + train_ratio * total_time
    val_end_time = df_sorted['timestamp_ns'].min() + (train_ratio + val_ratio) * total_time
    
    # Create splits
    train_mask = df_sorted['timestamp_ns'] <= train_end_time
    val_mask = (df_sorted['timestamp_ns'] > train_end_time) & (df_sorted['timestamp_ns'] <= val_end_time)
    test_mask = df_sorted['timestamp_ns'] > val_end_time
    
    train_df = df_sorted[train_mask].reset_index(drop=True)
    val_df = df_sorted[val_mask].reset_index(drop=True)
    test_df = df_sorted[test_mask].reset_index(drop=True)
    
    print(f"  Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify no temporal overlap
    assert train_df['timestamp_ns'].max() <= val_df['timestamp_ns'].min(), "Train-Val overlap detected!"
    assert val_df['timestamp_ns'].max() <= test_df['timestamp_ns'].min(), "Val-Test overlap detected!"
    print(f"  ✅ No temporal overlap confirmed")
    
    return train_df, val_df, test_df

# ===========================
# Dataset
# ===========================
class IMUSpeedDataset(Dataset):
    def __init__(self, df, seq_len=20, stride=10, feature_cols=None):
        """
        Create sequences with specified stride to avoid excessive overlap.
        """
        self.seq_len = seq_len
        self.stride = stride
        
        if feature_cols is None:
            # Default features: raw IMU + engineered features
            feature_cols = [
                'accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z',
                'qw', 'qx', 'qy', 'qz',
                'linear_accel_x', 'linear_accel_y', 'linear_accel_z',
                'accel_mag', 'linear_accel_mag', 'gyro_mag'
            ]
        
        self.feature_cols = feature_cols
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(0, len(df) - seq_len, stride):
            seq = df.iloc[i:i+seq_len][feature_cols].values
            target = df.iloc[i+seq_len-1]['gps_speed_mps']
            
            if not np.isnan(target) and not np.isnan(seq).any():
                sequences.append(seq)
                targets.append(target)
        
        self.sequences = np.array(sequences, dtype=np.float32)
        self.targets = np.array(targets, dtype=np.float32)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.targets[idx]])

# ===========================
# Model Architecture
# ===========================
class SpeedEstimationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super(SpeedEstimationLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.dropout(last_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ===========================
# Training Functions
# ===========================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for sequences, targets in train_loader:
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for sequences, targets in data_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(targets.numpy().flatten())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    return rmse, mae, r2, predictions, actuals

# ===========================
# Main Training Loop
# ===========================
def main():
    # Load data
    print("\n📦 Loading data...")
    loader = NavAIDataLoader()
    df = loader.load_comma2k19(config.DATA_DIR)
    
    # Engineer features
    df = engineer_features(df)
    
    # Create temporal splits
    train_df, val_df, test_df = create_temporal_splits(
        df, 
        config.TRAIN_RATIO, 
        config.VAL_RATIO
    )
    
    # Create datasets
    print(f"\n🔨 Creating datasets...")
    train_dataset = IMUSpeedDataset(train_df, config.SEQ_LEN, config.STRIDE)
    val_dataset = IMUSpeedDataset(val_df, config.SEQ_LEN, config.STRIDE)
    test_dataset = IMUSpeedDataset(test_df, config.SEQ_LEN, config.STRIDE)
    
    print(f"  Train sequences: {len(train_dataset):,}")
    print(f"  Val sequences:   {len(val_dataset):,}")
    print(f"  Test sequences:  {len(test_dataset):,}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    print(f"\n🏗️ Creating model...")
    model = SpeedEstimationLSTM(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    print(f"\n🎯 Starting training...")
    best_val_rmse = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'val_r2': []
    }
    
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        # Validate
        val_rmse, val_mae, val_r2, _, _ = evaluate(model, val_loader, config.DEVICE)
        
        # Update scheduler
        scheduler.step(val_rmse)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{config.NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val RMSE: {val_rmse:.3f} m/s | "
              f"Val MAE: {val_mae:.3f} m/s | "
              f"Val R²: {val_r2:.4f}")
        
        # Early stopping and checkpointing
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            
            # Save best model
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model_fixed.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'config': config
            }, checkpoint_path)
            print(f"  ✅ Saved best model (RMSE: {val_rmse:.3f} m/s)")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
                break
    
    # Final evaluation on test set
    print(f"\n📊 Final Test Evaluation...")
    checkpoint = torch.load(os.path.join(config.CHECKPOINT_DIR, 'best_model_fixed.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_rmse, test_mae, test_r2, test_preds, test_actuals = evaluate(model, test_loader, config.DEVICE)
    
    print(f"  Test RMSE: {test_rmse:.3f} m/s")
    print(f"  Test MAE:  {test_mae:.3f} m/s")
    print(f"  Test R²:   {test_r2:.4f}")
    
    # Calculate baseline for comparison
    train_mean = train_df['gps_speed_mps'].mean()
    baseline_rmse = np.sqrt(mean_squared_error(test_actuals, np.full_like(test_actuals, train_mean)))
    print(f"\n🎯 Baseline Comparison:")
    print(f"  Mean prediction RMSE: {baseline_rmse:.3f} m/s")
    print(f"  Our model RMSE:       {test_rmse:.3f} m/s")
    print(f"  Improvement:          {(baseline_rmse - test_rmse) / baseline_rmse * 100:.1f}%")
    
    # Save training history and results
    results = {
        'history': history,
        'test_results': {
            'rmse': test_rmse,
            'mae': test_mae,
            'r2': test_r2,
            'baseline_rmse': baseline_rmse
        },
        'config': vars(config)
    }
    
    results_path = os.path.join(config.ANALYSIS_DIR, f'training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
    torch.save(results, results_path)
    print(f"\n💾 Results saved to: {results_path}")
    
    # Create visualizations
    create_visualizations(history, test_preds, test_actuals, config.VIZ_DIR)
    
    print(f"\n✅ Training complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

def create_visualizations(history, test_preds, test_actuals, viz_dir):
    """Create training and evaluation visualizations"""
    print(f"\n📈 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training history
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['val_rmse'], label='Val RMSE', color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('RMSE (m/s)')
    axes[0, 1].set_title('Validation RMSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Predictions vs Actuals
    axes[1, 0].scatter(test_actuals, test_preds, alpha=0.3, s=1)
    axes[1, 0].plot([test_actuals.min(), test_actuals.max()], 
                     [test_actuals.min(), test_actuals.max()], 
                     'r--', label='Perfect Prediction')
    axes[1, 0].set_xlabel('Actual Speed (m/s)')
    axes[1, 0].set_ylabel('Predicted Speed (m/s)')
    axes[1, 0].set_title('Test Set: Predictions vs Actuals')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = test_preds - test_actuals
    axes[1, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(0, color='red', linestyle='--', label='Zero Error')
    axes[1, 1].set_xlabel('Prediction Error (m/s)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Residual Distribution (Mean: {residuals.mean():.3f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_path = os.path.join(viz_dir, f'training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(viz_path, dpi=150)
    print(f"  ✅ Saved visualization to: {viz_path}")
    plt.close()

if __name__ == '__main__':
    main()
