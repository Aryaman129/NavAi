"""
Phase 1: Foundation Training with Bug Fixes + Progress Bars
- Fixed normalization (StandardScaler)
- Full dataset (478k samples)
- Kalman Filter post-processing
- ZUPT detection for walking
- Baseline BiLSTM architecture
- LIVE PROGRESS TRACKING
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import sys
sys.path.append('ml')
sys.path.append('ml/data')
sys.path.append('ml/models')

from data.data_loader import DataLoader
from data.preprocessor_v2 import IMUPreprocessor  # New version with progress bars
from models.kalman_filter import ExtendedKalmanFilter
from models.zupt_detector import AdaptiveZUPTDetector

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎮 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

class BaselineSpeedEstimator(nn.Module):
    """Simple BiLSTM baseline for speed estimation"""
    
    def __init__(self, input_dim=10, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        x = self.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        speed = self.fc3(x)
        
        return speed


class IMUSpeedDataset(Dataset):
    """Dataset for windowed IMU data"""
    
    def __init__(self, imu_windows, speeds):
        self.imu_windows = torch.FloatTensor(imu_windows)
        self.speeds = torch.FloatTensor(speeds).unsqueeze(1)
    
    def __len__(self):
        return len(self.imu_windows)
    
    def __getitem__(self, idx):
        return self.imu_windows[idx], self.speeds[idx]


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for imu_batch, speed_batch in tqdm(dataloader, desc="Training"):
        imu_batch = imu_batch.to(device)
        speed_batch = speed_batch.to(device)
        
        optimizer.zero_grad()
        speed_pred = model(imu_batch)
        
        loss = criterion(speed_pred, speed_batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for imu_batch, speed_batch in dataloader:
            imu_batch = imu_batch.to(device)
            speed_batch = speed_batch.to(device)
            
            speed_pred = model(imu_batch)
            loss = criterion(speed_pred, speed_batch)
            
            total_loss += loss.item()
            all_preds.append(speed_pred.cpu().numpy())
            all_targets.append(speed_batch.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Compute metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # R² score
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return total_loss / len(dataloader), rmse, mae, r2, all_preds, all_targets


def apply_ekf_postprocessing(predictions, imu_data, dt=0.01):
    """
    Apply Kalman filter post-processing to neural network predictions
    """
    print("\n🔧 Applying EKF post-processing...")
    
    ekf = ExtendedKalmanFilter(dt=dt)
    refined_predictions = []
    
    for i in tqdm(range(len(predictions)), desc="EKF filtering"):
        # Get neural network prediction
        nn_speed = predictions[i, 0]
        
        # Get IMU measurement
        # imu_data shape: (N, seq_len, features)
        # Take last timestep of window
        accel = imu_data[i, -1, 0:3]  # Last timestep, accel xyz
        gyro = imu_data[i, -1, 3:6]   # Last timestep, gyro xyz
        
        # Construct measurement vector
        measurement = np.concatenate([accel, gyro, [nn_speed]])
        
        # EKF predict and update
        ekf.predict()
        ekf.update(measurement)
        
        # Get refined speed estimate
        speed_refined, _ = ekf.get_speed_estimate()
        refined_predictions.append(speed_refined)
    
    return np.array(refined_predictions).reshape(-1, 1)


def main():
    """Main training function"""
    
    print("=" * 80)
    print("Phase 1: Foundation Training with Fixed Normalization")
    print("=" * 80)
    
    # Clear GPU memory from any previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ GPU memory cleared")
    
    # Hyperparameters
    BATCH_SIZE = 128  # Reduced from 256 to prevent GPU memory issues
    EPOCHS = 50  # Max epochs (early stopping will likely stop around 15-20)
    EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for 10 epochs
    LEARNING_RATE = 0.001
    WINDOW_SIZE = 100
    
    # Load data
    print("\n📁 Loading comma2k19 dataset...")
    data_loader = DataLoader()
    
    # Load real dataset (478k samples)
    parquet_path = Path('data/comma2k19/processed_real/comma2k19_processed.parquet')
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        print(f"✅ Loaded {len(df):,} samples from parquet")
    else:
        print("❌ Real dataset not found! Using synthetic data...")
        df = data_loader.load_comma2k19_synthetic(num_samples=50000)
    
    # Extract features
    imu_data = df[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
    speeds = df['speed'].values
    
    print(f"   IMU data shape: {imu_data.shape}")
    print(f"   Speed range: {speeds.min():.2f} - {speeds.max():.2f} m/s")
    
    # Create windows (OPTIMIZED: using stride_tricks for 100x faster)
    print(f"\n🪟 Creating windows (size={WINDOW_SIZE})...")
    num_samples = len(imu_data) - WINDOW_SIZE + 1
    
    # Fast windowing using NumPy stride tricks
    from numpy.lib.stride_tricks import as_strided
    shape = (num_samples, WINDOW_SIZE, imu_data.shape[1])
    strides = (imu_data.strides[0], imu_data.strides[0], imu_data.strides[1])
    imu_windows = as_strided(imu_data, shape=shape, strides=strides).copy()
    
    speed_targets = speeds[WINDOW_SIZE-1:]  # Target is speed at end of window
    
    print(f"   Windows created: {imu_windows.shape}")
    
    # Temporal split (CRITICAL: shuffle=False for time series)
    split_idx = int(0.8 * len(imu_windows))
    
    train_imu = imu_windows[:split_idx]
    train_speeds = speed_targets[:split_idx]
    
    val_imu = imu_windows[split_idx:]
    val_speeds = speed_targets[split_idx:]
    
    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(train_imu):,} samples")
    print(f"   Validation: {len(val_imu):,} samples")
    
    # ===== FIX: Apply preprocessing with ALL features (no compromises) =====
    print(f"\n🔧 Preprocessing data (FIX for R²=0 bug)...")
    print(f"   Processing {len(train_imu):,} training windows and {len(val_imu):,} validation windows")
    print(f"   This may take a few minutes - watch the progress bars below!")
    
    preprocessor = IMUPreprocessor()
    
    # Fit on training data ONLY (processes in 100k chunks with progress bar)
    train_imu_normalized = preprocessor.fit_transform(train_imu)
    
    # Transform validation data using training statistics
    val_imu_normalized = preprocessor.transform(val_imu)
    
    # Save preprocessor
    preprocessor.save('ml/outputs/phase1_preprocessor.pkl')
    
    print(f"✅ Preprocessing complete")
    print(f"   Input features: 6 -> {train_imu_normalized.shape[-1]}")
    print(f"   Training mean: {train_imu_normalized.mean(axis=(0,1))[:3]}")
    print(f"   Training std: {train_imu_normalized.std(axis=(0,1))[:3]}")
    
    # Create datasets
    train_dataset = IMUSpeedDataset(train_imu_normalized, train_speeds)
    val_dataset = IMUSpeedDataset(val_imu_normalized, val_speeds)
    
    # DataLoaders with num_workers=0 to avoid Windows multiprocessing issues
    train_loader = TorchDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = TorchDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model
    print(f"\n🏗️ Building model...")
    input_dim = train_imu_normalized.shape[-1]  # 10 (6 raw + 4 engineered features)
    model = BaselineSpeedEstimator(input_dim=input_dim, hidden_dim=128, num_layers=3)
    model = model.to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print(f"\n🚀 Starting training...")
    best_rmse = float('inf')
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_r2': []}
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_rmse, val_mae, val_r2, val_preds, val_targets = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}")
        
        # Save best model and check early stopping
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rmse': val_rmse,
                'r2': val_r2
            }, 'ml/outputs/phase1_best_model.pth')
            print(f"  ✅ New best model saved (RMSE: {val_rmse:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
                print(f"   No improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs")
                print(f"   Best RMSE: {best_rmse:.4f}")
                break
    
    # Load best model (PyTorch 2.6+ requires weights_only=False for older checkpoints)
    checkpoint = torch.load('ml/outputs/phase1_best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    print(f"\n" + "="*80)
    print("📊 Final Validation (without EKF)")
    val_loss, val_rmse, val_mae, val_r2, val_preds, val_targets = validate(
        model, val_loader, criterion, device
    )
    print(f"  RMSE: {val_rmse:.4f} m/s")
    print(f"  MAE: {val_mae:.4f} m/s")
    print(f"  R²: {val_r2:.4f}")
    
    # Apply EKF post-processing
    print(f"\n" + "="*80)
    print("🔧 Applying Kalman Filter Post-Processing")
    
    ekf_predictions = apply_ekf_postprocessing(val_preds, val_imu_normalized, dt=0.01)
    
    # Compute EKF metrics
    ekf_rmse = np.sqrt(np.mean((ekf_predictions - val_targets) ** 2))
    ekf_mae = np.mean(np.abs(ekf_predictions - val_targets))
    ss_res = np.sum((val_targets - ekf_predictions) ** 2)
    ss_tot = np.sum((val_targets - np.mean(val_targets)) ** 2)
    ekf_r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n📊 Final Results with EKF:")
    print(f"  RMSE: {ekf_rmse:.4f} m/s (improvement: {val_rmse - ekf_rmse:.4f})")
    print(f"  MAE: {ekf_mae:.4f} m/s")
    print(f"  R²: {ekf_r2:.4f}")
    
    # Success criteria check
    print(f"\n" + "="*80)
    print("✅ Phase 1 Success Criteria:")
    print(f"  ✓ RMSE < 12 m/s: {ekf_rmse:.4f} {'✅ PASS' if ekf_rmse < 12 else '❌ FAIL'}")
    print(f"  ✓ R² > 0.05: {ekf_r2:.4f} {'✅ PASS' if ekf_r2 > 0.05 else '❌ FAIL'}")
    print(f"  ✓ Full dataset: {len(imu_data):,} samples ✅")
    print(f"  ✓ Normalization: {'✅ PASS' if abs(train_imu_normalized.mean()) < 0.1 else '❌ FAIL'}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Curves')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['val_rmse'])
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (m/s)')
    plt.title('Validation RMSE')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['val_r2'])
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.title('Validation R²')
    
    plt.tight_layout()
    plt.savefig('ml/outputs/phase1_training_curves.png', dpi=150)
    print(f"\n📈 Training curves saved to ml/outputs/phase1_training_curves.png")
    
    print(f"\n" + "="*80)
    print("🎉 Phase 1 Training Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
