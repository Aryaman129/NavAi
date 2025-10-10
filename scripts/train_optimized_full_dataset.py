"""
Optimized training on full comma2k19 dataset with best practices
- Full dataset utilization (all available samples)
- Advanced architecture (deeper network, attention mechanisms)
- Physics-informed loss functions
- Learning rate scheduling
- Early stopping with model checkpointing
- Comprehensive validation and metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, random_split
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import sys
sys.path.append('ml')

from data.data_loader import DataLoader

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎮 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

class AttentionLayer(nn.Module):
    """Self-attention mechanism for temporal sequences"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        # x shape: [batch, seq_len, hidden_dim]
        attn_output, _ = self.attention(x, x, x)
        return self.norm(x + attn_output)  # Residual connection

class OptimizedSpeedEstimator(nn.Module):
    """
    Advanced architecture for speed estimation:
    - Deeper LSTM layers
    - Self-attention mechanism
    - Residual connections
    - Physics-aware output heads
    """
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        
        # Deeper LSTM with more capacity
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better context
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_dim * 2)  # *2 for bidirectional
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        
        # Fully connected layers with residual
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 32)
        
        # Output heads
        self.speed_head = nn.Linear(32, 1)
        self.uncertainty_head = nn.Linear(32, 1)  # Uncertainty estimation
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim*2]
        
        # Attention mechanism
        attn_out = self.attention(lstm_out)  # [batch, seq_len, hidden_dim*2]
        
        # Take last timestep
        last_hidden = attn_out[:, -1, :]  # [batch, hidden_dim*2]
        
        # Batch normalization
        normalized = self.batch_norm(last_hidden)
        
        # Fully connected layers
        x = self.relu(self.fc1(normalized))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        
        # Output heads
        speed = self.speed_head(x)
        uncertainty = torch.abs(self.uncertainty_head(x))  # Positive uncertainty
        
        return speed, uncertainty

class PhysicsInformedLoss(nn.Module):
    """
    Custom loss function with physics constraints:
    1. Prediction accuracy (MSE)
    2. Temporal smoothness (acceleration continuity)
    3. Physical bounds (speed limits)
    """
    def __init__(self, smoothness_weight=0.1, bounds_weight=0.05, uncertainty_weight=0.1):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.bounds_weight = bounds_weight
        self.uncertainty_weight = uncertainty_weight
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, uncertainties, targets, prev_speeds=None):
        # Main prediction loss
        pred_loss = self.mse(predictions, targets)
        
        # Uncertainty loss (encourage confident predictions when correct)
        # Lower uncertainty when prediction is close to target
        uncertainty_loss = torch.mean(uncertainties * torch.abs(predictions - targets))
        
        # Temporal smoothness (if previous speeds available)
        smoothness_loss = 0
        if prev_speeds is not None:
            # Penalize large accelerations (speed changes)
            speed_changes = predictions - prev_speeds
            smoothness_loss = torch.mean(speed_changes ** 2)
        
        # Physical bounds penalty (speeds shouldn't be negative or unrealistically high)
        bounds_loss = torch.mean(torch.relu(-predictions))  # Negative speeds penalty
        bounds_loss += torch.mean(torch.relu(predictions - 50))  # >50 m/s penalty (180 km/h)
        
        # Combined loss
        total_loss = (pred_loss + 
                     self.uncertainty_weight * uncertainty_loss +
                     self.smoothness_weight * smoothness_loss +
                     self.bounds_weight * bounds_loss)
        
        return total_loss, pred_loss, uncertainty_loss, smoothness_loss, bounds_loss

def create_dataloaders(data, batch_size=64, val_split=0.15, test_split=0.15):
    """Create train/val/test dataloaders"""
    
    # Split into train/val/test
    total_size = len(data)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    train_data, val_data, test_data = random_split(
        data, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n📊 Dataset splits:")
    print(f"   Training:   {len(train_data):,} samples ({len(train_data)/total_size*100:.1f}%)")
    print(f"   Validation: {len(val_data):,} samples ({len(val_data)/total_size*100:.1f}%)")
    print(f"   Test:       {len(test_data):,} samples ({len(test_data)/total_size*100:.1f}%)")
    
    # Create dataloaders
    train_loader = TorchDataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = TorchDataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = TorchDataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_pred_loss = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    prev_speeds = None
    
    for batch_idx, (sequences, targets) in enumerate(pbar):
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        # Forward pass
        predictions, uncertainties = model(sequences)
        
        # Calculate loss with physics constraints
        loss, pred_loss, unc_loss, smooth_loss, bounds_loss = criterion(
            predictions, uncertainties, targets, prev_speeds
        )
        
        # Store predictions for smoothness loss in next batch
        prev_speeds = predictions.detach()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        # Statistics
        batch_size = sequences.size(0)
        total_loss += loss.item() * batch_size
        total_pred_loss += pred_loss.item() * batch_size
        total_samples += batch_size
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'pred': f'{pred_loss.item():.4f}'
        })
    
    return total_loss / total_samples, total_pred_loss / total_samples

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_pred_loss = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, targets in tqdm(val_loader, desc="Validating", leave=False):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            predictions, uncertainties = model(sequences)
            loss, pred_loss, _, _, _ = criterion(predictions, uncertainties, targets)
            
            batch_size = sequences.size(0)
            total_loss += loss.item() * batch_size
            total_pred_loss += pred_loss.item() * batch_size
            total_samples += batch_size
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    predictions = np.array(all_predictions).flatten()
    targets = np.array(all_targets).flatten()
    
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    
    return total_loss / total_samples, rmse, mae

def train_optimized_model(train_loader, val_loader, test_loader, 
                         num_epochs=100, patience=15):
    """
    Train optimized model with:
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Comprehensive logging
    """
    
    # Initialize model
    model = OptimizedSpeedEstimator(
        input_dim=6,
        hidden_dim=128,
        num_layers=3,
        dropout=0.3
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model architecture:")
    print(f"   Parameters: {num_params:,}")
    print(f"   Architecture: 3-layer Bidirectional LSTM + Attention + Physics-aware")
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Loss function
    criterion = PhysicsInformedLoss(
        smoothness_weight=0.1,
        bounds_weight=0.05,
        uncertainty_weight=0.1
    )
    
    # Training tracking
    best_val_loss = float('inf')
    best_rmse = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'lr': []
    }
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print(f"   Early stopping patience: {patience} epochs")
    print(f"   Device: {device}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_pred_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_rmse, val_mae = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)
        
        # Print epoch results
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val RMSE:   {val_rmse:.4f} m/s")
        print(f"  Val MAE:    {val_mae:.4f} m/s")
        print(f"  LR:         {current_lr:.6f}")
        
        # Model checkpointing
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'history': history
            }, 'ml/outputs/best_optimized_model.pth')
            
            print(f"  ✅ New best RMSE! Model saved.")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
            break
    
    total_time = time.time() - start_time
    print(f"\n✅ Training complete in {total_time/60:.1f} minutes")
    print(f"   Best Val RMSE: {best_rmse:.4f} m/s")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    
    # Load best model for testing
    checkpoint = torch.load('ml/outputs/best_optimized_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test on test set
    print(f"\n📊 Testing on test set...")
    test_loss, test_rmse, test_mae = validate(model, test_loader, criterion, device)
    print(f"   Test RMSE: {test_rmse:.4f} m/s")
    print(f"   Test MAE:  {test_mae:.4f} m/s")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history, test_rmse

def plot_training_history(history):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # RMSE curve
    axes[0, 1].plot(history['val_rmse'], label='Val RMSE', color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('RMSE (m/s)')
    axes[0, 1].set_title('Validation RMSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # MAE curve
    axes[1, 0].plot(history['val_mae'], label='Val MAE', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE (m/s)')
    axes[1, 0].set_title('Validation MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate
    axes[1, 1].plot(history['lr'], label='Learning Rate', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('ml/outputs/training_history.png', dpi=150)
    print(f"\n📈 Training history saved to ml/outputs/training_history.png")

def prepare_sequences(df, window_size=150, stride=50):
    """
    Prepare training sequences from dataframe
    Args:
        df: DataFrame with IMU and speed data
        window_size: Number of timesteps in each sequence
        stride: Step size between sequences
    Returns:
        sequences: numpy array of shape (n_samples, window_size, 6)
        targets: numpy array of shape (n_samples,)
    """
    # Extract IMU features (accel_xyz + gyro_xyz)
    imu_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    imu_data = df[imu_cols].values
    speed_data = df['gps_speed_mps'].values
    
    sequences = []
    targets = []
    
    for i in range(0, len(imu_data) - window_size, stride):
        seq = imu_data[i:i+window_size]
        target = speed_data[i+window_size]  # Predict speed at end of window
        
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def main():
    print("="*70)
    print("🚀 OPTIMIZED TRAINING ON FULL COMMA2K19 DATASET")
    print("="*70)
    
    # Load data using DataLoader class
    print("\n[1/4] Loading comma2k19 dataset...")
    loader = DataLoader(target_sample_rate=100)
    data = loader.load_comma2k19("data/comma2k19")
    
    if data is None or len(data) == 0:
        print("❌ No data loaded! Check dataset location.")
        return
    
    print(f"✓ Loaded {len(data):,} samples")
    print(f"  Speed range: {data['gps_speed_mps'].min():.2f} - {data['gps_speed_mps'].max():.2f} m/s")
    print(f"  Speed mean: {data['gps_speed_mps'].mean():.2f} m/s")
    
    # Prepare sequences
    print("\n[2/4] Preparing training sequences...")
    sequences, targets = prepare_sequences(data, window_size=150, stride=50)
    
    print(f"✓ Created {len(sequences):,} sequences")
    
    # Create custom dataset
    class SpeedDataset(Dataset):
        def __init__(self, sequences, targets):
            self.sequences = torch.FloatTensor(sequences)
            self.targets = torch.FloatTensor(targets).unsqueeze(1)
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return self.sequences[idx], self.targets[idx]
    
    dataset = SpeedDataset(sequences, targets)
    
    # Create dataloaders
    print("\n[3/4] Creating train/val/test splits...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, batch_size=128, val_split=0.15, test_split=0.15
    )
    
    # Train model
    print("\n[4/4] Training optimized model...")
    model, history, test_rmse = train_optimized_model(
        train_loader, val_loader, test_loader,
        num_epochs=100,  # Will use early stopping
        patience=15
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"📁 Best model saved to: ml/outputs/best_optimized_model.pth")
    print(f"📈 Training plots saved to: ml/outputs/training_history.png")
    print(f"🎯 Final Test RMSE: {test_rmse:.4f} m/s")
    print("="*70)

if __name__ == "__main__":
    main()
