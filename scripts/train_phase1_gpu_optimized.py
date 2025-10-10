"""
Phase 1: GPU-Optimized Training for RTX 4050 (6GB VRAM, 16GB RAM)
- TensorFlow GPU acceleration
- Memory-efficient batch processing
- Early stopping (25 epochs max)
- Best configuration for 6GB VRAM
- Mixed precision training for speed
"""

# Suppress all TensorFlow warnings FIRST
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress ALL TF logs (including NUMA/cuDNN warnings)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for stability

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
from tqdm import tqdm

# Enable GPU memory growth (CRITICAL for 6GB VRAM)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
        print(f"   GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"⚠️  GPU config error: {e}")
else:
    print("⚠️  No GPU detected, using CPU")

# Mixed precision for faster training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"✅ Mixed precision enabled: {policy.name}")


class OptimizedPreprocessor:
    """Memory-efficient preprocessor for large datasets"""
    
    def __init__(self):
        self.means = None
        self.stds = None
        self.feature_names = None
    
    def fit_transform(self, data, chunk_size=100000):
        """Fit on training data and transform in chunks"""
        print("\n🔧 Computing statistics on training data...")
        
        # Compute statistics in chunks to save memory
        n_samples = data.shape[0]
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        
        # Compute means
        sums = np.zeros(data.shape[-1])
        for i in tqdm(range(n_chunks), desc="Computing means"):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_samples)
            chunk = data[start:end]
            sums += chunk.sum(axis=(0, 1))
        
        total_elements = n_samples * data.shape[1]
        self.means = sums / total_elements
        
        # Compute stds
        squared_diffs = np.zeros(data.shape[-1])
        for i in tqdm(range(n_chunks), desc="Computing stds"):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_samples)
            chunk = data[start:end]
            squared_diffs += ((chunk - self.means) ** 2).sum(axis=(0, 1))
        
        self.stds = np.sqrt(squared_diffs / total_elements)
        self.stds = np.where(self.stds < 1e-8, 1.0, self.stds)  # Avoid division by zero
        
        print(f"   Means: {self.means[:3]}")
        print(f"   Stds: {self.stds[:3]}")
        
        # Transform data
        return self.transform(data, chunk_size)
    
    def transform(self, data, chunk_size=100000):
        """Transform data in chunks"""
        n_samples = data.shape[0]
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        
        normalized = np.zeros_like(data, dtype=np.float32)
        
        for i in tqdm(range(n_chunks), desc="Normalizing"):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_samples)
            normalized[start:end] = (data[start:end] - self.means) / self.stds
        
        return normalized
    
    def save(self, path):
        """Save preprocessor parameters"""
        params = {
            'means': self.means.tolist(),
            'stds': self.stds.tolist()
        }
        with open(path, 'w') as f:
            json.dump(params, f)
        print(f"✅ Preprocessor saved to {path}")


def create_windows_fast(data, window_size):
    """Fast windowing using stride tricks"""
    from numpy.lib.stride_tricks import as_strided
    
    num_samples = len(data) - window_size + 1
    shape = (num_samples, window_size, data.shape[1])
    strides = (data.strides[0], data.strides[0], data.strides[1])
    
    return as_strided(data, shape=shape, strides=strides).copy()


def build_optimized_model(input_shape, hidden_dim=96, num_layers=2, dropout=0.3):
    """
    Build memory-efficient BiLSTM model
    
    Optimizations for 6GB VRAM:
    - Reduced hidden_dim: 128 -> 96 (saves ~30% memory)
    - Reduced layers: 3 -> 2 (saves ~25% memory)
    - Increased dropout: 0.2 -> 0.3 (better generalization)
    - BatchNormalization for stability
    """
    
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)
    
    # LSTM layers
    x = inputs
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                hidden_dim,
                return_sequences=return_sequences,
                dropout=dropout if num_layers > 1 else 0,
                recurrent_dropout=0.1,
                name=f'lstm_{i}'
            )
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(hidden_dim, activation='relu', name='fc1')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation='relu', name='fc2')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    # Output layer (float32 for numerical stability)
    outputs = tf.keras.layers.Dense(1, dtype='float32', name='output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='BiLSTM_SpeedEstimator')
    
    return model


def main():
    print("=" * 80)
    print("🚀 Phase 1: GPU-Optimized Training (RTX 4050 6GB)")
    print("=" * 80)
    
    # Clear any existing GPU memory
    tf.keras.backend.clear_session()
    
    # Hyperparameters (optimized for 6GB VRAM)
    WINDOW_SIZE = 100
    BATCH_SIZE = 256  # Optimal for RTX 4050
    MAX_EPOCHS = 25
    EARLY_STOPPING_PATIENCE = 7
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Load dataset
    print("\n📁 Loading comma2k19 dataset...")
    parquet_path = Path('data/comma2k19/processed_real/comma2k19_processed.parquet')
    
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        print(f"✅ Loaded {len(df):,} samples from parquet")
    else:
        print("❌ Dataset not found!")
        return
    
    # Extract features
    feature_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    imu_data = df[feature_cols].values.astype(np.float32)
    speeds = df['speed'].values.astype(np.float32)
    
    print(f"   IMU shape: {imu_data.shape}")
    print(f"   Speed range: {speeds.min():.2f} - {speeds.max():.2f} m/s")
    
    # Create windows
    print(f"\n🪟 Creating windows (size={WINDOW_SIZE})...")
    imu_windows = create_windows_fast(imu_data, WINDOW_SIZE)
    speed_targets = speeds[WINDOW_SIZE-1:]
    
    print(f"   Windows: {imu_windows.shape}")
    print(f"   Targets: {speed_targets.shape}")
    
    # Temporal split (CRITICAL: no shuffling for time series)
    split_idx = int((1 - VALIDATION_SPLIT) * len(imu_windows))
    
    train_imu = imu_windows[:split_idx]
    train_speeds = speed_targets[:split_idx]
    val_imu = imu_windows[split_idx:]
    val_speeds = speed_targets[split_idx:]
    
    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(train_imu):,} samples ({len(train_imu)/len(imu_windows)*100:.1f}%)")
    print(f"   Validation: {len(val_imu):,} samples ({len(val_imu)/len(imu_windows)*100:.1f}%)")
    
    # Preprocess data
    print(f"\n🔧 Preprocessing...")
    preprocessor = OptimizedPreprocessor()
    train_imu_norm = preprocessor.fit_transform(train_imu, chunk_size=100000)
    val_imu_norm = preprocessor.transform(val_imu, chunk_size=100000)
    
    # Save preprocessor
    preprocessor.save('ml/outputs/preprocessor_params.json')
    
    # Memory cleanup
    del imu_windows, imu_data, df, train_imu, val_imu
    import gc
    gc.collect()
    
    print(f"   Normalized shapes: {train_imu_norm.shape}, {val_imu_norm.shape}")
    print(f"   Memory usage: ~{(train_imu_norm.nbytes + val_imu_norm.nbytes) / 1e9:.2f} GB")
    
    # Build model
    print(f"\n🏗️ Building model...")
    input_shape = (WINDOW_SIZE, train_imu_norm.shape[-1])
    model = build_optimized_model(input_shape, hidden_dim=96, num_layers=2, dropout=0.3)
    
    # Print model summary
    model.summary()
    
    total_params = model.count_params()
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Estimated model size: ~{total_params * 4 / 1e6:.1f} MB")
    
    # Compile model
    print(f"\n⚙️ Compiling model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # Use MSE loss with float32 for numerical stability
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            tf.keras.metrics.MeanAbsoluteError(name='mae')
        ]
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'ml/outputs/phase1_best_model_checkpoint.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            'ml/outputs/training_history.csv'
        )
    ]
    
    # Train model
    print(f"\n🚀 Starting training...")
    print(f"   Max epochs: {MAX_EPOCHS}")
    print(f"   Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    
    start_time = time.time()
    
    history = model.fit(
        train_imu_norm, train_speeds,
        validation_data=(val_imu_norm, val_speeds),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1  # Show progress bars
    )
    
    train_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {train_time/60:.1f} minutes")
    
    # Final evaluation
    print(f"\n📊 Final Evaluation:")
    val_loss, val_rmse, val_mae = model.evaluate(val_imu_norm, val_speeds, batch_size=BATCH_SIZE, verbose=0)
    
    # Compute R²
    val_predictions = model.predict(val_imu_norm, batch_size=BATCH_SIZE, verbose=0)
    ss_res = np.sum((val_speeds - val_predictions.flatten()) ** 2)
    ss_tot = np.sum((val_speeds - np.mean(val_speeds)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"   Val Loss: {val_loss:.6f}")
    print(f"   Val RMSE: {val_rmse:.4f} m/s")
    print(f"   Val MAE: {val_mae:.4f} m/s")
    print(f"   Val R²: {r2_score:.4f}")
    
    # Save final model in multiple formats
    print(f"\n💾 Saving models...")
    
    # 1. Keras format (best for TensorFlow)
    model.save('ml/outputs/phase1_model.keras')
    print(f"   ✅ Saved Keras model: ml/outputs/phase1_model.keras")
    
    # 2. PyTorch-compatible format (for export script)
    import torch
    
    # Create dummy PyTorch state dict structure
    torch_checkpoint = {
        'epoch': len(history.history['loss']),
        'model_state_dict': None,  # Will be filled by conversion
        'optimizer_state_dict': None,
        'rmse': float(val_rmse),
        'r2': float(r2_score),
        'metadata': {
            'framework': 'tensorflow',
            'keras_model_path': 'ml/outputs/phase1_model.keras',
            'input_shape': input_shape,
            'hidden_dim': 96,
            'num_layers': 2
        }
    }
    
    torch.save(torch_checkpoint, 'ml/outputs/phase1_best_model.pth')
    print(f"   ✅ Saved PyTorch checkpoint: ml/outputs/phase1_best_model.pth")
    
    # Save training summary
    summary = {
        'training_time_minutes': train_time / 60,
        'total_epochs': len(history.history['loss']),
        'best_epoch': np.argmin(history.history['val_loss']) + 1,
        'final_metrics': {
            'val_loss': float(val_loss),
            'val_rmse': float(val_rmse),
            'val_mae': float(val_mae),
            'val_r2': float(r2_score)
        },
        'hyperparameters': {
            'window_size': WINDOW_SIZE,
            'batch_size': BATCH_SIZE,
            'max_epochs': MAX_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'hidden_dim': 96,
            'num_layers': 2,
            'dropout': 0.3
        },
        'dataset': {
            'train_samples': len(train_speeds),
            'val_samples': len(val_speeds),
            'total_samples': len(train_speeds) + len(val_speeds)
        }
    }
    
    with open('ml/outputs/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✅ Saved training summary: ml/outputs/training_summary.json")
    
    # Plot training curves
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss', alpha=0.8)
    plt.plot(history.history['val_loss'], label='Val Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    plt.grid(True, alpha=0.3)
    
    # RMSE curves
    plt.subplot(1, 3, 2)
    plt.plot(history.history['rmse'], label='Train RMSE', alpha=0.8)
    plt.plot(history.history['val_rmse'], label='Val RMSE', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (m/s)')
    plt.legend()
    plt.title('Training & Validation RMSE')
    plt.grid(True, alpha=0.3)
    
    # MAE curves
    plt.subplot(1, 3, 3)
    plt.plot(history.history['mae'], label='Train MAE', alpha=0.8)
    plt.plot(history.history['val_mae'], label='Val MAE', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MAE (m/s)')
    plt.legend()
    plt.title('Training & Validation MAE')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml/outputs/phase1_training_curves.png', dpi=150)
    print(f"   ✅ Saved training curves: ml/outputs/phase1_training_curves.png")
    
    # Success criteria
    print(f"\n" + "=" * 80)
    print("✅ Phase 1 Success Criteria:")
    print(f"   ✓ RMSE < 12 m/s: {val_rmse:.4f} {'✅ PASS' if val_rmse < 12 else '❌ FAIL'}")
    print(f"   ✓ R² > 0.05: {r2_score:.4f} {'✅ PASS' if r2_score > 0.05 else '❌ FAIL'}")
    print(f"   ✓ Training time: {train_time/60:.1f} min {'✅ FAST' if train_time < 600 else '⚠️ SLOW'}")
    print(f"   ✓ GPU utilized: {'✅ YES' if gpus else '❌ NO'}")
    
    print(f"\n" + "=" * 80)
    print("🎉 Training Complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Run export script: python ml/export_phase1_tflite.py")
    print(f"  2. Check outputs in: ml/outputs/")
    print(f"  3. View training curves: ml/outputs/phase1_training_curves.png")


if __name__ == '__main__':
    main()
