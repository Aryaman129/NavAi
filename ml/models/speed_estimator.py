"""
IMU-based speed estimation models for NavAI project
Supports both PyTorch and TensorFlow implementations for mobile deployment
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class SpeedCNN(nn.Module):
    """
    1D CNN for IMU-based speed estimation
    Input: [batch_size, sequence_length, num_features]
    Output: [batch_size, 1] (speed in m/s)
    """
    
    def __init__(self, 
                 input_channels: int = 6,  # accel_x,y,z + gyro_x,y,z
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_channels = input_channels
        
        # 1D Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = hidden_dim * (2 ** i) if i < 2 else hidden_dim * 4
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            in_channels = out_channels
        
        # Global pooling and final layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, sequence_length, features]
        # Convert to [batch_size, features, sequence_length] for Conv1d
        x = x.permute(0, 2, 1)
        
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global pooling
        x = self.global_pool(x)  # [batch_size, channels, 1]
        x = x.squeeze(-1)        # [batch_size, channels]
        
        # Final classification
        speed = self.classifier(x)
        
        # Ensure positive speed
        return F.relu(speed)

class SpeedLSTM(nn.Module):
    """
    LSTM-based speed estimator for comparison
    """
    
    def __init__(self,
                 input_size: int = 6,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, sequence_length, features]
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        speed = self.classifier(last_output)
        return F.relu(speed)

def create_tensorflow_model(input_shape: Tuple[int, int], 
                          model_type: str = 'cnn') -> tf.keras.Model:
    """
    Create TensorFlow model for mobile deployment
    
    Args:
        input_shape: (sequence_length, num_features)
        model_type: 'cnn' or 'lstm'
    """
    
    inputs = tf.keras.Input(shape=input_shape, name='imu_input')
    
    if model_type == 'cnn':
        # 1D CNN architecture
        x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        x = tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
    elif model_type == 'lstm':
        # LSTM architecture
        x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.1)(inputs)
        x = tf.keras.layers.LSTM(64, dropout=0.1)(x)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Final layers
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='relu', name='speed_output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f'speed_{model_type}')
    
    return model

def convert_to_tflite(model: tf.keras.Model, 
                     quantize: bool = True,
                     representative_dataset: Optional[np.ndarray] = None) -> bytes:
    """
    Convert TensorFlow model to TensorFlow Lite format
    
    Args:
        model: Trained TensorFlow model
        quantize: Whether to apply quantization
        representative_dataset: Sample data for quantization calibration
    """
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if representative_dataset is not None:
            def representative_data_gen():
                for sample in representative_dataset[:100]:  # Use subset for calibration
                    yield [sample.astype(np.float32)]
            
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    logger.info(f"TFLite model size: {len(tflite_model) / 1024:.1f} KB")
    
    return tflite_model

class WindowGenerator:
    """
    Generate sliding windows from sensor data for training
    """
    
    def __init__(self, 
                 window_size_sec: float = 1.5,
                 stride_sec: float = 0.25,
                 sample_rate: int = 100,
                 feature_cols: List[str] = None,
                 target_col: str = 'gps_speed_mps'):
        
        self.window_size = int(window_size_sec * sample_rate)
        self.stride = int(stride_sec * sample_rate)
        self.sample_rate = sample_rate
        
        if feature_cols is None:
            self.feature_cols = ['accel_x', 'accel_y', 'accel_z', 
                               'gyro_x', 'gyro_y', 'gyro_z']
        else:
            self.feature_cols = feature_cols
            
        self.target_col = target_col
        
    def create_windows(self, df) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from DataFrame
        
        Returns:
            X: [num_windows, window_size, num_features]
            y: [num_windows] - target values
        """
        
        if len(df) < self.window_size:
            logger.warning(f"DataFrame too short ({len(df)}) for window size ({self.window_size})")
            return np.array([]), np.array([])
        
        # Extract features and targets
        features = df[self.feature_cols].values.astype(np.float32)
        targets = df[self.target_col].values.astype(np.float32)
        
        X, y = [], []
        
        for start_idx in range(0, len(df) - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            
            # Feature window
            window_features = features[start_idx:end_idx]
            
            # Target (use value at end of window)
            target_value = targets[end_idx - 1]
            
            # Skip windows with invalid GPS data
            if np.isnan(target_value) or target_value < 0:
                continue
                
            X.append(window_features)
            y.append(target_value)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} windows of shape {X.shape[1:]} from {len(df)} samples")
        
        return X, y

# Example usage and testing
if __name__ == "__main__":
    # Test PyTorch model
    model = SpeedCNN(input_channels=6, hidden_dim=64)
    
    # Test input
    batch_size, seq_len, features = 32, 150, 6
    x = torch.randn(batch_size, seq_len, features)
    
    with torch.no_grad():
        output = model(x)
        print(f"PyTorch model output shape: {output.shape}")
    
    # Test TensorFlow model
    tf_model = create_tensorflow_model((seq_len, features), 'cnn')
    tf_model.summary()
    
    # Test window generator
    import pandas as pd
    
    # Create dummy data
    dummy_data = {
        'accel_x': np.random.randn(1000),
        'accel_y': np.random.randn(1000),
        'accel_z': np.random.randn(1000),
        'gyro_x': np.random.randn(1000),
        'gyro_y': np.random.randn(1000),
        'gyro_z': np.random.randn(1000),
        'gps_speed_mps': np.abs(np.random.randn(1000)) * 10
    }
    
    df = pd.DataFrame(dummy_data)
    
    window_gen = WindowGenerator()
    X, y = window_gen.create_windows(df)
    
    print(f"Generated windows: X shape {X.shape}, y shape {y.shape}")
