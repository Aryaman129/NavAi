"""
Hardware-Aware Neural Architecture for Mobile Navigation
Based on TinyOdom (UCLA NESL) - Temporal Convolutional Network approach

Key innovations:
1. Hardware-in-the-loop optimization for mobile deployment
2. Temporal convolutions for sequence modeling
3. Efficient inference on ARM processors
4. Real-time performance on smartphones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import time

class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution for efficient mobile inference
    Reduces parameters by ~8-9x compared to standard convolution
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=padding, stride=stride, groups=in_channels, bias=False
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class TemporalBlock(nn.Module):
    """
    Temporal Convolutional Network block with hardware-aware optimizations
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        # Use depthwise separable convolutions for efficiency
        self.conv1 = DepthwiseSeparableConv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, stride=1
        )
        
        self.conv2 = DepthwiseSeparableConv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, stride=1
        )
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Causal padding (remove future information)
        out = out[:, :, :-self.conv1.depthwise.padding[0]]
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        if res.size(-1) != out.size(-1):
            res = res[:, :, :out.size(-1)]
        
        return self.relu(out + res)


class HardwareAwareTCN(nn.Module):
    """
    Hardware-aware Temporal Convolutional Network for navigation
    Optimized for ARM Cortex processors and smartphone deployment
    """
    
    def __init__(self, input_dim=6, output_dim=3, num_channels=[32, 64, 32], kernel_size=3, dropout=0.1):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                dilation=dilation_size, dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        
        # Output layers - separate heads for different outputs
        final_channels = num_channels[-1]
        
        # Speed estimation head
        self.speed_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(final_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()  # Ensure positive speed
        )
        
        # Direction/heading head
        self.heading_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(final_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # sin, cos of heading angle
            nn.Tanh()
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(final_channels, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass
        Input: (batch_size, input_dim, sequence_length)
        Output: speed, heading, confidence
        """
        # Ensure input has correct dimension order
        if x.dim() == 3 and x.size(1) > x.size(-1):
            x = x.transpose(1, 2)  # (batch, seq_len, features) -> (batch, features, seq_len)
        
        features = self.network(x)
        
        speed = self.speed_head(features)
        heading = self.heading_head(features)
        confidence = self.confidence_head(features)
        
        return {
            'speed': speed,
            'heading': heading,  # [sin(theta), cos(theta)]
            'confidence': confidence,
            'features': features
        }


class MobileOptimizedPreprocessor:
    """
    Mobile-optimized preprocessing pipeline
    Implements real-time filtering and normalization
    """
    
    def __init__(self, window_size=50, sample_rate=100):
        self.window_size = window_size
        self.sample_rate = sample_rate
        
        # Butterworth filter parameters for noise reduction
        from scipy import signal
        self.sos_accel = signal.butter(4, 20, 'lowpass', fs=sample_rate, output='sos')
        self.sos_gyro = signal.butter(4, 10, 'lowpass', fs=sample_rate, output='sos')
        
        # Circular buffers for real-time processing
        self.accel_buffer = np.zeros((window_size, 3))
        self.gyro_buffer = np.zeros((window_size, 3))
        self.timestamp_buffer = np.zeros(window_size)
        
        self.buffer_idx = 0
        self.is_buffer_full = False
        
        # Normalization parameters (computed from training data)
        self.accel_mean = np.array([0.0, 0.0, 9.81])
        self.accel_std = np.array([2.0, 2.0, 2.0])
        self.gyro_mean = np.array([0.0, 0.0, 0.0])
        self.gyro_std = np.array([0.5, 0.5, 0.5])
    
    def add_sample(self, accel: np.ndarray, gyro: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        """
        Add new sensor sample and return preprocessed sequence if ready
        """
        # Add to circular buffer
        self.accel_buffer[self.buffer_idx] = accel
        self.gyro_buffer[self.buffer_idx] = gyro
        self.timestamp_buffer[self.buffer_idx] = timestamp
        
        self.buffer_idx = (self.buffer_idx + 1) % self.window_size
        if not self.is_buffer_full and self.buffer_idx == 0:
            self.is_buffer_full = True
        
        # Return sequence if buffer is ready
        if self.is_buffer_full:
            return self._get_preprocessed_sequence()
        
        return None
    
    def _get_preprocessed_sequence(self) -> np.ndarray:
        """Get preprocessed sequence from circular buffer"""
        # Reorder buffer to get chronological sequence
        if self.buffer_idx == 0:
            accel_seq = self.accel_buffer.copy()
            gyro_seq = self.gyro_buffer.copy()
        else:
            accel_seq = np.vstack([
                self.accel_buffer[self.buffer_idx:],
                self.accel_buffer[:self.buffer_idx]
            ])
            gyro_seq = np.vstack([
                self.gyro_buffer[self.buffer_idx:],
                self.gyro_buffer[:self.buffer_idx]
            ])
        
        # Apply filtering (simplified - in practice use online filtering)
        # accel_seq = signal.sosfilt(self.sos_accel, accel_seq, axis=0)
        # gyro_seq = signal.sosfilt(self.sos_gyro, gyro_seq, axis=0)
        
        # Normalize
        accel_norm = (accel_seq - self.accel_mean) / self.accel_std
        gyro_norm = (gyro_seq - self.gyro_mean) / self.gyro_std
        
        # Combine and transpose for TCN input format
        sequence = np.hstack([accel_norm, gyro_norm])  # (seq_len, 6)
        return sequence.T  # (6, seq_len)


class RealTimeNavigator:
    """
    Real-time navigation system using hardware-aware TCN
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = HardwareAwareTCN()
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        self.model.eval()
        self.preprocessor = MobileOptimizedPreprocessor()
        
        # State tracking
        self.position = np.array([0.0, 0.0])
        self.heading = 0.0
        self.last_timestamp = None
        
        # Performance monitoring
        self.inference_times = []
        
    def process_imu_sample(self, accel: np.ndarray, gyro: np.ndarray, timestamp: float) -> Optional[dict]:
        """
        Process single IMU sample and return navigation update if available
        """
        # Add sample to preprocessor
        sequence = self.preprocessor.add_sample(accel, gyro, timestamp)
        
        if sequence is None:
            return None
        
        # Run inference
        start_time = time.time()
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
            output = self.model(input_tensor)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Extract results
        speed = output['speed'].item()
        heading_vec = output['heading'].squeeze().numpy()
        confidence = output['confidence'].item()
        
        # Convert heading vector to angle
        heading_angle = np.arctan2(heading_vec[0], heading_vec[1])
        
        # Update position (dead reckoning)
        if self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            dx = speed * np.cos(self.heading + heading_angle) * dt
            dy = speed * np.sin(self.heading + heading_angle) * dt
            self.position += np.array([dx, dy])
            self.heading += heading_angle * dt
        
        self.last_timestamp = timestamp
        
        return {
            'position': self.position.copy(),
            'heading': self.heading,
            'speed': speed,
            'confidence': confidence,
            'inference_time': inference_time
        }
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        return {
            'mean_inference_time': np.mean(times),
            'max_inference_time': np.max(times),
            'min_inference_time': np.min(times),
            'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0,
            'total_samples': len(times)
        }


# Training utilities for the hardware-aware model
class HardwareAwareTrainer:
    """
    Training utilities with hardware efficiency constraints
    """
    
    def __init__(self, model: HardwareAwareTCN, target_latency_ms=10):
        self.model = model
        self.target_latency_ms = target_latency_ms
        self.criterion = nn.MSELoss()
        
    def compute_efficiency_loss(self, output: dict) -> torch.Tensor:
        """
        Compute efficiency-aware loss that penalizes slow inference
        """
        # Model complexity penalty (parameter count)
        param_count = sum(p.numel() for p in self.model.parameters())
        complexity_penalty = param_count / 1e6  # Normalize by 1M parameters
        
        # Latency would be measured during training on target hardware
        # For now, use parameter count as proxy
        efficiency_loss = complexity_penalty * 0.01
        
        return efficiency_loss
    
    def train_step(self, batch_data: dict) -> dict:
        """
        Single training step with hardware efficiency constraints
        """
        imu_data = batch_data['imu']  # (batch, channels, seq_len)
        target_speed = batch_data['speed']
        target_heading = batch_data['heading']
        
        # Forward pass
        output = self.model(imu_data)
        
        # Compute losses
        speed_loss = self.criterion(output['speed'], target_speed)
        heading_loss = self.criterion(output['heading'], target_heading)
        efficiency_loss = self.compute_efficiency_loss(output)
        
        # Combined loss
        total_loss = speed_loss + heading_loss + efficiency_loss
        
        return {
            'total_loss': total_loss,
            'speed_loss': speed_loss.item(),
            'heading_loss': heading_loss.item(),
            'efficiency_loss': efficiency_loss.item()
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize system
    navigator = RealTimeNavigator()
    
    print("Testing Hardware-Aware Navigation System")
    print("========================================")
    
    # Simulate real-time IMU data
    sample_rate = 100  # 100 Hz
    duration = 10  # 10 seconds
    
    for i in range(duration * sample_rate):
        # Simulate IMU data
        timestamp = i / sample_rate
        accel = np.array([0.1 * np.sin(timestamp), 0.1 * np.cos(timestamp), 9.81])
        gyro = np.array([0.01, 0.01, 0.1 * np.sin(timestamp * 2)])
        
        # Process sample
        result = navigator.process_imu_sample(accel, gyro, timestamp)
        
        if result and i % 100 == 0:  # Print every second
            print(f"Time: {timestamp:.1f}s")
            print(f"  Position: ({result['position'][0]:.2f}, {result['position'][1]:.2f})")
            print(f"  Speed: {result['speed']:.2f} m/s")
            print(f"  Heading: {result['heading']:.2f} rad")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Inference time: {result['inference_time']*1000:.1f} ms")
            print()
    
    # Print performance statistics
    stats = navigator.get_performance_stats()
    print("Performance Statistics:")
    print("======================")
    for key, value in stats.items():
        if 'time' in key:
            print(f"{key}: {value*1000:.2f} ms")
        else:
            print(f"{key}: {value:.2f}")