"""
Physics-Informed Speed Estimator for NavAI
Enhanced CNN with physics constraints and uncertainty quantification
Based on 2024-2025 research findings on PINN and hardware-aware architectures
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PhysicsInformedSpeedCNN(nn.Module):
    """
    Enhanced CNN with physics constraints and uncertainty quantification
    
    Features:
    - Physics-informed loss function with kinematic constraints
    - Uncertainty quantification (mean + variance output)
    - Scenario-aware processing (walking/cycling/vehicle/stationary)
    - Mount-aware feature extraction
    """
    
    def __init__(self, 
                 input_channels: int = 6,  # accel_x,y,z + gyro_x,y,z
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_channels = input_channels
        
        # Enhanced feature extractor with depthwise separable convolutions
        self.feature_extractor = nn.ModuleList()
        in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = hidden_dim * (2 ** min(i, 2))
            
            # Depthwise separable convolution for mobile efficiency
            self.feature_extractor.append(
                nn.Sequential(
                    # Depthwise convolution
                    nn.Conv1d(in_channels, in_channels, kernel_size=5, 
                             padding=2, groups=in_channels, bias=False),
                    # Pointwise convolution
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            in_channels = out_channels
        
        # Scenario classifier
        self.scenario_classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # walking, cycling, vehicle, stationary
        )
        
        # Mount type classifier
        self.mount_classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # pocket, dashboard, handlebar, handheld
        )
        
        # Physics constraint network
        self.physics_constraint = nn.Sequential(
            nn.Linear(in_channels + 8, 128),  # features + scenario + mount
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # mean, log_variance
        )
        
        # Scenario-specific physics heads
        self.walking_physics = self._create_scenario_physics_head()
        self.cycling_physics = self._create_scenario_physics_head()
        self.vehicle_physics = self._create_scenario_physics_head()
        self.stationary_physics = self._create_scenario_physics_head()
        
    def _create_scenario_physics_head(self):
        """Create scenario-specific physics constraint head"""
        return nn.Sequential(
            nn.Linear(256, 64),  # Match the final feature dimension
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with physics constraints
        
        Args:
            x: [batch_size, sequence_length, features] IMU data
            
        Returns:
            Dictionary with speed predictions, uncertainties, and auxiliary outputs
        """
        batch_size, seq_len, features = x.shape
        
        # Convert to [batch_size, features, sequence_length] for Conv1d
        x_conv = x.permute(0, 2, 1)
        
        # Feature extraction
        features_tensor = x_conv
        for layer in self.feature_extractor:
            features_tensor = layer(features_tensor)
        
        # Extract global features
        global_features = F.adaptive_avg_pool1d(features_tensor, 1).squeeze(-1)
        
        # Scenario classification
        scenario_logits = self.scenario_classifier(features_tensor)
        scenario_probs = F.softmax(scenario_logits, dim=1)
        
        # Mount classification
        mount_logits = self.mount_classifier(features_tensor)
        mount_probs = F.softmax(mount_logits, dim=1)
        
        # Combine features with context
        combined_features = torch.cat([global_features, scenario_probs, mount_probs], dim=1)
        
        # Physics-informed prediction
        physics_params = self.physics_constraint(combined_features)
        speed_mean = torch.clamp(physics_params[:, 0:1], min=0.0)  # Keep as [batch, 1]
        speed_log_var = physics_params[:, 1:2]  # Keep as [batch, 1] 
        speed_var = torch.exp(speed_log_var)
        
        # Scenario-specific physics constraints
        walking_constraint = self.walking_physics(global_features)
        cycling_constraint = self.cycling_physics(global_features)
        vehicle_constraint = self.vehicle_physics(global_features)
        stationary_constraint = self.stationary_physics(global_features)
        
        # Weighted combination based on scenario probabilities
        scenario_constraint = (
            scenario_probs[:, 0:1] * walking_constraint +
            scenario_probs[:, 1:2] * cycling_constraint +
            scenario_probs[:, 2:3] * vehicle_constraint +
            scenario_probs[:, 3:4] * stationary_constraint
        )
        
        # Apply scenario constraints
        constrained_speed = speed_mean * torch.sigmoid(scenario_constraint)
        
        return {
            'speed_mean': constrained_speed,
            'speed_variance': speed_var,
            'scenario_probs': scenario_probs,
            'mount_probs': mount_probs,
            'features': global_features
        }


class TemporalPhysicsValidator:
    """
    Validates predictions against recent motion patterns using physics constraints
    """
    
    def __init__(self, memory_length: int = 50, dt: float = 0.01):
        self.memory_length = memory_length
        self.dt = dt
        
        # Circular buffers for efficient memory management
        self.speed_history = deque(maxlen=memory_length)
        self.accel_history = deque(maxlen=memory_length)
        self.scenario_history = deque(maxlen=memory_length)
        self.mount_history = deque(maxlen=memory_length)
        
        # Physics parameters by scenario
        self.scenario_params = {
            0: {'max_accel': 2.0, 'max_jerk': 2.0, 'name': 'walking'},
            1: {'max_accel': 3.0, 'max_jerk': 3.0, 'name': 'cycling'},
            2: {'max_accel': 8.0, 'max_jerk': 5.0, 'name': 'vehicle'},
            3: {'max_accel': 0.2, 'max_jerk': 0.5, 'name': 'stationary'}
        }
        
    def update_history(self, speed: float, accel: np.ndarray, scenario: int, mount: int):
        """Update temporal memory with new measurements"""
        self.speed_history.append(speed)
        self.accel_history.append(np.linalg.norm(accel))  # Magnitude
        self.scenario_history.append(scenario)
        self.mount_history.append(mount)
        
    def validate_prediction(self, predicted_speed: float, current_accel: np.ndarray, 
                          scenario: int) -> Tuple[float, float]:
        """
        Validate prediction against physics constraints
        
        Returns:
            Tuple of (corrected_speed, confidence_score)
        """
        if len(self.speed_history) < 3:
            return predicted_speed, 1.0
            
        recent_speeds = np.array(list(self.speed_history)[-5:])
        recent_accels = np.array(list(self.accel_history)[-5:])
        
        # Physics consistency checks
        consistency_scores = []
        
        # 1. Kinematic consistency
        if len(recent_speeds) >= 2:
            expected_speed = recent_speeds[-1] + np.linalg.norm(current_accel) * self.dt
            kinematic_error = abs(predicted_speed - expected_speed)
            kinematic_score = np.exp(-kinematic_error / max(predicted_speed, 1.0))
            consistency_scores.append(kinematic_score)
            
        # 2. Acceleration limits
        accel_magnitude = np.linalg.norm(current_accel)
        max_accel = self.scenario_params[scenario]['max_accel']
        accel_score = np.exp(-max(0, accel_magnitude - max_accel) / max_accel)
        consistency_scores.append(accel_score)
        
        # 3. Jerk limits (rate of acceleration change)
        if len(recent_accels) >= 2:
            jerk = abs(accel_magnitude - recent_accels[-1]) / self.dt
            max_jerk = self.scenario_params[scenario]['max_jerk']
            jerk_score = np.exp(-max(0, jerk - max_jerk) / max_jerk)
            consistency_scores.append(jerk_score)
            
        # 4. Speed continuity
        if len(recent_speeds) >= 1:
            speed_change_rate = abs(predicted_speed - recent_speeds[-1]) / self.dt
            continuity_score = np.exp(-speed_change_rate / max_accel)
            consistency_scores.append(continuity_score)
            
        # Overall confidence
        confidence = np.mean(consistency_scores) if consistency_scores else 1.0
        
        # Apply correction if confidence is low
        if confidence < 0.3 and len(recent_speeds) >= 3:
            # Use physics-based correction
            corrected_speed = self._physics_correction(
                predicted_speed, recent_speeds, current_accel, scenario
            )
            return corrected_speed, confidence
            
        return predicted_speed, confidence
        
    def _physics_correction(self, predicted_speed: float, recent_speeds: np.ndarray,
                           current_accel: np.ndarray, scenario: int) -> float:
        """Apply physics-based correction to prediction"""
        # Simple kinematic correction
        expected_speed = recent_speeds[-1] + np.linalg.norm(current_accel) * self.dt
        max_accel = self.scenario_params[scenario]['max_accel']
        
        # Limit speed change based on maximum acceleration
        max_speed_change = max_accel * self.dt
        speed_change = predicted_speed - recent_speeds[-1]
        
        if abs(speed_change) > max_speed_change:
            corrected_change = np.sign(speed_change) * max_speed_change
            corrected_speed = recent_speeds[-1] + corrected_change
        else:
            corrected_speed = predicted_speed
            
        return max(0.0, corrected_speed)  # Ensure non-negative


def physics_informed_loss(predictions: Dict[str, torch.Tensor], 
                         targets: torch.Tensor,
                         imu_data: torch.Tensor,
                         lambda_weights: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Physics-informed loss function with multiple constraint terms
    
    Args:
        predictions: Model outputs including speed_mean, speed_variance, etc.
        targets: Ground truth speeds
        imu_data: Raw IMU data for physics calculations
        lambda_weights: Weights for different loss terms
        
    Returns:
        Tuple of (total_loss, loss_components)
    """
    if lambda_weights is None:
        lambda_weights = {
            'nll': 1.0,
            'kinematic': 0.1, 
            'smoothness': 0.05,
            'scenario': 0.1
        }
        
    speed_mean = predictions['speed_mean']
    speed_var = predictions['speed_variance']
    scenario_probs = predictions['scenario_probs']
    
    # 1. Negative log-likelihood loss with uncertainty
    nll_loss = 0.5 * (torch.log(speed_var) + (speed_mean - targets)**2 / speed_var).mean()
    
    # 2. Kinematic consistency loss
    accel_data = imu_data[:, :, 0:3]  # First 3 channels are accelerations
    dt = 0.01  # 100Hz sampling
    
    # Integrate acceleration magnitude over sequence
    accel_magnitude = torch.norm(accel_data, dim=2)  # [B, T]
    integrated_speed = torch.cumsum(accel_magnitude * dt, dim=1)[:, -1].unsqueeze(1)  # [B, 1]
    kinematic_loss = F.mse_loss(speed_mean, integrated_speed.detach())
    
    # 3. Temporal smoothness loss (penalize rapid changes)
    if speed_mean.shape[0] > 1:
        speed_diff = torch.diff(speed_mean.squeeze())
        smoothness_loss = torch.mean(speed_diff**2)
    else:
        smoothness_loss = torch.tensor(0.0, device=speed_mean.device)
    
    # 4. Scenario classification loss (if ground truth scenarios available)
    # For now, encourage confident predictions
    scenario_entropy = -torch.sum(scenario_probs * torch.log(scenario_probs + 1e-8), dim=1)
    scenario_loss = torch.mean(scenario_entropy)
    
    # Combined loss
    total_loss = (
        lambda_weights['nll'] * nll_loss +
        lambda_weights['kinematic'] * kinematic_loss +
        lambda_weights['smoothness'] * smoothness_loss +
        lambda_weights['scenario'] * scenario_loss
    )
    
    loss_components = {
        'nll': nll_loss.item(),
        'kinematic': kinematic_loss.item(),
        'smoothness': smoothness_loss.item(),
        'scenario': scenario_loss.item(),
        'total': total_loss.item()
    }
    
    return total_loss, loss_components


class MountAwarePreprocessor:
    """
    Mount-aware preprocessing that adapts to different phone placements
    """
    
    def __init__(self):
        self.mount_params = {
            0: {'name': 'pocket', 'noise_scale': 2.0, 'bias_adapt': True},
            1: {'name': 'dashboard', 'noise_scale': 1.0, 'bias_adapt': False}, 
            2: {'name': 'handlebar', 'noise_scale': 1.5, 'bias_adapt': True},
            3: {'name': 'handheld', 'noise_scale': 1.2, 'bias_adapt': False}
        }
        
    def preprocess(self, imu_data: np.ndarray, mount_type: int) -> np.ndarray:
        """Apply mount-specific preprocessing"""
        params = self.mount_params.get(mount_type, self.mount_params[1])
        
        # Apply noise scaling based on mount type
        if params['noise_scale'] != 1.0:
            noise_factor = 1.0 / params['noise_scale']
            imu_data = imu_data * noise_factor
            
        # Additional mount-specific processing can be added here
        
        return imu_data
    
    def classify_mount_type(self, imu_window: np.ndarray) -> int:
        """
        Simple rule-based mount classification
        Returns mount type index
        """
        accel_data = imu_window[:, 0:3]
        gyro_data = imu_window[:, 3:6]
        
        # Calculate statistics
        accel_variance = np.var(accel_data, axis=0).mean()
        gyro_variance = np.var(gyro_data, axis=0).mean()
        gravity_alignment = self._calculate_gravity_alignment(accel_data)
        
        # Simple classification rules
        if accel_variance > 3.0 and gyro_variance > 0.2:
            return 0  # pocket (high noise, frequent orientation changes)
        elif gravity_alignment > 0.9 and accel_variance < 1.0:
            return 1  # dashboard (stable, forward-facing)
        elif gyro_variance > 0.15 and accel_variance < 2.0:
            return 2  # handlebar (bike/motorcycle)
        else:
            return 3  # handheld (walking with phone)
            
    def _calculate_gravity_alignment(self, accel_data: np.ndarray) -> float:
        """Calculate how well acceleration aligns with expected gravity"""
        gravity_vector = np.array([0, 0, -9.81])
        mean_accel = np.mean(accel_data, axis=0)
        
        # Normalize vectors
        mean_accel_norm = mean_accel / (np.linalg.norm(mean_accel) + 1e-8)
        gravity_norm = gravity_vector / np.linalg.norm(gravity_vector)
        
        # Calculate alignment (dot product)
        alignment = np.dot(mean_accel_norm, gravity_norm)
        return max(0.0, alignment)


if __name__ == "__main__":
    # Test the enhanced model
    print("Testing Physics-Informed Speed Estimator...")
    
    # Create model
    model = PhysicsInformedSpeedCNN()
    
    # Create test data
    batch_size, seq_len, features = 8, 150, 6
    test_imu = torch.randn(batch_size, seq_len, features)
    test_speeds = torch.rand(batch_size, 1) * 20  # 0-20 m/s
    
    # Forward pass
    predictions = model(test_imu)
    
    print(f"Input shape: {test_imu.shape}")
    print(f"Speed predictions: {predictions['speed_mean'].shape}")
    print(f"Speed variance: {predictions['speed_variance'].shape}")
    print(f"Scenario probabilities: {predictions['scenario_probs'].shape}")
    print(f"Mount probabilities: {predictions['mount_probs'].shape}")
    
    # Test loss function
    loss, components = physics_informed_loss(predictions, test_speeds, test_imu)
    print(f"\nLoss components:")
    for name, value in components.items():
        print(f"  {name}: {value:.4f}")
    
    # Test temporal validator
    validator = TemporalPhysicsValidator()
    
    # Simulate some predictions
    for i in range(10):
        speed = float(predictions['speed_mean'][0, 0])  # [batch, 1] -> scalar
        accel = test_imu[0, -1, 0:3].numpy()
        scenario = torch.argmax(predictions['scenario_probs'][0]).item()
        mount = torch.argmax(predictions['mount_probs'][0]).item()
        
        validator.update_history(speed, accel, scenario, mount)
        corrected_speed, confidence = validator.validate_prediction(speed, accel, scenario)
        
        if i % 3 == 0:
            print(f"Step {i}: Original={speed:.2f}, Corrected={corrected_speed:.2f}, Confidence={confidence:.3f}")
    
    print("\nPhysics-informed speed estimator test completed!")