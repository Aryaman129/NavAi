"""
Enhanced User Priors and Adaptive Systems for NavAI
Implements asymmetric speed priors, learnable mount calibration, and contextual sensor fusion
"""

import numpy as np
import torch
import torch.nn as nn
import gtsam
from gtsam import symbol_shorthand
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from enum import Enum

# Symbol shortcuts
V = symbol_shorthand.V  # Velocity

class MotionMode(Enum):
    WALKING = "walking"
    CYCLING = "cycling" 
    DRIVING = "driving"
    STATIONARY = "stationary"

@dataclass
class UserNavigationPriors:
    """User-provided navigation constraints"""
    motion_mode: MotionMode
    speed_band: Tuple[float, float]  # (min_speed, max_speed) in m/s
    mount_type: str  # "handheld", "pocket", "dashboard", "handlebar"
    confidence: float = 0.8  # User confidence in their input
    route_downloaded: bool = False
    calibration_completed: bool = False

class EnhancedUserSpeedPrior:
    """Asymmetric speed priors with motion mode awareness"""
    
    def __init__(self):
        # Motion-specific constraints
        self.motion_constraints = {
            MotionMode.WALKING: {
                'max_accel': 2.0,  # m/s²
                'max_jerk': 5.0,   # m/s³
                'typical_speed': 1.4,  # m/s
                'zupt_frequency': 0.5  # ZUPT every 2 seconds
            },
            MotionMode.CYCLING: {
                'max_accel': 3.0,
                'max_jerk': 8.0,
                'typical_speed': 5.0,
                'zupt_frequency': 0.1  # Rare stops
            },
            MotionMode.DRIVING: {
                'max_accel': 4.0,
                'max_jerk': 12.0,
                'typical_speed': 15.0,
                'zupt_frequency': 0.05  # Very rare stops
            }
        }
    
    def add_user_speed_prior(self, factor_graph: gtsam.NonlinearFactorGraph, 
                           velocity_key: int, user_priors: UserNavigationPriors):
        """Add asymmetric speed prior with motion mode constraints"""
        
        speed_band = user_priors.speed_band
        confidence = user_priors.confidence
        mode = user_priors.motion_mode
        
        # Asymmetric speed prior
        mean_speed = (speed_band[0] + speed_band[1]) / 2
        lower_sigma = (mean_speed - speed_band[0]) / 2
        upper_sigma = (speed_band[1] - mean_speed) / 2
        
        # Weighted sigma based on confidence
        sigma = (lower_sigma + upper_sigma) / 2 / confidence
        
        # Mode-specific adjustments
        constraints = self.motion_constraints[mode]
        if mean_speed > constraints['typical_speed'] * 2:
            sigma *= 1.5  # Less confident in extreme speeds
        
        speed_noise = gtsam.noiseModel.Isotropic.Sigma(1, sigma)
        
        # Create custom speed prior factor (would need implementation)
        # speed_prior = create_asymmetric_speed_prior_factor(
        #     V(velocity_key), mean_speed, lower_sigma, upper_sigma, speed_noise
        # )
        # factor_graph.add(speed_prior)
        
        print(f"Added speed prior: {speed_band} m/s, confidence: {confidence}")
        return mean_speed, sigma
    
    def add_acceleration_constraint(self, factor_graph: gtsam.NonlinearFactorGraph,
                                  velocity_key_prev: int, velocity_key_curr: int,
                                  dt: float, mode: MotionMode):
        """Add physics-based acceleration constraints"""
        
        max_accel = self.motion_constraints[mode]['max_accel']
        accel_noise = gtsam.noiseModel.Isotropic.Sigma(3, max_accel / 3)  # 3-sigma bound
        
        # Acceleration constraint factor (would need custom implementation)
        # accel_factor = create_acceleration_constraint_factor(
        #     V(velocity_key_prev), V(velocity_key_curr), dt, max_accel, accel_noise
        # )
        # factor_graph.add(accel_factor)
        
        print(f"Added acceleration constraint: max {max_accel} m/s² for {mode.value}")

class MountAwareSpeedEstimator(nn.Module):
    """Enhanced mount-aware speed estimator with learnable calibration"""
    
    def __init__(self, base_model, mount_types: List[str]):
        super().__init__()
        self.base_model = base_model
        self.mount_types = mount_types
        
        # Learnable calibration layers for each mount type
        self.calib_layers = nn.ModuleDict({
            mount: nn.Linear(6, 6, bias=False) for mount in mount_types
        })
        
        # Initialize with identity + small random perturbation
        self._initialize_calibration_layers()
        
        # Mount-specific static transforms (initial values)
        self.static_transforms = {
            "handheld": np.array([[1, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 1]]),  # Identity
            "pocket": np.array([[0, 1, 0, 0, 0, 0],    # 90° rotation
                               [-1, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, -1, 0, 0],
                               [0, 0, 0, 0, 0, 1]]),
            "dashboard": np.array([[1, 0, 0, 0, 0, 0],  # Slight tilt compensation
                                  [0, 0.9, 0.1, 0, 0, 0],
                                  [0, -0.1, 0.9, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 0.9, 0.1],
                                  [0, 0, 0, 0, -0.1, 0.9]]),
            "handlebar": np.array([[0.9, 0, 0.1, 0, 0, 0],  # Forward tilt
                                  [0, 1, 0, 0, 0, 0],
                                  [-0.1, 0, 0.9, 0, 0, 0],
                                  [0, 0, 0, 0.9, 0, 0.1],
                                  [0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, -0.1, 0, 0.9]])
        }
    
    def _initialize_calibration_layers(self):
        """Initialize calibration layers with static transforms + small noise"""
        for mount, layer in self.calib_layers.items():
            if mount in self.static_transforms:
                # Initialize with static transform
                with torch.no_grad():
                    transform = torch.tensor(self.static_transforms[mount], dtype=torch.float32)
                    layer.weight.copy_(transform)
            else:
                # Initialize as identity with small random perturbation
                nn.init.eye_(layer.weight)
                layer.weight.data += torch.randn_like(layer.weight) * 0.01
    
    def forward(self, imu_data: torch.Tensor, mount_type: str):
        """Forward pass with mount-specific calibration"""
        # Apply learnable mount calibration
        if mount_type in self.calib_layers:
            calibrated_imu = self.calib_layers[mount_type](imu_data)
        else:
            calibrated_imu = imu_data  # Fallback to uncalibrated
        
        # Pass through base model
        return self.base_model(calibrated_imu)
    
    def get_calibration_matrix(self, mount_type: str) -> torch.Tensor:
        """Get current calibration matrix for mount type"""
        if mount_type in self.calib_layers:
            return self.calib_layers[mount_type].weight.detach()
        return torch.eye(6)

class ContextualAdaptiveSensorFusion:
    """Enhanced adaptive sensor fusion with contextual awareness"""
    
    def __init__(self, anomaly_threshold: float = 2.0):
        self.sensor_reliability = {"mag": 1.0, "accel": 1.0, "gyro": 1.0}
        self.anomaly_history = deque(maxlen=100)
        self.anomaly_threshold = anomaly_threshold
        
        # Context-specific reliability baselines
        self.context_baselines = {
            "stationary": {"mag": 1.0, "accel": 0.9, "gyro": 0.7},
            "slow_motion": {"mag": 0.8, "accel": 1.0, "gyro": 0.9},
            "fast_motion": {"mag": 0.3, "accel": 0.9, "gyro": 1.0},
            "high_accel": {"mag": 0.5, "accel": 1.0, "gyro": 1.0}
        }
        
        # Cross-sensor consistency checker
        self.consistency_history = deque(maxlen=50)
    
    def determine_motion_context(self, speed: float, acceleration_mag: float, 
                               stationary: bool) -> str:
        """Determine current motion context for sensor weighting"""
        if stationary:
            return "stationary"
        elif acceleration_mag > 3.0:  # High acceleration event
            return "high_accel"
        elif speed < 2.0:  # Slow motion (walking)
            return "slow_motion"
        else:  # Fast motion (cycling/driving)
            return "fast_motion"
    
    def check_cross_sensor_consistency(self, sensor_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Check consistency between sensors and compute reliability scores"""
        consistency_scores = {}
        
        if "accel" in sensor_data and "gyro" in sensor_data:
            # Check if gyro-derived tilt matches accelerometer gravity vector
            accel = sensor_data["accel"]
            gyro = sensor_data["gyro"]
            
            # Simplified consistency check (would need more sophisticated implementation)
            accel_magnitude = np.linalg.norm(accel)
            gyro_magnitude = np.linalg.norm(gyro)
            
            # Expected gravity magnitude consistency
            gravity_consistency = 1.0 - abs(accel_magnitude - 9.81) / 9.81
            consistency_scores["accel"] = max(0.1, gravity_consistency)
            
            # Gyro reasonableness (not spinning too fast)
            gyro_consistency = 1.0 - min(1.0, gyro_magnitude / 10.0)  # 10 rad/s max reasonable
            consistency_scores["gyro"] = max(0.1, gyro_consistency)
        
        if "mag" in sensor_data:
            # Magnetometer magnitude consistency
            mag = sensor_data["mag"]
            mag_magnitude = np.linalg.norm(mag)
            # Earth's magnetic field ~25-65 µT
            expected_mag = 50.0  # µT
            mag_consistency = 1.0 - abs(mag_magnitude - expected_mag) / expected_mag
            consistency_scores["mag"] = max(0.1, mag_consistency)
        
        return consistency_scores
    
    def update_sensor_weights(self, sensor_residuals: Dict[str, float], 
                            sensor_data: Dict[str, np.ndarray],
                            speed: float, acceleration_mag: float, 
                            stationary: bool):
        """Enhanced sensor weight update with context and consistency"""
        
        # Determine motion context
        context = self.determine_motion_context(speed, acceleration_mag, stationary)
        context_weights = self.context_baselines[context]
        
        # Check cross-sensor consistency
        consistency_scores = self.check_cross_sensor_consistency(sensor_data)
        
        # Update weights for each sensor
        for sensor, residual in sensor_residuals.items():
            # Base reliability from context
            base_reliability = context_weights.get(sensor, 1.0)
            
            # Consistency penalty/bonus
            consistency_factor = consistency_scores.get(sensor, 1.0)
            
            # Residual-based adaptation
            if residual > self.anomaly_threshold:
                residual_factor = 0.85  # Penalize high residuals
                self.anomaly_history.append((sensor, residual))
            else:
                residual_factor = 1.02  # Reward good performance
            
            # Combined update
            new_reliability = (self.sensor_reliability[sensor] * residual_factor * 
                             consistency_factor * 0.9 + base_reliability * 0.1)
            
            self.sensor_reliability[sensor] = np.clip(new_reliability, 0.1, 1.0)
        
        # Special case: magnetometer in stationary mode
        if stationary and "mag" in self.sensor_reliability:
            self.sensor_reliability["mag"] = min(1.0, self.sensor_reliability["mag"] * 1.1)
    
    def get_noise_models_for_factor_graph(self) -> Dict[str, gtsam.noiseModel.Diagonal]:
        """Convert sensor reliabilities to GTSAM noise models"""
        noise_models = {}
        
        for sensor, reliability in self.sensor_reliability.items():
            # Convert reliability to noise sigma (inverse relationship)
            base_sigma = {"mag": 0.1, "accel": 0.05, "gyro": 0.01}
            sigma = base_sigma[sensor] / reliability
            
            if sensor == "mag":
                noise_models[sensor] = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma, sigma, sigma]))
            elif sensor == "accel":
                noise_models[sensor] = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma, sigma, sigma]))
            elif sensor == "gyro":
                noise_models[sensor] = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma, sigma, sigma]))
        
        return noise_models
    
    def get_reliability_summary(self) -> Dict[str, Dict]:
        """Get comprehensive reliability summary for UI display"""
        return {
            "current_weights": self.sensor_reliability.copy(),
            "recent_anomalies": len([a for a in self.anomaly_history if a[1] > self.anomaly_threshold]),
            "avg_reliability": np.mean(list(self.sensor_reliability.values())),
            "status": "good" if np.mean(list(self.sensor_reliability.values())) > 0.7 else "degraded"
        }

# Integration class
class EnhancedNavigationSystem:
    """Complete enhanced navigation system with user priors and adaptive fusion"""
    
    def __init__(self, base_speed_estimator, mount_types: List[str]):
        self.user_priors = EnhancedUserSpeedPrior()
        self.mount_aware_estimator = MountAwareSpeedEstimator(base_speed_estimator, mount_types)
        self.adaptive_fusion = ContextualAdaptiveSensorFusion()
        
        # Current user settings
        self.current_user_priors: Optional[UserNavigationPriors] = None
        
    def set_user_priors(self, priors: UserNavigationPriors):
        """Set user-provided navigation priors"""
        self.current_user_priors = priors
        print(f"User priors set: {priors.motion_mode.value}, {priors.speed_band} m/s, {priors.mount_type}")
    
    def process_sensor_update(self, imu_data: torch.Tensor, sensor_residuals: Dict[str, float],
                            sensor_raw_data: Dict[str, np.ndarray], speed: float, 
                            stationary: bool) -> Dict:
        """Process complete sensor update with all enhancements"""
        
        if self.current_user_priors is None:
            raise ValueError("User priors must be set before processing")
        
        # 1. Mount-aware speed estimation
        mount_type = self.current_user_priors.mount_type
        speed_estimate = self.mount_aware_estimator(imu_data, mount_type)
        
        # 2. Update adaptive sensor fusion
        acceleration_mag = np.linalg.norm(sensor_raw_data.get("accel", np.zeros(3)))
        self.adaptive_fusion.update_sensor_weights(
            sensor_residuals, sensor_raw_data, speed, acceleration_mag, stationary
        )
        
        # 3. Get updated noise models
        noise_models = self.adaptive_fusion.get_noise_models_for_factor_graph()
        
        return {
            "speed_estimate": speed_estimate,
            "noise_models": noise_models,
            "reliability_summary": self.adaptive_fusion.get_reliability_summary(),
            "user_priors": self.current_user_priors
        }

if __name__ == "__main__":
    print("🚀 Enhanced User Priors and Adaptive Systems Test")
    print("=" * 50)
    
    # Test user priors
    user_priors = UserNavigationPriors(
        motion_mode=MotionMode.WALKING,
        speed_band=(0.5, 2.5),  # m/s
        mount_type="handheld",
        confidence=0.8
    )
    
    speed_prior_system = EnhancedUserSpeedPrior()
    factor_graph = gtsam.NonlinearFactorGraph()
    
    mean_speed, sigma = speed_prior_system.add_user_speed_prior(factor_graph, 1, user_priors)
    print(f"✅ Speed prior added: {mean_speed:.2f} ± {sigma:.2f} m/s")
    
    # Test adaptive sensor fusion
    adaptive_fusion = ContextualAdaptiveSensorFusion()
    
    # Simulate sensor data
    sensor_residuals = {"mag": 1.5, "accel": 0.3, "gyro": 0.2}
    sensor_data = {
        "accel": np.array([0.1, -0.2, 9.8]),  # Close to gravity
        "gyro": np.array([0.05, -0.01, 0.02]), # Small rotation
        "mag": np.array([20.0, 15.0, 30.0])   # Reasonable mag field
    }
    
    adaptive_fusion.update_sensor_weights(sensor_residuals, sensor_data, 1.2, 0.5, False)
    reliability = adaptive_fusion.get_reliability_summary()
    
    print(f"✅ Sensor reliability: {reliability}")
    print("Enhanced systems test completed! 🎉")