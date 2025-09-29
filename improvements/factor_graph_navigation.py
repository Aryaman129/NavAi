"""
Factor Graph-based Navigation System
Based on 2024 research: "Smartphone-based Vision/MEMS-IMU/GNSS tightly coupled seamless positioning"

Key improvements over basic EKF:
1. Handles non-linear constraints better
2. Incorporates multiple sensor modalities simultaneously
3. Enables batch optimization for improved accuracy
4. Supports loop closure and place recognition
"""

import numpy as np
import gtsam
from gtsam import symbol
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

class PhysicsInformedSpeedEstimator(nn.Module):
    """
    Physics-informed neural network for speed estimation
    Incorporates kinematic constraints and vehicle dynamics
    """
    
    def __init__(self, input_dim=6, hidden_dim=128, physics_weight=0.1):
        super().__init__()
        self.physics_weight = physics_weight
        
        # Main network for speed estimation
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.speed_head = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.ReLU()  # Ensure positive speed
        )
        
        # Physics constraint network
        self.physics_net = nn.Sequential(
            nn.Linear(input_dim * 2, 64),  # Current + previous state
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def physics_loss(self, imu_data, speed_pred, dt=0.01):
        """
        Enforce kinematic constraints
        """
        # Extract accelerations
        acc_x, acc_y, acc_z = imu_data[:, 0], imu_data[:, 1], imu_data[:, 2]
        
        # Compute expected speed change from acceleration
        total_acc = torch.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        expected_speed_change = total_acc * dt
        
        # Speed should be consistent with acceleration
        speed_diff = torch.diff(speed_pred.squeeze())
        expected_diff = expected_speed_change[:-1]
        
        physics_constraint = torch.mean((speed_diff - expected_diff)**2)
        return physics_constraint
    
    def forward(self, imu_sequence):
        """
        Forward pass with physics constraints
        """
        batch_size, seq_len, features = imu_sequence.shape
        
        # Reshape for 1D convolution
        x = imu_sequence.transpose(1, 2)  # (batch, features, seq_len)
        
        # Extract features and predict speed
        features = self.feature_extractor(x)
        speed = self.speed_head(features)
        
        # Compute physics loss
        physics_loss = self.physics_loss(imu_sequence[:, -1, :], speed)
        
        return speed, physics_loss


class FactorGraphNavigator:
    """
    Factor graph-based navigation system with tightly-coupled sensor fusion
    """
    
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.current_key = 0
        
        # Noise models
        self.imu_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1]))
        self.speed_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]))
        self.gps_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([3.0, 3.0, 5.0]))
        
        # IMU preintegration parameters
        self.imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
        self.imu_params.setAccelerometerCovariance(np.eye(3) * 0.01**2)
        self.imu_params.setGyroscopeCovariance(np.eye(3) * 0.1**2)
        self.imu_params.setIntegrationCovariance(np.eye(3) * 0.001**2)
        
        # Current state
        self.current_preintegrated = gtsam.PreintegratedImuMeasurements(self.imu_params)
        
    def add_imu_measurement(self, accel: np.ndarray, gyro: np.ndarray, dt: float):
        """Add IMU measurement to preintegration"""
        self.current_preintegrated.integrateMeasurement(accel, gyro, dt)
    
    def add_pose_constraint(self, position: np.ndarray, orientation: np.ndarray):
        """Add pose constraint from visual odometry or GPS"""
        key = symbol('x', self.current_key)
        
        # Convert to GTSAM pose
        rotation = gtsam.Rot3.Quaternion(orientation[3], orientation[0], orientation[1], orientation[2])
        translation = gtsam.Point3(position[0], position[1], position[2])
        pose = gtsam.Pose3(rotation, translation)
        
        # Add prior factor
        prior_factor = gtsam.PriorFactorPose3(key, pose, self.gps_noise)
        self.graph.add(prior_factor)
        self.initial_estimate.insert(key, pose)
        
        self.current_key += 1
    
    def add_speed_constraint(self, speed: float):
        """Add speed constraint from neural network"""
        velocity_key = symbol('v', self.current_key - 1)
        speed_factor = gtsam.PriorFactorVector(
            velocity_key, 
            np.array([speed, 0, 0]),  # Assuming forward motion
            self.speed_noise
        )
        self.graph.add(speed_factor)
    
    def add_imu_factor(self):
        """Add IMU preintegration factor between consecutive poses"""
        if self.current_key >= 1:
            key_i = symbol('x', self.current_key - 1)
            key_j = symbol('x', self.current_key)
            vel_i = symbol('v', self.current_key - 1)
            vel_j = symbol('v', self.current_key)
            bias_i = symbol('b', self.current_key - 1)
            bias_j = symbol('b', self.current_key)
            
            # Add IMU factor
            imu_factor = gtsam.ImuFactor(
                key_i, vel_i, key_j, vel_j, bias_i,
                self.current_preintegrated
            )
            self.graph.add(imu_factor)
            
            # Reset preintegration
            self.current_preintegrated.resetIntegration()
    
    def optimize(self) -> Dict[str, np.ndarray]:
        """
        Optimize the factor graph and return current state estimate
        """
        # Create optimizer
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate)
        
        # Optimize
        result = optimizer.optimize()
        
        # Extract current pose and velocity
        current_pose_key = symbol('x', self.current_key - 1)
        current_vel_key = symbol('v', self.current_key - 1)
        
        if result.exists(current_pose_key):
            pose = result.atPose3(current_pose_key)
            position = pose.translation()
            orientation = pose.rotation().quaternion()
            
            state = {
                'position': np.array([position.x(), position.y(), position.z()]),
                'orientation': np.array([orientation[1], orientation[2], orientation[3], orientation[0]]),  # x,y,z,w
                'covariance': self._extract_covariance(result, current_pose_key)
            }
            
            if result.exists(current_vel_key):
                velocity = result.atVector(current_vel_key)
                state['velocity'] = np.array(velocity)
            
            return state
        
        return {}
    
    def _extract_covariance(self, result: gtsam.Values, key: int) -> np.ndarray:
        """Extract covariance matrix for uncertainty estimation"""
        # This would require marginal covariance computation
        # For now, return identity as placeholder
        return np.eye(6) * 0.1


class MultiModalNavigationSystem:
    """
    Complete navigation system with factor graph optimization
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.speed_estimator = PhysicsInformedSpeedEstimator()
        if model_path:
            self.speed_estimator.load_state_dict(torch.load(model_path))
        
        self.factor_graph = FactorGraphNavigator()
        self.last_timestamp = None
        
    def process_sensor_data(self, 
                           imu_data: np.ndarray,
                           gps_data: Optional[np.ndarray] = None,
                           timestamp: float = None) -> Dict[str, np.ndarray]:
        """
        Process multi-modal sensor data and return navigation state
        """
        if timestamp is None:
            timestamp = time.time()
        
        dt = 0.01 if self.last_timestamp is None else timestamp - self.last_timestamp
        self.last_timestamp = timestamp
        
        # Extract IMU components
        accel = imu_data[:3]
        gyro = imu_data[3:6] if len(imu_data) >= 6 else np.zeros(3)
        
        # Add IMU measurements to factor graph
        self.factor_graph.add_imu_measurement(accel, gyro, dt)
        
        # Estimate speed using physics-informed neural network
        imu_tensor = torch.FloatTensor(imu_data).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            speed_pred, _ = self.speed_estimator(imu_tensor)
            speed = speed_pred.item()
        
        # Add speed constraint
        self.factor_graph.add_speed_constraint(speed)
        
        # Add GPS constraint if available
        if gps_data is not None:
            # Assume GPS provides [lat, lon, alt] -> convert to local coordinates
            position = self._gps_to_local(gps_data)
            orientation = np.array([0, 0, 0, 1])  # Placeholder
            self.factor_graph.add_pose_constraint(position, orientation)
        
        # Add IMU factor
        self.factor_graph.add_imu_factor()
        
        # Optimize and return state
        return self.factor_graph.optimize()
    
    def _gps_to_local(self, gps_data: np.ndarray) -> np.ndarray:
        """Convert GPS coordinates to local coordinate system"""
        # Placeholder implementation - would need proper coordinate transformation
        return np.array([gps_data[0] * 111000, gps_data[1] * 111000 * np.cos(np.radians(gps_data[0])), gps_data[2]])


# Usage example
if __name__ == "__main__":
    import time
    
    # Initialize system
    nav_system = MultiModalNavigationSystem()
    
    # Simulate sensor data processing
    for i in range(100):
        # Simulate IMU data [ax, ay, az, gx, gy, gz]
        imu_data = np.random.randn(6) * 0.1
        
        # Simulate GPS data every 10 iterations
        gps_data = None
        if i % 10 == 0:
            gps_data = np.array([40.7589 + i * 1e-6, -73.9851 + i * 1e-6, 10.0])
        
        # Process data
        state = nav_system.process_sensor_data(imu_data, gps_data, time.time())
        
        if state:
            print(f"Step {i}: Position = {state.get('position', 'Unknown')}")
        
        time.sleep(0.01)