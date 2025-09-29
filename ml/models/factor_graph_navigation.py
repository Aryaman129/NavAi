"""
GTSAM-based Factor Graph Navigation System for NavAI
Integrates IMU preintegration, ML speed estimates, and optional VIO/GPS updates
"""

import numpy as np
import gtsam
from gtsam import symbol_shorthand
from typing import Dict, List, Optional, Tuple, NamedTuple
import logging
from dataclasses import dataclass
from collections import deque
import time

# Symbol shortcuts
X = symbol_shorthand.X  # Pose3 (x,y,z,r,p,y)
V = symbol_shorthand.V  # Velocity3 (vx,vy,vz)
B = symbol_shorthand.B  # Bias (accel_bias, gyro_bias)
L = symbol_shorthand.L  # Landmark positions (for VIO)

logger = logging.getLogger(__name__)


@dataclass
class IMUMeasurement:
    """Single IMU measurement"""
    timestamp: float
    accel: np.ndarray  # [3] acceleration in m/s²
    gyro: np.ndarray   # [3] angular velocity in rad/s


@dataclass
class SpeedMeasurement:
    """ML-based speed measurement"""
    timestamp: float
    speed: float
    variance: float
    confidence: float
    scenario: int  # 0=walk, 1=cycle, 2=vehicle, 3=stationary


@dataclass
class NavigationState:
    """Complete navigation state"""
    timestamp: float
    position: np.ndarray      # [3] x, y, z in meters
    orientation: np.ndarray   # [4] quaternion w, x, y, z
    velocity: np.ndarray      # [3] vx, vy, vz in m/s
    angular_velocity: np.ndarray  # [3] wx, wy, wz in rad/s
    confidence: float
    covariance: np.ndarray    # [6x6] pose covariance


def create_speed_factor(velocity_key: int, speed_measurement: float, noise_model) -> gtsam.CustomFactor:
    """
    Create custom factor for ML speed estimates
    Constrains velocity magnitude to match ML prediction
    """
    def speed_error(this: gtsam.CustomFactor, values: gtsam.Values, jacobians: Optional[List] = None):
        """Error function for speed constraint"""
        try:
            velocity = values.atVector(velocity_key)
            predicted_speed = np.linalg.norm([velocity[0], velocity[1], velocity[2]])
            error = np.array([predicted_speed - speed_measurement])
            
            if jacobians is not None:
                # Jacobian computation (simplified)
                v = np.array([velocity[0], velocity[1], velocity[2]])
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-6:
                    jacobian = v.reshape(1, -1) / v_norm
                else:
                    jacobian = np.zeros((1, 3))
                jacobians[0] = jacobian
                
            return error
        except Exception as e:
            logger.error(f"Speed factor error: {e}")
            return np.array([0.0])
    
    return gtsam.CustomFactor(noise_model, [velocity_key], speed_error)


def create_nonholonomic_factor(velocity_key: int, heading: float, noise_model) -> gtsam.CustomFactor:
    """
    Create non-holonomic constraint factor for vehicles
    Constrains lateral velocity to be small
    """
    def nonhol_error(this: gtsam.CustomFactor, values: gtsam.Values, jacobians: Optional[List] = None):
        """Error function for non-holonomic constraint"""
        try:
            velocity = values.atVector(velocity_key)
            
            # Rotate velocity to body frame
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            
            # Body frame velocity (forward, lateral, vertical)  
            v_lateral = -velocity[0] * sin_h + velocity[1] * cos_h
            error = np.array([v_lateral])
            
            if jacobians is not None:
                # Jacobian for lateral velocity constraint
                jacobian = np.array([[-sin_h, cos_h, 0.0]]).reshape(1, -1)
                jacobians[0] = jacobian
                
            return error
        except Exception as e:
            logger.error(f"Non-holonomic factor error: {e}")
            return np.array([0.0])
    
    return gtsam.CustomFactor(noise_model, [velocity_key], nonhol_error)


class FactorGraphNavigation:
    """
    GTSAM-based factor graph navigation system
    """
    
    def __init__(self, 
                 imu_params: Optional[Dict] = None,
                 window_size: int = 20,
                 optimization_frequency: float = 1.0):  # Hz
        
        # Factor graph and optimization
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.current_key = 0
        self.window_size = window_size
        self.optimization_frequency = optimization_frequency
        self.last_optimization_time = 0.0
        
        # IMU preintegration setup
        self.imu_params = self._setup_imu_params(imu_params)
        self.current_preintegrated = gtsam.PreintegratedImuMeasurements(self.imu_params)
        self.prev_bias = gtsam.imuBias.ConstantBias()
        
        # State tracking
        self.current_state = None
        self.trajectory = deque(maxlen=1000)
        
        # Measurement buffers
        self.imu_buffer = deque(maxlen=1000)
        self.speed_buffer = deque(maxlen=100)
        
        # Noise models
        self.prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # 10cm position, ~6° rotation
        )
        self.prior_velocity_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1])  # 10cm/s velocity
        )
        self.prior_bias_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])  # accel + gyro bias
        )
        
    def _setup_imu_params(self, params: Optional[Dict] = None) -> gtsam.PreintegrationParams:
        """Setup IMU preintegration parameters"""
        if params is None:
            params = {
                'gravity': 9.81,
                'accel_noise_sigma': 0.01,  # m/s²
                'gyro_noise_sigma': 0.001,  # rad/s
                'accel_bias_rw_sigma': 0.001,
                'gyro_bias_rw_sigma': 0.0001,
                'integration_noise_sigma': 0.0001
            }
            
        # Create preintegration parameters
        preint_params = gtsam.PreintegrationParams.MakeSharedU(params['gravity'])
        
        # Set noise models
        accel_cov = np.eye(3) * (params['accel_noise_sigma'] ** 2)
        gyro_cov = np.eye(3) * (params['gyro_noise_sigma'] ** 2)
        integration_cov = np.eye(3) * (params['integration_noise_sigma'] ** 2)
        
        preint_params.setAccelerometerCovariance(accel_cov)
        preint_params.setGyroscopeCovariance(gyro_cov)
        preint_params.setIntegrationCovariance(integration_cov)
        
        return preint_params
    
    def initialize_state(self, 
                        initial_pose: gtsam.Pose3,
                        initial_velocity: np.ndarray,
                        initial_bias: Optional[gtsam.imuBias.ConstantBias] = None):
        """Initialize the navigation state"""
        
        if initial_bias is None:
            initial_bias = gtsam.imuBias.ConstantBias()
            
        # Add prior factors
        self.graph.add(gtsam.PriorFactorPose3(X(0), initial_pose, self.prior_pose_noise))
        self.graph.add(gtsam.PriorFactorVector(V(0), initial_velocity, self.prior_velocity_noise))
        self.graph.add(gtsam.PriorFactorConstantBias(B(0), initial_bias, self.prior_bias_noise))
        
        # Add to initial estimate
        self.initial_estimate.insert(X(0), initial_pose)
        self.initial_estimate.insert(V(0), initial_velocity)
        self.initial_estimate.insert(B(0), initial_bias)
        
        self.prev_bias = initial_bias
        self.current_key = 1
        
        logger.info(f"Initialized factor graph navigation at key {self.current_key}")
    
    def add_imu_measurement(self, imu: IMUMeasurement):
        """Add IMU measurement to preintegration"""
        self.imu_buffer.append(imu)
        
        # Add to current preintegration
        self.current_preintegrated.integrateMeasurement(
            gtsam.Point3(imu.accel[0], imu.accel[1], imu.accel[2]),
            gtsam.Point3(imu.gyro[0], imu.gyro[1], imu.gyro[2]),
            0.01  # Assuming 100Hz IMU
        )
        
    def add_speed_measurement(self, speed_measurement: SpeedMeasurement):
        """Add ML-based speed measurement"""
        self.speed_buffer.append(speed_measurement)
        
    def add_keyframe_and_optimize(self, timestamp: float) -> Optional[NavigationState]:
        """Add new keyframe with accumulated measurements and optimize"""
        
        if self.current_key == 0:
            logger.warning("Navigation not initialized, cannot add keyframe")
            return None
            
        # Add IMU factor from preintegration
        self._add_imu_factor()
        
        # Add speed measurements
        self._add_speed_factors()
        
        # Add scenario-specific constraints
        self._add_scenario_constraints(timestamp)
        
        # Initialize new state variables
        self._initialize_new_state()
        
        # Optimize if needed
        should_optimize = (timestamp - self.last_optimization_time) > (1.0 / self.optimization_frequency)
        
        if should_optimize:
            result = self._optimize()
            if result is not None:
                self.last_optimization_time = timestamp
                return self._extract_navigation_state(result, timestamp)
                
        return None
    
    def _add_imu_factor(self):
        """Add IMU preintegration factor between consecutive poses"""
        if self.current_key <= 0:
            return
            
        # Create IMU factor
        imu_factor = gtsam.ImuFactor(
            X(self.current_key - 1), V(self.current_key - 1),
            X(self.current_key), V(self.current_key),
            B(self.current_key - 1),
            self.current_preintegrated
        )
        
        self.graph.add(imu_factor)
        
        # Add bias evolution factor
        bias_factor = gtsam.BetweenFactorConstantBias(
            B(self.current_key - 1), B(self.current_key),
            gtsam.imuBias.ConstantBias(),  # Zero bias change
            gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.001, 0.001, 0.001]))
        )
        
        self.graph.add(bias_factor)
        
        # Reset preintegration
        self.current_preintegrated.resetIntegrationAndSetBias(self.prev_bias)
        
    def _add_speed_factors(self):
        """Add recent speed measurements as factors"""
        if not self.speed_buffer:
            return
            
        # Use most recent speed measurements
        recent_speeds = list(self.speed_buffer)[-5:]  # Last 5 measurements
        
        for speed_meas in recent_speeds:
            # Convert confidence to noise (higher confidence = lower noise)
            speed_noise_sigma = max(0.1, 2.0 / speed_meas.confidence)
            speed_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([speed_noise_sigma]))
            
            # Add speed factor
            speed_factor = create_speed_factor(V(self.current_key), speed_meas.speed, speed_noise)
            self.graph.add(speed_factor)
            
            # Add non-holonomic constraint for vehicles
            if speed_meas.scenario == 2 and speed_meas.speed > 1.0:  # Vehicle moving
                # Estimate heading from recent trajectory
                heading = self._estimate_current_heading()
                nonhol_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5]))  # 0.5 m/s lateral
                nonhol_factor = create_nonholonomic_factor(V(self.current_key), heading, nonhol_noise)
                self.graph.add(nonhol_factor)
                
    def _add_scenario_constraints(self, timestamp: float):
        """Add scenario-specific physics constraints"""
        if not self.speed_buffer:
            return
            
        recent_speed = self.speed_buffer[-1]
        
        # Zero velocity constraint for stationary scenario
        if recent_speed.scenario == 3 and recent_speed.speed < 0.1:
            zero_vel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01]))
            zero_vel_factor = gtsam.PriorFactorVector(
                V(self.current_key), 
                gtsam.Point3(0, 0, 0), 
                zero_vel_noise
            )
            self.graph.add(zero_vel_factor)
            
    def _initialize_new_state(self):
        """Initialize new state variables in the graph"""
        # Check if state already exists for current key
        if self.initial_estimate.exists(X(self.current_key)):
            return  # Already initialized
            
        if self.current_key == 1:
            # For first keyframe after initialization, predict from priors
            prev_pose = self.initial_estimate.atPose3(X(0))
            prev_velocity = self.initial_estimate.atPoint3(V(0))  # Use atPoint3 for velocity
        else:
            # Predict from previous optimized state
            prev_pose = self.initial_estimate.atPose3(X(self.current_key - 1))
            prev_velocity = self.initial_estimate.atPoint3(V(self.current_key - 1))  # Use atPoint3 for velocity
        
        # Simple constant velocity prediction
        dt = 0.5  # Assume 0.5s between keyframes
        predicted_position = prev_pose.translation() + prev_velocity * dt
        predicted_pose = gtsam.Pose3(prev_pose.rotation(), predicted_position)
        
        # Add to initial estimate only if keys don't exist
        if not self.initial_estimate.exists(X(self.current_key)):
            self.initial_estimate.insert(X(self.current_key), predicted_pose)
        if not self.initial_estimate.exists(V(self.current_key)):
            self.initial_estimate.insert(V(self.current_key), prev_velocity)
        if not self.initial_estimate.exists(B(self.current_key)):
            self.initial_estimate.insert(B(self.current_key), self.prev_bias)
        
    def _optimize(self) -> Optional[gtsam.Values]:
        """Optimize the factor graph"""
        try:
            # Use Levenberg-Marquardt optimizer
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate)
            result = optimizer.optimize()
            
            # Update initial estimate with optimized values
            self.initial_estimate = result
            
            # Maintain sliding window
            self._maintain_sliding_window()
            
            return result
            
        except Exception as e:
            logger.error(f"Factor graph optimization failed: {e}")
            return None
            
    def _maintain_sliding_window(self):
        """Maintain sliding window by marginalizing old states"""
        if self.current_key <= self.window_size:
            return
            
        # Marginalize oldest state (this is simplified - proper marginalization is more complex)
        oldest_key = self.current_key - self.window_size
        
        # Remove factors connected to old states (simplified)
        # In practice, you'd use incremental smoothing (iSAM2)
        
    def _extract_navigation_state(self, result: gtsam.Values, timestamp: float) -> NavigationState:
        """Extract navigation state from optimization result"""
        try:
            # Get current pose and velocity
            current_pose = result.atPose3(X(self.current_key))
            current_velocity = result.atPoint3(V(self.current_key))  # Use atPoint3 for velocity
            
            # Convert pose to position and orientation
            position = np.array([
                current_pose.x(), current_pose.y(), current_pose.z()
            ])
            
            # Convert rotation to quaternion
            rotation = current_pose.rotation()
            quat = rotation.toQuaternion()  # Correct method name
            orientation = np.array([quat.w(), quat.x(), quat.y(), quat.z()])
            
            # Velocity
            velocity = np.array([current_velocity[0], current_velocity[1], current_velocity[2]])
            
            # Angular velocity (would need more sophisticated extraction)
            angular_velocity = np.array([0.0, 0.0, 0.0])  # Placeholder
            
            # Confidence (simplified - would compute from covariance)
            confidence = self._compute_confidence(result)
            
            # Covariance (simplified)
            covariance = np.eye(6) * 0.1  # Placeholder
            
            nav_state = NavigationState(
                timestamp=timestamp,
                position=position,
                orientation=orientation,
                velocity=velocity,
                angular_velocity=angular_velocity,
                confidence=confidence,
                covariance=covariance
            )
            
            # Add to trajectory
            self.trajectory.append(nav_state)
            self.current_state = nav_state
            
            # Increment key for next iteration
            self.current_key += 1
            
            return nav_state
            
        except Exception as e:
            logger.error(f"Failed to extract navigation state: {e}")
            return None
            
    def _estimate_current_heading(self) -> float:
        """Estimate current heading from recent trajectory"""
        if len(self.trajectory) < 2:
            return 0.0
            
        recent_states = list(self.trajectory)[-2:]
        
        # Calculate heading from position change
        pos_diff = recent_states[-1].position[:2] - recent_states[-2].position[:2]
        heading = np.arctan2(pos_diff[1], pos_diff[0])
        
        return heading
        
    def _compute_confidence(self, result: gtsam.Values) -> float:
        """Compute navigation confidence from optimization result"""
        # Simplified confidence calculation
        # In practice, would use marginal covariances
        return 0.8  # Placeholder
        
    def get_trajectory(self) -> List[NavigationState]:
        """Get complete trajectory"""
        return list(self.trajectory)
        
    def get_current_state(self) -> Optional[NavigationState]:
        """Get current navigation state"""
        return self.current_state


# Integration with physics-informed speed estimator
class IntegratedNavigationSystem:
    """
    Complete navigation system integrating physics-informed ML and factor graphs
    """
    
    def __init__(self, 
                 speed_model_path: Optional[str] = None,
                 factor_graph_params: Optional[Dict] = None):
        
        # Import the physics-informed speed estimator
        from physics_informed_speed_estimator import PhysicsInformedSpeedCNN, TemporalPhysicsValidator
        import torch
        
        # Initialize ML speed estimator
        self.speed_estimator = PhysicsInformedSpeedCNN()
        if speed_model_path:
            self.speed_estimator.load_state_dict(torch.load(speed_model_path, weights_only=False))
        self.speed_estimator.eval()
        
        # Initialize physics validator
        self.physics_validator = TemporalPhysicsValidator()
        
        # Initialize factor graph
        self.factor_graph = FactorGraphNavigation(**(factor_graph_params or {}))
        
        # State tracking
        self.initialized = False
        self.imu_sequence_buffer = deque(maxlen=150)  # 1.5 seconds at 100Hz
        
    def initialize(self, 
                   initial_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
                   initial_orientation: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])):
        """Initialize the navigation system"""
        
        # Create initial pose
        rotation = gtsam.Rot3.Quaternion(
            initial_orientation[0], initial_orientation[1], 
            initial_orientation[2], initial_orientation[3]
        )
        translation = gtsam.Point3(initial_position[0], initial_position[1], initial_position[2])
        initial_pose = gtsam.Pose3(rotation, translation)
        
        # Initialize with zero velocity
        initial_velocity = gtsam.Point3(0.0, 0.0, 0.0)
        
        # Initialize factor graph
        self.factor_graph.initialize_state(initial_pose, initial_velocity)
        self.initialized = True
        
        logger.info("Integrated navigation system initialized")
        
    def process_imu_measurement(self, imu: IMUMeasurement) -> Optional[NavigationState]:
        """Process single IMU measurement"""
        if not self.initialized:
            logger.warning("System not initialized")
            return None
            
        # Add to factor graph
        self.factor_graph.add_imu_measurement(imu)
        
        # Add to sequence buffer for ML processing
        imu_vector = np.concatenate([imu.accel, imu.gyro])
        self.imu_sequence_buffer.append(imu_vector)
        
        # Process ML speed estimation when we have enough data
        if len(self.imu_sequence_buffer) >= 150:
            speed_measurement = self._estimate_speed_with_ml(imu.timestamp)
            if speed_measurement:
                self.factor_graph.add_speed_measurement(speed_measurement)
                
        return None  # ML estimation is asynchronous
        
    def _estimate_speed_with_ml(self, timestamp: float) -> Optional[SpeedMeasurement]:
        """Estimate speed using physics-informed ML"""
        import torch
        
        # Prepare input sequence
        sequence = np.array(list(self.imu_sequence_buffer))  # [150, 6]
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # [1, 150, 6]
        
        # Run ML prediction
        with torch.no_grad():
            predictions = self.speed_estimator(input_tensor)
            
        # Extract results
        speed = predictions['speed_mean'][0, 0].item()
        variance = predictions['speed_variance'][0, 0].item()
        scenario_probs = predictions['scenario_probs'][0].numpy()
        scenario = np.argmax(scenario_probs)
        confidence = float(np.max(scenario_probs))
        
        # Validate with temporal physics
        current_accel = sequence[-1, 0:3]
        corrected_speed, physics_confidence = self.physics_validator.validate_prediction(
            speed, current_accel, scenario
        )
        
        # Update physics validator history
        self.physics_validator.update_history(corrected_speed, current_accel, scenario, 0)
        
        # Combine confidences
        final_confidence = confidence * physics_confidence
        
        return SpeedMeasurement(
            timestamp=timestamp,
            speed=corrected_speed,
            variance=variance,
            confidence=final_confidence,
            scenario=scenario
        )
        
    def add_keyframe_and_get_state(self, timestamp: float) -> Optional[NavigationState]:
        """Add keyframe and get optimized navigation state"""
        if not self.initialized:
            return None
            
        return self.factor_graph.add_keyframe_and_optimize(timestamp)
        
    def get_trajectory(self) -> List[NavigationState]:
        """Get complete trajectory"""
        return self.factor_graph.get_trajectory()


if __name__ == "__main__":
    print("Testing GTSAM Factor Graph Navigation System...")
    
    # Test basic factor graph functionality
    nav_system = FactorGraphNavigation()
    
    # Initialize with simple state
    initial_pose = gtsam.Pose3()  # Identity pose
    initial_velocity = gtsam.Point3(0, 0, 0)
    nav_system.initialize_state(initial_pose, initial_velocity)
    
    print("Factor graph initialized successfully")
    
    # Simulate IMU measurements
    for i in range(10):
        timestamp = i * 0.01
        accel = np.array([0.1, 0.0, 9.81]) + np.random.randn(3) * 0.01
        gyro = np.array([0.0, 0.0, 0.1]) + np.random.randn(3) * 0.001
        
        imu = IMUMeasurement(timestamp, accel, gyro)
        nav_system.add_imu_measurement(imu)
        
        # Add speed measurement
        if i % 5 == 0:
            speed_meas = SpeedMeasurement(
                timestamp=timestamp,
                speed=1.0 + 0.1 * np.sin(timestamp),
                variance=0.1,
                confidence=0.8,
                scenario=0  # Walking
            )
            nav_system.add_speed_measurement(speed_meas)
            
            # Try optimization
            result = nav_system.add_keyframe_and_optimize(timestamp)
            if result:
                print(f"Optimized state at {timestamp:.2f}s: pos={result.position[:2]}, vel_mag={np.linalg.norm(result.velocity):.2f}")
    
    print(f"Final trajectory length: {len(nav_system.get_trajectory())}")
    print("GTSAM factor graph navigation test completed!")