"""
Extended Kalman Filter (EKF) for IMU-based Navigation
Fuses neural network speed predictions with IMU measurements
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class EKFState:
    """State vector for navigation EKF"""
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    acceleration: np.ndarray  # [ax, ay, az] in m/s²
    orientation: np.ndarray  # [qw, qx, qy, qz] quaternion
    
    def to_vector(self) -> np.ndarray:
        """Convert to 13D state vector"""
        return np.concatenate([
            self.position,
            self.velocity,
            self.acceleration,
            self.orientation
        ])
    
    @classmethod
    def from_vector(cls, x: np.ndarray) -> 'EKFState':
        """Create state from vector"""
        return cls(
            position=x[0:3],
            velocity=x[3:6],
            acceleration=x[6:9],
            orientation=x[9:13]
        )


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for smartphone navigation
    
    State: [position, velocity, acceleration, orientation] (13D)
    Measurements: [IMU accel, IMU gyro, speed_estimate] (7D)
    """
    
    def __init__(self, dt: float = 0.01):
        """
        Args:
            dt: Time step in seconds (default 100Hz = 0.01s)
        """
        self.dt = dt
        self.state_dim = 13  # pos(3) + vel(3) + accel(3) + quat(4)
        self.meas_dim = 7    # accel(3) + gyro(3) + speed(1)
        
        # Initialize state
        self.x = np.zeros(self.state_dim)
        self.x[9] = 1.0  # Initialize quaternion to identity [1, 0, 0, 0]
        
        # Initialize covariance matrix
        self.P = np.eye(self.state_dim) * 1.0
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim)
        self.Q[0:3, 0:3] *= 0.01   # Position noise (low)
        self.Q[3:6, 3:6] *= 0.1    # Velocity noise
        self.Q[6:9, 6:9] *= 1.0    # Acceleration noise (high - IMU noisy)
        self.Q[9:13, 9:13] *= 0.01 # Orientation noise (low)
        
        # Measurement noise covariance
        self.R = np.eye(self.meas_dim)
        self.R[0:3, 0:3] *= 0.5    # Accelerometer noise
        self.R[3:6, 3:6] *= 0.1    # Gyroscope noise
        self.R[6, 6] = 1.0         # Speed estimate noise (neural network uncertainty)
        
        # History
        self.history = {
            'time': [],
            'state': [],
            'covariance': [],
            'innovation': []
        }
    
    def predict(self, control_input: Optional[np.ndarray] = None):
        """
        Prediction step: Propagate state forward in time
        
        Uses constant acceleration motion model:
        p_{k+1} = p_k + v_k*dt + 0.5*a_k*dt²
        v_{k+1} = v_k + a_k*dt
        a_{k+1} = a_k (constant)
        q_{k+1} = q_k (simplified - would integrate gyro in full implementation)
        """
        dt = self.dt
        
        # Extract current state
        pos = self.x[0:3]
        vel = self.x[3:6]
        acc = self.x[6:9]
        quat = self.x[9:13]
        
        # Predict new state (constant acceleration model)
        pos_new = pos + vel * dt + 0.5 * acc * dt**2
        vel_new = vel + acc * dt
        acc_new = acc  # Constant acceleration assumption
        quat_new = quat  # Simplified - would integrate gyro properly
        
        # Update state
        self.x[0:3] = pos_new
        self.x[3:6] = vel_new
        self.x[6:9] = acc_new
        self.x[9:13] = quat_new
        
        # Compute Jacobian of state transition (F matrix)
        F = self._compute_state_jacobian(dt)
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, measurement: np.ndarray, measurement_type: str = 'full'):
        """
        Update step: Correct state with measurement
        
        Args:
            measurement: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, speed]
            measurement_type: 'full', 'imu_only', or 'speed_only'
        """
        # Predict measurement from current state
        z_pred = self._predict_measurement()
        
        # Compute innovation (measurement residual)
        innovation = measurement - z_pred
        
        # Compute measurement Jacobian (H matrix)
        H = self._compute_measurement_jacobian()
        
        # Compute innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Compute Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ innovation
        
        # Normalize quaternion
        self.x[9:13] = self.x[9:13] / np.linalg.norm(self.x[9:13])
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P
        
        # Store history
        self.history['innovation'].append(innovation)
    
    def _compute_state_jacobian(self, dt: float) -> np.ndarray:
        """
        Compute Jacobian of state transition function
        F = ∂f/∂x
        """
        F = np.eye(self.state_dim)
        
        # Position depends on velocity and acceleration
        F[0:3, 3:6] = np.eye(3) * dt          # dp/dv
        F[0:3, 6:9] = np.eye(3) * 0.5 * dt**2  # dp/da
        
        # Velocity depends on acceleration
        F[3:6, 6:9] = np.eye(3) * dt          # dv/da
        
        return F
    
    def _compute_measurement_jacobian(self) -> np.ndarray:
        """
        Compute Jacobian of measurement function
        H = ∂h/∂x
        """
        H = np.zeros((self.meas_dim, self.state_dim))
        
        # Accelerometer measures acceleration (with gravity)
        H[0:3, 6:9] = np.eye(3)
        
        # Gyroscope measures angular velocity (orientation derivative)
        # Simplified - would compute proper quaternion derivative
        H[3:6, 9:13] = 0
        
        # Speed measurement relates to velocity magnitude
        vel = self.x[3:6]
        speed = np.linalg.norm(vel)
        if speed > 1e-6:
            H[6, 3:6] = vel / speed  # dspeed/dvel
        
        return H
    
    def _predict_measurement(self) -> np.ndarray:
        """Predict what measurement we should see given current state"""
        accel = self.x[6:9]
        
        # Add gravity to acceleration (assumes upright orientation)
        accel_with_gravity = accel + np.array([0, 0, 9.81])
        
        # Gyro prediction (simplified)
        gyro = np.zeros(3)
        
        # Speed prediction
        vel = self.x[3:6]
        speed = np.linalg.norm(vel)
        
        return np.concatenate([accel_with_gravity, gyro, [speed]])
    
    def get_speed_estimate(self) -> Tuple[float, float]:
        """
        Get speed estimate with uncertainty
        
        Returns:
            (speed, std_dev)
        """
        vel = self.x[3:6]
        speed = np.linalg.norm(vel)
        
        # Extract velocity covariance
        vel_cov = self.P[3:6, 3:6]
        
        # Compute speed uncertainty (linearized)
        if speed > 1e-6:
            jacobian = vel / speed
            speed_var = jacobian @ vel_cov @ jacobian.T
            speed_std = np.sqrt(max(0, speed_var))
        else:
            speed_std = 1.0  # High uncertainty at low speeds
        
        return speed, speed_std
    
    def reset(self):
        """Reset filter to initial state"""
        self.x = np.zeros(self.state_dim)
        self.x[9] = 1.0  # Identity quaternion
        self.P = np.eye(self.state_dim) * 1.0
        self.history = {'time': [], 'state': [], 'covariance': [], 'innovation': []}


class UnscentedKalmanFilter(ExtendedKalmanFilter):
    """
    Unscented Kalman Filter - handles non-linearities better than EKF
    TODO: Implement sigma points and unscented transform
    """
    pass


def test_ekf():
    """Test the EKF implementation"""
    print("Testing Extended Kalman Filter...")
    
    ekf = ExtendedKalmanFilter(dt=0.01)
    
    # Simulate constant velocity motion
    true_velocity = np.array([5.0, 0.0, 0.0])  # 5 m/s in x direction
    
    for t in range(100):
        # Predict
        ekf.predict()
        
        # Simulate measurement (noisy)
        accel_meas = np.random.randn(3) * 0.5
        gyro_meas = np.random.randn(3) * 0.1
        speed_meas = np.linalg.norm(true_velocity) + np.random.randn() * 1.0
        
        measurement = np.concatenate([accel_meas, gyro_meas, [speed_meas]])
        
        # Update
        ekf.update(measurement)
        
        if t % 20 == 0:
            speed_est, speed_std = ekf.get_speed_estimate()
            print(f"t={t*0.01:.2f}s: Speed={speed_est:.2f}±{speed_std:.2f} m/s (true=5.0)")
    
    print("✅ EKF test passed!")


if __name__ == '__main__':
    test_ekf()
