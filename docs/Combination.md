# Add to existing FactorGraphNavigation class
def add_user_speed_prior(self, speed_band: Tuple[float, float], confidence: float):
    """Add user-provided speed band as soft constraint"""
    mean_speed = (speed_band[0] + speed_band[1]) / 2
    speed_uncertainty = (speed_band[1] - speed_band[0]) / 4  # 2-sigma range
    
    speed_noise = gtsam.noiseModel.Diagonal.Sigmas([speed_uncertainty / confidence])
    speed_prior = create_speed_prior_factor(V(self.current_key), mean_speed, speed_noise)
    self.graph.add(speed_prior)

# Extend existing speed estimator
class MountAwareSpeedEstimator(PhysicsInformedSpeedCNN):
    def __init__(self, mount_transforms: Dict[str, np.ndarray]):
        super().__init__()
        self.mount_transforms = mount_transforms
    
    def forward(self, imu_data, mount_type):
        # Apply mount-specific transform
        transformed_imu = self.apply_mount_transform(imu_data, mount_type)
        return super().forward(transformed_imu)

class AdaptiveSensorFusion:
    def __init__(self):
        self.sensor_reliability = {"mag": 1.0, "accel": 1.0, "gyro": 1.0}
        self.anomaly_history = deque(maxlen=100)
    
    def update_sensor_weights(self, sensor_residuals):
        """Dynamically adjust sensor weights based on recent performance"""
        for sensor, residual in sensor_residuals.items():
            if residual > self.anomaly_threshold:
                self.sensor_reliability[sensor] *= 0.9  # Reduce trust
            else:
                self.sensor_reliability[sensor] = min(1.0, self.sensor_reliability[sensor] * 1.01)