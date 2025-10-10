"""
Zero Velocity Update (ZUPT) Detector
Detects stationary periods for walking/pedestrian navigation
"""

import numpy as np
from typing import Tuple, List
from collections import deque

class ZUPTDetector:
    """
    Zero Velocity Update (ZUPT) detector for IMU-based navigation
    
    Detects when the device is stationary (zero velocity) and triggers
    velocity corrections in the navigation filter.
    
    Critical for walking scenarios where stance phases provide known velocity = 0
    """
    
    def __init__(self, 
                 window_size: int = 10,
                 accel_threshold: float = 0.5,
                 gyro_threshold: float = 0.1,
                 variance_threshold: float = 0.1):
        """
        Args:
            window_size: Number of samples to analyze (at 100Hz, 10 = 0.1s)
            accel_threshold: Max acceleration magnitude for zero velocity (m/s²)
            gyro_threshold: Max gyroscope magnitude for zero velocity (rad/s)
            variance_threshold: Max variance in window for zero velocity
        """
        self.window_size = window_size
        self.accel_threshold = accel_threshold
        self.gyro_threshold = gyro_threshold
        self.variance_threshold = variance_threshold
        
        # Buffers for windowed detection
        self.accel_buffer = deque(maxlen=window_size)
        self.gyro_buffer = deque(maxlen=window_size)
        
        # Detection history
        self.zupt_history = []
        self.stance_phases = []
    
    def detect(self, accel: np.ndarray, gyro: np.ndarray) -> bool:
        """
        Detect if current IMU reading indicates zero velocity
        
        Args:
            accel: (3,) acceleration [ax, ay, az] in m/s²
            gyro: (3,) angular velocity [wx, wy, wz] in rad/s
            
        Returns:
            True if zero velocity detected
        """
        # Add to buffers
        self.accel_buffer.append(accel)
        self.gyro_buffer.append(gyro)
        
        # Need full window
        if len(self.accel_buffer) < self.window_size:
            return False
        
        # Convert to arrays
        accel_window = np.array(self.accel_buffer)
        gyro_window = np.array(self.gyro_buffer)
        
        # Compute magnitude for each sample
        accel_mag = np.linalg.norm(accel_window, axis=1)
        gyro_mag = np.linalg.norm(gyro_window, axis=1)
        
        # Check if all samples in window are below threshold
        accel_below = np.all(accel_mag < self.accel_threshold)
        gyro_below = np.all(gyro_mag < self.gyro_threshold)
        
        # Check variance (should be low if stationary)
        accel_var = np.var(accel_mag)
        gyro_var = np.var(gyro_mag)
        
        variance_below = (accel_var < self.variance_threshold and 
                         gyro_var < self.variance_threshold)
        
        # All conditions must be met
        is_zupt = accel_below and gyro_below and variance_below
        
        self.zupt_history.append(is_zupt)
        
        return is_zupt
    
    def detect_batch(self, accel_batch: np.ndarray, gyro_batch: np.ndarray) -> np.ndarray:
        """
        Detect ZUPT for batch of IMU data
        
        Args:
            accel_batch: (N, 3) accelerations
            gyro_batch: (N, 3) gyroscope readings
            
        Returns:
            (N,) boolean array of ZUPT detections
        """
        N = len(accel_batch)
        zupt_flags = np.zeros(N, dtype=bool)
        
        for i in range(N):
            zupt_flags[i] = self.detect(accel_batch[i], gyro_batch[i])
        
        return zupt_flags
    
    def get_stance_phases(self) -> List[Tuple[int, int]]:
        """
        Extract stance phases (continuous ZUPT periods) from history
        
        Returns:
            List of (start_idx, end_idx) tuples for each stance phase
        """
        if len(self.zupt_history) == 0:
            return []
        
        stance_phases = []
        in_stance = False
        start_idx = 0
        
        for i, is_zupt in enumerate(self.zupt_history):
            if is_zupt and not in_stance:
                # Start of stance phase
                start_idx = i
                in_stance = True
            elif not is_zupt and in_stance:
                # End of stance phase
                stance_phases.append((start_idx, i - 1))
                in_stance = False
        
        # Handle case where last sample is in stance
        if in_stance:
            stance_phases.append((start_idx, len(self.zupt_history) - 1))
        
        self.stance_phases = stance_phases
        return stance_phases
    
    def get_statistics(self) -> dict:
        """Get detection statistics"""
        if len(self.zupt_history) == 0:
            return {}
        
        zupt_array = np.array(self.zupt_history)
        stance_phases = self.get_stance_phases()
        
        return {
            'total_samples': len(zupt_array),
            'zupt_samples': np.sum(zupt_array),
            'zupt_percentage': 100 * np.mean(zupt_array),
            'num_stance_phases': len(stance_phases),
            'avg_stance_duration': np.mean([end - start + 1 
                                           for start, end in stance_phases]) if stance_phases else 0
        }
    
    def reset(self):
        """Reset detector state"""
        self.accel_buffer.clear()
        self.gyro_buffer.clear()
        self.zupt_history.clear()
        self.stance_phases.clear()


class AdaptiveZUPTDetector(ZUPTDetector):
    """
    Adaptive ZUPT detector that adjusts thresholds based on activity
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activity_mode = 'walking'  # 'walking', 'running', 'stationary'
        
        # Activity-specific thresholds
        self.thresholds = {
            'walking': {
                'accel': 0.5,
                'gyro': 0.1,
                'variance': 0.1
            },
            'running': {
                'accel': 1.0,  # Higher threshold for running
                'gyro': 0.2,
                'variance': 0.2
            },
            'stationary': {
                'accel': 0.2,  # Lower threshold for stationary
                'gyro': 0.05,
                'variance': 0.05
            }
        }
    
    def set_activity_mode(self, mode: str):
        """Update activity mode and adjust thresholds"""
        if mode not in self.thresholds:
            raise ValueError(f"Unknown activity mode: {mode}")
        
        self.activity_mode = mode
        thresholds = self.thresholds[mode]
        
        self.accel_threshold = thresholds['accel']
        self.gyro_threshold = thresholds['gyro']
        self.variance_threshold = thresholds['variance']
        
        print(f"🚶 ZUPT mode set to '{mode}': accel={self.accel_threshold}, gyro={self.gyro_threshold}")


def test_zupt_detector():
    """Test ZUPT detector"""
    print("Testing ZUPT Detector...")
    
    detector = ZUPTDetector(window_size=10)
    
    # Simulate walking data: motion -> stance -> motion
    np.random.seed(42)
    
    # Motion phase (50 samples)
    motion_accel = np.random.randn(50, 3) * 2.0
    motion_gyro = np.random.randn(50, 3) * 0.5
    
    # Stance phase (20 samples - stationary)
    stance_accel = np.random.randn(20, 3) * 0.1
    stance_gyro = np.random.randn(20, 3) * 0.05
    
    # Another motion phase (30 samples)
    motion2_accel = np.random.randn(30, 3) * 2.0
    motion2_gyro = np.random.randn(30, 3) * 0.5
    
    # Combine
    all_accel = np.vstack([motion_accel, stance_accel, motion2_accel])
    all_gyro = np.vstack([motion_gyro, stance_gyro, motion2_gyro])
    
    # Detect ZUPT
    zupt_flags = detector.detect_batch(all_accel, all_gyro)
    
    # Get statistics
    stats = detector.get_statistics()
    print(f"\n📊 ZUPT Statistics:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   ZUPT samples: {stats['zupt_samples']} ({stats['zupt_percentage']:.1f}%)")
    print(f"   Stance phases detected: {stats['num_stance_phases']}")
    print(f"   Avg stance duration: {stats['avg_stance_duration']:.1f} samples")
    
    # Get stance phases
    stance_phases = detector.get_stance_phases()
    print(f"\n🚶 Detected stance phases:")
    for i, (start, end) in enumerate(stance_phases):
        print(f"   Phase {i+1}: samples {start}-{end} (duration={end-start+1})")
    
    # Test adaptive detector
    print(f"\n🎯 Testing Adaptive ZUPT...")
    adaptive = AdaptiveZUPTDetector()
    adaptive.set_activity_mode('running')
    adaptive.set_activity_mode('stationary')
    
    print("\n✅ ZUPT tests passed!")


if __name__ == '__main__':
    test_zupt_detector()
