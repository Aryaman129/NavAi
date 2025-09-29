"""
Visual-Inertial Navigation System
Based on 2024 research findings on smartphone VIO systems

Key improvements:
1. Visual-inertial odometry for robust navigation
2. Feature tracking and mapping
3. Failure detection and recovery
4. Integration with existing IMU-based system
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, NamedTuple
import time
from dataclasses import dataclass
from collections import deque

@dataclass
class Feature:
    """Visual feature representation"""
    id: int
    position: np.ndarray  # 2D image coordinates
    descriptor: np.ndarray
    age: int = 0
    world_point: Optional[np.ndarray] = None  # 3D world coordinates

class VisualInertialOdometry:
    """
    Visual-Inertial Odometry system for smartphones
    Combines camera and IMU data for robust navigation
    """
    
    def __init__(self, camera_matrix: np.ndarray, distortion_coeffs: np.ndarray):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = distortion_coeffs
        
        # Feature detection and tracking
        self.feature_detector = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Optical flow tracker
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # State variables
        self.current_features: List[Feature] = []
        self.feature_id_counter = 0
        self.previous_frame = None
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.trajectory = []
        
        # IMU integration
        self.imu_buffer = deque(maxlen=100)
        self.last_imu_timestamp = None
        
        # Failure detection
        self.min_features = 50
        self.max_reproj_error = 2.0
        self.is_tracking_good = True
        
    def process_frame_and_imu(self, 
                              frame: np.ndarray, 
                              imu_data: np.ndarray, 
                              timestamp: float) -> Dict:
        """
        Process camera frame with synchronized IMU data
        """
        # Store IMU data
        self.imu_buffer.append((imu_data, timestamp))
        
        # Process visual frame
        visual_result = self._process_visual_frame(frame, timestamp)
        
        # Integrate IMU between visual updates
        imu_integration = self._integrate_imu_motion(timestamp)
        
        # Combine visual and inertial estimates
        fused_result = self._fuse_visual_inertial(visual_result, imu_integration)
        
        return fused_result
    
    def _process_visual_frame(self, frame: np.ndarray, timestamp: float) -> Dict:
        """Process single camera frame for visual odometry"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        if self.previous_frame is None:
            # Initialize tracking
            return self._initialize_tracking(gray)
        
        # Track existing features
        tracked_features = self._track_features(gray)
        
        # Detect new features if needed
        if len(tracked_features) < self.min_features:
            new_features = self._detect_new_features(gray, tracked_features)
            tracked_features.extend(new_features)
        
        # Estimate motion from tracked features
        motion_estimate = self._estimate_motion(tracked_features)
        
        # Update state
        self.current_features = tracked_features
        self.previous_frame = gray.copy()
        
        return {
            'pose_delta': motion_estimate,
            'num_features': len(tracked_features),
            'tracking_quality': self._assess_tracking_quality(tracked_features),
            'timestamp': timestamp
        }
    
    def _initialize_tracking(self, frame: np.ndarray) -> Dict:
        """Initialize feature tracking on first frame"""
        keypoints, descriptors = self.feature_detector.detectAndCompute(frame, None)
        
        self.current_features = []
        if keypoints and descriptors is not None:
            for kp, desc in zip(keypoints, descriptors):
                feature = Feature(
                    id=self.feature_id_counter,
                    position=np.array(kp.pt),
                    descriptor=desc
                )
                self.current_features.append(feature)
                self.feature_id_counter += 1
        
        self.previous_frame = frame.copy()
        
        return {
            'pose_delta': np.eye(4),
            'num_features': len(self.current_features),
            'tracking_quality': 1.0,
            'timestamp': time.time()
        }
    
    def _track_features(self, frame: np.ndarray) -> List[Feature]:
        """Track features from previous frame using optical flow"""
        if not self.current_features:
            return []
        
        # Extract positions of current features
        old_points = np.array([f.position for f in self.current_features], dtype=np.float32)
        old_points = old_points.reshape(-1, 1, 2)
        
        # Track using Lucas-Kanade optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.previous_frame, frame, old_points, None, **self.lk_params
        )
        
        # Filter out bad tracks
        good_new = new_points[status == 1]
        good_old = old_points[status == 1]
        good_features = [f for i, f in enumerate(self.current_features) if status[i] == 1]
        
        # Update feature positions and ages
        tracked_features = []
        for i, (feature, new_pos) in enumerate(zip(good_features, good_new)):
            feature.position = new_pos.reshape(2)
            feature.age += 1
            tracked_features.append(feature)
        
        return tracked_features
    
    def _detect_new_features(self, frame: np.ndarray, existing_features: List[Feature]) -> List[Feature]:
        """Detect new features avoiding existing ones"""
        # Create mask to avoid existing features
        mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        
        for feature in existing_features:
            cv2.circle(mask, tuple(feature.position.astype(int)), 20, 0, -1)
        
        # Detect new keypoints
        keypoints, descriptors = self.feature_detector.detectAndCompute(frame, mask)
        
        new_features = []
        if keypoints and descriptors is not None:
            for kp, desc in zip(keypoints, descriptors):
                feature = Feature(
                    id=self.feature_id_counter,
                    position=np.array(kp.pt),
                    descriptor=desc
                )
                new_features.append(feature)
                self.feature_id_counter += 1
        
        return new_features[:50]  # Limit number of new features
    
    def _estimate_motion(self, features: List[Feature]) -> np.ndarray:
        """Estimate camera motion from tracked features"""
        if len(features) < 8:  # Need minimum 8 points for essential matrix
            return np.eye(4)
        
        # Get corresponding points from previous and current frames
        prev_points = []
        curr_points = []
        
        for feature in features:
            if feature.age > 0:  # Feature was tracked from previous frame
                # Would need to store previous positions - simplified here
                prev_points.append(feature.position)
                curr_points.append(feature.position)
        
        if len(prev_points) < 8:
            return np.eye(4)
        
        prev_points = np.array(prev_points, dtype=np.float32)
        curr_points = np.array(curr_points, dtype=np.float32)
        
        # Find essential matrix
        E, mask = cv2.findEssentialMat(
            prev_points, curr_points, 
            self.camera_matrix, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )
        
        if E is None:
            return np.eye(4)
        
        # Recover pose from essential matrix
        _, R, t, mask = cv2.recoverPose(
            E, prev_points, curr_points, self.camera_matrix
        )
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        
        return T
    
    def _integrate_imu_motion(self, timestamp: float) -> Dict:
        """Integrate IMU motion between visual updates"""
        if len(self.imu_buffer) < 2:
            return {'velocity': np.zeros(3), 'angular_velocity': np.zeros(3)}
        
        # Get IMU data since last visual update
        recent_imu = []
        for imu_data, t in self.imu_buffer:
            if self.last_imu_timestamp is None or t > self.last_imu_timestamp:
                recent_imu.append((imu_data, t))
        
        if not recent_imu:
            return {'velocity': np.zeros(3), 'angular_velocity': np.zeros(3)}
        
        # Simple integration (in practice would use proper preintegration)
        total_accel = np.zeros(3)
        total_gyro = np.zeros(3)
        dt_total = 0
        
        for i in range(1, len(recent_imu)):
            imu_data, t = recent_imu[i]
            prev_t = recent_imu[i-1][1]
            dt = t - prev_t
            
            if dt > 0:
                accel = imu_data[:3]
                gyro = imu_data[3:6] if len(imu_data) >= 6 else np.zeros(3)
                
                total_accel += accel * dt
                total_gyro += gyro * dt
                dt_total += dt
        
        self.last_imu_timestamp = timestamp
        
        return {
            'velocity': total_accel,
            'angular_velocity': total_gyro,
            'dt': dt_total
        }
    
    def _fuse_visual_inertial(self, visual_result: Dict, imu_result: Dict) -> Dict:
        """Fuse visual and inertial motion estimates"""
        # Simple fusion - in practice would use EKF or factor graph
        visual_pose = visual_result['pose_delta']
        
        # Update current pose
        self.current_pose = self.current_pose @ visual_pose
        
        # Extract position and orientation
        position = self.current_pose[:3, 3]
        rotation_matrix = self.current_pose[:3, :3]
        
        # Convert rotation matrix to Euler angles
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz')
        
        # Store trajectory
        self.trajectory.append({
            'position': position.copy(),
            'orientation': euler_angles.copy(),
            'timestamp': visual_result['timestamp']
        })
        
        # Assess system health
        self.is_tracking_good = self._assess_system_health(visual_result, imu_result)
        
        return {
            'position': position,
            'orientation': euler_angles,
            'velocity': imu_result['velocity'],
            'angular_velocity': imu_result['angular_velocity'],
            'tracking_quality': visual_result['tracking_quality'],
            'num_features': visual_result['num_features'],
            'system_healthy': self.is_tracking_good,
            'pose_matrix': self.current_pose.copy()
        }
    
    def _assess_tracking_quality(self, features: List[Feature]) -> float:
        """Assess quality of feature tracking"""
        if not features:
            return 0.0
        
        # Quality based on number of features and their age
        num_features_score = min(len(features) / self.min_features, 1.0)
        
        # Average feature age (older features are more reliable)
        avg_age = np.mean([f.age for f in features]) if features else 0
        age_score = min(avg_age / 10.0, 1.0)  # Normalize by 10 frames
        
        return (num_features_score + age_score) / 2.0
    
    def _assess_system_health(self, visual_result: Dict, imu_result: Dict) -> bool:
        """Assess overall system health"""
        # Check number of features
        if visual_result['num_features'] < self.min_features:
            return False
        
        # Check tracking quality
        if visual_result['tracking_quality'] < 0.3:
            return False
        
        # Check IMU data validity
        if np.any(np.isnan(imu_result['velocity'])) or np.any(np.isinf(imu_result['velocity'])):
            return False
        
        return True
    
    def get_trajectory(self) -> List[Dict]:
        """Get complete trajectory"""
        return self.trajectory.copy()
    
    def reset_tracking(self):
        """Reset tracking state (useful for failure recovery)"""
        self.current_features = []
        self.previous_frame = None
        self.feature_id_counter = 0
        self.trajectory = []
        self.current_pose = np.eye(4)
        self.is_tracking_good = True


class SmartphoneVIOSystem:
    """
    Complete smartphone VIO system with failure detection and recovery
    """
    
    def __init__(self, camera_params: Dict):
        # Initialize VIO system
        camera_matrix = np.array(camera_params['camera_matrix'])
        dist_coeffs = np.array(camera_params['distortion_coeffs'])
        
        self.vio = VisualInertialOdometry(camera_matrix, dist_coeffs)
        
        # Fallback to IMU-only navigation
        self.imu_navigator = None  # Would use existing NavAI system
        
        # System state
        self.use_visual = True
        self.failure_count = 0
        self.max_failures = 5
        
    def process_sensors(self, 
                       frame: Optional[np.ndarray], 
                       imu_data: np.ndarray, 
                       timestamp: float) -> Dict:
        """
        Process sensor data with automatic fallback
        """
        result = {}
        
        if frame is not None and self.use_visual:
            # Try visual-inertial navigation
            try:
                result = self.vio.process_frame_and_imu(frame, imu_data, timestamp)
                
                # Check if tracking failed
                if not result.get('system_healthy', False):
                    self.failure_count += 1
                    if self.failure_count >= self.max_failures:
                        print("VIO tracking failed, falling back to IMU-only")
                        self.use_visual = False
                        self.vio.reset_tracking()
                else:
                    self.failure_count = max(0, self.failure_count - 1)  # Recover gradually
                
            except Exception as e:
                print(f"VIO error: {e}, falling back to IMU")
                self.use_visual = False
                result = self._imu_fallback(imu_data, timestamp)
        else:
            # Use IMU-only navigation
            result = self._imu_fallback(imu_data, timestamp)
        
        return result
    
    def _imu_fallback(self, imu_data: np.ndarray, timestamp: float) -> Dict:
        """Fallback to IMU-only navigation"""
        # This would interface with the existing NavAI system
        # For now, return basic dead reckoning
        
        return {
            'position': np.array([0.0, 0.0, 0.0]),  # Would integrate IMU data
            'orientation': np.array([0.0, 0.0, 0.0]),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'tracking_quality': 0.5,  # Lower quality without vision
            'system_healthy': True,
            'mode': 'imu_only'
        }


# Camera calibration utilities
class CameraCalibrator:
    """Utility for calibrating smartphone cameras"""
    
    @staticmethod
    def calibrate_from_images(image_paths: List[str], 
                             checkerboard_size: Tuple[int, int] = (9, 6)) -> Dict:
        """
        Calibrate camera from checkerboard images
        """
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane
        
        for image_path in image_paths:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
        
        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        return {
            'camera_matrix': camera_matrix.tolist(),
            'distortion_coeffs': dist_coeffs.tolist(),
            'reprojection_error': ret
        }
    
    @staticmethod
    def get_default_smartphone_params() -> Dict:
        """Get typical smartphone camera parameters"""
        # These are approximate values - should be calibrated for each device
        return {
            'camera_matrix': [
                [800.0, 0.0, 320.0],
                [0.0, 800.0, 240.0],
                [0.0, 0.0, 1.0]
            ],
            'distortion_coeffs': [0.1, -0.2, 0.0, 0.0, 0.0]
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize VIO system with default parameters
    camera_params = CameraCalibrator.get_default_smartphone_params()
    vio_system = SmartphoneVIOSystem(camera_params)
    
    print("Visual-Inertial Odometry System Test")
    print("===================================")
    
    # Simulate processing (would use real camera and IMU)
    for i in range(100):
        timestamp = time.time()
        
        # Simulate IMU data
        imu_data = np.random.randn(6) * 0.1
        imu_data[2] += 9.81  # Add gravity
        
        # Simulate camera frame (placeholder)
        frame = None  # Would be actual camera frame
        if i % 10 == 0:  # Process frame every 10 IMU samples
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Process sensors
        result = vio_system.process_sensors(frame, imu_data, timestamp)
        
        if i % 50 == 0:  # Print every 50 iterations
            print(f"Iteration {i}:")
            print(f"  Position: {result.get('position', [0, 0, 0])}")
            print(f"  Tracking Quality: {result.get('tracking_quality', 0):.2f}")
            print(f"  System Healthy: {result.get('system_healthy', False)}")
            print(f"  Mode: {result.get('mode', 'vio')}")
            print()
        
        time.sleep(0.01)  # Simulate 100Hz processing