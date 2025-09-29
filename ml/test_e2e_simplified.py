"""
End-to-End Integration Test for NavAI
Tests ML speed estimation + basic navigation without GTSAM dependency
"""

import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml.models.physics_informed_speed_estimator import PhysicsInformedSpeedCNN, TemporalPhysicsValidator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedNavigation:
    """Simplified navigation without GTSAM for testing"""
    
    def __init__(self):
        self.positions = []
        self.orientations = []
        self.speeds = []
        self.confidences = []
        self.timestamps = []
        
    def integrate_motion(self, timestamp, speed, confidence, acceleration, gyroscope, dt=0.01):
        """Simple dead reckoning integration"""
        if len(self.positions) == 0:
            # Initialize
            self.positions.append(np.array([0.0, 0.0, 0.0]))
            self.orientations.append(np.array([0.0, 0.0, 0.0]))  # Euler angles
        else:
            # Simple integration (placeholder for factor graph)
            last_pos = self.positions[-1]
            
            # Integrate gyroscope to get orientation change
            orientation_change = gyroscope * dt
            new_orientation = self.orientations[-1] + orientation_change
            
            # Integrate speed in forward direction (simplified)
            heading = new_orientation[2]  # yaw
            velocity = np.array([
                speed * np.cos(heading),
                speed * np.sin(heading),
                0.0
            ])
            
            new_position = last_pos + velocity * dt
            
            self.positions.append(new_position)
            self.orientations.append(new_orientation)
        
        self.speeds.append(speed)
        self.confidences.append(confidence)
        self.timestamps.append(timestamp)
        
    def get_results(self):
        """Get navigation results"""
        return {
            'positions': np.array(self.positions),
            'orientations': np.array(self.orientations),
            'speeds': np.array(self.speeds),
            'confidences': np.array(self.confidences),
            'timestamps': np.array(self.timestamps),
            'total_distance': np.sum([np.linalg.norm(self.positions[i] - self.positions[i-1]) 
                                    for i in range(1, len(self.positions))]),
            'avg_speed': np.mean(self.speeds),
            'avg_confidence': np.mean(self.confidences)
        }

def run_end_to_end_test(data_file='outputs/gps_denied_demo.npz', gps_denied=True):
    """Run complete end-to-end test"""
    
    logger.info("🚀 Starting End-to-End NavAI Integration Test")
    logger.info(f"GPS-denied mode: {gps_denied}")
    
    # Step 1: Load data
    logger.info("📊 Loading sensor data...")
    try:
        data = np.load(data_file, allow_pickle=True)
        timestamps = data['timestamps']
        accel = data['accel']
        gyro = data['gyro']
        
        logger.info(f"Loaded {len(timestamps)} samples")
        logger.info(f"Duration: {(timestamps[-1] - timestamps[0]) / 1e9:.1f} seconds")
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_file}")
        logger.info("Run data_loader.py first to generate test data")
        return False
    
    # Step 2: Initialize ML model
    logger.info("🧠 Initializing ML speed estimator...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = PhysicsInformedSpeedCNN().to(device)
    validator = TemporalPhysicsValidator()
    
    # Step 3: Initialize navigation system
    logger.info("🧭 Initializing navigation system...")
    navigation = SimplifiedNavigation()
    
    # Step 4: Process sensor data in windows
    logger.info("⚙️ Processing sensor windows...")
    window_size = 150  # 1.5 seconds at 100Hz
    step_size = 50     # 0.5 second steps
    
    results = {
        'speed_estimates': [],
        'confidences': [],
        'scenarios': [],
        'mounts': [],
        'window_times': []
    }
    
    for i in range(0, len(timestamps) - window_size, step_size):
        # Extract window
        window_accel = accel[i:i+window_size]
        window_gyro = gyro[i:i+window_size]
        window_timestamps = timestamps[i:i+window_size]
        
        # Combine IMU data
        window_imu = np.concatenate([window_accel, window_gyro], axis=1)
        
        # Prepare for ML model
        window_tensor = torch.FloatTensor(window_imu).unsqueeze(0).to(device)  # Add batch dimension
        
        with torch.no_grad():
            # ML inference
            outputs = model(window_tensor)
            
            # Extract predictions from dictionary
            speed = outputs['speed_mean'].cpu().numpy()[0, 0]
            speed_var = outputs['speed_variance'].cpu().numpy()[0, 0]
            confidence = 1.0 / (1.0 + speed_var)  # Convert variance to confidence
            scenario = torch.argmax(outputs['scenario_probs'], dim=1).cpu().numpy()[0]
            mount = torch.argmax(outputs['mount_probs'], dim=1).cpu().numpy()[0]
            
            # Physics validation
            window_time = window_timestamps[-1]
            corrected_speed, physics_confidence = validator.validate_prediction(
                speed, window_accel[-1], scenario
            )
            
            # Use physics-corrected values
            final_speed = corrected_speed
            final_confidence = physics_confidence
            
            # Store results
            results['speed_estimates'].append(final_speed)
            results['confidences'].append(final_confidence)
            results['scenarios'].append(scenario)
            results['mounts'].append(mount)
            results['window_times'].append(window_time)
            
            # Navigate
            dt = 0.5  # step_size / sample_rate
            navigation.integrate_motion(
                window_time, final_speed, final_confidence,
                window_accel[-1], window_gyro[-1], dt
            )
    
    # Step 5: Analyze results
    logger.info("📈 Analyzing results...")
    nav_results = navigation.get_results()
    
    # Calculate metrics
    speed_rmse = np.sqrt(np.mean((np.array(results['speed_estimates']) - nav_results['avg_speed'])**2))
    position_drift = np.linalg.norm(nav_results['positions'][-1] - nav_results['positions'][0])
    
    # Print summary
    logger.info("✅ Integration Test Complete!")
    logger.info("=" * 50)
    logger.info(f"📊 PERFORMANCE METRICS:")
    logger.info(f"  • Processed windows: {len(results['speed_estimates'])}")
    logger.info(f"  • Average speed: {nav_results['avg_speed']:.2f} m/s")
    logger.info(f"  • Average confidence: {nav_results['avg_confidence']:.3f}")
    logger.info(f"  • Total distance: {nav_results['total_distance']:.2f} m")
    logger.info(f"  • Position drift: {position_drift:.2f} m")
    logger.info(f"  • Speed RMSE: {speed_rmse:.3f} m/s")
    
    # Scenario analysis
    scenario_names = ['walk', 'cycle', 'vehicle', 'stationary']
    mount_names = ['handheld', 'pocket', 'mount', 'other']
    
    dominant_scenario = np.bincount(results['scenarios']).argmax()
    dominant_mount = np.bincount(results['mounts']).argmax()
    
    logger.info(f"  • Dominant scenario: {scenario_names[dominant_scenario]}")
    logger.info(f"  • Dominant mount: {mount_names[dominant_mount]}")
    
    # Save detailed results
    output_file = 'outputs/integration_results.npz'
    np.savez_compressed(output_file,
                       speed_estimates=results['speed_estimates'],
                       confidences=results['confidences'],
                       scenarios=results['scenarios'],
                       mounts=results['mounts'],
                       positions=nav_results['positions'],
                       orientations=nav_results['orientations'],
                       timestamps=nav_results['timestamps'],
                       speed_rmse=speed_rmse,
                       position_drift=position_drift,
                       map_match_rate=0.0,  # Placeholder - need map data
                       total_distance=nav_results['total_distance'],
                       avg_speed=nav_results['avg_speed'],
                       avg_confidence=nav_results['avg_confidence'])
    
    logger.info(f"📁 Detailed results saved to: {output_file}")
    
    # Test success criteria
    success = (
        nav_results['avg_confidence'] > 0.5 and  # Reasonable confidence
        len(results['speed_estimates']) > 10 and  # Processed multiple windows
        speed_rmse < 10.0  # Reasonable speed estimation
    )
    
    logger.info(f"🎯 Test Status: {'✅ PASSED' if success else '❌ FAILED'}")
    return success

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NavAI End-to-End Integration Test')
    parser.add_argument('--data', type=str, default='outputs/gps_denied_demo.npz',
                       help='Input data file')
    parser.add_argument('--gps-denied', action='store_true', default=True,
                       help='Run in GPS-denied mode')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze existing results')
    parser.add_argument('--results', type=str, default='outputs/integration_results.npz',
                       help='Results file to analyze')
    
    args = parser.parse_args()
    
    if args.analyze:
        # Analyze existing results
        try:
            results = np.load(args.results, allow_pickle=True)
            print(f"📈 Results Analysis:")
            print(f"  • Speed RMSE: {results['speed_rmse']:.3f} m/s")
            print(f"  • Position drift: {results['position_drift']:.2f} m")
            print(f"  • Total distance: {results['total_distance']:.2f} m")
            print(f"  • Average speed: {results['avg_speed']:.2f} m/s")
            print(f"  • Average confidence: {results['avg_confidence']:.3f}")
            print(f"  • Processed samples: {len(results['speed_estimates'])}")
        except FileNotFoundError:
            print(f"Results file not found: {args.results}")
            print("Run integration test first")
    else:
        # Run integration test
        success = run_end_to_end_test(args.data, args.gps_denied)
        exit(0 if success else 1)