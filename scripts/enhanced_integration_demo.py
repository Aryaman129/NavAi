#!/usr/bin/env python3
"""
Enhanced NavAI Integration Demo
Demonstrates user priors, mount awareness, and adaptive fusion
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml.models.physics_informed_speed_estimator import PhysicsInformedSpeedCNN, TemporalPhysicsValidator
from improvements.enhanced_user_priors import (
    EnhancedNavigationSystem, UserNavigationPriors, MotionMode,
    ContextualAdaptiveSensorFusion, MountAwareSpeedEstimator
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedNavAIDemo:
    """Comprehensive demo of enhanced NavAI features"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize base models
        self.base_speed_estimator = PhysicsInformedSpeedCNN().to(self.device)
        self.validator = TemporalPhysicsValidator()
        
        # Mount types supported
        self.mount_types = ["handheld", "pocket", "dashboard", "handlebar"]
        
        # Initialize enhanced system
        self.enhanced_system = EnhancedNavigationSystem(
            self.base_speed_estimator, self.mount_types
        )
        
        logger.info("Enhanced NavAI system initialized")
    
    def simulate_user_onboarding(self) -> UserNavigationPriors:
        """Simulate user onboarding process"""
        logger.info("🔧 Simulating User Onboarding Process")
        
        # Simulate user selections (in real app, these would come from UI)
        scenarios = [
            {
                "mode": MotionMode.WALKING,
                "speed_band": (0.5, 2.5),  # m/s
                "mount": "handheld",
                "description": "Walking with phone in hand"
            },
            {
                "mode": MotionMode.CYCLING,
                "speed_band": (3.0, 8.0),  # m/s  
                "mount": "handlebar",
                "description": "Cycling with phone on handlebar"
            },
            {
                "mode": MotionMode.DRIVING,
                "speed_band": (5.0, 25.0),  # m/s
                "mount": "dashboard", 
                "description": "Driving with phone on dashboard"
            }
        ]
        
        # Select scenario (in real app, user would choose)
        selected_scenario = scenarios[1]  # Cycling scenario
        
        user_priors = UserNavigationPriors(
            motion_mode=selected_scenario["mode"],
            speed_band=selected_scenario["speed_band"],
            mount_type=selected_scenario["mount"],
            confidence=0.8,
            route_downloaded=True,
            calibration_completed=True
        )
        
        logger.info(f"✅ User selected: {selected_scenario['description']}")
        logger.info(f"   Speed range: {user_priors.speed_band} m/s")
        logger.info(f"   Mount type: {user_priors.mount_type}")
        
        return user_priors
    
    def simulate_calibration_phase(self, user_priors: UserNavigationPriors):
        """Simulate brief calibration phase"""
        logger.info("🎯 Simulating Calibration Phase")
        
        # Simulate calibration data collection (3 seconds)
        calibration_samples = 30  # 10Hz for 3 seconds
        
        if user_priors.motion_mode == MotionMode.WALKING:
            logger.info("   📱 Hold phone steady for 3 seconds...")
            # Simulate stationary IMU data
            imu_data = torch.randn(1, calibration_samples, 6) * 0.1
            imu_data[:, :, 2] += 9.81  # Add gravity to Z-axis
            
        elif user_priors.motion_mode == MotionMode.CYCLING:
            logger.info("   🚴 Secure phone to handlebar and start cycling...")
            # Simulate cycling motion
            imu_data = torch.randn(1, calibration_samples, 6) * 0.5
            imu_data[:, :, 2] += 9.81
            # Add cycling cadence pattern
            t = torch.linspace(0, 3, calibration_samples)
            cadence = torch.sin(2 * np.pi * 1.5 * t) * 2  # 1.5 Hz cadence
            imu_data[:, :, 0] += cadence.unsqueeze(0)
            
        elif user_priors.motion_mode == MotionMode.DRIVING:
            logger.info("   🚗 Place phone on dashboard and drive 50m...")
            # Simulate driving motion
            imu_data = torch.randn(1, calibration_samples, 6) * 0.3
            imu_data[:, :, 2] += 9.81
            # Add forward acceleration
            imu_data[:, :, 1] += 2.0  # Forward acceleration
        
        # Process calibration data
        with torch.no_grad():
            mount_type = user_priors.mount_type
            calibrated_output = self.enhanced_system.mount_aware_estimator(imu_data, mount_type)
        
        logger.info(f"✅ Calibration completed for {mount_type} mount")
        logger.info(f"   Calibration data shape: {imu_data.shape}")
        
        return imu_data
    
    def simulate_navigation_session(self, user_priors: UserNavigationPriors, duration: int = 60):
        """Simulate a navigation session with various scenarios"""
        logger.info(f"🚀 Starting {duration}s Navigation Session")
        logger.info(f"   Mode: {user_priors.motion_mode.value}")
        logger.info(f"   Expected speed: {user_priors.speed_band} m/s")
        
        # Set user priors in system
        self.enhanced_system.set_user_priors(user_priors)
        
        # Simulation parameters
        sample_rate = 10  # Hz
        total_samples = duration * sample_rate
        
        # Generate realistic IMU data based on motion mode
        imu_sequence = self._generate_realistic_imu_sequence(
            user_priors.motion_mode, total_samples, sample_rate
        )
        
        # Process navigation session
        results = []
        window_size = 150  # 15 seconds at 10Hz
        
        for i in range(0, total_samples - window_size, window_size // 2):  # 50% overlap
            # Extract window
            window_data = imu_sequence[i:i + window_size]
            window_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)
            
            # Simulate sensor residuals and raw data
            sensor_residuals = self._simulate_sensor_residuals(i, total_samples)
            sensor_raw_data = {
                "accel": window_data[-1, :3],  # Last accelerometer sample
                "gyro": window_data[-1, 3:],   # Last gyroscope sample
                "mag": np.random.normal([25, 15, 45], [5, 5, 5])  # Simulated magnetometer
            }
            
            # Determine if stationary
            accel_var = np.var(window_data[:, :3], axis=0).mean()
            stationary = accel_var < 0.1
            
            # Current speed estimate (simplified)
            current_speed = np.linalg.norm(np.mean(window_data[:, :3], axis=0)) - 9.81
            current_speed = max(0, current_speed)
            
            # Process with enhanced system
            try:
                result = self.enhanced_system.process_sensor_update(
                    window_tensor, sensor_residuals, sensor_raw_data, 
                    current_speed, stationary
                )
                
                # Extract speed estimate
                speed_output = result["speed_estimate"]
                if isinstance(speed_output, tuple):
                    estimated_speed = speed_output[0].item() if hasattr(speed_output[0], 'item') else speed_output[0]
                else:
                    estimated_speed = speed_output.item() if hasattr(speed_output, 'item') else speed_output
                
                results.append({
                    "timestamp": i / sample_rate,
                    "estimated_speed": estimated_speed,
                    "true_speed": current_speed,
                    "stationary": stationary,
                    "reliability": result["reliability_summary"],
                    "sensor_weights": result["reliability_summary"]["current_weights"]
                })
                
                if i % (sample_rate * 10) == 0:  # Log every 10 seconds
                    rel_summary = result["reliability_summary"]
                    logger.info(f"   t={i//sample_rate:2d}s: speed={estimated_speed:.2f} m/s, "
                              f"reliability={rel_summary['avg_reliability']:.2f}, "
                              f"status={rel_summary['status']}")
                
            except Exception as e:
                logger.warning(f"Processing failed at t={i//sample_rate}s: {e}")
        
        return results
    
    def _generate_realistic_imu_sequence(self, mode: MotionMode, total_samples: int, 
                                       sample_rate: int) -> np.ndarray:
        """Generate realistic IMU data sequence for given motion mode"""
        
        # Initialize with gravity
        imu_data = np.zeros((total_samples, 6))
        imu_data[:, 2] = 9.81  # Gravity on Z-axis
        
        # Add noise
        noise_level = {"walking": 0.2, "cycling": 0.4, "driving": 0.3}[mode.value]
        imu_data += np.random.normal(0, noise_level, imu_data.shape)
        
        # Add mode-specific patterns
        t = np.linspace(0, total_samples / sample_rate, total_samples)
        
        if mode == MotionMode.WALKING:
            # Walking pattern: ~2 Hz step frequency
            step_freq = 2.0
            step_pattern = np.sin(2 * np.pi * step_freq * t) * 1.5
            imu_data[:, 0] += step_pattern  # Forward acceleration
            imu_data[:, 2] += step_pattern * 0.3  # Vertical bounce
            
        elif mode == MotionMode.CYCLING:
            # Cycling pattern: ~1.5 Hz cadence + road vibrations
            cadence_freq = 1.5
            cadence_pattern = np.sin(2 * np.pi * cadence_freq * t) * 2.0
            imu_data[:, 0] += cadence_pattern
            
            # Road vibrations (higher frequency)
            road_vibrations = np.sin(2 * np.pi * 15 * t) * 0.5
            imu_data[:, 1] += road_vibrations
            
        elif mode == MotionMode.DRIVING:
            # Driving pattern: smooth with occasional turns/braking
            # Add some acceleration events
            for accel_start in range(0, total_samples, sample_rate * 15):  # Every 15 seconds
                accel_end = min(accel_start + sample_rate * 3, total_samples)  # 3 second events
                imu_data[accel_start:accel_end, 1] += 2.0  # Forward acceleration
            
            # Add turn events
            for turn_start in range(sample_rate * 5, total_samples, sample_rate * 20):  # Every 20 seconds
                turn_end = min(turn_start + sample_rate * 2, total_samples)  # 2 second turns
                imu_data[turn_start:turn_end, 0] += 1.5  # Lateral acceleration
                imu_data[turn_start:turn_end, 5] += 0.3  # Yaw rate
        
        return imu_data
    
    def _simulate_sensor_residuals(self, current_sample: int, total_samples: int) -> Dict[str, float]:
        """Simulate realistic sensor residuals"""
        
        # Base residuals
        residuals = {
            "mag": np.random.exponential(0.5),
            "accel": np.random.exponential(0.2),
            "gyro": np.random.exponential(0.1)
        }
        
        # Simulate magnetometer interference in urban areas
        if 0.3 < current_sample / total_samples < 0.7:  # Middle portion of journey
            residuals["mag"] *= 3.0  # High interference
        
        # Simulate accelerometer issues during high acceleration
        if np.random.random() > 0.9:  # 10% chance
            residuals["accel"] *= 2.0
        
        return residuals
    
    def analyze_results(self, results: List[Dict]):
        """Analyze navigation session results"""
        logger.info("📊 Analyzing Navigation Results")
        
        if not results:
            logger.warning("No results to analyze")
            return
        
        # Extract metrics
        timestamps = [r["timestamp"] for r in results]
        estimated_speeds = [r["estimated_speed"] for r in results]
        true_speeds = [r["true_speed"] for r in results]
        reliabilities = [r["reliability"]["avg_reliability"] for r in results]
        
        # Compute statistics
        speed_rmse = np.sqrt(np.mean([(est - true)**2 for est, true in zip(estimated_speeds, true_speeds)]))
        avg_reliability = np.mean(reliabilities)
        min_reliability = np.min(reliabilities)
        
        # Sensor weight evolution
        mag_weights = [r["sensor_weights"]["mag"] for r in results]
        accel_weights = [r["sensor_weights"]["accel"] for r in results]
        gyro_weights = [r["sensor_weights"]["gyro"] for r in results]
        
        logger.info(f"✅ Performance Metrics:")
        logger.info(f"   Speed RMSE: {speed_rmse:.3f} m/s")
        logger.info(f"   Average reliability: {avg_reliability:.3f}")
        logger.info(f"   Minimum reliability: {min_reliability:.3f}")
        logger.info(f"   Total windows processed: {len(results)}")
        
        logger.info(f"📈 Sensor Weight Evolution:")
        logger.info(f"   Magnetometer: {mag_weights[0]:.2f} → {mag_weights[-1]:.2f}")
        logger.info(f"   Accelerometer: {accel_weights[0]:.2f} → {accel_weights[-1]:.2f}")
        logger.info(f"   Gyroscope: {gyro_weights[0]:.2f} → {gyro_weights[-1]:.2f}")
        
        # Count degraded reliability periods
        degraded_periods = sum(1 for r in reliabilities if r < 0.7)
        degraded_percentage = degraded_periods / len(reliabilities) * 100
        
        logger.info(f"⚠️  Degraded reliability periods: {degraded_periods}/{len(results)} ({degraded_percentage:.1f}%)")
        
        return {
            "speed_rmse": speed_rmse,
            "avg_reliability": avg_reliability,
            "min_reliability": min_reliability,
            "degraded_percentage": degraded_percentage,
            "total_windows": len(results)
        }
    
    def run_complete_demo(self):
        """Run complete enhanced NavAI demonstration"""
        logger.info("🎯 Starting Enhanced NavAI Complete Demo")
        logger.info("=" * 60)
        
        # Step 1: User onboarding
        user_priors = self.simulate_user_onboarding()
        
        # Step 2: Calibration
        calibration_data = self.simulate_calibration_phase(user_priors)
        
        # Step 3: Navigation session
        results = self.simulate_navigation_session(user_priors, duration=120)  # 2 minutes
        
        # Step 4: Analysis
        metrics = self.analyze_results(results)
        
        # Step 5: Summary
        logger.info("🎉 Enhanced NavAI Demo Completed!")
        logger.info("=" * 60)
        logger.info("🔑 Key Enhancements Demonstrated:")
        logger.info(f"   ✅ User speed priors: {user_priors.speed_band} m/s band")
        logger.info(f"   ✅ Mount-aware processing: {user_priors.mount_type} calibration")
        logger.info(f"   ✅ Adaptive sensor fusion: dynamic reliability weighting")
        logger.info(f"   ✅ Motion mode awareness: {user_priors.motion_mode.value} constraints")
        
        if metrics:
            logger.info("📊 Final Performance:")
            logger.info(f"   Speed accuracy: {metrics['speed_rmse']:.3f} m/s RMSE")
            logger.info(f"   System reliability: {metrics['avg_reliability']:.1%}")
            logger.info(f"   Robustness: {100-metrics['degraded_percentage']:.1f}% uptime")
        
        return metrics

def main():
    """Main demo execution"""
    print("🚀 Enhanced NavAI Integration Demo")
    print("Demonstrating user priors, mount awareness, and adaptive fusion")
    print()
    
    try:
        demo = EnhancedNavAIDemo()
        metrics = demo.run_complete_demo()
        
        print()
        print("🎯 Demo completed successfully!")
        print("Ready for real-world testing and mobile deployment.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())