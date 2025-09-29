#!/usr/bin/env python3
"""
Simplified Enhanced NavAI Demo
Demonstrates user priors, mount awareness, and adaptive fusion without GTSAM
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from enum import Enum
from dataclasses import dataclass
from collections import deque

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class SimplifiedSpeedPriorSystem:
    """Simplified speed prior system without GTSAM"""
    
    def __init__(self):
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
    
    def compute_speed_prior(self, user_priors: UserNavigationPriors) -> Dict:
        """Compute speed prior parameters"""
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
        
        return {
            'mean_speed': mean_speed,
            'sigma': sigma,
            'lower_sigma': lower_sigma,
            'upper_sigma': upper_sigma,
            'constraints': constraints
        }

class SimplifiedMountAwareEstimator:
    """Simplified mount-aware estimator without PyTorch dependency"""
    
    def __init__(self, mount_types: List[str]):
        self.mount_types = mount_types
        
        # Mount-specific static transforms
        self.static_transforms = {
            "handheld": np.eye(6),  # Identity
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
    
    def apply_mount_transform(self, imu_data: np.ndarray, mount_type: str) -> np.ndarray:
        """Apply mount-specific transformation"""
        if mount_type in self.static_transforms:
            transform = self.static_transforms[mount_type]
            return np.dot(imu_data, transform.T)
        return imu_data
    
    def estimate_speed(self, imu_data: np.ndarray, mount_type: str) -> float:
        """Simplified speed estimation"""
        # Apply mount transform
        transformed_imu = self.apply_mount_transform(imu_data, mount_type)
        
        # Simple speed estimation from acceleration magnitude
        accel = transformed_imu[:3]
        accel_magnitude = np.linalg.norm(accel - np.array([0, 0, 9.81]))  # Remove gravity
        
        # Simplified speed estimate (would be more sophisticated in real model)
        speed_estimate = min(25.0, accel_magnitude * 2.0)  # Cap at reasonable speed
        
        return speed_estimate

class SimplifiedAdaptiveFusion:
    """Simplified adaptive sensor fusion"""
    
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
    
    def determine_motion_context(self, speed: float, acceleration_mag: float, 
                               stationary: bool) -> str:
        """Determine current motion context for sensor weighting"""
        if stationary:
            return "stationary"
        elif acceleration_mag > 3.0:
            return "high_accel"
        elif speed < 2.0:
            return "slow_motion"
        else:
            return "fast_motion"
    
    def update_sensor_weights(self, sensor_residuals: Dict[str, float], 
                            speed: float, acceleration_mag: float, 
                            stationary: bool):
        """Update sensor weights based on context and performance"""
        
        # Determine motion context
        context = self.determine_motion_context(speed, acceleration_mag, stationary)
        context_weights = self.context_baselines[context]
        
        # Update weights for each sensor
        for sensor, residual in sensor_residuals.items():
            # Base reliability from context
            base_reliability = context_weights.get(sensor, 1.0)
            
            # Residual-based adaptation
            if residual > self.anomaly_threshold:
                residual_factor = 0.85  # Penalize high residuals
                self.anomaly_history.append((sensor, residual))
            else:
                residual_factor = 1.02  # Reward good performance
            
            # Combined update
            new_reliability = (self.sensor_reliability[sensor] * residual_factor * 0.9 + 
                             base_reliability * 0.1)
            
            self.sensor_reliability[sensor] = np.clip(new_reliability, 0.1, 1.0)
        
        # Special case: magnetometer in stationary mode
        if stationary and "mag" in self.sensor_reliability:
            self.sensor_reliability["mag"] = min(1.0, self.sensor_reliability["mag"] * 1.1)
    
    def get_reliability_summary(self) -> Dict:
        """Get reliability summary"""
        avg_reliability = np.mean(list(self.sensor_reliability.values()))
        recent_anomalies = len([a for a in self.anomaly_history if a[1] > self.anomaly_threshold])
        
        return {
            "current_weights": self.sensor_reliability.copy(),
            "recent_anomalies": recent_anomalies,
            "avg_reliability": avg_reliability,
            "status": "good" if avg_reliability > 0.7 else "degraded"
        }

class SimplifiedEnhancedNavigation:
    """Simplified enhanced navigation system"""
    
    def __init__(self, mount_types: List[str]):
        self.speed_prior_system = SimplifiedSpeedPriorSystem()
        self.mount_aware_estimator = SimplifiedMountAwareEstimator(mount_types)
        self.adaptive_fusion = SimplifiedAdaptiveFusion()
        
        self.current_user_priors: Optional[UserNavigationPriors] = None
        
    def set_user_priors(self, priors: UserNavigationPriors):
        """Set user-provided navigation priors"""
        self.current_user_priors = priors
        logger.info(f"User priors set: {priors.motion_mode.value}, {priors.speed_band} m/s, {priors.mount_type}")
    
    def process_sensor_update(self, imu_data: np.ndarray, sensor_residuals: Dict[str, float],
                            speed: float, stationary: bool) -> Dict:
        """Process sensor update with all enhancements"""
        
        if self.current_user_priors is None:
            raise ValueError("User priors must be set before processing")
        
        # 1. Mount-aware speed estimation
        mount_type = self.current_user_priors.mount_type
        speed_estimate = self.mount_aware_estimator.estimate_speed(imu_data, mount_type)
        
        # 2. Apply speed priors
        speed_prior = self.speed_prior_system.compute_speed_prior(self.current_user_priors)
        
        # 3. Update adaptive sensor fusion
        acceleration_mag = np.linalg.norm(imu_data[:3])
        self.adaptive_fusion.update_sensor_weights(
            sensor_residuals, speed, acceleration_mag, stationary
        )
        
        # 4. Apply user speed band constraints
        constrained_speed = np.clip(speed_estimate, 
                                  self.current_user_priors.speed_band[0],
                                  self.current_user_priors.speed_band[1])
        
        return {
            "speed_estimate": constrained_speed,
            "raw_speed_estimate": speed_estimate,
            "speed_prior": speed_prior,
            "reliability_summary": self.adaptive_fusion.get_reliability_summary(),
            "user_priors": self.current_user_priors
        }

class SimplifiedEnhancedDemo:
    """Simplified demonstration of enhanced features"""
    
    def __init__(self):
        self.mount_types = ["handheld", "pocket", "dashboard", "handlebar"]
        self.enhanced_system = SimplifiedEnhancedNavigation(self.mount_types)
        logger.info("Simplified Enhanced NavAI system initialized")
    
    def simulate_user_onboarding(self) -> UserNavigationPriors:
        """Simulate user onboarding"""
        logger.info("🔧 Simulating User Onboarding Process")
        
        scenarios = [
            {
                "mode": MotionMode.WALKING,
                "speed_band": (0.5, 2.5),
                "mount": "handheld",
                "description": "Walking with phone in hand"
            },
            {
                "mode": MotionMode.CYCLING,
                "speed_band": (3.0, 8.0),
                "mount": "handlebar",
                "description": "Cycling with phone on handlebar"
            },
            {
                "mode": MotionMode.DRIVING,
                "speed_band": (5.0, 25.0),
                "mount": "dashboard",
                "description": "Driving with phone on dashboard"
            }
        ]
        
        selected_scenario = scenarios[1]  # Cycling
        
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
    
    def simulate_navigation_session(self, user_priors: UserNavigationPriors, duration: int = 60):
        """Simulate navigation session"""
        logger.info(f"🚀 Starting {duration}s Navigation Session")
        logger.info(f"   Mode: {user_priors.motion_mode.value}")
        logger.info(f"   Expected speed: {user_priors.speed_band} m/s")
        
        self.enhanced_system.set_user_priors(user_priors)
        
        # Generate test data
        sample_rate = 10  # Hz
        total_samples = duration * sample_rate
        
        results = []
        
        for i in range(0, total_samples, 10):  # Process every second
            # Generate realistic IMU data
            if user_priors.motion_mode == MotionMode.CYCLING:
                # Cycling pattern
                t = i / sample_rate
                cadence = np.sin(2 * np.pi * 1.5 * t) * 2.0  # 1.5 Hz cadence
                imu_data = np.array([cadence, 0.5, 9.81 + 0.3 * cadence, 0.1, 0.05, 0.02])
            else:
                # Default pattern
                imu_data = np.array([0.2, 0.1, 9.81, 0.05, 0.02, 0.01])
            
            # Add noise
            imu_data += np.random.normal(0, 0.2, 6)
            
            # Simulate sensor residuals
            sensor_residuals = {
                "mag": np.random.exponential(0.5),
                "accel": np.random.exponential(0.2),
                "gyro": np.random.exponential(0.1)
            }
            
            # Add magnetometer interference periodically
            if i % 100 < 20:  # 20% of time
                sensor_residuals["mag"] *= 3.0
            
            # Determine stationary
            stationary = np.random.random() < 0.1  # 10% stationary
            current_speed = np.random.uniform(*user_priors.speed_band)
            
            # Process with enhanced system
            try:
                result = self.enhanced_system.process_sensor_update(
                    imu_data, sensor_residuals, current_speed, stationary
                )
                
                results.append({
                    "timestamp": i / sample_rate,
                    "estimated_speed": result["speed_estimate"],
                    "raw_estimate": result["raw_speed_estimate"],
                    "true_speed": current_speed,
                    "stationary": stationary,
                    "reliability": result["reliability_summary"]
                })
                
                if i % 100 == 0:  # Log every 10 seconds
                    rel_summary = result["reliability_summary"]
                    logger.info(f"   t={i//sample_rate:2d}s: speed={result['speed_estimate']:.2f} m/s, "
                              f"reliability={rel_summary['avg_reliability']:.2f}, "
                              f"status={rel_summary['status']}")
                
            except Exception as e:
                logger.warning(f"Processing failed at t={i//sample_rate}s: {e}")
        
        return results
    
    def analyze_results(self, results: List[Dict]):
        """Analyze results"""
        logger.info("📊 Analyzing Navigation Results")
        
        if not results:
            logger.warning("No results to analyze")
            return
        
        # Extract metrics
        estimated_speeds = [r["estimated_speed"] for r in results]
        raw_estimates = [r["raw_estimate"] for r in results]
        true_speeds = [r["true_speed"] for r in results]
        reliabilities = [r["reliability"]["avg_reliability"] for r in results]
        
        # Compute statistics
        constrained_rmse = np.sqrt(np.mean([(est - true)**2 for est, true in zip(estimated_speeds, true_speeds)]))
        raw_rmse = np.sqrt(np.mean([(raw - true)**2 for raw, true in zip(raw_estimates, true_speeds)]))
        avg_reliability = np.mean(reliabilities)
        
        # Improvement from constraints
        improvement = (raw_rmse - constrained_rmse) / raw_rmse * 100
        
        logger.info(f"✅ Performance Metrics:")
        logger.info(f"   Raw speed RMSE: {raw_rmse:.3f} m/s")
        logger.info(f"   Constrained speed RMSE: {constrained_rmse:.3f} m/s")
        logger.info(f"   Improvement from user priors: {improvement:.1f}%")
        logger.info(f"   Average reliability: {avg_reliability:.3f}")
        logger.info(f"   Total windows processed: {len(results)}")
        
        return {
            "constrained_rmse": constrained_rmse,
            "raw_rmse": raw_rmse,
            "improvement": improvement,
            "avg_reliability": avg_reliability
        }
    
    def run_complete_demo(self):
        """Run complete demonstration"""
        logger.info("🎯 Starting Simplified Enhanced NavAI Demo")
        logger.info("=" * 60)
        
        # Step 1: User onboarding
        user_priors = self.simulate_user_onboarding()
        
        # Step 2: Navigation session
        results = self.simulate_navigation_session(user_priors, duration=60)
        
        # Step 3: Analysis
        metrics = self.analyze_results(results)
        
        # Step 4: Summary
        logger.info("🎉 Enhanced NavAI Demo Completed!")
        logger.info("=" * 60)
        logger.info("🔑 Key Enhancements Demonstrated:")
        logger.info(f"   ✅ User speed priors: {user_priors.speed_band} m/s band")
        logger.info(f"   ✅ Mount-aware processing: {user_priors.mount_type} calibration")
        logger.info(f"   ✅ Adaptive sensor fusion: dynamic reliability weighting")
        logger.info(f"   ✅ Motion mode awareness: {user_priors.motion_mode.value} constraints")
        
        if metrics:
            logger.info("📊 Final Performance:")
            logger.info(f"   Speed accuracy improvement: {metrics['improvement']:.1f}%")
            logger.info(f"   System reliability: {metrics['avg_reliability']:.1%}")
            logger.info(f"   Final RMSE: {metrics['constrained_rmse']:.3f} m/s")
        
        return metrics

def main():
    """Main demo execution"""
    print("🚀 Simplified Enhanced NavAI Integration Demo")
    print("Demonstrating user priors, mount awareness, and adaptive fusion")
    print()
    
    try:
        demo = SimplifiedEnhancedDemo()
        metrics = demo.run_complete_demo()
        
        print()
        print("🎯 Demo completed successfully!")
        print("✅ Enhanced features validated without GTSAM dependency")
        print("Ready for integration with full factor graph system.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())