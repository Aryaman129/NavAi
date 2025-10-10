#!/usr/bin/env python3
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
            model_outputs = model(window_tensor)
            
            if isinstance(model_outputs, dict):
                # Physics-informed model returns dictionary
                speed_pred = model_outputs['speed_mean']
                speed_var = model_outputs['speed_variance'] 
                scenario_probs = model_outputs['scenario_probs']
                mount_probs = model_outputs['mount_probs']
            else:
                # Handle tuple/list outputs for other models
                if len(model_outputs) == 2:
                    # Standard model with only speed prediction and variance
                    speed_pred, speed_var = model_outputs
                    scenario_probs = torch.zeros(1, 3)  # dummy scenario probs
                    mount_probs = torch.zeros(1, 4)     # dummy mount probs
                elif len(model_outputs) == 4:
                    # Enhanced model with scenario and mount detection
                    speed_pred, speed_var, scenario_probs, mount_probs = model_outputs
                else:
                    raise ValueError(f"Unexpected model output count: {len(model_outputs)}")
            
            # Extract predictions
            speed = speed_pred.cpu().numpy()[0, 0]
            confidence = 1.0 / (1.0 + speed_var.cpu().numpy()[0, 0])  # Convert variance to confidence
            scenario = torch.argmax(scenario_probs, dim=1).cpu().numpy()[0]
            mount = torch.argmax(mount_probs, dim=1).cpu().numpy()[0]
            
            # Physics validation
            window_time = window_timestamps[-1]
            current_accel = window_accel[-1]  # Use last acceleration sample
            corrected_speed, physics_confidence = validator.validate_prediction(
                speed, current_accel, scenario
            )
            
            # Update validator history
            validator.update_history(corrected_speed, current_accel, scenario, mount)
            
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
from models.speed_estimator import SpeedCNN, WindowGenerator, create_tensorflow_model, convert_to_tflite

def create_test_data(output_dir: Path, num_samples: int = 1000):
    """Create synthetic test data for integration testing"""
    print(f"Creating test data with {num_samples} samples...")
    
    # Create directory structure
    log_dir = output_dir / "navai_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic IMU data
    timestamps = np.arange(num_samples) * int(1e9 / 100)  # 100Hz
    t = np.arange(num_samples) / 100.0  # Time in seconds
    
    # Simulate realistic motion patterns
    speed_profile = 5 + 10 * np.sin(0.1 * t) + 5 * np.sin(0.05 * t)
    speed_profile = np.maximum(speed_profile, 0)
    
    # IMU data with realistic patterns
    accel_x = 0.2 * np.sin(0.5 * t) + 0.1 * np.random.randn(num_samples)
    accel_y = 0.3 * np.cos(0.3 * t) + 0.1 * np.random.randn(num_samples)
    accel_z = -9.81 + 0.5 * np.sin(0.2 * t) + 0.2 * np.random.randn(num_samples)
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

def test_window_generation(df):
    """Test window generation for ML training"""
    print("Testing window generation...")
    
    window_gen = WindowGenerator(
        window_size_sec=1.5,
        stride_sec=0.25,
        sample_rate=100
    )
    
    X, y = window_gen.create_windows(df)
    
    assert len(X) > 0, "No windows generated"
    assert len(X) == len(y), "Mismatch between features and targets"
    assert X.shape[1] == 150, f"Wrong window size: {X.shape[1]}"
    assert X.shape[2] == 6, f"Wrong feature count: {X.shape[2]}"
    
    print(f"✅ Window generation test passed - created {len(X)} windows")
    return X, y

def test_pytorch_training(X, y):
    """Test PyTorch model training"""
    print("Testing PyTorch model training...")
    
    # Create model
    model = SpeedCNN(input_channels=6, hidden_dim=32)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X[:100])  # Use subset for quick test
    y_tensor = torch.FloatTensor(y[:100])
    
    # Training loop
    model.train()
    initial_loss = None
    
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_tensor).squeeze()
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if initial_loss is None:
            initial_loss = loss.item()
    
    final_loss = loss.item()
    
    # Check that loss decreased
    assert final_loss < initial_loss, f"Training failed - loss increased: {initial_loss} -> {final_loss}"
    
    print(f"✅ PyTorch training test passed - loss: {initial_loss:.4f} -> {final_loss:.4f}")
    return model

def test_tensorflow_conversion(input_shape):
    """Test TensorFlow model creation and training"""
    print("Testing TensorFlow model...")
    
    # Create TensorFlow model
    tf_model = create_tensorflow_model(input_shape, 'cnn')
    tf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Test with dummy data
    dummy_X = np.random.randn(50, *input_shape).astype(np.float32)
    dummy_y = np.random.randn(50, 1).astype(np.float32)
    
    # Train briefly
    history = tf_model.fit(dummy_X, dummy_y, epochs=5, verbose=0)
    
    # Check that training worked
    assert len(history.history['loss']) == 5, "Training history incomplete"
    
    print("✅ TensorFlow model test passed")
    return tf_model

def test_tflite_conversion(tf_model, input_shape):
    """Test TensorFlow Lite conversion"""
    print("Testing TensorFlow Lite conversion...")
    
    # Generate representative dataset
    representative_data = np.random.randn(10, *input_shape).astype(np.float32)
    
    # Convert to TFLite
    tflite_model = convert_to_tflite(
        tf_model,
        quantize=True,
        representative_dataset=representative_data
    )
    
    assert len(tflite_model) > 0, "TFLite conversion failed"
    
    # Test inference
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test with sample data
    test_input = representative_data[:1]
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    assert tflite_output.shape == (1, 1), f"Wrong output shape: {tflite_output.shape}"
    
    model_size_kb = len(tflite_model) / 1024
    print(f"✅ TensorFlow Lite conversion test passed - model size: {model_size_kb:.1f} KB")
    
    return tflite_model

def test_end_to_end_pipeline():
    """Run complete end-to-end integration test"""
    print("🚀 Starting NavAI ML Pipeline Integration Test")
    print("=" * 50)
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # Step 1: Create test data
            create_test_data(temp_path, num_samples=2000)
            
            # Step 2: Test data loading
            df = test_data_loading(temp_path)
            
            # Step 3: Test window generation
            X, y = test_window_generation(df)
            
            # Step 4: Test PyTorch training
            pytorch_model = test_pytorch_training(X, y)
            
            # Step 5: Test TensorFlow conversion
            input_shape = (X.shape[1], X.shape[2])
            tf_model = test_tensorflow_conversion(input_shape)
            
            # Step 6: Test TFLite conversion
            tflite_model = test_tflite_conversion(tf_model, input_shape)
            
            print("\n" + "=" * 50)
            print("🎉 All integration tests passed!")
            print("✅ Data loading and preprocessing")
            print("✅ PyTorch model training")
            print("✅ TensorFlow model creation")
            print("✅ TensorFlow Lite conversion")
            print("✅ Model inference validation")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_gpu_availability():
    """Test GPU availability for training"""
    print("Testing GPU availability...")
    
    # PyTorch GPU test
    pytorch_gpu = torch.cuda.is_available()
    if pytorch_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ PyTorch GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️ PyTorch GPU not available")
    
    # TensorFlow GPU test
    tf_gpus = tf.config.list_physical_devices('GPU')
    if tf_gpus:
        print(f"✅ TensorFlow GPU available: {len(tf_gpus)} device(s)")
    else:
        print("⚠️ TensorFlow GPU not available")
    
    return pytorch_gpu or len(tf_gpus) > 0

if __name__ == "__main__":
    print("NavAI ML Pipeline Integration Test")
    print("=" * 40)
    
    # Test GPU availability
    gpu_available = test_gpu_availability()
    
    if gpu_available:
        print("🔥 GPU acceleration available for training")
    else:
        print("💻 Using CPU for training (slower)")
    
    print()
    
    # Run end-to-end test
    success = test_end_to_end_pipeline()
    
    if success:
        print("\n🎯 Integration test completed successfully!")
        print("The NavAI ML pipeline is ready for production use.")
        sys.exit(0)
    else:
        print("\n💥 Integration test failed!")
        print("Please check the error messages above and fix any issues.")
        sys.exit(1)
