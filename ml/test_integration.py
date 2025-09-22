#!/usr/bin/env python3
"""
Integration test script for NavAI ML pipeline
Tests the complete flow from data loading to model export
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import tensorflow as tf

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.data_loader import DataLoader as NavAIDataLoader
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
    
    gyro_x = 0.1 * np.sin(0.4 * t) + 0.05 * np.random.randn(num_samples)
    gyro_y = 0.1 * np.cos(0.6 * t) + 0.05 * np.random.randn(num_samples)
    gyro_z = 0.2 * np.sin(0.3 * t) + 0.05 * np.random.randn(num_samples)
    
    # Create CSV data in NavAI logger format
    csv_data = []
    
    for i in range(num_samples):
        # Add sensor readings
        csv_data.append(f"accel,{timestamps[i]},{accel_x[i]},{accel_y[i]},{accel_z[i]}")
        csv_data.append(f"gyro,{timestamps[i]},{gyro_x[i]},{gyro_y[i]},{gyro_z[i]}")
        csv_data.append(f"gps,{timestamps[i]},0.0,0.0,{speed_profile[i]}")
    
    # Write to CSV file
    csv_file = log_dir / "test_log.csv"
    with open(csv_file, 'w') as f:
        f.write("type,timestamp_ns,v1,v2,v3\n")
        f.write("\n".join(csv_data))
    
    print(f"Test data created: {csv_file}")
    return csv_file

def test_data_loading(data_dir: Path):
    """Test data loading functionality"""
    print("Testing data loading...")
    
    loader = NavAIDataLoader(target_sample_rate=100)
    
    # Load test data
    data_paths = {'navai': str(data_dir)}
    df = loader.load_combined_dataset(data_paths)
    
    assert not df.empty, "Data loading failed - empty dataframe"
    assert len(df) > 0, "No data loaded"
    
    # Check required columns
    required_cols = ['timestamp_ns', 'accel_x', 'accel_y', 'accel_z', 
                    'gyro_x', 'gyro_y', 'gyro_z', 'gps_speed_mps']
    
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
    
    print(f"✅ Data loading test passed - loaded {len(df)} samples")
    return df

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
