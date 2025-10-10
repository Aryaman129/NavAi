"""
Export Phase 1 TensorFlow model to TensorFlow Lite for Android deployment
Configured for the ultra-fast GPU-trained model
"""

import tensorflow as tf
import numpy as np
import json
from pathlib import Path
import pandas as pd

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ TensorFlow GPU available: {gpus[0].name}")
    except RuntimeError as e:
        print(f"⚠️ GPU setup error: {e}")
else:
    print("⚠️ Running on CPU")

def export_to_tflite():
    """Export trained TensorFlow model to TFLite for Android"""
    
    print("\n🚀 Exporting TensorFlow Model to TFLite")
    print("=" * 70)
    
    # Paths
    keras_model_path = Path('ml/outputs/phase1_best_model.keras')
    preprocessor_path = Path('ml/outputs/preprocessor_params.json')
    output_dir = Path('ml/outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model exists
    if not keras_model_path.exists():
        print(f"❌ Model not found: {keras_model_path}")
        print("   Please train the model first using train_phase1_ultra_fast.py")
        return
    
    # Load TensorFlow model
    print(f"\n📥 Loading TensorFlow model from {keras_model_path.name}...")
    try:
        model = tf.keras.models.load_model(keras_model_path)
        print("   ✅ Model loaded successfully!")
        
        # Print model summary
        print("\n📊 Model Architecture:")
        model.summary()
        
        # Get model info
        input_shape = model.input_shape
        output_shape = model.output_shape
        total_params = model.count_params()
        
        print(f"\n   Input shape: {input_shape}")
        print(f"   Output shape: {output_shape}")
        print(f"   Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load preprocessor parameters
    print(f"\n📥 Loading preprocessor parameters...")
    if preprocessor_path.exists():
        with open(preprocessor_path, 'r') as f:
            preprocessor_params_raw = json.load(f)
        
        # Convert from training format to export format
        preprocessor_params = {
            'mean': preprocessor_params_raw['accel_mean'] + preprocessor_params_raw['gyro_mean'],
            'std': preprocessor_params_raw['accel_std'] + preprocessor_params_raw['gyro_std'],
            'window_size': preprocessor_params_raw['window_size'],
            'num_features': 6,  # 3 accel + 3 gyro
            'input_features': [
                'accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z'
            ]
        }
        
        print("   ✅ Preprocessor parameters loaded:")
        print(f"      - Accel mean: [{preprocessor_params['mean'][0]:.3f}, {preprocessor_params['mean'][1]:.3f}, {preprocessor_params['mean'][2]:.3f}]")
        print(f"      - Gyro mean: [{preprocessor_params['mean'][3]:.6f}, {preprocessor_params['mean'][4]:.6f}, {preprocessor_params['mean'][5]:.6f}]")
        print(f"      - Window size: {preprocessor_params['window_size']}")
    else:
        print("   ⚠️ Preprocessor params not found, creating default...")
        # Create default preprocessor params
        preprocessor_params = {
            'mean': [0.0] * 6,  # 6 features from ultra_fast training
            'std': [1.0] * 6,
            'window_size': 100,
            'num_features': 6,
            'input_features': [
                'accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z'
            ]
        }
    
    # Load sample data for representative dataset (for quantization)
    print(f"\n📊 Loading sample data for quantization...")
    data_path = Path('data/comma2k19/comma2k19_processed.parquet')
    
    if data_path.exists():
        try:
            df = pd.read_parquet(data_path)
            print(f"   ✅ Loaded {len(df)} samples from dataset")
            
            # Extract features
            feature_cols = [col for col in df.columns if col != 'speed']
            X_sample = df[feature_cols].values
            
            # Create windows (take first 1000 for representative dataset)
            window_size = preprocessor_params['window_size']
            num_samples = min(1000, len(X_sample) - window_size)
            
            X_windows = []
            for i in range(num_samples):
                window = X_sample[i:i+window_size]
                X_windows.append(window)
            
            X_windows = np.array(X_windows, dtype=np.float32)
            print(f"   ✅ Created {len(X_windows)} windows for representative dataset")
            print(f"   ✅ Window shape: {X_windows.shape}")
            
        except Exception as e:
            print(f"   ⚠️ Error loading data: {e}")
            print(f"   ⚠️ Creating synthetic data instead...")
            # Create synthetic data
            window_size = preprocessor_params['window_size']
            num_features = preprocessor_params['num_features']
            X_windows = np.random.randn(1000, window_size, num_features).astype(np.float32)
    else:
        print(f"   ⚠️ Data file not found: {data_path}")
        print(f"   ⚠️ Creating synthetic data...")
        window_size = preprocessor_params['window_size']
        num_features = preprocessor_params['num_features']
        X_windows = np.random.randn(1000, window_size, num_features).astype(np.float32)
    
    # Test model inference before conversion
    print(f"\n🧪 Testing TensorFlow model inference...")
    test_input = X_windows[0:1]
    try:
        test_output = model.predict(test_input, verbose=0)
        print(f"   ✅ Test prediction: {test_output[0][0]:.4f} m/s")
    except Exception as e:
        print(f"   ❌ Model inference error: {e}")
        return
    
    # Convert to TensorFlow Lite
    print(f"\n📦 Converting to TensorFlow Lite...")
    print("   Configuration:")
    print("   - Target: Mobile/Android")
    print("   - Ops: TFLite builtins + Select TF ops (for LSTM)")
    print("   - Optimization: Dynamic range quantization")
    
    try:
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable necessary ops for LSTM
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard TFLite ops
            tf.lite.OpsSet.SELECT_TF_OPS     # TensorFlow ops (needed for LSTM)
        ]
        
        # Optimization settings
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Representative dataset for quantization
        def representative_dataset():
            for i in range(min(100, len(X_windows))):
                yield [X_windows[i:i+1]]
        
        converter.representative_dataset = representative_dataset
        
        # Disable experimental lowering (can cause LSTM issues)
        converter._experimental_lower_tensor_list_ops = False
        
        # Convert
        print("\n   🔄 Converting (this may take a moment)...")
        tflite_model = converter.convert()
        
        print("   ✅ Conversion successful!")
        
    except Exception as e:
        print(f"   ❌ Conversion error: {e}")
        print("\n   Trying fallback conversion (without quantization)...")
        
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter._experimental_lower_tensor_list_ops = False
            
            tflite_model = converter.convert()
            print("   ✅ Fallback conversion successful!")
            
        except Exception as e2:
            print(f"   ❌ Fallback conversion also failed: {e2}")
            return
    
    # Save TFLite model
    tflite_path = output_dir / 'phase1_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    tflite_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"\n💾 Saved TFLite model:")
    print(f"   Path: {tflite_path}")
    print(f"   Size: {tflite_size_mb:.2f} MB")
    
    # Test TFLite model
    print(f"\n🧪 Testing TFLite model...")
    try:
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   Input details:")
        print(f"      - Shape: {input_details[0]['shape']}")
        print(f"      - Type: {input_details[0]['dtype']}")
        
        print(f"   Output details:")
        print(f"      - Shape: {output_details[0]['shape']}")
        print(f"      - Type: {output_details[0]['dtype']}")
        
        # Test inference
        test_input = X_windows[0:1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        # Compare with TensorFlow model
        tf_output = model.predict(test_input, verbose=0)[0][0]
        
        error = abs(tf_output - tflite_output)
        error_percent = (error / (abs(tf_output) + 1e-6)) * 100
        
        print(f"\n   Inference comparison:")
        print(f"      - TensorFlow output: {tf_output:.6f} m/s")
        print(f"      - TFLite output:     {tflite_output:.6f} m/s")
        print(f"      - Absolute error:    {error:.6f} m/s")
        print(f"      - Relative error:    {error_percent:.2f}%")
        
        if error < 0.01:
            print(f"   ✅ TFLite model matches TensorFlow perfectly!")
        elif error < 0.1:
            print(f"   ✅ TFLite model matches TensorFlow well (acceptable error)")
        else:
            print(f"   ⚠️ TFLite conversion may have precision issues")
        
    except Exception as e:
        print(f"   ❌ TFLite testing error: {e}")
    
    # Save metadata
    print(f"\n💾 Saving model metadata...")
    metadata = {
        'model_type': 'BiLSTM Speed Estimator',
        'input_shape': list(model.input_shape),
        'output_shape': list(model.output_shape),
        'total_parameters': int(total_params),
        'tflite_size_mb': round(tflite_size_mb, 2),
        'window_size': preprocessor_params['window_size'],
        'num_features': preprocessor_params['num_features'],
        'feature_names': preprocessor_params['input_features'],
        'normalization': {
            'mean': preprocessor_params['mean'],
            'std': preprocessor_params['std']
        },
        'usage': {
            'description': 'Speed estimation from IMU sensor data',
            'input': 'Window of IMU readings (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)',
            'output': 'Predicted speed in m/s',
            'preprocessing': 'Normalize using (value - mean) / std for each feature'
        }
    }
    
    metadata_path = output_dir / 'phase1_model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Metadata saved: {metadata_path}")
    
    # Print deployment instructions
    print("\n" + "=" * 70)
    print("✅ EXPORT COMPLETE!")
    print("=" * 70)
    print(f"\n📱 Files ready for Android deployment:")
    print(f"   1. {tflite_path.name} ({tflite_size_mb:.2f} MB)")
    print(f"   2. {preprocessor_path.name}")
    print(f"   3. {metadata_path.name}")
    
    print(f"\n📋 Deployment steps:")
    print(f"   1. Copy files to Android project:")
    print(f"      - Copy {tflite_path.name} to app/src/main/assets/")
    print(f"      - Copy {preprocessor_path.name} to app/src/main/assets/")
    
    print(f"\n   2. Add TensorFlow Lite dependencies to build.gradle:")
    print(f"      implementation 'org.tensorflow:tensorflow-lite:2.13.0'")
    print(f"      implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.13.0'")
    
    print(f"\n   3. Load model in Android:")
    print(f"      ```kotlin")
    print(f"      val model = Interpreter(loadModelFile('phase1_model.tflite'))")
    print(f"      val input = Array(1) {{ FloatArray({window_size} * {preprocessor_params['num_features']}) }}")
    print(f"      val output = Array(1) {{ FloatArray(1) }}")
    print(f"      model.run(input, output)")
    print(f"      val speed = output[0][0]  // Predicted speed in m/s")
    print(f"      ```")
    
    print(f"\n   4. See ANDROID_DEPLOYMENT_GUIDE.md for complete integration")
    
    print("\n🎉 Model is ready for mobile deployment!")


if __name__ == '__main__':
    export_to_tflite()
