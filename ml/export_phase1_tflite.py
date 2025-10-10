"""
Export Phase 1 BiLSTM model to TensorFlow Lite for Android deployment
Optimized to use PyTorch GPU for distillation data generation
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm
import os

# Configure TensorFlow to use GPU if available
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Reduce warnings
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ TensorFlow will use GPU: {gpus}")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
else:
    print("⚠️ TensorFlow GPU not available, using CPU")

# Phase 1 BiLSTM model architecture (must match training)
class BaselineSpeedEstimator(nn.Module):
    """Simple BiLSTM baseline for speed estimation"""
    
    def __init__(self, input_dim=10, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        x = self.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        speed = self.fc3(x)
        
        return speed


def create_tf_lstm_model(input_shape=(100, 10), hidden_dim=128, num_layers=3):
    """Create TensorFlow LSTM model matching PyTorch architecture"""
    
    inputs = tf.keras.Input(shape=input_shape)
    
    # Bidirectional LSTM layers
    x = inputs
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                hidden_dim,
                return_sequences=return_sequences,
                dropout=0.2 if i < num_layers - 1 else 0
            )
        )(x)
    
    # Fully connected layers
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def generate_distillation_data(pytorch_model, num_samples=5000, window_size=100, num_features=10):
    """Generate training data for knowledge distillation from PyTorch to TensorFlow"""
    
    print(f"\n📊 Generating {num_samples} distillation samples...")
    
    X = []
    y = []
    
    pytorch_model.eval()
    device = next(pytorch_model.parameters()).device
    
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Generating samples"):
            # Generate realistic IMU-like data
            sample = np.random.randn(window_size, num_features).astype(np.float32)
            
            # Normalize to realistic IMU ranges
            # Accel: -20 to 20 m/s², Gyro: -5 to 5 rad/s
            sample[:, 0:3] *= 5  # Accelerometer
            sample[:, 3:6] *= 1  # Gyroscope
            sample[:, 6:8] *= 3  # Magnitude features
            sample[:, 8:10] *= 2  # Derivative features
            
            # Get PyTorch prediction
            pytorch_input = torch.FloatTensor(sample).unsqueeze(0).to(device)
            pytorch_output = pytorch_model(pytorch_input).cpu().numpy()
            
            X.append(sample)
            y.append(pytorch_output[0])
    
    return np.array(X), np.array(y)


def export_phase1_model():
    """Export Phase 1 model to TensorFlow Lite"""
    
    print("\n🚀 Phase 1 Model Export to TensorFlow Lite")
    print("=" * 60)
    
    # Paths
    model_path = Path('ml/outputs/phase1_best_model.pth')
    preprocessor_path = Path('ml/outputs/phase1_preprocessor.pkl')
    output_dir = Path('ml/outputs')
    
    # Check files exist
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    if not preprocessor_path.exists():
        print(f"❌ Preprocessor not found: {preprocessor_path}")
        return
    
    # Load PyTorch model
    print("\n📥 Loading PyTorch model...")
    
    # Use GPU if available for faster distillation data generation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    pytorch_model = BaselineSpeedEstimator(
        input_dim=10,
        hidden_dim=128,
        num_layers=3,
        dropout=0.2
    )
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    pytorch_model = pytorch_model.to(device)
    pytorch_model.eval()
    
    print(f"   ✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"   ✅ RMSE: {checkpoint['rmse']:.4f} m/s")
    print(f"   ✅ R²: {checkpoint['r2']:.4f}")
    
    # Load preprocessor
    print("\n📥 Loading preprocessor...")
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Extract normalization parameters
    if hasattr(preprocessor, 'scaler_'):
        mean = preprocessor.scaler_.mean_.tolist()
        std = preprocessor.scaler_.scale_.tolist()
        print(f"   ✅ Extracted normalization params (mean/std for {len(mean)} features)")
    else:
        print("   ⚠️ No scaler found, using default values")
        mean = [0.0] * 10
        std = [1.0] * 10
    
    # Save preprocessor params as JSON for Android
    preprocessor_json = {
        'mean': mean,
        'std': std,
        'window_size': 100,
        'num_features': 10,
        'input_features': [
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'accel_magnitude', 'gyro_magnitude',
            'accel_derivative', 'gyro_derivative'
        ]
    }
    
    preprocessor_json_path = output_dir / 'preprocessor_params.json'
    with open(preprocessor_json_path, 'w') as f:
        json.dump(preprocessor_json, f, indent=2)
    
    print(f"   ✅ Saved preprocessor params: {preprocessor_json_path}")
    
    # Create TensorFlow model
    print("\n🔧 Creating TensorFlow model...")
    tf_model = create_tf_lstm_model(input_shape=(100, 10), hidden_dim=128, num_layers=3)
    tf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("   ✅ TensorFlow model created")
    
    # Generate distillation data
    X_distill, y_distill = generate_distillation_data(pytorch_model, num_samples=5000)
    
    # Train TensorFlow model to match PyTorch (knowledge distillation)
    print("\n🎓 Training TensorFlow model (knowledge distillation)...")
    history = tf_model.fit(
        X_distill, y_distill,
        epochs=30,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    
    # Verify conversion accuracy
    print("\n✅ Verifying conversion accuracy...")
    test_samples = 100
    errors = []
    
    with torch.no_grad():
        for i in range(test_samples):
            sample = X_distill[i:i+1]
            
            # PyTorch prediction
            pytorch_input = torch.FloatTensor(sample).to(device)
            pytorch_output = pytorch_model(pytorch_input).cpu().numpy()[0][0]
            
            # TensorFlow prediction
            tf_output = tf_model.predict(sample, verbose=0)[0][0]
            
            error = abs(pytorch_output - tf_output)
            errors.append(error)
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"   Mean conversion error: {mean_error:.6f} m/s")
    print(f"   Max conversion error: {max_error:.6f} m/s")
    
    if mean_error > 0.5:
        print("   ⚠️ Warning: High conversion error! Model may not match PyTorch well.")
    else:
        print("   ✅ Conversion accuracy is good!")
    
    # Convert to TensorFlow Lite
    print("\n📦 Converting to TensorFlow Lite...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    
    # LSTM requires SELECT_TF_OPS (hybrid mode)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS     # Enable TF ops (needed for LSTM)
    ]
    
    # Disable experimental tensor list ops lowering (causes LSTM issues)
    converter._experimental_lower_tensor_list_ops = False
    
    # Enable optimizations for mobile
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set representative dataset for quantization
    def representative_dataset():
        for i in range(100):
            yield [X_distill[i:i+1].astype(np.float32)]
    
    converter.representative_dataset = representative_dataset
    
    # Convert
    tflite_model = converter.convert()
    
    # Save TFLite model
    tflite_path = output_dir / 'phase1_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    tflite_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"   ✅ TFLite model saved: {tflite_path}")
    print(f"   ✅ Model size: {tflite_size_mb:.2f} MB")
    
    # Test TFLite model
    print("\n🧪 Testing TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    
    # Test inference
    test_input = X_distill[0:1].astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    # Compare with TensorFlow
    tf_output = tf_model.predict(test_input, verbose=0)[0][0]
    tflite_error = abs(tf_output - tflite_output)
    
    print(f"   TensorFlow output: {tf_output:.4f} m/s")
    print(f"   TFLite output: {tflite_output:.4f} m/s")
    print(f"   TFLite conversion error: {tflite_error:.6f} m/s")
    
    if tflite_error < 0.1:
        print("   ✅ TFLite model working correctly!")
    else:
        print("   ⚠️ TFLite conversion may have issues")
    
    print("\n" + "=" * 60)
    print("✅ Export Complete!")
    print("=" * 60)
    print(f"\n📱 Ready for Android deployment:")
    print(f"   1. TFLite model: {tflite_path}")
    print(f"   2. Preprocessor params: {preprocessor_json_path}")
    print(f"\n📋 Next steps:")
    print(f"   1. Copy {tflite_path.name} to Android app assets/")
    print(f"   2. Copy {preprocessor_json_path.name} to Android app assets/")
    print(f"   3. Build Android app using ANDROID_DEPLOYMENT_GUIDE.md")
    print(f"   4. Test on device!")


if __name__ == '__main__':
    export_phase1_model()
