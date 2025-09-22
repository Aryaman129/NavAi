#!/usr/bin/env python3
"""
Export trained PyTorch models to TensorFlow Lite for mobile deployment
Optimized for RTX 4050 training and mobile inference
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.metrics import mean_squared_error

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.speed_estimator import SpeedCNN, SpeedLSTM, create_tensorflow_model, convert_to_tflite
from data.data_loader import DataLoader as NavAIDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelExporter:
    """Export PyTorch models to TensorFlow Lite"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_pytorch_model(self, model_path: str, model_type: str, input_channels: int = 6):
        """Load trained PyTorch model"""
        logger.info(f"Loading PyTorch model from {model_path}")
        
        if model_type == 'cnn':
            model = SpeedCNN(input_channels=input_channels, hidden_dim=64)
        elif model_type == 'lstm':
            model = SpeedLSTM(input_size=input_channels, hidden_size=64)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    
    def pytorch_to_tensorflow(self, pytorch_model, input_shape, model_type: str):
        """Convert PyTorch model to TensorFlow"""
        logger.info("Converting PyTorch to TensorFlow...")
        
        # Create equivalent TensorFlow model
        tf_model = create_tensorflow_model(input_shape, model_type)
        
        # Generate dummy data for tracing
        dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
        
        # Get PyTorch prediction for reference
        with torch.no_grad():
            pytorch_input = torch.FloatTensor(dummy_input).to(self.device)
            pytorch_output = pytorch_model(pytorch_input).cpu().numpy()
        
        # Compile TensorFlow model
        tf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train TensorFlow model to match PyTorch (knowledge distillation)
        logger.info("Training TensorFlow model to match PyTorch...")
        
        # Generate training data from PyTorch model
        X_distill, y_distill = self.generate_distillation_data(pytorch_model, input_shape, 1000)
        
        # Train TensorFlow model
        tf_model.fit(
            X_distill, y_distill,
            epochs=50,
            batch_size=32,
            verbose=0,
            validation_split=0.2
        )
        
        # Verify conversion accuracy
        tf_output = tf_model.predict(dummy_input, verbose=0)
        conversion_error = abs(pytorch_output[0][0] - tf_output[0][0])
        
        logger.info(f"Conversion error: {conversion_error:.6f}")
        
        if conversion_error > 0.1:
            logger.warning(f"High conversion error: {conversion_error}")
        
        return tf_model
    
    def generate_distillation_data(self, pytorch_model, input_shape, num_samples: int):
        """Generate training data for knowledge distillation"""
        logger.info(f"Generating {num_samples} distillation samples...")
        
        X = []
        y = []
        
        pytorch_model.eval()
        with torch.no_grad():
            for _ in range(num_samples):
                # Generate realistic IMU-like data
                sample = self.generate_realistic_imu_sample(input_shape)
                
                # Get PyTorch prediction
                pytorch_input = torch.FloatTensor(sample).unsqueeze(0).to(self.device)
                pytorch_output = pytorch_model(pytorch_input).cpu().numpy()
                
                X.append(sample)
                y.append(pytorch_output[0])
        
        return np.array(X), np.array(y)
    
    def generate_realistic_imu_sample(self, input_shape):
        """Generate realistic IMU data sample"""
        seq_len, features = input_shape
        
        # Simulate realistic IMU patterns
        t = np.linspace(0, 1.5, seq_len)  # 1.5 seconds
        
        # Simulate vehicle motion
        speed = 5 + 10 * np.sin(0.5 * t[0])  # Varying speed
        
        # Accelerometer (with gravity and motion)
        accel_x = 0.2 * np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(seq_len)
        accel_y = 0.3 * np.cos(2 * np.pi * 0.3 * t) + 0.1 * np.random.randn(seq_len)
        accel_z = -9.81 + 0.5 * np.sin(2 * np.pi * 0.2 * t) + 0.2 * np.random.randn(seq_len)
        
        # Gyroscope
        gyro_x = 0.1 * np.sin(2 * np.pi * 0.4 * t) + 0.05 * np.random.randn(seq_len)
        gyro_y = 0.1 * np.cos(2 * np.pi * 0.6 * t) + 0.05 * np.random.randn(seq_len)
        gyro_z = 0.2 * np.sin(2 * np.pi * 0.3 * t) + 0.05 * np.random.randn(seq_len)
        
        # Combine features
        sample = np.column_stack([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
        
        return sample.astype(np.float32)
    
    def export_to_tflite(self, tf_model, output_path: str, quantize: bool = True):
        """Export TensorFlow model to TensorFlow Lite"""
        logger.info(f"Exporting to TensorFlow Lite: {output_path}")
        
        # Generate representative dataset for quantization
        if quantize:
            representative_data = []
            for _ in range(100):
                sample = self.generate_realistic_imu_sample((150, 6))
                representative_data.append(sample)
            representative_data = np.array(representative_data)
        else:
            representative_data = None
        
        # Convert to TFLite
        tflite_model = convert_to_tflite(
            tf_model,
            quantize=quantize,
            representative_dataset=representative_data
        )
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"TFLite model saved: {output_path}")
        logger.info(f"Model size: {len(tflite_model) / 1024:.1f} KB")
        
        return tflite_model
    
    def validate_tflite_model(self, tflite_model_path: str, tf_model):
        """Validate TFLite model accuracy"""
        logger.info("Validating TFLite model...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Generate test data
        test_samples = []
        tf_predictions = []
        tflite_predictions = []
        
        for _ in range(100):
            sample = self.generate_realistic_imu_sample((150, 6))
            test_samples.append(sample)
            
            # TensorFlow prediction
            tf_pred = tf_model.predict(sample[np.newaxis, ...], verbose=0)
            tf_predictions.append(tf_pred[0][0])
            
            # TFLite prediction
            interpreter.set_tensor(input_details[0]['index'], sample[np.newaxis, ...])
            interpreter.invoke()
            tflite_pred = interpreter.get_tensor(output_details[0]['index'])
            tflite_predictions.append(tflite_pred[0][0])
        
        # Calculate accuracy metrics
        mse = mean_squared_error(tf_predictions, tflite_predictions)
        rmse = np.sqrt(mse)
        max_error = np.max(np.abs(np.array(tf_predictions) - np.array(tflite_predictions)))
        
        logger.info(f"TFLite validation results:")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  Max error: {max_error:.6f}")
        logger.info(f"  Mean TF prediction: {np.mean(tf_predictions):.3f}")
        logger.info(f"  Mean TFLite prediction: {np.mean(tflite_predictions):.3f}")
        
        if rmse < 0.1:
            logger.info("✅ TFLite model validation passed")
        else:
            logger.warning("⚠️ TFLite model validation failed - high error")
        
        return rmse < 0.1

def main():
    parser = argparse.ArgumentParser(description='Export PyTorch models to TensorFlow Lite')
    parser.add_argument('--pytorch-model', type=str, required=True, help='Path to PyTorch model (.pth)')
    parser.add_argument('--model-type', choices=['cnn', 'lstm'], default='cnn', help='Model type')
    parser.add_argument('--output-dir', type=Path, default=Path('models'), help='Output directory')
    parser.add_argument('--quantize', action='store_true', help='Apply quantization')
    parser.add_argument('--validate', action='store_true', help='Validate converted model')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize exporter
    exporter = ModelExporter(args)
    
    try:
        # Load PyTorch model
        pytorch_model = exporter.load_pytorch_model(
            args.pytorch_model, 
            args.model_type
        )
        
        # Convert to TensorFlow
        input_shape = (150, 6)  # 1.5 seconds at 100Hz, 6 IMU features
        tf_model = exporter.pytorch_to_tensorflow(
            pytorch_model, 
            input_shape, 
            args.model_type
        )
        
        # Export to TFLite
        tflite_output_path = args.output_dir / f'speed_estimator_{args.model_type}.tflite'
        tflite_model = exporter.export_to_tflite(
            tf_model,
            str(tflite_output_path),
            quantize=args.quantize
        )
        
        # Validate if requested
        if args.validate:
            validation_passed = exporter.validate_tflite_model(
                str(tflite_output_path),
                tf_model
            )
            
            if not validation_passed:
                logger.error("Model validation failed!")
                return 1
        
        # Copy to Android assets
        android_assets_path = Path('mobile/app/src/main/assets/speed_estimator.tflite')
        android_assets_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(tflite_output_path, 'rb') as src, open(android_assets_path, 'wb') as dst:
            dst.write(src.read())
        
        logger.info(f"✅ Model exported successfully!")
        logger.info(f"TFLite model: {tflite_output_path}")
        logger.info(f"Android assets: {android_assets_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
