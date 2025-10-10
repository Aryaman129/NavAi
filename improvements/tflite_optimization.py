"""
TensorFlow Lite Optimization for Mobile Navigation
Based on 2024-2025 research on efficient neural networks for smartphones

Key optimizations:
1. Model quantization (INT8, FP16)
2. Hardware acceleration (GPU, NNAPI)
3. Dynamic inference optimization
4. Memory-efficient architectures
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import time

class MobileNavigationOptimizer:
    """
    TensorFlow Lite optimization pipeline for mobile navigation models
    """
    
    def __init__(self):
        # Optimization configurations
        self.optimization_configs = {
            'ultra_light': {
                'quantization': 'dynamic',
                'target_spec': tf.lite.TargetSpec(
                    supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                ),
                'inference_type': tf.int8,
                'representative_dataset_size': 100
            },
            'balanced': {
                'quantization': 'float16',
                'target_spec': tf.lite.TargetSpec(
                    supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
                ),
                'inference_type': tf.float16,
                'representative_dataset_size': 500
            },
            'high_accuracy': {
                'quantization': 'none',
                'target_spec': tf.lite.TargetSpec(
                    supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
                ),
                'inference_type': tf.float32,
                'representative_dataset_size': 1000
            }
        }
    
    def optimize_model(self, 
                      model_path: str, 
                      optimization_level: str = 'balanced',
                      representative_data: Optional[np.ndarray] = None,
                      output_path: Optional[str] = None) -> Dict:
        """
        Optimize model for mobile deployment
        """
        if optimization_level not in self.optimization_configs:
            raise ValueError(f"Unknown optimization level: {optimization_level}")
        
        config = self.optimization_configs[optimization_level]
        
        # Load model
        if model_path.endswith('.h5') or model_path.endswith('.keras'):
            model = tf.keras.models.load_model(model_path)
        else:
            model = tf.saved_model.load(model_path)
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model) if hasattr(model, 'layers') else tf.lite.TFLiteConverter.from_saved_model(model_path)
        
        # Apply optimizations
        converter = self._configure_converter(converter, config, representative_data)
        
        # Convert
        try:
            tflite_model = converter.convert()
        except Exception as e:
            print(f"Conversion failed: {e}")
            # Fallback to less aggressive optimization
            if optimization_level != 'high_accuracy':
                print("Falling back to high_accuracy mode")
                return self.optimize_model(model_path, 'high_accuracy', representative_data, output_path)
            raise
        
        # Save optimized model
        if output_path is None:
            output_path = model_path.replace('.h5', f'_optimized_{optimization_level}.tflite')
            output_path = output_path.replace('.keras', f'_optimized_{optimization_level}.tflite')
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Benchmark the optimized model
        benchmark_results = self._benchmark_model(tflite_model, representative_data)
        
        return {
            'output_path': output_path,
            'model_size_mb': len(tflite_model) / (1024 * 1024),
            'optimization_level': optimization_level,
            'benchmark_results': benchmark_results
        }
    
    def _configure_converter(self, 
                           converter: tf.lite.TFLiteConverter, 
                           config: Dict, 
                           representative_data: Optional[np.ndarray]) -> tf.lite.TFLiteConverter:
        """Configure TFLite converter with optimization settings"""
        
        # Set target specification
        converter.target_spec = config['target_spec']
        
        # Configure optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Configure quantization
        if config['quantization'] == 'dynamic':
            # Dynamic range quantization (smallest size)
            pass  # Already set with DEFAULT optimization
            
        elif config['quantization'] == 'float16':
            # Float16 quantization (good balance)
            converter.target_spec.supported_types = [tf.float16]
            
        elif config['quantization'] == 'int8':
            # Full integer quantization (requires representative dataset)
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            if representative_data is not None:
                converter.representative_dataset = lambda: self._representative_data_gen(
                    representative_data, config['representative_dataset_size']
                )
        
        # Enable experimental features for better optimization
        converter.experimental_new_converter = True
        converter.allow_custom_ops = False
        
        return converter
    
    def _representative_data_gen(self, data: np.ndarray, num_samples: int):
        """Generate representative data for quantization"""
        if len(data) > num_samples:
            indices = np.random.choice(len(data), num_samples, replace=False)
            data = data[indices]
        
        for sample in data:
            yield [sample.astype(np.float32)]
    
    def _benchmark_model(self, tflite_model: bytes, test_data: Optional[np.ndarray] = None) -> Dict:
        """Benchmark TensorFlow Lite model performance"""
        
        # Create interpreter
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Generate test data if not provided
        if test_data is None:
            input_shape = input_details[0]['shape']
            test_data = np.random.randn(100, *input_shape[1:]).astype(np.float32)
        
        # Warm up
        for _ in range(10):
            sample = test_data[0:1]
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
        
        # Benchmark inference time
        inference_times = []
        for i in range(min(100, len(test_data))):
            sample = test_data[i:i+1]
            
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            end_time = time.time()
            
            inference_times.append(end_time - start_time)
        
        return {
            'mean_inference_time_ms': np.mean(inference_times) * 1000,
            'std_inference_time_ms': np.std(inference_times) * 1000,
            'max_inference_time_ms': np.max(inference_times) * 1000,
            'min_inference_time_ms': np.min(inference_times) * 1000,
            'fps': 1.0 / np.mean(inference_times),
            'input_shape': input_details[0]['shape'].tolist(),
            'output_shape': output_details[0]['shape'].tolist(),
            'input_dtype': str(input_details[0]['dtype']),
            'output_dtype': str(output_details[0]['dtype'])
        }


class HardwareAcceleratedInference:
    """
    Hardware-accelerated inference using GPU and NNAPI
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.interpreters = {}
        self._initialize_interpreters()
    
    def _initialize_interpreters(self):
        """Initialize interpreters for different hardware backends"""
        
        # CPU interpreter (fallback)
        self.interpreters['cpu'] = tf.lite.Interpreter(
            model_path=self.model_path,
            num_threads=4
        )
        
        # GPU interpreter (if available)
        try:
            self.interpreters['gpu'] = tf.lite.Interpreter(
                model_path=self.model_path,
                experimental_delegates=[tf.lite.experimental.load_delegate('libtensorflowlite_gpu_delegate.so')]
            )
        except Exception as e:
            print(f"GPU delegate not available: {e}")
        
        # NNAPI interpreter (Android)
        try:
            self.interpreters['nnapi'] = tf.lite.Interpreter(
                model_path=self.model_path,
                experimental_delegates=[tf.lite.experimental.load_delegate('libnnapi_delegate.so')]
            )
        except Exception as e:
            print(f"NNAPI delegate not available: {e}")
        
        # Allocate tensors for all interpreters
        for name, interpreter in self.interpreters.items():
            try:
                interpreter.allocate_tensors()
                print(f"Initialized {name} interpreter successfully")
            except Exception as e:
                print(f"Failed to initialize {name} interpreter: {e}")
                del self.interpreters[name]
    
    def benchmark_backends(self, test_data: np.ndarray, num_runs: int = 100) -> Dict:
        """Benchmark different hardware backends"""
        results = {}
        
        for backend, interpreter in self.interpreters.items():
            print(f"Benchmarking {backend} backend...")
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Warm up
            for _ in range(10):
                sample = test_data[0:1]
                interpreter.set_tensor(input_details[0]['index'], sample)
                interpreter.invoke()
            
            # Benchmark
            inference_times = []
            for i in range(min(num_runs, len(test_data))):
                sample = test_data[i:i+1]
                
                start_time = time.time()
                interpreter.set_tensor(input_details[0]['index'], sample)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
            
            results[backend] = {
                'mean_time_ms': np.mean(inference_times) * 1000,
                'std_time_ms': np.std(inference_times) * 1000,
                'fps': 1.0 / np.mean(inference_times)
            }
        
        # Find best backend
        best_backend = min(results.keys(), key=lambda k: results[k]['mean_time_ms'])
        results['recommended_backend'] = best_backend
        
        return results
    
    def predict(self, input_data: np.ndarray, backend: str = 'auto') -> np.ndarray:
        """Run inference with specified backend"""
        
        if backend == 'auto':
            backend = 'gpu' if 'gpu' in self.interpreters else 'cpu'
        
        if backend not in self.interpreters:
            print(f"Backend {backend} not available, falling back to CPU")
            backend = 'cpu'
        
        interpreter = self.interpreters[backend]
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Ensure input has correct shape
        if len(input_data.shape) == len(input_details[0]['shape']) - 1:
            input_data = np.expand_dims(input_data, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        return interpreter.get_tensor(output_details[0]['index'])


class AdaptiveInferenceEngine:
    """
    Adaptive inference engine that adjusts based on device performance
    """
    
    def __init__(self, model_configs: Dict[str, str]):
        """
        Initialize with different model configurations
        model_configs: {'ultra_light': 'path/to/ultra_light.tflite', 'balanced': 'path/to/balanced.tflite'}
        """
        self.model_configs = model_configs
        self.inference_engines = {}
        self.performance_history = []
        self.current_model = 'balanced'  # Default
        
        # Initialize inference engines
        for config_name, model_path in model_configs.items():
            if os.path.exists(model_path):
                self.inference_engines[config_name] = HardwareAcceleratedInference(model_path)
        
        # Adaptive parameters
        self.target_fps = 20  # Target 20 FPS
        self.performance_window = 50  # Number of samples for performance assessment
        self.adaptation_threshold = 5  # Number of poor performances before adaptation
        
    def predict(self, input_data: np.ndarray) -> Dict:
        """Run adaptive inference"""
        
        if self.current_model not in self.inference_engines:
            raise ValueError(f"Model {self.current_model} not available")
        
        start_time = time.time()
        
        # Run inference
        engine = self.inference_engines[self.current_model]
        output = engine.predict(input_data)
        
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time
        
        # Track performance
        self.performance_history.append({
            'model': self.current_model,
            'inference_time': inference_time,
            'fps': fps,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.performance_history) > self.performance_window * 2:
            self.performance_history = self.performance_history[-self.performance_window:]
        
        # Adapt model if needed
        self._adapt_model_if_needed()
        
        return {
            'output': output,
            'model_used': self.current_model,
            'inference_time': inference_time,
            'fps': fps
        }
    
    def _adapt_model_if_needed(self):
        """Adapt model based on performance history"""
        
        if len(self.performance_history) < self.performance_window:
            return
        
        recent_performance = self.performance_history[-self.performance_window:]
        avg_fps = np.mean([p['fps'] for p in recent_performance])
        
        # Count poor performances
        poor_performances = sum(1 for p in recent_performance[-self.adaptation_threshold:] 
                              if p['fps'] < self.target_fps)
        
        # Adapt model
        if poor_performances >= self.adaptation_threshold:
            # Performance is poor, try lighter model
            if self.current_model == 'high_accuracy' and 'balanced' in self.inference_engines:
                self.current_model = 'balanced'
                print("Adapted to balanced model due to performance issues")
            elif self.current_model == 'balanced' and 'ultra_light' in self.inference_engines:
                self.current_model = 'ultra_light'
                print("Adapted to ultra_light model due to performance issues")
                
        elif avg_fps > self.target_fps * 1.5:
            # Performance is good, try more accurate model
            if self.current_model == 'ultra_light' and 'balanced' in self.inference_engines:
                self.current_model = 'balanced'
                print("Adapted to balanced model due to good performance")
            elif self.current_model == 'balanced' and 'high_accuracy' in self.inference_engines:
                self.current_model = 'high_accuracy'
                print("Adapted to high_accuracy model due to good performance")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.performance_history:
            return {}
        
        fps_values = [p['fps'] for p in self.performance_history]
        inference_times = [p['inference_time'] for p in self.performance_history]
        
        return {
            'mean_fps': np.mean(fps_values),
            'std_fps': np.std(fps_values),
            'mean_inference_time_ms': np.mean(inference_times) * 1000,
            'current_model': self.current_model,
            'total_samples': len(self.performance_history)
        }


# Model training with mobile optimization in mind
class MobileOptimizedTraining:
    """
    Training pipeline optimized for mobile deployment
    """
    
    @staticmethod
    def create_mobile_optimized_model(input_shape: Tuple[int, ...], 
                                    output_dim: int,
                                    complexity: str = 'balanced') -> tf.keras.Model:
        """Create mobile-optimized model architecture"""
        
        complexity_configs = {
            'ultra_light': {'filters': [16, 32, 16], 'dense_units': [32, 16]},
            'balanced': {'filters': [32, 64, 32], 'dense_units': [64, 32]},
            'high_accuracy': {'filters': [64, 128, 64], 'dense_units': [128, 64]}
        }
        
        config = complexity_configs[complexity]
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            
            # Depthwise separable convolutions for efficiency
            tf.keras.layers.SeparableConv1D(config['filters'][0], 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            
            tf.keras.layers.SeparableConv1D(config['filters'][1], 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            
            tf.keras.layers.SeparableConv1D(config['filters'][2], 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
            
            # Dense layers
            tf.keras.layers.Dense(config['dense_units'][0], activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(config['dense_units'][1], activation='relu'),
            tf.keras.layers.Dropout(0.1),
            
            # Output layer
            tf.keras.layers.Dense(output_dim, activation='linear')
        ])
        
        return model
    
    @staticmethod
    def train_with_quantization_aware(model: tf.keras.Model, 
                                    train_data: Tuple[np.ndarray, np.ndarray],
                                    val_data: Tuple[np.ndarray, np.ndarray],
                                    epochs: int = 100) -> tf.keras.Model:
        """Train model with quantization-aware training"""
        
        import tensorflow_model_optimization as tfmot
        
        # Apply quantization-aware training
        quantize_model = tfmot.quantization.keras.quantize_model
        q_aware_model = quantize_model(model)
        
        # Compile
        q_aware_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train
        history = q_aware_model.fit(
            train_data[0], train_data[1],
            validation_data=val_data,
            epochs=epochs,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        
        return q_aware_model


# Example usage and comprehensive testing
if __name__ == "__main__":
    print("Mobile Navigation Optimization Pipeline")
    print("=====================================")
    
    # Create a sample model for testing
    print("1. Creating sample model...")
    input_shape = (100, 6)  # 100 timesteps, 6 IMU features
    model = MobileOptimizedTraining.create_mobile_optimized_model(
        input_shape, output_dim=3, complexity='balanced'
    )
    model.compile(optimizer='adam', loss='mse')
    
    # Save model
    model_path = 'sample_navigation_model.h5'
    model.save(model_path)
    print(f"Saved sample model to {model_path}")
    
    # Test optimization pipeline
    print("\n2. Testing optimization pipeline...")
    optimizer = MobileNavigationOptimizer()
    
    # Generate representative data
    representative_data = np.random.randn(1000, *input_shape).astype(np.float32)
    
    # Test different optimization levels
    for level in ['ultra_light', 'balanced', 'high_accuracy']:
        print(f"\nOptimizing with {level} level...")
        try:
            result = optimizer.optimize_model(
                model_path, 
                optimization_level=level, 
                representative_data=representative_data
            )
            
            print(f"  Output: {result['output_path']}")
            print(f"  Size: {result['model_size_mb']:.2f} MB")
            print(f"  Mean inference time: {result['benchmark_results']['mean_inference_time_ms']:.2f} ms")
            print(f"  FPS: {result['benchmark_results']['fps']:.1f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Clean up
    try:
        os.remove(model_path)
        print(f"\nCleaned up {model_path}")
    except:
        pass
    
    print("\nOptimization pipeline testing completed!")