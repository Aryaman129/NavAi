import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('='*70)
print('TensorFlow GPU Configuration Check')
print('='*70)

print(f'\nTensorFlow version: {tf.__version__}')

# 1. List physical devices
print('\n1. Physical Devices:')
gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')
print(f'   GPUs detected: {len(gpus)}')
for i, gpu in enumerate(gpus):
    print(f'     [{i}] {gpu}')
print(f'   CPUs detected: {len(cpus)}')

# 2. Logical devices
print('\n2. Logical Devices:')
logical_gpus = tf.config.list_logical_devices('GPU')
print(f'   Logical GPUs: {len(logical_gpus)}')
for i, gpu in enumerate(logical_gpus):
    print(f'     [{i}] {gpu}')

# 3. Check if GPU is being used
print('\n3. GPU Availability:')
print(f'   tf.test.is_built_with_cuda(): {tf.test.is_built_with_cuda()}')
print(f'   tf.test.is_gpu_available(): {tf.test.is_gpu_available(cuda_only=True)}')

# 4. Device placement test
print('\n4. Device Placement Test:')
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print(f'   Matrix multiply device: {c.device}')
    print(f'   Result on GPU: {"GPU" in c.device}')

# 5. Check XLA JIT compilation
print('\n5. XLA JIT:')
print(f'   XLA enabled: {tf.config.optimizer.get_jit() is not None}')

# 6. Mixed precision
print('\n6. Mixed Precision Policy:')
from tensorflow.keras import mixed_precision
print(f'   Current policy: {mixed_precision.global_policy()}')

# 7. Memory test
if gpus:
    print('\n7. GPU Memory Test:')
    try:
        # Try to allocate a tensor on GPU
        with tf.device('/GPU:0'):
            test_tensor = tf.random.normal([1000, 1000])
            result = tf.matmul(test_tensor, test_tensor)
        print(f'   Success: Allocated and computed on GPU')
        print(f'   Tensor device: {result.device}')
    except Exception as e:
        print(f'   Failed: GPU computation error: {e}')

# 8. Performance test - CPU vs GPU
print('\n8. Performance Test (1000x1000 matrix multiply):')
import time

# CPU test
with tf.device('/CPU:0'):
    cpu_tensor = tf.random.normal([1000, 1000])
    start = time.time()
    for _ in range(100):
        _ = tf.matmul(cpu_tensor, cpu_tensor)
    cpu_time = time.time() - start
    print(f'   CPU time: {cpu_time:.3f}s')

# GPU test
if gpus:
    with tf.device('/GPU:0'):
        gpu_tensor = tf.random.normal([1000, 1000])
        start = time.time()
        for _ in range(100):
            _ = tf.matmul(gpu_tensor, gpu_tensor)
        gpu_time = time.time() - start
        print(f'   GPU time: {gpu_time:.3f}s')
        print(f'   Speedup: {cpu_time/gpu_time:.2f}x')

print('\n' + '='*70)
