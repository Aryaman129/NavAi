import tensorflow as tf
import torch

print('=== TensorFlow ===')
tf_gpus = tf.config.list_physical_devices('GPU')
print(f'GPU Count: {len(tf_gpus)}')
for gpu in tf_gpus:
    print(f'  - {gpu}')

print('\n=== PyTorch ===')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Count: {torch.cuda.device_count()}')
