import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'TensorFlow GPU Count: {len(gpus)}')
for gpu in gpus:
    print(f'  - {gpu}')
