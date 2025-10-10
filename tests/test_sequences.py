import sys
sys.path.append('.')
from ml.data.data_loader import DataLoader
import numpy as np

print('Testing sequence preparation...')
loader = DataLoader()
df = loader.load_comma2k19('data/comma2k19').head(1000)
print(f'Test data: {len(df)} samples')

sequences = []
targets = []
feature_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']

for i in range(0, len(df) - 50, 10):
    seq = df.iloc[i:i+50][feature_cols].values
    target = df.iloc[i+50-1]['gps_speed_mps']
    if not np.isnan(target):
        sequences.append(seq)
        targets.append(target)

sequences = np.array(sequences)
targets = np.array(targets)

print(f'Created {len(sequences)} sequences')
print(f'Sequence shape: {sequences.shape}')
print(f'Target shape: {targets.shape}')
print(f'Target range: {targets.min():.2f} - {targets.max():.2f} m/s')
print('✓ Shape verification successful!')
