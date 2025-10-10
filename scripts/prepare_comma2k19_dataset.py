"""
Prepare comma2k19 dataset by merging IMU and CAN data
Creates a single parquet file with aligned timestamps
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def prepare_comma2k19_data():
    """Merge IMU and CAN data by timestamp"""
    
    print("="*80)
    print("Preparing comma2k19 Dataset")
    print("="*80)
    
    data_dir = Path('data/comma2k19/processed_real/parsed')
    
    # Load IMU data
    print(f"\n📁 Loading IMU data...")
    imu_df = pd.read_csv(data_dir / 'imu_data.csv')
    print(f"   Loaded {len(imu_df):,} IMU samples")
    print(f"   Columns: {list(imu_df.columns)}")
    print(f"   Time range: {imu_df['timestamp_ns'].min()} - {imu_df['timestamp_ns'].max()}")
    
    # Load CAN data (has speed)
    print(f"\n📁 Loading CAN data...")
    can_df = pd.read_csv(data_dir / 'can_data.csv')
    print(f"   Loaded {len(can_df):,} CAN samples")
    print(f"   Columns: {list(can_df.columns)}")
    print(f"   Speed range: {can_df['car_speed_mps'].min():.2f} - {can_df['car_speed_mps'].max():.2f} m/s")
    
    # Merge by timestamp (nearest neighbor)
    print(f"\n🔗 Merging IMU and CAN data...")
    print(f"   Using merge_asof for nearest timestamp matching...")
    
    # Sort by timestamp
    imu_df = imu_df.sort_values('timestamp_ns')
    can_df = can_df.sort_values('timestamp_ns')
    
    # Merge using pandas merge_asof (forward fill within tolerance)
    merged_df = pd.merge_asof(
        imu_df,
        can_df[['timestamp_ns', 'car_speed_mps']],
        on='timestamp_ns',
        direction='nearest',
        tolerance=100_000_000  # 100ms tolerance
    )
    
    # Drop rows where speed couldn't be matched
    initial_len = len(merged_df)
    merged_df = merged_df.dropna(subset=['car_speed_mps'])
    print(f"   Matched {len(merged_df):,} / {initial_len:,} samples ({100*len(merged_df)/initial_len:.1f}%)")
    
    # Rename columns for consistency
    merged_df = merged_df.rename(columns={'car_speed_mps': 'speed'})
    
    # Convert timestamp to seconds (relative to first sample)
    merged_df['timestamp_sec'] = (merged_df['timestamp_ns'] - merged_df['timestamp_ns'].iloc[0]) / 1e9
    
    # Keep only relevant columns
    final_df = merged_df[[
        'timestamp_sec',
        'accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'speed'
    ]].copy()
    
    # Basic statistics
    print(f"\n📊 Final Dataset Statistics:")
    print(f"   Total samples: {len(final_df):,}")
    print(f"   Duration: {final_df['timestamp_sec'].max():.1f} seconds ({final_df['timestamp_sec'].max()/60:.1f} minutes)")
    print(f"   Sampling rate: {len(final_df) / final_df['timestamp_sec'].max():.1f} Hz")
    print(f"\n   IMU Statistics:")
    print(f"     Accel X: {final_df['accel_x'].mean():.3f} ± {final_df['accel_x'].std():.3f} m/s²")
    print(f"     Accel Y: {final_df['accel_y'].mean():.3f} ± {final_df['accel_y'].std():.3f} m/s²")
    print(f"     Accel Z: {final_df['accel_z'].mean():.3f} ± {final_df['accel_z'].std():.3f} m/s² (should be ~-9.81)")
    print(f"     Gyro X: {final_df['gyro_x'].mean():.3f} ± {final_df['gyro_x'].std():.3f} rad/s")
    print(f"     Gyro Y: {final_df['gyro_y'].mean():.3f} ± {final_df['gyro_y'].std():.3f} rad/s")
    print(f"     Gyro Z: {final_df['gyro_z'].mean():.3f} ± {final_df['gyro_z'].std():.3f} rad/s")
    print(f"\n   Speed Statistics:")
    print(f"     Mean: {final_df['speed'].mean():.2f} m/s ({final_df['speed'].mean()*3.6:.1f} km/h)")
    print(f"     Std: {final_df['speed'].std():.2f} m/s")
    print(f"     Min: {final_df['speed'].min():.2f} m/s")
    print(f"     Max: {final_df['speed'].max():.2f} m/s ({final_df['speed'].max()*3.6:.1f} km/h)")
    print(f"     Median: {final_df['speed'].median():.2f} m/s")
    
    # Save to parquet
    output_path = Path('data/comma2k19/processed_real/comma2k19_processed.parquet')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to {output_path}...")
    final_df.to_parquet(output_path, index=False, compression='snappy')
    
    # Check file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Compression: snappy")
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"   Output: {output_path}")
    print(f"   Samples: {len(final_df):,}")
    print(f"   Features: {list(final_df.columns)}")
    
    return final_df

if __name__ == '__main__':
    df = prepare_comma2k19_data()
