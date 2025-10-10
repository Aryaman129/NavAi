"""
DEBUGGING SCRIPT: Let's understand what's really happening with our data

This script will help us:
1. Examine speed distribution 
2. Check for data leakage
3. Verify evaluation metrics
4. Set realistic expectations
5. Test simpler approaches

We'll work through this together step by step.
"""

import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml.data.data_loader import DataLoader

def analyze_speed_data():
    """First, let's understand what we're predicting"""
    print("="*80)
    print("SPEED DATA ANALYSIS")
    print("="*80)
    
    # Load data
    loader = DataLoader()
    df = loader.load_comma2k19('data/comma2k19')
    
    speeds = df['gps_speed_mps'].dropna()
    
    print(f"Total samples: {len(df):,}")
    print(f"Valid speed samples: {len(speeds):,}")
    print(f"Speed range: {speeds.min():.2f} - {speeds.max():.2f} m/s")
    print(f"Speed range (km/h): {speeds.min()*3.6:.1f} - {speeds.max()*3.6:.1f} km/h")
    
    # Speed distribution
    print("\nSpeed distribution:")
    speed_bins = [0, 2, 5, 10, 15, 25, 50]
    for i in range(len(speed_bins)-1):
        low, high = speed_bins[i], speed_bins[i+1]
        count = ((speeds >= low) & (speeds < high)).sum()
        percent = count / len(speeds) * 100
        print(f"  {low:2d}-{high:2d} m/s: {count:8,} samples ({percent:5.1f}%)")
    
    # Plot distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(speeds, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Count')
    plt.title('Speed Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(speeds, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Count')
    plt.title('Speed Distribution (Log Scale)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    time_hours = (df['timestamp_ns'] - df['timestamp_ns'].min()) / 1e9 / 3600
    plt.plot(time_hours, speeds, alpha=0.3, linewidth=0.5)
    plt.xlabel('Time (hours)')
    plt.ylabel('Speed (m/s)')
    plt.title('Speed Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('speed_analysis.png', dpi=150)
    print(f"\n✓ Speed analysis saved to speed_analysis.png")
    
    return df

def test_data_leakage():
    """Check if our train/val/test splits have temporal overlap"""
    print("\n" + "="*80)
    print("DATA LEAKAGE ANALYSIS")
    print("="*80)
    
    # Create sequences like we do in training
    df = analyze_speed_data()
    
    print(f"\nCreating sequences with seq_len=20, stride=5...")
    sequences = []
    targets = []
    timestamps = []
    
    feature_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'qw', 'qx', 'qy', 'qz']
    
    for i in range(0, len(df) - 20, 5):
        seq = df.iloc[i:i+20][feature_cols].values
        target = df.iloc[i+20-1]['gps_speed_mps']
        start_time = df.iloc[i]['timestamp_ns']
        end_time = df.iloc[i+19]['timestamp_ns']
        
        if not np.isnan(target) and not np.isnan(seq).any():
            sequences.append(seq)
            targets.append(target)
            timestamps.append((start_time, end_time))
    
    print(f"Created {len(sequences):,} sequences")
    
    # Split like we do in training
    n_samples = len(sequences)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_times = timestamps[:train_size]
    val_times = timestamps[train_size:train_size+val_size]
    test_times = timestamps[train_size+val_size:]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_times):,} sequences")
    print(f"  Val:   {len(val_times):,} sequences") 
    print(f"  Test:  {len(test_times):,} sequences")
    
    # Check overlap
    last_train_end = train_times[-1][1]
    first_val_start = val_times[0][0]
    last_val_end = val_times[-1][1]
    first_test_start = test_times[0][0]
    
    print(f"\nTemporal boundaries:")
    print(f"  Last train ends:   {last_train_end}")
    print(f"  First val starts:  {first_val_start}")
    print(f"  Overlap: {max(0, last_train_end - first_val_start):,} ns")
    print(f"  Last val ends:     {last_val_end}")
    print(f"  First test starts: {first_test_start}")
    print(f"  Overlap: {max(0, last_val_end - first_test_start):,} ns")
    
    # Time gaps
    train_val_gap = (first_val_start - last_train_end) / 1e9
    val_test_gap = (first_test_start - last_val_end) / 1e9
    
    print(f"\nTime gaps:")
    print(f"  Train → Val gap: {train_val_gap:.2f} seconds")
    print(f"  Val → Test gap:  {val_test_gap:.2f} seconds")
    
    if train_val_gap < 0 or val_test_gap < 0:
        print("\n🚨 CRITICAL: TEMPORAL DATA LEAKAGE DETECTED!")
        print("   Sequences overlap between train/val/test splits")
        print("   This causes the model to 'cheat' by seeing similar data")
    else:
        print("\n✓ No temporal overlap detected")

def test_simple_baselines():
    """Test simple prediction methods to set expectations"""
    print("\n" + "="*80)  
    print("BASELINE COMPARISONS")
    print("="*80)
    
    df = analyze_speed_data()
    speeds = df['gps_speed_mps'].dropna()
    
    # Split speeds into train/test by time (proper temporal split)
    split_point = int(0.8 * len(speeds))
    train_speeds = speeds.iloc[:split_point]
    test_speeds = speeds.iloc[split_point:]
    
    print(f"\nTemporal split:")
    print(f"  Train: {len(train_speeds):,} samples")
    print(f"  Test:  {len(test_speeds):,} samples")
    
    # Baseline 1: Always predict mean
    mean_speed = train_speeds.mean()
    mean_predictions = np.full(len(test_speeds), mean_speed)
    mean_rmse = np.sqrt(np.mean((test_speeds - mean_predictions) ** 2))
    
    # Baseline 2: Always predict median
    median_speed = train_speeds.median()  
    median_predictions = np.full(len(test_speeds), median_speed)
    median_rmse = np.sqrt(np.mean((test_speeds - median_predictions) ** 2))
    
    # Baseline 3: Previous speed (simple temporal model)
    prev_predictions = test_speeds.shift(1).fillna(mean_speed)
    prev_rmse = np.sqrt(np.mean((test_speeds - prev_predictions) ** 2))
    
    print(f"\nBaseline performance:")
    print(f"  Always predict mean ({mean_speed:.2f} m/s):   RMSE = {mean_rmse:.3f} m/s")
    print(f"  Always predict median ({median_speed:.2f} m/s): RMSE = {median_rmse:.3f} m/s")
    print(f"  Predict previous speed:                      RMSE = {prev_rmse:.3f} m/s")
    
    print(f"\nFor comparison, our model achieved:")
    print(f"  Validation RMSE: ~15-18 m/s")
    print(f"  Test RMSE: 8.139 m/s (possibly due to data leakage)")
    
    if mean_rmse < 15:
        print(f"\n✓ Our model is learning (better than mean prediction)")
    else:
        print(f"\n❌ Our model is worse than just predicting the mean!")

def analyze_imu_features():
    """Look at the actual IMU data to understand what we're working with"""
    print("\n" + "="*80)
    print("IMU FEATURE ANALYSIS") 
    print("="*80)
    
    df = analyze_speed_data()
    
    # Sample 1000 points for visualization
    sample_df = df.sample(n=1000, random_state=42)
    
    imu_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    quat_cols = ['qw', 'qx', 'qy', 'qz']
    
    print(f"\nIMU statistics (on {len(sample_df)} samples):")
    for col in imu_cols:
        data = sample_df[col]
        print(f"  {col:8s}: mean={data.mean():7.3f}, std={data.std():6.3f}, range=[{data.min():7.3f}, {data.max():7.3f}]")
    
    print(f"\nQuaternion statistics:")
    for col in quat_cols:
        data = sample_df[col]
        print(f"  {col:8s}: mean={data.mean():7.3f}, std={data.std():6.3f}, range=[{data.min():7.3f}, {data.max():7.3f}]")
    
    # Check quaternion normalization
    quat_norms = np.sqrt(sample_df[quat_cols].pow(2).sum(axis=1))
    print(f"\nQuaternion norm statistics:")
    print(f"  Mean norm: {quat_norms.mean():.6f} (should be 1.0)")
    print(f"  Std norm:  {quat_norms.std():.6f} (should be ~0)")
    
    if quat_norms.std() > 0.01:
        print(f"  ⚠️ Warning: Quaternions may not be properly normalized")
    
    # Correlation with speed
    speed_col = sample_df['gps_speed_mps']
    print(f"\nCorrelation with speed:")
    for col in imu_cols + quat_cols:
        corr = sample_df[col].corr(speed_col)
        print(f"  {col:8s}: {corr:7.3f}")

def main():
    """Run all diagnostic tests"""
    print("🔍 DEBUGGING OUR SPEED ESTIMATION MODEL")
    print("Let's understand what's really happening...")
    
    # Run all analyses
    analyze_speed_data()
    test_data_leakage()
    test_simple_baselines()
    analyze_imu_features()
    
    print("\n" + "="*80)
    print("SUMMARY & NEXT STEPS")
    print("="*80)
    print("""
Based on this analysis, here's what we should do:

1. FIX DATA LEAKAGE: Use time-based splits, not random splits
2. SET REALISTIC GOALS: Compare to baseline predictions  
3. ENGINEER FEATURES: Use physics to create better inputs
4. TRY SIMPLER TARGETS: Predict speed changes, not absolute speed
5. HANDLE DISTRIBUTION: Address speed imbalance (lots of low speeds)

The task is harder than the fake baseline suggested, but achievable!
    """)

if __name__ == '__main__':
    main()