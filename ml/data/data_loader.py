"""
Unified data loader for NavAI project supporting multiple datasets:
- NavAI logger CSV files
- IO-VNBD dataset
- OxIOD dataset  
- comma2k19 dataset
"""

import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Unified schema columns
SCHEMA_COLS = [
    'timestamp_ns', 'accel_x', 'accel_y', 'accel_z', 
    'gyro_x', 'gyro_y', 'gyro_z',
    'mag_x', 'mag_y', 'mag_z', 
    'qw', 'qx', 'qy', 'qz',
    'gps_lat', 'gps_lon', 'gps_speed_mps', 
    'device', 'source'
]

class DataLoader:
    """Unified data loader for all supported datasets"""
    
    def __init__(self, target_sample_rate: int = 100):
        self.target_sample_rate = target_sample_rate
        self.target_period_ns = int(1e9 / target_sample_rate)
        
    def empty_df(self) -> pd.DataFrame:
        """Create empty DataFrame with unified schema"""
        return pd.DataFrame(columns=SCHEMA_COLS)
    
    def load_navai_logs(self, log_dir: str) -> pd.DataFrame:
        """Load NavAI logger CSV files"""
        logger.info(f"Loading NavAI logs from {log_dir}")
        
        csv_files = glob.glob(os.path.join(log_dir, "*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {log_dir}")
            return self.empty_df()
        
        all_data = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Parse NavAI logger format: type,timestamp_ns,v1,v2,v3,v4,v5
                unified_data = self._parse_navai_format(df)
                if not unified_data.empty:
                    all_data.append(unified_data)
                    
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                continue
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"Loaded {len(result)} samples from {len(all_data)} files")
            return result
        else:
            return self.empty_df()
    
    def _parse_navai_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse NavAI logger CSV format into unified schema"""
        unified_df = self.empty_df()
        
        # Group by timestamp to merge sensor readings
        sensor_data = {}
        
        for _, row in df.iterrows():
            timestamp = row['timestamp_ns']
            sensor_type = row['type']
            
            if timestamp not in sensor_data:
                sensor_data[timestamp] = {
                    'timestamp_ns': timestamp,
                    'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0,
                    'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0,
                    'mag_x': 0.0, 'mag_y': 0.0, 'mag_z': 0.0,
                    'qw': 1.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0,
                    'gps_lat': 0.0, 'gps_lon': 0.0, 'gps_speed_mps': 0.0,
                    'device': 'navai_logger', 'source': 'navai'
                }
            
            if sensor_type == 'accel':
                sensor_data[timestamp].update({
                    'accel_x': row['v1'], 'accel_y': row['v2'], 'accel_z': row['v3']
                })
            elif sensor_type == 'gyro':
                sensor_data[timestamp].update({
                    'gyro_x': row['v1'], 'gyro_y': row['v2'], 'gyro_z': row['v3']
                })
            elif sensor_type == 'mag':
                sensor_data[timestamp].update({
                    'mag_x': row['v1'], 'mag_y': row['v2'], 'mag_z': row['v3']
                })
            elif sensor_type == 'rotvec':
                sensor_data[timestamp].update({
                    'qx': row['v1'], 'qy': row['v2'], 'qz': row['v3'], 'qw': row.get('v4', 1.0)
                })
            elif sensor_type == 'gps':
                sensor_data[timestamp].update({
                    'gps_lat': row['v1'], 'gps_lon': row['v2'], 'gps_speed_mps': row['v3']
                })
        
        # Convert to DataFrame
        if sensor_data:
            unified_df = pd.DataFrame(list(sensor_data.values()))
            unified_df = unified_df.sort_values('timestamp_ns').reset_index(drop=True)
        
        return unified_df
    
    def load_oxiod(self, data_dir: str) -> pd.DataFrame:
        """Load Oxford Inertial Odometry Dataset (OxIOD)"""
        logger.info(f"Loading OxIOD dataset from {data_dir}")
        
        # TODO: Implement OxIOD parser based on actual dataset format
        # This is a placeholder - actual implementation depends on OxIOD file structure
        
        df = self.empty_df()
        # Add OxIOD specific parsing logic here
        
        return df
    
    def load_iovnbd(self, data_dir: str) -> pd.DataFrame:
        """Load IO-VNBD (Inertial and Odometry Vehicle Navigation Benchmark Dataset)"""
        logger.info(f"Loading IO-VNBD dataset from {data_dir}")
        
        # TODO: Implement IO-VNBD parser based on actual dataset format
        # This is a placeholder - actual implementation depends on IO-VNBD file structure
        
        df = self.empty_df()
        # Add IO-VNBD specific parsing logic here
        
        return df
    
    def load_comma2k19(self, data_dir: str) -> pd.DataFrame:
        """Load comma2k19 dataset"""
        logger.info(f"Loading comma2k19 dataset from {data_dir}")
        
        # TODO: Implement comma2k19 parser based on actual dataset format
        # This is a placeholder - actual implementation depends on comma2k19 file structure
        
        df = self.empty_df()
        # Add comma2k19 specific parsing logic here
        
        return df
    
    def resample_to_target_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample data to target sample rate using interpolation"""
        if df.empty:
            return df
        
        logger.info(f"Resampling to {self.target_sample_rate} Hz")
        
        df = df.sort_values('timestamp_ns').reset_index(drop=True)
        
        # Create target timestamp grid
        t_start = df['timestamp_ns'].iloc[0]
        t_end = df['timestamp_ns'].iloc[-1]
        target_timestamps = np.arange(t_start, t_end, self.target_period_ns)
        
        # Interpolate numeric columns
        numeric_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z',
                       'mag_x', 'mag_y', 'mag_z', 'qw', 'qx', 'qy', 'qz',
                       'gps_lat', 'gps_lon', 'gps_speed_mps']
        
        resampled_data = {'timestamp_ns': target_timestamps}
        
        for col in numeric_cols:
            if col in df.columns:
                resampled_data[col] = np.interp(target_timestamps, df['timestamp_ns'], df[col])
            else:
                resampled_data[col] = np.zeros(len(target_timestamps))
        
        # Forward fill categorical columns
        resampled_data['device'] = df['device'].iloc[0] if 'device' in df.columns else 'unknown'
        resampled_data['source'] = df['source'].iloc[0] if 'source' in df.columns else 'unknown'
        
        resampled_df = pd.DataFrame(resampled_data)
        
        logger.info(f"Resampled from {len(df)} to {len(resampled_df)} samples")
        return resampled_df
    
    def load_combined_dataset(self, data_paths: Dict[str, str]) -> pd.DataFrame:
        """Load and combine multiple datasets"""
        logger.info("Loading combined dataset")
        
        all_datasets = []
        
        for dataset_name, path in data_paths.items():
            if not os.path.exists(path):
                logger.warning(f"Path not found: {path}")
                continue
                
            if dataset_name == 'navai':
                df = self.load_navai_logs(path)
            elif dataset_name == 'oxiod':
                df = self.load_oxiod(path)
            elif dataset_name == 'iovnbd':
                df = self.load_iovnbd(path)
            elif dataset_name == 'comma2k19':
                df = self.load_comma2k19(path)
            else:
                logger.warning(f"Unknown dataset type: {dataset_name}")
                continue
            
            if not df.empty:
                df = self.resample_to_target_rate(df)
                df['source'] = dataset_name
                all_datasets.append(df)
        
        if all_datasets:
            combined_df = pd.concat(all_datasets, ignore_index=True)
            logger.info(f"Combined dataset: {len(combined_df)} total samples from {len(all_datasets)} datasets")
            return combined_df
        else:
            logger.warning("No datasets loaded successfully")
            return self.empty_df()

    def strip_gps_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove GPS data for GPS-denied navigation testing"""
        logger.info("Stripping GPS data for GPS-denied mode")
        gps_df = df.copy()
        gps_cols = ['gps_lat', 'gps_lon', 'gps_speed_mps']
        for col in gps_cols:
            if col in gps_df.columns:
                gps_df[col] = np.nan
        logger.info("GPS data removed - operating in GPS-denied mode")
        return gps_df

# Example usage and CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NavAI Data Loader')
    parser.add_argument('--dataset', choices=['navai', 'oxiod', 'iovnbd', 'comma2k19', 'synthetic'], 
                       default='synthetic', help='Dataset to load')
    parser.add_argument('--path', type=str, help='Path to dataset')
    parser.add_argument('--output', type=str, help='Output file path (.npz)')
    parser.add_argument('--ignore-gps', action='store_true', 
                       help='Strip GPS data for GPS-denied navigation testing')
    parser.add_argument('--sample-rate', type=int, default=100, 
                       help='Target sample rate in Hz')
    
    args = parser.parse_args()
    
    loader = DataLoader(target_sample_rate=args.sample_rate)
    
    # Load dataset
    if args.dataset == 'synthetic':
        # Create synthetic data for testing
        logger.info("Generating synthetic dataset")
        n_samples = 1000
        timestamps = np.arange(n_samples) * (1e9 / args.sample_rate)
        
        df = pd.DataFrame({
            'timestamp_ns': timestamps,
            'accel_x': np.random.normal(0, 1, n_samples),
            'accel_y': np.random.normal(0, 1, n_samples), 
            'accel_z': np.random.normal(9.81, 1, n_samples),
            'gyro_x': np.random.normal(0, 0.1, n_samples),
            'gyro_y': np.random.normal(0, 0.1, n_samples),
            'gyro_z': np.random.normal(0, 0.1, n_samples),
            'mag_x': np.random.normal(25, 5, n_samples),
            'mag_y': np.random.normal(0, 5, n_samples),
            'mag_z': np.random.normal(-40, 5, n_samples),
            'qw': np.ones(n_samples),
            'qx': np.zeros(n_samples),
            'qy': np.zeros(n_samples), 
            'qz': np.zeros(n_samples),
            'gps_lat': np.linspace(40.7128, 40.7138, n_samples),
            'gps_lon': np.linspace(-74.0060, -74.0050, n_samples),
            'gps_speed_mps': np.abs(np.random.normal(5, 2, n_samples)),
            'device': 'synthetic',
            'source': 'synthetic'
        })
    else:
        if not args.path:
            logger.error(f"Path required for dataset {args.dataset}")
            exit(1)
        
        if args.dataset == 'navai':
            df = loader.load_navai_logs(args.path)
        elif args.dataset == 'oxiod':
            df = loader.load_oxiod(args.path)
        elif args.dataset == 'iovnbd':
            df = loader.load_iovnbd(args.path)
        elif args.dataset == 'comma2k19':
            df = loader.load_comma2k19(args.path)
    
    # Apply GPS-denied mode if requested
    if args.ignore_gps:
        df = loader.strip_gps_data(df)
    
    # Save output
    if args.output:
        logger.info(f"Saving data to {args.output}")
        np.savez_compressed(args.output, 
                           timestamps=df['timestamp_ns'].values,
                           accel=df[['accel_x', 'accel_y', 'accel_z']].values,
                           gyro=df[['gyro_x', 'gyro_y', 'gyro_z']].values,
                           mag=df[['mag_x', 'mag_y', 'mag_z']].values,
                           quaternion=df[['qw', 'qx', 'qy', 'qz']].values,
                           gps_coords=df[['gps_lat', 'gps_lon']].values if not args.ignore_gps else None,
                           gps_speed=df['gps_speed_mps'].values if not args.ignore_gps else None,
                           device=df['device'].iloc[0] if len(df) > 0 else 'unknown',
                           source=df['source'].iloc[0] if len(df) > 0 else 'unknown')
        logger.info(f"Data saved: {len(df)} samples, GPS-denied: {args.ignore_gps}")
    else:
        print(f"Loaded {len(df)} samples from {args.dataset}")
        print(f"GPS-denied mode: {args.ignore_gps}")
        print("Data loader ready for use")
