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

# Example usage
if __name__ == "__main__":
    loader = DataLoader(target_sample_rate=100)
    
    # Example: Load NavAI logs
    # df = loader.load_navai_logs("path/to/navai/logs")
    
    # Example: Load combined datasets
    data_paths = {
        'navai': 'data/navai_logs/',
        'oxiod': 'data/oxiod/',
        'iovnbd': 'data/iovnbd/',
        'comma2k19': 'data/comma2k19/'
    }
    
    # combined_df = loader.load_combined_dataset(data_paths)
    print("Data loader ready for use")
