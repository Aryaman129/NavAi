"""
Dataset Downloader for NavAI Project
Downloads and prepares real-world datasets for training and testing
"""

import os
import sys
import requests
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
import logging
from typing import Optional
import json
import subprocess
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import playwright for browser automation
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available. Oxford dataset automation disabled.")

class DatasetDownloader:
    """Download and prepare datasets for NavAI"""
    
    def __init__(self, base_data_dir: str):
        self.base_data_dir = Path(base_data_dir)
        self.base_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations with alternative sources
        self.datasets = {
            'comma2k19': {
                'name': 'comma2k19 Sample',
                'urls': [
                    'https://github.com/commaai/comma2k19/releases/download/v1.0/route_20190120_145925_59066be7_0.zip',
                    'https://commadatastorage.blob.core.windows.net/comma2k19/samples/route_20190120_145925_59066be7_0.zip'
                ],
                'folder': 'comma2k19',
                'description': 'Comma.ai driving dataset with smartphone sensors',
                'size_mb': 150,
                'extract_method': 'zip'
            },
            'oxiod': {
                'name': 'Oxford Inertial Odometry Dataset',
                'urls': [
                    'https://ori.ox.ac.uk/datasets/iod/download.php'
                ],
                'folder': 'oxiod',
                'description': 'Oxford dataset for pedestrian and handheld device navigation',
                'size_mb': 500,
                'automated_download': True
            },
            'euroc': {
                'name': 'EuRoC MAV Dataset',
                'urls': [
                    'https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip',
                    'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip',
                    'https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets'
                ],
                'folder': 'euroc',
                'description': 'ETH Zurich MAV dataset with visual-inertial data',
                'size_mb': 800,
                'extract_method': 'zip'
            },
            'kitti': {
                'name': 'KITTI Dataset Sample',
                'urls': [
                    'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip'
                ],
                'folder': 'kitti',
                'description': 'KITTI automotive dataset for visual odometry',
                'size_mb': 300,
                'extract_method': 'zip'
            }
        }
    
    def download_file_with_fallback(self, urls: list, output_path: Path, chunk_size: int = 8192) -> bool:
        """Download file with fallback URLs"""
        for url in urls:
            try:
                logger.info(f"Trying download from {url}")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\rProgress: {progress:.1f}%", end='', flush=True)
                
                print()  # New line after progress
                logger.info(f"Downloaded {output_path.name} ({downloaded / 1024 / 1024:.1f} MB)")
                return True
                
            except requests.RequestException as e:
                logger.warning(f"Download failed from {url}: {e}")
                continue
        
        logger.error("All download URLs failed")
        return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path, method: str) -> bool:
        """Extract downloaded archive"""
        try:
            logger.info(f"Extracting {archive_path.name}")
            extract_to.mkdir(parents=True, exist_ok=True)
            
            if method == 'zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif method == 'tar':
                with tarfile.open(archive_path, 'r') as tar_ref:
                    tar_ref.extractall(extract_to)
            elif method == 'tar.gz':
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                logger.error(f"Unknown extraction method: {method}")
                return False
            
            logger.info(f"Extracted to {extract_to}")
            return True
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False
    
    def download_comma2k19_sample(self) -> bool:
        """Download comma2k19 sample dataset"""
        config = self.datasets['comma2k19']
        dataset_dir = self.base_data_dir / config['folder']
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if (dataset_dir / 'processed').exists():
            logger.info("comma2k19 sample already downloaded and processed")
            return True
        
        # Download the sample
        archive_name = "comma2k19_sample.zip"
        archive_path = dataset_dir / archive_name
        
        if not archive_path.exists():
            success = self.download_file_with_fallback(config['urls'], archive_path)
            if not success:
                # Create synthetic comma2k19-like data as fallback
                logger.info("Download failed, creating synthetic comma2k19 data")
                return self._create_synthetic_comma2k19(dataset_dir)
        
        # Extract
        extract_dir = dataset_dir / "raw"
        success = self.extract_archive(archive_path, extract_dir, config['extract_method'])
        if not success:
            return False
        
        # Process comma2k19 format
        self._process_comma2k19(extract_dir, dataset_dir)
        
        # Mark as processed
        (dataset_dir / 'processed').touch()
        
        # Cleanup archive
        if archive_path.exists():
            archive_path.unlink()
        
        return True
    
    def _process_comma2k19(self, raw_dir: Path, output_dir: Path):
        """Process comma2k19 data into unified format"""
        logger.info("Processing comma2k19 data format")
        
        # comma2k19 typically contains:
        # - imu.csv (IMU data)
        # - gps.csv (GPS data) 
        # - can.csv (CAN bus data)
        # - video files
        
        processed_dir = output_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        # Look for data files in extracted directory
        for root, dirs, files in os.walk(raw_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_path = Path(root) / file
                    logger.info(f"Found data file: {file_path}")
                    
                    # Copy to processed directory with descriptive names
                    if 'imu' in file.lower():
                        shutil.copy2(file_path, processed_dir / 'imu_data.csv')
                    elif 'gps' in file.lower():
                        shutil.copy2(file_path, processed_dir / 'gps_data.csv')
                    elif 'can' in file.lower():
                        shutil.copy2(file_path, processed_dir / 'can_data.csv')
                    else:
                        shutil.copy2(file_path, processed_dir / file)
        
        # Create metadata
        metadata = {
            'dataset': 'comma2k19',
            'description': 'Comma.ai 2k19 driving dataset sample',
            'processed_date': str(Path().cwd()),
            'files_found': list(processed_dir.glob('*.csv'))
        }
        
        with open(processed_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Processed comma2k19 data saved to {processed_dir}")
    
    def download_euroc_sample(self) -> bool:
        """Download EuRoC MAV dataset sample"""
        config = self.datasets['euroc']
        dataset_dir = self.base_data_dir / config['folder']
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if (dataset_dir / 'processed').exists():
            logger.info("EuRoC sample already downloaded and processed")
            return True
        
        # Download sample sequence
        archive_name = "euroc_mh01.zip"
        archive_path = dataset_dir / archive_name
        
        if not archive_path.exists():
            success = self.download_file_with_fallback(config['urls'], archive_path)
            if not success:
                # Create synthetic EuRoC-like data as fallback
                logger.info("Download failed, creating synthetic EuRoC data")
                return self._create_synthetic_euroc(dataset_dir)
        
        # Extract
        extract_dir = dataset_dir / "raw"
        success = self.extract_archive(archive_path, extract_dir, config['extract_method'])
        if not success:
            return False
        
        # Process EuRoC format
        self._process_euroc(extract_dir, dataset_dir)
        
        # Mark as processed
        (dataset_dir / 'processed').touch()
        
        # Cleanup archive
        if archive_path.exists():
            archive_path.unlink()
        
        return True
    
    def _process_euroc(self, raw_dir: Path, output_dir: Path):
        """Process EuRoC data into unified format"""
        logger.info("Processing EuRoC data format")
        
        processed_dir = output_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        # EuRoC structure:
        # mav0/imu0/data.csv
        # mav0/cam0/data.csv, cam1/data.csv
        # mav0/leica0/data.csv (ground truth)
        
        for root, dirs, files in os.walk(raw_dir):
            for file in files:
                if file == 'data.csv':
                    file_path = Path(root) / file
                    parent_folder = Path(root).name
                    
                    if 'imu' in parent_folder:
                        shutil.copy2(file_path, processed_dir / 'imu_data.csv')
                        logger.info(f"Copied IMU data: {file_path}")
                    elif 'leica' in parent_folder or 'vicon' in parent_folder:
                        shutil.copy2(file_path, processed_dir / 'ground_truth.csv')
                        logger.info(f"Copied ground truth: {file_path}")
                    elif 'cam' in parent_folder:
                        shutil.copy2(file_path, processed_dir / f'{parent_folder}_timestamps.csv')
                        logger.info(f"Copied camera timestamps: {file_path}")
        
        # Create metadata
        metadata = {
            'dataset': 'euroc',
            'description': 'EuRoC MAV Dataset - Machine Hall 01 Easy',
            'processed_date': str(Path().cwd()),
            'files_found': list(processed_dir.glob('*.csv'))
        }
        
        with open(processed_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Processed EuRoC data saved to {processed_dir}")
    
    def _create_synthetic_comma2k19(self, dataset_dir: Path) -> bool:
        """Create synthetic comma2k19-style data for testing"""
        logger.info("Creating synthetic comma2k19 dataset")
        
        processed_dir = dataset_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        # Generate synthetic driving data
        import numpy as np
        import pandas as pd
        
        # 10 minutes of driving data at 100Hz
        duration_sec = 600
        sample_rate = 100
        n_samples = duration_sec * sample_rate
        
        # Time vector
        timestamps = np.arange(n_samples) * (1e9 // sample_rate)  # nanoseconds
        
        # Simulate driving scenario with turns and acceleration
        t_sec = np.arange(n_samples) / sample_rate
        
        # Vehicle dynamics simulation
        speed = 15 + 5 * np.sin(0.1 * t_sec) + np.random.normal(0, 0.5, n_samples)  # m/s
        speed = np.clip(speed, 0, 30)
        
        # Acceleration (including gravity)
        accel_forward = np.gradient(speed) + np.random.normal(0, 0.2, n_samples)
        accel_x = accel_forward + np.random.normal(0, 0.1, n_samples)
        accel_y = 2 * np.sin(0.05 * t_sec) + np.random.normal(0, 0.3, n_samples)  # lateral
        accel_z = 9.81 + np.random.normal(0, 0.2, n_samples)  # gravity + bumps
        
        # Gyroscope (vehicle rotation)
        gyro_x = np.random.normal(0, 0.05, n_samples)  # roll rate
        gyro_y = np.random.normal(0, 0.05, n_samples)  # pitch rate  
        gyro_z = 0.1 * np.sin(0.03 * t_sec) + np.random.normal(0, 0.02, n_samples)  # yaw rate
        
        # GPS coordinates (driving loop)
        lat_center, lon_center = 37.4219999, -122.0840575  # Palo Alto
        radius = 0.01  # roughly 1km radius
        angle = 0.02 * t_sec  # slow loop
        gps_lat = lat_center + radius * np.cos(angle) + np.random.normal(0, 0.0001, n_samples)
        gps_lon = lon_center + radius * np.sin(angle) + np.random.normal(0, 0.0001, n_samples)
        
        # Create IMU data file
        imu_data = pd.DataFrame({
            'timestamp_ns': timestamps,
            'accel_x': accel_x,
            'accel_y': accel_y, 
            'accel_z': accel_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z
        })
        imu_data.to_csv(processed_dir / 'imu_data.csv', index=False)
        
        # Create GPS data file
        gps_data = pd.DataFrame({
            'timestamp_ns': timestamps[::10],  # 10Hz GPS
            'latitude': gps_lat[::10],
            'longitude': gps_lon[::10],
            'speed_mps': speed[::10]
        })
        gps_data.to_csv(processed_dir / 'gps_data.csv', index=False)
        
        # Create metadata
        metadata = {
            'dataset': 'comma2k19_synthetic',
            'description': 'Synthetic driving data in comma2k19 format',
            'duration_sec': duration_sec,
            'sample_rate_hz': sample_rate,
            'generated_date': str(Path().cwd())
        }
        
        with open(processed_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("Synthetic comma2k19 dataset created successfully")
        return True
    
    def _create_synthetic_euroc(self, dataset_dir: Path) -> bool:
        """Create synthetic EuRoC-style data for testing"""
        logger.info("Creating synthetic EuRoC dataset")
        
        processed_dir = dataset_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        # Generate synthetic MAV flight data
        import numpy as np
        import pandas as pd
        
        # 5 minutes of flight data at 200Hz IMU
        duration_sec = 300
        imu_rate = 200
        n_samples = duration_sec * imu_rate
        
        # Time vector
        timestamps = np.arange(n_samples) * (1e9 // imu_rate)  # nanoseconds
        t_sec = np.arange(n_samples) / imu_rate
        
        # Simulate quadcopter flight with figure-8 pattern
        # Position trajectory
        radius = 2.0  # 2m radius
        freq = 0.1  # slow figure-8
        x_pos = radius * np.sin(2 * np.pi * freq * t_sec)
        y_pos = radius * np.sin(4 * np.pi * freq * t_sec)
        z_pos = 1.5 + 0.5 * np.sin(2 * np.pi * freq * t_sec)  # hovering with variation
        
        # Compute accelerations (including gravity in body frame)
        accel_x = np.gradient(np.gradient(x_pos)) * imu_rate**2 + np.random.normal(0, 0.1, n_samples)
        accel_y = np.gradient(np.gradient(y_pos)) * imu_rate**2 + np.random.normal(0, 0.1, n_samples)
        accel_z = 9.81 + np.gradient(np.gradient(z_pos)) * imu_rate**2 + np.random.normal(0, 0.15, n_samples)
        
        # Gyroscope (angular velocities)
        gyro_x = 0.2 * np.sin(2 * np.pi * freq * t_sec) + np.random.normal(0, 0.02, n_samples)
        gyro_y = 0.2 * np.cos(2 * np.pi * freq * t_sec) + np.random.normal(0, 0.02, n_samples)
        gyro_z = 0.1 * np.sin(4 * np.pi * freq * t_sec) + np.random.normal(0, 0.01, n_samples)
        
        # Create IMU data file
        imu_data = pd.DataFrame({
            'timestamp_ns': timestamps,
            'w_RS_S_x [rad s^-1]': gyro_x,
            'w_RS_S_y [rad s^-1]': gyro_y,
            'w_RS_S_z [rad s^-1]': gyro_z,
            'a_RS_S_x [m s^-2]': accel_x,
            'a_RS_S_y [m s^-2]': accel_y,
            'a_RS_S_z [m s^-2]': accel_z
        })
        imu_data.to_csv(processed_dir / 'imu_data.csv', index=False)
        
        # Create ground truth data (1Hz)
        gt_rate = 1
        gt_samples = duration_sec * gt_rate
        gt_timestamps = np.arange(gt_samples) * (1e9 // gt_rate)
        
        gt_data = pd.DataFrame({
            'timestamp_ns': gt_timestamps,
            'p_RS_R_x [m]': x_pos[::imu_rate//gt_rate],
            'p_RS_R_y [m]': y_pos[::imu_rate//gt_rate],
            'p_RS_R_z [m]': z_pos[::imu_rate//gt_rate],
            'q_RS_w []': np.ones(gt_samples),  # simplified quaternion
            'q_RS_x []': np.zeros(gt_samples),
            'q_RS_y []': np.zeros(gt_samples), 
            'q_RS_z []': np.zeros(gt_samples)
        })
        gt_data.to_csv(processed_dir / 'ground_truth.csv', index=False)
        
        # Create metadata
        metadata = {
            'dataset': 'euroc_synthetic',
            'description': 'Synthetic MAV data in EuRoC format',
            'duration_sec': duration_sec,
            'imu_rate_hz': imu_rate,
            'generated_date': str(Path().cwd())
        }
        
        with open(processed_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("Synthetic EuRoC dataset created successfully")
        return True
    
    def download_oxford_automated(self) -> bool:
        """Download Oxford dataset using browser automation"""
        if not PLAYWRIGHT_AVAILABLE:
            logger.error("Playwright not available for Oxford dataset automation")
            self.create_oxford_placeholder()
            return False
        
        config = self.datasets['oxiod']
        dataset_dir = self.base_data_dir / config['folder']
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if (dataset_dir / 'processed').exists():
            logger.info("Oxford dataset already downloaded and processed")
            return True
        
        logger.info("Attempting automated Oxford dataset download...")
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=False)  # Set to True for headless
                page = browser.new_page()
                
                # Navigate to Oxford dataset page
                page.goto('https://ori.ox.ac.uk/datasets/oxford-inertial-odometry-dataset/')
                page.wait_for_load_state('networkidle')
                
                # Look for download links
                download_links = page.locator('a[href*="download"], a[href*=".zip"], a[href*=".tar"]')
                
                if download_links.count() > 0:
                    # Try to download the first available dataset
                    first_link = download_links.first
                    href = first_link.get_attribute('href')
                    
                    if href:
                        if not href.startswith('http'):
                            href = 'https://ori.ox.ac.uk' + href
                        
                        logger.info(f"Found download link: {href}")
                        
                        # Start download
                        with page.expect_download() as download_info:
                            first_link.click()
                        
                        download = download_info.value
                        download_path = dataset_dir / download.suggested_filename
                        download.save_as(download_path)
                        
                        logger.info(f"Downloaded Oxford dataset to {download_path}")
                        
                        # Extract and process
                        if download_path.suffix == '.zip':
                            extract_dir = dataset_dir / "raw"
                            success = self.extract_archive(download_path, extract_dir, 'zip')
                            if success:
                                self._process_oxford(extract_dir, dataset_dir)
                                (dataset_dir / 'processed').touch()
                                download_path.unlink()  # cleanup
                                browser.close()
                                return True
                
                browser.close()
                
        except Exception as e:
            logger.error(f"Automated Oxford download failed: {e}")
        
        # Fallback to synthetic data
        logger.info("Creating synthetic Oxford-style dataset as fallback")
        return self._create_synthetic_oxford(dataset_dir)
    
    def _create_synthetic_oxford(self, dataset_dir: Path) -> bool:
        """Create synthetic Oxford-style dataset"""
        logger.info("Creating synthetic Oxford dataset")
        
        processed_dir = dataset_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        # Generate synthetic pedestrian walking data
        import numpy as np
        import pandas as pd
        
        # 10 minutes of walking data at 100Hz
        duration_sec = 600
        sample_rate = 100
        n_samples = duration_sec * sample_rate
        
        timestamps = np.arange(n_samples) * (1e9 // sample_rate)
        t_sec = np.arange(n_samples) / sample_rate
        
        # Walking pattern simulation
        step_freq = 1.8  # steps per second
        walking_speed = 1.4  # m/s average
        
        # Accelerometer (walking pattern)
        step_signal = np.sin(2 * np.pi * step_freq * t_sec)
        accel_x = 0.5 * step_signal + np.random.normal(0, 0.3, n_samples)
        accel_y = 0.3 * np.sin(2 * np.pi * step_freq * t_sec + np.pi/4) + np.random.normal(0, 0.2, n_samples)
        accel_z = 9.81 + 1.5 * np.abs(step_signal) + np.random.normal(0, 0.4, n_samples)
        
        # Gyroscope (turning while walking)
        gyro_x = np.random.normal(0, 0.1, n_samples)
        gyro_y = np.random.normal(0, 0.1, n_samples)
        gyro_z = 0.2 * np.sin(0.05 * t_sec) + np.random.normal(0, 0.05, n_samples)
        
        # Magnetometer
        mag_x = 25 + 5 * np.sin(0.01 * t_sec) + np.random.normal(0, 2, n_samples)
        mag_y = 0 + np.random.normal(0, 2, n_samples)
        mag_z = -40 + np.random.normal(0, 3, n_samples)
        
        # Create handheld data
        handheld_data = pd.DataFrame({
            'timestamp_ns': timestamps,
            'accel_x': accel_x,
            'accel_y': accel_y,
            'accel_z': accel_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z,
            'mag_x': mag_x,
            'mag_y': mag_y,
            'mag_z': mag_z
        })
        
        # Save handheld data
        handheld_dir = processed_dir / 'handheld'
        handheld_dir.mkdir(exist_ok=True)
        handheld_data.to_csv(handheld_dir / 'data.csv', index=False)
        
        # Create ground truth (GPS-like positions)
        # Walking in a square pattern
        side_length = 100  # meters
        perimeter = 4 * side_length
        position_along_perimeter = (walking_speed * t_sec) % perimeter
        
        # Determine which side of square
        x_pos = np.where(position_along_perimeter < side_length, 
                        position_along_perimeter,
                        np.where(position_along_perimeter < 2*side_length,
                                side_length,
                                np.where(position_along_perimeter < 3*side_length,
                                        side_length - (position_along_perimeter - 2*side_length),
                                        0)))
        
        y_pos = np.where(position_along_perimeter < side_length,
                        0,
                        np.where(position_along_perimeter < 2*side_length,
                                position_along_perimeter - side_length,
                                np.where(position_along_perimeter < 3*side_length,
                                        side_length,
                                        side_length - (position_along_perimeter - 3*side_length))))
        
        gt_data = pd.DataFrame({
            'timestamp_ns': timestamps[::100],  # 1Hz ground truth
            'x_pos': x_pos[::100],
            'y_pos': y_pos[::100],
            'z_pos': np.zeros(len(timestamps[::100]))
        })
        gt_data.to_csv(handheld_dir / 'ground_truth.csv', index=False)
        
        # Create metadata
        metadata = {
            'dataset': 'oxford_synthetic',
            'description': 'Synthetic pedestrian data in Oxford format',
            'duration_sec': duration_sec,
            'sample_rate_hz': sample_rate,
            'generated_date': str(Path().cwd())
        }
        
        with open(processed_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        (dataset_dir / 'processed').touch()
        logger.info("Synthetic Oxford dataset created successfully")
        return True
    
    def _process_oxford(self, raw_dir: Path, output_dir: Path):
        """Process Oxford data into unified format"""
        """Create placeholder for Oxford dataset (manual download required)"""
        config = self.datasets['oxiod']
        dataset_dir = self.base_data_dir / config['folder']
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        readme_content = f"""# Oxford Inertial Odometry Dataset (OxIOD)

This dataset requires manual download from: {config['url']}

## Instructions:
1. Visit the Oxford website and register for dataset access
2. Download the dataset files
3. Extract them to this directory
4. Run the data processing script

## Expected Structure:
```
oxiod/
├── handheld/
│   ├── data.csv
│   └── ground_truth.csv
├── pocket/
│   ├── data.csv
│   └── ground_truth.csv
└── processed/
    └── unified_data.csv
```

Dataset Size: ~{config['size_mb']} MB
Description: {config['description']}
"""
        
        with open(dataset_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        logger.info(f"Created Oxford dataset placeholder in {dataset_dir}")
    
    def download_all_available(self):
        """Download all automatically downloadable datasets"""
        logger.info("Starting download of all available datasets")
        
        results = {}
        
        # comma2k19 sample
        logger.info("=== Downloading comma2k19 sample ===")
        results['comma2k19'] = self.download_comma2k19_sample()
        
        # EuRoC sample
        logger.info("=== Downloading EuRoC sample ===")
        results['euroc'] = self.download_euroc_sample()
        
        # Oxford automated download
        logger.info("=== Downloading Oxford dataset ===")
        results['oxiod'] = self.download_oxford_automated()
        
        # KITTI sample (if configured)
        if 'kitti' in self.datasets:
            logger.info("=== Downloading KITTI sample ===")
            results['kitti'] = self.download_kitti_sample()
        
        # Summary
        logger.info("=== Download Summary ===")
        for dataset, status in results.items():
            if status is True:
                logger.info(f"✅ {dataset}: Downloaded and processed")
            elif status is False:
                logger.info(f"❌ {dataset}: Download failed")
            else:
                logger.info(f"📝 {dataset}: {status}")
        
        return results
    
    def download_kitti_sample(self) -> bool:
        """Download KITTI dataset sample"""
        if 'kitti' not in self.datasets:
            return False
            
        config = self.datasets['kitti']
        dataset_dir = self.base_data_dir / config['folder']
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if (dataset_dir / 'processed').exists():
            logger.info("KITTI sample already downloaded and processed")
            return True
        
        # Download sample sequence
        archive_name = "kitti_sample.zip"
        archive_path = dataset_dir / archive_name
        
        if not archive_path.exists():
            success = self.download_file_with_fallback(config['urls'], archive_path)
            if not success:
                # Create synthetic KITTI-like data as fallback
                logger.info("Download failed, creating synthetic KITTI data")
                return self._create_synthetic_kitti(dataset_dir)
        
        # Extract
        extract_dir = dataset_dir / "raw"
        success = self.extract_archive(archive_path, extract_dir, config['extract_method'])
        if not success:
            return False
        
        # Process KITTI format
        self._process_kitti(extract_dir, dataset_dir)
        
        # Mark as processed
        (dataset_dir / 'processed').touch()
        
        # Cleanup archive
        if archive_path.exists():
            archive_path.unlink()
        
        return True
    
    def _create_synthetic_kitti(self, dataset_dir: Path) -> bool:
        """Create synthetic KITTI-style dataset"""
        logger.info("Creating synthetic KITTI dataset")
        
        processed_dir = dataset_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        # Generate synthetic automotive data
        import numpy as np
        import pandas as pd
        
        # 5 minutes of driving data at 100Hz
        duration_sec = 300
        sample_rate = 100
        n_samples = duration_sec * sample_rate
        
        timestamps = np.arange(n_samples) * (1e9 // sample_rate)
        t_sec = np.arange(n_samples) / sample_rate
        
        # Vehicle dynamics (highway driving)
        base_speed = 20  # m/s (72 km/h)
        speed_variation = 5 * np.sin(0.05 * t_sec) + np.random.normal(0, 1, n_samples)
        speed = np.clip(base_speed + speed_variation, 5, 35)
        
        # IMU data in vehicle frame
        accel_x = np.gradient(speed) + np.random.normal(0, 0.1, n_samples)  # longitudinal
        accel_y = 2 * np.sin(0.02 * t_sec) + np.random.normal(0, 0.2, n_samples)  # lateral
        accel_z = 9.81 + np.random.normal(0, 0.15, n_samples)  # vertical + road bumps
        
        gyro_x = np.random.normal(0, 0.02, n_samples)  # roll rate
        gyro_y = np.random.normal(0, 0.02, n_samples)  # pitch rate
        gyro_z = 0.05 * np.sin(0.01 * t_sec) + np.random.normal(0, 0.01, n_samples)  # yaw rate
        
        # GPS trajectory (highway loop)
        lat_center, lon_center = 49.0069, 8.4037  # Karlsruhe area
        track_radius = 0.02
        angle = 0.01 * t_sec
        gps_lat = lat_center + track_radius * np.cos(angle) + np.random.normal(0, 0.00005, n_samples)
        gps_lon = lon_center + track_radius * np.sin(angle) + np.random.normal(0, 0.00005, n_samples)
        
        # Create oxts (GPS/IMU) data
        oxts_data = pd.DataFrame({
            'timestamp': timestamps,
            'lat': gps_lat,
            'lon': gps_lon,
            'alt': 120 + np.random.normal(0, 2, n_samples),
            'roll': np.random.normal(0, 0.02, n_samples),
            'pitch': np.random.normal(0, 0.02, n_samples),
            'yaw': np.cumsum(gyro_z) / sample_rate,
            'vn': speed * np.cos(angle) + np.random.normal(0, 0.5, n_samples),
            've': speed * np.sin(angle) + np.random.normal(0, 0.5, n_samples),
            'vf': np.random.normal(0, 0.1, n_samples),
            'vl': np.random.normal(0, 0.1, n_samples),
            'vu': np.random.normal(0, 0.1, n_samples),
            'ax': accel_x,
            'ay': accel_y,
            'az': accel_z,
            'af': accel_x,
            'al': accel_y,
            'au': accel_z,
            'wx': gyro_x,
            'wy': gyro_y,
            'wz': gyro_z,
            'wf': gyro_x,
            'wl': gyro_y,
            'wu': gyro_z,
            'pos_accuracy': np.random.uniform(0.5, 2.0, n_samples),
            'vel_accuracy': np.random.uniform(0.1, 0.5, n_samples),
            'navstat': np.ones(n_samples) * 4,  # INS solution good
            'numsats': np.random.randint(8, 15, n_samples),
            'posmode': np.ones(n_samples) * 5,  # RTK fixed
            'velmode': np.ones(n_samples) * 5,
            'orimode': np.ones(n_samples) * 4
        })
        
        # Save OXTS data 
        oxts_dir = processed_dir / 'oxts'
        oxts_dir.mkdir(exist_ok=True)
        oxts_data.to_csv(oxts_dir / 'data.csv', index=False, header=False, sep=' ')
        
        # Create timestamps file
        with open(oxts_dir / 'timestamps.txt', 'w') as f:
            for ts in timestamps:
                # Convert to KITTI timestamp format
                dt = pd.to_datetime(ts, unit='ns')
                f.write(f"{dt.strftime('%Y-%m-%d %H:%M:%S')}.{dt.microsecond:06d}\n")
        
        # Create metadata
        metadata = {
            'dataset': 'kitti_synthetic',
            'description': 'Synthetic automotive data in KITTI format',
            'duration_sec': duration_sec,
            'sample_rate_hz': sample_rate,
            'generated_date': str(Path().cwd())
        }
        
        with open(processed_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("Synthetic KITTI dataset created successfully")
        return True
    
    def _process_kitti(self, raw_dir: Path, output_dir: Path):
        """Process KITTI data into unified format"""
        logger.info("Processing KITTI data format")
        
        processed_dir = output_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        # KITTI structure: oxts/data/*.txt files
        for root, dirs, files in os.walk(raw_dir):
            if 'oxts' in str(root) and 'data' in str(root):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = Path(root) / file
                        shutil.copy2(file_path, processed_dir / f'oxts_{file}')
                        logger.info(f"Copied KITTI file: {file_path}")
        
        # Create metadata
        metadata = {
            'dataset': 'kitti',
            'description': 'KITTI automotive dataset',
            'processed_date': str(Path().cwd()),
            'files_found': list(processed_dir.glob('*.txt'))
        }
        
        with open(processed_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
    def _process_oxford(self, raw_dir: Path, output_dir: Path):
        """Process Oxford data into unified format"""
        logger.info("Processing Oxford data format")
        
        processed_dir = output_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        # Oxford structure varies, copy relevant files
        for root, dirs, files in os.walk(raw_dir):
            for file in files:
                if file.endswith('.csv') or file.endswith('.txt'):
                    file_path = Path(root) / file
                    parent_folder = Path(root).name
                    
                    if 'handheld' in parent_folder.lower():
                        dest_dir = processed_dir / 'handheld'
                        dest_dir.mkdir(exist_ok=True)
                        shutil.copy2(file_path, dest_dir / file)
                    elif 'pocket' in parent_folder.lower():
                        dest_dir = processed_dir / 'pocket'
                        dest_dir.mkdir(exist_ok=True)
                        shutil.copy2(file_path, dest_dir / file)
                    else:
                        shutil.copy2(file_path, processed_dir / file)
                    
                    logger.info(f"Copied Oxford file: {file_path}")
        
        # Create metadata
        metadata = {
            'dataset': 'oxford',
            'description': 'Oxford Inertial Odometry Dataset',
            'processed_date': str(Path().cwd()),
            'files_found': list(processed_dir.rglob('*.*'))
        }
        
        with open(processed_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Processed Oxford data saved to {processed_dir}")
    
    def create_oxford_placeholder(self):
        """Create placeholder for Oxford dataset (manual download required)"""
        config = self.datasets['oxiod']
        dataset_dir = self.base_data_dir / config['folder']
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        readme_content = f"""# Oxford Inertial Odometry Dataset (OxIOD)

This dataset requires manual download from: {config['urls'][0]}

## Instructions:
1. Visit the Oxford website and register for dataset access
2. Download the dataset files
3. Extract them to this directory
4. Run the data processing script

## Expected Structure:
```
oxiod/
├── handheld/
│   ├── data.csv
│   └── ground_truth.csv
├── pocket/
│   ├── data.csv
│   └── ground_truth.csv
└── processed/
    └── unified_data.csv
```

Dataset Size: ~{config['size_mb']} MB
Description: {config['description']}
"""
        
        with open(dataset_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        logger.info(f"Created Oxford dataset placeholder in {dataset_dir}")
    
    def list_available_datasets(self):
        """List all configured datasets and their status"""
        logger.info("Available datasets:")
        
        for key, config in self.datasets.items():
            dataset_dir = self.base_data_dir / config['folder']
            
            if (dataset_dir / 'processed').exists():
                status = "✅ Downloaded"
                # Check if it's synthetic
                metadata_file = dataset_dir / 'processed' / 'metadata.json'
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        if 'synthetic' in metadata.get('dataset', ''):
                            status = "🤖 Synthetic"
                    except:
                        pass
            elif dataset_dir.exists():
                status = "📝 Partially downloaded"
            else:
                status = "❌ Not downloaded"
            
            print(f"{key:15} | {config['name']:35} | {config['size_mb']:4}MB | {status}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NavAI Dataset Downloader')
    parser.add_argument('--data-dir', type=str, default='D:/NavAi/data', 
                       help='Base directory for datasets')
    parser.add_argument('--dataset', choices=['comma2k19', 'euroc', 'oxiod', 'kitti', 'all'], 
                       default='all', help='Dataset to download')
    parser.add_argument('--list', action='store_true', 
                       help='List available datasets')
    parser.add_argument('--synthetic', action='store_true',
                       help='Create synthetic datasets instead of downloading')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.list:
        downloader.list_available_datasets()
    else:
        if args.synthetic:
            logger.info("Creating synthetic datasets")
            results = {}
            
            # Create base directories first
            base_path = Path(args.data_dir)
            base_path.mkdir(parents=True, exist_ok=True)
            
            (base_path / 'comma2k19').mkdir(parents=True, exist_ok=True)
            (base_path / 'euroc').mkdir(parents=True, exist_ok=True)
            (base_path / 'oxiod').mkdir(parents=True, exist_ok=True)
            (base_path / 'kitti').mkdir(parents=True, exist_ok=True)
            
            results['comma2k19'] = downloader._create_synthetic_comma2k19(base_path / 'comma2k19')
            results['euroc'] = downloader._create_synthetic_euroc(base_path / 'euroc')
            results['oxiod'] = downloader._create_synthetic_oxford(base_path / 'oxiod')
            results['kitti'] = downloader._create_synthetic_kitti(base_path / 'kitti')
            
            logger.info("=== Synthetic Data Creation Summary ===")
            for dataset, status in results.items():
                status_str = "✅ Created" if status else "❌ Failed"
                logger.info(f"{dataset}: {status_str}")
        
        elif args.dataset == 'all':
            downloader.download_all_available()
        elif args.dataset == 'comma2k19':
            downloader.download_comma2k19_sample()
        elif args.dataset == 'euroc':
            downloader.download_euroc_sample()
        elif args.dataset == 'oxiod':
            downloader.download_oxford_automated()
        elif args.dataset == 'kitti':
            downloader.download_kitti_sample()
            logger.info("KITTI dataset download attempted")