# Fast GPU Training for NavAI with Live Updates
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path
import sys
sys.path.append('.')

from ml.data.data_loader import DataLoader
from tqdm import tqdm
import logging
import psutil
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUTrainer:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_loader = DataLoader(target_sample_rate=100)
        self.model = None
        
        print(f' GPU Trainer initialized on: {self.device}')
        if torch.cuda.is_available():
            print(f' GPU: {torch.cuda.get_device_name(0)}')
            print(f' VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
        
    def load_data(self):
        print(' Loading datasets...')
        datasets = {}
        
        # ONLY comma2k19 as requested by user
        for name in ['comma2k19']:
            path = Path(self.data_dir) / name
            try:
                df = self.data_loader.load_comma2k19(str(path))
                
                if not df.empty:
                    # Downsample if too large to fit in memory
                    if len(df) > 100000:
                        print(f" Downsampling {name} from {len(df)} to 100k samples...")
                        df = df.iloc[::len(df)//100000]
                    datasets[name] = df
                    print(f' {name}: {len(df)} samples')
                else:
                    print(f' {name}: empty')
            except Exception as e:
                print(f' {name}: {str(e)[:30]}')
        
        if datasets:
            combined = pd.concat(datasets.values(), ignore_index=True)
            print(f' Combined: {len(combined)} total samples')
            return combined
        else:
            raise ValueError('No datasets loaded!')
    
    def prepare_data(self, df):
        print(' Preparing training data...')
        
        feature_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        inputs, targets = [], []
        
        seq_len = 8
        for i in range(0, len(df) - seq_len, 5):  # Skip for speed
            sequence = df.iloc[i:i+seq_len][feature_cols].values
            target_row = df.iloc[i + seq_len]
            
            if 'gps_speed_mps' in target_row and target_row['gps_speed_mps'] > 0:
                target = target_row['gps_speed_mps']
            else:
                accel_mag = np.linalg.norm(sequence[-1, :3])
                target = max(0, accel_mag - 9.8)
            
            inputs.append(sequence)
            targets.append(target)
        
        X = torch.FloatTensor(inputs).to(self.device)
        y = torch.FloatTensor(targets).to(self.device)
        
        gpu_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        print(f' Data ready: {len(inputs)} sequences, GPU: {gpu_mem:.0f}MB')
        
        return X, y
    
    def create_model(self):
        class FastSpeedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(6, 64, batch_first=True, num_layers=2, dropout=0.1)
                self.fc1 = nn.Linear(64, 32)
                self.fc2 = nn.Linear(32, 1)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                x = lstm_out[:, -1, :]
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return torch.abs(x)
        
        model = FastSpeedModel().to(self.device)
        params = sum(p.numel() for p in model.parameters())
        print(f' Model created: {params:,} parameters')
        return model
    
    def train(self, X, y, epochs=20, batch_size=256):
        print(' Starting GPU training...')
        
        self.model = self.create_model()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        start_time = time.time()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0
            num_batches = (len(X) + batch_size - 1) // batch_size
            
            self.model.train()
            
            with tqdm(range(num_batches), desc=f'Epoch {epoch+1:2d}/{epochs}', leave=False) as pbar:
                for batch_idx in pbar:
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(X))
                    
                    batch_X = X[start_idx:end_idx]
                    batch_y = y[start_idx:end_idx].unsqueeze(1)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    gpu_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                    samples_per_sec = batch_size * (batch_idx + 1) / (time.time() - epoch_start)
                    
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.6f}',
                        'GPU': f'{gpu_mem:.0f}MB',
                        'Speed': f'{samples_per_sec:.0f}/s'
                    })
            
            avg_loss = epoch_loss / num_batches
            epoch_time = time.time() - epoch_start
            
            print(f'Epoch {epoch+1:2d} | Loss: {avg_loss:.6f} | Time: {epoch_time:.1f}s | Speed: {len(X)/epoch_time:.0f} samples/s')
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), 'best_gpu_model.pth')
            
            if avg_loss < 0.001:
                print(f' Early stopping! Excellent loss: {avg_loss:.6f}')
                break
        
        total_time = time.time() - start_time
        print(f' Training complete! Total: {total_time:.1f}s, Best loss: {best_loss:.6f}')
        
        return self.model
    
    def evaluate(self, X, y):
        print(' Evaluating performance...')
        
        self.model.eval()
        eval_size = min(3000, len(X))
        
        with torch.no_grad():
            predictions = self.model(X[:eval_size]).cpu().numpy().flatten()
            targets = y[:eval_size].cpu().numpy()
            
            rmse = np.sqrt(np.mean((predictions - targets) ** 2))
            mae = np.mean(np.abs(predictions - targets))
            
            print(f' Results: RMSE={rmse:.6f}, MAE={mae:.6f}')
            
            if rmse < 0.01:
                print(' EXCELLENT performance!')
            elif rmse < 0.05:
                print(' GOOD performance!')
            else:
                print(' MODERATE performance')
        
        return {'rmse': rmse, 'mae': mae}

def main():
    try:
        trainer = GPUTrainer()
        df = trainer.load_data()
        X, y = trainer.prepare_data(df)
        model = trainer.train(X, y, epochs=20, batch_size=512)
        metrics = trainer.evaluate(X, y)
        
        print(' GPU Training Pipeline Complete!')
        print(' Ready for real-world testing!')
        
        return True
    except Exception as e:
        print(f' Training failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
