import torch
import torch.nn as nn

class BaselineSpeedEstimator(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0, 
                          bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

model = BaselineSpeedEstimator(input_dim=10, hidden_dim=128, num_layers=3)
total_params = sum(p.numel() for p in model.parameters())

print("=" * 60)
print("📊 MODEL CONFIGURATION")
print("=" * 60)
print(f"Total Parameters: {total_params:,}")
print(f"Model Size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")
print()
print("Architecture:")
print(f"  - Input: 10 features (6 raw IMU + 4 engineered)")
print(f"  - BiLSTM: 3 layers x 128 hidden x 2 directions")
print(f"  - FC layers: 256 → 128 → 64 → 1")
print("=" * 60)
