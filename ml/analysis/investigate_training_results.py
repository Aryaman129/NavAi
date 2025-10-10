"""
Investigate the suspicious training results:
- Why is R² = 0.0000?
- Why is RMSE = MAE exactly?
- Are predictions actually varying?
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("🔍 INVESTIGATING TRAINING RESULTS")
print("="*80)

# Load checkpoint
checkpoint_path = '../training/checkpoints/best_model_fixed.pth'
checkpoint = torch.load(checkpoint_path, weights_only=False)

print(f"\n📦 Checkpoint Info:")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Val RMSE: {checkpoint['val_rmse']:.4f} m/s")

# Load results
results_path = './reports/training_results_20251009_013119.pt'
results = torch.load(results_path, weights_only=False)

print(f"\n📊 Training History:")
history = results['history']
print(f"  Epochs trained: {len(history['train_loss'])}")
print(f"  Best val RMSE: {min(history['val_rmse']):.4f} m/s")
print(f"  Final train loss: {history['train_loss'][-1]:.4f}")

print(f"\n🎯 Test Results:")
test_res = results['test_results']
print(f"  RMSE: {test_res['rmse']:.4f} m/s")
print(f"  MAE: {test_res['mae']:.4f} m/s")
print(f"  R²: {test_res['r2']:.6f}")
print(f"  Baseline RMSE: {test_res['baseline_rmse']:.4f} m/s")

# Check if RMSE = MAE (suspicious!)
if abs(test_res['rmse'] - test_res['mae']) < 0.001:
    print(f"\n⚠️  WARNING: RMSE = MAE exactly! This suggests:")
    print(f"     - All errors might be the same magnitude")
    print(f"     - OR there's a calculation bug")
    print(f"     - RMSE should be >= MAE always")

# Check R² = 0
if abs(test_res['r2']) < 0.001:
    print(f"\n⚠️  WARNING: R² = 0.0 exactly! This means:")
    print(f"     - Model is exactly as good as predicting the mean")
    print(f"     - But we claimed 1.9% improvement over baseline?")
    print(f"     - This is contradictory!")

print(f"\n📈 Detailed Training History:")
print(f"\n  Epoch | Train Loss | Val RMSE | Val MAE | Val R²")
print(f"  ------|------------|----------|---------|--------")
for i in range(len(history['train_loss'])):
    print(f"  {i+1:5d} | {history['train_loss'][i]:10.4f} | "
          f"{history['val_rmse'][i]:8.3f} | {history['val_mae'][i]:7.3f} | "
          f"{history['val_r2'][i]:8.4f}")

# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Train loss
axes[0, 0].plot(history['train_loss'], 'b-', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MSE Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].grid(True, alpha=0.3)

# Val RMSE
axes[0, 1].plot(history['val_rmse'], 'r-', linewidth=2)
axes[0, 1].axhline(test_res['baseline_rmse'], color='k', linestyle='--', label='Baseline')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('RMSE (m/s)')
axes[0, 1].set_title('Validation RMSE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Val MAE
axes[1, 0].plot(history['val_mae'], 'g-', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('MAE (m/s)')
axes[1, 0].set_title('Validation MAE')
axes[1, 0].grid(True, alpha=0.3)

# Val R²
axes[1, 1].plot(history['val_r2'], 'purple', linewidth=2)
axes[1, 1].axhline(0, color='k', linestyle='--', label='Baseline (R²=0)')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('R²')
axes[1, 1].set_title('Validation R²')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./visualizations/training_history_analysis.png', dpi=150)
print(f"\n💾 Saved detailed history plot to: ./visualizations/training_history_analysis.png")

print("\n" + "="*80)
print("✅ Investigation complete!")
print("="*80)
