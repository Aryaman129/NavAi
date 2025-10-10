"""Quick investigation of training results bugs"""
import torch
import numpy as np

print("=" * 80)
print("🔍 INVESTIGATING TRAINING RESULTS BUGS")
print("=" * 80)

# Load results
results_path = './reports/training_results_20251009_013119.pt'
print(f"\n📂 Loading: {results_path}")
results = torch.load(results_path, weights_only=False, map_location='cpu')

# Extract test results
test_res = results['test_results']

print(f"\n📊 Test Results:")
print(f"  RMSE:          {test_res['rmse']:.10f} m/s")
print(f"  MAE:           {test_res['mae']:.10f} m/s")
print(f"  R²:            {test_res['r2']:.10f}")
print(f"  Baseline RMSE: {test_res['baseline_rmse']:.10f} m/s")

# Check for bugs
print(f"\n🐛 Bug Detection:")

# Bug 1: RMSE = MAE
rmse_mae_diff = abs(test_res['rmse'] - test_res['mae'])
print(f"\n1. RMSE vs MAE:")
print(f"   Difference: {rmse_mae_diff:.10f}")
if rmse_mae_diff < 0.0001:
    print(f"   ⚠️  BUG CONFIRMED: RMSE = MAE exactly!")
    print(f"   ⚠️  This is mathematically impossible!")
    print(f"   ⚠️  RMSE must be >= MAE (RMSE² = mean of squared errors)")
else:
    print(f"   ✅ No bug: RMSE > MAE as expected")

# Bug 2: R² = 0
print(f"\n2. R² Score:")
print(f"   Value: {test_res['r2']:.10f}")
if abs(test_res['r2']) < 0.0001:
    print(f"   ⚠️  BUG CONFIRMED: R² = 0.0 exactly!")
    print(f"   ⚠️  But report claims 1.9% improvement over baseline")
    print(f"   ⚠️  If RMSE (15.356) < Baseline (15.648), R² should be > 0")
else:
    print(f"   ✅ No bug: R² has reasonable value")

# Calculate what R² SHOULD be
expected_r2 = 1 - (test_res['rmse'] ** 2) / (test_res['baseline_rmse'] ** 2)
print(f"\n3. Expected R² Calculation:")
print(f"   R² = 1 - (RMSE² / Baseline²)")
print(f"   R² = 1 - ({test_res['rmse']:.4f}² / {test_res['baseline_rmse']:.4f}²)")
print(f"   R² = 1 - ({test_res['rmse']**2:.4f} / {test_res['baseline_rmse']**2:.4f})")
print(f"   R² = 1 - {(test_res['rmse']**2)/(test_res['baseline_rmse']**2):.6f}")
print(f"   R² = {expected_r2:.6f}")
print(f"   Actual R²: {test_res['r2']:.6f}")
print(f"   Discrepancy: {abs(expected_r2 - test_res['r2']):.6f}")

# Check training history
history = results['history']
print(f"\n📈 Training History:")
print(f"   Epochs: {len(history['train_loss'])}")
print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
print(f"   Best val RMSE: {min(history['val_rmse']):.4f} m/s")
print(f"   Best val MAE: {min(history['val_mae']):.4f} m/s")
print(f"   Best val R²: {max(history['val_r2']):.6f}")

# Check if predictions are varying
print(f"\n🎯 Prediction Analysis:")
if 'predictions' in test_res and 'actuals' in test_res:
    preds = np.array(test_res['predictions'])
    actuals = np.array(test_res['actuals'])
    
    print(f"   Predictions - min: {preds.min():.2f}, max: {preds.max():.2f}, std: {preds.std():.2f}")
    print(f"   Actuals - min: {actuals.min():.2f}, max: {actuals.max():.2f}, std: {actuals.std():.2f}")
    
    # Check if predicting constant
    if preds.std() < 0.1:
        print(f"   ⚠️  WARNING: Predictions have very low variance!")
        print(f"   ⚠️  Model might be predicting nearly constant value")
    else:
        print(f"   ✅ Predictions are varying reasonably")
else:
    print(f"   ⚠️  Predictions not saved in results")

print(f"\n" + "=" * 80)
print("✅ Investigation Complete!")
print("=" * 80)

# Summary
print(f"\n📋 SUMMARY:")
if rmse_mae_diff < 0.0001:
    print(f"   ❌ RMSE = MAE bug: YES (calculation error)")
else:
    print(f"   ✅ RMSE = MAE bug: NO")

if abs(test_res['r2']) < 0.0001:
    print(f"   ❌ R² = 0 bug: YES (should be {expected_r2:.4f})")
else:
    print(f"   ✅ R² = 0 bug: NO")

print(f"\n💡 Next Steps:")
if rmse_mae_diff < 0.0001 or abs(test_res['r2']) < 0.0001:
    print(f"   1. Check evaluation code in train_speed_estimation_fixed.py")
    print(f"   2. Look for metric calculation functions")
    print(f"   3. Fix the bugs and re-run training")
else:
    print(f"   ✅ No bugs found - results are valid!")
