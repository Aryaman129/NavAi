# NavAI Quick Start Guide

## 🚀 Immediate Action Plan for RTX 4050 Setup

### Phase 1: Environment Setup (5 minutes)

```bash
# 1. Setup Python environment with CUDA support
python setup_environment.py

# 2. Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### Phase 2: Android App Build & Deploy (10 minutes)

```bash
# 1. Build Android app
cd mobile
./gradlew assembleDebug

# 2. Install on OnePlus 11R (enable USB debugging first)
adb devices  # Verify device connected
adb install app/build/outputs/apk/debug/app-debug.apk

# 3. Grant permissions on device:
# - Location (Fine & Coarse)
# - High sampling rate sensors
```

### Phase 3: Data Collection (15-30 minutes)

```bash
# 1. Open NavAI Logger app on OnePlus 11R
# 2. Start logging
# 3. Take a 5-10 minute drive/walk with GPS enabled
# 4. Stop logging
# 5. Export logs via app or pull from device:

adb pull /sdcard/Android/data/com.navai.logger/files/logs/ ml/data/navai_logs/
```

### Phase 4: Local GPU Training (10-20 minutes)

```bash
# 1. Train CNN model on RTX 4050
cd ml
python train_local.py --model-type cnn --batch-size 16 --num-epochs 50

# 2. Train LSTM model for comparison
python train_local.py --model-type lstm --batch-size 8 --num-epochs 30

# 3. Check results
cat models/training_results.txt
```

### Phase 5: Model Export & Integration (15 minutes)

```bash
# 1. Convert to TensorFlow Lite
python -c "
import sys; sys.path.append('.')
from models.speed_estimator import create_tensorflow_model, convert_to_tflite
import numpy as np

# Create and convert model
model = create_tensorflow_model((150, 6), 'cnn')
model.compile(optimizer='adam', loss='mse')

# Dummy training for structure
X_dummy = np.random.randn(100, 150, 6)
y_dummy = np.random.randn(100, 1)
model.fit(X_dummy, y_dummy, epochs=1, verbose=0)

# Convert to TFLite
tflite_model = convert_to_tflite(model, quantize=True, representative_dataset=X_dummy)

# Save
with open('models/speed_estimator.tflite', 'wb') as f:
    f.write(tflite_model)

print('TFLite model saved!')
"

# 2. Copy to Android assets
cp models/speed_estimator.tflite mobile/app/src/main/assets/
```

## 🎯 Success Criteria

After completing these steps, you should have:

1. ✅ **Working Android app** installed on OnePlus 11R
2. ✅ **Sensor data collected** from real device usage
3. ✅ **Trained ML model** with <15% speed estimation error
4. ✅ **TFLite model** ready for mobile deployment
5. ✅ **GPU utilization** confirmed on RTX 4050

## 🔧 Troubleshooting

### CUDA Issues
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Android Build Issues
```bash
# Clean and rebuild
cd mobile
./gradlew clean
./gradlew assembleDebug

# Check Android SDK
echo $ANDROID_HOME
```

### Memory Issues (RTX 4050 6GB)
```bash
# Reduce batch size
python train_local.py --batch-size 8 --model-type cnn

# Monitor GPU memory
nvidia-smi -l 1
```

## 📊 Expected Performance

### RTX 4050 Training Times:
- **CNN Model**: ~5-10 minutes (50 epochs)
- **LSTM Model**: ~10-15 minutes (30 epochs)
- **Memory Usage**: ~2-3GB VRAM

### Model Accuracy Targets:
- **Speed RMSE**: <10% with real data, ~15% with synthetic
- **Model Size**: <1MB (TFLite quantized)
- **Inference Time**: <10ms on mobile

## 🔄 Next Steps After Quick Start

1. **Collect more diverse data** (different speeds, routes, conditions)
2. **Implement EKF sensor fusion** (Phase 2)
3. **Add map matching** with MapLibre
4. **Integrate ARCore VIO** for enhanced accuracy
5. **Optimize for production** deployment

## 📞 When to Ask for Help

Contact me if you encounter:
- CUDA installation issues
- Android app crashes
- Training convergence problems
- Model accuracy below 20% RMSE
- Any step taking significantly longer than expected

## 🎉 Success Validation

Run this validation script after setup:

```bash
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

import tensorflow as tf
print(f'✅ TensorFlow: {tf.__version__}')
print(f'✅ TF-GPU: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')
"
```

Expected output:
```
✅ PyTorch: 2.1.0+cu121
✅ CUDA: True
✅ GPU: NVIDIA GeForce RTX 4050 Laptop GPU
✅ VRAM: 6.0GB
✅ TensorFlow: 2.15.0
✅ TF-GPU: True
```
