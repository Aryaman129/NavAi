# NavAI - Advanced Mobile Navigation System

**Status**: Phase 1 Complete ✅ | **Version**: 1.0.0 | **Last Updated**: October 10, 2025

A real-time IMU-based navigation system using machine learning and sensor fusion for accurate positioning without GPS dependency.

---

## 🎯 Project Overview

NavAI combines:
- **IMU Sensor Fusion** with Extended Kalman Filter (EKF)
- **Machine Learning Speed Estimation** using TensorFlow Lite (✅ Deployed)
- **High-Frequency Sensor Logging** at 100Hz (✅ Working)
- **Real-time Android Implementation** optimized for mobile devices
- **Offline Map Matching** (Planned for Phase 2)
- **Optional ARCore VIO** for enhanced accuracy (Planned)

---

## 📊 Current Performance (Phase 1 Complete)

### ML Model Performance
- **Speed RMSE**: 2.8251 m/s (Target: <12 m/s) ✅ **4.2x better than target**
- **R² Score**: 0.9306 (93% variance explained) ✅
- **MAE**: 0.39 m/s
- **Model Size**: 3.72 MB (975K parameters)
- **Inference Time**: ~12ms per prediction
- **Training Data**: 478,891 samples from Comma2k19 dataset

### Android App Performance
- **Sensor Sampling**: 100Hz (Accelerometer, Gyroscope, Magnetometer, Rotation)
- **GPS Updates**: 5Hz
- **Prediction Rate**: 10Hz
- **Memory Usage**: <100MB RAM
- **Battery Impact**: Minimal (foreground service optimization)

---

## 🏗️ Project Structure

```
NavAi/
├── mobile/                 # Android application
│   ├── app/               # Main app (sensor logger + speed prediction)
│   └── sensor-fusion/     # Sensor fusion library module
├── ml/                    # Machine learning pipeline
│   ├── models/           # Model architectures (BiLSTM)
│   ├── training/         # Training scripts
│   ├── evaluation/       # Evaluation utilities
│   ├── outputs/          # Trained models & results
│   └── requirements.txt
├── docs/                 # 📚 Documentation (start here!)
│   └── README.md         # Documentation index
├── scripts/              # Utility scripts
│   ├── android/          # Android build/debug scripts
│   └── [various].py      # Python utilities
├── data/                 # Datasets
│   └── comma2k19/        # Comma2k19 dataset (478K samples)
├── tests/                # Test files
├── improvements/         # Experimental features
└── quick_debug.ps1       # Quick Android debug script
```

---

## 🚀 Quick Start

### 📱 Option 1: Use the Android App (Recommended)

**Prerequisites:**
- Android device (Android 8.0+ / API 26+)
- USB debugging enabled
- ADB installed

**Build & Install:**
```powershell
# Navigate to mobile directory
cd mobile

# Set environment variables
$env:JAVA_HOME = "D:\Android SDK\jbr"
$env:GRADLE_USER_HOME = "D:\.gradle"

# Build debug APK
.\gradlew.bat assembleDebug --no-daemon

# Install on device
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

**Or use the quick debug script:**
```powershell
.\quick_debug.ps1
```

**Using the App:**
1. Open NavAI app on your device
2. Grant permissions (Location, Sensors)
3. Choose:
   - **Speed Prediction**: Real-time AI speed estimation
   - **Sensor Logger**: Record raw sensor data to CSV

---

### 🧠 Option 2: Train Your Own Model

**Prerequisites:**
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

**Setup:**
```bash
cd ml
pip install -r requirements.txt
```

**Train:**
```bash
python training/train_phase1_gpu_optimized.py
```

**Export to TFLite:**
```bash
python export_phase1_tflite_from_keras.py
```

---

## � Documentation

All documentation is in the `docs/` folder. **Start here:**

- **[docs/README.md](docs/README.md)** - Documentation index and navigation
- **[docs/CURRENT_APP_STATUS.md](docs/CURRENT_APP_STATUS.md)** - Android app features & testing
- **[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** - Current metrics and next steps
- **[docs/ANDROID_DEPLOYMENT_GUIDE.md](docs/ANDROID_DEPLOYMENT_GUIDE.md)** - Build and deployment instructions
- **[docs/SYSTEM_OVERVIEW.md](docs/SYSTEM_OVERVIEW.md)** - Architecture and design

---

## 📈 Development Status

### ✅ Phase 1: Foundation (COMPLETE)
- ✅ Android sensor logger with 100Hz sampling
- ✅ BiLSTM model trained on 478K samples
- ✅ TFLite model export and integration
- ✅ Real-time speed prediction
- ✅ EKF post-processing
- ✅ CSV data export
- ✅ Foreground services for continuous logging

### 🔄 Phase 2: Advanced Features (In Progress)
- 🔨 Service-UI broadcast communication (90% complete)
- 🔨 Real-time prediction visualization
- 🔨 File sharing and export UI
- ⏳ Attention mechanism for model
- ⏳ Physics-informed loss functions
- ⏳ Uncertainty estimation

### 📋 Phase 3: Sensor Fusion (Planned)
- ⏳ EKF sensor fusion engine
- ⏳ Zero Velocity Update (ZUPT)
- ⏳ Map matching integration
- ⏳ ARCore VIO (optional)

### 📋 Phase 4: Production (Planned)
- ⏳ Battery optimization
- ⏳ Model quantization
- ⏳ Cloud sync (optional)
- ⏳ Release build

---

## 🛠️ Technology Stack

**Mobile App:**
- **Language**: Kotlin
- **UI**: Jetpack Compose
- **ML**: TensorFlow Lite
- **Sensors**: Android SensorManager, FusedLocationClient
- **Persistence**: CSV files (External Storage)

**Machine Learning:**
- **Framework**: TensorFlow/Keras
- **Architecture**: BiLSTM (3 layers, 128 hidden units)
- **Preprocessing**: StandardScaler, Feature Engineering
- **Post-processing**: Extended Kalman Filter (EKF)
- **Export**: TFLite converter with optimization

**Development Tools:**
- **Android**: Gradle, ADB, Android Studio
- **Python**: NumPy, Pandas, Matplotlib, Jupyter
- **Datasets**: Comma2k19 (478K samples)

---

## 🔬 Research Foundation

Based on proven datasets and methods:
- **comma2k19**: Driving dataset with IMU+GPS+camera (primary dataset)
- **IO-VNBD**: Vehicle inertial navigation benchmark dataset
- **OxIOD**: Oxford inertial odometry dataset  
- **AVNet**: Learned attitude & velocity estimation with InEKF

---

## 📱 Supported Devices

- **Primary Target**: OnePlus 11R (tested and verified)
- **Requirements**: 
  - Android 8.0+ (API 26+)
  - IMU sensors (accelerometer, gyroscope, magnetometer)
  - GPS (for ground truth and augmentation)
  - 2GB+ RAM (for TFLite inference)

---

## 🐛 Known Issues & Fixes

### All Critical Issues Resolved ✅
- ✅ Android SDK ROOT not set → Fixed (set to proper SDK path)
- ✅ App buttons not working → Fixed (permission logic updated)
- ✅ Service state not syncing with UI → Fixed (broadcast communication added)
- ✅ Build errors → Fixed (class structure and ViewModel usage)

---

## 📞 Support & Documentation

**Need Help?**
1. Check **[docs/README.md](docs/README.md)** for documentation index
2. Read **[docs/CURRENT_APP_STATUS.md](docs/CURRENT_APP_STATUS.md)** for app usage
3. Review **[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** for current metrics

**Quick Commands:**
```powershell
# Build and install app
.\quick_debug.ps1

# Check Android setup
.\scripts\android\check_android_setup.ps1

# Train model
python ml/training/train_phase1_gpu_optimized.py
```

---

## 📄 License

Apache 2.0 - See LICENSE file for details

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Submit pull request

---

## 📊 Project Stats

- **Lines of Code**: ~15,000 (Kotlin + Python)
- **Model Parameters**: 975,105 (BiLSTM)
- **Training Samples**: 478,891 (Comma2k19)
- **Test RMSE**: 2.8251 m/s
- **APK Size**: ~8 MB (with TFLite model)

---

**Status**: � Phase 1 Complete - App deployed and functional!

**Made with ❤️ for advanced mobile navigation**
