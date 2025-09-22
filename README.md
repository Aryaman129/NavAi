# NavAI - Advanced Mobile Navigation System

A real-time IMU-based navigation system using machine learning and sensor fusion for accurate positioning without GPS dependency.

## 🎯 Project Overview

NavAI combines:
- **IMU Sensor Fusion** with Extended Kalman Filter (EKF)
- **Machine Learning Speed Estimation** using TensorFlow Lite
- **Offline Map Matching** with MapLibre
- **Optional ARCore VIO** for enhanced accuracy
- **Real-time Android Implementation** optimized for mobile devices

## 📊 Expected Performance
- **Speed Accuracy**: 5-15% RMS error
- **Position Drift**: <20m after 5 minutes without GPS
- **Battery Life**: >8 hours continuous operation
- **Real-time Latency**: <100ms position updates

## 🏗️ Project Structure

```
NavAi/
├── mobile/                 # Android sensor logger & fusion app
│   ├── app/               # Main Android application
│   ├── sensor-fusion/     # Core sensor fusion library
│   └── build.gradle.kts
├── ml/                    # Python ML pipeline
│   ├── data/             # Dataset processing
│   ├── models/           # Model training & export
│   ├── notebooks/        # Jupyter experiments
│   └── requirements.txt
├── docs/                 # Documentation
├── scripts/              # Utility scripts
└── README.md
```

## 🚀 Quick Start

### 1. Android Sensor Logger
```bash
cd mobile
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

### 2. Python ML Pipeline
```bash
cd ml
pip install -r requirements.txt
jupyter lab notebooks/
```

### 3. Data Collection
1. Install Android app on your device
2. Start logging sensors during drives/walks
3. Export CSV files for training

## 📱 Supported Devices
- **Primary Target**: OnePlus 11R
- **Requirements**: Android 8.0+ (API 26+)
- **Sensors**: IMU (accel, gyro, mag), GPS, Camera (optional)

## 🔬 Research Foundation

Based on proven datasets and methods:
- **IO-VNBD**: Vehicle inertial navigation dataset
- **OxIOD**: Oxford inertial odometry dataset  
- **comma2k19**: Driving dataset with IMU+GPS+camera
- **AVNet**: Learned attitude & velocity with InEKF

## 📈 Development Phases

- [x] **Phase 1**: Sensor logger + data pipeline *(Current)*
- [ ] **Phase 2**: ML speed estimation model
- [ ] **Phase 3**: EKF sensor fusion engine
- [ ] **Phase 4**: Map integration & VIO
- [ ] **Phase 5**: Production optimization

## 🛠️ Technology Stack

**Mobile**: Kotlin, Jetpack Compose, TensorFlow Lite, ARCore, MapLibre
**ML**: Python, PyTorch/TensorFlow, NumPy, Pandas, Jupyter
**Data**: Public datasets (IO-VNBD, OxIOD, comma2k19)

## 📄 License

Apache 2.0 - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Submit pull request

---

**Status**: 🟡 Phase 1 Implementation - Ready for testing
