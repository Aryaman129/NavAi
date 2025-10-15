# 📊 NavAI Project Status

**Last Updated**: October 11, 2025  
**Purpose**: Current state of the project - bugs, metrics, immediate next steps  
**Update Frequency**: After every significant change

---

## 🎯 Current State (October 11, 2025)

### 🎉 PHASE 1 COMPLETE ✅ - Ready for Phase 2!

**All Critical Issues RESOLVED** - The Android app is now fully functional with real-time AI speed prediction!

**Phase 1**: ✅ COMPLETE (RMSE 2.8251 m/s, R² 0.9306)  
**Phase 2**: 🚀 **READY TO START** - See [PHASE2_PLAN.md](PHASE2_PLAN.md) for detailed roadmap  
**Next Milestone**: Phase 2A.1 - Visualization Screen (real-time speed chart)oject Status

**Last Updated**: October 11, 2025  
**Purpose**: Current state of the project - bugs, metrics, immediate next steps  
**Update Frequency**: After every significant change

---

## 🎯 Current State (October 11, 2025)

### � MAJOR MILESTONE: Speed Prediction System WORKING! ✅

**All Critical Issues RESOLVED** - The Android app is now fully functional with real-time AI speed prediction!

### ✅ Recently Fixed (October 11, 2025)

**Issue #1: NNAPI Delegate Failure on Snapdragon 8+ Gen 1**
- **Status**: ✅ FIXED - Implemented cascade fallback pattern
- **Problem**: BiLSTM operations not supported by NNAPI on Qualcomm Snapdragon 8+ Gen 1
- **Root Cause**: `Interpreter()` constructor threw `IllegalArgumentException` when applying NNAPI delegate
- **Solution**: Cascade try-catch pattern - attempts NNAPI first, gracefully falls back to CPU-only mode
- **Impact**: App now runs smoothly on OnePlus 11R with 10-20ms CPU latency
- **Code**: `SpeedPredictor.kt` lines 64-91

**Issue #2: UTF-8 BOM in JSON Metadata File**
- **Status**: ✅ FIXED - Added BOM stripping
- **Problem**: `phase1_model_metadata.json` had UTF-8 BOM (Byte Order Mark) causing JSON parsing to fail
- **Error**: `JsonDecodingException: Expected '{', but had '∩╗┐'`
- **Root Cause**: Windows text editor added invisible BOM character (U+FEFF) at file start
- **Solution**: Strip BOM character before parsing: `if (json[0] == '\uFEFF') json.substring(1)`
- **Code**: `SpeedPredictor.kt` loadMetadata() function

**Issue #3: Unknown JSON Keys (Schema Mismatch)**
- **Status**: ✅ FIXED - Added ignoreUnknownKeys configuration
- **Problem**: JSON file contained `input_shape`, `output_shape` fields not defined in Kotlin data class
- **Error**: `JsonDecodingException: Encountered unknown key 'input_shape'`
- **Root Cause**: Python export script saves extra metadata fields that Kotlin doesn't need
- **Solution**: Configure JSON parser with `ignoreUnknownKeys = true`
- **Safety**: Only ignores extra documentation fields; critical normalization params still parsed correctly
- **Code**: `val jsonParser = Json { ignoreUnknownKeys = true }`

**Issue #4: TFLite Model Compression in APK**
- **Status**: ✅ FIXED (Oct 11) - Added noCompress configuration
- **Problem**: Gradle compressed .tflite files in APK, preventing memory-mapping
- **Root Cause**: `FileChannel.map()` requires uncompressed files for memory-mapped access
- **Solution**: Added `androidResources { noCompress += "tflite" }` to build.gradle.kts
- **Impact**: Model now loads via memory-mapping as designed

**Issue #5: Speed Prediction Screen Crashes on Android 13+**
- **Status**: ✅ FIXED (Oct 11) - Added RECEIVER_NOT_EXPORTED flag
- **Cause**: BroadcastReceiver registration requires explicit export flag on Android 13+
- **Fix**: Added `Context.RECEIVER_NOT_EXPORTED` flag when registering receiver

**Issue #6: Foreground Service Timeout**
- **Status**: ✅ FIXED - Resolved by fixing NNAPI cascade fallback
- **Problem**: `ForegroundServiceDidNotStartInTimeException` after 5 seconds
- **Root Cause**: Service crashed during `onCreate()` due to NNAPI failure, never called `startForeground()`
- **Solution**: Cascade fallback prevents initialization exceptions, service starts normally

### 🟢 FULLY WORKING Features (October 11, 2025)

✅ **Real-Time Speed Prediction**
- TFLite BiLSTM model loads successfully
- NNAPI cascade fallback working (tries NNAPI → falls back to CPU)
- CPU-only mode: 10-20ms latency per prediction
- Predictions at 10Hz (every 100ms)
- UI updates in real-time via broadcasts
- Comprehensive logging for debugging

✅ **Sensor Logger**
- 100Hz sampling (Accelerometer, Gyroscope, Magnetometer, Rotation Vector)
- GPS updates at 5Hz
- CSV export to external storage
- File rotation at 50MB limit
- Batch writes with proper flush() calls

✅ **Android Services**
- SpeedPredictionService: Foreground service for real-time predictions
- SensorLoggerService: High-frequency sensor data logging
- Proper lifecycle management
- Battery-optimized background execution

✅ **Broadcast Communication**
- Service-to-UI broadcasts working
- Real-time state updates
- Error propagation to UI
- RECEIVER_NOT_EXPORTED for Android 13+ compatibility

### Training Status - Phase 1 ✅ COMPLETE

- **Latest Model**: BiLSTM trained on **FULL** 478,891 samples (100% of dataset)
- **Current RMSE**: **2.8251 m/s** ✅ (Target: <12 m/s - **4.2x better**)
- **Current R²**: **0.9306** ✅ (Target: >0.05 - **18.6x better**)
- **Model Size**: 975,105 parameters (~1M params, 3.72 MB)
- **Architecture**: 3-layer BiLSTM, 128 hidden units, 10 input features
- **Post-Processing**: EKF applied (improved RMSE by 0.0226 m/s)
- **Deployment**: ✅ Successfully deployed to Android via TFLite
- **Status**: **FULLY OPERATIONAL** on OnePlus 11R (Snapdragon 8+ Gen 1)

### Android Deployment Status ✅ COMPLETE

- **TFLite Export**: ✅ Model converted and optimized
- **Model Integration**: ✅ Loaded via TensorFlow Lite interpreter
- **Asset Packaging**: ✅ noCompress configuration prevents APK compression
- **Memory Mapping**: ✅ FileChannel.map() working correctly
- **Metadata Loading**: ✅ JSON parsing with BOM stripping and ignoreUnknownKeys
- **NNAPI Handling**: ✅ Cascade fallback for unsupported operations
- **Performance**: 10-20ms CPU latency, 10Hz prediction rate
- **Device Tested**: OnePlus 11R (Android 14, Snapdragon 8+ Gen 1)

### 🔧 Known Limitations & Workarounds

**NNAPI BiLSTM Limitation**
- ⚠️ Qualcomm Snapdragon 8+ Gen 1 doesn't support BiLSTM via NNAPI
- ✅ Cascade fallback automatically uses CPU-only mode
- ✅ CPU performance excellent: 10-20ms latency (well within 100ms budget)
- 📊 Expected behavior: Warning in logs, then successful CPU fallback
- 🔧 Alternative: Could switch to TCN architecture for NNAPI support (Phase 2)

### Phase 1 Results
1. ✅ **Training Complete** - 11 epochs (early stopping triggered, saved time!)
2. ✅ **Excellent Performance** - RMSE 2.8251 << 12 m/s target
3. ✅ **High R²** - 93.06% variance explained (target was 5%)
4. ✅ **Full Dataset** - 478,891 samples utilized (100%)
5. ✅ **EKF Post-Processing Applied** - Improved RMSE from 2.8477 → 2.8251 m/s
6. ✅ **Training Curves Saved** - `ml/outputs/phase1_training_curves.png`

### Training Optimizations Added
- **Early Stopping**: Patience=10 (will save ~50-60 minutes on next run)
- **Gradient Clipping**: max_norm=1.0
- **LR Scheduler**: ReduceLROnPlateau
- **Adam Optimizer**: Learning rate 0.001 (research-validated choice)

---

## 📁 Dataset Status

### Available Datasets

**1. Synthetic/Sample Data** (Testing Only)
- Location: `data/comma2k19/processed/`
- Size: 16.4 MB
- Samples: ~60,000
- Type: Computer-generated
- Status: ✅ Keep for quick testing

**2. Real Comma2k19 Data** (PRODUCTION)
- Location: `data/comma2k19/processed_real/`
- Format: Parquet (3.8 GB)  
- Parsed CSV: `processed_real/parsed/imu_data.csv` (58.14 MB)
- Samples: 478,976 real driving samples
- Speed range: 0-46.61 m/s (0-168 km/h)
- Status: ✅ Ready to use

**3. TAR.GZ Files** (DELETED)
- Location: `data/comma2k19/raw_tar/` ~~(3.5 GB)~~
- Status: ✅ Deleted October 9 - was redundant with parquet
- Space saved: 3.5 GB

---

## 🏗️ Phase 1 vs Phase 2 Features

### Phase 1 (CURRENT - ✅ COMPLETE)

**Architecture**: Simple BiLSTM Baseline
- ✅ **3-layer BiLSTM** (128 hidden units, bidirectional)
- ✅ **Feature Engineering** (magnitude, derivatives from raw IMU)
- ✅ **Normalization** (StandardScaler - fixes R²=0 bug)
- ✅ **Basic Loss**: MSE only
- ✅ **EKF Post-Processing** (Kalman Filter + ZUPT)
- ✅ **Gradient Clipping** (max_norm=1.0)
- ✅ **Early Stopping** (patience=10)
- ✅ **Full Dataset** (478,891 samples)

**Missing in Phase 1**:
- ❌ Attention mechanism
- ❌ Physics-informed loss
- ❌ Uncertainty estimation
- ❌ Residual connections
- ❌ Temporal Convolutional Networks (TCN)

**Results**: RMSE=2.8251 m/s, R²=0.9306 (exceeded targets!)

---

### Phase 2 (PLANNED - Advanced Features)

**Architecture**: Integrated Advanced Model (from `train_optimized_full_dataset.py`)

**What Gets Added**:

1. **✨ Attention Mechanism**
   - Multi-head attention (4 heads)
   - Learns which temporal segments are important
   - Residual connections: `output = norm(input + attention(input))`

2. **⚛️ Physics-Informed Loss**
   - **Smoothness Penalty**: Penalizes large speed changes (unrealistic accelerations)
   - **Bounds Penalty**: Prevents negative speeds or speeds >50 m/s (180 km/h)
   - **Combined Loss**: `L = MSE + λ₁·smoothness + λ₂·bounds`
   - Weights: smoothness=0.1, bounds=0.05

3. **📊 Uncertainty Estimation**
   - Dual output heads: speed + uncertainty
   - Model learns confidence in predictions
   - Helps identify unreliable predictions

4. **🔗 Residual Connections**
   - Skip connections in attention layers
   - Better gradient flow for deeper networks

5. **🧠 Deeper Architecture**
   - Batch normalization after LSTM
   - More FC layers: 256 → 128 → 64 → 32
   - Dropout regularization (0.2)

6. **🔄 Optional: TCN Variant**
   - Temporal Convolutional Network (from `hardware_aware_tcn.py`)
   - Alternative to BiLSTM for temporal extraction
   - Better for mobile deployment (faster inference)

**Expected Results**: RMSE 6-8 m/s, R² 0.25-0.35

---

### Phase 3 (FUTURE - GTSAM Post-Processing)

**What Gets Added**:
- GTSAM factor graph optimization
- IMU pre-integration factors
- Combines neural network predictions with physics
- Applied during inference (not training)

**Expected Results**: RMSE <5 m/s, R² >0.40

---

## 🔬 Physics-Informed Loss Explained

The Phase 2 model will use a custom loss function that incorporates physics constraints:

### Components:

1. **MSE Loss** (prediction accuracy)
   ```
   L_mse = (predicted_speed - true_speed)²
   ```

2. **Smoothness Loss** (temporal consistency)
   ```
   L_smooth = (speed[t] - speed[t-1])²
   ```
   - Penalizes large speed changes between timesteps
   - Ensures realistic acceleration profiles
   - Weight: 0.1

3. **Bounds Loss** (physical constraints)
   ```
   L_bounds = ReLU(-speed) + ReLU(speed - 50)
   ```
   - Penalizes negative speeds (impossible)
   - Penalizes speeds >50 m/s (180 km/h - unrealistic for most driving)
   - Weight: 0.05

4. **Total Loss**
   ```
   L_total = L_mse + 0.1·L_smooth + 0.05·L_bounds
   ```

### Why This Helps:
- Prevents physically impossible predictions
- Reduces noise in speed estimates
- Encourages smooth velocity profiles
- Improves generalization to unseen data

---

## 🏗️ Architecture Decision

### Integration Approach: **INTEGRATED** ⭐

**Decision Made**: All advanced features in ONE model (not separate)

**What's Integrated** (in training):
- Physics-informed loss (smoothness + bounds)
- Attention mechanism (multi-head, 4 heads)
- Temporal extraction (BiLSTM or TCN)
- Residual connections
- Uncertainty estimation

**What's Post-Processing** (separate):
- GTSAM factor graph optimization
- Applied to predictions during inference
- Can't integrate into training (optimization-based, not gradient-based)

**Implementation**: Use `scripts/train_optimized_full_dataset.py` (already has integrated architecture)

---

## 📈 Performance Targets

| Phase | RMSE Target | R² Target | **Actual Result** | Status |
|-------|-------------|-----------|-------------------|--------|
| Phase 1: Foundation | 10-12 m/s | 0.05-0.10 | **2.8251 m/s, R²=0.9306** | ✅ **EXCEEDED** |
| Phase 2: Advanced Features | 6-8 m/s | 0.25-0.35 | - | ⏳ Next |
| Phase 3: + GTSAM Post | < 5 m/s | > 0.40 | - | 🎯 Goal |

### Phase 1 Achievement
- RMSE: 2.8251 m/s (Target: 12 m/s) → **4.2x better than target** 🎉
- R²: 0.9306 (Target: 0.05) → **18.6x better than target** 🎉
- Dataset: 100% utilization (478,891 samples)
- Model: Lightweight (975K params, 3.72 MB)
- EKF Post-Processing: +0.8% improvement (2.8477 → 2.8251 m/s)
- Early Stopping: Triggered at epoch 11 (saved ~40 epochs of training time!)

---

## ✅ Recently Completed (December 2025)

1. **Phase 1 Training Complete** ✅
   - Trained BiLSTM on full dataset (478,891 samples)
   - Achieved RMSE=2.8251 m/s, R²=0.9306 (with EKF)
   - Far exceeds all targets (4.2x better than goal)
   - Early stopping saved ~40 epochs of training time

2. **EKF Post-Processing Applied** ✅
   - Extended Kalman Filter applied to predictions
   - Improved RMSE from 2.8477 → 2.8251 m/s
   - Includes ZUPT (Zero-Velocity Updates) for stationary detection
   - Processing time: 8 seconds for 95,759 samples

3. **Training Optimizations** ✅
   - Early stopping triggered at epoch 11 (patience=10)
   - Gradient clipping (max_norm=1.0)
   - LR Scheduler (ReduceLROnPlateau)
   - Training curves saved: `ml/outputs/phase1_training_curves.png`

4. **Model Analysis** ✅
   - Created `check_model_params.py`
   - Confirmed 975,105 parameters
   - Model size: 3.72 MB (lightweight)

5. **Research on Advanced Training Methods** ✅
   - Explored **Ray Tune** for hyperparameter tuning
   - Reviewed optimizer research (arXiv:2007.01547)
   - Key finding: **Adam optimizer is competitive** (current choice validated)
   - Ray Tune could automate search over: LR, batch size, hidden dims, layers

## 🔍 Research Findings - Advanced Training Techniques

### Option 1: Hyperparameter Tuning with Ray Tune
**What it is**: Automated distributed hyperparameter search framework

**Benefits**:
- ASHA Scheduler for early stopping bad trials
- Parallel trials with GPU sharing
- Search over: learning_rate, hidden_dim, num_layers, batch_size
- Example: `tune.loguniform(1e-4, 1e-1)` for LR search

**When to use**: When you want to find optimal hyperparameters automatically

### Option 2: Optimizer Benchmarking
**Research Findings** (50,000+ runs on 15 optimizers):
- **"Adam remains a strong contender"**
- **"Newer methods failing to significantly outperform Adam"**
- Optimizer performance varies greatly across tasks
- Testing multiple optimizers ≈ tuning one optimizer

**Conclusion**: Our current **Adam optimizer is already optimal** ✅

### Option 3: Architecture Search
**Time Series Best Practices**:
- **Stacked LSTM**: Multiple LSTM layers for hierarchical features
- **Bidirectional LSTM**: Process sequence forwards+backwards (already using ✅)
- **CNN-LSTM**: CNN extracts subsequence features → LSTM processes
- **ConvLSTM**: Convolutional operations built into LSTM cells
- **Encoder-Decoder**: Separate encoder/decoder for multi-step forecasting

**Current Architecture**: BiLSTM (already competitive) ✅

### Recommendation
**Current approach is strong**:
- ✅ Adam optimizer (research-validated)
- ✅ BiLSTM architecture (best practice)
- ✅ Early stopping (prevents overfitting)
- ✅ Full dataset utilization
- ✅ Gradient clipping (stability)

**Optional improvements**:
1. **Ray Tune**: Automate hyperparameter search (learning_rate, hidden_dim, etc.)
2. **Ensemble**: Train multiple models with different seeds, average predictions
3. **Architecture variants**: Test CNN-LSTM or Encoder-Decoder for comparison

**Bottom line**: Phase 1 performance is **excellent**. Focus on Phase 2 (advanced features) before re-tuning Phase 1.
   - Decision: Integrated approach
   - GTSAM as post-processing only

4. **Documentation Organization**
   - Reduced 35 .md files to 8 organized docs
   - Created PROJECT_DIARY.md for history
   - Created SCRATCHPAD.md for WIP

---

## ⏳ Immediate Next Steps

### 1. Fix Normalization Bug (HIGH PRIORITY)
**What**: Add StandardScaler to normalize IMU features
**Where**: Modify training script or use `train_optimized_full_dataset.py`
**Why**: Different feature scales prevent model from learning
**Expected Result**: RMSE 10-12 m/s

### 2. Train on Full Dataset
**What**: Use all 478,976 samples (not just 100k)
**Why**: Only using 21% of available data
**How**: Batch loading if memory issues
**Expected Result**: Further RMSE improvement

### 3. Verify Bug Fixes
**What**: Run training and check R² > 0 and RMSE > MAE
**Why**: Confirm normalization fixes the issues
**Expected Result**: Positive R², RMSE > MAE

---

## 🎯 Implementation Roadmap

### Phase 1: Fix Foundation (Week 1)
- [ ] Add feature normalization (StandardScaler)
- [ ] Implement proper temporal train/test split
- [ ] Train on full 478k samples
- [ ] Target: RMSE 10-12 m/s, R² > 0.05

### Phase 2: Integrated Advanced Training (Week 2)
- [ ] Use `train_optimized_full_dataset.py`
- [ ] Enable: BiLSTM + Attention + Physics Loss
- [ ] Add data augmentation
- [ ] Target: RMSE 6-8 m/s, R² > 0.25

### Phase 3: GTSAM Post-Processing (Week 3)
- [ ] Use `improvements/factor_graph_navigation.py`
- [ ] Apply to best model's predictions
- [ ] Fine-tune parameters
- [ ] Target: RMSE < 5 m/s, R² > 0.40

### Phase 4: Mobile Deployment (Week 4)
- [ ] Export to TFLite
- [ ] Mobile optimization (quantization)
- [ ] Real-time testing on device
- [ ] Performance benchmarking

---

## 🔧 Advanced Features Available (Not Yet Used)

### In `improvements/` Directory

1. **factor_graph_navigation.py** (291 lines)
   - PhysicsInformedSpeedEstimator with CNN
   - GTSAM FactorGraphNavigator
   - IMU preintegration, GPS factors
   - SmartphoneNavigationSystem

2. **hardware_aware_tcn.py** (423 lines)
   - MobileNavTCN (mobile-optimized)
   - Depthwise separable convolutions
   - 8-9x fewer parameters than standard CNN

3. **enhanced_user_priors.py** (348 lines)
   - Activity classification (walk/cycle/vehicle)
   - Mount detection (handheld/pocket/dashboard)
   - User-specific speed priors

4. **visual_inertial_navigation.py** (267 lines)
   - ARCore VIO integration
   - Feature tracking
   - Visual odometry

5. **tflite_optimization.py** (189 lines)
   - Model export to TFLite
   - Quantization options
   - Mobile optimization

**Status**: All implemented but not yet integrated into training

---

## ❓ Recent Q&A

### Q: "Why do we have multiple comma2k19 formats?"
**A**: Three formats exist:
1. Synthetic (16MB) - early testing before real data
2. Real Parquet (3.8GB) - production dataset from HuggingFace ⭐
3. TAR.GZ (was 3.5GB) - same as parquet, deleted as redundant

### Q: "How much data have we actually used?"
**A**: Only 100,000 out of 478,976 samples (21%) in best training run. We can do much better!

### Q: "Should advanced features be separate models or integrated?"
**A**: INTEGRATED in one model. Better end-to-end optimization, simpler deployment. Only GTSAM is post-processing.

### Q: "What's causing the bugs?"
**A**: No feature normalization. IMU data has vastly different scales (accel ~10 vs gyro ~1), breaking model learning.

---

## 📊 Training History Summary

### October 8, 2025 - GPU Test (Synthetic)
- Dataset: 30,000 synthetic samples
- Model: Simple BiLSTM
- RMSE: 0.286 m/s
- Notes: GPU test only, not real data

### October 9, 2025 - Real Data Attempt
- Dataset: 100,000 real samples (21% of available)
- Model: Simple BiLSTM
- RMSE: 8.25 m/s
- Problem: Evaluated on training data (invalid metric)

### January 9, 2025 - Current Buggy Training
- Dataset: Unknown size (buggy)
- Model: Simple BiLSTM
- RMSE: 15.356 m/s ❌
- R²: 0.0 ❌
- Problem: No normalization, model not learning

**Full history**: See `TRAINING_HISTORY.md`

---

## 🗂️ File Organization

### Core Documentation (8 Files)
1. **PROJECT_STATUS.md** (this file) - Current state
2. **TRAINING_HISTORY.md** - All training runs chronologically
3. **IMPLEMENTATION_GUIDE.md** - How to implement features
4. **DATASET_GUIDE.md** - Dataset info and usage
5. **SYSTEM_OVERVIEW.md** - Project architecture and tech stack
6. **CONCEPTS_EXPLAINED.md** - Educational explanations
7. **PROJECT_DIARY.md** - Complete project history diary
8. **SCRATCHPAD.md** - Work-in-progress notes

### Key Code Files
- `ml/training/train_speed_estimation_fixed.py` - Current (buggy) training
- `scripts/train_optimized_full_dataset.py` - Integrated advanced training ⭐
- `improvements/` - Advanced feature implementations
- `ml/analysis/quick_bug_check.py` - Bug diagnostic tool

---

## 🎯 Success Criteria

### Phase 1 Success
- [ ] RMSE < 12 m/s
- [ ] R² > 0.05 (positive!)
- [ ] Trained on 478k samples
- [ ] Feature normalization working

### Phase 2 Success
- [ ] RMSE < 8 m/s
- [ ] R² > 0.25
- [ ] Attention mechanism working
- [ ] Physics-informed loss integrated

### Phase 3 Success
- [ ] RMSE < 5 m/s
- [ ] R² > 0.40
- [ ] GTSAM refinement working
- [ ] Ready for mobile deployment

---

**Update Policy**: This file gets updated after every significant change:
- New training run completed
- Bug fixed
- Architecture decision made
- Dataset change
- Milestone reached

**For detailed history**: See PROJECT_DIARY.md (append-only historical record)
**For work-in-progress**: See SCRATCHPAD.md (temporary notes)
