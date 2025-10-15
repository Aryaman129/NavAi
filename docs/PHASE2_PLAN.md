# 🚀 NavAI Phase 2 Implementation Plan

**Created**: October 11, 2025  
**Status**: Ready to Begin  
**Previous Phase**: Phase 1 Complete ✅ (RMSE 2.8251 m/s, R² 0.9306)

---

## 📋 Executive Summary

### Phase 1 Review

Phase 1 **EXCEEDED** all targets:
- ✅ RMSE: 2.8251 m/s (target: <12 m/s) - **4.2x better**
- ✅ R²: 0.9306 (target: >0.05) - **18.6x better**
- ✅ Model size: 3.72 MB (target: <5 MB) - **26% better**
- ✅ Inference: 10-20ms (target: <50ms) - **2-5x better**
- ✅ Android app: Fully operational on OnePlus 11R
- ✅ All 6 critical bugs fixed (NNAPI, BOM, JSON keys, etc.)

### Phase 2 Strategy Shift

**Important Insight**: Phase 1 already exceeded Phase 2's original accuracy targets (RMSE 6-8 m/s, R² 0.25-0.35).

Therefore, **Phase 2 is NOT about improving accuracy**. Instead, it's about:

1. **User Experience**: Visualization, file sharing, settings
2. **Performance**: TCN model for NNAPI support (2-3x faster)
3. **Interpretability**: Attention mechanism, uncertainty estimation
4. **Robustness**: Physics-informed loss, confidence scores

**Goal**: Maintain Phase 1's excellent accuracy while adding advanced features that make the system more usable, faster, and interpretable.

---

## 🎯 Phase 2 Overview

### Three Sub-Phases

**Phase 2A: UI Enhancements** (Week 1 - Quick Wins)
- Visualization screen (speed chart)
- File sharing UI
- Settings screen
- **Duration**: 3-4 days
- **Risk**: Low
- **Value**: High (immediate user benefit)

**Phase 2B: TCN Model Variant** (Week 2 - Performance)
- Hardware-aware Temporal Convolutional Network
- NNAPI-compatible architecture
- 2-3x faster inference
- **Duration**: 4-5 days
- **Risk**: Medium (model accuracy must match Phase 1)
- **Value**: High (solves NNAPI limitation)

**Phase 2C: Advanced ML Features** (Week 3-4 - Research)
- Attention mechanism
- Physics-informed loss
- Uncertainty estimation
- **Duration**: 6-8 days
- **Risk**: Medium-High (research-oriented)
- **Value**: Medium (interpretability, not user-facing)

---

## 📊 Phase 2A: UI Enhancements (Week 1)

### Motivation

Phase 1 works perfectly, but users need:
- **Visual feedback**: See predictions over time (validate system)
- **Data export**: Share CSV files easily (no manual file copying)
- **Customization**: Control sampling rates, file sizes, model selection

### Task 2A.1: Visualization Screen 📈

**Goal**: Real-time speed prediction chart

**What to Build**:
- Line chart showing speed predictions over time (last 100 points = 10 seconds)
- X-axis: Time (seconds)
- Y-axis: Speed (m/s and km/h)
- Color-coded: Green (low speed), Yellow (medium), Red (high)
- Live updates every 100ms

**Implementation**:

1. **Add MPAndroidChart Dependency**
   - File: `mobile/app/build.gradle.kts`
   - Add: `implementation("com.github.PhilJay:MPAndroidChart:v3.1.0")`
   - Add: `maven { url = uri("https://jitpack.io") }` to repositories

2. **Create VisualizationScreen.kt**
   - File: `mobile/app/src/main/java/com/navai/logger/ui/screens/VisualizationScreen.kt`
   - Composable with `AndroidView` wrapping `LineChart`
   - Subscribe to speed prediction broadcasts
   - Maintain `CircularBuffer<SpeedDataPoint>(100)` for last 10 seconds

3. **Create Data Models**
   - File: `mobile/app/src/main/java/com/navai/logger/data/SpeedDataPoint.kt`
   ```kotlin
   data class SpeedDataPoint(
       val timestamp: Long,        // milliseconds since epoch
       val speed: Float,           // m/s
       val uncertainty: Float? = null  // Phase 2C addition
   )
   ```

4. **Update SpeedPredictionService**
   - File: `mobile/app/src/main/java/com/navai/logger/service/SpeedPredictionService.kt`
   - Add: `private val recentPredictions = CircularBuffer<SpeedDataPoint>(100)`
   - Store each prediction with timestamp
   - Broadcast predictions for visualization

5. **Add Navigation**
   - File: `mobile/app/src/main/java/com/navai/logger/MainActivity.kt`
   - Add "Visualization" button to main screen
   - Navigate to `VisualizationScreen`

**Success Criteria**:
- ✅ Chart displays real-time speed predictions
- ✅ Smooth 60fps updates (no UI lag)
- ✅ Last 100 points (10 seconds) visible
- ✅ Dual Y-axis: m/s and km/h
- ✅ Color-coded speed ranges

**Estimated Effort**: 1 day

---

### Task 2A.2: File Sharing UI 📤

**Goal**: Share CSV files via Android share sheet

**What to Build**:
- "Share Files" button in MainActivity
- File picker to select CSV files
- Android Intent to share via email, WhatsApp, Drive, etc.

**Implementation**:

1. **Create FileShareHelper.kt**
   - File: `mobile/app/src/main/java/com/navai/logger/utils/FileShareHelper.kt`
   ```kotlin
   object FileShareHelper {
       fun shareCSVFiles(context: Context, files: List<File>) {
           val uris = files.map { 
               FileProvider.getUriForFile(context, "${context.packageName}.provider", it)
           }
           
           val shareIntent = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
               type = "text/csv"
               putParcelableArrayListExtra(Intent.EXTRA_STREAM, ArrayList(uris))
               addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
           }
           
           context.startActivity(Intent.createChooser(shareIntent, "Share CSV Files"))
       }
   }
   ```

2. **Update MainActivity.kt**
   - Add "Share Files" button
   - List available CSV files from external storage
   - Call `FileShareHelper.shareCSVFiles()` on click

3. **Update AndroidManifest.xml**
   - Add `FileProvider` configuration if not already present
   - Ensure READ_EXTERNAL_STORAGE permission

**Success Criteria**:
- ✅ Users can select CSV files to share
- ✅ Android share sheet appears with all apps
- ✅ Files successfully sent via WhatsApp, email, Drive
- ✅ No permission errors

**Estimated Effort**: 0.5 days

---

### Task 2A.3: Settings Screen ⚙️

**Goal**: User-configurable app settings

**What to Build**:
- Settings screen with preferences:
  - Sampling rate: 50Hz, 100Hz, 200Hz (default: 100Hz)
  - File size limit: 10MB, 50MB, 100MB (default: 50MB)
  - Model selection: BiLSTM, TCN (Phase 2B)
  - Prediction rate: 5Hz, 10Hz, 20Hz (default: 10Hz)
  - Enable/disable EKF post-processing

**Implementation**:

1. **Create UserPreferences.kt**
   - File: `mobile/app/src/main/java/com/navai/logger/data/UserPreferences.kt`
   ```kotlin
   class UserPreferences(context: Context) {
       private val prefs = context.getSharedPreferences("navai_settings", Context.MODE_PRIVATE)
       
       var samplingRateHz: Int
           get() = prefs.getInt("sampling_rate_hz", 100)
           set(value) = prefs.edit().putInt("sampling_rate_hz", value).apply()
       
       var fileSizeLimitMB: Int
           get() = prefs.getInt("file_size_limit_mb", 50)
           set(value) = prefs.edit().putInt("file_size_limit_mb", value).apply()
       
       var selectedModel: String
           get() = prefs.getString("selected_model", "bilstm") ?: "bilstm"
           set(value) = prefs.edit().putString("selected_model", value).apply()
       
       var predictionRateHz: Int
           get() = prefs.getInt("prediction_rate_hz", 10)
           set(value) = prefs.edit().putInt("prediction_rate_hz", value).apply()
   }
   ```

2. **Create SettingsScreen.kt**
   - File: `mobile/app/src/main/java/com/navai/logger/ui/screens/SettingsScreen.kt`
   - Compose UI with:
     - `DropdownMenu` for sampling rate
     - `DropdownMenu` for file size limit
     - `RadioButton` for model selection
     - `Slider` for prediction rate
     - "Save" button to persist changes

3. **Update Services to Read Settings**
   - File: `SensorLoggerService.kt`
     - Read `samplingRateHz` on service start
     - Read `fileSizeLimitMB` for file rotation
   - File: `SpeedPredictionService.kt`
     - Read `selectedModel` to load BiLSTM or TCN
     - Read `predictionRateHz` to adjust prediction interval

4. **Add Navigation**
   - File: `MainActivity.kt`
   - Add "Settings" button with gear icon
   - Navigate to `SettingsScreen`

**Success Criteria**:
- ✅ Settings persist across app restarts
- ✅ Changing sampling rate updates sensor manager immediately
- ✅ Changing file size limit applies to new files
- ✅ Model selection switches between BiLSTM/TCN (Phase 2B)
- ✅ UI follows Material 3 design

**Estimated Effort**: 1.5 days

---

### Phase 2A Summary

**Total Duration**: 3-4 days  
**Dependencies**: None (builds on Phase 1)  
**Risk**: Low (pure UI/UX improvements)  
**Value**: High (immediate user benefit)

**Deliverables**:
- ✅ Visualization screen with real-time speed chart
- ✅ File sharing via Android Intent
- ✅ Settings screen with 5 configurable preferences
- ✅ Updated documentation

**Testing Plan**:
1. Manual testing on OnePlus 11R
2. Verify chart performance (60fps)
3. Test file sharing with multiple apps
4. Verify settings persistence

---

## 🧠 Phase 2B: TCN Model Variant (Week 2)

### Motivation

**Problem**: BiLSTM not supported by NNAPI on Snapdragon 8+ Gen 1
- Current: 10-20ms CPU inference (acceptable but not optimal)
- Target: 5-10ms NNAPI inference (2-3x faster)
- Benefit: Lower power consumption, faster predictions

**Solution**: Temporal Convolutional Network (TCN)
- NNAPI-compatible operations (Conv1D, BatchNorm, ReLU)
- Similar accuracy to BiLSTM (expected RMSE ≤ 4 m/s)
- Faster inference on mobile GPUs/NPUs

### Architecture

**Based on**: `improvements/hardware_aware_tcn.py` (already scaffolded)

**Key Components**:
1. **Depthwise Separable Convolutions**: 8-9x fewer parameters
2. **Temporal Blocks**: Causal convolutions with residual connections
3. **Dilation**: Exponential receptive field growth (1, 2, 4, 8, 16...)
4. **Batch Normalization**: Faster convergence, better NNAPI support

**Architecture Details**:
- Input: (batch, 100, 10) - 100 timesteps, 10 features
- 5 Temporal Blocks: [128, 128, 256, 256, 512] channels
- Dilation: [1, 2, 4, 8, 16]
- Receptive field: ~50 timesteps (0.5 seconds at 100Hz)
- Output: (batch, 1) - speed prediction

### Task 2B.1: Implement TCN Architecture

**File**: `ml/models/tcn_speed_estimator.py`

**Implementation**:
```python
"""
Hardware-Aware TCN for Speed Estimation
Optimized for mobile NNAPI deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv1d(nn.Module):
    """Efficient convolution for mobile deployment"""
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1):
        super().__init__()
        
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=padding, stride=stride, dilation=dilation,
            groups=in_channels, bias=False
        )
        
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class TemporalBlock(nn.Module):
    """TCN block with residual connections"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = DepthwiseSeparableConv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        
        self.conv2 = DepthwiseSeparableConv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        residual = x if self.residual is None else self.residual(x)
        
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        
        # Trim to match residual (causal convolution)
        out = out[:, :, :residual.size(2)]
        
        return self.relu(out + residual)


class TCNSpeedEstimator(nn.Module):
    """Temporal Convolutional Network for speed estimation"""
    
    def __init__(self, input_dim=10, hidden_channels=[128, 128, 256, 256, 512], kernel_size=3):
        super().__init__()
        
        layers = []
        in_channels = input_dim
        
        for i, out_channels in enumerate(hidden_channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation))
            in_channels = out_channels
        
        self.tcn = nn.Sequential(*layers)
        
        # Global average pooling + FC
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # TCN
        x = self.tcn(x)  # (batch, channels, seq_len)
        
        # Global average pooling
        x = x.mean(dim=2)  # (batch, channels)
        
        # Speed prediction
        speed = self.fc(x)  # (batch, 1)
        
        return speed
```

**Success Criteria**:
- ✅ Model compiles without errors
- ✅ Forward pass produces (batch, 1) output
- ✅ Parameter count < 1M (similar to BiLSTM)

**Estimated Effort**: 1 day

---

### Task 2B.2: Train TCN on Comma2k19

**File**: `ml/training/train_phase2_tcn.py`

**Training Configuration**:
- Dataset: Comma2k19 (478,891 samples) - same as Phase 1
- Batch size: 64
- Epochs: 30 (early stopping patience=10)
- Optimizer: Adam (lr=0.001)
- Loss: MSE
- Hardware: GPU (NVIDIA RTX 3050 Ti)

**Expected Results**:
- RMSE: 3-4 m/s (slightly worse than Phase 1 BiLSTM, but acceptable)
- Training time: 3-4 hours
- Model size: 1-2 MB

**Implementation**:
```python
# Load preprocessed data (same as Phase 1)
X_train, y_train = load_comma2k19_data()

# Create TCN model
model = TCNSpeedEstimator(
    input_dim=10,
    hidden_channels=[128, 128, 256, 256, 512],
    kernel_size=3
)

# Train with early stopping
train_model(
    model, X_train, y_train,
    epochs=30, batch_size=64,
    early_stopping_patience=10
)

# Evaluate
rmse = evaluate_model(model, X_test, y_test)
print(f"TCN RMSE: {rmse:.2f} m/s")
```

**Success Criteria**:
- ✅ RMSE ≤ 4 m/s (close to Phase 1)
- ✅ R² > 0.80 (good variance explained)
- ✅ Training converges (no NaN losses)

**Estimated Effort**: 0.5 days (mostly waiting for training)

---

### Task 2B.3: Export TCN to TFLite with NNAPI

**File**: `ml/export_phase2_tcn_tflite.py`

**Export Configuration**:
- Format: TensorFlow Lite
- Quantization: None (FP32 for accuracy)
- NNAPI: Enabled and verified
- Metadata: JSON with normalization params

**Implementation**:
```python
# Convert PyTorch → ONNX → TensorFlow → TFLite
import torch
import tf2onnx
import tensorflow as tf

# Export to ONNX
torch.onnx.export(
    model, dummy_input, "tcn_model.onnx",
    input_names=['input'], output_names=['output']
)

# Convert to TensorFlow
tf_model = tf2onnx.convert.from_onnx("tcn_model.onnx")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open("phase2_tcn_model.tflite", "wb") as f:
    f.write(tflite_model)

# Verify NNAPI compatibility
verify_nnapi_compatibility("phase2_tcn_model.tflite")
```

**NNAPI Verification**:
- Use `nnapi_verify.py` to check all operations are NNAPI-supported
- Test on OnePlus 11R with NNAPI delegate
- Measure inference latency (target: 5-10ms)

**Success Criteria**:
- ✅ TFLite export successful
- ✅ All operations NNAPI-compatible
- ✅ Inference 2-3x faster than BiLSTM (5-10ms)
- ✅ Accuracy maintained (RMSE ≤ 4 m/s on Android)

**Estimated Effort**: 1 day

---

### Task 2B.4: Integrate TCN into Android App

**Files to Modify**:
- `SpeedPredictor.kt` - Add TCN model loading
- `SpeedPredictionService.kt` - Support model switching
- `SettingsScreen.kt` - Add model selection (Phase 2A.3)

**Implementation**:

1. **Update SpeedPredictor.kt**
   ```kotlin
   class SpeedPredictor(context: Context) {
       private var currentModel: String = "bilstm"  // or "tcn"
       private lateinit var interpreter: Interpreter
       
       fun loadModel(modelType: String) {
           currentModel = modelType
           
           val modelFile = when (modelType) {
               "bilstm" -> "phase1_model.tflite"
               "tcn" -> "phase2_tcn_model.tflite"
               else -> "phase1_model.tflite"
           }
           
           // Load with NNAPI for TCN
           val options = Interpreter.Options()
           if (modelType == "tcn") {
               options.setUseNNAPI(true)
           }
           
           interpreter = Interpreter(loadModelFile(context, modelFile), options)
       }
   }
   ```

2. **Update SpeedPredictionService.kt**
   - Read `selectedModel` from UserPreferences
   - Call `speedPredictor.loadModel(selectedModel)` on service start
   - Handle model loading failures gracefully

3. **Test Model Switching**
   - Open Settings → Select TCN model
   - Restart service
   - Verify 5-10ms inference latency in logcat
   - Verify predictions are accurate

**Success Criteria**:
- ✅ App supports both BiLSTM and TCN models
- ✅ Users can switch models in settings
- ✅ TCN model runs 2-3x faster than BiLSTM
- ✅ Graceful fallback if NNAPI fails

**Estimated Effort**: 1.5 days

---

### Phase 2B Summary

**Total Duration**: 4-5 days  
**Dependencies**: Phase 2A.3 (Settings screen)  
**Risk**: Medium (TCN accuracy must match BiLSTM)  
**Value**: High (solves NNAPI limitation, faster inference)

**Deliverables**:
- ✅ TCN model architecture implemented
- ✅ TCN trained on Comma2k19 (RMSE ≤ 4 m/s)
- ✅ TFLite export with NNAPI support verified
- ✅ Android app supports model switching
- ✅ 2-3x faster inference on OnePlus 11R

**Testing Plan**:
1. Benchmark TCN vs BiLSTM inference latency
2. Validate TCN accuracy on test set
3. Test NNAPI delegate on OnePlus 11R
4. Verify graceful fallback if NNAPI fails

---

## 🔬 Phase 2C: Advanced ML Features (Week 3-4)

### Motivation

Phase 1 and 2B provide excellent **accuracy** and **performance**. Phase 2C adds:

1. **Interpretability**: Which time segments matter? (Attention)
2. **Safety**: How confident is the prediction? (Uncertainty)
3. **Robustness**: Prevent physically implausible predictions (Physics loss)

These features are important for:
- Debugging model failures
- Safety-critical applications (autonomous vehicles)
- Research insights (understanding what the model learns)

### Task 2C.1: Attention Mechanism

**Goal**: Add multi-head attention to BiLSTM model

**Architecture**:
- BiLSTM (3 layers, 128 hidden units)
- Multi-head attention (4 heads, 256-dim)
- Residual connection: `output = norm(lstm_out + attention(lstm_out))`
- FC layers: 256 → 128 → 64 → 32 → 1

**File**: `ml/models/attention_bilstm.py`

**Implementation**:
```python
class AttentionBiLSTM(nn.Module):
    """BiLSTM with multi-head attention"""
    
    def __init__(self, input_dim=10, hidden_dim=128, num_layers=3, num_heads=4):
        super().__init__()
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        
        # Attention
        lstm_output_dim = hidden_dim * 2  # bidirectional
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer normalization for residual
        self.norm = nn.LayerNorm(lstm_output_dim)
        
        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention with residual
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        x = self.norm(lstm_out + attn_out)  # residual connection
        
        # Use last timestep
        x = x[:, -1, :]  # (batch, hidden*2)
        
        # Speed prediction
        speed = self.fc(x)  # (batch, 1)
        
        return speed, attn_weights
```

**Training**:
- Same as Phase 1 (Comma2k19, MSE loss, Adam optimizer)
- Expected RMSE: 2.5-3.0 m/s (similar to Phase 1)

**Visualization**:
- Use Phase 2A.1 visualization screen
- Add heatmap showing attention weights over 1-second window
- Color: Red (high attention), Blue (low attention)

**Success Criteria**:
- ✅ Attention mechanism trains successfully
- ✅ RMSE ≤ 3 m/s (maintains Phase 1 accuracy)
- ✅ Attention weights visualized in UI
- ✅ Attention focuses on relevant time segments (e.g., recent data)

**Estimated Effort**: 2 days

---

### Task 2C.2: Physics-Informed Loss

**Goal**: Prevent unrealistic predictions

**Physics Constraints**:
1. **Smoothness**: Speed shouldn't change too quickly
   - Max acceleration: 4 m/s² (typical car)
   - Penalty: Large speed jumps between timesteps

2. **Bounds**: Speed must be realistic
   - Min: 0 m/s (can't go backward)
   - Max: 50 m/s (180 km/h, typical highway limit)

3. **Temporal Consistency**: Predictions should be smooth

**File**: `ml/models/physics_loss.py`

**Implementation**:
```python
class PhysicsInformedLoss(nn.Module):
    """Combined MSE + Physics constraints"""
    
    def __init__(self, lambda_smoothness=0.1, lambda_bounds=0.05):
        super().__init__()
        self.lambda_smoothness = lambda_smoothness
        self.lambda_bounds = lambda_bounds
    
    def forward(self, predictions, targets, dt=0.01):
        # MSE loss (accuracy)
        mse_loss = F.mse_loss(predictions, targets)
        
        # Smoothness loss (penalize large accelerations)
        if predictions.size(0) > 1:
            speed_diff = predictions[1:] - predictions[:-1]
            acceleration = speed_diff / dt
            smoothness_loss = torch.mean(torch.abs(acceleration))
        else:
            smoothness_loss = 0.0
        
        # Bounds loss (penalize negative or excessive speeds)
        bounds_loss = torch.mean(F.relu(-predictions))  # negative speeds
        bounds_loss += torch.mean(F.relu(predictions - 50))  # > 50 m/s
        
        # Combined loss
        total_loss = (
            mse_loss + 
            self.lambda_smoothness * smoothness_loss +
            self.lambda_bounds * bounds_loss
        )
        
        return total_loss, {
            'mse': mse_loss.item(),
            'smoothness': smoothness_loss if isinstance(smoothness_loss, float) else smoothness_loss.item(),
            'bounds': bounds_loss.item()
        }
```

**Training**:
- Use with AttentionBiLSTM model
- Hyperparameters: λ_smoothness=0.1, λ_bounds=0.05
- Monitor loss components separately

**Success Criteria**:
- ✅ No negative speed predictions
- ✅ No predictions > 50 m/s
- ✅ Smooth predictions (no sudden jumps)
- ✅ RMSE maintains ≤ 3 m/s

**Estimated Effort**: 1 day

---

### Task 2C.3: Uncertainty Estimation

**Goal**: Model predicts speed + confidence score

**Approach**: Dual output heads
- Head 1: Speed prediction (regression)
- Head 2: Uncertainty estimate (regression)

**File**: `ml/models/uncertainty_bilstm.py`

**Implementation**:
```python
class UncertaintyBiLSTM(nn.Module):
    """BiLSTM with uncertainty estimation"""
    
    def __init__(self, input_dim=10, hidden_dim=128, num_layers=3):
        super().__init__()
        
        # Shared BiLSTM
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        
        # Shared FC layers
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Dual output heads
        self.speed_head = nn.Linear(32, 1)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Softplus()  # ensures positive uncertainty
        )
    
    def forward(self, x):
        # BiLSTM
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # last timestep
        
        # Shared features
        features = self.shared_fc(x)
        
        # Dual outputs
        speed = self.speed_head(features)
        uncertainty = self.uncertainty_head(features)
        
        return speed, uncertainty
```

**Loss Function**: Negative Log Likelihood
```python
def uncertainty_loss(speed_pred, uncertainty_pred, speed_true):
    """NLL loss for uncertainty estimation"""
    # Assume Gaussian distribution: N(speed_pred, uncertainty_pred^2)
    nll = 0.5 * torch.log(2 * np.pi * uncertainty_pred**2)
    nll += 0.5 * ((speed_true - speed_pred)**2 / uncertainty_pred**2)
    return torch.mean(nll)
```

**Visualization**:
- Update Phase 2A.1 visualization screen
- Show error bars: `speed ± uncertainty`
- Color: Green (low uncertainty), Red (high uncertainty)

**Success Criteria**:
- ✅ Uncertainty correlates with prediction error
- ✅ High uncertainty during rapid speed changes
- ✅ Low uncertainty during steady speed
- ✅ RMSE ≤ 3 m/s

**Estimated Effort**: 2 days

---

### Task 2C.4: Integrated Training and Evaluation

**Goal**: Train full Phase 2C model with all features

**Architecture**: AttentionBiLSTM + PhysicsLoss + Uncertainty

**File**: `ml/training/train_phase2c_integrated.py`

**Training Configuration**:
- Model: AttentionBiLSTM with uncertainty
- Loss: PhysicsInformedLoss + UncertaintyLoss
- Dataset: Comma2k19 (478,891 samples)
- Epochs: 30 (early stopping patience=10)
- Optimizer: Adam (lr=0.001)

**Evaluation Metrics**:
- RMSE (speed accuracy)
- R² (variance explained)
- Uncertainty calibration (reliability diagram)
- Attention interpretability (qualitative)

**Expected Results**:
- RMSE: 2.5-3.0 m/s (maintains Phase 1 accuracy)
- R²: 0.90-0.93 (similar to Phase 1)
- Uncertainty: Well-calibrated (correlates with error)

**Success Criteria**:
- ✅ Training converges (no NaN losses)
- ✅ RMSE ≤ 3 m/s
- ✅ Uncertainty meaningful (high during errors)
- ✅ Physics constraints satisfied (smooth, bounded)

**Estimated Effort**: 1 day (training), 1 day (evaluation)

---

### Phase 2C Summary

**Total Duration**: 6-8 days  
**Dependencies**: Phase 2A.1 (Visualization for attention weights)  
**Risk**: Medium-High (research-oriented, accuracy critical)  
**Value**: Medium (interpretability, not directly user-facing)

**Deliverables**:
- ✅ Attention mechanism implemented and visualized
- ✅ Physics-informed loss prevents unrealistic predictions
- ✅ Uncertainty estimation provides confidence scores
- ✅ Integrated model maintains Phase 1 accuracy (RMSE ≤ 3 m/s)
- ✅ Updated documentation and visualization

**Testing Plan**:
1. Validate attention weights make sense (recent data weighted higher)
2. Test physics constraints (no negative speeds, smooth predictions)
3. Evaluate uncertainty calibration (reliability diagram)
4. Compare Phase 2C vs Phase 1 accuracy

---

## 📅 Overall Phase 2 Timeline

### Week 1: UI Enhancements (Phase 2A)
- **Day 1**: Task 2A.1 - Visualization Screen
- **Day 2**: Task 2A.2 - File Sharing UI (0.5 days)
- **Day 2-3**: Task 2A.3 - Settings Screen (1.5 days)
- **Day 4**: Testing and documentation

**Deliverable**: Enhanced Android app with visualization, sharing, settings ✅

---

### Week 2: TCN Model (Phase 2B)
- **Day 1**: Task 2B.1 - Implement TCN architecture
- **Day 2**: Task 2B.2 - Train TCN (0.5 days active, 3-4 hours waiting)
- **Day 3**: Task 2B.3 - Export to TFLite with NNAPI
- **Day 4-5**: Task 2B.4 - Integrate TCN into Android app
- **Day 5**: Testing and benchmarking

**Deliverable**: NNAPI-compatible TCN model, 2-3x faster inference ✅

---

### Week 3-4: Advanced ML (Phase 2C)
- **Day 1-2**: Task 2C.1 - Attention mechanism
- **Day 3**: Task 2C.2 - Physics-informed loss
- **Day 4-5**: Task 2C.3 - Uncertainty estimation
- **Day 6**: Task 2C.4 - Integrated training
- **Day 7**: Task 2C.4 - Evaluation and analysis
- **Day 8**: Testing, documentation, wrap-up

**Deliverable**: Interpretable model with attention, uncertainty, physics ✅

---

## ✅ Success Criteria

### Phase 2A Success Metrics
- ✅ Visualization: 60fps, real-time updates, 10-second window
- ✅ File sharing: Works with WhatsApp, email, Drive
- ✅ Settings: 5 preferences, persist across restarts

### Phase 2B Success Metrics
- ✅ TCN accuracy: RMSE ≤ 4 m/s (within 1.2 m/s of Phase 1)
- ✅ TCN performance: 5-10ms inference (2-3x faster than BiLSTM)
- ✅ NNAPI: All operations supported, no fallback

### Phase 2C Success Metrics
- ✅ Attention: Meaningful weights, visualized in UI
- ✅ Physics: No negative speeds, no >50 m/s, smooth predictions
- ✅ Uncertainty: Correlates with error, calibrated
- ✅ Overall accuracy: RMSE ≤ 3 m/s (maintains Phase 1)

---

## 🚧 Risks and Mitigation

### Risk 1: TCN Accuracy Lower Than Expected
**Mitigation**: Keep BiLSTM as default, TCN as optional  
**Fallback**: If TCN RMSE > 5 m/s, skip Phase 2B and move to 2C

### Risk 2: NNAPI Still Doesn't Work
**Mitigation**: Verify NNAPI compatibility before training  
**Fallback**: Use CPU-only TCN (still faster than BiLSTM due to fewer params)

### Risk 3: Phase 2C Doesn't Improve Accuracy
**Mitigation**: Phase 2C is about interpretability, not accuracy  
**Acceptance Criteria**: RMSE ≤ 3 m/s (maintain Phase 1), add features

### Risk 4: UI Changes Cause Performance Issues
**Mitigation**: Benchmark chart rendering, optimize if needed  
**Fallback**: Reduce chart update rate from 10Hz to 5Hz

---

## 📊 Resource Requirements

### Hardware
- ✅ Development: Windows 11 PC (already available)
- ✅ GPU Training: NVIDIA RTX 3050 Ti (already available)
- ✅ Testing: OnePlus 11R (Snapdragon 8+ Gen 1, already available)

### Software
- ✅ Android Studio (already installed)
- ✅ TensorFlow/Keras (already installed)
- ✅ PyTorch (for TCN, needs installation)
- 📦 MPAndroidChart (add to build.gradle.kts)

### Datasets
- ✅ Comma2k19: 478,891 samples (already downloaded)
- ❌ EuRoC, KITTI, OxIOD: Not needed for Phase 2

---

## 📚 Documentation Updates

After each sub-phase:

1. **Update PROJECT_STATUS.md**: Current state, bugs, metrics
2. **Update README.md**: Features, screenshots, usage
3. **Update ANDROID_APP_STATUS.md**: New UI features, model selection
4. **Update TRAINING_HISTORY.md**: TCN training, Phase 2C results
5. **Create PHASE2_SUMMARY.md**: Final results, learnings

---

## 🎯 Next Immediate Action

**Start with Phase 2A - Task 2A.1: Visualization Screen**

**Why this first?**
1. ✅ Highest user value - seeing predictions validates the system
2. ✅ Lowest risk - pure UI work, no ML changes
3. ✅ Enables Phase 2C - visualization needed for attention weights
4. ✅ Quick win - can complete in 1 day
5. ✅ Testable - immediate visual feedback

**Implementation Steps**:
1. Add MPAndroidChart dependency to `build.gradle.kts`
2. Create `VisualizationScreen.kt` with line chart
3. Create `SpeedDataPoint.kt` data class
4. Modify `SpeedPredictionService.kt` to store last 100 predictions
5. Add navigation from `MainActivity.kt`
6. Test on OnePlus 11R

**Expected Outcome**: Real-time speed chart showing last 10 seconds of predictions ✅

---

## 🏁 Phase 2 Completion Criteria

Phase 2 is complete when:

- ✅ **Phase 2A**: Visualization, file sharing, settings working
- ✅ **Phase 2B**: TCN model deployed, NNAPI working, 2-3x faster
- ✅ **Phase 2C**: Attention, physics loss, uncertainty implemented
- ✅ **Documentation**: All .md files updated
- ✅ **Testing**: Manual testing on OnePlus 11R passed
- ✅ **Git**: All changes committed with clear messages

**Estimated Total Duration**: 3-4 weeks (15-20 working days)

**After Phase 2**: Move to Phase 3 (GTSAM sensor fusion) or Phase 4 (Production release)

---

*Phase 2 Plan Created: October 11, 2025*  
*Next Milestone: Phase 2A.1 - Visualization Screen*  
*Let's build! 🚀*
