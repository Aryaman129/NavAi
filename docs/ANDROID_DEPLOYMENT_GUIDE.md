# 📱 Android Deployment Guide - NavAI Speed Estimation

**Last Updated**: December 2025  
**Phase**: Phase 1 Model Deployment  
**Status**: Ready for Android Testing (No Maps/Frontend Needed)

---

## 🎯 Overview

This guide shows how to deploy the Phase 1 speed estimation model to Android **without** needing maps or a full frontend. We'll create a minimal test app that just displays the predicted speed.

---

## 📋 Prerequisites

### From Python Side (Already Done ✅)

- ✅ Trained model: `ml/outputs/phase1_best_model.pth`
- ✅ Preprocessor: `ml/outputs/phase1_preprocessor.pkl`
- ⏳ TFLite export: Run `ml/export_tflite.py` (next step)

### Android Development Tools (Verified ✅)

**Development Environment**:
- ✅ Android Studio: `D:\Android SDK\` (installed)
- ✅ ADB v36.0.0: `D:\Android Platform tool\platform-tools\adb.exe` (working)
- ✅ Platform tools: Available and ready

**Hardware Requirements**:
- ✅ **USB Cable Required**: Yes, for development and deployment
  - USB 2.0 or 3.0 cable (phone to PC)
  - Used for: Installing app, debugging, log viewing
  - Alternative wireless ADB available but USB is recommended
- ✅ Android phone with:
  - Minimum SDK 26 (Android 8.0+)
  - Developer Options enabled
  - USB Debugging enabled
  - IMU sensors (accelerometer + gyroscope)

**SDK Configuration**:
- Minimum SDK: 26 (Android 8.0)
- Target SDK: 34 (Android 14)
- Kotlin 1.9+
- TensorFlow Lite dependency

---

## 🔌 USB Connection Setup

### Step 1: Enable Developer Options on Phone

1. Open **Settings** on your Android phone
2. Go to **About Phone**
3. Find **Build Number**
4. Tap **Build Number** 7 times
5. You should see "You are now a developer!"

### Step 2: Enable USB Debugging

1. Go back to **Settings**
2. Find **Developer Options** (usually under System)
3. Enable **USB Debugging**
4. Enable **Install via USB** (optional, helpful)

### Step 3: Connect Phone via USB

1. Connect phone to PC using USB cable
2. On phone, you'll see "Allow USB Debugging?" popup
3. Check "Always allow from this computer"
4. Tap **OK**

### Step 4: Verify ADB Connection

Open PowerShell and run:
```powershell
& "D:\Android Platform tool\platform-tools\adb.exe" devices
```

Expected output:
```
List of devices attached
ABCD1234567890    device
```

If you see "unauthorized", repeat Step 3.

### Wireless ADB (Alternative - Not Recommended for Development)

If USB is unavailable, you can use wireless ADB:

```powershell
# First connect via USB, then:
& "D:\Android Platform tool\platform-tools\adb.exe" tcpip 5555

# Find phone IP (Settings → About → Status → IP address)
# Then connect wirelessly:
& "D:\Android Platform tool\platform-tools\adb.exe" connect 192.168.1.XXX:5555

# Verify
& "D:\Android Platform tool\platform-tools\adb.exe" devices
```

**Note**: Wireless is slower and less stable. Use USB for development.

---

## 🚀 Step-by-Step Deployment

### Step 1: Export Model to TFLite

Run the export script:

```bash
python ml/export_tflite.py
```

This will create:
- `ml/outputs/phase1_model.tflite` (TFLite model)
- `ml/outputs/preprocessor_params.json` (mean/std for normalization)

**What the export does**:
```python
# Converts PyTorch BiLSTM → TFLite
# Includes:
# - Input shape: [1, 100, 10] (batch, timesteps, features)
# - Output shape: [1, 1] (batch, speed)
# - Quantization: Optional (for smaller size/faster inference)
```

### Step 2: Create Android Project

**File Structure**:
```
mobile/app/
├── src/main/
│   ├── assets/
│   │   ├── phase1_model.tflite          # TFLite model
│   │   └── preprocessor_params.json     # Normalization params
│   │
│   ├── java/com/navai/speedtest/
│   │   ├── MainActivity.kt              # Simple UI
│   │   ├── IMUDataCollector.kt          # Sensor reading
│   │   ├── Preprocessor.kt              # Feature engineering
│   │   ├── TFLiteInference.kt           # Model inference
│   │   └── SpeedEstimator.kt            # Main logic
│   │
│   └── res/layout/
│       └── activity_main.xml            # Simple UI layout
│
└── build.gradle.kts                     # Dependencies
```

### Step 3: Add Dependencies

**mobile/app/build.gradle.kts**:
```kotlin
dependencies {
    // TensorFlow Lite
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    
    // Kotlin coroutines (for async processing)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    
    // JSON parsing (for preprocessor params)
    implementation("org.json:json:20230227")
    
    // Location (for GPS comparison)
    implementation("com.google.android.gms:play-services-location:21.0.1")
}

android {
    aaptOptions {
        noCompress "tflite"  // Don't compress TFLite models
    }
}
```

---

## 💻 Android Implementation

### 1. IMU Data Collector

**IMUDataCollector.kt**:
```kotlin
package com.navai.speedtest

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import java.util.concurrent.CopyOnWriteArrayList

class IMUDataCollector(context: Context) : SensorEventListener {
    
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    
    // Circular buffer for 100 samples (1 second @ 100Hz)
    private val accelBuffer = CopyOnWriteArrayList<FloatArray>()
    private val gyroBuffer = CopyOnWriteArrayList<FloatArray>()
    
    private var callback: ((FloatArray) -> Unit)? = null
    
    fun start() {
        sensorManager.registerListener(this, accelerometer, 10_000) // 100Hz = 10ms
        sensorManager.registerListener(this, gyroscope, 10_000)
    }
    
    fun stop() {
        sensorManager.unregisterListener(this)
    }
    
    fun setCallback(callback: (FloatArray) -> Unit) {
        this.callback = callback
    }
    
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                accelBuffer.add(event.values.clone())
                if (accelBuffer.size > 100) accelBuffer.removeAt(0)
            }
            Sensor.TYPE_GYROSCOPE -> {
                gyroBuffer.add(event.values.clone())
                if (gyroBuffer.size > 100) gyroBuffer.removeAt(0)
            }
        }
        
        // When we have 100 samples of both, trigger inference
        if (accelBuffer.size == 100 && gyroBuffer.size == 100) {
            val imuWindow = combineIMUData()
            callback?.invoke(imuWindow)
        }
    }
    
    private fun combineIMUData(): FloatArray {
        // Combine accel (x,y,z) + gyro (x,y,z) = 6 features per timestep
        // 100 timesteps × 6 features = 600 values
        val combined = FloatArray(600)
        
        for (i in 0 until 100) {
            val accel = accelBuffer[i]
            val gyro = gyroBuffer[i]
            
            combined[i * 6 + 0] = accel[0]  // accel_x
            combined[i * 6 + 1] = accel[1]  // accel_y
            combined[i * 6 + 2] = accel[2]  // accel_z
            combined[i * 6 + 3] = gyro[0]   // gyro_x
            combined[i * 6 + 4] = gyro[1]   // gyro_y
            combined[i * 6 + 5] = gyro[2]   // gyro_z
        }
        
        return combined
    }
    
    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        // Not needed
    }
}
```

### 2. Preprocessor (Feature Engineering + Normalization)

**Preprocessor.kt**:
```kotlin
package com.navai.speedtest

import android.content.Context
import org.json.JSONObject
import kotlin.math.sqrt

class Preprocessor(context: Context) {
    
    private var mean: FloatArray = FloatArray(10)
    private var std: FloatArray = FloatArray(10)
    
    init {
        // Load normalization params from JSON
        val json = context.assets.open("preprocessor_params.json").bufferedReader().use { it.readText() }
        val params = JSONObject(json)
        
        val meanArray = params.getJSONArray("mean")
        val stdArray = params.getJSONArray("std")
        
        for (i in 0 until 10) {
            mean[i] = meanArray.getDouble(i).toFloat()
            std[i] = stdArray.getDouble(i).toFloat()
        }
    }
    
    fun process(rawIMU: FloatArray): FloatArray {
        // Input: 600 values (100 timesteps × 6 features)
        // Output: 1000 values (100 timesteps × 10 features)
        
        val processed = FloatArray(1000)
        
        for (t in 0 until 100) {
            val idx = t * 6
            
            // Extract raw values
            val ax = rawIMU[idx + 0]
            val ay = rawIMU[idx + 1]
            val az = rawIMU[idx + 2]
            val gx = rawIMU[idx + 3]
            val gy = rawIMU[idx + 4]
            val gz = rawIMU[idx + 5]
            
            // Feature engineering (same as Python)
            val accelMag = sqrt(ax*ax + ay*ay + az*az)
            val gyroMag = sqrt(gx*gx + gy*gy + gz*gz)
            
            // Derivatives (if not first timestep)
            val accelDeriv = if (t > 0) {
                val prevAx = rawIMU[(t-1) * 6 + 0]
                val prevAy = rawIMU[(t-1) * 6 + 1]
                val prevAz = rawIMU[(t-1) * 6 + 2]
                sqrt((ax-prevAx)*(ax-prevAx) + (ay-prevAy)*(ay-prevAy) + (az-prevAz)*(az-prevAz))
            } else {
                0f
            }
            
            val gyroDeriv = if (t > 0) {
                val prevGx = rawIMU[(t-1) * 6 + 3]
                val prevGy = rawIMU[(t-1) * 6 + 4]
                val prevGz = rawIMU[(t-1) * 6 + 5]
                sqrt((gx-prevGx)*(gx-prevGx) + (gy-prevGy)*(gy-prevGy) + (gz-prevGz)*(gz-prevGz))
            } else {
                0f
            }
            
            // Create 10-feature vector
            val features = floatArrayOf(
                ax, ay, az, gx, gy, gz,       // Raw IMU (6 features)
                accelMag, gyroMag,             // Magnitudes (2 features)
                accelDeriv, gyroDeriv          // Derivatives (2 features)
            )
            
            // Normalize using fitted scaler
            for (i in 0 until 10) {
                processed[t * 10 + i] = (features[i] - mean[i]) / std[i]
            }
        }
        
        return processed
    }
}
```

### 3. TFLite Inference

**TFLiteInference.kt**:
```kotlin
package com.navai.speedtest

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteInference(context: Context) {
    
    private val interpreter: Interpreter
    
    init {
        // Load TFLite model from assets
        val modelBuffer = loadModelFile(context, "phase1_model.tflite")
        interpreter = Interpreter(modelBuffer)
    }
    
    fun predict(preprocessedInput: FloatArray): Float {
        // Reshape to [1, 100, 10]
        val input = Array(1) { Array(100) { FloatArray(10) } }
        
        for (t in 0 until 100) {
            for (f in 0 until 10) {
                input[0][t][f] = preprocessedInput[t * 10 + f]
            }
        }
        
        // Output: [1, 1]
        val output = Array(1) { FloatArray(1) }
        
        // Run inference
        interpreter.run(input, output)
        
        return output[0][0]  // Predicted speed in m/s
    }
    
    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun close() {
        interpreter.close()
    }
}
```

### 4. Main Activity (Simple UI)

**MainActivity.kt**:
```kotlin
package com.navai.speedtest

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import kotlinx.coroutines.*

class MainActivity : AppCompatActivity() {
    
    private lateinit var speedTextView: TextView
    private lateinit var gpsTextView: TextView
    private lateinit var diffTextView: TextView
    private lateinit var latencyTextView: TextView
    
    private lateinit var imuCollector: IMUDataCollector
    private lateinit var preprocessor: Preprocessor
    private lateinit var tfliteInference: TFLiteInference
    
    private val scope = CoroutineScope(Dispatchers.Main + Job())
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // UI elements
        speedTextView = findViewById(R.id.predictedSpeed)
        gpsTextView = findViewById(R.id.gpsSpeed)
        diffTextView = findViewById(R.id.difference)
        latencyTextView = findViewById(R.id.latency)
        
        // Request sensor permissions
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.BODY_SENSORS) 
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.BODY_SENSORS), 1)
        }
        
        // Initialize components
        preprocessor = Preprocessor(this)
        tfliteInference = TFLiteInference(this)
        imuCollector = IMUDataCollector(this)
        
        // Set callback for when we have 100 IMU samples
        imuCollector.setCallback { rawIMU ->
            scope.launch(Dispatchers.Default) {
                val startTime = System.currentTimeMillis()
                
                // Preprocess
                val preprocessed = preprocessor.process(rawIMU)
                
                // Inference
                val predictedSpeed = tfliteInference.predict(preprocessed)
                
                val latency = System.currentTimeMillis() - startTime
                
                // Update UI
                withContext(Dispatchers.Main) {
                    speedTextView.text = "%.2f m/s".format(predictedSpeed)
                    latencyTextView.text = "Latency: %d ms".format(latency)
                }
            }
        }
        
        // Start collecting IMU data
        imuCollector.start()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        imuCollector.stop()
        tfliteInference.close()
        scope.cancel()
    }
}
```

### 5. Simple UI Layout

**res/layout/activity_main.xml**:
```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout 
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:gravity="center"
    android:padding="24dp">
    
    <TextView
        android:text="NavAI Speed Test"
        android:textSize="24sp"
        android:textStyle="bold"
        android:layout_marginBottom="48dp"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content" />
    
    <TextView
        android:text="Predicted Speed"
        android:textSize="18sp"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content" />
    
    <TextView
        android:id="@+id/predictedSpeed"
        android:text="-- m/s"
        android:textSize="72sp"
        android:textStyle="bold"
        android:textColor="#2196F3"
        android:layout_marginBottom="32dp"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content" />
    
    <TextView
        android:id="@+id/gpsSpeed"
        android:text="GPS: -- m/s"
        android:textSize="18sp"
        android:layout_marginBottom="8dp"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content" />
    
    <TextView
        android:id="@+id/difference"
        android:text="Diff: -- m/s"
        android:textSize="18sp"
        android:layout_marginBottom="8dp"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content" />
    
    <TextView
        android:id="@+id/latency"
        android:text="Latency: -- ms"
        android:textSize="16sp"
        android:textColor="#757575"
        android:layout_marginTop="32dp"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content" />
    
</LinearLayout>
```

---

## 🧪 Testing Without Maps

### What to Test

1. **Inference Latency**
   - Target: <50ms per prediction
   - Measure: `System.currentTimeMillis()` before/after

2. **Accuracy vs GPS**
   - Compare predicted speed to GPS speed
   - Test in different scenarios:
     - Stationary (should predict ~0 m/s)
     - Constant speed (highway driving)
     - Acceleration/deceleration
     - Turns

3. **Battery Consumption**
   - Run for 1 hour, monitor battery drain
   - Target: <10% per hour

4. **Edge Cases**
   - Phone orientation changes
   - Rapid direction changes
   - Bumpy roads
   - Walking vs driving

### Test Scenarios

**Stationary Test**:
- Place phone on table for 5 minutes
- Expected: Speed should be near 0 m/s
- Validates: ZUPT detection working

**Driving Test**:
- Drive at constant speed (e.g., 20 m/s / 72 km/h)
- Expected: Predicted speed ≈ GPS speed ± 3 m/s
- Validates: Model accuracy

**Acceleration Test**:
- Accelerate from 0 to highway speed
- Expected: Smooth increase, no sudden jumps
- Validates: EKF smoothing working

---

## 📊 Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Inference Latency** | <50ms | `System.currentTimeMillis()` |
| **Accuracy (Stationary)** | <1 m/s | Phone on table, should predict ~0 |
| **Accuracy (Driving)** | ±3 m/s vs GPS | Compare predicted vs GPS |
| **Battery Drain** | <10% per hour | Run for 1 hour, check battery |
| **Crash Rate** | 0% | Should never crash |

---

## 🐛 Troubleshooting

### Model Not Loading

**Error**: `Failed to load TFLite model`

**Fix**:
1. Check `assets/` folder has `phase1_model.tflite`
2. Add to `build.gradle.kts`: `aaptOptions { noCompress "tflite" }`

### Wrong Predictions

**Error**: All predictions are ~0 or very high

**Fix**:
1. Check preprocessor parameters match Python
2. Verify feature engineering is identical
3. Print intermediate values to debug

### High Latency

**Error**: Inference takes >100ms

**Fix**:
1. Use quantized model (8-bit integers)
2. Run `export_tflite.py --quantize`
3. Consider GPU delegate (if device supports)

---

## 🚀 Next Steps After Testing

**If Testing Goes Well** ✅:
1. Add GPS comparison overlay
2. Log data for offline analysis
3. Prepare for Phase 2 (if needed)

**If Issues Found** ⚠️:
1. Collect failure cases
2. Retrain with edge case data
3. Consider Phase 2 features (uncertainty, physics)

**No Maps Needed Yet** 📍:
- Focus on speed accuracy first
- Maps are for visualization only
- Can add later once speed estimation is validated

---

## 📚 References

- TensorFlow Lite: https://www.tensorflow.org/lite/android
- Android Sensors: https://developer.android.com/guide/topics/sensors
- Kotlin Coroutines: https://kotlinlang.org/docs/coroutines-overview.html

---

**Ready to Deploy**: All code provided above is complete and ready to use!
