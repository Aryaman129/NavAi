package com.navai.logger.ml

import android.content.Context
import android.util.Log
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/**
 * TFLite-based speed predictor using IMU data (accelerometer + gyroscope)
 * 
 * Model Architecture: BiLSTM with 100-sample sliding window
 * Input: [1, 100, 6] (batch, timesteps, features)
 * Features: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
 * Output: [1, 1] (predicted speed in m/s)
 */
class SpeedPredictor(context: Context) {
    
    companion object {
        private const val TAG = "SpeedPredictor"
        private const val MODEL_FILE = "phase1_model.tflite"
        private const val METADATA_FILE = "phase1_model_metadata.json"
        private const val WINDOW_SIZE = 100
        private const val NUM_FEATURES = 6
    }
    
    private var interpreter: Interpreter? = null
    private var metadata: ModelMetadata? = null
    
    // Sliding window for IMU data
    private val dataWindow = ArrayDeque<FloatArray>(WINDOW_SIZE)
    
    // Performance metrics
    private var inferenceCount = 0
    private var totalInferenceTime = 0L
    private var minInferenceTime = Long.MAX_VALUE
    private var maxInferenceTime = 0L
    
    init {
        try {
            // Load model
            val modelBuffer = loadModelFile(context, MODEL_FILE)
            val options = Interpreter.Options().apply {
                setNumThreads(4) // Use 4 CPU threads for inference
                setUseNNAPI(true) // Try to use Android NNAPI for acceleration
            }
            interpreter = Interpreter(modelBuffer, options)
            Log.i(TAG, "✅ TFLite model loaded successfully")
            Log.i(TAG, "   Model file: $MODEL_FILE (${modelBuffer.capacity() / 1024 / 1024}MB)")
            
            // Load metadata (normalization parameters)
            metadata = loadMetadata(context, METADATA_FILE)
            Log.i(TAG, "✅ Model metadata loaded")
            Log.i(TAG, "   Window size: ${metadata?.window_size}")
            Log.i(TAG, "   Features: ${metadata?.num_features}")
            Log.i(TAG, "   Accel mean: ${metadata?.accel_mean}")
            Log.i(TAG, "   Accel std: ${metadata?.accel_std}")
            Log.i(TAG, "   Gyro mean: ${metadata?.gyro_mean}")
            Log.i(TAG, "   Gyro std: ${metadata?.gyro_std}")
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to initialize SpeedPredictor", e)
            Log.e(TAG, "   Error type: ${e.javaClass.simpleName}")
            Log.e(TAG, "   Error message: ${e.message}")
            Log.e(TAG, "   Stack trace: ${e.stackTraceToString()}")
        }
    }
    
    /**
     * Add IMU sample to sliding window
     * @param accel Accelerometer [x, y, z] in m/s²
     * @param gyro Gyroscope [x, y, z] in rad/s
     */
    fun addSample(accel: FloatArray, gyro: FloatArray) {
        require(accel.size == 3) { "Accelerometer must have 3 values" }
        require(gyro.size == 3) { "Gyroscope must have 3 values" }
        
        // Combine features: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        val sample = floatArrayOf(
            accel[0], accel[1], accel[2],
            gyro[0], gyro[1], gyro[2]
        )
        
        // Add to window (remove oldest if full)
        if (dataWindow.size >= WINDOW_SIZE) {
            dataWindow.removeFirst()
        }
        dataWindow.addLast(sample)
        
        // Log when window fills up for the first time
        if (dataWindow.size == WINDOW_SIZE && inferenceCount == 0) {
            Log.i(TAG, "🎯 Window full ($WINDOW_SIZE samples) - ready for inference!")
        }
    }
    
    /**
     * Predict speed from current window
     * @return Predicted speed in m/s, or null if window not full
     */
    fun predictSpeed(): PredictionResult? {
        if (dataWindow.size < WINDOW_SIZE) {
            // Log buffering progress every 20 samples
            if (dataWindow.size % 20 == 0 && dataWindow.size > 0) {
                Log.d(TAG, "⏳ Buffering: ${dataWindow.size}/$WINDOW_SIZE samples")
            }
            return null // Not enough data yet
        }
        
        val meta = metadata ?: return null
        val interp = interpreter ?: return null
        
        val startTime = System.nanoTime()
        
        try {
            // Normalize and prepare input tensor
            val inputBuffer = ByteBuffer.allocateDirect(1 * WINDOW_SIZE * NUM_FEATURES * 4).apply {
                order(ByteOrder.nativeOrder())
            }
            
            dataWindow.forEachIndexed { timestep, sample ->
                for (i in 0 until NUM_FEATURES) {
                    val normalizedValue = if (i < 3) {
                        // Accelerometer (first 3 features)
                        (sample[i] - meta.accel_mean[i]) / meta.accel_std[i]
                    } else {
                        // Gyroscope (last 3 features)
                        (sample[i] - meta.gyro_mean[i - 3]) / meta.gyro_std[i - 3]
                    }
                    inputBuffer.putFloat(normalizedValue)
                }
            }
            
            // Output tensor
            val outputBuffer = ByteBuffer.allocateDirect(1 * 1 * 4).apply {
                order(ByteOrder.nativeOrder())
            }
            
            // Run inference
            interp.run(inputBuffer, outputBuffer)
            
            // Parse output
            outputBuffer.rewind()
            val predictedSpeed = outputBuffer.float
            
            val inferenceTime = (System.nanoTime() - startTime) / 1_000_000 // Convert to ms
            
            // Update metrics
            inferenceCount++
            totalInferenceTime += inferenceTime
            minInferenceTime = minOf(minInferenceTime, inferenceTime)
            maxInferenceTime = maxOf(maxInferenceTime, inferenceTime)
            
            return PredictionResult(
                speedMps = predictedSpeed,
                speedKmh = predictedSpeed * 3.6f,
                inferenceTimeMs = inferenceTime,
                windowSize = dataWindow.size
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Inference failed", e)
            return null
        }
    }
    
    /**
     * Get performance statistics
     */
    fun getStats(): PerformanceStats {
        return PerformanceStats(
            inferenceCount = inferenceCount,
            avgInferenceTimeMs = if (inferenceCount > 0) totalInferenceTime / inferenceCount else 0,
            minInferenceTimeMs = if (inferenceCount > 0) minInferenceTime else 0,
            maxInferenceTimeMs = maxInferenceTime
        )
    }
    
    /**
     * Reset sliding window
     */
    fun reset() {
        dataWindow.clear()
        Log.i(TAG, "Window reset")
    }
    
    /**
     * Get current window size
     */
    fun getWindowSize(): Int = dataWindow.size
    
    /**
     * Clean up resources
     */
    fun close() {
        interpreter?.close()
        interpreter = null
        Log.i(TAG, "SpeedPredictor closed")
    }
    
    // Helper functions
    
    private fun loadModelFile(context: Context, filename: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    private fun loadMetadata(context: Context, filename: String): ModelMetadata {
        val json = context.assets.open(filename).bufferedReader().use { it.readText() }
        return Json.decodeFromString<ModelMetadata>(json)
    }
}

/**
 * Model metadata (normalization parameters)
 */
@Serializable
data class ModelMetadata(
    val accel_mean: List<Float>,
    val accel_std: List<Float>,
    val gyro_mean: List<Float>,
    val gyro_std: List<Float>,
    val speed_mean: Float,
    val speed_std: Float,
    val window_size: Int,
    val num_features: Int,
    val model_info: ModelInfo? = null
)

@Serializable
data class ModelInfo(
    val architecture: String? = null,
    val best_epoch: Int? = null,
    val best_val_mae: Float? = null,
    val training_time: String? = null
)

/**
 * Prediction result
 */
data class PredictionResult(
    val speedMps: Float,        // Speed in m/s
    val speedKmh: Float,        // Speed in km/h
    val inferenceTimeMs: Long,  // Inference latency
    val windowSize: Int         // Actual window size used
)

/**
 * Performance statistics
 */
data class PerformanceStats(
    val inferenceCount: Int,
    val avgInferenceTimeMs: Long,
    val minInferenceTimeMs: Long,
    val maxInferenceTimeMs: Long
)
