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
    private var initializationError: String? = null
    
    // Sliding window for IMU data
    private val dataWindow = ArrayDeque<FloatArray>(WINDOW_SIZE)
    
    // Performance metrics
    private var inferenceCount = 0
    private var totalInferenceTime = 0L
    private var minInferenceTime = Long.MAX_VALUE
    private var maxInferenceTime = 0L
    
    init {
        Log.i(TAG, "🔄 SpeedPredictor initialization starting...")
        
        try {
            // Check if model files exist in assets
            val assetList = context.assets.list("")
            Log.i(TAG, "📂 Assets available: ${assetList?.take(10)?.joinToString()}")
            
            if (assetList?.contains(MODEL_FILE) != true) {
                throw RuntimeException("❌ Model file $MODEL_FILE not found in assets!")
            }
            
            if (assetList?.contains(METADATA_FILE) != true) {
                throw RuntimeException("❌ Metadata file $METADATA_FILE not found in assets!")
            }
            
            // Load model
            Log.i(TAG, "📥 Loading model file: $MODEL_FILE...")
            val modelBuffer = loadModelFile(context, MODEL_FILE)
            Log.i(TAG, "✅ Model loaded: ${modelBuffer.capacity()} bytes (${modelBuffer.capacity() / 1024 / 1024}MB)")
            
            // CRITICAL: Cascade fallback for delegate initialization
            // NNAPI may fail on some devices (e.g., Snapdragon 8+ Gen 1 doesn't support BiLSTM)
            interpreter = try {
                // ATTEMPT 1: Try NNAPI acceleration first
                Log.i(TAG, "🚀 Attempting NNAPI delegate...")
                val nnApiOptions = Interpreter.Options().apply {
                    setNumThreads(4)
                    setUseNNAPI(true)
                }
                val interp = Interpreter(modelBuffer, nnApiOptions)
                Log.i(TAG, "✅ SUCCESS: NNAPI delegate active!")
                interp
                
            } catch (e: Exception) {
                // NNAPI failed - fall back to CPU-only mode
                Log.w(TAG, "⚠️ NNAPI failed: ${e.javaClass.simpleName}: ${e.message}")
                Log.i(TAG, "🔄 Falling back to CPU-only mode...")
                
                // ATTEMPT 2: Pure CPU mode (always works)
                val cpuOptions = Interpreter.Options().apply {
                    setNumThreads(4)  // Utilize all 4 high-performance cores
                    // NO delegates - pure CPU inference
                }
                val interp = Interpreter(modelBuffer, cpuOptions)
                Log.i(TAG, "✅ SUCCESS: CPU-only mode active (4 threads)")
                Log.i(TAG, "   Expected latency: 10-20ms per prediction")
                interp
            }
            
            Log.i(TAG, "✅ Interpreter created successfully")
            
            // Verify tensor shapes
            val inputTensor = interpreter!!.getInputTensor(0)
            val outputTensor = interpreter!!.getOutputTensor(0)
            Log.i(TAG, "📊 Input tensor shape: ${inputTensor.shape().contentToString()}")
            Log.i(TAG, "📊 Output tensor shape: ${outputTensor.shape().contentToString()}")
            Log.i(TAG, "📊 Input data type: ${inputTensor.dataType()}")
            
            // Load metadata (normalization parameters)
            Log.i(TAG, "📥 Loading metadata file: $METADATA_FILE...")
            metadata = loadMetadata(context, METADATA_FILE)
            Log.i(TAG, "✅ Model metadata loaded")
            Log.i(TAG, "   Window size: ${metadata?.window_size}")
            Log.i(TAG, "   Features: ${metadata?.num_features}")
            Log.i(TAG, "   Accel mean: ${metadata?.accel_mean}")
            Log.i(TAG, "   Accel std: ${metadata?.accel_std}")
            Log.i(TAG, "   Gyro mean: ${metadata?.gyro_mean}")
            Log.i(TAG, "   Gyro std: ${metadata?.gyro_std}")
            
            Log.i(TAG, "🎉 SpeedPredictor initialization COMPLETE!")
            
        } catch (e: Exception) {
            val errorMsg = "Failed to initialize SpeedPredictor: ${e.javaClass.simpleName}: ${e.message}"
            initializationError = errorMsg
            
            Log.e(TAG, "❌ $errorMsg", e)
            Log.e(TAG, "   Exception type: ${e.javaClass.name}")
            Log.e(TAG, "   Message: ${e.message}")
            Log.e(TAG, "   Stack trace:")
            e.printStackTrace()
            
            // Re-throw so service knows initialization failed
            throw RuntimeException("SpeedPredictor initialization failed", e)
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
     * Detect if device is stationary using IMU variance (ZUPT)
     * Returns true if gyro variance < threshold (device not rotating)
     */
    private fun isStationaryState(): Boolean {
        if (dataWindow.size < 10) return false // Need at least 10 samples
        
        // Calculate gyroscope variance over recent samples
        val recentSamples = dataWindow.takeLast(10)
        
        // Gyro is features 3, 4, 5
        val gyroX = recentSamples.map { it[3] }
        val gyroY = recentSamples.map { it[4] }
        val gyroZ = recentSamples.map { it[5] }
        
        val gyroXVar = variance(gyroX)
        val gyroYVar = variance(gyroY)
        val gyroZVar = variance(gyroZ)
        
        // ZUPT thresholds (rad/s)² - tuned for stationary detection
        // Typical sensor noise: ~0.01 rad/s, so variance ~0.0001
        // Moving phone: variance > 0.01
        val GYRO_VARIANCE_THRESHOLD = 0.005f
        
        val isStationary = gyroXVar < GYRO_VARIANCE_THRESHOLD && 
                          gyroYVar < GYRO_VARIANCE_THRESHOLD && 
                          gyroZVar < GYRO_VARIANCE_THRESHOLD
        
        return isStationary
    }
    
    /**
     * Calculate variance of a list of values
     */
    private fun variance(values: List<Float>): Float {
        if (values.isEmpty()) return 0f
        val mean = values.average().toFloat()
        return values.map { (it - mean) * (it - mean) }.average().toFloat()
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
            var predictedSpeed = outputBuffer.float
            
            // Apply ZUPT (Zero Velocity Update) for stationary detection
            // Calculate IMU variance to detect stationary state
            val isStationary = isStationaryState()
            if (isStationary) {
                Log.d(TAG, "ZUPT applied: IMU variance low, forcing speed to 0")
                predictedSpeed = 0f
            }
            
            // Clip negative speeds (model should never predict negative)
            predictedSpeed = maxOf(0f, predictedSpeed)
            
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
     * Check if predictor is properly initialized
     */
    fun isInitialized(): Boolean = interpreter != null && metadata != null
    
    /**
     * Get initialization error if any
     */
    fun getInitializationError(): String? = initializationError
    
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
        try {
            // Read JSON from assets
            val json = context.assets.open(filename).bufferedReader().use { it.readText() }
            
            // Strip UTF-8 BOM if present (EF BB BF = \uFEFF)
            // Some text editors add this automatically to UTF-8 files
            // JSON parsers don't handle it - must be removed manually
            val cleanJson = if (json.isNotEmpty() && json[0] == '\uFEFF') {
                Log.w(TAG, "⚠️ UTF-8 BOM detected in $filename - removing it")
                json.substring(1)  // Remove first character
            } else {
                json
            }
            
            Log.d(TAG, "📄 JSON first 50 chars: ${cleanJson.take(50)}")
            
            // Parse JSON to ModelMetadata
            // ignoreUnknownKeys = true allows extra fields in JSON (e.g., input_shape, output_shape)
            // that aren't defined in the Kotlin data class - this is safe because we only need
            // the normalization parameters, not the architectural metadata
            val jsonParser = Json { ignoreUnknownKeys = true }
            return jsonParser.decodeFromString<ModelMetadata>(cleanJson)
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to load metadata from $filename", e)
            throw RuntimeException("Metadata loading failed: ${e.message}", e)
        }
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
