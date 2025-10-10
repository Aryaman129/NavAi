package com.navai.sensorfusion

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ArrayBlockingQueue
import kotlin.math.sqrt

/**
 * TensorFlow Lite-based speed estimator using IMU data
 */
class MLSpeedEstimator(
    private val context: Context,
    private val modelPath: String = "speed_estimator.tflite",
    private val windowSizeSamples: Int = 150, // 1.5 seconds at 100Hz
    private val useGPU: Boolean = true
) {
    
    private var interpreter: Interpreter? = null
    private var inputBuffer: ByteBuffer? = null
    private var outputBuffer: ByteBuffer? = null
    
    private val sensorWindow = ArrayBlockingQueue<IMUMeasurement>(windowSizeSamples)
    private var isInitialized = false
    
    // Feature normalization parameters (should match training data)
    private val featureMeans = floatArrayOf(0.0f, 0.0f, -9.81f, 0.0f, 0.0f, 0.0f)
    private val featureStds = floatArrayOf(2.0f, 2.0f, 2.0f, 0.5f, 0.5f, 0.5f)
    
    init {
        initializeModel()
    }
    
    private fun initializeModel() {
        try {
            val modelBuffer = loadModelFile()
            val options = Interpreter.Options()
            
            // Configure GPU acceleration if available
            if (useGPU && CompatibilityList().isDelegateSupportedOnThisDevice) {
                val gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
            }
            
            // Use multiple threads for CPU inference
            options.setNumThreads(4)
            
            interpreter = Interpreter(modelBuffer, options)
            
            // Allocate input/output buffers
            allocateBuffers()
            
            isInitialized = true
            
        } catch (e: Exception) {
            throw RuntimeException("Failed to initialize ML speed estimator: ${e.message}", e)
        }
    }
    
    private fun loadModelFile(): MappedByteBuffer {
        return try {
            // Try to load from assets first
            val assetFileDescriptor = context.assets.openFd(modelPath)
            val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        } catch (e: Exception) {
            throw RuntimeException("Failed to load model file: $modelPath", e)
        }
    }
    
    private fun allocateBuffers() {
        val interpreter = this.interpreter ?: throw IllegalStateException("Interpreter not initialized")
        
        // Input buffer: [1, windowSizeSamples, 6] float32
        val inputShape = interpreter.getInputTensor(0).shape()
        val inputSize = inputShape.fold(1) { acc, dim -> acc * dim }
        inputBuffer = ByteBuffer.allocateDirect(inputSize * 4) // 4 bytes per float
        inputBuffer?.order(ByteOrder.nativeOrder())
        
        // Output buffer: [1, 1] float32
        val outputShape = interpreter.getOutputTensor(0).shape()
        val outputSize = outputShape.fold(1) { acc, dim -> acc * dim }
        outputBuffer = ByteBuffer.allocateDirect(outputSize * 4)
        outputBuffer?.order(ByteOrder.nativeOrder())
    }
    
    /**
     * Add IMU measurement to the sliding window
     */
    fun addIMUMeasurement(measurement: IMUMeasurement) {
        if (!isInitialized) return
        
        // Add to window, removing oldest if full
        if (sensorWindow.size >= windowSizeSamples) {
            sensorWindow.poll()
        }
        sensorWindow.offer(measurement)
    }
    
    /**
     * Estimate speed from current sensor window
     * Returns null if insufficient data
     */
    fun estimateSpeed(): SpeedMeasurement? {
        if (!isInitialized || sensorWindow.size < windowSizeSamples) {
            return null
        }
        
        val interpreter = this.interpreter ?: return null
        val inputBuffer = this.inputBuffer ?: return null
        val outputBuffer = this.outputBuffer ?: return null
        
        try {
            // Prepare input data
            inputBuffer.rewind()
            
            val windowData = sensorWindow.toList()
            for (i in 0 until windowSizeSamples) {
                val measurement = windowData[i]
                
                // Normalize features
                val features = floatArrayOf(
                    (measurement.accelX.toFloat() - featureMeans[0]) / featureStds[0],
                    (measurement.accelY.toFloat() - featureMeans[1]) / featureStds[1],
                    (measurement.accelZ.toFloat() - featureMeans[2]) / featureStds[2],
                    (measurement.gyroX.toFloat() - featureMeans[3]) / featureStds[3],
                    (measurement.gyroY.toFloat() - featureMeans[4]) / featureStds[4],
                    (measurement.gyroZ.toFloat() - featureMeans[5]) / featureStds[5]
                )
                
                for (feature in features) {
                    inputBuffer.putFloat(feature)
                }
            }
            
            // Run inference
            outputBuffer.rewind()
            interpreter.run(inputBuffer, outputBuffer)
            
            // Extract result
            outputBuffer.rewind()
            val speedEstimate = outputBuffer.float.toDouble()
            
            // Calculate confidence based on recent speed consistency
            val confidence = calculateConfidence(speedEstimate)
            
            val latestTimestamp = windowData.last().timestamp
            
            return SpeedMeasurement(
                timestamp = latestTimestamp,
                speed = maxOf(speedEstimate, 0.0), // Ensure non-negative
                confidence = confidence
            )
            
        } catch (e: Exception) {
            // Log error but don't crash
            return null
        }
    }
    
    /**
     * Calculate confidence based on speed estimate consistency
     */
    private fun calculateConfidence(currentSpeed: Double): Double {
        // Simple confidence metric based on acceleration consistency
        if (sensorWindow.size < windowSizeSamples) return 0.5
        
        val windowData = sensorWindow.toList()
        
        // Calculate acceleration magnitude variance
        val accelMagnitudes = windowData.map { measurement ->
            sqrt(
                measurement.accelX * measurement.accelX +
                measurement.accelY * measurement.accelY +
                measurement.accelZ * measurement.accelZ
            )
        }
        
        val meanAccel = accelMagnitudes.average()
        val accelVariance = accelMagnitudes.map { (it - meanAccel) * (it - meanAccel) }.average()
        
        // Lower variance = higher confidence
        val baseConfidence = 1.0 / (1.0 + accelVariance / 10.0)
        
        // Adjust confidence based on speed magnitude
        val speedConfidence = when {
            currentSpeed < 0.5 -> 0.7 // Lower confidence at very low speeds
            currentSpeed > 30.0 -> 0.8 // Slightly lower confidence at high speeds
            else -> 1.0
        }
        
        return (baseConfidence * speedConfidence).coerceIn(0.1, 1.0)
    }
    
    /**
     * Update feature normalization parameters
     */
    fun updateNormalizationParameters(means: FloatArray, stds: FloatArray) {
        if (means.size == 6 && stds.size == 6) {
            means.copyInto(featureMeans)
            stds.copyInto(featureStds)
        }
    }
    
    /**
     * Get model information
     */
    fun getModelInfo(): ModelInfo? {
        val interpreter = this.interpreter ?: return null
        
        return ModelInfo(
            inputShape = interpreter.getInputTensor(0).shape(),
            outputShape = interpreter.getOutputTensor(0).shape(),
            inputType = interpreter.getInputTensor(0).dataType().toString(),
            outputType = interpreter.getOutputTensor(0).dataType().toString()
        )
    }
    
    /**
     * Clean up resources
     */
    fun close() {
        interpreter?.close()
        interpreter = null
        inputBuffer = null
        outputBuffer = null
        sensorWindow.clear()
        isInitialized = false
    }
    
    data class ModelInfo(
        val inputShape: IntArray,
        val outputShape: IntArray,
        val inputType: String,
        val outputType: String
    )
}

/**
 * Factory for creating ML speed estimators with different configurations
 */
object MLSpeedEstimatorFactory {
    
    fun createDefault(context: Context): MLSpeedEstimator {
        return MLSpeedEstimator(
            context = context,
            modelPath = "speed_estimator.tflite",
            windowSizeSamples = 150,
            useGPU = true
        )
    }
    
    fun createCPUOnly(context: Context): MLSpeedEstimator {
        return MLSpeedEstimator(
            context = context,
            modelPath = "speed_estimator.tflite",
            windowSizeSamples = 150,
            useGPU = false
        )
    }
    
    fun createCustom(
        context: Context,
        modelPath: String,
        windowSizeSec: Float,
        sampleRate: Int = 100,
        useGPU: Boolean = true
    ): MLSpeedEstimator {
        return MLSpeedEstimator(
            context = context,
            modelPath = modelPath,
            windowSizeSamples = (windowSizeSec * sampleRate).toInt(),
            useGPU = useGPU
        )
    }
}
