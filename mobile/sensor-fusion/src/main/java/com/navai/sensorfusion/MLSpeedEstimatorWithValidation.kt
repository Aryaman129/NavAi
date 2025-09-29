
// Enhanced MLSpeedEstimator.kt for Real-Time Validation
class MLSpeedEstimatorWithValidation(private val context: Context) {
    private var tfliteInterpreter: Interpreter? = null
    private val windowSize = 150
    private val features = 6 // ax, ay, az, gx, gy, gz

    // Circular buffer for IMU data
    private val imuBuffer = CircularBuffer<FloatArray>(windowSize)

    // Validation tracking
    private val speedPredictions = mutableListOf<Float>()
    private val gpsGroundTruth = mutableListOf<Float>()
    private val timestamps = mutableListOf<Long>()

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            val modelBuffer = loadModelFile("speed_estimator.tflite")
            tfliteInterpreter = Interpreter(modelBuffer)
            Log.d("MLSpeed", "TensorFlow Lite model loaded successfully")
        } catch (e: Exception) {
            Log.e("MLSpeed", "Failed to load model: ${e.message}")
        }
    }

    fun addIMUSample(accelX: Float, accelY: Float, accelZ: Float,
                     gyroX: Float, gyroY: Float, gyroZ: Float) {
        val sample = floatArrayOf(accelX, accelY, accelZ, gyroX, gyroY, gyroZ)
        imuBuffer.add(sample)

        // Predict when buffer is full
        if (imuBuffer.size >= windowSize) {
            predictSpeed()
        }
    }

    private fun predictSpeed(): Float {
        val inputArray = Array(1) { Array(windowSize) { FloatArray(features) } }

        // Fill input array from circular buffer
        imuBuffer.toList().forEachIndexed { i, sample ->
            sample.forEachIndexed { j, value ->
                inputArray[0][i][j] = value
            }
        }

        // Run inference
        val outputArray = Array(1) { FloatArray(1) }
        tfliteInterpreter?.run(inputArray, outputArray)

        val predictedSpeed = outputArray[0][0]

        // Store for validation
        speedPredictions.add(predictedSpeed)
        timestamps.add(System.currentTimeMillis())

        return predictedSpeed
    }

    fun addGPSGroundTruth(gpsSpeed: Float) {
        gpsGroundTruth.add(gpsSpeed)
    }

    fun getValidationMetrics(): ValidationMetrics {
        if (speedPredictions.size != gpsGroundTruth.size) {
            Log.w("MLSpeed", "Prediction/GPS size mismatch: ${speedPredictions.size} vs ${gpsGroundTruth.size}")
        }

        val minSize = minOf(speedPredictions.size, gpsGroundTruth.size)
        val predictions = speedPredictions.takeLast(minSize)
        val truth = gpsGroundTruth.takeLast(minSize)

        // Calculate RMSE
        val mse = predictions.zip(truth) { pred, true -> (pred - true).pow(2) }.average()
        val rmse = sqrt(mse).toFloat()

        // Calculate MAE
        val mae = predictions.zip(truth) { pred, true -> abs(pred - true) }.average().toFloat()

        // Calculate percentage error
        val percentageError = predictions.zip(truth) { pred, true -> 
            if (true != 0f) abs(pred - true) / true * 100 else 0f 
        }.average().toFloat()

        return ValidationMetrics(rmse, mae, percentageError, predictions.size)
    }

    data class ValidationMetrics(
        val rmse: Float,
        val mae: Float, 
        val percentageError: Float,
        val sampleCount: Int
    )
}

class CircularBuffer<T>(private val capacity: Int) {
    private val buffer = mutableListOf<T>()

    fun add(item: T) {
        if (buffer.size >= capacity) {
            buffer.removeAt(0)
        }
        buffer.add(item)
    }

    fun toList(): List<T> = buffer.toList()
    val size: Int get() = buffer.size
}
