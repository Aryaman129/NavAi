package com.navai.logger.service

import android.app.*
import android.content.Context
import android.content.Intent
import android.hardware.*
import android.location.Location
import android.os.*
import androidx.core.app.NotificationCompat
import com.google.android.gms.location.*
import com.navai.logger.R
import com.navai.logger.ml.SpeedPredictor
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlin.math.abs
import android.util.Log

/**
 * Service for real-time speed prediction using IMU sensors + TFLite model
 * Also collects GPS data for ground truth comparison
 */
class SpeedPredictionService : Service(), SensorEventListener {
    
    companion object {
        private const val TAG = "SpeedPredictionService"
        const val ACTION_START_PREDICTION = "START_PREDICTION"
        const val ACTION_STOP_PREDICTION = "STOP_PREDICTION"
        const val NOTIFICATION_ID = 2001
        const val CHANNEL_ID = "speed_prediction_channel"
        
        // Broadcast actions for UI updates
        const val BROADCAST_PREDICTION_UPDATE = "com.navai.logger.PREDICTION_UPDATE"
        const val EXTRA_PREDICTED_SPEED = "predicted_speed"
        const val EXTRA_GPS_SPEED = "gps_speed"
        const val EXTRA_INFERENCE_TIME = "inference_time"
        const val EXTRA_ERROR = "error"
        const val EXTRA_AVG_ERROR = "avg_error"
        const val EXTRA_SAMPLE_COUNT = "sample_count"
        const val EXTRA_STATE = "state" // "running", "stopped", "error"
        const val EXTRA_ERROR_MESSAGE = "error_message"
        
        private const val TARGET_SAMPLE_RATE_HZ = 100
        private const val GPS_UPDATE_INTERVAL_MS = 1000L  // 1 Hz for comparison
        
        // Shared state flow for UI updates (DEPRECATED - use broadcasts instead)
        private val _predictionState = MutableStateFlow<PredictionState>(PredictionState.Idle)
        val predictionState: StateFlow<PredictionState> = _predictionState.asStateFlow()
    }
    
    private lateinit var sensorManager: SensorManager
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private var speedPredictor: SpeedPredictor? = null
    
    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    private var isRunning = false
    private var startTime = 0L
    
    // Sensor references
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    
    // Latest sensor values (for synchronized updates)
    private var latestAccel: FloatArray? = null
    private var latestGyro: FloatArray? = null
    private var lastUpdateTime = 0L
    
    // GPS ground truth
    private var gpsSpeed: Float = 0f
    private var gpsAccuracy: Float = 0f
    
    // Performance tracking
    private var predictionCount = 0
    private var totalError = 0.0
    private var totalAbsError = 0.0
    
    // Location callback
    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            result.lastLocation?.let { location ->
                gpsSpeed = location.speed  // m/s
                gpsAccuracy = location.accuracy
            }
        }
    }
    
    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "🚀 SpeedPredictionService onCreate()")
        
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        
        // Initialize TFLite predictor
        try {
            speedPredictor = SpeedPredictor(this)
            Log.i(TAG, "✅ SpeedPredictor initialized successfully")
            
            // Verify predictor initialized correctly
            if (speedPredictor == null) {
                Log.e(TAG, "❌ SpeedPredictor returned null")
                _predictionState.value = PredictionState.Error(
                    message = "Failed to initialize speed predictor",
                    details = "SpeedPredictor returned null"
                )
                broadcastError("Failed to initialize speed predictor", "SpeedPredictor returned null")
            }
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error initializing TFLite model: ${e.message}", e)
            _predictionState.value = PredictionState.Error(
                message = "Error initializing TFLite model",
                details = "${e.javaClass.simpleName}: ${e.message}"
            )
            broadcastError("Error initializing TFLite model", "${e.javaClass.simpleName}: ${e.message}")
        }
        
        // Initialize sensors
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        
        // Verify sensors exist
        if (accelerometer == null) {
            Log.e(TAG, "❌ Accelerometer not available")
            _predictionState.value = PredictionState.Error(
                message = "Accelerometer not available",
                details = "Device does not have an accelerometer sensor"
            )
            broadcastError("Accelerometer not available", "Device does not have an accelerometer sensor")
        } else {
            Log.i(TAG, "✅ Accelerometer available")
        }
        if (gyroscope == null) {
            Log.e(TAG, "❌ Gyroscope not available")
            _predictionState.value = PredictionState.Error(
                message = "Gyroscope not available",
                details = "Device does not have a gyroscope sensor"
            )
            broadcastError("Gyroscope not available", "Device does not have a gyroscope sensor")
        } else {
            Log.i(TAG, "✅ Gyroscope available")
        }
        
        createNotificationChannel()
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.i(TAG, "📥 onStartCommand() called - action=${intent?.action}")
        when (intent?.action) {
            ACTION_START_PREDICTION -> {
                Log.i(TAG, "➡️ Starting prediction...")
                startPrediction()
            }
            ACTION_STOP_PREDICTION -> {
                Log.i(TAG, "⏹️ Stopping prediction...")
                stopPrediction()
            }
            else -> {
                Log.w(TAG, "⚠️ Unknown action: ${intent?.action}")
            }
        }
        return START_STICKY
    }
    
    override fun onBind(intent: Intent?): IBinder? = null
    
    /**
     * Broadcast prediction state to UI
     */
    private fun broadcastPredictionUpdate(
        predictedSpeed: Float,
        gpsSpeed: Float,
        inferenceTimeMs: Int,
        error: Float,
        avgError: Float,
        sampleCount: Int,
        state: String = "running"
    ) {
        val intent = Intent(BROADCAST_PREDICTION_UPDATE).apply {
            putExtra(EXTRA_PREDICTED_SPEED, predictedSpeed)
            putExtra(EXTRA_GPS_SPEED, gpsSpeed)
            putExtra(EXTRA_INFERENCE_TIME, inferenceTimeMs)
            putExtra(EXTRA_ERROR, error)
            putExtra(EXTRA_AVG_ERROR, avgError)
            putExtra(EXTRA_SAMPLE_COUNT, sampleCount)
            putExtra(EXTRA_STATE, state)
        }
        sendBroadcast(intent)
        Log.d(TAG, "📡 Broadcast sent: speed=$predictedSpeed km/h, GPS=$gpsSpeed km/h, error=$error m/s")
    }
    
    /**
     * Broadcast error state to UI
     */
    private fun broadcastError(message: String, details: String = "") {
        val intent = Intent(BROADCAST_PREDICTION_UPDATE).apply {
            putExtra(EXTRA_STATE, "error")
            putExtra(EXTRA_ERROR_MESSAGE, "$message: $details")
        }
        sendBroadcast(intent)
        Log.e(TAG, "❌ Broadcast error: $message - $details")
    }
    
    private fun startPrediction() {
        Log.i(TAG, "🎬 startPrediction() called - isRunning=$isRunning")
        if (isRunning) {
            Log.w(TAG, "⚠️ Already running, ignoring start request")
            return
        }
        
        isRunning = true
        startTime = System.currentTimeMillis()
        predictionCount = 0
        totalError = 0.0
        totalAbsError = 0.0
        
        Log.d(TAG, "🔄 Resetting predictor...")
        // Reset predictor
        speedPredictor?.reset()
        
        Log.d(TAG, "🔔 Starting foreground service...")
        // Start foreground service
        startForeground(NOTIFICATION_ID, createNotification("Starting speed prediction..."))
        
        // Register sensors at high frequency
        val sensorDelay = SensorManager.SENSOR_DELAY_FASTEST
        
        Log.d(TAG, "📊 Registering sensors...")
        accelerometer?.let { 
            sensorManager.registerListener(this, it, sensorDelay)
            Log.i(TAG, "✅ Accelerometer registered")
        } ?: Log.e(TAG, "❌ Accelerometer is null!")
        
        gyroscope?.let { 
            sensorManager.registerListener(this, it, sensorDelay)
            Log.i(TAG, "✅ Gyroscope registered")
        } ?: Log.e(TAG, "❌ Gyroscope is null!")
        
        Log.d(TAG, "📡 Starting GPS updates...")
        // Start GPS updates for ground truth
        startGpsUpdates()
        
        Log.d(TAG, "🔄 Starting prediction loop...")
        // Start prediction loop
        serviceScope.launch {
            predictionLoop()
        }
        
        _predictionState.value = PredictionState.Running(
            predictedSpeed = 0f,
            gpsSpeed = 0f,
            inferenceTimeMs = 0,
            error = 0f,
            avgError = 0f,
            sampleCount = 0
        )
        
        updateNotification("Speed prediction running...")
        Log.i(TAG, "✅ Speed prediction started successfully!")
    }
    
    private fun stopPrediction() {
        Log.i(TAG, "🛑 stopPrediction() called - isRunning=$isRunning")
        if (!isRunning) {
            Log.w(TAG, "⚠️ Not running, ignoring stop request")
            return
        }
        
        isRunning = false
        
        Log.d(TAG, "📊 Unregistering sensors...")
        // Unregister sensors
        sensorManager.unregisterListener(this)
        
        Log.d(TAG, "📡 Stopping GPS updates...")
        // Stop GPS updates
        fusedLocationClient.removeLocationUpdates(locationCallback)
        
        // Get final stats
        val stats = speedPredictor?.getStats()
        Log.i(TAG, "📊 Final stats: ${stats?.inferenceCount ?: 0} predictions, avg ${stats?.avgInferenceTimeMs ?: 0}ms")
        
        _predictionState.value = PredictionState.Stopped(
            avgInferenceTimeMs = stats?.avgInferenceTimeMs ?: 0,
            minInferenceTimeMs = stats?.minInferenceTimeMs ?: 0,
            maxInferenceTimeMs = stats?.maxInferenceTimeMs ?: 0,
            totalPredictions = predictionCount,
            avgAbsError = if (predictionCount > 0) (totalAbsError / predictionCount).toFloat() else 0f
        )
        
        // Stop foreground service
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }
    
    private fun startGpsUpdates() {
        // Check permission first
        if (checkSelfPermission(android.Manifest.permission.ACCESS_FINE_LOCATION) 
            != android.content.pm.PackageManager.PERMISSION_GRANTED) {
            Log.e(TAG, "❌ GPS permission NOT granted!")
            broadcastError("GPS permission denied", "Location permission is required for speed comparison")
            return
        }
        
        val locationRequest = LocationRequest.Builder(
            Priority.PRIORITY_HIGH_ACCURACY,
            GPS_UPDATE_INTERVAL_MS
        ).apply {
            setMinUpdateIntervalMillis(GPS_UPDATE_INTERVAL_MS)
            setMaxUpdateDelayMillis(GPS_UPDATE_INTERVAL_MS * 2)
        }.build()
        
        try {
            fusedLocationClient.requestLocationUpdates(
                locationRequest,
                locationCallback,
                Looper.getMainLooper()
            )
            Log.i(TAG, "✅ GPS updates started successfully")
        } catch (e: SecurityException) {
            Log.e(TAG, "❌ GPS SecurityException: ${e.message}", e)
            broadcastError("GPS error", e.message ?: "Unknown error")
        }
    }
    
    override fun onSensorChanged(event: SensorEvent) {
        if (!isRunning) return
        
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                latestAccel = event.values.clone()
            }
            Sensor.TYPE_GYROSCOPE -> {
                latestGyro = event.values.clone()
            }
        }
        
        // Add sample when both sensors are updated
        val accel = latestAccel
        val gyro = latestGyro
        
        if (accel != null && gyro != null) {
            val currentTime = System.nanoTime()
            
            // Throttle to ~100Hz to avoid overwhelming the predictor
            if (currentTime - lastUpdateTime >= 10_000_000) { // 10ms = 100Hz
                speedPredictor?.addSample(accel, gyro)
                lastUpdateTime = currentTime
            }
        }
    }
    
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Handle accuracy changes if needed
    }
    
    private suspend fun predictionLoop() {
        Log.i(TAG, "🔄 Prediction loop STARTED")
        var loopCount = 0
        
        while (isRunning) {
            loopCount++
            
            // Try to predict speed
            val result = speedPredictor?.predictSpeed()
            
            if (result != null) {
                predictionCount++
                
                // Calculate error vs GPS ground truth
                val error = result.speedMps - gpsSpeed
                val absError = abs(error)
                
                totalError += error
                totalAbsError += absError
                
                val avgError = if (predictionCount > 0) (totalAbsError / predictionCount).toFloat() else 0f
                
                // Broadcast update to UI (replaces StateFlow)
                broadcastPredictionUpdate(
                    predictedSpeed = result.speedKmh,
                    gpsSpeed = gpsSpeed * 3.6f,  // Convert to km/h
                    inferenceTimeMs = result.inferenceTimeMs.toInt(),
                    error = error,
                    avgError = avgError,
                    sampleCount = predictionCount
                )
                
                // Also update StateFlow for backward compatibility
                _predictionState.value = PredictionState.Running(
                    predictedSpeed = result.speedKmh,
                    gpsSpeed = gpsSpeed * 3.6f,
                    inferenceTimeMs = result.inferenceTimeMs,
                    error = error,
                    avgError = avgError,
                    sampleCount = predictionCount
                )
                
                // Log every 10th prediction
                if (predictionCount % 10 == 0) {
                    Log.d(TAG, "📊 Prediction #$predictionCount: ${String.format("%.1f", result.speedKmh)} km/h, " +
                            "GPS: ${String.format("%.1f", gpsSpeed * 3.6f)} km/h, error: ${String.format("%.2f", error)} m/s")
                }
                
                // Update notification
                if (predictionCount % 10 == 0) {
                    updateNotification(
                        "Speed: ${String.format("%.1f", result.speedKmh)} km/h " +
                        "(GPS: ${String.format("%.1f", gpsSpeed * 3.6f)} km/h) " +
                        "| Latency: ${result.inferenceTimeMs}ms"
                    )
                }
            }
            
            // Predict at ~10Hz (every 100ms)
            delay(100)
        }
        
        Log.i(TAG, "🛑 Prediction loop STOPPED after $loopCount iterations, $predictionCount predictions made")
    }
    
    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Speed Prediction",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "NavAI real-time speed prediction"
            setShowBadge(false)
        }
        
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.createNotificationChannel(channel)
    }
    
    private fun createNotification(text: String): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("NavAI Speed Prediction")
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
    }
    
    private fun updateNotification(text: String) {
        val notification = createNotification(text)
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(NOTIFICATION_ID, notification)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
        speedPredictor?.close()
        stopPrediction()
    }
}

/**
 * Prediction state for UI updates
 */
sealed class PredictionState {
    object Idle : PredictionState()
    
    data class Running(
        val predictedSpeed: Float,      // km/h
        val gpsSpeed: Float,             // km/h
        val inferenceTimeMs: Long,       // Latency
        val error: Float,                // Current error (m/s)
        val avgError: Float,             // Average absolute error (m/s)
        val sampleCount: Int             // Number of predictions
    ) : PredictionState()
    
    data class Stopped(
        val avgInferenceTimeMs: Long,
        val minInferenceTimeMs: Long,
        val maxInferenceTimeMs: Long,
        val totalPredictions: Int,
        val avgAbsError: Float
    ) : PredictionState()
    
    data class Error(
        val message: String,
        val details: String? = null
    ) : PredictionState()
}
