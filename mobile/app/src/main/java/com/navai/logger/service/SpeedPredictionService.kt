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

/**
 * Service for real-time speed prediction using IMU sensors + TFLite model
 * Also collects GPS data for ground truth comparison
 */
class SpeedPredictionService : Service(), SensorEventListener {
    
    companion object {
        const val ACTION_START_PREDICTION = "START_PREDICTION"
        const val ACTION_STOP_PREDICTION = "STOP_PREDICTION"
        const val NOTIFICATION_ID = 2001
        const val CHANNEL_ID = "speed_prediction_channel"
        
        private const val TARGET_SAMPLE_RATE_HZ = 100
        private const val GPS_UPDATE_INTERVAL_MS = 1000L  // 1 Hz for comparison
        
        // Shared state flow for UI updates
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
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        
        // Initialize TFLite predictor
        try {
            speedPredictor = SpeedPredictor(this)
            
            // Verify predictor initialized correctly
            if (speedPredictor == null) {
                _predictionState.value = PredictionState.Error(
                    message = "Failed to initialize speed predictor",
                    details = "SpeedPredictor returned null"
                )
            }
        } catch (e: Exception) {
            _predictionState.value = PredictionState.Error(
                message = "Error initializing TFLite model",
                details = "${e.javaClass.simpleName}: ${e.message}"
            )
        }
        
        // Initialize sensors
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        
        // Verify sensors exist
        if (accelerometer == null) {
            _predictionState.value = PredictionState.Error(
                message = "Accelerometer not available",
                details = "Device does not have an accelerometer sensor"
            )
        }
        if (gyroscope == null) {
            _predictionState.value = PredictionState.Error(
                message = "Gyroscope not available",
                details = "Device does not have a gyroscope sensor"
            )
        }
        
        createNotificationChannel()
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_PREDICTION -> startPrediction()
            ACTION_STOP_PREDICTION -> stopPrediction()
        }
        return START_STICKY
    }
    
    override fun onBind(intent: Intent?): IBinder? = null
    
    private fun startPrediction() {
        if (isRunning) return
        
        isRunning = true
        startTime = System.currentTimeMillis()
        predictionCount = 0
        totalError = 0.0
        totalAbsError = 0.0
        
        // Reset predictor
        speedPredictor?.reset()
        
        // Start foreground service
        startForeground(NOTIFICATION_ID, createNotification("Starting speed prediction..."))
        
        // Register sensors at high frequency
        val sensorDelay = SensorManager.SENSOR_DELAY_FASTEST
        
        accelerometer?.let { 
            sensorManager.registerListener(this, it, sensorDelay)
        }
        gyroscope?.let { 
            sensorManager.registerListener(this, it, sensorDelay)
        }
        
        // Start GPS updates for ground truth
        startGpsUpdates()
        
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
    }
    
    private fun stopPrediction() {
        if (!isRunning) return
        
        isRunning = false
        
        // Unregister sensors
        sensorManager.unregisterListener(this)
        
        // Stop GPS updates
        fusedLocationClient.removeLocationUpdates(locationCallback)
        
        // Get final stats
        val stats = speedPredictor?.getStats()
        
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
        } catch (e: SecurityException) {
            // Handle permission not granted
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
        while (isRunning) {
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
                
                // Update state for UI
                _predictionState.value = PredictionState.Running(
                    predictedSpeed = result.speedKmh,
                    gpsSpeed = gpsSpeed * 3.6f,  // Convert to km/h
                    inferenceTimeMs = result.inferenceTimeMs,
                    error = error,
                    avgError = avgError,
                    sampleCount = predictionCount
                )
                
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
