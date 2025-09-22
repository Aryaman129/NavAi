package com.navai.logger.integration

import android.app.Service
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.IBinder
import android.os.Binder
import androidx.core.app.NotificationCompat
import com.navai.logger.R
import com.navai.sensorfusion.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

/**
 * Main navigation service that integrates sensor logging with real-time navigation
 */
class NavigationService : Service(), SensorEventListener {
    
    companion object {
        const val NOTIFICATION_ID = 2001
        const val ACTION_START_NAVIGATION = "START_NAVIGATION"
        const val ACTION_STOP_NAVIGATION = "STOP_NAVIGATION"
    }
    
    private val binder = NavigationBinder()
    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    // Core components
    private lateinit var sensorManager: SensorManager
    private lateinit var fusionEngine: NavigationFusionEngine
    
    // Sensors
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var magnetometer: Sensor? = null
    private var rotationVector: Sensor? = null
    
    // State
    private var isNavigating = false
    private val _navigationState = MutableStateFlow(NavigationState())
    val navigationState: StateFlow<NavigationState> = _navigationState.asStateFlow()
    
    private val _performanceMetrics = MutableStateFlow(PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    val performanceMetrics: StateFlow<PerformanceMetrics> = _performanceMetrics.asStateFlow()
    
    inner class NavigationBinder : Binder() {
        fun getService(): NavigationService = this@NavigationService
    }
    
    override fun onCreate() {
        super.onCreate()
        
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        fusionEngine = NavigationFusionEngineFactory.createDefault(this)
        
        // Initialize sensors
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        
        // Start monitoring fusion engine state
        serviceScope.launch {
            fusionEngine.navigationState.collect { state ->
                _navigationState.value = state
            }
        }
        
        serviceScope.launch {
            while (isActive) {
                _performanceMetrics.value = fusionEngine.getPerformanceMetrics()
                delay(1000) // Update every second
            }
        }
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_NAVIGATION -> startNavigation()
            ACTION_STOP_NAVIGATION -> stopNavigation()
        }
        return START_STICKY
    }
    
    override fun onBind(intent: Intent?): IBinder = binder
    
    private fun startNavigation() {
        if (isNavigating) return
        
        isNavigating = true
        
        // Start foreground service
        val notification = NotificationCompat.Builder(this, "navigation_channel")
            .setContentTitle("NavAI Navigation")
            .setContentText("Real-time navigation active")
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setOngoing(true)
            .build()
        
        startForeground(NOTIFICATION_ID, notification)
        
        // Register sensors
        val sensorDelay = SensorManager.SENSOR_DELAY_FASTEST
        
        accelerometer?.let { 
            sensorManager.registerListener(this, it, sensorDelay)
        }
        gyroscope?.let { 
            sensorManager.registerListener(this, it, sensorDelay)
        }
        magnetometer?.let { 
            sensorManager.registerListener(this, it, sensorDelay)
        }
        rotationVector?.let { 
            sensorManager.registerListener(this, it, sensorDelay)
        }
    }
    
    private fun stopNavigation() {
        if (!isNavigating) return
        
        isNavigating = false
        
        // Unregister sensors
        sensorManager.unregisterListener(this)
        
        // Stop foreground service
        stopForeground(STOP_FOREGROUND_REMOVE)
    }
    
    override fun onSensorChanged(event: SensorEvent) {
        if (!isNavigating) return
        
        val timestamp = event.timestamp
        
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER,
            Sensor.TYPE_GYROSCOPE -> {
                // Create IMU measurement
                val imuMeasurement = createIMUMeasurement(event)
                if (imuMeasurement != null) {
                    fusionEngine.addIMUMeasurement(imuMeasurement)
                }
            }
        }
    }
    
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Handle accuracy changes if needed
    }
    
    private fun createIMUMeasurement(event: SensorEvent): IMUMeasurement? {
        // We need to collect data from multiple sensors to create a complete IMU measurement
        // This is a simplified version - in practice, you'd buffer sensor data and combine them
        
        return when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                IMUMeasurement(
                    timestamp = event.timestamp,
                    accelX = event.values[0].toDouble(),
                    accelY = event.values[1].toDouble(),
                    accelZ = event.values[2].toDouble(),
                    gyroX = 0.0, // Would need to get from gyroscope
                    gyroY = 0.0,
                    gyroZ = 0.0
                )
            }
            else -> null
        }
    }
    
    /**
     * Add GPS measurement from location services
     */
    fun addGPSMeasurement(latitude: Double, longitude: Double, accuracy: Float) {
        // Convert lat/lon to local coordinates (simplified)
        val x = longitude * 111320.0 // Rough conversion
        val y = latitude * 110540.0
        
        val gpsMeasurement = GPSMeasurement(
            timestamp = System.nanoTime(),
            x = x,
            y = y,
            accuracy = accuracy.toDouble()
        )
        
        fusionEngine.addGPSMeasurement(gpsMeasurement)
    }
    
    /**
     * Reset navigation to initial state
     */
    fun resetNavigation() {
        fusionEngine.reset()
    }
    
    /**
     * Get current navigation state
     */
    fun getCurrentNavigationState(): NavigationState {
        return _navigationState.value
    }
    
    /**
     * Get performance metrics
     */
    fun getCurrentPerformanceMetrics(): PerformanceMetrics {
        return _performanceMetrics.value
    }
    
    override fun onDestroy() {
        super.onDestroy()
        stopNavigation()
        fusionEngine.stop()
        serviceScope.cancel()
    }
}
