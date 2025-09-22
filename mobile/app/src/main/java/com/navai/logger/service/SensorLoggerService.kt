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
import com.navai.logger.data.CsvWriter
import com.navai.logger.data.SensorData
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.math.sqrt

class SensorLoggerService : Service(), SensorEventListener {
    
    companion object {
        const val ACTION_START_LOGGING = "START_LOGGING"
        const val ACTION_STOP_LOGGING = "STOP_LOGGING"
        const val NOTIFICATION_ID = 1001
        const val CHANNEL_ID = "sensor_logger_channel"
        
        private const val TARGET_SAMPLE_RATE_HZ = 100
        private const val GPS_UPDATE_INTERVAL_MS = 200L
        private const val BATCH_WRITE_SIZE = 100
    }
    
    private lateinit var sensorManager: SensorManager
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private var csvWriter: CsvWriter? = null
    
    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val sensorDataQueue = ConcurrentLinkedQueue<SensorData>()
    
    private var isLogging = false
    private var startTime = 0L
    private var sampleCount = 0L
    
    // Sensor references
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var magnetometer: Sensor? = null
    private var rotationVector: Sensor? = null
    
    // Location callback
    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            result.lastLocation?.let { location ->
                val timestamp = System.nanoTime()
                val sensorData = SensorData.GpsData(
                    timestamp = timestamp,
                    latitude = location.latitude,
                    longitude = location.longitude,
                    speed = location.speed,
                    accuracy = location.accuracy,
                    bearing = location.bearing
                )
                sensorDataQueue.offer(sensorData)
            }
        }
    }
    
    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        
        // Initialize sensors
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        
        createNotificationChannel()
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_LOGGING -> startLogging()
            ACTION_STOP_LOGGING -> stopLogging()
        }
        return START_STICKY
    }
    
    override fun onBind(intent: Intent?): IBinder? = null
    
    private fun startLogging() {
        if (isLogging) return
        
        isLogging = true
        startTime = System.currentTimeMillis()
        sampleCount = 0
        
        // Initialize CSV writer
        csvWriter = CsvWriter(this)
        
        // Start foreground service
        startForeground(NOTIFICATION_ID, createNotification("Starting sensor logging..."))
        
        // Register sensors with high frequency
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
        
        // Start GPS updates
        startGpsUpdates()
        
        // Start data writing coroutine
        serviceScope.launch {
            writeDataLoop()
        }
        
        // Update notification
        updateNotification("Logging sensors at ${TARGET_SAMPLE_RATE_HZ}Hz")
    }
    
    private fun stopLogging() {
        if (!isLogging) return
        
        isLogging = false
        
        // Unregister sensors
        sensorManager.unregisterListener(this)
        
        // Stop GPS updates
        fusedLocationClient.removeLocationUpdates(locationCallback)
        
        // Write remaining data and close file
        serviceScope.launch {
            writeRemainingData()
            csvWriter?.close()
            csvWriter = null
            
            // Stop foreground service
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
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
        if (!isLogging) return
        
        val timestamp = event.timestamp
        val sensorData = when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> SensorData.AccelerometerData(
                timestamp = timestamp,
                x = event.values[0],
                y = event.values[1],
                z = event.values[2]
            )
            Sensor.TYPE_GYROSCOPE -> SensorData.GyroscopeData(
                timestamp = timestamp,
                x = event.values[0],
                y = event.values[1],
                z = event.values[2]
            )
            Sensor.TYPE_MAGNETIC_FIELD -> SensorData.MagnetometerData(
                timestamp = timestamp,
                x = event.values[0],
                y = event.values[1],
                z = event.values[2]
            )
            Sensor.TYPE_ROTATION_VECTOR -> SensorData.RotationVectorData(
                timestamp = timestamp,
                x = event.values[0],
                y = event.values[1],
                z = event.values[2],
                w = event.values.getOrNull(3) ?: 0f
            )
            else -> return
        }
        
        sensorDataQueue.offer(sensorData)
        sampleCount++
        
        // Update notification periodically
        if (sampleCount % 1000 == 0L) {
            val duration = (System.currentTimeMillis() - startTime) / 1000
            updateNotification("Logged ${sampleCount} samples (${duration}s)")
        }
    }
    
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Handle accuracy changes if needed
    }
    
    private suspend fun writeDataLoop() {
        val batch = mutableListOf<SensorData>()
        
        while (isLogging) {
            // Collect batch of data
            while (batch.size < BATCH_WRITE_SIZE && sensorDataQueue.isNotEmpty()) {
                sensorDataQueue.poll()?.let { batch.add(it) }
            }
            
            // Write batch to file
            if (batch.isNotEmpty()) {
                csvWriter?.writeBatch(batch)
                batch.clear()
            }
            
            // Small delay to prevent busy waiting
            delay(10)
        }
    }
    
    private suspend fun writeRemainingData() {
        val remaining = mutableListOf<SensorData>()
        while (sensorDataQueue.isNotEmpty()) {
            sensorDataQueue.poll()?.let { remaining.add(it) }
        }
        
        if (remaining.isNotEmpty()) {
            csvWriter?.writeBatch(remaining)
        }
    }
    
    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Sensor Logger",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "NavAI sensor data logging"
            setShowBadge(false)
        }
        
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.createNotificationChannel(channel)
    }
    
    private fun createNotification(text: String): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("NavAI Sensor Logger")
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
        stopLogging()
    }
}
