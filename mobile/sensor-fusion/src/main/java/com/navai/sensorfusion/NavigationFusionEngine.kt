package com.navai.sensorfusion

import android.content.Context
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.math.*

/**
 * Main navigation fusion engine that combines EKF, ML speed estimation, and sensor data
 */
class NavigationFusionEngine(
    private val context: Context,
    private val config: FusionConfig = FusionConfig()
) {
    
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    // Core components
    private val ekfEngine = EKFNavigationEngine()
    private val mlSpeedEstimator = MLSpeedEstimatorFactory.createDefault(context)
    
    // Data queues for processing
    private val imuQueue = ConcurrentLinkedQueue<IMUMeasurement>()
    private val gpsQueue = ConcurrentLinkedQueue<GPSMeasurement>()
    
    // State flows for real-time updates
    private val _navigationState = MutableStateFlow(NavigationState())
    val navigationState: StateFlow<NavigationState> = _navigationState.asStateFlow()
    
    private val _fusionStatus = MutableStateFlow(FusionStatus())
    val fusionStatus: StateFlow<FusionStatus> = _fusionStatus.asStateFlow()
    
    private var isRunning = false
    private var lastProcessTime = 0L
    
    init {
        startFusionLoop()
    }
    
    /**
     * Start the main fusion processing loop
     */
    private fun startFusionLoop() {
        scope.launch {
            isRunning = true
            
            while (isRunning) {
                try {
                    processSensorData()
                    delay(config.processingIntervalMs)
                } catch (e: Exception) {
                    // Log error but continue processing
                    updateFusionStatus(error = e.message)
                }
            }
        }
    }
    
    /**
     * Process queued sensor data
     */
    private suspend fun processSensorData() {
        val currentTime = System.nanoTime()
        
        // Process IMU data
        processIMUQueue()
        
        // Process GPS data
        processGPSQueue()
        
        // Generate ML speed estimates
        generateSpeedEstimate()
        
        // Update status
        updateFusionStatus(
            lastUpdateTime = currentTime,
            imuQueueSize = imuQueue.size,
            gpsQueueSize = gpsQueue.size
        )
        
        lastProcessTime = currentTime
    }
    
    /**
     * Process all queued IMU measurements
     */
    private fun processIMUQueue() {
        var processedCount = 0
        
        while (imuQueue.isNotEmpty() && processedCount < config.maxIMUBatchSize) {
            val imuData = imuQueue.poll() ?: break
            
            // Add to ML estimator window
            mlSpeedEstimator.addIMUMeasurement(imuData)
            
            // EKF prediction step
            ekfEngine.predict(imuData)
            
            processedCount++
        }
        
        if (processedCount > 0) {
            // Update navigation state
            _navigationState.value = ekfEngine.getCurrentState()
        }
    }
    
    /**
     * Process all queued GPS measurements
     */
    private fun processGPSQueue() {
        while (gpsQueue.isNotEmpty()) {
            val gpsData = gpsQueue.poll() ?: break
            
            // EKF update with GPS
            ekfEngine.updateWithGPS(gpsData)
            
            // Update navigation state
            _navigationState.value = ekfEngine.getCurrentState()
        }
    }
    
    /**
     * Generate and apply ML speed estimates
     */
    private fun generateSpeedEstimate() {
        val speedEstimate = mlSpeedEstimator.estimateSpeed()
        
        if (speedEstimate != null && speedEstimate.confidence > config.minSpeedConfidence) {
            // Apply speed estimate to EKF
            ekfEngine.updateWithSpeedEstimate(speedEstimate)
            
            // Update navigation state
            _navigationState.value = ekfEngine.getCurrentState()
        }
    }
    
    /**
     * Add IMU measurement to processing queue
     */
    fun addIMUMeasurement(measurement: IMUMeasurement) {
        if (imuQueue.size < config.maxQueueSize) {
            imuQueue.offer(measurement)
        } else {
            // Queue full, remove oldest
            imuQueue.poll()
            imuQueue.offer(measurement)
        }
    }
    
    /**
     * Add GPS measurement to processing queue
     */
    fun addGPSMeasurement(measurement: GPSMeasurement) {
        if (gpsQueue.size < config.maxQueueSize) {
            gpsQueue.offer(measurement)
        } else {
            // Queue full, remove oldest
            gpsQueue.poll()
            gpsQueue.offer(measurement)
        }
    }
    
    /**
     * Reset navigation state to initial position
     */
    fun reset(initialState: NavigationState = NavigationState()) {
        scope.launch {
            // Clear queues
            imuQueue.clear()
            gpsQueue.clear()
            
            // Reset EKF
            // Note: Would need to add reset method to EKFNavigationEngine
            
            // Update state
            _navigationState.value = initialState
            
            updateFusionStatus(reset = true)
        }
    }
    
    /**
     * Get current position in lat/lon coordinates
     */
    fun getCurrentLatLon(): LatLon? {
        val state = _navigationState.value
        
        // Convert from local coordinates to lat/lon
        // This would require a reference point and coordinate transformation
        // For now, return null if no GPS reference is available
        return null
    }
    
    /**
     * Update fusion status
     */
    private fun updateFusionStatus(
        lastUpdateTime: Long = _fusionStatus.value.lastUpdateTime,
        imuQueueSize: Int = _fusionStatus.value.imuQueueSize,
        gpsQueueSize: Int = _fusionStatus.value.gpsQueueSize,
        error: String? = null,
        reset: Boolean = false
    ) {
        val currentStatus = _fusionStatus.value
        
        _fusionStatus.value = FusionStatus(
            isRunning = isRunning,
            lastUpdateTime = lastUpdateTime,
            imuQueueSize = imuQueueSize,
            gpsQueueSize = gpsQueueSize,
            processingRate = calculateProcessingRate(),
            lastError = error ?: (if (reset) null else currentStatus.lastError),
            mlModelLoaded = true // Assume loaded if no error
        )
    }
    
    /**
     * Calculate current processing rate
     */
    private fun calculateProcessingRate(): Double {
        val currentTime = System.nanoTime()
        return if (lastProcessTime > 0) {
            1e9 / (currentTime - lastProcessTime)
        } else {
            0.0
        }
    }
    
    /**
     * Stop the fusion engine
     */
    fun stop() {
        isRunning = false
        scope.cancel()
        mlSpeedEstimator.close()
    }
    
    /**
     * Get performance metrics
     */
    fun getPerformanceMetrics(): PerformanceMetrics {
        val state = _navigationState.value
        val status = _fusionStatus.value
        
        return PerformanceMetrics(
            speed = state.speed,
            heading = state.heading,
            uncertainty = state.uncertainty,
            processingRate = status.processingRate,
            queueUtilization = (status.imuQueueSize + status.gpsQueueSize) / (2.0 * config.maxQueueSize),
            mlConfidence = 0.8 // Would need to track this from ML estimator
        )
    }
}

/**
 * Configuration for the fusion engine
 */
data class FusionConfig(
    val processingIntervalMs: Long = 10, // 100Hz processing
    val maxQueueSize: Int = 1000,
    val maxIMUBatchSize: Int = 10,
    val minSpeedConfidence: Double = 0.5
)

/**
 * Status information for the fusion engine
 */
data class FusionStatus(
    val isRunning: Boolean = false,
    val lastUpdateTime: Long = 0,
    val imuQueueSize: Int = 0,
    val gpsQueueSize: Int = 0,
    val processingRate: Double = 0.0,
    val lastError: String? = null,
    val mlModelLoaded: Boolean = false
)

/**
 * Performance metrics for monitoring
 */
data class PerformanceMetrics(
    val speed: Double,
    val heading: Double,
    val uncertainty: Double,
    val processingRate: Double,
    val queueUtilization: Double,
    val mlConfidence: Double
)

/**
 * Latitude/Longitude coordinates
 */
data class LatLon(
    val latitude: Double,
    val longitude: Double
)

/**
 * Factory for creating navigation fusion engines
 */
object NavigationFusionEngineFactory {
    
    fun createDefault(context: Context): NavigationFusionEngine {
        return NavigationFusionEngine(
            context = context,
            config = FusionConfig()
        )
    }
    
    fun createHighFrequency(context: Context): NavigationFusionEngine {
        return NavigationFusionEngine(
            context = context,
            config = FusionConfig(
                processingIntervalMs = 5, // 200Hz processing
                maxQueueSize = 2000,
                maxIMUBatchSize = 20
            )
        )
    }
    
    fun createLowPower(context: Context): NavigationFusionEngine {
        return NavigationFusionEngine(
            context = context,
            config = FusionConfig(
                processingIntervalMs = 50, // 20Hz processing
                maxQueueSize = 200,
                maxIMUBatchSize = 5
            )
        )
    }
}
