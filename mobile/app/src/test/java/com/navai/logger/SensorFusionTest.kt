package com.navai.logger

import com.navai.sensorfusion.*
import org.junit.Test
import org.junit.Assert.*
import org.junit.Before
import kotlin.math.*

/**
 * Unit tests for sensor fusion components
 */
class SensorFusionTest {
    
    private lateinit var ekfEngine: EKFNavigationEngine
    
    @Before
    fun setup() {
        ekfEngine = EKFNavigationEngine()
    }
    
    @Test
    fun testEKFInitialization() {
        val initialState = NavigationState(x = 10.0, y = 20.0, vx = 1.0, vy = 2.0)
        val ekf = EKFNavigationEngine(initialState)
        
        val state = ekf.getCurrentState()
        assertEquals(10.0, state.x, 0.001)
        assertEquals(20.0, state.y, 0.001)
        assertEquals(1.0, state.vx, 0.001)
        assertEquals(2.0, state.vy, 0.001)
    }
    
    @Test
    fun testIMUPrediction() {
        val initialTime = System.nanoTime()
        
        // First prediction to initialize
        val imu1 = IMUMeasurement(
            timestamp = initialTime,
            accelX = 1.0, accelY = 0.0, accelZ = -9.81,
            gyroX = 0.0, gyroY = 0.0, gyroZ = 0.1
        )
        ekfEngine.predict(imu1)
        
        // Second prediction with time step
        val imu2 = IMUMeasurement(
            timestamp = initialTime + 10_000_000, // 10ms later
            accelX = 1.0, accelY = 0.0, accelZ = -9.81,
            gyroX = 0.0, gyroY = 0.0, gyroZ = 0.1
        )
        ekfEngine.predict(imu2)
        
        val state = ekfEngine.getCurrentState()
        
        // Should have some velocity from acceleration
        assertTrue("Velocity should increase", state.vx > 0)
        
        // Should have some yaw from gyroscope
        assertTrue("Yaw should change", abs(state.yaw) > 0)
    }
    
    @Test
    fun testSpeedMeasurementUpdate() {
        // Initialize with some velocity
        val initialState = NavigationState(vx = 5.0, vy = 0.0)
        val ekf = EKFNavigationEngine(initialState)
        
        // Apply speed measurement
        val speedMeasurement = SpeedMeasurement(
            timestamp = System.nanoTime(),
            speed = 10.0,
            confidence = 1.0
        )
        
        ekf.updateWithSpeedEstimate(speedMeasurement)
        
        val state = ekf.getCurrentState()
        val actualSpeed = sqrt(state.vx * state.vx + state.vy * state.vy)
        
        // Speed should be closer to measurement
        assertTrue("Speed should be updated", actualSpeed > 5.0)
    }
    
    @Test
    fun testGPSUpdate() {
        // Initialize at origin
        val ekf = EKFNavigationEngine()
        
        // Apply GPS measurement
        val gpsMeasurement = GPSMeasurement(
            timestamp = System.nanoTime(),
            x = 100.0,
            y = 200.0,
            accuracy = 5.0
        )
        
        ekf.updateWithGPS(gpsMeasurement)
        
        val state = ekf.getCurrentState()
        
        // Position should be updated towards GPS measurement
        assertTrue("X position should be updated", abs(state.x - 100.0) < 50.0)
        assertTrue("Y position should be updated", abs(state.y - 200.0) < 50.0)
    }
    
    @Test
    fun testNavigationStateCalculations() {
        val state = NavigationState(vx = 3.0, vy = 4.0, yaw = PI/4)
        
        // Test speed calculation
        assertEquals(5.0, state.speed, 0.001)
        
        // Test heading
        assertEquals(PI/4, state.heading, 0.001)
    }
    
    @Test
    fun testIMUMeasurementCreation() {
        val timestamp = System.nanoTime()
        val imu = IMUMeasurement(
            timestamp = timestamp,
            accelX = 1.0, accelY = 2.0, accelZ = 3.0,
            gyroX = 0.1, gyroY = 0.2, gyroZ = 0.3
        )
        
        assertEquals(timestamp, imu.timestamp)
        assertEquals(1.0, imu.accelX, 0.001)
        assertEquals(0.3, imu.gyroZ, 0.001)
    }
    
    @Test
    fun testSpeedMeasurementCreation() {
        val timestamp = System.nanoTime()
        val speed = SpeedMeasurement(
            timestamp = timestamp,
            speed = 15.5,
            confidence = 0.8
        )
        
        assertEquals(timestamp, speed.timestamp)
        assertEquals(15.5, speed.speed, 0.001)
        assertEquals(0.8, speed.confidence, 0.001)
    }
    
    @Test
    fun testProcessNoiseConfig() {
        val config = ProcessNoiseConfig(
            positionNoise = 0.5,
            velocityNoise = 1.0,
            yawNoise = 0.02
        )
        
        assertEquals(0.5, config.positionNoise, 0.001)
        assertEquals(1.0, config.velocityNoise, 0.001)
        assertEquals(0.02, config.yawNoise, 0.001)
    }
    
    @Test
    fun testMeasurementNoiseConfig() {
        val config = MeasurementNoiseConfig(
            speedVariance = 2.0,
            gpsVariance = 10.0
        )
        
        assertEquals(2.0, config.speedVariance, 0.001)
        assertEquals(10.0, config.gpsVariance, 0.001)
    }
    
    @Test
    fun testEKFWithRealisticSequence() {
        val ekf = EKFNavigationEngine()
        val startTime = System.nanoTime()
        
        // Simulate 1 second of IMU data at 100Hz
        for (i in 0 until 100) {
            val timestamp = startTime + i * 10_000_000L // 10ms intervals
            
            // Simulate constant acceleration forward
            val imu = IMUMeasurement(
                timestamp = timestamp,
                accelX = 2.0, // 2 m/s² forward
                accelY = 0.0,
                accelZ = -9.81, // Gravity
                gyroX = 0.0,
                gyroY = 0.0,
                gyroZ = 0.0 // No rotation
            )
            
            ekf.predict(imu)
        }
        
        val finalState = ekf.getCurrentState()
        
        // After 1 second of 2 m/s² acceleration:
        // Expected velocity: 2 m/s
        // Expected position: 1 m
        assertTrue("Should have forward velocity", finalState.vx > 1.0)
        assertTrue("Should have moved forward", finalState.x > 0.5)
        
        // Should not have moved sideways significantly
        assertTrue("Should not move sideways much", abs(finalState.y) < 0.1)
        assertTrue("Should not have sideways velocity", abs(finalState.vy) < 0.1)
    }
    
    @Test
    fun testEKFWithTurning() {
        val ekf = EKFNavigationEngine()
        val startTime = System.nanoTime()
        
        // Simulate turning motion
        for (i in 0 until 100) {
            val timestamp = startTime + i * 10_000_000L
            
            val imu = IMUMeasurement(
                timestamp = timestamp,
                accelX = 1.0, // Forward acceleration
                accelY = 0.0,
                accelZ = -9.81,
                gyroX = 0.0,
                gyroY = 0.0,
                gyroZ = 0.1 // Turning at 0.1 rad/s
            )
            
            ekf.predict(imu)
        }
        
        val finalState = ekf.getCurrentState()
        
        // Should have turned
        assertTrue("Should have rotated", abs(finalState.yaw) > 0.05)
        
        // Should have moved in a curved path
        assertTrue("Should have moved", finalState.x > 0 || finalState.y > 0)
    }
}
