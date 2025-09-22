package com.navai.sensorfusion

import org.ejml.simple.SimpleMatrix
import kotlin.math.*

/**
 * Extended Kalman Filter for IMU-based navigation
 * State: [x, y, vx, vy, yaw, bias_ax, bias_ay, bias_az, bias_gz]
 */
class EKFNavigationEngine(
    private val initialState: NavigationState = NavigationState(),
    private val processNoise: ProcessNoiseConfig = ProcessNoiseConfig(),
    private val measurementNoise: MeasurementNoiseConfig = MeasurementNoiseConfig()
) {
    
    companion object {
        private const val STATE_SIZE = 9
        private const val GRAVITY = 9.81
    }
    
    // State vector: [x, y, vx, vy, yaw, bias_ax, bias_ay, bias_az, bias_gz]
    private var state = SimpleMatrix(STATE_SIZE, 1)
    private var covariance = SimpleMatrix.identity(STATE_SIZE).scale(1.0)
    
    private var lastUpdateTime = 0L
    private var isInitialized = false
    
    init {
        initializeState(initialState)
    }
    
    private fun initializeState(initial: NavigationState) {
        state.set(0, 0, initial.x)
        state.set(1, 0, initial.y)
        state.set(2, 0, initial.vx)
        state.set(3, 0, initial.vy)
        state.set(4, 0, initial.yaw)
        state.set(5, 0, initial.biasAx)
        state.set(6, 0, initial.biasAy)
        state.set(7, 0, initial.biasAz)
        state.set(8, 0, initial.biasGz)
        
        // Initialize covariance with reasonable uncertainties
        covariance.set(0, 0, 10.0)  // x position uncertainty
        covariance.set(1, 1, 10.0)  // y position uncertainty
        covariance.set(2, 2, 5.0)   // vx velocity uncertainty
        covariance.set(3, 3, 5.0)   // vy velocity uncertainty
        covariance.set(4, 4, 0.5)   // yaw uncertainty
        covariance.set(5, 5, 1.0)   // accel bias uncertainty
        covariance.set(6, 6, 1.0)   // accel bias uncertainty
        covariance.set(7, 7, 1.0)   // accel bias uncertainty
        covariance.set(8, 8, 0.1)   // gyro bias uncertainty
    }
    
    /**
     * Prediction step using IMU measurements
     */
    fun predict(imuData: IMUMeasurement) {
        val currentTime = imuData.timestamp
        
        if (!isInitialized) {
            lastUpdateTime = currentTime
            isInitialized = true
            return
        }
        
        val dt = (currentTime - lastUpdateTime) / 1e9 // Convert to seconds
        if (dt <= 0 || dt > 1.0) { // Skip invalid or too large time steps
            lastUpdateTime = currentTime
            return
        }
        
        // Extract current state
        val x = state.get(0, 0)
        val y = state.get(1, 0)
        val vx = state.get(2, 0)
        val vy = state.get(3, 0)
        val yaw = state.get(4, 0)
        val biasAx = state.get(5, 0)
        val biasAy = state.get(6, 0)
        val biasAz = state.get(7, 0)
        val biasGz = state.get(8, 0)
        
        // Remove bias and transform accelerations to world frame
        val accelBodyX = imuData.accelX - biasAx
        val accelBodyY = imuData.accelY - biasAy
        val accelBodyZ = imuData.accelZ - biasAz
        
        // Transform to world frame (assuming phone is roughly horizontal)
        val cosYaw = cos(yaw)
        val sinYaw = sin(yaw)
        
        val accelWorldX = accelBodyX * cosYaw - accelBodyY * sinYaw
        val accelWorldY = accelBodyX * sinYaw + accelBodyY * cosYaw
        
        // Remove gravity (assuming Z-axis points up)
        val accelWorldZ = accelBodyZ + GRAVITY
        
        // Angular velocity (remove bias)
        val gyroZ = imuData.gyroZ - biasGz
        
        // State prediction (constant acceleration model)
        val newX = x + vx * dt + 0.5 * accelWorldX * dt * dt
        val newY = y + vy * dt + 0.5 * accelWorldY * dt * dt
        val newVx = vx + accelWorldX * dt
        val newVy = vy + accelWorldY * dt
        val newYaw = normalizeAngle(yaw + gyroZ * dt)
        
        // Bias prediction (random walk model)
        val newBiasAx = biasAx
        val newBiasAy = biasAy
        val newBiasAz = biasAz
        val newBiasGz = biasGz
        
        // Update state
        state.set(0, 0, newX)
        state.set(1, 0, newY)
        state.set(2, 0, newVx)
        state.set(3, 0, newVy)
        state.set(4, 0, newYaw)
        state.set(5, 0, newBiasAx)
        state.set(6, 0, newBiasAy)
        state.set(7, 0, newBiasAz)
        state.set(8, 0, newBiasGz)
        
        // Jacobian of state transition
        val F = createStateTransitionJacobian(dt, yaw, accelBodyX, accelBodyY, cosYaw, sinYaw)
        
        // Process noise matrix
        val Q = createProcessNoiseMatrix(dt)
        
        // Covariance prediction: P = F * P * F^T + Q
        covariance = F.mult(covariance).mult(F.transpose()).plus(Q)
        
        lastUpdateTime = currentTime
    }
    
    /**
     * Update step using ML speed estimate
     */
    fun updateWithSpeedEstimate(speedEstimate: SpeedMeasurement) {
        val vx = state.get(2, 0)
        val vy = state.get(3, 0)
        val predictedSpeed = sqrt(vx * vx + vy * vy)
        
        // Measurement residual
        val innovation = speedEstimate.speed - predictedSpeed
        
        // Measurement Jacobian: H = [0, 0, vx/speed, vy/speed, 0, 0, 0, 0, 0]
        val H = SimpleMatrix(1, STATE_SIZE)
        val speed = max(predictedSpeed, 0.1) // Avoid division by zero
        H.set(0, 2, vx / speed)
        H.set(0, 3, vy / speed)
        
        // Measurement noise
        val R = SimpleMatrix(1, 1)
        R.set(0, 0, measurementNoise.speedVariance)
        
        // Kalman gain: K = P * H^T * (H * P * H^T + R)^-1
        val S = H.mult(covariance).mult(H.transpose()).plus(R)
        val K = covariance.mult(H.transpose()).mult(S.invert())
        
        // State update: x = x + K * innovation
        val innovationMatrix = SimpleMatrix(1, 1)
        innovationMatrix.set(0, 0, innovation)
        state = state.plus(K.mult(innovationMatrix))
        
        // Covariance update: P = (I - K * H) * P
        val I = SimpleMatrix.identity(STATE_SIZE)
        covariance = I.minus(K.mult(H)).mult(covariance)
    }
    
    /**
     * Update step using GPS measurement
     */
    fun updateWithGPS(gpsData: GPSMeasurement) {
        // Position measurement
        val predictedX = state.get(0, 0)
        val predictedY = state.get(1, 0)
        
        val innovationX = gpsData.x - predictedX
        val innovationY = gpsData.y - predictedY
        
        // Measurement Jacobian for position: H = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        //                                         [0, 1, 0, 0, 0, 0, 0, 0, 0]
        val H = SimpleMatrix(2, STATE_SIZE)
        H.set(0, 0, 1.0)
        H.set(1, 1, 1.0)
        
        // Measurement noise based on GPS accuracy
        val R = SimpleMatrix(2, 2)
        R.set(0, 0, gpsData.accuracy * gpsData.accuracy)
        R.set(1, 1, gpsData.accuracy * gpsData.accuracy)
        
        // Kalman update
        val S = H.mult(covariance).mult(H.transpose()).plus(R)
        val K = covariance.mult(H.transpose()).mult(S.invert())
        
        val innovation = SimpleMatrix(2, 1)
        innovation.set(0, 0, innovationX)
        innovation.set(1, 0, innovationY)
        
        state = state.plus(K.mult(innovation))
        
        val I = SimpleMatrix.identity(STATE_SIZE)
        covariance = I.minus(K.mult(H)).mult(covariance)
    }
    
    /**
     * Get current navigation state
     */
    fun getCurrentState(): NavigationState {
        return NavigationState(
            x = state.get(0, 0),
            y = state.get(1, 0),
            vx = state.get(2, 0),
            vy = state.get(3, 0),
            yaw = state.get(4, 0),
            biasAx = state.get(5, 0),
            biasAy = state.get(6, 0),
            biasAz = state.get(7, 0),
            biasGz = state.get(8, 0),
            uncertainty = sqrt(covariance.get(0, 0) + covariance.get(1, 1))
        )
    }
    
    private fun createStateTransitionJacobian(
        dt: Double,
        yaw: Double,
        accelBodyX: Double,
        accelBodyY: Double,
        cosYaw: Double,
        sinYaw: Double
    ): SimpleMatrix {
        val F = SimpleMatrix.identity(STATE_SIZE)
        
        // Position derivatives
        F.set(0, 2, dt)  // dx/dvx
        F.set(1, 3, dt)  // dy/dvy
        
        // Velocity derivatives with respect to yaw
        F.set(2, 4, dt * (-accelBodyX * sinYaw - accelBodyY * cosYaw))
        F.set(3, 4, dt * (accelBodyX * cosYaw - accelBodyY * sinYaw))
        
        // Velocity derivatives with respect to accel bias
        F.set(2, 5, -dt * cosYaw)  // dvx/dbias_ax
        F.set(2, 6, dt * sinYaw)   // dvx/dbias_ay
        F.set(3, 5, -dt * sinYaw)  // dvy/dbias_ax
        F.set(3, 6, -dt * cosYaw)  // dvy/dbias_ay
        
        // Yaw derivative with respect to gyro bias
        F.set(4, 8, -dt)  // dyaw/dbias_gz
        
        return F
    }
    
    private fun createProcessNoiseMatrix(dt: Double): SimpleMatrix {
        val Q = SimpleMatrix(STATE_SIZE, STATE_SIZE)
        
        // Position noise (from velocity uncertainty)
        Q.set(0, 0, processNoise.positionNoise * dt * dt)
        Q.set(1, 1, processNoise.positionNoise * dt * dt)
        
        // Velocity noise (from acceleration uncertainty)
        Q.set(2, 2, processNoise.velocityNoise * dt)
        Q.set(3, 3, processNoise.velocityNoise * dt)
        
        // Yaw noise (from gyro uncertainty)
        Q.set(4, 4, processNoise.yawNoise * dt)
        
        // Bias noise (random walk)
        Q.set(5, 5, processNoise.accelBiasNoise * dt)
        Q.set(6, 6, processNoise.accelBiasNoise * dt)
        Q.set(7, 7, processNoise.accelBiasNoise * dt)
        Q.set(8, 8, processNoise.gyroBiasNoise * dt)
        
        return Q
    }
    
    private fun normalizeAngle(angle: Double): Double {
        var normalized = angle
        while (normalized > PI) normalized -= 2 * PI
        while (normalized < -PI) normalized += 2 * PI
        return normalized
    }
}

data class NavigationState(
    val x: Double = 0.0,
    val y: Double = 0.0,
    val vx: Double = 0.0,
    val vy: Double = 0.0,
    val yaw: Double = 0.0,
    val biasAx: Double = 0.0,
    val biasAy: Double = 0.0,
    val biasAz: Double = 0.0,
    val biasGz: Double = 0.0,
    val uncertainty: Double = 0.0
) {
    val speed: Double get() = sqrt(vx * vx + vy * vy)
    val heading: Double get() = yaw
}

data class IMUMeasurement(
    val timestamp: Long,
    val accelX: Double,
    val accelY: Double,
    val accelZ: Double,
    val gyroX: Double,
    val gyroY: Double,
    val gyroZ: Double
)

data class SpeedMeasurement(
    val timestamp: Long,
    val speed: Double,
    val confidence: Double = 1.0
)

data class GPSMeasurement(
    val timestamp: Long,
    val x: Double,
    val y: Double,
    val accuracy: Double
)

data class ProcessNoiseConfig(
    val positionNoise: Double = 0.1,
    val velocityNoise: Double = 0.5,
    val yawNoise: Double = 0.01,
    val accelBiasNoise: Double = 0.01,
    val gyroBiasNoise: Double = 0.001
)

data class MeasurementNoiseConfig(
    val speedVariance: Double = 1.0,
    val gpsVariance: Double = 25.0
)
