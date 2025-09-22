package com.navai.logger.data

/**
 * Sealed class representing different types of sensor data
 */
sealed class SensorData {
    abstract val timestamp: Long
    abstract fun toCsvRow(): String
    
    data class AccelerometerData(
        override val timestamp: Long,
        val x: Float,
        val y: Float,
        val z: Float
    ) : SensorData() {
        override fun toCsvRow(): String = "accel,$timestamp,$x,$y,$z"
    }
    
    data class GyroscopeData(
        override val timestamp: Long,
        val x: Float,
        val y: Float,
        val z: Float
    ) : SensorData() {
        override fun toCsvRow(): String = "gyro,$timestamp,$x,$y,$z"
    }
    
    data class MagnetometerData(
        override val timestamp: Long,
        val x: Float,
        val y: Float,
        val z: Float
    ) : SensorData() {
        override fun toCsvRow(): String = "mag,$timestamp,$x,$y,$z"
    }
    
    data class RotationVectorData(
        override val timestamp: Long,
        val x: Float,
        val y: Float,
        val z: Float,
        val w: Float
    ) : SensorData() {
        override fun toCsvRow(): String = "rotvec,$timestamp,$x,$y,$z,$w"
    }
    
    data class GpsData(
        override val timestamp: Long,
        val latitude: Double,
        val longitude: Double,
        val speed: Float,
        val accuracy: Float,
        val bearing: Float
    ) : SensorData() {
        override fun toCsvRow(): String = "gps,$timestamp,$latitude,$longitude,$speed,$accuracy,$bearing"
    }
}

/**
 * Data class for unified sensor schema used in ML pipeline
 */
data class UnifiedSensorData(
    val timestampNs: Long,
    val accelX: Float = 0f,
    val accelY: Float = 0f,
    val accelZ: Float = 0f,
    val gyroX: Float = 0f,
    val gyroY: Float = 0f,
    val gyroZ: Float = 0f,
    val magX: Float = 0f,
    val magY: Float = 0f,
    val magZ: Float = 0f,
    val qw: Float = 1f,
    val qx: Float = 0f,
    val qy: Float = 0f,
    val qz: Float = 0f,
    val gpsLat: Double = 0.0,
    val gpsLon: Double = 0.0,
    val gpsSpeedMps: Float = 0f,
    val device: String = "unknown",
    val source: String = "navai_logger"
) {
    fun toCsvRow(): String {
        return "$timestampNs,$accelX,$accelY,$accelZ,$gyroX,$gyroY,$gyroZ," +
                "$magX,$magY,$magZ,$qw,$qx,$qy,$qz,$gpsLat,$gpsLon,$gpsSpeedMps,$device,$source"
    }
    
    companion object {
        fun csvHeader(): String {
            return "timestamp_ns,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z," +
                    "mag_x,mag_y,mag_z,qw,qx,qy,qz,gps_lat,gps_lon,gps_speed_mps,device,source"
        }
    }
}
