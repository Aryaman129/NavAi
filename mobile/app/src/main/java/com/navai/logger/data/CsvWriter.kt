package com.navai.logger.data

import android.content.Context
import android.os.Build
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*

/**
 * High-performance CSV writer for sensor data with automatic file rotation
 */
class CsvWriter(private val context: Context) {
    
    companion object {
        private const val MAX_FILE_SIZE_MB = 50
        private const val MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
        private const val LOG_DIR_NAME = "logs"
    }
    
    private val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
    private val logDir: File = File(context.getExternalFilesDir(null), LOG_DIR_NAME).apply { 
        mkdirs() 
    }
    
    private var currentFile: File? = null
    private var fileWriter: FileWriter? = null
    private var currentFileSize = 0L
    
    init {
        createNewFile()
    }
    
    /**
     * Write a batch of sensor data efficiently
     */
    suspend fun writeBatch(sensorDataList: List<SensorData>) = withContext(Dispatchers.IO) {
        try {
            val writer = fileWriter ?: return@withContext
            
            sensorDataList.forEach { sensorData ->
                val csvRow = sensorData.toCsvRow()
                writer.appendLine(csvRow)
                currentFileSize += csvRow.length + 1 // +1 for newline
            }
            
            writer.flush()
            
            // Check if file rotation is needed
            if (currentFileSize > MAX_FILE_SIZE_BYTES) {
                rotateFile()
            }
            
        } catch (e: IOException) {
            // Handle write error - could log to system or create new file
            rotateFile()
        }
    }
    
    /**
     * Write a single sensor data point
     */
    suspend fun write(sensorData: SensorData) = withContext(Dispatchers.IO) {
        writeBatch(listOf(sensorData))
    }
    
    /**
     * Close the current file and cleanup resources
     */
    fun close() {
        try {
            fileWriter?.flush()
            fileWriter?.close()
        } catch (e: IOException) {
            // Log error if needed
        } finally {
            fileWriter = null
            currentFile = null
        }
    }
    
    /**
     * Get list of all log files
     */
    fun getLogFiles(): List<File> {
        return logDir.listFiles()?.filter { it.isFile && it.extension == "csv" }?.sortedByDescending { it.lastModified() } ?: emptyList()
    }
    
    /**
     * Clear all log files
     */
    fun clearAllLogs(): Boolean {
        close()
        return try {
            logDir.listFiles()?.forEach { it.delete() }
            createNewFile()
            true
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Get total size of all log files in MB
     */
    fun getTotalLogSizeMB(): Double {
        val totalBytes = logDir.listFiles()?.sumOf { it.length() } ?: 0L
        return totalBytes / (1024.0 * 1024.0)
    }
    
    private fun createNewFile() {
        try {
            close() // Close current file if open
            
            val timestamp = dateFormat.format(Date())
            val deviceModel = Build.MODEL.replace(" ", "_")
            val fileName = "navai_${deviceModel}_${timestamp}.csv"
            
            currentFile = File(logDir, fileName)
            fileWriter = FileWriter(currentFile!!, false) // false = overwrite if exists
            currentFileSize = 0L
            
            // Write CSV header
            val header = "type,timestamp_ns,v1,v2,v3,v4,v5"
            fileWriter!!.appendLine(header)
            fileWriter!!.flush()
            currentFileSize += header.length + 1
            
        } catch (e: IOException) {
            // Handle file creation error
            fileWriter = null
            currentFile = null
        }
    }
    
    private fun rotateFile() {
        createNewFile()
    }
    
    /**
     * Export logs to a shareable format
     */
    suspend fun exportLogs(): File? = withContext(Dispatchers.IO) {
        try {
            val exportFile = File(context.getExternalFilesDir(null), "navai_export_${System.currentTimeMillis()}.zip")
            
            // For now, just return the log directory
            // In a full implementation, you'd create a ZIP file here
            logDir
            
        } catch (e: Exception) {
            null
        }
    }
}
