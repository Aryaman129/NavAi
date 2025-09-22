package com.navai.logger.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.navai.logger.data.CsvWriter
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

data class LoggerUiState(
    val isLogging: Boolean = false,
    val sampleCount: Long = 0,
    val duration: String = "00:00",
    val logFiles: List<File> = emptyList(),
    val totalLogSizeMB: Double = 0.0,
    val isExporting: Boolean = false,
    val message: String? = null
)

class LoggerViewModel(application: Application) : AndroidViewModel(application) {
    
    private val _uiState = MutableStateFlow(LoggerUiState())
    val uiState: StateFlow<LoggerUiState> = _uiState.asStateFlow()
    
    private val csvWriter = CsvWriter(application)
    private val timeFormat = SimpleDateFormat("mm:ss", Locale.US)
    
    init {
        refreshLogFiles()
        // In a real implementation, you'd observe the service state here
        // For now, we'll simulate with periodic updates
        startPeriodicUpdates()
    }
    
    private fun startPeriodicUpdates() {
        viewModelScope.launch {
            while (true) {
                refreshLogFiles()
                kotlinx.coroutines.delay(2000) // Update every 2 seconds
            }
        }
    }
    
    fun refreshLogFiles() {
        viewModelScope.launch {
            val files = csvWriter.getLogFiles()
            val totalSize = csvWriter.getTotalLogSizeMB()
            
            _uiState.value = _uiState.value.copy(
                logFiles = files,
                totalLogSizeMB = totalSize
            )
        }
    }
    
    fun exportLogs() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isExporting = true)
            
            try {
                val exportFile = csvWriter.exportLogs()
                if (exportFile != null) {
                    _uiState.value = _uiState.value.copy(
                        isExporting = false,
                        message = "Logs exported successfully"
                    )
                } else {
                    _uiState.value = _uiState.value.copy(
                        isExporting = false,
                        message = "Export failed"
                    )
                }
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    isExporting = false,
                    message = "Export error: ${e.message}"
                )
            }
        }
    }
    
    fun clearLogs() {
        viewModelScope.launch {
            val success = csvWriter.clearAllLogs()
            if (success) {
                _uiState.value = _uiState.value.copy(
                    logFiles = emptyList(),
                    totalLogSizeMB = 0.0,
                    message = "Logs cleared successfully"
                )
            } else {
                _uiState.value = _uiState.value.copy(
                    message = "Failed to clear logs"
                )
            }
        }
    }
    
    fun clearMessage() {
        _uiState.value = _uiState.value.copy(message = null)
    }
    
    // Simulate logging state updates
    // In a real implementation, this would come from the service
    fun updateLoggingState(isLogging: Boolean, sampleCount: Long = 0, startTime: Long = 0) {
        val duration = if (isLogging && startTime > 0) {
            val elapsed = (System.currentTimeMillis() - startTime) / 1000
            String.format("%02d:%02d", elapsed / 60, elapsed % 60)
        } else {
            "00:00"
        }
        
        _uiState.value = _uiState.value.copy(
            isLogging = isLogging,
            sampleCount = sampleCount,
            duration = duration
        )
    }
    
    override fun onCleared() {
        super.onCleared()
        csvWriter.close()
    }
}
