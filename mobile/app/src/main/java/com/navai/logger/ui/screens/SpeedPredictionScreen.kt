package com.navai.logger.ui.screens

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import androidx.core.content.ContextCompat
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.navai.logger.service.SpeedPredictionService
import kotlin.math.abs

@Composable
fun SpeedPredictionScreen() {
    val context = LocalContext.current
    
    // State managed via broadcasts instead of StateFlow
    var isRunning by remember { mutableStateOf(false) }
    var predictedSpeed by remember { mutableStateOf(0f) }
    var gpsSpeed by remember { mutableStateOf(0f) }
    var inferenceTimeMs by remember { mutableStateOf(0L) }
    var error by remember { mutableStateOf(0f) }
    var avgError by remember { mutableStateOf(0f) }
    var sampleCount by remember { mutableStateOf(0) }
    var serviceState by remember { mutableStateOf("idle") } // "running", "stopped", "error", "idle"
    var errorMessage by remember { mutableStateOf("") }
    
    // Register broadcast receiver for service updates
    DisposableEffect(Unit) {
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                val state = intent.getStringExtra(SpeedPredictionService.EXTRA_STATE) ?: "running"
                serviceState = state
                
                when (state) {
                    "running" -> {
                        isRunning = true
                        predictedSpeed = intent.getFloatExtra(SpeedPredictionService.EXTRA_PREDICTED_SPEED, 0f)
                        gpsSpeed = intent.getFloatExtra(SpeedPredictionService.EXTRA_GPS_SPEED, 0f)
                        inferenceTimeMs = intent.getIntExtra(SpeedPredictionService.EXTRA_INFERENCE_TIME, 0).toLong()
                        error = intent.getFloatExtra(SpeedPredictionService.EXTRA_ERROR, 0f)
                        avgError = intent.getFloatExtra(SpeedPredictionService.EXTRA_AVG_ERROR, 0f)
                        sampleCount = intent.getIntExtra(SpeedPredictionService.EXTRA_SAMPLE_COUNT, 0)
                    }
                    "stopped" -> {
                        isRunning = false
                    }
                    "error" -> {
                        isRunning = false
                        errorMessage = intent.getStringExtra(SpeedPredictionService.EXTRA_ERROR_MESSAGE) ?: "Unknown error"
                    }
                }
            }
        }
        
        val filter = IntentFilter(SpeedPredictionService.BROADCAST_PREDICTION_UPDATE)
        
        // Android 13+ requires RECEIVER_NOT_EXPORTED flag for local broadcasts
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.registerReceiver(receiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            context.registerReceiver(receiver, filter)
        }
        
        onDispose {
            context.unregisterReceiver(receiver)
        }
    }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Title
        Text(
            text = "🚗 NavAI Speed Prediction",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold
        )
        
        Spacer(modifier = Modifier.height(8.dp))
        
        // Main speed display based on service state
        when (serviceState) {
            "running" -> {
                SpeedDisplayCard(
                    predictedSpeed = predictedSpeed,
                    gpsSpeed = gpsSpeed,
                    inferenceTimeMs = inferenceTimeMs,
                    error = error,
                    avgError = avgError,
                    sampleCount = sampleCount
                )
            }
            "stopped" -> {
                // Show final stats when stopped
                Text(
                    text = "Prediction Stopped",
                    style = MaterialTheme.typography.titleLarge
                )
                Text(
                    text = "Total predictions: $sampleCount",
                    style = MaterialTheme.typography.bodyLarge
                )
                Text(
                    text = "Avg error: ${String.format("%.2f", avgError)} m/s",
                    style = MaterialTheme.typography.bodyLarge
                )
            }
            "error" -> {
                ErrorCard(
                    message = "Prediction Error",
                    details = errorMessage
                )
            }
            else -> {
                IdleCard()
            }
        }
        
        Spacer(modifier = Modifier.weight(1f))
        
        // Control button
        Button(
            onClick = {
                if (isRunning) {
                    // Stop prediction
                    val intent = Intent(context, SpeedPredictionService::class.java).apply {
                        action = SpeedPredictionService.ACTION_STOP_PREDICTION
                    }
                    context.stopService(intent)
                    isRunning = false
                } else {
                    // Start prediction
                    val intent = Intent(context, SpeedPredictionService::class.java).apply {
                        action = SpeedPredictionService.ACTION_START_PREDICTION
                    }
                    context.startForegroundService(intent)
                    isRunning = true
                }
            },
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = if (isRunning) Color.Red else Color(0xFF4CAF50)
            )
        ) {
            Text(
                text = if (isRunning) "⏹ STOP PREDICTION" else "▶ START PREDICTION",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )
        }
        
        // Info text
        Text(
            text = if (isRunning) {
                "🔴 Service running in background\nCheck notification for real-time updates"
            } else {
                "📱 Press START to begin real-time speed prediction\nUsing TFLite model with IMU sensors"
            },
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(horizontal = 16.dp)
        )
    }
}

@Composable
fun SpeedDisplayCard(
    predictedSpeed: Float,
    gpsSpeed: Float,
    inferenceTimeMs: Long,
    error: Float,
    avgError: Float,
    sampleCount: Int
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Predicted speed (large display)
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = "Predicted Speed",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.primary
                )
                Text(
                    text = String.format("%.1f", predictedSpeed),
                    fontSize = 64.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                )
                Text(
                    text = "km/h",
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            
            Divider()
            
            // GPS comparison
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("GPS Speed", style = MaterialTheme.typography.bodyMedium)
                    Text(
                        String.format("%.1f km/h", gpsSpeed),
                        fontWeight = FontWeight.Bold,
                        fontSize = 20.sp
                    )
                }
                
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("Error", style = MaterialTheme.typography.bodyMedium)
                    Text(
                        String.format("%.2f m/s", error),
                        fontWeight = FontWeight.Bold,
                        fontSize = 20.sp,
                        color = if (abs(error) < 1.0) Color(0xFF4CAF50) else Color.Red
                    )
                }
            }
            
            Divider()
            
            // Performance metrics
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                MetricItem("Latency", "${inferenceTimeMs}ms")
                MetricItem("Avg Error", String.format("%.2f m/s", avgError))
                MetricItem("Samples", "$sampleCount")
            }
        }
    }
}

@Composable
fun StatsCard(
    avgInferenceTimeMs: Long,
    minInferenceTimeMs: Long,
    maxInferenceTimeMs: Long,
    totalPredictions: Int,
    avgAbsError: Float
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "📊 Session Statistics",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )
            
            Divider()
            
            StatRow("Total Predictions", "$totalPredictions")
            StatRow("Avg Inference Time", "${avgInferenceTimeMs}ms")
            StatRow("Min Inference Time", "${minInferenceTimeMs}ms")
            StatRow("Max Inference Time", "${maxInferenceTimeMs}ms")
            StatRow("Avg Absolute Error", String.format("%.3f m/s", avgAbsError))
            
            // Performance assessment
            val performance = when {
                avgInferenceTimeMs < 20 -> "🟢 Excellent"
                avgInferenceTimeMs < 50 -> "🟡 Good"
                avgInferenceTimeMs < 100 -> "🟠 Acceptable"
                else -> "🔴 Slow"
            }
            
            val accuracy = when {
                avgAbsError < 0.5 -> "🟢 Excellent"
                avgAbsError < 1.0 -> "🟡 Good"
                avgAbsError < 2.0 -> "🟠 Acceptable"
                else -> "🔴 Poor"
            }
            
            Divider()
            
            StatRow("Performance", performance)
            StatRow("Accuracy", accuracy)
        }
    }
}

@Composable
fun IdleCard() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Text(
                text = "⏸️",
                fontSize = 64.sp
            )
            Text(
                text = "Ready to Start",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "Press START below to begin real-time speed prediction using your phone's accelerometer and gyroscope.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun ErrorCard(message: String, details: String?) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Text(
                text = "❌",
                fontSize = 64.sp
            )
            Text(
                text = "Error",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onErrorContainer
            )
            Text(
                text = message,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onErrorContainer,
                fontWeight = FontWeight.Medium
            )
            
            details?.let {
                Divider(color = MaterialTheme.colorScheme.onErrorContainer.copy(alpha = 0.3f))
                Text(
                    text = "Details:",
                    style = MaterialTheme.typography.bodySmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onErrorContainer
                )
                Text(
                    text = it,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onErrorContainer,
                    fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace
                )
            }
            
            Text(
                text = "💡 Share this error message with the developer for troubleshooting",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onErrorContainer.copy(alpha = 0.7f)
            )
        }
    }
}

@Composable
fun MetricItem(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            fontWeight = FontWeight.Bold,
            fontSize = 16.sp
        )
    }
}

@Composable
fun StatRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(text = label, style = MaterialTheme.typography.bodyMedium)
        Text(
            text = value,
            fontWeight = FontWeight.Bold,
            style = MaterialTheme.typography.bodyMedium
        )
    }
}
