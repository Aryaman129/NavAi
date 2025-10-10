package com.navai.logger

import android.Manifest
import android.content.Intent
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.navai.logger.ui.theme.NavAITheme

/**
 * Main launcher activity with navigation to different features
 */
class LauncherActivity : ComponentActivity() {
    
    // Only require ESSENTIAL permissions - others are optional
    private val requiredPermissions = arrayOf(
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.ACCESS_COARSE_LOCATION
    )
    
    // Optional permissions (nice to have, but not required)
    private val optionalPermissions = mutableListOf<String>().apply {
        add(Manifest.permission.HIGH_SAMPLING_RATE_SENSORS)
        
        // Add background location for Android 10+ (for continuous tracking)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            add(Manifest.permission.ACCESS_BACKGROUND_LOCATION)
        }
        // Add notification permission for Android 13+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            add(Manifest.permission.POST_NOTIFICATIONS)
        }
    }.toTypedArray()

    private var permissionsGranted by mutableStateOf(false)
    private var deniedPermissions by mutableStateOf<List<String>>(emptyList())

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        // Only check REQUIRED permissions
        val essentialGranted = requiredPermissions.all { perm ->
            permissions[perm] == true
        }
        permissionsGranted = essentialGranted
        
        if (!essentialGranted) {
            deniedPermissions = requiredPermissions.filter { 
                permissions[it] != true 
            }
        } else {
            deniedPermissions = emptyList()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Request only REQUIRED permissions on start
        permissionLauncher.launch(requiredPermissions)
        
        setContent {
            NavAITheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    LauncherScreen(
                        permissionsGranted = permissionsGranted,
                        deniedPermissions = deniedPermissions,
                        onRequestPermissions = {
                            permissionLauncher.launch(requiredPermissions)
                        }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LauncherScreen(
    permissionsGranted: Boolean,
    deniedPermissions: List<String>,
    onRequestPermissions: () -> Unit
) {
    val context = LocalContext.current
    val scrollState = rememberScrollState()
    
    var showDebugInfo by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("NavAI - Navigation AI") },
                actions = {
                    IconButton(onClick = { showDebugInfo = !showDebugInfo }) {
                        Icon(
                            imageVector = if (showDebugInfo) Icons.Default.BugReport else Icons.Default.Info,
                            contentDescription = "Debug Info"
                        )
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp)
                .verticalScroll(scrollState),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // App Header
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            ) {
                Column(
                    modifier = Modifier.padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "🚗",
                        style = MaterialTheme.typography.displayLarge
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "NavAI",
                        style = MaterialTheme.typography.headlineLarge,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "IMU-based Speed Estimation & Sensor Logging",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                    )
                }
            }
            
            // Permissions Status
            if (!permissionsGranted) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Icon(
                                Icons.Default.Warning,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.error
                            )
                            Text(
                                text = "Permissions Required",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                        }
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Text(
                            text = "The following permissions are needed:",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onErrorContainer
                        )
                        
                        deniedPermissions.forEach { permission ->
                            Text(
                                text = "• ${permission.split(".").lastOrNull() ?: permission}",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                        }
                        
                        Spacer(modifier = Modifier.height(12.dp))
                        
                        Button(
                            onClick = onRequestPermissions,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Grant Permissions")
                        }
                    }
                }
            } else {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = Color(0xFF4CAF50).copy(alpha = 0.1f)
                    )
                ) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            Icons.Default.CheckCircle,
                            contentDescription = null,
                            tint = Color(0xFF4CAF50)
                        )
                        Text(
                            text = "All permissions granted ✓",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Medium
                        )
                    }
                }
            }
            
            // Feature Cards
            Text(
                text = "Features",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )
            
            // Speed Prediction Feature
            FeatureCard(
                icon = "⚡",
                title = "Speed Prediction (TFLite)",
                description = "Real-time speed estimation using trained TensorFlow Lite model with IMU sensors",
                enabled = permissionsGranted,
                onClick = {
                    try {
                        val intent = Intent(context, SpeedTestActivity::class.java)
                        context.startActivity(intent)
                    } catch (e: Exception) {
                        errorMessage = "Failed to launch Speed Prediction: ${e.message}"
                    }
                },
                badge = "NEW"
            )
            
            // Sensor Logging Feature
            FeatureCard(
                icon = "📊",
                title = "Sensor Data Logger",
                description = "High-frequency IMU and GPS data collection for training and analysis",
                enabled = permissionsGranted,
                onClick = {
                    try {
                        val intent = Intent(context, MainActivity::class.java)
                        context.startActivity(intent)
                    } catch (e: Exception) {
                        errorMessage = "Failed to launch Sensor Logger: ${e.message}"
                    }
                }
            )
            
            // Debug Information
            if (showDebugInfo) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Text(
                            text = "🔧 Debug Information",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Divider(modifier = Modifier.padding(vertical = 8.dp))
                        
                        DebugInfoRow("Android Version", "${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
                        DebugInfoRow("Device", "${Build.MANUFACTURER} ${Build.MODEL}")
                        DebugInfoRow("Permissions", if (permissionsGranted) "✓ All granted" else "✗ ${deniedPermissions.size} denied")
                        DebugInfoRow("TFLite Model", "phase1_model.tflite (1.28 MB)")
                        DebugInfoRow("Package", context.packageName)
                        
                        // Check if assets are accessible (moved outside composable)
                        val assetCheckResult = remember {
                            try {
                                val modelExists = context.assets.list("")?.contains("phase1_model.tflite") == true
                                val metadataExists = context.assets.list("")?.contains("phase1_model_metadata.json") == true
                                Triple(true, modelExists, metadataExists)
                            } catch (e: Exception) {
                                Triple(false, false, false)
                            }
                        }
                        
                        if (assetCheckResult.first) {
                            DebugInfoRow("Model File", if (assetCheckResult.second) "✓ Found" else "✗ Missing")
                            DebugInfoRow("Metadata File", if (assetCheckResult.third) "✓ Found" else "✗ Missing")
                        } else {
                            DebugInfoRow("Asset Check", "✗ Error checking assets")
                        }
                    }
                }
            }
            
            // Error Display
            errorMessage?.let { error ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "❌ Error",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                            IconButton(onClick = { errorMessage = null }) {
                                Icon(Icons.Default.Close, contentDescription = "Dismiss")
                            }
                        }
                        Text(
                            text = error,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onErrorContainer
                        )
                    }
                }
            }
            
            // Footer
            Text(
                text = "Version 0.1.0 | Phase 1 Complete",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 16.dp)
            )
        }
    }
}

@Composable
fun FeatureCard(
    icon: String,
    title: String,
    description: String,
    enabled: Boolean,
    onClick: () -> Unit,
    badge: String? = null
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        onClick = onClick,
        enabled = enabled
    ) {
        Column(
            modifier = Modifier.padding(20.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    text = icon,
                    style = MaterialTheme.typography.displaySmall
                )
                
                Column(modifier = Modifier.weight(1f)) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = title,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        
                        badge?.let {
                            Surface(
                                color = MaterialTheme.colorScheme.primary,
                                shape = MaterialTheme.shapes.small
                            ) {
                                Text(
                                    text = it,
                                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onPrimary
                                )
                            }
                        }
                    }
                    
                    Text(
                        text = description,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                
                Icon(
                    Icons.Default.ArrowForward,
                    contentDescription = null,
                    tint = if (enabled) MaterialTheme.colorScheme.primary 
                           else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
                )
            }
        }
    }
}

@Composable
fun DebugInfoRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium
        )
    }
}
