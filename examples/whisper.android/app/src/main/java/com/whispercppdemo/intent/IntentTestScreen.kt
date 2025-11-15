package com.whispercppdemo.intent

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun IntentTestScreen(
    viewModel: IntentTestViewModel = viewModel()
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Header
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Intent Classification Test",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Test the intent classifier with your commands",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                )
            }
        }
        
        // Initialization Status
        InitializationStatus(viewModel)
        
        // Input Section
        if (viewModel.isInitialized) {
            InputSection(viewModel)
        }
        
        // Example Commands
        if (viewModel.isInitialized) {
            ExampleCommands(viewModel)
        }
        
        // Results Section
        ResultsSection(viewModel)
        
        // Available Intents
        if (viewModel.isInitialized && viewModel.intentList.isNotEmpty()) {
            AvailableIntents(viewModel.intentList)
        }
    }
}

@Composable
private fun InitializationStatus(viewModel: IntentTestViewModel) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (viewModel.isInitialized) {
                Color(0xFF4CAF50).copy(alpha = 0.1f)
            } else if (viewModel.isLoading) {
                Color(0xFFFF9800).copy(alpha = 0.1f)
            } else {
                Color(0xFFF44336).copy(alpha = 0.1f)
            }
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (viewModel.isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.size(24.dp),
                    strokeWidth = 2.dp
                )
            } else {
                Text(
                    text = if (viewModel.isInitialized) "✅" else "❌",
                    fontSize = 24.sp
                )
            }
            
            Spacer(modifier = Modifier.width(12.dp))
            
            Column {
                Text(
                    text = when {
                        viewModel.isLoading -> "Initializing TensorFlow Lite models..."
                        viewModel.isInitialized -> "Intent Classifier Ready"
                        else -> "Initialization Failed"
                    },
                    fontWeight = FontWeight.Medium
                )
                
                if (viewModel.isInitialized) {
                    Text(
                        text = "${viewModel.intentList.size} intents loaded",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                    )
                }
                
                viewModel.errorMessage?.let { error ->
                    Text(
                        text = error,
                        style = MaterialTheme.typography.bodySmall,
                        color = Color.Red
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun InputSection(viewModel: IntentTestViewModel) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Enter Text to Classify",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Medium
            )
            
            OutlinedTextField(
                value = viewModel.inputText,
                onValueChange = viewModel::updateInputText,
                modifier = Modifier.fillMaxWidth(),
                placeholder = { Text("Enter a voice command here...") },
                minLines = 2,
                maxLines = 4,
                enabled = !viewModel.isLoading
            )
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = viewModel::classifyIntent,
                    modifier = Modifier.weight(1f),
                    enabled = !viewModel.isLoading && viewModel.inputText.isNotBlank()
                ) {
                    if (viewModel.isLoading) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(16.dp),
                            strokeWidth = 2.dp,
                            color = MaterialTheme.colorScheme.onPrimary
                        )
                    } else {
                        Icon(Icons.Default.Send, contentDescription = null)
                    }
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Classify Intent")
                }
                
                OutlinedButton(
                    onClick = viewModel::clearResults,
                    enabled = !viewModel.isLoading
                ) {
                    Icon(Icons.Default.Clear, contentDescription = null)
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Clear")
                }
            }
        }
    }
}

@Composable
private fun ExampleCommands(viewModel: IntentTestViewModel) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "Example Commands",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Medium
            )
            
            val examples = listOf(
                "How many steps did I take today?",  // QueryPoint: metric=steps, time_ref=today
                "What's my average heart rate yesterday?",  // QueryPoint: metric=heart rate, qualifier=average, time_ref=yesterday
                "Set my daily step goal to 10000",  // SetGoal: metric=steps, target=10000, unit=count
                "Set a threshold for heart rate above 100 bpm",  // SetThreshold: metric=heart rate, threshold=100, type=high, unit=bpm
                "Set a timer for 15 minutes",  // TimerStopwatch: tool=timer, action=set, value=15
                "Turn on do not disturb",  // ToggleFeature: feature=do not disturb, state=on
                "Start a running workout",  // StartActivity: activity_type=outdoor run
                "Call mom",  // PhoneAction: action=call, contact=mom
                "What's the weather in London?",  // WeatherQuery: location=london
                "How far did I walk last week?"  // QueryPoint: metric=distance, time_ref=last week
            )
            
            LazyColumn(
                modifier = Modifier.height(120.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                items(examples) { example ->
                    OutlinedButton(
                        onClick = { viewModel.tryExampleText(example) },
                        modifier = Modifier.fillMaxWidth(),
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp)
                    ) {
                        Text(
                            text = example,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ResultsSection(viewModel: IntentTestViewModel) {
    viewModel.result?.let { result ->
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF4CAF50).copy(alpha = 0.1f))
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    text = "Classification Result",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Medium
                )
                
                // Main result
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = result.intent,
                            style = MaterialTheme.typography.headlineSmall,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF4CAF50)
                        )
                        Text(
                            text = "Confidence: ${"%.1f".format(result.confidence * 100)}%",
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
                
                // Slot extraction results
                if (result.slots.isNotEmpty()) {
                    Divider()
                    
                    Text(
                        text = "Extracted Slots:",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Medium
                    )
                    
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = Color(0xFF2196F3).copy(alpha = 0.1f))
                    ) {
                        Column(
                            modifier = Modifier.padding(12.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Text(
                                text = "Slot Confidence: ${"%.1f".format(result.slotConfidence * 100)}%",
                                style = MaterialTheme.typography.bodySmall,
                                color = Color(0xFF2196F3)
                            )
                            
                            result.slots.forEach { (slotName, slotValue) ->
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .background(
                                            Color(0xFF2196F3).copy(alpha = 0.1f),
                                            RoundedCornerShape(4.dp)
                                        )
                                        .padding(8.dp),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        text = slotName,
                                        style = MaterialTheme.typography.bodySmall,
                                        fontWeight = FontWeight.Medium,
                                        color = Color(0xFF1976D2)
                                    )
                                    Text(
                                        text = slotValue.toString(),
                                        style = MaterialTheme.typography.bodySmall,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                            }
                        }
                    }
                }
                
                Divider()
                
                // All probabilities
                Text(
                    text = "All Intent Probabilities:",
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Medium
                )
                
                LazyColumn(
                    modifier = Modifier.height(200.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    items(result.allProbabilities.toList().sortedByDescending { it.second }) { (intent, probability) ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .background(
                                    if (intent == result.intent) Color(0xFF4CAF50).copy(alpha = 0.2f)
                                    else Color.Transparent,
                                    RoundedCornerShape(4.dp)
                                )
                                .padding(vertical = 4.dp, horizontal = 8.dp),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = intent,
                                style = MaterialTheme.typography.bodySmall,
                                fontWeight = if (intent == result.intent) FontWeight.Bold else FontWeight.Normal
                            )
                            Text(
                                text = "${"%.1f".format(probability * 100)}%",
                                style = MaterialTheme.typography.bodySmall,
                                fontWeight = if (intent == result.intent) FontWeight.Bold else FontWeight.Normal
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun AvailableIntents(intents: List<String>) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "Available Intents (${intents.size})",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Medium
            )
            
            LazyColumn(
                modifier = Modifier.height(150.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                items(intents.sorted()) { intent ->
                    Text(
                        text = "• $intent",
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(vertical = 2.dp)
                    )
                }
            }
        }
    }
}