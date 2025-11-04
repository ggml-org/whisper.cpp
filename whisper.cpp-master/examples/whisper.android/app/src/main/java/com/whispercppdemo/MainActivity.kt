package com.whispercppdemo

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import com.whispercppdemo.intent.IntentTestScreen
import com.whispercppdemo.intent.IntentTestViewModel
import com.whispercppdemo.ui.main.MainScreen
import com.whispercppdemo.ui.main.MainScreenViewModel
import com.whispercppdemo.ui.theme.WhisperCppDemoTheme

class MainActivity : ComponentActivity() {
    private val viewModel: MainScreenViewModel by viewModels {
        MainScreenViewModel.factory(application)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            WhisperCppDemoTheme {
                WhisperAppWithTabs(viewModel)
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun WhisperAppWithTabs(mainViewModel: MainScreenViewModel) {
    var selectedTabIndex by remember { mutableIntStateOf(0) }
    
    val tabs = listOf(
        "🎤 Whisper",
        "🎯 Intent Test"
    )
    
    Column(
        modifier = Modifier.fillMaxSize()
    ) {
        TabRow(
            selectedTabIndex = selectedTabIndex
        ) {
            tabs.forEachIndexed { index, title ->
                Tab(
                    selected = selectedTabIndex == index,
                    onClick = { selectedTabIndex = index },
                    text = { Text(title) }
                )
            }
        }
        
        when (selectedTabIndex) {
            0 -> MainScreen(mainViewModel)
            1 -> {
                val intentViewModel: IntentTestViewModel = viewModel()
                IntentTestScreen(intentViewModel)
            }
        }
    }
}