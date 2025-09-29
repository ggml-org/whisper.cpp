package com.whispercppdemo.intent

import android.app.Application
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch

private const val LOG_TAG = "IntentTestViewModel"

class IntentTestViewModel(private val application: Application) : AndroidViewModel(application) {
    
    var inputText by mutableStateOf("")
        private set
    
    var isLoading by mutableStateOf(false)
        private set
    
    var result by mutableStateOf<IntentResult?>(null)
        private set
    
    var errorMessage by mutableStateOf<String?>(null)
        private set
    
    var isInitialized by mutableStateOf(false)
        private set
    
    var intentList by mutableStateOf<List<String>>(emptyList())
        private set
    
    private var intentClassifier: IntentClassifier? = null
    
    init {
        initializeClassifier()
    }
    
    private fun initializeClassifier() {
        viewModelScope.launch {
            isLoading = true
            errorMessage = null
            
            try {
                Log.d(LOG_TAG, "Initializing Intent Classifier...")
                intentClassifier = IntentClassifier(application)
                
                val success = intentClassifier!!.initialize()
                
                if (success) {
                    isInitialized = true
                    intentList = intentClassifier!!.getIntentList()
                    Log.d(LOG_TAG, "✅ Intent Classifier initialized with ${intentList.size} intents")
                } else {
                    errorMessage = "Failed to initialize Intent Classifier"
                    Log.e(LOG_TAG, "❌ Failed to initialize Intent Classifier")
                }
            } catch (e: Exception) {
                errorMessage = "Error initializing: ${e.localizedMessage}"
                Log.e(LOG_TAG, "❌ Exception during initialization", e)
            } finally {
                isLoading = false
            }
        }
    }
    
    fun updateInputText(text: String) {
        inputText = text
        // Clear previous results when text changes
        result = null
        errorMessage = null
    }
    
    fun classifyIntent() {
        if (inputText.isBlank()) {
            errorMessage = "Please enter some text to classify"
            return
        }
        
        if (!isInitialized) {
            errorMessage = "Intent classifier not initialized"
            return
        }
        
        viewModelScope.launch {
            isLoading = true
            errorMessage = null
            result = null
            
            try {
                Log.d(LOG_TAG, "🔍 Classifying text: '$inputText'")
                
                val classificationResult = intentClassifier?.classifyIntent(inputText)
                
                if (classificationResult != null) {
                    result = classificationResult
                    Log.d(LOG_TAG, "✅ Classification successful: ${classificationResult.intent}")
                } else {
                    errorMessage = "Failed to classify intent"
                    Log.e(LOG_TAG, "❌ Classification returned null")
                }
            } catch (e: Exception) {
                errorMessage = "Classification error: ${e.localizedMessage}"
                Log.e(LOG_TAG, "❌ Exception during classification", e)
            } finally {
                isLoading = false
            }
        }
    }
    
    fun clearResults() {
        result = null
        errorMessage = null
        inputText = ""
    }
    
    fun tryExampleText(example: String) {
        updateInputText(example)
    }
    
    override fun onCleared() {
        super.onCleared()
        intentClassifier?.close()
        Log.d(LOG_TAG, "🔒 ViewModel cleared, classifier closed")
    }
}