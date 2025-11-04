package com.whispercppdemo.intent

import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.math.exp

private const val LOG_TAG = "IntentClassifier"

data class IntentResult(
    val intent: String,
    val confidence: Float,
    val allProbabilities: Map<String, Float>,
    val slots: Map<String, Any> = emptyMap(),
    val slotConfidence: Float = 0f
)

class IntentClassifier(private val context: Context) {
    
    private var intentClassifier: Interpreter? = null
    private var vocabulary: Map<String, Int> = emptyMap()
    private var intentMapping: Map<Int, String> = emptyMap()
    private var specialTokens: Map<String, Int> = emptyMap()
    private var isInitialized = false
    private val slotExtractor = SlotExtractor()
    
    companion object {
        private const val MAX_SEQUENCE_LENGTH = 256  // Updated to match your model
        private const val CLS_TOKEN = "[CLS]"
        private const val SEP_TOKEN = "[SEP]"
        private const val PAD_TOKEN = "[PAD]"
        private const val UNK_TOKEN = "[UNK]"
    }
    
    suspend fun initialize(): Boolean {
        return try {
            Log.d(LOG_TAG, "Initializing Intent Classifier...")
            
            // Load metadata
            loadMetadata()
            
            // Load vocabulary and special tokens
            loadVocabulary()
            loadSpecialTokens()
            
            // Load the complete TFLite model (end-to-end)
            loadIntentClassifier()
            
            isInitialized = true
            Log.d(LOG_TAG, "✅ Intent Classifier initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(LOG_TAG, "❌ Failed to initialize Intent Classifier", e)
            false
        }
    }
    
    private fun loadMetadata() {
        val metadataJson = context.assets.open("model_metadata.json").bufferedReader().use { it.readText() }
        val metadata = JSONObject(metadataJson)
        
        // Load intent mapping
        val intentMappingObj = metadata.getJSONObject("intent_mapping")
        val tempIntentMapping = mutableMapOf<Int, String>()
        
        intentMappingObj.keys().forEach { key ->
            tempIntentMapping[key.toInt()] = intentMappingObj.getString(key)
        }
        
        intentMapping = tempIntentMapping
        Log.d(LOG_TAG, "📋 Loaded ${intentMapping.size} intent mappings")
    }
    
    private fun loadVocabulary() {
        val vocabText = context.assets.open("tokenizer/vocab.txt").bufferedReader().use { it.readText() }
        val tempVocabulary = mutableMapOf<String, Int>()
        
        vocabText.lines().forEachIndexed { index, line ->
            val token = line.trim()
            if (token.isNotEmpty()) {
                tempVocabulary[token] = index
            }
        }
        
        vocabulary = tempVocabulary
        Log.d(LOG_TAG, "📝 Loaded vocabulary with ${vocabulary.size} tokens")
    }
    
    private fun loadSpecialTokens() {
        val tempSpecialTokens = mutableMapOf<String, Int>()
        tempSpecialTokens[CLS_TOKEN] = vocabulary[CLS_TOKEN] ?: 101
        tempSpecialTokens[SEP_TOKEN] = vocabulary[SEP_TOKEN] ?: 102
        tempSpecialTokens[PAD_TOKEN] = vocabulary[PAD_TOKEN] ?: 0
        tempSpecialTokens[UNK_TOKEN] = vocabulary[UNK_TOKEN] ?: 100
        
        specialTokens = tempSpecialTokens
        Log.d(LOG_TAG, "🎯 Loaded special tokens: $specialTokens")
    }
    
    private fun loadIntentClassifier() {
        val model = loadModelFile("intent_classifier.tflite")
        intentClassifier = Interpreter(model)
        
        val inputDetails = intentClassifier!!.inputTensorCount
        val outputDetails = intentClassifier!!.outputTensorCount
        
        Log.d(LOG_TAG, "🎯 Complete intent classifier loaded - Inputs: $inputDetails, Outputs: $outputDetails")
        
        // Log input/output shapes
        for (i in 0 until inputDetails) {
            val shape = intentClassifier!!.getInputTensor(i).shape()
            Log.d(LOG_TAG, "  Input $i shape: ${shape.contentToString()}")
        }
        
        for (i in 0 until outputDetails) {
            val shape = intentClassifier!!.getOutputTensor(i).shape()
            Log.d(LOG_TAG, "  Output $i shape: ${shape.contentToString()}")
        }
    }
    
    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(fileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun classifyIntent(text: String): IntentResult? {
        if (!isInitialized) {
            Log.e(LOG_TAG, "❌ Intent classifier not initialized")
            return null
        }

        return try {
            Log.d(LOG_TAG, "🔍 Classifying: '$text'")
            
            // Step 1: Tokenize text using BERT tokenizer
            val tokenization = tokenizeText(text)
            val inputIds = tokenization.first
            val attentionMask = tokenization.second
            
            Log.d(LOG_TAG, "📊 Tokenized - InputIds: ${inputIds.take(10).joinToString()}")
            Log.d(LOG_TAG, "📊 AttentionMask: ${attentionMask.take(10).joinToString()}")
            
            // Step 2: Run the complete end-to-end model
            val logits = runCompleteModel(inputIds, attentionMask)
            
            // Step 3: Apply softmax and get predictions
            val probabilities = softmax(logits)
            val bestIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            val bestIntent = intentMapping[bestIndex] ?: "Unknown"
            val confidence = probabilities[bestIndex]
            
            // Create probability map for all intents
            val allProbabilities = mutableMapOf<String, Float>()
            probabilities.forEachIndexed { index, probability ->
                val intent = intentMapping[index] ?: "Unknown_$index"
                allProbabilities[intent] = probability
            }
            
            Log.d(LOG_TAG, "✅ Final result: $bestIntent (confidence: ${"%.3f".format(confidence)})")
            
            // Log top 3 predictions
            val sortedIndices = probabilities.indices.sortedByDescending { probabilities[it] }
            Log.d(LOG_TAG, "🏆 Top 3 predictions:")
            sortedIndices.take(3).forEach { index ->
                val intent = intentMapping[index] ?: "Unknown_$index"
                val prob = probabilities[index]
                Log.d(LOG_TAG, "  ${intent}: ${"%.3f".format(prob)}")
            }
            
            // Step 4: Extract slots for the predicted intent
            val slotResult = slotExtractor.extractSlots(text, bestIntent)
            
            IntentResult(
                intent = bestIntent,
                confidence = confidence,
                allProbabilities = allProbabilities,
                slots = slotResult.slots,
                slotConfidence = slotResult.confidence
            )
        } catch (e: Exception) {
            Log.e(LOG_TAG, "❌ Error classifying intent", e)
            null
        }
    }
    
    private fun tokenizeText(text: String): Pair<IntArray, IntArray> {
        // Preprocess text (lowercase, basic cleaning)
        val processedText = text.lowercase(Locale.getDefault()).trim()
        
        // Basic BERT-like tokenization
        val tokens = mutableListOf<String>()
        tokens.add(CLS_TOKEN)  // Add [CLS] token at the beginning
        
        // Simple word tokenization (you might want to implement subword tokenization)
        val words = processedText.split("\\s+".toRegex())
        for (word in words) {
            if (word.isNotEmpty()) {
                // Simple tokenization - can be improved with subword tokenization
                val cleanWord = word.replace(Regex("[^a-zA-Z0-9]"), "")
                if (cleanWord.isNotEmpty()) {
                    tokens.add(cleanWord)
                }
            }
        }
        
        tokens.add(SEP_TOKEN)  // Add [SEP] token at the end
        
        // Convert tokens to IDs
        val inputIds = IntArray(MAX_SEQUENCE_LENGTH) { specialTokens[PAD_TOKEN] ?: 0 }
        val attentionMask = IntArray(MAX_SEQUENCE_LENGTH) { 0 }
        
        val maxTokens = minOf(tokens.size, MAX_SEQUENCE_LENGTH)
        for (i in 0 until maxTokens) {
            val token = tokens[i]
            inputIds[i] = vocabulary[token] ?: specialTokens[UNK_TOKEN] ?: 100
            attentionMask[i] = 1
        }
        
        Log.d(LOG_TAG, "🔤 Tokenized '${processedText}' -> ${tokens.take(10).joinToString()}")
        Log.d(LOG_TAG, "🔤 Token count: ${tokens.size}, Valid tokens: ${attentionMask.sum()}")
        
        return Pair(inputIds, attentionMask)
    }
    
    private fun runCompleteModel(inputIds: IntArray, attentionMask: IntArray): FloatArray {
        // Prepare inputs for TFLite model
        val inputIdsArray = Array(1) { inputIds }
        val attentionMaskArray = Array(1) { attentionMask }
        
        // Prepare output
        val outputArray = Array(1) { FloatArray(intentMapping.size) }
        
        // Run inference
        val inputs = arrayOf(inputIdsArray, attentionMaskArray)
        val outputs = mapOf(0 to outputArray)
        
        intentClassifier!!.runForMultipleInputsOutputs(inputs, outputs)
        
        val logits = outputArray[0]
        
        Log.d(LOG_TAG, "🎯 Model output logits: ${logits.take(5).map { "%.3f".format(it) }.joinToString()}")
        
        return logits
    }
    
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val expValues = logits.map { exp(it - maxLogit) }
        val sumExp = expValues.sum()
        return expValues.map { (it / sumExp).toFloat() }.toFloatArray()
    }
    
    fun getIntentList(): List<String> {
        return intentMapping.values.sorted()
    }
    
    fun close() {
        intentClassifier?.close()
        isInitialized = false
        Log.d(LOG_TAG, "🔒 Intent classifier closed")
    }
}