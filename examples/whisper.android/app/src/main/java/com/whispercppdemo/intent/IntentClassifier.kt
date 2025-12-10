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
    private var hfTokenizer: HFTokenizer? = null
    private var intentMapping: Map<Int, String> = emptyMap()
    private var isInitialized = false
    private val slotExtractor = SlotExtractor()
    
    companion object {
        private const val MAX_SEQUENCE_LENGTH = 256  // Updated to match your model
    }
    
    suspend fun initialize(): Boolean {
        return try {
            Log.d(LOG_TAG, "Initializing Intent Classifier...")
            
            // Load metadata
            loadMetadata()
            
            // Load HuggingFace tokenizer
            loadHFTokenizer()
            
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
    
    private fun loadHFTokenizer() {
        // Load tokenizer.json file as bytes
        val tokenizerBytes = context.assets.open("tokenizer/tokenizer.json").use { inputStream ->
            inputStream.readBytes()
        }
        
        // Initialize HuggingFace tokenizer
        hfTokenizer = HFTokenizer(tokenizerBytes)
        Log.d(LOG_TAG, "🤗 HuggingFace tokenizer loaded successfully")
    }

    private fun loadIntentClassifier() {
        val model = loadModelFile("Intent classifier 10-12-2025.tflite")
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
        // Use HuggingFace tokenizer for proper BERT WordPiece tokenization
        val result = hfTokenizer!!.tokenize(text)
        
        // Ensure the sequences are exactly MAX_SEQUENCE_LENGTH
        val inputIds = IntArray(MAX_SEQUENCE_LENGTH)
        val attentionMask = IntArray(MAX_SEQUENCE_LENGTH)
        
        // Copy tokens up to MAX_SEQUENCE_LENGTH, padding or truncating as needed
        val tokenCount = minOf(result.ids.size, MAX_SEQUENCE_LENGTH)
        for (i in 0 until tokenCount) {
            inputIds[i] = result.ids[i].toInt()
            attentionMask[i] = result.attentionMask[i].toInt()
        }
        
        // The rest remain as 0 (padding)
        Log.d(LOG_TAG, "🔤 HF Tokenized '${text.take(50)}${if(text.length > 50) "..." else ""}' -> ${tokenCount} tokens")
        Log.d(LOG_TAG, "🔤 First 10 input_ids: ${inputIds.take(10).joinToString()}")
        Log.d(LOG_TAG, "🔤 Valid tokens: ${attentionMask.sum()}")
        
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
        hfTokenizer?.close()
        isInitialized = false
        Log.d(LOG_TAG, "🔒 Intent classifier closed")
    }
}