package com.whispercppdemo.intent

import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*

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
    private var isInitialized = false
    private val slotExtractor = SlotExtractor()
    
    // BERT tokenizer constants
    private var clsTokenId = 101
    private var sepTokenId = 102
    private var padTokenId = 0
    private var unkTokenId = 100
    private var maxLength = 128
    
    companion object {
        private const val EMBEDDING_DIM = 768  // Updated for your BERT model
    }
    
    suspend fun initialize(): Boolean {
        return try {
            Log.d(LOG_TAG, "Initializing Intent Classifier...")
            
            // Load metadata
            loadMetadata()
            
            // Load BERT vocabulary
            loadBertVocabulary()
            
            // Load TFLite model
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
        
        // Load tokenizer info
        val tokenizerInfo = metadata.optJSONObject("tokenizer_info")
        if (tokenizerInfo != null) {
            maxLength = tokenizerInfo.optInt("max_length", 128)
        }
        
        Log.d(LOG_TAG, "📋 Loaded ${intentMapping.size} intent mappings, max_length: $maxLength")
    }
    
    private fun loadBertVocabulary() {
        val vocabText = context.assets.open("tokenizer/vocab.txt").bufferedReader().use { it.readText() }
        val vocabLines = vocabText.split("\n").filter { it.isNotBlank() }
        
        val tempVocabulary = mutableMapOf<String, Int>()
        vocabLines.forEachIndexed { index, token ->
            tempVocabulary[token.trim()] = index
        }
        
        vocabulary = tempVocabulary
        
        // Update special token IDs based on actual vocabulary
        clsTokenId = vocabulary["[CLS]"] ?: 101
        sepTokenId = vocabulary["[SEP]"] ?: 102
        padTokenId = vocabulary["[PAD]"] ?: 0
        unkTokenId = vocabulary["[UNK]"] ?: 100
        
        Log.d(LOG_TAG, "📝 Loaded BERT vocabulary with ${vocabulary.size} tokens")
        Log.d(LOG_TAG, "🔑 Special tokens - CLS:$clsTokenId, SEP:$sepTokenId, PAD:$padTokenId, UNK:$unkTokenId")
    }
    
    private fun loadIntentClassifier() {
        val model = loadModelFile("intent_classifier.tflite")
        intentClassifier = Interpreter(model)
        
        val inputShape = intentClassifier!!.getInputTensor(0).shape()
        val outputShape = intentClassifier!!.getOutputTensor(0).shape()
        
        Log.d(LOG_TAG, "🎯 Intent classifier loaded - Input: ${inputShape.contentToString()}, Output: ${outputShape.contentToString()}")
    }
    
    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(fileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
        
        val inputShape = intentClassifier!!.getInputTensor(0).shape()
        val outputShape = intentClassifier!!.getOutputTensor(0).shape()
        
        Log.d(LOG_TAG, "🎯 Intent classifier loaded - Input: ${inputShape.contentToString()}, Output: ${outputShape.contentToString()}")
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
            val tokenIds = tokenizeText(text)
            Log.d(LOG_TAG, "📊 Generated token IDs: ${tokenIds.take(10).joinToString()}")
            
            // Step 2: Classify intent using token IDs directly
            val probabilities = classifyWithTokens(tokenIds)
            
            // Step 3: Get best prediction
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
    
    private fun tokenizeText(text: String): IntArray {
        // Simple BERT-like tokenization
        val processedText = text.lowercase(Locale.getDefault()).trim()
        
        // Basic word tokenization (this is simplified - a full BERT tokenizer would do subword tokenization)
        val words = processedText.split("\\s+".toRegex()).filter { it.isNotBlank() }
        
        Log.d(LOG_TAG, "🔤 Original text: '$text'")
        Log.d(LOG_TAG, "🔤 Processed text: '$processedText'")
        Log.d(LOG_TAG, "🔤 Words: ${words.joinToString(", ")}")
        
        // Convert to token IDs with BERT format: [CLS] + tokens + [SEP] + padding
        val tokenIds = mutableListOf<Int>()
        
        // Add [CLS] token
        tokenIds.add(clsTokenId)
        
        // Add word tokens (simplified - in real BERT this would be subword tokens)
        var knownWords = 0
        for (word in words) {
            if (tokenIds.size >= maxLength - 1) break // Reserve space for [SEP]
            
            val tokenId = vocabulary[word] ?: unkTokenId
            tokenIds.add(tokenId)
            if (tokenId != unkTokenId) knownWords++
            Log.d(LOG_TAG, "🔤 Word '$word' -> token $tokenId")
        }
        
        // Add [SEP] token
        if (tokenIds.size < maxLength) {
            tokenIds.add(sepTokenId)
        }
        
        // Pad to maxLength
        while (tokenIds.size < maxLength) {
            tokenIds.add(padTokenId)
        }
        
        // Truncate if too long
        if (tokenIds.size > maxLength) {
            tokenIds[maxLength - 1] = sepTokenId // Ensure [SEP] at the end
        }
        
        Log.d(LOG_TAG, "🔤 Known words: $knownWords/${words.size}")
        Log.d(LOG_TAG, "� Token sequence length: ${tokenIds.size}")
        Log.d(LOG_TAG, "� First 10 tokens: ${tokenIds.take(10).joinToString()}")
        
        return tokenIds.take(maxLength).toIntArray()
    }
    
    private fun classifyWithTokens(tokenIds: IntArray): FloatArray {
        // Your model expects the input as input_ids with shape [batch_size, max_length]
        val inputArray = Array(1) { tokenIds }
        val outputArray = Array(1) { FloatArray(intentMapping.size) }
        
        intentClassifier!!.run(inputArray, outputArray)
        
        val probabilities = outputArray[0]
        
        // Log detailed probability information
        Log.d(LOG_TAG, "🎯 Raw predictions:")
        probabilities.forEachIndexed { index, prob ->
            val intent = intentMapping[index] ?: "Unknown_$index"
            Log.d(LOG_TAG, "  $index. $intent: ${"%.3f".format(prob)}")
        }
        
        // Find top 3 predictions
        val sortedIndices = probabilities.indices.sortedByDescending { probabilities[it] }
        Log.d(LOG_TAG, "🏆 Top 3 predictions:")
        sortedIndices.take(3).forEach { index ->
            val intent = intentMapping[index] ?: "Unknown_$index"
            val prob = probabilities[index]
            Log.d(LOG_TAG, "  ${intent}: ${"%.3f".format(prob)}")
        }
        
        return probabilities
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