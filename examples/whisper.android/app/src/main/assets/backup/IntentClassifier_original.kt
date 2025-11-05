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
    
    private var sentenceEncoder: Interpreter? = null
    private var intentClassifier: Interpreter? = null
    private var vocabulary: Map<String, Int> = emptyMap()
    private var intentMapping: Map<Int, String> = emptyMap()
    private var isInitialized = false
    private val slotExtractor = SlotExtractor()  // Add slot extractor
    
    companion object {
        private const val MAX_SEQUENCE_LENGTH = 32
        private const val EMBEDDING_DIM = 384
    }
    
    suspend fun initialize(): Boolean {
        return try {
            Log.d(LOG_TAG, "Initializing Intent Classifier...")
            
            // Load metadata
            loadMetadata()
            
            // Load vocabulary
            loadVocabulary()
            
            // Load TFLite models
            loadSentenceEncoder()
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
        val vocabularyJson = context.assets.open("vocabulary.json").bufferedReader().use { it.readText() }
        val vocabObject = JSONObject(vocabularyJson)
        
        val tempVocabulary = mutableMapOf<String, Int>()
        vocabObject.keys().forEach { word ->
            tempVocabulary[word] = vocabObject.getInt(word)
        }
        
        vocabulary = tempVocabulary
        Log.d(LOG_TAG, "📝 Loaded vocabulary with ${vocabulary.size} words")
    }
    
    private fun loadSentenceEncoder() {
        val model = loadModelFile("lightweight_sentence_encoder.tflite")
        sentenceEncoder = Interpreter(model)
        
        val inputShape = sentenceEncoder!!.getInputTensor(0).shape()
        val outputShape = sentenceEncoder!!.getOutputTensor(0).shape()
        
        Log.d(LOG_TAG, "🔤 Sentence encoder loaded - Input: ${inputShape.contentToString()}, Output: ${outputShape.contentToString()}")
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
    
    fun classifyIntent(text: String): IntentResult? {
        if (!isInitialized) {
            Log.e(LOG_TAG, "❌ Intent classifier not initialized")
            return null
        }

        return try {
            Log.d(LOG_TAG, "🔍 Classifying: '$text'")
            
            // Step 1: Convert text to embeddings
            val embeddings = textToEmbeddings(text)
            Log.d(LOG_TAG, "📊 Generated embeddings: ${embeddings.size} dimensions")
            
            // Check if embeddings are all zeros (from dummy encoder)
            val isAllZeros = embeddings.all { it == 0f }
            val nonZeroCount = embeddings.count { it != 0f }
            Log.d(LOG_TAG, "🔍 Embedding analysis: isAllZeros=$isAllZeros, nonZeroCount=$nonZeroCount")
            
            // Step 2: Classify intent using embeddings
            val probabilities = classifyWithEmbeddings(embeddings)
            
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
            Log.d(LOG_TAG, "🔍 Intent mapping check - bestIndex: $bestIndex, mapping size: ${intentMapping.size}")
            Log.d(LOG_TAG, "🔍 Expected for zero embeddings: OpenApp (from Python test)")
            
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
    }    private fun textToEmbeddings(text: String): FloatArray {
        // Preprocess text
        val processedText = text.lowercase(Locale.getDefault()).trim()
        val words = processedText.split("\\s+".toRegex())
        
        Log.d(LOG_TAG, "🔤 Original text: '$text'")
        Log.d(LOG_TAG, "🔤 Processed text: '$processedText'")
        Log.d(LOG_TAG, "🔤 Words: ${words.joinToString(", ")}")
        
        // Convert to token sequence
        val tokenSequence = FloatArray(MAX_SEQUENCE_LENGTH) { 0f }  // Changed to FloatArray
        
        var knownWords = 0
        words.take(MAX_SEQUENCE_LENGTH).forEachIndexed { index, word ->
            val tokenId = vocabulary[word] ?: 0
            tokenSequence[index] = tokenId.toFloat()  // Convert to float
            if (tokenId != 0) knownWords++
            Log.d(LOG_TAG, "🔤 Word '$word' -> token $tokenId")
        }
        
        Log.d(LOG_TAG, "🔤 Known words: $knownWords/${words.size} (${if (words.isNotEmpty()) (knownWords * 100 / words.size) else 0}%)")
        Log.d(LOG_TAG, "🔤 Token sequence: ${tokenSequence.take(10).joinToString { "%.0f".format(it) }}")
        
        // Run sentence encoder with float input
        val inputArray = Array(1) { tokenSequence }
        val outputArray = Array(1) { FloatArray(EMBEDDING_DIM) }
        
        sentenceEncoder!!.run(inputArray, outputArray)
        
        val embeddings = outputArray[0]
        
        // Log embedding statistics for debugging
        val mean = embeddings.average().toFloat()
        val std = kotlin.math.sqrt(embeddings.map { (it - mean) * (it - mean) }.average()).toFloat()
        val min = embeddings.minOrNull() ?: 0f
        val max = embeddings.maxOrNull() ?: 0f
        
        Log.d(LOG_TAG, "📊 Embedding stats: mean=${"%.6f".format(mean)}, std=${"%.6f".format(std)}, range=[${"%.3f".format(min)}, ${"%.3f".format(max)}]")
        Log.d(LOG_TAG, "📊 First 10 embedding values: ${embeddings.take(10).map { "%.3f".format(it) }.joinToString()}")
        
        return embeddings
    }
    
    private fun classifyWithEmbeddings(embeddings: FloatArray): FloatArray {
        val inputArray = Array(1) { embeddings }
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
        sentenceEncoder?.close()
        intentClassifier?.close()
        isInitialized = false
        Log.d(LOG_TAG, "🔒 Intent classifier closed")
    }
}