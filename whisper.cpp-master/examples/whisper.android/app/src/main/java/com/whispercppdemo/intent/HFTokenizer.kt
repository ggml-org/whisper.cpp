package com.whispercppdemo.intent

import android.util.Log
import org.json.JSONObject

/**
 * HuggingFace Tokenizer wrapper using Rust-based implementation
 * Provides proper BERT WordPiece tokenization for improved accuracy
 */
class HFTokenizer(tokenizerBytes: ByteArray) {
    
    private val TAG = "HFTokenizer"
    
    data class Result(
        val ids: LongArray = longArrayOf(),
        val attentionMask: LongArray = longArrayOf()
    )

    private val tokenizerPtr: Long = createTokenizer(tokenizerBytes)

    init {
        if (tokenizerPtr == 0L) {
            throw RuntimeException("Failed to create tokenizer from provided bytes")
        }
        Log.d(TAG, "✅ HuggingFace tokenizer initialized successfully")
    }

    /**
     * Tokenize text using proper BERT WordPiece tokenization
     * @param text Input text to tokenize
     * @return Result containing token ids and attention mask
     */
    fun tokenize(text: String): Result {
        val output = tokenize(tokenizerPtr, text)
        
        return try {
            // Deserialize the JSON response from Rust
            val jsonObject = JSONObject(output)
            val idsArray = jsonObject.getJSONArray("ids")
            val ids = LongArray(idsArray.length())
            for (i in 0 until idsArray.length()) {
                ids[i] = (idsArray.get(i) as Int).toLong()
            }
            
            val attentionMaskArray = jsonObject.getJSONArray("attention_mask")
            val attentionMask = LongArray(attentionMaskArray.length())
            for (i in 0 until attentionMaskArray.length()) {
                attentionMask[i] = (attentionMaskArray.get(i) as Int).toLong()
            }
            
            Log.d(TAG, "🔤 Tokenized '${text.take(50)}${if(text.length > 50) "..." else ""}' -> ${ids.size} tokens")
            
            Result(ids, attentionMask)
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error parsing tokenization result", e)
            Result() // Return empty result
        }
    }

    /**
     * Close the tokenizer and free native resources
     */
    fun close() {
        if (tokenizerPtr != 0L) {
            deleteTokenizer(tokenizerPtr)
            Log.d(TAG, "🔒 HuggingFace tokenizer closed")
        }
    }

    // Native method declarations - implemented in Rust
    private external fun createTokenizer(tokenizerBytes: ByteArray): Long
    private external fun tokenize(tokenizerPtr: Long, text: String): String
    private external fun deleteTokenizer(tokenizerPtr: Long)

    companion object {
        init {
            System.loadLibrary("hftokenizer")
        }
    }
}
