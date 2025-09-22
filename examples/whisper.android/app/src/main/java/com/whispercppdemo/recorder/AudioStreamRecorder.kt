package com.whispercppdemo.recorder

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class AudioStreamRecorder {
    private val scope: CoroutineScope = CoroutineScope(
        Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    )
    
    companion object {
        private const val SAMPLE_RATE = 16000 // Required by Whisper
        private const val CHANNELS = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_FLOAT
        private const val BUFFER_SIZE_MS = 5000 // 5 seconds buffer for commands
        private const val CHUNK_SIZE_MS = 1000 // Process 1 second chunks for better context
        private const val MIN_AUDIO_LEVEL = 0.01f // Minimum audio level to consider as speech
    }

    private var audioRecord: AudioRecord? = null
    private val isRecording = AtomicBoolean(false)
    
    @SuppressLint("MissingPermission")
    suspend fun startStreamingAudio(): Flow<FloatArray> = flow {
        val minBufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            CHANNELS,
            AUDIO_FORMAT
        )
        
        // Buffer for 2 seconds of audio
        val bufferSize = (SAMPLE_RATE * (BUFFER_SIZE_MS / 1000f)).toInt()
        val chunkSize = (SAMPLE_RATE * (CHUNK_SIZE_MS / 1000f)).toInt()
        
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            SAMPLE_RATE,
            CHANNELS,
            AUDIO_FORMAT,
            minBufferSize.coerceAtLeast(bufferSize)
        )
        
        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            throw IllegalStateException("Failed to initialize AudioRecord")
        }
        
        val buffer = FloatArray(chunkSize)
        val accumulator = FloatArray(SAMPLE_RATE * 5) // 5 seconds max
        var accumulatorPos = 0
        var silenceFrames = 0
        var hasVoice = false
        
        isRecording.set(true)
        audioRecord?.startRecording()
        
        try {
            while (isRecording.get()) {
                val readResult = audioRecord?.read(buffer, 0, buffer.size, AudioRecord.READ_BLOCKING) ?: 0
                if (readResult > 0) {
                    // Check for voice activity
                    val maxAmplitude = buffer.maxOf { kotlin.math.abs(it) }
                    
                    if (maxAmplitude > MIN_AUDIO_LEVEL) {
                        hasVoice = true
                        silenceFrames = 0
                    } else if (hasVoice) {
                        silenceFrames++
                    }
                    
                    // Copy to accumulator
                    buffer.copyInto(accumulator, accumulatorPos)
                    accumulatorPos += readResult
                    
                    // If we have voice and either hit max buffer or have enough silence, emit
                    if (hasVoice && (accumulatorPos >= accumulator.size || silenceFrames >= 2)) {
                        emit(accumulator.copyOfRange(0, accumulatorPos))
                        accumulatorPos = 0
                        hasVoice = false
                        silenceFrames = 0
                    }
                }
            }
            // Emit any remaining audio if we have voice activity
            if (hasVoice && accumulatorPos > 0) {
                emit(accumulator.copyOfRange(0, accumulatorPos))
            }
        } finally {
            withContext(scope.coroutineContext) {
                stopRecording()
            }
        }
    }
    
    fun stopRecording() {
        isRecording.set(false)
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }
}