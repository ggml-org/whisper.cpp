package com.whispercppdemo.recorder

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import com.whispercppdemo.media.encodeWaveFile
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class Recorder {
    private val scope: CoroutineScope = CoroutineScope(
        Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    )
    private var recorder: AudioRecordThread? = null

    suspend fun startRecording(outputFile: File, onError: (Exception) -> Unit) = withContext(scope.coroutineContext) {
        recorder = AudioRecordThread(outputFile, onError)
        recorder?.start()
    }

    suspend fun stopRecording() = withContext(scope.coroutineContext) {
        recorder?.stopRecording()
        @Suppress("BlockingMethodInNonBlockingContext")
        recorder?.join()
        recorder = null
    }
}

private class AudioRecordThread(
    private val outputFile: File,
    private val onError: (Exception) -> Unit
) :
    Thread("AudioRecorder") {
    private var quit = AtomicBoolean(false)

    @SuppressLint("MissingPermission")
    override fun run() {
        try {
            val bufferSize = AudioRecord.getMinBufferSize(
                16000,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            ) * 4
            val buffer = ShortArray(bufferSize / 2)

            val audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                16000,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )

            try {
                audioRecord.startRecording()

                val allData = mutableListOf<Short>()

                while (!quit.get()) {
                    val read = audioRecord.read(buffer, 0, buffer.size)
                    if (read > 0) {
                        for (i in 0 until read) {
                            allData.add(buffer[i])
                        }
                    } else {
                        throw java.lang.RuntimeException("audioRecord.read returned $read")
                    }
                }

                audioRecord.stop()
                encodeWaveFile(outputFile, allData.toShortArray())
            } finally {
                audioRecord.release()
            }
        } catch (e: Exception) {
            onError(e)
        }
    }

    fun stopRecording() {
        quit.set(true)
    }
}