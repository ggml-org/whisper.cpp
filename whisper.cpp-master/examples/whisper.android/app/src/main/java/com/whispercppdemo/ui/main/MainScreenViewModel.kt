package com.whispercppdemo.ui.main

import android.app.Application
import android.content.Context
import android.content.pm.PackageManager
import android.media.MediaPlayer
import android.os.Build
import android.os.Environment
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.net.toUri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.whispercppdemo.media.decodeWaveFile
import com.whispercppdemo.recorder.AudioStreamRecorder
import com.whispercppdemo.recorder.Recorder
import com.whispercppdemo.intent.IntentClassifier
import com.whispercppdemo.intent.IntentResult
import com.whispercpp.whisper.WhisperContext
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val LOG_TAG = "MainScreenViewModel"

// Prompt to improve short command recognition
private const val PROMPT = "Voice commands, Start Running, stop, start, record, play, pause, resume, next, previous, open, left, right, go, back, help, exit, set an, alarm, timer, stopwatch, set an alarm, set alarm, set timer 5 min, start stopwatch, stop timer, reset stopwatch, pause timer, resume timer, call mom, call dad, call john, call alex, call sister, call brother, call wife, call husband, call shreya, call best friend, mute, unmute, volume up, volume down, brightness up, brightness down, increase, decrease, dim, silence, music, song, track, what's the weather today, weather tomorrow, forecast, rain, snow, air quality, steps today, steps this week, weekly steps, sleep score, sleep score yesterday, sleep score last week, what is my heart rate today, weekly heart rate, how much calories today, calories yesterday, spo2 level today, spo2 yesterday, stress alert, set stress 80%, set heart rate high 120, set heart rate low 50, set spo2 low 90, set distance goal 5 km, set steps goal 10000, set calories goal 2000, update sleep goal 8 hours, hiking, running, walking, treadmill, swimming, rowing, yoga, meditation, cycling, indoor cycling, strength training, workout start, workout stop, workout pause, open weather, open spo2, measure spo2, show trend, weekly trend, last week, yesterday, today, tomorrow, DND, enable DND, disable DND, AOD on, AOD off, raise to wake on, raise to wake off, vibration on, vibration off"

class MainScreenViewModel(private val application: Application) : AndroidViewModel(application) {
    var canTranscribe by mutableStateOf(false)
        private set
    var dataLog by mutableStateOf("")
        private set
    var isRecording by mutableStateOf(false)
        private set

    private val modelsPath = File(application.filesDir, "models")
    private val samplesPath = File(application.filesDir, "samples")
    private val recordingsPath = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "WhisperRecordings")
    private val csvFile = File(recordingsPath, "transcriptions.csv")
    private var recorder: Recorder = Recorder()
    private var whisperContext: WhisperContext? = null
    private var intentClassifier: IntentClassifier? = null
    private var mediaPlayer: MediaPlayer? = null
    private var recordedFile: File? = null
    private var currentRecordingTimestamp: String? = null

    companion object {
        fun factory(application: Application): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                MainScreenViewModel(application)
            }
        }
    }
    
    // For real-time transcription
    private val audioRecorder = AudioStreamRecorder()
    var currentTranscript by mutableStateOf("")
        private set
    var transcriptionTime by mutableStateOf("")
        private set
    private var isProcessing by mutableStateOf(false)
        private set
        
    // For intent classification results
    var currentIntent by mutableStateOf("")
        private set
    var intentConfidence by mutableStateOf(0f)
        private set
    var intentSlots by mutableStateOf<Map<String, Any>>(emptyMap())
        private set
    var intentProcessingTime by mutableStateOf("")
        private set

    init {
        viewModelScope.launch {
            setupStorageDirectories()
            printSystemInfo()
            loadData()
        }
    }

    private suspend fun setupStorageDirectories() = withContext(Dispatchers.IO) {
        try {
            // Check if we have storage permission for Downloads folder
            if (!hasStoragePermission()) {
                Log.w(LOG_TAG, "Storage permission not granted for Downloads access")
                printMessage("Warning: Storage permission required for Downloads access\n")
                return@withContext
            }

            // Create recordings directory in Downloads
            if (!recordingsPath.exists()) {
                val created = recordingsPath.mkdirs()
                if (created) {
                    Log.d(LOG_TAG, "Created recordings directory: ${recordingsPath.absolutePath}")
                    printMessage("Created recordings directory in Downloads\n")
                } else {
                    Log.w(LOG_TAG, "Failed to create recordings directory: ${recordingsPath.absolutePath}")
                    printMessage("Warning: Failed to create recordings directory\n")
                }
            } else {
                Log.d(LOG_TAG, "Recordings directory already exists: ${recordingsPath.absolutePath}")
            }
            
            // Initialize CSV file with headers if it doesn't exist
            if (!csvFile.exists()) {
                initializeCsvFile()
            } else {
                Log.d(LOG_TAG, "CSV file already exists: ${csvFile.absolutePath}")
            }
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Error setting up storage directories", e)
            printMessage("Error setting up storage: ${e.localizedMessage}\n")
        }
    }

    private fun hasStoragePermission(): Boolean {
        return when {
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.R -> {
                // Android 11+ - Check if we can write to public external storage
                Environment.isExternalStorageManager()
            }
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.M -> {
                // Android 6+ - Check WRITE_EXTERNAL_STORAGE permission
                ContextCompat.checkSelfPermission(
                    application,
                    android.Manifest.permission.WRITE_EXTERNAL_STORAGE
                ) == PackageManager.PERMISSION_GRANTED
            }
            else -> {
                // Below Android 6 - Permission granted by default
                true
            }
        }
    }

    private suspend fun initializeCsvFile() = withContext(Dispatchers.IO) {
        try {
            FileWriter(csvFile).use { writer ->
                writer.append("timestamp,audio_filename,transcription,intent\n")
            }
            Log.d(LOG_TAG, "Initialized CSV file: ${csvFile.absolutePath}")
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Error initializing CSV file", e)
        }
    }

    private fun generateTimestamp(): String {
        val dateFormat = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss-SSS", Locale.getDefault())
        return dateFormat.format(Date())
    }

    private fun getCurrentTimeString(): String {
        val timeFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
        return timeFormat.format(Date())
    }

    private suspend fun saveAudioToFile(audioData: FloatArray, timestamp: String): File? = withContext(Dispatchers.IO) {
        try {
            val filename = "${timestamp}.wav"
            val audioFile = File(recordingsPath, filename)
            
            // Convert float array to WAV format
            saveAsWavFile(audioData, audioFile)
            
            Log.d(LOG_TAG, "Saved audio file: ${audioFile.absolutePath}")
            audioFile
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Error saving audio file", e)
            null
        }
    }

    private fun saveAsWavFile(audioData: FloatArray, file: File) {
        val sampleRate = 16000
        val bitsPerSample = 16
        val channels = 1
        
        // Convert float to 16-bit PCM
        val pcmData = ByteArray(audioData.size * 2)
        val buffer = ByteBuffer.wrap(pcmData).order(ByteOrder.LITTLE_ENDIAN)
        
        for (sample in audioData) {
            val pcmSample = (sample * 32767).toInt().coerceIn(-32768, 32767)
            buffer.putShort(pcmSample.toShort())
        }
        
        FileOutputStream(file).use { fos ->
            // WAV header
            fos.write("RIFF".toByteArray())
            fos.write(intToBytes(36 + pcmData.size))
            fos.write("WAVE".toByteArray())
            fos.write("fmt ".toByteArray())
            fos.write(intToBytes(16)) // PCM format chunk size
            fos.write(shortToBytes(1)) // PCM format
            fos.write(shortToBytes(channels.toShort()))
            fos.write(intToBytes(sampleRate))
            fos.write(intToBytes(sampleRate * channels * bitsPerSample / 8))
            fos.write(shortToBytes((channels * bitsPerSample / 8).toShort()))
            fos.write(shortToBytes(bitsPerSample.toShort()))
            fos.write("data".toByteArray())
            fos.write(intToBytes(pcmData.size))
            fos.write(pcmData)
        }
    }

    private fun intToBytes(value: Int): ByteArray {
        return byteArrayOf(
            (value and 0xFF).toByte(),
            ((value shr 8) and 0xFF).toByte(),
            ((value shr 16) and 0xFF).toByte(),
            ((value shr 24) and 0xFF).toByte()
        )
    }

    private fun shortToBytes(value: Short): ByteArray {
        return byteArrayOf(
            (value.toInt() and 0xFF).toByte(),
            ((value.toInt() shr 8) and 0xFF).toByte()
        )
    }

    private suspend fun saveToCsv(timestamp: String, audioFilename: String, transcription: String, intent: String) = withContext(Dispatchers.IO) {
        try {
            FileWriter(csvFile, true).use { writer ->
                // Escape any commas or quotes in the transcription and intent
                val escapedTranscription = transcription.replace("\"", "\"\"")
                val escapedIntent = intent.replace("\"", "\"\"")
                writer.append("$timestamp,\"$audioFilename\",\"$escapedTranscription\",\"$escapedIntent\"\n")
            }
            Log.d(LOG_TAG, "Saved to CSV: $audioFilename -> $transcription -> $intent")
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Error saving to CSV", e)
        }
    }

    private suspend fun printSystemInfo() {
        printMessage(String.format("System Info: %s\n", com.whispercpp.whisper.WhisperContext.getSystemInfo()))
    }

    private suspend fun loadData() {
        printMessage("Loading data...\n")
        try {
            Log.d(LOG_TAG, "Starting to copy assets...")
            copyAssets()
            Log.d(LOG_TAG, "Assets copied successfully. Loading base model...")
            loadBaseModel()
            Log.d(LOG_TAG, "Base model loaded successfully. Initializing intent classifier...")
            loadIntentClassifier()
            Log.d(LOG_TAG, "Intent classifier initialized successfully")
            canTranscribe = true
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Error during initialization", e)
            printMessage("Error: ${e.localizedMessage}\n")
            printMessage("Stack trace: ${e.stackTraceToString()}\n")
        }
    }

    private suspend fun printMessage(msg: String) = withContext(Dispatchers.Main) {
        dataLog += msg
    }

    private suspend fun copyAssets() = withContext(Dispatchers.IO) {
        // Clean and recreate directories to ensure fresh copy
        if (modelsPath.exists()) {
            modelsPath.deleteRecursively()
        }
        if (samplesPath.exists()) {
            samplesPath.deleteRecursively()
        }
        
        modelsPath.mkdirs()
        samplesPath.mkdirs()
        
        Log.d(LOG_TAG, "Copying models to: ${modelsPath.absolutePath}")
        Log.d(LOG_TAG, "Copying samples to: ${samplesPath.absolutePath}")
        
        application.copyData("models", modelsPath, ::printMessage)
        application.copyData("samples", samplesPath, ::printMessage)
        
        // Log what was actually copied
        Log.d(LOG_TAG, "Models copied:")
        modelsPath.listFiles()?.forEach { file ->
            Log.d(LOG_TAG, "  - ${file.name} (${file.length()} bytes)")
        }
        
        Log.d(LOG_TAG, "Samples copied:")
        samplesPath.listFiles()?.forEach { file ->
            Log.d(LOG_TAG, "  - ${file.name} (${file.length()} bytes)")
        }
        
        printMessage("All data copied to working directory.\n")
    }

    private suspend fun loadBaseModel() = withContext(Dispatchers.IO) {
        printMessage("Loading model...\n")
        Log.d(LOG_TAG, "Checking models directory...")
        val models = application.assets.list("models/")
        if (models != null && models.isNotEmpty()) {
            Log.d(LOG_TAG, "Found models: ${models.joinToString()}")
            try {
                whisperContext = WhisperContext.createContextFromAsset(application.assets, "models/" + models[0])
                Log.d(LOG_TAG, "Successfully created WhisperContext for model ${models[0]}")
                printMessage("Loaded model ${models[0]}.\n")
            } catch (e: Exception) {
                Log.e(LOG_TAG, "Failed to create WhisperContext", e)
                throw IllegalStateException("Failed to initialize WhisperContext: ${e.message}", e)
            }
        } else {
            Log.e(LOG_TAG, "No models found in assets/models directory")
            throw IllegalStateException("No models found in assets/models directory")
        }
    }
    
    private suspend fun loadIntentClassifier() = withContext(Dispatchers.IO) {
        printMessage("Loading intent classifier...\n")
        try {
            intentClassifier = IntentClassifier(application)
            val success = intentClassifier!!.initialize()
            if (success) {
                printMessage("Intent classifier loaded successfully.\n")
                Log.d(LOG_TAG, "Intent classifier initialized successfully")
            } else {
                printMessage("Failed to initialize intent classifier.\n")
                Log.e(LOG_TAG, "Failed to initialize intent classifier")
                intentClassifier = null
            }
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Error loading intent classifier", e)
            printMessage("Error loading intent classifier: ${e.localizedMessage}\n")
            intentClassifier = null
        }
    }

    fun benchmark() = viewModelScope.launch {
        runBenchmark(6)
    }

    fun transcribeSample() = viewModelScope.launch {
        transcribeAudio(getFirstSample())
    }

    private suspend fun runBenchmark(nthreads: Int) {
        if (!canTranscribe) {
            return
        }

        canTranscribe = false

        printMessage("Running benchmark. This will take minutes...\n")
        whisperContext?.benchMemory(nthreads)?.let{ printMessage(it) }
        printMessage("\n")
        whisperContext?.benchGgmlMulMat(nthreads)?.let{ printMessage(it) }

        canTranscribe = true
    }

    private suspend fun getFirstSample(): File = withContext(Dispatchers.IO) {
        val files = samplesPath.listFiles()
        Log.d(LOG_TAG, "Available sample files in ${samplesPath.absolutePath}:")
        files?.forEach { file ->
            Log.d(LOG_TAG, "  - ${file.name} (${file.length()} bytes, exists: ${file.exists()})")
        }
        
        // Look for samples_jfk.wav specifically first
        val samplesJfkFile = files?.find { it.name.equals("samples_jfk.wav", ignoreCase = true) }
        if (samplesJfkFile != null && samplesJfkFile.exists()) {
            Log.d(LOG_TAG, "Selected samples_jfk.wav: ${samplesJfkFile.absolutePath}")
            return@withContext samplesJfkFile
        }
        
        // Look for any file containing "jfk" in the name
        val jfkFile = files?.find { it.name.contains("jfk", ignoreCase = true) }
        if (jfkFile != null && jfkFile.exists()) {
            Log.d(LOG_TAG, "Selected JFK sample: ${jfkFile.absolutePath}")
            return@withContext jfkFile
        }
        
        // Fall back to first available wav file
        val firstFile = files?.firstOrNull { it.isFile && it.name.endsWith(".wav", ignoreCase = true) }
        if (firstFile != null && firstFile.exists()) {
            Log.d(LOG_TAG, "Selected first sample: ${firstFile.absolutePath}")
            return@withContext firstFile
        }
        
        throw IllegalStateException("No sample files found in ${samplesPath.absolutePath}")
    }

    private suspend fun readAudioSamples(file: File): FloatArray = withContext(Dispatchers.IO) {
        stopPlayback()
        startPlayback(file)
        return@withContext decodeWaveFile(file)
    }

    private suspend fun stopPlayback() = withContext(Dispatchers.Main) {
        mediaPlayer?.stop()
        mediaPlayer?.release()
        mediaPlayer = null
    }

    private suspend fun startPlayback(file: File) = withContext(Dispatchers.Main) {
        mediaPlayer = MediaPlayer.create(application, file.absolutePath.toUri())
        mediaPlayer?.start()
    }

    private suspend fun transcribeAudio(file: File) {
        if (!canTranscribe) {
            return
        }

        canTranscribe = false

        try {
            printMessage("Processing file: ${file.name} (${file.absolutePath})\n")
            printMessage("File exists: ${file.exists()}, Size: ${file.length()} bytes\n")
            printMessage("Reading wave samples... ")
            val data = readAudioSamples(file)
            printMessage("${data.size / (16000 / 1000)} ms\n")
            printMessage("Transcribing data...\n")
            val start = System.currentTimeMillis()
            val text = whisperContext?.transcribeData(data, printTimestamp = true, prompt = PROMPT)
            val elapsed = System.currentTimeMillis() - start
            printMessage("Done ($elapsed ms): \n$text\n")
            
            // Classify intent if text is not null and not empty (run asynchronously)
            text?.let { transcript ->
                if (transcript.isNotBlank()) {
                    viewModelScope.launch {
                        classifyIntentFromTranscript(transcript)
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(LOG_TAG, e)
            printMessage("Error transcribing ${file.name}: ${e.localizedMessage}\n")
        }

        canTranscribe = true
    }
    
    private suspend fun classifyIntentFromTranscript(transcript: String): String = withContext(Dispatchers.Default) {
        try {
            val classificationStart = System.currentTimeMillis()
            val result = intentClassifier?.classifyIntent(transcript)
            val classificationElapsed = System.currentTimeMillis() - classificationStart
            
            withContext(Dispatchers.Main) {
                if (result != null) {
                    // Handle irrelevant input specially
                    if (result.intent == "IrrelevantInput") {
                        currentIntent = result.slots["message"] as? String ?: "Sorry, please say again"
                        intentConfidence = result.confidence
                        intentSlots = emptyMap()  // Don't show slots for irrelevant input
                        intentProcessingTime = "Intent classified in ${classificationElapsed}ms"
                        
                        Log.d(LOG_TAG, "Irrelevant input detected: ${"%.3f".format(result.confidence)}")
                        printMessage("Irrelevant input detected\n")
                        return@withContext "IrrelevantInput"
                    } else {
                        currentIntent = result.intent
                        intentConfidence = result.confidence
                        intentSlots = result.slots
                        intentProcessingTime = "Intent classified in ${classificationElapsed}ms"
                        
                        Log.d(LOG_TAG, "Intent classified: ${result.intent} (confidence: ${"%.3f".format(result.confidence)})")
                        printMessage("Intent: ${result.intent} (${"%.1f".format(result.confidence * 100)}% confidence)\n")
                        
                        if (result.slots.isNotEmpty()) {
                            printMessage("Slots: ${result.slots}\n")
                        }
                        return@withContext result.intent
                    }
                } else {
                    currentIntent = "Classification failed"
                    intentConfidence = 0f
                    intentSlots = emptyMap()
                    intentProcessingTime = "Classification failed"
                    printMessage("Intent classification failed\n")
                    return@withContext "Classification failed"
                }
            }
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Error classifying intent", e)
            withContext(Dispatchers.Main) {
                currentIntent = "Error"
                intentConfidence = 0f
                intentSlots = emptyMap()
                intentProcessingTime = "Error: ${e.localizedMessage}"
                printMessage("Intent classification error: ${e.localizedMessage}\n")
            }
            return@withContext "Error"
        }
    }

    private val processingScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    fun toggleRecord() {
        viewModelScope.launch {
            try {
                if (isRecording) {
                    stopRecording()
                } else {
                    startRecording()
                }
            } catch (e: Exception) {
                Log.e(LOG_TAG, "Error toggling recording", e)
                printMessage("Error: ${e.localizedMessage}\n")
                cleanup()
            }
        }
    }

    private suspend fun startRecording() {
        stopPlayback()
        isRecording = true
        currentTranscript = "Listening..."
        transcriptionTime = ""
        // Clear previous intent results
        currentIntent = ""
        intentConfidence = 0f
        intentSlots = emptyMap()
        intentProcessingTime = ""
        isProcessing = true
        
        // Generate timestamp for this recording session
        currentRecordingTimestamp = generateTimestamp()
        
        processingScope.launch {
            try {
                whisperContext?.let { context ->
                    audioRecorder.startStreamingAudio()
                        .collect { audioChunk ->
                            // Process audio in background
                            try {
                                // Clear previous transcript since we're processing a new command
                                withContext(Dispatchers.Main) {
                                    currentTranscript = "Processing..."
                                    transcriptionTime = ""
                                }
                                
                                // Measure transcription time
                                val transcriptionStart = System.currentTimeMillis()
                                val result = context.transcribeData(audioChunk, printTimestamp = false, prompt = PROMPT)
                                val transcriptionElapsed = System.currentTimeMillis() - transcriptionStart
                                
                                if (result.isNotBlank()) {
                                    val timestamp = currentRecordingTimestamp ?: generateTimestamp()
                                    
                                    withContext(Dispatchers.Main) {
                                        // Replace transcript with new result since it's a command
                                        currentTranscript = result.trim()
                                        transcriptionTime = "Transcribed in ${transcriptionElapsed}ms at ${getCurrentTimeString()}"
                                        printMessage("Command detected: $result\n")
                                        printMessage("Transcription time: ${transcriptionElapsed}ms\n")
                                    }
                                    
                                    // Classify intent for the transcribed text and save to CSV
                                    launch {
                                        try {
                                            // Classify intent first
                                            val intentResult = classifyIntentFromTranscript(result.trim())
                                            
                                            // Save files with intent information
                                            val audioFile = saveAudioToFile(audioChunk, timestamp)
                                            audioFile?.let { file ->
                                                saveToCsv(timestamp, file.name, result.trim(), intentResult)
                                                withContext(Dispatchers.Main) {
                                                    printMessage("Saved to: ${file.name}\n")
                                                }
                                            }
                                        } catch (e: Exception) {
                                            Log.e(LOG_TAG, "Error saving files", e)
                                        }
                                    }
                                }
                            } catch (e: Exception) {
                                Log.e(LOG_TAG, "Error processing audio chunk", e)
                                withContext(Dispatchers.Main) {
                                    currentTranscript = "Error processing audio"
                                }
                            }
                        }
                }
            } catch (e: Exception) {
                Log.e(LOG_TAG, "Error in audio processing", e)
                withContext(Dispatchers.Main) {
                    cleanup()
                }
            }
        }
    }

    private suspend fun stopRecording() {
        isRecording = false
        isProcessing = false
        audioRecorder.stopRecording()
        recorder.stopRecording()
    }

    private fun cleanup() {
        currentRecordingTimestamp = null
        viewModelScope.launch { stopRecording() }
    }

    // Public function to get storage location for UI display
    fun getStorageLocation(): String {
        return recordingsPath.absolutePath
    }

    // Public function to get CSV file location
    fun getCsvLocation(): String {
        return csvFile.absolutePath
    }

    // Public function to check if storage is accessible
    fun isStorageAccessible(): Boolean {
        return hasStoragePermission() && recordingsPath.exists() && recordingsPath.canWrite()
    }

    override fun onCleared() {
        super.onCleared()
        processingScope.cancel()
        viewModelScope.launch {
            try {
                cleanup()
                whisperContext?.release()
                whisperContext = null
                intentClassifier?.close()
                intentClassifier = null
                stopPlayback()
            } catch (e: Exception) {
                Log.e(LOG_TAG, "Error during cleanup", e)
            }
        }
    }

    private suspend fun getTempFileForRecording() = withContext(Dispatchers.IO) {
        File.createTempFile("recording", "wav")
    }
}

private suspend fun Context.copyData(
    assetDirName: String,
    destDir: File,
    printMessage: suspend (String) -> Unit
) = withContext(Dispatchers.IO) {
    assets.list(assetDirName)?.forEach { name ->
        val assetPath = "$assetDirName/$name"
        Log.v(LOG_TAG, "Processing $assetPath...")
        val destination = File(destDir, name)
        
        // Always copy fresh - delete existing file if it exists
        if (destination.exists()) {
            destination.delete()
            Log.v(LOG_TAG, "Deleted existing file: $destination")
        }
        
        Log.v(LOG_TAG, "Copying $assetPath to $destination...")
        printMessage("Copying $name...\n")
        
        try {
            assets.open(assetPath).use { input ->
                destination.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            Log.v(LOG_TAG, "Successfully copied $assetPath to $destination (${destination.length()} bytes)")
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Failed to copy $assetPath to $destination", e)
            printMessage("Error copying $name: ${e.localizedMessage}\n")
        }
    }
}