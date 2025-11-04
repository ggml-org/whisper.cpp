const path = require("path");
const { whisper } = require(path.join(
  __dirname,
  "../../build/Release/addon.node"
));
const { promisify } = require("util");

const whisperAsync = promisify(whisper);

// Example with VAD enabled
const vadParams = {
  language: "en",
  model: path.join(__dirname, "../../models/ggml-base.en.bin"),
  fname_inp: path.join(__dirname, "../../samples/jfk.wav"),
  use_gpu: true,
  flash_attn: false,
  no_prints: false,
  comma_in_time: true,
  translate: false,
  no_timestamps: false,
  detect_language: false,
  audio_ctx: 0,
  max_len: 0,
  // VAD parameters
  vad: true,
  vad_model: path.join(__dirname, "../../models/ggml-silero-v5.1.2.bin"), // You need to download this model
  vad_threshold: 0.5,
  vad_min_speech_duration_ms: 250,
  vad_min_silence_duration_ms: 100,
  vad_max_speech_duration_s: 30.0,
  vad_speech_pad_ms: 30,
  vad_samples_overlap: 0.1,
  progress_callback: (progress) => {
    console.log(`VAD Transcription progress: ${progress}%`);
  }
};

// Example without VAD (traditional approach)
const traditionalParams = {
  language: "en",
  model: path.join(__dirname, "../../models/ggml-base.en.bin"),
  fname_inp: path.join(__dirname, "../../samples/jfk.wav"),
  use_gpu: true,
  flash_attn: false,
  no_prints: false,
  comma_in_time: true,
  translate: false,
  no_timestamps: false,
  detect_language: false,
  audio_ctx: 0,
  max_len: 0,
  vad: false, // Explicitly disable VAD
  progress_callback: (progress) => {
    console.log(`Traditional transcription progress: ${progress}%`);
  }
};

async function runVADExample() {
  try {
    console.log("=== Whisper.cpp Node.js VAD Example ===\n");
    
    // Check if VAD model exists
    const fs = require('fs');
    if (!fs.existsSync(vadParams.vad_model)) {
      console.log("⚠️  VAD model not found. Please download the VAD model first:");
      console.log("   ./models/download-vad-model.sh silero-v5.1.2");
      console.log("   Or run: python models/convert-silero-vad-to-ggml.py");
      console.log("\n   Falling back to traditional transcription without VAD...\n");
      
      // Run without VAD
      console.log("🎵 Running traditional transcription...");
      const traditionalResult = await whisperAsync(traditionalParams);
      console.log("\n📝 Traditional transcription result:");
      console.log(traditionalResult);
      return;
    }
    
    console.log("🎵 Running transcription with VAD enabled...");
    console.log("VAD Parameters:");
    console.log(`  - Threshold: ${vadParams.vad_threshold}`);
    console.log(`  - Min speech duration: ${vadParams.vad_min_speech_duration_ms}ms`);
    console.log(`  - Min silence duration: ${vadParams.vad_min_silence_duration_ms}ms`);
    console.log(`  - Max speech duration: ${vadParams.vad_max_speech_duration_s}s`);
    console.log(`  - Speech padding: ${vadParams.vad_speech_pad_ms}ms`);
    console.log(`  - Samples overlap: ${vadParams.vad_samples_overlap}\n`);
    
    const startTime = Date.now();
    const vadResult = await whisperAsync(vadParams);
    const vadDuration = Date.now() - startTime;
    
    console.log("\n✅ VAD transcription completed!");
    console.log(`⏱️  Processing time: ${vadDuration}ms`);
    console.log("\n📝 VAD transcription result:");
    console.log(vadResult);
    
    // Compare with traditional approach
    console.log("\n🔄 Running traditional transcription for comparison...");
    const traditionalStartTime = Date.now();
    const traditionalResult = await whisperAsync(traditionalParams);
    const traditionalDuration = Date.now() - traditionalStartTime;
    
    console.log("\n✅ Traditional transcription completed!");
    console.log(`⏱️  Processing time: ${traditionalDuration}ms`);
    console.log("\n📝 Traditional transcription result:");
    console.log(traditionalResult);
    
    // Performance comparison
    console.log("\n📊 Performance Comparison:");
    console.log(`VAD:         ${vadDuration}ms`);
    console.log(`Traditional: ${traditionalDuration}ms`);
    const speedup = traditionalDuration / vadDuration;
    if (speedup > 1) {
      console.log(`🚀 VAD is ${speedup.toFixed(2)}x faster!`);
    } else {
      console.log(`ℹ️  Traditional approach was ${(1/speedup).toFixed(2)}x faster in this case.`);
    }
    
  } catch (error) {
    console.error("❌ Error during transcription:", error);
  }
}

// Run the example
if (require.main === module) {
  runVADExample();
}

module.exports = {
  runVADExample,
  vadParams,
  traditionalParams
}; 