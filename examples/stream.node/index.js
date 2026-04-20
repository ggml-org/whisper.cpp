const path = require('path');
const os = require('os');
const portAudio = require('naudiodon2');

const addonPath = path.join(__dirname, '..', '..', 'build', 'Release', 'stream.node');

const { WhisperStream } = require(addonPath);

const modelPath = path.join(__dirname, '..', '..', 'models', 'ggml-base.en.bin');
const SAMPLE_RATE = 16000;

// --- Main Application ---
async function main() {
  const whisper = new WhisperStream();
  let pendingText = ''; // Buffer for the current unconfirmed text

  console.log('Loading model...');
  whisper.startModel({
    modelPath: modelPath, 
    language: 'en',
    nThreads: 4,
    stepMs: 3000,
    lengthMs: 10000,
    keepMs: 200,
    useGpu: true, 
  });
  console.log('Model loaded.');

  const ai = new portAudio.AudioIO({
    inOptions: {
      channelCount: 1,
      sampleFormat: portAudio.SampleFormatFloat32,
      sampleRate: SAMPLE_RATE,
      deviceId: -1, 
      closeOnError: true,
    }
  });

  ai.on('data', (chunk) => {
    const floatCount = chunk.length / Float32Array.BYTES_PER_ELEMENT;
    const float32 = new Float32Array(chunk.buffer, chunk.byteOffset, floatCount);

    try {
      const result = whisper.processChunk(float32);
      if (!result || !result.text) return;

      const { text, isFinal } = result;

      if (isFinal) {
        process.stdout.write(`\r${text}\n`);
        pendingText = ''; // Reset for the next utterance
      } else {
        pendingText = text;
        // '\r' moves cursor to the start, '\x1B[K' clears the rest of the line.
        process.stdout.write(`\r${pendingText}\x1B[K`);
      }
    } catch (err) {
      console.error('Error during processing:', err);
    }
  });

  ai.on('error', (err) => console.error('Audio input error:', err));

  ai.start();
  console.log('Recording from microphone. Speak now.');
  process.stdout.write('> '); 

  const shutdown = () => {
    console.log('\nShutting down...');
    ai.quit(() => {
      whisper.freeModel();
      process.exit(0);
    });
  };

  process.on('SIGINT', shutdown);  
  process.on('SIGTERM', shutdown); 
}

main().catch((err) => {
  console.error('An unexpected error occurred:', err);
  process.exit(1);
});