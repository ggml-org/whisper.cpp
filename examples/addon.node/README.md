# whisper.cpp Node.js addon

This is an addon demo that can **perform whisper model reasoning in `node` and `electron` environments**, based on [cmake-js](https://github.com/cmake-js/cmake-js).
It can be used as a reference for using the whisper.cpp project in other node projects.

This addon now supports **Voice Activity Detection (VAD)** for improved transcription performance.

## Install

```shell
npm install
```

## Compile

Make sure it is in the project root directory and compiled with make-js.

```shell
npx cmake-js compile -T addon.node -B Release
```

For Electron addon and cmake-js options, you can see [cmake-js](https://github.com/cmake-js/cmake-js) and make very few configuration changes.

> Such as appointing special cmake path:
> ```shell
> npx cmake-js compile -c 'xxx/cmake' -T addon.node -B Release
> ```

## Run

### Basic Usage

```shell
cd examples/addon.node

node index.js --language='language' --model='model-path' --fname_inp='file-path'
```

### VAD (Voice Activity Detection) Usage

Run the VAD example with performance comparison:

```shell
node vad-example.js
```

## Voice Activity Detection (VAD) Support

VAD can significantly improve transcription performance by only processing speech segments, which is especially beneficial for audio files with long periods of silence.

### VAD Model Setup

Before using VAD, download a VAD model:

```shell
# From the whisper.cpp root directory
./models/download-vad-model.sh silero-v5.1.2
```

### VAD Parameters

All VAD parameters are optional and have sensible defaults:

- `vad`: Enable VAD (default: false)
- `vad_model`: Path to VAD model file (required when VAD enabled)
- `vad_threshold`: Speech detection threshold 0.0-1.0 (default: 0.5)
- `vad_min_speech_duration_ms`: Min speech duration in ms (default: 250)
- `vad_min_silence_duration_ms`: Min silence duration in ms (default: 100)
- `vad_max_speech_duration_s`: Max speech duration in seconds (default: FLT_MAX)
- `vad_speech_pad_ms`: Speech padding in ms (default: 30)
- `vad_samples_overlap`: Sample overlap 0.0-1.0 (default: 0.1)

### JavaScript API Example

```javascript
const path = require("path");
const { whisper } = require(path.join(__dirname, "../../build/Release/addon.node"));
const { promisify } = require("util");

const whisperAsync = promisify(whisper);

// With VAD enabled
const vadParams = {
  language: "en",
  model: path.join(__dirname, "../../models/ggml-base.en.bin"),
  fname_inp: path.join(__dirname, "../../samples/jfk.wav"),
  vad: true,
  vad_model: path.join(__dirname, "../../models/ggml-silero-v5.1.2.bin"),
  vad_threshold: 0.5,
  progress_callback: (progress) => console.log(`Progress: ${progress}%`)
};

whisperAsync(vadParams).then(result => console.log(result));
```

## Supported Parameters

Both traditional whisper.cpp parameters and new VAD parameters are supported:

- `language`: Language code (e.g., "en", "es", "fr")
- `model`: Path to whisper model file
- `fname_inp`: Path to input audio file
- `use_gpu`: Enable GPU acceleration (default: true)
- `flash_attn`: Enable flash attention (default: false)
- `no_prints`: Disable console output (default: false)
- `no_timestamps`: Disable timestamps (default: false)
- `detect_language`: Auto-detect language (default: false)
- `audio_ctx`: Audio context size (default: 0)
- `max_len`: Maximum segment length (default: 0)
- `max_context`: Maximum context size (default: -1)
- `prompt`: Initial prompt for decoder
- `comma_in_time`: Use comma in timestamps (default: true)
- `print_progress`: Print progress info (default: false)
- `progress_callback`: Progress callback function
- VAD parameters (see above section)
