# NPM package wrapping the whisper.cpp Node.js addon

This package is useful in a node.js environment (node, Electron, etc.) where it will provide access to the C++ implementation of whisper.cpp which should be the fastest possible way to run it.

## Install
To install you need to have [cmake v4+](https://cmake.org/download/) (and [Visual Studio](https://github.com/nodejs/node-gyp#on-windows) on Windows) already available on you machine - it will be used to build the addon for your environment.

```shell
npm install whisper.cpp.node
npx whisper.cpp.node install
```

This will download the whisper.cpp repository and will build the addon. There are few commands that can be used:

- "install" - downloads the latest released tag of the whisper.cpp repository at the time of publishing or does nothing if the repo is already downloaded;
- "install latest" - downloads the latest master;
- "reinstall" - downloads the latest released tag of the whisper.cpp repository at the time of publishing even if the repo is already downloaded;
- "reinstall latest" - will produce the same result as "install latest";
- "rebuild" - will not download anything but will simply try to rebuild the addon (if for example you change the version of node).

The addon will then be available as the package's main export for use like:

```javascript
const whisper = require("whisper.cpp.node");

const transcription = await whisper({
  language: 'en',
  model: './models/ggml-base.en.bin',
  fname_inp: './your/file/here'
});

console.log(transcription);
```

Check the [Supported Parameters section](#supported-parameters) for more parameter information.

## What the package will not provide for you

It will not download any models for inference. As noted in many other packages, there are models ready for download and use in the [Hugging Face repo of whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main).

It will also not work if you try to bundle it for the browser (you should use the [whisper.cpp](https://www.npmjs.com/package/whisper.cpp) package instead which provides the WASM version).

## Links

- [GitHub Repository](https://github.com/gkostov/whisper.cpp.node)
- [NPM Package](https://www.npmjs.com/package/whisper.cpp.node)


And following is the original README of the addon where you can see details for using it.
______

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
./models/download-vad-model.sh silero-v6.2.0
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
  vad_model: path.join(__dirname, "../../models/ggml-silero-v6.2.0.bin"),
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