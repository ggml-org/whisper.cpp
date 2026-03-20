# whisper.cpp/examples/mic

This example captures live microphone audio and performs manual start/stop transcription.
Unlike `whisper-stream`, it records a full segment first and then transcribes that segment.

## Run

```bash
./build/bin/whisper-mic -m ./models/ggml-base.bin
```

Press Enter to start recording, then press Enter again to stop and transcribe.

## Options

```text
  -h,   --help           show help and exit
  -m F, --model F        model path
  -t N, --timeout N      max recording time in seconds
  -c N, --capture N      capture device ID
  -l S, --language S     language (for example: zh, en)
  -ng,  --no-gpu         disable GPU inference
```

## Build

```bash
cmake -B build
cmake --build build --config Release -j
```

## GPU build (optional)

```bash
cmake -S . -B build_gpu -DGGML_CUDA=ON
cmake --build build_gpu --config Release -j
```

Then run:

```bash
./build_gpu/bin/whisper-mic -m ./models/ggml-base.bin
```
