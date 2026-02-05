# whisper.cpp/examples/stream-pcm

This example performs real-time inference on raw PCM audio streamed via stdin or a pipe.
It mirrors the behavior of `whisper-stream`, but does not require SDL or a microphone device.

## Usage

Stream raw PCM (16 kHz, mono) into the tool:

```bash
./build/bin/whisper-stream-pcm -m ./models/ggml-base.en.bin --format s16 --sample-rate 16000 --step 500 --length 5000
```

You can also read from a named pipe (FIFO):

```bash
mkfifo /tmp/whisper.pcm
./build/bin/whisper-stream-pcm -m ./models/ggml-base.en.bin --input /tmp/whisper.pcm --format s16 --sample-rate 16000 --step 500 --length 5000
```

Example of piping a WAV file using ffmpeg (optional):

```bash
ffmpeg -i samples/jfk.wav -f s16le -ac 1 -ar 16000 - | \
  ./build/bin/whisper-stream-pcm -m ./models/ggml-base.en.bin --format s16 --sample-rate 16000 --step 500 --length 5000
```

## Notes

- Input must be raw PCM, mono, 16 kHz. The tool does not resample.
- Supported formats: `f32` or `s16` (little-endian).
- Use `--input -` (default) for stdin.

## Building

`whisper-stream-pcm` does not depend on SDL and builds with the default examples:

```bash
cmake -B build
cmake --build build --config Release
```
