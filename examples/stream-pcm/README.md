# whisper.cpp/examples/stream-pcm

This example performs real-time inference on raw PCM audio streamed via stdin, a pipe, or a file.
It is PCM-first (input is consumed once) and does not require SDL or a microphone device.

## Usage

Stream raw PCM (16 kHz, mono) into the tool (non-VAD):

```bash
./build/bin/whisper-stream-pcm -m ./models/ggml-base.en.bin --format s16 --sample-rate 16000 --step 1000 --length 10000 --keep 500
```

Enable VAD-based segmentation (optional, recommended for speech bursts):

```bash
./build/bin/whisper-stream-pcm -m ./models/ggml-base.en.bin --format s16 --sample-rate 16000 --vad --vad-probe-ms 200 --vad-silence-ms 800 --vad-pre-roll-ms 300 --length 8000
```

You can also read from a named pipe (FIFO):

```bash
mkfifo /tmp/whisper.pcm
./build/bin/whisper-stream-pcm -m ./models/ggml-base.en.bin --input /tmp/whisper.pcm --format s16 --sample-rate 16000 --step 1000 --length 10000 --keep 500
```

Example of piping a WAV file using ffmpeg (optional, `-re` for realtime pacing):

```bash
ffmpeg -re -i samples/jfk.wav -f s16le -ac 1 -ar 16000 - | \
  ./build/bin/whisper-stream-pcm -m ./models/ggml-base.en.bin --format s16 --sample-rate 16000 --step 1000 --length 10000 --keep 500
```

Windows (PowerShell + `cmd /c`) pipe example:

```powershell
cmd /c "ffmpeg -re -hide_banner -loglevel error -i samples\jfk.wav -f s16le -ac 1 -ar 16000 - | build-cpu\bin\Release\whisper-stream-pcm.exe -m models\ggml-base.en.bin --format s16 --sample-rate 16000 --step 1000 --length 10000 --keep 500"
```

## Notes

- Input must be raw PCM, mono, 16 kHz. The tool does not resample.
- Supported formats: `f32` or `s16` (little-endian).
- Use `--input -` (default) for stdin.
- `--step` must be > 0 unless `--vad` is enabled.
- For VAD, `--vad-probe-ms` should be at least 200 ms; very small probes can fail to trigger.

## Building

`whisper-stream-pcm` does not depend on SDL and builds with the default examples:

```bash
cmake -B build
cmake --build build --config Release
```
