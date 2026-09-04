# whisper.cpp/examples/stream

This is a naive example of performing real-time inference on audio from your microphone.
The `whisper-stream` tool samples the audio every half a second and runs the transcription continuously.
More info is available in [issue #10](https://github.com/ggerganov/whisper.cpp/issues/10).

```bash
./build/bin/whisper-stream -m ./models/ggml-base.en.bin -t 8 --step 500 --length 5000
```

https://user-images.githubusercontent.com/1991296/194935793-76afede7-cfa8-48d8-a80f-28ba83be7d09.mp4

## Sliding window mode with VAD

Setting the `--step` argument to `0` enables the sliding window mode:

```bash
 ./build/bin/whisper-stream -m ./models/ggml-base.en.bin -t 6 --step 0 --length 30000 -vth 0.6
```

In this mode, the tool will transcribe only after some speech activity is detected. A very
basic VAD detector is used, but in theory a more sophisticated approach can be added. The
`-vth` argument determines the VAD threshold - higher values will make it detect silence more often.
It's best to tune it to the specific use case, but a value around `0.6` should be OK in general.
When silence is detected, it will transcribe the last `--length` milliseconds of audio and output
a transcription block that is suitable for parsing.

## Output format

The shape depends on --step, and there is no flag to override it.

Default (--step > 0): one rolling segment, no timestamps, rewritten in
place with ANSI erase-line escapes. Fine for a terminal, wrong for a pipe.

VAD mode (--step 0): timestamped blocks meant for parsing —

    ### Transcription 0 START | t0 = 0 ms | t1 = 4000 ms

    [00:00:00.000 --> 00:00:03.480]   Hello there.

    ### Transcription 0 END

- timestamps are enabled implicitly by --step 0; there is no -nt here
- strip the [t0 --> t1] prefix; don't discard those lines as log noise
- -tdrz appends [SPEAKER_TURN] on a speaker change
- the ### markers and [Start speaking] are on stdout

## Model path

-m is relative to the process CWD and defaults to models/ggml-base.en.bin.
Spawning whisper-stream from a parent with a different CWD needs an
absolute path, or you get:

    error: failed to initialize whisper context

which is also what a corrupt model prints — check the path first.

## Building

The `whisper-stream` tool depends on SDL2 library to capture audio from the microphone. You can build it like this:

```bash
# Install SDL2
# On Debian based linux distributions:
sudo apt-get install libsdl2-dev

# On Fedora Linux:
sudo dnf install SDL2 SDL2-devel

# Install SDL2 on Mac OS
brew install sdl2

cmake -B build -DWHISPER_SDL2=ON
cmake --build build --config Release

./build/bin/whisper-stream
```

## Web version

This tool can also run in the browser: [examples/stream.wasm](/examples/stream.wasm)
