# whisper.cpp/examples/vad-speech-segments

This examples demonstrates how to use a VAD (Voice Activity Detection) model to
segment an audio file into speech segments.

### Building the example
The example can be built using the following command:
```console
cmake -S . -B build
cmake --build build -j8 --target vad-speech-segments
```

### Running the example
The examples can be run using the following command, which uses a model
that we use internally for testing:
```console
./build/bin/vad-speech-segments \
    -vad-model models/for-tests-silero-v5.1.2-ggml.bin \
    --file samples/jfk.wav \
    --no-prints

Detected 5 speech segments:
Speech segment 0: start = 0.29, end = 2.21
Speech segment 1: start = 3.30, end = 3.77
Speech segment 2: start = 4.00, end = 4.35
Speech segment 3: start = 5.38, end = 7.65
Speech segment 4: start = 8.16, end = 10.59
```
To see more output from whisper.cpp remove the `--no-prints` argument.


### Command line options
```console
./build/bin/vad-speech-segments --help

usage: ./build/bin/vad-speech-segments [options] file
supported audio formats: flac, mp3, ogg, wav

options:
  -h,        --help                          [default] show this help message and exit
  -f FNAME,  --file FNAME                    [       ] input audio file path
  -t N,      --threads N                     [4      ] number of threads to use during computation
  -ug,       --use-gpu                       [true   ] use GPU
  -vm FNAME, --vad-model FNAME               [       ] VAD model path
  -vt N,     --vad-threshold N               [0.50   ] VAD threshold for speech recognition
  -vspd N,   --vad-min-speech-duration-ms  N [250    ] VAD min speech duration (0.0-1.0)
  -vsd N,    --vad-min-silence-duration-ms N [100    ] VAD min silence duration (to split segments)
  -vmsd N,   --vad-max-speech-duration-s   N [FLT_MAX] VAD max speech duration (auto-split longer)
  -vp N,     --vad-speech-pad-ms           N [30     ] VAD speech padding (extend segments)
  -vo N,     --vad-samples-overlap         N [0.10   ] VAD samples overlap (seconds between segments)
  -np,       --no-prints                     [false  ] do not print anything other than the results
```
