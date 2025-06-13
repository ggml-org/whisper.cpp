# whisper.cpp/examples/server

Simple http server. WAV Files are passed to the inference model via http requests.

https://github.com/ggerganov/whisper.cpp/assets/1991296/e983ee53-8741-4eb5-9048-afe5e4594b8f

## Usage

```
./build/bin/whisper-server -h

usage: ./build/bin/whisper-server [options]

options:
  -h,        --help              [default] show this help message and exit
  -t N,      --threads N         [4      ] number of threads to use during computation
  -p N,      --processors N      [1      ] number of processors to use during computation
  -ot N,     --offset-t N        [0      ] time offset in milliseconds
  -on N,     --offset-n N        [0      ] segment index offset
  -d  N,     --duration N        [0      ] duration of audio to process in milliseconds
  -mc N,     --max-context N     [-1     ] maximum number of text context tokens to store
  -ml N,     --max-len N         [0      ] maximum segment length in characters
  -sow,      --split-on-word     [false  ] split on word rather than on token
  -bo N,     --best-of N         [2      ] number of best candidates to keep
  -bs N,     --beam-size N       [-1     ] beam size for beam search
  -ac N,     --audio-ctx N       [0      ] audio context size (0 - all)
  -wt N,     --word-thold N      [0.01   ] word timestamp probability threshold
  -et N,     --entropy-thold N   [2.40   ] entropy threshold for decoder fail
  -lpt N,    --logprob-thold N   [-1.00  ] log probability threshold for decoder fail
  -debug,    --debug-mode        [false  ] enable debug mode (eg. dump log_mel)
  -tr,       --translate         [false  ] translate from source language to english
  -di,       --diarize           [false  ] stereo audio diarization
  -tdrz,     --tinydiarize       [false  ] enable tinydiarize (requires a tdrz model)
  -nf,       --no-fallback       [false  ] do not use temperature fallback while decoding
  -ps,       --print-special     [false  ] print special tokens
  -pc,       --print-colors      [false  ] print colors
  -pr,       --print-realtime    [false  ] print output in realtime
  -pp,       --print-progress    [false  ] print progress
  -nt,       --no-timestamps     [false  ] do not print timestamps
  -l LANG,   --language LANG     [en     ] spoken language ('auto' for auto-detect)
  -dl,       --detect-language   [false  ] exit after automatically detecting language
             --prompt PROMPT     [       ] initial prompt
  -m FNAME,  --model FNAME       [models/ggml-base.en.bin] model path
  -oved D,   --ov-e-device DNAME [CPU    ] the OpenVINO device used for encode inference
  -dtw MODEL --dtw MODEL         [       ] compute token-level timestamps
  --host HOST,                   [127.0.0.1] Hostname/ip-adress for the server
  --port PORT,                   [8080   ] Port number for the server
  --public PATH,                 [examples/server/public] Path to the public folder
  --request-path PATH,           [       ] Request path for all requests
  --inference-path PATH,         [/inference] Inference path for all requests
  --convert,                     [false  ] Convert audio to WAV, requires ffmpeg on the server
  -sns,      --suppress-nst      [false  ] suppress non-speech tokens
  -nth N,    --no-speech-thold N [0.60   ] no speech threshold
  -nc,       --no-context        [false  ] do not use previous audio context
  -ng,       --no-gpu            [false  ] do not use gpu
  -fa,       --flash-attn        [false  ] flash attention

Voice Activity Detection (VAD) options:
             --vad                           [false  ] enable Voice Activity Detection (VAD)
  -vm FNAME, --vad-model FNAME               [       ] VAD model path
  -vt N,     --vad-threshold N               [0.50   ] VAD threshold for speech recognition
  -vspd N,   --vad-min-speech-duration-ms  N [250    ] VAD min speech duration (0.0-1.0)
  -vsd N,    --vad-min-silence-duration-ms N [100    ] VAD min silence duration (to split segments)
  -vmsd N,   --vad-max-speech-duration-s   N [FLT_MAX] VAD max speech duration (auto-split longer)
  -vp N,     --vad-speech-pad-ms           N [30     ] VAD speech padding (extend segments)
  -vo N,     --vad-samples-overlap         N [0.10   ] VAD samples overlap (seconds between segments)
```

> [!WARNING]
> **Do not run the server example with administrative privileges and ensure it's operated in a sandbox environment, especially since it involves risky operations like accepting user file uploads and using ffmpeg for format conversions. Always validate and sanitize inputs to guard against potential security threats.**

## request examples

**/inference**
```
curl 127.0.0.1:8080/inference \
-H "Content-Type: multipart/form-data" \
-F file="@<file-path>" \
-F temperature="0.0" \
-F temperature_inc="0.2" \
-F response_format="json"
```

**/load**
```
curl 127.0.0.1:8080/load \
-H "Content-Type: multipart/form-data" \
-F model="<path-to-model-file>"
```

## Load testing with k6

> **Note:** Install [k6](https://k6.io/docs/get-started/installation/) before running the benchmark script.

You can benchmark the Whisper server using the provided bench.js script with [k6](https://k6.io/). This script sends concurrent multipart requests to the /inference endpoint and is fully configurable via environment variables.

**Example usage:**

```
k6 run bench.js \
  --env FILE_PATH=/absolute/path/to/samples/jfk.wav \
  --env BASE_URL=http://127.0.0.1:8080 \
  --env ENDPOINT=/inference \
  --env CONCURRENCY=4 \
  --env TEMPERATURE=0.0 \
  --env TEMPERATURE_INC=0.2 \
  --env RESPONSE_FORMAT=json
```

**Environment variables:**
- `FILE_PATH`: Path to the audio file to send (must be absolute or relative to the k6 working directory)
- `BASE_URL`: Server base URL (default: `http://127.0.0.1:8080`)
- `ENDPOINT`: API endpoint (default: `/inference`)
- `CONCURRENCY`: Number of concurrent requests (default: 4)
- `TEMPERATURE`: Decoding temperature (default: 0.0)
- `TEMPERATURE_INC`: Temperature increment (default: 0.2)
- `RESPONSE_FORMAT`: Response format (default: `json`)

**Note:**
- The server must be running and accessible at the specified `BASE_URL` and `ENDPOINT`.
- The script is located in the same directory as this README: `bench.js`.
