# whisper.cpp/examples/cli

This is the main example demonstrating most of the functionality of the Whisper model.
It can be used as a reference for using the `whisper.cpp` library in other projects.

```
./build/bin/whisper-cli -h

usage: ./build/bin/whisper-cli [options] file0 file1 ...
supported audio formats: flac, mp3, ogg, wav

options:
  -h,        --help              [default] show this help message and exit
             --version           [       ] show version information and exit
  -t N,      --threads N         [4      ] number of threads to use during computation
  -p N,      --processors N      [1      ] number of processors to use during computation
  -ot N,     --offset-t N        [0      ] time offset in milliseconds
  -on N,     --offset-n N        [0      ] segment index offset
  -d  N,     --duration N        [0      ] duration of audio to process in milliseconds
  -mc N,     --max-context N     [-1     ] maximum number of text context tokens to store
  -ml N,     --max-len N         [0      ] maximum segment length in characters
  -sow,      --split-on-word     [false  ] split on word rather than on token
  -bo N,     --best-of N         [5      ] number of best candidates to keep
  -bs N,     --beam-size N       [5      ] beam size for beam search
  -ac N,     --audio-ctx N       [0      ] audio context size (0 - all)
  -wt N,     --word-thold N      [0.01   ] word timestamp probability threshold
  -et N,     --entropy-thold N   [2.40   ] entropy threshold for decoder fail
  -lpt N,    --logprob-thold N   [-1.00  ] log probability threshold for decoder fail
  -nth N,    --no-speech-thold N [0.60   ] no speech threshold
  -tp,       --temperature N     [0.00   ] The sampling temperature, between 0 and 1
  -tpi,      --temperature-inc N [0.20   ] The increment of temperature, between 0 and 1
  -debug,    --debug-mode        [false  ] enable debug mode (eg. dump log_mel)
  -tr,       --translate         [false  ] translate from source language to english
  -di,       --diarize           [false  ] stereo audio diarization
  -tdrz,     --tinydiarize       [false  ] enable tinydiarize (requires a tdrz model)
  -nf,       --no-fallback       [false  ] do not use temperature fallback while decoding
  -otxt,     --output-txt        [false  ] output result in a text file
  -ovtt,     --output-vtt        [false  ] output result in a vtt file
  -osrt,     --output-srt        [false  ] output result in a srt file
  -olrc,     --output-lrc        [false  ] output result in a lrc file
  -owts,     --output-words      [false  ] output script for generating karaoke video
  -fp,       --font-path         [/System/Library/Fonts/Supplemental/Courier New Bold.ttf] path to a monospace font for karaoke video
  -ocsv,     --output-csv        [false  ] output result in a CSV file
  -oj,       --output-json       [false  ] output result in a JSON file
  -ojf,      --output-json-full  [false  ] include more information in the JSON file
  -of FNAME, --output-file FNAME [       ] output file path (without file extension)
  -np,       --no-prints         [false  ] do not print anything other than the results
  -ps,       --print-special     [false  ] print special tokens
  -pc,       --print-colors      [false  ] print colors
             --print-confidence  [false  ] print confidence
  -pp,       --print-progress    [false  ] print progress
  -nt,       --no-timestamps     [false  ] do not print timestamps
  -l LANG,   --language LANG     [en     ] spoken language ('auto' for auto-detect)
  -dl,       --detect-language   [false  ] exit after automatically detecting language
             --prompt PROMPT     [       ] initial prompt (max n_text_ctx/2 tokens)
             --carry-initial-prompt [false] always prepend initial prompt
  -m FNAME,  --model FNAME       [models/ggml-base.en.bin] model path
  -f FNAME,  --file FNAME        [       ] input audio file path
  -oved D,   --ov-e-device DNAME [CPU    ] the OpenVINO device used for encode inference
  -dtw MODEL --dtw MODEL         [       ] compute token-level timestamps
  -ls,       --log-score         [false  ] log best decoder scores of tokens
  -ng,       --no-gpu            [false  ] disable GPU
  -dev N,    --device N          [0      ] GPU device ID (default: 0)
  -fa,       --flash-attn        [true   ] enable flash attention
  -nfa,      --no-flash-attn     [false  ] disable flash attention
  -sns,      --suppress-nst      [false  ] suppress non-speech tokens
  --suppress-regex REGEX         [       ] regular expression matching tokens to suppress
  --grammar GRAMMAR              [       ] GBNF grammar to guide decoding
  --grammar-rule RULE            [       ] top-level GBNF grammar rule name
  --grammar-penalty N            [100.0  ] scales down logits of nongrammar tokens

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

## Non-speech audio (music/silence) hallucinations

On audio that is mostly music or silence, the decoder can emit phantom text:
repeated lines, or fluent sentences in an unexpected language (commonly CJK),
even when no actual speech is present (see
[#1724](https://github.com/ggml-org/whisper.cpp/issues/1724)). The options below
reduce this behaviour:

- **Pin the language** with `-l LANG, --language LANG` instead of relying on
  auto-detection: on non-speech audio the language detector can pick a language
  at random, and decoding then produces text matching that language.
- **`-sns, --suppress-nst`** suppresses non-speech tokens.
- **`-nf, --no-fallback`** disables the temperature fallback: when the decoder
  is unsure (as it is on silence), the fallback retries at higher temperatures,
  which tends to turn uncertainty into fluent hallucinations.
- **Tighten the decoder-fail thresholds** `-et, --entropy-thold` and
  `-lpt, --logprob-thold` so low-confidence output is rejected rather than
  printed.
- **Use VAD** (`--vad` with `--vad-model`) to cut non-speech regions before
  they reach the decoder — nothing is hallucinated from audio that is never
  decoded.
- As a last resort, **`--suppress-regex REGEX`** filters tokens matching a
  known phantom pattern.
