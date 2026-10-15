# tts engine: command-line usage

Part of the [tts engine documentation](../README.md).

## Command-line tools

`TTS_CPP_BUILD_EXECUTABLES` creates:

| Binary | What it does |
|--------|--------------|
| `build/tts-cli` | Chatterbox batch/streaming, Supertonic batch, and limited Parler batch dispatch |
| `build/mel2wav` | Chatterbox HiFT-only mel-to-WAV demo |
| `build/cosyvoice-hift` / `cosyvoice-flow` / `cosyvoice-llm` | CosyVoice3 stage tools |
| `build/cosyvoice-cli` / `cosyvoice-bench` | CosyVoice3 end-to-end synthesis and benchmark |
| `build/supertonic-cli` | full Supertonic batch and streaming CLI |
| `build/parler-cli` / `parler-bench` | Parler end-to-end synthesis and benchmark |
| `build/lavasr-bench` | LavaSR denoiser/enhancer benchmark and by-ear harness |
| `build/audio8-cli` | Audio8 text-to-speech and zero-shot voice cloning |
| `build/pocket-cli` | Pocket text-to-speech and metadata-only memory preflight |
| `build/moss-cli` | MOSS Delay text-to-speech and zero-shot voice cloning; `--mode sfx` for MOSS-SoundEffect text-to-sound-effects; `--mode s2s` for MOSS-Speech speech-to-speech |

For parity harnesses and their build/fixture gates, see
[testing.md](testing.md#validation-harness-index). ggml acquisition and build
paths are documented in [build.md](build.md#build-paths).

Commands below run from `engines/tts` after a direct engine build. For umbrella
builds substitute `build/engines/tts/` from the repository root; multi-config
generators also add the selected configuration directory.

## One-time: convert weights

```bash
# Activate the Python environment from the Conversion tooling step
. chatterbox-ref/.venv/bin/activate

# --- Turbo (English, default) ---
python scripts/convert-t3-turbo-to-gguf.py --out models/chatterbox-t3-turbo.gguf
python scripts/convert-s3gen-to-gguf.py    --out models/chatterbox-s3gen.gguf

# --- Multilingual (23 languages) ---
python scripts/convert-t3-mtl-to-gguf.py            --out models/chatterbox-t3-mtl.gguf
python scripts/convert-s3gen-to-gguf.py --variant mtl --out models/chatterbox-s3gen-mtl.gguf

# --- Multilingual, quantised (recommended for speed) ---
# Used by the dated CI benchmark in docs/performance.md. --quant accepts
# {f32,f16,q8_0,q5_0,q4_0} on convert-s3gen-to-gguf.py (default f16) and
# {f16,q8_0,q5_0,q4_0} on convert-t3-mtl-to-gguf.py (default f16, since
# the T3 storage baseline is already F16).  The flag controls the large
# matmul weights only — biases, LayerNorm gammas/betas, embedding tables,
# voice encoders, and built-in voice conditioning always stay at full
# precision (see the deny-list in scripts/requantize-gguf.py for the
# exact policy; the same policy is used by all three tools).
python scripts/convert-t3-mtl-to-gguf.py --quant q4_0 \
       --out models/chatterbox-t3-mtl-q4_0.gguf
python scripts/convert-s3gen-to-gguf.py  --variant mtl --quant q4_0 \
       --out models/chatterbox-s3gen-mtl-q4_0.gguf
```

The Turbo converter pulls `ResembleAI/chatterbox-turbo` (~1.5 GB), the MTL
converter pulls `ResembleAI/chatterbox` (~3 GB).  The BPE tokenizer for
Turbo (`vocab.json` + `merges.txt` + `added_tokens.json`) is **embedded
directly into the T3 GGUF** as `tokenizer.ggml.*` metadata; for MTL we
embed the full HuggingFace `tokenizers.json` blob (plus a Korean-Jamo /
NFKD Unicode table for offline preprocessing), so in both cases you don't
need to keep the source tokenizer files around on disk.

The quantisation flag on `convert-s3gen-to-gguf.py` changes storage
without requiring a different runtime. CPU, Metal, Vulkan and CUDA use
their corresponding quantized matmul kernels.  The per-tensor decision lives in `should_quantize()`
inside `scripts/requantize-gguf.py` (single source of truth shared
with the offline rewriter): biases, norm scales, embedding tables,
spectral filterbanks, voice-cloning preprocessors (CAMPPlus,
VoiceEncoder, S3TokenizerV2) and any tensor whose reduction dim isn't
block-aligned all stay at full precision.  See the [archived Chatterbox port report](history/chatterbox-port.md) §3.20 for
the full deny-list and resulting size / speed numbers.

You should now have (either pair is usable on its own):

```
models/
  chatterbox-t3-turbo.gguf   (~742 MB) — T3 GPT-2 Medium + embedded GPT-2 BPE
                               tokenizer + VoiceEncoder weights + built-in voice
  chatterbox-s3gen.gguf      (~1.0 GB) — S3Gen encoder/CFM (meanflow 2-step)
                               + HiFT vocoder + CAMPPlus + S3TokenizerV2
                               + built-in voice (everything needed for voice
                               cloning Turbo-side)

  chatterbox-t3-mtl.gguf          (~1.1 GB) — T3 Llama-520M + perceiver resampler
                                    + emotion adv + learned pos embs + embedded
                                    MTL grapheme tokenizer JSON + VoiceEncoder
                                    + built-in voice
  chatterbox-s3gen-mtl.gguf       (~1.0 GB) — S3Gen encoder/CFM (standard 10-step
                                    + CFG inside, cfg_rate=0.7) + HiFT vocoder
                                    + CAMPPlus + S3TokenizerV2 + built-in voice

  # Optional quantised multilingual variants — numerically very close to F16 but
  # lower weight bandwidth. Recorded speedups and quality tradeoffs are in
  # docs/performance.md; validate the chosen tier for your workload.
  chatterbox-t3-mtl-q4_0.gguf     (~344 MB) — Q4_0 T3 Llama-520M
  chatterbox-s3gen-mtl-q4_0.gguf  (~685 MB) — Q4_0 MTL S3Gen
```

For numerical validation against PyTorch (optional, step 4), also run:

```bash
python scripts/dump-s3gen-reference.py \
  --text "Hello from ggml." --out artifacts/s3gen-ref \
  --seed 42 --n-predict 64 --device cpu
```

### Optional: quantize the models (smaller + faster)

Both GGUFs can be quantized to `Q8_0` (near-lossless), `Q5_0`, or
`Q4_0` (different CFM sample but same subjective quality, smaller).
The same machinery works on the multilingual GGUFs too — the
[dated CI benchmarks](performance.md#ci-benchmarks-2026-08-12-linux-x86-64)
use q4_0 variants.  `llama-quantize` doesn't recognize the `chatterbox` /
`chatterbox-s3gen` custom architectures, so we ship a small standalone
rewriter that works on any of the four GGUFs:

```bash
# T3
python scripts/requantize-gguf.py \
  models/chatterbox-t3-turbo.gguf \
  models/t3-q8_0.gguf q8_0

# S3Gen
python scripts/requantize-gguf.py \
  models/chatterbox-s3gen.gguf \
  models/chatterbox-s3gen-q8_0.gguf q8_0
```

Swap `q8_0` → `q4_0` (or `q5_0`) for a more aggressive variant.  T3's
original converter also accepts `--quant` if you prefer to quantize at
conversion time instead of after.

Measured on a representative paragraph (M3 Ultra, Metal, streaming mode
`--stream-chunk-tokens 25 --max-sentence-chars 100`):

| T3 / S3Gen                | total size | first-audio | total wall | cos sim¹ |
|---------------------------|-----------:|------------:|-----------:|--------:|
| F16 / F32 (baseline)      |  1 757 MB  |  1 604 ms   |   28.6 s   |  1.000  |
| Q8_0 / F32                |  1 476 MB  |  1 451 ms   |   27.2 s   |   —     |
| F16 / Q8_0                |  1 532 MB  |  1 646 ms   |   28.2 s   |  0.991  |
| **Q8_0 / Q8_0**           | **1 251 MB** | **1 399 ms** | **26.4 s** | 0.991 |
| **Q4_0 / Q4_0**           | **1 071 MB** | **1 510 ms** | **26.7 s** | 0.66²  |

¹ Cosine similarity of the final waveform vs the F16/F32 baseline.

² Q4_0 quantization shifts the CFM diffusion ODE's trajectory enough
  to land on a *different sample* from the same noise seed.  Subjective
  quality is essentially the same (in-distribution speech, correct
  phonemes, stable voice); it's just a different legitimate sample
  rather than a lower-fidelity version of the baseline.

Using both Q8_0 variants cuts **~500 MB off disk**, drops first-audio
latency **~13 %**, speeds total wall-clock **~8 %**, and produces
audibly-identical output (cos-sim > 0.99 vs F32 reference waveform).
Q4_0 trims another ~180 MB on top for roughly the same speed — best
choice on memory-constrained targets (mobile, low-end CPUs) when you
don't need per-seed reproducibility against the F32 baseline.

Note: the S3Gen requantize script only compresses the 385 big 2-D
matmul weights (encoder attention/MLPs + CFM projections + flow FFs).
The 1 664 other tensors — biases, norms, spectral filterbanks, the
input-embedding table, the 3-D convolution weights — remain at their
source dtype to keep numerics clean.  That's why Q4_0 ends up only
~15 % smaller than Q8_0 rather than 2× smaller; the bulk not covered
by block quantization dominates.

Pass the quantized GGUFs to `tts-cli` exactly like the defaults:

```bash
./build/tts-cli \
  --model      models/t3-q8_0.gguf \
  --s3gen-gguf models/chatterbox-s3gen-q8_0.gguf \
  --text "Hello from the quantized port." \
  --n-gpu-layers 99 --out out.wav
```

## Run — end-to-end text → wav

The easiest way:

```bash
./scripts/synthesize.sh "Hello from native C plus plus." /tmp/out.wav
```

That's equivalent to running the binary directly:

```bash
./build/tts-cli \
  --model       models/chatterbox-t3-turbo.gguf \
  --s3gen-gguf  models/chatterbox-s3gen.gguf \
  --text        "Hello from native C plus plus." \
  --out         /tmp/out.wav
```

Native language codes: `en`, `es`, `fr`, `de`, `it`, `pt`, `nl`, `pl`,
`tr`, `sv`, `da`, `fi`, `no`, `el`, `ms`, `sw`, `ar`, `ko`. Japanese,
Hebrew, Russian, Chinese and Hindi require external preprocessing; the
following sections describe those paths.

**Multilingual** takes the same flags plus a required `--language CODE`
(one of the native language codes listed below) and runs all the
CFG / perceiver / 10-step-CFM machinery automatically based on the GGUF's
`chatterbox.variant` metadata:

```bash
./build/tts-cli \
  --model       models/chatterbox-t3-mtl.gguf \
  --s3gen-gguf  models/chatterbox-s3gen-mtl.gguf \
  --text        "Hola, esto es una demostración multilingüe." \
  --language    es \
  --out         /tmp/mtl_es.wav
```

Extra MTL-only knobs: `--cfg-weight F` (default 0.5, must be ≥ 0),
`--min-p F` (0.05, in [0, 1]), `--exaggeration F` (0.5 — emotion
intensity, in [0, 1]).  `--reference-audio` works
the same way on both variants.

`--cfm-steps N` lowers the CFM Euler step count for non-streaming
synthesis (default 10 for Multilingual's standard CFM).  N=7 saves ~22%
of S3Gen wall time at log-mel cosine 0.995 vs the N=10 reference and is
the recommended quality knee on M3 Ultra (see the [Chatterbox port archive §3.21](history/chatterbox-port.md));
N=6 is too aggressive (cosine 0.990 right at the threshold, PCM cosine
drops to 0.88).  Streaming chunks ignore this flag and use
`--stream-cfm-steps` instead.

`--output-sample-rate HZ` selects the output frequency. The
pipeline natively emits 24 kHz (Chatterbox) / the model's metadata rate
(Supertonic); pass a positive rate in `8000..192000` to resample the final
PCM with the in-tree Kaiser-windowed sinc resampler before it's written or
streamed (e.g. `--output-sample-rate 16000` for a 16 kHz wav).  `0` (the
default) keeps the native rate — zero behaviour change.  Works on the
single-shot, auto-split, and streaming paths and on both engines (the
`tts_cpp::*::EngineOptions::output_sample_rate` field exposes the same knob
to library callers; `SynthesisResult::sample_rate` always reports the actual
rate).

Everything is self-contained in the two `.gguf` files:

- `chatterbox-t3-turbo.gguf` embeds the BPE tokenizer (vocab + merges +
  added tokens) as standard `tokenizer.ggml.*` metadata, which the C++
  binary loads out of GGUF at startup. The tokenizer itself (merge chains,
  byte-level encoding, added-token spans, `punc_norm`) is pinned model-free
  by `test-gpt2-bpe` on a hand-built vocabulary.
- `chatterbox-s3gen.gguf` embeds the built-in reference voice (embedding,
  prompt token, prompt mel) under `s3gen/builtin/*`.

Advanced modes:

- **T3 only** — drop `--s3gen-gguf` + `--out`; write tokens with
  `--output tokens.txt`. Useful for piping into other tools.
- **S3Gen + HiFT only** — pass `--s3gen-gguf` + `--tokens-file FILE` with
  already-generated speech tokens and no `--model`.
- **Custom voice (voice cloning)** — point `--reference-audio` at a
  reference `.wav` and the C++ binary does everything else natively
  (no Python, no preprocessing step):

  ```bash
  ./build/tts-cli --model models/chatterbox-t3-turbo.gguf \
                     --s3gen-gguf models/chatterbox-s3gen.gguf \
                     --reference-audio me.wav \
                     --text "Hello in my voice." \
                     --out out.wav
  ```

  Requirements for the reference wav:
  - **Strictly more than 5 s** of clean mono speech (the binary enforces
    this and fails fast; 10–15 s gives the best similarity).
  - Any sample rate, any PCM bit-depth (binary resamples + downmixes).

  **Prep helper** — `scripts/extract-voice.py` automates the usual
  chore of picking a good clip out of a messy recording (podcast,
  WhatsApp voice note, `.mov` screen capture, etc.):

  ```bash
  # auto-detect codec, pick the best 10 s speech block, write voices/alice.wav:
  ./scripts/extract-voice.py ~/Downloads/alice.m4a --name alice
  # same, but also bake the .npy profile in one go:
  ./scripts/extract-voice.py ~/Downloads/alice.m4a --name alice --bake
  ```

  It probes the file, runs `silencedetect` to find speech regions,
  picks the longest clean 5–15 s block from the middle of the
  recording (or concatenates the two best short blocks if no single
  long block exists), then applies a codec-aware filter chain:

  | source codec                         | chain applied                                                                 |
  |--------------------------------------|-------------------------------------------------------------------------------|
  | WAV / FLAC / ≥ 96 kbps AAC / ≥ 128 kbps MP3 | `highpass + alimiter` — minimal, trusts the source                      |
  | Opus / Vorbis at any bitrate, low-bitrate AAC/MP3 | `highpass + afftdn + 3-band EQ + loudnorm + alimiter` — restores presence/air past the codec's brick-wall low-pass |

  The lossy chain is what takes an 18 kbps Opus voice note from
  "clone sounds wrong" to "clone sounds like the speaker".  See
  `./scripts/extract-voice.py --help` for the full flag set.

  Loudness is normalised to **-27 LUFS** (ITU-R BS.1770-4 / EBU R 128)
  internally before preprocessing, so a quiet recording like a phone
  memo works as well as a studio track.  All five voice-conditioning
  tensors are produced in C++:

  | tensor                         | source                           |
  |--------------------------------|----------------------------------|
  | `speaker_emb`                  | C++ VoiceEncoder  (T3 GGUF)      |
  | `cond_prompt_speech_tokens`    | C++ S3TokenizerV2 (S3Gen GGUF)   |
  | `prompt_token`                 | C++ S3TokenizerV2 (S3Gen GGUF)   |
  | `embedding`                    | C++ CAMPPlus      (S3Gen GGUF)   |
  | `prompt_feat`                  | C++ mel extraction               |

- **Cache a voice for fast reuse (`--save-voice`)** — voice preprocessing
  (VoiceEncoder + CAMPPlus + S3TokenizerV2 + mel) adds ≈ 2 minutes on a
  Mac before every synthesis.  The five tensors don't depend on the
  text, so bake them once:

  ```bash
  # Bake the profile (no --text needed; just preprocesses + saves).
  ./build/tts-cli --model models/chatterbox-t3-turbo.gguf \
                     --s3gen-gguf models/chatterbox-s3gen.gguf \
                     --reference-audio me.wav \
                     --save-voice voices/me/
  # Writes voices/me/{speaker_emb, cond_prompt_speech_tokens,
  # embedding, prompt_token, prompt_feat}.npy (~160 KB total).

  # Reuse (≈ 17× faster; VoiceEncoder / CAMPPlus / S3TokenizerV2
  # / mel extraction are all skipped).
  ./build/tts-cli --model models/chatterbox-t3-turbo.gguf \
                     --s3gen-gguf models/chatterbox-s3gen.gguf \
                     --ref-dir voices/me/ \
                     --text "Anything you want." \
                     --out  out.wav
  ```

  You can mix the two: `--ref-dir D --reference-audio X.wav` will load
  any `.npy` present in `D` and compute the rest from `X.wav`.  Useful
  during development when you want to iterate on one tensor.

Play the result:

```bash
afplay /tmp/out.wav         # macOS
aplay  /tmp/out.wav         # Linux (alsa)
ffplay /tmp/out.wav         # any OS with ffmpeg
```

### Live / streaming input

When you want a long-running process that keeps the model loaded and
synthesises whatever text arrives as it arrives — e.g. the output of
a streaming LLM, a live transcription, or just a human typing —
use `--input-file`.  The binary `tail -f`'s the file, splits on
sentence terminators (or `\n` in `--input-by-line` mode), and pipes
raw PCM (s16le, 24 kHz, mono) to stdout chunk-by-chunk.

```bash
# Two-process demo: background writer appends sentences, tts-cli
# tail-follows, sox plays in real time.
./build/tts-cli \
    --model       models/t3-q8_0.gguf \
    --s3gen-gguf  models/chatterbox-s3gen-q8_0.gguf \
    --ref-dir     voices/alice \
    --input-file  ./speech.txt \
    --input-by-line \
    --stream-chunk-tokens 25 --stream-cfm-steps 2 \
    --n-gpu-layers 99 \
    --out -                     \
  | play -q -t raw -r 24000 -b 16 -e signed -c 1 -   # sox(1)

# Another process (LLM, transcriber, shell, etc.) writes here:
echo "First request." >> speech.txt
echo "Second request with, internal, punctuation." >> speech.txt
```

**Interactive mode on a TTY** — pass `--input-file -` to read from
stdin.  On a terminal you get a `> ` prompt; each Enter-terminated
line is spoken immediately, Ctrl-D exits:

```bash
./build/tts-cli \
    --model       models/t3-q8_0.gguf \
    --s3gen-gguf  models/chatterbox-s3gen-q8_0.gguf \
    --ref-dir     voices/alice \
    --input-file  - --input-by-line \
    --stream-chunk-tokens 25 --stream-cfm-steps 2 \
    --n-gpu-layers 99 \
    --out -                     \
  | play -q -t raw -r 24000 -b 16 -e signed -c 1 -
```

Relevant flags:

| flag                          | effect                                                                                                               |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `--input-file PATH`           | Tail-follow `PATH`; `-` means read stdin (interactive on a TTY).                                                     |
| `--input-by-line`             | One Enter-terminated line = one request.  `. ! ?` inside a line stay part of the same utterance (no mid-line restart). |
| `--input-eof-marker STR`      | Exit cleanly after seeing `STR` anywhere in the input (useful for scripted pipelines).                                |
| `--stream-chunk-tokens N`     | Speech-token chunk granularity for the S3Gen streaming loop.  25 is a good default.                                   |
| `--stream-cfm-steps N`        | CFM Euler steps per chunk.  2 is the minimum the model was designed for; 4–5 gives crisper word endings on cloned voices. |
| `--stream-first-chunk-tokens N` | Override the first chunk's size to minimise first-audio-out latency.                                                |

The process keeps the T3 + S3Gen models warm across requests, so
after the initial load (~150 ms), each request only pays T3 + S3Gen
inference cost (well under real-time on any GPU backend).

### Useful flags

- `--seed N` — change the RNG seed for the CFM initial noise and the SineGen
  excitation (same text, different voice "take").
- `--threads N` — override the CLI default, which caps automatic selection at
  4. Library Chatterbox, Parler, and Audio8 engines use the same cap; do not
  infer CosyVoice stage threading from this flag.
- `--n-gpu-layers N` — move layers to the GPU backend when built with
  `-DGGML_METAL=ON` / `-DGGML_CUDA=ON` / `-DGGML_VULKAN=ON`.  Pass `99`
  (or any large number) to move everything.
- `--backend NAME` (`supertonic-cli`, `supertonic-bench`,
  `supertonic-fit-params`, `parler-cli`, `parler-bench`,
  `parler-fit-params`, `cosyvoice-cli`, `cosyvoice-bench`,
  `cosyvoice-fit-params`, and `moss-cli` in its default TTS mode) — run on
  one device: `cpu`, `opencl`, `hexagon` (the Snapdragon HTP0 NPU) or an
  exact ggml device name; MOSS is validated on `cpu` and `hexagon` only. It overrides `--n-gpu-layers` (`--gpu` on
  `moss-cli`) and fails instead of falling back when the device is missing;
  the automatic walk never picks Hexagon.
- `--reference-audio PATH` — voice cloning input (see the Custom voice
  section above).
- `--save-voice DIR` — cache the five voice-conditioning tensors for
  reuse via `--ref-dir DIR`.
- `--ref-dir DIR` — load previously-baked voice tensors (or a subset)
  from `DIR/*.npy`.
- `--input-file PATH` — long-running mode; tail-follow `PATH` and
  synthesise text as it arrives.  Pass `-` to read from stdin (see the
  Live / streaming input section above).
- `--input-by-line` — treat one newline as one complete request; `. ! ?`
  inside a line stay part of the same utterance.
- `--debug` (requires `--ref-dir`) — substitute Python-dumped reference
  values for the random bits so every stage can be bit-exactly compared
  to PyTorch.

<a id="performance"></a>
