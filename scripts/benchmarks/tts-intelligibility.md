# TTS intelligibility benchmark

The `chatterbox`, `chatterbox-multilingual`, `supertonic`, `supertonic1`,
`supertonic2`, `parler`, `parler-indic`, `cosyvoice`, `audio8-tts`, and
`moss-tts` families synthesize the same checked-in English prompt. The final
measured output is transcribed by Whisper Tiny and compared with the prompt
using the existing English WER normalizer. `audio8-tts` is the full synthesis
workload; the separate Audio8 codec reconstruction benchmark in PR #244
remains separate. `moss-tts` is macOS-only and opt-in because its registered
f16 backbone is about 17 GB.

WER is `(substitutions + insertions + deletions) / reference words`.
Lower is better; zero means the reference ASR recovered all normalized words.
Insertions can produce WER above 100%. This is an intelligibility proxy that
includes errors from the reference ASR. It does not measure speaker similarity,
naturalness, or overall sound quality. No quality pass threshold is configured.

## Workload and timing

`fixtures/tts-intelligibility.txt` supplies both synthesis text and reference.
Changing this prompt resets the performance baseline. Parler uses its existing
sampled mode with seed 42 because its default greedy bench mode is documented
as producing degenerate audio. The other families retain their synthesis
settings with fixed seeds; GPU offload is requested with CPU fallback.

Native benchmarks retain final measured PCM and write `--wav-out` after timing.
Time-wrapped CLIs write a unique WAV for each warmup and timed invocation;
their existing wall timing includes CLI startup and WAV output. In both cases,
reference-ASR preparation and transcription are outside synthesis timing.
The score represents one final measured output, not the median of three scores.

## Reference ASR

`correctness.asr_model` pins `ggerganov/whisper.cpp` at commit
`5359861c739e955e79d9a303bcbc70fb988958b1`, file `ggml-tiny.bin`, SHA-256
`be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21`.
The task calls this a GGUF; the repository's Whisper CLI uses this GGML file.
`prepare-tts-asr.py` verifies cached files and atomically publishes verified
downloads under `bench-models/reference-asr/<sha256>/`.

The built-in Whisper audio decoder converts the generated WAV to mono 16 kHz.
The reference invocation is CPU-only on both runners, English, four threads,
temperature zero, best-of/beam size one, and no temperature fallback. The
expected text is never supplied as an ASR prompt. Transcription has a 300-second
timeout. No new Python package or cloud ASR service is required.

## Run and inspect

With the umbrella build configured, build the family target and `whisper-cli`:

```sh
cmake --build build --target supertonic-bench whisper-cli -j4
mkdir -p artifacts
scripts/benchmarks/run-family.sh --family supertonic --runs 3 --warmup 1 \
  --build-dir build --models-root "$PWD/bench-models" \
  --runner linux --out artifacts/result.json
```

The driver uses the existing S3 model download configuration for TTS weights.
On GitHub Actions, dispatch `speech-benchmark-desktop.yml` with
`model_families=chatterbox,chatterbox-multilingual,supertonic,supertonic1,supertonic2,parler,parler-indic,cosyvoice,audio8-tts`,
`runners=linux,macos`, and `runs_per_bench=3`; add `moss-tts` to the list to
include the opt-in MOSS cell on macOS.

Artifacts alongside `result.json` are `.tts.wav`, `.tts-reference.txt`,
`.tts-transcript.txt`, `.tts-asr.log`, and `.tts-intelligibility.json`.
The score JSON includes ASR model hash/settings, WAV metadata, word counts,
edit counts, and execution status. Successful empty transcription scores all
reference words as deletions. Missing dependencies, invalid audio, invocation
failure, or timeout produce explicit unavailable/error results with null WER.
Synthesis performance remains reported when ASR scoring fails. A green workflow
or synthesis `status: ok` alone does not establish scoring success or quality.

## Model-free checks

```sh
scripts/benchmarks/test-run-family.sh
python3 -B scripts/benchmarks/test_tts_intelligibility.py
python3 -B scripts/benchmarks/test-tts-intelligibility-driver.py
```

Native CTest `test-bench-wav` verifies PCM16 encoding and output failures.
