# Stable Timestamps 005 - Build + Verification (Synthetic 5m Audio)

## 1) Build whisper-cli

Run from `whisper.cpp/`:

```bash
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target whisper-cli -j8
```

## 2) Download Models

### Whisper model (requested)

```bash
mkdir -p models
wget -O models/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

### VAD model (required for `--stable-timestamps`)

```bash
./models/download-vad-model.sh silero-v6.2.0
# expected: models/ggml-silero-v6.2.0.bin
```

## 3) Generate 5-Minute Synthetic Audio + Timeline

This generates:
- WAV audio with normal pauses and occasional 20s pauses
- exact timeline CSV (`speech`, `short_silence`, `long_silence`) for ground-truth silence checks
- reference transcript text assembled from generated speech rows

```bash
mkdir -p plans/stable-timestamps/out
uv run plans/stable-timestamps/stable-timestamps_005.py \
  --output plans/stable-timestamps/out/synth_5min.wav \
  --timeline-csv plans/stable-timestamps/out/synth_5min_timeline.csv \
  --reference-txt plans/stable-timestamps/out/synth_5min_reference.txt \
  --target-seconds 300 \
  --long-pause-seconds 20
```

## 4) Transcribe Baseline (No Stable Timestamps)

```bash
./build/bin/whisper-cli \
  -m models/ggml-large-v3-turbo.bin \
  -f plans/stable-timestamps/out/synth_5min.wav \
  -l en \
  -ojf \
  -of plans/stable-timestamps/out/baseline
```

Output JSON:
- `plans/stable-timestamps/out/baseline.json`

## 5) Transcribe With Stable Timestamps

```bash
./build/bin/whisper-cli \
  -m models/ggml-large-v3-turbo.bin \
  -f plans/stable-timestamps/out/synth_5min.wav \
  -l en \
  --stable-timestamps \
  --vad-model models/ggml-silero-v6.2.0.bin \
  -ojf \
  -of plans/stable-timestamps/out/stable
```

Output JSON:
- `plans/stable-timestamps/out/stable.json`

## 6) Compare Baseline vs Stable (Pandas)

This uses the synthetic timeline CSV as exact silence ground truth.
It also compares transcript completeness against the generated reference text using WER/CER.

```bash
uv run plans/stable-timestamps/stable-timestamps_005_compare.py \
  --timeline-csv plans/stable-timestamps/out/synth_5min_timeline.csv \
  --reference-txt plans/stable-timestamps/out/synth_5min_reference.txt \
  --baseline-json plans/stable-timestamps/out/baseline.json \
  --stable-json plans/stable-timestamps/out/stable.json \
  --out-csv plans/stable-timestamps/out/metrics_005.csv
```

Primary metrics to inspect:
- `n_start_in_silence` / `pct_start_in_silence`
- `n_end_in_silence`
- `n_overlap_silence`
- `n_start_in_long_silence`
- `start_to_speech_edge_ms_mean`, `end_to_speech_edge_ms_mean`
- `wer`, `cer`
- `word_recall`, `word_precision`
- `hyp_to_ref_word_ratio` and `hyp_words` vs `ref_words`

Expected direction:
- stable mode should reduce silence-placement metrics and boundary error while keeping text metrics close to baseline.
