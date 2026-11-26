# Benchmark Harness

This harness is the single source of truth for baseline and optimization performance measurements.

## Layout

```text
benchmark/
  bench.sh
  audio/
    short.wav
    medium.wav
    long.wav
    lock.json
  references/
    short.txt
    medium.txt
    long.txt
  parse_results.py
  results/
```

## Requirements

- `build/bin/whisper-cli` exists (build first with CMake).
- Audio files exist at:
  - `benchmark/audio/short.wav` (~30s)
  - `benchmark/audio/medium.wav` (~5m)
  - `benchmark/audio/long.wav` (~30m)
- Audio files must be 16 kHz, mono, 16-bit WAV.

## Fixed Benchmark Policy

- Warm-up runs: 1
- Measured runs: 5
- Sequential execution only
- Fixed model: `models/ggml-small.en.bin`
- Fixed decode flags:
  - `-l en -tp 0 -tpi 0 -nf -bs 1 -bo 1 -fa`
- Fixed threading config:
  - `-t 8 -p 1`

## Usage

1. Create the lock file (checksums + durations):

```bash
./benchmark/bench.sh --create-lock
```

2. Run benchmark (after lock exists):

```bash
./benchmark/bench.sh --variant metal-baseline
```

## Outputs

Each run writes to `benchmark/results/<timestamp>_<variant>/`:

- `config.json`: exact benchmark config + environment metadata
- `validated_inputs.json`: lock validation snapshot
- `raw/...`: per-run logs and metadata (`warmup_*.log`, `run_*.log`, `*.meta.json`)
- `runs.csv`: per-run metrics
- `summary.csv`: aggregated metrics
- `summary.json`: detailed aggregates
- `summary.md`: required table format
- `correctness.json`: WER/CER gate report against references

## Correctness Gate

- References are read from `benchmark/references/{short,medium,long}.txt`.
- The parser extracts transcript text from each measured run log and computes:
  - WER (word error rate)
  - CER (character error rate)
- Default enforcement thresholds from `bench.sh`:
  - `MAX_WER=0.02`
  - `MAX_CER=0.02`
- If enforcement is enabled and references are missing or thresholds are exceeded, the run exits non-zero.
