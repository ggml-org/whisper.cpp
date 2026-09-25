# Whisper Metal Optimization Handoff (2026-03-08)

## Scope and Guardrails
- Repository: `/Users/shaihi/Downloads/whisper_optimization`
- Target: Apple Silicon + Metal backend performance
- Benchmark policy: use only `benchmark/bench.sh`
- Current user constraint: run `short.wav` only unless explicitly asked to run `medium.wav` or `long.wav`
- Success criterion for an optimization: median runtime improvement >= 5% with stable repeated runs and acceptable correctness (WER/CER gates)

## Canonical Benchmark Harness
- Script: `/Users/shaihi/Downloads/whisper_optimization/benchmark/bench.sh`
- Parser/reporting: `/Users/shaihi/Downloads/whisper_optimization/benchmark/parse_results.py`
- Fixed config (from harness):
  - Warm-up runs: 1
  - Measured runs: 5
  - Model: `models/ggml-small.en.bin`
  - CLI decode flags: `-l en -tp 0 -tpi 0 -nf -bs 1 -bo 1 -fa`
  - Thread/process: `-t 8 -p 1`
- Short-only run command:
  - `./benchmark/bench.sh --variant <name> --audio short`

## Measurement Definitions
- Runtime metrics:
  - wall clock runtime
  - first inference latency
  - full transcription runtime
  - throughput in audio-seconds per second
  - tokens/s if available in logs
- Efficiency in this project:
  - primary: lower median runtime
  - secondary: higher throughput, lower variability (std dev)
- Correctness:
  - WER = Word Error Rate
  - CER = Character Error Rate
  - reference texts in `/Users/shaihi/Downloads/whisper_optimization/benchmark/references/`
  - default thresholds in harness: `MAX_WER=0.02`, `MAX_CER=0.02`

## Current Short Reference Size
- File: `/Users/shaihi/Downloads/whisper_optimization/benchmark/references/short.txt`
- Content: `"The Town Hose Story"`
- Count snapshot (`wc -w -m`):
  - words: 4
  - characters: 22

## Established Baseline and Latest Opt3 Comparison (short only)
### Baseline (current conditions)
- Variant: `metal-baseline-r2`
- Result dir: `/Users/shaihi/Downloads/whisper_optimization/benchmark/results/20260308_232244_metal-baseline-r2`
- Summary file: `/Users/shaihi/Downloads/whisper_optimization/benchmark/results/20260308_232244_metal-baseline-r2/summary.csv`
- Key metrics:
  - init_mean_ms: 509.688
  - first_inference_mean_s: 3.5896615666
  - runtime_median_s: 3.6510759580
  - throughput_mean_audio_s_per_s: 7.7969541540
  - runtime_std_dev_s: 0.3323240978
  - correctness_pass: True
  - notes: encode mean=2179.93 ms; decode mean=664.96 ms

### Optimization #3 (latest tuned run)
- Variant: `metal-opt3-v5`
- Result dir: `/Users/shaihi/Downloads/whisper_optimization/benchmark/results/20260308_232524_metal-opt3-v5`
- Summary file: `/Users/shaihi/Downloads/whisper_optimization/benchmark/results/20260308_232524_metal-opt3-v5/summary.csv`
- Key metrics:
  - init_mean_ms: 565.146
  - first_inference_mean_s: 3.3695700166
  - runtime_median_s: 3.6233550420
  - throughput_mean_audio_s_per_s: 8.2722771621
  - runtime_std_dev_s: 0.0125376431
  - correctness_pass: True
  - notes: encode mean=2041.77 ms; decode mean=536.73 ms

### Delta (opt3-v5 vs baseline-r2)
- runtime median: 3.6511s -> 3.6234s (about 0.76% faster)
- throughput mean: 7.797 -> 8.272 (about 6.10% higher)
- decode mean (from notes): 664.96 ms -> 536.73 ms (about 19.3% lower)
- success criterion status: NOT met on median runtime (needs >=5%)

## Optimization #3 Code Changes (currently uncommitted)
- File: `/Users/shaihi/Downloads/whisper_optimization/ggml/src/ggml-metal/ggml-metal-ops.cpp`
- Changes:
  - `const int ne11_mm_min = props_dev->supports_gpu_family_apple7 ? 6 : 8;`
  - `const int ne21_mm_id_min = props_dev->supports_gpu_family_apple7 ? 24 : 32;`
- Intent:
  - allow simdgroup matmul kernel path at slightly smaller matrix thresholds on Apple7+ GPUs to reduce decode-side overhead

## Working Tree Snapshot
- Modified: `/Users/shaihi/Downloads/whisper_optimization/ggml/src/ggml-metal/ggml-metal-ops.cpp`
- Untracked benchmark artifacts exist under `/Users/shaihi/Downloads/whisper_optimization/benchmark/`

## Immediate Task List for Fresh Thread
1. Choose next optimization branch to test next: #1 (decoder input staging/mask overhead) or #2 (Metal command buffer/scheduling overhead).
2. Keep `short`-only runs for dev iterations:
   - baseline refresh command: `./benchmark/bench.sh --variant metal-baseline-r3 --audio short`
   - candidate command: `./benchmark/bench.sh --variant metal-optX-v1 --audio short`
3. Compare candidate against the refreshed baseline using `summary.csv` only; report median runtime delta first.
4. Keep correctness gates enabled; reject changes with WER/CER threshold failures.
5. If short-only looks strong and user explicitly asks, run medium/long to verify scalability.

## Notes to Carry Forward
- User requested: test short only by default.
- User requested workflow: prove improvement for one optimization first, then ask before moving to others.

## 2026-03-09 Isolation Experiment (User-requested: short+medium+long)

### Isolated options tested (single-line edits only)
- Option A only:
  - `/Users/shaihi/Downloads/whisper_optimization/ggml/src/ggml-metal/ggml-metal-ops.cpp`
  - `ne11_mm_min: 8 -> props_dev->supports_gpu_family_apple7 ? 6 : 8`
- Option B only:
  - `/Users/shaihi/Downloads/whisper_optimization/ggml/src/ggml-metal/ggml-metal-ops.cpp`
  - `ne21_mm_id_min: 32 -> props_dev->supports_gpu_family_apple7 ? 24 : 32`

### Benchmark runs executed
- Baseline: `/Users/shaihi/Downloads/whisper_optimization/benchmark/results/20260308_233303_metal-baseline-r3`
- Option A: `/Users/shaihi/Downloads/whisper_optimization/benchmark/results/20260308_235028_metal-optA-r1`
- Option B: `/Users/shaihi/Downloads/whisper_optimization/benchmark/results/20260309_000042_metal-optB-r1`
- Consolidated table: `/Users/shaihi/Downloads/whisper_optimization/benchmark/results/20260309_isolation_summary.md`

### Result snapshot
- Short:
  - Option A and B are effectively neutral/slightly worse on runtime median (-0.27%, -0.13% vs baseline)
- Medium:
  - Option A is inconclusive (runtime median slightly worse, very high run variance)
  - Option B is strongly faster (+35.65% runtime median improvement vs baseline)
- Long:
  - Option A and B both improve runtime median (+31.41%, +34.50% vs baseline)
  - Option B has better stability (std dev 0.457s vs Option A 2.080s)
- Correctness:
  - All runs pass WER/CER gates (all `correctness_pass=true`)

### Certainty assessment
- Option B is the strongest candidate from this pass.
- Baseline long variance was high in this run set, so final certainty is moderate (not final-proof).
- Recommended confirmation before merging:
  - rerun ordered A/B/A baseline sandwich (baseline -> option -> baseline) on medium+long
  - keep identical thermal/system conditions as much as possible

### Repository state after experiment
- Source reverted to baseline thresholds.
- Rebuilt so binaries match baseline source.
- No optimization currently left applied in source.

## 2026-03-09 Option B Kept + Run
- Source state: Option B is currently applied in `/Users/shaihi/Downloads/whisper_optimization/ggml/src/ggml-metal/ggml-metal-ops.cpp`
  - `ne21_mm_id_min = props_dev->supports_gpu_family_apple7 ? 24 : 32;`
- Build completed after applying Option B.
- Benchmark executed (short only):
  - Command: `./benchmark/bench.sh --variant metal-optB-final --audio short`
  - Result dir: `/Users/shaihi/Downloads/whisper_optimization/benchmark/results/20260309_084640_metal-optB-final`
  - Summary: `/Users/shaihi/Downloads/whisper_optimization/benchmark/results/20260309_084640_metal-optB-final/summary.csv`
- Key metrics (short):
  - runtime_median_s: `1.061812124986318`
  - throughput_mean_audio_s_per_s: `28.18097234137864`
  - runtime_std_dev_s: `0.00702706432431074`
  - correctness_pass: `True`
