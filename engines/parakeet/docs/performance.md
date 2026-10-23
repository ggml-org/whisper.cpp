# parakeet engine: performance deep dive

Part of the [parakeet engine documentation](../README.md).

## Multi-machine benchmark: method and caveats

Method and caveats:

- **Two-point slope**: each cell runs the clip `R_lo=1` and `R_hi` times in
  one process; `slope = (median_wall(R_hi) - median_wall(R_lo)) / (R_hi -
  R_lo)`. `R_hi` = 11, or 51–101 for the short clip on the MacBook. 3 timed
  reps per point (5 on the MacBook) plus an untimed warm-up, repeats via the
  CLI's native `--bench-runs`, one job at a time, `CUDA_VISIBLE_DEVICES=0`,
  `GGML_VK_VISIBLE_DEVICES=0`, thread count fixed per host (16 / 10 / 8).
- **Backend proven per cell** by `.backend` in the bench JSON.
- **Gate**: worst rep spread at `R_hi` per row — at most 10 % clean; rows
  under proven host contention are withheld. The M5 CPU long row is withheld
  because the measurement session drove the 16 GiB machine into swap, so
  that row would measure the SSD; M5 rows in general carry fanless-throttling
  spread of 7–24 % (three Metal sessions put the short clip at 51 / 61 /
  81 ms), so treat them as directional.
- **Linux CPU lanes are out of scope**: the shipping x86/Strix CPU path has
  a known graph-structure inefficiency, fixed in engine PR #227 / registry
  PR #360, neither of which is in the `46afe7d9` tree measured here. The M5
  CPU short row is kept.
- One build note: ggml needed `#include <cuda/iterator>` in
  `src/ggml-cuda/{top-k,argsort}.cu` to compile under CUDA 13.3 (build fix
  only).

### speech-cpp CI (2026-09-07)

Fresh CPU-baseline snapshot from `speech-benchmark-desktop.yml` on the
hosted-Linux and self-hosted macOS runners (5 timed runs + 1 warmup, `jfk.wav`
fixture, ~11 s).

| Model | Runner | Backend | Median wall ms | Median RTF | Peak RSS MiB |
|---|---|---|---:|---:|---:|
| Parakeet CTC 0.6b q8_0 | linux | ggml-cpu | 2111 | 0.192 | 858 |
| Parakeet CTC 0.6b q8_0 | macos | ggml-cpu | 184 | 0.0170 | 914 |
| Whisper base | linux | (CPU) | 1906 | 0.173 | 298 |
| Whisper base | macos | Metal | 584 | 0.0530 | 360 |
| Whisper small | linux | (CPU) | 6122 | 0.557 | 797 |
| Whisper small | macos | Metal | 564 | 0.0510 | 878 |
| Whisper tiny | linux | (CPU) | 1035 | 0.0940 | 182 |
| Whisper tiny | macos | Metal | 565 | 0.0510 | 240 |

Source: [workflow run 34113144218](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/34113144218) (2026-09-07).

### TDT decode on CUDA and Metal

The fused decoder ops, the unrolled greedy loop and the per-backend lowerings
were measured against the engine as published before them, on the same
machine in the same session: parakeet-tdt-0.6b-v3, `n_gpu_layers 1`,
`--bench --bench-warmup 10 --bench-runs 20`, the two builds alternated, the
median run time of each build kept. Cells read `before -> after ms (speedup)`.
Ten warmup runs matter on Metal: the first two or three timed runs of a
process are up to 30 % faster than the rest, so a short warmup reports a
transient, not the sustained rate.

| Backend | Quant | 30 s clip | 90 s clip | 306 s clip |
|---|---|---:|---:|---:|
| CUDA, RTX 3080, CUDA graphs on | q8_0 | 80.0 -> 32.0 (2.50x) | 308 -> 119 (2.59x) | 1292 -> 495 (2.61x) |
| CUDA, RTX 3080, CUDA graphs on | q4_0 | 79.4 -> 30.8 (2.58x) | 305 -> 116 (2.63x) | 1275 -> 484 (2.64x) |
| CUDA, RTX 3080, CUDA graphs off | q8_0 | 80.0 -> 34.9 (2.29x) | 308 -> 127 (2.42x) | not measured |
| CUDA, RTX 3080, CUDA graphs off | q4_0 | 79.4 -> 33.5 (2.37x) | 305 -> 125 (2.45x) | not measured |
| Metal, Apple M5 | q8_0 | 518 -> 281 (1.84x) | 2561 -> 1095 (2.34x) | 18130 -> 5531 (3.28x) |
| Metal, Apple M5 | q4_0 | 486 -> 254 (1.91x) | 2416 -> 1020 (2.37x) | 17559 -> 5216 (3.37x) |

The 306 s cells come from an earlier run with two warmups and seven timed
runs, medians over six rounds; that clip's run-to-run drift is under 2 %, so
the shorter warmup does not move it. CUDA graphs are a ggml build option
(`GGML_CUDA_GRAPHS`) that is off by default; the 11 s clip lands between
2.2x and 2.3x on CUDA and around 2x on Metal. The Metal host carried a
1-minute load between 2 and 3 during these runs.

Word error rate on the same 500-utterance LibriSpeech subset, Whisper text
normalisation: CPU 2.22 %, CUDA q8_0 2.21 % and q4_0 2.28 %, Metal q8_0
2.15 % and q4_0 2.24 %. Sortformer, streaming and ASR-plus-diarization
outputs are byte-equal to the previous build on both backends.

### Nemotron 3 Diarization on GPU

Nemotron 3 Diarization runs its whole graph on CUDA, Vulkan, and Metal as one
scheduler split with no CPU fallback. Medians of warm runs of the official
q8_0 GGUF with a prewarmed engine on `diarization-sample-16k.wav` (27.3 s)
and the same clip repeated ten times (273 s). Streaming uses the default live
geometry (1040 ms chunks, 80 ms right context, 264 speaker-cache and 80 FIFO
rows) and feeds 100 ms blocks as fast as the engine accepts them; the time
covers the whole session including `finalize()`.

| Backend | Offline 27.3 s | Offline 273 s | Streaming 27.3 s | Streaming 273 s | Slowest chunk |
|---|---:|---:|---:|---:|---:|
| CPU, Ryzen 9 9950X3D, 32 threads | 118-246 ms | 1169 ms | 2.1 s | 36.3 s | 459 ms |
| CUDA, RTX 5090 | 5.9 ms | 40.4 ms | 91 ms | 1.0 s | 4.1 ms |
| Vulkan, RTX 5090 | 5.9 ms | 59.4 ms | 106 ms | 1.09 s | 8.9 ms |
| Metal, Apple M3 Ultra | 16.2 ms | 199 ms | 333 ms | 3.87 s | 15.7 ms |

The CPU host was shared with other jobs, so its row is indicative only.
Offline probabilities against the NeMo-Speech.cpp reference on the 27.3 s
clip, as maximum absolute error and relative L2 error: CPU 0.023 and 0.0016,
CUDA 0.037 and 0.0021, Vulkan 0.031 and 0.0016, Metal 0.033 and 0.0015. The
test tolerance is 0.04 and 0.003. On the 273 s input, the GPU probabilities
differ from CPU by 0.0005 to 0.0006 on average, and 0.02 % to 0.03 % of
frame decisions at the 0.5 threshold flip. The per-block error stays flat
across the input, so long contexts do not drift. Setting `GGML_PREC_F32` on
the matmuls lowers the CUDA maximum error to 0.033 at a 10 % cost and doubles
the Vulkan time, so the graph keeps the default precision.

The first call compiles GPU kernels. With the NVIDIA shader disk cache
disabled, Vulkan spends 2.3 s on the first offline call and 0.25 s on the
first streaming chunk; CUDA spends 67 ms. `EngineOptions::prewarm` runs one
offline pass of `prewarm_audio_seconds` and one chunk at the default live
geometry during construction, after which the first offline call takes 8 ms
on both backends and no streaming chunk exceeds 5 ms.

Backends without fused flash attention for this graph, such as ggml-opencl
on Adreno, use an unfused attention on the GPU. Forced on the RTX 5090, it
costs 15 % per streaming chunk (CUDA 3.80 -> 4.35 ms, Vulkan 3.90 -> 4.48 ms)
and about 3x on the 273 s offline graph (CUDA 16.6 -> 50.8 ms, Vulkan 28.3 ->
61.1 ms), because its score matrix is quadratic in the input length. Sending
the fused attention to the CPU instead, which is what the scheduler would do
without the fallback, makes a streaming chunk 10x slower (CUDA 38.7 ms,
Vulkan 40.0 ms). The unfused path passes the same NeMo-Speech.cpp reference
check on CPU, CUDA, Vulkan, Metal, and Adreno OpenCL.

On a Samsung Galaxy S25 (Snapdragon 8 Elite, Adreno 830), the same 27.3 s
clip and harness, with the F32 matmul precision Adreno receives:

| Backend | Offline 27.3 s | Streaming 27.3 s | Slowest chunk | Max abs | Rel L2 |
|---|---:|---:|---:|---:|---:|
| CPU, Oryon | 331 ms | 5.7 s | 478 ms | 0.027 | 0.0017 |
| OpenCL, Adreno 830 | 176 ms | 2.3 s | 154 ms | 0.033 | 0.0015 |
| Vulkan, Adreno 830 | 726 ms | 12.4 s | 712 ms | 0.035 | 0.0016 |

Without the F32 request, OpenCL measured 0.037 and 0.0030 and Vulkan 0.042
and 0.0040, at 2-3 % less time. OpenCL runs the unfused attention because
ggml-opencl rejects flash attention on Adreno. With
`GGML_OPENCL_FA_ADRENO=1` it takes 108 ms offline and 86 ms per chunk, but it
fails the test's negative-peak check and changes the streaming segments, so
leave it off.

### Nemotron 3 Diarization quantizations

Measured on 2026-10-14 with speech base `e2c3d8bc`, ggml-speech
`6bb85ad673ccf18850524a4d4d4cf5e781a374c7`, RTX 5090 CUDA 13.3, and an
8-thread CPU run on a Ryzen 9 9950X VM exposing 10 CPUs. All formats were
converted directly from the pinned NeMo checkpoint by NeMo-Speech.cpp
`97a15afa5caa9bce5baaa86c1184103877af4101`; these are not requantizations
of the published Q8 file. F32 is a linear-weight reference: the converter
still stores its convolution kernel in F16. The published Q8 fixture also
passes the existing CPU and CUDA probability and streaming regression tests.

`bench-nemotron-quantization` checks finite [0, 1] probabilities and a
consistent speaker count (eight channels for Nemotron 3 Diarization, four for
Sortformer), writes the offline probabilities, warms the full input,
and reports the median of five offline runs plus one streaming session.
The input is `diarization-sample-16k.wav` (27.3 s); streaming feeds 100 ms
blocks with the default 1040 ms chunks. Model load is excluded. Streaming
is a single observation, not a median; small timing differences should not
be treated as reliable speedups. Sizes are decimal MB.

| Precision | MB | CPU offline ms | CUDA offline ms | CPU stream ms | CUDA stream ms |
|---|---:|---:|---:|---:|---:|
| F32 | 395.30 | 139.26 | 5.12 | 2143 | 87.44 |
| F16 | 199.02 | 151.33 | 5.01 | 2134 | 91.10 |
| BF16 | 199.02 | 112.95 | 4.98 | 1597 | 92.61 |
| Q8_0 | 107.01 | 138.94 | 6.12 | 1939 | 105.59 |
| Q6_K | 83.24 | 135.17 | 6.05 | 1883 | 104.45 |
| Q5_K | 70.21 | 169.61 | 5.79 | 2362 | 99.89 |
| Q4_K | 57.94 | 135.02 | 5.94 | 1871 | 99.79 |
| MXFP4 | 54.88 | 167.67 | 5.37 | 2343 | 92.84 |
| NVFP4 | 57.94 | 203.46 | 5.12 | 2848 | 86.65 |

Frame DER against the existing RTTMs uses the test's 10 ms grid, no collar,
optimal speaker mapping, and 200 ms minimum segment. Each pair below is
`abcba` / `abcdba` (160.6 / 191.2 s). Offline uses automatic long-form
chunking; streaming uses 1040 ms chunks. Coverage and speaker re-entry are
also checked. These two fixtures do not establish quality on overlapping
speech, eight simultaneous speakers, languages, or general meeting audio.

| Precision | CPU offline DER % | CUDA offline DER % | CUDA streaming DER % |
|---|---:|---:|---:|
| F32 | 3.71 / 4.36 | 3.71 / 4.36 | 6.09 / 15.35 |
| F16 | 3.71 / 4.36 | 3.71 / 4.36 | 4.76 / 17.56 |
| BF16 | 3.71 / 4.36 | 3.71 / 4.36 | 8.18 / 8.77 |
| Q8_0 | 3.71 / 8.02 | 3.70 / 4.35 | 4.78 / 17.59 |
| Q6_K | 3.71 / 4.36 | 3.72 / 4.37 | 13.60 / 13.37 |
| Q5_K | 3.72 / 4.36 | 3.71 / 4.36 | 4.75 / 15.22 |
| Q4_K | 3.68 / 4.30 | 3.68 / 4.30 | 4.74 / 5.68 |
| MXFP4 | 26.17 / re-entry failure | 7.23 / 7.34 | 20.65 / 22.83 |
| NVFP4 | 3.63 / 4.26 | 3.31 / 3.99 | 4.51 / 4.71 |

On the short clip, CUDA probability relative L2 errors against F32 after
optimal speaker permutation were Q8 0.0049, Q6_K 0.0065, Q5_K 0.0096,
Q4_K 0.0340, MXFP4 0.0503, and NVFP4 0.0420. Corresponding CUDA-versus-CPU
errors using the same GGUF were 0.0014, 0.0020, 0.0089, 0.0099, 0.0671,
and 0.0265. Quantized activation arithmetic can differ across backends;
these comparisons are diagnostics, not independent NVIDIA reference
parity for each precision. Do not reuse the Q8 probability reference as
an acceptance target for a differently quantized model or loosen its
existing tolerances to admit them.

Q4_K is the strongest size-oriented candidate in this fixture set: 46%
smaller than Q8 with comparable offline latency and quality. Q6_K saves
22% with less offline probability drift, but its live `abcba` quality
regresses. Q5_K saves 34%, with slower CPU inference here. F16/BF16 offer
higher precision at larger size. NVFP4 is a promising RTX 5090 candidate,
but is slower on this CPU and needs wider device/quality validation.
MXFP4 saves only another 3.07 MB versus Q4_K and fails CPU quality and
speaker continuity; no MXFP4-specific runtime patch is justified by these
results. The tested formats require no graph/kernel changes. Keep Q8 as
the default until broader validation supports changing it.

Reproduce conversion and measurements from the repository root:

```bash
python engines/parakeet/scripts/convert_nemotron_diarization.py --outtype q4_k
cmake --build build-parakeet --target bench-nemotron-quantization test-sortformer-aosc-speakers
./build-parakeet/bench-nemotron-quantization \
  engines/parakeet/models/Nemotron-3-Diarization.converted.q4_k.gguf \
  engines/parakeet/test/samples/diarization-sample-16k.wav /tmp/q4-k-probs.bin 99
ctest --test-dir build-parakeet --output-on-failure \
  -R 'test-nemotron-diarization-(f16|bf16|q6_k|q5_k|q4_k|nvfp4)-'
./build-parakeet/test-sortformer-aosc-speakers \
  --model engines/parakeet/models/Nemotron-3-Diarization.converted.q4_k.gguf \
  --wav engines/parakeet/test/samples/abcdba.wav \
  --ref-rttm engines/parakeet/test/samples/abcdba.rttm \
  --chunk-ms 1040 --n-gpu-layers 99 --der-max 15
```

Use `0` instead of `99` for CPU benchmarking. Install the pinned converter's
Python requirements and provide `NEMO_SPEECH_GGML_BASE` pointing to a built
ggml-base shared library for K-quants when required by the converter.
Configure CMake after generating the models: missing fixture tests are
disabled at configure time. `PARAKEET_TEST_MODEL_DIR` selects their directory;
`PARAKEET_TEST_NEMOTRON_DIARIZATION_QUANTIZATIONS` selects the format list.
The default six extra formats register 24 offline CPU/GPU tests, all passing
on the host above. MXFP4 is available for explicit experiments but excluded
from that default list because it failed. Live failures above remain visible
and are not relabeled as passing offline gates. The follow-up below covers
Q4_K/Q6_K/NVFP4 on desktop Metal and Vulkan. Additional-precision mobile
and downstream `qvac/` validation have not been run.

#### Desktop Metal and Vulkan follow-up

Measured on 2026-10-14 from merged speech revision `dc1e7957`, with the
same ggml revision, converter, and fixtures as above. Model SHA256 hashes
matched between hosts for Q8_0, Q4_K, Q6_K, and NVFP4. Metal runs on Apple
M3 Ultra; Vulkan runs on RTX 5090. No engine or kernel changes were needed
for Q4_K/Q6_K. These measurements extend offline validation, not a claim
that all precisions are interchangeable for live diarization.

Metal timings use the existing harness: median of five warm offline passes
and one streaming session on the 27.3 s clip. Peak RSS comes from macOS
`/usr/bin/time -l` for the entire benchmark process, including loading and
prewarming; it is not an isolated GPU allocation or a portable VRAM metric.
NVFP4's row includes CPU fallback and must not be read as Metal acceleration.

| Precision | Metal offline ms | Metal stream ms | Peak process RSS MB | Graph placement |
|---|---:|---:|---:|---|
| Q8_0 | 14.79 | 331.11 | 190.84 | 1 Metal split |
| Q4_K | 15.07 | 326.66 | 136.63 | 1 Metal split |
| Q6_K | 14.99 | 326.41 | 162.09 | 1 Metal split |
| NVFP4 | 220.10 | 3687.83 | 155.73 | 252 CPU/Metal splits |

Q4_K lowers observed peak process RSS by about 28% relative to Q8 here,
while reducing file size by 46%; neither percentage establishes a general
runtime-memory saving for other input lengths or devices. The pinned
Metal backend explicitly rejects NVFP4 in its matmul support check
(`ggml/src/ggml-metal/ggml-metal-device.m`). Scheduler diagnostics confirm
CPU execution of those weights, so a passing GPU-requested correctness
test does not demonstrate full GPU placement. Prefer Q4_K/Q6_K for Metal;
adding NVFP4 Metal kernels would be a separate ggml change.

The initial Vulkan timing sweep encountered substantial host contention
and is excluded from the timing table. Timings below use three interleaved
rounds on the second, idle RTX 5090 (`GGML_VK_VISIBLE_DEVICES=1`). Each
round contains five warm offline passes and one streaming session; report
the median across rounds and the observed range. The host remained shared,
so these values support comparable performance, not a precise speedup claim.
The earlier quality runs used the first RTX 5090. Placement diagnostics
show one Vulkan GPU split for all four formats, including streaming graphs.

| Precision | Vulkan offline median ms (range) | Vulkan stream median ms (range) |
|---|---:|---:|
| Q8_0 | 7.14 (6.42–8.54) | 114.38 (109.49–125.26) |
| Q4_K | 6.71 (6.18–7.65) | 103.38 (103.03–118.93) |
| Q6_K | 7.15 (6.62–7.67) | 114.68 (113.69–115.75) |
| NVFP4 | 7.31 (6.48–7.66) | 117.52 (112.16–121.24) |

DER uses the same RTTM scoring as above; pairs are `abcba` / `abcdba`.
Metal NVFP4 quality includes its CPU matmuls. The 1040 ms streaming results
are deliberately reported separately from offline acceptance.

| Precision | Metal offline DER % | Vulkan offline DER % | Metal live DER % | Vulkan live DER % |
|---|---:|---:|---:|---:|
| Q8_0 | 3.70 / 4.36 | 3.70 / 4.36 | 13.79 / 12.90 | 4.80 / 17.34 |
| Q4_K | 3.68 / 4.32 | 3.68 / 4.31 | 6.17 / 18.18 | 5.77 / 17.12 |
| Q6_K | 3.72 / 4.37 | 3.72 / 4.37 | 5.82 / 18.94 | 5.50 / 15.08 |
| NVFP4 | 3.63 / 4.26 | 3.63 / 4.26 | 3.96 / 4.32 | 3.97 / 4.53 |

On the short clip, relative L2 differences from each host's CPU output for
the identical GGUF were Q8 0.00180, Q4_K 0.00255, Q6_K 0.00253, and
NVFP4 0.00105 for Metal; Vulkan measured 0.00172, 0.00261, 0.00261, and
0.00110 respectively. Speaker permutations are aligned before comparison.
These diagnostics do not replace independent reference parity or long-form
quality testing: small local numeric differences can accompany markedly
different live speaker-cache outcomes across backends. The root cause of
the live divergence has not been isolated, and no precision policy or
streaming algorithm was changed to make these fixtures pass.

Both hosts passed 14 registered regression tests: the official Q8 CPU/GPU
reference tests plus the two offline RTTM fixtures for Q4_K/Q6_K/NVFP4 on
CPU and the requested GPU backend. The Metal NVFP4 passes include fallback.
Q4_K and Q6_K fail the standalone live `abcdba` 15% DER gate on both
backends; Q8 also fails that gate on Vulkan. Keep Q8 as the default and
limit the new K-quant recommendation to the validated offline use case.

After generating or copying these models, configure the desired backend
with the pinned ggml source, then run the existing regression targets:

```bash
ctest --test-dir build-parakeet --output-on-failure \
  -R '^test-nemotron-diarization(-gpu|-(q4_k|q6_k|nvfp4)-(abcba|abcdba)-(cpu|gpu))?$'
GGML_SCHED_DEBUG=1 ./build-parakeet/bench-nemotron-quantization \
  engines/parakeet/models/Nemotron-3-Diarization.converted.q4_k.gguf \
  engines/parakeet/test/samples/diarization-sample-16k.wav /tmp/q4-k-probs.bin 99
```

The second command exposes scheduler splits; use it separately from timing
runs so logging does not contaminate latency. The live command in the
previous section reproduces the four-speaker failure. Other Vulkan GPUs,
other Metal precisions, mobile, broader meeting audio, overlap/eight-speaker
quality, and downstream QVAC integration remain outside this validation.

### Nemotron 3 Diarization on long offline inputs

A single offline graph loses speaker identity on inputs longer than about
two minutes, so offline inputs past the 90 s long-form window run through
the speaker-cache path in 30 s chunks with 1040 ms of left and right
context. Frame-level DER (no collar, optimal speaker mapping, 200 ms
minimum segment) against the fixture RTTMs, RTX 5090 CUDA:

| Input | Single graph | Long-form, 1.04 s chunks | 10 s chunks | 30 s chunks | 60 s chunks |
|---|---:|---:|---:|---:|---:|
| `abcba.wav`, 160.6 s, 3 speakers | 28.3 % | 12.4 % | 26.9 % | 3.7 % | 3.9 % |
| `abcdba.wav`, 191.2 s, 4 speakers | 47.5 % | 23.4 % | 14.5 % | 4.3 % | 4.4 % |

On prefixes of the same files the single graph matches the 30 s long-form
path to within 0.2 points up to 105 s (3.2-4.1 %), slips at 120 s (5.2-5.4 %
against 4.4-4.5 %), and collapses at 140 s (34-36 % against 4.1-5.1 %); the
single graph also fails the re-entry check of `test-sortformer-aosc-speakers`
on both full files. The 90 s auto window keeps single-graph results for short
inputs and leaves a margin below the measured degradation. The long-form
path reaches 3.70 % and 4.36 % on CUDA, Vulkan, Metal, Adreno OpenCL and
Vulkan, and the Snapdragon CPU; the x86 CPU reaches 3.71 % on `abcba` but
8.0 % on `abcdba`, where one near-tie flips a speaker, which is why the tests
gate at 15 %. On the RTX 5090 a
273 s input takes 82 ms long-form against 57 ms for the single graph, and a
one-hour input takes 1.0 s. The engine feeds the long-form session in 30 s
blocks, so its sample ring stays bounded instead of holding a copy of the
whole input.

### Nemotron 3 Diarization and Sortformer v2.1 on the Apple Neural Engine

[Core ML diarizer sidecars](backends.md#core-ml-diarizer-sidecars) on an Apple
M4 mac mini (10-core GPU, 16 GB, macOS 15.3.2), against ggml Metal on the same
machine (`PARAKEET_COREML_DISABLE=1`). The Nemotron 3 Diarization rows use the
official q8_0 GGUF with 358- and 745-frame sidecars; the Sortformer v2.1 rows
use the f16 GGUF with a 410-frame sidecar. `bench-nemotron-quantization`
produced the offline medians and the 27.3 s streaming sessions as described
above. The 273 s input repeats `diarization-sample-16k.wav` ten times.
Sortformer streams with its default 2 s chunks. Its 273 s offline input exceeds
the sidecar capacity, so both columns run ggml. Its 273 s streaming row comes
from a second session in the same process, because the harness's first session
would include the sidecar load.

| Model | Input | ggml Metal | Neural Engine | Speedup |
|---|---|---:|---:|---:|
| Nemotron 3 Diarization | offline 27.3 s | 36.9 ms | 14.1 ms | 2.6x |
| Nemotron 3 Diarization | offline 273 s, 30 s long-form blocks | 725 ms | 331 ms | 2.2x |
| Nemotron 3 Diarization | streaming 27.3 s | 560 ms | 327 ms | 1.7x |
| Nemotron 3 Diarization | streaming 273 s | 8.55 s | 3.13 s | 2.7x |
| Sortformer v2.1 | offline 27.3 s | 64.3 ms | 38.0 ms | 1.7x |
| Sortformer v2.1 | offline 273 s | 2.09 s | 2.10 s | ggml in both |
| Sortformer v2.1 | streaming 27.3 s | 489 ms | 352 ms | 1.4x |
| Sortformer v2.1 | streaming 273 s | 7.49 s | 3.54 s | 2.1x |

The same bundles and GGUFs on an Apple M3 Ultra Mac Studio (60-core GPU,
96 GB, macOS 26.7.1), with the diarizers on CPU and Neural Engine compute
units, the default. Each cell is the median of three runs of the harness
above, with the host otherwise idle:

| Model | Input | ggml Metal | Neural Engine | Speedup |
|---|---|---:|---:|---:|
| Nemotron 3 Diarization | offline 27.3 s | 15.0 ms | 13.0 ms | 1.15x |
| Nemotron 3 Diarization | offline 273 s, 30 s long-form blocks | 226 ms | 325 ms | 0.70x |
| Nemotron 3 Diarization | streaming 27.3 s | 295 ms | 298 ms | 0.99x |
| Nemotron 3 Diarization | streaming 273 s | 3.37 s | 2.89 s | 1.17x |
| Sortformer v2.1 | offline 27.3 s | 25.9 ms | 30.7 ms | 0.84x |
| Sortformer v2.1 | offline 273 s | 431 ms | 432 ms | ggml in both |
| Sortformer v2.1 | streaming 27.3 s | 287 ms | 352 ms | 0.82x |
| Sortformer v2.1 | streaming 273 s | 3.38 s | 3.68 s | 0.92x |

The Neural Engine takes about as long as on the M4 (13.0 against 14.1 ms for
the 27.3 s Nemotron input); the difference is the Metal baseline, 2.5 times
faster on the Ultra. With `PARAKEET_COREML_COMPUTE_UNITS=all`, Core ML places
every operation of the three bundles on the GPU instead, and the same rows run
1.4-1.8x slower than ggml Metal. The first load on this machine compiles each
Nemotron capacity in 9.6-11.6 s and the Sortformer bundle in 17.9 s. The
parity tests pass there with maximum errors of 0.0049 (Nemotron) and 0.0023
(Sortformer), and relative L2 errors up to 0.00043 and 0.00051.

`test-nemotron-diarization-coreml-parity` and
`test-sortformer-diarizer-coreml-parity` compare each sidecar with the ggml
graph on the same inputs, as maximum absolute and relative L2 error of the
probabilities. Nemotron offline: 0.0033 and 0.00041; live chunk with a full
speaker cache: 0.0037 and 0.00030; 300-row cache plus a 342-frame block: 0.0045
and 0.00044. Sortformer batch: 0.0023 and 0.00046; 409-row AOSC slab: 0.0022
and 0.00048. Against the NeMo-Speech.cpp reference on the 27.3 s clip, the
Nemotron sidecar measures 0.030 and 0.0016 and ggml Metal 0.033 and 0.0015,
under the 0.04 and 0.003 test tolerance.

Speaker-cache DER on the long-form fixtures, frame level as above:

| Model | Fixture and path | ggml Metal | Neural Engine |
|---|---|---:|---:|
| Nemotron 3 Diarization | `abcba`, offline long-form | 3.70 % | 7.98 % |
| Nemotron 3 Diarization | `abcdba`, offline long-form | 4.36 % | 4.35 % |
| Nemotron 3 Diarization | `abcba`, live 1040 ms | 13.79 % | 6.72 % |
| Nemotron 3 Diarization | `abcdba`, live 1040 ms | 12.90 % | 5.83 % |
| Sortformer v2.1 | `abcba`, streaming 2 s | 26.13 % | 24.18 % |
| Sortformer v2.1 | `abcdba`, streaming 2 s | 21.14 % | 21.06 % |

The speaker cache feeds each chunk's probabilities into the next chunk's
input, so a near-tie in its compression can flip a speaker for the rest of a
recording, as the x86 CPU does on `abcdba` above. On the `abcba` long-form
step where the Neural Engine and Metal trajectories separate, the two differ
by a relative 0.95 % on identical inputs, while the CPU differs from them by
2.1 % and 3.0 %. The 6.7 s of speaker confusion on that fixture is such a flip,
not a systematic error: the Neural Engine scores better than Metal on both
live runs.

## ASR addon CI (2026-08-12)


`RTF = inference_time / audio_duration`; lower is faster. The recorded 2026-08-12
Linux x86-64 CI run used q8_0 registry models, one warmup, and five timed runs
on an NVIDIA RTX 4000 SFF Ada:

| Model | CPU RTF | CPU wall | Vulkan RTF | Vulkan wall |
|---|---:|---:|---:|---:|
| CTC | 0.112 | 2256 ms | 0.0022 | 43 ms |
| TDT | 0.130 | 2607 ms | 0.0044 | 88 ms |
| EOU | 0.051 | 1034 ms | 0.0034 | 68 ms |
| Sortformer | 0.046 | 922 ms | 0.0019 | 38 ms |
| Sortformer streaming | 0.032 | 646 ms | 0.0034 | 69 ms |

The same run also covers the self-hosted Apple M4 Mac mini (Metal, q8_0):
CTC 0.0113, TDT 0.0150, EOU 0.0097, Sortformer 0.0061, Sortformer
streaming 0.0070.

Source: [workflow run 31603189415](https://github.com/tetherto/qvac/actions/runs/31603189415),
12 August 2026, runner `qvac-ubuntu2204-x64-gpu`, benchmarking the published
`@qvac/asr-ggml@0.1.1` addon (released 2026-08-03, pinning `parakeet-cpp`
2026-08-03).

## TranscribePsy long-form CPU windows (2026-10)

[TranscribePsy-250M-EU-sub](https://huggingface.co/qvac/TranscribePsy-250M-EU-sub-gguf)
q8_0 on a 412.7 s recording (the three `test/samples` clips concatenated twelve
times), x86-64 Linux, eight threads, ggml CPU encoder and decoder. The engine's
automatic long-form window (300 s, 256 frames of context) against the 60 s /
25 % windows the model was validated with on phones
(`--window-sec 60 --overlap 0.25`), and the memory work that lets a windowed
pass hold one encoder graph and release the audio and STFT scratch once the mel
exists:

| Build | Windows | Mel | Encoder | Decoder | Peak RSS |
|---|---|--:|--:|--:|--:|
| before this work, automatic window | 2 x 300 s | 38 ms | 19.6 s | 92 ms | 4.4 GB |
| this work, automatic window | 2 x 300 s | 19 ms | 18.0 s | 102 ms | 2.4 GB |
| this work, `--window-sec 60 --overlap 0.25` | 10 x 60 s | 19 ms | 10.8 s | 86 ms | 0.9 GB |
| TranscribePsy fork (q8_0 host decoder), 60 s / 25 % default | 10 x 60 s | 141 ms | 12.8 s | 379 ms | 1.6 GB |

The ggml-graph CPU decoder is 3 to 4 times faster than the fork's packed q8_0
host decoder on the same input (92 ms against 379 ms here; 5 ms against 12 ms
on `jfk.wav`), which is why that decoder was not brought over. The blocked STFT
halves the mel time and bounds its scratch by a 2048-frame block instead of the
input length. Short clips are unchanged: `jfk.wav` (11 s) encodes in 170 ms and
decodes in 5 ms, RTF 0.016.

### TranscribePsy on the Hexagon NPU (Galaxy S25, 2026-10-22)

TranscribePsy q8_0 on a Galaxy S25 (Snapdragon 8 Elite, HTP v79), six CPU
threads, default 60 s / 25 % windows, median of three runs after one warm-up,
each run gated on thermal status 0; the transcripts match the CPU's on every
clip:

| Clip | Backend | Encoder | Decoder | Inference | RTF |
|---|---|--:|--:|--:|--:|
| `jfk.wav`, 11 s | CPU | 153 ms | 16 ms | 173 ms | 0.016 |
| `jfk.wav`, 11 s | Hexagon | 122 ms | 15 ms | 140 ms | 0.013 |
| 67.5 s, two windows | CPU | 4.18 s | 96 ms | 4.29 s | 0.064 |
| 67.5 s, two windows | Hexagon | 1.57 s | 122 ms | 1.71 s | 0.025 |
| `abcdba.wav`, 191 s, five windows | CPU | 14.5 s | 267 ms | 14.8 s | 0.078 |
| `abcdba.wav`, 191 s, five windows | Hexagon | 4.13 s | 355 ms | 4.52 s | 0.024 |

The CPU's 67.5 s runs spread from 3.7 to 6.2 s with the phone's temperature;
the Hexagon runs stay within a millisecond. Where the encoder time goes on one
844-frame window (`GGML_HEXAGON_PROFILE=1`, `--window-sec 0` on the 67.5 s
clip):

| Attention graph | Encoder | Attention ops | Encoder cosine vs CPU |
|---|--:|--:|--:|
| fused, HVX (head size 80) | 3.50 s | 2.00 s flash attention + 0.91 s position matmul | 0.99995 |
| fused, HMX (head padded to 128) | 1.56 s | 0.04 s flash attention + 0.91 s position matmul | 0.81 |
| unfused, HVX (head size 80) | 6.16 s | 5.34 s in the three batched matmuls | 0.99995 |
| unfused, HMX (head padded to 96), shipped | 0.91 s | 0.06 s in the three batched matmuls + 0.09 s softmax | 0.99995 |

Parakeet-TDT-0.6B, whose 128-wide heads already suit HMX, gains the same way
on that window: the fused HMX attention it took before ran the encoder in
1.60 s at cosine 0.76 against the CPU, the unfused HMX graph in 0.94 s at
0.9996.

### TranscribePsy on the Apple Neural Engine (Mac mini M4, 2026-10)

The same 412.7 s recording on a Mac mini M4 (macOS 15.3, six threads), built
with `PARAKEET_COREML=ON`, with the published
[Core ML distribution](https://huggingface.co/qvac/TranscribePsy-250M-EU-sub-coreml)
(encoder-less GGUF, fixed-window Neural Engine encoder, 16-lane TDT decoder)
against the q8_0 GGUF on the ggml CPU encoder and decoder. Engine times are
`EngineResult` stage times from `test-transcribepsy --long-wav`; CLI times are
the `[BENCH]` line and `/usr/bin/time -l`:

| Path | Encoder | Decoder | Total | RTF | Peak RSS |
|---|--:|--:|--:|--:|--:|
| `Engine::transcribe`, Core ML GGUF: 34 pause-cut segments on the Neural Engine encoder, batched Neural Engine decoder | 529 ms | 420 ms | 968 ms | 0.0023 | |
| `Engine::transcribe`, q8_0 GGUF: ggml CPU, automatic 300 s windows | 27.5 s | 202 ms | 27.7 s | 0.067 | |
| CLI, Core ML GGUF: 34 Neural Engine encoder windows (15 s, 16 frames of context), ggml CPU decoder | 501 ms | 191 ms | 709 ms | 0.0017 | 251 MB |
| CLI, q8_0 GGUF: ggml CPU, `--window-sec 60 --overlap 0.25` | 12.0 s | 192 ms | 12.2 s | 0.030 | 1.0 GB |

`jfk.wav` (11 s): 18 ms on the Neural Engine encoder against 145 ms on the M4
CPU; the batched decoder takes 59 ms for that single segment against 5 ms for
the sequential CPU decoder, since one segment fills one of its 16 lanes. The
Neural Engine decoder pays off on phones, where the prediction network is
memory-bandwidth bound; on the M4 the CPU decoder is faster, and
`PARAKEET_ANE_BATCH_DECODE=0` keeps the Neural Engine encoder with the
sequential decoder.

Accuracy on that recording, word error rate against the three clips' own
single-clip transcripts repeated twelve times (the clips are concatenated
without pauses, so every seam falls in running speech). The 15 s Neural
Engine windows with 1.28 s of seam context are the fork's published geometry;
the same geometry on the ggml encoder is given for reference, and the TDT
greedy decoder's sensitivity to it is large (187 against 188 encoder frames per
window changes the result by three points):

| Path | WER |
|---|--:|
| CLI, q8_0 GGUF, ggml CPU, `--window-sec 60 --overlap 0.25` | 5.1 % |
| `Engine::transcribe`, q8_0 GGUF, ggml CPU, automatic 300 s windows | 7.1 % |
| CLI, ggml CPU at the Neural Engine geometry (187-frame windows, 16 frames of context) | 8.1 % |
| CLI, ggml CPU, 188-frame windows, 16 frames of context | 11.2 % |
| CLI, Core ML GGUF, Neural Engine encoder windows, ggml decoder | 10.5 % |
| CLI, Core ML GGUF, Neural Engine encoder windows with 32 frames of context (`PARAKEET_COREML_ANE_CONTEXT_FRAMES=32`) | 8.4 % |
| `Engine::transcribe`, Core ML GGUF, pause-cut segments and the batched Neural Engine decoder | 11.6 % |
| The TranscribePsy fork's own CLI on the Core ML GGUF (its internal 1500-frame tiler and q8_0 host decoder) | 7.1 % |

The committed frames nearest a window's right edge differ most from the
full-context encoder (cosine 0.63 to 0.72 against 0.84 mid-window), and a
single 15 s window matches the ggml encoder frame for frame (cosine 0.98) and
transcribes the sample clips identically, so the loss comes from where the
seams fall, not from the Neural Engine arithmetic. The per-backend accuracy
section below shows what the seams cost on single-language speech (nothing to
0.6 %) and why this recording exaggerates it: every seam that lands near the
French sentence at a language switch can flip it.

### TranscribePsy: port against the research fork (2026-10)

The port on this branch (`58bf7d07`, master's parakeet engine with the ggml
speech head) against the research fork
[tether-ai-research/qvac-transcribepsy](https://github.com/tether-ai-research/qvac-transcribepsy)
at `364ec36` (2026-10-15, its vendored ggml), both built Release with the same
flags per machine (CUDA on Linux, Metal and `PARAKEET_COREML=ON` on the Macs):

- Linux x86-64 (Ubuntu 25.10), AMD Ryzen 9 9950X with 10 cores exposed, RTX
  5090 32 GB, CUDA 13.3; 8 threads, plus a 4-thread long-mix row.
- Mac Studio M3 Ultra (20 performance and 8 efficiency cores, 96 GB), macOS
  15.7; 8 threads, plus a 16-thread long-mix row.
- Mac mini M4 (4 performance and 6 efficiency cores, 16 GB), macOS 15.3; 6
  threads, plus a 4-thread long-mix row.

Inputs are the three `test/samples` clips (`jfk.wav` 11 s, `sample-16k.wav`
20 s, `fr-multipiece-16k.wav` 3.3 s), the 412.7 s long-mix recording (the
three clips concatenated twelve times) and a 619 s recording (eighteen times);
the long recordings have no pauses, so every window seam falls in running
speech. Every number is the median of three timed runs after one warm-up in
the same process: `parakeet --bench --bench-runs 3 --bench-warmup 1` for the
CLI, and for the engine API a small driver that constructs `parakeet::Engine`
with the same `EngineOptions` on both trees and reads the `EngineResult` stage
times. Peak RSS is the process maximum reported by `/usr/bin/time`, so it
covers the model load and all four runs. WER is against each clip's own
transcript, which both trees produce identically on every short clip and
which the long recordings repeat. The fork's 60 s / 25 % windows are its CLI
default and the setting it validated on phones, so the port runs them as
`--window-sec 60 --overlap 0.25` on the CLI and as 750-frame windows with 94
frames of context through the engine API. The two short clips not shown below
behave like `jfk.wav`.

| Machine | Path, long-mix (412.7 s) | Fork: total / peak RSS / WER | Port: total / peak RSS / WER |
|---|---|---|---|
| Linux, Ryzen 9 9950X, RTX 5090 | CLI, ggml CPU, 8 threads, 60 s / 25 % windows (fork default; port `--window-sec 60 --overlap 0.25`) | 13.4 s / 1.8 GB / 5.3 % | 12.1 s / 1.1 GB / 5.1 % |
|  | CLI, ggml CPU, 8 threads, port default (automatic 300 s window) | - | 20.6 s / 2.6 GB / 7.1 % |
|  | Engine API, ggml CPU, 8 threads, default options (automatic 300 s window on both) | 19.8 s / 9.7 GB / 6.3 % | 20.6 s / 2.6 GB / 7.1 % |
|  | Engine API, ggml CPU, 8 threads, 750-frame windows with 94 frames of context | 12.0 s / 3.2 GB / 5.8 % | 12.2 s / 1.1 GB / 5.1 % |
|  | CLI, CUDA (`--n-gpu-layers 999`), 60 s / 25 % windows | fails (`tdt_greedy_decode` rc=6) | 207 ms / 686 MB / 5.3 % |
|  | Engine API, CUDA, default options | fails (`tdt_greedy_decode` rc=6) | 295 ms / 704 MB / 6.9 % |
| Mac Studio M3 Ultra | CLI, ggml CPU, 8 threads, 60 s / 25 % windows (fork default; port `--window-sec 60 --overlap 0.25`) | 8.6 s / 1.7 GB / 5.3 % | 7.6 s / 1019 MB / 5.1 % |
|  | CLI, ggml CPU, 8 threads, port default (automatic 300 s window) | - | 17.4 s / 2.5 GB / 7.1 % |
|  | Engine API, ggml CPU, 8 threads, default options (automatic 300 s window on both) | 12.8 s / 9.6 GB / 6.2 % | 17.3 s / 2.4 GB / 7.1 % |
|  | Engine API, ggml CPU, 8 threads, 750-frame windows with 94 frames of context | 7.9 s / 3.0 GB / 5.8 % | 7.6 s / 1008 MB / 5.1 % |
|  | CLI, Metal (`--n-gpu-layers 999`), 60 s / 25 % windows | fails (`tdt_greedy_decode` rc=6) | 876 ms / 495 MB / 5.1 % |
|  | Engine API, Metal, default options | fails (`tdt_greedy_decode` rc=6) | 1.3 s / 583 MB / 6.9 % |
|  | CLI, Core ML GGUF (Neural Engine encoder; fork: q8_0 host decoder, port: ggml decoder) | 999 ms / 209 MB / 7.1 % | 688 ms / 238 MB / 10.5 % |
|  | Engine API, Core ML GGUF (Neural Engine encoder and batched Neural Engine decoder) | 1.1 s / 196 MB / 12.9 % | 1.1 s / 194 MB / 11.6 % |
| Mac mini M4 | CLI, ggml CPU, 6 threads, 60 s / 25 % windows (fork default; port `--window-sec 60 --overlap 0.25`) | 13.7 s / 1.7 GB / 5.3 % | 12.3 s / 965 MB / 5.1 % |
|  | CLI, ggml CPU, 6 threads, port default (automatic 300 s window) | - | 27.8 s / 2.4 GB / 7.1 % |
|  | Engine API, ggml CPU, 6 threads, default options (automatic 300 s window on both) | 24.1 s / 6.9 GB / 6.2 % | 27.8 s / 2.4 GB / 7.1 % |
|  | Engine API, ggml CPU, 6 threads, 750-frame windows with 94 frames of context | 12.5 s / 3.0 GB / 5.8 % | 12.3 s / 959 MB / 5.1 % |
|  | CLI, Metal (`--n-gpu-layers 999`), 60 s / 25 % windows | fails (`tdt_greedy_decode` rc=6) | 2.5 s / 482 MB / 5.1 % |
|  | Engine API, Metal, default options | fails (`tdt_greedy_decode` rc=6) | 5.7 s / 524 MB / 6.9 % |
|  | CLI, Core ML GGUF (Neural Engine encoder; fork: q8_0 host decoder, port: ggml decoder) | 986 ms / 210 MB / 7.1 % | 700 ms / 237 MB / 10.5 % |
|  | Engine API, Core ML GGUF (Neural Engine encoder and batched Neural Engine decoder) | 1.0 s / 205 MB / 12.9 % | 952 ms / 202 MB / 11.6 % |

What the numbers say:

- Short clips: the port is 20 to 30 % faster on the CPU (`jfk.wav` 154 ms
  against 192 ms on the Ryzen, 90 against 128 ms on the M3 Ultra, 145 against
  189 ms on the M4) and about twice as fast on CUDA and Metal (8 against 17 ms
  on the RTX 5090, 25 against 61 ms on the M3 Ultra, 46 against 106 ms on the
  M4). Master's encoder graph is 16 to 27 % faster than the fork's, the
  ggml-graph TDT decoder takes 3 to 5 ms where the fork's packed q8_0 host
  decoder takes 9 ms, and the blocked mel takes 1 ms against 4 ms.
- Long recordings at the 60 s / 25 % windows: the port is equal or faster
  (CLI long-mix 12.1 against 13.4 s on the Ryzen, 7.6 against 8.6 s on the M3
  Ultra, 12.3 against 13.7 s on the M4; engine API within 2 % either way;
  16 threads on the M3 Ultra 4.8 against 6.0 s), with about 40 % less peak
  memory through the CLI (1.0 to 1.1 against 1.7 to 1.8 GB) and a third of the
  fork's through the engine API (1.0 to 1.2 against 3.0 to 3.8 GB), a 2 to 3.5
  times faster decoder (112 against 408 ms on the Ryzen) and a 7 to 11 times
  faster mel (19 against 151 ms), at a slightly lower WER (5.1 against 5.3 %
  CLI, 5.1 against 5.8 % engine API; 4.5 to 4.7 against 4.7 to 4.9 % on the
  619 s recording).
- Engine API defaults, the automatic 300 s window on both trees: the port's
  peak memory is 2.9 to 4 times lower (2.6 against 9.7 GB on the Ryzen, 2.4
  against 9.6 GB on the M3 Ultra, 2.4 against 6.9 GB on the M4). The fork is
  4 % faster on the Ryzen (19.8 against 20.6 s), 15 % faster on the M4 (24.1
  against 27.8 s) and 35 % faster on the M3 Ultra (12.8 against 17.3 s), at
  6.2 to 6.3 % against 7.1 % WER. The difference is master's planner,
  unchanged by this PR: it keeps every automatic window at the full 3750
  frames by extending the boundary windows into the input, so the 412.7 s
  recording is encoded as two 300 s windows (600 s of encoder input) where the
  fork's planner shrinks the boundary windows and encodes 280 s plus 174 s.
  The 60 s / 25 % windows sidestep it.
- CLI defaults differ: the fork's CLI defaults to the 60 s / 25 % windows,
  this branch keeps master's automatic window. With the defaults on both
  sides the port's CLI is 1.5 to 2 times slower than the fork's on long
  recordings (20.6 against 13.4 s on the Ryzen, 17.4 against 8.6 s on the M3
  Ultra, 27.8 against 13.7 s on the M4), uses 2.4 to 2.6 GB against 1.7 to
  1.8 GB and loses two WER points; with `--window-sec 60 --overlap 0.25` it is
  ahead on every count. Short clips are unaffected either way.
- GPU: the fork fails on both long recordings on CUDA and on Metal (CLI exit
  code 16; the engine throws `tdt_greedy_decode failed (rc=6)` from the
  on-device joint projection of its decoder); its short clips run. The port
  transcribes long-mix in 207 ms on the RTX 5090 (RTF 0.0005), 876 ms on the
  M3 Ultra and 2.5 s on the M4 with the 60 s windows.
- Core ML distribution (Macs): the encoder sidecar is the same `.mlmodelc`,
  so encoder times match (about 500 ms for long-mix on both Macs). The port's
  CLI is 30 % faster end to end (688 to 700 ms against 986 to 999 ms) through
  the ggml decoder (168 to 190 ms against 368 to 372 ms) and the mel; its
  engine path is 7 to 8 % faster (952 ms to 1.06 s against 1.04 to 1.14 s) at
  the same or lower WER (11.6 against 12.9 % on long-mix, 11.8 against 11.7 %
  on the 619 s recording); `jfk.wav` takes 20 to 22 ms against 27 ms on the
  CLI and 74 to 88 ms on both engine paths, where the batched decoder fills
  one lane. On these mixed-language recordings the port's CLI trails the
  fork's CLI (10.5 against 7.1 % on long-mix, 9.7 against 6.4 % on the 619 s
  recording) while on English it leads it (0.6 against 2.7 % without pauses,
  0.1 against 0.6 % with pauses, next section): both tile the same 15 s
  Neural Engine windows with 16 frames of context on both sides, the fork
  inside its 60 s CLI windows and this branch directly, so their seams fall
  in different places, and what a seam costs depends on what it lands on.

Per-machine tables for `jfk.wav`, long-mix and the 619 s recording; the
fork's failed GPU runs on the long recordings have no row:

**Linux x86-64 (Ubuntu 25.10), AMD Ryzen 9 9950X (10 cores exposed), RTX 5090 32 GB, CUDA 13.3**

CLI (`parakeet --bench`), ggml CPU, 8 threads:

| Clip | Build and path | Mel | Encoder | Decoder | Total | RTF | Peak RSS | WER |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| jfk.wav, 11 s | fork CLI, its default 60 s / 25 % windows | 4 ms | 179 ms | 9 ms | 192 ms | 0.0175 | 764 MB | 0.0 % |
| jfk.wav, 11 s | port CLI, `--window-sec 60 --overlap 0.25` | 1 ms | 150 ms | 3 ms | 154 ms | 0.0140 | 620 MB | 0.0 % |
| jfk.wav, 11 s | port CLI, default (automatic 300 s window) | 1 ms | 152 ms | 3 ms | 156 ms | 0.0142 | 619 MB | 0.0 % |
| long-mix, 412.7 s | fork CLI, its default 60 s / 25 % windows | 151 ms | 12.8 s | 408 ms | 13.4 s | 0.0324 | 1.8 GB | 5.3 % |
| long-mix, 412.7 s | port CLI, `--window-sec 60 --overlap 0.25` | 19 ms | 12.0 s | 112 ms | 12.1 s | 0.0294 | 1.1 GB | 5.1 % |
| long-mix, 412.7 s | port CLI, default (automatic 300 s window) | 20 ms | 20.4 s | 117 ms | 20.6 s | 0.0498 | 2.6 GB | 7.1 % |
| long-10min, 619 s | fork CLI, its default 60 s / 25 % windows | 218 ms | 17.8 s | 576 ms | 18.6 s | 0.0300 | 1.8 GB | 4.7 % |
| long-10min, 619 s | port CLI, `--window-sec 60 --overlap 0.25` | 25 ms | 16.7 s | 165 ms | 16.9 s | 0.0273 | 1.1 GB | 4.7 % |
| long-10min, 619 s | port CLI, default (automatic 300 s window) | 27 ms | 30.0 s | 176 ms | 30.2 s | 0.0488 | 2.6 GB | 7.1 % |
| long-mix, 412.7 s | fork CLI, default windows, 4 threads | 151 ms | 21.7 s | 400 ms | 22.3 s | 0.0540 | 1.8 GB | 5.3 % |
| long-mix, 412.7 s | port CLI, `--window-sec 60 --overlap 0.25`, 4 threads | 17 ms | 21.9 s | 169 ms | 22.1 s | 0.0536 | 1.1 GB | 5.1 % |
| long-mix, 412.7 s | port CLI, default, 4 threads | 18 ms | 36.9 s | 181 ms | 37.1 s | 0.0898 | 2.5 GB | 7.1 % |

Engine API (`bench_engine.cpp` driver), ggml CPU, 8 threads:

| Clip | Build and path | Mel | Encoder | Decoder | Total | RTF | Peak RSS | WER |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| jfk.wav, 11 s | fork `Engine::transcribe`, default options | 4 ms | 181 ms | 9 ms | 195 ms | 0.0177 | 764 MB | 0.0 % |
| jfk.wav, 11 s | port `Engine::transcribe`, default options | 1 ms | 150 ms | 3 ms | 154 ms | 0.0140 | 617 MB | 0.0 % |
| jfk.wav, 11 s | fork engine, 750-frame windows, 94 frames of context | 4 ms | 179 ms | 9 ms | 192 ms | 0.0174 | 764 MB | 0.0 % |
| jfk.wav, 11 s | port engine, 750-frame windows, 94 frames of context | 1 ms | 149 ms | 3 ms | 153 ms | 0.0139 | 617 MB | 0.0 % |
| long-mix, 412.7 s | fork `Engine::transcribe`, default options | 145 ms | 19.3 s | 404 ms | 19.8 s | 0.0481 | 9.7 GB | 6.3 % |
| long-mix, 412.7 s | port `Engine::transcribe`, default options | 15 ms | 20.4 s | 115 ms | 20.6 s | 0.0499 | 2.6 GB | 7.1 % |
| long-mix, 412.7 s | fork engine, 750-frame windows, 94 frames of context | 145 ms | 11.4 s | 381 ms | 12.0 s | 0.0290 | 3.2 GB | 5.8 % |
| long-mix, 412.7 s | port engine, 750-frame windows, 94 frames of context | 17 ms | 12.0 s | 112 ms | 12.2 s | 0.0295 | 1.1 GB | 5.1 % |
| long-10min, 619 s | fork `Engine::transcribe`, default options | 235 ms | 31.7 s | 651 ms | 32.6 s | 0.0526 | 14.7 GB | 6.6 % |
| long-10min, 619 s | port `Engine::transcribe`, default options | 32 ms | 30.1 s | 174 ms | 30.3 s | 0.0490 | 2.6 GB | 7.1 % |
| long-10min, 619 s | fork engine, 750-frame windows, 94 frames of context | 228 ms | 17.2 s | 604 ms | 18.1 s | 0.0292 | 3.7 GB | 4.8 % |
| long-10min, 619 s | port engine, 750-frame windows, 94 frames of context | 25 ms | 16.7 s | 166 ms | 16.9 s | 0.0273 | 1.1 GB | 4.7 % |

CUDA (`--n-gpu-layers 999`), q8_0 GGUF:

| Clip | Build and path | Mel | Encoder | Decoder | Total | RTF | Peak RSS | WER |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| jfk.wav, 11 s | fork CLI, default windows | 4 ms | 10 ms | 3 ms | 17 ms | 0.0015 | 603 MB | 0.0 % |
| jfk.wav, 11 s | port CLI, `--window-sec 60 --overlap 0.25` | 1 ms | 6 ms | 2 ms | 8 ms | 0.0007 | 619 MB | 0.0 % |
| jfk.wav, 11 s | port CLI, default | 1 ms | 6 ms | 2 ms | 8 ms | 0.0007 | 619 MB | 0.0 % |
| jfk.wav, 11 s | fork engine, default options | 4 ms | 10 ms | 3 ms | 17 ms | 0.0015 | 604 MB | 0.0 % |
| jfk.wav, 11 s | port engine, default options | 1 ms | 6 ms | 2 ms | 8 ms | 0.0007 | 619 MB | 0.0 % |
| long-mix, 412.7 s | port CLI, `--window-sec 60 --overlap 0.25` | 16 ms | 131 ms | 60 ms | 207 ms | 0.0005 | 686 MB | 5.3 % |
| long-mix, 412.7 s | port CLI, default | 16 ms | 210 ms | 66 ms | 292 ms | 0.0007 | 716 MB | 6.9 % |
| long-mix, 412.7 s | port engine, default options | 16 ms | 209 ms | 66 ms | 295 ms | 0.0007 | 704 MB | 6.9 % |
| long-10min, 619 s | port CLI, `--window-sec 60 --overlap 0.25` | 23 ms | 181 ms | 93 ms | 297 ms | 0.0005 | 682 MB | 4.8 % |
| long-10min, 619 s | port CLI, default | 30 ms | 309 ms | 96 ms | 436 ms | 0.0007 | 703 MB | 7.1 % |
| long-10min, 619 s | port engine, default options | 34 ms | 309 ms | 96 ms | 445 ms | 0.0007 | 679 MB | 7.1 % |

**Mac Studio M3 Ultra (20 P + 8 E cores, 96 GB), macOS 15.7**

CLI (`parakeet --bench`), ggml CPU, 8 threads:

| Clip | Build and path | Mel | Encoder | Decoder | Total | RTF | Peak RSS | WER |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| jfk.wav, 11 s | fork CLI, its default 60 s / 25 % windows | 3 ms | 116 ms | 9 ms | 128 ms | 0.0116 | 635 MB | 0.0 % |
| jfk.wav, 11 s | port CLI, `--window-sec 60 --overlap 0.25` | 0 ms | 85 ms | 4 ms | 90 ms | 0.0081 | 468 MB | 0.0 % |
| jfk.wav, 11 s | port CLI, default (automatic 300 s window) | 1 ms | 86 ms | 5 ms | 91 ms | 0.0083 | 459 MB | 0.0 % |
| long-mix, 412.7 s | fork CLI, its default 60 s / 25 % windows | 126 ms | 8.1 s | 374 ms | 8.6 s | 0.0209 | 1.7 GB | 5.3 % |
| long-mix, 412.7 s | port CLI, `--window-sec 60 --overlap 0.25` | 11 ms | 7.4 s | 160 ms | 7.6 s | 0.0184 | 1019 MB | 5.1 % |
| long-mix, 412.7 s | port CLI, default (automatic 300 s window) | 10 ms | 17.2 s | 172 ms | 17.4 s | 0.0421 | 2.5 GB | 7.1 % |
| long-10min, 619 s | fork CLI, its default 60 s / 25 % windows | 190 ms | 11.4 s | 569 ms | 12.2 s | 0.0196 | 1.7 GB | 4.9 % |
| long-10min, 619 s | port CLI, `--window-sec 60 --overlap 0.25` | 14 ms | 10.4 s | 243 ms | 10.7 s | 0.0173 | 1.0 GB | 4.5 % |
| long-10min, 619 s | port CLI, default (automatic 300 s window) | 15 ms | 25.5 s | 255 ms | 25.8 s | 0.0417 | 2.5 GB | 7.2 % |
| long-mix, 412.7 s | fork CLI, default windows, 16 threads | 135 ms | 5.5 s | 402 ms | 6.0 s | 0.0145 | 1.7 GB | 5.3 % |
| long-mix, 412.7 s | port CLI, `--window-sec 60 --overlap 0.25`, 16 threads | 10 ms | 4.6 s | 240 ms | 4.8 s | 0.0117 | 998 MB | 5.1 % |
| long-mix, 412.7 s | port CLI, default, 16 threads | 10 ms | 10.9 s | 257 ms | 11.1 s | 0.0270 | 2.5 GB | 7.1 % |

Engine API (`bench_engine.cpp` driver), ggml CPU, 8 threads:

| Clip | Build and path | Mel | Encoder | Decoder | Total | RTF | Peak RSS | WER |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| jfk.wav, 11 s | fork `Engine::transcribe`, default options | 4 ms | 112 ms | 9 ms | 124 ms | 0.0113 | 623 MB | 0.0 % |
| jfk.wav, 11 s | port `Engine::transcribe`, default options | 1 ms | 85 ms | 4 ms | 90 ms | 0.0082 | 461 MB | 0.0 % |
| jfk.wav, 11 s | fork engine, 750-frame windows, 94 frames of context | 4 ms | 113 ms | 9 ms | 126 ms | 0.0114 | 627 MB | 0.0 % |
| jfk.wav, 11 s | port engine, 750-frame windows, 94 frames of context | 0 ms | 86 ms | 4 ms | 90 ms | 0.0082 | 456 MB | 0.0 % |
| long-mix, 412.7 s | fork `Engine::transcribe`, default options | 126 ms | 12.2 s | 417 ms | 12.8 s | 0.0310 | 9.6 GB | 6.2 % |
| long-mix, 412.7 s | port `Engine::transcribe`, default options | 10 ms | 17.1 s | 166 ms | 17.3 s | 0.0419 | 2.4 GB | 7.1 % |
| long-mix, 412.7 s | fork engine, 750-frame windows, 94 frames of context | 128 ms | 7.4 s | 397 ms | 7.9 s | 0.0191 | 3.0 GB | 5.8 % |
| long-mix, 412.7 s | port engine, 750-frame windows, 94 frames of context | 9 ms | 7.5 s | 161 ms | 7.6 s | 0.0185 | 1008 MB | 5.1 % |
| long-10min, 619 s | fork `Engine::transcribe`, default options | 198 ms | 20.2 s | 639 ms | 21.0 s | 0.0340 | 14.6 GB | 6.6 % |
| long-10min, 619 s | port `Engine::transcribe`, default options | 15 ms | 25.5 s | 257 ms | 25.8 s | 0.0417 | 2.5 GB | 7.2 % |
| long-10min, 619 s | fork engine, 750-frame windows, 94 frames of context | 199 ms | 11.0 s | 605 ms | 11.8 s | 0.0190 | 3.6 GB | 4.9 % |
| long-10min, 619 s | port engine, 750-frame windows, 94 frames of context | 14 ms | 10.3 s | 242 ms | 10.6 s | 0.0171 | 1.0 GB | 4.5 % |

Metal (`--n-gpu-layers 999`), q8_0 GGUF:

| Clip | Build and path | Mel | Encoder | Decoder | Total | RTF | Peak RSS | WER |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| jfk.wav, 11 s | fork CLI, default windows | 4 ms | 40 ms | 17 ms | 61 ms | 0.0055 | 393 MB | 0.0 % |
| jfk.wav, 11 s | port CLI, `--window-sec 60 --overlap 0.25` | 1 ms | 18 ms | 6 ms | 25 ms | 0.0023 | 372 MB | 0.0 % |
| jfk.wav, 11 s | port CLI, default | 0 ms | 19 ms | 6 ms | 25 ms | 0.0023 | 373 MB | 0.0 % |
| jfk.wav, 11 s | fork engine, default options | 3 ms | 40 ms | 17 ms | 60 ms | 0.0055 | 394 MB | 0.0 % |
| jfk.wav, 11 s | port engine, default options | 1 ms | 19 ms | 6 ms | 25 ms | 0.0023 | 378 MB | 0.0 % |
| long-mix, 412.7 s | port CLI, `--window-sec 60 --overlap 0.25` | 10 ms | 610 ms | 254 ms | 876 ms | 0.0021 | 495 MB | 5.1 % |
| long-mix, 412.7 s | port CLI, default | 10 ms | 994 ms | 273 ms | 1.3 s | 0.0031 | 541 MB | 6.9 % |
| long-mix, 412.7 s | port engine, default options | 10 ms | 991 ms | 272 ms | 1.3 s | 0.0031 | 583 MB | 6.9 % |
| long-10min, 619 s | port CLI, `--window-sec 60 --overlap 0.25` | 15 ms | 834 ms | 384 ms | 1.2 s | 0.0020 | 525 MB | 4.7 % |
| long-10min, 619 s | port CLI, default | 15 ms | 1.4 s | 410 ms | 1.8 s | 0.0030 | 587 MB | 7.1 % |
| long-10min, 619 s | port engine, default options | 14 ms | 1.4 s | 409 ms | 1.8 s | 0.0030 | 569 MB | 7.1 % |

Core ML distribution (encoder-less GGUF, Neural Engine encoder and decoder sidecars):

| Clip | Build and path | Mel | Encoder | Decoder | Total | RTF | Peak RSS | WER |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| jfk.wav, 11 s | fork CLI, Core ML GGUF (its 1500-frame tiler, q8_0 host decoder) | 3 ms | 15 ms | 8 ms | 27 ms | 0.0025 | 298 MB | 0.0 % |
| jfk.wav, 11 s | port CLI, Core ML GGUF (Neural Engine windows, ggml decoder) | 1 ms | 16 ms | 4 ms | 22 ms | 0.0020 | 140 MB | 0.0 % |
| jfk.wav, 11 s | fork engine, Core ML GGUF | 3 ms | 15 ms | 69 ms | 88 ms | 0.0080 | 294 MB | 0.0 % |
| jfk.wav, 11 s | port engine, Core ML GGUF (pause-cut segments, batched Neural Engine decoder) | 0 ms | 15 ms | 72 ms | 88 ms | 0.0080 | 270 MB | 0.0 % |
| long-mix, 412.7 s | fork CLI, Core ML GGUF (its 1500-frame tiler, q8_0 host decoder) | 126 ms | 504 ms | 368 ms | 999 ms | 0.0024 | 209 MB | 7.1 % |
| long-mix, 412.7 s | port CLI, Core ML GGUF (Neural Engine windows, ggml decoder) | 10 ms | 508 ms | 168 ms | 688 ms | 0.0017 | 238 MB | 10.5 % |
| long-mix, 412.7 s | fork engine, Core ML GGUF | 126 ms | 534 ms | 480 ms | 1.1 s | 0.0028 | 196 MB | 12.9 % |
| long-mix, 412.7 s | port engine, Core ML GGUF (pause-cut segments, batched Neural Engine decoder) | 10 ms | 530 ms | 516 ms | 1.1 s | 0.0026 | 194 MB | 11.6 % |
| long-10min, 619 s | fork CLI, Core ML GGUF (its 1500-frame tiler, q8_0 host decoder) | 189 ms | 747 ms | 556 ms | 1.5 s | 0.0024 | 242 MB | 6.4 % |
| long-10min, 619 s | port CLI, Core ML GGUF (Neural Engine windows, ggml decoder) | 14 ms | 745 ms | 246 ms | 1.0 s | 0.0016 | 284 MB | 9.7 % |
| long-10min, 619 s | fork engine, Core ML GGUF | 188 ms | 799 ms | 613 ms | 1.6 s | 0.0026 | 219 MB | 11.7 % |
| long-10min, 619 s | port engine, Core ML GGUF (pause-cut segments, batched Neural Engine decoder) | 16 ms | 795 ms | 632 ms | 1.4 s | 0.0023 | 226 MB | 11.8 % |

**Mac mini M4 (4 P + 6 E cores, 16 GB), macOS 15.3**

CLI (`parakeet --bench`), ggml CPU, 6 threads:

| Clip | Build and path | Mel | Encoder | Decoder | Total | RTF | Peak RSS | WER |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| jfk.wav, 11 s | fork CLI, its default 60 s / 25 % windows | 4 ms | 176 ms | 9 ms | 189 ms | 0.0171 | 627 MB | 0.0 % |
| jfk.wav, 11 s | port CLI, `--window-sec 60 --overlap 0.25` | 1 ms | 139 ms | 5 ms | 145 ms | 0.0132 | 468 MB | 0.0 % |
| jfk.wav, 11 s | port CLI, default (automatic 300 s window) | 1 ms | 139 ms | 5 ms | 145 ms | 0.0131 | 458 MB | 0.0 % |
| long-mix, 412.7 s | fork CLI, its default 60 s / 25 % windows | 121 ms | 13.2 s | 386 ms | 13.7 s | 0.0332 | 1.7 GB | 5.3 % |
| long-mix, 412.7 s | port CLI, `--window-sec 60 --overlap 0.25` | 18 ms | 12.1 s | 188 ms | 12.3 s | 0.0298 | 965 MB | 5.1 % |
| long-mix, 412.7 s | port CLI, default (automatic 300 s window) | 18 ms | 27.6 s | 201 ms | 27.8 s | 0.0673 | 2.4 GB | 7.1 % |
| long-10min, 619 s | fork CLI, its default 60 s / 25 % windows | 179 ms | 18.7 s | 579 ms | 19.4 s | 0.0313 | 1.7 GB | 4.9 % |
| long-10min, 619 s | port CLI, `--window-sec 60 --overlap 0.25` | 21 ms | 16.9 s | 286 ms | 17.2 s | 0.0278 | 1017 MB | 4.5 % |
| long-10min, 619 s | port CLI, default (automatic 300 s window) | 20 ms | 41.4 s | 303 ms | 41.7 s | 0.0673 | 2.5 GB | 7.2 % |
| long-mix, 412.7 s | fork CLI, default windows, 4 threads | 122 ms | 15.2 s | 390 ms | 15.8 s | 0.0382 | 1.7 GB | 5.3 % |
| long-mix, 412.7 s | port CLI, `--window-sec 60 --overlap 0.25`, 4 threads | 15 ms | 12.3 s | 186 ms | 12.5 s | 0.0303 | 967 MB | 5.1 % |
| long-mix, 412.7 s | port CLI, default, 4 threads | 18 ms | 31.5 s | 203 ms | 31.7 s | 0.0768 | 2.4 GB | 7.1 % |

Engine API (`bench_engine.cpp` driver), ggml CPU, 6 threads:

| Clip | Build and path | Mel | Encoder | Decoder | Total | RTF | Peak RSS | WER |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| jfk.wav, 11 s | fork `Engine::transcribe`, default options | 3 ms | 169 ms | 9 ms | 182 ms | 0.0166 | 621 MB | 0.0 % |
| jfk.wav, 11 s | port `Engine::transcribe`, default options | 0 ms | 147 ms | 5 ms | 152 ms | 0.0138 | 458 MB | 0.0 % |
| jfk.wav, 11 s | fork engine, 750-frame windows, 94 frames of context | 3 ms | 174 ms | 9 ms | 186 ms | 0.0169 | 625 MB | 0.0 % |
| jfk.wav, 11 s | port engine, 750-frame windows, 94 frames of context | 0 ms | 138 ms | 5 ms | 143 ms | 0.0130 | 455 MB | 0.0 % |
| long-mix, 412.7 s | fork `Engine::transcribe`, default options | 121 ms | 23.6 s | 429 ms | 24.1 s | 0.0585 | 6.9 GB | 6.2 % |
| long-mix, 412.7 s | port `Engine::transcribe`, default options | 18 ms | 27.5 s | 200 ms | 27.8 s | 0.0673 | 2.4 GB | 7.1 % |
| long-mix, 412.7 s | fork engine, 750-frame windows, 94 frames of context | 118 ms | 12.0 s | 379 ms | 12.5 s | 0.0302 | 3.0 GB | 5.8 % |
| long-mix, 412.7 s | port engine, 750-frame windows, 94 frames of context | 18 ms | 12.1 s | 199 ms | 12.3 s | 0.0298 | 959 MB | 5.1 % |
| long-10min, 619 s | fork `Engine::transcribe`, default options | 181 ms | 40.0 s | 629 ms | 40.9 s | 0.0660 | 7.0 GB | 6.6 % |
| long-10min, 619 s | port `Engine::transcribe`, default options | 20 ms | 41.5 s | 307 ms | 41.9 s | 0.0676 | 2.4 GB | 7.2 % |
| long-10min, 619 s | fork engine, 750-frame windows, 94 frames of context | 180 ms | 17.8 s | 578 ms | 18.6 s | 0.0300 | 3.6 GB | 4.9 % |
| long-10min, 619 s | port engine, 750-frame windows, 94 frames of context | 26 ms | 16.9 s | 291 ms | 17.2 s | 0.0278 | 991 MB | 4.5 % |

Metal (`--n-gpu-layers 999`), q8_0 GGUF:

| Clip | Build and path | Mel | Encoder | Decoder | Total | RTF | Peak RSS | WER |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| jfk.wav, 11 s | fork CLI, default windows | 3 ms | 84 ms | 19 ms | 106 ms | 0.0097 | 400 MB | 0.0 % |
| jfk.wav, 11 s | port CLI, `--window-sec 60 --overlap 0.25` | 1 ms | 37 ms | 8 ms | 46 ms | 0.0042 | 377 MB | 0.0 % |
| jfk.wav, 11 s | port CLI, default | 1 ms | 37 ms | 8 ms | 46 ms | 0.0042 | 375 MB | 0.0 % |
| jfk.wav, 11 s | fork engine, default options | 4 ms | 84 ms | 19 ms | 106 ms | 0.0097 | 396 MB | 0.0 % |
| jfk.wav, 11 s | port engine, default options | 0 ms | 37 ms | 8 ms | 46 ms | 0.0042 | 379 MB | 0.0 % |
| long-mix, 412.7 s | port CLI, `--window-sec 60 --overlap 0.25` | 17 ms | 2.2 s | 325 ms | 2.5 s | 0.0061 | 482 MB | 5.1 % |
| long-mix, 412.7 s | port CLI, default | 16 ms | 5.3 s | 341 ms | 5.6 s | 0.0137 | 551 MB | 6.9 % |
| long-mix, 412.7 s | port engine, default options | 16 ms | 5.3 s | 340 ms | 5.7 s | 0.0138 | 524 MB | 6.9 % |
| long-10min, 619 s | port CLI, `--window-sec 60 --overlap 0.25` | 27 ms | 3.0 s | 484 ms | 3.5 s | 0.0057 | 518 MB | 4.7 % |
| long-10min, 619 s | port CLI, default | 24 ms | 7.9 s | 513 ms | 8.4 s | 0.0136 | 594 MB | 7.1 % |
| long-10min, 619 s | port engine, default options | 26 ms | 7.9 s | 513 ms | 8.4 s | 0.0136 | 555 MB | 7.1 % |

Core ML distribution (encoder-less GGUF, Neural Engine encoder and decoder sidecars):

| Clip | Build and path | Mel | Encoder | Decoder | Total | RTF | Peak RSS | WER |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| jfk.wav, 11 s | fork CLI, Core ML GGUF (its 1500-frame tiler, q8_0 host decoder) | 3 ms | 14 ms | 9 ms | 27 ms | 0.0025 | 142 MB | 0.0 % |
| jfk.wav, 11 s | port CLI, Core ML GGUF (Neural Engine windows, ggml decoder) | 1 ms | 14 ms | 5 ms | 20 ms | 0.0018 | 136 MB | 0.0 % |
| jfk.wav, 11 s | fork engine, Core ML GGUF | 3 ms | 14 ms | 56 ms | 73 ms | 0.0067 | 294 MB | 0.0 % |
| jfk.wav, 11 s | port engine, Core ML GGUF (pause-cut segments, batched Neural Engine decoder) | 1 ms | 14 ms | 58 ms | 74 ms | 0.0067 | 270 MB | 0.0 % |
| long-mix, 412.7 s | fork CLI, Core ML GGUF (its 1500-frame tiler, q8_0 host decoder) | 118 ms | 496 ms | 372 ms | 986 ms | 0.0024 | 210 MB | 7.1 % |
| long-mix, 412.7 s | port CLI, Core ML GGUF (Neural Engine windows, ggml decoder) | 19 ms | 495 ms | 190 ms | 700 ms | 0.0017 | 237 MB | 10.5 % |
| long-mix, 412.7 s | fork engine, Core ML GGUF | 119 ms | 522 ms | 389 ms | 1.0 s | 0.0025 | 205 MB | 12.9 % |
| long-mix, 412.7 s | port engine, Core ML GGUF (pause-cut segments, batched Neural Engine decoder) | 13 ms | 520 ms | 412 ms | 952 ms | 0.0023 | 202 MB | 11.6 % |
| long-10min, 619 s | fork CLI, Core ML GGUF (its 1500-frame tiler, q8_0 host decoder) | 179 ms | 730 ms | 566 ms | 1.5 s | 0.0024 | 248 MB | 6.4 % |
| long-10min, 619 s | port CLI, Core ML GGUF (Neural Engine windows, ggml decoder) | 26 ms | 728 ms | 272 ms | 1.0 s | 0.0017 | 282 MB | 9.7 % |
| long-10min, 619 s | fork engine, Core ML GGUF | 178 ms | 782 ms | 498 ms | 1.5 s | 0.0024 | 232 MB | 11.7 % |
| long-10min, 619 s | port engine, Core ML GGUF (pause-cut segments, batched Neural Engine decoder) | 26 ms | 779 ms | 517 ms | 1.3 s | 0.0021 | 231 MB | 11.8 % |

### TranscribePsy: accuracy per backend (2026-10)

The same recordings transcribed on every backend of this branch, with the
fork for reference, on the machines of the previous section. The CPU and
Metal rows are from the M4 and the M3 Ultra, which agree with each other and
with the Ryzen to the word; CUDA is the RTX 5090; the Core ML rows are
identical on both Macs. Four recordings were built from the sample clips so
that the reference is exact: `jfk.wav` and `sample-16k.wav` concatenated
twelve times, with and without 0.7 s pauses; `fr-multipiece-16k.wav` thirty
times, with and without pauses; plus the mixed long-mix recording of the
previous sections. WER is against the clips' own single-clip transcripts,
repeated. Time is not shown; it is the previous section's.

| Build, backend and path | English, no pauses, 374 s | English, pauses, 390 s | French, no pauses, 98 s | French, pauses, 119 s | mixed EN+FR, no pauses, 413 s |
|---|--:|--:|--:|--:|--:|
| ggml CPU, 60 s / 25 % windows (CLI `--window-sec 60 --overlap 0.25`) | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.1 % |
| ggml CPU, engine API, 750-frame windows with 94 frames of context | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.1 % |
| ggml CPU, default (automatic 300 s window) | 0.0 % | 0.0 % | 28.6 % | 28.6 % | 7.1 % |
| CUDA, RTX 5090, 60 s / 25 % windows | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.3 % |
| CUDA, RTX 5090, default | 0.0 % | 0.0 % | 28.6 % | 28.6 % | 6.9 % |
| Metal, M4 and M3 Ultra, 60 s / 25 % windows | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.1 % |
| Metal, M4 and M3 Ultra, default | 0.0 % | 0.0 % | 28.6 % | 28.6 % | 6.9 % |
| Core ML, CLI: 15 s Neural Engine windows, 16 frames of seam context, ggml decoder | 0.6 % | 0.1 % | 21.0 % | 29.5 % | 10.5 % |
| Core ML, CLI with `PARAKEET_COREML_ANE_CONTEXT_FRAMES=32` | 0.8 % | 0.0 % | 22.4 % | 27.6 % | 8.4 % |
| Core ML, engine API: pause-cut segments, batched Neural Engine decoder | 0.0 % | 0.0 % | 20.0 % | 27.6 % | 11.6 % |
| Core ML, engine API with `PARAKEET_ANE_BATCH_DECODE=0` (windows, ggml decoder) | 0.6 % | 0.1 % | 21.0 % | 29.5 % | 10.5 % |
| fork, ggml CPU, its default 60 s / 25 % windows | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.3 % |
| fork, CUDA and Metal | fails | fails | 21.9 % | 28.6 % | fails |
| fork, Core ML CLI | 2.7 % | 0.6 % | 21.0 % | 35.7 % | 7.1 % |
| fork, Core ML engine API | 0.1 % | 0.0 % | 21.0 % | 27.1 % | 12.9 % |

What the table says:

- The ggml backends agree. CPU, CUDA and Metal give the same words on every
  recording at the same window settings; CUDA differs by two words on the
  mixed recording.
- The Core ML paths match the ggml backends on English speech. The engine API
  path (pause-cut segments and the Neural Engine decoder) is exact on both
  English recordings. The CLI's fixed 15 s Neural Engine windows lose 0.6 %
  without any pause in 374 s (six words: "thought Alice" dropped where it
  straddles a window seam) and 0.1 % with pauses. The fork's Core ML CLI
  loses 2.7 % and 0.6 % on the same recordings.
- The French rows measure the model, not the backends. The 3.3 s clip's
  seven words come out differently when the clip is repeated than in
  isolation ("ces" for "ses", "rest" for "restent"), on the CPU single pass
  as much as anywhere (28.6 % is two words in seven), and every backend lands
  within one such variant of the others.
- The mixed recording is the one where the Core ML paths trail (10.5 to
  11.6 % against 5.1 to 7.1 %), and it is the only recording that is both
  pause-free and language-switching: the French sentence is transcribed,
  dropped or anglicised depending on what the encoder window around it holds,
  so a path with a seam every 12.4 s (the 15 s export) disturbs it far more
  often than one with a seam every 45 s (the 60 s windows) or none (the 300 s
  window). A sweep of the seam context (16, 24, 32, 40 and 47 frames give
  10.5, 12.3, 8.4, 9.1 and 10.9 %) is not monotonic, which is what seam
  placement on an unstable sentence looks like; on English the same sweep
  moves nothing (0.6 to 0.8 %). This supersedes the earlier reading of that
  gap as a seam-context shortfall.
- `test-transcribepsy` holds every backend to the same bar: a 187 s
  pause-free English recording (the two clips six times) must transcribe
  within 1 % word error with the default options and with the 60 s / 25 %
  windows. It is registered for the ggml CPU, for the GPU backend of the
  build (`test-transcribepsy-gpu`, `--n-gpu-layers 999`) and for the Core ML
  GGUF with its sidecars (`test-transcribepsy-coreml`), and passes on all
  three on the M4 with no word error.

### TranscribePsy: port against the research fork, final pass (2026-10-19)

The port at `2a0d4df1` (with the balanced automatic windows and
TranscribePsy's validated 60 s / 25 % windows as its default) against the
fork at `364ec36`, both built Release with the same flags per machine, on the
four machines available for this pass: the Linux box (Ubuntu 25.10, Ryzen 9
9950X with 10 cores exposed, RTX 5090 on GPU 0, CUDA 13.3, 8 threads and a
4-thread row), a DGX Spark (GB10, 20 cores, CUDA 13.3 with
`CMAKE_CUDA_ARCHITECTURES=121`, 10 threads and a 20-thread row), the Mac mini
M4 (16 GB, macOS 15.3, Metal and the Core ML distribution, 6 threads and a
4-thread row) and a Samsung S25 (SM-S931B, Snapdragon 8 Elite, 8 cores,
12 GB) over adb, both trees cross-compiled with the fork's own Android
recipe (NDK r29, arm64-v8a, CPU only, static; 6 threads and a 4-thread row).
Method and recordings are those of the previous sections: three timed runs
after a warm-up (two on the phone), peak RSS from `/usr/bin/time` or, on
the phone, the engine driver's own `getrusage`, WER against the clips' own
transcripts. The phone throttles under a 40-minute run, so its default and
engine rows were measured again in an interleaved fork/port round with cool-downs before every run, best of two rounds; its 60 s / 25 % CLI
rows are from the first pass, where fork and port ran back to back. The
fork's engine API default was not run on the phone for the 619 s recording:
its automatic 300 s windows take 8.1 GB on the 413 s one, and on the 619 s
one the phone (11.4 GB) rebooted twice, each time during that row.

Total time, port over fork (below 100 %: the port is faster); "fails" is
the fork's crash on long recordings on CUDA and Metal:

| Machine | Backend / path | Audio | Port | Fork | Port / fork |
|---|---|---|--:|--:|--:|
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CPU, CLI, 60 s / 25 % windows | 11 s | 152 ms | 190 ms | 80 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CPU, CLI, 60 s / 25 % windows | 413 s | 12.1 s | 13.2 s | 92 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CPU, CLI, 60 s / 25 % windows | 619 s | 17.0 s | 18.8 s | 90 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CPU, CLI, port default (the model's validated 60 s / 25 % windows) vs fork default | 11 s | 154 ms | 190 ms | 81 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CPU, CLI, port default (the model's validated 60 s / 25 % windows) vs fork default | 413 s | 12.2 s | 13.2 s | 92 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CPU, CLI, port default (the model's validated 60 s / 25 % windows) vs fork default | 619 s | 16.9 s | 18.8 s | 90 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CPU, engine API, default options on both | 11 s | 153 ms | 192 ms | 80 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CPU, engine API, default options on both | 413 s | 12.2 s | 19.9 s | 61 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CPU, engine API, default options on both | 619 s | 16.9 s | 32.6 s | 52 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CPU, engine API, 750-frame windows / 94 context | 11 s | 153 ms | 194 ms | 79 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CPU, engine API, 750-frame windows / 94 context | 413 s | 12.1 s | 12.1 s | 99 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CPU, engine API, 750-frame windows / 94 context | 619 s | 16.8 s | 17.9 s | 94 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CPU, CLI, 60 s / 25 % windows, 4 threads | 413 s | 22.1 s | 22.1 s | 100 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CUDA, CLI, 60 s / 25 % windows | 11 s | 8 ms | 17 ms | 47 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CUDA, CLI, 60 s / 25 % windows | 413 s | 211 ms | fails | - |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CUDA, CLI, 60 s / 25 % windows | 619 s | 296 ms | fails | - |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CUDA, engine API, default options | 11 s | 8 ms | 17 ms | 45 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CUDA, engine API, default options | 413 s | 209 ms | fails | - |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | CUDA, engine API, default options | 619 s | 297 ms | fails | - |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CPU, CLI, 60 s / 25 % windows | 11 s | 181 ms | 234 ms | 77 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CPU, CLI, 60 s / 25 % windows | 413 s | 12.7 s | 14.0 s | 90 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CPU, CLI, 60 s / 25 % windows | 619 s | 17.6 s | 19.8 s | 89 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CPU, CLI, port default (the model's validated 60 s / 25 % windows) vs fork default | 11 s | 178 ms | 234 ms | 76 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CPU, CLI, port default (the model's validated 60 s / 25 % windows) vs fork default | 413 s | 12.6 s | 14.0 s | 90 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CPU, CLI, port default (the model's validated 60 s / 25 % windows) vs fork default | 619 s | 17.7 s | 19.8 s | 90 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CPU, engine API, default options on both | 11 s | 183 ms | 235 ms | 78 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CPU, engine API, default options on both | 413 s | 12.7 s | 17.3 s | 74 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CPU, engine API, default options on both | 619 s | 17.6 s | 27.8 s | 63 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CPU, engine API, 750-frame windows / 94 context | 11 s | 181 ms | 235 ms | 77 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CPU, engine API, 750-frame windows / 94 context | 413 s | 12.7 s | 12.7 s | 100 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CPU, engine API, 750-frame windows / 94 context | 619 s | 17.7 s | 19.2 s | 92 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CPU, CLI, 60 s / 25 % windows, 20 threads | 413 s | 11.3 s | 12.7 s | 89 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CUDA, CLI, 60 s / 25 % windows | 11 s | 15 ms | 33 ms | 46 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CUDA, CLI, 60 s / 25 % windows | 413 s | 604 ms | fails | - |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CUDA, CLI, 60 s / 25 % windows | 619 s | 830 ms | fails | - |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CUDA, engine API, default options | 11 s | 15 ms | 34 ms | 45 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CUDA, engine API, default options | 413 s | 602 ms | fails | - |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | CUDA, engine API, default options | 619 s | 835 ms | fails | - |
| Mac mini M4 (16 GB), 6 threads | CPU, CLI, 60 s / 25 % windows | 11 s | 138 ms | 189 ms | 73 % |
| Mac mini M4 (16 GB), 6 threads | CPU, CLI, 60 s / 25 % windows | 413 s | 11.6 s | 13.7 s | 85 % |
| Mac mini M4 (16 GB), 6 threads | CPU, CLI, 60 s / 25 % windows | 619 s | 16.4 s | 19.4 s | 85 % |
| Mac mini M4 (16 GB), 6 threads | CPU, CLI, port default (the model's validated 60 s / 25 % windows) vs fork default | 11 s | 142 ms | 189 ms | 75 % |
| Mac mini M4 (16 GB), 6 threads | CPU, CLI, port default (the model's validated 60 s / 25 % windows) vs fork default | 413 s | 11.8 s | 13.7 s | 86 % |
| Mac mini M4 (16 GB), 6 threads | CPU, CLI, port default (the model's validated 60 s / 25 % windows) vs fork default | 619 s | 16.4 s | 19.4 s | 85 % |
| Mac mini M4 (16 GB), 6 threads | CPU, engine API, default options on both | 11 s | 140 ms | 182 ms | 77 % |
| Mac mini M4 (16 GB), 6 threads | CPU, engine API, default options on both | 413 s | 11.8 s | 24.1 s | 49 % |
| Mac mini M4 (16 GB), 6 threads | CPU, engine API, default options on both | 619 s | 16.4 s | 40.9 s | 40 % |
| Mac mini M4 (16 GB), 6 threads | CPU, engine API, 750-frame windows / 94 context | 11 s | 140 ms | 186 ms | 75 % |
| Mac mini M4 (16 GB), 6 threads | CPU, engine API, 750-frame windows / 94 context | 413 s | 11.8 s | 12.5 s | 94 % |
| Mac mini M4 (16 GB), 6 threads | CPU, engine API, 750-frame windows / 94 context | 619 s | 16.4 s | 18.6 s | 88 % |
| Mac mini M4 (16 GB), 6 threads | CPU, CLI, 60 s / 25 % windows, 4 threads | 413 s | 12.2 s | 15.8 s | 78 % |
| Mac mini M4 (16 GB), 6 threads | Metal, CLI, 60 s / 25 % windows | 11 s | 48 ms | 106 ms | 45 % |
| Mac mini M4 (16 GB), 6 threads | Metal, CLI, 60 s / 25 % windows | 413 s | 2.6 s | fails | - |
| Mac mini M4 (16 GB), 6 threads | Metal, CLI, 60 s / 25 % windows | 619 s | 3.6 s | fails | - |
| Mac mini M4 (16 GB), 6 threads | Metal, engine API, default options | 11 s | 48 ms | 106 ms | 45 % |
| Mac mini M4 (16 GB), 6 threads | Metal, engine API, default options | 413 s | 2.5 s | fails | - |
| Mac mini M4 (16 GB), 6 threads | Metal, engine API, default options | 619 s | 3.6 s | fails | - |
| Mac mini M4 (16 GB), 6 threads | Core ML (Neural Engine), CLI | 11 s | 20 ms | 27 ms | 73 % |
| Mac mini M4 (16 GB), 6 threads | Core ML (Neural Engine), CLI | 413 s | 689 ms | 986 ms | 70 % |
| Mac mini M4 (16 GB), 6 threads | Core ML (Neural Engine), CLI | 619 s | 1.0 s | 1.5 s | 68 % |
| Mac mini M4 (16 GB), 6 threads | Core ML (Neural Engine), engine API | 11 s | 76 ms | 77 ms | 99 % |
| Mac mini M4 (16 GB), 6 threads | Core ML (Neural Engine), engine API | 413 s | 955 ms | 1.0 s | 92 % |
| Mac mini M4 (16 GB), 6 threads | Core ML (Neural Engine), engine API | 619 s | 1.3 s | 1.5 s | 91 % |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads | CPU, CLI, 60 s / 25 % windows | 11 s | 171 ms | 199 ms | 86 % |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads | CPU, CLI, 60 s / 25 % windows | 413 s | 21.8 s | 22.3 s | 98 % |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads | CPU, CLI, 60 s / 25 % windows | 619 s | 38.9 s | 40.0 s | 97 % |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads | CPU, CLI, 60 s / 25 % windows, 4 threads | 413 s | 36.0 s | 40.4 s | 89 % |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads, best of two interleaved rounds | CPU, CLI, port default (the model's validated 60 s / 25 % windows) vs fork default | 11 s | 169 ms | 204 ms | 83 % |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads, best of two interleaved rounds | CPU, CLI, port default (the model's validated 60 s / 25 % windows) vs fork default | 413 s | 25.6 s | 26.1 s | 98 % |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads, best of two interleaved rounds | CPU, CLI, port default (the model's validated 60 s / 25 % windows) vs fork default | 619 s | 41.1 s | 46.8 s | 88 % |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads, best of two interleaved rounds | CPU, engine API, default options on both | 11 s | 155 ms | 202 ms | 77 % |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads, best of two interleaved rounds | CPU, engine API, default options on both | 413 s | 23.1 s | 77.1 s | 30 % |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads, best of two interleaved rounds | CPU, engine API, default options on both | 619 s | 40.7 s | reboots the phone (out of memory) | - |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads, best of two interleaved rounds | CPU, engine API, 750-frame windows / 94 context | 11 s | 167 ms | 202 ms | 83 % |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads, best of two interleaved rounds | CPU, engine API, 750-frame windows / 94 context | 413 s | 26.5 s | 26.5 s | 100 % |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads, best of two interleaved rounds | CPU, engine API, 750-frame windows / 94 context | 619 s | 40.7 s | 47.0 s | 87 % |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads, one run | CPU, CLI, port default (the model's validated 60 s / 25 % windows) vs fork default | 2063 s | 135.7 s | 165.7 s | 82 % |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads, one run | CPU, engine API, port default vs the fork's 750-frame windows (its default exceeds the phone's memory) | 2063 s | 134.2 s | 165.0 s | 81 % |
| Samsung S25, Adreno 830 OpenCL, 6 CPU threads | OpenCL, CLI, port default (60 s / 25 % windows) vs fork default | 11 s | 255 ms | 327 ms | 78 % |
| Samsung S25, Adreno 830 OpenCL, 6 CPU threads | OpenCL, CLI, port default (60 s / 25 % windows) vs fork default | 413 s | 16.1 s | 18.6 s | 87 % |
| Samsung S25, Adreno 830 OpenCL, 6 CPU threads | OpenCL, CLI, port default (60 s / 25 % windows) vs fork default | 619 s | 27.5 s | 31.1 s | 88 % |
| Samsung S25, Adreno 830 OpenCL, 6 CPU threads | OpenCL, engine API, port default vs fork default (fork: 11 s only, its default exceeds the phone's memory on long recordings) | 11 s | 252 ms | 337 ms | 75 % |
| Samsung S25, Adreno 830 OpenCL, 6 CPU threads | OpenCL, engine API, port default vs fork default (fork: 11 s only, its default exceeds the phone's memory on long recordings) | 413 s | 16.7 s | not run | - |
| Samsung S25, Adreno 830 OpenCL, 6 CPU threads | OpenCL, engine API, port default vs fork default (fork: 11 s only, its default exceeds the phone's memory on long recordings) | 619 s | 28.6 s | not run | - |
| Samsung S25, Adreno 830 OpenCL, 6 CPU threads | OpenCL, engine API, port default vs the fork's 750-frame windows | 11 s | 252 ms | 343 ms | 74 % |
| Samsung S25, Adreno 830 OpenCL, 6 CPU threads | OpenCL, engine API, port default vs the fork's 750-frame windows | 413 s | 16.7 s | 19.4 s | 86 % |
| Samsung S25, Adreno 830 OpenCL, 6 CPU threads | OpenCL, engine API, port default vs the fork's 750-frame windows | 619 s | 28.6 s | 35.8 s | 80 % |

What changed since the first pass and what the table says:

- The port's CLI without flags and the engine API with default options now
  plan TranscribePsy's validated 60 s / 25 % windows, so its default rows
  equal its `--window-sec 60 --overlap 0.25` rows: 8 to 15 % faster than the
  fork's CLI default on long recordings on the Linux box, the Spark and the
  M4, and 20 to 27 % faster on short clips. Against the fork's engine API
  default (its automatic 300 s window) the port is 1.3 to 2.5 times faster
  on long recordings.
- The engine API at the same 750-frame windows is equal on the 413 s
  recording (99 to 100 %) and 6 to 12 % faster on the 619 s one.
- CUDA and Metal: twice as fast on short clips; the fork fails on the long
  recordings, the port transcribes them (RTF 0.0005 on the RTX 5090, 0.0015
  on the GB10, 0.006 on the M4).
- Core ML on the M4: the CLI is 27 to 32 % faster, the engine path 8 to 9 %
  faster on long recordings and equal on `jfk.wav` (76 against 77 ms, ten
  runs), where the single-lane batched decoder dominates on both.
- Samsung S25, CPU only, best of two interleaved rounds: port default
  against fork default 83 %, 98 %, 88 % (11 s, 413 s, 619 s); engine API
  defaults 77 %, 30 %, -; engine API at 750-frame windows 83 %, 100 %, 87 %.
  The phone throttles, so the first pass had put the port's re-measured
  rows at a disadvantage; the interleaved rounds remove it.
- Samsung S25 on its Adreno 830 through OpenCL (both trees cross-compiled
  with the OpenCL backend against the Khronos ICD loader, the phone's own
  `libOpenCL.so` at run time; encoder on the GPU): port default against
  fork default on the CLI 78 %, 87 %, 88 % (11 s, 413 s, 619 s), engine
  API against the fork's 750-frame windows 74 %, 86 %, 80 %. Neither tree
  can run the decoder graphs on ggml-opencl (it drops the in-place state
  writes); the fork falls back to its packed q8_0 host decoder and the port
  now runs its ggml-graph decoder on the CPU backend from a CPU copy of the
  decoder weights (`c86079ec`), which decodes the 413 s recording in half a
  second where the scalar host fallback it replaced took two.
- Accuracy on that OpenCL path is wrong for long recordings, in both trees
  alike: single-window inputs are correct (`jfk.wav`, `sample-16k.wav`, and
  two different 11 s clips transcribed one after the other by one engine),
  but every recording that needs more than one encoder window loses most
  of its words from the second window on (fork and port both 36 % WER on
  long-mix, 54 % on the pause-free English recording, where the CPU is at
  0 %; the two trees' outputs are word-for-word the same). The defect is in
  the shared ggml-opencl backend on this device, not in either tree's
  windowing: the same code is exact on the CPU, CUDA and Metal; it is
  deterministic and survived disabling the port's graph reuse, copying each
  window's mel into its own buffer, and every backend switch
  (`GGML_OPENCL_FLUSH_WORK_MB`, `GGML_OPENCL_FA_MAX_NQ`,
  `GGML_OPENCL_FA_ADRENO`, `GGML_OPENCL_ADRENO_USE_LARGE_BUFFER`); it
  worsens with the window size (25 s windows: 16 % WER on the 374 s file,
  60 s: 54 %, 120 s: everything after the first window is empty) while a
  62 s single pass is exact. Reproduction: `en-nopause-2x.wav` (jfk and
  sample-16k twice, 62 s) with `--window-sec 60 --overlap 0.25` on the
  Adreno 830 gives 104 words instead of 158 on both trees. Until the backend
  is fixed, recordings longer than one window must not be transcribed on
  Adreno OpenCL; the OpenCL speed rows above stand as measurements of the
  work done, not of a usable result on long audio.

Peak memory (engine API, the path the addon uses; the engine driver's own
`getrusage` on the phone, `/usr/bin/time` on the desktops):

| Build and path | S25 11 s | S25 413 s | S25 619 s | S25 2063 s | M4 413 s | M4 619 s | Ryzen 413 s | Ryzen 619 s |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| port, engine API, default (TranscribePsy's 60 s / 25 % windows) | 0.4 GB | 0.9 GB | 0.9 GB | 1.0 GB | 0.9 GB | 1.0 GB | 1.1 GB | 1.1 GB |
| port, engine API, 750-frame windows / 94 context | 0.4 GB | 0.9 GB | 0.9 GB | - | 0.9 GB | 1.0 GB | 1.1 GB | 1.1 GB |
| fork, engine API, 750-frame windows / 94 context | 0.5 GB | 2.9 GB | 3.5 GB | 3.7 GB | 3.0 GB | 3.6 GB | 3.2 GB | 3.7 GB |
| fork, engine API, default (automatic 300 s windows) | 0.5 GB | 8.1 GB | reboots the phone (> 11.4 GB) | not run (would exceed 11.4 GB) | 6.9 GB | 7.0 GB | 9.7 GB | 14.7 GB |
| port, engine API on OpenCL, default (process RSS; OpenCL buffers are not in it) | 0.1 GB | 0.2 GB | 0.2 GB | - | | | | |
| fork, engine API on OpenCL, 750-frame windows (process RSS; OpenCL buffers are not in it) | 0.1 GB | 0.2 GB | 0.2 GB | - | | | | |

The port's memory does not grow with the recording: 0.9 GB on the phone from
413 s to 34 minutes, because a windowed pass holds one encoder graph, the
60 s window bounds the attention buffers, and the waveform and STFT scratch
are released once the mel exists. The fork needs three to four times as
much at the same 60 s windows (it keeps three graphs and the full scratch)
and, at its engine default, 8 GB at 413 s and more than the phone has at
619 s. The memory the addon will see is the port's default row.

Accuracy per backend on this pass (same recordings and references as the
previous section; the port's default rows now coincide with its 60 s rows):

| Machine | Build, backend and path | EN no pauses 374 s | EN pauses 390 s | FR no pauses 98 s | FR pauses 119 s | EN+FR no pauses 413 s |
|---|---|--:|--:|--:|--:|--:|
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | port, CPU, 60 s / 25 % windows | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.1 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | port, CPU, default | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.1 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | fork, CPU, its default 60 s / 25 % windows | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.3 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | port, CUDA, 60 s / 25 % windows | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.3 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | port, CUDA, default | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.3 % |
| Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads | fork, CUDA | fails | fails | 21.9 % | 28.6 % | fails |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | port, CPU, 60 s / 25 % windows | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.3 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | port, CPU, default | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.3 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | fork, CPU, its default 60 s / 25 % windows | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.3 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | port, CUDA, 60 s / 25 % windows | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 4.9 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | port, CUDA, default | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 4.9 % |
| DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads | fork, CUDA | fails | fails | 21.9 % | 28.6 % | fails |
| Mac mini M4 (16 GB), 6 threads | port, CPU, 60 s / 25 % windows | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.1 % |
| Mac mini M4 (16 GB), 6 threads | port, CPU, default | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.1 % |
| Mac mini M4 (16 GB), 6 threads | fork, CPU, its default 60 s / 25 % windows | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.3 % |
| Mac mini M4 (16 GB), 6 threads | port, Metal, 60 s / 25 % windows | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.1 % |
| Mac mini M4 (16 GB), 6 threads | port, Metal, default | 0.0 % | 0.2 % | 21.9 % | 28.6 % | 5.1 % |
| Mac mini M4 (16 GB), 6 threads | fork, Metal | fails | fails | 21.9 % | 28.6 % | fails |
| Mac mini M4 (16 GB), 6 threads | port, Core ML, CLI windows | 0.6 % | 0.1 % | 21.0 % | 29.5 % | 10.5 % |
| Mac mini M4 (16 GB), 6 threads | port, Core ML, engine API | 0.0 % | 0.0 % | 20.0 % | 27.6 % | 11.6 % |
| Mac mini M4 (16 GB), 6 threads | fork, Core ML, CLI | 2.7 % | 0.6 % | 21.0 % | 35.7 % | 7.1 % |
| Mac mini M4 (16 GB), 6 threads | fork, Core ML, engine API | 0.1 % | 0.0 % | 21.0 % | 27.1 % | 12.9 % |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads | port, CPU, 60 s / 25 % windows | 0.0 % | 0.2 % | 21.9 % | 28.6 % | - |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads | port, CPU, default | 0.0 % | 0.2 % | 21.9 % | 28.6 % | - |
| Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads | fork, CPU, its default 60 s / 25 % windows | 0.0 % | 0.2 % | 21.9 % | 28.6 % | - |
| Samsung S25, Adreno 830 OpenCL, 6 CPU threads | port, OpenCL, default (60 s / 25 % windows) | 54.4 % | 65.8 % | 21.9 % | 28.6 % | - |
| Samsung S25, Adreno 830 OpenCL, 6 CPU threads | fork, OpenCL, its default 60 s / 25 % windows | 54.4 % | 65.8 % | 21.9 % | 28.6 % | - |

Per-machine detail (`jfk.wav`, long-mix and the 619 s recording):
**Linux, Ryzen 9 9950X + RTX 5090 (GPU0), 8 threads**

| Clip | Build and path | Mel | Encoder | Decoder | Total | RTF | Peak RSS | WER |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| 11 s | fork CLI, default 60 s / 25 % windows | 4 ms | 177 ms | 9 ms | 190 ms | 0.0173 | 765 MB | 0.0 % |
| 11 s | port CLI, `--window-sec 60 --overlap 0.25` | 1 ms | 147 ms | 3 ms | 152 ms | 0.0138 | 620 MB | 0.0 % |
| 11 s | port CLI, default (balanced automatic windows) | 1 ms | 150 ms | 3 ms | 154 ms | 0.0140 | 619 MB | 0.0 % |
| 11 s | fork engine API, default options | 4 ms | 179 ms | 9 ms | 192 ms | 0.0175 | 765 MB | 0.0 % |
| 11 s | port engine API, default options | 1 ms | 149 ms | 3 ms | 153 ms | 0.0140 | 617 MB | 0.0 % |
| 11 s | fork engine API, 750-frame windows, 94 context | 4 ms | 181 ms | 9 ms | 194 ms | 0.0177 | 764 MB | 0.0 % |
| 11 s | port engine API, 750-frame windows, 94 context | 1 ms | 149 ms | 3 ms | 153 ms | 0.0139 | 617 MB | 0.0 % |
| 11 s | fork CLI, CUDA | 4 ms | 10 ms | 3 ms | 17 ms | 0.0015 | 603 MB | 0.0 % |
| 11 s | port CLI, CUDA, 60 s / 25 % windows | 1 ms | 6 ms | 2 ms | 8 ms | 0.0007 | 619 MB | 0.0 % |
| 11 s | port CLI, CUDA, default | 1 ms | 6 ms | 1 ms | 8 ms | 0.0007 | 619 MB | 0.0 % |
| 11 s | fork engine API, CUDA | 4 ms | 10 ms | 3 ms | 17 ms | 0.0016 | 604 MB | 0.0 % |
| 11 s | port engine API, CUDA, default | 1 ms | 6 ms | 2 ms | 8 ms | 0.0007 | 619 MB | 0.0 % |
| 413 s | fork CLI, default 60 s / 25 % windows | 144 ms | 12.7 s | 388 ms | 13.2 s | 0.0320 | 1.8 GB | 5.3 % |
| 413 s | port CLI, `--window-sec 60 --overlap 0.25` | 19 ms | 12.0 s | 111 ms | 12.1 s | 0.0293 | 1.1 GB | 5.1 % |
| 413 s | port CLI, default (balanced automatic windows) | 19 ms | 12.0 s | 111 ms | 12.2 s | 0.0295 | 1.1 GB | 5.1 % |
| 413 s | fork engine API, default options | 151 ms | 19.3 s | 421 ms | 19.9 s | 0.0483 | 9.7 GB | 6.3 % |
| 413 s | port engine API, default options | 16 ms | 12.0 s | 112 ms | 12.2 s | 0.0295 | 1.1 GB | 5.1 % |
| 413 s | fork engine API, 750-frame windows, 94 context | 152 ms | 11.6 s | 397 ms | 12.1 s | 0.0294 | 3.2 GB | 5.8 % |
| 413 s | port engine API, 750-frame windows, 94 context | 15 ms | 11.9 s | 110 ms | 12.1 s | 0.0293 | 1.1 GB | 5.1 % |
| 413 s | fork CLI, CUDA | fails | | | | | | |
| 413 s | port CLI, CUDA, 60 s / 25 % windows | 17 ms | 132 ms | 62 ms | 211 ms | 0.0005 | 679 MB | 5.3 % |
| 413 s | port CLI, CUDA, default | 16 ms | 132 ms | 62 ms | 210 ms | 0.0005 | 681 MB | 5.3 % |
| 413 s | fork engine API, CUDA | fails | | | | | | |
| 413 s | port engine API, CUDA, default | 16 ms | 132 ms | 60 ms | 209 ms | 0.0005 | 676 MB | 5.3 % |
| 619 s | fork CLI, default 60 s / 25 % windows | 227 ms | 18.0 s | 595 ms | 18.8 s | 0.0303 | 1.8 GB | 4.7 % |
| 619 s | port CLI, `--window-sec 60 --overlap 0.25` | 29 ms | 16.8 s | 166 ms | 17.0 s | 0.0274 | 1.1 GB | 4.7 % |
| 619 s | port CLI, default (balanced automatic windows) | 28 ms | 16.7 s | 167 ms | 16.9 s | 0.0273 | 1.1 GB | 4.7 % |
| 619 s | fork engine API, default options | 235 ms | 31.7 s | 646 ms | 32.6 s | 0.0526 | 14.7 GB | 6.6 % |
| 619 s | port engine API, default options | 24 ms | 16.7 s | 166 ms | 16.9 s | 0.0273 | 1.1 GB | 4.7 % |
| 619 s | fork engine API, 750-frame windows, 94 context | 217 ms | 17.1 s | 580 ms | 17.9 s | 0.0290 | 3.7 GB | 4.8 % |
| 619 s | port engine API, 750-frame windows, 94 context | 28 ms | 16.6 s | 166 ms | 16.8 s | 0.0271 | 1.1 GB | 4.7 % |
| 619 s | fork CLI, CUDA | fails | | | | | | |
| 619 s | port CLI, CUDA, 60 s / 25 % windows | 23 ms | 183 ms | 90 ms | 296 ms | 0.0005 | 682 MB | 4.8 % |
| 619 s | port CLI, CUDA, default | 23 ms | 183 ms | 93 ms | 299 ms | 0.0005 | 682 MB | 4.8 % |
| 619 s | fork engine API, CUDA | fails | | | | | | |
| 619 s | port engine API, CUDA, default | 23 ms | 183 ms | 91 ms | 297 ms | 0.0005 | 675 MB | 4.8 % |

**DGX Spark, GB10 (20 cores, CUDA 13.3), 10 threads**

| Clip | Build and path | Mel | Encoder | Decoder | Total | RTF | Peak RSS | WER |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| 11 s | fork CLI, default 60 s / 25 % windows | 4 ms | 211 ms | 18 ms | 234 ms | 0.0213 | 846 MB | 0.0 % |
| 11 s | port CLI, `--window-sec 60 --overlap 0.25` | 2 ms | 175 ms | 4 ms | 181 ms | 0.0164 | 701 MB | 0.0 % |
| 11 s | port CLI, default (balanced automatic windows) | 2 ms | 173 ms | 4 ms | 178 ms | 0.0162 | 701 MB | 0.0 % |
| 11 s | fork engine API, default options | 4 ms | 212 ms | 19 ms | 235 ms | 0.0214 | 846 MB | 0.0 % |
| 11 s | port engine API, default options | 2 ms | 178 ms | 4 ms | 183 ms | 0.0166 | 699 MB | 0.0 % |
| 11 s | fork engine API, 750-frame windows, 94 context | 4 ms | 213 ms | 18 ms | 235 ms | 0.0214 | 846 MB | 0.0 % |
| 11 s | port engine API, 750-frame windows, 94 context | 2 ms | 176 ms | 4 ms | 181 ms | 0.0165 | 699 MB | 0.0 % |
| 11 s | fork CLI, CUDA | 4 ms | 24 ms | 5 ms | 33 ms | 0.0030 | 560 MB | 0.0 % |
| 11 s | port CLI, CUDA, 60 s / 25 % windows | 1 ms | 12 ms | 2 ms | 15 ms | 0.0014 | 569 MB | 0.0 % |
| 11 s | port CLI, CUDA, default | 1 ms | 11 ms | 2 ms | 15 ms | 0.0013 | 569 MB | 0.0 % |
| 11 s | fork engine API, CUDA | 4 ms | 24 ms | 5 ms | 34 ms | 0.0031 | 560 MB | 0.0 % |
| 11 s | port engine API, CUDA, default | 1 ms | 11 ms | 2 ms | 15 ms | 0.0014 | 570 MB | 0.0 % |
| 413 s | fork CLI, default 60 s / 25 % windows | 150 ms | 13.1 s | 766 ms | 14.0 s | 0.0340 | 1.9 GB | 5.3 % |
| 413 s | port CLI, `--window-sec 60 --overlap 0.25` | 25 ms | 12.5 s | 139 ms | 12.7 s | 0.0308 | 1.2 GB | 5.3 % |
| 413 s | port CLI, default (balanced automatic windows) | 24 ms | 12.5 s | 138 ms | 12.6 s | 0.0306 | 1.2 GB | 5.3 % |
| 413 s | fork engine API, default options | 151 ms | 16.3 s | 826 ms | 17.3 s | 0.0419 | 9.8 GB | 6.2 % |
| 413 s | port engine API, default options | 24 ms | 12.6 s | 138 ms | 12.7 s | 0.0308 | 1.2 GB | 5.3 % |
| 413 s | fork engine API, 750-frame windows, 94 context | 151 ms | 11.9 s | 696 ms | 12.7 s | 0.0309 | 3.3 GB | 5.8 % |
| 413 s | port engine API, 750-frame windows, 94 context | 25 ms | 12.6 s | 143 ms | 12.7 s | 0.0309 | 1.2 GB | 5.3 % |
| 413 s | fork CLI, CUDA | fails | | | | | | |
| 413 s | port CLI, CUDA, 60 s / 25 % windows | 25 ms | 483 ms | 96 ms | 604 ms | 0.0015 | 637 MB | 4.9 % |
| 413 s | port CLI, CUDA, default | 24 ms | 479 ms | 96 ms | 600 ms | 0.0015 | 637 MB | 4.9 % |
| 413 s | fork engine API, CUDA | fails | | | | | | |
| 413 s | port engine API, CUDA, default | 24 ms | 478 ms | 96 ms | 602 ms | 0.0015 | 651 MB | 4.9 % |
| 619 s | fork CLI, default 60 s / 25 % windows | 227 ms | 18.5 s | 1.1 s | 19.8 s | 0.0319 | 1.9 GB | 4.8 % |
| 619 s | port CLI, `--window-sec 60 --overlap 0.25` | 36 ms | 17.3 s | 210 ms | 17.6 s | 0.0284 | 1.2 GB | 4.7 % |
| 619 s | port CLI, default (balanced automatic windows) | 36 ms | 17.5 s | 214 ms | 17.7 s | 0.0286 | 1.2 GB | 4.7 % |
| 619 s | fork engine API, default options | 234 ms | 26.5 s | 1.1 s | 27.8 s | 0.0450 | 14.8 GB | 6.6 % |
| 619 s | port engine API, default options | 37 ms | 17.4 s | 209 ms | 17.6 s | 0.0285 | 1.2 GB | 4.7 % |
| 619 s | fork engine API, 750-frame windows, 94 context | 230 ms | 17.8 s | 1.2 s | 19.2 s | 0.0311 | 3.8 GB | 4.9 % |
| 619 s | port engine API, 750-frame windows, 94 context | 38 ms | 17.4 s | 216 ms | 17.7 s | 0.0285 | 1.2 GB | 4.7 % |
| 619 s | fork CLI, CUDA | fails | | | | | | |
| 619 s | port CLI, CUDA, 60 s / 25 % windows | 38 ms | 648 ms | 145 ms | 830 ms | 0.0013 | 661 MB | 4.4 % |
| 619 s | port CLI, CUDA, default | 37 ms | 647 ms | 145 ms | 829 ms | 0.0013 | 661 MB | 4.4 % |
| 619 s | fork engine API, CUDA | fails | | | | | | |
| 619 s | port engine API, CUDA, default | 37 ms | 649 ms | 145 ms | 835 ms | 0.0013 | 693 MB | 4.4 % |

**Mac mini M4 (16 GB), 6 threads**

| Clip | Build and path | Mel | Encoder | Decoder | Total | RTF | Peak RSS | WER |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| 11 s | fork CLI, default 60 s / 25 % windows | 4 ms | 176 ms | 9 ms | 189 ms | 0.0171 | 627 MB | 0.0 % |
| 11 s | port CLI, `--window-sec 60 --overlap 0.25` | 1 ms | 133 ms | 5 ms | 138 ms | 0.0126 | 458 MB | 0.0 % |
| 11 s | port CLI, default (balanced automatic windows) | 1 ms | 137 ms | 5 ms | 142 ms | 0.0129 | 457 MB | 0.0 % |
| 11 s | fork engine API, default options | 3 ms | 169 ms | 9 ms | 182 ms | 0.0166 | 621 MB | 0.0 % |
| 11 s | port engine API, default options | 0 ms | 135 ms | 5 ms | 140 ms | 0.0127 | 459 MB | 0.0 % |
| 11 s | fork engine API, 750-frame windows, 94 context | 3 ms | 174 ms | 9 ms | 186 ms | 0.0169 | 625 MB | 0.0 % |
| 11 s | port engine API, 750-frame windows, 94 context | 0 ms | 135 ms | 5 ms | 140 ms | 0.0127 | 461 MB | 0.0 % |
| 11 s | fork CLI, Metal | 3 ms | 84 ms | 19 ms | 106 ms | 0.0097 | 400 MB | 0.0 % |
| 11 s | port CLI, Metal, 60 s / 25 % windows | 0 ms | 39 ms | 9 ms | 48 ms | 0.0043 | 370 MB | 0.0 % |
| 11 s | port CLI, Metal, default | 0 ms | 39 ms | 9 ms | 48 ms | 0.0044 | 375 MB | 0.0 % |
| 11 s | fork engine API, Metal | 4 ms | 84 ms | 19 ms | 106 ms | 0.0097 | 396 MB | 0.0 % |
| 11 s | port engine API, Metal, default | 0 ms | 39 ms | 9 ms | 48 ms | 0.0043 | 375 MB | 0.0 % |
| 11 s | fork CLI, Core ML | 3 ms | 14 ms | 9 ms | 27 ms | 0.0025 | 142 MB | 0.0 % |
| 11 s | port CLI, Core ML | 0 ms | 14 ms | 5 ms | 20 ms | 0.0018 | 139 MB | 0.0 % |
| 11 s | fork engine API, Core ML | 3 ms | 14 ms | 59 ms | 77 ms | 0.0070 | 147 MB | 0.0 % |
| 11 s | port engine API, Core ML | 1 ms | 14 ms | 61 ms | 76 ms | 0.0069 | 152 MB | 0.0 % |
| 413 s | fork CLI, default 60 s / 25 % windows | 121 ms | 13.2 s | 386 ms | 13.7 s | 0.0332 | 1.7 GB | 5.3 % |
| 413 s | port CLI, `--window-sec 60 --overlap 0.25` | 13 ms | 11.5 s | 175 ms | 11.6 s | 0.0282 | 968 MB | 5.1 % |
| 413 s | port CLI, default (balanced automatic windows) | 13 ms | 11.6 s | 177 ms | 11.8 s | 0.0285 | 972 MB | 5.1 % |
| 413 s | fork engine API, default options | 121 ms | 23.6 s | 429 ms | 24.1 s | 0.0585 | 6.9 GB | 6.2 % |
| 413 s | port engine API, default options | 12 ms | 11.6 s | 179 ms | 11.8 s | 0.0285 | 951 MB | 5.1 % |
| 413 s | fork engine API, 750-frame windows, 94 context | 118 ms | 12.0 s | 379 ms | 12.5 s | 0.0302 | 3.0 GB | 5.8 % |
| 413 s | port engine API, 750-frame windows, 94 context | 12 ms | 11.6 s | 177 ms | 11.8 s | 0.0285 | 951 MB | 5.1 % |
| 413 s | fork CLI, Metal | fails | | | | | | |
| 413 s | port CLI, Metal, 60 s / 25 % windows | 13 ms | 2.2 s | 344 ms | 2.6 s | 0.0062 | 486 MB | 5.1 % |
| 413 s | port CLI, Metal, default | 13 ms | 2.2 s | 345 ms | 2.6 s | 0.0062 | 481 MB | 5.1 % |
| 413 s | fork engine API, Metal | fails | | | | | | |
| 413 s | port engine API, Metal, default | 12 ms | 2.2 s | 345 ms | 2.5 s | 0.0062 | 474 MB | 5.1 % |
| 413 s | fork CLI, Core ML | 118 ms | 496 ms | 372 ms | 986 ms | 0.0024 | 210 MB | 7.1 % |
| 413 s | port CLI, Core ML | 13 ms | 493 ms | 181 ms | 689 ms | 0.0017 | 236 MB | 10.5 % |
| 413 s | fork engine API, Core ML | 119 ms | 522 ms | 389 ms | 1.0 s | 0.0025 | 205 MB | 12.9 % |
| 413 s | port engine API, Core ML | 12 ms | 513 ms | 428 ms | 955 ms | 0.0023 | 201 MB | 11.6 % |
| 619 s | fork CLI, default 60 s / 25 % windows | 179 ms | 18.7 s | 579 ms | 19.4 s | 0.0313 | 1.7 GB | 4.9 % |
| 619 s | port CLI, `--window-sec 60 --overlap 0.25` | 18 ms | 16.2 s | 268 ms | 16.4 s | 0.0266 | 1010 MB | 4.5 % |
| 619 s | port CLI, default (balanced automatic windows) | 18 ms | 16.2 s | 269 ms | 16.4 s | 0.0266 | 1014 MB | 4.5 % |
| 619 s | fork engine API, default options | 181 ms | 40.0 s | 629 ms | 40.9 s | 0.0660 | 7.0 GB | 6.6 % |
| 619 s | port engine API, default options | 18 ms | 16.1 s | 267 ms | 16.4 s | 0.0266 | 990 MB | 4.5 % |
| 619 s | fork engine API, 750-frame windows, 94 context | 180 ms | 17.8 s | 578 ms | 18.6 s | 0.0300 | 3.6 GB | 4.9 % |
| 619 s | port engine API, 750-frame windows, 94 context | 19 ms | 16.1 s | 267 ms | 16.4 s | 0.0265 | 993 MB | 4.5 % |
| 619 s | fork CLI, Metal | fails | | | | | | |
| 619 s | port CLI, Metal, 60 s / 25 % windows | 20 ms | 3.0 s | 517 ms | 3.6 s | 0.0058 | 528 MB | 4.7 % |
| 619 s | port CLI, Metal, default | 19 ms | 3.1 s | 516 ms | 3.6 s | 0.0058 | 532 MB | 4.7 % |
| 619 s | fork engine API, Metal | fails | | | | | | |
| 619 s | port engine API, Metal, default | 18 ms | 3.1 s | 519 ms | 3.6 s | 0.0058 | 507 MB | 4.7 % |
| 619 s | fork CLI, Core ML | 179 ms | 730 ms | 566 ms | 1.5 s | 0.0024 | 248 MB | 6.4 % |
| 619 s | port CLI, Core ML | 19 ms | 722 ms | 266 ms | 1.0 s | 0.0016 | 279 MB | 9.7 % |
| 619 s | fork engine API, Core ML | 178 ms | 782 ms | 498 ms | 1.5 s | 0.0024 | 232 MB | 11.7 % |
| 619 s | port engine API, Core ML | 18 ms | 769 ms | 547 ms | 1.3 s | 0.0022 | 226 MB | 11.8 % |

**Samsung S25, Snapdragon 8 Elite (CPU only), 6 threads**

| Clip | Build and path | Mel | Encoder | Decoder | Total | RTF | Peak RSS | WER |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| 11 s | fork CLI, default 60 s / 25 % windows | 5 ms | 182 ms | 11 ms | 199 ms | 0.0181 | - | 0.0 % |
| 11 s | port CLI, `--window-sec 60 --overlap 0.25` | 4 ms | 154 ms | 13 ms | 171 ms | 0.0155 | - | 0.0 % |
| 11 s | port CLI, default (balanced automatic windows) | 3 ms | 153 ms | 10 ms | 166 ms | 0.0151 | - | 0.0 % |
| 11 s | fork engine API, default options | 6 ms | 195 ms | 13 ms | 213 ms | 0.0194 | 547 MB | 0.0 % |
| 11 s | port engine API, default options | 3 ms | 227 ms | 11 ms | 241 ms | 0.0220 | 396 MB | 0.0 % |
| 11 s | fork engine API, 750-frame windows, 94 context | 6 ms | 186 ms | 12 ms | 204 ms | 0.0186 | 547 MB | 0.0 % |
| 11 s | port engine API, 750-frame windows, 94 context | 1 ms | 147 ms | 8 ms | 157 ms | 0.0143 | 398 MB | 0.0 % |
| 413 s | fork CLI, default 60 s / 25 % windows | 225 ms | 21.5 s | 613 ms | 22.3 s | 0.0541 | - | 5.1 % |
| 413 s | port CLI, `--window-sec 60 --overlap 0.25` | 81 ms | 21.3 s | 430 ms | 21.8 s | 0.0528 | - | 4.9 % |
| 413 s | port CLI, default (balanced automatic windows) | 118 ms | 29.7 s | 539 ms | 30.3 s | 0.0735 | - | 4.9 % |
| 413 s | fork engine API, default options | 557 ms | 96.3 s | 1.4 s | 98.3 s | 0.2382 | 7.8 GB | 6.5 % |
| 413 s | port engine API, default options | 119 ms | 28.1 s | 541 ms | 28.8 s | 0.0698 | 875 MB | 4.9 % |
| 413 s | fork engine API, 750-frame windows, 94 context | 495 ms | 30.3 s | 1.1 s | 31.9 s | 0.0773 | 2.9 GB | 5.6 % |
| 413 s | port engine API, 750-frame windows, 94 context | 98 ms | 31.1 s | 539 ms | 31.7 s | 0.0768 | 875 MB | 4.9 % |
| 619 s | fork CLI, default 60 s / 25 % windows | 488 ms | 38.1 s | 1.4 s | 40.0 s | 0.0646 | - | 4.7 % |
| 619 s | port CLI, `--window-sec 60 --overlap 0.25` | 159 ms | 37.9 s | 794 ms | 38.9 s | 0.0628 | - | 4.5 % |
| 619 s | port CLI, default (balanced automatic windows) | 167 ms | 41.3 s | 801 ms | 42.3 s | 0.0683 | - | 4.5 % |

## Multi-machine benchmark (2026-09)

Maintainer-run measurement of Parakeet TDT 0.6b v3 across four machines and
seven device-backend lanes, timed from outside the process — the engine's own
timer is not quoted. Compute per transcription is an external two-point
slope, so model load, process start-up, and wav decode cancel out of the
number. Clips are byte-identical on all machines: `jfk.wav` 11.00 s,
`ls90.wav` 98.49 s. Build: engine `46afe7d9`, ggml `speech@157b299f`,
`parakeet-tdt-0.6b-v3.q8_0.gguf` (715 MiB, 100 % Q8_0 body).

| Device | Backend | short 11.0 s | long 98.49 s |
|---|---|--:|--:|
| MacBook Air M5 | Metal | 60.53 ms (RTF 0.0055) | 755.80 ms (RTF 0.0077) |
| MacBook Air M5 | CPU | 445.44 ms (RTF 0.0405) | withheld |
| RTX 3080 desktop | CUDA | 13.49 ms (RTF 0.0012) | 115.51 ms (RTF 0.0012) |
| RTX 3080 desktop | Vulkan | 15.41 ms (RTF 0.0014) | 143.98 ms (RTF 0.0015) |
| Strix Halo | Vulkan | 30.21 ms (RTF 0.0027) | 228.27 ms (RTF 0.0023) |
| RTX 5090 box | CUDA | 8.00 ms (RTF 0.0007) | 58.46 ms (RTF 0.0006) |
| RTX 5090 box | Vulkan | 11.56 ms (RTF 0.0011) | 69.51 ms (RTF 0.0007) |

Peak GPU memory on the long clip (`nvidia-smi` per-process sampling at 5 Hz,
warm run only; RADV and Metal expose no equivalent counter): 2008 MiB on the
RTX 3080 under CUDA and 1748 MiB under Vulkan; 2386 MiB on the RTX 5090
under CUDA and 1935 MiB under Vulkan.

Accuracy: jfk 0.00 % and ls90 0.80 % WER (`compute-wer.py`, `english`
normaliser) on every lane; the statement of record stays the 500-utterance
LibriSpeech run at 2.15–2.22 % across backends and quantisation tiers.

Method, caveats, CI snapshots, and the CUDA/Metal decode
optimization history: [docs/performance.md](performance.md).
