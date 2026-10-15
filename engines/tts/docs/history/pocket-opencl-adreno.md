# Pocket OpenCL validation on OnePlus 13s

Measured 2026-10-15 for QVAC-26902, Fabric `c99e116c` with registry-resolved
ggml-speech `06f1b66806626a635904f31db6914ad2d0e71197`. The implementation is
independent of Fabric PR #338. No ggml or Hexagon source changes are required.
CPU remains the default.

The [initial machine-readable record](pocket-opencl-adreno.json) contains each warm
measurement, first-process measurements, PCM/transcription results, model
hashes, build flags and the complete memory-preflight reports.
The [expanded performance record](pocket-opencl-adreno-perf.json) contains a
subsequent 60-request comparison. Both datasets are retained because the
normally running phone showed substantial timing variability.

## Device and assets

- OnePlus 13s, CPH2723, SM8750, Adreno 830 v2, Android 16; 11,384,676 KiB RAM.
- OpenCL 3.0 QUALCOMM build `0800.74`, compiler `E031.47.18.51`.
- NDK r28b / Clang 19, arm64-v8a, Android API 28, Release. Shared dynamic
  ggml backends, OpenCL/Adreno kernels and kernel cache enabled; ggml OpenMP,
  Vulkan, CUDA, Metal and Hexagon disabled.
- Kyutai `pocket-tts-without-voice-cloning`, English 2026-04 checkpoint and
  tokenizer at `d29db7978e464fb90cb3359ee0c69a273b9142cc`; Alba embedding at
  `e81d79e8194ad4c7ce879c87a4258ef20cbf2487`.
- Source checkpoint SHA-256:
  `be9c6b4876d3f30740a8225dfcaa2e43dc4aeb753c15272735bee16bbb4abb0a`.
  Conversion and reference generation used the requirements-file upstream pin
  `0c2db3bdea7c991c568989cc11b503f14483fabc`.
- F32 converted weights and arithmetic, prepared Alba voice, one flow step,
  temperature 0.3, seed 1234, two threads per worker. Native RNG is not
  PyTorch seed-equivalent.

These are native process measurements on an otherwise normally running phone.
Frequencies, app activity and thermal conditions were not controlled. The
battery temperature read after the benchmarks was 33.7 C. Three warm runs are
a small baseline, not a platform-wide performance guarantee.

## Correctness

All **19 device checks passed**: backend selection, loading, graph-memory
accounting, EOS generation, reference-audio parsing, 322 frontend cases,
small and real FlowLM fixtures, real and random Mimi fixtures, CPU/OpenCL fit,
and CPU/OpenCL public-engine tests.

FlowLM exercised embedding lookup, prefill, cached decoding, 1/2/4/16 flow
steps, sampled trajectories, reset/reuse, exact context capacity and invalid
input/overflow recovery. The existing tolerances were unchanged: F32 `1e-4`,
F16 storage `1e-2`. The real F32 fixture's largest error was `7.8529119e-5`.

Mimi exercised individual and variable-size chunks, convolution seams, its
250-position attention window, resets and failure recovery. Each latent frame
advances 16 transformer positions, so the 37-frame random fixture crosses that
window. Real decoder versus PyTorch: max absolute error `2.4922192e-6`, SNR
120.215 dB. The random encoder also met its existing `0.003` tolerance. The
released checkpoint's disabled voice encoder was explicitly rejected.

The engine tests verified batch/streaming agreement, 24 kHz mono output,
cancellation, repeated requests, callback errors, overlapping-call rejection,
and parallel engine construction/synthesis/destruction. Both stage backend
names were `OpenCL` on the Adreno device. CPU tests also passed.

Device testing exposed corruption in the pinned backend's vectorized
`diag_mask_inf` kernel at full-context prefill. FlowLM's OpenCL graph therefore
uses an explicit causal mask with `soft_max_ext`; the CPU graph is unchanged.
The existing full-context-versus-chunks test reproduced the fault before this
fix and passed afterward. The mask's device arena and host upload staging are
included in preflight.

[CI at the tested commit](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/38778583428)
passed 161 tests in each of Ubuntu portable, Ubuntu tinyBLAS and macOS, plus
the macOS Metal and Core ML parity checks. Model-backed Pocket tests were
disabled in CI because it lacked their assets; host/device fixture runs supply
that evidence. Independent code review found no outstanding issues.

## Speech output

The two prompts were:

1. "Hello Alok. Your meeting starts in ten minutes." — 3.04 seconds of audio.
2. "The research team announced a new speech model today. It can generate
   natural English speech directly on your phone." — 6.16 seconds of audio.

Both backends produced finite, non-silent 24 kHz mono PCM with no clipped
samples. CPU/OpenCL durations and independent Whisper base.en transcripts
matched for each prompt. The news prompt had zero word errors. The reminder
transcript was "Hello, Alach. Your meeting starts in 10 minutes." on both
backends: raw normalized-word WER was 25%, including the name recognition and
the `ten`/`10` representation difference. Numbers were not canonicalized for
this metric.

CPU/OpenCL waveform SNR was 80.60 dB for the reminder and 54.61 dB for news;
correlation was 0.999999996 and 0.999998271, respectively. These are numerical
and intelligibility checks. The comparison samples were also provided in the
user's Windows Downloads folder. The user approved both outputs as equivalent
after reviewing them. This checks the two supplied samples, rather than broad
voice or naturalness coverage.

## Expanded performance comparison

A second pass used the same native binary, ggml pin, assets and synthesis
settings. For each of three prompts, the process order was CPU, OpenCL,
OpenCL, CPU. Each process loaded its models, performed one untimed warmup,
then measured five requests. This gives ten warm requests per backend per
prompt, 60 measured requests overall. Ordering was counterbalanced, but
frequencies, background apps and thermal conditions remained uncontrolled.
Battery temperatures observed around the processes ranged from 35.4 to 37.0 C.

Generation time below excludes engine loading and warmup. All requests for a
given prompt produced the same audio duration on both backends. Throughput
relative to CPU is CPU median generation time / OpenCL median generation time.

| Prompt | Speech duration (s) | CPU generation (s) | OpenCL generation (s) | CPU RTF | OpenCL RTF | OpenCL throughput vs CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reminder | 3.04 | 1.113 | 1.033 | 0.366 | 0.340 | 1.08x |
| News | 6.16 | 2.368 | 1.921 | 0.384 | 0.312 | 1.23x |
| Paragraph | 14.08 | 5.147 | 4.084 | 0.366 | 0.290 | 1.26x |

The new paragraph was:

> Good morning. Here is your schedule for today. The engineering meeting
> begins at ten, followed by lunch with the team at noon. In the afternoon,
> we will review the latest results and plan the next release. Please bring
> your notes and leave enough time to travel between meetings.

OpenCL's median throughput across the prompts was 2.94–3.45 seconds of speech
per second of processing. Its observed median advantage over this CPU configuration
was 8–26%. This is a device/run observation, not a consistent platform-wide
speedup claim. The initial smaller pass below had slower OpenCL generation.

| Prompt | CPU first audio, median (ms) | OpenCL first audio, median (ms) | CPU generation range (s) | OpenCL generation range (s) |
| --- | ---: | ---: | ---: | ---: |
| Reminder | 152.3 | 68.0 | 1.029–1.180 | 0.971–1.072 |
| News | 255.0 | 63.4 | 2.300–2.407 | 1.546–2.408 |
| Paragraph | 314.8 | 64.4 | 3.667–10.154 | 3.450–5.173 |

First-audio timing starts after engine loading. The paragraph's large CPU
range demonstrates the need to retain raw measurements and control device
conditions before making stronger comparisons. OpenCL paragraph first audio
also varied from 56.9 to 262.9 ms despite its 64.4 ms median. This run does not
identify the cause of those outliers or isolate individual kernel costs.

All 12 saved comparison WAVs were finite, non-silent 24 kHz mono and had no
clipped samples. Independent Whisper base.en transcripts for the new CPU and
OpenCL paragraph were identical. Both rendered `ten` as `10`, giving a raw
word-error rate of 1/48; no other word differed from the input.

The [expanded JSON record](pocket-opencl-adreno-perf.json) preserves all 60
request measurements, backend identities, process loads, RSS, temperatures,
asset/binary hashes and paragraph ASR results. The source binary predates
documentation-only commit `cb4af728`; no execution code changed between the
initial and expanded passes.

To reproduce the ordering in the ADB-shell setup below, run each prompt with
`--runs 5` in the order `cpu`, `opencl`, `opencl`, `cpu`, writing a separate
report for each process. Retain all ten requests for each backend when
computing the median. Keep the existing two-thread, one-step, temperature
0.3, seed 1234 settings and prepared voice unchanged.

## Initial timing and memory baseline

Warm first-audio, synthesis and RTF results below are medians of three requests
after one untimed warmup. Load time is one engine construction per process;
peak RSS covers that process, including loading and warmup. First-audio latency
starts at the synthesis call, after model loading. RTF is generation time /
audio duration; values below 1 mean faster than real time.

| Prompt | Backend | Load, cached kernels (s) | First audio (ms) | Synthesis (s) | RTF | Peak process RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Reminder | CPU | 0.726 | 148.9 | 0.962 | 0.316 | 507.2 |
| Reminder | OpenCL | 1.082 | 220.9 | 1.853 | 0.610 | 485.0 |
| News | CPU | 0.760 | 384.3 | 2.233 | 0.363 | 523.2 |
| News | OpenCL | 1.257 | 77.2 | 2.410 | 0.391 | 485.2 |

The first fresh-process reminder request, without a warmup, measured:

| Backend | Load (s) | First audio after load (ms) | Synthesis (s) | RTF |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.587 | 138.3 | 0.839 | 0.276 |
| OpenCL, initially empty kernel cache | 26.070 | 71.9 | 1.704 | 0.561 |

The empty-cache OpenCL load includes driver kernel compilation. Later OpenCL
loads used previously cached binaries. Model files could already be in the OS
page cache; these are not cold-storage measurements.

OpenCL was slower in total synthesis time on both prompts in this initial baseline.
The news first-audio latency improved; reminder latency varied considerably
(71–271 ms across warm runs). No general speedup is established. Both stages
use F32, and Pocket must serialize submissions because the pinned OpenCL
backend shares mutable kernel arguments and scratch across handles. Kernel
launch overhead, shared-queue scheduling and graph submission granularity are
profiling candidates, rather than measured per-component bottlenecks.

For the reminder, metadata-only preflight with the default 256 MiB margin
reported:

| Backend | Device/primary buffers (MiB) | Host extras (MiB) | Required including margin (MiB) |
| --- | ---: | ---: | ---: |
| CPU | 579.0 | 121.5 | 956.5 |
| OpenCL | 1017.0 | 121.7 | 1394.6 |

OpenCL's state includes one conservative shared-backend scratch bound of
459,112,448 bytes. Adreno shares system RAM, so primary and host allocations
are combined against OS availability. These projections are deliberately
conservative. Peak RSS comes from Android `time -v`; it does not measure all
driver-owned GPU allocations. Separate per-process GPU allocation counters
were unavailable to the ADB shell, so no measured total GPU-memory claim is
made.

## ggml dependency and separate backend fixes

The Fabric implementation and both benchmark passes use the unchanged
registry ggml pin. Its existing OpenCL operations are sufficient for both
FlowLM and Mimi with Fabric's explicit causal-mask workaround, so this PR
does not require a ggml PR or a new ggml source pin.

The vectorized mask defect remains in
[ggml speech at e1fb17af](https://github.com/tetherto/qvac-ext-ggml/blob/e1fb17af921c04a3004b7850021c4b44397a37dd/src/ggml-opencl/kernels/diag_mask_inf.cl):
`(&dst[i+1])[k]` and `(&dst[i])[k]` index whole `float4` objects instead of
their individual lanes. A separate ggml correctness fix is warranted for
other callers of that kernel. It is independent of this Fabric PR and was
not applied during these measurements. Likewise, the effect of replacing
shared backend state or optimizing kernels requires separate profiling and
validation; these timings do not establish a specific backend bottleneck.

## Reproduction

Prepare the bundle and fixtures using the commands in the
[Pocket guide](../pocket-tts.md), with the asset revisions above. Build the
reviewed ggml pin, then Fabric, with your Android NDK and OpenCL headers/link
library. This validation used these key configuration options:

```sh
cmake -S "$GGML_SOURCE" -B build/pocket-ggml-android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$GGML_PREFIX" \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF \
  -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_OPENMP=OFF \
  -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=ON -DGGML_OPENCL_KERNEL_CACHE=ON \
  -DGGML_HEXAGON=OFF -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_METAL=OFF \
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF \
  -DOpenCL_INCLUDE_DIR="$OPENCL_HEADERS" -DOpenCL_LIBRARY="$OPENCL_LIBRARY"
cmake --build build/pocket-ggml-android --parallel 4
cmake --install build/pocket-ggml-android
cmake -S engines/tts -B build/pocket-android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$GGML_PREFIX" \
  -Dggml_DIR="$GGML_PREFIX/share/ggml" -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
  -DTTS_CPP_BUILD_TESTS=ON -DTTS_CPP_BUILD_EXECUTABLES=ON \
  -DTTS_CPP_POCKET_OPENCL_TESTS=ON -DGGML_HEXAGON=OFF \
  -DOpenCL_INCLUDE_DIR="$OPENCL_HEADERS" -DOpenCL_LIBRARY="$OPENCL_LIBRARY"
cmake --build build/pocket-android --parallel 4 --target pocket-cli \
  test-pocket-flow-lm test-pocket-mimi test-pocket-engine test-pocket-fit
```

Stage the binaries, dynamic ggml libraries, NDK arm64 `libomp.so`, converted
bundle and reference directories into a task directory on the phone. The
following commands run **inside an ADB shell**, from that directory. Set
`POCKET_DIR` to its absolute device path:

```sh
export LD_LIBRARY_PATH="$POCKET_DIR:/system/lib64:/vendor/lib64"
export TMPDIR="$POCKET_DIR/tmp"
export GGML_OPENCL_CACHE_DIR="$POCKET_DIR/cl-cache"
export GGML_OPENCL_FORCE_LOAD=1
export OCL_ICD_FILENAMES=/vendor/lib64/libOpenCL_adreno.so
mkdir -p "$TMPDIR" "$GGML_OPENCL_CACHE_DIR"
./test-pocket-flow-lm real-flow-reference/flow-lm-f32.gguf real-flow-reference 0.0001 opencl
./test-pocket-flow-lm real-flow-reference/flow-lm-f16.gguf real-flow-reference 0.01 opencl
./test-pocket-mimi bundle/mimi.gguf real-mimi-reference opencl
./test-pocket-engine bundle bundle/flow-lm.gguf opencl
./test-pocket-fit bundle opencl
for backend in cpu opencl; do
  /system/bin/time -v ./pocket-cli --model-dir bundle --backend "$backend" \
    --threads 2 --steps 1 --temperature 0.3 --seed 1234 --runs 3 \
    --text "Hello Alok. Your meeting starts in ten minutes." \
    --output "$backend.wav" --report "$backend.json"
done
```

The force-load setting is needed for this OpenCL-only build because the
reviewed Android registry policy uses Vulkan enumeration to detect Adreno.
It does not change Pocket's explicit backend selection. This phone's ICD
loader also needed the explicit vendor ICD path. System libraries precede
vendor libraries to avoid incompatible binder dependencies during vendor
driver loading. These are standalone shell setup details, not SDK settings.

Use `--runs 1` without a prior synthesis for first-request measurements and
an unused `GGML_OPENCL_CACHE_DIR` for initial kernel compilation. Keep the
same assets, prompt, threads, steps, temperature and seed for both backends.
For independent ASR, resample copies to 16 kHz PCM16 and run Whisper base.en
with English, beam size 5 and two CPU threads. Inspect original 24 kHz PCM
for finite values, amplitude, clipping and duration, then listen to the
original files. ASR does not score naturalness.
