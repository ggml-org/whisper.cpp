# tts-cpp

> **In-tree subtree of [`chatterbox.cpp`](https://github.com/gianni-cor/chatterbox.cpp)
> integrated into `qvac-ext-lib-whisper.cpp`.** The standalone development
> repo carries `scripts/setup-ggml.sh` + `patches/` and supports
> bundled-ggml dev builds; this in-tree subtree drops both and consumes
> ggml exclusively through the QVAC speech-stack `ggml-speech` vcpkg
> port (the [`qvac-ext-ggml/speech`](https://github.com/tetherto/qvac-ext-ggml/tree/speech)
> branch, which ships the patches pre-applied).  See [§1 below](#1-build-from-the-qvac-speech-stack)
> for the build flow.

**Chatterbox** (Resemble AI, MIT-licensed zero-shot text-to-speech) ported
to [`ggml`](https://github.com/ggml-org/ggml).  Pure C++/ggml inference on
CPU / Metal / CUDA / Vulkan, no runtime dependency on Python or PyTorch.
Ships both variants out of one binary, autodetected from GGUF metadata:

- **Turbo** — English, GPT-2 Medium T3, meanflow 2-step CFM.  Optimised for
  lowest-latency CLI use.
- **Multilingual** — 23 languages, Llama-520M T3 + perceiver resampler +
  classifier-free guidance, standard 10-step CFM with CFG inside.  Tier-1
  subset wired up natively (`en, es, fr, de, it, pt, nl, pl, tr, sv, da,
  fi, no, el, ms, sw, ar, ko`); `ja/he/ru/zh/hi` stay on the backlog.

End-to-end inference on a short sentence with voice cloning from an 11 s
reference wav (T3 + S3Gen + HiFT, warm runs, excludes model load):

**Turbo:**

| Backend                              | Wall      | `RTF`  | vs real-time | vs ONNX Runtime |
|--------------------------------------|----------:|-------:|-------------:|----------------:|
| Vulkan (RTX 5090, Q4_0)              |   463 ms  | 0.07   | **14.2×**    | **13.8× faster** |
| Metal (Mac Studio M3 Ultra, Q4_0)    |   985 ms  | 0.16   | 6.4×         | 17.5× faster    |
| CPU (AMD Ryzen 9 9950X, AVX, Q4_0)   | 5 397 ms  | 0.82   | 1.2×         | 1.2× faster     |
| CPU (Mac Studio M3 Ultra, NEON)      | 7 568 ms  | 1.05   | 0.96×        | 2.3× faster     |
| Reference (ONNX Runtime, CPU Q4)     | 6.4–17 s  | 1.2–3.2 | 0.3–0.85×   | —               |

**Multilingual** (same Spanish prompt, seed 42, built-in voice;
ONNX reference uses `jfk.wav` via the [multilingual-bench][bench] script):

| Backend                              | Wall      | `RTF` | vs real-time | vs ONNX Runtime |
|--------------------------------------|----------:|------:|-------------:|----------------:|
| **Metal (M3 Ultra, Q4_0, `--cfm-steps 7`)** | **1.05 s**| **0.30** | **3.3×**     | **48.4× faster**¹ |
| Metal (M3 Ultra, Q4_0)               |  **1.22 s** | 0.35  | 2.9×        | **42.0× faster**¹ |
| Metal (M3 Ultra, F16, `--cfm-steps 7`)| 1.16 s   |  0.32  | 3.2×        | **45.9× faster**¹ |
| Metal (M3 Ultra, F16)                |  1.41 s   |  0.38  | 2.6×        | **37.5× faster**¹ |
| **Metal (M4, Q4_0)**                 |  **3.0 s**| 1.37  | 0.73×        | **10.6× faster**¹ |
| Metal (M4, F16)                      |   4.0 s   | 1.65  | 0.61×        | **14.2× faster**¹ |
| CPU (M4, 4t NEON, Q4_0)              |  10.7 s²  | 4.32² | 0.23×        | **3.0× faster**¹  |
| CPU (M4, 4t NEON, F16)               |  17.1 s²  | 6.70² | 0.15×        | **3.1× faster**¹  |
| Reference (ONNX Runtime, CPU 4t, q4) |  31.7 s   |14.55  | 0.07×        | —                |
| Reference (ONNX Runtime, CPU 4t, fp16)|53.3 s   |23.50  | 0.04×        | —                |

The M3 Ultra rows reflect the §3.21 optimisation pass — CFG cond+uncond
batched into one Metal forward (B=2) on T3, the new `--cfm-steps N` knob
on the standard 10-step CFM (N=7 is the recommended quality knee, log-mel
cosine vs N=10 = **0.995**), and `ggml_swiglu_split` on the Llama MLP.
The M4 rows are kept for continuity with §3.19/§3.20.

¹ ONNX Runtime's multilingual ONNX export ships **without** the
`text_emb_weight.bin` tensor and logs `CFG disabled` at load, so it's
running half the compute of the ggml pipeline (1 T3 forward per token
instead of 2 — no classifier-free guidance, no CFG-combined CFM). If
the ONNX CFG path were wired up, its RTF would roughly double and the
gap vs ggml would widen to ~10–14× (CPU) / ~20–28× (Metal). ggml runs
the full CFG pipeline in every row above. Reproduction + per-stage
breakdown in [`PROGRESS.md §3.19–3.20`](PROGRESS.md).

² **CPU multilingual rows re-measured after restoring CFG on the
non-Metal CFM path.**  The previous numbers in this row (`6.0 s / 2.69`
and `7.8 s / 3.24`) were captured between Apr 23–May 4 while the
non-`use_b2` branch in the CFM step loop was silently running only the
conditional pass — i.e. half the CFM compute, no classifier-free
guidance steering on CPU/Vulkan/CUDA.  Fixed in commit `6d9b42b`; the
two rows above are now end-to-end re-measurements on the same M4 host
with CFG correctly applied (12 CFM steps × 2 forward calls).
S3Gen wall-time roughly doubled, RTF went up ~2×.  Metal rows are
unaffected (the `use_b2` branched path always carried the CFG combine).

See the [full benchmark](#performance) section below for the per-stage
breakdown, or [`PROGRESS.md`](PROGRESS.md) for the full chronological
development journal — every numerical-parity stage and optimization pass
(T3 Flash Attention, KV-cache layout rework, Metal kernel patches,
CAMPPlus + VoiceEncoder + S3TokenizerV2 ported to ggml graphs, mel
extraction via STFT matmul, T3 Q4/Q5/Q8 quantization, the multilingual
Llama-520M port + CFG dual-cache (§3.19), and the shared S3Gen weight-
quantisation pass that ships in this repo (§3.20)).

---

## API overview

The two high-level entry points exported through the `TTS_CPP_API`
public surface (defined in
[`include/tts-cpp/export.h`](include/tts-cpp/export.h)):

| Surface | Role |
|---------|------|
| `tts_cpp::chatterbox::Engine::synthesize` | One-shot text → 24 kHz PCM for the Chatterbox pipeline (T3 + S3Gen + HiFT). Loads the T3 + S3Gen GGUFs once at construction, optionally bakes a voice-cloning profile from a reference wav, and reuses everything across calls. Header: [`<tts-cpp/chatterbox/engine.h>`](include/tts-cpp/chatterbox/engine.h). Variant (Turbo / Multilingual) is autodetected from `chatterbox.variant` GGUF metadata. |
| `tts_cpp::supertonic::synthesize` | One-shot text → 44.1 kHz PCM for the Supertonic CPU TTS family. Header: [`<tts-cpp/supertonic/engine.h>`](include/tts-cpp/supertonic/engine.h). |

Lower-level helpers also exposed via `TTS_CPP_API`, useful when
embedding into a host that already manages model lifetime:
**`s3gen_synthesize_to_wav`** (file-output S3Gen path,
[`<tts-cpp/chatterbox/s3gen_pipeline.h>`](include/tts-cpp/chatterbox/s3gen_pipeline.h)),
**`s3gen_preload`** / **`s3gen_unload`** (host-side weight pre-staging
+ explicit teardown — required for long-running processes that cycle
through models or that need deterministic teardown before destroying
their own ggml backend), and **`tts_cpp_cli_main`** (the same
entry point used by the `tts-cli` binary — useful for embedding the
end-to-end CLI into a downstream binary without forking
[`src/cli_main.cpp`](src/cli_main.cpp)).  Anything outside the
`TTS_CPP_API`-annotated surface (in particular the
`tts_cpp::supertonic::detail::*` and `tts_cpp::chatterbox::*_internal`
helpers reachable from the supertonic / chatterbox test harnesses) is
**not** part of the public API and is hidden from `SHARED` builds.

### Consumer integration

Downstream projects in the QVAC speech stack consume `tts-cpp` via
the matching `tts-cpp` vcpkg port (this in-tree subtree).  ggml comes
from the [`ggml-speech`](https://github.com/tetherto/qvac-registry-vcpkg)
sister port, which vendors the
[`qvac-ext-ggml/speech`](https://github.com/tetherto/qvac-ext-ggml/tree/speech)
branch with all Metal / OpenCL / Vulkan patches pre-applied.  Once
both ports are installed, integration on the consumer side is one
`find_package` call:

```cmake
find_package(tts-cpp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE tts-cpp::tts-cpp)
```

```cpp
#include <tts-cpp/chatterbox/engine.h>

tts_cpp::chatterbox::EngineOptions opts;
opts.t3_gguf_path    = "models/chatterbox-t3-turbo.gguf";
opts.s3gen_gguf_path = "models/chatterbox-s3gen.gguf";
opts.n_gpu_layers    = 99;
tts_cpp::chatterbox::Engine engine(opts);
auto result = engine.synthesize("Hello, world.");
// result.pcm is 24 kHz mono float32 PCM ready to be written as wav.
```

The supertonic test/bench harnesses link against `tts-cpp` directly
and use detail-namespaced symbols outside the `TTS_CPP_API` public
surface, so the integrated port keeps the default
`TTS_CPP_BUILD_SHARED=OFF` and `TTS_CPP_BUILD_TESTS=OFF`.  See
[**Useful CMake options**](#useful-cmake-options) below for the full
flag table.

## Pipeline at a glance

```
      text                                                 24 kHz wav
       │                                                        ▲
       ▼                                                        │
  ┌────────────────────────────────────────────────────────────────┐
  │                       tts-cli (libtts-cpp)                     │
  │                                                                │
  │      T3      ──►   S3Gen encoder   ──►        CFM              │
  │  text → toks       toks → h                   h → mel          │
  │                                                                │
  │                         HiFT vocoder  ──►  24 kHz wav          │
  └────────────────────────────────────────────────────────────────┘
       ▲                                              ▲
   text tokenizer                              reference voice
   (embedded in T3 GGUF metadata)              (embedded in S3Gen GGUF)
```

`tts-cli` (and the back-compat `chatterbox` binary, same code) handles
both Chatterbox variants via `chatterbox.variant` GGUF metadata.  Supertonic
GGUFs are also autodetected from `supertonic.arch` and routed to the
Supertonic engine.

| Stage         | Turbo                                      | Multilingual                                        |
|---------------|--------------------------------------------|-----------------------------------------------------|
| Tokenizer     | GPT-2 byte-level BPE (English)             | HuggingFace `tokenizers.json` (23 langs, NFKD pre)  |
| T3 backbone   | GPT-2 Medium, 24 layers, single forward    | Llama-520M, 30 layers, CFG cond+uncond per token    |
| CFM solver    | Meanflow, 2 Euler steps                    | Standard, 10 Euler steps with `cfg_rate=0.7`        |
| HiFT vocoder  | shared (same checkpoint format)            | shared (same checkpoint format)                     |

One binary, one invocation, end to end — `scripts/synthesize.sh` is a
thin convenience wrapper that fills in the two GGUF paths.

## Experimental: Supertonic GGUF / CPU

This branch also contains an experimental Supertonic path.  It is
model-specific: the official Supertone ONNX files and assets are converted
into one GGUF, then a CPU C++ runtime runs the known Supertonic stages.

There are two related upstream bundles:

- `Supertone/supertonic` is the stable English bundle.  It should be used for
  English and does **not** wrap text in language tags.
- `Supertone/supertonic-2` is the multilingual bundle.  It should use the
  open/close language-tag path (`<lang>...</lang>`).  The older prefix-only
  form (`<lang>... `) can make English prompts stutter.

Current status:

- `scripts/dump-supertonic-reference.py` dumps ONNX Runtime reference tensors.
- `scripts/setup-supertonic2.sh` downloads the official Hugging Face bundle
  through `huggingface_hub` and writes the local GGUF.
- `scripts/convert-supertonic2-to-gguf.py` writes `models/supertonic.gguf`
  (English) or `models/supertonic2.gguf` (multilingual), depending on flags.
  When `--onnx-dir` is omitted, it downloads the selected repo just like the
  Chatterbox converters.
- `build/tts-cli` autodetects Supertonic GGUFs from `supertonic.arch` and can
  synthesize a 44.1 kHz wav on CPU.  `build/supertonic-cli` remains as a
  focused compatibility/debug wrapper.
- All four stages pass numerical parity against the ONNX reference
  (preprocess, duration, text encoder, vector estimator, vocoder), and the
  full pipeline (`test-supertonic-pipeline`) reproduces the ONNX reference
  waveform when fed the same initial noise tensor.
- The production path is GGML-backed for duration, text encoder, vector
  estimator, and vocoder.  Text relative-position self-attention and FFN blocks
  are expressed with stock GGML ops, and the speech-prompted text attention /
  vector attention blocks use `ggml_flash_attn_ext` where the math allows it.
- Vector attention uses strided Q/K/V GGML views where the time-channel layout
  permits it.  The vector runtime also keeps persistent graph/allocr caches for
  attention, ConvNeXt group, and tail islands, plus fused ConvNeXt boundary /
  tail update graphs and portable custom CPU kernels for pointwise Conv1D,
  depthwise Conv1D, row-wise layer norm, dense time matmul, and fused
  bias/GELU/residual elementwise work.
- The vocoder keeps a persistent GGML graph cache and uses portable
  BLAS/Accelerate-backed causal Conv1D custom ops for the hot projection paths.
  BLAS worker threads are capped by default to avoid nested oversubscription
  under GGML task-level threading.
- `SUPERTONIC_VECTOR_PROFILE=1` and `SUPERTONIC_TEXT_PROFILE=1` print
  per-island timings for tuning graph boundaries.  Current text profiling shows
  stock-op relpos is ~0.7-0.8 ms/layer on the quick prompt, so a fused relpos
  op is deferred until backend profiling proves it necessary.
- CPU thread count is controlled by `--threads`; the default caps at 4 threads
  because the current small-graph Supertonic path regresses when oversubscribed.
- Current CPU benchmark artifacts live in
  `artifacts/supertonic-thread-matrix/`.  The final matched matrix on this
  machine uses F1, 5 denoise steps, speed `1.05`, `runs=3`, `warmup=1`, and
  ONNX Runtime `CPUExecutionProvider` only.  GGML wins 10 of 12 matched
  thread/prompt comparisons.  The only end-to-end losses are quick English at
  4 threads (`157.7 ms` vs `148.8 ms`) and long English at 4 threads
  (`361.2 ms` vs `351.5 ms`).
- The current quick prompt 4-thread medians are `13.5 ms` text encoder,
  `96.3 ms` vector estimator, `43.6 ms` vocoder, and `157.7 ms` total
  (`RTF 0.050`).  Portuguese 4-thread now wins end to end (`234.3 ms` GGML vs
  `250.8 ms` ONNX), with GGML vocoder at `68.9 ms` vs ONNX `95.6 ms`.

Latest matched CPU matrix, median total milliseconds:

| Prompt | GGML 1t | GGML 2t | GGML 3t | GGML 4t | ONNX 1t | ONNX 2t | ONNX 3t | ONNX 4t |
|--------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| quick English | 298.0 | 189.4 | 157.7 | 157.7 | 373.8 | 218.5 | 168.3 | 148.8 |
| longer English | 757.5 | 491.2 | 390.3 | 361.2 | 1103.0 | 580.6 | 555.7 | 351.5 |
| Portuguese smoke | 457.2 | 292.9 | 251.0 | 234.3 | 610.6 | 344.6 | 268.3 | 250.8 |

Example:

```bash
# Stable English bundle: no language wrapping.
bash scripts/setup-supertonic2.sh --arch supertonic

cmake --build build --target tts-cli
./build/tts-cli \
  --model models/supertonic.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice F1 --language en --steps 5 --speed 1.05 \
  --out /tmp/supertonic.wav

# Multilingual bundle: uses the <lang>...</lang> wrapping path.
bash scripts/setup-supertonic2.sh

cmake --build build --target tts-cli
./build/tts-cli \
  --model models/supertonic2.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice M1 --language en --steps 5 --speed 1.05 \
  --out /tmp/supertonic.wav

# Bit-exact reproduction of the ONNX reference run (pass the same noise tensor)
./build/tts-cli --model models/supertonic2.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice M1 --language en --steps 5 --speed 1.05 \
  --noise-npy artifacts/supertonic-ref-quick/noise.npy \
  --out /tmp/supertonic.wav

# Matched GGML benchmark with machine-readable metrics.
./build/supertonic-bench \
  --model models/supertonic2.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice F1 --language en --steps 5 --speed 1.05 \
  --threads 4 --runs 5 --warmup 1 \
  --json-out artifacts/supertonic-bench.json

# Matched ONNX Runtime benchmark.  Use open_close wrapping for Supertonic 2.
python scripts/bench-supertonic-onnx.py \
  --onnx-dir /path/to/supertonic-pytorch/onnx_models/onnx \
  --assets-dir /path/to/supertonic-pytorch/assets \
  --voice-style /path/to/supertonic-pytorch/assets/voice_styles/F1.json \
  --text "The quick brown fox jumps over the lazy dog." \
  --lang en --language-wrap-mode open_close \
  --steps 5 --speed 1.05 --threads 1 --runs 5 --warmup 1 \
  --json-out artifacts/supertonic-onnx-bench.json
```

## Prerequisites

- C++17 compiler (clang or gcc)
- cmake ≥ 3.14
- Python 3.10+ with `torch`, `numpy`, `onnx`, `gguf`, `huggingface_hub`,
  `safetensors`, `scipy`, `librosa`, `resampy` — needed **once**, at setup time only, to run the
  weight converters (which bake the precomputed mel filterbanks into the
  GGUFs) and the optional reference-dump scripts. Once the GGUFs exist,
  the C++ binary has zero runtime dependency on Python.

The easiest way to get the Python side is:

```bash
git clone https://github.com/resemble-ai/chatterbox.git chatterbox-ref
cd chatterbox-ref
python -m venv .venv && . .venv/bin/activate
pip install -e .
pip install onnx gguf huggingface_hub safetensors scipy librosa resampy
cd -
```

## 1. Build from the qvac speech stack

This in-tree subtree is built against the [`ggml-speech`](https://github.com/tetherto/qvac-registry-vcpkg)
vcpkg port (which vendors the [`qvac-ext-ggml/speech`](https://github.com/tetherto/qvac-ext-ggml/tree/speech)
branch with all patches pre-applied).  `tts-cpp` itself is consumed
through the matching `tts-cpp` port; downstream applications in the
QVAC speech stack add both to their `vcpkg.json` and call
`find_package(tts-cpp CONFIG REQUIRED)`:

```cmake
find_package(tts-cpp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE tts-cpp::tts-cpp)
```

For development out of this in-tree subtree (running the parity
harnesses, prototyping API changes, etc.) the canonical build is:

```bash
# Install the speech-stack ggml port via vcpkg first; then:
cmake -S tts-cpp -B tts-cpp/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=<vcpkg_root>/scripts/buildsystems/vcpkg.cmake
cmake --build tts-cpp/build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

`TTS_CPP_USE_SYSTEM_GGML` defaults to `ON` here so the build picks
up the patched ggml from vcpkg automatically; flipping it `OFF` in
this subtree is rejected at configure time (no `patches/` to apply).
GPU acceleration is selected at the ggml-port level - the
`ggml-speech` port already carries the Metal / Vulkan / OpenCL
backend support its consumers ask for; pass `--n-gpu-layers 99` at
runtime to actually use the compiled GPU backend.

If you need a bundled-ggml dev build (`add_subdirectory(ggml)` with
patches applied locally rather than coming from vcpkg), use the
standalone [`chatterbox.cpp`](https://github.com/gianni-cor/chatterbox.cpp)
repo - the source-of-truth this subtree was copied from - which keeps
`scripts/setup-ggml.sh` + `patches/` for that flow.

### Useful CMake options

Project-namespaced flags (all default to a sensible standalone build;
override with `-D<flag>=...` at configure time):

| Flag | Default | Meaning |
|------|---------|---------|
| `TTS_CPP_BUILD_LIBRARY` | `ON` | Build the `tts-cpp` library target itself (linkage controlled by `TTS_CPP_BUILD_SHARED`, not `BUILD_SHARED_LIBS` — see below) |
| `TTS_CPP_BUILD_SHARED` | `OFF` | Build `tts-cpp` as `SHARED` instead of `STATIC`. Decoupled from `BUILD_SHARED_LIBS` because ggml's own CMake declares its own `option(BUILD_SHARED_LIBS)` defaulting to `ON` on Windows non-MinGW; using a project-namespaced option keeps the two independent. The supertonic test/bench harnesses link against `tts-cpp` directly and use detail-namespaced symbols outside the `TTS_CPP_API` public surface, so `SHARED` builds hide them and disable those targets — leave OFF for development, flip ON only for downstream packaging where the test harnesses aren't built |
| `TTS_CPP_BUILD_EXECUTABLES` | `ON` standalone / `OFF` subdir | `tts-cli`, `mel2wav`, `supertonic-cli`, `supertonic-bench` |
| `TTS_CPP_BUILD_TESTS` | `ON` standalone / `OFF` subdir | `test-*` parity / unit harnesses, registered with CTest (label-filterable via `ctest -L unit` / `ctest -L fixture` / `ctest -L gpu`) |
| `TTS_CPP_INSTALL` | `ON` | Generate `install` rules + the `tts-cpp` CMake package config so consumers can `find_package(tts-cpp CONFIG REQUIRED)` |
| `TTS_CPP_USE_SYSTEM_GGML` | `ON` (this in-tree subtree) | `find_package(ggml CONFIG REQUIRED)` against the QVAC speech-stack `ggml-speech` vcpkg port (the [`qvac-ext-ggml/speech`](https://github.com/tetherto/qvac-ext-ggml/tree/speech) branch). Flipping `OFF` is rejected at configure time here because the standalone `patches/` directory is intentionally absent. The standalone `chatterbox.cpp` repo defaults this to `OFF` for its bundled-ggml dev flow |
| `TTS_CPP_GGML_LIB_PREFIX` | `ON` | Rename bundled ggml libraries to `libspeech-ggml-*` (filename only; CMake target names and C symbols are unchanged). The `speech-` prefix is shared with `parakeet.cpp` so the QVAC speech stack (whisper, parakeet, chatterbox, supertonic) co-vendors a single ggml file set when several speech libraries are loaded into the same host process. Set to OFF to keep stock upstream filenames; no-op when `TTS_CPP_USE_SYSTEM_GGML=ON` |
| `TTS_CPP_CCACHE` | `ON` | Use ccache as compiler launcher for `tts-cpp`'s own targets when `find_program(ccache)` succeeds. Scoped per target; ggml's independent `GGML_CCACHE` option handles the ggml subdirectory |

When tests are enabled, the build also exposes three cache-overridable
fixture roots that the registered ctest entries resolve against:

| CACHE PATH | Default | Used by |
|------------|---------|---------|
| `TTS_CPP_TEST_MODEL_DIR` | `${CMAKE_CURRENT_SOURCE_DIR}/models`               | GGUF checkpoints (produced by `scripts/convert-*-to-gguf.py` or `bash scripts/setup-supertonic2.sh`) |
| `TTS_CPP_TEST_AUDIO_DIR` | `${CMAKE_CURRENT_SOURCE_DIR}/test/reference-audio` | Reference WAV fixtures (e.g. `test/reference-audio/jfk.wav`) |
| `TTS_CPP_TEST_REF_DIR`   | `${CMAKE_CURRENT_SOURCE_DIR}/artifacts`            | `.npy` reference dumps from `scripts/dump-{s3gen,supertonic,t3-mtl}-reference.py` |

Tests whose required fixtures don't exist at configure time are
auto-marked `DISABLED` (they still appear in `ctest -N` output but
return `Not Run` instead of failing), so a fresh checkout still gives
a green `ctest` run on the harnesses whose fixtures it does have.

This produces the end-to-end binary plus a set of per-stage validation
harnesses:

| Binary | What it does |
|--------|--------------|
| `build/tts-cli`            | End-to-end Chatterbox text → speech tokens (T3) → wav (S3Gen + HiFT), plus Supertonic text → wav. Handles Chatterbox voice cloning via `--reference-audio`, autodetects Chatterbox Turbo/Multilingual from `chatterbox.variant`, and autodetects Supertonic from `supertonic.arch`. |
| `build/mel2wav`               | HiFT only: mel.npy → wav (demo) |
| `build/supertonic-cli`        | Supertonic-only end-to-end CLI (text → wav) — the same engine `tts-cli` invokes when it sees a Supertonic GGUF, exposed standalone for scripting and parity work |
| `build/supertonic-bench`      | Per-stage Supertonic benchmark harness (`--text` / `--out` / `--runs`); machine-readable RTF + per-stage timings |
| `build/test-s3gen`            | Staged numerical validation of S3Gen encoder + CFM vs Python dumps |
| `build/test-resample`         | Round-trip SNR of the C++ Kaiser-windowed sinc resampler |
| `build/test-voice-features`   | 24 kHz 80-ch mel parity (prompt_feat) |
| `build/test-fbank`            | 16 kHz 80-ch Kaldi fbank parity |
| `build/test-voice-encoder`    | VoiceEncoder 256-d speaker embedding parity |
| `build/test-campplus`         | CAMPPlus 192-d embedding parity |
| `build/test-voice-embedding`  | wav → fbank → CAMPPlus end-to-end parity |
| `build/test-s3tokenizer`      | S3TokenizerV2 log-mel + speech-token parity |
| `build/test-streaming`        | Per-chunk CFM + HiFT parity for the streaming pipeline (B1) |
| `build/test-mtl-tokenizer`    | Multilingual grapheme tokenizer parity vs the HF reference |
| `build/test-t3-mtl`           | End-to-end MTL T3 (Llama-520M) forward-pass parity |
| `build/test-t3-mtl-stages`    | Staged MTL T3 parity (cond/text/inputs/layers/head) |
| `build/test-cpu-caches`       | CPU-side persistent-cache validation (time_mlp / time_emb / cfm_estimator / weight_mirror caches that amortise per-synth overhead on the multilingual CPU path); no-arg invocation runs the bit-cast cache-key + initial-state checks, GGUF arg runs the warm-cache + bit-exact pipeline check |
| `build/test-t3-caches`        | T3 step-graph cache validation (no-arg = initial-state, GGUF arg = warm-cache + bit-exact end-to-end check) |
| `build/test-supertonic-*`     | Per-stage Supertonic parity harnesses (`preprocess`, `vocoder` ± `trace` / `pointwise`, `duration` ± `trace`, `text-encoder` ± `trace`, `vector` ± `trace`, `pipeline`); each takes `MODEL.gguf REF_DIR` |
| `build/test-metal-ops`        | Metal-only: parity check for `diag_mask_inf`, `pad_ext`, and fast `conv_transpose_1d` (only useful when built with `-DGGML_METAL=ON`) |

You'll normally only need `build/tts-cli`; the `test-*` binaries are
there for the staged-verification methodology in `PROGRESS.md`, and
register with CTest so you can run `ctest -C Release -L unit` /
`ctest -C Release -L fixture` from the build directory after fixtures
are in place.

### How `TTS_CPP_USE_SYSTEM_GGML=ON` resolves ggml

When the option is `ON` (the default in this in-tree subtree), the
top-level `CMakeLists.txt` swaps `add_subdirectory(ggml)` for
`find_package(ggml CONFIG REQUIRED)` and aliases the imported
`ggml::ggml` target onto the plain `ggml` name that the rest of the
build uses.  No local `./ggml/` clone is read.  The imported package
ships the equivalent of the patches the standalone `chatterbox.cpp`
repo applies via `patches/` - the `qvac-ext-ggml/speech` branch
carries them pre-applied so consumers don't maintain a patch trail.
This shape mirrors `stable-diffusion.cpp`'s `SD_USE_SYSTEM_GGML`.

## 2. One-time: convert weights

```bash
# Activate the Python environment from the Prerequisites step
. ../chatterbox-ref/.venv/bin/activate

# --- Turbo (English, default) ---
python scripts/convert-t3-turbo-to-gguf.py --out models/chatterbox-t3-turbo.gguf
python scripts/convert-s3gen-to-gguf.py    --out models/chatterbox-s3gen.gguf

# --- Multilingual (23 languages) ---
python scripts/convert-t3-mtl-to-gguf.py            --out models/chatterbox-t3-mtl.gguf
python scripts/convert-s3gen-to-gguf.py --variant mtl --out models/chatterbox-s3gen-mtl.gguf

# --- Multilingual, quantised (recommended for speed) ---
# Matches the RTF numbers in the benchmark table above.  --quant accepts
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

The quantisation flag on `convert-s3gen-to-gguf.py` is new as of §3.20 —
it's pure data-format work, so the binary needs no changes and every
backend (CPU, Metal, Vulkan, CUDA) picks up the faster matmul kernels
transparently.  The per-tensor decision lives in `should_quantize()`
inside `scripts/requantize-gguf.py` (single source of truth shared
with the offline rewriter): biases, norm scales, embedding tables,
spectral filterbanks, voice-cloning preprocessors (CAMPPlus,
VoiceEncoder, S3TokenizerV2) and any tensor whose reduction dim isn't
block-aligned all stay at full precision.  See `PROGRESS.md §3.20` for
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
  # ~1.5-2x faster on every backend (CPU/Metal/Vulkan/CUDA) due to lower weight
  # memory bandwidth.  Recommended for production use; see benchmark table above.
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
benchmark numbers at the top of this README use the q4_0 variants
shown there.  `llama-quantize` doesn't recognize the `chatterbox` /
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

## 3. Run — end-to-end text → wav

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

**Multilingual** takes the same flags plus a required `--language CODE`
(one of the tier-1 codes listed at the top of the README) and runs all the
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
the recommended quality knee on M3 Ultra (see [`PROGRESS.md §3.21`](PROGRESS.md));
N=6 is too aggressive (cosine 0.990 right at the threshold, PCM cosine
drops to 0.88).  Streaming chunks ignore this flag and use
`--stream-cfm-steps` instead.

Everything is self-contained in the two `.gguf` files:

- `chatterbox-t3-turbo.gguf` embeds the BPE tokenizer (vocab + merges +
  added tokens) as standard `tokenizer.ggml.*` metadata, which the C++
  binary loads out of GGUF at startup.
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
- `--threads N` — override the default `std::thread::hardware_concurrency()`.
  The sweet spot on a 10-core CPU is 10.
- `--n-gpu-layers N` — move layers to the GPU backend when built with
  `-DGGML_METAL=ON` / `-DGGML_CUDA=ON` / `-DGGML_VULKAN=ON`.  Pass `99`
  (or any large number) to move everything.
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
## Performance

Reproducible perf check vs an ONNX Runtime Q4 baseline (same
architecture) on the same machine.  Shared setup:

- Text: *"Hello from native C plus plus. This audio was generated end
  to end on CPU using ggml."*
- Reference voice: `test/reference-audio/jfk.wav` (11 s mono 16 kHz)
- Seed: 42, warm 3-run average, inference only (excludes model load)

### Mac Studio M3 Ultra (96 GB unified memory)

| Implementation                        | Backend         | T3 gen             | S3Gen+HiFT gen | Total inference | RTF   | vs real-time |
|---------------------------------------|-----------------|-------------------:|---------------:|----------------:|------:|-------------:|
| **`tts-cpp` Q4_0**                    | **Metal**       |  573 ms / 155 tok  |    412 ms      |   **985 ms**    | 0.16  | **6.4×**     |
| `tts-cpp` Q4_0                        | CPU (NEON+Accel)| 2 045 ms / 178 tok |  5 523 ms      |    7 568 ms     | 1.05  | 0.96×        |
| ONNX Runtime Q4 baseline              | CPU             |        —           |      —         |   17 190 ms     | 3.18  | 0.31×        |

`tts-cpp` (Metal) is **17.5× faster than ONNX Runtime** on the
same machine; the CPU-only build is still 2.3× faster.

### Linux RTX 5090 + AMD Ryzen 9 9950X

| Implementation                        | Backend         | T3 gen             | S3Gen+HiFT gen | Total inference | RTF   | vs real-time |
|---------------------------------------|-----------------|-------------------:|---------------:|----------------:|------:|-------------:|
| **`tts-cpp` Q4_0**                    | **Vulkan**      |  241 ms / 161 tok  |    222 ms      |    **463 ms**   | 0.07  | **14.2×**    |
| `tts-cpp` Q4_0                        | CPU (AVX)       | 2 161 ms / 161 tok |  3 236 ms      |    5 397 ms     | 0.82  | 1.2×         |
| ONNX Runtime Q4 baseline              | CPU             |        —           |      —         |    6 373 ms     | 1.18  | 0.85×        |

`tts-cpp` (Vulkan) is **13.8× faster than ONNX Runtime** on the
same machine.  Note that the ONNX Runtime baseline here only uses the
CPU execution provider; a CUDA build would narrow the gap, but is not
included in this comparison.

### Per-stage S3Gen + HiFT breakdown (GPU builds)

| Stage           | M3 Ultra Metal  | RTX 5090 Vulkan |
|-----------------|----------------:|----------------:|
| T3 per token    | 3.70 ms / tok   | **1.50 ms/tok** |
| encoder         |    38 ms        |    35 ms        |
| cfm_step0       |    69 ms        |    84 ms        |
| cfm_step1       |    49 ms        |    13 ms        |
| cfm_total       |   124 ms        |   100 ms        |
| f0_predictor    |   3.1 ms        |   1.1 ms        |
| sinegen (CPU)   |    15 ms        |    16 ms        |
| stft            |   3.1 ms        |   1.0 ms        |
| **hift_decode** |   225 ms        |    66 ms        |
| hift_total      |   246 ms        |    84 ms        |

HiFT `conv_transpose_1d` upsampling is the single biggest stage on
Metal today; the 5090 chews through it 3.4× faster, which is where the
remaining end-to-end gap comes from.

### Multilingual (Apple M4, F16 weights)

Same prompt + seed run through both variants on the same M4, for apples-to-
apples comparison.  MTL is 30 transformer layers vs Turbo's 24 plus CFG on
both T3 and CFM (2 forward passes per step), and it samples standard CFM
for 10 Euler steps instead of Turbo's meanflow 2.

| Config                              | T3 infer            | S3Gen infer | Audio | **RTF** |
|-------------------------------------|---------------------:|-------------:|------:|--------:|
| Turbo, Metal                        |  788 ms /  73 tok   |    768 ms    | 3.04 s| 0.51    |
| Turbo, CPU 4t                       | 1 721 ms /  73 tok  |  3 334 ms    | 3.04 s| 1.66    |
| Multilingual, Metal *(batched CFM)* | 1 865 ms /  61 tok  |  2 247 ms    | 2.56 s| 1.61    |
| Multilingual, CPU 4t *(2-call CFM)*³| 3 210 ms /  85 tok  | 25 660 ms    | 3.52 s| 8.20    |

The MTL Metal path packs the CFG cond+uncond into a single batch=2
decoder forward (`use_b2 = !ggml_backend_is_cpu(...)`), since kernel
dispatch overhead amortises well across the bigger workload; on ggml-cpu
the extra permute+cont ops that a batched attention block needs regress
throughput, so CPU keeps the two-call path.  See
[`PROGRESS.md §3.19`](PROGRESS.md) for the measurement and a discussion
of where the MTL slowdown lives relative to Turbo.

³ Re-measured on the same M4 host after commit `6d9b42b` restored the
CFG combine on the non-`use_b2` (CPU) CFM path.  The previous row in
this table (`2 711 ms / 71 tok`, `8 029 ms`, `RTF 3.63`) was captured
while CPU was silently running only the conditional CFM pass — i.e.
half the CFM compute and no classifier-free guidance steering.  S3Gen
wall-time roughly doubled (one extra forward call per CFM step) and
RTF went from 3.63 → 8.20.  The remeasurement here uses a different
local reference wav (the original `jfk.wav` was not present on the
host) — that accounts for the slightly larger token count (85 vs the
original 71); the per-audio-second cost ratio is what shifted, ~2×,
in line with restoring the missing pass.  Turbo and the Metal
multilingual row are
unaffected — they always carried the CFG combine.

### Multilingual (Mac Studio M3 Ultra, after §3.21 optimisation pass)

Same Spanish prompt (`"Hola, esto es una demostración multilingüe."`,
`--language es`), `jfk.wav` voice, seed 42, greedy (`--temp 0 --top-k 1`),
3 warm runs averaged.  T3 is now CFG-batched into a single Metal forward
(B=2, mirrors S3Gen's `use_b2`); MLP uses `ggml_swiglu_split` so the 30
SiLU+Mul element-wise pairs collapse into one fused Metal kernel per
layer.  The new `--cfm-steps N` flag exposes the standard CFM step count
(default 10); N=7 is the recommended quality knee (log-mel cosine vs N=10
= **0.995**).

| Config                              | T3 infer           | S3Gen infer | Audio | **RTF** |
|-------------------------------------|-------------------:|------------:|------:|--------:|
| MTL, Metal Q4_0, `--cfm-steps 7`    |  478 ms /  84 tok  |    576 ms   | 3.48 s|  0.30   |
| MTL, Metal Q4_0 (default N=10)      |  482 ms /  84 tok  |    730 ms   | 3.48 s|  0.35   |
| MTL, Metal F16, `--cfm-steps 7`     |  579 ms /  89 tok  |    586 ms   | 3.68 s|  0.32   |
| MTL, Metal F16 (default N=10)       |  613 ms /  89 tok  |    752 ms   | 3.68 s|  0.37   |

Compared to the M4 multilingual numbers above, the M3 Ultra hits
**RTF 0.30** on Q4_0 — a 4.6× speedup.  The CFG-batching alone drops T3
by 42–45% (see PROGRESS.md §3.21 for the full bench matrix and the
NEGATIVE results for F16 KV cache and SwiGLU on F16).

### Multilingual (M3 Ultra, post §3.24–§3.31 Metal kernel portfolio)

Same prompt, voice, seed as §3.21 above.  Adds, on top of §3.21:

- **§3.24** — HiFT conv-kernel F16 quantisation (64 tensors).
- **§3.26** — `kernel_mul_mv_f32_f16{,_4,_short}` Metal kernel variants
  to unblock 21 more HiFT `source_*` F16 tensors
  (GGUF shrinks **754 → 747 MB**, WAV cos 1.000000 vs §3.24).
- **§3.27** — `kernel_mul_mm` + `ADD(bias)` [+ `ADD(residual)`] fusion
  for the CFM transformer Q4_0 mat-muls (1820 saved `ggml_add`
  dispatches per synth).
- **§3.28** — extends the fusion to absorb `GELU_ERF` (CFM FF ff0
  activation path; 1120 additional saved dispatches).
- **§3.30** — `test-metal-ops` fused-mul_mm parity harness + bias-only
  direct-store variant.
- **§3.31** — iOS-arm64 cross-build portability +
  `scripts/bench-m4-validation.sh` for M4 hand-off.

5-invocation averages (`default N=10` CFM — compare to the §3.21 N=10 row):

| Config                              | T3 infer            | S3Gen infer | Audio | **RTF** |
|-------------------------------------|--------------------:|------------:|------:|--------:|
| MTL, Metal Q4_0 + HiFT F16 v2 (§3.28) |  433 ms / 84 tok  |    706 ms   | 3.48 s| **0.33** |
| MTL, Metal Q4_0 baseline (§3.21 N=10) |  482 ms / 84 tok  |    730 ms   | 3.48 s|  0.35    |
| **Δ §3.21 → §3.28**                   |  **−49 ms / −10.2 %** | **−24 ms / −3.3 %** | — | **−0.02** |

WAV is byte-exact deterministic across runs (md5
`d8a1b22375dbcb2259c686426a7d76c5` ×5).  Parity harness
`test-metal-ops` passes 14 gates (3 base + 3 conv_transpose_1d + 8
fused `mul_mm`).  Patch `patches/ggml-metal-chatterbox-ops.patch`
(1088 lines) applies cleanly on a fresh ggml clone at pinned
`58c38058`.  All §3.24–§3.30 kernel changes cross-compile cleanly
for iOS-arm64 (portability verified; runtime measurement deferred
until an M4 / iPhone / iPad run of
[`scripts/bench-m4-validation.sh`](scripts/bench-m4-validation.sh)).

M3 Ultra CFM time specifically drops from 541.9 ms → 534.0 ms
(**−1.5 %**) — modest on this chip because per-dispatch overhead
is very low; expected to be larger on bandwidth-limited silicon
(M4 / A-series) where each saved `ggml_add` dispatch is worth more
relative to compute.

### Reference comparison vs onnxruntime (Multilingual, M4 CPU, F16)

Same prompt, seed, and reference audio fed through ONNX Runtime and
our ggml build back-to-back, 4 CPU threads on both.  ONNX Runtime's
multilingual export currently ships without the `text_emb_weight.bin`
tensor and emits `CFG disabled` at load — so its numbers are already
against a half-compute pipeline:

```
                     onnxruntime-fp16   ggml-cpu-f16⁴
  -------------------------------------------------
  cold load               42 829 ms        ~500 ms   (85x faster)
  inference wall          51 447 ms     10 168 ms   (5.06x faster)
  audio produced           2 740 ms      2 400 ms
  RTF                        18.78          4.24
  CFG enabled                  no           yes⁴
```

ggml is **5.06× faster per utterance and ~85× faster on cold load**,
while doing the full CFG pipeline (2 CFM estimator passes + 2 T3 passes
per step) that ONNX skips.  If the ONNX CFG path were wired up, its RTF
would roughly double and the gap would be ~10×.  Quality is comparable
— the two wavs (`bench-onnx.wav` / `bench-ggml.wav`) sound like the same
Spanish sentence in the JFK-cloned voice.

⁴ The ggml-cpu-f16 column above was captured during the same Apr 23–May 4
window where the non-`use_b2` CFM path was silently dropping the
unconditional pass on CPU; "CFG enabled: yes" overstated what was running
in this specific column.  After commit `6d9b42b` restored CFG on the CPU
path, the per-utterance ggml number on M4 CPU F16 should roughly double
(see footnotes ² and ³ above for the 2× ratio observed on standalone CPU
runs).  Re-running the bench against current `multilingual_merged`
will produce a corrected column; the ONNX side is unchanged so the
ratio should land near `~2.5–3× faster` rather than the historical
`5.06×`.  Cold-load (`~85×`) is a load-time figure and unaffected by
the runtime fix.

### Reproducing these numbers

```bash
# Build tts-cpp, then:
./build/tts-cli \
    --model       models/chatterbox-t3-turbo.gguf \
    --s3gen-gguf  models/chatterbox-s3gen.gguf \
    --reference-audio test/reference-audio/jfk.wav \
    --text "Hello from native C plus plus. This audio was generated end to end on CPU using ggml." \
    --out /tmp/bench.wav \
    --seed 42 \
    --n-gpu-layers 99   # 0 or omit for CPU
```

The binary prints both the per-stage timings and `BENCH:` lines that
scripts can scrape.  Note: the binary also prints an inner
`=== pipeline: … RTF=… ===` line — that RTF covers **only the
S3Gen + HiFT phase** (the timer around `s3gen_synthesize_to_wav`, which
runs after T3 is already done).  The tables above report the full
end-to-end number (T3_INFER + S3GEN_INFER).

`gen_RTF = (T3_INFER_MS + S3GEN_INFER_MS) / AUDIO_MS`

Token counts vary slightly across backends because the CPU-side
sampler reads logits that come out of different float-reduction orders
per backend; per-token T3 cost is the directly-comparable figure.
Full development history and older backend combinations (F16 vs
Q4_0 / Q5_0 / Q8_0, plus other machines) are in
[`PROGRESS.md §3.10 / §3.13`](PROGRESS.md).

### Streaming mode — low-latency playback

For interactive use cases, the binary can emit audio **chunk-by-chunk**
as it's generated instead of waiting for the whole sentence to finish.
Any non-zero `--stream-chunk-tokens N` turns streaming on.

**Flags:**

- `--stream-chunk-tokens N` — main knob; N speech tokens per chunk
  (25 ≈ 1 s of audio, 50 ≈ 2 s).
- `--stream-first-chunk-tokens N` — override the *first* chunk's size
  so first-audio-out lands early while later chunks stay big and keep
  overall RTF low.  Typical: 10.
- `--stream-cfm-steps N` — CFM Euler step count.  Default 2 (matches
  Python meanflow).  `1` halves CFM cost with a small quality penalty;
  Turbo's meanflow training makes 1-step a valid sampling mode per the
  paper.
- `--out -` — emit raw `s16le` mono @ 24 kHz to stdout instead of
  writing a wav file, so the output can be piped straight into a
  player.

**Recommended low-latency preset for interactive use:**

```bash
brew install sox      # one-time, for the `play` command

./build/tts-cli \
    --model      models/chatterbox-t3-turbo.gguf \
    --s3gen-gguf models/chatterbox-s3gen.gguf \
    --text       "Hello from streaming Chatterbox." \
    --stream-first-chunk-tokens 10 \
    --stream-chunk-tokens       25 \
    --stream-cfm-steps          1 \
    --n-gpu-layers              99 \
    --out - \
  | play -q -t raw -r 24000 -b 16 -e signed -c 1 -
```

`play` ships with `sox` and routes straight to CoreAudio.  If you
prefer, the same stdout stream works with `ffplay -f s16le -ar 24000
-ch_layout mono -nodisp -i -` or piped through a Python
`sounddevice.play()` one-liner; on some macOS 26 builds ffplay's SDL
output is silent for raw piped audio, so `sox play` is the safest
default.

You can also drop the `--out -` to get a regular wav:

```bash
./build/tts-cli … --stream-chunk-tokens 50 --out out.wav
afplay out.wav
```

**Latency and throughput** on an Apple M4 with the Metal backend and
the preset above, feeding the sentence *"Hello from streaming
Chatterbox, I am John and I work in Google since 2010. I love to go
out with my friends, eat some pizza and also drink some wine. I also
love to travel around the world alone."* (produces 317 speech tokens,
~12.7 s of audio):

| metric | value |
|---|---|
| first-audio-out latency | **279 ms** |
| chunk 1 (10-token bootstrap) | RTF 0.99 |
| chunks 2–13 (steady-state, 25 tokens each) | **RTF 0.30 – 0.63** |
| chunk 14 (tail finalise) | RTF 1.42 |
| total wall time | 11.5 s for 12.7 s of audio |
| overall RTF | **0.90** |

The steady-state RTFs stay comfortably below 1.0, so the streamer
sustainably pushes audio faster than real-time playback consumes it.
Chunk 1 is small by design so first audio lands in ~280 ms; the final
chunk is short and relatively slow (fixed encoder/CFM overhead
amortised over only 0.4 s of audio).

For the full journal of how streaming got there — bit-exact CFM parity,
`cache_source` + `trim_fade` port, `--out -` stdout wiring, per-chunk
tuning — see [`PROGRESS.md §B1`](PROGRESS.md).

## 4. Optional: validate against PyTorch

Every stage of the pipeline has a numerical regression test against
Python-dumped reference tensors:

```bash
./build/test-s3gen models/chatterbox-s3gen.gguf artifacts/s3gen-ref ALL
```

Expected output (rel error per stage):

```
Stage A  speaker_emb_affine    rel ≈ 1e-7
Stage B  input_embedded        rel = 0
Stage C  encoder_embed         rel ≈ 4e-7
Stage D  pre_lookahead         rel ≈ 3e-7
Stage E  enc_block0_out        rel ≈ 1e-7
Stage F  encoder_proj (mu)     rel ≈ 5e-7
Stage G1 time_mixer            rel ≈ 7e-7
Stage G2 cfm_resnet_out        rel ≈ 3e-7
Stage G3 tfm_out               rel ≈ 2e-7
Stage G4 cfm_step0_dxdt        rel ≈ 1e-6
Stage H1 f0                    rel ≈ 4e-6
Stage H3 conv_post             rel ≈ 6e-7
Stage H4 stft                  rel ≈ 8e-3 (boundary-bound)
Stage H5 waveform              rel ≈ 1e-4
```

For T3 bit-exact validation against the Python reference:

```bash
python scripts/reference-t3-turbo.py \
  --text "Hello from ggml." \
  --out-dir artifacts \
  --cpp-bin ./build/tts-cli \
  --cpp-model models/chatterbox-t3-turbo.gguf
```

## Repository layout

```
tts-cpp/                         in-tree subtree of github.com/gianni-cor/chatterbox.cpp
  include/tts-cpp/               installed public headers (Engine API)
    tts-cpp.h                    library entry; declares tts_cpp_cli_main()
    chatterbox/engine.h          Engine + EngineOptions (text → wav)
    chatterbox/s3gen_pipeline.h  low-level S3Gen + HiFT pipeline entries
  src/
    main.cpp                     T3 turbo runtime + shared helpers (libtts-cpp)
    t3_mtl.{h,cpp}               T3 multilingual (Llama-520M) runtime + stage builders
    chatterbox_t3_internal.h     internal T3 declarations shared by main/engine/CLI
    chatterbox_engine.cpp        public Engine API impl (libtts-cpp)
    chatterbox_cli.cpp           CLI entry (`tts-cli` binary)
    cli_main.cpp                 thin int-main forwarder; calls tts_cpp_cli_main()
    chatterbox_tts.cpp           S3Gen + HiFT pipeline        (libtts-cpp)
    mel2wav.cpp                  HiFT-only demo              (mel2wav)
    gpt2_bpe.{h,cpp}             self-contained GPT-2 BPE tokenizer (Turbo)
    mtl_tokenizer.{h,cpp}        multilingual grapheme tokenizer
                                   (HF tokenizers.json + NFKD lowercasing)
    mtl_unicode_tables.inc       embedded NFKD + Korean Jamo lookup tables

    voice_features.{h,cpp}       WAV I/O, sinc resampler, LUFS meter,
                                   24 kHz & 16 kHz log-mel extraction,
                                   Kaldi-style 80-ch fbank
    mel_extract_stft.cpp         STFT-based mel extraction shared by C++ pipelines
    voice_encoder.{h,cpp}        3-layer LSTM → 256-d speaker_emb
                                   (matches Resemble VoiceEncoder)
    campplus.{h,cpp}             FunASR x-vector port (FCM + 3× CAMDense
                                   TDNN) → 192-d embedding
    s3tokenizer.{h,cpp}          6-layer FSMN-attn transformer + FSQ →
                                   25-Hz speech tokens
    dr_wav.h                     vendored single-header WAV reader
    npy.h                        minimal .npy load / save + compare

    test_*.cpp                   per-stage numerical-parity harnesses
                                   (S3Gen / HiFT / streaming / MTL T3 /
                                    MTL tokenizer / voice features / Metal ops)
  scripts/
    synthesize.sh                text → wav wrapper around tts-cli
    convert-t3-turbo-to-gguf.py  Turbo T3 weights + GPT-2 BPE + VE + builtin
                                   voice → T3 GGUF (--quant)
    convert-t3-mtl-to-gguf.py    MTL T3 (Llama-520M) + perceiver + emotion-adv
                                   + tokenizers.json + builtin voice → T3 GGUF (--quant)
    convert-s3gen-to-gguf.py     S3Gen encoder + CFM + HiFT + CAMPPlus +
                                   S3TokenizerV2 + mel filterbanks → S3Gen GGUF
                                   (--variant {turbo,mtl}, --quant)
    requantize-gguf.py           in-place block-quantise of an existing
                                   T3/S3Gen GGUF (canonical deny-list lives here)
    extract-voice.py             one-shot voice-clone prep (silencedetect +
                                   codec-aware EQ + optional `--save-voice` bake)
    gen-nfkd-table.py            generates src/mtl_unicode_tables.inc
    dump-*-reference.py          PyTorch → .npy intermediates for the
                                   per-stage harnesses (S3Gen, CAMPPlus,
                                   S3TokenizerV2, streaming, MTL T3)
    reference-t3-turbo.py        PyTorch T3 bit-exact compare vs C++
    compare-tokenizer.py         10-case BPE tokenizer compare vs HF
  patches/
    ggml-metal-chatterbox-ops.patch   ggml-metal fixes
    ggml-opencl-chatterbox-ops.patch  OpenCL: HiFT (CONV_TRANSPOSE_1D, SIN, …)
    README.md                    applies-to / what-it-does notes
  voices/                        baked voice profiles (not tracked; populated
                                   by --save-voice)
  models/                        generated GGUFs (not tracked)
  artifacts/                     .npy dumps for validation (not tracked)
  CMakeLists.txt                 top-level build
  README.md                      this file
  PROGRESS.md                    chronological development journal
```

## Troubleshooting

**`error: this GGUF has no embedded tokenizer`** — you're running against
a legacy T3 GGUF built before the tokenizer was embedded. Re-run the
converter to produce a fresh GGUF:

```bash
python scripts/convert-t3-turbo-to-gguf.py --out models/chatterbox-t3-turbo.gguf
```

**`warning: s3gen GGUF lacks variant keys`** — you're running against a
legacy S3Gen GGUF produced before the variant metadata was added in
§3.19/§3.20. The defaults (`meanflow=true, n_timesteps=2, cfg_rate=0`)
match the historical Turbo behaviour, so legacy Turbo GGUFs continue
to work.  For a Multilingual S3Gen GGUF, however, those defaults are
wrong and the output will be garbage — re-run the converter:

```bash
python scripts/convert-s3gen-to-gguf.py --variant mtl --out models/chatterbox-s3gen-mtl.gguf
```

**`error: --min-p must be in [0, 1]`** / `--cfg-weight must be >= 0` /
`--exaggeration must be in [0, 1]` — the MTL sampling knobs reject
out-of-range values up front instead of producing wrong-but-not-crashing
output.  Pass values inside the documented ranges (see "Run" above).

**`--debug requires --ref-dir`** — debug mode substitutes Python-dumped
random bits to make every intermediate tensor bit-exactly comparable.
Run `python scripts/dump-s3gen-reference.py --out artifacts/s3gen-ref …`
first, then pass `--ref-dir artifacts/s3gen-ref`.

**Output is much louder than the Python reference** — expected: the Python
reference dump uses a very short utterance (mostly silence). Generate a
longer sentence and compare RMS. Differences up to ~2.5 % in spectrogram
magnitude are from the stochastic SineGen excitation (non-bit-exact RNG
between `std::mt19937` and `torch.rand`).

**Slower than real-time** — make sure you built `-DCMAKE_BUILD_TYPE=Release`
and that `--threads` picks up all your cores. The binary defaults to
`std::thread::hardware_concurrency()`.

## License

Released under the [MIT License](LICENSE) — Copyright (c) 2026 Gianfranco
Cordella. The bundled `ggml/` is also MIT-licensed
([ggml/LICENSE](ggml/LICENSE)). The upstream Python implementation
([Chatterbox](https://github.com/resemble-ai/chatterbox), Copyright (c) 2025
Resemble AI) is likewise MIT-licensed; see `LICENSE` for the third-party
attribution block.
