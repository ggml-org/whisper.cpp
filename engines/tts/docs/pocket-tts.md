# tts engine: Pocket TTS

Part of the [tts documentation](../README.md).

Pocket provides English text-to-speech through a persistent CPU, GPU or Hexagon
FlowLM/Mimi engine and incremental 24 kHz mono PCM callbacks. Its public API is
`<tts-cpp/pocket/engine.h>`; `pocket-cli` uses that API. F32 arithmetic is used
throughout. F16 is a storage option whose weights expand to F32 at load.

Prepared voices are checkpoint-specific. The tested released checkpoint has
a disabled reference encoder, so voice cloning requires separate
encoder-enabled weights; random-weight encoder parity does not establish
cloning quality. Each instance admits one synthesis at a time. Callbacks run
on the calling thread; returning false or calling `cancel()` ends streaming.
Reentrant/overlapping synthesis throws. Each request resets voice, codec and
RNG; native seeds are not PyTorch-equivalent.

## Conversion

Run these commands from the repository root. Prepare the conversion environment:

```sh
python3.12 -m venv /tmp/pocket-venv
/tmp/pocket-venv/bin/pip install -r engines/tts/scripts/pocket-flow-requirements.txt
```

Python 3.11+ and PyTorch are conversion/reference dependencies only. Use an
installed speech-branch ggml as described in the [build guide](../../../docs/BUILD.md).

For a complete bundle, supply explicit local upstream assets. The output
directory must not already exist; it is published after all stages succeed.

```sh
/tmp/pocket-venv/bin/python engines/tts/scripts/convert-pocket-to-gguf.py \
  --weights /path/to/model.safetensors --config /path/to/config.yaml \
  --tokenizer /path/to/tokenizer.model --voice /path/to/alba.safetensors \
  --output /path/to/pocket-bundle
GGML_PREFIX=/path/to/ggml-install
cmake -S engines/tts -B build/pocket -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$GGML_PREFIX" -DTTS_CPP_BUILD_EXECUTABLES=ON
cmake --build build/pocket --target pocket-cli
build/pocket/pocket-cli --model-dir /path/to/pocket-bundle \
  --text "Hello from Pocket TTS." --output /tmp/pocket.wav
build/pocket/pocket-cli --model-dir /path/to/pocket-bundle --gpu \
  --text "Hello from Pocket TTS on Metal." --output /tmp/pocket-metal.wav
```

The bundle contains `flow-lm.gguf`, `mimi.gguf`, `frontend.json`, `voice.gguf`
and a SHA-256 manifest. Use a prepared voice from the same upstream checkpoint
revision as the weights: legacy voice files do not contain a checkpoint hash,
so the converter binds the caller-selected assets, rather than proving the
provenance of arbitrary voice files. Native loading rejects mixed bundle
hashes and invalid cache lengths, types or payloads.

The identity SentencePiece unigram frontend includes byte fallback, Unicode
case/whitespace/digit metadata, punctuation repair and decimal-aware sentence
splitting. It matches 322 upstream corpus cases. Generate the fixture using
`dump-pocket-frontend-reference.py --tokenizer /path/to/tokenizer.model
--output /path/to/pocket-bundle/frontend-corpus.json`, then configure
`TTS_CPP_POCKET_MODEL_DIR` to enable frontend and public-engine CTest cases.

The persistent engine restores the chosen voice for each text chunk and runs
FlowLM/Mimi on separate workers with at most 16 queued latent frames.
CPU remains the default. Set `EngineOptions::use_gpu = true` or pass `--gpu`
to select Metal, CUDA, Vulkan or OpenCL for both the language model and the codec,
including their persistent weights and caches. A supported GPU backend must be available;
the request fails explicitly otherwise. `backend_name()` and CLI JSON reports
identify the selected backend. `backends_dir` (CLI: `--backends-dir`) locates
dynamically loaded ggml backends using the library's existing process-wide
loader. For backend-specific validation, set `TTS_CPP_GPU_BACKEND` to
`metal`, `cuda`, `vulkan` or `opencl`; the selected backend must match the request.
The Fabric seed uses a specified Box–Muller transform. An encoder-enabled
checkpoint can use `--reference-audio` instead of the prepared voice; the
tested public checkpoint rejects that path.

## OpenCL execution

`EngineOptions.backend = "opencl"` (CLI: `--backend opencl`) explicitly
selects OpenCL; `"cpu"` explicitly selects CPU. Empty/`"auto"` follows `use_gpu`.
Explicit OpenCL requests fail if OpenCL is unavailable. Automatic GPU selection
also supports Metal, CUDA and Vulkan, retaining the registry's Adreno tier
policy. Hexagon is selected explicitly as described below. Both FlowLM and Mimi
use the same selected device, retaining
F32 weights and persistent caches on that device. Prepared voices and small
conditioning/output values still cross the host boundary.

Use an installed ggml-speech build with `GGML_OPENCL=ON` and its OpenCL backend
available to the existing registry loader. Dynamic builds must ship the
backend libraries beside the application, or set the public backend directory
before constructing the first engine (see `<tts-cpp/backend.h>`).

```sh
build/pocket/pocket-cli --model-dir /path/to/pocket-bundle --backend opencl \
  --text "Hello from Pocket TTS." --output /tmp/pocket-opencl.wav \
  --runs 3 --report /tmp/pocket-opencl.json
```

The CLI prints both stage backends and the device description; its JSON records
`flow_lm_backend`, `mimi_backend`, and `device`. Native callers can inspect
`flow_lm_backend_name()`, `mimi_backend_name()`, and `device_name()`.
Every graph checks backend operation support before allocation or execution;
an incompatible ggml build reports the unsupported operation instead of
executing a partial graph or falling back to CPU.

FlowLM supplies an explicit causal mask to OpenCL softmax. This avoids the
reviewed ggml-speech `DIAG_MASK_INF` vector kernel, whose lane writes corrupt
neighboring vectors at attention widths divisible by eight. The existing
64-position full-context/chunked-prefill regression exposed this on Adreno 830;
CPU retains its original graph. Preflight prices both the device mask and its
host upload staging.

The ggml OpenCL backend shares kernel arguments, scratch and lifecycle state
between handles on a device. Pocket serializes its OpenCL submissions,
initialization and backend release across its workers and engine instances.
Each worker still owns its weights, persistent state and graph arena, so
their memory can overlap. CPU workers remain concurrent. This Pocket gate
does not cover another Fabric engine using the same ggml OpenCL context;
applications must serialize such mixed-engine use until ggml provides that
protection itself.

## Hexagon execution

`EngineOptions.backend = "hexagon"` (CLI: `--backend hexagon`) explicitly
selects Qualcomm Hexagon `HTP0`; `"HTP0"` is an equivalent alias. Both stages use
the selected session's RPC buffers and execute their complete graphs on the
DSP. CPU remains the default, and `--gpu` uses the GPU selection described above.
An unavailable HTP runtime or unsupported graph operation is an error with no CPU
fallback. The stage backend names must both report `HTP0`.

Use an Android arm64 build of the speech ggml dependency with
`GGML_HEXAGON=ON`, built using matching Hexagon SDK, Hexagon Tools and Android
NDK versions. The reference toolchain is SDK 6.6.0.0, Tools 19.0.07 and NDK r28b.
Ship `libggml-hexagon.so` with the ggml libraries and the architecture-matched
`libggml-htp-v*.so` DSP skeletons from that same build. The loader extends
`ADSP_LIBRARY_PATH` from the configured backend directory. Android FastRPC
access and compatible vendor `libcdsprpc.so`/drivers are required; detecting a
Snapdragon GPU does not establish DSP availability.

```sh
build/pocket/pocket-cli --model-dir /path/to/pocket-bundle --backend hexagon \
  --text "Hello from Pocket TTS." --output /tmp/pocket-hexagon.wav \
  --runs 3 --report /tmp/pocket-hexagon.json
```

The baseline retains expanded F32 weights, F32 caches and `GGML_PREC_F32`
matmul requests, which select HVX computation. HMX/FP16 acceleration and
quantization are separate tuning work. FlowLM uses an explicit causal mask
because Hexagon has no `DIAG_MASK_INF` operation, and lowers `MEAN` into F32
`SUM_ROWS` followed by division by the row width. Mimi retains its existing
tap-wise F32 convolution graphs and explicit sliding-window mask.

The HTP backend shares one queue, graph cache and mapping state per registered
device. Pocket serializes HTP graph construction through buffer destruction,
weight loading, tensor transfers, reset and backend release across all Pocket
workers and instances. A recursive gate permits FlowLM's nested projection and
attention graphs. CPU workers remain concurrent. Other Fabric engines using
the same HTP session need application-level serialization until ggml supplies
backend-wide protection. Different processes have separate sessions.

Pocket rejects a buffer type whose packing limit exceeds the runtime's
effective RPC payload cap, before model allocation or memory preflight. This
also covers packed weights/state and graph arenas containing several valid
tensors. With the current 2 GiB mapping limit and 4 KiB guard,
`GGML_HEXAGON_MBUF=2048` or larger is incompatible; leave the 1024 MiB default
or use at most 2047 MiB. Set this before backend initialization: the retained
session and buffer type settings, rather than the current environment, are
checked.

Enable `TTS_CPP_POCKET_HEXAGON_TESTS=ON` with the three Pocket fixture directories
to register explicit Hexagon FlowLM F32/F16, Mimi, fit and engine tests. These
reuse the existing CPU tolerances and cover cached/chunked prefill, Mimi's
sliding-window boundary, streaming, cancellation, reuse and parallel instances.
Missing fixture files remain visibly disabled; an enabled hardware test fails
if Hexagon is unavailable. Hardware validation and timing results must be
reported separately from successful host builds.

`frames_after_eos` (CLI: `--frames-after-eos`) accepts an explicit 0–100-frame
tail, or -1 for the checkpoint/text default. Synthesis and memory preflight
reserve this tail in addition to the text-duration estimate. All chunks must
fit the context before any PCM is emitted; detecting EOS near the estimate's
end still permits the complete tail. If EOS never arrives within the original
text estimate, generation fails rather than spending the reserved tail.

`scripts/convert-pocket-flow-lm-to-gguf.py` takes a local Kyutai checkpoint and
its YAML configuration. It selects `flow_lm.*` tensors from the safetensors
bundle, leaving Mimi out. BF16 source weights expand to F32. Unknown/missing
FlowLM weights, unsupported flow types, and invalid dimensions fail explicitly.
The output is replaced only after a successful complete export.

GGUF architecture: `pocket-tts-flow-lm`; `pocket.schema_version = 1`.
Keys under `pocket.*` record dimensions, RoPE base, BOS policy, flow type,
source SHA-256, and the input configuration. Tensor names retain upstream
names with the `flow_lm.` prefix removed. PyTorch tensor dimensions are reversed
by GGUF without transposing data. `bos_before_voice` is flattened to a vector.
This format is specific to Fabric's FlowLM stage and is not interchangeable
with llama.cpp's full Pocket TTS model/mmproj files.

```sh
/tmp/pocket-venv/bin/python engines/tts/scripts/convert-pocket-flow-lm-to-gguf.py \
  --weights /path/to/model.safetensors --config /path/to/config.yaml \
  --output /path/to/flow-lm-f32.gguf
```

The C++ core does not load Python, safetensors, ONNX Runtime, or a second ggml copy.


## Reproduce numerical validation


The fixture generator calls Kyutai's actual neural modules at commit
`0c2db3bdea7c991c568989cc11b503f14483fabc`, pinned by the requirements file.
It verifies that revision before generating data. Its default small-weight
fixture needs no downloaded model or tokenizer. Its PyTorch oracle recomputes
full prefixes while the native implementation uses cached decoding.

```sh
/tmp/pocket-venv/bin/python engines/tts/scripts/dump-pocket-flow-reference.py \
  --output /tmp/pocket-reference
/tmp/pocket-venv/bin/python engines/tts/test/pocket/test_converter.py /tmp/pocket-reference

# GGML_PREFIX must contain an installed qvac-ext-ggml speech build compatible
# with the rest of this checkout, including the CPU registry graph-planner export.
cmake -S engines/tts -B build/pocket -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$GGML_PREFIX" -DTTS_CPP_BUILD_EXECUTABLES=OFF \
  -DTTS_CPP_BUILD_TESTS=ON -DTTS_CPP_POCKET_REFERENCE_DIR=/tmp/pocket-reference
cmake --build build/pocket --target test-pocket-flow-lm test-pocket-load
ctest --test-dir build/pocket -R test-pocket- --output-on-failure
```

The native tests check embedding lookup, prefill, four cached audio steps,
EOS, 1/2/4/16 flow steps, denormalization, chunked prefill, reset, invalid
inputs, and a trajectory that feeds native sampled latents back into the LM.
The loader test needs no model files. Numerical tests are visibly disabled
by CMake when fixtures are absent; no reference downloads occur during build.

On a configured OpenCL host, set `-DTTS_CPP_POCKET_OPENCL_TESTS=ON` and all three
fixture directories to add explicit OpenCL runs of the existing FlowLM F32/F16,
Mimi, public engine, and metadata-only fit tests. These retain the CPU numerical
tolerances and cover cache/context boundaries, streaming, reset, cancellation,
repeated synthesis and error recovery. A missing OpenCL backend fails an enabled
OpenCL test; it cannot pass through CPU fallback. The fixture-free backend test
also checks the CPU default, allowlist and unavailable-OpenCL behavior.

For example, on the target host after installing its compatible ggml build:

```sh
cmake -S engines/tts -B build/pocket -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$GGML_PREFIX" -DTTS_CPP_BUILD_TESTS=ON \
  -DTTS_CPP_POCKET_OPENCL_TESTS=ON \
  -DTTS_CPP_POCKET_REFERENCE_DIR=/path/to/pocket-reference \
  -DTTS_CPP_POCKET_MIMI_REFERENCE_DIR=/path/to/pocket-mimi-reference \
  -DTTS_CPP_POCKET_MODEL_DIR=/path/to/pocket-bundle
cmake --build build/pocket --parallel
ctest --test-dir build/pocket -R 'test-pocket-.*opencl' --output-on-failure
```

For a real checkpoint, use the same generator with `--weights` and `--config`:

```sh
/tmp/pocket-venv/bin/python engines/tts/scripts/dump-pocket-flow-reference.py \
  --weights /path/to/model.safetensors --config /path/to/config.yaml \
  --output /tmp/pocket-real-reference
build/pocket/test-pocket-flow-lm /tmp/pocket-real-reference/flow-lm-f32.gguf \
  /tmp/pocket-real-reference 0.0001
```


For real-checkpoint and complete bundle tests, supply the fixtures expected
by `TTS_CPP_POCKET_REFERENCE_DIR`, `TTS_CPP_POCKET_MIMI_REFERENCE_DIR` and
`TTS_CPP_POCKET_MODEL_DIR`, then reconfigure. Inspect `ctest -N` and distinguish
disabled tests from executed tests. Exact test counts are revision-dependent.

## Native memory preflight


`tts-cpp/pocket/fit.h` exposes `pocket::fit_params(FitOptions)`. It tokenizes the
requested text, validates voice/context capacity, and prices the actual FlowLM
and Mimi graphs with ggml's size-only allocator on the selected CPU or GPU
backend. CPU projections additionally include the CPU work-buffer planner.
That planner is resolved through the CPU backend's registry, so preflight
works when CPU variants are dynamically loaded (Linux/Android).
CPU projection requires the `ggml_graph_plan` registry export from
[qvac-ext-ggml#92](https://github.com/tetherto/qvac-ext-ggml/pull/92);
older backends report an explicit preflight error instead of omitting scratch memory.
GGUF tensor payloads are never read and model/graph buffers are never allocated
or executed during preflight. Backend initialization can allocate its own scratch.
The estimate includes expanded F32 weights, state, overlapping worker compute,
host containers, thread stacks, load staging, and native batch PCM buffers.
It excludes application and SDK/JS copies. This is a memory projection, not a
validation of tensor contents, checkpoint encoder capability, or speech quality.

OpenCL fit uses the same selection rules as synthesis. Adreno OpenCL is treated
as shared system memory even though ggml advertises it as a GPU. Other devices
use ggml's shared-memory classification; discrete devices must fit both device
and host memory independently. Host availability is sampled before backend
initialization to charge that allocation once. OpenCL's reported free memory
is a backend estimate/cap, currently global memory minus 1 GiB in the reviewed
speech backend; on Adreno the result is also limited by OS available RAM.
Missing/invalid memory telemetry produces a preflight error.

The OpenCL state total includes one conservative shared-backend scratch bound,
reported separately in the text report. Its constants and F32 dispatch audit
come from qvac-ext-ggml speech `06f1b668`: the Adreno initialization pools total
395,149,312 bytes before device allocation caps. The estimate also bounds
independently retained weight/activation transpose and strided GEMM input
pools, including old allocations retained by queued work during growth.
Builds without those Adreno pools can therefore be overestimated. Re-audit
these constants and F32 dispatch paths when updating ggml; driver/kernel
compiler overhead is implementation-dependent and covered by the requested
margin rather than an exact measurement. This is not a live allocation trace.

Hexagon preflight uses the versioned ggml registry procedure
`ggml_backend_hexagon_get_runtime_info_v1`. Older builds without this procedure
fail explicitly for Hexagon, while CPU and OpenCL remain usable. The query
reads retained session configuration, so changing environment variables after
backend initialization cannot silently underprice existing queues or tracing
buffers. A configuration that disables HTP queueing or compute is rejected.

HTP RPC buffers share OS RAM; its zero-valued device telemetry is replaced by
OS available RAM sampled before backend loading. Backend-reported bounds cover
all initialized sessions' RPC queues and retained/prospective graph metadata.
Model and state estimates include backend tensor alignment, a conservative RPC
allocation count, guard pages and host page rounding. Each graph arena reserves
16 chunks' guard/page overhead, matching the reviewed allocator, and both
workers' arenas are counted concurrently. Individual tensors must fit the
selected session's buffer limit; the conservative sum of live model buffers and
arenas must fit its batch mapping limit. VTCM is reported separately and graph
capability checks enforce the backend's available VTCM constraints. VTCM is not
charged as ordinary OS RAM. Opaque driver/DSP heaps and app copies still require
the requested safety margin. Re-audit graph chunking when updating ggml.

```sh
build/pocket/pocket-cli --model-dir /path/to/pocket-bundle \
  --text "Hello from Pocket TTS." --fit --fit-margin-mib 256 \
  --fit-budget-mib 1024 --report /tmp/pocket-fit.json
build/pocket/pocket-cli --model-dir /path/to/pocket-bundle --gpu \
  --text "Hello from Pocket TTS." --fit --report /tmp/pocket-metal-fit.json
```

The optional budget further limits available OS memory. On iOS, the projection
also respects `os_proc_available_memory()` when supported; a zero app allowance
returns does-not-fit. Linux uses MemAvailable; container/cgroup limits are not
automatically detected, so hosts must supply their application ceiling. The CLI
returns 0 for fit, 1 for no-fit, and 2 for a preflight error. Native callers receive
`FitStatus` and a reason/report. Fit remains a point-in-time estimate; another
process can consume memory between preflight and loading.

For an OpenCL preflight add `--backend opencl` to that command. For device
benchmarks, run the CPU and OpenCL commands with identical assets, text,
threads, steps and seed. Record the model hashes, phone/GPU, OS and driver,
build options, load time, first-run and warm first-audio latency/RTF, and peak
host/device memory. `--runs 1` records the first synthesis; `--runs 3` performs
one warmup and records three warm runs. Validate output with the existing
`scripts/validate-pocket-audio.py` workflow and listen for defects. Hardware
correctness, audio quality and speedup require measured results on the target
device; host CPU checks do not establish them.


Reference input is bounded before model allocation. Accepted reference input is uncompressed
PCM8/16/24/32 or float32/64 WAV, 1–64 channels, 8–192 kHz, at most 30 seconds and
64 MiB on disk.

## Performance and integration evidence

The [OnePlus 13s OpenCL report](history/pocket-opencl-adreno.md) records
CPU/OpenCL correctness, speech checks, first-use and warm timings, and memory
preflight on Adreno 830. Both stages passed device validation. It retains the
initial timings and an expanded 60-request comparison, which observed a
1.08–1.26x median OpenCL throughput advantage with substantial run variability.
The report includes the standalone Android build and driver setup used for
reproduction.

The [archived integration report](history/pocket-integration.md#final-registry-build-and-release-review)
records the September 2026 pinned-registry Apple M2 comparison: generation
5–16% above upstream, native RTF about 0.18–0.19, and first audio 74–116 ms.
Those warm-cache measurements supersede the earlier manual-build slowdown;
SDK startup and transport latency are additional.

The archive also records native, addon, Node SDK socket transport and iOS
Simulator/worklet validation at specific source pins. Physical-device,
Android and broader voice/listening coverage remain limitations of that
record; it does not establish current package publication or platform parity.
For public SDK usage and current package versions, see
[QVAC](https://github.com/tetherto/qvac).
