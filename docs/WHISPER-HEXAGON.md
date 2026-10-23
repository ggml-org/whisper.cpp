# Native Whisper on Snapdragon Hexagon

Whisper uses the Hexagon backend from the external shared
[`qvac-ext-ggml` speech branch](https://github.com/tetherto/qvac-ext-ggml/tree/speech).
Build the host backend and its DSP skeletons from one immutable revision.
The `third_party/whisper.cpp/ggml` copy is not the Speech dependency and must
remain unchanged. This guide covers native Fabric builds and validation.

## Build Android arm64

Install an Android NDK, Qualcomm Hexagon SDK and matching Hexagon Tools. Set
`ANDROID_NDK`, `HEXAGON_SDK_ROOT` and `HEXAGON_TOOLS_ROOT` to those installations.
The commands below use NDK r28b, SDK 6.6.0.0 and Tools 19.0.07. OpenCL also
requires Khronos headers and an Android arm64 `libOpenCL.so` compatible with
the target device. Set `OPENCL_HEADERS` and `OPENCL_LIBRARY` accordingly; a
desktop OpenCL library cannot be used for this cross build.

From the Speech repository root:

```sh
git clone --branch speech https://github.com/tetherto/qvac-ext-ggml ggml-src
git -C ggml-src checkout a6122941d1d851833ccff7200aac205e92d481e0
git -C ggml-src rev-parse HEAD

cmake -S ggml-src -B build-ggml-android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_HEXAGON=ON -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON \
  -DGGML_OPENMP=OFF -DGGML_BUILD_TESTS=ON -DGGML_BUILD_EXAMPLES=OFF \
  -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
  -DHEXAGON_TOOLS_ROOT="$HEXAGON_TOOLS_ROOT" \
  -DPREBUILT_LIB_DIR=android_aarch64 \
  -DOpenCL_INCLUDE_DIR="$OPENCL_HEADERS" -DOpenCL_LIBRARY="$OPENCL_LIBRARY" \
  -DCMAKE_INSTALL_PREFIX="$PWD/ggml-android-install"
cmake --build build-ggml-android --parallel 4
cmake --install build-ggml-android

cmake -S . -B build-whisper-android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_FIND_ROOT_PATH="$PWD/ggml-android-install" \
  -DSPEECH_BUILD_PARAKEET=OFF -DSPEECH_BUILD_TTS=OFF \
  -DSPEECH_BUILD_AUDIOGEN=OFF -DSPEECH_BUILD_EXECUTABLES=ON \
  -DCMAKE_INSTALL_PREFIX="$PWD/whisper-android-install"
cmake --build build-whisper-android --parallel 4
cmake --install build-whisper-android
```

`PREBUILT_LIB_DIR` is needed by SDK 6.6's CMake helpers. Android package
discovery uses `CMAKE_FIND_ROOT_PATH` because the NDK restricts package lookup
to its cross-compilation roots. The installed libraries use the
`libqvac-speech-ggml*` prefix. The unprefixed `libggml-htp-v*.so` libraries
execute on the DSP. This source builds v73, v75, v79 and v81 skeletons; the
runtime selects the detected architecture. Preserve the matching skeleton
and host libraries together, and record the detected architecture in results.

## Stage and select a device

Example with an already connected ADB device (`ANDROID_SERIAL` may select it):

```sh
adb shell mkdir -p /data/local/tmp/whisper-hexagon/lib
adb push ggml-android-install/lib/. /data/local/tmp/whisper-hexagon/lib/
adb push whisper-android-install/lib/. /data/local/tmp/whisper-hexagon/lib/
adb push whisper-android-install/bin/whisper-cli /data/local/tmp/whisper-hexagon/
adb push whisper-android-install/bin/whisper-fit-params /data/local/tmp/whisper-hexagon/
adb push "$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" /data/local/tmp/whisper-hexagon/lib/
adb push third_party/whisper.cpp/samples/jfk.wav /data/local/tmp/whisper-hexagon/
adb push /path/to/ggml-tiny.en.bin /data/local/tmp/whisper-hexagon/
adb shell
```

Inside the Android shell:

```sh
cd /data/local/tmp/whisper-hexagon
export LD_LIBRARY_PATH="$PWD/lib:/system/lib64:/vendor/lib64"
export ADSP_LIBRARY_PATH="$PWD/lib;/vendor/lib/rfsa/adsp;/dsp"
chmod +x whisper-cli whisper-fit-params
./whisper-cli --list-devices
./whisper-fit-params --list-devices
./whisper-cli -m ggml-tiny.en.bin -f jfk.wav -nt
./whisper-cli --device HTP0 -m ggml-tiny.en.bin -f jfk.wav -nt
./whisper-cli --no-gpu -m ggml-tiny.en.bin -f jfk.wav -nt
```

Use the exact listed OpenCL device name for the GPU comparison. Numeric
`--device N` / `-dev N` remain supported and count only GPU/IGPU devices,
not all registered devices. `whisper-fit-params` also retains `--gpu-device`.
`WHISPER_ARG_DEVICE` accepts the same name/index; an explicit flag overrides
it and `--no-gpu` ignores it. An explicit device flag combined with
`--no-gpu` is an error. Without a flag or environment target, both native tools
prefer registered `HTP0`, then retain the existing GPU-index-0 default when HTP
is absent or its name is ambiguous. Thus a Hexagon-enabled Snapdragon build with
matching DSP libraries uses HTP automatically. Automatic initialization retains
the existing CPU fallback; explicit HTP selection below rejects initialization
fallback. The C API's numeric-device contract is unchanged; library callers
choose the desired GPU ordinal themselves.

`GGML_HEXAGON_NDEV` already defaults to `1`, exposing one logical HTP device,
`HTP0`; it does not choose operators or count DSP cores. Keep the default
`GGML_HEXAGON_HOSTBUF=1` for Whisper. The focused operator-only run with
`NDEV=1 HOSTBUF=0` exercises another storage mode, not the full-model default.

An explicit CLI target must be registered and must initialize successfully;
initialization falling back to another primary backend is an error. The
additive native API `whisper_get_primary_backend_name(ctx)` reports the first
initialized device of the default state. It returns null when there is no
default state. Its name does not establish execution of every graph node.
`--no-gpu` retains Whisper's existing automatic ACCEL policy; inspect the
actual backend when a CPU-only measurement is required. HTP is registered as
a GPU device in this shared dependency, so CPU mode excludes HTP.

Missing `libcdsprpc.so`, inaccessible DSP skeletons, incompatible host/DSP
revisions and device FastRPC restrictions must be reported as initialization
or deployment failures. A CPU fallback is not successful HTP validation.
Keep system libraries ahead of vendor libraries in `LD_LIBRARY_PATH`: on the
validated Android 16 device, vendor-first lookup selects an incompatible
Binder dependency and prevents OpenCL/Hexagon initialization.

## Confirm execution and correctness

Use existing scheduler and HTP diagnostics in a separate, untimed run:

```sh
WHISPER_LOG_DEBUG=1 GGML_SCHED_DEBUG=2 \
GGML_HEXAGON_PROFILE=3 GGML_HEXAGON_OPTRACE=65536 \
./whisper-cli --device HTP0 -m ggml-tiny.en.bin -f jfk.wav -nt \
  >transcript.txt 2>placement.log
```

`WHISPER_LOG_DEBUG=1` exposes debug messages through Whisper's default logger
in release builds. Scheduler lines show actual node assignment and split
inputs/copies; DSP profiling records executed operations. Inspect convolution,
encoder, cross projection and decoder separately. Unsupported operations may
run on CPU. Report those placements, rather than treating backend discovery,
allocated buffers or process exit code as proof of DSP execution. A caller
that installs its own log callback controls its own debug filtering.

Use the same matching shared-GGML build's `test-backend-ops` for relevant
operator comparisons against CPU. Select cases available in that revision;
record full commands/output, return codes and passed/failed/unsupported counts.
Do not change numerical tolerances to make the backend pass.

Model validation uses real tiny/base checkpoints, both English and multilingual,
F16 and Q8_0, fixed reference transcripts, silence, zero samples and audio longer
than 30 seconds. Whisper consumes its GGML `.bin` models, not a new GGUF graph.
Create Q8_0 with the matching host `whisper-quantize` tool:

```sh
./build/bin/whisper-quantize ggml-tiny.en.bin ggml-tiny.en-q8_0.bin q8_0
```

Record source/model/audio hashes and identical decoding parameters for
CPU/OpenCL/HTP. Report corpus WER with a declared normalizer; the validation
gate is expected transcript fixtures passing and HTP WER no more than
1.0 absolute percentage point above the same-model/quant CPU baseline.
Committed weightless fixtures cover contracts, not recognition accuracy.
Build success alone does not establish model support on a device.

With the shared backend's default host buffers, canonical quantized bytes can
be read safely by CPU fallback but are not the tiled layout required by HTP
quantized matmul. For quantized weights used by `MUL_MAT`, Whisper now tries
the selected GPU/IGPU's advertised extra buffers and checks type/shape support
with that buffer attached. The existing Hexagon `HTP0-REPACK` extra is non-host;
the normal tensor-upload API repacks those weights without changing the model
file. Unsupported projections retain the existing canonical fallback placement.
F16/F32 weights, quantized `GET_ROWS`/tied token embeddings, VAD and OpenCL's
default placement are unchanged. The memory-fit preflight uses the same
selection path as real loading; compare projected and actual bytes on HTP
after a buffer-layout change. A primary `HTP0` label alone does not prove that
Q8 matmul executed on the DSP: confirm it with the diagnostic run above.

Memory fit requires the paired shared-GGML allocation-size callback: context
and scheduler measurement apply it to each actual payload chunk, including
the HTP guard without allocating tensor payloads. Repeated buffer types are
charged once. The older `a6122941` producer excluded guards and produced
8/12/16 KiB weight/KV/compute mismatches for tiny/base Q8; retain those logs
as the before-fix evidence, not as a successful parity check.

HTP declares `memory_unified`: FastRPC buffers and host extras compete for
system RAM. When HTP has no separate capacity estimate, fit uses the CPU
backend's OS available/total RAM query and compares it with the combined
device/host requirement. Known capacity with zero available bytes remains
memory pressure; unavailable capacity is an error. Prospective Hexagon
metadata growth is added to the host budget, while retained session queues
and metadata are already reflected in available RAM. The memory margin still
covers unknown future driver/DSP heaps. This is an estimate, not an allocation
guarantee or a claim that all available RAM can be mapped at once. Validate
both component byte parity and the shared-memory verdict on the real device
after updating the producer; CPU parity alone does not establish either.

`GGML_HEXAGON_HOSTBUF=0` can make a full Whisper graph unschedulable even when
its individual matmuls pass: a preallocated non-host KV destination may need
a copy operation unsupported by HTP. A rejected scheduler measurement is an
Error (`measurement-failed`), with no partial compute budget accepted as a
fit. This configuration is not a supported Whisper fallback path merely
because its focused backend operator checks pass.

In the focused 73-instance operator selection, `NDEV=1 HOSTBUF=0` passes 65
cases and reports eight unsupported cases: four F16 GELU cases (two shapes,
each contiguous and viewed) and four F32 NORM cases with permuted rows (two
shapes, each at epsilon `1e-6` and `1e-4`). The shared HTP unary contract accepts
F32 input/output and rejects permuted inputs. These are unsupported type/layout
combinations, not failed numerical comparisons; the corresponding supported
F32 GELU and ordinary-row NORM cases pass. Report unsupported cases separately
from actual HTP execution and CPU fallback.

## Measure transcription time

The standalone installed-library consumer in
[`scripts/benchmarks/whisper-hexagon`](../scripts/benchmarks/whisper-hexagon/CMakeLists.txt)
loads one model and emits JSONL around native `whisper_full` calls. Build it
for Android against the same installed Whisper and GGML:

```sh
cmake -S scripts/benchmarks/whisper-hexagon -B build-whisper-measure \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_FIND_ROOT_PATH="$PWD/whisper-android-install;$PWD/ggml-android-install"
cmake --build build-whisper-measure --parallel 4
adb push build-whisper-measure/whisper-hexagon /data/local/tmp/whisper-hexagon/
```

Run inside the prepared Android shell, substituting the GPU ordinal listed
by `whisper-cli --list-devices` (`HTP_INDEX` below):

```sh
./whisper-hexagon --model ggml-tiny.en.bin --audio jfk.wav \
  --device "$HTP_INDEX" --threads 4 --language en \
  --warmup 1 --runs 3 --duration-seconds 300 >htp.jsonl 2>htp.log
```

Run the CPU reference with `--no-gpu`, and OpenCL with its listed GPU ordinal.
The driver accepts the CPU backend or the known host BLAS backend (registered
as ACCEL on macOS), retaining the actual name in every record. Other ACCEL
devices are not accepted as CPU references. Use a mono
16 kHz PCM16 or float32 WAV; this driver does not resample. Loading is timed
separately, `inference_s` times only `whisper_full`, and RTF is inference
seconds divided by audio seconds. `*_mean_s` fields are per-invocation means
from Whisper's timing API; the bracketed stderr timing reports retain stage
totals and invocation counts. `peak_rss_bytes` is the process high-water mark,
not a device-wide or per-call memory allocation measurement.

Preserve stdout/stderr separately and record the process return code. Use
three warm measurements and at least five minutes of repeated transcription
per backend under matched starting battery, charging, screen and thermal
conditions. Record thermal status/temperature throughout and compare initial
versus final-window inference time. Do not force suspend. Keep debug/profile
logging disabled for timing and record CLI process time separately if needed.
The driver verifies primary initialization, but execution confirmation still
requires the separate diagnostic run. Report measured results even if HTP
is slower; a short smoke is not a sustained-performance result.

## Native CI coverage

`whisper-ci.yml` covers native tests, real-model smokes and installed exports.
`engines-cross-ci.yml` includes an Android Whisper CLI compile lane against
CPU-only shared GGML. Hosted Android compilation does not exercise HTP or
OpenCL. Real-device results, matching DSP builds and human review remain
separate requirements.
