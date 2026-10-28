# Chatterbox native Hexagon validation

This procedure extends the implementation merged in [PR #355](https://github.com/tetherto/qvac-fabric-speech.cpp/pull/355).
It provides reproducible collection for Turbo and Multilingual on one recorded
Snapdragon device. It does not introduce another Hexagon implementation or change
automatic backend selection. The commands, manifests and retained artifacts are
part of the evidence; a successful process exit alone is insufficient.

The scope is Fabric engines, shared GGML when needed, native builds and native
validation. QVAC/SDK integration, consumer prebuilds, registry rollout and releases
remain deferred until legal review and separately authorized integration.

## Existing execution policy

Request the NPU explicitly with `--backend hexagon` or
`tts_cpp::chatterbox::EngineOptions::backend = "hexagon"`. An unavailable
explicit backend is an error. Empty/automatic selection keeps its existing
policy and does not automatically prefer HTP0.

T3 and S3Gen use supported Hexagon repack storage; HTP permits the existing
F32/F16/Q8_0/Q4_0 weight paths and rejects unsupported types. Hexagon forces
the T3 KV cache to F16 even when the caller requests F32. Reference-audio
conditioning (including S3Tokenizer/CAM++) remains on CPU. Shape-dependent
unsupported graph operations may also use the scheduler's CPU path. See
[Chatterbox's backend guide](chatterbox.md#snapdragon-hexagon-npu) for the
graph adaptations and existing numerical gates.

Backend discovery, the primary backend name, buffer placement, and a valid WAV
do not prove that each graph node executed on the DSP. Collect scheduler/HTP
diagnostics separately in an untimed run and inspect stage placement.

## Build and pin the native artifacts

Use one immutable `qvac-ext-ggml` **speech** revision for all matching host-side
Hexagon libraries and DSP skeletons. The vendored Whisper GGML copy is not the
Speech dependency. Record source SHA, clean/dirty source state, compiler/NDK,
Hexagon SDK/Tools versions, CMake cache, and artifact hashes. Never label an
uncommitted build as an exact clean-commit validation.

Build/install GGML, then configure native TTS with that install. For numerical
tests use a static TTS library because the existing harnesses call private
detail symbols:

```sh
cmake -S engines/tts -B build-tts-tests \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/absolute/ggml-install \
  -DTTS_CPP_USE_SYSTEM_GGML=ON -DTTS_CPP_BUILD_SHARED=OFF \
  -DTTS_CPP_BUILD_TESTS=ON -DTTS_CPP_BUILD_EXECUTABLES=ON \
  -DTTS_CPP_TEST_MODEL_DIR=/absolute/chatterbox-models
cmake --build build-tts-tests --parallel 4 --target \
  tts-cli chatterbox-fit-params test-backend-selection test-conv-columns \
  test-attention-fallback test-t3-backend-logits test-s3gen-backend-agreement \
  test-chatterbox-backend-parity test-chatterbox-explicit-backend \
  test-chatterbox-fit-params test-chatterbox-engine-stream
```

Build a second installed shared TTS library with `TTS_CPP_BUILD_SHARED=ON`,
`TTS_CPP_BUILD_TESTS=OFF` and an explicit writable `CMAKE_INSTALL_PREFIX`.
The measurement consumer uses only installed public headers and that library:

```sh
cmake -S scripts/benchmarks/chatterbox-hexagon -B build-chatterbox-measure \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/absolute/tts-install;/absolute/ggml-install"
cmake --build build-chatterbox-measure --parallel 4
```

For Android, configure GGML and both TTS builds with the same Android arm64
toolchain. GGML must enable `GGML_HEXAGON=ON`, `GGML_OPENCL=ON`, matching
Hexagon SDK/Tools and the target OpenCL headers/library. The documented
[native Snapdragon build/deployment procedure](../../../docs/WHISPER-HEXAGON.md#build-android-arm64)
shows the shared GGML configuration and loader requirements; select the TTS
targets above rather than Whisper targets. Record the detected HTP architecture
and deploy its skeleton from the same build. A skeleton filename alone is not
device detection evidence.

## Freeze the primary matrix

`scripts/benchmarks/fixtures/chatterbox-hexagon.json` contains the fixed prompts,
model basenames and common requested settings. Its primary matrix has:

| Group | Coverage | Cells |
|---|---|---:|
| Built-in voice | Q4/Q8 × Turbo English and MTL English/French/Spanish × CPU/OpenCL/HTP × batch/stream | 48 |
| Reference cloning | Both variants in English × Q4/Q8 × CPU/OpenCL/HTP × batch/stream | 24 |
| Paragraph | Both variants in English × Q4/Q8 × CPU/OpenCL/HTP, batch | 12 |
| Total | Primary synthesis cells; stage tests/retries are separate | 84 |

Common requested settings include seed 42, four CPU threads, context 1024,
sentence split 180, and default variant-specific CFM settings. Batch and stream
remain distinct workloads. The requested F32 KV policy is common, but HTP's
forced F16 representation must remain disclosed. Full-context HTP diagnostics
are separate from the common-context comparison.

List cells before evaluation:

```sh
python3 scripts/benchmarks/chatterbox-hexagon-validation.py list
```

Prepare the eight named Q4/Q8 T3/S3Gen GGUFs plus the same PCM reference voice.
Use the existing official source/conversion tooling when an asset is missing;
record source revision, converter and hash. Do not swap checkpoints after a
failure. Freeze the hashes of the **deployed Android** measurement executable,
models, voice and native runtime libraries before a device collection:

```sh
python3 scripts/benchmarks/chatterbox-hexagon-validation.py freeze \
  --models-root /absolute/chatterbox-models \
  --reference-audio /absolute/reference-jfk.wav \
  --driver-artifact /absolute/android/chatterbox-hexagon \
  --runtime-artifact /absolute/android/libtts-cpp.so \
  --runtime-artifact /absolute/android/libqvac-speech-ggml-hexagon.so \
  --runtime-artifact /absolute/android/libggml-htp-v79.so \
  --speech-sha FULL_SPEECH_SHA --ggml-sha FULL_GGML_SHA \
  --out /absolute/evidence/frozen-android.json
```

Include every deployed GGML/CPU/OpenCL/FastRPC/C++ runtime artifact relevant to
the run, rather than only the illustrative three `--runtime-artifact` entries.
The architecture suffix above is an example; use the device's matching skel.
The freeze command refuses overwriting an existing manifest. `--asset-map`
accepts a JSON object mapping standard model basenames to local files that
retain those names. `--case ID` may intentionally freeze a subset for early
smokes; this remains a subset and never establishes 84-cell coverage.

## Collect locally or on Android

The C++ consumer reports JSON lines with `config`, `load`, `run`, `end` and
`error` kinds. One **excluded `first_call` always occurs**, even with
`--warmup 0`, to finish asynchronous/lazy S3Gen loading. Constructor time is
not full model-loading time. Each native call writes a unique WAV; the final
measured/sustained WAV is also copied to the requested output outside timing.
Thus 84 primary cells with `--runs 1 --warmup 0` produce at least 168 native
calls. Count and retain those phases separately.

For host smokes, freeze the matching host binary/libraries separately, then:

```sh
python3 scripts/benchmarks/chatterbox-hexagon-validation.py collect \
  --frozen /absolute/evidence/frozen-host.json \
  --case builtin-turbo-q4_0-en-cpu-batch \
  --transport local --out-dir /absolute/evidence/host-attempts
```

Stage Android files in an isolated deployment directory: the measurement
executable at its root, libraries under `lib/`, and models/reference WAV under
`models/`. Supply a device JSON recording model, SoC, Android build and detected
HTP architecture, with the discovery evidence retained alongside it.

```sh
python3 scripts/benchmarks/chatterbox-hexagon-validation.py collect \
  --frozen /absolute/evidence/frozen-android.json \
  --transport adb --adb adb --serial DEVICE_IP:DEBUG_PORT \
  --device-info /absolute/evidence/device.json \
  --remote-root /data/local/tmp/chatterbox-hexagon \
  --env LD_LIBRARY_PATH=/data/local/tmp/chatterbox-hexagon/lib:/system/lib64:/vendor/lib64 \
  --env 'ADSP_LIBRARY_PATH=/data/local/tmp/chatterbox-hexagon/lib;/vendor/lib/rfsa/adsp;/dsp' \
  --out-dir /absolute/evidence/device-attempts
```

`--adb-port` selects a nondefault host ADB server port. `--remote-driver` and
`--remote-models-root` override deployment locations. Remote execution uses
Android's `timeout` utility and records the driver's status in `native.status`;
the ADB transport status and artifact-pull status are retained separately.
Exit 0 from ADB without a valid native receipt cannot become a validated cell.
Remote timeouts include model loading and artifact writes, unlike native
inference timing. A lost transport can leave a remote outcome unknown until
its bounded process finishes; investigate it before starting another workload.

Each invocation creates a new attempt directory. The union of required inputs
is hash-checked on the host and deployed device before and after the complete
collection. These are collection-wide brackets, not per-case hashes. Hashing
each large model around every cell would add CPU heat/work to the comparison.
The after bracket runs even when collection fails or is interrupted; a missing
or mismatched bracket invalidates every otherwise successful contained cell.
Missing output, partial event sequences, mismatched settings, nonzero
native status, stale or truncated WAVs, failed stream contracts, and changed
input hashes remain errors. Native failures stop collection by default;
`--continue-on-error` collects other cells only when a native status is known.
Raw outputs, command arrays, transport/native receipts and PCM hashes remain
available. Do not modify phone settings without authorization.

## Numerical tests and execution evidence

Run the existing focused harnesses on the device, preserving all output and
native status. Examples (with matching deployed models/library paths):

```sh
./test-conv-columns hexagon
./test-attention-fallback opencl
./test-t3-backend-logits T3.gguf hexagon
./test-t3-backend-logits T3.gguf opencl --n-ctx 1024
./test-s3gen-backend-agreement S3GEN.gguf --backend hexagon
./test-s3gen-backend-agreement S3GEN_Q4.gguf --backend hexagon \
  --mel-tolerance 5e-2 --max-rms-ratio 0.5
./test-chatterbox-backend-parity T3_MTL.gguf S3GEN_MTL.gguf en --backend hexagon
```

Run both variants/tiers and explicit OpenCL comparisons as applicable, plus
backend selection, streaming, cloning, unavailable-device/unsupported-weight
errors and memory-fit contracts. Preserve each harness's registered numerical
tolerances, including the existing Q4 bounds above. Report failed, unsupported,
disabled and unavailable-fixture cases separately. `ctest -N` is inventory,
not execution; no full backend suite is part of this procedure.

The known greedy Turbo stop-token behavior affects CPU and accelerators on
some prompts. Keep the standard sampling mode as the primary workload and
record any separate greedy diagnostic. The public `SynthesisResult` does not
expose the stop reason: the consumer reports `termination_status` as
`not_exposed_by_public_api` and a token-limit suspicion only as a heuristic.
Do not promote that heuristic into proof of correct termination.

Collect scheduler/HTP traces in separate untimed runs. State which stages and
fallbacks were actually observed. Re-run failures and compare exact cases to
the same-build CPU/OpenCL baseline before proposing a runtime fix.

## Performance, fit and quality interpretation

For warm performance use the fixed Q4 English short and paragraph workloads on
both variants and each backend, excluding `first_call` and one warmup, with
three measured repeats. For example, select one case with `--warmup 1 --runs 3`.
Use `--duration-seconds 300` for each of the six primary backend/variant
sustained sessions. Start under comparable thermal/battery conditions and
retain separate screen, charging and temperature telemetry. The collector
does not automatically grant a thermal/idle guard pass or change DND settings.

Compare native mean/median/min/max, generated audio duration, RTF (native
seconds/audio seconds), audio-seconds/native-second, T3/S3Gen times and streamed
first-audio latency. Native calls use the consumer's steady clock; file output,
ASR, model load and excluded phases are not included. Generated audio lengths
can differ, so report duration and RTF beside raw seconds. Process RSS is not
total GPU/FastRPC/shared-memory consumption. Temperature or first/last time
changes alone do not establish thermal throttling or energy savings.

Check fit's projected and measured components using the same variant, model,
context and cache policy. Unknown capacity and unsupported graph failures must
remain explicit errors. A successful fit invocation is not byte-parity proof.

Run pinned multilingual reference ASR on host CPU after synthesis with explicit
language and fixed decoding settings; expected text must not be supplied as an
ASR prompt. Save original prompt, WAV, transcript, model/settings hashes and
edit/reference-word counts. The existing scorer calls its normalizer `english`,
but its implementation is Unicode-basic: NFC, lowercase, Unicode word/ASCII
apostrophe preservation, punctuation-to-space and whitespace collapse. Before
model evaluation, a nine-check preflight verified French/Spanish accents,
decomposed Unicode and punctuation, and that distinct word forms remain
distinct. Reuse that unchanged policy explicitly for English/French/Spanish.
It performs no contraction, number or semantic rewriting; curly and straight
apostrophes remain distinct. Retain the normalization examples and scorer hash
with the quality evidence. The existing English-only ASR wrapper must not be
used unchanged for French/Spanish: invoke the pinned multilingual ASR with the
correct language and pass its transcript to the unchanged scorer.

WER includes ASR errors; this procedure sets no new quality threshold or
speaker-similarity gate. A French exception accepted for a different model/task
does not apply.
Provide representative WAVs for human listening; do not claim a listening
pass before review occurs.

## Summarize and preserve failures

```sh
python3 scripts/benchmarks/chatterbox-hexagon-validation.py summarize \
  --attempt-dir /absolute/evidence/device-attempts/ATTEMPT_ID \
  --out /absolute/evidence/summary.json
python3 -B scripts/benchmarks/test-chatterbox-hexagon-validation.py
```

The summary rechecks saved WAV hashes and before/after input brackets. It
groups comparisons only when frozen settings, model/source/runtime identities,
device and environment agree, keeping different phases separate.
The intentional offload-layer difference (CPU 0, OpenCL/HTP 99) is excluded
from the comparison key; each native receipt retains the full requested
settings and the explicit primary-backend check. All other settings remain
part of the comparison key. Repeated
backend/case attempts are flagged rather than silently replacing a failure
with the latest result. Missing primary cells remain visible.

`execution_validated` means the native measurement and retained audio passed
structural checks. It does not mean every kernel passed, quality is accepted,
HTP execution was observed, or CI/device validation is complete. Attach the
separate numerical, dispatch, fit, ASR/listening, thermal and native-CI results
before making those claims. No real-device result is implied by this document.
