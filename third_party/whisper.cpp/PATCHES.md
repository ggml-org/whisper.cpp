# QVAC patches on vendored whisper.cpp

This directory is a **git subtree** of [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp),
currently pinned at **v1.9.4** (`927cfce3`; machine-readable pin in
`UPSTREAM_PIN`, consumed by the divergence-guard CI). It is *upstream-tracking,
minimally divergent*: the only files that may differ from the pinned upstream
tag are the ones listed below. CI enforces this (divergence-guard job) —
if you change anything in this directory, either it's an upstream sync
(`git subtree pull`, see `docs/UPSTREAM-SYNC.md`) or this manifest must be
updated in the same PR.

**ggml is special:** `ggml/` in this subtree must stay byte-identical to
upstream. All QVAC ggml patches live in
[`qvac-ext-ggml`](https://github.com/tetherto/qvac-ext-ggml) (`speech` branch),
which ships as the `ggml-speech` vcpkg port; the umbrella build forces
`WHISPER_USE_SYSTEM_GGML=ON`, so the vendored `ggml/` is never compiled.

## Manifest

Each row must be self-contained: state what the change does and why it
exists, so the manifest is meaningful without access to any tracker.

| File | Change | Why |
|---|---|---|
| `src/whisper.cpp` | vocab-logits slice + decoder QKV matmul fusion | decode-time perf on mobile GPUs (quantized decode on Adreno-class hardware): the vocab-logits matmul is sliced to only the rows sampling actually needs instead of the full vocab, and the decoder's three Q/K/V matmuls are fused into one — upstream does neither |
| `src/whisper.cpp` | quantized matmul GPU extra-buffer selection | try the selected GPU/IGPU's existing extra buffers for quantized projection weights, probing operation support with the real buffer type before choosing one; Hexagon's non-host repack buffer enables its tiled quantized matmul while canonical default buffers preserve CPU fallback. F16/F32, GET_ROWS/tied embeddings, VAD and default device ordering are unchanged; loading and memory-fit projection share this selector |
| `src/whisper.cpp` | BCI model loading and windowed attention | preserve the BCI-specific model header, convolution kernel size, per-layer attention window/mask, and decoder prompt used by `@qvac/bci-whispercpp`; these are existing fork behavior, retained through the v1.9.4 sync |
| `src/whisper.cpp` | VAD honors `use_gpu` | upstream places the VAD weights per `use_gpu` but forces the compute backends to CPU, so `use_gpu=true` aborts in ggml_backend_sched on every GPU build (ggml-org/whisper.cpp#3508); the backends now follow the same flag, and the LSTM input is made contiguous before its matmul because the CUDA mul-mat-vec kernel rejects the odd-stride transposed view. Default stays CPU. Upstreaming candidate |
| `src/whisper.cpp`, `include/whisper.h` | memory-fit preflight (`whisper_fit_params` / `whisper_fit_actual`) | project whether a model + workload fits the free device memory WITHOUT reading weight data (QVAC-24283, the SDK's @qvac/model-fit contract: Success=0/Failure=1/Error=2). The projection mirrors the runtime by construction: `whisper_model_load`, `whisper_kv_cache_init` and the VAD init gain an optional measure mode (size buffers via ggml's size-only APIs, seek past tensor payloads, mark tensors externally-allocated), and a size-only twin of `whisper_init_state` prices the four resident schedulers through `ggml_backend_sched_reserve_size`. `whisper_fit_actual` measures the same breakdown from a real load, reproducing `whisper_full`'s worst-case KV recreation when `n_decoders > 1` -- the byte-parity oracle for the tests |
| `src/whisper.cpp`, `src/whisper-fit-memory.h`, `tests/test-whisper-fit-memory.cpp`, `tests/CMakeLists.txt` | shared-memory fit capacity and backend metadata budgeting | use the backend's `memory_unified` property instead of registry-name matching; an unknown unified-device capacity falls back to the CPU's OS/system-RAM query, while zero available bytes with a known total remains memory pressure. Unknown capacity is an Error. Include only prospective Hexagon metadata growth beyond retained queues/metadata already reflected in available RAM; test these policies without drivers or models. Physical tensor-buffer guards are accounted per actual allocation chunk by the shared allocator |
| `src/whisper.cpp`, `src/whisper-fit-memory.h`, `tests/test-whisper-fit-graph-error.cpp`, `tests/CMakeLists.txt` | reject failed scheduler measurements | propagate a rejected size-only graph reservation to the existing measurement-error path instead of using partial compute sizes to report a false fit. A real CPU scheduler regression uses an unsupported preallocated copy, then checks supported-graph recovery and exact size parity; it needs no model or accelerator |
| `src/whisper.cpp`, `include/whisper.h` | memory-fit preflight from a weightless description | `whisper_fit_params` takes, for the model and for the VAD model, either the file or the GGUF description qvac's registry publishes beside it (the file's header settings, a vocabulary length histogram and the tensor inventory, with no data section) and replays the description as the byte stream the file opens with, payloads left out, so the same metadata-only load projects both identically before any weights are downloaded; a real load refuses a description (bad magic) |
| `examples/whisper-fit-params/` | new example (`whisper-fit-params` CLI + `whisper-fit-util.h`) | thin front-end over `whisper_fit_params`: exit code == fit status, `--json` for the SDK, `--verify` prints projected-vs-real; `--model` / `--vad-model` also take weightless descriptions; the header keeps the strict ERANGE-checked parsers and UTF-8-validating JSON escaper testable model-free |
| `examples/cli/cli.cpp`, `examples/whisper-fit-params/whisper-fit-params.cpp`, `examples/whisper-device-selection.h` | shared native device listing, exact-name selection and automatic HTP preference | preserve numeric GPU/IGPU ordinals, prefer registered HTP0 only when no flag/environment target is given, and retain the existing default if HTP is unavailable or ambiguous. Resolve names through Whisper's filtered device order, reject invalid explicit targets, and honor CPU mode; listing reads no model and explicit flags override `WHISPER_ARG_DEVICE`. The C API's numeric-device contract is unchanged |
| `src/whisper.cpp`, `include/whisper.h`, `examples/cli/cli.cpp`, `examples/whisper-fit-params/whisper-fit-params.cpp`, `examples/whisper-device-selection.h` | primary-backend metadata and strict explicit CLI initialization | additive `whisper_get_primary_backend_name` reports the default state's first initialized device without changing parameter structs; both native frontends reject an explicit flag/environment target that fell back during initialization, including a fit projection for another device. Node-level CPU fallback remains supported and the primary name is not evidence that every operation executed on that device |
| `src/whisper.cpp` | opt-in release debug logging | `WHISPER_LOG_DEBUG=1` lets the default logger expose existing GGML scheduler and HTP diagnostics in release builds; ordinary runs retain their existing log level |
| `tests/test-whisper-device-selection.cpp`, `tests/test-whisper-device-cli.cmake`, `tests/CMakeLists.txt` | native device-selection regression tests | cover mixed registry ordering, strict indices/names, environment/flag precedence, CPU overrides, model-free listing and CLI errors using the existing CTest harness without changing numerical grading |
| `tests/test-whisper-zero-samples.cpp` | primary-backend metadata regression checks | reuse the existing CPU-mode fixture state to check its registered non-GPU primary device, plus null-context and no-default-state handling; fixture loads do not establish model accuracy |
| `examples/CMakeLists.txt` | register `whisper-fit-params` | build glue for the example above |
| `tests/test-whisper-fit-cli.cpp` | new file | model-free unit test: strict numeric parsers, saturating margin arithmetic, JSON escaping, and `whisper_fit_params`' invalid-argument / missing-model status contract |
| `tests/test-whisper-fit-params.cpp` | new file | byte-parity gates: a metadata-only projection must equal what a real load + state init actually allocates (weights, KV caches, all four scheduler compute buffers, VAD, host-overflow split), byte-for-byte including the `n_decoders > 1` KV worst case, and longer audio must grow only host bytes; the `gpu` arm additionally anchors the device/host classifier to the driver's own allocation counter (parity alone cannot see a misclassification shared by both sides) and exits 77 (ctest SKIP) without a GPU device; runs against the committed tiny + silero fixtures |
| `tests/test-whisper-fit-description.cpp` | new file | parity gate for descriptions: projecting each committed description must equal projecting the file it describes on every component (weights, KV, compute, VAD, host-overflow split, host bytes), alone and with a VAD model or VAD description; a description of the wrong kind, or a VAD description without the four encoder layers the Silero loader builds, is refused with the documented reason, and a real load refuses a description. Takes `<vad> <vad.fit.gguf> <model> <model.fit.gguf>...` so real downloaded models can be checked the same way; the `gpu` arm exits 77 (ctest SKIP) without a GPU device |
| `models/for-tests-ggml-tiny.fit.gguf`, `models/for-tests-silero-v6.2.0-ggml.fit.gguf`, `models/for-tests-ggml-bci.bin`, `models/for-tests-ggml-bci.fit.gguf` | new test fixtures | the descriptions qvac's registry-server writes for the committed tiny and silero fixtures, plus a header-only BCI model (n_mels 512 with the three BCI header fields, no tensors) and its description, so the BCI header branch of the replay is covered model-download-free |
| `tests/CMakeLists.txt` | register the three fit tests | the model-free/CPU forms carry the `unit` label and use only committed fixtures, so they run model-download-free in CI; the two parity tests are registered a second time as `test-whisper-fit-params-gpu` and `test-whisper-fit-description-gpu` under the `gpu` label -- excluded by the CPU lanes' `-LE 'gpu|...'`, picked up by any `ctest -L gpu` GPU lane (the engines' self-hosted-lane convention), and a ctest SKIP where no GPU exists |
| `tests/CMakeLists.txt` | drop `../ggml/include` from every test target; link `whisper` where only `common` was linked | the tests take the ggml headers from the `whisper` target they link (it exports ggml publicly, `common` does not); under `WHISPER_USE_SYSTEM_GGML` the vendored `ggml/` headers are never compiled and describe a different `gguf_init_params` than the speech-branch library (an extra field), so a test built against them passed the struct with the wrong layout and `test-whisper-fit-description` segfaulted on arm64 |
| `src/whisper-logits-slice.h` | new file | helper implementing the vocab-logits slice above |
| `src/whisper.cpp`, `include/whisper.h` | API additions | public controls for the logits-slice behavior and the sampling seed so bindings/addons can opt in per-decode; retain `params.seed` when applying upstream v1.9.4's decoder-0 reset |
| `src/CMakeLists.txt` | build glue | wire whisper-logits-slice.h; install/export adjustments for the vcpkg port |
| `src/CMakeLists.txt`, `cmake/whisper-config.cmake.in` | VitisAI helper export | export the internal object helper without its source include path and find its FlexMLRT dependency for installed static consumers; otherwise enabling VitisAI breaks CMake generation with QVAC's `whisper-targets` export |
| `CMakeLists.txt` | install/export + robustness | export `whisper-targets`, headers under `include/whisper/`; guard `git-vars`/js-bindings config steps so source-tarball (vcpkg) builds work |
| `cmake/git-vars.cmake` | robustness | tolerate non-git source trees (vcpkg tarballs) |
| `cmake/whisper-config.cmake.in` | config fixes | correct find_package config for system-ggml consumers |
| `CMakeLists.txt`, `src/CMakeLists.txt`, `examples/CMakeLists.txt` | `WHISPER_BUILD_PARAKEET` option (default ON) | gate upstream's bundled parakeet: its `parakeet`/`parakeet-cli` target names collide with `engines/parakeet`; the umbrella sets it OFF. Upstreaming candidate |
| `CMakeLists.txt` | `include(GNUInstallDirs)` moved before `add_subdirectory(src)` | the whisper INSTALL_INTERFACE expands `${CMAKE_INSTALL_INCLUDEDIR}`; unset it exports a bogus `/whisper` path breaking install-tree `find_package(whisper)`. Absorbed from the registry port's patch. Upstreaming candidate |
| `tests/test-whisper-logits-slice.cpp` | new file | header-only unit test for the vocab-logits slice patch (suffix-offset arithmetic); needs no model |
| `tests/test-vad-streaming.cpp` | new file | pins the streaming-VAD contract: `whisper_vad_detect_speech_no_reset` + `whisper_vad_reset_state` must reproduce `whisper_vad_detect_speech` per-chunk probabilities exactly; uses only the committed silero test fixture |
| `tests/test-whisper-lang.cpp` | new file | pins the `whisper_lang_*` lookup API (code/full-name round-trip over all 100 entries, unknown-input handling) — the multilingual surface every `-l` / `--detect-language` option resolves through; needs no model |
| `tests/test-whisper-params.cpp` | new file | pins the default-parameter contract (`whisper_full_default_params` greedy/beam, context, VAD, `_by_ref`) so an upstream sync cannot silently change decode defaults; needs no model |
| `tests/test-vad-gpu.cpp` | new file | pins the VAD `use_gpu=true` contract for the fix above: init must succeed and produce CPU-matching probabilities; exercises the real GPU path on GPU builds and degrades to CPU-vs-CPU elsewhere; uses only the committed silero test fixture |
| `tests/CMakeLists.txt` | register the five tests above; gate the parakeet tests behind `WHISPER_BUILD_PARAKEET` | the new tests carry `unit` labels so they run model-free in CI; the upstream parakeet tests link the gated `parakeet` target and must follow its option |
| `bindings/java/src/main/java/io/github/ggerganov/whispercpp/params/WhisperFullParams.java`, `bindings/java/src/main/java/io/github/ggerganov/whispercpp/params/WhisperVadParams.java` | VAD parameters exposed to the Java binding | keeps the Java surface in step with the VAD options in whisper_full_params |
| `examples/addon.node/package.json`, `examples/addon.node/package-lock.json` | cmake-js `^7.1.1` → `^8.0.0` + npm `overrides` pinning transitive `tar` to `^7.5.21` | security: cmake-js 7.x depends on `tar ^6.2.0`, vulnerable to a decompression/parse DoS via unlimited input (GHSA-23hp-3jrh-7fpw) and further criticals fixed only in tar >= 7.5.21; the override guarantees a safe resolution, and the upstream lockfile is regenerated to match these constraints and update vulnerable transitive dependencies flagged by npm audit. Upstreaming candidate |

**Deliberately dropped from the pre-subtree fork** (not upstream, not restored):
`examples/talk-llama/llama-hparams.{h,cpp}` carried a stray `kv_only_nextn`
MTP-head hunk referencing `nextn_predict_layers`, which does not exist in
the talk-llama snapshot upstream ships with v1.9.1 — it did not compile and
was the cause of the `ubuntu-22-*` failures in the (now inert) upstream
`build.yml` on every master push. The demo builds clean without it.

Marker convention: new QVAC code blocks carry a `QVAC` reference in a nearby
comment where practical.

## Updating this pin

See `docs/UPSTREAM-SYNC.md`. Short version: `git subtree pull
--prefix third_party/whisper.cpp upstream <tag> --squash`, resolve conflicts
(expected only in the files above), re-verify this manifest, run the full CI
matrix. If a repository squash merge removed the subtree ancestry, use the
runbook's release-to-release three-way patch fallback (used for v1.9.4).
