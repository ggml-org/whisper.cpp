# parakeet engine: tests and parity

Part of the [parakeet engine documentation](../README.md).

## Tests and NeMo parity

Set up fixtures in this order:

1. Download the required `.nemo` archive.
2. Convert it to the exact q8_0 runtime or f16 parity GGUF filename expected by
   `CMakeLists.txt`.
3. Generate NeMo `.npy` references with the matching `dump-*-reference.py`
   script when the test requires references.
4. Configure after fixtures exist so CTest registrations detect them.
5. Build, inspect `ctest -N`, then run CTest.

```bash
engines/parakeet/scripts/download-all-models.sh

python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt engines/parakeet/models/parakeet-ctc-0.6b.nemo \
  --out engines/parakeet/models/parakeet-ctc-0.6b.f16.gguf --quant f16

python engines/parakeet/scripts/dump-ctc-reference.py \
  --wav engines/parakeet/test/samples/jfk.wav

# Hybrid checkpoint: select its RNN-T branch in both conversion and NeMo.
python engines/parakeet/scripts/dump-rnnt-reference.py \
  --nemo-model engines/parakeet/models/stt_ka_fastconformer_hybrid_large_pc.nemo \
  --wav engines/parakeet/test/samples/rnnt-ka-16k.wav \
  --out engines/parakeet/artifacts/rnnt-ref

cmake -S engines/parakeet -B build-parakeet -DCMAKE_BUILD_TYPE=Release
cmake --build build-parakeet -j
ctest --test-dir build-parakeet -N
ctest --test-dir build-parakeet --output-on-failure
```

`download-all-models.sh` produces `.nemo` files, not runnable GGUFs. Missing
model, audio, or reference fixtures cause individual tests to be registered as
`DISABLED`, not failed. Configure again after adding a fixture. Each test
carries exactly one label out of `unit`, `fixture`, `cpu`, `gpu`, and `perf`.
The fixture-free suite (the one CI runs) is selected by excluding the
GPU-bound and timing-bound labels:

```bash
ctest --test-dir build-parakeet -LE 'gpu|perf' --output-on-failure
```

`test-rnnt-decoder-parity` is enabled when the hybrid RNN-T GGUF, its WAV, and
the NeMo `token_ids.npy` dump are available. It requires bit-exact token IDs
between the reference and the current shared RNN-T/TDT decoder.

Model-free logic tests (`-L unit`) cover the CTC language mask, mel FFT
parity, the blocked STFT's independence from its block size, a reused
`MelState` or `IncrementalMelState` following tables that changed at the same
address, per-feature CMVN, the fixed-window Core ML export's valid lengths for
a slab with zeroed trailing padding, RNN-T graph construction, long-form window planning and the CLI's
`--window-sec` / `--overlap` conversions, the pause-cut segment planner and
the batched TDT lane walk behind the Core ML decoder sidecar (driven by a
scripted stepper), Sortformer finalization and probability thresholding, the
streaming energy VAD, the SentencePiece detokenizer, the NeMo converter and
the Neural Engine Core ML exporters (Python).

`test-transcribepsy` (`-L fixture`) runs when
`models/transcribepsy-250m-eu-sub.gguf` is staged
(`scripts/download-all-models.sh transcribepsy`): it checks the published
GGUF's transcript of `jfk.wav`, `transcribe_mel` parity with `transcribe`
including a mel computed on a worker thread, the rejection of a
`PrecomputedMel` whose buffer does not match its declared shape, the
60 s / 25 % long-form windows against the single pass, and a 187 s pause-free
English recording (`jfk.wav` and `sample-16k.wav` six times) held to one
percent word error with the default options and with the 60 s / 25 % windows.
`test-transcribepsy-gpu` (`-L gpu`) runs the same binary with
`--n-gpu-layers 999` on the GPU backend of the build,
`test-transcribepsy-hexagon` runs it with `--backend hexagon` on HTP0 (see
[Hexagon NPU tests](#hexagon-npu-tests)), and on a Core ML build
`test-transcribepsy-coreml` (`-L coreml`) runs it on the encoder-less Core ML
GGUF with its sidecars, so every backend meets the same accuracy bar. Two more fixture tests
use the same GGUF: `test-encoder-graph-cache` fills the ggml encoder graph
cache with several shapes, lowers the limit and checks the eviction, then
checks that a windowed pass holds one graph and restores the limit on both a
complete and a cancelled run; `test-tdt-cpu-copies` builds the CPU-resident
decoder weight copies an OpenCL load makes and checks them byte for byte
against the originals on the CPU backend; `test-coreml-encoder-owner-reject` checks that
the encoder-less Core ML GGUF (`transcribepsy-250m-eu-sub-coreml.gguf`) is
refused with a clear error when no sidecar can take the encoder over, and on
a Core ML build with the `-encoder.mlmodelc` sidecar staged
`test-coreml-encoder-owner` (`-L coreml`) loads it, runs the encoder,
checks that a ggml encoder request is refused rather than reading unallocated
weights, and checks that `PARAKEET_COREML_ONLY=1` skips the encoder weights of
the full GGUF.

### Hexagon NPU tests

The Hexagon arms are registered on every build and exit with the skip code
(ctest "Skipped") where HTP0 cannot be opened, so they run for real only on a
Snapdragon device with the GGML Hexagon backend and its DSP library staged.
Cross-build with `-DGGML_HEXAGON=ON` in the GGML build, push the binaries,
`libqvac-speech-ggml*.so`, the `libggml-htp-v*.so` skeletons, `libc++_shared.so`
and `libomp.so`, then run each binary with `LD_LIBRARY_PATH` and
`DSP_LIBRARY_PATH` pointing at that directory:

| Test | Checks |
|---|---|
| `test-transcribepsy-hexagon` | The `test-transcribepsy` accuracy bar with encoder and TDT decoder on HTP0 |
| `test-encoder-backend-parity-transcribepsy` | TranscribePsy encoder on HTP0 against the CPU on `hi-16k.wav`, whose first feed-forward exceeds the F16 range; turns non-finite without the matmul headroom |
| `test-attn-path-graph-transcribepsy-hexagon` | The loader picks the unfused attention and the 96-wide HMX head size from the model's own 80-wide heads |
| `test-tdt-backend-graphs-hexagon` | HTP0 accepts every node of the TDT decoder graphs, including the stacked LSTM GEMV; `test-tdt-backend-graphs` runs the same check on the CPU |
| `test-matmul-headroom-hexagon` | A q8_0 matmul whose result is 76800 comes back finite with the Hexagon headroom; `test-matmul-headroom` checks the CPU builds the plain matmul |
| `test-streaming-tdt-hexagon`, `test-streaming-eou-hexagon`, `test-encoder-backend-parity-{ctc,tdt,eou,sortformer}`, `test-fit-params-ctc-hexagon` | The other Parakeet families on HTP0 |

## Mosaic integration

`test-mosaic-*` loads the exact 486,693,152-byte Mosaic GGUF and
transcribes two complete LibriSpeech dev-clean recordings (2.245 and 3.525 s).
The committed references are checked after case/punctuation normalization;
tokens and raw text must repeat exactly on a second call. CPU and explicit
Metal/CUDA/HIP lanes require native execution. The Core ML lane verifies the matching
masked512 sidecar's provenance/inventory and requires actual per-call Core ML
use, with the decoder on Metal; fallback fails. There are no timing gates.
These smoke tests do not establish broader WER or ANE placement.

Set `PARAKEET_TEST_MOSAIC_DIR` to a fixture root containing `bundle/` (the
outputs of `scripts/prepare-mosaic.py`) and `audio/` (the pinned WAVs and
`receipt.json`). The provisioner is `test/prepare_mosaic_fixtures.py`; it
downloads the hash-pinned original archive, uses the production Mosaic
exporter, and verifies the source FLACs before lossless PCM16 WAV conversion.
An S3/registry mirror can supply the same verified bundle later without
changing the inference tests. No model weights or audio are committed.

Like other model tests, missing fixtures disable these registrations in
ordinary PR CI. The opt-in Apple CI job provisions them and requires all six
arms to execute successfully; missing/disabled/skipped coverage cannot pass:

```bash
gh workflow run parakeet-ci.yml --ref YOUR_BRANCH -f run_mosaic=true
```

The `mosaic-integration-*` workflow artifact contains CTest results, runtime
route/transcript logs, dependency SHA and model/sidecar/audio receipts.

Fixture roots are configurable with `PARAKEET_TEST_MODEL_DIR`,
`PARAKEET_TEST_AUDIO_DIR`, and `PARAKEET_TEST_REF_DIR`. `test-gpu-vs-cpu`
(formerly `test-vk-vs-cpu`) is available when Vulkan or Metal is configured.
`verify-gguf-roundtrip.py`,
`ref-encoder-from-gguf.py`, and `streaming-reference.py` support converter and
parity investigation.
