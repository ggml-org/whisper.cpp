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
`--n-gpu-layers 999` on the GPU backend of the build, and on a Core ML build
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

Fixture roots are configurable with `PARAKEET_TEST_MODEL_DIR`,
`PARAKEET_TEST_AUDIO_DIR`, and `PARAKEET_TEST_REF_DIR`. `test-gpu-vs-cpu`
(formerly `test-vk-vs-cpu`) is available when Vulkan or Metal is configured.
`verify-gguf-roundtrip.py`,
`ref-encoder-from-gguf.py`, and `streaming-reference.py` support converter and
parity investigation.
