# tts engine: LavaSR enhancement

Part of the [tts engine documentation](../README.md).

## LavaSR enhancement

LavaSR is post-processing for synthesized or captured speech, not another
text-to-speech engine. The denoiser uses a 16 kHz internal STFT and preserves
the requested output rate; the bandwidth-extension enhancer emits 48 kHz.
Public APIs are installed under `<tts-cpp/lavasr/>`, and `lavasr-bench`
exercises the denoiser, enhancer, or two-stage pipeline. Backend support differs
between the two stages, so use the capability matrix rather than assuming every
compiled ggml backend applies to both.

### Convert

`scripts/download-lavasr-onnx.sh` fetches the LavaSRcpp ONNX release assets
the two converters read (the original torch weights live at
[YatharthS/LavaSR](https://huggingface.co/YatharthS/LavaSR)); both converters
accept `--ftype {f32,f16}`:

```bash
scripts/download-lavasr-onnx.sh --dir checkpoints/lavasr

python3 scripts/convert-lavasr-denoiser-to-gguf.py \
    --denoiser checkpoints/lavasr/denoiser_core_legacy_fixed63.onnx \
    --out models/lavasr-denoiser-f32.gguf
python3 scripts/convert-lavasr-enhancer-to-gguf.py \
    --backbone checkpoints/lavasr/enhancer_backbone.onnx \
    --spec-head checkpoints/lavasr/enhancer_spec_head.onnx \
    --out models/lavasr-enhancer-f32.gguf
```

### Run

`lavasr-bench` runs the two-stage pipeline over a wav and can write both
stage outputs:

```bash
./build/lavasr-bench --denoiser models/lavasr-denoiser-f32.gguf \
                     --enhancer models/lavasr-enhancer-f32.gguf \
                     --in noisy.wav \
                     --out-denoised denoised.wav --out-enhanced enhanced-48k.wav
```

`test-lavasr-gguf-load` needs no fixtures: it synthesizes tiny metadata-only
GGUFs and asserts both stage loaders fail closed on a missing, empty,
truncated, unmarked, cross-architecture, or tensorless file. The denoiser and
enhancer GGUFs differ only by `general.architecture`, so each loader has to
refuse the other's file.

### Core ML

Build with `-DTTS_CPP_COREML=ON` on Apple to enable either or both neural cores.
The GGUF remains required for architecture and DSP parameters. Supply an
explicit compiled model path and compute policy for each selected stage;
there is no automatic discovery or fallback if a requested Core ML model fails
to load or predict. Existing callers keep their CPU/ggml/GPU routes.

The models exported below require macOS 13+ or iOS 16+. Export and compile on a
Mac with Xcode command-line tools. The converter uses `torch`, `numpy`, `gguf`
and `coremltools`; conversion was checked with Python 3.11, torch 2.7.0,
coremltools 9.0 and numpy 2.2.6. These are conversion dependencies, not runtime
dependencies of the C++ library.

```bash
python3 scripts/export-lavasr-coreml.py \
    --stage denoiser --gguf models/lavasr-denoiser-f32.gguf \
    --precision float32 --output models/denoiser-f32.mlpackage \
    --compile-dir models
python3 scripts/export-lavasr-coreml.py \
    --stage enhancer --gguf models/lavasr-enhancer-f32.gguf \
    --frames 256 --precision float32 --output models/enhancer-f32.mlpackage \
    --compile-dir models
```

Precision is required: `float32` preserves the reference numerics, while
`float16` permits lower precision execution including Neural Engine placement.
The denoiser's magnitude/log calculation retains FP32 precision to preserve
near-silence values and avoid overflow when squaring full-scale STFT bins.
Changing precision requires an explicit export; the loader never converts the
GGUF or changes the selected model precision at runtime.

`--frames` sets the enhancer's fixed inference window, not a maximum audio
duration. C++ processes longer inputs in overlapping windows and retains the
convolution context. The exporter masks padded activations after each block,
preserving short inputs and final-window boundaries. The denoiser retains its
GGUF-defined chunk size and resets recurrent state for each chunk, matching
the existing scalar and ggml implementations.
All chunks are submitted together through Core ML's batch prediction API;
the compiled model and its tensor shapes stay unchanged, and chunks keep
independent zero recurrent state. No audio-duration cap is imposed.

| Policy | Eligible Core ML compute units |
|---|---|
| `CPU_ONLY` | CPU |
| `CPU_AND_GPU` | CPU and GPU |
| `CPU_AND_NE` | CPU and Neural Engine |
| `ALL` | CPU, GPU and Neural Engine |

These are Apple's allowed compute units, not guarantees that every operation
runs on a particular device. `backend_name()` reports the requested policy
as `coreml-cpu`, `coreml-gpu`, `coreml-ane`, or `coreml-all`;
`Enhancer::backend_device()` reports `BackendDevice::CoreML`.
Resampling, STFT/ISTFT, overlap-add and FastLR remain in the existing C++ DSP
pipeline. See Apple's [compute units](https://developer.apple.com/documentation/coreml/mlcomputeunits)
and [precision guide](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html).

```cpp
#include <tts-cpp/lavasr/denoiser.h>
#include <tts-cpp/lavasr/enhancer.h>

tts_cpp::lavasr::DenoiserOptions dn_options;
dn_options.coreml = {"models/denoiser-f32.mlmodelc", "CPU_ONLY"};
auto dn = tts_cpp::lavasr::Denoiser::load("models/lavasr-denoiser-f32.gguf", dn_options);

tts_cpp::lavasr::EnhancerOptions enh_options;
enh_options.coreml = {"models/enhancer-f32.mlmodelc", "CPU_ONLY"};
auto enh = tts_cpp::lavasr::Enhancer::load("models/lavasr-enhancer-f32.gguf", enh_options);

auto cleaned = dn->denoise(pcm, sample_rate);
auto enhanced_48k = enh->enhance(cleaned, sample_rate);
```

Reuse these objects across calls: each compiled model is loaded once, and
access to its reusable inference buffers is serialized. A selected Core ML
stage cannot also select `use_gpu` or a nonzero `n_gpu_layers`; use the Core ML
compute policy instead. Other stages can independently keep their original
backend.

```bash
./build/lavasr-bench \
    --denoiser models/lavasr-denoiser-f32.gguf \
    --enhancer models/lavasr-enhancer-f32.gguf --in noisy.wav \
    --denoiser-coreml models/denoiser-f32.mlmodelc \
    --enhancer-coreml models/enhancer-f32.mlmodelc \
    --coreml-compute-units CPU_ONLY --runs 3 --warmup 0 --verify \
    --out-denoised denoised.wav --out-enhanced enhanced-48k.wav
```

Load time is reported separately from processing time. `--verify` prints
per-stage comparisons to scalar/ggml CPU; it is diagnostic, not a quality gate.
`test-lavasr-coreml-parity` enforces FP32 waveform parity and output lengths for
empty, short and multi-window inputs, repeated calls, and incompatible models:

```bash
./build/test-lavasr-coreml-parity \
    models/lavasr-denoiser-f32.gguf models/lavasr-enhancer-f32.gguf \
    models/denoiser-f32.mlmodelc models/enhancer-f32.mlmodelc
python3 test/test_export_lavasr_coreml.py
./build/test-lavasr-coreml-batch \
    models/lavasr-denoiser-f32.gguf models/denoiser-f32.mlmodelc CPU_ONLY
```

The native parity fixture is disabled in CTest until those four files exist in
`TTS_CPP_TEST_MODEL_DIR`. The options and exporter tests need no real weights;
the exporter tests report a skip when numpy/torch are unavailable.
The batch fixture checks serial/batch parity, output ordering, changing batch
sizes, invalid dimensions and concurrent calls on one loaded model.
