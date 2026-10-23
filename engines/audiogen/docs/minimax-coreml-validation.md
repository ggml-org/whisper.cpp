# MiniMax full Core ML validation

This experimental replay harness runs the LM, depth decoder, feedback,
conditioner, DiT and vocoder through the native Core ML API in C++ and
Objective-C++. Python is used to export the original F16 weights and run
conversion tests. The public MiniMax engine retains its native ggml route
and its existing optional conditioner sidecar.

The validation build requires Apple Silicon, macOS 15 or later, and an
installed speech-branch ggml with CPU, Metal and Accelerate backends. Model
conversion additionally requires PyTorch, coremltools, NumPy and gguf in the
active Python environment.

## Build

From the repository root, set `ggml_prefix` to the absolute install prefix
of the speech-branch ggml used for this build:

```sh
ggml_prefix=/absolute/path/to/ggml-install
cmake -S engines/audiogen -B build/minimax-coreml-validation \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$ggml_prefix" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 -DBUILD_SHARED_LIBS=OFF \
  -DAUDIOGEN_BUILD_MINIMAX=ON -DAUDIOGEN_BUILD_TESTS=ON
cmake --build build/minimax-coreml-validation --target audiogen-cpp --parallel 8

clang++ -std=c++17 -O2 -fobjc-arc -DMM3_ANE_VALIDATION \
  -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_BLAS -DGGML_USE_CPU -DGGML_USE_METAL \
  -mmacosx-version-min=15.0 \
  -Iengines/audiogen/include -Iengines/audiogen/src -Iengines/audiogen/src/acestep \
  -I"$ggml_prefix/include" \
  engines/audiogen/src/minimax/mm3_replay_cli.cpp \
  engines/audiogen/src/coreml/minimax-ane-validation.mm \
  build/minimax-coreml-validation/libaudiogen-cpp.a \
  -framework CoreML -framework Foundation -L"$ggml_prefix/lib" \
  -lqvac-speech-ggml -lqvac-speech-ggml-base -lqvac-speech-ggml-cpu \
  -lqvac-speech-ggml-metal -lqvac-speech-ggml-blas \
  -Wl,-rpath,"$ggml_prefix/lib" \
  -o build/minimax-coreml-validation/mm3-ane-replay
```

`MM3_ANE_VALIDATION` enables this internal replay route. Standard CMake
targets compile the existing native paths. `AUDIOGEN_COREML` continues to
control the production sidecars independently.

## Export and compose a profile

Set `models` to the directory containing `mm3-lm-f16.gguf` and
`mm3-synth-f16.gguf`. Keep conversion variants in separate directories:
LM-specific switches are part of that directory's manifest. A composed
profile uses links to the selected compiled programs, so it does not
duplicate their weights.

The following profile uses 256 LM cache positions, matching the native
cache bucket for the six-second request below. The original model context
remains in the manifest. Other requests need a profile matching their
native cache bucket.

```sh
models=/absolute/path/to/original-f16-models
profile_root="$(pwd)/models/minimax-coreml-validation"
exporter=engines/audiogen/scripts/export-minimax-ane-validation.py

python "$exporter" --models "$models" --out "$profile_root/lm" \
  --stage lm --lm-cache-positions 256 \
  --disable-conv-scale-fusion --stable-lm-silu
python "$exporter" --models "$models" --out "$profile_root/base" \
  --stage aux --lm-cache-positions 256
python "$exporter" --models "$models" --out "$profile_root/base" \
  --stage depth --lm-cache-positions 256
python "$exporter" --models "$models" --out "$profile_root/base" \
  --stage dit --lm-cache-positions 256
python "$exporter" --models "$models" --out "$profile_root/base" \
  --stage vocoder --lm-cache-positions 256 --vocoder-precision fp32-activations

python - "$profile_root" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
profile = root/"combined"
profile.mkdir(parents=True, exist_ok=True)
for stage in ("base", "lm"):
    for compiled in (root/stage).glob("*.mlmodelc"):
        destination = profile/compiled.name
        if destination.is_symlink() and destination.resolve() == compiled.resolve():
            continue
        destination.symlink_to(compiled.resolve(), target_is_directory=True)
manifest = json.loads((root/"lm/manifest.json").read_text())
(profile/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
PY
```

The LM conversion disables scale fusion and uses stable RMS/SiLU
expressions to preserve finite FP16 activations. The exporter also supports
isolated `lm-pair` experiments, validated with
`scripts/validate-minimax-ane-pair.py`; the full replay loader uses individual
LM blocks. Pair profiles are separate from full replay profiles.

Vocoder precision must be explicit. Reusing an existing compiled vocoder
with different or unrecorded precision fails with a conversion-settings
error; use a separate export directory for that variant.

## Run

```sh
build/minimax-coreml-validation/mm3-ane-replay \
  --models "$models" --ane-validation "$profile_root/combined" \
  --ane-compute-units CPU_AND_NE --mode full --device auto \
  --caption "A warm instrumental piano melody with gentle acoustic guitar, soft drums and a relaxed cinematic atmosphere." \
  --seed 7 --steps 8 --max-frames 150 --threads 8 \
  --dump-iters 1 --out output/minimax-coreml/cpu-ane
```

The 150-frame request produces approximately six seconds; this is a replay
workload parameter. The output includes `audio.wav`, generated codes,
latents, initial AR probes and `ane-timings.json`. Load time is separate
from generation and its AR, conditioner, flow and vocoder components. All
compiled programs stay resident during generation.

| `--ane-compute-units` | Devices permitted by native Core ML |
|---|---|
| `CPU_ONLY` | CPU |
| `CPU_AND_GPU` | CPU and GPU |
| `CPU_AND_NE` | CPU and Neural Engine; existing validation default |
| `ALL` | CPU, GPU and Neural Engine |

The same selection is passed to every compiled model. Core ML chooses
operation placement within the permitted devices; the option alone does
not establish actual ANE activity or exclusive GPU placement. `--device`
controls the native backend configuration independently.

## Validation and measurements, 2026-10-22

Both Macs used the same runtime sources, 90 corrected compiled programs,
F16 GGUF weights, caption, seed, eight flow steps and 150-frame request.
Each configuration ran in a fresh process, sequentially on its Mac. File,
OS and compiler caches could be reused. The machines have different macOS
and SDK versions, so the table is not a silicon-only comparison.

All eight runs completed with finite latents, no nonfinite candidate
logits, and a 264,192-frame stereo PCM16 WAV at 44.1 kHz (5.99 seconds).
These timings establish completion and performance, not perceptual quality.

| Configuration | M1 Ultra load | M1 Ultra generation | M3 Ultra load | M3 Ultra generation |
|---|---:|---:|---:|---:|
| CPU_ONLY | 148.645 s | 105.214 s | 53.588 s | 89.125 s |
| CPU_AND_GPU | 120.593 s | 2435.390 s | 17.860 s | 48.779 s |
| CPU_AND_NE | 403.591 s | 898.795 s | 132.476 s | 47.201 s |
| ALL | 283.587 s | 948.067 s | 134.236 s | 62.435 s |

The existing numerical thresholds remain relative RMS error <= 0.002 and
cosine >= 0.9999. CPU_AND_GPU passed both initial hidden-state and semantic
logit comparisons against native Metal on both Macs. CPU_ONLY and the ANE
configurations failed these initial comparisons. Full waveform equivalence
and perceptual quality remain unconfirmed.

The M3 CPU_AND_NE and ALL runs also emitted ANE compiler diagnostics after
generation. Successful WAV generation does not establish that every model
ran on the Neural Engine. The final M3 CPU_ONLY measurement was repeated
after disk-cache cleanup; the earlier disk-full attempt is excluded.

The detailed timing and initial numerical checks are in
[minimax-coreml-20261022.csv](minimax-coreml-20261022.csv). Process RSS
excludes external Core ML/ANE services. M1 GPU/ANE generation remains much
slower than CPU_ONLY in this workload.

Run the focused tests from the repository root, with the conversion
dependencies installed in the active Python environment:

```sh
python engines/audiogen/test/test_minimax_ane_models.py
python engines/audiogen/test/test_minimax_ane_export.py
python engines/audiogen/test/test_minimax_ane_cli.py \
  --executable build/minimax-coreml-validation/mm3-ane-replay
```
