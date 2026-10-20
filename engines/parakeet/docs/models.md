# parakeet engine: models and conversion

Part of the [parakeet engine documentation](../README.md).

## Acquisition by family

Commands in this guide run from the repository root unless a section explicitly
selects the engine directory. Install the selected model's conversion
dependencies before running its converter.

| Family | Acquisition | Runtime artifact |
|---|---|---|
| Parakeet CTC / Unified RNN-T / TDT / EOU; IndicConformer; Sortformer v1/v2/v2.1 | `engines/parakeet/scripts/download-all-models.sh`, then `convert-nemo-to-gguf.py` | Converted GGUF; IndicConformer defaults to CTC, with optional `--head rnnt` |
| Nemotron 3.5 ASR | `convert-nemo-to-gguf.py` with explicit `--hf-repo nvidia/nemotron-3.5-asr-streaming-0.6b` and checkpoint/output paths | Converted ASR GGUF |
| Nemotron 3 Diarization | `engines/parakeet/scripts/download_nemotron_diarization.py` | Official pinned Q8_0 GGUF; optional source conversion is separate |
| MOSS-Transcribe-Diarize | Download the [Hugging Face checkpoint](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize), then [convert-moss-transcribe-to-gguf.py](moss-transcribe.md#convert) | Separate MOSS GGUF, API and CLI |
| TranscribePsy-250M-EU-sub | `engines/parakeet/scripts/download-all-models.sh transcribepsy` | Published q8_0 GGUF; on Apple, the encoder-less GGUF with its Neural Engine encoder and decoder sidecars |

### TranscribePsy

[TranscribePsy-250M-EU-sub](https://huggingface.co/qvac/TranscribePsy-250M-EU-sub)
is a 254M FastConformer-TDT distilled from Parakeet-TDT-0.6B-v3 for Dutch,
English, French, German, Italian, Polish, Portuguese and Spanish. It is
published ready to run, so there is nothing to convert; the engine recognises
it and transcribes long recordings with the 60 s / 25 % windows the research
team validated unless the caller asks for a window (a re-export can state the
window itself with the converter's `--long-form-window-sec 60
--long-form-overlap 0.25`):

```bash
engines/parakeet/scripts/download-all-models.sh transcribepsy

build-parakeet/parakeet \
  --model engines/parakeet/models/transcribepsy-250m-eu-sub.gguf \
  --wav engines/parakeet/test/samples/jfk.wav --threads 6
```

The selector also downloads the Core ML distribution: a 17 MB GGUF that holds
the vocabulary, the mel filterbank, the decoder weights and the configuration
but no encoder weights, plus `transcribepsy-250m-eu-sub-coreml-encoder.mlmodelc`
and `transcribepsy-250m-eu-sub-coreml-decoder.mlmodelc`. The three stay in one
directory under these names; the engine finds the sidecars next to the GGUF.
That GGUF loads only on an Apple build with `PARAKEET_COREML=ON` and its encoder
sidecar present; anywhere else it fails with a message saying so. See
[Neural Engine sidecars](backends.md#neural-engine-fixed-window-encoder-and-tdt-decoder-sidecars).

The published GGUF is a direct conversion of the NeMo checkpoint, reproduced
byte for byte with `--description`, which replaces the converter's default
`general.description` text:

```bash
hf download qvac/TranscribePsy-250M-EU-sub TranscribePsy-250M-EU-sub.nemo \
  --local-dir engines/parakeet/models
python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt engines/parakeet/models/TranscribePsy-250M-EU-sub.nemo \
  --out  engines/parakeet/models/transcribepsy-250m-eu-sub.gguf --quant q8_0 \
  --description "TranscribePsy-250M-EU-sub, FastConformer-TDT speech recognition"
```

The converter reads the TDT durations from `model_defaults`, as NeMo's released
checkpoints keep them, or from `loss.tdt_kwargs` / `decoding`, where some
training configs keep them instead. The Core ML models are exported from the
GGUF with the Neural Engine exporters described in
[backends.md](backends.md#neural-engine-fixed-window-encoder-and-tdt-decoder-sidecars).

### Nemotron 3.5 ASR conversion

```bash
python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --hf-repo nvidia/nemotron-3.5-asr-streaming-0.6b \
  --ckpt engines/parakeet/models/nemotron-3.5-asr-streaming-0.6b.nemo \
  --out engines/parakeet/models/nemotron-3.5-asr-streaming-0.6b.f16.gguf \
  --quant f16
```

If the checkpoint is absent, the converter downloads from the explicit HF
repository. Passing that repository prevents fallback to the CTC default.

## Models and conversion

The engine runs GGUF, not `.nemo`. `scripts/download-all-models.sh` downloads
NeMo archives for the CTC, Unified RNN-T, TDT, EOU, IndicConformer and
Sortformer v1/v2/v2.1 families only; those downloads must still be converted.
Nemotron and MOSS use the separate acquisition paths below.
Python is required only for conversion, Core ML export, and NeMo parity tooling;
runtime inference remains pure C++.

Create an isolated environment using a Python version supported by the selected
NeMo release:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch gguf numpy pyyaml soundfile librosa sentencepiece \
  "nemo_toolkit[asr]" huggingface_hub
```

Convert the downloaded checkpoint:

```bash
python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt engines/parakeet/models/parakeet-ctc-0.6b.nemo \
  --out engines/parakeet/models/parakeet-ctc-0.6b.q8_0.gguf \
  --quant q8_0
```

The converter defaults to CTC 0.6B and `q8_0`. When `--out` is omitted, it
derives `models/parakeet-ctc-0.6b.<quant>.gguf` from `--quant`. For every other
checkpoint, pass an explicit `--ckpt`, `--hf-repo`, and `--out`; otherwise a
missing local checkpoint can cause the default CTC repository to be downloaded.
Keep the quantization in explicit filenames:

```text
<model>.q8_0.gguf
<model>.f16.gguf
```

Use f16 for numerical parity against NeMo references and q8_0 for normal runtime
fixtures. Small tensors or dimensions unsuitable for block quantization remain
f16. Hybrid RNNT+CTC checkpoints export their CTC branch by default; pass
`--head rnnt` to export the Transducer branch instead. RNN-T conversion rejects
checkpoints whose joint output is not exactly vocabulary plus blank, preventing
a duration-bearing TDT head from being mislabeled as plain RNN-T.

Recorded CTC 0.6B quantization results on an M4 Air CPU:

| Quantization | Size | 20-second encoder | 11-second encoder | Transcript |
|---|---:|---:|---:|---|
| f32 | 2.4 GiB | n/a | n/a | exact |
| f16 | 1.3 GiB | 1221 ms | ~680 ms | bit-equal |
| q8_0 | 697 MiB | 839 ms | 460 ms | bit-equal |
| q5_0 | 453 MiB | 1475 ms | ~650 ms | bit-equal |
| q4_0 | 372 MiB | 1080 ms | 595 ms | bit-equal |

For the NeMo families covered by `scripts/download-all-models.sh`, the
script fetches the archive (including the AI4Bharat IndicConformer hybrid) and
`scripts/convert-nemo-to-gguf.py` converts it to GGUF. The downloader also caches
a forward-looking TDT+CTC checkpoint; downloading it does not establish support. The IndicConformer conversion emits the per-language
token ranges the run-time `--language` selection needs:

```bash
python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt engines/parakeet/models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo \
  --out engines/parakeet/models/indic-conformer-600m-multilingual.q8_0.gguf \
  --quant q8_0
```

To select the same checkpoint's RNN-T branch, add `--head rnnt` and use a
distinct output filename. The auxiliary `ctc_decoder.*` tensors are then
ignored.

## Nemotron 3 Diarization

The commands in this section run from `engines/parakeet` (`cd engines/parakeet`).

Nemotron 3 Diarization loads NVIDIA's official `Nemotron-3-Diarization.q8_0.gguf`
directly. Use it with `--model` for standalone diarization or
`--diarization-model` with a Parakeet ASR model for speaker attribution. Its
native output contains eight independent speaker probabilities every 10 ms.
The streaming API uses a speaker cache and 80 ms encoder frames; its default
chunk is 1040 ms with 0 ms left and 80 ms right context. Explicit 80 ms left
context retains one encoder frame. Streaming uses a gain from its first window
for the whole session; offline inference normalizes its full input. Custom chunk and context durations
must be multiples of 80 ms. The Q8 checkpoint is covered by a numerical
reference test against NVIDIA's C++ implementation. With `--n-gpu-layers` or
`EngineOptions::n_gpu_layers` above zero the whole graph runs on CUDA, Vulkan,
Metal, or OpenCL; offline inference and cached streaming pass the same
numerical and session tests on an RTX 5090 (CUDA and Vulkan), an AMD Radeon
RX 7600 XT (Vulkan), an Apple M3 Ultra (Metal), and the Adreno 830 of a
Snapdragon 8 Elite (OpenCL and Vulkan). Backends without fused flash
attention for this graph, such as ggml-opencl on Adreno, keep attention on
the GPU with an unfused path, and Adreno GPUs run the matmuls at F32
precision.
`EngineOptions::prewarm` runs one offline pass and one chunk at the default
live geometry, and `parakeet-fit-params` projects the model's memory. GPU
timings and accuracy are in
[docs/performance.md](performance.md#nemotron-3-diarization-on-gpu).
Offline inputs longer than 90 s (`EngineOptions::long_form_window_frames`)
are diarized by the cached long-form path in 30 s chunks instead of one
graph: a single graph loses speaker identity past about two minutes (DER
28-48 % on the 160 s and 191 s `abcba`/`abcdba` fixtures against 3.7-4.4 %
long-form), and inputs past 400 s now run instead of failing.

Install `huggingface_hub` and download the pinned official GGUF into `models/`:

```bash
python -m pip install huggingface_hub
python scripts/download_nemotron_diarization.py
```

Add `--include-source` to also download the original `.nemo` checkpoint. To
produce a separate Q8 GGUF from that checkpoint, prepare the pinned NVIDIA
converter checkout, install its Python requirements, and run:

```bash
python scripts/convert_nemotron_diarization.py --prepare-only
python -m pip install -r models/sources/NeMo-Speech.cpp/requirements.txt
python scripts/convert_nemotron_diarization.py
```

The conversion script downloads the checkpoint and checks the converter source
revision, updating a stale checkout to the pinned revision. It writes
`models/Nemotron-3-Diarization.converted.q8_0.gguf`
and leaves the official GGUF intact.

## Documented storage tiers

The table distinguishes recorded model tiers from additional converter
outputs. Format availability does not establish equivalent quality or backend
parity; see the [checkpoint table](../README.md#supported-checkpoints),
[backend validation](backends.md) and [testing guide](testing.md).

| Model | Storage and quantization |
|---|---|
| `nvidia/parakeet-ctc-0.6b` | `f32`, `f16`, `q8_0`, `q5_0`, `q4_0` |
| `nvidia/parakeet-ctc-1.1b` | `f16`, `q8_0` |
| `ai4bharat/indic-conformer-600m-multilingual` | `f16`, `q8_0`, `q4_0` |
| `nvidia/parakeet-unified-en-0.6b` | `q8_0` |
| `nvidia/parakeet-tdt-0.6b-v3` | `f32`, `f16`, `q8_0`, `q5_0`, `q4_0` |
| `nvidia/parakeet-tdt-1.1b` | `f16`, `q8_0` |
| `qvac/TranscribePsy-250M-EU-sub` | published `q8_0` GGUF; encoder-less GGUF with 6-bit palettized Core ML encoder and decoder |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | `f16` |
| `OpenMOSS-Team/MOSS-Transcribe-Diarize` | `f16`, `q8_0`, `q5_0`; converter also writes `f32`, `bf16` |
| `nvidia/parakeet_realtime_eou_120m-v1` | `f16`, `q8_0` |
| `nvidia/diar_sortformer_4spk-v1` | `f16`, `q8_0`, `q4_0` |
| `nvidia/diar_streaming_sortformer_4spk-v2` | `f16`, `q8_0`, `q4_0` |
| `nvidia/diar_streaming_sortformer_4spk-v2.1` | `f16`, `q8_0`, `q4_0` |
| `nvidia/Nemotron-3-Diarization` | official `q8_0` GGUF |

## EOU artifact preparation

From the repository root, download EOU, verify an F16 export, and create the
Q8_0 runtime file. Install the [conversion environment](#models-and-conversion)
first. Optional Core ML export and exact-shape routing are documented in
[backends.md](backends.md#core-ml-encoder-sidecar).

```sh
engines/parakeet/scripts/download-all-models.sh eou
python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt engines/parakeet/models/parakeet_realtime_eou_120m-v1.nemo \
  --hf-repo nvidia/parakeet_realtime_eou_120m-v1 \
  --out engines/parakeet/models/parakeet_realtime_eou_120m-v1.f16.gguf --quant f16
python engines/parakeet/scripts/verify-gguf-roundtrip.py \
  --nemo engines/parakeet/models/parakeet_realtime_eou_120m-v1.nemo \
  --gguf engines/parakeet/models/parakeet_realtime_eou_120m-v1.f16.gguf
python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt engines/parakeet/models/parakeet_realtime_eou_120m-v1.nemo \
  --hf-repo nvidia/parakeet_realtime_eou_120m-v1 \
  --out engines/parakeet/models/parakeet_realtime_eou_120m-v1.q8_0.gguf --quant q8_0

```
