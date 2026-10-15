# tts engine: APIs, pipelines, and surfaces

Part of the [tts engine documentation](../README.md).

### API, streaming, and CLI matrix

| Family | Installed header | Streaming API | CLI surface |
|---|---|---|---|
| Chatterbox | `<tts-cpp/chatterbox/engine.h>` | true incremental S3Gen/HiFT chunks | `tts-cli` batch and streaming |
| Supertonic 1/2/3 | `<tts-cpp/supertonic/engine.h>` | text-chunk pipeline streaming | `supertonic-cli` batch/streaming; `tts-cli` batch only and rejects streaming flags |
| Parler | `<tts-cpp/parler/engine.h>` | incremental callback via `stream_chunk_frames` | `parler-cli` and `tts-cli` batch only; neither exposes streaming |
| CosyVoice3 | `<tts-cpp/cosyvoice/engine.h>` | callback is post-hoc chunking after full generation | `cosyvoice-cli` |
| Audio8 | `<tts-cpp/audio8/engine.h>` | no | `audio8-cli` |
| Pocket TTS | `<tts-cpp/pocket/engine.h>` | incremental FlowLM/Mimi PCM callbacks | `pocket-cli` batch WAV and memory preflight |
| MOSS Delay / TTSD | `<tts-cpp/moss/engine.h>` | incremental codec PCM callbacks | `moss-cli` batch/streaming and dialogue |
| MOSS-SoundEffect | `<tts-cpp/moss/sound_effect.h>` | progress callback; no incremental PCM | `moss-cli --mode sfx` |
| MOSS-Speech | `<tts-cpp/moss/speech.h>` | progress callback; no incremental PCM | `moss-cli --mode s2s` |
| LavaSR | `<tts-cpp/lavasr/{denoiser,enhancer}.h>` | block-oriented enhancement APIs | `lavasr-bench` |

`tts-cli` is intentionally a limited metadata dispatcher: it handles
Chatterbox, batch Supertonic, and a reduced Parler surface. Use the dedicated
CLIs for the complete Supertonic, Parler, CosyVoice3, Audio8, Pocket and MOSS options.

### Defaults that differ by surface

| Setting | `tts-cli` | Library engines and dedicated CLIs |
|---|---|---|
| Seed | `0` unless `--seed` is supplied | model-specific: most synthesis engines use `42`; Pocket uses `1234`; see the model header/CLI for exceptions |
| Threads | model-specific; explicit `--threads` wins | Chatterbox, Parler and Audio8 cap automatic selection at 4. Supertonic uses 4 on GPU or CPU BLAS paths; its fused CPU path leaves an eighth of logical CPUs free. Pocket defaults to one thread per worker; CosyVoice stage defaults are model-specific |
| Parler greedy | repaired to sampling with a warning because argmax does not terminate | same; use `seed` for reproducibility |
| Chatterbox streaming CFM | Turbo accepts 1 or 2 steps | Multilingual standard CFM requests below the model timestep count are floored to 10 |

## Public APIs

The installed CMake target is `tts-cpp::tts-cpp`. Persistent engine classes
load their model once and reuse it across synthesis calls:

| Namespace | Primary surface | Result |
|---|---|---|
| `tts_cpp::chatterbox` | `Engine::synthesize` | 24 kHz PCM by default; Turbo/Multilingual selected by GGUF metadata |
| `tts_cpp::supertonic` | `Engine::synthesize` | model-rate PCM, usually 44.1 kHz; `Engine::vocoder_on_coreml()` reports whether the Core ML vocoder sidecar loaded and `SynthesisResult::vocoder_synthesis_backend` where each call's vocoder ran (`"ggml"`, a `coreml-*` label, or `"mixed"` across streamed chunks), see the [Supertonic guide](supertonic.md#core-ml-vocoder-sidecar) |
| `tts_cpp::parler` | `Engine::synthesize` | 44.1 kHz PCM conditioned by a description |
| `tts_cpp::cosyvoice` | `Engine::synthesize` | 24 kHz PCM |
| `tts_cpp::audio8` | `Engine::synthesize` | 44.1 kHz PCM, optionally cloned from `VoicePrompt`; `Engine::codec_on_coreml()` reports whether the Core ML codec sidecar is attached (false once a failed call retires it) and `SynthesisResult::codec_synthesis_backend` where each call's codec synthesis ran (`"ggml"` or a `coreml-*` label), see the [Audio8 guide](audio8.md#core-ml-codec-sidecar) |
| `tts_cpp::pocket` | `Engine::synthesize` / `synthesize_stream` | CPU FlowLM/Mimi, 24 kHz by default; prepared voice or encoder-enabled cloning |
| `tts_cpp::moss` | `Engine::synthesize` / `synthesize_stream` | MOSS Delay / TTSD synthesis, cloning and dialogue; `EngineOptions::backend = "hexagon"` runs it on the [Snapdragon NPU](moss.md#hexagon-npu); `Engine::codec_on_coreml()` reports whether the [Core ML codec sidecar](moss.md#core-ml-codec-decoder) is attached and `SynthesisResult::codec_backend` where the request decoded (`"ggml"`, a `coreml-*` label, or `"mixed"`) |
| `tts_cpp::moss` | `SoundEffectEngine::generate` | 48 kHz sound-effect PCM; `dit_on_coreml()` / `vae_on_coreml()` and `SoundEffectResult::dit_backend` / `vae_backend` report the [Core ML sidecars](moss.md#core-ml-dit-and-vae) |
| `tts_cpp::moss` | `SpeechEngine::respond` | Spoken reply PCM or text, with generation/progress metadata; `tokenizer_on_coreml()` and `SpeechResult::tokenizer_backend` report the [Core ML speech tokenizer](moss.md#core-ml-speech-tokenizer) |
| `tts_cpp::lavasr` | `Denoiser` / `Enhancer` | enhanced PCM |

The public surface also includes Chatterbox's lower-level
`s3gen_synthesize_to_wav`, `s3gen_preload`, and `s3gen_unload`, plus
`tts_cpp_cli_main`. Symbols without `TTS_CPP_API`, including `detail`
namespaces used by tests, are private and hidden from shared-library consumers.

### Consumer integration

Downstream projects consume this engine through `speech-cpp[tts]`, which
depends on the shared `ggml-speech` package. The installed CMake package and
imported target retain the `tts-cpp` name. See the
[package guide](../../../docs/BUILD.md#consumable-packages) for features.
Integration uses:

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
[Useful CMake options](build.md#useful-cmake-options) for the full
flag table. The streaming text chunker (`split_for_streaming`:
sentence/clause/whitespace boundary priority, first-chunk latency knob,
tiny-tail merge, CJK sentence ends) is pinned model-free by
`test-supertonic-chunker`.

## Pipelines

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

`tts-cli` handles both Chatterbox variants via `chatterbox.variant` metadata.
There is no separate `chatterbox` executable target. Supertonic and Parler
GGUFs are autodetected from their architecture metadata.

| Stage         | Turbo                                      | Multilingual                                        |
|---------------|--------------------------------------------|-----------------------------------------------------|
| Tokenizer     | GPT-2 byte-level BPE (English)             | HuggingFace `tokenizers.json` (23 langs, NFKD pre)  |
| T3 backbone   | GPT-2 Medium, 24 layers, single forward    | Llama-520M, 30 layers, CFG cond+uncond per token    |
| CFM solver    | Meanflow, 2 Euler steps                    | Standard, 10 Euler steps with `cfg_rate=0.7`        |
| HiFT vocoder  | shared (same checkpoint format)            | shared (same checkpoint format)                     |

The other synthesis pipelines are:

| Family | Pipeline |
|---|---|
| Supertonic | text preprocessing → duration → text encoder → vector estimator → vocoder |
| Parler | Flan-T5 description encoder → delay-pattern decoder → DAC |
| CosyVoice3 | Qwen2.5 LM → DiT flow → CausalHiFT |
| Audio8 | DualAR semantic/fast LM → 10-codebook codec decoder |
| Pocket | FlowLM flow sampling → streaming Mimi codec |
| MOSS Delay / TTSD | Qwen3 delay-pattern backbone → RVQ codec |
| MOSS-SoundEffect | Text encoder → flow DiT → DAC decoder |
| MOSS-Speech | Speech tokenizer → split Qwen3 text/audio branches → CosyVoice2 flow/HiFT |
| LavaSR | optional UL-UNAS denoiser → Vocos bandwidth-extension enhancer |

## Memory-fit preflight

Metadata-only preflight projects model/workload memory before loading weights.
Result statuses and reasons are declared in `<tts-cpp/fit.h>`; preflight does
not validate tensor payloads or guarantee memory remains free until loading.

| Family | Public header | Entry point / CLI |
|---|---|---|
| Chatterbox | `<tts-cpp/chatterbox/fit.h>` | `chatterbox::fit_params`, `chatterbox-fit-params` |
| Supertonic | `<tts-cpp/supertonic/fit.h>` | `supertonic::fit_params`, `supertonic-fit-params` |
| Parler | `<tts-cpp/parler/fit.h>` | `parler::fit_params`, `parler-fit-params` |
| CosyVoice3 | `<tts-cpp/cosyvoice/fit.h>` | `cosyvoice::fit_params`, `cosyvoice-fit-params` |
| Audio8 | `<tts-cpp/audio8/fit.h>` | `audio8::fit_params`, `audio8-fit-params` |
| Pocket | `<tts-cpp/pocket/fit.h>` | `pocket::fit_params`, `pocket-cli --fit`; [workload guide](pocket-tts.md#native-memory-preflight) |
| MOSS Delay / TTSD | `<tts-cpp/moss/fit.h>` | `moss::fit_params(options, workload)` |
| MOSS-SoundEffect | `<tts-cpp/moss/sound_effect_fit.h>` | `moss::fit_params(options, request, margin_bytes)` |

### MOSS memory fit


`tts-cpp/moss/fit.h` provides `tts_cpp::moss::fit_params(options, workload)`
for MOSS-TTS-v1.5 and MOSS-TTSD. Pass the same `EngineOptions` as synthesis,
and specify all four workload fields explicitly:

```cpp
#include <tts-cpp/moss/fit.h>

tts_cpp::moss::EngineOptions options;
options.backbone_path = "moss-tts-delay-f16.gguf";
options.decoder_path = "moss-codec-decoder-f16.gguf";
options.use_gpu = true;
const tts_cpp::moss::FitWorkload workload(128, 0, false, 256ull * 1024 * 1024);
const auto result = tts_cpp::moss::fit_params(options, workload);
```

The fields are the full native prompt row count, total mono reference samples
(0 without a reference), streaming intent, and memory headroom in bytes.
Prompt rows must include the frontend's special tokens and any encoded speaker
references or TTSD continuation rows. The fitter reads no reference recordings;
the caller supplies their sample count. A positive reference count requires
`encoder_path`. Use the TTSD backbone and include the total reference workload
for dialogue. The existing runtime context, duration and reference validation
also applies to the projection; generation settings remain unchanged.

The projection reads GGUF metadata without reading weights, allocating device
buffers, or running inference. Weightless GGUFs with the same tensor descriptors
produce the same estimate. It uses the runtime LM and codec graph builders and
size-only ggml allocators, including KV state, decoder streaming state, encoder
work, host staging and generated PCM. Batch decoding follows the engine's
existing windowed path for long sequences. Component peaks are summed as a
conservative upper bound, including encoder allocations that can be released
before synthesis. This can report `does-not-fit` for a workload whose actual
peak is smaller. On CPU, Metal unified memory and Hexagon, host and device
requirements compete for the same free memory. A Hexagon session reports no
memory of its own, so its projection reads the host's, and the embedding
tables and text head it keeps on the host count as host bytes.

The result follows `tts-cpp/fit.h`: `Success`/`fits`, `Failure`/`does-not-fit`,
or `Error` for invalid arguments, an unreadable model, or failed measurement.
Run `test-moss-fit` for model-free metadata, workload, streaming, reference,
validation and runtime-regression coverage.

### MOSS-SoundEffect memory preflight

The sound-effect fitter uses the same GGUF metadata, tokenizer, request validation
and text, DiT and VAE graph builders as generation. It neither uploads weights nor
executes a graph, and accepts a GGUF containing only its metadata and tensor table.

```cpp
#include <tts-cpp/moss/sound_effect_fit.h>

tts_cpp::moss::SoundEffectOptions options;
options.model_path = "moss-sfx-v2-q8_0.gguf";
tts_cpp::moss::SoundEffectRequest request;
request.prompt = "Rain falling on a tin roof.";
request.seconds = 8;
const auto result = tts_cpp::moss::fit_params(options, request, 0);
```

`request` accepts the existing generation controls and model defaults. DiT always
uses the model's full latent duration; requested seconds affect decoder windows
and output storage. The shared graph allocator is priced at its peak across phases:
`lm_compute_bytes` covers text/DiT, and `codec_compute_bytes` is any additional
VAE demand. Their sum is the device compute peak. Host bytes include CPU fallback
arenas, graph/scheduler descriptors, tokenizer storage, conditioning, diffusion
vectors and decoded audio. The margin is explicit in the native API.
