# audiogen-cpp

Native [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) text-to-music in pure C++ on [ggml](https://github.com/tetherto/qvac-ext-ggml): caption and lyrics in, 48 kHz stereo audio out, no Python or PyTorch at inference time.

| Property | Value |
|---|---|
| CMake project | `audiogen-cpp` v0.1.0 |
| Public API | `tts_cpp::acestep::Engine` (`include/audiogen-cpp/acestep/engine.h`) |
| Output | interleaved stereo PCM, 48 kHz, `pcm[t * 2 + ch]` |
| Backends | CPU, Vulkan (including Android Mali iGPUs), Metal |
| ggml | requires the `ggml-speech` port for the custom `ggml_snake` and `ggml_col2im_1d` ops |
| Consumed by | the `@qvac/audiogen-ggml` addon in [QVAC](https://github.com/tetherto/qvac) |

## Pipeline

```
caption + lyrics (+ bpm, key, time signature, language)
   |
   v
text encoder (Qwen3-Embedding)  -> prompt embeddings
LM (acestep-5Hz-lm, Qwen3 causal) -> lyric + acoustic codes
FSQ detokenizer                 -> DiT context latents
text encoder + cond encoder     -> cross-attention states
DiT (flow matching, Euler)      -> 64-channel acoustic latent
VAE (AutoencoderOobleck)        -> 48 kHz stereo PCM
```

The VAE upsamples by 1920, so `T_audio = T_latent * 1920`. Latents are time-major, `latent[t * 64 + c]`.

## Model stages

Four GGUFs, one per stage. Point `--models <dir>` at a directory and the engine classifies by filename stem, or pass explicit per-stage paths, which always win over the scan.

| Stage | Upstream model | Filename stems matched | Flag |
|---|---|---|---|
| Text encoder | Qwen3-Embedding | `embedding`, `text-enc`, `textenc` | `--text` |
| LM | ACE-Step 5 Hz LM (Qwen3 causal) | `-lm`, `lm-`, `_lm`, `ace-lm`, `5hz-lm` | `--lm` |
| DiT | ACE-Step v1.5 diffusion transformer | `turbo`, `dit`, `v15`, `sft` | `--dit` |
| VAE | AutoencoderOobleck | `vae` | `--vae` |

The most specific stems (`embedding`, `vae`) are tested first so no short token such as `lm` can claim an unrelated file.

### DiT variants

Detected from the `acestep.is_turbo` GGUF key; absent means base or sft.

| Variant | Default steps | Default shift | Notes |
|---|--:|--:|---|
| turbo | 8 | 3.0 | fastest, no CFG on the DiT |
| base / sft | 50 | 1.0 | DiT CFG and APG (`guidance > 1`) are deferred |

Weights load quantized. `f32`, `f16`, and `bf16` are handled for norms and biases, and the detokenizer's `special_tokens` may be `q8_0`. This tree ships no converter: the stage GGUFs come from `convert.py` in [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp), the upstream C++/ggml implementation this port follows, which also publishes pre-quantized GGUFs at [Serveurperso/ACE-Step-1.5-GGUF](https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF).

## Backends

`n_gpu_layers > 0` (`--gpu` on the CLI) selects a GPU backend through the ggml
registry. Discrete GPUs are preferred, then integrated GPUs; this second class
is required because Vulkan reports Android UMA adapters such as Pixel's
Mali-G715 as `IGPU`.

| Stage | Placement when a GPU is selected |
|---|---|
| DiT, VAE | GPU |
| Text encoder, cond encoder | GPU, unless `ACESTEP_ENCODERS_CPU=1` |
| LM | GPU on Metal; CPU on Vulkan and every unmeasured backend |
| FSQ detokenizer | GPU on Vulkan and Metal, CPU on every other backend |

The LM and detokenizer use an allowlist rather than a denylist: a backend keeps the CPU placement until the stage has been measured on it, so adding one cannot silently regress generated audio. Metal is validated for both stages. Vulkan is validated for the detokenizer, but the autoregressive LM remains on CPU because Mali-G715 testing showed code collapse and early termination on Vulkan.

Measurement is against an F32-dequantized reference (`scripts/dequant_gguf.py`), not against CPU. CPU is not automatically ground truth for a quantized model — ggml's CPU matmul quantizes activations to Q8_1 internally. On Metal the LM reproduces the F32 argmax trajectory exactly where CPU Q8_0 diverges at the first token. On Mali Vulkan, however, the LM produces repeated semantic codes and can terminate at roughly half the requested duration, while the same device produces a diverse full-length sequence with the LM on CPU.

On a GPU that supports F32 flash attention, Phase 2 also decodes the conditional and unconditional CFG paths in one batched graph. This matches the reference `acestep.cpp` LM path: the same prompt, model, sampler settings, and seed produce the same semantic-code sequence. Unsupported backends keep the separate F32 manual-attention path.

The policy itself lives in [`src/acestep/stage_placement.h`](src/acestep/stage_placement.h), separate from the engine, so it is unit tested without a GPU.

### Environment overrides

Applied after the allowlist. CPU wins when both are set for the same stage.

| Variable | Effect |
|---|---|
| `ACESTEP_LM_GPU` / `ACESTEP_LM_CPU` | force the LM onto the GPU or the CPU |
| `ACESTEP_DETOK_GPU` / `ACESTEP_DETOK_CPU` | force the detokenizer onto the GPU or the CPU |
| `ACESTEP_ENCODERS_CPU` | move the encoders to the CPU to trim wired memory |
| `ACESTEP_VAE_GPU` | force the VAE onto the GPU |
| `ACESTEP_KEEP_STAGES` | keep every stage resident instead of loading sequentially |
| `ACESTEP_LM_DUMP_LAYERS` | write the LM's per-layer prefill hidden states to this file path |
| `ACESTEP_LM_DUMP_TOKENS` | write the Phase-2 prompt token IDs as CSV |
| `ACESTEP_LM_DUMP_LOGITS` | write the conditional Phase-2 prefill logits as raw F32 |
| `ACESTEP_VAE_PROFILE` | print the VAE per-op-type time inventory |

Use `ACESTEP_LM_GPU` or `ACESTEP_DETOK_GPU` to take the measurement that would widen the allowlist for a new backend, without a rebuild.

### Parity debug hooks

`ACESTEP_PARITY_DEBUG` is a compile-time macro, not an environment variable, and there is no CMake option for it: configure with `-DCMAKE_CXX_FLAGS=-DACESTEP_PARITY_DEBUG`. The hooks below are `#ifdef`-ed out of `generate()` without it, so setting them on a default build does nothing.

| Variable | Effect |
|---|---|
| `ACESTEP_DUMP_DIR` | log the tokenizer shapes and write `our_dit_output.bin`, `our_noise.bin`, `our_context.bin`, `our_enc_hidden.bin` here for tensor-level comparison against acestep.cpp |
| `ACESTEP_INJECT_NOISE`, `ACESTEP_INJECT_ENC`, `ACESTEP_INJECT_CONTEXT` | replace the noise, encoder hidden states, or DiT context with an acestep.cpp `--dump` tensor to isolate a diverging stage |

Dumps use a flat `[ndim, d0, d1]` int32 header followed by the `f32` payload, the same format the injection hooks read.

### Memory

By default no stage stays resident after `create()`: `generate()` loads each stage immediately before its step and frees it right after, so the peak is one stage for the LM, detokenizer, DiT, and VAE steps rather than all six at once, which is what keeps a non-entitled iOS app inside its memory budget. The one overlap is the condition encoder: it supplies the silence frame that pads the DiT context before the text encoder runs and its own forward comes after text encoding, so it is loaded first and the two encoders are co-resident until the text encoder is freed. That pair is the documented peak. `ACESTEP_KEEP_STAGES=1` opts out and keeps every stage resident. VAE tiling for long tracks is handled internally with windows tuned to the Metal buffer limit and is deliberately not configurable.

## Build

Standalone, against an installed `ggml-speech`:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/ggml-install
cmake --build build -j
```

From the repo root the umbrella builds this engine behind `-DSPEECH_BUILD_AUDIOGEN=ON` (the default); see the [top-level README](../../README.md).

| Option | Default | Effect |
|---|---|---|
| `AUDIOGEN_BUILD_LIBRARY` | `ON` | build the library; linkage follows `BUILD_SHARED_LIBS` |
| `AUDIOGEN_BUILD_EXECUTABLES` | `ON` standalone, `OFF` as a subdirectory | CLIs and per-stage smoke harnesses |
| `AUDIOGEN_BUILD_TESTS` | `ON` standalone, `OFF` as a subdirectory | CPU-only unit tests |
| `AUDIOGEN_INSTALL` | `ON` | generate install rules |
| `AUDIOGEN_USE_SYSTEM_GGML` | `ON` | `find_package(ggml)`; required, there is no supported vendored ggml in this tree |
| `AUDIOGEN_CCACHE` | `ON` | use ccache when available |

Consume the installed package with `find_package(audiogen-cpp CONFIG REQUIRED)` and link `audiogen-cpp::audiogen-cpp`.

## Command line tools

| Binary | Purpose |
|---|---|
| `music-cli` | end-to-end text-to-music |
| `acestep-cli` | VAE decode and reconstruction roundtrip harness |
| `textenc-smoke`, `lm-smoke`, `lmgen-smoke`, `bpe-smoke`, `detok-smoke`, `cond-smoke`, `dit-smoke` | per-stage smoke harnesses |

### music-cli

```sh
# all four GGUFs in one directory
./build/music-cli --models models/acestep --out song.wav --dur 8 --seed 42

# explicit per-stage paths, GPU, custom prompt
./build/music-cli --dit dit.gguf --lm lm.gguf --text emb.gguf --vae vae.gguf \
                  --caption "driving synth pop, bright analog leads" \
                  --lyrics "[Instrumental]" --bpm 128 --key "C major" --tsig 4/4 \
                  --lang en --steps 8 --shift 3.0 --gpu --out song.wav
```

| Flag | Default | Meaning |
|---|---|---|
| `--models DIR` | | directory holding the four stage GGUFs |
| `--dit`, `--lm`, `--text`, `--vae` | | explicit per-stage GGUF paths |
| `--caption TEXT` | built-in pop-rock prompt | text prompt |
| `--lyrics TEXT` | `[Instrumental]` | lyrics |
| `--dur SECONDS` | `8` | target length, drives the LM code count |
| `--seed N` | `42` | negative means random |
| `--bpm N`, `--key STR`, `--tsig STR`, `--lang CODE` | inferred | optional metadata hints |
| `--steps N`, `--shift F` | per variant | sampler overrides |
| `--no-dcw` | DCW enabled | disable the official Haar low/high correction applied after each DiT step |
| `--temp F`, `--topp F`, `--topk N`, `--cfg F` | `0.85`, `0.9`, off, `2.0` | LM sampling for the audio codes |
| `--no-phase1` | off | skip the LM metadata auto-fill pass |
| `--req FILE` | | request JSON; pre-supplied `audio_codes` skip the LM stage |
| `--gpu`, `--threads N` | CPU, hardware concurrency | compute placement |
| `--dump-stages DIR` | | write one `.bin` per stage into an existing directory |

The sampler enables ACE-Step's single-level Haar DCW `double` mode by default.
At timestep `t`, the low band uses `t * 0.05` and the high band uses
`(1 - t) * 0.02`, matching the official Python defaults. The correction runs
on the host latent between DiT steps, so it is backend-independent and adds
negligible work compared with a transformer forward.

### acestep-cli

```sh
./build/acestep-cli --model vae.gguf --t-latent 32 --out out.wav
./build/acestep-cli --model vae.gguf --roundtrip --in in.wav --seconds 2.56 --out out.wav
```

The first form is the default mode: it decodes a synthetic latent, which checks that real weights load and that the decode graph (`ggml_col2im_1d` + `ggml_snake`) runs on the selected backend. `--roundtrip` encodes a real WAV and prints the per-channel reconstruction correlation, the audible end-to-end VAE check. Both forms take `--gpu`.

## Tests and parity

```sh
cmake -S . -B build -DAUDIOGEN_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/path/to/ggml-install
cmake --build build -j
ctest --test-dir build
```

`test-acestep-units` covers the weight-free CPU logic and needs no GGUFs. That includes the stage-placement policy above: the backend allowlist (both the `MTL` and `Metal` registry names), the CPU fallback for every unmeasured backend, and the environment override precedence.

Stage dumps are the tool for localising a backend divergence. Run the same prompt twice with `--dump-stages`, then compare:

| Script | Purpose |
|---|---|
| `scripts/stage_cos.py` | per-stage cosine and rel_l2 between two dump directories; fails under a worst-stage cosine of 0.999, and also on a missing, truncated, or mismatched counterpart |
| `scripts/dequant_gguf.py` | rewrite a GGUF with every quantized tensor dequantized to F32, giving a ground-truth trajectory to compare a quantized run against |

Both need `numpy`; `dequant_gguf.py` also needs `gguf`.

Reference parity against the upstream `acestep.cpp`: end-to-end synthesis correlates 0.98 to 0.99 on identical codes, and LM greedy decoding matches upstream argmax, with the residual difference attributable to CPU F32 versus Metal F16 arithmetic.

## Performance

There is no CI benchmark suite for this engine yet. `music-cli` and the engine print per-stage wall clock to stderr (`[music-cli]`, `[acestep-timing]`), which is the current way to compare backends and step counts.

## License

`audiogen-cpp` is MIT licensed; see [LICENSE](LICENSE). Model weights and upstream model code carry their own terms, listed in [NOTICE](NOTICE).
