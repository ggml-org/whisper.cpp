# audiogen engine: memory-fit preflight

Part of the [audiogen engine documentation](../README.md).

Both engines can project whether a model and a generation workload fit the
memory available right now, before any weight is read.

| Engine | Library entry point | Command line |
|---|---|---|
| ACE-Step | `tts_cpp::acestep::fit_params` in `audiogen-cpp/acestep/fit.h` | [`acestep-fit-params`](cli.md#acestep-fit-params) |
| MiniMax-Music3 | `tts_cpp::minimax::fit_params` in `audiogen-cpp/minimax/fit.h` | none |

## MiniMax-Music3

```cpp
#include <audiogen-cpp/minimax/fit.h>

tts_cpp::minimax::EngineOptions options;
options.model_dir = "/path/to/minimax-music3";
options.device    = "gpu";

tts_cpp::minimax::FitWorkload workload;
workload.max_frames    = 3000;  // 120 s at 25 frames per second
workload.prompt_tokens = 900;   // tokenized caption, lyrics and prompt template

const tts_cpp::minimax::FitResult fit = tts_cpp::minimax::fit_params(options, workload);
```

`EngineOptions` is the struct `Engine::create` takes, with the same model
resolution (`model_dir` or both `lm_model_path` and `synth_model_path`), the
same `device` semantics (`cpu`, `gpu` or `auto`; empty defers to
`MM3_DEVICE`, then `cpu`) and the same `backends_dir`. `device = "gpu"` without
a usable GPU is an error, as it is for a load.

When `coreml_conditioner_path` is set, the preflight returns `Error` with
reason `coreml-memory-estimate-unavailable`. It cannot project Core ML's
resident model and prediction allocations, so it does not report the native
estimate as a hybrid fit verdict. This does not prevent `Engine::create` or
`generate` from using the Core ML conditioner.

| `FitWorkload` field | Default | Meaning |
|---|---|---|
| `max_frames` | 300 (12 s) | `GenerateParams::max_frames`; capped at the checkpoint's `mm3.max_audio_frames` exactly as `generate()` caps it |
| `prompt_tokens` | 1024 | Tokenized prompt length; above `mm3.max_prompt_tokens`, or when the prompt and the frames exceed `qwen3.context_length`, the result is `workload-too-large` |
| `margin_bytes` | 256 MiB | Free memory that must remain for the projection to count as fitting |

The fitter opens both GGUFs metadata-only, so a weightless copy of the pair
gives the same answer as the full files. It wires every tensor through the
engine's own loaders, builds the runtime's LM, depth, condition, DiT and
vocoder graphs at the workload's shapes, and prices each one through the same
ggml scheduler layout the engine allocates it with, so weight, KV-cache and
compute buffers come out as a real load allocates them, including the host
buffer in which a GPU scheduler stages graph inputs. Nothing is allocated on a
device and no graph runs.

The projection follows the engine's residency. The weights stay resident from
`Engine::create`, and the depth, condition, DiT and vocoder graphs stay
allocated from the first `generate()` until the engine is destroyed. The LM's
KV cache and graphs are freed after the autoregressive stage and rebuilt by the
next call, so from the second call on they sit beside everything else. That
steady state is the device peak. Host memory is the largest of the stage
phases: the autoregressive stage with its per-frame hidden states, the flow
windows, the vocoder preparation, the stitched waveform, and the final
interleaved PCM. On the CPU and on devices that share system memory, both
requirements are charged against the same free memory.

`FitResult` has the ACE-Step shape. Its `stages` rows are `lm`, `depth`,
`cond`, `dit` and `vocoder`: `weights_bytes` includes the derived buffers a
stage keeps beside its weights, `state_bytes` is a KV cache, `compute_bytes`
is the device side of the stage's graph buffers, and `host_bytes` is what the
stage holds in system memory. `status` is `Success` (`fits`), `Failure`
(`does-not-fit`, a valid answer) or `Error` with one of `invalid-arguments`,
`model-unreadable`, `workload-too-large`, `no-backend-device` or
`measurement-failed` or `coreml-memory-estimate-unavailable`; `report` carries
a readable summary or the error.

The tokenizer tables, ggml's per-graph CPU work buffers and the scheduler's own
bookkeeping are not projected; the margin covers them.
