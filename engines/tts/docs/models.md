# tts engine: models and storage formats

Part of the [tts documentation](../README.md).

All native models use GGUF. A synthesis may require several stage files;
the per-model guide owns acquisition, conversion, filenames and validated
combinations. The [capability table](../README.md#capabilities) records voice
sources, sample rates and backend validation.

## Storage tiers

Converter output types and validated runtime tiers are not interchangeable
claims. Where a converter writes additional types, the table says so; use the
model guide for quality limits and stage-specific restrictions.

| Model | Storage and quantization | Acquisition and conversion |
|---|---|---|
| Chatterbox Turbo | `f16`, `q8_0`, `q5_0`, `q4_0` | [Guide](chatterbox.md) |
| Chatterbox Multilingual | `f16`, `q8_0`, `q5_0`, `q4_0` | [Guide](chatterbox.md) |
| Supertonic v1 | `f32`, `f16`, `q8_0`; `q4_0` runtime via requantization | [Guide](supertonic.md) |
| Supertonic v2 | `f32`, `f16`, `q8_0`; `q4_0` runtime via requantization | [Guide](supertonic.md) |
| Supertonic v3 | `f32`, `f16`, `q8_0`; `q4_0` runtime via requantization | [Guide](supertonic.md) |
| Parler-TTS mini-v1 | `f32`, `f16`, `q8_0`, `q6_k` | [Guide](parler.md) |
| Parler-TTS large-v1 | `f32`, `f16`, `q8_0`, `q6_k` | [Guide](parler.md) |
| Indic Parler-TTS | `f32`, `f16`, `q8_0`, `q6_k` | [Guide](parler.md) |
| Fun-CosyVoice3-0.5B | `f32`; LM and flow also `q8_0`, `q4_0`; flow and HiFT also `f16`; flow also `bf16` | [Guide](cosyvoice3.md) |
| Audio8-TTS-Preview-0.6B | `f32`, `f16`, `q8_0`; LM also `q4_0` | [Guide](audio8.md) |
| Pocket TTS | `f32`; `f16` as storage | [Guide](pocket-tts.md) |
| MOSS Delay (MOSS-TTS-v1.5 / MOSS-TTSD) | `f32`, `f16` | [Guide](moss.md) |
| MOSS-SoundEffect-v2 | `f16`, `q8_0`, `q8_0` with `q4_0` DiT blocks; converter also writes `f32`, `bf16`, `q4_0` | [Guide](moss.md) |
| MOSS-Speech | LM `bf16`, `q8_0`; converter also writes `f16`, `f32`; codec `f16` | [Guide](moss.md) |
| LavaSR denoiser (UL-UNAS) | `f32`, `f16` | [Guide](lavasr.md) |
| LavaSR enhancer (Vocos BWE) | `f32`, `f16` | [Guide](lavasr.md) |

Pocket F16 storage expands to F32 at load and does not reduce resident weight
memory. Parler's mixed Q6 tier preserves sensitive tensors at higher precision;
it is not a uniform six-bit model. CosyVoice's preferred flow tier depends on
the backend; [its guide](cosyvoice3.md#convert) documents that choice and the
unsupported LM F16 caveat. Supertonic's converter offers F32/F16/Q8_0;
low-bit vocoders do not automatically use the reference-precision Core ML sidecar.
