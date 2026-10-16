# tts engine: backend selection

Part of the [tts documentation](../README.md).

Backends are compiled into, or loaded by, the shared speech-branch ggml.
See the [capability table](../README.md#capabilities) for model validation;
the ability to compile a ggml backend does not establish model parity.

## Runtime selection

Where a build carries both CUDA and Vulkan, selection prefers CUDA on NVIDIA:
measured on one binary on an RTX 3090, Chatterbox Turbo is 8.6x faster on CUDA
than on that card's Vulkan adapter. Set `TTS_CPP_GPU_BACKEND` to `cuda`,
`vulkan`, `metal` or `opencl` to pin one for a test arm or a comparison; an
unrecognised value is rejected rather than silently dropping to the CPU.
Among several Vulkan adapters, Audio8 takes the one with the most free memory
and never an integrated adapter while a discrete one is visible; Supertonic and
CosyVoice3 take `EngineOptions::vulkan_device` (default: the first adapter).


Explicit Hexagon placement is supported for [Supertonic 3](supertonic.md#snapdragon-hexagon-npu),
[Parler mini](parler.md#snapdragon-hexagon-npu),
[CosyVoice3](cosyvoice3.md#snapdragon-hexagon-npu),
[Audio8](audio8.md#hexagon-npu-snapdragon) and
[MOSS-TTS / MOSS-TTSD](moss.md#hexagon-npu),
[MOSS-SoundEffect](moss.md#hexagon-npu-generation) and
[MOSS-Speech](moss.md#hexagon-npu-speech-to-speech); it is not an automatic GPU fallback.

## Apple Core ML sidecars

`TTS_CPP_COREML=ON` enables optional Supertonic vocoder, Audio8 codec and
MOSS sidecars on Apple. Their guides own naming, export commands, accepted
shapes, low-bit restrictions, placement and runtime status:

- [Supertonic vocoder](supertonic.md#core-ml-vocoder-sidecar)
- [Audio8 codec](audio8.md#core-ml-codec-sidecar)
- [MOSS-TTS / MOSS-TTSD codec decoder](moss.md#core-ml-codec-decoder)
- [MOSS-SoundEffect DiT and VAE](moss.md#core-ml-dit-and-vae)
- [MOSS-Speech speech tokenizer](moss.md#core-ml-speech-tokenizer)

Other TTS model families run entirely through their ggml paths.
