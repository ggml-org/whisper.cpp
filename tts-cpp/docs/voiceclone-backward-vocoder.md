# Voice-clone backward — vocoder (QVAC-20983)

Scope for ticket *"6. GGML backward pass: vocoder"*. Make the Supertonic vocoder
differentiable for enrollment, with the **transposed convolution** called out as
the main risk op. This doc records the op × backend gap for the vocoder path and
the CPU-fallback behavior, and is committed alongside the deliverable: an
analytic, gradchecked C++ backward of the full vocoder
(`src/supertonic_vocoder_backward.{h,cpp}`).

It is the vocoder counterpart of `voiceclone-backward-gap-matrix.md` (text
encoder); the *why analytic* rationale and the Task-2 gradcheck contract are
shared.

## Why the gap exists

Voice cloning optimizes only `style_ttl` (model weights frozen). The vocoder
maps the CFM latent to a waveform, so the gradient enrollment needs from it is
`d(loss)/d(latent)` — the audio-loss gradient backpropagated through the frozen
vocoder into the latent, which then flows back through the vector estimator and
text encoder to `style_ttl`.

A fully GGML-native backward (the on-device goal) needs every op on the vocoder
path to have a case in `ggml_compute_backward` (`ggml/src/ggml.c`) **and** a CPU
kernel for the ops the backward expands into. Several are missing, and the
vocoder additionally leans on custom `ggml_supertonic_*` ops that only run on the
CPU backend.

## The "transposed convolution" risk, resolved

The ticket flags the transposed convolution as the main risk. In the Supertonic
vocoder there is **no `ggml_conv_transpose_*` op**: the time upsampling (factor
`ttl_chunk_compress_factor`) is realized as a fixed `reshape + permute + cont`
(the latent unpack, `build_supertonic_vocoder_cache`), i.e. a pure permutation
`x[(t*factor+r)*C_latent + c] = latent[(c*factor+r)*latent_len + t]`. Its
backward is the transpose gather (`latent_unpack_backward`), with no kernel risk
— the feared conv-transpose backward does not arise here.

## Forward ops on the vocoder path

Source: `src/supertonic_vocoder.cpp`.

| Forward op | Where (forward) |
| --- | --- |
| reshape/permute/cont | latent unpack (the notional "transposed conv" upsample) |
| `ggml_scale`, `ggml_mul`, `ggml_add` (broadcast) | denorm, BN affine, residuals, scalar gamma |
| `ggml_im2col` + `ggml_mul_mat` | causal conv1d (embed, pw1/pw2, head1/head2) |
| custom causal depthwise (`ggml_custom` / `ggml_supertonic_depthwise_1d_causal_ct`) | ConvNeXt depthwise |
| `ggml_norm` (+ `ggml_supertonic_layer_norm_channel*`) | ConvNeXt channel layer norm |
| `ggml_gelu_erf` (+ `ggml_supertonic_bias_gelu*`) | ConvNeXt FFN |
| leaky-relu lowering (`leaky_relu_portable_ggml`) | head PReLU |
| `ggml_supertonic_edge_pad_1d`, `convnext_block_fused` | fused CPU/Metal fast paths |

## Gap matrix

Legend: **OK** = implemented; **MISSING** = aborts / not implemented.

"Graph backward" = a case in `ggml_compute_backward` (`ggml/src/ggml.c`),
backend-agnostic. "CPU bwd kernel" = the kernels the backward expands into exist
for the CPU backend, which is the only backend enrollment needs. GPU columns are
out of scope for Phase 2 (enrollment runs on CPU).

| Op | Graph backward (ggml.c) | CPU bwd kernel | CUDA / Metal / Vulkan / OpenCL |
| --- | --- | --- | --- |
| `RESHAPE`/`PERMUTE`/`CONT` (latent unpack) | OK | OK | out of scope |
| `SCALE` / `ADD` / `MUL` (denorm, BN, gamma) | OK | OK | out of scope |
| `MUL_MAT` / `IM2COL` (conv1d) | OK (`mul_mat`/`im2col_back`) | OK | out of scope |
| `NORM` (channel layer norm) | **MISSING** | — | — |
| `GELU_ERF` (unary) | **MISSING** | — | — |
| leaky-relu / PReLU | partial (`STEP`/`RELU` only) | — | — |
| custom `ggml_supertonic_*` ops | **MISSING** | — | CPU-only forward (see below) |

Confirmed against the `ggml_compute_backward` switch: `NORM`, `GELU`/`GELU_ERF`
fall through to `GGML_ABORT`; the custom ops have no backward at all. This
mirrors the text-encoder matrix — the blocking gaps are the same `NORM` and the
elementwise activation (`GELU_ERF` here; the vector estimator adds it too).

## CPU fallback behavior (enrollment)

Two layers of CPU-only behavior matter for enrollment:

1. **Forward custom ops are CPU-only.** `GGML_OP_CUSTOM` is rejected on every GPU
   backend (CUDA / Metal / Vulkan / OpenCL), so the vocoder's custom causal
   depthwise, fused ConvNeXt block, edge-pad and `_ct` fused ops only execute on
   the CPU backend. On GPU backends the forward already falls back to the
   pure-GGML `im2col + mul_mat` / granular-op chain (see
   `supertonic_use_cpu_custom_ops()` / `supertonic_use_fused_supertonic_ops()`
   guards in `supertonic_vocoder.cpp`).
2. **The backward runs on CPU, analytically.** Because `NORM`, `GELU_ERF` and the
   custom ops have no GGML backward, the enrollment gradient cannot be produced
   by `ggml`'s autodiff on any backend today. The differentiable vocoder is
   therefore provided as the analytic C++ backward in this PR, which runs on the
   CPU (the enrollment target). **Every backend must fall back to CPU for the
   vocoder backward during enrollment.** This is acceptable: enrollment is a
   one-time, offline optimization loop, not the realtime synthesis path (which
   keeps its GPU fast paths unchanged).

## Solution shipped in this PR

The `VocoderBackward` class (`src/supertonic_vocoder_backward.{h,cpp}`) owns the
frozen weights and caches per-call activations as state: `forward(latent)` runs
the chain and `backward(d_wav)` consumes the cached activations to return
`d(loss)/d(latent)`. It is model-free and validated component-wise against
central finite differences via the Task-2 gradcheck harness
(`test/test_supertonic_vocoder_backward.cpp`, always-on `unit` tier). The
stateless math primitives are exposed as static members, each gradchecked
individually:

- `denorm_backward_input` — latent denormalization
- `conv1d_causal_backward_input` — full causal conv1d (embed / head1)
- `depthwise_causal_backward_input` — causal depthwise conv1d
- `batch_norm_backward_input` — affine BN at inference
- `leaky_relu_backward` — head PReLU
- `latent_unpack_backward` — the "transposed conv" upsample (permutation)
- `convnext_backward_input` — full scalar-gamma ConvNeXt block
- `VocoderBackward::backward` — the whole chain → `d(loss)/d(latent)`

Channel layer norm, erf-GELU and pointwise (1x1) convs are shared with the
vector-estimator backward (`tts_cpp::ve_grad`), since the math is identical.

This is mathematically exact, runs on CPU, and serves as the reference oracle for
the per-stage gradcheck once the GGML-native ops (`NORM`, `GELU_ERF`, custom-op
backward / lowering) are implemented.
