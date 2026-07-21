# Voice-clone backward — op × backend gap matrix (Phase 2)

Scope for ticket *"GGML backward pass: Supertonic text encoder"*. This doc
scopes the remaining Phase-2 work to make the Supertonic text encoder
**differentiable in GGML** on the CPU path used for enrollment, and records which
backward ops are still missing in the vendored `ggml`.

It is committed alongside the interim deliverable of this PR: an analytic,
gradchecked C++ backward of the style-dependent tail
(`src/supertonic_text_encoder_backward.{h,cpp}`). See
[Interim vs Phase-2](#interim-solution-shipped-in-this-pr) below for how the two
relate.

## Why the gap exists

Voice cloning optimizes only `style_ttl` (model weights frozen). `style_ttl`
enters the text encoder in exactly one place — the two speech-prompted attention
layers plus the final channel layer norm — so the gradient we need is
`d(text_emb_out)/d(style_ttl)` flowing through that tail.

A fully GGML-native backward (the Phase-2 goal, needed by the on-device
enrollment loop) requires every op on that path to have a backward in
`ggml_compute_backward` (`ggml/src/ggml.c`) **and** a CPU kernel for the ops the
backward expands into. Several are missing today.

## Forward ops on the style-dependent path

Source: `src/supertonic_text_encoder.cpp` (line refs approximate).

| Forward op | Where (forward) |
| --- | --- |
| `ggml_mul_mat` | linear/dense projections (q/v/o, dense_time) |
| `ggml_add` (broadcast) | bias adds, residuals |
| `ggml_mul`, `ggml_scale` | attention scaling, rel-pos terms |
| `ggml_soft_max` | rel-pos attention (`relpos_attention`, ~L550) |
| `ggml_flash_attn_ext` | speech-prompted attention (~L818, ~L928) |
| `ggml_norm` | channel layer norm (`layer_norm_ggml`, ~L233) |
| `ggml_gelu_erf` | convnext FFN (`text_convnext_ggml`, ~L276) |
| `ggml_concat` | edge-clamp padding (`edge_clamp_pad_1d`, ~L185) |
| `ggml_im2col` | conv1d / depthwise conv (~L164, ~L216) |
| `ggml_cont`/`reshape`/`permute`/`transpose` | layout shuffles |

## Gap matrix

Legend: **OK** = implemented; **MISSING** = aborts / not implemented; **n/a** =
not on the enrollment path.

"Graph backward" = a case in `ggml_compute_backward` (`ggml/src/ggml.c`). It is
backend-agnostic: if it aborts, no backend can differentiate the op. "CPU bwd
kernel" = the kernels the backward expands into exist for the CPU backend
(`ggml-cpu`), which is the only backend enrollment needs in Phase 2. GPU columns
are out of scope for Phase 2 (enrollment runs on CPU) and tracked here only for
visibility.

| Op | Graph backward (ggml.c) | CPU bwd kernel | CUDA / Metal / Vulkan / OpenCL |
| --- | --- | --- | --- |
| `MUL_MAT` | OK | OK (`out_prod`/`mul_mat`) | out of scope |
| `ADD` / `MUL` / `SCALE` | OK | OK | out of scope |
| `CONT`/`RESHAPE`/`PERMUTE`/`TRANSPOSE` | OK | OK | out of scope |
| `SOFT_MAX` | OK (`soft_max_back`) | OK | out of scope |
| `IM2COL` | OK (`im2col_back`) | OK | out of scope |
| `RMS_NORM` | OK | OK | out of scope |
| `NORM` (layer norm) | **MISSING** | — | — |
| `GELU_ERF` (unary) | **MISSING** | — | — |
| `CONCAT` | **MISSING** | — | — |
| `FLASH_ATTN_EXT` | **MISSING** | — | — |
| custom `ggml_supertonic_*` ops | **MISSING** | — | — |

Confirmed against the `ggml_compute_backward` switch: handled ops include `ADD`,
`MUL`, `SCALE`, `CPY`, `CONT`, `RESHAPE`, `PERMUTE`, `TRANSPOSE`, `GET_ROWS`,
`DIAG_MASK_INF`, `RMS_NORM`, `MUL_MAT`, `SOFT_MAX`, `IM2COL`, and a subset of
`UNARY` (`ABS`, `SGN`, `NEG`, `STEP`, `RELU`, `SILU`, `EXP`, `EXPM1`,
`SOFTPLUS`). `GELU`/`GELU_ERF`, `NORM`, `CONCAT`, and `FLASH_ATTN_EXT` fall
through to `GGML_ABORT`.

## Remaining Phase-2 work items

To reach a fully GGML-native, on-device backward of the style tail:

1. **`NORM` backward** — add layer-norm backward to `ggml_compute_backward`
   (analytic form: like `rms_norm_back` but mean-centered) + CPU kernel.
2. **`GELU_ERF` backward** — add the erf-GELU derivative to the `UNARY` switch +
   CPU kernel. Only needed if the convnext stack is included; the style tail
   itself does not use GELU, so this is deferrable unless the upstream stack is
   also made differentiable.
3. **`FLASH_ATTN_EXT` backward** — no fused backward exists. Lower
   speech-prompted attention to the explicit `mul_mat` + `soft_max` +
   `mul_mat` form (all of which already have backward) for the differentiable
   enrollment graph, keeping the fused kernel for inference.
4. **`CONCAT` backward** — slice-and-route grad to each input. Only on the
   padding path (upstream stack); deferrable like GELU.
5. **Custom `ggml_supertonic_*` ops** — provide backward or lower them to
   primitive ops on the enrollment graph.
6. **Per-stage gradcheck** — wire each lowered stage into the Task 2 harness
   (finite differences vs reference) — the analytic backward from this PR is the
   reference oracle.

Items 1 and 3 are the minimum to make the **style tail** differentiable in GGML;
items 2, 4, 5 are only required if differentiability is extended upstream of the
style tail (not needed for cloning, which only optimizes `style_ttl`).

## Interim solution shipped in this PR

Because the gaps above (`NORM`, `FLASH_ATTN_EXT` in particular) block a
GGML-native backward today, this PR ships an **analytic C++ backward** of the
style tail, validated component-wise against finite differences via the Task 2
gradcheck harness:

- `dense_time_backward_input` — linear (ONNX layout)
- `layer_norm_channel_backward` — channel layer norm
- `speech_attention_backward` — softmax attention (`d_x`, `d_style`)
- `speech_tail_backward` — 2 speech-prompted layers + residual + final LN →
  `d(loss)/d(style)`

This is mathematically exact, runs on CPU (the enrollment target), and serves as
the **reference oracle** for the per-stage gradcheck once the GGML-native ops in
the work items above are implemented.
