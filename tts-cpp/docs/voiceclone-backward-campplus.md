# Voice-clone backward — CAMPPlus speaker encoder (op × backend gap matrix)

Scope for ticket *"GGML backward pass: CAMPPlus speaker encoder"*.
This doc scopes the work to make the CAMPPlus speaker encoder **differentiable in
GGML** on the CPU path used for enrollment, and records which backward ops are
still missing in the vendored `ggml`.

It is committed alongside the interim deliverable of this PR: an analytic,
gradchecked C++ backward of the whole CAMPPlus chain
(`src/campplus_backward.{h,cpp}`). See
[Interim vs Phase-2](#interim-solution-shipped-in-this-pr) for how the two
relate.

## Why the gap exists

In the enrollment loop CAMPPlus provides the **speaker-similarity loss** between
the target-WAV embedding (constant, forward-only) and the generated-audio
embedding. Only the generated-audio path needs gradients, so the gradient we
need is `d(loss)/d(fbank)` — the input gradient with the model weights frozen.
The fbank is differentiated further back to the waveform by a separate stage;
this module stops at the CAMPPlus input.

A fully GGML-native backward (the Phase-2 goal, needed by the on-device
enrollment loop) requires every op on the forward graph to have a backward in
`ggml_compute_backward` (`ggml/src/ggml.c`) **and** a CPU kernel for the ops the
backward expands into. Several are missing today.

## Forward ops on the CAMPPlus path

Source: `src/campplus_forward.inc` (the GGML graph) and `src/campplus.cpp` (the
scalar CPU reference `campplus_embed_cpu`).

| Forward op | Where (forward) |
| --- | --- |
| `ggml_conv_2d` / `ggml_im2col` + `ggml_mul_mat` | FCM Conv2d head + residual blocks |
| `conv1d_f32` (`ggml_im2col` + `ggml_mul_mat`) | TDNN, linear1, linear_local, cam linear1/2, transits, dense |
| `ggml_mul` / `ggml_add` (broadcast) | pre-fused BN (scale/shift), bias adds, residuals |
| `ggml_relu` | every nonlinear1/2, transit, out_nonlinear, FCM |
| `ggml_sigmoid` | CAMLayer context gate |
| `ggml_mean` | CAMLayer global context, stats-pool mean + variance |
| `ggml_sum_rows` | CAMLayer seg-pool reduction |
| `ggml_pad` / `ggml_repeat` | CAMLayer seg-pool reshape + broadcast |
| `ggml_sqrt` | stats-pool std |
| `ggml_concat` | dense concat (CAMDenseTDNN), stats-pool mean‖std |
| `ggml_cont`/`reshape`/`view` | layout shuffles, FCM (32,10,T)→(320,T) flatten |

## Gap matrix

Legend: **OK** = implemented; **MISSING** = aborts / not implemented; **n/a** =
not on the enrollment path.

"Graph backward" = a case in `ggml_compute_backward` (`ggml/src/ggml.c`). It is
backend-agnostic: if it aborts, no backend can differentiate the op. "CPU bwd
kernel" = the kernels the backward expands into exist for the CPU backend
(`ggml-cpu`), the only backend enrollment needs in Phase 2. GPU columns are out
of scope for Phase 2 (enrollment runs on CPU) and tracked only for visibility.

| Op | Graph backward (ggml.c) | CPU bwd kernel | CUDA / Metal / Vulkan / OpenCL |
| --- | --- | --- | --- |
| `MUL_MAT` | OK | OK (`out_prod`/`mul_mat`) | out of scope |
| `ADD` / `MUL` | OK | OK | out of scope |
| `CONT`/`RESHAPE`/`VIEW`/`PERMUTE` | OK | OK | out of scope |
| `IM2COL` | OK (`im2col_back`) | OK | out of scope |
| `RELU` (unary) | OK | OK | out of scope |
| `SIGMOID` (unary) | **MISSING** | — | — |
| `MEAN` | **MISSING** | — | — |
| `SUM_ROWS` | **MISSING** | — | — |
| `SQRT` (unary) | **MISSING** | — | — |
| `PAD` | **MISSING** | — | — |
| `REPEAT` | **MISSING** | — | — |
| `CONCAT` | **MISSING** | — | — |

Confirmed against the `ggml_compute_backward` switch: handled ops include `ADD`,
`MUL`, `SCALE`, `CPY`, `CONT`, `RESHAPE`, `PERMUTE`, `TRANSPOSE`, `GET_ROWS`,
`DIAG_MASK_INF`, `RMS_NORM`, `MUL_MAT`, `SOFT_MAX`, `IM2COL`, and a subset of
`UNARY` (`ABS`, `SGN`, `NEG`, `STEP`, `RELU`, `SILU`, `EXP`, `EXPM1`,
`SOFTPLUS`). `SIGMOID`, `SQRT`, `MEAN`, `SUM_ROWS`, `PAD`, `REPEAT`, and `CONCAT`
fall through to `GGML_ABORT`.

## Remaining Phase-2 work items

To reach a fully GGML-native, on-device backward of CAMPPlus:

1. **`SIGMOID` backward** — add `s*(1-s)` to the `UNARY` switch + CPU kernel
   (needed by the CAMLayer gate).
2. **`SQRT` backward** — add `1/(2*sqrt(x))` to the `UNARY` switch + CPU kernel
   (stats-pool std).
3. **`MEAN` / `SUM_ROWS` backward** — broadcast the upstream grad back over the
   reduced axis (`1/N` for mean) + CPU kernels.
4. **`PAD` / `REPEAT` backward** — slice off the padding / sum over the repeated
   axis (`ggml_repeat_back` already exists; wire it into `ggml_compute_backward`).
5. **`CONCAT` backward** — slice-and-route the grad to each input (dense concat
   and stats-pool concat).
6. **Per-stage gradcheck** — wire each lowered stage into the Task 2 harness;
   the analytic backward from this PR is the reference oracle.

Alternatively, the seg-pool / stats-pool subgraphs can be lowered to
`mul_mat`-based reductions (which already have backward), avoiding new kernels for
`MEAN`/`SUM_ROWS`/`REPEAT`.

## Interim solution shipped in this PR

Because the gaps above block a GGML-native backward today, this PR ships an
**analytic C++ backward** of the whole CAMPPlus chain, validated component-wise
against finite differences via the Task 2 gradcheck harness
(`src/voiceclone_gradcheck.{h,cpp}`):

- `conv1d_backward_input` / `conv2d_backward_input` — transpose-conv input grad
  (stride / pad / dilation aware)
- `bn_backward_input` — pre-fused affine BN (per-channel scale)
- `relu_backward` / `sigmoid_backward` — pointwise nonlinearities
- `mean_T_backward` / `seg_pool_backward` — CAMLayer context reductions
- `stats_pool_backward_input` — mean + unbiased std pooling
- `fcm_resblock_backward` — Conv2d residual block (with optional shortcut)
- `cam_layer_backward` — CAMDenseTDNN layer (gate + dense-concat split)
- `CampplusBackward::backward` — full chain → `d(loss)/d(fbank)`

It mirrors the layout and conventions of `campplus_embed_cpu` exactly. Two tests
guard it (both in the always-on `unit` ctest tier, model-free):

- `test-campplus-backward` — gradchecks every primitive and the full chain
  against central finite differences.
- `test-campplus-backward-parity` — asserts the analytic double forward matches
  the production scalar forward (`campplus_embed_cpu`) on synthetic weights
  (multi-layer CAM blocks, 2/3/2, so the dense-concat accumulation is exercised),
  anchoring the gradcheck's relevance to the real model.

The scalar CPU forward is the path every `campplus_embed` caller uses today
(production `main.cpp`, `test-campplus`, `test-voice-embedding` all pass
`backend==nullptr`), and `test-campplus` / `test-voice-embedding` validate it
against the Python reference embedding. So the trust chain is complete:
Python → `campplus_embed_cpu` → analytic forward → gradchecked backward. The
`campplus_embed_ggml` graph path is not wired to any caller yet; when it is, it
gets its own fixture parity against the CPU/Python path.

This is mathematically exact, runs on CPU (the enrollment target), and serves as
the **reference oracle** for the per-stage gradcheck once the GGML-native ops in
the work items above are implemented.

> Note: `campplus_embed_cpu`'s `fcm_forward` hardcodes the input feature
> dimension to 80 (the production fbank width), so the production scalar path is
> only self-consistent at `feat_dim=80`; the parity test uses that. The analytic
> backward derives every dimension from `feat_dim`, so it is geometry-agnostic.
