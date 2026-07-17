# QVAC-21921 — [AudioGen] ACE-Step (acestep.cpp) CPU bring-up — Recon notes

> Phase-0 reconnaissance. No engine code written yet. This documents *what* to port,
> *where* it goes, and *how much divergence* there is, so implementation can start from facts.
> Branch: `feat/QVAC-21921-acestep-cpu` (in `qvac-ext-lib-whisper.cpp`).

## 0. What the model is

ACE-Step 1.5 = **music generation** (text → music), not speech. Ticket is tagged `[AudioGen]`.
The repo `ServeurpersoCom/acestep.cpp` already ships a **working C++ / ggml engine** (CPU/CUDA/Metal/Vulkan/SYCL/termux)
with **pre-quantized GGUF weights**, so this is a **reconciliation / bring-up**, not a from-scratch port.

Repo language split (confirms it's already C++):
`C++ 774 KB` (engine) · `C 301 KB` (ggml) · `Python 52 KB` (**only** `convert.py` + parity/debug scripts) · Svelte/TS (demo web UI).

## 1. Scope recap (this ticket, CPU-only)

1. Land acestep's **2 custom ggml ops** into our `ggml-speech` fork (`qvac-ext-ggml`, branch `speech`) — **CPU kernels only**.
2. Rebase acestep's C++ engine onto **our** `ggml-speech` base (the two ggml forks track different upstream points → divergence work).
3. Port the pipeline into `tts-cpp/src` layout: text-enc (Qwen3-Embedding 0.6B) → LM (ace-lm, Qwen3-arch) → DiT → Oobleck VAE. **Only the VAE uses the 2 custom ops**; the rest is stock ggml.
4. Reuse their reference-parity scripts as our fixture/cosine-sim tests vs the PyTorch reference.
5. GGUF: consume the existing quantized weights; verify CPU load + end-to-end generation. Convert script comes from acestep.cpp.
6. Wire through vcpkg to the monorepo NPM package.

**Out of scope now:** GPU backends, q4 quantization (DiT is quant-sensitive; LM-4B breaks at Q4_K_M).
**Dominant risk:** VAE memory — graphs up to ~491,520 timesteps. Need sane `--vae-chunk`/overlap defaults + short default durations from the start.

## 2. acestep.cpp engine layout (`src/`, header-heavy)

Mostly header-implemented, a few `.cpp`. Grouped by pipeline stage:

| Stage | Files |
|---|---|
| Text encoder | `qwen3-enc.h`, `cond-enc.h` |
| LM (ace-lm) | `qwen3-lm.h`, `pipeline-lm.cpp/.h`, `sampling.h`, `metadata-fsm.h` (FSM-constrained decode) |
| DiT | `dit.h`, `dit-graph.h`, `dit-sampler.h`, `solvers/` (euler, dpm, sde, stork/stork4) |
| VAE (Oobleck) | `vae.h`, `vae-enc.h`, `fsq-tok.h`, `fsq-detok.h`, `dwt-haar.h` ← **the only stage needing the 2 custom ops** |
| Tokenizer | `bpe.h` |
| Weights / IO | `gguf-weights.h`, `safetensors.h`, `weight-ctx.h`, `model-store.cpp/.h`, `model-registry.h`, `audio-io.h`, `wav.h`, `audio-resample.h` |
| Orchestration | `pipeline-synth*.{h,cpp}`, `pipeline-understand.*`, `request.*`, `task-types.h`, `static-graph.h`, `graph-arena.h`, `backend.h`, `timer.h`, `philox.h` |

## 3. The 2 custom ggml ops — exact nature (important nuance)

Their ggml submodule is pinned at `ServeurpersoCom/ggml@9e2947f17583acc2f657a77c29b6593ca0fbc6c4`.

### 3a. `COL2IM_1D` — a REAL new op enum
Scatter-add of GEMM columns back to a 1D signal (GEMM-based `conv_transpose_1d`). Files to port (CPU only):
- `include/ggml.h` → `GGML_OP_COL2IM_1D` enum (line ~538) + `ggml_col2im_1d(...)` API (~2011-2014)
- `src/ggml.c` → op name table (`"COL2IM_1D"` ~1034, `"col2im_1d(x)"` ~1145) + `ggml_col2im_1d()` builder (~4546-4575)
- `src/ggml-cpu/ops.cpp` → `ggml_compute_forward_col2im_1d[_impl]` (~6786-6854; F32/F16/BF16)
- `src/ggml-cpu/ops.h` → decl (~71)
- `src/ggml-cpu/ggml-cpu.c` → dispatch case (~1917)

→ **Maps 1:1 to how our Supertonic ops are already carried** (new enum + name + builder + CPU kernel + dispatch case). Straightforward.

### 3b. `SNAKE` — NOT an op enum, it's an AUTOFUSE
Snake activation `y = x + sin^2(a*x) * inv_b`. Implemented by **detecting a stock-op pattern** and fusing it, not by adding an enum:
- `src/ggml-cpu/ggml-cpu.c` (~3016-3056): detects `{ MUL, SIN, SQR, MUL, ADD }` via `ggml_can_fuse(...)` and runs the fused kernel.
- `src/ggml-cpu/ops.cpp` (~6733-6782): `ggml_compute_forward_snake_fused[_impl]` (F32/F16/BF16).
- `src/ggml-cpu/ops.h` (~72): decl.
- (CUDA `snake.cu/.cuh` + Metal fuse exist too — **GPU phase, out of scope**.)

→ Our `ggml-speech` **already has `ggml_can_fuse`** (in `ggml-impl.h` / `ggml.c`), so the autofuse approach is viable on our base.
**Open decision:** land Snake as an autofuse (copy their approach, no new enum) — recommended, matches upstream direction — vs. as an explicit `GGML_OP_SUPERTONIC_*`-style enum to match our current Supertonic convention. Autofuse is cleaner and is what their engine expects.

## 4. Fork divergence (rebase sizing)

Common ancestor of our `speech` and their pinned ggml: `19eac6f0` (**2026-05-02**, upstream ggml).
- Our `origin/speech`: **54 commits** ahead of base (our speech ops + backend fixes; tip `d7e27ac7`, 2026-07-15).
- Their ggml `9e2947f`: **283 commits** ahead of base (they tracked upstream ggml much further; tip 2026-07-14, msg "metal: absorb snake fusion...").

Implication: **do NOT wholesale-adopt their ggml.** Their engine may call ggml APIs that landed in those ~283 upstream commits but aren't in our 54-commit base. The reconciliation work = (a) cherry-pick/reimplement just the 2 ops onto our `speech`, and (b) adapt their engine's ggml API usage to our older base where signatures differ. Expect the API-surface gap to be the main friction, exactly as the ticket warns.

## 5. Target patterns in our repos (the mold to copy)

### `qvac-ext-ggml` (branch `speech`) — how custom CPU ops are carried
Supertonic family = explicit enums, each with: enum in `include/ggml.h`, name in `src/ggml.c`, CPU kernel in `src/ggml-cpu/ops.cpp` (e.g. `ggml_compute_forward_supertonic_depthwise_1d` ~7871), decl in `ops.h`, dispatch + can-run cases in `src/ggml-cpu/ggml-cpu.c` (~1945-1963, ~2386-2390).
`GGML_OP_SUPERTONIC_{DEPTHWISE_1D, LAYER_NORM_CHANNEL, PW2_RESIDUAL, BIAS_GELU, EDGE_PAD_1D}`.

### `tts-cpp/src` — how an engine is structured
Two conventions in `src/`: older models are flat prefixed files (`chatterbox_*.cpp`, `supertonic_*.cpp`); the newest (`lavasr`) is a **subfolder** `src/lavasr/` with a clean split per component:
`*_core.cpp/.h` (logic) · `*_ggml.cpp/.h` (ggml graph) · `*_gguf.cpp/.h` (weight load) · `*_api.cpp` · `dsp/` subfolder.

→ **Recommended:** ACE-Step goes in a new **`src/acestep/`** subfolder following the lavasr split, one component per stage (text-enc / lm / dit / vae), keeping the 4-model pipeline tidy. Plus CLI wiring (`*_cli.cpp` + register in `cli_main.cpp`) and vcpkg later.

## 6. Parity / test assets available to reuse (`acestep.cpp/tests/`)
- `debug-lm-logits.py` (+ `.sh`), `debug-detok-cossim.py`, `debug-dit-cossim.py` (+ `.sh`), `debug-tok-cossim.py` — cosine-sim vs PyTorch reference, per stage.
- C++ tests: `test-lm-prompt.cpp`, `test-model-store.cpp`, `test-philox.cpp` + `request0.json` fixture.
- Reference logs per backend×quant (`CPU-*.log`, `Metal-*.log`, etc.) — golden outputs to compare against.

## 7. Proposed order of work (for when we start coding)
1. Land `COL2IM_1D` (enum, easy) + Snake autofuse onto `qvac-ext-ggml@speech`, CPU-only, with a tiny unit test each.
2. Bring up the VAE (Oobleck) stage first in isolation (it's the only op consumer + the memory risk) with `debug-detok-cossim.py` parity + `--vae-chunk`/overlap defaults.
3. Bring up LM → DiT → text-enc against their parity scripts, one stage at a time.
4. End-to-end CPU generation from the existing quantized GGUF; validate on mobile early (VAE memory).
5. CLI + vcpkg + NPM wiring.

## Local recon artifacts (not committed)
- Reference clone: `/Users/freddy/Work/Tether/repos/acestep.cpp` (with pinned `ggml` submodule checked out).
- Added remote `acestep-ggml` in `qvac-ext-ggml` (for merge-base sizing) — can be removed with `git remote remove acestep-ggml`.
