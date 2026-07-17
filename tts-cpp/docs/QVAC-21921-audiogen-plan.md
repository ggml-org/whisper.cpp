# QVAC-21921 — ACE-Step music generation: native addon plan

Goal: `@qvac/audiogen-ggml`, a native addon (no binaries) that takes a text
prompt and returns stereo 48 kHz audio, following the exact pattern of
`@qvac/tts-ggml` (native C++ compiled per-platform, linking `tts-cpp` via
vcpkg).

## Architecture (target)

```
@qvac/audiogen-ggml (JS)
  index.js  ── native binding (BARE_MODULE) ──►  AcestepModel (model-interface)
                                                     │
                                                     ▼
                                          tts_cpp::acestep::Engine   (tts-cpp)
                                                     │
              text-enc ─► LM ─► DiT ─► VAE  (ggml graphs, ggml-speech fork, CPU)
```

- Addon layer mirrors `tts-ggml/addon/src`: `js-interface/binding.cpp`
  (`BARE_MODULE`), `addon/AddonJs.hpp` (`createInstance`/`activate`/`runJob`),
  `model-interface/acestep/{AcestepConfig,AcestepModel}.{hpp,cpp}` implementing
  `qvac_lib_inference_addon_cpp::model::IModel` (+ `IModelCancel`,
  `IModelAsyncLoad`).
- Build: `cmake-bare` + `cmake-vcpkg`, `find_package(tts-cpp CONFIG REQUIRED)`,
  `add_bare_module(audiogen-ggml ...)`, `target_link_libraries(... tts-cpp::tts-cpp)`.
- `vcpkg.json` depends on `tts-cpp` (+ `qvac-lib-inference-addon-cpp`,
  `qvac-lint-cpp`), same registry baseline as `tts-ggml`.

The engine contract already exists: `tts-cpp/include/tts-cpp/acestep/engine.h`
(`tts_cpp::acestep::Engine::create()` / `generate()`), designed to match
`vae.h`'s style.

## What is DONE

- Custom ggml ops in `qvac-ext-ggml` (ggml-speech), CPU kernels + unit tests:
  - `ggml_col2im_1d` (transposed-conv scatter-add for the VAE decoder)
  - `ggml_snake` (snake activation)
- VAE stage in `tts-cpp` (`src/acestep/vae_*`, `tts_cpp::acestep::Vae`):
  decode + encode validated on CPU (roundtrip correlation ~0.96 on real music).
- **DiT stage in `tts-cpp`** (`src/acestep/dit_gguf.{h,cpp}`, `dit_ggml.{h,cpp}`):
  - Loads the real turbo GGUF (`acestep-v15-turbo-Q8_0`, 1.6 GB of weights,
    24 layers, H=2048, GQA 16/8, head_dim 128, in=192/out=64, patch=2,
    sliding_window=128, rope theta 1e6) onto the CPU backend.
  - Single `dit_model_forward` runs the full 24-layer graph (AdaLN + GQA
    self-attn with sliding-window/full alternation + cross-attn + SwiGLU) and
    returns finite velocities. Verified by `dit-smoke` (PASS).
  - `dit_sample` = full flow-matching loop (Euler, turbo 8 steps, schedule
    `t_i = shift·t/(1+(shift-1)·t)` shift=3, SA sliding-window + CA padding
    masks, no CFG since turbo runs guidance=1.0). Verified by `dit-smoke
    --sample` (schedule + finite latent, PASS).
  - **Root fix:** DiT tensor names exceed 64 chars, so the ggml-speech fork's
    `GGML_MAX_NAME` was bumped 64→128 (matches upstream acestep.cpp). Part of
    the `ggml_tensor` ABI, so lib + all consumers rebuilt together.
  - Harness: `src/acestep/dit_smoke.cpp` + `dit-smoke` CMake target.
- **Shared Qwen3 block** (`src/acestep/qwen3_block.h`): the loaders + graph
  builders (RMSNorm, GQA self-attn with per-head QK-norm + NEOX RoPE, SwiGLU,
  mask-driven causality/windowing) reused by the text/lyric/timbre encoders.
- **Text encoder in `tts-cpp`** (`textenc_ggml.{h,cpp}`): loads the real
  `Qwen3-Embedding-0.6B-Q8_0` (742.7 MB, 28 layers, H=1024, GQA 16/8, D=128) and
  runs a full causal forward → finite hidden states. Verified by `textenc-smoke`
  (PASS). Harness: `textenc_smoke.cpp`.
- **Condition encoder in `tts-cpp`** (`cond_ggml.{h,cpp}`): lyric encoder (8L,
  H=2048, bidirectional sliding-window) + timbre encoder (4L, +CLS on XL) + text
  projector (1024→2048) + `null_condition_emb`, loaded from the DiT GGUF
  (`encoder.*`, 616 MB). Packs `enc_hidden [2048, S_total]` = [lyric | timbre[0] |
  text_proj]. Verified by `cond-smoke` (PASS). Harness: `cond_smoke.cpp`.
- **LM core in `tts-cpp`** (`lm_ggml.{h,cpp}`): ace-lm Qwen3 0.6B causal LM with a
  persistent f16 KV cache (set_rows write + windowed read), tied LM head, config
  derived from tensor shapes. Loads the real `acestep-5Hz-lm-0.6B-Q8_0` (671.9 MB,
  28 layers, H=1024, V=217204, GQA 16/8) and runs prefill + multi-step greedy
  decode with the KV cache advancing → finite logits (argmax in the audio-code
  range). Verified by `lm-smoke` (PASS). Harness: `lm_smoke.cpp`.
- **BPE tokenizer in `tts-cpp`** (`bpe_tokenizer.{h,cpp}`): Qwen3/GPT-2
  byte-level BPE loaded from the LM GGUF KV (`tokenizer.ggml.tokens` /
  `.merges`, 151643 vocab / 151387 merges), GPT-2 pre-tokenizer + merges +
  byte decoder. Encode↔decode roundtrip verified on ASCII/unicode/CJK/digits/
  punctuation/YAML metadata. Verified by `bpe-smoke` (PASS). Ported from
  `acestep.cpp/src/bpe.h` + `sampling.h::bpe_decode`.
- **LM pipeline in `tts-cpp`** (`lm_pipeline.{h,cpp}`): `sample_top_k_p`
  (temperature→top_k→top_p→multinomial), CoT/`build_lm_prompt_with_cot` chat
  template, and the Phase-2 audio-code decode loop (restricted to EOS + audio
  codes, single KV set, no CFG). Runs the real turbo text2music prompt end to
  end → **~5 codes/s** FSQ semantic tokens in-range, LM emits `<|im_end|>` on
  its own. Verified by `lmgen-smoke` (PASS). Phase 1 (CoT/lyric gen + metadata
  FSM) still deferred — first path uses a fully-specified prompt.
- **FSQ detokenizer in `tts-cpp`** (`detok_ggml.{h,cpp}`): LM codes → DiT
  context latents. FSQ decode (6 dims, levels 8/8/8/5/5/5) → project_out
  (6→2048) → embed + special_tokens broadcast (5 frames) → 2L Qwen3 encoder →
  norm → proj_out (2048→64) ⇒ `[64, T_25Hz]` (T_25Hz = T_5Hz·5). Loaded from
  the DiT GGUF (`tokenizer.*` / `detokenizer.*`). Verified by `detok-smoke`
  (PASS). Ported from `acestep.cpp/src/fsq-detok.h`, reuses `qwen3_block.h`.
- **All eight neural stages now run natively on real weights**: VAE, DiT (+ Euler
  sampler), text-encoder, cond-encoder, LM core, BPE tokenizer, LM pipeline
  (Phase 2 codes), FSQ detokenizer.
- **END-TO-END ENGINE WORKS** (`src/acestep/engine.cpp`, `tts_cpp::acestep::Engine`):
  `Engine::generate()` runs the full native chain — caption+lyrics → LM Phase-2
  codes → FSQ detok context `[128,T]` (latent[64] + mask[64]=1) → text-encoder
  (`# Instruction`/`# Caption`/`# Metas` + lyric embed lookup) → cond-encoder →
  DiT flow-matching (Euler, turbo 8 steps, Philox `torch.randn`-parity noise) →
  VAE decode → stereo 48 kHz PCM. Driven by `music-cli`: a 6 s prompt renders a
  **5.44 s stereo WAV in ~28 s on CPU** (27 LM codes), audio has real musical
  content (RMS ~2638/peak 29490, ~307 zero-crossings/s, stereo). No acestep.cpp
  binaries — the same C++ the addon links.
- `engine.h` facade contract + `@qvac/audiogen-ggml` package scaffold + stable
  JS API.

## What remains (the port — this is the multi-day work)

Order chosen to unblock end-to-end as early as possible (DiT+VAE first render
audio from precomputed codes; LM/text-enc add prompt→codes):

1. **DiT stage** — DONE (load + forward + sampler run natively; see "What is
   DONE"). Remaining: numerical parity vs upstream on a fixed seed (dump
   `enc_hidden`/`context`/`noise` from `ace-synth` with a fixed seed and compare
   our `dit_sample` latent), plus wiring the DiT→VAE bridge (latent layout
   `[64, T]` is already VAE-`decode` compatible).
2. **text-encoder stage** — DONE (Qwen3-Embedding load + forward PASS).
3. **cond-encoder stage** — DONE (lyric/timbre/text-proj + null_emb, PASS).
4. **LM stage** — DONE for the text2music turbo path: LM core (KV cache) + BPE
   tokenizer + Phase-2 pipeline (CoT prompt, top-k/p sampling, code decode) +
   FSQ detokenizer (codes → `[64, T_25Hz]` context). `lmgen-smoke` +
   `detok-smoke` PASS. Deferred: Phase 1 (CoT/lyric generation) + metadata FSM
   + CFG (needs multi-set KV in the core).
5. **Engine glue** — DONE. `Engine::create()` loads all stages + both tokenizers
   on a shared CPU backend; `Engine::generate()` wires textenc → cond →
   LM(codes) → detok(context [128,T]) → DiT `dit_sample` → VAE → stereo 48 kHz,
   with progress + cooperative cancel. Verified end-to-end by `music-cli`.
   Remaining polish: Phase-1 auto-metadata (so a bare caption works without
   defaults), CFG, numerical parity vs acestep.cpp, and model store/eviction.
6. **Addon build** — flip `AcestepModel` from stub to real `Engine` calls;
   bump the `tts-cpp` vcpkg REF; green build on macOS/Linux/Windows/Android/iOS.

### ggml op audit

acestep.cpp's ggml fork is ~283 commits ahead of ggml-speech. Audit result
per stage (checked against `qvac-ext-ggml/include/ggml.h`):

- **DiT — no new ops needed.** The graph (dit-graph.h) uses only ops already
  present in ggml-speech: `ggml_rms_norm`, `ggml_mul_mat`, `ggml_rope_ext`,
  `ggml_flash_attn_ext` (+ `ggml_soft_max_ext` F32 fallback for CPU),
  `ggml_swiglu` / `ggml_swiglu_split`, `ggml_timestep_embedding`, `ggml_silu`,
  `ggml_conv_transpose_1d`, and stock add/mul/scale/view/reshape/permute/cont/
  cast/sub. So the DiT port is pure C++ plumbing (GGUF load + graph + sampler),
  **zero ggml fork changes**.
- **LM / text-enc** — audit pending; Qwen3 attention/RoPE/rms_norm are all
  standard, so no new ops are expected, but confirm before porting.

If any gap appears, land a CPU kernel with the same procedure as snake/col2im
(`ggml.h` enum + builder in `ggml.c`, CPU kernel in `ops.cpp`, dispatch in
`ggml-cpu.c`, standalone test).

### DiT architecture (from acestep.cpp/src/dit*.h)

24-layer transformer, AdaLN modulation, GQA self-attention (sliding-window on
even layers, full on odd) + cross-attention to the text-encoder states, SwiGLU
MLP. Flow-matching sampler, turbo = 8 Euler steps. Config from GGUF metadata
(`acestep-dit.*`). Weight loading fuses QKV / gate-up when types match and
pre-permutes proj_in/proj_out convs at load time. Port target files in
`tts-cpp/src/acestep/`: `dit_gguf.{h,cpp}` (IO), `dit_ggml.{h,cpp}` (graph +
run), plus the sampler loop and conditioning (cond-enc) wired by the engine.

## Interim (today)

`@qvac/audiogen-ggml` scaffold is in place with the native structure and a
stable JS API. Generation is gated on the DiT/LM/text-enc port above. A working
end-to-end demo exists via the upstream `acestep.cpp` build (proof the model +
our ported ops work), but that path is **not** part of the shippable addon —
the addon must compile the C++ per-platform, not call a binary.

## Model set (smallest working, ~3.8 GB)

| Stage | GGUF | Size |
|------|------|------|
| text-enc | Qwen3-Embedding-0.6B-Q8_0 | 748 MB |
| LM | acestep-5Hz-lm-0.6B-Q8_0 | 710 MB |
| DiT | acestep-v15-turbo-Q8_0 | 2.4 GB |
| VAE | vae-BF16 | 322 MB |
