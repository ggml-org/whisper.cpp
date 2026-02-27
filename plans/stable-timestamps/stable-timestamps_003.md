# Stable Timestamps - Integration Plan

Reference: [stable-ts](https://github.com/jianfch/stable-ts) | See `_001.md` for how stable-ts works, `_002.md` for whisper.cpp internals.

## Design Principles

- **Opt-in only** — new param `stable_timestamps`, off by default, zero impact on existing users
- **Isolated** — `src/whisper-stable.cpp` + `src/whisper-stable.h` (internal header, follows `whisper-arch.h` pattern)
- **Minimal internal changes** — Phase 1: one line. Phase 2: a few lines calling into our file. Phase 3: zero lines.
- **VAD required** — `stable_timestamps = true` requires a loaded VAD model. Fail with clear error if missing. No half-baked loudness fallback.
- **Reuse existing infrastructure** — VAD system, `logits_filter_callback`, DTW pipeline, `whisper_state` structs

## Resolved Technical Details

### VAD Timing Resolution
- 1 probability per `n_window` samples. Official Silero models: `n_window = 512` = 32ms at 16kHz.
- `n_window` is model-defined (loaded from GGML file), not hardcoded.
- Prob index `i` maps to sample `n_window * i` → time `(n_window * i) / 16000.0` seconds.

### State Access
- `whisper_state` is opaque in public API, defined only in `src/whisper.cpp`.
- Pattern: `src/whisper-stable.h` internal header (like `src/whisper-arch.h`). Our functions receive the data they need as parameters — passed from `whisper.cpp` which has full struct access.
- No need to expose `whisper_state` — just pass VAD probs, segment/token data via function args.

### DTW Head Tensor Layout
- Shape: `[n_tokens, n_audio_ctx, n_selected_heads]` (contiguous).
- Heads concatenated layer-by-layer in mask construction order.
- Raw index: `data[head * (n_tokens * n_audio) + token * n_audio + audio_pos]`.

### VAD vs Decode Timeline
- VAD runs in `whisper_full()` BEFORE `whisper_full_with_state()`.
- VAD probs are on the **original** audio timeline.
- Decode timestamps in `state->result_all` are on the **processed** (VAD-stripped) timeline.
- Segment API getters remap to original timeline via `vad_mapping_table`. Token timestamps do NOT get remapped.
- **Our snapping must work on the processed timeline** (matching token timestamps), then the existing segment getter remapping handles the rest. OR we snap after remapping using the public getters. Decision: snap using public getters (original timeline) — simpler, avoids timeline confusion.

---

## Phase 1 — Post-Hoc Silence Snapping (~80% of stable-ts quality)

### What It Does

After whisper.cpp produces all segments and word timestamps, iterate every word/segment boundary and snap it away from silence using VAD speech probabilities.

### New Files

- `src/whisper-stable.h` — internal header (follows `whisper-arch.h` pattern)
- `src/whisper-stable.cpp` — implementation

### New Param

Add to `whisper_full_params` in `whisper.h`:
- `bool stable_timestamps` (default `false`)

### Algorithm

1. Get VAD per-frame speech probabilities (already computed when `vad = true`)
2. Convert to silence map: frames where `prob < vad_threshold` are silent. Each frame = `n_window` samples = 32ms (model-defined).
3. Merge adjacent silent frames into silence regions with start/end times (in seconds)
4. Filter out silence regions shorter than `min_silence_dur` (e.g. 0.1s)
5. For each word in each segment:
   - **Start in silence:** If `word.t0` falls inside a silence region (silence_start <= t0 < silence_end <= t1), move `t0` to `silence_end`
   - **End in silence:** If `word.t1` falls inside a silence region (t0 <= silence_start < t1 <= silence_end), move `t1` to `silence_start`
   - **Silence contained in word:** If silence fully inside word, compute error ratio on each side. Snap the side with less error. First word in segment: prefer snapping start forward. Last word: prefer snapping end backward.
   - **Clamp:** Never let word duration go below `min_word_dur` (e.g. 0.05s)
6. Update segment `t0`/`t1` from first/last word boundaries

### Function Signature (internal)

```c
// in whisper-stable.h
void whisper_stable_snap_timestamps(
    int n_segments,                          // from whisper_full_n_segments()
    std::vector<whisper_segment> & segments,  // state->result_all (passed from whisper.cpp)
    const float * vad_probs,                 // VAD per-frame probabilities
    int n_vad_frames,                        // number of VAD frames
    int vad_n_window,                        // samples per VAD frame (512)
    float vad_threshold,                     // silence threshold (0.35)
    float min_silence_dur,                   // min silence region duration (0.1s)
    float min_word_dur                       // min word duration after snapping (0.05s)
);
```

whisper.cpp calls this with data it has access to. No need for whisper-stable.cpp to see whisper_state.

### Internal Changes (1 line + include)

In `whisper.cpp`, add `#include "whisper-stable.h"` and at end of `whisper_full_with_state()`:

```c
if (params.stable_timestamps) {
    whisper_stable_snap_timestamps(...);  // pass required data from state
}
```

### Automatic Side Effects When `stable_timestamps = true`

- Force `params.vad = true` (need VAD probabilities)
- Force `params.token_timestamps = true` (need word-level timestamps to snap)
- Set `params.max_initial_ts = 0.0f` (remove the 1.0s constraint like stable-ts)

### Error Handling

- If no VAD model loaded: log error, return without snapping, don't crash
- If no word timestamps available: log warning, snap segment-level only

---

## Phase 2 — DTW Improvements (~15% more quality)

### What It Does

Two improvements to the DTW word-timestamp extraction, behind the same `stable_timestamps` flag. Only active when DTW is enabled (`ctx_params.dtw_token_timestamps = true`).

### 2A. Gap Padding

**Problem:** Without padding, DTW cross-attention energy can "leak" early, causing words to start before actual speech.

**Solution:** Prepend `" ..."` tokens before the text tokens in the DTW token sequence.

**Change location:** `whisper_exp_compute_token_level_timestamps_dtw()` at line 8843-8860.

Currently builds: `[sot, lang, no_timestamps, text_tokens..., eot]`

With stable_timestamps: `[sot, lang, no_timestamps, GAP_TOKENS, text_tokens..., eot]`

Where `GAP_TOKENS` = tokenized `" ..."` (typically 2-3 tokens).

**Implementation:** Add a function in `whisper-stable.cpp` that returns the gap token IDs. Call it from the DTW function when `stable_timestamps = true`. Adjust `sot_sequence_length` to account for gap tokens. After DTW, discard the gap token timestamps.

**Internal changes:** ~5 lines in `whisper_exp_compute_token_level_timestamps_dtw()` to conditionally call our function.

### 2B. Dynamic Head Selection

**Problem:** Hardcoded alignment heads are suboptimal for some audio/models. Some heads may produce noisy attention.

**Solution:** Score all captured attention heads by monotonicity, select top-k.

**Change location:** `whisper_exp_compute_token_level_timestamps_dtw()` at line 8919 (currently takes mean across all heads).

**Tensor access:** Data is `[n_tokens, n_audio_ctx, n_selected_heads]`. Index: `data[head * (n_tokens * n_audio) + token * n_audio + audio_pos]`.

**Algorithm:**
1. For each head, compute attention peaks (argmax per token across audio frames)
2. Score = how monotonically peaks increase (correlation with `[0, 1, 2, ...]`)
3. Select top-k heads (default 6)
4. Take weighted mean of selected heads only (instead of mean of all)
5. Proceed with DTW on the averaged matrix

**Implementation:** New function in `whisper-stable.cpp`: `whisper_stable_select_heads()`. Called from DTW function instead of the plain mean. Receives raw float pointer + dimensions.

**Internal changes:** ~3 lines: replace the mean-across-heads call with a conditional call to our head selection function.

**Prerequisite:** To score all heads, we need all heads captured, not just preset ones. When `stable_timestamps = true`, override `dtw_aheads_preset` to `WHISPER_AHEADS_N_TOP_MOST` with a high N (or all layers). This exposes more heads for scoring. Fallback: if user explicitly set custom heads, respect that.

---

## Phase 3 — Constrained Decoding (~5% more quality, zero internal changes)

### What It Does

During decoding, suppress timestamp tokens that correspond to silent audio positions. The decoder cannot predict a timestamp landing in silence.

### Implementation

Uses the existing `params.logits_filter_callback` — no internal changes needed.

**Setup:** Done internally in `whisper.cpp` when `stable_timestamps = true`. Before decoding begins, set `logits_filter_callback` to a filter function defined in `whisper-stable.cpp`. The silence mask from Phase 1 VAD probs is reused and passed via `logits_filter_callback_user_data`.

**Filter function:**
```
For each timestamp token position t:
    if silence_mask[t] == true:
        logits[token_beg + t] = -INFINITY
```

**Note:** This overwrites `logits_filter_callback`. If a user needs both custom filtering and stable timestamps, they handle chaining themselves. Not worth the complexity for an edge case with ~zero users.

---

## Summary

| Phase | Quality | Internal Changes | New Code Location |
|-------|---------|-----------------|-------------------|
| 1. Silence snapping | ~80% | 1 line (+ 1 param) | `src/whisper-stable.cpp` |
| 2. DTW improvements | ~15% | ~8 lines | `src/whisper-stable.cpp` |
| 3. Constrained decoding | ~5% | 0 lines | `src/whisper-stable.cpp` |

### Files Touched in whisper.cpp Internals

- `include/whisper.h` — add `stable_timestamps` bool to `whisper_full_params`
- `src/whisper.cpp` — `#include "whisper-stable.h"` + Phase 1: 1 call at end of `whisper_full_with_state()`. Phase 2: ~8 lines in DTW function.
- That's it.

### New Files

- `src/whisper-stable.h` — internal header (follows `whisper-arch.h` pattern)
- `src/whisper-stable.cpp` — all stable-ts logic
- `CMakeLists.txt` — add new source file to build

### CLI Integration

Add `--stable-timestamps` flag to `examples/cli/cli.cpp` that sets `params.stable_timestamps = true`. Requires `--vad-model` to also be set.

### Dependencies

- VAD model file (~2MB, same one whisper.cpp already supports)
- No new libraries, no new build dependencies

### Rollout

Phase 1 first as a PR. Small, clean, easy to review. Phase 2 and 3 as follow-up PRs. Each phase is independently useful and independently reviewable.

### Effort Estimation

~300 lines total. Plan fully resolved, no open questions.

| Phase | Lines | Time |
|-------|-------|------|
| Phase 1 (silence snapping) | ~150 | ~3 min |
| Phase 2 (DTW improvements) | ~100 | ~5 min |
| Phase 3 (constrained decoding) | ~30 | ~2 min |
| Plumbing (whisper.h, CMake, CLI) | ~20 | ~1 min |
| **Total** | **~300** | **~11 min** |
