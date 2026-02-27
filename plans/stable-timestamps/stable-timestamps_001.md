# Stable Timestamps - How stable-ts Works

Reference repo: https://github.com/jianfch/stable-ts

## Overview

stable-ts improves Whisper's word-level timestamps with near-zero performance cost. The core idea: Whisper gives rough timestamps, then stable-ts clips them to where sound actually exists. No model weights are changed.

## The 5 Mechanisms

### 1. Post-Hoc Silence Snapping (main workhorse, always on)

**Files:** `stabilization/__init__.py`, `stabilization/nonvad.py`, `stabilization/silero_vad.py`

After Whisper produces timestamps, every word boundary is checked against a silence map and snapped to speech edges.

**Silence map construction (non-VAD mode):**
1. `abs(waveform)` -> normalize by 99.9th percentile
2. Interpolate down to one value per audio token position (320 samples per token at 16kHz)
3. Average-pool with kernel size 5 (reflection padding) to smooth
4. Quantize: `mask = (mask * 20).round()` -> anything rounding to 0 = silent
5. Convert boolean mask to start/end silence timing arrays

**Snapping logic:**
- If word.start falls inside silence -> move start to silence_end
- If word.end falls inside silence -> move end to silence_start
- If silence is contained within a word -> snap the boundary with less "error" (ratio of overshoot vs silence duration, threshold 10%)
- First word in segment: prefer keeping end (snap start forward)
- Last word in segment: prefer keeping start (snap end backward)
- Minimum word duration is enforced during snapping

### 2. Better Cross-Attention / DTW Alignment

**File:** `timing.py`

Three improvements to how word timestamps are extracted from cross-attention:

**a) Gap padding:**
Prepend `" ..."` tokens before each segment's tokens in DTW. This absorbs early cross-attention energy that would otherwise cause timestamps to start too early.

**b) Dynamic head selection (`dynamic_heads`):**
Instead of hardcoded `model.alignment_heads`, score ALL attention heads by how monotonically their peaks track the DTW path. Select best k (default 6) per token. Can run multiple iterations where each pass refines head selection using previous DTW result.

**c) `max_initial_timestamp=None`:**
Vanilla Whisper forces first timestamp <= 1s. stable-ts removes this constraint so speech starting later in a 30s chunk isn't forced early.

**d) New alignment algorithm (`aligner='new'`, from arxiv:2509.09987):**
Score all (layer, head) pairs by column-norm and row-norm of attention matrix. Select top-k (default 20) globally, normalize each by column norm, average, then DTW.

### 3. Constrained Decoding (opt-in, off by default)

**File:** `decode.py`

Subclasses Whisper's `DecodingTask`. During token sampling, timestamp tokens corresponding to silent audio regions are set to `-inf`. The decoder literally cannot predict a timestamp in silence.

```
ts_logits[:, ts_token_mask] = -inf
```

Controlled by `suppress_ts_tokens=True` (defaults to `False`).

Also caches audio features across temperature fallbacks (vanilla Whisper re-encodes mel each time).

### 4. Binary-Search Refinement (optional, expensive)

**File:** `non_whisper/refinement.py`

Called explicitly via `model.refine()`. For each word boundary:
1. Progressively mute audio inward from the boundary
2. Run inference, monitor token probability
3. If probability holds -> mute more (boundary can be tighter)
4. If probability drops -> restore (speech is there)
5. Binary search converges to latest-possible-start / earliest-possible-end

Precision ~0.1s default. Runs inference dozens of times per word - slow but optional.

### 5. Hallucination Filtering

**File:** `whisper_word_level/original_whisper.py`

- Segments with >50% zero-duration words -> discarded
- Segments below avg probability threshold -> discarded
- Entirely silent 30s chunks -> skipped without running decoder
- Long silence gaps within chunks -> audio truncated to prevent hallucinated text
- Punctuation-only segments -> deleted

## Cost Summary

| Mechanism | Speed Cost | Always On? | Benefit |
|-----------|-----------|------------|---------|
| Silence snapping | ~0 | Yes | 60% of improvement |
| Better DTW (gap padding, dynamic heads) | ~0 | Yes | 20% of improvement |
| Hallucination filtering | ~0 | Yes | Cleaner output |
| Constrained decoding | ~0 | No (opt-in) | Prevents silent timestamps |
| Binary-search refinement | Very high | No (explicit call) | Tightest possible boundaries |

## What to Port to whisper.cpp

**Priority 1 (easy, high impact):** Post-hoc silence snapping. ~100 lines of C. No model changes needed. Just audio analysis + timestamp adjustment on existing output.

**Priority 2 (medium effort):** Gap padding in DTW step. Requires touching `whisper_exp_compute_token_level_timestamps()`.

**Priority 3 (medium effort):** Dynamic attention head selection. whisper.cpp already extracts cross-attention for DTW. Need to expose all heads and score them.

**Priority 4 (low priority):** Constrained decoding. Invasive to sampling loop.

**Priority 5 (skip):** Binary-search refinement. Too expensive, wrong fit for whisper.cpp's use case.
