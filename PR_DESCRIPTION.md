# DTW Timestamps: Universal Onset Shift & Plosive Optimization

## Problem Description

The experimental DTW (Dynamic Time Warping) implementation previously exhibited latency issues and unnatural token durations:
1.  **Late Onset Latency**: Words starting with Plosives (e.g., "country", "people") and Vowels ("And", "ask") aligned to the energy peak/burst rather than the articulatory onset, causing a perceived lag of 100-200ms.
2.  **Short Token Durations**: Semivowels (e.g., "you") were treated as consonants, resulting in truncated durations (80ms).
3.  **Timestamp Regressions**: The previous Minimum Duration logic pushed start times *forward*, causing cascading delays in fast speech.
4.  **Conflicting Segment Logic**: Legacy code for segment wrapping overwrote accurate DTW timestamps.

## Solution

This PR implements a comprehensive fix:

### 1. Universal Onset Shift (Phonetic-Aware)
A targeted backward shift strategy is applied based on the phonetic properties of the first character:
*   **Aggressive Shift (150ms)**: Applied to:
    *   **Vowels** (`a,e,i,o,u`): Captures soft glottal onsets.
    *   **Semivowels** (`y,w`): Captures glides (fixing strictly short "you").
    *   **Plosives** (`b,c,d,g,k,p,q,t`): Covers the "closure silence" interval before the stop burst.
*   **Standard Shift (80ms)**: Applied to remaining consonants (Fricatives, Nasals, Liquids) for snappy alignment without overshooting.

### 2. Backward-Extending Minimum Duration
The minimum duration enforcement (50ms) now extends the token **backward** (earlier start time) into the preceding gap, rather than pushing the end time forward. This prevents the "domino effect" of delaying subsequent tokens.

### 3. Cross-Segment Lookahead
Restored logic to correctly calculate the duration of the last token in a segment by peeking at the start time of the *next* segment's first token. This eliminates artificially short 10ms tokens at segment boundaries.

### 4. Codebase Simplification
Removed the redundant "Block A" segment wrapping logic (`whisper_full_with_state`) which conflicted with the internal DTW propagation.

## Verification Results

The fix was verified using `jfk.wav`. The comparisons below show significant accuracy improvements, particularly for the problematic "And" and "country" tokens.

**Word-level Timestamp Comparison:**

| Word | Metric | Original (Bugged) | **Optimized (This PR)** | Reference (CTC) |
| :--- | :--- | :--- | :--- | :--- |
| **And** | Start Time | `0.50s` (Late) | **0.31s** (Perfect) | `0.30s` |
| **so** | Latency | `0.45s` (Early) | **0.64s** (Snappy) | `0.66s` |
| **country** | Start Time | `6.20s` (Lag) | **6.05s** (Fixed) | `5.98s` |
| **you** | Duration | `0.08s` (Truncated) | **0.15s** (Natural) | `0.15s` |
| **can** | Alignment | `6.56s` | **6.41s** | `6.52s` |

*Note: Timestamps derived from `whisper-cli` output with `-dtw base.en -nfa -ml 1`.*

## Usage

No API changes required. The fixes are active automatically when DTW is used.

```bash
./main -m models/ggml-base.en.bin -f samples/jfk.wav --dtw base.en -nfa -ml 1
```
