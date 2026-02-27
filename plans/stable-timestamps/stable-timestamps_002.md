# Stable Timestamps - How whisper.cpp Works (Relevant Internals)

## Codebase Structure

- `include/whisper.h` (741 lines) -- Public C API
- `src/whisper.cpp` (9016 lines) -- Entire implementation in one file
- `src/whisper-arch.h` -- Tensor name maps (encoder/decoder/VAD)
- `ggml/` -- Tensor library backend
- `examples/cli/cli.cpp` -- Main CLI

## Key Data Structures (all in `src/whisper.cpp`)

### Token Data (`whisper_token_data`, whisper.h:131)
```c
id, tid (timestamp token), p (probability), plog, pt (timestamp prob),
ptsum (sum of timestamp probs), t0/t1, t_dtw, vlen (voice length)
```

### Segment (`whisper_segment`, line 460)
```c
t0, t1, text, no_speech_prob, tokens (vector<whisper_token_data>)
```

### State (`whisper_state`, line 834)
Holds: `mel`, `kv_self/kv_cross`, `decoders[8]`, `result_all` (segments), `energy` (PCM signal energy), `aheads_masks`, `aheads_cross_QKs`, `vad_context/segments/mapping`

## Decoding Pipeline

Entry: **`whisper_full_with_state()`** at line 6805

1. **PCM -> Mel** (line 6818): `whisper_pcm_to_mel_with_state()` -- FFT + mel filterbank, 80 bands, hop=160 (10ms/frame)
2. **Signal energy** (line 6847): `get_signal_energy(samples, n_samples, 32)` -- smoothed abs amplitude for token timestamps
3. **Main loop** (line 7012): `while(true)` over 30s chunks, advancing by `seek`
4. **Encoder** (line 7033): `whisper_encode_internal()` -- conv + encoder + cross-attn KV cache
5. **Prompt setup** (line 7098-7157): `[<prev>] + past + [<sot>] + [<lang>] + [<transcribe>]`
6. **Token-by-token** (line 7197): `for (i = 0; i < n_max; ++i)` where `n_max = n_text_ctx/2 - 4`

### Logit Processing -- `whisper_process_logits()` at line 6155

This is WHERE ALL LOGIT FILTERING HAPPENS:

- **Line 6232**: `logits_filter_callback` -- user-supplied callback (external injection point)
- **Line 6268-6308**: Timestamp pairing constraints (must come in pairs, must increase)
- **Line 6291-6298**: `max_initial_ts` constraint -- limits first timestamp to <= 1.0s
  - **stable-ts removes this** by setting it to `None`
  - whisper.cpp param: `params.max_initial_ts` (default 1.0f, line 5950)
- **Line 6300-6308**: Increasing timestamp enforcement via `decoder.seek_delta/2`
- **Line 6314-6365**: Force timestamp when `sum(ts_probs) > max(text_probs)`

**INJECTION POINT for constrained decoding:** Between lines 6300-6308 (after increasing-ts check), add `logits[token_beg + t] = -INFINITY` for silent positions. Or use the existing `logits_filter_callback` externally.

### Sampling -- `whisper_sample_token()` at line 6438
Greedy: argmax. Also computes `tid` (best timestamp), `pt` (timestamp prob), `ptsum` (sum timestamp probs).

## Word-Level Timestamps

### Method 1: Non-DTW (simpler, existing)

**`whisper_exp_compute_token_level_timestamps()`** at line 8433

1. Uses `state.energy` (smoothed PCM amplitude)
2. Confident timestamps from `token.tid` when `pt > thold_pt && ptsum > thold_ptsum`
3. Fills gaps by proportional splitting based on `vlen`
4. **Energy-based refinement** (lines 8563-8631): Expands/contracts token boundaries using signal energy. This is a PRIMITIVE form of silence snapping already present -- but crude.

### Method 2: DTW (experimental, more accurate)

**`whisper_exp_compute_token_level_timestamps_dtw()`** at line 8815

1. Build token sequence: `[sot] + [lang] + [no_timestamps] + all_text_tokens + [eot]`
2. Full decoder pass with `save_alignment_heads_QKs=true`
3. Copy cross-attention QKs to CPU: shape `[n_tokens, n_audio_tokens, n_heads]`
4. Normalize (line 8907)
5. Median filter width 7 over audio dimension (line 8914)
6. **Mean across heads** (line 8919) -- all selected heads weighted equally
7. Scale by -1 (line 8920)
8. Standard DTW + backtrace via `dtw_and_backtrace()` (line 8690)
9. Assign timestamps from DTW path (lines 8940-8963)

**IMPORTANT:** DTW does NOT work with `flash_attn=true` (line 3708-3710) because flash attention doesn't expose intermediate attention weights.

Called at lines 7725-7728 after all segments created for a 30s window.

### Alignment Heads -- Hardcoded (lines 384-409)

```c
static const whisper_ahead g_aheads_large_v3[] = {
    {7,0}, {10,17}, {12,18}, {13,12}, {16,1}, {17,14}, {19,11}, {21,4}, {24,1}, {25,6}
};
static const whisper_ahead g_aheads_large_v3_turbo[] = {
    {2,4}, {2,11}, {3,3}, {3,6}, {3,11}, {3,14}
};
```

Selected via `get_alignment_heads_by_layer()` (line 8666). Modes: preset-specific, N-top-most layers, or custom user-provided heads.

Masks built in `aheads_masks_init()` (line 1160), used during decoder graph construction at lines 2720-2734 in the cross-attention block.

### WHERE TO ADD IMPROVEMENTS:

**Gap padding:** In DTW function at line 8843-8860 when building token sequence. Insert `" ..."` tokens after `no_timestamps` but before text tokens. Adjust `sot_sequence_length`.

**Dynamic head selection:** At line 8919 (currently takes mean). Instead: score each head for monotonicity, select top-k, then average only those. Would need to expose all heads first (currently only preset heads captured).

## VAD Support (Already Exists!)

whisper.cpp has full Silero-style neural VAD:

- **`whisper_vad()`** at line 6621 -- called from `whisper_full()` when `params.vad == true`
- Strips silence, concatenates speech segments with overlap
- Builds `vad_mapping_table` to remap timestamps back to original audio
- **Per-frame speech probabilities** available via `whisper_vad_probs()` API
- Params: `threshold`, `min_speech_duration_ms`, `min_silence_duration_ms`, etc.

This is relevant because: we could use the existing VAD probabilities as input for the silence mask instead of building our own loudness-based detector (or offer both options like stable-ts).

## Segment Creation & Output

### How Segments Are Created (lines 7616-7718)
1. Scan tokens for timestamp tokens (`id > whisper_token_beg()`)
2. Text between timestamps -> segment with `t0`, `t1`, text, tokens
3. Pushed to `result_all`
4. If `token_timestamps == true`: per-segment token timestamps computed
5. If DTW enabled: DTW timestamps computed per-window after all segments

### WHERE TO HOOK POST-HOC SNAPPING:

**Option A -- Internal:** After DTW (line 7735) or after non-DTW token timestamps (lines 7663/7708), iterate all segments and snap word boundaries to speech edges using silence mask.

**Option B -- End of pipeline:** Before `whisper_full_with_state()` returns (line 7753), as a final pass over all `result_all`.

**Option C -- New public API:** `whisper_snap_timestamps(ctx, state)` that callers invoke after `whisper_full()`. Cleanest, non-invasive.

## Existing Energy-Based "Snapping" (Primitive)

Lines 8563-8631 in `whisper_exp_compute_token_level_timestamps()`:
- Computes energy sum in token's time range
- Expands/contracts boundaries based on energy threshold
- Already exists but is crude compared to stable-ts

## Key Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `WHISPER_SAMPLE_RATE` | 16000 | Hz |
| `WHISPER_HOP_LENGTH` | 160 | samples per mel frame = 10ms |
| `WHISPER_CHUNK_SIZE` | 30 | seconds per chunk |
| `WHISPER_N_FFT` | 400 | FFT window size |
| Audio token resolution | 320 samples = 20ms | Each audio ctx position |
| Timestamp token resolution | 20ms | Each increment of timestamp token |
| `n_audio_ctx` | 1500 | Audio tokens per 30s chunk |
| `n_text_ctx` | 448 | Max text tokens |

## Public API Surface (relevant)

```c
// After transcription:
whisper_full_n_segments(ctx)
whisper_full_get_segment_t0/t1(ctx, i)        // centiseconds (1 = 10ms)
whisper_full_get_segment_text(ctx, i)
whisper_full_n_tokens(ctx, i)
whisper_full_get_token_data(ctx, i, j)        // -> whisper_token_data
whisper_full_get_segment_no_speech_prob(ctx, i)

// Params:
params.token_timestamps      // enable non-DTW word timestamps
params.max_initial_ts        // default 1.0s (stable-ts sets to 0)
params.logits_filter_callback // can inject custom logit filters externally
ctx_params.dtw_token_timestamps // enable DTW mode
ctx_params.dtw_aheads_preset    // which alignment heads
params.vad                   // enable built-in VAD
```
