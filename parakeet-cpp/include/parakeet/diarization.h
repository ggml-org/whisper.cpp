#pragma once

// Sortformer diarization: offline results and sliding-history streaming sessions.
//
// Offline: segments plus per-frame speaker_probs. Streaming: push PCM; each step runs Sortformer
// over the last `history_ms`, emits overlapping segments, advances the chunk cursor.
// Optional StreamEvent VadStateChanged from speaker_probs (see <parakeet/streaming.h>).

#include "export.h"
#include "streaming.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace parakeet {

struct DiarizationOptions {
    float threshold      = 0.5f;
    int   min_segment_ms = 0;
};

struct DiarizationSegment {
    int    speaker_id = 0;
    double start_s    = 0.0;
    double end_s      = 0.0;
};

struct DiarizationResult {
    std::vector<DiarizationSegment> segments;
    std::vector<float> speaker_probs;
    int    n_frames        = 0;
    int    num_spks        = 0;
    double frame_stride_s  = 0.08;

    int    audio_samples   = 0;
    int    sample_rate     = 16000;
    double preprocess_ms   = 0.0;
    double encoder_ms      = 0.0;
    double decode_ms       = 0.0;
    double total_ms        = 0.0;
};

// One speaker span emitted by SortformerStreamSession. `is_final=true`
// flags the LAST callback the session will fire; it is either:
//   - a real segment from the trailing partial chunk (when audio ends
//     mid-chunk), or
//   - a synthetic terminator with `speaker_id = -1` and `start_s ==
//     end_s` (when audio ended exactly on a chunk boundary, so all
//     real segments were already delivered as `is_final=false`).
// Consumers should treat `speaker_id < 0` as "session done, no new
// segment" and skip any text/append logic for it.
struct StreamingDiarizationSegment {
    int    speaker_id  = 0;
    double start_s     = 0.0;
    double end_s       = 0.0;
    int    chunk_index = 0;
    bool   is_final    = false;
};

struct SortformerStreamingOptions {
    int   sample_rate     = 16000;

    int   chunk_ms        = 2000;
    int   history_ms      = 30000;

    float threshold       = 0.5f;
    int   min_segment_ms  = 200;

    bool  emit_partials   = true;

    // Optional StreamEvent delivery (VadStateChanged from speaker_probs); nullptr disables.
    StreamEventCallback on_event = nullptr;

    // === AOSC (Audio-Online Speaker Cache, Sortformer v2.1) ===
    // Cache-aware streaming forward (port of NeMo's `forward_streaming_step` +
    // `streaming_update` + `_compress_spkcache`). On v2.1 models (auto-detected
    // from encoder shape) and spkcache_enable=true, the engine concatenates the
    // speaker cache + FIFO + current chunk's pre-encode embeddings, runs the
    // conformer layers over the concat, then the diariser head, before updating
    // the runtime cache. This preserves speaker identity across silences far
    // longer than `history_ms`. v1 models always take the legacy path.
    //
    // `mean_sil_emb` is RUNTIME state (zeros at session start, EMA of detected
    // silence frames), NOT a learned tensor -- no converter changes required.
    // Defaults below are NeMo's inference defaults (see
    // examples/speaker_tasks/diarization/neural_diarizer/e2e_diarize_speech.py).
    bool  spkcache_enable        = true;
    int   spkcache_len           = 188;    // total cache rows (encoder frames)
    int   fifo_len               = 188;    // FIFO warmup buffer (encoder frames)
    int   chunk_left_context_ms  = 80;     // ~1 encoder frame at v2.1 (80ms)
    int   chunk_right_context_ms = 560;    // ~7 encoder frames at v2.1 (560ms)
    int   spkcache_update_period = 144;    // pop_out_len on FIFO overflow
};

using SortformerSegmentCallback =
    std::function<void(const StreamingDiarizationSegment &)>;

// Live Sortformer session: ring buffer, periodic diarize over trailing history_ms,
// segments overlapping the new audio window. Speaker ids come from each chunk pass and can
// drift slightly until the history buffer is filled; larger history_ms stabilizes sooner.
class PARAKEET_API SortformerStreamSession {
public:
    struct Impl;
    explicit SortformerStreamSession(std::unique_ptr<Impl> impl);
    ~SortformerStreamSession();

    SortformerStreamSession(const SortformerStreamSession &)            = delete;
    SortformerStreamSession & operator=(const SortformerStreamSession &) = delete;
    SortformerStreamSession(SortformerStreamSession &&) noexcept;
    SortformerStreamSession & operator=(SortformerStreamSession &&) noexcept;

    void feed_pcm_f32(const float * samples, int n_samples);
    void feed_pcm_i16(const int16_t * samples, int n_samples);
    void finalize();
    void cancel();

    const SortformerStreamingOptions & options() const;

    // True when the session is running v2.1 NeMo-style speaker-cache
    // streaming (AOSC). False on v1 sortformer GGUFs, or on v2.x with
    // `SortformerStreamingOptions::spkcache_enable=false`. Mirrors the
    // internal `cache_active` flag; useful for CLI banners / logs that
    // want to differentiate the two streaming modes for the user.
    bool aosc_active() const;

private:
    std::unique_ptr<Impl> pimpl_;
};

}
