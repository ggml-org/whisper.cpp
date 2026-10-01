#include "whisper-diarization-arch.h"
#include "whisper.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <map>
#include <cstring>
#include <string>
#include <vector>

static void whisper_diar_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
#ifndef WHISPER_DEBUG
    if (level == GGML_LOG_LEVEL_DEBUG) {
        return;
    }
#endif
    fputs(text, stderr);
    fflush(stderr);
}

GGML_ATTRIBUTE_FORMAT(2, 3)
static void whisper_diar_log_internal(ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    char buffer[1024];
    const int len = vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    if (len < 0) {
        return;
    }
    if (static_cast<size_t>(len) < sizeof(buffer)) {
        whisper_diar_log_callback_default(level, buffer, nullptr);
    } else {
        std::vector<char> buffer2(static_cast<size_t>(len) + 1);
        va_start(args, format);
        vsnprintf(buffer2.data(), buffer2.size(), format, args);
        va_end(args);
        whisper_diar_log_callback_default(level, buffer2.data(), nullptr);
    }
}

#define WHISPER_LOG_ERROR(...) whisper_diar_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define WHISPER_LOG_WARN(...)  whisper_diar_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define WHISPER_LOG_INFO(...)  whisper_diar_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)

struct whisper_diar_scoring {
    int   sil_frames_per_spk   = 1;
    float pred_score_threshold = 0.25f;
    float scores_boost_latest  = 0.05f;
    float strong_boost_rate    = 0.75f;
    float weak_boost_rate      = 1.5f;
    float min_pos_scores_rate  = 0.5f;
};

struct whisper_diar_cache_params {
    int spkcache_len       = 264;
    int fifo_len           = 80;
    int chunk_len          = 13;
    int spkcache_batch_len = 40;
};

struct whisper_diar_cache {
    whisper_diar_cache_params cache_params;
    whisper_diar_scoring scoring;

    int n_speakers = 0;
    int n_embd = 0;

    // Speaker cache which stores 
    std::vector<float> spkcache;
    int n_spk_frames = 0;

    std::vector<float> spkcache_preds;

    // Holds a rolling buffer of hidden states (the input to the graph computation,
    // the log melspectrograms projected to the models hidden vector space
    std::vector<float> fifo;
    int n_fifo_frames = 0;

    std::vector<float> mean_sil_emb;
};

struct whisper_diar_hparams {
    int32_t n_audio_state      = 0;
    int32_t n_audio_head       = 0;
    int32_t n_head_state       = 0;
    int32_t n_ff               = 0;
    int32_t n_mels             = 0;
    int32_t n_fft              = 0;
    int32_t n_audio_layer      = 0;
    int32_t n_speakers         = 0;
    int32_t n_audio_ctx        = 0;
    int32_t subsampling_factor = 0;
    float   preemph            = 0;
    float   log_guard          = 0;
    float   rope_base          = 0;
};

struct whisper_diar_layer_encoder {
    ggml_tensor * norm_attn_w = nullptr;
    ggml_tensor * norm_attn_b = nullptr;
    ggml_tensor * attn_qkv_w  = nullptr;
    ggml_tensor * attn_out_w  = nullptr;
    ggml_tensor * attn_out_b  = nullptr;
    ggml_tensor * norm_ff_w   = nullptr;
    ggml_tensor * norm_ff_b   = nullptr;
    ggml_tensor * ff1_w       = nullptr;
    ggml_tensor * ff1_b       = nullptr;
    ggml_tensor * ff2_w       = nullptr;
    ggml_tensor * ff2_b       = nullptr;
};

struct whisper_diar_model {
    whisper_diar_hparams hparams;
    whisper_diar_scoring scoring;

    ggml_tensor * mel_filters    = nullptr;
    ggml_tensor * silence_emb    = nullptr;
    ggml_tensor * enc_pre_w      = nullptr;
    ggml_tensor * enc_norm_w     = nullptr;
    ggml_tensor * enc_norm_b     = nullptr;
    ggml_tensor * enc_out_norm_w = nullptr;
    ggml_tensor * enc_out_norm_b = nullptr;
    ggml_tensor * enc_proj_w     = nullptr;
    ggml_tensor * enc_proj_b     = nullptr;
    ggml_tensor * upsample_w     = nullptr;
    ggml_tensor * upsample_b     = nullptr;
    ggml_tensor * head_hidden_w  = nullptr;
    ggml_tensor * head_hidden_b  = nullptr;
    ggml_tensor * head_spks_w    = nullptr;
    ggml_tensor * head_spks_b    = nullptr;

    std::vector<whisper_diar_layer_encoder> layers;
    std::map<std::string, ggml_tensor *> tensors;
    int  n_loaded = 0;

    std::vector<ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> buffers;

    std::vector<float> filters;
    std::vector<float> silence;
    float window[400];
};

struct whisper_diar_state {
    std::vector<ggml_backend_t> backends;
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> meta;
    whisper_diar_cache cache;
    std::vector<float> probs;

    // Buffers reused across chunks to avoid per-chunk heap allocation.
    std::vector<float>   buf_padded;
    std::vector<float>   buf_cache;
    std::vector<int32_t> buf_pos;
    std::vector<float>   buf_probs;
    std::vector<float>   buf_hidden;
    std::vector<float>   buf_probs_80ms;
};

struct whisper_diar_context {
    whisper_diar_context_params params;
    std::string path_model;

    whisper_diar_model model;
    whisper_diar_state state;
};

static const int WHISPER_DIAR_MAX_NODES = 8192;
static const double WHISPER_DIAR_PI = 3.14159265358979323846;

static constexpr float WHISPER_DIAR_NEG_INF = -std::numeric_limits<float>::infinity();
static constexpr float WHISPER_DIAR_POS_INF = std::numeric_limits<float>::infinity();
static constexpr int64_t WHISPER_DIAR_MAX_INDEX = 99999;

static std::vector<int> whisper_diar_topk_column(const std::vector<float> & scores,
        int n, int n_spk, int spk, int k) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    if (k >= n) {
        return idx;
    }

    std::nth_element(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
        const float sa = scores[static_cast<size_t>(a) * n_spk + spk];
        const float sb = scores[static_cast<size_t>(b) * n_spk + spk];
        if (sa != sb) {
            return sa > sb;
        }
        return a < b;
    });

    idx.resize(k);
    return idx;
}

static void whisper_diar_cache_init(whisper_diar_cache & state, const whisper_diar_cache_params & cache_params,
        const whisper_diar_scoring & scoring, int n_speakers, int n_embd,
        const std::vector<float> & learned_silence) {
    state              = {};
    state.cache_params = cache_params;
    state.scoring      = scoring;
    state.n_speakers   = n_speakers;
    state.n_embd       = n_embd;
    state.mean_sil_emb = learned_silence;
}

static void whisper_diar_cache_compact(whisper_diar_cache & state, const std::vector<float> &  cache_preds);

static void whisper_diar_cache_update(whisper_diar_cache & state,
        const float * hidden, int frames, const float * probs, int rc) {
    const int chunk_valid = frames - rc;
    if (chunk_valid <= 0) {
        return;
    }

    // probs will contain the models predictions, which is the processed output
    // of [speaker_cache | fifo | hidden] and this will produce the output in
    // the following shape:
    // [ processed_speaker_cache | processed_fifo | new_chunk]
    //
    // Where new_chunk is the processed output of hidden.

    const int n_spk_frames  = state.n_spk_frames;
    const int n_fifo_frames = state.n_fifo_frames;

    // Append the new hidden state into the fifo. So the contents of the fifo
    // is the projected input not the processed input.
    // [hidden0, hidden1, ... ]
    state.fifo.insert(state.fifo.end(),
                      hidden,
                      hidden + static_cast<size_t>(chunk_valid) * state.n_embd);

    state.n_fifo_frames = n_fifo_frames + chunk_valid;

    if (state.n_fifo_frames > state.cache_params.fifo_len) {
        // The following vector will contain the processed fifo and the new_chunk.
        // state.fifo now holds [old fifo hiddens | hidden] (we just appended the
        // new hidden state above). This array mirrors that same layout but in
        // prediction space, so that pop_embs[f] and pop_preds[f] below always
        // refer to the same physical frame.
        const float * processed_fifo = probs + static_cast<size_t>(n_spk_frames) * state.n_speakers;
        const float * new_chunk = probs + static_cast<size_t>(n_spk_frames + n_fifo_frames) * state.n_speakers;
        std::vector<float> processed_fifo_and_new_chunk;
        processed_fifo_and_new_chunk.reserve(static_cast<size_t>(n_fifo_frames + chunk_valid) * state.n_speakers);
        processed_fifo_and_new_chunk.insert(processed_fifo_and_new_chunk.end(),
                           processed_fifo,
                           processed_fifo + static_cast<size_t>(n_fifo_frames) * state.n_speakers);
        processed_fifo_and_new_chunk.insert(processed_fifo_and_new_chunk.end(),
                           new_chunk,
                           new_chunk + static_cast<size_t>(chunk_valid) * state.n_speakers);

        int n_pop = state.cache_params.spkcache_batch_len;
        n_pop = std::max(n_pop, chunk_valid - state.cache_params.fifo_len + n_fifo_frames);
        n_pop = std::min(n_pop, state.n_fifo_frames);

        const float * pop_embs  = state.fifo.data();

        // Append the new fifo frames to the speaker cache.
        state.spkcache.insert(state.spkcache.end(),
                pop_embs,
                pop_embs + static_cast<size_t>(n_pop) * state.n_embd);

        const float * pop_preds = processed_fifo_and_new_chunk.data();
        if (!state.spkcache_preds.empty()) {
            // Append the processed fifo and new chunks (prediction space) to
            // the states speaker cache for predictions.
            state.spkcache_preds.insert(state.spkcache_preds.end(),
                                        pop_preds,
                                        pop_preds + static_cast<size_t>(n_pop) * state.n_speakers);
        }

        state.n_spk_frames += n_pop;

        // if we exceeded the length of the speaker cache and the speaker cache preditions is empty.
        if (state.n_spk_frames > state.cache_params.spkcache_len && state.spkcache_preds.empty()) {
            state.spkcache_preds.reserve(static_cast<size_t>(state.n_spk_frames) * state.n_speakers);
            // insert probs:
            // [ processed_speaker_cache | processed_fifo | new_chunk] into the
            // speaker cache for predictions.
            state.spkcache_preds.insert(state.spkcache_preds.end(),
                    probs,
                    probs + static_cast<size_t>(n_spk_frames) * state.n_speakers);

            // Append the processed fifo and new chunk.
            state.spkcache_preds.insert(state.spkcache_preds.end(),
                    pop_preds,
                    pop_preds + static_cast<size_t>(n_pop) * state.n_speakers);
        }

        // Remove the frames that have been popped.
        state.fifo.erase(state.fifo.begin(), state.fifo.begin() + static_cast<size_t>(n_pop) * state.n_embd);
        state.n_fifo_frames -= n_pop;

        if (state.n_spk_frames > state.cache_params.spkcache_len) {
            whisper_diar_cache_compact(state, state.spkcache_preds);
        }
    }
}

static void whisper_diar_cache_compact(whisper_diar_cache & state,
                                        const std::vector<float> & cache_preds) {
    const int n        = state.n_spk_frames;
    const int cap      = state.cache_params.spkcache_len;
    const int per_spk  = cap / state.n_speakers - state.scoring.sil_frames_per_spk;
    const int min_pos  = static_cast<int>(std::floor(per_spk * state.scoring.min_pos_scores_rate));
    const float log_half = std::log(0.5f);

    // Populate log-odds scores for each frame/speaker. For each 80ms frame a
    // score will be inserted representing the log-odds that only that speaker
    // is speaking.
    // score[0] = frame0 speaker0: log-odds that only speaker 0 is speaking
    // score[1] = frame0 speaker1: log-odds that only speaker 1 is speaking
    // score[2] = frame0 speaker2: log-odds that only speaker 2 is speaking
    // score[3] = frame0 speaker3: log-odds that only speaker 3 is speaking
    // score[4] = frame0 speaker4: log-odds that only speaker 4 is speaking
    // score[5] = frame0 speaker5: log-odds that only speaker 5 is speaking
    // score[6] = frame0 speaker6: log-odds that only speaker 6 is speaking
    // score[7] = frame0 speaker7: log-odds that only speaker 7 is speaking
    //
    // score[8] = frame1 speaker8: log-odds that only speaker 8 is speaking
    // score[9] = frame1 speaker9: log-odds that only speaker 9 is speaking
    // ...
    std::vector<float> scores(static_cast<size_t>(n) * state.n_speakers);
    for (int f = 0; f < n; ++f) {
        float sum_log1p = 0.f;
        for (int s = 0; s < state.n_speakers; s++) {
            const float p = cache_preds[static_cast<size_t>(f) * state.n_speakers + s];
            sum_log1p += std::log(std::max(1.f - p, state.scoring.pred_score_threshold));
        }

        for (int s = 0; s < state.n_speakers; s++) {
            const float p     = cache_preds[static_cast<size_t>(f) * state.n_speakers + s];
            const float logp  = std::log(std::max(p, state.scoring.pred_score_threshold));
            const float log1p = std::log(std::max(1.f - p, state.scoring.pred_score_threshold));
            scores[static_cast<size_t>(f) * state.n_speakers + s] = logp - log1p + sum_log1p - log_half;
        }
    }

    // The pos_count vector will store a count for each speaker for which its
    // scores value is greater than 0 meaning that this was a highly probable
    // solo speech with no cross-talk and no back ground ambiguiity.
    std::vector<int> pos_count(state.n_speakers, 0);
    for (int f = 0; f < n; ++f) {
        for (int s = 0; s < state.n_speakers; s++) {
            const size_t i = static_cast<size_t>(f) * state.n_speakers + s;

            // if the current speaker did not speak, then set to -inf to avoid
            // negative values in later operations.
            const bool is_speech = cache_preds[i] > 0.5f;
            if (!is_speech) {
                scores[i] = WHISPER_DIAR_NEG_INF;
            }

            if (scores[i] > 0.f) {
                pos_count[s]++;
            }
        }
    }

    for (int s = 0; s < state.n_speakers; s++) {
        // If the count for a speaker is less that min_pos then skip it.
        // Just one or two 80ms frames do not contain enough accoustic information
        // to define human voice so if we have less that the defined min we
        // skip this speaker. For example, the current min_pos is 16 which is
        // about 1.28 seconds (16*80ms ≈ 1.28s).
        if (pos_count[s] < min_pos) {
            continue;
        }

        for (int f = 0; f < n; ++f) {
            const size_t i = static_cast<size_t>(f) * state.n_speakers + s;
            const bool is_speech = cache_preds[i] > 0.5f;
            // If the probability of cache_pred[i] is greater than 0.5 we consider it
            // that someone is talking. But we also want to make sure that the
            // score for this index (speaker) is greater than 0, because if it
            // is not then there is some kind of cross-talk or background noice
            // and it is not a pure solo speaker.
            if (is_speech && !(scores[i] > 0.f)) {
                scores[i] = WHISPER_DIAR_NEG_INF;
            }
        }
    }

    // Increase/boost more recent speaker frames to rotate in the current
    // representations of a speakers voice to account for changes in the acoustic
    // profile (like a speaker turning their head away from the microphone, or
    // leaning back in their chair).
    if (state.scoring.scores_boost_latest > 0.f) {
        for (int f = cap; f < n; ++f) {
            for (int s = 0; s < state.n_speakers; s++) {
                scores[static_cast<size_t>(f) * state.n_speakers + s] += state.scoring.scores_boost_latest;
            }
        }
    }

    // Boost top k of each speaker's score so that all speakers are condidered
    // even if some speakers have acoustic characteristics that gave then a higher
    // score (like load, close to the microphone) compared to a speaker with a
    // lower score (more softly spoken, head away from the mic). This avoid a
    // strong speaker from drowning out other speakers completely.
    const int strong_k = static_cast<int>(std::floor(per_spk * state.scoring.strong_boost_rate));
    for (int s = 0; s < state.n_speakers; ++s) {
        for (int f : whisper_diar_topk_column(scores, n, state.n_speakers, s, strong_k)) {
            scores[static_cast<size_t>(f) * state.n_speakers + s] -= 2.f * log_half;
        }
    }

    // The following will use the boosted scores from above and gives a boost
    // to runner ups that were not included in the above strong_k boost.
    const int weak_k = static_cast<int>(std::floor(per_spk * state.scoring.weak_boost_rate));
    for (int s = 0; s < state.n_speakers; ++s) {
        for (int f : whisper_diar_topk_column(scores, n, state.n_speakers, s, weak_k)) {
            scores[static_cast<size_t>(f) * state.n_speakers + s] -= log_half;
        }
    }

    // Softformer requires that there is a silence anchor for every speaker, 
    // even inacactive ones (for self-attention to have a negative refernece point
    // and not hallucinate speech during pauses). By setting the silence frames
    // to +INF means that those silence frames will be at the front of the flat
    // vector.
    const int n_pad = n + state.scoring.sil_frames_per_spk;
    std::vector<int64_t> flat(static_cast<size_t>(state.n_speakers) * n_pad);
    std::iota(flat.begin(), flat.end(), 0);
    auto flat_score = [&](int64_t i) -> float {
        const int f = static_cast<int>(i % n_pad);
        if (f >= n) {
            // Set silence/padding frames to positive infinity.
            return WHISPER_DIAR_POS_INF;
        }
        const int s = static_cast<int>(i / n_pad);
        return scores[static_cast<size_t>(f) * state.n_speakers + s];
    };
    std::partial_sort(flat.begin(), flat.begin() + cap, flat.end(), [&](int64_t a, int64_t b) {
        const float sa = flat_score(a);
        const float sb = flat_score(b);
        if (sa != sb) {
            return sa > sb;
        }
        return a < b;
    });

    // Copy the flat vector up to the cap silence frames and the frames with
    // the highest scores (top cap).
    std::vector<int64_t> picked(flat.begin(), flat.begin() + cap);
    for (auto & i : picked) {
        if (flat_score(i) == WHISPER_DIAR_NEG_INF) {
            i = WHISPER_DIAR_MAX_INDEX * static_cast<int64_t>(n_pad) + WHISPER_DIAR_MAX_INDEX;
        }
    }
    // Sort by speaker and sequence in time.
    std::sort(picked.begin(), picked.end());

    std::vector<float> new_cache;
    new_cache.reserve(static_cast<size_t>(cap) * state.n_embd);

    std::vector<float> new_preds;
    new_preds.reserve(static_cast<size_t>(cap) * state.n_speakers);

    for (int j = 0; j < cap; ++j) {
        const int64_t i = picked[j];
        const int f = static_cast<int>(i % n_pad);

        bool silence_anchor = f >= n;
        bool sentinel_id = i >= static_cast<int64_t>(state.n_speakers) * n_pad;
        if (silence_anchor || sentinel_id) {
            // Insert a reference vector represeting pure silence.
            new_cache.insert(new_cache.end(), state.mean_sil_emb.begin(), state.mean_sil_emb.end());
            new_preds.insert(new_preds.end(), state.n_speakers, 0.f);
        } else {
            const auto emb_begin = state.spkcache.begin() + static_cast<size_t>(f) * state.n_embd;
            new_cache.insert(new_cache.end(), emb_begin, emb_begin + state.n_embd);

            const auto pred_begin = cache_preds.begin() + static_cast<size_t>(f) * state.n_speakers;
            new_preds.insert(new_preds.end(), pred_begin, pred_begin + state.n_speakers);
        }
    }

    // release and replace the old speaker cache
    state.spkcache = std::move(new_cache);

    // release and replace the old predictions cache
    state.spkcache_preds = std::move(new_preds);

    state.n_spk_frames   = cap;
}

static bool whisper_diar_read_str(const gguf_context * meta, const char * name, std::string & value) {
    const int64_t id = gguf_find_key(meta, name);
    if (id < 0 || gguf_get_kv_type(meta, id) != GGUF_TYPE_STRING) {
        WHISPER_LOG_ERROR("%s: missing or invalid metadata '%s'\n", __func__, name);
        return false;
    }
    value = gguf_get_val_str(meta, id);
    return true;
}

static bool whisper_diar_read_i32(const gguf_context * meta, const char * name, int32_t & value) {
    const int64_t id = gguf_find_key(meta, name);
    if (id < 0 || gguf_get_kv_type(meta, id) != GGUF_TYPE_UINT32) {
        WHISPER_LOG_ERROR("%s: missing or invalid metadata '%s'\n", __func__, name);
        return false;
    }
    if (gguf_get_val_u32(meta, id) > INT32_MAX) {
        WHISPER_LOG_ERROR("%s: metadata '%s' exceeds INT32_MAX\n", __func__, name);
        return false;
    }
    value = gguf_get_val_u32(meta, id);
    return true;
}

static bool whisper_diar_read_f32(const gguf_context * meta, const char * name, float & value) {
    const int64_t id = gguf_find_key(meta, name);
    if (id < 0 || gguf_get_kv_type(meta, id) != GGUF_TYPE_FLOAT32) {
        WHISPER_LOG_ERROR("%s: missing or invalid metadata '%s'\n", __func__, name);
        return false;
    }
    value = gguf_get_val_f32(meta, id);
    if (!std::isfinite(value)) {
        WHISPER_LOG_ERROR("%s: non-finite metadata '%s'\n", __func__, name);
        return false;
    }
    return true;
}

static bool whisper_diar_read_bool(const gguf_context * meta, const char * name, bool & value) {
    const int64_t id = gguf_find_key(meta, name);
    if (id < 0 || gguf_get_kv_type(meta, id) != GGUF_TYPE_BOOL) {
        WHISPER_LOG_ERROR("%s: missing or invalid metadata '%s'\n", __func__, name);
        return false;
    }
    value = gguf_get_val_bool(meta, id);
    return true;
}

static bool whisper_diar_is_probability(float p) {
    return std::isfinite(p) && p >= 0 && p <= 1;
}

static bool whisper_diar_validate_hparams(const whisper_diar_model & model) {
    const auto & hparams = model.hparams;
    const auto & scoring = model.scoring;

    const uint32_t nl = hparams.n_audio_layer;
    const uint32_t ns = hparams.n_speakers;
    const uint32_t np = hparams.n_audio_ctx;
    if (!(nl > 0 && nl <= 31 && ns > 0 && ns <= 8 && np >= 400 && np <= 1000000)) {
        WHISPER_LOG_ERROR("%s: invalid model dimensions\n", __func__);
        return false;
    }

    if (!(hparams.preemph >= 0 && hparams.preemph < 1 && hparams.log_guard > 0 && hparams.rope_base > 0)) {
        WHISPER_LOG_ERROR("%s: invalid frontend/RoPE parameters\n", __func__);
        return false;
    }

    if (!((uint32_t)scoring.sil_frames_per_spk < 264 / ns)) {
        WHISPER_LOG_ERROR("%s: invalid silence cache budget\n", __func__);
        return false;
    }

    if (!(scoring.pred_score_threshold > 0 && scoring.pred_score_threshold < 1 &&
          scoring.scores_boost_latest >= 0 &&
          scoring.strong_boost_rate >= 0 && scoring.strong_boost_rate <= 4 &&
          scoring.weak_boost_rate   >= 0 && scoring.weak_boost_rate   <= 4 &&
          whisper_diar_is_probability(scoring.min_pos_scores_rate))) {
        WHISPER_LOG_ERROR("%s: invalid cache scoring parameters\n", __func__);
        return false;
    }

    return true;
}

static bool whisper_diar_backend_init(whisper_diar_context & dctx) {
    auto & backends = dctx.state.backends;
    const auto & params = dctx.params;
    backends.reserve(2);
    ggml_backend_load_all();

    if (params.use_gpu) {
        int index = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU && index++ == params.gpu_device) {
                auto backend = ggml_backend_dev_init(dev, nullptr);
                if (backend) {
                    backends.push_back(backend);
                }
                break;
            }
        }
    }

    auto cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        WHISPER_LOG_ERROR("%s: CPU backend unavailable\n", __func__);
        return false;
    }

    auto cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    if (!cpu) {
        WHISPER_LOG_ERROR("%s: cannot initialize CPU backend\n", __func__);
        return false;
    }
    backends.push_back(cpu);

    for (auto backend : backends) {
        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto fn  = (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (fn) {
            fn(backend, params.n_threads);
        }
    }

    return true;
}

static bool whisper_diar_model_load(whisper_diar_context & dctx) {
    WHISPER_LOG_INFO("%s: loading model from '%s'\n", __func__, dctx.path_model.c_str());
    auto & model = dctx.model;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_file(dctx.path_model.c_str(), {true, nullptr}), gguf_free);
    gguf_context * meta = metadata.get();
    if (!meta) {
        return false;
    }

    {
        auto & hparams = model.hparams;
        auto & scoring = model.scoring;
        bool ok = true;

        auto read_str  = [&](whisper_diar_hparam hparam, std::string & value) {
            ok = ok && whisper_diar_read_str(meta, WHISPER_DIAR_HPARAM_NAMES.at(hparam), value);
        };
        auto read_i32  = [&](whisper_diar_hparam hparam, int32_t & value) {
            ok = ok && whisper_diar_read_i32(meta, WHISPER_DIAR_HPARAM_NAMES.at(hparam), value);
        };
        auto read_f32  = [&](whisper_diar_hparam hparam, float & value) {
            ok = ok && whisper_diar_read_f32(meta, WHISPER_DIAR_HPARAM_NAMES.at(hparam), value);
        };
        auto read_bool = [&](whisper_diar_hparam hparam, bool & value) {
            ok = ok && whisper_diar_read_bool(meta, WHISPER_DIAR_HPARAM_NAMES.at(hparam), value);
        };
        auto check_i32 = [](int32_t actual, whisper_diar_hparam hparam) {
            return actual == WHISPER_DIAR_HPARAM_MODEL_VALUES.at(hparam);
        };
        auto check_f32 = [](float actual, whisper_diar_hparam hparam) {
            return std::fabs(actual - WHISPER_DIAR_HPARAM_MODEL_FLOAT_VALUES.at(hparam)) < 1e-7f;
        };

        // Validation of field from the model that we don't actually use but still
        // want to make sure that future models don't mismatch.
        std::string architecture;
        std::string version;
        std::string type;
        std::string subsampling_type;
        std::string normalize;
        int32_t n_features = 0;
        int32_t n_transformer_layer = 0;
        int32_t output_subsampling_factor = 0;
        int32_t upsample_factor = 0;
        int32_t sample_rate = 0;
        bool qkv_bias = false;
        bool qk_norm = false;
        bool xscaling = false;
        bool pre_block_norm = false;
        bool learnable_silence = false;
        bool high_resolution = false;
        float rotary_fraction = 0.f;
        float window_size = 0.f;
        float window_stride = 0.f;
        int32_t sil_frames_per_spk = 0;

        read_str (WHISPER_DIAR_HPARAM_ARCHITECTURE,                         architecture);
        read_str (WHISPER_DIAR_HPARAM_VERSION,                              version);
        read_str (WHISPER_DIAR_HPARAM_ENCODER_TYPE,                         type);
        read_str (WHISPER_DIAR_HPARAM_SUBSAMPLING_TYPE,                     subsampling_type);
        read_i32 (WHISPER_DIAR_HPARAM_ENCODER_N_HEADS,                      hparams.n_audio_head);
        read_i32 (WHISPER_DIAR_HPARAM_ENCODER_SUBSAMPLING_FACTOR,           hparams.subsampling_factor);
        read_i32 (WHISPER_DIAR_HPARAM_ENCODER_FEAT_IN,                      n_features);
        read_i32 (WHISPER_DIAR_HPARAM_TRANSFORMER_N_LAYERS,                 n_transformer_layer);
        read_bool(WHISPER_DIAR_HPARAM_ENCODER_QKV_BIAS,                     qkv_bias);
        read_bool(WHISPER_DIAR_HPARAM_ENCODER_QK_NORM,                      qk_norm);
        read_bool(WHISPER_DIAR_HPARAM_ENCODER_XSCALING,                     xscaling);
        read_bool(WHISPER_DIAR_HPARAM_ENCODER_PRE_BLOCK_NORM,               pre_block_norm);
        read_bool(WHISPER_DIAR_HPARAM_LEARNABLE_SILENCE,                    learnable_silence);
        read_bool(WHISPER_DIAR_HPARAM_HIGH_RESOLUTION,                      high_resolution);
        read_f32 (WHISPER_DIAR_HPARAM_ENCODER_ROTARY_FRACTION,              rotary_fraction);
        read_i32 (WHISPER_DIAR_HPARAM_OUTPUT_SUBSAMPLING_FACTOR,            output_subsampling_factor);
        read_i32 (WHISPER_DIAR_HPARAM_UPSAMPLE_FACTOR,                      upsample_factor);
        read_i32 (WHISPER_DIAR_HPARAM_PREPROCESSOR_SAMPLE_RATE,             sample_rate);
        read_f32 (WHISPER_DIAR_HPARAM_PREPROCESSOR_WINDOW_SIZE,             window_size);
        read_f32 (WHISPER_DIAR_HPARAM_PREPROCESSOR_WINDOW_STRIDE,           window_stride);
        read_str (WHISPER_DIAR_HPARAM_PREPROCESSOR_NORMALIZE,               normalize);

        // Runtime fields → hparams
        read_i32 (WHISPER_DIAR_HPARAM_ENCODER_D_MODEL,                      hparams.n_audio_state);
        read_i32 (WHISPER_DIAR_HPARAM_ENCODER_D_FF,                         hparams.n_ff);
        read_i32 (WHISPER_DIAR_HPARAM_TRANSFORMER_HIDDEN_SIZE,              hparams.n_head_state);
        read_i32 (WHISPER_DIAR_HPARAM_ENCODER_N_LAYERS,                     hparams.n_audio_layer);
        read_i32 (WHISPER_DIAR_HPARAM_NUM_SPEAKERS,                         hparams.n_speakers);
        read_i32 (WHISPER_DIAR_HPARAM_ENCODER_POS_EMB_MAX_LEN,              hparams.n_audio_ctx);
        read_i32 (WHISPER_DIAR_HPARAM_PREPROCESSOR_N_FFT,                   hparams.n_fft);
        read_i32 (WHISPER_DIAR_HPARAM_PREPROCESSOR_FEATURES,                hparams.n_mels);
        read_f32 (WHISPER_DIAR_HPARAM_PREPROCESSOR_PREEMPH,                 hparams.preemph);
        read_f32 (WHISPER_DIAR_HPARAM_PREPROCESSOR_LOG_ZERO_GUARD,          hparams.log_guard);
        read_f32 (WHISPER_DIAR_HPARAM_ENCODER_ROPE_BASE,                    hparams.rope_base);

        // Scoring fields → scoring
        read_i32 (WHISPER_DIAR_HPARAM_SCORING_SPKCACHE_SIL_FRAMES_PER_SPK,  sil_frames_per_spk);
        read_f32 (WHISPER_DIAR_HPARAM_SCORING_PRED_SCORE_THRESHOLD,         scoring.pred_score_threshold);
        read_f32 (WHISPER_DIAR_HPARAM_SCORING_SCORES_BOOST_LATEST,          scoring.scores_boost_latest);
        read_f32 (WHISPER_DIAR_HPARAM_SCORING_STRONG_BOOST_RATE,            scoring.strong_boost_rate);
        read_f32 (WHISPER_DIAR_HPARAM_SCORING_WEAK_BOOST_RATE,              scoring.weak_boost_rate);
        read_f32 (WHISPER_DIAR_HPARAM_SCORING_MIN_POS_SCORES_RATE,          scoring.min_pos_scores_rate);

        if (!ok) {
            return false;
        }

        // Validate fixed-value fields.
        if (!(architecture == "sortformer" && version == "v3")) {
            WHISPER_LOG_ERROR("%s: expected Sortformer V3 GGUF\n", __func__);
            return false;
        }

        if (!(type == "transformer_rope" && subsampling_type == "feature_stacking")) {
            WHISPER_LOG_ERROR("%s: unsupported encoder\n", __func__);
            return false;
        }

        if (!(check_i32(hparams.n_audio_state,      WHISPER_DIAR_HPARAM_ENCODER_D_MODEL)            &&
              check_i32(hparams.n_audio_head,       WHISPER_DIAR_HPARAM_ENCODER_N_HEADS)            &&
              check_i32(hparams.n_ff,               WHISPER_DIAR_HPARAM_ENCODER_D_FF)               &&
              check_i32(hparams.subsampling_factor, WHISPER_DIAR_HPARAM_ENCODER_SUBSAMPLING_FACTOR) &&
              check_i32(n_features,                 WHISPER_DIAR_HPARAM_ENCODER_FEAT_IN)            &&
              check_i32(hparams.n_head_state,       WHISPER_DIAR_HPARAM_TRANSFORMER_HIDDEN_SIZE)    &&
              check_i32(n_transformer_layer,        WHISPER_DIAR_HPARAM_TRANSFORMER_N_LAYERS))) {
            WHISPER_LOG_ERROR("%s: unsupported V3 dimensions\n", __func__);
            return false;
        }

        if (!(!qkv_bias && !qk_norm && !xscaling && pre_block_norm                                &&
              learnable_silence && high_resolution && rotary_fraction == 1.0f                     &&
              check_i32(output_subsampling_factor, WHISPER_DIAR_HPARAM_OUTPUT_SUBSAMPLING_FACTOR) &&
              check_i32(upsample_factor,           WHISPER_DIAR_HPARAM_UPSAMPLE_FACTOR))) {
            WHISPER_LOG_ERROR("%s: unsupported V3 graph configuration\n", __func__);
            return false;
        }

        if (!(check_i32(sample_rate,     WHISPER_DIAR_HPARAM_PREPROCESSOR_SAMPLE_RATE)    &&
              check_i32(hparams.n_fft,   WHISPER_DIAR_HPARAM_PREPROCESSOR_N_FFT)          &&
              check_i32(hparams.n_mels,  WHISPER_DIAR_HPARAM_PREPROCESSOR_FEATURES)       &&
              check_f32(window_size,     WHISPER_DIAR_HPARAM_PREPROCESSOR_WINDOW_SIZE)    &&
              check_f32(window_stride,   WHISPER_DIAR_HPARAM_PREPROCESSOR_WINDOW_STRIDE)  &&
              normalize == "NA")) {
            WHISPER_LOG_ERROR("%s: unsupported audio frontend\n", __func__);
            return false;
        }

        scoring.sil_frames_per_spk = (int)sil_frames_per_spk;

        if (!whisper_diar_validate_hparams(model)) {
            return false;
        }
    }

    const auto & hparams = model.hparams;
    WHISPER_LOG_INFO("%s: n_audio_layer = %d\n", __func__, hparams.n_audio_layer);
    WHISPER_LOG_INFO("%s: n_audio_state = %d\n", __func__, hparams.n_audio_state);
    WHISPER_LOG_INFO("%s: n_mels        = %d\n", __func__, hparams.n_mels);
    WHISPER_LOG_INFO("%s: n_speakers    = %d\n", __func__, hparams.n_speakers);

    const size_t n_tensors = 15 + 11 * hparams.n_audio_layer;
    model.ctxs.reserve(1);
    model.buffers.reserve(1);

    ggml_init_params params = {
            /*.mem_size   =*/n_tensors * ggml_tensor_overhead(),
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        WHISPER_LOG_ERROR("%s: failed to allocate tensor context\n", __func__);
        return false;
    }
    model.ctxs.push_back(ctx);

    ggml_init_params meta_params = {
            /*.mem_size   =*/n_tensors * ggml_tensor_overhead(),
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
    };

    std::unique_ptr<ggml_context, decltype(&ggml_free)> meta_ctx(ggml_init(meta_params), ggml_free);
    if (!meta_ctx) {
        WHISPER_LOG_ERROR("%s: failed to allocate tensor metadata context\n", __func__);
        return false;
    }

    const int n_audio_state = hparams.n_audio_state;
    const int n_head_state  = hparams.n_head_state;
    const int n_ff          = hparams.n_ff;
    const int n_mels        = hparams.n_mels;
    const int n_fft         = hparams.n_fft;
    const int sf            = hparams.subsampling_factor;

    bool tensors_ok = true;
    auto create_tensor = [&](whisper_diar_tensor tensor_type, ggml_tensor * meta_tensor,
                             int layer = -1) -> ggml_tensor * {
        std::string name = WHISPER_DIAR_TENSOR_NAMES.at(tensor_type);
        if (layer >= 0) {
            const size_t marker = name.find("%d");
            if (marker != std::string::npos) {
                name.replace(marker, 2, std::to_string(layer));
            }
        }

        const int64_t id = gguf_find_tensor(meta, name.c_str());
        if (id < 0) {
            WHISPER_LOG_ERROR("%s: missing tensor '%s'\n", __func__, name.c_str());
            tensors_ok = false;
            return nullptr;
        }

        const int64_t * ne   = gguf_get_tensor_ne(meta, id);
        const ggml_type type = gguf_get_tensor_type(meta, id);

        const bool is_linear = tensor_type == WHISPER_DIAR_TENSOR_ENC_PRE_ENCODE_WEIGHT ||
                               tensor_type == WHISPER_DIAR_TENSOR_ENC_PROJ_WEIGHT       ||
                               tensor_type == WHISPER_DIAR_TENSOR_HEAD_HIDDEN_WEIGHT    ||
                               tensor_type == WHISPER_DIAR_TENSOR_HEAD_SPEAKERS_WEIGHT  ||
                               tensor_type == WHISPER_DIAR_TENSOR_ENC_ATTN_QKV_WEIGHT   ||
                               tensor_type == WHISPER_DIAR_TENSOR_ENC_ATTN_OUT_WEIGHT   ||
                               tensor_type == WHISPER_DIAR_TENSOR_ENC_FFN1_WEIGHT       ||
                               tensor_type == WHISPER_DIAR_TENSOR_ENC_FFN2_WEIGHT;
        const bool is_conv   = tensor_type == WHISPER_DIAR_TENSOR_UPSAMPLE_WEIGHT;

        const bool type_ok = is_conv ? type == GGML_TYPE_F16 :
                             is_linear ? (type == GGML_TYPE_F32 ||
                                          type == GGML_TYPE_F16 ||
                                          type == GGML_TYPE_Q8_0) : type == GGML_TYPE_F32;

        if (!type_ok || ne[0] != meta_tensor->ne[0] ||
            ne[1] != meta_tensor->ne[1] ||
            ne[2] != meta_tensor->ne[2] ||
            ne[3] != meta_tensor->ne[3]) {
            WHISPER_LOG_ERROR("%s: tensor '%s' has invalid type or shape\n", __func__, name.c_str());
            tensors_ok = false;
            return nullptr;
        }

        ggml_tensor * tensor = ggml_new_tensor(ctx, type, ggml_n_dims(meta_tensor), meta_tensor->ne);
        ggml_set_name(tensor, name.c_str());
        model.tensors[name] = tensor;
        return tensor;
    };

    model.mel_filters = create_tensor(WHISPER_DIAR_TENSOR_MEL_FILTERS,
            ggml_new_tensor_2d(meta_ctx.get(), GGML_TYPE_F32, n_fft / 2 + 1, n_mels));
    model.silence_emb = create_tensor(WHISPER_DIAR_TENSOR_SILENCE_EMBEDDING,
            ggml_new_tensor_1d(meta_ctx.get(), GGML_TYPE_F32, n_audio_state));
    model.enc_pre_w = create_tensor(WHISPER_DIAR_TENSOR_ENC_PRE_ENCODE_WEIGHT,
            ggml_new_tensor_2d(meta_ctx.get(), GGML_TYPE_F32, sf * n_mels, n_audio_state));
    model.enc_norm_w = create_tensor(WHISPER_DIAR_TENSOR_ENC_EMBED_NORM_WEIGHT,
            ggml_new_tensor_1d(meta_ctx.get(), GGML_TYPE_F32, n_audio_state));
    model.enc_norm_b = create_tensor(WHISPER_DIAR_TENSOR_ENC_EMBED_NORM_BIAS,
            ggml_new_tensor_1d(meta_ctx.get(), GGML_TYPE_F32, n_audio_state));
    model.enc_out_norm_w = create_tensor(WHISPER_DIAR_TENSOR_ENC_FINAL_NORM_WEIGHT,
            ggml_new_tensor_1d(meta_ctx.get(), GGML_TYPE_F32, n_audio_state));
    model.enc_out_norm_b = create_tensor(WHISPER_DIAR_TENSOR_ENC_FINAL_NORM_BIAS,
            ggml_new_tensor_1d(meta_ctx.get(), GGML_TYPE_F32, n_audio_state));
    model.enc_proj_w = create_tensor(WHISPER_DIAR_TENSOR_ENC_PROJ_WEIGHT,
            ggml_new_tensor_2d(meta_ctx.get(), GGML_TYPE_F32, n_audio_state, n_head_state));
    model.enc_proj_b = create_tensor(WHISPER_DIAR_TENSOR_ENC_PROJ_BIAS,
            ggml_new_tensor_1d(meta_ctx.get(), GGML_TYPE_F32, n_head_state));
    model.upsample_w = create_tensor(WHISPER_DIAR_TENSOR_UPSAMPLE_WEIGHT,
            ggml_new_tensor_3d(meta_ctx.get(), GGML_TYPE_F32, 3, n_head_state, sf * n_head_state));
    model.upsample_b = create_tensor(WHISPER_DIAR_TENSOR_UPSAMPLE_BIAS,
            ggml_new_tensor_1d(meta_ctx.get(), GGML_TYPE_F32, sf * n_head_state));
    model.head_hidden_w = create_tensor(WHISPER_DIAR_TENSOR_HEAD_HIDDEN_WEIGHT,
            ggml_new_tensor_2d(meta_ctx.get(), GGML_TYPE_F32, n_head_state, n_head_state));
    model.head_hidden_b = create_tensor(WHISPER_DIAR_TENSOR_HEAD_HIDDEN_BIAS,
            ggml_new_tensor_1d(meta_ctx.get(), GGML_TYPE_F32, n_head_state));
    model.head_spks_w = create_tensor(WHISPER_DIAR_TENSOR_HEAD_SPEAKERS_WEIGHT,
            ggml_new_tensor_2d(meta_ctx.get(), GGML_TYPE_F32, n_head_state, hparams.n_speakers));
    model.head_spks_b = create_tensor(WHISPER_DIAR_TENSOR_HEAD_SPEAKERS_BIAS,
            ggml_new_tensor_1d(meta_ctx.get(), GGML_TYPE_F32, hparams.n_speakers));

    model.layers.resize(hparams.n_audio_layer);
    for (int i = 0; i < hparams.n_audio_layer; ++i) {
        auto & layer = model.layers[i];
        layer.norm_attn_w = create_tensor(WHISPER_DIAR_TENSOR_ENC_NORM1_WEIGHT,
                ggml_new_tensor_1d(meta_ctx.get(), GGML_TYPE_F32, n_audio_state), i);
        layer.norm_attn_b = create_tensor(WHISPER_DIAR_TENSOR_ENC_NORM1_BIAS,
                ggml_new_tensor_1d(meta_ctx.get(), GGML_TYPE_F32, n_audio_state), i);
        layer.attn_qkv_w = create_tensor(WHISPER_DIAR_TENSOR_ENC_ATTN_QKV_WEIGHT,
                ggml_new_tensor_2d(meta_ctx.get(), GGML_TYPE_F32, n_audio_state, n_audio_state * 3), i);
        layer.attn_out_w = create_tensor(WHISPER_DIAR_TENSOR_ENC_ATTN_OUT_WEIGHT,
                ggml_new_tensor_2d(meta_ctx.get(), GGML_TYPE_F32, n_audio_state, n_audio_state), i);
        layer.attn_out_b = create_tensor(WHISPER_DIAR_TENSOR_ENC_ATTN_OUT_BIAS,
                ggml_new_tensor_1d(meta_ctx.get(), GGML_TYPE_F32, n_audio_state), i);
        layer.norm_ff_w = create_tensor(WHISPER_DIAR_TENSOR_ENC_NORM2_WEIGHT,
                ggml_new_tensor_1d(meta_ctx.get(), GGML_TYPE_F32, n_audio_state), i);
        layer.norm_ff_b = create_tensor(WHISPER_DIAR_TENSOR_ENC_NORM2_BIAS,
                ggml_new_tensor_1d(meta_ctx.get(), GGML_TYPE_F32, n_audio_state), i);
        layer.ff1_w = create_tensor(WHISPER_DIAR_TENSOR_ENC_FFN1_WEIGHT,
                ggml_new_tensor_2d(meta_ctx.get(), GGML_TYPE_F32, n_audio_state, n_ff), i);
        layer.ff1_b = create_tensor(WHISPER_DIAR_TENSOR_ENC_FFN1_BIAS,
                ggml_new_tensor_1d(meta_ctx.get(), GGML_TYPE_F32, n_ff), i);
        layer.ff2_w = create_tensor(WHISPER_DIAR_TENSOR_ENC_FFN2_WEIGHT,
                ggml_new_tensor_2d(meta_ctx.get(), GGML_TYPE_F32, n_ff, n_audio_state), i);
        layer.ff2_b = create_tensor(WHISPER_DIAR_TENSOR_ENC_FFN2_BIAS,
                ggml_new_tensor_1d(meta_ctx.get(), GGML_TYPE_F32, n_audio_state), i);
    }

    meta_ctx.reset();

    if (!tensors_ok) {
        return false;
    }

    std::ifstream file(dctx.path_model, std::ios::binary | std::ios::ate);
    const auto end = file.tellg();
    if (!file || end < 0) {
        WHISPER_LOG_ERROR("%s: cannot determine GGUF file size\n", __func__);
        return false;
    }

    const uint64_t size = (uint64_t)end;
    const uint64_t base = gguf_get_data_offset(meta);
    for (int64_t i = 0; i < gguf_get_n_tensors(meta); ++i) {
        const char * name = gguf_get_tensor_name(meta, i);
        if (model.tensors.count(name) == 0 &&
            std::strcmp(name, WHISPER_DIAR_TENSOR_NAMES.at(WHISPER_DIAR_TENSOR_ACTIVITY_HEAD0_WEIGHT)) != 0 &&
            std::strcmp(name, WHISPER_DIAR_TENSOR_NAMES.at(WHISPER_DIAR_TENSOR_ACTIVITY_HEAD0_BIAS)) != 0 &&
            std::strcmp(name, WHISPER_DIAR_TENSOR_NAMES.at(WHISPER_DIAR_TENSOR_ACTIVITY_HEAD1_WEIGHT)) != 0 &&
            std::strcmp(name, WHISPER_DIAR_TENSOR_NAMES.at(WHISPER_DIAR_TENSOR_ACTIVITY_HEAD1_BIAS)) != 0) {
            WHISPER_LOG_ERROR("%s: unknown tensor '%s'\n", __func__, name);
            return false;
        }
        const uint64_t offset = gguf_get_tensor_offset(meta, i);
        const uint64_t bytes  = gguf_get_tensor_size(meta, i);
        if (base > size || offset > size - base || bytes > size - base - offset) {
            WHISPER_LOG_ERROR("%s: truncated payload for '%s'\n", __func__, name);
            return false;
        }
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, dctx.state.backends.front());
    if (!buffer) {
        WHISPER_LOG_ERROR("%s: failed to allocate weight buffer\n", __func__);
        return false;
    }
    model.buffers.push_back(buffer);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    size_t total_size = 0;
    std::vector<char> read_buf;
    for (const auto & entry : model.tensors) {
        ggml_tensor * tensor = entry.second;
        const int64_t id     = gguf_find_tensor(meta, entry.first.c_str());
        const size_t  bytes  = ggml_nbytes(tensor);
        file.seekg(base + gguf_get_tensor_offset(meta, id));
        if (ggml_backend_buffer_is_host(tensor->buffer)) {
            file.read((char *)tensor->data, bytes);
        } else {
            read_buf.resize(bytes);
            file.read(read_buf.data(), bytes);
            if (file) {
                ggml_backend_tensor_set(tensor, read_buf.data(), 0, bytes);
            }
        }
        if (!file) {
            WHISPER_LOG_ERROR("%s: failed to read tensor '%s'\n", __func__, entry.first.c_str());
            return false;
        }
        total_size += bytes;
        model.n_loaded++;
    }

    if (model.n_loaded != (int)model.tensors.size()) {
        WHISPER_LOG_ERROR("%s: not all model tensors loaded\n", __func__);
        return false;
    }

    WHISPER_LOG_INFO("%s: model size = %.2f MB (%d tensors)\n", __func__, total_size / 1e6, model.n_loaded);
    model.filters.resize((hparams.n_fft / 2 + 1) * hparams.n_mels);
    model.silence.resize(hparams.n_audio_state);
    ggml_backend_tensor_get(model.mel_filters, model.filters.data(), 0, model.filters.size() * sizeof(float));
    ggml_backend_tensor_get(model.silence_emb, model.silence.data(), 0, model.silence.size() * sizeof(float));

    for (int i = 0; i < 400; ++i) {
        model.window[i] = 0.5f * (1 - std::cos(2 * WHISPER_DIAR_PI * i / (400 - 1)));
    }

    return true;
}

static void whisper_diar_fft(float * re, float * im) {
    for (int i = 1, j = 0; i < 512; ++i) {
        int bit = 256;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2; len <= 512; len *= 2) {
        const float wr = std::cos(-2 * WHISPER_DIAR_PI / len);
        const float wi = std::sin(-2 * WHISPER_DIAR_PI / len);
        for (int i = 0; i < 512; i += len) {
            float cr = 1, ci = 0;
            for (int j = 0; j < len / 2; ++j) {
                const int   a = i + j, b = a + len / 2;
                const float tr = cr * re[b] - ci * im[b];
                const float ti = cr * im[b] + ci * re[b];
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                const float next = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = next;
            }
        }
    }
}

static std::vector<float> whisper_diar_pcm_to_mel(const whisper_diar_model & model,
        const float * audio, int64_t n, int64_t first, int count) {
    const int n_mels    = model.hparams.n_mels;
    const int n_fft_out = model.hparams.n_fft / 2 + 1;
    std::vector<float> out(count * n_mels);
    for (int f = 0; f < count; ++f) {
        float re[512] = {}, im[512] = {};
        for (int j = 0; j < 400; ++j) {
            const int64_t sample = (first + f) * 160 - 200 + j;
            if (sample >= 0 && sample < n) {
                re[j + 56] = (audio[sample] - (sample ? model.hparams.preemph * audio[sample - 1] : 0)) * model.window[j];
            }
        }
        whisper_diar_fft(re, im);
        for (int m = 0; m < n_mels; ++m) {
            float sum = 0;
            for (int k = 0; k < n_fft_out; ++k) {
                sum += model.filters[m * n_fft_out + k] * (re[k] * re[k] + im[k] * im[k]);
            }
            out[f * n_mels + m] = std::log(sum + model.hparams.log_guard);
        }
    }
    return out;
}

static ggml_cgraph * whisper_diar_build_graph(whisper_diar_context & dctx, int frames, int prefix) {
    const whisper_diar_model & model = dctx.model;
    const auto & hparams = model.hparams;
    const int sf      = hparams.subsampling_factor;
    const int time    = prefix + frames;
    const int n_heads = hparams.n_audio_head;
    const int d_head  = hparams.n_audio_state / n_heads;

    ggml_init_params params = {
        /*.mem_size   =*/ dctx.state.meta.size(),
        /*.mem_buffer =*/ dctx.state.meta.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return nullptr;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, WHISPER_DIAR_MAX_NODES, false);

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sf * hparams.n_mels, frames);
    ggml_set_input(input);
    ggml_set_name(input, "input");

    // project down into the model hidden vector space.
    ggml_tensor * hidden = ggml_mul_mat(ctx, model.enc_pre_w, input);
    ggml_set_output(hidden);
    ggml_set_name(hidden, "hidden");

    ggml_tensor * cur = hidden;

    // This is conditionally adding an input which will cause the graph to be
    // reshaped and not be reused. We should fix this and always add a cache
    // but I need to understand the operations first to see how to to this.
    ggml_tensor * cache = nullptr;
    if (prefix) {
        cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hparams.n_audio_state, prefix);
        ggml_set_input(cache);
        ggml_set_name(cache, "cache");
        cur = ggml_concat(ctx, cache, cur, 1);
    }

    auto * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, time);
    ggml_set_input(positions);
    ggml_set_name(positions, "positions");

    cur = ggml_norm(ctx, cur, 1e-5f);
    cur = ggml_add(ctx,
            ggml_mul(ctx, cur, model.enc_norm_w),
            model.enc_norm_b);

    struct ggml_tensor * inpL = cur;

    for (int i = 0; i < model.hparams.n_audio_layer; ++i) {
        const auto & layer = model.layers[i];

        // norm
        {
            cur = ggml_norm(ctx, inpL, 1e-5f);

            cur = ggml_add(ctx,
                    ggml_mul(ctx, cur, layer.norm_attn_w),
                    layer.norm_attn_b);
        }

        // self attention
        {
            ggml_tensor * qkv = ggml_mul_mat(ctx, layer.attn_qkv_w, cur);

            ggml_tensor * Qcur = ggml_view_3d(ctx, qkv, d_head, n_heads, time,
                    d_head * sizeof(float), qkv->nb[1], 0);
            Qcur = ggml_rope_ext(ctx, Qcur, positions, nullptr, d_head, GGML_ROPE_TYPE_NEOX,
                    hparams.n_audio_ctx, hparams.rope_base, 1, 0, 1, 0, 0);
            Qcur = ggml_permute(ctx, Qcur, 0, 2, 1, 3);

            ggml_tensor * Kcur = ggml_view_3d(ctx, qkv, d_head, n_heads, time,
                    d_head * sizeof(float), qkv->nb[1], (size_t)hparams.n_audio_state * sizeof(float));
            Kcur = ggml_rope_ext(ctx, Kcur, positions, nullptr, d_head, GGML_ROPE_TYPE_NEOX,
                    hparams.n_audio_ctx, hparams.rope_base, 1, 0, 1, 0, 0);
            Kcur = ggml_permute(ctx, Kcur, 0, 2, 1, 3);

            ggml_tensor * Vcur = ggml_view_3d(ctx, qkv, d_head, n_heads, time,
                    d_head * sizeof(float), qkv->nb[1], (size_t)2 * hparams.n_audio_state * sizeof(float));
            Vcur = ggml_permute(ctx, Vcur, 0, 2, 1, 3);

            ggml_tensor * attn;
            if (dctx.params.flash_attn) {
                // ggml_flash_attn_ext output shape is [v.ne[0], q.ne[2], q.ne[1]] =
                // [head_dim, n_heads, seq_len] — permute is already baked in.
                attn = ggml_flash_attn_ext(ctx, Qcur, Kcur, Vcur, nullptr,
                        1.0f / sqrtf((float)d_head), 0.0f, 0.0f);
            } else {
                ggml_tensor * KQ         = ggml_mul_mat(ctx, Kcur, Qcur);
                ggml_tensor * KQ_softmax = ggml_soft_max_ext(ctx, KQ, nullptr, 1.0f / sqrtf((float)d_head), 0);
                ggml_tensor * V_trans    = ggml_cont(ctx, ggml_transpose(ctx, Vcur));
                attn = ggml_mul_mat(ctx, V_trans, KQ_softmax);
                // manual path gives [head_dim, seq_len, n_heads]; permute to [head_dim, n_heads, seq_len].
                attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
            }
            attn = ggml_reshape_2d(ctx, attn, hparams.n_audio_state, time);

            attn = ggml_mul_mat(ctx, layer.attn_out_w, attn);
            attn = ggml_add(ctx, attn, layer.attn_out_b);

            cur  = ggml_add(ctx, inpL, attn);
        }

        struct ggml_tensor * inp_ff = cur;

        // feed-forward network
        {
            // norm
            cur = ggml_norm(ctx, inp_ff, 1e-5f);

            cur = ggml_add(ctx,
                    ggml_mul(ctx, cur, layer.norm_ff_w),
                    layer.norm_ff_b);

            cur = ggml_mul_mat(ctx, layer.ff1_w, cur);
            cur = ggml_add(ctx, cur, layer.ff1_b);

            cur = ggml_gelu_erf(ctx, cur);

            cur = ggml_mul_mat(ctx, layer.ff2_w, cur);
            cur = ggml_add(ctx, cur, layer.ff2_b);

        }
        inpL = ggml_add(ctx, cur, inp_ff);
    }

    cur = inpL;

    // norm
    {
        cur = ggml_norm(ctx, cur, 1e-5f);

        cur = ggml_add(ctx,
                ggml_mul(ctx, cur, model.enc_out_norm_w),
                model.enc_out_norm_b);
    }

    // we currently have 512 dimension which was required by the self-attention
    // but to output 8 speaker probabilities we don't need all of those dimensions.
    // The following down projection will project this down to 192 dimensions.
    cur = ggml_mul_mat(ctx, model.enc_proj_w, cur);
    cur = ggml_add(ctx, cur, model.enc_proj_b);

    // transpose for convolution
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
    // convolution to look accross neighboring frames to interpolate between them.
    cur = ggml_conv_1d(ctx, model.upsample_w, cur, 1, 1, 1);
    // add bias (broadcasting first which is what reshape is doing)
    cur = ggml_add(ctx, cur, ggml_reshape_2d(ctx, model.upsample_b, 1, hparams.n_head_state * sf));
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));

    // reshape back to the original 10ms resolution.
    cur = ggml_reshape_2d(ctx, cur, hparams.n_head_state, time * sf);

    // eliminate noise or background by zeroing out negative values, but keep
    // vocal characteristics.
    cur = ggml_relu(ctx, cur);

    // mix features all 192 acoustic features together.
    cur = ggml_mul_mat(ctx, model.head_hidden_w, cur);
    cur = ggml_add(ctx, cur, model.head_hidden_b);
    cur = ggml_relu(ctx, cur);

    // map the 192 accoustic rules to 8 speaker slots.
    cur = ggml_mul_mat(ctx, model.head_spks_w, cur);
    cur = ggml_add(ctx, cur, model.head_spks_b);

    // Sigmoid is used as speakers can be talking at the same time and we need
    // to have a probability for each channel at each timeframe. The output
    // shape will be be [8, 112], where we will have 112 rows representing 10ms
    // and each will have 8 values, a probability for each speaker.
    ggml_tensor * output = ggml_sigmoid(ctx, cur);
    ggml_set_output(output);
    ggml_set_name(output, "output");

    ggml_build_forward_expand(gf, output);

    ggml_free(ctx);

    return gf;
}

static bool whisper_diar_process(whisper_diar_context & dctx,
        const std::vector<float> & mel, int valid_mel, int right_mel) {
    whisper_diar_state & state = dctx.state;
    whisper_diar_cache & cache = state.cache;

    const auto & hparams    = dctx.model.hparams;
    const int    n_speakers = hparams.n_speakers;
    const int    sf         = hparams.subsampling_factor;
    const int    n_frames   = (valid_mel + right_mel + sf - 1) / sf;
    const int    prefix     = cache.n_spk_frames + cache.n_fifo_frames;
    const int    time       = prefix + n_frames;

    if (time > hparams.n_audio_ctx) {
        WHISPER_LOG_ERROR("%s: chunk exceeds position limit\n", __func__);
        return false;
    }

    ggml_backend_sched_t sched = state.sched;
    ggml_backend_sched_reset(sched);

    ggml_cgraph * gf = whisper_diar_build_graph(dctx, n_frames, prefix);
    if (!gf) {
        WHISPER_LOG_ERROR("%s: failed to build graph\n", __func__);
        return false;
    }

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        WHISPER_LOG_ERROR("%s: failed to allocate compute graph\n", __func__);
        return false;
    }

    ggml_tensor * input      = ggml_graph_get_tensor(gf, "input");
    ggml_tensor * positions  = ggml_graph_get_tensor(gf, "positions");
    ggml_tensor * hidden     = ggml_graph_get_tensor(gf, "hidden");
    ggml_tensor * output     = ggml_graph_get_tensor(gf, "output");

    state.buf_padded.assign(n_frames * sf * hparams.n_mels, 0.0f);
    std::copy(mel.begin(), mel.end(), state.buf_padded.begin());
    ggml_backend_tensor_set(input, state.buf_padded.data(), 0, state.buf_padded.size() * sizeof(float));

    if (prefix) {
        // set the cache input by copying the current speaker cache and then the fifo.
        state.buf_cache.resize(cache.spkcache.size() + cache.fifo.size());
        // copy the speaker cache.
        std::copy(cache.spkcache.begin(), cache.spkcache.end(), state.buf_cache.begin());
        // copy the fifo.
        std::copy(cache.fifo.begin(), cache.fifo.end(), state.buf_cache.begin() + cache.spkcache.size());

        ggml_tensor * cache_t = ggml_graph_get_tensor(gf, "cache");
        ggml_backend_tensor_set(cache_t, state.buf_cache.data(), 0, state.buf_cache.size() * sizeof(float));
    }

    state.buf_pos.resize(time);
    std::iota(state.buf_pos.begin(), state.buf_pos.end(), 0);
    ggml_backend_tensor_set(positions, state.buf_pos.data(), 0, state.buf_pos.size() * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        WHISPER_LOG_ERROR("%s: graph compute failed\n", __func__);
        return false;
    }

    state.buf_probs.resize(time * sf * n_speakers);
    ggml_backend_tensor_get(output, state.buf_probs.data(), 0, state.buf_probs.size() * sizeof(float));

    // write probabilites to probs.
    state.probs.insert(state.probs.end(),
                       state.buf_probs.begin() + prefix * sf * n_speakers,
                       state.buf_probs.begin() + (prefix * sf + valid_mel) * n_speakers);

    // Update cache.

    // Get the hidden state of the previous chunk, this is the log-mel
    // spectrogram after it has been projected into the models hidden vector space.
    // So this is 80ms of state saved before the model processes it.
    state.buf_hidden.resize(n_frames * hparams.n_audio_state);
    ggml_backend_tensor_get(hidden, state.buf_hidden.data(), 0, state.buf_hidden.size() * sizeof(float));

    // The following will go through and average sf, 10ms frames for each speaker,
    // bringing the value to the 80ms time scale, which will make it easier for the
    // calculations required for updating the cache.
    state.buf_probs_80ms.assign(time * n_speakers, 0.0f);
    for (int f = 0; f < time; ++f) {
        for (int j = 0; j < sf; ++j) {
            for (int s = 0; s < n_speakers; ++s) {
                state.buf_probs_80ms[f * n_speakers + s] += state.buf_probs[(f * sf + j) * n_speakers + s] / sf;
            }
        }
    }
    whisper_diar_cache_update(cache, state.buf_hidden.data(), n_frames, state.buf_probs_80ms.data(), (right_mel + sf - 1) / sf);

    return true;
}

struct whisper_diar_segment {
    int64_t t0;
    int64_t t1;
    int     speaker;
};

struct whisper_diar_segments {
    std::vector<whisper_diar_segment> data;
};

whisper_diar_context_params whisper_diar_default_context_params(void) {
    whisper_diar_context_params result = {
        /*.n_threads  =*/ 4,
        /*.use_gpu    =*/ false,
        /*.gpu_device =*/ 0,
        /*.flash_attn =*/ true,
    };
    return result;
}

whisper_diar_params whisper_diar_default_params(void) {
    whisper_diar_params result = {
        /*.start_threshold         =*/ 0.641f,
        /*.stop_threshold          =*/ 0.561f,
        /*.start_pad_ms            =*/ 229,
        /*.end_pad_ms              =*/ 79,
        /*.min_speech_duration_ms  =*/ 511,
        /*.min_silence_duration_ms =*/ 296,
    };
    return result;
}

whisper_diar_context * whisper_diar_init_from_file_with_params(const char * path,
        whisper_diar_context_params params) {
    if (!path || !path[0] || params.n_threads <= 0 || params.gpu_device < 0) {
        WHISPER_LOG_ERROR("%s: invalid context parameters\n", __func__);
        return nullptr;
    }

    whisper_diar_context * ctx = nullptr;
    try {
        ctx = new whisper_diar_context;
        ctx->params = params;
        ctx->path_model = path;

        if (whisper_diar_backend_init(*ctx) && whisper_diar_model_load(*ctx)) {
            auto & state = ctx->state;
            state.meta.resize(WHISPER_DIAR_MAX_NODES * ggml_tensor_overhead() +
                              ggml_graph_overhead_custom(WHISPER_DIAR_MAX_NODES, false));
            state.sched = ggml_backend_sched_new(state.backends.data(), nullptr, state.backends.size(),
                                                 WHISPER_DIAR_MAX_NODES, false, true);
            if (state.sched) {
                return ctx;
            }
            WHISPER_LOG_ERROR("%s: failed to allocate scheduler\n", __func__);
        }
    } catch (const std::exception & e) {
        WHISPER_LOG_ERROR("%s: exception during initialization: %s\n", __func__, e.what());
    } catch (...) {
        WHISPER_LOG_ERROR("%s: unknown exception during initialization\n", __func__);
    }
    whisper_diar_free(ctx);

    return nullptr;
}

bool whisper_diar_detect_speakers(whisper_diar_context * ctx, const float * samples, int64_t n) {
    if (!ctx) {
        return false;
    }

    ctx->state.probs.clear();

    if (!n) {
        return true;
    }

    try {
        if (!(n >= 0 && (n == 0 || samples))) {
            WHISPER_LOG_ERROR("%s: invalid audio input\n", __func__);
            return false;
        }

        if (!((uint64_t)n <= std::numeric_limits<size_t>::max() / sizeof(float))) {
            WHISPER_LOG_ERROR("%s: audio too large\n", __func__);
            return false;
        }

        for (int64_t i = 0; i < n; ++i) {
            if (!(std::isfinite(samples[i]) && std::fabs(samples[i]) <= 1.0f)) {
                WHISPER_LOG_ERROR("%s: expected finite normalized PCM\n", __func__);
                return false;
            }
        }

        whisper_diar_cache_params cache_params;
        whisper_diar_cache_init(ctx->state.cache,
                cache_params, ctx->model.scoring, ctx->model.hparams.n_speakers, ctx->model.hparams.n_audio_state, ctx->model.silence);

        // 16000 samples/sec * 0.010 sec = 160 samples per frame
        const int   sf         = ctx->model.hparams.subsampling_factor;
        const int64_t n_mel_frames = n / 160 + 1;
        for (int64_t i = 0; i < n_mel_frames; i += cache_params.chunk_len * sf) {
            const int count = (int)std::min<int64_t>(cache_params.chunk_len * sf, n_mel_frames - i);
            const int right = (int)std::min<int64_t>(sf, n_mel_frames - i - count);

            const auto mel = whisper_diar_pcm_to_mel(ctx->model, samples, n, i, count + right);

            if (!whisper_diar_process(*ctx, mel, count, right)) {
                ctx->state.probs.clear();
                ggml_backend_sched_reset(ctx->state.sched);
                return false;
            }
        }

        ctx->state.probs.resize((n / 160 + (n % 160 != 0)) * ctx->model.hparams.n_speakers);
        return true;
    } catch (const std::exception & e) {
        WHISPER_LOG_ERROR("%s: %s\n", __func__, e.what());
        ctx->state.probs.clear();
        if (ctx->state.sched) {
            ggml_backend_sched_reset(ctx->state.sched);
        }
        return false;
    } catch (...) {
        WHISPER_LOG_ERROR("%s: unknown exception during inference\n", __func__);
        ctx->state.probs.clear();
        ggml_backend_sched_reset(ctx->state.sched);
        return false;
    }
}

int whisper_diar_n_speakers(const whisper_diar_context * ctx) {
    return ctx ? ctx->model.hparams.n_speakers : 0;
}

int64_t whisper_diar_n_frames(const whisper_diar_context * ctx) {
    return ctx ? ctx->state.probs.size() / ctx->model.hparams.n_speakers : 0;
}

const float * whisper_diar_probs(const whisper_diar_context * ctx) {
    return ctx && !ctx->state.probs.empty() ? ctx->state.probs.data() : nullptr;
}

int whisper_diar_speaker_for_time(const whisper_diar_context * ctx, int64_t t0, int64_t t1, float threshold) {
    if (!ctx || !whisper_diar_is_probability(threshold)) {
        return -1;
    }
    t0 = std::max<int64_t>(0, t0);
    t1 = std::min(t1, whisper_diar_n_frames(ctx));
    if (t1 <= t0) {
        return -1;
    }
    double best = threshold;
    int speaker = -1;
    for (int s = 0; s < ctx->model.hparams.n_speakers; ++s) {
        double sum = 0;
        for (int64_t f = t0; f < t1; ++f) {
            sum += ctx->state.probs[f * ctx->model.hparams.n_speakers + s];
        }
        const double mean = sum / (t1 - t0);
        if (mean >= threshold && (speaker < 0 || mean > best)) {
            best = mean;
            speaker = s;
        }
    }
    return speaker;
}

whisper_diar_segments * whisper_diar_segments_from_probs(const whisper_diar_context * ctx, whisper_diar_params p) {
    if (!ctx || !whisper_diar_is_probability(p.start_threshold)  ||
                !whisper_diar_is_probability(p.stop_threshold)   ||
                p.stop_threshold > p.start_threshold             ||
                p.start_pad_ms < 0                               ||
                p.end_pad_ms < 0                                 ||
                p.min_speech_duration_ms < 0                     ||
                p.min_silence_duration_ms < 0) {
        return nullptr;
    }

    try {
        std::unique_ptr<whisper_diar_segments> result(new whisper_diar_segments);
        const int64_t n_frames    = whisper_diar_n_frames(ctx);

        // since speech does not start and stop at exact thresholds, there is
        // usually valid audio just before the probability raises and just after
        // it falls. So we add some padding to the start and end of each detected
        // segment. So the start of a segment (t0) is pulled back by start_pad
        // and the end of a segment (t1) is pushed forward by end_pad.
        const int64_t start_pad = (p.start_pad_ms + 5) / 10;
        const int64_t end_pad   = (p.end_pad_ms   + 5) / 10;

        for (int s = 0; s < ctx->model.hparams.n_speakers; ++s) {
            std::vector<whisper_diar_segment> segments;

            // Find contigous frames where speaker s is active
            int64_t start = -1; // -1 = not currently in a segment
            for (int64_t f = 0; f <= n_frames; ++f) {
                const float v = f < n_frames ? ctx->state.probs[f * ctx->model.hparams.n_speakers + s] : -1.0f;

                // if we are not currently in a segment and f is a valid frame,
                // and the start threshold is met, then start a new segment.
                if (start < 0 && f < n_frames && v >= p.start_threshold) {
                    start = f;
                }

                // if we are currently in a segment and f is not a valid frame,
                // or the stop threshold is met, then end the current segment.
                if (start >= 0 && (f == n_frames || v < p.stop_threshold)) {
                    segments.push_back({start, f, s});
                    start = -1;
                }
            }

            // padding of t0 and t1
            for (auto & seg : segments) {
                // pull back the start of the segment but clamp to zero.
                seg.t0 = std::max<int64_t>(0, seg.t0 - start_pad);
                // push forward the end of the segment but clamp to n_frames.
                seg.t1 = std::min<int64_t>(n_frames, seg.t1 + end_pad);
            }

            // by applying the padding above it is now possible that segments
            // overlap. So we merge overlapping segments here.
            std::vector<whisper_diar_segment> merged;
            for (const auto & seg : segments) {
                if (merged.empty()) {
                    merged.push_back(seg);
                    continue;
                }

                bool t0_overlap = seg.t0 <= merged.back().t1;
                bool sil_to_short = (seg.t0 - merged.back().t1) * 10 < p.min_silence_duration_ms;

                if (t0_overlap || sil_to_short) {
                    merged.back().t1 = std::max(merged.back().t1, seg.t1);
                } else {
                    merged.push_back(seg);
                }
            }

            for (const auto & seg : merged) {
                // only add a segments if it is longer than the min speech duration.
                // (t0/t1 are in centiseconds so we multiply by 10 to convert to ms)
                if ((seg.t1 - seg.t0) * 10 >= p.min_speech_duration_ms) {
                    result->data.push_back(seg);
                }
            }
        }

        std::sort(result->data.begin(), result->data.end(),
            [](const whisper_diar_segment & a, const whisper_diar_segment & b) {
                // sort by start time (t0), if tied then sort by speaker.
                return a.t0 != b.t0 ? a.t0 < b.t0 : a.speaker < b.speaker;
        });
        return result.release();
    } catch (const std::exception & e) {
        WHISPER_LOG_ERROR("%s: %s\n", __func__, e.what());
        return nullptr;
    } catch (...) {
        WHISPER_LOG_ERROR("%s: unknown exception during segmentation\n", __func__);
        return nullptr;
    }
}

int whisper_diar_segments_n_segments(const whisper_diar_segments * s) {
    return s ? (int)s->data.size() : 0;
}

int64_t whisper_diar_segments_get_segment_t0(const whisper_diar_segments * s, int i) {
    return s && i >= 0 && (size_t)i < s->data.size() ? s->data[i].t0 : -1;
}

int64_t whisper_diar_segments_get_segment_t1(const whisper_diar_segments * s, int i) {
    return s && i >= 0 && (size_t)i < s->data.size() ? s->data[i].t1 : -1;
}

int whisper_diar_segments_get_speaker(const whisper_diar_segments * s, int i) {
    return s && i >= 0 && (size_t)i < s->data.size() ? s->data[i].speaker : -1;
}

void whisper_diar_free_segments(whisper_diar_segments * s) {
    delete s;
}

void whisper_diar_free(whisper_diar_context * ctx) {
    if (!ctx) {
        return;
    }

    auto & state = ctx->state;
    if (state.sched) {
        ggml_backend_sched_free(state.sched);
    }

    for (ggml_backend_buffer_t buffer : ctx->model.buffers) {
        ggml_backend_buffer_free(buffer);
    }

    for (ggml_context * context : ctx->model.ctxs) {
        ggml_free(context);
    }

    for (ggml_backend_t backend : state.backends) {
        ggml_backend_free(backend);
    }
    delete ctx;
}
