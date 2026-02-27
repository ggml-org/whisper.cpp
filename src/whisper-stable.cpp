#include "whisper-stable.h"
#include "whisper.h"

#include <algorithm>
#include <cmath>
#include <limits>

std::vector<std::pair<int64_t, int64_t>> whisper_stable_build_silence_map(
        const float * vad_probs,
        int           n_probs,
        int           n_window,
        int           sample_rate,
        float         threshold,
        int64_t       min_silence_dur_cs) {

    std::vector<std::pair<int64_t, int64_t>> silence;
    if (!vad_probs || n_probs <= 0) {
        return silence;
    }

    // Each VAD frame = n_window samples. Convert frame index to centiseconds.
    // cs = frame_index * n_window * 100 / sample_rate
    const double cs_per_frame = (double)n_window * 100.0 / sample_rate;

    int64_t region_start = -1;

    for (int i = 0; i < n_probs; i++) {
        const bool is_silent = vad_probs[i] < threshold;

        if (is_silent && region_start < 0) {
            region_start = (int64_t)(i * cs_per_frame);
        } else if (!is_silent && region_start >= 0) {
            const int64_t region_end = (int64_t)(i * cs_per_frame);
            if (region_end - region_start >= min_silence_dur_cs) {
                silence.push_back({region_start, region_end});
            }
            region_start = -1;
        }
    }

    // Handle trailing silence
    if (region_start >= 0) {
        const int64_t region_end = (int64_t)(n_probs * cs_per_frame);
        if (region_end - region_start >= min_silence_dur_cs) {
            silence.push_back({region_start, region_end});
        }
    }

    return silence;
}

std::vector<std::pair<int64_t, int64_t>> whisper_stable_copy_vad_mapping(
        const std::vector<vad_time_mapping> & mapping_table) {
    std::vector<std::pair<int64_t, int64_t>> mapping;
    mapping.reserve(mapping_table.size());

    for (const auto & m : mapping_table) {
        mapping.push_back({m.processed_time, m.original_time});
    }

    return mapping;
}

int64_t whisper_stable_silence_overlap_len(
        int64_t t0,
        int64_t t1,
        const std::vector<std::pair<int64_t, int64_t>> & silence_regions_cs) {
    if (t1 <= t0 || silence_regions_cs.empty()) {
        return 0;
    }

    int64_t overlap = 0;
    auto it = std::lower_bound(
        silence_regions_cs.begin(), silence_regions_cs.end(), t0,
        [](const std::pair<int64_t, int64_t> & region, int64_t t) {
            return region.second <= t;
        });

    for (; it != silence_regions_cs.end() && it->first < t1; ++it) {
        const int64_t ss = std::max<int64_t>(t0, it->first);
        const int64_t se = std::min<int64_t>(t1, it->second);
        if (se > ss) {
            overlap += (se - ss);
        }
    }

    return overlap;
}

static std::vector<std::pair<int64_t, int64_t>> whisper_stable_build_speech_regions(
        const std::vector<std::pair<int64_t, int64_t>> & silence,
        int64_t total_end_cs,
        int64_t min_silence_dur_cs = 0) {
    std::vector<std::pair<int64_t, int64_t>> speech;
    if (total_end_cs <= 0) {
        return speech;
    }

    int64_t cur = 0;
    for (const auto & r : silence) {
        const int64_t ss = std::max<int64_t>(0, r.first);
        const int64_t se = std::max<int64_t>(ss, r.second);

        // Skip short silences — treat them as speech for snapping purposes.
        if (se - ss < min_silence_dur_cs) {
            continue;
        }

        if (se <= cur) {
            continue;
        }

        if (ss > cur) {
            speech.push_back({cur, ss});
        }

        cur = se;
        if (cur >= total_end_cs) {
            break;
        }
    }

    if (cur < total_end_cs) {
        speech.push_back({cur, total_end_cs});
    }

    return speech;
}

void whisper_stable_snap_timestamps(
        int64_t     * words_t0,
        int64_t     * words_t1,
        int           n_words,
        const int   * seg_first_word,
        const int   * seg_word_count,
        int           n_segments,
        const std::vector<std::pair<int64_t, int64_t>> & silence,
        int64_t       min_word_dur_cs,
        int64_t       min_snap_silence_dur_cs) {
    if (n_words <= 0) {
        return;
    }

    // Build per-word flags: is_first_in_seg, is_last_in_seg
    std::vector<bool> is_first(n_words, false);
    std::vector<bool> is_last(n_words, false);
    for (int s = 0; s < n_segments; s++) {
        const int first = seg_first_word[s];
        const int count = seg_word_count[s];
        if (count > 0) {
            is_first[first] = true;
            is_last[first + count - 1] = true;
        }
    }

    int64_t total_end_cs = 0;
    for (int i = 0; i < n_words; ++i) {
        total_end_cs = std::max(total_end_cs, words_t1[i]);
    }
    if (!silence.empty()) {
        total_end_cs = std::max(total_end_cs, silence.back().second);
    }

    const auto speech_regions = whisper_stable_build_speech_regions(silence, total_end_cs, min_snap_silence_dur_cs);

    for (int w = 0; w < n_words; w++) {
        const int64_t t0 = words_t0[w];
        const int64_t t1 = words_t1[w];

        if (t0 >= t1) {
            continue;
        }

        int64_t new_t0 = t0;
        int64_t new_t1 = t1;
        bool anchor_left = false;
        bool anchor_right = false;

        int64_t best_overlap = 0;
        int64_t best_start = 0;
        int64_t best_end = 0;
        for (const auto & sr : speech_regions) {
            if (sr.second <= t0) {
                continue;
            }
            if (sr.first >= t1) {
                break;
            }

            const int64_t ov0 = std::max<int64_t>(t0, sr.first);
            const int64_t ov1 = std::min<int64_t>(t1, sr.second);
            const int64_t ovd = ov1 - ov0;
            if (ovd > best_overlap) {
                best_overlap = ovd;
                best_start = ov0;
                best_end = ov1;
            }
        }

        if (best_overlap > 0) {
            new_t0 = best_start;
            new_t1 = best_end;
            anchor_left = (best_start <= t0);
            anchor_right = (best_end >= t1);
        } else {
            const int prev_w = is_first[w] ? -1 : (w - 1);
            const int next_w = is_last[w]  ? -1 : (w + 1);

            const bool has_prev = prev_w >= 0 && words_t1[prev_w] > words_t0[prev_w];
            const bool has_next = next_w >= 0 && words_t1[next_w] > words_t0[next_w];

            bool choose_left = true;
            if (has_prev && has_next) {
                const int64_t d_prev = std::llabs(t0 - words_t1[prev_w]);
                const int64_t d_next = std::llabs(words_t0[next_w] - t1);
                choose_left = d_prev <= d_next;
            } else if (!has_prev && has_next) {
                choose_left = false;
            }

            if (choose_left) {
                new_t0 = t0;
                new_t1 = t0 + min_word_dur_cs;
                anchor_left = true;
            } else {
                new_t0 = t1 - min_word_dur_cs;
                new_t1 = t1;
                anchor_right = true;
            }
        }

        if (new_t1 - new_t0 < min_word_dur_cs) {
            if (anchor_left && !anchor_right) {
                new_t1 = std::min<int64_t>(t1, new_t0 + min_word_dur_cs);
            } else if (anchor_right && !anchor_left) {
                new_t0 = std::max<int64_t>(t0, new_t1 - min_word_dur_cs);
            } else {
                const int64_t centered = (new_t0 + new_t1 - min_word_dur_cs) / 2;
                new_t0 = std::max<int64_t>(t0, std::min<int64_t>(centered, t1 - min_word_dur_cs));
                new_t1 = std::min<int64_t>(t1, new_t0 + min_word_dur_cs);
            }
        }

        if (new_t1 <= new_t0) {
            continue;
        }

        words_t0[w] = new_t0;
        words_t1[w] = new_t1;
    }
}

std::vector<int32_t> whisper_stable_get_gap_tokens(struct whisper_context * ctx) {
    static const char * k_gap_text = " ...";

    std::vector<int32_t> result;
    if (!ctx) {
        return result;
    }

    std::vector<whisper_token> gap_tokens(16);
    int n_written = whisper_tokenize(ctx, k_gap_text, gap_tokens.data(), (int) gap_tokens.size());
    if (n_written < 0) {
        gap_tokens.resize(-n_written);
        n_written = whisper_tokenize(ctx, k_gap_text, gap_tokens.data(), (int) gap_tokens.size());
    }
    if (n_written <= 0) {
        return result;
    }

    result.reserve(n_written);
    for (int i = 0; i < n_written; ++i) {
        result.push_back(gap_tokens[i]);
    }

    return result;
}

void whisper_stable_select_heads(
        float * data,
        int     n_tokens,
        int     n_audio,
        int     n_heads,
        int     top_k) {
    if (!data || n_tokens <= 1 || n_audio <= 0 || n_heads <= 0) {
        return;
    }

    top_k = std::max(1, std::min(top_k, n_heads));
    if (top_k >= n_heads) {
        return;
    }

    struct head_score {
        int head;
        float score;
    };

    std::vector<head_score> scores;
    scores.reserve(n_heads);

    const double mean_x = 0.5 * (n_tokens - 1);
    double var_x = 0.0;
    for (int t = 0; t < n_tokens; ++t) {
        const double dx = t - mean_x;
        var_x += dx * dx;
    }
    if (var_x <= 0.0) {
        return;
    }

    const int head_stride  = n_audio * n_tokens;
    const int audio_stride = n_tokens;

    for (int h = 0; h < n_heads; ++h) {
        const float * head_data = data + h * head_stride;

        std::vector<int> peaks(n_tokens, 0);
        for (int t = 0; t < n_tokens; ++t) {
            float best = -std::numeric_limits<float>::infinity();
            int best_a = 0;
            for (int a = 0; a < n_audio; ++a) {
                const float v = head_data[a * audio_stride + t];
                if (v > best) {
                    best = v;
                    best_a = a;
                }
            }
            peaks[t] = best_a;
        }

        double mean_y = 0.0;
        for (int t = 0; t < n_tokens; ++t) {
            mean_y += peaks[t];
        }
        mean_y /= n_tokens;

        double cov = 0.0;
        double var_y = 0.0;
        for (int t = 0; t < n_tokens; ++t) {
            const double dx = t - mean_x;
            const double dy = peaks[t] - mean_y;
            cov += dx * dy;
            var_y += dy * dy;
        }

        float corr = -1.0f;
        if (var_y > 0.0) {
            corr = (float) (cov / std::sqrt(var_x * var_y));
        }

        scores.push_back({h, corr});
    }

    std::sort(scores.begin(), scores.end(), [](const head_score & a, const head_score & b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.head < b.head;
    });

    std::vector<uint8_t> keep(n_heads, 0);
    for (int i = 0; i < top_k; ++i) {
        keep[scores[i].head] = 1;
    }

    for (int h = 0; h < n_heads; ++h) {
        if (keep[h]) {
            continue;
        }
        float * head_data = data + h * head_stride;
        std::fill(head_data, head_data + head_stride, 0.0f);
    }
}

static int64_t whisper_stable_map_processed_to_original_time(
        int64_t processed_time,
        const std::vector<std::pair<int64_t, int64_t>> & mapping_table) {
    if (mapping_table.empty()) {
        return processed_time;
    }

    if (processed_time <= mapping_table.front().first) {
        return mapping_table.front().second;
    }

    if (processed_time >= mapping_table.back().first) {
        return mapping_table.back().second;
    }

    auto upper = std::lower_bound(
        mapping_table.begin(), mapping_table.end(), processed_time,
        [](const std::pair<int64_t, int64_t> & entry, int64_t t) {
            return entry.first < t;
        });

    if (upper != mapping_table.end() && upper->first == processed_time) {
        return upper->second;
    }

    if (upper == mapping_table.begin() || upper == mapping_table.end()) {
        return processed_time;
    }

    auto lower = upper - 1;

    const int64_t processed_diff = upper->first - lower->first;
    const int64_t original_diff  = upper->second - lower->second;
    const int64_t offset         = processed_time - lower->first;

    if (processed_diff == 0) {
        return lower->second;
    }

    return lower->second + (offset * original_diff) / processed_diff;
}

static bool whisper_stable_is_time_in_silence(
        int64_t t_cs,
        const std::vector<std::pair<int64_t, int64_t>> & silence_regions_cs) {
    if (silence_regions_cs.empty()) {
        return false;
    }

    auto it = std::lower_bound(
        silence_regions_cs.begin(), silence_regions_cs.end(), t_cs,
        [](const std::pair<int64_t, int64_t> & region, int64_t t) {
            return region.second <= t;
        });

    if (it == silence_regions_cs.end()) {
        return false;
    }

    return it->first <= t_cs && t_cs < it->second;
}

std::vector<uint8_t> whisper_stable_build_timestamp_silence_mask(
        const std::vector<std::pair<int64_t, int64_t>> & silence_regions_cs,
        const std::vector<std::pair<int64_t, int64_t>> & mapping_processed_to_original,
        int64_t total_processed_cs,
        int64_t token_step_cs) {
    std::vector<uint8_t> mask;

    if (token_step_cs <= 0 || total_processed_cs <= 0 || silence_regions_cs.empty()) {
        return mask;
    }

    const int64_t n_bins = total_processed_cs / token_step_cs + 2;
    if (n_bins <= 0) {
        return mask;
    }

    mask.assign((size_t) n_bins, 0);

    for (int64_t i = 0; i < n_bins; ++i) {
        const int64_t t_processed_cs = i * token_step_cs;
        const int64_t t_original_cs = whisper_stable_map_processed_to_original_time(
            t_processed_cs, mapping_processed_to_original);

        if (whisper_stable_is_time_in_silence(t_original_cs, silence_regions_cs)) {
            mask[(size_t) i] = 1;
        }
    }

    return mask;
}

bool whisper_stable_setup_filter(
        struct whisper_full_params & params,
        const std::vector<std::pair<int64_t, int64_t>> & silence_regions_cs,
        const std::vector<std::pair<int64_t, int64_t>> & mapping_processed_to_original,
        int64_t total_processed_cs,
        struct whisper_stable_ts_filter_ctx * filter_ctx) {
    if (!filter_ctx) {
        return false;
    }

    filter_ctx->timestamp_silence_mask.clear();
    filter_ctx->seek_cs = 0;
    filter_ctx->token_step_cs = 2;
    filter_ctx->wrapped_callback = params.logits_filter_callback;
    filter_ctx->wrapped_user_data = params.logits_filter_callback_user_data;

    if (silence_regions_cs.empty()) {
        return false;
    }

    filter_ctx->timestamp_silence_mask = whisper_stable_build_timestamp_silence_mask(
        silence_regions_cs,
        mapping_processed_to_original,
        total_processed_cs,
        filter_ctx->token_step_cs);

    if (filter_ctx->timestamp_silence_mask.empty()) {
        return false;
    }

    params.logits_filter_callback = whisper_stable_logits_filter_callback;
    params.logits_filter_callback_user_data = filter_ctx;
    return true;
}

bool whisper_stable_prepare(
        struct whisper_full_params & params,
        const float * vad_probs,
        int n_vad_probs,
        int vad_n_window,
        int sample_rate,
        int64_t total_processed_cs,
        float vad_threshold,
        int64_t min_silence_dur_cs,
        bool has_vad_mapping,
        const std::vector<vad_time_mapping> & mapping_table,
        struct whisper_stable_runtime * runtime) {
    if (!runtime) {
        return false;
    }

    *runtime = whisper_stable_runtime {};

    runtime->silence_regions_cs = whisper_stable_build_silence_map(
        vad_probs,
        n_vad_probs,
        vad_n_window,
        sample_rate,
        vad_threshold,
        min_silence_dur_cs);

    if (runtime->silence_regions_cs.empty()) {
        return false;
    }

    if (has_vad_mapping && !mapping_table.empty()) {
        runtime->mapping_processed_to_original = whisper_stable_copy_vad_mapping(mapping_table);
        runtime->has_vad_mapping = !runtime->mapping_processed_to_original.empty();
    }

    runtime->filter_installed = whisper_stable_setup_filter(
        params,
        runtime->silence_regions_cs,
        runtime->mapping_processed_to_original,
        total_processed_cs,
        &runtime->filter_ctx);

    runtime->active = true;
    return true;
}

bool whisper_stable_prepare_from_ctx(
        struct whisper_full_params & params,
        struct whisper_vad_context * vctx,
        int n_samples,
        bool has_vad_mapping,
        const std::vector<vad_time_mapping> & mapping_table,
        struct whisper_stable_runtime * runtime) {
    if (!vctx || !runtime || n_samples <= 0) {
        if (runtime) {
            *runtime = whisper_stable_runtime {};
        }
        return false;
    }

    const int64_t total_processed_cs = (int64_t) ((double) n_samples * 100.0 / WHISPER_SAMPLE_RATE + 0.5);

    return whisper_stable_prepare(
        params,
        whisper_vad_probs(vctx),
        whisper_vad_n_probs(vctx),
        whisper_stable_vad_n_window(vctx),
        WHISPER_SAMPLE_RATE,
        total_processed_cs,
        params.vad_params.threshold,
        /*min_silence_dur_cs=*/10,
        has_vad_mapping,
        mapping_table,
        runtime);
}

void whisper_stable_snap_segments(
        struct whisper_context * ctx,
        std::vector<whisper_segment> & result_all,
        const std::vector<std::pair<int64_t, int64_t>> & silence_regions_cs,
        const std::vector<std::pair<int64_t, int64_t>> & mapping_processed_to_original,
        bool has_vad_mapping,
        int64_t min_word_dur_cs,
        int64_t min_snap_silence_dur_cs) {
    if (!ctx || result_all.empty() || silence_regions_cs.empty()) {
        return;
    }

    struct word_ref {
        int64_t * t0 = nullptr;
        int64_t * t1 = nullptr;
    };

    std::vector<word_ref> words;
    std::vector<int> seg_first_word;
    std::vector<int> seg_word_count;
    words.reserve(result_all.size() * 4);
    seg_first_word.reserve(result_all.size());
    seg_word_count.reserve(result_all.size());

    const int token_eot = whisper_token_eot(ctx);
    int word_idx = 0;
    for (auto & segment : result_all) {
        seg_first_word.push_back(word_idx);

        int seg_words = 0;
        for (auto & tok : segment.tokens) {
            if (tok.id >= token_eot) {
                continue;
            }

            words.push_back({&tok.t0, &tok.t1});
            ++seg_words;
            ++word_idx;
        }

        seg_word_count.push_back(seg_words);
    }

    const int n_words = (int) words.size();
    const int n_segments = (int) result_all.size();
    if (n_words <= 0 || n_segments <= 0) {
        return;
    }

    std::vector<int64_t> words_t0(n_words, 0);
    std::vector<int64_t> words_t1(n_words, 0);
    for (int i = 0; i < n_words; ++i) {
        auto & w = words[i];
        if (!w.t0 || !w.t1) {
            continue;
        }

        int64_t t0 = *w.t0;
        int64_t t1 = *w.t1;
        if (has_vad_mapping && !mapping_processed_to_original.empty()) {
            t0 = whisper_stable_map_processed_to_original_time(t0, mapping_processed_to_original);
            t1 = whisper_stable_map_processed_to_original_time(t1, mapping_processed_to_original);
        }

        words_t0[i] = t0;
        words_t1[i] = t1;
    }

    whisper_stable_snap_timestamps(
        words_t0.data(),
        words_t1.data(),
        n_words,
        seg_first_word.data(),
        seg_word_count.data(),
        n_segments,
        silence_regions_cs,
        min_word_dur_cs,
        min_snap_silence_dur_cs);

    for (int i = 0; i < n_words; ++i) {
        auto & w = words[i];
        if (!w.t0 || !w.t1) {
            continue;
        }

        *w.t0 = words_t0[i];
        *w.t1 = words_t1[i];
    }

    for (int s = 0; s < n_segments; ++s) {
        const int first = seg_first_word[s];
        const int count = seg_word_count[s];
        if (count <= 0 || first < 0 || first >= n_words) {
            continue;
        }

        int first_valid = -1;
        int last_valid = -1;
        for (int j = 0; j < count; ++j) {
            const int wi = first + j;
            if (wi < 0 || wi >= n_words) {
                continue;
            }
            if (words_t1[wi] <= words_t0[wi]) {
                continue;
            }

            if (first_valid < 0) {
                first_valid = wi;
            }
            last_valid = wi;
        }

        if (first_valid < 0 || last_valid < 0) {
            continue;
        }

        int64_t seg_t0 = words_t0[first_valid];
        int64_t seg_t1 = words_t1[last_valid];
        if (seg_t1 < seg_t0) {
            seg_t1 = seg_t0;
        }

        auto & seg = result_all[s];
        seg.t0 = seg_t0;
        seg.t1 = seg_t1;
    }
}

void whisper_stable_finalize(
        struct whisper_context * ctx,
        std::vector<whisper_segment> & result_all,
        std::vector<vad_time_mapping> & vad_mapping_table,
        bool & has_vad_segments,
        const struct whisper_stable_runtime & runtime,
        int64_t min_word_dur_cs,
        int64_t min_snap_silence_dur_cs) {
    if (!runtime.active) {
        return;
    }

    whisper_stable_snap_segments(
        ctx,
        result_all,
        runtime.silence_regions_cs,
        runtime.mapping_processed_to_original,
        runtime.has_vad_mapping,
        min_word_dur_cs,
        min_snap_silence_dur_cs);

    // Mapping has already been applied during stable word-level snapping.
    vad_mapping_table.clear();
    has_vad_segments = false;
}

void whisper_stable_set_filter_seek(void * user_data, int64_t seek_cs) {
    if (!user_data) {
        return;
    }

    auto * stable = reinterpret_cast<whisper_stable_ts_filter_ctx *>(user_data);
    stable->seek_cs = seek_cs;
}

void whisper_stable_logits_filter_callback(
        struct whisper_context * ctx,
          struct whisper_state * state,
      const whisper_token_data * tokens,
                           int   n_tokens,
                         float * logits,
                          void * user_data) {
    auto * stable = reinterpret_cast<whisper_stable_ts_filter_ctx *>(user_data);
    if (!stable || !ctx || !logits) {
        return;
    }

    if (stable->wrapped_callback) {
        stable->wrapped_callback(ctx, state, tokens, n_tokens, logits, stable->wrapped_user_data);
    }

    if (stable->timestamp_silence_mask.empty() || stable->token_step_cs <= 0) {
        return;
    }

    const int token_beg = whisper_token_beg(ctx);
    const int n_vocab   = whisper_n_vocab(ctx);
    if (token_beg < 0 || token_beg >= n_vocab) {
        return;
    }

    for (int id = token_beg; id < n_vocab; ++id) {
        const int64_t rel_token_idx = id - token_beg;
        const int64_t abs_cs = stable->seek_cs + rel_token_idx * stable->token_step_cs;
        if (abs_cs < 0) {
            continue;
        }

        const int64_t abs_bin = abs_cs / stable->token_step_cs;
        if (abs_bin < 0 || (size_t) abs_bin >= stable->timestamp_silence_mask.size()) {
            continue;
        }

        if (stable->timestamp_silence_mask[(size_t) abs_bin]) {
            logits[id] = -INFINITY;
        }
    }
}
