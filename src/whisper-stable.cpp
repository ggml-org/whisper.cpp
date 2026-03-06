#include "whisper-stable.h"
#include "whisper.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ---------------------------------------------------------------------------
// Silence map from VAD probabilities
// ---------------------------------------------------------------------------

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

    if (region_start >= 0) {
        const int64_t region_end = (int64_t)(n_probs * cs_per_frame);
        if (region_end - region_start >= min_silence_dur_cs) {
            silence.push_back({region_start, region_end});
        }
    }

    return silence;
}

// ---------------------------------------------------------------------------
// Silence map from raw PCM energy — mirrors stable-ts wav2mask
// ---------------------------------------------------------------------------

std::vector<std::pair<int64_t, int64_t>> whisper_stable_build_silence_map_from_pcm(
        const float * pcm,
        int           n_samples,
        int           sample_rate,
        int64_t       min_silence_dur_cs) {

    std::vector<std::pair<int64_t, int64_t>> silence;
    if (!pcm || n_samples <= 0 || sample_rate <= 0) {
        return silence;
    }

    // Audio token size matches Whisper's resolution: 320 samples @ 16kHz = 20ms
    const int samples_per_token = 320;
    const int n_tokens = (int)std::round((double)n_samples / samples_per_token) + 1;
    if (n_tokens < 2) {
        return silence;
    }

    // Step 1+2: abs amplitude, find 99.9th percentile (top 0.1% of samples)
    const int k = std::max(1, n_samples / 1000);
    std::vector<float> abs_vals(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        abs_vals[i] = std::fabs(pcm[i]);
    }
    std::nth_element(abs_vals.begin(), abs_vals.begin() + (n_samples - k), abs_vals.end());
    float threshold = abs_vals[n_samples - k];
    if (threshold < 1e-5f) {
        // Entirely silent audio — everything is silence
        const double cs_total = (double)n_samples * 100.0 / sample_rate;
        if ((int64_t)cs_total >= min_silence_dur_cs) {
            silence.push_back({0, (int64_t)cs_total});
        }
        return silence;
    }

    // Step 3: average abs amplitude per token window
    std::vector<float> token_energy(n_tokens, 0.0f);
    for (int t = 0; t < n_tokens; ++t) {
        const int s0 = t * samples_per_token;
        const int s1 = std::min(s0 + samples_per_token, n_samples);
        if (s0 >= n_samples) break;
        float sum = 0.0f;
        for (int s = s0; s < s1; ++s) {
            sum += std::fabs(pcm[s]);
        }
        token_energy[t] = sum / (float)(s1 - s0);
    }

    // Normalize: divide by min(1.0, threshold * 1.75), clamp to [0, 1]
    const float norm_denom = std::min(1.0f, threshold * 1.75f);
    for (auto & v : token_energy) {
        v = std::min(1.0f, v / norm_denom);
    }

    // Step 4: avg-pool with kernel=5, reflection padding
    std::vector<float> smoothed(n_tokens, 0.0f);
    const int k_half = 2; // kernel 5
    for (int t = 0; t < n_tokens; ++t) {
        float sum = 0.0f;
        for (int d = -k_half; d <= k_half; ++d) {
            int idx = t + d;
            if (idx < 0) idx = -idx;
            if (idx >= n_tokens) idx = 2 * n_tokens - 2 - idx;
            idx = std::max(0, std::min(n_tokens - 1, idx));
            sum += token_energy[idx];
        }
        smoothed[t] = sum / 5.0f;
    }

    // Step 5: quantize to 20 levels — anything rounding to 0 is silent
    // Step 6: merge adjacent silent tokens into regions, filter by min duration
    const double cs_per_token = (double)samples_per_token * 100.0 / sample_rate;
    int64_t region_start = -1;

    for (int t = 0; t < n_tokens; ++t) {
        const bool is_silent = std::roundf(smoothed[t] * 20.0f) == 0.0f;

        if (is_silent && region_start < 0) {
            region_start = (int64_t)(t * cs_per_token);
        } else if (!is_silent && region_start >= 0) {
            const int64_t region_end = (int64_t)(t * cs_per_token);
            if (region_end - region_start >= min_silence_dur_cs) {
                silence.push_back({region_start, region_end});
            }
            region_start = -1;
        }
    }

    if (region_start >= 0) {
        const int64_t region_end = (int64_t)(n_tokens * cs_per_token);
        if (region_end - region_start >= min_silence_dur_cs) {
            silence.push_back({region_start, region_end});
        }
    }

    return silence;
}

// ---------------------------------------------------------------------------
// Silence overlap utility
// ---------------------------------------------------------------------------

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
        [](const std::pair<int64_t, int64_t> & r, int64_t t) {
            return r.second <= t;
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

// ---------------------------------------------------------------------------
// Timestamp snapping — stable-ts boundary-moving algorithm
// ---------------------------------------------------------------------------

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

    if (n_words <= 0 || silence.empty()) {
        return;
    }

    std::vector<bool> is_first(n_words, false);
    std::vector<bool> is_last(n_words, false);
    for (int s = 0; s < n_segments; ++s) {
        const int first = seg_first_word[s];
        const int count = seg_word_count[s];
        if (count > 0) {
            is_first[first] = true;
            is_last[first + count - 1] = true;
        }
    }

    for (int w = 0; w < n_words; ++w) {
        const int64_t t0 = words_t0[w];
        const int64_t t1 = words_t1[w];
        if (t0 >= t1) {
            continue;
        }

        int64_t new_t0    = t0;
        int64_t new_t1    = t1;
        bool moved_start  = false;
        bool moved_end    = false;

        auto it = std::lower_bound(
            silence.begin(), silence.end(), new_t0,
            [](const std::pair<int64_t, int64_t> & r, int64_t t) {
                return r.second <= t;
            });

        for (; it != silence.end() && it->first < new_t1; ++it) {
            if (it->second - it->first < min_snap_silence_dur_cs) {
                continue;
            }

            const int64_t si_s = it->first;
            const int64_t si_e = it->second;

            if (si_s <= new_t0 && si_e > new_t0 && si_e <= new_t1) {
                // Start is inside silence → move start forward
                new_t0 = si_e;
                moved_start = true;
            } else if (si_s >= new_t0 && si_s < new_t1 && si_e >= new_t1) {
                // End is inside silence → move end backward
                new_t1 = si_s;
                moved_end = true;
                break;
            } else if (si_s > new_t0 && si_e < new_t1) {
                // Silence fully inside word → snap boundary with less overshoot
                const int64_t sil_len    = si_e - si_s;
                const double left_ratio  = (double)(si_s - new_t0) / sil_len;
                const double right_ratio = (double)(new_t1 - si_e) / sil_len;

                bool snap_start;
                if (is_first[w]) {
                    snap_start = true;
                } else if (is_last[w]) {
                    snap_start = false;
                } else {
                    snap_start = (left_ratio >= right_ratio);
                }

                if (snap_start) {
                    new_t0 = si_e;
                    moved_start = true;
                } else {
                    new_t1 = si_s;
                    moved_end = true;
                    break;
                }
            }
        }

        // Enforce minimum word duration
        if (new_t1 - new_t0 < min_word_dur_cs) {
            if (moved_start && !moved_end) {
                new_t1 = std::min(t1, new_t0 + min_word_dur_cs);
                if (new_t1 - new_t0 < min_word_dur_cs) {
                    new_t0 = std::max<int64_t>(0, new_t1 - min_word_dur_cs);
                }
            } else if (moved_end && !moved_start) {
                new_t0 = std::max(t0, new_t1 - min_word_dur_cs);
                if (new_t1 - new_t0 < min_word_dur_cs) {
                    new_t1 = new_t0 + min_word_dur_cs;
                }
            } else {
                const int64_t span = t1 - t0;
                if (span >= min_word_dur_cs) {
                    const int64_t mid = (t0 + t1) / 2;
                    new_t0 = mid - min_word_dur_cs / 2;
                    new_t1 = new_t0 + min_word_dur_cs;
                } else {
                    new_t0 = t0;
                    new_t1 = t1;
                }
            }
        }

        if (new_t1 <= new_t0) {
            continue;
        }

        words_t0[w] = new_t0;
        words_t1[w] = new_t1;
    }
}

// ---------------------------------------------------------------------------
// Segment-level snapping
// ---------------------------------------------------------------------------

void whisper_stable_snap_segments(
        struct whisper_context * ctx,
        std::vector<whisper_segment> & result_all,
        const std::vector<std::pair<int64_t, int64_t>> & silence_regions_cs,
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
    words.reserve(result_all.size() * 8);
    seg_first_word.reserve(result_all.size());
    seg_word_count.reserve(result_all.size());

    const int token_eot = whisper_token_eot(ctx);
    int word_idx = 0;

    for (auto & seg : result_all) {
        seg_first_word.push_back(word_idx);
        int count = 0;
        for (auto & tok : seg.tokens) {
            if (tok.id >= token_eot) {
                continue;
            }
            words.push_back({&tok.t0, &tok.t1});
            ++count;
            ++word_idx;
        }
        seg_word_count.push_back(count);
    }

    const int n_words = (int)words.size();
    const int n_segs  = (int)result_all.size();
    if (n_words <= 0) {
        return;
    }

    // Token timestamps are already in original timeline (offset applied by per-segment VAD decode)
    std::vector<int64_t> t0_arr(n_words);
    std::vector<int64_t> t1_arr(n_words);
    for (int i = 0; i < n_words; ++i) {
        t0_arr[i] = words[i].t0 ? *words[i].t0 : 0;
        t1_arr[i] = words[i].t1 ? *words[i].t1 : 0;
    }

    whisper_stable_snap_timestamps(
        t0_arr.data(), t1_arr.data(), n_words,
        seg_first_word.data(), seg_word_count.data(), n_segs,
        silence_regions_cs, min_word_dur_cs, min_snap_silence_dur_cs);

    // Write back — values are now on original timeline
    for (int i = 0; i < n_words; ++i) {
        if (words[i].t0) *words[i].t0 = t0_arr[i];
        if (words[i].t1) *words[i].t1 = t1_arr[i];
    }

    // Update segment t0/t1 from first/last valid word
    for (int s = 0; s < n_segs; ++s) {
        const int first = seg_first_word[s];
        const int count = seg_word_count[s];
        if (count <= 0) {
            continue;
        }

        int64_t seg_t0 = std::numeric_limits<int64_t>::max();
        int64_t seg_t1 = std::numeric_limits<int64_t>::min();

        for (int j = 0; j < count; ++j) {
            const int wi = first + j;
            if (t1_arr[wi] <= t0_arr[wi]) {
                continue;
            }
            seg_t0 = std::min(seg_t0, t0_arr[wi]);
            seg_t1 = std::max(seg_t1, t1_arr[wi]);
        }

        if (seg_t0 != std::numeric_limits<int64_t>::max() &&
            seg_t1 != std::numeric_limits<int64_t>::min()) {
            result_all[s].t0 = seg_t0;
            result_all[s].t1 = seg_t1;
        }
    }
}

// ---------------------------------------------------------------------------
// DTW gap padding tokens
// ---------------------------------------------------------------------------

std::vector<int32_t> whisper_stable_get_gap_tokens(struct whisper_context * ctx) {
    static const char * k_gap_text = " ...";

    std::vector<int32_t> result;
    if (!ctx) {
        return result;
    }

    std::vector<whisper_token> gap_tokens(16);
    int n_written = whisper_tokenize(ctx, k_gap_text, gap_tokens.data(), (int)gap_tokens.size());
    if (n_written < 0) {
        gap_tokens.resize(-n_written);
        n_written = whisper_tokenize(ctx, k_gap_text, gap_tokens.data(), (int)gap_tokens.size());
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

// ---------------------------------------------------------------------------
// Dynamic head selection — score heads by monotonicity, keep top-k
// ---------------------------------------------------------------------------

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

    struct head_score { int head; float score; };

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
                if (v > best) { best = v; best_a = a; }
            }
            peaks[t] = best_a;
        }

        double mean_y = 0.0;
        for (int t = 0; t < n_tokens; ++t) mean_y += peaks[t];
        mean_y /= n_tokens;

        double cov = 0.0, var_y = 0.0;
        for (int t = 0; t < n_tokens; ++t) {
            const double dx = t - mean_x;
            const double dy = peaks[t] - mean_y;
            cov   += dx * dy;
            var_y += dy * dy;
        }

        float corr = -1.0f;
        if (var_y > 0.0) {
            corr = (float)(cov / std::sqrt(var_x * var_y));
        }
        scores.push_back({h, corr});
    }

    std::sort(scores.begin(), scores.end(), [](const head_score & a, const head_score & b) {
        return a.score != b.score ? a.score > b.score : a.head < b.head;
    });

    std::vector<uint8_t> keep(n_heads, 0);
    for (int i = 0; i < top_k; ++i) keep[scores[i].head] = 1;

    for (int h = 0; h < n_heads; ++h) {
        if (keep[h]) continue;
        float * head_data = data + h * head_stride;
        std::fill(head_data, head_data + head_stride, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Constrained decoding filter
// ---------------------------------------------------------------------------

bool whisper_stable_setup_filter(
        struct whisper_full_params & params,
        const std::vector<std::pair<int64_t, int64_t>> & silence_regions_cs,
        int64_t total_audio_cs,
        struct whisper_stable_ts_filter_ctx * filter_ctx) {

    if (!filter_ctx) return false;

    filter_ctx->timestamp_silence_mask.clear();
    filter_ctx->seek_cs        = 0;
    filter_ctx->token_step_cs  = 2;
    filter_ctx->wrapped_callback  = params.logits_filter_callback;
    filter_ctx->wrapped_user_data = params.logits_filter_callback_user_data;

    if (silence_regions_cs.empty() || total_audio_cs <= 0) return false;

    const int64_t step   = filter_ctx->token_step_cs;
    const int64_t n_bins = total_audio_cs / step + 2;
    filter_ctx->timestamp_silence_mask.assign((size_t)n_bins, 0);

    for (const auto & r : silence_regions_cs) {
        const int64_t bin_start = r.first / step;
        const int64_t bin_end   = (r.second + step - 1) / step;
        for (int64_t b = bin_start; b < bin_end && b < n_bins; ++b) {
            filter_ctx->timestamp_silence_mask[(size_t)b] = 1;
        }
    }

    params.logits_filter_callback           = whisper_stable_logits_filter_callback;
    params.logits_filter_callback_user_data  = filter_ctx;
    return true;
}

void whisper_stable_set_filter_seek(void * user_data, int64_t seek_cs) {
    if (!user_data) return;
    reinterpret_cast<whisper_stable_ts_filter_ctx *>(user_data)->seek_cs = seek_cs;
}

void whisper_stable_logits_filter_callback(
        struct whisper_context * ctx,
          struct whisper_state * /*state*/,
      const whisper_token_data * /*tokens*/,
                           int   /*n_tokens*/,
                         float * logits,
                          void * user_data) {

    auto * stable = reinterpret_cast<whisper_stable_ts_filter_ctx *>(user_data);
    if (!stable || !ctx || !logits) return;

    if (stable->wrapped_callback) {
        stable->wrapped_callback(ctx, nullptr, nullptr, 0, logits, stable->wrapped_user_data);
    }

    if (stable->timestamp_silence_mask.empty() || stable->token_step_cs <= 0) return;

    const int token_beg = whisper_token_beg(ctx);
    const int n_vocab   = whisper_n_vocab(ctx);
    if (token_beg < 0 || token_beg >= n_vocab) return;

    for (int id = token_beg; id < n_vocab; ++id) {
        const int64_t rel_idx = id - token_beg;
        const int64_t abs_cs  = stable->seek_cs + rel_idx * stable->token_step_cs;
        if (abs_cs < 0) continue;

        const int64_t bin = abs_cs / stable->token_step_cs;
        if (bin < 0 || (size_t)bin >= stable->timestamp_silence_mask.size()) continue;

        if (stable->timestamp_silence_mask[(size_t)bin]) {
            logits[id] = -INFINITY;
        }
    }
}
