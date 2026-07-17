#include "lm_pipeline.h"

#include "metadata_fsm.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Ported from acestep.cpp: sampling.h (sample_top_k_p), prompt.h (CoT/prompt
// building) and pipeline-lm.cpp (run_phase2_batch + generate_phase1_batch,
// single-sequence subset).

namespace tts_cpp::acestep {

// LM system instructions (task-types.h).
static const char * LM_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:";
static const char * LM_INSPIRE_INSTRUCTION =
    "Expand the user's input into a more detailed and specific musical description:";

struct TokenProb {
    int   id;
    float prob;
};

int sample_top_k_p(float * logits, int V, float temperature, float top_p, int top_k, std::mt19937 & rng) {
    if (temperature <= 0.0f) {
        return (int) (std::max_element(logits, logits + V) - logits);
    }

    static thread_local std::vector<float>     tmp_buf;
    static thread_local std::vector<TokenProb> sorted_buf;

    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < V; i++) logits[i] *= inv_temp;

    if (top_k > 0 && top_k < V) {
        tmp_buf.resize(V);
        memcpy(tmp_buf.data(), logits, V * sizeof(float));
        std::nth_element(tmp_buf.begin(), tmp_buf.begin() + (top_k - 1), tmp_buf.end(), std::greater<float>());
        float threshold = tmp_buf[top_k - 1];
        for (int i = 0; i < V; i++) {
            if (logits[i] < threshold) logits[i] = -INFINITY;
        }
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        float max_logit = -INFINITY;
        for (int i = 0; i < V; i++) {
            if (logits[i] > max_logit) max_logit = logits[i];
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < V; i++) sum_exp += expf(logits[i] - max_logit);
        float inv_sum = 1.0f / sum_exp;

        float cutoff = max_logit - 16.0f;
        sorted_buf.clear();
        for (int i = 0; i < V; i++) {
            if (logits[i] >= cutoff) {
                float prob = expf(logits[i] - max_logit) * inv_sum;
                sorted_buf.push_back({ i, prob });
            } else {
                logits[i] = -INFINITY;
            }
        }

        int K = (int) sorted_buf.size();
        if (K > 0) {
            std::sort(sorted_buf.begin(), sorted_buf.end(),
                      [](const TokenProb & a, const TokenProb & b) { return a.prob > b.prob; });
            float cum = 0.0f;
            for (int i = 0; i < K; i++) {
                if (i > 0 && cum >= top_p) logits[sorted_buf[i].id] = -INFINITY;
                cum += sorted_buf[i].prob;
            }
        }
    }

    float max_val = -INFINITY;
    for (int i = 0; i < V; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }

    std::uniform_real_distribution<float> dist(0.0f, sum);
    float                                 r   = dist(rng);
    float                                 acc = 0.0f;
    for (int i = 0; i < V; i++) {
        acc += logits[i];
        if (acc >= r) return i;
    }
    return 0;
}

// YAML CoT block (acestep.cpp prompt.h build_cot_yaml). Matches Python
// yaml.dump(allow_unicode=True, sort_keys=True): break + 2-space indent past col 80.
static std::string build_cot_yaml(const AcePrompt & prompt) {
    auto yaml_wrap = [](const std::string & key, const std::string & val) -> std::string {
        std::string result = key + ":";
        int         col    = (int) (key.size() + 1);
        size_t      i      = 0;
        while (i < val.size()) {
            size_t end = val.find(' ', i);
            if (end == std::string::npos) end = val.size();
            std::string word = val.substr(i, end - i);
            if (col > 80) {
                result += "\n  ";
                col = 2;
            } else {
                result += " ";
                col += 1;
            }
            result += word;
            col += (int) word.size();
            i = (end < val.size()) ? end + 1 : val.size();
        }
        result += "\n";
        return result;
    };

    std::string yaml;
    if (prompt.bpm > 0) yaml += "bpm: " + std::to_string(prompt.bpm) + "\n";
    if (!prompt.caption.empty()) yaml += yaml_wrap("caption", prompt.caption);
    if (prompt.duration > 0) yaml += "duration: " + std::to_string((int) prompt.duration) + "\n";
    if (!prompt.keyscale.empty()) yaml += "keyscale: " + prompt.keyscale + "\n";
    if (!prompt.vocal_language.empty()) yaml += "language: " + prompt.vocal_language + "\n";
    if (!prompt.timesignature.empty()) yaml += "timesignature: " + prompt.timesignature + "\n";
    return yaml;
}

std::vector<int> build_lm_prompt_with_cot(const BpeTokenizer & bpe, const AcePrompt & prompt) {
    std::string      cot = build_cot_yaml(prompt);
    std::vector<int> ids;
    auto             append = [&](const std::string & text) {
        auto t = bpe_encode(bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append(std::string("system\n# Instruction\n") + LM_INSTRUCTION + "\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("user\n# Caption\n" + prompt.caption + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    ids.push_back(TOKEN_THINK);
    append("\n" + cot);
    ids.push_back(TOKEN_THINK_END);
    append("\n\n");
    return ids;
}

// Unconditional prompt for CFG (acestep.cpp build_lm_prompt_uncond_with_cot):
// bare user turn (no Caption/Lyric wrapper), empty CoT. Matches the training
// CFG dropout. negative_prompt is optional.
static std::vector<int> build_lm_prompt_uncond_with_cot(const BpeTokenizer & bpe, const std::string & negative_prompt) {
    std::vector<int> ids;
    auto             append = [&](const std::string & text) {
        auto t = bpe_encode(bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append(std::string("system\n# Instruction\n") + LM_INSTRUCTION + "\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("user\n" + negative_prompt);
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    ids.push_back(TOKEN_THINK);
    append("\n\n");
    ids.push_back(TOKEN_THINK_END);
    append("\n\n");
    return ids;
}

// ---------------------------------------------------------------------------
// Phase 1: metadata / CoT auto-generation (acestep.cpp prompt.h + pipeline-lm)
// ---------------------------------------------------------------------------

// Parse the LM Phase-1 output text into metadata + lyrics (prompt.h
// parse_cot_and_lyrics). Returns true if any of bpm/duration was found.
static bool parse_cot_and_lyrics(const std::string & text, AcePrompt & out) {
    size_t ts = text.find("<think>");
    size_t te = text.find("</think>");

    std::string cot;
    std::string lyrics_after;
    if (ts != std::string::npos && te != std::string::npos) {
        cot          = text.substr(ts + 7, te - ts - 7);
        lyrics_after = text.substr(te + 8);
    } else if (te != std::string::npos) {
        cot          = text.substr(0, te);
        lyrics_after = text.substr(te + 8);
    } else {
        cot = text;
    }

    auto get_field = [&](const std::string & key) -> std::string {
        std::string needle = key + ":";
        size_t      p      = cot.find(needle);
        if (p == std::string::npos) return "";
        p += needle.size();
        while (p < cot.size() && (cot[p] == ' ' || cot[p] == '\'')) p++;
        size_t end = cot.find('\n', p);
        if (end == std::string::npos) end = cot.size();
        std::string val = cot.substr(p, end - p);
        while (!val.empty() && (val.back() == ' ' || val.back() == '\'' || val.back() == '\r')) val.pop_back();
        return val;
    };

    std::string bpm_s = get_field("bpm");
    if (!bpm_s.empty()) out.bpm = atoi(bpm_s.c_str());
    std::string dur_s = get_field("duration");
    if (!dur_s.empty()) out.duration = (float) atof(dur_s.c_str());
    std::string ks = get_field("keyscale");
    if (!ks.empty()) out.keyscale = ks;
    std::string ts_s = get_field("timesignature");
    if (!ts_s.empty()) out.timesignature = ts_s;
    std::string lang = get_field("language");
    if (!lang.empty()) out.vocal_language = lang;

    std::string cap = get_field("caption");
    if (!cap.empty()) {
        size_t cp = cot.find("caption:");
        if (cp != std::string::npos) {
            cp += 8;
            size_t end = cot.find("\nduration:", cp);
            if (end == std::string::npos) end = cot.find("\nkeyscale:", cp);
            if (end == std::string::npos) end = cot.size();
            std::string full_cap = cot.substr(cp, end - cp);
            std::string cleaned;
            bool        in_space = true;
            for (char ch : full_cap) {
                if (ch == '\n' || ch == '\r') ch = ' ';
                if (ch == ' ') {
                    if (!in_space) cleaned += ' ';
                    in_space = true;
                } else {
                    cleaned += ch;
                    in_space = false;
                }
            }
            while (!cleaned.empty() && cleaned.back() == ' ') cleaned.pop_back();
            while (!cleaned.empty() && cleaned.front() == ' ') cleaned.erase(cleaned.begin());
            if (!cleaned.empty()) out.caption = cleaned;
        }
    }

    if (!lyrics_after.empty()) {
        size_t s = lyrics_after.find_first_not_of(" \t\n\r");
        if (s != std::string::npos) lyrics_after = lyrics_after.substr(s);
        size_t lp = lyrics_after.find("# Lyric\n");
        if (lp != std::string::npos && lp < 64) lyrics_after = lyrics_after.substr(lp + 8);
        while (!lyrics_after.empty() &&
               (lyrics_after.back() == ' ' || lyrics_after.back() == '\n' || lyrics_after.back() == '\r'))
            lyrics_after.pop_back();
        if (!lyrics_after.empty()) out.lyrics = lyrics_after;
    }

    return (out.bpm > 0 || out.duration > 0);
}

// Phase-1 gap-fill prompt with lyrics/metadata (prompt.h build_lm_prompt): the
// assistant turn is left OPEN (no <think>) — the FSM drives the CoT YAML.
static std::vector<int> build_lm_prompt_phase1(const BpeTokenizer & bpe, const AcePrompt & prompt) {
    std::vector<int> ids;
    auto             append = [&](const std::string & text) {
        auto t = bpe_encode(bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append(std::string("system\n# Instruction\n") + LM_INSTRUCTION + "\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("user\n# Caption\n" + prompt.caption + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    return ids;
}

// INSPIRE prompt for bare captions (prompt.h build_custom_prompt).
static std::vector<int> build_custom_prompt(const BpeTokenizer & bpe, const std::string & sys,
                                            const std::string & user) {
    std::vector<int> ids;
    auto             append = [&](const std::string & text) {
        auto t = bpe_encode(bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append("system\n" + sys + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("user\n" + user + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    return ids;
}

bool lm_generate_phase1(LMModel * m, const BpeTokenizer & bpe, AcePrompt & prompt, const LmSampleParams & params,
                        bool use_fsm, bool use_cot_caption) {
    const LMConfig & cfg = lm_model_config(m);
    const int        V   = cfg.vocab_size;

    const AcePrompt base        = prompt;
    const bool      need_lyrics = base.lyrics.empty();
    const bool      gen_lyrics  = need_lyrics;
    const bool      lyrics_mode = gen_lyrics;
    const bool      stop_at_reasoning = !gen_lyrics;
    const bool      inspire     = need_lyrics;  // bare caption -> INSPIRE expansion

    std::vector<int> prompt_tokens;
    if (inspire) {
        std::string user_msg = base.caption;
        prompt_tokens        = build_custom_prompt(bpe, std::string("# Instruction\n") + LM_INSPIRE_INSTRUCTION,
                                                    user_msg);
    } else {
        prompt_tokens = build_lm_prompt_phase1(bpe, base);
    }

    // FSM setup: constrain the CoT YAML. Force any user-provided metadata.
    MetadataFSM fsm;
    if (use_fsm) {
        fsm.init(bpe, V);
        fsm.skip_caption = !use_cot_caption && !inspire;
        if (base.bpm > 0)               fsm.force_field(bpe, MetadataFSM::BPM_VALUE, std::to_string(base.bpm));
        if (base.duration > 0)          fsm.force_field(bpe, MetadataFSM::DURATION_VALUE,
                                                        std::to_string((int) base.duration));
        if (!base.keyscale.empty())     fsm.force_field(bpe, MetadataFSM::KEYSCALE_VALUE, base.keyscale);
        if (!base.vocal_language.empty()) fsm.force_field(bpe, MetadataFSM::LANGUAGE_VALUE, base.vocal_language);
        if (!base.timesignature.empty()) fsm.force_field(bpe, MetadataFSM::TIMESIG_VALUE, base.timesignature);
    }

    std::mt19937       rng(params.seed);
    std::vector<int>   gen_tokens;
    std::vector<float> lg;

    lm_reset(m, 0);
    if (!lm_model_forward(m, prompt_tokens.data(), (int) prompt_tokens.size(), lg, 0)) {
        fprintf(stderr, "[lm-phase1] prefill failed\n");
        return false;
    }
    fprintf(stderr, "[lm-phase1] prefill %zu tokens, fsm=%d inspire=%d gen_lyrics=%d\n", prompt_tokens.size(),
            (int) use_fsm, (int) inspire, (int) gen_lyrics);

    bool codes_phase = false;
    int  tok;
    {
        if (use_fsm && fsm.enabled) fsm.apply_mask(lg.data());
        tok = sample_top_k_p(lg.data(), V, params.temperature, params.top_p, params.top_k, rng);
        if (tok != TOKEN_IM_END) {
            if (use_fsm && fsm.enabled) fsm.update(tok);
            if (tok == TOKEN_THINK_END) codes_phase = true;
            gen_tokens.push_back(tok);
        }
    }

    const int max_new = params.max_new_tokens > 0 ? params.max_new_tokens : 2048;
    for (int step = 0; step < max_new && tok != TOKEN_IM_END; step++) {
        if (codes_phase && stop_at_reasoning) break;
        int32_t t = (int32_t) tok;
        if (!lm_model_forward(m, &t, 1, lg, 0)) {
            fprintf(stderr, "[lm-phase1] decode step %d failed\n", step);
            return false;
        }
        float * lc = lg.data();
        if (use_fsm && fsm.enabled && !codes_phase) fsm.apply_mask(lc);
        if (codes_phase && !lyrics_mode) {
            for (int v = TOKEN_IM_END + 1; v < AUDIO_CODE_BASE; v++) lc[v] = -1e9f;
        }

        if (codes_phase && !lyrics_mode) {
            int V_eff = V - TOKEN_IM_END;
            tok       = sample_top_k_p(lc + TOKEN_IM_END, V_eff, params.temperature, params.top_p, params.top_k, rng) +
                  TOKEN_IM_END;
        } else {
            tok = sample_top_k_p(lc, V, params.temperature, params.top_p, params.top_k, rng);
        }

        if (tok == TOKEN_IM_END) break;
        if (use_fsm && fsm.enabled && !codes_phase) fsm.update(tok);
        if (tok == TOKEN_THINK_END && !codes_phase) {
            codes_phase = true;
            gen_tokens.push_back(tok);
            if (stop_at_reasoning) break;
            continue;
        }
        gen_tokens.push_back(tok);
    }

    std::string text = bpe_decode(bpe, gen_tokens);
    fprintf(stderr, "[lm-phase1] generated %zu tokens:\n%s\n", gen_tokens.size(), text.c_str());

    AcePrompt parsed = {};
    parse_cot_and_lyrics(text, parsed);

    // Gap-fill: only overwrite fields the user left empty.
    if (parsed.bpm > 0 && base.bpm <= 0)                              prompt.bpm = parsed.bpm;
    if (parsed.duration > 0 && base.duration <= 0)                    prompt.duration = parsed.duration;
    if (!parsed.keyscale.empty() && base.keyscale.empty())            prompt.keyscale = parsed.keyscale;
    if (!parsed.timesignature.empty() && base.timesignature.empty())  prompt.timesignature = parsed.timesignature;
    if (!parsed.vocal_language.empty() && (base.vocal_language.empty() || base.vocal_language == "unknown"))
        prompt.vocal_language = parsed.vocal_language;
    if (!parsed.caption.empty() && use_cot_caption)                   prompt.caption = parsed.caption;
    if (gen_lyrics && !parsed.lyrics.empty())                         prompt.lyrics = parsed.lyrics;
    if (prompt.duration <= 0)  prompt.duration = 120.0f;
    if (prompt.duration > 600) prompt.duration = 600.0f;

    fprintf(stderr, "[lm-phase1] filled: bpm=%d dur=%.0f key='%s' tsig='%s' lang='%s'\n", prompt.bpm, prompt.duration,
            prompt.keyscale.c_str(), prompt.timesignature.c_str(), prompt.vocal_language.c_str());
    return true;
}

bool lm_generate_codes(LMModel *              m,
                       const BpeTokenizer &   bpe,
                       const AcePrompt &      prompt,
                       const LmSampleParams & params,
                       std::vector<int> &     codes_out) {
    codes_out.clear();

    const LMConfig & cfg = lm_model_config(m);
    const int        V   = cfg.vocab_size;

    const bool use_cfg = params.cfg_scale > 1.0f && lm_num_kv_sets(m) >= 2;

    std::vector<int> tokens = build_lm_prompt_with_cot(bpe, prompt);
    std::vector<int> uncond;
    if (use_cfg) uncond = build_lm_prompt_uncond_with_cot(bpe, /*negative_prompt=*/"");

    int max_tokens = params.max_new_tokens > 0 ? params.max_new_tokens : (int) (prompt.duration * 5) + 100;

    std::mt19937 rng(params.seed);

    std::vector<float> lc, lu;  // cond / uncond logits

    // CFG combine (cond + w*(cond-uncond)) then restrict to EOS + audio codes,
    // then sample. When CFG is off, `lu` is unused.
    auto combine_mask_sample = [&]() -> int {
        if (use_cfg) {
            const float w = params.cfg_scale;
            for (int v = AUDIO_CODE_BASE; v < V; v++) lc[v] = lu[v] + w * (lc[v] - lu[v]);
            lc[TOKEN_IM_END] = lu[TOKEN_IM_END] + w * (lc[TOKEN_IM_END] - lu[TOKEN_IM_END]);
        }
        for (int v = 0; v < AUDIO_CODE_BASE; v++) {
            if (v != TOKEN_IM_END) lc[v] = -1e9f;
        }
        return sample_top_k_p(lc.data(), V, params.temperature, params.top_p, params.top_k, rng);
    };

    lm_reset(m, 0);
    if (!lm_model_forward(m, tokens.data(), (int) tokens.size(), lc, 0)) {
        fprintf(stderr, "[lm-pipeline] cond prefill failed\n");
        return false;
    }
    if (use_cfg) {
        lm_reset(m, 1);
        if (!lm_model_forward(m, uncond.data(), (int) uncond.size(), lu, 1)) {
            fprintf(stderr, "[lm-pipeline] uncond prefill failed\n");
            return false;
        }
    }
    fprintf(stderr, "[lm-pipeline] prefill cond=%zu uncond=%zu, max_new=%d, temp=%.2f top_p=%.2f top_k=%d cfg=%.2f%s\n",
            tokens.size(), uncond.size(), max_tokens, params.temperature, params.top_p, params.top_k,
            use_cfg ? params.cfg_scale : 1.0f, use_cfg ? "" : " (CFG off)");

    int tok = combine_mask_sample();
    if (tok != TOKEN_IM_END && tok >= AUDIO_CODE_BASE && tok < AUDIO_CODE_BASE + AUDIO_CODE_COUNT) {
        codes_out.push_back(tok - AUDIO_CODE_BASE);
    }

    for (int step = 0; step < max_tokens && tok != TOKEN_IM_END; step++) {
        int32_t t = (int32_t) tok;
        if (!lm_model_forward(m, &t, 1, lc, 0)) {
            fprintf(stderr, "[lm-pipeline] cond decode step %d failed\n", step);
            return false;
        }
        if (use_cfg && !lm_model_forward(m, &t, 1, lu, 1)) {
            fprintf(stderr, "[lm-pipeline] uncond decode step %d failed\n", step);
            return false;
        }
        tok = combine_mask_sample();
        if (tok == TOKEN_IM_END) break;
        if (tok >= AUDIO_CODE_BASE && tok < AUDIO_CODE_BASE + AUDIO_CODE_COUNT) {
            codes_out.push_back(tok - AUDIO_CODE_BASE);
        }
        if ((step + 1) % 100 == 0) {
            fprintf(stderr, "[lm-pipeline]   step %d, %zu codes\n", step + 1, codes_out.size());
        }
    }

    fprintf(stderr, "[lm-pipeline] generated %zu audio codes\n", codes_out.size());
    {
        std::string csv;
        for (size_t i = 0; i < codes_out.size(); i++) {
            if (i) csv += ',';
            csv += std::to_string(codes_out[i]);
        }
        fprintf(stderr, "[lm-pipeline] codes: %s\n", csv.c_str());
    }
    return true;
}

} // namespace tts_cpp::acestep
