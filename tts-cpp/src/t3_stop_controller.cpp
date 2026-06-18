#include "t3_stop_controller.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace tts_cpp::chatterbox::detail {

namespace {

// --- small env-override helpers (calibration without a recompile) ----------

bool env_present(const char * name) {
    const char * v = std::getenv(name);
    return v && v[0] != '\0';
}

// Treats unset / "" / "0" / "false" / "off" as "not disabled".
bool env_truthy(const char * name) {
    const char * v = std::getenv(name);
    if (!v || v[0] == '\0') return false;
    if (std::strcmp(v, "0") == 0) return false;
    if (std::strcmp(v, "false") == 0 || std::strcmp(v, "FALSE") == 0) return false;
    if (std::strcmp(v, "off") == 0 || std::strcmp(v, "OFF") == 0) return false;
    return true;
}

int env_int(const char * name, int fallback) {
    const char * v = std::getenv(name);
    if (!v || v[0] == '\0') return fallback;
    char * end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v) return fallback;
    return (int) parsed;
}

float env_float(const char * name, float fallback) {
    const char * v = std::getenv(name);
    if (!v || v[0] == '\0') return fallback;
    char * end = nullptr;
    float parsed = std::strtof(v, &end);
    if (end == v) return fallback;
    return parsed;
}

// argmax of the CFG-combined logits.  combined[i] = cond[i] + w*(cond[i]-uncond[i]).
// When `uncond` is empty (non-CFG callers), combined == cond.  Returns -1 for
// an empty / size-mismatched input.
int combined_argmax(const std::vector<float> & cond,
                    const std::vector<float> & uncond,
                    float                      cfg_weight) {
    const size_t V = cond.size();
    if (V == 0) return -1;
    const bool use_cfg = (!uncond.empty() && uncond.size() == V && cfg_weight != 0.0f);

    int   best_i = 0;
    float best_v = -INFINITY;
    for (size_t i = 0; i < V; ++i) {
        float l = use_cfg ? cond[i] + cfg_weight * (cond[i] - uncond[i]) : cond[i];
        if (l > best_v) { best_v = l; best_i = (int) i; }
    }
    return best_i;
}

// Softmax probability of `idx` over the CFG-combined logits.  Only called when
// the probability gate is enabled.
float combined_softmax_prob(const std::vector<float> & cond,
                            const std::vector<float> & uncond,
                            float                      cfg_weight,
                            int                        idx) {
    const size_t V = cond.size();
    if (V == 0 || idx < 0 || (size_t) idx >= V) return 0.0f;
    const bool use_cfg = (!uncond.empty() && uncond.size() == V && cfg_weight != 0.0f);

    auto comb = [&](size_t i) -> float {
        return use_cfg ? cond[i] + cfg_weight * (cond[i] - uncond[i]) : cond[i];
    };

    float maxl = -INFINITY;
    for (size_t i = 0; i < V; ++i) maxl = std::max(maxl, comb(i));
    if (!std::isfinite(maxl)) return 0.0f;

    double sum = 0.0;
    for (size_t i = 0; i < V; ++i) sum += std::exp((double) (comb(i) - maxl));
    if (sum <= 0.0) return 0.0f;
    return (float) (std::exp((double) (comb((size_t) idx) - maxl)) / sum);
}

} // namespace

void t3_stop_controller::reset(const t3_stop_params & p) {
    params_     = p;
    eos_streak_ = 0;
}

bool t3_stop_controller::force_eos(int                        n_generated,
                                   const std::vector<float> & logits_cond,
                                   const std::vector<float> & logits_uncond) {
    if (!params_.enabled || params_.eos_argmax_streak <= 0) {
        eos_streak_ = 0;
        return false;
    }

    const int argmax = combined_argmax(logits_cond, logits_uncond, params_.cfg_weight);
    if (argmax == params_.stop_token) {
        ++eos_streak_;
    } else {
        eos_streak_ = 0;
        return false;
    }

    if (n_generated < params_.min_tokens) return false;
    if (eos_streak_ < params_.eos_argmax_streak) return false;

    if (params_.eos_prob_threshold > 0.0f) {
        const float p = combined_softmax_prob(logits_cond, logits_uncond,
                                               params_.cfg_weight, params_.stop_token);
        if (p < params_.eos_prob_threshold) return false;
    }

    return true;
}

t3_post_result t3_stop_controller::post_check(const std::vector<int32_t> & generated) const {
    t3_post_result result;
    if (!params_.enabled) return result;

    const int n = (int) generated.size();
    if (n < params_.min_tokens) return result;

    // 1. Repetition: smallest period first so a period-1 stutter (the common
    //    near-silent cadence) is caught and reported as period 1.
    if (params_.rep_repeats > 1 && params_.rep_max_period >= 1) {
        const int max_p = std::min(params_.rep_max_period, n / params_.rep_repeats);
        for (int p = 1; p <= max_p; ++p) {
            // Are the last (rep_repeats * p) tokens `rep_repeats` copies of the
            // trailing p-token block?
            bool periodic = true;
            const int span = params_.rep_repeats * p;
            for (int k = 0; k < span; ++k) {
                if (generated[n - 1 - k] != generated[n - 1 - (k % p)]) {
                    periodic = false;
                    break;
                }
            }
            if (!periodic) continue;

            // Extend the run backwards to count every trailing period, then
            // trim all but one copy so the stutter collapses cleanly.
            int periods = params_.rep_repeats;
            while ((periods + 1) * p <= n) {
                bool ok = true;
                for (int k = periods * p; k < (periods + 1) * p; ++k) {
                    if (generated[n - 1 - k] != generated[n - 1 - (k % p)]) {
                        ok = false;
                        break;
                    }
                }
                if (!ok) break;
                ++periods;
            }
            result.reason    = t3_stop_reason::repetition;
            result.trim_tail = (periods - 1) * p;
            return result;
        }
    }

    // 2. Budget backstop.
    if (params_.max_tokens > 0 && n >= params_.max_tokens) {
        result.reason = t3_stop_reason::budget;
        return result;
    }

    return result;
}

t3_stop_params make_mtl_stop_params(int32_t stop_token,
                                    float   cfg_weight,
                                    int     n_text_tokens,
                                    int     n_predict_cap) {
    t3_stop_params p;
    p.enabled    = true;
    p.stop_token = stop_token;
    p.cfg_weight = cfg_weight;

    // Skip the speech onset before any heuristic may fire.
    p.min_tokens = std::max(0, env_int("CHATTERBOX_STOP_MIN_TOKENS", 16));

    // Proportional budget: generous multiple of the text length so genuinely
    // long inputs are never clipped, with an absolute floor for very short
    // inputs and an absolute ceiling at the caller's n_predict.  Real MTL
    // speech runs ~5-8 tokens per text token; 20x leaves a wide safety margin
    // while still bounding a total EOS failure to a few seconds.
    constexpr int   kBudgetFloor = 96;
    const float ratio = std::max(0.0f, env_float("CHATTERBOX_STOP_MAX_RATIO", 20.0f));
    int budget = (int) std::lround((double) ratio * (double) std::max(0, n_text_tokens))
                 + 64;
    budget = std::max(budget, kBudgetFloor);
    if (n_predict_cap > 0) budget = std::min(budget, n_predict_cap);
    budget = env_int("CHATTERBOX_STOP_MAX_TOKENS", budget);
    p.max_tokens = budget;

    p.rep_max_period    = std::max(0, env_int("CHATTERBOX_STOP_REP_PERIOD", 8));
    p.rep_repeats       = std::max(0, env_int("CHATTERBOX_STOP_REP_REPEATS", 3));
    p.eos_argmax_streak = env_int("CHATTERBOX_STOP_EOS_STREAK", 2);
    p.eos_prob_threshold = std::max(0.0f, env_float("CHATTERBOX_STOP_EOS_PROB", 0.0f));

    if (env_present("CHATTERBOX_STOP_DISABLE") && env_truthy("CHATTERBOX_STOP_DISABLE")) {
        p.enabled = false;
    }
    return p;
}

} // namespace tts_cpp::chatterbox::detail
