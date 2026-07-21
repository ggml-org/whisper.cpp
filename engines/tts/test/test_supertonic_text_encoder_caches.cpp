// TDD harness for the audit follow-up #2 caches added to
// `supertonic_text_encoder`'s GPU hot path.
//
// Two findings checked here, both fixture-bound (require the
// Supertonic GGUF + auto-DISABLED when the model isn't present):
//
//   F13  Text-encoder layer-norm weight host-side cache.
//        The text-encoder GGML production path runs four
//        `relpos + LN + ffn + LN` iterations followed by a final
//        speech-prompted LN.  Pre-audit, each LN downloaded its
//        γ + β tensors from the backend via `read_f32(...)` on
//        every synth — 18 downloads / synth = 18 sync points on
//        a non-CPU backend.  Caching them once at load (same
//        pattern as F1 RoPE θ) drops that to zero.
//
//   F16  Speech-prompted attention `tanh_k` host-side cache.
//        The two speech-prompted attention layers each pull a
//        constant `tanh_k` tensor (~50 × 256 = 51.2 KiB) on
//        every synth.  Cache it once at load and consume the
//        host pointer at both call sites.
//
// Validation strategy:
//   1. After `load_supertonic_gguf` returns, the new cache
//      fields on `supertonic_model` are populated with the right
//      shapes (size + content match a direct backend read of the
//      source tensor).
//   2. The roster of cached LN weights covers exactly the 10
//      hot-path LN pairs the text encoder consumes per synth
//      (4 × `norm_layers_1.X` + 4 × `norm_layers_2.X` +
//       final `speech_prompted_text_encoder.norm.norm`).
//
// Registered with `LABEL "fixture"` in CMakeLists.txt.

#include "supertonic_internal.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace tts_cpp::supertonic::detail;

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond) do {                                              \
    ++g_checks;                                                       \
    if (!(cond)) {                                                    \
        ++g_failures;                                                 \
        std::fprintf(stderr, "FAIL %s:%d  %s\n",                     \
                     __FILE__, __LINE__, #cond);                      \
    }                                                                 \
} while (0)

std::vector<float> dump_f32(ggml_tensor * tensor) {
    std::vector<float> out((size_t) ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, out.data(), 0, ggml_nbytes(tensor));
    return out;
}

ggml_tensor * find_source(const supertonic_model & model, const std::string & key) {
    auto it = model.source_tensors.find(key);
    return it == model.source_tensors.end() ? nullptr : it->second;
}

// F13 — text-encoder layer-norm weights host-side cache.
//
// The expected roster (10 LN pairs) is the union of:
//   - the four `attn_encoder.norm_layers_1.X` (post-relpos
//     residual norms, X ∈ {0..3})
//   - the four `attn_encoder.norm_layers_2.X` (post-FFN residual
//     norms, X ∈ {0..3})
//   - the two `attn_encoder.norm_layers_*.X` for the speech-
//     prompted block exists only as the final
//     `speech_prompted_text_encoder.norm.norm` so it counts as
//     one extra cache entry in the production path, but the
//     "norm_layers" naming convention covers the first 8.
//
// Test asserts:
//   - `model.text_encoder_ln_weights` is populated with at least
//     the 8 attn_encoder pairs + the 1 speech-prompted final.
//   - Each cached vector matches a direct backend read of the
//     corresponding source tensor bit-exactly.
void test_f13_text_encoder_ln_cache(const supertonic_model & model) {
    std::fprintf(stderr, "[F13 text-encoder LN weight cache]\n");

    // Contract: helper accessor + map populated for at least the
    // four attn_encoder norm_layers_{1,2}.{0..3} pairs.  Allows
    // additional entries (the final speech-prompted norm, future
    // audit roster expansions) without trip-wiring the test.
    int matched = 0;
    int bad = 0;
    static const char * const kRosterStems[] = {
        "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_1.0",
        "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_1.1",
        "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_1.2",
        "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_1.3",
        "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_2.0",
        "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_2.1",
        "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_2.2",
        "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_2.3",
        "text_encoder:tts.ttl.speech_prompted_text_encoder.norm",
    };

    for (const char * stem : kRosterStems) {
        const std::string g_name = std::string(stem) + ".norm.weight";
        const std::string b_name = std::string(stem) + ".norm.bias";

        // Each entry in the cache map is keyed on the SOURCE name
        // (the `text_encoder:...` string), value is the cached
        // host vector ready for `layer_norm_channel` to consume.
        auto gamma_it = model.text_encoder_ln_weights.find(g_name);
        auto beta_it  = model.text_encoder_ln_weights.find(b_name);

        ggml_tensor * gamma_src = find_source(model, g_name);
        ggml_tensor * beta_src  = find_source(model, b_name);
        if (!gamma_src || !beta_src) {
            std::fprintf(stderr, "  SKIP %s (source tensor missing)\n", stem);
            continue;
        }
        ++matched;
        CHECK(gamma_it != model.text_encoder_ln_weights.end());
        CHECK(beta_it  != model.text_encoder_ln_weights.end());
        if (gamma_it == model.text_encoder_ln_weights.end() ||
            beta_it  == model.text_encoder_ln_weights.end()) {
            continue;
        }

        // Contract: cached size matches the source tensor.
        CHECK(gamma_it->second.size() == (size_t) ggml_nelements(gamma_src));
        CHECK(beta_it->second.size()  == (size_t) ggml_nelements(beta_src));

        // Contract: cached bytes match a direct backend read.
        auto gamma_direct = dump_f32(gamma_src);
        auto beta_direct  = dump_f32(beta_src);
        for (size_t i = 0; i < gamma_direct.size(); ++i) {
            if (gamma_it->second[i] != gamma_direct[i]) {
                if (bad < 2) {
                    std::fprintf(stderr,
                                 "  %s gamma mismatch @ %zu: cached=%g direct=%g\n",
                                 stem, i, gamma_it->second[i], gamma_direct[i]);
                }
                ++bad;
            }
        }
        for (size_t i = 0; i < beta_direct.size(); ++i) {
            if (beta_it->second[i] != beta_direct[i]) {
                if (bad < 2) {
                    std::fprintf(stderr,
                                 "  %s beta mismatch @ %zu: cached=%g direct=%g\n",
                                 stem, i, beta_it->second[i], beta_direct[i]);
                }
                ++bad;
            }
        }
    }
    CHECK(bad == 0);
    std::fprintf(stderr,
                 "  matched %d / %zu pairs, bad=%d\n",
                 matched, sizeof(kRosterStems)/sizeof(kRosterStems[0]), bad);
}

// F16 — speech-prompted attention `tanh_k` host-side cache.
//
// Two `tanh_k` tensors (one per speech-prompted attention layer)
// were previously downloaded via `read_f32(...)` inside
// `speech_prompted_attention_ggml` on every synth.  Caching them
// at load drops 2 GPU→host sync points per synth.
//
// Source names match the production path (lines 622 / 796 in
// `supertonic_text_encoder.cpp` pre-fix):
//   text_encoder:/speech_prompted_text_encoder/attention1/tanh/Tanh_output_0
//   text_encoder:/speech_prompted_text_encoder/attention2/tanh/Tanh_output_0
void test_f16_speech_tanh_k_cache(const supertonic_model & model) {
    std::fprintf(stderr, "[F16 speech tanh_k cache]\n");

    static const char * const kTanhSources[2] = {
        "text_encoder:/speech_prompted_text_encoder/attention1/tanh/Tanh_output_0",
        "text_encoder:/speech_prompted_text_encoder/attention2/tanh/Tanh_output_0",
    };
    int matched = 0;
    int bad = 0;
    for (int i = 0; i < 2; ++i) {
        ggml_tensor * src = find_source(model, kTanhSources[i]);
        if (!src) {
            std::fprintf(stderr, "  SKIP %s (not in GGUF)\n", kTanhSources[i]);
            continue;
        }
        ++matched;
        const std::vector<float> & cached = model.speech_tanh_k_cache[i];
        CHECK(cached.size() == (size_t) ggml_nelements(src));
        if (cached.size() != (size_t) ggml_nelements(src)) continue;

        auto direct = dump_f32(src);
        for (size_t j = 0; j < direct.size(); ++j) {
            if (cached[j] != direct[j]) {
                if (bad < 2) {
                    std::fprintf(stderr,
                                 "  tanh_k[%d] mismatch @ %zu: cached=%g direct=%g\n",
                                 i, j, cached[j], direct[j]);
                }
                ++bad;
            }
        }
    }
    CHECK(bad == 0);
    std::fprintf(stderr, "  matched %d / 2 tanh_k tensors, bad=%d\n", matched, bad);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }
    supertonic_model model;
    if (!load_supertonic_gguf(argv[1], model)) {
        std::fprintf(stderr, "failed to load model: %s\n", argv[1]);
        return 1;
    }

    test_f13_text_encoder_ln_cache(model);
    test_f16_speech_tanh_k_cache(model);

    free_supertonic_model(model);

    std::fprintf(stderr,
                 "test_supertonic_text_encoder_caches: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
