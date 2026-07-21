// TDD harness for the audit follow-up #3 caches: F17 (duration
// scalar-continuation weight cache), F18 (text-encoder convnext-
// front graph cache), and F19 (vector-estimator front-block graph
// cache).
//
// Each finding is a "make the second call cheaper" change: the
// graph or weight bytes that the per-synth code path reaches for
// are pulled out into model-lifetime storage on first touch, then
// reused on every subsequent call.  Math is unchanged; the
// test gate is a strict "two consecutive calls with identical
// inputs produce bit-exact identical outputs" — if the cache
// accidentally aliases buffers or resets state across calls, this
// test trips.
//
//   F17 — Duration scalar-continuation `read_f32` cache.
//         `supertonic_duration_forward_ggml` runs ~30 backend
//         tensor reads in its scalar continuation (after the
//         cached graph computes Q/K/V).  Validates that the
//         `model.scalar_weight_cache` map is populated after the
//         first synth and reused on the second.
//
//   F18 — Text-encoder convnext-front graph cache.
//         `supertonic_text_encoder_forward_ggml` previously
//         allocated a fresh `ggml_context` + `gallocr` for the
//         front-half ConvNeXt graph on every synth.  Validates
//         that the second synth produces bit-exact output.
//
//   F19 — Vector-estimator front-block graph cache.
//         `supertonic_vector_trace_proj_ggml` allocated a fresh
//         ~200-node graph per denoise step (5 alloc/free per
//         synth on the default schedule).  Validates that two
//         consecutive `supertonic_vector_step_ggml` calls with
//         identical inputs are bit-exact (already partially
//         covered by F8 / F11 tests; this extends with the front
//         block being the new cached island).
//
// Registered with `LABEL "fixture"` — needs the Supertonic GGUF.

#include "supertonic_internal.h"

#include <cstdio>
#include <random>
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

std::vector<float> make_synthetic(int n, uint32_t seed) {
    std::vector<float> out((size_t) n);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & v : out) v = dist(rng);
    return out;
}

// F17 — Duration scalar weight cache.
//
// Contract:
//   - After the first `supertonic_duration_forward_ggml` call,
//     `model.scalar_weight_cache` contains at least one rostered
//     entry (the relpos K/V embeddings + conv_o weight/bias are
//     the audit's hot list).
//   - A second call with the same input produces bit-exactly the
//     same duration scalar (the cache must not corrupt values).
//   - Cache size does NOT grow on the second call (every entry
//     was a cache hit).
void test_f17_duration_scalar_weight_cache(const supertonic_model & model) {
    std::fprintf(stderr, "[F17 duration scalar weight cache]\n");

    if (model.voices.empty()) {
        std::fprintf(stderr, "  SKIP: no voices in model\n");
        return;
    }
    const auto & voice = model.voices.begin()->second;
    std::vector<float> style_dp((size_t) ggml_nelements(voice.dp));
    ggml_backend_tensor_get(voice.dp, style_dp.data(), 0, ggml_nbytes(voice.dp));

    std::vector<int64_t> text_ids;
    for (int i = 1; i <= 16; ++i) text_ids.push_back(i);

    std::string err;
    float dur1 = 0.0f;
    const size_t cache_before = model.scalar_weight_cache.size();
    if (!supertonic_duration_forward_ggml(model, text_ids.data(),
                                           (int) text_ids.size(),
                                           style_dp.data(), dur1, &err)) {
        std::fprintf(stderr, "  SKIP duration call 1: %s\n", err.c_str());
        return;
    }
    const size_t cache_after_one = model.scalar_weight_cache.size();
    std::fprintf(stderr, "  cache size: before=%zu  after-1=%zu\n",
                 cache_before, cache_after_one);
    CHECK(cache_after_one > cache_before);

    // Specific rostered entries we expect (matches the call sites
    // that `cached_read_f32` replaced).  Sub-rostered: not every
    // GGUF carries every key, so we accept >= 4 of the 6 spotchecks.
    static const char * const kRostered[] = {
        "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.emb_rel_k",
        "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.emb_rel_v",
        "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_o.weight",
        "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.conv_o.bias",
        "duration:tts.dp.sentence_encoder.proj_out.net.weight",
        "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_1.0.norm.weight",
    };
    int hits = 0;
    for (const char * key : kRostered) {
        if (model.scalar_weight_cache.find(key) != model.scalar_weight_cache.end()) {
            ++hits;
        }
    }
    std::fprintf(stderr, "  spot-check rostered entries: %d / %zu present\n",
                 hits, sizeof(kRostered) / sizeof(kRostered[0]));
    CHECK(hits >= 4);

    // Second call must NOT grow the cache (every entry is a hit).
    float dur2 = 0.0f;
    if (!supertonic_duration_forward_ggml(model, text_ids.data(),
                                           (int) text_ids.size(),
                                           style_dp.data(), dur2, &err)) {
        std::fprintf(stderr, "  SKIP duration call 2: %s\n", err.c_str());
        return;
    }
    const size_t cache_after_two = model.scalar_weight_cache.size();
    CHECK(cache_after_two == cache_after_one);
    std::fprintf(stderr, "  cache size: after-2=%zu (must == after-1)\n", cache_after_two);

    // Bit-exact duration across the two calls.
    CHECK(dur1 == dur2);
    std::fprintf(stderr, "  dur1=%.6g  dur2=%.6g\n", dur1, dur2);
}

// F18 — Text-encoder convnext-front graph cache.
//
// Contract: two consecutive `supertonic_text_encoder_forward_ggml`
// calls with identical inputs produce bit-exact identical output
// vectors.  The first call rebuilds the cached graph; the second
// reuses it.  If the cache state leaks across calls (e.g. allocator
// re-aliases an input tensor's buffer with an intermediate's), this
// test trips.
void test_f18_text_encoder_convnext_cache(const supertonic_model & model) {
    std::fprintf(stderr, "[F18 text-encoder convnext-front graph cache]\n");

    if (model.voices.empty()) {
        std::fprintf(stderr, "  SKIP: no voices in model\n");
        return;
    }
    const auto & voice = model.voices.begin()->second;
    std::vector<float> style_ttl((size_t) ggml_nelements(voice.ttl));
    ggml_backend_tensor_get(voice.ttl, style_ttl.data(), 0, ggml_nbytes(voice.ttl));

    std::vector<int64_t> text_ids;
    for (int i = 1; i <= 24; ++i) text_ids.push_back(i);

    std::string err;
    std::vector<float> emb1, emb2;
    if (!supertonic_text_encoder_forward_ggml(model, text_ids.data(),
                                               (int) text_ids.size(),
                                               style_ttl.data(), emb1, &err)) {
        std::fprintf(stderr, "  SKIP call 1: %s\n", err.c_str());
        return;
    }
    if (!supertonic_text_encoder_forward_ggml(model, text_ids.data(),
                                               (int) text_ids.size(),
                                               style_ttl.data(), emb2, &err)) {
        std::fprintf(stderr, "  SKIP call 2: %s\n", err.c_str());
        return;
    }

    CHECK(emb1.size() == emb2.size());
    int bad = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < emb1.size() && i < emb2.size(); ++i) {
        const float d = std::fabs(emb1[i] - emb2[i]);
        if (d > 0.0f) ++bad;
        max_abs = std::max(max_abs, d);
    }
    std::fprintf(stderr,
                 "  emb.size=%zu  max_abs_diff=%.3e  bad=%d (must be 0)\n",
                 emb1.size(), max_abs, bad);
    CHECK(bad == 0);
}

// F19 — Vector-estimator front-block graph cache.
//
// Contract: same as F18.  `supertonic_vector_step_ggml` invokes
// `supertonic_vector_trace_proj_ggml` internally, which has the
// front-block graph.  Two consecutive calls with identical inputs
// must yield bit-exact identical outputs.  Builds on the F8 / F11
// tests with the new front-block cache as the additional gate.
void test_f19_vector_front_block_cache(const supertonic_model & model) {
    std::fprintf(stderr, "[F19 vector-estimator front-block cache]\n");

    if (model.voices.empty()) {
        std::fprintf(stderr, "  SKIP: no voices in model\n");
        return;
    }
    const auto & voice = model.voices.begin()->second;
    std::vector<float> style_ttl((size_t) ggml_nelements(voice.ttl));
    ggml_backend_tensor_get(voice.ttl, style_ttl.data(), 0, ggml_nbytes(voice.ttl));

    const int text_len   = 24;
    const int latent_len = 12;
    const int Cin        = model.hparams.latent_channels;

    auto latent     = make_synthetic(Cin * latent_len, 0xF00D);
    auto text_emb   = make_synthetic(256 * text_len,   0xBEEF);
    std::vector<float> latent_mask((size_t) latent_len, 1.0f);

    std::string err;
    std::vector<float> next1, next2;
    if (!supertonic_vector_step_ggml(model, latent.data(), latent_len,
                                     text_emb.data(), text_len,
                                     style_ttl.data(), latent_mask.data(),
                                     /*current_step=*/0, /*total_steps=*/5,
                                     next1, &err)) {
        std::fprintf(stderr, "  SKIP step 1: %s\n", err.c_str());
        return;
    }
    if (!supertonic_vector_step_ggml(model, latent.data(), latent_len,
                                     text_emb.data(), text_len,
                                     style_ttl.data(), latent_mask.data(),
                                     /*current_step=*/0, /*total_steps=*/5,
                                     next2, &err)) {
        std::fprintf(stderr, "  SKIP step 2: %s\n", err.c_str());
        return;
    }
    CHECK(next1.size() == next2.size());
    int bad = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < next1.size() && i < next2.size(); ++i) {
        const float d = std::fabs(next1[i] - next2[i]);
        if (d > 0.0f) ++bad;
        max_abs = std::max(max_abs, d);
    }
    std::fprintf(stderr,
                 "  next.size=%zu  max_abs_diff=%.3e  bad=%d (must be 0)\n",
                 next1.size(), max_abs, bad);
    CHECK(bad == 0);
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

    test_f17_duration_scalar_weight_cache(model);
    test_f18_text_encoder_convnext_cache(model);
    test_f19_vector_front_block_cache(model);

    free_supertonic_model(model);

    std::fprintf(stderr,
                 "test_supertonic_audit3_caches: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
