// Decoding-knob resolution: argmax-forcing requests must be repaired to
// sampling (parler cannot terminate under argmax), everything else passed through.

#include "parler/sampler.h"

#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace tts_cpp::parler::detail;

static int g_failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            ++g_failures;                                                      \
            fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__);   \
        }                                                                      \
    } while (0)

// the shipped checkpoints: do_sample true, temperature 1.0, top_k 50
static parler_gen_defaults shipped() { return parler_gen_defaults(); }

static void test_resolution() {
    // 1. nothing requested -> the model's own sampled defaults, no repair
    {
        std::string rep;
        const parler_sampling_params p = parler_resolve_sampling(
            parler_sampling_request(), shipped(), &rep);
        CHECK(!p.greedy, "default: samples");
        CHECK(p.top_k == 50, "default: top_k defers to the GGUF");
        CHECK(p.temperature == 1.0f, "default: temperature defers to the GGUF");
        CHECK(p.top_p == 1.0f, "default: top_p untouched");
        CHECK(rep.empty(), "default: nothing reported");
    }

    // 2. greedy -> repaired to the model's sampled defaults
    {
        parler_sampling_request r;
        r.greedy = true;
        std::string rep;
        const parler_sampling_params p = parler_resolve_sampling(r, shipped(), &rep);
        CHECK(!p.greedy, "greedy: repaired to sampling");
        CHECK(p.top_k == 50, "greedy: top_k falls back to the GGUF default");
        CHECK(!rep.empty(), "greedy: trigger reported");
    }

    // 3. top_k = 1 is argmax too (see test_top_k_1_is_argmax below)
    {
        parler_sampling_request r;
        r.top_k = 1;
        std::string rep;
        const parler_sampling_params p = parler_resolve_sampling(r, shipped(), &rep);
        CHECK(!p.greedy, "top_k=1: samples");
        CHECK(p.top_k == 50, "top_k=1: repaired to the GGUF default");
        CHECK(!rep.empty(), "top_k=1: trigger reported");
    }

    // 4. only the argmax-forcing knob is repaired -- an unrelated top_k survives
    {
        parler_sampling_request r;
        r.greedy = true;
        r.top_k  = 25;
        std::string rep;
        const parler_sampling_params p = parler_resolve_sampling(r, shipped(), &rep);
        CHECK(!p.greedy, "greedy+top_k=25: samples");
        CHECK(p.top_k == 25, "greedy+top_k=25: caller's top_k is not clobbered");
        CHECK(!rep.empty(), "greedy+top_k=25: trigger reported");
    }

    // 5. no-regression pin: an ordinary sampled request passes through untouched
    {
        parler_sampling_request r;
        r.top_k       = 50;
        r.temperature = 0.7f;
        r.top_p       = 0.9f;
        std::string rep;
        const parler_sampling_params p = parler_resolve_sampling(r, shipped(), &rep);
        CHECK(!p.greedy, "ordinary: samples");
        CHECK(p.top_k == 50, "ordinary: top_k preserved");
        CHECK(p.temperature == 0.7f, "ordinary: temperature preserved");
        CHECK(p.top_p == 0.9f, "ordinary: top_p preserved");
        CHECK(rep.empty(), "ordinary: nothing reported");
    }

    // 6. a GGUF asking for greedy is overridden the same way
    {
        parler_gen_defaults d = shipped();
        d.do_sample = false;
        std::string rep;
        const parler_sampling_params p = parler_resolve_sampling(
            parler_sampling_request(), d, &rep);
        CHECK(!p.greedy, "do_sample=false: samples anyway");
        CHECK(!rep.empty(), "do_sample=false: trigger reported");
    }

    // 7. a GGUF whose own top_k is 1 must not resolve back to argmax
    {
        parler_gen_defaults d = shipped();
        d.top_k = 1;
        parler_sampling_request r;
        r.top_k = 1;
        std::string rep;
        const parler_sampling_params p = parler_resolve_sampling(r, d, &rep);
        CHECK(!p.greedy, "degenerate GGUF: samples");
        CHECK(p.top_k == 0, "degenerate GGUF: top_k drops the filter, never stays 1");
        CHECK(!rep.empty(), "degenerate GGUF: trigger reported");
    }

    // repaired is optional
    parler_resolve_sampling(parler_sampling_request(), shipped(), nullptr);
}

// The premise the guard rests on: top_k = 1 masks everything below the largest
// logit, so the multinomial draw has one candidate and is argmax for any seed.
static void test_top_k_1_is_argmax() {
    const int n_cb = 3, vocab = 8;
    std::vector<float> logits((size_t) n_cb * vocab, 0.0f);
    const int argmax[n_cb] = { 5, 0, 7 };
    for (int k = 0; k < n_cb; ++k) {
        for (int v = 0; v < vocab; ++v) logits[(size_t) k * vocab + v] = 0.1f * (float) v;
        logits[(size_t) k * vocab + argmax[k]] = 10.0f;
    }

    parler_sampling_params p;   // sampling, but with a single-token nucleus
    p.top_k = 1;
    bool all_argmax = true;
    for (int seed = 0; seed < 32; ++seed) {
        std::mt19937 rng((uint32_t) seed);
        const std::vector<int32_t> f = parler_sample_frame(logits.data(), n_cb, vocab, p, rng);
        for (int k = 0; k < n_cb; ++k) if (f[k] != argmax[k]) all_argmax = false;
    }
    CHECK(all_argmax, "top_k=1 is argmax for every seed (this is why it is repaired)");

    // and the repaired configuration really does sample: a flat distribution
    // must yield more than one distinct token across many draws
    parler_sampling_params s;
    s.top_k = 50;   // >= vocab, so no filtering
    std::vector<float> flat((size_t) vocab, 0.0f);
    std::set<int32_t> seen;
    std::mt19937 rng(1234);
    for (int i = 0; i < 200; ++i) {
        seen.insert(parler_sample_frame(flat.data(), 1, vocab, s, rng)[0]);
    }
    CHECK(seen.size() > 1, "the repaired configuration actually samples");
}

int main() {
    test_resolution();
    test_top_k_1_is_argmax();

    if (g_failures == 0) {
        fprintf(stderr, "parler sampler: PASS\n");
        return 0;
    }
    fprintf(stderr, "parler sampler: %d failure(s)\n", g_failures);
    return 1;
}
