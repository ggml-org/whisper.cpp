// Decoding-knob resolution: argmax-forcing requests must be repaired to
// sampling (parler cannot terminate under argmax), everything else passed
// through.  Plus RNG-stream parity of the candidate-based sampler against a
// full-vocabulary reference of the HF warper chain.

#include "parler/sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
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
    parler_sampler_scratch scratch;
    std::vector<int32_t> f;
    bool all_argmax = true;
    for (int seed = 0; seed < 32; ++seed) {
        std::mt19937 rng((uint32_t) seed);
        parler_sample_frame(logits.data(), n_cb, vocab, p, rng, scratch, f);
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
        parler_sample_frame(flat.data(), 1, vocab, s, rng, scratch, f);
        seen.insert(f[0]);
    }
    CHECK(seen.size() > 1, "the repaired configuration actually samples");
}

// The full-vocabulary HF warper chain the shipped sampler used before the
// candidate-based rewrite: copy the row, temperature, top-k mask, top-p mask,
// softmax + one multinomial draw.  The rewrite must pick the same token from
// the same RNG stream.
static int32_t reference_sample_row(const float * row, int vocab,
                                    const parler_sampling_params & p,
                                    std::mt19937 & rng) {
    if (p.greedy) {
        int best = 0;
        for (int i = 1; i < vocab; ++i) {
            if (row[i] > row[best]) best = i;
        }
        return best;
    }

    std::vector<float> l(row, row + vocab);
    if (p.temperature > 0.0f && p.temperature != 1.0f) {
        for (float & v : l) v /= p.temperature;
    }
    if (p.top_k > 0 && p.top_k < vocab) {
        std::vector<float> sorted(l);
        std::nth_element(sorted.begin(), sorted.begin() + (p.top_k - 1), sorted.end(),
                         std::greater<float>());
        const float thresh = sorted[p.top_k - 1];
        for (float & v : l) {
            if (v < thresh) v = -std::numeric_limits<float>::infinity();
        }
    }
    if (p.top_p < 1.0f) {
        std::vector<int> idx(vocab);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return l[a] > l[b]; });
        float max_l = l[idx[0]];
        double denom = 0.0;
        std::vector<double> probs(vocab);
        for (int i = 0; i < vocab; ++i) {
            probs[i] = std::exp((double) l[idx[i]] - max_l);
            denom += probs[i];
        }
        double cum = 0.0;
        for (int i = 0; i < vocab; ++i) {
            cum += probs[i] / denom;
            if (cum > p.top_p) {
                for (int j = i + 1; j < vocab; ++j) {
                    l[idx[j]] = -std::numeric_limits<float>::infinity();
                }
                break;
            }
        }
    }
    float max_l = -std::numeric_limits<float>::infinity();
    for (float v : l) max_l = std::max(max_l, v);
    std::vector<double> probs(vocab, 0.0);
    double denom = 0.0;
    for (int i = 0; i < vocab; ++i) {
        if (std::isinf(l[i]) && l[i] < 0) continue;
        probs[i] = std::exp((double) l[i] - max_l);
        denom += probs[i];
    }
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng) * denom;
    double cum = 0.0;
    for (int i = 0; i < vocab; ++i) {
        cum += probs[i];
        if (r <= cum) return i;
    }
    return vocab - 1;
}

// Rows mimic what the delay-pattern processor hands the sampler: finite
// logits, EOS forced to -inf on some rows, and (top_p == 1 cases) tie blocks
// straddling the top-k threshold, where nth_element's keep-at-threshold
// semantics must be reproduced exactly.
static void fill_row(std::vector<float> & row, std::mt19937 & gen,
                     bool inject_ties, int eos_id) {
    std::uniform_real_distribution<float> logit(-8.0f, 8.0f);
    for (float & v : row) v = logit(gen);
    if (inject_ties) {
        const float tie = logit(gen);
        for (size_t i = 0; i < row.size(); i += 7) row[i] = tie;
    }
    if (gen() % 2 == 0) row[(size_t) eos_id] = -std::numeric_limits<float>::infinity();
}

static void test_matches_full_vocab_reference() {
    const int vocab = 259;
    const int eos_id = vocab - 2;
    const float temps[] = { 1.0f, 0.7f, 1.3f, 0.0f };
    const int   ks[]    = { 0, 2, 50, vocab, vocab + 7 };
    const float ps[]    = { 1.0f, 0.9f, 0.4f };

    parler_sampler_scratch scratch;
    std::vector<int32_t> frame;
    std::mt19937 gen(20260921);
    std::vector<float> row((size_t) vocab);
    bool tokens_match = true, streams_match = true;
    for (float temperature : temps) {
        for (int top_k : ks) {
            for (float top_p : ps) {
                parler_sampling_params p;
                p.temperature = temperature;
                p.top_k       = top_k;
                p.top_p       = top_p;
                for (int trial = 0; trial < 64; ++trial) {
                    // exact float ties have no defined order under top-p's
                    // sort, so tie injection stays on the top_p == 1 lanes
                    fill_row(row, gen, top_p == 1.0f, eos_id);
                    const uint32_t seed = gen();
                    std::mt19937 rng_ref(seed), rng_new(seed);
                    const int32_t a = reference_sample_row(row.data(), vocab, p, rng_ref);
                    parler_sample_frame(row.data(), 1, vocab, p, rng_new, scratch, frame);
                    if (a != frame[0]) tokens_match = false;
                    if (rng_ref() != rng_new()) streams_match = false;
                }
            }
        }
    }
    CHECK(tokens_match, "candidate-based sampler draws the reference tokens");
    CHECK(streams_match, "candidate-based sampler consumes the same RNG stream");

    // greedy bypasses the RNG entirely
    parler_sampling_params g;
    g.greedy = true;
    fill_row(row, gen, false, eos_id);
    std::mt19937 rng_a(7), rng_b(7);
    parler_sample_frame(row.data(), 1, vocab, g, rng_b, scratch, frame);
    CHECK(reference_sample_row(row.data(), vocab, g, rng_a) == frame[0],
          "greedy matches the reference argmax");
    CHECK(rng_a() == rng_b(), "greedy leaves the RNG untouched");
}

// A backend that miscomputes the decoder hands the sampler NaN logits; the
// candidate walk must refuse them instead of indexing past the vocabulary.
// Masked tokens arrive as -inf and must still sample normally.
static void test_non_finite_logits_rejected() {
    const int n_codebooks = 2;
    const int vocab       = 8;
    std::vector<float> logits((size_t) n_codebooks * vocab, 0.5f);
    logits[3] = -std::numeric_limits<float>::infinity();
    parler_sampler_scratch scratch;
    std::vector<int32_t>   frame;
    std::mt19937           rng(7);
    const parler_sampling_params p = parler_resolve_sampling(parler_sampling_request(), shipped());

    bool threw = false;
    try {
        parler_sample_frame(logits.data(), n_codebooks, vocab, p, rng, scratch, frame);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    CHECK(!threw, "masked -inf logits sample normally");
    CHECK(frame.size() == (size_t) n_codebooks, "one token per codebook");
    CHECK(frame[0] != 3, "a masked token is never drawn");

    for (float poison : { std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity() }) {
        logits[(size_t) vocab + 5] = poison;
        threw = false;
        try {
            parler_sample_frame(logits.data(), n_codebooks, vocab, p, rng, scratch, frame);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw, "NaN or +inf logits raise instead of sampling");
    }
}

int main() {
    test_resolution();
    test_top_k_1_is_argmax();
    test_matches_full_vocab_reference();
    test_non_finite_logits_rejected();

    if (g_failures == 0) {
        fprintf(stderr, "parler sampler: PASS\n");
        return 0;
    }
    fprintf(stderr, "parler sampler: %d failure(s)\n", g_failures);
    return 1;
}
