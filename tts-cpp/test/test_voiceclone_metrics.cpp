// Self-tests for the voice-clone metrics module (QVAC-20979).  Pure host logic;
// runs in the always-on `unit` ctest tier.  Two layers of coverage:
//
//   1. Closed-form self-tests with values computed by hand, so the metric
//      implementations are pinned independently of any fixture.
//   2. Fixture reproduction: load the committed golden fixtures under
//      test/fixtures/voiceclone/v1/ and assert each metric reproduces the
//      committed expected score.  This proves the .npy ingestion path and locks
//      the documented numbers (see that folder's README + dump script).
//
// Standalone build:
//   g++ -std=c++17 -I src test/test_voiceclone_metrics.cpp src/voiceclone_metrics.cpp -o /tmp/t
//   /tmp/t test/fixtures/voiceclone/v1

#include "voiceclone_metrics.h"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace tts_cpp::voiceclone;

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, ...) do {                                            \
    ++g_checks;                                                          \
    if (!(cond)) {                                                       \
        ++g_failures;                                                    \
        fprintf(stderr, "FAIL %s:%d  ", __FILE__, __LINE__);            \
        fprintf(stderr, __VA_ARGS__);                                    \
        fprintf(stderr, "\n");                                          \
    }                                                                    \
} while (0)

bool close(double a, double b, double tol = 1e-6) {
    return std::fabs(a - b) <= tol;
}

// ----------------------------------------------------------------------------
// 1. Closed-form self-tests.
// ----------------------------------------------------------------------------

void test_cosine_known_values() {
    CHECK(close(cosine_similarity({1, 0, 0}, {1, 0, 0}), 1.0), "identical -> 1");
    CHECK(close(cosine_similarity({1, 0}, {0, 1}), 0.0), "orthogonal -> 0");
    CHECK(close(cosine_similarity({1, 0}, {-1, 0}), -1.0), "anti-parallel -> -1");
    CHECK(close(cosine_similarity({1, 0}, {1, 1}), 1.0 / std::sqrt(2.0)),
          "45 degrees -> 1/sqrt(2)");
    // Scale invariance.
    CHECK(close(cosine_similarity({2, 4, 6}, {1, 2, 3}), 1.0), "scaled copy -> 1");
    // Zero-norm input defined as 0 (silent clip), not NaN.
    CHECK(close(cosine_similarity({0, 0}, {1, 1}), 0.0), "zero vector -> 0");
}

void test_cosine_rejects_bad_input() {
    bool threw = false;
    try { cosine_similarity({1, 2}, {1, 2, 3}); } catch (const std::invalid_argument &) { threw = true; }
    CHECK(threw, "size mismatch must throw");
}

void test_time_avg_stats_known() {
    // H = [[0, 0], [2, 4]]  (T=2, D=2)
    //   mu    = [1, 2]
    //   sigma = [sqrt(((0-1)^2+(2-1)^2)/2), sqrt(((0-2)^2+(4-2)^2)/2)] = [1, 2]
    const std::vector<float> H = {0, 0, 2, 4};
    const TimeAvgStats s = time_avg_stats(H, /*T=*/2, /*D=*/2);
    CHECK(close(s.mu[0], 1.0) && close(s.mu[1], 2.0), "mu = [1, 2]");
    CHECK(close(s.sigma[0], 1.0) && close(s.sigma[1], 2.0), "sigma = [1, 2]");
}

void test_time_avg_stats_single_frame_zero_std() {
    const std::vector<float> H = {3, -7};
    const TimeAvgStats s = time_avg_stats(H, /*T=*/1, /*D=*/2);
    CHECK(close(s.mu[0], 3.0) && close(s.mu[1], -7.0), "single-frame mu = the frame");
    CHECK(close(s.sigma[0], 0.0) && close(s.sigma[1], 0.0), "single-frame sigma = 0");
}

void test_style_loss_known() {
    // a: mu=[1,2] sigma=[1,2]   (from H above)
    // b: identical              -> loss 0
    const std::vector<float> Ha = {0, 0, 2, 4};
    CHECK(close(wavlm_style_loss(Ha, 2, Ha, 2, 2), 0.0), "identical hidden states -> 0 loss");

    // c: mu=[0,0] sigma=[0,0]  (single zero frame repeated)
    // loss vs a = ||[1,2]||^2 + ||[1,2]||^2 = (1+4) + (1+4) = 10
    const std::vector<float> Hc = {0, 0, 0, 0};
    CHECK(close(wavlm_style_loss(Ha, 2, Hc, 2, 2), 10.0), "style loss = 10");
}

void test_wer_known_cases() {
    // Perfect match.
    CHECK(close(word_error_rate("the cat sat", "the cat sat").wer, 0.0), "perfect -> 0");

    // One deletion out of 3 ref words.
    {
        WerResult r = word_error_rate("the cat sat", "the cat");
        CHECK(r.deletions == 1 && r.substitutions == 0 && r.insertions == 0, "1 deletion");
        CHECK(close(r.wer, 1.0 / 3.0), "wer = 1/3");
    }
    // One substitution.
    {
        WerResult r = word_error_rate("the cat sat", "the dog sat");
        CHECK(r.substitutions == 1 && r.deletions == 0 && r.insertions == 0, "1 substitution");
        CHECK(close(r.wer, 1.0 / 3.0), "wer = 1/3");
    }
    // One insertion.
    {
        WerResult r = word_error_rate("the cat sat", "the big cat sat");
        CHECK(r.insertions == 1 && r.deletions == 0 && r.substitutions == 0, "1 insertion");
        CHECK(close(r.wer, 1.0 / 3.0), "wer = 1/3");
    }
    // Mixed: ref 4 words, hyp drops one and substitutes one -> S=1, D=1, WER=2/4.
    {
        WerResult r = word_error_rate("a b c d", "a x c");
        CHECK(r.ref_len == 4, "ref_len 4");
        CHECK(close(r.wer, 2.0 / 4.0), "wer = 0.5 (got %.4f, S=%d D=%d I=%d)",
              r.wer, r.substitutions, r.deletions, r.insertions);
    }
    // Empty reference, non-empty hypothesis -> all insertions.
    {
        WerResult r = word_error_rate("", "hello world");
        CHECK(r.insertions == 2 && r.ref_len == 0, "empty ref -> 2 insertions");
    }
}

// ----------------------------------------------------------------------------
// 2. Golden-fixture reproduction.
// ----------------------------------------------------------------------------

std::string read_text_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    // Trim a single trailing newline so the tokenizer doesn't see it as content.
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

std::vector<float> load_f32(const std::string & path) {
    npy_array a = npy_load(path);
    const float * p = npy_as_f32(a);
    return std::vector<float>(p, p + a.n_elements());
}

void test_fixture_reproduction(const std::string & dir) {
    // Cosine.
    {
        const std::vector<float> a = load_f32(dir + "/embed_a.npy");
        const std::vector<float> b = load_f32(dir + "/embed_b.npy");
        const double expected = load_f32(dir + "/expected_cosine.npy")[0];
        const double got = cosine_similarity(a, b);
        CHECK(close(got, expected, 1e-5),
              "fixture cosine: got %.6f expected %.6f", got, expected);
    }
    // WavLM-L3 style loss.
    {
        npy_array ref = npy_load(dir + "/hidden_ref.npy");
        npy_array tgt = npy_load(dir + "/hidden_target.npy");
        CHECK(ref.shape.size() == 2 && tgt.shape.size() == 2, "hidden states are 2-D");
        const int Tr = (int)ref.shape[0], Dr = (int)ref.shape[1];
        const int Tt = (int)tgt.shape[0], Dt = (int)tgt.shape[1];
        CHECK(Dr == Dt, "hidden feature dims match (%d vs %d)", Dr, Dt);
        const std::vector<float> hr = load_f32(dir + "/hidden_ref.npy");
        const std::vector<float> ht = load_f32(dir + "/hidden_target.npy");
        const double expected = load_f32(dir + "/expected_style_loss.npy")[0];
        const double got = wavlm_style_loss(hr, Tr, ht, Tt, Dr);
        CHECK(close(got, expected, 1e-3),
              "fixture style loss: got %.6f expected %.6f", got, expected);
    }
    // WER.
    {
        const std::string ref = read_text_file(dir + "/wer_ref.txt");
        const std::string hyp = read_text_file(dir + "/wer_hyp.txt");
        const double expected = load_f32(dir + "/expected_wer.npy")[0];
        const WerResult got = word_error_rate(ref, hyp);
        CHECK(close(got.wer, expected, 1e-5),
              "fixture wer: got %.6f expected %.6f", got.wer, expected);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    // Whole body wrapped so an unexpected throw becomes a clean test failure
    // (exit 1) rather than std::terminate with no diagnostics.
    try {
        test_cosine_known_values();
        test_cosine_rejects_bad_input();
        test_time_avg_stats_known();
        test_time_avg_stats_single_frame_zero_std();
        test_style_loss_known();
        test_wer_known_cases();

        if (argc >= 2) {
            test_fixture_reproduction(argv[1]);
        } else {
            fprintf(stderr, "WARN: no fixture dir argument; skipped fixture reproduction "
                            "(closed-form self-tests still ran)\n");
        }
    } catch (const std::exception & e) {
        ++g_failures;
        fprintf(stderr, "FAIL uncaught exception: %s\n", e.what());
    }

    fprintf(stderr, "\n%s: %d/%d checks passed\n",
            g_failures == 0 ? "PASS" : "FAIL",
            g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
