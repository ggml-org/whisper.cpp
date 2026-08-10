// Model-free unit tests for multilingual CTC language masking:
// find_ctc_language_range, resolve_ctc_decode_options, and masked
// ctc_greedy_decode (range argmax, blank outside range, repeat collapse,
// invalid range fallback to full vocab).
//
// Exit 0 on success; non-zero with FAIL lines otherwise.

#include "parakeet_ctc.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using parakeet::CtcDecodeOptions;
using parakeet::ParakeetCtcModel;
using parakeet::ctc_greedy_decode;
using parakeet::find_ctc_language_range;
using parakeet::resolve_ctc_decode_options;

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

void expect_eq_vec(const std::vector<int32_t> & got,
                   const std::vector<int32_t> & want,
                   const std::string & what) {
    if (got != want) {
        std::string detail = what + ": got=[";
        for (size_t i = 0; i < got.size(); ++i) {
            if (i) detail += ",";
            detail += std::to_string(got[i]);
        }
        detail += "] want=[";
        for (size_t i = 0; i < want.size(); ++i) {
            if (i) detail += ",";
            detail += std::to_string(want[i]);
        }
        detail += "]";
        fail(detail);
    }
}

ParakeetCtcModel make_multilingual_model() {
    ParakeetCtcModel model;
    model.vocab_size = 8;
    model.blank_id = 7;
    model.ctc_lang_ranges = {
        {"hi", 0, 3},
        {"ta", 3, 6},
    };
    return model;
}

ParakeetCtcModel make_monolingual_model() {
    ParakeetCtcModel model;
    model.vocab_size = 8;
    model.blank_id = 7;
    return model;
}

// One row: vocab_size floats. Higher score = more likely.
void set_row(std::vector<float> & logits, int frame, int vocab, int best_id, float score = 1.0f) {
    float * row = logits.data() + static_cast<size_t>(frame) * vocab;
    for (int i = 0; i < vocab; ++i) row[i] = 0.0f;
    row[best_id] = score;
}

void test_language_lookup() {
    const ParakeetCtcModel multi = make_multilingual_model();
    int32_t start = -1;
    int32_t end = -1;
    expect(find_ctc_language_range(multi, "hi", start, end), "lookup hi");
    expect(start == 0 && end == 3, "hi range [0,3)");
    expect(find_ctc_language_range(multi, "ta", start, end), "lookup ta");
    expect(start == 3 && end == 6, "ta range [3,6)");
    expect(!find_ctc_language_range(multi, "gu", start, end), "unknown lang");
    expect(!find_ctc_language_range(multi, "", start, end), "empty lang");

    const ParakeetCtcModel mono = make_monolingual_model();
    expect(!find_ctc_language_range(mono, "hi", start, end), "mono has no ranges");
}

void test_resolve_options() {
    const ParakeetCtcModel multi = make_multilingual_model();
    const ParakeetCtcModel mono = make_monolingual_model();

    bool threw = false;
    try {
        (void) resolve_ctc_decode_options(multi, "");
    } catch (const std::runtime_error &) {
        threw = true;
    }
    expect(threw, "multi empty language must throw");

    threw = false;
    try {
        (void) resolve_ctc_decode_options(multi, "gu");
    } catch (const std::runtime_error &) {
        threw = true;
    }
    expect(threw, "multi unknown language must throw");

    const CtcDecodeOptions hi = resolve_ctc_decode_options(multi, "hi");
    expect(hi.token_start == 0 && hi.token_end == 3, "resolve hi range");

    const CtcDecodeOptions ignored = resolve_ctc_decode_options(mono, "hi");
    expect(ignored.token_end < 0 || ignored.token_end <= ignored.token_start,
           "mono language ignored -> full vocab opts");

    const CtcDecodeOptions empty_mono = resolve_ctc_decode_options(mono, "");
    expect(empty_mono.token_end < 0 || empty_mono.token_end <= empty_mono.token_start,
           "mono empty language ok");
}

void test_masked_decode_range_and_blank() {
    constexpr int vocab = 8;
    constexpr int32_t blank = 7;
    constexpr int n_frames = 4;
    std::vector<float> logits(static_cast<size_t>(n_frames) * vocab, 0.0f);

    // Frame 0: in-range hi token 1 wins over out-of-range 5.
    set_row(logits, 0, vocab, 5, 2.0f);
    logits[0 * vocab + 1] = 1.5f;
    // Frame 1: same token 1 again (must collapse while prev is still 1).
    set_row(logits, 1, vocab, 1, 2.0f);
    // Frame 2: blank outside range beats in-range token.
    set_row(logits, 2, vocab, 1, 1.0f);
    logits[2 * vocab + blank] = 3.0f;
    // Frame 3: new token 2 (allowed after blank resets prev).
    set_row(logits, 3, vocab, 2, 2.0f);

    CtcDecodeOptions hi;
    hi.token_start = 0;
    hi.token_end = 3;
    const std::vector<int32_t> got =
        ctc_greedy_decode(logits.data(), n_frames, vocab, blank, &hi);
    // Frame0 emits 1 (not 5), frame1 collapses, frame2 blank, frame3 emits 2.
    expect_eq_vec(got, {1, 2}, "masked hi decode");
}

void test_masked_decode_invalid_range_falls_back() {
    constexpr int vocab = 8;
    constexpr int32_t blank = 7;
    std::vector<float> logits(static_cast<size_t>(2) * vocab, 0.0f);
    set_row(logits, 0, vocab, 5, 2.0f);
    set_row(logits, 1, vocab, blank, 2.0f);

    CtcDecodeOptions bad;
    bad.token_start = 4;
    bad.token_end = 4; // empty -> decoder falls back to full vocab
    const std::vector<int32_t> got =
        ctc_greedy_decode(logits.data(), 2, vocab, blank, &bad);
    expect_eq_vec(got, {5}, "invalid range falls back to full-vocab argmax");
}

void test_full_vocab_unchanged() {
    constexpr int vocab = 8;
    constexpr int32_t blank = 7;
    std::vector<float> logits(static_cast<size_t>(3) * vocab, 0.0f);
    set_row(logits, 0, vocab, 4, 2.0f);
    set_row(logits, 1, vocab, 4, 2.0f);
    set_row(logits, 2, vocab, blank, 2.0f);
    const std::vector<int32_t> got =
        ctc_greedy_decode(logits.data(), 3, vocab, blank, nullptr);
    expect_eq_vec(got, {4}, "full-vocab collapse");
}

}  // namespace

int main() {
    test_language_lookup();
    test_resolve_options();
    test_masked_decode_range_and_blank();
    test_masked_decode_invalid_range_falls_back();
    test_full_vocab_unchanged();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "ok: ctc language mask unit tests passed\n");
    return 0;
}
