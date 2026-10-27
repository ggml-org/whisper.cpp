// Model-free unit tests for the independent-segment path behind the Core ML
// TDT decoder sidecar: the pause-cut planner (src/independent_segments.h) and
// the batched lane walk (tdt_decode_lanes, src/parakeet_tdt.cpp) driven by a
// scripted stepper instead of Core ML.
//
// They assert:
//   1. mel_frame_energy is the per-frame mean, and plan_independent_segments
//      tiles [0, n) gap-free with every segment within the cap, cut at the
//      quietest frame of the search span, and never cut without a search;
//   2. the lane walk reproduces the sequential TDT walk window by window,
//      whatever the lane count, with the embedding of the last emitted token
//      fed back and the state only advancing on an emission;
//   3. a window that emits nothing is re-decoded warm-started from its
//      predecessor's retired state, and the first window never is;
//   4. max_symbols_per_step forces the frame cursor on;
//   5. a stepper whose geometry does not match the checkpoint is refused.
//
// Exit 0 on success; non-zero (with a FAIL line per broken invariant) otherwise.

#include "independent_segments.h"
#include "parakeet_tdt.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using parakeet::SegmentRange;
using parakeet::TdtDecodeOptions;
using parakeet::TdtLaneGeometry;
using parakeet::TdtLaneStats;
using parakeet::TdtLaneStepper;
using parakeet::TdtLaneWindow;
using parakeet::mel_frame_energy;
using parakeet::plan_independent_segments;
using parakeet::tdt_decode_lanes;

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

// ── planner ───────────────────────────────────────────────────────────────

void check_energy() {
    const std::vector<float> mel = {1.0f, 3.0f,  -2.0f, 2.0f,  0.0f, 0.0f};
    const std::vector<float> energy = mel_frame_energy(mel.data(), 3, 2);
    expect(energy.size() == 3, "energy: one value per frame");
    expect(energy.size() == 3 && energy[0] == 2.0f && energy[1] == 0.0f && energy[2] == 0.0f,
           "energy: per-frame mean over the mel bins");
    expect(mel_frame_energy(mel.data(), 0, 2).empty(), "energy: no frames -> empty");
}

void check_segment_cover(const std::vector<SegmentRange> & segs, int n, int cap,
                         const std::string & tag) {
    expect(!segs.empty(), tag + ": at least one segment");
    if (segs.empty()) return;
    expect(segs.front().begin == 0, tag + ": first segment starts at 0");
    expect(segs.back().end == n, tag + ": last segment ends at n");
    for (size_t i = 0; i < segs.size(); ++i) {
        expect(segs[i].begin < segs[i].end, tag + ": empty segment");
        expect(segs[i].end - segs[i].begin <= cap, tag + ": segment exceeds the cap");
        if (i + 1 < segs.size()) {
            expect(segs[i].end == segs[i + 1].begin, tag + ": gap or overlap between segments");
        }
    }
}

void check_planner() {
    // Flat energy: with no search the cuts land exactly on the cap.
    std::vector<float> flat(100, 1.0f);
    const std::vector<SegmentRange> no_search = plan_independent_segments(flat, 100, 30, 0);
    check_segment_cover(no_search, 100, 30, "planner flat");
    expect(no_search.size() == 4 && no_search[0].end == 30 && no_search[1].end == 60 &&
           no_search[2].end == 90,
           "planner flat: cuts on the cap without a search");

    // A clear pause inside the search span attracts the cut.
    std::vector<float> energy(100, 1.0f);
    energy[25] = -5.0f;
    const std::vector<SegmentRange> paused = plan_independent_segments(energy, 100, 30, 10);
    check_segment_cover(paused, 100, 30, "planner pause");
    expect(!paused.empty() && paused[0].end == 25, "planner pause: cut at the quietest frame");
    expect(paused.size() >= 2 && paused[1].begin == 25, "planner pause: next segment starts at the cut");

    // The search never looks back past the cap minus the span, and never
    // forward: a pause just outside the span is ignored.
    energy.assign(100, 1.0f);
    energy[19] = -5.0f;
    const std::vector<SegmentRange> outside = plan_independent_segments(energy, 100, 30, 10);
    expect(!outside.empty() && outside[0].end == 30, "planner: pause outside the search span is ignored");

    // Ties keep the latest (closest to the cap) frame; a search wider than the
    // segment cannot make an empty segment.
    const std::vector<SegmentRange> wide = plan_independent_segments(flat, 100, 30, 1000);
    check_segment_cover(wide, 100, 30, "planner wide search");
    expect(!wide.empty() && wide[0].end == 30, "planner: ties keep the frame on the cap");

    // Inputs within one cap stay whole; degenerate inputs give nothing.
    const std::vector<SegmentRange> single = plan_independent_segments(flat, 30, 30, 10);
    expect(single.size() == 1 && single[0].begin == 0 && single[0].end == 30,
           "planner: input of exactly one cap is one segment");
    expect(plan_independent_segments(flat, 0, 30, 10).empty(), "planner: no frames -> no segments");
    expect(plan_independent_segments(flat, 100, 0, 10).empty(), "planner: no cap -> no segments");
}

// ── lane walk ─────────────────────────────────────────────────────────────

// Toy checkpoint: tokens 0..2, blank 3, durations {0, 1, 2}; h_pred 2, one
// layer, d_enc 1.
constexpr int kVocabPlusBlank = 4;
constexpr int kBlank          = 3;
constexpr int kHPred          = 2;
constexpr int kDEnc           = 1;

TdtLaneGeometry toy_geometry() {
    TdtLaneGeometry g;
    g.blank_id         = kBlank;
    g.vocab_plus_blank = kVocabPlusBlank;
    g.durations        = {0, 1, 2};
    g.h_pred           = kHPred;
    g.layers           = 1;
    g.d_enc            = kDEnc;
    return g;
}

parakeet::TdtEmbeddingRows f32_rows(const std::vector<float> & table) {
    const auto * first = reinterpret_cast<const uint8_t *>(table.data());
    return parakeet::tdt_embedding_rows_from_bytes(
        GGML_TYPE_F32, (int64_t) table.size() / kHPred, kHPred,
        std::vector<uint8_t>(first, first + table.size() * sizeof(float)));
}

// Row `token` is {token, 100 + token}, so an embedding fed back is traceable.
parakeet::TdtEmbeddingRows toy_embed_table() {
    std::vector<float> table;
    for (int token = 0; token < kVocabPlusBlank; ++token) {
        table.push_back((float) token);
        table.push_back(100.0f + (float) token);
    }
    return f32_rows(table);
}

// Encoder frames carry the script: value = token * 10 + duration index, so the
// stepper "decodes" a frame into (token, duration) with no model at all.
float frame_code(int token, int dur_idx) { return (float) (token * 10 + dur_idx); }

// Scripted stepper. Per lane it reads the frame code, writes argmax logits for
// its token and duration, and advances h by one so state commits are visible:
// h_out = h_in + 1 (both components), c_out = c_in. It also records the first
// embedding component it saw per dispatch per lane, which the walk must have
// set to the last emitted token (or blank).
struct ScriptedStepper : TdtLaneStepper {
    int64_t n_lanes;
    bool    emit_only_when_warm = false;  // emit only once the state is non-zero
    std::vector<float> seen_embed;         // appended per dispatch: lanes values

    explicit ScriptedStepper(int64_t lanes) : n_lanes(lanes) {}

    int64_t lanes()  const override { return n_lanes; }
    int64_t h_pred() const override { return kHPred; }
    int64_t h_dim()  const override { return kHPred; }
    int64_t d_enc()  const override { return kDEnc; }
    int64_t v_out()  const override { return kVocabPlusBlank + 3; }

    int step(const float * embed, const float * h_in, const float * c_in, const float * enc_step,
             float * logits, float * h_out, float * c_out) override {
        for (int64_t l = 0; l < n_lanes; ++l) {
            seen_embed.push_back(embed[l * kHPred]);
            const int code    = (int) enc_step[l * kDEnc];
            int       token   = code / 10;
            const int dur_idx = code % 10;
            if (emit_only_when_warm && h_in[l * kHPred] == 0.0f) token = kBlank;
            float * row = logits + l * v_out();
            std::fill(row, row + v_out(), 0.0f);
            row[token]                    = 1.0f;
            row[kVocabPlusBlank + dur_idx] = 1.0f;
            for (int k = 0; k < kHPred; ++k) {
                h_out[l * kHPred + k] = h_in[l * kHPred + k] + 1.0f;
                c_out[l * kHPred + k] = c_in[l * kHPred + k];
            }
        }
        return 0;
    }
};

// The sequential TDT walk over one scripted window, as tdt_decode_window does it.
std::vector<int32_t> sequential_walk(const std::vector<float> & frames, int max_symbols) {
    const std::vector<int> durations = {0, 1, 2};
    std::vector<int32_t> out;
    int t = 0, symbols = 0;
    while (t < (int) frames.size()) {
        const int code  = (int) frames[(size_t) t];
        const int token = code / 10;
        const int dur   = durations[(size_t) (code % 10)];
        if (token == kBlank) {
            t += std::max(1, dur);
            symbols = 0;
            continue;
        }
        out.push_back(token);
        ++symbols;
        if (dur > 0 || symbols >= max_symbols) {
            t += std::max(1, dur);
            symbols = 0;
        }
    }
    return out;
}

std::vector<float> script_a() {
    // emit 1 (dur 1), blank (dur 2), emit 2 (dur 0) then emit 0 (dur 1), blank, emit 1 (dur 2)
    return {frame_code(1, 1), frame_code(kBlank, 2), frame_code(2, 0), frame_code(0, 1),
            frame_code(kBlank, 1), frame_code(1, 2), frame_code(kBlank, 1)};
}

std::vector<float> script_b() {
    return {frame_code(kBlank, 1), frame_code(2, 1), frame_code(2, 1), frame_code(kBlank, 2),
            frame_code(0, 1)};
}

std::vector<float> script_silent() {
    return {frame_code(kBlank, 1), frame_code(kBlank, 1), frame_code(kBlank, 1)};
}

std::vector<TdtLaneWindow> windows_of(const std::vector<std::vector<float>> & scripts) {
    std::vector<TdtLaneWindow> windows;
    for (const std::vector<float> & s : scripts) {
        windows.push_back({s.data(), (int) s.size()});
    }
    return windows;
}

void check_lanes_match_sequential(int lanes) {
    const std::string tag = "lanes=" + std::to_string(lanes);
    const std::vector<std::vector<float>> scripts = {script_a(), script_b(), script_a(),
                                                     script_silent(), script_b()};
    ScriptedStepper stepper(lanes);
    std::vector<std::vector<int32_t>> out;
    TdtLaneStats stats;
    TdtDecodeOptions opts;
    const int rc = tdt_decode_lanes(toy_geometry(), toy_embed_table(), stepper, windows_of(scripts),
                                    opts, out, &stats);
    expect(rc == 0, tag + ": decode succeeds");
    expect(out.size() == scripts.size(), tag + ": one token vector per window");
    for (size_t i = 0; i < scripts.size() && i < out.size(); ++i) {
        expect(out[i] == sequential_walk(scripts[i], opts.max_symbols_per_step),
               tag + ": window " + std::to_string(i) + " matches the sequential walk");
    }
    expect(stats.dispatches > 0, tag + ": dispatches counted");
    // The silent window emits nothing, has a predecessor with a retired state,
    // and therefore gets a rescue pass; the rescue cannot make it emit here.
    expect(stats.rescued == 1, tag + ": the silent window is rescued once");
    expect(out.size() > 3 && out[3].empty(), tag + ": the silent window stays empty");
}

void check_embedding_feedback() {
    // One lane, script_a: the embedding fed to each dispatch is blank until
    // the first emission, then the last emitted token.
    const std::vector<float> script = script_a();
    ScriptedStepper stepper(1);
    std::vector<std::vector<int32_t>> out;
    const TdtDecodeOptions opts;
    expect(tdt_decode_lanes(toy_geometry(), toy_embed_table(), stepper, windows_of({script}), opts,
                            out, nullptr) == 0,
           "feedback: decode succeeds");
    // Dispatches: t0 emit 1 (dur 1) -> t1 blank (dur 2) -> t3 emit 0 (dur 1) ->
    // t4 blank -> t5 emit 1 (dur 2) -> done. Embeddings seen: blank, 1, 1, 0, 0.
    const std::vector<float> want = {(float) kBlank, 1.0f, 1.0f, 0.0f, 0.0f};
    expect(stepper.seen_embed == want, "feedback: last emitted token's row is fed back");
}

// Lane 1 is forced silent while its state is cold: the scripted stepper
// emits only once an earlier emission advanced h. Window 1 therefore emits
// nothing in the first pass and is re-decoded warm-started from window 0's
// retired state, in lane 0, where it decodes its script in full.
struct HalfWarmStepper : ScriptedStepper {
    using ScriptedStepper::ScriptedStepper;
    int step(const float * embed, const float * h_in, const float * c_in, const float * enc,
             float * logits, float * h_out, float * c_out) override {
        ScriptedStepper::step(embed, h_in, c_in, enc, logits, h_out, c_out);
        for (int64_t l = 1; l < n_lanes; ++l) {
            if (h_in[l * kHPred] == 0.0f) {
                float * row = logits + l * v_out();
                std::fill(row, row + kVocabPlusBlank, 0.0f);
                row[kBlank] = 1.0f;
            }
        }
        return 0;
    }
};

void check_rescue_warm_start() {
    const TdtDecodeOptions opts;
    std::vector<std::vector<int32_t>> out;
    TdtLaneStats stats;

    const std::vector<std::vector<float>> scripts = {script_a(), script_a()};
    HalfWarmStepper half(2);
    expect(tdt_decode_lanes(toy_geometry(), toy_embed_table(), half, windows_of(scripts), opts,
                            out, &stats) == 0,
           "rescue: decode succeeds");
    expect(out.size() == 2 && !out[0].empty(), "rescue: window 0 emits on its own");
    expect(stats.rescued == 1, "rescue: the silent second window is re-decoded once");
    expect(out.size() == 2 && out[1] == out[0], "rescue: warm-started window decodes its script");

    // A silent predecessor still retires a (cold) state, so the rescue is
    // attempted once and changes nothing; the first window has no predecessor
    // and is never rescued.
    ScriptedStepper cold(2);
    cold.emit_only_when_warm = true;
    expect(tdt_decode_lanes(toy_geometry(), toy_embed_table(), cold, windows_of(scripts), opts,
                            out, &stats) == 0,
           "rescue: cold decode succeeds");
    expect(out.size() == 2 && out[0].empty() && out[1].empty(),
           "rescue: a cold seed leaves both windows silent");
    expect(stats.rescued == 1, "rescue: only the second window is tried");
}

void check_max_symbols() {
    // Three zero-duration emissions on one frame, then a blank: with
    // max_symbols_per_step 2 the cursor is forced on after two.
    const std::vector<float> script = {frame_code(1, 0), frame_code(kBlank, 1)};
    ScriptedStepper stepper(1);
    std::vector<std::vector<int32_t>> out;
    TdtDecodeOptions opts;
    opts.max_symbols_per_step = 2;
    expect(tdt_decode_lanes(toy_geometry(), toy_embed_table(), stepper, windows_of({script}), opts,
                            out, nullptr) == 0,
           "max symbols: decode succeeds");
    expect(out.size() == 1 && out[0] == std::vector<int32_t>{1, 1},
           "max symbols: two emissions per frame, then the cursor moves on");
    expect(out.size() == 1 && out[0] == sequential_walk(script, 2),
           "max symbols: matches the sequential walk");
}

void check_geometry_mismatch() {
    struct WrongWidth : ScriptedStepper {
        using ScriptedStepper::ScriptedStepper;
        int64_t h_pred() const override { return kHPred + 1; }
    };
    WrongWidth wrong(1);
    std::vector<std::vector<int32_t>> out;
    const std::vector<float> script = script_a();
    const TdtDecodeOptions opts;
    expect(tdt_decode_lanes(toy_geometry(), toy_embed_table(), wrong, windows_of({script}), opts,
                            out, nullptr) != 0,
           "geometry: a stepper with the wrong h_pred is refused");

    ScriptedStepper ok(1);
    const parakeet::TdtEmbeddingRows short_table = f32_rows(std::vector<float>(kHPred, 0.0f));
    expect(tdt_decode_lanes(toy_geometry(), short_table, ok, windows_of({script}), opts,
                            out, nullptr) != 0,
           "geometry: a short embedding table is refused");
}

}  // namespace

int main() {
    check_energy();
    check_planner();
    check_lanes_match_sequential(1);
    check_lanes_match_sequential(2);
    check_lanes_match_sequential(16);
    check_embedding_feedback();
    check_rescue_warm_start();
    check_max_symbols();
    check_geometry_mismatch();

    if (g_failures == 0) {
        std::printf("test-independent-segments: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test-independent-segments: %d failure(s)\n", g_failures);
    return 1;
}
