// Sortformer v2.1 AOSC speaker-correctness regression.
//
// Runs the streaming Sortformer engine with the default AOSC config on
// a multi-speaker re-entry fixture (abcba / abcdba) and asserts three
// invariants against the RTTM ground truth:
//
//   1. Speaker coverage     — every reference speaker has at least one
//                             frame covered by some emitted hyp.
//   2. Re-entry slot        — every reference speaker that appears in
//      continuity             multiple non-contiguous segments lands in
//                             the SAME hyp_<id> across all of those
//                             segments. This is the AOSC contract.
//   3. DER ceiling          — frame-level (10 ms grid) confusion + miss +
//                             false-alarm rate under the optimal
//                             hyp -> ref permutation is below
//                             `--der-max` (default 30 %).
//
// Speakers themselves are identified by their natural ground-truth label
// (A / B / C / D in the canonical fixtures). The test does not gate on
// perfect per-frame identity — encoder-side acoustic similarity between
// two voices can legitimately confuse one for another even with the
// cache running correctly. The three checks above isolate the AOSC
// contract (cache tracks slot continuity over time) from the upstream
// model-quality question.
//
// Usage:
//   test-sortformer-aosc-speakers --model <gguf> --wav <wav>
//                                 --ref-rttm <rttm>
//                                 [--chunk-ms 2000]
//                                 [--der-max 30.0]
//
// Exit codes:
//    0 = PASS  (all three invariants satisfied)
//    2 = bad CLI / missing required arg
//   11 = ref RTTM unreadable / empty
//   13 = WAV file unreadable / not 16 kHz mono s16le
//   14 = engine reports the GGUF isn't a Sortformer model
//   20 = speaker coverage failed (one or more ref speakers had zero
//        frames covered by any hyp)
//   21 = re-entry slot continuity failed (an AOSC contract break — a
//        speaker that returned was rebound to a different hyp_<id>)
//   22 = DER ceiling exceeded
//   30 = no hyp segments emitted at all
//
//   0 also covers SKIP-equivalent (missing model / wav / rttm at
//   runtime) so this binary behaves the same way the other parakeet-cpp
//   ctest fixtures behave when their fixtures aren't on disk.

#include "parakeet/engine.h"
#include "test_utils.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr double FRAME_S = 0.01;  // 10 ms grid

using parakeet_test::file_exists;
using parakeet_test::load_wav_pcm16le_mono;

struct RttmSeg {
    double      start_s;
    double      dur_s;
    std::string speaker;
};

// Minimal RTTM v1 parser. Reads only `SPEAKER` lines and pulls out the
// 4th, 5th and 8th tokens (start_s, duration_s, speaker_label).
// Comments (lines starting with `;`) and blank lines are ignored. Other
// RTTM types (LEXEME, SPKR-INFO, NOSCORE, ...) are skipped.
std::vector<RttmSeg> parse_rttm(const std::string & path) {
    std::vector<RttmSeg> out;
    std::ifstream f(path);
    if (!f) return out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == ';') continue;
        std::istringstream iss(line);
        std::string type, uri, channel, start_str, dur_str, ortho, stype, spk;
        if (!(iss >> type)) continue;
        if (type != "SPEAKER") continue;
        if (!(iss >> uri >> channel >> start_str >> dur_str >> ortho >> stype >> spk)) {
            continue;
        }
        RttmSeg s;
        s.start_s = std::stod(start_str);
        s.dur_s   = std::stod(dur_str);
        s.speaker = spk;
        out.push_back(std::move(s));
    }
    return out;
}

// Pre-built frame timeline at 10 ms grid. -1 = silence, otherwise the
// remapped integer speaker id (assignment order = first appearance in
// the segment list).
struct Timeline {
    std::vector<int>                       frame_to_id;  // size n_frames
    std::unordered_map<std::string, int>   label_to_id;
    std::vector<std::string>               id_to_label;
};

Timeline timeline_from_ref(const std::vector<RttmSeg> & segs, int n_frames) {
    Timeline t;
    t.frame_to_id.assign(n_frames, -1);
    for (const auto & s : segs) {
        auto it = t.label_to_id.find(s.speaker);
        int id;
        if (it == t.label_to_id.end()) {
            id = (int) t.id_to_label.size();
            t.label_to_id[s.speaker] = id;
            t.id_to_label.push_back(s.speaker);
        } else {
            id = it->second;
        }
        const int s_frame = std::max(0, (int) (s.start_s / FRAME_S));
        const int e_frame = std::min(n_frames,
                                     (int) ((s.start_s + s.dur_s) / FRAME_S));
        for (int i = s_frame; i < e_frame; ++i) t.frame_to_id[i] = id;
    }
    return t;
}

struct HypTimeline {
    std::vector<int> frame_to_id;  // -1 silence, otherwise raw hyp_id
    int              n_speakers;   // max hyp_id + 1 across emitted segs
};

HypTimeline timeline_from_hyp(
    const std::vector<parakeet::StreamingDiarizationSegment> & segs,
    int n_frames) {

    HypTimeline t;
    t.frame_to_id.assign(n_frames, -1);
    int max_id = -1;
    for (const auto & s : segs) {
        if (s.speaker_id < 0) continue;
        const int s_frame = std::max(0, (int) (s.start_s / FRAME_S));
        const int e_frame = std::min(n_frames, (int) (s.end_s / FRAME_S));
        for (int i = s_frame; i < e_frame; ++i) {
            t.frame_to_id[i] = s.speaker_id;
        }
        if (s.speaker_id > max_id) max_id = s.speaker_id;
    }
    t.n_speakers = max_id + 1;
    return t;
}

// Brute-force the hyp_id -> ref_id permutation that maximises the
// number of frames where the assigned ref_id matches the actual ref_id.
// Returns a vector mapping hyp_id to ref_id (or -1 if the hyp is
// spurious / unassigned). K_hyp * K_ref! work, fine for K <= 4.
std::vector<int> best_perm(const Timeline    & ref,
                           const HypTimeline & hyp) {
    const int K_ref = (int) ref.id_to_label.size();
    const int K_hyp = hyp.n_speakers;

    // co[(h, r)] = # frames where ref=r AND hyp=h
    std::vector<int> co(K_hyp * K_ref, 0);
    const int n_frames = (int) ref.frame_to_id.size();
    for (int i = 0; i < n_frames; ++i) {
        const int r = ref.frame_to_id[i];
        const int h = hyp.frame_to_id[i];
        if (r < 0 || h < 0) continue;
        co[h * K_ref + r]++;
    }

    // Enumerate all permutations of ref ids; for each, sum the matching
    // co counts for the first K_pick hyp ids (extra hyps map to -1).
    const int K_pick = std::min(K_hyp, K_ref);
    std::vector<int> ref_perm(K_ref);
    std::iota(ref_perm.begin(), ref_perm.end(), 0);

    int best_correct = -1;
    std::vector<int> best(K_hyp, -1);

    do {
        int correct = 0;
        for (int h = 0; h < K_pick; ++h) {
            correct += co[h * K_ref + ref_perm[h]];
        }
        if (correct > best_correct) {
            best_correct = correct;
            std::fill(best.begin(), best.end(), -1);
            for (int h = 0; h < K_pick; ++h) best[h] = ref_perm[h];
        }
    } while (std::next_permutation(ref_perm.begin(), ref_perm.end()));

    return best;
}

// Per-ref-segment dominant hyp_id. Returns -1 if the segment was
// entirely uncovered by any hyp.
int dominant_hyp_in_range(const HypTimeline & hyp,
                          double start_s, double dur_s) {
    const int s_frame = std::max(0, (int) (start_s / FRAME_S));
    const int e_frame = std::min((int) hyp.frame_to_id.size(),
                                 (int) ((start_s + dur_s) / FRAME_S));
    std::unordered_map<int, int> counts;
    int best_id = -1;
    int best_cnt = 0;
    for (int i = s_frame; i < e_frame; ++i) {
        const int h = hyp.frame_to_id[i];
        if (h < 0) continue;
        int c = ++counts[h];
        if (c > best_cnt) {
            best_cnt = c;
            best_id = h;
        }
    }
    return best_id;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string gguf;
    std::string wav;
    std::string ref_rttm;
    int    chunk_ms = 2000;
    double der_max  = 30.0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--model"    && i + 1 < argc) gguf     = argv[++i];
        else if (a == "--wav"      && i + 1 < argc) wav      = argv[++i];
        else if (a == "--ref-rttm" && i + 1 < argc) ref_rttm = argv[++i];
        else if (a == "--chunk-ms" && i + 1 < argc) chunk_ms = std::atoi(argv[++i]);
        else if (a == "--der-max"  && i + 1 < argc) der_max  = std::atof(argv[++i]);
        else {
            std::fprintf(stderr,
                "[aosc-spk-test] unknown / incomplete option: %s\n", a.c_str());
            return 2;
        }
    }
    if (gguf.empty() || wav.empty() || ref_rttm.empty()) {
        std::fprintf(stderr,
            "[aosc-spk-test] Usage: --model <gguf> --wav <wav> "
            "--ref-rttm <rttm> [--chunk-ms 2000] [--der-max 30.0]\n");
        return 2;
    }

    if (!file_exists(gguf) || !file_exists(wav) || !file_exists(ref_rttm)) {
        std::fprintf(stderr,
            "[aosc-spk-test] SKIP: fixture missing (model=%s%s wav=%s%s rttm=%s%s)\n",
            gguf.c_str(),     file_exists(gguf)     ? "" : " (missing)",
            wav.c_str(),      file_exists(wav)      ? "" : " (missing)",
            ref_rttm.c_str(), file_exists(ref_rttm) ? "" : " (missing)");
        return 0;
    }

    std::vector<float> samples; int sr = 0;
    if (!load_wav_pcm16le_mono(wav, samples, sr)) {
        std::fprintf(stderr, "[aosc-spk-test] FAIL: could not load wav %s\n", wav.c_str());
        return 13;
    }

    auto ref_segs = parse_rttm(ref_rttm);
    if (ref_segs.empty()) {
        std::fprintf(stderr,
            "[aosc-spk-test] FAIL: reference rttm empty / unreadable %s\n",
            ref_rttm.c_str());
        return 11;
    }

    parakeet::EngineOptions eopts;
    eopts.model_gguf_path = gguf;
    eopts.verbose         = false;
    parakeet::Engine engine(eopts);
    if (!engine.is_diarization_model()) {
        std::fprintf(stderr,
            "[aosc-spk-test] FAIL: %s isn't a Sortformer model\n", gguf.c_str());
        return 14;
    }

    // Defaults pull the new AOSC config (spkcache_enable=true, fifo_len=188,
    // chunk_left_context_ms=80, chunk_right_context_ms=560, etc.) from
    // the public SortformerStreamingOptions struct. We only override the
    // bits that follow the WAV + the CLI chunk knob.
    parakeet::SortformerStreamingOptions sopts;
    sopts.sample_rate    = sr;
    sopts.chunk_ms       = chunk_ms;
    // min_segment_ms 200 matches the other streaming test; otherwise
    // very-short transient segments inflate the segment count without
    // contributing to the speaker-correctness verdict.
    sopts.min_segment_ms = 200;

    std::fprintf(stderr,
        "[aosc-spk-test] model=%s wav=%s samples=%zu sr=%d chunk_ms=%d "
        "der_max=%.2f%% (AOSC: spkcache=%d len=%d fifo=%d)\n",
        gguf.c_str(), wav.c_str(), samples.size(), sr, chunk_ms, der_max,
        (int) sopts.spkcache_enable, sopts.spkcache_len, sopts.fifo_len);

    std::vector<parakeet::StreamingDiarizationSegment> hyp_segs;
    auto on_seg = [&](const parakeet::StreamingDiarizationSegment & s) {
        if (s.speaker_id < 0) return;
        if (s.end_s <= s.start_s) return;
        hyp_segs.push_back(s);
    };

    auto session = engine.diarize_start(sopts, on_seg);
    const int feed_samples = std::max(1, (sr * chunk_ms) / 1000);
    size_t off = 0;
    while (off < samples.size()) {
        const int n = std::min(feed_samples, (int) (samples.size() - off));
        session->feed_pcm_f32(samples.data() + off, n);
        off += n;
    }
    try { session->finalize(); } catch (...) { /* same as streaming test */ }

    if (hyp_segs.empty()) {
        std::fprintf(stderr,
            "[aosc-spk-test] FAIL: engine emitted no diarization segments\n");
        return 30;
    }

    // Build frame timelines.
    const double audio_s = (double) samples.size() / sr;
    const int    n_frames = (int) (audio_s / FRAME_S) + 1;
    const Timeline    ref = timeline_from_ref(ref_segs, n_frames);
    const HypTimeline hyp = timeline_from_hyp(hyp_segs, n_frames);
    const int K_ref = (int) ref.id_to_label.size();
    const int K_hyp = hyp.n_speakers;
    std::fprintf(stderr,
        "[aosc-spk-test] ref speakers=%d  hyp speakers emitted=%d  audio=%.2fs\n",
        K_ref, K_hyp, audio_s);

    // ── Assertion 1: speaker coverage ─────────────────────────────────
    std::vector<int> ref_speech_frames(K_ref, 0);
    std::vector<int> ref_covered_frames(K_ref, 0);
    for (int i = 0; i < n_frames; ++i) {
        const int r = ref.frame_to_id[i];
        if (r < 0) continue;
        ref_speech_frames[r]++;
        if (hyp.frame_to_id[i] >= 0) ref_covered_frames[r]++;
    }
    bool coverage_ok = true;
    for (int rid = 0; rid < K_ref; ++rid) {
        const double pct = ref_speech_frames[rid] > 0
            ? 100.0 * ref_covered_frames[rid] / ref_speech_frames[rid]
            : 0.0;
        std::fprintf(stderr,
            "[aosc-spk-test]   coverage: ref '%s'  %d / %d frames (%.1f%%)\n",
            ref.id_to_label[rid].c_str(),
            ref_covered_frames[rid], ref_speech_frames[rid], pct);
        if (ref_covered_frames[rid] == 0) coverage_ok = false;
    }
    if (!coverage_ok) {
        std::fprintf(stderr,
            "[aosc-spk-test] FAIL[coverage]: at least one ref speaker has 0 hyp frames\n");
        return 20;
    }

    // ── Assertion 2: re-entry slot continuity ─────────────────────────
    // Compute dominant hyp_id per ref segment; collect by ref speaker;
    // require every ref speaker's set of dominant hyp_ids to be a
    // singleton.
    std::unordered_map<std::string, std::vector<int>> per_speaker_hyps;
    std::unordered_map<std::string, int>              per_speaker_segcount;
    for (const auto & rs : ref_segs) {
        per_speaker_segcount[rs.speaker]++;
        const int dom = dominant_hyp_in_range(hyp, rs.start_s, rs.dur_s);
        if (dom < 0) continue;
        auto & vec = per_speaker_hyps[rs.speaker];
        if (std::find(vec.begin(), vec.end(), dom) == vec.end()) {
            vec.push_back(dom);
        }
    }
    bool continuity_ok = true;
    for (const auto & kv : per_speaker_segcount) {
        const std::string & spk = kv.first;
        const int n_segs = kv.second;
        const auto & doms = per_speaker_hyps[spk];
        std::ostringstream dom_str;
        for (size_t i = 0; i < doms.size(); ++i) {
            if (i) dom_str << ",";
            dom_str << "hyp_" << doms[i];
        }
        std::fprintf(stderr,
            "[aosc-spk-test]   continuity: ref '%s'  %d segment(s)  -> {%s}\n",
            spk.c_str(), n_segs, dom_str.str().c_str());
        if (n_segs >= 2 && doms.size() > 1) continuity_ok = false;
    }
    if (!continuity_ok) {
        std::fprintf(stderr,
            "[aosc-spk-test] FAIL[continuity]: a re-entering speaker was "
            "rebound to a different hyp_<id> — AOSC contract broken\n");
        return 21;
    }

    // ── Assertion 3: DER ceiling ──────────────────────────────────────
    const std::vector<int> perm = best_perm(ref, hyp);
    int miss = 0, fa = 0, conf = 0, ref_total = 0;
    for (int i = 0; i < n_frames; ++i) {
        const int r = ref.frame_to_id[i];
        const int h = hyp.frame_to_id[i];
        if (r >= 0) {
            ref_total++;
            if (h < 0) {
                miss++;
            } else if (h >= (int) perm.size() || perm[h] != r) {
                conf++;
            }
        } else if (h >= 0) {
            fa++;
        }
    }
    const double der = ref_total > 0
        ? 100.0 * (miss + fa + conf) / ref_total
        : 0.0;
    std::fprintf(stderr,
        "[aosc-spk-test]   DER: %.2f%%  miss=%.2fs  fa=%.2fs  conf=%.2fs  ref=%.2fs\n",
        der, miss * FRAME_S, fa * FRAME_S, conf * FRAME_S, ref_total * FRAME_S);
    std::ostringstream perm_str;
    for (int h = 0; h < (int) perm.size(); ++h) {
        if (h) perm_str << ", ";
        perm_str << "hyp_" << h << "->";
        if (perm[h] < 0) perm_str << "(unassigned)";
        else perm_str << ref.id_to_label[perm[h]];
    }
    std::fprintf(stderr,
        "[aosc-spk-test]   mapping: %s\n", perm_str.str().c_str());

    if (der > der_max) {
        std::fprintf(stderr,
            "[aosc-spk-test] FAIL[DER]: %.2f%% > ceiling %.2f%%\n", der, der_max);
        return 22;
    }

    std::fprintf(stderr, "[aosc-spk-test] PASS\n");
    return 0;
}
