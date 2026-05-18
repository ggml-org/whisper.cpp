// Sortformer streaming session sanity (Mode 3-style chunking).
//
// Usage:
//   test-sortformer-streaming [--model <gguf>] [--wav <wav>]
//                             [--history-ms <ms>] [--chunk-ms <ms>]
//                             [--rttm-out <path>]
//
// Exit 0 on success or skip when defaults missing; non-zero on failure.
//
// `--history-ms` and `--chunk-ms` override the default streaming knobs
// (30000 / 2000); used to reproduce drift scenarios at different
// rolling-window sizes. `--rttm-out` writes a NIST RTTM hypothesis
// file of every emitted real streaming segment (terminators and
// `is_final` synthetic markers excluded). The URI column is taken
// from the WAV path's stem so the JS benchmark's DER evaluator can
// ingest the file directly against the matching reference RTTM.

#include "parakeet/engine.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool file_exists(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

bool load_wav_pcm16le_mono(const std::string & path, std::vector<float> & samples, int & sample_rate) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char riff[4]; f.read(riff, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0) return false;
    f.ignore(4);
    char wave[4]; f.read(wave, 4);
    if (std::memcmp(wave, "WAVE", 4) != 0) return false;

    bool fmt_ok = false; uint16_t channels = 0; uint16_t bits = 0; uint32_t srate = 0;
    std::vector<char> data;
    while (f) {
        char id[4]; f.read(id, 4);
        if (!f) break;
        uint32_t sz = 0; f.read((char *) &sz, 4);
        if (std::memcmp(id, "fmt ", 4) == 0) {
            std::vector<char> hdr(sz);
            f.read(hdr.data(), sz);
            uint16_t fmt = *(uint16_t *) hdr.data();
            channels    = *(uint16_t *) (hdr.data() + 2);
            srate       = *(uint32_t *) (hdr.data() + 4);
            bits        = *(uint16_t *) (hdr.data() + 14);
            if (fmt != 1 || channels != 1 || bits != 16) return false;
            fmt_ok = true;
        } else if (std::memcmp(id, "data", 4) == 0) {
            data.resize(sz);
            f.read(data.data(), sz);
            break;
        } else {
            f.ignore(sz);
        }
    }
    if (!fmt_ok || data.empty()) return false;
    sample_rate = (int) srate;
    const int n = (int) (data.size() / 2);
    samples.resize(n);
    const int16_t * s16 = reinterpret_cast<const int16_t *>(data.data());
    for (int i = 0; i < n; ++i) samples[i] = (float) s16[i] / 32768.0f;
    return true;
}

using namespace parakeet;

int run_basic(const std::string & gguf_path,
              const std::string & wav_path,
              int history_ms,
              int chunk_ms,
              const std::string & rttm_out_path,
              bool  spkcache_enable,
              int   spkcache_len,
              int   fifo_len,
              float threshold) {

    std::vector<float> samples; int sr = 0;
    if (!load_wav_pcm16le_mono(wav_path, samples, sr)) {
        std::fprintf(stderr, "[sf-stream-test] could not load wav %s\n", wav_path.c_str());
        return 1;
    }
    std::fprintf(stderr, "[sf-stream-test] wav=%s samples=%zu sr=%d\n",
                 wav_path.c_str(), samples.size(), sr);

    EngineOptions eopts;
    eopts.model_gguf_path = gguf_path;
    eopts.verbose         = false;
    Engine engine(eopts);
    if (!engine.is_diarization_model()) {
        std::fprintf(stderr, "[sf-stream-test] %s is not a Sortformer model\n", gguf_path.c_str());
        return 2;
    }

    DiarizationResult offline = engine.diarize_samples(
        samples.data(), (int) samples.size(), sr, {});
    std::fprintf(stderr, "[sf-stream-test] offline segments=%zu\n", offline.segments.size());
    for (const auto & s : offline.segments) {
        std::fprintf(stderr, "  offline [%.2f-%.2f] speaker_%d\n",
                     s.start_s, s.end_s, s.speaker_id);
    }

    int n_real_callbacks = 0;
    int n_terminators    = 0;
    int n_finals         = 0;
    double max_end       = 0.0;
    int max_chunk_index  = -1;
    std::vector<StreamingDiarizationSegment> all;

    auto on_seg = [&](const StreamingDiarizationSegment & s) {
        if (s.speaker_id < 0) {
            ++n_terminators;
        } else {
            ++n_real_callbacks;
            if (s.end_s > max_end) max_end = s.end_s;
        }
        if (s.is_final) ++n_finals;
        if (s.chunk_index > max_chunk_index) max_chunk_index = s.chunk_index;
        all.push_back(s);
    };

    SortformerStreamingOptions sopts;
    sopts.sample_rate    = sr;
    sopts.chunk_ms       = chunk_ms;
    sopts.history_ms     = history_ms;
    sopts.threshold      = threshold;
    sopts.min_segment_ms = 200;
    sopts.spkcache_enable = spkcache_enable;
    sopts.spkcache_len    = spkcache_len;
    sopts.fifo_len        = fifo_len;
    std::fprintf(stderr,
        "[sf-stream-test] streaming opts: chunk_ms=%d history_ms=%d "
        "spkcache_enable=%d spkcache_len=%d fifo_len=%d threshold=%.2f\n",
        sopts.chunk_ms, sopts.history_ms,
        (int) sopts.spkcache_enable, sopts.spkcache_len, sopts.fifo_len,
        (double) sopts.threshold);

    int n_vad_events       = 0;
    int n_speaking_events  = 0;
    sopts.on_event = [&](const StreamEvent & ev) {
        if (ev.type == StreamEventType::VadStateChanged) {
            ++n_vad_events;
            if (ev.vad_state == VadState::Speaking) ++n_speaking_events;
            std::fprintf(stderr,
                "[sf-stream-test] EVT VadStateChanged @ %.2fs chunk=%d -> %s "
                "speaker_id=%d score=%.3f\n",
                ev.timestamp_s, ev.chunk_index,
                ev.vad_state == VadState::Speaking ? "Speaking" :
                ev.vad_state == VadState::Silent   ? "Silent"   : "Unknown",
                ev.speaker_id, ev.vad_score);
        }
    };

    auto session = engine.diarize_start(sopts, on_seg);

    std::mt19937 rng(0xC0FFEE);
    std::uniform_int_distribution<int> burst_dist(1, 5000);
    size_t off = 0;
    while (off < samples.size()) {
        const int n = std::min<int>(burst_dist(rng), (int) (samples.size() - off));
        session->feed_pcm_f32(samples.data() + off, n);
        off += n;
    }
    session->finalize();

    std::fprintf(stderr,
        "[sf-stream-test] streaming real=%d terminators=%d final_flags=%d max_end=%.3fs chunks=%d\n",
        n_real_callbacks, n_terminators, n_finals, max_end, max_chunk_index + 1);

    if (!rttm_out_path.empty()) {
        std::ofstream rttm(rttm_out_path);
        if (!rttm) {
            std::fprintf(stderr,
                "[sf-stream-test] FAIL: could not open --rttm-out path %s\n",
                rttm_out_path.c_str());
            return 11;
        }
        // Derive a stable URI from the WAV path's stem (filename without
        // dirs and without the .wav extension). The JS benchmark's DER
        // evaluator matches reference and hypothesis by this URI.
        std::string uri = wav_path;
        const size_t slash = uri.find_last_of("/\\");
        if (slash != std::string::npos) uri = uri.substr(slash + 1);
        const size_t dot = uri.find_last_of('.');
        if (dot != std::string::npos) uri = uri.substr(0, dot);

        int dumped = 0;
        for (const auto & s : all) {
            // Skip the synthetic terminator (speaker_id < 0) and zero-
            // length is_final markers. Real trailing-chunk segments have
            // is_final=true AND speaker_id>=0 AND positive duration --
            // those we keep.
            if (s.speaker_id < 0) continue;
            const double dur = s.end_s - s.start_s;
            if (dur <= 0.0) continue;
            char line[256];
            std::snprintf(line, sizeof(line),
                "SPEAKER %s 1 %.3f %.3f <NA> <NA> hyp_%d <NA> <NA>\n",
                uri.c_str(), s.start_s, dur, s.speaker_id);
            rttm << line;
            ++dumped;
        }
        rttm.close();
        std::fprintf(stderr,
            "[sf-stream-test] wrote %d hypothesis segments to %s (uri=%s)\n",
            dumped, rttm_out_path.c_str(), uri.c_str());
    }

    if (n_real_callbacks == 0) {
        std::fprintf(stderr, "[sf-stream-test] FAIL: no real segments emitted\n");
        return 3;
    }
    if (n_finals != 1) {
        std::fprintf(stderr,
            "[sf-stream-test] FAIL: expected exactly one is_final callback, got %d\n",
            n_finals);
        return 4;
    }
    {
        int dup = 0;
        for (size_t i = 1; i < all.size(); ++i) {
            const auto & a = all[i - 1];
            const auto & b = all[i];
            if (a.speaker_id == b.speaker_id &&
                std::abs(a.start_s - b.start_s) < 1e-6 &&
                std::abs(a.end_s   - b.end_s)   < 1e-6) {
                ++dup;
            }
        }
        if (dup > 0) {
            std::fprintf(stderr,
                "[sf-stream-test] FAIL: %d duplicate segment(s) detected\n", dup);
            return 7;
        }
    }
    const double audio_s = (double) samples.size() / sr;
    if (max_end > audio_s + 0.5) {
        std::fprintf(stderr, "[sf-stream-test] FAIL: max_end=%.3f > audio=%.3f\n",
                     max_end, audio_s);
        return 5;
    }
    if (max_end < audio_s - 5.0) {
        std::fprintf(stderr, "[sf-stream-test] FAIL: max_end=%.3f << audio=%.3f (lost segments?)\n",
                     max_end, audio_s);
        return 6;
    }

    // Phase 13: at least one VadStateChanged event should fire on a
    // wav that contains speech. We don't gate the exact count -- on
    // single-speaker fixtures it's commonly just two (Unknown ->
    // Speaking on chunk 0, possibly Speaking -> Silent on the trailing
    // silence chunk) -- but zero events on a wav with audible speech
    // means the event plumbing is broken.
    if (n_vad_events == 0) {
        std::fprintf(stderr,
            "[sf-stream-test] FAIL: no VadStateChanged events on a wav "
            "with audible speech (Phase 13 plumbing broken?)\n");
        return 9;
    }
    if (n_speaking_events == 0) {
        std::fprintf(stderr,
            "[sf-stream-test] FAIL: no Speaking transitions among %d "
            "VadStateChanged events\n", n_vad_events);
        return 10;
    }

    {
        auto session2 = engine.diarize_start(sopts, [](const StreamingDiarizationSegment &) {});
        session2->feed_pcm_f32(samples.data(), (int) std::min<size_t>(samples.size(), 4 * sr));
        session2->cancel();
        session2->cancel();
    }

    std::fprintf(stderr, "[sf-stream-test] PASS\n");
    return 0;
}

}

int main(int argc, char ** argv) {
    std::string gguf = "models/sortformer-4spk-v1.f16.gguf";
    std::string wav  = "test/samples/diarization-sample-16k.wav";
    int         history_ms   = 30000;
    int         chunk_ms     = 2000;
    std::string rttm_out;
    // Mirror the public SortformerStreamingOptions defaults so the test
    // binary reflects the production v2.1 AOSC config out of the box.
    parakeet::SortformerStreamingOptions sopts_defaults;
    bool        spkcache_enable = sopts_defaults.spkcache_enable;
    int         spkcache_len    = sopts_defaults.spkcache_len;
    int         fifo_len        = sopts_defaults.fifo_len;
    float       threshold       = sopts_defaults.threshold;
    bool gguf_user = false;
    bool wav_user  = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--model"      && i + 1 < argc) { gguf = argv[++i]; gguf_user = true; }
        else if (a == "--wav"        && i + 1 < argc) { wav  = argv[++i]; wav_user  = true; }
        else if (a == "--history-ms" && i + 1 < argc) { history_ms = std::atoi(argv[++i]); }
        else if (a == "--chunk-ms"   && i + 1 < argc) { chunk_ms   = std::atoi(argv[++i]); }
        else if (a == "--rttm-out"   && i + 1 < argc) { rttm_out   = argv[++i]; }
        else if (a == "--spkcache-enable")            { spkcache_enable = true; }
        else if (a == "--no-spkcache")                { spkcache_enable = false; }
        else if (a == "--spkcache-len" && i + 1 < argc) { spkcache_len = std::atoi(argv[++i]); }
        else if (a == "--fifo-len"     && i + 1 < argc) { fifo_len     = std::atoi(argv[++i]); }
        else if (a == "--threshold"    && i + 1 < argc) { threshold    = (float) std::atof(argv[++i]); }
        else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 2;
        }
    }
    if (history_ms <= 0 || chunk_ms <= 0 || history_ms < chunk_ms) {
        std::fprintf(stderr,
            "[sf-stream-test] FAIL: invalid --history-ms / --chunk-ms (history=%d chunk=%d)\n",
            history_ms, chunk_ms);
        return 8;
    }
    const bool model_missing = !file_exists(gguf);
    const bool wav_missing   = !file_exists(wav);

    if ((gguf_user && model_missing) || (wav_user && wav_missing)) {
        std::fprintf(stderr,
            "[sf-stream-test] FAIL: explicit input missing (model=%s%s wav=%s%s)\n",
            gguf.c_str(), model_missing ? " (missing)" : "",
            wav.c_str(),  wav_missing   ? " (missing)" : "");
        return 8;
    }
    if (model_missing || wav_missing) {
        std::fprintf(stderr,
            "[sf-stream-test] SKIP: default fixture not present (model=%s%s wav=%s%s).\n"
            "                Set --model and --wav to exercise the streaming path in CI.\n",
            gguf.c_str(), model_missing ? " (missing)" : "",
            wav.c_str(),  wav_missing   ? " (missing)" : "");
        return 0;
    }
    try {
        return run_basic(gguf, wav, history_ms, chunk_ms, rttm_out,
                         spkcache_enable, spkcache_len, fifo_len, threshold);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[sf-stream-test] EXCEPTION: %s\n", e.what());
        return 99;
    }
}
