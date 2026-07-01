// Shared-scheduler lifecycle across the diarization pipeline: running three
// distinct graphs (subsampling -> encoder -> Sortformer head) through the one
// shared ggml_backend_sched must not corrupt downstream stages.
//
// The sched migration reset-at-head + download-output-before-next-reset ordering
// is only exercised implicitly by the production one-shot path (encoder -> head).
// This test adds a run_subsampling graph in front of that sequence and asserts the
// resulting speaker_probs / segments are byte-identical to a clean reference model
// that runs only encoder -> head. A wrong reset ordering (freeing a stage's
// allocation before its output is downloaded, or the extra subsampling graph
// clobbering the encoder) makes the two diverge. Repeating N times also locks
// determinism across cached-graph reuse.
//
// Usage:
//   test-sched-diarize-lifecycle --model <sortformer gguf> --wav <wav>
//                                [--runs N] [--n-gpu-layers N]
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet_ctc.h"
#include "parakeet_sortformer.h"
#include "mel_preprocess.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <sortformer gguf> --wav <wav> [--runs N] [--n-gpu-layers N]\n"
        "\n"
        "Runs run_subsampling -> run_encoder -> sortformer_diarize_ggml through the\n"
        "shared scheduler and asserts the speaker_probs / segments match a clean\n"
        "encoder -> head reference model, byte-identical, across --runs (default 5).\n",
        argv0);
}

bool bit_equal(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool non_degenerate(const std::vector<float> & v) {
    if (v.empty()) return false;
    bool any_nonzero = false;
    for (float x : v) {
        if (!std::isfinite(x)) return false;
        if (x != 0.0f) any_nonzero = true;
    }
    return any_nonzero;
}

bool segments_equal(const std::vector<parakeet::SortformerSegment> & a,
                    const std::vector<parakeet::SortformerSegment> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].speaker_id != b[i].speaker_id ||
            a[i].start_s    != b[i].start_s    ||
            a[i].end_s      != b[i].end_s) {
            return false;
        }
    }
    return true;
}

// One-shot diarization: encoder(mel) -> Sortformer head. `pre_subsample` inserts a
// run_subsampling graph (result discarded) in front, exercising a third graph type
// through the shared scheduler before the encoder runs.
int diarize_once(parakeet::ParakeetCtcModel & model,
                 const std::vector<float> & mel, int n_mel_frames,
                 bool pre_subsample,
                 parakeet::SortformerDiarizationResult & out) {
    using namespace parakeet;
    if (pre_subsample) {
        std::vector<float> feats;
        int n_feat = 0;
        if (int rc = run_subsampling(model, mel.data(), n_mel_frames,
                                     model.mel_cfg.n_mels, feats, n_feat); rc != 0) {
            std::fprintf(stderr, "  run_subsampling rc=%d\n", rc);
            return rc;
        }
    }
    EncoderOutputs enc;
    if (int rc = run_encoder(model, mel.data(), n_mel_frames, model.mel_cfg.n_mels,
                             enc, /*max_layers=*/-1, /*capture_intermediates=*/false); rc != 0) {
        std::fprintf(stderr, "  run_encoder rc=%d\n", rc);
        return rc;
    }
    SortformerDiarizationOptions dopts;
    if (int rc = sortformer_diarize_ggml(model, enc.encoder_out.data(),
                                         enc.n_enc_frames, enc.d_model, dopts, out); rc != 0) {
        std::fprintf(stderr, "  sortformer_diarize_ggml rc=%d\n", rc);
        return rc;
    }
    return 0;
}

}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string wav_path;
    int n_runs       = 5;
    int n_gpu_layers = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model"        && i + 1 < argc) model_path   = argv[++i];
        else if (a == "--wav"          && i + 1 < argc) wav_path     = argv[++i];
        else if (a == "--runs"         && i + 1 < argc) n_runs       = std::atoi(argv[++i]);
        else if (a == "--n-gpu-layers" && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else if (a == "-h" || a == "--help")            { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (model_path.empty() || wav_path.empty()) { usage(argv[0]); return 2; }
    if (n_runs < 1) n_runs = 1;

    using namespace parakeet;

    // Load the wav + mel once (shared by both models).
    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        std::fprintf(stderr, "[test-sched-diarize-lifecycle] load_wav rc=%d\n", rc);
        return 1;
    }

    // Reference model: clean encoder -> head, no subsampling graph in front. A
    // separate instance so its scheduler never sees the extra graph.
    ParakeetCtcModel model_ref;
    if (int rc = load_from_gguf(model_path, model_ref, /*n_threads=*/0,
                                n_gpu_layers, /*verbose=*/false); rc != 0) {
        std::fprintf(stderr, "[test-sched-diarize-lifecycle] load_from_gguf(ref) rc=%d\n", rc);
        return 1;
    }
    if (model_ref.model_type != ParakeetModelType::SORTFORMER) {
        std::fprintf(stderr, "[test-sched-diarize-lifecycle] FAIL: expected a Sortformer model\n");
        return 1;
    }
    if (sr != model_ref.mel_cfg.sample_rate) {
        std::fprintf(stderr, "[test-sched-diarize-lifecycle] FAIL: wav sr %d != model sr %d\n",
                     sr, model_ref.mel_cfg.sample_rate);
        return 1;
    }

    std::vector<float> mel;
    int                n_mel_frames = 0;
    if (int rc = compute_log_mel(samples.data(), (int) samples.size(),
                                 model_ref.mel_cfg, mel, n_mel_frames); rc != 0) {
        std::fprintf(stderr, "[test-sched-diarize-lifecycle] compute_log_mel rc=%d\n", rc);
        return 1;
    }

    SortformerDiarizationResult ref;
    if (int rc = diarize_once(model_ref, mel, n_mel_frames, /*pre_subsample=*/false, ref); rc != 0) {
        std::fprintf(stderr, "[test-sched-diarize-lifecycle] reference diarize failed rc=%d\n", rc);
        return 1;
    }
    if (!non_degenerate(ref.speaker_probs)) {
        std::fprintf(stderr,
            "[test-sched-diarize-lifecycle] FAIL: reference speaker_probs degenerate "
            "(empty / all-zero / non-finite)\n");
        return 1;
    }
    std::fprintf(stderr,
        "[test-sched-diarize-lifecycle] backend=%s ref: %d frames, %d spks, %zu segments\n",
        model_active_backend_name(model_ref).c_str(), ref.n_frames, ref.num_spks,
        ref.segments.size());

    // Test model: subsampling -> encoder -> head through the shared scheduler,
    // repeated to also cover cached-graph reuse.
    ParakeetCtcModel model;
    if (int rc = load_from_gguf(model_path, model, /*n_threads=*/0,
                                n_gpu_layers, /*verbose=*/false); rc != 0) {
        std::fprintf(stderr, "[test-sched-diarize-lifecycle] load_from_gguf(test) rc=%d\n", rc);
        return 1;
    }

    for (int k = 0; k < n_runs; ++k) {
        SortformerDiarizationResult r;
        if (int rc = diarize_once(model, mel, n_mel_frames, /*pre_subsample=*/true, r); rc != 0) {
            std::fprintf(stderr, "[test-sched-diarize-lifecycle] run %d diarize failed rc=%d\n", k, rc);
            return 1;
        }
        if (!bit_equal(r.speaker_probs, ref.speaker_probs)) {
            std::fprintf(stderr,
                "[test-sched-diarize-lifecycle] FAIL: run %d speaker_probs differ from the clean "
                "encoder->head reference -- running subsampling -> encoder -> head through the "
                "shared scheduler corrupted a downstream stage\n", k);
            return 1;
        }
        if (!segments_equal(r.segments, ref.segments)) {
            std::fprintf(stderr,
                "[test-sched-diarize-lifecycle] FAIL: run %d produced %zu segments, reference has "
                "%zu (or they differ)\n", k, r.segments.size(), ref.segments.size());
            return 1;
        }
    }

    std::fprintf(stderr,
        "[test-sched-diarize-lifecycle] PASS  %d runs of subsampling->encoder->head "
        "byte-identical to the clean encoder->head reference (%zu probs, %zu segments)\n",
        n_runs, ref.speaker_probs.size(), ref.segments.size());
    return 0;
}
