#include "parakeet_ctc.h"
#include "parakeet_tdt.h"
#include "mel_preprocess.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using parakeet::EncoderOutputs;
using parakeet::ParakeetCtcModel;
using parakeet::TdtDecodeOptions;
using parakeet::TdtDecodeResult;
using parakeet::TdtDecodeState;
using parakeet::TdtRuntimeWeights;

namespace {

constexpr const char * kTag          = "[tdt-deferred-runtime]";
constexpr int          kOddChunk     = 97;
constexpr int          kExitFailure  = 1;
constexpr int          kExitUsage    = 2;

struct EncodedSequence {
    std::vector<float> frames;
    int                n_frames = 0;
    int                d_model  = 0;
};

int fail(const std::string & what) {
    std::fprintf(stderr, "%s FAIL: %s\n", kTag, what.c_str());
    return kExitFailure;
}

bool decode_resources_absent(const TdtRuntimeWeights & rt) {
    return rt.persist_buffer == nullptr && rt.gctx == nullptr && rt.lstm_cat_buffer == nullptr &&
           rt.host_lstm.empty() && rt.embed.empty();
}

bool decode_resources_present(const TdtRuntimeWeights & rt) {
    return rt.use_graphs ? rt.persist_buffer != nullptr && rt.gctx != nullptr
                         : !rt.host_lstm.empty() && !rt.embed.empty();
}

int encode_wav(ParakeetCtcModel & model, const char * wav, EncodedSequence & out) {
    std::vector<float> samples;
    int sample_rate = 0;
    if (int rc = parakeet::load_wav_mono_f32(wav, samples, sample_rate); rc != 0) return rc;
    std::vector<float> mel;
    int n_mel_frames = 0;
    if (int rc = parakeet::compute_log_mel(samples.data(), (int) samples.size(), model.mel_cfg,
                                           mel, n_mel_frames);
        rc != 0) {
        return rc;
    }
    EncoderOutputs enc;
    if (int rc = parakeet::run_encoder(model, mel.data(), n_mel_frames, model.mel_cfg.n_mels, enc);
        rc != 0) {
        return rc;
    }
    out.frames   = std::move(enc.encoder_out);
    out.n_frames = enc.n_enc_frames;
    out.d_model  = enc.d_model;
    return 0;
}

EncodedSequence tile_past(const EncodedSequence & clip, int min_frames) {
    EncodedSequence tiled;
    tiled.d_model = clip.d_model;
    const size_t clip_floats = (size_t) clip.n_frames * (size_t) clip.d_model;
    while (tiled.n_frames <= min_frames) {
        tiled.frames.insert(tiled.frames.end(), clip.frames.begin(), clip.frames.begin() + clip_floats);
        tiled.n_frames += clip.n_frames;
    }
    return tiled;
}

int greedy_tokens(const ParakeetCtcModel & model, TdtRuntimeWeights & rt,
                  const EncodedSequence & seq, std::vector<int32_t> & tokens) {
    TdtDecodeResult result;
    const int rc = parakeet::tdt_greedy_decode(model, rt, seq.frames.data(), seq.n_frames,
                                               seq.d_model, TdtDecodeOptions{}, result);
    tokens = std::move(result.token_ids);
    return rc;
}

int chunked_window_tokens(const ParakeetCtcModel & model, TdtRuntimeWeights & rt,
                          const EncodedSequence & seq, int chunk,
                          std::vector<int32_t> & tokens) {
    TdtDecodeState state;
    tokens.clear();
    for (int start = 0; start < seq.n_frames; start += chunk) {
        const int frames = std::min(chunk, seq.n_frames - start);
        int steps = 0;
        if (int rc = parakeet::tdt_decode_window(
                model, rt, seq.frames.data() + (size_t) start * seq.d_model, frames, seq.d_model,
                TdtDecodeOptions{}, state, tokens, steps);
            rc != 0) {
            return rc;
        }
    }
    return 0;
}

int check_deferred_matches_eager(const ParakeetCtcModel & model, const EncodedSequence & clip,
                                 const std::vector<int32_t> & reference) {
    TdtRuntimeWeights rt;
    if (int rc = parakeet::tdt_prepare_runtime_deferred(model, rt); rc != 0) {
        return fail("deferred preparation rc=" + std::to_string(rc));
    }
    if (!rt.decode_resources_deferred || !decode_resources_absent(rt)) {
        return fail("a deferred runtime allocated decoder resources before its first decode");
    }
    const int expected_capacity = rt.use_graphs ? parakeet::tdt_enc_proj_capacity(model) : 0;
    if (rt.enc_proj_T_max != expected_capacity) {
        return fail("enc-projection capacity " + std::to_string(rt.enc_proj_T_max) +
                    " instead of the model's window " + std::to_string(expected_capacity));
    }
    std::vector<int32_t> tokens;
    if (int rc = greedy_tokens(model, rt, clip, tokens); rc != 0) {
        return fail("deferred decode rc=" + std::to_string(rc));
    }
    if (rt.decode_resources_deferred || !decode_resources_present(rt)) {
        return fail("the first decode did not build the decoder resources");
    }
    if (tokens != reference) return fail("deferred and eager runtimes decode differently");
    return 0;
}

int check_bounded_windows(const ParakeetCtcModel & model, TdtRuntimeWeights & rt,
                          const EncodedSequence & clip) {
    const EncodedSequence long_input = tile_past(clip, std::max(rt.enc_proj_T_max, clip.n_frames));
    std::vector<int32_t> reference;
    if (int rc = greedy_tokens(model, rt, long_input, reference); rc != 0) {
        return fail("greedy decode past the projection capacity rc=" + std::to_string(rc));
    }
    std::vector<int32_t> whole;
    if (int rc = chunked_window_tokens(model, rt, long_input, long_input.n_frames, whole); rc != 0) {
        return fail("one window past the projection capacity rc=" + std::to_string(rc));
    }
    std::vector<int32_t> chunked;
    if (int rc = chunked_window_tokens(model, rt, long_input, kOddChunk, chunked); rc != 0) {
        return fail("chunked windows rc=" + std::to_string(rc));
    }
    if (whole != reference || chunked != reference) {
        return fail("bounded windows change the token stream");
    }
    std::printf("%s %d frames past a %d-frame projection decode to %zu identical tokens\n",
                kTag, long_input.n_frames, rt.enc_proj_T_max, reference.size());
    return 0;
}

int parse_gpu_layers(int argc, char ** argv) {
    int n_gpu_layers = 0;
    for (int i = 3; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--n-gpu-layers") == 0) n_gpu_layers = std::atoi(argv[i + 1]);
        if (std::strcmp(argv[i], "--backends-dir") == 0) parakeet::set_backends_directory(argv[i + 1]);
    }
    return n_gpu_layers;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <tdt.gguf> <wav> [--n-gpu-layers N] [--backends-dir DIR]\n",
                     argv[0]);
        return kExitUsage;
    }
    const int n_gpu_layers = parse_gpu_layers(argc, argv);

    ParakeetCtcModel model;
    if (int rc = parakeet::load_from_gguf(argv[1], model, 0, n_gpu_layers, false); rc != 0) {
        return fail("load_from_gguf rc=" + std::to_string(rc));
    }
    EncodedSequence clip;
    if (int rc = encode_wav(model, argv[2], clip); rc != 0) {
        return fail("encoding the clip rc=" + std::to_string(rc));
    }

    TdtRuntimeWeights eager;
    if (int rc = parakeet::tdt_prepare_runtime(model, eager); rc != 0) {
        return fail("eager preparation rc=" + std::to_string(rc));
    }
    std::vector<int32_t> reference;
    if (int rc = greedy_tokens(model, eager, clip, reference); rc != 0) {
        return fail("eager decode rc=" + std::to_string(rc));
    }
    if (int rc = check_deferred_matches_eager(model, clip, reference); rc != 0) return rc;
    if (int rc = check_bounded_windows(model, eager, clip); rc != 0) return rc;

    std::printf("%s PASS: %zu tokens, deferred runtime identical to eager\n", kTag,
                reference.size());
    return 0;
}
