// Fixture test for the CPU-resident copies of the TDT decoder weights that
// the loader builds when the encoder runs on OpenCL (ggml-opencl cannot run
// the decoder graphs, so they run on the CPU backend from these copies).
//
// Usage:
//   test-tdt-cpu-copies --model <tdt.gguf> --wav <input.wav>
//
// It asserts that:
//   1. on this build's backend the loader leaves the copies empty and the
//      decoder reports the active backend;
//   2. building the copies duplicates every decoder tensor byte for byte on
//      the CPU backend, with the model's type and shape;
//   3. once the copies exist the decoder reports the CPU backend, and a
//      transcription still produces the expected words, so the copies are a
//      drop-in for the originals.
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet/engine.h"

#include "mel_preprocess.h"
#include "parakeet_ctc.h"
#include "parakeet_tdt.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace parakeet;

namespace {

int g_failures = 0;

void expect(bool cond, const std::string & what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

std::vector<uint8_t> bytes_of(const ggml_tensor * t) {
    std::vector<uint8_t> out(ggml_nbytes(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size());
    return out;
}

void expect_same_tensor(const ggml_tensor * original, const ggml_tensor * copy, const char * name) {
    if (!original) {
        expect(copy == nullptr, std::string(name) + ": an absent original must stay absent");
        return;
    }
    expect(copy != nullptr, std::string(name) + ": missing copy");
    if (!copy) return;
    expect(copy->type == original->type && std::memcmp(copy->ne, original->ne, sizeof(original->ne)) == 0,
           std::string(name) + ": type or shape differs");
    expect(bytes_of(copy) == bytes_of(original), std::string(name) + ": bytes differ");
    expect(copy->buffer != original->buffer, std::string(name) + ": the copy shares the original's buffer");
}

void check_copies_match(const TdtWeights & src, const TdtWeights & dst) {
    expect_same_tensor(src.predict_embed, dst.predict_embed, "predict_embed");
    expect(src.lstm.size() == dst.lstm.size(), "lstm layer count differs");
    for (size_t i = 0; i < src.lstm.size() && i < dst.lstm.size(); ++i) {
        expect_same_tensor(src.lstm[i].w_ih, dst.lstm[i].w_ih, "lstm.w_ih");
        expect_same_tensor(src.lstm[i].w_hh, dst.lstm[i].w_hh, "lstm.w_hh");
        expect_same_tensor(src.lstm[i].b_ih, dst.lstm[i].b_ih, "lstm.b_ih");
        expect_same_tensor(src.lstm[i].b_hh, dst.lstm[i].b_hh, "lstm.b_hh");
    }
    expect_same_tensor(src.joint_enc_w,  dst.joint_enc_w,  "joint_enc_w");
    expect_same_tensor(src.joint_enc_b,  dst.joint_enc_b,  "joint_enc_b");
    expect_same_tensor(src.joint_pred_w, dst.joint_pred_w, "joint_pred_w");
    expect_same_tensor(src.joint_pred_b, dst.joint_pred_b, "joint_pred_b");
    expect_same_tensor(src.joint_out_w,  dst.joint_out_w,  "joint_out_w");
    expect_same_tensor(src.joint_out_b,  dst.joint_out_b,  "joint_out_b");
}

std::string transcribe(ParakeetCtcModel & model, const std::string & wav) {
    std::vector<float> samples;
    int sample_rate = 0;
    if (load_wav_mono_f32(wav, samples, sample_rate) != 0) return "";
    std::vector<float> mel;
    int n_frames = 0;
    if (compute_log_mel(samples.data(), (int) samples.size(), model.mel_cfg, mel, n_frames) != 0) return "";
    EncoderOutputs enc;
    if (run_encoder(model, mel.data(), n_frames, model.mel_cfg.n_mels, enc, -1, false) != 0) return "";
    TdtRuntimeWeights rt;
    if (tdt_prepare_runtime(model, rt) != 0) return "";
    TdtDecodeResult result;
    if (tdt_greedy_decode(model, rt, enc.encoder_out.data(), enc.n_enc_frames, enc.d_model,
                          TdtDecodeOptions{}, result) != 0) return "";
    return result.text;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model_path, wav_path;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--model") == 0) model_path = argv[i + 1];
        else if (std::strcmp(argv[i], "--wav") == 0) wav_path = argv[i + 1];
    }
    if (model_path.empty() || wav_path.empty()) {
        std::fprintf(stderr, "usage: test-tdt-cpu-copies --model <gguf> --wav <wav>\n");
        return 2;
    }

    ParakeetCtcModel model;
    if (load_from_gguf(model_path, model, 0, 0, false) != 0 || model.model_type != ParakeetModelType::TDT) {
        std::fprintf(stderr, "FAIL: could not load a TDT model from %s\n", model_path.c_str());
        return 1;
    }

    expect(!model_tdt_decoder_on_cpu(model), "the copies must not exist on a non-OpenCL load");
    expect(model_decoder_backend_name(model) == model_active_backend_name(model),
           "without copies the decoder reports the active backend");
    const std::string before = transcribe(model, wav_path);
    expect(!before.empty(), "the original decoder produced no text");

    expect(model_build_tdt_cpu_weights(model), "building the CPU copies failed");
    expect(model_tdt_decoder_on_cpu(model), "the copies must register as the decoder's weights");
    check_copies_match(model.tdt, model.tdt_cpu);
    expect(model_decoder_backend_name(model) == ggml_backend_name(model.backend_cpu()),
           "with copies the decoder reports the CPU backend");

    if (g_failures == 0) {
        std::printf("test-tdt-cpu-copies: OK (%s)\n", before.c_str());
        return 0;
    }
    std::fprintf(stderr, "test-tdt-cpu-copies: %d failure(s)\n", g_failures);
    return 1;
}
