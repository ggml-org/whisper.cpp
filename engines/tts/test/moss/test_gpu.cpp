#include "gguf_fixtures.h"
#include "sfx_fixtures.h"
#include "speech_fixtures.h"
#include "../test_env_portable.h"
#include "../../../test/moss_gpu_arm.h"
#include "../../../test/moss_precision.h"
#include "backend_selection.h"
#include "moss/codec.h"
#include "moss/delay_lm.h"
#include "moss/sfx_networks.h"
#include "moss/speech_lm.h"
#include "moss/speech_tokenizer.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace tts_cpp::moss::detail;

namespace {

struct Fixture {
    std::filesystem::path path;
    ~Fixture() { std::error_code ec; std::filesystem::remove(path, ec); }
    std::string str() const { return path.string(); }
};

void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

MossGpuArm arm;

void compare(const char * stage, const std::vector<float> & cpu, const std::vector<float> & gpu) {
    require(!cpu.empty() && cpu.size() == gpu.size(), std::string(stage) + ": output shape");
    double error = 0, norm = 0;
    float worst = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        require(std::isfinite(cpu[i]) && std::isfinite(gpu[i]), std::string(stage) + ": nonfinite output");
        const double d = (double) cpu[i] - gpu[i];
        error += d * d;
        norm += (double) cpu[i] * cpu[i];
        worst = std::max(worst, (float) std::fabs(d));
    }
    const double relative = std::sqrt(error / std::max(norm, 1e-20));
    std::printf("%s: max_abs=%g rel_l2=%g\n", stage, worst, relative);
    require(worst < 2e-4f || relative < 0.01, std::string(stage) + ": CPU/" + arm.prefix + " mismatch");
}

std::vector<float> signal(size_t count) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) values[i] = 0.2f * std::sin((float) i * 0.31f);
    return values;
}

void delay(ggml_type precision) {
    using namespace moss_fixtures;
    std::mt19937 rng(26342);
    Fixture lm_file{temp_gguf("gpu-delay")};
    {
        GgufBuilder b;
        b.random = &rng;
        add_backbone_meta(b.file);
        add_backbone_tensors(b);
        b.write(lm_file.path);
    }
    moss_fixture_precision(lm_file.path, precision);
    DelayLM cpu(lm_file.str(), false, 2, 512), gpu(lm_file.str(), true, 2, 512);
    arm.require(gpu.backend_name());
    const std::vector<DelayRow> rows{{1, {0, 1}}, {3, {2, 3}}, {6, {4, 5}}};
    auto check = [](const DelayLogits & a, const DelayLogits & b) {
        compare("delay text", a.text, b.text);
        require(a.audio.size() == b.audio.size(), "delay audio heads");
        for (size_t i = 0; i < a.audio.size(); ++i) compare("delay audio", a.audio[i], b.audio[i]);
    };
    check(cpu.prefill(rows), gpu.prefill(rows));
    for (int i = 0; i < 5; ++i) check(cpu.step(rows.back()), gpu.step(rows.back()));
    cpu.reset(); gpu.reset();
    check(cpu.prefill(rows), gpu.prefill(rows));

    Fixture decoder{write_decoder("gpu-codec", N_VQ, 3 * D_MODEL, [](gguf_context *) {}, &rng)};
    moss_fixture_precision(decoder.path, precision);
    Codec c(decoder.str(), false, 2), g(decoder.str(), true, 2);
    arm.require(g.backend_name());
    std::vector<int32_t> codes(40 * N_VQ);
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = (int32_t) (i % CODE_SIZE);
    const auto full = c.decode(codes);
    compare("codec batch", full, g.decode(codes));
    compare("codec streaming", full, decode_in_windows(g, codes, 3));
    // TTSD uses fewer residual quantizers than the codec has available.
    codes.resize(19);
    compare("codec dialogue channels", c.decode(codes, 1), g.decode(codes, 1));

    Fixture encoder{temp_gguf("gpu-codec-encoder")};
    {
        GgufBuilder b;
        b.random = &rng;
        add_encoder_meta(b.file);
        add_encoder_tensors(b);
        b.write(encoder.path);
    }
    moss_fixture_precision(encoder.path, precision);
    Codec ec(encoder.str(), false, 2), eg(encoder.str(), true, 2);
    arm.require(eg.backend_name());
    require(ec.encode(signal(160)) == eg.encode(signal(160)), "codec encoder codes differ");
}

void sfx(ggml_type precision) {
    Fixture file{moss_sfx_fixtures::write_sfx_model("gpu-sfx", 26342)};
    moss_fixture_precision(file.path, precision);
    SfxModel cpu(file.str(), false, 2), gpu(file.str(), true, 2);
    arm.require(gpu.backend_name());
    const auto context = encode_text(cpu, {1, 8, 9, 2});
    compare("sfx text", context, encode_text(gpu, {1, 8, 9, 2}));
    const int frames = 17;
    auto latents = signal(frames * moss_sfx_fixtures::LATENT);
    {
        SfxDitSession c(cpu, frames), g(gpu, frames);
        for (float time : {1.0f, 0.5f, 0.0f}) {
            compare("sfx dit", c.velocity(latents, time, context), g.velocity(latents, time, context));
        }
    }
    compare("sfx vae", decode_latents(cpu, latents, frames, frames, 8, {}),
            decode_latents(gpu, latents, frames, frames, 8, {}));
    // Reuse the scheduler after switching from diffusion to the VAE.
    compare("sfx text reload", context, encode_text(gpu, {1, 8, 9, 2}));
}

void speech(ggml_type precision) {
    using namespace moss_speech_fixtures;
    Fixture file{write_lm("gpu-speech", 26342)};
    moss_fixture_precision(file.path, precision);
    SpeechLM cpu(file.str(), false, 2), gpu(file.str(), true, 2);
    arm.require(gpu.backend_name());
    const std::vector<SpeechRow> rows{{TOKEN_IM_START, TOKEN_AUDIO_PAD},
        {TOKEN_SPEECH_START, TOKEN_AUDIO_PAD}, {TOKEN_TEXT_PLACEHOLDER, 7}, {TOKEN_TEXT_PLACEHOLDER, 8}};
    auto check = [](const SpeechLogits & a, const SpeechLogits & b) {
        compare("speech text", a.text, b.text);
        compare("speech audio", a.audio, b.audio);
    };
    for (int batch : {1, 3, 4}) {
        cpu.begin(512); gpu.begin(512);
        const SpeechLogits expected = cpu.prefill(rows, batch);
        check(expected, gpu.prefill(rows, batch));
        gpu.release_generation();
        require(gpu.context() == 0 && gpu.position() == 0, "speech release leaves no KV state");
        gpu.begin(512);
        check(expected, gpu.prefill(rows, batch));
        gpu.release_weights();
        arm.require(gpu.backend_name());
        gpu.begin(512);
        check(expected, gpu.prefill(rows, batch));
        for (int i = 0; i < 5; ++i) check(cpu.step(rows.back()), gpu.step(rows.back()));
        compare("speech audio-only", cpu.step(rows.back(), {false, true}).audio,
                gpu.step(rows.back(), {false, true}).audio);
    }
    Fixture vq{write_vq("gpu-speech-vq", 26342)};
    moss_fixture_precision(vq.path, precision == GGML_TYPE_BF16 ? GGML_TYPE_F16 : precision);
    SpeechTokenizer c(vq.str(), false, 2), g(vq.str(), true, 2);
    const auto pcm = signal(VQ_CHUNK + 37);
    require(c.encode(pcm) == g.encode(pcm), "speech tokenizer codes differ");
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s cuda|vulkan\n", argv[0]);
        return 2;
    }
    try {
        arm = moss_gpu_arm(argv[1]);
        setenv("TTS_CPP_GPU_BACKEND", arm.request.c_str(), 1);
        tts_cpp::detail::ensure_backends_loaded();
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
            std::printf("precision: %s\n", ggml_type_name(type));
            delay(type);
            sfx(type);
            speech(type);
        }
        // MOSS-Speech's default LM export is BF16. Devices without native
        // BF16 must execute it correctly through the scheduler's CPU fallback.
        speech(GGML_TYPE_BF16);
        std::printf("MOSS CPU/%s parity: OK\n", arm.prefix.c_str());
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
