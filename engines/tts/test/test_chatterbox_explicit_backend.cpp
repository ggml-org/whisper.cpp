// Explicit Chatterbox backend requests (EngineOptions::backend and friends): an
// explicit "cpu" request lands on the CPU and matches the automatic CPU path
// exactly; a request for a missing device fails at every entry point instead of
// falling back.
//
// usage: test-chatterbox-explicit-backend T3.gguf S3GEN.gguf

#include "backend_util.h"
#include "chatterbox_t3_internal.h"
#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/fit.h"
#include "tts-cpp/chatterbox/s3gen_pipeline.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace tts_cpp::chatterbox::detail;

namespace {

constexpr const char * CPU_REQUEST     = "cpu";
constexpr const char * MISSING_REQUEST = "no-such-device";
constexpr int          SPEECH_TOKENS   = 20;
constexpr int          TOKEN_STRIDE    = 73;
constexpr int          TOKEN_SPAN      = 4200;
constexpr int          SEED            = 42;
constexpr int          THREADS         = 2;

int failures = 0;

void check(bool condition, const char * message) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
}

void release_model(chatterbox_model & model) {
    tts_cpp::detail::sched_fallback_free(model.sched_fb);
    if (model.buffer_w)  ggml_backend_buffer_free(model.buffer_w);
    if (model.buffer_kv) ggml_backend_buffer_free(model.buffer_kv);
    if (model.ctx_w)     ggml_free(model.ctx_w);
    if (model.ctx_kv)    ggml_free(model.ctx_kv);
    if (model.backend)   ggml_backend_free(model.backend);
}

bool load_t3(const std::string & path, const char * backend, bool & on_cpu) {
    chatterbox_model model;
    const bool loaded = load_model_gguf(path, model, /*requested_ctx=*/0, /*n_gpu_layers=*/0,
                                        GGML_TYPE_F32, /*measure=*/nullptr, backend);
    on_cpu = loaded && tts_cpp::detail::backend_is_cpu(model.backend);
    if (loaded) {
        check(voice_frontend_backend(model.backend) == model.backend,
              "the reference-audio frontend must share a CPU T3 backend");
        release_model(model);
    }
    return loaded;
}

std::vector<int32_t> make_tokens() {
    std::vector<int32_t> tokens;
    for (int i = 0; i < SPEECH_TOKENS; ++i) tokens.push_back((i * TOKEN_STRIDE + 11) % TOKEN_SPAN);
    return tokens;
}

int synthesize(const std::string & s3gen, const std::string & backend, std::vector<float> & pcm) {
    s3gen_synthesize_opts opts;
    opts.s3gen_gguf_path = s3gen;
    opts.pcm_out         = &pcm;
    opts.seed            = SEED;
    opts.n_threads       = THREADS;
    opts.backend         = backend;
    return s3gen_synthesize_to_wav(make_tokens(), opts);
}

// s3gen_synthesize_to_wav reports model-load failures by throwing rather than
// by return code, so a refused request may surface either way.
bool synthesis_fails(const std::string & s3gen, const std::string & backend) {
    std::vector<float> pcm;
    try {
        return synthesize(s3gen, backend, pcm) != 0;
    } catch (const std::exception &) {
        return true;
    }
}

void check_t3(const std::string & t3) {
    bool on_cpu = false;
    check(load_t3(t3, CPU_REQUEST, on_cpu), "an explicit cpu T3 request must load");
    check(on_cpu, "an explicit cpu T3 request must place T3 on the CPU");
    check(!load_t3(t3, MISSING_REQUEST, on_cpu), "a T3 request for a missing device must fail");
}

void check_s3gen(const std::string & s3gen) {
    std::vector<float> automatic, explicit_cpu;
    check(synthesize(s3gen, "", automatic) == 0, "the automatic S3Gen path must synthesize");
    check(synthesize(s3gen, CPU_REQUEST, explicit_cpu) == 0, "an explicit cpu S3Gen request must synthesize");
    check(!automatic.empty() && automatic == explicit_cpu,
          "an explicit cpu S3Gen request must match the automatic CPU path exactly");
    check(synthesis_fails(s3gen, MISSING_REQUEST), "an S3Gen request for a missing device must fail");
    s3gen_unload();
}

void check_engine(const std::string & t3, const std::string & s3gen) {
    tts_cpp::chatterbox::EngineOptions opts;
    opts.t3_gguf_path    = t3;
    opts.s3gen_gguf_path = s3gen;
    opts.backend         = MISSING_REQUEST;
    bool threw = false;
    try {
        tts_cpp::chatterbox::Engine engine(opts);
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, "an Engine for a missing device must throw instead of falling back");
}

void check_fit(const std::string & t3, const std::string & s3gen) {
    tts_cpp::chatterbox::FitOptions opts;
    opts.t3_gguf_path    = t3;
    opts.s3gen_gguf_path = s3gen;
    opts.backend         = CPU_REQUEST;
    const tts_cpp::FitResult cpu = tts_cpp::chatterbox::fit_params(opts);
    check(cpu.status != tts_cpp::FitStatus::Error, "an explicit cpu fit must project");
    check(cpu.device_is_cpu, "an explicit cpu fit must project onto the CPU");
    opts.backend = MISSING_REQUEST;
    const tts_cpp::FitResult missing = tts_cpp::chatterbox::fit_params(opts);
    check(missing.status == tts_cpp::FitStatus::Error && missing.reason == "no-backend-device",
          "a fit for a missing device must report no-backend-device");
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s T3.gguf S3GEN.gguf\n", argv[0]);
        return 2;
    }
    const std::string t3    = argv[1];
    const std::string s3gen = argv[2];
    check_t3(t3);
    check_s3gen(s3gen);
    check_engine(t3, s3gen);
    check_fit(t3, s3gen);
    std::fprintf(stderr, failures == 0 ? "PASS\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
