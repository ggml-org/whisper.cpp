// T3 (Turbo or multilingual) logits on an explicit backend against the CPU.
// Both models run the same prompt and then the same teacher-forced speech
// tokens, so sampling never lets the two trajectories part; every prompt and
// step logit vector is compared by cosine and argmax.  On Hexagon it also pins
// the KV cache to F16 (an F32 cache overflows the DSP mapping budget).
//
// usage: test-t3-backend-logits T3.gguf BACKEND [--min-cosine C] [--min-argmax-match F] [--n-ctx N]

#include "backend_util.h"
#include "chatterbox_t3_internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tts_cpp::chatterbox::detail;

namespace {

constexpr int    kThreads             = 4;
constexpr int    kTextTokens          = 12;
constexpr int    kForcedSteps         = 24;
constexpr int    kTextTokenStride     = 37;
constexpr int    kSpeechTokenStride   = 97;
constexpr int    kSpeechTokenOffset   = 13;
constexpr int    kSpeechTokenSpan     = 6561;
constexpr float  kExaggeration        = 0.5f;
constexpr double kDefaultMinCosine    = 0.995;
constexpr double kDefaultArgmaxMatch  = 0.75;

struct logit_trace {
    std::vector<std::vector<float>> calls;
};

struct agreement {
    double min_cosine   = 1.0;
    int    argmax_equal = 0;
    int    compared     = 0;
};

std::vector<int32_t> make_text_tokens(const chatterbox_model & model) {
    std::vector<int32_t> tokens;
    for (int i = 0; i < kTextTokens; ++i) {
        tokens.push_back(1 + (i * kTextTokenStride) % (model.hparams.n_text_vocab - 1));
    }
    return tokens;
}

int32_t forced_token(int step) {
    return (step * kSpeechTokenStride + kSpeechTokenOffset) % kSpeechTokenSpan;
}

bool trace_turbo(const chatterbox_model & model, ggml_gallocr_t allocr, logit_trace & out) {
    std::vector<float> logits;
    int prompt_len = 0;
    if (!eval_prompt(model, allocr, kThreads, make_text_tokens(model), logits, prompt_len)) return false;
    out.calls.push_back(logits);
    for (int s = 0; s < kForcedSteps; ++s) {
        if (!eval_step(model, allocr, kThreads, prompt_len + s, forced_token(s), logits)) return false;
        out.calls.push_back(logits);
    }
    return true;
}

bool trace_mtl(const chatterbox_model & model, ggml_gallocr_t allocr, logit_trace & out) {
    std::vector<float> cond, uncond;
    int prompt_len = 0;
    if (!eval_prompt_mtl(model, allocr, kThreads, make_text_tokens(model), kExaggeration,
                         cond, uncond, prompt_len)) {
        return false;
    }
    out.calls.push_back(cond);
    out.calls.push_back(uncond);
    for (int s = 0; s < kForcedSteps; ++s) {
        if (!eval_step_mtl(model, allocr, kThreads, prompt_len + s, forced_token(s), cond, uncond)) return false;
        out.calls.push_back(cond);
        out.calls.push_back(uncond);
    }
    return true;
}

bool trace(const chatterbox_model & model, logit_trace & out) {
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    const bool ok = model.hparams.variant == CHBX_VARIANT_MTL ? trace_mtl(model, allocr, out)
                                                              : trace_turbo(model, allocr, out);
    ggml_gallocr_free(allocr);
    t3_release_caches();
    return ok;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * b[i];
        na  += (double) a[i] * a[i];
        nb  += (double) b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

size_t argmax(const std::vector<float> & v) {
    return (size_t) (std::max_element(v.begin(), v.end()) - v.begin());
}

agreement compare_traces(const logit_trace & cpu, const logit_trace & device) {
    agreement a;
    for (size_t i = 0; i < cpu.calls.size() && i < device.calls.size(); ++i) {
        a.min_cosine = std::min(a.min_cosine, cosine(cpu.calls[i], device.calls[i]));
        a.argmax_equal += argmax(cpu.calls[i]) == argmax(device.calls[i]) ? 1 : 0;
        a.compared += 1;
    }
    return a;
}

void release_model(chatterbox_model & model) {
    tts_cpp::detail::sched_fallback_free(model.sched_fb);
    if (model.buffer_w)        ggml_backend_buffer_free(model.buffer_w);
    if (model.buffer_kv)       ggml_backend_buffer_free(model.buffer_kv);
    if (model.ctx_w)           ggml_free(model.ctx_w);
    if (model.ctx_kv)          ggml_free(model.ctx_kv);
    model.buffer_w  = nullptr;
    model.buffer_kv = nullptr;
    model.ctx_w     = nullptr;
    model.ctx_kv    = nullptr;
}

bool load_and_trace(const std::string & path, const std::string & backend, int n_ctx, logit_trace & out,
                    ggml_type & kv_type, bool & is_hexagon) {
    chatterbox_model model;
    if (!load_model_gguf(path, model, n_ctx, /*n_gpu_layers=*/0, GGML_TYPE_F32,
                         /*measure=*/nullptr, backend)) {
        fprintf(stderr, "FAIL: could not load %s on %s\n", path.c_str(), backend.c_str());
        return false;
    }
    kv_type    = model.hparams.kv_type;
    is_hexagon = tts_cpp::detail::backend_is_hexagon(model.backend);
    const bool ok = trace(model, out);
    if (!ok) fprintf(stderr, "FAIL: evaluation failed on %s\n", backend.c_str());
    // MTL stacks register their buffer for an exit-time release, so only the
    // buffers this test owns are freed here; the backends live to exit.
    release_model(model);
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s T3.gguf BACKEND [--min-cosine C] [--min-argmax-match F] [--n-ctx N]\n",
                argv[0]);
        return 2;
    }
    const std::string path    = argv[1];
    const std::string backend = argv[2];
    double min_cosine   = kDefaultMinCosine;
    double argmax_match = kDefaultArgmaxMatch;
    int    n_ctx        = 0;
    for (int i = 3; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--min-cosine" && i + 1 < argc)            min_cosine   = atof(argv[++i]);
        else if (flag == "--min-argmax-match" && i + 1 < argc) argmax_match = atof(argv[++i]);
        else if (flag == "--n-ctx" && i + 1 < argc)            n_ctx        = atoi(argv[++i]);
    }

    logit_trace cpu, device;
    ggml_type cpu_kv = GGML_TYPE_F32, device_kv = GGML_TYPE_F32;
    bool cpu_hexagon = false, device_hexagon = false;
    if (!load_and_trace(path, "cpu", n_ctx, cpu, cpu_kv, cpu_hexagon) ||
        !load_and_trace(path, backend, n_ctx, device, device_kv, device_hexagon)) {
        return 1;
    }

    bool ok = true;
    if (device_hexagon && device_kv != GGML_TYPE_F16) {
        fprintf(stderr, "FAIL: Hexagon KV cache is %s, expected f16\n", ggml_type_name(device_kv));
        ok = false;
    }
    const agreement a = compare_traces(cpu, device);
    const double match = a.compared ? (double) a.argmax_equal / a.compared : 0.0;
    fprintf(stderr, "  %d logit vectors: min cosine %.6f, argmax match %d/%d (%.3f)\n",
            a.compared, a.min_cosine, a.argmax_equal, a.compared, match);
    if (a.compared != (int) cpu.calls.size() || a.compared == 0) {
        fprintf(stderr, "FAIL: %zu CPU vs %zu device logit vectors\n", cpu.calls.size(), device.calls.size());
        ok = false;
    }
    if (a.min_cosine < min_cosine) {
        fprintf(stderr, "FAIL: min cosine %.6f below %.6f\n", a.min_cosine, min_cosine);
        ok = false;
    }
    if (match < argmax_match) {
        fprintf(stderr, "FAIL: argmax match %.3f below %.3f\n", match, argmax_match);
        ok = false;
    }
    fprintf(stderr, ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
