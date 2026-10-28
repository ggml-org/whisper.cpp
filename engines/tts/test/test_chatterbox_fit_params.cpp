// Fit-projection parity tests (include/tts-cpp/chatterbox/fit.h): assert
// that the metadata-only memory projection matches what a REAL load and a
// REAL synthesis actually allocate, byte for byte where the projection is
// exact by construction:
//
//   1. fit_params returns a projection (never Error) for a readable pair,
//      with non-zero weight / KV / compute figures and a report;
//   2. T3 load parity -- projected weight, KV-slab, and (MTL, GPU) wqkv-stack
//      bytes equal the buffers a real load_model_gguf allocates on the same
//      backend (both sides run the same buffer-type sizing sweep);
//   3. T3 compute parity -- the projected prompt/step arena equals the
//      gallocr buffer a real eval_prompt + eval_step actually reserve at the
//      same shapes (direct-dispatch path; skipped when the backend needs the
//      scheduler fallback, whose CPU portion the projection reports as host);
//   4. S3Gen parity -- projected weights and each resident stage arena
//      (encoder / CFM / F0 / STFT / HiFT) equal what one real
//      s3gen_synthesize_to_wav of the same token count leaves allocated;
//   5. the projection grows with n_predict (no false saturation: every
//      chatterbox stage scales with the utterance);
//   6. a missing model file is Error/"model-unreadable", never Success;
//   7. a near-INT_MAX --text-tokens is Error/"workload-too-large" -- the
//      prompt sum is widened before the n_ctx comparison, so it can never
//      sign-overflow into a negative tensor dim and a ggml abort (the
//      preflight must not crash on the inputs it exists to reject).
//
// Usage: test-chatterbox-fit-params <t3.gguf> <s3gen.gguf> [n_gpu_layers]
//        [--backend NAME] [--n-ctx N]
// (CMake registers the CPU form; pass e.g. 99 manually to check parity on a
// GPU backend.)  Exit 0 on success; non-zero with a FAIL line per broken
// invariant.

#include "tts-cpp/chatterbox/fit.h"
#include "tts-cpp/chatterbox/s3gen_pipeline.h"

#include "chatterbox_fit_internal.h"
#include "chatterbox_t3_internal.h"
#include "chatterbox_tts_test_hooks.h"
#include "t3_mtl.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <limits>
#include <string>
#include <vector>

namespace detail = tts_cpp::chatterbox::detail;
namespace hooks  = tts_cpp::chatterbox::test_hooks;

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

void expect_eq(uint64_t projected, uint64_t real, const std::string & what) {
    std::printf("%s: projected %llu allocated %llu\n", what.c_str(),
                static_cast<unsigned long long>(projected),
                static_cast<unsigned long long>(real));
    if (projected != real) {
        fail(what + ": projected " + std::to_string(projected) +
             " != allocated " + std::to_string(real));
    }
}

bool parse_int(const char * text, int & value) {
    if (!text || !*text) return false;
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno || *end || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) return false;
    value = static_cast<int>(parsed);
    return true;
}

void usage(const char * executable) {
    std::fprintf(stderr,
        "usage: %s <t3.gguf> <s3gen.gguf> [n_gpu_layers] [--backend NAME] [--n-ctx N]\n",
        executable);
}

void free_t3(detail::chatterbox_model & m) {
    // The real MTL GPU stack is also registered for process-exit cleanup.
    // Remove that ownership before releasing it here.
    if (m.buffer_stack || m.ctx_stack) {
        detail::t3_stack_unregister(m.buffer_stack, m.ctx_stack);
    }
    ::tts_cpp::detail::sched_fallback_free(m.sched_fb);
    detail::t3_release_caches();
    if (m.buffer_w)        ggml_backend_buffer_free(m.buffer_w);
    if (m.buffer_kv)       ggml_backend_buffer_free(m.buffer_kv);
    if (m.buffer_stack)    ggml_backend_buffer_free(m.buffer_stack);
    if (m.buffer_override) ggml_backend_buffer_free(m.buffer_override);
    if (m.ctx_w)           ggml_free(m.ctx_w);
    if (m.ctx_kv)          ggml_free(m.ctx_kv);
    if (m.ctx_stack)       ggml_free(m.ctx_stack);
    if (m.ctx_override)    ggml_free(m.ctx_override);
    if (m.backend)         ggml_backend_free(m.backend);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 2;
    }
    const std::string t3_path    = argv[1];
    const std::string s3gen_path = argv[2];
    int n_gpu_layers = 0;
    int n_ctx = 0;
    bool have_layers = false;
    std::string backend;
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc && argv[i + 1][0] &&
            argv[i + 1][0] != '-') {
            backend = argv[++i];
        } else if (arg == "--n-ctx" && i + 1 < argc &&
                   parse_int(argv[i + 1], n_ctx) && n_ctx >= 0) {
            ++i;
        } else if (!have_layers && parse_int(argv[i], n_gpu_layers)) {
            have_layers = true;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    constexpr int kTextTokens  = 8;
    constexpr int kSpeechToks  = 40;

    tts_cpp::chatterbox::FitOptions fopts;
    fopts.t3_gguf_path    = t3_path;
    fopts.s3gen_gguf_path = s3gen_path;
    fopts.n_gpu_layers    = n_gpu_layers;
    fopts.backend         = backend;
    fopts.n_ctx           = n_ctx;
    fopts.text_tokens     = kTextTokens;
    fopts.n_predict       = kSpeechToks;

    // 1. A readable pair always yields a projection.
    const tts_cpp::FitResult fit = tts_cpp::chatterbox::fit_params(fopts);
    expect(fit.status != tts_cpp::FitStatus::Error,
           "fit_params returned Error (" + fit.reason + ") for a readable pair");
    expect(fit.device.weights_bytes > 0,       "projected weights_bytes == 0");
    expect(fit.device.state_bytes > 0,         "projected state_bytes == 0");
    expect(fit.device.lm_compute_bytes > 0,    "projected lm_compute_bytes == 0");
    expect(fit.device.codec_compute_bytes > 0, "projected codec_compute_bytes == 0");
    expect(!fit.report.empty(),                "empty report");
    expect(!fit.model_variant.empty(),         "empty model_variant");
    expect(fit.device_total_bytes > 0,         "device_total_bytes == 0");
    if (g_failures) {
        return g_failures;  // nothing below is meaningful without a projection
    }
    std::printf("%s", fit.report.c_str());

    // 2 + 3. T3 parity against a real load + eval on the same backend.
    {
        detail::t3_load_measure t3m;
        detail::chatterbox_model mm;  // metadata-only
        if (!detail::load_model_gguf(t3_path, mm, n_ctx, n_gpu_layers,
                                     GGML_TYPE_F32, &t3m, backend)) {
            fail("metadata-only load_model_gguf failed");
            return g_failures;
        }
        detail::chatterbox_model real;
        if (!detail::load_model_gguf(t3_path, real, n_ctx, n_gpu_layers,
                                     GGML_TYPE_F32, nullptr, backend)) {
            fail("real load_model_gguf failed");
            free_t3(mm);
            return g_failures;
        }

        expect_eq(t3m.weights_bytes, ggml_backend_buffer_get_size(real.buffer_w),
                  "T3 weights parity");
        expect_eq(t3m.kv_bytes, ggml_backend_buffer_get_size(real.buffer_kv),
                  "T3 KV parity");
        expect_eq(t3m.stack_bytes,
                  real.buffer_stack ? ggml_backend_buffer_get_size(real.buffer_stack) : 0,
                  "T3 wqkv stack parity");

        // Real prompt + one decode step through the Engine-shaped gallocr.
        const bool mtl = real.hparams.variant == detail::CHBX_VARIANT_MTL;
        ggml_gallocr_t allocr = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(real.backend));
        std::vector<int32_t> text(kTextTokens, 1);
        std::vector<float>   logits, logits_u;
        int prompt_len = 0;
        bool eval_ok;
        if (mtl) {
            eval_ok = detail::eval_prompt_mtl(real, allocr, /*n_threads=*/4, text,
                                              /*exaggeration=*/0.5f, logits, logits_u,
                                              prompt_len) &&
                      detail::eval_step_mtl(real, allocr, 4, prompt_len,
                                            real.hparams.start_speech_token,
                                            logits, logits_u);
        } else {
            eval_ok = detail::eval_prompt(real, allocr, 4, text, logits, prompt_len) &&
                      detail::eval_step(real, allocr, 4, prompt_len,
                                        real.hparams.start_speech_token, logits);
        }
        if (!eval_ok) {
            fail("real eval_prompt/eval_step failed");
        } else {
            uint64_t dev = 0, host = 0;
            if (!detail::t3_measure_compute(mm, kTextTokens, prompt_len, dev, host)) {
                fail("t3_measure_compute failed");
            } else if (host == 0) {
                // Direct-dispatch path: the projection must equal the real
                // gallocr buffer byte for byte.  (A sched-routed backend
                // reports a host portion instead; its split is priced with
                // the sched's own size-only API and not comparable here.)
                expect_eq(dev, ggml_gallocr_get_buffer_size(allocr, 0),
                          "T3 compute parity");
            }
        }
        ggml_gallocr_free(allocr);
        free_t3(real);
        free_t3(mm);
    }

    // 4. S3Gen parity: one real synthesis leaves the weights + all five
    //    stage arenas resident; the projection must match each, byte for
    //    byte (HiFT only on the direct-dispatch path -- the sched rebuilds
    //    per call and holds no gallocr).
    {
        std::vector<int32_t> toks(kSpeechToks, 42);
        std::vector<float> pcm;
        s3gen_synthesize_opts sopts;
        sopts.s3gen_gguf_path = s3gen_path;
        sopts.pcm_out         = &pcm;
        sopts.n_threads       = 4;
        sopts.n_gpu_layers    = n_gpu_layers;
        sopts.backend         = backend;
        if (s3gen_synthesize_to_wav(toks, sopts) != 0) {
            fail("real s3gen_synthesize_to_wav failed");
        } else {
            expect(!pcm.empty(), "real synthesis produced no PCM");
            detail::s3gen_fit_measure sm;
            std::string error;
            if (!detail::s3gen_measure_fit(s3gen_path, n_gpu_layers, backend, kSpeechToks,
                                           sm, &error)) {
                fail("s3gen_measure_fit failed: " + error);
            } else {
                expect_eq(sm.weights_bytes, hooks::s3gen_cached_weights_bytes(),
                          "S3Gen weights parity");
                expect_eq(sm.encoder_bytes, hooks::encoder_graph_cache_buffer_bytes(),
                          "S3Gen encoder arena parity");
                expect_eq(sm.cfm_bytes, hooks::cfm_estimator_cache_buffer_bytes(),
                          "S3Gen CFM arena parity");
                expect_eq(sm.f0_bytes, hooks::f0_graph_cache_buffer_bytes(),
                          "S3Gen F0 arena parity");
                expect_eq(sm.stft_bytes, hooks::stft_graph_cache_buffer_bytes(),
                          "S3Gen STFT arena parity");
                const uint64_t real_hift = hooks::hift_graph_cache_buffer_bytes();
                if (real_hift > 0) {
                    expect_eq(sm.hift_device_bytes, real_hift,
                              "S3Gen HiFT arena parity");
                }
            }
        }
        s3gen_unload();
    }

    // 5. No false saturation: every chatterbox stage scales with the
    //    utterance, so a bigger token budget must project strictly more.
    {
        tts_cpp::chatterbox::FitOptions big = fopts;
        big.n_predict = 4 * kSpeechToks;
        const tts_cpp::FitResult fb = tts_cpp::chatterbox::fit_params(big);
        expect(fb.status != tts_cpp::FitStatus::Error, "bigger projection errored");
        expect(fb.device.total_bytes > fit.device.total_bytes,
               "projection did not grow with n_predict");
    }

    // 6. Errors are reported as Error, never Success.
    {
        tts_cpp::chatterbox::FitOptions bad = fopts;
        bad.t3_gguf_path = t3_path + ".does-not-exist";
        const tts_cpp::FitResult fr = tts_cpp::chatterbox::fit_params(bad);
        expect(fr.status == tts_cpp::FitStatus::Error, "missing model was not Error");
        expect(fr.reason == "model-unreadable",
               "missing model reason was '" + fr.reason + "'");
        expect(!fr.fits, "missing model reported fits");
    }

    // 7. A near-INT_MAX workload is rejected strictly, never evaluated: the
    //    prompt sum is widened before the n_ctx check, so it cannot
    //    sign-overflow into a negative tensor dim (a preflight that crashes
    //    on its input violates its own contract).
    {
        tts_cpp::chatterbox::FitOptions huge = fopts;
        huge.text_tokens = std::numeric_limits<int>::max() - 1;
        const tts_cpp::FitResult fr = tts_cpp::chatterbox::fit_params(huge);
        expect(fr.status == tts_cpp::FitStatus::Error,
               "near-INT_MAX text_tokens was not Error");
        expect(fr.reason == "workload-too-large",
               "near-INT_MAX text_tokens reason was '" + fr.reason + "'");
        expect(!fr.fits, "near-INT_MAX text_tokens reported fits");
    }

    if (g_failures == 0) {
        std::printf("test-chatterbox-fit-params: all checks passed\n");
    }
    return g_failures;
}
