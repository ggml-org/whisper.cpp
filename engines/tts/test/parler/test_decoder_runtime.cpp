// Native generation recovery gates, runnable with the tiny DAC CI GGUF.
// A fake sidecar leaves ggml KV stale; recovery must replay prompt and consumed
// frames, rather than silently continue from the stale cache.
#include "parler/decoder_runtime.h"
#include "parler/internal.h"
#include "tts-cpp/parler/fit.h"
#include "../test_env_portable.h"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>

using namespace tts_cpp::parler::detail;

namespace {
int failures = 0;
void check(bool value, const char * text) {
    fprintf(stderr, "[%s] %s\n", value ? "PASS" : "FAIL", text);
    if (!value) ++failures;
}

struct fake_step final : parler_decoder_step_model {
    int count = 0, seeds = 0, fail_at = 2;
    bool seed_ok = true, nonfinite = false;
    bool native_timing = true;
    int profiled_calls = 0;
    int context = 4096, cross = 4096;
    size_t logit_count;
    explicit fake_step(size_t n) : logit_count(n) {}
    int context_capacity() const override { return context; }
    int cross_capacity() const override { return cross; }
    const char * label() const override { return "coreml-fake"; }
    bool seed(const std::vector<float> & k, const std::vector<float> & v, int past,
              const std::vector<float> & ck, const std::vector<float> & cv,
              int cross, std::string &) override {
        ++seeds;
        count = 0;
        return seed_ok && past > 0 && cross > 0 && !k.empty() && k.size() == v.size() &&
               !ck.empty() && ck.size() == cv.size();
    }
    bool step(const std::vector<float> & hidden, int, std::vector<float> & logits,
              std::string & error, parler_decoder_step_profile * profile) override {
        if (profile) {
            ++profiled_calls;
            *profile = {};
            profile->native_timing_available = native_timing;
            profile->row_bytes_written = 128;
        }
        if (++count == fail_at) { error = "injected failure"; return false; }
        if (hidden.empty()) return false;
        logits.assign(logit_count, nonfinite ? std::numeric_limits<float>::quiet_NaN() : 0.0f);
        return true;
    }
};

bool equal(const parler_step_logits & logits, const std::vector<float> & want) {
    if (!logits.view) return false;
    for (size_t i = 0; i < want.size(); ++i) {
        if (!std::isfinite(logits.view[i]) || std::fabs(logits.view[i] - want[i]) > 1e-6f) return false;
    }
    return true;
}
} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) { fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]); return 77; }
    parler_model model;
    std::string error;
    if (!parler_load_gguf(argv[1], model, 0, &error)) { fprintf(stderr, "%s\n", error.c_str()); return 1; }
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    const auto & hp = model.hparams;
    auto contract = nlohmann::json{
        {"contract", 2}, {"kind", "parler-decoder-step"}, {"normalization", "last-axis"}, {"mask_off", -30000},
        {"variant", hp.variant}, {"n_layer", hp.dec_n_layer}, {"d_model", hp.dec_d_model},
        {"n_head", hp.dec_n_head}, {"head_dim", hp.dec_d_model / hp.dec_n_head},
        {"n_q", hp.n_codebooks}, {"vocab", hp.dec_vocab}, {"ctx", 64}, {"cross_context", 8},
        {"states", {{"k_layout", "rows"}, {"layers_per_state", 1}}}};
    check(parler_validate_decoder_contract(contract.dump(), hp, 64, 8, 1, error), "supported decoder contract");
    check(!parler_validate_decoder_contract("", hp, 64, 8, 1, error), "missing contract rejected");
    auto legacy_contract = contract;
    legacy_contract["contract"] = 1;
    legacy_contract.erase("normalization");
    check(!parler_validate_decoder_contract(legacy_contract.dump(), hp, 64, 8, 1, error) &&
          error.find("re-export") != std::string::npos, "old channel-axis export requires re-export");
    auto unsafe_contract = contract;
    unsafe_contract["normalization"] = "channel-axis";
    check(!parler_validate_decoder_contract(unsafe_contract.dump(), hp, 64, 8, 1, error) &&
          error.find("last-axis") != std::string::npos, "unsupported normalization recipe rejected");
    unsafe_contract.erase("normalization");
    check(!parler_validate_decoder_contract(unsafe_contract.dump(), hp, 64, 8, 1, error) &&
          error.find("re-export") != std::string::npos,
          "missing normalization recipe rejected");
    for (const auto * key : {"contract", "mask_off", "n_layer", "d_model", "n_head", "head_dim", "n_q", "vocab", "ctx", "cross_context"}) {
        auto bad = contract;
        bad[key] = -7;
        check(!parler_validate_decoder_contract(bad.dump(), hp, 64, 8, 1, error), "incompatible contract field rejected");
    }
    auto bad_contract = contract;
    bad_contract["states"]["k_layout"] = "cols";
    check(!parler_validate_decoder_contract(bad_contract.dump(), hp, 64, 8, 1, error) &&
          error.find("k_layout=rows") != std::string::npos, "unsupported key layout rejected explicitly");
    bad_contract = contract;
    bad_contract["states"]["layers_per_state"] = 2;
    check(!parler_validate_decoder_contract(bad_contract.dump(), hp, 64, 8, 1, error), "inconsistent group metadata rejected");
    auto variant_hp = hp;
    variant_hp.variant = "expected-checkpoint";
    check(!parler_validate_decoder_contract(contract.dump(), variant_hp, 64, 8, 1, error), "wrong checkpoint variant rejected");
    const std::vector<int32_t> prompt{2, 3}, start(hp.n_codebooks, hp.bos_id);
    const size_t n = (size_t) hp.n_codebooks * hp.dec_vocab;
    std::vector<std::vector<int32_t>> frames(3, std::vector<int32_t>(hp.n_codebooks));
    for (size_t i = 0; i < frames.size(); ++i) std::fill(frames[i].begin(), frames[i].end(), 4 + (int) i);
    if (!parler_encode_description(model, {2, 1}, 2)) return 1;
    parler_step_logits logits;
    int past = 0;
    check(parler_dec_prefill(model, prompt, start, allocator, 2, logits, past), "reference prefill");
    std::vector<std::vector<float>> reference;
    for (const auto & frame : frames) {
        if (!parler_dec_step(model, frame, past++, allocator, 2, logits)) return 1;
        reference.emplace_back(logits.view, logits.view + n);
    }

    parler_decoder_runtime runtime(model);
    auto sidecar = std::make_unique<fake_step>(n);
    auto * probe = sidecar.get();
    runtime.attach(std::move(sidecar));
    check(runtime.prefill(prompt, start, allocator, 2, logits, past) && probe->seeds == 1,
          "prefill seeds native caches");
    // Destroy the ggml cache to ensure an accidental direct fallback fails.
    ggml_backend_buffer_clear(model.buffer_kv, 0);
    check(parler_dec_step(model, frames[1], past + 1, allocator, 2, logits) &&
          !equal(logits, reference[1]), "fixture detects stale-cache direct fallback");
    check(runtime.step(frames[0], past++, allocator, 2, logits) && runtime.backend() == "coreml-fake",
          "first step uses sidecar");
    check(probe->profiled_calls == 0, "default runtime step does not request native profiling");
    parler_decoder_step_profile profile;
    profile.valid = true;
    check(runtime.step(frames[1], past++, allocator, 2, logits, &profile) && !runtime.active() &&
          runtime.backend() == "mixed" && equal(logits, reference[1]),
          "mid-utterance failure replays exact history into a destroyed ggml cache");
    check(!profile.valid && profile.row_bytes_written == 0,
          "failed prediction and successful replay publish no native profile");
    profile.valid = true;
    check(runtime.step(frames[2], past++, allocator, 2, logits, &profile) && equal(logits, reference[2]),
          "later steps continue from the rebuilt ggml cache");
    check(!profile.valid, "ggml steps clear stale native profile validity");
    check(runtime.prefill(prompt, start, allocator, 2, logits, past) && runtime.backend() == "ggml",
          "later synthesis stays on ggml after retirement");

    sidecar = std::make_unique<fake_step>(n);
    sidecar->fail_at = 1;
    runtime.attach(std::move(sidecar), true);
    check(runtime.prefill(prompt, start, allocator, 2, logits, past), "strict prefill");
    check(!runtime.step(frames[0], past, allocator, 2, logits), "strict prediction fails call");
    check(!runtime.prefill(prompt, start, allocator, 2, logits, past), "strict retirement fails later calls");

    sidecar = std::make_unique<fake_step>(n);
    sidecar->nonfinite = true;
    runtime.attach(std::move(sidecar));
    check(runtime.prefill(prompt, start, allocator, 2, logits, past), "nonfinite prefill");
    ggml_backend_buffer_clear(model.buffer_kv, 0);
    check(runtime.step(frames[0], past, allocator, 2, logits, &profile) && equal(logits, reference[0]),
          "nonfinite sidecar logits trigger replay");
    check(!profile.valid && profile.row_bytes_written == 0,
          "native timings are not published when finite-logit validation fails");

    sidecar = std::make_unique<fake_step>(n - 1);
    runtime.attach(std::move(sidecar));
    check(runtime.prefill(prompt, start, allocator, 2, logits, past) &&
          runtime.step(frames[0], past, allocator, 2, logits, &profile) && !profile.valid,
          "wrong native output geometry cannot publish a profile");

    sidecar = std::make_unique<fake_step>(n);
    sidecar->native_timing = false;
    runtime.attach(std::move(sidecar));
    check(runtime.prefill(prompt, start, allocator, 2, logits, past) &&
          runtime.step(frames[0], past, allocator, 2, logits, &profile) && !profile.valid,
          "unknown timing support is unavailable rather than zero-cost native");

    sidecar = std::make_unique<fake_step>(n);
    probe = sidecar.get();
    runtime.attach(std::move(sidecar));
    check(runtime.prefill(prompt, start, allocator, 2, logits, past) &&
          runtime.prefill(prompt, start, allocator, 2, logits, past) && probe->seeds == 2,
          "repeated synthesis seeds fresh state for unchanged descriptions");
    check(!runtime.step(frames[0], past + 1, allocator, 2, logits), "out-of-order position rejected");
    check(!runtime.step(frames[0], past, allocator, 2, logits), "rejected position invalidates current call");
    check(runtime.prefill(prompt, start, allocator, 2, logits, past) &&
          runtime.step(frames[0], past, allocator, 2, logits, &profile), "fresh prefill recovers after invalid position");
    check(profile.valid && profile.native_timing_available && profile.row_bytes_written == 128 &&
          profile.embedding_ms >= -1e-6 && profile.step_wall_ms + 1e-6 >= profile.embedding_ms &&
          probe->profiled_calls == 1,
          "opt-in profile published only after a successful validated native step");

    sidecar = std::make_unique<fake_step>(n);
    sidecar->seed_ok = false;
    runtime.attach(std::move(sidecar));
    check(runtime.prefill(prompt, start, allocator, 2, logits, past) && !runtime.active(),
          "seed failure keeps successful ggml prefill");
    check(runtime.step(frames[0], past, allocator, 2, logits) && equal(logits, reference[0]),
          "seed failure continues on ggml");

    sidecar = std::make_unique<fake_step>(n);
    sidecar->context = (int) prompt.size() + hp.gen_max_length - 2;
    runtime.attach(std::move(sidecar));
    check(runtime.prefill(prompt, start, allocator, 2, logits, past) && runtime.active() &&
          runtime.step(frames[0], past, allocator, 2, logits) && runtime.backend() == "ggml",
          "oversized call stays on ggml without retiring sidecar");
    check(runtime.prefill(prompt, start, allocator, 2, logits, past, hp.gen_max_length - 1) &&
          runtime.step(frames[0], past, allocator, 2, logits) && runtime.backend() == "coreml-fake",
          "later bounded call uses retained sidecar");
    sidecar = std::make_unique<fake_step>(n);
    sidecar->context = (int) prompt.size() + hp.gen_max_length - 2;
    runtime.attach(std::move(sidecar), true);
    check(!runtime.prefill(prompt, start, allocator, 2, logits, past) && runtime.active(),
          "strict oversized call rejects request without retiring sidecar");
    check(runtime.prefill(prompt, start, allocator, 2, logits, past, hp.gen_max_length - 1) &&
          runtime.step(frames[0], past, allocator, 2, logits), "strict later bounded call succeeds");

    sidecar = std::make_unique<fake_step>(n);
    sidecar->context = (int) prompt.size() + hp.gen_max_length - 1;
    runtime.attach(std::move(sidecar), true);
    check(runtime.prefill(prompt, start, allocator, 2, logits, past) &&
          runtime.step(frames[0], past, allocator, 2, logits), "exact-fit generation capacity uses sidecar");

    sidecar = std::make_unique<fake_step>(n);
    sidecar->cross = model.cross_len - 1;
    runtime.attach(std::move(sidecar));
    check(runtime.prefill(prompt, start, allocator, 2, logits, past) && runtime.active() &&
          runtime.step(frames[0], past, allocator, 2, logits) && runtime.backend() == "ggml",
          "oversized description does not retire sidecar");
    check(parler_encode_description(model, {1}, 2) &&
          runtime.prefill(prompt, start, allocator, 2, logits, past) &&
          runtime.step(frames[0], past, allocator, 2, logits) && runtime.backend() == "coreml-fake",
          "later shorter description uses retained sidecar");
    check(parler_encode_description(model, {2, 1}, 2), "restore reference description");

    sidecar = std::make_unique<fake_step>(n);
    sidecar->fail_at = 1;
    runtime.attach(std::move(sidecar));
    check(runtime.prefill(prompt, start, allocator, 2, logits, past), "cancelled replay prefill");
    int checks = 0;
    runtime.set_cancel_check([&] { return ++checks >= 3; });
    check(!runtime.step(frames[0], past, allocator, 2, logits), "cancellation interrupts replay");
    runtime.set_cancel_check({});
    check(!runtime.step(frames[1], past + 1, allocator, 2, logits), "cancelled replay cannot resume stale cache");
    check(runtime.prefill(prompt, start, allocator, 2, logits, past), "new call recovers after cancelled replay");

    if (argc == 3 && std::string(argv[2]) == "--coreml") {
        fprintf(stderr, "native parity model=%s attention=%s kv=%s units=%s\n", argv[1],
                model.use_fa ? "fa" : "manual", ggml_type_name(model.kv_type),
                std::getenv(PARLER_COREML_UNITS_ENV) ? std::getenv(PARLER_COREML_UNITS_ENV) : "all");
        auto native = parler_open_decoder_sidecar(model, argv[1], error);
        check(native != nullptr, "actual decoder sidecar opens");
        if (native) {
            runtime.attach(std::move(native), true);
            for (int repeat = 0; repeat < 2; ++repeat) {
                check(runtime.prefill(prompt, start, allocator, 2, logits, past), "actual sidecar prefill/reset");
                for (size_t i = 0; i < frames.size(); ++i) {
                    const bool ok = runtime.step(frames[i], past++, allocator, 2, logits, &profile);
                    check(ok && profile.valid && profile.row_bytes_written > 0 &&
                          profile.session_total_ms + 1e-6 >= profile.predict_ms + profile.row_write_ms &&
                          profile.step_other_ms() >= -1e-6,
                          "actual native profile includes row bytes and nested timing intervals");
                    float peak = 0;
                    size_t peak_index = 0;
                    if (ok) for (size_t j = 0; j < n; ++j) {
                        const float difference = std::isfinite(logits.view[j]) && std::isfinite(reference[i][j])
                            ? std::fabs(logits.view[j] - reference[i][j]) : std::numeric_limits<float>::infinity();
                        if (difference > peak) { peak = difference; peak_index = j; }
                    }
                    check(ok && runtime.backend().find("coreml-") == 0 && peak < 0.02f,
                          "actual sidecar seeded step matches ggml within fp16 tolerance");
                    fprintf(stderr, "actual sidecar repeat %d step %zu peak logit error %.6f index %zu actual %.6f reference %.6f\n",
                            repeat, i, peak, peak_index, ok ? logits.view[peak_index] : 0.0f, reference[i][peak_index]);
                }
            }
        } else {
            fprintf(stderr, "%s\n", error.c_str());
        }
    }

#ifdef TTS_CPP_USE_COREML
    tts_cpp::parler::FitOptions fit;
    fit.model_gguf_path = argv[1];
    fit.coreml_decoder = true;
    check(tts_cpp::parler::fit_params(fit).reason == "coreml-memory-estimate-unavailable",
          "fit refuses to project requested decoder Core ML memory");
    setenv(PARLER_COREML_DISABLE_ENV, "1", 1);
    check(tts_cpp::parler::fit_params(fit).reason != "coreml-memory-estimate-unavailable",
          "disabled decoder uses native fit projection");
    unsetenv(PARLER_COREML_DISABLE_ENV);
#endif

    ggml_gallocr_free(allocator);
    parler_free_model(model);
    return failures ? 1 : 0;
}
