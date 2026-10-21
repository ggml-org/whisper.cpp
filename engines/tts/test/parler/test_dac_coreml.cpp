// Parler DAC on the Core ML sidecar (src/parler/dac_coreml.h).
//
//   test-parler-dac-coreml --gguf MODEL.gguf [--fallback-gguf MODEL.gguf]
//                          [--ref-dir parler-ref] [--min-snr 45] [--require-coreml]
//
// Every build (fake sidecars, no Core ML needed):
//   * the host quantizer the sidecar decodes from equals ggml's latent;
//   * a sidecar that fails mid-range (on its second window) is retired once
//     and the WHOLE range is redone on ggml -- bit-identical to a ggml-only
//     decode, nothing of the failed call's earlier windows survives -- and
//     later calls stay on ggml;
//   * PARLER_COREML_STRICT turns the failure into an error (for later calls
//     too), and a missing sidecar into a throw at attach;
//   * empty ranges and out-of-range codes keep their ggml-path behaviour.
// TTS_CPP_COREML builds with `<stem>-dac.mlmodelc` next to --gguf:
//   * fit_params refuses the projection (coreml-memory-estimate-unavailable)
//     unless PARLER_COREML_DISABLE is set;
//   * the sidecar attaches, reports a coreml-* label, and decodes whole
//     utterances (several windows plus a partial tail, and one shorter than a
//     window) and arbitrary sub-ranges within --min-snr of the ggml CPU decode;
//   * with --ref-dir holding dump-reference.py's case0 fixtures, case0 against
//     the HF wav (45 dB for speech; bounded absolute error for near-silence);
//   * an Engine reports the sidecar (dac_on_coreml, dac_backend, whole and
//     streamed) and PARLER_COREML_DISABLE keeps it on ggml;
//   * --fallback-gguf: a sidecar whose window cannot hold its context is
//     refused at attach (and throws under PARLER_COREML_STRICT).
// Exit 77 (skip) without --gguf; --require-coreml fails instead of skipping
// the Core ML gates (CI).

#include "tts-cpp/parler/engine.h"
#include "tts-cpp/parler/fit.h"
#include "parler/internal.h"
#include "coreml_sidecar_path.h"
#include "npy.h"
#include "../test_env_portable.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tts_cpp::parler::detail;

namespace {

int g_failures = 0;

void check(bool ok, const std::string & what) {
    fprintf(stderr, "  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

// Fills every window with a constant, or fails on call `fail_on` (1-based).
class fake_sidecar final : public parler_dac_sidecar {
public:
    fake_sidecar(int window, int hop, int fail_on) : window_(window), hop_(hop), fail_on_(fail_on) {}
    int window() const override { return window_; }
    const char * label() const override { return "coreml-fake"; }
    bool decode(const float *, const float * valid, float * pcm) override {
        ++calls;
        if (calls == fail_on_) return false;
        for (int t = 0; t < window_; ++t) {
            if (valid[t] != 0.0f && valid[t] != 1.0f) return false;
        }
        std::fill(pcm, pcm + (size_t) window_ * hop_, 0.25f);
        return true;
    }
    int calls = 0;

private:
    int window_, hop_, fail_on_;
};

double snr_db(const std::vector<float> & got, const std::vector<float> & ref, double & max_abs) {
    double sig = 0, noise = 0;
    max_abs = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) got[i] - ref[i];
        sig += (double) ref[i] * ref[i];
        noise += d * d;
        max_abs = std::max(max_abs, std::fabs(d));
    }
    if (!std::isfinite(noise)) return NAN;
    return noise > 0 ? 10.0 * std::log10(sig / noise) : INFINITY;
}

// Relative SNR is ill-conditioned near silence: the Indic Hindi greedy
// fixture is mostly DC at -78.5 dBFS. Keep speech on the relative gate and
// bound BOTH RMS (-70 dBFS) and peak (-60 dBFS) error on quiet HF fixtures.
// Random-code parity always uses the requested relative gate.
struct waveform_quality {
    double snr = NAN;
    double peak = 0;
    double ref_rms = 0;
    double error_rms = 0;
    bool quiet = false;
    bool pass = false;
};

waveform_quality hf_quality(const std::vector<float> & pcm, const std::vector<float> & ref,
                            double min_snr) {
    waveform_quality q;
    if (pcm.empty() || pcm.size() != ref.size()) return q;
    double signal = 0, error = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(pcm[i]) || !std::isfinite(ref[i])) return q;
        const double d = (double) pcm[i] - ref[i];
        signal += (double) ref[i] * ref[i];
        error += d * d;
        q.peak = std::max(q.peak, std::fabs(d));
    }
    q.ref_rms = std::sqrt(signal / ref.size());
    q.error_rms = std::sqrt(error / ref.size());
    q.snr = error > 0 ? 10.0 * std::log10(signal / error) : INFINITY;
    q.quiet = q.ref_rms < 1e-3; // below -60 dBFS, based only on the reference
    q.pass = q.quiet ? (q.error_rms <= std::pow(10.0, -70.0 / 20) && q.peak <= 1e-3)
                     : q.snr >= min_snr;
    return q;
}

void waveform_quality_gates() {
    const std::vector<float> silence(100, 0.0f), speech(100, 0.1f);
    check(hf_quality(silence, silence, 45).pass, "quiet gate: exact silence");
    check(hf_quality(std::vector<float>(100, 1.5e-4f), silence, 45).pass,
          "quiet gate: bounded fp16 floor");
    check(!hf_quality(std::vector<float>(100, 4e-4f), silence, 45).pass,
          "quiet gate: excessive RMS rejected");
    auto spike = silence;
    spike[0] = 2e-3f;
    check(!hf_quality(spike, silence, 45).pass, "quiet gate: isolated peak rejected");
    check(!hf_quality(std::vector<float>(100, 0.101f), speech, 45).pass,
          "speech gate: relative SNR required");
    check(hf_quality(std::vector<float>(100, 0.1001f), speech, 45).pass,
          "speech gate: small error passes");
    const std::vector<float> below(100, 0.000999f), above(100, 0.001001f);
    check(hf_quality(below, below, 45).quiet && !hf_quality(above, above, 45).quiet,
          "quiet gate: reference RMS selects the branch at -60 dBFS");
    auto transient = silence;
    transient[0] = 0.0079f; // low RMS, but the quiet gate must preserve this peak
    check(!hf_quality(silence, transient, 45).pass,
          "quiet gate: erasing a quiet reference transient fails");
    check(!hf_quality({}, silence, 45).pass, "quiet gate: wrong length rejected");
    spike[0] = NAN;
    check(!hf_quality(spike, silence, 45).pass && !hf_quality(silence, spike, 45).pass,
          "quiet gate: nonfinite PCM and reference rejected");
}

std::vector<int32_t> random_codes(const parler_hparams & hp, int n_frames, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> code(0, hp.dac_codebook_size - 1);
    std::vector<int32_t> codes((size_t) hp.dac_n_q * n_frames);
    for (int32_t & c : codes) c = code(rng);
    return codes;
}

bool decode(const parler_model & model, const std::vector<int32_t> & codes, int n_frames,
            std::vector<float> & pcm, std::vector<float> * latent = nullptr, int begin = 0,
            int end = -1, std::string * backend = nullptr, std::string * err = nullptr) {
    return parler_dac_decode(model, codes.data(), n_frames, 2, pcm, latent, begin, end, err, backend);
}

void fake_sidecar_gates(parler_model & model) {
    fprintf(stderr, "fake sidecar gates\n");
    const parler_hparams & hp = model.hparams;
    const int rf = parler_dac_rf_frames(model);
    const int window = 2 * rf + 8;
    const int n_frames = 3 * (window - 2 * rf) + 5;
    const std::vector<int32_t> codes = random_codes(hp, n_frames, 11);

    std::vector<float> ref, ref_lat;
    std::string backend;
    check(decode(model, codes, n_frames, ref, &ref_lat, 0, -1, &backend) && backend == "ggml",
          "ggml reference decode reports ggml");

    // host quantizer == ggml's latent; the fake's pcm is its constant
    auto ok_fake = std::make_shared<fake_sidecar>(window, hp.dac_hop, 0);
    check(parler_attach_dac_sidecar(model, ok_fake), "attach a fake sidecar");
    std::vector<float> pcm, lat;
    check(decode(model, codes, n_frames, pcm, &lat, 0, -1, &backend) && backend == "coreml-fake",
          "sidecar decode reports the sidecar label");
    check(ok_fake->calls == 4, "sidecar walks ceil(n / (W - 2 rf)) windows (" +
          std::to_string(ok_fake->calls) + ")");
    double max_abs = 0;
    for (size_t i = 0; i < lat.size() && i < ref_lat.size(); ++i) {
        max_abs = std::max(max_abs, (double) std::fabs(lat[i] - ref_lat[i]));
    }
    check(lat.size() == ref_lat.size() && max_abs < 1e-4,
          "host RVQ latent == ggml latent (max_abs " + std::to_string(max_abs) + ")");
    check(pcm.size() == ref.size() && pcm.front() == 0.25f && pcm.back() == 0.25f,
          "sidecar pcm covers exactly the range");
    check(decode(model, codes, n_frames, pcm, nullptr, 7, 7) && pcm.empty(), "empty range");
    std::vector<int32_t> bad = codes;
    bad[3] = hp.dac_codebook_size;
    check(!decode(model, bad, n_frames, pcm) && model.dac_sidecar, "out-of-range code rejected before the sidecar");

    // fails on its second window: retired once, whole range redone on ggml
    auto failing = std::make_shared<fake_sidecar>(window, hp.dac_hop, 2);
    check(parler_attach_dac_sidecar(model, failing), "attach a failing fake sidecar");
    check(decode(model, codes, n_frames, pcm, nullptr, 0, -1, &backend) && backend == "ggml",
          "mid-range failure falls back to ggml");
    check(pcm == ref, "fallback output is bit-identical to the ggml-only decode");
    check(!model.dac_sidecar && failing->calls == 2, "failing sidecar retired after one failure");
    check(decode(model, codes, n_frames, pcm, nullptr, 5, n_frames - 3, &backend) && backend == "ggml" &&
          std::equal(pcm.begin(), pcm.end(), ref.begin() + 5 * hp.dac_hop),
          "later ranges stay on ggml");

    // strict: a failure is an error, a missing sidecar throws at attach
    setenv(PARLER_COREML_STRICT_ENV, "1", 1);
    auto strict_fail = std::make_shared<fake_sidecar>(window, hp.dac_hop, 1);
    parler_attach_dac_sidecar(model, strict_fail);
    std::string err;
    check(!decode(model, codes, n_frames, pcm, nullptr, 0, -1, nullptr, &err) && pcm.empty() &&
          err.find(PARLER_COREML_STRICT_ENV) != std::string::npos && !model.dac_sidecar,
          "strict: sidecar failure fails the call");
    check(!decode(model, codes, n_frames, pcm) && pcm.empty(), "strict: later calls keep failing after a retirement");
    bool threw = false;
    try {
        parler_attach_dac_sidecar(model, std::string("/nonexistent/parler-none-f32.gguf"));
    } catch (const std::runtime_error &) {
        threw = true;
    }
    check(threw, "strict: no sidecar throws at attach");
    unsetenv(PARLER_COREML_STRICT_ENV);
    check(decode(model, codes, n_frames, pcm) && pcm == ref, "without strict, a retired sidecar decodes on ggml");
    check(!parler_attach_dac_sidecar(model, std::string("/nonexistent/parler-none-f32.gguf")),
          "no sidecar: attach reports ggml");
}

void coreml_gates(parler_model & model, const std::string & gguf, const std::string & ref_dir,
                  double min_snr) {
    fprintf(stderr, "Core ML sidecar gates (%s)\n", parler_dac_sidecar_path(gguf).c_str());
    model.dac_sidecar.reset();
    if (!parler_attach_dac_sidecar(model, gguf)) {
        check(false, "sidecar attaches");
        return;
    }
    const std::string label = model.dac_sidecar->label();
    check(label.rfind("coreml-", 0) == 0, "label " + label);
    const parler_hparams & hp = model.hparams;
    const int window = model.dac_sidecar->window();
    const int keep = window - 2 * parler_dac_rf_frames(model);

    for (const int n_frames : {2 * keep + keep / 2, window / 3, 1}) {
        const std::vector<int32_t> codes = random_codes(hp, n_frames, 100 + n_frames);
        auto sidecar = model.dac_sidecar;
        model.dac_sidecar.reset();
        std::vector<float> ref, pcm;
        decode(model, codes, n_frames, ref);
        model.dac_sidecar = sidecar;
        std::string backend;
        const bool ok = decode(model, codes, n_frames, pcm, nullptr, 0, -1, &backend);
        double max_abs = 0;
        const double snr = ok && pcm.size() == ref.size() ? snr_db(pcm, ref, max_abs) : NAN;
        check(ok && backend == label && snr >= min_snr,
              std::to_string(n_frames) + " frames vs ggml: SNR " + std::to_string(snr) +
              " dB, max_abs " + std::to_string(max_abs));
        if (n_frames > keep) {
            // three sub-ranges that split windows differently stitch to the whole
            const int a = keep / 3, b = keep + keep / 2;
            std::vector<float> stitched, part;
            for (const auto & r : {std::make_pair(0, a), std::make_pair(a, b), std::make_pair(b, n_frames)}) {
                decode(model, codes, n_frames, part, nullptr, r.first, r.second);
                stitched.insert(stitched.end(), part.begin(), part.end());
            }
            const double ssnr = stitched.size() == ref.size() ? snr_db(stitched, ref, max_abs) : NAN;
            check(ssnr >= min_snr, "sub-ranges stitch: SNR " + std::to_string(ssnr) + " dB");
        }
    }
    check(model.dac_sidecar != nullptr, "sidecar still attached");

    tts_cpp::parler::FitOptions fit;
    fit.model_gguf_path = gguf;
    check(tts_cpp::parler::fit_params(fit).reason == "coreml-memory-estimate-unavailable",
          "fit refuses to project an enabled sidecar");
    setenv(PARLER_COREML_DISABLE_ENV, "1", 1);
    const std::string reason = tts_cpp::parler::fit_params(fit).reason;
    unsetenv(PARLER_COREML_DISABLE_ENV);
    check(reason != "coreml-memory-estimate-unavailable", "PARLER_COREML_DISABLE: fit projects ggml (" + reason + ")");

    if (!ref_dir.empty() && std::ifstream(ref_dir + "/case0_codes.npy").good() &&
        std::ifstream(ref_dir + "/case0_wav_greedy.npy").good()) {
        npy_array codes_npy = npy_load(ref_dir + "/case0_codes.npy");
        npy_array wav_npy = npy_load(ref_dir + "/case0_wav_greedy.npy");
        const int n_frames = (int) codes_npy.shape[2];
        const int64_t * c64 = reinterpret_cast<const int64_t *>(codes_npy.data.data());
        std::vector<int32_t> codes(codes_npy.n_elements());
        for (size_t i = 0; i < codes.size(); ++i) codes[i] = (int32_t) c64[i];
        std::vector<float> pcm;
        std::string backend;
        const bool ok = decode(model, codes, n_frames, pcm, nullptr, 0, -1, &backend);
        const float * w = npy_as_f32(wav_npy);
        const std::vector<float> want(w, w + wav_npy.n_elements());
        const auto q = hf_quality(pcm, want, min_snr);
        check(ok && backend == label && q.pass, "case0 vs HF wav (" + backend + "): SNR " + std::to_string(q.snr) +
              " dB, max_abs " + std::to_string(q.peak) + ", ref RMS " +
              std::to_string(q.ref_rms) + ", error RMS " + std::to_string(q.error_rms) +
              (q.quiet ? " (quiet absolute gate)" : " (relative gate)"));
    }
}

// The public surface: an Engine on the model attaches the sidecar and reports
// it per call, streamed or not; PARLER_COREML_DISABLE keeps it on ggml.
void engine_gates(const std::string & gguf) {
    fprintf(stderr, "Engine gates\n");
    tts_cpp::parler::EngineOptions opts;
    opts.model_gguf_path = gguf;
    opts.default_description = "A calm voice.";
    opts.max_frames = 40;
    opts.stream_chunk_frames = 12;
    for (const bool disabled : {false, true}) {
        if (disabled) setenv(PARLER_COREML_DISABLE_ENV, "1", 1);
        tts_cpp::parler::Engine engine(opts);
        const auto whole = engine.synthesize("a b");
        const auto streamed = engine.synthesize("a b", opts.default_description,
                                                [](const float *, std::size_t, int, bool) {});
        const std::string want = disabled ? "ggml" : "coreml-";
        check(engine.dac_on_coreml() == !disabled &&
              whole.dac_backend.rfind(want, 0) == 0 && streamed.dac_backend.rfind(want, 0) == 0,
              std::string(disabled ? "disabled" : "enabled") + ": dac_on_coreml " +
              (engine.dac_on_coreml() ? "true" : "false") + ", dac_backend " + whole.dac_backend + " / " +
              streamed.dac_backend);
        if (disabled) unsetenv(PARLER_COREML_DISABLE_ENV);
    }
}

void fallback_gates(const std::string & gguf) {
    fprintf(stderr, "short-window sidecar gates (%s)\n", parler_dac_sidecar_path(gguf).c_str());
    parler_model model;
    std::string err;
    if (!parler_load_gguf(gguf, model, 0, &err)) {
        check(false, "load " + gguf + ": " + err);
        return;
    }
    check(!parler_attach_dac_sidecar(model, gguf) && !model.dac_sidecar,
          "a window no wider than its context is refused");
    setenv(PARLER_COREML_STRICT_ENV, "1", 1);
    bool threw = false;
    try {
        parler_attach_dac_sidecar(model, gguf);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    unsetenv(PARLER_COREML_STRICT_ENV);
    check(threw, "strict: refused sidecar throws");
    parler_free_model(model);
}

} // namespace

int main(int argc, char ** argv) {
    std::string gguf, fallback, ref_dir;
    double min_snr = 45.0;
    bool require_coreml = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--gguf") gguf = next();
        else if (a == "--fallback-gguf") fallback = next();
        else if (a == "--ref-dir") ref_dir = next();
        else if (a == "--min-snr") min_snr = std::atof(next().c_str());
        else if (a == "--require-coreml") require_coreml = true;
    }
    if (gguf.empty() || !std::ifstream(gguf).good()) {
        fprintf(stderr, "SKIP: no --gguf model (%s)\n", gguf.c_str());
        return 77;
    }
    unsetenv(PARLER_COREML_STRICT_ENV);
    unsetenv(PARLER_COREML_DISABLE_ENV);

    parler_model model;
    std::string err;
    if (!parler_load_gguf(gguf, model, 0, &err)) {
        fprintf(stderr, "FAIL: load %s: %s\n", gguf.c_str(), err.c_str());
        return 1;
    }
    waveform_quality_gates();
    fake_sidecar_gates(model);
#ifdef TTS_CPP_USE_COREML
    if (::tts_cpp::detail::coreml_sidecar_exists(parler_dac_sidecar_path(gguf))) {
        coreml_gates(model, gguf, ref_dir, min_snr);
        engine_gates(gguf);
    } else {
        fprintf(stderr, "no sidecar next to %s: Core ML gates skipped\n", gguf.c_str());
        check(!require_coreml, "--require-coreml: sidecar present");
    }
    if (!fallback.empty()) fallback_gates(fallback);
#else
    (void) ref_dir;
    (void) min_snr;
    (void) fallback;
    fprintf(stderr, "built without TTS_CPP_COREML: Core ML gates skipped\n");
    check(!require_coreml, "--require-coreml: built with TTS_CPP_COREML");
#endif
    parler_free_model(model);
    fprintf(stderr, g_failures ? "FAILED (%d)\n" : "OK\n", g_failures);
    return g_failures ? 1 : 0;
}
