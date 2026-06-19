// QVAC-20978 — external voice injection.
//
// Two modes, selected by argv:
//
//   * no args        -> model-free unit tests for the voice-JSON parser /
//                       schema validation (always runs on a fresh checkout
//                       under `ctest -L unit`).
//
//   * argv[1] = GGUF -> preset-as-JSON parity: synthesize a sentence with a
//                       baked-in preset selected by name, then again with
//                       the *same* voice injected (a) as in-memory tensors
//                       and (b) via a voice JSON file, and assert the PCM is
//                       identical.  This is the acceptance test from the
//                       ticket ("preset-as-JSON output is identical to the
//                       baked-in preset").  Fixture-gated, so it auto-disables
//                       when the GGUF isn't staged.

#include "tts-cpp/supertonic/engine.h"

#include "supertonic_internal.h"
#include "supertonic_voice_json.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using tts_cpp::supertonic::Engine;
using tts_cpp::supertonic::EngineOptions;
using tts_cpp::supertonic::SynthesisResult;
using tts_cpp::supertonic::detail::external_voice;
using tts_cpp::supertonic::detail::parse_supertonic_voice_json;

namespace {

int g_failures = 0;

void check(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_failures;
    }
}

int run_parser_unit_tests() {
    // Happy path: canonical object form with data + shape, name in metadata.
    {
        external_voice v;
        std::string    err;
        const std::string j =
            R"({"style_ttl":{"data":[1.0,2.0,3.0,4.0],"shape":[1,2,2]},)"
            R"("style_dp":{"data":[0.5,-0.5]},"metadata":{"name":"marco"}})";
        check(parse_supertonic_voice_json(j, v, &err), "object-form parse should succeed");
        check(v.ttl.size() == 4, "ttl size == 4");
        check(v.dp.size() == 2, "dp size == 2");
        check(v.ttl[2] == 3.0f, "ttl[2] == 3.0");
        check(v.dp[1] == -0.5f, "dp[1] == -0.5");
        check(v.name == "marco", "name read from metadata");
    }

    // Bare-array form is also accepted.
    {
        external_voice v;
        std::string    err;
        const std::string j = R"({"style_ttl":[1,2,3],"style_dp":[9]})";
        check(parse_supertonic_voice_json(j, v, &err), "bare-array parse should succeed");
        check(v.ttl.size() == 3 && v.dp.size() == 1, "bare-array sizes");
    }

    // Missing required key fails.
    {
        external_voice v;
        std::string    err;
        const std::string j = R"({"style_ttl":{"data":[1]}})";
        check(!parse_supertonic_voice_json(j, v, &err), "missing style_dp should fail");
        check(!err.empty(), "missing-key error message is set");
    }

    // shape product != data length fails.
    {
        external_voice v;
        std::string    err;
        const std::string j = R"({"style_ttl":{"data":[1,2,3],"shape":[2,2]},"style_dp":[1]})";
        check(!parse_supertonic_voice_json(j, v, &err), "shape/data mismatch should fail");
    }

    // A zero dimension in shape is invalid (would make prod==0 and bypass
    // the data-length cross-check) — must be rejected.
    {
        external_voice v;
        std::string    err;
        const std::string j = R"({"style_ttl":{"data":[1,2,3],"shape":[0,3]},"style_dp":[1]})";
        check(!parse_supertonic_voice_json(j, v, &err), "zero shape dim should fail");
    }

    // A negative dimension in shape is invalid — must be rejected.
    {
        external_voice v;
        std::string    err;
        const std::string j = R"({"style_ttl":{"data":[1,2,3],"shape":[-1,3]},"style_dp":[1]})";
        check(!parse_supertonic_voice_json(j, v, &err), "negative shape dim should fail");
    }

    // Malformed JSON fails (no crash).
    {
        external_voice v;
        std::string    err;
        check(!parse_supertonic_voice_json("{not json", v, &err), "malformed json should fail");
    }

    // Top-level name wins over metadata.name.
    {
        external_voice v;
        std::string    err;
        const std::string j =
            R"({"name":"top","style_ttl":[1],"style_dp":[2],"metadata":{"name":"meta"}})";
        check(parse_supertonic_voice_json(j, v, &err), "name-precedence parse");
        check(v.name == "top", "top-level name wins over metadata.name");
    }

    if (g_failures == 0) {
        std::printf("test-supertonic-external-voice: parser unit tests OK\n");
    }
    return g_failures == 0 ? 0 : 1;
}

std::string make_voice_json(const std::vector<float> & ttl, const std::vector<float> & dp) {
    std::ostringstream oss;
    oss << std::setprecision(9);  // FLT_DECIMAL_DIG: round-trips binary32 exactly
    oss << "{\"style_ttl\":{\"data\":[";
    for (size_t i = 0; i < ttl.size(); ++i) {
        if (i) oss << ',';
        oss << ttl[i];
    }
    oss << "]},\"style_dp\":{\"data\":[";
    for (size_t i = 0; i < dp.size(); ++i) {
        if (i) oss << ',';
        oss << dp[i];
    }
    oss << "]},\"metadata\":{\"name\":\"parity\"}}";
    return oss.str();
}

float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return 1e30f;
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::abs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int run_parity_test(const char * gguf_path) {
    using tts_cpp::supertonic::detail::load_supertonic_gguf;
    using tts_cpp::supertonic::detail::free_supertonic_model;
    using tts_cpp::supertonic::detail::supertonic_model;

    // Pull the default preset's baked tensors out to host vectors so we can
    // re-inject the very same numbers through the external-voice path.
    supertonic_model model;
    if (!load_supertonic_gguf(gguf_path, model)) {
        std::fprintf(stderr, "parity: failed to load GGUF: %s\n", gguf_path);
        return 1;
    }
    std::string voice_name = model.hparams.default_voice;
    auto it = model.voices.find(voice_name);
    if (it == model.voices.end()) {
        it = model.voices.begin();
    }
    if (it == model.voices.end()) {
        std::fprintf(stderr, "parity: model has no voices\n");
        free_supertonic_model(model);
        return 1;
    }
    voice_name = it->first;
    std::vector<float> ttl((size_t) ggml_nelements(it->second.ttl));
    std::vector<float> dp((size_t) ggml_nelements(it->second.dp));
    ggml_backend_tensor_get(it->second.ttl, ttl.data(), 0, ggml_nbytes(it->second.ttl));
    ggml_backend_tensor_get(it->second.dp, dp.data(), 0, ggml_nbytes(it->second.dp));
    free_supertonic_model(model);

    const std::string text = "Hello world, this is a parity test.";

    // A — baked-in preset selected by name (the reference).
    EngineOptions oa;
    oa.model_gguf_path = gguf_path;
    oa.voice           = voice_name;
    oa.n_gpu_layers    = 0;
    Engine          ea(oa);
    SynthesisResult pa = ea.synthesize(text);
    check(!pa.pcm.empty(), "baked-preset synthesis produced audio");

    // B — same voice injected as in-memory tensors.
    EngineOptions ob;
    ob.model_gguf_path  = gguf_path;
    ob.voice_style_ttl  = ttl;
    ob.voice_style_dp   = dp;
    ob.n_gpu_layers     = 0;
    Engine          eb(ob);
    SynthesisResult pb = eb.synthesize(text);

    // C — same voice injected through a voice JSON file.
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path() / "supertonic_voice_parity.json";
    {
        std::ofstream f(tmp, std::ios::binary);
        f << make_voice_json(ttl, dp);
    }
    EngineOptions oc;
    oc.model_gguf_path = gguf_path;
    oc.voice_json_path = tmp.string();
    oc.n_gpu_layers    = 0;
    Engine          ec(oc);
    SynthesisResult pc = ec.synthesize(text);
    std::error_code ec_rm;
    std::filesystem::remove(tmp, ec_rm);

    // Same backend + seed + identical style vectors => bit-exact for the
    // tensor path; the JSON path round-trips float32 at 9 sig-digits so it
    // is exact too (allow a hair of slack for the textual round-trip).
    const float ab = max_abs_diff(pa.pcm, pb.pcm);
    const float ac = max_abs_diff(pa.pcm, pc.pcm);
    std::printf("parity: voice=%s samples=%zu  max|A-B|=%.3e  max|A-C|=%.3e\n",
                voice_name.c_str(), pa.pcm.size(), ab, ac);
    check(pa.pcm.size() == pb.pcm.size(), "A/B sample count matches");
    check(pa.pcm.size() == pc.pcm.size(), "A/C sample count matches");
    check(ab == 0.0f, "injected tensors are bit-exact vs baked preset");
    check(ac <= 1e-5f, "injected JSON matches baked preset within fp round-trip");

    if (g_failures == 0) {
        std::printf("test-supertonic-external-voice: parity OK (%s)\n", gguf_path);
    }
    return g_failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc >= 2) {
        return run_parity_test(argv[1]);
    }
    return run_parser_unit_tests();
}
