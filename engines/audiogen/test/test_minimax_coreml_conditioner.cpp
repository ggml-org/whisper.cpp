#include "minimax/mm3-cond-graph.h"
#include "minimax/mm3-replay-io.h"

#include <cstdlib>
#include <stdexcept>

namespace {
void require(bool ok, const std::string & error) {
    if (!ok) throw std::runtime_error(error);
}
}

// Original GGUF weights + recorded AR hiddens. Loads only the conditioner,
// keeping this parity test independent of the 22 GB language/synth stack.
int main() {
    const char * synth = std::getenv("AUDIOGEN_TEST_MINIMAX_SYNTH");
    const char * sidecar = std::getenv("AUDIOGEN_TEST_MINIMAX_COREML_CONDITIONER");
    const char * hiddens_path = std::getenv("AUDIOGEN_TEST_MINIMAX_HIDDENS");
    if (!synth || !sidecar || !hiddens_path) {
        std::puts("SKIP: set AUDIOGEN_TEST_MINIMAX_SYNTH, AUDIOGEN_TEST_MINIMAX_COREML_CONDITIONER and AUDIOGEN_TEST_MINIMAX_HIDDENS");
        return 77;
    }
    MM3Model model;
    GGUFModel file{};
    MM3CoreMLConditioner * coreml = nullptr;
    int status = 0;
    try {
        backend_configure_cpu(8, "");
        const char * device = std::getenv("AUDIOGEN_TEST_MINIMAX_DEVICE");
        backend_configure_device(device ? device : "cpu");
        const auto backend = backend_init("conditioner-parity");
        model.backend = backend.backend;
        model.cpu_backend = backend.cpu_backend;
        model.backend_ref = true;
        require(gf_load(&file, synth), "Cannot map synth GGUF");
        mm3_parse_synth_config(file, &model.synth_cfg);
        const auto & c = model.synth_cfg.cond;
        require(c.num_layers && c.hidden_dim && c.out_dim, "Invalid conditioner configuration");
        std::vector<std::string> errors;
        wctx_init(&model.wctx_synth, 4);
        MM3Loader loader{&model.wctx_synth, &file, &model.tmap_synth, &errors};
        model.synth.cond.layer_logits = loader.req("cond.layer_logits", c.num_layers);
        model.synth.cond.layer_scale = loader.req("cond.layer_scale", 1);
        model.synth.cond.proj_w = loader.req("cond.proj.weight", c.kernel_size, c.hidden_dim, c.out_dim);
        model.synth.cond.proj_b = loader.req("cond.proj.bias", c.out_dim);
        require(errors.empty(), errors.empty() ? "" : errors.front());
        require(wctx_alloc(&model.wctx_synth, model.backend), "Cannot allocate conditioner weights");
        gf_close(&file);
        model.loaded = true;
        std::string error;
        coreml = mm3_coreml_conditioner_load(sidecar, c.hidden_dim, c.out_dim, &error);
        require(coreml, error);
        auto * incompatible = mm3_coreml_conditioner_load(sidecar, c.hidden_dim + 1, c.out_dim, &error);
        const bool rejected_shape = incompatible == nullptr;
        mm3_coreml_conditioner_free(incompatible);
        require(rejected_shape, "Accepted a projection with incompatible hidden dimension");
        std::vector<float> invalid_output;
        const float empty_input = 0;
        require(!mm3_coreml_conditioner_project(coreml, &empty_input, 0, invalid_output, &error),
                "Accepted zero projection frames");
        require(!mm3_coreml_conditioner_project(coreml, nullptr, 1, invalid_output, &error),
                "Accepted a null projection input");

        MM3Model missing_native;
        missing_native.lm_file.found = missing_native.lm_file.probe_ok = true;
        missing_native.synth_file.found = missing_native.synth_file.probe_ok = true;
        missing_native.lm_file.path = "missing-coreml-test-lm.gguf";
        missing_native.coreml_conditioner_path = sidecar;
        missing_native.synth_cfg = model.synth_cfg;
        const int refs_before = g_backend_refs;
        const bool native_load_failed = !mm3_load(&missing_native, &error);
        const bool released = !missing_native.coreml_conditioner && !missing_native.backend_ref;
        mm3_unload(&missing_native);
        require(native_load_failed && released && g_backend_refs == refs_before,
                "Native load failure leaked its Core ML model or backend");
        std::vector<float> fixture;
        require(mm3_replay_read_raw(hiddens_path, fixture), "Cannot read recorded AR hiddens");
        const size_t row = (size_t) c.num_layers * c.hidden_dim;
        require(fixture.size() >= row && fixture.size() % row == 0, "Invalid recorded hidden layout");
        for (int64_t frames : {1, 149, 150, 151, 301}) {
            std::vector<float> input((size_t) frames * row);
            for (size_t i = 0; i < input.size(); ++i) input[i] = fixture[i % fixture.size()];
            std::vector<float> reference, actual;
            int64_t native_length = 0, coreml_length = 0;
            model.coreml_conditioner = nullptr;
            require(mm3_cond_encode(model, input.data(), frames, reference, &native_length, &error), error);
            model.coreml_conditioner = coreml;
            const auto start = std::chrono::steady_clock::now();
            require(mm3_cond_encode(model, input.data(), frames, actual, &coreml_length, &error), error);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            require(native_length == coreml_length && reference.size() == actual.size(), "Resampling shape differs");
            double aa = 0, bb = 0, ab = 0, diff = 0;
            for (size_t i = 0; i < reference.size(); ++i) {
                require(std::isfinite(actual[i]), "Non-finite Core ML output");
                const double a = reference[i], b = actual[i];
                aa += a * a; bb += b * b; ab += a * b; diff += (a - b) * (a - b);
            }
            const double relative_rms = std::sqrt(diff / aa), cosine = ab / std::sqrt(aa * bb);
            std::printf("frames=%lld latents=%lld relative_rms=%.9g cosine=%.9g coreml_ms=%.3f\n",
                        (long long) frames, (long long) coreml_length, relative_rms, cosine, ms);
            require(relative_rms <= 0.002 && cosine >= 0.9999, "Core ML conditioner failed quality parity");
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        status = 1;
    }
    gf_close(&file);
    model.coreml_conditioner = coreml;
    mm3_cond_free(&g_mm3_cond);
    mm3_unload(&model);
    return status;
}
