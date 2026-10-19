#include "cosyvoice_coreml.h"
#include "cosyvoice_pipeline.h"
#include "coreml_sidecar_path.h"
#include "tts-cpp/cosyvoice/fit.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <cmath>
#include <fstream>
#include <filesystem>
#include "test_env_portable.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>

static bool compare(const char *, const std::vector<float> &, const std::vector<float> &, double, double);

struct compute_units_guard {
    const char * name = "COSYVOICE_COREML_COMPUTE_UNITS";
    bool present = std::getenv(name) != nullptr;
    std::string value = present ? std::getenv(name) : "";
    ~compute_units_guard() {
        if (present) setenv(name, value.c_str(), 1);
        else unsetenv(name);
    }
};

static void require_fit_reason(const tts_cpp::cosyvoice::FitOptions & options, const std::string & reason) {
    const auto result = tts_cpp::cosyvoice::fit_params(options);
    if (result.status != tts_cpp::FitStatus::Error || result.fits || result.reason != reason ||
        result.device.total_bytes != 0)
        throw std::runtime_error("unexpected Core ML fit result: " + result.reason);
    if (reason == "coreml-memory-estimate-unavailable" && result.report.empty())
        throw std::runtime_error("missing Core ML fit diagnostic");
}

static void check_coreml_fit() {
    compute_units_guard units;
    setenv(units.name, "cpu_and_ane", 1);
    const auto directory = std::filesystem::path(test_tmpdir()) / ("cosyvoice-coreml-fit-" + test_process_tag());
    std::filesystem::create_directories(directory);
    struct cleanup {
        std::filesystem::path path;
        ~cleanup() { std::error_code error; std::filesystem::remove_all(path, error); }
    } guard{directory};
    tts_cpp::cosyvoice::FitOptions options;
    options.llm_gguf_path = (directory / "llm-f32.gguf").string();
    options.flow_gguf_path = (directory / "flow-f32.gguf").string();
    options.hift_gguf_path = (directory / "hift-f32.gguf").string();
    options.voice_gguf_path = (directory / "voice.gguf").string();
    options.backend = "cpu";
    require_fit_reason(options, "model-unreadable");
    const std::pair<std::string, std::string> stages[] = {
        {options.llm_gguf_path, ""}, {options.flow_gguf_path, ""},
        {options.flow_gguf_path, "-frontend"}, {options.hift_gguf_path, ""},
        {options.hift_gguf_path, "-f0"}};
    for (const auto & stage : stages) {
        const auto path = tts_cpp::detail::coreml_sidecar_path(stage.first, stage.second);
        std::filesystem::create_directory(path);
#ifdef TTS_CPP_USE_COREML
        require_fit_reason(options, "coreml-memory-estimate-unavailable");
#else
        require_fit_reason(options, "model-unreadable");
#endif
        setenv(units.name, "disabled", 1);
        require_fit_reason(options, "model-unreadable");
        setenv(units.name, "cpu_and_ane", 1);
        std::filesystem::remove(path);
    }
}

static std::vector<int> generate_lm(model_ctx & model, const std::vector<int> & text, int steps) {
    constexpr int seed = 42;
    return cosyvoice_llm_generate(model, cosyvoice_qwen_hp(model), text, {2}, steps,
                                  true, seed, steps, nullptr, false);
}

static std::string token_sequence(const std::vector<int> & tokens) {
    std::ostringstream output;
    std::copy(tokens.begin(), tokens.end(), std::ostream_iterator<int>(output, " "));
    return output.str();
}

static void require_tokens(const char * label, const std::vector<int> & actual,
                           const std::vector<int> & expected) {
    if (actual != expected)
        throw std::runtime_error(std::string(label) + ": expected " + token_sequence(expected) +
                                 "; actual " + token_sequence(actual));
}

static void check_lm_routing(const std::string & path) {
    auto model = cosyvoice_load_gguf(path);
    struct cleanup {
        model_ctx & model;
        ~cleanup() { cosyvoice_free(model); }
    } guard{model};
    ggml_backend_cpu_set_n_threads(model.backend, 2);
    constexpr int short_steps = 3;
    constexpr int prefix_rows = 5;
    const std::vector<int> text{1, 2};
    const auto native = generate_lm(model, text, short_steps);
    model.coreml = cosyvoice_coreml::load(path);
    if (!model.coreml) throw std::runtime_error("missing LM generation fixture");
    const auto first = generate_lm(model, text, short_steps);
    const auto first_predictions = model.coreml->predictions();
    const auto alternate = generate_lm(model, {3, 4}, short_steps);
    const auto second = generate_lm(model, text, short_steps);
    require_tokens("Core ML LM differs from native generation", first, native);
    require_tokens("Core ML LM cache reset differs", second, first);
    if (native.size() != short_steps || alternate.size() != short_steps ||
        first_predictions != prefix_rows + short_steps ||
        model.coreml->predictions() != first_predictions)
        throw std::runtime_error("LM generation routing/cache reset failed");
    const auto capacity = model.coreml->dimensions("mask").back();
    constexpr int fixture_context = 16;
    if (capacity != fixture_context) throw std::runtime_error("unexpected routing fixture context");
    model.coreml->clear_predictions();
    const auto fallback = generate_lm(model, text, fixture_context);
    if (model.coreml->predictions() != 0)
        throw std::runtime_error("undersized LM sidecar was selected");
    model.coreml.reset();
    if (fallback.size() != fixture_context || fallback != generate_lm(model, text, fixture_context))
        throw std::runtime_error("LM fallback differs from native generation");
    std::cout << "LM generation: Core ML selected, caches reset, undersized context retained ggml; tokens "
              << token_sequence(first) << '\n';
}

static void check_unavailable_stages() {
    std::vector<float> output{12.0f};
    if (compare("zero-output-regression", {1.0f}, {0.0f}, 0.998, 2.0))
        throw std::runtime_error("zero parity output was accepted");
    if (cosyvoice_coreml::load(test_tmpdir() + "/unavailable-cosyvoice.gguf"))
        throw std::runtime_error("missing Core ML model was selected");
    if (cosyvoice_coreml_frontend({}, {1}, {1}, output, output) ||
        cosyvoice_coreml_euler({}, 64, {}, {}, {}, 1, output) ||
        cosyvoice_coreml_f0({}, {}, 1, output) ||
        cosyvoice_coreml_spectrum({}, {}, 1, {}, 1, output))
        throw std::runtime_error("unavailable Core ML stage must retain ggml");
    if (output != std::vector<float>{12.0f})
        throw std::runtime_error("fallback changed its output before execution");
}

static std::vector<float> read_floats(const model_ctx & model, const char * name) {
    auto * tensor = cosyvoice_get(model, name);
    std::vector<float> data(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(float));
    return data;
}

static bool compare(const char * label, const std::vector<float> & reference,
                     const std::vector<float> & actual, double minimum_cosine, double maximum_error) {
    if (reference.size() != actual.size()) throw std::runtime_error("parity shape mismatch");
    double dot = 0, aa = 0, bb = 0, error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error("non-finite parity output");
        dot += double(reference[i]) * actual[i];
        aa += double(reference[i]) * reference[i];
        bb += double(actual[i]) * actual[i];
        error = std::max(error, double(std::abs(reference[i] - actual[i])));
    }
    const double cosine = dot / std::sqrt(aa * bb);
    std::cerr << label << " cosine=" << cosine << " max_abs=" << error << '\n';
    return std::isfinite(cosine) && cosine >= minimum_cosine && error <= maximum_error;
}

static void dump(const std::string & directory, const char * name, const std::vector<float> & data) {
    if (directory.empty()) return;
    std::filesystem::create_directories(directory);
    std::ofstream out(directory + "/" + name, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
}

static bool hift_parity(const std::string & directory, const std::vector<float> & reference,
                         int native_frames, const std::string & output) {
    auto hift = cosyvoice_load_gguf(directory + "/cosyvoice3-hift-f32.gguf");
    ggml_backend_cpu_set_n_threads(hift.backend, 8);
    const auto f0 = cosyvoice_hift_f0(hift, reference, native_frames);
    const auto waveform = cosyvoice_hift_synth(hift, reference, native_frames, 42, nullptr, &f0);
    hift.coreml_f0 = cosyvoice_coreml::load(directory + "/cosyvoice3-hift-f32.gguf", "-f0");
    const auto coreml_f0 = cosyvoice_hift_f0(hift, reference, native_frames);
    bool passed = compare("f0", f0, coreml_f0, 0.9999, 5.0);
    dump(output, "f0-native.f32", f0);
    dump(output, "f0-coreml.f32", coreml_f0);
    hift.coreml = cosyvoice_coreml::load(directory + "/cosyvoice3-hift-f32.gguf");
    const auto coreml_waveform = cosyvoice_hift_synth(hift, reference, native_frames, 42, nullptr, &f0);
    if (!hift.coreml || !hift.coreml_f0 || !hift.coreml->predictions() || !hift.coreml_f0->predictions())
        throw std::runtime_error("Core ML HiFT did not run");
    passed = compare("hift", waveform, coreml_waveform, 0.999, 0.01) && passed;
    dump(output, "hift-native.f32", waveform);
    dump(output, "hift-coreml.f32", coreml_waveform);
    cosyvoice_free(hift);
    return passed;
 }

static void stage_parity(const std::string & directory, const std::string & output = "") {
    auto voice = cosyvoice_load_gguf(directory + "/voice.gguf");
    auto * token_tensor = cosyvoice_get(voice, "voice/prompt_token");
    std::vector<int> prompt(ggml_nelements(token_tensor));
    ggml_backend_tensor_get(token_tensor, prompt.data(), 0, prompt.size() * sizeof(int));
    const auto features = read_floats(voice, "voice/prompt_feat");
    const auto embedding = read_floats(voice, "voice/embedding");
    cosyvoice_free(voice);
    std::vector<int> tokens;
    std::ifstream stream(directory + "/parity-tokens.txt");
    for (int token; stream >> token;) tokens.push_back(token);
    if (tokens.empty()) throw std::runtime_error("missing parity-tokens.txt");
    auto flow = cosyvoice_load_gguf(directory + "/cosyvoice3-flow-f32.gguf");
    ggml_backend_cpu_set_n_threads(flow.backend, 8);
    int native_frames = 0, coreml_frames = 0;
    const auto reference = cosyvoice_flow_run(flow, prompt, tokens, features, features.size() / 80,
                                             embedding, native_frames);
    bool passed = hift_parity(directory, reference, native_frames, output);
    flow.coreml_frontend = cosyvoice_coreml::load(directory + "/cosyvoice3-flow-f32.gguf", "-frontend");
    flow.coreml = cosyvoice_coreml::load(directory + "/cosyvoice3-flow-f32.gguf");
    const auto actual = cosyvoice_flow_run(flow, prompt, tokens, features, features.size() / 80,
                                          embedding, coreml_frames);
    if (!flow.coreml || !flow.coreml_frontend || !flow.coreml->predictions() ||
        !flow.coreml_frontend->predictions()) throw std::runtime_error("Core ML flow did not run");
    passed = compare("flow", reference, actual, 0.998, 2.0) && passed;
    dump(output, "flow-native.f32", reference);
    dump(output, "flow-coreml.f32", actual);
    cosyvoice_free(flow);
    if (!passed) throw std::runtime_error("Core ML stage parity failed");
}

static std::vector<float> lm_prefix(const std::string & path) {
    auto native = cosyvoice_load_gguf(path);
    auto * speech = cosyvoice_get(native, "lm/speech_embedding/weight");
    auto * text = cosyvoice_get(native, "lm/embed_tokens/weight");
    const auto sos = cosyvoice_meta_i(native, "cosyvoice3.llm.sos", 6561);
    const size_t width = speech->ne[0];
    std::vector<float> prefix(width * 2);
    ggml_backend_tensor_get(speech, prefix.data(), sos * width * sizeof(float), width * sizeof(float));
    ggml_backend_tensor_get(text, prefix.data() + width, 0, width * sizeof(float));
    cosyvoice_free(native);
    return prefix;
}

static std::vector<float> native_lm_logits(const std::string & path, const std::vector<float> & input) {
    auto native = cosyvoice_load_gguf(path);
    const auto hp = cosyvoice_qwen_hp(native);
    constexpr size_t graph_nodes = 8192;
    constexpr int rows = 2;
    const size_t bytes = (graph_nodes + 16) * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_nodes, false);
    auto * context = ggml_init({bytes, nullptr, true});
    auto * graph = ggml_new_graph_custom(context, graph_nodes, false);
    auto * x = ggml_new_tensor_2d(context, GGML_TYPE_F32, hp.hidden, rows);
    auto * positions = ggml_new_tensor_1d(context, GGML_TYPE_I32, rows);
    auto * mask = ggml_new_tensor_2d(context, GGML_TYPE_F32, rows, rows);
    ggml_set_input(x);
    ggml_set_input(positions);
    ggml_set_input(mask);
    auto * logits = build_qwen(context, native, hp, x, positions, mask, rows);
    ggml_build_forward_expand(graph, logits);
    auto * allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(native.backend));
    if (!ggml_gallocr_alloc_graph(allocator, graph)) throw std::runtime_error("native LM fixture allocation failed");
    const int32_t position_values[] = {0, 1};
    const float mask_values[] = {0, -INFINITY, 0, 0};
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(positions, position_values, 0, sizeof(position_values));
    ggml_backend_tensor_set(mask, mask_values, 0, sizeof(mask_values));
    ggml_backend_cpu_set_n_threads(native.backend, 8);
    const auto status = ggml_backend_graph_compute(native.backend, graph);
    std::vector<float> result(ggml_nelements(logits));
    ggml_backend_tensor_get(logits, result.data(), 0, result.size() * sizeof(float));
    ggml_gallocr_free(allocator);
    ggml_free(context);
    cosyvoice_free(native);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("native LM reference failed");
    return result;
}

int main(int argc, char ** argv) {
    try {
#ifndef TTS_CPP_USE_COREML
        if (argc == 2 && (std::string(argv[1]) == "--stages-fixture" ||
                          std::string(argv[1]) == "--lm-fixture" ||
                          std::string(argv[1]) == "--routing-fixture")) return 77;
#endif
        if (argc == 2 && std::string(argv[1]) == "--routing-fixture") {
            const char * file = std::getenv("COSYVOICE_COREML_ROUTING_GGUF");
            if (!file || !*file) return 77;
            check_lm_routing(file);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--stages-fixture") {
            const char * root = std::getenv("COSYVOICE_COREML_TEST_MODELS_DIR");
            if (!root || !*root) return 77;
            stage_parity(root);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--lm-fixture") {
            const char * file = std::getenv("COSYVOICE_COREML_TEST_LM_GGUF");
            if (!file || !*file) return 77;
            argv[1] = const_cast<char *>(file);
        }
        if (argc == 4 && std::string(argv[1]) == "--hift") {
            const auto bytes = std::filesystem::file_size(argv[3]);
            if (!bytes || bytes % (80 * sizeof(float))) throw std::runtime_error("invalid mel fixture");
            std::vector<float> mel(bytes / sizeof(float));
            std::ifstream in(argv[3], std::ios::binary);
            in.read(reinterpret_cast<char *>(mel.data()), bytes);
            if (!in || !hift_parity(argv[2], mel, mel.size() / 80, ""))
                throw std::runtime_error("Core ML HiFT parity failed");
            return 0;
        }
        if ((argc == 3 || argc == 4) && std::string(argv[1]) == "--parity") { stage_parity(argv[2], argc == 4 ? argv[3] : ""); return 0; }
        if (argc == 2) {
            auto model = cosyvoice_coreml::load(argv[1]);
            if (!model) throw std::runtime_error("Core ML fixture did not load");
            model->reset();
            const auto x_shape = model->dimensions("x");
            const auto cosine_shape = model->dimensions("cosine");
            const auto mask_shape = model->dimensions("mask");
            const auto logits_shape = model->dimensions("logits");
            if (model->stateful() || x_shape.size() != 3 || cosine_shape.size() != 4 ||
                mask_shape.size() != 4 || logits_shape.size() != 2)
                throw std::runtime_error("expected explicit-cache LM fixture");
            const auto prefix = lm_prefix(argv[1]);
            std::vector<float> x(prefix.begin(), prefix.begin() + x_shape.back());
            std::vector<float> cosine(cosine_shape.back(), 1), sine(cosine.size());
            std::vector<float> mask(mask_shape.back(), -1e4f), write(mask.size()), first(logits_shape.back()), second(first.size());
            const auto cache_shape = model->dimensions("keys");
            if (cache_shape.size() != 5 || !model->matches("keys_out", cache_shape) ||
                !model->matches("values", cache_shape) || !model->matches("values_out", cache_shape))
                throw std::runtime_error("unexpected LM cache schema");
            size_t count = 1;
            for (int64_t dimension : cache_shape) count *= dimension;
            std::vector<float> keys(count), values(count);
            mask[0] = 0;
            write[0] = 1;
            const auto predict = [&](std::vector<float> & output) {
                model->predict({{"x", x.data(), x_shape}, {"cosine", cosine.data(), cosine_shape},
                                {"sine", sine.data(), cosine_shape}, {"mask", mask.data(), mask_shape},
                                {"write_mask", write.data(), {1, 1, static_cast<int64_t>(write.size()), 1}},
                                {"keys", keys.data(), cache_shape}, {"values", values.data(), cache_shape}},
                               {{"logits", output.data(), logits_shape}, {"keys_out", keys.data(), cache_shape},
                                {"values_out", values.data(), cache_shape}});
            };
            predict(first);
            model->clear_predictions();
            std::vector<float> unchanged{12};
            if (model->predictions() != 0 || cosyvoice_coreml_f0(model, {}, 1, unchanged) ||
                model->predictions() != 0 || unchanged != std::vector<float>{12})
                throw std::runtime_error("Core ML activity survived fallback");
            std::fill(keys.begin(), keys.end(), 0);
            std::fill(values.begin(), values.end(), 0);
            model->reset();
            predict(second);
            if (first != second || model->predictions() != 1)
                throw std::runtime_error("Core ML reset/prediction contract failed");
            std::copy(prefix.begin() + x.size(), prefix.end(), x.begin());
            mask[1] = 0;
            write[0] = 0;
            write[1] = 1;
            auto native = cosyvoice_load_gguf(argv[1]);
            const auto hp = cosyvoice_qwen_hp(native);
            cosyvoice_free(native);
            for (size_t i = 0; i < cosine.size(); ++i) {
                const float angle = std::pow(hp.theta, -2.0f * i / hp.head_dim);
                cosine[i] = std::cos(angle);
                sine[i] = std::sin(angle);
            }
            std::vector<float> next(first.size());
            predict(next);
            const auto expected = native_lm_logits(argv[1], prefix);
            const std::vector<float> first_expected(expected.begin(), expected.begin() + first.size());
            const std::vector<float> next_expected(expected.begin() + first.size(), expected.end());
            const char * output = std::getenv("COSYVOICE_COREML_PARITY_OUTPUT");
            dump(output ? output : "", "lm-native.f32", expected);
            bool lm_passed = compare("lm-position-0", first_expected, first, 0.999, 0.5);
            lm_passed = compare("lm-position-1", next_expected, next, 0.999, 0.5) && lm_passed;
            if (!lm_passed)
                throw std::runtime_error("Core ML LM native parity failed");
            std::cout << model->label() << " stateful=" << model->stateful() << '\n';
            return 0;
        }
        check_coreml_fit();
        check_unavailable_stages();
        setenv("COSYVOICE_COREML_COMPUTE_UNITS", "disabled", 1);
        if (cosyvoice_coreml::load(test_tmpdir() + "/unavailable-cosyvoice.gguf"))
            throw std::runtime_error("disabled Core ML model was selected");
        unsetenv("COSYVOICE_COREML_COMPUTE_UNITS");
        std::cout << "CosyVoice Core ML unavailable-stage and disable checks passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
