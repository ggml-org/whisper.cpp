#include "pocket/flow_lm.h"
#include "test_backend.h"
#include "npy.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using tts_cpp::pocket::detail::FlowLM;

namespace {
std::vector<float> load(const std::string & directory, const std::string & name) {
    const auto a = npy_load(directory + "/" + name + ".npy");
    if (a.dtype != "<f4") throw std::runtime_error("expected F32 fixture: " + name);
    std::vector<float> result(a.n_elements());
    std::memcpy(result.data(), a.data.data(), result.size() * sizeof(float));
    return result;
}
void check(const std::string & name, const std::vector<float> & got,
           const std::vector<float> & expected, double tolerance) {
    if (got.size() != expected.size()) throw std::runtime_error(name + ": size mismatch");
    double error = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(expected[i])) throw std::runtime_error(name + ": non-finite value");
        error = std::max(error, std::abs(double(got[i]) - expected[i]));
    }
    std::printf("%s max_abs=%.8g tolerance=%.8g\n", name.c_str(), error, tolerance);
    if (error > tolerance) throw std::runtime_error(name + ": parity failure");
}
template<class F> void rejects(const char * name, F fn) {
    try { fn(); } catch (const std::exception &) { return; }
    throw std::runtime_error(std::string("expected rejection: ") + name);
}
struct CorruptModel {
    std::filesystem::path path;
    explicit CorruptModel(const std::string & source) {
        path = std::filesystem::temp_directory_path() / ("pocket-payload-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
        std::filesystem::copy_file(source, path);
    }
    ~CorruptModel() { std::error_code e; std::filesystem::remove(path, e); }
    bool replace(const char * name, const std::vector<float> & values) {
        ggml_context * c = nullptr;
        auto * f = gguf_init_from_file(path.string().c_str(), {true, &c});
        if (!f || !c) throw std::runtime_error("cannot read corrupt-model fixture");
        auto * t = ggml_get_tensor(c, name);
        const bool f32 = t && t->type == GGML_TYPE_F32;
        const auto index = gguf_find_tensor(f, name);
        const auto offset = index >= 0 ? gguf_get_data_offset(f) + gguf_get_tensor_offset(f, index) : 0;
        const bool size_ok = t && size_t(ggml_nelements(t)) == values.size();
        gguf_free(f); ggml_free(c);
        if (!size_ok) throw std::runtime_error("payload fixture shape mismatch");
        if (!f32) return false;
        std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out);
        out.seekp(offset); out.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float));
        if (!out) throw std::runtime_error("payload fixture write failed");
        return true;
    }
};
} // namespace

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr, "usage: test-pocket-flow-lm model.gguf fixture-directory absolute-tolerance [cpu|opencl|hexagon|HTP0|--gpu]\n");
        return 2;
    }
    try {
        const std::string model = argv[1], dir = argv[2];
        const double tol = std::stod(argv[3]);
        if (!std::isfinite(tol) || tol <= 0) throw std::runtime_error("invalid tolerance");
        auto backend = pocket_test_backend(argc == 5 ? argv[4] : "cpu");
        const auto device = ggml_backend_get_device(backend.get());
        FlowLM lm(model, 64, 2, device);
        if (std::string(lm.backend_name()) != ggml_backend_name(backend.get()))
            throw std::runtime_error("FlowLM fixture ran on a different backend");
        if (argc != 5 || std::string(argv[4]) == "--gpu") check_pocket_backend(lm.backend_name(), argc == 5);
        std::printf("FlowLM backend: %s\n", lm.backend_name());
        const auto embeddings = load(dir, "embeddings");
        const auto expected = load(dir, "prefill");
        check("text embeddings", lm.text_embeddings({1, 7, 3, 12, 5}), load(dir, "text_embeddings"), tol);
        check("prefill", lm.prefill(embeddings), expected, tol);
        const int rows = static_cast<int>(embeddings.size() / lm.config().dim);
        if (lm.position() != rows) throw std::runtime_error("prefill position mismatch");
        std::vector<float> previous;
        for (int step = 0; step < 4; ++step) {
            const std::string p = "step" + std::to_string(step);
            auto frame = lm.advance(previous);
            check(p + " hidden", frame.hidden, load(dir, p + "_hidden"), tol);
            check(p + " eos", {frame.eos_logit}, load(dir, p + "_eos"), tol);
            const auto noise = load(dir, p + "_noise");
            for (int steps : {1, 2, 4, 16}) {
                // Isolate flow parity from accumulation of backbone error.
                auto latent = lm.sample(load(dir, p + "_hidden"), noise, steps);
                check(p + " sample" + std::to_string(steps), latent,
                      load(dir, p + "_sample" + std::to_string(steps)), tol);
            }
            previous = load(dir, p + "_sample4");
            check(p + " denormalize", lm.denormalize(previous), load(dir, p + "_denormalized"), tol);
        }
        // Chunking the same prefix must preserve positions and causal history.
        lm.reset();
        const size_t split = 3 * lm.config().dim;
        auto first = lm.prefill({embeddings.begin(), embeddings.begin() + split});
        auto second = lm.prefill({embeddings.begin() + split, embeddings.end()});
        first.insert(first.end(), second.begin(), second.end());
        check("chunked prefill", first, expected, tol);
        const int before = lm.position();
        rejects("empty prefill", [&] { lm.prefill({}); });
        rejects("wrong latent width", [&] { lm.advance({1, 2}); });
        rejects("bad token id", [&] { lm.text_embeddings({lm.config().vocab_size}); });
        rejects("negative token id", [&] { lm.text_embeddings({-1}); });
        auto bad = embeddings; bad[0] = std::numeric_limits<float>::quiet_NaN();
        rejects("NaN embeddings", [&] { lm.prefill(bad); });
        rejects("context overflow", [&] { lm.prefill(std::vector<float>(65 * lm.config().dim)); });
        rejects("zero flow steps", [&] { lm.sample(load(dir, "step0_hidden"), load(dir, "step0_noise"), 0); });
        if (lm.position() != before) throw std::runtime_error("invalid input changed cache position");
        lm.reset();
        check("reset and reuse", lm.prefill(embeddings), expected, tol);
        // Also replay an end-to-end native latent trajectory: each sampled
        // latent, rather than the oracle's latent, feeds the next AR step.
        previous.clear();
        for (int step = 0; step < 4; ++step) {
            const std::string p = "step" + std::to_string(step);
            const auto frame = lm.advance(previous);
            previous = lm.sample(frame.hidden, load(dir, p + "_noise"), 4);
            check(p + " native trajectory", previous, load(dir, p + "_sample4"), tol);
        }
        rejects("zero context", [&] { FlowLM invalid(model, 0); });
        rejects("zero threads", [&] { FlowLM invalid(model, 64, 0); });
        // Exact capacity remains usable, and rejection cannot poison it.
        lm.reset();
        std::vector<float> full(64*lm.config().dim);
        for (size_t i = 0; i < full.size(); ++i) full[i] = embeddings[i%embeddings.size()];
        const auto full_result = lm.prefill(full);
        if (lm.position() != 64) throw std::runtime_error("exact capacity not reached");
        rejects("advance at capacity", [&] { lm.advance(); });
        if (lm.position() != 64) throw std::runtime_error("overflow poisoned cache");
        lm.reset();
        std::vector<float> split_result;
        for (int first = 0; first < 64; first += 8) {
            auto part = lm.prefill({full.begin()+first*lm.config().dim, full.begin()+(first+8)*lm.config().dim});
            split_result.insert(split_result.end(), part.begin(), part.end());
        }
        check("full context vs eight chunks", split_result, full_result, tol);
        for (float bad_value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
            CorruptModel bad(model); bad.replace("out_eos.bias", {bad_value});
            rejects("non-finite EOS weight payload", [&] { FlowLM invalid(bad.path.string(), 64, 2, device); });
        }
        {
            CorruptModel bad(model);
            auto weights = load(dir, "step0_hidden");
            for (auto & w : weights) w = std::copysign(std::numeric_limits<float>::max(), w);
            if (bad.replace("out_eos.weight", weights)) {
                bad.replace("out_eos.bias", {std::numeric_limits<float>::max()});
                FlowLM overflow(bad.path.string(), 64, 2, device);
                overflow.prefill(embeddings);
                rejects("finite weights causing EOS overflow", [&] { overflow.advance(); });
                if (overflow.position() != 0) throw std::runtime_error("EOS overflow did not reset state");
                check("reuse after EOS overflow", overflow.prefill(embeddings), expected, tol);
            }
        }
        std::puts("Pocket FlowLM parity and lifecycle checks passed");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
