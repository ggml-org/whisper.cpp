// Model-backed checks of repacked projection weights and fixed-input CPU/OpenCL parity.
#include "audio8/internal.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
using namespace tts_cpp::audio8::detail;
struct model_owner {
    lm_model model;
    ~model_owner() { free_lm(model); }
};
void require(bool pass, const std::string & message) {
    if (!pass)
        throw std::runtime_error(message);
}
struct graph_owner {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~graph_owner() {
        if (buffer)
            ggml_backend_buffer_free(buffer);
        if (ctx)
            ggml_free(ctx);
    }
};
std::vector<float> product(lm_model & model, ggml_tensor * weight, const std::vector<float> & input,
                           int width) {
    graph_owner held;
    held.ctx = ggml_init(
        {ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(4, false), nullptr, true});
    require(held.ctx, "graph context");
    auto * x = ggml_new_tensor_2d(held.ctx, GGML_TYPE_F32, weight->ne[0], width);
    auto * y = ggml_mul_mat(held.ctx, weight, x);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    auto * graph = ggml_new_graph_custom(held.ctx, 4, false);
    ggml_build_forward_expand(graph, y);
    require(ggml_backend_supports_op(model.backend, y), "unsupported projection");
    held.buffer = ggml_backend_alloc_ctx_tensors(held.ctx, model.backend);
    require(held.buffer, "graph buffer");
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(model.backend));
    auto threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
    if (threads)
        threads(model.backend, 4);
    require(ggml_backend_graph_compute(model.backend, graph) == GGML_STATUS_SUCCESS,
            "projection compute");
    std::vector<float> output(ggml_nelements(y));
    ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
    return output;
}
bool metric(const std::string & label, const std::vector<float> & reference,
            const std::vector<float> & candidate) {
    require(reference.size() == candidate.size(), "output shape");
    double dot = 0, a = 0, b = 0, e = 0, max_abs = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        require(std::isfinite(reference[i]) && std::isfinite(candidate[i]), "nonfinite output");
        double r = reference[i], g = candidate[i];
        dot += r * g;
        a += r * r;
        b += g * g;
        e += (r - g) * (r - g);
        max_abs = std::max(max_abs, std::abs(r - g));
    }
    double cosine = a && b ? dot / std::sqrt(a * b) : (a == b ? 1 : 0);
    double nmse = a ? e / a : (e ? INFINITY : 0);
    bool pass = std::isfinite(cosine) && std::isfinite(nmse) && cosine >= 0.9999 && nmse <= 2e-4;
    std::printf("projection label=%s cosine=%.12f nmse=%.12g max_abs=%.12g gate=%s\n",
                label.c_str(), cosine, nmse, max_abs, pass ? "PASS" : "FAIL");
    return pass;
}
int main(int argc, char ** argv) {
    try {
        std::setvbuf(stdout, nullptr, _IOLBF, 0);
        const char * path = argc == 2 ? argv[1] : std::getenv("AUDIO8_PARITY_LM");
        if (!path || !std::ifstream(path).good()) {
            std::fprintf(stderr,
                         "usage: test-audio8-hexagon-repack LM.gguf (or AUDIO8_PARITY_LM)\n");
            return 77;
        }
        std::string error;
        model_owner cpu, hexagon, opencl;
        require(load_lm(path, 0, "cpu", cpu.model, &error), error);
        if (!ggml_backend_reg_by_name("HTP") || !ggml_backend_reg_by_name("OpenCL")) {
            std::fprintf(stderr, "Hexagon/OpenCL providers unavailable; skipping device fixture\n");
            return 77;
        }
        require(load_lm(path, 99, "hexagon", hexagon.model, &error), error);
        require(load_lm(path, 99, "opencl", opencl.model, &error), error);
        require(!hexagon.model.repacked_projection_buffers.empty(), "no repack buffers");
        model_owner metadata;
        fit_load_measure measured;
        require(load_lm_metadata_only(path, 99, "hexagon", metadata.model, measured, &error),
                error);
        std::set<ggml_backend_buffer_t> weight_buffers;
        for (auto * t = ggml_get_first_tensor(hexagon.model.ctx_w); t;
             t = ggml_get_next_tensor(hexagon.model.ctx_w, t))
            weight_buffers.insert(t->buffer);
        size_t allocated = 0;
        for (auto buffer : weight_buffers)
            allocated += ggml_backend_buffer_get_size(buffer);
        require(allocated == measured.weights_bytes + 4096 * weight_buffers.size(),
                "metadata weight size does not match allocation plus guard pages");
        std::printf(
            "memory buffers=%zu payload_bytes=%zu allocated_bytes=%zu guard_bytes=%zu PASS\n",
            weight_buffers.size(), measured.weights_bytes, allocated, 4096 * weight_buffers.size());
        std::set<std::string> expected_projections;
        for (const auto * branch : {&cpu.model.blocks, &cpu.model.fast_blocks}) {
            for (const auto & block : *branch) {
                for (const auto * tensor :
                     {block.attn.wqkv, block.attn.wq, block.attn.wk, block.attn.wv, block.attn.wo,
                      block.w13, block.w1, block.w2, block.w3}) {
                    if (tensor && tensor->type == GGML_TYPE_Q8_0 && tensor->ne[0] % 32 == 0 &&
                        tensor->ne[1] <= 32768)
                        expected_projections.insert(ggml_get_name(tensor));
                }
            }
        }
        int cases = 0, failures = 0, roundtrips = 0;
        for (auto * w = ggml_get_first_tensor(cpu.model.ctx_w); w;
             w = ggml_get_next_tensor(cpu.model.ctx_w, w)) {
            auto * h = ggml_get_tensor(hexagon.model.ctx_w, ggml_get_name(w));
            auto * c = ggml_get_tensor(opencl.model.ctx_w, ggml_get_name(w));
            require(h && c, "missing weight");
            const bool repacked = std::strstr(ggml_backend_buffer_name(h->buffer), "-REPACK");
            require(repacked == bool(expected_projections.count(ggml_get_name(w))),
                    "wrong weight buffer: " + std::string(ggml_get_name(w)));
            if (!repacked)
                continue;
            std::vector<unsigned char> raw(ggml_nbytes(w)), restored(raw.size());
            ggml_backend_tensor_get(w, raw.data(), 0, raw.size());
            ggml_backend_tensor_get(h, restored.data(), 0, restored.size());
            require(raw == restored, "repack roundtrip: " + std::string(ggml_get_name(w)));
            ++roundtrips;
            for (int width : {1, 2, 30}) {
                std::vector<float> input(w->ne[0] * width);
                unsigned state = 0x67543219u;
                for (auto & value : input) {
                    state = state * 1664525u + 1013904223u;
                    value = (float(int(state >> 8) % 65536) - 32768) / 8192.0f;
                }
                auto reference = product(cpu.model, w, input, width),
                     hex = product(hexagon.model, h, input, width),
                     cl = product(opencl.model, c, input, width);
                auto label = std::string(ggml_get_name(w)) + ".width" + std::to_string(width);
                failures += !metric(label + ".hexagon_vs_cpu", reference, hex);
                failures += !metric(label + ".hexagon_vs_opencl", cl, hex);
                failures += !metric(label + ".opencl_vs_cpu", reference, cl);
                cases += 3;
            }
        }
        require(roundtrips > 0 && size_t(roundtrips) == expected_projections.size(),
                "not all eligible projections were tested");
        std::printf("summary roundtrip_weights=%d comparisons=%d failures=%d %s\n", roundtrips,
                    cases, failures, failures ? "FAIL" : "PASS");
        return failures ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "projection parity: %s\n", e.what());
        return 1;
    }
}
