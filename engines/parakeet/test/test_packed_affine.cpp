// Exact GPU reconstruction audit. Fails on unsupported ops; no CPU fallback.
#include "packed_affine.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

static bool schema_self_test() {
    ggml_init_params params = {ggml_tensor_overhead() * 8, nullptr, true};
    auto * ctx = ggml_init(params);
    auto * file = gguf_init_empty();
    const char * good = "encoder.blk.0.ff1.linear1.weight";
    auto * codes = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 2);
    auto * minimum = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    auto * step = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_set_name(codes, good);
    ggml_set_name(minimum, (std::string(good) + ".affine.minimum").c_str());
    ggml_set_name(step, (std::string(good) + ".affine.step").c_str());
    gguf_set_val_u32(file, "parakeet.affine.version", 1);
    gguf_set_arr_str(file, "parakeet.affine.names", &good, 1);
    bool valid = parakeet::load_packed_affine(file, ctx).size() == 1;
    const char * bad = "tdt.predict.embed.weight";
    gguf_set_arr_str(file, "parakeet.affine.names", &bad, 1);
    bool rejected = false;
    try { parakeet::load_packed_affine(file, ctx); }
    catch (const std::runtime_error &) { rejected = true; }
    gguf_set_arr_str(file, "parakeet.affine.names", &good, 1);
    gguf_set_val_u32(file, "parakeet.affine.version", 99);
    bool version_rejected = false;
    try { parakeet::load_packed_affine(file, ctx); }
    catch (const std::runtime_error &) { version_rejected = true; }
    gguf_free(file); ggml_free(ctx);
    return valid && rejected && version_rejected;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--self-test") return schema_self_test() ? 0 : 1;
    if (argc != 3) { std::cerr << "usage: test-packed-affine <gguf> <backend-name>\n"; return 2; }
    ggml_backend_load_all();
    auto * device = ggml_backend_dev_by_name(argv[2]);
    if (!device) { std::cerr << "missing requested device\n"; return 3; }
    auto backend = ggml_backend_dev_init(device, nullptr);
    ggml_context * weights = nullptr;
    gguf_init_params gp = {true, &weights};
    auto * file = gguf_init_from_file(argv[1], gp);
    if (!file || !backend) return 4;
    auto records = parakeet::load_packed_affine(file, weights);
    if (records.empty()) return 5;
    // Load only one carrier and its metadata at a time, so auditing does not
    // duplicate the full deployed model or materialize all dense matrices.
    std::ifstream input(argv[1], std::ios::binary);
    size_t tested = 0, elements = 0;
    for (const auto & entry : records) {
        auto * original = const_cast<ggml_tensor *>(entry.first);
        ggml_init_params cp = {ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false), nullptr, true};
        auto * ctx = ggml_init(cp);
        auto * codes = ggml_dup_tensor(ctx, original);
        auto * minimum = ggml_dup_tensor(ctx, entry.second.minimum);
        auto * step = ggml_dup_tensor(ctx, entry.second.step);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<unsigned char> payload;
        for (auto pair : {std::make_pair(original, codes),
                          std::make_pair(entry.second.minimum, minimum),
                          std::make_pair(entry.second.step, step)}) {
            const int id = gguf_find_tensor(file, ggml_get_name(pair.first));
            payload.resize(ggml_nbytes(pair.first));
            input.seekg(gguf_get_data_offset(file) + gguf_get_tensor_offset(file, id));
            input.read(reinterpret_cast<char *>(payload.data()), payload.size());
            if (!input) return 6;
            ggml_backend_tensor_set(pair.second, payload.data(), 0, payload.size());
        }
        auto * graph = ggml_new_graph_custom(ctx, 64, false);
        parakeet::PackedAffineWeights one{{codes, {minimum, step, entry.second.shift}}};
        parakeet::PackedAffineGraphScope scope(ctx, graph, one);
        auto * output = parakeet::affine_weight(ctx, codes, codes);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, i))) return 7;
        }
        auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(allocator, graph) ||
            ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 8;
        std::vector<ggml_fp16_t> actual(ggml_nelements(output));
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(actual[0]));
        std::vector<float> decoded(ggml_nelements(codes)), minima(original->ne[1]), steps(original->ne[1]);
        payload.resize(ggml_nbytes(codes));
        ggml_backend_tensor_get(codes, payload.data(), 0, payload.size());
        // Unit carrier scales are part of the format, never learned weights.
        const size_t block_bytes = ggml_type_size(codes->type);
        for (size_t offset = 0; offset < payload.size(); offset += block_bytes) {
            uint16_t scale; std::memcpy(&scale, payload.data() + offset, 2);
            if (scale != 0x3c00) return 9;
        }
        ggml_get_type_traits(codes->type)->to_float(payload.data(), decoded.data(), decoded.size());
        ggml_backend_tensor_get(minimum, minima.data(), 0, minima.size() * sizeof(float));
        ggml_backend_tensor_get(step, steps.data(), 0, steps.size() * sizeof(float));
        for (size_t i = 0; i < actual.size(); ++i) {
            // volatile prevents the host compiler folding these into an FMA.
            volatile float scaled = (decoded[i] + entry.second.shift) * steps[i / original->ne[0]];
            const auto expected = ggml_fp32_to_fp16(scaled + minima[i / original->ne[0]]);
            if (actual[i] != expected) {
                std::cerr << "MISMATCH " << ggml_get_name(original) << " index=" << i << '\n';
                return 10;
            }
        }
        ++tested; elements += actual.size();
        std::cout << "EXACT " << ggml_get_name(original) << ' ' << actual.size() << '\n';
        ggml_gallocr_free(allocator);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    std::cout << "PASS tensors=" << tested << " elements=" << elements << " backend=" << argv[2] << '\n';
    gguf_free(file); ggml_free(weights); ggml_backend_free(backend);
    return 0;
}
