#include "cosyvoice_pipeline.h"
#include "backend_util.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kHeadDim = 64;
constexpr int kHeads = 16;
constexpr int kBatches = 2;
constexpr size_t kContextBytes = 1024 * 1024;
constexpr double kTolerance = 0.005;
constexpr float kQueryFrequency = 0.031f;
constexpr float kKeyFrequency = 0.017f;
constexpr float kValueFrequency = 0.023f;
constexpr int kUnsupportedHeadDim = 7;

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct attention_graph {
    ggml_context * ctx = ggml_init({kContextBytes, nullptr, true});
    ggml_backend_buffer_t buffer = nullptr;
    ~attention_graph() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
};

std::vector<float> make_values(size_t count, float frequency) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) values[i] = std::sin(float(i) * frequency);
    return values;
}

size_t input_offset(int length, int batch, int head, int token) {
    return ((size_t(batch) * kHeads + head) * length + token) * kHeadDim;
}

double dot(const float * query, const float * key) {
    double value = 0;
    for (int d = 0; d < kHeadDim; ++d) value += double(query[d]) * key[d];
    return value / std::sqrt(double(kHeadDim));
}

std::vector<double> scores(const float * query, const float * keys, int length) {
    std::vector<double> values(length);
    for (int t = 0; t < length; ++t) values[t] = dot(query, keys + t * kHeadDim);
    return values;
}

double normalize(std::vector<double> & weights) {
    const double maximum = *std::max_element(weights.begin(), weights.end());
    double total = 0;
    for (double & weight : weights) {
        weight = std::exp(weight - maximum);
        total += weight;
    }
    return total;
}

double weighted_value(const std::vector<double> & weights, double total, const float * values, int dimension) {
    double result = 0;
    for (size_t t = 0; t < weights.size(); ++t) result += weights[t] * values[t * kHeadDim + dimension];
    return result / total;
}

void check_query(const float * got, const float * query, const float * keys, const float * values, int length) {
    auto weights = scores(query, keys, length);
    const double total = normalize(weights);
    for (int d = 0; d < kHeadDim; ++d) {
        const double expected = weighted_value(weights, total, values, d);
        require(std::isfinite(got[d]) && std::abs(got[d] - expected) < kTolerance,
                "flow attention differs from the F32 scalar reference");
    }
}

void check_output(const std::vector<float> & got, const std::vector<float> & query,
                  const std::vector<float> & keys, const std::vector<float> & values, int length) {
    for (int batch = 0; batch < kBatches; ++batch) {
        for (int head = 0; head < kHeads; ++head) {
            for (int token : {0, length / 2, length - 1}) {
                const size_t out_offset = ((size_t(batch) * length + token) * kHeads + head) * kHeadDim;
                check_query(got.data() + out_offset, query.data() + input_offset(length, batch, head, token),
                    keys.data() + input_offset(length, batch, head, 0),
                    values.data() + input_offset(length, batch, head, 0), length);
            }
        }
    }
}

bool supports_half_attention(ggml_backend_t backend) {
    attention_graph owner;
    require(owner.ctx != nullptr, "cannot create capability context");
    auto q = ggml_new_tensor_4d(owner.ctx, GGML_TYPE_F32, kHeadDim, 16, kHeads, kBatches);
    auto k = ggml_new_tensor_4d(owner.ctx, GGML_TYPE_F16, kHeadDim, 16, kHeads, kBatches);
    auto v = ggml_dup_tensor(owner.ctx, k);
    auto output = ggml_flash_attn_ext(owner.ctx, q, k, v, nullptr, 1.0f, 0, 0);
    ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
    return ggml_backend_supports_op(backend, output);
}

void check_attention(ggml_backend_t backend, ggml_type kv_type, int length) {
    attention_graph owner;
    require(owner.ctx != nullptr, "cannot create attention context");
    auto q = ggml_new_tensor_4d(owner.ctx, GGML_TYPE_F32, kHeadDim, length, kHeads, kBatches);
    auto k = ggml_dup_tensor(owner.ctx, q);
    auto v = ggml_dup_tensor(owner.ctx, q);
    auto output = ggml_flash_attn_ext(owner.ctx, q, ggml_cast(owner.ctx, k, kv_type),
        ggml_cast(owner.ctx, v, kv_type), nullptr, 1.0f / std::sqrt(float(kHeadDim)), 0, 0);
    ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
    auto graph = ggml_new_graph(owner.ctx);
    ggml_build_forward_expand(graph, output);
    owner.buffer = ggml_backend_alloc_ctx_tensors(owner.ctx, backend);
    require(owner.buffer != nullptr, "cannot allocate attention tensors");
    auto queries = make_values(ggml_nelements(q), kQueryFrequency);
    auto keys = make_values(ggml_nelements(k), kKeyFrequency);
    auto values = make_values(ggml_nelements(v), kValueFrequency);
    ggml_backend_tensor_set(q, queries.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, keys.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, values.data(), 0, ggml_nbytes(v));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "attention compute failed");
    std::vector<float> got(ggml_nelements(output));
    ggml_backend_tensor_get(output, got.data(), 0, ggml_nbytes(output));
    check_output(got, queries, keys, values, length);
}

void check_lengths(ggml_backend_t backend, ggml_type kv_type) {
    for (int length : {1, 31, 65, 257}) check_attention(backend, kv_type, length);
}

void check_backend(ggml_backend_t backend) {
    dit_hp hp;
    ggml_type kv_type;
    const bool enabled = cosyvoice_dit_fa_config(backend, hp, kv_type);
    const bool prefers_half = tts_cpp::detail::backend_is_vulkan(backend) || tts_cpp::detail::backend_is_metal(backend);
    const auto expected = prefers_half && supports_half_attention(backend) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    require(kv_type == expected, "unexpected flow attention operand type");
    if (enabled) check_lengths(backend, kv_type);
    if (tts_cpp::detail::backend_is_vulkan(backend)) {
        hp.dim_head = kUnsupportedHeadDim;
        require(!cosyvoice_dit_fa_config(backend, hp, kv_type), "unsupported dimensions must use regular attention");
        require(kv_type == GGML_TYPE_F32, "unsupported F16 attention must try F32 fallback");
    }
}

}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name(argc > 1 ? argv[1] : "CPU", nullptr);
    if (!backend) return 1;
    int result = 0;
    try {
        check_backend(backend);
        fprintf(stderr, "PASS: CosyVoice flow attention\n");
    } catch (const std::exception & error) {
        fprintf(stderr, "FAIL: %s\n", error.what());
        result = 1;
    }
    ggml_backend_free(backend);
    return result;
}
