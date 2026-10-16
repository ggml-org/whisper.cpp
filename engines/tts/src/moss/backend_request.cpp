#include "moss/backend_request.h"

#include "backend_selection.h"
#include "backend_util.h"

#include <stdexcept>

namespace tts_cpp::moss::detail {
namespace {

constexpr int GPU_REQUESTED = 1;

} // namespace

BackendRequest backend_request(const EngineOptions & options) {
    return {options.backend, options.use_gpu};
}

BackendRequest backend_request(const SoundEffectOptions & options) {
    return {options.backend, options.use_gpu};
}

bool places_tables_on_host(TablePlacement placement, ggml_backend_t backend) {
    return placement == TablePlacement::Host ||
           (placement == TablePlacement::Automatic && ::tts_cpp::detail::backend_is_hexagon(backend));
}

bool fuses_attention(AttentionLayout layout, ggml_backend_t backend) {
    return layout == AttentionLayout::Fused ||
           (layout == AttentionLayout::Automatic && ::tts_cpp::detail::backend_is_hexagon(backend));
}

ggml_backend_t init_backend(const BackendRequest & request, const char * owner) {
    namespace det = ::tts_cpp::detail;
    det::ensure_backends_loaded();
    if (!det::backend_request_is_auto(request.name)) {
        ggml_backend_t backend = det::init_requested_backend(request.name, false, owner);
        if (!backend) {
            throw std::runtime_error(std::string(owner) + ": no compute backend available for '" +
                                     request.name + "'");
        }
        return backend;
    }
    ggml_backend_t backend = request.use_gpu ? det::init_gpu_backend(GPU_REQUESTED, false, owner) : nullptr;
    if (!backend) {
        backend = det::init_cpu_backend();
    }
    if (!backend) {
        throw std::runtime_error(std::string(owner) + ": no compute backend available");
    }
    return backend;
}

bool hexagon_runs_weight_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0 ||
           type == GGML_TYPE_Q4_0;
}

const ggml_tensor * first_tensor_hexagon_cannot_run(ggml_context * weights) {
    for (auto * t = ggml_get_first_tensor(weights); t; t = ggml_get_next_tensor(weights, t)) {
        if (!hexagon_runs_weight_type(t->type)) {
            return t;
        }
    }
    return nullptr;
}

std::string hexagon_weight_refusal(const ggml_tensor * tensor, const char * supported_tiers) {
    return std::string("Hexagon cannot run ") + ggml_get_name(tensor) + " (" + ggml_type_name(tensor->type) +
           "); use a " + supported_tiers + " GGUF";
}

} // namespace tts_cpp::moss::detail
