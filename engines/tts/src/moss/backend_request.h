#pragma once

#include "tts-cpp/moss/engine.h"

#include "ggml-backend.h"

#include <string>

namespace tts_cpp::moss::detail {

struct BackendRequest {
    std::string name;
    bool use_gpu = false;
};

BackendRequest backend_request(const EngineOptions & options);

ggml_backend_t init_backend(const BackendRequest & request, const char * owner);

bool hexagon_runs_weight_type(ggml_type type);

const ggml_tensor * first_tensor_hexagon_cannot_run(ggml_context * weights);

std::string hexagon_weight_refusal(const ggml_tensor * tensor);

} // namespace tts_cpp::moss::detail
