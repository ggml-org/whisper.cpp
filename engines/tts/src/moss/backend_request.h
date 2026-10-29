#pragma once

#include "tts-cpp/moss/engine.h"
#include "tts-cpp/moss/sound_effect.h"
#include "tts-cpp/moss/speech.h"

#include "ggml-backend.h"

#include <string>

namespace tts_cpp::moss::detail {

struct BackendRequest {
    std::string name;
    bool use_gpu = false;
};

enum class TablePlacement { Automatic, Device, Host };

enum class AttentionLayout { Automatic, Unfused, Fused };

inline constexpr const char * HEXAGON_TTS_WEIGHT_TIERS = "q8_0, q4_0, f16 or f32";
inline constexpr const char * HEXAGON_SFX_WEIGHT_TIERS = "q8_0, q4_0, f16 or f32";
inline constexpr const char * HEXAGON_SPEECH_WEIGHT_TIERS = "q8_0, q4_0, f16 or f32";
inline constexpr const char * HEXAGON_SPEECH_TOKENIZER_WEIGHT_TIERS = "q8_0, f16 or f32";

BackendRequest backend_request(const EngineOptions & options);

BackendRequest backend_request(const SoundEffectOptions & options);

BackendRequest backend_request(const SpeechOptions & options);

ggml_backend_t init_backend(const BackendRequest & request, const char * owner);

bool places_tables_on_host(TablePlacement placement, ggml_backend_t backend);

bool fuses_attention(AttentionLayout layout, ggml_backend_t backend);

bool hexagon_runs_weight_type(ggml_type type);

const ggml_tensor * first_tensor_hexagon_cannot_run(ggml_context * weights);

std::string hexagon_weight_refusal(const ggml_tensor * tensor, const char * supported_tiers);

} // namespace tts_cpp::moss::detail
