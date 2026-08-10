#pragma once

// Supertonic speaking-rate resolution, split out from the engine so it carries
// no ggml dependency and can be pinned by a model-free test.

#include <string>

namespace tts_cpp::supertonic::detail {

// Multiplier the canonical pace step contributes, exactly 1.0f when no pace is
// set. Throws std::invalid_argument when both rate controls are set or the step
// is not one Supertonic supports; the engine calls this at construction so a
// bad request fails at load rather than at the first synthesized chunk.
float resolve_pace_factor(float speed, const std::string & pace);

// The rate to synthesize at. `speed` is the exact multiplier and wins when set;
// otherwise the checkpoint's own `default_speed` carries the pace factor, which
// makes "moderate" bit-identical to leaving both unset on any checkpoint.
float apply_pace_factor(float speed, float pace_factor, float default_speed);

}  // namespace tts_cpp::supertonic::detail
