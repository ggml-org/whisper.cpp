#include "supertonic_pace.h"

#include "tts-cpp/voice_controls.h"

#include <stdexcept>

namespace tts_cpp::supertonic::detail {

namespace {

namespace ctl = ::tts_cpp::controls;

constexpr ctl::EngineId k_engine = ctl::EngineId::Supertonic;
constexpr float         k_no_pace_factor = 1.0f;

}  // namespace

float resolve_pace_factor(float speed, const std::string & pace) {
    if (pace.empty()) return k_no_pace_factor;
    if (speed > 0.0f) {
        throw std::invalid_argument(
            "Supertonic Engine: set either speed (exact multiplier) or pace "
            "(slow|moderate|fast), not both");
    }
    return ctl::pace_rate_factor(ctl::canon_pace(k_engine, pace));
}

float apply_pace_factor(float speed, float pace_factor, float default_speed) {
    return speed > 0.0f ? speed : default_speed * pace_factor;
}

}  // namespace tts_cpp::supertonic::detail
