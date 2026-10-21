#pragma once

#include "decoder_coreml.h"

#include <cstdio>

namespace tts_cpp { namespace parler { namespace detail {

// Internal constructor helper. The sidecar opener has already validated
// positive geometry with per_state dividing layers. Only row changes at step.
inline std::vector<decoder_coreml_row_update> parler_decoder_update_plan(int layers, int per_state) {
    if (per_state == layers) {
        return {{"new_k", "self_k_00", 2, 0, -1, -1},
                {"new_v", "self_v_00", 2, 0, -1, -1}};
    }
    std::vector<decoder_coreml_row_update> updates;
    for (int l = 0; l < layers; ++l) {
        char suffix[24];
        std::snprintf(suffix, sizeof(suffix), "_%02d", l / per_state);
        for (int v = 0; v < 2; ++v) {
            updates.push_back({v ? "new_v" : "new_k", std::string(v ? "self_v" : "self_k") + suffix,
                               2, 0, l, l % per_state});
        }
    }
    return updates;
}

}}} // namespace tts_cpp::parler::detail
