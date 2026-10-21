#include "coreml-conditioner.h"

#if !defined(AUDIOGEN_USE_COREML)
MM3CoreMLConditioner * mm3_coreml_conditioner_load(const std::string &, int64_t, int64_t,
                                                 std::string * error) {
    if (error) *error = "MiniMax Core ML conditioner requires an Apple build with AUDIOGEN_COREML=ON";
    return nullptr;
}

void mm3_coreml_conditioner_free(MM3CoreMLConditioner *) {}

bool mm3_coreml_conditioner_project(MM3CoreMLConditioner *, const float *, int64_t,
                                   std::vector<float> &, std::string * error) {
    if (error) *error = "MiniMax Core ML conditioner is unavailable in this build";
    return false;
}
#endif
