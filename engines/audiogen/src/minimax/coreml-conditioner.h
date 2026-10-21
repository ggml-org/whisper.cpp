#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct MM3CoreMLConditioner;

// Explicitly requested Core ML projection. CPU+ANE; the remaining stages use
// the configured ggml backend. Model and IO allocations remain resident.
MM3CoreMLConditioner * mm3_coreml_conditioner_load(const std::string & path,
                                                 int64_t hidden, int64_t output,
                                                 std::string * error);
void mm3_coreml_conditioner_free(MM3CoreMLConditioner * context);

// Frame-major input [frames, hidden], output [frames, output]. Fixed Core ML
// windows use both convolution halos, so arbitrary frame counts are supported.
bool mm3_coreml_conditioner_project(MM3CoreMLConditioner * context, const float * mixed,
                                   int64_t frames, std::vector<float> & projected,
                                   std::string * error);
