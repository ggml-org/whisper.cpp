#pragma once
#include <cstdint>
#include <string>

namespace tts_cpp::pocket::detail {
// Reviewed qvac-ext-ggml speech 06f1b668, ggml_cl_init(): uncapped Adreno
// initialization pools. Device allocation limits/build flags can only lower
// these reservations. Re-audit when the backend's F32 dispatch/pools change.
constexpr uint64_t opencl_init_quant_bytes = 311164928;
constexpr uint64_t opencl_init_scales_bytes = 38895616;
constexpr uint64_t opencl_init_activation_bytes = 45088768;
constexpr uint64_t opencl_init_bytes = opencl_init_quant_bytes + opencl_init_scales_bytes + opencl_init_activation_bytes;
// ggml-opencl retains these four pools independently across graphs. Graph
// submissions share them; in-flight growth can retain older generations.
struct OpenCLScratch {
    uint64_t weights = 0, activations = 0, src0 = 0, src1 = 0, growth = 0;
};
// Internal metadata-only measurements. Compute includes CPU work-plan scratch;
// the CPU backend retains its largest work buffer across graph executions.
struct MemoryMeasure {
    uint64_t weights = 0, state = 0, compute = 0;
    uint64_t metadata = 0, graph_metadata = 0, load_staging = 0;
    uint64_t host_compute = 0; // graph-input upload staging, outside device arenas
    OpenCLScratch opencl_scratch;
    std::string source_hash;
    std::string backend_name;
    uint64_t device_free = 0, device_total = 0;
    bool shares_host_memory = true;
};
}
