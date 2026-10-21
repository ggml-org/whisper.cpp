#include "pocket/backend.h"
#include <iostream>
#include <stdexcept>

using namespace tts_cpp::pocket::detail;
static void check(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }
int main() {
    try {
        for (const auto & selection : {std::string(), std::string("auto"), std::string("cpu"), std::string("CPU")}) {
            auto backend = select_backend(false, selection);
            check(tts_cpp::detail::backend_is_cpu(backend.get()), "CPU default changed");
            check(shares_host_memory(backend.get()), "CPU must share host memory");
        }
        auto explicit_cpu = select_backend(true, "cpu");
        check(tts_cpp::detail::backend_is_cpu(explicit_cpu.get()), "explicit CPU selection ignored");
        for (const auto * selection : {"cuda", "metal", "vulkan", "HTP1", "no-such-device"}) {
            bool rejected = false;
            try { auto backend = select_backend(true, selection); }
            catch (const std::invalid_argument &) { rejected = true; }
            check(rejected, "unsupported backend accepted");
        }
        check(backend_request("HTP0") == "hexagon" && backend_request("HEXAGON") == "hexagon", "Hexagon aliases differ");
        check(backend_request("WebGPU") == "webgpu", "WebGPU request is not case-folded");
        bool opencl_present = false, webgpu_present = false, hexagon_present = false;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            const auto device = ggml_backend_dev_get(i);
            const auto registry = ggml_backend_dev_backend_reg(device);
            opencl_present |= tts_cpp::detail::reg_name_is_opencl(ggml_backend_reg_name(registry));
            webgpu_present |= tts_cpp::detail::reg_name_is_webgpu(ggml_backend_reg_name(registry));
            hexagon_present |= tts_cpp::detail::reg_name_is_hexagon(ggml_backend_reg_name(registry)) &&
                std::string(ggml_backend_dev_name(device)) == "HTP0";
        }
        if (!opencl_present) {
            for (bool gpu_flag : {false, true}) {
                bool rejected = false;
                try { auto backend = select_backend(gpu_flag, "opencl"); }
                catch (const BackendUnavailable & error) {
                    rejected = std::string(error.what()).find("OpenCL") != std::string::npos;
                }
                check(rejected, "missing OpenCL silently fell back to CPU");
            }
        }
        if (webgpu_present) {
            for (bool gpu_flag : {false, true}) {
                auto backend = select_backend(gpu_flag, "webgpu");
                check(tts_cpp::detail::backend_is_webgpu(backend.get()), "explicit WebGPU selected another backend");
            }
        } else {
            for (bool gpu_flag : {false, true}) {
                bool rejected = false;
                try { auto backend = select_backend(gpu_flag, "webgpu"); }
                catch (const BackendUnavailable & error) {
                    rejected = std::string(error.what()).find("WebGPU") != std::string::npos;
                }
                check(rejected, "missing WebGPU silently fell back to CPU");
            }
        }
        if (hexagon_present) {
            for (const auto * selection : {"hexagon", "HTP0"}) {
                auto backend = select_backend(true, selection);
                check(tts_cpp::detail::backend_is_hexagon(backend.get()), "explicit Hexagon selected another backend");
                check(std::string(ggml_backend_name(backend.get())) == "HTP0", "Hexagon alias selected another session");
                check(shares_host_memory(backend.get()), "Hexagon RPC memory must share host RAM");
            }
        } else {
            for (const auto * selection : {"hexagon", "HEXAGON", "HTP0", "htp0"}) {
                for (bool gpu_flag : {false, true}) {
                    bool rejected = false;
                    try { auto backend = select_backend(gpu_flag, selection); }
                    catch (const BackendUnavailable & error) {
                        rejected = std::string(error.what()).find("Hexagon") != std::string::npos;
                    }
                    check(rejected, "missing Hexagon silently fell back to CPU/OpenCL");
                }
            }
        }
        std::cout << "Pocket CPU default, aliases, allowlist and unavailable accelerator checks passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
