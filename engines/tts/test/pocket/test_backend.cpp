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
        for (const auto * selection : {"cuda", "metal", "vulkan", "no-such-device"}) {
            bool rejected = false;
            try { auto backend = select_backend(true, selection); }
            catch (const std::invalid_argument &) { rejected = true; }
            check(rejected, "unsupported backend accepted");
        }
        bool opencl_present = false;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            const auto device = ggml_backend_dev_get(i);
            const auto registry = ggml_backend_dev_backend_reg(device);
            opencl_present |= tts_cpp::detail::reg_name_is_opencl(ggml_backend_reg_name(registry));
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
        std::cout << "Pocket CPU default, allowlist and unavailable OpenCL checks passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
