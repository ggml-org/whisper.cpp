// supertonic-fit-params CLI: project whether the Supertonic GGUF fits the
// device memory available right now, without reading weight data. Thin
// front-end over tts_cpp::supertonic::fit_params (include/tts-cpp/supertonic/
// fit.h); lives in the library so hosts can link `supertonic_fit_cli_main`
// directly. Exit code == fit status: 0 fits, 1 does not fit, 2 error.

#include "tts-cpp/supertonic/fit.h"

#include "fit_util.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

using tts_cpp::fitutil::json_escape;
using tts_cpp::fitutil::margin_mib_to_bytes;
using tts_cpp::fitutil::parse_f32_positive;
using tts_cpp::fitutil::parse_i32;
using tts_cpp::fitutil::parse_u64;

void print_usage(const char * argv0) {
    std::printf(
        "usage: %s --model SUPERTONIC.gguf [options]\n"
        "\n"
        "Projects the Supertonic model's memory needs against the free device\n"
        "memory, reading only GGUF metadata (no weights are loaded, nothing runs).\n"
        "Models fused GPU/CPU graph paths. The CPU Accelerate/CBLAS pointwise path\n"
        "exits 2 with reason compute-path-not-supported instead of guessing.\n"
        "Exit code: 0 = fits, 1 = does not fit, 2 = error.\n"
        "\n"
        "options:\n"
        "  --model PATH          Supertonic GGUF (required)\n"
        "  --text-tokens N       workload: text length in Supertonic tokens\n"
        "                        (Unicode code points; default 128)\n"
        "  --audio-seconds S     workload: projected utterance length (default 10;\n"
        "                        the runtime derives it from the duration model)\n"
        "  --steps N             CFM steps (default 0 = the GGUF's default_steps)\n"
        "  --precision P         auto | f32 | f16 | q8_0, as EngineOptions::precision\n"
        "  --f16-weights N       -1 auto / 0 off / 1 on, as EngineOptions::f16_weights\n"
        "  --n-gpu-layers N      request the validated GPU backend when > 0\n"
        "                        (default 0 = CPU; support depends on graph path)\n"
        "  --vulkan-device N     Vulkan adapter index (default 0)\n"
        "  --backend NAME        auto | cpu | opencl | webgpu | hexagon | exact ggml device name,\n"
        "                        as EngineOptions::backend (no fallback)\n"
        "  --margin-mib MIB      free-memory headroom to require (default 256)\n"
        "  --backends-dir DIR    directory scanned for dynamically-loaded ggml backends\n"
        "  --json                emit the projection as JSON on stdout\n"
        "  --help                this text\n",
        argv0);
}

void print_json(const tts_cpp::FitResult & r, uint64_t margin_bytes) {
    auto b = [](bool v) { return v ? "true" : "false"; };
    std::printf("{\n");
    std::printf("  \"status\": %d,\n", (int) r.status);
    std::printf("  \"statusName\": \"%s\",\n", tts_cpp::fit_status_name(r.status));
    std::printf("  \"fits\": %s,\n", b(r.fits));
    std::printf("  \"reason\": \"%s\",\n", json_escape(r.reason).c_str());
    std::printf("  \"modelVariant\": \"%s\",\n", json_escape(r.model_variant).c_str());
    std::printf("  \"deviceName\": \"%s\",\n", json_escape(r.device_name).c_str());
    std::printf("  \"deviceIsCpu\": %s,\n", b(r.device_is_cpu));
    std::printf("  \"deviceSharesHostMemory\": %s,\n", b(r.device_shares_host_memory));
    std::printf("  \"deviceFreeBytes\": %" PRIu64 ",\n", r.device_free_bytes);
    std::printf("  \"deviceTotalBytes\": %" PRIu64 ",\n", r.device_total_bytes);
    std::printf("  \"weightsBytes\": %" PRIu64 ",\n", r.device.weights_bytes);
    std::printf("  \"stateBytes\": %" PRIu64 ",\n", r.device.state_bytes);
    std::printf("  \"lmComputeBytes\": %" PRIu64 ",\n", r.device.lm_compute_bytes);
    std::printf("  \"codecComputeBytes\": %" PRIu64 ",\n", r.device.codec_compute_bytes);
    std::printf("  \"deviceProjectedBytes\": %" PRIu64 ",\n", r.device.total_bytes);
    std::printf("  \"hostBytes\": %" PRIu64 ",\n", r.host_bytes);
    std::printf("  \"marginBytes\": %" PRIu64 "\n", margin_bytes);
    std::printf("}\n");
}

}  // namespace

extern "C" int supertonic_fit_cli_main(int argc, char ** argv) {
    tts_cpp::supertonic::FitOptions opts;
    bool json = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "--model" && i + 1 < argc) {
            opts.model_gguf_path = argv[++i];
        } else if (a == "--text-tokens" && i + 1 < argc) {
            // Strict like every numeric flag here: atoi would coerce a typo
            // to 0 and silently project a different workload.
            if (!parse_i32(argv[++i], opts.text_tokens) || opts.text_tokens <= 0) {
                std::fprintf(stderr, "--text-tokens: '%s' is not a positive integer\n",
                             argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
        } else if (a == "--audio-seconds" && i + 1 < argc) {
            if (!parse_f32_positive(argv[++i], opts.audio_seconds)) {
                std::fprintf(stderr, "--audio-seconds: '%s' is not a positive number\n",
                             argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
        } else if (a == "--steps" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.steps) || opts.steps < 0) {
                std::fprintf(stderr, "--steps: '%s' is not a non-negative integer\n", argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
        } else if (a == "--precision" && i + 1 < argc) {
            opts.precision = argv[++i];
            if (opts.precision != "auto" && opts.precision != "f32" &&
                opts.precision != "f16" && opts.precision != "q8_0") {
                std::fprintf(stderr, "--precision: '%s' is not auto|f32|f16|q8_0\n",
                             opts.precision.c_str());
                return 2;
            }
        } else if (a == "--f16-weights" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.f16_weights) ||
                opts.f16_weights < -1 || opts.f16_weights > 1) {
                std::fprintf(stderr, "--f16-weights: '%s' is not -1|0|1\n", argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
        } else if (a == "--n-gpu-layers" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.n_gpu_layers)) {
                std::fprintf(stderr, "--n-gpu-layers: '%s' is not an integer\n", argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
        } else if (a == "--backend" && i + 1 < argc) {
            opts.backend = argv[++i];
        } else if (a == "--vulkan-device" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.vulkan_device) || opts.vulkan_device < 0) {
                std::fprintf(stderr, "--vulkan-device: '%s' is not a non-negative integer\n",
                             argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
        } else if (a == "--margin-mib" && i + 1 < argc) {
            uint64_t mib = 0;
            if (!parse_u64(argv[++i], mib)) {
                std::fprintf(stderr, "--margin-mib: '%s' is not a non-negative integer\n",
                             argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
            opts.margin_bytes = margin_mib_to_bytes(mib);
        } else if (a == "--backends-dir" && i + 1 < argc) {
            opts.backends_dir = argv[++i];
        } else if (a == "--json") {
            json = true;
        } else {
            std::fprintf(stderr, "unknown or incomplete argument: %s\n\n", a.c_str());
            print_usage(argv[0]);
            return (int) tts_cpp::FitStatus::Error;
        }
    }

    if (opts.model_gguf_path.empty()) {
        std::fprintf(stderr, "--model is required\n\n");
        print_usage(argv[0]);
        return (int) tts_cpp::FitStatus::Error;
    }

    const tts_cpp::FitResult r = tts_cpp::supertonic::fit_params(opts);

    if (json) {
        print_json(r, opts.margin_bytes);
    } else if (r.status == tts_cpp::FitStatus::Error) {
        std::fprintf(stderr, "supertonic-fit-params: error: %s\n", r.reason.c_str());
    } else {
        std::printf("%s", r.report.c_str());
    }

    return (int) r.status;
}
