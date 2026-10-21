// parler-fit-params CLI: project whether the Parler GGUF fits the device
// memory available right now, without reading weight data. Thin front-end
// over tts_cpp::parler::fit_params (include/tts-cpp/parler/fit.h); lives in
// the library so hosts can link `parler_fit_cli_main` directly. Exit code ==
// fit status: 0 fits, 1 does not fit, 2 error.

#include "tts-cpp/parler/fit.h"

#include "fit_util.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

using tts_cpp::fitutil::json_escape;
using tts_cpp::fitutil::margin_mib_to_bytes;
using tts_cpp::fitutil::parse_i32;
using tts_cpp::fitutil::parse_u64;

void print_usage(const char * argv0) {
    std::printf(
        "usage: %s --model PARLER.gguf [options]\n"
        "\n"
        "Projects the Parler model's memory needs against the free device memory,\n"
        "reading only GGUF metadata (no weights are loaded, nothing runs).\n"
        "Exit code: 0 = fits, 1 = does not fit, 2 = error.\n"
        "\n"
        "options:\n"
        "  --model PATH             Parler GGUF (required)\n"
        "  --description-tokens N   workload: voice-description length in T5 tokens\n"
        "                           (default 50; sizes the cross-K/V and T5 graphs)\n"
        "  --prompt-tokens N        workload: transcript prompt length in tokens\n"
        "                           (default 128)\n"
        "  --max-frames N           workload: generation cap in delayed decoder\n"
        "                           steps, as EngineOptions::max_frames (default 0 =\n"
        "                           the GGUF's max_length, ~30 s)\n"
        "  --n-gpu-layers N         request the validated GPU backend when > 0\n"
        "                           (default 0 = CPU), with the same runtime\n"
        "                           fallbacks a real load applies\n"
        "  --backend NAME           auto | cpu | opencl | hexagon | exact ggml device\n"
        "                           name, as EngineOptions::backend (no fallback)\n"
        "  --margin-mib MIB         free-memory headroom to require (default 256)\n"
        "  --coreml-decoder         account for the optional Core ML decoder (cannot project)\n"
        "  --backends-dir DIR       directory scanned for dynamically-loaded ggml backends\n"
        "  --json                   emit the projection as JSON on stdout\n"
        "  --help                   this text\n",
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

extern "C" int parler_fit_cli_main(int argc, char ** argv) {
    tts_cpp::parler::FitOptions opts;
    bool json = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "--model" && i + 1 < argc) {
            opts.model_gguf_path = argv[++i];
        } else if (a == "--description-tokens" && i + 1 < argc) {
            // Strict like every numeric flag here: atoi would coerce a typo
            // to 0 and silently project a different workload.
            if (!parse_i32(argv[++i], opts.description_tokens) ||
                opts.description_tokens <= 0) {
                std::fprintf(stderr, "--description-tokens: '%s' is not a positive integer\n",
                             argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
        } else if (a == "--prompt-tokens" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.prompt_tokens) || opts.prompt_tokens <= 0) {
                std::fprintf(stderr, "--prompt-tokens: '%s' is not a positive integer\n",
                             argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
        } else if (a == "--max-frames" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.max_frames) || opts.max_frames < 0) {
                std::fprintf(stderr, "--max-frames: '%s' is not a non-negative integer\n",
                             argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
        } else if (a == "--n-gpu-layers" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.n_gpu_layers)) {
                std::fprintf(stderr, "--n-gpu-layers: '%s' is not an integer\n", argv[i]);
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
        } else if (a == "--backend" && i + 1 < argc) {
            opts.backend = argv[++i];
        } else if (a == "--backends-dir" && i + 1 < argc) {
            opts.backends_dir = argv[++i];
        } else if (a == "--coreml-decoder") {
            opts.coreml_decoder = true;
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

    const tts_cpp::FitResult r = tts_cpp::parler::fit_params(opts);

    if (json) {
        print_json(r, opts.margin_bytes);
    } else if (r.status == tts_cpp::FitStatus::Error) {
        std::fprintf(stderr, "parler-fit-params: error: %s\n", r.reason.c_str());
    } else {
        std::printf("%s", r.report.c_str());
    }

    return (int) r.status;
}
