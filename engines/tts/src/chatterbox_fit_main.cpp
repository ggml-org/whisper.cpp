// chatterbox-fit-params CLI: project whether the T3 + S3Gen GGUF pair fits
// the device memory available right now, without reading weight data.  Thin
// front-end over tts_cpp::chatterbox::fit_params
// (include/tts-cpp/chatterbox/fit.h); lives in the library so hosts can link
// `chatterbox_fit_cli_main` directly.  Exit code == fit status: 0 fits, 1
// does not fit, 2 error.

#include "tts-cpp/chatterbox/fit.h"

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
        "usage: %s --t3 T3.gguf --s3gen S3GEN.gguf [options]\n"
        "\n"
        "Projects the Chatterbox model pair's memory needs against the free\n"
        "device memory, reading only GGUF metadata (no weights are loaded,\n"
        "nothing runs).  The T3 variant (turbo / mtl) is picked from the GGUF\n"
        "exactly as a real load picks it.\n"
        "Exit code: 0 = fits, 1 = does not fit, 2 = error.\n"
        "\n"
        "options:\n"
        "  --t3 PATH               chatterbox T3 GGUF (required)\n"
        "  --s3gen PATH            chatterbox S3Gen GGUF (required)\n"
        "  --text-tokens N         workload: text tokens of the longest segment (default 128)\n"
        "  --n-predict N           workload: speech-token budget, as EngineOptions::\n"
        "                          n_predict (default 1000)\n"
        "  --n-ctx N               EngineOptions::n_ctx (default 0 = the GGUF's value);\n"
        "                          the KV cache is allocated up-front at this length\n"
        "  --kv-cache-type T       EngineOptions::kv_cache_type: f32|f16|q8_0, resolved\n"
        "                          against the backend with the same downgrades a real\n"
        "                          load applies (default f32)\n"
        "  --n-gpu-layers N        request the GPU backend when > 0 (default 0 = CPU),\n"
        "                          with the same runtime fallbacks a real load applies\n"
        "  --backend NAME          auto | cpu | opencl | webgpu | hexagon | exact ggml device name;\n"
        "                          overrides --n-gpu-layers, no fallback\n"
        "  --margin-mib MIB        free-memory headroom to require (default 256)\n"
        "  --backends-dir DIR      directory scanned for dynamically-loaded ggml backends\n"
        "  --json                  emit the projection as JSON on stdout\n"
        "  --help                  this text\n",
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

extern "C" int chatterbox_fit_cli_main(int argc, char ** argv) {
    tts_cpp::chatterbox::FitOptions opts;
    bool json = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "--t3" && i + 1 < argc) {
            opts.t3_gguf_path = argv[++i];
        } else if (a == "--s3gen" && i + 1 < argc) {
            opts.s3gen_gguf_path = argv[++i];
        } else if (a == "--text-tokens" && i + 1 < argc) {
            // Strict like every numeric flag here: atoi would coerce a typo
            // to 0 and silently project a different workload.
            if (!parse_i32(argv[++i], opts.text_tokens) || opts.text_tokens <= 0) {
                std::fprintf(stderr, "--text-tokens: '%s' is not a positive integer\n", argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
        } else if (a == "--n-predict" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.n_predict) || opts.n_predict <= 0) {
                std::fprintf(stderr, "--n-predict: '%s' is not a positive integer\n", argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
        } else if (a == "--n-ctx" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.n_ctx) || opts.n_ctx < 0) {
                std::fprintf(stderr, "--n-ctx: '%s' is not a non-negative integer\n", argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
        } else if (a == "--kv-cache-type" && i + 1 < argc) {
            opts.kv_cache_type = argv[++i];
        } else if (a == "--n-gpu-layers" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.n_gpu_layers)) {
                std::fprintf(stderr, "--n-gpu-layers: '%s' is not an integer\n", argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
        } else if (a == "--margin-mib" && i + 1 < argc) {
            uint64_t mib = 0;
            if (!parse_u64(argv[++i], mib)) {
                std::fprintf(stderr, "--margin-mib: '%s' is not a non-negative integer\n", argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
            opts.margin_bytes = margin_mib_to_bytes(mib);
        } else if (a == "--backend" && i + 1 < argc) {
            opts.backend = argv[++i];
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

    if (opts.t3_gguf_path.empty() || opts.s3gen_gguf_path.empty()) {
        std::fprintf(stderr, "--t3 and --s3gen are required\n\n");
        print_usage(argv[0]);
        return (int) tts_cpp::FitStatus::Error;
    }

    const tts_cpp::FitResult r = tts_cpp::chatterbox::fit_params(opts);

    if (json) {
        print_json(r, opts.margin_bytes);
    } else if (r.status == tts_cpp::FitStatus::Error) {
        std::fprintf(stderr, "chatterbox-fit-params: error: %s\n", r.reason.c_str());
    } else {
        std::printf("%s", r.report.c_str());
    }

    return (int) r.status;
}
