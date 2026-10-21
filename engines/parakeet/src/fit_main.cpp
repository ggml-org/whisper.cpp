// parakeet-fit-params CLI: project whether a Parakeet GGUF fits the device
// memory available right now, without reading weight data. Thin front-end
// over parakeet::fit_params (include/parakeet/fit.h); lives in the library so
// hosts can link `parakeet_fit_cli_main` directly (same shape as
// parakeet_cli_main). Exit code == fit status: 0 fits, 1 does not fit,
// 2 error.

#include "parakeet/cli.h"
#include "parakeet/fit.h"

#include "fit_util.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

using parakeet::fitutil::json_escape;
using parakeet::fitutil::margin_mib_to_bytes;
using parakeet::fitutil::parse_f32_positive;
using parakeet::fitutil::parse_i32;
using parakeet::fitutil::parse_u64;

void print_usage(const char * argv0) {
    std::printf(
        "usage: %s --model MODEL.gguf [options]\n"
        "\n"
        "Projects the model's memory needs against the free device memory,\n"
        "reading only GGUF metadata (no weights are loaded, nothing runs).\n"
        "Exit code: 0 = fits, 1 = does not fit, 2 = error.\n"
        "\n"
        "options:\n"
        "  --model PATH            Parakeet GGUF to project (required)\n"
        "  --audio-seconds S       workload: longest single transcribe input\n"
        "                          (default 300; device memory saturates at the\n"
        "                          long-form window for transcription models,\n"
        "                          but keeps growing for Sortformer diarization)\n"
        "  --n-gpu-layers N        request the GPU backend when > 0 (default 0 = CPU),\n"
        "                          with the same runtime fallbacks a real load applies\n"
        "  --threads N             CPU thread count used for backend resolution\n"
        "  --backend NAME          auto|cpu|opencl|hexagon or an exact ggml device name\n"
        "                          (default auto follows --n-gpu-layers; an unavailable\n"
        "                          backend is an error, as in a real load)\n"
        "  --margin-mib MIB        free-memory headroom to require (default 256)\n"
        "  --window-frames N       EngineOptions::long_form_window_frames (default 0 = auto)\n"
        "  --context-frames N      EngineOptions::long_form_context_frames (default 0 = auto)\n"
        "  --nemotron-chunk-ms N   Nemotron only: streaming operating point\n"
        "                          (StreamingOptions::chunk_ms) whose live session the\n"
        "                          projection must also fit; must be one of the GGUF's\n"
        "                          allowed values (80/160/320/560/1120 on the shipped\n"
        "                          checkpoint). Default 0 projects the largest allowed\n"
        "                          operating point. Ignored for other model types\n"
        "  --backends-dir DIR      directory scanned for dynamically-loaded ggml backends\n"
        "  --json                  emit the projection as JSON on stdout\n"
        "  --verbose               loader diagnostics on stderr\n"
        "  --help                  this text\n",
        argv0);
}

void print_json(const parakeet::FitResult & r, uint64_t margin_bytes) {
    auto b = [](bool v) { return v ? "true" : "false"; };
    std::printf("{\n");
    std::printf("  \"status\": %d,\n", (int) r.status);
    std::printf("  \"statusName\": \"%s\",\n", parakeet::fit_status_name(r.status));
    std::printf("  \"fits\": %s,\n", b(r.fits));
    std::printf("  \"reason\": \"%s\",\n", json_escape(r.reason).c_str());
    std::printf("  \"modelType\": \"%s\",\n", json_escape(r.model_type).c_str());
    std::printf("  \"modelVariant\": \"%s\",\n", json_escape(r.model_variant).c_str());
    std::printf("  \"deviceName\": \"%s\",\n", json_escape(r.device_name).c_str());
    std::printf("  \"deviceIsCpu\": %s,\n", b(r.device_is_cpu));
    std::printf("  \"deviceSharesHostMemory\": %s,\n", b(r.device_shares_host_memory));
    std::printf("  \"deviceFreeBytes\": %" PRIu64 ",\n", r.device_free_bytes);
    std::printf("  \"deviceTotalBytes\": %" PRIu64 ",\n", r.device_total_bytes);
    std::printf("  \"weightsBytes\": %" PRIu64 ",\n", r.device.weights_bytes);
    std::printf("  \"encoderComputeBytes\": %" PRIu64 ",\n", r.device.encoder_compute_bytes);
    std::printf("  \"decoderStateBytes\": %" PRIu64 ",\n", r.device.decoder_state_bytes);
    std::printf("  \"decoderComputeBytes\": %" PRIu64 ",\n", r.device.decoder_compute_bytes);
    std::printf("  \"deviceProjectedBytes\": %" PRIu64 ",\n", r.device.total_bytes);
    std::printf("  \"hostBytes\": %" PRIu64 ",\n", r.host_bytes);
    std::printf("  \"coremlBytes\": %" PRIu64 ",\n", r.coreml_bytes);
    std::printf("  \"marginBytes\": %" PRIu64 "\n", margin_bytes);
    std::printf("}\n");
}

}  // namespace

extern "C" int parakeet_fit_cli_main(int argc, char ** argv) {
    parakeet::FitOptions opts;
    bool json = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "--model" && i + 1 < argc) {
            opts.model_gguf_path = argv[++i];
        } else if (a == "--audio-seconds" && i + 1 < argc) {
            if (!parse_f32_positive(argv[++i], opts.audio_seconds)) {
                std::fprintf(stderr, "--audio-seconds: '%s' is not a positive number\n", argv[i]);
                return (int) parakeet::FitStatus::Error;
            }
        } else if (a == "--n-gpu-layers" && i + 1 < argc) {
            // Strict like every numeric flag here: atoi would coerce a typo
            // ('1x') to 0 and silently project the CPU backend instead.
            if (!parse_i32(argv[++i], opts.n_gpu_layers)) {
                std::fprintf(stderr, "--n-gpu-layers: '%s' is not an integer\n", argv[i]);
                return (int) parakeet::FitStatus::Error;
            }
        } else if (a == "--threads" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.n_threads)) {
                std::fprintf(stderr, "--threads: '%s' is not an integer\n", argv[i]);
                return (int) parakeet::FitStatus::Error;
            }
        } else if (a == "--margin-mib" && i + 1 < argc) {
            uint64_t mib = 0;
            if (!parse_u64(argv[++i], mib)) {
                std::fprintf(stderr, "--margin-mib: '%s' is not a non-negative integer\n", argv[i]);
                return (int) parakeet::FitStatus::Error;
            }
            opts.margin_bytes = margin_mib_to_bytes(mib);
        } else if (a == "--window-frames" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.long_form_window_frames)) {
                std::fprintf(stderr, "--window-frames: '%s' is not an integer\n", argv[i]);
                return (int) parakeet::FitStatus::Error;
            }
        } else if (a == "--context-frames" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.long_form_context_frames)) {
                std::fprintf(stderr, "--context-frames: '%s' is not an integer\n", argv[i]);
                return (int) parakeet::FitStatus::Error;
            }
        } else if (a == "--nemotron-chunk-ms" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.nemotron_chunk_ms)) {
                std::fprintf(stderr, "--nemotron-chunk-ms: '%s' is not an integer\n", argv[i]);
                return (int) parakeet::FitStatus::Error;
            }
        } else if (a == "--backend" && i + 1 < argc) {
            opts.backend = argv[++i];
        } else if (a == "--backends-dir" && i + 1 < argc) {
            opts.backends_dir = argv[++i];
        } else if (a == "--json") {
            json = true;
        } else if (a == "--verbose" || a == "-v") {
            opts.verbose = true;
        } else {
            std::fprintf(stderr, "unknown or incomplete argument: %s\n\n", a.c_str());
            print_usage(argv[0]);
            return (int) parakeet::FitStatus::Error;
        }
    }

    if (opts.model_gguf_path.empty()) {
        std::fprintf(stderr, "--model is required\n\n");
        print_usage(argv[0]);
        return (int) parakeet::FitStatus::Error;
    }

    const parakeet::FitResult r = parakeet::fit_params(opts);

    if (json) {
        print_json(r, opts.margin_bytes);
    } else if (r.status == parakeet::FitStatus::Error) {
        std::fprintf(stderr, "parakeet-fit-params: error: %s\n", r.reason.c_str());
    } else {
        std::printf("%s", r.report.c_str());
    }

    return (int) r.status;
}
