// cosyvoice-fit-params CLI: project whether the CosyVoice3 GGUF set fits the
// device memory available right now, without reading weight data. Thin
// front-end over tts_cpp::cosyvoice::fit_params (include/tts-cpp/cosyvoice/
// fit.h); lives in the library so hosts can link `cosyvoice_fit_cli_main`
// directly. Exit code == fit status: 0 fits, 1 does not fit, 2 error.

#include "tts-cpp/cosyvoice/fit.h"

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
        "usage: %s --llm LLM.gguf --flow FLOW.gguf --hift HIFT.gguf --voice VOICE.gguf [options]\n"
        "\n"
        "Projects the CosyVoice3 model set's memory needs against the free device\n"
        "memory, reading only GGUF metadata (no weights are loaded, nothing runs).\n"
        "The pipeline is staged (llm/flow/hift load one at a time); the verdict\n"
        "follows the peak phase. Exit code: 0 = fits, 1 = does not fit, 2 = error.\n"
        "\n"
        "options:\n"
        "  --llm PATH            CosyVoice3 LM GGUF (required)\n"
        "  --flow PATH           DiT flow GGUF (required)\n"
        "  --hift PATH           CausalHiFT vocoder GGUF (required)\n"
        "  --voice PATH          baked-voice GGUF (required)\n"
        "  --text-tokens N       workload: total LM text ids -- template +\n"
        "                        transcript + text (default 30)\n"
        "  --speech-tokens N     workload: generated speech tokens, 25/s of audio\n"
        "                        (default 0 = the runtime cap 50*(text_tokens+1);\n"
        "                        the KV cache is always priced at that cap)\n"
        "  --n-gpu-layers N      request the validated GPU backend when > 0\n"
        "                        (default 0 = CPU), with the same runtime\n"
        "                        fallbacks a real load applies\n"
        "  --vulkan-device N     Vulkan adapter index (default 0)\n"
        "  --backend NAME        auto | cpu | opencl | webgpu | hexagon | exact ggml device\n"
        "                        name, as EngineOptions::backend (no fallback)\n"
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

extern "C" int cosyvoice_fit_cli_main(int argc, char ** argv) {
    tts_cpp::cosyvoice::FitOptions opts;
    bool json = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "--llm" && i + 1 < argc) {
            opts.llm_gguf_path = argv[++i];
        } else if (a == "--flow" && i + 1 < argc) {
            opts.flow_gguf_path = argv[++i];
        } else if (a == "--hift" && i + 1 < argc) {
            opts.hift_gguf_path = argv[++i];
        } else if (a == "--voice" && i + 1 < argc) {
            opts.voice_gguf_path = argv[++i];
        } else if (a == "--text-tokens" && i + 1 < argc) {
            // Strict like every numeric flag here: atoi would coerce a typo
            // to 0 and silently project a different workload.
            if (!parse_i32(argv[++i], opts.text_tokens) || opts.text_tokens <= 0) {
                std::fprintf(stderr, "--text-tokens: '%s' is not a positive integer\n",
                             argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
        } else if (a == "--speech-tokens" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.speech_tokens) || opts.speech_tokens < 0) {
                std::fprintf(stderr, "--speech-tokens: '%s' is not a non-negative integer\n",
                             argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
        } else if (a == "--n-gpu-layers" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.n_gpu_layers)) {
                std::fprintf(stderr, "--n-gpu-layers: '%s' is not an integer\n", argv[i]);
                return (int) tts_cpp::FitStatus::Error;
            }
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

    if (opts.llm_gguf_path.empty() || opts.flow_gguf_path.empty() ||
        opts.hift_gguf_path.empty() || opts.voice_gguf_path.empty()) {
        std::fprintf(stderr, "--llm, --flow, --hift and --voice are required\n\n");
        print_usage(argv[0]);
        return (int) tts_cpp::FitStatus::Error;
    }

    const tts_cpp::FitResult r = tts_cpp::cosyvoice::fit_params(opts);

    if (json) {
        print_json(r, opts.margin_bytes);
    } else if (r.status == tts_cpp::FitStatus::Error) {
        std::fprintf(stderr, "cosyvoice-fit-params: error: %s\n", r.reason.c_str());
    } else {
        std::printf("%s", r.report.c_str());
    }

    return (int) r.status;
}
