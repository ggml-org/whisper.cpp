// acestep-fit-params CLI: project whether the ACE-Step stage GGUFs + a
// generation workload fit the memory available right now, without reading
// weight data. Thin front-end over tts_cpp::acestep::fit_params
// (include/audiogen-cpp/acestep/fit.h); lives in the library so hosts can link
// `acestep_fit_cli_main` directly (the thin executable is fit_cli_main.cpp).
// Exit code == fit status: 0 fits, 1 does not fit, 2 error.

#include "audiogen-cpp/acestep/fit.h"

#include "fit_util.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using tts_cpp::acestep::fitutil::json_escape;
using tts_cpp::acestep::fitutil::margin_mib_to_bytes;
using tts_cpp::acestep::fitutil::parse_f32_nonnegative;
using tts_cpp::acestep::fitutil::parse_f32_positive;
using tts_cpp::acestep::fitutil::parse_i32;
using tts_cpp::acestep::fitutil::parse_u64;

void print_usage(const char * argv0) {
    std::printf(
        "usage: %s --models-dir DIR | --text-enc/--lm/--dit/--vae PATHS [options]\n"
        "\n"
        "Projects the ACE-Step pipeline's memory needs against the free device\n"
        "memory, reading only GGUF metadata (no weights load, nothing runs).\n"
        "Exit code: 0 = fits, 1 = does not fit, 2 = error.\n"
        "\n"
        "models:\n"
        "  --models-dir DIR        directory with the four stage GGUFs (same\n"
        "                          filename classification as the engine)\n"
        "  --text-enc PATH         Qwen3-Embedding GGUF (overrides the scan)\n"
        "  --lm PATH               acestep LM GGUF\n"
        "  --dit PATH              acestep DiT GGUF\n"
        "  --vae PATH              acestep VAE GGUF\n"
        "\n"
        "workload:\n"
        "  --duration S            longest generate() to accommodate, seconds\n"
        "                          (default 60; the DiT graph grows ~quadratically)\n"
        "  --text-tokens N         prompt tokens fed to the text encoder (default 160)\n"
        "  --lyric-tokens N        lyric tokens (default 512; quadratic in the\n"
        "                          cond encoder's sliding-window attention)\n"
        "  --lm-prompt-tokens N    LM prefill length (default: text+lyric+64)\n"
        "  --lm-max-new-tokens N   LM code budget (default: duration*5 + 100)\n"
        "  --lm-cfg SCALE          LM CFG scale (default 2.0; >1 projects the\n"
        "                          2-stream CFG decode)\n"
        "  --guidance SCALE        DiT guidance (default 0 = auto: turbo 1.0,\n"
        "                          base/sft 7.0; >1 adds the APG host buffers)\n"
        "  --with-source-audio     project a cover/lego/reference request\n"
        "                          (adds the VAE-encoder phase)\n"
        "  --keep-stages 0|1       residency to project (default: mirror the\n"
        "                          engine's ACESTEP_KEEP_STAGES environment)\n"
        "\n"
        "options:\n"
        "  --n-gpu-layers N        request the GPU stack when > 0 (default 0 = CPU),\n"
        "                          with the same fallbacks a real load applies\n"
        "  --backend NAME          auto (default), cpu, opencl, hexagon, or a ggml device\n"
        "                          name; explicit requests have no fallback\n"
        "  --lm-backend NAME       put the LM on this device (same names; default auto)\n"
        "  --threads N             CPU thread count used for backend resolution\n"
        "  --margin-mib MIB        free-memory headroom to require (default 256)\n"
        "  --backends-dir DIR      directory scanned for dynamically-loaded ggml backends\n"
        "  --json                  emit the projection as JSON on stdout\n"
        "  --verbose               loader diagnostics on stderr\n"
        "  --help                  this text\n",
        argv0);
}

const char * json_bool(bool v) {
    return v ? "true" : "false";
}

void print_extra_devices_json(const std::vector<tts_cpp::acestep::FitDevicePool> & extras) {
    std::printf("  \"extraDevices\": [");
    for (size_t i = 0; i < extras.size(); ++i) {
        const tts_cpp::acestep::FitDevicePool & d = extras[i];
        std::printf("%s\n    {\"name\": \"%s\", \"sharesHostMemory\": %s, \"freeBytes\": %" PRIu64
                    ", \"totalBytes\": %" PRIu64 ", \"peakBytes\": %" PRIu64 "}",
                    i ? "," : "", json_escape(d.name).c_str(), json_bool(d.shares_host_memory), d.free_bytes,
                    d.total_bytes, d.peak_bytes);
    }
    std::printf("%s],\n", extras.empty() ? "" : "\n  ");
}

void print_json(const tts_cpp::acestep::FitResult & r, uint64_t margin_bytes) {
    auto b = [](bool v) { return v ? "true" : "false"; };
    std::printf("{\n");
    std::printf("  \"status\": %d,\n", (int) r.status);
    std::printf("  \"statusName\": \"%s\",\n", tts_cpp::acestep::fit_status_name(r.status));
    std::printf("  \"fits\": %s,\n", b(r.fits));
    std::printf("  \"reason\": \"%s\",\n", json_escape(r.reason).c_str());
    std::printf("  \"modelName\": \"%s\",\n", json_escape(r.model_name).c_str());
    std::printf("  \"isTurbo\": %s,\n", b(r.is_turbo));
    std::printf("  \"deviceName\": \"%s\",\n", json_escape(r.device_name).c_str());
    std::printf("  \"deviceIsCpu\": %s,\n", b(r.device_is_cpu));
    std::printf("  \"deviceSharesHostMemory\": %s,\n", b(r.device_shares_host_memory));
    std::printf("  \"deviceFreeBytes\": %" PRIu64 ",\n", r.device_free_bytes);
    std::printf("  \"deviceTotalBytes\": %" PRIu64 ",\n", r.device_total_bytes);
    std::printf("  \"hostFreeBytes\": %" PRIu64 ",\n", r.host_free_bytes);
    std::printf("  \"hostTotalBytes\": %" PRIu64 ",\n", r.host_total_bytes);
    std::printf("  \"stagesResident\": %s,\n", b(r.stages_resident));
    std::printf("  \"stages\": [");
    for (size_t i = 0; i < r.stages.size(); ++i) {
        const tts_cpp::acestep::FitStageProjection & s = r.stages[i];
        std::printf("%s\n    {\"name\": \"%s\", \"deviceName\": \"%s\", \"onGpu\": %s, "
                    "\"weightsBytes\": %" PRIu64 ", \"weightsMmapBytes\": %" PRIu64 ", "
                    "\"stateBytes\": %" PRIu64 ", \"computeBytes\": %" PRIu64 ", "
                    "\"hostBytes\": %" PRIu64 "}",
                    i ? "," : "", json_escape(s.name).c_str(), json_escape(s.device_name).c_str(),
                    b(s.on_gpu), s.weights_bytes, s.weights_mmap_bytes, s.state_bytes,
                    s.compute_bytes, s.host_bytes);
    }
    std::printf("\n  ],\n");
    std::printf("  \"peakDeviceBytes\": %" PRIu64 ",\n", r.peak_device_bytes);
    std::printf("  \"peakHostBytes\": %" PRIu64 ",\n", r.peak_host_bytes);
    print_extra_devices_json(r.extra_devices);
    std::printf("  \"marginBytes\": %" PRIu64 "\n", margin_bytes);
    std::printf("}\n");
}

}  // namespace

extern "C" int acestep_fit_cli_main(int argc, char ** argv) {
    tts_cpp::acestep::FitOptions opts;
    bool json = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        // Strict on every numeric flag: atoi would coerce a typo ('1x') to 0
        // and silently project a different device or workload.
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "--models-dir" && i + 1 < argc) {
            opts.models_dir = argv[++i];
        } else if (a == "--text-enc" && i + 1 < argc) {
            opts.text_enc_model_path = argv[++i];
        } else if (a == "--lm" && i + 1 < argc) {
            opts.lm_model_path = argv[++i];
        } else if (a == "--dit" && i + 1 < argc) {
            opts.dit_model_path = argv[++i];
        } else if (a == "--vae" && i + 1 < argc) {
            opts.vae_model_path = argv[++i];
        } else if (a == "--duration" && i + 1 < argc) {
            if (!parse_f32_positive(argv[++i], opts.duration_seconds)) {
                std::fprintf(stderr, "--duration: '%s' is not a positive number\n", argv[i]);
                return (int) tts_cpp::acestep::FitStatus::Error;
            }
        } else if (a == "--text-tokens" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.text_tokens)) {
                std::fprintf(stderr, "--text-tokens: '%s' is not an integer\n", argv[i]);
                return (int) tts_cpp::acestep::FitStatus::Error;
            }
        } else if (a == "--lyric-tokens" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.lyric_tokens)) {
                std::fprintf(stderr, "--lyric-tokens: '%s' is not an integer\n", argv[i]);
                return (int) tts_cpp::acestep::FitStatus::Error;
            }
        } else if (a == "--lm-prompt-tokens" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.lm_prompt_tokens)) {
                std::fprintf(stderr, "--lm-prompt-tokens: '%s' is not an integer\n", argv[i]);
                return (int) tts_cpp::acestep::FitStatus::Error;
            }
        } else if (a == "--lm-max-new-tokens" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.lm_max_new_tokens)) {
                std::fprintf(stderr, "--lm-max-new-tokens: '%s' is not an integer\n", argv[i]);
                return (int) tts_cpp::acestep::FitStatus::Error;
            }
        } else if (a == "--lm-cfg" && i + 1 < argc) {
            if (!parse_f32_nonnegative(argv[++i], opts.lm_cfg_scale)) {
                std::fprintf(stderr, "--lm-cfg: '%s' is not a non-negative number\n", argv[i]);
                return (int) tts_cpp::acestep::FitStatus::Error;
            }
        } else if (a == "--guidance" && i + 1 < argc) {
            if (!parse_f32_nonnegative(argv[++i], opts.guidance_scale)) {
                std::fprintf(stderr, "--guidance: '%s' is not a non-negative number\n", argv[i]);
                return (int) tts_cpp::acestep::FitStatus::Error;
            }
        } else if (a == "--with-source-audio") {
            opts.with_source_audio = true;
        } else if (a == "--keep-stages" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.keep_stages) ||
                (opts.keep_stages != 0 && opts.keep_stages != 1)) {
                std::fprintf(stderr, "--keep-stages: '%s' is not 0 or 1\n", argv[i]);
                return (int) tts_cpp::acestep::FitStatus::Error;
            }
        } else if (a == "--n-gpu-layers" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.n_gpu_layers)) {
                std::fprintf(stderr, "--n-gpu-layers: '%s' is not an integer\n", argv[i]);
                return (int) tts_cpp::acestep::FitStatus::Error;
            }
        } else if (a == "--threads" && i + 1 < argc) {
            if (!parse_i32(argv[++i], opts.n_threads)) {
                std::fprintf(stderr, "--threads: '%s' is not an integer\n", argv[i]);
                return (int) tts_cpp::acestep::FitStatus::Error;
            }
        } else if (a == "--margin-mib" && i + 1 < argc) {
            uint64_t mib = 0;
            if (!parse_u64(argv[++i], mib)) {
                std::fprintf(stderr, "--margin-mib: '%s' is not a non-negative integer\n", argv[i]);
                return (int) tts_cpp::acestep::FitStatus::Error;
            }
            opts.margin_bytes = margin_mib_to_bytes(mib);
        } else if (a == "--backends-dir" && i + 1 < argc) {
            opts.backends_dir = argv[++i];
        } else if (a == "--backend" && i + 1 < argc) {
            opts.backend = argv[++i];
        } else if (a == "--lm-backend" && i + 1 < argc) {
            opts.lm_backend = argv[++i];
        } else if (a == "--json") {
            json = true;
        } else if (a == "--verbose" || a == "-v") {
            opts.verbose = true;
        } else {
            std::fprintf(stderr, "unknown or incomplete argument: %s\n\n", a.c_str());
            print_usage(argv[0]);
            return (int) tts_cpp::acestep::FitStatus::Error;
        }
    }

    if (opts.models_dir.empty() &&
        (opts.text_enc_model_path.empty() || opts.lm_model_path.empty() ||
         opts.dit_model_path.empty() || opts.vae_model_path.empty())) {
        std::fprintf(stderr, "--models-dir (or all four stage paths) is required\n\n");
        print_usage(argv[0]);
        return (int) tts_cpp::acestep::FitStatus::Error;
    }

    const tts_cpp::acestep::FitResult r = tts_cpp::acestep::fit_params(opts);

    if (json) {
        print_json(r, opts.margin_bytes);
    } else if (r.status == tts_cpp::acestep::FitStatus::Error) {
        std::fprintf(stderr, "acestep-fit-params: error: %s\n", r.reason.c_str());
        if (!r.report.empty()) std::fprintf(stderr, "%s\n", r.report.c_str());
    } else {
        std::printf("%s", r.report.c_str());
    }

    return (int) r.status;
}
