// QVAC (see PATCHES.md): whisper-fit-params CLI -- project whether a whisper
// model (+ optional VAD model) fits the device memory available right now,
// without reading weight data. Thin front-end over whisper_fit_params
// (include/whisper.h). Exit code == fit status: 0 fits, 1 does not fit,
// 2 error (the SDK's @qvac/model-fit contract).

#include "whisper.h"
#include "ggml-backend.h"

#include "whisper-fit-util.h"
#include "../whisper-device-selection.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

using whisper_fit_util::json_escape;
using whisper_fit_util::margin_mib_to_bytes;
using whisper_fit_util::parse_f32_positive;
using whisper_fit_util::parse_i32;
using whisper_fit_util::parse_u64;

void print_usage(const char * argv0) {
    printf(
        "usage: %s --model MODEL.bin [options]\n"
        "\n"
        "Projects the model's memory needs against the free device memory,\n"
        "reading only model metadata (no weights are loaded, nothing runs).\n"
        "Exit code: 0 = fits, 1 = does not fit, 2 = error.\n"
        "\n"
        "options:\n"
        "  --model PATH        whisper ggml model to project (required); its weightless\n"
        "                      description (.fit.gguf) projects the same\n"
        "  --vad-model PATH    also project this VAD model or its description\n"
        "                      (whisper_full's VAD path)\n"
        "  --audio-seconds S   workload: longest single whisper_full input (default 300;\n"
        "                      only host-side buffers scale with it -- the device set is\n"
        "                      fixed by the model's 30 s window at state init)\n"
        "  --n-decoders N      worst-case resident decoders (greedy best_of / beam size;\n"
        "                      default 5 = whisper_full_default_params). The temperature\n"
        "                      fallback grows the KV cache (N+2)x the first time it runs\n"
        "  --no-gpu            project for the CPU backend (default follows\n"
        "                      whisper_context_default_params: GPU when available)\n"
        "  --device DEV        GPU/IGPU index or exact name; default prefers HTP0\n"
        "                      when available, otherwise GPU index 0; overrides\n"
        "                      WHISPER_ARG_DEVICE; --gpu-device is an alias\n"
        "  --list-devices      list devices and exit without a model\n"
        "  --no-flash-attn     project with flash_attn disabled\n"
        "  --margin-mib MIB    free-memory headroom to require (default 256)\n"
        "  --backends-dir DIR  directory scanned for dynamically-loaded ggml backends\n"
        "                      (default: ggml's standard search path)\n"
        "  --verify            ALSO load the model for real and print the measured\n"
        "                      breakdown next to the projection (allocates memory!;\n"
        "                      needs the model itself, not its description)\n"
        "  --json              emit the projection as JSON on stdout\n"
        "  --verbose           loader diagnostics on stderr\n"
        "  --help              this text\n",
        argv0);
}

void print_json(const whisper_fit_result & r, uint64_t margin_bytes) {
    auto b = [](bool v) { return v ? "true" : "false"; };
    printf("{\n");
    printf("  \"status\": %d,\n", (int) r.status);
    printf("  \"statusName\": \"%s\",\n",
           r.status == WHISPER_FIT_SUCCESS ? "success" :
           r.status == WHISPER_FIT_FAILURE ? "failure" : "error");
    printf("  \"fits\": %s,\n", b(r.fits));
    printf("  \"reason\": \"%s\",\n", json_escape(r.reason).c_str());
    printf("  \"modelType\": \"%s\",\n", json_escape(r.model_type).c_str());
    printf("  \"deviceName\": \"%s\",\n", json_escape(r.device_name).c_str());
    printf("  \"deviceIsCpu\": %s,\n", b(r.device_is_cpu));
    printf("  \"deviceSharesHostMemory\": %s,\n", b(r.device_shares_host_memory));
    printf("  \"deviceFreeBytes\": %" PRIu64 ",\n", r.device_free_bytes);
    printf("  \"deviceTotalBytes\": %" PRIu64 ",\n", r.device_total_bytes);
    printf("  \"weightsBytes\": %" PRIu64 ",\n", r.device.weights_bytes);
    printf("  \"kvBytes\": %" PRIu64 ",\n", r.device.kv_bytes);
    printf("  \"computeBytes\": %" PRIu64 ",\n", r.device.compute_bytes);
    printf("  \"vadBytes\": %" PRIu64 ",\n", r.device.vad_bytes);
    printf("  \"deviceProjectedBytes\": %" PRIu64 ",\n", r.device.total_bytes);
    printf("  \"hostBytes\": %" PRIu64 ",\n", r.host_bytes);
    printf("  \"marginBytes\": %" PRIu64 "\n", margin_bytes);
    printf("}\n");
}

void log_callback_null(ggml_log_level, const char *, void *) {}

} // namespace

int main(int argc, char ** argv) {
    whisper_fit_options opts = whisper_fit_default_options();

    std::string model_path;
    std::string vad_model_path;
    std::string backends_dir;
    std::string device;
    bool device_explicit = false;
    bool list_devices = false;
    bool json   = false;
    bool verify = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (a == "--vad-model" && i + 1 < argc) {
            vad_model_path = argv[++i];
        } else if (a == "--audio-seconds" && i + 1 < argc) {
            if (!parse_f32_positive(argv[++i], opts.audio_seconds)) {
                fprintf(stderr, "--audio-seconds: '%s' is not a positive number\n", argv[i]);
                return (int) WHISPER_FIT_ERROR;
            }
        } else if (a == "--n-decoders" && i + 1 < argc) {
            // strict like every numeric flag here: atoi would coerce a typo
            // ('5x') to 0 and silently project a different workload
            if (!parse_i32(argv[++i], opts.n_decoders)) {
                fprintf(stderr, "--n-decoders: '%s' is not an integer\n", argv[i]);
                return (int) WHISPER_FIT_ERROR;
            }
        } else if ((a == "--gpu-device" || a == "--device" || a == "-dev") && i + 1 < argc) {
            device = argv[++i];
            device_explicit = true;
        } else if (a == "--list-devices") {
            list_devices = true;
        } else if (a == "--margin-mib" && i + 1 < argc) {
            uint64_t mib = 0;
            if (!parse_u64(argv[++i], mib)) {
                fprintf(stderr, "--margin-mib: '%s' is not a non-negative integer\n", argv[i]);
                return (int) WHISPER_FIT_ERROR;
            }
            opts.margin_bytes = margin_mib_to_bytes(mib);
        } else if (a == "--backends-dir" && i + 1 < argc) {
            backends_dir = argv[++i];
        } else if (a == "--no-gpu") {
            opts.use_gpu = false;
        } else if (a == "--no-flash-attn") {
            opts.flash_attn = false;
        } else if (a == "--verify") {
            verify = true;
        } else if (a == "--json") {
            json = true;
        } else if (a == "--verbose" || a == "-v") {
            opts.verbose = true;
        } else if (a == "--model" || a == "--vad-model" || a == "--audio-seconds" ||
                   a == "--n-decoders" || a == "--gpu-device" || a == "--device" || a == "-dev" || a == "--margin-mib" ||
                   a == "--backends-dir") {
            // a value-taking flag in last position: every such flag only
            // matches its branch above when a value follows it
            fprintf(stderr, "%s requires a value\n\n", a.c_str());
            print_usage(argv[0]);
            return (int) WHISPER_FIT_ERROR;
        } else {
            fprintf(stderr, "unknown argument: %s\n\n", a.c_str());
            print_usage(argv[0]);
            return (int) WHISPER_FIT_ERROR;
        }
    }

    if (model_path.empty() && !list_devices) {
        fprintf(stderr, "--model is required\n\n");
        print_usage(argv[0]);
        return (int) WHISPER_FIT_ERROR;
    }

    if (!opts.verbose) {
        whisper_log_set(log_callback_null, nullptr);
    }

    // same dynamic-backend discovery whisper-cli performs
    if (!backends_dir.empty()) {
        ggml_backend_load_all_from_path(backends_dir.c_str());
    } else {
        ggml_backend_load_all();
    }

    if (list_devices) {
        whisper_device_selection::list(stdout);
        return 0;
    }
    std::string device_error;
    if (!whisper_device_selection::apply(device, device_explicit, opts.use_gpu,
                                          opts.gpu_device, device_error)) {
        fprintf(stderr, "whisper-fit-params: error: %s\n", device_error.c_str());
        return (int) WHISPER_FIT_ERROR;
    }
    const std::string requested_backend = whisper_device_selection::requested_backend(
        device_explicit, opts.use_gpu, opts.gpu_device);

    opts.model_path     = model_path.c_str();
    opts.vad_model_path = vad_model_path.empty() ? nullptr : vad_model_path.c_str();

    whisper_fit_result r;
    whisper_fit_params(&opts, &r);
    // A registered device can still fail initialization. Do not emit a CPU
    // memory verdict for an explicit GPU request, or obscure an earlier error.
    if (r.status != WHISPER_FIT_ERROR &&
        !whisper_device_selection::check_primary_backend(requested_backend, r.device_name, device_error)) {
        r.status = WHISPER_FIT_ERROR;
        r.fits = false;
        std::snprintf(r.reason, sizeof(r.reason), "requested-device-init-failed");
    }

    if (json) {
        print_json(r, opts.margin_bytes);
        if (!device_error.empty()) {
            fprintf(stderr, "whisper-fit-params: error: %s\n", device_error.c_str());
        }
    } else if (r.status == WHISPER_FIT_ERROR) {
        fprintf(stderr, "whisper-fit-params: error: %s\n", device_error.empty() ? r.reason : device_error.c_str());
    } else {
        printf("%s", r.report);
    }

    if (verify && r.status != WHISPER_FIT_ERROR) {
        whisper_fit_breakdown actual;
        // measure the post-init resident set from a REAL load; compare against
        // an n_decoders = 1 projection -- the one point where every line,
        // compute included, is byte-comparable (whisper_fit_actual's contract:
        // at n_decoders > 1 the schedulers still hold their post-init size)
        whisper_fit_options vopts = opts;
        vopts.n_decoders = 1;
        whisper_fit_result pr1;
        whisper_fit_params(&vopts, &pr1);
        if (whisper_fit_actual(&vopts, &actual) != 0) {
            fprintf(stderr, "whisper-fit-params: --verify: real load failed\n");
            return (int) WHISPER_FIT_ERROR;
        }
        printf("verify (projected @ n_decoders=1 vs real load):\n");
        printf("  weights:  %14" PRIu64 " vs %14" PRIu64 "%s\n", pr1.device.weights_bytes, actual.weights_bytes,
               pr1.device.weights_bytes == actual.weights_bytes ? "" : "  MISMATCH");
        printf("  kv cache: %14" PRIu64 " vs %14" PRIu64 "%s\n", pr1.device.kv_bytes, actual.kv_bytes,
               pr1.device.kv_bytes == actual.kv_bytes ? "" : "  MISMATCH");
        printf("  compute:  %14" PRIu64 " vs %14" PRIu64 "%s\n", pr1.device.compute_bytes, actual.compute_bytes,
               pr1.device.compute_bytes == actual.compute_bytes ? "" : "  MISMATCH");
        printf("  vad:      %14" PRIu64 " vs %14" PRIu64 "%s\n", pr1.device.vad_bytes, actual.vad_bytes,
               pr1.device.vad_bytes == actual.vad_bytes ? "" : "  MISMATCH");
        printf("  host ovf: %14" PRIu64 " vs %14" PRIu64 "%s\n", pr1.device.host_overflow_bytes, actual.host_overflow_bytes,
               pr1.device.host_overflow_bytes == actual.host_overflow_bytes ? "" : "  MISMATCH");
    }

    return (int) r.status;
}
