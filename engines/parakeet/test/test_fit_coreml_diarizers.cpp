// fit_params must charge the Core ML diarizer sidecars beside a Nemotron 3
// Diarization or Sortformer v2.1 GGUF: a short offline input loads only the
// smallest bundle, a long Nemotron input reaches every bundle through its
// long-form blocks, a long Sortformer input exceeds them all and runs on
// ggml, the charge decides the verdict on unified memory, and
// PARAKEET_COREML_DIARIZER_DISABLE removes it. parakeet-fit-params --json
// reports the same figure. Exits 77 without sidecars.

#include "parakeet/cli.h"
#include "parakeet/fit.h"

#include "coreml/coreml_diarizers.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

constexpr int kSkip = 77;
constexpr int kGpuLayers = 1;
constexpr float kShortSeconds = 27.3f;
constexpr float kLongSeconds = 273.0f;
constexpr uint64_t kDefaultMargin = 256ull * 1024 * 1024;
constexpr int kAllFrames = std::numeric_limits<int>::max();
constexpr const char * kNemotronType = "nemotron-diarization";
constexpr const char * kReportLine = "core ml sidecars:";
constexpr const char * kJsonKey = "\"coremlBytes\": ";
constexpr const char * kJsonCapture = "fit-coreml-diarizers.json";

bool g_passed = true;

void check(bool condition, const char * label) {
    std::printf("[%s] %s\n", condition ? "ok  " : "FAIL", label);
    g_passed = g_passed && condition;
}

parakeet::FitResult project(const std::string & gguf, float seconds, uint64_t margin) {
    parakeet::FitOptions opts;
    opts.model_gguf_path = gguf;
    opts.audio_seconds = seconds;
    opts.n_gpu_layers = kGpuLayers;
    opts.margin_bytes = margin;
    return parakeet::fit_params(opts);
}

uint64_t charge(const std::vector<parakeet::CoremlDiarizerBundle> & bundles) {
    return parakeet::coreml_diarizer_resident_bytes(bundles, {{1, kAllFrames}});
}

bool reports_sidecars(const parakeet::FitResult & result) {
    return result.report.find(kReportLine) != std::string::npos;
}

uint64_t margin_between(const parakeet::FitResult & result) {
    const uint64_t without_sidecars = result.device.total_bytes + result.host_bytes;
    return result.device_free_bytes - without_sidecars - result.coreml_bytes / 2;
}

std::vector<char *> pointers_to(std::vector<std::string> & args) {
    std::vector<char *> pointers(args.size());
    std::transform(args.begin(), args.end(), pointers.begin(),
                   [](std::string & arg) { return arg.data(); });
    return pointers;
}

std::string captured_cli_json(const std::string & gguf) {
    const std::string seconds = std::to_string(kShortSeconds);
    const std::string layers = std::to_string(kGpuLayers);
    std::vector<std::string> args = {"parakeet-fit-params", "--model", gguf, "--audio-seconds",
                                     seconds, "--n-gpu-layers", layers, "--json"};
    std::vector<char *> argv = pointers_to(args);
    std::fflush(stdout);
    const int saved = dup(STDOUT_FILENO);
    if (std::freopen(kJsonCapture, "w", stdout) != nullptr) {
        parakeet_fit_cli_main(static_cast<int>(argv.size()), argv.data());
        std::fflush(stdout);
    }
    dup2(saved, STDOUT_FILENO);
    close(saved);
    std::ifstream file(kJsonCapture);
    std::stringstream text;
    text << file.rdbuf();
    std::remove(kJsonCapture);
    return text.str();
}

void check_cli_json(const std::string & gguf, uint64_t expected) {
    const std::string json = captured_cli_json(gguf);
    const std::string field = kJsonKey + std::to_string(expected) + ",";
    check(json.find(field) != std::string::npos, "fit CLI json reports coremlBytes");
}

void check_workloads(const std::string & gguf,
                     const std::vector<parakeet::CoremlDiarizerBundle> & bundles) {
    const parakeet::FitResult offline = project(gguf, kShortSeconds, kDefaultMargin);
    check(offline.coreml_bytes == charge({bundles.front()}), "short input charges the smallest bundle");
    check(reports_sidecars(offline), "report lists the sidecar charge");
    check_cli_json(gguf, offline.coreml_bytes);
    const parakeet::FitResult long_input = project(gguf, kLongSeconds, kDefaultMargin);
    const bool nemotron = long_input.model_type == kNemotronType;
    check(long_input.coreml_bytes == (nemotron ? charge(bundles) : 0),
          nemotron ? "long-form input charges every bundle" : "input above every bundle charges none");
}

void check_verdict(const std::string & gguf) {
    const parakeet::FitResult baseline = project(gguf, kShortSeconds, kDefaultMargin);
    if (!baseline.device_shares_host_memory || !baseline.fits) return;
    const uint64_t margin = margin_between(baseline);
    check(!project(gguf, kShortSeconds, margin).fits, "sidecar charge makes a tight budget fail");
    setenv(parakeet::kCoremlDiarizerDisableEnv, "1", 1);
    const parakeet::FitResult disabled = project(gguf, kShortSeconds, margin);
    unsetenv(parakeet::kCoremlDiarizerDisableEnv);
    check(disabled.fits, "the same budget fits with the diarizers disabled");
    check(disabled.coreml_bytes == 0 && !reports_sidecars(disabled),
          "disable env removes the sidecar charge");
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 1;
    }
    const std::vector<parakeet::CoremlDiarizerBundle> bundles =
        parakeet::list_coreml_diarizer_bundles(argv[1]);
    if (bundles.empty()) {
        std::printf("SKIP: no Core ML diarizer beside %s\n", argv[1]);
        return kSkip;
    }
    check_workloads(argv[1], bundles);
    check_verdict(argv[1]);
    std::printf("[fit-coreml-diarizers] %s\n", g_passed ? "PASS" : "FAIL");
    return g_passed ? 0 : 1;
}
