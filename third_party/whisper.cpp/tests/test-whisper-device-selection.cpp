// QVAC (see PATCHES.md): native device selection contracts, without a model
// or backend initialization. Synthetic devices make ordinal checks independent
// of which accelerators are installed on the test machine.
#include "whisper-device-selection.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

void expect(bool ok, const std::string & what) {
    ++g_checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

// Do not leak changes into other tests or depend on the caller's environment.
void set_device_env(const char * value) {
#if defined(_WIN32)
    _putenv_s("WHISPER_ARG_DEVICE", value ? value : "");
#else
    if (value) {
        setenv("WHISPER_ARG_DEVICE", value, 1);
    } else {
        unsetenv("WHISPER_ARG_DEVICE");
    }
#endif
}

struct device_env_guard {
    bool present;
    std::string value;

    explicit device_env_guard(const char * replacement) {
        const char * previous = std::getenv("WHISPER_ARG_DEVICE");
        present = previous != nullptr;
        value = previous ? previous : "";
        set_device_env(replacement);
    }

    ~device_env_guard() {
        set_device_env(present ? value.c_str() : nullptr);
    }
};

using whisper_device_selection::device;
using whisper_device_selection::resolve;

void expect_device(const std::vector<device> & candidates, const std::string & target, int expected) {
    int selected = -123;
    std::string error = "previous error";
    const bool ok = resolve(target, candidates, selected, error);
    expect(ok && selected == expected && error.empty(), "select '" + target + "' => " + std::to_string(expected));
}

void expect_rejected(const std::vector<device> & candidates, const std::string & target,
                     const std::string & diagnostic) {
    int selected = 37;
    std::string error;
    const bool ok = resolve(target, candidates, selected, error);
    expect(!ok && selected == 37 && error.find(diagnostic) != std::string::npos,
           "reject '" + target + "' without changing the selected device: " + diagnostic);
}

void test_filtered_ordinals() {
    const std::vector<device> candidates = {
        {"CPU",    GGML_BACKEND_DEVICE_TYPE_CPU,   "host"},
        {"NPU",    GGML_BACKEND_DEVICE_TYPE_ACCEL, "automatic accelerator"},
        {"HTP0",   GGML_BACKEND_DEVICE_TYPE_GPU,   "Hexagon"},
        {"CPU2",   GGML_BACKEND_DEVICE_TYPE_CPU,   "another host"},
        {"Adreno", GGML_BACKEND_DEVICE_TYPE_IGPU,  "integrated GPU"},
        {"GPU1",   GGML_BACKEND_DEVICE_TYPE_GPU,   "discrete GPU"},
    };

    // Registry indices are 2, 4 and 5; Whisper's GPU ordinals must be 0, 1, 2.
    expect_device(candidates, "HTP0",   0);
    expect_device(candidates, "Adreno", 1);
    expect_device(candidates, "GPU1",   2);
    expect_device(candidates, "0",      0);
    expect_device(candidates, "1",      1);
    expect_device(candidates, "2",      2);
    expect_device(candidates, "+1",     1);
    expect_device(candidates, "0002",   2);

    expect_rejected(candidates, "3", "unavailable");
    expect_rejected(candidates, "4", "unavailable");
    expect_rejected(candidates, "CPU", "not a selectable GPU/IGPU");
    expect_rejected(candidates, "NPU", "not a selectable GPU/IGPU");
    expect_rejected(candidates, "htp0", "unavailable");
    expect_rejected(candidates, "HTP", "unavailable");
    expect_rejected(candidates, "missing-device", "unavailable");
    expect_rejected({}, "0", "unavailable");
}

void test_strict_targets() {
    const std::vector<device> candidates = {{"HTP0", GGML_BACKEND_DEVICE_TYPE_GPU, "Hexagon"}};
    expect_rejected(candidates, "", "device must be");
    for (const char * target : {"-1", "-0", "+", "1x", "0.0", "0x0", "+-1", "0 ", "2147483648",
                                "999999999999999999999999999999999999999999999999"}) {
        expect_rejected(candidates, target, "invalid GPU device index");
    }
    // Leading whitespace and padded names must not turn into a successful
    // partial integer/name parse either.
    expect_rejected(candidates, " 0", "unavailable");
    expect_rejected(candidates, "HTP0 ", "unavailable");
    expect_rejected(candidates, "2147483647", "unavailable");
}

void test_ambiguous_names() {
    const std::vector<device> candidates = {
        {"duplicate", GGML_BACKEND_DEVICE_TYPE_GPU,  "first"},
        {"CPU",       GGML_BACKEND_DEVICE_TYPE_CPU,  "host"},
        {"duplicate", GGML_BACKEND_DEVICE_TYPE_IGPU, "second"},
    };
    expect_rejected(candidates, "duplicate", "ambiguous");
    expect_device(candidates, "0", 0);
    expect_device(candidates, "1", 1);
}

void test_environment_and_cpu_override() {
    using whisper_device_selection::apply;
    const char * absent_device = "whisper-test-device-that-is-not-registered";

    {
        device_env_guard env(nullptr);
        int selected = 19;
        std::string error = "stale";
        expect(apply("", false, true, selected, error, {}) && selected == 19 && error.empty(),
               "without HTP or an explicit/environment request, preserve the default selection");
    }
    {
        device_env_guard env("1invalid");
        int selected = 19;
        std::string error;
        expect(!apply("", false, true, selected, error) && selected == 19 &&
               error.find("WHISPER_ARG_DEVICE: invalid GPU device index") != std::string::npos,
               "invalid environment target fails clearly");
        expect(!apply(absent_device, true, true, selected, error) && selected == 19 &&
               error.find(absent_device) != std::string::npos &&
               error.find("WHISPER_ARG_DEVICE") == std::string::npos,
               "explicit target takes precedence over invalid environment target");
        expect(apply("", false, false, selected, error) && selected == 19 && error.empty(),
               "CPU mode ignores invalid environment target");
        expect(!apply("0", true, false, selected, error) && selected == 19 &&
               error.find("cannot be combined with --no-gpu") != std::string::npos,
               "explicit numeric device conflicts with CPU mode");
        expect(!apply("HTP0", true, false, selected, error) && selected == 19 &&
               error.find("cannot be combined with --no-gpu") != std::string::npos,
               "explicit named device conflicts with CPU mode");
        expect(!apply("", true, true, selected, error) && selected == 19 &&
               error.find("WHISPER_ARG_DEVICE") == std::string::npos,
               "empty explicit target cannot fall back to the environment");
    }
    {
        device_env_guard env("");
#if !defined(_WIN32)
        // Windows _putenv_s(name, "") removes the variable instead of setting
        // an empty value. POSIX exposes empty and absent as distinct states.
        int selected = 19;
        std::string error;
        expect(!apply("", false, true, selected, error) && selected == 19 &&
               error.find("WHISPER_ARG_DEVICE") != std::string::npos,
               "empty environment target is invalid rather than default selection");
#endif
    }
}

void test_automatic_htp_preference() {
    using whisper_device_selection::apply;
    const std::vector<device> candidates = {
        {"CPU",    GGML_BACKEND_DEVICE_TYPE_CPU,  "host"},
        {"Adreno", GGML_BACKEND_DEVICE_TYPE_IGPU, "OpenCL"},
        {"CPU2",   GGML_BACKEND_DEVICE_TYPE_CPU,  "another host"},
        {"HTP0",   GGML_BACKEND_DEVICE_TYPE_GPU,  "Hexagon"},
    };
    device_env_guard env(nullptr);
    int selected = 0;
    std::string error;
    expect(apply("", false, true, selected, error, candidates) && selected == 1 && error.empty(),
           "automatic HTP selection beats an earlier OpenCL device using the filtered GPU ordinal");
    expect(apply("0", true, true, selected, error, candidates) && selected == 0 && error.empty(),
           "explicit OpenCL ordinal overrides automatic HTP selection");
    expect(apply("Adreno", true, true, selected, error, candidates) && selected == 0,
           "explicit OpenCL name overrides automatic HTP selection");
    expect(apply("HTP0", true, true, selected, error, candidates) && selected == 1,
           "explicit HTP selection remains available");
    selected = 0;
    expect(apply("", false, false, selected, error, candidates) && selected == 0 && error.empty(),
           "CPU mode does not automatically select HTP");

    {
        device_env_guard target("Adreno");
        selected = 1;
        expect(apply("", false, true, selected, error, candidates) && selected == 0,
               "environment OpenCL selection overrides automatic HTP selection");
        expect(apply("HTP0", true, true, selected, error, candidates) && selected == 1,
               "explicit HTP overrides an OpenCL environment target");
    }
    {
        device_env_guard target("missing-device");
        selected = 0;
        expect(!apply("", false, true, selected, error, candidates) && selected == 0 &&
                   error.find("WHISPER_ARG_DEVICE:") != std::string::npos,
               "invalid environment target is not hidden by automatic HTP selection");
    }
    const std::vector<device> no_htp = {
        {"Adreno", GGML_BACKEND_DEVICE_TYPE_IGPU, "OpenCL"},
        {"GPU1",   GGML_BACKEND_DEVICE_TYPE_GPU,  "other GPU"},
    };
    selected = 1;
    expect(apply("", false, true, selected, error, no_htp) && selected == 1 && error.empty(),
           "non-HTP topology preserves the caller's default device");
    selected = 0;
    expect(apply("", false, true, selected, error,
                 {{"HTP0", GGML_BACKEND_DEVICE_TYPE_ACCEL, "not a selectable GPU"}}) && selected == 0,
           "HTP-like name on a non-GPU device does not change selection");
    expect(apply("", false, true, selected, error,
                 {{"HTP0", GGML_BACKEND_DEVICE_TYPE_GPU, "first"},
                  {"HTP0", GGML_BACKEND_DEVICE_TYPE_GPU, "duplicate"}}) && selected == 0 && error.empty(),
           "ambiguous HTP name preserves the default without an explicit-selection error");
}

void test_primary_initialization() {
    using whisper_device_selection::check_primary_backend;
    using whisper_device_selection::requested_backend;
    const std::vector<device> candidates = {
        {"CPU", GGML_BACKEND_DEVICE_TYPE_CPU, "host"},
        {"HTP0", GGML_BACKEND_DEVICE_TYPE_GPU, "Hexagon"},
    };
    device_env_guard env(nullptr);
    int selected = -1;
    std::string error;
    expect(resolve("HTP0", candidates, selected, error) && selected == 0,
           "registered explicit target resolves before initialization");
    const std::string requested = requested_backend(true, true, selected, candidates);
    expect(requested == "HTP0", "explicit target retains its expected primary backend");
    expect(!check_primary_backend(requested, "CPU", error) &&
           error.find("HTP0") != std::string::npos && error.find("CPU") != std::string::npos,
           "registered target whose init falls back to CPU is rejected");
    expect(!check_primary_backend(requested, nullptr, error) && error.find("none") != std::string::npos,
           "explicit target with no initialized primary backend is rejected");
    expect(check_primary_backend(requested, "HTP0", error) && error.empty(),
           "successfully initialized explicit target is accepted");
    const std::string default_target = requested_backend(false, true, selected, candidates);
    expect(default_target.empty() && check_primary_backend(default_target, "CPU", error),
           "default selection retains its existing initialization fallback");
    {
        device_env_guard env_target("HTP0");
        const std::string environment_target = requested_backend(false, true, selected, candidates);
        expect(environment_target == "HTP0" && !check_primary_backend(environment_target, "CPU", error),
               "environment target also rejects initialization fallback");
        const std::string cpu_target = requested_backend(false, false, selected, candidates);
        expect(cpu_target.empty() && check_primary_backend(cpu_target, "CPU", error),
               "CPU mode ignores the environment target during initialization checks");
    }
}

} // namespace

int main() {
    test_filtered_ordinals();
    test_strict_targets();
    test_ambiguous_names();
    test_environment_and_cpu_override();
    test_automatic_htp_preference();
    test_primary_initialization();
    if (g_failures) {
        std::fprintf(stderr, "test-whisper-device-selection: %d/%d checks failed\n", g_failures, g_checks);
        return 1;
    }
    std::printf("test-whisper-device-selection: %d checks passed\n", g_checks);
    return 0;
}
