// Memory-cycle regression test for tts_cpp::supertonic::Engine.
//
// Constructs + destroys N Engines back-to-back on the same thread,
// synthesizing once per engine so every per-stage thread_local graph
// cache populates before teardown.  Verifies that resident memory
// (RSS on Linux, phys_footprint on macOS) does not drift more than a
// small tolerance across cycles — which guards against the per-cycle
// gallocr leak in `supertonic_safe_gallocr_free`'s dead-backend skip
// path (fixed by registering each thread_local cache for engine-
// destructor release; see comments on
// `release_*_thread_local_caches` in supertonic_internal.h).
//
// Usage: test-supertonic-engine-cycle <supertonic.gguf> [REF_DIR_ignored]
//
// We accept (and ignore) the REF_DIR argument so this harness can
// reuse the existing `add_supertonic_harness` CMake helper without
// special-casing.
//
// Pass criteria: max(RSS) over cycles ≥ 2 ≤ first-cycle RSS + 5 MB.
// First cycle is excluded from the drift calculation because the
// initial backend init (loading metal-library, vulkan-icd, openmp
// pool, etc.) inflates RSS once on first construction; subsequent
// cycles reuse the process-singleton backend registry and should
// stay flat.

#include "tts-cpp/supertonic/engine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__APPLE__)
  #include <mach/mach.h>
#elif defined(__linux__)
  #include <unistd.h>
#endif

namespace {

size_t rss_bytes() {
#if defined(__APPLE__)
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  (task_info_t)&info, &count) != KERN_SUCCESS) {
        return 0;
    }
    return (size_t)info.phys_footprint;
#elif defined(__linux__)
    FILE * f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long size = 0, resident = 0;
    if (std::fscanf(f, "%ld %ld", &size, &resident) != 2) {
        std::fclose(f);
        return 0;
    }
    std::fclose(f);
    return (size_t)resident * (size_t)sysconf(_SC_PAGESIZE);
#else
    return 0;
#endif
}

double mb(size_t bytes) {
    return (double)bytes / (1024.0 * 1024.0);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <supertonic.gguf> [REF_DIR_ignored] "
            "[n_cycles=20] [n_gpu_layers=0]\n", argv[0]);
        return 2;
    }
    const std::string model = argv[1];
    // argv[2] is REF_DIR — ignored here; accepted for parity with the
    // other Supertonic harnesses so `add_supertonic_harness` works.
    const int n_cycles     = argc >= 4 ? std::atoi(argv[3]) : 20;
    const int n_gpu_layers = argc >= 5 ? std::atoi(argv[4]) : 0;

    if (n_cycles < 2) {
        std::fprintf(stderr, "n_cycles must be >= 2\n");
        return 2;
    }

    const size_t baseline = rss_bytes();
    std::printf("test-supertonic-engine-cycle: baseline RSS = %.1f MB, "
                "n_cycles=%d, n_gpu_layers=%d\n",
                mb(baseline), n_cycles, n_gpu_layers);

    std::vector<size_t> samples;
    samples.reserve((size_t)n_cycles);

    for (int i = 0; i < n_cycles; ++i) {
        tts_cpp::supertonic::EngineOptions opts;
        opts.model_gguf_path = model;
        opts.n_gpu_layers    = n_gpu_layers;
        opts.n_threads       = 4;
        opts.steps           = 5;

        tts_cpp::supertonic::Engine engine(opts);
        auto result = engine.synthesize("The quick brown fox jumps over the lazy dog.");
        if (result.pcm.empty()) {
            std::fprintf(stderr, "FAIL: cycle %d produced empty PCM\n", i);
            return 1;
        }
        // Engine destroyed at scope exit; per-stage caches released
        // via release_*_thread_local_caches() inside free_supertonic_model.

        const size_t rss = rss_bytes();
        samples.push_back(rss);
    }

    const size_t first_rss = samples.front();
    const size_t max_rss   = *std::max_element(samples.begin() + 1,
                                               samples.end());
    const size_t last_rss  = samples.back();
    const double drift_mb  = mb(max_rss) - mb(first_rss);
    const double last_mb   = mb(last_rss) - mb(first_rss);

    std::printf("  cycle 1: %.1f MB\n", mb(first_rss));
    std::printf("  max(cycle 2..%d): %.1f MB  (delta vs cycle 1: %+.2f MB)\n",
                n_cycles, mb(max_rss), drift_mb);
    std::printf("  cycle %d: %.1f MB  (delta vs cycle 1: %+.2f MB)\n",
                n_cycles, mb(last_rss), last_mb);

    // Tolerance: 5 MB.  The first cycle's RSS captures one-time
    // process-singleton initialisation (metal library compile, vulkan
    // ICD load, openmp pool, ggml backend registry); cycles 2..N
    // reuse all of that and should stay flat.
    constexpr double kDriftMb = 5.0;
    if (drift_mb > kDriftMb) {
        std::fprintf(stderr,
            "FAIL: max RSS drift %.2f MB across cycles 2..%d exceeds "
            "%.2f MB threshold\n", drift_mb, n_cycles, kDriftMb);
        return 1;
    }

    std::printf("PASS: max RSS drift %.2f MB is within %.2f MB threshold\n",
                drift_mb, kDriftMb);
    return 0;
}
