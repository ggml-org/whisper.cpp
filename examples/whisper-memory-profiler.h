// Standalone memory profiler for whisper.cpp
// Usage: Set environment variable WHISPER_PROFILE_MEMORY=1 before running
//
// Features:
//   - Peak memory: Uses OS-tracked PeakWorkingSetSize (Windows) - most accurate!
//   - Average memory: Background polling every 10ms
//   - Cross-platform: Windows, Linux, macOS
//
// Example:
//   Windows: set WHISPER_PROFILE_MEMORY=1 && main.exe -f audio.wav
//   Linux:   WHISPER_PROFILE_MEMORY=1 ./main -f audio.wav

#pragma once

#include <atomic>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

class WhisperMemoryProfiler {
private:
    std::atomic<bool> running{false};
    std::atomic<size_t> sum_memory{0};
    std::atomic<int> sample_count{0};
    std::thread profiler_thread;
    int poll_interval_ms;

    static size_t get_current_memory() {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            return (size_t)pmc.WorkingSetSize;
        }
        return 0;
#else
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
#ifdef __APPLE__
            return (size_t)usage.ru_maxrss;
#else
            return (size_t)usage.ru_maxrss * 1024;
#endif
        }
        return 0;
#endif
    }

    static size_t get_peak_memory() {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            return (size_t)pmc.PeakWorkingSetSize;
        }
        return 0;
#else
        // On Unix, ru_maxrss already gives peak, same as current
        return get_current_memory();
#endif
    }

    void profiler_loop() {
        while (running.load()) {
            size_t current = get_current_memory();
            if (current > 0) {
                // Only accumulate for average (peak tracked by OS)
                sum_memory.fetch_add(current);
                sample_count.fetch_add(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
        }
    }

public:
    WhisperMemoryProfiler(int poll_ms = 10) : poll_interval_ms(poll_ms) {}

    ~WhisperMemoryProfiler() {
        stop();
    }

    void start() {
        if (!running.load()) {
            running.store(true);
            profiler_thread = std::thread(&WhisperMemoryProfiler::profiler_loop, this);
        }
    }

    void stop() {
        if (running.load()) {
            running.store(false);
            if (profiler_thread.joinable()) {
                profiler_thread.join();
            }
        }
    }

    void print_stats() {
        int samples = sample_count.load();
        if (samples == 0) {
            fprintf(stderr, "\nMemory profiling: No samples collected\n");
            return;
        }

        const double to_mb = 1.0 / (1024.0 * 1024.0);
        const size_t peak = get_peak_memory();
        const double peak_mb = peak * to_mb;
        const double avg_mb = (sum_memory.load() / (double)samples) * to_mb;

        fprintf(stderr, "\n");
        fprintf(stderr, "========= Memory Profiling =========\n");
        fprintf(stderr, "Peak memory:    %8.2f MB (OS-tracked)\n", peak_mb);
        fprintf(stderr, "Average memory: %8.2f MB\n", avg_mb);
        fprintf(stderr, "Samples taken:  %8d\n", samples);
        fprintf(stderr, "====================================\n");
    }

    void reset() {
        sum_memory.store(0);
        sample_count.store(0);
        // Note: Cannot reset OS-tracked peak (PeakWorkingSetSize)
    }

    size_t get_peak_memory_snapshot() const {
        return get_peak_memory();
    }

    double get_average_memory() const {
        int samples = sample_count.load();
        if (samples == 0) return 0.0;
        return (double)sum_memory.load() / samples;
    }
};

// Global instance (optional, for convenience)
static WhisperMemoryProfiler* g_memory_profiler = nullptr;

// Helper functions for easy integration
inline bool whisper_should_profile_memory() {
    const char* env = std::getenv("WHISPER_PROFILE_MEMORY");
    return env != nullptr && (env[0] == '1' || env[0] == 't' || env[0] == 'T');
}

inline void whisper_profiler_start() {
    if (whisper_should_profile_memory()) {
        if (g_memory_profiler == nullptr) {
            g_memory_profiler = new WhisperMemoryProfiler();
        }
        g_memory_profiler->start();
        fprintf(stderr, "[Memory profiling enabled - polling every 10ms]\n");
    }
}

inline void whisper_profiler_stop_and_print() {
    if (g_memory_profiler != nullptr) {
        g_memory_profiler->stop();
        g_memory_profiler->print_stats();
        delete g_memory_profiler;
        g_memory_profiler = nullptr;
    }
}
