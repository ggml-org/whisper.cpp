#include "tts-cpp/pocket/fit.h"
#include "pocket/flow_lm.h"
#include "pocket/mimi.h"
#include "pocket/frontend.h"
#include "pocket/generation.h"
#include "fit_util.h"
#include "backend_selection.h"
#include "pocket/reference_audio.h"
#include "pocket/backend.h"
#include "pocket/ggml_utils.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <mach/mach.h>
#if TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
#include <os/proc.h>
#endif
#include <sys/sysctl.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace tts_cpp::pocket {
namespace {
using ::tts_cpp::fitutil::sat_add;
using ::tts_cpp::fitutil::sat_mul;
constexpr uint64_t mib = 1024*1024;
// ggml CPU reports total RAM as free RAM. Use OS availability instead.
bool available_memory(uint64_t & available, uint64_t & total) {
    bool known = false;
#if defined(__APPLE__)
    size_t size = sizeof(total);
    if (sysctlbyname("hw.memsize", &total, &size, nullptr, 0)) return false;
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    const auto host = mach_host_self();
    vm_size_t page_size = 0;
    const bool ok = host_page_size(host, &page_size) == KERN_SUCCESS &&
        host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) == KERN_SUCCESS;
    mach_port_deallocate(mach_task_self(), host);
    if (!ok) return false;
    // Inactive pages can be reclaimed; do not count compressed or wired pages.
    available = sat_mul(sat_add(vm.free_count, vm.inactive_count), page_size); known = true;
#if TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
    // Mobile apps can hit a per-process dirty-memory limit while the device
    // still has RAM. Zero means no remaining app allowance (or a non-app host).
    if (__builtin_available(iOS 13.0, tvOS 13.0, watchOS 6.0, *))
        available = std::min(available, uint64_t(os_proc_available_memory()));
    else return false;
#endif
#elif defined(_WIN32)
    MEMORYSTATUSEX status{}; status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return false;
    available = status.ullAvailPhys; total = status.ullTotalPhys; known = true;
#elif defined(__linux__)
    std::ifstream input("/proc/meminfo");
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line); std::string name, unit; uint64_t value;
        if (!(row >> name >> value >> unit) || unit != "kB") continue;
        if (name == "MemAvailable:") { available = sat_mul(value, 1024); known = true; }
        if (name == "MemTotal:") total = sat_mul(value, 1024);
    }
#else
    return false;
#endif
    return known && total > 0 && available <= total;
}
void validate(const FitOptions & o) {
    const auto & e = o.engine;
    detail::backend_request(e.backend);
    if (o.text.empty() || o.text.size() > 65536 || e.context < 1 || e.context > 8192 ||
        e.n_threads < 1 || e.n_threads > 1024 || e.output_sample_rate < 8000 || e.output_sample_rate > 192000 ||
        e.steps < 1 || e.steps > 64 || !std::isfinite(e.temperature) || e.temperature < 0 || e.temperature > 10 ||
        !std::isfinite(e.noise_clamp) || e.noise_clamp < 0 || !std::isfinite(e.eos_threshold) ||
        e.frames_after_eos < -1 || e.frames_after_eos > 100 || e.max_tokens < 1 || e.max_tokens > 1024 ||
        e.voice_path.empty() == e.reference_audio_path.empty())
        throw std::invalid_argument("invalid Pocket fit options");
}
}
FitResult fit_params(const FitOptions & o) {
    FitResult r; r.model_variant = "pocket-tts";
    try {
        validate(o);
        const auto & e = o.engine;
        // Snapshot before backend initialization and metadata/container loads:
        // all of those allocations are charged in the projection below.
        if (!e.backends_dir.empty()) ::tts_cpp::detail::set_backends_directory(e.backends_dir);
        uint64_t host_free = 0, host_total = 0;
        if (!available_memory(host_free, host_total)) {
            r.reason = "memory-unavailable"; r.report = "Cannot determine currently available system memory"; return r;
        }
        auto backend = detail::select_backend(e.use_gpu, e.backend);
        const auto device = ggml_backend_get_device(backend.get());
        r.device_name = ggml_backend_name(backend.get());
        r.device_is_cpu = ::tts_cpp::detail::backend_is_cpu(backend.get());
        const bool is_hexagon = ::tts_cpp::detail::backend_is_hexagon(backend.get());
        const auto hexagon = is_hexagon ? detail::hexagon_runtime(backend.get()) : detail::HexagonRuntime{};
        r.device_shares_host_memory = detail::shares_host_memory(backend.get());
        if (!r.device_is_cpu) r.device_name += std::string(" (") + ggml_backend_dev_description(device) + ')';
        detail::Frontend frontend(e.frontend_path);
        const auto chunks = frontend.split(o.text, e.max_tokens);
        if (chunks.empty()) throw std::invalid_argument("no text chunks");
        int prefill = 1;
        uint64_t frames = 0;
        std::vector<std::pair<size_t, int>> shapes;
        for (const auto & chunk : chunks) {
            const auto tokens = frontend.encode(chunk.text).size();
            const int tail = e.frames_after_eos >= 0 ? e.frames_after_eos : chunk.tail_frames;
            const int count = detail::frame_budget(tokens, tail).max_frames();
            if (tokens + count > size_t(e.context)) {
                r.reason = "workload-too-large"; r.report = "Text chunk exceeds the context capacity"; return r;
            }
            prefill = std::max(prefill, int(tokens));
            shapes.push_back({tokens, count}); frames = sat_add(frames, count);
        }
        auto flow = detail::FlowLM::measure(e.flow_lm_path, e.context, prefill, e.n_threads, device);
        const bool reference = !e.reference_audio_path.empty();
        uint64_t reference_host = 0;
        int voice_frames;
        if (reference) {
            const detail::ReferenceAudio audio(e.reference_audio_path);
            const auto file_bytes = audio.file_bytes();
            const auto rate = audio.sample_rate();
            const auto audio_frames = audio.frames();
            const auto samples = uint64_t(std::ceil(double(audio_frames)*detail::Mimi::sample_rate()/rate));
            voice_frames = int((samples+detail::Mimi::frame_samples()-1)/detail::Mimi::frame_samples()) + int(flow.config.bos_before_voice);
            reference_host = sat_add(sat_mul(file_bytes, 8), sat_mul(30ull*192000+samples, 8));
            if (voice_frames > prefill && voice_frames < e.context)
                flow = detail::FlowLM::measure(e.flow_lm_path, e.context, voice_frames, e.n_threads, device);
        } else {
            voice_frames = detail::FlowLM::measure_voice(e.voice_path, flow, e.context);
        }
        for (const auto & shape : shapes) if (voice_frames + shape.first + shape.second > size_t(e.context)) {
            r.reason = "workload-too-large"; r.report = "Voice, text and maximum audio frames exceed context capacity"; return r;
        }
        const auto codec = detail::Mimi::measure(e.mimi_path, e.n_threads, reference, device);
        if (flow.source_hash != frontend.source_hash() || flow.source_hash != codec.source_hash ||
            flow.config.vocab_size != frontend.vocab_size() || flow.config.latent_dim != detail::Mimi::latent_dim())
            throw std::invalid_argument("Pocket artifacts do not belong to the same checkpoint");
        r.device.weights_bytes = sat_add(flow.weights, codec.weights);
        r.device.state_bytes = sat_add(flow.state, codec.state);
        const bool opencl = ::tts_cpp::detail::backend_is_opencl(backend.get());
        const auto backend_scratch = is_hexagon ? hexagon.shared_bytes :
            opencl ? detail::opencl_scratch_bound(flow.opencl_scratch, codec.opencl_scratch) : uint64_t(0);
        r.device.state_bytes = sat_add(r.device.state_bytes, backend_scratch);
        r.device.lm_compute_bytes = flow.compute;
        r.device.codec_compute_bytes = codec.compute;
        r.device.total_bytes = sat_add(sat_add(r.device.weights_bytes, r.device.state_bytes), sat_add(flow.compute, codec.compute));

        const auto voice_bytes = sat_mul(sat_mul(uint64_t(voice_frames), flow.config.dim), uint64_t(8)*flow.config.layers);
        uint64_t persistent_host = sat_add(frontend.memory_bytes(), voice_bytes);
        persistent_host = sat_add(persistent_host, sat_add(flow.metadata, codec.metadata));
        // Conservative STL/allocator bookkeeping and CPU thread stack bounds.
        persistent_host = sat_add(persistent_host, sat_mul(8*mib, 2ull*e.n_threads+2));
        auto runtime_host = sat_mul(sat_add(flow.graph_metadata, codec.graph_metadata), 2);
        runtime_host = sat_add(runtime_host, flow.host_compute);
        runtime_host = sat_add(runtime_host, sat_mul(uint64_t(prefill)*flow.config.dim, 16));
        runtime_host = sat_add(runtime_host, sat_mul(o.text.size()+1, 64));
        runtime_host = sat_add(runtime_host, 2*mib); // codec mask, latent queue, chunk copies
        const auto native_samples = sat_mul(frames, detail::Mimi::frame_samples());
        const auto output_samples = uint64_t(std::ceil(double(native_samples)*e.output_sample_rate/detail::Mimi::sample_rate()));
        // Batch-vector growth (old+new arenas), output chunks, and the sinc
        // resampler's retained input. At 24 kHz the latter is an overestimate.
        runtime_host = sat_add(runtime_host, sat_add(sat_mul(native_samples, 12), sat_mul(output_samples, 16)));
        auto load_host = sat_add(std::max(flow.load_staging, codec.load_staging), reference_host);
        load_host = sat_add(load_host, sat_mul(std::filesystem::file_size(e.frontend_path), 64));
        r.host_bytes = sat_add(sat_add(persistent_host, std::max(runtime_host, load_host)), hexagon.host_bytes_bound);
        if (is_hexagon && r.device.total_bytes > hexagon.max_batch_vmem_bytes) {
            r.reason = "workload-too-large";
            r.report = "Pocket Hexagon weights, state and overlapping arenas exceed the conservative session mapping budget";
            return r;
        }
        if (r.device_is_cpu || is_hexagon) {
            r.device_free_bytes = host_free; r.device_total_bytes = host_total;
        } else {
            size_t free = 0, total = 0;
            ::tts_cpp::detail::backend_memory(backend.get(), free, total);
            if (!total || free > total) {
                r.reason = "memory-unavailable"; r.report = "Cannot determine device memory"; return r;
            }
            r.device_free_bytes = free; r.device_total_bytes = total;
            if (r.device_shares_host_memory) {
                r.device_free_bytes = std::min(r.device_free_bytes, host_free);
                r.device_total_bytes = std::min(r.device_total_bytes, host_total);
            }
        }
        if (o.memory_budget_bytes) {
            r.device_free_bytes = std::min(r.device_free_bytes, o.memory_budget_bytes);
            host_free = std::min(host_free, o.memory_budget_bytes);
        }
        const auto required = sat_add(sat_add(r.device.total_bytes,
            r.device_shares_host_memory ? r.host_bytes : 0), o.margin_bytes);
        const auto host_required = sat_add(r.host_bytes, o.margin_bytes);
        r.fits = required <= r.device_free_bytes && (r.device_shares_host_memory || host_required <= host_free);
        r.status = r.fits ? FitStatus::Success : FitStatus::Failure;
        r.reason = r.fits ? "fits" : "does-not-fit";
        std::ostringstream report;
        report << "Pocket " << r.device_name << " memory preflight (bytes)\nweights: " << r.device.weights_bytes
               << "\nstate: " << r.device.state_bytes << "\nFlowLM compute: " << flow.compute
               << "\nMimi compute: " << codec.compute << "\nhost extras: " << r.host_bytes
               << "\nmargin: " << o.margin_bytes << "\nrequired: " << required
               << "\navailable/budget: " << r.device_free_bytes << "\nmaximum audio frames: " << frames
               << "\nshared host memory: " << (r.device_shares_host_memory ? "yes" : "no")
               << "\nhost required/available: " << host_required << '/' << host_free
               << "\nresult: " << r.reason << '\n';
        if (is_hexagon)
            report << "Hexagon registry session RPC queues (included in state): " << hexagon.shared_bytes
                   << "\nHexagon registry host/graph metadata bound (included in host): " << hexagon.host_bytes_bound
                   << "\nHexagon sessions: " << hexagon.session_count
                   << "; RPC buffer/batch mapping limits: " << hexagon.max_buffer_bytes << '/' << hexagon.max_batch_vmem_bytes
                   << "; VTCM: " << hexagon.vtcm_bytes
                   << "\nShared OS available RAM is used because HTP device memory telemetry is unavailable.\n";
        else if (opencl)
            report << "OpenCL shared backend scratch bound (included in state): " << backend_scratch
                   << "\nConservative F32 backend scratch bound for reviewed ggml-speech; includes backend init and pool growth."
                   << "\nOpenCL available memory is a driver estimate/cap, limited by OS RAM on Adreno.\n";
        r.report = report.str();
    } catch (const std::invalid_argument & error) {
        r.reason = "invalid-arguments"; r.report = error.what();
    } catch (const detail::BackendUnavailable & error) {
        r.reason = "no-backend-device"; r.report = error.what();
    } catch (const std::exception & error) {
        r.reason = "measurement-failed"; r.report = error.what();
    } catch (...) { r.reason = "measurement-failed"; r.report = "Unknown Pocket preflight failure"; }
    return r;
}
}
