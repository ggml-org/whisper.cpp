#include "tts-cpp/lavasr/enhancer.h"

#include "../backend_selection.h" // init_gpu_backend, init_cpu_backend
#include "enhancer.h"             // internal: enhance(EnhancerWeights, ...)
#include "enhancer_core.h"        // internal: EnhancerWeights
#include "enhancer_ggml.h"        // internal: EnhancerGgml, enhancer_ggml_*
#include "enhancer_gguf.h"        // internal: load_enhancer_gguf

#include "ggml-backend.h"

#include <cstdio>
#include <mutex>
#include <stdexcept>

namespace tts_cpp::lavasr {

struct Enhancer::Impl {
    EnhancerWeights weights;

    // ggml compute-graph engine.  `backend` is a GPU backend when one was
    // requested and available, otherwise the ggml-CPU backend (still markedly
    // faster than the scalar core and validated bit-comparable to it).  Both are
    // null only when even the ggml-CPU backend could not be initialised, in
    // which case enhance() runs the scalar core directly.  `backend` is owned
    // here and freed after `graph` (which references it).
    ggml_backend_t     backend = nullptr;
    EnhancerGgml *     graph   = nullptr;
    mutable std::mutex graph_mu; // serialises the single reusable graph allocator

    // Telemetry: registered backend name ("Vulkan0"/"CUDA0"/"Metal"/"CPU"/...)
    // and the resolved compute device.  `device` stays CPU for the ggml-CPU
    // graph (and the scalar fallback) so a host can tell a real GPU run from an
    // accelerated-CPU run.
    std::string   backend_name = "CPU";
    BackendDevice device       = BackendDevice::CPU;

    ~Impl() {
        if (graph) {
            enhancer_ggml_free(graph);
        }
        if (backend) {
            ggml_backend_free(backend);
        }
    }
};

Enhancer::Enhancer() : impl_(std::make_unique<Impl>()) {}
Enhancer::~Enhancer() = default;

std::unique_ptr<Enhancer> Enhancer::load(const std::string & gguf_path,
                                         const EnhancerOptions & opts) {
    std::unique_ptr<Enhancer> e(new Enhancer());
    std::string               err;
    if (!load_enhancer_gguf(gguf_path, e->impl_->weights, &err)) {
        throw std::runtime_error(err.empty() ? "lavasr: failed to load enhancer GGUF"
                                             : err);
    }

    // Pick a compute backend for the ggml graph: a GPU when requested and
    // available, otherwise the ggml-CPU backend.  The scalar core (below) stays
    // as the correctness oracle and the last-resort fallback.
    ggml_backend_t backend = nullptr;
    BackendDevice  device  = BackendDevice::CPU;
    if (opts.use_gpu) {
        // n_gpu_layers is a whole-network switch for the enhancer (the graph
        // runs entirely on one backend); pass a positive value to request a GPU.
        backend = tts_cpp::detail::init_gpu_backend(
            /*n_gpu_layers=*/99, opts.verbose, "lavasr-enhancer", opts.vulkan_device);
        if (backend) {
            device = BackendDevice::GPU;
        }
        // backend == nullptr: no GPU available/selected — fall through to CPU.
    }
    if (!backend) {
        // Run the graph on the ggml-CPU backend: ~7x faster than the scalar core
        // and validated bit-comparable to it (test-lavasr-enhancer-ggml).
        backend = tts_cpp::detail::init_cpu_backend();
        device  = BackendDevice::CPU;
    }

    if (backend) {
        EnhancerGgml * graph = enhancer_ggml_create(e->impl_->weights, backend);
        if (graph) {
            e->impl_->backend      = backend;
            e->impl_->graph        = graph;
            e->impl_->device       = device;
            const char * bn        = ggml_backend_name(backend);
            e->impl_->backend_name = bn ? bn : (device == BackendDevice::GPU ? "GPU" : "CPU");
        } else {
            // Graph / weight setup failed — drop the backend and run the scalar
            // core directly.  Surface it under verbose so a host that asked for a
            // GPU can tell why backend_device() ends up CPU instead of the
            // fallback happening silently.
            if (opts.verbose) {
                const char * bn = ggml_backend_name(backend);
                std::fprintf(stderr,
                             "lavasr-enhancer: ggml graph creation failed on the %s "
                             "backend; falling back to the scalar CPU core\n",
                             bn ? bn : "?");
            }
            ggml_backend_free(backend);
        }
    }

    return e;
}

std::vector<float> Enhancer::enhance(const std::vector<float> & pcm_in,
                                     int sr_in) const {
    if (impl_->graph) {
        // The reusable graph allocator is mutated per call, so serialise
        // concurrent enhance() calls on this instance (CPU and GPU alike).
        std::lock_guard<std::mutex> lock(impl_->graph_mu);
        return tts_cpp::lavasr::enhance(impl_->weights, impl_->graph, pcm_in, sr_in);
    }
    return tts_cpp::lavasr::enhance(impl_->weights, pcm_in, sr_in);
}

int Enhancer::output_sample_rate() const {
    return enhancer_work_sample_rate(impl_->weights);
}

std::string Enhancer::backend_name() const {
    return impl_->backend_name;
}

BackendDevice Enhancer::backend_device() const {
    return impl_->device;
}

} // namespace tts_cpp::lavasr
