#include "audiogen-cpp/minimax/engine.h"

#include "minimax/backend.h"
#include "minimax/coreml-options.h"
#include "minimax/logic.h"
#include "minimax/mm3-pipeline.h"
#include "minimax/model-files.h"
#include "minimax/request-utils.h"

#include <atomic>
#include <cmath>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>

namespace tts_cpp::minimax {
namespace {

std::recursive_mutex & engine_mutex() {
    static std::recursive_mutex mutex;
    return mutex;
}

int & active_engine_count() {
    static int count = 0;
    return count;
}

void require_coreml_support(const EngineOptions & options) {
    detail::validate_coreml_options(options);
    if (!detail::coreml_requested(options)) {
        return;
    }
#ifdef AUDIOGEN_USE_MINIMAX_COREML
    if (mm3_ane_validation::available()) {
        return;
    }
    throw std::runtime_error("MiniMax full Core ML requires macOS 15 or later");
#else
    throw std::runtime_error("MiniMax full Core ML support was not built; enable "
                             "AUDIOGEN_MINIMAX_COREML");
#endif
}

class CoreMLCancellationScope {
  public:
    CoreMLCancellationScope(const MM3Model & model, std::atomic<bool> & cancelled)
#ifdef AUDIOGEN_USE_MINIMAX_COREML
        : runtime_(model.ane_validation) {
        if (runtime_) {
            mm3_ane_validation::set_cancel_handler(*runtime_, [&cancelled] { return cancelled.load(); });
        }
#else
    {
        (void)model;
        (void)cancelled;
#endif
    }

    ~CoreMLCancellationScope() {
#ifdef AUDIOGEN_USE_MINIMAX_COREML
        if (runtime_) {
            mm3_ane_validation::set_cancel_handler(*runtime_, {});
        }
#endif
    }

  private:
#ifdef AUDIOGEN_USE_MINIMAX_COREML
    std::shared_ptr<mm3_ane_validation::Runtime> runtime_;
#endif
};

int resolve_thread_count(int requested) {
    if (requested > 0) {
        return requested;
    }
    const unsigned int available = std::thread::hardware_concurrency();
    return available > 0 ? static_cast<int>(available) : 1;
}

uint64_t resolve_seed(int64_t seed) {
    if (seed >= 0) {
        return static_cast<uint64_t>(seed);
    }
    std::random_device random;
    return (static_cast<uint64_t>(random()) << 32) ^ static_cast<uint64_t>(random());
}

void interleave_planar_stereo(const std::vector<float> & planar, int64_t samples, std::vector<float> & interleaved) {
    interleaved.resize(static_cast<size_t>(samples) * 2);
    for (int64_t sample = 0; sample < samples; ++sample) {
        interleaved[static_cast<size_t>(sample) * 2] = planar[static_cast<size_t>(sample)];
        interleaved[static_cast<size_t>(sample) * 2 + 1] =
            planar[static_cast<size_t>(samples + sample)];
    }
}

int64_t progress_current(const MM3GenProgress & progress) {
    if (progress.window < 0 || progress.n_windows <= 1) {
        return progress.step;
    }
    return progress.window * progress.n_steps + progress.step;
}

int64_t progress_total(const MM3GenProgress & progress) {
    if (progress.window < 0 || progress.n_windows <= 1) {
        return progress.n_steps;
    }
    return progress.n_windows * progress.n_steps;
}

void release_graphs() {
    mm3_lm_free(&g_mm3_lm);
    mm3_depth_free(&g_mm3_depth);
    mm3_cond_free(&g_mm3_cond);
    mm3_dit_free(&g_mm3_dit);
    mm3_vocoder_free(&g_mm3_voc);
}

bool abort_when_cancelled(void * user_data) {
    return static_cast<std::atomic<bool> *>(user_data)->load();
}

class AbortScope {
public:
    AbortScope(ggml_backend_t backend, ggml_backend_t cpu_backend, std::atomic<bool> & cancelled)
        : backend_(backend), cpu_backend_(cpu_backend) {
        backend_set_abort_handler(cpu_backend_, abort_when_cancelled, &cancelled);
        if (backend_ != cpu_backend_) {
            backend_set_abort_handler(backend_, abort_when_cancelled, &cancelled);
        }
    }

    ~AbortScope() {
        backend_set_abort_handler(cpu_backend_, nullptr, nullptr);
        if (backend_ != cpu_backend_) {
            backend_set_abort_handler(backend_, nullptr, nullptr);
        }
    }

private:
    ggml_backend_t backend_;
    ggml_backend_t cpu_backend_;
};

class GenerationScope {
public:
    explicit GenerationScope(bool & generating) : generating_(generating) {
        if (generating_) {
            throw std::logic_error("minimax engine: recursive generate() is not allowed");
        }
        generating_ = true;
    }

    ~GenerationScope() {
        generating_ = false;
    }

private:
    bool & generating_;
};

// A cancellation is consumed by the generation that observes it, never erased
// at startup: cancel() arriving between a caller's own precheck and generate()
// entry must cancel that run instead of stalling until the first progress
// callback re-arms the flag.
class CancellationScope {
public:
    explicit CancellationScope(std::atomic<bool> & cancelled) : cancelled_(cancelled) {}

    ~CancellationScope() {
        cancelled_.store(false);
    }

private:
    std::atomic<bool> & cancelled_;
};

}

struct Engine::Impl {
    EngineOptions options;
    MM3Model model;
    mutable MM3Tokenizer tokenizer;
    mutable std::atomic<bool> cancelled{false};
    mutable bool generating = false;
    bool registered = false;

    ~Impl() {
        std::lock_guard<std::recursive_mutex> lock(engine_mutex());
        release_graphs();
        mm3_unload(&model);
        if (registered) {
            --active_engine_count();
        }
    }
};

Engine::Engine() : impl_(std::make_unique<Impl>()) {}
Engine::~Engine() = default;

std::unique_ptr<Engine> Engine::create(const EngineOptions & input) {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex());
    require_coreml_support(input);
    if (!detail::engine_instance_available(active_engine_count())) {
        throw std::runtime_error("minimax engine: only one active engine instance is supported");
    }
    std::unique_ptr<Engine> engine(new Engine());
    engine->impl_->options = detail::resolve_model_paths(input);
    engine->impl_->options.n_threads = resolve_thread_count(input.n_threads);
    backend_configure_cpu(engine->impl_->options.n_threads, input.backends_dir);
    backend_configure_device(input.device);
    detail::probe_model_files(engine->impl_->model, engine->impl_->options);
    engine->impl_->model.coreml_conditioner_path = input.coreml_conditioner_path;
#ifdef AUDIOGEN_USE_MINIMAX_COREML
    engine->impl_->model.ane_validation_directory = input.coreml_profile_dir;
    engine->impl_->model.ane_validation_compute_units = input.coreml_compute_units;
#endif
    std::string error;
    if (!mm3_load(&engine->impl_->model, &error)) {
        throw std::runtime_error("minimax engine: " + error);
    }
    engine->impl_->registered = true;
    ++active_engine_count();
    return engine;
}

GenerateResult Engine::generate(const GenerateParams & params, const ProgressFn & progress) const {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex());
    GenerationScope generation_scope(impl_->generating);
    CancellationScope cancellation_scope(impl_->cancelled);
    CoreMLCancellationScope coreml_cancellation_scope(impl_->model, impl_->cancelled);
    AbortScope abort_scope(impl_->model.backend, impl_->model.cpu_backend, impl_->cancelled);

    const int model_max_frames = static_cast<int>(impl_->model.lm_cfg.max_audio_frames);
    const int64_t max_frames = detail::validate_frames(params.max_frames, model_max_frames);
    const int steps = detail::resolve_flow_steps(params.inference_steps);
    if (steps > 1000) {
        throw std::invalid_argument("inference steps must be in 1..1000");
    }
    const float cfg_scale =
        params.cfg_scale > 0.0f ? params.cfg_scale : impl_->model.synth_cfg.flow.cfg_scale;
    if (!std::isfinite(cfg_scale) || cfg_scale <= 0.0f) {
        throw std::invalid_argument("CFG scale must be finite and greater than zero");
    }

    if (impl_->cancelled.load()) {
        return {};
    }

    MM3GenRequest request;
    request.prompt = detail::build_prompt(params.caption, params.lyrics);
    request.max_frames = max_frames;
    request.seed = resolve_seed(params.seed);
    request.steps = steps;
    request.cfg_flow = cfg_scale;
    request.should_cancel = [this] { return impl_->cancelled.load(); };

    MM3ProgressCb callback;
    if (progress) {
        callback = [this, &progress](const MM3GenProgress & state) {
            if (!progress(state.stage, progress_current(state), progress_total(state))) {
                impl_->cancelled.store(true);
            }
        };
    }

    MM3GenResult generated;
    std::string error;
    bool success = false;
    try {
        success = mm3_generate(impl_->model, request, &impl_->tokenizer, callback, &generated, &error);
    } catch (...) {
        if (impl_->cancelled.load()) {
            return {};
        }
        throw;
    }
    if (!success) {
        if (error == MM3_ERR_CANCELLED || impl_->cancelled.load()) {
            return {};
        }
        throw std::runtime_error("minimax engine: " + error);
    }

    GenerateResult result;
    interleave_planar_stereo(generated.audio, generated.n_samples, result.pcm);
    result.sample_rate = generated.sample_rate;
    result.channels = 2;
    result.emitted_frames = generated.frames;
    result.ar_ms = generated.ar_ms;
    result.condition_ms = generated.cond_ms;
    result.flow_ms = generated.flow_ms;
    result.vocoder_ms = generated.voc_ms;
    result.total_ms = generated.total_ms;
    return result;
}

void Engine::cancel() const {
    impl_->cancelled.store(true);
}

int Engine::sample_rate() const {
    return static_cast<int>(impl_->model.synth_cfg.voc.sampling_rate);
}

std::string Engine::backend_name() const {
#ifdef AUDIOGEN_USE_MINIMAX_COREML
    if (impl_->model.ane_validation) {
        return "CoreML/" + impl_->options.coreml_compute_units;
    }
#endif
    const ggml_backend_t backend = impl_->model.backend;
    if (!backend || backend == impl_->model.cpu_backend) {
        return "CPU";
    }
    const char * name = tts_cpp::acestep::backend_reg_name(backend);
    return name && *name ? name : "CPU";
}

GpuFallbackReason Engine::gpu_fallback_reason() const {
#ifdef AUDIOGEN_USE_MINIMAX_COREML
    if (impl_->model.ane_validation) {
        return GpuFallbackReason::not_requested;
    }
#endif
    return g_backend_gpu_fallback_reason;
}

}
