#include "audiogen-cpp/acestep/vae.h"

#include "vae_ggml.h"              // internal: VaeModel + vae_model_*
#include "vae_encode_windows.h"

#include "acestep/backend_registry.h"
#include "acestep/engine_backends.h"

#ifdef AUDIOGEN_USE_COREML
#include "coreml/vae-decoder.h"
#include "vae_coreml_path.h"
#include "vae_coreml_windows.h"
#endif

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace tts_cpp::acestep {

struct Vae::Impl {
    ggml_backend_t backend = nullptr;  // owned
    VaeModel *     model   = nullptr;
    std::string    backend_name = "CPU";
#ifdef AUDIOGEN_USE_COREML
    acestep_coreml_vae_context * coreml = nullptr;  // owned; nullptr = ggml decode
#endif

    ~Impl() {
        if (model) vae_model_free(model);
        if (backend) ggml_backend_free(backend);
#ifdef AUDIOGEN_USE_COREML
        if (coreml) acestep_coreml_vae_free(coreml);
#endif
    }
};

#ifdef AUDIOGEN_USE_COREML
enum class CoremlDecodeStatus { done, unavailable, cancelled };

// ACESTEP_COREML_STRICT turns the silent ggml fallback into a decode failure,
// so a parity or benchmark run cannot accidentally measure ggml and report it
// as Core ML. A production decode never sets it.
static bool coreml_strict() { return std::getenv("ACESTEP_COREML_STRICT") != nullptr; }

static acestep_coreml_vae_context * load_coreml_sidecar(const std::string & gguf_path, bool verbose) {
    if (std::getenv("ACESTEP_COREML_DISABLE")) return nullptr;
    const std::string path = coreml_vae_sidecar_path(gguf_path);
    acestep_coreml_vae_context * ctx = acestep_coreml_vae_init(path.c_str());
    if (verbose) {
        if (ctx) {
            fprintf(stderr, "[acestep-vae] Core ML decoder sidecar loaded (%s, window %lld frames, %s)\n",
                    path.c_str(), (long long) acestep_coreml_vae_window_frames(ctx),
                    acestep_coreml_vae_backend_label(ctx));
        } else {
            fprintf(stderr, "[acestep-vae] no Core ML decoder sidecar at %s; using ggml\n", path.c_str());
        }
    }
    return ctx;
}

static void copy_window_core(const VaeCoremlWindow & w, const std::vector<float> & pcm_win,
                             std::vector<float> & pcm_out) {
    const size_t samples_per_frame = (size_t) VAE_ENCODER_UPSAMPLE * VAE_PCM_CHANNELS;
    const size_t core_off = (size_t) (w.core_a - w.win_a) * samples_per_frame;
    const size_t core_len = (size_t) (w.core_b - w.core_a) * samples_per_frame;
    const size_t dst      = (size_t) w.core_a * samples_per_frame;
    std::copy(pcm_win.begin() + core_off, pcm_win.begin() + core_off + core_len, pcm_out.begin() + dst);
}

static CoremlDecodeStatus coreml_decode(acestep_coreml_vae_context * ctx, const float * latent, int T_latent,
                                        std::vector<float> & pcm_out, const Vae::ProgressCb & on_progress) {
    const int window_frames = (int) acestep_coreml_vae_window_frames(ctx);
    const std::vector<VaeCoremlWindow> plan =
        vae_coreml_plan_windows(T_latent, window_frames, VAE_COREML_OVERLAP);
    if (plan.empty()) return CoremlDecodeStatus::unavailable;

    pcm_out.assign((size_t) T_latent * VAE_ENCODER_UPSAMPLE * VAE_PCM_CHANNELS, 0.0f);
    std::vector<float> pcm_win((size_t) window_frames * VAE_ENCODER_UPSAMPLE * VAE_PCM_CHANNELS);
    const int n_windows = (int) plan.size();
    for (int i = 0; i < n_windows; ++i) {
        const VaeCoremlWindow & w = plan[i];
        if (acestep_coreml_vae_decode(ctx, latent + (size_t) w.win_a * VAE_LATENT_CHANNELS, pcm_win.data()) != 0) {
            pcm_out.clear();
            return CoremlDecodeStatus::unavailable;
        }
        copy_window_core(w, pcm_win, pcm_out);
        if (on_progress && !on_progress(i + 1, n_windows)) {
            pcm_out.clear();
            return CoremlDecodeStatus::cancelled;
        }
    }
    return CoremlDecodeStatus::done;
}
#endif

Vae::Vae() : impl_(std::make_unique<Impl>()) {}
Vae::~Vae() = default;

std::unique_ptr<Vae> Vae::load(const std::string & gguf_path, const VaeOptions & opts) {
    std::unique_ptr<Vae> v(new Vae());

    // The two custom ops (col2im_1d, snake) have CPU, Metal and Vulkan kernels
    // in the ggml-speech fork, so the decode/encode graph can run on a GPU
    // backend. n_gpu_layers > 0 opts in (Metal on Apple, Vulkan elsewhere);
    // falls back to CPU when no GPU backend is registered/available. Backends are
    // acquired through the ggml registry (see backend_registry.h) so the CPU path
    // resolves on arm64 dlopen (GGML_CPU_ALL_VARIANTS) builds too. When Vae::load
    // runs standalone (not via Engine), backends_dir loads the modules first.
    load_backends(opts.backends_dir);

    // Resolution shared with the engine and the memory-fit projection
    // (engine_backends.h), so all three agree by construction.
    BackendRequest request;
    request.n_gpu_layers   = opts.n_gpu_layers;
    request.backend        = opts.backend;
    request.n_threads      = opts.n_threads;
    request.verbose        = opts.verbose;
    ggml_backend_t backend = resolve_vae_backend(request);
    if (!backend) throw std::runtime_error("acestep-vae: failed to init backend '" + opts.backend + "'");
    v->impl_->backend = backend;

    VaeModel * model = vae_model_load(gguf_path, backend, opts.with_encoder, opts.verbose);
    if (!model) throw std::runtime_error("acestep-vae: failed to load VAE GGUF: " + gguf_path);
    v->impl_->model = model;
    const char * bn   = ggml_backend_name(backend);
    v->impl_->backend_name = bn ? bn : "CPU";
#ifdef AUDIOGEN_USE_COREML
    v->impl_->coreml = load_coreml_sidecar(gguf_path, opts.verbose);
#endif
    return v;
}

std::vector<float> Vae::decode(const std::vector<float> & latent, int T_latent,
                               const ProgressCb & on_progress) const {
    std::vector<float> pcm;
    if (T_latent <= 0 || (int) latent.size() < T_latent * 64) return {};
#ifdef AUDIOGEN_USE_COREML
    if (impl_->coreml) {
        switch (coreml_decode(impl_->coreml, latent.data(), T_latent, pcm, on_progress)) {
            case CoremlDecodeStatus::done:      return pcm;
            case CoremlDecodeStatus::cancelled: return {};
            case CoremlDecodeStatus::unavailable:
                if (coreml_strict()) {
                    fprintf(stderr, "[acestep-vae] Core ML decode unavailable for T_latent=%d and "
                                    "ACESTEP_COREML_STRICT is set; failing instead of ggml fallback\n", T_latent);
                    return {};
                }
                if (std::getenv("AUDIOGEN_VERBOSE"))
                    fprintf(stderr, "[acestep-vae] Core ML decode unavailable for T_latent=%d; using ggml\n", T_latent);
                break;
        }
    } else if (coreml_strict()) {
        fprintf(stderr, "[acestep-vae] no Core ML sidecar loaded and ACESTEP_COREML_STRICT is set; "
                        "failing instead of ggml fallback\n");
        return {};
    }
#endif
    int T_audio = vae_model_decode(impl_->model, latent.data(), T_latent, pcm, on_progress);
    if (T_audio < 0) return {};
    return pcm;
}

std::vector<float> Vae::encode(const std::vector<float> & pcm_interleaved, int frames,
                               int * T_latent_out,
                               const ProgressCb & on_progress) const {
    const int window_count = vae_encode_window_count(frames);
    int window_index = 0;
    const VaeWindowEncoder encode = [this, &on_progress, &window_index, window_count](
                                        const float * pcm, int window_frames,
                                        std::vector<float> & latent) {
        ProgressCb window_progress;
        if (on_progress) {
            window_progress = [&on_progress, &window_index, window_count](
                                  int done, int total) {
                return on_progress(window_index * total + done,
                                   window_count * total);
            };
        }
        const int encoded = vae_model_encode(
            impl_->model, pcm, window_frames, latent, window_progress);
        ++window_index;
        return encoded;
    };
    return encode_vae_pcm_bounded(pcm_interleaved, frames, encode, T_latent_out);
}

bool        Vae::has_encoder() const   { return vae_model_has_encoder(impl_->model); }
int         Vae::sample_rate() const   { return 48000; }
int         Vae::upsample_factor() const { return 1920; }
std::string Vae::backend_name() const  { return impl_->backend_name; }

} // namespace tts_cpp::acestep
