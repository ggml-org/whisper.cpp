#include "audiogen-cpp/acestep/vae.h"

#include "vae_ggml.h"              // internal: VaeModel + vae_model_*

#include "acestep/backend_registry.h"

#include "ggml-backend.h"

#include <cstdio>
#include <stdexcept>
#include <thread>

namespace tts_cpp::acestep {

struct Vae::Impl {
    ggml_backend_t backend = nullptr;  // owned
    VaeModel *     model   = nullptr;
    std::string    backend_name = "CPU";

    ~Impl() {
        if (model) vae_model_free(model);
        if (backend) ggml_backend_free(backend);
    }
};

static void vae_set_cpu_threads(ggml_backend_t backend, int n_threads) {
    int nthreads = n_threads > 0 ? n_threads : (int) std::thread::hardware_concurrency();
    if (nthreads <= 0) nthreads = 4;
    backend_set_n_threads(backend, nthreads);
}

Vae::Vae() : impl_(std::make_unique<Impl>()) {}
Vae::~Vae() = default;

std::unique_ptr<Vae> Vae::load(const std::string & gguf_path, const VaeOptions & opts) {
    std::unique_ptr<Vae> v(new Vae());

    // The two custom ops (col2im_1d, snake) now have both CPU and Metal kernels
    // in the ggml-speech fork, so the decode/encode graph can run on a GPU
    // backend. n_gpu_layers > 0 opts in (Metal on Apple, CUDA/Vulkan elsewhere);
    // falls back to CPU when no GPU backend is registered/available. Backends are
    // acquired through the ggml registry (see backend_registry.h) so the CPU path
    // resolves on arm64 dlopen (GGML_CPU_ALL_VARIANTS) builds too. When Vae::load
    // runs standalone (not via Engine), backends_dir loads the modules first.
    load_backends(opts.backends_dir);

    ggml_backend_t backend = nullptr;
    if (opts.n_gpu_layers > 0) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!backend && opts.verbose) {
            fprintf(stderr, "[acestep-vae] GPU requested but no GPU backend available; using CPU\n");
        }
    }
    if (!backend) {
        backend = backend_cpu_init();
        if (!backend) throw std::runtime_error("acestep-vae: failed to init CPU backend");
        vae_set_cpu_threads(backend, opts.n_threads);
    }

    VaeModel * model = vae_model_load(gguf_path, backend, opts.with_encoder, opts.verbose);
    if (!model) {
        ggml_backend_free(backend);
        throw std::runtime_error("acestep-vae: failed to load VAE GGUF: " + gguf_path);
    }

    v->impl_->backend = backend;
    v->impl_->model   = model;
    const char * bn   = ggml_backend_name(backend);
    v->impl_->backend_name = bn ? bn : "CPU";
    return v;
}

std::vector<float> Vae::decode(const std::vector<float> & latent, int T_latent,
                               const ProgressCb & on_progress) const {
    std::vector<float> pcm;
    if (T_latent <= 0 || (int) latent.size() < T_latent * 64) return {};
    int T_audio = vae_model_decode(impl_->model, latent.data(), T_latent, pcm, on_progress);
    if (T_audio < 0) return {};
    return pcm;
}

std::vector<float> Vae::encode(const std::vector<float> & pcm_interleaved, int frames, int * T_latent_out) const {
    if (T_latent_out) *T_latent_out = 0;
    if (frames <= 0 || (int) pcm_interleaved.size() < frames * 2) return {};
    std::vector<float> latent;
    int T_lat = vae_model_encode(impl_->model, pcm_interleaved.data(), frames, latent);
    if (T_lat < 0) return {};
    if (T_latent_out) *T_latent_out = T_lat;
    return latent;
}

bool        Vae::has_encoder() const   { return vae_model_has_encoder(impl_->model); }
int         Vae::sample_rate() const   { return 48000; }
int         Vae::upsample_factor() const { return 1920; }
std::string Vae::backend_name() const  { return impl_->backend_name; }

} // namespace tts_cpp::acestep
