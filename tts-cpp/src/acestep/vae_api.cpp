#include "tts-cpp/acestep/vae.h"

#include "../backend_selection.h"  // init_cpu_backend
#include "vae_ggml.h"              // internal: VaeModel + vae_model_*

#include "ggml-backend.h"
#include "ggml-cpu.h"

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

Vae::Vae() : impl_(std::make_unique<Impl>()) {}
Vae::~Vae() = default;

std::unique_ptr<Vae> Vae::load(const std::string & gguf_path, const VaeOptions & opts) {
    std::unique_ptr<Vae> v(new Vae());

    // CPU-only for now: the two custom ops (col2im_1d, snake) have CPU kernels
    // only in the ggml-speech fork. Prefer the registry CPU backend, fall back
    // to the direct ggml-cpu init.
    ggml_backend_t backend = tts_cpp::detail::init_cpu_backend();
    if (!backend) backend = ggml_backend_cpu_init();
    if (!backend) throw std::runtime_error("acestep-vae: failed to init CPU backend");

    int nthreads = opts.n_threads > 0 ? opts.n_threads : (int) std::thread::hardware_concurrency();
    if (nthreads <= 0) nthreads = 4;
    ggml_backend_cpu_set_n_threads(backend, nthreads);

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

std::vector<float> Vae::decode(const std::vector<float> & latent, int T_latent) const {
    std::vector<float> pcm;
    if (T_latent <= 0 || (int) latent.size() < T_latent * 64) return {};
    int T_audio = vae_model_decode(impl_->model, latent.data(), T_latent, pcm);
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
