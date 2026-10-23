#include "dac_coreml.h"

#include "internal.h"
#include "../coreml_sidecar_path.h"

#ifdef TTS_CPP_USE_COREML
#include "../moss/coreml/runner.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace tts_cpp {
namespace parler {
namespace detail {
namespace {

constexpr int SINGLE_BATCH = 1;
constexpr int MONO = 1;
constexpr int FEATURE_RANK = 3;

bool read_f32(const ggml_tensor * t, int64_t n, std::vector<float> & out) {
    if (t == nullptr || t->type != GGML_TYPE_F32 || ggml_nelements(t) != n) return false;
    out.resize((size_t) n);
    ggml_backend_tensor_get(t, out.data(), 0, (size_t) n * sizeof(float));
    return true;
}

#ifdef TTS_CPP_USE_COREML
class coreml_dac final : public parler_dac_sidecar {
public:
    coreml_dac(moss_coreml_model * model, int window) : model_(model), window_(window) {}
    ~coreml_dac() override { moss_coreml_free(model_); }
    coreml_dac(const coreml_dac &) = delete;
    coreml_dac & operator=(const coreml_dac &) = delete;

    int window() const override { return window_; }
    const char * label() const override { return moss_coreml_label(model_); }

    bool decode(const float * latent, const float * valid, float * pcm) override {
        const char * in_names[] = {"latent", "valid"};
        const float * in_data[] = {latent, valid};
        const char * out_names[] = {"pcm"};
        float * out_data[] = {pcm};
        return moss_coreml_predict(model_, in_names, in_data, 2, out_names, out_data, 1) == 0;
    }

private:
    moss_coreml_model * model_;
    int window_;
};

std::vector<int64_t> feature_dims(moss_coreml_model * model, const char * name) {
    int64_t dims[8] = {};
    const int rank = moss_coreml_feature_dims(model, name, dims, 8);
    return std::vector<int64_t>(dims, dims + std::max(rank, 0));
}
#endif

} // namespace

std::string parler_dac_sidecar_path(const std::string & gguf_path) {
    return ::tts_cpp::detail::coreml_sidecar_path(gguf_path, PARLER_DAC_SIDECAR_SUFFIX);
}

bool parler_coreml_disabled() {
    return std::getenv(PARLER_COREML_DISABLE_ENV) != nullptr;
}

bool parler_coreml_strict() {
    return std::getenv(PARLER_COREML_STRICT_ENV) != nullptr;
}

bool parler_dac_sidecar_enabled(const std::string & gguf_path) {
#ifdef TTS_CPP_USE_COREML
    return !parler_coreml_disabled() &&
           ::tts_cpp::detail::coreml_sidecar_exists(parler_dac_sidecar_path(gguf_path));
#else
    (void) gguf_path;
    return false;
#endif
}

void parler_dac_rvq::latent(const int32_t * codes, int n_frames, int f0, int f1,
                            float * out, int stride) const {
    const int n = f1 - f0;
    for (int c = 0; c < latent_dim; ++c) {
        float bsum = 0.0f;
        for (int k = 0; k < n_q; ++k) bsum += bias[(size_t) k][(size_t) c];
        std::fill(out + (size_t) c * stride, out + (size_t) c * stride + n, bsum);
    }
    std::vector<float> vec((size_t) cb_dim);
    for (int k = 0; k < n_q; ++k) {
        const float * cb = codebook[(size_t) k].data();
        const float * w  = proj[(size_t) k].data();
        for (int t = 0; t < n; ++t) {
            const int32_t code = codes[(size_t) k * n_frames + f0 + t];
            std::memcpy(vec.data(), cb + (size_t) code * cb_dim, (size_t) cb_dim * sizeof(float));
            for (int c = 0; c < latent_dim; ++c) {
                const float * wr = w + (size_t) c * cb_dim;
                float acc = 0.0f;
                for (int i = 0; i < cb_dim; ++i) acc += wr[i] * vec[(size_t) i];
                out[(size_t) c * stride + t] += acc;
            }
        }
    }
}

bool parler_dac_rvq_load(const parler_model & model, parler_dac_rvq & rvq, std::string * err) {
    const parler_hparams & hp = model.hparams;
    parler_dac_rvq r;
    r.n_q           = hp.dac_n_q;
    r.codebook_size = hp.dac_codebook_size;
    r.latent_dim    = hp.dac_latent;
    if ((int) model.dac_quant.size() != r.n_q || r.n_q <= 0 || model.dac_quant[0].codebook == nullptr) {
        if (err) *err = "DAC quantizer tensors missing";
        return false;
    }
    r.cb_dim = (int) model.dac_quant[0].codebook->ne[0];
    r.codebook.resize((size_t) r.n_q);
    r.proj.resize((size_t) r.n_q);
    r.bias.resize((size_t) r.n_q);
    for (int k = 0; k < r.n_q; ++k) {
        const parler_dac_quant & q = model.dac_quant[(size_t) k];
        if (!read_f32(q.codebook, (int64_t) r.codebook_size * r.cb_dim, r.codebook[(size_t) k]) ||
            !read_f32(q.out_proj_w, (int64_t) r.latent_dim * r.cb_dim, r.proj[(size_t) k]) ||
            !read_f32(q.out_proj_b, r.latent_dim, r.bias[(size_t) k])) {
            if (err) *err = "DAC quantizer " + std::to_string(k) + " is not F32 of the expected shape";
            return false;
        }
    }
    rvq = std::move(r);
    return true;
}

std::shared_ptr<parler_dac_sidecar> parler_open_dac_sidecar(const std::string & gguf_path,
                                                            const parler_model & model,
                                                            std::string * why) {
    return parler_open_dac_sidecar(gguf_path, model, parler_resolve_coreml_policy(), why);
}

std::shared_ptr<parler_dac_sidecar> parler_open_dac_sidecar(const std::string & gguf_path,
                                                            const parler_model & model,
                                                            const parler_coreml_policy & policy,
                                                            std::string * why) {
    auto reject = [&](const std::string & reason) -> std::shared_ptr<parler_dac_sidecar> {
        if (why) *why = reason;
        return nullptr;
    };
    if (policy.disabled) return reject(std::string(PARLER_COREML_DISABLE_ENV) + " is set");
#ifdef TTS_CPP_USE_COREML
    const std::string path = parler_dac_sidecar_path(gguf_path);
    if (!::tts_cpp::detail::coreml_sidecar_exists(path)) return reject("no sidecar at " + path);
    moss_coreml_model * m = moss_coreml_load(path.c_str(), policy.compute_units.c_str());
    if (m == nullptr) return reject("Core ML could not load " + path);

    const parler_hparams & hp = model.hparams;
    const std::vector<int64_t> lat = feature_dims(m, "latent");
    const int64_t window = lat.size() == FEATURE_RANK ? lat[2] : 0;
    const bool ok = lat.size() == FEATURE_RANK && lat[0] == SINGLE_BATCH && lat[1] == hp.dac_latent &&
                    feature_dims(m, "valid") == std::vector<int64_t>{SINGLE_BATCH, MONO, window} &&
                    feature_dims(m, "pcm") == std::vector<int64_t>{SINGLE_BATCH, MONO, window * hp.dac_hop};
    if (!ok) {
        moss_coreml_free(m);
        return reject(path + " does not declare latent [1, " + std::to_string(hp.dac_latent) +
                      ", W], valid [1, 1, W] and pcm [1, 1, W * " + std::to_string(hp.dac_hop) + "]");
    }
    return std::make_shared<coreml_dac>(m, (int) window);
#else
    (void) gguf_path;
    (void) model;
    return reject("built without TTS_CPP_COREML");
#endif
}

bool parler_attach_dac_sidecar(parler_model & model, std::shared_ptr<parler_dac_sidecar> sidecar,
                               std::string * err) {
    return parler_attach_dac_sidecar(model, std::move(sidecar), parler_resolve_coreml_policy(), err);
}

bool parler_attach_dac_sidecar(parler_model & model, std::shared_ptr<parler_dac_sidecar> sidecar,
                               const parler_coreml_policy & policy, std::string * err) {
    model.dac_sidecar.reset();
    model.dac_sidecar_retired = false;
    model.dac_coreml_strict = policy.strict;
    if (!sidecar || policy.disabled) return true;
    if (!parler_dac_rvq_load(model, model.dac_rvq, err)) return false;
    model.dac_sidecar = std::move(sidecar);
    return true;
}

bool parler_attach_dac_sidecar(parler_model & model, const std::string & gguf_path) {
    return parler_attach_dac_sidecar(model, gguf_path, parler_resolve_coreml_policy());
}

bool parler_attach_dac_sidecar(parler_model & model, const std::string & gguf_path,
                               const parler_coreml_policy & policy) {
    // Reset retirement even if the new sidecar cannot be opened. Strictness
    // belongs to this attachment, never to the environment at decode time.
    parler_attach_dac_sidecar(model, std::shared_ptr<parler_dac_sidecar>{}, policy);
    if (policy.disabled) {
        if (policy.strict) {
            throw std::runtime_error("parler: Core ML is disabled but strict policy is enabled");
        }
        return false;
    }
    std::string why;
    std::shared_ptr<parler_dac_sidecar> sidecar = parler_open_dac_sidecar(gguf_path, model, policy, &why);
    if (sidecar && sidecar->window() <= 2 * parler_dac_rf_frames(model)) {
        why = "sidecar window " + std::to_string(sidecar->window()) + " leaves no frames after " +
              std::to_string(parler_dac_rf_frames(model)) + " context frames per side";
        sidecar.reset();
    }
    if (sidecar && !parler_attach_dac_sidecar(model, sidecar, policy, &why)) sidecar.reset();
    if (sidecar) {
        fprintf(stderr, "parler: DAC on Core ML (%s, %d-frame window)\n", sidecar->label(),
                sidecar->window());
        return true;
    }
    if (policy.strict) {
        throw std::runtime_error(std::string("parler: no usable Core ML DAC sidecar (") + why +
                                 "); Core ML strict policy is enabled (coreml_strict / " +
                                 PARLER_COREML_STRICT_ENV + ")");
    }
    return false;
}

bool parler_dac_decode_sidecar(const parler_model & model, const int32_t * codes, int n_frames,
                               std::vector<float> & pcm_out, std::vector<float> * latent_out,
                               int out_begin, int out_end) {
    parler_dac_sidecar & sidecar = *model.dac_sidecar;
    const parler_dac_rvq & rvq = model.dac_rvq;
    const int hop = model.hparams.dac_hop;
    const int window = sidecar.window();
    const int rf = parler_dac_rf_frames(model);
    const int keep = window - 2 * rf;
    if (keep <= 0 || rvq.latent_dim != model.hparams.dac_latent) return false;

    std::vector<float> latent((size_t) rvq.latent_dim * window);
    std::vector<float> valid((size_t) window);
    std::vector<float> pcm((size_t) window * hop);
    for (int a = out_begin; a < out_end; a += keep) {
        const int b  = std::min(a + keep, out_end);
        const int w0 = std::max(0, a - rf);
        const int w1 = std::min(n_frames, w0 + window);
        const int n  = w1 - w0;
        // the unused tail is masked out, so its contents never reach the kept span
        std::fill(latent.begin(), latent.end(), 0.0f);
        rvq.latent(codes, n_frames, w0, w1, latent.data(), window);
        std::fill(valid.begin(), valid.begin() + n, 1.0f);
        std::fill(valid.begin() + n, valid.end(), 0.0f);
        if (!sidecar.decode(latent.data(), valid.data(), pcm.data())) return false;

        const size_t off = (size_t) (a - w0) * hop;
        pcm_out.insert(pcm_out.end(), pcm.begin() + (ptrdiff_t) off,
                       pcm.begin() + (ptrdiff_t) (off + (size_t) (b - a) * hop));
        if (latent_out) {
            for (int t = a; t < b; ++t) {
                for (int c = 0; c < rvq.latent_dim; ++c) {
                    latent_out->push_back(latent[(size_t) c * window + (t - w0)]);
                }
            }
        }
    }
    return true;
}

} // namespace detail
} // namespace parler
} // namespace tts_cpp
