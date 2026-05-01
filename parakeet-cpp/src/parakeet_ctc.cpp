// FastConformer encoder ggml graph, GGUF load, CTC head, and encoder execution.

#include "parakeet_ctc.h"
#include "parakeet_log.h"
#include "backend_util.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace parakeet {

struct EncoderGraph {
    ggml_context * graph_ctx = nullptr;
    ggml_cgraph  * cgraph    = nullptr;
    ggml_gallocr_t alloc     = nullptr;
    int            T_mel     = 0;
    int            n_run_layers = 0;
    bool           all_valid = false;

    std::vector<float> pe_host;
    std::vector<float> att_mask_host;   // (T_enc, T_enc) row-major; 0 for visible, -inf for masked

    // Subsampling time-pad masks. When `all_valid == true` these are
    // pre-built once at graph construction (every value is 1.0 over
    // the corresponding L_i). When `all_valid == false` they are
    // re-built per call from the actual `mel_valid` count and cached
    // in `mN_dynamic`; the cached buffers are reused across calls
    // with the same `(L_i, V_i)` layout to avoid the per-call
    // std::vector allocations. See `run_encoder` for the cache
    // invalidation logic.
    std::vector<float> m0_host;
    std::vector<float> m1_host;
    std::vector<float> m2_host;
    std::vector<float> m3_host;
    int                m0_v = -1;
    int                m1_v = -1;
    int                m2_v = -1;
    int                m3_v = -1;

    ggml_tensor * mel_in   = nullptr;
    ggml_tensor * mask_t0  = nullptr;
    ggml_tensor * mask_t1  = nullptr;
    ggml_tensor * mask_t2  = nullptr;
    ggml_tensor * mask_t3  = nullptr;
    ggml_tensor * pe_in    = nullptr;
    ggml_tensor * att_mask = nullptr;   // null when the encoder uses unrestricted attention

    ggml_tensor * sub_out_node         = nullptr;
    ggml_tensor * post_ff1_0_node      = nullptr;
    ggml_tensor * post_attn_0_node     = nullptr;
    ggml_tensor * post_conv_0_node     = nullptr;
    ggml_tensor * post_ff2_0_node      = nullptr;
    ggml_tensor * block_0_out_node     = nullptr;
    ggml_tensor * block_last_out_node  = nullptr;
    ggml_tensor * encoder_out_node     = nullptr;
    ggml_tensor * logits_node          = nullptr;

    void free_() {
        if (alloc)     { ggml_gallocr_free(alloc); alloc = nullptr; }
        if (graph_ctx) { ggml_free(graph_ctx);     graph_ctx = nullptr; }
        cgraph = nullptr;
        mel_in = mask_t0 = mask_t1 = mask_t2 = mask_t3 = pe_in = nullptr;
        sub_out_node = post_ff1_0_node = post_attn_0_node = nullptr;
        post_conv_0_node = post_ff2_0_node = block_0_out_node = nullptr;
        block_last_out_node = encoder_out_node = logits_node = nullptr;
        T_mel = 0;
        all_valid = false;
        pe_host.clear();
        att_mask_host.clear();
        m0_host.clear(); m1_host.clear(); m2_host.clear(); m3_host.clear();
        m0_v = m1_v = m2_v = m3_v = -1;
    }
};

struct ParakeetCtcModel::Impl {
    gguf_context         * gguf           = nullptr;
    ggml_context         * ctx            = nullptr;
    ggml_backend_t         backend_cpu    = nullptr;
    ggml_backend_t         backend_blas   = nullptr;
    ggml_backend_t         backend_gpu    = nullptr;
    ggml_backend_t         backend_active = nullptr;
    ggml_backend_buffer_t  weights_buffer = nullptr;
    std::vector<std::unique_ptr<EncoderGraph>> encoder_graphs;
    static constexpr size_t k_encoder_graph_cache_max = 3;

    ~Impl() {
        for (auto & g : encoder_graphs) {
            if (g) g->free_();
        }
        encoder_graphs.clear();
        if (weights_buffer) ggml_backend_buffer_free(weights_buffer);
        if (ctx)            ggml_free(ctx);
        if (gguf)           gguf_free(gguf);
        if (backend_blas)   ggml_backend_free(backend_blas);
        if (backend_gpu)    ggml_backend_free(backend_gpu);
        if (backend_cpu)    ggml_backend_free(backend_cpu);
    }
};

ggml_backend_t ParakeetCtcModel::backend_active() const {
    return impl ? impl->backend_active : nullptr;
}

ggml_context * ParakeetCtcModel::weights_ctx() const {
    return impl ? impl->ctx : nullptr;
}


namespace {

// Trigger one-time discovery + load of every available ggml backend.
// Idempotent: repeated calls inside the same process are no-ops once
// the registry is populated. Routed through a static guard so we don't
// pay the directory-walk cost on every model load.
//
// Why this instead of the per-backend ggml_backend_<x>_init() entry
// points the cascade used to call directly: with GGML_BACKEND_DL=ON
// (the dynamic-loader mode embedded host applications typically
// ship with) the CUDA / Metal / Vulkan / OpenCL / BLAS / ggml-cpu
// backends live in separate shared libraries that are dlopened at
// runtime; their concrete init symbols are not linkable from
// libparakeet, and the only supported entry point is the registry.
// With GGML_BACKEND_DL=OFF the backends are statically linked into
// libggml, registered at constructor time, and
// ggml_backend_load_all() is a cheap no-op. Both modes therefore
// reach the same registry walk below, matching the convention used
// by llama.cpp and other ggml-based libraries.
void ensure_backends_loaded() {
    static const bool loaded = []() {
        ggml_backend_load_all();
        return true;
    }();
    (void) loaded;
}

bool is_adreno_6xx(const char * s) {
    if (!s) return false;
    if (!strstr(s, "Adreno")) return false;
    for (const char * q = s; *q; ++q) {
        if (*q == '6' && q[1] >= '0' && q[1] <= '9' && q[2] >= '0' && q[2] <= '9') {
            return true;
        }
    }
    return false;
}

const char * dev_reg_name(ggml_backend_dev_t dev) {
    if (!dev) return "";
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    return reg ? ggml_backend_reg_name(reg) : "";
}


ggml_backend_t init_gpu_backend(int n_gpu_layers, bool verbose) {
    if (n_gpu_layers <= 0) return nullptr;

    ensure_backends_loaded();

    // Walk the registry in registration order and pick the first
    // GPU/IGPU device. Registry order is defined by the ggml-backend
    // registry's static init list (CUDA -> Metal -> Vulkan -> OpenCL
    // -> ...), so this preserves the priority of the legacy direct-
    // init cascade. The Adreno-6xx fallback policy stays on top:
    // ggml-opencl produces incorrect results on Adreno 6xx; force-
    // skip and continue the walk (or fall through to CPU) unless
    // `PARAKEET_ALLOW_ADRENO_6XX=1` is set.
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        const char * name     = ggml_backend_dev_name(dev);
        const char * desc     = ggml_backend_dev_description(dev);
        const char * reg_name = dev_reg_name(dev);
        const bool   is_opencl = std::strcmp(reg_name, "OpenCL") == 0;

        if (is_opencl && (is_adreno_6xx(name) || is_adreno_6xx(desc))) {
            const char * reported = name ? name : (desc ? desc : "unknown");
            const char * override_env = getenv("PARAKEET_ALLOW_ADRENO_6XX");
            if (!override_env || override_env[0] != '1') {
                if (verbose) PARAKEET_LOG_WARN(
                    "parakeet: OpenCL device '%s' is Adreno 6xx; "
                    "skipping (7xx/8xx/X1E supported, set "
                    "PARAKEET_ALLOW_ADRENO_6XX=1 to override)\n",
                    reported);
                continue;
            }
            if (verbose) PARAKEET_LOG_INFO(
                "parakeet: PARAKEET_ALLOW_ADRENO_6XX=1 set; "
                "keeping OpenCL backend on '%s' anyway\n", reported);
        }

        ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
        if (!b) continue;
        if (verbose) PARAKEET_LOG_INFO(
            "parakeet: using %s backend (%s)\n",
            reg_name && *reg_name ? reg_name : "GPU",
            name ? name : (desc ? desc : "unknown"));
        return b;
    }

    if (verbose) PARAKEET_LOG_INFO("parakeet: no GPU backend available, falling back to CPU\n");
    return nullptr;
}

ggml_backend_t init_cpu_backend() {
    ensure_backends_loaded();
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

ggml_backend_t init_blas_backend() {
    ensure_backends_loaded();
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_ACCEL) continue;
        const char * reg_name = dev_reg_name(dev);
        if (std::strcmp(reg_name, "BLAS") != 0) continue;
        return ggml_backend_dev_init(dev, nullptr);
    }
    return nullptr;
}

int find_key(const gguf_context * g, const std::string & k) {
    return (int) gguf_find_key(g, k.c_str());
}

uint32_t get_u32(const gguf_context * g, const std::string & k, uint32_t fallback) {
    const int id = find_key(g, k);
    if (id < 0) return fallback;
    return gguf_get_val_u32(g, id);
}

float get_f32(const gguf_context * g, const std::string & k, float fallback) {
    const int id = find_key(g, k);
    if (id < 0) return fallback;
    return gguf_get_val_f32(g, id);
}

bool get_bool(const gguf_context * g, const std::string & k, bool fallback) {
    const int id = find_key(g, k);
    if (id < 0) return fallback;
    return gguf_get_val_bool(g, id);
}

ggml_tensor * require_tensor(ggml_context * ctx, const std::string & name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t) throw std::runtime_error("gguf: missing required tensor '" + name + "'");
    return t;
}

ggml_tensor * maybe_tensor(ggml_context * ctx, const std::string & name) {
    return ggml_get_tensor(ctx, name.c_str());
}

std::string get_str(const gguf_context * g, const std::string & k, const std::string & fallback) {
    const int id = find_key(g, k);
    if (id < 0) return fallback;
    return gguf_get_val_str(g, id);
}

std::vector<float> read_filterbank_to_vector(ggml_tensor * t) {
    const size_t n_elts = ggml_nelements(t);
    std::vector<float> out(n_elts);
    if (t->type != GGML_TYPE_F32) {
        throw std::runtime_error("preproc tensor type must be f32");
    }
    // Tensor storage may live on a non-CPU backend (Vulkan/CUDA/Metal), in
    // which case `t->data` is not a host-accessible pointer. Always go via
    // the backend buffer API; it copies device->host where needed and is a
    // no-op for CPU buffers.
    ggml_backend_tensor_get(t, out.data(), 0, n_elts * sizeof(float));
    return out;
}

}

int load_from_gguf(const std::string & gguf_path,
                   ParakeetCtcModel  & out_model,
                   int                 n_threads,
                   int                 n_gpu_layers,
                   bool                verbose) {
    auto impl = std::make_shared<ParakeetCtcModel::Impl>();

    impl->backend_cpu = init_cpu_backend();
    if (!impl->backend_cpu) {
        PARAKEET_LOG_ERROR("gguf: failed to initialize CPU backend (no CPU device registered?)\n");
        return 10;
    }
    int resolved_threads = n_threads;
    if (resolved_threads <= 0) {
        const unsigned hc = std::thread::hardware_concurrency();
        resolved_threads = hc > 0 ? (int) hc : 4;
    }
    backend_set_n_threads(impl->backend_cpu, resolved_threads);

    impl->backend_blas = init_blas_backend();
    if (impl->backend_blas) {
        backend_set_n_threads(impl->backend_blas, resolved_threads);
    }

    impl->backend_gpu    = init_gpu_backend(n_gpu_layers, verbose);
    impl->backend_active = impl->backend_gpu ? impl->backend_gpu : impl->backend_cpu;

    gguf_init_params params = { /*no_alloc=*/ true, &impl->ctx };
    impl->gguf = gguf_init_from_file(gguf_path.c_str(), params);
    if (!impl->gguf) {
        PARAKEET_LOG_ERROR("gguf: failed to open %s\n", gguf_path.c_str());
        return 1;
    }

    gguf_context * g = impl->gguf;

    impl->weights_buffer = ggml_backend_alloc_ctx_tensors(impl->ctx, impl->backend_active);
    if (!impl->weights_buffer) {
        PARAKEET_LOG_ERROR("gguf: ggml_backend_alloc_ctx_tensors failed\n");
        return 12;
    }
    ggml_backend_buffer_set_usage(impl->weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    {
        std::ifstream f(gguf_path, std::ios::binary);
        if (!f) {
            PARAKEET_LOG_ERROR("gguf: cannot reopen %s for tensor data\n", gguf_path.c_str());
            return 13;
        }
        const size_t data_offset = gguf_get_data_offset(g);
        const int64_t n_tensors = gguf_get_n_tensors(g);
        std::vector<char> buf;
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char  * name = gguf_get_tensor_name(g, i);
            ggml_tensor * t    = ggml_get_tensor(impl->ctx, name);
            if (!t) continue;
            const size_t off   = gguf_get_tensor_offset(g, i);
            const size_t nbytes = ggml_nbytes(t);
            buf.resize(nbytes);
            f.seekg((std::streamoff)(data_offset + off), std::ios::beg);
            if (!f.read(buf.data(), nbytes)) {
                PARAKEET_LOG_ERROR("gguf: short read on tensor '%s' (%zu bytes)\n", name, nbytes);
                return 14;
            }
            ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
        }
    }

    {
        const int id = find_key(g, "general.architecture");
        if (id < 0) {
            PARAKEET_LOG_ERROR("gguf: missing general.architecture\n");
            return 2;
        }
        const char * arch = gguf_get_val_str(g, id);
        if (std::strcmp(arch, "parakeet-ctc") != 0) {
            PARAKEET_LOG_ERROR("gguf: expected arch=parakeet-ctc, got '%s'\n", arch);
            return 2;
        }
    }

    out_model.encoder_cfg.d_model        = get_u32(g, "parakeet.encoder.d_model", 1024);
    out_model.encoder_cfg.n_layers       = get_u32(g, "parakeet.encoder.n_layers", 24);
    out_model.encoder_cfg.n_heads        = get_u32(g, "parakeet.encoder.n_heads", 8);
    out_model.encoder_cfg.head_dim       = get_u32(g, "parakeet.encoder.head_dim",
                                                   out_model.encoder_cfg.d_model / out_model.encoder_cfg.n_heads);
    out_model.encoder_cfg.ff_dim         = get_u32(g, "parakeet.encoder.ff_dim", 4096);
    out_model.encoder_cfg.conv_kernel    = get_u32(g, "parakeet.encoder.conv_kernel", 9);
    out_model.encoder_cfg.subsampling_factor    = get_u32(g, "parakeet.encoder.subsampling_factor", 8);
    out_model.encoder_cfg.subsampling_channels  = get_u32(g, "parakeet.encoder.subsampling_conv_channels", 256);
    out_model.encoder_cfg.subsampling_freq_bins = get_u32(g, "parakeet.encoder.subsampling_freq_bins", 10);
    out_model.encoder_cfg.pos_emb_max_len = get_u32(g, "parakeet.encoder.pos_emb_max_len", 5000);
    out_model.encoder_cfg.xscaling        = get_bool(g, "parakeet.encoder.xscaling", true);
    out_model.encoder_cfg.untie_biases    = get_bool(g, "parakeet.encoder.untie_biases", true);
    out_model.encoder_cfg.use_bias        = get_bool(g, "parakeet.encoder.use_bias", true);

    {
        const std::string conv_norm = get_str(g, "parakeet.encoder.conv_norm_type", "batch_norm");
        out_model.encoder_cfg.conv_norm_type = (conv_norm == "layer_norm")
                                                 ? ConvNormType::LayerNorm
                                                 : ConvNormType::BatchNorm;
    }
    out_model.encoder_cfg.causal_downsampling =
        get_bool(g, "parakeet.encoder.causal_downsampling", false);
    {
        const std::string conv_ctx = get_str(g, "parakeet.encoder.conv_context_size", "default");
        out_model.encoder_cfg.conv_causal = (conv_ctx == "causal");
    }
    {
        const int id_l = find_key(g, "parakeet.encoder.att_context_size_left");
        const int id_r = find_key(g, "parakeet.encoder.att_context_size_right");
        if (id_l >= 0) out_model.encoder_cfg.att_context_left  = gguf_get_val_i32(g, id_l);
        if (id_r >= 0) out_model.encoder_cfg.att_context_right = gguf_get_val_i32(g, id_r);
    }
    {
        const std::string style = get_str(g, "parakeet.encoder.att_context_style", "regular");
        out_model.encoder_cfg.att_chunked_limited = (style == "chunked_limited");
    }

    out_model.supports_streaming = get_bool(g, "parakeet.encoder.streaming.enabled", false);

    const std::string mtype_str = get_str(g, "parakeet.model.type", "ctc");
    if      (mtype_str == "tdt")        out_model.model_type = ParakeetModelType::TDT;
    else if (mtype_str == "eou")        out_model.model_type = ParakeetModelType::EOU;
    else if (mtype_str == "sortformer") out_model.model_type = ParakeetModelType::SORTFORMER;
    else                                out_model.model_type = ParakeetModelType::CTC;

    if (out_model.model_type == ParakeetModelType::TDT) {
        out_model.encoder_cfg.tdt_pred_hidden     = get_u32(g, "parakeet.tdt.pred_hidden",     640);
        out_model.encoder_cfg.tdt_pred_rnn_layers = get_u32(g, "parakeet.tdt.pred_rnn_layers", 2);
        out_model.encoder_cfg.tdt_joint_hidden    = get_u32(g, "parakeet.tdt.joint_hidden",    640);
        out_model.encoder_cfg.tdt_num_durations   = get_u32(g, "parakeet.tdt.num_durations",   5);
        out_model.vocab_size = get_u32(g, "parakeet.tdt.vocab_size", 8192);
        out_model.blank_id   = get_u32(g, "parakeet.tdt.blank_id",   out_model.vocab_size);

        const int did = find_key(g, "parakeet.tdt.durations");
        if (did >= 0) {
            const size_t n = gguf_get_arr_n(g, did);
            const int32_t * data = static_cast<const int32_t *>(gguf_get_arr_data(g, did));
            out_model.tdt_durations.assign(data, data + n);
        } else {
            out_model.tdt_durations = {0, 1, 2, 3, 4};
        }
    }

    if (out_model.model_type == ParakeetModelType::EOU) {
        out_model.encoder_cfg.eou_pred_hidden           = get_u32(g, "parakeet.eou.pred_hidden",           640);
        out_model.encoder_cfg.eou_pred_rnn_layers       = get_u32(g, "parakeet.eou.pred_rnn_layers",       1);
        out_model.encoder_cfg.eou_joint_hidden          = get_u32(g, "parakeet.eou.joint_hidden",          640);
        out_model.encoder_cfg.eou_chunk_mel_frames      = get_u32(g, "parakeet.eou.encoder_chunk_mel_frames", 25);
        out_model.encoder_cfg.eou_cache_lookback_frames = get_u32(g, "parakeet.eou.cache_lookback_frames", 70);
        out_model.encoder_cfg.eou_cache_time_steps      = get_u32(g, "parakeet.eou.cache_time_steps",      8);
        out_model.encoder_cfg.eou_max_symbols_per_step  = get_u32(g, "parakeet.eou.max_symbols_per_step",  5);

        out_model.vocab_size = get_u32(g, "parakeet.eou.vocab_size", 1026);
        out_model.blank_id   = get_u32(g, "parakeet.eou.blank_id",   out_model.vocab_size);

        const int id_eou = find_key(g, "parakeet.eou.eou_id");
        const int id_eob = find_key(g, "parakeet.eou.eob_id");
        out_model.eou_id = id_eou >= 0 ? gguf_get_val_i32(g, id_eou) : -1;
        out_model.eob_id = id_eob >= 0 ? gguf_get_val_i32(g, id_eob) : -1;
    }

    if (out_model.model_type == ParakeetModelType::SORTFORMER) {
        out_model.encoder_cfg.sortformer_num_spks      = get_u32 (g, "parakeet.sortformer.num_spks",      4);
        out_model.encoder_cfg.sortformer_fc_d_model    = get_u32 (g, "parakeet.sortformer.fc_d_model",    512);
        out_model.encoder_cfg.sortformer_tf_d_model    = get_u32 (g, "parakeet.sortformer.tf_d_model",    192);
        out_model.encoder_cfg.sortformer_tf_n_layers   = get_u32 (g, "parakeet.sortformer.tf_n_layers",   18);
        out_model.encoder_cfg.sortformer_tf_n_heads    = get_u32 (g, "parakeet.sortformer.tf_n_heads",    8);
        out_model.encoder_cfg.sortformer_tf_inner_size = get_u32 (g, "parakeet.sortformer.tf_inner_size", 768);
        out_model.encoder_cfg.sortformer_tf_pre_ln     = get_bool(g, "parakeet.sortformer.tf_pre_ln",     false);
    }

    out_model.mel_cfg.sample_rate = get_u32(g, "parakeet.preproc.sample_rate", 16000);
    out_model.mel_cfg.n_fft       = get_u32(g, "parakeet.preproc.n_fft",       512);
    out_model.mel_cfg.win_length  = get_u32(g, "parakeet.preproc.win_length",  400);
    out_model.mel_cfg.hop_length  = get_u32(g, "parakeet.preproc.hop_length",  160);
    out_model.mel_cfg.n_mels      = get_u32(g, "parakeet.preproc.n_mels",      80);
    out_model.mel_cfg.preemph     = get_f32(g, "parakeet.preproc.preemph",     0.97f);
    out_model.mel_cfg.log_zero_guard_value =
        get_f32(g, "parakeet.preproc.log_zero_guard_value", kDefaultLogZeroGuard);
    {
        const std::string norm = get_str(g, "parakeet.preproc.normalize", "per_feature");
        out_model.mel_cfg.normalize = (norm == "NA" || norm == "none" || norm == "None")
                                        ? MelNormalize::None
                                        : MelNormalize::PerFeature;
    }

    if (out_model.model_type == ParakeetModelType::CTC) {
        out_model.vocab_size = get_u32(g, "parakeet.ctc.vocab_size", 1025);
        out_model.blank_id   = get_u32(g, "parakeet.ctc.blank_id",   1024);
    }
    out_model.vocab.blank_id = out_model.blank_id;

    {
        const int id = find_key(g, "tokenizer.ggml.tokens");
        if (id >= 0 && gguf_get_arr_type(g, id) == GGUF_TYPE_STRING) {
            const size_t n = gguf_get_arr_n(g, id);
            out_model.vocab.pieces.resize(n);
            for (size_t i = 0; i < n; ++i) {
                const char * s = gguf_get_arr_str(g, id, i);
                if (s) out_model.vocab.pieces[i] = s;
            }
        }
        const int id_sc = find_key(g, "tokenizer.ggml.scores");
        if (id_sc >= 0 && gguf_get_arr_n(g, id_sc) == out_model.vocab.pieces.size()) {
            const float * p = (const float *) gguf_get_arr_data(g, id_sc);
            out_model.vocab.scores.assign(p, p + out_model.vocab.pieces.size());
        }
        const int id_tp = find_key(g, "tokenizer.ggml.token_type");
        if (id_tp >= 0 && gguf_get_arr_n(g, id_tp) == out_model.vocab.pieces.size()) {
            const int8_t * p = (const int8_t *) gguf_get_arr_data(g, id_tp);
            out_model.vocab.piece_types.assign(p, p + out_model.vocab.pieces.size());
        }
        out_model.vocab.unk_id = (int32_t) get_u32(g, "tokenizer.ggml.unk_token_id", (uint32_t) -1);
        out_model.vocab.bos_id = (int32_t) get_u32(g, "tokenizer.ggml.bos_token_id", (uint32_t) -1);
        out_model.vocab.eos_id = (int32_t) get_u32(g, "tokenizer.ggml.eos_token_id", (uint32_t) -1);
        out_model.vocab.pad_id = (int32_t) get_u32(g, "tokenizer.ggml.pad_token_id", (uint32_t) -1);
    }

    out_model.mel_filterbank = require_tensor(impl->ctx, "preproc.mel_filterbank");
    out_model.window         = require_tensor(impl->ctx, "preproc.window");

    out_model.mel_cfg.filterbank = read_filterbank_to_vector(out_model.mel_filterbank);
    out_model.mel_cfg.window     = read_filterbank_to_vector(out_model.window);

    out_model.subsampling.conv0_w    = require_tensor(impl->ctx, "encoder.subsampling.conv0.weight");
    out_model.subsampling.conv0_b    = maybe_tensor(impl->ctx, "encoder.subsampling.conv0.bias");
    out_model.subsampling.conv1_dw_w = require_tensor(impl->ctx, "encoder.subsampling.conv1_dw.weight");
    out_model.subsampling.conv1_dw_b = maybe_tensor(impl->ctx, "encoder.subsampling.conv1_dw.bias");
    out_model.subsampling.conv1_pw_w = require_tensor(impl->ctx, "encoder.subsampling.conv1_pw.weight");
    out_model.subsampling.conv1_pw_b = maybe_tensor(impl->ctx, "encoder.subsampling.conv1_pw.bias");
    out_model.subsampling.conv2_dw_w = require_tensor(impl->ctx, "encoder.subsampling.conv2_dw.weight");
    out_model.subsampling.conv2_dw_b = maybe_tensor(impl->ctx, "encoder.subsampling.conv2_dw.bias");
    out_model.subsampling.conv2_pw_w = require_tensor(impl->ctx, "encoder.subsampling.conv2_pw.weight");
    out_model.subsampling.conv2_pw_b = maybe_tensor(impl->ctx, "encoder.subsampling.conv2_pw.bias");
    out_model.subsampling.out_w      = require_tensor(impl->ctx, "encoder.subsampling.out.weight");
    out_model.subsampling.out_b      = maybe_tensor(impl->ctx, "encoder.subsampling.out.bias");

    // Use converter-pre-stacked encoder.blk.*.attn.qkv on GPU when the wide
    // M=3*n_embd matmul helps; keep unstacked Q/K/V on CPU for cache locality.
    // (Heuristic: stacked wins where separate matmuls under-fill the device;
    // Metal stays unstacked here.)
    const bool gate_qkv_stack =
        impl->backend_active &&
        !backend_is_cpu(impl->backend_active) &&
        !backend_is_metal(impl->backend_active);

    out_model.blocks.resize(out_model.encoder_cfg.n_layers);
    for (int i = 0; i < out_model.encoder_cfg.n_layers; ++i) {
        BlockWeights & b = out_model.blocks[i];
        const std::string p = "encoder.blk." + std::to_string(i) + ".";

        b.norm_ff1_w  = require_tensor(impl->ctx, p + "norm_ff1.weight");
        b.norm_ff1_b  = require_tensor(impl->ctx, p + "norm_ff1.bias");
        b.ff1_l1_w    = require_tensor(impl->ctx, p + "ff1.linear1.weight");
        b.ff1_l1_b    = maybe_tensor(impl->ctx, p + "ff1.linear1.bias");
        b.ff1_l2_w    = require_tensor(impl->ctx, p + "ff1.linear2.weight");
        b.ff1_l2_b    = maybe_tensor(impl->ctx, p + "ff1.linear2.bias");

        b.norm_attn_w = require_tensor(impl->ctx, p + "norm_attn.weight");
        b.norm_attn_b = require_tensor(impl->ctx, p + "norm_attn.bias");
        b.attn_q_w    = require_tensor(impl->ctx, p + "attn.q.weight");
        b.attn_q_b    = maybe_tensor(impl->ctx, p + "attn.q.bias");
        b.attn_k_w    = require_tensor(impl->ctx, p + "attn.k.weight");
        b.attn_k_b    = maybe_tensor(impl->ctx, p + "attn.k.bias");
        b.attn_v_w    = require_tensor(impl->ctx, p + "attn.v.weight");
        b.attn_v_b    = maybe_tensor(impl->ctx, p + "attn.v.bias");
        b.attn_qkv_w  = gate_qkv_stack ? maybe_tensor(impl->ctx, p + "attn.qkv.weight") : nullptr;
        b.attn_qkv_b  = gate_qkv_stack ? maybe_tensor(impl->ctx, p + "attn.qkv.bias")   : nullptr;
        b.attn_out_w  = require_tensor(impl->ctx, p + "attn.out.weight");
        b.attn_out_b  = maybe_tensor(impl->ctx, p + "attn.out.bias");
        b.attn_pos_w  = require_tensor(impl->ctx, p + "attn.pos.weight");
        b.pos_bias_u  = require_tensor(impl->ctx, p + "attn.pos_bias_u");
        b.pos_bias_v  = require_tensor(impl->ctx, p + "attn.pos_bias_v");

        b.norm_conv_w = require_tensor(impl->ctx, p + "norm_conv.weight");
        b.norm_conv_b = require_tensor(impl->ctx, p + "norm_conv.bias");
        b.conv_pw1_w  = require_tensor(impl->ctx, p + "conv.pw1.weight");
        b.conv_pw1_b  = maybe_tensor(impl->ctx, p + "conv.pw1.bias");
        b.conv_dw_w   = require_tensor(impl->ctx, p + "conv.dw.weight");
        b.conv_dw_b   = maybe_tensor(impl->ctx, p + "conv.dw.bias");
        if (out_model.encoder_cfg.conv_norm_type == ConvNormType::LayerNorm) {
            b.conv_norm_w = require_tensor(impl->ctx, p + "conv.norm.weight");
            b.conv_norm_b = require_tensor(impl->ctx, p + "conv.norm.bias");
        } else {
            b.conv_bn_scale = require_tensor(impl->ctx, p + "conv.bn.scale");
            b.conv_bn_shift = require_tensor(impl->ctx, p + "conv.bn.shift");
        }
        b.conv_pw2_w  = require_tensor(impl->ctx, p + "conv.pw2.weight");
        b.conv_pw2_b  = maybe_tensor(impl->ctx, p + "conv.pw2.bias");

        b.norm_ff2_w  = require_tensor(impl->ctx, p + "norm_ff2.weight");
        b.norm_ff2_b  = require_tensor(impl->ctx, p + "norm_ff2.bias");
        b.ff2_l1_w    = require_tensor(impl->ctx, p + "ff2.linear1.weight");
        b.ff2_l1_b    = maybe_tensor(impl->ctx, p + "ff2.linear1.bias");
        b.ff2_l2_w    = require_tensor(impl->ctx, p + "ff2.linear2.weight");
        b.ff2_l2_b    = maybe_tensor(impl->ctx, p + "ff2.linear2.bias");

        b.norm_out_w  = require_tensor(impl->ctx, p + "norm_out.weight");
        b.norm_out_b  = require_tensor(impl->ctx, p + "norm_out.bias");
    }

    if (out_model.model_type == ParakeetModelType::CTC) {
        out_model.ctc.w = require_tensor(impl->ctx, "ctc.decoder.weight");
        out_model.ctc.b = require_tensor(impl->ctx, "ctc.decoder.bias");
    } else if (out_model.model_type == ParakeetModelType::EOU) {
        out_model.eou.predict_embed = require_tensor(impl->ctx, "eou.predict.embed.weight");
        for (int l = 0; l < out_model.encoder_cfg.eou_pred_rnn_layers; ++l) {
            const std::string pl = "eou.predict.lstm." + std::to_string(l) + ".";
            TdtLstmLayer lyr;
            lyr.w_ih = require_tensor(impl->ctx, pl + "w_ih");
            lyr.w_hh = require_tensor(impl->ctx, pl + "w_hh");
            lyr.b_ih = require_tensor(impl->ctx, pl + "b_ih");
            lyr.b_hh = require_tensor(impl->ctx, pl + "b_hh");
            out_model.eou.lstm.push_back(lyr);
        }
        out_model.eou.joint_enc_w  = require_tensor(impl->ctx, "eou.joint.enc.weight");
        out_model.eou.joint_enc_b  = require_tensor(impl->ctx, "eou.joint.enc.bias");
        out_model.eou.joint_pred_w = require_tensor(impl->ctx, "eou.joint.pred.weight");
        out_model.eou.joint_pred_b = require_tensor(impl->ctx, "eou.joint.pred.bias");
        out_model.eou.joint_out_w  = require_tensor(impl->ctx, "eou.joint.out.weight");
        out_model.eou.joint_out_b  = require_tensor(impl->ctx, "eou.joint.out.bias");
    } else if (out_model.model_type == ParakeetModelType::SORTFORMER) {
        out_model.sortformer.encoder_proj_w = require_tensor(impl->ctx, "sortformer.encoder_proj.weight");
        out_model.sortformer.encoder_proj_b = require_tensor(impl->ctx, "sortformer.encoder_proj.bias");
        out_model.sortformer.transformer.resize(out_model.encoder_cfg.sortformer_tf_n_layers);
        for (int i = 0; i < out_model.encoder_cfg.sortformer_tf_n_layers; ++i) {
            const std::string p = "sortformer.transformer.blk." + std::to_string(i) + ".";
            SortformerTransformerBlock & b = out_model.sortformer.transformer[i];
            b.attn_q_w  = require_tensor(impl->ctx, p + "attn.q.weight");
            b.attn_q_b  = require_tensor(impl->ctx, p + "attn.q.bias");
            b.attn_k_w  = require_tensor(impl->ctx, p + "attn.k.weight");
            b.attn_k_b  = require_tensor(impl->ctx, p + "attn.k.bias");
            b.attn_v_w  = require_tensor(impl->ctx, p + "attn.v.weight");
            b.attn_v_b  = require_tensor(impl->ctx, p + "attn.v.bias");
            b.attn_o_w  = require_tensor(impl->ctx, p + "attn.out.weight");
            b.attn_o_b  = require_tensor(impl->ctx, p + "attn.out.bias");
            b.ln1_w     = require_tensor(impl->ctx, p + "ln1.weight");
            b.ln1_b     = require_tensor(impl->ctx, p + "ln1.bias");
            b.ffn_in_w  = require_tensor(impl->ctx, p + "ffn.in.weight");
            b.ffn_in_b  = require_tensor(impl->ctx, p + "ffn.in.bias");
            b.ffn_out_w = require_tensor(impl->ctx, p + "ffn.out.weight");
            b.ffn_out_b = require_tensor(impl->ctx, p + "ffn.out.bias");
            b.ln2_w     = require_tensor(impl->ctx, p + "ln2.weight");
            b.ln2_b     = require_tensor(impl->ctx, p + "ln2.bias");
        }
        out_model.sortformer.head_h2h_w = require_tensor(impl->ctx, "sortformer.head.first_hidden_to_hidden.weight");
        out_model.sortformer.head_h2h_b = require_tensor(impl->ctx, "sortformer.head.first_hidden_to_hidden.bias");
        out_model.sortformer.head_h2s_w = require_tensor(impl->ctx, "sortformer.head.single_hidden_to_spks.weight");
        out_model.sortformer.head_h2s_b = require_tensor(impl->ctx, "sortformer.head.single_hidden_to_spks.bias");
    } else {
        out_model.tdt.predict_embed = require_tensor(impl->ctx, "tdt.predict.embed.weight");
        for (int l = 0; l < out_model.encoder_cfg.tdt_pred_rnn_layers; ++l) {
            const std::string pl = "tdt.predict.lstm." + std::to_string(l) + ".";
            TdtLstmLayer lyr;
            lyr.w_ih = require_tensor(impl->ctx, pl + "w_ih");
            lyr.w_hh = require_tensor(impl->ctx, pl + "w_hh");
            lyr.b_ih = require_tensor(impl->ctx, pl + "b_ih");
            lyr.b_hh = require_tensor(impl->ctx, pl + "b_hh");
            out_model.tdt.lstm.push_back(lyr);
        }
        out_model.tdt.joint_enc_w  = require_tensor(impl->ctx, "tdt.joint.enc.weight");
        out_model.tdt.joint_enc_b  = require_tensor(impl->ctx, "tdt.joint.enc.bias");
        out_model.tdt.joint_pred_w = require_tensor(impl->ctx, "tdt.joint.pred.weight");
        out_model.tdt.joint_pred_b = require_tensor(impl->ctx, "tdt.joint.pred.bias");
        out_model.tdt.joint_out_w  = require_tensor(impl->ctx, "tdt.joint.out.weight");
        out_model.tdt.joint_out_b  = require_tensor(impl->ctx, "tdt.joint.out.bias");
    }

    if (impl->backend_blas) {
        ggml_backend_free(impl->backend_blas);
        impl->backend_blas = nullptr;
    }

    out_model.impl = impl;

    if (verbose) {
        print_model_summary(out_model);
        const char * be = impl->backend_gpu
                            ? ggml_backend_name(impl->backend_gpu)
                            : "CPU";
        PARAKEET_LOG_INFO("  backend: %s  (threads=%d)\n", be, resolved_threads);
    }
    return 0;
}

bool model_has_gpu_backend(const ParakeetCtcModel & m) {
    return m.impl && m.impl->backend_gpu != nullptr;
}

std::string model_active_backend_name(const ParakeetCtcModel & m) {
    if (!m.impl) return "CPU";
    ggml_backend_t b = m.impl->backend_active;
    if (!b) return "CPU";
    const char * name = ggml_backend_name(b);
    return name ? std::string(name) : std::string("CPU");
}

ggml_backend_t model_active_backend(ParakeetCtcModel & m) {
    if (!m.impl) return nullptr;
    return m.impl->backend_active;
}

void print_model_summary(const ParakeetCtcModel & m) {
    const char * mt = "ctc";
    if (m.model_type == ParakeetModelType::TDT)        mt = "tdt";
    else if (m.model_type == ParakeetModelType::EOU)        mt = "eou";
    else if (m.model_type == ParakeetModelType::SORTFORMER) mt = "sortformer";
    PARAKEET_LOG_INFO("parakeet-%s loaded:\n", mt);
    const char * conv_norm = m.encoder_cfg.conv_norm_type == ConvNormType::LayerNorm ? "ln" : "bn";
    PARAKEET_LOG_INFO("  encoder: d_model=%d n_layers=%d n_heads=%d head_dim=%d ff_dim=%d conv_k=%d sub=%dx xscaling=%d untie=%d use_bias=%d conv_norm=%s\n",
                      m.encoder_cfg.d_model, m.encoder_cfg.n_layers, m.encoder_cfg.n_heads,
                      m.encoder_cfg.head_dim, m.encoder_cfg.ff_dim, m.encoder_cfg.conv_kernel,
                      m.encoder_cfg.subsampling_factor,
                      (int) m.encoder_cfg.xscaling, (int) m.encoder_cfg.untie_biases,
                      (int) m.encoder_cfg.use_bias, conv_norm);
    if (m.encoder_cfg.att_chunked_limited || m.encoder_cfg.causal_downsampling || m.encoder_cfg.conv_causal) {
        PARAKEET_LOG_INFO("  streaming: att_ctx=[%d,%d] style=%s causal_ds=%d conv_ctx=%s\n",
                          m.encoder_cfg.att_context_left, m.encoder_cfg.att_context_right,
                          m.encoder_cfg.att_chunked_limited ? "chunked_limited" : "regular",
                          (int) m.encoder_cfg.causal_downsampling,
                          m.encoder_cfg.conv_causal ? "causal" : "default");
    }
    PARAKEET_LOG_INFO("  preproc: sr=%d n_fft=%d win=%d hop=%d n_mels=%d preemph=%.2f log_guard=%.2e\n",
                      m.mel_cfg.sample_rate, m.mel_cfg.n_fft, m.mel_cfg.win_length,
                      m.mel_cfg.hop_length, m.mel_cfg.n_mels, m.mel_cfg.preemph,
                      (double) m.mel_cfg.log_zero_guard_value);
    if (m.model_type == ParakeetModelType::CTC) {
        PARAKEET_LOG_INFO("  ctc:     vocab=%d blank=%d\n", m.vocab_size, m.blank_id);
    } else if (m.model_type == ParakeetModelType::EOU) {
        PARAKEET_LOG_INFO("  eou:     vocab=%d blank=%d eou_id=%d eob_id=%d "
                          "pred_hidden=%d pred_layers=%d joint_hidden=%d "
                          "chunk_mel=%d cache_lookback=%d cache_time=%d max_syms=%d\n",
                          m.vocab_size, m.blank_id, m.eou_id, m.eob_id,
                          m.encoder_cfg.eou_pred_hidden, m.encoder_cfg.eou_pred_rnn_layers,
                          m.encoder_cfg.eou_joint_hidden,
                          m.encoder_cfg.eou_chunk_mel_frames,
                          m.encoder_cfg.eou_cache_lookback_frames,
                          m.encoder_cfg.eou_cache_time_steps,
                          m.encoder_cfg.eou_max_symbols_per_step);
    } else if (m.model_type == ParakeetModelType::SORTFORMER) {
        PARAKEET_LOG_INFO("  sortformer: num_spks=%d  fc_d_model=%d  tf=%dlx%dh d_model=%d inner=%d pre_ln=%d\n",
                          m.encoder_cfg.sortformer_num_spks,
                          m.encoder_cfg.sortformer_fc_d_model,
                          m.encoder_cfg.sortformer_tf_n_layers,
                          m.encoder_cfg.sortformer_tf_n_heads,
                          m.encoder_cfg.sortformer_tf_d_model,
                          m.encoder_cfg.sortformer_tf_inner_size,
                          (int) m.encoder_cfg.sortformer_tf_pre_ln);
    } else {
        PARAKEET_LOG_INFO("  tdt:     vocab=%d blank=%d pred_hidden=%d pred_layers=%d joint_hidden=%d durations=[",
                          m.vocab_size, m.blank_id,
                          m.encoder_cfg.tdt_pred_hidden, m.encoder_cfg.tdt_pred_rnn_layers,
                          m.encoder_cfg.tdt_joint_hidden);
        for (size_t i = 0; i < m.tdt_durations.size(); ++i) {
            PARAKEET_LOG_INFO("%s%d", i ? "," : "", m.tdt_durations[i]);
        }
        PARAKEET_LOG_INFO("]\n");
    }
    PARAKEET_LOG_INFO("  tensors: filterbank=%ldx%ld window=%ld blocks=%zu\n",
                      (long) m.mel_filterbank->ne[0], (long) m.mel_filterbank->ne[1],
                      (long) m.window->ne[0], m.blocks.size());
}

namespace {

ggml_tensor * zero_pad_dim0(ggml_context * ctx, ggml_tensor * x, int p_front, int p_back);
ggml_tensor * zero_pad_dim1(ggml_context * ctx, ggml_tensor * x, int p_front, int p_back);

ggml_tensor * conv_bias_bcast(ggml_context * ctx, ggml_tensor * bias, int64_t C) {
    return ggml_reshape_4d(ctx, bias, 1, 1, C, 1);
}

ggml_tensor * apply_time_mask(ggml_context * ctx, ggml_tensor * x, ggml_tensor * mask) {
    return ggml_mul(ctx, x, mask);
}

ggml_tensor * subsampling_graph(ggml_context    * gctx,
                                ggml_tensor     * mel_in,
                                const SubsamplingWeights & S,
                                int               subsampling_channels,
                                int               /*d_model*/,
                                ggml_tensor     * mask_t0,
                                ggml_tensor     * mask_t1,
                                ggml_tensor     * mask_t2,
                                ggml_tensor     * mask_t3,
                                bool              all_valid,
                                bool              causal_downsampling) {
    ggml_tensor * x = mel_in;

    auto maybe_mask = [&](ggml_tensor * xin, ggml_tensor * m) {
        return all_valid ? xin : apply_time_mask(gctx, xin, m);
    };

    // NeMo's CausalConv2D for `causal_downsampling=true` applies an
    // asymmetric (L=stride, R=stride-1) zero-pad on **both** the freq
    // (ne[0]) and time (ne[1]) axes per stride-2 conv (kernel=3), then
    // calls the conv with `padding=0`. Total pad = stride+stride-1 = 3,
    // so each layer's spatial output is `(F+3-3)/2 + 1 = F/2 + 1` --
    // freq goes 128 -> 65 -> 33 -> 17 (instead of the 128 -> 64 -> 32
    // -> 16 the symmetric `pad=1` baseline produces). The trained
    // `encoder.subsampling.out.weight` has 17 freq-bin slots, so this
    // asymmetric padding is mandatory for the matmul to line up.
    auto causal_pad = [&](ggml_tensor * xin) {
        if (!causal_downsampling) return xin;
        xin = zero_pad_dim0(gctx, xin, /*L=*/2, /*R=*/1);
        xin = zero_pad_dim1(gctx, xin, /*L=*/2, /*R=*/1);
        return xin;
    };
    const int conv_pad = causal_downsampling ? 0 : 1;

    x = maybe_mask(x, mask_t0);
    x = causal_pad(x);
    x = ggml_conv_2d(gctx, S.conv0_w, x, 2, 2, conv_pad, conv_pad, 1, 1);
    x = ggml_add(gctx, x, conv_bias_bcast(gctx, S.conv0_b, subsampling_channels));
    x = maybe_mask(x, mask_t1);
    x = ggml_relu(gctx, x);

    x = maybe_mask(x, mask_t1);
    x = causal_pad(x);
    x = ggml_conv_2d_dw(gctx, S.conv1_dw_w, x, 2, 2, conv_pad, conv_pad, 1, 1);
    x = ggml_add(gctx, x, conv_bias_bcast(gctx, S.conv1_dw_b, subsampling_channels));
    x = maybe_mask(x, mask_t2);
    x = ggml_conv_2d(gctx, S.conv1_pw_w, x, 1, 1, 0, 0, 1, 1);
    x = ggml_add(gctx, x, conv_bias_bcast(gctx, S.conv1_pw_b, subsampling_channels));
    x = maybe_mask(x, mask_t2);
    x = ggml_relu(gctx, x);

    x = maybe_mask(x, mask_t2);
    x = causal_pad(x);
    x = ggml_conv_2d_dw(gctx, S.conv2_dw_w, x, 2, 2, conv_pad, conv_pad, 1, 1);
    x = ggml_add(gctx, x, conv_bias_bcast(gctx, S.conv2_dw_b, subsampling_channels));
    x = maybe_mask(x, mask_t3);
    x = ggml_conv_2d(gctx, S.conv2_pw_w, x, 1, 1, 0, 0, 1, 1);
    x = ggml_add(gctx, x, conv_bias_bcast(gctx, S.conv2_pw_b, subsampling_channels));
    x = maybe_mask(x, mask_t3);
    x = ggml_relu(gctx, x);

    const int64_t W = x->ne[0];
    const int64_t H = x->ne[1];
    const int64_t C = x->ne[2];

    x = ggml_permute(gctx, x, 0, 2, 1, 3);
    x = ggml_cont(gctx, x);
    x = ggml_reshape_2d(gctx, x, W * C, H);

    x = ggml_mul_mat(gctx, S.out_w, x);
    x = ggml_add(gctx, x, S.out_b);
    return x;
}

int _conv_out_len(int L, int k, int s, int p) {
    return (L + 2 * p - k) / s + 1;
}

std::vector<float> compute_rel_pos_encoding(int T, int D) {
    const int L = 2 * T - 1;
    std::vector<float> pe((size_t) L * D, 0.0f);
    const float log10000 = std::log(10000.0f);
    std::vector<float> div_term(D / 2);
    for (int i = 0; i < D / 2; ++i) {
        div_term[i] = std::exp(-((float)(2 * i) * log10000 / (float) D));
    }
    std::vector<std::vector<float>> pos_pe(T, std::vector<float>(D, 0.0f));
    std::vector<std::vector<float>> neg_pe(T, std::vector<float>(D, 0.0f));
    for (int i = 0; i < T; ++i) {
        for (int k = 0; k < D / 2; ++k) {
            pos_pe[i][2*k]     = std::sin( (float) i * div_term[k]);
            pos_pe[i][2*k + 1] = std::cos( (float) i * div_term[k]);
            neg_pe[i][2*k]     = std::sin(-(float) i * div_term[k]);
            neg_pe[i][2*k + 1] = std::cos(-(float) i * div_term[k]);
        }
    }
    for (int t = 0; t < T; ++t) {
        int src = T - 1 - t;
        for (int d = 0; d < D; ++d) pe[(size_t) t * D + d] = pos_pe[src][d];
    }
    for (int t = 1; t < T; ++t) {
        for (int d = 0; d < D; ++d) pe[(size_t) (T - 1 + t) * D + d] = neg_pe[t][d];
    }
    return pe;
}

ggml_tensor * zero_pad_dim0(ggml_context * ctx, ggml_tensor * x, int p_front, int p_back) {
    if (p_front <= 0 && p_back <= 0) return x;
    ggml_tensor * y = x;
    if (p_front > 0) {
        ggml_tensor * head = ggml_view_4d(ctx, x, p_front, x->ne[1], x->ne[2], x->ne[3],
                                          x->nb[1], x->nb[2], x->nb[3], 0);
        ggml_tensor * z = ggml_scale(ctx, ggml_cont(ctx, head), 0.0f);
        y = ggml_concat(ctx, z, y, 0);
    }
    if (p_back > 0) {
        ggml_tensor * tail = ggml_view_4d(ctx, y, p_back, y->ne[1], y->ne[2], y->ne[3],
                                          y->nb[1], y->nb[2], y->nb[3], 0);
        ggml_tensor * z = ggml_scale(ctx, ggml_cont(ctx, tail), 0.0f);
        y = ggml_concat(ctx, y, z, 0);
    }
    return y;
}

// Asymmetric zero-pad along the second axis (ne[1]). Mirrors
// ``zero_pad_dim0`` but for the H dim so it can pre-pad the
// time axis of the (W=freq, H=time) mel layout used in the
// dw_striding subsampler.
ggml_tensor * zero_pad_dim1(ggml_context * ctx, ggml_tensor * x, int p_front, int p_back) {
    if (p_front <= 0 && p_back <= 0) return x;
    ggml_tensor * y = x;
    if (p_front > 0) {
        ggml_tensor * head = ggml_view_4d(ctx, x, x->ne[0], p_front, x->ne[2], x->ne[3],
                                          x->nb[1], x->nb[2], x->nb[3], 0);
        ggml_tensor * z = ggml_scale(ctx, ggml_cont(ctx, head), 0.0f);
        y = ggml_concat(ctx, z, y, 1);
    }
    if (p_back > 0) {
        ggml_tensor * tail = ggml_view_4d(ctx, y, y->ne[0], p_back, y->ne[2], y->ne[3],
                                          y->nb[1], y->nb[2], y->nb[3], 0);
        ggml_tensor * z = ggml_scale(ctx, ggml_cont(ctx, tail), 0.0f);
        y = ggml_concat(ctx, y, z, 1);
    }
    return y;
}

ggml_tensor * conv1d_via_matmul(ggml_context * ctx,
                                ggml_tensor * kernel, ggml_tensor * input,
                                int stride, int padding, int dilation) {
    ggml_tensor * kf32 = kernel->type == GGML_TYPE_F32
                       ? kernel
                       : ggml_cast(ctx, kernel, GGML_TYPE_F32);
    ggml_tensor * im2col = ggml_im2col(ctx, kf32, input,
                                       stride, 0, padding, 0, dilation, 0,
                                       false, GGML_TYPE_F32);
    ggml_tensor * r = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kf32, kf32->ne[0] * kf32->ne[1], kf32->ne[2]));
    return ggml_reshape_3d(ctx, r, im2col->ne[1], kf32->ne[2], im2col->ne[2]);
}

ggml_tensor * layer_norm_affine(ggml_context * ctx, ggml_tensor * x,
                                ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

ggml_tensor * maybe_add_bias(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias) {
    return bias ? ggml_add(ctx, x, bias) : x;
}

ggml_tensor * conformer_ff_graph(ggml_context * ctx, ggml_tensor * x,
                                 ggml_tensor * norm_w, ggml_tensor * norm_b,
                                 ggml_tensor * l1_w,  ggml_tensor * l1_b,
                                 ggml_tensor * l2_w,  ggml_tensor * l2_b,
                                 float eps) {
    x = layer_norm_affine(ctx, x, norm_w, norm_b, eps);
    x = maybe_add_bias(ctx, ggml_mul_mat(ctx, l1_w, x), l1_b);
    x = ggml_silu(ctx, x);
    x = maybe_add_bias(ctx, ggml_mul_mat(ctx, l2_w, x), l2_b);
    return x;
}

ggml_tensor * rel_pos_mha_graph(ggml_context * ctx, ggml_tensor * xn,
                                ggml_tensor * pos_emb,
                                ggml_tensor * att_mask,
                                const BlockWeights & W,
                                int H, int HD, int T) {
    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;
    if (W.attn_qkv_w) {
        // Pre-stacked encoder.blk.*.attn.qkv.weight from the converter:
        // one Q8_0 mat-mul produces (3 * n_embd, T) and Q / K / V are
        // strided ggml_view_3d slices straight into the (HD, H, T)
        // shape the reshape branch below produces.  Parent T-stride is
        // 3 * n_embd * f instead of n_embd * f, but the next ops
        // (permute → cont) walk by per-element nb01 / nb02 strides so
        // the wider stride is transparent.
        //
        // Whether `W.attn_qkv_w` was loaded at all is decided in the
        // model loader (gated to non-Apple-Metal backends — see the
        // `backend_is_metal` branch in `load_model_gguf`).  M3
        // Ultra Metal already saturates the un-stacked path's tile
        // grid (M=1024 / T=252 → 16 row × 8 col tiles ≈ 128 chunks vs
        // 60 cores: one wave fills the GPU); the stacked M=3072 path
        // adds 3× row tiles where one wave was sufficient, which is
        // measured neutral-to-slightly-bad on Apple.  CUDA / Vulkan
        // hardware with higher per-dispatch overhead and proportionally
        // smaller tiles relative to SM count is the predicted-positive
        // case; that's what the loader gate is for.
        ggml_tensor * qkv = ggml_mul_mat(ctx, W.attn_qkv_w, xn);
        if (W.attn_qkv_b) qkv = ggml_add(ctx, qkv, W.attn_qkv_b);
        const int n_embd = HD * H;
        const size_t f = sizeof(float);
        const size_t row_stride = (size_t) 3 * n_embd * f;
        q = ggml_view_3d(ctx, qkv, HD, H, T, HD * f, row_stride, 0 * (size_t) n_embd * f);
        k = ggml_view_3d(ctx, qkv, HD, H, T, HD * f, row_stride, 1 * (size_t) n_embd * f);
        v = ggml_view_3d(ctx, qkv, HD, H, T, HD * f, row_stride, 2 * (size_t) n_embd * f);
    } else {
        q = maybe_add_bias(ctx, ggml_mul_mat(ctx, W.attn_q_w, xn), W.attn_q_b);
        k = maybe_add_bias(ctx, ggml_mul_mat(ctx, W.attn_k_w, xn), W.attn_k_b);
        v = maybe_add_bias(ctx, ggml_mul_mat(ctx, W.attn_v_w, xn), W.attn_v_b);
        q = ggml_reshape_3d(ctx, q, HD, H, T);
        k = ggml_reshape_3d(ctx, k, HD, H, T);
        v = ggml_reshape_3d(ctx, v, HD, H, T);
    }
    ggml_tensor * p = ggml_mul_mat(ctx, W.attn_pos_w, pos_emb);
    p = ggml_reshape_3d(ctx, p, HD, H, pos_emb->ne[1]);

    ggml_tensor * q_perm = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * k_perm = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_tensor * p_perm = ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3));

    ggml_tensor * u_bias = ggml_reshape_3d(ctx, W.pos_bias_u, HD, 1, H);
    ggml_tensor * v_bias = ggml_reshape_3d(ctx, W.pos_bias_v, HD, 1, H);
    ggml_tensor * q_u = ggml_add(ctx, q_perm, u_bias);
    ggml_tensor * q_v = ggml_add(ctx, q_perm, v_bias);

    ggml_tensor * bd = ggml_mul_mat(ctx, p_perm, q_v);

    ggml_tensor * bd_padded   = zero_pad_dim0(ctx, bd, 1, 0);
    ggml_tensor * bd_viewed   = ggml_reshape_3d(ctx, bd_padded, T, 2 * T, H);
    ggml_tensor * bd_sliced   = ggml_view_3d(ctx, bd_viewed, T, 2 * T - 1, H,
                                             bd_viewed->nb[1], bd_viewed->nb[2], bd_viewed->nb[1]);
    ggml_tensor * bd_reshaped = ggml_reshape_3d(ctx, ggml_cont(ctx, bd_sliced), 2 * T - 1, T, H);
    ggml_tensor * bd_final    = ggml_view_3d(ctx, bd_reshaped, T, T, H,
                                             bd_reshaped->nb[1], bd_reshaped->nb[2], 0);
    bd_final = ggml_cont(ctx, bd_final);

    const float scale = 1.0f / std::sqrt((float) HD);

#ifdef PARAKEET_EXPERIMENTAL_FLASH_ATTN
    // Non-flash path computes:
    //   attn = softmax(scale * (q*k^T + bd_final) + att_mask)
    //        = softmax(scale * q*k^T + scale * bd_final + att_mask)
    // ggml_flash_attn_ext computes:
    //   attn = softmax(scale * q*k^T + mask)
    // so the equivalent mask is `scale * bd_final + att_mask`. Mode 1
    // (full-window) sets att_mask = nullptr so the fall-through to
    // bd_scaled alone is byte-exact vs the non-flash path. Mode 2 / 3
    // streaming windows pass a non-null (T_k, T_q, 1, 1) f32 chunked
    // mask that must be folded in here, otherwise FA attends to
    // positions outside the streaming window and produces token
    // duplication / EOU-detection failures (test-streaming Mode 3
    // chunk=1000 right=500 -> WER 10.5 % regression; test-eou-streaming
    // Mode 2 -> no is_eou_boundary). Broadcast: bd_scaled is
    // (T, T, H, 1); att_mask is (T, T, 1, 1); ggml_can_repeat(att_mask,
    // bd_scaled) holds so the sum is (T, T, H, 1).
    ggml_tensor * bd_scaled = ggml_scale(ctx, bd_final, scale);
    ggml_tensor * fa_mask   = att_mask ? ggml_add(ctx, bd_scaled, att_mask) : bd_scaled;
    ggml_tensor * bd_mask   = ggml_cast(ctx, fa_mask, GGML_TYPE_F16);
    ggml_tensor * attn_out  = ggml_flash_attn_ext(ctx, q_u, k_perm, v_perm, bd_mask,
                                                  scale, 0.0f, 0.0f);
    ggml_tensor * flat      = ggml_reshape_2d(ctx, attn_out, HD * H, T);
    return maybe_add_bias(ctx, ggml_mul_mat(ctx, W.attn_out_w, flat), W.attn_out_b);
#else
    ggml_tensor * ac     = ggml_mul_mat(ctx, k_perm, q_u);
    ggml_tensor * scores = ggml_add(ctx, ac, bd_final);

    ggml_tensor * attn;
    if (att_mask) {
        // ggml_soft_max_ext computes softmax(scale * x + mask) along the
        // last dim of `scores`. The kernel requires an f16 mask whose
        // shape broadcasts over the head axis: (T_k, T_q, 1, 1).
        attn = ggml_soft_max_ext(ctx, scores, att_mask, scale, 0.0f);
    } else {
        scores = ggml_scale(ctx, scores, scale);
        attn   = ggml_soft_max(ctx, scores);
    }

    ggml_tensor * v_for_mm = ggml_cont(ctx, ggml_permute(ctx, v_perm, 1, 0, 2, 3));
    ggml_tensor * attn_v   = ggml_mul_mat(ctx, v_for_mm, attn);
    ggml_tensor * merged   = ggml_cont(ctx, ggml_permute(ctx, attn_v, 0, 2, 1, 3));
    ggml_tensor * flat     = ggml_reshape_2d(ctx, merged, HD * H, T);

    return maybe_add_bias(ctx, ggml_mul_mat(ctx, W.attn_out_w, flat), W.attn_out_b);
#endif
}

ggml_tensor * conformer_conv_graph(ggml_context * ctx, ggml_tensor * xn,
                                   const BlockWeights & W,
                                   int d_model, int /*T*/, int conv_kernel,
                                   bool use_conv2d_dw,
                                   ConvNormType conv_norm_type,
                                   bool conv_causal,
                                   float layer_norm_eps) {
    ggml_tensor * pw1_w_2d = ggml_reshape_2d(ctx, W.conv_pw1_w, d_model, 2 * d_model);
    ggml_tensor * y = ggml_mul_mat(ctx, pw1_w_2d, xn);
    y = maybe_add_bias(ctx, y, W.conv_pw1_b);

    // Conformer GLU: split (2*d_model, T, B) along the channel axis into two
    // halves, then y = half1 * sigmoid(half2). The two halves are strided
    // views over the same source tensor and ggml-vulkan miscomputes the
    // sigmoid / mul kernels when fed strided inputs (validated on RTX 5060;
    // see test-vk-vs-cpu bisect: block0_conv_post_glu rel jumps from 1e-3
    // to ~0.8 without the contig). Materialising each half with ggml_cont
    // before the elementwise ops fixes the divergence on Vulkan and is a
    // no-op-cheap memcpy on CPU.
    ggml_tensor * half1 = ggml_cont(ctx,
        ggml_view_3d(ctx, y, d_model, y->ne[1], y->ne[2],
                     y->nb[1], y->nb[2], 0));
    ggml_tensor * half2 = ggml_cont(ctx,
        ggml_view_3d(ctx, y, d_model, y->ne[1], y->ne[2],
                     y->nb[1], y->nb[2],
                     (size_t) d_model * y->nb[0]));
    y = ggml_mul(ctx, half1, ggml_sigmoid(ctx, half2));

    ggml_tensor * yt = ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));

    const int pad_left  = conv_causal ? (conv_kernel - 1) : ((conv_kernel - 1) / 2);
    const int pad_right = conv_causal ? 0                 : ((conv_kernel - 1) / 2);
    if (use_conv2d_dw) {
        const int T_local = (int) yt->ne[0];
        ggml_tensor * yt_4d = ggml_reshape_4d(ctx, yt, T_local, 1, d_model, 1);
        if (conv_causal && pad_left > 0) {
            yt_4d = zero_pad_dim0(ctx, yt_4d, pad_left, pad_right);
        }
        ggml_tensor * dw_kernel_f32 = W.conv_dw_w->type == GGML_TYPE_F32
                                    ? W.conv_dw_w
                                    : ggml_cast(ctx, W.conv_dw_w, GGML_TYPE_F32);
        ggml_tensor * dw_kernel_4d = ggml_reshape_4d(ctx, dw_kernel_f32, conv_kernel, 1, 1, d_model);
        const int dw_pad = conv_causal ? 0 : pad_left;
        ggml_tensor * dw_out = ggml_conv_2d_dw_direct(ctx, dw_kernel_4d, yt_4d, 1, 1, dw_pad, 0, 1, 1);
        yt = ggml_reshape_3d(ctx, dw_out, dw_out->ne[0], d_model, 1);
    } else {
        if (conv_causal && pad_left > 0) {
            yt = zero_pad_dim0(ctx, yt, pad_left, pad_right);
            yt = ggml_conv_1d_dw(ctx, W.conv_dw_w, yt, 1, 0, 1);
        } else {
            yt = ggml_conv_1d_dw(ctx, W.conv_dw_w, yt, 1, pad_left, 1);
        }
    }
    if (W.conv_dw_b) {
        yt = ggml_add(ctx, yt, ggml_reshape_2d(ctx, W.conv_dw_b, 1, d_model));
    }

    if (conv_norm_type == ConvNormType::LayerNorm) {
        // NeMo applies LayerNorm over the channel axis. After the depthwise
        // conv we are in (T, d_model) layout; transpose to (d_model, T) so
        // ggml_norm reduces across the channel dim, run LN, then continue
        // in (d_model, T) (saves one permute vs the BN path).
        y = ggml_cont(ctx, ggml_permute(ctx, yt, 1, 0, 2, 3));
        y = layer_norm_affine(ctx, y, W.conv_norm_w, W.conv_norm_b, layer_norm_eps);
        y = ggml_silu(ctx, y);
    } else {
        yt = ggml_mul(ctx, yt, ggml_reshape_2d(ctx, W.conv_bn_scale, 1, d_model));
        yt = ggml_add(ctx, yt, ggml_reshape_2d(ctx, W.conv_bn_shift, 1, d_model));
        yt = ggml_silu(ctx, yt);
        y  = ggml_cont(ctx, ggml_permute(ctx, yt, 1, 0, 2, 3));
    }

    ggml_tensor * pw2_w_2d = ggml_reshape_2d(ctx, W.conv_pw2_w, d_model, d_model);
    y = ggml_mul_mat(ctx, pw2_w_2d, y);
    y = maybe_add_bias(ctx, y, W.conv_pw2_b);

    return y;
}

ggml_tensor * conformer_block_graph(ggml_context * ctx, ggml_tensor * x,
                                    ggml_tensor * pos_emb,
                                    ggml_tensor * att_mask,
                                    const BlockWeights & W,
                                    int d_model, int H, int HD, int T,
                                    int conv_kernel, float eps,
                                    bool use_conv2d_dw,
                                    ConvNormType conv_norm_type,
                                    bool conv_causal) {
    ggml_tensor * residual = x;
    ggml_tensor * y = conformer_ff_graph(ctx, x,
                                         W.norm_ff1_w, W.norm_ff1_b,
                                         W.ff1_l1_w,   W.ff1_l1_b,
                                         W.ff1_l2_w,   W.ff1_l2_b, eps);
    y = ggml_scale(ctx, y, 0.5f);
    x = ggml_add(ctx, residual, y);

    residual = x;
    ggml_tensor * xn = layer_norm_affine(ctx, x, W.norm_attn_w, W.norm_attn_b, eps);
    y = rel_pos_mha_graph(ctx, xn, pos_emb, att_mask, W, H, HD, T);
    x = ggml_add(ctx, residual, y);

    residual = x;
    xn = layer_norm_affine(ctx, x, W.norm_conv_w, W.norm_conv_b, eps);
    y = conformer_conv_graph(ctx, xn, W, d_model, T, conv_kernel, use_conv2d_dw,
                             conv_norm_type, conv_causal, eps);
    x = ggml_add(ctx, residual, y);

    residual = x;
    y = conformer_ff_graph(ctx, x,
                           W.norm_ff2_w, W.norm_ff2_b,
                           W.ff2_l1_w,   W.ff2_l1_b,
                           W.ff2_l2_w,   W.ff2_l2_b, eps);
    y = ggml_scale(ctx, y, 0.5f);
    x = ggml_add(ctx, residual, y);

    x = layer_norm_affine(ctx, x, W.norm_out_w, W.norm_out_b, eps);
    return x;
}

}

int run_subsampling(ParakeetCtcModel   & model,
                    const float        * mel,
                    int                  n_mel_frames,
                    int                  n_mels,
                    std::vector<float> & out_feats,
                    int                & out_n_frames) {
    if (!model.impl || !model.impl->backend_active) return -1;

    ggml_backend_t backend = model.impl->backend_active;
    const int C_sub = model.encoder_cfg.subsampling_channels;
    const int d_model = model.encoder_cfg.d_model;

    int mel_valid = 0;
    for (int t = 0; t < n_mel_frames; ++t) {
        bool nonzero = false;
        for (int m = 0; m < n_mels; ++m) {
            if (mel[(size_t) t * n_mels + m] != 0.0f) { nonzero = true; break; }
        }
        if (nonzero) mel_valid = t + 1;
    }
    if (mel_valid == 0) mel_valid = n_mel_frames;

    const bool causal_ds = model.encoder_cfg.causal_downsampling;
    auto sub_out_len = [&](int Lin) {
        return causal_ds ? (Lin / 2 + 1) : _conv_out_len(Lin, 3, 2, 1);
    };

    const int L0 = n_mel_frames;
    const int L1 = sub_out_len(L0);
    const int L2 = sub_out_len(L1);
    const int L3 = sub_out_len(L2);

    const int V0 = mel_valid;
    const int V1 = sub_out_len(V0);
    const int V2 = sub_out_len(V1);
    const int V3 = sub_out_len(V2);

    auto make_mask = [](int L, int V) {
        std::vector<float> m(L, 0.0f);
        for (int t = 0; t < L && t < V; ++t) m[t] = 1.0f;
        return m;
    };
    std::vector<float> m0 = make_mask(L0, V0);
    std::vector<float> m1 = make_mask(L1, V1);
    std::vector<float> m2 = make_mask(L2, V2);
    std::vector<float> m3 = make_mask(L3, V3);

    const size_t overhead = ggml_tensor_overhead() * (GGML_DEFAULT_GRAPH_SIZE + 64)
                          + ggml_graph_overhead();
    ggml_init_params gp = { overhead, nullptr, /*no_alloc=*/ true };
    ggml_context * gctx = ggml_init(gp);
    if (!gctx) return -2;

    ggml_tensor * mel_in  = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, n_mels, L0, 1, 1);
    ggml_set_name(mel_in, "mel_in");
    ggml_tensor * mask_t0 = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, L0, 1, 1);
    ggml_tensor * mask_t1 = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, L1, 1, 1);
    ggml_tensor * mask_t2 = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, L2, 1, 1);
    ggml_tensor * mask_t3 = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, L3, 1, 1);
    ggml_set_name(mask_t0, "mask_t0");
    ggml_set_name(mask_t1, "mask_t1");
    ggml_set_name(mask_t2, "mask_t2");
    ggml_set_name(mask_t3, "mask_t3");

    ggml_tensor * out = subsampling_graph(gctx, mel_in, model.subsampling, C_sub, d_model,
                                          mask_t0, mask_t1, mask_t2, mask_t3, false,
                                          causal_ds);
    ggml_set_name(out, "sub_out");

    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
        if (alloc) ggml_gallocr_free(alloc);
        ggml_free(gctx);
        return -3;
    }

    ggml_backend_tensor_set(mel_in, mel, 0, (size_t) n_mels * L0 * sizeof(float));
    ggml_backend_tensor_set(mask_t0, m0.data(), 0, m0.size() * sizeof(float));
    ggml_backend_tensor_set(mask_t1, m1.data(), 0, m1.size() * sizeof(float));
    ggml_backend_tensor_set(mask_t2, m2.data(), 0, m2.size() * sizeof(float));
    ggml_backend_tensor_set(mask_t3, m3.data(), 0, m3.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(alloc);
        ggml_free(gctx);
        return -4;
    }

    const int H_out = (int) out->ne[1];
    const int W_out = (int) out->ne[0];
    out_feats.resize((size_t) W_out * H_out);
    ggml_backend_tensor_get(out, out_feats.data(), 0, out_feats.size() * sizeof(float));
    out_n_frames = H_out;

    ggml_gallocr_free(alloc);
    ggml_free(gctx);
    return 0;
}

static int build_encoder_graph_cached(const ParakeetCtcModel & model,
                                      EncoderGraph & g,
                                      int n_mel_frames, int n_mels,
                                      int n_run_layers_override,
                                      bool all_valid,
                                      ggml_backend_t backend) {
    const EncoderConfig & enc = model.encoder_cfg;
    const int C_sub = enc.subsampling_channels;
    const int d_model = enc.d_model;
    const int H  = enc.n_heads;
    const int HD = enc.head_dim;
    const int N_LAYERS = enc.n_layers;
    const int conv_kernel = enc.conv_kernel;
    const float eps = enc.layer_norm_eps;

    // Metal / CUDA / Vulkan don't implement CONV_2D_DW yet; use the
    // im2col+matmul lowering on any non-CPU backend.
    const bool use_conv2d_dw = backend_is_cpu(backend);

    auto sub_out_len = [&](int Lin) {
        return enc.causal_downsampling ? (Lin / 2 + 1) : _conv_out_len(Lin, 3, 2, 1);
    };

    const int L0 = n_mel_frames;
    const int L1 = sub_out_len(L0);
    const int L2 = sub_out_len(L1);
    const int L3 = sub_out_len(L2);
    const int T = L3;

    g.pe_host = compute_rel_pos_encoding(T, d_model);

    // Build the chunked-limited attention mask host-side once per graph.
    // For an `att_context_size = [left, right]` with `att_context_style =
    // chunked_limited`, frames are grouped into chunks of `chunk_size =
    // right + 1`. A query frame at position i (in chunk c = i / chunk_size)
    // attends to keys in [c*chunk_size - left, (c+1)*chunk_size - 1].
    // Mask is `0.0f` for visible positions and `-INFINITY` for masked.
    //
    // The mask is shape (T, T) row-major in NumPy / (T, T, 1, 1) in ggml
    // (ne[0]=T_k, ne[1]=T_q). Stored as f32; ggml_soft_max_ext accepts f32
    // mask tensors and broadcasts over the head axis.
    const bool use_chunked_mask = enc.att_chunked_limited &&
                                  enc.att_context_left  >= 0 &&
                                  enc.att_context_right >= 0;
    if (use_chunked_mask) {
        const int left  = enc.att_context_left;
        const int right = enc.att_context_right;
        const int chunk = right + 1;
        // Use a large finite "very negative" sentinel rather than -inf:
        // Apple Clang at -O3 emits `-Wnan-infinity-disabled` because some
        // FP optimisations treat infinity as UB, which empirically
        // corrupts the chunked-limited mask on the EOU offline encoder
        // (CTC / TDT use full attention so they're unaffected). Softmax
        // with -1e30 saturates to ~0 just like -inf, with no UB risk.
        g.att_mask_host.assign((size_t) T * T, -1.0e30f);
        for (int i = 0; i < T; ++i) {
            const int c          = i / chunk;
            const int win_start  = c * chunk - left;
            const int win_end    = (c + 1) * chunk - 1;
            const int j0 = std::max(0, win_start);
            const int j1 = std::min(T - 1, win_end);
            float * row = g.att_mask_host.data() + (size_t) i * T;
            for (int j = j0; j <= j1; ++j) row[j] = 0.0f;
        }
    } else {
        g.att_mask_host.clear();
    }

    const size_t graph_slots = GGML_DEFAULT_GRAPH_SIZE * 16;
    const size_t overhead = ggml_tensor_overhead() * graph_slots
                          + ggml_graph_overhead_custom(graph_slots, false);
    ggml_init_params gp = { overhead, nullptr, /*no_alloc=*/ true };
    g.graph_ctx = ggml_init(gp);
    if (!g.graph_ctx) return -2;
    ggml_context * gctx = g.graph_ctx;

    g.mel_in  = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, n_mels, L0, 1, 1);
    g.mask_t0 = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, L0, 1, 1);
    g.mask_t1 = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, L1, 1, 1);
    g.mask_t2 = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, L2, 1, 1);
    g.mask_t3 = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, L3, 1, 1);
    g.pe_in   = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d_model, 2 * T - 1);
    if (use_chunked_mask) {
        g.att_mask = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, T, T, 1, 1);
        ggml_set_name(g.att_mask, "att_mask");
    } else {
        g.att_mask = nullptr;
    }
    ggml_set_name(g.mel_in,  "mel_in");
    ggml_set_name(g.mask_t0, "mask_t0");
    ggml_set_name(g.mask_t1, "mask_t1");
    ggml_set_name(g.mask_t2, "mask_t2");
    ggml_set_name(g.mask_t3, "mask_t3");
    ggml_set_name(g.pe_in,   "pe_in");

    ggml_tensor * x = subsampling_graph(gctx, g.mel_in, model.subsampling, C_sub, d_model,
                                        g.mask_t0, g.mask_t1, g.mask_t2, g.mask_t3, all_valid,
                                        enc.causal_downsampling);
    g.sub_out_node = x;
    ggml_set_name(g.sub_out_node, "subsampling_out");
    ggml_set_output(g.sub_out_node);

    if (enc.xscaling) {
        x = ggml_scale(gctx, x, std::sqrt((float) d_model));
    }

    int n_run_layers = n_run_layers_override;
    if (n_run_layers < 0) {
        n_run_layers = std::getenv("PARAKEET_MAX_LAYERS")
                     ? std::atoi(std::getenv("PARAKEET_MAX_LAYERS"))
                     : N_LAYERS;
    }
    if (n_run_layers > N_LAYERS) n_run_layers = N_LAYERS;
    if (n_run_layers < 0)        n_run_layers = 0;
    g.n_run_layers = n_run_layers;

    for (int i = 0; i < n_run_layers; ++i) {
        if (i == 0) {
            const BlockWeights & W = model.blocks[0];
            ggml_tensor * residual = x;
            ggml_tensor * y = conformer_ff_graph(gctx, x,
                                                 W.norm_ff1_w, W.norm_ff1_b,
                                                 W.ff1_l1_w,   W.ff1_l1_b,
                                                 W.ff1_l2_w,   W.ff1_l2_b, eps);
            y = ggml_scale(gctx, y, 0.5f);
            x = ggml_add(gctx, residual, y);
            g.post_ff1_0_node = x;
            ggml_set_name(g.post_ff1_0_node, "block_0_post_ff1");
            ggml_set_output(g.post_ff1_0_node);

            residual = x;
            ggml_tensor * xn = layer_norm_affine(gctx, x, W.norm_attn_w, W.norm_attn_b, eps);
            y = rel_pos_mha_graph(gctx, xn, g.pe_in, g.att_mask, W, H, HD, T);
            x = ggml_add(gctx, residual, y);
            g.post_attn_0_node = x;
            ggml_set_name(g.post_attn_0_node, "block_0_post_attn");
            ggml_set_output(g.post_attn_0_node);

            residual = x;
            xn = layer_norm_affine(gctx, x, W.norm_conv_w, W.norm_conv_b, eps);
            y = conformer_conv_graph(gctx, xn, W, d_model, T, conv_kernel, use_conv2d_dw,
                                     enc.conv_norm_type, enc.conv_causal, eps);
            x = ggml_add(gctx, residual, y);
            g.post_conv_0_node = x;
            ggml_set_name(g.post_conv_0_node, "block_0_post_conv");
            ggml_set_output(g.post_conv_0_node);

            residual = x;
            y = conformer_ff_graph(gctx, x,
                                   W.norm_ff2_w, W.norm_ff2_b,
                                   W.ff2_l1_w,   W.ff2_l1_b,
                                   W.ff2_l2_w,   W.ff2_l2_b, eps);
            y = ggml_scale(gctx, y, 0.5f);
            x = ggml_add(gctx, residual, y);
            g.post_ff2_0_node = x;
            ggml_set_name(g.post_ff2_0_node, "block_0_post_ff2");
            ggml_set_output(g.post_ff2_0_node);

            x = layer_norm_affine(gctx, x, W.norm_out_w, W.norm_out_b, eps);

            g.block_0_out_node = x;
            ggml_set_name(g.block_0_out_node, "block_0_out");
            ggml_set_output(g.block_0_out_node);
        } else {
            x = conformer_block_graph(gctx, x, g.pe_in, g.att_mask, model.blocks[i],
                                      d_model, H, HD, T, conv_kernel, eps,
                                      use_conv2d_dw,
                                      enc.conv_norm_type, enc.conv_causal);
        }
        if (i == n_run_layers - 1) {
            g.block_last_out_node = x;
            ggml_set_name(g.block_last_out_node, "block_last_out");
            ggml_set_output(g.block_last_out_node);
        }
    }

    g.encoder_out_node = x;
    ggml_set_name(g.encoder_out_node, "encoder_out");
    ggml_set_output(g.encoder_out_node);

    if (model.model_type == ParakeetModelType::CTC && model.ctc.w && model.ctc.b) {
        g.logits_node = ggml_add(gctx, ggml_mul_mat(gctx, model.ctc.w, x), model.ctc.b);
        ggml_set_name(g.logits_node, "logits");
        ggml_set_output(g.logits_node);
    } else {
        g.logits_node = nullptr;
    }

    g.cgraph = ggml_new_graph_custom(gctx, graph_slots, false);
    ggml_build_forward_expand(g.cgraph, g.sub_out_node);
    if (g.post_ff1_0_node)     ggml_build_forward_expand(g.cgraph, g.post_ff1_0_node);
    if (g.post_attn_0_node)    ggml_build_forward_expand(g.cgraph, g.post_attn_0_node);
    if (g.post_conv_0_node)    ggml_build_forward_expand(g.cgraph, g.post_conv_0_node);
    if (g.post_ff2_0_node)     ggml_build_forward_expand(g.cgraph, g.post_ff2_0_node);
    if (g.block_0_out_node)    ggml_build_forward_expand(g.cgraph, g.block_0_out_node);
    if (g.block_last_out_node) ggml_build_forward_expand(g.cgraph, g.block_last_out_node);
    ggml_build_forward_expand(g.cgraph, g.encoder_out_node);
    if (g.logits_node) ggml_build_forward_expand(g.cgraph, g.logits_node);

    g.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!g.alloc || !ggml_gallocr_reserve(g.alloc, g.cgraph)) {
        g.free_();
        return -3;
    }

    g.T_mel = n_mel_frames;
    g.all_valid = all_valid;
    return 0;
}

int run_encoder(ParakeetCtcModel   & model,
                const float        * mel,
                int                  n_mel_frames,
                int                  n_mels,
                EncoderOutputs     & out,
                int                  max_layers,
                bool                 capture_intermediates) {
    if (!model.impl || !model.impl->backend_active) return -1;

    ggml_backend_t backend = model.impl->backend_active;
    const EncoderConfig & enc = model.encoder_cfg;
    const int d_model = enc.d_model;

    // Scan from the end for the last non-zero mel frame: the mel
    // pre-processor zeros out trailing pad frames (per-feature CMVN
    // sets them to 0), and `compute_log_mel` always produces at least
    // one valid frame. Reverse-scan with early-exit replaces the
    // worst-case O(n_mel_frames * n_mels) full sweep with the typical
    // case "the last frame is valid -> 1 inner iteration only" path
    // -- the fast path for non-streaming `Engine::transcribe()` calls
    // and the tail of any chunk in Mode 2 / Mode 3 streaming.
    int mel_valid = 0;
    for (int t = n_mel_frames - 1; t >= 0; --t) {
        const float * row = mel + (size_t) t * n_mels;
        for (int m = 0; m < n_mels; ++m) {
            if (row[m] != 0.0f) { mel_valid = t + 1; break; }
        }
        if (mel_valid != 0) break;
    }
    if (mel_valid == 0) mel_valid = n_mel_frames;
    const bool all_valid = (mel_valid == n_mel_frames);

    auto & cache = model.impl->encoder_graphs;
    const int layers_key = (max_layers >= 0) ? max_layers : -1;

    EncoderGraph * g_ptr = nullptr;
    for (size_t i = 0; i < cache.size(); ++i) {
        EncoderGraph & e = *cache[i];
        const bool layers_match = (layers_key < 0) || (e.n_run_layers == layers_key);
        if (e.T_mel == n_mel_frames && layers_match && e.all_valid == all_valid) {
            if (i + 1 != cache.size()) {
                auto moved = std::move(cache[i]);
                cache.erase(cache.begin() + i);
                cache.push_back(std::move(moved));
            }
            g_ptr = cache.back().get();
            break;
        }
    }

    if (!g_ptr) {
        while (cache.size() >= ParakeetCtcModel::Impl::k_encoder_graph_cache_max) {
            cache.front()->free_();
            cache.erase(cache.begin());
        }
        cache.push_back(std::make_unique<EncoderGraph>());
        EncoderGraph & e = *cache.back();
        if (int rc = build_encoder_graph_cached(model, e, n_mel_frames, n_mels, max_layers, all_valid, backend); rc != 0) {
            cache.pop_back();
            return rc;
        }
        g_ptr = &e;
    }
    EncoderGraph & g = *g_ptr;

    auto sub_out_len = [&](int Lin) {
        return enc.causal_downsampling ? (Lin / 2 + 1) : _conv_out_len(Lin, 3, 2, 1);
    };

    const int L0 = n_mel_frames;
    const int L1 = sub_out_len(L0);
    const int L2 = sub_out_len(L1);
    const int L3 = sub_out_len(L2);

    const int V0 = mel_valid;
    const int V1 = sub_out_len(V0);
    const int V2 = sub_out_len(V1);
    const int V3 = sub_out_len(V2);

    const int T = L3;
    const int vocab_size = model.vocab_size;

    // Refresh the cached subsampling masks if the valid-frame count
    // changed since last call (or if this is the first call against
    // this cached graph). For long-form transcribe / Mode 2 streaming
    // the same `g_ptr` graph is reused across many `run_encoder` calls
    // with `all_valid == true` (graph cache key includes `all_valid`),
    // so the cache hit-rate here is essentially 100 % and we save four
    // std::vector allocations + the corresponding fills every call.
    auto refresh_mask = [](std::vector<float> & buf, int & v_cache, int L, int V) {
        if (v_cache == V && (int) buf.size() == L) return;
        buf.assign((size_t) L, 0.0f);
        for (int t = 0; t < L && t < V; ++t) buf[t] = 1.0f;
        v_cache = V;
    };
    refresh_mask(g.m0_host, g.m0_v, L0, V0);
    refresh_mask(g.m1_host, g.m1_v, L1, V1);
    refresh_mask(g.m2_host, g.m2_v, L2, V2);
    refresh_mask(g.m3_host, g.m3_v, L3, V3);

    if (!ggml_gallocr_alloc_graph(g.alloc, g.cgraph)) {
        return -3;
    }

    auto safe_set = [](ggml_tensor * t, const void * src, size_t bytes) {
        if (t && t->buffer) ggml_backend_tensor_set(t, src, 0, bytes);
    };
    safe_set(g.mel_in,  mel,                 (size_t) n_mels * L0 * sizeof(float));
    safe_set(g.mask_t0, g.m0_host.data(),    g.m0_host.size()     * sizeof(float));
    safe_set(g.mask_t1, g.m1_host.data(),    g.m1_host.size()     * sizeof(float));
    safe_set(g.mask_t2, g.m2_host.data(),    g.m2_host.size()     * sizeof(float));
    safe_set(g.mask_t3, g.m3_host.data(),    g.m3_host.size()     * sizeof(float));
    safe_set(g.pe_in,   g.pe_host.data(),    g.pe_host.size()     * sizeof(float));
    if (g.att_mask) {
        safe_set(g.att_mask, g.att_mask_host.data(),
                 g.att_mask_host.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(backend, g.cgraph) != GGML_STATUS_SUCCESS) {
        return -4;
    }

    out.n_enc_frames = T;
    out.d_model      = d_model;
    out.vocab_size   = vocab_size;

    auto copy_tensor = [&](ggml_tensor * t, std::vector<float> & dst) {
        if (!t) { dst.clear(); return; }
        dst.resize((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, dst.data(), 0, dst.size() * sizeof(float));
    };
    // The production transcribe/diarize/stream path only consumes
    // `encoder_out` (TDT/EOU/Sortformer decoders) and `logits` (CTC).
    // Skip the per-stage host copies in that case -- saves roughly
    // 7 * d_model * T_enc * 4 bytes per inference call (~4-5 MB on
    // 0.6B at T_enc=137), which is on the GPU<->host critical path
    // for OpenCL / CUDA / Vulkan / Metal streaming workloads where
    // each chunk drives a fresh `run_encoder()` round-trip.
    if (capture_intermediates) {
        copy_tensor(g.sub_out_node,         out.subsampling_out);
        copy_tensor(g.post_ff1_0_node,      out.block_0_post_ff1);
        copy_tensor(g.post_attn_0_node,     out.block_0_post_attn);
        copy_tensor(g.post_conv_0_node,     out.block_0_post_conv);
        copy_tensor(g.post_ff2_0_node,      out.block_0_post_ff2);
        copy_tensor(g.block_0_out_node,     out.block_0_out);
        copy_tensor(g.block_last_out_node,  out.block_last_out);
    } else {
        out.subsampling_out.clear();
        out.block_0_post_ff1.clear();
        out.block_0_post_attn.clear();
        out.block_0_post_conv.clear();
        out.block_0_post_ff2.clear();
        out.block_0_out.clear();
        out.block_last_out.clear();
    }
    copy_tensor(g.encoder_out_node,     out.encoder_out);
    copy_tensor(g.logits_node,          out.logits);

    return 0;
}

namespace {

struct SubstageGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * cgraph = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_tensor  * x_in  = nullptr;
    ggml_tensor  * pe_in = nullptr;
    ggml_tensor  * out   = nullptr;
    void free_() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx)   ggml_free(ctx);
        *this = SubstageGraph{};
    }
};

enum class Substage { FF1, ATTN, CONV, FF2, NORM_OUT, FULL_BLOCK };

static int build_substage_graph(const ParakeetCtcModel & model,
                                SubstageGraph & g,
                                int T, Substage stage,
                                ggml_backend_t backend) {
    const EncoderConfig & enc = model.encoder_cfg;
    const int d_model = enc.d_model;
    const int H  = enc.n_heads;
    const int HD = enc.head_dim;
    const int conv_kernel = enc.conv_kernel;
    const float eps = enc.layer_norm_eps;

    const size_t graph_slots = 4096;
    const size_t overhead = ggml_tensor_overhead() * graph_slots
                          + ggml_graph_overhead_custom(graph_slots, false);
    ggml_init_params gp = { overhead, nullptr, /*no_alloc=*/ true };
    g.ctx = ggml_init(gp);
    if (!g.ctx) return -1;

    g.x_in  = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, d_model, T, 1, 1);
    g.pe_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, d_model, 2 * T - 1);
    ggml_set_name(g.x_in, "x_in");
    ggml_set_name(g.pe_in, "pe_in");

    const BlockWeights & W = model.blocks[0];
    ggml_tensor * x = g.x_in;

    if (stage == Substage::FF1 || stage == Substage::FULL_BLOCK) {
        ggml_tensor * r = x;
        ggml_tensor * y = conformer_ff_graph(g.ctx, x,
                                             W.norm_ff1_w, W.norm_ff1_b,
                                             W.ff1_l1_w,   W.ff1_l1_b,
                                             W.ff1_l2_w,   W.ff1_l2_b, eps);
        y = ggml_scale(g.ctx, y, 0.5f);
        x = ggml_add(g.ctx, r, y);
    }
    if (stage == Substage::ATTN || stage == Substage::FULL_BLOCK) {
        ggml_tensor * r = x;
        ggml_tensor * xn = layer_norm_affine(g.ctx, x, W.norm_attn_w, W.norm_attn_b, eps);
        ggml_tensor * y = rel_pos_mha_graph(g.ctx, xn, g.pe_in, /*att_mask=*/nullptr,
                                            W, H, HD, T);
        x = ggml_add(g.ctx, r, y);
    }
    if (stage == Substage::CONV || stage == Substage::FULL_BLOCK) {
        ggml_tensor * r = x;
        ggml_tensor * xn = layer_norm_affine(g.ctx, x, W.norm_conv_w, W.norm_conv_b, eps);
        const bool use_conv2d_dw = backend_is_cpu(backend);
        ggml_tensor * y = conformer_conv_graph(g.ctx, xn, W, d_model, T, conv_kernel, use_conv2d_dw,
                                               enc.conv_norm_type, enc.conv_causal, eps);
        x = ggml_add(g.ctx, r, y);
    }
    if (stage == Substage::FF2 || stage == Substage::FULL_BLOCK) {
        ggml_tensor * r = x;
        ggml_tensor * y = conformer_ff_graph(g.ctx, x,
                                             W.norm_ff2_w, W.norm_ff2_b,
                                             W.ff2_l1_w,   W.ff2_l1_b,
                                             W.ff2_l2_w,   W.ff2_l2_b, eps);
        y = ggml_scale(g.ctx, y, 0.5f);
        x = ggml_add(g.ctx, r, y);
    }
    if (stage == Substage::NORM_OUT || stage == Substage::FULL_BLOCK) {
        x = layer_norm_affine(g.ctx, x, W.norm_out_w, W.norm_out_b, eps);
    }

    g.out = x;
    ggml_set_output(g.out);

    g.cgraph = ggml_new_graph_custom(g.ctx, graph_slots, false);
    ggml_build_forward_expand(g.cgraph, g.out);

    g.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!g.alloc || !ggml_gallocr_reserve(g.alloc, g.cgraph)) {
        g.free_();
        return -2;
    }
    return 0;
}

}

int profile_block_substages(ParakeetCtcModel & model,
                            int T_enc,
                            int warmup_runs,
                            int timed_runs,
                            BlockSubstageTimes & out) {
    if (!model.impl || !model.impl->backend_active) return -1;
    ggml_backend_t backend = model.impl->backend_active;
    const int d_model = model.encoder_cfg.d_model;

    std::vector<float> x_host((size_t) d_model * T_enc);
    for (size_t i = 0; i < x_host.size(); ++i) {
        x_host[i] = 1e-3f * (float)((int)(i * 2654435761u) % 1000 - 500);
    }
    std::vector<float> pe_host = compute_rel_pos_encoding(T_enc, d_model);

    auto measure = [&](Substage stage) -> double {
        SubstageGraph g;
        if (build_substage_graph(model, g, T_enc, stage, backend) != 0) return -1.0;

        std::vector<double> timings;
        timings.reserve(timed_runs);
        for (int r = 0; r < warmup_runs + timed_runs; ++r) {
            if (!ggml_gallocr_alloc_graph(g.alloc, g.cgraph)) { g.free_(); return -1.0; }
            ggml_backend_tensor_set(g.x_in,  x_host.data(),  0, x_host.size()  * sizeof(float));
            if (g.pe_in->buffer) {
                ggml_backend_tensor_set(g.pe_in, pe_host.data(), 0, pe_host.size() * sizeof(float));
            }

            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(backend, g.cgraph);
            const double dt = std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::steady_clock::now() - t0).count() / 1000.0;
            if (r >= warmup_runs) timings.push_back(dt);
        }
        g.free_();

        std::sort(timings.begin(), timings.end());
        return timings.empty() ? -1.0
             : (timings.size() % 2 == 1 ? timings[timings.size()/2]
                                        : 0.5 * (timings[timings.size()/2 - 1] + timings[timings.size()/2]));
    };

    out.ff1_ms      = measure(Substage::FF1);
    out.attn_ms     = measure(Substage::ATTN);
    out.conv_ms     = measure(Substage::CONV);
    out.ff2_ms      = measure(Substage::FF2);
    out.norm_out_ms = measure(Substage::NORM_OUT);
    out.block_full_ms = measure(Substage::FULL_BLOCK);
    return 0;
}

std::vector<int32_t> ctc_greedy_decode(const float * logits,
                                       int           n_frames,
                                       int           vocab_size,
                                       int32_t       blank_id) {
    std::vector<int32_t> decoded;
    int32_t prev = -1;
    ctc_greedy_decode_window(logits, 0, n_frames, vocab_size, blank_id,
                             prev, decoded, nullptr);
    return decoded;
}

void ctc_greedy_decode_window(const float * logits,
                              int           start_frame,
                              int           end_frame,
                              int           vocab_size,
                              int32_t       blank_id,
                              int32_t     & inout_prev_token,
                              std::vector<int32_t> & out_tokens,
                              std::vector<int>     * out_first_frame) {
    if (start_frame < 0) start_frame = 0;
    if (end_frame < start_frame) end_frame = start_frame;

    int32_t prev = inout_prev_token;
    for (int t = start_frame; t < end_frame; ++t) {
        const float * __restrict row = logits + static_cast<size_t>(t) * vocab_size;
        int32_t best       = 0;
        float   best_score = row[0];
        // The argmax-with-index reduction has a loop-carried dep on
        // `best_score` / `best` so it doesn't auto-vectorise as cleanly
        // as a plain reduction. `__restrict` + the explicit read into a
        // register at least lets the compiler use a fused max-with-mask
        // pattern on AVX2 / AVX-512. Same shape as the gemv treatment in
        // parakeet_tdt.cpp::gemv_f32.
        #pragma GCC ivdep
        for (int i = 1; i < vocab_size; ++i) {
            const float v = row[i];
            if (v > best_score) { best_score = v; best = i; }
        }
        if (best != blank_id && best != prev) {
            out_tokens.push_back(best);
            if (out_first_frame) out_first_frame->push_back(t);
        }
        prev = best;
    }
    inout_prev_token = prev;
}

}
