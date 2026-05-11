#include "supertonic_internal.h"

#include "ggml-cpu.h"
#include "gguf.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_OPENCL
#include "ggml-opencl.h"
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <unordered_set>
#include <stdexcept>
#include <thread>

namespace tts_cpp::supertonic::detail {
namespace {

int64_t require_key(const gguf_context * ctx, const char * key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) throw std::runtime_error(std::string("missing GGUF key: ") + key);
    return id;
}

uint32_t get_u32(const gguf_context * ctx, const char * key) {
    return gguf_get_val_u32(ctx, require_key(ctx, key));
}

float get_f32(const gguf_context * ctx, const char * key) {
    return gguf_get_val_f32(ctx, require_key(ctx, key));
}

bool get_bool_u32(const gguf_context * ctx, const char * key, bool fallback = false) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return fallback;
    return gguf_get_val_u32(ctx, id) != 0;
}

std::string get_string(const gguf_context * ctx, const char * key, const std::string & fallback = {}) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return fallback;
    return gguf_get_val_str(ctx, id);
}

std::vector<std::string> get_string_array(const gguf_context * ctx, const char * key) {
    int64_t id = require_key(ctx, key);
    size_t n = gguf_get_arr_n(ctx, id);
    std::vector<std::string> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.emplace_back(gguf_get_arr_str(ctx, id, i));
    }
    return out;
}

ggml_tensor * get_tensor_or_null(const supertonic_model & model, const std::string & name) {
    auto it = model.tensors.find(name);
    return it == model.tensors.end() ? nullptr : it->second;
}

bool should_expand_supertonic_tensor(enum ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0;
}

std::vector<float> expand_supertonic_tensor_to_f32(const ggml_tensor * src) {
    const int64_t n = ggml_nelements(src);
    std::vector<float> out((size_t) n);
    const void * data = ggml_get_data(src);
    // Use the public ggml_get_type_traits() API instead of the
    // internal ggml-quants.h helpers.  ggml-quants.h lives under
    // ggml/src/ and isn't shipped by the ggml-speech vcpkg port,
    // so direct includes break system-ggml builds (the integrated
    // tts-cpp port path).  The type-traits to_float function pointer
    // is the public dequantization entry-point and covers F16, Q8_0
    // and every other ggml type uniformly.
    const ggml_type_traits * tr = ggml_get_type_traits(src->type);
    if (!tr || !tr->to_float) {
        throw std::runtime_error(std::string("unsupported Supertonic tensor expansion type ") +
                                 ggml_type_name(src->type));
    }
    tr->to_float(data, out.data(), n);
    return out;
}

ggml_backend_t init_supertonic_backend(int n_gpu_layers, bool verbose) {
#ifdef GGML_USE_CUDA
    if (n_gpu_layers > 0) {
        ggml_backend_t b = ggml_backend_cuda_init(0);
        if (b) { if (verbose) fprintf(stderr, "supertonic: using CUDA backend\n"); return b; }
    }
#endif
#ifdef GGML_USE_METAL
    if (n_gpu_layers > 0) {
        ggml_backend_t b = ggml_backend_metal_init();
        if (b) { if (verbose) fprintf(stderr, "supertonic: using Metal backend\n"); return b; }
    }
#endif
#ifdef GGML_USE_VULKAN
    if (n_gpu_layers > 0) {
        ggml_backend_t b = ggml_backend_vk_init(0);
        if (b) {
            if (verbose) fprintf(stderr, "supertonic: using Vulkan backend\n");
            return b;
        }
    }
#endif
#ifdef GGML_USE_OPENCL
    if (n_gpu_layers > 0) {
        ggml_backend_reg_t reg = ggml_backend_opencl_reg();
        if (reg && ggml_backend_reg_dev_count(reg) > 0) {
            ggml_backend_t b = ggml_backend_opencl_init();
            if (b) { if (verbose) fprintf(stderr, "supertonic: using OpenCL backend\n"); return b; }
        }
    }
#endif
    ggml_backend_t b = ggml_backend_cpu_init();
    if (!b) throw std::runtime_error("ggml_backend_cpu_init failed");
    if (verbose) fprintf(stderr, "supertonic: using CPU backend\n");
    return b;
}

void set_env_if_unset(const char * name, const char * value) {
    if (std::getenv(name) != nullptr) return;
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 0);
#endif
}

void configure_supertonic_blas_threads_once() {
#if defined(TTS_CPP_USE_ACCELERATE)
    static bool configured = false;
    if (configured) return;
    configured = true;
    // The Supertonic CPU graphs already parallelize across GGML tasks. Letting
    // Accelerate spawn a second worker pool for every small pointwise matmul
    // hurts vector scaling on 3-4 thread runs.
    set_env_if_unset("VECLIB_MAXIMUM_THREADS", "1");
#elif defined(TTS_CPP_USE_CBLAS)
    static bool configured = false;
    if (configured) return;
    configured = true;
    set_env_if_unset("OPENBLAS_NUM_THREADS", "1");
    set_env_if_unset("MKL_NUM_THREADS", "1");
    set_env_if_unset("BLIS_NUM_THREADS", "1");
#endif
}

void print_supertonic_setup_hint() {
    fprintf(stderr,
            "Supertonic GGUFs are generated locally and intentionally ignored by git.\n"
            "Create the multilingual Supertonic 2 GGUF with:\n"
            "  bash scripts/setup-supertonic2.sh\n"
            "or create the English-only Supertonic GGUF with:\n"
            "  bash scripts/setup-supertonic2.sh --arch supertonic\n");
}

uint64_t next_supertonic_generation_id() {
    static std::atomic<uint64_t> next_id{1};
    return next_id.fetch_add(1, std::memory_order_relaxed);
}

// Process-wide alive-set keyed on generation_id.  See
// supertonic_internal.h for the rationale; contract is local to the
// register_supertonic_alive / unregister_supertonic_alive /
// is_supertonic_alive triple defined further down at the
// detail-namespace scope (so the symbols match the header
// declarations and aren't accidentally hidden in this TU's anon
// namespace).
inline std::mutex & supertonic_alive_mu() {
    static std::mutex m;
    return m;
}
inline std::unordered_set<uint64_t> & supertonic_alive_ids() {
    static std::unordered_set<uint64_t> s;
    return s;
}

} // namespace

void register_supertonic_alive(uint64_t generation_id) {
    std::lock_guard<std::mutex> lk(supertonic_alive_mu());
    supertonic_alive_ids().insert(generation_id);
}

void unregister_supertonic_alive(uint64_t generation_id) {
    std::lock_guard<std::mutex> lk(supertonic_alive_mu());
    supertonic_alive_ids().erase(generation_id);
}

bool is_supertonic_alive(uint64_t generation_id) {
    if (generation_id == 0) return false;
    std::lock_guard<std::mutex> lk(supertonic_alive_mu());
    return supertonic_alive_ids().find(generation_id) != supertonic_alive_ids().end();
}

// Thread-local dispatch flags consulted by the GGML graph builders to
// pick between the CBLAS-backed `ggml_custom_4d` fast paths (CPU only)
// and the portable pure-GGML fallbacks (any backend).  See the
// supertonic_op_dispatch_scope comment in supertonic_internal.h.
namespace {
thread_local bool g_supertonic_use_cpu_custom_ops = true;
thread_local bool g_supertonic_use_f16_attn      = false;
}

bool supertonic_use_cpu_custom_ops() {
    return g_supertonic_use_cpu_custom_ops;
}

bool supertonic_use_f16_attn() {
    return g_supertonic_use_f16_attn;
}

supertonic_op_dispatch_scope::supertonic_op_dispatch_scope(const supertonic_model & model)
    : prev_use_cpu_custom_ops(g_supertonic_use_cpu_custom_ops),
      prev_use_f16_attn(g_supertonic_use_f16_attn) {
    g_supertonic_use_cpu_custom_ops = model.backend_is_cpu;
    g_supertonic_use_f16_attn       = model.use_f16_attn;
}

supertonic_op_dispatch_scope::~supertonic_op_dispatch_scope() {
    g_supertonic_use_cpu_custom_ops = prev_use_cpu_custom_ops;
    g_supertonic_use_f16_attn       = prev_use_f16_attn;
}

ggml_tensor * require_tensor(const supertonic_model & model, const std::string & name) {
    ggml_tensor * t = get_tensor_or_null(model, name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    return t;
}

ggml_tensor * require_source_tensor(const supertonic_model & model, const std::string & source_name) {
    auto it = model.source_tensors.find(source_name);
    if (it == model.source_tensors.end() || !it->second) {
        throw std::runtime_error("missing source tensor: " + source_name);
    }
    return it->second;
}

void supertonic_set_n_threads(supertonic_model & model, int n_threads) {
    configure_supertonic_blas_threads_once();
    if (n_threads <= 0) {
        const int hw = (int) std::thread::hardware_concurrency();
        n_threads = std::min(std::max(1, hw), 4);
    }
    model.n_threads = std::max(1, n_threads);
}

void supertonic_graph_compute(const supertonic_model & model, ggml_cgraph * graph) {
    if (ggml_backend_is_cpu(model.backend) && model.n_threads > 0) {
        ggml_backend_cpu_set_n_threads(model.backend, model.n_threads);
    }
    ggml_backend_graph_compute(model.backend, graph);
}

static void bind_vocoder_weights(supertonic_model & model) {
    auto & v = model.vocoder;
    v.normalizer_scale = require_source_tensor(model, "vocoder:tts.ttl.normalizer.scale");
    v.latent_mean = require_source_tensor(model, "vocoder:tts.ae.latent_mean");
    v.latent_std = require_source_tensor(model, "vocoder:tts.ae.latent_std");
    v.embed_w = require_source_tensor(model, "vocoder:onnx::Conv_1440");
    v.embed_b = require_source_tensor(model, "vocoder:onnx::Conv_1441");
    for (int i = 0; i < 10; ++i) {
        const std::string p = "vocoder:tts.ae.decoder.convnext." + std::to_string(i);
        auto & c = v.convnext[(size_t) i];
        c.dw_w = require_source_tensor(model, p + ".dwconv.net.weight");
        c.dw_b = require_source_tensor(model, p + ".dwconv.net.bias");
        c.norm_g = require_source_tensor(model, p + ".norm.norm.weight");
        c.norm_b = require_source_tensor(model, p + ".norm.norm.bias");
        c.pw1_w = require_source_tensor(model, p + ".pwconv1.weight");
        c.pw1_b = require_source_tensor(model, p + ".pwconv1.bias");
        c.pw2_w = require_source_tensor(model, p + ".pwconv2.weight");
        c.pw2_b = require_source_tensor(model, p + ".pwconv2.bias");
        c.gamma = require_source_tensor(model, p + ".gamma");
    }
    v.final_norm_g = require_source_tensor(model, "vocoder:tts.ae.decoder.final_norm.norm.weight");
    v.final_norm_b = require_source_tensor(model, "vocoder:tts.ae.decoder.final_norm.norm.bias");
    v.final_norm_running_mean = require_source_tensor(model, "vocoder:tts.ae.decoder.final_norm.norm.running_mean");
    v.final_norm_running_var = require_source_tensor(model, "vocoder:tts.ae.decoder.final_norm.norm.running_var");
    v.head1_w = require_source_tensor(model, "vocoder:tts.ae.decoder.head.layer1.net.weight");
    v.head1_b = require_source_tensor(model, "vocoder:tts.ae.decoder.head.layer1.net.bias");
    v.head_prelu = require_source_tensor(model, "vocoder:onnx::PRelu_1505");
    v.head2_w = require_source_tensor(model, "vocoder:tts.ae.decoder.head.layer2.weight");
}

bool load_supertonic_gguf(const std::string & path,
                          supertonic_model & model,
                          int n_gpu_layers,
                          bool verbose) {
    model.generation_id = next_supertonic_generation_id();
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp_ctx };
    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), gp);
    if (!gguf_ctx) {
        fprintf(stderr, "load_supertonic_gguf: failed to open '%s'\n", path.c_str());
        print_supertonic_setup_hint();
        return false;
    }

    try {
        std::string arch = get_string(gguf_ctx, "supertonic.arch");
        if (arch != "supertonic2" && arch != "supertonic") {
            throw std::runtime_error("unexpected supertonic.arch: " + arch);
        }

        model.hparams.arch = arch;
        model.hparams.ftype = get_string(gguf_ctx, "supertonic.ftype", "f32");
        model.hparams.sample_rate = (int) get_u32(gguf_ctx, "supertonic.sample_rate");
        model.hparams.base_chunk_size = (int) get_u32(gguf_ctx, "supertonic.base_chunk_size");
        model.hparams.ttl_chunk_compress_factor =
            (int) get_u32(gguf_ctx, "supertonic.ttl_chunk_compress_factor");
        model.hparams.latent_dim = (int) get_u32(gguf_ctx, "supertonic.latent_dim");
        model.hparams.latent_channels = (int) get_u32(gguf_ctx, "supertonic.latent_channels");
        model.hparams.default_steps = (int) get_u32(gguf_ctx, "supertonic.default_steps");
        model.hparams.default_speed = get_f32(gguf_ctx, "supertonic.default_speed");
        model.hparams.language_wrap_mode = get_string(gguf_ctx, "supertonic.language_wrap_mode");
        if (model.hparams.language_wrap_mode.empty()) {
            bool language_wrap = get_bool_u32(gguf_ctx, "supertonic.language_wrap", arch != "supertonic");
            model.hparams.language_wrap_mode = language_wrap ? (arch == "supertonic2" ? "open_close" : "prefix") : "none";
        }
        model.hparams.default_voice = get_string(gguf_ctx, "supertonic.default_voice", "F1");
        model.languages = get_string_array(gguf_ctx, "supertonic.languages");
        model.tts_json = get_string(gguf_ctx, "supertonic.tts_json");

        model.backend = init_supertonic_backend(n_gpu_layers, verbose);
        // The graph builders below dispatch between CBLAS-backed
        // `ggml_custom_4d` fast paths (CPU only) and pure-GGML fallbacks
        // (any backend) based on this flag.  Stable for the model's
        // lifetime; see the supertonic_op_dispatch_scope comment in
        // supertonic_internal.h for the threading contract.
        model.backend_is_cpu = ggml_backend_is_cpu(model.backend);
        if (verbose) {
            fprintf(stderr, "supertonic: backend_is_cpu=%s\n", model.backend_is_cpu ? "true" : "false");
        }

        const int64_t num_tensors = gguf_get_n_tensors(gguf_ctx);
        // Reserve a small surplus of tensor-overhead slots for the
        // audit-driven pre-baked tensors that load_supertonic_gguf
        // appends to `model.ctx_w` below: F2 vocoder bn_scale_pre +
        // bn_shift_pre, plus F6's pre-transposed companions for the
        // five hot t_proj weights.  A surplus of 16 covers the
        // current roster + headroom for follow-up audit phases.
        constexpr int64_t kPrebakedTensorSurplus = 16;
        ggml_init_params params = {
            /*.mem_size=*/ ggml_tensor_overhead() * (size_t)(num_tensors + kPrebakedTensorSurplus),
            /*.mem_buffer=*/ nullptr,
            /*.no_alloc=*/ true,
        };
        model.ctx_w = ggml_init(params);
        if (!model.ctx_w) throw std::runtime_error("ggml_init failed");

        std::unordered_map<std::string, std::vector<float>> expanded_f32_tensors;
        for (int64_t i = 0; i < num_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
            if (!src) throw std::runtime_error(std::string("missing tmp tensor: ") + name);
            ggml_tensor * dst = should_expand_supertonic_tensor(src->type)
                ? ggml_new_tensor(model.ctx_w, GGML_TYPE_F32, ggml_n_dims(src), src->ne)
                : ggml_dup_tensor(model.ctx_w, src);
            ggml_set_name(dst, name);
            model.tensors[name] = dst;
            if (should_expand_supertonic_tensor(src->type)) {
                expanded_f32_tensors[name] = expand_supertonic_tensor_to_f32(src);
            }
        }

        // Audit finding F2 — declare the pre-baked vocoder BN
        // tensors BEFORE `ggml_backend_alloc_ctx_tensors` so they
        // get a slot in the same backend buffer as the rest of the
        // model weights.  Data is uploaded after the source-tensor
        // upload loop further down; see the F2 hook after
        // `bind_vocoder_weights`.
        model.vocoder.bn_scale_pre = ggml_new_tensor_1d(model.ctx_w, GGML_TYPE_F32, 512);
        ggml_set_name(model.vocoder.bn_scale_pre, "vocoder/bn_scale_pre");
        model.vocoder.bn_shift_pre = ggml_new_tensor_1d(model.ctx_w, GGML_TYPE_F32, 512);
        ggml_set_name(model.vocoder.bn_shift_pre, "vocoder/bn_shift_pre");

        // Audit finding F6 — declare the pre-transposed companion
        // tensors for the four t_proj matmul weights.  Each one has
        // shape [512, 64] in the GGUF (matches the Supertonic-2
        // architecture's time-embedding projection); the transposed
        // form is [64, 512], i.e. axes 0/1 swapped.  Data uploaded
        // after `bind_vocoder_weights` in the F6 post-bind hook.
        // The roster matches AUDIT_SUPERTONIC_OPENCL.md F6 + the
        // test in test_supertonic_load_caches.cpp.
        ggml_tensor * pretrans_t_proj[4] = {nullptr, nullptr, nullptr, nullptr};
        static const char * const kF6PretransNames[4] = {
            "vector_estimator:onnx::MatMul_3095__T",
            "vector_estimator:onnx::MatMul_3140__T",
            "vector_estimator:onnx::MatMul_3185__T",
            "vector_estimator:onnx::MatMul_3230__T",
        };
        for (int i = 0; i < 4; ++i) {
            pretrans_t_proj[i] = ggml_new_tensor_2d(model.ctx_w, GGML_TYPE_F32, 64, 512);
            ggml_set_name(pretrans_t_proj[i], kF6PretransNames[i]);
        }

        model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
        if (!model.buffer_w) throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed");

        for (ggml_tensor * cur = ggml_get_first_tensor(model.ctx_w);
             cur;
             cur = ggml_get_next_tensor(model.ctx_w, cur)) {
            ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
            if (!src) {
                // Pre-baked tensor (F2 / F6 / future audit phases):
                // declared in model.ctx_w earlier in this function but
                // doesn't have a GGUF source row — data is uploaded by
                // the dedicated post-bind hook further down.  Skip
                // here so we don't deref a null `src`.
                continue;
            }
            auto expanded = expanded_f32_tensors.find(ggml_get_name(cur));
            if (expanded != expanded_f32_tensors.end()) {
                ggml_backend_tensor_set(cur, expanded->second.data(), 0,
                                        expanded->second.size() * sizeof(float));
            } else {
                ggml_backend_tensor_set(cur, ggml_get_data(src), 0, ggml_nbytes(src));
            }
        }

        {
            ggml_tensor * unicode = require_tensor(model, "supertonic/unicode_indexer");
            model.unicode_indexer.resize((size_t) ggml_nelements(unicode));
            ggml_backend_tensor_get(unicode, model.unicode_indexer.data(), 0, ggml_nbytes(unicode));
        }

        std::vector<std::string> tensor_names = get_string_array(gguf_ctx, "supertonic.tensor_names");
        std::vector<std::string> source_names = get_string_array(gguf_ctx, "supertonic.source_names");
        if (tensor_names.size() != source_names.size()) {
            throw std::runtime_error("supertonic tensor/source metadata length mismatch");
        }
        for (size_t i = 0; i < tensor_names.size(); ++i) {
            ggml_tensor * t = require_tensor(model, tensor_names[i]);
            model.source_tensors[source_names[i]] = t;
        }

        for (const std::string & voice_name : get_string_array(gguf_ctx, "supertonic.voice_names")) {
            supertonic_voice_style voice;
            voice.name = voice_name;
            voice.ttl = require_tensor(model, "supertonic/voices/" + voice_name + "/ttl");
            voice.dp  = require_tensor(model, "supertonic/voices/" + voice_name + "/dp");
            model.voices[voice_name] = voice;
        }

        bind_vocoder_weights(model);

        // Audit finding F1 — cache the vector-estimator RoPE θ
        // tensor on the host once at load time.  All four group
        // attention sites in `supertonic_vector_step_ggml`'s
        // production GGML path read from the same source tensor;
        // caching here avoids 4 × N_STEPS GPU→host downloads per
        // synth on a non-CPU backend.  Tensor is small (64 floats
        // typical), so the host-side copy cost is negligible
        // compared with the sync-point savings.  See
        // AUDIT_SUPERTONIC_OPENCL.md F1 + PLAN Phase 2F.
        //
        // The source tensor is mandatory for any production
        // Supertonic GGUF (all four group attention sites depend
        // on it); fail-fast at load time so the call-site
        // assumption "model.vector_rope_theta.data() is non-null"
        // can stay assertion-free.  Matches the previous behaviour
        // where the same tensor was looked up via
        // `read_f32(model, "...theta")` on the hot path and would
        // throw `runtime_error("missing source tensor: ...")`.
        {
            ggml_tensor * theta_src = require_source_tensor(model,
                "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
            model.vector_rope_theta.resize((size_t) ggml_nelements(theta_src));
            ggml_backend_tensor_get(theta_src,
                                    model.vector_rope_theta.data(),
                                    0, ggml_nbytes(theta_src));
        }

        // Audit finding F2 — compute the vocoder BN scale / shift
        // pre-bake.  Downloads the four final_norm.* tensors that
        // were just uploaded a few lines above (so this is a single
        // round-trip at load time, not per-synth), folds them into
        // the BN-fused form, and uploads to bn_scale_pre /
        // bn_shift_pre which the vocoder graph cache references
        // directly as weights.  Every subsequent synth call skips
        // the 4 reads + CPU compute + 2 uploads that the old path
        // did.  See AUDIT_SUPERTONIC_OPENCL.md F2.
        {
            auto download = [](ggml_tensor * t, std::vector<float> & out) {
                out.resize((size_t) ggml_nelements(t));
                ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
            };
            std::vector<float> gamma, beta, mean, var;
            download(model.vocoder.final_norm_g, gamma);
            download(model.vocoder.final_norm_b, beta);
            download(model.vocoder.final_norm_running_mean, mean);
            download(model.vocoder.final_norm_running_var,  var);
            if (gamma.size() != 512 || beta.size() != 512 ||
                mean.size() != 512  || var.size()  != 512) {
                throw std::runtime_error(
                    "vocoder final_norm.* size mismatch (expected 512 each)");
            }
            std::vector<float> bn_scale_pre(512), bn_shift_pre(512);
            for (int c = 0; c < 512; ++c) {
                bn_scale_pre[c] = gamma[c] / std::sqrt(var[c] + 1e-5f);
                bn_shift_pre[c] = beta[c] - mean[c] * bn_scale_pre[c];
            }
            ggml_backend_tensor_set(model.vocoder.bn_scale_pre,
                                    bn_scale_pre.data(), 0, 512 * sizeof(float));
            ggml_backend_tensor_set(model.vocoder.bn_shift_pre,
                                    bn_shift_pre.data(), 0, 512 * sizeof(float));
        }

        // Audit finding F6 — populate the pre-transposed t_proj
        // companions from the source tensors.  At the four call
        // sites in supertonic_vector_estimator.cpp the original
        // matmul weight is consumed as `ggml_cont(ggml_transpose(W))`
        // every graph build; storing the transposed form in the
        // backend buffer once at load eliminates both the in-graph
        // transpose op and the ~640 KiB of compute-buffer copies
        // that came with it.  Each source is downloaded once,
        // transposed host-side, and uploaded into the companion.
        // The mapping from `<name>` to `<name>__T` is added to
        // `model.source_tensors` so `require_source_tensor` works
        // at the rewritten call sites.
        {
            static const char * const kF6Sources[4] = {
                "vector_estimator:onnx::MatMul_3095",
                "vector_estimator:onnx::MatMul_3140",
                "vector_estimator:onnx::MatMul_3185",
                "vector_estimator:onnx::MatMul_3230",
            };
            for (int i = 0; i < 4; ++i) {
                if (!pretrans_t_proj[i]) continue;
                auto it = model.source_tensors.find(kF6Sources[i]);
                if (it == model.source_tensors.end() || !it->second) continue;
                ggml_tensor * orig = it->second;
                // Defensive: only pre-transpose the F32 [512, 64]
                // shape the audit roster targets.  Any other layout
                // means the GGUF doesn't fit the assumed
                // architecture (or has already been quantized below
                // F32, in which case the call-site rewrite would
                // need a different lowering anyway).
                if (orig->type != GGML_TYPE_F32 ||
                    orig->ne[0] != 512 || orig->ne[1] != 64 ||
                    orig->ne[2] != 1   || orig->ne[3] != 1) {
                    continue;
                }
                std::vector<float> src((size_t) ggml_nelements(orig));
                ggml_backend_tensor_get(orig, src.data(), 0, ggml_nbytes(orig));
                std::vector<float> dst((size_t) 64 * 512);
                // Transpose: dst[i, j] = src[j, i] where source ne=
                // [512, 64].  Memory: src[j * 512 + i],
                // dst[i * 64 + j].
                for (int j = 0; j < 64; ++j) {
                    for (int ii = 0; ii < 512; ++ii) {
                        dst[(size_t) ii * 64 + j] = src[(size_t) j * 512 + ii];
                    }
                }
                ggml_backend_tensor_set(pretrans_t_proj[i], dst.data(), 0, dst.size() * sizeof(float));
                model.source_tensors[std::string(kF6Sources[i]) + "__T"] = pretrans_t_proj[i];
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "load_supertonic_gguf: %s\n", e.what());
        gguf_free(gguf_ctx);
        if (tmp_ctx) ggml_free(tmp_ctx);
        free_supertonic_model(model);
        return false;
    }

    gguf_free(gguf_ctx);
    ggml_free(tmp_ctx);
    // Mark this model alive only after all the load steps succeeded.
    // The per-stage thread_local graph caches consult is_supertonic_alive()
    // before calling ggml_gallocr_free() to skip the free path against a
    // backend that's already been torn down.
    register_supertonic_alive(model.generation_id);
    return true;
}

void free_supertonic_model(supertonic_model & model) {
    // Unregister BEFORE freeing the backend so any concurrent / subsequent
    // free_*_cache() call on a stale thread_local cache sees the
    // generation as no-longer-alive and skips ggml_gallocr_free against
    // the soon-to-be-dead backend.
    if (model.generation_id != 0) {
        unregister_supertonic_alive(model.generation_id);
    }
    if (model.buffer_w) {
        ggml_backend_buffer_free(model.buffer_w);
        model.buffer_w = nullptr;
    }
    if (model.backend) {
        ggml_backend_free(model.backend);
        model.backend = nullptr;
    }
    if (model.ctx_w) {
        ggml_free(model.ctx_w);
        model.ctx_w = nullptr;
    }
    model.tensors.clear();
    model.source_tensors.clear();
    model.vocoder = {};
    model.voices.clear();
    model.unicode_indexer.clear();
    model.languages.clear();
    model.tts_json.clear();
    // Reset the OpenCL optimization caches (audit F1 / F9) added to
    // supertonic_model.  The vector-estimator RoPE θ cache is a
    // bare std::vector so its clear() is sufficient; the time
    // embedding cache map is mutable so we clear it explicitly here
    // even though dtor would handle it on the next load reuse.
    model.vector_rope_theta.clear();
    model.time_emb_cache.clear();
    model.generation_id = 0;
}

} // namespace tts_cpp::supertonic::detail
