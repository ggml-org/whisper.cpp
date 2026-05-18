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

        const int64_t num_tensors = gguf_get_n_tensors(gguf_ctx);
        ggml_init_params params = {
            /*.mem_size=*/ ggml_tensor_overhead() * (size_t) num_tensors,
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

        model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
        if (!model.buffer_w) throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed");

        for (ggml_tensor * cur = ggml_get_first_tensor(model.ctx_w);
             cur;
             cur = ggml_get_next_tensor(model.ctx_w, cur)) {
            ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
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
    model.generation_id = 0;
}

} // namespace tts_cpp::supertonic::detail
