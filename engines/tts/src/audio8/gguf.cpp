#include "audio8/internal.h"

#include "audio8/coreml_path.h"
#include "coreml_sidecar_path.h"
#include "audio8/graph.h"
#include "backend_selection.h"
#include "backend_util.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include "gguf_stream.h"

#ifdef TTS_CPP_USE_COREML
#include "audio8/coreml/codec-synth.h"
#endif

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace tts_cpp {
namespace audio8 {
namespace detail {
namespace {

// Reads the GGUF's metadata and tensor headers; the payload stays on disk
// until a gguf_stream_reader pulls it into the backend buffer.
class gguf_file {
public:
    explicit gguf_file(const std::string & path) : path_(path) {
        gguf_init_params params = {/*no_alloc=*/true, &meta_};
        ctx_ = gguf_init_from_file(path.c_str(), params);
    }
    ~gguf_file() {
        if (ctx_) gguf_free(ctx_);
        if (meta_) ggml_free(meta_);
    }
    gguf_file(const gguf_file &) = delete;
    gguf_file & operator=(const gguf_file &) = delete;

    bool ok() const { return ctx_ != nullptr; }
    const gguf_context * ctx() const { return ctx_; }
    const ggml_context * meta() const { return meta_; }
    const std::string & path() const { return path_; }

private:
    std::string path_;
    gguf_context * ctx_ = nullptr;
    ggml_context * meta_ = nullptr;
};

// Collects the first failure so a whole block of reads can be written as a
// straight list and checked once.
class metadata {
public:
    metadata(const gguf_context * ctx, const std::string & prefix)
        : ctx_(ctx), prefix_(prefix) {}

    // gguf_get_val_* / gguf_get_arr_* GGML_ABORT on a type mismatch, so a
    // mistyped key would kill the process. Every reader here keeps its
    // caller-supplied default when the stored type is not the one asked for.
    void u32(const char * key, int & out) {
        const int64_t id = find(key);
        if (id >= 0 && gguf_get_kv_type(ctx_, id) == GGUF_TYPE_UINT32) {
            out = static_cast<int>(gguf_get_val_u32(ctx_, id));
        }
    }
    void f32(const char * key, float & out) {
        const int64_t id = find(key);
        if (id >= 0 && gguf_get_kv_type(ctx_, id) == GGUF_TYPE_FLOAT32) {
            out = gguf_get_val_f32(ctx_, id);
        }
    }
    void boolean(const char * key, bool & out) {
        const int64_t id = find(key);
        if (id >= 0 && gguf_get_kv_type(ctx_, id) == GGUF_TYPE_BOOL) {
            out = gguf_get_val_bool(ctx_, id);
        }
    }
    void ints(const char * key, std::vector<int> & out) {
        const int64_t id = find(key);
        if (id < 0) return;
        if (gguf_get_kv_type(ctx_, id) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(ctx_, id) != GGUF_TYPE_INT32) {
            return;
        }
        const int count = static_cast<int>(gguf_get_arr_n(ctx_, id));
        const int32_t * values = static_cast<const int32_t *>(gguf_get_arr_data(ctx_, id));
        out.assign(values, values + count);
    }
    void strings(const char * key, std::vector<std::string> & out) {
        const int64_t id = find(key);
        if (id < 0) return;
        if (gguf_get_kv_type(ctx_, id) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(ctx_, id) != GGUF_TYPE_STRING) {
            return;
        }
        const int count = static_cast<int>(gguf_get_arr_n(ctx_, id));
        out.resize(count);
        for (int index = 0; index < count; ++index) {
            out[index] = gguf_get_arr_str(ctx_, id, index);
        }
    }
    std::string text(const char * key) {
        const int64_t id = find(key);
        if (id < 0 || gguf_get_kv_type(ctx_, id) != GGUF_TYPE_STRING) return std::string();
        return gguf_get_val_str(ctx_, id);
    }

    bool ok() const { return error_.empty(); }
    const std::string & error() const { return error_; }

private:
    int64_t find(const char * key) {
        const std::string full = prefix_ + key;
        const int64_t id = gguf_find_key(ctx_, full.c_str());
        if (id < 0 && error_.empty()) error_ = "missing GGUF key: " + full;
        return id;
    }

    const gguf_context * ctx_;
    std::string prefix_;
    std::string error_;
};

// Absolute keys (tokenizer, architecture) sit outside the model's namespace.
class raw_metadata : public metadata {
public:
    explicit raw_metadata(const gguf_context * ctx) : metadata(ctx, "") {}
};

bool architecture_is(const gguf_file & file, const char * expected) {
    const int64_t id = gguf_find_key(file.ctx(), "general.architecture");
    if (id < 0 || gguf_get_kv_type(file.ctx(), id) != GGUF_TYPE_STRING) return false;
    return std::string(gguf_get_val_str(file.ctx(), id)) == expected;
}

ggml_context * new_weight_context(int64_t n_tensors) {
    const size_t size = static_cast<size_t>(n_tensors + 8) * ggml_tensor_overhead();
    ggml_init_params params = {size, nullptr, /*no_alloc=*/true};
    return ggml_init(params);
}

// A weight the loader assembles by concatenating the rows of several GGUF
// tensors, so one matmul replaces several; see docs/audio8.md ("The language
// model's projections are fused at load").
struct fused_weight {
    std::string name;
    std::vector<std::string> parts;
};

using fusion_plan = std::vector<fused_weight>;

const std::vector<std::string> QKV_PARTS = {"wq", "wk", "wv"};
const std::vector<std::string> QKV_BIAS_PARTS = {"wq_b", "wk_b", "wv_b"};
const std::vector<std::string> GATE_UP_PARTS = {"w1", "w3"};

bool fusion_disabled() {
    return std::getenv("AUDIO8_LM_FUSION_DISABLE") != nullptr;
}

ggml_tensor * meta_tensor(const gguf_file & file, const std::string & name) {
    return ggml_get_tensor(const_cast<ggml_context *>(file.meta()), name.c_str());
}

constexpr int VECTOR_DIMS = 1;
constexpr int MATRIX_DIMS = 2;

// Rows are whole blocks in every storage type, so matrices that share a type
// and a row width, or vectors that share a type, stack as a byte concatenation.
bool stacks_with(const ggml_tensor * first, const ggml_tensor * part) {
    if (!part || part->type != first->type || ggml_n_dims(part) != ggml_n_dims(first)) {
        return false;
    }
    if (ggml_n_dims(first) == VECTOR_DIMS) return true;
    return ggml_n_dims(first) == MATRIX_DIMS && part->ne[0] == first->ne[0];
}

bool rows_stack(const gguf_file & file, const fused_weight & fused) {
    const ggml_tensor * first = meta_tensor(file, fused.parts.front());
    if (!first) return false;
    for (const std::string & name : fused.parts) {
        if (!stacks_with(first, meta_tensor(file, name))) return false;
    }
    return true;
}

fused_weight named_fusion(const std::string & prefix, const std::string & name,
                          const std::vector<std::string> & suffixes) {
    fused_weight fused;
    fused.name = prefix + name;
    for (const std::string & suffix : suffixes) fused.parts.push_back(prefix + suffix);
    return fused;
}

// A file that already carries a tensor under the stacked name keeps its own;
// two tensors of one name would leave the loader writing the stack into the
// file's.
bool plan_fusion(const gguf_file & file, const fused_weight & fused, fusion_plan & plan) {
    if (meta_tensor(file, fused.name) || !rows_stack(file, fused)) return false;
    plan.push_back(fused);
    return true;
}

// The biases only stack together with their weights: the graph reads either
// both stacked or both separate.
void plan_layer_fusions(const gguf_file & file, const std::string & prefix,
                        fusion_plan & plan) {
    if (plan_fusion(file, named_fusion(prefix, "wqkv", QKV_PARTS), plan)) {
        plan_fusion(file, named_fusion(prefix, "wqkv_b", QKV_BIAS_PARTS), plan);
    }
    plan_fusion(file, named_fusion(prefix, "w13", GATE_UP_PARTS), plan);
}

void plan_branch_fusions(const gguf_file & file, const std::string & branch, int depth,
                         fusion_plan & plan) {
    for (int index = 0; index < depth; ++index) {
        plan_layer_fusions(file, branch + "/blk/" + std::to_string(index) + "/", plan);
    }
}

fusion_plan plan_lm_fusions(const gguf_file & file, const lm_hparams & hp) {
    fusion_plan plan;
    if (fusion_disabled()) return plan;
    plan_branch_fusions(file, "lm", hp.depth, plan);
    plan_branch_fusions(file, "fast", hp.fast_depth, plan);
    return plan;
}

std::unordered_set<std::string> fused_parts(const fusion_plan & plan) {
    std::unordered_set<std::string> parts;
    for (const fused_weight & fused : plan) parts.insert(fused.parts.begin(), fused.parts.end());
    return parts;
}

void clone_tensor_headers(const ggml_context * meta, ggml_context * dst,
                          const std::unordered_set<std::string> & skipped) {
    ggml_context * source = const_cast<ggml_context *>(meta);
    for (ggml_tensor * t = ggml_get_first_tensor(source); t;
         t = ggml_get_next_tensor(source, t)) {
        if (skipped.count(ggml_get_name(t))) continue;
        ggml_set_name(ggml_dup_tensor(dst, t), ggml_get_name(t));
    }
}

int64_t stacked_rows(const gguf_file & file, const fused_weight & fused, int dim) {
    int64_t total = 0;
    for (const std::string & name : fused.parts) total += meta_tensor(file, name)->ne[dim];
    return total;
}

void add_fused_header(const gguf_file & file, const fused_weight & fused, ggml_context * dst) {
    const ggml_tensor * first = meta_tensor(file, fused.parts.front());
    ggml_tensor * t = ggml_n_dims(first) == VECTOR_DIMS
                          ? ggml_new_tensor_1d(dst, first->type, stacked_rows(file, fused, 0))
                          : ggml_new_tensor_2d(dst, first->type, first->ne[0],
                                               stacked_rows(file, fused, 1));
    ggml_set_name(t, fused.name.c_str());
}

void add_fused_headers(const gguf_file & file, const fusion_plan & plan, ggml_context * dst) {
    for (const fused_weight & fused : plan) add_fused_header(file, fused, dst);
}

bool read_parts(::tts_cpp::detail::gguf_stream_reader & reader, const gguf_file & file,
                const fused_weight & fused, std::vector<uint8_t> & bytes) {
    size_t offset = 0;
    for (const std::string & name : fused.parts) {
        const size_t size = ggml_nbytes(meta_tensor(file, name));
        if (offset + size > bytes.size()) return false;
        if (!reader.to_host(name.c_str(), bytes.data() + offset, size)) return false;
        offset += size;
    }
    return offset == bytes.size();
}

bool stream_parts_in_place(::tts_cpp::detail::gguf_stream_reader & reader,
                           const gguf_file & file, const fused_weight & fused,
                           ggml_tensor * dst) {
    size_t offset = 0;
    for (const std::string & name : fused.parts) {
        const size_t size = ggml_nbytes(meta_tensor(file, name));
        if (!reader.to_backend_at(name.c_str(), dst, offset, size)) return false;
        offset += size;
    }
    return offset == ggml_nbytes(dst);
}

// Plain types stream each part in chunks at its offset; a quantised stack is
// staged on the host and uploaded in one call, which is what the OpenCL
// quantised layouts need (gguf_stream.h). The staging buffer is reused.
bool stream_fused(::tts_cpp::detail::gguf_stream_reader & reader, const gguf_file & file,
                  const fused_weight & fused, ggml_context * ctx,
                  std::vector<uint8_t> & bytes) {
    ggml_tensor * dst = ggml_get_tensor(ctx, fused.name.c_str());
    if (!ggml_is_quantized(dst->type)) return stream_parts_in_place(reader, file, fused, dst);
    bytes.resize(ggml_nbytes(dst));
    if (!read_parts(reader, file, fused, bytes)) return false;
    ggml_backend_tensor_set(dst, bytes.data(), 0, bytes.size());
    return true;
}

bool stream_plain(::tts_cpp::detail::gguf_stream_reader & reader, ggml_context * ctx,
                  const std::unordered_set<std::string> & fused_names) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (fused_names.count(ggml_get_name(t))) continue;
        if (!reader.to_backend(ggml_get_name(t), t)) return false;
    }
    return true;
}

bool stream_all_fused(::tts_cpp::detail::gguf_stream_reader & reader, const gguf_file & file,
                      const fusion_plan & plan, ggml_context * ctx) {
    std::vector<uint8_t> bytes;
    for (const fused_weight & fused : plan) {
        if (!stream_fused(reader, file, fused, ctx, bytes)) return false;
    }
    return true;
}

std::unordered_set<std::string> fused_names(const fusion_plan & plan) {
    std::unordered_set<std::string> names;
    for (const fused_weight & fused : plan) names.insert(fused.name);
    return names;
}

bool stream_weights(const gguf_file & file, ggml_context * ctx, const fusion_plan & plan) {
    ::tts_cpp::detail::gguf_stream_reader reader(file.ctx(), file.path());
    if (!reader.ok()) return false;
    return stream_plain(reader, ctx, fused_names(plan)) &&
           stream_all_fused(reader, file, plan, ctx);
}

// Mark every unallocated tensor in `ctx` externally allocated (dummy non-null
// data, the same trick ggml's own measure paths use) so graph pricing via
// ggml_gallocr / ggml_backend_sched excludes it from the measured compute
// buffers instead of counting it as a graph-owned leaf. The context must
// never have tensor data read or written after this.
void mark_externally_allocated(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t;
         t = ggml_get_next_tensor(ctx, t)) {
        if (!t->data && !t->view_src) {
            t->data = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
        }
    }
}

ggml_backend_buffer_type_t hexagon_projection_buffer_type(ggml_backend_t backend) {
    // Repacking is the validated Audio8 Hexagon default. Keep an explicit
    // fallback for comparisons and devices that need the canonical layout.
    const char * setting = std::getenv("AUDIO8_LM_HEXAGON_REPACK");
    if (setting && std::strcmp(setting, "0") == 0) return nullptr;
    const auto device = ggml_backend_get_device(backend);
    if (!device) return nullptr;
    const auto registry = ggml_backend_dev_backend_reg(device);
    if (!registry) return nullptr;
    if (std::strcmp(ggml_backend_reg_name(registry), "HTP") != 0) return nullptr;
    const auto extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(registry, "ggml_backend_dev_get_extra_bufts"));
    if (!extra) return nullptr;
    const auto types = extra(device);
    if (!types) return nullptr;
    for (size_t i = 0; types[i]; ++i) {
        if (std::strstr(ggml_backend_buft_name(types[i]), "-REPACK") &&
            ggml_backend_dev_supports_buft(device, types[i])) return types[i];
    }
    return nullptr;
}

bool is_q8_projection(const ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_Q8_0 || ggml_n_dims(tensor) != MATRIX_DIMS ||
        tensor->ne[0] % 32 != 0 || tensor->ne[1] > 32768) return false;
    const std::string name = ggml_get_name(tensor);
    const size_t prefix = name.compare(0, 7, "lm/blk/") == 0 ? 7 :
                          (name.compare(0, 9, "fast/blk/") == 0 ? 9 : 0);
    if (!prefix) return false;
    const size_t slash = name.find('/', prefix);
    if (slash == std::string::npos || slash == prefix ||
        name.find_first_not_of("0123456789", prefix) != slash) return false;
    const std::string role = name.substr(slash + 1);
    return role == "wqkv" || role == "wo" || role == "w13" || role == "w2" ||
           role == "wq" || role == "wk" || role == "wv" || role == "w1" || role == "w3";
}

// Allocate eligible projections first. The normal context allocator then skips
// their resident tensors and allocates every other weight in its usual layout.
// The metadata-only path applies the same partition and backend padding.
bool allocate_projection_weights(ggml_context * ctx, ggml_backend_buffer_type_t type,
                                 std::vector<ggml_backend_buffer_t> & buffers,
                                 size_t * measure_bytes, std::string * error) {
    if (!type) return true;
    const size_t alignment = ggml_backend_buft_get_alignment(type);
    const size_t max_size = ggml_backend_buft_get_max_size(type);
    std::vector<ggml_tensor *> tensors;
    size_t chunk_bytes = 0;
    auto allocate_chunk = [&]() {
        if (tensors.empty()) return true;
        if (measure_bytes) {
            if (*measure_bytes > std::numeric_limits<size_t>::max() - chunk_bytes) return false;
            *measure_bytes += chunk_bytes;
            for (auto * tensor : tensors) {
                tensor->data = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
            }
        } else {
            auto buffer = ggml_backend_buft_alloc_buffer(type, chunk_bytes);
            if (!buffer) return false;
            buffers.push_back(buffer);
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            auto allocator = ggml_tallocr_new(buffer);
            for (auto * tensor : tensors) {
                if (ggml_tallocr_alloc(&allocator, tensor) != GGML_STATUS_SUCCESS) return false;
            }
        }
        tensors.clear();
        chunk_bytes = 0;
        return true;
    };
    bool success = alignment > 0 && max_size > 0;
    for (auto * tensor = ggml_get_first_tensor(ctx); success && tensor;
         tensor = ggml_get_next_tensor(ctx, tensor)) {
        if (!is_q8_projection(tensor)) continue;
        const size_t bytes = ggml_backend_buft_get_alloc_size(type, tensor);
        if (bytes > std::numeric_limits<size_t>::max() - (alignment - 1)) {
            success = false;
            break;
        }
        const size_t padded = ((bytes + alignment - 1) / alignment) * alignment;
        if (padded > max_size) {
            success = false;
            break;
        }
        if (padded > max_size - chunk_bytes && !allocate_chunk()) {
            success = false;
            break;
        }
        tensors.push_back(tensor);
        chunk_bytes += padded;
    }
    if (success) success = allocate_chunk();
    if (!success && error) *error = "audio8: failed to allocate repacked projection weights";
    return success;
}

// When `measure_bytes` is non-null the load is metadata-only: the weight
// buffer the real path allocates is SIZED instead (the size-only twin of
// ggml_backend_alloc_ctx_tensors runs the same buffer-type sizing sweep) and
// no tensor data leaves the disk.
bool load_weights(const gguf_file & file, ggml_backend_t backend, ggml_context ** ctx,
                  ggml_backend_buffer_t * buffer, std::string * error,
                  size_t * measure_bytes = nullptr, const fusion_plan & plan = {},
                  std::vector<ggml_backend_buffer_t> * projection_buffers = nullptr) {
    *ctx = new_weight_context(gguf_get_n_tensors(file.ctx()) +
                              static_cast<int64_t>(plan.size()));
    if (!*ctx) {
        if (error) *error = "audio8: failed to create the weight context";
        return false;
    }
    clone_tensor_headers(file.meta(), *ctx, fused_parts(plan));
    add_fused_headers(file, plan, *ctx);
    if (measure_bytes) *measure_bytes = 0;
    if (projection_buffers && !allocate_projection_weights(
            *ctx, hexagon_projection_buffer_type(backend), *projection_buffers,
            measure_bytes, error)) return false;
    if (measure_bytes) {
        const size_t canonical_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(
            *ctx, ggml_backend_get_default_buffer_type(backend));
        if (*measure_bytes > std::numeric_limits<size_t>::max() - canonical_bytes) {
            if (error) *error = "audio8: weight buffer size overflow";
            return false;
        }
        *measure_bytes += canonical_bytes;
        mark_externally_allocated(*ctx);
        return true;
    }
    *buffer = ggml_backend_alloc_ctx_tensors(*ctx, backend);
    if (!*buffer) {
        if (error) *error = "audio8: failed to allocate the weight buffer";
        return false;
    }
    if (!stream_weights(file, *ctx, plan)) {
        if (error) *error = "audio8: failed to read tensor data from " + file.path();
        return false;
    }
    return true;
}

// A capability probe only ever holds its operand and the op asked about.
constexpr int PROBE_NODES = 2;

// init_gpu_backend's auto-pick: the Vulkan adapter with the most free memory,
// never an integrated one while a discrete one is visible.
constexpr int AUTO_VULKAN_DEVICE = -1;

// Resolve a device's registry name without pulling in a shared helper
// (dev_reg_name is Parakeet-local). Returns an empty string on nullptr
// so string comparisons are safe either way.
const char * audio8_dev_reg_name(ggml_backend_dev_t dev) {
    if (!dev) return "";
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char * n = reg ? ggml_backend_reg_name(reg) : nullptr;
    return n ? n : "";
}

// Walks the ggml device registry and inits the first device whose reg +
// device name match `requested`. Returns nullptr when no device matches or
// init itself failed. On failure the caller emits a user-visible
// "failed to init a compute backend" error.
ggml_backend_t init_explicit_backend(const std::string & requested) {
    ::tts_cpp::detail::ensure_backends_loaded();
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const char * reg_name  = audio8_dev_reg_name(dev);
        const char * dev_name  = ggml_backend_dev_name(dev);
        const enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
        if (!backend_selection_matches(requested, reg_name, dev_name, dev_type)) {
            continue;
        }
        if (ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr)) {
            std::fprintf(stderr,
                "[audio8] explicitly selected %s backend (%s)\n",
                reg_name ? reg_name : "?", dev_name ? dev_name : "unknown");
            return backend;
        }
    }
    std::fprintf(stderr,
        "[audio8] requested backend \"%s\" unavailable or failed initialization; "
        "no fallback\n", requested.c_str());
    return nullptr;
}

// Audio8's GPU path is enabled only on backends its whole graph set has been
// validated against, stage by stage, against the F32 reference. Anything else
// falls back to CPU rather than running unverified kernels.
//
// `backend` is EngineOptions::backend (CLI --backend): "" / "auto" takes the
// legacy tier-based GPU path; "cpu" short-circuits to the CPU backend;
// "hexagon" or an exact ggml device name is resolved against the ggml device
// registry via init_explicit_backend. Explicit requests never fall back to
// CPU on failure -- the caller sees a hard "failed to init a compute
// backend" error instead of running on the wrong device.
ggml_backend_t init_backend(int n_gpu_layers, const std::string & backend) {
    if (backend == "cpu") {
        return ::tts_cpp::detail::init_cpu_backend();
    }
    if (!backend.empty() && backend != "auto") {
        return init_explicit_backend(backend);
    }
    using ::tts_cpp::detail::GpuBackendRequirement;
    ggml_backend_t gpu = ::tts_cpp::detail::init_gpu_backend(
        n_gpu_layers, true, "audio8", AUTO_VULKAN_DEVICE, false, nullptr,
        GpuBackendRequirement::Vulkan | GpuBackendRequirement::Metal |
            GpuBackendRequirement::OpenCL | GpuBackendRequirement::CUDA);
    return gpu ? gpu : ::tts_cpp::detail::init_cpu_backend();
}

// Looks tensors up by name and remembers the first one that was missing, so a
// wiring block reads as a plain list of assignments.
class tensor_map {
public:
    explicit tensor_map(ggml_context * ctx) : ctx_(ctx) {}

    ggml_tensor * get(const std::string & name) {
        ggml_tensor * t = ggml_get_tensor(ctx_, name.c_str());
        if (!t && error_.empty()) error_ = "audio8: missing tensor " + name;
        return t;
    }
    ggml_tensor * maybe(const std::string & name) const {
        return ggml_get_tensor(ctx_, name.c_str());
    }
    void reject(const std::string & why) {
        if (error_.empty()) error_ = why;
    }
    bool ok() const { return error_.empty(); }
    const std::string & error() const { return error_; }

private:
    ggml_context * ctx_;
    std::string error_;
};

// What a branch's projections must measure, from its hparams: the graph cuts
// the q, k and v heads and the gate and up halves by these counts, so a file
// that disagrees is refused here instead of aborting in a reshape.
struct projection_geometry {
    int hidden = 0;
    int n_head = 0;
    int n_kv = 0;
    int head_dim = 0;
    int inter = 0;
    bool bias = false;
};

std::string extent(int64_t cols, int64_t rows) {
    return "[" + std::to_string(cols) + ", " + std::to_string(rows) + "]";
}

void expect_shape(tensor_map & map, const std::string & name, const ggml_tensor * t,
                  int64_t cols, int64_t rows) {
    if (!t || (t->ne[0] == cols && t->ne[1] == rows && t->ne[2] == 1)) return;
    map.reject("audio8: " + name + " is " + extent(t->ne[0], t->ne[1]) +
               ", the model's head geometry needs " + extent(cols, rows));
}

void expect_length(tensor_map & map, const std::string & name, const ggml_tensor * t,
                   int64_t length) {
    if (!t || (t->ne[0] == length && ggml_n_dims(t) == 1)) return;
    map.reject("audio8: " + name + " holds " + std::to_string(ggml_nelements(t)) +
               " values, the model's head geometry needs " + std::to_string(length));
}

void check_stacked_attention(tensor_map & map, const std::string & prefix,
                             const projection_geometry & g, const attention_weights & attn) {
    const int64_t rows = static_cast<int64_t>(g.n_head + 2 * g.n_kv) * g.head_dim;
    expect_shape(map, prefix + "wqkv", attn.wqkv, g.hidden, rows);
    if (g.bias) expect_length(map, prefix + "wqkv_b", attn.wqkv_b, rows);
}

void check_separate_attention(tensor_map & map, const std::string & prefix,
                              const projection_geometry & g, const attention_weights & attn) {
    const int64_t q_rows = static_cast<int64_t>(g.n_head) * g.head_dim;
    const int64_t kv_rows = static_cast<int64_t>(g.n_kv) * g.head_dim;
    expect_shape(map, prefix + "wq", attn.wq, g.hidden, q_rows);
    expect_shape(map, prefix + "wk", attn.wk, g.hidden, kv_rows);
    expect_shape(map, prefix + "wv", attn.wv, g.hidden, kv_rows);
    if (!g.bias) return;
    expect_length(map, prefix + "wq_b", attn.wq_b, q_rows);
    expect_length(map, prefix + "wk_b", attn.wk_b, kv_rows);
    expect_length(map, prefix + "wv_b", attn.wv_b, kv_rows);
}

void check_gate_up(tensor_map & map, const std::string & prefix, const projection_geometry & g,
                   const block_weights & block) {
    if (block.w13) {
        expect_shape(map, prefix + "w13", block.w13, g.hidden, 2 * static_cast<int64_t>(g.inter));
        return;
    }
    expect_shape(map, prefix + "w1", block.w1, g.hidden, g.inter);
    expect_shape(map, prefix + "w3", block.w3, g.hidden, g.inter);
}

void check_projections(tensor_map & map, const std::string & prefix,
                       const projection_geometry & g, const block_weights & block) {
    if (block.attn.wqkv) {
        check_stacked_attention(map, prefix, g, block.attn);
    } else {
        check_separate_attention(map, prefix, g, block.attn);
    }
    check_gate_up(map, prefix, g, block);
}

void read_projections(tensor_map & map, const std::string & prefix, bool has_bias,
                      attention_weights & attn) {
    attn.wqkv = map.maybe(prefix + "wqkv");
    if (attn.wqkv) {
        if (has_bias) attn.wqkv_b = map.get(prefix + "wqkv_b");
        return;
    }
    attn.wq = map.get(prefix + "wq");
    attn.wk = map.get(prefix + "wk");
    attn.wv = map.get(prefix + "wv");
    if (!has_bias) return;
    attn.wq_b = map.get(prefix + "wq_b");
    attn.wk_b = map.get(prefix + "wk_b");
    attn.wv_b = map.get(prefix + "wv_b");
}

void read_attention(tensor_map & map, const std::string & prefix, bool has_bias,
                    attention_weights & attn) {
    read_projections(map, prefix, has_bias, attn);
    attn.wo = map.get(prefix + "wo");
    attn.attn_norm = map.get(prefix + "attn_norm");
}

void read_gate_up(tensor_map & map, const std::string & prefix, block_weights & block) {
    block.w13 = map.maybe(prefix + "w13");
    if (block.w13) return;
    block.w1 = map.get(prefix + "w1");
    block.w3 = map.get(prefix + "w3");
}

void read_block(tensor_map & map, const std::string & prefix, const projection_geometry & g,
                block_weights & block) {
    read_attention(map, prefix, g.bias, block.attn);
    read_gate_up(map, prefix, block);
    block.w2 = map.get(prefix + "w2");
    block.ffn_norm = map.get(prefix + "ffn_norm");
    check_projections(map, prefix, g, block);
}

void read_branch(tensor_map & map, const std::string & branch, int depth,
                 const projection_geometry & g, std::vector<block_weights> & blocks) {
    blocks.resize(depth);
    for (int index = 0; index < depth; ++index) {
        read_block(map, branch + "/blk/" + std::to_string(index) + "/", g, blocks[index]);
    }
}

projection_geometry slow_geometry(const lm_hparams & hp) {
    return {hp.hidden, hp.n_head, hp.n_kv, hp.head_dim, hp.inter, hp.qkv_bias};
}

projection_geometry fast_geometry(const lm_hparams & hp) {
    return {hp.fast_hidden, hp.fast_n_head, hp.fast_n_kv, hp.fast_head_dim, hp.fast_inter,
            hp.fast_qkv_bias};
}

void read_lm_hparams(metadata & meta, lm_hparams & hp) {
    meta.u32("depth", hp.depth);
    meta.u32("hidden", hp.hidden);
    meta.u32("n_head", hp.n_head);
    meta.u32("n_kv", hp.n_kv);
    meta.u32("head_dim", hp.head_dim);
    meta.u32("inter", hp.inter);
    meta.u32("vocab", hp.vocab);
    meta.u32("fast_depth", hp.fast_depth);
    meta.u32("fast_hidden", hp.fast_hidden);
    meta.u32("fast_n_head", hp.fast_n_head);
    meta.u32("fast_n_kv", hp.fast_n_kv);
    meta.u32("fast_head_dim", hp.fast_head_dim);
    meta.u32("fast_inter", hp.fast_inter);
    meta.u32("num_codebooks", hp.num_codebooks);
    meta.u32("codebook_size", hp.codebook_size);
    meta.u32("semantic_begin", hp.semantic_begin);
    meta.u32("semantic_end", hp.semantic_end);
    meta.u32("eos", hp.eos);
    meta.u32("pad", hp.pad);
    meta.u32("max_seq_len", hp.max_seq_len);
    meta.u32("ras_window", hp.ras_window);
    meta.f32("rope_theta", hp.rope_theta);
    meta.f32("rms_eps", hp.rms_eps);
    meta.f32("ras_top_p", hp.ras_top_p);
    meta.f32("ras_temperature", hp.ras_temperature);
    meta.boolean("norm_fast_input", hp.norm_fast_input);
    meta.boolean("qkv_bias", hp.qkv_bias);
    meta.boolean("fast_qkv_bias", hp.fast_qkv_bias);
}

void read_tokenizer(raw_metadata & meta, TokenizerData & data) {
    meta.strings("tokenizer.ggml.tokens", data.tokens);
    meta.strings("tokenizer.ggml.merges", data.merges);
    meta.ints("tokenizer.ggml.added_token_ids", data.added_token_ids);
}

// K is position-major. V uses the same layout for F32 flash attention, or
// channel-major storage for the separate value matmul.
// In measure mode (`measure_bytes` non-null) the slab is sized, not
// allocated, and the tensors come back marked externally allocated so graph
// pricing can still build the cache views over them.
bool alloc_kv(ggml_backend_t backend, int layers, int width, int capacity, kv_cache & cache,
              size_t * measure_bytes = nullptr) {
    ggml_init_params params = {4 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true};
    cache.ctx = ggml_init(params);
    if (!cache.ctx) return false;
    cache.capacity = capacity;
    cache.stride = width;
    cache.k = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, width,
                                 static_cast<int64_t>(capacity) * layers);
    cache.v = cache.flash_attention
        ? ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, width, static_cast<int64_t>(capacity) * layers)
        : ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, capacity, static_cast<int64_t>(width) * layers);
    if (measure_bytes) {
        *measure_bytes += ggml_backend_alloc_ctx_tensors_from_buft_size(
            cache.ctx, ggml_backend_get_default_buffer_type(backend));
        mark_externally_allocated(cache.ctx);
        return true;
    }
    cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
    return cache.buffer != nullptr;
}

void free_kv(kv_cache & cache) {
    if (cache.buffer) ggml_backend_buffer_free(cache.buffer);
    if (cache.ctx) ggml_free(cache.ctx);
    cache = kv_cache{};
}

void read_transformer_spec(metadata & meta, const std::string & name, transformer_spec & spec) {
    const std::string prefix = name + ".";
    meta.u32((prefix + "n_layer").c_str(), spec.n_layer);
    meta.u32((prefix + "n_head").c_str(), spec.n_head);
    meta.u32((prefix + "n_kv").c_str(), spec.n_kv);
    meta.u32((prefix + "dim").c_str(), spec.dim);
    meta.u32((prefix + "inter").c_str(), spec.inter);
    meta.u32((prefix + "window").c_str(), spec.window);
    meta.u32((prefix + "head_dim").c_str(), spec.head_dim);
    meta.f32((prefix + "rope_theta").c_str(), spec.rope_theta);
    meta.f32((prefix + "norm_eps").c_str(), spec.norm_eps);
}

void read_scaled_block(tensor_map & map, const std::string & prefix,
                       scaled_block_weights & block) {
    read_attention(map, prefix, /*has_bias=*/false, block.attn);
    block.attn_scale = map.get(prefix + "attn_scale");
    block.w1 = map.get(prefix + "w1");
    block.w2 = map.get(prefix + "w2");
    block.w3 = map.get(prefix + "w3");
    block.ffn_norm = map.get(prefix + "ffn_norm");
    block.ffn_scale = map.get(prefix + "ffn_scale");
}

void read_window_transformer(tensor_map & map, const std::string & prefix,
                             const std::string & rope, window_transformer & tf) {
    tf.blocks.resize(tf.spec.n_layer);
    for (int index = 0; index < tf.spec.n_layer; ++index) {
        read_scaled_block(map, prefix + "/blk/" + std::to_string(index) + "/", tf.blocks[index]);
    }
    tf.norm = map.get(prefix + "/norm/weight");
    tf.rope_cos = map.get("rope/" + rope + "_cos");
    tf.rope_sin = map.get("rope/" + rope + "_sin");
}

void read_conv(tensor_map & map, const std::string & prefix, conv_weights & conv) {
    conv.w = map.get(prefix + "/weight");
    conv.b = map.get(prefix + "/bias");
}

void read_convnext(tensor_map & map, const std::string & prefix, convnext_weights & block) {
    read_conv(map, prefix + "/dwconv/conv", block.dwconv);
    block.norm_w = map.get(prefix + "/norm/weight");
    block.norm_b = map.get(prefix + "/norm/bias");
    read_conv(map, prefix + "/pwconv1", block.pwconv1);
    read_conv(map, prefix + "/pwconv2", block.pwconv2);
    block.gamma = map.get(prefix + "/gamma");
}

// The quantizer's resamplers are a fixed pair of stride-2 stages, each a
// conv followed by a ConvNeXt block.
constexpr int RESAMPLE_STAGES = 2;
constexpr int RESAMPLE_STRIDE = 2;

void read_resamplers(tensor_map & map, const std::string & prefix,
                     std::vector<resample_stage> & stages) {
    stages.resize(RESAMPLE_STAGES);
    for (int index = 0; index < RESAMPLE_STAGES; ++index) {
        const std::string base = prefix + "/" + std::to_string(index);
        read_conv(map, base + "/0/conv", stages[index].conv);
        read_convnext(map, base + "/1", stages[index].convnext);
        stages[index].stride = RESAMPLE_STRIDE;
    }
}

void read_quantizers(tensor_map & map, const std::string & prefix, int count,
                     bool encoding, std::vector<quantizer_weights> & out) {
    out.resize(count);
    for (int index = 0; index < count; ++index) {
        const std::string base = prefix + "/" + std::to_string(index);
        out[index].codebook = map.get(base + "/codebook/weight");
        read_conv(map, base + "/out_proj", out[index].out_proj);
        if (encoding) read_conv(map, base + "/in_proj", out[index].in_proj);
    }
}

void read_residual_units(tensor_map & map, const std::string & prefix, int first, int count,
                         std::vector<residual_unit> & units) {
    units.resize(count);
    for (int index = 0; index < count; ++index) {
        const std::string base = prefix + "/" + std::to_string(index + first) + "/block";
        units[index].alpha1 = map.get(base + "/0/alpha");
        read_conv(map, base + "/1/conv", units[index].conv1);
        units[index].alpha2 = map.get(base + "/2/alpha");
        read_conv(map, base + "/3/conv", units[index].conv2);
    }
}

// The GGUF keeps the reference's Sequential indices, so stage n of a part sits
// at <part>/(n+1) and the entries below are that Sequential's own positions.
std::string stage_prefix(const std::string & part, size_t index) {
    return part + "/" + std::to_string(index + 1) + "/block";
}

void read_decoder_stages(tensor_map & map, const std::vector<int> & strides, int dilations,
                         std::vector<dac_stage> & stages) {
    stages.resize(strides.size());
    for (size_t index = 0; index < strides.size(); ++index) {
        const std::string base = stage_prefix("dec", index);
        stages[index].alpha = map.get(base + "/0/alpha");
        read_conv(map, base + "/1/conv", stages[index].conv);
        stages[index].stride = strides[index];
        read_residual_units(map, base, /*first=*/2, dilations, stages[index].units);
    }
}

void read_encoder_stages(tensor_map & map, const std::vector<int> & strides, int dilations,
                         const transformer_spec & spec, std::vector<dac_stage> & stages) {
    stages.resize(strides.size());
    for (size_t index = 0; index < strides.size(); ++index) {
        const std::string base = stage_prefix("enc", index);
        read_residual_units(map, base, /*first=*/0, dilations, stages[index].units);
        stages[index].alpha = map.get(base + "/" + std::to_string(dilations) + "/alpha");
        read_conv(map, base + "/" + std::to_string(dilations + 1) + "/conv", stages[index].conv);
        stages[index].stride = strides[index];
        const std::string nested = base + "/" + std::to_string(dilations + 2);
        stages[index].has_transformer = map.maybe(nested + "/norm/weight") != nullptr;
        if (!stages[index].has_transformer) continue;
        stages[index].transformer.spec = spec;
        read_window_transformer(map, nested, "enc_tf", stages[index].transformer);
    }
}

void read_codec_hparams(metadata & meta, codec_hparams & hp) {
    meta.u32("sample_rate", hp.sample_rate);
    meta.u32("frame_size", hp.frame_size);
    meta.u32("num_codebooks", hp.num_codebooks);
    meta.u32("latent_dim", hp.latent_dim);
    meta.u32("codebook_dim", hp.codebook_dim);
    meta.u32("max_frames", hp.max_frames);
    meta.u32("semantic_codebook_size", hp.semantic_codebook_size);
    meta.u32("residual_codebook_size", hp.residual_codebook_size);
    meta.u32("residual_codebooks", hp.residual_codebooks);
    meta.f32("convnext_norm_eps", hp.convnext_norm_eps);
    meta.f32("snake_epsilon", hp.snake_epsilon);
    meta.ints("encoder_strides", hp.encoder_strides);
    meta.ints("decoder_strides", hp.decoder_strides);
    meta.ints("residual_dilations", hp.residual_dilations);
}

bool read_codec_header(const gguf_file & file, codec_header & header,
                       std::string * error) {
    if (!architecture_is(file, "audio8-codec")) {
        if (error) *error = "audio8: " + file.path() + " is not an audio8-codec GGUF";
        return false;
    }
    metadata meta(file.ctx(), "audio8.codec.");
    read_codec_hparams(meta, header.hp);
    header.part = meta.text("part");
    if (!meta.ok()) {
        if (error) *error = meta.error();
        return false;
    }
    if (header.part != "encoder" && header.part != "decoder") {
        if (error) {
            *error = "audio8: unexpected codec part '" + header.part + "' in " +
                     file.path();
        }
        return false;
    }
    return true;
}

void read_quantizer_bank(tensor_map & map, codec_model & model, bool encoding) {
    read_quantizers(map, "q/sem", 1, encoding, model.semantic_quantizers);
    read_quantizers(map, "q/res", model.hp.residual_codebooks, encoding,
                    model.residual_quantizers);
}

void read_decoder(tensor_map & map, codec_model & model) {
    const codec_hparams & hp = model.hp;
    const int dilations = static_cast<int>(hp.residual_dilations.size());
    read_quantizer_bank(map, model, /*encoding=*/false);
    read_window_transformer(map, "q/post", "post_tf", model.post);
    read_resamplers(map, "q/up", model.upsample);
    read_conv(map, "dec/0/conv", model.dec_in);
    read_decoder_stages(map, hp.decoder_strides, dilations, model.dec_stages);
    const size_t tail = hp.decoder_strides.size() + 1;
    model.dec_out_alpha = map.get("dec/" + std::to_string(tail) + "/alpha");
    read_conv(map, "dec/" + std::to_string(tail + 1) + "/conv", model.dec_out);
}

void read_encoder(tensor_map & map, codec_model & model, const transformer_spec & nested) {
    const codec_hparams & hp = model.hp;
    const int dilations = static_cast<int>(hp.residual_dilations.size());
    read_conv(map, "enc/0/conv", model.enc_in);
    read_encoder_stages(map, hp.encoder_strides, dilations, nested, model.enc_stages);
    const size_t tail = hp.encoder_strides.size() + 1;
    model.enc_out_alpha = map.get("enc/" + std::to_string(tail) + "/alpha");
    read_conv(map, "enc/" + std::to_string(tail + 1) + "/conv", model.enc_out);
    read_window_transformer(map, "q/pre", "pre_tf", model.pre);
    read_resamplers(map, "q/down", model.downsample);
    read_quantizer_bank(map, model, /*encoding=*/true);
}

// Chaining a frame into one graph buys exactly one thing: nine fewer command
// buffers to submit and wait on. The CPU backend has none to collapse -- its
// compute is a direct call -- so it would take a different greedy tie-break for
// no gain, because ggml's argmax settles a tie on the last equal logit where
// argmax_of keeps the first. It stays on the per-position path.
//
// The kernel still has to be asked for: without argmax the scheduler would
// reroute every frame through the CPU, which is slower than the path it replaces.
bool backend_picks_codes(ggml_backend_t backend, const lm_hparams & hp) {
    if (::tts_cpp::detail::backend_is_cpu(backend)) return false;
    scratch probe(PROBE_NODES);
    if (!probe.ok()) return false;
    ggml_tensor * logits = ggml_new_tensor_2d(probe.ctx, GGML_TYPE_F32, hp.codebook_size, 1);
    return ggml_backend_supports_op(backend, ggml_argmax(probe.ctx, logits));
}

bool hexagon_decode_attention(ggml_backend_t backend, int head_dim, int heads, int kv_heads,
                              int capacity) {
    // Select supported Hexagon decode attention by default, with an explicit opt-out.
    const char * flag = std::getenv("AUDIO8_LM_HEXAGON_FLASH_ATTN");
    if (flag && std::strcmp(flag, "0") == 0) return false;
    const auto device = ggml_backend_get_device(backend);
    if (!device || std::strcmp(audio8_dev_reg_name(device), "HTP") != 0 ||
        head_dim < 32 || head_dim > 256 || head_dim % 32 != 0 ||
        heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 || capacity <= 0) return false;
    scratch probe(16);
    if (!probe.ok()) return false;
    auto q = ggml_new_tensor_4d(probe.ctx, GGML_TYPE_F32, head_dim, 1, heads, 1);
    auto k = ggml_new_tensor_3d(probe.ctx, GGML_TYPE_F32, head_dim, kv_heads, capacity);
    auto v = ggml_new_tensor_3d(probe.ctx, GGML_TYPE_F32, head_dim, kv_heads, capacity);
    auto out = ggml_flash_attn_ext(probe.ctx, q, ggml_permute(probe.ctx, k, 0, 2, 1, 3),
                                  ggml_permute(probe.ctx, v, 0, 2, 1, 3), nullptr,
                                  1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    return ggml_backend_supports_op(backend, out);
}

bool coreml_sidecar_enabled() {
#ifdef TTS_CPP_USE_COREML
    return std::getenv("AUDIO8_COREML_DISABLE") == nullptr;
#else
    return false;
#endif
}

// The metadata-only load only checks for the sidecar, so a fit preflight stays a preflight.
void attach_coreml_sidecar(const std::string & gguf_path, codec_model & model,
                           bool metadata_only) {
    model.coreml = nullptr;
    model.synthesis_on_coreml = false;
    if (!model.has_decoder || !coreml_sidecar_enabled()) return;
    const std::string sidecar = coreml_codec_sidecar_path(gguf_path);
    if (!::tts_cpp::detail::coreml_sidecar_exists(sidecar)) return;
    if (metadata_only) {
        model.synthesis_on_coreml = true;
        return;
    }
#ifdef TTS_CPP_USE_COREML
    model.coreml = audio8_coreml_codec_init(sidecar.c_str(), model.hp.latent_dim,
                                            model.hp.frame_size);
    model.synthesis_on_coreml = model.coreml != nullptr;
#endif
}

}  // namespace

// Registry names and exact device names are distinct in ggml. Hexagon
// registers as "HTP" and exposes a single device "HTP0". "cpu" and "opencl"
// select by registry; "hexagon" selects by matching BOTH; anything else is
// treated as an exact ggml device name (e.g. "HTP0", "CUDA0"). Mirrors the
// Parakeet matcher in engines/parakeet/src/backend_util.h so a user who
// learned the names on one engine finds them identical on the other.
// Declared in internal.h so tests can exercise the policy without having
// to spin up a ggml device.
bool backend_selection_matches(const std::string & requested,
                               const char * reg, const char * device,
                               enum ggml_backend_dev_type type) {
    if (requested.empty() || requested == "auto") return false;
    if (requested == "cpu")    return type == GGML_BACKEND_DEVICE_TYPE_CPU;
    if (requested == "opencl") return reg && std::strcmp(reg, "OpenCL") == 0;
    if (requested == "webgpu") return ::tts_cpp::detail::reg_name_is_webgpu(reg);
    if (requested == "hexagon") {
        return reg && std::strcmp(reg, "HTP") == 0 &&
               device && std::strcmp(device, "HTP0") == 0;
    }
    return device && requested == device;
}

// Shared body of load_lm and load_lm_metadata_only. When `measure` is
// non-null the load is metadata-only: every allocation the real path makes is
// sized into `measure` instead of performed and no tensor data is read; see
// the declaration comments in internal.h.
static bool load_lm_impl(const std::string & path, int n_gpu_layers,
                         const std::string & backend, lm_model & model,
                         std::string * error, fit_load_measure * measure) {
    gguf_file file(path);
    if (!file.ok()) {
        if (error) *error = "audio8: failed to open " + path;
        return false;
    }
    if (!architecture_is(file, "audio8-lm")) {
        if (error) *error = "audio8: " + path + " is not an audio8-lm GGUF";
        return false;
    }

    metadata meta(file.ctx(), "audio8.lm.");
    read_lm_hparams(meta, model.hp);
    raw_metadata shared(file.ctx());
    // A weightless fit-measure GGUF may omit the vocabulary (a measurement
    // never tokenizes); find() would record the absent keys as errors.
    if (!measure || gguf_find_key(file.ctx(), "tokenizer.ggml.tokens") >= 0) {
        read_tokenizer(shared, model.tokenizer);
    }
    if (!meta.ok() || !shared.ok()) {
        if (error) *error = meta.ok() ? shared.error() : meta.error();
        return false;
    }
    if (model.hp.num_codebooks < 1 || model.hp.num_codebooks > AUDIO8_MAX_CODEBOOKS) {
        if (error) {
            *error = "audio8: the model declares " +
                     std::to_string(model.hp.num_codebooks) +
                     " codebooks, outside the supported 1.." +
                     std::to_string(AUDIO8_MAX_CODEBOOKS);
        }
        return false;
    }

    model.backend = init_backend(n_gpu_layers, backend);
    if (!model.backend) {
        if (error) *error = "audio8: failed to init a compute backend";
        return false;
    }
    model.precise_outputs = ::tts_cpp::detail::backend_is_metal(model.backend);
    if (!load_weights(file, model.backend, &model.ctx_w, &model.buffer_w, error,
                      measure ? &measure->weights_bytes : nullptr,
                      plan_lm_fusions(file, model.hp), &model.repacked_projection_buffers)) {
        return false;
    }

    tensor_map map(model.ctx_w);
    model.tok_emb = map.get("lm/tok_emb");
    model.codebook_emb = map.get("lm/codebook_emb");
    model.norm = map.get("lm/norm");
    model.sem_head = map.get("lm/sem_head");
    model.rope_cos = map.get("lm/rope_cos");
    model.rope_sin = map.get("lm/rope_sin");
    read_branch(map, "lm", model.hp.depth, slow_geometry(model.hp), model.blocks);
    model.fast_emb = map.get("fast/emb");
    model.fast_norm = map.get("fast/norm");
    model.fast_out = map.get("fast/out");
    model.fast_rope_cos = map.get("fast/rope_cos");
    model.fast_rope_sin = map.get("fast/rope_sin");
    read_branch(map, "fast", model.hp.fast_depth, fast_geometry(model.hp), model.fast_blocks);
    if (!map.ok()) {
        if (error) *error = map.error();
        return false;
    }

    const lm_hparams & hp = model.hp;
    size_t * kv_measure = measure ? &measure->kv_bytes : nullptr;
    model.slow_kv.flash_attention = hexagon_decode_attention(
        model.backend, hp.head_dim, hp.n_head, hp.n_kv, hp.max_seq_len);
    model.fast_kv.flash_attention = hexagon_decode_attention(
        model.backend, hp.fast_head_dim, hp.fast_n_head, hp.fast_n_kv, hp.num_codebooks);
    if (!alloc_kv(model.backend, hp.depth, hp.n_kv * hp.head_dim, hp.max_seq_len,
                  model.slow_kv, kv_measure) ||
        !alloc_kv(model.backend, hp.fast_depth, hp.fast_n_kv * hp.fast_head_dim,
                  hp.num_codebooks, model.fast_kv, kv_measure)) {
        if (error) *error = "audio8: failed to allocate the KV cache";
        return false;
    }
    ggml_backend_buffer_type_t buffer_type = ggml_backend_get_default_buffer_type(model.backend);
    model.slow_allocr = ggml_gallocr_new(buffer_type);
    model.fast_allocr = ggml_gallocr_new(buffer_type);
    model.frame_allocr = ggml_gallocr_new(buffer_type);
    if (!model.slow_allocr || !model.fast_allocr || !model.frame_allocr) {
        if (error) *error = "audio8: failed to create the graph allocators";
        return false;
    }
    model.picks_codes = backend_picks_codes(model.backend, model.hp);
    return true;
}

bool load_lm(const std::string & path, int n_gpu_layers,
             const std::string & backend, lm_model & model,
             std::string * error) {
    return load_lm_impl(path, n_gpu_layers, backend, model, error, /*measure=*/nullptr);
}

bool load_lm_metadata_only(const std::string & path, int n_gpu_layers,
                           const std::string & backend, lm_model & model,
                           fit_load_measure & measure, std::string * error) {
    measure = fit_load_measure{};
    return load_lm_impl(path, n_gpu_layers, backend, model, error, &measure);
}

void free_fast_graphs(lm_model & model) {
    for (lm_model::fast_graph & cached : model.fast_graphs) {
        if (cached.ctx) ggml_free(cached.ctx);
        if (cached.allocr) ggml_gallocr_free(cached.allocr);
    }
    model.fast_graphs.clear();
}

void free_lm(lm_model & model) {
    free_fast_graphs(model);
    ::tts_cpp::detail::sched_fallback_free(model.sched);
    if (model.slow_allocr) ggml_gallocr_free(model.slow_allocr);
    if (model.fast_allocr) ggml_gallocr_free(model.fast_allocr);
    if (model.frame_allocr) ggml_gallocr_free(model.frame_allocr);
    free_kv(model.slow_kv);
    free_kv(model.fast_kv);
    if (model.buffer_w) ggml_backend_buffer_free(model.buffer_w);
    for (auto buffer : model.repacked_projection_buffers) ggml_backend_buffer_free(buffer);
    model.repacked_projection_buffers.clear();
    if (model.ctx_w) ggml_free(model.ctx_w);
    if (model.backend) ggml_backend_free(model.backend);
    model.slow_allocr = nullptr;
    model.fast_allocr = nullptr;
    model.frame_allocr = nullptr;
    model.buffer_w = nullptr;
    model.ctx_w = nullptr;
    model.backend = nullptr;
}

bool peek_codec_header(const std::string & path, codec_header & header,
                       std::string * error) {
    gguf_file file(path);
    if (!file.ok()) {
        if (error) *error = "audio8: failed to open " + path;
        return false;
    }
    return read_codec_header(file, header, error);
}

// Shared body of load_codec and load_codec_metadata_only; same measure
// semantics as load_lm_impl.
static bool load_codec_impl(const std::string & path, int n_gpu_layers,
                            const std::string & backend, codec_model & model,
                            std::string * error, fit_load_measure * measure) {
    gguf_file file(path);
    if (!file.ok()) {
        if (error) *error = "audio8: failed to open " + path;
        return false;
    }
    codec_header header;
    if (!read_codec_header(file, header, error)) return false;
    model.hp = header.hp;
    model.has_decoder = header.part == "decoder";
    model.has_encoder = header.part == "encoder";

    metadata meta(file.ctx(), "audio8.codec.");
    transformer_spec nested;
    if (model.has_decoder) {
        read_transformer_spec(meta, "post_tf", model.post.spec);
    } else {
        read_transformer_spec(meta, "pre_tf", model.pre.spec);
        read_transformer_spec(meta, "enc_tf", nested);
    }
    if (!meta.ok()) {
        if (error) *error = meta.error();
        return false;
    }

    model.backend = init_backend(n_gpu_layers, backend);
    if (!model.backend) {
        if (error) *error = "audio8: failed to init a compute backend";
        return false;
    }
    model.precise_outputs = ::tts_cpp::detail::backend_is_metal(model.backend);
    model.synthesis_convs = {};
    const auto device = ggml_backend_get_device(model.backend);
    const char * tiled = std::getenv("AUDIO8_CODEC_HEXAGON_CONV_TILE");
    const char * fp16 = std::getenv("AUDIO8_CODEC_HEXAGON_FP16");
    const bool hexagon_decoder = model.has_decoder && device &&
                                 std::strcmp(audio8_dev_reg_name(device), "HTP") == 0;
    if (hexagon_decoder && (!fp16 || std::strcmp(fp16, "0") != 0)) {
        // Default to the validated F16 activation/HMX path for eligible
        // synthesis products. The explicit opt-out restores the F32 request.
        model.synthesis_convs.precision = GGML_PREC_DEFAULT;
        std::fprintf(stderr, "[audio8] Hexagon codec FP16 synthesis: enabled\n");
    } else if (hexagon_decoder && tiled && std::strcmp(tiled, "1") == 0) {
        const auto registry = ggml_backend_dev_backend_reg(device);
        model.synthesis_convs.backend = model.backend;
        model.synthesis_convs.tiled = reinterpret_cast<codec_conv_options::tiled_fn>(
            ggml_backend_reg_get_proc_address(registry, "ggml_backend_hexagon_conv_taps_f32"));
        std::fprintf(stderr, "[audio8] Hexagon codec F32 convolution tiling: %s\n",
                     model.synthesis_convs.tiled ? "requested" : "unavailable; using separate taps");
    }
    if (!load_weights(file, model.backend, &model.ctx_w, &model.buffer_w, error,
                      measure ? &measure->weights_bytes : nullptr)) {
        return false;
    }

    tensor_map map(model.ctx_w);
    if (model.has_decoder) {
        read_decoder(map, model);
    } else {
        read_encoder(map, model, nested);
    }
    if (!map.ok()) {
        if (error) *error = map.error();
        return false;
    }

    ggml_backend_buffer_type_t buffer_type =
        ggml_backend_get_default_buffer_type(model.backend);
    model.allocr = ggml_gallocr_new(buffer_type);
    model.block_allocr = ggml_gallocr_new(buffer_type);
    if (!model.allocr || !model.block_allocr) {
        if (error) *error = "audio8: failed to create the graph allocators";
        return false;
    }
    attach_coreml_sidecar(path, model, measure != nullptr);
    return true;
}

bool load_codec(const std::string & path, int n_gpu_layers,
                const std::string & backend, codec_model & model,
                std::string * error) {
    return load_codec_impl(path, n_gpu_layers, backend, model, error, /*measure=*/nullptr);
}

bool load_codec_metadata_only(const std::string & path, int n_gpu_layers,
                              const std::string & backend, codec_model & model,
                              fit_load_measure & measure, std::string * error) {
    measure = fit_load_measure{};
    return load_codec_impl(path, n_gpu_layers, backend, model, error, &measure);
}

void free_codec(codec_model & model) {
#ifdef TTS_CPP_USE_COREML
    if (model.coreml) audio8_coreml_codec_free(model.coreml);
#endif
    model.coreml = nullptr;
    model.synthesis_on_coreml = false;
    ::tts_cpp::detail::sched_fallback_free(model.sched);
    if (model.allocr) ggml_gallocr_free(model.allocr);
    if (model.block_allocr) ggml_gallocr_free(model.block_allocr);
    if (model.buffer_w) ggml_backend_buffer_free(model.buffer_w);
    if (model.ctx_w) ggml_free(model.ctx_w);
    if (model.backend) ggml_backend_free(model.backend);
    model.allocr = nullptr;
    model.block_allocr = nullptr;
    model.buffer_w = nullptr;
    model.ctx_w = nullptr;
    model.backend = nullptr;
}

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
