#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_sched;
struct gguf_context;

namespace parakeet::moss::detail {

constexpr size_t STREAM_CHUNK_BYTES = 8u << 20;

enum class TensorUpload { InPlace, Whole, Chunked };

TensorUpload plan_upload(bool host_buffer, bool quantized);

struct TranscribeMemory {
    uint64_t device_bytes = 0;
    uint64_t host_compute_bytes = 0;
    uint64_t host_bytes = 0;
    uint64_t cpu_work_bytes = 0;
};

class TranscribeGraph {
public:
    explicit TranscribeGraph(int max_nodes);
    ~TranscribeGraph();
    TranscribeGraph(const TranscribeGraph &) = delete;
    TranscribeGraph & operator=(const TranscribeGraph &) = delete;

    ggml_context * ctx() const;
    ggml_cgraph * graph() const;
    ggml_tensor * input_f32(int64_t ne0, int64_t ne1);
    ggml_tensor * input_f16(int64_t ne0, int64_t ne1);
    ggml_tensor * input_i32(int64_t ne0);

private:
    ggml_context * ctx_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
};

class TranscribeScheduler {
public:
    TranscribeScheduler(ggml_backend * backend, int max_nodes);
    ~TranscribeScheduler();
    TranscribeScheduler(const TranscribeScheduler &) = delete;
    TranscribeScheduler & operator=(const TranscribeScheduler &) = delete;

    bool allocate(ggml_cgraph * graph);
    TranscribeMemory measure(ggml_cgraph *graph, int n_threads);
    TranscribeMemory allocated_memory() const;
    bool compute(ggml_cgraph * graph, int n_threads);

private:
    ggml_backend * backend_ = nullptr;
    ggml_backend * cpu_ = nullptr;
    ggml_backend_sched * sched_ = nullptr;
};

struct TensorSource {
    ggml_tensor * tensor = nullptr;
    size_t offset = 0;
};

class GgufRows {
public:
    GgufRows(const std::string & path, size_t offset, ggml_type type, int64_t width, int64_t rows);
    void dequantize(int64_t row, float * out);

private:
    std::ifstream file_;
    size_t offset_ = 0;
    ggml_type type_;
    int64_t width_ = 0;
    int64_t rows_ = 0;
    std::vector<uint8_t> row_;
};

using CopySupport = bool (*)(ggml_backend * backend, const ggml_tensor * copy);

// Adreno OpenCL refuses a strided copy source, and the CPU cannot write a
// cache that lives in that backend's buffer; such a source is materialized
// first. Copies the backend accepts keep their single node.
ggml_tensor * copy_into_cache(ggml_context * ctx, ggml_backend * backend, ggml_tensor * source,
                              ggml_tensor * destination, CopySupport supports = ggml_backend_supports_op);

ggml_backend * open_transcribe_backend(bool use_gpu, const std::string & requested);
size_t gguf_tensor_data_offset(const gguf_context * file, const char * name, size_t expected_bytes);
void upload_file_ranges(const std::string & path, const std::vector<TensorSource> & sources);
void upload_gguf_tensors(const gguf_context * file, const std::string & path, ggml_context * weights);

} // namespace parakeet::moss::detail
