#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_sched;
struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;
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

ggml_backend * open_transcribe_backend(bool use_gpu, const std::string & requested);
void upload_gguf_tensors(const gguf_context * file, const std::string & path, ggml_context * weights);

} // namespace parakeet::moss::detail
