#include "moss/transcribe_runtime.h"

#include "backend_util.h"
#include "parakeet_ctc.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace parakeet::moss::detail {
namespace {


[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss transcribe: " + message);
}

void read_exactly(std::ifstream & file, void * data, size_t bytes, const char * name) {
    if (bytes != 0 && !file.read(static_cast<char *>(data), (std::streamsize) bytes)) {
        fail(std::string("short read on tensor ") + name);
    }
}

void read_in_place(std::ifstream & file, ggml_tensor * tensor, size_t bytes) {
    read_exactly(file, tensor->data, bytes, ggml_get_name(tensor));
}

void upload_whole(std::ifstream & file, ggml_tensor * tensor, size_t bytes, std::vector<uint8_t> & scratch) {
    scratch.resize(bytes);
    read_exactly(file, scratch.data(), bytes, ggml_get_name(tensor));
    ggml_backend_tensor_set(tensor, scratch.data(), 0, bytes);
}

void upload_chunked(std::ifstream & file, ggml_tensor * tensor, size_t bytes, std::vector<uint8_t> & scratch) {
    scratch.resize(std::min(bytes, STREAM_CHUNK_BYTES));
    for (size_t done = 0; done < bytes; done += scratch.size()) {
        const size_t count = std::min(scratch.size(), bytes - done);
        read_exactly(file, scratch.data(), count, ggml_get_name(tensor));
        ggml_backend_tensor_set(tensor, scratch.data(), done, count);
    }
}

void seek(std::ifstream & file, size_t offset, const char * name) {
    if (!file.seekg((std::streamoff) offset)) {
        fail(std::string("cannot seek to tensor ") + name);
    }
}

void upload_tensor(std::ifstream & file, const TensorSource & source, std::vector<uint8_t> & scratch) {
    ggml_tensor * tensor = source.tensor;
    const size_t bytes = ggml_nbytes(tensor);
    seek(file, source.offset, ggml_get_name(tensor));
    switch (plan_upload(ggml_backend_buffer_is_host(tensor->buffer), ggml_is_quantized(tensor->type))) {
        case TensorUpload::InPlace: read_in_place(file, tensor, bytes); break;
        case TensorUpload::Whole:   upload_whole(file, tensor, bytes, scratch); break;
        case TensorUpload::Chunked: upload_chunked(file, tensor, bytes, scratch); break;
    }
}

} // namespace

TensorUpload plan_upload(bool host_buffer, bool quantized) {
    if (host_buffer) {
        return TensorUpload::InPlace;
    }
    return quantized ? TensorUpload::Whole : TensorUpload::Chunked;
}

TranscribeGraph::TranscribeGraph(int max_nodes) {
    const size_t bytes = (size_t) max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false);
    ctx_ = ggml_init({bytes, nullptr, true});
    if (!ctx_) {
        fail("graph context allocation failed");
    }
    graph_ = ggml_new_graph_custom(ctx_, max_nodes, false);
}

TranscribeGraph::~TranscribeGraph() {
    ggml_free(ctx_);
}

ggml_context * TranscribeGraph::ctx() const { return ctx_; }
ggml_cgraph * TranscribeGraph::graph() const { return graph_; }

ggml_tensor * TranscribeGraph::input_f32(int64_t ne0, int64_t ne1) {
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, ne0, ne1);
    ggml_set_input(tensor);
    return tensor;
}

ggml_tensor * TranscribeGraph::input_f16(int64_t ne0, int64_t ne1) {
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx_, GGML_TYPE_F16, ne0, ne1);
    ggml_set_input(tensor);
    return tensor;
}

ggml_tensor * TranscribeGraph::input_i32(int64_t ne0) {
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, ne0);
    ggml_set_input(tensor);
    return tensor;
}

ggml_tensor * TranscribeGraph::input_i64(int64_t ne0) {
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx_, GGML_TYPE_I64, ne0);
    ggml_set_input(tensor);
    return tensor;
}

TranscribeScheduler::TranscribeScheduler(ggml_backend * backend, int max_nodes) : backend_(backend) {
    ggml_backend_t backends[2] = {backend, nullptr};
    int count = 1;
    if (!backend_is_cpu(backend)) {
        cpu_ = init_engine_cpu_backend();
        if (!cpu_) {
            fail("cannot open the CPU backend for the scheduler");
        }
        backends[count++] = cpu_;
    }
    sched_ = ggml_backend_sched_new(backends, nullptr, count, (size_t) max_nodes, false, false);
    if (!sched_) {
        fail("scheduler initialization failed");
    }
}

TranscribeScheduler::~TranscribeScheduler() {
    if (sched_) ggml_backend_sched_free(sched_);
    if (cpu_) ggml_backend_free(cpu_);
}

bool TranscribeScheduler::allocate(ggml_cgraph * graph) {
    ggml_backend_sched_reset(sched_);
    return ggml_backend_sched_alloc_graph(sched_, graph);
}

TranscribeMemory TranscribeScheduler::measure(ggml_cgraph *graph,
                                              int n_threads) {
  auto *cpu = cpu_ ? cpu_ : backend_;
  backend_set_n_threads(cpu, n_threads);
  size_t sizes[2] = {0, 0};
  if (!ggml_backend_sched_reserve_size(sched_, graph, sizes)) {
    fail("scheduler measurement failed");
  }
  return {sizes[0], sizes[1], ggml_backend_sched_get_host_size(sched_),
          ggml_backend_sched_get_work_size(sched_, cpu)};
}

TranscribeMemory TranscribeScheduler::allocated_memory() const {
  return {ggml_backend_sched_get_buffer_size(sched_, backend_),
          cpu_ ? ggml_backend_sched_get_buffer_size(sched_, cpu_) : 0,
          ggml_backend_sched_get_host_size(sched_),
          ggml_backend_get_work_size(cpu_ ? cpu_ : backend_)};
}

bool TranscribeScheduler::compute(ggml_cgraph * graph, int n_threads) {
    backend_set_n_threads(cpu_ ? cpu_ : backend_, n_threads);
    return ggml_backend_sched_graph_compute(sched_, graph) == GGML_STATUS_SUCCESS;
}

ggml_tensor * copy_into_cache(ggml_context * ctx, ggml_backend * backend, ggml_tensor * source,
                              ggml_tensor * destination, CopySupport supports) {
    ggml_tensor * copy = ggml_cpy(ctx, source, destination);
    if (ggml_is_contiguous(source) || supports(backend, copy)) {
        return copy;
    }
    return ggml_cpy(ctx, ggml_cont(ctx, source), destination);
}

ggml_backend * open_transcribe_backend(bool use_gpu, const std::string & requested) {
    ggml_backend_t backend = init_engine_backend(use_gpu, requested);
    if (!backend) {
        fail("no compute backend available for request: " + requested);
    }
    return backend;
}

size_t gguf_tensor_data_offset(const gguf_context * file, const char * name, size_t expected_bytes) {
    const int64_t id = gguf_find_tensor(file, name);
    if (id < 0 || gguf_get_tensor_size(file, id) != expected_bytes) {
        fail(std::string("GGUF tensor does not match the model: ") + name);
    }
    return gguf_get_data_offset(file) + gguf_get_tensor_offset(file, id);
}

void upload_file_ranges(const std::string & path, const std::vector<TensorSource> & sources) {
    std::ifstream handle(path, std::ios::binary);
    if (!handle) {
        fail("cannot reopen GGUF for streaming: " + path);
    }
    std::vector<uint8_t> scratch;
    for (const TensorSource & source : sources) {
        upload_tensor(handle, source, scratch);
    }
}

void upload_gguf_tensors(const gguf_context * file, const std::string & path, ggml_context * weights) {
    std::vector<TensorSource> sources;
    for (ggml_tensor * tensor = ggml_get_first_tensor(weights); tensor; tensor = ggml_get_next_tensor(weights, tensor)) {
        sources.push_back({tensor, gguf_tensor_data_offset(file, ggml_get_name(tensor), ggml_nbytes(tensor))});
    }
    upload_file_ranges(path, sources);
}

GgufRows::GgufRows(const std::string & path, size_t offset, ggml_type type, int64_t width, int64_t rows)
    : file_(path, std::ios::binary), offset_(offset), type_(type), width_(width), rows_(rows),
      row_(ggml_row_size(type, width)) {
    if (!file_) {
        fail("cannot reopen GGUF for embedding lookup: " + path);
    }
    if (type != GGML_TYPE_F32 && !ggml_get_type_traits(type)->to_float) {
        fail(std::string("cannot dequantize ") + ggml_type_name(type) + " embedding rows");
    }
}

void GgufRows::dequantize(int64_t row, float * out) {
    if (row < 0 || row >= rows_) {
        fail("token id outside the text vocabulary");
    }
    seek(file_, offset_ + (size_t) row * row_.size(), "text embedding row");
    read_exactly(file_, row_.data(), row_.size(), "text embedding row");
    if (type_ == GGML_TYPE_F32) {
        std::memcpy(out, row_.data(), row_.size());
        return;
    }
    ggml_get_type_traits(type_)->to_float(row_.data(), out, width_);
}

} // namespace parakeet::moss::detail
