#include "moss/host_tables.h"

#include "gguf_stream.h"

#include "gguf.h"

#include <stdexcept>

namespace tts_cpp::moss::detail {
namespace {

constexpr size_t HOST_TENSOR_ALIGNMENT = 32;

} // namespace

HostTables::HostTables(const char * owner) : owner_(owner) {}

HostTables::~HostTables() {
    release();
}

void HostTables::release() {
    if (buffer_) ggml_backend_buffer_free(buffer_);
    buffer_ = nullptr;
    ::tts_cpp::cosyvoice::mapped_file_close(mapping_);
}

ggml_backend_buffer_t HostTables::buffer() const {
    return buffer_;
}

void HostTables::fail(const std::string & message) const {
    throw std::runtime_error(owner_ + ": " + message);
}

const uint8_t * HostTables::mapped_address(gguf_context * file, const ggml_tensor * tensor) const {
    const int64_t id = gguf_find_tensor(file, ggml_get_name(tensor));
    if (id < 0) {
        return nullptr;
    }
    const size_t offset = gguf_get_data_offset(file) + gguf_get_tensor_offset(file, id);
    if (offset > mapping_.size || ggml_nbytes(tensor) > mapping_.size - offset) {
        return nullptr;
    }
    const uint8_t * address = mapping_.data + offset;
    return (uintptr_t) address % HOST_TENSOR_ALIGNMENT == 0 ? address : nullptr;
}

bool HostTables::mappable(ggml_context * tables, gguf_context * file) const {
    for (auto * t = ggml_get_first_tensor(tables); t; t = ggml_get_next_tensor(tables, t)) {
        if (!mapped_address(file, t)) {
            return false;
        }
    }
    return true;
}

void HostTables::bind(ggml_context * tables, gguf_context * file) {
    for (auto * t = ggml_get_first_tensor(tables); t; t = ggml_get_next_tensor(tables, t)) {
        if (ggml_backend_tensor_alloc(buffer_, t, (void *) mapped_address(file, t)) != GGML_STATUS_SUCCESS) {
            fail(std::string("cannot map tensor: ") + ggml_get_name(t));
        }
    }
}

bool HostTables::map(ggml_context * tables, gguf_context * file, const std::string & path) {
    if (!::tts_cpp::cosyvoice::mapped_file_open(mapping_, path, owner_.c_str())) {
        return false;
    }
    if (mappable(tables, file)) {
        buffer_ = ggml_backend_cpu_buffer_from_ptr((void *) mapping_.data, mapping_.size);
    }
    if (!buffer_) {
        ::tts_cpp::cosyvoice::mapped_file_close(mapping_);
        return false;
    }
    bind(tables, file);
    return true;
}

void HostTables::allocate(ggml_context * tables) {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    buffer_ = cpu ? ggml_backend_alloc_ctx_tensors_from_buft(tables, ggml_backend_dev_buffer_type(cpu)) : nullptr;
    if (!buffer_) {
        fail("host table allocation failed");
    }
}

void HostTables::stream(ggml_context * tables, ::tts_cpp::detail::gguf_stream_reader & reader) const {
    for (auto * t = ggml_get_first_tensor(tables); t; t = ggml_get_next_tensor(tables, t)) {
        if (!reader.to_backend(ggml_get_name(t), t)) {
            fail(std::string("failed to load tensor: ") + ggml_get_name(t));
        }
    }
}

void HostTables::read(ggml_context * tables, ::tts_cpp::detail::gguf_stream_reader & reader) {
    allocate(tables);
    stream(tables, reader);
}

void HostTables::place(ggml_context * tables, gguf_context * file, const std::string & path,
                       ::tts_cpp::detail::gguf_stream_reader & reader) {
    if (!map(tables, file, path)) {
        read(tables, reader);
    }
}

} // namespace tts_cpp::moss::detail
