#pragma once

#include "cosyvoice_mmap.h"

#include "ggml-backend.h"

#include <string>

struct gguf_context;

namespace tts_cpp::detail {
class gguf_stream_reader;
}

namespace tts_cpp::moss::detail {

class HostTables {
public:
    explicit HostTables(const char * owner);
    ~HostTables();
    HostTables(const HostTables &) = delete;
    HostTables & operator=(const HostTables &) = delete;

    void place(ggml_context * tables, gguf_context * file, const std::string & path,
               ::tts_cpp::detail::gguf_stream_reader & reader);
    ggml_backend_buffer_t buffer() const;

private:
    const uint8_t * mapped_address(gguf_context * file, const ggml_tensor * tensor) const;
    bool mappable(ggml_context * tables, gguf_context * file) const;
    void bind(ggml_context * tables, gguf_context * file);
    bool map(ggml_context * tables, gguf_context * file, const std::string & path);
    void read(ggml_context * tables, ::tts_cpp::detail::gguf_stream_reader & reader);
    void allocate(ggml_context * tables);
    void stream(ggml_context * tables, ::tts_cpp::detail::gguf_stream_reader & reader) const;
    [[noreturn]] void fail(const std::string & message) const;

    std::string owner_;
    ::tts_cpp::cosyvoice::MappedFile mapping_;
    ggml_backend_buffer_t buffer_ = nullptr;
};

} // namespace tts_cpp::moss::detail
