#pragma once

#include "ggml.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

struct gguf_context;

namespace tts_cpp::moss::detail {

class GgufRows {
public:
    GgufRows(const std::string & path, const gguf_context * file, const ggml_tensor * table, const char * owner);

    std::vector<float> gather(const std::vector<int32_t> & ids);

private:
    void locate(const gguf_context * file, const ggml_tensor * table);
    void open();
    bool file_holds_table() const;
    void read_row(int32_t id, float * out);
    void convert_row(float * out) const;
    [[noreturn]] void fail(const std::string & message) const;

    std::string owner_;
    std::string path_;
    std::ifstream stream_;
    uint64_t offset_ = 0;
    ggml_type type_ = GGML_TYPE_F32;
    int64_t width_ = 0;
    int64_t rows_ = 0;
    std::vector<uint8_t> row_;
};

} // namespace tts_cpp::moss::detail
