#include "moss/gguf_rows.h"

#include "gguf.h"

#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <system_error>

namespace tts_cpp::moss::detail {

GgufRows::GgufRows(const std::string & path, const gguf_context * file, const ggml_tensor * table,
                   const char * owner)
    : owner_(owner), path_(path) {
    locate(file, table);
    open();
}

void GgufRows::fail(const std::string & message) const {
    throw std::runtime_error(owner_ + ": " + message);
}

void GgufRows::locate(const gguf_context * file, const ggml_tensor * table) {
    const int64_t id = gguf_find_tensor(file, ggml_get_name(table));
    if (id < 0 || gguf_get_tensor_size(file, id) != ggml_nbytes(table) || table->ne[2] != 1 || table->ne[3] != 1) {
        fail(std::string("cannot locate the rows of ") + ggml_get_name(table));
    }
    type_ = table->type;
    if (type_ != GGML_TYPE_F32 && !ggml_get_type_traits(type_)->to_float) {
        fail(std::string("cannot read rows of type ") + ggml_type_name(type_));
    }
    offset_ = gguf_get_data_offset(file) + gguf_get_tensor_offset(file, id);
    width_ = table->ne[0];
    rows_ = table->ne[1];
    row_.resize(ggml_row_size(type_, width_));
}

bool GgufRows::file_holds_table() const {
    std::error_code error;
    const uint64_t size = std::filesystem::file_size(path_, error);
    return !error && size >= offset_ && (uint64_t) rows_ * row_.size() <= size - offset_;
}

void GgufRows::open() {
    if (!file_holds_table()) {
        fail("the token table extends past the end of " + path_);
    }
    stream_.open(path_, std::ios::binary);
    if (!stream_) {
        fail("cannot open " + path_);
    }
}

void GgufRows::convert_row(float * out) const {
    if (type_ == GGML_TYPE_F32) {
        std::memcpy(out, row_.data(), row_.size());
        return;
    }
    ggml_get_type_traits(type_)->to_float(row_.data(), out, width_);
}

void GgufRows::read_row(int32_t id, float * out) {
    if (id < 0 || id >= rows_) {
        fail("token row " + std::to_string(id) + " is outside the table");
    }
    stream_.clear();
    stream_.seekg((std::streamoff) (offset_ + (uint64_t) id * row_.size()));
    stream_.read(reinterpret_cast<char *>(row_.data()), (std::streamsize) row_.size());
    if (!stream_) {
        fail("cannot read token row " + std::to_string(id) + "; the model file changed while loaded");
    }
    convert_row(out);
}

std::vector<float> GgufRows::gather(const std::vector<int32_t> & ids) {
    if (!file_holds_table()) {
        fail("cannot read token rows; the model file changed while loaded");
    }
    std::vector<float> rows(ids.size() * (size_t) width_);
    for (size_t i = 0; i < ids.size(); ++i) {
        read_row(ids[i], rows.data() + i * (size_t) width_);
    }
    return rows;
}

} // namespace tts_cpp::moss::detail
