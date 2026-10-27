#include "long_form.h"
#include "parakeet_ctc.h"
#include "parakeet_tdt.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using parakeet::ParakeetCtcModel;
using parakeet::TdtEmbeddingRows;
using parakeet::TdtRuntimeWeights;
using parakeet::tdt_embedding_rows;
using parakeet::tdt_embedding_rows_from_bytes;
using parakeet::tdt_enc_proj_capacity;

namespace {

constexpr int     kDefaultPositionRange = 5000;
constexpr int     kShortPositionRange   = 1000;
constexpr int     kWidePositionRange    = 10000;
constexpr int     kOversizedHint        = 5000;
constexpr int64_t kRows                 = 5;
constexpr int64_t kWidth                = 64;
constexpr int64_t kUnalignedWidth       = 30;
constexpr float   kValueStep            = 0.37f;
constexpr size_t  kContextBytes         = 1024 * 1024;

int g_failures = 0;

void expect(bool cond, const std::string & what) {
    if (cond) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

ParakeetCtcModel model_with_window(int window_hint, int position_range) {
    ParakeetCtcModel model;
    model.encoder_cfg.long_form_window_hint = window_hint;
    model.encoder_cfg.pos_emb_max_len       = position_range;
    return model;
}

void check_enc_proj_capacity() {
    expect(tdt_enc_proj_capacity(model_with_window(parakeet::kTranscribePsyWindowFrames,
                                                   kDefaultPositionRange)) ==
               parakeet::kTranscribePsyWindowFrames,
           "capacity: the model's recommended window sizes the projection");
    expect(tdt_enc_proj_capacity(model_with_window(0, kDefaultPositionRange)) ==
               parakeet::kLongFormAutoWindowFrames,
           "capacity: no recommendation takes the automatic window");
    expect(tdt_enc_proj_capacity(model_with_window(0, kShortPositionRange)) == kShortPositionRange,
           "capacity: the positional range caps the window");
    expect(tdt_enc_proj_capacity(model_with_window(kOversizedHint, kWidePositionRange)) ==
               TdtRuntimeWeights::k_enc_proj_T_max,
           "capacity: the projection never exceeds its ceiling");
}

std::vector<float> table_values() {
    std::vector<float> values((size_t) (kRows * kWidth));
    for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin((float) i * kValueStep);
    return values;
}

std::vector<uint8_t> quantize_q8_0(const std::vector<float> & values) {
    std::vector<uint8_t> bytes(ggml_row_size(GGML_TYPE_Q8_0, kWidth) * (size_t) kRows);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(), bytes.data(), 0, kRows, kWidth, nullptr);
    return bytes;
}

std::vector<uint8_t> encode_f16(const std::vector<float> & values) {
    std::vector<ggml_fp16_t> halves(values.size());
    ggml_fp32_to_fp16_row(values.data(), halves.data(), (int64_t) values.size());
    const auto * first = reinterpret_cast<const uint8_t *>(halves.data());
    return std::vector<uint8_t>(first, first + halves.size() * sizeof(ggml_fp16_t));
}

std::vector<uint8_t> encode_f32(const std::vector<float> & values) {
    const auto * first = reinterpret_cast<const uint8_t *>(values.data());
    return std::vector<uint8_t>(first, first + values.size() * sizeof(float));
}

std::vector<float> dequantize(ggml_type type, const std::vector<uint8_t> & bytes) {
    std::vector<float> values((size_t) (kRows * kWidth));
    if (type == GGML_TYPE_F32) {
        std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
        return values;
    }
    ggml_get_type_traits(type)->to_float(bytes.data(), values.data(), (int64_t) values.size());
    return values;
}

bool rows_match(const TdtEmbeddingRows & rows, const std::vector<float> & expected) {
    std::vector<float> row((size_t) kWidth);
    for (int64_t r = 0; r < kRows; ++r) {
        if (!rows.read(r, row.data())) return false;
        if (std::memcmp(row.data(), expected.data() + r * kWidth, row.size() * sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}

void check_rows_match_dequantised(ggml_type type, const std::vector<uint8_t> & bytes) {
    const std::string name = ggml_type_name(type);
    const std::vector<float> expected = dequantize(type, bytes);
    const TdtEmbeddingRows rows = tdt_embedding_rows_from_bytes(type, kRows, kWidth, bytes);
    expect(rows.rows == kRows && rows.width == kWidth, "rows: " + name + " table shape");
    expect(rows_match(rows, expected), "rows: " + name + " rows equal the dequantised table");
}

void check_row_bounds() {
    const TdtEmbeddingRows rows =
        tdt_embedding_rows_from_bytes(GGML_TYPE_F32, kRows, kWidth, encode_f32(table_values()));
    std::vector<float> row((size_t) kWidth);
    expect(!rows.read(-1, row.data()), "bounds: a negative row is refused");
    expect(!rows.read(kRows, row.data()), "bounds: a row past the table is refused");
    expect(tdt_embedding_rows_from_bytes(GGML_TYPE_Q8_0, kRows, kUnalignedWidth,
                                         quantize_q8_0(table_values())).rows == 0,
           "bounds: a width off the q8_0 block size is refused");
    expect(tdt_embedding_rows_from_bytes(GGML_TYPE_F32, kRows, kWidth, {}).rows == 0,
           "bounds: a byte buffer shorter than the table is refused");
}

void check_borrowed_tensor_rows() {
    ggml_init_params params{};
    params.mem_size = kContextBytes;
    ggml_context * ctx = ggml_init(params);
    const std::vector<uint8_t> bytes = quantize_q8_0(table_values());
    ggml_tensor * table = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, kWidth, kRows);
    std::memcpy(table->data, bytes.data(), bytes.size());

    const TdtEmbeddingRows rows = tdt_embedding_rows(table);
    expect(rows.borrowed == table->data && rows.owned.empty(),
           "tensor: host rows are read in place, not copied");
    expect(rows_match(rows, dequantize(GGML_TYPE_Q8_0, bytes)),
           "tensor: borrowed q8_0 rows equal the dequantised table");

    ggml_tensor * cube = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kWidth, kRows, kRows);
    expect(tdt_embedding_rows(cube).rows == 0, "tensor: a 3-D tensor is not an embedding table");
    ggml_free(ctx);
}

}  // namespace

int main() {
    check_enc_proj_capacity();
    check_rows_match_dequantised(GGML_TYPE_F32, encode_f32(table_values()));
    check_rows_match_dequantised(GGML_TYPE_F16, encode_f16(table_values()));
    check_rows_match_dequantised(GGML_TYPE_Q8_0, quantize_q8_0(table_values()));
    check_row_bounds();
    check_borrowed_tensor_rows();

    if (g_failures == 0) {
        std::printf("test-tdt-runtime-memory: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test-tdt-runtime-memory: %d failure(s)\n", g_failures);
    return 1;
}
