#include "whisper.h"

#include <cstdint>
#include <cstdio>
#include <climits>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

static std::string log_text;

static void capture_log(enum ggml_log_level, const char * text, void *) {
    log_text += text;
}

// Model files are little-endian, including on the big-endian CI targets.
static void set_u32(std::vector<uint8_t> & model, size_t offset, uint32_t value) {
    assert(offset + 4 <= model.size());
    for (int i = 0; i < 4; ++i) {
        model[offset + i] = uint8_t(value >> (8*i));
    }
}

static uint32_t get_u32(const std::vector<uint8_t> & model, size_t offset) {
    assert(offset + 4 <= model.size());
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        value |= uint32_t(model[offset + i]) << (8*i);
    }
    return value;
}

static void check_model(std::vector<uint8_t> model, bool valid, const char * error = "invalid model") {
    {
        std::ofstream file(WHISPER_TEST_PATH, std::ios::binary);
        assert(file.good());
        file.write(reinterpret_cast<const char *>(model.data()), model.size());
        assert(file.good());
    }

    auto params = whisper_context_default_params();
    params.use_gpu = false;

    // Check both public entry points, with and without state initialization.
    // An abort/divide-by-zero fails the test, rather than counting as rejection.
    for (int entry = 0; entry < 4; ++entry) {
        log_text.clear();
        whisper_context * ctx = nullptr;
        switch (entry) {
            case 0: ctx = whisper_init_from_buffer_with_params_no_state(model.data(), model.size(), params); break;
            case 1: ctx = whisper_init_from_file_with_params_no_state(WHISPER_TEST_PATH, params); break;
            case 2: ctx = whisper_init_from_buffer_with_params(model.data(), model.size(), params); break;
            case 3: ctx = whisper_init_from_file_with_params(WHISPER_TEST_PATH, params); break;
        }
        if ((ctx != nullptr) != valid || (!valid && log_text.find(error) == std::string::npos)) {
            fprintf(stderr, "entry %d, expected %s:\n%s", entry, valid ? "success" : error, log_text.c_str());
            assert(false);
        }
        whisper_free(ctx);
    }
    assert(std::remove(WHISPER_TEST_PATH) == 0);
}

int main() {
    ggml_backend_load_all();
    whisper_log_set(capture_log, nullptr);

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;

    struct whisper_context * ctx_empty = whisper_init_from_buffer_with_params(nullptr, 1, cparams);
    assert(ctx_empty == nullptr);

    uint8_t truncated[8] = { 0 };
    struct whisper_context * ctx_trunc = whisper_init_from_buffer_with_params(truncated, sizeof(truncated), cparams);
    assert(ctx_trunc == nullptr);

    std::ifstream file(WHISPER_MODEL_PATH, std::ios::binary);
    assert(file.good());
    const std::vector<uint8_t> base((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    assert(base.size() > 56);
    check_model(base, true);

    // All 13 fields, including both filter dimensions and packed ftype.
    const int32_t maxima[] = { 65536, 2048, 2048, 32, 32, 512, 2048, 32, 32, 128,
                              (GGML_QNT_VERSION + 1)*GGML_QNT_VERSION_FACTOR - 1, 128, 201 };
    for (size_t field = 0; field < 13; ++field) {
        const int32_t values[] = { 0, -1, INT32_MIN, 1 << 30, INT32_MAX, maxima[field] + 1 };
        for (int32_t value : values) {
            if (field == 10 && value == 0) {
                continue; // F32 is valid and is tested below.
            }
            auto model = base;
            set_u32(model, 4*(field + 1), uint32_t(value));
            check_model(model, false);
        }
    }

    // Small positive values also need relational and quantization checks.
    for (size_t offset : { 16, 32 }) {
        auto model = base;
        set_u32(model, offset, 5); // 384 is not divisible by 5.
        check_model(model, false, "incompatible state and head dimensions");
    }
    {
        auto model = base;
        set_u32(model, 28, 2);
        check_model(model, false, "incompatible state and head dimensions");
        set_u32(model, 12, 30);
        set_u32(model, 28, 30); // divisible by 6, but not by a Q5_1 block.
        set_u32(model, 44, 1000 + GGML_FTYPE_MOSTLY_Q5_1);
        check_model(model, false, "weight block size");
    }
    for (uint32_t ftype : { 4u, 5u, 6u, 999u, 1004u, 2006u, 3001u, uint32_t(-1000) }) {
        auto model = base;
        set_u32(model, 44, ftype);
        check_model(model, false, "bad ftype");
    }
    for (size_t offset : { 48, 52 }) {
        auto model = base;
        set_u32(model, offset, offset == 48 ? 79 : 200);
        check_model(model, false, "bad mel filter dimensions");
    }

    // The serialized vocabulary count is separate from the header's n_vocab.
    const size_t vocab_offset = 56 + 4*get_u32(base, 48)*get_u32(base, 52);
    for (uint32_t count : { uint32_t(-1), get_u32(base, 4) + 1, uint32_t(INT32_MAX) }) {
        auto model = base;
        set_u32(model, vocab_offset, count);
        check_model(model, false, "bad vocab size");
    }
    for (uint32_t length : { 1025u, 1u << 30, UINT32_MAX }) {
        auto model = base;
        set_u32(model, vocab_offset + 4, length);
        check_model(model, false, "too long");
    }
    for (uint32_t length : { 0u, 1024u }) {
        auto model = base;
        const size_t token_offset = vocab_offset + 8;
        const size_t token_length = get_u32(base, vocab_offset + 4);
        model.erase(model.begin() + token_offset, model.begin() + token_offset + token_length);
        model.insert(model.begin() + token_offset, length, 'x');
        set_u32(model, vocab_offset + 4, length);
        check_model(model, true);
    }

    // Accept real large-v3/turbo/distilled headers without allocating their GBs
    // of weights: an intentionally invalid filter must be the first rejection.
    const int32_t profiles[][10] = {
        { 51865, 1500, 1280, 20, 32, 448, 1280, 20, 32,  80 }, // large-v2
        { 51866, 1500, 1280, 20, 32, 448, 1280, 20, 32, 128 }, // large-v3
        { 51866, 1500, 1280, 20, 32, 448, 1280, 20,  4, 128 }, // turbo
        { 51866, 1500, 1280, 20, 32, 448, 1280, 20,  2, 128 }, // distil-large-v3
        { 51864, 1500,  768, 12, 12, 448,  768, 12,  4,  80 }, // distil-small.en
        { 65536, 2048, 2048, 32, 32, 512, 2048, 32, 32, 128 }, // inclusive bounds
    };
    for (const auto & profile : profiles) {
        auto model = base;
        for (size_t i = 0; i < 10; ++i) {
            set_u32(model, 4*(i + 1), profile[i]);
        }
        set_u32(model, 48, profile[9]);
        set_u32(model, 52, 0);
        check_model(model, false, "bad mel filter dimensions");
    }

    // Keep the full set of existing weight types and legacy version encodings.
    for (int32_t version = 0; version <= GGML_QNT_VERSION; ++version) {
        for (int32_t type = 0; type <= GGML_FTYPE_MOSTLY_Q2_0; ++type) {
            if (type == 4 || type == 5 || type == 6) {
                continue;
            }
            auto model = base;
            set_u32(model, 12, 1024);
            set_u32(model, 16, 16);
            set_u32(model, 28, 1024);
            set_u32(model, 32, 16);
            set_u32(model, 44, version*GGML_QNT_VERSION_FACTOR + type);
            set_u32(model, 52, 0);
            check_model(model, false, "bad mel filter dimensions");
        }
    }

    // A complete 128-mel model with unequal layer counts and legacy Q5_1.
    // Empty weights are intentionally supported by the repository's fixtures.
    auto model = base;
    model.insert(model.begin() + 56, 4*(128 - 80)*201, 0);
    set_u32(model, 4, 51866);
    set_u32(model, 36, 2);
    set_u32(model, 40, 128);
    set_u32(model, 48, 128);
    set_u32(model, 44, 1000 + GGML_FTYPE_MOSTLY_Q5_1);
    check_model(model, true);
    set_u32(model, 44, GGML_FTYPE_ALL_F32);
    check_model(model, true);

    printf("test-whisper-buffer-loader: OK\n");
    return 0;
}
