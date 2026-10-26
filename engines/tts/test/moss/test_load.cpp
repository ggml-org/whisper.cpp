#include "gguf_fixtures.h"

#include "moss/codec.h"
#include "moss/delay_lm.h"
#include "moss/reference_wav.h"
#include "tts-cpp/moss/engine.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>

using tts_cpp::moss::detail::Codec;
using tts_cpp::moss::detail::DelayLM;
using tts_cpp::moss::detail::DelayLogits;
using tts_cpp::moss::detail::DelayRow;
using tts_cpp::moss::detail::require_reference_total;
using tts_cpp::moss::detail::validate_reference_shape;
using tts_cpp::moss::detail::WavShape;
using namespace moss_fixtures;

namespace {

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

void expect_backbone_failure(const char * name,
        const std::function<void(gguf_context *)> & mutate, const std::string & expected) {
    const auto path = write_backbone(name, mutate);
    try {
        DelayLM model(path.string(), {}, 1, 64);
        check(false, std::string(name) + ": accepted invalid GGUF");
    } catch (const std::runtime_error & e) {
        check(std::string(e.what()).find(expected) != std::string::npos,
                std::string(name) + ": wrong failure: " + e.what());
    }
    std::filesystem::remove(path);
}

void expect_decoder_failure(const char * name, int64_t qkv_rows,
        const std::function<void(gguf_context *)> & mutate_meta, const std::string & expected) {
    const auto path = write_decoder(name, N_VQ, qkv_rows, mutate_meta);
    try {
        Codec codec(path.string(), {}, 1);
        check(false, std::string(name) + ": accepted invalid GGUF");
    } catch (const std::runtime_error & e) {
        check(std::string(e.what()).find(expected) != std::string::npos,
                std::string(name) + ": wrong failure: " + e.what());
    }
    std::filesystem::remove(path);
}

void test_valid_backbone() {
    const auto path = write_backbone("valid-backbone", [](gguf_context *) {});
    DelayLM model(path.string(), {}, 1, 64);
    check(model.config().text_vocab == TEXT_VOCAB, "backbone text vocab from tensor");
    check(model.config().audio_vocab == AUDIO_HEAD, "backbone audio vocab from tensor");
    check(model.config().n_vq == N_VQ, "backbone n_vq");
    std::filesystem::remove(path);
}

void test_valid_decoder() {
    const auto path = write_decoder("valid-decoder", N_VQ, 3 * D_MODEL, [](gguf_context *) {});
    Codec codec(path.string(), {}, 1);
    check(!codec.is_encoder(), "decoder role");
    check(codec.num_quantizers() == N_VQ, "decoder quantizer count");
    check(codec.samples_per_frame() == OUT_DIM, "decoder upsample from patch product");
    const std::vector<int32_t> narrow(4, 1);
    check(codec.decode(narrow, 1).size() == (size_t) (4 * OUT_DIM),
            "partial-channel decode renders one sample block per frame");
    std::filesystem::remove(path);
}

void test_engine_rejects_quantizer_mismatch() {
    const auto backbone = write_backbone("engine-backbone", [](gguf_context *) {});
    const auto decoder = write_decoder("engine-decoder", N_VQ - 1, 3 * D_MODEL,
            [](gguf_context *) {});
    tts_cpp::moss::EngineOptions options;
    options.backbone_path = backbone.string();
    options.decoder_path = decoder.string();
    options.n_threads = 1;
    options.context = 64;
    try {
        tts_cpp::moss::Engine engine(options);
        check(false, "engine accepted a decoder with mismatched quantizers");
    } catch (const std::runtime_error & e) {
        check(std::string(e.what()).find("fewer quantizers") != std::string::npos,
                std::string("engine mismatch: wrong failure: ") + e.what());
    }
    std::filesystem::remove(backbone);
    std::filesystem::remove(decoder);
}

constexpr uint32_t OVERSIZED = 0x7FFFFFFFu;
constexpr int BUDGET_MAX_NEW_TOKENS = 24;
constexpr int TERMINATION_ROWS = 2;
constexpr float OVERFLOW_EMBEDDING = 1.0f;
constexpr float OVERFLOW_PROJECTION = 12.5f;
constexpr float OVERFLOW_DOWN = 1.0f;

tts_cpp::moss::EngineOptions pair_options(const std::filesystem::path & backbone,
                                          const std::filesystem::path & decoder) {
    tts_cpp::moss::EngineOptions options;
    options.backbone_path = backbone.string();
    options.decoder_path = decoder.string();
    options.n_threads = 1;
    options.context = 64;
    options.max_new_tokens = BUDGET_MAX_NEW_TOKENS;
    return options;
}

bool engine_accepts_duration(const std::filesystem::path & backbone,
                             const std::filesystem::path & decoder, int duration_tokens,
                             std::string & error) {
    tts_cpp::moss::EngineOptions options = pair_options(backbone, decoder);
    options.duration_tokens = duration_tokens;
    try {
        tts_cpp::moss::Engine engine(options);
        return true;
    } catch (const std::runtime_error & e) {
        error = e.what();
        return false;
    }
}

void test_duration_budget() {
    const auto backbone = write_backbone("budget-backbone", [](gguf_context *) {});
    const auto decoder = write_decoder("budget-decoder", N_VQ, 3 * D_MODEL, [](gguf_context *) {});
    const int largest = BUDGET_MAX_NEW_TOKENS - (N_VQ - 1) - TERMINATION_ROWS;
    std::string error;
    check(engine_accepts_duration(backbone, decoder, largest, error),
            "a duration that leaves room for the delay drain is accepted: " + error);
    check(!engine_accepts_duration(backbone, decoder, largest + 1, error) &&
            error.find("raise max_new_tokens") != std::string::npos,
            "a duration without room for the delay drain is rejected");
    check(!engine_accepts_duration(backbone, decoder, -1, error) &&
            error.find("must not be negative") != std::string::npos,
            "a negative duration is rejected");
    check(engine_accepts_duration(backbone, decoder, 0, error), "free length needs no budget");
    std::filesystem::remove(backbone);
    std::filesystem::remove(decoder);
}

void test_engine_accepts_backends_dir() {
    const auto backbone = write_backbone("backends-backbone", [](gguf_context *) {});
    const auto decoder = write_decoder("backends-decoder", N_VQ, 3 * D_MODEL, [](gguf_context *) {});
    tts_cpp::moss::EngineOptions options = pair_options(backbone, decoder);
    options.backends_dir = std::filesystem::temp_directory_path().string();
    tts_cpp::moss::Engine engine(options);
    check(std::string(engine.backend_name()).size() > 0, "engine loads with a backends directory");
    std::filesystem::remove(backbone);
    std::filesystem::remove(decoder);
}

ConstantTensors overflowing_feed_forward() {
    return {
        {"token_embd.weight", {GGML_TYPE_F32, OVERFLOW_EMBEDDING}},
        {"blk.0.ffn_norm.weight", {GGML_TYPE_F32, OVERFLOW_EMBEDDING}},
        {"blk.0.ffn_gate.weight", {GGML_TYPE_F16, OVERFLOW_PROJECTION}},
        {"blk.0.ffn_up.weight", {GGML_TYPE_F16, OVERFLOW_PROJECTION}},
        {"blk.0.ffn_down.weight", {GGML_TYPE_F16, OVERFLOW_DOWN}},
    };
}

bool all_finite(const std::vector<float> & values) {
    for (float value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

void test_feed_forward_beyond_half_precision() {
    const auto path = write_backbone("overflow-backbone", [](gguf_context *) {},
            overflowing_feed_forward());
    DelayLM model(path.string(), {}, 1, 64);
    DelayRow row;
    row.text = SPECIALS;
    row.audio.assign(N_VQ, 0);
    const DelayLogits logits = model.prefill({row, row});
    check(all_finite(logits.text), "a feed-forward sum past the f16 range keeps text logits finite");
    check(all_finite(logits.audio[0]), "a feed-forward sum past the f16 range keeps audio logits finite");
    std::filesystem::remove(path);
}

constexpr uint32_t INFLATED_FRAMES = 1000000000u;

constexpr uint32_t MOSS_RATE = 24000;
constexpr uint32_t PERMISSIVE_RATE = 250000;
constexpr uint64_t SIXTY_SECONDS_AT_MAX_RATE = 60ull * 192000ull;

bool shape_rejected(const WavShape & shape, int expected_rate, const std::string & expected) {
    try {
        validate_reference_shape(shape, expected_rate);
        return false;
    } catch (const std::runtime_error & e) {
        return std::string(e.what()).find(expected) != std::string::npos;
    }
}

void test_reference_shape_limits() {
    check(shape_rejected({INFLATED_FRAMES, MOSS_RATE, 1}, MOSS_RATE, "seconds of audio"),
            "a header declaring a billion frames is rejected before allocation");
    check(shape_rejected({SIXTY_SECONDS_AT_MAX_RATE + 1, PERMISSIVE_RATE, 1}, PERMISSIVE_RATE,
            "more samples than the engine decodes"),
            "the decoded-sample cap holds even where the declared rate allows more");
    check(shape_rejected({MOSS_RATE, MOSS_RATE, 0}, MOSS_RATE, "channel count"),
            "a zero-channel header is rejected");
    validate_reference_shape({MOSS_RATE, MOSS_RATE, 2}, MOSS_RATE);
    try {
        require_reference_total((size_t) MOSS_RATE * 61, (int) MOSS_RATE);
        check(false, "dialogue references past sixty seconds are rejected");
    } catch (const std::runtime_error & e) {
        check(std::string(e.what()).find("total at most") != std::string::npos,
                std::string("dialogue total: wrong failure: ") + e.what());
    }
}

void test_reference_loudness_normalization() {
    using tts_cpp::moss::detail::normalize_reference_pcm;
    constexpr float TOLERANCE = 1e-6f;
    const std::vector<float> quiet{0.01f, -0.01f};
    const auto raised = normalize_reference_pcm(quiet);
    const auto lowered = normalize_reference_pcm({0.5f, -0.5f});
    const auto targeted = normalize_reference_pcm({0.08f, -0.08f});
    check(normalize_reference_pcm({}).empty(), "empty reference remains empty");
    check(normalize_reference_pcm({0.0f, 0.0f}) == std::vector<float>({0.0f, 0.0f}),
            "silent reference remains silent");
    check(std::abs(raised[0] - 0.014125375f) < TOLERANCE && raised[1] == -raised[0],
            "quiet reference gain is limited to three dB");
    check(std::abs(lowered[0] - 0.353972892f) < TOLERANCE && lowered[1] == -lowered[0],
            "loud reference attenuation is limited to three dB");
    check(std::abs(targeted[0] - 0.1f) < TOLERANCE && targeted[1] == -targeted[0],
            "reference reaches minus twenty dBFS within the gain limits");
    check(quiet[0] == 0.01f, "individual normalization preserves raw PCM for dialogue concatenation");
}

void expect_shaped_backbone_failure(const char * name, const TensorShapes & shapes,
                                    const std::string & expected) {
    const auto path = write_backbone(name, [](gguf_context *) {}, {}, shapes);
    try {
        DelayLM model(path.string(), {}, 1, 64);
        check(false, std::string(name) + ": accepted invalid GGUF");
    } catch (const std::runtime_error & e) {
        check(std::string(e.what()).find(expected) != std::string::npos,
                std::string(name) + ": wrong failure: " + e.what());
    }
    std::filesystem::remove(path);
}

void test_engine_accepts_matching_pair() {
    const auto backbone = write_backbone("engine-backbone-ok", [](gguf_context *) {});
    const auto decoder = write_decoder("engine-decoder-ok", N_VQ, 3 * D_MODEL,
            [](gguf_context *) {});
    tts_cpp::moss::EngineOptions options;
    options.backbone_path = backbone.string();
    options.decoder_path = decoder.string();
    options.n_threads = 1;
    options.context = 64;
    tts_cpp::moss::Engine engine(options);
    check(engine.sample_rate() == 24000, "engine sample rate");
    std::filesystem::remove(backbone);
    std::filesystem::remove(decoder);
}

} // namespace

int main() {
    try {
        test_engine_accepts_backends_dir();
        test_valid_backbone();
        test_valid_decoder();
        test_engine_accepts_matching_pair();
        test_engine_rejects_quantizer_mismatch();
        test_duration_budget();
        test_feed_forward_beyond_half_precision();
        test_reference_shape_limits();
        test_reference_loudness_normalization();

        expect_shaped_backbone_failure("text vocabulary mismatch",
                {{"token_embd.weight", {N_EMBD, TEXT_VOCAB + SPECIALS}}},
                "disagree on the text vocabulary");
        expect_shaped_backbone_failure("audio vocabulary mismatch",
                {{"token_embd_audio.1.weight", {N_EMBD, AUDIO_HEAD + 1}}},
                "disagrees on the audio vocabulary");
        expect_shaped_backbone_failure("down projection shape",
                {{"blk.0.ffn_down.weight", {N_FF, N_EMBD + 1}}}, "unexpected dimensions");
        expect_backbone_failure("kv heads do not divide heads", [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-delay.attention.head_count_kv", 3);
        }, "invalid model geometry");
        expect_decoder_failure("inflated sampling rate", 3 * D_MODEL, [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-audio-decoder.sampling_rate", INFLATED_FRAMES);
        }, "sampling rate must be");

        expect_backbone_failure("oversized block count", [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-delay.block_count", OVERSIZED);
        }, "invalid model geometry");
        expect_backbone_failure("oversized channel count", [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-delay.n_vq", OVERSIZED);
        }, "invalid model geometry");
        expect_backbone_failure("oversized feed forward", [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-delay.feed_forward_length", OVERSIZED);
        }, "invalid model geometry");
        expect_decoder_failure("oversized module count", 3 * D_MODEL, [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.block_count", OVERSIZED);
        }, "invalid module count");
        expect_decoder_failure("oversized layer count", 3 * D_MODEL, [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.0.num_layers", OVERSIZED);
        }, "invalid transformer geometry");
        expect_decoder_failure("oversized quantizer count", 3 * D_MODEL, [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-audio-decoder.quantizer.num_quantizers", OVERSIZED);
        }, "invalid quantizer geometry");
        expect_decoder_failure("zero decoder context", 3 * D_MODEL, [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.0.context", 0);
        }, "invalid transformer geometry");

        expect_backbone_failure("zero heads", [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-delay.attention.head_count", 0);
        }, "invalid model geometry");
        expect_backbone_failure("token id outside vocab", [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-delay.audio_assistant_gen_slot_token_id", 999);
        }, "outside the text vocabulary");
        expect_backbone_failure("huge token id", [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-delay.audio_end_token_id", 0xFFFFFFFFu);
        }, "outside the text vocabulary");
        expect_backbone_failure("pad code outside audio vocab", [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-delay.audio_pad_code", 99);
        }, "outside the audio vocabulary");

        expect_decoder_failure("zero patch size", 3 * D_MODEL, [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-audio-decoder.decoder.1.patch_size", 0);
        }, "invalid patch size");
        expect_decoder_failure("zero quantizers", 3 * D_MODEL, [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-audio-decoder.quantizer.num_quantizers", 0);
        }, "invalid quantizer geometry");
        expect_decoder_failure("oversized codebook_size", 3 * D_MODEL, [](gguf_context * f) {
            gguf_set_val_u32(f, "moss-tts-audio-decoder.quantizer.codebook_size", 4 * CODE_SIZE);
        }, "does not match the declared quantizer geometry");
        expect_decoder_failure("truncated qkv tensor", 2 * D_MODEL, [](gguf_context *) {},
                "unexpected dimensions");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    if (failures == 0) {
        std::printf("moss load validation: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss load validation: %d failures\n", failures);
    return 1;
}
