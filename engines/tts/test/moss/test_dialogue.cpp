#include "gguf_fixtures.h"

#include "moss/frontend.h"
#include "moss/generation.h"
#include "tts-cpp/moss/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using namespace moss_fixtures;
using namespace tts_cpp::moss::detail;

namespace {

constexpr int CONTINUATION_FRAMES = 10;
constexpr uint32_t DIALOGUE_WEIGHT_SEED = 29;
constexpr int TTSD_AUDIO_CHANNELS = 16;
constexpr int TTS_AUDIO_CHANNELS = 32;

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

DelayConfig prompt_config() {
    DelayConfig config;
    config.n_vq = 3;
    config.text_vocab = 200;
    config.audio_vocab = 11;
    config.audio_pad_code = 10;
    config.audio_start_token_id = 100;
    config.audio_end_token_id = 101;
    config.audio_user_slot_token_id = 102;
    config.audio_assistant_gen_slot_token_id = 103;
    config.audio_assistant_delay_slot_token_id = 104;
    return config;
}

std::vector<int32_t> ramp_codes(const DelayConfig & config, int frames, int32_t base) {
    std::vector<int32_t> codes((size_t) frames * config.n_vq);
    for (size_t i = 0; i < codes.size(); ++i) {
        codes[i] = (int32_t) ((base + i) % 9);
    }
    return codes;
}

int count_rows_with_text(const std::vector<DelayRow> & rows, int32_t token) {
    int count = 0;
    for (const DelayRow & row : rows) {
        if (row.text == token) {
            count++;
        }
    }
    return count;
}

void test_dialogue_prompt_rows() {
    const DelayConfig config = prompt_config();
    PromptTokens tokens;
    tokens.pad = 0;
    tokens.im_start = 2;
    tokens.im_end = 3;
    std::string captured;
    const TextEncoder recorder = [&captured](const std::string & span) {
        captured += span;
        return std::vector<int32_t>{(int32_t) (span.size() % 90)};
    };

    PromptAudio audio;
    audio.speaker_codes.push_back(ramp_codes(config, 4, 1));
    audio.speaker_codes.push_back(ramp_codes(config, 6, 2));
    audio.continuation_codes = ramp_codes(config, 10, 3);

    const std::vector<DelayRow> rows = build_prompt_rows(config, tokens, recorder,
            "[S1] hola [S2] que tal", "es", 0, audio);

    check(captured.find("[S1]:\n") != std::string::npos &&
            captured.find("[S2]:\n") != std::string::npos,
            "reference field lists both speakers");
    check(count_rows_with_text(rows, config.audio_start_token_id) == 3,
            "two user blocks plus the continuation block open with audio_start");
    check(count_rows_with_text(rows, config.audio_user_slot_token_id) ==
            (4 + config.n_vq - 1) + (6 + config.n_vq - 1),
            "each speaker expands to frames + n_vq - 1 user slots");
    check(count_rows_with_text(rows, config.audio_end_token_id) == 2,
            "only the user blocks close with audio_end");
    check(count_rows_with_text(rows, config.audio_assistant_gen_slot_token_id) == 10,
            "the continuation block carries one gen row per reference frame");
    check(rows.back().text == config.audio_assistant_gen_slot_token_id,
            "the prompt ends mid-stream on a gen row");
    check(rows.back().audio[0] != config.audio_pad_code,
            "the final continuation row carries delayed codes");
}

std::string capture_prompt_template(int audio_channels) {
    DelayConfig config = prompt_config();
    config.n_vq = audio_channels;
    const PromptTokens tokens{0, 2, 3};
    std::string captured;
    const TextEncoder recorder = [&captured](const std::string & span) {
        captured += span;
        return std::vector<int32_t>{1};
    };
    build_prompt_rows(config, tokens, recorder, "[S1] Hello.", "English", 0, {});
    return captured;
}

void test_reference_prompt_templates() {
    const std::string expected =
            "user\n<user_inst>\n- Reference(s):\nNone\n- Instruction:\nNone\n"
            "- Tokens:\nNone\n- Quality:\nNone\n- Sound Event:\nNone\n"
            "- Ambient Sound:\nNone\n- Language:\nEnglish\n- Scene:\nNone\n"
            "- Text:\n[S1] Hello.\n</user_inst>\nassistant\n";
    check(capture_prompt_template(TTSD_AUDIO_CHANNELS) == expected,
            "TTSD matches the upstream UserMessage template including Scene");
    const std::string tts = capture_prompt_template(TTS_AUDIO_CHANNELS);
    check(tts.find("- Scene:") == std::string::npos,
            "TTS v1.5 retains its template without Scene");
    check(tts.find("- Language:\nEnglish\n- Text:") != std::string::npos,
            "TTS v1.5 text follows language directly");
}

void test_continuation_state_and_extraction() {
    const DelayConfig config = prompt_config();
    PromptTokens tokens;
    tokens.pad = 0;
    tokens.im_start = 2;
    tokens.im_end = 3;
    const TextEncoder encode = [](const std::string & span) {
        return std::vector<int32_t>{(int32_t) (span.size() % 90)};
    };

    PromptAudio audio;
    audio.speaker_codes.push_back(ramp_codes(config, 4, 1));
    audio.continuation_codes = ramp_codes(config, CONTINUATION_FRAMES, 3);
    const std::vector<DelayRow> prompt = build_prompt_rows(config, tokens, encode,
            "hola", "es", 0, audio);

    DelayState state(config, prompt, tokens.pad, tokens.im_end);
    SamplingConfig sampling;
    sampling.text_temperature = 0.0f;
    sampling.audio_temperature = 0.0f;
    std::mt19937 rng(1);

    DelayLogits logits;
    logits.text.assign((size_t) config.text_vocab, 0.0f);
    logits.text[(size_t) config.audio_assistant_gen_slot_token_id] = 50.0f;
    logits.audio.assign((size_t) config.n_vq,
            std::vector<float>((size_t) config.audio_vocab, 0.0f));
    for (auto & channel : logits.audio) {
        channel[7] = 50.0f;
    }

    const DelayRow row = state.step(logits, sampling, rng);
    check(row.audio[0] == 7 && row.audio[1] == 7 && row.audio[2] == 7,
            "every channel is already open when generation continues past the references");

    for (int step = 0; step < 5; ++step) {
        state.step(logits, sampling, rng);
    }
    const int base_row = (int) prompt.size() - CONTINUATION_FRAMES;
    const AudioSegments segments = state.generated_audio(base_row);
    check(segments.size() == 1, "references and their continuation form one segment");
    const std::vector<int32_t> & segment = segments.front();
    const size_t context = (size_t) CONTINUATION_FRAMES * config.n_vq;
    check(segment.size() > context, "continuation emits audio beyond the references");
    const size_t prompt_only = (size_t) (CONTINUATION_FRAMES - (config.n_vq - 1)) * config.n_vq;
    check(std::equal(audio.continuation_codes.begin(),
            audio.continuation_codes.begin() + (std::ptrdiff_t) prompt_only, segment.begin()),
            "the segment opens with the reference codes as decoder context");
    check(std::all_of(segment.begin() + (std::ptrdiff_t) context, segment.end(),
            [](int32_t code) { return code == 7; }),
            "frames after the context are generated codes, not reference codes");
}

tts_cpp::moss::EngineOptions dialogue_options(const std::filesystem::path & backbone,
                                              const std::filesystem::path & decoder,
                                              const std::filesystem::path & encoder,
                                              const std::filesystem::path & s1,
                                              const std::filesystem::path & s2) {
    tts_cpp::moss::EngineOptions options;
    options.backbone_path = backbone.string();
    options.decoder_path = decoder.string();
    options.encoder_path = encoder.string();
    options.dialogue_reference_paths = {s1.string(), s2.string()};
    options.n_threads = 1;
    options.context = 512;
    options.max_new_tokens = 24;
    options.stream_chunk_frames = 2;
    return options;
}

void test_engine_dialogue(const tts_cpp::moss::EngineOptions & options) {
    tts_cpp::moss::Engine engine(options);
    const tts_cpp::moss::SynthesisResult batch = engine.synthesize("[S1] hi [S2] hello");
    check(!batch.pcm.empty(), "dialogue synthesis produces PCM");
    check(std::any_of(batch.pcm.begin(), batch.pcm.end(), [](float s) { return s != 0.0f; }),
            "random codec weights make the dialogue parity check meaningful");

    std::vector<float> streamed;
    const tts_cpp::moss::SynthesisResult stream = engine.synthesize_stream("[S1] hi [S2] hello",
            [&](const float * samples, size_t count, int) {
                streamed.insert(streamed.end(), samples, samples + count);
                return true;
            });
    check(!stream.cancelled, "dialogue streaming completes");
    check(streamed.size() == batch.pcm.size(), "dialogue stream sample count matches batch");
    float worst = 0.0f;
    for (size_t i = 0; i < streamed.size(); ++i) {
        worst = std::max(worst, std::fabs(streamed[i] - batch.pcm[i]));
    }
    check(worst < 1e-4f, "dialogue stream matches the batch render");
}

void test_engine_rejects_mixed_references(const tts_cpp::moss::EngineOptions & base) {
    tts_cpp::moss::EngineOptions options = base;
    options.reference_audio_path = options.dialogue_reference_paths[0];
    try {
        tts_cpp::moss::Engine engine(options);
        check(false, "mixed single and dialogue references were accepted");
    } catch (const std::runtime_error & e) {
        check(std::string(e.what()).find("exclusive") != std::string::npos,
                std::string("mixed references: wrong failure: ") + e.what());
    }
}

} // namespace

int main() {
    int rc = 0;
    try {
        test_dialogue_prompt_rows();
        test_reference_prompt_templates();
        test_continuation_state_and_extraction();

        const auto backbone = write_backbone("dlg-backbone", [](gguf_context *) {});
        std::mt19937 weights(DIALOGUE_WEIGHT_SEED);
        const auto decoder = write_decoder("dlg-decoder", N_VQ, 3 * D_MODEL,
                [](gguf_context *) {}, &weights);
        const auto encoder = write_encoder("dlg-encoder");
        const auto s1 = write_tiny_wav("dlg-s1", 64);
        const auto s2 = write_tiny_wav("dlg-s2", 96);
        const tts_cpp::moss::EngineOptions options =
                dialogue_options(backbone, decoder, encoder, s1, s2);
        test_engine_dialogue(options);
        test_engine_rejects_mixed_references(options);
        std::filesystem::remove(backbone);
        std::filesystem::remove(decoder);
        std::filesystem::remove(encoder);
        std::filesystem::remove(s1);
        std::filesystem::remove(s2);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        rc = 1;
    }
    if (rc != 0) {
        return rc;
    }
    if (failures == 0) {
        std::printf("moss dialogue: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss dialogue: %d failures\n", failures);
    return 1;
}
