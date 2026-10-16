#include "transcribe_fixtures.h"
#include "../../../test/moss_precision.h"

#include "moss/transcribe_cli.h"
#include "moss/transcribe_audio.h"
#include "moss/transcribe_bpe.h"
#include "moss/transcribe_coreml.h"
#include "moss/transcribe_model.h"
#include "moss/transcribe_networks.h"
#include "moss/transcribe_runtime.h"
#include "moss/transcribe_text.h"
#include "parakeet/moss_transcribe.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <locale>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace moss_transcribe_fixtures;
using namespace parakeet::moss::detail;

namespace {

constexpr uint32_t WEIGHT_SEED = 20260925;
constexpr uint32_t AUDIO_SEED = 7;
constexpr float FFT_TOLERANCE = 1e-3f;
constexpr float LOGIT_TOLERANCE = 1e-4f;
constexpr double PI = 3.14159265358979323846;
constexpr int WHISPER_SAMPLE_RATE = 16000;
constexpr int WHISPER_CHUNK = 480000;
constexpr int WHISPER_HOP = 160;
constexpr int WHISPER_MERGE = 4;
constexpr int REFERENCE_SAMPLES = 1980480;
constexpr int REFERENCE_AUDIO_TOKENS = 1548;
constexpr int REFERENCE_SPAN = 1600;
constexpr int CACHE_ALIGNMENT = 256;
constexpr int SHORT_CONTEXT = 64;
constexpr int OVER_CONTEXT_CHUNKS = 10;
constexpr int NEVER_FAIL = -1;
constexpr const char * FAKE_SIDECAR_LABEL = "coreml-fake";
constexpr int SPLIT_HEAD_ROWS = 100;
constexpr int SPLIT_HEAD_CHUNKS = 3;
constexpr int SPLIT_TAIL_ROWS = TEXT_VOCAB - 2 * SPLIT_HEAD_ROWS;
constexpr int DECODE_STEPS = 5;
constexpr double FUSED_RELATIVE_TOLERANCE = 1e-2;
constexpr int CACHE_WIDTH = 4;
constexpr int CACHE_POSITIONS = 8;
constexpr int COPY_TOKENS = 3;
constexpr int COPY_POSITION = 2;
constexpr float UNTOUCHED = -1.0f;
constexpr int COPY_GRAPH_NODES = 16;
constexpr int MASK_POSITION = 3;
constexpr int MASK_TOKENS = 4;
const char * const COMMA_DECIMAL_LOCALES[] = {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8", "de_DE",
                                              "fr_FR", "German_Germany.1252"};

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

template <typename Fn>
void expect_failure(Fn && fn, const std::string & needle, const std::string & label) {
    try {
        fn();
        check(false, label + ": accepted");
    } catch (const std::runtime_error & e) {
        check(std::string(e.what()).find(needle) != std::string::npos, label + ": wrong failure: " + e.what());
    }
}

float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return INFINITY;
    }
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

std::vector<float> random_signal(size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> draw(-1.0f, 1.0f);
    std::vector<float> values(count);
    for (float & value : values) {
        value = draw(rng);
    }
    return values;
}

std::vector<std::complex<float>> naive_dft(const std::vector<float> & input) {
    const size_t n = input.size();
    std::vector<std::complex<float>> output(n);
    for (size_t k = 0; k < n; ++k) {
        std::complex<double> sum = 0.0;
        for (size_t j = 0; j < n; ++j) {
            sum += (double) input[j] * std::polar(1.0, -2.0 * PI * (double) (j * k) / (double) n);
        }
        output[k] = std::complex<float>((float) sum.real(), (float) sum.imag());
    }
    return output;
}

float spectrum_error(const std::vector<std::complex<float>> & a, const std::vector<std::complex<float>> & b) {
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::abs(a[i] - b[i]));
    }
    return worst;
}

void test_fft_matches_dft() {
    for (int size : {12, 25, 400}) {
        const std::vector<float> signal = random_signal((size_t) size, AUDIO_SEED);
        check(spectrum_error(TranscribeFft(size).transform(signal), naive_dft(signal)) < FFT_TOLERANCE,
              "mixed-radix FFT matches the direct DFT for size " + std::to_string(size));
    }
}

void test_log_mel_normalization() {
    std::vector<float> mel = {-12.0f, -3.0f, 0.0f, 1.0f};
    normalize_log_mel(mel);
    check(std::fabs(mel[0] - ((1.0f - 8.0f + 4.0f) / 4.0f)) < 1e-6f, "values below max-8 clamp to the floor");
    check(std::fabs(mel[3] - (5.0f / 4.0f)) < 1e-6f, "the maximum maps to (max + 4) / 4");
}

TranscribeConfig whisper_like_config() {
    TranscribeConfig config;
    config.audio.sample_rate = WHISPER_SAMPLE_RATE;
    config.audio.hop_length = WHISPER_HOP;
    config.audio.chunk_samples = WHISPER_CHUNK;
    config.audio.chunk_frames = WHISPER_CHUNK / WHISPER_HOP;
    config.merge_size = WHISPER_MERGE;
    config.audio_tokens_per_second = 12.5f;
    config.time_marker_every_seconds = 5;
    config.time_markers = true;
    config.tokens.audio_pad = TOKEN_AUDIO_PAD;
    return config;
}

void test_chunk_arithmetic() {
    const TranscribeConfig config = whisper_like_config();
    check(transcribe_chunk_count(config.audio, 0) == 0, "no audio has no chunks");
    check(transcribe_chunk_count(config.audio, WHISPER_CHUNK) == 1, "a full window is one chunk");
    check(transcribe_chunk_count(config.audio, WHISPER_CHUNK + 1) == 2, "one extra sample opens a chunk");
    check(transcribe_chunk_tokens(config, WHISPER_CHUNK) == 375, "a 30 s chunk yields 375 tokens");
    check(transcribe_chunk_tokens(config, 1) == 1, "a single sample yields one token");
    check(transcribe_audio_tokens(config, REFERENCE_SAMPLES) == REFERENCE_AUDIO_TOKENS,
          "audio token count matches the Hugging Face processor");
}

TranscribeTokenizer fixture_tokenizer() {
    return TranscribeTokenizer(transcribe_vocab(), {"a b"}, transcribe_token_types());
}

size_t count_pads(const std::vector<int32_t> & span) {
    return (size_t) std::count(span.begin(), span.end(), TOKEN_AUDIO_PAD);
}

void test_audio_span_markers() {
    const TranscribeConfig config = whisper_like_config();
    const TranscribeTokenizer tokenizer = fixture_tokenizer();
    const std::vector<int32_t> span = transcribe_audio_span(config, tokenizer, REFERENCE_AUDIO_TOKENS);
    check(span.size() == REFERENCE_SPAN, "span length matches the Hugging Face processor");
    check(count_pads(span) == REFERENCE_AUDIO_TOKENS, "every audio token keeps its placeholder");
    check(span[62] == byte_token('5'), "the 5 s marker follows 62 placeholders");
    check(span[125] == byte_token('1') && span[126] == byte_token('0'), "the 10 s marker is spelled digit by digit");
    TranscribeConfig plain = config;
    plain.time_markers = false;
    check(transcribe_audio_span(plain, tokenizer, 10) == std::vector<int32_t>(10, TOKEN_AUDIO_PAD),
          "disabled markers leave only placeholders");
    check(transcribe_audio_span(config, tokenizer, 0).empty(), "no audio tokens give an empty span");
}

std::vector<int32_t> bytes_of(const std::string & text) {
    std::vector<int32_t> ids;
    for (unsigned char ch : text) {
        ids.push_back(byte_token(ch));
    }
    return ids;
}

std::vector<int32_t> expected_prompt(const std::vector<int32_t> & span, const std::vector<int32_t> & body) {
    std::vector<int32_t> ids{TOKEN_IM_START};
    for (const auto & part : {bytes_of("system\nsys"), std::vector<int32_t>{TOKEN_IM_END}, bytes_of("\n"),
                              std::vector<int32_t>{TOKEN_IM_START}, bytes_of("user\n"),
                              std::vector<int32_t>{TOKEN_AUDIO_START}, span,
                              std::vector<int32_t>{TOKEN_AUDIO_END}, body, std::vector<int32_t>{TOKEN_IM_END},
                              bytes_of("\n"), std::vector<int32_t>{TOKEN_IM_START}, bytes_of("assistant\n")}) {
        ids.insert(ids.end(), part.begin(), part.end());
    }
    return ids;
}

void test_prompt_layout() {
    TranscribeConfig config = whisper_like_config();
    config.tokens = {TOKEN_AUDIO_START, TOKEN_AUDIO_END, TOKEN_AUDIO_PAD, TOKEN_IM_START, TOKEN_IM_END, 0};
    config.system_prompt = SYSTEM_PROMPT;
    config.default_prompt_ids = {byte_token('h'), byte_token('i')};
    config.time_markers = false;
    const TranscribeTokenizer tokenizer = fixture_tokenizer();
    const std::vector<int32_t> span(3, TOKEN_AUDIO_PAD);
    check(transcribe_prompt(config, tokenizer, 3, "") == expected_prompt(span, bytes_of("\nhi")),
          "empty prompt uses the stored default prompt ids");
    check(transcribe_prompt(config, tokenizer, 3, "ok") == expected_prompt(span, bytes_of("\nok")),
          "a custom prompt replaces the default");
    check(transcribe_prompt(config, tokenizer, 3, " \n\t") == expected_prompt(span, bytes_of("\nhi")),
          "a whitespace-only prompt falls back to the default");
}

void test_detokenizer() {
    const TranscribeTokenizer tokenizer = fixture_tokenizer();
    std::vector<int32_t> ids = bytes_of("h\xc3\xa9");
    ids.insert(ids.begin(), TOKEN_IM_START);
    ids.push_back(TOKEN_USER_DEFINED);
    ids.push_back(TOKEN_IM_END);
    check(tokenizer.decode(ids) == "h\xc3\xa9<think>", "decode restores bytes, keeps user tokens, skips control");
    check(tokenizer.decode({-1, 100000}) == "", "out-of-range ids decode to nothing");
    check(tokenizer.encode("h\xc3\xa9") == bytes_of("h\xc3\xa9"), "byte-level encode round-trips");
}

using Pieces = std::vector<std::string>;

void test_pretokenizer_unicode() {
    check(qwen_pretokenize("\xe8\x8c\x83\xe5\x9b\xb4\xe3\x80\x82\xe7\x83\xad\xe8\xaf\x8d") ==
          Pieces{"\xe8\x8c\x83\xe5\x9b\xb4", "\xe3\x80\x82\xe7\x83\xad\xe8\xaf\x8d"},
          "CJK punctuation is not a letter and prefixes the next word");
    check(qwen_pretokenize("  \n  x") == Pieces{"  \n", " ", " x"}, "whitespace ending in a newline splits there");
    check(qwen_pretokenize("a\xe3\x80\x80" "b") == Pieces{"a", "\xe3\x80\x80" "b"},
          "the ideographic space is whitespace");
    check(qwen_pretokenize("x\xef\xbc\x91\xef\xbc\x92") == Pieces{"x", "\xef\xbc\x91", "\xef\xbc\x92"},
          "full-width digits are single numbers");
    check(qwen_pretokenize("caf\xc3\xa9 it's") == Pieces{"caf\xc3\xa9", " it", "'s"},
          "accented letters stay in the word and contractions split");
}

void test_hotword_sanitation() {
    check(sanitize_hotword("  Q\nV[A]C  <|x|>  ") == "Q V A C x", "breakers become spaces and runs collapse");
    check(sanitize_hotwords({"QVAC", " ", "QVAC", "vcpkg"}) == std::vector<std::string>{"QVAC", "vcpkg"},
          "empty and duplicate hotwords are dropped");
    expect_failure([] { sanitize_hotword(std::string(MAX_HOTWORD_BYTES + 1, 'x')); }, "at most",
                   "an oversized hotword is rejected");
    expect_failure([] { sanitize_hotwords(std::vector<std::string>(MAX_HOTWORDS + 1, "x")); }, "at most",
                   "too many hotwords are rejected");
    TranscribeConfig config;
    config.default_prompt = DEFAULT_PROMPT;
    config.hotword_prefix = HOTWORD_PREFIX;
    config.hotword_separator = HOTWORD_SEPARATOR;
    check(hotword_prompt(config, {"QVAC", "vcpkg"}) == "hi Hotwords: QVAC, vcpkg",
          "hotwords extend the default prompt with the model's prefix and separator");
}

void test_parser_basic() {
    const auto segments = parse_transcript("[0.50][S01] hello there [1.25][1.25][S02]world[2]");
    check(segments.size() == 2, "two segments parse");
    check(segments.size() == 2 && segments[0].start_s == 0.5 && segments[0].end_s == 1.25 &&
          segments[0].speaker == "S01" && segments[0].text == "hello there", "first segment fields");
    check(segments.size() == 2 && segments[1].speaker == "S02" && segments[1].end_s == 2.0,
          "the last segment is emitted on close");
}

void test_parser_edge_cases() {
    check(parse_transcript("[1.0][X1]nope[2.0]").empty(), "an invalid speaker drops the segment");
    check(parse_transcript("[1.0][S1]   [2.0]").empty(), "empty text is skipped");
    const auto backwards = parse_transcript("[2.0][S1]a[1.0]b[3.0]");
    check(backwards.size() == 1 && backwards[0].text == "a[1.0]b", "an end before the start stays in the text");
    const auto spaced = parse_transcript("[0][S1]a[1] b[2]");
    check(spaced.size() == 1 && spaced[0].text == "a[1] b" && spaced[0].end_s == 2.0,
          "text after an end timestamp reopens the segment");
    check(parse_transcript("[0][S1]dangling").empty(), "an unterminated segment is not emitted");
    check(parse_transcript("noise [0][S1]ok[1]").size() == 1, "leading noise is ignored");
    check(parse_transcript("[1.2.3][S1]x[4]").empty(), "a timestamp with two dots is rejected");
}

void test_parser_streaming() {
    const std::string text = "[0.50][S01] hello [1.25][1.25][S02]world[2.00]";
    TranscriptParser parser;
    std::vector<parakeet::moss::TranscriptSegment> streamed;
    for (char ch : text) {
        const auto emitted = parser.feed(std::string(1, ch));
        streamed.insert(streamed.end(), emitted.begin(), emitted.end());
    }
    const auto tail = parser.close();
    streamed.insert(streamed.end(), tail.begin(), tail.end());
    check(streamed.size() == parse_transcript(text).size(), "character streaming matches a whole parse");
}

void test_model_loads() {
    const auto path = write_transcribe_model("transcribe-load", WEIGHT_SEED);
    TranscribeModel model(path.string(), false, 1);
    const TranscribeConfig & config = model.config();
    check(config.text.vocab == TEXT_VOCAB, "vocabulary size comes from the embedding table");
    check(config.samples_per_token() == HOP * 2 * MERGE, "samples per token follow hop, stride and merge");
    check(config.default_prompt_ids.size() == 2, "default prompt ids load");
    validate_transcribe_encoder(model);
    validate_transcribe_decoder(model);
    std::filesystem::remove(path);
}

void expect_model_rejects(const std::filesystem::path & path, const std::string & needle, const std::string & label) {
    expect_failure([&] { TranscribeModel model(path.string(), false, 1); }, needle, label);
    std::filesystem::remove(path);
}

void test_model_rejections() {
    expect_model_rejects(write_transcribe_model("transcribe-arch", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_str(f, "general.architecture", "moss-sfx");
    }), "unsupported architecture", "wrong architecture");
    expect_model_rejects(write_transcribe_model("transcribe-window", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-transcribe.encoder.context_length", ENC_CTX + 1);
    }), "encoder geometry", "encoder window must be half the mel frames");
    expect_model_rejects(write_transcribe_model("transcribe-frames", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-transcribe.audio.chunk_frames", CHUNK_FRAMES + 1);
    }), "audio front-end", "chunk frames must match samples / hop");
    expect_model_rejects(write_transcribe_model("transcribe-gqa", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-transcribe.text.attention.head_count_kv", 3);
    }), "decoder geometry", "query heads must be a multiple of key heads");
    expect_model_rejects(write_transcribe_model("transcribe-merge", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-transcribe.adaptor.merge_size", 3);
    }), "prompt metadata", "merge size must divide the encoder window");
    expect_model_rejects(write_transcribe_model("transcribe-prompt", WEIGHT_SEED, [](gguf_context * f) {
        gguf_remove_key(f, "moss-transcribe.default_prompt_ids");
    }), "default_prompt_ids", "default prompt ids are required");
    expect_model_rejects(write_transcribe_model("transcribe-context", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-transcribe.text.context_length", 0x7fffffffu);
    }), "decoder geometry", "an oversized training context is rejected");
    expect_model_rejects(write_transcribe_model("transcribe-new-tokens", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-transcribe.default_max_new_tokens", TEXT_CTX + 1);
    }), "prompt metadata", "a default token budget past the context is rejected");
    expect_model_rejects(write_transcribe_model("transcribe-no-hotwords", WEIGHT_SEED, [](gguf_context * f) {
        gguf_remove_key(f, "moss-transcribe.hotword_prefix");
    }), "hotword_prefix", "the hotword prefix is required");
    expect_model_rejects(write_transcribe_model("transcribe-rope", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_f32(f, "moss-transcribe.text.rope.freq_base", 0.0f);
    }), "decoder geometry", "a zero RoPE base is rejected");
    expect_model_rejects(write_transcribe_model("transcribe-rms-eps", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_f32(f, "moss-transcribe.text.attention.layer_norm_rms_epsilon", NAN);
    }), "decoder geometry", "a non-finite RMS epsilon is rejected");
    expect_model_rejects(write_transcribe_model("transcribe-enc-eps", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_f32(f, "moss-transcribe.encoder.attention.layer_norm_epsilon", -1.0f);
    }), "encoder geometry", "a negative encoder epsilon is rejected");
    expect_model_rejects(write_transcribe_model("transcribe-adaptor-eps", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_f32(f, "moss-transcribe.adaptor.layer_norm_epsilon", INFINITY);
    }), "prompt metadata", "an infinite adaptor epsilon is rejected");
    expect_model_rejects(write_transcribe_model("transcribe-rate-high", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_f32(f, "moss-transcribe.audio_tokens_per_second", 1e30f);
    }), "prompt metadata", "a huge token rate is rejected");
    expect_model_rejects(write_transcribe_model("transcribe-rate-low", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_f32(f, "moss-transcribe.audio_tokens_per_second", 1e-30f);
    }), "prompt metadata", "a vanishing token rate is rejected");
    const auto healthy = write_transcribe_model("transcribe-threads", WEIGHT_SEED);
    expect_failure([&] { TranscribeModel model(healthy.string(), false, 0); }, "threads", "zero threads rejected");
    std::filesystem::remove(healthy);
}

void test_token_table_width_rejected() {
    const auto shapes = TensorShapes{{"text.token_embd.weight", {TEXT_EMBD * 2, TEXT_VOCAB}}};
    const auto path = write_transcribe_model("transcribe-table-width", WEIGHT_SEED, [](gguf_context *) {}, shapes);
    expect_failure([&] { TranscribeModel model(path.string(), false, 1); }, "text.token_embd.weight",
                   "a token table wider than the decoder is rejected at load");
    expect_failure([&] {
        TranscribeModel model(path.string(), false, 1, "auto", false, TranscribeLayout{SPLIT_HEAD_ROWS});
    }, "text.token_embd.weight", "a split token table never reads rows wider than the decoder");
    std::filesystem::remove(path);
}

void test_network_rejections() {
    const auto shapes = TensorShapes{{"adaptor.fc1.weight", {ENC_EMBD, TEXT_EMBD}}};
    const auto path = write_transcribe_model("transcribe-adaptor", WEIGHT_SEED, [](gguf_context *) {}, shapes);
    TranscribeModel model(path.string(), false, 1);
    expect_failure([&] { validate_transcribe_encoder(model); }, "adaptor.fc1.weight", "adaptor width is validated");
    std::filesystem::remove(path);
}

std::vector<float> tiny_mel(const TranscribeModel & model, const std::vector<float> & audio) {
    const TranscribeMel mel(model.config().audio, read_mel_filters(model));
    return mel.chunk(audio.data(), audio.size());
}

void test_encoder_chunk() {
    const auto path = write_transcribe_model("transcribe-encoder", WEIGHT_SEED);
    TranscribeModel model(path.string(), false, 1);
    const std::vector<float> mel = tiny_mel(model, random_signal(CHUNK_SAMPLES, AUDIO_SEED));
    check(mel.size() == (size_t) N_MELS * CHUNK_FRAMES, "mel chunk has n_mels x frames values");
    const TranscribeChunkEncoding full = encode_audio_chunk(model, mel, CHUNK_TOKENS, true);
    check(full.embeddings.size() == (size_t) CHUNK_TOKENS * TEXT_EMBD, "a full chunk yields one row per token");
    check(full.encoder_states.size() == (size_t) ENC_CTX * ENC_EMBD, "encoder states cover the window");
    const TranscribeChunkEncoding partial = encode_audio_chunk(model, mel, 2, false);
    check(partial.encoder_states.empty(), "encoder states are dropped unless requested");
    const std::vector<float> head(full.embeddings.begin(), full.embeddings.begin() + 2 * TEXT_EMBD);
    check(max_abs_diff(partial.embeddings, head) < LOGIT_TOLERANCE, "a short chunk keeps the leading tokens");
    expect_failure([&] { encode_audio_chunk(model, mel, CHUNK_TOKENS + 1, false); }, "encoder window",
                   "too many tokens rejected");
    expect_failure([&] { encode_audio_chunk(model, {1.0f}, 1, false); }, "wrong size", "bad mel size rejected");
    std::filesystem::remove(path);
}

std::vector<int32_t> tiny_prompt(int audio_tokens) {
    std::vector<int32_t> ids{TOKEN_IM_START, byte_token('a'), TOKEN_AUDIO_START};
    ids.insert(ids.end(), (size_t) audio_tokens, TOKEN_AUDIO_PAD);
    ids.push_back(TOKEN_AUDIO_END);
    ids.push_back(byte_token('b'));
    return ids;
}

std::vector<float> prefill_logits(TranscribeModel & model, const std::vector<int32_t> & prompt,
                                  const std::vector<float> & audio, int batch) {
    TranscribeDecoder decoder(model, (int) prompt.size() + 1);
    return decoder.prefill(prompt, audio, batch);
}

void test_decoder_batches() {
    const auto path = write_transcribe_model("transcribe-decoder", WEIGHT_SEED);
    TranscribeModel model(path.string(), false, 1);
    const std::vector<int32_t> prompt = tiny_prompt(5);
    const std::vector<float> audio = random_signal((size_t) 5 * TEXT_EMBD, AUDIO_SEED);
    const std::vector<float> whole = prefill_logits(model, prompt, audio, (int) prompt.size());
    check(whole.size() == (size_t) TEXT_VOCAB, "prefill returns one logit per vocabulary entry");
    check(max_abs_diff(prefill_logits(model, prompt, audio, 1), whole) < LOGIT_TOLERANCE,
          "token-by-token prefill matches a single batch");
    check(max_abs_diff(prefill_logits(model, prompt, audio, 3), whole) < LOGIT_TOLERANCE,
          "uneven batches match a single batch");
    const std::vector<float> silent(audio.size(), 0.0f);
    check(max_abs_diff(prefill_logits(model, prompt, silent, 3), whole) > LOGIT_TOLERANCE,
          "audio embeddings replace the placeholder rows");
    TranscribeDecoder decoder(model, 3);
    check(decoder.context() % CACHE_ALIGNMENT == 0 && decoder.context() >= 3, "context is padded for alignment");
    expect_failure([&] { prefill_logits(model, prompt, std::vector<float>(TEXT_EMBD), 4); }, "audio",
                   "missing audio rows rejected");
    expect_failure([&] { prefill_logits(model, prompt, std::vector<float>(7), 4); }, "decoder width",
                   "ragged audio rows rejected");
    expect_failure([&] { TranscribeDecoder oversized(model, TEXT_CTX + 1); }, "context", "context past training");
    std::filesystem::remove(path);
}

parakeet::moss::TranscribeOptions tiny_options(const std::filesystem::path & path) {
    parakeet::moss::TranscribeOptions options;
    options.model_path = path.string();
    options.n_threads = 1;
    return options;
}

void test_engine_end_to_end() {
    const auto path = write_transcribe_model("transcribe-engine", WEIGHT_SEED);
    parakeet::moss::TranscribeEngine engine(tiny_options(path));
    const std::vector<float> audio = random_signal(CHUNK_SAMPLES * 2 + 40, AUDIO_SEED);
    const auto result = engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE);
    check(!result.cancelled, "a tiny transcription completes");
    check(result.audio_tokens == 2 * CHUNK_TOKENS + 2, "audio tokens cover full and partial chunks");
    check(result.generated_tokens <= DEFAULT_MAX_NEW_TOKENS, "the model default caps generation");
    check(result.prompt_tokens > result.audio_tokens, "the prompt wraps the audio span");
    parakeet::moss::TranscribeRequest request;
    request.max_new_tokens = 2;
    check(engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE, request).generated_tokens <= 2,
          "max_new_tokens overrides the default");
    check(engine.sample_rate() == SAMPLE_RATE, "sample rate comes from the model");
    std::filesystem::remove(path);
}

void test_engine_rejections() {
    const auto path = write_transcribe_model("transcribe-reject", WEIGHT_SEED);
    parakeet::moss::TranscribeEngine engine(tiny_options(path));
    const std::vector<float> audio = random_signal(CHUNK_SAMPLES, AUDIO_SEED);
    expect_failure([&] { engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE * 2); }, "sampled at",
                   "wrong sample rate rejected");
    expect_failure([&] { engine.transcribe(audio.data(), 0, SAMPLE_RATE); }, "empty", "empty audio rejected");
    parakeet::moss::TranscribeRequest negative;
    negative.max_new_tokens = -1;
    expect_failure([&] { engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE, negative); }, "max_new_tokens",
                   "negative max_new_tokens rejected");
    parakeet::moss::TranscribeRequest huge;
    huge.max_new_tokens = 0x7fffffff;
    expect_failure([&] { engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE, huge); }, "decoder holds",
                   "a token budget past the decoder context is rejected");
    parakeet::moss::TranscribeRequest long_prompt;
    long_prompt.prompt = std::string(10000, 'x');
    expect_failure([&] { engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE, long_prompt); }, "prompt",
                   "oversized prompt rejected");
    expect_failure([] { parakeet::moss::TranscribeEngine missing(parakeet::moss::TranscribeOptions{}); }, "model_path",
                   "model path required");
    std::filesystem::remove(path);
}

void test_engine_hotwords() {
    const auto path = write_transcribe_model("transcribe-hotwords", WEIGHT_SEED);
    parakeet::moss::TranscribeEngine engine(tiny_options(path));
    const std::vector<float> audio = random_signal(CHUNK_SAMPLES, AUDIO_SEED);
    parakeet::moss::TranscribeRequest plain;
    plain.max_new_tokens = 1;
    parakeet::moss::TranscribeRequest hinted = plain;
    hinted.hotwords = {"QVAC", "vcpkg"};
    const int extra = (int) std::string(" Hotwords: QVAC, vcpkg").size();
    check(engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE, hinted).prompt_tokens ==
          engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE, plain).prompt_tokens + extra,
          "hotwords add their byte tokens to the prompt");
    parakeet::moss::TranscribeRequest blank = plain;
    blank.hotwords = {" ", "[]"};
    check(engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE, blank).prompt_tokens ==
          engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE, plain).prompt_tokens,
          "hotwords that sanitize to nothing keep the default prompt");
    parakeet::moss::TranscribeRequest mixed = hinted;
    mixed.prompt = "custom";
    expect_failure([&] { engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE, mixed); }, "custom prompt",
                   "hotwords with a custom prompt are rejected");
    std::filesystem::remove(path);
}

void test_engine_rejects_over_context_audio() {
    const auto path = write_transcribe_model("transcribe-short-context", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-transcribe.text.context_length", SHORT_CONTEXT);
    });
    parakeet::moss::TranscribeEngine engine(tiny_options(path));
    const std::vector<float> audio = random_signal((size_t) CHUNK_SAMPLES * OVER_CONTEXT_CHUNKS, AUDIO_SEED);
    expect_failure([&] { engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE); }, "decoder holds",
                   "audio past the decoder context is rejected");
    std::filesystem::remove(path);
}

void test_engine_cancel() {
    const auto path = write_transcribe_model("transcribe-cancel", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-transcribe.token.im_end", TEXT_VOCAB - 1);
    });
    parakeet::moss::TranscribeEngine engine(tiny_options(path));
    const std::vector<float> audio = random_signal(CHUNK_SAMPLES, AUDIO_SEED);
    int calls = 0;
    const auto result = engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE, {},
            [&](int, int) { return ++calls < 2; });
    check(result.cancelled && result.generated_tokens == 2, "the progress callback stops generation");
    std::filesystem::remove(path);
}

void test_cli_flags() {
    using parakeet::moss::cli::TranscribeCliArgs;
    using parakeet::moss::cli::parse_args;
    const char * full[] = {"moss-transcribe", "--model", "t.gguf", "--audio", "a.wav", "--prompt", "p",
                           "--max-new-tokens", "9", "--gpu", "--threads", "3", "--out", "o.json"};
    TranscribeCliArgs args;
    check(parse_args(14, full, args), "CLI flags parse");
    check(args.options.model_path == "t.gguf" && args.audio_path == "a.wav" && args.request.prompt == "p" &&
          args.request.max_new_tokens == 9 && args.options.use_gpu && args.options.n_threads == 3 &&
          args.out_path == "o.json", "CLI values land in the options and the request");
    const char * explicit_backend[] = {"moss-transcribe", "--model", "t.gguf", "--audio", "a.wav",
                                       "--backend", "vulkan"};
    TranscribeCliArgs vulkan;
    check(parse_args(7, explicit_backend, vulkan) && vulkan.options.backend == "vulkan" && !vulkan.options.use_gpu,
          "--backend selects Vulkan independently of --gpu");
    const char * hinted[] = {"moss-transcribe", "--model", "t.gguf", "--audio", "a.wav", "--hotwords", "QVAC,,vcpkg"};
    TranscribeCliArgs hotwords;
    check(parse_args(7, hinted, hotwords) &&
          hotwords.request.hotwords == std::vector<std::string>{"QVAC", "", "vcpkg"},
          "--hotwords splits on commas");
    const char * no_audio[] = {"moss-transcribe", "--model", "t.gguf"};
    TranscribeCliArgs missing;
    check(!parse_args(3, no_audio, missing), "--audio is required");
    const char * unknown[] = {"moss-transcribe", "--model", "t.gguf", "--audio", "a.wav", "--stream"};
    TranscribeCliArgs rejected;
    check(!parse_args(6, unknown, rejected), "unknown flags are rejected");
    const char * dangling[] = {"moss-transcribe", "--model"};
    TranscribeCliArgs incomplete;
    expect_failure([&] { parse_args(2, dangling, incomplete); }, "missing value", "a flag without a value");
}

void test_transcript_json_escapes() {
    parakeet::moss::TranscribeResult result;
    result.text = "say \"hi\"\n";
    const std::string json = parakeet::moss::cli::transcript_json(result);
    check(json.find("\"say \\\"hi\\\"\\n\"") != std::string::npos, "transcript JSON escapes quotes and newlines");
    check(json.find("\"segments\": []") != std::string::npos, "an empty segment list stays valid JSON");
}

void test_transcript_json() {
    parakeet::moss::TranscribeResult result;
    result.text = "[0][S01]hi[1]";
    result.segments = {{0.0, 1.0, "S01", "hi"}};
    const std::string json = parakeet::moss::cli::transcript_json(result);
    check(json.find("\"speaker\": \"S01\"") != std::string::npos && json.find("\"text\": \"hi\"") != std::string::npos,
          "transcript JSON carries the segments");
}

bool comma_decimal_active() {
    return std::strtod("1.25", nullptr) != 1.25;
}

bool try_comma_decimal_locale(const char * name) {
    if (std::setlocale(LC_ALL, name) == nullptr) {
        return false;
    }
    try {
        std::locale::global(std::locale(name));
    } catch (const std::runtime_error &) {
        std::setlocale(LC_ALL, "C");
        return false;
    }
    return comma_decimal_active();
}

bool enter_comma_decimal_locale() {
    for (const char * name : COMMA_DECIMAL_LOCALES) {
        if (try_comma_decimal_locale(name)) {
            return true;
        }
    }
    return false;
}

void leave_comma_decimal_locale() {
    std::setlocale(LC_ALL, "C");
    std::locale::global(std::locale::classic());
}

void test_locale_independent_numbers() {
    const bool exercised = enter_comma_decimal_locale();
    if (!exercised) {
        std::printf("note: no comma-decimal locale installed; numbers checked in the C locale only\n");
    }
    const auto segments = parse_transcript("[0.50][S01] hello [1.25]");
    check(segments.size() == 1 && segments[0].start_s == 0.5 && segments[0].end_s == 1.25,
          "timestamps parse the same under a comma-decimal locale");
    parakeet::moss::TranscribeResult result;
    result.segments = {{0.5, 1.25, "S01", "hello"}};
    const std::string json = parakeet::moss::cli::transcript_json(result);
    check(json.find("\"start\": 0.5, \"end\": 1.25") != std::string::npos,
          "transcript JSON keeps a dot decimal under a comma-decimal locale");
    check(parse_decimal("2.75") == 2.75 && format_decimal(2.75) == "2.75", "decimal helpers ignore the locale");
    leave_comma_decimal_locale();
}

class FakeEncoderSidecar final : public TranscribeEncoderSidecar {
public:
    FakeEncoderSidecar(std::vector<float> rows, int fail_on_call, int * calls)
        : rows_(std::move(rows)), fail_on_call_(fail_on_call), calls_(calls) {}

    bool encode(const std::vector<float> &, std::vector<float> & embeddings) override {
        const int call = (*calls_)++;
        if (call == fail_on_call_) {
            return false;
        }
        embeddings = rows_;
        return true;
    }

    const char * label() const override {
        return FAKE_SIDECAR_LABEL;
    }

private:
    std::vector<float> rows_;
    int fail_on_call_;
    int * calls_;
};

std::unique_ptr<TranscribeEncoderSidecar> fake_sidecar(std::vector<float> rows, int fail_on_call, int * calls) {
    return std::make_unique<FakeEncoderSidecar>(std::move(rows), fail_on_call, calls);
}

std::vector<float> shifted(const std::vector<float> & values, float offset) {
    std::vector<float> out(values);
    std::transform(out.begin(), out.end(), out.begin(), [offset](float v) { return v + offset; });
    return out;
}

void test_coreml_frame_major_mel() {
    const std::vector<float> feature_major = {1, 2, 3, 4, 5, 6};
    check(frame_major_mel(feature_major, 2, 3) == std::vector<float>({1, 4, 2, 5, 3, 6}),
          "the Core ML input is the mel window with frames outermost");
    expect_failure([&] { frame_major_mel(feature_major, 4, 2); }, "wrong size", "a ragged mel window is rejected");
}

void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void clear_env(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

void test_coreml_policy() {
    clear_env(COREML_DISABLE_ENV);
    clear_env(COREML_STRICT_ENV);
    const CoremlPolicy defaults = read_coreml_policy();
    check(!defaults.disabled && !defaults.strict, "the sidecar policy defaults to enabled and lenient");
    set_env(COREML_DISABLE_ENV, "1");
    set_env(COREML_STRICT_ENV, "1");
    const CoremlPolicy forced = read_coreml_policy();
    check(forced.disabled && forced.strict, "MOSS_COREML_DISABLE and MOSS_COREML_STRICT are read from the environment");
    clear_env(COREML_DISABLE_ENV);
    clear_env(COREML_STRICT_ENV);
}

void test_coreml_sidecar_lookup() {
    const auto path = write_transcribe_model("transcribe-sidecar-lookup", WEIGHT_SEED);
    TranscribeModel model(path.string(), false, 1);
    check(transcribe_encoder_sidecar_path("models/moss-transcribe-diarize-q8_0.gguf") ==
          "models/moss-transcribe-diarize-encoder.mlmodelc", "the sidecar name drops the quantization tag");
    check(open_transcribe_encoder_sidecar(path.string(), model.config(), {}) == nullptr,
          "no sidecar directory keeps the ggml encoder");
    const std::filesystem::path sidecar = transcribe_encoder_sidecar_path(path.string());
    std::filesystem::create_directories(sidecar);
    check(open_transcribe_encoder_sidecar(path.string(), model.config(), {true, false}) == nullptr,
          "MOSS_COREML_DISABLE skips a staged sidecar");
    check(open_transcribe_encoder_sidecar(path.string(), model.config(), {}) == nullptr,
          "a directory that is not a loadable model keeps the ggml encoder");
    std::filesystem::remove_all(sidecar);
    std::filesystem::remove(path);
}

void test_coreml_encoder_routing() {
    const auto path = write_transcribe_model("transcribe-sidecar-routing", WEIGHT_SEED);
    TranscribeModel model(path.string(), false, 1);
    const std::vector<float> mel = tiny_mel(model, random_signal(CHUNK_SAMPLES, AUDIO_SEED));
    const std::vector<float> ggml = encode_audio_chunk(model, mel, CHUNK_TOKENS, false).embeddings;
    const std::vector<float> sidecar_rows = shifted(ggml, 1.0f);
    int calls = 0;
    TranscribeAudioEncoder encoder(model, fake_sidecar(sidecar_rows, NEVER_FAIL, &calls), false);
    encoder.begin_run();
    const std::vector<float> two = encoder.encode(mel, 2);
    check(encoder.on_coreml() && calls == 1, "an attached sidecar encodes the window");
    check(two == std::vector<float>(sidecar_rows.begin(), sidecar_rows.begin() + 2 * TEXT_EMBD),
          "a short window keeps the sidecar's leading tokens");
    check(encoder.run_backend() == FAKE_SIDECAR_LABEL, "the run reports the sidecar label");
    expect_failure([&] { encoder.encode(mel, CHUNK_TOKENS + 1); }, "encoder window",
                   "more tokens than the sidecar window are rejected");
    std::filesystem::remove(path);
}

void test_coreml_encoder_fallback() {
    const auto path = write_transcribe_model("transcribe-sidecar-fallback", WEIGHT_SEED);
    TranscribeModel model(path.string(), false, 1);
    const std::vector<float> mel = tiny_mel(model, random_signal(CHUNK_SAMPLES, AUDIO_SEED));
    const std::vector<float> ggml = encode_audio_chunk(model, mel, CHUNK_TOKENS, false).embeddings;
    int calls = 0;
    TranscribeAudioEncoder encoder(model, fake_sidecar(shifted(ggml, 1.0f), 1, &calls), false);
    encoder.begin_run();
    encoder.encode(mel, CHUNK_TOKENS);
    const std::vector<float> fallback = encoder.encode(mel, CHUNK_TOKENS);
    check(max_abs_diff(fallback, ggml) == 0.0f, "a failed prediction reruns the window on ggml");
    check(!encoder.on_coreml(), "a failed sidecar is retired");
    check(encoder.run_backend() == MIXED_ENCODER_BACKEND, "a run that used both encoders reports mixed");
    encoder.begin_run();
    encoder.encode(mel, CHUNK_TOKENS);
    check(calls == 2 && encoder.run_backend() == GGML_ENCODER_BACKEND,
          "a retired sidecar is not tried again and later runs report ggml");
    std::filesystem::remove(path);
}

void test_coreml_encoder_strict() {
    const auto path = write_transcribe_model("transcribe-sidecar-strict", WEIGHT_SEED);
    TranscribeModel model(path.string(), false, 1);
    const std::vector<float> mel = tiny_mel(model, random_signal(CHUNK_SAMPLES, AUDIO_SEED));
    int calls = 0;
    TranscribeAudioEncoder failing(model, fake_sidecar({}, 0, &calls), true);
    expect_failure([&] { failing.encode(mel, CHUNK_TOKENS); }, COREML_STRICT_ENV,
                   "a failed prediction fails the window under MOSS_COREML_STRICT");
    TranscribeAudioEncoder absent(model, nullptr, true);
    expect_failure([&] { absent.encode(mel, CHUNK_TOKENS); }, COREML_STRICT_ENV,
                   "a missing sidecar fails the window under MOSS_COREML_STRICT");
    std::filesystem::remove(path);
}

void test_engine_reports_ggml_encoder() {
    const auto path = write_transcribe_model("transcribe-encoder-backend", WEIGHT_SEED);
    parakeet::moss::TranscribeEngine engine(tiny_options(path));
    const std::vector<float> audio = random_signal(CHUNK_SAMPLES, AUDIO_SEED);
    parakeet::moss::TranscribeRequest request;
    request.max_new_tokens = 1;
    const auto result = engine.transcribe(audio.data(), audio.size(), SAMPLE_RATE, request);
    check(!engine.encoder_on_coreml(), "an engine without a sidecar encodes on ggml");
    check(result.encoder_backend == GGML_ENCODER_BACKEND, "the result reports the ggml encoder");
    check(parakeet::moss::cli::transcript_json(result).find("\"encoder_backend\": \"ggml\"") != std::string::npos,
          "transcript JSON carries the encoder backend");
    std::filesystem::remove(path);
}

double relative_l2(const std::vector<float> & reference, const std::vector<float> & value) {
    if (reference.size() != value.size() || reference.empty()) {
        return INFINITY;
    }
    double error = 0.0, norm = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double d = (double) reference[i] - value[i];
        error += d * d;
        norm += (double) reference[i] * reference[i];
    }
    return std::sqrt(error / std::max(norm, 1e-20));
}

bool chunks_tile(const std::vector<TranscribeHeadChunk> & chunks, int vocab, int max_rows) {
    int next = 0;
    for (const TranscribeHeadChunk & chunk : chunks) {
        if (chunk.row0 != next || chunk.rows < 1 || chunk.rows > max_rows) return false;
        next += chunk.rows;
    }
    return next == vocab;
}

void test_head_plan() {
    const auto split = plan_transcribe_head(TEXT_VOCAB, SPLIT_HEAD_ROWS);
    check(split.size() == SPLIT_HEAD_CHUNKS && split.back().rows == SPLIT_TAIL_ROWS,
          "head chunks stop at the row cap and keep the remainder");
    check(chunks_tile(split, TEXT_VOCAB, SPLIT_HEAD_ROWS), "head chunks tile the vocabulary without gaps");
    check(plan_transcribe_head(TEXT_VOCAB, TEXT_VOCAB).size() == 1, "a backend with no row cap keeps one chunk");
    expect_failure([] { plan_transcribe_head(0, SPLIT_HEAD_ROWS); }, "head chunks", "empty vocabulary rejected");
    expect_failure([] { plan_transcribe_head(TEXT_VOCAB, 0); }, "head chunks", "zero-row chunks rejected");
}

std::vector<float> decode_steps(TranscribeDecoder & decoder) {
    std::vector<float> logits;
    for (int step = 0; step < DECODE_STEPS; ++step) {
        const std::vector<float> next = decoder.step(byte_token('a' + step));
        logits.insert(logits.end(), next.begin(), next.end());
    }
    return logits;
}

std::vector<float> decoded_logits(TranscribeModel & model, int batch) {
    const std::vector<int32_t> prompt = tiny_prompt(CHUNK_TOKENS);
    TranscribeDecoder decoder(model, (int) prompt.size() + DECODE_STEPS);
    std::vector<float> logits = decoder.prefill(prompt, random_signal((size_t) CHUNK_TOKENS * TEXT_EMBD, AUDIO_SEED),
            batch);
    const std::vector<float> steps = decode_steps(decoder);
    logits.insert(logits.end(), steps.begin(), steps.end());
    return logits;
}

void check_split_table(const std::filesystem::path & path, const std::string & precision) {
    TranscribeModel whole(path.string(), false, 1);
    TranscribeModel split(path.string(), false, 1, "auto", false, TranscribeLayout{SPLIT_HEAD_ROWS});
    check(!whole.embeds_on_host() && whole.head().size() == 1, precision + ": the CPU gathers from one table");
    check(split.embeds_on_host() && split.head().size() == SPLIT_HEAD_CHUNKS,
          precision + ": a capped head splits the table");
    check(max_abs_diff(decoded_logits(split, 3), decoded_logits(whole, 3)) == 0.0f,
          precision + ": host lookup and chunked head reproduce the whole table");
    expect_failure([&] { split.embed_tokens({TEXT_VOCAB}); }, "vocabulary", precision + ": host lookup bounds ids");
    TranscribeModel measured(path.string(), false, 1, "auto", true, TranscribeLayout{SPLIT_HEAD_ROWS});
    check(measured.measure_weights().device.weights_bytes == split.weight_bytes(),
          precision + ": the projection counts the head chunks");
}

void test_split_token_table() {
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        const auto path = write_transcribe_model("transcribe-split", WEIGHT_SEED);
        moss_fixture_precision(path, type);
        check_split_table(path, ggml_type_name(type));
        std::filesystem::remove(path);
    }
}

void test_fused_attention() {
    const auto path = write_transcribe_model("transcribe-fused", WEIGHT_SEED);
    TranscribeModel unfused(path.string(), false, 1);
    TranscribeModel fused(path.string(), false, 1, "auto", false, TranscribeLayout{0, TranscribeAttention::Fused});
    check(!unfused.fused_attention() && fused.fused_attention(), "attention layout follows the override");
    const std::vector<float> mel = tiny_mel(unfused, random_signal(CHUNK_SAMPLES, AUDIO_SEED));
    const auto a = encode_audio_chunk(unfused, mel, CHUNK_TOKENS, true);
    const auto b = encode_audio_chunk(fused, mel, CHUNK_TOKENS, true);
    check(relative_l2(a.encoder_states, b.encoder_states) < FUSED_RELATIVE_TOLERANCE,
          "fused encoder attention matches the unfused graph");
    const std::vector<float> reference = decoded_logits(unfused, 3);
    for (int batch : {1, 3, CHUNK_TOKENS + 5}) {
        check(relative_l2(reference, decoded_logits(fused, batch)) < FUSED_RELATIVE_TOLERANCE,
              "fused decoder attention matches with prefill batch " + std::to_string(batch));
    }
    std::filesystem::remove(path);
}

void test_contiguous_attention_heads() {
    const auto path = write_transcribe_model("transcribe-contiguous", WEIGHT_SEED);
    TranscribeModel views(path.string(), false, 1);
    TranscribeModel contiguous(path.string(), false, 1, "auto", false,
                               TranscribeLayout{0, TranscribeAttention::UnfusedContiguousHeads});
    check(!views.contiguous_attention_heads() && contiguous.contiguous_attention_heads() &&
          !contiguous.fused_attention(), "contiguous heads keep the unfused graph");
    for (int batch : {1, 3, CHUNK_TOKENS + 5}) {
        check(max_abs_diff(decoded_logits(views, batch), decoded_logits(contiguous, batch)) < LOGIT_TOLERANCE,
              "contiguous attention heads match the cache views with prefill batch " + std::to_string(batch));
    }
    std::filesystem::remove(path);
}

bool rejects_strided_copy(ggml_backend * backend, const ggml_tensor * copy) {
    return ggml_is_contiguous(copy->src[0]) && ggml_backend_supports_op(backend, copy);
}

std::vector<float> counting_values(size_t count) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = (float) (i + 1);
    }
    return values;
}

std::vector<ggml_fp16_t> untouched_cache() {
    std::vector<ggml_fp16_t> cache((size_t) CACHE_POSITIONS * CACHE_WIDTH);
    std::fill(cache.begin(), cache.end(), ggml_fp32_to_fp16(UNTOUCHED));
    return cache;
}

struct CacheCopyRun {
    std::vector<float> cache;
    bool materialized = false;
};

std::vector<float> read_cache(ggml_tensor * cache) {
    std::vector<ggml_fp16_t> halves((size_t) ggml_nelements(cache));
    ggml_backend_tensor_get(cache, halves.data(), 0, ggml_nbytes(cache));
    std::vector<float> values(halves.size());
    ggml_fp16_to_fp32_row(halves.data(), values.data(), (int64_t) halves.size());
    return values;
}

CacheCopyRun run_cache_copy(ggml_backend_t cpu, CopySupport supports) {
    ggml_context * state = ggml_init({ggml_tensor_overhead(), nullptr, true});
    ggml_tensor * cache = ggml_new_tensor_2d(state, GGML_TYPE_F16, CACHE_POSITIONS, CACHE_WIDTH);
    ggml_backend_buffer_t cache_buffer = ggml_backend_alloc_ctx_tensors(state, cpu);
    const std::vector<ggml_fp16_t> initial = untouched_cache();
    ggml_backend_tensor_set(cache, initial.data(), 0, ggml_nbytes(cache));

    TranscribeGraph graph(COPY_GRAPH_NODES);
    ggml_tensor * rows = graph.input_f32(CACHE_WIDTH, COPY_TOKENS);
    ggml_tensor * columns = ggml_view_2d(graph.ctx(), cache, COPY_TOKENS, CACHE_WIDTH, cache->nb[1],
            (size_t) COPY_POSITION * ggml_element_size(cache));
    ggml_tensor * copy = copy_into_cache(graph.ctx(), cpu, ggml_transpose(graph.ctx(), rows), columns, supports);
    ggml_build_forward_expand(graph.graph(), copy);
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    ggml_gallocr_alloc_graph(allocator, graph.graph());
    const std::vector<float> values = counting_values((size_t) CACHE_WIDTH * COPY_TOKENS);
    ggml_backend_tensor_set(rows, values.data(), 0, ggml_nbytes(rows));
    ggml_backend_graph_compute(cpu, graph.graph());

    CacheCopyRun run{read_cache(cache), copy->src[0]->op == GGML_OP_CONT};
    ggml_gallocr_free(allocator);
    ggml_backend_buffer_free(cache_buffer);
    ggml_free(state);
    return run;
}

float expected_cache_value(int channel, int position) {
    const int token = position - COPY_POSITION;
    if (token < 0 || token >= COPY_TOKENS) {
        return UNTOUCHED;
    }
    return (float) (token * CACHE_WIDTH + channel + 1);
}

bool cache_holds_columns(const std::vector<float> & cache) {
    for (int channel = 0; channel < CACHE_WIDTH; ++channel) {
        for (int position = 0; position < CACHE_POSITIONS; ++position) {
            if (cache[(size_t) channel * CACHE_POSITIONS + position] != expected_cache_value(channel, position)) {
                return false;
            }
        }
    }
    return true;
}

void test_cache_copy_materializes_rejected_source() {
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    check(cpu != nullptr, "the CPU backend initializes");
    if (!cpu) return;
    const CacheCopyRun direct = run_cache_copy(cpu, ggml_backend_supports_op);
    const CacheCopyRun materialized = run_cache_copy(cpu, rejects_strided_copy);
    check(!direct.materialized, "a backend that copies strided sources keeps one copy node");
    check(materialized.materialized, "a backend that rejects strided sources copies a materialized source");
    check(cache_holds_columns(direct.cache), "the transposed copy fills only its cache columns");
    check(materialized.cache == direct.cache, "the materialized copy writes the same cache");
    ggml_backend_free(cpu);
}

std::vector<float> mask_values(const std::vector<uint8_t> & bytes, ggml_type type) {
    const size_t count = bytes.size() / ggml_type_size(type);
    std::vector<float> values(count);
    if (type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(bytes.data()), values.data(), (int64_t) count);
    } else {
        std::memcpy(values.data(), bytes.data(), bytes.size());
    }
    return values;
}

bool mask_is_causal(const std::vector<float> & values) {
    const int total = MASK_POSITION + MASK_TOKENS;
    for (int query = 0; query < MASK_TOKENS; ++query) {
        for (int key = 0; key < total; ++key) {
            const float value = values[(size_t) query * total + key];
            const bool visible = key <= MASK_POSITION + query;
            if (visible ? value != 0.0f : !(std::isinf(value) && value < 0.0f)) return false;
        }
    }
    return true;
}

void check_mask_projection(const TranscribeModel & model, ggml_type type, const std::string & layout) {
    const int64_t total = MASK_POSITION + MASK_TOKENS;
    const std::vector<uint8_t> mask = build_causal_mask(model, MASK_POSITION, MASK_TOKENS);
    check(transcribe_mask_type(model) == type, layout + ": the mask has the attention input's type");
    check(mask.size() == transcribe_mask_bytes(model, total, MASK_TOKENS) &&
          mask.size() == (size_t) total * MASK_TOKENS * ggml_type_size(type),
          layout + ": the projection counts exactly the one mask buffer the decoder uploads");
    check(mask_is_causal(mask_values(mask, type)), layout + ": the mask hides only later positions");
}

void test_mask_projection() {
    const auto path = write_transcribe_model("transcribe-mask", WEIGHT_SEED);
    TranscribeModel unfused(path.string(), false, 1);
    TranscribeModel fused(path.string(), false, 1, "auto", false, TranscribeLayout{0, TranscribeAttention::Fused});
    check_mask_projection(unfused, GGML_TYPE_F32, "unfused");
    check_mask_projection(fused, GGML_TYPE_F16, "fused");
    std::filesystem::remove(path);
}

void test_upload_plan() {
    check(plan_upload(true, true) == TensorUpload::InPlace && plan_upload(true, false) == TensorUpload::InPlace,
          "host buffers read tensors in place without a staging copy");
    check(plan_upload(false, false) == TensorUpload::Chunked, "device float tensors stream in bounded chunks");
    check(plan_upload(false, true) == TensorUpload::Whole,
          "device quantized tensors upload whole, as the OpenCL and repacking backends require");
}

} // namespace

int main() {
    try {
        test_fft_matches_dft();
        test_log_mel_normalization();
        test_chunk_arithmetic();
        test_audio_span_markers();
        test_prompt_layout();
        test_detokenizer();
        test_pretokenizer_unicode();
        test_hotword_sanitation();
        test_parser_basic();
        test_parser_edge_cases();
        test_parser_streaming();
        test_model_loads();
        test_model_rejections();
        test_network_rejections();
        test_token_table_width_rejected();
        test_encoder_chunk();
        test_decoder_batches();
        test_engine_end_to_end();
        test_engine_rejections();
        test_engine_hotwords();
        test_engine_rejects_over_context_audio();
        test_engine_cancel();
        test_cli_flags();
        test_transcript_json();
        test_transcript_json_escapes();
        test_locale_independent_numbers();
        test_upload_plan();
        test_cache_copy_materializes_rejected_source();
        test_contiguous_attention_heads();
        test_head_plan();
        test_split_token_table();
        test_fused_attention();
        test_mask_projection();
        test_coreml_frame_major_mel();
        test_coreml_policy();
        test_coreml_sidecar_lookup();
        test_coreml_encoder_routing();
        test_coreml_encoder_fallback();
        test_coreml_encoder_strict();
        test_engine_reports_ggml_encoder();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "unexpected: %s\n", e.what());
        return 1;
    }
    if (failures == 0) {
        std::printf("moss transcribe: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss transcribe: %d failures\n", failures);
    return 1;
}
