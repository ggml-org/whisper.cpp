#include "speech_fixtures.h"

#include "moss/cli.h"
#include "moss/speech_generation.h"
#include "moss/speech_lm.h"
#include "moss/speech_prompt.h"
#include "moss/speech_request.h"
#include "moss/speech_tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace moss_speech_fixtures;
using namespace tts_cpp::moss::detail;

namespace {

constexpr uint32_t WEIGHT_SEED = 20260928;
constexpr uint32_t AUDIO_SEED = 11;
constexpr float TOLERANCE = 1e-4f;
constexpr float LOW = -10.0f;
constexpr float HIGH = 10.0f;
constexpr int NEVER_FAIL = -1;
constexpr int CODE_STRIDE = 5;
constexpr int CODE_OFFSET = 2;
constexpr const char * FAKE_SIDECAR_LABEL = "coreml-fake";

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

SpeechTokens fixture_tokens() {
    return {TOKEN_TEXT_PLACEHOLDER, TOKEN_AUDIO_PAD, TOKEN_SPEECH_START, TOKEN_SPEECH_END, TOKEN_IM_START,
            TOKEN_IM_END, TOKEN_PAD};
}

std::vector<float> peaked(size_t size, int32_t winner) {
    std::vector<float> logits(size, LOW);
    logits[(size_t) winner] = HIGH;
    return logits;
}

SpeechLogits logits_for(int32_t text, int32_t audio) {
    return {peaked(moss_fixtures::TEXT_VOCAB, text), peaked(AUDIO_VOCAB, audio)};
}

const SpeechSampling GREEDY{false, 1.0f, 1.0f, 0};

void test_generation_switches_channels() {
    std::mt19937 rng(0);
    SpeechGenerationState state(fixture_tokens(), {TOKEN_SPEECH_START, TOKEN_AUDIO_PAD}, {20, 0, 0});
    const SpeechRow first = state.next(logits_for(byte_token('x'), 7), GREEDY, rng);
    check(state.channel() == SpeechChannel::Audio, "a speech-start prompt opens the audio channel");
    check(first.text == TOKEN_TEXT_PLACEHOLDER && first.audio == 7, "audio rows carry the text placeholder");
    state.next(logits_for(byte_token('x'), TOKEN_SPEECH_END), GREEDY, rng);
    const SpeechRow text = state.next(logits_for(byte_token('y'), 3), GREEDY, rng);
    check(state.channel() == SpeechChannel::Text && text.text == byte_token('y'),
          "a speech-end code hands the turn back to text");
    state.next(logits_for(TOKEN_IM_END, 3), GREEDY, rng);
    check(state.stopping(), "im_end stops generation");
    check(speech_reply_codes(state.generated(), fixture_tokens()) == std::vector<int32_t>{7},
          "reply codes stop at the first speech-end");
}

void test_generation_constraints() {
    std::mt19937 rng(0);
    SpeechGenerationState early(fixture_tokens(), {TOKEN_SPEECH_START, TOKEN_AUDIO_PAD}, {20, 3, 0});
    std::vector<float> audio = peaked(AUDIO_VOCAB, TOKEN_SPEECH_END);
    audio[9] = HIGH - 1.0f;
    check(early.next({peaked(moss_fixtures::TEXT_VOCAB, 0), audio}, GREEDY, rng).audio == 9,
          "speech-end is masked before min_new_tokens");
    std::vector<float> beyond = peaked(AUDIO_VOCAB, AUDIO_VOCAB - 1);
    beyond[2] = 0.0f;
    check(early.next({peaked(moss_fixtures::TEXT_VOCAB, 0), beyond}, GREEDY, rng).audio == 2,
          "codes past speech-end are never sampled");
}

void test_generation_limits() {
    std::mt19937 rng(0);
    SpeechGenerationState capped(fixture_tokens(), {TOKEN_SPEECH_START, TOKEN_AUDIO_PAD}, {20, 0, 2});
    capped.next(logits_for(0, 7), GREEDY, rng);
    capped.next(logits_for(0, 8), GREEDY, rng);
    const SpeechRow forced = capped.next(logits_for(0, 9), GREEDY, rng);
    check(forced.audio == TOKEN_SPEECH_END && capped.truncated(), "max_reply_codes forces speech-end");
    SpeechGenerationState bounded(fixture_tokens(), {TOKEN_SPEECH_START, TOKEN_AUDIO_PAD}, {2, 0, 0});
    bounded.next(logits_for(0, 7), GREEDY, rng);
    bounded.next(logits_for(0, 7), GREEDY, rng);
    check(bounded.stopping() && bounded.truncated(), "max_new_tokens stops generation and marks it truncated");
    check(speech_reply_codes(bounded.generated(), fixture_tokens()) == std::vector<int32_t>{7, 7},
          "a length-limited reply keeps its last code");
    SpeechGenerationState single(fixture_tokens(), {TOKEN_SPEECH_START, TOKEN_AUDIO_PAD}, {1, 0, 0});
    single.next(logits_for(0, 9), GREEDY, rng);
    check(speech_reply_codes(single.generated(), fixture_tokens()) == std::vector<int32_t>{9},
          "a one-token limit still yields its code");
    SpeechGenerationState text(fixture_tokens(), {byte_token('a'), TOKEN_AUDIO_PAD}, {20, 0, 0});
    text.next(logits_for(TOKEN_PAD, 7), GREEDY, rng);
    check(text.stopping() && text.channel() == SpeechChannel::Text, "the pad token stops a text reply");
}

void test_generation_text_minimum() {
    std::mt19937 rng(0);
    SpeechGenerationState state(fixture_tokens(), {byte_token('a'), TOKEN_AUDIO_PAD}, {20, 2, 0});
    std::vector<float> text = peaked(moss_fixtures::TEXT_VOCAB, TOKEN_IM_END);
    text[(size_t) byte_token('b')] = HIGH - 1.0f;
    check(state.next({text, peaked(AUDIO_VOCAB, 3)}, GREEDY, rng).text == byte_token('b'),
          "im_end is masked before min_new_tokens");
    check(state.next({text, peaked(AUDIO_VOCAB, 3)}, GREEDY, rng).text == byte_token('b') && !state.stopping(),
          "im_end stays masked for min_new_tokens steps");
    state.next({text, peaked(AUDIO_VOCAB, 3)}, GREEDY, rng);
    check(state.stopping(), "im_end stops generation once the minimum is met");
}

void test_reply_codes_follow_speech() {
    const std::vector<SpeechRow> spoken_after_text{{byte_token('a'), 9}, {TOKEN_SPEECH_START, 9},
        {TOKEN_TEXT_PLACEHOLDER, 7}, {TOKEN_TEXT_PLACEHOLDER, 8}, {TOKEN_TEXT_PLACEHOLDER, TOKEN_SPEECH_END},
        {TOKEN_IM_END, 3}};
    check(speech_reply_codes(spoken_after_text, fixture_tokens()) == std::vector<int32_t>{7, 8},
          "reply codes start at the first speech row");
    const std::vector<SpeechRow> text_only{{byte_token('a'), 9}, {byte_token('b'), 10}, {TOKEN_IM_END, 3}};
    check(speech_reply_codes(text_only, fixture_tokens()).empty(), "a text reply carries no speech codes");
}

void test_generation_heads() {
    std::mt19937 rng(0);
    SpeechGenerationState state(fixture_tokens(), {TOKEN_SPEECH_START, TOKEN_AUDIO_PAD}, {20, 0, 0});
    check(state.upcoming_channel() == SpeechChannel::Audio, "speech-start announces the audio channel");
    const SpeechRow row = state.next({{}, peaked(AUDIO_VOCAB, 7)}, GREEDY, rng);
    check(row.text == TOKEN_TEXT_PLACEHOLDER && row.audio == 7, "an audio step needs no text logits");
    state.next({{}, peaked(AUDIO_VOCAB, TOKEN_SPEECH_END)}, GREEDY, rng);
    check(state.upcoming_channel() == SpeechChannel::Text, "speech-end announces the text channel");
    expect_failure([&] { state.next({{}, peaked(AUDIO_VOCAB, 3)}, GREEDY, rng); }, "text logits",
                   "a text step without text logits");
}

SpeechLmConfig fixture_config() {
    SpeechLmConfig config;
    config.tokens = fixture_tokens();
    config.audio_system_prompt = AUDIO_SYSTEM_PROMPT;
    config.text_system_prompt = TEXT_SYSTEM_PROMPT;
    return config;
}

SpeechTextTokenizer fixture_tokenizer() {
    return SpeechTextTokenizer(moss_fixtures::byte_complete_vocab(), {"a b"}, token_types());
}

std::vector<SpeechRow> text_rows(const std::string & text) {
    std::vector<SpeechRow> rows;
    for (unsigned char ch : text) {
        rows.push_back({byte_token(ch), TOKEN_AUDIO_PAD});
    }
    return rows;
}

void append(std::vector<SpeechRow> & rows, const std::vector<SpeechRow> & more) {
    rows.insert(rows.end(), more.begin(), more.end());
}

bool same_rows(const std::vector<SpeechRow> & a, const std::vector<SpeechRow> & b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](const SpeechRow & x, const SpeechRow & y) {
        return x.text == y.text && x.audio == y.audio;
    });
}

std::vector<SpeechRow> header(const char * role) {
    std::vector<SpeechRow> rows{{TOKEN_IM_START, TOKEN_AUDIO_PAD}};
    append(rows, text_rows(std::string(role) + "\n"));
    return rows;
}

std::vector<SpeechRow> footer() {
    std::vector<SpeechRow> rows{{TOKEN_IM_END, TOKEN_AUDIO_PAD}};
    append(rows, text_rows("\n"));
    return rows;
}

void test_prompt_layout() {
    const SpeechTextTokenizer tokenizer = fixture_tokenizer();
    SpeechTurn user{SpeechTurnRole::User, {}, {7, 8}, true};
    std::vector<SpeechRow> expected = header("user");
    append(expected, {{TOKEN_SPEECH_START, TOKEN_AUDIO_PAD}, {TOKEN_TEXT_PLACEHOLDER, 7}, {TOKEN_TEXT_PLACEHOLDER, 8},
                      {TOKEN_TEXT_PLACEHOLDER, TOKEN_SPEECH_END}});
    append(expected, footer());
    append(expected, header("system"));
    append(expected, text_rows(AUDIO_SYSTEM_PROMPT));
    append(expected, footer());
    append(expected, header("assistant"));
    expected.push_back({TOKEN_SPEECH_START, TOKEN_AUDIO_PAD});
    check(same_rows(speech_prompt(fixture_config(), tokenizer, {user}, true), expected),
          "a spoken turn follows the processor grid, with the default system turn appended");
    SpeechTurn system{SpeechTurnRole::System, "hi", {}, false};
    const std::vector<SpeechRow> text = speech_prompt(fixture_config(), tokenizer, {system, user}, false);
    std::vector<SpeechRow> tail = header("assistant");
    check(std::equal(tail.rbegin(), tail.rend(), text.rbegin(),
                     [](const SpeechRow & x, const SpeechRow & y) { return x.text == y.text; }),
          "a text reply ends with the bare assistant header");
    check(same_rows(std::vector<SpeechRow>(text.begin(), text.begin() + 1), {{TOKEN_IM_START, TOKEN_AUDIO_PAD}}),
          "an explicit system turn comes first");
    expect_failure([&] { speech_prompt(fixture_config(), tokenizer, {}, true); }, "at least one turn",
                   "an empty conversation is rejected");
}

void test_reply_text() {
    const SpeechTextTokenizer tokenizer = fixture_tokenizer();
    std::vector<SpeechRow> rows = text_rows("ok");
    rows.insert(rows.begin(), {TOKEN_TEXT_PLACEHOLDER, 7});
    rows.push_back({TOKEN_IM_END, TOKEN_AUDIO_PAD});
    check(speech_reply_text(tokenizer, rows, fixture_tokens()) == "ok",
          "reply text skips control tokens and the stop row");
    check(speech_reply_text(tokenizer, text_rows("cut"), fixture_tokens()) == "cut",
          "a length-limited text reply keeps its last token");
}

void test_lm_loads_and_validates() {
    const auto path = write_lm("speech-lm", WEIGHT_SEED);
    SpeechLM lm(path.string(), false, 1);
    check(lm.config().n_shared_layers == 1 && lm.config().audio_vocab == AUDIO_VOCAB, "configuration loads");
    expect_failure([&] { lm.step({0, 0}); }, "begin()", "running before begin() is rejected");
    lm.begin(3);
    check(lm.context() % 256 == 0 && lm.context() >= 3, "the KV context is aligned");
    expect_failure([&] { lm.begin(CONTEXT + 1); }, "context", "a context past training is rejected");
    std::filesystem::remove(path);
}

void expect_lm_rejects(const std::filesystem::path & path, const std::string & needle, const std::string & label) {
    expect_failure([&] { SpeechLM lm(path.string(), false, 1); }, needle, label);
    std::filesystem::remove(path);
}

void test_lm_rejections() {
    expect_lm_rejects(write_lm("speech-arch", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_str(f, "general.architecture", "moss-tts-delay");
    }), "unsupported architecture", "wrong architecture");
    expect_lm_rejects(write_lm("speech-gqa", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-speech.attention.head_count_kv", 3);
    }), "geometry", "heads must divide by key heads");
    expect_lm_rejects(write_lm("speech-rope", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_f32(f, "moss-speech.rope.freq_base", 0.0f);
    }), "geometry", "a zero RoPE base");
    expect_lm_rejects(write_lm("speech-token", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-speech.token.speech_end", AUDIO_VOCAB);
    }), "special tokens", "a speech-end code outside the audio vocabulary");
    expect_lm_rejects(write_lm("speech-shape", WEIGHT_SEED, [](gguf_context *) {},
            {{"audio.output.weight", {EMBD, AUDIO_VOCAB - 1}}}), "audio.output.weight", "a mis-sized audio head");
    const auto healthy = write_lm("speech-threads", WEIGHT_SEED);
    expect_failure([&] { SpeechLM lm(healthy.string(), false, 0); }, "threads", "zero threads");
    std::filesystem::remove(healthy);
}

std::vector<SpeechRow> mixed_rows() {
    std::vector<SpeechRow> rows = text_rows("hi");
    rows.push_back({TOKEN_SPEECH_START, TOKEN_AUDIO_PAD});
    rows.push_back({TOKEN_TEXT_PLACEHOLDER, 7});
    rows.push_back({TOKEN_TEXT_PLACEHOLDER, 9});
    append(rows, text_rows("ok"));
    return rows;
}

SpeechLogits prefill(SpeechLM & lm, const std::vector<SpeechRow> & rows, int batch) {
    lm.begin((int) rows.size() + 1);
    return lm.prefill(rows, batch);
}

bool close(const SpeechLogits & a, const SpeechLogits & b) {
    return max_abs_diff(a.text, b.text) < TOLERANCE && max_abs_diff(a.audio, b.audio) < TOLERANCE;
}

void test_lm_batches_and_steps() {
    const auto path = write_lm("speech-batches", WEIGHT_SEED);
    SpeechLM lm(path.string(), false, 1);
    const std::vector<SpeechRow> rows = mixed_rows();
    const SpeechLogits whole = prefill(lm, rows, (int) rows.size());
    check(whole.text.size() == (size_t) moss_fixtures::TEXT_VOCAB && whole.audio.size() == AUDIO_VOCAB,
          "prefill returns both heads");
    lm.release_generation();
    lm.release_generation();
    check(lm.position() == 0 && lm.context() == 0, "released generation has no KV state");
    expect_failure([&] { lm.step(rows.back()); }, "begin()", "decode after release requires begin()");
    check(close(prefill(lm, rows, (int) rows.size()), whole),
          "releasing generation retains weights and permits an identical new request");
    const std::string backend = lm.backend_name();
    lm.release_weights();
    lm.release_weights();
    check(backend == lm.backend_name() && lm.config().text_vocab == moss_fixtures::TEXT_VOCAB,
          "weight eviction retains backend identity and model metadata");
    check(!lm.tokenizer_tokens().empty(), "tokenizer metadata survives weight eviction");
    expect_failure([&] { lm.step(rows.back()); }, "begin()", "decode after weight eviction requires begin()");
    check(close(prefill(lm, rows, (int) rows.size()), whole), "reloaded weights reproduce original logits");
    lm.release_weights();
    const auto moved = path.string() + ".held";
    std::filesystem::rename(path, moved);
    expect_failure([&] { lm.begin((int) rows.size() + 1); }, "reopen GGUF", "a missing reload checkpoint");
    std::filesystem::rename(moved, path);
    check(lm.context() == 0, "failed weight reload leaves no generation state");
    check(close(prefill(lm, rows, (int) rows.size()), whole), "a failed weight reload can be retried");
    check(close(prefill(lm, rows, 1), whole), "token-by-token prefill matches one batch");
    check(close(prefill(lm, rows, 3), whole), "uneven batches match one batch");
    lm.begin((int) rows.size() + 1);
    lm.prefill(std::vector<SpeechRow>(rows.begin(), rows.end() - 1), 4);
    check(close(lm.step(rows.back()), whole), "a cached step matches a full prefill");
    std::vector<SpeechRow> swapped = rows;
    swapped[3].audio = 8;
    check(!close(prefill(lm, swapped, 4), whole), "placeholder rows read the audio embedding");
    std::vector<SpeechRow> ignored = rows;
    ignored[0].audio = 8;
    check(close(prefill(lm, ignored, 4), whole), "text rows ignore the audio channel");
    int batches = 0;
    lm.begin((int) rows.size() + 1);
    lm.prefill(rows, 2, [&] { return batches++ > 0; });
    check(lm.position() == 2, "a stop request ends prefill between batches");
    const std::vector<SpeechRow> overlong((size_t) lm.context() + 1, {byte_token('a'), TOKEN_AUDIO_PAD});
    expect_failure([&] { lm.begin(1); lm.prefill(overlong, (int) overlong.size()); }, "overflow",
                   "a prefill past the aligned context");
    expect_failure([&] { lm.begin(4); lm.step({0, AUDIO_VOCAB}); }, "vocabulary", "an out-of-range code");
    std::filesystem::remove(path);
}

void test_lm_skips_the_text_head() {
    const auto path = write_lm("speech-heads", WEIGHT_SEED);
    SpeechLM lm(path.string(), false, 1);
    const std::vector<SpeechRow> rows = mixed_rows();
    const SpeechLogits whole = prefill(lm, rows, (int) rows.size());
    const std::vector<SpeechRow> head(rows.begin(), rows.end() - 2);
    lm.begin((int) rows.size() + 1);
    lm.prefill(head, 4);
    const SpeechLogits audio_only = lm.step(rows[rows.size() - 2], {false, true});
    check(audio_only.text.empty() && !audio_only.audio.empty(), "an audio-only step returns only the audio head");
    check(close(lm.step(rows.back()), whole), "an audio-only step still fills the text branch cache");
    std::filesystem::remove(path);
}

void test_generation_loop() {
    const auto path = write_lm("speech-loop", WEIGHT_SEED);
    SpeechLM lm(path.string(), false, 1);
    const std::vector<SpeechRow> prompt = mixed_rows();
    const SpeechSampling sampled{true, 0.9f, 0.95f, 5};
    auto run = [&](const SpeechContinue & keep_going, std::vector<SpeechRow> & generated) {
        lm.begin((int) prompt.size() + 8);
        SpeechGenerationState state(fixture_tokens(), prompt.back(), {8, 0, 0});
        const bool finished = run_speech_generation(lm, state, lm.prefill(prompt, 4), sampled, AUDIO_SEED, keep_going);
        generated = state.generated();
        return finished;
    };
    std::vector<SpeechRow> first;
    std::vector<SpeechRow> second;
    check(run([](int) { return true; }, first) && run([](int) { return true; }, second) &&
          same_rows(first, second), "a seed reproduces the reply");
    std::vector<SpeechRow> stopped;
    check(!run([](int generated) { return generated < 3; }, stopped) && stopped.size() == 3,
          "the progress callback stops generation");
    std::filesystem::remove(path);
}

SpeechVqConfig vq_config() {
    SpeechVqConfig config;
    config.hop_length = 160;
    config.pooling_kernel = 4;
    return config;
}

void test_segment_arithmetic() {
    const SpeechVqConfig config = vq_config();
    check(config.samples_per_token() == 1280, "one token spans 1280 samples at 16 kHz");
    check(speech_segment_tokens(config, 57600) == 45, "3.6 s of audio yields 45 tokens");
    check(speech_segment_tokens(config, 1) == 1, "a single sample yields one token");
    check(speech_padded_samples(config, 1281) == 2560, "segments pad to whole tokens");
    std::vector<float> mel = {-12.0f, 0.0f, 1.0f};
    normalize_whisper_log_mel(mel);
    check(std::fabs(mel[0] - (1.0f - 8.0f + 4.0f) / 4.0f) < 1e-6f && std::fabs(mel[2] - 1.25f) < 1e-6f,
          "log-mel clamps to max-8 and scales by (x+4)/4");
}

void test_tokenizer_encodes() {
    const auto path = write_vq("speech-vq", WEIGHT_SEED);
    SpeechTokenizer tokenizer(path.string(), false, 1);
    const std::vector<float> audio = random_signal(VQ_CHUNK + 40, AUDIO_SEED);
    const std::vector<int32_t> codes = tokenizer.encode(audio);
    const size_t per_token = (size_t) tokenizer.config().samples_per_token();
    check(codes.size() == VQ_CHUNK / per_token + (40 + per_token - 1) / per_token, "every segment emits its tokens");
    check(std::all_of(codes.begin(), codes.end(), [](int32_t c) { return c >= 0 && c < VQ_CODES; }),
          "codes index the codebook");
    const std::vector<int32_t> head = tokenizer.encode(std::vector<float>(audio.begin(), audio.begin() + VQ_CHUNK));
    check(std::equal(head.begin(), head.end(), codes.begin()), "segments encode independently");
    check(tokenizer.log_mel(audio.data(), 16).size() == (size_t) VQ_MELS * (per_token / VQ_HOP),
          "the log-mel covers the padded segment");
    expect_failure([&] { tokenizer.encode_segment(audio.data(), VQ_CHUNK + 1); }, "segment", "an oversized segment");
    int segments = 0;
    const std::vector<int32_t> first = tokenizer.encode(audio, [&] { return segments++ > 0; });
    check(first.size() == VQ_CHUNK / per_token, "a stop request ends encoding between segments");
    std::filesystem::remove(path);
}

void expect_tokenizer_rejects(const std::filesystem::path & path, const std::string & label) {
    expect_failure([&] { SpeechTokenizer tokenizer(path.string(), false, 1); }, "geometry", label);
    std::filesystem::remove(path);
}

void test_tokenizer_rejections() {
    expect_tokenizer_rejects(write_vq("speech-vq-geometry", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-speech-codec.vq.context_length", VQ_CTX - 1);
    }), "a window too short for the chunk");
    expect_tokenizer_rejects(write_vq("speech-vq-padding", WEIGHT_SEED, [](gguf_context * f) {
        gguf_set_val_u32(f, "moss-speech-codec.vq.chunk_samples", VQ_CHUNK + 1);
    }), "a chunk whose padded segment outgrows the window");
}

class FakeTokenizerSidecar final : public SpeechTokenizerSidecar {
public:
    FakeTokenizerSidecar(std::vector<float> states, int fail_on_call, int * calls, std::vector<float> * seen)
        : states_(std::move(states)), fail_on_call_(fail_on_call), calls_(calls), seen_(seen) {}

    bool pooled_states(const std::vector<float> & mel, std::vector<float> & states) override {
        *seen_ = mel;
        if ((*calls_)++ == fail_on_call_) {
            return false;
        }
        states = states_;
        return true;
    }

    const char * label() const override {
        return FAKE_SIDECAR_LABEL;
    }

private:
    std::vector<float> states_;
    int fail_on_call_;
    int * calls_;
    std::vector<float> * seen_;
};

std::vector<float> read_f32_tensor(const std::filesystem::path & path, const char * name) {
    ggml_context * data = nullptr;
    gguf_context * file = gguf_init_from_file(path.string().c_str(), {false, &data});
    const ggml_tensor * tensor = ggml_get_tensor(data, name);
    const float * values = (const float *) tensor->data;
    std::vector<float> out(values, values + ggml_nelements(tensor));
    ggml_free(data);
    gguf_free(file);
    return out;
}

std::vector<int32_t> planted_codes(int tokens) {
    std::vector<int32_t> codes((size_t) tokens);
    for (int t = 0; t < tokens; ++t) {
        codes[(size_t) t] = (t * CODE_STRIDE + CODE_OFFSET) % VQ_CODES;
    }
    return codes;
}

std::vector<float> codebook_rows(const std::vector<float> & codebook, const std::vector<int32_t> & codes) {
    std::vector<float> rows;
    for (int32_t code : codes) {
        rows.insert(rows.end(), codebook.begin() + (std::ptrdiff_t) code * VQ_EMBD,
                    codebook.begin() + (std::ptrdiff_t) (code + 1) * VQ_EMBD);
    }
    return rows;
}

std::unique_ptr<SpeechTokenizerSidecar> planted_sidecar(const std::filesystem::path & path, int fail_on_call,
                                                        int * calls, std::vector<float> * seen) {
    const SpeechTokenizer probe(path.string(), false, 1);
    const SpeechSegmentCapacity capacity = speech_segment_capacity(probe.config());
    const std::vector<float> rows = codebook_rows(read_f32_tensor(path, "whispervq.codebook"),
                                                  planted_codes(capacity.tokens));
    return std::make_unique<FakeTokenizerSidecar>(rows, fail_on_call, calls, seen);
}

void test_tokenizer_capacity() {
    const auto path = write_vq("speech-vq-capacity", WEIGHT_SEED);
    const SpeechTokenizer tokenizer(path.string(), false, 1);
    const SpeechSegmentCapacity capacity = speech_segment_capacity(tokenizer.config());
    check(capacity.tokens == VQ_CHUNK / tokenizer.config().samples_per_token(),
          "the sidecar window holds every token of a full segment");
    check(capacity.mel_frames == VQ_CHUNK / VQ_HOP, "the sidecar window holds every mel frame of a full segment");
    check(open_speech_tokenizer_sidecar(path.string(), tokenizer.config(), {}) == nullptr,
          "no staged sidecar keeps the ggml tokenizer");
    const std::filesystem::path staged = speech_tokenizer_sidecar_path(path.string());
    std::filesystem::create_directories(staged);
    check(open_speech_tokenizer_sidecar(path.string(), tokenizer.config(), {true, false}) == nullptr,
          "MOSS_COREML_DISABLE skips a staged sidecar");
    check(open_speech_tokenizer_sidecar(path.string(), tokenizer.config(), {}) == nullptr,
          "a directory that is not a loadable model keeps the ggml tokenizer");
    std::filesystem::remove_all(staged);
    std::filesystem::remove(path);
}

void test_tokenizer_sidecar_routing() {
    const auto path = write_vq("speech-vq-sidecar", WEIGHT_SEED);
    SpeechTokenizer tokenizer(path.string(), false, 1);
    int calls = 0;
    std::vector<float> seen;
    tokenizer.attach_sidecar(planted_sidecar(path, NEVER_FAIL, &calls, &seen), false);
    tokenizer.begin_run();
    const std::vector<float> audio = random_signal(VQ_CHUNK, AUDIO_SEED);
    const std::vector<int32_t> codes = tokenizer.encode(audio);
    const std::vector<int32_t> expected = planted_codes((int) codes.size());
    check(tokenizer.on_coreml() && calls == 1, "an attached sidecar encodes a full segment");
    check(codes.size() == speech_segment_tokens(tokenizer.config(), audio.size()) && codes == expected,
          "the sidecar's pooled states are quantized to their nearest codes");
    const SpeechSegmentCapacity capacity = speech_segment_capacity(tokenizer.config());
    check(seen.size() == (size_t) VQ_MELS * capacity.mel_frames, "the sidecar sees the whole segment");
    check(tokenizer.run_backend() == FAKE_SIDECAR_LABEL, "the run reports the sidecar label");
    std::filesystem::remove(path);
}

void test_tokenizer_partial_segments_stay_on_ggml() {
    const auto path = write_vq("speech-vq-partial", WEIGHT_SEED);
    SpeechTokenizer reference(path.string(), false, 1);
    SpeechTokenizer tokenizer(path.string(), false, 1);
    int calls = 0;
    std::vector<float> seen;
    tokenizer.attach_sidecar(planted_sidecar(path, NEVER_FAIL, &calls, &seen), true);
    tokenizer.begin_run();
    const std::vector<float> partial = random_signal(VQ_CHUNK / 2, AUDIO_SEED);
    check(tokenizer.encode(partial) == reference.encode(partial) && calls == 0,
          "a partial segment runs on ggml, even under MOSS_COREML_STRICT");
    check(tokenizer.on_coreml() && tokenizer.run_backend() == GGML_STAGE_BACKEND,
          "the sidecar stays attached and the run reports ggml");
    tokenizer.begin_run();
    tokenizer.encode(random_signal(VQ_CHUNK + VQ_CHUNK / 2, AUDIO_SEED));
    check(calls == 1 && tokenizer.run_backend() == MIXED_STAGE_BACKEND,
          "a full segment and a partial one report mixed");
    std::filesystem::remove(path);
}

void test_tokenizer_sidecar_fallback() {
    const auto path = write_vq("speech-vq-fallback", WEIGHT_SEED);
    SpeechTokenizer reference(path.string(), false, 1);
    SpeechTokenizer tokenizer(path.string(), false, 1);
    int calls = 0;
    std::vector<float> seen;
    tokenizer.attach_sidecar(planted_sidecar(path, 1, &calls, &seen), false);
    tokenizer.begin_run();
    const std::vector<float> audio = random_signal(VQ_CHUNK, AUDIO_SEED);
    tokenizer.encode(audio);
    check(tokenizer.encode(audio) == reference.encode(audio), "a failed prediction reruns the segment on ggml");
    check(!tokenizer.on_coreml() && tokenizer.run_backend() == MIXED_STAGE_BACKEND,
          "a failed sidecar is retired and the run reports mixed");
    tokenizer.begin_run();
    tokenizer.encode(audio);
    check(calls == 2 && tokenizer.run_backend() == GGML_STAGE_BACKEND, "a retired sidecar is not tried again");
    std::filesystem::remove(path);
}

void test_tokenizer_sidecar_strict() {
    const auto path = write_vq("speech-vq-strict", WEIGHT_SEED);
    SpeechTokenizer tokenizer(path.string(), false, 1);
    int calls = 0;
    std::vector<float> seen;
    const std::vector<float> audio = random_signal(VQ_CHUNK, AUDIO_SEED);
    tokenizer.attach_sidecar(planted_sidecar(path, 0, &calls, &seen), true);
    expect_failure([&] { tokenizer.encode(audio); }, COREML_STRICT_ENV,
                   "a failed prediction fails the segment under MOSS_COREML_STRICT");
    tokenizer.attach_sidecar(nullptr, true);
    expect_failure([&] { tokenizer.encode(audio); }, COREML_STRICT_ENV,
                   "a missing sidecar fails the segment under MOSS_COREML_STRICT");
    tokenizer.begin_run();
    check(tokenizer.run_backend().empty(), "a run that tokenized nothing reports no backend");
    std::filesystem::remove(path);
}

tts_cpp::moss::SpeechRequest valid_request() {
    tts_cpp::moss::SpeechRequest request;
    request.messages.push_back({tts_cpp::moss::SpeechRole::User, {}, std::vector<float>(1600, 0.1f), 16000});
    return request;
}

void test_request_validation() {
    validate_speech_request(valid_request());
    auto reject = [](auto mutate, const char * needle, const char * label) {
        tts_cpp::moss::SpeechRequest request = valid_request();
        mutate(request);
        expect_failure([&] { validate_speech_request(request); }, needle, label);
    };
    reject([](auto & r) { r.messages.clear(); }, "messages", "no messages");
    reject([](auto & r) { r.messages.back().role = tts_cpp::moss::SpeechRole::Assistant; }, "last message",
           "an assistant turn last");
    reject([](auto & r) { r.messages.back().text = "both"; }, "exactly one", "text and audio together");
    reject([](auto & r) { r.messages.back().sample_rate = 100; }, "sample rate", "an invalid sample rate");
    reject([](auto & r) { r.temperature = 0.0f; }, "temperature", "zero temperature when sampling");
    reject([](auto & r) { r.top_p = 1.5f; }, "top_p", "top_p above one");
    reject([](auto & r) { r.max_new_tokens = -1; }, "max_new_tokens", "negative max_new_tokens");
    reject([](auto & r) { r.max_reply_seconds = NAN; }, "max_reply_seconds", "a non-finite reply limit");
    reject([](auto & r) { r.voice = {0.1f}; r.voice_sample_rate = 0; }, "voice prompt", "a voice without a rate");
    reject([](auto & r) { r.voice.assign(16000 * 61, 0.1f); r.voice_sample_rate = 16000; }, "voice prompt",
           "a voice longer than 60 s");
    auto large_budget = valid_request();
    large_budget.max_new_tokens = 4097;
    validate_speech_request(large_budget);
    check(speech_generation_budget(100, 0, 32768) == 32668,
          "default reply budget uses remaining context instead of stopping at 1000 or 4096 tokens");
    check(speech_generation_budget(30000, 0, 32768) == 2768,
          "default reply budget accounts for conversation history");
    check(speech_generation_budget(100, 4097, 32768) == 4097,
          "an explicitly requested budget above 4096 is allowed when it fits the context");
    check(speech_generation_budget(10, 20, 30) == 20, "an explicit budget fits exactly in remaining context");
    expect_failure([] { speech_generation_budget(11, 20, 30); }, "context", "explicit budget exceeds remaining context");
    expect_failure([] { speech_generation_budget(30, 0, 30); }, "context", "a full context cannot start a reply");
    expect_failure([] { speech_generation_budget(31, 0, 30); }, "context", "an oversized prompt cannot start a reply");
    expect_failure([] { speech_generation_budget(std::numeric_limits<size_t>::max(), 0, 30); }, "context",
                   "an oversized prompt is checked before converting its size");
    expect_failure([] { speech_generation_budget(10, -1, 30); }, "max_new_tokens", "a negative generation budget");
    check_speech_context(10, 20, 30);
    expect_failure([] { check_speech_context(11, 20, 30); }, "context", "a conversation past the model context");
    tts_cpp::moss::SpeechRequest greedy = valid_request();
    greedy.greedy = true;
    greedy.temperature = 0.0f;
    validate_speech_request(greedy);
}

void test_cli_flags() {
    using tts_cpp::moss::cli::CliArgs;
    using tts_cpp::moss::cli::parse_args;
    const char * full[] = {"moss-cli", "--mode", "s2s", "--model", "lm.gguf", "--codec", "codec.gguf", "--audio",
                           "q.wav", "--voice", "v.wav", "--system", "be brief", "--max-reply-seconds", "12",
                           "--max-new-tokens", "300", "--greedy", "--top-k", "5", "--seed", "7", "--text-reply"};
    CliArgs args;
    check(parse_args(23, full, args), "s2s flags parse");
    check(args.s2s_options.model_path == "lm.gguf" && args.s2s_options.codec_path == "codec.gguf" &&
          args.audio_path == "q.wav" && args.voice_path == "v.wav" && args.system_prompt == "be brief" &&
          args.s2s_request.max_reply_seconds == 12.0f && args.s2s_request.max_new_tokens == 300 &&
          args.s2s_request.greedy && args.s2s_request.top_k == 5 && args.s2s_request.seed == 7 &&
          args.s2s_request.text_reply, "s2s values land in the request");
    const char * defaults[] = {"moss-cli", "--mode", "s2s", "--model", "lm.gguf", "--codec", "c.gguf", "--audio",
                               "q.wav"};
    CliArgs plain;
    check(parse_args(9, defaults, plain) && plain.s2s_request.max_new_tokens == 0,
          "without --max-new-tokens the engine default applies");
    const char * no_codec[] = {"moss-cli", "--mode", "s2s", "--model", "lm.gguf", "--audio", "q.wav"};
    CliArgs missing;
    check(!parse_args(7, no_codec, missing), "--codec is required");
    const char * streaming[] = {"moss-cli", "--mode", "s2s", "--model", "lm.gguf", "--codec", "c.gguf", "--audio",
                                "q.wav", "--stream"};
    CliArgs stream;
    check(!parse_args(10, streaming, stream), "--stream is rejected in s2s mode");
}

} // namespace

int main() {
    try {
        test_generation_switches_channels();
        test_generation_constraints();
        test_generation_limits();
        test_generation_heads();
        test_generation_text_minimum();
        test_reply_codes_follow_speech();
        test_prompt_layout();
        test_reply_text();
        test_lm_loads_and_validates();
        test_lm_rejections();
        test_lm_batches_and_steps();
        test_lm_skips_the_text_head();
        test_generation_loop();
        test_segment_arithmetic();
        test_tokenizer_encodes();
        test_tokenizer_rejections();
        test_request_validation();
        test_cli_flags();
        test_tokenizer_capacity();
        test_tokenizer_sidecar_routing();
        test_tokenizer_partial_segments_stay_on_ggml();
        test_tokenizer_sidecar_fallback();
        test_tokenizer_sidecar_strict();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "unexpected: %s\n", e.what());
        return 1;
    }
    if (failures == 0) {
        std::printf("moss speech: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss speech: %d failures\n", failures);
    return 1;
}
