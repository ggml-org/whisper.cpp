#include "moss/speech_generation.h"
#include "moss/speech_lm.h"
#include "moss/speech_prompt.h"
#include "moss/speech_codec.h"
#include "moss/speech_tokenizer.h"
#include "chatterbox_tts_test_hooks.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tts_cpp::moss::detail;

namespace {

constexpr int SKIP = 77;
constexpr int PARITY_THREADS = 8;
constexpr int PREFILL_BATCH = 64;
constexpr double LOGIT_COSINE = 0.99;
constexpr double MEL_COSINE = 0.9999;
constexpr double TOKEN_AGREEMENT = 0.95;
constexpr double STAGE_COSINE = 0.999;
constexpr double SPECTRUM_COSINE = 0.99;
constexpr double TEACHER_AGREEMENT = 0.9;
constexpr int GREEDY_TOKENS = 600;
constexpr int MIN_NEW_TOKENS = 10;

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

template <typename T>
std::vector<T> read_bin(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot read " + path.string());
    }
    const std::streamsize bytes = input.tellg();
    input.seekg(0);
    std::vector<T> data((size_t) bytes / sizeof(T));
    input.read(reinterpret_cast<char *>(data.data()), bytes);
    return data;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return -2.0;
    }
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

void report(const char * stage, double value, double threshold) {
    std::printf("%-24s %.6f (threshold %.3f)\n", stage, value, threshold);
    check(value >= threshold, std::string(stage) + " diverges from the PyTorch reference");
}

std::vector<SpeechRow> rows_of(const std::vector<int32_t> & grid) {
    std::vector<SpeechRow> rows;
    for (size_t i = 0; i + 1 < grid.size(); i += 2) {
        rows.push_back({grid[i], grid[i + 1]});
    }
    return rows;
}

double agreement(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    size_t same = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        same += a[i] == b[i] ? 1 : 0;
    }
    return (double) same / (double) std::max<size_t>(std::max(a.size(), b.size()), 1);
}

const char * const REFERENCE_SYSTEM_PROMPT =
        "You are a helpful voice assistant. Answer the user's questions with spoken responses.";

std::vector<SpeechRow> reference_prompt(const SpeechLM & lm, const SpeechTextTokenizer & tokenizer,
                                        const std::filesystem::path & codec_dir) {
    SpeechTurn system{SpeechTurnRole::System, REFERENCE_SYSTEM_PROMPT, {}, false};
    SpeechTurn user{SpeechTurnRole::User, {}, read_bin<int32_t>(codec_dir / "user_tokens.bin"), true};
    return speech_prompt(lm.config(), tokenizer, {system, user}, true);
}

bool same_rows(const std::vector<SpeechRow> & a, const std::vector<SpeechRow> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].text != b[i].text || a[i].audio != b[i].audio) {
            return false;
        }
    }
    return true;
}

struct GreedyReply {
    std::vector<int32_t> codes;
    bool finished = false;
};

GreedyReply greedy_reply(SpeechLM & lm, const std::vector<SpeechRow> & prompt, int max_new_tokens) {
    SpeechLimits limits{max_new_tokens, MIN_NEW_TOKENS, 0};
    lm.begin((int) prompt.size() + max_new_tokens);
    SpeechLogits logits = lm.prefill(prompt, PREFILL_BATCH);
    SpeechGenerationState state(lm.config().tokens, prompt.back(), limits);
    std::mt19937 rng(0);
    SpeechSampling greedy{false, 1.0f, 1.0f, 0};
    while (!state.stopping()) {
        const SpeechRow row = state.next(std::move(logits), greedy, rng);
        if (!state.stopping()) {
            logits = lm.step(row);
        }
    }
    return {speech_reply_codes(state.generated(), lm.config().tokens), !state.truncated()};
}

int32_t constrained_argmax(std::vector<float> audio, const SpeechTokens & tokens) {
    audio.resize((size_t) tokens.speech_end + 1);
    return (int32_t) std::distance(audio.begin(), std::max_element(audio.begin(), audio.end()));
}

size_t teacher_forced_matches(SpeechLM & lm, SpeechLogits logits, const std::vector<SpeechRow> & reference) {
    size_t matches = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        matches += constrained_argmax(logits.audio, lm.config().tokens) == reference[i].audio ? 1 : 0;
        if (i + 1 < reference.size()) {
            logits = lm.step(reference[i]);
        }
    }
    return matches;
}

void test_teacher_forcing(SpeechLM & lm, const std::vector<SpeechRow> & prompt, const std::filesystem::path & dir) {
    std::vector<SpeechRow> reference = rows_of(read_bin<int32_t>(dir / "lm" / "generated_ids.bin"));
    reference.resize(read_bin<int32_t>(dir / "codec" / "reply_codes.bin").size());
    lm.begin((int) (prompt.size() + reference.size()) + 1);
    const SpeechLogits logits = lm.prefill(prompt, PREFILL_BATCH);
    const size_t matches = teacher_forced_matches(lm, logits, reference);
    report("teacher-forced codes", (double) matches / (double) reference.size(), TEACHER_AGREEMENT);
}

void test_greedy(SpeechLM & lm, const std::filesystem::path & dir) {
    const SpeechTextTokenizer tokenizer(lm.tokenizer_tokens(), lm.tokenizer_merges(), lm.tokenizer_types());
    const std::vector<SpeechRow> prompt = reference_prompt(lm, tokenizer, dir / "codec");
    check(same_rows(prompt, rows_of(read_bin<int32_t>(dir / "lm" / "input_ids.bin"))),
          "prompt grid reproduces the Hugging Face processor");
    const std::vector<int32_t> reference = read_bin<int32_t>(dir / "codec" / "reply_codes.bin");
    const GreedyReply reply = greedy_reply(lm, prompt, GREEDY_TOKENS);
    std::printf("greedy reply: %zu codes (reference %zu), %s\n", reply.codes.size(), reference.size(),
                reply.finished ? "ended by the model" : "cut at the token limit");
    std::printf("greedy reply codes agreement %.3f\n", agreement(reply.codes, reference));
    check(!reply.codes.empty(), "greedy decoding produces speech");
    test_teacher_forcing(lm, prompt, dir);
}

void test_prefill(SpeechLM & lm, const std::filesystem::path & dir) {
    const std::vector<SpeechRow> rows = rows_of(read_bin<int32_t>(dir / "input_ids.bin"));
    lm.begin((int) rows.size() + 1);
    const SpeechLogits logits = lm.prefill(rows, PREFILL_BATCH);
    report("prefill text logits", cosine(logits.text, read_bin<float>(dir / "prefill_text_logits.bin")), LOGIT_COSINE);
    report("prefill audio logits", cosine(logits.audio, read_bin<float>(dir / "prefill_audio_logits.bin")),
           LOGIT_COSINE);
}

void test_tokenizer(const std::string & codec_path, const BackendRequest & backend,
                    const std::filesystem::path & codec_dir) {
    SpeechTokenizer tokenizer(codec_path, backend, PARITY_THREADS);
    std::printf("tokenizer backend: %s\n", tokenizer.backend_name());
    const std::vector<float> audio = read_bin<float>(codec_dir / "user_audio_16k.bin");
    const std::vector<float> mel = tokenizer.log_mel(audio.data(), audio.size());
    report("user log-mel", cosine(mel, read_bin<float>(codec_dir / "user_mel.bin")), MEL_COSINE);
    const std::vector<int32_t> codes = tokenizer.encode(audio);
    const std::vector<int32_t> reference = read_bin<int32_t>(codec_dir / "user_tokens.bin");
    std::printf("user tokens: %zu (reference %zu)\n", codes.size(), reference.size());
    report("user speech tokens", agreement(codes, reference), TOKEN_AGREEMENT);
}

void test_decoder(const std::string & codec_path, const BackendRequest & backend, const std::filesystem::path & dir) {
    SpeechCodec codec(codec_path, backend, PARITY_THREADS);
    const std::vector<float> feat = codec.prompt_feat(read_bin<float>(dir / "voice_audio_24k.bin"));
    const std::vector<float> reference_feat = read_bin<float>(dir / "voice_prompt_feat.bin");
    const std::vector<float> head(feat.begin(), feat.begin() + (std::ptrdiff_t) std::min(feat.size(), reference_feat.size()));
    report("voice prompt mel", cosine(head, reference_feat), STAGE_COSINE);
    report("speaker embedding", cosine(codec.speaker_embedding(read_bin<float>(dir / "voice_audio_16k.bin")),
           read_bin<float>(dir / "voice_embedding.bin")), STAGE_COSINE);
    SpeechVoice voice;
    voice.tokens = read_bin<int32_t>(dir / "voice_prompt_tokens.bin");
    voice.feat = reference_feat;
    voice.feat_frames = (int) (reference_feat.size() / 80);
    voice.embedding = read_bin<float>(dir / "voice_embedding.bin");
    SpeechDecodeOptions options;
    options.noise = read_bin<float>(dir / "flow_noise.bin");
    options.release_decoder = true;
    if (const char * dump = std::getenv("MOSS_SPEECH_DUMP_MEL")) {
        options.dump_mel_path = dump;
    }
    const std::vector<float> wav = codec.decode(read_bin<int32_t>(dir / "reply_codes.bin"), voice, options);
    const std::vector<float> reference_wav = read_bin<float>(dir / "reply_audio_24k.bin");
    std::printf("reply samples: %zu (reference %zu)\n", wav.size(), reference_wav.size());
    check(!tts_cpp::chatterbox::test_hooks::cfm_estimator_cache_built() &&
          !tts_cpp::chatterbox::test_hooks::encoder_graph_cache_built(),
          "a released decoder keeps no S3Gen graph cache");
    report("reply log-mel", cosine(codec.prompt_feat(wav), codec.prompt_feat(reference_wav)), SPECTRUM_COSINE);
}

BackendRequest parity_backend() {
    const char * name = std::getenv("MOSS_SPEECH_BACKEND");
    return {name ? name : "", std::getenv("MOSS_SPEECH_GPU") != nullptr};
}

} // namespace

int main() {
    const char * model_path = std::getenv("MOSS_SPEECH_MODEL");
    const char * reference_dir = std::getenv("MOSS_SPEECH_REFERENCE_DIR");
    if (!model_path || !reference_dir) {
        std::printf("SKIP: set MOSS_SPEECH_MODEL and MOSS_SPEECH_REFERENCE_DIR\n");
        return SKIP;
    }
    try {
        const BackendRequest backend = parity_backend();
        const std::filesystem::path dir(reference_dir);
        if (const char * codec_path = std::getenv("MOSS_SPEECH_CODEC")) {
            test_tokenizer(codec_path, backend, dir / "codec");
            test_decoder(codec_path, backend, dir / "codec");
        }
        SpeechLM lm(model_path, backend, PARITY_THREADS);
        std::printf("backend: %s\n", lm.backend_name());
        test_prefill(lm, dir / "lm");
        test_greedy(lm, dir);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    if (failures == 0) {
        std::printf("moss speech parity: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss speech parity: %d failures\n", failures);
    return 1;
}
