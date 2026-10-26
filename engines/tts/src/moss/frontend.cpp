#include "moss/frontend.h"

#include "moss/generation.h"
#include "qwen_tokenizer.h"

#include <stdexcept>

namespace tts_cpp::moss::detail {
namespace {

constexpr const char * PAD_TOKEN      = "<|endoftext|>";
constexpr const char * IM_START_TOKEN = "<|im_start|>";
constexpr const char * IM_END_TOKEN   = "<|im_end|>";
constexpr const char * AUDIO_PLACEHOLDER = "<|audio|>";
constexpr int TTSD_AUDIO_CHANNELS = 16;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss frontend: " + message);
}

std::string tokens_field(int duration_tokens) {
    return duration_tokens > 0 ? std::to_string(duration_tokens) : "None";
}

std::string user_instruction(const std::string & reference, const std::string & language,
                             const std::string & text, int duration_tokens, int audio_channels) {
    const std::string scene = audio_channels == TTSD_AUDIO_CHANNELS ? "- Scene:\nNone\n" : "";
    return "<user_inst>\n"
           "- Reference(s):\n" + reference + "\n"
           "- Instruction:\nNone\n"
           "- Tokens:\n" + tokens_field(duration_tokens) + "\n"
           "- Quality:\nNone\n"
           "- Sound Event:\nNone\n"
           "- Ambient Sound:\nNone\n"
           "- Language:\n" + language + "\n" + scene +
           "- Text:\n" + text + "\n"
           "</user_inst>";
}

} // namespace

int frames_of(const DelayConfig & config, const std::vector<int32_t> & codes) {
    if (codes.size() % (size_t) config.n_vq != 0) {
        fail("audio codes are not a whole number of frames");
    }
    return (int) (codes.size() / (size_t) config.n_vq);
}

std::string reference_field_for(size_t speakers) {
    if (speakers == 0) {
        return "None";
    }
    std::string field;
    for (size_t s = 0; s < speakers; ++s) {
        if (s > 0) {
            field += "\n";
        }
        field += "[S" + std::to_string(s + 1) + "]:\n" + AUDIO_PLACEHOLDER;
    }
    return field;
}

struct RowBuilder {
    const DelayConfig & config;
    const TextEncoder & encode;
    std::vector<DelayRow> rows;

    void append_text(const std::vector<int32_t> & ids) {
        for (int32_t id : ids) {
            DelayRow row;
            row.text = id;
            row.audio.assign((size_t) config.n_vq, config.audio_pad_code);
            rows.push_back(row);
        }
    }

    void append_encoded(const std::string & span) {
        append_text(encode(span));
    }

    void append_delayed(int32_t slot_token, const std::vector<int32_t> & delayed, int64_t count) {
        for (int64_t r = 0; r < count; ++r) {
            DelayRow row;
            row.text = slot_token;
            row.audio.assign((size_t) config.n_vq, config.audio_pad_code);
            for (int channel = 0; channel < config.n_vq; ++channel) {
                row.audio[(size_t) channel] = delayed[(size_t) r * config.n_vq + channel];
            }
            rows.push_back(row);
        }
    }

    void append_user_audio_block(const std::vector<int32_t> & codes) {
        const int frames = frames_of(config, codes);
        append_text({(int32_t) config.audio_start_token_id});
        const std::vector<int32_t> delayed = apply_delay_pattern(codes, frames,
                config.n_vq, config.audio_pad_code);
        append_delayed((int32_t) config.audio_user_slot_token_id, delayed,
                (int64_t) frames + config.n_vq - 1);
        append_text({(int32_t) config.audio_end_token_id});
    }

    void append_continuation_block(const std::vector<int32_t> & codes) {
        const int frames = frames_of(config, codes);
        append_text({(int32_t) config.audio_start_token_id});
        const std::vector<int32_t> delayed = apply_delay_pattern(codes, frames,
                config.n_vq, config.audio_pad_code);
        append_delayed((int32_t) config.audio_assistant_gen_slot_token_id, delayed, frames);
    }

    void append_user_content(const std::string & content,
                             const std::vector<std::vector<int32_t>> & speaker_codes) {
        size_t cursor = 0;
        for (const std::vector<int32_t> & codes : speaker_codes) {
            const size_t split = content.find(AUDIO_PLACEHOLDER, cursor);
            if (split == std::string::npos) {
                fail("reference placeholders do not match the speaker count");
            }
            append_encoded(content.substr(cursor, split - cursor));
            append_user_audio_block(codes);
            cursor = split + std::string(AUDIO_PLACEHOLDER).size();
        }
        append_encoded(content.substr(cursor));
    }
};

std::vector<DelayRow> build_prompt_rows(const DelayConfig & config, const PromptTokens & tokens,
                                        const TextEncoder & encode, const std::string & text,
                                        const std::string & language, int duration_tokens,
                                        const PromptAudio & audio) {
    const std::string reference_field = reference_field_for(audio.speaker_codes.size());
    const std::string content = user_instruction(reference_field, language, text, duration_tokens, config.n_vq);

    RowBuilder builder{config, encode, {}};
    builder.append_text({tokens.im_start});
    builder.append_encoded("user\n");
    builder.append_user_content(content, audio.speaker_codes);
    builder.append_text({tokens.im_end});
    builder.append_encoded("\n");
    builder.append_text({tokens.im_start});
    builder.append_encoded("assistant\n");

    if (audio.continuation_codes.empty()) {
        DelayRow seed;
        seed.text = config.audio_start_token_id;
        seed.audio.assign((size_t) config.n_vq, config.audio_pad_code);
        builder.rows.push_back(seed);
    } else {
        builder.append_continuation_block(audio.continuation_codes);
    }
    return builder.rows;
}

struct Frontend::Impl {
    QwenTokenizer tokenizer;
    PromptTokens tokens;

    void load(const DelayLM & model) {
        const std::vector<std::string> vocab = model.tokenizer_tokens();
        const std::vector<std::string> merges = model.tokenizer_merges();
        if (vocab.empty() || merges.empty()) {
            fail("backbone GGUF is missing the tokenizer vocabulary");
        }
        tokenizer.build_byte_map();
        tokenizer.vocab.reserve(vocab.size());
        for (size_t id = 0; id < vocab.size(); ++id) {
            tokenizer.vocab.emplace(vocab[id], (int) id);
        }
        tokenizer.merge_rank.reserve(merges.size());
        for (size_t rank = 0; rank < merges.size(); ++rank) {
            tokenizer.merge_rank.emplace(merges[rank], (int) rank);
        }
        tokens.pad      = lookup(PAD_TOKEN);
        tokens.im_start = lookup(IM_START_TOKEN);
        tokens.im_end   = lookup(IM_END_TOKEN);
        const int32_t text_vocab = model.config().text_vocab;
        if (tokens.pad >= text_vocab || tokens.im_start >= text_vocab ||
            tokens.im_end >= text_vocab) {
            fail("special tokens fall outside the model's text vocabulary");
        }
    }

    int32_t lookup(const std::string & token) const {
        const auto it = tokenizer.vocab.find(token);
        if (it == tokenizer.vocab.end()) {
            fail("tokenizer is missing the special token " + token);
        }
        return it->second;
    }

    std::vector<int32_t> encode(const std::string & text) {
        std::vector<int32_t> ids;
        for (int id : tokenizer.encode(text)) {
            ids.push_back((int32_t) id);
        }
        return ids;
    }
};

Frontend::Frontend(const DelayLM & model) : impl_(new Impl) {
    impl_->load(model);
}

Frontend::~Frontend() = default;

const PromptTokens & Frontend::tokens() const { return impl_->tokens; }

std::vector<int32_t> Frontend::encode(const std::string & text) const {
    return impl_->encode(text);
}

std::vector<DelayRow> Frontend::build_prompt(const DelayConfig & config, const std::string & text,
                                             const std::string & language, int duration_tokens,
                                             const PromptAudio & audio) const {
    return build_prompt_rows(config, impl_->tokens,
            [this](const std::string & span) { return impl_->encode(span); },
            text, language, duration_tokens, audio);
}

} // namespace tts_cpp::moss::detail
