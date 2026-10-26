#pragma once

#include "moss/delay_lm.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss::detail {

struct PromptTokens {
    int32_t pad = 0;
    int32_t im_start = 0;
    int32_t im_end = 0;
};

using TextEncoder = std::function<std::vector<int32_t>(const std::string &)>;

struct PromptAudio {
    std::vector<std::vector<int32_t>> speaker_codes;
    std::vector<int32_t> continuation_codes;

    void prepare_continuation(const std::function<std::vector<int32_t>()> & encode_joined);
};

std::vector<DelayRow> build_prompt_rows(const DelayConfig & config, const PromptTokens & tokens,
                                        const TextEncoder & encode, const std::string & text,
                                        const std::string & language, int duration_tokens,
                                        const PromptAudio & audio);

class Frontend {
public:
    explicit Frontend(const DelayLM & model);
    ~Frontend();
    Frontend(const Frontend &) = delete;
    Frontend & operator=(const Frontend &) = delete;

    const PromptTokens & tokens() const;
    std::vector<int32_t> encode(const std::string & text) const;
    std::vector<DelayRow> build_prompt(const DelayConfig & config, const std::string & text,
                                       const std::string & language, int duration_tokens,
                                       const PromptAudio & audio) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::moss::detail
