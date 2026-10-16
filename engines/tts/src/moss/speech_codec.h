#pragma once

#include "moss/speech_tokenizer.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss::detail {

struct SpeechVoice {
    std::vector<int32_t> tokens;
    std::vector<float> feat;
    int feat_frames = 0;
    std::vector<float> embedding;
};

struct SpeechDecodeOptions {
    std::vector<float> noise;
    const std::atomic<bool> * cancel = nullptr;
    std::string dump_mel_path;
    bool release_decoder = false;
};

bool decoder_uses_gpu(const BackendRequest & backend);

class SpeechCodec {
public:
    SpeechCodec(const std::string & codec_path, const BackendRequest & backend, int n_threads);
    ~SpeechCodec();
    SpeechCodec(const SpeechCodec &) = delete;
    SpeechCodec & operator=(const SpeechCodec &) = delete;

    int sample_rate() const;
    int token_mel_ratio() const;
    SpeechTokenizer & tokenizer();
    std::vector<int32_t> encode(const std::vector<float> & pcm, int sample_rate,
                                const std::function<bool()> & stop = {});
    std::vector<float> prompt_feat(const std::vector<float> & pcm_24k) const;
    std::vector<float> speaker_embedding(const std::vector<float> & pcm_16k) const;
    SpeechVoice voice(const std::vector<float> & pcm, int sample_rate);
    const SpeechVoice & default_voice();
    std::vector<float> decode(const std::vector<int32_t> & codes, const SpeechVoice & voice,
                              const SpeechDecodeOptions & options = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::moss::detail
