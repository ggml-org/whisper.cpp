#include "moss/speech_codec.h"

#include "moss/gguf_metadata.h"

#include "campplus.h"
#include "voice_features.h"

#include "tts-cpp/chatterbox/s3gen_pipeline.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace tts_cpp::moss::detail {
namespace {

constexpr const char * OWNER = "moss speech codec";
constexpr const char * ARCH = "moss-speech-codec";
constexpr const char * PROMPT_MEL_FILTERS = "s3gen/mel_fb/24k_80";
constexpr const char * FBANK_FILTERS = "campplus/mel_fb_kaldi_80";
constexpr const char * DEFAULT_VOICE = "voice.default_audio";
constexpr int OUTPUT_SAMPLE_RATE = 24000;
constexpr int MEL_BANDS = 80;
constexpr float PROMPT_MAGNITUDE_EPS = 1e-9f;
constexpr int CANCELLED = 2;
constexpr int DEFAULT_SEED = 0;
constexpr int MIN_VOICE_RATE = 8000;
constexpr int MAX_VOICE_RATE = 192000;
constexpr double MAX_VOICE_SECONDS = 60.0;
constexpr size_t DECODER_SPEAKER_DIM = 192;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(std::string(OWNER) + ": " + message);
}

struct GgufFile {
    gguf_context * file = nullptr;
    ~GgufFile() { if (file) gguf_free(file); }
};

std::vector<float> read_f32_tensor(const gguf_context * file, const std::string & path, const char * name) {
    const int64_t id = gguf_find_tensor(file, name);
    if (id < 0 || gguf_get_tensor_type(file, id) != GGML_TYPE_F32) {
        fail(std::string("missing f32 tensor: ") + name);
    }
    const size_t offset = gguf_get_data_offset(file) + gguf_get_tensor_offset(file, id);
    const size_t bytes = gguf_get_tensor_size(file, id);
    if (offset + bytes > std::filesystem::file_size(path)) {
        fail(std::string("tensor runs past the end of the file: ") + name);
    }
    std::vector<float> values(bytes / sizeof(float));
    std::ifstream input(path, std::ios::binary);
    input.seekg((std::streamoff) offset);
    if (!input.read(reinterpret_cast<char *>(values.data()), (std::streamsize) (values.size() * sizeof(float)))) {
        fail(std::string("cannot read tensor: ") + name);
    }
    return values;
}

std::vector<float> resampled(const std::vector<float> & pcm, int from, int to) {
    return from == to ? pcm : resample_sinc(pcm, from, to);
}

void add_row(std::vector<double> & sums, const float * row) {
    for (size_t c = 0; c < sums.size(); ++c) {
        sums[c] += row[c];
    }
}

void subtract_row(float * row, const std::vector<double> & sums, double rows) {
    for (size_t c = 0; c < sums.size(); ++c) {
        row[c] -= (float) (sums[c] / rows);
    }
}

std::vector<double> column_sums(const std::vector<float> & frames, size_t rows, size_t columns) {
    std::vector<double> sums(columns, 0.0);
    for (size_t r = 0; r < rows; ++r) {
        add_row(sums, frames.data() + r * columns);
    }
    return sums;
}

void subtract_column_means(std::vector<float> & frames, size_t rows, size_t columns) {
    const std::vector<double> sums = column_sums(frames, rows, columns);
    for (size_t r = 0; r < rows; ++r) {
        subtract_row(frames.data() + r * columns, sums, (double) rows);
    }
}

bool valid_voice(const std::vector<float> & pcm, int rate) {
    return !pcm.empty() && rate >= MIN_VOICE_RATE && rate <= MAX_VOICE_RATE &&
           (double) pcm.size() / rate <= MAX_VOICE_SECONDS;
}

} // namespace

struct SpeechCodec::Impl {
    std::string path;
    bool use_gpu = false;
    int n_threads = 1;
    std::unique_ptr<SpeechTokenizer> tokenizer;
    std::vector<float> prompt_mel_filters;
    std::vector<float> fbank_filters;
    std::vector<float> default_audio;
    int default_rate = 0;
    int token_mel_ratio = 0;
    size_t speaker_dim = 0;
    campplus_weights speaker;
    std::unique_ptr<SpeechVoice> cached_default;

    void read_assets() {
        GgufFile handle;
        handle.file = gguf_init_from_file(path.c_str(), {true, nullptr});
        if (!handle.file) {
            fail("cannot read GGUF: " + path);
        }
        const GgufMetadata meta(handle.file, OWNER);
        if (meta.str("general.architecture") != ARCH) {
            fail("unsupported architecture");
        }
        token_mel_ratio = (int) meta.u32("s3gen.encoder.token_mel_ratio");
        speaker_dim = meta.u32("s3gen.spk_embed_dim");
        default_rate = (int) meta.u32(std::string(ARCH) + ".voice.sample_rate");
        prompt_mel_filters = read_f32_tensor(handle.file, path, PROMPT_MEL_FILTERS);
        fbank_filters = read_f32_tensor(handle.file, path, FBANK_FILTERS);
        default_audio = read_f32_tensor(handle.file, path, DEFAULT_VOICE);
        if (token_mel_ratio < 1 || speaker_dim != DECODER_SPEAKER_DIM || !valid_voice(default_audio, default_rate)) {
            fail("invalid decoder metadata");
        }
    }

    void load() {
        read_assets();
        tokenizer = std::make_unique<SpeechTokenizer>(path, use_gpu, n_threads);
        if (!campplus_load(path, speaker)) {
            fail("the codec GGUF carries no CAM++ speaker encoder");
        }
    }

    std::vector<int32_t> encode(const std::vector<float> & pcm, int rate, const std::function<bool()> & stop = {}) {
        return tokenizer->encode(resampled(pcm, rate, tokenizer->config().sample_rate), stop);
    }

    std::vector<float> prompt_feat(const std::vector<float> & pcm_24k) const {
        std::vector<float> feat = mel_extract_24k_80(pcm_24k, prompt_mel_filters, PROMPT_MAGNITUDE_EPS);
        if (feat.empty()) {
            fail("voice prompt is too short");
        }
        return feat;
    }

    std::vector<float> speaker_embedding(const std::vector<float> & pcm_16k) const {
        std::vector<float> fbank = fbank_kaldi_80(pcm_16k, fbank_filters);
        const size_t frames = fbank.size() / MEL_BANDS;
        if (frames == 0) {
            fail("voice prompt is too short for the speaker encoder");
        }
        subtract_column_means(fbank, frames, MEL_BANDS);
        std::vector<float> embedding;
        if (!campplus_embed(fbank, (int) frames, speaker, nullptr, embedding) || embedding.size() != speaker_dim) {
            fail("speaker encoder failed or does not match the decoder's speaker width");
        }
        return embedding;
    }

    SpeechVoice voice(const std::vector<float> & pcm, int rate) {
        if (!valid_voice(pcm, rate)) {
            fail("voice prompt must hold up to 60 s of audio at 8 to 192 kHz");
        }
        const std::vector<float> pcm_24k = resampled(pcm, rate, OUTPUT_SAMPLE_RATE);
        SpeechVoice result;
        result.feat = prompt_feat(pcm_24k);
        result.tokens = encode(pcm, rate);
        const size_t frames = result.feat.size() / MEL_BANDS;
        const size_t tokens = std::min(frames / (size_t) token_mel_ratio, result.tokens.size());
        result.tokens.resize(tokens);
        result.feat_frames = (int) (tokens * (size_t) token_mel_ratio);
        result.feat.resize((size_t) result.feat_frames * MEL_BANDS);
        result.embedding = speaker_embedding(resampled(pcm_24k, OUTPUT_SAMPLE_RATE, tokenizer->config().sample_rate));
        if (tokens == 0) {
            fail("voice prompt is too short");
        }
        return result;
    }

    const SpeechVoice & default_voice() {
        if (!cached_default) {
            cached_default = std::make_unique<SpeechVoice>(voice(default_audio, default_rate));
        }
        return *cached_default;
    }

    ::s3gen_synthesize_opts synth_options(const SpeechVoice & voice, std::vector<float> & pcm,
                                                             const SpeechDecodeOptions & options) const {
        ::s3gen_synthesize_opts opts;
        opts.s3gen_gguf_path = path;
        opts.pcm_out = &pcm;
        opts.prompt_token_view_data = voice.tokens.data();
        opts.prompt_token_view_size = voice.tokens.size();
        opts.prompt_feat_view_data = voice.feat.data();
        opts.prompt_feat_view_size = voice.feat.size();
        opts.prompt_feat_view_rows = voice.feat_frames;
        opts.embedding_view_data = voice.embedding.data();
        opts.embedding_view_size = voice.embedding.size();
        opts.seed = DEFAULT_SEED;
        opts.n_threads = n_threads;
        opts.sr = OUTPUT_SAMPLE_RATE;
        opts.n_gpu_layers = use_gpu ? 1 : 0;
        opts.finalize = true;
        opts.append_lookahead_silence = false;
        opts.apply_trim_fade = false;
        opts.cfm_z0_override = options.noise;
        opts.cancel_flag = options.cancel;
        opts.dump_mel_path = options.dump_mel_path;
        return opts;
    }

    std::vector<float> decode(const std::vector<int32_t> & codes, const SpeechVoice & voice,
                              const SpeechDecodeOptions & options) {
        if (codes.empty()) {
            return {};
        }
        // Match every acquired cache reference, including exceptions and
        // cancellation. The decoder must not remain resident when the next
        // Speech turn reloads the LM. Other owners' references are preserved.
        struct DecoderLease {
            bool acquired = false;
            ~DecoderLease() { if (acquired) ::s3gen_unload(); }
        } lease;
        if (use_gpu) {
            if (::s3gen_preload(path, 1) != 0) {
                fail("token-to-wave decoder load failed");
            }
            lease.acquired = true;
        }
        std::vector<float> pcm;
        const int status = ::s3gen_synthesize_to_wav(codes, synth_options(voice, pcm, options));
        if (status == CANCELLED) {
            return {};
        }
        if (status != 0) {
            fail("token-to-wave decoding failed");
        }
        return pcm;
    }
};

SpeechCodec::SpeechCodec(const std::string & codec_path, bool use_gpu, int n_threads) : impl_(new Impl) {
    impl_->path = codec_path;
    impl_->use_gpu = use_gpu;
    impl_->n_threads = n_threads;
    impl_->load();
}

SpeechCodec::~SpeechCodec() = default;

int SpeechCodec::sample_rate() const { return OUTPUT_SAMPLE_RATE; }
int SpeechCodec::token_mel_ratio() const { return impl_->token_mel_ratio; }
SpeechTokenizer & SpeechCodec::tokenizer() { return *impl_->tokenizer; }

std::vector<int32_t> SpeechCodec::encode(const std::vector<float> & pcm, int sample_rate,
                                         const std::function<bool()> & stop) {
    return impl_->encode(pcm, sample_rate, stop);
}

std::vector<float> SpeechCodec::prompt_feat(const std::vector<float> & pcm_24k) const {
    return impl_->prompt_feat(pcm_24k);
}

std::vector<float> SpeechCodec::speaker_embedding(const std::vector<float> & pcm_16k) const {
    return impl_->speaker_embedding(pcm_16k);
}

SpeechVoice SpeechCodec::voice(const std::vector<float> & pcm, int sample_rate) {
    return impl_->voice(pcm, sample_rate);
}

const SpeechVoice & SpeechCodec::default_voice() { return impl_->default_voice(); }

std::vector<float> SpeechCodec::decode(const std::vector<int32_t> & codes, const SpeechVoice & voice,
                                       const SpeechDecodeOptions & options) {
    return impl_->decode(codes, voice, options);
}

} // namespace tts_cpp::moss::detail
