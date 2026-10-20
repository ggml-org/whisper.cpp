#include "moss/transcribe_coreml.h"

#include "moss/transcribe_networks.h"
#include "parakeet_coreml_path.h"

#ifdef PARAKEET_USE_COREML
#include "coreml/parakeet-encoder.h"
#endif

#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace parakeet::moss::detail {
namespace {

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss transcribe: " + message);
}

bool env_set(const char * name) {
    return std::getenv(name) != nullptr;
}

int chunk_token_capacity(const TranscribeConfig & config) {
    return config.encoder.n_ctx / config.merge_size;
}

void scatter_band(const std::vector<float> & mel, int band, int n_mels, int frames, std::vector<float> & out) {
    const size_t row = (size_t) band * frames;
    for (int frame = 0; frame < frames; ++frame) {
        out[(size_t) frame * n_mels + band] = mel[row + frame];
    }
}

#ifdef PARAKEET_USE_COREML
class CoremlTranscribeEncoder final : public TranscribeEncoderSidecar {
public:
    CoremlTranscribeEncoder(parakeet_coreml_context * context, const TranscribeConfig & config)
        : context_(context), n_mels_(config.audio.n_mels), frames_(config.audio.chunk_frames),
          tokens_(chunk_token_capacity(config)), width_(config.text.n_embd) {}

    ~CoremlTranscribeEncoder() override {
        parakeet_coreml_free(context_);
    }

    bool encode(const std::vector<float> & mel, std::vector<float> & embeddings) override {
        const std::vector<float> frames = frame_major_mel(mel, n_mels_, frames_);
        embeddings.assign((size_t) tokens_ * width_, 0.0f);
        return parakeet_coreml_encode(context_, frames_, /*n_mel_valid=*/frames_, n_mels_,
                frames.data(), tokens_, width_, embeddings.data()) == 0;
    }

    const char * label() const override {
        return parakeet_coreml_backend_label(context_);
    }

private:
    parakeet_coreml_context * context_;
    int n_mels_;
    int frames_;
    int tokens_;
    int width_;
};

bool matches_window(parakeet_coreml_context * context, const TranscribeConfig & config) {
    return parakeet_coreml_fixed_mel_frames(context, config.audio.n_mels) == config.audio.chunk_frames &&
           parakeet_coreml_fixed_output_rows(context, config.text.n_embd) == chunk_token_capacity(config);
}

std::unique_ptr<TranscribeEncoderSidecar> load_coreml_encoder(const std::string & path,
                                                              const TranscribeConfig & config) {
    parakeet_coreml_context * context = parakeet_coreml_init_with_units_env(path.c_str(), COREML_UNITS_ENV);
    if (context == nullptr) {
        return nullptr;
    }
    if (!matches_window(context, config)) {
        parakeet_coreml_free(context);
        return nullptr;
    }
    return std::make_unique<CoremlTranscribeEncoder>(context, config);
}
#else
std::unique_ptr<TranscribeEncoderSidecar> load_coreml_encoder(const std::string &, const TranscribeConfig &) {
    return nullptr;
}
#endif

} // namespace

CoremlPolicy read_coreml_policy() {
    CoremlPolicy policy;
    policy.disabled = env_set(COREML_DISABLE_ENV);
    policy.strict = env_set(COREML_STRICT_ENV);
    return policy;
}

std::string transcribe_encoder_sidecar_path(const std::string & model_path) {
    return ::parakeet::coreml_encoder_sidecar_path(model_path);
}

std::vector<float> frame_major_mel(const std::vector<float> & mel, int n_mels, int frames) {
    if (mel.size() != (size_t) n_mels * frames) {
        fail("mel window has the wrong size for the Core ML encoder");
    }
    std::vector<float> out(mel.size());
    for (int band = 0; band < n_mels; ++band) {
        scatter_band(mel, band, n_mels, frames, out);
    }
    return out;
}

std::unique_ptr<TranscribeEncoderSidecar> open_transcribe_encoder_sidecar(const std::string & model_path,
                                                                          const TranscribeConfig & config,
                                                                          const CoremlPolicy & policy) {
    const std::string path = transcribe_encoder_sidecar_path(model_path);
    if (policy.disabled || !std::filesystem::is_directory(path)) {
        return nullptr;
    }
    return load_coreml_encoder(path, config);
}

TranscribeAudioEncoder::TranscribeAudioEncoder(TranscribeModel & model,
                                               std::unique_ptr<TranscribeEncoderSidecar> sidecar, bool strict)
    : model_(model), sidecar_(std::move(sidecar)), strict_(strict) {}

void TranscribeAudioEncoder::begin_run() {
    run_backends_.clear();
}

std::vector<float> TranscribeAudioEncoder::encode(const std::vector<float> & mel, int tokens) {
    std::vector<float> embeddings;
    if (try_sidecar(mel, tokens, embeddings)) {
        return embeddings;
    }
    if (strict_) {
        fail(std::string("no working Core ML encoder sidecar and ") + COREML_STRICT_ENV + " is set");
    }
    run_backends_.insert(GGML_ENCODER_BACKEND);
    return encode_audio_chunk(model_, mel, tokens, false).embeddings;
}

bool TranscribeAudioEncoder::on_coreml() const {
    return sidecar_ != nullptr;
}

std::string TranscribeAudioEncoder::run_backend() const {
    if (run_backends_.empty()) {
        return std::string();
    }
    return run_backends_.size() == 1 ? *run_backends_.begin() : MIXED_ENCODER_BACKEND;
}

bool TranscribeAudioEncoder::try_sidecar(const std::vector<float> & mel, int tokens,
                                         std::vector<float> & embeddings) {
    if (sidecar_ == nullptr) {
        return false;
    }
    std::vector<float> rows;
    if (!sidecar_->encode(mel, rows)) {
        sidecar_.reset();
        return false;
    }
    run_backends_.insert(sidecar_->label());
    embeddings = leading_rows(rows, tokens);
    return true;
}

std::vector<float> TranscribeAudioEncoder::leading_rows(const std::vector<float> & rows, int tokens) const {
    const size_t count = (size_t) tokens * model_.config().text.n_embd;
    if (tokens < 1 || count > rows.size()) {
        fail("chunk token count is outside the Core ML encoder window");
    }
    return std::vector<float>(rows.begin(), rows.begin() + (std::ptrdiff_t) count);
}

} // namespace parakeet::moss::detail
