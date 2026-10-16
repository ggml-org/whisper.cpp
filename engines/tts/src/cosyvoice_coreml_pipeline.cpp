#include "cosyvoice_coreml.h"
#include "cosyvoice_pipeline.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <stdexcept>

namespace {
constexpr int mel_channels = 80;
constexpr int time_channels = 256;
constexpr int cfg_batch = 2;
constexpr int euler_steps = 10;
constexpr float guidance = 0.7f;
constexpr float masked_score = -1e4f;
constexpr float rope_theta = 10000.0f;
constexpr int spectrum_channels = 18;
constexpr int stft_frames_per_mel = 120;
constexpr double pi = 3.14159265358979323846;

int frame_capacity(const std::shared_ptr<cosyvoice_coreml> & model, const char * feature, int axis,
                    int rank, int requested) {
    if (!model || requested < 1) return 0;
    const auto shape = model->dimensions(feature);
    if (shape.size() != static_cast<size_t>(rank) || shape[axis] < requested || shape[axis] > INT_MAX) return 0;
    return static_cast<int>(shape[axis]);
}

std::vector<float> pad_channels(const std::vector<float> & source, int channels, int frames, int capacity) {
    if (source.size() != static_cast<size_t>(channels) * frames)
        throw std::invalid_argument("cosyvoice: invalid Core ML channel input");
    std::vector<float> padded(static_cast<size_t>(channels) * capacity);
    for (int channel = 0; channel < channels; ++channel)
        std::copy_n(source.data() + static_cast<size_t>(channel) * frames, frames,
                    padded.data() + static_cast<size_t>(channel) * capacity);
    return padded;
}

std::vector<float> crop_channels(const std::vector<float> & source, int channels, int frames, int capacity) {
    std::vector<float> cropped(static_cast<size_t>(channels) * frames);
    for (int channel = 0; channel < channels; ++channel)
        std::copy_n(source.data() + static_cast<size_t>(channel) * capacity, frames,
                    cropped.data() + static_cast<size_t>(channel) * frames);
    return cropped;
}

void fill_rotary(std::vector<float> & cosine, std::vector<float> & sine, int frames, int head_dim) {
    for (int frame = 0; frame < frames; ++frame) {
        for (int pair = 0; pair < head_dim / 2; ++pair) {
            const auto index = static_cast<size_t>(frame) * (head_dim / 2) + pair;
            const float angle = frame * std::pow(rope_theta, -2.0f * pair / head_dim);
            cosine[index] = std::cos(angle);
            sine[index] = std::sin(angle);
        }
    }
}

bool flow_interface(const cosyvoice_coreml & model, int capacity, int head_dim) {
    for (const char * name : {"x", "mu", "cond", "velocity"})
        if (!model.matches(name, {cfg_batch, capacity, mel_channels})) return false;
    return model.matches("spks", {cfg_batch, mel_channels}) &&
           model.matches("time_sin", {cfg_batch, time_channels}) &&
           model.matches("cosine", {1, capacity, head_dim / 2}) &&
           model.matches("sine", {1, capacity, head_dim / 2}) &&
           model.matches("mask", {1, 1, 1, capacity});
}

void update_euler(std::vector<float> & x, const std::vector<float> & velocity, size_t batch_stride, float dt) {
    for (size_t i = 0; i < x.size(); ++i)
        x[i] += dt * ((1 + guidance) * velocity[i] - guidance * velocity[i + batch_stride]);
}
}

bool cosyvoice_coreml_frontend(const std::shared_ptr<cosyvoice_coreml> & model,
                               const std::vector<int32_t> & tokens, const std::vector<float> & speaker,
                               std::vector<float> & mu, std::vector<float> & spks) {
    const int capacity = frame_capacity(model, "token_ids", 1, 2, static_cast<int>(tokens.size()));
    if (!capacity || !model->matches("token_ids", {1, capacity}) ||
        !model->matches("token_mask", {1, capacity, 1}) ||
        !model->matches("speaker", {1, static_cast<int64_t>(speaker.size())}) ||
        !model->matches("mu", {1, int64_t(capacity) * 2, mel_channels}) ||
        !model->matches("spks", {1, mel_channels})) return false;
    std::vector<float> ids(capacity), mask(capacity), features(static_cast<size_t>(capacity) * 2 * mel_channels);
    std::copy(tokens.begin(), tokens.end(), ids.begin());
    std::fill_n(mask.begin(), tokens.size(), 1);
    spks.resize(mel_channels);
    model->predict({{"token_ids", ids.data(), {1, capacity}}, {"token_mask", mask.data(), {1, capacity, 1}},
                    {"speaker", speaker.data(), {1, static_cast<int64_t>(speaker.size())}}},
                   {{"mu", features.data(), {1, int64_t(capacity) * 2, mel_channels}},
                    {"spks", spks.data(), {1, mel_channels}}});
    features.resize(tokens.size() * 2 * mel_channels);
    mu = std::move(features);
    return true;
}

bool cosyvoice_coreml_euler(const std::shared_ptr<cosyvoice_coreml> & model, int head_dim,
                           const std::vector<float> & mu, const std::vector<float> & cond,
                           const std::vector<float> & spks, int frames, std::vector<float> & x) {
    const int capacity = frame_capacity(model, "x", 1, 3, frames);
    if (!capacity || !flow_interface(*model, capacity, head_dim)) return false;
    const size_t actual = static_cast<size_t>(frames) * mel_channels;
    if (mu.size() != actual || cond.size() != actual || x.size() != actual || spks.size() != mel_channels)
        throw std::invalid_argument("cosyvoice: invalid Core ML flow inputs");
    const size_t stride = static_cast<size_t>(capacity) * mel_channels;
    std::vector<float> mu_in(stride * cfg_batch), cond_in(stride * cfg_batch), spks_in(mel_channels * cfg_batch);
    std::copy(mu.begin(), mu.end(), mu_in.begin());
    std::copy(cond.begin(), cond.end(), cond_in.begin());
    std::copy(spks.begin(), spks.end(), spks_in.begin());
    std::vector<float> cosine(static_cast<size_t>(capacity) * (head_dim / 2)), sine(cosine.size());
    fill_rotary(cosine, sine, capacity, head_dim);
    std::vector<float> mask(capacity, masked_score), xin(stride * cfg_batch), velocity(xin.size());
    std::fill_n(mask.begin(), frames, 0);
    for (int step = 0; step < euler_steps; ++step) {
        const float time = 1 - std::cos(static_cast<float>(step) / euler_steps * 0.5f * static_cast<float>(pi));
        const float next = 1 - std::cos(static_cast<float>(step + 1) / euler_steps * 0.5f * static_cast<float>(pi));
        const auto time_sin = sinus_time_emb({time, time}, time_channels);
        std::copy(x.begin(), x.end(), xin.begin());
        std::copy(x.begin(), x.end(), xin.begin() + stride);
        model->predict({{"x", xin.data(), {cfg_batch, capacity, mel_channels}},
                        {"mu", mu_in.data(), {cfg_batch, capacity, mel_channels}},
                        {"cond", cond_in.data(), {cfg_batch, capacity, mel_channels}},
                        {"spks", spks_in.data(), {cfg_batch, mel_channels}},
                        {"time_sin", time_sin.data(), {cfg_batch, time_channels}},
                        {"cosine", cosine.data(), {1, capacity, head_dim / 2}},
                        {"sine", sine.data(), {1, capacity, head_dim / 2}},
                        {"mask", mask.data(), {1, 1, 1, capacity}}},
                       {{"velocity", velocity.data(), {cfg_batch, capacity, mel_channels}}});
        update_euler(x, velocity, stride, next - time);
    }
    return true;
}

bool cosyvoice_coreml_f0(const std::shared_ptr<cosyvoice_coreml> & model,
                        const std::vector<float> & mel, int frames, std::vector<float> & f0) {
    const int capacity = frame_capacity(model, "mel", 2, 3, frames);
    if (!capacity || !model->matches("mel", {1, mel_channels, capacity}) ||
        !model->matches("f0", {1, capacity})) return false;
    const auto padded = pad_channels(mel, mel_channels, frames, capacity);
    std::vector<float> output(capacity);
    model->predict({{"mel", padded.data(), {1, mel_channels, capacity}}},
                   {{"f0", output.data(), {1, capacity}}});
    output.resize(frames);
    f0 = std::move(output);
    return true;
}

bool cosyvoice_coreml_spectrum(const std::shared_ptr<cosyvoice_coreml> & model,
                              const std::vector<float> & mel, int frames,
                              const std::vector<float> & source, int source_frames,
                              std::vector<float> & spectrum) {
    const int capacity = frame_capacity(model, "mel", 2, 3, frames);
    if (!capacity || capacity > (INT_MAX - 1) / stft_frames_per_mel) return false;
    const int source_capacity = capacity * stft_frames_per_mel + 1;
    if (source_frames > source_capacity || !model->matches("mel", {1, mel_channels, capacity}) ||
        !model->matches("source_stft", {1, spectrum_channels, source_capacity}) ||
        !model->matches("spec", {1, spectrum_channels, source_capacity})) return false;
    const auto padded_mel = pad_channels(mel, mel_channels, frames, capacity);
    const auto padded_source = pad_channels(source, spectrum_channels, source_frames, source_capacity);
    std::vector<float> output(static_cast<size_t>(spectrum_channels) * source_capacity);
    model->predict({{"mel", padded_mel.data(), {1, mel_channels, capacity}},
                    {"source_stft", padded_source.data(), {1, spectrum_channels, source_capacity}}},
                   {{"spec", output.data(), {1, spectrum_channels, source_capacity}}});
    spectrum = crop_channels(output, spectrum_channels, source_frames, source_capacity);
    return true;
}
