#pragma once
// Chatterbox S3Gen Core ML sidecars (TTS_CPP_COREML builds, Apple only).
//
// scripts/export-chatterbox-coreml.py exports three stages of an S3Gen GGUF
// next to it, named by the shared rule (coreml_sidecar_path.h):
//
//   chatterbox-s3gen-q8_0.gguf -> chatterbox-s3gen-encoder.mlmodelc
//                                 chatterbox-s3gen-cfm.mlmodelc
//                                 chatterbox-s3gen-hift.mlmodelc
//
// The encoder and CFM estimator are multi-function models, one function per
// token ("tokens_<T>") or frame ("frames_<F>") width; a call takes the
// narrowest function that holds it and masks the padding. HiFT is one fixed
// window walked over the utterance with HIFT_CONTEXT_FRAMES of context on
// each side. A stage with no function wide enough, a missing sidecar or a
// retired one runs on ggml; tokenization, the time embedding, the Euler
// update, the F0 predictor, SineGen, STFT and inverse STFT always do.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

struct moss_coreml_model;

namespace tts_cpp::chatterbox::detail {

constexpr const char * COREML_DISABLE_ENV = "CHATTERBOX_COREML_DISABLE";
constexpr const char * COREML_STRICT_ENV = "CHATTERBOX_COREML_STRICT";
constexpr const char * COREML_UNITS_ENV = "CHATTERBOX_COREML_COMPUTE_UNITS";
// CPU + Neural Engine: measured fastest for every stage on an M2, and it keeps
// the GPU free for T3.
constexpr const char * COREML_DEFAULT_UNITS = "cpu_and_ane";
constexpr const char * COREML_ENCODER_SUFFIX = "-encoder";
constexpr const char * COREML_CFM_SUFFIX = "-cfm";
constexpr const char * COREML_HIFT_SUFFIX = "-hift";
constexpr const char * COREML_ENCODER_PREFIX = "tokens_";
constexpr const char * COREML_CFM_PREFIX = "frames_";
constexpr const char * GGML_STAGE_BACKEND = "ggml";
constexpr const char * MIXED_STAGE_BACKEND = "mixed";

// Support radius of a HiFT output frame in mel frames; must equal
// hift_receptive_frames() in scripts/chatterbox_coreml_models.py.
constexpr int HIFT_CONTEXT_FRAMES = 15;
constexpr int HIFT_STFT_PER_FRAME = 120;
constexpr int HIFT_STFT_BINS = 18;

// Resolved once per Engine (or per s3gen_synthesize_to_wav call for direct
// callers). Explicit values never write the environment.
struct coreml_policy {
    std::string compute_units = COREML_DEFAULT_UNITS;
    bool strict = false;
    bool disabled = false;
};

bool coreml_valid_units(const std::string & units);

// Empty `compute_units` inherits CHATTERBOX_COREML_COMPUTE_UNITS (an invalid
// inherited value warns and keeps the default); an explicit invalid value
// throws. Unset `strict` inherits CHATTERBOX_COREML_STRICT (presence means
// true). CHATTERBOX_COREML_DISABLE disables every stage.
coreml_policy resolve_coreml_policy(const std::string & compute_units = {},
                                    std::optional<bool> strict = std::nullopt);

std::string coreml_stage_path(const std::string & s3gen_gguf, const char * suffix);

// True when this build has Core ML, the policy leaves it enabled and any
// stage sidecar sits next to the GGUF (presence only; nothing is loaded).
bool coreml_sidecars_enabled(const std::string & s3gen_gguf, const coreml_policy & policy);

// True when both the CFM and HiFT sidecars are enabled, i.e. S3Gen's GPU work
// moves off the GPU, so S3Gen of one sentence can overlap T3 of the next
// without the two contending for it.
bool coreml_offloads_s3gen(const std::string & s3gen_gguf, const coreml_policy & policy);

// "frames_512,frames_640" + "frames_" -> {512, 640} (sorted, unique).
std::vector<int> coreml_buckets(const std::string & function_names, const std::string & prefix);

// The narrowest bucket >= needed, or -1.
int coreml_pick_bucket(const std::vector<int> & buckets, int needed);

// One HiFT sidecar call: frames [start, start + window) go in, the output of
// frames [keep_begin, keep_end) is kept. Kept spans tile [0, frames).
struct hift_window {
    int start;
    int keep_begin;
    int keep_end;
};

std::vector<hift_window> coreml_hift_windows(int frames, int window, int context);

class s3gen_coreml {
public:
    enum stage { ENCODER, CFM, HIFT, STAGES };

    s3gen_coreml(const std::string & s3gen_gguf, const std::string & compute_units, int token_mel_ratio = 2);
    ~s3gen_coreml();
    s3gen_coreml(const s3gen_coreml &) = delete;
    s3gen_coreml & operator=(const s3gen_coreml &) = delete;

    // A sidecar file exists for the stage (it may still be unusable).
    bool present(stage which) const { return present_[which]; }
    // Present, listed and not retired after a failure.
    bool available(stage which) const;
    void retire(stage which);
    const std::string & label() const { return label_; }

    // Loads HiFT and, for each expected call's S3Gen input token count, the
    // encoder and CFM functions it would use, ahead of the calls (a function
    // costs a one-time Neural Engine compile per executable, then ~0.2-1 s per
    // process). For a background thread; `stop` ends it between loads.
    void warm(const std::vector<int> & tokens, const std::atomic<bool> * stop = nullptr);
    const std::string & compute_units() const { return units_; }

    // Each returns false when its sidecar cannot serve the call (absent,
    // retired, no wide-enough function); the caller then runs ggml. Every
    // input and output shape the sidecar declares is checked against the
    // buffers the call packs before anything is copied: a mismatch (an
    // incompatible export, or the other variant's sidecar) throws
    // std::runtime_error, as does a failing prediction.

    // embedding: [tokens, 512] row-major -> mu: [tokens * ratio, 80].
    bool encode(const std::vector<float> & embedding, int tokens, int ratio, std::vector<float> & mu);

    // x / mu / cond: `batch` planes of [80, frames]; spks: batch x 80;
    // t_emb: batch x 1024 -> velocity: `batch` planes of [80, frames].
    bool velocity(int batch, int frames, const std::vector<const float *> & x,
                  const std::vector<const float *> & mu, const std::vector<const float *> & cond,
                  const std::vector<const float *> & spks, const std::vector<const float *> & t_emb,
                  std::vector<std::vector<float>> & out);

    // mel: [80, frames]; source_stft: [18, frames * 120 + 1] -> spectrum:
    // conv_post rows [18, frames * 120 + 1] (log magnitude, then phase).
    bool vocode(const std::vector<float> & mel, int frames, const std::vector<float> & source_stft,
                std::vector<float> & spectrum);

private:
    struct stage_state {
        std::string path;
        std::vector<int> buckets;  // empty for the single-function HiFT model
        std::map<int, std::shared_ptr<moss_coreml_model>> loaded;
        std::set<int> loading;
        std::atomic<bool> retired{true};
    };

    std::shared_ptr<moss_coreml_model> model_for(stage which, int bucket);

    std::string units_;
    int ratio_ = 2;
    std::string label_ = "coreml";
    stage_state stages_[STAGES];
    bool present_[STAGES] = {};
    std::mutex mutex_;
    std::condition_variable loaded_;
};

// Expected speech tokens T3 produces for `text_bytes` of text (~1.15 per
// byte: Turbo 1.17-1.21, Multilingual 0.95-1.05 on English), for choosing
// which functions to warm.
int coreml_expected_speech_tokens(size_t text_bytes);

// Speech-token windows of the S3Gen calls a streamed sentence of
// `speech_tokens` makes (Engine::synthesize_streaming_segment's chunking:
// cumulative, or capped by `left_context` when > 0).
std::vector<int> coreml_stream_windows(int speech_tokens, int first_chunk, int chunk, int left_context);

// Opens the S3Gen GGUF's sidecars under `policy` in the process-wide cache
// and warms them (s3gen_coreml::warm) for S3Gen calls on `speech_tokens`
// each, after `prompt_tokens` of voice prompt (< 0: the GGUF's built-in
// voice). Meant for a background thread started before T3; a no-op when no
// sidecar is enabled. Defined in chatterbox_tts.cpp beside the cache.
void s3gen_coreml_warm(const std::string & s3gen_gguf, const coreml_policy & policy,
                       const std::vector<int> & speech_tokens, int prompt_tokens,
                       const std::atomic<bool> * stop = nullptr);

} // namespace tts_cpp::chatterbox::detail
