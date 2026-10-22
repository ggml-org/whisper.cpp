#pragma once

// Library-internal S3Gen measurement surface for the chatterbox memory-fit
// preflight (include/tts-cpp/chatterbox/fit.h).  Implemented in
// chatterbox_tts.cpp next to the stage graph builders it prices; consumed by
// src/chatterbox_fit.cpp.

#include <cstdint>
#include <string>

namespace tts_cpp::chatterbox::detail {

struct s3gen_fit_measure {
    // Weight buffer of the whole S3Gen GGUF (flow encoder + CFM + HiFT +
    // s3tokenizer + campplus + built-in voice), sized metadata-only.
    uint64_t weights_bytes = 0;

    // Sum of the five per-stage graph arenas (encoder, CFM estimator, F0,
    // STFT, HiFT) at the projected shapes.  Every one of these caches stays
    // resident once its stage has run, so they add.  The HiFT stage is
    // priced through the same [backend, CPU-last] scheduler shape the
    // runtime falls back to when the backend cannot run conv_transpose_1d;
    // its CPU portion lands in host_compute_bytes.
    uint64_t device_compute_bytes = 0;
    uint64_t host_compute_bytes   = 0;

    // Per-stage device figures behind device_compute_bytes, for the parity
    // test's byte-for-byte comparison against a real synthesis' caches.
    uint64_t encoder_bytes     = 0;
    uint64_t cfm_bytes         = 0;
    uint64_t f0_bytes          = 0;
    uint64_t stft_bytes        = 0;
    uint64_t hift_device_bytes = 0;

    // Host graph-metadata arenas the caches keep resident (64+64+8+4+64 MB).
    uint64_t host_arena_bytes = 0;

    // Host-side working slabs of one synthesis at the projected shapes:
    // input_embed, the mu/cond/z/dxdt family (they coexist mid-CFM), the mel
    // slice, the f0/source/wav sample-rate buffers, the persistent F32 CPU
    // weight mirrors (flow/input_embedding, spk_embed_affine) and the cached
    // positional embeddings / window sums.
    uint64_t host_slab_bytes = 0;

    // Projected shapes, for the caller's host-slab arithmetic and report.
    int  n_prompt_token = 0;  // built-in voice prompt tokens (reference audio is capped to the same)
    int  mel_len1       = 0;  // built-in prompt_feat mel frames
    int  n_total        = 0;
    int  T_mu           = 0;
    int  T_mel          = 0;
    bool meanflow       = true;
    bool used_b2        = false;
};

// Project one s3gen_synthesize_to_wav of `n_speech_tokens` speech tokens
// (batch path: +3 lookahead-silence tokens, built-in-voice conditioning
// lengths) against the S3Gen GGUF at `gguf_path`, without reading weight
// data.  Resolves the backend with the same policy a real load applies.
// Returns false with `error` set when the model is unreadable or a graph
// cannot be priced.
bool s3gen_measure_fit(const std::string & gguf_path, int n_gpu_layers,
                       const std::string & backend,
                       int n_speech_tokens, s3gen_fit_measure & out,
                       std::string * error);

}  // namespace tts_cpp::chatterbox::detail
