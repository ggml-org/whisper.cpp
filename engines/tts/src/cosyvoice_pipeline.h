// CosyVoice3 native pipeline — shared library core (CPU / ggml).
//
// Single source of truth for the three CosyVoice3 back-half stages, factored
// out of the standalone parity CLIs (cosyvoice_llm.cpp / cosyvoice_flow.cpp /
// cosyvoice_hift.cpp) so that both those CLIs and the public Engine
// (cosyvoice_engine.cpp) share one validated implementation.
//
//   text ids + prompt speech tokens --(Qwen2.5 LM)--> speech tokens [0,6561)
//   speech tokens (+prompt) --------(DiT flow, Euler)--> mel [80, T]
//   mel ---------------------------(CausalHiFT vocoder)--> 24 kHz PCM
//
// Everything here is in-memory (no npy / no file I/O) and depends only on ggml,
// so it compiles unchanged against both the in-tree ggml (CLIs) and the
// ggml-speech vcpkg port (library / addon).  Numerics validated against the
// PyTorch reference (flow_mel cosine 1.0, LM prefill cosine 1.0).

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Native output sample rate of the CausalHiFT vocoder (Hz). Single source of
// truth for the native rate: the vocoder source module runs at it, and the
// public engine re-exports it on SynthesisResult::sample_rate (a static_assert
// in cosyvoice_engine.cpp keeps the public constant in lock-step with this one).
constexpr int kCosyvoiceNativeSampleRate = 24000;

// ---- resident model (weights + CPU backend) -------------------------------
struct model_ctx {
    ggml_backend_t        backend  = nullptr;
    ggml_context *        ctx_w    = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    // GGUF scalar metadata captured at load (e.g. cosyvoice3.llm.sos,
    // cosyvoice3.llm.depth, ...), so the graph reads the ids/sizes the weights
    // were converted with instead of trusting hardcoded constants.
    std::map<std::string, int64_t> kv_i;   // uint/int-typed KV
    std::map<std::string, float>   kv_f;   // float-typed KV
};

// Load a GGUF into a resident CPU model_ctx (all tensors materialised).
model_ctx cosyvoice_load_gguf(const std::string & path);
// Free the backend + weight context held by a model_ctx.
void cosyvoice_free(model_ctx & m);
// Tensor lookup by name (throws std::runtime_error if absent).
ggml_tensor * cosyvoice_get(const model_ctx & m, const std::string & name);

// Read a string metadata value from a GGUF file (returns `fallback` if the key
// is absent or not a string).  Used to read a baked voice's prompt transcript.
std::string cosyvoice_gguf_meta_str(const std::string & path, const std::string & key,
                                    const std::string & fallback = "");

// Read a scalar metadata value captured at load (see model_ctx::kv_i / kv_f).
// Returns `fallback` when the key is absent — so an older GGUF written before
// the key existed still loads with the historical constant.
int64_t cosyvoice_meta_i(const model_ctx & m, const std::string & key, int64_t fallback);
float   cosyvoice_meta_f(const model_ctx & m, const std::string & key, float fallback);

// ---- hyper-parameters -----------------------------------------------------
struct qwen_hp {
    int   depth = 24, hidden = 896, n_head = 14, n_kv = 2, head_dim = 64, inter = 4864;
    float theta = 1000000.0f, eps = 1e-6f;
};

// Build the Qwen2 hyper-parameters from a loaded LM GGUF's cosyvoice3.llm.* KV,
// falling back to the struct defaults for any key the GGUF doesn't carry. Use
// this instead of a default-constructed qwen_hp so the graph shape follows the
// weights rather than hardcoded numbers.
qwen_hp cosyvoice_qwen_hp(const model_ctx & m);
struct dit_hp {
    int depth = 22, dim = 1024, heads = 16, dim_head = 64, ff_inner = 2048;
    int conv_k = 31, conv_groups = 16;
};

// ---- low-level graph builders (exposed for the parity CLIs) ---------------
// Qwen2 prefill: x [hidden, L] -> logits [6761, L] (also tags "hidden" output).
ggml_tensor * build_qwen(ggml_context * c, const model_ctx & m, const qwen_hp & hp,
                         ggml_tensor * x, ggml_tensor * pos, ggml_tensor * mask, int L);
// DiT estimator: (x,mu,cond,spks,time_sin,pos) -> dit_out [80, N, B].
ggml_tensor * build_dit(ggml_context * c, const model_ctx & m, const dit_hp & hp,
                        ggml_tensor * x, ggml_tensor * mu, ggml_tensor * cond,
                        ggml_tensor * spks, ggml_tensor * time_sin, ggml_tensor * pos,
                        ggml_tensor * one, int N, int B);
// SinusPositionEmbedding(dim, scale=1000); returns [dim, B] (b outer).
std::vector<float> sinus_time_emb(const std::vector<float> & t, int dim);

// ---- high-level engine stages (in-memory, no file I/O) --------------------
// Autoregressive speech-token decode.  `text_ids` = tokenized (prompt+tts)
// text; `prompt_stok` = reference-voice S3 tokens.  Returns speech tokens in
// [0, 6561).  min_len suppresses EOS for the first min_len steps.
std::vector<int> cosyvoice_llm_generate(model_ctx & m, const qwen_hp & hp,
                                        const std::vector<int> & text_ids,
                                        const std::vector<int> & prompt_stok,
                                        int max_steps, bool greedy, int seed, int min_len);

// DiT flow: (prompt_token ++ speech_tokens) -> mel.  prompt_feat is the prompt
// mel [mel_len1][80] row-major (mel-fastest); embedding is the 192-d CAM++
// speaker vector.  Returns mel [80, out_mel_len] channel-major (mel[ch*T + t]).
std::vector<float> cosyvoice_flow_run(model_ctx & m,
                                      const std::vector<int> & prompt_token,
                                      const std::vector<int> & speech_tokens,
                                      const std::vector<float> & prompt_feat, int mel_len1,
                                      const std::vector<float> & embedding, int & out_mel_len);

// CausalHiFT vocoder: mel [80, mel_len] channel-major (mel[ch*T + t]) -> 24 kHz
// float PCM.  Runs f0_predictor + SineGen2 excitation + STFT + decode.
std::vector<float> cosyvoice_hift_synth(model_ctx & m,
                                        const std::vector<float> & mel, int mel_len, int seed);
