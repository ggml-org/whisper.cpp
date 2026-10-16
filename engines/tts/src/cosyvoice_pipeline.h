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
#include "backend_selection.h"
#include "sched_dispatch.h"
#include "cosyvoice_mmap.h"
#include "cosyvoice_coreml.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Native output sample rate of the CausalHiFT vocoder (Hz). Single source of
// truth for the native rate: the vocoder source module runs at it, and the
// public engine re-exports it on SynthesisResult::sample_rate (a static_assert
// in cosyvoice_engine.cpp keeps the public constant in lock-step with this one).
constexpr int kCosyvoiceNativeSampleRate = 24000;

// GPU admission policy shared by the engine and the per-stage parity
// harnesses (test-cosyvoice-{flow,llm,hift}), so the backends a test
// validates are exactly the backends the engine will select. Vulkan and
// CUDA share one platform gate: both are admitted positively on Windows
// and non-Android Linux -- the platforms their reference runs cover --
// rather than "everywhere but Android":
// on Apple a MoltenVK registration with Metal unavailable must not let
// Vulkan win, and on Android an always-vendor-allowed Samsung Xclipse
// device must keep declining to CPU instead of offloading through a
// backend no reference run covers. TTS_CPP_COSYVOICE_GPU in
// CMakeLists.txt mirrors this condition for gpu test registration.
inline ::tts_cpp::detail::GpuBackendRequirement cosyvoice_gpu_requirement() {
    using ::tts_cpp::detail::GpuBackendRequirement;
#if defined(_WIN32) || (defined(__linux__) && !defined(__ANDROID__))
    return GpuBackendRequirement::Metal | GpuBackendRequirement::OpenCL |
           GpuBackendRequirement::Vulkan | GpuBackendRequirement::CUDA;
#else
    return GpuBackendRequirement::Metal | GpuBackendRequirement::OpenCL;
#endif
}

ggml_backend_t cosyvoice_init_backend(const std::string & requested, int n_gpu_layers,
                                      int vulkan_device, bool * gpu_present_but_unused);

bool cosyvoice_hexagon_runs_weight_type(ggml_type type);

ggml_backend_t cosyvoice_frontend_backend(ggml_backend_t engine_backend);

// ---- resident model (weights + compute backend) ----------------------------
// Move-only: owns a sched bundle that cannot be copied.
struct model_ctx {
    // Compute backend.  Borrowed when owns_backend is false — the creator frees
    // it, and only after cosyvoice_free() has run on every model_ctx borrowing
    // it (the sched holds backend refs; see sched_dispatch.h).
    ggml_backend_t        backend      = nullptr;
    bool                  owns_backend = false;

    ggml_context *        ctx_w    = nullptr;   // graph-visible weights
    ggml_backend_buffer_t buffer_w = nullptr;

    // Host-resident lookup tables (see cosyvoice_host_resident in the .cpp).
    // Null when the GGUF carries none, and unused when the backend is the CPU.
    ggml_context *        ctx_h    = nullptr;
    ggml_backend_buffer_t buffer_h = nullptr;

    // Map-in-place GGUF backing on the CPU backend (null on the GPU path).
    // map_buf wraps the whole mapping; freeing it does not munmap.
    tts_cpp::cosyvoice::MappedFile mapped;
    ggml_backend_buffer_t          map_buf = nullptr;

    std::map<std::string, ggml_tensor*> tensors;   // spans both contexts
    // GGUF scalar metadata captured at load (e.g. cosyvoice3.llm.sos,
    // cosyvoice3.llm.depth, ...), so the graph reads the ids/sizes the weights
    // were converted with instead of trusting hardcoded constants.
    std::map<std::string, int64_t> kv_i;   // uint/int-typed KV
    std::map<std::string, float>   kv_f;   // float-typed KV

    // Lazily-built [backend, CPU-last] scheduler, used only for graphs the
    // backend cannot fully run (HiFT decode's CONV_TRANSPOSE_1D on OpenCL).
    // unique_ptr because sched_fallback is non-copyable and non-movable.
    std::unique_ptr<::tts_cpp::detail::sched_fallback> sched_fb;
    // Hash-set size the cached sched was built with.  One model_ctx feeds
    // several graphs of different sizes (flow: front-end + DiT; hift: f0 +
    // STFT + decode), and ggml_backend_sched_alloc_graph asserts
    // hash_set.size >= n_nodes + n_leafs -- so a sched built for the smallest
    // graph must be rebuilt before a larger one runs through it.
    size_t sched_graph_size = 0;

    int n_threads = 0;   // 0 = leave the backend default
    std::shared_ptr<cosyvoice_coreml> coreml;
    std::shared_ptr<cosyvoice_coreml> coreml_frontend;
    std::shared_ptr<cosyvoice_coreml> coreml_f0;
};

// Load a GGUF into a resident model_ctx (all tensors materialised).
//   backend == nullptr : create and OWN a CPU backend.
//   backend != nullptr : borrow it; the caller owns it and must outlive every
//                        model_ctx that borrows it.
model_ctx cosyvoice_load_gguf(const std::string & path, ggml_backend_t backend = nullptr);
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
bool cosyvoice_lm_fa_enabled(ggml_backend_t backend, const qwen_hp & hp);
bool cosyvoice_lm_replay_enabled(ggml_backend_t backend, const qwen_hp & hp);
struct dit_hp {
    int depth = 22, dim = 1024, heads = 16, dim_head = 64, ff_inner = 2048;
    int conv_k = 31, conv_groups = 16;
};

// Build the DiT hyper-parameters from a loaded flow GGUF's cosyvoice3.flow.*
// KV, falling back to the struct defaults for any key the GGUF doesn't carry
// (real converted GGUFs predate these keys, so their graph shape is
// unchanged).  Mirrors cosyvoice_qwen_hp.
dit_hp cosyvoice_dit_hp(const model_ctx & m);
bool cosyvoice_dit_fa_config(ggml_backend_t backend, const dit_hp & hp, ggml_type & kv_type);

// ---- low-level graph builders (exposed for the parity CLIs) ---------------
// Qwen2 prefill: x [hidden, L] -> logits [6761, L] (also tags "hidden" output).
ggml_tensor * build_qwen(ggml_context * c, const model_ctx & m, const qwen_hp & hp,
                         ggml_tensor * x, ggml_tensor * pos, ggml_tensor * mask, int L);
// DiT estimator: (x,mu,cond,spks,time_sin,pos) -> dit_out [80, N - n_cut, B].
// n_cut > 0 treats the first n_cut positions as attention conditioning only
// (block 0 attends over the full sequence, later blocks and the output cover
// just the generated region); 0 (the default) is the full reference path.
ggml_tensor * build_dit(ggml_context * c, const model_ctx & m, const dit_hp & hp,
                        ggml_tensor * x, ggml_tensor * mu, ggml_tensor * cond,
                        ggml_tensor * spks, ggml_tensor * time_sin, ggml_tensor * pos,
                        int N, int B, int n_cut = 0);
// SinusPositionEmbedding(dim, scale=1000); returns [dim, B] (b outer).
std::vector<float> sinus_time_emb(const std::vector<float> & t, int dim);

// ---- optional per-stage instrumentation -----------------------------------
// Pass nullptr (the default) and nothing is measured.  Times are wall-clock
// milliseconds taken AFTER ggml_backend_synchronize, so a GPU backend's async
// submission cannot be mistaken for speed.  The counters exist so a CPU leg and
// a GPU leg can be proven to have done the same work before their times are
// compared -- a leg that decoded a different number of tokens is void, not slow.
struct cosyvoice_timings {
    double lm_prefill_ms    = 0;
    double lm_decode_ms     = 0;   // summed over all decode steps
    double flow_frontend_ms = 0;
    double dit_euler_ms     = 0;   // summed over the 10 Euler steps
    double hift_f0_ms       = 0;
    double hift_source_ms   = 0;   // CPU-side SineGen2 excitation
    double hift_stft_ms     = 0;
    double hift_decode_ms   = 0;

    int n_decode_steps  = 0;   // LM decode iterations actually run
    int n_speech_tokens = 0;   // tokens the LM emitted
    int tm              = 0;   // flow frames in  (prompt + generated)
    int mel_len         = 0;   // mel frames out  (after the prompt trim)

    // LM prefill composition (filled by the engine even when a pinned
    // trajectory skips the LM): template + transcript + text ids, and the
    // reference/baked prompt speech tokens the LM continues from.
    int n_text_ids             = 0;
    int n_prompt_speech_tokens = 0;
};

// ---- high-level engine stages (in-memory, no file I/O) --------------------
// Autoregressive speech-token decode.  `text_ids` = tokenized (prompt+tts)
// text; `prompt_stok` = reference-voice S3 tokens.  Returns speech tokens in
// [0, 6561).  min_len suppresses EOS for the first min_len steps.
std::vector<int> cosyvoice_llm_generate(model_ctx & m, const qwen_hp & hp,
                                        const std::vector<int> & text_ids,
                                        const std::vector<int> & prompt_stok,
                                        int max_steps, bool greedy, int seed, int min_len,
                                        cosyvoice_timings * tmg = nullptr, bool allow_replay = true);

// DiT flow: (prompt_token ++ speech_tokens) -> mel.  prompt_feat is the prompt
// mel [mel_len1][80] row-major (mel-fastest); embedding is the 192-d CAM++
// speaker vector.  Returns mel [80, out_mel_len] channel-major (mel[ch*T + t]).
// n_cut > 0 (clamped to mel_len1) turns on the opt-in prompt cut: the prompt
// frames condition attention only and the Euler solve integrates just the
// generated region.  0 (the default) is the full reference path.
std::vector<float> cosyvoice_flow_run(model_ctx & m,
                                      const std::vector<int> & prompt_token,
                                      const std::vector<int> & speech_tokens,
                                      const std::vector<float> & prompt_feat, int mel_len1,
                                      const std::vector<float> & embedding, int & out_mel_len,
                                      cosyvoice_timings * tmg = nullptr,
                                      int n_cut = 0);

// HiFT f0 predictor alone: mel [80, mel_len] channel-major -> per-frame f0 [Hz].
std::vector<float> cosyvoice_hift_f0(model_ctx & m,
                                     const std::vector<float> & mel, int mel_len);

// Plain conv1d graph builder: input [Nlen, Cin, B] (ne0=time), weight
// [K, Cin, Cout]. Guards its im2col with ggml_cont, so a strided view input
// stays on the backend's direct path instead of demoting the whole stage to
// the sched-fallback (ggml-vulkan's IM2COL requires a contiguous signal).
// Exposed for test-cosyvoice-conv1d, which pins exactly that guard.
ggml_tensor * cosyvoice_conv1d_f32(ggml_context * c, ggml_tensor * w, ggml_tensor * x,
                                   int stride, int padding, int dilation);

// Grouped conv1d over time: x [Nlen, Cin, B] (ne0=time), weight
// [K, Cin/groups, Cout].  The per-group form emits one im2col + matmul per
// (group, batch) — the shape Adreno's im2col fusion requires; the batched
// form folds (group, batch) into one im2col and one batched matmul and is
// selected where per-dispatch launch overhead dominates (Metal).  Both are
// exposed for test-cosyvoice-conv1d, which pins their equivalence.
ggml_tensor * cosyvoice_conv1d_grouped(ggml_context * c, ggml_tensor * w, ggml_tensor * x, int groups);
ggml_tensor * cosyvoice_conv1d_grouped_batched(ggml_context * c, ggml_tensor * w, ggml_tensor * x, int groups);

ggml_tensor * cosyvoice_dit_rope_first_head(ggml_context * c, ggml_tensor * z, ggml_tensor * pos,
                                            int dim_head, int NL, int B, bool rows_in_place);

ggml_tensor * cosyvoice_istft_columns(ggml_context * c, ggml_tensor * kernel_cols, ggml_tensor * spec, int hop);
std::vector<float> cosyvoice_istft_kernel_columns(const std::vector<float> & kernel, int n_fft, int n_ch);

// SineGen2 NSF source excitation (host-side): sample-rate f0 [T_wav] ->
// source [T_wav].  The per-harmonic sine tracks and the mix/tanh arithmetic
// run on up to n_threads host threads (<= 0 selects all hardware threads);
// the RNG draws stay one sequential pass in the original order, so the output
// is bit-identical for a given seed at any thread count.  Exposed for
// test-cosyvoice-sinegen, which pins that equality.
std::vector<float> cosyvoice_sinegen2_source(const std::vector<float> & f0_wav,
                                             int sampling_rate, int harmonic_num,
                                             float sine_amp, float noise_std,
                                             float voiced_threshold, int upsample_scale,
                                             const std::vector<float> & l_linear_w,
                                             float l_linear_b, uint32_t seed,
                                             int n_threads);

// CausalHiFT vocoder: mel [80, mel_len] channel-major (mel[ch*T + t]) -> 24 kHz
// float PCM.  Runs f0_predictor + SineGen2 excitation + STFT + decode.
// f0_override (parity tests) skips the predictor: SineGen2 integrates f0 into
// sine phases, so cross-backend f0 noise decorrelates the raw waveform unless
// pinned.
std::vector<float> cosyvoice_hift_synth(model_ctx & m,
                                        const std::vector<float> & mel, int mel_len, int seed,
                                        cosyvoice_timings * tmg = nullptr,
                                        const std::vector<float> * f0_override = nullptr);
