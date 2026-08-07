// Forward-parity test for the Supertonic vocoder backward (voice-clone ticket
// "GGML backward pass: vocoder", QVAC-20983).
//
// Why this test exists
// --------------------
// `test_supertonic_vocoder_backward.cpp` gradchecks the analytic backward
// against the *in-file* `VocoderBackward::forward`. That proves the backward is
// the correct derivative of that forward, but it is self-referential: if
// `VocoderBackward::forward` itself drifted from the production vocoder, the
// gradcheck would still pass while the gradients flow through the wrong
// function (this is exactly how a `gamma` dimensionality bug slipped past it).
//
// This test closes that gap. It builds a synthetic `supertonic_model` on a CPU
// backend with deterministic weights, feeds the *identical* raw weight buffers
// to both `supertonic_vocoder_forward_cpu` (production) and
// `VocoderBackward::forward` (the backward's reference forward), and asserts the
// two waveforms match. Any divergence between the reference forward and the
// production forward — wrong gamma layout, wrong dilation schedule, swapped
// weight index order — fails here.
//
// Model-free: weights are synthesized in-memory, so it always runs in the
// always-on `unit` ctest tier (no GGUF, no fixtures).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "supertonic_internal.h"
#include "supertonic_vocoder_backward.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace tts_cpp::supertonic::detail;
using tts_cpp::voc_grad::VocConvNextWeights;
using tts_cpp::voc_grad::VocoderBackward;
using tts_cpp::voc_grad::VocoderWeights;

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, ...) do {                                  \
    ++g_checks;                                                \
    if (!(cond)) {                                             \
        ++g_failures;                                          \
        std::fprintf(stderr, "FAIL %s:%d  ", __FILE__, __LINE__); \
        std::fprintf(stderr, __VA_ARGS__);                     \
        std::fprintf(stderr, "\n");                            \
    }                                                          \
} while (0)

// Production ConvNeXt dilation schedule (`convnext_block` in
// supertonic_vocoder.cpp). The 10-block count is fixed by the model.
constexpr int kNumBlocks = 10;
constexpr int kDilations[kNumBlocks] = {1, 2, 4, 1, 2, 4, 1, 1, 1, 1};

// Topology kept tiny (microsecond runtime) but structurally faithful: real
// dilation variety, per-channel gamma, the full denorm -> embed -> 10x convnext
// -> batch-norm -> head1 -> prelu -> head2 chain.
struct VocoderDims {
    int C_latent  = 4;
    int factor    = 2;   // latent_channels = C_latent * factor = 8
    int latent_len = 5;  // T0 = latent_len * factor = 10
    int C         = 8;   // ConvNeXt width
    int K_embed   = 3;
    int hidden    = 16;
    int K_dw      = 3;
    int Hh        = 4;   // head1 output channels
    int K_head1   = 3;
    int OUT       = 1;   // waveform channels

    int latent_channels() const { return C_latent * factor; }
    int T0() const { return latent_len * factor; }
};

// Deterministic, bounded weight generator. Small magnitudes keep the 10-block
// residual chain well-scaled so float (production) vs double (reference)
// rounding stays in the sub-1e-3 band; the values still vary per element so the
// test is not degenerate.
float gen_value(int index, double phase, double scale) {
    return (float) (scale * std::sin(index * 0.7 + phase));
}

std::vector<float> gen_buffer(int n, double phase, double scale = 0.25) {
    std::vector<float> v((std::size_t) n);
    for (int i = 0; i < n; ++i) v[(std::size_t) i] = gen_value(i, phase, scale);
    return v;
}

// Strictly positive buffer for batch-norm running variance.
std::vector<float> gen_positive(int n, double phase) {
    std::vector<float> v((std::size_t) n);
    for (int i = 0; i < n; ++i) v[(std::size_t) i] = 0.5f + 0.3f * (gen_value(i, phase, 1.0) + 1.0f);
    return v;
}

struct BlockBuffers {
    std::vector<float> dw_w, dw_b, norm_g, norm_b, pw1_w, pw1_b, pw2_w, pw2_b, gamma;
    int dilation = 1;
};

// All vocoder weight buffers in their raw (production / ggml-linear) layout.
// The same buffers feed both forwards, so any layout mismatch is a real bug,
// not a test artifact.
struct VocoderBuffers {
    VocoderDims dims;
    std::vector<float> normalizer_scale, latent_mean, latent_std, embed_w, embed_b;
    std::vector<BlockBuffers> blocks;
    std::vector<float> final_g, final_b, final_mean, final_var;
    std::vector<float> head1_w, head1_b, head_prelu, head2_w;
    std::vector<float> latent;
};

BlockBuffers gen_block(const VocoderDims & d, int dilation, double phase) {
    BlockBuffers b;
    b.dilation = dilation;
    b.dw_w   = gen_buffer(d.C * d.K_dw, phase + 0.1);
    b.dw_b   = gen_buffer(d.C, phase + 0.2);
    b.norm_g = gen_buffer(d.C, phase + 0.3);
    b.norm_b = gen_buffer(d.C, phase + 0.4);
    b.pw1_w  = gen_buffer(d.hidden * d.C, phase + 0.5);
    b.pw1_b  = gen_buffer(d.hidden, phase + 0.6);
    b.pw2_w  = gen_buffer(d.C * d.hidden, phase + 0.7);
    b.pw2_b  = gen_buffer(d.C, phase + 0.8);
    b.gamma  = gen_buffer(d.C, phase + 0.9, 0.15);  // per-channel residual scale
    return b;
}

VocoderBuffers gen_vocoder_buffers() {
    VocoderBuffers vb;
    const VocoderDims & d = vb.dims;

    vb.normalizer_scale = {1.7f};
    vb.latent_mean = gen_buffer(d.C_latent, 0.2);
    vb.latent_std  = gen_buffer(d.C_latent, 0.5);
    vb.embed_w = gen_buffer(d.C * d.C_latent * d.K_embed, 0.9);
    vb.embed_b = gen_buffer(d.C, 1.0);

    for (int i = 0; i < kNumBlocks; ++i) {
        vb.blocks.push_back(gen_block(d, kDilations[i], 1.0 + 0.31 * i));
    }

    vb.final_g    = gen_buffer(d.C, 0.4);
    vb.final_b    = gen_buffer(d.C, 0.7);
    vb.final_mean = gen_buffer(d.C, 0.1);
    vb.final_var  = gen_positive(d.C, 0.5);

    vb.head1_w = gen_buffer(d.Hh * d.C * d.K_head1, 0.3);
    vb.head1_b = gen_buffer(d.Hh, 0.45);
    vb.head_prelu = {0.1f};
    vb.head2_w = gen_buffer(d.OUT * d.Hh, 0.65);

    vb.latent = gen_buffer(d.latent_channels() * d.latent_len, 1.3, 0.6);
    return vb;
}

// --- production-side model assembly -----------------------------------------

// Owns the ggml resources backing the synthetic model so the test can release
// them deterministically (free_supertonic_model is avoided — this model never
// went through load_supertonic_gguf and carries no scheduler / source map).
struct GgmlModelArena {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ~GgmlModelArena() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};

void set_tensor(ggml_tensor * t, const std::vector<float> & data) {
    if ((std::size_t) ggml_nelements(t) != data.size()) {
        throw std::runtime_error("tensor element count mismatch while uploading weights");
    }
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
}

void build_block_tensors(ggml_context * ctx, const VocoderDims & d,
                         supertonic_vocoder_convnext_weights & w) {
    w.dw_w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d.K_dw, 1, d.C);
    w.dw_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.C);
    w.norm_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.C);
    w.norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.C);
    w.pw1_w  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, d.C, d.hidden);
    w.pw1_b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.hidden);
    w.pw2_w  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, d.hidden, d.C);
    w.pw2_b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.C);
    w.gamma  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.C);
}

void build_vocoder_tensors(ggml_context * ctx, const VocoderDims & d, supertonic_model & model) {
    supertonic_vocoder_weights & v = model.vocoder;
    v.normalizer_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    v.latent_mean = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.C_latent);
    v.latent_std  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.C_latent);
    v.embed_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d.K_embed, d.C_latent, d.C);
    v.embed_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.C);
    for (int i = 0; i < kNumBlocks; ++i) build_block_tensors(ctx, d, v.convnext[(std::size_t) i]);
    v.final_norm_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.C);
    v.final_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.C);
    v.final_norm_running_mean = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.C);
    v.final_norm_running_var  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.C);
    v.head1_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d.K_head1, d.C, d.Hh);
    v.head1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.Hh);
    v.head_prelu = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    v.head2_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, d.Hh, d.OUT);
}

void upload_vocoder_weights(const VocoderBuffers & vb, supertonic_model & model) {
    const supertonic_vocoder_weights & v = model.vocoder;
    set_tensor(v.normalizer_scale, vb.normalizer_scale);
    set_tensor(v.latent_mean, vb.latent_mean);
    set_tensor(v.latent_std, vb.latent_std);
    set_tensor(v.embed_w, vb.embed_w);
    set_tensor(v.embed_b, vb.embed_b);
    for (int i = 0; i < kNumBlocks; ++i) {
        const BlockBuffers & b = vb.blocks[(std::size_t) i];
        const supertonic_vocoder_convnext_weights & w = v.convnext[(std::size_t) i];
        set_tensor(w.dw_w, b.dw_w);
        set_tensor(w.dw_b, b.dw_b);
        set_tensor(w.norm_g, b.norm_g);
        set_tensor(w.norm_b, b.norm_b);
        set_tensor(w.pw1_w, b.pw1_w);
        set_tensor(w.pw1_b, b.pw1_b);
        set_tensor(w.pw2_w, b.pw2_w);
        set_tensor(w.pw2_b, b.pw2_b);
        set_tensor(w.gamma, b.gamma);
    }
    set_tensor(v.final_norm_g, vb.final_g);
    set_tensor(v.final_norm_b, vb.final_b);
    set_tensor(v.final_norm_running_mean, vb.final_mean);
    set_tensor(v.final_norm_running_var, vb.final_var);
    set_tensor(v.head1_w, vb.head1_w);
    set_tensor(v.head1_b, vb.head1_b);
    set_tensor(v.head_prelu, vb.head_prelu);
    set_tensor(v.head2_w, vb.head2_w);
}

// Builds a CPU-backed `supertonic_model` whose vocoder weights are the synthetic
// buffers. `arena` owns the ggml resources for deterministic teardown.
void build_synthetic_model(const VocoderBuffers & vb, supertonic_model & model, GgmlModelArena & arena) {
    const VocoderDims & d = vb.dims;
    model.hparams.latent_dim = d.C_latent;
    model.hparams.ttl_chunk_compress_factor = d.factor;
    model.hparams.latent_channels = d.latent_channels();

    arena.backend = ggml_backend_cpu_init();
    if (!arena.backend) throw std::runtime_error("ggml_backend_cpu_init failed");

    constexpr int kMaxTensors = 256;  // 5 + 10*9 + 4 + 4 = 103 tensors, padded.
    const std::size_t mem = ggml_tensor_overhead() * kMaxTensors;
    ggml_init_params params = { mem, nullptr, /*no_alloc=*/true };
    arena.ctx = ggml_init(params);
    if (!arena.ctx) throw std::runtime_error("ggml_init failed");

    build_vocoder_tensors(arena.ctx, d, model);
    arena.buffer = ggml_backend_alloc_ctx_tensors(arena.ctx, arena.backend);
    if (!arena.buffer) throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed");

    upload_vocoder_weights(vb, model);

    model.backend = arena.backend;
    model.backend_is_cpu = true;
}

// --- reference-side weight assembly -----------------------------------------

std::vector<double> to_double(const std::vector<float> & v) {
    return std::vector<double>(v.begin(), v.end());
}

VocoderWeights build_reference_weights(const VocoderBuffers & vb) {
    const VocoderDims & d = vb.dims;
    VocoderWeights w;
    w.latent_len = d.latent_len;
    w.C_latent   = d.C_latent;
    w.factor     = d.factor;
    w.C          = d.C;
    w.normalizer_scale = vb.normalizer_scale[0];
    w.latent_mean = to_double(vb.latent_mean);
    w.latent_std  = to_double(vb.latent_std);

    w.K_embed = d.K_embed;
    w.embed_w = to_double(vb.embed_w);
    w.embed_b = to_double(vb.embed_b);

    for (int i = 0; i < kNumBlocks; ++i) {
        const BlockBuffers & b = vb.blocks[(std::size_t) i];
        VocConvNextWeights c;
        c.C = d.C; c.hidden = d.hidden; c.K = d.K_dw; c.dilation = b.dilation;
        c.dw_w = to_double(b.dw_w);
        c.dw_b = to_double(b.dw_b);
        c.ln_gamma = to_double(b.norm_g);
        c.ln_beta = to_double(b.norm_b);
        c.pw1_w = to_double(b.pw1_w);
        c.pw1_b = to_double(b.pw1_b);
        c.pw2_w = to_double(b.pw2_w);
        c.pw2_b = to_double(b.pw2_b);
        c.gamma = to_double(b.gamma);
        w.convnext.push_back(std::move(c));
    }

    w.bn_gamma        = to_double(vb.final_g);
    w.bn_beta         = to_double(vb.final_b);
    w.bn_running_mean = to_double(vb.final_mean);
    w.bn_running_var  = to_double(vb.final_var);

    w.Hh = d.Hh;
    w.K_head1 = d.K_head1;
    w.head1_w = to_double(vb.head1_w);
    w.head1_b = to_double(vb.head1_b);
    w.prelu_slope = vb.head_prelu[0];

    w.OUT = d.OUT;
    w.head2_w = to_double(vb.head2_w);
    return w;
}

// --- the parity check --------------------------------------------------------

void test_forward_parity() {
    const VocoderBuffers vb = gen_vocoder_buffers();
    const VocoderDims & d = vb.dims;

    supertonic_model model;
    GgmlModelArena arena;
    build_synthetic_model(vb, model, arena);

    std::vector<float> wav_prod;
    std::string error;
    const bool ok = supertonic_vocoder_forward_cpu(model, vb.latent.data(), d.latent_len, wav_prod, &error);
    CHECK(ok, "supertonic_vocoder_forward_cpu failed: %s", error.c_str());
    if (!ok) return;

    VocoderBackward backward(build_reference_weights(vb));
    const std::vector<double> wav_ref = backward.forward(to_double(vb.latent));

    const std::size_t expected = (std::size_t) d.T0() * d.OUT;
    CHECK(wav_prod.size() == expected, "production wav size %zu != expected %zu", wav_prod.size(), expected);
    CHECK(wav_ref.size() == expected, "reference wav size %zu != expected %zu", wav_ref.size(), expected);
    if (wav_prod.size() != expected || wav_ref.size() != expected) return;

    double max_abs = 0.0;
    double max_mag = 0.0;
    for (std::size_t i = 0; i < expected; ++i) {
        max_abs = std::max(max_abs, std::fabs((double) wav_prod[i] - wav_ref[i]));
        max_mag = std::max(max_mag, std::fabs(wav_ref[i]));
    }
    // float production vs double reference over a 10-block chain; observed error
    // is ~2e-8, so this tight bar is a meaningful parity check (~500x margin)
    // while the bug it guards (per-channel gamma modeled as scalar, wrong
    // dilation, swapped weight index) shifts the output by O(output magnitude).
    constexpr double kAbsTol = 1e-5;
    std::fprintf(stderr, "[vocoder forward parity] max_abs_err=%.3e max_ref_mag=%.3e atol=%.0e\n",
                 max_abs, max_mag, kAbsTol);
    CHECK(max_abs <= kAbsTol, "forward parity exceeded tolerance: max_abs=%.3e > %.0e", max_abs, kAbsTol);
}

}  // namespace

int main() {
    try {
        test_forward_parity();
    } catch (const std::exception & e) {
        ++g_failures;
        std::fprintf(stderr, "FAIL uncaught exception: %s\n", e.what());
    }
    std::fprintf(stderr, "\n%s: %d/%d checks passed\n",
                 g_failures == 0 ? "PASS" : "FAIL", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
