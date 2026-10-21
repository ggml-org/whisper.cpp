#include "internal.h"

#include "backend_selection.h"
#include "backend_util.h"
#include "gguf_stream.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace tts_cpp {
namespace parler {
namespace detail {

namespace {

// gguf_get_val_* GGML_ABORTs when the stored type differs from the requested
// one, so a mistyped key would kill the host process instead of surfacing
// through the error out-param. Check the type first and fail closed.
bool kv_type_is(const gguf_context * g, int64_t id, gguf_type want) {
    return gguf_get_kv_type(g, id) == want;
}

bool kv_u32(const gguf_context * g, const char * key, int & out, std::string * err) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        if (err) *err = std::string("missing GGUF key: ") + key;
        return false;
    }
    if (!kv_type_is(g, id, GGUF_TYPE_UINT32)) {
        if (err) *err = std::string("GGUF key is not uint32: ") + key;
        return false;
    }
    out = (int) gguf_get_val_u32(g, id);
    return true;
}

bool kv_f32(const gguf_context * g, const char * key, float & out, std::string * err) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        if (err) *err = std::string("missing GGUF key: ") + key;
        return false;
    }
    if (!kv_type_is(g, id, GGUF_TYPE_FLOAT32)) {
        if (err) *err = std::string("GGUF key is not float32: ") + key;
        return false;
    }
    out = gguf_get_val_f32(g, id);
    return true;
}

bool kv_bool(const gguf_context * g, const char * key, bool & out, std::string * err) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        if (err) *err = std::string("missing GGUF key: ") + key;
        return false;
    }
    if (!kv_type_is(g, id, GGUF_TYPE_BOOL)) {
        if (err) *err = std::string("GGUF key is not bool: ") + key;
        return false;
    }
    out = gguf_get_val_bool(g, id);
    return true;
}

// FA probe: build a representative flash_attn_ext node over `kv_type` K/V and
// ask the backend if it supports it.
bool parler_probe_fa(ggml_backend_t backend, int head_dim, int n_heads, ggml_type kv_type) {
    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/ true };
    ggml_context * c = ggml_init(ip);
    if (!c) return false;
    ggml_tensor * q = ggml_new_tensor_3d(c, GGML_TYPE_F32, head_dim, 16, n_heads);
    ggml_tensor * k = ggml_new_tensor_3d(c, kv_type, head_dim, 16, n_heads);
    ggml_tensor * v = ggml_new_tensor_3d(c, kv_type, head_dim, 16, n_heads);
    ggml_tensor * op = ggml_flash_attn_ext(c, q, k, v, nullptr, 1.0f / (float) head_dim, 0.0f, 0.0f);
    const bool ok = op && ggml_backend_supports_op(backend, op);
    ggml_free(c);
    return ok;
}

// Mark every unallocated tensor in `ctx` externally allocated (dummy non-null
// data, the same trick ggml's own measure paths use) so graph pricing via
// ggml_gallocr / ggml_backend_sched excludes it from the measured compute
// buffers instead of counting it as a graph-owned leaf. The context must
// never have tensor data read or written after this.
void mark_externally_allocated(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t;
         t = ggml_get_next_tensor(ctx, t)) {
        if (!t->data && !t->view_src) {
            t->data = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
        }
    }
}

std::vector<std::pair<ggml_tensor *, ggml_tensor **>> dac_snake_slots(parler_model & model) {
    std::vector<std::pair<ggml_tensor *, ggml_tensor **>> slots;
    for (parler_dac_block & b : model.dac_blocks) {
        slots.emplace_back(b.snake_alpha, &b.snake_inv);
        for (parler_dac_residual & r : b.res) {
            slots.emplace_back(r.snake1_alpha, &r.snake1_inv);
            slots.emplace_back(r.snake2_alpha, &r.snake2_inv);
        }
    }
    slots.emplace_back(model.dac_snake_out_alpha, &model.dac_snake_out_inv);
    return slots;
}

bool all_f32(const std::vector<std::pair<ggml_tensor *, ggml_tensor **>> & slots) {
    for (const auto & slot : slots) {
        if (slot.first->type != GGML_TYPE_F32 || !ggml_is_contiguous(slot.first)) return false;
    }
    return true;
}

void fill_snake_inverse(ggml_tensor * alpha, ggml_tensor * inv) {
    std::vector<float> v((size_t) ggml_nelements(alpha));
    ggml_backend_tensor_get(alpha, v.data(), 0, v.size() * sizeof(float));
    for (float & a : v) a = 1.0f / (a + PARLER_DAC_SNAKE_EPS);
    ggml_backend_tensor_set(inv, v.data(), 0, v.size() * sizeof(float));
}

void fill_snake_inverses(const std::vector<std::pair<ggml_tensor *, ggml_tensor **>> & slots) {
    for (const auto & slot : slots) fill_snake_inverse(slot.first, *slot.second);
}

void transpose_kernel_row(const float * src, int64_t rows, int64_t IC, float * dst) {
    for (int64_t r = 0; r < rows; ++r) dst[r * IC] = src[r];
}

bool dac_fuses_snakes(const parler_model & model) {
    return !model.on_gpu || model.on_hexagon;
}

bool convt_kernels_f32(const parler_model & model) {
    for (const parler_dac_block & b : model.dac_blocks) {
        if (b.convt_w->type != GGML_TYPE_F32 || !ggml_is_contiguous(b.convt_w)) return false;
    }
    return true;
}

void create_convt_cols_weights(parler_model & model) {
    for (parler_dac_block & b : model.dac_blocks) {
        const ggml_tensor * w = b.convt_w;
        b.convt_cols_w = ggml_new_tensor_2d(model.ctx_dac_derived, GGML_TYPE_F32, w->ne[2], w->ne[0] * w->ne[1]);
    }
}

void fill_convt_cols_weight(const parler_dac_block & b) {
    const ggml_tensor * w = b.convt_w;
    std::vector<float> kernel((size_t) ggml_nelements(w));
    std::vector<float> cols(kernel.size());
    ggml_backend_tensor_get(w, kernel.data(), 0, kernel.size() * sizeof(float));
    parler_dac_convt_cols_from_kernel(kernel.data(), w->ne[0], w->ne[1], w->ne[2], cols.data());
    ggml_backend_tensor_set(b.convt_cols_w, cols.data(), 0, cols.size() * sizeof(float));
}

void fill_convt_cols_weights(const parler_model & model) {
    for (const parler_dac_block & b : model.dac_blocks) fill_convt_cols_weight(b);
}

// The reciprocals the fused ggml_snake nodes consume (CPU and Hexagon) and the
// Hexagon columns-GEMM kernels.  A GGUF with non-F32 snake alphas keeps the
// elementwise chain; one with non-F32 transposed-conv kernels keeps the
// phase-GEMM lowering.
bool build_dac_derived(parler_model & model, parler_fit_measure * measure) {
    const auto slots = dac_snake_slots(model);
    const bool snakes = dac_fuses_snakes(model) && all_f32(slots);
    const bool cols   = model.on_hexagon && convt_kernels_f32(model);
    if (!snakes && !cols) return true;
    const size_t n_tensors = slots.size() + model.dac_blocks.size();
    ggml_init_params ip = { n_tensors * ggml_tensor_overhead(), nullptr, /*no_alloc=*/ true };
    model.ctx_dac_derived = ggml_init(ip);
    if (!model.ctx_dac_derived) return false;
    if (snakes) {
        for (const auto & slot : slots) *slot.second = ggml_dup_tensor(model.ctx_dac_derived, slot.first);
    }
    if (cols) create_convt_cols_weights(model);
    if (measure) {
        measure->dac_derived_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(
            model.ctx_dac_derived, ggml_backend_get_default_buffer_type(model.backend));
        mark_externally_allocated(model.ctx_dac_derived);
        return true;
    }
    model.buffer_dac_derived = ggml_backend_alloc_ctx_tensors(model.ctx_dac_derived, model.backend);
    if (!model.buffer_dac_derived) return false;
    if (snakes) fill_snake_inverses(slots);
    if (cols) fill_convt_cols_weights(model);
    return true;
}

const ggml_tensor * first_tensor_hexagon_cannot_run(const ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!parler_hexagon_runs_weight_type(t->type)) return t;
    }
    return nullptr;
}

ggml_backend_t init_policy_backend(int n_gpu_layers) {
    // Pixel 9-class Mali GPUs are validated through Vulkan for the complete
    // Parler pipeline, including T5, autoregressive decode, and DAC synthesis.
    // Parler's GPU path (FA + DAC phase-GEMM) is enabled only on
    // backends it has been validated against end-to-end (reference-fixture parity
    // per stage, plus greedy-token identity); anything else falls back to CPU.
    // Metal was validated in PR #103, Vulkan and OpenCL (Adreno) since, and CUDA
    // against the same reference fixtures on an RTX 3090. Feature-level
    // probes such as parler_probe_fa() still apply on top, so a validated backend
    // that lacks an individual capability degrades rather than breaking.
    // Filtering at the walk rather than after it matters on a host carrying an
    // unvalidated backend the walk would otherwise return: rejecting the winner
    // afterwards drops straight to CPU instead of yielding to a validated one.
    using ::tts_cpp::detail::GpuBackendRequirement;
    ggml_backend_t backend = ::tts_cpp::detail::init_gpu_backend(
        n_gpu_layers, /*verbose=*/false, "parler", /*vulkan_device=*/0,
        /*allow_arm_mali=*/true, /*out_gpu_present_but_unused=*/nullptr,
        GpuBackendRequirement::Metal | GpuBackendRequirement::Vulkan |
            GpuBackendRequirement::OpenCL | GpuBackendRequirement::CUDA);
    return backend ? backend : ::tts_cpp::detail::init_cpu_backend();
}

} // namespace

// Shared body of parler_load_gguf and parler_load_gguf_metadata_only.  When
// `measure` is non-null the load is metadata-only: every buffer the real path
// allocates is sized instead and no tensor data leaves the disk.
bool parler_load_gguf_impl(const std::string & path, parler_model & model,
                           int n_gpu_layers, std::string * error,
                           parler_fit_measure * measure, const std::string & requested_backend) {
    ggml_context * ctx_meta = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ true, /*.ctx=*/ &ctx_meta };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) {
        if (error) *error = "failed to open GGUF: " + path;
        return false;
    }

    auto fail = [&](const std::string & msg) {
        if (error) *error = msg;
        gguf_free(g);
        if (ctx_meta) ggml_free(ctx_meta);
        parler_free_model(model);
        return false;
    };

    if (gguf_find_key(g, "parler.arch") < 0) {
        return fail("not a parler GGUF (missing parler.arch)");
    }

    parler_hparams & hp = model.hparams;
    hp.variant.clear();
    const int64_t variant_id = gguf_find_key(g, "parler.variant");
    if (variant_id >= 0) {
        if (!kv_type_is(g, variant_id, GGUF_TYPE_STRING)) return fail("GGUF key is not string: parler.variant");
        hp.variant = gguf_get_val_str(g, variant_id);
    }
    std::string err;
    if (!kv_u32(g, "parler.t5.n_layer", hp.t5_n_layer, &err) ||
        !kv_u32(g, "parler.t5.d_model", hp.t5_d_model, &err) ||
        !kv_u32(g, "parler.t5.d_ff", hp.t5_d_ff, &err) ||
        !kv_u32(g, "parler.t5.n_head", hp.t5_n_head, &err) ||
        !kv_u32(g, "parler.t5.d_kv", hp.t5_d_kv, &err) ||
        !kv_u32(g, "parler.t5.rel_buckets", hp.t5_rel_buckets, &err) ||
        !kv_u32(g, "parler.t5.rel_max_dist", hp.t5_rel_max_dist, &err) ||
        !kv_f32(g, "parler.t5.rms_eps", hp.t5_rms_eps, &err) ||
        !kv_u32(g, "parler.t5.vocab_size", hp.t5_vocab, &err) ||
        !kv_u32(g, "parler.dec.n_layer", hp.dec_n_layer, &err) ||
        !kv_u32(g, "parler.dec.d_model", hp.dec_d_model, &err) ||
        !kv_u32(g, "parler.dec.n_head", hp.dec_n_head, &err) ||
        !kv_u32(g, "parler.dec.d_ff", hp.dec_d_ff, &err) ||
        !kv_u32(g, "parler.dec.n_codebooks", hp.n_codebooks, &err) ||
        !kv_u32(g, "parler.dec.vocab_size", hp.dec_vocab, &err) ||
        !kv_f32(g, "parler.dec.ln_eps", hp.dec_ln_eps, &err) ||
        !kv_u32(g, "parler.dec.bos_token_id", hp.bos_id, &err) ||
        !kv_u32(g, "parler.dec.eos_token_id", hp.eos_id, &err) ||
        !kv_u32(g, "parler.dec.pad_token_id", hp.pad_id, &err) ||
        !kv_u32(g, "parler.dec.decoder_start_token_id", hp.dec_start_id, &err) ||
        !kv_u32(g, "parler.dec.max_position", hp.max_position, &err) ||
        !kv_bool(g, "parler.enc_to_dec", hp.enc_to_dec, &err) ||
        !kv_u32(g, "parler.gen.max_length", hp.gen_max_length, &err) ||
        !kv_u32(g, "parler.gen.min_new_tokens", hp.gen_min_new_tokens, &err) ||
        !kv_bool(g, "parler.gen.do_sample", hp.gen_do_sample, &err) ||
        !kv_f32(g, "parler.gen.temperature", hp.gen_temperature, &err) ||
        !kv_u32(g, "parler.gen.top_k", hp.gen_top_k, &err) ||
        !kv_u32(g, "parler.dac.sample_rate", hp.dac_sample_rate, &err) ||
        !kv_u32(g, "parler.dac.n_quantizers", hp.dac_n_q, &err) ||
        !kv_u32(g, "parler.dac.codebook_size", hp.dac_codebook_size, &err) ||
        !kv_u32(g, "parler.dac.latent_dim", hp.dac_latent, &err) ||
        !kv_u32(g, "parler.dac.decoder_dim", hp.dac_decoder_dim, &err) ||
        !kv_u32(g, "parler.dac.hop", hp.dac_hop, &err)) {
        return fail(err);
    }
    {
        const int64_t id = gguf_find_key(g, "parler.dac.rates");
        if (id < 0) return fail("missing GGUF key: parler.dac.rates");
        const int n = (int) gguf_get_arr_n(g, id);
        hp.dac_rates.resize(n);
        for (int i = 0; i < n; ++i) {
            hp.dac_rates[i] = (int) ((const int32_t *) gguf_get_arr_data(g, id))[i];
        }
    }

    // decoder KV capacity: full generation budget plus prompt headroom,
    // capped at the sinusoidal table size.
    hp.n_ctx = std::min(hp.max_position, hp.gen_max_length + 640);

    // ---- tokenizer payload ----
    // A weightless fit-measure GGUF may omit the vocabulary: a measurement
    // never tokenizes text.
    if (!measure || gguf_find_key(g, "tokenizer.ggml.tokens") >= 0) {
        int64_t id = gguf_find_key(g, "tokenizer.ggml.tokens");
        if (id < 0) return fail("missing tokenizer.ggml.tokens");
        const int n = (int) gguf_get_arr_n(g, id);
        model.tok_pieces.resize(n);
        for (int i = 0; i < n; ++i) {
            model.tok_pieces[i] = gguf_get_arr_str(g, id, i);
        }
        id = gguf_find_key(g, "tokenizer.ggml.scores");
        if (id < 0) return fail("missing tokenizer.ggml.scores");
        if ((int) gguf_get_arr_n(g, id) != n) return fail("tokenizer scores/tokens length mismatch");
        model.tok_scores.resize(n);
        std::memcpy(model.tok_scores.data(), gguf_get_arr_data(g, id), n * sizeof(float));
        id = gguf_find_key(g, "tokenizer.ggml.precompiled_charsmap");
        if (id >= 0) {
            const size_t nb = gguf_get_arr_n(g, id);
            model.tok_charsmap.resize(nb);
            std::memcpy(model.tok_charsmap.data(), gguf_get_arr_data(g, id), nb);
        }
        int v = 0;
        if (kv_u32(g, "tokenizer.ggml.unknown_token_id", v, nullptr)) model.tok_unk_id = v;
        if (kv_u32(g, "tokenizer.ggml.eos_token_id", v, nullptr)) model.tok_eos_id = v;
        bool b = true;
        if (kv_bool(g, "parler.tokenizer.add_eos", b, nullptr)) model.tok_add_eos = b;
    }

    // ---- optional BPE prompt tokenizer (indic-class checkpoints; its vocab
    //      may be stripped from a weightless fit-measure GGUF too) ----
    if (gguf_find_key(g, "parler.prompt_tokenizer.model") >= 0 &&
        (!measure || gguf_find_key(g, "parler.prompt_tokenizer.tokens") >= 0)) {
        int64_t id = gguf_find_key(g, "parler.prompt_tokenizer.tokens");
        if (id < 0) return fail("missing parler.prompt_tokenizer.tokens");
        const int n = (int) gguf_get_arr_n(g, id);
        model.ptok_pieces.resize(n);
        for (int i = 0; i < n; ++i) {
            model.ptok_pieces[i] = gguf_get_arr_str(g, id, i);
        }
        id = gguf_find_key(g, "parler.prompt_tokenizer.merges");
        if (id < 0) return fail("missing parler.prompt_tokenizer.merges");
        const int nm = (int) gguf_get_arr_n(g, id);
        model.ptok_merges.resize(nm);
        for (int i = 0; i < nm; ++i) {
            model.ptok_merges[i] = gguf_get_arr_str(g, id, i);
        }
        int v = 0;
        if (kv_u32(g, "parler.prompt_tokenizer.unknown_token_id", v, nullptr)) model.ptok_unk_id = v;
        if (kv_u32(g, "parler.prompt_tokenizer.bos_token_id", v, nullptr)) model.ptok_bos_id = v;
        bool b = true;
        if (kv_bool(g, "parler.prompt_tokenizer.add_bos", b, nullptr)) model.ptok_add_bos = b;
        model.has_prompt_tok = true;
    }

    // ---- weights: dup metadata tensors into ctx_w, alloc, stream ----
    const int64_t n_tensors = gguf_get_n_tensors(g);
    {
        const size_t ctx_size = (size_t)(n_tensors + 8) * ggml_tensor_overhead();
        ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/ true };
        model.ctx_w = ggml_init(ip);
        if (!model.ctx_w) return fail("ggml_init(ctx_w) failed");
    }
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t; t = ggml_get_next_tensor(ctx_meta, t)) {
        ggml_tensor * d = ggml_dup_tensor(model.ctx_w, t);
        ggml_set_name(d, ggml_get_name(t));
    }

    ::tts_cpp::detail::ensure_backends_loaded();
    const bool explicit_backend = !::tts_cpp::detail::backend_request_is_auto(requested_backend);
    model.backend = explicit_backend
        ? ::tts_cpp::detail::init_requested_backend(requested_backend, /*verbose=*/false, "parler")
        : init_policy_backend(n_gpu_layers);
    if (!model.backend && explicit_backend) {
        return fail("requested backend '" + requested_backend + "' is not available");
    }
    if (!model.backend) return fail("failed to init backend");

    model.on_gpu     = !::tts_cpp::detail::backend_is_cpu(model.backend);
    model.on_hexagon = ::tts_cpp::detail::backend_is_hexagon(model.backend);
    if (model.on_hexagon) {
        if (const ggml_tensor * t = first_tensor_hexagon_cannot_run(model.ctx_w)) {
            return fail(std::string("Hexagon cannot run ") + ggml_get_name(t) + " (" + ggml_type_name(t->type) +
                        "); use a q8_0, f16 or f32 Parler GGUF");
        }
    }
    const ggml_type fa_kv_type = parler_fa_kv_type(model.on_gpu);
    model.use_fa  = parler_probe_fa(model.backend, hp.dec_d_model / hp.dec_n_head, hp.dec_n_head,
                                    fa_kv_type)
                    && std::getenv("PARLER_NO_FA") == nullptr;
    model.kv_type = model.use_fa ? fa_kv_type : GGML_TYPE_F32;

    // On the CPU backend, back each verbatim ctx_w tensor with the mmap'd GGUF
    // (bounds + 32B-align guarded, alloc+stream fallback) instead of a dirty
    // buffer + stream copy. GPU path unchanged (weights uploaded to a device
    // buffer); ctx_fused is GPU-only.
    // Measure mode bypasses the mmap: the projection prices the allocate-and-
    // stream fallback, which bounds the fully-touched mapping from above.
    const bool on_cpu = ::tts_cpp::detail::backend_is_cpu(model.backend);
    bool mapping = false;
    if (on_cpu && !measure && tts_cpp::cosyvoice::mapped_file_open(model.mapped, path, "parler")) {
        model.map_buf = ggml_backend_cpu_buffer_from_ptr(
            (void *) model.mapped.data, model.mapped.size);
        mapping = model.map_buf != nullptr;
        if (!mapping) tts_cpp::cosyvoice::mapped_file_close(model.mapped);
    }
    if (mapping) {
        const size_t data_off = gguf_get_data_offset(g);
        for (ggml_tensor * t = ggml_get_first_tensor(model.ctx_w); t;
             t = ggml_get_next_tensor(model.ctx_w, t)) {
            const int64_t gi = gguf_find_tensor(g, ggml_get_name(t));
            if (gi < 0) continue;
            const size_t off = data_off + (size_t) gguf_get_tensor_offset(g, gi);
            const size_t nb  = ggml_nbytes(t);
            if (off > model.mapped.size || nb > model.mapped.size - off) continue;
            const uint8_t * src = model.mapped.data + off;
            if (((uintptr_t) src % 32) != 0) continue;  // quant kernels need 32B alignment
            t->data   = (void *) src;
            t->buffer = model.map_buf;
        }
    }

    // Allocate + stream whatever stayed unmapped (all on GPU; none on a full CPU
    // map, where buffer_w stays null).  In measure mode the buffer is sized
    // instead of allocated and nothing is streamed.
    bool any_unmapped = false;
    for (ggml_tensor * t = ggml_get_first_tensor(model.ctx_w); t;
         t = ggml_get_next_tensor(model.ctx_w, t)) {
        if (t->data == nullptr) { any_unmapped = true; break; }
    }
    ggml_backend_buffer_type_t weight_buft = ::tts_cpp::detail::weight_buffer_type(model.backend);
    if (measure) {
        measure->weights_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(model.ctx_w, weight_buft);
        mark_externally_allocated(model.ctx_w);
    } else if (any_unmapped) {
        model.buffer_w = ggml_backend_alloc_ctx_tensors_from_buft(model.ctx_w, weight_buft);
        if (!model.buffer_w) return fail("failed to allocate weight buffer");

        ::tts_cpp::detail::gguf_stream_reader rd(g, path);
        if (!rd.ok()) return fail("failed to reopen GGUF for streaming");
        for (ggml_tensor * t = ggml_get_first_tensor(model.ctx_w); t;
             t = ggml_get_next_tensor(model.ctx_w, t)) {
            if (mapping && t->buffer == model.map_buf) continue;
            if (!rd.to_backend(ggml_get_name(t), t)) {
                return fail(std::string("failed to stream tensor ") + ggml_get_name(t));
            }
        }
    }

    // ---- wire named tensors into the typed structs ----
    auto get = [&](const std::string & name, bool required = true) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(model.ctx_w, name.c_str());
        if (!t && required) {
            fprintf(stderr, "parler: missing tensor %s\n", name.c_str());
        }
        return t;
    };
    bool ok = true;
    auto req = [&](const std::string & name) {
        ggml_tensor * t = get(name);
        if (!t) ok = false;
        return t;
    };

    model.t5_embed       = req("t5.embed_tokens.weight");
    model.t5_rel_b       = req("t5.blk.0.attn_rel_b.weight");
    model.t5_output_norm = req("t5.output_norm.weight");
    model.t5_layers.resize(hp.t5_n_layer);
    for (int i = 0; i < hp.t5_n_layer; ++i) {
        auto & l = model.t5_layers[i];
        const std::string p = "t5.blk." + std::to_string(i) + ".";
        l.attn_norm = req(p + "attn_norm.weight");
        l.q = req(p + "attn_q.weight"); l.k = req(p + "attn_k.weight");
        l.v = req(p + "attn_v.weight"); l.o = req(p + "attn_o.weight");
        l.ffn_norm = req(p + "ffn_norm.weight");
        l.gate = req(p + "ffn_gate.weight");
        l.up   = req(p + "ffn_up.weight");
        l.down = req(p + "ffn_down.weight");
    }

    if (hp.enc_to_dec) {
        model.enc_to_dec_w = req("enc_to_dec.weight");
        model.enc_to_dec_b = req("enc_to_dec.bias");
    }
    model.embed_prompts   = req("dec.embed_prompts.weight");
    model.embed_positions = req("dec.embed_positions.weight");

    model.dec_embed.resize(hp.n_codebooks);
    model.lm_heads.resize(hp.n_codebooks);
    for (int k = 0; k < hp.n_codebooks; ++k) {
        model.dec_embed[k] = req("dec.embed_tokens." + std::to_string(k) + ".weight");
        model.lm_heads[k]  = req("dec.lm_heads." + std::to_string(k) + ".weight");
    }
    model.dec_output_norm_w = req("dec.output_norm.weight");
    model.dec_output_norm_b = req("dec.output_norm.bias");
    model.dec_layers.resize(hp.dec_n_layer);
    for (int i = 0; i < hp.dec_n_layer; ++i) {
        auto & l = model.dec_layers[i];
        const std::string p = "dec.blk." + std::to_string(i) + ".";
        l.attn_norm_w = req(p + "attn_norm.weight");
        l.attn_norm_b = req(p + "attn_norm.bias");
        l.q = req(p + "attn_q.weight"); l.k = req(p + "attn_k.weight");
        l.v = req(p + "attn_v.weight"); l.o = req(p + "attn_o.weight");
        l.cross_norm_w = req(p + "cross_norm.weight");
        l.cross_norm_b = req(p + "cross_norm.bias");
        l.cq = req(p + "cross_q.weight"); l.ck = req(p + "cross_k.weight");
        l.cv = req(p + "cross_v.weight"); l.co = req(p + "cross_o.weight");
        l.ffn_norm_w = req(p + "ffn_norm.weight");
        l.ffn_norm_b = req(p + "ffn_norm.bias");
        l.up   = req(p + "ffn_up.weight");
        l.down = req(p + "ffn_down.weight");
    }

    model.dac_quant.resize(hp.dac_n_q);
    for (int k = 0; k < hp.dac_n_q; ++k) {
        auto & q = model.dac_quant[k];
        const std::string p = "dac.quant." + std::to_string(k) + ".";
        q.codebook   = req(p + "codebook.weight");
        q.out_proj_w = req(p + "out_proj.weight");
        q.out_proj_b = req(p + "out_proj.bias");
    }
    model.dac_conv_in_w = req("dac.dec.conv_in.weight");
    model.dac_conv_in_b = req("dac.dec.conv_in.bias");
    model.dac_blocks.resize(hp.dac_rates.size());
    for (size_t i = 0; i < hp.dac_rates.size(); ++i) {
        auto & b = model.dac_blocks[i];
        const std::string p = "dac.dec.blk." + std::to_string(i) + ".";
        b.stride      = hp.dac_rates[i];
        b.snake_alpha = req(p + "snake.alpha");
        b.convt_w     = req(p + "convt.weight");
        b.convt_b     = req(p + "convt.bias");
        for (int j = 0; j < 3; ++j) {
            auto & r = b.res[j];
            const std::string rp = p + "res." + std::to_string(j) + ".";
            r.snake1_alpha = req(rp + "snake1.alpha");
            r.conv1_w = req(rp + "conv1.weight"); r.conv1_b = req(rp + "conv1.bias");
            r.snake2_alpha = req(rp + "snake2.alpha");
            r.conv2_w = req(rp + "conv2.weight"); r.conv2_b = req(rp + "conv2.bias");
        }
    }
    model.dac_snake_out_alpha = req("dac.dec.snake_out.alpha");
    model.dac_conv_out_w = req("dac.dec.conv_out.weight");
    model.dac_conv_out_b = req("dac.dec.conv_out.bias");

    if (!ok) return fail("one or more expected tensors missing (see log)");

    // shape spot-checks against hparams (catches converter/loader drift)
    if (model.t5_embed->ne[0] != hp.t5_d_model || model.t5_embed->ne[1] != hp.t5_vocab) {
        return fail("t5.embed_tokens shape mismatch");
    }
    if (model.lm_heads[0]->ne[0] != hp.dec_d_model || model.lm_heads[0]->ne[1] != hp.dec_vocab) {
        return fail("dec.lm_heads shape mismatch");
    }
    if (model.dec_embed[0]->ne[0] != hp.dec_d_model || model.dec_embed[0]->ne[1] < hp.dec_vocab) {
        return fail("dec.embed_tokens shape mismatch");
    }
    if (model.embed_positions->ne[0] != hp.dec_d_model ||
        model.embed_positions->ne[1] != hp.max_position) {
        return fail("dec.embed_positions shape mismatch");
    }
    if (model.embed_prompts->ne[0] != hp.dec_d_model ||
        model.embed_prompts->ne[1] < (int64_t) (model.has_prompt_tok ? model.ptok_pieces.size()
                                                                     : model.tok_pieces.size())) {
        return fail("dec.embed_prompts smaller than the prompt tokenizer vocab");
    }

    // ---- decoder self-KV cache slabs ----
    {
        ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/ true };
        model.ctx_kv = ggml_init(ip);
        if (!model.ctx_kv) return fail("ggml_init(ctx_kv) failed");
        const int64_t rows = (int64_t) hp.n_ctx * hp.dec_n_layer;
        model.memory_k = ggml_new_tensor_2d(model.ctx_kv, model.kv_type, hp.dec_d_model, rows);
        model.memory_v = ggml_new_tensor_2d(model.ctx_kv, model.kv_type, hp.dec_d_model, rows);
        ggml_set_name(model.memory_k, "parler_kv_k");
        ggml_set_name(model.memory_v, "parler_kv_v");
        if (measure) {
            measure->kv_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(
                model.ctx_kv, ggml_backend_get_default_buffer_type(model.backend));
            mark_externally_allocated(model.ctx_kv);
        } else {
            model.buffer_kv = ggml_backend_alloc_ctx_tensors(model.ctx_kv, model.backend);
            if (!model.buffer_kv) return fail("failed to allocate KV buffer");
        }
    }

    if (parler_should_fuse_decode_weights(model, measure != nullptr) &&
        !parler_fuse_decode_weights(model, measure)) {
        return fail("failed to fuse decode projection weights");
    }
    if (!build_dac_derived(model, measure)) {
        return fail("failed to build the DAC snake reciprocals and transposed-conv weights");
    }

    gguf_free(g);
    ggml_free(ctx_meta);
    return true;
}

bool parler_load_gguf(const std::string & path, parler_model & model,
                      int n_gpu_layers, std::string * error, const std::string & backend) {
    return parler_load_gguf_impl(path, model, n_gpu_layers, error, /*measure=*/nullptr, backend);
}

bool parler_load_gguf_metadata_only(const std::string & path, parler_model & model,
                                    int n_gpu_layers, parler_fit_measure & measure,
                                    std::string * error, const std::string & backend) {
    measure = parler_fit_measure{};
    return parler_load_gguf_impl(path, model, n_gpu_layers, error, &measure, backend);
}

void parler_free_model(parler_model & model) {
    ::tts_cpp::detail::sched_fallback_free(model.sched_fb);
    if (model.dac_allocr)   { ggml_gallocr_free(model.dac_allocr); model.dac_allocr = nullptr; }
    model.dac_sidecar.reset();
    model.dac_sidecar_retired = false;
    model.dac_rvq = {};
    if (model.buffer_dac_derived) {
        ggml_backend_buffer_free(model.buffer_dac_derived);
        model.buffer_dac_derived = nullptr;
    }
    if (model.ctx_dac_derived) { ggml_free(model.ctx_dac_derived); model.ctx_dac_derived = nullptr; }
    if (model.buffer_cross) { ggml_backend_buffer_free(model.buffer_cross); model.buffer_cross = nullptr; }
    if (model.ctx_cross)    { ggml_free(model.ctx_cross); model.ctx_cross = nullptr; }
    if (model.buffer_fused) { ggml_backend_buffer_free(model.buffer_fused); model.buffer_fused = nullptr; }
    if (model.ctx_fused)    { ggml_free(model.ctx_fused); model.ctx_fused = nullptr; }
    if (model.buffer_kv)    { ggml_backend_buffer_free(model.buffer_kv); model.buffer_kv = nullptr; }
    if (model.ctx_kv)       { ggml_free(model.ctx_kv); model.ctx_kv = nullptr; }
    if (model.buffer_w)     { ggml_backend_buffer_free(model.buffer_w); model.buffer_w = nullptr; }
    if (model.ctx_w)        { ggml_free(model.ctx_w); model.ctx_w = nullptr; }
    // Free the wrapper (not the mapping) after ctx_w, then munmap.
    if (model.map_buf)      { ggml_backend_buffer_free(model.map_buf); model.map_buf = nullptr; }
    tts_cpp::cosyvoice::mapped_file_close(model.mapped);
    if (model.backend)      { ggml_backend_free(model.backend); model.backend = nullptr; }
    model.cross_k.clear();
    model.cross_v_t.clear();
    model.cross_len = 0;
}

static bool qkv_projections_fusable(const parler_model & model) {
    const int64_t d = model.hparams.dec_d_model;
    for (const auto & l : model.dec_layers) {
        if (l.q->type != l.k->type || l.q->type != l.v->type ||
            l.q->ne[0] != d || l.q->ne[1] != d ||
            l.k->ne[0] != d || l.k->ne[1] != d ||
            l.v->ne[0] != d || l.v->ne[1] != d) return false;
    }
    return true;
}

static bool lm_heads_fusable(const parler_model & model) {
    const int64_t d = model.hparams.dec_d_model;
    const int64_t vocab = model.lm_heads[0]->ne[1];
    for (const ggml_tensor * h : model.lm_heads) {
        if (h->type != model.lm_heads[0]->type ||
            h->ne[0] != d || h->ne[1] != vocab) return false;
    }
    return true;
}

static void create_fused_tensors(parler_model & model, bool fuse_qkv, bool fuse_heads,
                                 std::vector<ggml_tensor *> & qkv, ggml_tensor *& heads) {
    const parler_hparams & hp = model.hparams;
    const int64_t d = hp.dec_d_model;
    qkv.assign((size_t) hp.dec_n_layer, nullptr);
    if (fuse_qkv) {
        for (int i = 0; i < hp.dec_n_layer; ++i) {
            qkv[i] = ggml_new_tensor_2d(model.ctx_fused, model.dec_layers[i].q->type, d, 3 * d);
        }
    }
    heads = fuse_heads
        ? ggml_new_tensor_2d(model.ctx_fused, model.lm_heads[0]->type, d,
                             model.lm_heads[0]->ne[1] * hp.n_codebooks)
        : nullptr;
}

static void size_fused_buffer(parler_model & model, parler_fit_measure & measure) {
    measure.fused_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(
        model.ctx_fused, ::tts_cpp::detail::weight_buffer_type(model.backend));
    mark_externally_allocated(model.ctx_fused);
}

// Assemble the row concat in host memory and upload it in one whole-tensor
// call: backends that repack quantized weights on set_tensor (OpenCL's
// struct-of-arrays) rebuild from the entire tensor and ignore a byte window.
static void concat_weight_rows(ggml_tensor * dst, const ggml_tensor * const * parts,
                               int n, std::vector<uint8_t> & tmp) {
    tmp.resize(ggml_nbytes(dst));
    size_t off = 0;
    for (int p = 0; p < n; ++p) {
        const size_t nb = ggml_nbytes(parts[p]);
        ggml_backend_tensor_get(parts[p], tmp.data() + off, 0, nb);
        off += nb;
    }
    GGML_ASSERT(off == tmp.size() && "fused weight size mismatch");
    ggml_backend_tensor_set(dst, tmp.data(), 0, tmp.size());
}

static void fill_fused_tensors(parler_model & model,
                               const std::vector<ggml_tensor *> & qkv, ggml_tensor * heads) {
    std::vector<uint8_t> tmp;
    for (size_t i = 0; i < qkv.size(); ++i) {
        if (!qkv[i]) continue;
        const parler_dec_layer & l = model.dec_layers[i];
        const ggml_tensor * parts[3] = { l.q, l.k, l.v };
        concat_weight_rows(qkv[i], parts, 3, tmp);
    }
    if (heads) {
        concat_weight_rows(heads, model.lm_heads.data(), (int) model.lm_heads.size(), tmp);
    }
}

static void wire_fused_pointers(parler_model & model,
                                const std::vector<ggml_tensor *> & qkv, ggml_tensor * heads) {
    for (size_t i = 0; i < qkv.size(); ++i) {
        if (qkv[i]) model.dec_layers[i].qkv = qkv[i];
    }
    if (heads) model.lm_head_stacked = heads;
}

void parler_dac_convt_cols_from_kernel(const float * kernel, int64_t K, int64_t OC,
                                       int64_t IC, float * cols) {
    const int64_t rows = K * OC;
    for (int64_t ic = 0; ic < IC; ++ic) {
        transpose_kernel_row(kernel + ic * rows, rows, IC, cols + ic);
    }
}

bool parler_hexagon_runs_weight_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0;
}

bool parler_dac_uses_fused_snake(const parler_model & model) {
    return model.dac_snake_out_inv != nullptr;
}

ggml_type parler_fa_kv_type(bool on_gpu) {
    return on_gpu ? GGML_TYPE_F16 : GGML_TYPE_F32;
}

bool parler_should_fuse_decode_weights(const parler_model & model, bool measuring) {
    if (std::getenv("PARLER_NO_FUSED") != nullptr) return false;
    // The CPU allocate-and-stream fallback would hold the projections twice in
    // anonymous memory. mmap'd originals are file-backed and evictable once the
    // fused copies exist, and GPU loads always uploaded a device copy, so those
    // paths fuse; measure mode prices the mapped path it stands in for.
    return model.on_gpu || model.map_buf != nullptr || measuring;
}

bool parler_fuse_decode_weights(parler_model & model, parler_fit_measure * measure) {
    const parler_hparams & hp = model.hparams;
    if (hp.dec_n_layer <= 0 || hp.n_codebooks <= 0 ||
        model.lm_heads.empty() || !model.lm_heads[0]) {
        return true;
    }
    const bool fuse_qkv   = qkv_projections_fusable(model);
    const bool fuse_heads = lm_heads_fusable(model);
    // Skip the fused context entirely when nothing fuses: an empty ctx makes
    // ggml_backend_alloc_ctx_tensors return NULL (0 buffers) and abort the load.
    if (!fuse_qkv && !fuse_heads) return true;

    ggml_init_params ip = { (size_t) (hp.dec_n_layer + 2) * ggml_tensor_overhead(),
                            nullptr, /*no_alloc=*/ true };
    model.ctx_fused = ggml_init(ip);
    if (!model.ctx_fused) return false;
    std::vector<ggml_tensor *> qkv;
    ggml_tensor * heads = nullptr;
    create_fused_tensors(model, fuse_qkv, fuse_heads, qkv, heads);
    if (measure) {
        size_fused_buffer(model, *measure);
        wire_fused_pointers(model, qkv, heads);
        return true;
    }
    model.buffer_fused = ggml_backend_alloc_ctx_tensors_from_buft(
        model.ctx_fused, ::tts_cpp::detail::weight_buffer_type(model.backend));
    if (!model.buffer_fused) return false;
    fill_fused_tensors(model, qkv, heads);
    wire_fused_pointers(model, qkv, heads);
    return true;
}

bool parler_graph_prepare(const parler_model & model, ggml_cgraph * gf,
                          ggml_gallocr_t allocr, bool & use_sched, const char * caller) {
    use_sched = ::tts_cpp::detail::sched_force_enabled() ||
                !::tts_cpp::detail::graph_fully_supported(model.backend, gf);
    if (!use_sched) {
        if (!ggml_gallocr_alloc_graph(allocr, gf)) {
            fprintf(stderr, "%s: gallocr alloc failed\n", caller);
            return false;
        }
        return true;
    }
    if (::tts_cpp::detail::graph_has_unsupported_preallocated_op(model.backend, gf)) {
        fprintf(stderr, "%s: op writing a persistent buffer is unsupported; "
                        "sched fallback impossible\n", caller);
        return false;
    }
    if (!::tts_cpp::detail::sched_fallback_ensure(model.sched_fb, model.backend,
            /*graph_size=*/ 2 * PARLER_MAX_NODES,
            {model.buffer_w, model.buffer_kv, model.buffer_cross})) {
        fprintf(stderr, "%s: scheduler creation failed\n", caller);
        return false;
    }
    if (!::tts_cpp::detail::sched_fallback_alloc(model.sched_fb, gf)) {
        fprintf(stderr, "%s: sched graph alloc failed\n", caller);
        return false;
    }
    return true;
}

bool parler_graph_compute(const parler_model & model, ggml_cgraph * gf,
                          bool use_sched, int n_threads, const char * caller) {
    const ggml_status status = use_sched
        ? ::tts_cpp::detail::sched_fallback_compute(model.sched_fb, model.backend, gf, n_threads)
        : ::tts_cpp::detail::direct_compute(model.backend, gf, n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed (ggml_status=%d)\n", caller, (int) status);
        return false;
    }
    return true;
}

} // namespace detail
} // namespace parler
} // namespace tts_cpp
