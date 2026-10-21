// Fit-projection parity tests (include/tts-cpp/parler/fit.h): assert that the
// metadata-only memory projection matches what a REAL load and REAL runs
// actually allocate, byte for byte where the projection is exact by
// construction.
//
// Two modes:
//
//   * No arguments (the always-on CI form): a tiny but complete parler GGUF is
//     synthesized on the fly (same trick as test/audio8/test_fit_params.cpp),
//     so the parity gates run with no model fixture at all:
//       1. projected weight bytes == a real allocation of the same tensor set
//          (the real CPU load maps the file in place, so the anchor is a
//          genuine ggml_backend_alloc_ctx_tensors of identical headers);
//       2. projected KV-slab bytes == the real load's KV buffer;
//       3. projected cross-K/V bytes == the buffer a real description encode
//          allocates at the same token count;
//       4. projected decoder arena (max of prefill and step) == the gallocr
//          buffer real parler_dec_prefill + parler_dec_step calls reserve;
//       5. projected T5 arena == a real gallocr reservation of the same graph;
//       6. projected DAC window arena == the arena a real parler_dac_decode
//          leaves in model.dac_allocr;
//       7. errors and oversized workloads surface as Error, never Success;
//       8. a weightless description GGUF with the vocabulary stripped loads
//          metadata-only and projects, while a real load still refuses it.
//
//   * With arguments <parler.gguf> [n_gpu_layers]: the same gates on a real
//     fixture (plus the GPU fused-weight stack when n_gpu_layers > 0).
//
// Exit 0 on success; non-zero with a FAIL line per broken invariant.

#include "tts-cpp/parler/fit.h"

#include "fit_price.h"
#include "parler/internal.h"
#include "../test_env_portable.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace tts_cpp::parler::detail;

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

void expect_eq(uint64_t projected, uint64_t real, const std::string & what) {
    if (projected != real) {
        fail(what + ": projected " + std::to_string(projected) +
             " != allocated " + std::to_string(real));
    }
}

// ── Tiny synthetic parler GGUF ──────────────────────────────────────────────
// The smallest hparam set the loader, the graph builders, and the shape
// checks accept: every dimension real models scale up, none of the structure
// changed (1 T5 layer, 2 decoder layers, a 2-block DAC).

struct tiny_parler {
    // t5
    int t5_n_layer = 1, t5_d_model = 32, t5_d_ff = 48, t5_n_head = 2, t5_d_kv = 8;
    int t5_rel_buckets = 8, t5_rel_max_dist = 16, t5_vocab = 64;
    // decoder
    int dec_n_layer = 2, dec_d_model = 32, dec_n_head = 4, dec_d_ff = 48;
    int n_codebooks = 3, dec_vocab = 44;
    int bos_id = 41, eos_id = 40, pad_id = 40, dec_start_id = 41;
    int max_position = 96, gen_max_length = 24, gen_min_new_tokens = 2;
    // dac (hop == product of rates)
    int dac_sample_rate = 44100, dac_n_q = 3, dac_codebook_size = 24;
    int dac_latent = 8, dac_decoder_dim = 16, dac_hop = 4;
    int dac_rates[2] = { 2, 2 };
    int dac_cb_dim = 4;
};

void add_f32(gguf_context * g, ggml_context * ctx, const std::string & name,
             std::initializer_list<int64_t> ne) {
    ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, (int) ne.size(),
                                      std::vector<int64_t>(ne).data());
    ggml_set_name(t, name.c_str());
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = 0.01f;
    gguf_add_tensor(g, t);
}

std::string write_tiny_parler_gguf(const tiny_parler & p, bool with_vocab = true,
                                   bool only_meta = false) {
    const std::string path =
        (fs::temp_directory_path() /
         (with_vocab ? "test-parler-fit-tiny.gguf" : "test-parler-fit-tiny-desc.gguf"))
            .string();
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "parler.arch", "parler");
    auto u32 = [&](const char * k, int v) { gguf_set_val_u32(g, k, (uint32_t) v); };
    auto f32 = [&](const char * k, float v) { gguf_set_val_f32(g, k, v); };
    auto bl  = [&](const char * k, bool v) { gguf_set_val_bool(g, k, v); };
    u32("parler.t5.n_layer", p.t5_n_layer);
    u32("parler.t5.d_model", p.t5_d_model);
    u32("parler.t5.d_ff", p.t5_d_ff);
    u32("parler.t5.n_head", p.t5_n_head);
    u32("parler.t5.d_kv", p.t5_d_kv);
    u32("parler.t5.rel_buckets", p.t5_rel_buckets);
    u32("parler.t5.rel_max_dist", p.t5_rel_max_dist);
    f32("parler.t5.rms_eps", 1e-6f);
    u32("parler.t5.vocab_size", p.t5_vocab);
    u32("parler.dec.n_layer", p.dec_n_layer);
    u32("parler.dec.d_model", p.dec_d_model);
    u32("parler.dec.n_head", p.dec_n_head);
    u32("parler.dec.d_ff", p.dec_d_ff);
    u32("parler.dec.n_codebooks", p.n_codebooks);
    u32("parler.dec.vocab_size", p.dec_vocab);
    f32("parler.dec.ln_eps", 1e-5f);
    u32("parler.dec.bos_token_id", p.bos_id);
    u32("parler.dec.eos_token_id", p.eos_id);
    u32("parler.dec.pad_token_id", p.pad_id);
    u32("parler.dec.decoder_start_token_id", p.dec_start_id);
    u32("parler.dec.max_position", p.max_position);
    bl("parler.enc_to_dec", false);
    u32("parler.gen.max_length", p.gen_max_length);
    u32("parler.gen.min_new_tokens", p.gen_min_new_tokens);
    bl("parler.gen.do_sample", true);
    f32("parler.gen.temperature", 1.0f);
    u32("parler.gen.top_k", 10);
    u32("parler.dac.sample_rate", p.dac_sample_rate);
    u32("parler.dac.n_quantizers", p.dac_n_q);
    u32("parler.dac.codebook_size", p.dac_codebook_size);
    u32("parler.dac.latent_dim", p.dac_latent);
    u32("parler.dac.decoder_dim", p.dac_decoder_dim);
    u32("parler.dac.hop", p.dac_hop);
    const int32_t rates[2] = { p.dac_rates[0], p.dac_rates[1] };
    gguf_set_arr_data(g, "parler.dac.rates", GGUF_TYPE_INT32, rates, 2);

    if (with_vocab) {
        const char * toks[] = { "<unk>", "</s>", "\xe2\x96\x81" "a", "\xe2\x96\x81" "b",
                                "a", "b", "c", "d" };
        gguf_set_arr_str(g, "tokenizer.ggml.tokens", toks, 8);
        const float scores[8] = { 0, 0, -1, -1, -2, -2, -2, -2 };
        gguf_set_arr_data(g, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores, 8);
        gguf_set_val_u32(g, "tokenizer.ggml.unknown_token_id", 0);
        gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", 1);
    }

    ggml_init_params ip = { 16u * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context * ctx = ggml_init(ip);

    const int inner = p.t5_n_head * p.t5_d_kv;
    add_f32(g, ctx, "t5.embed_tokens.weight", { p.t5_d_model, p.t5_vocab });
    add_f32(g, ctx, "t5.output_norm.weight", { p.t5_d_model });
    for (int i = 0; i < p.t5_n_layer; ++i) {
        const std::string pre = "t5.blk." + std::to_string(i) + ".";
        if (i == 0) add_f32(g, ctx, pre + "attn_rel_b.weight", { p.t5_n_head, p.t5_rel_buckets });
        add_f32(g, ctx, pre + "attn_norm.weight", { p.t5_d_model });
        add_f32(g, ctx, pre + "attn_q.weight", { p.t5_d_model, inner });
        add_f32(g, ctx, pre + "attn_k.weight", { p.t5_d_model, inner });
        add_f32(g, ctx, pre + "attn_v.weight", { p.t5_d_model, inner });
        add_f32(g, ctx, pre + "attn_o.weight", { inner, p.t5_d_model });
        add_f32(g, ctx, pre + "ffn_norm.weight", { p.t5_d_model });
        add_f32(g, ctx, pre + "ffn_gate.weight", { p.t5_d_model, p.t5_d_ff });
        add_f32(g, ctx, pre + "ffn_up.weight", { p.t5_d_model, p.t5_d_ff });
        add_f32(g, ctx, pre + "ffn_down.weight", { p.t5_d_ff, p.t5_d_model });
    }

    const int d = p.dec_d_model;
    add_f32(g, ctx, "dec.embed_prompts.weight", { d, 8 });  // >= tokenizer vocab
    add_f32(g, ctx, "dec.embed_positions.weight", { d, p.max_position });
    for (int k = 0; k < p.n_codebooks; ++k) {
        add_f32(g, ctx, "dec.embed_tokens." + std::to_string(k) + ".weight",
                { d, p.dec_vocab });
        add_f32(g, ctx, "dec.lm_heads." + std::to_string(k) + ".weight",
                { d, p.dec_vocab });
    }
    add_f32(g, ctx, "dec.output_norm.weight", { d });
    add_f32(g, ctx, "dec.output_norm.bias", { d });
    for (int i = 0; i < p.dec_n_layer; ++i) {
        const std::string pre = "dec.blk." + std::to_string(i) + ".";
        add_f32(g, ctx, pre + "attn_norm.weight", { d });
        add_f32(g, ctx, pre + "attn_norm.bias", { d });
        add_f32(g, ctx, pre + "attn_q.weight", { d, d });
        add_f32(g, ctx, pre + "attn_k.weight", { d, d });
        add_f32(g, ctx, pre + "attn_v.weight", { d, d });
        add_f32(g, ctx, pre + "attn_o.weight", { d, d });
        add_f32(g, ctx, pre + "cross_norm.weight", { d });
        add_f32(g, ctx, pre + "cross_norm.bias", { d });
        add_f32(g, ctx, pre + "cross_q.weight", { d, d });
        add_f32(g, ctx, pre + "cross_k.weight", { d, d });
        add_f32(g, ctx, pre + "cross_v.weight", { d, d });
        add_f32(g, ctx, pre + "cross_o.weight", { d, d });
        add_f32(g, ctx, pre + "ffn_norm.weight", { d });
        add_f32(g, ctx, pre + "ffn_norm.bias", { d });
        add_f32(g, ctx, pre + "ffn_up.weight", { d, p.dec_d_ff });
        add_f32(g, ctx, pre + "ffn_down.weight", { p.dec_d_ff, d });
    }

    for (int k = 0; k < p.dac_n_q; ++k) {
        const std::string pre = "dac.quant." + std::to_string(k) + ".";
        add_f32(g, ctx, pre + "codebook.weight", { p.dac_cb_dim, p.dac_codebook_size });
        add_f32(g, ctx, pre + "out_proj.weight", { 1, p.dac_cb_dim, p.dac_latent });
        add_f32(g, ctx, pre + "out_proj.bias", { p.dac_latent });
    }
    // channel plan: conv_in -> C0 = decoder_dim, each block halves it
    const int C0 = p.dac_decoder_dim;
    add_f32(g, ctx, "dac.dec.conv_in.weight", { 7, p.dac_latent, C0 });
    add_f32(g, ctx, "dac.dec.conv_in.bias", { C0 });
    int C = C0;
    for (int i = 0; i < 2; ++i) {
        const std::string pre = "dac.dec.blk." + std::to_string(i) + ".";
        const int s  = p.dac_rates[i];
        const int OC = C / 2;
        add_f32(g, ctx, pre + "snake.alpha", { 1, C, 1 });
        add_f32(g, ctx, pre + "convt.weight", { 2 * s, OC, C });
        add_f32(g, ctx, pre + "convt.bias", { OC });
        for (int j = 0; j < 3; ++j) {
            const std::string rp = pre + "res." + std::to_string(j) + ".";
            add_f32(g, ctx, rp + "snake1.alpha", { 1, OC, 1 });
            add_f32(g, ctx, rp + "conv1.weight", { 7, OC, OC });
            add_f32(g, ctx, rp + "conv1.bias", { OC });
            add_f32(g, ctx, rp + "snake2.alpha", { 1, OC, 1 });
            add_f32(g, ctx, rp + "conv2.weight", { 1, OC, OC });
            add_f32(g, ctx, rp + "conv2.bias", { OC });
        }
        C = OC;
    }
    add_f32(g, ctx, "dac.dec.snake_out.alpha", { 1, C, 1 });
    add_f32(g, ctx, "dac.dec.conv_out.weight", { 7, C, 1 });
    add_f32(g, ctx, "dac.dec.conv_out.bias", { 1 });

    if (!gguf_write_to_file(g, path.c_str(), only_meta)) {
        std::fprintf(stderr, "FATAL: cannot write %s\n", path.c_str());
        std::exit(2);
    }
    ggml_free(ctx);
    gguf_free(g);
    return path;
}

// Price a freshly built graph the way the projector does (fit_price.h mirrors
// parler_graph_prepare's dispatch).
bool price(const parler_model & m, ggml_cgraph * gf,
           ::tts_cpp::detail::fit_graph_price & out) {
    return gf && ::tts_cpp::detail::fit_price_graph(m.backend, gf,
                                                    2 * PARLER_MAX_NODES, out);
}

void run_parity_gates(const std::string & path, int n_gpu_layers) {
    // Metadata-only vs real load on the same backend.
    parler_model mm, real;
    parler_fit_measure fm;
    std::string error;
    if (!parler_load_gguf_metadata_only(path, mm, n_gpu_layers, fm, &error)) {
        fail("parler_load_gguf_metadata_only failed: " + error);
        return;
    }
    if (!parler_load_gguf(path, real, n_gpu_layers, &error)) {
        fail("real parler_load_gguf failed: " + error);
        parler_free_model(mm);
        return;
    }

    // 1. Weight parity.  A real CPU load maps the GGUF in place (buffer_w
    //    null), so anchor the sizing against a genuine allocation of the
    //    identical tensor headers; on GPU the real buffer itself is the
    //    anchor.  The fused stack (GPU only) is byte-compared directly.
    if (real.buffer_w) {
        expect_eq(fm.weights_bytes, ggml_backend_buffer_get_size(real.buffer_w),
                  "weights parity (real buffer)");
    } else {
        // duplicate the real model's weight tensor headers and allocate them
        // for real -- the strongest anchor available on the mmap path
        size_t n = 8;
        for (ggml_tensor * t = ggml_get_first_tensor(real.ctx_w); t;
             t = ggml_get_next_tensor(real.ctx_w, t)) {
            ++n;
        }
        ggml_init_params wp = { n * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };
        ggml_context * dup = ggml_init(wp);
        for (ggml_tensor * t = ggml_get_first_tensor(real.ctx_w); t;
             t = ggml_get_next_tensor(real.ctx_w, t)) {
            ggml_set_name(ggml_dup_tensor(dup, t), ggml_get_name(t));
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(dup, real.backend);
        if (!buf) {
            fail("anchor allocation for weight parity failed");
        } else {
            expect_eq(fm.weights_bytes, ggml_backend_buffer_get_size(buf),
                      "weights parity (mmap path, anchored allocation)");
            ggml_backend_buffer_free(buf);
        }
        ggml_free(dup);
    }
    if (real.buffer_fused) {
        expect_eq(fm.fused_bytes, ggml_backend_buffer_get_size(real.buffer_fused),
                  "fused-weight parity");
        expect(mm.lm_head_stacked != nullptr || mm.dec_layers[0].qkv != nullptr,
               "measure-mode load did not wire the fused path the real load took");
    } else {
        expect_eq(fm.fused_bytes, 0, "fused bytes on a path with no fused buffer");
    }

    // 2. KV parity, byte for byte.
    expect_eq(fm.kv_bytes, ggml_backend_buffer_get_size(real.buffer_kv), "KV parity");

    // 3. Cross-K/V parity at the same description length.
    const int T = 7;
    {
        std::vector<int32_t> desc_ids(T);
        for (int i = 0; i < T; ++i) desc_ids[i] = i % 4;
        if (!parler_encode_description(real, desc_ids, /*n_threads=*/2)) {
            fail("real parler_encode_description failed");
        } else {
            size_t cross_bytes = 0;
            if (!parler_alloc_cross(mm, T, &cross_bytes)) {
                fail("measure-mode parler_alloc_cross failed");
            } else {
                expect_eq(cross_bytes, ggml_backend_buffer_get_size(real.buffer_cross),
                          "cross-K/V parity");
            }
        }
    }

    // 4. Decoder arena parity: real prefill + one step vs the projection of
    //    the same two shapes (one gallocr, so max).
    const int P = 5;
    {
        ggml_gallocr_t allocr =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(real.backend));
        std::vector<int32_t> prompt_ids(P, 1);
        std::vector<int32_t> start_frame(real.hparams.n_codebooks, 0);
        parler_step_logits logits;
        int n_past = 0;
        if (!parler_dec_prefill(real, prompt_ids, start_frame, allocr, 2, logits, n_past) ||
            !parler_dec_step(real, start_frame, n_past, allocr, 2, logits)) {
            fail("real prefill/step failed");
        } else {
            ::tts_cpp::detail::fit_graph_price prefill, step;
            if (!price(mm, parler_build_prefill_fit_graph(mm, P), prefill)) {
                fail("pricing the prefill graph failed");
            }
            if (!price(mm, parler_build_step_fit_graph(mm, n_past), step)) {
                fail("pricing the step graph failed");
            }
            if (prefill.host_bytes == 0 && step.host_bytes == 0) {
                expect_eq(std::max(prefill.device_bytes, step.device_bytes),
                          ggml_gallocr_get_buffer_size(allocr, 0),
                          "decoder arena parity");
            }
        }
        ggml_gallocr_free(allocr);
    }

    // 5. T5 arena: the size-only reservation must equal a real gallocr
    //    reservation of the same graph (the runtime's transient allocr is
    //    freed inside the encode, so this cross-API gate is the anchor).
    {
        ::tts_cpp::detail::fit_graph_price t5;
        if (!price(mm, parler_build_t5_fit_graph(mm, T), t5)) {
            fail("pricing the T5 graph failed");
        } else if (t5.host_bytes == 0) {
            ggml_gallocr_t allocr =
                ggml_gallocr_new(ggml_backend_get_default_buffer_type(real.backend));
            ggml_cgraph * gf = parler_build_t5_fit_graph(real, T);
            if (!allocr || !gf || !ggml_gallocr_reserve(allocr, gf)) {
                fail("real T5 reservation failed");
            } else {
                expect_eq(t5.device_bytes, ggml_gallocr_get_buffer_size(allocr, 0),
                          "T5 arena parity");
            }
            if (allocr) ggml_gallocr_free(allocr);
        }
    }

    // 6. DAC window arena parity: a real decode leaves the arena in
    //    model.dac_allocr; the projection of the same frame count must match
    //    byte for byte on the direct-dispatch path.
    {
        const int n_frames = 16;
        std::vector<int32_t> codes((size_t) real.hparams.dac_n_q * n_frames, 0);
        std::vector<float> pcm;
        std::string dac_err;
        if (!parler_dac_decode(real, codes.data(), n_frames, 2, pcm, nullptr, 0, -1,
                               &dac_err)) {
            fail("real parler_dac_decode failed: " + dac_err);
        } else {
            ggml_context * dac_ctx = nullptr;
            ggml_cgraph * gf = parler_build_dac_fit_graph(mm, n_frames, &dac_ctx);
            ::tts_cpp::detail::fit_graph_price dac;
            if (!price(mm, gf, dac)) {
                fail("pricing the DAC graph failed");
            } else if (dac.host_bytes == 0) {
                expect_eq(dac.device_bytes,
                          ggml_gallocr_get_buffer_size(real.dac_allocr, 0),
                          "DAC window arena parity");
            }
            if (dac_ctx) ggml_free(dac_ctx);
        }
    }

    parler_free_model(real);
    parler_free_model(mm);
}

void run_fit_gates(const std::string & path, int n_gpu_layers) {
    tts_cpp::parler::FitOptions fopts;
    fopts.model_gguf_path    = path;
    fopts.n_gpu_layers       = n_gpu_layers;
    fopts.description_tokens = 7;
    fopts.prompt_tokens      = 5;
    fopts.max_frames         = 16;

    const tts_cpp::FitResult fit = tts_cpp::parler::fit_params(fopts);
    expect(fit.status != tts_cpp::FitStatus::Error,
           "fit_params returned Error (" + fit.reason + ") for a readable model");
    expect(fit.device.weights_bytes > 0,       "projected weights_bytes == 0");
    expect(fit.device.state_bytes > 0,         "projected state_bytes == 0");
    expect(fit.device.lm_compute_bytes > 0,    "projected lm_compute_bytes == 0");
    expect(fit.device.codec_compute_bytes > 0, "projected codec_compute_bytes == 0");
    expect(fit.host_bytes > 0,                 "projected host_bytes == 0");
    expect(!fit.report.empty(),                "empty report");
    if (g_failures == 0) std::printf("%s", fit.report.c_str());

    // The per-description cross-K/V and T5 graphs grow with the description.
    {
        tts_cpp::parler::FitOptions longer = fopts;
        longer.description_tokens = 21;
        const tts_cpp::FitResult big = tts_cpp::parler::fit_params(longer);
        expect(big.status != tts_cpp::FitStatus::Error,
               "longer description was Error (" + big.reason + ")");
        expect(big.device.state_bytes > fit.device.state_bytes,
               "state did not grow with description_tokens");
    }

    // Near-INT_MAX workloads are rejected strictly (Error /
    // "workload-too-large"), never priced through wrapped-int graph shapes.
    {
        tts_cpp::parler::FitOptions huge = fopts;
        huge.prompt_tokens = std::numeric_limits<int>::max();
        tts_cpp::FitResult fr = tts_cpp::parler::fit_params(huge);
        expect(fr.status == tts_cpp::FitStatus::Error,
               "near-INT_MAX prompt_tokens was not Error");
        expect(fr.reason == "workload-too-large",
               "near-INT_MAX prompt_tokens reason was '" + fr.reason + "'");

        huge = fopts;
        huge.description_tokens = std::numeric_limits<int>::max();
        fr = tts_cpp::parler::fit_params(huge);
        expect(fr.status == tts_cpp::FitStatus::Error,
               "near-INT_MAX description_tokens was not Error");
        expect(fr.reason == "workload-too-large",
               "near-INT_MAX description_tokens reason was '" + fr.reason + "'");
    }

    // Engine::run refuses max_frames <= n_codebooks; the projection must too.
    {
        tts_cpp::parler::FitOptions bad = fopts;
        bad.max_frames = 1;
        const tts_cpp::FitResult fr = tts_cpp::parler::fit_params(bad);
        expect(fr.status == tts_cpp::FitStatus::Error &&
                   fr.reason == "workload-too-large",
               "max_frames <= n_codebooks was not workload-too-large");
    }

    // Errors surface as Error, never Success.
    {
        tts_cpp::parler::FitOptions bad = fopts;
        bad.model_gguf_path = path + ".does-not-exist";
        const tts_cpp::FitResult fr = tts_cpp::parler::fit_params(bad);
        expect(fr.status == tts_cpp::FitStatus::Error, "missing model was not Error");
        expect(fr.reason == "model-unreadable",
               "missing model reason was '" + fr.reason + "'");
    }
}

// A weightless description GGUF -- tensor headers only, vocabulary stripped --
// must load metadata-only and yield a projection (a fit measurement never
// tokenizes text), while a real load still refuses the missing vocabulary.
void run_weightless_description_gates() {
    const std::string path =
        write_tiny_parler_gguf(tiny_parler{}, /*with_vocab=*/false, /*only_meta=*/true);

    parler_model mm;
    parler_fit_measure fm;
    std::string error;
    if (!parler_load_gguf_metadata_only(path, mm, /*n_gpu_layers=*/0, fm, &error)) {
        fail("metadata-only load refused a vocab-less description: " + error);
    } else {
        expect(fm.weights_bytes > 0, "vocab-less description sized 0 weight bytes");
    }
    parler_free_model(mm);

    parler_model real;
    expect(!parler_load_gguf(path, real, /*n_gpu_layers=*/0, &error),
           "a real load accepted a vocab-less GGUF");
    expect(error.find("tokenizer.ggml.tokens") != std::string::npos,
           "the real-load rejection does not name the vocabulary: '" + error + "'");
    parler_free_model(real);

    tts_cpp::parler::FitOptions fopts;
    fopts.model_gguf_path    = path;
    fopts.n_gpu_layers       = 0;
    fopts.description_tokens = 7;
    fopts.prompt_tokens      = 5;
    fopts.max_frames         = 16;
    const tts_cpp::FitResult fit = tts_cpp::parler::fit_params(fopts);
    expect(fit.status != tts_cpp::FitStatus::Error,
           "fit_params on a vocab-less description was Error (" + fit.reason + ")");

    fs::remove(path);
}

}  // namespace

int main(int argc, char ** argv) {
    // These gates price the ggml DAC; a Core ML sidecar next to a fixture
    // model makes fit_params refuse instead (test-parler-dac-coreml covers it).
    setenv(PARLER_COREML_DISABLE_ENV, "1", 1);
    if (argc >= 2 && std::string(argv[1]) == "--synthetic-gpu") {
        // Same synthetic model, GPU backend requested: exercises the fused
        // qkv/head stack, the FA probe's KV/cross dtype, and the phase-GEMM
        // DAC path (falls back to CPU parity when no GPU is present).
        const std::string path = write_tiny_parler_gguf(tiny_parler{});
        run_parity_gates(path, /*n_gpu_layers=*/99);
        run_fit_gates(path, /*n_gpu_layers=*/99);
        fs::remove(path);
    } else if (argc >= 2) {
        const int n_gpu_layers = argc > 2 ? std::atoi(argv[2]) : 0;
        run_parity_gates(argv[1], n_gpu_layers);
        run_fit_gates(argv[1], n_gpu_layers);
    } else {
        const std::string path = write_tiny_parler_gguf(tiny_parler{});
        run_parity_gates(path, /*n_gpu_layers=*/0);
        run_fit_gates(path, /*n_gpu_layers=*/0);
        fs::remove(path);
        run_weightless_description_gates();
    }
    if (g_failures == 0) {
        std::printf("test-parler-fit-params: all checks passed\n");
    }
    return g_failures;
}
