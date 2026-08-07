#include "audio8/graph.h"
#include "audio8/internal.h"

#include <vector>

namespace tts_cpp {
namespace audio8 {
namespace detail {
namespace {

// The codebook rows only contribute where the semantic row holds a codec
// token; elsewhere the reference zeroes them, so a per-position gate stands in
// for the reference's torch.where.
ggml_tensor * sum_codebooks(ggml_context * ctx, ggml_tensor * table, ggml_tensor * ids,
                            ggml_tensor * gate, int hidden, int codebooks, int width) {
    ggml_tensor * rows = ggml_get_rows(ctx, table, ids);
    ggml_tensor * shaped = ggml_reshape_3d(ctx, rows, hidden, codebooks, width);
    const size_t stride = static_cast<size_t>(hidden) * codebooks * sizeof(float);
    ggml_tensor * total = nullptr;
    for (int index = 0; index < codebooks; ++index) {
        ggml_tensor * slice = ggml_view_2d(ctx, shaped, hidden, width, stride,
                                           static_cast<size_t>(index) * hidden * sizeof(float));
        total = total ? ggml_add(ctx, total, slice) : ggml_cont(ctx, slice);
    }
    return ggml_mul(ctx, total, gate);
}

ggml_tensor * embed_frames(ggml_context * ctx, const lm_model & model, int width) {
    const lm_hparams & hp = model.hp;
    ggml_tensor * text = input_i32(ctx, "text_ids", width);
    ggml_tensor * codes = input_i32(ctx, "codebook_ids", width * hp.num_codebooks);
    ggml_tensor * gate = input_f32(ctx, "semantic_gate", 1, width);
    ggml_tensor * base = ggml_get_rows(ctx, model.tok_emb, text);
    ggml_tensor * stacked = sum_codebooks(ctx, model.codebook_emb, codes, gate, hp.hidden,
                                          hp.num_codebooks, width);
    return ggml_add(ctx, base, stacked);
}

ggml_tensor * run_block(ggml_context * ctx, ggml_cgraph * graph, const block_weights & block,
                        ggml_tensor * x, const rope_planes & rope, const kv_cache & cache,
                        const attention_shape & shape, ggml_tensor * mask, float eps) {
    ggml_tensor * normed = rms_norm(ctx, x, block.attn.attn_norm, eps);
    ggml_tensor * hidden = ggml_add(
        ctx, x, attention(ctx, graph, block.attn, normed, rope, cache, shape, mask));
    ggml_tensor * gated = rms_norm(ctx, hidden, block.ffn_norm, eps);
    return ggml_add(ctx, hidden, swiglu(ctx, block.w1, block.w2, block.w3, gated));
}

ggml_tensor * run_blocks(ggml_context * ctx, ggml_cgraph * graph,
                         const std::vector<block_weights> & blocks, ggml_tensor * x,
                         const rope_planes & rope, const kv_cache & cache,
                         attention_shape shape, ggml_tensor * mask, float eps) {
    for (size_t index = 0; index < blocks.size(); ++index) {
        shape.layer = static_cast<int>(index);
        x = run_block(ctx, graph, blocks[index], x, rope, cache, shape, mask, eps);
    }
    return x;
}

ggml_tensor * last_column(ggml_context * ctx, ggml_tensor * x) {
    const size_t offset = static_cast<size_t>(x->ne[1] - 1) * x->nb[1];
    return ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], offset);
}

attention_shape slow_shape(const lm_hparams & hp, int width, int n_past) {
    return {hp.n_head, hp.n_kv, hp.head_dim, width, n_past, 0};
}

attention_shape fast_shape(const lm_hparams & hp, int position) {
    return {hp.fast_n_head, hp.fast_n_kv, hp.fast_head_dim, 1, position, 0};
}

bool is_semantic(const lm_hparams & hp, int32_t token) {
    return token >= hp.semantic_begin && token <= hp.semantic_end;
}

// Row 0 carries text or semantic ids; the codebook rows are read per position
// with the reference's per-codebook offset already applied.
struct frame_inputs {
    std::vector<int32_t> text;
    std::vector<int32_t> codes;
    std::vector<float> gate;
    std::vector<float> mask;
};

void fill_frame_inputs(const lm_hparams & hp, const int32_t * frames, int width, int n_past,
                       frame_inputs & inputs) {
    const int rows = hp.num_codebooks + 1;
    inputs.text.resize(width);
    inputs.codes.resize(static_cast<size_t>(width) * hp.num_codebooks);
    inputs.gate.resize(width);
    for (int column = 0; column < width; ++column) {
        const int32_t * frame = frames + static_cast<size_t>(column) * rows;
        inputs.text[column] = frame[0];
        inputs.gate[column] = is_semantic(hp, frame[0]) ? 1.0f : 0.0f;
        for (int book = 0; book < hp.num_codebooks; ++book) {
            inputs.codes[static_cast<size_t>(column) * hp.num_codebooks + book] =
                frame[book + 1] + book * hp.codebook_size;
        }
    }
    inputs.mask.resize(static_cast<size_t>(width) * (n_past + width));
    fill_causal_mask(inputs.mask.data(), n_past + width, width, n_past, /*window=*/0);
}

void set_frame_inputs(ggml_cgraph * graph, const frame_inputs & inputs) {
    write_input(graph, "text_ids", inputs.text.data(), inputs.text.size() * sizeof(int32_t));
    write_input(graph, "codebook_ids", inputs.codes.data(),
                inputs.codes.size() * sizeof(int32_t));
    write_input(graph, "semantic_gate", inputs.gate.data(), inputs.gate.size() * sizeof(float));
    write_input(graph, "mask", inputs.mask.data(), inputs.mask.size() * sizeof(float));
}

}  // namespace

int argmax_of(const std::vector<float> & values) {
    int best = 0;
    for (size_t index = 1; index < values.size(); ++index) {
        if (values[index] > values[best]) best = static_cast<int>(index);
    }
    return best;
}

prompt_frames build_frames(const lm_hparams & hp, const PromptSegments & segments,
                           const std::vector<int32_t> & reference_codes, int reference_len) {
    prompt_frames frames;
    frames.rows = hp.num_codebooks + 1;
    frames.width = static_cast<int>(segments.prefix.size() + segments.suffix.size()) +
                   reference_len;
    frames.values.assign(static_cast<size_t>(frames.rows) * frames.width, 0);

    int column = 0;
    for (int32_t token : segments.prefix) {
        frames.values[static_cast<size_t>(column++) * frames.rows] = token;
    }
    for (int index = 0; index < reference_len; ++index) {
        const size_t base = static_cast<size_t>(column + index) * frames.rows;
        frames.values[base] = reference_codes[index] + hp.semantic_begin;
        for (int book = 0; book < hp.num_codebooks; ++book) {
            frames.values[base + book + 1] =
                reference_codes[static_cast<size_t>(book) * reference_len + index];
        }
    }
    column += reference_len;
    for (int32_t token : segments.suffix) {
        frames.values[static_cast<size_t>(column++) * frames.rows] = token;
    }
    return frames;
}

bool slow_step(lm_model & model, const int32_t * frames, int width, int n_past,
               int n_threads, std::vector<float> & sem_logits,
               std::vector<float> & fast_input, std::string * error) {
    const lm_hparams & hp = model.hp;
    if (n_past + width > hp.max_seq_len) {
        if (error) *error = "audio8: prompt and generation exceed the model's context";
        return false;
    }
    scratch build(AUDIO8_MAX_NODES);
    if (!build.ok()) {
        if (error) *error = "audio8: failed to create the slow graph context";
        return false;
    }
    ggml_context * ctx = build.ctx;
    ggml_tensor * mask = input_f32(ctx, "mask", n_past + width, width);
    ggml_tensor * hidden = embed_frames(ctx, model, width);
    const rope_planes rope = rope_window(ctx, model.rope_cos, model.rope_sin, n_past, width);
    hidden = run_blocks(ctx, build.graph, model.blocks, hidden, rope, model.slow_kv,
                        slow_shape(hp, width, n_past), mask, hp.rms_eps);

    ggml_tensor * tail = ggml_cont(ctx, last_column(ctx, hidden));
    ggml_tensor * normed = rms_norm(ctx, tail, model.norm, hp.rms_eps);
    ggml_tensor * logits = mark_output(build.graph, ggml_mul_mat(ctx, model.sem_head, normed));
    ggml_tensor * carried = mark_output(build.graph, hp.norm_fast_input ? normed : tail);

    if (!allocate_graph(model.slow_allocr, build.graph, "slow", error)) return false;
    frame_inputs inputs;
    fill_frame_inputs(hp, frames, width, n_past, inputs);
    set_frame_inputs(build.graph, inputs);
    if (!compute_graph(model.backend, build.graph, n_threads, "slow", error)) return false;

    read_output(logits, sem_logits);
    read_output(carried, fast_input);
    return true;
}

namespace {

// Position 0 consumes the slow transformer's hidden state and only primes the
// cache; later positions read the previous codebook's embedding.
struct fast_source {
    const void * data = nullptr;
    size_t bytes = 0;
    bool is_code = false;
};

ggml_tensor * fast_input_tensor(ggml_context * ctx, const lm_model & model,
                                const fast_source & source) {
    if (!source.is_code) {
        return input_f32(ctx, "input", model.hp.fast_hidden, 1);
    }
    return ggml_get_rows(ctx, model.fast_emb, input_i32(ctx, "input", 1));
}

bool fast_pass(lm_model & model, const fast_source & source, int position, int n_threads,
               std::vector<float> * logits_out, std::string * error) {
    const lm_hparams & hp = model.hp;
    scratch build(AUDIO8_MAX_NODES);
    if (!build.ok()) {
        if (error) *error = "audio8: failed to create the fast graph context";
        return false;
    }
    ggml_context * ctx = build.ctx;
    const int keys = position + 1;
    ggml_tensor * mask = input_f32(ctx, "mask", keys, 1);
    ggml_tensor * hidden = fast_input_tensor(ctx, model, source);
    const rope_planes rope = rope_window(ctx, model.fast_rope_cos, model.fast_rope_sin,
                                         position, 1);
    hidden = run_blocks(ctx, build.graph, model.fast_blocks, hidden, rope, model.fast_kv,
                        fast_shape(hp, position), mask, hp.rms_eps);
    ggml_tensor * logits = mark_output(
        build.graph,
        ggml_mul_mat(ctx, model.fast_out, rms_norm(ctx, hidden, model.fast_norm, hp.rms_eps)));

    if (!allocate_graph(model.fast_allocr, build.graph, "fast", error)) return false;
    std::vector<float> mask_values(keys);
    fill_causal_mask(mask_values.data(), keys, 1, position, /*window=*/0);
    write_input(build.graph, "mask", mask_values.data(), mask_values.size() * sizeof(float));
    write_input(build.graph, "input", source.data, source.bytes);
    if (!compute_graph(model.backend, build.graph, n_threads, "fast", error)) return false;

    if (logits_out) read_output(logits, *logits_out);
    return true;
}

int clamp_to_codebook(const lm_hparams & hp, int semantic) {
    const int index = semantic - hp.semantic_begin;
    if (index < 0) return 0;
    return index >= hp.codebook_size ? hp.codebook_size - 1 : index;
}

}  // namespace

bool fast_step(lm_model & model, const std::vector<float> & fast_input, int semantic,
               int n_threads, const code_picker & pick, std::vector<int32_t> & codes_out,
               std::string * error) {
    const lm_hparams & hp = model.hp;
    const fast_source prime = {fast_input.data(), fast_input.size() * sizeof(float), false};
    if (!fast_pass(model, prime, 0, n_threads, nullptr, error)) return false;

    codes_out.assign(hp.num_codebooks, 0);
    codes_out[0] = clamp_to_codebook(hp, semantic);
    std::vector<float> logits;
    for (int position = 1; position < hp.num_codebooks; ++position) {
        const int32_t token = codes_out[position - 1];
        const fast_source step = {&token, sizeof(token), true};
        if (!fast_pass(model, step, position, n_threads, &logits, error)) return false;
        codes_out[position] = pick(logits, position);
    }
    return true;
}

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
