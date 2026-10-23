// Audio8 codec synthesis: ten codebook rows per frame in, waveform out.
//
// The path mirrors the reference exactly -- look each code up, project it to
// the latent width and sum, run the windowed post transformer, upsample by
// four, then the DAC stack's four transposed-conv stages and a tanh.
//
// It runs as two graphs rather than one, because the two halves want opposite
// things. The post transformer is eight windowed layers reaching 128 frames
// back each, so between them they see roughly a thousand frames and the whole
// sequence has to go through at once -- cheap, since it works at one column
// per frame. The synthesis stack then expands each of those columns into 2048
// samples, which is where the memory goes, and it is causal with a context of
// only a dozen frames, so it runs in blocks sized to a memory budget. That
// keeps the scratch bounded instead of the 1.5 GB a whole 24 s take would need.

#include "audio8/codec_ops.h"
#include "audio8/graph.h"
#include "audio8/window_stream.h"

#include "fit_price.h"
#include "fit_util.h"

#ifdef TTS_CPP_USE_COREML
#include "audio8/coreml/codec-synth.h"
#include "audio8/coreml_windows.h"
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace tts_cpp {
namespace audio8 {
namespace detail {

namespace {
constexpr double SCRATCH_BUDGET_SHARE = 0.25;
constexpr size_t SCRATCH_BUDGET_CAP = 384u * 1024 * 1024;
// The narrowest block that still produces audio, so the floor of every search
// and the width taken when even it is over budget.
constexpr int MIN_BLOCK_FRAMES = 1;
}  // namespace

size_t synthesis_scratch_budget(size_t configured, size_t free_bytes, size_t total_bytes) {
    if (configured > 0) return configured;
    // A zero total is a backend saying it cannot tell rather than a device with
    // no memory: ggml-metal reports zero for both figures on the OS versions
    // predating the working-set query. Real pressure shows as a small free
    // against a nonzero total, and that does have to narrow the blocks.
    if (total_bytes == 0) return SCRATCH_BUDGET_CAP;
    const size_t share = static_cast<size_t>(free_bytes * SCRATCH_BUDGET_SHARE);
    return std::min(share, SCRATCH_BUDGET_CAP);
}

namespace {

const char * const CODE_INPUT = "codes";
const char * const MASK_INPUT = "post_mask";
const char * const POST_INPUT = "post";

struct latent_graph {
    ggml_tensor * semantic = nullptr;
    ggml_tensor * residual = nullptr;
    ggml_tensor * post = nullptr;
};

struct synthesis_graph {
    ggml_tensor * latent = nullptr;
    ggml_tensor * pcm = nullptr;
};

ggml_tensor * code_row(ggml_context * ctx, ggml_tensor * codes, int row) {
    return ggml_view_1d(ctx, codes, codes->ne[0], static_cast<size_t>(row) * codes->nb[1]);
}

ggml_tensor * reconstruct(ggml_context * ctx, const quantizer_weights & quantizer,
                          ggml_tensor * ids) {
    return project(ctx, quantizer.out_proj, ggml_get_rows(ctx, quantizer.codebook, ids));
}

ggml_tensor * sum_quantizers(ggml_context * ctx, const std::vector<quantizer_weights> & bank,
                             ggml_tensor * codes, int first_row) {
    ggml_tensor * total = nullptr;
    for (size_t index = 0; index < bank.size(); ++index) {
        ggml_tensor * part = reconstruct(
            ctx, bank[index], code_row(ctx, codes, first_row + static_cast<int>(index)));
        total = total ? ggml_add(ctx, total, part) : part;
    }
    return total;
}

ggml_tensor * run_upsample(ggml_context * ctx, const std::vector<resample_stage> & stages,
                           ggml_tensor * signal, float eps, const codec_conv_options & options) {
    for (const resample_stage & stage : stages) {
        signal = causal_conv_transpose(ctx, stage.conv, signal, stage.stride, options);
        signal = convnext_block(ctx, stage.convnext, signal, eps, options);
    }
    return signal;
}

ggml_tensor * run_decoder_stack(ggml_context * ctx, const codec_model & model,
                                ggml_tensor * signal) {
    const codec_hparams & hp = model.hp;
    signal = causal_conv(ctx, model.dec_in, signal, /*stride=*/1, /*dilation=*/1, model.synthesis_convs);
    for (const dac_stage & stage : model.dec_stages) {
        signal = snake(ctx, signal, stage.alpha, hp.snake_epsilon);
        signal = causal_conv_transpose(ctx, stage.conv, signal, stage.stride, model.synthesis_convs);
        signal = residual_stack(ctx, stage.units, hp.residual_dilations, signal,
                                hp.snake_epsilon, model.synthesis_convs);
    }
    signal = snake(ctx, signal, model.dec_out_alpha, hp.snake_epsilon);
    return ggml_tanh(ctx, causal_conv(ctx, model.dec_out, signal, /*stride=*/1,
                                      /*dilation=*/1, model.synthesis_convs));
}

int upsample_factor(const codec_model & model) {
    int factor = 1;
    for (const resample_stage & stage : model.upsample) factor *= stage.stride;
    return factor;
}

int synthesis_context(const codec_model & model) {
    return synthesis_context_frames(model);
}

}  // namespace

int synthesis_context_frames(const codec_model & model) {
    int span = span_through_conv(1, conv_taps(model.dec_out), 1, 1);
    for (size_t index = model.dec_stages.size(); index-- > 0;) {
        const dac_stage & stage = model.dec_stages[index];
        span = span_through_units(span, stage.units, model.hp.residual_dilations);
        span = span_through_transpose(span, conv_taps(stage.conv), stage.stride);
    }
    span = span_through_conv(span, conv_taps(model.dec_in), 1, 1);
    for (size_t index = model.upsample.size(); index-- > 0;) {
        span = span_through_convnext(span, model.upsample[index].convnext);
        span = span_through_transpose(span, conv_taps(model.upsample[index].conv),
                                      model.upsample[index].stride);
    }
    return span - 1;
}

namespace {

latent_graph build_latents(ggml_context * ctx, const codec_model & model, int n_frames) {
    ggml_tensor * codes = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_frames,
                                             model.hp.num_codebooks);
    ggml_set_name(codes, CODE_INPUT);
    ggml_set_input(codes);
    ggml_tensor * mask = input_f32(ctx, MASK_INPUT, n_frames, n_frames);

    latent_graph built;
    built.semantic = sum_quantizers(ctx, model.semantic_quantizers, codes, 0);
    built.residual = sum_quantizers(ctx, model.residual_quantizers, codes, 1);
    built.post =
        window_forward(ctx, model.post, ggml_add(ctx, built.semantic, built.residual),
                       mask, model.precise_outputs);
    return built;
}

synthesis_graph build_synthesis(ggml_context * ctx, const codec_model & model,
                                int n_frames) {
    ggml_tensor * post = input_f32(ctx, POST_INPUT, model.hp.latent_dim, n_frames);
    synthesis_graph built;
    built.latent = run_upsample(ctx, model.upsample, post, model.hp.convnext_norm_eps, model.synthesis_convs);
    built.pcm = run_decoder_stack(ctx, model, built.latent);
    return built;
}

int row_limit(const codec_hparams & hp, int row) {
    return (row == 0 ? hp.semantic_codebook_size : hp.residual_codebook_size) - 1;
}

void clamp_row(const int32_t * codes, int n_frames, int limit, int32_t * out) {
    for (int frame = 0; frame < n_frames; ++frame) {
        out[frame] = std::min(std::max(codes[frame], 0), limit);
    }
}

// Codes outside a codebook would index past the embedding table, so they are
// clamped the way the reference clamps them before the lookup.
std::vector<int32_t> clamped_codes(const codec_hparams & hp, const int32_t * codes,
                                   int n_frames) {
    std::vector<int32_t> out(static_cast<size_t>(hp.num_codebooks) * n_frames);
    for (int row = 0; row < hp.num_codebooks; ++row) {
        const size_t at = static_cast<size_t>(row) * n_frames;
        clamp_row(codes + at, n_frames, row_limit(hp, row), out.data() + at);
    }
    return out;
}

bool run_latents(codec_model & model, const int32_t * codes, int n_frames, int n_threads,
                 std::vector<float> & post, decode_taps * taps, std::string * error) {
    scratch work(AUDIO8_MAX_NODES);
    if (!work.ok()) {
        if (error) *error = "audio8: failed to create the latent graph context";
        return false;
    }
    const latent_graph built = build_latents(work.ctx, model, n_frames);
    mark_output(work.graph, built.post);
    if (taps) {
        mark_output(work.graph, built.semantic);
        mark_output(work.graph, built.residual);
    }
    bool use_sched = false;
    if (!prepare_graph(model.backend, model.sched, model.buffer_w, model.allocr,
                       work.graph, "latent", use_sched, error)) {
        return false;
    }

    const std::vector<int32_t> clamped = clamped_codes(model.hp, codes, n_frames);
    write_input(work.graph, CODE_INPUT, clamped.data(), clamped.size() * sizeof(int32_t));
    const std::vector<float> mask = window_mask(model.post.spec, n_frames);
    write_input(work.graph, MASK_INPUT, mask.data(), mask.size() * sizeof(float));
    if (!compute_graph(model.backend, model.sched, work.graph, use_sched, n_threads,
                       "latent", error)) {
        return false;
    }

    read_output(built.post, post);
    if (taps) {
        read_output(built.semantic, taps->semantic);
        read_output(built.residual, taps->residual);
        taps->post = post;
    }
    return true;
}

// A block covers post-transformer columns [first - context, first + count) and
// keeps only what belongs to [first, first + count): the context is there to
// give the causal convolutions real history instead of the zeros they would
// otherwise pad with.
struct block {
    int begin = 0;
    int count = 0;
    int dropped = 0;
};

block block_at(int first, int n_frames, int context, int block_frames) {
    block span;
    span.begin = std::max(0, first - context);
    span.dropped = first - span.begin;
    span.count = std::min(block_frames, n_frames - first) + span.dropped;
    return span;
}

size_t device_scratch_budget(const codec_model & model) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(model.backend), &free_bytes,
                            &total_bytes);
    return synthesis_scratch_budget(model.synthesis_scratch_budget, free_bytes,
                                    total_bytes);
}

size_t block_scratch(codec_model & model, ggml_gallocr_t pricer, int columns,
                     bool with_taps) {
    scratch work(AUDIO8_MAX_NODES);
    if (!work.ok()) return SIZE_MAX;
    const synthesis_graph built = build_synthesis(work.ctx, model, columns);
    mark_output(work.graph, built.pcm);
    if (with_taps) mark_output(work.graph, built.latent);
    size_t size = 0;
    ggml_gallocr_reserve_n_size(pricer, work.graph, nullptr, nullptr, &size);
    return size;
}

int span_of(int block_frames, int n_frames, int context) {
    return std::min(block_frames + context, n_frames);
}

int widest_block(codec_model & model, ggml_gallocr_t pricer, int context, int n_frames,
                 bool with_taps, size_t budget) {
    int narrowest = MIN_BLOCK_FRAMES;
    int widest = n_frames;
    while (narrowest < widest) {
        const int mid = narrowest + (widest - narrowest + 1) / 2;
        const size_t cost =
            block_scratch(model, pricer, span_of(mid, n_frames, context), with_taps);
        if (cost <= budget) {
            narrowest = mid;
        } else {
            widest = mid - 1;
        }
    }
    return narrowest;
}

struct block_plan {
    int frames = MIN_BLOCK_FRAMES;
    size_t scratch = 0;
};

block_plan plan_blocks(codec_model & model, int context, int n_frames, bool with_taps) {
    ggml_gallocr_t pricer =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    block_plan plan;
    plan.frames = model.synthesis_block_frames > 0
                      ? std::min(model.synthesis_block_frames, n_frames)
                      : widest_block(model, pricer, context, n_frames, with_taps,
                                     device_scratch_budget(model));
    plan.scratch =
        block_scratch(model, pricer, span_of(plan.frames, n_frames, context), with_taps);
    ggml_gallocr_free(pricer);
    return plan;
}

void append_tail(const std::vector<float> & produced, int dropped, int per_column,
                 std::vector<float> & out) {
    const size_t skip = static_cast<size_t>(dropped) * per_column;
    out.insert(out.end(), produced.begin() + skip, produced.end());
}

bool run_block(codec_model & model, const std::vector<float> & post, const block & span,
               int n_threads, std::vector<float> & pcm_out, decode_taps * taps,
               std::string * error) {
    const codec_hparams & hp = model.hp;
    scratch work(AUDIO8_MAX_NODES);
    if (!work.ok()) {
        if (error) *error = "audio8: failed to create the synthesis graph context";
        return false;
    }
    const synthesis_graph built = build_synthesis(work.ctx, model, span.count);
    mark_output(work.graph, built.pcm);
    if (taps) mark_output(work.graph, built.latent);
    bool use_sched = false;
    if (!prepare_graph(model.backend, model.sched, model.buffer_w, model.block_allocr,
                       work.graph, "synthesis", use_sched, error)) {
        return false;
    }

    const size_t offset = static_cast<size_t>(span.begin) * hp.latent_dim;
    write_input(work.graph, POST_INPUT, post.data() + offset,
                static_cast<size_t>(span.count) * hp.latent_dim * sizeof(float));
    if (!compute_graph(model.backend, model.sched, work.graph, use_sched, n_threads,
                       "synthesis", error)) {
        return false;
    }

    std::vector<float> produced;
    read_output(built.pcm, produced);
    append_tail(produced, span.dropped, hp.frame_size, pcm_out);
    if (taps) {
        read_output(built.latent, produced);
        append_tail(produced, span.dropped * upsample_factor(model), hp.latent_dim,
                    taps->latent);
    }
    return true;
}

#ifdef TTS_CPP_USE_COREML
enum class coreml_status { done, unavailable, cancelled };

// Test-only: fail instead of falling back to ggml.
bool coreml_strict() {
    return std::getenv("AUDIO8_COREML_STRICT") != nullptr;
}

bool run_coreml_window(codec_model & model, const std::vector<float> & post,
                       const coreml_window & span, std::vector<float> & in,
                       std::vector<float> & out, std::vector<float> & pcm_out) {
    const codec_hparams & hp = model.hp;
    const size_t latent = static_cast<size_t>(hp.latent_dim);
    const size_t frame = static_cast<size_t>(hp.frame_size);
    std::fill(in.begin(), in.end(), 0.0f);
    std::copy(post.begin() + static_cast<size_t>(span.begin) * latent,
              post.begin() + static_cast<size_t>(span.begin + span.filled) * latent,
              in.begin());
    if (audio8_coreml_codec_synthesize(model.coreml, in.data(), out.data()) != 0) return false;
    const size_t skip = static_cast<size_t>(span.core_begin - span.begin) * frame;
    const size_t keep = static_cast<size_t>(span.core_end - span.core_begin) * frame;
    std::copy(out.begin() + skip, out.begin() + skip + keep,
              pcm_out.begin() + static_cast<size_t>(span.core_begin) * frame);
    return true;
}

// A sidecar that failed once is retired: the plan and the model are fixed, so
// the failure would repeat on every call, each time ahead of a full ggml pass.
void retire_coreml_sidecar(codec_model & model, const char * why) {
    std::fprintf(stderr, "[audio8] retiring the Core ML sidecar (%s); synthesis stays on ggml\n", why);
    audio8_coreml_codec_free(model.coreml);
    model.coreml = nullptr;
    model.synthesis_on_coreml = false;
}

// Cancel keeps the completed windows; unavailable leaves pcm_out empty for the fallback.
coreml_status run_synthesis_coreml(codec_model & model, const std::vector<float> & post,
                                   int n_frames, const cancel_hook & cancel,
                                   std::vector<float> & pcm_out, decode_timing & clock,
                                   std::string * error) {
    const codec_hparams & hp = model.hp;
    const int window = static_cast<int>(audio8_coreml_codec_window_frames(model.coreml));
    const std::vector<coreml_window> plan =
        plan_coreml_windows(n_frames, window, synthesis_context_frames(model));
    if (plan.empty()) {
        retire_coreml_sidecar(model, "its window cannot carry the causal context");
        return coreml_status::unavailable;
    }

    pcm_out.assign(static_cast<size_t>(n_frames) * hp.frame_size, 0.0f);
    std::vector<float> in(static_cast<size_t>(window) * hp.latent_dim);
    std::vector<float> out(static_cast<size_t>(window) * hp.frame_size);
    int completed = 0;
    for (const coreml_window & span : plan) {
        if (cancelled(cancel, error)) {
            pcm_out.resize(static_cast<size_t>(completed) * hp.frame_size);
            return coreml_status::cancelled;
        }
        if (!run_coreml_window(model, post, span, in, out, pcm_out)) {
            pcm_out.clear();
            retire_coreml_sidecar(model, "prediction failed");
            return coreml_status::unavailable;
        }
        completed = span.core_end;
    }
    clock.block_frames = window;
    clock.block_scratch = 0;
    clock.synthesis_backend = model.coreml ? audio8_coreml_codec_backend_label(model.coreml) : "ggml";
    return coreml_status::done;
}
#endif

bool run_synthesis_blocks(codec_model & model, const std::vector<float> & post,
                          int n_frames, int n_threads, const cancel_hook & cancel,
                          std::vector<float> & pcm_out, decode_taps * taps,
                          decode_timing & clock, std::string * error) {
#ifdef TTS_CPP_USE_COREML
    if (model.coreml && !taps) {
        switch (run_synthesis_coreml(model, post, n_frames, cancel, pcm_out, clock, error)) {
            case coreml_status::done:      return true;
            case coreml_status::cancelled: return false;
            case coreml_status::unavailable:
                if (coreml_strict()) {
                    if (error) {
                        *error = "audio8: Core ML synthesis unavailable for " +
                                 std::to_string(n_frames) +
                                 " frames and AUDIO8_COREML_STRICT is set; failing instead "
                                 "of the ggml fallback";
                    }
                    return false;
                }
                std::fprintf(stderr, "[audio8] Core ML synthesis unavailable for %d frames; "
                                     "using ggml\n", n_frames);
                break;
        }
    } else if (coreml_strict() && !taps) {
        if (error) {
            *error = "audio8: no Core ML sidecar loaded and AUDIO8_COREML_STRICT is set; "
                     "failing instead of the ggml fallback";
        }
        return false;
    }
#endif
    pcm_out.reserve(static_cast<size_t>(n_frames) * model.hp.frame_size);
    if (taps) taps->latent.clear();
    const int context = synthesis_context(model);
    const block_plan plan = plan_blocks(model, context, n_frames, taps != nullptr);
    const int block_frames = plan.frames;
    clock.block_frames = plan.frames;
    clock.block_scratch = plan.scratch;
    clock.synthesis_backend = "ggml";
    for (int first = 0; first < n_frames; first += block_frames) {
        if (cancelled(cancel, error)) return false;
        if (!run_block(model, post, block_at(first, n_frames, context, block_frames),
                       n_threads, pcm_out, taps, error)) {
            return false;
        }
    }
    return true;
}

bool check_decodable(const codec_model & model, int n_frames, std::string * error) {
    if (!model.has_decoder) {
        if (error) *error = "audio8: this codec GGUF has no decoder";
        return false;
    }
    if (n_frames > model.hp.max_frames) {
        if (error) {
            *error = "audio8: " + std::to_string(n_frames) +
                     " frames exceeds the baked RoPE table of " +
                     std::to_string(model.hp.max_frames);
        }
        return false;
    }
    return true;
}

}  // namespace

bool decode_codes(codec_model & model, const int32_t * codes, int n_frames,
                  int n_threads, const cancel_hook & cancel,
                  std::vector<float> & pcm_out, std::string * error,
                  decode_taps * taps, decode_timing * timing) {
    pcm_out.clear();
    if (!check_decodable(model, n_frames, error)) return false;
    if (n_frames <= 0) return true;

    decode_timing discarded;
    decode_timing & clock = timing ? *timing : discarded;

    std::vector<float> post;
    {
        stage_timer measure(clock.latent_ms);
        if (!run_latents(model, codes, n_frames, n_threads, post, taps, error)) return false;
    }
    stage_timer measure(clock.synthesis_ms);
    return run_synthesis_blocks(model, post, n_frames, n_threads, cancel, pcm_out, taps,
                                clock, error);
}

namespace {

const char * const HELD_KEYS = "held_keys_";
const char * const HELD_VALUES = "held_values_";

std::string held_name(const char * kind, size_t layer) {
    return std::string(kind) + std::to_string(layer);
}

ggml_tensor * held_input(ggml_context * ctx, const transformer_spec & spec, const char * kind,
                         size_t layer, int held) {
    ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, spec.head_dim, spec.n_kv, held);
    ggml_set_name(t, held_name(kind, layer).c_str());
    ggml_set_input(t);
    return t;
}

window_history held_inputs(ggml_context * ctx, const window_transformer & post, int held) {
    window_history inputs;
    if (held == 0) return inputs;
    for (size_t layer = 0; layer < post.blocks.size(); ++layer) {
        inputs.keys.push_back(held_input(ctx, post.spec, HELD_KEYS, layer, held));
        inputs.values.push_back(held_input(ctx, post.spec, HELD_VALUES, layer, held));
    }
    return inputs;
}

struct post_chunk_graph {
    ggml_tensor * post = nullptr;
    window_history reached;
};

void mark_history(ggml_cgraph * graph, const window_history & reached) {
    for (size_t layer = 0; layer < reached.keys.size(); ++layer) {
        mark_output(graph, reached.keys[layer]);
        mark_output(graph, reached.values[layer]);
    }
}

post_chunk_graph build_post_chunk(ggml_context * ctx, ggml_cgraph * graph,
                                  const codec_model & model, int first, int width, int held) {
    ggml_tensor * codes = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, width, model.hp.num_codebooks);
    ggml_set_name(codes, CODE_INPUT);
    ggml_set_input(codes);
    ggml_tensor * mask = input_f32(ctx, MASK_INPUT, held + width, width);
    ggml_tensor * latent =
        ggml_add(ctx, sum_quantizers(ctx, model.semantic_quantizers, codes, 0),
                 sum_quantizers(ctx, model.residual_quantizers, codes, 1));
    post_chunk_graph built;
    built.post = window_forward_after(ctx, model.post, latent, mask, first,
                                      held_inputs(ctx, model.post, held), built.reached,
                                      model.precise_outputs);
    mark_output(graph, built.post);
    mark_history(graph, built.reached);
    return built;
}

void write_held(ggml_cgraph * graph, const post_history & history) {
    for (size_t layer = 0; history.held > 0 && layer < history.keys.size(); ++layer) {
        write_input(graph, held_name(HELD_KEYS, layer).c_str(), history.keys[layer].data(),
                    history.keys[layer].size() * sizeof(float));
        write_input(graph, held_name(HELD_VALUES, layer).c_str(),
                    history.values[layer].data(),
                    history.values[layer].size() * sizeof(float));
    }
}

// What the next chunk needs: the last window - 1 positions, which its first
// query reaches back to. Only those are read back.
void keep_tail(ggml_tensor * t, int keep, std::vector<float> & out) {
    const size_t per_position = static_cast<size_t>(t->ne[0]) * static_cast<size_t>(t->ne[1]);
    const size_t skipped = static_cast<size_t>(t->ne[2] - keep) * per_position;
    out.resize(per_position * static_cast<size_t>(keep));
    ggml_backend_tensor_get(t, out.data(), skipped * sizeof(float), out.size() * sizeof(float));
}

void remember(const window_history & reached, int keep, post_history & history) {
    history.keys.resize(reached.keys.size());
    history.values.resize(reached.values.size());
    for (size_t layer = 0; layer < reached.keys.size(); ++layer) {
        keep_tail(reached.keys[layer], keep, history.keys[layer]);
        keep_tail(reached.values[layer], keep, history.values[layer]);
    }
    history.held = keep;
}

void append_post(ggml_tensor * t, std::vector<float> & post) {
    std::vector<float> chunk;
    read_output(t, chunk);
    post.insert(post.end(), chunk.begin(), chunk.end());
}

}  // namespace

bool post_has_window(const codec_model & model) {
    return model.post.spec.window > 0;
}

bool extend_post(codec_model & model, const int32_t * codes, int end, int n_threads,
                 post_history & history, std::vector<float> & post, std::string * error) {
    const int width = end - history.done;
    if (width <= 0) return true;
    if (!post_has_window(model)) {
        if (error) *error = "audio8: the post transformer has no attention window to carry";
        return false;
    }
    if (!check_decodable(model, end, error)) return false;
    scratch work(AUDIO8_MAX_NODES);
    if (!work.ok()) {
        if (error) *error = "audio8: failed to create the post chunk context";
        return false;
    }
    const post_chunk_graph built =
        build_post_chunk(work.ctx, work.graph, model, history.done, width, history.held);
    bool use_sched = false;
    if (!prepare_graph(model.backend, model.sched, model.buffer_w, model.allocr, work.graph,
                       "post chunk", use_sched, error)) {
        return false;
    }
    const std::vector<int32_t> clamped = clamped_codes(model.hp, codes, width);
    write_input(work.graph, CODE_INPUT, clamped.data(), clamped.size() * sizeof(int32_t));
    const std::vector<float> mask = window_mask_after(model.post.spec, history.held, width);
    write_input(work.graph, MASK_INPUT, mask.data(), mask.size() * sizeof(float));
    write_held(work.graph, history);
    if (!compute_graph(model.backend, model.sched, work.graph, use_sched, n_threads,
                       "post chunk", error)) {
        return false;
    }
    append_post(built.post, post);
    remember(built.reached, std::min(model.post.spec.window - 1, history.held + width), history);
    history.done = end;
    return true;
}

namespace {

void place_frame(const std::vector<int32_t> & frames, int books, int frame, int first,
                 int count, std::vector<int32_t> & rows) {
    for (int book = 0; book < books; ++book) {
        rows[static_cast<size_t>(book) * count + (frame - first)] =
            frames[static_cast<size_t>(frame) * books + book];
    }
}

}  // namespace

std::vector<int32_t> codebook_rows(const std::vector<int32_t> & frames, int books, int first,
                                   int end) {
    const int count = end - first;
    std::vector<int32_t> rows(static_cast<size_t>(books) * count);
    for (int frame = first; frame < end; ++frame) {
        place_frame(frames, books, frame, first, count, rows);
    }
    return rows;
}

namespace {

#ifdef TTS_CPP_USE_COREML
bool coreml_streaming_enabled() {
    return std::getenv("AUDIO8_COREML_STREAM_DISABLE") == nullptr;
}

#endif

constexpr int STREAM_CORE_FRAMES = 16;

window_stream_shape stream_shape(const codec_model & model) {
    window_stream_shape shape;
    shape.context = synthesis_context_frames(model);
    shape.window = shape.context + STREAM_CORE_FRAMES;
    shape.asynchronous = false;
#ifdef TTS_CPP_USE_COREML
    if (model.coreml) {
        shape.window = static_cast<int>(audio8_coreml_codec_window_frames(model.coreml));
        shape.asynchronous = true;
    }
#endif
    shape.latent_dim = model.hp.latent_dim;
    shape.frame_size = model.hp.frame_size;
    return shape;
}

window_stream_hooks stream_hooks(codec_model & model, const std::vector<int32_t> & frames,
                                 int n_threads, const cancel_hook & cancel) {
    window_stream_hooks hooks;
    const std::shared_ptr<post_history> history = std::make_shared<post_history>();
    hooks.extend_post = [&model, &frames, n_threads, history](int end,
                                                              std::vector<float> & post,
                                                              std::string * error) {
        const std::vector<int32_t> rows =
            codebook_rows(frames, model.hp.num_codebooks, history->done, end);
        return extend_post(model, rows.data(), end, n_threads, *history, post, error);
    };
    const window_stream_shape shape = stream_shape(model);
    hooks.synthesize = [&model, n_threads, shape](const float * post, float * pcm) {
#ifdef TTS_CPP_USE_COREML
        if (model.coreml) return audio8_coreml_codec_synthesize(model.coreml, post, pcm) == 0;
#endif
        const std::vector<float> input(post, post + static_cast<size_t>(shape.window) * shape.latent_dim);
        std::vector<float> output;
        std::string error;
        if (!run_block(model, input, {0, shape.window, 0}, n_threads, output, nullptr, &error)) return false;
        std::copy(output.begin(), output.end(), pcm);
        return true;
    };
    hooks.cancelled = cancel;
    return hooks;
}

void note_streamed_synthesis(const codec_model & model, const window_stream & stream,
                             decode_timing & clock) {
    clock.block_frames = stream.window_frames();
    clock.block_scratch = 0;
    clock.streamed_windows = stream.streamed_windows();
#ifdef TTS_CPP_USE_COREML
    clock.synthesis_backend = model.coreml ? audio8_coreml_codec_backend_label(model.coreml) : "ggml";
#else
    (void) model;
#endif
}

bool fall_back_from_stream(codec_model & model, const int32_t * codes, int n_frames,
                           int n_threads, const cancel_hook & cancel,
                           std::vector<float> & pcm_out, std::string * error,
                           decode_timing * timing) {
    pcm_out.clear();
    if (timing) *timing = decode_timing{};
#ifdef TTS_CPP_USE_COREML
    retire_coreml_sidecar(model, "prediction failed");
    if (coreml_strict()) {
        if (error) {
            *error = "audio8: Core ML synthesis unavailable for " + std::to_string(n_frames) +
                     " frames and AUDIO8_COREML_STRICT is set; failing instead of the ggml "
                     "fallback";
        }
        return false;
    }
    std::fprintf(stderr, "[audio8] Core ML synthesis unavailable for %d frames; using ggml\n",
                 n_frames);
#endif
    return decode_codes(model, codes, n_frames, n_threads, cancel, pcm_out, error, nullptr,
                        timing);
}

}  // namespace

std::unique_ptr<window_stream> open_synthesis_stream(codec_model & model,
                                                     const std::vector<int32_t> & frames,
                                                     int n_threads, const cancel_hook & cancel,
                                                     const std::function<void(const std::vector<float> &)> & emit) {
    if (!post_has_window(model)) return nullptr;
    bool enabled = static_cast<bool>(emit);
#ifdef TTS_CPP_USE_COREML
    enabled = enabled || (model.coreml && coreml_streaming_enabled());
#endif
    if (!enabled) return nullptr;
    const window_stream_shape shape = stream_shape(model);
    if (shape.context >= shape.window) return nullptr;
    window_stream_hooks hooks = stream_hooks(model, frames, n_threads, cancel);
    hooks.emit = emit;
    return std::unique_ptr<window_stream>(new window_stream(shape, std::move(hooks)));
}

bool finish_synthesis_stream(codec_model & model, window_stream & stream,
                             const int32_t * codes, int n_frames, int n_threads,
                             const cancel_hook & cancel, std::vector<float> & pcm_out,
                             std::string * error, decode_timing * timing) {
    decode_timing discarded;
    decode_timing & clock = timing ? *timing : discarded;
    const double post_before = stream.post_ms();
    double finish_ms = 0.0;
    window_stream_status status = window_stream_status::failed;
    {
        stage_timer measure(finish_ms);
        status = stream.finish(n_frames, pcm_out, error);
    }
    clock.latent_ms = stream.post_ms();
    clock.synthesis_ms = stream.asynchronous()
        ? finish_ms - (stream.post_ms() - post_before) : stream.synthesis_ms();
    switch (status) {
        case window_stream_status::done:
            note_streamed_synthesis(model, stream, clock);
            return true;
        case window_stream_status::cancelled:
            if (error) *error = CANCELLED;
            return false;
        case window_stream_status::failed:
            return false;
        case window_stream_status::unavailable:
            break;
    }
    if (stream.emitted()) {
        if (error) *error = "audio8: codec failed after PCM delivery began";
        return false;
    }
    return fall_back_from_stream(model, codes, n_frames, n_threads, cancel, pcm_out, error,
                                 timing);
}

// Fit measurement (include/tts-cpp/audio8/fit.h): price what one decode_codes
// leaves resident, without allocating. The latent graph is the full-sequence
// pass run_latents makes; the synthesis block is the width the runtime's own
// planner (plan_blocks, itself already size-only) would settle on against the
// device memory available right now, priced through the same dual-path
// dispatch prepare_graph allocates with. The two arenas (model.allocr /
// model.block_allocr) are separate and both stay resident, so they sum.
bool measure_decode_memory(codec_model & model, int n_frames, codec_fit_measure & out) {
    using ::tts_cpp::fitutil::sat_add;
    out = codec_fit_measure{};
    if (!model.has_decoder || n_frames <= 0) return false;

    {
        scratch work(AUDIO8_MAX_NODES);
        if (!work.ok()) return false;
        const latent_graph built = build_latents(work.ctx, model, n_frames);
        mark_output(work.graph, built.post);
        ::tts_cpp::detail::fit_graph_price price;
        if (!fit_price_graph(model.backend, work.graph, 2 * AUDIO8_MAX_NODES, price)) {
            return false;
        }
        out.device_bytes = sat_add(out.device_bytes, price.device_bytes);
        out.host_bytes   = sat_add(out.host_bytes, price.host_bytes);
    }

    const int context = synthesis_context(model);
    const block_plan plan = plan_blocks(model, context, n_frames, /*with_taps=*/false);
    out.block_frames = plan.frames;
    out.block_span   = span_of(plan.frames, n_frames, context);
    {
        scratch work(AUDIO8_MAX_NODES);
        if (!work.ok()) return false;
        const synthesis_graph built = build_synthesis(work.ctx, model, out.block_span);
        mark_output(work.graph, built.pcm);
        ::tts_cpp::detail::fit_graph_price price;
        if (!fit_price_graph(model.backend, work.graph, 2 * AUDIO8_MAX_NODES, price)) {
            return false;
        }
        out.device_bytes = sat_add(out.device_bytes, price.device_bytes);
        out.host_bytes   = sat_add(out.host_bytes, price.host_bytes);
    }
    return true;
}

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
