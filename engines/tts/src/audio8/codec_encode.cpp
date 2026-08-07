// Audio8 codec analysis: a reference waveform in, ten codebook rows per frame
// out, which is what enrolling a voice needs.
//
// The DAC stack reduces 44.1 kHz mono by 512, a windowed transformer runs over
// that, two stride-2 stages take it to one column per 2048 samples, and a
// residual quantizer walks ten codebooks over what is left. Because each
// quantizer subtracts its own reconstruction before the next one looks, the
// whole cascade -- nearest-neighbour search included -- is one graph.
//
// As in synthesis, it takes two graphs rather than one, and for the same
// reason. The convolutional stack still works at the sample rate, so its
// activations dwarf everything downstream, and it is causal with a context of
// a few thousand samples, so it runs in blocks. Everything from the encoder's
// own transformer on runs at one column per 512 samples and has to see the
// whole reference at once anyway, its four windowed layers reaching 512
// columns back each.

#include "audio8/codec_ops.h"
#include "audio8/graph.h"

#include <algorithm>
#include <string>
#include <vector>

namespace tts_cpp {
namespace audio8 {
namespace detail {
namespace {

const char * const AUDIO_INPUT = "audio";
const char * const FEATURE_INPUT = "features";
const char * const ENCODER_MASK_INPUT = "enc_mask";
const char * const PRE_MASK_INPUT = "pre_mask";

// F.normalize's floor on the vector norm. Both sides of the nearest-neighbour
// search come out unit length, so the search is over cosine similarity: the
// squared norms the reference also carries are 1 for every candidate and
// cannot move the argmax.
constexpr float NORMALIZE_EPS = 1e-12f;

struct analysis_graph {
    ggml_tensor * encoded = nullptr;
    ggml_tensor * downsampled = nullptr;
    std::vector<ggml_tensor *> codes;
};

const dac_stage * transformer_stage(const codec_model & model) {
    for (const dac_stage & stage : model.enc_stages) {
        if (stage.has_transformer) return &stage;
    }
    return nullptr;
}

int total_encoder_stride(const codec_hparams & hp) {
    int stride = 1;
    for (int value : hp.encoder_strides) stride *= value;
    return stride;
}

// The encoder's own transformer runs one column per 512 samples, four per
// frame, so it is the first RoPE table to run out.
int encoder_positions(const codec_hparams & hp, int n_frames) {
    return n_frames * hp.frame_size / total_encoder_stride(hp);
}

// What the convolutional stack hands the transformer, which is also what a
// block of it has to declare as its output width.
int feature_width(const codec_model & model) {
    return static_cast<int>(model.enc_stages.back().conv.w->ne[1]);
}

// The columns of history one block has to be handed before its own. The walk
// is in samples and the blocks are cut on column boundaries, so it rounds up.
int analysis_context(const codec_model & model) {
    int span = 1;
    for (size_t index = model.enc_stages.size(); index-- > 0;) {
        const dac_stage & stage = model.enc_stages[index];
        span = span_through_conv(span, conv_taps(stage.conv), stage.stride, /*dilation=*/1);
        span = span_through_units(span, stage.units, model.hp.residual_dilations);
    }
    span = span_through_conv(span, conv_taps(model.enc_in), 1, /*dilation=*/1);
    const int stride = total_encoder_stride(model.hp);
    return (span - 1 + stride - 1) / stride;
}

ggml_tensor * run_conv_stack(ggml_context * ctx, const codec_model & model,
                             ggml_tensor * signal) {
    const codec_hparams & hp = model.hp;
    signal = causal_conv(ctx, model.enc_in, signal, /*stride=*/1, /*dilation=*/1);
    for (const dac_stage & stage : model.enc_stages) {
        signal = residual_stack(ctx, stage.units, hp.residual_dilations, signal,
                                hp.snake_epsilon);
        signal = snake(ctx, signal, stage.alpha, hp.snake_epsilon);
        signal = causal_conv(ctx, stage.conv, signal, stage.stride, /*dilation=*/1);
    }
    return signal;
}

ggml_tensor * run_encoder_tail(ggml_context * ctx, const codec_model & model,
                               ggml_tensor * signal, ggml_tensor * mask) {
    signal = window_forward(ctx, transformer_stage(model)->transformer, signal, mask);
    signal = snake(ctx, signal, model.enc_out_alpha, model.hp.snake_epsilon);
    return causal_conv(ctx, model.enc_out, signal, /*stride=*/1, /*dilation=*/1);
}

ggml_tensor * run_downsample(ggml_context * ctx, const std::vector<resample_stage> & stages,
                             ggml_tensor * signal, float eps) {
    for (const resample_stage & stage : stages) {
        signal = causal_conv(ctx, stage.conv, signal, stage.stride, /*dilation=*/1);
        signal = convnext_block(ctx, stage.convnext, signal, eps);
    }
    return signal;
}

ggml_tensor * unit_length(ggml_context * ctx, ggml_tensor * x) {
    return ggml_rms_norm(ctx, x, NORMALIZE_EPS);
}

ggml_tensor * nearest_codes(ggml_context * ctx, const quantizer_weights & quantizer,
                            ggml_tensor * latents) {
    ggml_tensor * similarity = ggml_mul_mat(ctx, unit_length(ctx, quantizer.codebook),
                                            unit_length(ctx, latents));
    ggml_mul_mat_set_prec(similarity, GGML_PREC_F32);
    return ggml_argmax(ctx, similarity);
}

// The reference passes on `latents + (chosen - latents)` rather than the
// chosen vector itself, which is the straight-through estimator left over from
// training. The two differ in the last bit, so the port keeps the detour.
ggml_tensor * straight_through(ggml_context * ctx, ggml_tensor * latents,
                               ggml_tensor * chosen) {
    return ggml_add(ctx, latents, ggml_sub(ctx, chosen, latents));
}

struct quantized_step {
    ggml_tensor * codes = nullptr;
    ggml_tensor * reconstruction = nullptr;
};

quantized_step quantize(ggml_context * ctx, const quantizer_weights & quantizer,
                        ggml_tensor * x) {
    ggml_tensor * latents = project(ctx, quantizer.in_proj, x);
    quantized_step step;
    step.codes = nearest_codes(ctx, quantizer, latents);
    ggml_tensor * chosen = ggml_get_rows(ctx, quantizer.codebook, step.codes);
    step.reconstruction =
        project(ctx, quantizer.out_proj, straight_through(ctx, latents, chosen));
    return step;
}

// Each codebook only sees what the ones before it could not explain.
ggml_tensor * quantize_bank(ggml_context * ctx, const std::vector<quantizer_weights> & bank,
                            ggml_tensor * residual, std::vector<ggml_tensor *> & codes) {
    for (const quantizer_weights & quantizer : bank) {
        const quantized_step step = quantize(ctx, quantizer, residual);
        codes.push_back(step.codes);
        residual = ggml_sub(ctx, residual, step.reconstruction);
    }
    return residual;
}

ggml_tensor * build_conv_stack(ggml_context * ctx, const codec_model & model,
                               int n_samples) {
    return run_conv_stack(ctx, model, input_f32(ctx, AUDIO_INPUT, /*ne0=*/1, n_samples));
}

analysis_graph build_analysis(ggml_context * ctx, const codec_model & model, int positions,
                              int n_frames) {
    ggml_tensor * features = input_f32(ctx, FEATURE_INPUT, feature_width(model), positions);
    ggml_tensor * enc_mask = input_f32(ctx, ENCODER_MASK_INPUT, positions, positions);
    ggml_tensor * pre_mask = input_f32(ctx, PRE_MASK_INPUT, n_frames, n_frames);

    analysis_graph built;
    built.encoded = run_encoder_tail(ctx, model, features, enc_mask);
    built.downsampled = window_forward(
        ctx, model.pre,
        run_downsample(ctx, model.downsample, built.encoded, model.hp.convnext_norm_eps),
        pre_mask);

    ggml_tensor * residual = quantize_bank(ctx, model.semantic_quantizers,
                                           built.downsampled, built.codes);
    quantize_bank(ctx, model.residual_quantizers, residual, built.codes);
    return built;
}

// A block covers feature columns [first - context, first + count) and keeps
// only what belongs to [first, first + count): the context is there to give
// the causal convolutions real history instead of the zeros they would
// otherwise pad with.
struct block {
    int begin = 0;
    int count = 0;
    int dropped = 0;
};

block block_at(int first, int positions, int context, int block_columns) {
    block span;
    span.begin = std::max(0, first - context);
    span.dropped = first - span.begin;
    span.count = std::min(block_columns, positions - first) + span.dropped;
    return span;
}

bool run_block(codec_model & model, const std::vector<float> & audio, const block & span,
               int stride, int n_threads, std::vector<float> & features,
               std::string * error) {
    scratch work(AUDIO8_MAX_NODES);
    if (!work.ok()) {
        if (error) *error = "audio8: failed to create the convolution graph context";
        return false;
    }
    ggml_tensor * built = build_conv_stack(work.ctx, model, span.count * stride);
    mark_output(work.graph, built);
    if (!allocate_graph(model.block_allocr, work.graph, "convolution", error)) return false;

    write_input(work.graph, AUDIO_INPUT, audio.data() + span.begin * stride,
                static_cast<size_t>(span.count) * stride * sizeof(float));
    if (!compute_graph(model.backend, work.graph, n_threads, "convolution", error)) {
        return false;
    }

    std::vector<float> produced;
    read_output(built, produced);
    const size_t skip = static_cast<size_t>(span.dropped) * feature_width(model);
    features.insert(features.end(), produced.begin() + skip, produced.end());
    return true;
}

bool run_conv_blocks(codec_model & model, const std::vector<float> & audio, int positions,
                     int n_threads, const cancel_hook & cancel,
                     std::vector<float> & features, std::string * error) {
    features.clear();
    features.reserve(static_cast<size_t>(positions) * feature_width(model));
    const int stride = total_encoder_stride(model.hp);
    const int context = analysis_context(model);
    const int block_columns = std::max(1, model.analysis_block_columns);
    for (int first = 0; first < positions; first += block_columns) {
        if (cancelled(cancel, error)) return false;
        if (!run_block(model, audio, block_at(first, positions, context, block_columns),
                       stride, n_threads, features, error)) {
            return false;
        }
    }
    return true;
}

void set_analysis_inputs(ggml_cgraph * graph, const codec_model & model,
                         const std::vector<float> & features, int positions,
                         int n_frames) {
    write_input(graph, FEATURE_INPUT, features.data(), features.size() * sizeof(float));
    const std::vector<float> encoder =
        window_mask(transformer_stage(model)->transformer.spec, positions);
    write_input(graph, ENCODER_MASK_INPUT, encoder.data(), encoder.size() * sizeof(float));
    const std::vector<float> pre = window_mask(model.pre.spec, n_frames);
    write_input(graph, PRE_MASK_INPUT, pre.data(), pre.size() * sizeof(float));
}

// The reference pads the waveform up to a whole number of frames before the
// encoder ever sees it.
std::vector<float> padded_audio(const float * pcm, int n_samples, int n_frames,
                                int frame_size) {
    std::vector<float> audio(static_cast<size_t>(n_frames) * frame_size, 0.0f);
    std::copy(pcm, pcm + n_samples, audio.begin());
    return audio;
}

void collect_codes(const analysis_graph & built, int n_frames, std::vector<int32_t> & out) {
    out.resize(built.codes.size() * static_cast<size_t>(n_frames));
    std::vector<int32_t> row;
    for (size_t index = 0; index < built.codes.size(); ++index) {
        read_ids(built.codes[index], row);
        std::copy(row.begin(), row.end(),
                  out.begin() + static_cast<long>(index * n_frames));
    }
}

bool run_analysis(codec_model & model, const std::vector<float> & features, int positions,
                  int n_frames, int n_threads, std::vector<int32_t> & codes_out,
                  encode_taps * taps, std::string * error) {
    scratch work(AUDIO8_MAX_NODES);
    if (!work.ok()) {
        if (error) *error = "audio8: failed to create the analysis graph context";
        return false;
    }
    const analysis_graph built = build_analysis(work.ctx, model, positions, n_frames);
    for (ggml_tensor * codes : built.codes) mark_output(work.graph, codes);
    if (taps) {
        mark_output(work.graph, built.encoded);
        mark_output(work.graph, built.downsampled);
    }
    if (!allocate_graph(model.allocr, work.graph, "analysis", error)) return false;

    set_analysis_inputs(work.graph, model, features, positions, n_frames);
    if (!compute_graph(model.backend, work.graph, n_threads, "analysis", error)) {
        return false;
    }

    collect_codes(built, n_frames, codes_out);
    if (taps) {
        read_output(built.encoded, taps->encoded);
        read_output(built.downsampled, taps->downsampled);
    }
    return true;
}

bool check_encodable(const codec_model & model, int n_frames, std::string * error) {
    if (!model.has_encoder) {
        if (error) *error = "audio8: this codec GGUF has no encoder";
        return false;
    }
    if (!transformer_stage(model)) {
        if (error) *error = "audio8: the encoder GGUF has no window transformer";
        return false;
    }
    const int positions = encoder_positions(model.hp, n_frames);
    if (positions > model.hp.max_frames) {
        if (error) {
            *error = "audio8: " + std::to_string(n_frames) + " frames needs " +
                     std::to_string(positions) +
                     " encoder positions, past the baked RoPE table of " +
                     std::to_string(model.hp.max_frames);
        }
        return false;
    }
    return true;
}

}  // namespace

bool encode_audio(codec_model & model, const float * pcm, int n_samples, int n_threads,
                  const cancel_hook & cancel, std::vector<int32_t> & codes_out,
                  int & n_frames, std::string * error, encode_taps * taps) {
    codes_out.clear();
    n_frames = (n_samples + model.hp.frame_size - 1) / model.hp.frame_size;
    if (!check_encodable(model, n_frames, error)) return false;
    if (n_frames <= 0) return true;

    const std::vector<float> audio =
        padded_audio(pcm, n_samples, n_frames, model.hp.frame_size);
    const int positions = encoder_positions(model.hp, n_frames);

    std::vector<float> features;
    if (!run_conv_blocks(model, audio, positions, n_threads, cancel, features, error)) {
        return false;
    }
    if (cancelled(cancel, error)) return false;
    return run_analysis(model, features, positions, n_frames, n_threads, codes_out, taps,
                        error);
}

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
