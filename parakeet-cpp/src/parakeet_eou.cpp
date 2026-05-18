// EOU runtime preparation and greedy decoding.

#include "parakeet_eou.h"
#include "parakeet_log.h"
#include "sentencepiece_bpe.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace parakeet {

namespace {

void dequantize_to_f32(const ggml_tensor * t, std::vector<float> & out) {
    if (!t) throw std::runtime_error("eou_prepare_runtime: missing tensor");
    const size_t n = (size_t) ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        return;
    }
    const auto * tr = ggml_get_type_traits(t->type);
    if (!tr || !tr->to_float) {
        throw std::runtime_error(std::string("eou_prepare_runtime: no to_float for type ") +
                                 ggml_type_name(t->type));
    }
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> host_raw(nbytes);
    ggml_backend_tensor_get(t, host_raw.data(), 0, nbytes);
    tr->to_float(host_raw.data(), out.data(), (int64_t) n);
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// See `parakeet_tdt.cpp::gemv_f32` for the same vectorisation pattern
// + rationale. EOU is the same shape on a smaller weight matrix
// (1L LSTM 640 vs TDT 2L LSTM 640) but is the only path that runs
// per-token at low single-digit ms latency targets, so the SIMD
// pragma matters here too.
void gemv_f32(const float * __restrict W, const float * __restrict x,
              const float * __restrict b, float * __restrict y,
              int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; ++i) {
        const float * __restrict row = W + (size_t) i * in_dim;
        float acc = b ? b[i] : 0.0f;
        #pragma GCC ivdep
        for (int j = 0; j < in_dim; ++j) acc += row[j] * x[j];
        y[i] = acc;
    }
}

void gemv_add_f32(const float * __restrict W, const float * __restrict x,
                  float * __restrict y, int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; ++i) {
        const float * __restrict row = W + (size_t) i * in_dim;
        float acc = 0.0f;
        #pragma GCC ivdep
        for (int j = 0; j < in_dim; ++j) acc += row[j] * x[j];
        y[i] += acc;
    }
}

// `layer_input_scratch` lifted out of the per-call allocator. EOU
// emits ~250 tokens per 11s utterance, each calling lstm_step once,
// so a single-decode allocation count of 250 vs. 1 at H_pred=640
// (2.5 KB each) shaves ~1.6 MB of malloc/free traffic off the inner
// loop. Byte-equal output: each call resizes to H and the inner gemv
// writes every byte before reading, so no stale state can leak
// between calls.
void lstm_step(const EouRuntimeWeights & W,
               const float * __restrict x_input,
               float * __restrict h_state,
               float * __restrict c_state,
               std::vector<float> & scratch,
               std::vector<float> & layer_input_scratch) {
    const int H = W.H_pred;
    const int L = W.L;
    const int G = 4 * H;

    scratch.resize((size_t) G);
    layer_input_scratch.resize((size_t) H);

    const float * x = x_input;

    for (int layer = 0; layer < L; ++layer) {
        const auto & w = W.lstm[layer];
        const float * h_l = h_state + (size_t) layer * H;
        float * c_l = c_state + (size_t) layer * H;

        gemv_f32(w.w_ih.data(), x, w.b_ih.data(), scratch.data(), G, H);
        for (int i = 0; i < G; ++i) scratch[i] += w.b_hh[i];
        gemv_add_f32(w.w_hh.data(), h_l, scratch.data(), G, H);

        float * h_new = (float *) (h_state + (size_t) layer * H);
        for (int i = 0; i < H; ++i) {
            const float i_g = sigmoid(scratch[0 * H + i]);
            const float f_g = sigmoid(scratch[1 * H + i]);
            const float g_g = std::tanh(scratch[2 * H + i]);
            const float o_g = sigmoid(scratch[3 * H + i]);
            const float c_new = f_g * c_l[i] + i_g * g_g;
            c_l[i] = c_new;
            h_new[i] = o_g * std::tanh(c_new);
        }

        std::memcpy(layer_input_scratch.data(), h_new, (size_t) H * sizeof(float));
        x = layer_input_scratch.data();
    }
}

// `tmp_scratch` lifted out of the per-call allocator. Same rationale
// as `lstm_step`: ~250 calls per utterance, each previously
// fresh-allocating an H-sized vector. gemv_f32 writes every output
// byte before any read, so re-using the buffer is byte-equal to the
// per-call allocation.
void joint_step(const EouRuntimeWeights & W,
                const float * __restrict enc,
                const float * __restrict pred,
                std::vector<float> & hidden,
                std::vector<float> & logits,
                std::vector<float> & tmp_scratch) {
    const int H   = W.H_joint;
    const int De  = W.D_enc;
    const int Hp  = W.H_pred;
    const int Vp1 = W.V_plus_1;

    hidden.resize(H);
    tmp_scratch.resize(H);
    gemv_f32(W.joint_enc_w.data(),  enc,  W.joint_enc_b.data(),  hidden.data(), H, De);
    gemv_f32(W.joint_pred_w.data(), pred, W.joint_pred_b.data(), tmp_scratch.data(), H, Hp);
    for (int i = 0; i < H; ++i) hidden[i] += tmp_scratch[i];
    for (int i = 0; i < H; ++i) hidden[i] = std::max(0.0f, hidden[i]);

    logits.resize(Vp1);
    gemv_f32(W.joint_out_w.data(), hidden.data(), W.joint_out_b.data(), logits.data(), Vp1, H);
}

int argmax_f32(const float * data, int n) {
    int best = 0;
    float best_val = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > best_val) { best_val = data[i]; best = i; }
    }
    return best;
}

void replace_sentencepiece_space(std::string & piece) {
    static const std::string sp = "\xe2\x96\x81";
    size_t pos = 0;
    while ((pos = piece.find(sp, pos)) != std::string::npos) {
        piece.replace(pos, sp.size(), " ");
        ++pos;
    }
}

std::string trim_spaces(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return std::string();
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

}

int eou_prepare_runtime(const ParakeetCtcModel & model, EouRuntimeWeights & W) {
    if (model.model_type != ParakeetModelType::EOU) {
        return 1;
    }
    W.H_pred   = model.encoder_cfg.eou_pred_hidden;
    W.H_joint  = model.encoder_cfg.eou_joint_hidden;
    W.D_enc    = model.encoder_cfg.d_model;
    W.L        = model.encoder_cfg.eou_pred_rnn_layers;
    W.V_plus_1 = (int) model.vocab_size + 1;
    W.blank_id = (int) model.blank_id;
    W.eou_id   = model.eou_id >= 0 ? model.eou_id : (int) model.vocab_size - 2;
    W.eob_id   = model.eob_id >= 0 ? model.eob_id : (int) model.vocab_size - 1;

    dequantize_to_f32(model.eou.predict_embed, W.embed);

    W.lstm.clear();
    W.lstm.resize(W.L);
    for (int l = 0; l < W.L; ++l) {
        dequantize_to_f32(model.eou.lstm[l].w_ih, W.lstm[l].w_ih);
        dequantize_to_f32(model.eou.lstm[l].w_hh, W.lstm[l].w_hh);
        dequantize_to_f32(model.eou.lstm[l].b_ih, W.lstm[l].b_ih);
        dequantize_to_f32(model.eou.lstm[l].b_hh, W.lstm[l].b_hh);
    }

    dequantize_to_f32(model.eou.joint_enc_w,  W.joint_enc_w);
    dequantize_to_f32(model.eou.joint_enc_b,  W.joint_enc_b);
    dequantize_to_f32(model.eou.joint_pred_w, W.joint_pred_w);
    dequantize_to_f32(model.eou.joint_pred_b, W.joint_pred_b);
    dequantize_to_f32(model.eou.joint_out_w,  W.joint_out_w);
    dequantize_to_f32(model.eou.joint_out_b,  W.joint_out_b);

    return 0;
}

void eou_init_state(const EouRuntimeWeights & W, EouDecodeState & state) {
    const int H = W.H_pred;
    const int L = W.L;

    state.h_state.assign((size_t) L * H, 0.0f);
    state.c_state.assign((size_t) L * H, 0.0f);
    state.pred_out.assign(H, 0.0f);
    state.last_token        = -1;
    state.symbols_this_step = 0;
    state.segment_start_token = 0;

    // Prime the predictor with the blank token so the first joint call
    // sees a sensible context (matches NeMo's `initialize_state` +
    // first decoder forward with `targets=[blank]`).
    std::vector<float> scratch;
    std::vector<float> layer_input_scratch;
    const float * embed_row = W.embed.data() + (size_t) W.blank_id * H;
    lstm_step(W, embed_row, state.h_state.data(), state.c_state.data(),
              scratch, layer_input_scratch);
    std::memcpy(state.pred_out.data(),
                state.h_state.data() + (size_t) (L - 1) * H,
                (size_t) H * sizeof(float));

    state.initialized = true;
}

int eou_decode_window(const ParakeetCtcModel & model,
                      const EouRuntimeWeights & W,
                      const float * encoder_out_window,
                      int n_frames, int D_enc,
                      const EouDecodeOptions & opts,
                      EouDecodeState & state,
                      std::vector<int32_t> & out_tokens,
                      std::vector<EouSegmentBoundary> & out_segments,
                      int & out_steps) {
    if (D_enc != W.D_enc) {
        PARAKEET_LOG_ERROR("eou_decode_window: encoder d_model mismatch (%d vs %d)\n",
                           D_enc, W.D_enc);
        return 1;
    }
    if (!state.initialized) {
        eou_init_state(W, state);
    }

    const int H     = W.H_pred;
    const int L     = W.L;
    const int V_p1  = W.V_plus_1;
    const int blank = W.blank_id;
    const int eou   = W.eou_id;
    const int eob   = W.eob_id;
    const int max_syms = std::max(1, opts.max_symbols_per_step);

    std::vector<float> scratch_lstm;
    std::vector<float> scratch_lstm_layer_input;
    std::vector<float> scratch_joint_tmp;
    std::vector<float> scratch_joint_hidden;
    std::vector<float> scratch_joint_logits;

    out_steps = 0;
    const size_t n_vocab = model.vocab.pieces.size();

    for (int t = 0; t < n_frames; ++t) {
        const float * enc_frame = encoder_out_window + (size_t) t * D_enc;
        state.symbols_this_step = 0;

        while (state.symbols_this_step < max_syms) {
            joint_step(W, enc_frame, state.pred_out.data(),
                       scratch_joint_hidden, scratch_joint_logits,
                       scratch_joint_tmp);
            ++out_steps;

            const int best = argmax_f32(scratch_joint_logits.data(), V_p1);
            if (best == blank) {
                break;
            }

            // <EOB>: training-time block boundary; treat as a no-op skip.
            if (best == eob) {
                break;
            }

            // <EOU>: flush the current segment, reset LSTM state, drop
            // back to the blank token as the predictor input. Match the
            // NeMo `eouDecodeChunk` reference: do NOT feed `<EOU>`
            // back into the predictor; reset h/c to zero and lastToken
            // to blank.
            if (best == eou) {
                if ((int) out_tokens.size() > state.segment_start_token) {
                    EouSegmentBoundary boundary;
                    boundary.token_index  = (int) out_tokens.size();
                    boundary.is_eou_flush = true;
                    out_segments.push_back(boundary);
                    state.segment_start_token = (int) out_tokens.size();
                }
                state.h_state.assign((size_t) L * H, 0.0f);
                state.c_state.assign((size_t) L * H, 0.0f);
                state.last_token = blank;
                const float * embed_row = W.embed.data() + (size_t) blank * H;
                lstm_step(W, embed_row, state.h_state.data(),
                          state.c_state.data(), scratch_lstm,
                          scratch_lstm_layer_input);
                std::memcpy(state.pred_out.data(),
                            state.h_state.data() + (size_t) (L - 1) * H,
                            (size_t) H * sizeof(float));
                break;
            }

            // Skip any other special token defensively (e.g. <unk>);
            // any vocab piece wrapped in `<...>` is treated as special.
            if (best >= 0 && (size_t) best < n_vocab) {
                const std::string & piece = model.vocab.pieces[best];
                if (!piece.empty() && piece.front() == '<' && piece.back() == '>') {
                    break;
                }
            }

            out_tokens.push_back((int32_t) best);

            const float * embed_row = W.embed.data() + (size_t) best * H;
            lstm_step(W, embed_row, state.h_state.data(),
                      state.c_state.data(), scratch_lstm,
                      scratch_lstm_layer_input);
            std::memcpy(state.pred_out.data(),
                        state.h_state.data() + (size_t) (L - 1) * H,
                        (size_t) H * sizeof(float));
            state.last_token = best;
            ++state.symbols_this_step;
        }
    }
    return 0;
}

int eou_greedy_decode(const ParakeetCtcModel & model,
                      const EouRuntimeWeights & W,
                      const float * encoder_out,
                      int T_enc, int D_enc,
                      const EouDecodeOptions & opts,
                      EouDecodeResult & result) {
    const auto t0 = std::chrono::steady_clock::now();

    EouDecodeState state;
    result.token_ids.clear();
    result.segments.clear();
    result.token_ids.reserve(T_enc);

    if (int rc = eou_decode_window(model, W, encoder_out, T_enc, D_enc,
                                   opts, state,
                                   result.token_ids, result.segments,
                                   result.steps);
        rc != 0) {
        return rc;
    }

    result.eou_count = (int) result.segments.size();

    int seg_start = 0;
    std::string out_text;
    for (size_t i = 0; i <= result.segments.size(); ++i) {
        const int seg_end = (i < result.segments.size())
                              ? result.segments[i].token_index
                              : (int) result.token_ids.size();
        if (seg_end <= seg_start) {
            seg_start = seg_end;
            continue;
        }
        std::vector<int32_t> seg_tokens(
            result.token_ids.begin() + seg_start,
            result.token_ids.begin() + seg_end);
        std::string seg_text = detokenize(model.vocab, seg_tokens);
        seg_text = trim_spaces(seg_text);
        if (!seg_text.empty()) {
            if (!out_text.empty()) out_text += "\n";
            out_text += seg_text;
        }
        seg_start = seg_end;
    }
    result.text = std::move(out_text);

    const auto t1 = std::chrono::steady_clock::now();
    result.decode_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    return 0;
}

}
