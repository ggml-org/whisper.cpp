#include "parakeet.h"
#include "parakeet-arch.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <atomic>
#include <algorithm>
#include <cassert>
#include <cfloat>
#define _USE_MATH_DEFINES
#include <cmath>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <cctype>
#include <map>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#include <codecvt>
#endif

#if defined(PARAKEET_BIG_ENDIAN)
template<typename T>
static T byteswap(T value) {
    T value_swapped;
    char * source = reinterpret_cast<char *>(&value);
    char * target = reinterpret_cast<char *>(&value_swapped);
    int size = sizeof(T);
    for (int i = 0; i < size; i++) {
        target[size - 1 - i] = source[i];
    }
    return value_swapped;
}

template<typename T>
static void byteswap_tensor_data(ggml_tensor * tensor) {
    T * datum = reinterpret_cast<T *>(tensor->data);
    for (int i = 0; i < ggml_nelements(tensor); i++) {
        datum[i] = byteswap(datum[i]);
    }
}

static void byteswap_tensor(ggml_tensor * tensor) {
    switch (tensor->type) {
        case GGML_TYPE_I16: {
            byteswap_tensor_data<int16_t>(tensor);
            break;
        }
        case GGML_TYPE_F16: {
            byteswap_tensor_data<ggml_fp16_t>(tensor);
            break;
        }
        case GGML_TYPE_I32: {
            byteswap_tensor_data<int32_t>(tensor);
            break;
        }
        case GGML_TYPE_F32: {
            byteswap_tensor_data<float>(tensor);
            break;
        }
        default: { // GML_TYPE_I8
            break;
        }
    }
}

#define BYTESWAP_VALUE(d) d = byteswap(d)
#define BYTESWAP_FILTERS(f)           \
    do {                              \
        for (auto & datum : f.data) { \
            datum = byteswap(datum);  \
        }                             \
    } while (0)
#define BYTESWAP_TENSOR(t)  \
    do {                    \
        byteswap_tensor(t); \
    } while (0)
#else
#define BYTESWAP_VALUE(d) do {} while (0)
#define BYTESWAP_FILTERS(f) do {} while (0)
#define BYTESWAP_TENSOR(t) do {} while (0)
#endif

#ifdef __GNUC__
#ifdef __MINGW32__
#define PARAKEET_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define PARAKEET_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define PARAKEET_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

PARAKEET_ATTRIBUTE_FORMAT(2, 3)
static void parakeet_log_internal        (ggml_log_level level, const char * format, ...);
static void parakeet_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define PARAKEET_LOG_ERROR(...) parakeet_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define PARAKEET_LOG_WARN(...)  parakeet_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define PARAKEET_LOG_INFO(...)  parakeet_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)

// define this to enable verbose trace logging - useful for debugging purposes
//#define PARAKEET_DEBUG

#if defined(PARAKEET_DEBUG)
#define PARAKEET_LOG_DEBUG(...) parakeet_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#else
#define PARAKEET_LOG_DEBUG(...)
#endif

#define PARAKEET_ASSERT(x) \
    do { \
        if (!(x)) { \
            PARAKEET_LOG_ERROR("PARAKEET_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

#define PARAKEET_MAX_NODES 4096

static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

//
// ggml helpers
//

static bool ggml_graph_compute_helper(
          struct ggml_cgraph * graph,
                         int   n_threads,
         ggml_abort_callback   abort_callback,
                        void * abort_callback_data) {
    ggml_backend_ptr backend { ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr) };

    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));

    auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
    if (set_abort_callback_fn) {
        set_abort_callback_fn(backend.get(), abort_callback, abort_callback_data);
    }

    auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (ggml_backend_set_n_threads_fn) {
        ggml_backend_set_n_threads_fn(backend.get(), n_threads);
    }

    return ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS;
}

static bool ggml_graph_compute_helper(
      ggml_backend_sched_t   sched,
        struct ggml_cgraph * graph,
                       int   n_threads,
                      bool   sched_reset = true) {
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;

        auto * fn_set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (fn_set_n_threads) {
            fn_set_n_threads(backend, n_threads);
        }
    }

    const bool t = (ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);

    if (!t || sched_reset) {
        ggml_backend_sched_reset(sched);
    }

    return t;
}

// TODO: move these functions to ggml-base with support for ggml-backend?

static ggml_tensor * parakeet_set_f32(struct ggml_tensor * t, float v) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(t));
    size_t nels = ggml_nelements(t);
    for (size_t i = 0; i < nels; ++i) {
        ((float *) t->data)[i] = v;
    }
    return t;
}

static ggml_tensor * parakeet_set_i32(struct ggml_tensor * t, int32_t v) {
    GGML_ASSERT(t->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(t));
    size_t nels = ggml_nelements(t);
    for (size_t i = 0; i < nels; ++i) {
        ((int32_t *) t->data)[i] = v;
    }
    return t;
}

static float parakeet_get_f32_nd(const struct ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    void * data = (char *) t->data + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
    return *(float *) data;
}

static void parakeet_set_f32_nd(struct ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, float v) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    void * data = (char *) t->data + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
    *(float *) data = v;
}

static int32_t parakeet_get_i32_nd(const struct ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    GGML_ASSERT(t->type == GGML_TYPE_I32);
    void * data = (char *) t->data + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
    return *(int32_t *) data;
}

static void parakeet_set_i32_nd(struct ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, int32_t v) {
    GGML_ASSERT(t->type == GGML_TYPE_I32);
    void * data = (char *) t->data + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
    *(int32_t *) data = v;
}

struct parakeet_mel {
    int n_len     = 0;
    int n_len_org = 0;
    int n_mel     = 0;

    std::vector<float> data;
};

struct parakeet_filters {
    int32_t n_mel = 0;
    int32_t n_fb  = 0;  // number of frequency bins

    std::vector<float> data;
};

struct parakeet_vocab {
    using id    = int32_t;
    using token = std::string;

    int n_vocab = 8192;
    size_t max_token_length = 0;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;

    id token_unk;
    id token_bos;
    id token_blank;
    id token_eos;
};

struct parakeet_segment {
    int64_t t0;
    int64_t t1;

    std::string text;

    std::vector<parakeet_token_data> tokens;
};

struct parakeet_batch {
    int32_t n_tokens;

    parakeet_token  *  token;
    int32_t         *  i_time;   // index of the audio frame
    parakeet_pos    *  pos;
    int32_t         *  n_seq_id; // always 1, here for consistency with llama.cpp
    parakeet_seq_id ** seq_id;   // null terminated
    int8_t          *  logits;
};

// ggml_backend_sched wrapper for parakeet usage
struct parakeet_sched {
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> meta;
};

// TODO: Find out is there a multiple version types. It is not yet clear to me
// at this point.
enum parakeet_arch {
    PARAKEET_ARCH_UNKNOWN = 0,
    PARAKEET_ARCH_TDT     = 1,  // NVIDIA Parakeet TDT (RNN-T)
};

struct parakeet_hparams {
    int32_t n_vocab                = 8192;
    int32_t n_audio_ctx            = 0;  // 0 = unlimited, will be set based on input
    int32_t n_audio_state          = 1024;
    int32_t n_audio_head           = 8;
    int32_t n_audio_layer          = 24;
    int32_t n_mels                 = 128;
    int32_t ftype                  = 1;
    int32_t n_fft                  = 512;  // FFT size for mel spectrogram
    float   eps                    = 1e-5f;
    int32_t subsampling_factor     = 8;
    int32_t n_subsampling_channels = 256;
    int32_t n_pos_max_len          = 5000;
    int32_t n_pred_dim             = 640;
    int32_t n_pred_layers          = 2;
    int32_t n_tdt_durations        = 5;
    int32_t n_max_tokens           = 10;

    parakeet_arch arch     = PARAKEET_ARCH_TDT;
};

struct parakeet_layer_encoder {
    struct ggml_tensor * norm_ff1_w = nullptr;
    struct ggml_tensor * norm_ff1_b = nullptr;

    struct ggml_tensor * ff1_linear1_w = nullptr;
    struct ggml_tensor * ff1_linear2_w = nullptr;

    struct ggml_tensor * norm_conv_w = nullptr;
    struct ggml_tensor * norm_conv_b = nullptr;

    struct ggml_tensor * conv_pw1_w          = nullptr;  // pointwise_conv1
    struct ggml_tensor * conv_dw_w           = nullptr;  // depthwise_conv
    struct ggml_tensor * conv_bn_w           = nullptr;  // batch_norm weight
    struct ggml_tensor * conv_bn_b           = nullptr;  // batch_norm bias
    struct ggml_tensor * conv_bn_mean        = nullptr;  // batch_norm running_mean
    struct ggml_tensor * conv_bn_var         = nullptr;  // batch_norm running_var
    struct ggml_tensor * conv_bn_num_batches = nullptr;  // batch_norm num_batches_tracked
    struct ggml_tensor * conv_pw2_w          = nullptr;  // pointwise_conv2

    struct ggml_tensor * norm_attn_w = nullptr;
    struct ggml_tensor * norm_attn_b = nullptr;

    struct ggml_tensor * attn_pos_bias_u = nullptr;
    struct ggml_tensor * attn_pos_bias_v = nullptr;
    struct ggml_tensor * attn_q_w        = nullptr;
    struct ggml_tensor * attn_k_w        = nullptr;
    struct ggml_tensor * attn_v_w        = nullptr;
    struct ggml_tensor * attn_out_w      = nullptr;
    struct ggml_tensor * attn_pos_w      = nullptr;

    struct ggml_tensor * norm_ff2_w      = nullptr;
    struct ggml_tensor * norm_ff2_b      = nullptr;

    struct ggml_tensor * ff2_linear1_w = nullptr;
    struct ggml_tensor * ff2_linear2_w = nullptr;

    struct ggml_tensor * norm_out_w = nullptr;
    struct ggml_tensor * norm_out_b = nullptr;
};

struct parakeet_lsmt_layer {
    struct ggml_tensor * ih_w = nullptr;  // input-to-hidden weight
    struct ggml_tensor * ih_b = nullptr;  // input-to-hidden bias
    struct ggml_tensor * hh_w = nullptr;  // hidden-to-hidden weight
    struct ggml_tensor * hh_b = nullptr;  // hidden-to-hidden bias
};

struct parakeet_prediction_network {
    struct ggml_tensor * embed_w = nullptr;

    std::vector<parakeet_lsmt_layer> lstm_layer;
};

struct parakeet_joint_network {
    struct ggml_tensor * pred_w = nullptr;
    struct ggml_tensor * pred_b = nullptr;
    struct ggml_tensor * enc_w  = nullptr;
    struct ggml_tensor * enc_b  = nullptr;
    struct ggml_tensor * net_w  = nullptr;
    struct ggml_tensor * net_b  = nullptr;
};

struct parakeet_model {
    parakeet_filters filters;
    parakeet_hparams hparams;

    // Relative positional encoding (pe)
    struct ggml_tensor * pe = nullptr;

    struct ggml_tensor * enc_pre_out_w    = nullptr;
    struct ggml_tensor * enc_pre_out_b    = nullptr;
    struct ggml_tensor * enc_pre_conv_0_w = nullptr;
    struct ggml_tensor * enc_pre_conv_0_b = nullptr;
    struct ggml_tensor * enc_pre_conv_2_w = nullptr;
    struct ggml_tensor * enc_pre_conv_2_b = nullptr;
    struct ggml_tensor * enc_pre_conv_3_w = nullptr;
    struct ggml_tensor * enc_pre_conv_3_b = nullptr;
    struct ggml_tensor * enc_pre_conv_5_w = nullptr;
    struct ggml_tensor * enc_pre_conv_5_b = nullptr;
    struct ggml_tensor * enc_pre_conv_6_w = nullptr;
    struct ggml_tensor * enc_pre_conv_6_b = nullptr;

    std::vector<parakeet_layer_encoder> layers;

    parakeet_prediction_network prediction;

    parakeet_joint_network joint;

    std::vector<uint32_t> tdt_durations;

    std::vector<ggml_context *> ctxs;

    std::vector<ggml_backend_buffer_t> buffers;

    int n_loaded = 0;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct parakeet_lstm_state_layer {
    struct ggml_tensor * h_state = nullptr;
    struct ggml_tensor * c_state = nullptr;
};

struct parakeet_lstm_state {
    std::vector<parakeet_lstm_state_layer> layer;

    std::vector<uint8_t> ctx_buf;

    ggml_backend_buffer_t buffer = nullptr;
};

struct tdt_stream_state {
    parakeet_token last_token;  // last emitted token (for predictor input)
    int32_t time_step;          // overflow frames into next chunk
    int32_t decoded_length;     // total tokens decoded
    bool initialized;           // whether prediction LSTM state has been initialized
};

struct parakeet_stream {
    std::vector<float> buffer;
    int64_t n_samples_advanced = 0;

    int n_left_ctx  = 0;
    int n_chunk     = 0;
    int n_right_ctx = 0;

    parakeet_full_params params = {};
    bool initialized = false;
};

struct parakeet_state {
    int64_t t_sample_us = 0;
    int64_t t_encode_us = 0;
    int64_t t_decode_us = 0;
    int64_t t_mel_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_encode = 0; // number of encoder calls
    int32_t n_decode = 0; // number of decoder calls with n_tokens == 1  (text-generation)
    int32_t n_fail_p = 0; // number of logprob threshold failures
    int32_t n_fail_h = 0; // number of entropy threshold failures

    parakeet_mel mel;

    parakeet_batch batch;

    int n_frames = 0;

    std::vector<ggml_backend_t> backends;

    parakeet_sched sched_conv;
    parakeet_sched sched_encode;
    parakeet_sched sched_decode;

    // outputs from encoder stages
    struct ggml_tensor * pre_enc_out = nullptr;
    struct ggml_tensor * enc_out     = nullptr;
    struct ggml_tensor * pred_out    = nullptr;

    std::vector<uint8_t> pred_out_buf;
    ggml_backend_buffer_t pred_out_buffer = nullptr;

    struct ggml_tensor * attn_mask = nullptr;

    std::vector<float> inp_mel;
    std::vector<float> inp_mask;

    std::vector<float> enc_out_buffer;
    int enc_out_frames = 0;

    std::vector<float> logits;

    std::vector<parakeet_segment> result_all;

    std::vector<parakeet_token>      decoded_tokens;
    std::vector<parakeet_token_data> decoded_token_data;

    std::string path_model;

    int32_t n_audio_ctx = 0;

    parakeet_lstm_state lstm_state;

    struct tdt_stream_state tdt_stream_state = {0, 0, 0, false};

    parakeet_stream stream;
};

// FFT cache for mel spectrogram computation
struct parakeet_mel_cache {
    int n_fft = 0;

    // In FFT, we frequently use sine and cosine operations with the same values.
    // We can use precalculated values to speed up the process.
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;

    // Hann window (Use cosf to eliminate difference)
    // ref: https://pytorch.org/docs/stable/generated/torch.hann_window.html
    // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L147
    std::vector<float> hann_window;

    // Window function from model (Parakeet uses actual window from training)
    std::vector<float> window;

    void init(int fft_size) {
        n_fft = fft_size;
        sin_vals.resize(n_fft);
        cos_vals.resize(n_fft);
        hann_window.resize(n_fft);

        fill_sin_cos_table();
        fill_hann_window(n_fft, true, hann_window.data());
    }

    void fill_sin_cos_table() {
        for (int i = 0; i < n_fft; i++) {
            double theta = (2 * M_PI * i) / n_fft;
            sin_vals[i] = sinf(theta);
            cos_vals[i] = cosf(theta);
        }
    }

    void fill_hann_window(int length, bool periodic, float * output) {
        int offset = -1;
        if (periodic) {
            offset = 0;
        }
        for (int i = 0; i < length; i++) {
            output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
        }
    }
};

struct parakeet_context {
    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;

    ggml_type wtype = ggml_type::GGML_TYPE_F16;
    ggml_type itype = ggml_type::GGML_TYPE_F16;

    parakeet_context_params params;

    parakeet_model model;
    parakeet_vocab vocab;

    parakeet_state * state = nullptr;

    parakeet_mel_cache mel_cache;

    std::string path_model;
};

struct parakeet_global {
    // We save the log callback globally
    ggml_log_callback log_callback = parakeet_log_callback_default;
    void * log_callback_user_data = nullptr;
};

static parakeet_global g_state;

static const std::string PARAKEET_SPM_SPACE = "\xE2\x96\x81";

static inline int utf8_codepoint_len(unsigned char c) {
    if ((c & 0x80) == 0x00) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static bool is_sentencepiece_control(const std::string & piece) {
    return piece == "<unk>" || piece == "<s>" || piece == "</s>" || piece == "[BLANK]";
}

static std::string sentencepiece_normalize(const std::string & text) {
    std::string normalized;
    normalized.reserve(text.size() + PARAKEET_SPM_SPACE.size());
    normalized += PARAKEET_SPM_SPACE; // SentencePiece dummy prefix

    for (unsigned char c : text) {
        if (std::isspace(c)) {
            normalized += PARAKEET_SPM_SPACE;
        } else {
            normalized += static_cast<char>(c);
        }
    }

    return normalized;
}

static std::string sentencepiece_piece_to_text(const std::string & piece, bool is_first_piece) {
    if (is_sentencepiece_control(piece)) {
        return "";
    }

    std::string text;
    text.reserve(piece.size());

    size_t pos = 0;
    while (pos < piece.size()) {
        if (piece.compare(pos, PARAKEET_SPM_SPACE.size(), PARAKEET_SPM_SPACE) == 0) {
            if (!is_first_piece || !text.empty()) {
                text += ' ';
            }
            pos += PARAKEET_SPM_SPACE.size();
            continue;
        }

        text += piece[pos];
        ++pos;
    }

    return text;
}


static struct parakeet_batch parakeet_batch_init(int32_t n_tokens) {
    parakeet_batch batch = { 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, };

    batch.token    = (parakeet_token *  ) malloc(sizeof(parakeet_token)    * (n_tokens));
    batch.i_time   = (int32_t *)          malloc(sizeof(int32_t)           * (n_tokens));
    batch.pos      = (parakeet_pos *)     malloc(sizeof(parakeet_pos)      * (n_tokens));
    batch.n_seq_id = (int32_t *)          malloc(sizeof(int32_t)           * (n_tokens));
    batch.seq_id   = (parakeet_seq_id **) malloc(sizeof(parakeet_seq_id *) * (n_tokens + 1));
    for (int i = 0; i < n_tokens; ++i) {
        batch.seq_id[i] = (parakeet_seq_id *) malloc(sizeof(parakeet_seq_id));
    }
    batch.seq_id[n_tokens] = nullptr;
    batch.logits   = (int8_t *)          malloc(sizeof(int8_t)           * n_tokens);

    return batch;
}

static void parakeet_batch_free(struct parakeet_batch batch) {
    if (batch.token)    free(batch.token);
    if (batch.pos)      free(batch.pos);
    if (batch.n_seq_id) free(batch.n_seq_id);
    if (batch.seq_id) {
        for (int i = 0; batch.seq_id[i]; ++i) {
            free(batch.seq_id[i]);
        }
        free(batch.seq_id);
    }
    if (batch.logits)   free(batch.logits);
}

static void parakeet_batch_prep_legacy(parakeet_batch & batch, const parakeet_token * tokens, int n_tokens, int n_past, int seq_id) {
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        if (tokens) {
            batch.token[i] = tokens[i];
        }
        batch.pos     [i]    = n_past + i;
        batch.n_seq_id[i]    = 1;
        batch.seq_id  [i][0] = seq_id;
        batch.logits  [i]    = 0;
    }
    batch.logits[n_tokens - 1] = 1;
}


static size_t parakeet_sched_size(struct parakeet_sched & allocr) {
    size_t size = allocr.meta.size();
    for (int i = 0; i < ggml_backend_sched_get_n_backends(allocr.sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(allocr.sched, i);
        size += ggml_backend_sched_get_buffer_size(allocr.sched, backend);
    }
    return size;
}

// measure the memory usage of a graph and prepare the allocr's internal data buffer
static bool parakeet_sched_graph_init(struct parakeet_sched & allocr, std::vector<ggml_backend_t> backends, std::function<struct ggml_cgraph *()> && get_graph) {
    auto & sched = allocr.sched;
    auto & meta  = allocr.meta;

    sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), PARAKEET_MAX_NODES, false, true);

    if (!sched) {
        PARAKEET_LOG_ERROR("%s: failed to create scheduler\n", __func__);
        return false;
    }

    meta.resize(ggml_tensor_overhead()*PARAKEET_MAX_NODES + ggml_graph_overhead());

    // since there are dependencies between the different graphs,
    // we need to allocate them instead of only reserving to get the correct compute buffer size
    if (!ggml_backend_sched_alloc_graph(sched, get_graph())) {
        // failed to allocate the compute buffer
        PARAKEET_LOG_ERROR("%s: failed to allocate the compute buffer\n", __func__);
        ggml_backend_sched_free(sched);
        sched = nullptr;
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}


template<typename T>
static void read_safe(parakeet_model_loader * loader, T & dest) {
    loader->read(loader->context, &dest, sizeof(T));
    BYTESWAP_VALUE(dest);
}

static bool parakeet_lstm_state_init(
               struct parakeet_state & pstate,
                      ggml_backend_t   backend,
                                 int   n_layer) {
    parakeet_lstm_state & lstm_state = pstate.lstm_state;

    lstm_state.ctx_buf.resize(ggml_tensor_overhead() * n_layer * 2);
    lstm_state.layer.resize(n_layer);

    struct ggml_init_params params = {
        /*.mem_size   =*/ lstm_state.ctx_buf.size(),
        /*.mem_buffer =*/ lstm_state.ctx_buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);

    if (!ctx) {
        PARAKEET_LOG_ERROR("%s: failed to allocate memory for the lstm states context\n", __func__);
        return false;
    }


    for (int il = 0; il < n_layer; ++il) {
        lstm_state.layer[il].h_state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 640);
        lstm_state.layer[il].c_state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 640);
    }

    lstm_state.buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!lstm_state.buffer) {
        PARAKEET_LOG_ERROR("%s: failed to allocate memory for the lstm states\n", __func__);
        return false;
    }

    ggml_backend_buffer_clear(lstm_state.buffer, 0);

    ggml_free(ctx);

    return true;
}

static bool parakeet_pred_state_init(
               struct parakeet_state & pstate,
                      ggml_backend_t   backend,
                                 int   n_pred_dim) {
    pstate.pred_out_buf.resize(ggml_tensor_overhead());

    struct ggml_init_params params = {
        /*.mem_size   =*/ pstate.pred_out_buf.size(),
        /*.mem_buffer =*/ pstate.pred_out_buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        PARAKEET_LOG_ERROR("%s: failed to allocate memory for pred tensor context\n", __func__);
        return false;
    }

    pstate.pred_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_pred_dim);
    pstate.pred_out_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!pstate.pred_out_buffer) {
        PARAKEET_LOG_ERROR("%s: failed to allocate memory for pred tensor\n", __func__);
        ggml_free(ctx);
        return false;
    }

    ggml_free(ctx);

    return true;
}

static ggml_backend_t whisper_backend_init_gpu(const parakeet_context_params & params) {
    ggml_log_set(g_state.log_callback, g_state.log_callback_user_data);

    ggml_backend_dev_t dev = nullptr;

    int cnt = 0;
    if (params.use_gpu) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev_cur = ggml_backend_dev_get(i);
            enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev_cur);
            const char * dev_name = ggml_backend_dev_name(dev_cur);
            PARAKEET_LOG_INFO("%s: device %zu: %s (type: %d)\n", __func__, i, dev_name, dev_type);
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                PARAKEET_LOG_INFO("%s: found GPU device %zu: %s (type: %d, cnt: %d)\n", __func__, i, dev_name, dev_type, cnt);
                if (cnt == params.gpu_device) {
                    dev = dev_cur;
                }

                if (++cnt > params.gpu_device) {
                    break;
                }
            }
        }
    }

    if (dev == nullptr) {
        PARAKEET_LOG_INFO("%s: no GPU found\n", __func__);
        return nullptr;
    }

    PARAKEET_LOG_INFO("%s: using %s backend\n", __func__, ggml_backend_dev_name(dev));
    ggml_backend_t result = ggml_backend_dev_init(dev, nullptr);
    if (!result) {
        PARAKEET_LOG_ERROR("%s: failed to initialize %s backend\n", __func__, ggml_backend_dev_name(dev));
    }

    return result;
}

static std::vector<ggml_backend_t> whisper_backend_init(const parakeet_context_params & params) {
    std::vector<ggml_backend_t> result;

    ggml_backend_t backend_gpu = whisper_backend_init_gpu(params);

    if (backend_gpu) {
        result.push_back(backend_gpu);
    }

    // ACCEL backends
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            PARAKEET_LOG_INFO("%s: using %s backend\n", __func__, ggml_backend_dev_name(dev));
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (!backend) {
                PARAKEET_LOG_ERROR("%s: failed to initialize %s backend\n", __func__, ggml_backend_dev_name(dev));
                continue;
            }
            result.push_back(backend);
        }
    }

    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend_cpu == nullptr) {
        throw std::runtime_error("failed to initialize CPU backend");
    }
    result.push_back(backend_cpu);

    return result;
}

using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

static buft_list_t make_buft_list(parakeet_context_params & params) {
    // Prio order: GPU -> CPU Extra -> CPU
    buft_list_t buft_list;

    // GPU
    if (params.use_gpu) {
        int cnt = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                if (cnt == params.gpu_device) {
                    auto * buft = ggml_backend_dev_buffer_type(dev);
                    if (buft) {
                        buft_list.emplace_back(dev, buft);
                    }
                }

                if (++cnt > params.gpu_device) {
                    break;
                }
            }
        }
    }

    // CPU Extra
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
    if (get_extra_bufts_fn) {
        ggml_backend_buffer_type_t * extra_bufts = get_extra_bufts_fn(cpu_dev);
        while (extra_bufts && *extra_bufts) {
            buft_list.emplace_back(cpu_dev, *extra_bufts);
            ++extra_bufts;
        }
    }

    // CPU
    buft_list.emplace_back(cpu_dev, ggml_backend_cpu_buffer_type());

    return buft_list;
}

static bool weight_buft_supported(const parakeet_hparams & hparams, ggml_tensor * w, ggml_op op, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev) {
    bool op_supported = true;

    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ||
        ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU ||
        (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && buft == ggml_backend_cpu_buffer_type())) {
        // GPU and default CPU backend support all operators
        op_supported = true;
    } else {
        switch (op) {
            // The current extra_buffer_type implementations only support GGML_OP_MUL_MAT and GGML_OP_GET_ROWS
            case GGML_OP_GET_ROWS:
            case GGML_OP_MUL_MAT: {
                ggml_init_params params = {
                    /*.mem_size   =*/ 2 * ggml_tensor_overhead(),
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };

                ggml_context_ptr ctx_ptr { ggml_init(params) };
                if (!ctx_ptr) {
                    throw std::runtime_error("failed to create ggml context");
                }
                ggml_context * ctx = ctx_ptr.get();

                ggml_tensor * op_tensor = nullptr;

                if (op == GGML_OP_MUL_MAT) {
                    int64_t n_ctx = hparams.n_audio_ctx;
                    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], n_ctx, w->ne[2], w->ne[3]);
                    op_tensor = ggml_mul_mat(ctx, w, b);
                } else if (op == GGML_OP_GET_ROWS) {
                    int64_t num_indices = 8;
                    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, num_indices);
                    op_tensor = ggml_get_rows(ctx, w, indices);
                }

                // create a temporary dummy buffer for the weight so that supports_op can check the buffer type
                GGML_ASSERT(w->buffer == nullptr);
                w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
                op_supported = ggml_backend_dev_supports_op(dev, op_tensor);
                ggml_backend_buffer_free(w->buffer);
                w->buffer = nullptr;
                break;
            }
            default: {
                op_supported = false;
                break;
            }
        };
    }

    return op_supported;
}

static ggml_backend_buffer_type_t select_weight_buft(const parakeet_hparams & hparams, ggml_tensor * w, ggml_op op, buft_list_t buft_list) {
    GGML_ASSERT(!buft_list.empty());
    for (const auto & p : buft_list) {
        ggml_backend_dev_t dev = p.first;
        ggml_backend_buffer_type_t buft = p.second;
        if (weight_buft_supported(hparams, w, op, buft, dev)) {
            return buft;
        }
    }

    return nullptr;
}

// load the model from a ggml file
//

// see the convert-parakeet-to-ggml.py script for details
//
static bool parakeet_model_load(struct parakeet_model_loader * loader, parakeet_context & wctx) {
    PARAKEET_LOG_INFO("%s: loading model\n", __func__);

    const int64_t t_start_us = ggml_time_us();

    wctx.t_start_us = t_start_us;

    auto & model = wctx.model;
    auto & vocab = wctx.vocab;

    // verify magic
    {
        uint32_t magic;
        read_safe(loader, magic);
        if (magic != GGML_FILE_MAGIC) {
            PARAKEET_LOG_ERROR("%s: invalid model data (bad magic)\n", __func__);
            return false;
        }
    }

    //load hparams
    parakeet_hparams hparams;
    {
        read_safe(loader, hparams.n_vocab);
        read_safe(loader, hparams.n_audio_ctx);
        read_safe(loader, hparams.n_audio_state);
        read_safe(loader, hparams.n_audio_head);
        read_safe(loader, hparams.n_audio_layer);
        read_safe(loader, hparams.n_mels);
        read_safe(loader, hparams.ftype);
        read_safe(loader, hparams.n_fft);
        read_safe(loader, hparams.subsampling_factor);
        read_safe(loader, hparams.n_subsampling_channels);
        read_safe(loader, hparams.n_pos_max_len);
        read_safe(loader, hparams.n_pred_dim);
        read_safe(loader, hparams.n_pred_layers);
        read_safe(loader, hparams.n_tdt_durations);
        read_safe(loader, hparams.n_max_tokens);

        hparams.arch = PARAKEET_ARCH_TDT;
        wctx.model.hparams = hparams;

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;

        // for the big tensors, we have the option to store the data in 16-bit floats or quantized
        // in order to save memory and also to speed up the computation
        wctx.wtype = ggml_ftype_to_ggml_type((ggml_ftype) hparams.ftype);
        if (wctx.wtype == GGML_TYPE_COUNT) {
            PARAKEET_LOG_ERROR("%s: invalid model (bad ftype value %d)\n", __func__, hparams.ftype);
            return false;
        }

        const char* arch_name = hparams.arch == PARAKEET_ARCH_TDT ? "Parakeet TDT" : "unknown";
        PARAKEET_LOG_INFO("%s: arch          = %s\n", __func__, arch_name);
        PARAKEET_LOG_INFO("%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
        PARAKEET_LOG_INFO("%s: n_audio_ctx   = %d\n", __func__, hparams.n_audio_ctx);
        PARAKEET_LOG_INFO("%s: n_audio_state = %d\n", __func__, hparams.n_audio_state);
        PARAKEET_LOG_INFO("%s: n_audio_head  = %d\n", __func__, hparams.n_audio_head);
        PARAKEET_LOG_INFO("%s: n_audio_layer = %d\n", __func__, hparams.n_audio_layer);
        PARAKEET_LOG_INFO("%s: n_mels        = %d\n", __func__, hparams.n_mels);
        PARAKEET_LOG_INFO("%s: n_fft         = %d\n", __func__, hparams.n_fft);
        PARAKEET_LOG_INFO("%s: ftype         = %d\n", __func__, hparams.ftype);
        PARAKEET_LOG_INFO("%s: qntvr         = %d\n", __func__, qntvr);
        PARAKEET_LOG_INFO("%s: subsampling_factor = %d\n", __func__, hparams.subsampling_factor);
        PARAKEET_LOG_INFO("%s: n_subsampling_channels = %d\n", __func__, hparams.n_subsampling_channels);
        PARAKEET_LOG_INFO("%s: n_pos_max_len = %d\n", __func__, hparams.n_pos_max_len);
        PARAKEET_LOG_INFO("%s: n_pred_dim = %d\n", __func__, hparams.n_pred_dim);
        PARAKEET_LOG_INFO("%s: n_pred_layers = %d\n", __func__, hparams.n_pred_layers);
        PARAKEET_LOG_INFO("%s: n_tdt_durations = %d\n", __func__, hparams.n_tdt_durations);
        PARAKEET_LOG_INFO("%s: n_max_tokens = %d\n", __func__, hparams.n_max_tokens);
    }

    // load mel filters
    {
        auto & filters = wctx.model.filters;

        read_safe(loader, filters.n_mel);
        read_safe(loader, filters.n_fb);

        filters.data.resize(filters.n_mel * filters.n_fb);
        loader->read(loader->context, filters.data.data(), filters.data.size() * sizeof(float));
        BYTESWAP_FILTERS(filters);
    }

    // load window function
    {
        int32_t n_window = 0;
        read_safe(loader, n_window);

        wctx.mel_cache.window.resize(n_window);
        loader->read(loader->context, wctx.mel_cache.window.data(), n_window * sizeof(float));

#ifdef GGML_BIG_ENDIAN
        for (auto & datum : wctx.mel_cache.window) {
            datum = byteswap(datum);
        }
#endif

        PARAKEET_LOG_INFO("%s: loaded window function with %d samples\n", __func__, n_window);
    }

    // load tdt durations values
    {
        auto & tdt_durations = wctx.model.tdt_durations;
        tdt_durations.resize(hparams.n_tdt_durations);
        loader->read(loader->context, tdt_durations.data(), hparams.n_tdt_durations * sizeof(uint32_t));

        PARAKEET_LOG_INFO("%s: loaded tdt_durations: [", __func__);
        for (const auto value : tdt_durations) {
            PARAKEET_LOG_INFO("%u ", value);
        }
        PARAKEET_LOG_INFO("]\n");
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        read_safe(loader, n_vocab);

        std::string word;
        std::vector<char> tmp;

        tmp.reserve(128);

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            read_safe(loader, len);

            if (len > 0) {
                tmp.resize(len);
                loader->read(loader->context, &tmp[0], tmp.size()); // read to buffer
                word.assign(&tmp[0], tmp.size());
            } else {
                PARAKEET_LOG_WARN("%s: warning: empty-string token in vocab, i = %d\n", __func__, i);
                word = "";
            }

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
            vocab.max_token_length = std::max(vocab.max_token_length, word.size());
        }
        // Blank token for transducer is at index n_vocab (8192), outside the vocabulary
        int blank_id = n_vocab;
        vocab.token_blank = blank_id;
        vocab.id_to_token[blank_id] = "[BLANK]";
        vocab.token_to_id["[BLANK]"] = blank_id;

        // Set special token IDs by looking them up in the loaded vocabulary
        // These are from the SentencePiece vocab file loaded above
        if (vocab.token_to_id.find("<unk>") != vocab.token_to_id.end()) {
            vocab.token_unk = vocab.token_to_id.at("<unk>");
        } else {
            vocab.token_unk = 0;  // Fallback
        }

        if (vocab.token_to_id.find("<s>") != vocab.token_to_id.end()) {
            vocab.token_bos = vocab.token_to_id.at("<s>");
        } else if (vocab.token_to_id.find("<|startoftranscript|>") != vocab.token_to_id.end()) {
            vocab.token_bos = vocab.token_to_id.at("<|startoftranscript|>");
        } else {
            vocab.token_bos = 0;  // Fallback
        }

        if (vocab.token_to_id.find("</s>") != vocab.token_to_id.end()) {
            vocab.token_eos = vocab.token_to_id.at("</s>");
        } else if (vocab.token_to_id.find("<|endoftext|>") != vocab.token_to_id.end()) {
            vocab.token_eos = vocab.token_to_id.at("<|endoftext|>");
        } else {
            vocab.token_eos = 0;  // Fallback
        }

        vocab.n_vocab = model.hparams.n_vocab;

        PARAKEET_LOG_INFO("%s: loaded vocab with %d tokens (blank_id=%d, unk=%d, bos=%d, eos=%d)\n",
            __func__, n_vocab, blank_id, vocab.token_unk, vocab.token_bos, vocab.token_eos);
    }

    const ggml_type wtype = wctx.wtype;
    const ggml_type vtype = wctx.wtype == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16; // conv type

    const int n_audio_layer = hparams.n_audio_layer;

    // Calculate tensor count based on architecture
    size_t n_tensors = 2 + 14 + 1 + 29*n_audio_layer + 9 + 6;

    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto get_ctx = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to create ggml context");
            }

            ctx_map[buft] = ctx;
            wctx.model.ctxs.emplace_back(ctx);

            return ctx;
        }

        return it->second;
    };

    // Create a list of available bufts, in priority order
    buft_list_t buft_list = make_buft_list(wctx.params);

    auto create_tensor = [&](parakeet_tensor type, ggml_tensor * meta, int layer = -1) -> ggml_tensor * {
        ggml_op op = PARAKEET_TENSOR_INFO.at(type);
        ggml_backend_buffer_type_t buft = select_weight_buft(hparams, meta, op, buft_list);
        if (!buft) {
            throw std::runtime_error(format("failed to find a compatible buffer type for parakeet tensor %s", PARAKEET_TENSOR_NAMES.at(type)));
        }

        ggml_context * ctx = get_ctx(buft);
        ggml_tensor * tensor = ggml_dup_tensor(ctx, meta);

        std::string tensor_name;
        if (layer >= 0) {
            tensor_name = format(PARAKEET_TENSOR_NAMES.at(type), layer);
        } else {
            tensor_name = PARAKEET_TENSOR_NAMES.at(type);
        }

        wctx.model.tensors[tensor_name] = tensor;

        return tensor;
    };

    // prepare tensors for the weights

    // Count: preprocessor (2) + pre_encode (14) + positional encoding (1) + encoder layers (29 per layer) + prediction (9) + joint (6)
    const int n_parakeet_tensors = 2 + 14 + 1 + (29 * n_audio_layer) + 9 + 6;

    ggml_init_params params = {
        /*.mem_size   =*/ n_parakeet_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(params);

    const int n_audio_state = hparams.n_audio_state;

    model.layers.resize(n_audio_layer);

    // Encoder pre_encode
    const int n_subsampling_channels = hparams.n_subsampling_channels;
    model.enc_pre_out_w = create_tensor(PARAKEET_TENSOR_ENC_PRE_OUT_WEIGHT, ggml_new_tensor_2d(ctx, wtype, 4096, n_audio_state));
    ggml_set_name(model.enc_pre_out_w, "enc_pre_out_w");
    model.enc_pre_out_b = create_tensor(PARAKEET_TENSOR_ENC_PRE_OUT_BIAS, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state));
    ggml_set_name(model.enc_pre_out_b, "enc_pre_out_b");

    model.enc_pre_conv_0_w = create_tensor(PARAKEET_TENSOR_ENC_PRE_CONV_0_WEIGHT, ggml_new_tensor_4d(ctx, vtype, 3, 3, 1, n_subsampling_channels));
    ggml_set_name(model.enc_pre_conv_0_w, "enc_pre_conv_0_w");
    model.enc_pre_conv_0_b = create_tensor(PARAKEET_TENSOR_ENC_PRE_CONV_0_BIAS, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, n_subsampling_channels, 1));
    ggml_set_name(model.enc_pre_conv_0_b, "enc_pre_conv_0_b");

    model.enc_pre_conv_2_w = create_tensor(PARAKEET_TENSOR_ENC_PRE_CONV_2_WEIGHT, ggml_new_tensor_4d(ctx, vtype, 3, 3, 1, n_subsampling_channels));
    ggml_set_name(model.enc_pre_conv_2_w, "enc_pre_conv_2_w");
    model.enc_pre_conv_2_b = create_tensor(PARAKEET_TENSOR_ENC_PRE_CONV_2_BIAS, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, n_subsampling_channels, 1));
    ggml_set_name(model.enc_pre_conv_2_b, "enc_pre_conv_2_b");

    model.enc_pre_conv_3_w = create_tensor(PARAKEET_TENSOR_ENC_PRE_CONV_3_WEIGHT, ggml_new_tensor_4d(ctx, wtype, 1, 1, n_subsampling_channels, n_subsampling_channels));
    ggml_set_name(model.enc_pre_conv_3_w, "enc_pre_conv_3_w");
    model.enc_pre_conv_3_b = create_tensor(PARAKEET_TENSOR_ENC_PRE_CONV_3_BIAS, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, n_subsampling_channels, 1));
    ggml_set_name(model.enc_pre_conv_3_b, "enc_pre_conv_3_b");

    model.enc_pre_conv_5_w = create_tensor(PARAKEET_TENSOR_ENC_PRE_CONV_5_WEIGHT, ggml_new_tensor_4d(ctx, vtype, 3, 3, 1, n_subsampling_channels));
    ggml_set_name(model.enc_pre_conv_5_w, "enc_pre_conv_5_w");
    model.enc_pre_conv_5_b = create_tensor(PARAKEET_TENSOR_ENC_PRE_CONV_5_BIAS, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, n_subsampling_channels, 1));
    ggml_set_name(model.enc_pre_conv_5_b, "enc_pre_conv_5_b");

    model.enc_pre_conv_6_w = create_tensor(PARAKEET_TENSOR_ENC_PRE_CONV_6_WEIGHT, ggml_new_tensor_4d(ctx, wtype, 1, 1, n_subsampling_channels, n_subsampling_channels));
    ggml_set_name(model.enc_pre_conv_6_w, "enc_pre_conv_6_w");
    model.enc_pre_conv_6_b = create_tensor(PARAKEET_TENSOR_ENC_PRE_CONV_6_BIAS, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, n_subsampling_channels, 1));
    ggml_set_name(model.enc_pre_conv_6_b, "enc_pre_conv_6_b");

    // Encoder layers
    for (int i = 0; i < n_audio_layer; ++i) {
        auto & layer = model.layers[i];

        // Feed forward 1
        layer.norm_ff1_w    = create_tensor(PARAKEET_TENSOR_ENC_NORM_FF1_WEIGHT, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);
        layer.norm_ff1_b    = create_tensor(PARAKEET_TENSOR_ENC_NORM_FF1_BIAS, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);
        layer.ff1_linear1_w = create_tensor(PARAKEET_TENSOR_ENC_FF1_LINEAR1_WEIGHT, ggml_new_tensor_2d(ctx, wtype, n_audio_state, 4*n_audio_state), i);
        ggml_format_name(layer.ff1_linear1_w, "enc_%d_ff1_linear1_w", i);
        layer.ff1_linear2_w = create_tensor(PARAKEET_TENSOR_ENC_FF1_LINEAR2_WEIGHT, ggml_new_tensor_2d(ctx, wtype, 4*n_audio_state, n_audio_state), i);
        ggml_format_name(layer.ff1_linear2_w, "enc_%d_ff1_linear2_w", i);

        // Convolution module
        layer.norm_conv_w         = create_tensor(PARAKEET_TENSOR_ENC_NORM_CONV_WEIGHT, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);
        ggml_format_name(layer.norm_conv_w, "enc_%d_norm_conv_w", i);
        layer.norm_conv_b         = create_tensor(PARAKEET_TENSOR_ENC_NORM_CONV_BIAS, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);
        ggml_format_name(layer.norm_conv_b, "enc_%d_norm_conv_b", i);
        layer.conv_pw1_w          = create_tensor(PARAKEET_TENSOR_ENC_CONV_PW1_WEIGHT, ggml_new_tensor_2d(ctx, wtype, n_audio_state, 2*n_audio_state), i);
        ggml_format_name(layer.conv_pw1_w, "enc_%d_conv_pw1_w", i);
        layer.conv_dw_w           = create_tensor(PARAKEET_TENSOR_ENC_CONV_DW_WEIGHT, ggml_new_tensor_2d(ctx, wtype, 9, n_audio_state), i);
        ggml_format_name(layer.conv_dw_w, "enc_%d_conv_dw_w", i);
        layer.conv_bn_w           = create_tensor(PARAKEET_TENSOR_ENC_CONV_BN_WEIGHT, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);
        ggml_format_name(layer.conv_bn_w, "enc_%d_conv_bn_w", i);
        layer.conv_bn_b           = create_tensor(PARAKEET_TENSOR_ENC_CONV_BN_BIAS, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);
        ggml_format_name(layer.conv_bn_b, "enc_%d_conv_bn_b", i);
        layer.conv_bn_mean        = create_tensor(PARAKEET_TENSOR_ENC_CONV_BN_MEAN, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);
        layer.conv_bn_var         = create_tensor(PARAKEET_TENSOR_ENC_CONV_BN_VAR, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);
        ggml_format_name(layer.conv_bn_var, "enc_%d_conv_bn_var", i);
        layer.conv_bn_num_batches = create_tensor(PARAKEET_TENSOR_ENC_CONV_BN_NUM_BATCHES, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1), i);
        layer.conv_pw2_w          = create_tensor(PARAKEET_TENSOR_ENC_CONV_PW2_WEIGHT, ggml_new_tensor_2d(ctx, wtype, n_audio_state, n_audio_state), i);
        ggml_format_name(layer.conv_pw2_w, "enc_%d_conv_pw2_w", i);

        // Self attention
        layer.norm_attn_w      = create_tensor(PARAKEET_TENSOR_ENC_NORM_ATTN_WEIGHT, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);
        layer.norm_attn_b      = create_tensor(PARAKEET_TENSOR_ENC_NORM_ATTN_BIAS, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);
        layer.attn_pos_bias_u  = create_tensor(PARAKEET_TENSOR_ENC_ATTN_POS_BIAS_U, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 8), i);
        layer.attn_pos_bias_v  = create_tensor(PARAKEET_TENSOR_ENC_ATTN_POS_BIAS_V, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 8), i);
        layer.attn_q_w         = create_tensor(PARAKEET_TENSOR_ENC_ATTN_Q_WEIGHT, ggml_new_tensor_2d(ctx, wtype, n_audio_state, n_audio_state), i);
        layer.attn_k_w         = create_tensor(PARAKEET_TENSOR_ENC_ATTN_K_WEIGHT, ggml_new_tensor_2d(ctx, wtype, n_audio_state, n_audio_state), i);
        layer.attn_v_w         = create_tensor(PARAKEET_TENSOR_ENC_ATTN_V_WEIGHT, ggml_new_tensor_2d(ctx, wtype, n_audio_state, n_audio_state), i);
        layer.attn_out_w       = create_tensor(PARAKEET_TENSOR_ENC_ATTN_OUT_WEIGHT, ggml_new_tensor_2d(ctx, wtype, n_audio_state, n_audio_state), i);
        layer.attn_pos_w       = create_tensor(PARAKEET_TENSOR_ENC_ATTN_POS_WEIGHT, ggml_new_tensor_2d(ctx, wtype, n_audio_state, n_audio_state), i);
        ggml_format_name(layer.attn_pos_w, "enc_%d_attn_pos_w", i);

        // Feed forward 2
        layer.norm_ff2_w    = create_tensor(PARAKEET_TENSOR_ENC_NORM_FF2_WEIGHT, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);
        layer.norm_ff2_b    = create_tensor(PARAKEET_TENSOR_ENC_NORM_FF2_BIAS, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);
        layer.ff2_linear1_w = create_tensor(PARAKEET_TENSOR_ENC_FF2_LINEAR1_WEIGHT, ggml_new_tensor_2d(ctx, wtype, n_audio_state, 4*n_audio_state), i);
        layer.ff2_linear2_w = create_tensor(PARAKEET_TENSOR_ENC_FF2_LINEAR2_WEIGHT, ggml_new_tensor_2d(ctx, wtype, 4*n_audio_state, n_audio_state), i);

        // Output norm
        layer.norm_out_w = create_tensor(PARAKEET_TENSOR_ENC_NORM_OUT_WEIGHT, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);
        layer.norm_out_b = create_tensor(PARAKEET_TENSOR_ENC_NORM_OUT_BIAS, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state), i);
    }

    // Prediction network (decoder)
    const int dec_hidden = hparams.n_pred_dim;
    model.prediction.embed_w = create_tensor(PARAKEET_TENSOR_PRED_EMBED_WEIGHT, ggml_new_tensor_2d(ctx, wtype, dec_hidden, 8193));
    model.prediction.lstm_layer.resize(hparams.n_pred_layers);
    for (int i = 0; i < hparams.n_pred_layers; ++i) {
        auto & layer = model.prediction.lstm_layer[i];
        layer.ih_w = create_tensor(PARAKEET_TENSOR_PRED_LSTM_WEIGHT_IH, ggml_new_tensor_2d(ctx, wtype, dec_hidden, 2560), i);
        ggml_format_name(layer.ih_w, "pred_%d_ih_w", i);

        layer.ih_b = create_tensor(PARAKEET_TENSOR_PRED_LSTM_BIAS_IH, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2560), i);
        ggml_format_name(layer.ih_b, "pred_%d_ih_b", i);

        layer.hh_b = create_tensor(PARAKEET_TENSOR_PRED_LSTM_BIAS_HH, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2560), i);
        ggml_format_name(layer.hh_b, "pred_%d_hh_b", i);

        layer.hh_w = create_tensor(PARAKEET_TENSOR_PRED_LSTM_WEIGHT_HH, ggml_new_tensor_2d(ctx, wtype, dec_hidden, 2560), i);
        ggml_format_name(layer.hh_w, "pred_%d_hh_w", i);
    }

    // Joint network
    model.joint.pred_w = create_tensor(PARAKEET_TENSOR_JOINT_PRED_WEIGHT, ggml_new_tensor_2d(ctx, wtype, dec_hidden, dec_hidden));
    ggml_set_name(model.joint.pred_w, "pred_w");
    model.joint.pred_b = create_tensor(PARAKEET_TENSOR_JOINT_PRED_BIAS, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dec_hidden));
    ggml_set_name(model.joint.pred_b, "pred_b");
    model.joint.enc_w  = create_tensor(PARAKEET_TENSOR_JOINT_ENC_WEIGHT, ggml_new_tensor_2d(ctx, wtype, n_audio_state, dec_hidden));
    ggml_set_name(model.joint.enc_w, "enc_w");
    model.joint.enc_b  = create_tensor(PARAKEET_TENSOR_JOINT_ENC_BIAS, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dec_hidden));
    ggml_set_name(model.joint.enc_b, "enc_b");
    model.joint.net_w  = create_tensor(PARAKEET_TENSOR_JOINT_NET_WEIGHT, ggml_new_tensor_2d(ctx, wtype, dec_hidden, 8198));
    ggml_set_name(model.joint.net_w, "net_w");
    model.joint.net_b  = create_tensor(PARAKEET_TENSOR_JOINT_NET_BIAS, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8198));
    ggml_set_name(model.joint.net_b, "net_b");

    // Relative positional encoding
    {
        model.pe = create_tensor(PARAKEET_TENSOR_ENC_PE, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_state, hparams.n_pos_max_len * 2 - 1));
        ggml_set_name(model.pe, "pe");
    }

    ggml_free(ctx);

    // allocate tensors in the backend buffers
    for (auto & p : ctx_map) {
        ggml_backend_buffer_type_t buft = p.first;
        ggml_context * ctx = p.second;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (buf) {
            wctx.model.buffers.emplace_back(buf);

            size_t size_main = ggml_backend_buffer_get_size(buf);
            PARAKEET_LOG_INFO("%s: %12s total size = %8.2f MB\n", __func__, ggml_backend_buffer_name(buf), size_main / 1e6);
        }
    }

    // load weights
    {
        size_t total_size = 0;

        auto & tensors_map = wctx.model.tensors;
        int & n_loaded = wctx.model.n_loaded;

        n_loaded = 0;

        std::vector<char> read_buf;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            read_safe(loader, n_dims);
            read_safe(loader, length);
            read_safe(loader, ttype);

            if (loader->eof(loader->context)) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[4] = { 1, 1, 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                read_safe(loader, ne[i]);
                nelements *= ne[i];
            }

            std::string name;
            std::vector<char> tmp(length); // create a buffer
            loader->read(loader->context, &tmp[0], tmp.size()); // read to buffer
            name.assign(&tmp[0], tmp.size());

            if (tensors_map.find(name) == tensors_map.end()) {
                PARAKEET_LOG_ERROR("%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = tensors_map[name.data()];

            if (ggml_nelements(tensor) != nelements) {
                PARAKEET_LOG_ERROR("%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                PARAKEET_LOG_ERROR("%s: shape: [%d, %d, %d], expected: [%d, %d, %d]\n",
                        __func__, ne[0], ne[1], ne[2], (int) tensor->ne[0], (int) tensor->ne[1], (int) tensor->ne[2]);
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2] || tensor->ne[3] != ne[3]) {
                PARAKEET_LOG_ERROR("%s: tensor '%s' has wrong shape in model file: got [%d, %d, %d, %d], expected [%d, %d, %d, %d]\n",
                        __func__, name.data(), (int) tensor->ne[0], (int) tensor->ne[1], (int) tensor->ne[2], (int) tensor->ne[3], ne[0], ne[1], ne[2], ne[3]);
                return false;
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                PARAKEET_LOG_ERROR("%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            if (ggml_backend_buffer_is_host(tensor->buffer)) {
                // for the CPU and Metal backend, we can read directly into the tensor
                loader->read(loader->context, tensor->data, ggml_nbytes(tensor));
                BYTESWAP_TENSOR(tensor);
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(ggml_nbytes(tensor));

                loader->read(loader->context, read_buf.data(), read_buf.size());

                ggml_backend_tensor_set(tensor, read_buf.data(), 0, ggml_nbytes(tensor));
            }

            total_size += ggml_nbytes(tensor);
            n_loaded++;
        }

        PARAKEET_LOG_INFO("%s: model size    = %7.2f MB\n", __func__, total_size/1e6);

        if (n_loaded == 0) {
            PARAKEET_LOG_WARN("%s: WARN no tensors loaded from model file - assuming empty model for testing\n", __func__);
        } else if (n_loaded != (int) tensors_map.size()) {
            PARAKEET_LOG_ERROR("%s: ERROR not all tensors loaded from model file - expected %zu, got %d\n", __func__, tensors_map.size(), n_loaded);
            return false;
        }
    }

    auto & buffers = wctx.model.buffers;
    for (auto & buf : buffers) {
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    wctx.t_load_us = ggml_time_us() - t_start_us;

    return true;
}

//
// This function build the computation graph for the pre-encoder stage.
//
// It takes the generated mel spectrogram as an input tensor, which will be set
// during processing, and performs a number of convolutions to detect/filter
// features, reduces the time dimension by a factor of 8, and finally projects
// this information into the models abstract feature space.
//
static struct ggml_cgraph * parakeet_build_graph_conv(parakeet_context & pctx, parakeet_state & pstate) {
    const auto & model   = pctx.model;
    const auto & hparams = model.hparams;
    const int n_time     = pstate.n_audio_ctx > 0 ? pstate.n_audio_ctx : hparams.n_audio_ctx;
    const int n_mels     = hparams.n_mels;

    struct ggml_init_params params = {
        /*.mem_size   =*/ pstate.sched_conv.meta.size(),
        /*.mem_buffer =*/ pstate.sched_conv.meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    ggml_cgraph * gf = ggml_new_graph(ctx0);

    // [freq, time]
    struct ggml_tensor * mel = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_mels, n_time, 1, 1);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // [freq, time, channels, batch]
    struct ggml_tensor * cur = ggml_conv_2d(ctx0, model.enc_pre_conv_0_w, mel, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, model.enc_pre_conv_0_b);
    ggml_set_name(cur, "pre_conv_0");

    cur = ggml_relu(ctx0, cur);
    ggml_set_name(cur, "pre_conv_0_relu");

    // enc_pre_conv_2_w: {3, 3, 1, 256} (depthwise)
    // [freq, time, channels, batch]
    cur = ggml_conv_2d_dw_direct(ctx0, model.enc_pre_conv_2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, model.enc_pre_conv_2_b);
    ggml_set_name(cur, "pre_conv_2");

    // enc_pre_conv_3_w: {1, 1, 256, 256} (pointwise)
    // [freq, time, channels, batch]
    cur = ggml_conv_2d(ctx0, model.enc_pre_conv_3_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, model.enc_pre_conv_3_b);
    ggml_set_name(cur, "pre_conv_3");

    cur = ggml_relu(ctx0, cur);
    ggml_set_name(cur, "pre_conv_3_relu");

    // enc_pre_conv_5_w: {3, 3, 1, 256} (depthwise)
    // [freq, time, channels, batch]
    cur = ggml_conv_2d_dw_direct(ctx0, model.enc_pre_conv_5_w, cur, 2, 2, 1, 1, 1, 1);
    ggml_set_name(cur, "pre_conv_5_direct");
    cur = ggml_add(ctx0, cur, model.enc_pre_conv_5_b);
    ggml_set_name(cur, "pre_conv_5");

    // enc_pre_conv_6_w: {1, 1, 256, 256} (pointwise)
    // [freq, time, channels, batch]
    cur = ggml_conv_2d(ctx0, model.enc_pre_conv_6_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, model.enc_pre_conv_6_b);
    ggml_set_name(cur, "pre_conv_6");

    cur = ggml_relu(ctx0, cur);
    ggml_set_name(cur, "pre_conv_6_relu");

    // [freq, time, chan]
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
    // [freq, chan, time]
    cur = ggml_cont(ctx0, cur);

    const int n_freq   = cur->ne[0]; // 16
    const int n_chan   = cur->ne[1]; // 256
    const int n_frames = cur->ne[2]; // time

    // [freq, time, chan, batch] -> [(freq * chan), time]
    cur = ggml_reshape_2d(ctx0, cur, n_freq * n_chan, n_frames);

    cur = ggml_mul_mat(ctx0, model.enc_pre_out_w, cur);
    cur = ggml_add(ctx0, cur, model.enc_pre_out_b);

    ggml_set_name(cur, "pre_enc_out");
    ggml_set_output(cur);
    pstate.pre_enc_out = cur;

    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);

    return gf;
}

// This function builds the graph for the encoder part of the model.
//
// It takes the output of the pre-encoder convolution above which will have the
// shape of [hidden_dim, time_frames]
static struct ggml_cgraph * parakeet_build_graph_encoder(parakeet_context & pctx, parakeet_state & pstate) {
    const auto & model    = pctx.model;
    const auto & hparams  = model.hparams;
    const int n_layer     = hparams.n_audio_layer;
    const int n_state     = hparams.n_audio_state;
    const float fc_factor = 0.5f;

    struct ggml_init_params params = {
        /*.mem_size   =*/ pstate.sched_encode.meta.size(),
        /*.mem_buffer =*/ pstate.sched_encode.meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, PARAKEET_MAX_NODES, false);

    // Create a view of the output produced by parakeet_build_graph_conv
    // [feat, time_frames, 1, 1]
    struct ggml_tensor * cur = ggml_view_tensor(ctx0, pstate.pre_enc_out);
    ggml_set_name(cur, "encoder_inp");

    // [time_frames, time_frames, 1, 1]]
    struct ggml_tensor * attn_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cur->ne[1], cur->ne[1]);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    for (int il = 0; il < n_layer; ++il) {
        // FFN1
        {
            struct ggml_tensor * residual = cur;
            ggml_format_name(cur, "enc_%d_res", il);

            // norm
            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].norm_ff1_w), model.layers[il].norm_ff1_b);
            ggml_format_name(cur, "enc_%d_ffn_norm_1", il);

            // ffn_1
            cur = ggml_mul_mat(ctx0, model.layers[il].ff1_linear1_w, cur);
            cur = ggml_silu(ctx0, cur);
            ggml_format_name(cur, "enc_%d_silu", il);

            cur = ggml_mul_mat(ctx0, model.layers[il].ff1_linear2_w, cur);
            ggml_format_name(cur, "enc_%d_ffn_1", il);

            cur = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, fc_factor));
            ggml_format_name(cur, "enc_%d_res_ffn", il);
        }

        // self attention block using relative positional encoding from the model.pe tensor.
        {
            // [feat, time_frames, 1, 1]
            struct ggml_tensor * residual = cur;

            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].norm_attn_w), model.layers[il].norm_attn_b);
            ggml_format_name(cur, "enc_%d_attn_norm", il);

            const int n_head = hparams.n_audio_head;
            const int d_head = n_state / n_head;
            const int n_time = cur->ne[1];

            // [feat, time_frames, 1, 1]
            struct ggml_tensor * Q_cur = ggml_mul_mat(ctx0, model.layers[il].attn_q_w, cur);
            struct ggml_tensor * K_cur = ggml_mul_mat(ctx0, model.layers[il].attn_k_w, cur);
            struct ggml_tensor * V_cur = ggml_mul_mat(ctx0, model.layers[il].attn_v_w, cur);

            // [d_head, n_heads, time_frames, 1]
            Q_cur = ggml_reshape_3d(ctx0, Q_cur, d_head, n_head, n_time);
            K_cur = ggml_reshape_3d(ctx0, K_cur, d_head, n_head, n_time);
            V_cur = ggml_reshape_3d(ctx0, V_cur, d_head, n_head, n_time);

            const int input_len = cur->ne[1];
            const int center_pos = model.pe->ne[1] / 2 + 1;
            const int start_pos = center_pos - input_len;
            const int window_size = 2 * input_len - 1;
            const size_t offset = start_pos * model.pe->nb[1];

            // [feat, window_size]
            struct ggml_tensor * pos_emb = ggml_view_2d(ctx0, model.pe,
                                            n_state, window_size,
                                            model.pe->nb[1], offset);
            ggml_format_name(pos_emb, "enc_%d_attn_pos_emb", il);

            struct ggml_tensor * pos = ggml_mul_mat(ctx0, model.layers[il].attn_pos_w, pos_emb);
            ggml_format_name(pos, "enc_%d_attn_pos", il);

            // Add the content bias to Q. Like a baseline query for content. Like
            // regardless of my specific position all ways look for these base features.
            // So we shift the Q by the content bias.
            // [feat, head, time_frames, batch]
            struct ggml_tensor * Q_u = ggml_add(ctx0, Q_cur, model.layers[il].attn_pos_bias_u);
            ggml_format_name(Q_u, "enc_%d_attn_q_u", il);

            // [feat, time_frames, head, 1]
            struct ggml_tensor * K_prep = ggml_permute(ctx0, K_cur, 0, 2, 1, 3);
            // [feat, time_frames, head, 1]
            struct ggml_tensor * Q_prep = ggml_permute(ctx0, Q_u,   0, 2, 1, 3);
            // [feat, feat, head, 1]
            struct ggml_tensor * content_scores = ggml_mul_mat(ctx0, K_prep, Q_prep);
            ggml_format_name(content_scores, "enc_%d_attn_content_scores", il);

            // Add the position bias to Q. Like a baseline for query positions. Like
            // regardless of my specific content I'm always interested in tokens a certain
            // distance away. So we shift the Q by the position bias.
            // [feat, head, time_frames, batch]
            struct ggml_tensor * Q_v = ggml_add(ctx0, Q_cur, model.layers[il].attn_pos_bias_v);
            ggml_format_name(Q_v, "enc_%d_attn_q_v", il);

            // [feat, window_size, 1, 1] and we are doing multi-head attention so
            // we need to split this into heads.
            // [feat, head, window_size, 1]
            pos = ggml_reshape_3d(ctx0, pos, d_head, n_head, pos_emb->ne[1]);

            // [feat, window_size, head, 1]
            pos = ggml_permute(ctx0, pos, 0, 2, 1, 3);
            pos = ggml_cont(ctx0, pos);
            ggml_format_name(pos, "enc_%d_attn_pos_perm", il);
            // [feat, time, head, 1]
            Q_v = ggml_permute(ctx0, Q_v, 0, 2, 1, 3);
            Q_v = ggml_cont(ctx0, Q_v);
            ggml_format_name(Q_v, "enc_%d_attn_q_v_perm", il);

            // [window_size, time_frames, head, 1]
            struct ggml_tensor * rel_pos_scores = ggml_mul_mat(ctx0, pos, Q_v);
            ggml_format_name(rel_pos_scores, "enc_%d_attn_rel_pos", il);

            // Relative positional shift
            {

                const auto pos_window = rel_pos_scores->ne[0];
                const auto n_frame    = rel_pos_scores->ne[1];
                const auto n_head     = rel_pos_scores->ne[2];

                // [feat_padded, window_size, head, 1]
                rel_pos_scores = ggml_pad(ctx0, rel_pos_scores, 1, 0, 0, 0);
                rel_pos_scores = ggml_roll(ctx0, rel_pos_scores, 1, 0, 0, 0);

                rel_pos_scores = ggml_reshape_3d(ctx0, rel_pos_scores, n_frame, pos_window + 1, n_head);
                rel_pos_scores = ggml_cont(ctx0, rel_pos_scores);
                ggml_format_name(rel_pos_scores, "enc_%d_attn_rel_pos_reshaped", il);

                int center = pos_window / 2;
                size_t offset = rel_pos_scores->nb[0] * (center+1);

                rel_pos_scores = ggml_view_3d(ctx0, rel_pos_scores,
                                              n_frame, pos_window, n_head,
                                              (pos_window) * 4,
                                              rel_pos_scores->nb[2],
                                              offset);

                rel_pos_scores = ggml_cont(ctx0, rel_pos_scores);
                ggml_format_name(rel_pos_scores, "enc_%d_attn_rel_pos_shifted", il);

                rel_pos_scores = ggml_view_3d(ctx0, rel_pos_scores,
                                              content_scores->ne[0],
                                              content_scores->ne[1],
                                              rel_pos_scores->ne[2],
                                              rel_pos_scores->nb[1],
                                              rel_pos_scores->nb[2],
                                              0);
                rel_pos_scores = ggml_cont(ctx0, rel_pos_scores);
                ggml_format_name(rel_pos_scores, "enc_%d_attn_rel_pos_shifted_view", il);
            }

            struct ggml_tensor * attn_scores = ggml_add(ctx0, content_scores, rel_pos_scores);
            attn_scores = ggml_cont(ctx0, attn_scores);
            ggml_format_name(attn_scores, "enc_%d_attn_scores", il);
            attn_scores = ggml_scale(ctx0, attn_scores, 1.0f / std::sqrt(d_head));
            attn_scores = ggml_add(ctx0, attn_scores, attn_mask);
            ggml_format_name(attn_scores, "enc_%d_attn_scores_scaled", il);

            struct ggml_tensor * probs = ggml_soft_max(ctx0, attn_scores);
            ggml_format_name(probs, "enc_%d_attn_probs", il);

            V_cur = ggml_cont(ctx0, ggml_permute(ctx0, V_cur, 1, 2, 0, 3));
            ggml_format_name(V_cur, "enc_%d_attn_v_cur", il);
            cur = ggml_mul_mat(ctx0, probs, V_cur);
            ggml_format_name(cur, "enc_%d_attn_inp", il);

            cur = ggml_permute(ctx0, cur, 2, 0, 1, 3);
            cur = ggml_cont_2d(ctx0, cur, n_state, n_time);
            cur = ggml_mul_mat(ctx0, model.layers[il].attn_out_w, cur);
            ggml_format_name(cur, "enc_%d_attn_out", il);

            cur = ggml_add(ctx0, residual, cur);
            ggml_format_name(cur, "enc_%d_attn_res", il);
        }

        // Convolution
        {
            struct ggml_tensor * residual = cur;
            ggml_format_name(cur, "enc_%d_residual_conv", il);

            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].norm_conv_w), model.layers[il].norm_conv_b);
            ggml_format_name(cur, "enc_%d_norm_conv", il);

            // pointwise 1d convolution: [1024, 138] -> [2048, 138]
            cur = ggml_mul_mat(ctx0, model.layers[il].conv_pw1_w, cur);
            ggml_format_name(cur, "enc_%d_conv_pw1", il);

            {
                int64_t d = cur->ne[0] / 2;
                struct ggml_tensor * signal = ggml_view_2d(ctx0, cur, d, cur->ne[1], cur->nb[1], 0);
                struct ggml_tensor * gate   = ggml_view_2d(ctx0, cur, d, cur->ne[1], cur->nb[1], d * cur->nb[0]);

                cur = ggml_mul(ctx0, signal, ggml_sigmoid(ctx0, gate));
                ggml_format_name(cur, "enc_%d_conv_glu", il);
            }

            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

            // use ggml_ssm_conv for f32 precision
            cur = ggml_pad(ctx0, cur, 4, 0, 0, 0);
            cur = ggml_roll(ctx0, cur, 4, 0, 0, 0);
            cur = ggml_pad(ctx0, cur, 4, 0, 0, 0);
            ggml_format_name(cur, "enc_%d_conv_dw_pad", il);

            cur = ggml_ssm_conv(ctx0, cur, model.layers[il].conv_dw_w);
            ggml_format_name(cur, "enc_%d_conv_1d_dw", il);

            cur = ggml_sub(ctx0, cur, model.layers[il].conv_bn_mean);
            struct ggml_tensor * std = ggml_sqrt(ctx0, model.layers[il].conv_bn_var);
            cur = ggml_div(ctx0, cur, std);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].conv_bn_w), model.layers[il].conv_bn_b);
            ggml_format_name(cur, "enc_%d_conv_bn", il);

            cur = ggml_silu(ctx0, cur);
            ggml_format_name(cur, "enc_%d_conv_silu", il);

            cur = ggml_mul_mat(ctx0, model.layers[il].conv_pw2_w, cur);
            ggml_format_name(cur, "enc_%d_conv_pw2", il);

            cur = ggml_add(ctx0, residual, cur);
            ggml_format_name(cur, "enc_%d_conv_res", il);
        }

        // FFN2
        {
            struct ggml_tensor * residual = cur;
            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].norm_ff2_w), model.layers[il].norm_ff2_b);
            ggml_format_name(cur, "enc_%d_ffn_norm_2", il);

            cur = ggml_mul_mat(ctx0, model.layers[il].ff2_linear1_w, cur);
            cur = ggml_silu(ctx0, cur);
            cur = ggml_mul_mat(ctx0, model.layers[il].ff2_linear2_w, cur);
            cur = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, 0.5));
            ggml_format_name(cur, "enc_%d_ffn_res", il);
        }

        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].norm_out_w), model.layers[il].norm_out_b);
    }

    ggml_set_name(cur, "encoder_out");
    ggml_set_output(cur);
    pstate.enc_out = cur;
    pstate.n_frames = cur->ne[1];

    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);

    return gf;
}

static bool parakeet_encode_internal(
        parakeet_context & pctx,
          parakeet_state & pstate,
              const int   mel_offset,
              const int   n_threads,
    ggml_abort_callback   abort_callback,
                   void * abort_callback_data) {
    const int64_t t_start_us = ggml_time_us();

    // conv
    {
        auto & sched = pstate.sched_conv.sched;

        ggml_cgraph * gf = parakeet_build_graph_conv(pctx, pstate);

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }

        struct ggml_tensor * mel = ggml_graph_get_tensor(gf, "mel");

        // set the input
        {
            const auto & mel_inp = pstate.mel;
            const int n_ctx      = pstate.n_audio_ctx > 0 ? pstate.n_audio_ctx : pctx.model.hparams.n_audio_ctx;

            assert(mel->type == GGML_TYPE_F32);
            assert(mel_inp.n_mel == pctx.model.hparams.n_mels);

            pstate.inp_mel.resize(ggml_nelements(mel));

            float * dst = pstate.inp_mel.data();
            memset(dst, 0, ggml_nbytes(mel));

            const int i0 = std::min(mel_offset,         mel_inp.n_len);
            const int i1 = std::min(mel_offset + n_ctx, mel_inp.n_len);

            const int n_frames_to_copy = i1 - i0;
            memcpy(dst, mel_inp.data.data() + i0 * mel_inp.n_mel, n_frames_to_copy * mel_inp.n_mel * sizeof(float));

            ggml_backend_tensor_set(mel, pstate.inp_mel.data(), 0, ggml_nelements(mel)*sizeof(float));
        }

        if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            return false;
        }
    }

    // encoder
    auto & sched = pstate.sched_encode.sched;

    ggml_cgraph * gf = parakeet_build_graph_encoder(pctx, pstate);

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        // should never happen as we pre-allocate the memory
        return false;
    }

    // set the inputs
    {
        struct ggml_tensor * attn_mask = ggml_graph_get_tensor(gf, "attn_mask");
        const int n_q = attn_mask->ne[1];
        const int n_k = attn_mask->ne[0];

        const int32_t subsampl_factor = pctx.model.hparams.subsampling_factor;
        const int n_tokens_real = (pstate.mel.n_len_org + subsampl_factor-1) / subsampl_factor;

        std::vector<float> mask_data(n_q * n_k);
        const float mask_value = -1e30f;

        for (int q = 0; q < n_q; ++q) {
            for (int k = 0; k < n_k; ++k) {
                bool is_padding = (k >= n_tokens_real);

                mask_data[q * n_k + k] = (is_padding) ? mask_value : 0.0f;
            }
        }
        ggml_backend_tensor_set(attn_mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    }

    if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
        return false;
    }

    pstate.t_encode_us += ggml_time_us() - t_start_us;
    pstate.n_encode++;

    return !(abort_callback && abort_callback(abort_callback_data));
}

static struct ggml_tensor * parakeet_build_graph_lstm_layer(
        struct ggml_context * ctx0,
         struct ggml_cgraph * gf,
         struct ggml_tensor * x_t,       // the current input token embedding
         struct ggml_tensor * w_ih,      // input to hidden weights (4 weight tensors packed)
         struct ggml_tensor * b_ih,      // input to hidden bias (4 bias packed together)
         struct ggml_tensor * w_hh,      // hidden to hidden weights (4 weight tensors packed)
         struct ggml_tensor * b_hh,      // hidden to hidden bias (4 bias tensors packed)
         struct ggml_tensor * h_state,   // this layers hidden state
         struct ggml_tensor * c_state,   // this layers cell state
                        int   li) {      // layer index (for tensor naming)

    ggml_format_name(x_t, "lstm_layer_%d_x_t", li);
    ggml_format_name(h_state, "lstm_layer_%d_h_state", li);
    ggml_format_name(c_state, "lstm_layer_%d_c_state", li);

    // Input/Forget/Cell/Output gates are packed in the same weight tensor.
    struct ggml_tensor * inp_gates = ggml_mul_mat(ctx0, w_ih, x_t);
    inp_gates = ggml_add(ctx0, inp_gates, b_ih);

    // Hidden-to-Hidden Projections are also packed in the same weight tensor.
    struct ggml_tensor * hid_gates = ggml_mul_mat(ctx0, w_hh, h_state);
    hid_gates = ggml_add(ctx0, hid_gates, b_hh);

    // Combine the input and hidden contributions of the gates.
    struct ggml_tensor * gates = ggml_add(ctx0, inp_gates, hid_gates);
    ggml_format_name(gates, "lstm_layer_%d_gates", li);

    const int h_dim = h_state->ne[0];
    const size_t row_size = ggml_row_size(gates->type, h_dim);

    // 1. Input Gate at time t.
    struct ggml_tensor * i_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, gates, h_dim, 0 * row_size));
    ggml_format_name(i_t, "lstm_layer_%d_i_t", li);

    // Forget gate.
    struct ggml_tensor * f_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, gates, h_dim, 1 * row_size));
    ggml_format_name(f_t, "lstm_layer_%d_f_t", li);

    // Cell gate.
    struct ggml_tensor * c_t = ggml_tanh(ctx0,    ggml_view_1d(ctx0, gates, h_dim, 2 * row_size));
    ggml_format_name(c_t, "lstm_layer_%d_c_t", li);

    // Output gate.
    struct ggml_tensor * o_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, gates, h_dim, 3 * row_size));
    ggml_format_name(o_t, "lstm_layer_%d_o_t", li);

    // Calculate the new cell state.
    struct ggml_tensor * c_new = ggml_add(ctx0,
        ggml_mul(ctx0, f_t, c_state), // apply forget gate to cell state.
        ggml_mul(ctx0, i_t, c_t));    // apply input gate to cell gate.
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, c_new, c_state));

    // Calculate the new hidden state.
    struct ggml_tensor * h_new = ggml_mul(ctx0, o_t, ggml_tanh(ctx0, c_new));
    ggml_set_output(h_new);
    ggml_format_name(h_new, "lstm_layer_%d_h_new", li);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, h_new, h_state));

    return h_new;
}

static struct ggml_cgraph * parakeet_build_graph_prediction(
         parakeet_context & pctx,
           parakeet_state & pstate,
     const parakeet_batch & batch,
                    bool   worst_case) {
    GGML_UNUSED(worst_case);
    const auto & model   = pctx.model;
    const auto & hparams = model.hparams;
    const int n_tokens   = batch.n_tokens;

    struct ggml_init_params params = {
        /*.mem_size   =*/ pstate.sched_decode.meta.size(),
        /*.mem_buffer =*/ pstate.sched_decode.meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, PARAKEET_MAX_NODES, false);

    // Prediction Network
    struct ggml_tensor * token = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(token, "token_inp");
    ggml_set_input(token);

    struct ggml_tensor * token_embd = ggml_get_rows(ctx0, model.prediction.embed_w, token);
    ggml_set_input(token_embd);

    struct ggml_tensor * inpL = token_embd;

    for (int il = 0; il < hparams.n_pred_layers; ++il) {
        inpL = parakeet_build_graph_lstm_layer(ctx0, gf, inpL,
                model.prediction.lstm_layer[il].ih_w,
                model.prediction.lstm_layer[il].ih_b,
                model.prediction.lstm_layer[il].hh_w,
                model.prediction.lstm_layer[il].hh_b,
                pstate.lstm_state.layer[il].h_state,
                pstate.lstm_state.layer[il].c_state,
                il);
    }

    struct ggml_tensor * pred_out = inpL;
    ggml_format_name(pred_out, "lstm_pred_out");

    // Project the prediction network output to the joint network hidden dimension.
    struct ggml_tensor * pred = ggml_mul_mat(ctx0, model.joint.pred_w, pred_out);
    pred = ggml_add(ctx0, pred, model.joint.pred_b);
    ggml_set_output(pred);
    ggml_set_name(pred, "h_pred");

    ggml_build_forward_expand(gf, pred);

    ggml_free(ctx0);

    return gf;
}

static struct ggml_cgraph * parakeet_build_graph_joint(
         parakeet_context & pctx,
           parakeet_state & pstate,
     const parakeet_batch & batch,
                     bool   worst_case) {
    GGML_UNUSED(worst_case);
    GGML_UNUSED(batch);
    const auto & model   = pctx.model;
    const auto & hparams = model.hparams;

    struct ggml_init_params params = {
        /*.mem_size   =*/ pstate.sched_decode.meta.size(),
        /*.mem_buffer =*/ pstate.sched_decode.meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, PARAKEET_MAX_NODES, false);

    struct ggml_tensor * pred = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hparams.n_pred_dim);
    ggml_set_name(pred, "pred");
    ggml_set_input(pred);

    struct ggml_tensor * enc_out = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hparams.n_audio_state);
    ggml_set_name(enc_out, "enc_out");
    ggml_set_input(enc_out);

    // Project the encoder output to the joint network hidden dimension.
    struct ggml_tensor * enc  = ggml_mul_mat(ctx0, model.joint.enc_w, enc_out);
    enc = ggml_add(ctx0, enc, model.joint.enc_b);
    ggml_set_name(enc, "enc");

    struct ggml_tensor * joint = ggml_add(ctx0, enc, pred);
    ggml_set_name(joint, "joint");
    joint = ggml_relu(ctx0, joint);

    struct ggml_tensor * logits = ggml_mul_mat(ctx0, model.joint.net_w, joint);
    logits = ggml_add(ctx0, logits, model.joint.net_b);
    ggml_set_output(logits);
    ggml_set_name(logits, "logits");

    struct ggml_tensor * probs = ggml_soft_max(ctx0, logits);
    struct ggml_tensor * log_probs = ggml_log(ctx0, probs);
    ggml_set_output(log_probs);
    ggml_format_name(log_probs, "log_probs");

    ggml_build_forward_expand(gf, log_probs);

    ggml_free(ctx0);

    return gf;
}

static bool parakeet_predict(
        parakeet_context & pctx,
          parakeet_state & pstate,
    const parakeet_batch & batch,
               const int   n_threads,
     ggml_abort_callback   abort_callback,
                   void  * abort_callback_data) {

    const int n_tokens   = batch.n_tokens;

    {
        auto & sched = pstate.sched_decode.sched;

        ggml_cgraph * gf = parakeet_build_graph_prediction(pctx, pstate, batch, false);

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }

        // set the inputs
        {
            struct ggml_tensor * token_inp = ggml_graph_get_tensor(gf, "token_inp");
            ggml_backend_tensor_set(token_inp, batch.token, 0, n_tokens * ggml_element_size(token_inp));
        }

        if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            return false;
        }

        // Copy h_pred output to pstate.pred_out for use in joint network
        {
            struct ggml_tensor * h_pred = ggml_graph_get_tensor(gf, "h_pred");
            ggml_backend_tensor_copy(h_pred, pstate.pred_out);
        }

    }

    return !(abort_callback && abort_callback(abort_callback_data));
}

static bool parakeet_joint(
         parakeet_context & pctx,
           parakeet_state & pstate,
     const parakeet_batch & batch,
                const int   n_threads,
      ggml_abort_callback   abort_callback,
                     void * abort_callback_data) {
    const int64_t t_start_us = ggml_time_us();

    const auto & model   = pctx.model;
    const auto & hparams = model.hparams;
    const int n_tokens   = batch.n_tokens;

    auto & logits_out = pstate.logits;

    struct ggml_tensor * logits;

    {
        auto & sched = pstate.sched_decode.sched;

        ggml_cgraph * gf = parakeet_build_graph_joint(pctx, pstate, batch, false);

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }

        // set the inputs
        {
            struct ggml_tensor * pred = ggml_graph_get_tensor(gf, "pred");
            if (n_tokens == 1) {
                ggml_backend_tensor_copy(pstate.pred_out, pred);
            }

            struct ggml_tensor * enc_out_inp = ggml_graph_get_tensor(gf, "enc_out");
            if (n_tokens == 1) {
                const int t_idx = batch.i_time[0];
                const int d_enc = hparams.n_audio_state;

                std::vector<float> frame_buf(d_enc);

                // If we have accumulated encoder output (chunked processing), use that
                if (!pstate.enc_out_buffer.empty() && t_idx < pstate.enc_out_frames) {
                    const float * src = pstate.enc_out_buffer.data() + (t_idx * d_enc);
                    std::copy(src, src + d_enc, frame_buf.begin());
                } else {
                    ggml_backend_tensor_get(pstate.enc_out, frame_buf.data(), t_idx * d_enc * sizeof(float), d_enc * sizeof(float));
                }

                ggml_backend_tensor_set(enc_out_inp, frame_buf.data(), 0, d_enc * sizeof(float));
            }
        }

        logits = ggml_graph_node(gf, -1);

        if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            return false;
        }

    }

    const int n_logits = hparams.n_vocab + hparams.n_tdt_durations + 1; // one for the blank token
    logits_out.resize(n_tokens * n_logits);
    for (int i = 0; i < n_tokens; i++) {
        if (batch.logits[i] == 0) {
            continue;
        }
        ggml_backend_tensor_get(logits, logits_out.data() + (n_logits*i), sizeof(float)*(n_logits*i), sizeof(float)*n_logits);
    }

    if (batch.n_tokens == 1) {
        pstate.t_decode_us += ggml_time_us() - t_start_us;
        pstate.n_decode++;
    }

    return !(abort_callback && abort_callback(abort_callback_data));
}

static bool is_word_start_token(parakeet_vocab & vocab, parakeet_token token_id) {
    const std::string & token_str = vocab.id_to_token[token_id];
    // check if it starts with the SentencePiece meta-space "▁" (U+2581) or 3-byte UTF-8 character: 0xE2 0x96 0x81
    if (!token_str.empty()) {
        if (token_str.find("\xE2\x96\x81") == 0 || token_str[0] == '_') {
            return true;
        }
    }
    return false;
}

static bool is_punctuation_token(parakeet_vocab & vocab, parakeet_token token_id) {
    const std::string & token_str = vocab.id_to_token[token_id];
    static const std::string punct_chars = ".,!?;:'\"-()[]{}";

    if (token_str.empty()) {
        return false;
    }

    std::string clean_token = token_str;
    if (clean_token.find("\xE2\x96\x81") == 0) {
        clean_token = clean_token.substr(3); // Remove the 3-byte UTF-8 character
    } else if (clean_token[0] == '_') {
        clean_token = clean_token.substr(1);
    }

    return clean_token.length() == 1 && punct_chars.find(clean_token[0]) != std::string::npos;
}

// Collapse punctuation timestamps to match the original Parakeet model.
// Punctuations symbols like ',', '.' and others are not spoken words but the
// model will still produce a duration for these tokens. But since these are
// non-spoken we collapse the timesstamps so that they don't have an time duration.
static void refine_timestamps_tdt(parakeet_vocab & vocab, std::vector<parakeet_token_data> & tokens) {
    if (tokens.empty()) {
        return;
    }

    int64_t last_non_punct_t1 = -1;

    for (size_t i = 0; i < tokens.size(); ++i) {
        if (is_punctuation_token(vocab, tokens[i].id)) {
            if (last_non_punct_t1 >= 0) {
                tokens[i].t0 = last_non_punct_t1;
                tokens[i].t1 = last_non_punct_t1;
            }
        } else {
            last_non_punct_t1 = tokens[i].t1;
        }
    }
}

static parakeet_token_data create_token_data(
            parakeet_context & pctx,
              parakeet_state & pstate,
               parakeet_token   token_id,
                          int   duration_idx,
                          int   duration_value,
                          int   frame_index,
                        float   token_logit,
                          int   n_vocab_logits) {

    float token_sum = 0.0f;
    for (int i = 0; i < n_vocab_logits; ++i) {
        token_sum += expf(pstate.logits[i]);
    }
    float token_p = expf(token_logit) / token_sum;

    parakeet_token_data token_data;
    token_data.id = token_id;
    token_data.duration_idx = duration_idx;
    token_data.duration_value = duration_value;
    token_data.frame_index = frame_index;
    token_data.p = token_p;
    token_data.plog = token_logit;
    token_data.t0 = frame_index * pctx.model.hparams.subsampling_factor;
    token_data.t1 = (frame_index + duration_value) * pctx.model.hparams.subsampling_factor;
    token_data.is_word_start = is_word_start_token(pctx.vocab, token_id);

    return token_data;
}

static bool parakeet_decode(
              parakeet_context & pctx,
                parakeet_state & pstate,
                parakeet_batch & batch,
                     const int   n_threads,
    const parakeet_full_params * params = nullptr) {
    const auto & hparams       = pctx.model.hparams;
    const auto & tdt_durations = pctx.model.tdt_durations;

    const int  n_tdt_durations          = hparams.n_tdt_durations;
    const int  n_frames                 = pstate.n_frames;
    const int  blank_id                 = pctx.vocab.token_blank;
    const int  n_vocab_logits           = blank_id + 1;
    const int  max_tokens_per_timestep = hparams.n_max_tokens;

    // time index into the encoder frame (current time frame)
    int t = 0;
    // number of symbols emitted for the current time frame
    int tokens_emitted = 0;

    // Start with the blank token (8192)
    parakeet_token last_token = blank_id;

    PARAKEET_LOG_DEBUG("parakeet_decode: starting decode with n_frames=%d\n", n_frames);

    batch.n_tokens  = 1;
    batch.token[0]  = last_token;
    batch.logits[0] = 1;
    batch.i_time[0] = 0;

    // run the prediction network for the initial blank token. This will
    // initialize the LSTM state and produce an initial hidden state that can
    // be used in the joint network below.
    if (!parakeet_predict(pctx, pstate, batch, n_threads, nullptr, nullptr)) {
        return false;
    }

    // process all time frames of the encoder output
    while (t < n_frames) {
        batch.n_tokens  = 1;
        batch.i_time[0] = t;
        batch.logits[0] = 1;

        // Use the current encoder frame (t) and the output of the prediction to
        // generate probabilities for the next token and duration. batch.i_time
        // is used in to select the correct frame from the encoder output.
        // The joint network outputs logits for all the tokens in the vocabulary
        // plus the blank token, and also n_duration logits for the duration
        // tokens which contain information about how many frames to skip/advance forward.
        if (!parakeet_joint(pctx, pstate, batch, n_threads, nullptr, nullptr)) {
            return false;
        }

        // find the best token (greedy).
        // TODO: implement beam search?
        int best_token = 0;
        float max_logit = -1e10f;
        for (int i = 0; i < n_vocab_logits; ++i) {
            if (pstate.logits[i] > max_logit) {
                max_logit = pstate.logits[i];
                best_token = i;
            }
        }

        // find the max index of the duration logits, and look up that index
        // value in the tdt_durations array to get the actual duration value.
        int best_duration_idx = 0;
        float best_duration_logit = -1e10f;
        for (int i = 0; i < n_tdt_durations; ++i) {
            if (pstate.logits[n_vocab_logits + i] > best_duration_logit) {
                best_duration_logit = pstate.logits[n_vocab_logits + i];
                best_duration_idx = i;
            }
        }
        // look up that max duration index value in the tdt_durations array to
        // get the actual duration value.
        int duration = tdt_durations[best_duration_idx];

        if (best_token == blank_id) {
            if (duration == 0) {
                duration = 1;
            }
            // skip forward by duration time frames.
            t += duration;
            // reset symbols emitted counter
            tokens_emitted = 0;
            // continue without predicting.
            continue;
        }

        // Emit non-blank token at current frame t.
        pstate.decoded_tokens.push_back(best_token);
        pstate.n_sample++;

        parakeet_token_data token_data = create_token_data(
            pctx, pstate, best_token, best_duration_idx, duration, t,
            max_logit, n_vocab_logits);

        pstate.decoded_token_data.push_back(token_data);

        // Call token callback if registered (for real-time streaming)
        if (params && params->new_token_callback) {
            params->new_token_callback(&pctx, &pstate, &token_data, params->new_token_callback_user_data);
        }

        last_token = best_token;

        // advance predictor for the non-blank token.
        batch.token[0] = last_token;
        if (!parakeet_predict(pctx, pstate, batch, n_threads, nullptr, nullptr)) {
            return false;
        }

        // if duration greater than 0, continue looping over the encoder frames
        // and skip to the updated time frame (t).
        if (duration > 0) {
            t += duration;
            tokens_emitted = 0;
            continue;
        }

        // if duration is zero we stay on the current time frame.
        tokens_emitted++;
        if (tokens_emitted >= max_tokens_per_timestep) {
            t += 1; // forced blank/time advance behavior
            tokens_emitted = 0;
        }
    }

    return true;
}

static bool parakeet_decode_chunk(
                  parakeet_context & pctx,
                    parakeet_state & pstate,
                    parakeet_batch & batch,
                        const int    chunk_enc_frames,
                        const int    n_threads,
        const parakeet_full_params * params = nullptr) {
    const auto & hparams = pctx.model.hparams;
    const auto & tdt_durations = pctx.model.tdt_durations;

    const int blank_id = pctx.vocab.token_blank;
    const int n_vocab_logits = blank_id + 1;
    const int max_tokens_per_timestep = hparams.n_max_tokens;

    // time index into the encoder frame (current time frame)
    int t = pstate.tdt_stream_state.time_step;
    parakeet_token last_token = pstate.tdt_stream_state.last_token;

    PARAKEET_LOG_DEBUG("%s: chunk_frames=%d, start_t=%d, last_token=%d, initialized=%d\n",
        __func__, chunk_enc_frames, t, last_token, pstate.tdt_stream_state.initialized);

    if (!pstate.tdt_stream_state.initialized) {
        // Start with the blank token (8192)
        last_token = blank_id;
        t = 0;

        batch.n_tokens  = 1;
        batch.token[0]  = last_token;
        batch.logits[0] = 0;
        batch.i_time[0] = 0;

        // run the prediction network for the initial blank token. This will
        // initialize the LSTM state and produce an initial hidden state that can
        // be used in the joint network.
        if (!parakeet_predict(pctx, pstate, batch, n_threads, nullptr, nullptr)) {
            return false;
        }
        pstate.tdt_stream_state.initialized = true;
        pstate.tdt_stream_state.last_token  = last_token;
    }

    // Count non-blank emissions at the same encoder frame to mimic max_symbols handling.
    int last_emission_time = -1;
    int tokens_at_time    = 0;

    // loop over all the time frames the encoder produced for this chunk.
    while (t < chunk_enc_frames) {
        int current_token_time = t;

        int chosen_token = blank_id;
        int chosen_duration = 1;
        int chosen_duration_idx = 0;
        float chosen_token_logit = 0.0f;

        // inner loop handles the joint network
        while (t < chunk_enc_frames) {
            batch.n_tokens  = 1;
            batch.i_time[0] = std::min(t, chunk_enc_frames - 1);
            batch.logits[0] = 1;

            // Use the current encoder frame (t) and the output of the prediction to
            // generate probabilities for the next token and duration.
            // The joint network outputs logits for all the tokens in the vocabulary
            // and the blank token, and also addtionally N duration logits for the duration
            // tokens which contain information about how many frames to skip/advance forward.
            if (!parakeet_joint(pctx, pstate, batch, n_threads, nullptr, nullptr)) {
                return false;
            }

            // find the token with the highest value.
            int best_token         = 0;
            float best_token_logit = -1e10f;
            for (int i = 0; i < n_vocab_logits; ++i) {
                if (pstate.logits[i] > best_token_logit) {
                    best_token_logit = pstate.logits[i];
                    best_token = i;
                }
            }

            // find the max index of the duration logits, and look up that index
            // value in the tdt_durations array to get the actual duration value.
            int best_duration_idx = 0;
            float best_duration_logit = -1e10f;
            for (int i = 0; i < hparams.n_tdt_durations; ++i) {
                const float v = pstate.logits[n_vocab_logits + i];
                if (v > best_duration_logit) {
                    best_duration_logit = v;
                    best_duration_idx = i;
                }
            }
            // look up that max duration index value in the tdt_durations array to
            // get the actual duration value.
            int duration = tdt_durations[best_duration_idx];

            // handle blank token output from the joint network.
            if (best_token == blank_id) {
                if (duration == 0) {
                    duration = 1;
                }

                // always add the duration.
                t += duration;

                // if blank duration jumped past the chunk break.
                if (t >= chunk_enc_frames) {
                    break;
                }

                // we continue with the current frame
                current_token_time = t;
                continue;
            }

            // found the next non-blank token and label for this outer step.
            chosen_token = best_token;
            chosen_duration = duration;
            chosen_token_logit = best_token_logit;
            chosen_duration_idx = best_duration_idx;
            break;
        }

        // if we exited because blank durations ran beyond the chunk, no label to emit.
        if (chosen_token == blank_id) {
            break;
        }

        // store token at the frame where it was found.
        pstate.decoded_tokens.push_back(chosen_token);
        pstate.n_sample++;

        parakeet_token_data token_data = create_token_data(
            pctx, pstate, chosen_token, chosen_duration_idx, chosen_duration, current_token_time,
            chosen_token_logit, n_vocab_logits);

        pstate.decoded_token_data.push_back(token_data);

        // emit the token to the callback if registered.
        if (params && params->new_token_callback) {
            params->new_token_callback(&pctx, &pstate, &token_data, params->new_token_callback_user_data);
        }

        // advance predictor for the non-blank token.
        last_token = chosen_token;
        pstate.tdt_stream_state.last_token = last_token;

        batch.n_tokens  = 1;
        batch.token[0]  = last_token;
        batch.logits[0] = 0;
        batch.i_time[0] = 0;

        if (!parakeet_predict(pctx, pstate, batch, n_threads, nullptr, nullptr)) {
            return false;
        }

        if (chosen_duration > 0) {
            // if the duration looked up in the duration array was greater than zero
            t += chosen_duration;
            last_emission_time = -1;
            tokens_at_time = 0;
        } else {
            // the duration looked up in the duration array was zero
            if (current_token_time == last_emission_time) {
                // update the token count as it can be valid for the model
                // to emit a token with a zero duration.
                tokens_at_time++;
            } else {
                // update the current emission time and reset the token count.
                last_emission_time = current_token_time;
                tokens_at_time = 1;
            }

            // handle the case where the model is emitting too many tokens with
            // zero duration to force progress.
            if (tokens_at_time >= max_tokens_per_timestep) {
                t += 1;
                last_emission_time = -1;
                tokens_at_time = 0;
            }
        }
    }

    pstate.tdt_stream_state.time_step = t - chunk_enc_frames;
    pstate.tdt_stream_state.decoded_length += chunk_enc_frames;

    PARAKEET_LOG_DEBUG("%s: finished t=%d, time_step=%d, decoded_length=%d, total_tokens=%zu\n",
            __func__,
            t,
            pstate.tdt_stream_state.time_step,
            pstate.tdt_stream_state.decoded_length,
            pstate.decoded_tokens.size());

    return true;
}

//  500 -> 00:05.000
// 6000 -> 01:00.000
static std::string to_timestamp(int64_t t, bool comma = false) {
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);

    return std::string(buf);
}

// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued
static void dft(const float* in, int N, float* out, const parakeet_mel_cache & cache) {
    const int sin_cos_step = cache.n_fft / N;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int idx = (k * n * sin_cos_step) % cache.n_fft; // t = 2*M_PI*k*n/N
            re += in[n]*cache.cos_vals[idx]; // cos(t)
            im -= in[n]*cache.sin_vals[idx]; // sin(t)
        }

        out[k*2 + 0] = re;
        out[k*2 + 1] = im;
    }
}

// Cooley-Tukey FFT
// poor man's implementation - use something better
// input is real-valued
// output is complex-valued
static void fft(float* in, int N, float* out, const parakeet_mel_cache & cache) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    const int half_N = N / 2;
    if (N - half_N*2 == 1) {
        dft(in, N, out, cache);
        return;
    }

    float* even = in + N;
    for (int i = 0; i < half_N; ++i) {
        even[i]= in[2*i];
    }
    float* even_fft = out + 2 * N;
    fft(even, half_N, even_fft, cache);

    float* odd = even;
    for (int i = 0; i < half_N; ++i) {
        odd[i] = in[2*i + 1];
    }
    float* odd_fft = even_fft + N;
    fft(odd, half_N, odd_fft, cache);

    const int sin_cos_step = cache.n_fft / N;
    for (int k = 0; k < half_N; k++) {
        int idx = k * sin_cos_step; // t = 2*M_PI*k/N
        float re = cache.cos_vals[idx]; // cos(t)
        float im = -cache.sin_vals[idx]; // sin(t)

        float re_odd = odd_fft[2*k + 0];
        float im_odd = odd_fft[2*k + 1];

        out[2*k + 0] = even_fft[2*k + 0] + re*re_odd - im*im_odd;
        out[2*k + 1] = even_fft[2*k + 1] + re*im_odd + im*re_odd;

        out[2*(k + half_N) + 0] = even_fft[2*k + 0] - re*re_odd + im*im_odd;
        out[2*(k + half_N) + 1] = even_fft[2*k + 1] - re*im_odd - im*re_odd;
    }
}

static void log_mel_spectrogram_worker_thread(
                             int   ith,
                     const float * window_func,
                             int   window_size,
        const std::vector<float> & samples,
                             int   n_samples,
                             int   frame_size,
                             int   frame_step,
                             int   n_threads,
          const parakeet_filters & filters,
                    parakeet_mel & mel,
        const parakeet_mel_cache & cache) {
    std::vector<float> fft_in(frame_size * 2, 0.0);
    std::vector<float> fft_out(frame_size * 2 * 2 * 2);

    int n_fb = filters.n_fb;  // number of frequency bins
    int i = ith;

    // make sure n_fb == 1 + (frame_size / 2), bin_0 to bin_nyquist
    assert(n_fb == 1 + (frame_size / 2));

    const double eps = 5.960464477539063e-08;

    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, mel.n_len); i += n_threads) {
        const int offset = i * frame_step;

        const int window_pad_left = (frame_size - window_size) / 2;

        // Zero-pad left
        std::fill(fft_in.begin(), fft_in.begin() + window_pad_left, 0.0f);

        // Apply windowed samples in the center
        const int n_to_process = std::min({window_size, n_samples - offset});
        for (int j = 0; j < n_to_process; j++) {
            fft_in[window_pad_left + j] = window_func[j] * samples[offset + window_pad_left + j];
        }

        // Zero-pad right (and any samples we didn't have)
        std::fill(fft_in.begin() + window_pad_left + n_to_process, fft_in.begin() + frame_size, 0.0f);

        // FFT
        fft(fft_in.data(), frame_size, fft_out.data(), cache);

        // Calculate modulus^2 of complex numbers
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < n_fb; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }

        // mel spectrogram
        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;
            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fb - 3; k += 4) {
                sum +=
                        fft_out[k + 0] * filters.data[j * n_fb + k + 0] +
                        fft_out[k + 1] * filters.data[j * n_fb + k + 1] +
                        fft_out[k + 2] * filters.data[j * n_fb + k + 2] +
                        fft_out[k + 3] * filters.data[j * n_fb + k + 3];
            }
            // handle n_fb remainder
            for (; k < n_fb; k++) {
                sum += fft_out[k] * filters.data[j * n_fb + k];
            }

            mel.data[i * mel.n_mel + j] = std::log(sum + eps);
        }
    }

    // Otherwise fft_out are all zero - use log(eps) for consistency
    const double empty_sum = std::log(eps);
    for (; i < mel.n_len; i += n_threads) {
        for (int j = 0; j < mel.n_mel; j++) {
            mel.data[i * mel.n_mel + j] = empty_sum;
        }
    }
}

static bool log_mel_spectrogram(
                  parakeet_state & wstate,
                     const float * samples,
                       const int   n_samples,
                       const int   /*sample_rate*/,
                       const int   frame_size,
                       const int   frame_step,
                       const int   n_mel,
                       const int   n_threads,
          const parakeet_filters & filters,
                      const bool   debug,
                    parakeet_mel & mel,
        const parakeet_mel_cache & cache) {
    const int64_t t_start_us = ggml_time_us();

    const float * window_func = cache.window.empty() ? cache.hann_window.data() : cache.window.data();
    const int window_size = cache.window.empty() ? cache.n_fft : cache.window.size();

    std::vector<float> samples_preprocessed(samples, samples + n_samples);

    // Apply preemphasis filter (high-pass): x[i] = x[i] - 0.97 * x[i-1]
    {
        const float preemph = 0.97f;
        for (int i = n_samples - 1; i > 0; i--) {
            samples_preprocessed[i] = samples_preprocessed[i] - preemph * samples_preprocessed[i - 1];
        }
    }

    // Parakeet Pytorch implementation uses centered contant padding.
    const int pad = frame_size / 2;
    std::vector<float> samples_padded(n_samples + 2 * pad); // 176512
    std::fill(samples_padded.begin(), samples_padded.begin() + pad, 0.0f);
    std::fill(samples_padded.begin() + pad + n_samples, samples_padded.end(), 0.0f);
    std::copy(samples_preprocessed.begin(), samples_preprocessed.end(), samples_padded.begin() + pad);

    mel.n_mel = n_mel;
    mel.n_len = (samples_padded.size() - frame_size) / frame_step + 1;
    mel.n_len_org = mel.n_len;
    mel.data.resize(mel.n_mel * mel.n_len);

    // Worker Threads (STFT + Mel + Natural Log)
    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw] = std::thread(log_mel_spectrogram_worker_thread,
                    iw + 1,                      // thread index
                    window_func,
                    window_size,
                    std::cref(samples_padded),
                    samples_padded.size(),
                    frame_size,
                    frame_step,
                    n_threads,
                    std::cref(filters),
                    std::ref(mel),
                    std::cref(cache));
        }

        log_mel_spectrogram_worker_thread(
                0,
                window_func,
                window_size,
                samples_padded,
                samples_padded.size(),
                frame_size,
                frame_step,
                n_threads,
                filters,
                mel,
                cache);

        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw].join();
        }
    }

    {
        const double eps = 1e-5;
        int valid_frames = n_samples / frame_step;

        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;
            double sq_diff_sum = 0.0;

            // Calculate Mean ONLY on valid audio frames
            for (int i = 0; i < valid_frames; i++) {
                sum += (double)mel.data[i * mel.n_mel + j];
            }
            double mean = sum / valid_frames;

            // Calculate Variance ONLY on valid audio frames
            for (int i = 0; i < valid_frames; i++) {
                double diff = (double)mel.data[i * mel.n_mel + j] - mean;
                sq_diff_sum += diff * diff;
            }

            double std_dev = std::sqrt(sq_diff_sum / (valid_frames - 1.0));
            double denominator = std_dev + eps;

            // Apply to ALL frames (including the padded ones)
            for (int i = 0; i < mel.n_len; i++) {
                mel.data[i * mel.n_mel + j] = (float)((mel.data[i * mel.n_mel + j] - mean) / denominator);
            }
        }
    }

    wstate.t_mel_us += ggml_time_us() - t_start_us;

    if (debug) {
        std::ofstream outFile("log_mel_spectrogram.json");
        outFile << "[";
        for (uint64_t i = 0; i < mel.data.size() - 1; i++) {
            outFile << mel.data[i] << ", ";
        }
        outFile << mel.data[mel.data.size() - 1] << "]";
        outFile.close();
    }

    return true;
}

static std::vector<parakeet_vocab::id> tokenize(const parakeet_vocab & vocab, const std::string & text) {
    std::vector<parakeet_vocab::id> tokens;
    const std::string normalized = sentencepiece_normalize(text);

    size_t i = 0;
    while (i < normalized.size()) {
        const size_t remaining = normalized.size() - i;
        const size_t max_len = std::min(vocab.max_token_length, remaining);

        bool found = false;
        for (size_t len = max_len; len > 0; --len) {
            const auto it = vocab.token_to_id.find(normalized.substr(i, len));
            if (it != vocab.token_to_id.end() && !is_sentencepiece_control(it->first)) {
                tokens.push_back(it->second);
                i += len;
                found = true;
                break;
            }
        }

        if (!found) {
            if (vocab.token_unk >= 0) {
                tokens.push_back(vocab.token_unk);
            }

            const unsigned char c = static_cast<unsigned char>(normalized[i]);
            i += utf8_codepoint_len(c);
        }
    }

    return tokens;
}


//
// interface implementation
//

struct parakeet_state * parakeet_init_state(parakeet_context * ctx) {
    parakeet_state * state = new parakeet_state;

    state->backends = whisper_backend_init(ctx->params);
    if (state->backends.empty()) {
        PARAKEET_LOG_ERROR("%s: whisper_backend_init() failed\n", __func__);
        parakeet_free_state(state);
        return nullptr;
    }

    const int batch_size = ctx->model.hparams.n_audio_ctx;

    state->logits.reserve(ctx->vocab.n_vocab * batch_size);

    state->batch = parakeet_batch_init(batch_size);

    // conv allocator
    {
        bool ok = parakeet_sched_graph_init(state->sched_conv, state->backends,
                [&]() {
                    return parakeet_build_graph_conv(*ctx, *state);
                });

        if (!ok) {
            PARAKEET_LOG_ERROR("%s: failed to init conv allocator\n", __func__);
            parakeet_free_state(state);
            return nullptr;
        }

        PARAKEET_LOG_INFO("%s: compute buffer (conv)   = %7.2f MB\n", __func__, parakeet_sched_size(state->sched_conv) / 1e6);
    }

    // encoder allocator
    bool ok = parakeet_sched_graph_init(state->sched_encode, state->backends,
            [&]() {
                return parakeet_build_graph_encoder(*ctx, *state);
            });

    if (!ok) {
        PARAKEET_LOG_ERROR("%s: failed to init encoder allocator\n", __func__);
        parakeet_free_state(state);
        return nullptr;
    }

    if (!parakeet_lstm_state_init(*state, state->backends[0], ctx->model.hparams.n_pred_layers)) {
        PARAKEET_LOG_ERROR("%s: parakeet_lstm_states_init () failed\n", __func__);
        parakeet_free_state(state);
        return nullptr;
    }

    {
        const size_t mem_lstm_ctx = state->lstm_state.ctx_buf.size();
        const size_t mem_lstm_buf = ggml_backend_buffer_get_size(state->lstm_state.buffer);
        PARAKEET_LOG_INFO("%s: lstm state: %7.2f MB (meta) + %7.2f MB (data)\n", __func__,
                mem_lstm_ctx / 1024.0 / 1024.0, mem_lstm_buf / 1024.0 / 1024.0);
    }

    if (!parakeet_pred_state_init(*state, state->backends[0], ctx->model.hparams.n_pred_dim)) {
        PARAKEET_LOG_ERROR("%s: parakeet_pred_state_init() failed\n", __func__);
        parakeet_free_state(state);
        return nullptr;
    }

    {
        const size_t mem_pred_ctx = state->pred_out_buf.size();
        const size_t mem_pred_out_buf = ggml_backend_buffer_get_size(state->pred_out_buffer);
        PARAKEET_LOG_INFO("%s: pred state: %7.2f MB (meta) + %7.2f MB (data)\n", __func__,
                mem_pred_ctx / 1024.0 / 1024.0, mem_pred_out_buf / 1024.0 / 1024.0);
    }

    PARAKEET_LOG_INFO("%s: compute buffer (encode) = %7.2f MB\n", __func__, parakeet_sched_size(state->sched_encode) / 1e6);

    {
        bool ok = parakeet_sched_graph_init(state->sched_decode, state->backends,
                [&]() {
                    const auto & hparams = ctx->model.hparams;
                    const int n_tokens = hparams.n_audio_ctx; // Use audio ctx for Parakeet

                    parakeet_batch_prep_legacy(state->batch, nullptr, n_tokens, 0, 0);

                    return parakeet_build_graph_prediction(*ctx, *state, state->batch, true);
                });

        if (!ok) {
            PARAKEET_LOG_ERROR("%s: failed to init decoder allocator\n", __func__);
            parakeet_free_state(state);
            return nullptr;
        }

        PARAKEET_LOG_INFO("%s: compute buffer (decode) = %7.2f MB\n", __func__, parakeet_sched_size(state->sched_decode) / 1e6);
    }

    return state;
}

struct parakeet_context_params parakeet_context_default_params() {
    struct parakeet_context_params result = {
        /*.use_gpu              =*/ true,
        /*.flash_attn           =*/ true,
        /*.gpu_device           =*/ 0,
    };
    return result;
}

struct parakeet_context * parakeet_init_from_file_with_params_no_state(const char * path_model, struct parakeet_context_params params) {
    PARAKEET_LOG_INFO("%s: loading model from '%s'\n", __func__, path_model);
#ifdef _MSC_VER
    // Convert UTF-8 path to wide string (UTF-16) for Windows, resolving character encoding issues.
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring path_model_wide = converter.from_bytes(path_model);
    auto fin = std::ifstream(path_model_wide, std::ios::binary);
#else
    auto fin = std::ifstream(path_model, std::ios::binary);
#endif
    if (!fin) {
        PARAKEET_LOG_ERROR("%s: failed to open '%s'\n", __func__, path_model);
        return nullptr;
    }

    parakeet_model_loader loader = {};

    loader.context = &fin;

    loader.read = [](void * ctx, void * output, size_t read_size) {
        std::ifstream * fin = (std::ifstream*)ctx;
        fin->read((char *)output, read_size);
        return read_size;
    };

    loader.eof = [](void * ctx) {
        std::ifstream * fin = (std::ifstream*)ctx;
        return fin->eof();
    };

    loader.close = [](void * ctx) {
        std::ifstream * fin = (std::ifstream*)ctx;
        fin->close();
    };

    auto ctx = parakeet_init_with_params_no_state(&loader, params);

    if (ctx) {
        ctx->path_model = path_model;
    }

    return ctx;
}

struct parakeet_context * parakeet_init_from_buffer_with_params_no_state(void * buffer, size_t buffer_size, struct parakeet_context_params params) {
    struct buf_context {
        uint8_t* buffer;
        size_t size;
        size_t current_offset;
    };

    buf_context ctx = { reinterpret_cast<uint8_t*>(buffer), buffer_size, 0 };

    PARAKEET_LOG_INFO("%s: loading model from buffer\n", __func__);

    parakeet_model_loader loader = {};

    loader.context = &ctx;

    loader.read = [](void * ctx, void * output, size_t read_size) {
        buf_context * buf = reinterpret_cast<buf_context *>(ctx);

        size_t size_to_copy = buf->current_offset + read_size < buf->size ? read_size : buf->size - buf->current_offset;

        memcpy(output, buf->buffer + buf->current_offset, size_to_copy);
        buf->current_offset += size_to_copy;

        return size_to_copy;
    };

    loader.eof = [](void * ctx) {
        buf_context * buf = reinterpret_cast<buf_context *>(ctx);

        return buf->current_offset >= buf->size;
    };

    loader.close = [](void * /*ctx*/) { };

    return parakeet_init_with_params_no_state(&loader, params);
}

struct parakeet_context * parakeet_init_with_params_no_state(struct parakeet_model_loader * loader, struct parakeet_context_params params) {
    ggml_time_init();

    PARAKEET_LOG_INFO("%s: use gpu    = %d\n", __func__, params.use_gpu);
    PARAKEET_LOG_INFO("%s: flash attn = %d\n", __func__, params.flash_attn);
    PARAKEET_LOG_INFO("%s: gpu_device = %d\n", __func__, params.gpu_device);
    PARAKEET_LOG_INFO("%s: devices    = %zu\n", __func__, ggml_backend_dev_count());
    PARAKEET_LOG_INFO("%s: backends   = %zu\n", __func__, ggml_backend_reg_count());

    parakeet_context * ctx = new parakeet_context;
    ctx->params = params;

    if (!parakeet_model_load(loader, *ctx)) {
        loader->close(loader->context);
        PARAKEET_LOG_ERROR("%s: failed to load model\n", __func__);
        delete ctx;
        return nullptr;
    }

    loader->close(loader->context);

    // Initialize mel cache with model's FFT size
    ctx->mel_cache.init(ctx->model.hparams.n_fft);
    PARAKEET_LOG_INFO("%s: initialized mel cache with n_fft = %d\n", __func__, ctx->model.hparams.n_fft);

    return ctx;
}

struct parakeet_context * parakeet_init_from_file_with_params(const char * path_model, struct parakeet_context_params params) {
    parakeet_context * ctx = parakeet_init_from_file_with_params_no_state(path_model, params);
    if (!ctx) {
        return nullptr;
    }

    ctx->state = parakeet_init_state(ctx);
    if (!ctx->state) {
        parakeet_free(ctx);
        return nullptr;
    }

    return ctx;
}

struct parakeet_context * parakeet_init_from_buffer_with_params(void * buffer, size_t buffer_size, struct parakeet_context_params params) {
    parakeet_context * ctx = parakeet_init_from_buffer_with_params_no_state(buffer, buffer_size, params);
    if (!ctx) {
        return nullptr;
    }

    ctx->state = parakeet_init_state(ctx);
    if (!ctx->state) {
        parakeet_free(ctx);
        return nullptr;
    }

    return ctx;
}

struct parakeet_context * parakeet_init_with_params(struct parakeet_model_loader * loader, struct parakeet_context_params params) {
    parakeet_context * ctx = parakeet_init_with_params_no_state(loader, params);
    if (!ctx) {
        return nullptr;
    }

    ctx->state = parakeet_init_state(ctx);
    if (!ctx->state) {
        parakeet_free(ctx);
        return nullptr;
    }

    return ctx;
}

void parakeet_free_state(struct parakeet_state * state) {
    if (state) {
        ggml_backend_buffer_free(state->lstm_state.buffer);
        ggml_backend_buffer_free(state->pred_out_buffer);

        parakeet_batch_free(state->batch);

        ggml_backend_sched_free(state->sched_conv.sched);
        ggml_backend_sched_free(state->sched_encode.sched);
        ggml_backend_sched_free(state->sched_decode.sched);

        for (auto & backend : state->backends) {
            ggml_backend_free(backend);
        }

        delete state;
    }
}

void parakeet_free(struct parakeet_context * ctx) {
    if (ctx) {
        for (ggml_context * context : ctx->model.ctxs) {
            ggml_free(context);
        }

        for (ggml_backend_buffer_t buf : ctx->model.buffers) {
            ggml_backend_buffer_free(buf);
        }

        parakeet_free_state(ctx->state);

        delete ctx;
    }
}

void parakeet_free_context_params(struct parakeet_context_params * params) {
    if (params) {
        delete params;
    }
}

void parakeet_free_params(struct parakeet_full_params * params) {
    if (params) {
        delete params;
    }
}

int parakeet_pcm_to_mel_with_state(struct parakeet_context * ctx, struct parakeet_state * state, const float * samples, int n_samples, int n_threads) {
    if (!log_mel_spectrogram(*state,
                samples,
                n_samples,
                PARAKEET_SAMPLE_RATE,
                ctx->model.hparams.n_fft,
                PARAKEET_HOP_LENGTH,
                ctx->model.filters.n_mel,
                n_threads,
                ctx->model.filters,
                false,                        // debug
                state->mel,
                ctx->mel_cache)) {
        PARAKEET_LOG_ERROR("%s: failed to compute mel spectrogram\n", __func__);
        return -1;
    }

    return 0;
}

int parakeet_pcm_to_mel(struct parakeet_context * ctx, const float * samples, int n_samples, int n_threads) {
    return parakeet_pcm_to_mel_with_state(ctx, ctx->state, samples, n_samples, n_threads);
}

int parakeet_set_mel_with_state(
        struct parakeet_context * ctx,
          struct parakeet_state * state,
                   const float * data,
                           int   n_len,
                           int   n_mel) {
    if (n_mel != ctx->model.filters.n_mel) {
        PARAKEET_LOG_ERROR("%s: invalid number of mel bands: %d (expected %d)\n", __func__, n_mel, ctx->model.filters.n_mel);
        return -1;
    }

    state->mel.n_len     = n_len;
    state->mel.n_len_org = n_len;
    state->mel.n_mel     = n_mel;

    state->mel.data.resize(n_len*n_mel);
    memcpy(state->mel.data.data(), data, n_len*n_mel*sizeof(float));

    return 0;
}

int parakeet_set_mel(
        struct parakeet_context * ctx,
        const float * data,
        int n_len,
        int n_mel) {
    return parakeet_set_mel_with_state(ctx, ctx->state, data, n_len, n_mel);
}

int parakeet_encode_with_state(struct parakeet_context * ctx, struct parakeet_state * state, int offset, int n_threads) {
    if (!parakeet_encode_internal(*ctx, *state, offset, n_threads, nullptr, nullptr)) {
        PARAKEET_LOG_ERROR("%s: failed to eval\n", __func__);
        return -1;
    }

    return 0;
}

int parakeet_encode(struct parakeet_context * ctx, int offset, int n_threads) {
    if (!parakeet_encode_internal(*ctx, *ctx->state, offset, n_threads, nullptr, nullptr)) {
        PARAKEET_LOG_ERROR("%s: failed to eval\n", __func__);
        return -1;
    }

    return 0;
}

int parakeet_tokenize(struct parakeet_context * ctx, const char * text, parakeet_token * tokens, int n_max_tokens) {
    const auto res = tokenize(ctx->vocab, text);

    if (n_max_tokens < (int) res.size()) {
        PARAKEET_LOG_ERROR("%s: too many resulting tokens: %d (max %d)\n", __func__, (int) res.size(), n_max_tokens);
        return -(int) res.size();
    }

    for (int i = 0; i < (int) res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

int parakeet_token_count(struct parakeet_context * ctx, const char * text) {
    return -parakeet_tokenize(ctx, text, NULL, 0);
}

int parakeet_model_n_vocab(struct parakeet_context * ctx) {
    return ctx->model.hparams.n_vocab;
}

int parakeet_model_n_audio_ctx(struct parakeet_context * ctx) {
    return ctx->model.hparams.n_audio_ctx;
}

int parakeet_model_n_audio_state(struct parakeet_context * ctx) {
    return ctx->model.hparams.n_audio_state;
}

int parakeet_model_n_audio_head(struct parakeet_context * ctx) {
    return ctx->model.hparams.n_audio_head;
}

int parakeet_model_n_audio_layer(struct parakeet_context * ctx) {
    return ctx->model.hparams.n_audio_layer;
}

int parakeet_model_n_mels(struct parakeet_context * ctx) {
    return ctx->model.hparams.n_mels;
}

int parakeet_model_ftype(struct parakeet_context * ctx) {
    return ctx->model.hparams.ftype;
}

int parakeet_n_len_from_state(struct parakeet_state * state) {
    return state->mel.n_len_org;
}

int parakeet_n_len(struct parakeet_context * ctx) {
    return ctx->state->mel.n_len_org;
}

int parakeet_n_vocab(struct parakeet_context * ctx) {
    return ctx->vocab.n_vocab;
}

int parakeet_n_audio_ctx(struct parakeet_context * ctx) {
    return ctx->model.hparams.n_audio_ctx;
}

float * parakeet_get_logits(struct parakeet_context * ctx) {
    return ctx->state->logits.data();
}

float * parakeet_get_logits_from_state(struct parakeet_state * state) {
    return state->logits.data();
}

const char * parakeet_token_to_str(struct parakeet_context * ctx, parakeet_token token) {
    return ctx->vocab.id_to_token.at(token).c_str();
}

int parakeet_token_to_text(const char * token_str, bool is_first, char * output, int max_len) {
    std::string text = sentencepiece_piece_to_text(token_str, is_first);

    if (output == nullptr) {
        return text.size();
    }

    int bytes_to_copy = std::min((int)text.size(), max_len - 1);
    if (bytes_to_copy > 0) {
        memcpy(output, text.c_str(), bytes_to_copy);
        output[bytes_to_copy] = '\0';
    } else if (max_len > 0) {
        output[0] = '\0';
    }

    return text.size();
}

parakeet_token parakeet_token_bos(struct parakeet_context * ctx) {
    return ctx->vocab.token_bos;
}

parakeet_token parakeet_token_unk(struct parakeet_context * ctx) {
    return ctx->vocab.token_unk;
}

parakeet_token parakeet_token_blank(struct parakeet_context * ctx) {
    return ctx->vocab.token_blank;
}

struct parakeet_timings * parakeet_get_timings(struct parakeet_context * ctx) {
    if (ctx->state == nullptr) {
        return nullptr;
    }
    parakeet_timings * timings = new parakeet_timings;
    timings->sample_ms = 1e-3f * ctx->state->t_sample_us / std::max(1, ctx->state->n_sample);
    timings->encode_ms = 1e-3f * ctx->state->t_encode_us / std::max(1, ctx->state->n_encode);
    timings->decode_ms = 1e-3f * ctx->state->t_decode_us / std::max(1, ctx->state->n_decode);
    return timings;
}

void parakeet_print_timings(struct parakeet_context * ctx) {
    const int64_t t_end_us = ggml_time_us();

    PARAKEET_LOG_INFO("\n");
    PARAKEET_LOG_INFO("%s:     load time = %8.2f ms\n", __func__, ctx->t_load_us / 1000.0f);
    if (ctx->state != nullptr) {

        const int32_t n_sample = std::max(1, ctx->state->n_sample);
        const int32_t n_encode = std::max(1, ctx->state->n_encode);
        const int32_t n_decode = std::max(1, ctx->state->n_decode);

        PARAKEET_LOG_INFO("%s:     fallbacks = %3d p / %3d h\n", __func__, ctx->state->n_fail_p, ctx->state->n_fail_h);
        PARAKEET_LOG_INFO("%s:      mel time = %8.2f ms\n", __func__, ctx->state->t_mel_us / 1000.0f);
        PARAKEET_LOG_INFO("%s:   sample time = %8.2f ms / %5d runs ( %8.2f ms per run)\n", __func__, 1e-3f * ctx->state->t_sample_us, n_sample, 1e-3f * ctx->state->t_sample_us / n_sample);
        PARAKEET_LOG_INFO("%s:   encode time = %8.2f ms / %5d runs ( %8.2f ms per run)\n", __func__, 1e-3f * ctx->state->t_encode_us, n_encode, 1e-3f * ctx->state->t_encode_us / n_encode);
        PARAKEET_LOG_INFO("%s:   decode time = %8.2f ms / %5d runs ( %8.2f ms per run)\n", __func__, 1e-3f * ctx->state->t_decode_us, n_decode, 1e-3f * ctx->state->t_decode_us / n_decode);

    }
    PARAKEET_LOG_INFO("%s:    total time = %8.2f ms\n", __func__, (t_end_us - ctx->t_start_us)/1000.0f);
}

void parakeet_reset_timings(struct parakeet_context * ctx) {
    ctx->t_start_us = ggml_time_us();
    if (ctx->state != nullptr) {
        ctx->state->t_mel_us = 0;
        ctx->state->t_sample_us = 0;
        ctx->state->t_encode_us = 0;
        ctx->state->t_decode_us = 0;

        ctx->state->n_sample = 0;
        ctx->state->n_encode = 0;
        ctx->state->n_decode = 0;
    }
}

const char * parakeet_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "PARAKEET : ";

    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }
    return s.c_str();
}

struct parakeet_context_params * parakeet_context_default_params_by_ref(void) {
    struct parakeet_context_params params = parakeet_context_default_params();

    struct parakeet_context_params* result = new parakeet_context_params();
    *result = params;
    return result;
}

struct parakeet_full_params * parakeet_full_default_params_by_ref(enum parakeet_sampling_strategy strategy) {
    struct parakeet_full_params params = parakeet_full_default_params(strategy);

    struct parakeet_full_params* result = new parakeet_full_params();
    *result = params;
    return result;
}

struct parakeet_full_params parakeet_full_default_params(enum parakeet_sampling_strategy strategy) {
    struct parakeet_full_params result = {
        /*.strategy                         =*/ strategy,
        /*.n_threads                        =*/ std::min(4, (int32_t) std::thread::hardware_concurrency()),
        /*.offset_ms                        =*/ 0,
        /*.duration_ms                      =*/ 0,
        /*.no_context                       =*/ true,
        /*.audio_ctx                        =*/ 0,
        /*.chunk_length_ms                  =*/ 10000,  // 10 second chunks
        /*.left_context_ms                  =*/ 10000,  // 10 second left context
        /*.right_context_ms                 =*/ 4960,   // 4.96 second right context
        /*.new_token_callback               =*/ nullptr,
        /*.new_token_callback_user_data     =*/ nullptr,
        /*.new_segment_callback             =*/ nullptr,
        /*.new_segment_callback_user_data   =*/ nullptr,
        /*.progress_callback                =*/ nullptr,
        /*.progress_callback_user_data      =*/ nullptr,
        /*.encoder_begin_callback           =*/ nullptr,
        /*.encoder_begin_callback_user_data =*/ nullptr,
        /*.abort_callback                   =*/ nullptr,
        /*.abort_callback_user_data         =*/ nullptr,
    };

    return result;
}

static void parakeet_stream_reset_state(struct parakeet_state * state) {
    if (state == nullptr) {
        return;
    }

    if (state->lstm_state.buffer) {
        ggml_backend_buffer_clear(state->lstm_state.buffer, 0);
    }

    state->decoded_tokens.clear();
    state->decoded_token_data.clear();
    state->result_all.clear();

    state->tdt_stream_state.initialized    = false;
    state->tdt_stream_state.last_token     = 0;
    state->tdt_stream_state.time_step      = 0;
    state->tdt_stream_state.decoded_length = 0;

    state->stream.buffer.clear();
    state->stream.n_samples_advanced = 0;
    state->stream.n_left_ctx         = 0;
    state->stream.n_chunk            = 0;
    state->stream.n_right_ctx        = 0;
    state->stream.params             = {};
    state->stream.initialized        = false;

    state->enc_out_buffer.clear();
    state->enc_out_frames = 0;
    state->n_frames       = 0;
    state->n_audio_ctx    = 0;
}

static int parakeet_stream_process_window(
        struct parakeet_context * ctx,
          struct parakeet_state * state,
                    const float * samples,
                            int   n_samples,
                            int   n_chunk) {
    const parakeet_stream & stream = state->stream;
    const parakeet_full_params & params = stream.params;
    const int d_enc = ctx->model.hparams.n_audio_state;

    // process all the samples.
    if (parakeet_pcm_to_mel_with_state(ctx, state, samples, n_samples, params.n_threads) != 0) {
        return -2;
    }

    const int left_mel_frames = stream.n_left_ctx / PARAKEET_HOP_LENGTH;
    const int chunk_mel_frames = n_chunk / PARAKEET_HOP_LENGTH;

    state->n_audio_ctx = state->mel.n_len;
    // process entire log mel spectrogram.
    if (!parakeet_encode_internal(*ctx, *state, 0, params.n_threads,
                                  params.abort_callback, params.abort_callback_user_data)) {
        return -6;
    }

    const int left_enc_frames  = left_mel_frames  / ctx->model.hparams.subsampling_factor;
    const int chunk_enc_frames = chunk_mel_frames / ctx->model.hparams.subsampling_factor;

    if (chunk_enc_frames <= 0) {
        return 0;
    }

    // Copy the center chunk so that it is the only part that the joint network sees.
    state->enc_out_buffer.resize(chunk_enc_frames * d_enc);
    ggml_backend_tensor_get(state->enc_out, state->enc_out_buffer.data(),
                           left_enc_frames * d_enc * sizeof(float),
                           chunk_enc_frames * d_enc * sizeof(float));

    state->enc_out_frames = chunk_enc_frames;
    state->n_frames = chunk_enc_frames;

    const size_t tokens_before = state->decoded_tokens.size();

    // Run the prediction and joint network on the center chunk.
    if (!parakeet_decode_chunk(*ctx, *state, state->batch, chunk_enc_frames, params.n_threads, &params)) {
        return -7;
    }

    const size_t tokens_after = state->decoded_tokens.size();
    const size_t new_token_count = tokens_after - tokens_before;

    if (new_token_count > 0) {
        std::string text;
        std::vector<parakeet_token_data> result_tokens;
        const int64_t chunk_t0 = 100LL * stream.n_samples_advanced / PARAKEET_SAMPLE_RATE;
        const int64_t chunk_t1 = 100LL * (stream.n_samples_advanced + n_chunk) / PARAKEET_SAMPLE_RATE;
        const int frame_offset = chunk_t0 / ctx->model.hparams.subsampling_factor;

        result_tokens.reserve(new_token_count);

        for (size_t i = tokens_before; i < tokens_after; ++i) {
            const auto token_id = state->decoded_tokens[i];
            const char * token_str = parakeet_token_to_str(ctx, token_id);
            if (token_str) {
                const bool is_first_piece = (tokens_before == 0) && text.empty();
                text += sentencepiece_piece_to_text(token_str, is_first_piece);
            }

            auto token_data = state->decoded_token_data[i];
            token_data.frame_index += frame_offset;
            token_data.t0 += chunk_t0;
            token_data.t1 += chunk_t0;
            result_tokens.push_back(token_data);
        }

        refine_timestamps_tdt(ctx->vocab, result_tokens);

        if (!text.empty()) {
            parakeet_segment segment;
            segment.t0 = chunk_t0;
            segment.t1 = chunk_t1;
            segment.text = std::move(text);
            segment.tokens = std::move(result_tokens);

            state->result_all.push_back(std::move(segment));

            if (params.new_segment_callback) {
                params.new_segment_callback(ctx, state, 1, params.new_segment_callback_user_data);
            }
        }
    }

    return 0;
}

static int ms_to_n_samples(int ms) {
    return ms * PARAKEET_SAMPLE_RATE / 1000;
}

int parakeet_stream_init(
        struct parakeet_context * ctx,
          struct parakeet_state * state,
    struct parakeet_full_params   params) {
    if (ctx == nullptr || state == nullptr) {
        return -1;
    }

    const int n_left_ctx  = ms_to_n_samples(params.left_context_ms);
    const int n_chunk     = ms_to_n_samples(params.chunk_length_ms);
    const int n_right_ctx = ms_to_n_samples(params.right_context_ms);

    if (n_left_ctx < 0 || n_chunk <= 0 || n_right_ctx < 0) {
        return -1;
    }

    parakeet_stream_reset_state(state);

    state->stream.n_left_ctx  = n_left_ctx;
    state->stream.n_chunk     = n_chunk;
    state->stream.n_right_ctx = n_right_ctx;
    state->stream.params      = params;
    state->stream.initialized = true;

    if (n_left_ctx > 0) {
        state->stream.buffer.assign(n_left_ctx, 0.0f);
    }

    return 0;
}

int parakeet_stream_push(
        struct parakeet_context * ctx,
          struct parakeet_state * state,
                    const float * samples,
                            int   n_samples) {
    if (ctx == nullptr || state == nullptr || samples == nullptr || n_samples <= 0) {
        return -1;
    }

    if (!state->stream.initialized) {
        return -1;
    }

    const int n_total_samples = state->stream.n_left_ctx + state->stream.n_chunk + state->stream.n_right_ctx;

    // Insert the new chunk of samples as the new center and right context.
    state->stream.buffer.insert(state->stream.buffer.end(), samples, samples + n_samples);

    // As long as we have enough samples to form a complete window we process it.
    while (state->stream.buffer.size() >= (size_t) n_total_samples) {
        const int ret = parakeet_stream_process_window(
                ctx,
                state,
                state->stream.buffer.data(),
                n_total_samples,
                state->stream.n_chunk);
        if (ret != 0) {
            return ret;
        }

        // TODO: std::vector::erase is O(n) and not optimal. We should probably
        // use a ring buffer instead.
        // Shift the center and right context to the start of the buffer. This
        // allows the next call to have the current center chunk as its left
        // context, and the right context will become part of the next target
        // chunk together with the new samples which will make up the rest of
        // the target chunk and the new right context.
        state->stream.buffer.erase(state->stream.buffer.begin(), state->stream.buffer.begin() + state->stream.n_chunk);

        state->stream.n_samples_advanced += state->stream.n_chunk;
    }

    return 0;
}

int parakeet_stream_flush(
        struct parakeet_context * ctx,
          struct parakeet_state * state) {
    if (ctx == nullptr || state == nullptr) {
        return -1;
    }

    if (!state->stream.initialized) {
        return -1;
    }

    while (state->stream.buffer.size() > (size_t) state->stream.n_left_ctx) {
        const int n_remaining_samples = (int) state->stream.buffer.size() - state->stream.n_left_ctx;
        const int n_flush_chunk       = std::min(state->stream.n_chunk, n_remaining_samples);
        const int n_right_available   = std::min(state->stream.n_right_ctx, n_remaining_samples - n_flush_chunk);
        const int n_copied            = state->stream.n_left_ctx + n_flush_chunk + n_right_available;

        std::vector<float> flush_window(state->stream.n_left_ctx + n_flush_chunk + state->stream.n_right_ctx, 0.0f);

        std::copy_n(state->stream.buffer.begin(), n_copied, flush_window.begin());

        const int ret = parakeet_stream_process_window(
                ctx,
                state,
                flush_window.data(),
                (int) flush_window.size(),
                n_flush_chunk);
        if (ret != 0) {
            return ret;
        }

        state->stream.buffer.erase(state->stream.buffer.begin(), state->stream.buffer.begin() + n_flush_chunk);
        state->stream.n_samples_advanced += n_flush_chunk;
    }

    state->stream.buffer.clear();
    state->stream.n_samples_advanced = 0;
    state->stream.n_left_ctx         = 0;
    state->stream.n_chunk            = 0;
    state->stream.n_right_ctx        = 0;
    state->stream.params             = {};
    state->stream.initialized        = false;

    return 0;
}

int parakeet_full_with_state(
        struct parakeet_context * ctx,
          struct parakeet_state * state,
    struct parakeet_full_params   params,
                    const float * samples,
                           int    n_samples) {
    // clear old results
    auto & result_all = state->result_all;
    result_all.clear();

    // Clear any previous decoded tokens if no_context is set
    if (params.no_context) {
        state->decoded_tokens.clear();
        state->decoded_token_data.clear();
    }

    // If no chunking specified, delegate to parakeet_chunk (processes entire audio as one chunk)
    if (params.chunk_length_ms == 0) {
        return parakeet_chunk(ctx, state, params, samples, n_samples);
    }

    // Nemo sliding window chunking (chunk raw samples)
    {
        PARAKEET_LOG_INFO("%s: chunking: chunk_ms=%d, left_ctx=%d, right_ctx=%d\n",
                         __func__, params.chunk_length_ms, params.left_context_ms, params.right_context_ms);

        // reset tdt streaming state
        state->tdt_stream_state.initialized    = false;
        state->tdt_stream_state.last_token     = 0;
        state->tdt_stream_state.time_step      = 0;
        state->tdt_stream_state.decoded_length = 0;

        const int left_context_samples  = (params.left_context_ms  * PARAKEET_SAMPLE_RATE) / 1000;
        const int chunk_samples         = (params.chunk_length_ms  * PARAKEET_SAMPLE_RATE) / 1000;
        const int right_context_samples = (params.right_context_ms * PARAKEET_SAMPLE_RATE) / 1000;

        const int d_enc = ctx->model.hparams.n_audio_state;

        int left_sample = 0;

        // Process audio in sliding chunks of raw samples.
        while (left_sample < n_samples) {
            const int buffer_start_sample   = std::max(0, left_sample - left_context_samples);
            const int right_sample          = std::min(left_sample + chunk_samples, n_samples);
            const int buffer_end_sample     = std::min(right_sample + right_context_samples, n_samples);
            const int buffer_length_samples = buffer_end_sample - buffer_start_sample;
            const size_t tokens_before      = state->decoded_tokens.size();

            PARAKEET_LOG_DEBUG("%s: sample chunk: buffer=[%d-%d] samples, chunk=[%d-%d]\n",
                             __func__, buffer_start_sample, buffer_end_sample, left_sample, right_sample);

            // Compute mel spectrogram for the left context, the center chunk,
            // and the right context.
            if (parakeet_pcm_to_mel_with_state(ctx, state, samples + buffer_start_sample,
                                               buffer_length_samples, params.n_threads) != 0) {
                PARAKEET_LOG_ERROR("%s: failed to compute log mel spectrogram for chunk\n", __func__);
                return -2;
            }

            const int total_mel_frames = state->mel.n_len;

            // Calculate mel frame offsets within this chunks mel spectrogram
            // mel frames = raw samples / 160 (for 16kHz, 10ms hop)
            const int left_context_mel_frames = (left_sample - buffer_start_sample) / 160;
            const int chunk_mel_frames        = (right_sample - left_sample)        / 160;

            PARAKEET_LOG_DEBUG("%s: Mel features: total=%d frames, left_ctx=%d, chunk=%d\n",
                             __func__, total_mel_frames, left_context_mel_frames, chunk_mel_frames);

            // The encoder will see the left context, the main chunk, and the right
            // context so that it has access to nearby frames during processing.
            // Without this there would be hard cutoffs and the encoder might not
            // be able to detect speech near the edges.
            state->n_audio_ctx = total_mel_frames;
            if (!parakeet_encode_internal(*ctx, *state, 0, params.n_threads,
                                         params.abort_callback, params.abort_callback_user_data)) {
                PARAKEET_LOG_ERROR("%s: failed to encode chunk\n", __func__);
                return -6;
            }

            // Calculate which encoder frames correspond to the actual chunk (center chunk)
            const int left_context_enc_frames = left_context_mel_frames / ctx->model.hparams.subsampling_factor;
            const int chunk_enc_frames = chunk_mel_frames / ctx->model.hparams.subsampling_factor;

            PARAKEET_LOG_DEBUG("%s: encoder output: total=%d frames, left_ctx=%d, chunk=%d\n",
                             __func__, state->n_frames, left_context_enc_frames, chunk_enc_frames);

            // The joint network only sees the center chunk encoder frames and
            // not the left and right context that the encoder did.
            state->enc_out_buffer.resize(chunk_enc_frames * d_enc);
            ggml_backend_tensor_get(state->enc_out, state->enc_out_buffer.data(),
                                   left_context_enc_frames * d_enc * sizeof(float),
                                   chunk_enc_frames * d_enc * sizeof(float));
            state->enc_out_frames = chunk_enc_frames;

            state->n_frames = chunk_enc_frames;

            PARAKEET_LOG_DEBUG("%s: decoding %d encoder frames for this chunk\n", __func__, chunk_enc_frames);

            if (!parakeet_decode_chunk(*ctx, *state, state->batch, chunk_enc_frames, params.n_threads, &params)) {
                PARAKEET_LOG_ERROR("%s: failed to decode chunk\n", __func__);
                return -7;
            }

            const size_t tokens_after = state->decoded_tokens.size();
            const size_t new_token_count = tokens_after - tokens_before;

            if (new_token_count > 0) {
                std::string text;
                std::vector<parakeet_token_data> result_tokens;
                const int64_t chunk_t0 = 100LL * left_sample / PARAKEET_SAMPLE_RATE;
                const int64_t chunk_t1 = 100LL * right_sample / PARAKEET_SAMPLE_RATE;
                const int frame_offset = chunk_t0 / ctx->model.hparams.subsampling_factor;

                result_tokens.reserve(new_token_count);

                for (size_t i = tokens_before; i < tokens_after; ++i) {
                    const auto token_id = state->decoded_tokens[i];
                    const char * token_str = parakeet_token_to_str(ctx, token_id);
                    if (token_str) {
                        const bool is_first_piece = (tokens_before == 0) && text.empty();
                        text += sentencepiece_piece_to_text(token_str, is_first_piece);
                    }

                    auto token_data = state->decoded_token_data[i];
                    token_data.frame_index += frame_offset;
                    token_data.t0 += chunk_t0;
                    token_data.t1 += chunk_t0;
                    result_tokens.push_back(token_data);
                }

                refine_timestamps_tdt(ctx->vocab, result_tokens);

                if (!text.empty()) {
                    parakeet_segment segment;
                    segment.t0 = chunk_t0;
                    segment.t1 = chunk_t1;
                    segment.text = std::move(text);
                    segment.tokens = std::move(result_tokens);

                    state->result_all.push_back(std::move(segment));

                    if (params.new_segment_callback) {
                        params.new_segment_callback(ctx, state, 1, params.new_segment_callback_user_data);
                    }
                }
            }

            // shift the window.
            left_sample = right_sample;
        }

        // Clear buffers after all chunks processed
        state->enc_out_buffer.clear();
        state->enc_out_frames = 0;
    }

    return 0;
}

int parakeet_full(
        struct parakeet_context * ctx,
    struct parakeet_full_params   params,
                    const float * samples,
                            int   n_samples) {
    return parakeet_full_with_state(ctx, ctx->state, params, samples, n_samples);
}

int parakeet_chunk(
        struct parakeet_context * ctx,
          struct parakeet_state * state,
    struct parakeet_full_params   params,
                    const float * samples,
                            int   n_samples) {

    if (params.no_context) {
        ggml_backend_buffer_clear(state->lstm_state.buffer, 0);
        state->decoded_tokens.clear();
        state->decoded_token_data.clear();

        state->tdt_stream_state.initialized    = false;
        state->tdt_stream_state.last_token     = 0;
        state->tdt_stream_state.time_step      = 0;
        state->tdt_stream_state.decoded_length = 0;
    }

    if (n_samples > 0) {
        if (parakeet_pcm_to_mel_with_state(ctx, state, samples, n_samples, params.n_threads) != 0) {
            PARAKEET_LOG_ERROR("%s: failed to compute log mel spectrogram\n", __func__);
            return -2;
        }
    }

    if (params.audio_ctx == 0) {
        const int total_len = parakeet_n_len_from_state(state);
        const int model_max_ctx = parakeet_n_audio_ctx(ctx);
        params.audio_ctx = std::min(total_len, model_max_ctx);
        PARAKEET_LOG_DEBUG("Processing audio: total_frames=%d, chunk_size=%d\n", total_len, params.audio_ctx);
    }
    state->n_audio_ctx = params.audio_ctx;

    const int n_frames = parakeet_n_len_from_state(state);

    if (!parakeet_encode_internal(*ctx, *state, 0, params.n_threads, params.abort_callback, params.abort_callback_user_data)) {
        PARAKEET_LOG_ERROR("%s: failed to encode\n", __func__);
        return -6;
    }

    const size_t tokens_before = state->decoded_tokens.size();

    if (!parakeet_decode(*ctx, *state, state->batch, params.n_threads, &params)) {
        PARAKEET_LOG_ERROR("%s: failed to decode\n", __func__);
        return -7;
    }

    const size_t tokens_after = state->decoded_tokens.size();
    const size_t new_token_count = tokens_after - tokens_before;

    if (new_token_count > 0) {
        std::string text;
        std::vector<parakeet_token_data> result_tokens;

        for (size_t i = tokens_before; i < tokens_after; i++) {
            const auto token_id = state->decoded_tokens[i];
            const char * token_str = parakeet_token_to_str(ctx, token_id);
            if (token_str) {
                const bool is_first_piece = (tokens_before == 0) && text.empty();
                text += sentencepiece_piece_to_text(token_str, is_first_piece);
            }

            // Use the stored token data from parakeet_decode
            result_tokens.push_back(state->decoded_token_data[i]);
        }

        refine_timestamps_tdt(ctx->vocab, result_tokens);

        if (!text.empty()) {
            parakeet_segment segment;
            segment.t0 = 0; // Caller tracks timing
            segment.t1 = n_frames;
            segment.text = text;
            segment.tokens = result_tokens;

            state->result_all.push_back(std::move(segment));

            if (params.new_segment_callback) {
                params.new_segment_callback(ctx, state, 1, params.new_segment_callback_user_data);
            }
        }
    }

    return 0;
}

int parakeet_full_parallel(
            struct parakeet_context * ctx,
        struct parakeet_full_params   params,
                        const float * samples,
                                int   n_samples,
                                int   n_processors) {
    if (n_processors == 1) {
        return parakeet_full(ctx, params, samples, n_samples);
    }

    int ret = 0;

    // prepare separate states for each thread
    std::vector<parakeet_state*> states;

    const int offset_samples = (PARAKEET_SAMPLE_RATE*params.offset_ms)/1000;
    const int n_samples_per_processor = (n_samples - offset_samples)/n_processors;

    // the calling thread will process the first chunk
    // while the other threads will process the remaining chunks

    std::vector<std::thread> workers(n_processors - 1);
    for (int i = 0; i < n_processors - 1; ++i) {
        // create a new state for each thread
        states.push_back(parakeet_init_state(ctx));

        const int start_samples = offset_samples + (i + 1)*n_samples_per_processor;
        const int n_samples_cur = (i == n_processors - 2) ? n_samples - start_samples : n_samples_per_processor;

        auto params_cur = params;

        params_cur.offset_ms = 0;

        params_cur.new_segment_callback = nullptr;
        params_cur.new_segment_callback_user_data = nullptr;

        params_cur.progress_callback = nullptr;
        params_cur.progress_callback_user_data = nullptr;

        workers[i] = std::thread(parakeet_full_with_state, ctx, states[i], std::move(params_cur), samples + start_samples, n_samples_cur);
    }

    {
        auto params_cur = params;

        // Run the first transformation using default state but only for the first chunk.
        ret = parakeet_full_with_state(ctx, ctx->state, std::move(params_cur), samples, offset_samples + n_samples_per_processor);
    }

    for (int i = 0; i < n_processors - 1; ++i) {
        workers[i].join();
    }

    const int64_t offset_t = (int64_t) params.offset_ms/10.0;

    // combine results into result_state->result_all from all other states
    for (int i = 0; i < n_processors - 1; ++i) {
        auto& results_i = states[i]->result_all;

        for (auto& result : results_i) {
            // correct the segment timestamp taking into account the offset
            result.t0 += 100 * ((i + 1) * n_samples_per_processor) / PARAKEET_SAMPLE_RATE + offset_t;
            result.t1 += 100 * ((i + 1) * n_samples_per_processor) / PARAKEET_SAMPLE_RATE + offset_t;

            // make sure that segments are not overlapping
            if (!ctx->state->result_all.empty()) {
                result.t0 = std::max(result.t0, ctx->state->result_all.back().t1);
            }

            ctx->state->result_all.push_back(std::move(result));

            // call the new_segment_callback for each segment
            if (params.new_segment_callback) {
                params.new_segment_callback(ctx, ctx->state, 1, params.new_segment_callback_user_data);
            }
        }

        ctx->state->t_mel_us += states[i]->t_mel_us;
        ctx->state->t_sample_us += states[i]->t_sample_us;
        ctx->state->t_encode_us += states[i]->t_encode_us;
        ctx->state->t_decode_us += states[i]->t_decode_us;
        ctx->state->n_sample += states[i]->n_sample;
        ctx->state->n_encode += states[i]->n_encode;
        ctx->state->n_decode += states[i]->n_decode;

        parakeet_free_state(states[i]);
    }

    // average the timings
    ctx->state->t_mel_us    /= n_processors;
    ctx->state->t_sample_us /= n_processors;
    ctx->state->t_encode_us /= n_processors;
    ctx->state->t_decode_us /= n_processors;

    // print information about the audio boundaries
    PARAKEET_LOG_WARN("\n");
    PARAKEET_LOG_WARN("%s: the audio has been split into %d chunks at the following times:\n", __func__, n_processors);
    for (int i = 0; i < n_processors - 1; ++i) {
        PARAKEET_LOG_WARN("%s: split %d - %s\n", __func__, (i + 1), to_timestamp(100*((i + 1)*n_samples_per_processor)/PARAKEET_SAMPLE_RATE + offset_t).c_str());
    }
    PARAKEET_LOG_WARN("%s: the transcription quality may be degraded near these boundaries\n", __func__);

    return ret;
}

int parakeet_full_n_segments_from_state(struct parakeet_state * state) {
    return state->result_all.size();
}

int parakeet_full_n_segments(struct parakeet_context * ctx) {
    return ctx->state->result_all.size();
}

int64_t parakeet_full_get_segment_t0_from_state(struct parakeet_state * state, int i_segment) {
    return state->result_all[i_segment].t0;
}

int64_t parakeet_full_get_segment_t1_from_state(struct parakeet_state * state, int i_segment) {
    return state->result_all[i_segment].t1;
}

int64_t parakeet_full_get_segment_t0(struct parakeet_context * ctx, int i_segment) {
    return parakeet_full_get_segment_t0_from_state(ctx->state, i_segment);
}

int64_t parakeet_full_get_segment_t1(struct parakeet_context * ctx, int i_segment) {
    return parakeet_full_get_segment_t1_from_state(ctx->state, i_segment);
}

const char * parakeet_full_get_segment_text_from_state(struct parakeet_state * state, int i_segment) {
    return state->result_all[i_segment].text.c_str();
}

const char * parakeet_full_get_segment_text(struct parakeet_context * ctx, int i_segment) {
    return ctx->state->result_all[i_segment].text.c_str();
}

int parakeet_full_n_tokens_from_state(struct parakeet_state * state, int i_segment) {
    return state->result_all[i_segment].tokens.size();
}

int parakeet_full_n_tokens(struct parakeet_context * ctx, int i_segment) {
    return ctx->state->result_all[i_segment].tokens.size();
}

const char * parakeet_full_get_token_text_from_state(struct parakeet_context * ctx, struct parakeet_state * state, int i_segment, int i_token) {
    return ctx->vocab.id_to_token[state->result_all[i_segment].tokens[i_token].id].c_str();
}

const char* parakeet_full_get_token_text(struct parakeet_context * ctx, int i_segment, int i_token) {
    return ctx->vocab.id_to_token[ctx->state->result_all[i_segment].tokens[i_token].id].c_str();
}

parakeet_token parakeet_full_get_token_id_from_state(struct parakeet_state * state, int i_segment, int i_token) {
    return state->result_all[i_segment].tokens[i_token].id;
}

parakeet_token parakeet_full_get_token_id(struct parakeet_context * ctx, int i_segment, int i_token) {
    return ctx->state->result_all[i_segment].tokens[i_token].id;
}

struct parakeet_token_data parakeet_full_get_token_data_from_state(struct parakeet_state * state, int i_segment, int i_token) {
    return state->result_all[i_segment].tokens[i_token];
}

struct parakeet_token_data parakeet_full_get_token_data(struct parakeet_context * ctx, int i_segment, int i_token) {
    return ctx->state->result_all[i_segment].tokens[i_token];
}

float parakeet_full_get_token_p_from_state(struct parakeet_state * state, int i_segment, int i_token) {
    return state->result_all[i_segment].tokens[i_token].p;
}

float parakeet_full_get_token_p(struct parakeet_context * ctx, int i_segment, int i_token) {
    return ctx->state->result_all[i_segment].tokens[i_token].p;
}

void parakeet_log_set(ggml_log_callback log_callback, void * user_data) {
    g_state.log_callback = log_callback ? log_callback : parakeet_log_callback_default;
    g_state.log_callback_user_data = user_data;
    ggml_log_set(g_state.log_callback, g_state.log_callback_user_data);
}

const char * parakeet_version(void) {
    return PARAKEET_VERSION;
}

GGML_ATTRIBUTE_FORMAT(2, 3)
static void parakeet_log_internal(ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    char buffer[1024];
    int len = vsnprintf(buffer, 1024, format, args);
    if (len < 1024) {
        g_state.log_callback(level, buffer, g_state.log_callback_user_data);
    } else {
        char* buffer2 = new char[len+1];
        vsnprintf(buffer2, len+1, format, args);
        buffer2[len] = 0;
        g_state.log_callback(level, buffer2, g_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args);
}

static void parakeet_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
#ifndef WHISPER_DEBUG
    if (level == GGML_LOG_LEVEL_DEBUG) {
        return;
    }
#endif
    fputs(text, stderr);
    fflush(stderr);
}
