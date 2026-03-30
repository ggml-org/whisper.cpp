#include "whisper-diarize.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cstdio>

// Define logging macros for consistency with whisper.cpp
#define WHISPER_LOG_ERROR(...) fprintf(stderr, "[ERROR] " __VA_ARGS__)
#define WHISPER_LOG_WARN(...)  fprintf(stderr, "[WARN]  " __VA_ARGS__)
#define WHISPER_LOG_INFO(...)  fprintf(stderr, "[INFO]  " __VA_ARGS__)

// Mel-spectrogram computation (80-bin, compatible with SpeechBrain ECAPA-TDNN)

#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160
#define WHISPER_SAMPLE_RATE 16000

#define MEL_N_BINS          80
#define MEL_FMIN            0.0f
#define MEL_FMAX            8000.0f

// FFT constants
#define FFT_SIZE            512  // Next power of 2 for efficiency

static float g_hann_window[WHISPER_N_FFT] = {0};
static int g_hann_computed = 0;

static void compute_hann_window() {
    if (g_hann_computed) return;

    // Hamming window (matching SpeechBrain pretrained model)
    for (int i = 0; i < WHISPER_N_FFT; i++) {
        g_hann_window[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (WHISPER_N_FFT - 1));
    }
    g_hann_computed = 1;
}

// Cooley-Tukey radix-2 FFT (complex-to-complex, in-place)
// data: interleaved complex [re0, im0, re1, im1, ...], length 2*n
static void fft_radix2(float * data, int n) {
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = data[2*i]; data[2*i] = data[2*j]; data[2*j] = tr;
            float ti = data[2*i+1]; data[2*i+1] = data[2*j+1]; data[2*j+1] = ti;
        }
    }
    // Butterfly passes
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * (float)M_PI / len;
        float wr = cosf(ang), wi = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cur_r = 1.0f, cur_i = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int u = i + j, v = i + j + len / 2;
                float tr = data[2*v]*cur_r - data[2*v+1]*cur_i;
                float ti = data[2*v]*cur_i + data[2*v+1]*cur_r;
                data[2*v]   = data[2*u]   - tr;
                data[2*v+1] = data[2*u+1] - ti;
                data[2*u]   += tr;
                data[2*u+1] += ti;
                float nr = cur_r*wr - cur_i*wi;
                cur_i = cur_r*wi + cur_i*wr;
                cur_r = nr;
            }
        }
    }
}

// Convert frequency (Hz) to mel scale
static inline float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

// Convert mel scale back to frequency (Hz)
static inline float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Create triangular mel filterbank (librosa-style continuous frequency mapping)
// Returns pointer to filters array of size n_mels * n_fft
static float * create_mel_filters(int n_mels, int n_fft, int sample_rate) {
    float * filters = (float *)calloc(n_mels * n_fft, sizeof(float));
    if (!filters) return NULL;

    // Compute mel boundary frequencies
    float fmin_mel = hz_to_mel(MEL_FMIN);
    float fmax_mel = hz_to_mel(MEL_FMAX);

    float * mel_hz = (float *)malloc((n_mels + 2) * sizeof(float));
    for (int i = 0; i < n_mels + 2; i++) {
        mel_hz[i] = mel_to_hz(fmin_mel + (fmax_mel - fmin_mel) * i / (n_mels + 1));
    }

    // Compute frequency of each FFT bin
    int fft_size = 2 * (n_fft - 1);
    float * fft_freqs = (float *)malloc(n_fft * sizeof(float));
    for (int k = 0; k < n_fft; k++) {
        fft_freqs[k] = (float)k * sample_rate / fft_size;
    }

    // Build triangular filters (peak=1, no area normalization)
    for (int m = 0; m < n_mels; m++) {
        float lower = mel_hz[m];
        float center = mel_hz[m + 1];
        float upper = mel_hz[m + 2];

        for (int k = 0; k < n_fft; k++) {
            float freq = fft_freqs[k];
            if (freq >= lower && freq < center && center > lower) {
                filters[m * n_fft + k] = (freq - lower) / (center - lower);
            } else if (freq >= center && freq < upper && upper > center) {
                filters[m * n_fft + k] = (upper - freq) / (upper - center);
            }
        }

    }

    free(mel_hz);
    free(fft_freqs);
    return filters;
}

// Compute 80-bin mel-spectrogram from PCM samples
float * whisper_compute_mel_80(const float * samples, int n_samples) {
    if (!samples || n_samples <= 0) {
        return NULL;
    }

    compute_hann_window();

    // Center padding: add n_fft/2 samples on both sides
    int pad = WHISPER_N_FFT / 2;  // 200 samples
    int padded_len = n_samples + 2 * pad;

    // Create padded signal
    float * padded_samples = (float *)calloc(padded_len, sizeof(float));
    if (!padded_samples) {
        return NULL;
    }
    // Copy original samples to center (zero padding at edges already done by calloc)
    memcpy(padded_samples + pad, samples, n_samples * sizeof(float));

    // Calculate number of frames (now with padded length)
    int n_frames = (padded_len - WHISPER_N_FFT) / WHISPER_HOP_LENGTH + 1;
    if (n_frames <= 0) {
        free(padded_samples);
        return NULL;
    }

    // Allocate output mel array [n_frames, 80]
    float * mel = (float *)calloc(n_frames * MEL_N_BINS, sizeof(float));
    if (!mel) {
        return NULL;
    }

    // Create mel filterbank once (n_fft=400 → 201 bins)
    static float * mel_filters = NULL;
    static int mel_filters_initialized = 0;
    int n_fft_bins = 1 + WHISPER_N_FFT / 2;  // 201 bins for 400-point DFT
    if (!mel_filters_initialized) {
        mel_filters = create_mel_filters(MEL_N_BINS, n_fft_bins, WHISPER_SAMPLE_RATE);
        mel_filters_initialized = 1;
    }

    if (!mel_filters) {
        free(mel);
        return NULL;
    }

    // Process each frame
    for (int t = 0; t < n_frames; t++) {
        int offset = t * WHISPER_HOP_LENGTH;

        // Extract frame, apply Hamming window, zero-pad to 512 as complex interleaved
        float * fft_buf = (float *)calloc(FFT_SIZE * 2, sizeof(float));
        for (int i = 0; i < WHISPER_N_FFT; i++) {
            fft_buf[2*i] = padded_samples[offset + i] * g_hann_window[i];
            // imaginary = 0 (calloc)
        }
        // positions WHISPER_N_FFT..FFT_SIZE-1 are zero-padded (calloc)

        // In-place 512-point FFT
        fft_radix2(fft_buf, FFT_SIZE);

        // Extract power spectrum for first n_fft_bins=201 bins
        // Bin k of 512-point FFT → freq = k * sr / 512
        // Bin k of 400-point DFT → freq = k * sr / 400
        // Map: for target bin j (400-point), find 512-point bin at same frequency
        // freq_j = j * sr / 400 → k_512 = j * 512 / 400 = j * 1.28
        // Use linear interpolation between adjacent 512-point bins
        float * mag = (float *)malloc(n_fft_bins * sizeof(float));
        for (int j = 0; j < n_fft_bins; j++) {
            float k_f = j * (float)FFT_SIZE / WHISPER_N_FFT;  // fractional 512-bin index
            int k0 = (int)k_f;
            float frac = k_f - k0;
            int k1 = k0 + 1;
            if (k1 >= FFT_SIZE / 2 + 1) k1 = k0;

            // Power at each 512-bin
            float p0 = fft_buf[2*k0]*fft_buf[2*k0] + fft_buf[2*k0+1]*fft_buf[2*k0+1];
            float p1 = fft_buf[2*k1]*fft_buf[2*k1] + fft_buf[2*k1+1]*fft_buf[2*k1+1];

            // Interpolate
            mag[j] = p0 * (1.0f - frac) + p1 * frac;
        }

        // Apply mel filterbank
        for (int m = 0; m < MEL_N_BINS; m++) {
            float mel_val = 0.0f;
            for (int k = 0; k < n_fft_bins; k++) {
                mel_val += mag[k] * mel_filters[m * n_fft_bins + k];
            }

            // dB scale: 10 * log10(max(x, 1e-10))
            mel[t * MEL_N_BINS + m] = 10.0f * log10f(fmaxf(mel_val, 1e-10f));
        }

        free(fft_buf);
        free(mag);
    }

    // top_db clipping: clamp to (max_db - 80)
    float max_db = -1e30f;
    for (int i = 0; i < n_frames * MEL_N_BINS; i++) {
        if (mel[i] > max_db) max_db = mel[i];
    }
    float min_db = max_db - 80.0f;
    for (int i = 0; i < n_frames * MEL_N_BINS; i++) {
        if (mel[i] < min_db) mel[i] = min_db;
    }

    free(padded_samples);
    return mel;
}

// Get number of frames for given sample count
int whisper_get_mel_n_frames(int n_samples) {
    if (n_samples <= 0) {
        return 0;
    }
    // Match whisper_compute_mel_80: center padding adds WHISPER_N_FFT/2 on each side
    int padded_len = n_samples + WHISPER_N_FFT;  // 2 * (N_FFT/2)
    return (padded_len - WHISPER_N_FFT) / WHISPER_HOP_LENGTH + 1;
}

// Free mel-spectrogram buffer
void whisper_mel_free(float * mel) {
    if (mel) {
        free(mel);
    }
}

// Speaker encoder forward pass

static struct ggml_tensor * apply_simple_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x) {
    if (!x) {
        WHISPER_LOG_ERROR("apply_simple_norm: NULL input tensor\n");
        return x;
    }
    return ggml_norm(ctx, x, 1e-5f);
}

// Reshape conv1d output to 4D for broadcasting
static struct ggml_tensor * ensure_4d_from_conv1d(struct ggml_context * ctx, struct ggml_tensor * t) {
    // Conv1d outputs 3D: [OW, OC, batch]
    // Reshape to 4D: [OW, OC, batch, 1]
    return ggml_reshape_4d(ctx, t, t->ne[0], t->ne[1], t->ne[2], 1);
}

// Precompute BatchNorm: scale = gamma / sqrt(var + eps), offset = beta - mean * scale
static void precompute_bn_params(
    struct ggml_tensor * bn_mean,   // [C] (mu)
    struct ggml_tensor * bn_var,    // [C] (sigma²)
    struct ggml_tensor * bn_gamma,  // [C] (weight/scale parameter)
    struct ggml_tensor * bn_beta,   // [C] (bias/shift parameter)
    struct ggml_tensor * bn_scale,
    struct ggml_tensor * bn_offset)
{
    if (!bn_mean || !bn_var || !bn_gamma || !bn_beta || !bn_scale || !bn_offset) {
        WHISPER_LOG_ERROR("precompute_bn_params: NULL tensor\n");
        return;
    }

    int32_t C = bn_scale->ne[0];
    float * mean = (float *)bn_mean->data;
    float * var = (float *)bn_var->data;
    float * gamma = (float *)bn_gamma->data;
    float * beta = (float *)bn_beta->data;
    float * scale = (float *)bn_scale->data;
    float * offset = (float *)bn_offset->data;

    const float eps = 1e-5f;

    for (int32_t c = 0; c < C; c++) {
        scale[c] = gamma[c] / sqrtf(var[c] + eps);
        offset[c] = beta[c] - mean[c] * scale[c];
    }
}

// Apply BatchNorm: output = x * scale + offset
static struct ggml_tensor * apply_runtime_bn(
    struct ggml_context * ctx,
    struct ggml_tensor * x,           // [T, C, 1, 1]
    struct ggml_tensor * bn_scale,    // [C]
    struct ggml_tensor * bn_offset)   // [C]
{
    if (!x || !bn_scale || !bn_offset) {
        WHISPER_LOG_ERROR("apply_runtime_bn: NULL tensor\n");
        return x;
    }

    // Reshape scale and offset for broadcasting: [C] → [1, C, 1, 1]
    struct ggml_tensor * scale_reshaped = ggml_reshape_4d(ctx, bn_scale, 1, bn_scale->ne[0], 1, 1);
    struct ggml_tensor * offset_reshaped = ggml_reshape_4d(ctx, bn_offset, 1, bn_offset->ne[0], 1, 1);

    // Apply: output = (x * scale) + offset
    struct ggml_tensor * scaled = ggml_mul(ctx, x, scale_reshaped);
    struct ggml_tensor * output = ggml_add(ctx, scaled, offset_reshaped);

    return output;
}

static struct ggml_tensor * ggml_conv_weight_f32_to_f16(
    struct ggml_context * ctx,
    struct ggml_tensor * weight_f32) {

    // Convert F32 → F16 (required by ggml_conv_1d)
    int64_t ne0 = weight_f32->ne[0];
    int64_t ne1 = weight_f32->ne[1];
    int64_t ne2 = weight_f32->ne[2];



    struct ggml_tensor * weight_f16 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, ne0, ne1, ne2);
    if (!weight_f16) {
        WHISPER_LOG_ERROR("Failed to allocate F16 weight tensor\n");
        return NULL;
    }

    float * src = (float *)weight_f32->data;
    ggml_fp16_t * dst = (ggml_fp16_t *)weight_f16->data;

    int64_t n = ne0 * ne1 * ne2;
    for (int64_t i = 0; i < n; i++) {
        dst[i] = ggml_fp32_to_fp16(src[i]);
    }

    return weight_f16;
}

struct whisper_speaker_encoder {
    struct whisper_speaker_model * model;
    struct ggml_context * ctx;
    struct ggml_cgraph * graph;
    struct ggml_tensor * input_mel;
    struct ggml_tensor * output_embedding;
    int n_frames;
    int n_mels;
};

// Initialize encoder with loaded speaker model
struct whisper_speaker_encoder * whisper_speaker_encoder_new(
    struct whisper_speaker_model * model,
    int n_frames,
    int /*device*/) {

    if (!model || n_frames <= 0) {
        return NULL;
    }

    // Allocate encoder struct
    struct whisper_speaker_encoder * encoder =
        (struct whisper_speaker_encoder *)malloc(sizeof(struct whisper_speaker_encoder));
    if (!encoder) {
        return NULL;
    }

    encoder->model = model;
    encoder->n_frames = n_frames;
    encoder->n_mels = 80;  // Fixed for ECAPA-TDNN

    // Dynamic context size: base 200MB + ~0.5MB per frame for intermediate tensors
    size_t ctx_bytes = (size_t)200 * 1024 * 1024 + (size_t)n_frames * 512 * 1024;
    struct ggml_init_params params = {
        .mem_size = ctx_bytes,
        .mem_buffer = NULL,
        .no_alloc = false,
    };

    encoder->ctx = ggml_init(params);
    if (!encoder->ctx) {
        free(encoder);
        return NULL;
    }

    // Create ggml_cgraph
    encoder->graph = ggml_new_graph(encoder->ctx);
    if (!encoder->graph) {
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    // Input mel: [80, T] in ggml, transposed to [T, 80] in the graph
    encoder->input_mel = ggml_new_tensor_2d(encoder->ctx, GGML_TYPE_F32,
                                             encoder->n_mels, encoder->n_frames);
    if (!encoder->input_mel) {
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    // Set input tensor as model input
    ggml_set_name(encoder->input_mel, "input_mel");
    ggml_set_input(encoder->input_mel);

    // Allocate output tensor placeholder: [192]
    encoder->output_embedding = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, 192);
    if (!encoder->output_embedding) {
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    // Build ECAPA-TDNN forward computation graph
    struct ggml_tensor * cur = ggml_cont(encoder->ctx, ggml_transpose(encoder->ctx, encoder->input_mel));  // [T, 80]

    // Layer 0: Conv1d(80→1024, k=5)

    struct ggml_tensor * layer0_w = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.0.conv.conv.weight");
    struct ggml_tensor * layer0_b = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.0.conv.conv.bias");

    if (!layer0_w || !layer0_b) {
        WHISPER_LOG_ERROR("Layer 0: Failed to load weights\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }



    // Conv1d(k=5, s=1, p=2, d=1)
    struct ggml_tensor * layer0_w_ggml = ggml_conv_weight_f32_to_f16(encoder->ctx, layer0_w);
    if (!layer0_w_ggml) {
        WHISPER_LOG_ERROR("Layer 0: Failed to convert weight layout\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    // Verify weight data
    if (!layer0_w_ggml->data) {
        WHISPER_LOG_ERROR("Layer 0: Weight tensor has NULL data!\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    cur = ggml_conv_1d(encoder->ctx, layer0_w_ggml, cur, 1, 2, 1);
    if (!cur) {
        WHISPER_LOG_ERROR("Layer 0: ggml_conv_1d failed\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }
    cur = ensure_4d_from_conv1d(encoder->ctx, cur);

    struct ggml_tensor * layer0_b_reshaped = ggml_reshape_3d(encoder->ctx, layer0_b, 1, 1024, 1);
    cur = ggml_add(encoder->ctx, cur, layer0_b_reshaped);


    // ReLU
    cur = ggml_relu(encoder->ctx, cur);

    // BatchNorm
    struct ggml_tensor * bn0_mean = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.0.norm.norm.running_mean");
    struct ggml_tensor * bn0_var = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.0.norm.norm.running_var");
    struct ggml_tensor * bn0_gamma = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.0.norm.norm.weight");
    struct ggml_tensor * bn0_beta = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.0.norm.norm.bias");

    if (bn0_mean && bn0_var && bn0_gamma && bn0_beta) {
        int32_t bn0_channels = cur->ne[1];
        struct ggml_tensor * bn0_scale = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, bn0_channels);
        struct ggml_tensor * bn0_offset = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, bn0_channels);

            precompute_bn_params(bn0_mean, bn0_var, bn0_gamma, bn0_beta, bn0_scale, bn0_offset);
        cur = apply_runtime_bn(encoder->ctx, cur, bn0_scale, bn0_offset);
    } else {
        WHISPER_LOG_WARN("Layer 0: Missing BN tensors, skipping BN\n");
    }

    // Layers 1-3: SE-Res2Net blocks
    struct ggml_tensor * layer1_tdnn1_w = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.1.tdnn1.conv.conv.weight");
    struct ggml_tensor * layer1_tdnn1_b = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.1.tdnn1.conv.conv.bias");

    // Res2Net branches (7 branches, each with dilation 1,2,3,4,1,2,3)
    struct ggml_tensor * layer1_res2net[7] = {NULL};
    struct ggml_tensor * layer1_res2net_b[7] = {NULL};
    const char * res2net_names[] = {
        "mods.embedding_model.blocks.1.res2net_block.blocks.0.conv.conv.weight",
        "mods.embedding_model.blocks.1.res2net_block.blocks.1.conv.conv.weight",
        "mods.embedding_model.blocks.1.res2net_block.blocks.2.conv.conv.weight",
        "mods.embedding_model.blocks.1.res2net_block.blocks.3.conv.conv.weight",
        "mods.embedding_model.blocks.1.res2net_block.blocks.4.conv.conv.weight",
        "mods.embedding_model.blocks.1.res2net_block.blocks.5.conv.conv.weight",
        "mods.embedding_model.blocks.1.res2net_block.blocks.6.conv.conv.weight",
    };
    const char * res2net_bias_names[] = {
        "mods.embedding_model.blocks.1.res2net_block.blocks.0.conv.conv.bias",
        "mods.embedding_model.blocks.1.res2net_block.blocks.1.conv.conv.bias",
        "mods.embedding_model.blocks.1.res2net_block.blocks.2.conv.conv.bias",
        "mods.embedding_model.blocks.1.res2net_block.blocks.3.conv.conv.bias",
        "mods.embedding_model.blocks.1.res2net_block.blocks.4.conv.conv.bias",
        "mods.embedding_model.blocks.1.res2net_block.blocks.5.conv.conv.bias",
        "mods.embedding_model.blocks.1.res2net_block.blocks.6.conv.conv.bias",
    };

    for (int i = 0; i < 7; i++) {
        layer1_res2net[i] = whisper_speaker_find_tensor(encoder->model, res2net_names[i]);
        layer1_res2net_b[i] = whisper_speaker_find_tensor(encoder->model, res2net_bias_names[i]);
        if (!layer1_res2net[i] || !layer1_res2net_b[i]) {
            WHISPER_LOG_ERROR("Layer 1: Failed to load Res2Net branch %d\n", i);
            ggml_free(encoder->ctx);
            free(encoder);
            return NULL;
        }
    }

    // SE block weights
    struct ggml_tensor * layer1_se_fc1_w = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.1.se_block.conv1.conv.weight");
    struct ggml_tensor * layer1_se_fc1_b = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.1.se_block.conv1.conv.bias");
    struct ggml_tensor * layer1_se_fc2_w = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.1.se_block.conv2.conv.weight");
    struct ggml_tensor * layer1_se_fc2_b = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.1.se_block.conv2.conv.bias");

    // TDNN2
    struct ggml_tensor * layer1_tdnn2_w = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.1.tdnn2.conv.conv.weight");
    struct ggml_tensor * layer1_tdnn2_b = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.1.tdnn2.conv.conv.bias");

    if (!layer1_tdnn1_w || !layer1_tdnn1_b || !layer1_se_fc1_w || !layer1_se_fc1_b ||
        !layer1_se_fc2_w || !layer1_se_fc2_b || !layer1_tdnn2_w || !layer1_tdnn2_b) {
        WHISPER_LOG_ERROR("Layer 1: Failed to load SE block weights\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }


    // Current input
    struct ggml_tensor * layer1_input = cur;  // [T, 1024]

    // TDNN1: [T, 1024] → [T, 1024]
    struct ggml_tensor * layer1_tdnn1_w_ggml = ggml_conv_weight_f32_to_f16(encoder->ctx, layer1_tdnn1_w);
    if (!layer1_tdnn1_w_ggml) {
        WHISPER_LOG_ERROR("Layer 1: Failed to convert TDNN1 weight layout\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }
    struct ggml_tensor * layer1_tdnn1_out = ggml_conv_1d(encoder->ctx, layer1_tdnn1_w_ggml, layer1_input, 1, 0, 1);
    layer1_tdnn1_out = ggml_reshape_4d(encoder->ctx, layer1_tdnn1_out, layer1_tdnn1_out->ne[0], layer1_tdnn1_out->ne[1], layer1_tdnn1_out->ne[2], 1);
    struct ggml_tensor * layer1_tdnn1_b_reshaped = ggml_reshape_3d(encoder->ctx, layer1_tdnn1_b, 1, 1024, 1);
    layer1_tdnn1_out = ggml_add(encoder->ctx, layer1_tdnn1_out, layer1_tdnn1_b_reshaped);
    layer1_tdnn1_out = ggml_relu(encoder->ctx, layer1_tdnn1_out);

    // Layer 1 TDNN1: Runtime BatchNorm
    {
        struct ggml_tensor * bn1_tdnn1_mean = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.1.tdnn1.norm.norm.running_mean");
        struct ggml_tensor * bn1_tdnn1_var = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.1.tdnn1.norm.norm.running_var");
        struct ggml_tensor * bn1_tdnn1_gamma = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.1.tdnn1.norm.norm.weight");
        struct ggml_tensor * bn1_tdnn1_beta = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.1.tdnn1.norm.norm.bias");

        if (bn1_tdnn1_mean && bn1_tdnn1_var && bn1_tdnn1_gamma && bn1_tdnn1_beta) {
            int32_t bn1_channels = layer1_tdnn1_out->ne[1];
            struct ggml_tensor * bn1_scale = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, bn1_channels);
            struct ggml_tensor * bn1_offset = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, bn1_channels);

                precompute_bn_params(bn1_tdnn1_mean, bn1_tdnn1_var, bn1_tdnn1_gamma, bn1_tdnn1_beta, bn1_scale, bn1_offset);

            layer1_tdnn1_out = apply_runtime_bn(encoder->ctx, layer1_tdnn1_out, bn1_scale, bn1_offset);
        }
    }

    // Res2Net: Split [T, 1024] into 8 groups of [T, 128], apply 7 dilated convs + 1 identity, concatenate back
    // In ggml column-major: ne[0]=T varies fastest, so channel stride = nb[1] = T * elem_size
    struct ggml_tensor * layer1_res2net_splits[8];
    const int32_t group_channels = 128;

    size_t chan_stride = layer1_tdnn1_out->nb[1]; // bytes per channel = T * elem_size

    for (int g = 0; g < 8; g++) {
        size_t offset = g * group_channels * chan_stride;
        layer1_res2net_splits[g] = ggml_view_2d(encoder->ctx,
            layer1_tdnn1_out,
            layer1_tdnn1_out->ne[0],  // T frames
            group_channels,            // 128 channels
            chan_stride,               // stride per channel (T * elem_size)
            offset);
        ggml_set_name(layer1_res2net_splits[g], "res2net_split");
    }

    // Res2Net: chunk[0]=identity, chunk[i]=conv(chunk[i]+y[i-1]) for i>=2
    // All 7 blocks use dilation=2 for Layer 1
    struct ggml_tensor * layer1_res2net_branches[8];

    // Chunk 0: identity
    layer1_res2net_branches[0] = layer1_res2net_splits[0];

    // Chunks 1-7: apply blocks[0..6]
    for (int i = 1; i < 8; i++) {
        int b = i - 1;  // block index

        struct ggml_tensor * branch_w_ggml = ggml_conv_weight_f32_to_f16(
            encoder->ctx, layer1_res2net[b]);
        if (!branch_w_ggml) {
            WHISPER_LOG_ERROR("Layer 1: Failed to convert Res2Net branch %d weight\n", b);
            ggml_free(encoder->ctx);
            free(encoder);
            return NULL;
        }

        // Cumulative: input = chunk[i] + y[i-1] for i >= 2
        struct ggml_tensor * conv_input = layer1_res2net_splits[i];
        if (i >= 2) {
            conv_input = ggml_add(encoder->ctx, conv_input, layer1_res2net_branches[i - 1]);
        }

        // Conv1d(128→128, k=3, dilation=2, padding=2)
        struct ggml_tensor * branch_conv = ggml_conv_1d(encoder->ctx,
            branch_w_ggml, conv_input, 1, 2, 2);

        branch_conv = ensure_4d_from_conv1d(encoder->ctx, branch_conv);

        struct ggml_tensor * branch_b_reshaped = ggml_reshape_3d(encoder->ctx,
            layer1_res2net_b[b], 1, 128, 1);
        branch_conv = ggml_add(encoder->ctx, branch_conv, branch_b_reshaped);
        branch_conv = ggml_relu(encoder->ctx, branch_conv);

        // Runtime BatchNorm
        {
            char bn_name[256];
            snprintf(bn_name, sizeof(bn_name), "mods.embedding_model.blocks.1.res2net_block.blocks.%d.norm.norm.running_mean", b);
            struct ggml_tensor * bn_mean = whisper_speaker_find_tensor(encoder->model, bn_name);
            snprintf(bn_name, sizeof(bn_name), "mods.embedding_model.blocks.1.res2net_block.blocks.%d.norm.norm.running_var", b);
            struct ggml_tensor * bn_var = whisper_speaker_find_tensor(encoder->model, bn_name);
            snprintf(bn_name, sizeof(bn_name), "mods.embedding_model.blocks.1.res2net_block.blocks.%d.norm.norm.weight", b);
            struct ggml_tensor * bn_gamma = whisper_speaker_find_tensor(encoder->model, bn_name);
            snprintf(bn_name, sizeof(bn_name), "mods.embedding_model.blocks.1.res2net_block.blocks.%d.norm.norm.bias", b);
            struct ggml_tensor * bn_beta = whisper_speaker_find_tensor(encoder->model, bn_name);

            if (bn_mean && bn_var && bn_gamma && bn_beta) {
                struct ggml_tensor * bn_scale = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, 128);
                struct ggml_tensor * bn_offset = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, 128);
                precompute_bn_params(bn_mean, bn_var, bn_gamma, bn_beta, bn_scale, bn_offset);
                branch_conv = apply_runtime_bn(encoder->ctx, branch_conv, bn_scale, bn_offset);
            }
        }

        layer1_res2net_branches[i] = ggml_reshape_2d(encoder->ctx, branch_conv,
            branch_conv->ne[0], branch_conv->ne[1]);
    }

    // Step 3: Concatenate 8 branches back to [T, 1024]
    struct ggml_tensor * res2net_concat = layer1_res2net_branches[0];
    for (int g = 1; g < 8; g++) {
        res2net_concat = ggml_concat(encoder->ctx, res2net_concat,
            layer1_res2net_branches[g], 1);
        if (!res2net_concat) {
            WHISPER_LOG_ERROR("Layer 1: Failed to concatenate Res2Net branch %d\n", g);
            ggml_free(encoder->ctx);
            free(encoder);
            return NULL;
        }
    }

    if (res2net_concat->ne[1] != 1024) {
        WHISPER_LOG_ERROR("Layer 1 Res2Net: ERROR - concat output is %lld, expected 1024\n",
            res2net_concat->ne[1]);
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    // TDNN2

    // TDNN2: [T, 1024] → [T, 1024]
    struct ggml_tensor * layer1_tdnn2_w_ggml = ggml_conv_weight_f32_to_f16(encoder->ctx, layer1_tdnn2_w);
    if (!layer1_tdnn2_w_ggml) {
        WHISPER_LOG_ERROR("Layer 1: Failed to convert TDNN2 weight layout\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }
    struct ggml_tensor * layer1_tdnn2_out = ggml_conv_1d(encoder->ctx, layer1_tdnn2_w_ggml, res2net_concat, 1, 0, 1);
    layer1_tdnn2_out = ggml_reshape_4d(encoder->ctx, layer1_tdnn2_out, layer1_tdnn2_out->ne[0], layer1_tdnn2_out->ne[1], layer1_tdnn2_out->ne[2], 1);
    struct ggml_tensor * layer1_tdnn2_b_reshaped = ggml_reshape_3d(encoder->ctx, layer1_tdnn2_b, 1, 1024, 1);
    layer1_tdnn2_out = ggml_add(encoder->ctx, layer1_tdnn2_out, layer1_tdnn2_b_reshaped);
    layer1_tdnn2_out = ggml_relu(encoder->ctx, layer1_tdnn2_out);

    // Layer 1 TDNN2: Runtime BatchNorm
    {
        struct ggml_tensor * bn1_tdnn2_mean = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.1.tdnn2.norm.norm.running_mean");
        struct ggml_tensor * bn1_tdnn2_var = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.1.tdnn2.norm.norm.running_var");
        struct ggml_tensor * bn1_tdnn2_gamma = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.1.tdnn2.norm.norm.weight");
        struct ggml_tensor * bn1_tdnn2_beta = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.1.tdnn2.norm.norm.bias");

        if (bn1_tdnn2_mean && bn1_tdnn2_var && bn1_tdnn2_gamma && bn1_tdnn2_beta) {
            int32_t bn1_channels = layer1_tdnn2_out->ne[1];
            struct ggml_tensor * bn1_scale = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, bn1_channels);
            struct ggml_tensor * bn1_offset = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, bn1_channels);

                precompute_bn_params(bn1_tdnn2_mean, bn1_tdnn2_var, bn1_tdnn2_gamma, bn1_tdnn2_beta, bn1_scale, bn1_offset);

            layer1_tdnn2_out = apply_runtime_bn(encoder->ctx, layer1_tdnn2_out, bn1_scale, bn1_offset);
        }
    }

    // SE Block: GlobalAvgPool → FC1 → ReLU → FC2 → Sigmoid → Scale

    // Global average pooling: [T, 1024] → [1, 1024]
    struct ggml_tensor * se_gap = ggml_pool_1d(encoder->ctx,
        layer1_tdnn2_out,
        GGML_OP_POOL_AVG,
        (int)layer1_tdnn2_out->ne[0],  // kernel = full seq length
        (int)layer1_tdnn2_out->ne[0],  // stride = full seq length
        0);

    // Reshape gap to 1D: [1024]
    struct ggml_tensor * se_gap_1d = ggml_reshape_1d(encoder->ctx, se_gap, 1024);

    // FC1: [1024] → [128] with ReLU
    struct ggml_tensor * se_fc1_w_ggml = ggml_conv_weight_f32_to_f16(
        encoder->ctx, layer1_se_fc1_w);
    if (!se_fc1_w_ggml) {
        WHISPER_LOG_ERROR("Layer 1 SE: Failed to convert FC1 weight\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    struct ggml_tensor * se_fc1 = ggml_mul_mat(encoder->ctx,
        ggml_reshape_2d(encoder->ctx, se_fc1_w_ggml, 1024, 128),
        ggml_reshape_2d(encoder->ctx, se_gap_1d, 1024, 1));
    se_fc1 = ggml_reshape_1d(encoder->ctx, se_fc1, 128);
    se_fc1 = ggml_add(encoder->ctx, se_fc1, layer1_se_fc1_b);
    se_fc1 = ggml_relu(encoder->ctx, se_fc1);

    // FC2: [128] → [1024] with Sigmoid
    struct ggml_tensor * se_fc2_w_ggml = ggml_conv_weight_f32_to_f16(
        encoder->ctx, layer1_se_fc2_w);
    if (!se_fc2_w_ggml) {
        WHISPER_LOG_ERROR("Layer 1 SE: Failed to convert FC2 weight\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    struct ggml_tensor * se_fc2 = ggml_mul_mat(encoder->ctx,
        ggml_reshape_2d(encoder->ctx, se_fc2_w_ggml, 128, 1024),
        ggml_reshape_2d(encoder->ctx, se_fc1, 128, 1));
    se_fc2 = ggml_reshape_1d(encoder->ctx, se_fc2, 1024);
    se_fc2 = ggml_add(encoder->ctx, se_fc2, layer1_se_fc2_b);
    struct ggml_tensor * se_gates = ggml_sigmoid(encoder->ctx, se_fc2);

    // Scale: [T, 1024] × [1024] element-wise
    struct ggml_tensor * se_gates_reshaped = ggml_reshape_3d(encoder->ctx,
        se_gates, 1, 1024, 1);

    struct ggml_tensor * layer1_se_out = ggml_mul(encoder->ctx,
        layer1_tdnn2_out, se_gates_reshaped);

    // Residual connection
    cur = ggml_add(encoder->ctx, layer1_se_out, layer1_input);

    struct ggml_tensor * layer1_out = cur;  // [n_frames, 1024]

    // Layer 2
    struct ggml_tensor * layer2_tdnn1_w = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.2.tdnn1.conv.conv.weight");
    struct ggml_tensor * layer2_tdnn1_b = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.2.tdnn1.conv.conv.bias");

    // Res2Net branches (7 branches)
    struct ggml_tensor * layer2_res2net[7] = {NULL};
    struct ggml_tensor * layer2_res2net_b[7] = {NULL};
    const char * res2net_names_2[] = {
        "mods.embedding_model.blocks.2.res2net_block.blocks.0.conv.conv.weight",
        "mods.embedding_model.blocks.2.res2net_block.blocks.1.conv.conv.weight",
        "mods.embedding_model.blocks.2.res2net_block.blocks.2.conv.conv.weight",
        "mods.embedding_model.blocks.2.res2net_block.blocks.3.conv.conv.weight",
        "mods.embedding_model.blocks.2.res2net_block.blocks.4.conv.conv.weight",
        "mods.embedding_model.blocks.2.res2net_block.blocks.5.conv.conv.weight",
        "mods.embedding_model.blocks.2.res2net_block.blocks.6.conv.conv.weight",
    };
    const char * res2net_bias_names_2[] = {
        "mods.embedding_model.blocks.2.res2net_block.blocks.0.conv.conv.bias",
        "mods.embedding_model.blocks.2.res2net_block.blocks.1.conv.conv.bias",
        "mods.embedding_model.blocks.2.res2net_block.blocks.2.conv.conv.bias",
        "mods.embedding_model.blocks.2.res2net_block.blocks.3.conv.conv.bias",
        "mods.embedding_model.blocks.2.res2net_block.blocks.4.conv.conv.bias",
        "mods.embedding_model.blocks.2.res2net_block.blocks.5.conv.conv.bias",
        "mods.embedding_model.blocks.2.res2net_block.blocks.6.conv.conv.bias",
    };

    for (int i = 0; i < 7; i++) {
        layer2_res2net[i] = whisper_speaker_find_tensor(encoder->model, res2net_names_2[i]);
        layer2_res2net_b[i] = whisper_speaker_find_tensor(encoder->model, res2net_bias_names_2[i]);
        if (!layer2_res2net[i] || !layer2_res2net_b[i]) {
            WHISPER_LOG_ERROR("Layer 2: Failed to load Res2Net branch %d\n", i);
            ggml_free(encoder->ctx);
            free(encoder);
            return NULL;
        }
    }

    // SE block weights
    struct ggml_tensor * layer2_se_fc1_w = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.2.se_block.conv1.conv.weight");
    struct ggml_tensor * layer2_se_fc1_b = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.2.se_block.conv1.conv.bias");
    struct ggml_tensor * layer2_se_fc2_w = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.2.se_block.conv2.conv.weight");
    struct ggml_tensor * layer2_se_fc2_b = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.2.se_block.conv2.conv.bias");

    // TDNN2
    struct ggml_tensor * layer2_tdnn2_w = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.2.tdnn2.conv.conv.weight");
    struct ggml_tensor * layer2_tdnn2_b = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.2.tdnn2.conv.conv.bias");

    if (!layer2_tdnn1_w || !layer2_tdnn1_b || !layer2_se_fc1_w || !layer2_se_fc1_b ||
        !layer2_se_fc2_w || !layer2_se_fc2_b || !layer2_tdnn2_w || !layer2_tdnn2_b) {
        WHISPER_LOG_ERROR("Layer 2: Failed to load SE block weights\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    // Current input
    struct ggml_tensor * layer2_input = cur;  // [T, 1024]

    // TDNN1: [T, 1024] → [T, 1024]
    struct ggml_tensor * layer2_tdnn1_w_ggml = ggml_conv_weight_f32_to_f16(encoder->ctx, layer2_tdnn1_w);
    if (!layer2_tdnn1_w_ggml) {
        WHISPER_LOG_ERROR("Layer 2: Failed to convert TDNN1 weight layout\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }
    struct ggml_tensor * layer2_tdnn1_out = ggml_conv_1d(encoder->ctx, layer2_tdnn1_w_ggml, layer2_input, 1, 0, 1);
    layer2_tdnn1_out = ggml_reshape_4d(encoder->ctx, layer2_tdnn1_out, layer2_tdnn1_out->ne[0], layer2_tdnn1_out->ne[1], layer2_tdnn1_out->ne[2], 1);
    struct ggml_tensor * layer2_tdnn1_b_reshaped = ggml_reshape_3d(encoder->ctx, layer2_tdnn1_b, 1, 1024, 1);
    layer2_tdnn1_out = ggml_add(encoder->ctx, layer2_tdnn1_out, layer2_tdnn1_b_reshaped);
    layer2_tdnn1_out = ggml_relu(encoder->ctx, layer2_tdnn1_out);

    // Layer 2 TDNN1: Runtime BatchNorm
    {
        struct ggml_tensor * bn2_tdnn1_mean = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.2.tdnn1.norm.norm.running_mean");
        struct ggml_tensor * bn2_tdnn1_var = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.2.tdnn1.norm.norm.running_var");
        struct ggml_tensor * bn2_tdnn1_gamma = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.2.tdnn1.norm.norm.weight");
        struct ggml_tensor * bn2_tdnn1_beta = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.2.tdnn1.norm.norm.bias");

        if (bn2_tdnn1_mean && bn2_tdnn1_var && bn2_tdnn1_gamma && bn2_tdnn1_beta) {
            int32_t bn2_channels = layer2_tdnn1_out->ne[1];
            struct ggml_tensor * bn2_scale = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, bn2_channels);
            struct ggml_tensor * bn2_offset = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, bn2_channels);

                precompute_bn_params(bn2_tdnn1_mean, bn2_tdnn1_var, bn2_tdnn1_gamma, bn2_tdnn1_beta, bn2_scale, bn2_offset);

            layer2_tdnn1_out = apply_runtime_bn(encoder->ctx, layer2_tdnn1_out, bn2_scale, bn2_offset);
        }
    }

    // Res2Net: Split [T, 1024] into 8 groups of [T, 128]
    struct ggml_tensor * layer2_res2net_splits[8];
    size_t chan_stride_l2 = layer2_tdnn1_out->nb[1];

    for (int g = 0; g < 8; g++) {
        size_t offset = g * group_channels * chan_stride_l2;
        layer2_res2net_splits[g] = ggml_view_2d(encoder->ctx,
            layer2_tdnn1_out,
            layer2_tdnn1_out->ne[0],
            group_channels,
            chan_stride_l2,
            offset);
        ggml_set_name(layer2_res2net_splits[g], "layer2_res2net_split");
    }

    // Res2Net: chunk[0]=identity, chunk[i]=conv(chunk[i]+y[i-1]) for i>=2
    // All 7 blocks use dilation=3 for Layer 2
    struct ggml_tensor * layer2_res2net_branches[8];

    // Chunk 0: identity
    layer2_res2net_branches[0] = layer2_res2net_splits[0];

    // Chunks 1-7: apply blocks[0..6]
    for (int i = 1; i < 8; i++) {
        int b = i - 1;  // block index

        struct ggml_tensor * branch_w_ggml = ggml_conv_weight_f32_to_f16(
            encoder->ctx, layer2_res2net[b]);
        if (!branch_w_ggml) {
            WHISPER_LOG_ERROR("Layer 2: Failed to convert Res2Net branch %d weight\n", b);
            ggml_free(encoder->ctx);
            free(encoder);
            return NULL;
        }

        // Cumulative: input = chunk[i] + y[i-1] for i >= 2
        struct ggml_tensor * conv_input_l2 = layer2_res2net_splits[i];
        if (i >= 2) {
            conv_input_l2 = ggml_add(encoder->ctx, conv_input_l2, layer2_res2net_branches[i - 1]);
        }

        // Conv1d(128→128, k=3, dilation=3, padding=3)
        struct ggml_tensor * branch_conv = ggml_conv_1d(encoder->ctx,
            branch_w_ggml, conv_input_l2, 1, 3, 3);

        branch_conv = ensure_4d_from_conv1d(encoder->ctx, branch_conv);

        struct ggml_tensor * branch_b_reshaped = ggml_reshape_3d(encoder->ctx,
            layer2_res2net_b[b], 1, 128, 1);
        branch_conv = ggml_add(encoder->ctx, branch_conv, branch_b_reshaped);
        branch_conv = ggml_relu(encoder->ctx, branch_conv);

        // Layer 2 Res2Net branch runtime BatchNorm
        {
            char bn_name[256];
            snprintf(bn_name, sizeof(bn_name), "mods.embedding_model.blocks.2.res2net_block.blocks.%d.norm.norm.running_mean", b);
            struct ggml_tensor * bn_mean = whisper_speaker_find_tensor(encoder->model, bn_name);

            snprintf(bn_name, sizeof(bn_name), "mods.embedding_model.blocks.2.res2net_block.blocks.%d.norm.norm.running_var", b);
            struct ggml_tensor * bn_var = whisper_speaker_find_tensor(encoder->model, bn_name);

            snprintf(bn_name, sizeof(bn_name), "mods.embedding_model.blocks.2.res2net_block.blocks.%d.norm.norm.weight", b);
            struct ggml_tensor * bn_gamma = whisper_speaker_find_tensor(encoder->model, bn_name);

            snprintf(bn_name, sizeof(bn_name), "mods.embedding_model.blocks.2.res2net_block.blocks.%d.norm.norm.bias", b);
            struct ggml_tensor * bn_beta = whisper_speaker_find_tensor(encoder->model, bn_name);

            if (bn_mean && bn_var && bn_gamma && bn_beta) {
                struct ggml_tensor * bn_scale = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, 128);
                struct ggml_tensor * bn_offset = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, 128);

                    precompute_bn_params(bn_mean, bn_var, bn_gamma, bn_beta, bn_scale, bn_offset);

                branch_conv = apply_runtime_bn(encoder->ctx, branch_conv, bn_scale, bn_offset);
            }
        }

        layer2_res2net_branches[i] = ggml_reshape_2d(encoder->ctx, branch_conv,
            branch_conv->ne[0], branch_conv->ne[1]);
    }

    // Step 3: Concatenate 8 branches back to [T, 1024]
    struct ggml_tensor * layer2_res2net_concat = layer2_res2net_branches[0];
    for (int g = 1; g < 8; g++) {
        layer2_res2net_concat = ggml_concat(encoder->ctx, layer2_res2net_concat,
            layer2_res2net_branches[g], 1);
        if (!layer2_res2net_concat) {
            WHISPER_LOG_ERROR("Layer 2: Failed to concatenate Res2Net branch %d\n", g);
            ggml_free(encoder->ctx);
            free(encoder);
            return NULL;
        }
    }

    if (layer2_res2net_concat->ne[1] != 1024) {
        WHISPER_LOG_ERROR("Layer 2 Res2Net: ERROR - concat output is %lld, expected 1024\n",
            layer2_res2net_concat->ne[1]);
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    // TDNN2

    // TDNN2: [T, 1024] → [T, 1024]
    struct ggml_tensor * layer2_tdnn2_w_ggml = ggml_conv_weight_f32_to_f16(encoder->ctx, layer2_tdnn2_w);
    if (!layer2_tdnn2_w_ggml) {
        WHISPER_LOG_ERROR("Layer 2: Failed to convert TDNN2 weight layout\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }
    struct ggml_tensor * layer2_tdnn2_out = ggml_conv_1d(encoder->ctx, layer2_tdnn2_w_ggml, layer2_res2net_concat, 1, 0, 1);
    layer2_tdnn2_out = ggml_reshape_4d(encoder->ctx, layer2_tdnn2_out, layer2_tdnn2_out->ne[0], layer2_tdnn2_out->ne[1], layer2_tdnn2_out->ne[2], 1);
    struct ggml_tensor * layer2_tdnn2_b_reshaped = ggml_reshape_3d(encoder->ctx, layer2_tdnn2_b, 1, 1024, 1);
    layer2_tdnn2_out = ggml_add(encoder->ctx, layer2_tdnn2_out, layer2_tdnn2_b_reshaped);
    layer2_tdnn2_out = ggml_relu(encoder->ctx, layer2_tdnn2_out);

    // Layer 2 TDNN2: Runtime BatchNorm
    {
        struct ggml_tensor * bn2_tdnn2_mean = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.2.tdnn2.norm.norm.running_mean");
        struct ggml_tensor * bn2_tdnn2_var = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.2.tdnn2.norm.norm.running_var");
        struct ggml_tensor * bn2_tdnn2_gamma = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.2.tdnn2.norm.norm.weight");
        struct ggml_tensor * bn2_tdnn2_beta = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.2.tdnn2.norm.norm.bias");

        if (bn2_tdnn2_mean && bn2_tdnn2_var && bn2_tdnn2_gamma && bn2_tdnn2_beta) {
            int32_t bn2_channels = layer2_tdnn2_out->ne[1];
            struct ggml_tensor * bn2_scale = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, bn2_channels);
            struct ggml_tensor * bn2_offset = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, bn2_channels);

                precompute_bn_params(bn2_tdnn2_mean, bn2_tdnn2_var, bn2_tdnn2_gamma, bn2_tdnn2_beta, bn2_scale, bn2_offset);

            layer2_tdnn2_out = apply_runtime_bn(encoder->ctx, layer2_tdnn2_out, bn2_scale, bn2_offset);
        }
    }

    // SE Block: GlobalAvgPool → FC1 → ReLU → FC2 → Sigmoid → Scale

    struct ggml_tensor * layer2_se_gap = ggml_pool_1d(encoder->ctx,
        layer2_tdnn2_out,
        GGML_OP_POOL_AVG,
        (int)layer2_tdnn2_out->ne[0],
        (int)layer2_tdnn2_out->ne[0],
        0);

    struct ggml_tensor * layer2_se_gap_1d = ggml_reshape_1d(encoder->ctx, layer2_se_gap, 1024);

    struct ggml_tensor * layer2_se_fc1_w_ggml = ggml_conv_weight_f32_to_f16(
        encoder->ctx, layer2_se_fc1_w);
    if (!layer2_se_fc1_w_ggml) {
        WHISPER_LOG_ERROR("Layer 2 SE: Failed to convert FC1 weight\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    struct ggml_tensor * layer2_se_fc1 = ggml_mul_mat(encoder->ctx,
        ggml_reshape_2d(encoder->ctx, layer2_se_fc1_w_ggml, 1024, 128),
        ggml_reshape_2d(encoder->ctx, layer2_se_gap_1d, 1024, 1));
    layer2_se_fc1 = ggml_reshape_1d(encoder->ctx, layer2_se_fc1, 128);
    layer2_se_fc1 = ggml_add(encoder->ctx, layer2_se_fc1, layer2_se_fc1_b);
    layer2_se_fc1 = ggml_relu(encoder->ctx, layer2_se_fc1);

    struct ggml_tensor * layer2_se_fc2_w_ggml = ggml_conv_weight_f32_to_f16(
        encoder->ctx, layer2_se_fc2_w);
    if (!layer2_se_fc2_w_ggml) {
        WHISPER_LOG_ERROR("Layer 2 SE: Failed to convert FC2 weight\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    struct ggml_tensor * layer2_se_fc2 = ggml_mul_mat(encoder->ctx,
        ggml_reshape_2d(encoder->ctx, layer2_se_fc2_w_ggml, 128, 1024),
        ggml_reshape_2d(encoder->ctx, layer2_se_fc1, 128, 1));
    layer2_se_fc2 = ggml_reshape_1d(encoder->ctx, layer2_se_fc2, 1024);
    layer2_se_fc2 = ggml_add(encoder->ctx, layer2_se_fc2, layer2_se_fc2_b);
    struct ggml_tensor * layer2_se_gates = ggml_sigmoid(encoder->ctx, layer2_se_fc2);

    struct ggml_tensor * layer2_se_gates_reshaped = ggml_reshape_3d(encoder->ctx,
        layer2_se_gates, 1, 1024, 1);

    struct ggml_tensor * layer2_se_out = ggml_mul(encoder->ctx,
        layer2_tdnn2_out, layer2_se_gates_reshaped);

    // Residual connection
    cur = ggml_add(encoder->ctx, layer2_se_out, layer2_input);

    struct ggml_tensor * layer2_out = cur;  // [n_frames, 1024]

    // Layer 3
    struct ggml_tensor * layer3_tdnn1_w = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.3.tdnn1.conv.conv.weight");
    struct ggml_tensor * layer3_tdnn1_b = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.3.tdnn1.conv.conv.bias");

    // Res2Net branches (7 branches)
    struct ggml_tensor * layer3_res2net[7] = {NULL};
    struct ggml_tensor * layer3_res2net_b[7] = {NULL};
    const char * res2net_names_3[] = {
        "mods.embedding_model.blocks.3.res2net_block.blocks.0.conv.conv.weight",
        "mods.embedding_model.blocks.3.res2net_block.blocks.1.conv.conv.weight",
        "mods.embedding_model.blocks.3.res2net_block.blocks.2.conv.conv.weight",
        "mods.embedding_model.blocks.3.res2net_block.blocks.3.conv.conv.weight",
        "mods.embedding_model.blocks.3.res2net_block.blocks.4.conv.conv.weight",
        "mods.embedding_model.blocks.3.res2net_block.blocks.5.conv.conv.weight",
        "mods.embedding_model.blocks.3.res2net_block.blocks.6.conv.conv.weight",
    };
    const char * res2net_bias_names_3[] = {
        "mods.embedding_model.blocks.3.res2net_block.blocks.0.conv.conv.bias",
        "mods.embedding_model.blocks.3.res2net_block.blocks.1.conv.conv.bias",
        "mods.embedding_model.blocks.3.res2net_block.blocks.2.conv.conv.bias",
        "mods.embedding_model.blocks.3.res2net_block.blocks.3.conv.conv.bias",
        "mods.embedding_model.blocks.3.res2net_block.blocks.4.conv.conv.bias",
        "mods.embedding_model.blocks.3.res2net_block.blocks.5.conv.conv.bias",
        "mods.embedding_model.blocks.3.res2net_block.blocks.6.conv.conv.bias",
    };

    for (int i = 0; i < 7; i++) {
        layer3_res2net[i] = whisper_speaker_find_tensor(encoder->model, res2net_names_3[i]);
        layer3_res2net_b[i] = whisper_speaker_find_tensor(encoder->model, res2net_bias_names_3[i]);
        if (!layer3_res2net[i] || !layer3_res2net_b[i]) {
            WHISPER_LOG_ERROR("Layer 3: Failed to load Res2Net branch %d\n", i);
            ggml_free(encoder->ctx);
            free(encoder);
            return NULL;
        }
    }

    // SE block weights
    struct ggml_tensor * layer3_se_fc1_w = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.3.se_block.conv1.conv.weight");
    struct ggml_tensor * layer3_se_fc1_b = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.3.se_block.conv1.conv.bias");
    struct ggml_tensor * layer3_se_fc2_w = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.3.se_block.conv2.conv.weight");
    struct ggml_tensor * layer3_se_fc2_b = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.3.se_block.conv2.conv.bias");

    // TDNN2
    struct ggml_tensor * layer3_tdnn2_w = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.3.tdnn2.conv.conv.weight");
    struct ggml_tensor * layer3_tdnn2_b = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.3.tdnn2.conv.conv.bias");

    if (!layer3_tdnn1_w || !layer3_tdnn1_b || !layer3_se_fc1_w || !layer3_se_fc1_b ||
        !layer3_se_fc2_w || !layer3_se_fc2_b || !layer3_tdnn2_w || !layer3_tdnn2_b) {
        WHISPER_LOG_ERROR("Layer 3: Failed to load SE block weights\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    // Current input
    struct ggml_tensor * layer3_input = cur;  // [T, 1024]

    // TDNN1: [T, 1024] → [T, 1024]
    struct ggml_tensor * layer3_tdnn1_w_ggml = ggml_conv_weight_f32_to_f16(encoder->ctx, layer3_tdnn1_w);
    if (!layer3_tdnn1_w_ggml) {
        WHISPER_LOG_ERROR("Layer 3: Failed to convert TDNN1 weight layout\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }
    struct ggml_tensor * layer3_tdnn1_out = ggml_conv_1d(encoder->ctx, layer3_tdnn1_w_ggml, layer3_input, 1, 0, 1);
    layer3_tdnn1_out = ggml_reshape_4d(encoder->ctx, layer3_tdnn1_out, layer3_tdnn1_out->ne[0], layer3_tdnn1_out->ne[1], layer3_tdnn1_out->ne[2], 1);
    struct ggml_tensor * layer3_tdnn1_b_reshaped = ggml_reshape_3d(encoder->ctx, layer3_tdnn1_b, 1, 1024, 1);
    layer3_tdnn1_out = ggml_add(encoder->ctx, layer3_tdnn1_out, layer3_tdnn1_b_reshaped);
    layer3_tdnn1_out = ggml_relu(encoder->ctx, layer3_tdnn1_out);

    // Layer 3 TDNN1: Runtime BatchNorm
    {
        struct ggml_tensor * bn3_tdnn1_mean = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.3.tdnn1.norm.norm.running_mean");
        struct ggml_tensor * bn3_tdnn1_var = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.3.tdnn1.norm.norm.running_var");
        struct ggml_tensor * bn3_tdnn1_gamma = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.3.tdnn1.norm.norm.weight");
        struct ggml_tensor * bn3_tdnn1_beta = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.3.tdnn1.norm.norm.bias");

        if (bn3_tdnn1_mean && bn3_tdnn1_var && bn3_tdnn1_gamma && bn3_tdnn1_beta) {
            int32_t bn3_channels = layer3_tdnn1_out->ne[1];
            struct ggml_tensor * bn3_scale = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, bn3_channels);
            struct ggml_tensor * bn3_offset = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, bn3_channels);

                precompute_bn_params(bn3_tdnn1_mean, bn3_tdnn1_var, bn3_tdnn1_gamma, bn3_tdnn1_beta, bn3_scale, bn3_offset);

            layer3_tdnn1_out = apply_runtime_bn(encoder->ctx, layer3_tdnn1_out, bn3_scale, bn3_offset);
        }
    }

    // Res2Net: Apply 7 branches with different dilations
    // Res2Net: Split [T, 1024] into 8 groups of [T, 128]
    struct ggml_tensor * layer3_res2net_splits[8];
    size_t chan_stride_l3 = layer3_tdnn1_out->nb[1];

    for (int g = 0; g < 8; g++) {
        size_t offset = g * group_channels * chan_stride_l3;
        layer3_res2net_splits[g] = ggml_view_2d(encoder->ctx,
            layer3_tdnn1_out,
            layer3_tdnn1_out->ne[0],
            group_channels,
            chan_stride_l3,
            offset);
        ggml_set_name(layer3_res2net_splits[g], "layer3_res2net_split");
    }

    // Res2Net: chunk[0]=identity, chunk[i]=conv(chunk[i]+y[i-1]) for i>=2
    // All 7 blocks use dilation=4 for Layer 3
    struct ggml_tensor * layer3_res2net_branches[8];

    // Chunk 0: identity
    layer3_res2net_branches[0] = layer3_res2net_splits[0];

    // Chunks 1-7: apply blocks[0..6]
    for (int i = 1; i < 8; i++) {
        int b = i - 1;  // block index

        struct ggml_tensor * branch_w_ggml = ggml_conv_weight_f32_to_f16(
            encoder->ctx, layer3_res2net[b]);
        if (!branch_w_ggml) {
            WHISPER_LOG_ERROR("Layer 3: Failed to convert Res2Net branch %d weight\n", b);
            ggml_free(encoder->ctx);
            free(encoder);
            return NULL;
        }

        // Cumulative: input = chunk[i] + y[i-1] for i >= 2
        struct ggml_tensor * conv_input_l3 = layer3_res2net_splits[i];
        if (i >= 2) {
            conv_input_l3 = ggml_add(encoder->ctx, conv_input_l3, layer3_res2net_branches[i - 1]);
        }

        // Conv1d(128→128, k=3, dilation=4, padding=4)
        struct ggml_tensor * branch_conv = ggml_conv_1d(encoder->ctx,
            branch_w_ggml, conv_input_l3, 1, 4, 4);

        branch_conv = ensure_4d_from_conv1d(encoder->ctx, branch_conv);

        struct ggml_tensor * branch_b_reshaped = ggml_reshape_3d(encoder->ctx,
            layer3_res2net_b[b], 1, 128, 1);
        branch_conv = ggml_add(encoder->ctx, branch_conv, branch_b_reshaped);
        branch_conv = ggml_relu(encoder->ctx, branch_conv);

        // Layer 3 Res2Net branch runtime BatchNorm
        {
            char bn_name[256];
            snprintf(bn_name, sizeof(bn_name), "mods.embedding_model.blocks.3.res2net_block.blocks.%d.norm.norm.running_mean", b);
            struct ggml_tensor * bn_mean = whisper_speaker_find_tensor(encoder->model, bn_name);

            snprintf(bn_name, sizeof(bn_name), "mods.embedding_model.blocks.3.res2net_block.blocks.%d.norm.norm.running_var", b);
            struct ggml_tensor * bn_var = whisper_speaker_find_tensor(encoder->model, bn_name);

            snprintf(bn_name, sizeof(bn_name), "mods.embedding_model.blocks.3.res2net_block.blocks.%d.norm.norm.weight", b);
            struct ggml_tensor * bn_gamma = whisper_speaker_find_tensor(encoder->model, bn_name);

            snprintf(bn_name, sizeof(bn_name), "mods.embedding_model.blocks.3.res2net_block.blocks.%d.norm.norm.bias", b);
            struct ggml_tensor * bn_beta = whisper_speaker_find_tensor(encoder->model, bn_name);

            if (bn_mean && bn_var && bn_gamma && bn_beta) {
                struct ggml_tensor * bn_scale = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, 128);
                struct ggml_tensor * bn_offset = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, 128);

                    precompute_bn_params(bn_mean, bn_var, bn_gamma, bn_beta, bn_scale, bn_offset);

                branch_conv = apply_runtime_bn(encoder->ctx, branch_conv, bn_scale, bn_offset);
            }
        }

        layer3_res2net_branches[i] = ggml_reshape_2d(encoder->ctx, branch_conv,
            branch_conv->ne[0], branch_conv->ne[1]);
    }

    // Step 3: Concatenate 8 branches back to [T, 1024]
    struct ggml_tensor * layer3_res2net_concat = layer3_res2net_branches[0];
    for (int g = 1; g < 8; g++) {
        layer3_res2net_concat = ggml_concat(encoder->ctx, layer3_res2net_concat,
            layer3_res2net_branches[g], 1);
        if (!layer3_res2net_concat) {
            WHISPER_LOG_ERROR("Layer 3: Failed to concatenate Res2Net branch %d\n", g);
            ggml_free(encoder->ctx);
            free(encoder);
            return NULL;
        }
    }

    if (layer3_res2net_concat->ne[1] != 1024) {
        WHISPER_LOG_ERROR("Layer 3 Res2Net: ERROR - concat output is %lld, expected 1024\n",
            layer3_res2net_concat->ne[1]);
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    // TDNN2

    // TDNN2: [T, 1024] → [T, 1024]
    struct ggml_tensor * layer3_tdnn2_w_ggml = ggml_conv_weight_f32_to_f16(encoder->ctx, layer3_tdnn2_w);
    if (!layer3_tdnn2_w_ggml) {
        WHISPER_LOG_ERROR("Layer 3: Failed to convert TDNN2 weight layout\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }
    struct ggml_tensor * layer3_tdnn2_out = ggml_conv_1d(encoder->ctx, layer3_tdnn2_w_ggml, layer3_res2net_concat, 1, 0, 1);
    layer3_tdnn2_out = ggml_reshape_4d(encoder->ctx, layer3_tdnn2_out, layer3_tdnn2_out->ne[0], layer3_tdnn2_out->ne[1], layer3_tdnn2_out->ne[2], 1);
    struct ggml_tensor * layer3_tdnn2_b_reshaped = ggml_reshape_3d(encoder->ctx, layer3_tdnn2_b, 1, 1024, 1);
    layer3_tdnn2_out = ggml_add(encoder->ctx, layer3_tdnn2_out, layer3_tdnn2_b_reshaped);
    layer3_tdnn2_out = ggml_relu(encoder->ctx, layer3_tdnn2_out);

    // Layer 3 TDNN2: Runtime BatchNorm
    {
        struct ggml_tensor * bn3_tdnn2_mean = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.3.tdnn2.norm.norm.running_mean");
        struct ggml_tensor * bn3_tdnn2_var = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.3.tdnn2.norm.norm.running_var");
        struct ggml_tensor * bn3_tdnn2_gamma = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.3.tdnn2.norm.norm.weight");
        struct ggml_tensor * bn3_tdnn2_beta = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.blocks.3.tdnn2.norm.norm.bias");

        if (bn3_tdnn2_mean && bn3_tdnn2_var && bn3_tdnn2_gamma && bn3_tdnn2_beta) {
            int32_t bn3_channels = layer3_tdnn2_out->ne[1];
            struct ggml_tensor * bn3_scale = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, bn3_channels);
            struct ggml_tensor * bn3_offset = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, bn3_channels);

                precompute_bn_params(bn3_tdnn2_mean, bn3_tdnn2_var, bn3_tdnn2_gamma, bn3_tdnn2_beta, bn3_scale, bn3_offset);

            layer3_tdnn2_out = apply_runtime_bn(encoder->ctx, layer3_tdnn2_out, bn3_scale, bn3_offset);
        }
    }

    // SE Block: GlobalAvgPool → FC1 → ReLU → FC2 → Sigmoid → Scale

    struct ggml_tensor * layer3_se_gap = ggml_pool_1d(encoder->ctx,
        layer3_tdnn2_out,
        GGML_OP_POOL_AVG,
        (int)layer3_tdnn2_out->ne[0],
        (int)layer3_tdnn2_out->ne[0],
        0);

    struct ggml_tensor * layer3_se_gap_1d = ggml_reshape_1d(encoder->ctx, layer3_se_gap, 1024);

    struct ggml_tensor * layer3_se_fc1_w_ggml = ggml_conv_weight_f32_to_f16(
        encoder->ctx, layer3_se_fc1_w);
    if (!layer3_se_fc1_w_ggml) {
        WHISPER_LOG_ERROR("Layer 3 SE: Failed to convert FC1 weight\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    struct ggml_tensor * layer3_se_fc1 = ggml_mul_mat(encoder->ctx,
        ggml_reshape_2d(encoder->ctx, layer3_se_fc1_w_ggml, 1024, 128),
        ggml_reshape_2d(encoder->ctx, layer3_se_gap_1d, 1024, 1));
    layer3_se_fc1 = ggml_reshape_1d(encoder->ctx, layer3_se_fc1, 128);
    layer3_se_fc1 = ggml_add(encoder->ctx, layer3_se_fc1, layer3_se_fc1_b);
    layer3_se_fc1 = ggml_relu(encoder->ctx, layer3_se_fc1);

    struct ggml_tensor * layer3_se_fc2_w_ggml = ggml_conv_weight_f32_to_f16(
        encoder->ctx, layer3_se_fc2_w);
    if (!layer3_se_fc2_w_ggml) {
        WHISPER_LOG_ERROR("Layer 3 SE: Failed to convert FC2 weight\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    struct ggml_tensor * layer3_se_fc2 = ggml_mul_mat(encoder->ctx,
        ggml_reshape_2d(encoder->ctx, layer3_se_fc2_w_ggml, 128, 1024),
        ggml_reshape_2d(encoder->ctx, layer3_se_fc1, 128, 1));
    layer3_se_fc2 = ggml_reshape_1d(encoder->ctx, layer3_se_fc2, 1024);
    layer3_se_fc2 = ggml_add(encoder->ctx, layer3_se_fc2, layer3_se_fc2_b);
    struct ggml_tensor * layer3_se_gates = ggml_sigmoid(encoder->ctx, layer3_se_fc2);

    struct ggml_tensor * layer3_se_gates_reshaped = ggml_reshape_3d(encoder->ctx,
        layer3_se_gates, 1, 1024, 1);

    struct ggml_tensor * layer3_se_out = ggml_mul(encoder->ctx,
        layer3_tdnn2_out, layer3_se_gates_reshaped);

    // Residual connection
    cur = ggml_add(encoder->ctx, layer3_se_out, layer3_input);

    struct ggml_tensor * layer3_out = cur;  // [n_frames, 1024]

    // Layer 4: Multi-layer Feature Aggregation (MFA)
    struct ggml_tensor * mfa_w = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.mfa.conv.conv.weight");
    struct ggml_tensor * mfa_b = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.mfa.conv.conv.bias");

    if (!mfa_w || !mfa_b) {
        WHISPER_LOG_ERROR("Layer 4 (MFA): Failed to load weights\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    // Concatenate layers 1-3: [T, 1024] × 3 → [T, 3072]

    struct ggml_tensor * mfa_input = ggml_concat(encoder->ctx, layer1_out, layer2_out, 1);
    if (!mfa_input) {
        WHISPER_LOG_ERROR("Layer 4 (MFA): Failed to concatenate layers 1-2\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    mfa_input = ggml_concat(encoder->ctx, mfa_input, layer3_out, 1);
    if (!mfa_input) {
        WHISPER_LOG_ERROR("Layer 4 (MFA): Failed to concatenate layers 1-3\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    // Verify dimension
    if (mfa_input->ne[1] != 3072) {
        WHISPER_LOG_WARN("Layer 4 (MFA): WARNING - input dimension is %lld, expected 3072\n", mfa_input->ne[1]);
    }

    // Apply MFA Conv1d(3072→3072, k=1, padding=0) + BN
    struct ggml_tensor * mfa_w_ggml = ggml_conv_weight_f32_to_f16(encoder->ctx, mfa_w);
    if (!mfa_w_ggml) {
        WHISPER_LOG_ERROR("Layer 4 (MFA): Failed to convert weight layout\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    // MFA conv: [T, 3072] @ [1, 3072, 3072] → [T, 3072]
    struct ggml_tensor * mfa_conv_out = ggml_conv_1d(encoder->ctx, mfa_w_ggml, mfa_input, 1, 0, 1);
    mfa_conv_out = ggml_reshape_4d(encoder->ctx, mfa_conv_out, mfa_conv_out->ne[0], mfa_conv_out->ne[1], mfa_conv_out->ne[2], 1);

    // Add bias
    struct ggml_tensor * mfa_b_reshaped = ggml_reshape_3d(encoder->ctx, mfa_b, 1, 3072, 1);
    mfa_conv_out = ggml_add(encoder->ctx, mfa_conv_out, mfa_b_reshaped);

    // ReLU
    mfa_conv_out = ggml_relu(encoder->ctx, mfa_conv_out);

    // Runtime BN for MFA
    struct ggml_tensor * bn_mfa_mean = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.mfa.norm.norm.running_mean");
    struct ggml_tensor * bn_mfa_var = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.mfa.norm.norm.running_var");
    struct ggml_tensor * bn_mfa_gamma = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.mfa.norm.norm.weight");
    struct ggml_tensor * bn_mfa_beta = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.mfa.norm.norm.bias");

    if (bn_mfa_mean && bn_mfa_var && bn_mfa_gamma && bn_mfa_beta) {
        struct ggml_tensor * bn_mfa_scale = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, 3072);
        struct ggml_tensor * bn_mfa_offset = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, 3072);

            precompute_bn_params(bn_mfa_mean, bn_mfa_var, bn_mfa_gamma, bn_mfa_beta, bn_mfa_scale, bn_mfa_offset);

        mfa_conv_out = apply_runtime_bn(encoder->ctx, mfa_conv_out, bn_mfa_scale, bn_mfa_offset);
    } else {
        WHISPER_LOG_WARN("Layer 4 (MFA): Missing BN tensors, skipping BN\n");
    }

    struct ggml_tensor * mfa_out = mfa_conv_out;

    // Final check
    if (mfa_out->ne[1] != 3072) {
        WHISPER_LOG_ERROR("Layer 4 (MFA): ERROR - output dimension is %lld, expected 3072! Aborting.\n", mfa_out->ne[1]);
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    cur = mfa_out;  // [T, 3072]

    // Layer 5: Attentive Statistical Pooling (ASP)

    struct ggml_tensor * x = cur;
    int32_t n_features = x->ne[1];
    if (n_features != 3072) {
        WHISPER_LOG_ERROR("Layer 5: input dimension is %d, expected 3072\n", n_features);
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    int32_t T = x->ne[0];
    int32_t C = x->ne[1];

    // Global statistics for attention input
    struct ggml_tensor * global_mean = ggml_pool_1d(encoder->ctx, x, GGML_OP_POOL_AVG, T, T, 0);

    struct ggml_tensor * x_minus_mean = ggml_sub(encoder->ctx, x, global_mean);
    struct ggml_tensor * sq_dev = ggml_mul(encoder->ctx, x_minus_mean, x_minus_mean);
    struct ggml_tensor * var = ggml_pool_1d(encoder->ctx, sq_dev, GGML_OP_POOL_AVG, T, T, 0);
    struct ggml_tensor * global_std = ggml_sqrt(encoder->ctx, var);

    // Repeat stats for concatenation
    struct ggml_tensor * mean_repeated = ggml_repeat(encoder->ctx, global_mean, x);
    struct ggml_tensor * std_repeated = ggml_repeat(encoder->ctx, global_std, x);

    struct ggml_tensor * att_input = ggml_concat(encoder->ctx, x, mean_repeated, 1);
    att_input = ggml_concat(encoder->ctx, att_input, std_repeated, 1);

    // TDNN projection → attention weights
    struct ggml_tensor * tdnn_w = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.asp.tdnn.conv.conv.weight");
    struct ggml_tensor * tdnn_b = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.asp.tdnn.conv.conv.bias");

    if (!tdnn_w || !tdnn_b) {
        WHISPER_LOG_ERROR("Layer 5 ASP: Missing TDNN tensors\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    struct ggml_tensor * tdnn_w_ggml = ggml_conv_weight_f32_to_f16(encoder->ctx, tdnn_w);
    if (!tdnn_w_ggml) {
        WHISPER_LOG_ERROR("Layer 5 ASP: Failed to convert TDNN weight layout\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    struct ggml_tensor * tdnn_conv = ggml_conv_1d(encoder->ctx, tdnn_w_ggml, att_input, 1, 0, 1);
    tdnn_conv = ensure_4d_from_conv1d(encoder->ctx, tdnn_conv);
    struct ggml_tensor * tdnn_b_reshaped = ggml_reshape_3d(encoder->ctx, tdnn_b, 1, 128, 1);
    tdnn_conv = ggml_add(encoder->ctx, tdnn_conv, tdnn_b_reshaped);
    tdnn_conv = ggml_relu(encoder->ctx, tdnn_conv);

    {
        struct ggml_tensor * bn_mean = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.asp.tdnn.norm.norm.running_mean");
        struct ggml_tensor * bn_var  = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.asp.tdnn.norm.norm.running_var");
        struct ggml_tensor * bn_g    = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.asp.tdnn.norm.norm.weight");
        struct ggml_tensor * bn_b    = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.asp.tdnn.norm.norm.bias");
        if (bn_mean && bn_var && bn_g && bn_b) {
            struct ggml_tensor * s = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, 128);
            struct ggml_tensor * o = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, 128);
            precompute_bn_params(bn_mean, bn_var, bn_g, bn_b, s, o);
            tdnn_conv = apply_runtime_bn(encoder->ctx, tdnn_conv, s, o);
        }
    }

    // Attention conv
    struct ggml_tensor * att_conv_w = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.asp.conv.conv.weight");
    struct ggml_tensor * att_conv_b = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.asp.conv.conv.bias");
    if (!att_conv_w || !att_conv_b) {
        WHISPER_LOG_ERROR("Layer 5 ASP: Missing attention conv tensors\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    struct ggml_tensor * att_conv_w_f16 = ggml_conv_weight_f32_to_f16(encoder->ctx, att_conv_w);
    struct ggml_tensor * att_e = ggml_conv_1d(encoder->ctx, att_conv_w_f16, tdnn_conv, 1, 0, 1);
    att_e = ensure_4d_from_conv1d(encoder->ctx, att_e);
    struct ggml_tensor * att_conv_b_r = ggml_reshape_3d(encoder->ctx, att_conv_b, 1, C, 1);
    att_e = ggml_add(encoder->ctx, att_e, att_conv_b_r);

    // Softmax over time
    struct ggml_tensor * att_e_2d = ggml_reshape_2d(encoder->ctx, att_e, T, C);
    struct ggml_tensor * alpha = ggml_soft_max(encoder->ctx, att_e_2d);

    // Weighted aggregation
    struct ggml_tensor * x_2d = ggml_reshape_2d(encoder->ctx, x, T, C);

    struct ggml_tensor * x_weighted = ggml_mul(encoder->ctx, x_2d, alpha);
    struct ggml_tensor * weighted_mean_2d = ggml_sum_rows(encoder->ctx, x_weighted);
    struct ggml_tensor * x_minus_wmean = ggml_sub(encoder->ctx, x_2d, weighted_mean_2d);
    struct ggml_tensor * sq_diff = ggml_mul(encoder->ctx, x_minus_wmean, x_minus_wmean);
    struct ggml_tensor * w_sq_diff = ggml_mul(encoder->ctx, sq_diff, alpha);
    struct ggml_tensor * weighted_var_2d = ggml_sum_rows(encoder->ctx, w_sq_diff);
    struct ggml_tensor * eps_tensor = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, 1);
    ((float *)eps_tensor->data)[0] = 1e-5f;
    weighted_var_2d = ggml_add(encoder->ctx, weighted_var_2d, eps_tensor);
    struct ggml_tensor * weighted_std_2d = ggml_sqrt(encoder->ctx, weighted_var_2d);

    // Output: [mean; std] → [6144]
    weighted_mean_2d = ggml_reshape_4d(encoder->ctx, weighted_mean_2d, 1, C, 1, 1);
    weighted_std_2d = ggml_reshape_4d(encoder->ctx, weighted_std_2d, 1, C, 1, 1);

    struct ggml_tensor * asp_output = ggml_concat(encoder->ctx, weighted_mean_2d, weighted_std_2d, 1);
    asp_output = ggml_reshape_1d(encoder->ctx, asp_output, 2*C);  // [6144]

    if (asp_output->ne[0] != 6144) {
        WHISPER_LOG_ERROR("Layer 5 ASP: Output dimension mismatch: %lld != 6144\n", asp_output->ne[0]);
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    cur = asp_output;

    // ASP BatchNorm
    {
        struct ggml_tensor * bn_mean = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.asp_bn.norm.running_mean");
        struct ggml_tensor * bn_var  = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.asp_bn.norm.running_var");
        struct ggml_tensor * bn_g    = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.asp_bn.norm.weight");
        struct ggml_tensor * bn_b    = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.asp_bn.norm.bias");
        if (bn_mean && bn_var && bn_g && bn_b) {
            struct ggml_tensor * s = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, 6144);
            struct ggml_tensor * o = ggml_new_tensor_1d(encoder->ctx, GGML_TYPE_F32, 6144);
            precompute_bn_params(bn_mean, bn_var, bn_g, bn_b, s, o);
            cur = ggml_mul(encoder->ctx, cur, s);
            cur = ggml_add(encoder->ctx, cur, o);
        } else {
            WHISPER_LOG_WARN("ASP BN: Missing tensors, skipping\n");
        }
    }

    // Layer 6: Final FC [6144] → [192]

    struct ggml_tensor * embedding_w = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.fc.conv.weight");
    struct ggml_tensor * embedding_b = whisper_speaker_find_tensor(encoder->model, "mods.embedding_model.fc.conv.bias");

    if (!embedding_w || !embedding_b) {
        WHISPER_LOG_ERROR("Layer 6 (Final FC): Failed to load weights\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }


    if (cur->ne[0] != 6144) {
        WHISPER_LOG_ERROR("Layer 6 FC: ERROR - input dimension is %lld, expected 6144!\n", cur->ne[0]);
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    struct ggml_tensor * embedding_w_ggml = ggml_conv_weight_f32_to_f16(encoder->ctx, embedding_w);
    if (!embedding_w_ggml) {
        WHISPER_LOG_ERROR("Layer 6: Failed to convert weight layout\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    struct ggml_tensor * weight_2d = ggml_reshape_2d(encoder->ctx, embedding_w_ggml, 6144, 192);

    struct ggml_tensor * cur_fc_2d = ggml_reshape_2d(encoder->ctx, cur, 6144, 1);
    struct ggml_tensor * embedding = ggml_mul_mat(encoder->ctx, weight_2d, cur_fc_2d);
    embedding = ggml_reshape_1d(encoder->ctx, embedding, 192);
    if (!embedding) {
        WHISPER_LOG_ERROR("Layer 6: Failed to compute matrix multiplication\n");
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    embedding = ggml_add(encoder->ctx, embedding, embedding_b);

    if (embedding->ne[0] != 192) {
        WHISPER_LOG_ERROR("Layer 6 FC: ERROR - output dimension is %lld, expected 192!\n", embedding->ne[0]);
        ggml_free(encoder->ctx);
        free(encoder);
        return NULL;
    }

    ggml_set_name(embedding, "output_embedding");
    ggml_set_output(embedding);
    encoder->output_embedding = embedding;
    ggml_build_forward_expand(encoder->graph, embedding);

    return encoder;
}

// Free encoder context
void whisper_speaker_encoder_free(struct whisper_speaker_encoder * encoder) {
    if (!encoder) {
        return;
    }

    if (encoder->ctx) {
        ggml_free(encoder->ctx);
    }

    free(encoder);
}

// Run forward pass: mel [T, 80] → embedding [192]
bool whisper_speaker_encoder_compute(
    struct whisper_speaker_encoder * encoder,
    const float * mel,
    float * embedding) {

    if (!encoder || !mel || !embedding) {
        WHISPER_LOG_ERROR("encoder_compute: invalid arguments\n");
        return false;
    }

    // Validate input tensor dimensions (ne[0]=n_mels, ne[1]=n_frames)
    if (encoder->input_mel->ne[0] != encoder->n_mels ||
        encoder->input_mel->ne[1] != encoder->n_frames) {
        WHISPER_LOG_ERROR("encoder_compute: input shape mismatch\n");
        return false;
    }

    // Copy mel data into input tensor
    float * input_data = (float *)encoder->input_mel->data;
    int mel_size = encoder->n_frames * encoder->n_mels;
    memcpy(input_data, mel, mel_size * sizeof(float));

    // Mark input/output tensors
    ggml_set_input(encoder->input_mel);
    ggml_set_output(encoder->output_embedding);

    // Execute forward pass using CPU backend
    // Allocate work buffer for graph computation
    struct ggml_cplan plan = ggml_graph_plan(encoder->graph, 4, NULL);  // 4 threads, no custom threadpool
    if (plan.work_size > 0) {
        plan.work_data = (uint8_t *)malloc(plan.work_size);
        if (!plan.work_data) {
            WHISPER_LOG_ERROR("encoder_compute: failed to allocate work buffer (%zu bytes)\n", plan.work_size);
            return false;
        }
    }

    // Execute the graph (forward pass)
    enum ggml_status ret = ggml_graph_compute(encoder->graph, &plan);
    if (plan.work_data) {
        free(plan.work_data);
    }

    if (ret != GGML_STATUS_SUCCESS) {
        WHISPER_LOG_ERROR("encoder_compute: graph compute failed with status %d\n", (int)ret);
        return false;
    }

    // Check for NaN/Inf in output (sanity check on computed values)
    float * output_data = (float *)encoder->output_embedding->data;
    for (int i = 0; i < 192; i++) {
        if (isnan(output_data[i]) || isinf(output_data[i])) {
            WHISPER_LOG_ERROR("encoder_compute: output contains NaN/Inf at index %d\n", i);
            return false;
        }
    }

    // Copy output embedding to caller's buffer
    memcpy(embedding, output_data, 192 * sizeof(float));

    return true;
}

// Agglomerative hierarchical clustering

// Cosine distance (double precision)
static double cosine_distance_f64(const float* a, const float* b, int dim) {
    double dot_product = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (int i = 0; i < dim; i++) {
        double a_f64 = (double)a[i];
        double b_f64 = (double)b[i];
        dot_product += a_f64 * b_f64;
        norm_a += a_f64 * a_f64;
        norm_b += b_f64 * b_f64;
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    // Handle zero-norm case (protect against division by zero)
    if (norm_a < 1e-10 || norm_b < 1e-10) return 1.0;
    return 1.0 - (dot_product / (norm_a * norm_b));
}

double * compute_distance_matrix(
    const float * embeddings,
    int num_segments,
    int embedding_dim
) {
    // Allocate symmetric distance matrix
    double * dist_matrix = (double*)malloc(num_segments * num_segments * sizeof(double));
    if (!dist_matrix) return NULL;

    for (int i = 0; i < num_segments; i++) {
        dist_matrix[i * num_segments + i] = 0.0;  // distance to self is 0
        const float * embedding_i = embeddings + i * embedding_dim;
        for (int j = i + 1; j < num_segments; j++) {
            const float * embedding_j = embeddings + j * embedding_dim;
            double dist = cosine_distance_f64(embedding_i, embedding_j, embedding_dim);
            dist_matrix[i * num_segments + j] = dist;
            dist_matrix[j * num_segments + i] = dist;  // symmetric
        }
    }
    return dist_matrix;
}

struct whisper_clustering_context * whisper_clustering_context_create(int num_segments) {
    if (num_segments <= 0) return NULL;

    struct whisper_clustering_context * ctx =
        (struct whisper_clustering_context*)malloc(sizeof(struct whisper_clustering_context));
    if (!ctx) return NULL;

    ctx->num_segments = num_segments;
    ctx->embedding_dim = 192;  // ECAPA-TDNN output dimension
    ctx->distance_matrix = NULL;
    ctx->speaker_ids = (int*)malloc(num_segments * sizeof(int));
    ctx->num_speakers = 0;

    if (!ctx->speaker_ids) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

void whisper_clustering_context_free(struct whisper_clustering_context * ctx) {
    if (!ctx) return;
    if (ctx->distance_matrix) free(ctx->distance_matrix);
    if (ctx->speaker_ids) free(ctx->speaker_ids);
    free(ctx);
}

int whisper_clustering_cluster(
    struct whisper_clustering_context * ctx,
    const float * embeddings,
    int target_speakers,
    float threshold,
    int linkage_type
) {
    if (!ctx || !embeddings) return -1;
    if (ctx->num_segments <= 0) return -1;

    // Compute distance matrix
    ctx->distance_matrix = compute_distance_matrix(embeddings, ctx->num_segments, ctx->embedding_dim);
    if (!ctx->distance_matrix) {
        WHISPER_LOG_ERROR("clustering_cluster: failed to allocate distance matrix\n");
        return -1;
    }

    int num_segments = ctx->num_segments;
    double * dist = ctx->distance_matrix;

    int * cluster_assignment = (int*)malloc(num_segments * sizeof(int));
    if (!cluster_assignment) {
        free(ctx->distance_matrix);
        ctx->distance_matrix = NULL;
        return -1;
    }

    for (int i = 0; i < num_segments; i++) {
        cluster_assignment[i] = i;
    }

    bool * cluster_active = (bool*)malloc(num_segments * sizeof(bool));
    if (!cluster_active) {
        free(cluster_assignment);
        free(ctx->distance_matrix);
        ctx->distance_matrix = NULL;
        return -1;
    }

    for (int i = 0; i < num_segments; i++) {
        cluster_active[i] = true;
    }

    int num_active_clusters = num_segments;

    while (num_active_clusters > 1) {
        // Find two closest active clusters
        double min_distance = 1e10;
        int merge_cluster1 = -1;
        int merge_cluster2 = -1;

        for (int i = 0; i < num_segments; i++) {
            if (!cluster_active[i]) continue;
            for (int j = i + 1; j < num_segments; j++) {
                if (!cluster_active[j]) continue;
                double d = dist[i * num_segments + j];
                if (d < min_distance) {
                    min_distance = d;
                    merge_cluster1 = i;
                    merge_cluster2 = j;
                }
            }
        }

        if (merge_cluster1 == -1) break;  // No clusters to merge

        if (target_speakers > 0) {
            if (num_active_clusters <= target_speakers) break;
        } else {
            if (min_distance > (double)threshold) break;
        }

        // Merge cluster2 into cluster1
        for (int i = 0; i < num_segments; i++) {
            if (cluster_assignment[i] == merge_cluster2) {
                cluster_assignment[i] = merge_cluster1;
            }
        }
        cluster_active[merge_cluster2] = false;
        num_active_clusters--;

        // Update distances
        for (int k = 0; k < num_segments; k++) {
            if (!cluster_active[k] || k == merge_cluster1) continue;

            double new_distance = 0.0;

            if (linkage_type == WHISPER_LINKAGE_AVERAGE) {
                // Average linkage: mean distance between all pairs
                int pairs_count = 0;
                new_distance = 0.0;
                for (int i = 0; i < num_segments; i++) {
                    if (cluster_assignment[i] != merge_cluster1) continue;
                    for (int j = 0; j < num_segments; j++) {
                        if (cluster_assignment[j] != k) continue;
                        new_distance += dist[i * num_segments + j];
                        pairs_count++;
                    }
                }
                if (pairs_count > 0) {
                    new_distance /= (double)pairs_count;
                }
            } else {  // WHISPER_LINKAGE_COMPLETE
                // Complete linkage: max distance between any pair
                new_distance = 0.0;
                for (int i = 0; i < num_segments; i++) {
                    if (cluster_assignment[i] != merge_cluster1) continue;
                    for (int j = 0; j < num_segments; j++) {
                        if (cluster_assignment[j] != k) continue;
                        double d = dist[i * num_segments + j];
                        if (d > new_distance) {
                            new_distance = d;
                        }
                    }
                }
            }

            dist[merge_cluster1 * num_segments + k] = new_distance;
            dist[k * num_segments + merge_cluster1] = new_distance;
        }
    }

    // Map cluster IDs to 0-based speaker IDs
    int * cluster_to_speaker = (int*)malloc(num_segments * sizeof(int));
    if (!cluster_to_speaker) {
        free(cluster_assignment);
        free(cluster_active);
        if (ctx->distance_matrix) {
            free(ctx->distance_matrix);
            ctx->distance_matrix = NULL;
        }
        return -1;
    }

    for (int i = 0; i < num_segments; i++) {
        cluster_to_speaker[i] = -1;
    }

    int num_speakers = 0;
    for (int i = 0; i < num_segments; i++) {
        int cluster_id = cluster_assignment[i];
        if (cluster_to_speaker[cluster_id] == -1) {
            cluster_to_speaker[cluster_id] = num_speakers++;
        }
        ctx->speaker_ids[i] = cluster_to_speaker[cluster_id];
    }

    ctx->num_speakers = num_speakers;

    free(cluster_assignment);
    free(cluster_active);
    free(cluster_to_speaker);

    return 0;
}
