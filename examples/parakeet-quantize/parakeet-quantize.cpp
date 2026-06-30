#include "ggml.h"
#include "ggml-backend.h"

#include "common-ggml.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct parakeet_hparams {
    int32_t n_vocab                = 0;
    int32_t n_audio_ctx            = 0;
    int32_t n_audio_state          = 0;
    int32_t n_audio_head           = 0;
    int32_t n_audio_layer          = 0;
    int32_t n_mels                 = 0;
    int32_t ftype                  = 0;
    int32_t n_fft                  = 0;
    int32_t subsampling_factor     = 0;
    int32_t n_subsampling_channels = 0;
    int32_t n_conv_kernel          = 0;
    int32_t n_pred_dim             = 0;
    int32_t n_pred_layers          = 0;
    int32_t n_tdt_durations        = 0;
    int32_t n_max_tokens           = 0;
};

static bool parakeet_model_quantize(const std::string & fname_inp, const std::string & fname_out, ggml_ftype ftype) {
    printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

    auto finp = std::ifstream(fname_inp, std::ios::binary);
    if (!finp) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, fname_inp.c_str());
        return false;
    }

    auto fout = std::ofstream(fname_out, std::ios::binary);
    if (!fout) {
        fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname_out.c_str());
        return false;
    }

    // magic
    {
        uint32_t magic;
        finp.read((char *) &magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file (bad magic)\n", __func__);
            return false;
        }
        fout.write((char *) &magic, sizeof(magic));
    }

    // hparams
    parakeet_hparams hparams;
    {
        finp.read((char *) &hparams.n_vocab,                sizeof(hparams.n_vocab));
        finp.read((char *) &hparams.n_audio_ctx,            sizeof(hparams.n_audio_ctx));
        finp.read((char *) &hparams.n_audio_state,          sizeof(hparams.n_audio_state));
        finp.read((char *) &hparams.n_audio_head,           sizeof(hparams.n_audio_head));
        finp.read((char *) &hparams.n_audio_layer,          sizeof(hparams.n_audio_layer));
        finp.read((char *) &hparams.n_mels,                 sizeof(hparams.n_mels));
        finp.read((char *) &hparams.ftype,                  sizeof(hparams.ftype));
        finp.read((char *) &hparams.n_fft,                  sizeof(hparams.n_fft));
        finp.read((char *) &hparams.subsampling_factor,     sizeof(hparams.subsampling_factor));
        finp.read((char *) &hparams.n_subsampling_channels, sizeof(hparams.n_subsampling_channels));
        finp.read((char *) &hparams.n_conv_kernel,          sizeof(hparams.n_conv_kernel));
        finp.read((char *) &hparams.n_pred_dim,             sizeof(hparams.n_pred_dim));
        finp.read((char *) &hparams.n_pred_layers,          sizeof(hparams.n_pred_layers));
        finp.read((char *) &hparams.n_tdt_durations,        sizeof(hparams.n_tdt_durations));
        finp.read((char *) &hparams.n_max_tokens,           sizeof(hparams.n_max_tokens));

        const int32_t qntvr_src = hparams.ftype / GGML_QNT_VERSION_FACTOR;
        const int32_t ftype_dst = GGML_QNT_VERSION * GGML_QNT_VERSION_FACTOR + ftype;

        fprintf(stderr, "%s: n_vocab              = %d\n",  __func__, hparams.n_vocab);
        fprintf(stderr, "%s: n_audio_state        = %d\n",  __func__, hparams.n_audio_state);
        fprintf(stderr, "%s: n_audio_layer        = %d\n",  __func__, hparams.n_audio_layer);
        fprintf(stderr, "%s: n_mels               = %d\n",  __func__, hparams.n_mels);
        fprintf(stderr, "%s: ftype (src)          = %d\n",  __func__, hparams.ftype);
        fprintf(stderr, "%s: qntvr (src)          = %d\n",  __func__, qntvr_src);
        fprintf(stderr, "%s: ftype (dst)          = %d\n",  __func__, ftype_dst);
        fprintf(stderr, "%s: qntvr (dst)          = %d\n",  __func__, GGML_QNT_VERSION);

        fout.write((char *) &hparams.n_vocab,                sizeof(hparams.n_vocab));
        fout.write((char *) &hparams.n_audio_ctx,            sizeof(hparams.n_audio_ctx));
        fout.write((char *) &hparams.n_audio_state,          sizeof(hparams.n_audio_state));
        fout.write((char *) &hparams.n_audio_head,           sizeof(hparams.n_audio_head));
        fout.write((char *) &hparams.n_audio_layer,          sizeof(hparams.n_audio_layer));
        fout.write((char *) &hparams.n_mels,                 sizeof(hparams.n_mels));
        fout.write((char *) &ftype_dst,                      sizeof(ftype_dst));
        fout.write((char *) &hparams.n_fft,                  sizeof(hparams.n_fft));
        fout.write((char *) &hparams.subsampling_factor,     sizeof(hparams.subsampling_factor));
        fout.write((char *) &hparams.n_subsampling_channels, sizeof(hparams.n_subsampling_channels));
        fout.write((char *) &hparams.n_conv_kernel,          sizeof(hparams.n_conv_kernel));
        fout.write((char *) &hparams.n_pred_dim,             sizeof(hparams.n_pred_dim));
        fout.write((char *) &hparams.n_pred_layers,          sizeof(hparams.n_pred_layers));
        fout.write((char *) &hparams.n_tdt_durations,        sizeof(hparams.n_tdt_durations));
        fout.write((char *) &hparams.n_max_tokens,           sizeof(hparams.n_max_tokens));
    }

    // mel filterbank
    {
        int32_t n_mel, n_fb;
        finp.read((char *) &n_mel, sizeof(n_mel));
        fout.write((char *) &n_mel, sizeof(n_mel));
        finp.read((char *) &n_fb,  sizeof(n_fb));
        fout.write((char *) &n_fb,  sizeof(n_fb));

        const size_t n = (size_t) n_mel * n_fb;
        std::vector<float> buf(n);
        finp.read((char *) buf.data(), n * sizeof(float));
        fout.write((char *) buf.data(), n * sizeof(float));
    }

    // window function
    {
        int32_t n_window;
        finp.read((char *) &n_window, sizeof(n_window));
        fout.write((char *) &n_window, sizeof(n_window));

        std::vector<float> buf(n_window);
        finp.read((char *) buf.data(), n_window * sizeof(float));
        fout.write((char *) buf.data(), n_window * sizeof(float));
    }

    // TDT durations
    {
        std::vector<uint32_t> buf(hparams.n_tdt_durations);
        finp.read((char *) buf.data(), hparams.n_tdt_durations * sizeof(uint32_t));
        fout.write((char *) buf.data(), hparams.n_tdt_durations * sizeof(uint32_t));
    }

    // vocab
    {
        int32_t n_tokens;
        finp.read((char *) &n_tokens, sizeof(n_tokens));
        fout.write((char *) &n_tokens, sizeof(n_tokens));

        for (int i = 0; i < n_tokens; ++i) {
            int32_t len;
            finp.read((char *) &len, sizeof(len));
            fout.write((char *) &len, sizeof(len));

            std::string token(len, '\0');
            finp.read(&token[0], len);
            fout.write(&token[0], len);
        }
    }

    // tensors — quantize 2D weights skipping tensors that must stay F32:
    // ggml_ssm_conv / ggml_conv2d_dw CUDA kernels require F32 weights.
    // pos_bias_u / pos_bias_v are declared F32 in the loader.
    const std::vector<std::string> to_quant = { ".*" };
    std::vector<std::string> to_skip = {
        // CUDA kernel constraints (ggml_ssm_conv / ggml_conv2d_dw require F32 weights)
        "encoder\\.layers\\..+\\.conv\\.depthwise_conv\\.weight",
        // Declared F32 in loader (pos_bias tensors)
        "encoder\\.layers\\..+\\.self_attn\\.pos_bias_u",
        "encoder\\.layers\\..+\\.self_attn\\.pos_bias_v",
    };

    // Prediction/joint tensors use n_pred_dim as their inner dimension. K-quant
    // types (block size 256) cannot quantize 640 evenly, so keep them F32. For
    // other types (Q8_0, Q4_0, block size 32) 640 is divisible and they can be
    // quantized normally. The loader mirrors this logic at load time.
    {
        const ggml_type qtype = ggml_ftype_to_ggml_type(ftype);
        const int32_t   blck  = ggml_blck_size(qtype);
        if (blck > 1 && hparams.n_pred_dim % blck != 0) {
            to_skip.push_back("decoder\\.prediction\\.embed\\.weight");
            to_skip.push_back("decoder\\.prediction\\.dec_rnn\\.lstm\\.weight_ih_l.*");
            to_skip.push_back("decoder\\.prediction\\.dec_rnn\\.lstm\\.weight_hh_l.*");
            to_skip.push_back("joint\\.pred\\.weight");
            to_skip.push_back("joint\\.joint_net\\.2\\.weight");
        }
    }

    if (!ggml_common_quantize_0(finp, fout, ftype, to_quant, to_skip)) {
        fprintf(stderr, "%s: failed to quantize tensors\n", __func__);
        return false;
    }

    finp.close();
    fout.close();

    return true;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    if (argc != 4) {
        fprintf(stderr, "usage: %s model-f32.bin model-quant.bin type\n", argv[0]);
        ggml_print_ftypes(stderr);
        return 1;
    }

    // initialise F16 lookup tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];
    const ggml_ftype  ftype     = ggml_parse_ftype(argv[3]);

    if (ftype == GGML_FTYPE_UNKNOWN) {
        fprintf(stderr, "%s: invalid quantization type\n", argv[0]);
        ggml_print_ftypes(stderr);
        return 1;
    }

    const int64_t t_start_us = ggml_time_us();

    if (!parakeet_model_quantize(fname_inp, fname_out, ftype)) {
        fprintf(stderr, "%s: failed to quantize model from '%s'\n", argv[0], fname_inp.c_str());
        return 1;
    }

    printf("\n%s: quantize time = %8.2f ms\n", argv[0], (ggml_time_us() - t_start_us) / 1000.0f);
    printf("%s: output model  = %s\n",         argv[0], fname_out.c_str());

    return 0;
}
