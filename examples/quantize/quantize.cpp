#include "ggml.h"
#include "ggml-backend.h"

#include "common.h"
#include "common-ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <regex>

// default hparams (Whisper tiny)
struct whisper_hparams {
    int32_t n_vocab       = 51864;
    int32_t n_audio_ctx   = 1500;
    int32_t n_audio_state = 384;
    int32_t n_audio_head  = 6;
    int32_t n_audio_layer = 4;
    int32_t n_text_ctx    = 448;
    int32_t n_text_state  = 384;
    int32_t n_text_head   = 6;
    int32_t n_text_layer  = 4;
    int32_t n_mels        = 80;
    int32_t ftype         = 1;
};

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

// quantize a model
static bool whisper_model_quantize(
        const std::string & fname_inp, 
        const std::string & fname_out, 
        ggml_ftype ftype,
        const std::vector<tensor_quant_spec> & tensor_quant_specs = {}) {
    gpt_vocab vocab;

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

    // verify magic
    {
        uint32_t magic;
        finp.read((char *) &magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname_inp.c_str());
            return false;
        }

        fout.write((char *) &magic, sizeof(magic));
    }

    whisper_hparams hparams;

    // load hparams
    {
        finp.read((char *) &hparams.n_vocab,       sizeof(hparams.n_vocab));
        finp.read((char *) &hparams.n_audio_ctx,   sizeof(hparams.n_audio_ctx));
        finp.read((char *) &hparams.n_audio_state, sizeof(hparams.n_audio_state));
        finp.read((char *) &hparams.n_audio_head,  sizeof(hparams.n_audio_head));
        finp.read((char *) &hparams.n_audio_layer, sizeof(hparams.n_audio_layer));
        finp.read((char *) &hparams.n_text_ctx,    sizeof(hparams.n_text_ctx));
        finp.read((char *) &hparams.n_text_state,  sizeof(hparams.n_text_state));
        finp.read((char *) &hparams.n_text_head,   sizeof(hparams.n_text_head));
        finp.read((char *) &hparams.n_text_layer,  sizeof(hparams.n_text_layer));
        finp.read((char *) &hparams.n_mels,        sizeof(hparams.n_mels));
        finp.read((char *) &hparams.ftype,         sizeof(hparams.ftype));

        const int32_t qntvr_src =    hparams.ftype / GGML_QNT_VERSION_FACTOR;
        
        // For mixed precision quantization, use F16 as the base ftype to ensure
        // all tensor buffers are large enough to hold any quantization type
        const bool use_mixed_precision = !tensor_quant_specs.empty();
        const int32_t ftype_for_allocation = use_mixed_precision ? GGML_FTYPE_MOSTLY_F16 : ftype;
        const int32_t ftype_dst = GGML_QNT_VERSION * GGML_QNT_VERSION_FACTOR + ftype_for_allocation;

        fprintf(stderr, "%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
        fprintf(stderr, "%s: n_audio_ctx   = %d\n", __func__, hparams.n_audio_ctx);
        fprintf(stderr, "%s: n_audio_state = %d\n", __func__, hparams.n_audio_state);
        fprintf(stderr, "%s: n_audio_head  = %d\n", __func__, hparams.n_audio_head);
        fprintf(stderr, "%s: n_audio_layer = %d\n", __func__, hparams.n_audio_layer);
        fprintf(stderr, "%s: n_text_ctx    = %d\n", __func__, hparams.n_text_ctx);
        fprintf(stderr, "%s: n_text_state  = %d\n", __func__, hparams.n_text_state);
        fprintf(stderr, "%s: n_text_head   = %d\n", __func__, hparams.n_text_head);
        fprintf(stderr, "%s: n_text_layer  = %d\n", __func__, hparams.n_text_layer);
        fprintf(stderr, "%s: n_mels        = %d\n", __func__, hparams.n_mels);
        fprintf(stderr, "%s: ftype (src)   = %d\n", __func__, hparams.ftype);
        fprintf(stderr, "%s: qntvr (src)   = %d\n", __func__, qntvr_src);
        fprintf(stderr, "%s: ftype (dst)   = %d\n", __func__, ftype_dst);
        fprintf(stderr, "%s: qntvr (dst)   = %d\n", __func__, GGML_QNT_VERSION);
        if (use_mixed_precision) {
            fprintf(stderr, "%s: using mixed precision quantization (ftype for allocation = F16)\n", __func__);
        }

        fout.write((const char *) &hparams.n_vocab,       sizeof(hparams.n_vocab));
        fout.write((const char *) &hparams.n_audio_ctx,   sizeof(hparams.n_audio_ctx));
        fout.write((const char *) &hparams.n_audio_state, sizeof(hparams.n_audio_state));
        fout.write((const char *) &hparams.n_audio_head,  sizeof(hparams.n_audio_head));
        fout.write((const char *) &hparams.n_audio_layer, sizeof(hparams.n_audio_layer));
        fout.write((const char *) &hparams.n_text_ctx,    sizeof(hparams.n_text_ctx));
        fout.write((const char *) &hparams.n_text_state,  sizeof(hparams.n_text_state));
        fout.write((const char *) &hparams.n_text_head,   sizeof(hparams.n_text_head));
        fout.write((const char *) &hparams.n_text_layer,  sizeof(hparams.n_text_layer));
        fout.write((const char *) &hparams.n_mels,        sizeof(hparams.n_mels));
        fout.write((const char *) &ftype_dst,             sizeof(hparams.ftype));
    }

    // load mel filters
    {
        whisper_filters filters;

        finp.read ((char *) &filters.n_mel, sizeof(filters.n_mel));
        fout.write((char *) &filters.n_mel, sizeof(filters.n_mel));
        finp.read ((char *) &filters.n_fft, sizeof(filters.n_fft));
        fout.write((char *) &filters.n_fft, sizeof(filters.n_fft));

        filters.data.resize(filters.n_mel * filters.n_fft);
        finp.read ((char *) filters.data.data(), filters.data.size() * sizeof(float));
        fout.write((char *) filters.data.data(), filters.data.size() * sizeof(float));
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        finp.read ((char *) &n_vocab, sizeof(n_vocab));
        fout.write((char *) &n_vocab, sizeof(n_vocab));

        //if (n_vocab != hparams.n_vocab) {
        //    fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
        //            __func__, fname_inp.c_str(), n_vocab, hparams.n_vocab);
        //    return false;
        //}

        char word[129];

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            finp.read ((char *) &len, sizeof(len));
            fout.write((char *) &len, sizeof(len));

            word[len] = '\0';

            finp.read ((char *) word, len);
            fout.write((char *) word, len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // regexes of tensor names to not be quantized
    const std::vector<std::string> to_skip = {
        //"encoder.*",
        "encoder.conv1.bias",
        "encoder.conv2.bias",
        "encoder.positional_embedding",
        "decoder.positional_embedding",
    };

    // Use the extended quantization function if we have per-tensor specs
    bool success;
    if (!tensor_quant_specs.empty()) {
        success = ggml_common_quantize_0(finp, fout, ftype, { ".*" }, to_skip, tensor_quant_specs);
    } else {
        success = ggml_common_quantize_0(finp, fout, ftype, { ".*" }, to_skip);
    }
    
    if (!success) {
        fprintf(stderr, "%s: failed to quantize model '%s'\n", __func__, fname_inp.c_str());
        return false;
    }

    finp.close();
    fout.close();

    return true;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    if (argc < 4) {
        fprintf(stderr, "usage: %s [--tensor-type PATTERN=TYPE ...] model-f32.bin model-quant.bin type\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "  --tensor-type PATTERN=TYPE : specify quantization type for tensors matching PATTERN\n");
        fprintf(stderr, "      PATTERN is a regex pattern to match tensor names\n");
        fprintf(stderr, "      TYPE is a quantization type (e.g., q4_0, q8_0, f16)\n");
        fprintf(stderr, "      Example: --tensor-type 'encoder\\..*\\.weight'=q8_0 --tensor-type 'decoder\\..*\\.weight'=q4_0\n");
        fprintf(stderr, "\n");
        ggml_print_ftypes(stderr);
        return 1;
    }

    // Parse optional arguments
    std::vector<tensor_quant_spec> tensor_quant_specs;
    int arg_idx = 1;
    
    while (arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0) {
        if (strcmp(argv[arg_idx], "--tensor-type") == 0) {
            if (arg_idx + 1 >= argc) {
                fprintf(stderr, "error: --tensor-type requires an argument\n");
                return 1;
            }
            arg_idx++;
            
            // Parse PATTERN=TYPE
            const char * spec_str = argv[arg_idx];
            const char * eq = strchr(spec_str, '=');
            if (eq == nullptr) {
                fprintf(stderr, "error: invalid --tensor-type format '%s', expected PATTERN=TYPE\n", spec_str);
                return 1;
            }
            
            std::string pattern(spec_str, eq - spec_str);
            std::string type_str(eq + 1);
            
            ggml_type qtype = ggml_parse_qtype(type_str.c_str());
            if (qtype == GGML_TYPE_COUNT) {
                fprintf(stderr, "error: unknown quantization type '%s'\n", type_str.c_str());
                return 1;
            }
            
            tensor_quant_spec spec;
            spec.pattern = pattern;
            spec.quant_type = qtype;
            tensor_quant_specs.push_back(spec);
            
            printf("Added tensor quantization spec: pattern='%s' type=%s\n", 
                   pattern.c_str(), ggml_type_name(qtype));
        } else {
            fprintf(stderr, "error: unknown option '%s'\n", argv[arg_idx]);
            return 1;
        }
        arg_idx++;
    }
    
    if (argc - arg_idx < 3) {
        fprintf(stderr, "error: missing required arguments\n");
        fprintf(stderr, "usage: %s [--tensor-type PATTERN=TYPE ...] model-f32.bin model-quant.bin type\n", argv[0]);
        return 1;
    }

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    const std::string fname_inp = argv[arg_idx];
    const std::string fname_out = argv[arg_idx + 1];

    const ggml_ftype ftype = ggml_parse_ftype(argv[arg_idx + 2]);

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!whisper_model_quantize(fname_inp, fname_out, ggml_ftype(ftype), tensor_quant_specs)) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = ggml_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    return 0;
}
