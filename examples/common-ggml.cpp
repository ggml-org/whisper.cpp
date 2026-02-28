#include "common-ggml.h"

#include <regex>
#include <map>
#include <algorithm>
#include <cctype>

static const std::map<std::string, enum ggml_ftype> GGML_FTYPE_MAP = {
    {"q4_0", GGML_FTYPE_MOSTLY_Q4_0},
    {"q4_1", GGML_FTYPE_MOSTLY_Q4_1},
    {"q5_0", GGML_FTYPE_MOSTLY_Q5_0},
    {"q5_1", GGML_FTYPE_MOSTLY_Q5_1},
    {"q8_0", GGML_FTYPE_MOSTLY_Q8_0},
    {"q2_k", GGML_FTYPE_MOSTLY_Q2_K},
    {"q3_k", GGML_FTYPE_MOSTLY_Q3_K},
    {"q4_k", GGML_FTYPE_MOSTLY_Q4_K},
    {"q5_k", GGML_FTYPE_MOSTLY_Q5_K},
    {"q6_k", GGML_FTYPE_MOSTLY_Q6_K},
};

static const std::map<std::string, enum ggml_type> GGML_TYPE_MAP = {
    {"q4_0", GGML_TYPE_Q4_0},
    {"q4_1", GGML_TYPE_Q4_1},
    {"q5_0", GGML_TYPE_Q5_0},
    {"q5_1", GGML_TYPE_Q5_1},
    {"q8_0", GGML_TYPE_Q8_0},
    {"q2_k", GGML_TYPE_Q2_K},
    {"q3_k", GGML_TYPE_Q3_K},
    {"q4_k", GGML_TYPE_Q4_K},
    {"q5_k", GGML_TYPE_Q5_K},
    {"q6_k", GGML_TYPE_Q6_K},
    {"f16",  GGML_TYPE_F16},
    {"f32",  GGML_TYPE_F32},
};

void ggml_print_ftypes(FILE * fp) {
    for (auto it = GGML_FTYPE_MAP.begin(); it != GGML_FTYPE_MAP.end(); it++) {
        fprintf(fp, "  type = \"%s\" or %d\n", it->first.c_str(), it->second);
    }
}

enum ggml_ftype ggml_parse_ftype(const char * str) {
    enum ggml_ftype ftype;
    if (str[0] == 'q') {
        const auto it = GGML_FTYPE_MAP.find(str);
        if (it == GGML_FTYPE_MAP.end()) {
            fprintf(stderr, "%s: unknown ftype '%s'\n", __func__, str);
            return GGML_FTYPE_UNKNOWN;
        }
        ftype = it->second;
    } else {
        ftype = (enum ggml_ftype) atoi(str);
    }

    return ftype;
}

ggml_type ggml_parse_qtype(const char * str) {
    std::string str_lower(str);
    std::transform(str_lower.begin(), str_lower.end(), str_lower.begin(), ::tolower);
    
    const auto it = GGML_TYPE_MAP.find(str_lower);
    if (it == GGML_TYPE_MAP.end()) {
        fprintf(stderr, "%s: unknown qtype '%s'\n", __func__, str);
        return GGML_TYPE_COUNT;
    }
    return it->second;
}

bool ggml_common_quantize_0(
        std::ifstream & finp,
        std::ofstream & fout,
        const ggml_ftype ftype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip) {

    ggml_type qtype = GGML_TYPE_F32;

    switch (ftype) {
        case GGML_FTYPE_MOSTLY_Q4_0: qtype = GGML_TYPE_Q4_0; break;
        case GGML_FTYPE_MOSTLY_Q4_1: qtype = GGML_TYPE_Q4_1; break;
        case GGML_FTYPE_MOSTLY_Q5_0: qtype = GGML_TYPE_Q5_0; break;
        case GGML_FTYPE_MOSTLY_Q5_1: qtype = GGML_TYPE_Q5_1; break;
        case GGML_FTYPE_MOSTLY_Q8_0: qtype = GGML_TYPE_Q8_0; break;
        case GGML_FTYPE_MOSTLY_Q2_K: qtype = GGML_TYPE_Q2_K; break;
        case GGML_FTYPE_MOSTLY_Q3_K: qtype = GGML_TYPE_Q3_K; break;
        case GGML_FTYPE_MOSTLY_Q4_K: qtype = GGML_TYPE_Q4_K; break;
        case GGML_FTYPE_MOSTLY_Q5_K: qtype = GGML_TYPE_Q5_K; break;
        case GGML_FTYPE_MOSTLY_Q6_K: qtype = GGML_TYPE_Q6_K; break;
        case GGML_FTYPE_UNKNOWN:
        case GGML_FTYPE_ALL_F32:
        case GGML_FTYPE_MOSTLY_F16:
        case GGML_FTYPE_MOSTLY_Q4_1_SOME_F16:
        case GGML_FTYPE_MOSTLY_IQ2_XXS:
        case GGML_FTYPE_MOSTLY_IQ2_XS:
        case GGML_FTYPE_MOSTLY_IQ2_S:
        case GGML_FTYPE_MOSTLY_IQ3_XXS:
        case GGML_FTYPE_MOSTLY_IQ3_S:
        case GGML_FTYPE_MOSTLY_IQ1_S:
        case GGML_FTYPE_MOSTLY_IQ4_NL:
        case GGML_FTYPE_MOSTLY_IQ4_XS:
        case GGML_FTYPE_MOSTLY_IQ1_M:
        case GGML_FTYPE_MOSTLY_BF16:
        case GGML_FTYPE_MOSTLY_MXFP4:
                {
                    fprintf(stderr, "%s: invalid model type %d\n", __func__, ftype);
                    return false;
                }
    };

    if (!ggml_is_quantized(qtype)) {
        fprintf(stderr, "%s: invalid quantization type %d (%s)\n", __func__, qtype, ggml_type_name(qtype));
        return false;
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<float> work;

    std::vector<uint8_t>     data_u8;
    std::vector<ggml_fp16_t> data_f16;
    std::vector<float>       data_f32;

    while (true) {
        int32_t n_dims;
        int32_t length;
        int32_t ttype;

        finp.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        finp.read(reinterpret_cast<char *>(&length), sizeof(length));
        finp.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));

        if (finp.eof()) {
            break;
        }

        int32_t nelements = 1;
        int32_t ne[4] = { 1, 1, 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            finp.read (reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            nelements *= ne[i];
        }

        std::string name(length, 0);
        finp.read (&name[0], length);

        printf("%64s - [%5d, %5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ne[2], ggml_type_name((ggml_type) ttype));

        bool quantize = false;

        // check if we should quantize this tensor
        for (const auto & s : to_quant) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = true;
                break;
            }
        }

        // check if we should skip this tensor
        for (const auto & s : to_skip) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = false;
                break;
            }
        }

        // quantize only 2D tensors
        quantize &= (n_dims == 2);

        if (quantize) {
            if (ttype != GGML_TYPE_F32 && ttype != GGML_TYPE_F16) {
                fprintf(stderr, "%s: unsupported ttype %d (%s) for integer quantization\n", __func__, ttype, ggml_type_name((ggml_type) ttype));
                return false;
            }

            if (ttype == GGML_TYPE_F16) {
                data_f16.resize(nelements);
                finp.read(reinterpret_cast<char *>(data_f16.data()), nelements * sizeof(ggml_fp16_t));
                data_f32.resize(nelements);
                for (int i = 0; i < nelements; ++i) {
                    data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                }
            } else {
                data_f32.resize(nelements);
                finp.read(reinterpret_cast<char *>(data_f32.data()), nelements * sizeof(float));
            }

            ttype = qtype;
        } else {
            // For non-quantized tensors, we need to correctly calculate size based on type
            // Use ggml_row_size to get the correct size for the tensor's row
            const size_t row_size = ggml_row_size((ggml_type) ttype, ne[0]);
            const size_t data_size = row_size * (nelements / ne[0]);

            data_u8.resize(data_size);
            finp.read(reinterpret_cast<char *>(data_u8.data()), data_size);
        }

        fout.write(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        fout.write(reinterpret_cast<char *>(&length), sizeof(length));
        fout.write(reinterpret_cast<char *>(&ttype),  sizeof(ttype));
        for (int i = 0; i < n_dims; ++i) {
            fout.write(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        }
        fout.write(&name[0], length);

        if (quantize) {
            work.resize(nelements); // for quantization

            size_t cur_size = 0;
            switch ((ggml_type) ttype) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    {
                        cur_size = ggml_quantize_chunk((ggml_type) ttype, data_f32.data(), work.data(), 0, nelements/ne[0], ne[0], nullptr);
                    } break;
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_I8:
                case GGML_TYPE_I16:
                case GGML_TYPE_I32:
                case GGML_TYPE_I64:
                case GGML_TYPE_F64:
                case GGML_TYPE_Q8_1:
                case GGML_TYPE_Q8_K:
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                case GGML_TYPE_IQ3_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ1_S:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                case GGML_TYPE_IQ1_M:
                case GGML_TYPE_BF16:
                case GGML_TYPE_TQ1_0:
                case GGML_TYPE_TQ2_0:
                case GGML_TYPE_MXFP4:
                case GGML_TYPE_COUNT:
                    {
                        fprintf(stderr, "%s: unsupported quantization type %d (%s)\n", __func__, ttype, ggml_type_name((ggml_type) ttype));
                        return false;
                    }
            }

            fout.write(reinterpret_cast<char *>(work.data()), cur_size);
            total_size_new += cur_size;

            printf("size = %8.2f MB -> %8.2f MB\n", nelements * sizeof(float)/1024.0/1024.0, cur_size/1024.0/1024.0);
        } else {
            printf("size = %8.3f MB\n", data_u8.size()/1024.0/1024.0);
            fout.write(reinterpret_cast<char *>(data_u8.data()), data_u8.size());
            total_size_new += data_u8.size();
        }

        total_size_org += nelements * sizeof(float);
    }

    printf("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    printf("%s: quant size  = %8.2f MB | ftype = %d (%s)\n", __func__, total_size_new/1024.0/1024.0, ftype, ggml_type_name(qtype));

    return true;
}

// Extended quantization function with per-tensor quantization support
bool ggml_common_quantize_0(
        std::ifstream & finp,
        std::ofstream & fout,
        const ggml_ftype ftype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip,
        const std::vector<tensor_quant_spec> & tensor_quant_specs) {

    ggml_type default_qtype = GGML_TYPE_F32;

    switch (ftype) {
        case GGML_FTYPE_MOSTLY_Q4_0: default_qtype = GGML_TYPE_Q4_0; break;
        case GGML_FTYPE_MOSTLY_Q4_1: default_qtype = GGML_TYPE_Q4_1; break;
        case GGML_FTYPE_MOSTLY_Q5_0: default_qtype = GGML_TYPE_Q5_0; break;
        case GGML_FTYPE_MOSTLY_Q5_1: default_qtype = GGML_TYPE_Q5_1; break;
        case GGML_FTYPE_MOSTLY_Q8_0: default_qtype = GGML_TYPE_Q8_0; break;
        case GGML_FTYPE_MOSTLY_Q2_K: default_qtype = GGML_TYPE_Q2_K; break;
        case GGML_FTYPE_MOSTLY_Q3_K: default_qtype = GGML_TYPE_Q3_K; break;
        case GGML_FTYPE_MOSTLY_Q4_K: default_qtype = GGML_TYPE_Q4_K; break;
        case GGML_FTYPE_MOSTLY_Q5_K: default_qtype = GGML_TYPE_Q5_K; break;
        case GGML_FTYPE_MOSTLY_Q6_K: default_qtype = GGML_TYPE_Q6_K; break;
        case GGML_FTYPE_UNKNOWN:
        case GGML_FTYPE_ALL_F32:
        case GGML_FTYPE_MOSTLY_F16:
        case GGML_FTYPE_MOSTLY_Q4_1_SOME_F16:
        case GGML_FTYPE_MOSTLY_IQ2_XXS:
        case GGML_FTYPE_MOSTLY_IQ2_XS:
        case GGML_FTYPE_MOSTLY_IQ2_S:
        case GGML_FTYPE_MOSTLY_IQ3_XXS:
        case GGML_FTYPE_MOSTLY_IQ3_S:
        case GGML_FTYPE_MOSTLY_IQ1_S:
        case GGML_FTYPE_MOSTLY_IQ4_NL:
        case GGML_FTYPE_MOSTLY_IQ4_XS:
        case GGML_FTYPE_MOSTLY_IQ1_M:
        case GGML_FTYPE_MOSTLY_BF16:
        case GGML_FTYPE_MOSTLY_MXFP4:
                {
                    fprintf(stderr, "%s: unsupported model type %d (ftype=%d)\n", __func__, ftype, ftype);
                    return false;
                }
    };

    if (!ggml_is_quantized(default_qtype)) {
        fprintf(stderr, "%s: invalid quantization type %d (%s)\n", __func__, default_qtype, ggml_type_name(default_qtype));
        return false;
    }

    // Pre-compile regex patterns for efficiency
    struct compiled_pattern {
        std::regex regex;
        ggml_type quant_type;
    };
    std::vector<compiled_pattern> compiled_patterns;
    compiled_patterns.reserve(tensor_quant_specs.size());
    
    for (const auto & spec : tensor_quant_specs) {
        try {
            compiled_patterns.push_back({std::regex(spec.pattern), spec.quant_type});
        } catch (const std::regex_error & e) {
            fprintf(stderr, "%s: invalid regex pattern '%s': %s\n", __func__, spec.pattern.c_str(), e.what());
            return false;
        }
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<float> work;

    std::vector<uint8_t>     data_u8;
    std::vector<ggml_fp16_t> data_f16;
    std::vector<float>       data_f32;

    std::unordered_map<std::string, int> quant_type_counts;

    while (true) {
        int32_t n_dims;
        int32_t length;
        int32_t ttype;

        finp.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        finp.read(reinterpret_cast<char *>(&length), sizeof(length));
        finp.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));

        if (finp.eof()) {
            break;
        }

        int32_t nelements = 1;
        int32_t ne[4] = { 1, 1, 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            finp.read (reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            nelements *= ne[i];
        }

        std::string name(length, 0);
        finp.read (&name[0], length);

        printf("%64s - [%5d, %5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ne[2], ggml_type_name((ggml_type) ttype));

        bool quantize = false;
        ggml_type qtype = default_qtype;

        // check if we should quantize this tensor
        for (const auto & s : to_quant) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = true;
                break;
            }
        }

        // check if we should skip this tensor
        for (const auto & s : to_skip) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = false;
                break;
            }
        }

        // check for per-tensor quantization specification
        if (quantize) {
            for (const auto & cp : compiled_patterns) {
                if (std::regex_match(name, cp.regex)) {
                    qtype = cp.quant_type;
                    printf("matched pattern -> %s ", ggml_type_name(qtype));
                    break;
                }
            }
        }

        // quantize only 2D tensors
        quantize &= (n_dims == 2);

        if (quantize) {
            if (ttype != GGML_TYPE_F32 && ttype != GGML_TYPE_F16) {
                fprintf(stderr, "%s: unsupported ttype %d (%s) for integer quantization\n", __func__, ttype, ggml_type_name((ggml_type) ttype));
                return false;
            }

            if (ttype == GGML_TYPE_F16) {
                data_f16.resize(nelements);
                finp.read(reinterpret_cast<char *>(data_f16.data()), nelements * sizeof(ggml_fp16_t));
                data_f32.resize(nelements);
                for (int i = 0; i < nelements; ++i) {
                    data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                }
            } else {
                data_f32.resize(nelements);
                finp.read(reinterpret_cast<char *>(data_f32.data()), nelements * sizeof(float));
            }

            ttype = qtype;
            quant_type_counts[ggml_type_name(qtype)]++;
        } else {
            // For non-quantized tensors, we need to correctly calculate size based on type
            // Use ggml_row_size to get the correct size for the tensor's row
            const size_t row_size = ggml_row_size((ggml_type) ttype, ne[0]);
            const size_t data_size = row_size * (nelements / ne[0]);

            data_u8.resize(data_size);
            finp.read(reinterpret_cast<char *>(data_u8.data()), data_size);
        }

        fout.write(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        fout.write(reinterpret_cast<char *>(&length), sizeof(length));
        fout.write(reinterpret_cast<char *>(&ttype),  sizeof(ttype));
        for (int i = 0; i < n_dims; ++i) {
            fout.write(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        }
        fout.write(&name[0], length);

        if (quantize) {
            work.resize(nelements); // for quantization

            size_t cur_size = 0;
            switch ((ggml_type) ttype) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    {
                        cur_size = ggml_quantize_chunk((ggml_type) ttype, data_f32.data(), work.data(), 0, nelements/ne[0], ne[0], nullptr);
                    } break;
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_I8:
                case GGML_TYPE_I16:
                case GGML_TYPE_I32:
                case GGML_TYPE_I64:
                case GGML_TYPE_F64:
                case GGML_TYPE_Q8_1:
                case GGML_TYPE_Q8_K:
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                case GGML_TYPE_IQ3_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ1_S:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                case GGML_TYPE_IQ1_M:
                case GGML_TYPE_BF16:
                case GGML_TYPE_TQ1_0:
                case GGML_TYPE_TQ2_0:
                case GGML_TYPE_MXFP4:
                case GGML_TYPE_COUNT:
                    {
                        fprintf(stderr, "%s: unsupported quantization type %d (%s)\n", __func__, ttype, ggml_type_name((ggml_type) ttype));
                        return false;
                    }
            }

            fout.write(reinterpret_cast<char *>(work.data()), cur_size);
            total_size_new += cur_size;

            printf("size = %8.2f MB -> %8.2f MB\n", nelements * sizeof(float)/1024.0/1024.0, cur_size/1024.0/1024.0);
        } else {
            printf("size = %8.3f MB\n", data_u8.size()/1024.0/1024.0);
            fout.write(reinterpret_cast<char *>(data_u8.data()), data_u8.size());
            total_size_new += data_u8.size();
        }

        total_size_org += nelements * sizeof(float);
    }

    printf("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    printf("%s: quant size  = %8.2f MB | ftype = %d (%s)\n", __func__, total_size_new/1024.0/1024.0, ftype, ggml_type_name(default_qtype));
    
    printf("%s: quantization type summary:\n", __func__);
    for (const auto & kv : quant_type_counts) {
        printf("%s:   %s: %d tensors\n", __func__, kv.first.c_str(), kv.second);
    }

    return true;
}
