#include "whisper-speaker.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

#define WHISPER_LOG_ERROR(...) fprintf(stderr, "[ERROR] " __VA_ARGS__)
#define WHISPER_LOG_WARN(...)  fprintf(stderr, "[WARN]  " __VA_ARGS__)
#define WHISPER_LOG_INFO(...)  fprintf(stderr, "[INFO]  " __VA_ARGS__)

struct whisper_speaker_model {
    struct ggml_context * ctx;
    std::vector<struct ggml_tensor *> tensors;
    std::vector<std::string> tensor_names;
    int embedding_dim;
    int n_tensors;
};

// Load GGML speaker model from file
whisper_speaker_model * whisper_speaker_load_from_file(const char * path_model) {
    FILE * fin = fopen(path_model, "rb");
    if (!fin) {
        fprintf(stderr, "Failed to open model file: %s\n", path_model);
        return nullptr;
    }

    // Read magic number (must be 0x67676d6c = "ggml")
    uint32_t magic;
    if (fread(&magic, sizeof(magic), 1, fin) != 1) {
        fprintf(stderr, "Failed to read magic number\n");
        fclose(fin);
        return nullptr;
    }
    if (magic != 0x67676d6c) {  // "ggml"
        WHISPER_LOG_ERROR("invalid GGML magic: 0x%x (expected 0x67676d6c)\n", magic);
        fclose(fin);
        return nullptr;
    }

    // Read model type string (length-prefixed UTF-8)
    int str_len;
    if (fread(&str_len, sizeof(str_len), 1, fin) != 1) {
        WHISPER_LOG_ERROR("failed to read model type length\n");
        fclose(fin);
        return nullptr;
    }

    if (str_len < 0 || str_len > 256) {
        WHISPER_LOG_ERROR("invalid model type length: %d\n", str_len);
        fclose(fin);
        return nullptr;
    }

    char model_type[257];
    if (fread(model_type, str_len, 1, fin) != 1) {
        WHISPER_LOG_ERROR("failed to read model type\n");
        fclose(fin);
        return nullptr;
    }
    model_type[str_len] = '\0';
    WHISPER_LOG_INFO("speaker model type: %s\n", model_type);

    // Read version (major, minor, patch)
    int major, minor, patch;
    if (fread(&major, sizeof(major), 1, fin) != 1 ||
        fread(&minor, sizeof(minor), 1, fin) != 1 ||
        fread(&patch, sizeof(patch), 1, fin) != 1) {
        WHISPER_LOG_ERROR("failed to read version\n");
        fclose(fin);
        return nullptr;
    }
    WHISPER_LOG_INFO("speaker model version: %d.%d.%d\n", major, minor, patch);

    // Read hyperparameters
    int embedding_dim;
    if (fread(&embedding_dim, sizeof(embedding_dim), 1, fin) != 1) {
        WHISPER_LOG_ERROR("failed to read embedding_dim\n");
        fclose(fin);
        return nullptr;
    }

    int n_channels;
    if (fread(&n_channels, sizeof(n_channels), 1, fin) != 1) {
        WHISPER_LOG_ERROR("failed to read n_channels\n");
        fclose(fin);
        return nullptr;
    }

    // Read tensor count (for verification)
    int n_tensors_expected;
    if (fread(&n_tensors_expected, sizeof(n_tensors_expected), 1, fin) != 1) {
        WHISPER_LOG_ERROR("failed to read tensor count\n");
        fclose(fin);
        return nullptr;
    }

    WHISPER_LOG_INFO("speaker model: embedding_dim=%d, n_tensors=%d\n", embedding_dim, n_tensors_expected);

    // Estimate context size from file: file_size + overhead for tensor metadata
    long cur_pos = ftell(fin);
    fseek(fin, 0, SEEK_END);
    long file_size = ftell(fin);
    fseek(fin, cur_pos, SEEK_SET);
    size_t ctx_size = (size_t)(file_size - cur_pos) + (size_t)n_tensors_expected * 1024 + 16 * 1024 * 1024;

    struct ggml_init_params ggml_params = {
        ctx_size,
        nullptr,
        false,
    };

    struct ggml_context * ctx = ggml_init(ggml_params);
    if (!ctx) {
        WHISPER_LOG_ERROR("failed to create ggml context (%zu bytes)\n", ctx_size);
        fclose(fin);
        return nullptr;
    }

    // Create speaker model structure
    whisper_speaker_model * model = new whisper_speaker_model();
    model->ctx = ctx;
    model->embedding_dim = embedding_dim;
    model->n_tensors = 0;

    // Load tensors
    for (int t = 0; t < n_tensors_expected; ++t) {
        // Read tensor header: n_dims, name_len
        int n_dims;
        if (fread(&n_dims, sizeof(n_dims), 1, fin) != 1) {
            WHISPER_LOG_ERROR("failed to read n_dims for tensor %d\n", t);
            break;
        }

        int name_len;
        if (fread(&name_len, sizeof(name_len), 1, fin) != 1) {
            WHISPER_LOG_ERROR("failed to read name_len for tensor %d\n", t);
            break;
        }

        // Sanity checks
        if (n_dims < 0 || n_dims > 8) {
            WHISPER_LOG_ERROR("invalid n_dims for tensor %d: %d\n", t, n_dims);
            break;
        }
        if (name_len < 0 || name_len > 512) {
            WHISPER_LOG_ERROR("invalid name_len for tensor %d: %d\n", t, name_len);
            break;
        }

        // Read dimensions (convert to int64_t)
        int64_t dims[8] = {0};
        for (int i = 0; i < n_dims; ++i) {
            int dim;
            if (fread(&dim, sizeof(int), 1, fin) != 1) {
                WHISPER_LOG_ERROR("failed to read dim %d for tensor %d\n", i, t);
                break;
            }
            dims[i] = (int64_t)dim;
        }

        // Read tensor name (not null-terminated in binary)
        char name[513];
        if (fread(name, name_len, 1, fin) != 1) {
            WHISPER_LOG_ERROR("failed to read tensor name for tensor %d\n", t);
            break;
        }
        name[name_len] = '\0';

        // Create tensor in ggml context
        struct ggml_tensor * tensor = ggml_new_tensor(ctx, GGML_TYPE_F32, n_dims, dims);
        if (!tensor) {
            WHISPER_LOG_ERROR("failed to create tensor: %s\n", name);
            break;
        }

        // Read tensor data (float32)
        size_t nelements = ggml_nelements(tensor);
        size_t bytes_read = fread(tensor->data, sizeof(float), nelements, fin);
        if (bytes_read != nelements) {
            WHISPER_LOG_ERROR("failed to read tensor data for %s: got %zu, expected %zu\n",
                    name, bytes_read, nelements);
            break;
        }

        ggml_set_name(tensor, name);
        model->tensors.push_back(tensor);
        model->tensor_names.push_back(std::string(name));

        model->n_tensors++;
    }

    WHISPER_LOG_INFO("speaker model loaded: %d / %d tensors (%.1f MB)\n",
                     model->n_tensors, n_tensors_expected, ctx_size / 1024.0 / 1024.0);

    if (model->n_tensors != n_tensors_expected) {
        WHISPER_LOG_WARN("loaded %d tensors but expected %d\n", model->n_tensors, n_tensors_expected);
    }

    fclose(fin);
    return model;
}

void whisper_speaker_validate(whisper_speaker_model * model) {
    if (!model) {
        WHISPER_LOG_ERROR("speaker model is nullptr\n");
        return;
    }

    WHISPER_LOG_INFO("speaker model: embedding_dim=%d, n_tensors=%d\n",
                     model->embedding_dim, model->n_tensors);

    if (model->embedding_dim != 192) {
        WHISPER_LOG_WARN("unexpected embedding dimension: %d (expected 192)\n", model->embedding_dim);
    }

    if (model->n_tensors <= 0) {
        WHISPER_LOG_ERROR("no tensors loaded in speaker model\n");
    }
}

int whisper_speaker_get_embedding_dim(whisper_speaker_model * model) {
    return model ? model->embedding_dim : -1;
}

int whisper_speaker_get_tensor_count(whisper_speaker_model * model) {
    return model ? model->n_tensors : -1;
}

struct ggml_tensor * whisper_speaker_get_tensor(struct whisper_speaker_model * model, int idx) {
    if (!model || idx < 0 || idx >= model->n_tensors) {
        return nullptr;
    }
    return model->tensors[idx];
}

struct ggml_tensor * whisper_speaker_find_tensor(struct whisper_speaker_model * model, const char * name) {
    if (!model || !name) return nullptr;
    for (int i = 0; i < model->n_tensors; i++) {
        if (model->tensor_names[i] == name) {
            return model->tensors[i];
        }
    }
    return nullptr;
}

void whisper_speaker_free(whisper_speaker_model * model) {
    if (!model) return;

    if (model->ctx) {
        // Free context (buffer is managed internally by ggml)
        ggml_free(model->ctx);
    }

    delete model;
}
