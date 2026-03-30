#include "whisper-speaker.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

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
        fprintf(stderr, "Invalid GGML magic: 0x%x (expected 0x67676d6c)\n", magic);
        fclose(fin);
        return nullptr;
    }
    printf("GGML magic valid: 0x%08x\n", magic);

    // Read model type string (length-prefixed UTF-8)
    int str_len;
    if (fread(&str_len, sizeof(str_len), 1, fin) != 1) {
        fprintf(stderr, "Failed to read model type length\n");
        fclose(fin);
        return nullptr;
    }

    if (str_len < 0 || str_len > 256) {
        fprintf(stderr, "Invalid model type length: %d\n", str_len);
        fclose(fin);
        return nullptr;
    }

    char model_type[257];
    if (fread(model_type, str_len, 1, fin) != 1) {
        fprintf(stderr, "Failed to read model type\n");
        fclose(fin);
        return nullptr;
    }
    model_type[str_len] = '\0';
    printf("Model type: %s\n", model_type);

    // Read version (major, minor, patch)
    int major, minor, patch;
    if (fread(&major, sizeof(major), 1, fin) != 1 ||
        fread(&minor, sizeof(minor), 1, fin) != 1 ||
        fread(&patch, sizeof(patch), 1, fin) != 1) {
        fprintf(stderr, "Failed to read version\n");
        fclose(fin);
        return nullptr;
    }
    printf("Version: %d.%d.%d\n", major, minor, patch);

    // Read hyperparameters
    int embedding_dim;
    if (fread(&embedding_dim, sizeof(embedding_dim), 1, fin) != 1) {
        fprintf(stderr, "Failed to read embedding_dim\n");
        fclose(fin);
        return nullptr;
    }
    printf("Embedding dimension: %d\n", embedding_dim);

    int n_channels;
    if (fread(&n_channels, sizeof(n_channels), 1, fin) != 1) {
        fprintf(stderr, "Failed to read n_channels\n");
        fclose(fin);
        return nullptr;
    }
    printf("Internal channels: %d\n", n_channels);

    // Read tensor count (for verification)
    int n_tensors_expected;
    if (fread(&n_tensors_expected, sizeof(n_tensors_expected), 1, fin) != 1) {
        fprintf(stderr, "Failed to read tensor count\n");
        fclose(fin);
        return nullptr;
    }
    printf("Expected tensors: %d\n", n_tensors_expected);

    // Create ggml context with sufficient memory for all tensors
    // ~500 MB should accommodate ECAPA-TDNN model weights
    size_t ctx_size = 500 * 1024 * 1024;  // 500 MB
    struct ggml_init_params ggml_params = {
        .mem_size = ctx_size,
        .mem_buffer = malloc(ctx_size),
        .no_alloc = false,
    };

    if (!ggml_params.mem_buffer) {
        fprintf(stderr, "Failed to allocate GGML context buffer\n");
        fclose(fin);
        return nullptr;
    }

    struct ggml_context * ctx = ggml_init(ggml_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create ggml context\n");
        free(ggml_params.mem_buffer);
        fclose(fin);
        return nullptr;
    }
    printf("GGML context created (%zu bytes)\n", ctx_size);

    // Create speaker model structure
    whisper_speaker_model * model = new whisper_speaker_model();
    model->ctx = ctx;
    model->embedding_dim = embedding_dim;
    model->n_tensors = 0;

    printf("\nLoading tensors:\n");

    // Load tensors
    for (int t = 0; t < n_tensors_expected; ++t) {
        // Read tensor header: n_dims, name_len
        int n_dims;
        if (fread(&n_dims, sizeof(n_dims), 1, fin) != 1) {
            fprintf(stderr, "Failed to read n_dims for tensor %d\n", t);
            break;
        }

        int name_len;
        if (fread(&name_len, sizeof(name_len), 1, fin) != 1) {
            fprintf(stderr, "Failed to read name_len for tensor %d\n", t);
            break;
        }

        // Sanity checks
        if (n_dims < 0 || n_dims > 8) {
            fprintf(stderr, "Invalid n_dims for tensor %d: %d\n", t, n_dims);
            break;
        }
        if (name_len < 0 || name_len > 512) {
            fprintf(stderr, "Invalid name_len for tensor %d: %d\n", t, name_len);
            break;
        }

        // Read dimensions (convert to int64_t)
        int64_t dims[8] = {0};
        for (int i = 0; i < n_dims; ++i) {
            int dim;
            if (fread(&dim, sizeof(int), 1, fin) != 1) {
                fprintf(stderr, "Failed to read dim %d for tensor %d\n", i, t);
                break;
            }
            dims[i] = (int64_t)dim;
        }

        // Read tensor name (not null-terminated in binary)
        char name[513];
        if (fread(name, name_len, 1, fin) != 1) {
            fprintf(stderr, "Failed to read tensor name for tensor %d\n", t);
            break;
        }
        name[name_len] = '\0';

        // Create tensor in ggml context
        struct ggml_tensor * tensor = ggml_new_tensor(ctx, GGML_TYPE_F32, n_dims, dims);
        if (!tensor) {
            fprintf(stderr, "Failed to create tensor: %s\n", name);
            break;
        }

        // Read tensor data (float32)
        size_t nelements = ggml_nelements(tensor);
        size_t bytes_read = fread(tensor->data, sizeof(float), nelements, fin);
        if (bytes_read != nelements) {
            fprintf(stderr, "Failed to read tensor data for %s: got %zu, expected %zu\n",
                    name, bytes_read, nelements);
            break;
        }

        ggml_set_name(tensor, name);
        model->tensors.push_back(tensor);
        model->tensor_names.push_back(std::string(name));

        printf("  [%d] %s: ", t + 1, name);
        for (int i = 0; i < n_dims; ++i) {
            printf("%lld", (long long)dims[i]);
            if (i < n_dims - 1) printf("x");
        }
        printf(" (%zu elements, %.2f MB)\n", nelements, (nelements * sizeof(float)) / 1024.0 / 1024.0);

        model->n_tensors++;
    }

    printf("\nModel loaded: %d / %d tensors\n", model->n_tensors, n_tensors_expected);

    if (model->n_tensors != n_tensors_expected) {
        fprintf(stderr, "Warning: Loaded %d tensors but expected %d\n", model->n_tensors, n_tensors_expected);
    }

    fclose(fin);
    return model;
}

void whisper_speaker_validate(whisper_speaker_model * model) {
    if (!model) {
        fprintf(stderr, "Error: Model is nullptr\n");
        return;
    }

    printf("\n=== Model Validation ===\n");
    printf("Embedding dimension: %d\n", model->embedding_dim);
    printf("Total tensors loaded: %d\n", model->n_tensors);

    if (model->ctx) {
        printf("Context allocated\n");
    }

    if (model->embedding_dim == 192) {
        printf("Embedding dimension correct (192)\n");
    } else {
        printf("WARNING: Embedding dimension unexpected: %d (expected 192)\n", model->embedding_dim);
    }

    if (model->n_tensors > 0) {
        printf("Model structure valid (%d tensors)\n", model->n_tensors);
    } else {
        printf("ERROR: No tensors loaded\n");
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
