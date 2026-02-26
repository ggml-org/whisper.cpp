#include "vitisai/whisper-vitisai-encoder.h"
#include "FlexMLClient.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
#endif
#include <cstring>
#include <string>

struct whisper_vitisai_context {
    std::string model_path;
    std::shared_ptr<flexmlrt::client::Model> runner;
    uint8_t * fbs_buffer;
    size_t fbs_buffer_size;
};

// Function to mmap rai file for Linux and MapViewOfFile for Windows
static bool map_rai_file(const char * path, uint8_t ** buffer, size_t * size) {
#ifdef _WIN32
    // Open the file
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "%s: %d: Failed to open rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }

    // Get the file size
    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        std::fprintf(stderr, "%s: %d: Failed to get file size for rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }

    // Create a file mapping object
    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, fileSize.QuadPart, NULL);
    if (hMapping == NULL) {
        CloseHandle(hFile);
        std::fprintf(stderr, "%s: %d: Failed to create file mapping for rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }

    // Map the file
    *buffer = (uint8_t *)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, fileSize.QuadPart);
    if (*buffer == NULL) {
        CloseHandle(hMapping);
        CloseHandle(hFile);
        std::fprintf(stderr, "%s: %d: Failed to map rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }
    *size = fileSize.QuadPart;
    return true;
#else
    // Open the file
    FILE * fd = fopen(path, "rb");
    if (!fd) {
        std::fprintf(stderr, "%s: %d: Failed to open rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }

    // Get the file size
    struct stat st;
    if (fstat(fileno(fd), &st) == -1) {
        fclose(fd);
        std::fprintf(stderr, "%s: %d: Failed to get file size for rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }

    // Mmap the file
    *buffer = (uint8_t *)mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fileno(fd), 0);
    if (*buffer == MAP_FAILED) {
        fclose(fd);
        std::fprintf(stderr, "%s: %d: Failed to mmap rai file '%s'\n", __func__, __LINE__, path);
        return false;
    }
    *size = st.st_size;
    return true;
#endif // _WIN32
}

static void unmap_rai_file(uint8_t * buffer, size_t size) {
#ifdef _WIN32
    UnmapViewOfFile(buffer);
#else
    munmap(buffer, size);
#endif // _WIN32
}

struct whisper_vitisai_context * whisper_vitisai_init(const char * path_model) {
    if (!path_model) {
        std::fprintf(stderr, "%s: path_model is null\n", __func__);
        return nullptr;
    }

    auto * ctx = new whisper_vitisai_context;
    ctx->model_path = path_model;

    // Override the model path with the environment variable if it is set
    if (const char * env_model_path = std::getenv("OVERRIDE_VITISAI_MODEL_PATH")) {
        if (env_model_path[0] != '\0') {
            ctx->model_path = env_model_path;
        }
    }

    // Step 1: Set up the model
    flexmlrt::client::Options options;
    options.modelPath = ctx->model_path;
    options.deviceName = "stx";
    options.debug = false;
    options.executeMode = 2;
    options.extOptions["ai_analyzer_profiling"] = true; // Enable AIA profiling
    options.extOptions["enable_preemption"] = true;

    // Check if model_path is rai file and if so, add fbs_buffer and fbs_buffer_size to the options
    if (ctx->model_path.find(".rai") != std::string::npos) {
        // mmap rai file for both Linux and Windows and pass the buffer to the options
        ctx->fbs_buffer = nullptr;
        ctx->fbs_buffer_size = 0;
        if (map_rai_file(ctx->model_path.c_str(), &ctx->fbs_buffer, &ctx->fbs_buffer_size)) {
            options.extOptions["fbs_buffer"] = ctx->fbs_buffer;
            options.extOptions["fbs_buffer_size"] = ctx->fbs_buffer_size;
            options.subgraphName = "vaiml_par_0";
            options.extOptions["cache_dir"] = std::string(".");
        } else {
            std::fprintf(stderr, "%s: Failed to mmap rai file '%s'\n", __func__, ctx->model_path.c_str());
            delete ctx;
            return nullptr;
        }
    }

    try {
        ctx->runner = std::make_shared<flexmlrt::client::Model>(options);

        if (!ctx->runner->good()) {
            throw std::runtime_error("Runner creation ran into an error");
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s: Exception during Vitis AI runner creation: %s\n", __func__, e.what());
        delete ctx;
        return nullptr;
    }
    return ctx;
}

void whisper_vitisai_free(struct whisper_vitisai_context * ctx) {
    if (!ctx) {
        return;
    }

    std::fprintf(stderr, "%s: releasing Vitis AI encoder context for model '%s'\n", __func__, ctx->model_path.c_str());
    if (ctx->fbs_buffer) {
        unmap_rai_file(ctx->fbs_buffer, ctx->fbs_buffer_size);
    }
    delete ctx;
}

int whisper_vitisai_encode(struct whisper_vitisai_context * ctx, struct ggml_tensor * mel, struct ggml_tensor * out) {
    if (!ctx || !mel || !out) {
        std::fprintf(stderr, "%s: ctx/mel/out must not be null\n", __func__);
        return 0;
    }

    if (ggml_n_dims(mel) != 2) {
        std::fprintf(stderr, "%s: mel tensor expected to have 2 dims, got %d\n", __func__, ggml_n_dims(mel));
        return 0;
    }

    if (ggml_n_dims(out) != 2) {
        std::fprintf(stderr, "%s: out tensor expected to have 2 dims, got %d\n", __func__, ggml_n_dims(out));
        return 0;
    }

    // setup input and output tensors for Vitis AI model
    std::vector<flexmlrt::client::ErtTensorType> input_tensors, output_tensors;
    auto model = ctx->runner;

    // Get tensors as CPU tensors (hwTensor = false)
    input_tensors = model->getIOTensors("input", false);
    output_tensors = model->getIOTensors("output", false);

    // TODO: add assert checks for tensor numbers and shapes

    input_tensors[0].data = mel->data;
    output_tensors[0].data = out->data;

    try {
        model->forward(input_tensors, output_tensors);
#if defined(WHISPER_DEBUG)
        std::fprintf(stderr, "%s: Vitis AI model inference completed.\n", __func__);
#endif
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s: Exception during model inference: %s\n", __func__, e.what());
        return 0;
    }

    return 1;
}
