// Regression test for https://github.com/ggml-org/whisper.cpp/issues/3386
//
// When there is not enough free memory to allocate the model weight buffer,
// whisper_model_load() used to silently skip the failed backend buffer and
// continue loading weights into tensors whose ->buffer was still NULL. That
// dereferenced a null buffer while loading weights (and again while freeing the
// model, e.g. ggml_backend_metal_buffer_rset_free), crashing instead of failing
// gracefully.
//
// This test builds a valid-enough in-memory model whose weight buffer is far too
// large to allocate, caps the process address space so the allocation is
// guaranteed to fail, and checks that whisper_init returns NULL instead of
// crashing.

#include "whisper.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__) && !defined(__SANITIZE_ADDRESS__)

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

// hugely oversized model dimensions so the weight buffer cannot be allocated
static const int32_t HUGE_STATE   = 8192; // n_audio_state == n_text_state
static const int32_t N_VOCAB      = 512;
static const int32_t N_AUDIO_CTX  = 1500;
static const int32_t N_TEXT_CTX   = 448;
static const int32_t N_HEAD       = 8;
static const int32_t N_LAYER      = 4;   // MODEL_TINY
static const int32_t N_MELS       = 80;
static const int32_t FTYPE_F16    = 1;   // GGML_FTYPE_MOSTLY_F16
static const int32_t GGML_TYPE_F32_ID = 0;

// address space headroom left for backend init / metadata after capping (bytes);
// far smaller than the multi-GB weight buffer, so only the buffer alloc fails
static const size_t  AS_HEADROOM  = (size_t) 256 * 1024 * 1024;

static void put_i32(std::vector<uint8_t> & b, int32_t v) {
    b.insert(b.end(), (uint8_t *) &v, (uint8_t *) &v + sizeof(v));
}
static void put_u32(std::vector<uint8_t> & b, uint32_t v) {
    b.insert(b.end(), (uint8_t *) &v, (uint8_t *) &v + sizeof(v));
}

// build a whisper model buffer that parses far enough to attempt (and fail) the
// weight buffer allocation, then feeds one real tensor header to reach the code
// path that dereferences the (missing) buffer.
static std::vector<uint8_t> build_oversized_model() {
    std::vector<uint8_t> b;

    put_u32(b, GGML_FILE_MAGIC);

    // hparams
    put_i32(b, N_VOCAB);
    put_i32(b, N_AUDIO_CTX);
    put_i32(b, HUGE_STATE);   // n_audio_state
    put_i32(b, N_HEAD);
    put_i32(b, N_LAYER);      // n_audio_layer
    put_i32(b, N_TEXT_CTX);
    put_i32(b, HUGE_STATE);   // n_text_state (must equal n_audio_state)
    put_i32(b, N_HEAD);
    put_i32(b, N_LAYER);      // n_text_layer
    put_i32(b, N_MELS);
    put_i32(b, FTYPE_F16);

    // mel filters: n_mel, n_fft, then n_mel*n_fft floats
    const int32_t f_n_mel = 80;
    const int32_t f_n_fft = 1;
    put_i32(b, f_n_mel);
    put_i32(b, f_n_fft);
    for (int i = 0; i < f_n_mel * f_n_fft; ++i) {
        put_u32(b, 0); // one float each
    }

    // vocab: count, then [len, bytes] per token
    put_i32(b, N_VOCAB);
    for (int32_t i = 0; i < N_VOCAB; ++i) {
        put_u32(b, 1);                       // token length
        b.push_back((uint8_t) (i & 0xFF));   // token byte
    }

    // one real tensor header: encoder.positional_embedding, F32, [n_audio_state, n_audio_ctx]
    const std::string name = "encoder.positional_embedding";
    put_i32(b, 2);                       // n_dims
    put_i32(b, (int32_t) name.size());   // name length
    put_i32(b, GGML_TYPE_F32_ID);        // ttype
    put_i32(b, HUGE_STATE);              // ne[0]
    put_i32(b, N_AUDIO_CTX);             // ne[1]
    b.insert(b.end(), name.begin(), name.end());
    // no tensor data: load crashes (unfixed) / returns false (fixed) before it is read

    return b;
}

static size_t current_address_space_bytes() {
    // /proc/self/statm field 1 = total program size, in pages
    FILE * f = fopen("/proc/self/statm", "r");
    if (!f) {
        return 0;
    }
    unsigned long pages = 0;
    if (fscanf(f, "%lu", &pages) != 1) {
        pages = 0;
    }
    fclose(f);
    return (size_t) pages * (size_t) sysconf(_SC_PAGESIZE);
}

int main() {
    std::vector<uint8_t> model = build_oversized_model();

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    }

    if (pid == 0) {
        // child: cap the address space so the weight buffer allocation must fail,
        // then attempt to load the model. Never crash if the fix is present.
        size_t cap = current_address_space_bytes() + AS_HEADROOM;
        struct rlimit rl;
        rl.rlim_cur = cap;
        rl.rlim_max = cap;
        if (setrlimit(RLIMIT_AS, &rl) != 0) {
            perror("setrlimit");
            _exit(3);
        }

        struct whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = false;

        struct whisper_context * ctx =
            whisper_init_from_buffer_with_params(model.data(), model.size(), cparams);

        if (ctx == nullptr) {
            _exit(0); // graceful failure — expected with the fix
        }
        whisper_free(ctx);
        _exit(2); // unexpectedly succeeded (buffer somehow allocated)
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        return 1;
    }

    if (WIFSIGNALED(status)) {
        printf("test-whisper-oom-graceful: RED — model load crashed with signal %d on allocation failure\n",
               WTERMSIG(status));
        return 1;
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        printf("test-whisper-oom-graceful: GREEN — whisper_init returned NULL gracefully on allocation failure\n");
        return 0;
    }

    printf("test-whisper-oom-graceful: FAIL — unexpected child status (exit code %d)\n",
           WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    return 1;
}

#else // not Linux, or built with AddressSanitizer (incompatible with RLIMIT_AS)

int main() {
    printf("test-whisper-oom-graceful: SKIP — requires Linux without AddressSanitizer\n");
    return 0;
}

#endif
