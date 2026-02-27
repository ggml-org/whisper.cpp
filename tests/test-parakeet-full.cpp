#include "parakeet.h"
#include "common-whisper.h"

#include <cstdio>
#include <string>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

void token_callback(parakeet_context * ctx, parakeet_state * state, const parakeet_token_data * token_data, void * user_data) {
    static bool is_first = true;
    const char * token_str = parakeet_token_to_str(ctx, token_data->id);
    char text_buf[256];
    parakeet_token_to_text(token_str, is_first, text_buf, sizeof(text_buf));

    int32_t time_ms = token_data->frame_index * 10;

    printf("%s", text_buf);
    fflush(stdout);

    is_first = false;
}

int main() {
    std::string model_path  = PARAKEET_MODEL_PATH;
    std::string sample_path = SAMPLE_PATH;

    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    assert(read_audio_data(sample_path.c_str(), pcmf32, pcmf32s, false));
    assert(pcmf32.size() > 0);
    assert(pcmf32s.size() == 0); // no stereo vector

    printf("Loading Parakeet model from: %s\n", model_path.c_str());

    struct parakeet_context_params ctx_params = parakeet_context_default_params();

    struct parakeet_context * pctx = parakeet_init_from_file_with_params(model_path.c_str(), ctx_params);
    if (pctx == nullptr) {
        fprintf(stderr, "Failed to load Parakeet model\n");
        return 1;
    }
    printf("Successfully loaded Parakeet model\n");

    struct parakeet_full_params params = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
    params.new_token_callback = token_callback;
    params.new_token_callback_user_data = nullptr;

    params.chunk_length_ms  = 10000;
    params.left_context_ms  = 10000;
    params.right_context_ms = 4960;

    int ret = parakeet_full(pctx, params, pcmf32.data(), pcmf32.size());
    assert(ret == 0);

    parakeet_free(pctx);

    printf("\nTest passed: parakeet_full_parallel succeeded!\n");
    return 0;
}
