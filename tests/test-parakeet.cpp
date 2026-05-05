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

void segment_callback(parakeet_context * ctx, parakeet_state * state, int n_new, void * user_data) {
    const int n_segments = parakeet_full_n_segments_from_state(state);
    const int s0 = n_segments - n_new;

    printf("\nSegment Callback: %d new segment(s)\n", n_new);

    for (int i = s0; i < n_segments; i++) {
        const char * text = parakeet_full_get_segment_text_from_state(state, i);
        const int64_t t0 = parakeet_full_get_segment_t0_from_state(state, i);
        const int64_t t1 = parakeet_full_get_segment_t1_from_state(state, i);

        printf("Segment %d: [%lld -> %lld] \"%s\"\n", i, (long long)t0, (long long)t1, text);
        printf("Tokens:\n");

        const int n_tokens = parakeet_full_n_tokens_from_state(state, i);
        for (int j = 0; j < n_tokens; j++) {
            parakeet_token_data token_data = parakeet_full_get_token_data_from_state(state, i, j);
            const char * token_str = parakeet_token_to_str(ctx, token_data.id);

            printf("  [%2d] id=%5d frame=%3d dur_idx=%2d dur_val=%2d p=%.4f plog=%.4f t0=%4lld t1=%4lld word_start=%d \"%s\"\n",
                   j,
                   token_data.id,
                   token_data.frame_index,
                   token_data.duration_idx,
                   token_data.duration_value,
                   token_data.p,
                   token_data.plog,
                   (long long)token_data.t0,
                   (long long)token_data.t1,
                   token_data.is_word_start,
                   token_str);
        }
    }
    printf("\n");
}

int main() {
    std::string model_path  = PARAKEET_MODEL_PATH;
    std::string sample_path = SAMPLE_PATH;

    // Load the sample audio file
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    assert(read_audio_data(sample_path.c_str(), pcmf32, pcmf32s, false));
    assert(pcmf32.size() > 0);
    assert(pcmf32s.size() == 0);

    printf("Loading Parakeet model from: %s\n", model_path.c_str());

    struct parakeet_context_params ctx_params = parakeet_context_default_params();

    struct parakeet_context * pctx = parakeet_init_from_file_with_params_no_state(model_path.c_str(), ctx_params);
    if (pctx == nullptr) {
        fprintf(stderr, "Failed to load Parakeet model\n");
        return 1;
    }
    printf("Successfully loaded Parakeet model\n");

    struct parakeet_full_params params = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
    params.new_token_callback = token_callback;
    params.new_token_callback_user_data = nullptr;
    params.new_segment_callback = segment_callback;
    params.new_segment_callback_user_data = nullptr;
    parakeet_state * state = parakeet_init_state(pctx);

    int ret = parakeet_chunk(pctx, state, params, pcmf32.data(), pcmf32.size());
    assert(ret == 0);

    parakeet_free_state(state);
    parakeet_free(pctx);

    printf("\nTest passed: Parakeet model loaded and freed successfully\n");
    return 0;
}
