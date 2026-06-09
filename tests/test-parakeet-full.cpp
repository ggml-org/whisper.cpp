#include "parakeet.h"
#include "common-whisper.h"
#include "parakeet-verification.h"

#include <cstdio>
#include <string>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

struct test_state {
    bool is_first = true;
    std::string transcript;
};

void progress_callback(parakeet_context * ctx, parakeet_state * state, int progress, void * user_data) {
    bool * called = static_cast<bool *>(user_data);
    *called = true;
}

bool encoder_begin_callback(parakeet_context * ctx, parakeet_state * state, void * user_data) {
    bool * called = static_cast<bool *>(user_data);
    *called = true;
    return true;
}

bool abort_callback(void * user_data) {
    bool * called = static_cast<bool *>(user_data);
    *called = true;
    return false; // just continue without aborting.
}

void token_callback(parakeet_context * ctx, parakeet_state * state, const parakeet_token_data * token_data, void * user_data) {
    test_state * tstate = static_cast<test_state *>(user_data);

    const char * token_str = parakeet_token_to_str(ctx, token_data->id);
    char text_buf[256];
    parakeet_token_to_text(token_str, tstate->is_first, text_buf, sizeof(text_buf));

    printf("%s", text_buf);
    fflush(stdout);

    tstate->transcript += text_buf;
    tstate->is_first = false;
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
    test_state tstate;
    params.new_token_callback = token_callback;
    params.new_token_callback_user_data = &tstate;
    bool progress_callback_called = false;
    params.progress_callback = progress_callback;
    params.progress_callback_user_data = &progress_callback_called;
    bool encoder_begin_callback_called = false;
    params.encoder_begin_callback = encoder_begin_callback;
    params.encoder_begin_callback_user_data = &encoder_begin_callback_called;
    bool abort_callback_called = false;
    params.abort_callback = abort_callback;
    params.abort_callback_user_data = &abort_callback_called;

    int ret = parakeet_full(pctx, params, pcmf32.data(), pcmf32.size());
    assert(ret == 0);
    assert(progress_callback_called);
    assert(encoder_begin_callback_called);
    assert(abort_callback_called);

    const std::string expected = read_expected_transcription(EXPECTED_TRANSCRIPTION_PATH);
    const bool transcript_matches = verify_transcription(expected, tstate.transcript);

    parakeet_free(pctx);

    if (!transcript_matches) {
        return 1;
    }

    printf("\nTest passed: parakeet_full succeeded!\n");
    return 0;
}
