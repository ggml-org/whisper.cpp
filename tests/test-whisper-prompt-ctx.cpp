#include "whisper.h"
#include "common-whisper.h"

#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

struct PromptRecorder {
    std::vector<std::vector<whisper_token>> window_prompts;
    bool window_recorded = false;
};

static bool encoder_begin_cb(struct whisper_context * ctx, struct whisper_state * state, void * user_data) {
    (void) ctx;
    (void) state;
    auto * rec = (PromptRecorder *) user_data;
    rec->window_recorded = false;
    return true;
}

static void logits_filter_cb(
        struct whisper_context * ctx,
        struct whisper_state * state,
        const whisper_token_data * tokens,
        int n_tokens,
        float * logits,
        void * user_data) {
    (void) ctx;
    (void) tokens;
    (void) n_tokens;
    (void) logits;
    auto * rec = (PromptRecorder *) user_data;
    if (!rec->window_recorded) {
        rec->window_recorded = true;
        const int n = whisper_full_n_prompt_tokens_from_state(state);
        std::vector<whisper_token> p(n);
        for (int i = 0; i < n; ++i) {
            p[i] = whisper_full_get_prompt_token_from_state(state, i);
        }
        rec->window_prompts.push_back(p);
    }
}

static bool contains_subsequence(const std::vector<whisper_token> & vec, const std::vector<whisper_token> & sub) {
    if (sub.empty() || sub.size() > vec.size()) return false;
    return std::search(vec.begin(), vec.end(), sub.begin(), sub.end()) != vec.end();
}

int main() {
    ggml_backend_load_all();

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;

    struct whisper_context * ctx = whisper_init_from_file_with_params(WHISPER_MODEL_PATH, cparams);
    assert(ctx != nullptr);

    // Load sample audio and repeat it to guarantee at least 2 decode windows (> 30s)
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    assert(read_audio_data(SAMPLE_PATH, pcmf32, pcmf32s, false));
    assert(!pcmf32.empty());

    while (pcmf32.size() < 35 * WHISPER_SAMPLE_RATE) {
        pcmf32.insert(pcmf32.end(), pcmf32.begin(), pcmf32.end());
    }

    const char * prompt_text = "WhisperPromptRegressionTest";
    std::vector<whisper_token> prompt_tokens(128);
    const int n_prompt = whisper_tokenize(ctx, prompt_text, prompt_tokens.data(), prompt_tokens.size());
    assert(n_prompt > 0);
    prompt_tokens.resize(n_prompt);

    // Test 1: carry_initial_prompt = false
    // Prompt must be present on first window and absent on subsequent windows
    {
        PromptRecorder rec;
        struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        params.initial_prompt         = prompt_text;
        params.carry_initial_prompt   = false;
        params.n_max_text_ctx         = 64;
        params.print_progress         = false;
        params.print_realtime         = false;
        params.encoder_begin_callback           = encoder_begin_cb;
        params.encoder_begin_callback_user_data = &rec;
        params.logits_filter_callback           = logits_filter_cb;
        params.logits_filter_callback_user_data = &rec;

        const int rc = whisper_full(ctx, params, pcmf32.data(), pcmf32.size());
        assert(rc == 0);
        assert(rec.window_prompts.size() >= 2);

        // Window 1: must contain initial prompt tokens
        assert(contains_subsequence(rec.window_prompts[0], prompt_tokens));

        // Window 2: must NOT contain initial prompt tokens
        assert(!contains_subsequence(rec.window_prompts[1], prompt_tokens));
        printf("Test 1 (carry disabled): OK\n");
    }

    // Test 2a: n_max_text_ctx = 0 with carry_initial_prompt = false (#4098)
    // Initial prompt must be preserved on first window even when context is 0
    {
        PromptRecorder rec;
        struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        params.initial_prompt         = prompt_text;
        params.carry_initial_prompt   = false;
        params.n_max_text_ctx         = 0;
        params.print_progress         = false;
        params.print_realtime         = false;
        params.encoder_begin_callback           = encoder_begin_cb;
        params.encoder_begin_callback_user_data = &rec;
        params.logits_filter_callback           = logits_filter_cb;
        params.logits_filter_callback_user_data = &rec;

        const int rc = whisper_full(ctx, params, pcmf32.data(), pcmf32.size());
        assert(rc == 0);
        assert(rec.window_prompts.size() >= 2);

        // Window 1: must preserve initial prompt
        assert(contains_subsequence(rec.window_prompts[0], prompt_tokens));

        // Window 2: carry is off and context is 0 -> no previous prompt or context
        assert(!contains_subsequence(rec.window_prompts[1], prompt_tokens));
        assert(rec.window_prompts[1].front() != whisper_token_prev(ctx));
        printf("Test 2a (n_max_text_ctx = 0, carry off): OK\n");
    }

    // Test 2b: n_max_text_ctx = 0 with carry_initial_prompt = true
    // Initial prompt must be preserved on ALL windows
    {
        PromptRecorder rec;
        struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        params.initial_prompt         = prompt_text;
        params.carry_initial_prompt   = true;
        params.n_max_text_ctx         = 0;
        params.print_progress         = false;
        params.print_realtime         = false;
        params.encoder_begin_callback           = encoder_begin_cb;
        params.encoder_begin_callback_user_data = &rec;
        params.logits_filter_callback           = logits_filter_cb;
        params.logits_filter_callback_user_data = &rec;

        const int rc = whisper_full(ctx, params, pcmf32.data(), pcmf32.size());
        assert(rc == 0);
        assert(rec.window_prompts.size() >= 2);

        // Window 1: initial prompt present
        assert(contains_subsequence(rec.window_prompts[0], prompt_tokens));

        // Window 2: initial prompt preserved because carry is on
        assert(contains_subsequence(rec.window_prompts[1], prompt_tokens));
        printf("Test 2b (n_max_text_ctx = 0, carry on): OK\n");
    }

    // Test 3: History bounded by n_max_text_ctx
    {
        const int bound = 7;
        PromptRecorder rec;
        struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        params.initial_prompt         = prompt_text;
        params.carry_initial_prompt   = false;
        params.n_max_text_ctx         = bound;
        params.print_progress         = false;
        params.print_realtime         = false;
        params.encoder_begin_callback           = encoder_begin_cb;
        params.encoder_begin_callback_user_data = &rec;
        params.logits_filter_callback           = logits_filter_cb;
        params.logits_filter_callback_user_data = &rec;

        const int rc = whisper_full(ctx, params, pcmf32.data(), pcmf32.size());
        assert(rc == 0);
        assert(rec.window_prompts.size() >= 2);

        // In window 2, carry is off, so tokens before _SOT_ are dynamic history tokens
        const auto & p2 = rec.window_prompts[1];
        auto sot_it = std::find(p2.begin(), p2.end(), whisper_token_sot(ctx));
        assert(sot_it != p2.end());
        int history_tokens = 0;
        if (!p2.empty() && p2.front() == whisper_token_prev(ctx)) {
            history_tokens = std::distance(p2.begin() + 1, sot_it);
        }
        assert(history_tokens <= bound);
        printf("Test 3 (history bounded by %d tokens, actual = %d): OK\n", bound, history_tokens);
    }

    whisper_free(ctx);

    printf("test-whisper-prompt-ctx: ALL TESTS PASSED\n");
    return 0;
}
