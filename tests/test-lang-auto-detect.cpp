#include "whisper.h"
#include "common-whisper.h"

#include <cassert>
#include <cstring>
#include <vector>

static void assert_only_candidates_have_probability(const std::vector<float> & probs, int lang_a, int lang_b) {
    for (int i = 0; i < (int) probs.size(); ++i) {
        if (i == lang_a || i == lang_b) {
            assert(probs[i] >= 0.0f);
        } else {
            assert(probs[i] == 0.0f);
        }
    }
}

int main() {
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    assert(read_audio_data(SAMPLE_PATH, pcmf32, pcmf32s, false));

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;

    struct whisper_context * wctx = whisper_init_from_file_with_params(WHISPER_MODEL_PATH, cparams);
    assert(wctx != nullptr);
    assert(whisper_is_multilingual(wctx));
    assert(whisper_pcm_to_mel(wctx, pcmf32.data(), pcmf32.size(), 1) == 0);

    const char * candidates[] = {"fr", "en", "en"};
    std::vector<float> probs(whisper_lang_max_id() + 1, -1.0f);

    const int lang_id = whisper_lang_auto_detect_candidates(wctx, 0, 1, candidates, 3, probs.data());
    assert(lang_id == whisper_lang_id("en"));
    assert(probs[whisper_lang_id("en")] > 0.0f);
    assert(probs[whisper_lang_id("fr")] > 0.0f);
    assert_only_candidates_have_probability(probs, whisper_lang_id("en"), whisper_lang_id("fr"));

    const char * invalid_candidates[] = {"zz"};
    assert(whisper_lang_auto_detect_candidates(wctx, 0, 1, invalid_candidates, 1, nullptr) == -4);

    const char * null_candidate = nullptr;
    assert(whisper_lang_auto_detect_candidates(wctx, 0, 1, &null_candidate, 1, nullptr) == -4);

    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.language = "auto";
    wparams.detect_language = true;
    wparams.language_candidates = candidates;
    wparams.n_language_candidates = 3;

    assert(whisper_full(wctx, wparams, pcmf32.data(), pcmf32.size()) == 0);
    assert(whisper_full_lang_id(wctx) == whisper_lang_id("en"));

    whisper_free(wctx);

    struct whisper_context * wctx_en = whisper_init_from_file_with_params(WHISPER_MODEL_EN_PATH, cparams);
    assert(wctx_en != nullptr);
    assert(!whisper_is_multilingual(wctx_en));
    assert(whisper_pcm_to_mel(wctx_en, pcmf32.data(), pcmf32.size(), 1) == 0);
    assert(whisper_lang_auto_detect_candidates(wctx_en, 0, 1, candidates, 2, nullptr) == -5);

    whisper_free(wctx_en);

    return 0;
}
