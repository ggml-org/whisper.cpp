#include "whisper.h"
#include "common-whisper.h"

#include <cstdio>
#include <cfloat>
#include <cstdint>
#include <string>
#include <cstring>

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

static void assert_vad_segments_in_window(
        whisper_context * ctx,
        int offset_ms,
        int duration_ms) {

    const int n_vad = whisper_full_n_vad_segments(ctx);
    assert(n_vad > 0);

    const int64_t window_start_cs = offset_ms / 10;
    const int64_t window_end_cs =
        duration_ms > 0
            ? (offset_ms + duration_ms) / 10
            : INT64_MAX;

    int64_t prev_end = -1;

    for (int i = 0; i < n_vad; ++i) {
        const int64_t t0 = whisper_full_get_vad_segment_t0(ctx, i);
        const int64_t t1 = whisper_full_get_vad_segment_t1(ctx, i);

        assert(t1 > t0);
        assert(t0 >= window_start_cs);
        assert(t1 <= window_end_cs);
        assert(t0 >= prev_end);

        prev_end = t1;
    }
}

int main() {
    ggml_backend_load_all();

    std::string whisper_model_path = WHISPER_MODEL_PATH;
    std::string vad_model_path     = VAD_MODEL_PATH;
    std::string sample_path        = SAMPLE_PATH;

    // Load the sample audio file
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    assert(read_audio_data(sample_path.c_str(), pcmf32, pcmf32s, false));

    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context * wctx = whisper_init_from_file_with_params(
            whisper_model_path.c_str(),
            cparams);

    if (!wctx) {
        fprintf(stderr, "failed to load model '%s'\n", whisper_model_path.c_str());
        return 1;
    }

    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
    wparams.vad            = true;
    wparams.vad_model_path = vad_model_path.c_str();

    wparams.vad_params.threshold               = 0.5f;
    wparams.vad_params.min_speech_duration_ms  = 250;
    wparams.vad_params.min_silence_duration_ms = 100;
    wparams.vad_params.max_speech_duration_s   = FLT_MAX;
    wparams.vad_params.speech_pad_ms           = 30;

    assert(whisper_full_parallel(wctx, wparams, pcmf32.data(), pcmf32.size(), 1) == 0);

    const int n_segments = whisper_full_n_segments(wctx);
    assert(n_segments == 1);


    printf("Segment text:\n%s", whisper_full_get_segment_text(wctx, 0));
    assert(strcmp(" And so my fellow Americans, ask not what your country can do for you,"
                  " ask what you can do for your country.",
           whisper_full_get_segment_text(wctx, 0)) == 0);
    assert(whisper_full_get_segment_t0(wctx, 0) == 32);
    assert(whisper_full_get_segment_t1(wctx, 0) == 1051);

    // token times mapped back to the original timeline: ordered, non-negative duration
    const int n_tokens = whisper_full_n_tokens(wctx, 0);
    assert(n_tokens > 0);
    int64_t prev_t0 = -1;
    for (int j = 0; j < n_tokens; ++j) {
        const int64_t t0 = whisper_full_get_token_t0(wctx, 0, j);
        const int64_t t1 = whisper_full_get_token_t1(wctx, 0, j);
        assert(t0 >= 0 && t1 >= t0);
        assert(t0 >= prev_t0);
        prev_t0 = t0;
    }

    // internal VAD speech segments, on the original audio timeline (centiseconds)
    const int n_vad = whisper_full_n_vad_segments(wctx);
    assert(n_vad > 0);
    int64_t vad_prev_end = -1;
    for (int i = 0; i < n_vad; ++i) {
        const int64_t t0 = whisper_full_get_vad_segment_t0(wctx, i);
        const int64_t t1 = whisper_full_get_vad_segment_t1(wctx, i);
        assert(t1 > t0);
        assert(t0 >= vad_prev_end); // segments are ordered and non-overlapping
        vad_prev_end = t1;
    }

    // non-VAD call after a VAD call on the same context must not use stale VAD state.
    // ref: https://github.com/ggml-org/whisper.cpp/issues/4113
    struct whisper_full_params wparams_novad = wparams;
    wparams_novad.vad = false;

    struct whisper_context * wctx_ref = whisper_init_from_file_with_params(
            whisper_model_path.c_str(),
            cparams);

    assert(wctx_ref);

    for (int processors = 1; processors <= 2; ++processors) {
        assert(whisper_full_parallel(wctx_ref, wparams_novad, pcmf32.data(), pcmf32.size(), processors) == 0);

        assert(whisper_full_parallel(wctx, wparams,       pcmf32.data(), pcmf32.size(), 1)          == 0);
        assert(whisper_full_parallel(wctx, wparams_novad, pcmf32.data(), pcmf32.size(), processors) == 0);
        assert(whisper_full_n_vad_segments(wctx) == 0);

        const int n_segments_ref = whisper_full_n_segments(wctx_ref);
        assert(n_segments_ref > 0);
        assert(whisper_full_n_segments(wctx) == n_segments_ref);

        for (int i = 0; i < n_segments_ref; ++i) {
            assert(whisper_full_get_segment_t0(wctx, i) == whisper_full_get_segment_t0(wctx_ref, i));
            assert(whisper_full_get_segment_t1(wctx, i) == whisper_full_get_segment_t1(wctx_ref, i));
        }
    }


    // Build a 38 s recording from the 11 s jfk sample, so that offset_ms and duration_ms
    // have to be applied on the original timeline and not on the speech-only audio:
    //
    //    0 -  8 s   silence
    //    8 - 19 s   speech A (jfk.wav)
    //   19 - 27 s   silence
    //   27 - 38 s   speech B (jfk.wav)
    std::vector<float> pcmf32_long;
    {
        const std::vector<float> silence(8 * WHISPER_SAMPLE_RATE, 0.0f);
        pcmf32_long.insert(pcmf32_long.end(), silence.begin(), silence.end());
        pcmf32_long.insert(pcmf32_long.end(), pcmf32.begin(), pcmf32.end());
        pcmf32_long.insert(pcmf32_long.end(), silence.begin(), silence.end());
        pcmf32_long.insert(pcmf32_long.end(), pcmf32.begin(), pcmf32.end());
    }

    for (int processors = 1; processors <= 2; ++processors) {
        struct whisper_full_params params = wparams;

        params.offset_ms = 10000;
        params.duration_ms = 0;

        assert(whisper_full_parallel(
            wctx, params, pcmf32_long.data(), pcmf32_long.size(), processors) == 0);

        assert_vad_segments_in_window(wctx, 10000, 0);

        params.offset_ms = 20000;
        params.duration_ms = 0;

        assert(whisper_full_parallel(
            wctx, params, pcmf32_long.data(), pcmf32_long.size(), processors) == 0);

        assert_vad_segments_in_window(wctx, 20000, 0);

        params.offset_ms = 10000;
        params.duration_ms = 10000;

        assert(whisper_full_parallel(
            wctx, params, pcmf32_long.data(), pcmf32_long.size(), processors) == 0);

        assert_vad_segments_in_window(wctx, 10000, 10000);
    }

    whisper_free(wctx_ref);
    whisper_free(wctx);

    return 0;
}
