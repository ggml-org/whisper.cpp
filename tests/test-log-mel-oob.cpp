#include "whisper.h"

#include <string>
#include <vector>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

// Regression test for the heap out-of-bounds read in log_mel_spectrogram().
// Reflective padding reads samples[1 .. WHISPER_N_FFT/2], so an input shorter
// than WHISPER_N_FFT/2 + 1 (201) samples used to read past the end of the
// caller's buffer. Such inputs must now be rejected with a non-zero return.

int main() {
    std::string model_path = WHISPER_MODEL_PATH;

    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context * ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    assert(ctx != nullptr);

    std::vector<float> samples(256, 0.0f);

    // Shorter than the reflective-pad window: must be rejected, not read OOB.
    assert(whisper_pcm_to_mel(ctx, samples.data(),   4, 1) != 0);
    assert(whisper_pcm_to_mel(ctx, samples.data(), 200, 1) != 0);

    // Long enough to pad safely: must still succeed.
    assert(whisper_pcm_to_mel(ctx, samples.data(), 201, 1) == 0);

    whisper_free(ctx);

    return 0;
}
