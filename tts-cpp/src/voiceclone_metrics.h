#pragma once

// Speaker-similarity + WER metrics for the voice-clone test harness
// (QVAC-20979).
//
// Three families of metric, all of them pure host arithmetic so they run in the
// always-on `unit` test tier (no model, no fixture download required):
//
//   * cosine_similarity   - the building block of every speaker-verification
//                           score in the design (CAMPPlus on-device; WavLM-SV /
//                           ECAPA-TDNN / ResNet off-device).  The *embeddings*
//                           come from those models; the *score* is plain cosine.
//
//   * wavlm_style_loss    - the perceptual optimization objective from Kim,
//                           "Extracting Voice Styles from Frozen TTS Models via
//                           Gradient-Based Inverse Optimization": time-averaged
//                           mean (mu) and std (sigma) over WavLM layer-3 hidden
//                           states, compared with squared-L2.  The WavLM forward
//                           is off-device; this module consumes the hidden-state
//                           matrix it produces.
//
//   * word_error_rate     - Levenshtein edit distance over word tokens, used to
//                           guard intelligibility (Whisper transcription vs the
//                           ground-truth text).  Whisper supplies the hypothesis;
//                           this module scores it.

#include <string>
#include <vector>

namespace tts_cpp {
namespace voiceclone {

// Cosine similarity of two equal-length embeddings.  Returns a value in
// [-1, 1].  A zero-norm input has no direction, so similarity is defined as 0
// rather than NaN (matches how downstream averaging treats a silent clip).
// Throws std::invalid_argument on size mismatch or empty input.
double cosine_similarity(const std::vector<float> & a, const std::vector<float> & b);

// Time-averaged statistics of a hidden-state matrix.
//   mu[d]    = mean_t  H[t, d]
//   sigma[d] = sqrt( mean_t (H[t, d] - mu[d])^2 )   (population std, 1/T)
// Population (not sample) std is used so a single-frame clip yields sigma = 0
// instead of a divide-by-zero, and so the statistic is independent of T.
struct TimeAvgStats {
    std::vector<double> mu;
    std::vector<double> sigma;
};

// H is row-major [T, D] (frame-major, the layout WavLM / npy dumps produce).
// Throws std::invalid_argument if T or D is non-positive or the data size is not
// exactly T*D.
TimeAvgStats time_avg_stats(const std::vector<float> & H_t_by_d, int T, int D);

// Perceptual style loss: ||mu_a - mu_b||^2 + ||sigma_a - sigma_b||^2.
double wavlm_style_loss(const TimeAvgStats & a, const TimeAvgStats & b);

// Convenience overload computing the loss directly from two hidden-state
// matrices that share the feature dimension D but may differ in length.
double wavlm_style_loss(const std::vector<float> & Ha, int Ta,
                        const std::vector<float> & Hb, int Tb,
                        int D);

struct WerResult {
    int    substitutions = 0;
    int    deletions     = 0;
    int    insertions    = 0;
    int    ref_len       = 0;
    double wer           = 0.0;  // (S + D + I) / ref_len
};

// Word error rate via Levenshtein alignment over token sequences.  Reports S / D
// / I separately (not just the total distance) so the harness can distinguish a
// model that drops words from one that hallucinates them.  An empty reference is
// error-free only when the hypothesis is empty too; otherwise every hypothesis
// token is an insertion and `wer` is reported as the insertion count.
WerResult word_error_rate(const std::vector<std::string> & ref,
                          const std::vector<std::string> & hyp);

// Whitespace tokenization convenience overload.
WerResult word_error_rate(const std::string & ref, const std::string & hyp);

// Splits on runs of whitespace; no case-folding or punctuation stripping (the
// caller normalizes upstream so the metric stays a pure alignment).
std::vector<std::string> split_whitespace(const std::string & s);

}  // namespace voiceclone
}  // namespace tts_cpp
