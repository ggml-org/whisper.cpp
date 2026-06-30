#pragma once

// Voice-cloning preprocessing primitives: WAV I/O, resampling, loudness
// normalisation, mel extraction.  Together with the VoiceEncoder / CAMPPlus
// / S3TokenizerV2 ports in voice_encoder.{h,cpp}, campplus.{h,cpp}, and
// s3tokenizer.{h,cpp}, these cover every tensor Chatterbox needs for voice
// cloning — no Python runtime dependency.
//
// Exposes:
//   wav_load            — dr_wav-based multi-format WAV reader → mono f32
//   resample_sinc       — Kaiser-windowed sinc resampler
//   measure_lufs        — ITU-R BS.1770-4 loudness metering
//   normalise_lufs      — in-place gain to a target LUFS
//   mel_extract_24k_80  — 80-ch log-mel @ 24 kHz (S3Gen prompt_feat)
//   mel_extract_16k_40  — 40-ch power mel @ 16 kHz (VoiceEncoder input)
//   fbank_kaldi_80      — Kaldi-style fbank @ 16 kHz (CAMPPlus input)

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// WAV I/O
// -----------------------------------------------------------------------------

// Load a WAV file into a mono float32 buffer in [-1, 1].  Stereo / multi-channel
// inputs are averaged down to mono.  Returns false on IO or format errors.
// out_sr is the file's sample rate (may require resampling before use).
bool wav_load(const std::string & path,
              std::vector<float> & out_samples,
              int & out_sr);

// -----------------------------------------------------------------------------
// Resampling (rational-ratio windowed-sinc, Kaiser window, beta = 8.6 ~= -90 dB)
// -----------------------------------------------------------------------------

// Resample `in` from `sr_in` to `sr_out` using Kaiser-windowed sinc
// interpolation.  Quality is comparable to librosa's `res_type='kaiser_fast'`
// (32 taps) — good enough for voice preprocessing, not optimised for real-time.
std::vector<float> resample_sinc(const std::vector<float> & in,
                                 int sr_in, int sr_out,
                                 int taps_half = 16);

// -----------------------------------------------------------------------------
// Output-frequency selection (QVAC-21483)
// -----------------------------------------------------------------------------

// Supported explicit output sample-rate window, in Hz.  A value of 0 is also
// accepted everywhere and means "keep the engine's native rate".  The bounds
// mirror the @qvac/tts-ggml addon's JS-side validation so the contract is
// identical from the JS API down through the C++ engines and CLIs.
constexpr int kOutputSampleRateMin = 8000;
constexpr int kOutputSampleRateMax = 192000;

// Validate a requested output sample rate.  Accepts 0 (keep native) or any
// rate in [kOutputSampleRateMin, kOutputSampleRateMax]; throws
// std::runtime_error (prefixed with `who`) on anything else so callers fail
// fast at construction / arg-parse time rather than emitting a malformed wav.
void validate_output_sample_rate(int sr, const char * who = "tts-cpp");

// Resample final synthesized `pcm` from `native_sr` to `target_sr`.  Returns
// `pcm` unchanged (moved through) when `target_sr <= 0` or
// `target_sr == native_sr` — the "keep native rate" fast path — otherwise
// delegates to resample_sinc.  Centralises the output-frequency policy shared
// by the Chatterbox and Supertonic engines so both behave identically.
std::vector<float> resample_for_output(std::vector<float> pcm,
                                       int native_sr, int target_sr);

// Stateful streaming output-frequency converter (QVAC-21483).
//
// `resample_sinc` is stateless and whole-buffer: resampling each streaming
// chunk independently restarts the output grid at t=0 and truncates the sinc
// window at every chunk edge, so concatenating per-chunk resamples introduces a
// discontinuity at every seam (and the total length drifts) unless each chunk
// happens to be an exact multiple of the resample ratio's denominator — which
// the synthesis pipeline does not guarantee (e.g. the trim-faded first chunk,
// the finalized last chunk, or output rates such as 11025 Hz).
//
// OutputResampler removes those seams: feed native-rate chunks via `process()`
// and it returns only the output samples whose sinc window is fully covered by
// the native audio seen so far (the "stable" prefix); `finish()` flushes the
// final right-edge samples once no more input is coming.  Because a stable
// sample's window is identical whether computed on the prefix or on the full
// buffer, the concatenation of every `process()` result followed by `finish()`
// is bit-for-bit identical to resample_for_output(<all native samples>) — the
// streamed output equals the batch resample, with no per-chunk artifacts and no
// length drift.  When `target_sr` is 0 or equals `native_sr` the converter is a
// passthrough: `process()` returns its input unchanged and `finish()` is empty.
class OutputResampler {
public:
    OutputResampler(int native_sr, int target_sr);

    // Append `native_chunk` and return the newly-stable output samples.
    std::vector<float> process(const std::vector<float> & native_chunk);

    // Flush the remaining (right-edge) output samples.  Call once, after the
    // last process(); further process()/finish() calls return nothing new.
    std::vector<float> finish();

    bool passthrough() const { return passthrough_; }
    int  target_rate() const { return target_sr_; }   // resolved (== native when passthrough)

private:
    int                native_sr_;
    int                target_sr_;
    bool               passthrough_;
    int                taps_half_;   // mirrors resample_sinc's window half-width
    std::vector<float> native_;      // every native sample seen so far
    std::size_t        emitted_ = 0; // output samples already returned
};

// -----------------------------------------------------------------------------
// Loudness normalisation (ITU-R BS.1770-4 / EBU R 128)
// -----------------------------------------------------------------------------

// Measure the integrated loudness of a mono float32 signal in LUFS, using the
// K-weighting filter + 400 ms gated blocks described in ITU-R BS.1770-4.
// Matches `pyloudnorm.Meter(sr).integrated_loudness(wav)`.
//
// Returns -std::numeric_limits<double>::infinity() when the signal is too
// short or all blocks fall below the -70 LUFS absolute gate.
double measure_lufs(const std::vector<float> & wav, int sr);

// Normalise `wav` to `target_lufs` (default -27, which is what
// chatterbox.tts_turbo.ChatterboxTurboTTS.norm_loudness uses).  Modifies
// `wav` in-place.  No-op if the measured loudness is ±∞ or the gain is NaN.
void normalise_lufs(std::vector<float> & wav, int sr, double target_lufs = -27.0);

// -----------------------------------------------------------------------------
// Mel extraction
// -----------------------------------------------------------------------------

// Compute the 80-channel log-mel spectrogram at 24 kHz that S3Gen uses as
// `prompt_feat`, matching chatterbox.models.s3gen.utils.mel.mel_spectrogram:
//
//   n_fft=1920  hop=480  win=1920  fmin=0  fmax=8000  center=False
//   reflect-pad by (n_fft - hop) / 2 = 720 each side
//   magnitude (with 1e-9 floor), mel filterbank matmul, log(clip(x, 1e-5))
//
// `mel_filterbank` must be the (80 * 961 = 76880)-element filterbank that
// librosa.filters.mel produces for those parameters, flattened row-major (80
// rows of 961 columns).  It gets baked into the s3gen GGUF by
// convert-s3gen-to-gguf.py.
//
// Returns a row-major (T_mel, 80) tensor, where T_mel = (L_wav + 2*720 - 1920) / 480 + 1.
std::vector<float> mel_extract_24k_80(const std::vector<float> & wav_24k,
                                      const std::vector<float> & mel_filterbank);

// Compute the 40-channel mel *power* spectrogram at 16 kHz that VoiceEncoder
// consumes, matching voice_encoder.melspec.melspectrogram with its default
// hyperparameters:
//
//   n_fft=400  hop=160  win=400  center=True (reflect-pad n_fft/2 each side)
//   mel_power = 2 (POWER, not magnitude)
//   mel_type = "amp" (no log / db conversion)
//   normalized_mels = False
//
// `mel_filterbank` must be the librosa (40, 201)=8040-element filterbank
// produced with those parameters; convert-t3-turbo-to-gguf.py bakes it in as
// `voice_encoder/mel_fb`.
//
// Returns a row-major (T_mel, 40) tensor, where T_mel = 1 + L_wav / hop.
std::vector<float> mel_extract_16k_40(const std::vector<float> & wav_16k,
                                      const std::vector<float> & mel_filterbank);

// Kaldi-style 80-channel log-power mel filterbank at 16 kHz (matches
// torchaudio.compliance.kaldi.fbank with its default arguments + dither=0).
// Parameters baked in:
//
//   frame_length = 25 ms (400 samples)    frame_shift = 10 ms (160 samples)
//   num_mel_bins = 80                     low_freq    = 20 Hz
//   high_freq    = 8000 Hz (nyquist)      preemphasis = 0.97
//   window_type  = "povey" (0.85-exponent Hann)
//   round_to_power_of_two = True → n_fft = 512
//   remove_dc_offset = True, snip_edges = True
//   use_power = True, use_log_fbank = True, dither = 0
//   signed_16bit_max = 32768 (Kaldi int16 scaling)
//
// `mel_filterbank` must be the (80, 257)-element Kaldi filterbank baked into
// the s3gen GGUF as `campplus/mel_fb_kaldi_80`.
//
// Returns a row-major (T, 80) log-mel tensor,
// T = (L_wav - 400) / 160 + 1.
std::vector<float> fbank_kaldi_80(const std::vector<float> & wav_16k,
                                  const std::vector<float> & mel_filterbank);
