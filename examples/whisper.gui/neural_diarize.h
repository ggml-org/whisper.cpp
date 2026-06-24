// Bridge from the C++ GUI to the Python neural-diarization helper (diarize.py,
// which uses sherpa-onnx). Rather than linking onnxruntime into the GUI, we run
// the already-verified helper as a subprocess: write the segment timestamps it
// expects, run it, and read the assigned speaker ids back.

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Run the neural-diarization helper on already-decoded 16 kHz mono audio for the
// given segment time spans (in centiseconds, as whisper reports t0/t1). The PCM
// is written to a temp 16-bit WAV so the helper sees exactly the audio whisper
// transcribed, regardless of the original input format. On success, out_speakers
// holds a 0-based speaker id per span (-1 if that span got no speaker) and the
// function returns true. On failure it returns false and sets `error`.
//
//   python       - python interpreter (e.g. "python3")
//   script       - path to diarize.py
//   pcm16k_mono  - the decoded mono samples at 16 kHz (whisper's input)
//   num_speakers - fixed speaker count, or <= 0 to auto-detect
//   emb_model    - optional --emb-model path ("" = helper default)
bool neural_diarize(const std::string & python,
                    const std::string & script,
                    const std::vector<float> & pcm16k_mono,
                    const std::vector<std::pair<int64_t, int64_t>> & spans_cs,
                    int num_speakers,
                    const std::string & emb_model,
                    std::vector<int> & out_speakers,
                    std::string & error);
