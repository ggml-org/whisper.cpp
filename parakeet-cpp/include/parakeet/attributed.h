#pragma once

// Speaker-attributed transcription: Sortformer segments + ASR text per slice (CTC/TDT/EOU).

#include "export.h"
#include "engine.h"
#include "diarization.h"

#include <string>
#include <vector>

namespace parakeet {

struct AttributedSegment {
    int         speaker_id = 0;
    std::string text;
    double      start_s    = 0.0;
    double      end_s      = 0.0;
};

struct AttributedTranscriptionOptions {
    DiarizationOptions diarization;
    bool   merge_same_speaker  = true;
    int    min_segment_ms      = 200;
    int    pad_segment_ms      = 0;
};

struct AttributedTranscriptionResult {
    std::vector<AttributedSegment> segments;
    DiarizationResult              diarization;
    int                            asr_calls    = 0;
    double                         total_ms     = 0.0;
    int                            audio_samples = 0;
    int                            sample_rate   = 16000;
};

// Throws if engines are not Sortformer + transcription types or sample rates mismatch.
PARAKEET_API AttributedTranscriptionResult transcribe_with_speakers(
    Engine & sortformer_engine,
    Engine & asr_engine,
    const std::string & wav_path,
    const AttributedTranscriptionOptions & opts = {});

PARAKEET_API AttributedTranscriptionResult transcribe_samples_with_speakers(
    Engine & sortformer_engine,
    Engine & asr_engine,
    const float * samples,
    int n_samples,
    int sample_rate,
    const AttributedTranscriptionOptions & opts = {});

}
