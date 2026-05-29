// Test to verify that --no-timestamps flag doesn't affect transcription quality
// The flag should only control output formatting, not the decoding process

#include "whisper.h"
#include "common-whisper.h"
#include <string>
#include <vector>
#include <cstring>
#include <cstdio>

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

// Helper function to extract text from all segments
static std::string extract_text(whisper_context * ctx) {
    std::string result;
    const int n_segments = whisper_full_n_segments(ctx);
    
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);
        if (text) {
            result += text;
        }
    }
    return result;
}

// Helper function to normalize text for comparison (remove extra spaces, lowercase)
static std::string normalize_text(const std::string & text) {
    std::string result;
    bool prev_space = false;
    
    for (char c : text) {
        if (std::isspace(c)) {
            if (!prev_space && !result.empty()) {
                result += ' ';
                prev_space = true;
            }
        } else {
            result += std::tolower(c);
            prev_space = false;
        }
    }
    
    // Remove trailing space
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    
    return result;
}

// Helper to run transcription with given parameters
static std::string transcribe(whisper_context * ctx, const std::vector<float> & pcmf32, bool no_timestamps) {
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = false;
    wparams.print_special    = false;
    wparams.translate        = false;
    wparams.language         = "en";
    wparams.n_threads        = 1;
    wparams.no_timestamps    = no_timestamps;
    
    // Run inference
    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        fprintf(stderr, "error: failed to process audio\n");
        return "";
    }
    
    // Extract text from all segments
    return extract_text(ctx);
}

int main(int argc, char ** argv) {
    std::string model_path = WHISPER_MODEL_PATH;
    std::string sample_path = SAMPLE_PATH;
    
    fprintf(stderr, "Testing no_timestamps behavior\n");
    fprintf(stderr, "Model:  %s\n", model_path.c_str());
    fprintf(stderr, "Sample: %s\n", sample_path.c_str());
    fprintf(stderr, "\n");
    
    // Load model
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;  // Use CPU for consistent results
    
    whisper_context * ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    assert(ctx != nullptr);
    
    // Load audio
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    
    assert(read_audio_data(sample_path.c_str(), pcmf32, pcmf32s, false));
    
    fprintf(stderr, "Loaded audio: %.2f seconds\n", float(pcmf32.size()) / WHISPER_SAMPLE_RATE);
    fprintf(stderr, "\n");
    
    // Test 1: Transcribe with timestamps enabled (default)
    fprintf(stderr, "Test 1: Transcribing with timestamps enabled...\n");
    std::string text_with_ts = transcribe(ctx, pcmf32, false);
    fprintf(stderr, "Result: %s\n", text_with_ts.c_str());
    fprintf(stderr, "\n");
    
    // Test 2: Transcribe with no_timestamps flag
    fprintf(stderr, "Test 2: Transcribing with no_timestamps flag...\n");
    std::string text_no_ts = transcribe(ctx, pcmf32, true);
    fprintf(stderr, "Result: %s\n", text_no_ts.c_str());
    fprintf(stderr, "\n");
    
    // Compare results
    std::string normalized_with_ts = normalize_text(text_with_ts);
    std::string normalized_no_ts = normalize_text(text_no_ts);
    
    fprintf(stderr, "Comparison:\n");
    fprintf(stderr, "  With timestamps:    '%s'\n", normalized_with_ts.c_str());
    fprintf(stderr, "  Without timestamps: '%s'\n", normalized_no_ts.c_str());
    fprintf(stderr, "\n");
    
    // Verify that texts are identical
    bool success = (normalized_with_ts == normalized_no_ts);
    
    if (success) {
        fprintf(stderr, "✓ SUCCESS: Transcriptions are IDENTICAL\n");
        fprintf(stderr, "  The no_timestamps flag only affects output formatting,\n");
        fprintf(stderr, "  not the decoding process. Quality is preserved!\n");
    } else {
        fprintf(stderr, "✗ FAILURE: Transcriptions DIFFER\n");
        fprintf(stderr, "  The no_timestamps flag should not change transcription quality.\n");
        fprintf(stderr, "  This indicates a regression in the fix.\n");
    }
    
    // Cleanup
    whisper_free(ctx);
    
    return success ? 0 : 3;
}

