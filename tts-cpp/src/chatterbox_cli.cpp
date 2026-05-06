// CLI entry point + CLI-specific helpers for the tts-cpp executable.
//
// This file is part of the tts-cpp static library build, but none of its
// symbols are referenced by the Engine API in
// `include/tts-cpp/chatterbox/engine.h`.  Consumers that link libtts-cpp.a
// purely for `tts_cpp::chatterbox::Engine` pay nothing for the CLI code:
// the object file produced from this translation unit is left out of the
// final link by the linker's standard static-archive dead-code rule.
//
// Split out of src/main.cpp so the Engine-only TUs
// (chatterbox_engine.cpp + the T3 helpers still in main.cpp) stay lean
// and don't drag in the CLI's argv parser, signal handlers, live-input
// reader, save-voice dumper, multi-segment crossfade logic, etc.

#include "gpt2_bpe.h"
#include "mtl_tokenizer.h"
#include "tts-cpp/log.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "tts-cpp/tts-cpp.h"
#include "tts-cpp/chatterbox/s3gen_pipeline.h"
#include "tts-cpp/supertonic/engine.h"
#include "chatterbox_t3_internal.h"
#include "t3_mtl.h"
#include "npy.h"
#include "voice_features.h"
#include "voice_encoder.h"
#include "campplus.h"
#include "s3tokenizer.h"
#include "gguf.h"

#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#  include <io.h>
#  include <direct.h>
#  ifndef S_ISREG
#    define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#  endif
#  ifndef S_ISDIR
#    define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#  endif
#  define mkdir(path, mode) _mkdir(path)
#else
#  include <sys/select.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

using namespace tts_cpp::chatterbox::detail;

static bool file_exists(const std::string & path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// Sanity-check a --reference-audio file before we kick off the full voice-
// cloning pipeline.  The Python reference asserts `len(ref) / sr > 5.0` and
// fails hard otherwise; we silently accept any length, but produce undersized
// conditioning tensors (prompt_token=125 instead of 250, etc.) which falls
// back on whatever is in the built-in voice slots.  That's misleading — give
// a clear error instead.  Recommended length is 10–15 seconds.
// Minimal 16-bit PCM WAV writer; matches the one in chatterbox_tts.cpp / mel2wav.cpp.
// Used by the streaming synthesis path to write the final concatenated wav.
static void stream_write_wav(const std::string & path, const std::vector<float> & wav, int sr) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "error: cannot write %s\n", path.c_str()); return; }
    auto w32 = [&](uint32_t v){ f.write((const char*)&v, 4); };
    auto w16 = [&](uint16_t v){ f.write((const char*)&v, 2); };
    uint32_t data_bytes = (uint32_t)(wav.size() * sizeof(int16_t));
    f.write("RIFF", 4); w32(36 + data_bytes); f.write("WAVE", 4);
    f.write("fmt ", 4); w32(16); w16(1); w16(1); w32(sr); w32(sr * 2); w16(2); w16(16);
    f.write("data", 4); w32(data_bytes);
    for (float v : wav) {
        int s = (int)std::lround(v * 32767.0f);
        if (s > 32767) s = 32767; if (s < -32768) s = -32768;
        int16_t s16 = (int16_t)s;
        f.write((const char*)&s16, 2);
    }
}

// Emit a chunk of float samples to stdout as raw 16-bit little-endian PCM
// and flush so downstream players hear it immediately (stdio buffers would
// otherwise hold up to 4-8 KB, stalling real-time playback at chunk
// boundaries).  Used by `--out -` streaming mode; callers pipe into e.g.
// `ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -`.
static void stream_emit_pcm_stdout(const std::vector<float> & wav) {
    for (float v : wav) {
        int s = (int)std::lround(v * 32767.0f);
        if (s > 32767) s = 32767; if (s < -32768) s = -32768;
        int16_t s16 = (int16_t)s;
        std::fwrite(&s16, sizeof(s16), 1, stdout);
    }
    std::fflush(stdout);
}

// Split `text` into TTS-friendly segments of at most `max_chars` characters.
//
// Motivation: Chatterbox Turbo's T3 was trained on utterances of 5–15 s and
// degrades (prosody drift, hallucinated phonemes, timbre wandering) on much
// longer autoregressive outputs.  Reproducible on every backend (ggml / ONNX
// / upstream Python).  The only reliable fix is sentence-level segmentation
// above the model.
//
// The splitter does three passes:
//   1. Break at `. ? !` followed by whitespace / EOF.
//   2. For any sentence longer than `max_chars`, break further at `, : ;`
//      (preferring boundaries past max_chars/2 so we don't fragment into
//      unpronouncable stubs).  Last-resort: hard-break every max_chars.
//   3. Greedily merge consecutive short fragments forward while their
//      combined length stays <= max_chars, so very short sentences ride
//      with their neighbours rather than stand alone.
//
// Abbreviations like "e.g." are not treated specially; in practice the
// greedy merge pass absorbs false splits on them back into the next segment.
static std::vector<std::string> split_text_for_tts(const std::string & text, int max_chars) {
    std::vector<std::string> out;
    if (text.empty() || max_chars <= 0) { out.push_back(text); return out; }

    auto is_ws = [](unsigned char c) { return std::isspace(c) != 0; };

    // Pass 1: sentence split.
    std::vector<std::string> sentences;
    {
        std::string cur;
        size_t i = 0;
        while (i < text.size()) {
            cur += text[i];
            const char c = text[i];
            const bool at_end = (i + 1 == text.size());
            const bool nx_ws  = !at_end && is_ws((unsigned char)text[i + 1]);
            if ((c == '.' || c == '?' || c == '!') && (at_end || nx_ws)) {
                size_t j = i + 1;
                while (j < text.size() && is_ws((unsigned char)text[j])) { cur += text[j]; ++j; }
                sentences.push_back(cur);
                cur.clear();
                i = j;
            } else {
                ++i;
            }
        }
        if (!cur.empty()) sentences.push_back(cur);
    }

    // Pass 2: refine any sentence longer than max_chars.
    std::vector<std::string> refined;
    refined.reserve(sentences.size());
    for (auto & s : sentences) {
        if ((int)s.size() <= max_chars) { refined.push_back(std::move(s)); continue; }
        std::string acc;
        size_t k = 0;
        while (k < s.size()) {
            acc += s[k];
            const char c = s[k];
            const bool nx_ws = (k + 1 < s.size()) && is_ws((unsigned char)s[k + 1]);
            const bool soft_break = (c == ',' || c == ':' || c == ';') && nx_ws &&
                                    (int)acc.size() > max_chars / 2;
            if (soft_break) {
                size_t j = k + 1;
                while (j < s.size() && is_ws((unsigned char)s[j])) { acc += s[j]; ++j; }
                refined.push_back(acc);
                acc.clear();
                k = j;
                continue;
            }
            if ((int)acc.size() >= max_chars) {
                // Last-resort hard break at a space if we can find one in the
                // tail quarter; otherwise just cut.
                size_t back = acc.size();
                while (back > (size_t)(max_chars * 3 / 4) && !is_ws((unsigned char)acc[back - 1])) --back;
                if (back <= (size_t)(max_chars / 2)) back = acc.size();
                refined.push_back(acc.substr(0, back));
                acc.erase(0, back);
            }
            ++k;
        }
        if (!acc.empty()) refined.push_back(acc);
    }

    // Pass 3: greedy forward merge of short fragments.
    for (auto & s : refined) {
        if (!out.empty() && (int)(out.back().size() + s.size()) <= max_chars) {
            out.back() += s;
        } else {
            out.push_back(std::move(s));
        }
    }

    // Strip trailing whitespace per segment.
    for (auto & s : out) {
        while (!s.empty() && is_ws((unsigned char)s.back())) s.pop_back();
    }
    // Drop empty segments (paranoia).
    out.erase(std::remove_if(out.begin(), out.end(),
                             [](const std::string & s) { return s.empty(); }),
              out.end());
    if (out.empty()) out.push_back(text);
    return out;
}

// Append `src` PCM to `dst`, crossfading the last `fade_ms` of `dst` with the
// leading `fade_ms` of `src` via a raised-cosine ramp.  Removes clicks at
// segment seams in auto-split mode.
static void append_pcm_crossfade(std::vector<float> & dst, const std::vector<float> & src,
                                 int sr, int fade_ms) {
    if (src.empty()) return;
    if (dst.empty() || fade_ms <= 0) {
        dst.insert(dst.end(), src.begin(), src.end());
        return;
    }
    int fade_n = sr * fade_ms / 1000;
    fade_n = std::min(fade_n, (int)dst.size());
    fade_n = std::min(fade_n, (int)src.size());
    if (fade_n <= 0) { dst.insert(dst.end(), src.begin(), src.end()); return; }

    const size_t ofs = dst.size() - fade_n;
    for (int i = 0; i < fade_n; ++i) {
        const float t = (float)(i + 1) / (float)(fade_n + 1);
        const float w = 0.5f * (1.0f - std::cos((float)M_PI * t));  // 0 → 1 cosine ramp
        dst[ofs + i] = dst[ofs + i] * (1.0f - w) + src[i] * w;
    }
    dst.insert(dst.end(), src.begin() + fade_n, src.end());
}

// Save the five voice-conditioning tensors to a directory as .npy so later
// runs can reuse them via --ref-dir (no --reference-audio needed), skipping
// VoiceEncoder / CAMPPlus / S3TokenizerV2 / mel-extract entirely.
//
// Any of the five buffers may be empty; missing ones are silently skipped
// (we emit whatever we have and let the reuse path fall back to built-in
// or error cleanly if a required tensor is absent).
static void save_voice_profile(const std::string & dir,
                               const std::vector<float>   & speaker_emb,
                               const std::vector<int32_t> & cond_prompt_speech_tokens,
                               const std::vector<float>   & embedding,
                               const std::vector<int32_t> & prompt_token,
                               const std::vector<float>   & prompt_feat,
                               int prompt_feat_rows /* = pf_data.size() / 80 */)
{
    struct stat st;
    if (::stat(dir.c_str(), &st) != 0) {
        if (::mkdir(dir.c_str(), 0755) != 0) {
            fprintf(stderr, "save_voice_profile: cannot create %s\n", dir.c_str());
            return;
        }
    } else if (!S_ISDIR(st.st_mode)) {
        fprintf(stderr, "save_voice_profile: %s exists but is not a directory\n", dir.c_str());
        return;
    }

    int n_saved = 0;
    if (!speaker_emb.empty()) {
        npy_save_f32(dir + "/speaker_emb.npy",
                     {(int64_t)speaker_emb.size()}, speaker_emb.data());
        ++n_saved;
    }
    if (!cond_prompt_speech_tokens.empty()) {
        npy_save_i32(dir + "/cond_prompt_speech_tokens.npy",
                     {(int64_t)cond_prompt_speech_tokens.size()},
                     cond_prompt_speech_tokens.data());
        ++n_saved;
    }
    if (!embedding.empty()) {
        npy_save_f32(dir + "/embedding.npy",
                     {(int64_t)embedding.size()}, embedding.data());
        ++n_saved;
    }
    if (!prompt_token.empty()) {
        npy_save_i32(dir + "/prompt_token.npy",
                     {(int64_t)prompt_token.size()}, prompt_token.data());
        ++n_saved;
    }
    if (!prompt_feat.empty() && prompt_feat_rows > 0) {
        npy_save_f32(dir + "/prompt_feat.npy",
                     {(int64_t)prompt_feat_rows, 80}, prompt_feat.data());
        ++n_saved;
    }
    fprintf(stderr, "save_voice_profile: wrote %d .npy files into %s\n", n_saved, dir.c_str());
}

// --------------------------------------------------------------------------
// CLI
// --------------------------------------------------------------------------

struct cli_params {
    std::string model;           // T3 GGUF (required unless --tokens-file + --s3gen-gguf)
    std::string tokens_file;     // optional pre-tokenized speech tokens (skips T3)
    std::string text;            // input text for T3
    std::string output;          // legacy: speech-tokens output file (if set, write tokens)
    // S3Gen + HiFT vocoder:
    std::string s3gen_gguf;      // enables full text → wav pipeline
    std::string out_wav;         // wav output path (requires --s3gen-gguf)
    std::string ref_dir;         // override built-in voice with .npy reference dump
    std::string reference_audio; // wav file; computes prompt_feat natively in C++
    std::string save_voice_dir;  // if set, dump the 5 conditioning tensors here for reuse
    bool    debug          = false;  // --debug: load Python-dumped intermediates for validation
    bool    verbose        = false;  // --verbose: per-stage profile timings (human-readable)
    bool    dump_tokens_only = false;
    int32_t seed           = 0;
    bool    seed_set       = false;
    int32_t n_threads      = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_predict      = 1000;   // matches Python's default-ish output budget for paragraph-length text
    int32_t n_ctx          = 0;
    int32_t n_gpu_layers   = 0;
    // Sampling defaults matched to ChatterboxTurboTTS.generate() in tts_turbo.py:
    //   temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2
    // The previous greedy defaults (top_k=1) collapse into silence-token
    // repetition loops on any non-trivial text.
    int32_t top_k          = 1000;
    float   top_p          = 0.95f;
    float   temp           = 0.8f;
    float   repeat_penalty = 1.2f;
    // Experimental: route CFM flash-attn through the F32 Q + F16 K/V path
    // so backends with `flash_attn_f32_f16` (Adreno OpenCL) dispatch the
    // mixed-precision kernel.  Opt-in mobile latency knob.  See PROGRESS.md
    // "OpenCL / Adreno bring-up".
    bool    cfm_f16_kv_attn = false;

    // Multilingual-only knobs. Python ChatterboxMultilingualTTS.generate()
    // defaults: cfg_weight=0.5, temperature=0.8, repetition_penalty=2.0,
    // min_p=0.05, top_p=1.0 (top_k unused).
    float       cfg_weight   = 0.5f;   // classifier-free guidance strength
    float       min_p        = 0.05f;  // minimum-probability warp (0 = off)
    std::string language;              // tier-1 lang code when variant = t3_mtl
    float       exaggeration = 0.5f;   // emotion_adv scalar (0..1)

    // Supertonic-only knobs.  Supertonic stores built-in voices and default
    // steps/speed in its single GGUF, so these are only meaningful when
    // --model points at a GGUF with `supertonic.arch` metadata.
    std::string supertonic_voice;
    int32_t     supertonic_steps = 0;
    float       supertonic_speed = 0.0f;
    std::string supertonic_noise_npy;
    bool        has_supertonic_options = false;

    // Streaming synthesis (PROGRESS.md B1).  When > 0, speech tokens from
    // T3 are fed to S3Gen+HiFT in chunks of this size, with `cache_source`
    // carried across chunks for phase continuity and `trim_fade` only on
    // chunk 0.  Chunks are concatenated in memory and written to --out when
    // the loop finishes, or piped to stdout as soon as each chunk finishes
    // when --out is "-".  No per-chunk files are ever written.
    int32_t stream_chunk_tokens       = 0;
    // Optional: override first-chunk size (typically smaller than
    // stream_chunk_tokens so first-audio-out is fast, then the pipeline
    // switches to larger chunks to amortise the fixed per-chunk overhead).
    // 0 → same as stream_chunk_tokens.
    int32_t stream_first_chunk_tokens = 0;
    // Optional: override CFM Euler step count for streaming chunks.  Defaults
    // to 2 (matches Python's meanflow); setting 1 halves CFM cost at the
    // price of a bit of extra high-frequency noise.
    int32_t stream_cfm_steps          = 0;
    // Override CFM Euler step count for non-streaming synthesis.  Defaults
    // to 0 (= use the GGUF's `n_timesteps`: 10 for Multilingual standard
    // CFM, 2 for Turbo's meanflow).  Lowering N (e.g. 7-8 on Multilingual)
    // reduces S3Gen wall-clock proportionally; the §3.21 sweep documents
    // the audio-cosine knee.  Streaming uses --stream-cfm-steps instead.
    int32_t cfm_steps                 = 0;

    // Auto-split the input text into sentences before running the pipeline.
    // Chatterbox Turbo's T3 degrades badly on autoregressive outputs longer
    // than ~15 s (well outside its training distribution), so anything over
    // a few sentences comes out as garbled prosody, hallucinated phonemes
    // or drifting timbre — regardless of backend (reproduced on Python and
    // ONNX too).  Splitting at sentence boundaries keeps each T3 call
    // in-distribution.  Segments are concatenated with a short raised-cosine
    // crossfade at the seams.
    //
    //   max_sentence_chars      Target length per segment in characters.
    //                           When a sentence exceeds this, we split
    //                           further at `, : ;`.  Set to 0 to disable
    //                           auto-split entirely (single-shot, old
    //                           behaviour; matches --no-auto-split).
    //                           Default 180 ≈ 5–8 s of audio.
    //
    //   crossfade_ms            Raised-cosine crossfade length at segment
    //                           seams, in ms.  Default 30.
    int32_t max_sentence_chars        = 180;
    int32_t crossfade_ms              = 30;

    // Incremental streaming input.  When --input-file PATH is set, the binary
    // opens PATH for reading and follows it with tail -f semantics: as soon as
    // a complete sentence (ending in . ! ? or \n) has been read, it's
    // tokenised, fed to T3, and the resulting speech tokens are streamed
    // through S3Gen + HiFT to stdout.  Intended for pairing with an upstream
    // process (a streaming LLM, a live transcription, a human typing, …) that
    // writes text to the file while we synthesise it.
    //
    // Requires --s3gen-gguf, --stream-chunk-tokens > 0, --out -.
    // Exclusive with --text / --tokens-file.
    std::string input_file;
    std::string input_eof_marker;        // optional; stops reading when seen
    bool        input_by_line    = false; // one request per \n; don't split
                                          // on . ! ? within a line
};

static int32_t sample_next_token(
    const std::vector<float> & logits,
    const std::vector<int32_t> & generated,
    const cli_params & params,
    std::mt19937 & rng) {
    chatterbox_sampling_params sp;
    sp.top_k          = params.top_k;
    sp.top_p          = params.top_p;
    sp.temp           = params.temp;
    sp.repeat_penalty = params.repeat_penalty;
    return sample_next_token_ex(logits, generated, sp, rng);
}

static void print_usage(const char * argv0) {
    fprintf(stderr, "usage: %s --model MODEL.gguf [--text TEXT | --tokens-file tokens.txt] [options]\n", argv0);
    fprintf(stderr, "\noptions:\n");
    fprintf(stderr, "  --model PATH            GGUF model. Chatterbox T3 GGUFs use --s3gen-gguf for\n");
    fprintf(stderr, "                          full text -> wav; Supertonic GGUFs are all-in-one and\n");
    fprintf(stderr, "                          autodetected from supertonic.arch metadata.\n");
    fprintf(stderr, "  --text TEXT             Input text.\n");
    fprintf(stderr, "  --tokens-file PATH      Pre-tokenized text token ids (alternative to --text).\n");
    fprintf(stderr, "                          With --s3gen-gguf this is interpreted as *speech* tokens\n");
    fprintf(stderr, "                          and the T3 step is skipped.\n");
    fprintf(stderr, "  --output PATH           Write generated speech tokens to PATH (text mode).\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  --s3gen-gguf PATH       Enables the full text -> wav pipeline (S3Gen + HiFT).\n");
    fprintf(stderr, "  --out PATH              Output wav file when --s3gen-gguf is set.\n");
    fprintf(stderr, "                          Use `--out -` together with --stream-chunk-tokens to\n");
    fprintf(stderr, "                          pipe raw s16le mono @ 24 kHz to stdout as each chunk\n");
    fprintf(stderr, "                          is ready (for live playback, e.g. `| ffplay -f s16le`).\n");
    fprintf(stderr, "  --ref-dir DIR           Override built-in voice with embedding.npy /\n");
    fprintf(stderr, "                          prompt_token.npy / prompt_feat.npy from DIR, plus\n");
    fprintf(stderr, "                          T3 speaker_emb.npy / cond_prompt_speech_tokens.npy.\n");
    fprintf(stderr, "  --reference-audio PATH  Reference .wav; all five voice-conditioning tensors\n");
    fprintf(stderr, "                          (speaker_emb, cond_prompt_speech_tokens, embedding,\n");
    fprintf(stderr, "                          prompt_token, prompt_feat) are computed in C++.\n");
    fprintf(stderr, "  --save-voice DIR        Dump the 5 computed conditioning tensors as .npy into\n");
    fprintf(stderr, "                          DIR (created if missing).  Use --ref-dir DIR on later\n");
    fprintf(stderr, "                          runs to reuse the voice without --reference-audio —\n");
    fprintf(stderr, "                          skips VoiceEncoder/CAMPPlus/S3TokenizerV2 entirely.\n");
    fprintf(stderr, "  --debug                 Load reference intermediates from --ref-dir for\n");
    fprintf(stderr, "                          bit-exact numerical validation (requires --ref-dir).\n");
    fprintf(stderr, "  --verbose               Print per-stage wall-time breakdown for T3, S3Gen,\n");
    fprintf(stderr, "                          HiFT and (when --reference-audio is used) the voice-\n");
    fprintf(stderr, "                          cloning preprocessing pipeline.\n");
    fprintf(stderr, "  --seed N                RNG seed (default: 0)\n");
    fprintf(stderr, "  --threads N             CPU threads (default: %d)\n", std::min(4, (int32_t) std::thread::hardware_concurrency()));
    fprintf(stderr, "  --n-predict N           Max speech tokens (default: 1000)\n");
    fprintf(stderr, "  --context N             Override KV context length\n");
    fprintf(stderr, "  --n-gpu-layers N        GPU backend when N > 0\n");
    fprintf(stderr, "  --top-k N               (default: 1000, matches Python; use 1 for greedy)\n");
    fprintf(stderr, "  --top-p P               (default: 0.95)\n");
    fprintf(stderr, "  --temp T                (default: 0.8)\n");
    fprintf(stderr, "  --repeat-penalty R      (default: 1.2)\n");
    fprintf(stderr, "  --min-p P               Minimum-probability warp (default: 0.05; t3_mtl only)\n");
    fprintf(stderr, "  --cfg-weight W          Classifier-free guidance strength (default: 0.5;\n");
    fprintf(stderr, "                          t3_mtl only)\n");
    fprintf(stderr, "  --exaggeration X        Emotion-adv scalar in [0,1] (default: 0.5; t3_mtl only)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "multilingual (variant=t3_mtl) options:\n");
    fprintf(stderr, "  --language CODE         Required for t3_mtl GGUFs. Tier-1: en, es, fr, de, it,\n");
    fprintf(stderr, "                          pt, nl, pl, tr, sv, da, fi, no, el, ms, sw, ar, ko.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Supertonic options (when --model has supertonic.arch metadata):\n");
    fprintf(stderr, "  --voice NAME            Built-in Supertonic voice name. Defaults to GGUF metadata.\n");
    fprintf(stderr, "  --steps N               Denoising steps. Defaults to GGUF metadata.\n");
    fprintf(stderr, "  --speed X               Duration speed multiplier. Defaults to GGUF metadata.\n");
    fprintf(stderr, "  --noise-npy PATH        Fixed initial noise tensor for parity/debug runs.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  --stream-chunk-tokens N Synthesize the wav in streaming chunks of N speech\n");
    fprintf(stderr, "                          tokens each (~1 s audio per 25-token chunk).  With\n");
    fprintf(stderr, "                          --out PATH.wav, the concatenated wav is written at the\n");
    fprintf(stderr, "                          end; with --out -, each chunk's PCM is piped to stdout\n");
    fprintf(stderr, "                          as soon as it's produced.  No per-chunk files are\n");
    fprintf(stderr, "                          written.  Requires --s3gen-gguf.  (default: 0 = batch)\n");
    fprintf(stderr, "  --stream-first-chunk-tokens N  Override first-chunk size to minimise first-audio\n");
    fprintf(stderr, "                          latency.  Typical value: 10-15.  (default: 0 = same\n");
    fprintf(stderr, "                          as --stream-chunk-tokens)\n");
    fprintf(stderr, "  --stream-cfm-steps N    CFM Euler step count per chunk.  Python uses 2 for\n");
    fprintf(stderr, "                          meanflow; 1 halves CFM cost.  (default: 0 = 2)\n");
    fprintf(stderr, "  --cfm-steps N           Non-streaming CFM Euler step count.  Multilingual's\n");
    fprintf(stderr, "                          standard CFM ships at 10 steps; lower (e.g. 7-8)\n");
    fprintf(stderr, "                          trades small audio quality for proportional S3Gen\n");
    fprintf(stderr, "                          speedup.  Turbo's meanflow defaults to 2 steps.\n");
    fprintf(stderr, "                          See PROGRESS.md §3.21 for the quality knee sweep.\n");
    fprintf(stderr, "                          (default: 0 = GGUF's n_timesteps)\n");
    fprintf(stderr, "  --cfm-f16-kv-attn       Experimental: CFM flash-attn uses F32 Q + F16 K/V so\n");
    fprintf(stderr, "                          OpenCL/Adreno can dispatch flash_attn_f32_f16.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  --input-file PATH       Stream text from PATH as another process writes to it.\n");
    fprintf(stderr, "                          tail -f semantics: each complete sentence (ending in\n");
    fprintf(stderr, "                          . ! ? or newline) is tokenised and synthesised the\n");
    fprintf(stderr, "                          moment it arrives; audio streams chunk-by-chunk to\n");
    fprintf(stderr, "                          stdout as raw s16le @ 24 kHz mono.  Use '-' to read\n");
    fprintf(stderr, "                          from stdin instead of a file; on a TTY this gives an\n");
    fprintf(stderr, "                          interactive prompt where each Enter-terminated line\n");
    fprintf(stderr, "                          is spoken immediately (Ctrl-D exits).  Runs until\n");
    fprintf(stderr, "                          SIGINT or stdin EOF / --input-eof-marker.  Requires\n");
    fprintf(stderr, "                          --s3gen-gguf, --stream-chunk-tokens > 0, --out -.\n");
    fprintf(stderr, "                          Exclusive with --text / --tokens-file.\n");
    fprintf(stderr, "  --input-eof-marker STR  When this string is seen in the input, flush any\n");
    fprintf(stderr, "                          preceding text, synthesise it, and exit cleanly.\n");
    fprintf(stderr, "                          (default: none = run until SIGINT)\n");
    fprintf(stderr, "  --input-by-line         Treat one newline-terminated line as one request.\n");
    fprintf(stderr, "                          . ! ? inside a line no longer split it into multiple\n");
    fprintf(stderr, "                          synthesis runs (and the 150 ms gap that goes with them);\n");
    fprintf(stderr, "                          the full line is sent to T3 as a single utterance.\n");
    fprintf(stderr, "                          Ideal when each upstream message is one 'request' and\n");
    fprintf(stderr, "                          internal punctuation is meant as prosody, not as a\n");
    fprintf(stderr, "                          hard boundary.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  --max-sentence-chars N  Split --text into segments of at most N chars, running\n");
    fprintf(stderr, "                          T3+S3Gen+HiFT per segment and concatenating the PCM with\n");
    fprintf(stderr, "                          a raised-cosine crossfade.  Works around Chatterbox Turbo's\n");
    fprintf(stderr, "                          degradation on > ~15 s outputs.  (default: 180)\n");
    fprintf(stderr, "  --no-auto-split         Disable the above (single-shot T3 over the full text).\n");
    fprintf(stderr, "  --crossfade-ms N        Crossfade length between segments in ms.  (default: 30)\n");
    fprintf(stderr, "  -h, --help\n");
}

static bool parse_args(int argc, char ** argv, cli_params & params) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s requires an argument\n", flag); return nullptr; }
            return argv[++i];
        };
        // Safe numeric parsers: turn std::stoi / std::stof "no conversion"
        // exceptions into a user-friendly error + clean exit.  Catches the
        // common mistake of a flag being followed by *another flag* because
        // the intended value was forgotten (e.g. `--n-gpu-layers --out ...`).
        auto parse_int = [&](const char * flag, int32_t & out) -> bool {
            auto v = next(flag);
            if (!v) return false;
            try {
                size_t pos = 0;
                long long n = std::stoll(v, &pos);
                if (pos != std::strlen(v)) throw std::invalid_argument("trailing garbage");
                out = (int32_t) n;
                return true;
            } catch (const std::exception & e) {
                fprintf(stderr, "error: %s expects an integer value, got '%s' (%s)\n", flag, v, e.what());
                return false;
            }
        };
        auto parse_float = [&](const char * flag, float & out) -> bool {
            auto v = next(flag);
            if (!v) return false;
            try {
                size_t pos = 0;
                float f = std::stof(v, &pos);
                if (pos != std::strlen(v)) throw std::invalid_argument("trailing garbage");
                out = f;
                return true;
            } catch (const std::exception & e) {
                fprintf(stderr, "error: %s expects a number, got '%s' (%s)\n", flag, v, e.what());
                return false;
            }
        };

        if      (arg == "--model")          { auto v = next("--model");          if (!v) return false; params.model = v; }
        else if (arg == "--text")           { auto v = next("--text");           if (!v) return false; params.text = v; }
        else if (arg == "--tokens-file")    { auto v = next("--tokens-file");    if (!v) return false; params.tokens_file = v; }
        else if (arg == "--output")         { auto v = next("--output");         if (!v) return false; params.output = v; }
        else if (arg == "--s3gen-gguf")     { auto v = next("--s3gen-gguf");     if (!v) return false; params.s3gen_gguf = v; }
        else if (arg == "--out")            { auto v = next("--out");            if (!v) return false; params.out_wav = v; }
        else if (arg == "--ref-dir")        { auto v = next("--ref-dir");        if (!v) return false; params.ref_dir = v; }
        else if (arg == "--reference-audio"){ auto v = next("--reference-audio");if (!v) return false; params.reference_audio = v; }
        else if (arg == "--save-voice")     { auto v = next("--save-voice");     if (!v) return false; params.save_voice_dir = v; }
        else if (arg == "--debug")          { params.debug = true; }
        else if (arg == "--verbose" || arg == "-v") { params.verbose = true; }
        else if (arg == "--seed")           { if (!parse_int  ("--seed",           params.seed))           return false; params.seed_set = true; }
        else if (arg == "--threads")        { if (!parse_int  ("--threads",        params.n_threads))      return false; }
        else if (arg == "--n-predict")      { if (!parse_int  ("--n-predict",      params.n_predict))      return false; }
        else if (arg == "--context")        { if (!parse_int  ("--context",        params.n_ctx))          return false; }
        else if (arg == "--n-gpu-layers")   { if (!parse_int  ("--n-gpu-layers",   params.n_gpu_layers))   return false; }
        else if (arg == "--top-k")          { if (!parse_int  ("--top-k",          params.top_k))          return false; }
        else if (arg == "--top-p")          { if (!parse_float("--top-p",          params.top_p))          return false; }
        else if (arg == "--temp")           { if (!parse_float("--temp",           params.temp))           return false; }
        else if (arg == "--repeat-penalty") { if (!parse_float("--repeat-penalty", params.repeat_penalty)) return false; }
        else if (arg == "--min-p") {
            if (!parse_float("--min-p", params.min_p)) return false;
            if (params.min_p < 0.0f || params.min_p > 1.0f) {
                fprintf(stderr, "error: --min-p must be in [0, 1] (got %g)\n", (double) params.min_p);
                return false;
            }
        }
        else if (arg == "--cfg-weight") {
            if (!parse_float("--cfg-weight", params.cfg_weight)) return false;
            if (params.cfg_weight < 0.0f) {
                fprintf(stderr, "error: --cfg-weight must be >= 0 (got %g)\n", (double) params.cfg_weight);
                return false;
            }
        }
        else if (arg == "--exaggeration") {
            if (!parse_float("--exaggeration", params.exaggeration)) return false;
            if (params.exaggeration < 0.0f || params.exaggeration > 1.0f) {
                fprintf(stderr, "error: --exaggeration must be in [0, 1] (got %g)\n", (double) params.exaggeration);
                return false;
            }
        }
        else if (arg == "--language")       { auto v = next("--language");       if (!v) return false; params.language = v; }
        else if (arg == "--voice")          { auto v = next("--voice");          if (!v) return false; params.supertonic_voice = v; params.has_supertonic_options = true; }
        else if (arg == "--steps")          { if (!parse_int  ("--steps",          params.supertonic_steps)) return false; params.has_supertonic_options = true; }
        else if (arg == "--speed")          { if (!parse_float("--speed",          params.supertonic_speed)) return false; params.has_supertonic_options = true; }
        else if (arg == "--noise-npy")      { auto v = next("--noise-npy");      if (!v) return false; params.supertonic_noise_npy = v; params.has_supertonic_options = true; }
        else if (arg == "--cfm-f16-kv-attn") { params.cfm_f16_kv_attn = true; }
        else if (arg == "--max-sentence-chars") { if (!parse_int("--max-sentence-chars", params.max_sentence_chars)) return false; }
        else if (arg == "--no-auto-split")  { params.max_sentence_chars = 0; }
        else if (arg == "--crossfade-ms")   { if (!parse_int("--crossfade-ms",   params.crossfade_ms))   return false; }
        else if (arg == "--stream-chunk-tokens")       { if (!parse_int("--stream-chunk-tokens",       params.stream_chunk_tokens))       return false; }
        else if (arg == "--stream-first-chunk-tokens") { if (!parse_int("--stream-first-chunk-tokens", params.stream_first_chunk_tokens)) return false; }
        else if (arg == "--stream-cfm-steps")          { if (!parse_int("--stream-cfm-steps",          params.stream_cfm_steps))          return false; }
        else if (arg == "--cfm-steps")                 { if (!parse_int("--cfm-steps",                 params.cfm_steps))                 return false; }
        else if (arg == "--input-file")       { auto v = next("--input-file");       if (!v) return false; params.input_file = v; }
        else if (arg == "--input-eof-marker") { auto v = next("--input-eof-marker"); if (!v) return false; params.input_eof_marker = v; }
        else if (arg == "--input-by-line")    { params.input_by_line = true; }
        else if (arg == "--dump-tokens-only") { params.dump_tokens_only = true; }
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); std::exit(0); }
        else {
            // Surface two common shell typos that would otherwise produce
            // cryptic messages: (a) an argument that's entirely whitespace
            // — symptom of `\<space>` at end of a continuation line, the
            // backslash escapes the space instead of the newline; (b) a
            // leading backslash on the arg itself, symptom of the same
            // thing on the previous line.
            bool all_ws = !arg.empty();
            for (char c : arg) if (!std::isspace((unsigned char)c)) { all_ws = false; break; }
            if (all_ws) {
                fprintf(stderr, "error: empty / whitespace-only argument at position %d. "
                                "This usually means you have a trailing space after '\\' at the "
                                "end of a continuation line — remove it so the shell treats the "
                                "next newline as the line break.\n", i);
            } else if (!arg.empty() && arg[0] == '\\') {
                fprintf(stderr, "error: argument starts with a backslash: %s\n  "
                                "You probably have a trailing space after '\\' on the *previous* "
                                "line, which escaped the space instead of the newline.  Remove "
                                "the trailing space so the next line is treated as a continuation.\n",
                        arg.c_str());
            } else {
                fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            }
            return false;
        }
    }
    if (params.dump_tokens_only) {
        if (params.text.empty()) {
            fprintf(stderr, "error: --dump-tokens-only requires --text\n");
            return false;
        }
        return true;
    }
    // Bake-only mode: just save the 5 voice tensors and exit.
    const bool bake_only = !params.save_voice_dir.empty()
                        && !params.reference_audio.empty()
                        && params.text.empty()
                        && params.tokens_file.empty();
    // If we're only doing the S3Gen+HiFT back half (user already has speech tokens),
    // --model (T3) is optional; otherwise it's required.
    const bool skip_t3 = !params.s3gen_gguf.empty() && !params.tokens_file.empty() && params.text.empty();
    if (!skip_t3 && !bake_only && params.model.empty()) {
        fprintf(stderr, "error: --model is required (pass --s3gen-gguf + --tokens-file to skip T3, "
                        "or --save-voice + --reference-audio to bake only)\n");
        return false;
    }
    if (!bake_only && params.text.empty() && params.tokens_file.empty() && params.input_file.empty()) {
        fprintf(stderr, "error: one of --text / --tokens-file / --input-file is required "
                        "(or --save-voice + --reference-audio to bake a voice profile without synthesising)\n");
        return false;
    }
    if (!params.input_file.empty()) {
        if (params.s3gen_gguf.empty()) {
            fprintf(stderr, "error: --input-file requires --s3gen-gguf\n"); return false;
        }
        if (params.stream_chunk_tokens <= 0) {
            fprintf(stderr, "error: --input-file requires --stream-chunk-tokens > 0\n"); return false;
        }
        if (params.out_wav != "-") {
            fprintf(stderr, "error: --input-file requires --out - (stream raw PCM to stdout)\n"); return false;
        }
        if (!params.text.empty() || !params.tokens_file.empty()) {
            fprintf(stderr, "error: --input-file is mutually exclusive with --text / --tokens-file\n"); return false;
        }
    }
    if (!params.s3gen_gguf.empty() && !bake_only && params.out_wav.empty()) {
        fprintf(stderr, "error: --s3gen-gguf requires --out PATH.wav\n"); return false;
    }
    if (params.debug && params.ref_dir.empty()) {
        fprintf(stderr, "error: --debug requires --ref-dir\n"); return false;
    }
    return true;
}

// --------------------------------------------------------------------------
// I/O helpers
// --------------------------------------------------------------------------

static std::vector<int32_t> read_token_file(const std::string & path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("failed to open token file: " + path);
    std::string raw((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    for (char & ch : raw) if (ch == ',') ch = ' ';
    std::vector<int32_t> tokens;
    std::stringstream ss(raw);
    int32_t tok;
    while (ss >> tok) tokens.push_back(tok);
    return tokens;
}

static void write_token_file(const std::string & path, const std::vector<int32_t> & tokens) {
    std::ofstream fout(path);
    if (!fout) throw std::runtime_error("failed to open output file: " + path);
    for (size_t i = 0; i < tokens.size(); ++i) { if (i) fout << ','; fout << tokens[i]; }
    fout << '\n';
}

// --------------------------------------------------------------------------
// GGUF helpers

enum class cli_model_family {
    unknown,
    chatterbox,
    supertonic,
};

static cli_model_family detect_model_family(const std::string & path) {
    if (path.empty()) return cli_model_family::unknown;
    gguf_init_params gp = { /*.no_alloc=*/ true, /*.ctx=*/ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return cli_model_family::unknown;

    cli_model_family family = cli_model_family::unknown;
    if (gguf_find_key(g, "supertonic.arch") >= 0) {
        family = cli_model_family::supertonic;
    } else if (gguf_find_key(g, KEY_VARIANT) >= 0 ||
               gguf_find_key(g, KEY_TEXT_VOCAB_SIZE) >= 0 ||
               gguf_find_key(g, "tokenizer.ggml.tokens") >= 0) {
        family = cli_model_family::chatterbox;
    }
    gguf_free(g);
    return family;
}

static bool path_looks_supertonic(const std::string & path) {
    std::string lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });
    return lower.find("supertonic") != std::string::npos;
}

static int run_supertonic_cli_path(const cli_params & params) {
    if (!params.tokens_file.empty()) {
        fprintf(stderr, "error: Supertonic does not support --tokens-file; pass --text instead.\n");
        return 1;
    }
    if (!params.output.empty()) {
        fprintf(stderr, "error: Supertonic does not support --output token files; use --out PATH.wav.\n");
        return 1;
    }
    if (!params.s3gen_gguf.empty() || !params.ref_dir.empty() || !params.reference_audio.empty() ||
        !params.save_voice_dir.empty() || params.debug || !params.input_file.empty() ||
        params.stream_chunk_tokens > 0 || params.stream_first_chunk_tokens > 0 ||
        params.stream_cfm_steps > 0 || params.dump_tokens_only) {
        fprintf(stderr,
                "error: Supertonic GGUFs are all-in-one models; --s3gen-gguf, voice-cloning,\n"
                "       token streaming, --input-file, and debug reference-directory modes are\n"
                "       Chatterbox-only in tts-cli.\n");
        return 1;
    }
    if (params.text.empty()) {
        fprintf(stderr, "error: Supertonic requires --text TEXT\n");
        return 1;
    }
    if (params.out_wav.empty()) {
        fprintf(stderr, "error: Supertonic requires --out PATH.wav\n");
        return 1;
    }
    if (params.out_wav == "-") {
        fprintf(stderr, "error: Supertonic does not support --out - streaming yet; pass a wav path.\n");
        return 1;
    }
    if (params.supertonic_steps < 0) {
        fprintf(stderr, "error: --steps must be >= 0 for Supertonic\n");
        return 1;
    }
    if (params.supertonic_speed < 0.0f) {
        fprintf(stderr, "error: --speed must be >= 0 for Supertonic\n");
        return 1;
    }

    tts_cpp::supertonic::EngineOptions opts;
    opts.model_gguf_path = params.model;
    opts.voice = params.supertonic_voice;
    opts.language = params.language.empty() ? "en" : params.language;
    opts.steps = params.supertonic_steps;
    opts.speed = params.supertonic_speed;
    if (params.seed_set) opts.seed = params.seed;
    opts.n_threads = params.n_threads;
    opts.n_gpu_layers = params.n_gpu_layers;
    opts.noise_npy_path = params.supertonic_noise_npy;

    auto result = tts_cpp::supertonic::synthesize(opts, params.text);
    stream_write_wav(params.out_wav, result.pcm, result.sample_rate);
    fprintf(stderr, "wrote %s (%.2fs @ %d Hz, %zu samples)\n",
            params.out_wav.c_str(), result.duration_s, result.sample_rate, result.pcm.size());
    return 0;
}

int tts_cpp_cli_main(int argc, char ** argv) {
    ggml_time_init();
    cli_params params;
    if (!parse_args(argc, argv, params)) {
        // Don't dump the full usage here — parse_args already printed the
        // specific error (missing / malformed value, unknown flag).  Dumping
        // ~90 lines of option descriptions below it just pushes the actual
        // message off-screen.  Point users at --help if they want it.
        fprintf(stderr, "Run `%s --help` for the full list of options.\n", argv[0]);
        return 1;
    }

    // Apply the log filter BEFORE any ggml_backend_*_init() runs, otherwise
    // Metal / Vulkan device-init messages leak out.  The CLI installs
    // chatterbox_log_cb via the public tts_cpp_log_set hook so that
    // (a) the same path a downstream consumer would take is exercised
    // here, and (b) the verbose-or-error gating in chatterbox_log_cb
    // continues to apply without the Engine ctor having to clobber the
    // process-global sink for everyone.
    g_log_verbose = params.verbose ? 1 : 0;
    tts_cpp_log_set(chatterbox_log_cb, nullptr);

    try {
        const cli_model_family family = file_exists(params.model)
            ? detect_model_family(params.model)
            : cli_model_family::unknown;
        if (family == cli_model_family::supertonic ||
            (family == cli_model_family::unknown && path_looks_supertonic(params.model))) {
            return run_supertonic_cli_path(params);
        }
        if (params.has_supertonic_options) {
            fprintf(stderr,
                    "error: --voice / --steps / --speed / --noise-npy are Supertonic-only options,\n"
                    "       but --model was not detected as a Supertonic GGUF.\n");
            return 1;
        }

        // Early preflight: if the user supplied --reference-audio, make sure
        // it's long enough for real voice cloning.  Bail out now with a clear
        // message instead of silently falling back on the built-in voice when
        // the conditioning tensors come out undersized.
        if (!params.reference_audio.empty()) {
            if (!validate_reference_audio(params.reference_audio)) return 1;
        }

        // Bake-only mode: user passed --reference-audio + --save-voice but no
        // text to synthesise.  Compute the five voice tensors, dump them, and
        // exit.  Later runs can reuse with --ref-dir DIR (no preprocessing).
        if (!params.save_voice_dir.empty()
            && !params.reference_audio.empty()
            && params.text.empty()
            && params.tokens_file.empty()) {
            if (params.model.empty() || params.s3gen_gguf.empty()) {
                fprintf(stderr, "error: --save-voice needs both --model and --s3gen-gguf\n");
                return 1;
            }
            // Peek cond_prompt_len out of the T3 GGUF metadata (no weight load).
            int cond_prompt_len = 375;  // Turbo default
            {
                gguf_init_params gp = { /*.no_alloc=*/ true, /*.ctx=*/ nullptr };
                gguf_context * g = gguf_init_from_file(params.model.c_str(), gp);
                if (g) {
                    int64_t id = gguf_find_key(g, KEY_COND_PROMPT_LEN);
                    if (id >= 0) cond_prompt_len = (int)gguf_get_val_u32(g, id);
                    gguf_free(g);
                }
            }

            // Voice-cloning preprocessing shares a backend: on Mac we pick
            // Metal, on Linux + NVIDIA we pick CUDA / Vulkan.  Falls back to
            // the ggml-cpu NEON/AVX kernels when n_gpu_layers == 0.
            ggml_backend_t vc_backend = init_backend(params.n_gpu_layers);

            // (1) speaker_emb via VoiceEncoder (3-layer LSTM + proj + L2-norm
            //     on the chosen backend).
            std::vector<float> se_bake;
            {
                const int64_t _t0 = ggml_time_us();
                voice_encoder_weights vew;
                if (voice_encoder_load(params.model, vew)) {
                    std::vector<float> wav; int sr = 0;
                    if (!wav_load(params.reference_audio, wav, sr))
                        throw std::runtime_error("failed to load --reference-audio");
                    normalise_lufs(wav, sr, -27.0);
                    if (sr != 16000) wav = resample_sinc(wav, sr, 16000);
                    if (!voice_encoder_embed(wav, vew, vc_backend, se_bake))
                        throw std::runtime_error("VoiceEncoder forward failed");
                }
                fprintf(stderr, "BENCH: VC_STAGE_speaker_emb_ms=%lld\n", (long long)((ggml_time_us() - _t0)/1000));
            }

            // (2 + 4) cond_prompt_speech_tokens + prompt_token via S3TokenizerV2.
            std::vector<int32_t> pt_bake, ct_bake;
            {
                const int64_t _t0 = ggml_time_us();
                (void)compute_speech_tokens_native(params.reference_audio, params.s3gen_gguf,
                                                   cond_prompt_len, pt_bake, ct_bake,
                                                   params.n_threads, vc_backend,
                                                   params.verbose);
                fprintf(stderr, "BENCH: VC_STAGE_s3tokenizer_ms=%lld\n", (long long)((ggml_time_us() - _t0)/1000));
            }

            // (3) embedding via CAMPPlus.
            std::vector<float> emb_bake;
            {
                const int64_t _t0 = ggml_time_us();
                (void)compute_embedding_native(params.reference_audio, params.s3gen_gguf,
                                               emb_bake, vc_backend, params.verbose);
                fprintf(stderr, "BENCH: VC_STAGE_campplus_ms=%lld\n", (long long)((ggml_time_us() - _t0)/1000));
            }

            // (5) prompt_feat via mel_extract_24k_80.
            std::vector<float> pf_bake;
            int pf_rows = 0;
            {
                const int64_t _t0 = ggml_time_us();
                (void)compute_prompt_feat_native(params.reference_audio, params.s3gen_gguf,
                                                 pf_bake, pf_rows, params.verbose);
                fprintf(stderr, "BENCH: VC_STAGE_prompt_feat_ms=%lld\n", (long long)((ggml_time_us() - _t0)/1000));
            }

            save_voice_profile(params.save_voice_dir,
                               se_bake, ct_bake, emb_bake, pt_bake, pf_bake, pf_rows);
            fprintf(stderr,
                "done: voice profile written to %s.  Reuse it with "
                "--ref-dir %s (no --reference-audio needed).\n",
                params.save_voice_dir.c_str(), params.save_voice_dir.c_str());
            ggml_backend_free(vc_backend);
            return 0;
        }

        // Short-circuit: user gave us speech tokens directly + --s3gen-gguf. Skip T3 entirely.
        if (params.model.empty() && !params.s3gen_gguf.empty() && !params.tokens_file.empty()) {
            std::vector<int32_t> speech_tokens = read_token_file(params.tokens_file);
            if (speech_tokens.empty()) throw std::runtime_error("empty speech tokens file");
            s3gen_synthesize_opts opts;
            opts.s3gen_gguf_path = params.s3gen_gguf;
            opts.out_wav_path    = params.out_wav;
            opts.ref_dir         = params.ref_dir;
            opts.seed            = params.seed;
            opts.n_threads       = params.n_threads;
            opts.debug           = params.debug;
            opts.verbose         = params.verbose;
            opts.n_gpu_layers    = params.n_gpu_layers;
            opts.cfm_steps       = params.cfm_steps;
            opts.cfm_f16_kv_attn = params.cfm_f16_kv_attn;
            if (!params.reference_audio.empty()) {
                if (!compute_prompt_feat_native(params.reference_audio, params.s3gen_gguf,
                                                opts.prompt_feat_override,
                                                opts.prompt_feat_rows_override,
                                                params.verbose))
                    throw std::runtime_error("failed to compute prompt_feat from --reference-audio");
                // Best-effort: try to compute the S3Gen `embedding` natively too.
                // Falls through to ref_dir/embedding.npy if the s3gen GGUF is pre-A1-2d-a.
                (void)compute_embedding_native(params.reference_audio, params.s3gen_gguf,
                                               opts.embedding_override,
                                               /*backend=*/nullptr, params.verbose);
                // And the S3Gen-side prompt_token via S3TokenizerV2 (Phase 2e).
                // No backend available in this path yet (we haven't loaded T3);
                // fall back to ggml-cpu.  Callers going through the bake path
                // above or the main T3 path below pass the real backend.
                std::vector<int32_t> dummy_cond;
                (void)compute_speech_tokens_native(params.reference_audio, params.s3gen_gguf,
                                                   /*max_cond_tokens=*/-1,
                                                   opts.prompt_token_override, dummy_cond,
                                                   params.n_threads, /*backend=*/nullptr,
                                                   params.verbose);
            }
            return s3gen_synthesize_to_wav(speech_tokens, opts);
        }

        // Load model first so we can use the GGUF-embedded tokenizer (if any).
        chatterbox_model model;
        const int64_t _t3_load_t0 = ggml_time_us();
        if (!load_model_gguf(params.model, model, params.n_ctx, params.n_gpu_layers)) return 1;
        const int64_t _t3_load_ms = (ggml_time_us() - _t3_load_t0) / 1000;
        fprintf(stderr, "BENCH: T3_LOAD_MS=%lld\n", (long long)_t3_load_ms);

        // Warm the S3Gen GGUF cache in the background while T3 inference
        // runs.  This cuts first-audio-out latency by ~700 ms in streaming
        // mode — by the time T3 emits its first chunk of tokens, S3Gen is
        // already in RAM with its tensors allocated on the right backend.
        std::thread s3gen_preload_thread;
        if (!params.s3gen_gguf.empty()) {
            s3gen_preload_thread = std::thread([path = params.s3gen_gguf,
                                                ngpu = params.n_gpu_layers]() {
                s3gen_preload(path, ngpu);
            });
        }

        // Voice-profile override on the T3 side.  We resolve two tensors
        // independently:
        //
        //   speaker_emb           — take from ref_dir/speaker_emb.npy if
        //                           available, otherwise compute in C++ from
        //                           --reference-audio via VoiceEncoder.
        //   cond_prompt_tokens    — only available from ref_dir (until the
        //                           S3TokenizerV2 C++ port in Phase 2e).
        //
        // The S3Gen side is overridden later inside s3gen_synthesize_to_wav.
        bool have_se = false, have_ct = false;
        std::vector<float>   se_data;
        std::vector<int32_t> ct_data;
        if (!params.ref_dir.empty()) {
            const std::string se_path = params.ref_dir + "/speaker_emb.npy";
            const std::string ct_path = params.ref_dir + "/cond_prompt_speech_tokens.npy";
            if (file_exists(se_path)) {
                npy_array se = npy_load(se_path);
                se_data.assign((const float *)se.data.data(),
                               (const float *)se.data.data() + se.n_elements());
                have_se = true;
            }
            if (file_exists(ct_path)) {
                npy_array ct = npy_load(ct_path);
                ct_data.assign((const int32_t *)ct.data.data(),
                               (const int32_t *)ct.data.data() + ct.n_elements());
                have_ct = true;
            }
        }
        if (!have_se && !params.reference_audio.empty()) {
            voice_encoder_weights vew;
            if (voice_encoder_load(params.model, vew)) {
                if (params.verbose) fprintf(stderr, "voice_encoder: computing speaker_emb from %s\n",
                        params.reference_audio.c_str());
                std::vector<float> wav;
                int sr = 0;
                if (!wav_load(params.reference_audio, wav, sr))
                    throw std::runtime_error("failed to load --reference-audio for VoiceEncoder");
                normalise_lufs(wav, sr, -27.0);
                if (sr != 16000) wav = resample_sinc(wav, sr, 16000);
                // Reuse the T3 backend — already loaded & sitting on the GPU
                // at this point in the flow.
                if (!voice_encoder_embed(wav, vew, model.backend, se_data))
                    throw std::runtime_error("VoiceEncoder forward failed");
                have_se = true;
            } else {
                fprintf(stderr,
                    "voice_encoder: T3 GGUF has no VE weights; cannot synthesise speaker_emb natively "
                    "(re-run scripts/convert-t3-turbo-to-gguf.py)\n");
            }
        }

        // Speech-token overrides: compute both cond_prompt_speech_tokens
        // (T3 side) and prompt_token (S3Gen side, stashed for later) via
        // the C++ S3TokenizerV2 port if --reference-audio is given and the
        // s3gen GGUF has the tokenizer weights (Phase 2e).
        std::vector<int32_t> prompt_token_from_ref;
        bool ct_from_cpp = false;
        if (!have_ct && !params.reference_audio.empty() && !params.s3gen_gguf.empty()) {
            std::vector<int32_t> cond_tokens;
            if (compute_speech_tokens_native(params.reference_audio, params.s3gen_gguf,
                                             /*max_cond_tokens=*/model.hparams.cond_prompt_len,
                                             prompt_token_from_ref, cond_tokens,
                                             params.n_threads,
                                             /*backend=*/model.backend,
                                             params.verbose)) {
                ct_data = std::move(cond_tokens);
                have_ct = true;
                ct_from_cpp = true;
            }
        }

        if (have_se) {
            if ((int64_t)se_data.size() != ggml_nelements(model.builtin_speaker_emb)) {
                fprintf(stderr,
                    "error: speaker_emb has %zu elements but builtin_speaker_emb expects %lld\n",
                    se_data.size(), (long long)ggml_nelements(model.builtin_speaker_emb));
                return 1;
            }
            ggml_backend_tensor_set(model.builtin_speaker_emb,
                                    se_data.data(), 0, ggml_nbytes(model.builtin_speaker_emb));
        }

        if (have_ct) {
            if ((int64_t)ct_data.size() == ggml_nelements(model.builtin_cond_prompt_tokens)) {
                ggml_backend_tensor_set(model.builtin_cond_prompt_tokens,
                                        ct_data.data(), 0,
                                        ggml_nbytes(model.builtin_cond_prompt_tokens));
            } else {
                ggml_init_params op = { ggml_tensor_overhead() * 2, nullptr, true };
                model.ctx_override = ggml_init(op);
                if (!model.ctx_override) throw std::runtime_error("ggml_init(ctx_override) failed");
                ggml_tensor * new_ct = ggml_new_tensor_1d(model.ctx_override, GGML_TYPE_I32, (int64_t)ct_data.size());
                ggml_set_name(new_ct, "chatterbox/builtin/cond_prompt_speech_tokens_override");
                model.buffer_override = ggml_backend_alloc_ctx_tensors(model.ctx_override, model.backend);
                if (!model.buffer_override) throw std::runtime_error("alloc override buffer failed");
                ggml_backend_tensor_set(new_ct, ct_data.data(), 0, ct_data.size() * sizeof(int32_t));
                model.builtin_cond_prompt_tokens = new_ct;
                model.hparams.cond_prompt_len = (int32_t)ct_data.size();
            }
        }

        if (have_se || have_ct) {
            if (params.verbose) {
                fprintf(stderr,
                    "%s: T3 voice override — speaker_emb=%s, cond_prompt_tokens=%s\n",
                    __func__,
                    have_se ? (params.reference_audio.empty() ? "ref_dir" : "C++ VoiceEncoder") : "built-in",
                    have_ct ? (ct_from_cpp ? "C++ S3TokenizerV2" : "ref_dir") : "built-in");
            }
        } else if (!params.ref_dir.empty() || !params.reference_audio.empty()) {
            fprintf(stderr,
                "%s: no T3 override; keeping built-in T3 voice\n", __func__);
        }

        // -----------------------------------------------------------------
        // --input-file: incremental / tail -f streaming synthesis
        // -----------------------------------------------------------------
        //
        // When --input-file is set we bypass the static --text pipeline and
        // enter a loop that:
        //
        //   1. fread()s whatever bytes are currently available in the file,
        //   2. scans for complete sentences (ending in . ! ? or newline),
        //   3. tokenises and synthesises each sentence the moment it arrives,
        //      streaming the resulting PCM to stdout chunk-by-chunk, and
        //   4. sleeps briefly and polls again when the file has no new data.
        //
        // The file is expected to be written by another process. Termination:
        //   - SIGINT / SIGTERM (flag is polled between reads and chunks)
        //   - `--input-eof-marker STR` seen in the input (any text before it
        //     is flushed and synthesised, then we exit cleanly).
        //
        // Because the binary doesn't have daemon/server-mode plumbing, every
        // sentence reuses the same in-process T3 + S3Gen+HiFT state: no model
        // reloads, no warm-up between sentences. First-audio latency for a
        // new sentence ≈ T3 prompt eval (~100-300 ms with Metal+q8_0) +
        // first-chunk S3Gen (~250 ms) ≈ 400-550 ms.
        if (!params.input_file.empty()) {
#ifdef _WIN32
            fprintf(stderr, "error: --input-file streaming is not supported on Windows in this build.\n");
            return 1;
#else
            // Wait for the S3Gen background preload up-front so that any
            // early-return below (fopen failure, missing tokenizer, ...)
            // doesn't leave a joinable std::thread that std::terminate()s on
            // destruction.
            if (s3gen_preload_thread.joinable()) s3gen_preload_thread.join();

            // Free all T3-owned GPU resources before an early return.  Without
            // this Metal's static device destructor asserts at process exit
            // ("rsets->data count != 0") because the residency sets attached
            // to model.buffer_w / buffer_kv are still live when the dylib
            // tears the device down.  (S3Gen's cache registers its own
            // atexit hook; T3 has no such hook, main() is its owner.)
            auto free_t3 = [&]() {
                if (model.buffer_stack || model.ctx_stack) {
                    tts_cpp::chatterbox::detail::t3_stack_unregister(
                        model.buffer_stack, model.ctx_stack);
                }
                // Drop the T3 step-graph cache BEFORE freeing the
                // backend.  The cache holds gallocators that carry
                // backend references; freeing them against a dead
                // backend would assert inside the
                // ggml-metal / ggml-vulkan / ggml-cuda dylib finalisers.
                tts_cpp::chatterbox::detail::t3_release_caches();
                ggml_backend_buffer_free(model.buffer_w);
                ggml_backend_buffer_free(model.buffer_kv);
                if (model.buffer_stack)    ggml_backend_buffer_free(model.buffer_stack);
                if (model.buffer_override) ggml_backend_buffer_free(model.buffer_override);
                ggml_backend_free(model.backend);
                ggml_free(model.ctx_w);
                ggml_free(model.ctx_kv);
                if (model.ctx_stack)    ggml_free(model.ctx_stack);
                if (model.ctx_override) ggml_free(model.ctx_override);
                model.buffer_w = nullptr;
                model.buffer_kv = nullptr;
                model.buffer_stack = nullptr;
                model.buffer_override = nullptr;
                model.backend = nullptr;
                model.ctx_w = nullptr;
                model.ctx_kv = nullptr;
                model.ctx_stack = nullptr;
                model.ctx_override = nullptr;
            };

            if (model.tok_tokens.empty()) {
                fprintf(stderr,
                    "error: GGUF has no embedded tokenizer; --input-file requires one\n");
                free_t3();
                return 1;
            }
            gpt2_bpe bpe_live;
            bpe_live.load_from_arrays(model.tok_tokens, model.tok_merges);

            // Use open()/read() on a plain fd rather than fopen()/fread()
            // because macOS/Linux stdio keeps its own readahead buffer; once
            // a FILE* hits EOF the buffer can happily keep returning 0 for
            // many subsequent reads even after the writer appended new
            // bytes and we called clearerr().  read() always asks the
            // kernel for the current state, which is what `tail -f`
            // semantics require.
            //
            // Special case: `--input-file -` means "read from stdin", so the
            // user can run a single process, type (or pipe) text into the
            // terminal, and hear it spoken back.  We intentionally do not
            // open() "/dev/stdin" here because that gets a fresh fd whose
            // initial offset is 0 on some systems, which re-reads prior
            // bytes of the terminal session on TTYs.
            int in_fd;
            const bool input_from_stdin = (params.input_file == "-");
            if (input_from_stdin) {
                in_fd = STDIN_FILENO;
            } else {
                in_fd = open(params.input_file.c_str(), O_RDONLY);
                if (in_fd < 0) {
                    fprintf(stderr, "error: cannot open --input-file '%s': %s\n",
                            params.input_file.c_str(), std::strerror(errno));
                    free_t3();
                    return 1;
                }
            }
            const bool stdin_is_tty = input_from_stdin && isatty(STDIN_FILENO) != 0;

            // S3Gen opts: same setup as the regular streaming branch below.
            s3gen_synthesize_opts opts;
            opts.s3gen_gguf_path = params.s3gen_gguf;
            opts.out_wav_path    = params.out_wav;     // "-" → stdout
            opts.ref_dir         = params.ref_dir;
            opts.seed            = params.seed;
            opts.n_threads       = params.n_threads;
            opts.debug           = params.debug;
            opts.verbose         = params.verbose;
            opts.n_gpu_layers    = params.n_gpu_layers;
            // Live-input streaming: --stream-cfm-steps takes precedence per
            // chunk; --cfm-steps falls in as the per-chunk default below
            // (`stream_cfm_steps > 0 ? stream_cfm_steps : cfm_steps`).
            opts.cfm_steps       = params.cfm_steps;
            opts.cfm_f16_kv_attn = params.cfm_f16_kv_attn;
            if (!params.reference_audio.empty()) {
                if (!compute_prompt_feat_native(params.reference_audio, params.s3gen_gguf,
                                                opts.prompt_feat_override,
                                                opts.prompt_feat_rows_override,
                                                params.verbose))
                    throw std::runtime_error("failed to compute prompt_feat from --reference-audio");
                (void)compute_embedding_native(params.reference_audio, params.s3gen_gguf,
                                               opts.embedding_override,
                                               /*backend=*/model.backend, params.verbose);
                if (!prompt_token_from_ref.empty()) {
                    opts.prompt_token_override = std::move(prompt_token_from_ref);
                }
                if (!params.save_voice_dir.empty()) {
                    save_voice_profile(params.save_voice_dir,
                                       se_data, ct_data,
                                       opts.embedding_override,
                                       opts.prompt_token_override,
                                       opts.prompt_feat_override,
                                       opts.prompt_feat_rows_override);
                }
            }

            // Ctrl-C / SIGTERM: set a flag the read + synth loops poll.
            static std::atomic<bool> live_stop{false};
            {
                struct sigaction sa;
                std::memset(&sa, 0, sizeof(sa));
                sa.sa_handler = [](int){ live_stop.store(true); };
                sigemptyset(&sa.sa_mask);
                sigaction(SIGINT,  &sa, nullptr);
                sigaction(SIGTERM, &sa, nullptr);
            }

            ggml_gallocr_t allocr_live = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(model.backend));
            std::mt19937 rng_live(params.seed);

            constexpr int S3GEN_SIL = tts_cpp::chatterbox::kS3GenSilenceToken;
            const int sr            = opts.sr ? opts.sr : 24000;
            const int chunk_n       = params.stream_chunk_tokens;
            const int first_chunk_n = params.stream_first_chunk_tokens > 0
                                    ? params.stream_first_chunk_tokens
                                    : chunk_n;

            std::string       pending;
            std::vector<char> read_buf(4096);
            bool   saw_eof_marker          = false;
            size_t segments_done           = 0;
            size_t t3_tokens_total_live    = 0;
            int64_t t3_total_ms_live       = 0;
            double  first_audio_t_ms       = -1.0;
            const double live_t0_ms        = 1e-3 * ggml_time_us();

            const char * stop_hint =
                stdin_is_tty
                    ? (params.input_by_line
                          ? "type a request + Enter (Ctrl-D to exit)"
                          : "type a sentence + Enter (Ctrl-D to exit)")
                    : (params.input_eof_marker.empty()
                          ? "Ctrl-C to stop"
                          : "stops at eof-marker or Ctrl-C");
            const char * src_label =
                input_from_stdin ? "<stdin>" : params.input_file.c_str();
            const char * mode_label = params.input_by_line ? ", line-mode" : "";
            fprintf(stderr, "\n=== live input: %s (%s%s) ===\n",
                    src_label, stop_hint, mode_label);
            if (stdin_is_tty) {
                // Prompt lives on stderr so it doesn't collide with the
                // raw-PCM stream on stdout.
                fprintf(stderr, "> ");
                fflush(stderr);
            }

            auto is_ws = [](unsigned char c) { return std::isspace(c) != 0; };

            // Pop the next complete sentence out of `pending`. Returns false
            // if no terminator has been seen yet. When force_final is true we
            // return whatever non-empty tail remains (used for the final
            // flush after eof-marker / SIGINT).
            auto pop_sentence = [&](std::string & out, bool force_final) -> bool {
                if (pending.empty()) return false;
                for (size_t i = 0; i < pending.size(); ++i) {
                    char c = pending[i];
                    // --input-by-line: only a literal newline ends a
                    // request, so internal . ! ? stay inside the same
                    // utterance and T3 gets the whole thing as a single
                    // prompt (no mid-line restart + 150 ms gap).
                    if (params.input_by_line) {
                        if (c != '\n') continue;
                        size_t j = i + 1;
                        while (j < pending.size() && is_ws((unsigned char)pending[j])) ++j;
                        out.assign(pending, 0, j);
                        pending.erase(0, j);
                        return true;
                    }
                    if (c != '.' && c != '!' && c != '?' && c != '\n') continue;
                    const bool at_end = (i + 1 == pending.size());
                    const unsigned char nx =
                        at_end ? 0 : (unsigned char)pending[i + 1];
                    const bool nx_ws     = !at_end && is_ws(nx);
                    // Writers that pack sentences back-to-back without a
                    // space after the terminator (e.g. "Hello.World." from
                    // an LLM that forgot punctuation spacing) would
                    // otherwise bundle everything into one utterance.
                    // Accept "<.!?> + <uppercase letter>" as a sentence
                    // break too.
                    const bool nx_upper  = !at_end && nx >= 'A' && nx <= 'Z';
                    if (c == '\n' || at_end || nx_ws || nx_upper) {
                        size_t j = i + 1;
                        while (j < pending.size() && is_ws((unsigned char)pending[j])) ++j;
                        out.assign(pending, 0, j);
                        pending.erase(0, j);
                        return true;
                    }
                }
                if (force_final) {
                    out = std::move(pending);
                    pending.clear();
                    return !out.empty();
                }
                // Force-flush when the buffer overruns max_sentence_chars * 2
                // without any punctuation, so a writer streaming word soup
                // (no periods) still gets audio out.
                if (params.max_sentence_chars > 0 &&
                    (int)pending.size() > params.max_sentence_chars * 2) {
                    const size_t n = (size_t)params.max_sentence_chars;
                    out.assign(pending, 0, n);
                    pending.erase(0, n);
                    return true;
                }
                return false;
            };

            // Synthesize one sentence end-to-end: T3 -> S3Gen chunked
            // streaming -> stdout PCM. Returns non-zero to abort the loop.
            auto synth_sentence = [&](const std::string & raw_sentence) -> int {
                std::string normalized = gpt2_bpe::punc_norm(raw_sentence);
                if (normalized.empty()) return 0;

                // Reject inputs that are only punctuation / whitespace.
                // Otherwise T3 happily hallucinates ~1-2 s of speaker-biased
                // audio for e.g. the single token "." — which with a cloned
                // voice can come out sounding like a word from the previous
                // utterance (reported live: "i heard 'you?' in seg 3 audio
                // where seg 3 was just '.'").  Nothing sensible comes out of
                // such inputs, so just drop them and keep the prompt.
                bool has_word_char = false;
                for (unsigned char c : normalized) {
                    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c >= 0x80 /* UTF-8 cont */) {
                        has_word_char = true;
                        break;
                    }
                }
                if (!has_word_char) {
                    if (stdin_is_tty) {
                        fprintf(stderr, "[skipped: no word characters]\n");
                    }
                    return 0;
                }

                std::vector<int32_t> text_toks = bpe_live.tokenize(normalized);
                if (text_toks.empty()) return 0;

                {
                    std::string preview = normalized;
                    if (preview.size() > 80) preview = preview.substr(0, 77) + "...";
                    fprintf(stderr, "\n[live seg %zu] %s\n",
                            segments_done + 1, preview.c_str());
                }

                // --- T3: text tokens -> speech tokens ---
                //
                // Re-seed the RNG deterministically per sentence.  Otherwise
                // `rng_live` carries state across sentences: after a couple of
                // long (hundreds-of-tokens) sentences, the RNG is in a very
                // different state than a fresh run, and that state can
                // combine with --repeat-penalty to shift where T3 lands on
                // its stop token.  Deterministic per-segment reseed makes
                // each sentence's sampling independent of the history while
                // keeping everything reproducible (seed + segment index).
                rng_live.seed((uint32_t)params.seed + (uint32_t)segments_done);

                const int64_t t3_t0 = ggml_time_us();

                // Same early-stop / retry loop as batch mode's
                // run_t3_for_segment.  T3 occasionally samples
                // stop_speech_token way too soon when the speaker
                // conditioning is out-of-distribution - most visible
                // with cloned voices - which manifests at the waveform
                // level as the first / last word of a sentence being
                // dropped.  Replay with a different RNG stream up to 3
                // times, keep the longest result.
                const int min_tokens = std::max(8, (int)(text_toks.size() * 5));
                constexpr int MAX_RETRIES = 3;
                auto rng_snapshot = rng_live;

                std::vector<int32_t> generated, best_generated;
                for (int attempt = 0; attempt <= MAX_RETRIES; ++attempt) {
                    rng_live = rng_snapshot;
                    rng_live.discard((size_t)attempt * 1009);

                    std::vector<float> logits;
                    int prompt_len = 0;
                    if (!eval_prompt(model, allocr_live, params.n_threads,
                                     text_toks, logits, prompt_len))
                        throw std::runtime_error("prompt eval failed");

                    int n_past = prompt_len;
                    generated.clear();
                    generated.reserve(params.n_predict + 1);

                    int32_t current = sample_next_token(logits, generated, params, rng_live);
                    generated.push_back(current);

                    bool stopped_by_stop_token = false;
                    for (int i = 0; i < params.n_predict; ++i) {
                        if (current == model.hparams.stop_speech_token) { stopped_by_stop_token = true; break; }
                        if (n_past + 1 > model.hparams.n_ctx) {
                            fprintf(stderr, "KV cache full\n"); break;
                        }
                        if (!eval_step(model, allocr_live, params.n_threads,
                                       n_past, current, logits))
                            throw std::runtime_error("step eval failed");
                        ++n_past;
                        current = sample_next_token(logits, generated, params, rng_live);
                        generated.push_back(current);
                    }

                    if (!generated.empty() &&
                        generated.back() == model.hparams.stop_speech_token)
                        generated.pop_back();

                    if (generated.size() > best_generated.size()) best_generated = generated;

                    const bool plausible = (int)generated.size() >= min_tokens;
                    if (!stopped_by_stop_token || plausible) {
                        if (attempt > 0) {
                            fprintf(stderr, "  [live seg %zu] recovered after %d retries (%zu tokens)\n",
                                    segments_done + 1, attempt, generated.size());
                        }
                        break;
                    }

                    if (attempt < MAX_RETRIES) {
                        fprintf(stderr, "  [live seg %zu] early-stop at %zu tokens "
                                        "(expected >= %d for %zu BPE tokens); retrying %d/%d\n",
                                segments_done + 1, generated.size(), min_tokens,
                                text_toks.size(), attempt + 1, MAX_RETRIES);
                    } else {
                        fprintf(stderr, "  [live seg %zu] all %d retries produced short output; "
                                        "keeping longest (%zu tokens)\n",
                                segments_done + 1, MAX_RETRIES + 1, best_generated.size());
                        generated = best_generated;
                    }
                }

                t3_tokens_total_live += generated.size();
                t3_total_ms_live     += (ggml_time_us() - t3_t0) / 1000;

                // --- S3Gen + HiFT streaming (same boundary + cache logic
                //     as the multi-segment streaming path below) ---
                std::vector<int32_t> seg_toks = std::move(generated);
                for (int i = 0; i < tts_cpp::chatterbox::kS3GenLookaheadTokens; ++i) {
                    seg_toks.push_back(S3GEN_SIL);
                }
                const int total_n   = (int)seg_toks.size();
                const int seg_first = (segments_done == 0) ? first_chunk_n : chunk_n;

                std::vector<int> boundaries = {0};
                int cursor = std::min(seg_first, total_n);
                boundaries.push_back(cursor);
                while (cursor < total_n) {
                    cursor = std::min(cursor + chunk_n, total_n);
                    boundaries.push_back(cursor);
                }
                const int min_tail = std::max(6, chunk_n / 3);
                if (boundaries.size() >= 3) {
                    const int tail_len = boundaries.back()
                                       - boundaries[boundaries.size() - 2];
                    if (tail_len < min_tail) boundaries.erase(boundaries.end() - 2);
                }

                std::vector<float> hift_cache_source;
                int prev_mels_emitted = 0;

                for (int k = 1; k < (int)boundaries.size(); ++k) {
                    if (live_stop.load()) return 0;

                    const int end    = boundaries[k];
                    const bool is_last = (end == total_n);
                    std::vector<int32_t> toks(seg_toks.begin(), seg_toks.begin() + end);

                    s3gen_synthesize_opts copts = opts;
                    std::vector<float> chunk_pcm;
                    copts.out_wav_path             = "";
                    copts.pcm_out                  = &chunk_pcm;
                    copts.append_lookahead_silence = false;
                    copts.finalize                 = is_last;
                    copts.skip_mel_frames          = prev_mels_emitted;
                    copts.apply_trim_fade          = (k == 1);
                    copts.hift_cache_source        = hift_cache_source;
                    std::vector<float> tail_out;
                    copts.hift_source_tail_out     = &tail_out;
                    copts.source_tail_samples      = 480;
                    copts.cfm_steps                = params.stream_cfm_steps > 0 ? params.stream_cfm_steps : params.cfm_steps;
                    copts.cfm_f16_kv_attn          = params.cfm_f16_kv_attn;

                    int rc = s3gen_synthesize_to_wav(toks, copts);
                    if (rc != 0) return rc;

                    if (first_audio_t_ms < 0.0)
                        first_audio_t_ms = 1e-3 * ggml_time_us() - live_t0_ms;

                    stream_emit_pcm_stdout(chunk_pcm);
                    hift_cache_source  = std::move(tail_out);
                    prev_mels_emitted += (int)(chunk_pcm.size() / 480);
                }

                // Short silence between sentences — keeps the downstream
                // player's buffer fed while we wait for the next sentence
                // to arrive on the pipe.  Also reads as a natural pause.
                if (params.crossfade_ms > 0) {
                    const int gap_ms = std::max(150, 2 * params.crossfade_ms);
                    std::vector<float> gap((size_t)(sr * gap_ms / 1000), 0.0f);
                    stream_emit_pcm_stdout(gap);
                }

                ++segments_done;
                return 0;
            };

            // --- Main tail -f loop ---
            //
            // INPUT_POLL_MS is the quiet-period sleep between read() attempts
            // when the input file has no new bytes and between select()
            // wake-ups on a TTY (so SIGINT is noticed without the user also
            // pressing Enter).  25 ms gives ~25 ms first-byte latency for
            // interactive appends, which is well below perception threshold
            // (and the syscall is essentially free at that rate).
            constexpr int INPUT_POLL_MS = 25;
            int  loop_rc     = 0;
            bool stdin_eof   = false;  // only meaningful when reading from stdin
            // Re-prints the interactive prompt once per "ready for input"
            // state (after start-up, and after each synthesised sentence
            // when nothing is queued).
            auto reprompt = [&]() {
                if (stdin_is_tty && !stdin_eof && pending.empty() && !live_stop.load()) {
                    fprintf(stderr, "> ");
                    fflush(stderr);
                }
            };
            while (!live_stop.load()) {
                // For TTY stdin, read() blocks until the user types a line.
                // Use select() with the poll timeout so SIGINT (which sets
                // live_stop) is noticed promptly and the user doesn't need
                // to also press Enter to exit.
                if (stdin_is_tty) {
                    fd_set rf; FD_ZERO(&rf); FD_SET(in_fd, &rf);
                    struct timeval tv;
                    tv.tv_sec  = 0;
                    tv.tv_usec = INPUT_POLL_MS * 1000;
                    int sv = select(in_fd + 1, &rf, nullptr, nullptr, &tv);
                    if (sv <= 0) {
                        if (live_stop.load()) break;
                        continue;
                    }
                }
                ssize_t r;
                do {
                    r = read(in_fd, read_buf.data(), read_buf.size());
                    if (r < 0 && errno == EINTR && live_stop.load()) break;
                } while (r < 0 && errno == EINTR);
                const size_t n = (r > 0) ? (size_t)r : 0;
                if (n > 0) {
                    pending.append(read_buf.data(), n);
                    if (!params.input_eof_marker.empty()) {
                        const auto pos = pending.find(params.input_eof_marker);
                        if (pos != std::string::npos) {
                            pending.resize(pos);
                            saw_eof_marker = true;
                        }
                    }
                } else if (r == 0 && input_from_stdin) {
                    // EOF on stdin: the user pressed Ctrl-D (TTY) or the
                    // upstream pipe was closed.  Drain the buffer one
                    // last time and exit.  For regular files r == 0 just
                    // means "no new bytes appended yet", so we only
                    // treat it as terminal when reading from stdin.
                    stdin_eof = true;
                }

                // Drain every complete sentence currently in the buffer.
                bool synthesised_any = false;
                std::string sentence;
                while (pop_sentence(sentence, /*force_final=*/false)) {
                    loop_rc = synth_sentence(sentence);
                    sentence.clear();
                    synthesised_any = true;
                    if (loop_rc != 0 || live_stop.load()) break;
                }
                if (loop_rc != 0) break;

                if (saw_eof_marker || stdin_eof) {
                    // Final flush: synthesise any non-terminated tail.
                    std::string tail;
                    if (pop_sentence(tail, /*force_final=*/true)) {
                        loop_rc = synth_sentence(tail);
                    }
                    break;
                }

                if (synthesised_any) {
                    reprompt();
                }

                if (n == 0) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(INPUT_POLL_MS));
                }
            }

            // SIGINT with a non-empty buffer: flush what we have.
            if (live_stop.load() && !pending.empty() && loop_rc == 0) {
                std::string tail;
                if (pop_sentence(tail, /*force_final=*/true)) {
                    (void)synth_sentence(tail);
                }
            }
            // On TTY stdin, leave the shell cursor on a fresh line so the
            // post-run summary doesn't overwrite the last "> ".
            if (stdin_is_tty) {
                fprintf(stderr, "\n");
            }

            if (!input_from_stdin) {
                close(in_fd);
            }
            const double total_ms = 1e-3 * ggml_time_us() - live_t0_ms;
            fprintf(stderr,
                    "\n=== live input done: %zu sentences, "
                    "first-audio=%.1f ms, total=%.1f s ===\n",
                    segments_done,
                    first_audio_t_ms >= 0.0 ? first_audio_t_ms : 0.0,
                    total_ms / 1000.0);
            fprintf(stderr, "BENCH: T3_INFER_MS=%lld tokens=%zu\n",
                    (long long)t3_total_ms_live, t3_tokens_total_live);

            ggml_gallocr_free(allocr_live);
            free_t3();
            return loop_rc;
#endif
        }

        // ----------- Segment planning & tokenization ------------------
        //
        // When --text is given and --max-sentence-chars > 0, split the
        // text into TTS-friendly segments and tokenize each separately.
        // Auto-split is suppressed when the tokens-file path is used,
        // when --stream-chunk-tokens requests streaming output, and when
        // --dump-tokens-only prints a single token list.
        gpt2_bpe bpe;
        mtl_tokenizer mtl_tok;
        const bool is_mtl = (model.hparams.variant == CHBX_VARIANT_MTL);
        if (!params.text.empty()) {
            if (is_mtl) {
                if (model.mtl_tokenizer_json.empty()) {
                    fprintf(stderr,
                        "error: this t3_mtl GGUF has no embedded MTL tokenizer. Re-run\n"
                        "       scripts/convert-t3-mtl-to-gguf.py.\n");
                    return 1;
                }
                if (!mtl_tok.load_from_json(model.mtl_tokenizer_json)) {
                    fprintf(stderr, "error: failed to parse embedded MTL tokenizer\n");
                    return 1;
                }
                if (params.language.empty()) {
                    fprintf(stderr,
                        "error: t3_mtl variant requires --language CODE (tier-1: en, es, fr,\n"
                        "       de, it, pt, nl, pl, tr, sv, da, fi, no, el, ms, sw, ar, ko).\n");
                    return 1;
                }
                if (!mtl_tok.is_language_supported(params.language)) {
                    fprintf(stderr,
                        "error: language '%s' is not supported in this build.\n"
                        "       Supported tier-1 codes: ", params.language.c_str());
                    for (const auto & l : mtl_tokenizer::supported_languages()) fprintf(stderr, "%s ", l.c_str());
                    fprintf(stderr, "\n");
                    return 1;
                }
            } else {
                if (model.tok_tokens.empty()) {
                    fprintf(stderr,
                        "error: this GGUF has no embedded tokenizer. Re-run\n"
                        "       scripts/convert-t3-turbo-to-gguf.py to produce a fresh GGUF\n"
                        "       with tokenizer.ggml.* metadata.\n");
                    return 1;
                }
                bpe.load_from_arrays(model.tok_tokens, model.tok_merges);
            }
        }

        const bool auto_split_enabled =
            !params.text.empty() &&
            params.max_sentence_chars > 0 &&
            !params.dump_tokens_only;

        std::vector<std::string> text_segments;
        if (auto_split_enabled) {
            auto segs = split_text_for_tts(params.text, params.max_sentence_chars);
            if (segs.size() > 1) text_segments = std::move(segs);
        }
        if (text_segments.empty() && !params.text.empty()) text_segments.push_back(params.text);

        std::vector<std::vector<int32_t>> seg_text_tokens;
        if (!params.text.empty()) {
            seg_text_tokens.reserve(text_segments.size());
            for (size_t si = 0; si < text_segments.size(); ++si) {
                if (is_mtl) {
                    // MTLTokenizer applies its own normalization (NFKD +
                    // lowercase + language prefix); skip gpt2_bpe::punc_norm.
                    std::vector<int32_t> ids = mtl_tok.encode(text_segments[si], params.language);
                    // Python ChatterboxMultilingualTTS.generate pads with
                    // start_text_token (255) + ids + stop_text_token (0).
                    std::vector<int32_t> padded;
                    padded.reserve(ids.size() + 2);
                    padded.push_back(model.hparams.start_text_token);
                    padded.insert(padded.end(), ids.begin(), ids.end());
                    padded.push_back(model.hparams.stop_text_token);
                    seg_text_tokens.push_back(std::move(padded));
                    if (params.verbose) {
                        fprintf(stderr, "%s: text[%zu] [lang=%s]: \"%s\" (%zu tokens incl. SOT/EOT)\n",
                                __func__, si, params.language.c_str(),
                                text_segments[si].c_str(), seg_text_tokens.back().size());
                    }
                    continue;
                }
                std::string normalized = gpt2_bpe::punc_norm(text_segments[si]);
                seg_text_tokens.push_back(bpe.tokenize(normalized));
                if (params.verbose) {
                    fprintf(stderr, "%s: text[%zu]: \"%s\" (%zu bpe)\n", __func__,
                            si, normalized.c_str(), seg_text_tokens.back().size());
                }
            }
            if (params.dump_tokens_only) {
                // --dump-tokens-only only ever has one segment (auto-split disabled).
                for (size_t i = 0; i < seg_text_tokens[0].size(); ++i) {
                    if (i) printf(",");
                    printf("%d", seg_text_tokens[0][i]);
                }
                printf("\n");
                return 0;
            }
        } else {
            seg_text_tokens.push_back(read_token_file(params.tokens_file));
        }
        for (auto & tt : seg_text_tokens) {
            if (tt.empty()) throw std::runtime_error("empty token input");
        }

        const bool multi_seg = (seg_text_tokens.size() > 1);
        if (multi_seg) {
            fprintf(stderr, "main: auto-split: %zu segments (max-sentence-chars=%d)\n",
                    seg_text_tokens.size(), params.max_sentence_chars);
        }

        // ----------- T3 autoregressive decode (per segment) ------------
        //
        // The KV cache is indexed by position, so calling eval_prompt again
        // simply overwrites positions 0..prompt_len of the cache — no
        // explicit reset needed between segments.
        // T3 autoregressive decode is interleaved *per segment* with S3Gen
        // below — each segment's T3 runs immediately before its own S3Gen
        // call, not all up-front.  This gives low first-audio-out latency
        // (T3(seg0) + first S3Gen chunk ≈ 2–3 s regardless of paragraph
        // length) while avoiding GPU contention from running T3 and S3Gen
        // concurrently on the same device (which doubled per-chunk wall
        // time on Metal when we tried a concurrent background-thread T3).
        //
        // The per-segment T3 loop is wrapped in this closure so both the
        // batch and streaming S3Gen branches can call it uniformly.
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        std::mt19937 rng(params.seed);
        const size_t N_SEG = seg_text_tokens.size();
        std::vector<std::vector<int32_t>> seg_generated(N_SEG);
        size_t t3_tokens_total = 0;
        int64_t t3_total_ms    = 0;

        auto run_t3_for_segment = [&](size_t si) {
            const int64_t _t0 = ggml_time_us();

            chatterbox_sampling_params sp_mtl;
            if (is_mtl) {
                sp_mtl.top_k          = params.top_k;
                sp_mtl.top_p          = params.top_p;
                sp_mtl.temp           = params.temp;
                sp_mtl.repeat_penalty = params.repeat_penalty;
                sp_mtl.min_p          = params.min_p;
                sp_mtl.cfg_weight     = params.cfg_weight;
            }

            // Early-stop heuristic.  T3 occasionally samples `stop_speech_token`
            // way too soon when the speaker conditioning is out-of-distribution
            // (most visible with cloned voices).  Python doesn't hit this
            // because a different RNG stream happens to dodge the bad draw;
            // our std::mt19937 seeded the same way draws differently and
            // sometimes lands on a truncating sequence.
            //
            // Guard: if T3 exits via stop-token and the output length is
            // implausibly short vs. the input text, replay with a
            // different RNG offset.  Floor of 8 tokens covers very short
            // inputs ("Hi."); the 5x multiplier on BPE-token count is a
            // conservative lower bound (Python's reference averages ~5.5x
            // speech tokens per BPE token for English).  If every retry
            // still comes out short, we keep the *longest* attempt rather
            // than whatever the last draw happened to produce.
            const int min_tokens = std::max(8, (int)(seg_text_tokens[si].size() * 5));
            constexpr int MAX_RETRIES = 3;
            auto rng_snapshot = rng;

            std::vector<int32_t> generated, best_generated;
            for (int attempt = 0; attempt <= MAX_RETRIES; ++attempt) {
                rng = rng_snapshot;
                rng.discard((size_t)attempt * 1009);   // move to a different RNG stream each retry

                std::vector<float> logits;
                std::vector<float> logits_c, logits_u;
                int prompt_len = 0;
                if (is_mtl) {
                    if (!eval_prompt_mtl(model, allocr, params.n_threads,
                                         seg_text_tokens[si], params.exaggeration,
                                         logits_c, logits_u, prompt_len))
                        throw std::runtime_error("prompt eval failed");
                } else {
                    if (!eval_prompt(model, allocr, params.n_threads, seg_text_tokens[si], logits, prompt_len))
                        throw std::runtime_error("prompt eval failed");
                }

                int n_past = prompt_len;
                generated.clear();
                generated.reserve(params.n_predict + 1);

                int32_t current = is_mtl
                    ? sample_next_token_mtl(logits_c, logits_u, generated, sp_mtl, rng,
                                            model.hparams.stop_speech_token)
                    : sample_next_token(logits, generated, params, rng);
                generated.push_back(current);

                bool stopped_by_stop_token = false;
                bool stopped_by_repetition  = false;
                for (int i = 0; i < params.n_predict; ++i) {
                    if (current == model.hparams.stop_speech_token) { stopped_by_stop_token = true; break; }
                    if (n_past + 1 > model.hparams.n_ctx) { fprintf(stderr, "KV cache full\n"); break; }
                    bool step_ok;
                    if (is_mtl) {
                        step_ok = eval_step_mtl(model, allocr, params.n_threads, n_past, current,
                                                logits_c, logits_u);
                    } else {
                        step_ok = eval_step(model, allocr, params.n_threads, n_past, current, logits);
                    }
                    if (!step_ok) throw std::runtime_error("step eval failed");
                    ++n_past;
                    current = is_mtl
                        ? sample_next_token_mtl(logits_c, logits_u, generated, sp_mtl, rng,
                                                model.hparams.stop_speech_token)
                        : sample_next_token(logits, generated, params, rng);
                    generated.push_back(current);

                    // Port of the token_repetition check in the Python
                    // AlignmentStreamAnalyzer.  MTL T3 sometimes emits a
                    // plausible end-of-speech silence cadence mid-utterance
                    // and then hallucinates more low-energy content before
                    // eventually stopping.  Three consecutive identical
                    // tokens cleanly signal this cadence without firing on
                    // normal speech.  Gated to MTL because the turbo
                    // codebook has a different cadence signature.
                    if (is_mtl && generated.size() >= 3) {
                        size_t n = generated.size();
                        if (generated[n - 1] == generated[n - 2] &&
                            generated[n - 2] == generated[n - 3]) {
                            stopped_by_repetition = true;
                            break;
                        }
                    }
                }

                if (!generated.empty() && generated.back() == model.hparams.stop_speech_token)
                    generated.pop_back();

                if (stopped_by_repetition && params.verbose) {
                    fprintf(stderr, "  [t3 segment %zu/%zu] stopped on 3x repeated token (%d) "
                                    "at %zu tokens; MTL end-of-speech cadence\n",
                            si + 1, N_SEG, generated.empty() ? -1 : (int)generated.back(),
                            generated.size());
                }

                // Keep the longest attempt as the fallback in case every
                // retry still comes out short.
                if (generated.size() > best_generated.size()) best_generated = generated;

                // The 5x speech-tokens-per-BPE-token floor was calibrated for
                // English Turbo (GPT-2 BPE, ~5 speech tokens per text token).
                // MTL uses the Llama tokenizer with a ~1.7x ratio, so a clean
                // stop-token termination on a short MTL segment looks
                // "implausible" by this heuristic and would trigger up to 3
                // spurious retries (4x T3 wall time).  The 3x-repeated-token
                // early-stop (above) handles MTL's catastrophic case.
                const bool plausible = is_mtl || (int)generated.size() >= min_tokens;
                if (!stopped_by_stop_token || plausible) {
                    if (attempt > 0) {
                        fprintf(stderr, "  [t3 segment %zu/%zu] recovered after %d retries (%zu tokens)\n",
                                si + 1, N_SEG, attempt, generated.size());
                    }
                    break;
                }

                if (attempt < MAX_RETRIES) {
                    fprintf(stderr, "  [t3 segment %zu/%zu] early-stop at %zu tokens "
                                    "(expected >= %d for %zu BPE tokens); retrying %d/%d\n",
                            si + 1, N_SEG, generated.size(), min_tokens,
                            seg_text_tokens[si].size(), attempt + 1, MAX_RETRIES);
                } else {
                    fprintf(stderr, "  [t3 segment %zu/%zu] all %d retries produced short output; "
                                    "keeping longest (%zu tokens)\n",
                            si + 1, N_SEG, MAX_RETRIES + 1, best_generated.size());
                    generated = best_generated;
                }
            }

            if (multi_seg) {
                fprintf(stderr, "  [t3 segment %zu/%zu] %zu speech tokens\n",
                        si + 1, N_SEG, generated.size());
            }
            t3_tokens_total += generated.size();
            t3_total_ms     += (ggml_time_us() - _t0) / 1000;
            seg_generated[si] = std::move(generated);
        };

        // Legacy --output tokens file: run T3 on the first segment early and
        // dump its tokens.  Downstream S3Gen for that same segment will
        // reuse seg_generated[0] without re-running T3.
        if (!params.output.empty()) {
            run_t3_for_segment(0);
            write_token_file(params.output, seg_generated[0]);
        }
        std::vector<bool> seg_t3_done(N_SEG, false);
        if (!params.output.empty()) seg_t3_done[0] = true;

        // Background prefetch slot for the *next* segment's T3, used by the
        // streaming path to hide T3 cost behind the previous segment's
        // streaming audio.  Only one prefetch is in flight at a time because
        // run_t3_for_segment mutates the shared allocr / rng / KV cache.
        std::future<void> next_t3_future;
        size_t next_t3_segment = (size_t)-1;

        auto ensure_t3 = [&](size_t si) {
            if (seg_t3_done[si]) return;
            if (next_t3_segment == si && next_t3_future.valid()) {
                next_t3_future.get();   // propagates exceptions from the worker
                seg_t3_done[si] = true;
                next_t3_segment = (size_t)-1;
                return;
            }
            run_t3_for_segment(si);
            seg_t3_done[si] = true;
        };

        // Kick off T3 for segment `si` on a worker thread so it overlaps
        // with the caller's ongoing S3Gen streaming.  Caller must ensure
        // no previous prefetch is still in flight.
        auto prefetch_t3 = [&](size_t si) {
            if (si >= N_SEG || seg_t3_done[si] || next_t3_segment == si) return;
            next_t3_segment = si;
            next_t3_future = std::async(std::launch::async, [&, si]() {
                run_t3_for_segment(si);
            });
        };

        // If --s3gen-gguf is set, chain into the S3Gen + HiFT vocoder to write a wav.
        if (!params.s3gen_gguf.empty()) {
            // Make sure the background preload finished before we call into
            // s3gen_synthesize_to_wav — otherwise we race with its own lazy
            // load and pay the ~700 ms cost anyway.
            if (s3gen_preload_thread.joinable()) s3gen_preload_thread.join();
            s3gen_synthesize_opts opts;
            opts.s3gen_gguf_path = params.s3gen_gguf;
            opts.out_wav_path    = params.out_wav;
            opts.ref_dir         = params.ref_dir;
            opts.seed            = params.seed;
            opts.n_threads       = params.n_threads;
            opts.debug           = params.debug;
            opts.verbose         = params.verbose;
            opts.n_gpu_layers    = params.n_gpu_layers;
            // Non-streaming CFM Euler step count (0 = GGUF default).
            // Streaming chunks honour --stream-cfm-steps with --cfm-steps as
            // fallback when copts is set up further below.
            opts.cfm_steps       = params.cfm_steps;
            opts.cfm_f16_kv_attn = params.cfm_f16_kv_attn;
            if (!params.reference_audio.empty()) {
                if (!compute_prompt_feat_native(params.reference_audio, params.s3gen_gguf,
                                                opts.prompt_feat_override,
                                                opts.prompt_feat_rows_override,
                                                params.verbose))
                    throw std::runtime_error("failed to compute prompt_feat from --reference-audio");
                (void)compute_embedding_native(params.reference_audio, params.s3gen_gguf,
                                               opts.embedding_override,
                                               /*backend=*/model.backend, params.verbose);
                if (!prompt_token_from_ref.empty()) {
                    opts.prompt_token_override = std::move(prompt_token_from_ref);
                }

                // Optionally persist the five voice tensors to disk so later
                // runs can reuse them via --ref-dir DIR (no --reference-audio).
                if (!params.save_voice_dir.empty()) {
                    save_voice_profile(params.save_voice_dir,
                                       se_data, ct_data,
                                       opts.embedding_override,
                                       opts.prompt_token_override,
                                       opts.prompt_feat_override,
                                       opts.prompt_feat_rows_override);
                }
            }
            if (params.stream_chunk_tokens <= 0 && !multi_seg) {
                // Single-shot.  Two output modes:
                //   --out PATH  → s3gen_synthesize_to_wav writes the wav and
                //                 prints its own "Wrote ..." line.
                //   --out -     → capture PCM in-memory and emit raw s16le to
                //                 stdout (so it can be piped into `play` /
                //                 `ffplay` without streaming mode).
                ensure_t3(0);
                if (params.out_wav == "-") {
                    std::vector<float> pcm;
                    s3gen_synthesize_opts so = opts;
                    so.out_wav_path = "";
                    so.pcm_out      = &pcm;
                    int rc = s3gen_synthesize_to_wav(seg_generated[0], so);
                    if (rc != 0) return rc;
                    stream_emit_pcm_stdout(pcm);
                    fprintf(stderr, "Streamed to stdout (raw s16le @ 24 kHz mono)\n");
                } else {
                    int rc = s3gen_synthesize_to_wav(seg_generated[0], opts);
                    if (rc != 0) return rc;
                }
            } else if (params.stream_chunk_tokens <= 0) {
                // Auto-split multi-segment batch mode: render each segment
                // into in-memory PCM, concatenate with a raised-cosine
                // crossfade at the seams, write a single wav at the end.
                const int sr = opts.sr ? opts.sr : 24000;
                std::vector<float> full_pcm;
                const int64_t _seg_t0 = ggml_time_us();
                for (size_t si = 0; si < N_SEG; ++si) {
                    ensure_t3(si);
                    s3gen_synthesize_opts copts = opts;
                    std::vector<float> seg_pcm;
                    copts.out_wav_path = "";
                    copts.pcm_out      = &seg_pcm;
                    fprintf(stderr, "\n--- segment %zu/%zu: %zu speech tokens ---\n",
                            si + 1, N_SEG, seg_generated[si].size());
                    int rc = s3gen_synthesize_to_wav(seg_generated[si], copts);
                    if (rc != 0) return rc;
                    append_pcm_crossfade(full_pcm, seg_pcm, sr, params.crossfade_ms);
                }
                const double seg_total_ms = 1e-3 * (ggml_time_us() - _seg_t0);
                const double audio_ms = 1000.0 * (double)full_pcm.size() / (double)sr;
                fprintf(stderr,
                        "\n=== auto-split: %zu segments, %.0f ms for %.0f ms audio (RTF=%.2f) ===\n",
                        N_SEG, seg_total_ms, audio_ms,
                        audio_ms > 0.0 ? seg_total_ms / audio_ms : 0.0);
                if (params.out_wav == "-") {
                    stream_emit_pcm_stdout(full_pcm);
                    fprintf(stderr, "Streamed to stdout (raw s16le @ 24 kHz mono)\n");
                } else {
                    stream_write_wav(params.out_wav, full_pcm, sr);
                    fprintf(stderr, "Wrote %s\n", params.out_wav.c_str());
                }
            } else {
                // Streaming synthesis.  Runs the chunked S3Gen+HiFT loop on
                // each T3 segment.  Within a segment, `hift_cache_source`
                // keeps SineGen phase continuous across chunks and
                // `skip_mel_frames` trims the already-emitted mel tail.  At
                // segment boundaries both reset: a new utterance starts, the
                // crossfade (in file mode) or the natural sentence-break
                // pause (in stdout mode) masks any seam.
                //
                // Also: resetting per segment is precisely what keeps per-
                // chunk cost bounded — encoder and CFM only ever process the
                // current segment's cumulative mel, which stays small
                // (~200-300 tokens).  Single-segment streaming on a long
                // paragraph otherwise scales linearly with total length.
                //
                // Mirrors scripts/dump-streaming-reference.py.  Appends 3
                // S3GEN_SIL tokens at each segment's tail so every chunk's
                // `append_lookahead_silence=false` is safe.
                constexpr int S3GEN_SIL = tts_cpp::chatterbox::kS3GenSilenceToken;

                const int chunk_n       = params.stream_chunk_tokens;
                const int first_chunk_n = params.stream_first_chunk_tokens > 0
                                        ? params.stream_first_chunk_tokens
                                        : chunk_n;
                const bool to_stdout    = (params.out_wav == "-");
                const int  sr           = opts.sr ? opts.sr : 24000;

                if (multi_seg) {
                    fprintf(stderr, "\n=== streaming synthesis: %zu segments, %d-token chunks%s ===\n",
                            N_SEG, chunk_n,
                            to_stdout ? " → stdout (raw s16le @ 24 kHz mono)" : "");
                } else {
                    fprintf(stderr, "\n=== streaming synthesis: %d-token chunks%s ===\n",
                            chunk_n,
                            to_stdout ? " → stdout (raw s16le @ 24 kHz mono)" : "");
                }

                std::vector<float> full_streamed_wav;   // only used in file mode
                const double stream_t0_ms = 1e-3 * ggml_time_us();
                double first_chunk_t_ms = -1.0;
                int global_chunk_idx = 0;

                for (size_t si = 0; si < N_SEG; ++si) {
                    ensure_t3(si);
                    std::vector<int32_t> seg_toks = seg_generated[si];
                    for (int i = 0; i < tts_cpp::chatterbox::kS3GenLookaheadTokens; ++i) {
                        seg_toks.push_back(S3GEN_SIL);
                    }
                    const int total_n = (int)seg_toks.size();

                    // Chunk boundaries within this segment.  The small
                    // `first_chunk_n` override only applies to the very first
                    // segment — there it buys low first-audio-out latency.
                    // For later segments the player already has audio queued,
                    // so we use the bigger `chunk_n` for all chunks to keep
                    // per-chunk RTF < 1.0 and avoid a per-segment stutter.
                    const int seg_first = (si == 0) ? first_chunk_n : chunk_n;
                    std::vector<int> boundaries = {0};
                    int cursor = std::min(seg_first, total_n);
                    boundaries.push_back(cursor);
                    while (cursor < total_n) {
                        cursor = std::min(cursor + chunk_n, total_n);
                        boundaries.push_back(cursor);
                    }

                    // Absorb a short trailing chunk into the previous one.
                    // When total_n isn't a clean multiple of the chunk
                    // stride, the final chunk can emit only a handful of
                    // new tokens — but S3Gen still pays the full encoder+
                    // CFM cost on the cumulative mel, so it shows up as a
                    // cosmetically-bad RTF (e.g. "520 ms compute for 160 ms
                    // audio, RTF=3.25") and burns one wasted encoder+CFM
                    // dispatch.  Merging saves ~500 ms of compute per
                    // segment with no audible impact because the same
                    // audio is still emitted, just in one dispatch instead
                    // of two.
                    //
                    // Threshold: compute ≈ fixed (~500 ms) per chunk; audio
                    // emitted ≈ tail_len × 40 ms.  Break-even (RTF ≈ 1) is
                    // at ~12 tokens.  chunk_n / 3 catches anything
                    // noticeably worse than that for the usual 25-token
                    // stride, and scales for larger strides too.
                    const int min_tail = std::max(6, chunk_n / 3);
                    if (boundaries.size() >= 3) {
                        const int tail_len = boundaries.back() - boundaries[boundaries.size() - 2];
                        if (tail_len < min_tail) {
                            boundaries.erase(boundaries.end() - 2);
                        }
                    }

                    std::vector<float> hift_cache_source;       // reset per segment
                    std::vector<float> seg_streamed_wav;
                    int prev_mels_emitted = 0;
                    size_t last_streamed = 0;

                    if (multi_seg) {
                        fprintf(stderr, "\n[segment %zu/%zu: %d tokens → %d chunks]\n",
                                si + 1, N_SEG, total_n,
                                (int)boundaries.size() - 1);
                    }

                    for (int k = 1; k < (int)boundaries.size(); ++k) {
                        const int end    = boundaries[k];
                        const bool is_last_in_seg = (end == total_n);
                        std::vector<int32_t> toks(seg_toks.begin(), seg_toks.begin() + end);

                        s3gen_synthesize_opts copts = opts;
                        std::vector<float> chunk_pcm;
                        copts.out_wav_path              = "";
                        copts.pcm_out                   = &chunk_pcm;
                        copts.append_lookahead_silence  = false;
                        copts.finalize                  = is_last_in_seg;
                        copts.skip_mel_frames           = prev_mels_emitted;
                        // First chunk of EACH segment gets trim_fade, masking
                        // HiFT's per-utterance resnet cold start.
                        copts.apply_trim_fade           = (k == 1);
                        copts.hift_cache_source         = hift_cache_source;
                        std::vector<float> tail_out;
                        copts.hift_source_tail_out      = &tail_out;
                        copts.source_tail_samples       = 480;
                        copts.cfm_steps                 = params.stream_cfm_steps > 0 ? params.stream_cfm_steps : params.cfm_steps;
                        copts.cfm_f16_kv_attn           = params.cfm_f16_kv_attn;

                        ++global_chunk_idx;
                        if (params.verbose) {
                            fprintf(stderr, "\n--- %schunk %d/%d: tokens_total=%d finalize=%s ---\n",
                                    multi_seg ? "seg " : "",
                                    k, (int)boundaries.size() - 1, end,
                                    is_last_in_seg ? "true" : "false");
                        }
                        int rc = s3gen_synthesize_to_wav(toks, copts);
                        if (rc != 0) return rc;

                        if (first_chunk_t_ms < 0.0)
                            first_chunk_t_ms = 1e-3 * ggml_time_us() - stream_t0_ms;

                        if (to_stdout) stream_emit_pcm_stdout(chunk_pcm);
                        seg_streamed_wav.insert(seg_streamed_wav.end(),
                                                chunk_pcm.begin(), chunk_pcm.end());
                        hift_cache_source = std::move(tail_out);

                        size_t chunk_samples = seg_streamed_wav.size() - last_streamed;
                        prev_mels_emitted  += (int)(chunk_samples / 480);
                        last_streamed = seg_streamed_wav.size();

                        // Kick off T3 for the NEXT segment as soon as the
                        // first chunk of the current segment has been emitted.
                        // This hides the ~1–2 s T3 cost behind the remaining
                        // S3Gen chunks so the player doesn't stall waiting
                        // for the next segment's tokens at the boundary.
                        // First chunk runs on a "clean" GPU so its RTF stays
                        // low; subsequent chunks share the GPU with T3 but
                        // the queue grown by chunk 1 absorbs the slowdown.
                        if (k == 1 && si + 1 < N_SEG) prefetch_t3(si + 1);
                    }

                    if (to_stdout) {
                        // In stdout mode, emit a short inter-segment silence
                        // (unless this is the final segment) so the player's
                        // buffer stays non-empty while T3 decodes the next
                        // segment (~1.5 s with no PCM otherwise).  The
                        // silence reads as a natural sentence-break pause.
                        const bool more = (si + 1 < N_SEG);
                        if (more && params.crossfade_ms > 0) {
                            const int gap_ms = std::max(150, 2 * params.crossfade_ms);
                            const int gap_samples = sr * gap_ms / 1000;
                            std::vector<float> gap(gap_samples, 0.0f);
                            stream_emit_pcm_stdout(gap);
                        }
                    } else {
                        // File mode: concatenate segments with a raised-cosine
                        // crossfade at each boundary, identical to the
                        // non-streaming auto-split path.
                        append_pcm_crossfade(full_streamed_wav, seg_streamed_wav,
                                             sr, multi_seg ? params.crossfade_ms : 0);
                    }
                }

                if (!to_stdout) stream_write_wav(params.out_wav, full_streamed_wav, sr);
                fprintf(stderr, "\n=== streaming done: %d chunks, first-chunk latency=%.1f ms, total=%.1f ms ===\n",
                        global_chunk_idx, first_chunk_t_ms,
                        1e-3 * ggml_time_us() - stream_t0_ms);
                if (to_stdout)
                    fprintf(stderr, "Streamed to stdout (raw s16le @ 24 kHz mono)\n");
                else
                    fprintf(stderr, "Wrote %s\n", params.out_wav.c_str());
            }
        } else {
            // Legacy: print the tokens to stdout too (handy for piping).
            // In auto-split mode this prints the first segment's tokens only;
            // real callers who want all segments pass --s3gen-gguf and --out.
            ensure_t3(0);
            const auto & toks = seg_generated[0];
            for (size_t i = 0; i < toks.size(); ++i) { if (i) printf(","); printf("%d", toks[i]); }
            printf("\n");
        }

        fprintf(stderr, "BENCH: T3_INFER_MS=%lld tokens=%zu\n",
                (long long)t3_total_ms, t3_tokens_total);

        ggml_gallocr_free(allocr);
        // Drop T3 step-graph cache BEFORE freeing the backend
        // (gallocators in cached entries reference it).
        tts_cpp::chatterbox::detail::t3_release_caches();
        ggml_backend_buffer_free(model.buffer_w);
        ggml_backend_buffer_free(model.buffer_kv);
        if (model.buffer_override) ggml_backend_buffer_free(model.buffer_override);
        ggml_backend_free(model.backend);
        ggml_free(model.ctx_w);
        ggml_free(model.ctx_kv);
        if (model.ctx_override) ggml_free(model.ctx_override);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
