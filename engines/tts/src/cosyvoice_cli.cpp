// cosyvoice-cli: end-to-end CosyVoice3 text -> 24 kHz WAV via the public
// tts_cpp::cosyvoice::Engine (the same API the JS addon / @qvac/sdk drive).
//
// Point --model-dir at a folder assembled by scripts/assemble-cosyvoice3-model.py
// (cosyvoice3-{llm,flow,hift}*.gguf + voice.gguf + vocab.json + merges.txt):
//
//   cosyvoice-cli --model-dir models/cosyvoice3-0.5b \
//       --text "Hello from a fully on-device C++ pipeline." --out out.wav
//
// Pass --n-gpu-layers > 0 to select the GPU path (Metal on Apple, Vulkan on
// desktop Linux / Windows, OpenCL on Adreno; every other GPU is declined and
// falls back to CPU), or --backend to name the device (hexagon = the
// Snapdragon HTP0 NPU; no fallback).
//
// Voice cloning (needs cosyvoice3-s3tok*.gguf + cosyvoice3-campplus*.gguf in
// the model dir, or --s3tok-gguf / --campplus-gguf):
//
//   zero-shot (same language as the reference; best fidelity):
//     cosyvoice-cli --model-dir ... --reference-audio me.wav \
//         --prompt-text "verbatim transcript of me.wav" --text "..."
//   cross-lingual (no transcript; timbre only, target language differs):
//     cosyvoice-cli --model-dir ... --reference-audio me.wav --text "..."
//
// Without --reference-audio the baked default voice (voice.gguf) is used.

#include "tts-cpp/cosyvoice/engine.h"
#include "tts-cpp/voice_controls.h"
#include "voice_controls_cli.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

static void write_wav(const std::string & path, const std::vector<float> & wav, int sr) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    uint32_t num_samples = (uint32_t)wav.size();
    uint32_t byte_rate = sr * 2, data_size = num_samples * 2, chunk_size = 36 + data_size;
    uint32_t fcs = 16, sr32 = (uint32_t)sr; uint16_t af = 1, nc = 1, ba = 2, bps = 16;
    std::fwrite("RIFF", 1, 4, f); std::fwrite(&chunk_size, 4, 1, f); std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f); std::fwrite(&fcs, 4, 1, f); std::fwrite(&af, 2, 1, f); std::fwrite(&nc, 2, 1, f);
    std::fwrite(&sr32, 4, 1, f); std::fwrite(&byte_rate, 4, 1, f); std::fwrite(&ba, 2, 1, f); std::fwrite(&bps, 2, 1, f);
    std::fwrite("data", 1, 4, f); std::fwrite(&data_size, 4, 1, f);
    for (float x : wav) { float c = std::max(-1.0f, std::min(1.0f, x)); int16_t v = (int16_t)std::lrintf(c * 32767.0f); std::fwrite(&v, 2, 1, f); }
    std::fclose(f);
}

// Whole-string parse for --vulkan-device: atoi would turn "abc" into
// adapter 0 and "1abc" into adapter 1, defeating the option's
// fail-loud contract. Accepts -1 (auto-pick) or a nonnegative index.
static bool parse_vulkan_device(const char * s, int & out) {
    int v = 0;
    const auto r = std::from_chars(s, s + std::strlen(s), v);
    if (r.ec != std::errc() || r.ptr != s + std::strlen(s) || v < -1) return false;
    out = v;
    return true;
}

int main(int argc, char ** argv) {
    using namespace tts_cpp::cosyvoice;
    std::string model_dir, text = "Hello from a fully on-device C plus plus pipeline.";
    namespace ctl = tts_cpp::controls;
    constexpr ctl::EngineId kEngine = ctl::EngineId::CosyVoice;

    std::string out = "cosyvoice_out.wav", prompt_text, voice_gguf;
    std::string reference_audio, s3tok_gguf, campplus_gguf;
    std::string backends_dir, opencl_cache_dir, backend;
    VoiceControls controls;
    int seed = 42, n_gpu_layers = 0, n_threads = 0, vulkan_device = 0;
    bool greedy = false;
    bool flow_cut_prompt = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--text" && i + 1 < argc) text = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--prompt-text" && i + 1 < argc) prompt_text = argv[++i];
        else if (a == "--reference-audio" && i + 1 < argc) reference_audio = argv[++i];
        else if (a == "--s3tok-gguf" && i + 1 < argc) s3tok_gguf = argv[++i];
        else if (a == "--campplus-gguf" && i + 1 < argc) campplus_gguf = argv[++i];
        else if (a == "--instruct" && i + 1 < argc) controls.instruct_text = argv[++i];
        else if (a == "--emotion" && i + 1 < argc) controls.emotion = argv[++i];
        else if (a == "--pace" && i + 1 < argc) controls.pace = argv[++i];
        else if (a == "--list-emotions") { printf("%s\n", ctl::cli::describe_emotions(kEngine).c_str()); return 0; }
        else if (a == "--list-paces") { printf("%s\n", ctl::cli::describe_paces(kEngine).c_str()); return 0; }
        else if (a == "--voice-gguf" && i + 1 < argc) voice_gguf = argv[++i];
        else if (a == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
        else if ((a == "--n-gpu-layers" || a == "-ngl") && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else if (a == "--vulkan-device" && i + 1 < argc) {
            if (!parse_vulkan_device(argv[++i], vulkan_device)) {
                fprintf(stderr, "cosyvoice-cli: --vulkan-device expects -1 or a nonnegative "
                                "adapter index, got \"%s\"\n", argv[i]);
                return 1;
            }
        }
        else if ((a == "--threads" || a == "-t") && i + 1 < argc) n_threads = std::atoi(argv[++i]);
        else if (a == "--backend" && i + 1 < argc) backend = argv[++i];
        else if (a == "--backends-dir" && i + 1 < argc) backends_dir = argv[++i];
        else if (a == "--opencl-cache-dir" && i + 1 < argc) opencl_cache_dir = argv[++i];
        else if (a == "--greedy") greedy = true;
        else if (a == "--flow-cut-prompt") flow_cut_prompt = true;
        else {
            fprintf(stderr,
                "usage: %s --model-dir DIR [--text ...] [--voice-gguf voice.gguf]\n"
                "          [--reference-audio REF.wav [--prompt-text \"its transcript\"]]\n"
                "          [--s3tok-gguf S3TOK.gguf] [--campplus-gguf CAMPPLUS.gguf]\n"
                "          [--emotion NAME] [--pace slow|moderate|fast] [--instruct \"...\"]\n"
                "          [--list-emotions] [--list-paces]\n"
                "          [--out out.wav] [--seed N] [--greedy] [--flow-cut-prompt]\n"
                "          [--n-gpu-layers N] [--threads N]\n"
                "          [--backend auto|cpu|opencl|webgpu|hexagon|DEVICE] (explicit device;\n"
                "                      overrides --n-gpu-layers, no fallback)\n"
                "          [--vulkan-device N] [--backends-dir DIR] [--opencl-cache-dir DIR]\n"
                "\n"
                "Voice cloning: --reference-audio with --prompt-text (the reference's verbatim\n"
                "transcript) = zero-shot; without --prompt-text = cross-lingual (timbre only).\n"
                "Requires the s3tok + campplus GGUFs (auto-resolved from --model-dir).\n"
                "\n"
                "CosyVoice3 is trained on one instruction per synthesis: set at most one of\n"
                "--emotion / --pace / --instruct (pace=moderate counts as unset).\n"
                "--list-emotions prints the values this engine supports.\n", argv[0]);
            return 1;
        }
    }
    if (model_dir.empty()) { fprintf(stderr, "need --model-dir\n"); return 1; }

    EngineOptions opts;
    opts.model_dir = model_dir;
    opts.seed = seed;
    opts.greedy = greedy;
    opts.flow_cut_prompt = flow_cut_prompt;
    opts.n_gpu_layers = n_gpu_layers;
    opts.backend = backend;
    opts.vulkan_device = vulkan_device;
    opts.n_threads = n_threads;
    opts.default_controls = controls;
    if (!prompt_text.empty()) opts.prompt_text = prompt_text;
    if (!reference_audio.empty()) opts.reference_audio = reference_audio;
    if (!s3tok_gguf.empty()) opts.s3tok_gguf_path = s3tok_gguf;
    if (!campplus_gguf.empty()) opts.campplus_gguf_path = campplus_gguf;
    if (!voice_gguf.empty()) opts.voice_gguf_path = voice_gguf;
    if (!backends_dir.empty()) opts.backends_dir = backends_dir;
    if (!opencl_cache_dir.empty()) opts.opencl_cache_dir = opencl_cache_dir;

    try {
        fprintf(stderr, "loading model from %s ...\n", model_dir.c_str());
        Engine engine(opts);
        fprintf(stderr, "synthesizing: \"%s\"\n", text.c_str());
        auto res = engine.synthesize(text);
        fprintf(stderr, "  %zu samples  %.2fs  %d Hz (backend %s%s)\n",
                res.pcm.size(), res.duration_s, res.sample_rate, engine.backend_name().c_str(),
                engine.gpu_unsupported() ? ", GPU present but declined" : "");
        const auto & t = res.timings;
        fprintf(stderr,
                "  timings ms: lm_prefill %.0f lm_decode %.0f flow_frontend %.0f "
                "dit_euler %.0f hift_f0 %.0f hift_source %.0f hift_stft %.0f "
                "hift_decode %.0f total %.0f (decode_steps %d tokens %d)\n",
                t.lm_prefill_ms, t.lm_decode_ms, t.flow_frontend_ms, t.dit_euler_ms,
                t.hift_f0_ms, t.hift_source_ms, t.hift_stft_ms, t.hift_decode_ms,
                t.total_ms, t.n_decode_steps, t.n_speech_tokens);
        write_wav(out, res.pcm, res.sample_rate);
    } catch (const std::exception & e) {
        fprintf(stderr, "cosyvoice-cli: error: %s\n", e.what());
        return 1;
    }
    fprintf(stderr, "wrote %s\n", out.c_str());
    return 0;
}
