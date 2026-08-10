// cosyvoice-cli: end-to-end CosyVoice3 text -> 24 kHz WAV via the public
// tts_cpp::cosyvoice::Engine (the same API the JS addon / @qvac/sdk drive).
//
// Point --model-dir at a folder assembled by scripts/assemble-cosyvoice3-model.py
// (cosyvoice3-{llm,flow,hift}*.gguf + voice.gguf + vocab.json + merges.txt):
//
//   cosyvoice-cli --model-dir models/cosyvoice3-0.5b \
//       --text "Hello from a fully on-device C++ pipeline." --out out.wav
//
// Pass --n-gpu-layers > 0 to select the GPU path (OpenCL/Adreno only; every
// other GPU is declined and falls back to CPU).  Uses the baked default voice;
// zero-shot from arbitrary reference audio awaits the native S3/CAM++ port.

#include "tts-cpp/cosyvoice/engine.h"
#include "tts-cpp/voice_controls.h"
#include "voice_controls_cli.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

int main(int argc, char ** argv) {
    using namespace tts_cpp::cosyvoice;
    std::string model_dir, text = "Hello from a fully on-device C plus plus pipeline.";
    namespace ctl = tts_cpp::controls;
    constexpr ctl::EngineId kEngine = ctl::EngineId::CosyVoice;

    std::string out = "cosyvoice_out.wav", prompt_text, voice_gguf;
    std::string backends_dir, opencl_cache_dir;
    VoiceControls controls;
    int seed = 42, n_gpu_layers = 0, n_threads = 0;
    bool greedy = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--text" && i + 1 < argc) text = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--prompt-text" && i + 1 < argc) prompt_text = argv[++i];
        else if (a == "--instruct" && i + 1 < argc) controls.instruct_text = argv[++i];
        else if (a == "--emotion" && i + 1 < argc) controls.emotion = argv[++i];
        else if (a == "--pace" && i + 1 < argc) controls.pace = argv[++i];
        else if (a == "--list-emotions") { printf("%s\n", ctl::cli::describe_emotions(kEngine).c_str()); return 0; }
        else if (a == "--list-paces") { printf("%s\n", ctl::cli::describe_paces(kEngine).c_str()); return 0; }
        else if (a == "--voice-gguf" && i + 1 < argc) voice_gguf = argv[++i];
        else if (a == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
        else if ((a == "--n-gpu-layers" || a == "-ngl") && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else if ((a == "--threads" || a == "-t") && i + 1 < argc) n_threads = std::atoi(argv[++i]);
        else if (a == "--backends-dir" && i + 1 < argc) backends_dir = argv[++i];
        else if (a == "--opencl-cache-dir" && i + 1 < argc) opencl_cache_dir = argv[++i];
        else if (a == "--greedy") greedy = true;
        else {
            fprintf(stderr,
                "usage: %s --model-dir DIR [--text ...] [--voice-gguf voice.gguf]\n"
                "          [--emotion NAME] [--pace slow|moderate|fast] [--instruct \"...\"]\n"
                "          [--list-emotions] [--list-paces]\n"
                "          [--out out.wav] [--seed N] [--greedy] [--n-gpu-layers N] [--threads N]\n"
                "          [--backends-dir DIR] [--opencl-cache-dir DIR]\n"
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
    opts.n_gpu_layers = n_gpu_layers;
    opts.n_threads = n_threads;
    opts.default_controls = controls;
    if (!prompt_text.empty()) opts.prompt_text = prompt_text;
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
        write_wav(out, res.pcm, res.sample_rate);
    } catch (const std::exception & e) {
        fprintf(stderr, "cosyvoice-cli: error: %s\n", e.what());
        return 1;
    }
    fprintf(stderr, "wrote %s\n", out.c_str());
    return 0;
}
