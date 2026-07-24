// cosyvoice-cli: end-to-end CosyVoice3 text -> 24 kHz WAV via the public
// tts_cpp::cosyvoice::Engine (the same API the JS addon / @qvac/sdk drive).
//
// Point --model-dir at a folder assembled by scripts/assemble-cosyvoice3-model.py
// (cosyvoice3-{llm,flow,hift}*.gguf + voice.gguf + vocab.json + merges.txt):
//
//   cosyvoice-cli --model-dir models/cosyvoice3-0.5b \
//       --text "Hello from a fully on-device C++ pipeline." --out out.wav
//
// CPU-only (iteration 1).  Uses the baked default voice; zero-shot from
// arbitrary reference audio awaits the native S3/CAM++ port (stage 6).

#include "tts-cpp/cosyvoice/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    std::string out = "cosyvoice_out.wav", prompt_text, instruct_text, voice_gguf;
    int seed = 42;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--text" && i + 1 < argc) text = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--prompt-text" && i + 1 < argc) prompt_text = argv[++i];
        else if (a == "--instruct" && i + 1 < argc) instruct_text = argv[++i];
        else if (a == "--voice-gguf" && i + 1 < argc) voice_gguf = argv[++i];
        else if (a == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s --model-dir DIR [--text ...] [--instruct \"...\"] [--voice-gguf voice.gguf] [--out out.wav] [--seed N]\n", argv[0]); return 1; }
    }
    if (model_dir.empty()) { fprintf(stderr, "need --model-dir\n"); return 1; }

    EngineOptions opts;
    opts.model_dir = model_dir;
    opts.seed = seed;
    if (!prompt_text.empty()) opts.prompt_text = prompt_text;
    if (!instruct_text.empty()) opts.instruct_text = instruct_text;
    if (!voice_gguf.empty()) opts.voice_gguf_path = voice_gguf;

    fprintf(stderr, "loading model from %s ...\n", model_dir.c_str());
    Engine engine(opts);
    fprintf(stderr, "synthesizing: \"%s\"\n", text.c_str());
    auto res = engine.synthesize(text);
    fprintf(stderr, "  %zu samples  %.2fs  %d Hz (%s)\n",
            res.pcm.size(), res.duration_s, res.sample_rate, engine.backend_name().c_str());
    write_wav(out, res.pcm, res.sample_rate);
    fprintf(stderr, "wrote %s\n", out.c_str());
    return 0;
}
