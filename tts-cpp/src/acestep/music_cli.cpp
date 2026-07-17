// music-cli: end-to-end ACE-Step text-to-music harness (QVAC-21921).
//
// Drives tts_cpp::acestep::Engine (text prompt -> LM codes -> FSQ detok ->
// text/cond encoders -> DiT flow-matching -> VAE) and writes a 48 kHz stereo
// WAV. This is the first fully-native music generation path (no acestep.cpp
// binaries) — the same C++ that the @qvac/audiogen-ggml addon links.
//
// Usage:
//   music-cli --models <dir>            [--out song.wav] [--dur 8] [--seed 42]
//   music-cli --dit dit.gguf --lm lm.gguf --text emb.gguf --vae vae.gguf ...
//   optional: --caption "..." --lyrics "..." --steps 8 --shift 3.0
//             --bpm 128 --key "C major" --tsig 4/4 --lang en

#include "tts-cpp/acestep/engine.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * arg_val(int argc, char ** argv, const char * key) {
    for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], key)) return argv[i + 1];
    return nullptr;
}

static bool arg_flag(int argc, char ** argv, const char * key) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], key)) return true;
    return false;
}

// --- tiny JSON field readers (flat object; good enough for a request json) ---
static std::string read_file(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string s((size_t) n, '\0');
    size_t rd = fread(&s[0], 1, (size_t) n, f);
    fclose(f);
    s.resize(rd);
    return s;
}

// Return the raw value token after "key": . Handles "string" or bareword/number.
static bool json_field(const std::string & j, const char * key, std::string & out) {
    std::string needle = std::string("\"") + key + "\"";
    size_t      p      = j.find(needle);
    if (p == std::string::npos) return false;
    p = j.find(':', p + needle.size());
    if (p == std::string::npos) return false;
    p++;
    while (p < j.size() && (j[p] == ' ' || j[p] == '\t' || j[p] == '\n' || j[p] == '\r')) p++;
    if (p >= j.size()) return false;
    if (j[p] == '"') {
        size_t e = ++p;
        std::string v;
        while (e < j.size() && j[e] != '"') {
            if (j[e] == '\\' && e + 1 < j.size()) { v += j[e + 1]; e += 2; continue; }
            v += j[e++];
        }
        out = v;
    } else {
        size_t e = p;
        while (e < j.size() && j[e] != ',' && j[e] != '}' && j[e] != '\n') e++;
        out = j.substr(p, e - p);
        while (!out.empty() && (out.back() == ' ' || out.back() == '\r' || out.back() == '\t')) out.pop_back();
    }
    return true;
}

static void wav_write(const char * path, const std::vector<float> & pcm, int frames, int rate) {
    float peak = 1e-9f;
    for (int i = 0; i < frames * 2; i++) peak = std::fmax(peak, std::fabs(pcm[i]));
    float  gain = 0.9f / peak;
    FILE * f    = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    const int ch = 2, bits = 16;
    uint32_t  db  = (uint32_t) frames * ch * (bits / 8);
    auto      w32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto      w16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f); w32(36 + db); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); w32(16); w16(1); w16(ch); w32((uint32_t) rate);
    w32((uint32_t) rate * ch * (bits / 8)); w16(ch * (bits / 8)); w16(bits);
    fwrite("data", 1, 4, f); w32(db);
    for (int i = 0; i < frames * 2; i++) {
        float v = pcm[i] * gain * 32767.0f;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        w16((uint16_t) (int16_t) lrintf(v));
    }
    fclose(f);
    fprintf(stderr, "[music-cli] wrote %s: %d frames, %.2fs @ %d Hz stereo\n", path, frames,
            (float) frames / rate, rate);
}

int main(int argc, char ** argv) {
    using namespace tts_cpp::acestep;

    EngineOptions o;
    o.verbose = true;
    if (arg_val(argc, argv, "--models")) o.models_dir = arg_val(argc, argv, "--models");
    if (arg_val(argc, argv, "--dit"))    o.dit_model_path = arg_val(argc, argv, "--dit");
    if (arg_val(argc, argv, "--lm"))     o.lm_model_path = arg_val(argc, argv, "--lm");
    if (arg_val(argc, argv, "--text"))   o.text_enc_model_path = arg_val(argc, argv, "--text");
    if (arg_val(argc, argv, "--vae"))    o.vae_model_path = arg_val(argc, argv, "--vae");

    if (o.models_dir.empty() && o.dit_model_path.empty()) {
        fprintf(stderr, "usage: music-cli --models <dir> [--out song.wav] [--dur 8] [--seed 42]\n");
        return 1;
    }

    GenerateParams p;
    p.caption         = arg_val(argc, argv, "--caption") ? arg_val(argc, argv, "--caption")
                                                         : "Upbeat pop rock with driving electric guitars, punchy drums and a catchy hook";
    p.lyrics          = arg_val(argc, argv, "--lyrics") ? arg_val(argc, argv, "--lyrics") : "[Instrumental]";
    p.duration        = arg_val(argc, argv, "--dur")   ? (float) atof(arg_val(argc, argv, "--dur"))   : 8.0f;
    p.inference_steps = arg_val(argc, argv, "--steps") ? atoi(arg_val(argc, argv, "--steps"))         : 0;
    p.shift           = arg_val(argc, argv, "--shift") ? (float) atof(arg_val(argc, argv, "--shift")) : 0.0f;
    p.seed            = arg_val(argc, argv, "--seed")  ? atoi(arg_val(argc, argv, "--seed"))          : 42;
    if (arg_val(argc, argv, "--bpm"))  p.bpm = atoi(arg_val(argc, argv, "--bpm"));
    if (arg_val(argc, argv, "--key"))  p.keyscale = arg_val(argc, argv, "--key");
    if (arg_val(argc, argv, "--tsig")) p.timesignature = arg_val(argc, argv, "--tsig");
    if (arg_val(argc, argv, "--lang")) p.vocal_language = arg_val(argc, argv, "--lang");
    if (arg_val(argc, argv, "--temp")) p.lm_temperature = (float) atof(arg_val(argc, argv, "--temp"));
    if (arg_val(argc, argv, "--cfg"))  p.lm_cfg_scale = (float) atof(arg_val(argc, argv, "--cfg"));
    if (arg_val(argc, argv, "--topk")) p.lm_top_k = atoi(arg_val(argc, argv, "--topk"));
    if (arg_val(argc, argv, "--topp")) p.lm_top_p = (float) atof(arg_val(argc, argv, "--topp"));
    if (arg_flag(argc, argv, "--no-phase1")) p.lm_phase1 = false;

    // --req <json>: load caption/lyrics/metas and (if present) audio_codes to
    // bypass our LM — used for parity against acestep.cpp's ace-lm output.
    if (arg_val(argc, argv, "--req")) {
        std::string j = read_file(arg_val(argc, argv, "--req"));
        if (j.empty()) { fprintf(stderr, "[music-cli] cannot read --req json\n"); return 1; }
        std::string v;
        if (json_field(j, "caption", v)) p.caption = v;
        if (json_field(j, "lyrics", v)) p.lyrics = v;
        if (json_field(j, "keyscale", v)) p.keyscale = v;
        if (json_field(j, "timesignature", v)) p.timesignature = v;
        if (json_field(j, "vocal_language", v)) p.vocal_language = v;
        if (json_field(j, "bpm", v)) p.bpm = atoi(v.c_str());
        if (json_field(j, "duration", v)) p.duration = (float) atof(v.c_str());
        if (json_field(j, "shift", v)) p.shift = (float) atof(v.c_str());
        if (json_field(j, "inference_steps", v)) p.inference_steps = atoi(v.c_str());
        if (json_field(j, "seed", v)) p.seed = atoi(v.c_str());
        if (json_field(j, "audio_codes", v) && !v.empty()) {
            size_t start = 0;
            while (start < v.size()) {
                size_t comma = v.find(',', start);
                std::string tok = v.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                if (!tok.empty()) p.audio_codes.push_back(atoi(tok.c_str()));
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            fprintf(stderr, "[music-cli] --req: %zu pre-supplied audio codes (LM bypass)\n", p.audio_codes.size());
        }
    }

    const char * out_path = arg_val(argc, argv, "--out") ? arg_val(argc, argv, "--out") : "music_out.wav";

    std::unique_ptr<Engine> eng;
    try {
        eng = Engine::create(o);
    } catch (const std::exception & e) {
        fprintf(stderr, "[music-cli] engine create failed: %s\n", e.what());
        return 1;
    }

    auto progress = [](const std::string & stage, int step, int total) -> bool {
        fprintf(stderr, "[music-cli] %-4s %d/%d\n", stage.c_str(), step, total);
        return true;
    };

    GenerateResult r;
    try {
        r = eng->generate(p, progress);
    } catch (const std::exception & e) {
        fprintf(stderr, "[music-cli] generate failed: %s\n", e.what());
        return 1;
    }

    if (r.pcm.empty()) { fprintf(stderr, "[music-cli] no audio (cancelled?)\n"); return 1; }

    int frames = (int) (r.pcm.size() / 2);
    fprintf(stderr, "[music-cli] generated %d codes, seed=%lld, %d frames (%.2fs)\n", r.metadata.n_codes,
            r.metadata.seed, frames, (float) frames / r.sample_rate);
    wav_write(out_path, r.pcm, frames, r.sample_rate);
    return 0;
}
