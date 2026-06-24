#include "neural_diarize.h"

#include "json.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

using json = nlohmann::json;
namespace fs = std::filesystem;

// Quote a command argument so paths with spaces survive the shell that
// std::system() uses (cmd.exe on Windows, /bin/sh elsewhere).
static std::string shq(const std::string & s) {
#ifdef _WIN32
    // cmd.exe: wrap in double quotes, double any embedded quote
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else          out += c;
    }
    out += "\"";
    return out;
#else
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else           out += c;
    }
    out += "'";
    return out;
#endif
}

// write mono float PCM as a 16-bit little-endian WAV at the given sample rate
static bool write_wav16(const fs::path & path, const std::vector<float> & pcm, uint32_t sr) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const uint32_t n          = (uint32_t) pcm.size();
    const uint32_t data_bytes = n * 2;
    auto w32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) f.put((char) ((v >> (8 * i)) & 0xff)); };
    auto w16 = [&](uint16_t v) { f.put((char) (v & 0xff)); f.put((char) ((v >> 8) & 0xff)); };
    f.write("RIFF", 4); w32(36 + data_bytes); f.write("WAVE", 4);
    f.write("fmt ", 4); w32(16); w16(1); w16(1); w32(sr); w32(sr * 2); w16(2); w16(16);
    f.write("data", 4); w32(data_bytes);
    for (float s : pcm) {
        long v = std::lround(s * 32767.0f);
        if (v >  32767) v =  32767;
        if (v < -32768) v = -32768;
        w16((uint16_t) (int16_t) v);
    }
    return (bool) f;
}

bool neural_diarize(const std::string & python,
                    const std::string & script,
                    const std::vector<float> & pcm16k_mono,
                    const std::vector<std::pair<int64_t, int64_t>> & spans_cs,
                    int num_speakers,
                    const std::string & emb_model,
                    std::vector<int> & out_speakers,
                    std::string & error) {
    out_speakers.clear();
    if (spans_cs.empty()) {
        error = "nothing to diarize (no segments)";
        return false;
    }
    if (pcm16k_mono.empty()) {
        error = "no audio to diarize";
        return false;
    }

    std::error_code ec;
    const fs::path tmp = fs::temp_directory_path(ec);
    const fs::path in_json = tmp / "whisper-gui-diar.json";
    const fs::path in_wav  = tmp / "whisper-gui-diar.wav";
    const fs::path log     = tmp / "whisper-gui-diar.log";

    // write the exact audio whisper saw as a 16 kHz mono WAV for the helper
    if (!write_wav16(in_wav, pcm16k_mono, 16000)) {
        error = "cannot write temp audio " + in_wav.string();
        return false;
    }

    // write the whisper-format JSON the helper expects (timestamps only)
    {
        json j;
        j["transcription"] = json::array();
        for (const auto & s : spans_cs) {
            j["transcription"].push_back({{"from", s.first}, {"to", s.second}, {"text", ""}});
        }
        std::ofstream f(in_json);
        if (!f) { error = "cannot write temp file " + in_json.string(); return false; }
        f << j.dump();
    }

    std::ostringstream cmd;
    cmd << shq(python) << ' ' << shq(script) << ' ' << shq(in_wav.string())
        << " --json " << shq(in_json.string());
    if (num_speakers > 0)     cmd << " --speakers " << num_speakers;
    if (!emb_model.empty())   cmd << " --emb-model " << shq(emb_model);
    cmd << " > " << shq(log.string()) << " 2>&1";

    const int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::ifstream lf(log);
        std::stringstream ss;
        ss << lf.rdbuf();
        std::string tail = ss.str();
        if (tail.size() > 800) tail = "..." + tail.substr(tail.size() - 800);
        error = "diarization helper failed.\nCommand: " + cmd.str() + "\n" + tail;
        return false;
    }

    json j;
    try {
        std::ifstream f(in_json);
        f >> j;
    } catch (const std::exception & e) {
        error = std::string("could not parse helper output: ") + e.what();
        return false;
    }

    out_speakers.assign(spans_cs.size(), -1);
    if (j.contains("transcription") && j["transcription"].is_array()) {
        const auto & items = j["transcription"];
        for (size_t i = 0; i < items.size() && i < out_speakers.size(); ++i) {
            const auto & it = items[i];
            if (it.contains("speaker") && it["speaker"].is_number_integer()) {
                out_speakers[i] = it["speaker"].get<int>() - 1; // 1-based -> 0-based
            }
        }
    }
    return true;
}
