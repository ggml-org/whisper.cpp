#include "neural_diarize.h"

#include "json.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

using json = nlohmann::json;
namespace fs = std::filesystem;

// POSIX single-quote a string so paths with spaces/specials survive the shell
static std::string shq(const std::string & s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else           out += c;
    }
    out += "'";
    return out;
}

bool neural_diarize(const std::string & python,
                    const std::string & script,
                    const std::string & audio_path,
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

    std::error_code ec;
    const fs::path tmp = fs::temp_directory_path(ec);
    const fs::path in_json = tmp / "whisper-gui-diar.json";
    const fs::path log     = tmp / "whisper-gui-diar.log";

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
    cmd << shq(python) << ' ' << shq(script) << ' ' << shq(audio_path)
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
