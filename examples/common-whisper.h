#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Read WAV audio file and store the PCM data into pcmf32
// fname can be a buffer of WAV data instead of a filename
// The sample rate of the audio must be equal to COMMON_SAMPLE_RATE
// If stereo flag is set and the audio has 2 channels, the pcmf32s will contain 2 channel PCM
bool read_audio_data(
        const std::string & fname,
        std::vector<float> & pcmf32,
        std::vector<std::vector<float>> & pcmf32s,
        bool stereo);

// convert timestamp to string, 6000 -> 01:00.000
std::string to_timestamp(int64_t t, bool comma = false);

// given a timestamp get the sample
int timestamp_to_sample(int64_t t, int n_samples, int whisper_sample_rate);

// write text to file, and call system("command voice_id file")
bool speak_with_file(const std::string & command, const std::string & text, const std::string & path, int voice_id);

struct speak_metrics {
    bool ok = true;
    int64_t startup_ms = 0;
    int64_t total_ms = 0;
};

bool speak_with_file(const std::string & command, const std::string & text, const std::string & path, int voice_id, speak_metrics * metrics);

enum class tts_mode {
    script,
    piper_persistent,
};

struct tts_worker_params {
    tts_mode mode = tts_mode::script;
    std::string command;
    std::string file_path;
    std::string piper_model;
    std::string piper_output_cmd;
    int voice_id = 0;
};

struct tts_worker_turn_metrics {
    bool ok = true;
    int chunks = 0;
    int64_t startup_ms = 0;
    int64_t total_ms = 0;
};

class tts_worker {
public:
    explicit tts_worker(tts_worker_params params);
    ~tts_worker();

    bool start();
    void begin_turn();
    bool submit(const std::string & text);
    tts_worker_turn_metrics end_turn();
    void stop();

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
