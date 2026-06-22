// Common server utilities for whisper.cpp and parakeet.cpp servers
// Extracts shared HTTP infrastructure, response formatting, and request handling.
#pragma once

#include "httplib.h"
#include "json.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

enum server_state {
    SERVER_STATE_LOADING_MODEL,
    SERVER_STATE_READY,
};

struct server_params {
    std::string hostname       = "127.0.0.1";
    std::string public_path    = "examples/server/public";
    std::string request_path   = "";
    std::string inference_path = "/inference";
    std::string tmp_dir        = ".";

    int32_t port          = 8080;
    int32_t read_timeout  = 600;
    int32_t write_timeout = 600;

    bool ffmpeg_converter = false;
};

struct segment_token {
    int32_t id;
    std::string text;
    int64_t t0;      // in ms
    int64_t t1;      // in ms
    float prob;
};

struct segment_info {
    std::string text;
    int64_t t0;      // in ms
    int64_t t1;      // in ms
    float no_speech_prob;
    std::vector<segment_token> tokens;
};

struct transcription_result {
    virtual ~transcription_result() = default;

    virtual int          n_segments()                 const = 0;
    virtual segment_info get_segment(int i)           const = 0;

    virtual std::string  get_speaker(int /*i*/)       const { return {}; }
    virtual std::string  get_language()               const { return {}; }
    virtual json         get_language_probabilities() const { return {}; }
    virtual std::string  get_task()                   const { return "transcribe"; }
};

extern const std::string json_format;
extern const std::string text_format;
extern const std::string srt_format;
extern const std::string vjson_format;
extern const std::string vtt_format;

std::string format_text(const transcription_result & result);
std::string format_srt(const transcription_result & result, int offset_n = 0);
std::string format_vtt(const transcription_result & result);
std::string format_json(const transcription_result & result);
std::string format_verbose_json(
        const transcription_result & result,
        float temperature,
        float duration,
        bool no_timestamps,
        bool token_timestamps);


bool parse_str_to_bool(const std::string & s);

bool check_ffmpeg_availability();

std::string generate_temp_filename(const std::string & path, const std::string & prefix, const std::string & extension);

bool convert_to_wav(const std::string & temp_filename, std::string & error_resp, bool stereo);

void setup_signal_handler(std::function<void()> shutdown_callback);

// Set up common server configuration (CORS, error handlers, timeouts)
void setup_server_common(
        httplib::Server & svr,
        const server_params & sparams,
        std::atomic<server_state> & state,
        std::function<void(const httplib::Request &, httplib::Response &)> load_handler,
        std::function<void(const httplib::Request &, httplib::Response &)> inference_handler,
        const std::string & default_content,
        const std::string & server_name);
