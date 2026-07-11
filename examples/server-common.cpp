#include "server-common.h"
#include "common-whisper.h"

#include <cstdio>
#include <csignal>
#include <random>
#include <sstream>
#include <memory>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>

#if defined (_WIN32)
#include <windows.h>
#endif

const std::string json_format   = "json";
const std::string text_format   = "text";
const std::string srt_format    = "srt";
const std::string vjson_format  = "verbose_json";
const std::string vtt_format    = "vtt";

namespace {
    std::function<void()> g_shutdown_callback;
    std::atomic_flag g_is_terminating = ATOMIC_FLAG_INIT;

    void signal_handler(int /*signal*/) {
        if (g_is_terminating.test_and_set()) {
            fprintf(stderr, "Received second interrupt, terminating immediately.\n");
            exit(1);
        }
        if (g_shutdown_callback) {
            g_shutdown_callback();
        }
    }
}

bool parse_str_to_bool(const std::string & s) {
    if (s == "true" || s == "1" || s == "yes" || s == "y") {
        return true;
    }
    return false;
}

bool check_ffmpeg_availability() {
    int result = system("ffmpeg -version");
    if (result == 0) {
        std::cout << "ffmpeg is available." << std::endl;
    } else {
        std::cout << "ffmpeg is not found. Please ensure that ffmpeg is installed "
                  << "and that its executable is included in your system's PATH. ";
        exit(0);
    }
    return true;
}

std::string generate_temp_filename(const std::string & path, const std::string & prefix, const std::string & extension) {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);

    static std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<long long> dist(0, 1e9);

    std::stringstream ss;
    ss << path
       << std::filesystem::path::preferred_separator
       << prefix
       << "-"
       << std::put_time(std::localtime(&now_time_t), "%Y%m%d-%H%M%S")
       << "-"
       << dist(rng)
       << extension;

    return ss.str();
}

bool convert_to_wav(const std::string & temp_filename, std::string & error_resp, bool stereo) {
    std::ostringstream cmd_stream;
    std::string converted_filename_temp = temp_filename + "_temp.wav";
    cmd_stream << "ffmpeg -i \"" << temp_filename << "\" -y -ar 16000 -ac " << (stereo ? 2 : 1) << " -c:a pcm_s16le \"" << converted_filename_temp << "\" 2>&1";
    std::string cmd = cmd_stream.str();

    int status = std::system(cmd.c_str());
    if (status != 0) {
        error_resp = "{\"error\":\"FFmpeg conversion failed.\"}";
        return false;
    }

    if (remove(temp_filename.c_str()) != 0) {
        error_resp = "{\"error\":\"Failed to remove the original file.\"}";
        return false;
    }

    if (rename(converted_filename_temp.c_str(), temp_filename.c_str()) != 0) {
        error_resp = "{\"error\":\"Failed to rename the temporary file.\"}";
        return false;
    }
    return true;
}

void setup_signal_handler(std::function<void()> shutdown_callback) {
    g_shutdown_callback = std::move(shutdown_callback);

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
}

static std::string ms_to_timestamp(int64_t t_ms, bool comma = false) {
    // to_timestamp expects centiseconds, our adapter uses milliseconds
    return ::to_timestamp(t_ms / 10, comma);
}


std::string format_text(const transcription_result & result) {
    std::stringstream ss;
    const int n_segments = result.n_segments();
    for (int i = 0; i < n_segments; ++i) {
        auto seg = result.get_segment(i);
        auto speaker = result.get_speaker(i);
        ss << speaker << seg.text << "\n";
    }
    return ss.str();
}

std::string format_srt(const transcription_result & result, int offset_n) {
    std::stringstream ss;
    const int n_segments = result.n_segments();
    for (int i = 0; i < n_segments; ++i) {
        auto seg = result.get_segment(i);
        auto speaker = result.get_speaker(i);

        ss << i + 1 + offset_n << "\n";
        ss << ms_to_timestamp(seg.t0, true) << " --> " << ms_to_timestamp(seg.t1, true) << "\n";
        ss << speaker << seg.text << "\n\n";
    }
    return ss.str();
}

std::string format_vtt(const transcription_result & result) {
    std::stringstream ss;
    ss << "WEBVTT\n\n";

    const int n_segments = result.n_segments();
    for (int i = 0; i < n_segments; ++i) {
        auto seg = result.get_segment(i);
        std::string speaker_tag;

        auto speaker_id = result.get_speaker(i);
        if (!speaker_id.empty()) {
            speaker_tag = "<v Speaker" + speaker_id + ">";
        }

        ss << ms_to_timestamp(seg.t0) << " --> " << ms_to_timestamp(seg.t1) << "\n";
        ss << speaker_tag << seg.text << "\n\n";
    }
    return ss.str();
}

std::string format_json(const transcription_result & result) {
    std::string text = format_text(result);
    json jres = json{{"text", text}};
    return jres.dump(-1, ' ', false, json::error_handler_t::replace);
}

std::string format_verbose_json(
        const transcription_result & result,
        float temperature,
        float duration,
        bool no_timestamps,
        bool token_timestamps) {
    std::string text = format_text(result);
    std::string task     = result.get_task();
    std::string language = result.get_language();

    json jres = json{
        {"task", task},
        {"language", language},
        {"duration", duration},
        {"text", text},
        {"segments", json::array()}
    };

    // Merge language probability data into the top-level response.
    // Adapters return a json object whose keys are merged directly, allowing
    // model-specific fields (e.g. whisper's detected_language) to appear at
    // the top level alongside the standard language_probabilities map.
    json lang_data = result.get_language_probabilities();
    for (auto & [key, val] : lang_data.items()) {
        jres[key] = val;
    }

    const int n_segments = result.n_segments();
    for (int i = 0; i < n_segments; ++i) {
        auto seg = result.get_segment(i);

        json segment = json{
            {"id", i},
            {"text", seg.text},
        };

        if (!no_timestamps) {
            segment["start"] = seg.t0 * 0.001f;  // ms -> seconds
            segment["end"] = seg.t1 * 0.001f;
        }

        auto speaker_id = result.get_speaker(i);
        if (!speaker_id.empty()) {
            segment["speaker"] = speaker_id;
        }

        // Build word-level tokens by merging partial UTF-8 tokens
        std::vector<json> words;
        int n_tokens = (int)seg.tokens.size();
        float total_logprob = 0.0f;

        for (int j = 0; j < n_tokens; ++j) {
            auto & tok = seg.tokens[j];

            // Merge trailing partial UTF-8 bytes into complete words
            std::string word_text = tok.text;
            int64_t word_t1 = tok.t1;

            while (j + 1 < n_tokens) {
                int trailing = utf8_trailing_bytes_needed(word_text);
                if (trailing <= 0) break;

                ++j;
                auto & next_tok = seg.tokens[j];
                word_text += next_tok.text;
                if (next_tok.t1 > word_t1) {
                    word_t1 = next_tok.t1;
                }
            }

            json word = json{{"word", word_text}};
            if (!no_timestamps && token_timestamps) {
                word["start"] = tok.t0 * 0.001f;
                word["end"] = word_t1 * 0.001f;
            }
            word["probability"] = tok.prob;

            // Approximate logprob from probability
            float logprob = tok.prob > 0.0f ? std::log(tok.prob + 1e-10f) : -1e10f;
            total_logprob += logprob;

            words.push_back(word);
        }

        segment["words"] = words;
        segment["tokens"] = json::array();
        for (auto & tok : seg.tokens) {
            segment["tokens"].push_back(tok.id);
        }

        segment["temperature"] = temperature;
        int n_word_tokens = (int)seg.tokens.size();
        segment["avg_logprob"] = n_word_tokens > 0 ? total_logprob / n_word_tokens : 0.0f;
        segment["no_speech_prob"] = seg.no_speech_prob;

        jres["segments"].push_back(segment);
    }

    return jres.dump(-1, ' ', false, json::error_handler_t::replace);
}

void setup_server_common(
        httplib::Server & svr,
        const server_params & sparams,
        std::atomic<server_state> & state,
        std::function<void(const httplib::Request &, httplib::Response &)> load_handler,
        std::function<void(const httplib::Request &, httplib::Response &)> inference_handler,
        const std::string & default_content,
        const std::string & server_name) {

    svr.set_default_headers({
        {"Server", server_name},
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Headers", "content-type, authorization"}
    });

    // Default index page
    svr.Get(sparams.request_path + "/", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(default_content, "text/html");
        return false;
    });

    // CORS preflight
    svr.Options(sparams.request_path + sparams.inference_path,
                [&](const httplib::Request &, httplib::Response &) {});

    // Inference endpoint
    svr.Post(sparams.request_path + sparams.inference_path, inference_handler);

    // Model reload endpoint
    if (load_handler) {
        svr.Post(sparams.request_path + "/load", load_handler);
    }

    // Health check
    svr.Get(sparams.request_path + "/health", [&](const httplib::Request &, httplib::Response & res) {
        server_state current_state = state.load();
        if (current_state == SERVER_STATE_READY) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        } else {
            res.set_content("{\"status\":\"loading model\"}", "application/json");
            res.status = 503;
        }
    });

    // Exception handler
    svr.set_exception_handler([](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        const char fmt[] = "500 Internal Server Error\n%s";
        char buf[BUFSIZ];
        try {
            std::rethrow_exception(std::move(ep));
        } catch (std::exception & e) {
            snprintf(buf, sizeof(buf), fmt, e.what());
        } catch (...) {
            snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
        }
        res.set_content(buf, "text/plain");
        res.status = 500;
    });

    // Error handler
    svr.set_error_handler([](const httplib::Request & req, httplib::Response & res) {
        if (res.status == 400) {
            res.set_content("Invalid request", "text/plain");
        } else if (res.status != 500) {
            res.set_content("File Not Found (" + req.path + ")", "text/plain");
            res.status = 404;
        }
    });

    svr.set_read_timeout(sparams.read_timeout);
    svr.set_write_timeout(sparams.write_timeout);
}
