#include "mcp-handler.hpp"
#include "common.h"
#include "common-whisper.h"
#include <cstdio>
#include <stdexcept>

namespace mcp {

// JSON-RPC 2.0 error codes
enum class MCPError : int {
    // Standard JSON-RPC errors
    PARSE_ERROR      = -32700,
    INVALID_REQUEST  = -32600,
    METHOD_NOT_FOUND = -32601,
    INVALID_PARAMS   = -32602,
    INTERNAL_ERROR   = -32603,

    // MCP-specific errors
    MODEL_NOT_LOADED = 1001,
    AUDIO_FILE_ERROR = 1002,
    TRANSCRIPTION_FAILED = 1003
};

Handler::Handler(Transport * transport,
                 const struct mcp_params & mparams,
                 const struct whisper_params & wparams,
                 const std::string & model_path)
    : transport(transport), ctx(nullptr), model_path(model_path), model_loaded(false)
    ,mparams(mparams), wparams(wparams) {
    if (!transport) {
        throw std::invalid_argument("Transport cannot be null");
    }
}

Handler::~Handler() {
    if (ctx) {
        whisper_free(ctx);
    }
}

bool Handler::handle_message(const json & request) {
    // Validate JSON-RPC 2.0 format
    if (!request.contains("jsonrpc") || request["jsonrpc"] != "2.0") {
        fprintf(stderr, "Invalid JSON-RPC format\n");
        return false;
    }

    // Extract request ID (can be null for notifications)
    json id = nullptr;
    if (request.contains("id")) {
        id = request["id"];
    }

    // Extract method
    std::string method = request.value("method", "");
    if (method.empty()) {
        send_error(id, static_cast<int>(MCPError::INVALID_REQUEST), "Invalid request: missing method");
        return true;
    }

    fprintf(stderr, "Processing method: %s\n", method.c_str());

    // TODO: add missing methods from specification
    try {
        if (method == "initialize") {
            handle_initialize(id, request.value("params", json::object()));
        }
        else if (method == "tools/list") {
            handle_list_tools(id);
        }
        else if (method == "tools/call") {
            handle_tool_call(id, request.value("params", json::object()));
        }
        else if (method == "notifications/initialized") {
            handle_notification_initialized();
        }
        else {
            send_error(id, static_cast<int>(MCPError::METHOD_NOT_FOUND), "Method not found: " + method);
        }
        return true;
    } catch (const std::exception & e) {
        fprintf(stderr, "Exception in message handler: %s\n", e.what());
        send_error(id, static_cast<int>(MCPError::INTERNAL_ERROR), "Internal error: " + std::string(e.what()));
        return true;
    }
}

void Handler::handle_initialize(const json & id, const json & params) {
    fprintf(stderr, "Initializing whisper server with model: %s\n", model_path.c_str());

    if (!load_model()) {
        send_error(id, static_cast<int>(MCPError::INTERNAL_ERROR), "Failed to load whisper model");

        return;
    }

    json result = {
        {"protocolVersion", "2024-11-05"},
        {"capabilities", {
            {"tools", json::object()}
        }},
        {"serverInfo", {
            {"name", "whisper-mcp-server"},
            {"version", "1.0.0"}
        }}
    };

    send_result(id, result);
}

void Handler::handle_list_tools(const json & id) {
    fprintf(stderr, "Listing available tools\n");

    json result = {
        {"tools", json::array({
            {
                {"name", "transcribe"},
                {"description", "Transcribe audio file using whisper.cpp"},
                {"inputSchema", {
                    {"type", "object"},
                    {"properties", {
                        {"file", {
                            {"type", "string"},
                            {"description", "Path to audio file"}
                        }},
                        {"language", {
                            {"type", "string"},
                            {"description", "Language code (optional, auto-detect if not specified)"},
                            {"default", "auto"}
                        }},
                        {"translate", {
                            {"type", "boolean"},
                            {"description", "Translate to English"},
                            {"default", false}
                        }}
                    }},
                    {"required", json::array({"file"})}
                }}
            },
            {
                {"name", "model_info"},
                {"description", "Get information about loaded model"},
                {"inputSchema", {
                    {"type", "object"},
                    {"properties", json::object()}
                }}
            }
        })}
    };

    send_result(id, result);
}

void Handler::handle_tool_call(const json & id, const json & params) {
    if (!params.contains("name")) {
        send_error(id, static_cast<int>(MCPError::INVALID_PARAMS), "Missing required parameter: name");
        return;
    }

    std::string tool_name = params["name"];
    json arguments = params.value("arguments", json::object());

    if (tool_name == "transcribe") {
        json result = create_transcribe_result(arguments);
        send_result(id, result);
    }
    else if (tool_name == "model_info") {
        json result = create_model_info_result();
        send_result(id, result);
    }
    else {
        send_error(id, static_cast<int>(MCPError::METHOD_NOT_FOUND), "Unknown tool: " + tool_name);
    }
}

void Handler::handle_notification_initialized() {
    fprintf(stderr, "Client initialization completed\n");
}

void Handler::send_result(const json & id, const json & result) {
    json response = {
        {"jsonrpc", "2.0"},
        {"result", result}
    };

    if (!id.is_null()) {
        response["id"] = id;
    }

    transport->send_response(response);
}

void Handler::send_error(const json & id, int code, const std::string & message) {
    json response = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"error", {
            {"code", code},
            {"message", message}
        }}
    };

    transport->send_response(response);
}

bool Handler::load_model() {
    if (model_loaded) {
        return true;
    }

    fprintf(stderr, "Loading whisper model from: %s\n", model_path.c_str());

    whisper_context_params cparams = whisper_context_default_params();
    ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);

    if (!ctx) {
        fprintf(stderr, "Failed to load model: %s\n", model_path.c_str());
        return false;
    }

    model_loaded = true;
    fprintf(stderr, "Model loaded successfully!\n");
    return true;
}

std::string Handler::transcribe_file(const std::string & filepath,
                                     const std::string & language,
                                     bool translate) {
    if (!model_loaded) {
        throw std::runtime_error("Model not loaded");
    }

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    if (language != "auto" && whisper_lang_id(language.c_str()) == -1) {
        throw std::runtime_error("Unknown language: " + language);
    }

    if (language != "auto") {
        wparams.language = language.c_str();
    } else {
        wparams.language = "auto";
    }

    wparams.translate = translate;
    wparams.print_progress = false;
    wparams.print_timestamps = false;

    std::vector<float> pcmf32;
    if (!load_audio_file(filepath, pcmf32)) {
        throw std::runtime_error("Failed to load audio file: " + filepath);
    }

    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        throw std::runtime_error("Whisper inference failed");
    }

    std::string result;
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);
        result += text;
    }

    return result;
}

bool Handler::load_audio_file(const std::string & fname_inp, std::vector<float> & pcmf32) {
    fprintf(stderr, "Loading audio file: %s\n", fname_inp.c_str());
    std::vector<std::vector<float>> pcmf32s;

    if (!::read_audio_data(fname_inp, pcmf32, pcmf32s, wparams.diarize)) {
        fprintf(stderr, "Failed to read audio file: %s\n", fname_inp.c_str());
        return false;
    }

    fprintf(stderr, "Successfully loaded %s\n", fname_inp.c_str());
    return true;
}

json Handler::create_transcribe_result(const json & arguments) {
    try {
        if (!arguments.contains("file")) {
            throw std::runtime_error("Missing required parameter: file");
        }

        std::string file_path = arguments["file"];
        std::string language = arguments.value("language", "auto");
        bool translate = arguments.value("translate", false);

        std::string transcription = transcribe_file(file_path, language, translate);

        return json{
            {"content", json::array({
                {
                    {"type", "text"},
                    {"text", transcription}
                }
            })}
        };

    } catch (const std::exception & e) {
        throw std::runtime_error("Transcription failed: " + std::string(e.what()));
    }
}

json Handler::create_model_info_result() {
    if (!model_loaded) {
        throw std::runtime_error("No model loaded");
    }

    json model_info = {
        {"model_path", model_path},
        {"model_loaded", model_loaded},
        {"vocab_size", whisper_n_vocab(ctx)},
        {"n_text_ctx", whisper_n_text_ctx(ctx)},
        {"n_audio_ctx", whisper_n_audio_ctx(ctx)},
        {"is_multilingual", whisper_is_multilingual(ctx)}
    };

    return json{
        {"content", json::array({
            {
                {"type", "text"},
                {"text", "Model Information:\n" + model_info.dump(2)}
            }
        })}
    };
}

} // namespace mcp
