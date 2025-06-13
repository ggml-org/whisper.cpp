#pragma once

#include "mcp-transport.hpp"
#include "mcp-params.hpp"

#include "whisper.h"
#include <string>
#include <vector>

namespace mcp {

class Handler {
public:
    explicit Handler(mcp::Transport * transport,
                     const struct mcp_params & mparams,
                     const struct whisper_params & wparams,
                     const std::string & model_path);
    ~Handler();

    bool handle_message(const json & request);

private:
    // MCP protocol methods
    void handle_initialize(const json & id, const json & params);
    void handle_list_tools(const json & id);
    void handle_tool_call(const json & id, const json & params);
    void handle_notification_initialized();

    // Response helpers
    void send_result(const json & id, const json & result);
    void send_error(const json & id, int code, const std::string & message);

    bool load_model();
    std::string transcribe_file(const std::string & filepath,
                                const std::string & language = "auto",
                                bool translate = false);
    bool load_audio_file(const std::string & fname_inp, std::vector<float> & pcmf32);

    json create_transcribe_result(const json & arguments);
    json create_model_info_result();

    bool                     model_loaded;
    mcp::Transport         * transport;
    struct whisper_context * ctx;
    std::string              model_path;
    struct                   mcp_params mparams;
    struct                   whisper_params wparams;
};

} // namespace mcp
