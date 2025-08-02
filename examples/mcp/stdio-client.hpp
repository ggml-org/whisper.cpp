#pragma once

#include "json.hpp"
#include <string>
#include <vector>

using json = nlohmann::json;

namespace mcp {

class StdioClient {
private:
    pid_t server_pid;
    int   stdin_pipe[2];
    int   stdout_pipe[2];
    int   stderr_pipe[2];
    FILE* server_stdin;
    FILE* server_stdout;
    FILE* server_stderr;
    int   request_id_counter;
    bool  server_running;

    void cleanup();

public:
    StdioClient();
    ~StdioClient();

    bool start_server(const std::string & server_command, const std::vector<std::string> & args = {});
    void stop_server();
    bool is_server_running() const {
        return server_running;
    }

    // Core MCP communication
    json send_request(const json& request);
    void read_server_logs();

    // MCP protocol methods
    json initialize(const std::string& client_name = "mcp-test-client",
                    const std::string& client_version = "1.0.0");
    void send_initialized();
    json list_tools();
    json call_tool(const std::string& tool_name, const json& arguments);

    // Utilities
    int next_request_id();
    bool wait_for_server_ready(int timeout_ms = 1000);
    std::string get_last_server_logs();
};

} // namespace mcp
