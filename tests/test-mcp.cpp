#include "stdio-client.hpp"

#include "whisper.h"
#include "common-whisper.h"

#include <cassert>

template<typename T>
void assert_json_equals(const json & j, const std::string & key, const T & expected) {
    assert(j.contains(key));
    assert(j.at(key) == expected);
}

void assert_initialized(const json & response) {
    assert_json_equals(response, "id", 1);
    assert_json_equals(response, "jsonrpc", "2.0");

    json result = response.at("result");

    json cap = result.at("capabilities");
    assert(cap.at("tools").is_object());

    assert_json_equals(result, "protocolVersion", "2024-11-05");

    json server_info = result.at("serverInfo");
    assert_json_equals(server_info, "name", "whisper-mcp-server");
    assert_json_equals(server_info, "version", "1.0.0");
}

int main() {
    std::string server_bin = "../../build/bin/whisper-mcp-server";
    std::vector<std::string> args = {
        "--model", "../../models/ggml-base.en.bin"
    };
    mcp::StdioClient client;

    // Start server
    assert(client.start_server(server_bin, args));
    assert(client.wait_for_server_ready(2000));
    assert(client.is_server_running());


    // Send initialize request
    assert_initialized(client.initialize("mcp-test-client", "1.0.0"));
    // Send initialized notification
    client.send_initialized();

    // Read logs for debugging
    client.read_server_logs();

    return 0;
}
