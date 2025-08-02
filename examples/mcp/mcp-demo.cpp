#include "stdio-client.hpp"

#include <string>
#include <iostream>

void pretty_print_json(const json & j) {
    std::cout << j.dump(2) << std::endl;
}

int main(int argc, char ** argv) {
    std::string server_command = "build/bin/whisper-mcp-server";

    if (argc > 1) {
        server_command = argv[1];
    }

    printf("Starting MCP Demo\n");
    printf("Server command: %s\n", server_command.c_str());

    try {
        mcp::StdioClient client;

        // Start the server
        printf("Starting server...\n");
        if (!client.start_server(server_command)) {
            fprintf(stderr, "Failed to start server\n");
            return 1;
        }

        if (!client.wait_for_server_ready(2000)) {
            fprintf(stderr, "Server failed to start within timeout\n");
            return 1;
        }

        client.read_server_logs();

        // Initialize
        printf("Initializing...\n");
        json init_response = client.initialize("mcp-demo-client", "1.0.0");
        printf("Initialize response:\n");
        pretty_print_json(init_response);

        if (init_response.contains("error")) {
            fprintf(stderr, "Initialization failed!\n");
            return 1;
        }

        // Send initialized notification
        printf("Sending initialized notification...\n");
        client.send_initialized();
        client.read_server_logs();

        // List tools
        printf("Listing tools...\n");
        json tools_response = client.list_tools();
        printf("Tools list response:\n");
        pretty_print_json(tools_response);

        // Call transcribe tool
        printf("Calling transcribe tool...\n");
        json transcribe_args = {
            {"file", "samples/jfk.wav"}
        };

        json transcribe_response = client.call_tool("transcribe", transcribe_args);
        printf("Transcribe response:\n");
        pretty_print_json(transcribe_response);

        // Call model info tool
        printf("Calling model info tool...\n");
        json model_info_response = client.call_tool("model_info", json::object());
        printf("Model info response:\n");
        pretty_print_json(model_info_response);

        // Final logs
        printf("Final server logs:\n");
        client.read_server_logs();
    } catch (const std::exception & e) {
        fprintf(stderr, "Exception: %s\n", e.what());
        return 1;
    }

    return 0;
}
