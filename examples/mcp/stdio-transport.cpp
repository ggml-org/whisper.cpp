#include "stdio-transport.hpp"
#include "mcp-handler.hpp"

#include <iostream>
#include <string>
#include <cstdio>

namespace mcp {

void StdioTransport::send_response(const json & response) {
    std::cout << response.dump() << std::endl;
    std::cout.flush();
}

void StdioTransport::run(Handler * handler) {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }

        fprintf(stderr, "Received: %s\n", line.c_str());

        try {
            json request = json::parse(line);
            handler->handle_message(request);
        } catch (const json::parse_error & e) {
            fprintf(stderr, "JSON parse error: %s\n", e.what());
        } catch (const std::exception & e) {
            fprintf(stderr, "Error processing request: %s\n", e.what());
        }
    }
}

} // namespace mcp
