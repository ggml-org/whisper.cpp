#pragma once

#include "mcp-transport.hpp"

namespace mcp {

class Handler;

class StdioTransport : public Transport {
public:
    StdioTransport() = default;
    ~StdioTransport() = default;

    void send_response(const json & response) override;

    void run(Handler * handler);
};

} // namespace mcp
