#pragma once

#include "json.hpp"

using json = nlohmann::ordered_json;

namespace mcp {

class Transport {
public:
    virtual ~Transport() = default;
    virtual void send_response(const json & response) = 0;
};

} // namespace mcp
