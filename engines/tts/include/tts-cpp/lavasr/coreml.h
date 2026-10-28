#pragma once

#include <string>

namespace tts_cpp::lavasr {

struct CoreMLOptions {
    std::string model_path;
    std::string compute_units;
};

}
