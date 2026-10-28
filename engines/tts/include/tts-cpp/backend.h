#pragma once

namespace tts_cpp {

enum class BackendDevice : int {
    CPU = 0,
    GPU = 1,
    CoreML = 2,
};

}
