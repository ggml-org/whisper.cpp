#include "whisper.h"
#include "whisper-arch.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <atomic>
#include <algorithm>
#include <cassert>
#define _USE_MATH_DEFINES
#include <cmath>
#include <climits>
#include <codecvt>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "whisper-flat.h"

void whisper_flat_backend_load_all(void) {
    ggml_backend_load_all();
}

const char * whisper_flat_get_system_info_json(void) {
    return whisper_get_system_info_json();
}

struct whisper_state * whisper_flat_get_state_from_context(struct whisper_context * ctx) {
    return whisper_get_state_from_context(ctx);
}

struct whisper_timings * whisper_flat_get_timings_with_state(struct whisper_state * state) {
    return whisper_get_timings_with_state(state);
}