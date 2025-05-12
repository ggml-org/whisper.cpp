#include "whisper.h"
#include "ggml-backend.h"

#include "whisper-flat.h"

#ifdef WHISPER_BINDINGS_FLAT
void whisper_flat_backend_load_all(void) {
    ggml_backend_load_all();
}

const char * whisper_flat_get_system_info_json(void) {
    return whisper_get_system_info_json();
}

struct whisper_state * whisper_flat_get_state_from_context(struct whisper_context * ctx) {
    return whisper_get_state_from_context(ctx);
}

void whisper_flat_set_context_state(struct whisper_context * ctx, struct whisper_state * state) {
    whisper_set_context_state(ctx, state);
}

struct whisper_activity * whisper_flat_get_activity_with_state(struct whisper_state * state) {
    return whisper_get_activity_with_state(state);
}

ggml_backend_t whisper_flat_get_preferred_backend(struct whisper_state * state) {
    return whisper_get_preferred_backend(state);
}

ggml_backend_t whisper_flat_get_indexed_backend(struct whisper_state* state, size_t i) {
    return whisper_get_indexed_backend(state, i);
}

size_t whisper_flat_get_backend_count(struct whisper_state* state) {
    return whisper_get_backend_count(state);
}
#endif
