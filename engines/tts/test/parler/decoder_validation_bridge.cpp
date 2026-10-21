// Test-only entry points for validate-decoder-coreml.py's ctypes adapters.
// Core ML is STRICT: failures fail this sample, never silently benchmark ggml.
#include "decoder_validation_bridge.h"
#include "parler/decoder_runtime.h"
#include "parler/internal.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tts_cpp::parler::detail;

namespace {
struct bridge {
    parler_model model;
    std::unique_ptr<parler_decoder_runtime> runtime;
    ggml_gallocr_t allocator = nullptr;
    std::vector<int32_t> prompt;
    int threads = 1;
    int past = 0;
    int max_frames = 0;
    bool prepared = false;
    bool started = false;
    bool coreml = false;

    ~bridge() {
        runtime.reset();
        if (allocator) ggml_gallocr_free(allocator);
        parler_free_model(model);
    }
};

void message(char * output, size_t size, const char * text) {
    if (output && size) std::snprintf(output, size, "%s", text);
}

template<class F> int checked(char * error, size_t size, F && fn) {
    message(error, size, "");
    try {
        fn();
        return 1;
    } catch (const std::exception & e) {
        message(error, size, e.what());
    } catch (...) {
        message(error, size, "unknown native exception");
    }
    return 0;
}

bridge & get(void * handle) {
    if (!handle) throw std::invalid_argument("null decoder validation handle");
    return *static_cast<bridge *>(handle);
}

void validate_ids(const int32_t * ids, size_t count, int vocab, const char * label) {
    if (!ids || !count || count > 32768 || vocab <= 0) {
        throw std::invalid_argument(std::string("invalid ") + label + " pointer/count/vocab");
    }
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] < 0 || ids[i] >= vocab) {
            throw std::invalid_argument(std::string(label) + " token outside vocabulary");
        }
    }
}

void validate_output(bridge & b, float * output, size_t count) {
    const auto & hp = b.model.hparams;
    if (!output || count != (size_t) hp.n_codebooks * hp.dec_vocab) {
        throw std::invalid_argument("caller logits buffer has wrong size");
    }
}

void copy_logits(bridge & b, const parler_step_logits & logits, float * output, size_t count) {
    if (!logits.view) throw std::runtime_error("native decoder returned no logits");
    // read_logits()/Core ML copy_output() already materialize device outputs.
    // This final caller-owned copy is still inside Python's per-step timer.
    std::memcpy(output, logits.view, count * sizeof(float));
    if (b.coreml && !b.runtime->active()) {
        throw std::runtime_error("Core ML sidecar retired during strict validation");
    }
}
} // namespace

extern "C" {
int parler_validation_abi_version(void) { return 2; }

void * parler_validation_create(const char * gguf, int n_gpu_layers, int use_coreml,
                                int threads, const char * backend, char * error, size_t error_size) {
    bridge * result = nullptr;
    checked(error, error_size, [&] {
        if (!gguf || !*gguf || threads < 1 || threads > 256 || n_gpu_layers < 0 ||
            (use_coreml != 0 && use_coreml != 1)) throw std::invalid_argument("invalid create arguments");
        auto b = std::make_unique<bridge>();
        std::string why;
        if (!parler_load_gguf(gguf, b->model, n_gpu_layers, &why, backend ? backend : "")) {
            throw std::runtime_error("GGUF load failed: " + why);
        }
        b->threads = threads;
        b->coreml = use_coreml != 0;
        b->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(b->model.backend));
        if (!b->allocator) throw std::runtime_error("decoder allocator creation failed");
        b->runtime = std::make_unique<parler_decoder_runtime>(b->model);
        if (b->coreml) {
            auto sidecar = parler_open_decoder_sidecar(b->model, gguf, why);
            if (!sidecar) throw std::runtime_error("Core ML sidecar open failed: " + why);
            b->runtime->attach(std::move(sidecar), true);
        }
        result = b.release();
    });
    return result;
}

void parler_validation_destroy(void * handle) { delete static_cast<bridge *>(handle); }

int parler_validation_dimensions(void * handle, int * codebooks, int * vocab, int * on_gpu,
                                 char * error, size_t error_size) {
    return checked(error, error_size, [&] {
        auto & b = get(handle);
        if (!codebooks || !vocab || !on_gpu) throw std::invalid_argument("null dimensions output");
        *codebooks = b.model.hparams.n_codebooks;
        *vocab = b.model.hparams.dec_vocab;
        *on_gpu = b.model.on_gpu ? 1 : 0;
    });
}

int parler_validation_prepare(void * handle, const int32_t * description, size_t description_count,
                              const int32_t * prompt, size_t prompt_count, int max_frames,
                              char * error, size_t error_size) {
    return checked(error, error_size, [&] {
        auto & b = get(handle);
        b.prepared = b.started = false;
        validate_ids(description, description_count, b.model.hparams.t5_vocab, "description");
        validate_ids(prompt, prompt_count, (int) b.model.embed_prompts->ne[1], "prompt");
        const auto & hp = b.model.hparams;
        if (max_frames < 1 || max_frames == std::numeric_limits<int>::max() || prompt_count + (size_t) max_frames > (size_t) hp.n_ctx ||
            prompt_count + (size_t) max_frames > (size_t) hp.max_position) {
            throw std::invalid_argument("prompt plus actual trace bound exceeds native context");
        }
        b.prompt.assign(prompt, prompt + prompt_count);
        b.max_frames = max_frames;
        std::vector<int32_t> description_ids(description, description + description_count);
        if (!parler_encode_description(b.model, description_ids, b.threads, nullptr)) {
            throw std::runtime_error("native description/cross-KV preparation failed");
        }
        b.prepared = true;
    });
}

int parler_validation_prefill(void * handle, const int32_t * start, size_t start_count,
                              float * output, size_t output_count, char * error, size_t error_size) {
    return checked(error, error_size, [&] {
        auto & b = get(handle);
        b.started = false;
        if (!b.prepared) throw std::runtime_error("prepare must precede prefill");
        if (start_count != (size_t) b.model.hparams.n_codebooks) throw std::invalid_argument("wrong start frame size");
        validate_ids(start, start_count, b.model.hparams.dec_vocab, "start frame");
        validate_output(b, output, output_count);
        parler_step_logits logits;
        std::vector<int32_t> frame(start, start + start_count);
        if (!b.runtime->prefill(b.prompt, frame, b.allocator, b.threads, logits, b.past, b.max_frames + 1)) {
            throw std::runtime_error("native decoder prefill/Core ML seed failed");
        }
        copy_logits(b, logits, output, output_count);
        b.started = true;
    });
}

int parler_validation_step(void * handle, const int32_t * frame, size_t frame_count,
                           float * output, size_t output_count, char * error, size_t error_size) {
    return checked(error, error_size, [&] {
        auto & b = get(handle);
        if (!b.started) throw std::runtime_error("prefill must precede step");
        if (frame_count != (size_t) b.model.hparams.n_codebooks) throw std::invalid_argument("wrong delayed frame size");
        validate_ids(frame, frame_count, b.model.hparams.dec_vocab, "delayed frame");
        validate_output(b, output, output_count);
        parler_step_logits logits;
        std::vector<int32_t> ids(frame, frame + frame_count);
        if (!b.runtime->step(ids, b.past, b.allocator, b.threads, logits)) {
            b.started = false;
            throw std::runtime_error("native decoder/Core ML step failed; sample invalidated");
        }
        copy_logits(b, logits, output, output_count);
        ++b.past;
    });
}

int parler_validation_synchronize(void * handle, char * error, size_t error_size) {
    return checked(error, error_size, [&] {
        auto & b = get(handle);
        if (b.model.sched_fb.sched) ggml_backend_sched_synchronize(b.model.sched_fb.sched);
        ggml_backend_synchronize(b.model.backend);
        // Core ML predict()/partial state writes/copy_output() are synchronous
        // in decoder_coreml_session; no work is queued past their return.
    });
}

int parler_validation_backend(void * handle, char * output, size_t output_size,
                              char * error, size_t error_size) {
    return checked(error, error_size, [&] {
        auto & b = get(handle);
        if (!output || !output_size) throw std::invalid_argument("null backend output");
        const std::string label = std::string(ggml_backend_name(b.model.backend)) + "/" + b.runtime->backend();
        if (output_size <= label.size()) throw std::invalid_argument("backend output buffer too small");
        message(output, output_size, label.c_str());
    });
}
} // extern C
