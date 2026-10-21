#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp {
namespace parler {
namespace detail {

enum class decoder_coreml_dtype { float16, float32, int32 };
enum class decoder_coreml_units { cpu_only, cpu_and_gpu, cpu_and_ane, all };

struct decoder_coreml_tensor {
    std::string name;
    std::vector<int64_t> shape;
    decoder_coreml_dtype dtype = decoder_coreml_dtype::float32;
    size_t elements() const;
};

// Copy an output shaped [..., 1, ...] into one row of a state with the same
// rank/other dimensions after optional layer selection. For [L,H,CTX,HD]
// use axis=2; per-layer and grouped states work too.
struct decoder_coreml_row_update {
    std::string output;
    std::string state;
    size_t axis = 2;
    int64_t row = 0;
    // -1: copy the entire output. Otherwise slice dimension 0 at this layer,
    // keeping rank and replacing its extent with 1, e.g. [L,H,1,HD] ->
    // [1,H,1,HD] for a per-layer self_k_LL/self_v_LL state. No staging copy.
    int64_t source_layer = -1;
    // -1: use the entire state's first axis. Otherwise select this layer,
    // keeping rank and extent 1, e.g. write [1,H,1,HD] to layer l % G of
    // [G,H,CTX,HD]. Bounds are checked before prediction. Cannot select a
    // destination layer when axis=0 (row already selects that dimension).
    int64_t destination_layer = -1;
};

struct decoder_coreml_timing {
    double predict_ms = 0;
    double row_write_ms = 0;
    // Wall time of predict(), including validation, prediction and all writes.
    // Caller input preparation and copy_output() are outside this interval.
    double total_ms = 0;
    size_t row_bytes_written = 0;
};

// A single synthesis owns a session; serialize all access. Requires macOS 15
// (or iOS 18) and a compiled .mlmodelc with multi-array inputs/outputs. Uses
// their declared/default shapes; unexpected runtime shapes fail explicitly.
// No environment variables, shared state, Python bridge, or silent fallback.
// Compile decoder_coreml.mm with ARC and link Foundation and CoreML.
class decoder_coreml_session {
public:
    static std::unique_ptr<decoder_coreml_session> open(
        const std::string & model_path, decoder_coreml_units units, std::string & error);
    ~decoder_coreml_session();
    decoder_coreml_session(const decoder_coreml_session &) = delete;
    decoder_coreml_session & operator=(const decoder_coreml_session &) = delete;

    const std::vector<decoder_coreml_tensor> & inputs() const;
    const std::vector<decoder_coreml_tensor> & outputs() const;
    const std::vector<decoder_coreml_tensor> & states() const;
    // Creator-defined string metadata; empty when the key is absent.
    std::string metadata(const std::string & key) const;

    // Inputs and their feature provider are allocated once. Float data is
    // converted to the model's fp16/fp32 dtype; integer inputs require int32.
    // Every input must be set at least once after open/reset. Unchanged cross
    // K/V and masks may then be reused across predictions.
    bool set_input(const std::string & name, const float * data, size_t count, std::string & error);
    bool set_input_i32(const std::string & name, const int32_t * data, size_t count, std::string & error);

    // Validates the complete update plan before inference. After inference,
    // writes only the requested rows into MLState, using actual buffer strides.
    // Any inference/write failure invalidates the session until reset(); callers
    // decide whether to fall back. Invalid plans fail before inference.
    bool predict(const std::vector<decoder_coreml_row_update> & updates,
                 decoder_coreml_timing * timing, std::string & error);
    bool copy_output(const std::string & name, float * data, size_t count, std::string & error) const;

    // Initialization/diagnostics, not required per decoding step. Region shape
    // and origin have state rank; data is a contiguous row-major float tensor.
    bool write_state(const std::string & name, const std::vector<int64_t> & origin,
                     const std::vector<int64_t> & shape, const float * data,
                     size_t count, std::string & error);
    bool read_state(const std::string & name, float * data, size_t count, std::string & error) const;

    // Replaces MLState with fresh zero-initialized buffers, discards outputs,
    // and requires inputs to be set again. Does not reload/recompile the model.
    bool reset(std::string & error);

private:
    struct impl;
    explicit decoder_coreml_session(std::unique_ptr<impl> p);
    std::unique_ptr<impl> p_;
};

// Stride-aware copy primitive used by the session, also testable with padded
// layouts without a model. Strides are in elements; bounds include storage
// bytes and integer overflow. Source/destination must not alias. Only float
// types may be converted; int32 copies must have int32 on both sides.
struct decoder_coreml_buffer {
    void * data = nullptr;
    size_t bytes = 0;
    decoder_coreml_dtype dtype = decoder_coreml_dtype::float32;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
};
bool decoder_coreml_copy_region(const decoder_coreml_buffer & source,
                                const decoder_coreml_buffer & destination,
                                const std::vector<int64_t> & destination_origin,
                                std::string & error);

} // namespace detail
} // namespace parler
} // namespace tts_cpp
