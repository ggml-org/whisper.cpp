#pragma once

#include "moss/sfx_model.h"

struct ggml_backend_buffer;

namespace tts_cpp::moss::detail {

class SfxDitCache {
public:
    SfxDitCache(SfxModel & model, int contexts, bool measure_only);
    ~SfxDitCache();
    SfxDitCache(const SfxDitCache &) = delete;
    SfxDitCache & operator=(const SfxDitCache &) = delete;

    ggml_tensor * key(SfxGraph & graph, int layer, ggml_tensor * index) const;
    ggml_tensor * value(SfxGraph & graph, int layer, ggml_tensor * index) const;
    void store(SfxGraph & graph, int layer, int index, ggml_tensor * key, ggml_tensor * value) const;
    int contexts() const;
    uint64_t bytes() const;
    uint64_t host_bytes() const;

private:
    ggml_tensor * select(SfxGraph & graph, ggml_tensor * bank, ggml_tensor * index) const;
    void store_tensor(SfxGraph & graph, ggml_tensor * bank, ggml_tensor * source, int index) const;
    void bind_buffer();

    SfxGraph storage_;
    int width_;
    int tokens_;
    int contexts_;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    ggml_backend_buffer * buffer_ = nullptr;
    uint64_t bytes_ = 0;
};

} // namespace tts_cpp::moss::detail
