#include "moss/sfx_dit_cache.h"
#include "moss/fit_measure.h"

#include <stdexcept>

namespace tts_cpp::moss::detail {

SfxDitCache::SfxDitCache(SfxModel & model, int contexts, bool measure_only)
    : storage_(model.config().dit.n_layers * 2), width_(model.config().dit.n_embd),
      tokens_(model.config().text.max_tokens), contexts_(contexts) {
    if (contexts < 1) throw std::runtime_error("moss sfx: cache requires a context");
    for (int layer = 0; layer < model.config().dit.n_layers; ++layer) {
        keys_.push_back(ggml_new_tensor_2d(storage_.ctx(), GGML_TYPE_F32, int64_t(width_) * tokens_, contexts));
        values_.push_back(ggml_new_tensor_2d(storage_.ctx(), GGML_TYPE_F32, int64_t(width_) * tokens_, contexts));
        ggml_set_name(keys_.back(), ("dit.cache.key." + std::to_string(layer)).c_str());
        ggml_set_name(values_.back(), ("dit.cache.value." + std::to_string(layer)).c_str());
    }
    const auto buft = model.buffer_type();
    if (measure_only) {
        bytes_ = measure_fit_tensors(storage_.ctx(), buft);
        buffer_ = ggml_backend_buft_alloc_buffer(buft, 0);
    } else {
        buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(storage_.ctx(), buft);
        if (buffer_) bytes_ = ggml_backend_buffer_get_size(buffer_);
    }
    if (!buffer_) throw std::runtime_error("moss sfx: DiT cache allocation failed");
    if (measure_only) bind_buffer();
}

SfxDitCache::~SfxDitCache() { ggml_backend_buffer_free(buffer_); }

void SfxDitCache::bind_buffer() {
    for (auto * tensor = ggml_get_first_tensor(storage_.ctx()); tensor;
         tensor = ggml_get_next_tensor(storage_.ctx(), tensor)) {
        tensor->buffer = buffer_;
    }
}

ggml_tensor * SfxDitCache::select(SfxGraph & graph, ggml_tensor * bank, ggml_tensor * index) const {
    return ggml_reshape_2d(graph.ctx(), ggml_get_rows(graph.ctx(), bank, index), width_, tokens_);
}

ggml_tensor * SfxDitCache::key(SfxGraph & graph, int layer, ggml_tensor * index) const {
    return select(graph, keys_.at((size_t) layer), index);
}

ggml_tensor * SfxDitCache::value(SfxGraph & graph, int layer, ggml_tensor * index) const {
    return select(graph, values_.at((size_t) layer), index);
}

void SfxDitCache::store_tensor(SfxGraph & graph, ggml_tensor * bank, ggml_tensor * source, int index) const {
    auto * destination = ggml_view_2d(graph.ctx(), bank, width_, tokens_, size_t(width_) * sizeof(float),
                                     size_t(index) * bank->nb[1]);
    auto * copy = ggml_cpy(graph.ctx(), source, destination);
    ggml_build_forward_expand(graph.graph(), copy);
}

void SfxDitCache::store(SfxGraph & graph, int layer, int index, ggml_tensor * key, ggml_tensor * value) const {
    store_tensor(graph, keys_.at((size_t) layer), key, index);
    store_tensor(graph, values_.at((size_t) layer), value, index);
}

int SfxDitCache::contexts() const { return contexts_; }
uint64_t SfxDitCache::bytes() const { return bytes_; }
uint64_t SfxDitCache::host_bytes() const {
    return ggml_get_mem_size(storage_.ctx()) + (keys_.capacity() + values_.capacity()) * sizeof(ggml_tensor *);
}

} // namespace tts_cpp::moss::detail
