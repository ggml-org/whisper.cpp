#pragma once

// Streaming GGUF tensor-data access.
//
// gguf_init_from_file(no_alloc=false) materialises the ENTIRE tensor-data
// section of the file in host memory (one I8 blob tensor inside the
// returned ggml_context) before a single byte reaches its destination.
// For the chatterbox GGUFs that means a transient +0.5..1 GB host
// allocation per model load that coexists with the freshly-allocated
// backend weight buffer — a 2x peak that gets chatterbox jetsam-killed
// on iOS.  Loaders that only want a handful of small tensors (voice
// encoder, CAMPPlus, S3 tokenizer, mel filterbanks) paid the same full-
// file staging cost just to memcpy a few MB out of it.
//
// gguf_stream_reader replaces the staging blob with direct file reads:
// open the GGUF with no_alloc=true (metadata-only tensors, no data
// blob), allocate the destination (backend buffer or host vector), then
// stream each tensor's payload from the file in bounded chunks.  Peak
// host overhead drops from sizeof(data section) to CHUNK (8 MiB).
//
// Usage (backend weights):
//   ggml_context * meta = nullptr;
//   gguf_init_params p = { /*no_alloc=*/ true, &meta };
//   gguf_context * g = gguf_init_from_file(path, p);
//   ... dup metadata tensors into ctx_w, ggml_backend_alloc_ctx_tensors ...
//   gguf_stream_reader rd(g, path);
//   for (t : ctx_w tensors) rd.to_backend(ggml_get_name(t), t);
//
// Usage (host vectors):
//   gguf_stream_reader rd(g, path);
//   ggml_tensor * t = ggml_get_tensor(meta, name);   // metadata only
//   out.resize(ggml_nelements(t));
//   rd.to_host(name, out.data(), ggml_nbytes(t));

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace tts_cpp {
namespace detail {

class gguf_stream_reader {
public:
    // `g` must outlive the reader; `path` must be the same file `g` was
    // parsed from (offsets are resolved against `g`, bytes come from the
    // freshly-opened handle, so a swapped file shows up as a short read
    // or a size mismatch rather than silent corruption).
    gguf_stream_reader(const gguf_context * g, const std::string & path)
        : g_(g), f_(std::fopen(path.c_str(), "rb")) {
        if (!f_) {
            std::fprintf(stderr, "gguf_stream_reader: failed to reopen '%s' for tensor data\n",
                         path.c_str());
        }
    }
    ~gguf_stream_reader() { if (f_) std::fclose(f_); }
    gguf_stream_reader(const gguf_stream_reader &) = delete;
    gguf_stream_reader & operator=(const gguf_stream_reader &) = delete;

    bool ok() const { return f_ != nullptr; }

    // Stream tensor `name`'s payload into an already-allocated backend
    // tensor, CHUNK bytes at a time.  dst's nbytes must match the GGUF
    // entry exactly (catches metadata/destination drift).
    bool to_backend(const char * name, ggml_tensor * dst) {
        size_t nbytes = 0;
        if (!locate(name, ggml_nbytes(dst), nbytes)) return false;

        // Quantized tensors must be uploaded in a SINGLE whole-tensor call.
        // The OpenCL backend's Q4_0/Q8_0 set_tensor path rebuilds a struct-of-
        // arrays (scale/quant) layout from the ENTIRE tensor and ignores the
        // (offset, size) window — it reads ggml_nbytes(dst) from `data`
        // regardless of `size` (unlike the plain path, which honours it). A
        // chunked partial upload therefore makes it read past the end of the
        // CHUNK-sized scratch; on Adreno that copy lands straight in GPU shared
        // memory and SIGSEGVs in the driver at model load (QVAC-19557). Upload
        // quantized weights whole (peak overhead = largest quantized tensor,
        // still far below the old full-file staging) and keep chunk streaming
        // for f32/f16, where every backend honours (offset, size).
        if (ggml_is_quantized(dst->type)) {
            scratch_.resize(nbytes);
            if (nbytes != 0 && std::fread(scratch_.data(), 1, nbytes, f_) != nbytes) {
                std::fprintf(stderr, "gguf_stream_reader: short read on tensor '%s' "
                             "(%zu bytes)\n", name, nbytes);
                return false;
            }
            ggml_backend_tensor_set(dst, scratch_.data(), 0, nbytes);
            return true;
        }

        scratch_.resize(std::min(nbytes, (size_t) CHUNK));
        size_t done = 0;
        while (done < nbytes) {
            const size_t n = std::min((size_t) CHUNK, nbytes - done);
            if (std::fread(scratch_.data(), 1, n, f_) != n) {
                std::fprintf(stderr, "gguf_stream_reader: short read on tensor '%s' "
                             "(%zu of %zu bytes)\n", name, done, nbytes);
                return false;
            }
            ggml_backend_tensor_set(dst, scratch_.data(), done, n);
            done += n;
        }
        return true;
    }

    // Read tensor `name`'s payload into a host buffer of exactly `nbytes`.
    bool to_host(const char * name, void * dst, size_t nbytes) {
        size_t sz = 0;
        if (!locate(name, nbytes, sz)) return false;
        if (sz != 0 && std::fread(dst, 1, sz, f_) != sz) {
            std::fprintf(stderr, "gguf_stream_reader: short read on tensor '%s' (%zu bytes)\n",
                         name, sz);
            return false;
        }
        return true;
    }

private:
    // 8 MiB: large enough that fread/ggml_backend_tensor_set call overhead
    // is negligible against disk/flash bandwidth, small enough to be noise
    // next to the weight buffers themselves.
    static constexpr size_t CHUNK = 8u * 1024u * 1024u;

    // Find `name` in the GGUF, validate its payload size against what the
    // caller allocated, and seek the file handle to its payload.
    bool locate(const char * name, size_t expect_bytes, size_t & nbytes_out) {
        if (!f_) return false;
        const int64_t id = gguf_find_tensor(g_, name);
        if (id < 0) {
            std::fprintf(stderr, "gguf_stream_reader: tensor '%s' not found in GGUF\n", name);
            return false;
        }
        const size_t nbytes = gguf_get_tensor_size(g_, id);
        if (nbytes != expect_bytes) {
            std::fprintf(stderr, "gguf_stream_reader: tensor '%s' size mismatch "
                         "(file has %zu bytes, destination wants %zu)\n",
                         name, nbytes, expect_bytes);
            return false;
        }
        const uint64_t off = (uint64_t) gguf_get_data_offset(g_) + (uint64_t) gguf_get_tensor_offset(g_, id);
#if defined(_WIN32)
        if (_fseeki64(f_, (long long) off, SEEK_SET) != 0) {
#else
        if (fseeko(f_, (off_t) off, SEEK_SET) != 0) {
#endif
            std::fprintf(stderr, "gguf_stream_reader: seek to tensor '%s' (offset %llu) failed\n",
                         name, (unsigned long long) off);
            return false;
        }
        nbytes_out = nbytes;
        return true;
    }

    const gguf_context * g_;
    FILE * f_;
    std::vector<uint8_t> scratch_;
};

} // namespace detail
} // namespace tts_cpp
