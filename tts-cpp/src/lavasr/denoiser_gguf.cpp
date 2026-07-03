#include "denoiser_gguf.h"

#include "../gguf_stream.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tts_cpp::lavasr {

bool load_denoiser_gguf(const std::string & path, DenoiserWeights & out,
                        std::string * err) {
    auto fail = [&](const std::string & m) {
        if (err) {
            *err = "load_denoiser_gguf: " + m + " (" + path + ")";
        }
        return false;
    };

    // no_alloc=true + streamed host reads (mirrors enhancer_gguf.cpp); the
    // denoiser GGUF is tiny (~0.7 MB) but we keep the same low-peak-memory path.
    ggml_context *   tmp_ctx = nullptr;
    gguf_init_params gp = {/*.no_alloc=*/true, /*.ctx=*/&tmp_ctx};
    gguf_context *   g  = gguf_init_from_file(path.c_str(), gp);
    if (!g) {
        return fail("failed to open GGUF");
    }
    auto cleanup = [&](bool ok) {
        gguf_free(g);
        if (tmp_ctx) {
            ggml_free(tmp_ctx);
        }
        return ok;
    };

    {
        const int64_t kid = gguf_find_key(g, "general.architecture");
        if (kid < 0 || std::string(gguf_get_val_str(g, kid)) != "lavasr-denoiser") {
            return cleanup(fail("not a lavasr-denoiser GGUF"));
        }
    }

    tts_cpp::detail::gguf_stream_reader reader(g, path);
    if (!reader.ok()) {
        return cleanup(fail("stream reader init failed"));
    }

    auto u32 = [&](const char * k, int fallback) -> int {
        const int64_t id = gguf_find_key(g, k);
        return id < 0 ? fallback : static_cast<int>(gguf_get_val_u32(g, id));
    };
    auto f32 = [&](const char * k, float fallback) -> float {
        const int64_t id = gguf_find_key(g, k);
        return id < 0 ? fallback : gguf_get_val_f32(g, id);
    };

    out.n_fft            = u32("lavasr.denoiser.n_fft", 512);
    out.hop              = u32("lavasr.denoiser.hop", 256);
    out.win              = u32("lavasr.denoiser.win", 512);
    out.spec_bins        = u32("lavasr.denoiser.spec_bins", 257);
    out.work_sample_rate = u32("lavasr.denoiser.work_sample_rate", 16000);
    out.erb_low          = u32("lavasr.denoiser.erb_low", 65);
    out.erb_high         = u32("lavasr.denoiser.erb_high", 64);
    out.freq_comp_ratio  = u32("lavasr.denoiser.freq_comp_ratio", 4);
    out.chunk_frames     = u32("lavasr.denoiser.chunk_frames", 63);
    out.chunk_hop        = u32("lavasr.denoiser.chunk_hop", 21);
    out.bn_eps           = f32("lavasr.denoiser.batchnorm_eps", 1e-5f);
    out.ln_eps           = f32("lavasr.denoiser.layernorm_eps", 1e-8f);

    // The radix-2 FFT requires power-of-two n_fft/win (StftProcessor also
    // self-guards; this is the friendly early error).
    auto is_pow2 = [](int v) { return v > 0 && (v & (v - 1)) == 0; };
    if (!is_pow2(out.n_fft) || !is_pow2(out.win)) {
        return cleanup(fail("n_fft and win must be powers of two (got n_fft=" +
                            std::to_string(out.n_fft) + ", win=" +
                            std::to_string(out.win) + ")"));
    }
    if (out.spec_bins != out.n_fft / 2 + 1) {
        return cleanup(fail("spec_bins != n_fft/2+1"));
    }

    const int n_tensors = static_cast<int>(gguf_get_n_tensors(g));
    if (n_tensors <= 0) {
        return cleanup(fail("GGUF has no tensors"));
    }

    for (int i = 0; i < n_tensors; i++) {
        const char *  name = gguf_get_tensor_name(g, i);
        ggml_tensor * t    = ggml_get_tensor(tmp_ctx, name);
        if (!t) {
            return cleanup(fail(std::string("missing tensor handle '") + name + "'"));
        }
        DnTensor      et;
        const int64_t n = ggml_nelements(t);
        et.data.resize(static_cast<size_t>(n));
        if (t->type == GGML_TYPE_F32) {
            if (!reader.to_host(name, et.data.data(), ggml_nbytes(t))) {
                return cleanup(fail(std::string("failed to read tensor '") + name + "'"));
            }
        } else if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(static_cast<size_t>(n));
            if (!reader.to_host(name, tmp.data(), ggml_nbytes(t))) {
                return cleanup(fail(std::string("failed to read tensor '") + name + "'"));
            }
            for (int64_t k = 0; k < n; k++) {
                et.data[static_cast<size_t>(k)] = ggml_fp16_to_fp32(tmp[static_cast<size_t>(k)]);
            }
        } else {
            return cleanup(fail(std::string("tensor '") + name +
                                "' has unsupported dtype (expected f32/f16)"));
        }
        // Rebuild the numpy/C tensor shape from ggml's ne[] (reversed and
        // right-padded with unit dims).  Use ggml_n_dims() so the rank matches
        // the ONNX rank and interior unit dims are preserved (e.g. depthwise
        // conv [12,1,3,3]); only ggml's trailing unit padding is dropped.
        // Caveat: ggml cannot represent a *leading* numpy unit dim ([1,N] and
        // [N] share the same ne[]), so a mismatched export is caught by the
        // converter's fail-fast shape asserts + check_shape() below, not here.
        const int nd = ggml_n_dims(t) > 0 ? ggml_n_dims(t) : 1;
        et.shape.resize(static_cast<size_t>(nd));
        for (int d = 0; d < nd; d++) {
            et.shape[d] = static_cast<int>(t->ne[nd - 1 - d]);
        }
        out.t.emplace(name, std::move(et));
    }

    // Validate a few load-bearing tensors so a malformed GGUF fails loudly here
    // instead of indexing past the end in the forward core.  Shapes are numpy/C
    // order (matching the converter): conv [out,in/g,kt,kf]; linear/GRU [out,in].
    auto check_shape = [&](const std::string & name,
                           const std::vector<int> & want) -> bool {
        if (!out.has(name)) {
            return fail("GGUF missing tensor '" + name + "'");
        }
        const std::vector<int> & got = out.get(name).shape;
        if (got != want) {
            std::string s;
            for (int d : got) {
                s += (s.empty() ? "" : ",") + std::to_string(d);
            }
            return fail("tensor '" + name + "' has unexpected shape [" + s + "]");
        }
        return true;
    };

    if (out.erb_low + out.erb_high != 129) {
        return cleanup(fail("erb_low + erb_high must be 129"));
    }
    if (!check_shape("erb.erb_fc.weight", {out.erb_high, out.spec_bins - out.erb_low}) ||
        !check_shape("erb.ierb_fc.weight", {out.spec_bins - out.erb_low, out.erb_high}) ||
        !check_shape("encoder.en_convs.0.ops.1.weight", {12, 1, 3, 3}) ||
        !check_shape("decoder.de_convs.4.ops.1.weight", {12, 1, 3, 3}) ||
        !check_shape("dpgrnn.0.intra_fc.weight", {16, 16})) {
        return cleanup(false); // check_shape() already set *err via fail()
    }

    return cleanup(true);
}

} // namespace tts_cpp::lavasr
