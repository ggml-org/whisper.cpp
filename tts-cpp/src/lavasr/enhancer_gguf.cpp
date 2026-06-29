#include "enhancer_gguf.h"

#include "../gguf_stream.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tts_cpp::lavasr {

bool load_enhancer_gguf(const std::string & path, EnhancerWeights & out,
                        std::string * err) {
    auto fail = [&](const std::string & m) {
        if (err) {
            *err = "load_enhancer_gguf: " + m + " (" + path + ")";
        }
        return false;
    };

    // no_alloc=true + streamed host reads: the enhancer GGUF is ~56 MB, small
    // enough to stage, but the streamed reader keeps peak memory low and
    // mirrors the other tts-cpp loaders (voice_encoder / s3tokenizer).
    ggml_context *    tmp_ctx = nullptr;
    gguf_init_params  gp = {/*.no_alloc=*/true, /*.ctx=*/&tmp_ctx};
    gguf_context *    g  = gguf_init_from_file(path.c_str(), gp);
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
        if (kid < 0 || std::string(gguf_get_val_str(g, kid)) != "lavasr-enhancer") {
            return cleanup(fail("not a lavasr-enhancer GGUF"));
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

    out.dim       = u32("lavasr.enhancer.dim", 512);
    out.ffn_dim   = u32("lavasr.enhancer.ffn_dim", 1536);
    out.n_blocks  = u32("lavasr.enhancer.n_blocks", 8);
    out.n_mels    = u32("lavasr.enhancer.n_mels", 80);
    out.kernel    = u32("lavasr.enhancer.kernel", 7);
    out.n_fft     = u32("lavasr.enhancer.n_fft", 2048);
    out.hop       = u32("lavasr.enhancer.hop", 512);
    out.win       = u32("lavasr.enhancer.win", 2048);
    out.spec_bins = u32("lavasr.enhancer.spec_bins", 1025);
    out.clip_max  = f32("lavasr.enhancer.clip_max", 1000.0f);
    out.ln_eps    = f32("lavasr.enhancer.layernorm_eps", 1e-6f);
    out.work_sample_rate    = u32("lavasr.enhancer.work_sample_rate", 48000);
    out.mel_ref_sample_rate = u32("lavasr.enhancer.mel_ref_sample_rate", 44100);

    // The radix-2 FFT requires power-of-two n_fft/win. Fail loudly here rather
    // than let StftProcessor silently corrupt output on a future odd-n_fft
    // GGUF (the FFT also self-guards, this is the friendly early error).
    auto is_pow2 = [](int v) { return v > 0 && (v & (v - 1)) == 0; };
    if (!is_pow2(out.n_fft) || !is_pow2(out.win)) {
        return cleanup(fail("n_fft and win must be powers of two (got n_fft=" +
                            std::to_string(out.n_fft) + ", win=" +
                            std::to_string(out.win) + ")"));
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
        EnhTensor et;
        const int64_t n = ggml_nelements(t);
        et.data.resize(static_cast<size_t>(n));
        if (t->type == GGML_TYPE_F32) {
            if (!reader.to_host(name, et.data.data(), ggml_nbytes(t))) {
                return cleanup(fail(std::string("failed to read tensor '") + name + "'"));
            }
        } else if (t->type == GGML_TYPE_F16) {
            // f16 GGUF (--ftype f16): stream the raw halves and dequant to the
            // f32 the scalar core operates on. The scalar enhancer is the only
            // consumer, so we never keep the f16 copy resident.
            std::vector<ggml_fp16_t> tmp(static_cast<size_t>(n));
            if (!reader.to_host(name, tmp.data(), ggml_nbytes(t))) {
                return cleanup(fail(std::string("failed to read tensor '") + name + "'"));
            }
            for (int64_t i = 0; i < n; i++) {
                et.data[static_cast<size_t>(i)] = ggml_fp16_to_fp32(tmp[static_cast<size_t>(i)]);
            }
        } else {
            return cleanup(fail(std::string("tensor '") + name +
                                "' has unsupported dtype (expected f32/f16)"));
        }
        for (int d = GGML_MAX_DIMS - 1; d >= 0; d--) {
            if (t->ne[d] > 1 || !et.shape.empty()) {
                et.shape.push_back(static_cast<int>(t->ne[d]));
            }
        }
        out.t.emplace(name, std::move(et));
    }

    // Validate key tensor ranks/dims against the metadata so a malformed GGUF
    // fails loudly here instead of indexing past the end in the forward core.
    // Shapes are in numpy/C order (matching the converter): conv weights are
    // [out, in/groups, K]; linear weights are [out, in].
    auto check_shape = [&](const std::string & name,
                           const std::vector<int> & want) -> bool {
        if (!out.has(name)) {
            return fail("GGUF missing tensor '" + name + "'");
        }
        const std::vector<int> & got = out.get(name).shape;
        if (got != want) {
            std::string g;
            for (int d : got) {
                g += (g.empty() ? "" : ",") + std::to_string(d);
            }
            return fail("tensor '" + name + "' has unexpected shape [" + g + "]");
        }
        return true;
    };

    const int C = out.dim, M = out.n_mels, K = out.kernel, F = out.ffn_dim;
    if (!check_shape("enhancer.embed.weight", {C, M, K}) ||
        !check_shape("enhancer.block.0.dwconv.weight", {C, 1, K}) ||
        !check_shape("enhancer.block.0.pwconv1.weight", {F, C}) ||
        !check_shape("enhancer.block.0.pwconv2.weight", {C, F}) ||
        !check_shape("spec_head.out.weight", {2 * out.spec_bins, C})) {
        return cleanup(false); // check_shape() already set *err via fail()
    }

    return cleanup(true);
}

} // namespace tts_cpp::lavasr
