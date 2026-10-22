// Metadata-only fixture: packed profiling must fail before any graph executes.
#include "parakeet_ctc.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    auto * ctx = ggml_init({ggml_tensor_overhead() * 32, nullptr, true});
    auto * file = gguf_init_empty();
    gguf_set_val_str(file, "general.architecture", "parakeet-affine-v1");
    gguf_set_val_u32(file, "parakeet.encoder.n_layers", 0);
    gguf_set_val_u32(file, "parakeet.encoder.d_model", 32);
    gguf_set_val_u32(file, "parakeet.encoder.n_heads", 1);
    gguf_set_val_u32(file, "parakeet.affine.version", 1);
    const char * name = "encoder.blk.0.ff1.linear1.weight";
    gguf_set_arr_str(file, "parakeet.affine.names", &name, 1);
    auto add = [&](const std::string & n, ggml_tensor * t) {
        ggml_set_name(t, n.c_str());
        gguf_add_tensor(file, t);
    };
    add(name, ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 2));
    add(std::string(name) + ".affine.minimum", ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2));
    add(std::string(name) + ".affine.step", ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2));
    for (const char * n : {"preproc.mel_filterbank", "preproc.window",
            "encoder.subsampling.conv0.weight", "encoder.subsampling.conv1_dw.weight",
            "encoder.subsampling.conv1_pw.weight", "encoder.subsampling.conv2_dw.weight",
            "encoder.subsampling.conv2_pw.weight", "encoder.subsampling.out.weight",
            "ctc.decoder.weight", "ctc.decoder.bias"}) {
        add(n, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32));
    }
    const bool written = gguf_write_to_file(file, argv[1], true);
    gguf_free(file);
    ggml_free(ctx);
    if (!written) return 3;
    bool ok = true;
    try {
        parakeet::ParakeetCtcModel model;
        parakeet::GgufLoadMeasure measure;
        if (parakeet::load_from_gguf_metadata_only(argv[1], model, 1, 0, false, measure) != 0) {
            ok = false;
        } else {
            for (const char * fused : {"0", "1"}) {
#ifdef _WIN32
                _putenv_s("PARAKEET_AFFINE_FUSED", fused);
#else
                setenv("PARAKEET_AFFINE_FUSED", fused, 1);
#endif
                parakeet::BlockSubstageTimes times;
                times.ff1_ms = 123.0;
                const int rc = parakeet::profile_block_substages(model, 1, 0, 1, times);
                if (rc != -2 || times.ff1_ms != 123.0) {
                    std::fprintf(stderr, "FAIL: packed profiling was not rejected (fused=%s rc=%d)\n", fused, rc);
                    ok = false;
                }
            }
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        ok = false;
    }
    std::remove(argv[1]);
    return ok ? 0 : 1;
}
