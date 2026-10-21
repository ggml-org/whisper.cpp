// Fit-projection parity tests (include/tts-cpp/audio8/fit.h): assert that
// the metadata-only memory projection matches what a REAL load and REAL runs
// actually allocate, byte for byte where the projection is exact by
// construction.
//
// Two modes:
//
//   * No arguments (the always-on CI form): a tiny but complete audio8-lm
//     GGUF is synthesized on the fly (same trick as test_gguf_load.cpp), so
//     the LM-side parity gates run with no model fixture at all:
//       1. projected LM weight bytes == the buffer a real load_lm allocates;
//       2. projected KV-slab bytes == the slow+fast cache buffers;
//       3. projected slow arena (max of prompt prefill and deepest decode
//          step) == the gallocr buffer real slow_step calls reserve;
//       4. projected fast arena == the gallocr buffer real fast_step
//          reserves;
//       5. a missing / wrong-architecture model is Error, never Success;
//       8. a weightless description GGUF with the vocabulary stripped loads
//          metadata-only, while a real load still refuses it.
//
//   * With arguments <lm.gguf> <decoder.gguf> [encoder.gguf] [n_gpu_layers]:
//     the full-pipeline gates on real fixtures -- everything above via
//     fit_params, plus codec parity: projected decode arenas (latent +
//     planned synthesis block) == what a real decode_codes leaves in
//     model.allocr / model.block_allocr; and near-INT_MAX workloads
//     (prompt_tokens, reference_seconds) are Error/"workload-too-large",
//     never a sign-overflowed graph shape.
//
// Exit 0 on success; non-zero with a FAIL line per broken invariant.

#include "tts-cpp/audio8/fit.h"

#include "test_env_portable.h"
#include "tiny_lm.h"

#include "audio8/graph.h"
#include "audio8/internal.h"
#include "fit_price.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace tts_cpp::audio8::detail;

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

void expect_eq(uint64_t projected, uint64_t real, const std::string & what) {
    if (projected != real) {
        fail(what + ": projected " + std::to_string(projected) +
             " != allocated " + std::to_string(real));
    }
}

// Price a freshly built LM graph the way the projector does (fit_price.h
// mirrors prepare_graph's dispatch).
bool price(lm_model & lm, scratch & build, ::tts_cpp::detail::fit_graph_price & out) {
    return build.ok() &&
           ::tts_cpp::detail::fit_price_graph(lm.backend, build.graph,
                                              2 * AUDIO8_MAX_NODES, out);
}

// What a real fast_step leaves allocated: one arena per position it ran.
uint64_t real_fast_arena_bytes(const lm_model & model) {
    uint64_t total = 0;
    for (const lm_model::fast_graph & cached : model.fast_graphs) {
        if (cached.allocr) total += ggml_gallocr_get_buffer_size(cached.allocr, 0);
    }
    return total;
}

// And in host RAM, one graph context per position. The projector multiplies
// scratch_arena_bytes by this count, so the count is what has to be pinned --
// the size itself comes from the same function the arenas are built with.
uint64_t retained_fast_contexts(const lm_model & model) {
    uint64_t kept = 0;
    for (const lm_model::fast_graph & cached : model.fast_graphs) {
        if (cached.ctx) ++kept;
    }
    return kept;
}

void run_synthetic_lm_gates() {
    const audio8_test::tiny_lm p;
    const std::string path = audio8_test::write_tiny_lm_gguf(
        p, (fs::temp_directory_path() / "test-audio8-fit-tiny-lm.gguf").string());

    // Metadata-only vs real load on the same (CPU) backend.
    lm_model mm, real;
    fit_load_measure lmm;
    std::string error;
    if (!load_lm_metadata_only(path, /*n_gpu_layers=*/0, /*backend=*/"", mm, lmm, &error)) {
        fail("load_lm_metadata_only failed: " + error);
        return;
    }
    if (!load_lm(path, /*n_gpu_layers=*/0, /*backend=*/"", real, &error)) {
        fail("real load_lm failed: " + error);
        free_lm(mm);
        return;
    }

    // 1 + 2. Weight and KV parity, byte for byte.
    expect_eq(lmm.weights_bytes, ggml_backend_buffer_get_size(real.buffer_w),
              "LM weights parity");
    expect_eq(lmm.kv_bytes,
              (uint64_t) ggml_backend_buffer_get_size(real.slow_kv.buffer) +
                  ggml_backend_buffer_get_size(real.fast_kv.buffer),
              "LM KV parity");

    // 3. Slow arena parity: real prefill + deepest decode step vs the
    //    projection of the same two shapes (one gallocr, so max).
    const int width = p.num_codebooks + 1;
    const int prompt_w = 6;
    std::vector<int32_t> frames((size_t) width * prompt_w, 0);
    for (int c = 0; c < prompt_w; ++c) frames[(size_t) c * width] = 1;
    std::vector<float> sem_logits, fast_input;
    if (!slow_step(real, frames.data(), prompt_w, /*n_past=*/0, /*n_threads=*/2,
                   sem_logits, fast_input, &error) ||
        !slow_step(real, frames.data(), /*width=*/1, /*n_past=*/prompt_w, 2,
                   sem_logits, fast_input, &error)) {
        fail("real slow_step failed: " + error);
    } else {
        ::tts_cpp::detail::fit_graph_price prefill, step;
        {
            scratch build(AUDIO8_MAX_NODES);
            slow_graph_outputs outs;
            build_slow_graph(mm, build, prompt_w, 0, outs);
            if (!price(mm, build, prefill)) fail("pricing the prefill graph failed");
        }
        {
            scratch build(AUDIO8_MAX_NODES);
            slow_graph_outputs outs;
            build_slow_graph(mm, build, 1, prompt_w, outs);
            if (!price(mm, build, step)) fail("pricing the step graph failed");
        }
        if (prefill.host_bytes == 0 && step.host_bytes == 0) {
            expect_eq(std::max(prefill.device_bytes, step.device_bytes),
                      ggml_gallocr_get_buffer_size(real.slow_allocr, 0),
                      "LM slow arena parity");
        }
    }

    // 4. Fast arena parity: a real whole-frame fast_step vs the projected sum
    //    over the positions. Each position keeps its own graph and its own
    //    arena for the life of the model, so what a real frame leaves behind is
    //    every one of them, not the widest.
    {
        std::vector<int32_t> codes;
        std::vector<float> prime_in((size_t) p.hidden, 0.0f);
        const code_picker pick = [](const std::vector<float> &, int) { return 0; };
        if (!fast_step(real, prime_in, p.semantic_begin, 2, pick, codes, &error)) {
            fail("real fast_step failed: " + error);
        } else {
            ::tts_cpp::detail::fit_graph_price projected;
            bool priced_on_device = true;
            for (int position = 0; position < p.num_codebooks; ++position) {
                scratch build(AUDIO8_FAST_MAX_NODES);
                build_fast_fit_graph(mm, build, position, /*prime=*/position == 0);
                ::tts_cpp::detail::fit_graph_price one;
                if (!price(mm, build, one)) fail("pricing a fast position graph failed");
                projected.device_bytes += one.device_bytes;
                priced_on_device = priced_on_device && one.host_bytes == 0;
            }
            if (priced_on_device) {
                expect_eq(projected.device_bytes, real_fast_arena_bytes(real),
                          "LM fast arena parity");
                expect_eq((uint64_t) p.num_codebooks, retained_fast_contexts(real),
                          "LM retained fast graph contexts");
            }
        }
    }

    free_lm(real);
    free_lm(mm);

    // 5. Errors surface as Error, never Success.
    {
        tts_cpp::audio8::FitOptions bad;
        bad.lm_gguf_path            = path + ".does-not-exist";
        bad.codec_decoder_gguf_path = path;
        tts_cpp::FitResult fr = tts_cpp::audio8::fit_params(bad);
        expect(fr.status == tts_cpp::FitStatus::Error, "missing LM was not Error");
        expect(fr.reason == "model-unreadable",
               "missing LM reason was '" + fr.reason + "'");
        // The LM GGUF is not a codec GGUF: wrong architecture must also be
        // Error, not a projection of garbage.
        tts_cpp::audio8::FitOptions wrong;
        wrong.lm_gguf_path            = path;
        wrong.codec_decoder_gguf_path = path;
        fr = tts_cpp::audio8::fit_params(wrong);
        expect(fr.status == tts_cpp::FitStatus::Error,
               "wrong-architecture decoder was not Error");

        // An explicit backend selector that cannot be initialised on this
        // host must return a distinct "backend-unavailable" reason instead
        // of being conflated with model-unreadable. init_backend emits
        // "failed to init a compute backend" for an unresolved explicit
        // name, which the fit path maps to the dedicated reason.
        tts_cpp::audio8::FitOptions unknown_backend;
        unknown_backend.lm_gguf_path            = path;
        unknown_backend.codec_decoder_gguf_path = path;
        unknown_backend.backend = "this-backend-does-not-exist";
        fr = tts_cpp::audio8::fit_params(unknown_backend);
        expect(fr.status == tts_cpp::FitStatus::Error,
               "unknown backend was not Error");
        expect(fr.reason == "backend-unavailable",
               "unknown-backend reason was '" + fr.reason + "'");

        // Devices without reliable memory telemetry (Hexagon reports
        // free=total=0 today in ggml_backend_hexagon_device_get_memory)
        // must verdict as Error "device-memory-unknown" rather than being
        // conflated with "does-not-fit". We can't fabricate a backend from
        // this test, but when a Hex device IS visible in the registry we
        // can exercise the branch end to end.
        bool hex_visible = false;
        for (size_t i = 0, n = ggml_backend_dev_count(); i < n; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!dev) continue;
            if (std::strcmp(ggml_backend_dev_name(dev), "HTP0") == 0) {
                hex_visible = true;
                break;
            }
        }
        if (hex_visible) {
            tts_cpp::audio8::FitOptions hex_opts;
            hex_opts.lm_gguf_path            = path;
            hex_opts.codec_decoder_gguf_path = path;
            hex_opts.backend                 = "hexagon";
            fr = tts_cpp::audio8::fit_params(hex_opts);
            expect(fr.status == tts_cpp::FitStatus::Error,
                   "zero-telemetry Hex backend was not Error");
            expect(fr.reason == "device-memory-unknown",
                   "zero-telemetry reason was '" + fr.reason + "'");
        }

        // Backend-selector policy: aliases, exact device names, "cpu" by
        // type, "opencl" by registry, and the "auto"/"" bypass.
        using tts_cpp::audio8::detail::backend_selection_matches;
        expect(backend_selection_matches("hexagon", "HTP", "HTP0",
                                         GGML_BACKEND_DEVICE_TYPE_ACCEL),
               "'hexagon' alias did not match HTP/HTP0");
        expect(!backend_selection_matches("hexagon", "HTP", "HTP1",
                                          GGML_BACKEND_DEVICE_TYPE_ACCEL),
               "'hexagon' alias matched HTP/HTP1 (only HTP0 is accepted)");
        expect(backend_selection_matches("HTP0", "HTP", "HTP0",
                                         GGML_BACKEND_DEVICE_TYPE_ACCEL),
               "exact device name 'HTP0' did not match");
        expect(backend_selection_matches("cpu", "CPU", "CPU",
                                         GGML_BACKEND_DEVICE_TYPE_CPU),
               "'cpu' did not match by CPU type");
        expect(!backend_selection_matches("cpu", "CUDA", "CUDA0",
                                          GGML_BACKEND_DEVICE_TYPE_GPU),
               "'cpu' matched a GPU device by name");
        expect(backend_selection_matches("opencl", "OpenCL", "GPU0",
                                         GGML_BACKEND_DEVICE_TYPE_GPU),
               "'opencl' did not match the OpenCL registry");
        expect(backend_selection_matches("webgpu", "WebGPU", "WebGPU",
                                         GGML_BACKEND_DEVICE_TYPE_GPU),
               "'webgpu' did not match the WebGPU registry");
        expect(!backend_selection_matches("webgpu", "Vulkan", "Vulkan0",
                                          GGML_BACKEND_DEVICE_TYPE_GPU),
               "'webgpu' matched a Vulkan device");
        expect(!backend_selection_matches("auto", "CUDA", "CUDA0",
                                          GGML_BACKEND_DEVICE_TYPE_GPU),
               "'auto' matched a specific device (expected false; auto defers to tier policy)");
        expect(!backend_selection_matches("", "CUDA", "CUDA0",
                                          GGML_BACKEND_DEVICE_TYPE_GPU),
               "empty backend matched a specific device");
        expect(!backend_selection_matches("CUDA0", "HTP", "HTP0",
                                          GGML_BACKEND_DEVICE_TYPE_ACCEL),
               "'CUDA0' matched an HTP device");

        // CPU override end to end: "cpu" + positive n_gpu_layers must land
        // on the CPU backend, not a GPU. Drives load_lm_metadata_only
        // directly (the always-on path only has the tiny LM; fit_params
        // needs a real codec decoder for the full projection), which runs
        // the same init_backend the real engine uses.
        {
            tts_cpp::audio8::detail::lm_model cpu_lm;
            tts_cpp::audio8::detail::fit_load_measure cpu_measure;
            std::string cpu_error;
            const bool ok = tts_cpp::audio8::detail::load_lm_metadata_only(
                path, /*n_gpu_layers=*/32, /*backend=*/"cpu", cpu_lm,
                cpu_measure, &cpu_error);
            expect(ok, "backend='cpu' + n_gpu_layers=32 failed to load: " + cpu_error);
            if (ok) {
                ggml_backend_dev_t dev =
                    cpu_lm.backend ? ggml_backend_get_device(cpu_lm.backend) : nullptr;
                expect(dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU,
                       "'cpu' + n_gpu_layers=32 landed on a non-CPU device");
            }
            free_lm(cpu_lm);
        }
    }

    // 6. A codebook count outside the supported range never reaches the
    //    per-position pricing loop: the load rejects it first. The metadata
    //    lies while the tensors stay tiny, which is exactly the malformed-file
    //    shape the bound exists for.
    {
        const std::string absurd_path = path + ".absurd-codebooks";
        ggml_context * headers = nullptr;
        gguf_init_params open_params = {/*no_alloc=*/true, &headers};
        gguf_context * g = gguf_init_from_file(path.c_str(), open_params);
        expect(g != nullptr, "could not reopen the tiny LM to corrupt it");
        if (g) {
            gguf_set_val_u32(g, "audio8.lm.num_codebooks", 1u << 20);
            gguf_write_to_file(g, absurd_path.c_str(), /*only_meta=*/true);
            gguf_free(g);
        }
        if (headers) ggml_free(headers);

        lm_model rejected;
        fit_load_measure rejected_load;
        std::string load_error;
        expect(!load_lm_metadata_only(absurd_path, 0, /*backend=*/"", rejected, rejected_load, &load_error),
               "an absurd codebook count loaded anyway");
        expect(load_error.find("codebooks") != std::string::npos,
               "the rejection does not name the codebook count: '" + load_error + "'");
        free_lm(rejected);
        fs::remove(absurd_path);
    }

    // 7. How position prices combine, by dispatch path: replayed positions each
    //    keep an arena and add; once any position is scheduler-backed, the
    //    direct ones share one growing allocator and the scheduler ones share
    //    the scheduler's, so the widest of each coexist.
    {
        ::tts_cpp::detail::fit_graph_price direct_small{100, 0, false};
        ::tts_cpp::detail::fit_graph_price direct_large{300, 0, false};
        ::tts_cpp::detail::fit_graph_price sched_small{50, 10, true};
        ::tts_cpp::detail::fit_graph_price sched_large{200, 40, true};

        ::tts_cpp::detail::fit_price_aggregate all_direct;
        all_direct.add(direct_small);
        all_direct.add(direct_large);
        expect(all_direct.all_replayed(), "direct-only prices reported a scheduler");
        expect_eq(400, all_direct.total().device_bytes, "direct-only device sum");

        ::tts_cpp::detail::fit_price_aggregate mixed;
        mixed.add(direct_small);
        mixed.add(direct_large);
        mixed.add(sched_small);
        mixed.add(sched_large);
        expect(!mixed.all_replayed(), "a scheduler-backed price went unnoticed");
        expect_eq(500, mixed.total().device_bytes,
                  "mixed device total is not widest-direct plus widest-sched");
        expect_eq(40, mixed.total().host_bytes,
                  "mixed host total is not the widest scheduler-backed portion");

        ::tts_cpp::detail::fit_price_aggregate sched_only;
        sched_only.add(sched_small);
        sched_only.add(sched_large);
        expect_eq(200, sched_only.total().device_bytes, "sched-only device peak");
    }

    // 8. A weightless description GGUF with the vocabulary stripped loads
    //    metadata-only (a fit measurement never tokenizes), while a real load
    //    still refuses the missing tokenizer keys.
    {
        const std::string desc_path = audio8_test::write_tiny_lm_gguf(
            p, (fs::temp_directory_path() / "test-audio8-fit-tiny-lm-desc.gguf").string(),
            /*with_vocab=*/false, /*only_meta=*/true);

        lm_model meta_only;
        fit_load_measure desc_load;
        std::string desc_error;
        if (!load_lm_metadata_only(desc_path, /*n_gpu_layers=*/0, /*backend=*/"", meta_only, desc_load,
                                   &desc_error)) {
            fail("metadata-only load refused a vocab-less description: " + desc_error);
        } else {
            expect(desc_load.weights_bytes > 0,
                   "vocab-less description sized 0 weight bytes");
        }
        free_lm(meta_only);

        lm_model rejected;
        expect(!load_lm(desc_path, /*n_gpu_layers=*/0, /*backend=*/"", rejected, &desc_error),
               "a real load accepted a vocab-less GGUF");
        expect(desc_error.find("tokenizer.ggml.tokens") != std::string::npos,
               "the real-load rejection does not name the vocabulary: '" + desc_error + "'");
        free_lm(rejected);
        fs::remove(desc_path);
    }

    fs::remove(path);
}

void run_fixture_gates(const std::string & lm_path, const std::string & dec_path,
                       const std::string & enc_path, int n_gpu_layers) {
    tts_cpp::audio8::FitOptions fopts;
    fopts.lm_gguf_path            = lm_path;
    fopts.codec_decoder_gguf_path = dec_path;
    fopts.codec_encoder_gguf_path = enc_path;
    fopts.n_gpu_layers            = n_gpu_layers;
    fopts.prompt_tokens           = 16;
    fopts.max_frames              = 64;

    const tts_cpp::FitResult fit = tts_cpp::audio8::fit_params(fopts);
    expect(fit.status != tts_cpp::FitStatus::Error,
           "fit_params returned Error (" + fit.reason + ") for readable fixtures");
    expect(fit.device.weights_bytes > 0,       "projected weights_bytes == 0");
    expect(fit.device.state_bytes > 0,         "projected state_bytes == 0");
    expect(fit.device.lm_compute_bytes > 0,    "projected lm_compute_bytes == 0");
    expect(fit.device.codec_compute_bytes > 0, "projected codec_compute_bytes == 0");
    expect(!fit.report.empty(),                "empty report");
    if (g_failures) return;
    std::printf("%s", fit.report.c_str());

    // Codec decode parity: a real decode leaves the latent arena in
    // model.allocr and the block arena in model.block_allocr; the projection
    // of the same frame count must match both (their sum, since both stay
    // resident), byte for byte on the direct-dispatch path.
    {
        codec_model real;
        std::string error;
        if (!load_codec(dec_path, n_gpu_layers, /*backend=*/"", real, &error)) {
            fail("real load_codec failed: " + error);
            return;
        }
        const int n_frames = 16;
        std::vector<int32_t> codes((size_t) real.hp.num_codebooks * n_frames, 0);
        std::vector<float> pcm;
        if (!decode_codes(real, codes.data(), n_frames, /*n_threads=*/2,
                          cancel_hook(), pcm, &error)) {
            fail("real decode_codes failed: " + error);
        } else {
            codec_model mm;
            fit_load_measure dm;
            if (!load_codec_metadata_only(dec_path, n_gpu_layers, /*backend=*/"", mm, dm, &error)) {
                fail("load_codec_metadata_only failed: " + error);
            } else {
                expect_eq(dm.weights_bytes, ggml_backend_buffer_get_size(real.buffer_w),
                          "codec weights parity");
                codec_fit_measure cm;
                if (!measure_decode_memory(mm, n_frames, cm)) {
                    fail("measure_decode_memory failed");
                } else if (cm.host_bytes == 0) {
                    const uint64_t real_arenas =
                        (uint64_t) ggml_gallocr_get_buffer_size(real.allocr, 0) +
                        ggml_gallocr_get_buffer_size(real.block_allocr, 0);
                    expect_eq(cm.device_bytes, real_arenas, "codec decode arena parity");
                }
                free_codec(mm);
            }
        }
        free_codec(real);
    }

    // Near-INT_MAX workloads are rejected strictly (Error /
    // "workload-too-large"), never priced through wrapped-int graph shapes --
    // a preflight that crashes on the inputs it exists to reject violates its
    // own contract.
    {
        tts_cpp::audio8::FitOptions huge = fopts;
        huge.prompt_tokens = std::numeric_limits<int>::max();
        const tts_cpp::FitResult fr = tts_cpp::audio8::fit_params(huge);
        expect(fr.status == tts_cpp::FitStatus::Error,
               "near-INT_MAX prompt_tokens was not Error");
        expect(fr.reason == "workload-too-large",
               "near-INT_MAX prompt_tokens reason was '" + fr.reason + "'");
    }
    if (!enc_path.empty()) {
        // A reference whose sample count exceeds int must trip the widened
        // encode_positions product guard (or the RoPE-table check, whichever
        // bites first), never sign-overflow inside encoder_positions.
        tts_cpp::audio8::FitOptions huge = fopts;
        huge.reference_seconds = 2.0e5f;  // >= 2^31 samples at any real rate
        const tts_cpp::FitResult fr = tts_cpp::audio8::fit_params(huge);
        expect(fr.status == tts_cpp::FitStatus::Error,
               "over-int reference_seconds was not Error");
        expect(fr.reason == "workload-too-large",
               "over-int reference_seconds reason was '" + fr.reason + "'");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    // The arena gates compare against a real ggml decode, which a staged Core
    // ML sidecar would replace.
    setenv("AUDIO8_COREML_DISABLE", "1", 1);
    if (argc >= 3) {
        run_fixture_gates(argv[1], argv[2], argc > 3 ? argv[3] : "",
                          argc > 4 ? std::atoi(argv[4]) : 0);
    } else {
        run_synthetic_lm_gates();
    }
    if (g_failures == 0) {
        std::printf("test-audio8-fit-params: all checks passed\n");
    }
    return g_failures;
}
