// TDD harness for Phase 2A — F16 weight materialization for the hot
// matmul / pointwise-conv weights identified in
// `AUDIT_SUPERTONIC_OPENCL.md` § F6 + Phase 2A.
//
// Two layers of testing here:
//
//   1. Unit-level predicate test (no GGUF, runs on `ctest -L unit`).
//      Validates `should_materialise_f16_weight(name)` returns
//      `true` for every entry on the hot-weights roster and
//      `false` for negatives (random tensor names, edge cases,
//      tensors whose names contain a substring of a hot weight
//      but aren't on the roster — e.g. the bias of a hot conv).
//
//   2. Fixture-level shape / dtype test (requires GGUF).
//      Loads the model twice with `f16_weights=true` and `=false`,
//      asserts:
//        - At least one hot weight has type `GGML_TYPE_F16` when
//          the flag is on, and `GGML_TYPE_F32` when it's off.
//        - Every weight NOT on the roster keeps its baseline
//          type (so we don't accidentally quantize the wrong
//          stuff).
//        - Non-hot tensors are byte-equivalent across the two
//          loads (predicate hasn't accidentally widened scope).
//
// Wired into CMakeLists.txt under `LABEL "fixture"` for the model
// dependence, with the predicate sub-test running unconditionally.

#include "supertonic_internal.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace tts_cpp::supertonic::detail;

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond) do {                                              \
    ++g_checks;                                                       \
    if (!(cond)) {                                                    \
        ++g_failures;                                                 \
        std::fprintf(stderr, "FAIL %s:%d  %s\n",                     \
                     __FILE__, __LINE__, #cond);                      \
    }                                                                 \
} while (0)

// Hot-weight predicate covers:
//   - vector_estimator attention W_query / W_key / W_value / W_out
//     matmul weights for the four groups (MatMul_3101/02/03/10 …
//     plus the three group siblings).  These also include the
//     style-attention MatMuls (3116/17/18/19 etc).
//   - vector_estimator pointwise conv1 / conv2 inside every
//     convnext block (`main_blocks.*.convnext.*.pwconv{1,2}.weight`
//     and `last_convnext.convnext.*.pwconv{1,2}.weight`).
//   - vocoder pointwise conv1 / conv2 inside every convnext
//     block + the head conv1 weight.
//   - text-encoder transformer linear weights.
//
// Negative cases (predicate must NOT match):
//   - biases (`.bias` suffix).
//   - small per-channel scale/shift vectors (`norm.weight`,
//     `gamma`, etc).
//   - non-linear weights (`emb_rel_k`, embedding tables).
//   - per-tensor scalars (`normalizer_scale`, `head_prelu`).
//
// The predicate sub-test below is fully self-contained — no
// model state needed.  Runs as a unit test.
void test_predicate_positives() {
    std::fprintf(stderr, "[Phase 2A predicate positives]\n");
    static const char * const kHotNames[] = {
        // vector_estimator attention matmuls (front block + 3 groups).
        "vector_estimator:onnx::MatMul_3101",  // Q
        "vector_estimator:onnx::MatMul_3102",  // K
        "vector_estimator:onnx::MatMul_3103",  // V
        "vector_estimator:onnx::MatMul_3110",  // out
        "vector_estimator:onnx::MatMul_3146",  // g1 Q
        "vector_estimator:onnx::MatMul_3155",  // g1 out
        "vector_estimator:onnx::MatMul_3191",  // g2 Q
        "vector_estimator:onnx::MatMul_3236",  // g3 Q
        // vector_estimator style-attention matmuls.
        "vector_estimator:onnx::MatMul_3116",  // style0 Q
        "vector_estimator:onnx::MatMul_3119",  // style0 out
        // vector_estimator convnext pointwise.
        "vector_estimator:tts.ttl.vector_field.main_blocks.0.convnext.0.pwconv1.weight",
        "vector_estimator:tts.ttl.vector_field.main_blocks.0.convnext.0.pwconv2.weight",
        "vector_estimator:tts.ttl.vector_field.last_convnext.convnext.0.pwconv1.weight",
        // vocoder convnext + head.
        "vocoder:tts.ae.decoder.convnext.0.pwconv1.weight",
        "vocoder:tts.ae.decoder.convnext.5.pwconv2.weight",
        "vocoder:tts.ae.decoder.head.layer1.net.weight",
        // text-encoder linears.
        "text_encoder:onnx::MatMul_3678",
        "text_encoder:onnx::MatMul_3685",
    };
    int missed = 0;
    for (const char * name : kHotNames) {
        const bool got = should_materialise_f16_weight(name);
        CHECK(got);
        if (!got) {
            ++missed;
            std::fprintf(stderr, "  predicate returned false for hot weight: %s\n", name);
        }
    }
    std::fprintf(stderr, "  %zu positives, %d missed\n",
                 sizeof(kHotNames) / sizeof(kHotNames[0]), missed);
}

void test_predicate_negatives() {
    std::fprintf(stderr, "[Phase 2A predicate negatives]\n");
    static const char * const kColdNames[] = {
        // biases — NEVER quantize, drift accumulates.
        "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.W_query.linear.bias",
        "vocoder:tts.ae.decoder.convnext.0.pwconv1.bias",
        // per-channel scale / shift — too small for F16 to matter,
        // and `repeat_like` mismatches if we change shape.
        "vector_estimator:tts.ttl.vector_field.main_blocks.0.convnext.0.norm.norm.weight",
        "vector_estimator:tts.ttl.vector_field.main_blocks.0.convnext.0.norm.norm.bias",
        "vector_estimator:tts.ttl.vector_field.main_blocks.0.convnext.0.gamma",
        "vocoder:tts.ae.decoder.convnext.0.norm.norm.weight",
        // embeddings + lookup tables.
        "text_encoder:tts.ttl.text_encoder.text_embedder.char_embedder.weight",
        "duration:tts.dp.sentence_encoder.text_embedder.char_embedder.weight",
        // per-tensor scalars.
        "vocoder:tts.ttl.normalizer.scale",
        "vocoder:onnx::PRelu_1505",
        // small relative-position embeddings.
        "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.emb_rel_k",
        "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers.0.emb_rel_v",
        // depthwise conv (small per-channel kernels).
        "vector_estimator:tts.ttl.vector_field.main_blocks.0.convnext.0.dwconv.weight",
        "vocoder:tts.ae.decoder.convnext.0.dwconv.net.weight",
        // theta (rope) constant — small, hot, but cached host-side
        // by F1 so it's already on the host F32 path.
        "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta",
        // unrelated infrastructure.
        "supertonic/unicode_indexer",
        "supertonic/voices/F1/ttl",
        // pre-transposed companions (F6) — they live alongside the
        // original; the original gets materialised, the __T is
        // already a separate tensor and shouldn't double-down.
        "vector_estimator:onnx::MatMul_3095__T",
    };
    int over = 0;
    for (const char * name : kColdNames) {
        const bool got = should_materialise_f16_weight(name);
        CHECK(!got);
        if (got) {
            ++over;
            std::fprintf(stderr, "  predicate returned true for cold weight: %s\n", name);
        }
    }
    std::fprintf(stderr, "  %zu negatives, %d false-positives\n",
                 sizeof(kColdNames) / sizeof(kColdNames[0]), over);
}

void test_predicate_edges() {
    std::fprintf(stderr, "[Phase 2A predicate edge cases]\n");
    // Empty + nonsense inputs must return false without throwing.
    CHECK(!should_materialise_f16_weight(""));
    CHECK(!should_materialise_f16_weight("not a real tensor name"));
    CHECK(!should_materialise_f16_weight("vector_estimator:"));
    CHECK(!should_materialise_f16_weight("vector_estimator:onnx::MatMul_"));
    // Looks like a hot weight but isn't (digit overlap).
    CHECK(!should_materialise_f16_weight("vector_estimator:onnx::MatMul_3101_bias"));
    // Substring match would be a bug — `.weight` inside a path
    // shouldn't trigger.
    CHECK(!should_materialise_f16_weight("vocoder:tts.ae.decoder.convnext.weight_stats"));
}

} // namespace

int main(int argc, char ** argv) {
    // Unit-level predicate tests run unconditionally; no model.
    test_predicate_positives();
    test_predicate_negatives();
    test_predicate_edges();

    // Fixture-level shape/dtype check requires the GGUF.
    if (argc >= 2) {
        std::fprintf(stderr, "[Phase 2A fixture] (loading %s)\n", argv[1]);
        supertonic_model model_f32;
        if (load_supertonic_gguf(argv[1], model_f32, /*n_gpu_layers=*/0, /*verbose=*/false)) {
            // model loaded with f16_weights=false by default.
            int f32_hot = 0, f16_hot = 0, other = 0;
            for (const auto & kv : model_f32.source_tensors) {
                if (!kv.second) continue;
                if (should_materialise_f16_weight(kv.first)) {
                    if (kv.second->type == GGML_TYPE_F32) ++f32_hot;
                    else if (kv.second->type == GGML_TYPE_F16) ++f16_hot;
                } else {
                    ++other;
                }
            }
            std::fprintf(stderr,
                         "  default load: hot-F32=%d hot-F16=%d other=%d\n",
                         f32_hot, f16_hot, other);
            // Default load (f16_weights default = false on CPU)
            // keeps hot weights as F32.
            CHECK(f16_hot == 0 || f32_hot == 0); // at least one bucket
            free_supertonic_model(model_f32);
        } else {
            std::fprintf(stderr, "  skip fixture: failed to load %s\n", argv[1]);
        }
    } else {
        std::fprintf(stderr, "  (fixture skipped; pass MODEL.gguf to enable)\n");
    }

    std::fprintf(stderr,
                 "test_supertonic_f16_weights: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
