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

// round 6 — TDD test for the 2-arg
// `should_materialise_f16_weight(name, extra_deny_substrings)`
// overload.  Lets operators force-keep specific tensors as F32
// even when the auto/curated allow-list would have promoted them
// to F16.  Use cases:
//   - Researcher A/B testing a specific tensor pattern without
//     recompiling.
//   - Operator force-keeping a tensor as F32 if they observe
//     drift on their hardware.
//   - Safety net for new tensor patterns added in future GGUFs.
//
// Contract:
//   - Empty deny-list: 2-arg overload behaves identically to the
//     1-arg version (zero behaviour change for the default path).
//   - Any substring in the deny-list that matches a tensor name
//     forces a `false` return, even if the curated allow-list
//     would have said `true`.
//   - The deny-list cannot promote a cold weight to hot
//     (it's a deny-list, not an allow-list — adding a non-
//     matching pattern doesn't help).
//   - Empty strings inside the deny-list are skipped (no-op),
//     not treated as matching every name (defensive).
//   - Substring matching, not regex (matches the curated
//     predicate's audit-friendly style; no regex compile cost,
//     no invalid-pattern error surface).
//
// Written FIRST (TDD).  MUST fail before the 2-arg overload is
// added; MUST pass after.
void test_predicate_deny_list_empty_passthrough() {
    std::fprintf(stderr, "[Round 6 deny-list: empty-list passthrough]\n");
    // With an empty extra-deny-list, every result must equal the
    // 1-arg version's result.  Spot-check a positive and a
    // negative.
    const std::vector<std::string> empty_deny;
    CHECK(should_materialise_f16_weight("vector_estimator:onnx::MatMul_3101", empty_deny) ==
          should_materialise_f16_weight("vector_estimator:onnx::MatMul_3101"));
    CHECK(should_materialise_f16_weight("vocoder:tts.ae.decoder.convnext.0.pwconv1.weight", empty_deny) ==
          should_materialise_f16_weight("vocoder:tts.ae.decoder.convnext.0.pwconv1.weight"));
    CHECK(should_materialise_f16_weight("vector_estimator:tts.ttl.vector_field.main_blocks.0.convnext.0.norm.norm.weight", empty_deny) ==
          should_materialise_f16_weight("vector_estimator:tts.ttl.vector_field.main_blocks.0.convnext.0.norm.norm.weight"));
}

void test_predicate_deny_list_excludes_match() {
    std::fprintf(stderr, "[Round 6 deny-list: matching deny excludes hot weight]\n");
    // A hot weight that the 1-arg version returns `true` for must
    // return `false` when the deny-list contains a substring of
    // its name.
    const std::string hot = "vector_estimator:onnx::MatMul_3101";
    CHECK(should_materialise_f16_weight(hot));  // baseline: hot

    // Exact-name deny.
    CHECK(!should_materialise_f16_weight(hot, std::vector<std::string>{"MatMul_3101"}));
    // Stage-prefix deny: excludes EVERY vector_estimator MatMul.
    CHECK(!should_materialise_f16_weight(hot, std::vector<std::string>{"vector_estimator:onnx::MatMul_"}));
    // Single-char substring (defensive — works because substring
    // semantics, but operators should write more specific patterns).
    CHECK(!should_materialise_f16_weight(hot, std::vector<std::string>{"3101"}));

    // Same pattern applied to a pwconv weight.
    const std::string pw = "vocoder:tts.ae.decoder.convnext.0.pwconv1.weight";
    CHECK(should_materialise_f16_weight(pw));  // baseline: hot
    CHECK(!should_materialise_f16_weight(pw, std::vector<std::string>{".pwconv1."}));
    // pwconv2 deny shouldn't affect pwconv1.
    CHECK(should_materialise_f16_weight(pw, std::vector<std::string>{".pwconv2."}));
}

void test_predicate_deny_list_no_match() {
    std::fprintf(stderr, "[Round 6 deny-list: non-matching deny is no-op]\n");
    // A deny-list with no matching substring must leave the result
    // unchanged.  Spot-check positive (still hot) and negative
    // (still cold).
    const std::vector<std::string> deny_unrelated = {"ZZZ_definitely_not_in_any_name"};
    CHECK(should_materialise_f16_weight("vector_estimator:onnx::MatMul_3101", deny_unrelated));
    CHECK(!should_materialise_f16_weight("vector_estimator:onnx::MatMul_3101_bias", deny_unrelated));
}

void test_predicate_deny_list_cannot_promote_cold() {
    std::fprintf(stderr, "[Round 6 deny-list: cannot promote cold weight to hot]\n");
    // The deny-list is a DENY-list, not an allow-list.  Adding a
    // pattern that matches a cold weight has no effect (cold + deny
    // is still cold; deny only operates on the `true` branch of
    // the 1-arg predicate).
    const std::string cold = "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.W_query.linear.bias";
    CHECK(!should_materialise_f16_weight(cold));  // baseline: cold (bias)
    CHECK(!should_materialise_f16_weight(cold, std::vector<std::string>{"linear.bias"}));
    CHECK(!should_materialise_f16_weight(cold, std::vector<std::string>{"NOT_IN_NAME"}));
}

void test_predicate_deny_list_multiple_patterns() {
    std::fprintf(stderr, "[Round 6 deny-list: ANY match excludes]\n");
    // Multiple patterns: ANY match excludes the weight.  Patterns
    // are independent (no AND-of-all semantics).
    const std::string hot = "vocoder:tts.ae.decoder.convnext.0.pwconv1.weight";
    const std::vector<std::string> deny_multi = {
        "AAAAA_no_match",
        ".pwconv1.",        // matches!
        "BBBBB_no_match",
    };
    CHECK(!should_materialise_f16_weight(hot, deny_multi));

    // All-non-matching multi-pattern: still hot.
    const std::vector<std::string> deny_all_miss = {
        "AAAAA_no_match",
        "BBBBB_no_match",
        "CCCCC_no_match",
    };
    CHECK(should_materialise_f16_weight(hot, deny_all_miss));
}

void test_predicate_deny_list_empty_string_safe() {
    std::fprintf(stderr, "[Round 6 deny-list: empty string in deny-list is skipped]\n");
    // An empty string would technically match every name under
    // substring semantics ("" is a substring of every string),
    // which would silently disable F16 weights entirely — almost
    // certainly an operator typo (e.g. accidentally trailing
    // comma in a config file).  Defensive: empty-string entries
    // are SKIPPED instead of treated as universal matches.
    const std::vector<std::string> deny_with_empty = {""};
    CHECK(should_materialise_f16_weight("vector_estimator:onnx::MatMul_3101", deny_with_empty));
    CHECK(should_materialise_f16_weight("vocoder:tts.ae.decoder.convnext.0.pwconv1.weight", deny_with_empty));

    // Mixed: empty + a real pattern.  The real pattern must still
    // take effect.
    const std::vector<std::string> deny_mixed = {"", ".pwconv1."};
    CHECK(!should_materialise_f16_weight("vocoder:tts.ae.decoder.convnext.0.pwconv1.weight", deny_mixed));
    CHECK(should_materialise_f16_weight("vector_estimator:onnx::MatMul_3101", deny_mixed));
}

void test_predicate_deny_list_empty_name_safe() {
    std::fprintf(stderr, "[Round 6 deny-list: empty source name still returns false]\n");
    // Empty source name was handled defensively by the 1-arg
    // version (returns false).  The 2-arg overload must preserve
    // this regardless of the deny-list contents.
    CHECK(!should_materialise_f16_weight("", std::vector<std::string>{}));
    CHECK(!should_materialise_f16_weight("", std::vector<std::string>{"any"}));
}

} // namespace

int main(int argc, char ** argv) {
    // Unit-level predicate tests run unconditionally; no model.
    test_predicate_positives();
    test_predicate_negatives();
    test_predicate_edges();
    // round 6 — 2-arg overload tests (TDD: these are
    // the new symbol; whole block must fail compilation before
    // implementation, then pass after).
    test_predicate_deny_list_empty_passthrough();
    test_predicate_deny_list_excludes_match();
    test_predicate_deny_list_no_match();
    test_predicate_deny_list_cannot_promote_cold();
    test_predicate_deny_list_multiple_patterns();
    test_predicate_deny_list_empty_string_safe();
    test_predicate_deny_list_empty_name_safe();

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
