// CPU-only, weight-free unit tests for the ACE-Step pipeline's pure logic.
//
// None of these need a GGUF fixture — they exercise the deterministic math
// that the rest of the pipeline is built on, so a refactor that drifts from
// the acestep.cpp reference breaks here (fast, on a fresh checkout under
// `ctest -L unit`) before the fixture-bound integration tests get a chance to.
//
// Coverage:
//   1. dit_build_schedule  — flow-matching time schedule (shift/steps).
//   2. philox_randn        — Philox4x32-10 + Box-Muller (torch.randn parity).
//   3. fsq_decode_index    — FSQ index -> 6 normalized dims (strides 8/8/8/5/5/5).
//   4. sample_top_k_p      — top-k/top-p LM sampler (determinism + argmax).
//   5. vae_progress_pct    — VAE decode progress clamp/monotonicity.
//   6. GPU device types    — discrete and integrated GPUs are selectable.
//   7. stage placement     — which backend the LM / detokenizer / encoders run on.

#include "backend_registry.h"
#include "dit_ggml.h"
#include "detok_ggml.h"
#include "lm_pipeline.h"
#include "philox.h"
#include "stage_placement.h"
#include "vae_ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++g_checks;                                                                   \
        if (!(cond)) {                                                                \
            ++g_failures;                                                             \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        }                                                                             \
    } while (0)

bool approx(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) <= eps; }

// 1. dit_build_schedule ------------------------------------------------------
void test_schedule() {
    using tts_cpp::acestep::dit_build_schedule;
    std::vector<float> sched;

    // shift == 1.0 collapses to a linear ramp: schedule[i] = 1 - i/N.
    dit_build_schedule(1.0f, 8, sched);
    CHECK(sched.size() == 8);
    CHECK(approx(sched[0], 1.0f));                 // first step starts at pure noise
    CHECK(approx(sched[4], 1.0f - 4.0f / 8.0f));   // == 0.5
    for (size_t i = 1; i < sched.size(); ++i) CHECK(sched[i] < sched[i - 1]);  // strictly decreasing

    // shift > 1 (turbo uses 3.0) still starts at 1.0 and stays monotone.
    dit_build_schedule(3.0f, 8, sched);
    CHECK(sched.size() == 8);
    CHECK(approx(sched[0], 1.0f));
    for (size_t i = 1; i < sched.size(); ++i) CHECK(sched[i] < sched[i - 1]);
    for (float v : sched) CHECK(v > 0.0f && v <= 1.0f);
}

// 2. philox_randn ------------------------------------------------------------
void test_philox() {
    using tts_cpp::acestep::philox_randn;

    // Golden vector for seed 42 (bf16-rounded, the mode the engine uses for
    // the DiT initial noise). These values were validated corr=1.0 against
    // torch.randn() on CUDA via acestep.cpp's --dump; they lock the port.
    const float golden[8] = { 0.194335938f,  2.156250000f,  -0.171875000f, 0.847656250f,
                              -1.921875000f, 0.652343750f,  -0.648437500f, -0.816406250f };
    float out[8];
    philox_randn(42, out, 8, /*bf16_round=*/true);
    for (int i = 0; i < 8; ++i) CHECK(approx(out[i], golden[i], 1e-6f));

    // Determinism: same seed -> identical stream.
    float a[16], b[16];
    philox_randn(1234, a, 16, true);
    philox_randn(1234, b, 16, true);
    for (int i = 0; i < 16; ++i) CHECK(a[i] == b[i]);

    // Different seeds -> different stream (extremely unlikely to collide).
    float c[16];
    philox_randn(4321, c, 16, true);
    bool differs = false;
    for (int i = 0; i < 16; ++i) differs |= (a[i] != c[i]);
    CHECK(differs);

    // Statistical sanity over a large draw: mean ~ 0, std ~ 1.
    const int          N = 8192;
    std::vector<float> big(N);
    philox_randn(7, big.data(), N, false);
    double mean = 0.0;
    for (float v : big) mean += v;
    mean /= N;
    double var = 0.0;
    for (float v : big) var += (v - mean) * (v - mean);
    var /= N;
    CHECK(std::fabs(mean) < 0.05);
    CHECK(std::fabs(std::sqrt(var) - 1.0) < 0.05);
}

// 3. fsq_decode_index --------------------------------------------------------
void test_fsq() {
    using tts_cpp::acestep::fsq_decode_index;
    float out[6];

    // index 0 -> all digits 0 -> every dim at the low end (-1).
    fsq_decode_index(0, out);
    for (int d = 0; d < 6; ++d) CHECK(approx(out[d], -1.0f));

    // digit 1 in dim0 (L=8, half=3.5): 1/3.5 - 1.
    fsq_decode_index(1, out);
    CHECK(approx(out[0], 1.0f / 3.5f - 1.0f));
    for (int d = 1; d < 6; ++d) CHECK(approx(out[d], -1.0f));

    // Max index across all dims -> every dim at the high end (+1).
    int max_index = 8 * 8 * 8 * 5 * 5 * 5 - 1;
    fsq_decode_index(max_index, out);
    for (int d = 0; d < 6; ++d) CHECK(approx(out[d], 1.0f));

    // Output is always in [-1, 1] (small FP slack at the boundaries).
    const float slack = 1e-4f;
    for (int idx = 0; idx < 500; idx += 37) {
        fsq_decode_index(idx, out);
        for (int d = 0; d < 6; ++d) CHECK(out[d] >= -1.0f - slack && out[d] <= 1.0f + slack);
    }
}

// 4. sample_top_k_p ----------------------------------------------------------
void test_sampler() {
    using tts_cpp::acestep::sample_top_k_p;

    const int         V = 32;
    std::vector<float> logits(V, 0.0f);
    logits[17] = 10.0f;  // clearly dominant token

    // top_k = 1 keeps only the argmax, so the pick is deterministic regardless
    // of the RNG draw.
    for (uint32_t seed = 0; seed < 8; ++seed) {
        std::vector<float> l = logits;
        std::mt19937       rng(seed);
        CHECK(sample_top_k_p(l.data(), V, 1.0f, 1.0f, /*top_k=*/1, rng) == 17);
    }

    // Same RNG seed -> identical token sequence (reproducible generation).
    std::vector<float> l1 = logits, l2 = logits;
    std::mt19937       r1(99), r2(99);
    for (int step = 0; step < 5; ++step) {
        std::vector<float> a = l1, b = l2;
        int ta = sample_top_k_p(a.data(), V, 0.85f, 0.9f, 0, r1);
        int tb = sample_top_k_p(b.data(), V, 0.85f, 0.9f, 0, r2);
        CHECK(ta == tb);
        CHECK(ta >= 0 && ta < V);
    }
}

// 5. vae_progress_pct --------------------------------------------------------
// The VAE decode reports progress per computed graph node. A GPU+CPU scheduler
// can insert extra copy/split nodes, so the callback may fire MORE than
// ggml_graph_n_nodes(gf) times; the percentage must stay monotone and bounded
// to [0, 100] (no progress-bar overshoot). This locks that clamp/throttle math.
void test_vae_progress() {
    using tts_cpp::acestep::vae_progress_pct;

    // Endpoints and a midpoint.
    CHECK(vae_progress_pct(0, 200) == 0);
    CHECK(vae_progress_pct(100, 200) == 50);
    CHECK(vae_progress_pct(200, 200) == 100);

    // Overshoot: the scheduler fires past `total` -> clamped to 100, never more.
    CHECK(vae_progress_pct(201, 200) == 100);
    CHECK(vae_progress_pct(10000, 200) == 100);

    // Degenerate inputs are safe (no div-by-zero, no negative pct).
    CHECK(vae_progress_pct(5, 0) == 0);
    CHECK(vae_progress_pct(5, -1) == 0);
    CHECK(vae_progress_pct(-3, 200) == 0);

    // Monotone non-decreasing and bounded across a full sweep that overshoots,
    // mirroring the eval callback incrementing `done` once per fired node.
    const int total = 137;  // odd, to exercise integer division
    int       prev  = -1;
    int       last_emitted = -1;
    for (int done = 1; done <= total + 25; ++done) {  // +25 => simulate extra copy/split nodes
        int pct = vae_progress_pct(done, total);
        CHECK(pct >= 0 && pct <= 100);
        CHECK(pct >= prev);  // never goes backwards
        prev = pct;
        // Throttle: only distinct percentages would be surfaced to the user cb.
        if (pct != last_emitted) last_emitted = pct;
    }
    CHECK(prev == 100);          // ends exactly at 100
    CHECK(last_emitted == 100);  // the final surfaced value is 100
}

// 6. GPU device types --------------------------------------------------------
// Vulkan classifies UMA adapters such as Android Mali as IGPU. AceStep must
// accept both device classes while rejecting CPU and non-GPU accelerators.
void test_backend_device_types() {
    using tts_cpp::acestep::backend_device_type_is_gpu;
    using tts_cpp::acestep::backend_reg_name_is_validated_gpu;

    CHECK(backend_device_type_is_gpu(GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_device_type_is_gpu(GGML_BACKEND_DEVICE_TYPE_IGPU));
    CHECK(!backend_device_type_is_gpu(GGML_BACKEND_DEVICE_TYPE_CPU));
    CHECK(!backend_device_type_is_gpu(GGML_BACKEND_DEVICE_TYPE_ACCEL));

    CHECK(backend_reg_name_is_validated_gpu("Vulkan"));
    CHECK(backend_reg_name_is_validated_gpu("MTL"));
    CHECK(backend_reg_name_is_validated_gpu("Metal"));
    CHECK(!backend_reg_name_is_validated_gpu("OpenCL"));
    CHECK(!backend_reg_name_is_validated_gpu("CUDA"));
    CHECK(!backend_reg_name_is_validated_gpu(nullptr));
}

// 7. stage placement ---------------------------------------------------------
// Which backend each stage runs on decides which numerical path the generated
// audio takes, so the policy is locked here rather than only observed on a
// device lane. Mirrors the three branches Engine::create() relies on: the
// backend allowlist, the CPU fallback for everything else, and the env
// overrides layered on top.
void test_stage_placement() {
    using tts_cpp::acestep::backend_name_is_metal;
    using tts_cpp::acestep::backend_name_is_vulkan;
    using tts_cpp::acestep::PlacementOverrides;
    using tts_cpp::acestep::resolve_stage_placement;
    using tts_cpp::acestep::StagePlacement;

    // -- backend name matching ------------------------------------------------
    // ggml-metal registers as "MTL"; older ggml reported "Metal". Both must
    // match or the allowlist is silently dead on one of them.
    CHECK(backend_name_is_metal("MTL"));
    CHECK(backend_name_is_metal("Metal"));
    CHECK(backend_name_is_vulkan("Vulkan"));

    // The input is the REGISTRY name, which carries no device-index suffix.
    // ggml_backend_name() would hand over "MTL0" / "Vulkan0" and match nothing.
    CHECK(!backend_name_is_metal("MTL0"));
    CHECK(!backend_name_is_vulkan("Vulkan0"));

    // Exact compare: no case folding, no substring match, and null/empty safe.
    CHECK(!backend_name_is_metal("mtl"));
    CHECK(!backend_name_is_metal("metal"));
    CHECK(!backend_name_is_metal("MTLX"));
    CHECK(!backend_name_is_vulkan("vulkan"));
    CHECK(!backend_name_is_metal("CUDA"));
    CHECK(!backend_name_is_vulkan("MTL"));
    CHECK(!backend_name_is_metal("Vulkan"));
    CHECK(!backend_name_is_metal(""));
    CHECK(!backend_name_is_vulkan(""));
    CHECK(!backend_name_is_metal(nullptr));
    CHECK(!backend_name_is_vulkan(nullptr));

    const PlacementOverrides none;

    // -- allowlist: LM + detokenizer stay on the GPU ---------------------------
    for (const char * allowed : { "Vulkan", "MTL", "Metal" }) {
        StagePlacement p = resolve_stage_placement(allowed, none);
        CHECK(p.lm_on_gpu);
        CHECK(p.detok_on_gpu);
        CHECK(p.enc_on_gpu);  // encoders follow the GPU on every backend
    }

    // -- fallback: everything else keeps the shipping CPU placement -----------
    // Unmeasured backends must not silently pick up the GPU path. "MTL0" is in
    // this list on purpose: a suffixed name is NOT the allowlisted one.
    const char * const others[] = { "CUDA", "OpenCL", "SYCL", "BLAS", "CPU", "MTL0", "Vulkan0", "", nullptr };
    for (const char * other : others) {
        StagePlacement p = resolve_stage_placement(other, none);
        CHECK(!p.lm_on_gpu);
        CHECK(!p.detok_on_gpu);
        CHECK(p.enc_on_gpu);  // only the LM and the detokenizer are allowlisted
    }

    // -- env overrides: applied after the allowlist ---------------------------
    // GPU hatch lifts a non-allowlisted backend (this is how a new backend gets
    // measured without a rebuild).
    {
        PlacementOverrides ov;
        ov.lm_gpu        = true;
        StagePlacement p = resolve_stage_placement("CUDA", ov);
        CHECK(p.lm_on_gpu);
        CHECK(!p.detok_on_gpu);  // the LM hatch must not move the detokenizer
    }
    {
        PlacementOverrides ov;
        ov.detok_gpu     = true;
        StagePlacement p = resolve_stage_placement("CUDA", ov);
        CHECK(p.detok_on_gpu);
        CHECK(!p.lm_on_gpu);
    }

    // CPU hatch demotes an allowlisted backend.
    {
        PlacementOverrides ov;
        ov.lm_cpu        = true;
        StagePlacement p = resolve_stage_placement("MTL", ov);
        CHECK(!p.lm_on_gpu);
        CHECK(p.detok_on_gpu);
    }
    {
        PlacementOverrides ov;
        ov.detok_cpu     = true;
        StagePlacement p = resolve_stage_placement("Vulkan", ov);
        CHECK(!p.detok_on_gpu);
        CHECK(p.lm_on_gpu);
    }

    // Precedence: CPU wins when both hatches are set for the same stage, on an
    // allowlisted backend and on a fallback one alike.
    for (const char * name : { "MTL", "Metal", "Vulkan", "CUDA" }) {
        PlacementOverrides ov;
        ov.lm_gpu        = true;
        ov.lm_cpu        = true;
        ov.detok_gpu     = true;
        ov.detok_cpu     = true;
        StagePlacement p = resolve_stage_placement(name, ov);
        CHECK(!p.lm_on_gpu);
        CHECK(!p.detok_on_gpu);
    }

    // The encoder hatch is independent of the LM/detokenizer allowlist.
    for (const char * name : { "MTL", "Vulkan", "CUDA" }) {
        PlacementOverrides ov;
        ov.encoders_cpu  = true;
        StagePlacement p = resolve_stage_placement(name, ov);
        CHECK(!p.enc_on_gpu);
        CHECK(p.lm_on_gpu == (backend_name_is_metal(name) || backend_name_is_vulkan(name)));
    }
}

// 7b. env -> overrides -------------------------------------------------------
// Locks which variable drives which stage, and that PRESENCE is what counts:
// ACESTEP_LM_CPU=0 still forces the LM to the CPU (the getenv() semantics this
// policy inherited). Uses "0" as the "set" value so the assertion is identical
// on POSIX and Windows, where _putenv_s(k, "") removes the variable instead.
void set_env(const char * key, const char * value) {
#ifdef _WIN32
    _putenv_s(key, value ? value : "");
#else
    if (value) setenv(key, value, 1);
    else       unsetenv(key);
#endif
}

void test_placement_env() {
    using tts_cpp::acestep::placement_overrides_from_env;
    using tts_cpp::acestep::PlacementOverrides;

    const char * const vars[] = { "ACESTEP_LM_GPU",    "ACESTEP_LM_CPU", "ACESTEP_DETOK_GPU",
                                  "ACESTEP_DETOK_CPU", "ACESTEP_ENCODERS_CPU" };
    auto clear_all = [&] {
        for (const char * k : vars) set_env(k, nullptr);
    };

    // Nothing set -> no override.
    clear_all();
    {
        PlacementOverrides ov = placement_overrides_from_env();
        CHECK(!ov.lm_gpu);
        CHECK(!ov.lm_cpu);
        CHECK(!ov.detok_gpu);
        CHECK(!ov.detok_cpu);
        CHECK(!ov.encoders_cpu);
    }

    // One variable at a time -> exactly its own flag, nothing else.
    for (size_t i = 0; i < sizeof(vars) / sizeof(vars[0]); ++i) {
        clear_all();
        set_env(vars[i], "0");  // presence, not value
        PlacementOverrides ov  = placement_overrides_from_env();
        const bool         set[] = { ov.lm_gpu, ov.lm_cpu, ov.detok_gpu, ov.detok_cpu, ov.encoders_cpu };
        for (size_t j = 0; j < sizeof(set) / sizeof(set[0]); ++j) CHECK(set[j] == (i == j));
    }

    clear_all();  // leave the environment as found
}

}  // namespace

int main() {
    test_schedule();
    test_philox();
    test_fsq();
    test_sampler();
    test_vae_progress();
    test_backend_device_types();
    test_stage_placement();
    test_placement_env();

    std::fprintf(stderr, "[test-acestep-units] %d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
