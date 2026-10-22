// CPU-only, weight-free unit tests for the ACE-Step pipeline's pure logic.
//
// None of these need a GGUF fixture — they exercise the deterministic math
// that the rest of the pipeline is built on, so a refactor that drifts from
// the acestep.cpp reference breaks here (fast, on a fresh checkout under
// `ctest -L unit`) before the fixture-bound integration tests get a chance to.
//
// Coverage:
//   1. dit_build_schedule  — flow-matching time schedule (shift/steps).
//   2. dit_apply_haar_dcw  — official sampler-side low/high wavelet correction.
//   3. philox_randn        — Philox4x32-10 + Box-Muller (torch.randn parity).
//   4. fsq_decode_index    — FSQ index -> 6 normalized dims (strides 8/8/8/5/5/5).
//   5. sample_top_k_p      — top-k/top-p LM sampler (determinism + argmax).
//   6. vae_progress_pct    — VAE decode progress clamp/monotonicity.
//   6b. vae_shrink_window_core — chunked-decode window vs the backend alloc cap.
//   7. GPU device types    — discrete and integrated GPUs are selectable.
//   7b. GPU fallback reason — why a GPU request resolved to the CPU.
//   8. stage placement     — which backend the LM / detokenizer / encoders run on.
//   9. parallel_load       — weight-load row/chunk decomposition parity.
//   9b. fused load         — LM q|k|v / gate|up row blocks fail closed.
//   9c. vae measure        — metadata-only VAE yields decode sizes, no abort.
//   10. quantize policy    — acestep-quantize per-tensor type selection.
//   11. quantize roundtrip — synthetic GGUF through plan/stream/padding, read back.
//   12. bpe tokenizer      — byte-level BPE encode/decode on a hand-built vocab.

#include "backend_registry.h"
#include "parallel_load.h"
#include "audio_edit.h"
#include "cancellation_scope.h"
#include "bpe_tokenizer.h"
#include "cover_noise.h"
#include "dit_ggml.h"
#include "dit_gguf.h"
#include "engine_backends.h"
#include "detok_ggml.h"
#include "fit_measure.h"
#include "fit_pools.h"
#include "tok_ggml.h"
#include "generate_task.h"
#include "generation_conditioning.h"
#include "generation_plan.h"
#include "lm_ggml.h"
#include "loudness.h"
#include "lyrics_alignment.h"
#include "lm_pipeline.h"
#include "metadata_fsm.h"
#include "philox.h"
#include "quality_score.h"
#include "quantize_gguf.h"
#include "quantize_policy.h"
#include "qwen3_block.h"
#include "stage_dump_io.h"
#include "stage_placement.h"
#include "vae_coreml_path.h"
#include "vae_coreml_windows.h"
#include "vae_encode_windows.h"
#include "vae_ggml.h"
#include "wav_reader.h"

#include "audiogen-cpp/acestep/fit.h"
#include "audiogen-cpp/acestep/vae.h"

#include "ggml-alloc.h"
#include "gguf.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <random>
#include <sstream>
#include <utility>
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

// 2. Haar DCW ----------------------------------------------------------------
void test_haar_dcw() {
    using tts_cpp::acestep::dit_apply_haar_dcw;

    // Low-band correction moves both samples in a pair together.
    std::vector<float> x = { 2.0f, 4.0f };
    const std::vector<float> y_low = { 1.0f, 3.0f };
    dit_apply_haar_dcw(x, y_low, /*T=*/2, /*C=*/1, /*N=*/1, 0.1f, 0.0f);
    CHECK(approx(x[0], 2.1f));
    CHECK(approx(x[1], 4.1f));

    // High-band correction changes the contrast inside the temporal pair.
    x = { 2.0f, 4.0f };
    const std::vector<float> y_high = { 1.0f, 5.0f };
    dit_apply_haar_dcw(x, y_high, 2, 1, 1, 0.0f, 0.2f);
    CHECK(approx(x[0], 2.2f));
    CHECK(approx(x[1], 3.8f));

    // Odd temporal lengths use a zero-padded partner and discard it after IDWT.
    x = { 2.0f };
    const std::vector<float> y_odd = { 1.0f };
    dit_apply_haar_dcw(x, y_odd, 1, 1, 1, 0.1f, 0.2f);
    CHECK(approx(x[0], 2.15f));

    // Disabled coefficients are an exact no-op.
    x = { -3.0f, 7.0f };
    const std::vector<float> before = x;
    dit_apply_haar_dcw(x, y_low, 2, 1, 1, 0.0f, 0.0f);
    CHECK(x == before);
}

// 3. philox_randn ------------------------------------------------------------
void test_philox() {
    using tts_cpp::acestep::philox_randn;
    using tts_cpp::acestep::philox_randn_from;

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

    // Repeated draws from one generator continue at the next subsequence,
    // matching one torch.Generator whose state advances across randn calls.
    float first[7], second[9];
    philox_randn_from(1234, 0, first, 7, true);
    philox_randn_from(1234, 7, second, 9, true);
    for (int i = 0; i < 7; ++i)
      CHECK(first[i] == a[i]);
    for (int i = 0; i < 9; ++i)
      CHECK(second[i] == a[i + 7]);

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

// FSQ encode: the tokenizer-side inverse of fsq_decode_index.
void test_fsq_encode() {
    using tts_cpp::acestep::fsq_decode_index;
    using tts_cpp::acestep::fsq_encode_index;

    // Saturated projector outputs land on the level extremes.
    const float low[6]  = { -100.0f, -100.0f, -100.0f, -100.0f, -100.0f, -100.0f };
    const float high[6] = { 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f };
    CHECK(fsq_encode_index(low) == 0);
    CHECK(fsq_encode_index(high) == 8 * 8 * 8 * 5 * 5 * 5 - 1);

    // Zero raw values quantize to the middle level of every dim:
    // codes {4,4,4,2,2,2} with strides {1,8,64,512,2560,12800}.
    const float mid[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    CHECK(fsq_encode_index(mid) == 4 + 4 * 8 + 4 * 64 + 2 * 512 + 2 * 2560 + 2 * 12800);

    // Roundtrip through the decoder: interior indices survive
    // decode -> atanh -> encode exactly (the extremes would need atanh(1)).
    for (int idx = 0; idx < 64000; idx += 1237) {
        float decoded[6];
        fsq_decode_index(idx, decoded);
        bool interior = true;
        for (float v : decoded) {
            if (v <= -0.999f || v >= 0.999f) interior = false;
        }
        if (!interior) continue;
        float raw[6];
        for (int d = 0; d < 6; ++d) raw[d] = std::atanh(decoded[d]);
        CHECK(fsq_encode_index(raw) == idx);
    }
}

tts_cpp::acestep::BpeTokenizer make_test_bpe_tokenizer();

// Understand ("listener") prompt framing.
void test_understand_prompt() {
    using tts_cpp::acestep::AUDIO_CODE_BASE;
    using tts_cpp::acestep::lm_understand_prompt;
    using tts_cpp::acestep::lm_understand_unconditional_prompt;
    using tts_cpp::acestep::TOKEN_IM_END;
    using tts_cpp::acestep::TOKEN_IM_START;

    const tts_cpp::acestep::BpeTokenizer tok = make_test_bpe_tokenizer();
    const int codes[3] = { 7, 0, 63999 };

    const std::vector<int> prompt = lm_understand_prompt(tok, codes, 3);
    CHECK(prompt.front() == TOKEN_IM_START);
    CHECK(std::count(prompt.begin(), prompt.end(), TOKEN_IM_START) == 3);
    CHECK(std::count(prompt.begin(), prompt.end(), TOKEN_IM_END) == 2);
    const auto first_code = std::find(prompt.begin(), prompt.end(), AUDIO_CODE_BASE + 7);
    CHECK(first_code != prompt.end());
    CHECK(*(first_code + 1) == AUDIO_CODE_BASE + 0);
    CHECK(*(first_code + 2) == AUDIO_CODE_BASE + 63999);
    CHECK(*(first_code + 3) == TOKEN_IM_END);

    const std::vector<int> unconditional = lm_understand_unconditional_prompt(tok);
    CHECK(unconditional.front() == TOKEN_IM_START);
    CHECK(std::count(unconditional.begin(), unconditional.end(), TOKEN_IM_START) == 3);
    for (int id : unconditional) CHECK(id < AUDIO_CODE_BASE);

    // Identical framing up to the user turn's content.
    CHECK(std::equal(prompt.begin(), first_code, unconditional.begin()));
}

// Understand decode budget: capped by the KV room left after the prompt, so a
// long clip's code prompt (the historical 4096 default vs the LM's 2048
// max_seq) can never overflow the context mid-decode.
void test_understand_token_budget() {
    using tts_cpp::acestep::lm_understand_token_budget;

    CHECK(lm_understand_token_budget(2048, 100, 0) == 1948);
    CHECK(lm_understand_token_budget(2048, 1220, 0) == 828);
    CHECK(lm_understand_token_budget(2048, 2047, 0) == 1);
    CHECK(lm_understand_token_budget(2048, 2048, 0) == 0);
    CHECK(lm_understand_token_budget(2048, 3000, 0) == 0);
    CHECK(lm_understand_token_budget(2048, 100, 4096) == 1948);
    CHECK(lm_understand_token_budget(2048, 100, 500) == 500);
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

// Compact-slice sampling must match the historical full-vocab path exactly:
// entries below the EOS cut were only ever masked to -1e9, so sampling the
// [eos, V) slice via index_base is an identity, RNG stream included.
void test_sampler_compact_equivalence() {
    using tts_cpp::acestep::sample_top_k_p;

    const int V = 1000, eos = 100, base = 124;  // mirrors TOKEN_IM_END/AUDIO_CODE_BASE layout
    std::mt19937                          gen(1234);
    std::uniform_real_distribution<float> ud(-4.0f, 4.0f);

    for (int trial = 0; trial < 20; ++trial) {
        std::vector<float> raw(V);
        for (float & v : raw) v = ud(gen);

        std::vector<float> full(raw);
        for (int v = 0; v < base; ++v)
            if (v != eos) full[v] = -1e9f;
        std::vector<float> slice(raw.begin() + eos, raw.end());
        for (int v = 1; v < base - eos; ++v) slice[v] = -1e9f;

        std::mt19937 r_full(trial), r_slice(trial);
        const float  temp  = (trial % 5 == 0) ? 0.0f : 0.85f;  // argmax path every 5th trial
        const float  top_p = (trial % 7 == 0) ? 0.0f : 0.9f;   // also cover the disabled-top_p path
        int tf = sample_top_k_p(full.data(), V, temp, top_p, 0, r_full);
        int ts = sample_top_k_p(slice.data(), V - eos, temp, top_p, 0, r_slice, eos);
        CHECK(tf == ts);
        CHECK(r_full == r_slice);

        // Prefix form (phase-1 FSM shape): tail [base2, V) masked -> sampling
        // only [0, base2) is the same identity.
        const int          base2 = 900;
        std::vector<float> full2(raw);
        for (int v = base2; v < V; ++v) full2[v] = -1e9f;
        std::vector<float> pre(raw.begin(), raw.begin() + base2);
        std::mt19937       r_f2(trial + 100), r_p2(trial + 100);
        int tf2 = sample_top_k_p(full2.data(), V, temp, top_p, 0, r_f2);
        int tp2 = sample_top_k_p(pre.data(), base2, temp, top_p, 0, r_p2);
        CHECK(tf2 == tp2);
        CHECK(r_f2 == r_p2);
    }
}

// r==0 edge of the compact form. dist(0, sum) yields exactly 0.0f only when
// the raw 32-bit mt19937 output is 0 (sum >= 1.0, so no underflow path); an
// all-zero engine state produces that draw deterministically. The historical
// full-vector walk then lands on absolute index 0, and the index_base form
// must land on the same spot - NOT on index_base + 0 (= EOS in phase 2).
std::mt19937 make_zero_draw_rng() {
    std::stringstream ss;
    for (int i = 0; i < 624; ++i) ss << 0 << ' ';
    ss << 0;
    std::mt19937 rng;
    ss >> rng;
    return rng;
}

void test_sampler_r0_edge() {
    using tts_cpp::acestep::sample_top_k_p;

    {  // Portability guard: stream format / canonical mapping are stdlib-specific.
        std::mt19937                          probe = make_zero_draw_rng();
        std::uniform_real_distribution<float> d01(0.0f, 1.0f);
        if (d01(probe) != 0.0f) {
            std::fprintf(stderr, "[test-acestep-units] r==0 rng state not reproducible here, subtest skipped\n");
            return;
        }
    }

    const int                             V = 1000, eos = 100, base = 124;
    std::mt19937                          gen(4321);
    std::uniform_real_distribution<float> ud(-4.0f, 4.0f);
    std::vector<float>                    raw(V);
    for (float & v : raw) v = ud(gen);

    for (float top_p : { 0.9f, 0.0f }) {  // top_p and disabled-top_p branches
        std::vector<float> full(raw);
        for (int v = 0; v < base; ++v)
            if (v != eos) full[v] = -1e9f;
        std::vector<float> slice(raw.begin() + eos, raw.end());
        for (int v = 1; v < base - eos; ++v) slice[v] = -1e9f;

        std::mt19937 r_full = make_zero_draw_rng(), r_slice = make_zero_draw_rng();
        int tf = sample_top_k_p(full.data(), V, 0.85f, top_p, 0, r_full);
        int ts = sample_top_k_p(slice.data(), V - eos, 0.85f, top_p, 0, r_slice, eos);
        CHECK(tf == 0);
        CHECK(ts == 0);
        CHECK(r_full == r_slice);
    }
}

// lm_consume_forced must be indistinguishable from masking all but one token
// and running the full sampler: same pick, same RNG consumption.
void test_sampler_forced_fast_path() {
    using tts_cpp::acestep::lm_consume_forced;
    using tts_cpp::acestep::sample_top_k_p;

    const int V = 500;
    for (int trial = 0; trial < 12; ++trial) {
        const int   live = 37 + trial * 13;
        const float temp = (trial % 4 == 0) ? 0.0f : 0.85f;

        std::vector<float> l(V, -1e9f);
        l[live] = 1.5f;
        std::mt19937 r_full(trial), r_fast(trial);
        int tf = sample_top_k_p(l.data(), V, temp, 0.9f, 0, r_full);
        int tc = lm_consume_forced(live, temp, r_fast);
        CHECK(tf == tc);
        CHECK(r_full == r_fast);
    }
}

// MetadataFSM::forced_token must agree with apply_mask: it returns exactly the
// single surviving token when one exists, -1 when the LM still has a choice.
void test_fsm_forced_token() {
    using tts_cpp::acestep::MetadataFSM;

    const int V         = 64;
    auto      survivors = [&](MetadataFSM & f) {
        std::vector<float> l(V, 1.0f);
        f.apply_mask(l.data());
        std::vector<int> alive;
        for (int v = 0; v < V; ++v)
            if (l[v] > -1e8f) alive.push_back(v);
        return alive;
    };
    auto agree = [&](MetadataFSM & f) {
        MetadataFSM probe = f;  // forced_token may seed inject_queue; probe a copy
        int         t     = probe.forced_token();
        auto        alive = survivors(f);
        if (alive.size() == 1) CHECK(t == alive[0]);
        else CHECK(t == -1);
    };

    MetadataFSM fsm;
    fsm.enabled    = true;
    fsm.vocab_size = V;

    fsm.state    = MetadataFSM::BPM_NAME;  // name tokens force one token at a time
    fsm.bpm_name = { 5, 7 };
    fsm.name_pos = 0;
    agree(fsm);
    fsm.name_pos = 1;
    agree(fsm);

    fsm.inject_queue = { 9 };  // injected value: forced
    agree(fsm);
    fsm.inject_queue.clear();

    fsm.state         = MetadataFSM::THINK_END;  // </think> is forced
    fsm.think_end_tok = 20;                      // keep it inside the synthetic vocab
    agree(fsm);

    fsm.state       = MetadataFSM::BPM_VALUE;  // value tree: forced only when 1 child
    fsm.name_pos    = (int) fsm.bpm_name.size();
    fsm.newline_tok = 3;
    fsm.bpm_tree.add({ 11, 12 });
    fsm.value_acc.clear();
    agree(fsm);  // single child {11}
    fsm.bpm_tree.add({ 13, 12 });
    agree(fsm);  // two children {11,13} -> free choice
    fsm.value_acc = { 11 };
    agree(fsm);  // single child {12}
    fsm.value_acc = { 40 };
    agree(fsm);  // unknown prefix -> forced newline
}

// Verbatim pre-fusion sampler, kept as an oracle: the fused sample_top_k_p must
// match it token-for-token and draw-for-draw.
struct RefTokenProb {
    int   id;
    float prob;
};

int reference_sample_top_k_p(float * logits, int V, float temperature, float top_p, int top_k, std::mt19937 & rng) {
    if (temperature <= 0.0f) {
        return (int) (std::max_element(logits, logits + V) - logits);
    }

    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < V; i++) logits[i] *= inv_temp;

    if (top_k > 0 && top_k < V) {
        std::vector<float> tmp(logits, logits + V);
        std::nth_element(tmp.begin(), tmp.begin() + (top_k - 1), tmp.end(), std::greater<float>());
        float threshold = tmp[top_k - 1];
        for (int i = 0; i < V; i++) {
            if (logits[i] < threshold) logits[i] = -INFINITY;
        }
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        float max_logit = -INFINITY;
        for (int i = 0; i < V; i++) {
            if (logits[i] > max_logit) max_logit = logits[i];
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < V; i++) sum_exp += expf(logits[i] - max_logit);
        float inv_sum = 1.0f / sum_exp;

        float                     cutoff = max_logit - 16.0f;
        std::vector<RefTokenProb> sorted;
        for (int i = 0; i < V; i++) {
            if (logits[i] >= cutoff) {
                sorted.push_back({ i, expf(logits[i] - max_logit) * inv_sum });
            } else {
                logits[i] = -INFINITY;
            }
        }

        int K = (int) sorted.size();
        if (K > 0) {
            std::sort(sorted.begin(), sorted.end(),
                      [](const RefTokenProb & a, const RefTokenProb & b) { return a.prob > b.prob; });
            float cum = 0.0f;
            for (int i = 0; i < K; i++) {
                if (i > 0 && cum >= top_p) logits[sorted[i].id] = -INFINITY;
                cum += sorted[i].prob;
            }
        }
    }

    float max_val = -INFINITY;
    for (int i = 0; i < V; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }

    std::uniform_real_distribution<float> dist(0.0f, sum);
    float                                 r   = dist(rng);
    float                                 acc = 0.0f;
    for (int i = 0; i < V; i++) {
        acc += logits[i];
        if (acc >= r) return i;
    }
    return 0;
}

void test_sampler_fusion_oracle() {
    using tts_cpp::acestep::sample_top_k_p;

    const int                             V = 4096;
    std::mt19937                          gen(77);
    std::uniform_real_distribution<float> ud(-6.0f, 6.0f);

    for (int trial = 0; trial < 30; ++trial) {
        std::vector<float> raw(V);
        for (float & v : raw) v = ud(gen);
        // Adversarial shapes: -1e9 masses, -inf islands, exact ties.
        if (trial % 3 == 0)
            for (int i = 0; i < V; i += 2) raw[i] = -1e9f;
        if (trial % 4 == 0)
            for (int i = 0; i < V / 8; ++i) raw[i] = raw[V - 1 - i];
        if (trial % 5 == 0)
            for (int i = 100; i < 200; ++i) raw[i] = -INFINITY;
        const float temp  = (trial % 6 == 0) ? 0.0f : 0.85f;
        const float top_p = (trial % 7 == 0) ? 0.0f : 0.9f;  // also cover the disabled-top_p path
        const int   top_k = (trial % 8 == 0) ? 40 : 0;

        std::vector<float> a = raw, b = raw;
        std::mt19937       r1(trial), r2(trial);
        int ta = reference_sample_top_k_p(a.data(), V, temp, top_p, top_k, r1);
        int tb = sample_top_k_p(b.data(), V, temp, top_p, top_k, r2);
        CHECK(ta == tb);
        CHECK(r1 == r2);
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

// 5b. Core ML sidecar path ----------------------------------------------------
void test_coreml_sidecar_path() {
    using tts_cpp::acestep::coreml_vae_sidecar_path;

    CHECK(coreml_vae_sidecar_path("models/vae-BF16.gguf") == "models/vae-decoder.mlmodelc");
    CHECK(coreml_vae_sidecar_path("vae.f16.gguf") == "vae-decoder.mlmodelc");
    CHECK(coreml_vae_sidecar_path("models/vae.gguf") == "models/vae-decoder.mlmodelc");
    CHECK(coreml_vae_sidecar_path("C:\\m\\vae-BF16.gguf") == "C:\\m\\vae-decoder.mlmodelc");
    CHECK(coreml_vae_sidecar_path("my-vae.gguf") == "my-vae-decoder.mlmodelc");
}

// 5c. Core ML decode windows --------------------------------------------------
// The sidecar's input shape is fixed, so every planned window must be exactly
// window_frames long (the last one end-aligned), with the kept cores disjoint
// and covering the latent exactly.
void test_coreml_plan_windows() {
    using tts_cpp::acestep::vae_coreml_plan_windows;
    using tts_cpp::acestep::VaeCoremlWindow;

    using tts_cpp::acestep::VAE_COREML_OVERLAP;
    CHECK(VAE_COREML_OVERLAP == 8);   // measured bound, see vae_coreml_windows.h

    CHECK(vae_coreml_plan_windows(300, 352, 48).empty());   // shorter than one window
    CHECK(vae_coreml_plan_windows(352, 96, 48).empty());    // no core left after overlap

    const std::vector<VaeCoremlWindow> tiny = vae_coreml_plan_windows(740, 64, VAE_COREML_OVERLAP);
    CHECK(!tiny.empty());
    CHECK(tiny.front().core_a == 0 && tiny.back().core_b == 740);

    const std::vector<VaeCoremlWindow> single = vae_coreml_plan_windows(352, 352, 48);
    CHECK(single.size() == 1);
    CHECK(single[0].win_a == 0 && single[0].core_a == 0 && single[0].core_b == 352);

    for (int T : {353, 740, 1024, 4096}) {
        const std::vector<VaeCoremlWindow> plan = vae_coreml_plan_windows(T, 352, 48);
        CHECK(!plan.empty());
        CHECK(plan.front().core_a == 0);
        CHECK(plan.back().core_b == T);
        for (size_t i = 0; i < plan.size(); ++i) {
            const VaeCoremlWindow & w = plan[i];
            CHECK(w.win_a >= 0 && w.win_a + 352 <= T);              // window inside the latent
            CHECK(w.core_a >= w.win_a && w.core_b <= w.win_a + 352);  // core inside the window
            CHECK(w.core_a < w.core_b);
            if (i > 0) CHECK(w.core_a == plan[i - 1].core_b);       // disjoint, gapless cores
            if (w.core_a > 0) CHECK(w.core_a - w.win_a >= 48);      // left context except at 0
            if (w.core_b < T) CHECK(w.win_a + 352 - w.core_b >= 48);  // right context except at T
        }
    }
}

// 5d. VAE window sizing ------------------------------------------------------
// The decoder's im2col node grows linearly with the window and cannot be split
// across allocations, so a backend that caps allocation size caps the window.
// Adreno 740 reports 1024 MB and the 256+2*48 window needs 1155 MiB, which is
// what aborted a 30 s GPU decode; these are those measured numbers.
void test_vae_window_core() {
    using tts_cpp::acestep::vae_shrink_window_core;

    const int    core = 256, ov = 48, core_min = 64;
    const size_t adreno_cap = (size_t) 1024 * 1024 * 1024;
    const size_t peak_352   = (size_t) 1155 * 1024 * 1024;  // measured at 256 + 2*48 frames

    // A backend that does not cap allocations (CPU reports SIZE_MAX) keeps 256,
    // so CPU / Metal / iOS behaviour is untouched.
    CHECK(vae_shrink_window_core(core, ov, peak_352, SIZE_MAX, core_min) == core);
    CHECK(vae_shrink_window_core(core, ov, (size_t) 900 * 1024 * 1024, adreno_cap, core_min) == core);

    // Adreno: must shrink, and the result must actually fit once rescaled.
    const int fitted = vae_shrink_window_core(core, ov, peak_352, adreno_cap, core_min);
    CHECK(fitted < core);
    CHECK(fitted >= core_min);
    const double bytes_per_frame = (double) peak_352 / (double) (core + 2 * ov);
    CHECK((size_t) (bytes_per_frame * (fitted + 2 * ov)) <= adreno_cap);

    // Never below the floor, however small the cap.
    CHECK(vae_shrink_window_core(core, ov, peak_352, 1024 * 1024, core_min) == core_min);
    CHECK(vae_shrink_window_core(core_min, ov, peak_352, 1024 * 1024, core_min) == core_min);

    // Converges: re-applying with the rescaled peak is a fixed point.
    const size_t peak_fitted = (size_t) (bytes_per_frame * (fitted + 2 * ov));
    CHECK(vae_shrink_window_core(fitted, ov, peak_fitted, adreno_cap, core_min) == fitted);

    // Always makes progress while it does not fit, so the caller's loop ends.
    int c = core;
    for (int i = 0; i < 16 && c > core_min; ++i) {
        const int next = vae_shrink_window_core(c, ov, peak_352, adreno_cap, core_min);
        if (next == c) break;
        CHECK(next < c);
        c = next;
    }
}

// 6. GPU device types --------------------------------------------------------
// Vulkan classifies UMA adapters such as Android Mali as IGPU. AceStep must
// accept both device classes while rejecting CPU and non-GPU accelerators.
void test_backend_device_types() {
    using tts_cpp::acestep::backend_device_type_is_gpu;
    using tts_cpp::acestep::backend_reg_name_is_validated_gpu;
    using tts_cpp::acestep::parse_adreno_version;

    CHECK(backend_device_type_is_gpu(GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_device_type_is_gpu(GGML_BACKEND_DEVICE_TYPE_IGPU));
    CHECK(!backend_device_type_is_gpu(GGML_BACKEND_DEVICE_TYPE_CPU));
    CHECK(!backend_device_type_is_gpu(GGML_BACKEND_DEVICE_TYPE_ACCEL));

    CHECK(backend_reg_name_is_validated_gpu("Vulkan"));
    CHECK(backend_reg_name_is_validated_gpu("MTL"));
    CHECK(backend_reg_name_is_validated_gpu("Metal"));
    CHECK(backend_reg_name_is_validated_gpu("CUDA"));
    // OpenCL is deliberately absent: it is reached by its own Adreno pass in
    // backend_gpu_init(), not by this validated-backend preference.
    CHECK(!backend_reg_name_is_validated_gpu("OpenCL"));
    // The HIP/MUSA builds of ggml-cuda register under their own names and stay
    // unvalidated until measured.
    CHECK(!backend_reg_name_is_validated_gpu("ROCm"));
    CHECK(!backend_reg_name_is_validated_gpu("MUSA"));
    CHECK(!backend_reg_name_is_validated_gpu(nullptr));

    // That Adreno pass gates on the generation parsed out of the device
    // name/description, so the parse is what decides OpenCL-over-Vulkan.
    CHECK(parse_adreno_version("Adreno (TM) 740") == 740);
    CHECK(parse_adreno_version("QUALCOMM Adreno(TM) 830") == 830);
    CHECK(parse_adreno_version("Adreno X1-85") == 800);       // Snapdragon-X naming
    CHECK(parse_adreno_version("Adreno (TM) 640") == 640);    // parsed, but below the 700 gate
    // An embedded "OpenCL 3.0" must not be mistaken for the model number, and a
    // non-Adreno device must not be claimed at all.
    CHECK(parse_adreno_version("Adreno (TM) 750 (OpenCL 3.0)") == 750);
    CHECK(parse_adreno_version("Mali-G715") == -1);
    CHECK(parse_adreno_version("Apple M1 Pro") == -1);
    CHECK(parse_adreno_version("") == -1);
    CHECK(parse_adreno_version(nullptr) == -1);
}

// 6b. GPU fallback reason -----------------------------------------------------
// A CPU run after a GPU request is the symptom users report, so the reason must
// never disagree with what the registry actually returned.
void test_gpu_fallback_reason() {
    using tts_cpp::GpuFallbackReason;
    using tts_cpp::gpu_fallback_reason_name;

    CHECK(std::strcmp(gpu_fallback_reason_name(GpuFallbackReason::none), "none") == 0);
    CHECK(std::strcmp(gpu_fallback_reason_name(GpuFallbackReason::not_requested), "not-requested") == 0);
    CHECK(std::strcmp(gpu_fallback_reason_name(GpuFallbackReason::no_devices), "no-devices") == 0);
    CHECK(std::strcmp(gpu_fallback_reason_name(GpuFallbackReason::init_failed), "init-failed") == 0);

    // Holds on any machine: with a GPU the reason is `none`, without one it
    // names which half of the acquisition failed.
    GpuFallbackReason reason  = GpuFallbackReason::not_requested;
    ggml_backend_t    backend = tts_cpp::acestep::backend_gpu_init(&reason);
    if (backend) {
        CHECK(reason == GpuFallbackReason::none);
        ggml_backend_free(backend);
    } else {
        CHECK(reason == GpuFallbackReason::no_devices || reason == GpuFallbackReason::init_failed);
    }
}

// 6c. GPU tier policy --------------------------------------------------------
// Speech-engine GPU selection prefers CUDA over Vulkan on the same NVIDIA
// card, and prefers discrete over integrated adapters. gpu_tier_for is the
// pure ranking that captures that policy so a synthesised device topology can
// be scored without a live ggml-backend registry.
//
// SCOPE: this test covers the classifier (what tier a device lands in) and
// the enum-constant ordering (which tier outranks which). It does NOT
// exercise backend_gpu_init's actual walk in backend_registry.h: that
// function has its own parallel Adreno-OpenCL / CUDA-first / {require_validated,
// {GPU, IGPU}} sequence and does not call gpu_tier_for today, so a reordering
// of that walk would silently pass this test. Keeping the two in sync is a
// code-review contract; the deeper fix (drive backend_gpu_init off
// gpu_tier_for as the single source of truth) is tracked separately.
void test_gpu_tier_policy() {
    using tts_cpp::acestep::GpuTier;
    using tts_cpp::acestep::gpu_tier_for;

    // Adreno 700+ OpenCL wins outright — the OpenCL kernels are the validated
    // path on Snapdragon 8 Gen 2+.
    CHECK(gpu_tier_for("OpenCL", GGML_BACKEND_DEVICE_TYPE_GPU,  740) == GpuTier::AdrenoOpenCL700Plus);
    CHECK(gpu_tier_for("OpenCL", GGML_BACKEND_DEVICE_TYPE_IGPU, 740) == GpuTier::AdrenoOpenCL700Plus);
    // Adreno 6xx OpenCL is not routed to the OpenCL tier here (broken kernels)
    // — the calling code separately skips it; the ranking treats it as generic OpenCL.
    CHECK(gpu_tier_for("OpenCL", GGML_BACKEND_DEVICE_TYPE_GPU,  640) != GpuTier::AdrenoOpenCL700Plus);

    // CUDA outranks Vulkan on the same NVIDIA card.
    CHECK(gpu_tier_for("CUDA",   GGML_BACKEND_DEVICE_TYPE_GPU,  -1)  == GpuTier::CudaDiscrete);
    CHECK(gpu_tier_for("Vulkan", GGML_BACKEND_DEVICE_TYPE_GPU,  -1)  == GpuTier::ValidatedDiscrete);
    CHECK(static_cast<int>(GpuTier::CudaDiscrete) <
          static_cast<int>(GpuTier::ValidatedDiscrete));

    // Tegra / Jetson CUDA reports IGPU on some drivers — still preferred over
    // a validated Vulkan discrete via the CUDA-first rule.
    CHECK(gpu_tier_for("CUDA", GGML_BACKEND_DEVICE_TYPE_IGPU, -1) == GpuTier::CudaIntegrated);
    CHECK(static_cast<int>(GpuTier::CudaIntegrated) <
          static_cast<int>(GpuTier::ValidatedDiscrete));

    // Discrete outranks integrated at every tier that has both — the
    // other-addons behavior we are matching (llm/diffusion prefer dGPU).
    CHECK(static_cast<int>(GpuTier::CudaDiscrete)      < static_cast<int>(GpuTier::CudaIntegrated));
    CHECK(static_cast<int>(GpuTier::ValidatedDiscrete) < static_cast<int>(GpuTier::ValidatedIntegrated));
    CHECK(static_cast<int>(GpuTier::OtherDiscrete)     < static_cast<int>(GpuTier::OtherIntegrated));

    // Metal is validated; MTL is the new registry name for the same backend.
    CHECK(gpu_tier_for("Metal", GGML_BACKEND_DEVICE_TYPE_GPU,  -1) == GpuTier::ValidatedDiscrete);
    CHECK(gpu_tier_for("MTL",   GGML_BACKEND_DEVICE_TYPE_IGPU, -1) == GpuTier::ValidatedIntegrated);

    // Non-GPU device types are not selectable — the ranking rejects them so
    // the CPU/Accel devices never leak into the GPU pass.
    CHECK(gpu_tier_for("Vulkan", GGML_BACKEND_DEVICE_TYPE_CPU,   -1) == GpuTier::NotSelectable);
    CHECK(gpu_tier_for("BLAS",   GGML_BACKEND_DEVICE_TYPE_ACCEL, -1) == GpuTier::NotSelectable);
    CHECK(gpu_tier_for(nullptr,  GGML_BACKEND_DEVICE_TYPE_CPU,   -1) == GpuTier::NotSelectable);

    // A fake "ROCm" / "MUSA" registry lands in the Other tier — unvalidated
    // but still a candidate below every validated GPU, matching the current
    // {require_validated, ...} nest in backend_gpu_init.
    CHECK(gpu_tier_for("ROCm", GGML_BACKEND_DEVICE_TYPE_GPU,  -1) == GpuTier::OtherDiscrete);
    CHECK(gpu_tier_for("MUSA", GGML_BACKEND_DEVICE_TYPE_IGPU, -1) == GpuTier::OtherIntegrated);
}

// 6d. explicit backend requests ----------------------------------------------
void test_backend_request_matching() {
    using tts_cpp::acestep::backend_request_is_auto;
    using tts_cpp::acestep::backend_request_matches;
    using tts_cpp::acestep::GpuTier;
    using tts_cpp::acestep::gpu_tier_for;

    CHECK(backend_request_is_auto(""));
    CHECK(backend_request_is_auto("auto"));
    CHECK(!backend_request_is_auto("cpu"));
    CHECK(!backend_request_is_auto("hexagon"));

    CHECK(backend_request_matches("hexagon", "HTP", "HTP0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_request_matches("hexagon", "HTP", "HTP1", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_request_matches("hexagon", "OpenCL", "GPUOpenCL", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_request_matches("HTP1", "HTP", "HTP1", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_request_matches("opencl", "OpenCL", "GPUOpenCL", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_request_matches("opencl", "HTP", "HTP0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_request_matches("webgpu", "WebGPU", "WebGPU", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_request_matches("webgpu", "Vulkan", "Vulkan0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_request_matches("opencl", "WebGPU", "WebGPU", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_request_matches("cpu", "CPU", "CPU", GGML_BACKEND_DEVICE_TYPE_CPU));
    CHECK(!backend_request_matches("cpu", "HTP", "HTP0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_request_matches("auto", "HTP", "HTP0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_request_matches("Vulkan0", nullptr, nullptr, GGML_BACKEND_DEVICE_TYPE_GPU));

    CHECK(gpu_tier_for("HTP", GGML_BACKEND_DEVICE_TYPE_GPU, -1) == GpuTier::NotSelectable);
    CHECK(gpu_tier_for("HTP", GGML_BACKEND_DEVICE_TYPE_IGPU, -1) == GpuTier::NotSelectable);
}

void test_dsp_library_path() {
    using tts_cpp::acestep::prepend_dsp_library_directory;

    CHECK(prepend_dsp_library_directory("/app/lib", "") == "/app/lib");
    CHECK(prepend_dsp_library_directory("/app/lib", "/vendor/dsp") == "/app/lib;/vendor/dsp");
    CHECK(prepend_dsp_library_directory("/app/lib", "/app/lib;/vendor/dsp") == "/app/lib;/vendor/dsp");
    CHECK(prepend_dsp_library_directory("/app/lib", "/vendor/dsp;/app/lib") == "/vendor/dsp;/app/lib");
    CHECK(prepend_dsp_library_directory("", "/vendor/dsp") == "/vendor/dsp");
}

void test_backend_requested_init() {
    using tts_cpp::GpuFallbackReason;
    using tts_cpp::acestep::backend_is_cpu_device;
    using tts_cpp::acestep::backend_requested_init;

    GpuFallbackReason reason = GpuFallbackReason::not_requested;
    ggml_backend_t    cpu    = backend_requested_init("cpu", &reason);
    CHECK(cpu != nullptr);
    CHECK(backend_is_cpu_device(cpu));
    CHECK(reason == GpuFallbackReason::none);
    if (cpu) ggml_backend_free(cpu);

    CHECK(backend_requested_init("no-such-device", &reason) == nullptr);
    CHECK(reason == GpuFallbackReason::no_devices);
}

void test_backend_init_first_matching() {
    using tts_cpp::GpuFallbackReason;
    using tts_cpp::acestep::backend_device_matches_request;
    using tts_cpp::acestep::backend_init_first_matching;
    using tts_cpp::acestep::backend_request_outcome;
    using tts_cpp::acestep::RequestedBackendInit;

    const RequestedBackendInit cpu = backend_init_first_matching("cpu");
    CHECK(cpu.backend != nullptr);
    CHECK(cpu.saw_device);
    CHECK(backend_request_outcome(cpu) == GpuFallbackReason::none);
    if (cpu.backend) {
        CHECK(backend_device_matches_request("cpu", ggml_backend_get_device(cpu.backend)));
        CHECK(!backend_device_matches_request("hexagon", ggml_backend_get_device(cpu.backend)));
        ggml_backend_free(cpu.backend);
    }
    CHECK(!backend_device_matches_request("cpu", nullptr));

    const RequestedBackendInit missing = backend_init_first_matching("no-such-device");
    CHECK(missing.backend == nullptr);
    CHECK(!missing.saw_device);
    CHECK(backend_request_outcome(missing) == GpuFallbackReason::no_devices);

    RequestedBackendInit refused;
    refused.saw_device = true;
    CHECK(backend_request_outcome(refused) == GpuFallbackReason::init_failed);
}

void test_weight_placement_on_cpu() {
    using tts_cpp::acestep::backend_requested_init;
    using tts_cpp::acestep::dit_gguf_backend_maps_weights;
    using tts_cpp::acestep::dit_gguf_weight_buffer_type;

    ggml_backend_t cpu = backend_requested_init("cpu");
    CHECK(cpu != nullptr);
    if (!cpu) return;
    CHECK(dit_gguf_backend_maps_weights(cpu));
    CHECK(dit_gguf_weight_buffer_type(cpu) == ggml_backend_get_default_buffer_type(cpu));
    ggml_backend_free(cpu);
}

void test_lm_backend_request() {
    using tts_cpp::acestep::AcestepBackends;
    using tts_cpp::acestep::BackendRequest;
    using tts_cpp::acestep::free_acestep_backends;
    using tts_cpp::acestep::resolve_acestep_backends;

    BackendRequest req;
    req.backend    = "cpu";
    req.lm_backend = "cpu";
    AcestepBackends rb;
    CHECK(resolve_acestep_backends(req, rb));
    CHECK(rb.lm == rb.backend);
    CHECK(rb.lm_extra == nullptr);
    free_acestep_backends(rb);

    req.lm_backend = "no-such-device";
    CHECK(!resolve_acestep_backends(req, rb));
    CHECK(rb.backend == nullptr);
}

constexpr uint64_t FIT_TEST_GIB    = 1ull << 30;
constexpr uint64_t FIT_TEST_MARGIN = 256ull << 20;

ggml_backend_dev_t fit_test_device(int & tag) {
    return reinterpret_cast<ggml_backend_dev_t>(&tag);
}

tts_cpp::acestep::FitPool fit_test_pool(ggml_backend_dev_t dev, bool shares_host_memory, uint64_t free_gib) {
    tts_cpp::acestep::FitPool pool;
    pool.device             = dev;
    pool.shares_host_memory = shares_host_memory;
    pool.free_bytes         = free_gib * FIT_TEST_GIB;
    pool.total_bytes        = pool.free_bytes;
    return pool;
}

void test_fit_pool_sharing() {
    using tts_cpp::acestep::backend_requested_init;
    using tts_cpp::acestep::fit_device_shares_host_memory;
    using tts_cpp::acestep::fit_pool_for_device;
    using tts_cpp::acestep::FitPool;

    CHECK(fit_device_shares_host_memory(GGML_BACKEND_DEVICE_TYPE_CPU, "CPU", 32 * FIT_TEST_GIB));
    CHECK(fit_device_shares_host_memory(GGML_BACKEND_DEVICE_TYPE_IGPU, "Vulkan", 8 * FIT_TEST_GIB));
    CHECK(fit_device_shares_host_memory(GGML_BACKEND_DEVICE_TYPE_GPU, "MTL", 24 * FIT_TEST_GIB));
    CHECK(fit_device_shares_host_memory(GGML_BACKEND_DEVICE_TYPE_GPU, "HTP", 0));
    CHECK(!fit_device_shares_host_memory(GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA", 8 * FIT_TEST_GIB));
    CHECK(!fit_device_shares_host_memory(GGML_BACKEND_DEVICE_TYPE_GPU, "OpenCL", 4 * FIT_TEST_GIB));

    ggml_backend_t cpu = backend_requested_init("cpu");
    CHECK(cpu != nullptr);
    if (!cpu) return;
    const FitPool pool = fit_pool_for_device(ggml_backend_get_device(cpu));
    CHECK(pool.device == ggml_backend_get_device(cpu));
    CHECK(pool.shares_host_memory);
    CHECK(pool.total_bytes > 0);
    ggml_backend_free(cpu);
}

void test_fit_pools_charge_distinct_lm_device() {
    using tts_cpp::acestep::fit_budget_checks;
    using tts_cpp::acestep::fit_charge;
    using tts_cpp::acestep::fit_checks_headroom;
    using tts_cpp::acestep::fit_checks_hold;
    using tts_cpp::acestep::fit_pool_register;
    using tts_cpp::acestep::FIT_HOST_POOL;
    using tts_cpp::acestep::FitBudgetCheck;
    using tts_cpp::acestep::FitPool;
    using tts_cpp::acestep::PoolCharge;

    int                      host_tag = 0, lm_tag = 0, unknown_tag = 0;
    const ggml_backend_dev_t host = fit_test_device(host_tag);
    const ggml_backend_dev_t lm   = fit_test_device(lm_tag);

    std::vector<FitPool> pools;
    CHECK(fit_pool_register(pools, fit_test_pool(host, true, 16)) == FIT_HOST_POOL);
    const size_t primary = fit_pool_register(pools, fit_test_pool(host, true, 16));
    const size_t lm_pool = fit_pool_register(pools, fit_test_pool(lm, false, 1));
    CHECK(primary == FIT_HOST_POOL);
    CHECK(lm_pool == 1);
    CHECK(pools.size() == 2);

    PoolCharge peak = fit_charge(pools, host, 2 * FIT_TEST_GIB);
    peak.add(fit_charge(pools, lm, 3 * FIT_TEST_GIB));
    CHECK(peak.bytes[FIT_HOST_POOL] == 2 * FIT_TEST_GIB);
    CHECK(peak.bytes[lm_pool] == 3 * FIT_TEST_GIB);
    CHECK(!fit_checks_hold(fit_budget_checks(pools, peak, primary, FIT_TEST_MARGIN)));

    pools[lm_pool].free_bytes = 8 * FIT_TEST_GIB;
    const std::vector<FitBudgetCheck> checks = fit_budget_checks(pools, peak, primary, FIT_TEST_MARGIN);
    CHECK(checks.size() == 2);
    CHECK(fit_checks_hold(checks));
    CHECK(fit_checks_headroom(checks) == 5 * FIT_TEST_GIB - FIT_TEST_MARGIN);

    CHECK(fit_charge(pools, fit_test_device(unknown_tag), 1).bytes[FIT_HOST_POOL] == 1);
}

void test_fit_pools_shared_budget() {
    using tts_cpp::acestep::fit_budget_checks;
    using tts_cpp::acestep::fit_charge;
    using tts_cpp::acestep::fit_checks_headroom;
    using tts_cpp::acestep::fit_checks_hold;
    using tts_cpp::acestep::fit_pool_register;
    using tts_cpp::acestep::fit_shared_budget_pool;
    using tts_cpp::acestep::FIT_HOST_POOL;
    using tts_cpp::acestep::FitBudgetCheck;
    using tts_cpp::acestep::FitPool;
    using tts_cpp::acestep::PoolCharge;

    int                      host_tag = 0, npu_tag = 0, metal_tag = 0;
    const ggml_backend_dev_t host  = fit_test_device(host_tag);
    const ggml_backend_dev_t npu   = fit_test_device(npu_tag);
    const ggml_backend_dev_t metal = fit_test_device(metal_tag);

    std::vector<FitPool> npu_pools;
    fit_pool_register(npu_pools, fit_test_pool(host, true, 4));
    const size_t npu_primary = fit_pool_register(npu_pools, fit_test_pool(npu, true, 0));
    CHECK(fit_shared_budget_pool(npu_pools, npu_primary) == FIT_HOST_POOL);
    PoolCharge npu_peak = fit_charge(npu_pools, host, FIT_TEST_GIB);
    npu_peak.add(fit_charge(npu_pools, npu, 2 * FIT_TEST_GIB));
    const std::vector<FitBudgetCheck> npu_checks = fit_budget_checks(npu_pools, npu_peak, npu_primary, FIT_TEST_MARGIN);
    CHECK(npu_checks.size() == 1);
    CHECK(fit_checks_hold(npu_checks));
    CHECK(fit_checks_headroom(npu_checks) == FIT_TEST_GIB - FIT_TEST_MARGIN);
    npu_peak.add(fit_charge(npu_pools, npu, FIT_TEST_GIB));
    CHECK(!fit_checks_hold(fit_budget_checks(npu_pools, npu_peak, npu_primary, FIT_TEST_MARGIN)));

    std::vector<FitPool> metal_pools;
    fit_pool_register(metal_pools, fit_test_pool(host, true, 64));
    const size_t metal_primary = fit_pool_register(metal_pools, fit_test_pool(metal, true, 2));
    CHECK(fit_shared_budget_pool(metal_pools, metal_primary) == metal_primary);
    PoolCharge metal_peak = fit_charge(metal_pools, host, FIT_TEST_GIB);
    metal_peak.add(fit_charge(metal_pools, metal, 2 * FIT_TEST_GIB));
    CHECK(!fit_checks_hold(fit_budget_checks(metal_pools, metal_peak, metal_primary, FIT_TEST_MARGIN)));
}

void test_fit_pools_check_every_discrete_device() {
    using tts_cpp::acestep::fit_budget_checks;
    using tts_cpp::acestep::fit_charge;
    using tts_cpp::acestep::fit_checks_headroom;
    using tts_cpp::acestep::fit_checks_hold;
    using tts_cpp::acestep::fit_pool_register;
    using tts_cpp::acestep::FitBudgetCheck;
    using tts_cpp::acestep::FitPool;
    using tts_cpp::acestep::PoolCharge;

    int                      host_tag = 0, dit_tag = 0, lm_tag = 0;
    const ggml_backend_dev_t host = fit_test_device(host_tag);
    const ggml_backend_dev_t dit  = fit_test_device(dit_tag);
    const ggml_backend_dev_t lm   = fit_test_device(lm_tag);

    std::vector<FitPool> pools;
    fit_pool_register(pools, fit_test_pool(host, true, 32));
    const size_t primary = fit_pool_register(pools, fit_test_pool(dit, false, 8));
    const size_t lm_pool = fit_pool_register(pools, fit_test_pool(lm, false, 2));

    PoolCharge peak = fit_charge(pools, host, FIT_TEST_GIB);
    peak.add(fit_charge(pools, dit, 4 * FIT_TEST_GIB)).add(fit_charge(pools, lm, 3 * FIT_TEST_GIB));
    const std::vector<FitBudgetCheck> tight = fit_budget_checks(pools, peak, primary, FIT_TEST_MARGIN);
    CHECK(tight.size() == 3);
    CHECK(!fit_checks_hold(tight));

    pools[lm_pool].free_bytes = 4 * FIT_TEST_GIB;
    const std::vector<FitBudgetCheck> roomy = fit_budget_checks(pools, peak, primary, FIT_TEST_MARGIN);
    CHECK(fit_checks_hold(roomy));
    CHECK(fit_checks_headroom(roomy) == FIT_TEST_GIB - FIT_TEST_MARGIN);
}

// 7. stage placement ---------------------------------------------------------
// Which backend each stage runs on decides which numerical path the generated
// audio takes, so the policy is locked here rather than only observed on a
// device lane. Mirrors the three branches Engine::create() relies on: the
// backend allowlist, the CPU fallback for everything else, and the env
// overrides layered on top.

// A backend validated for every stage except the autoregressive LM keeps the
// LM on the CPU while the detokenizer and encoders follow the GPU.
void check_gpu_backend_keeps_lm_on_cpu(const char * name, const char * device_desc) {
    using tts_cpp::acestep::PlacementOverrides;
    using tts_cpp::acestep::resolve_stage_placement;
    using tts_cpp::acestep::StagePlacement;

    const PlacementOverrides none;
    StagePlacement p = resolve_stage_placement(name, device_desc, none);
    CHECK(!p.lm_on_gpu);
    CHECK(p.detok_on_gpu);
    CHECK(p.enc_on_gpu);
}

void test_stage_placement() {
    using tts_cpp::acestep::backend_name_is_cuda;
    using tts_cpp::acestep::backend_name_is_metal;
    using tts_cpp::acestep::backend_name_is_opencl;
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
    // ggml-opencl's REGISTRY name; its device reports as "GPUOpenCL", which is
    // not what reaches here.
    CHECK(backend_name_is_opencl("OpenCL"));
    CHECK(!backend_name_is_opencl("GPUOpenCL"));
    // ggml-cuda's REGISTRY name; the HIP/MUSA builds of the same backend
    // register as "ROCm"/"MUSA" and must not match.
    CHECK(backend_name_is_cuda("CUDA"));
    CHECK(!backend_name_is_cuda("ROCm"));
    CHECK(!backend_name_is_cuda("MUSA"));

    // The input is the REGISTRY name, which carries no device-index suffix.
    // ggml_backend_name() would hand over "MTL0" / "Vulkan0" and match nothing.
    CHECK(!backend_name_is_metal("MTL0"));
    CHECK(!backend_name_is_vulkan("Vulkan0"));
    CHECK(!backend_name_is_opencl("OpenCL0"));
    CHECK(!backend_name_is_cuda("CUDA0"));

    // Exact compare: no case folding, no substring match, and null/empty safe.
    CHECK(!backend_name_is_metal("mtl"));
    CHECK(!backend_name_is_metal("metal"));
    CHECK(!backend_name_is_metal("MTLX"));
    CHECK(!backend_name_is_vulkan("vulkan"));
    CHECK(!backend_name_is_opencl("opencl"));
    CHECK(!backend_name_is_metal("CUDA"));
    CHECK(!backend_name_is_cuda("cuda"));
    CHECK(!backend_name_is_vulkan("MTL"));
    CHECK(!backend_name_is_metal("Vulkan"));
    CHECK(!backend_name_is_opencl("Vulkan"));
    CHECK(!backend_name_is_cuda("Vulkan"));
    CHECK(!backend_name_is_metal(""));
    CHECK(!backend_name_is_vulkan(""));
    CHECK(!backend_name_is_opencl(""));
    CHECK(!backend_name_is_cuda(""));
    CHECK(!backend_name_is_metal(nullptr));
    CHECK(!backend_name_is_vulkan(nullptr));
    CHECK(!backend_name_is_opencl(nullptr));
    CHECK(!backend_name_is_cuda(nullptr));

    const PlacementOverrides none;
    const char * const radv_desc   = "Radeon 8060S Graphics (RADV GFX1151)";
    const char * const nvidia_desc = "NVIDIA GeForce RTX 3080";

    // -- device predicate: the Vulkan LM denylist is per-device ---------------
    using tts_cpp::acestep::vulkan_device_lm_blocked;
    CHECK(vulkan_device_lm_blocked("Mali-G715"));
    CHECK(vulkan_device_lm_blocked("Mali-G78"));
    CHECK(!vulkan_device_lm_blocked(radv_desc));
    CHECK(!vulkan_device_lm_blocked(nvidia_desc));
    CHECK(!vulkan_device_lm_blocked("Samsung Xclipse 920"));
    CHECK(!vulkan_device_lm_blocked("AMD Radeon RX 7900 XTX"));  // proprietary driver
    CHECK(!vulkan_device_lm_blocked("Intel(R) Arc(tm) A770 Graphics"));
    CHECK(!vulkan_device_lm_blocked(""));
    CHECK(!vulkan_device_lm_blocked(nullptr));

    // -- Hexagon: every stage runs on the NPU ---------------------------------
    using tts_cpp::acestep::backend_name_is_hexagon;
    CHECK(backend_name_is_hexagon("HTP"));
    CHECK(!backend_name_is_hexagon("HTP0"));
    CHECK(!backend_name_is_hexagon("htp"));
    CHECK(!backend_name_is_hexagon(nullptr));
    {
        PlacementOverrides ov;
        ov.lm_cpu        = true;
        StagePlacement p = resolve_stage_placement("HTP", "Hexagon", ov);
        CHECK(!p.lm_on_gpu);
        CHECK(p.detok_on_gpu);
    }

    // -- allowlist: Metal, OpenCL, CUDA, and Hexagon keep LM + detokenizer on GPU
    for (const char * allowed : { "MTL", "Metal", "OpenCL", "CUDA", "HTP" }) {
        StagePlacement p = resolve_stage_placement(allowed, "", none);
        CHECK(p.lm_on_gpu);
        CHECK(p.detok_on_gpu);
        CHECK(p.enc_on_gpu);  // encoders follow the GPU on every backend
    }

    // Vulkan runs every stage on the GPU on any device that is not denylisted.
    // RADV (Strix Halo) and NVIDIA (RTX 3080) are measured against the
    // F32-dequantized reference; the rest inherit the GPU placement.
    for (const char * device : { radv_desc, nvidia_desc, "Samsung Xclipse 920",
                                 "Intel(R) Arc(tm) A770 Graphics", "" }) {
        StagePlacement p = resolve_stage_placement("Vulkan", device, none);
        CHECK(p.lm_on_gpu);
        CHECK(p.detok_on_gpu);
        CHECK(p.enc_on_gpu);
    }

    // Mali keeps the LM on the CPU: the GPU LM collapses to repeated semantic
    // codes and truncates the song there (README "Backends").
    check_gpu_backend_keeps_lm_on_cpu("Vulkan", "Mali-G715");
    check_gpu_backend_keeps_lm_on_cpu("Vulkan", "Mali-G78");

    // -- fallback: everything else keeps the shipping CPU placement -----------
    // Unmeasured backends must not silently pick up the GPU path. "MTL0" is in
    // this list on purpose: a suffixed name is NOT the allowlisted one. "ROCm"
    // and "MUSA" are the HIP/MUSA builds of ggml-cuda: same code base, different
    // silicon, unmeasured.
    const char * const others[] = { "ROCm",     "MUSA",     "SYCL", "BLAS", "CPU",
                                    "MTL0",     "Vulkan0",  "OpenCL0", "CUDA0",
                                    "",         nullptr };
    for (const char * other : others) {
        // A RADV description must not rescue a non-Vulkan registry name either.
        StagePlacement p = resolve_stage_placement(other, radv_desc, none);
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
        StagePlacement p = resolve_stage_placement("SYCL", "", ov);
        CHECK(p.lm_on_gpu);
        CHECK(!p.detok_on_gpu);  // the LM hatch must not move the detokenizer
    }
    {
        PlacementOverrides ov;
        ov.detok_gpu     = true;
        StagePlacement p = resolve_stage_placement("SYCL", "", ov);
        CHECK(p.detok_on_gpu);
        CHECK(!p.lm_on_gpu);
    }

    // GPU hatch also lifts a non-validated Vulkan device (how a new device gets
    // measured without a rebuild).
    {
        PlacementOverrides ov;
        ov.lm_gpu        = true;
        StagePlacement p = resolve_stage_placement("Vulkan", "Mali-G715", ov);
        CHECK(p.lm_on_gpu);
    }

    // CPU hatch demotes an allowlisted backend.
    {
        PlacementOverrides ov;
        ov.lm_cpu        = true;
        StagePlacement p = resolve_stage_placement("MTL", "", ov);
        CHECK(!p.lm_on_gpu);
        CHECK(p.detok_on_gpu);
    }
    {
        PlacementOverrides ov;
        ov.lm_cpu        = true;
        StagePlacement p = resolve_stage_placement("Vulkan", radv_desc, ov);
        CHECK(!p.lm_on_gpu);  // CPU hatch demotes a validated RADV device too
        CHECK(p.detok_on_gpu);
    }
    {
        PlacementOverrides ov;
        ov.detok_cpu     = true;
        StagePlacement p = resolve_stage_placement("Vulkan", "", ov);
        CHECK(!p.detok_on_gpu);
        CHECK(p.lm_on_gpu);  // the detokenizer hatch leaves the LM where it was
    }

    // Precedence: CPU wins when both hatches are set for the same stage, on an
    // allowlisted backend and on a fallback one alike.
    for (const char * name : { "MTL", "Metal", "Vulkan", "OpenCL", "CUDA" }) {
        PlacementOverrides ov;
        ov.lm_gpu        = true;
        ov.lm_cpu        = true;
        ov.detok_gpu     = true;
        ov.detok_cpu     = true;
        StagePlacement p = resolve_stage_placement(name, radv_desc, ov);
        CHECK(!p.lm_on_gpu);
        CHECK(!p.detok_on_gpu);
    }

    // The encoder hatch is independent of the LM/detokenizer allowlist.
    for (const char * name : { "MTL", "Vulkan", "OpenCL", "CUDA" }) {
        PlacementOverrides ov;
        ov.encoders_cpu  = true;
        StagePlacement p = resolve_stage_placement(name, "", ov);
        CHECK(!p.enc_on_gpu);
        CHECK(p.lm_on_gpu == (backend_name_is_metal(name) || backend_name_is_opencl(name) ||
                              backend_name_is_cuda(name) || backend_name_is_vulkan(name)));
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

void test_vae_backend_request_env() {
    using tts_cpp::acestep::vae_backend_request_from_env;

    set_env("ACESTEP_VAE_GPU", nullptr);
    CHECK(vae_backend_request_from_env("hexagon") == "hexagon");
    CHECK(vae_backend_request_from_env("auto") == "auto");

    set_env("ACESTEP_VAE_GPU", "0");
    CHECK(vae_backend_request_from_env("hexagon") == "cpu");
    CHECK(vae_backend_request_from_env("auto") == "auto");

    set_env("ACESTEP_VAE_GPU", "1");
    CHECK(vae_backend_request_from_env("hexagon") == "hexagon");

    set_env("ACESTEP_VAE_GPU", nullptr);
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

// 9. parallel_load -----------------------------------------------------------
// Exactly-once row coverage and bit-parity with the single-threaded conversion.
void test_parallel_rows() {
    using tts_cpp::acestep::parallel_rows;
    for (int n : { 0, 1, 127, 128, 129, 255, 4096, 12345 }) {
        // Workers only touch atomics; every CHECK stays on the main thread.
        std::vector<std::atomic<int>> hits((size_t) n);
        std::atomic<bool>             bounds_ok{ true };
        parallel_rows(n, [&](int begin, int end) {
            if (begin < 0 || end > n) bounds_ok = false;
            for (int i = begin; i < end && i >= 0; i++) hits[(size_t) i].fetch_add(1);
        });
        CHECK(bounds_ok);
        bool all_once = true;
        for (const auto & h : hits) all_once = all_once && h.load() == 1;
        CHECK(all_once);
    }
}

void test_convert_f32_to_f16_rows() {
    using tts_cpp::acestep::convert_f32_to_f16_rows;
    using tts_cpp::acestep::F16_CONVERT_CHUNK;
    using tts_cpp::acestep::LOAD_SERIAL_MIN_ROWS;
    std::mt19937                          rng(123);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    // The last size crosses LOAD_SERIAL_MIN_ROWS chunks, exercising the
    // threaded branch with a ragged tail.
    const size_t threaded = (size_t) LOAD_SERIAL_MIN_ROWS * F16_CONVERT_CHUNK + 7;
    for (size_t count : { (size_t) 0, (size_t) 1, F16_CONVERT_CHUNK - 1, F16_CONVERT_CHUNK,
                          F16_CONVERT_CHUNK + 1, 2 * F16_CONVERT_CHUNK, (size_t) 65537, threaded }) {
        std::vector<float> src(count);
        for (auto & v : src) v = dist(rng);
        std::vector<ggml_fp16_t> serial(count);
        std::vector<ggml_fp16_t> chunked(count);
        if (count > 0) {
            ggml_fp32_to_fp16_row(src.data(), serial.data(), (int) count);
            convert_f32_to_f16_rows(src.data(), chunked.data(), count);
            CHECK(std::memcmp(serial.data(), chunked.data(), count * sizeof(ggml_fp16_t)) == 0);
        } else {
            convert_f32_to_f16_rows(src.data(), chunked.data(), count);
        }
    }
}

// 9b. fused-load fail-closed -------------------------------------------------
// lm_load_row_block copies one GGUF tensor into a row-concatenated fused tensor
// at a running byte offset; lm_load_layer_fused chains q|k|v and gate|up. A
// missing tensor must fail the load (false, offset untouched): skipping a block
// would byte-shift every later one into silently wrong weights. The tiny GGUF
// is written by the test itself; CPU buffers only.
// POSIX runners set TMPDIR; Windows runners set TEMP/TMP and have no /tmp, so
// a plain TMPDIR-or-/tmp fallback makes every file-writing test fail there.
std::string test_temp_dir() {
    for (const char * var : { "TMPDIR", "TEMP", "TMP" }) {
        const char * val = std::getenv(var);
        if (val && *val) {
            return val;
        }
    }
    return "/tmp";
}

std::string write_fused_test_gguf(const std::vector<std::pair<std::string, std::vector<float>>> & tensors, int ne0) {
    const std::string path = test_temp_dir() + "/qvac-acestep-fused-load-test.gguf";

    ggml_init_params ip{ 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context *   ctx = ggml_init(ip);
    gguf_context *   gc  = gguf_init_empty();
    for (const auto & [name, vals] : tensors) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, (int64_t) vals.size() / ne0);
        ggml_set_name(t, name.c_str());
        std::memcpy(t->data, vals.data(), vals.size() * sizeof(float));
        gguf_add_tensor(gc, t);
    }
    CHECK(gguf_write_to_file(gc, path.c_str(), /*only_meta=*/false));
    gguf_free(gc);
    ggml_free(ctx);
    return path;
}

void test_fused_load_fail_closed() {
    using tts_cpp::acestep::DitGGUF;
    using tts_cpp::acestep::Qwen3Layer;
    using tts_cpp::acestep::dit_gguf_close;
    using tts_cpp::acestep::dit_gguf_open;
    using tts_cpp::acestep::lm_load_layer_fused;
    using tts_cpp::acestep::lm_load_row_block;

    const int ne0 = 4;
    auto      seq = [](int n, float base) {
        std::vector<float> v((size_t) n);
        for (int i = 0; i < n; ++i) v[(size_t) i] = base + (float) i;
        return v;
    };
    const std::string p = "model.layers.0";
    const auto q = seq(8, 0), k = seq(4, 100), v = seq(4, 200), gate = seq(8, 300), up = seq(8, 400);

    const std::string path = write_fused_test_gguf(
        { { p + ".self_attn.q_proj.weight", q },
          { p + ".self_attn.k_proj.weight", k },
          { p + ".self_attn.v_proj.weight", v },
          { p + ".mlp.gate_proj.weight", gate },
          { p + ".mlp.up_proj.weight", up } },
        ne0);

    DitGGUF g;
    CHECK(dit_gguf_open(g, path));

    ggml_init_params      ip{ 8 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };
    ggml_context *        ctx = ggml_init(ip);
    ggml_tensor *         qkv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, 4);  // q(2) + k(1) + v(1) rows
    ggml_tensor *         gu  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, 4);  // gate(2) + up(2) rows
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    CHECK(buf != nullptr);

    // Happy path: all five blocks land back to back, in declaration order.
    Qwen3Layer ly{};  // norms/o/down stay null; q3_load_* skip a null dst
    CHECK(lm_load_layer_fused(g, p, ly, qkv, gu));
    std::vector<float> qkv_want(q);
    qkv_want.insert(qkv_want.end(), k.begin(), k.end());
    qkv_want.insert(qkv_want.end(), v.begin(), v.end());
    std::vector<float> gu_want(gate);
    gu_want.insert(gu_want.end(), up.begin(), up.end());
    CHECK(std::memcmp(qkv->data, qkv_want.data(), qkv_want.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(gu->data, gu_want.data(), gu_want.size() * sizeof(float)) == 0);

    // Missing tensor: false, and the offset must not advance.
    size_t off = 0;
    CHECK(lm_load_row_block(qkv, off, g, p + ".self_attn.q_proj.weight"));
    const size_t after_q = off;
    CHECK(!lm_load_row_block(qkv, off, g, p + ".self_attn.absent.weight"));
    CHECK(off == after_q);
    dit_gguf_close(g);

    // v_proj absent -> the whole fused layer load fails.
    const std::string path2 = write_fused_test_gguf(
        { { p + ".self_attn.q_proj.weight", q },
          { p + ".self_attn.k_proj.weight", k },
          { p + ".mlp.gate_proj.weight", gate },
          { p + ".mlp.up_proj.weight", up } },
        ne0);
    DitGGUF g2;
    CHECK(dit_gguf_open(g2, path2));
    CHECK(!lm_load_layer_fused(g2, p, ly, qkv, gu));
    dit_gguf_close(g2);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    std::remove(path.c_str());
}

std::string write_raw_stage_dump(const char * name, const std::vector<int32_t> & header, size_t n_floats) {
    const std::string path = test_temp_dir() + "/" + name;
    FILE *            f    = fopen(path.c_str(), "wb");
    CHECK(f != nullptr);
    if (!f) return path;
    const std::vector<float> payload(n_floats, 1.0f);
    fwrite(header.data(), sizeof(int32_t), header.size(), f);
    fwrite(payload.data(), sizeof(float), payload.size(), f);
    fclose(f);
    return path;
}

void expect_stage_dump_rejected(const char * name, const std::vector<int32_t> & header, size_t n_floats) {
    const std::string  path = write_raw_stage_dump(name, header, n_floats);
    std::vector<float> out;
    int                d0 = 0, d1 = 0;
    bool               read  = true;
    bool               threw = false;
    try {
        read = tts_cpp::acestep::read_stage_dump("test", path.c_str(), out, &d0, &d1);
    } catch (const std::exception &) {
        threw = true;
    }
    CHECK(!threw);
    CHECK(!read);
    CHECK(out.empty());
    std::remove(path.c_str());
}

void test_stage_dump_round_trip() {
    using tts_cpp::acestep::read_stage_dump;
    using tts_cpp::acestep::write_stage_dump;

    const std::string        path = test_temp_dir() + "/qvac-acestep-stage-dump.bin";
    const std::vector<float> v    = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    CHECK(write_stage_dump("test", path.c_str(), v, 3, 2));
    std::vector<float> out;
    int                d0 = 0, d1 = 0;
    CHECK(read_stage_dump("test", path.c_str(), out, &d0, &d1));
    CHECK(d0 == 3);
    CHECK(d1 == 2);
    CHECK(out == v);
    std::remove(path.c_str());
}

void test_stage_dump_rejects_malformed() {
    constexpr int32_t BIG = std::numeric_limits<int32_t>::max();
    expect_stage_dump_rejected("qvac-acestep-dump-huge.bin", { 2, BIG, BIG }, 0);
    expect_stage_dump_rejected("qvac-acestep-dump-huge-frames.bin", { 2, BIG, 64 }, 0);
    expect_stage_dump_rejected("qvac-acestep-dump-truncated.bin", { 2, 4, 64 }, 10);
    expect_stage_dump_rejected("qvac-acestep-dump-rank.bin", { 3, 2, 3 }, 6);
    expect_stage_dump_rejected("qvac-acestep-dump-zero.bin", { 2, 0, 3 }, 0);
    expect_stage_dump_rejected("qvac-acestep-dump-negative.bin", { 2, -1, 3 }, 3);
    expect_stage_dump_rejected("qvac-acestep-dump-short-header.bin", { 2, 3 }, 0);

    std::vector<float> out;
    int                d0 = 0, d1 = 0;
    const std::string  missing = test_temp_dir() + "/qvac-acestep-dump-missing.bin";
    std::remove(missing.c_str());
    CHECK(!tts_cpp::acestep::read_stage_dump("test", missing.c_str(), out, &d0, &d1));
}

constexpr int TINY_LM_HIDDEN     = 128;
constexpr int TINY_LM_FFN        = 256;
constexpr int TINY_LM_VOCAB      = 200;
constexpr int TINY_LM_MAX_SEQ    = 64;
constexpr int TINY_LM_CHUNK_ROWS = 32;
constexpr int TINY_LM_PREFIX     = 150;

float tiny_lm_weight(int64_t i, float scale) {
    return std::sin(0.37f * (float) i + scale) * scale;
}

std::vector<float> tiny_lm_values(int64_t n, float scale) {
    std::vector<float> values((size_t) n);
    for (int64_t i = 0; i < n; ++i) values[(size_t) i] = scale == 0.0f ? 1.0f : tiny_lm_weight(i, scale);
    return values;
}

void add_tiny_lm_tensor(ggml_context * ctx, gguf_context * gc, const std::string & name, ggml_type type, int64_t ne0,
                        int64_t ne1, float scale) {
    const std::vector<float> values = tiny_lm_values(ne0 * ne1, scale);
    ggml_tensor *            t      = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    ggml_set_name(t, name.c_str());
    if (type == GGML_TYPE_F32) {
        std::memcpy(t->data, values.data(), values.size() * sizeof(float));
    } else {
        ggml_quantize_chunk(type, values.data(), t->data, 0, ne1, ne0, nullptr);
    }
    gguf_add_tensor(gc, t);
}

std::string write_tiny_lm_gguf() {
    const std::string path = test_temp_dir() + "/qvac-acestep-tiny-lm-test.gguf";
    const int         H = TINY_LM_HIDDEN, F = TINY_LM_FFN, D = 128;
    const std::string l = "model.layers.0";

    ggml_init_params ip{ 16 * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context *   ctx = ggml_init(ip);
    gguf_context *   gc  = gguf_init_empty();
    add_tiny_lm_tensor(ctx, gc, "model.embed_tokens.weight", GGML_TYPE_Q8_0, H, TINY_LM_VOCAB, 0.5f);
    add_tiny_lm_tensor(ctx, gc, "model.norm.weight", GGML_TYPE_F32, H, 1, 0.0f);
    add_tiny_lm_tensor(ctx, gc, l + ".input_layernorm.weight", GGML_TYPE_F32, H, 1, 0.0f);
    add_tiny_lm_tensor(ctx, gc, l + ".post_attention_layernorm.weight", GGML_TYPE_F32, H, 1, 0.0f);
    add_tiny_lm_tensor(ctx, gc, l + ".self_attn.q_norm.weight", GGML_TYPE_F32, D, 1, 0.0f);
    add_tiny_lm_tensor(ctx, gc, l + ".self_attn.k_norm.weight", GGML_TYPE_F32, D, 1, 0.0f);
    add_tiny_lm_tensor(ctx, gc, l + ".self_attn.q_proj.weight", GGML_TYPE_Q8_0, H, D, 0.05f);
    add_tiny_lm_tensor(ctx, gc, l + ".self_attn.k_proj.weight", GGML_TYPE_Q8_0, H, D, 0.04f);
    add_tiny_lm_tensor(ctx, gc, l + ".self_attn.v_proj.weight", GGML_TYPE_Q8_0, H, D, 0.03f);
    add_tiny_lm_tensor(ctx, gc, l + ".self_attn.o_proj.weight", GGML_TYPE_Q8_0, D, H, 0.02f);
    add_tiny_lm_tensor(ctx, gc, l + ".mlp.gate_proj.weight", GGML_TYPE_Q8_0, H, F, 0.06f);
    add_tiny_lm_tensor(ctx, gc, l + ".mlp.up_proj.weight", GGML_TYPE_Q8_0, H, F, 0.07f);
    add_tiny_lm_tensor(ctx, gc, l + ".mlp.down_proj.weight", GGML_TYPE_Q8_0, F, H, 0.08f);
    CHECK(gguf_write_to_file(gc, path.c_str(), /*only_meta=*/false));
    gguf_free(gc);
    ggml_free(ctx);
    return path;
}

bool chunk_ranges_tile(const std::vector<std::pair<int, int>> & ranges, int vocab, int max_rows) {
    int next = 0;
    for (const auto & [row0, rows] : ranges) {
        if (row0 != next || rows <= 0 || rows > max_rows) return false;
        next = row0 + rows;
    }
    return next == vocab;
}

bool chunk_ranges_start_at(const std::vector<std::pair<int, int>> & ranges, int row) {
    for (const auto & range : ranges) {
        if (range.first == row) return true;
    }
    return false;
}

void test_lm_head_chunk_ranges() {
    using tts_cpp::acestep::AUDIO_CODE_BASE;
    using tts_cpp::acestep::AUDIO_CODE_COUNT;
    using tts_cpp::acestep::lm_head_chunk_ranges;
    using tts_cpp::acestep::TOKEN_IM_END;

    const int  ace_vocab = AUDIO_CODE_BASE + AUDIO_CODE_COUNT;
    const auto ace       = lm_head_chunk_ranges(ace_vocab, 32768);
    CHECK(chunk_ranges_tile(ace, ace_vocab, 32768));
    CHECK(chunk_ranges_start_at(ace, TOKEN_IM_END));
    CHECK(chunk_ranges_start_at(ace, AUDIO_CODE_BASE));

    const std::vector<std::pair<int, int>> small = { { 0, 32 }, { 32, 32 }, { 64, 32 }, { 96, 4 } };
    CHECK(lm_head_chunk_ranges(100, 32) == small);
    CHECK(lm_head_chunk_ranges(100, 100) == (std::vector<std::pair<int, int>>{ { 0, 100 } }));
}

void test_lm_split_table_matches_whole_table() {
    using tts_cpp::acestep::backend_cpu_init;
    using tts_cpp::acestep::LMModel;
    using tts_cpp::acestep::lm_model_forward;
    using tts_cpp::acestep::lm_model_free;
    using tts_cpp::acestep::lm_model_load;
    using tts_cpp::acestep::lm_model_load_split_table;

    const std::string path = write_tiny_lm_gguf();
    ggml_backend_t    cpu  = backend_cpu_init();
    CHECK(cpu != nullptr);
    LMModel * whole = cpu ? lm_model_load(path, cpu, TINY_LM_MAX_SEQ, /*verbose=*/false, /*n_kv_sets=*/1) : nullptr;
    LMModel * split = cpu ? lm_model_load_split_table(path, cpu, TINY_LM_MAX_SEQ, 1, TINY_LM_CHUNK_ROWS) : nullptr;
    CHECK(whole != nullptr);
    CHECK(split != nullptr);
    if (whole && split) {
        const int32_t      prompt[] = { 0, 3, 77, 150, TINY_LM_VOCAB - 1 };
        std::vector<float> whole_logits, split_logits;
        CHECK(lm_model_forward(whole, prompt, 5, whole_logits));
        CHECK(lm_model_forward(split, prompt, 5, split_logits));
        CHECK(whole_logits.size() == (size_t) TINY_LM_VOCAB);
        CHECK(whole_logits == split_logits);

        const int32_t next = 42;
        CHECK(lm_model_forward(whole, &next, 1, whole_logits, 0, nullptr, TINY_LM_PREFIX));
        CHECK(lm_model_forward(split, &next, 1, split_logits, 0, nullptr, TINY_LM_PREFIX));
        CHECK(split_logits.size() == (size_t) TINY_LM_PREFIX);
        CHECK(whole_logits == split_logits);

        const int32_t outside = TINY_LM_VOCAB;
        CHECK(!lm_model_forward(split, &outside, 1, split_logits));
    }
    lm_model_free(whole);
    lm_model_free(split);
    if (cpu) ggml_backend_free(cpu);
    std::remove(path.c_str());
}

// 9c. vae metadata-only measure ----------------------------------------------
// A metadata-only VAE (weights sized, never allocated) must still yield decode
// measurement sizes. The decoder architecture is hardcoded, so any parseable
// GGUF serves; the earlier sched-based pricing aborted on the buffer-less
// weights (GGML_ASSERT(buffer_id >= 0) in ggml-alloc).
void test_vae_metadata_only_measure() {
    using tts_cpp::acestep::AcestepStageMeasure;
    using tts_cpp::acestep::backend_cpu_init;
    using tts_cpp::acestep::vae_model_free;
    using tts_cpp::acestep::vae_model_load_metadata_only;
    using tts_cpp::acestep::vae_model_measure_decode;

    const std::string path = test_temp_dir() + "/qvac-acestep-vae-measure-test.gguf";
    gguf_context * gc = gguf_init_empty();
    CHECK(gguf_write_to_file(gc, path.c_str(), /*only_meta=*/true));
    gguf_free(gc);

    ggml_backend_t cpu = backend_cpu_init();
    CHECK(cpu != nullptr);
    if (cpu) {
        AcestepStageMeasure w{};
        tts_cpp::acestep::VaeModel * m =
            vae_model_load_metadata_only(path, cpu, /*with_encoder=*/false, /*verbose=*/false, w);
        CHECK(m != nullptr);
        if (m) {
            CHECK(w.weights_alloc_bytes > 0);
            size_t backend_bytes = 0, host_input_bytes = 0;
            CHECK(vae_model_measure_decode(m, /*T_latent=*/8, backend_bytes, host_input_bytes));
            CHECK(backend_bytes > 0);
            CHECK(host_input_bytes == 0);  // the input is part of the CPU measure
            vae_model_free(m);
        }
        ggml_backend_free(cpu);
    }
    std::remove(path.c_str());
}

void test_generate_task_kinds() {
    using tts_cpp::acestep::TASK_COVER;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::TASK_TEXT2MUSIC;
    using tts_cpp::acestep::is_cover_task;

    CHECK(is_cover_task(TASK_COVER));
    CHECK(is_cover_task(TASK_COVER_NOFSQ));
    CHECK(!is_cover_task(TASK_TEXT2MUSIC));
    CHECK(!is_cover_task(""));
}

void test_generate_task_defaults() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::TASK_TEXT2MUSIC;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    params.source_audio.assign(8, 0.25f);
    params.reference_audio.assign(8, 0.5f);
    const float * source_data = params.source_audio.data();
    const float * reference_data = params.reference_audio.data();
    GenerateTask task;
    CHECK(resolve_generate_task(params, task).empty());
    CHECK(task.type == TASK_TEXT2MUSIC);
    CHECK(approx(task.audio_cover_strength, 1.0f));
    CHECK(approx(task.cover_noise_strength, 0.0f));
    CHECK(params.source_audio.data() == source_data);
    CHECK(params.reference_audio.data() == reference_data);
}

void test_generate_task_audio_layout() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;

    params.task_type = TASK_COVER_NOFSQ;
    CHECK(resolve_generate_task(params, task).find("requires source_audio") != std::string::npos);

    params.source_audio.assign(3, 0.0f);
    CHECK(resolve_generate_task(params, task).find("source_audio must be interleaved stereo") != std::string::npos);

    params.source_audio.assign(4, 0.0f);
    params.reference_audio.assign(5, 0.0f);
    CHECK(resolve_generate_task(params, task).find("reference_audio must be interleaved stereo") != std::string::npos);

    params.reference_audio.assign(6, 0.0f);
    CHECK(resolve_generate_task(params, task).empty());
}

void test_generate_task_errors() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::TASK_COVER;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;

    params.task_type = "repaint";
    CHECK(resolve_generate_task(params, task).find("unsupported task_type") != std::string::npos);

    params.task_type = TASK_COVER;
    params.source_audio.assign(4, 0.0f);
    CHECK(resolve_generate_task(params, task).find("not implemented") != std::string::npos);

    params.task_type = TASK_COVER_NOFSQ;
    params.audio_cover_strength = 0.5f;
    CHECK(resolve_generate_task(params, task).empty());
    CHECK(approx(task.audio_cover_strength, 0.5f));
}

void test_simple_mode_policy() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;
    params.simple_mode = true;
    params.caption = "a romantic modern salsa with male lead vocals for a wedding";
    params.lyrics.clear();
    CHECK(resolve_generate_task(params, task).empty());

    params.lyrics = "[Instrumental]";
    CHECK(resolve_generate_task(params, task).empty());

    params.lyrics = "[verse]\nuser lyrics";
    CHECK(resolve_generate_task(params, task).find("lyrics must be empty") != std::string::npos);

    params.lyrics.clear();
    params.caption.clear();
    CHECK(resolve_generate_task(params, task).find("requires a caption") != std::string::npos);

    params.caption = "a short query";
    params.task_type = TASK_COVER_NOFSQ;
    params.source_audio.assign(4, 0.0f);
    CHECK(resolve_generate_task(params, task).find("only task 'text2music'") != std::string::npos);

    params.task_type.clear();
    params.source_audio.clear();
    params.audio_codes.assign(4, 1);
    CHECK(resolve_generate_task(params, task).find("pre-supplied audio_codes") != std::string::npos);

    params.audio_codes.clear();
    params.edit_plan.push_back(tts_cpp::acestep::RepaintParams{});
    CHECK(resolve_generate_task(params, task).find("combined with edit_plan") != std::string::npos);

    params.edit_plan.clear();
    params.lm_phase1 = false;
    CHECK(resolve_generate_task(params, task).find("requires lm_phase1") != std::string::npos);

    params.lm_phase1 = true;
    params.simple_mode = false;
    params.lyrics = "[verse]\nuser lyrics";
    CHECK(resolve_generate_task(params, task).empty());
}

void test_simple_mode_prompt_resolvers() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::RepaintParams;
    using tts_cpp::acestep::resolve_prompt_language;
    using tts_cpp::acestep::resolve_prompt_lyrics;

    GenerateParams params;
    params.lyrics.clear();
    CHECK(resolve_prompt_lyrics(params) == "[Instrumental]");
    CHECK(resolve_prompt_language(params) == "en");

    params.simple_mode = true;
    CHECK(resolve_prompt_lyrics(params).empty());
    CHECK(resolve_prompt_language(params).empty());

    params.simple_mode   = false;
    params.rewrite_query = true;
    CHECK(resolve_prompt_language(params).empty());
    params.rewrite_query = false;

    params.simple_mode = true;
    params.lyrics = "[Instrumental]";
    params.vocal_language = "es";
    CHECK(resolve_prompt_lyrics(params) == "[Instrumental]");
    CHECK(resolve_prompt_language(params) == "es");

    params.simple_mode = false;
    params.lyrics.clear();
    params.vocal_language.clear();
    params.edit_plan.push_back(RepaintParams{});
    CHECK(resolve_prompt_language(params) == "unknown");

    params.edit_plan.clear();
    params.task_type = tts_cpp::acestep::TASK_LEGO;
    CHECK(resolve_prompt_language(params) == "unknown");
}

void test_normalize_loudness() {
    using tts_cpp::acestep::normalize_loudness;

    std::vector<float> pcm = { 0.1f, -0.2f, 0.4f, 0.05f };
    normalize_loudness(pcm, 0);
    CHECK(approx(pcm[0], 0.25f));
    CHECK(approx(pcm[1], -0.5f));
    CHECK(approx(pcm[2], 1.0f));
    CHECK(approx(pcm[3], 0.125f));

    std::vector<float> clipped = { 0.1f, -0.2f, 0.4f, 0.05f };
    normalize_loudness(clipped, 10);
    CHECK(approx(clipped[0], 0.5f));
    CHECK(approx(clipped[1], -1.0f));
    CHECK(approx(clipped[2], 1.0f));
    CHECK(approx(clipped[3], 0.25f));

    std::vector<float> silence(16, 0.0f);
    normalize_loudness(silence);
    CHECK(silence == std::vector<float>(16, 0.0f));

    std::vector<float> empty;
    normalize_loudness(empty);
    CHECK(empty.empty());
}

bool near_value(double actual, double expected, double tolerance) {
    return std::fabs(actual - expected) <= tolerance;
}

void test_lyrics_matrix_conversion() {
    using tts_cpp::acestep::lyrics::Matrix;
    using tts_cpp::acestep::lyrics::matrix_from_column_major;

    const std::vector<float> ggml_values = {
        1, 5, 9,
        2, 6, 10,
        3, 7, 11,
        4, 8, 12,
    };
    const Matrix matrix = matrix_from_column_major(ggml_values.data(), 3, 4);
    CHECK(matrix.values == std::vector<float>({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }));
}

void test_lyrics_dtw() {
    using tts_cpp::acestep::lyrics::Matrix;
    using tts_cpp::acestep::lyrics::dtw;
    using tts_cpp::acestep::lyrics::DtwPath;

    const Matrix costs(3, 3, {
        0.0f, 2.0f, 2.0f,
        2.0f, 0.0f, 2.0f,
        2.0f, 2.0f, 0.0f,
    });
    DtwPath path = dtw(costs);
    CHECK(path.token_indices == std::vector<int32_t>({ 0, 1, 2 }));
    CHECK(path.frame_indices == std::vector<int32_t>({ 0, 1, 2 }));

    const Matrix rectangular(2, 5);
    path = dtw(rectangular);
    CHECK(path.token_indices.front() == 0 && path.token_indices.back() == 1);
    CHECK(path.frame_indices.front() == 0 && path.frame_indices.back() == 4);

    // Reference-parity tie-break (_dtw.py): with diagonal == vertical <
    // horizontal at the last cell, the strict comparison chain falls through
    // to horizontal, so the path detours through (1, 0) instead of stepping
    // diagonally.
    const Matrix tie(2, 2, {
        0.0f, 0.0f,
        1.0f, 0.0f,
    });
    path = dtw(tie);
    CHECK(path.token_indices == std::vector<int32_t>({ 0, 1, 1 }));
    CHECK(path.frame_indices == std::vector<int32_t>({ 0, 0, 1 }));
}

void test_lyrics_preprocessing() {
    using namespace tts_cpp::acestep::lyrics;

    const Matrix spike(1, 5, { 0.0f, 0.0f, 10.0f, 0.0f, 0.0f });
    const Matrix filtered = median_filter(spike, 3);
    CHECK(filtered.values == std::vector<float>({ 0, 0, 0, 0, 0 }));

    const Matrix head_a(2, 3, { 1, 2, 3, 4, 5, 6 });
    const Matrix head_b(2, 3, { 3, 4, 5, 6, 7, 8 });
    const auto scoring = preprocess_scoring({ head_a, head_b }, 1);
    CHECK(near_value(scoring.average_matrix(0, 0), 2.0, 1e-6));
    CHECK(near_value(scoring.energy_matrix(0, 0), 0.0, 1e-6));
    CHECK(near_value(scoring.energy_matrix(1, 2), 1.0, 1e-6));
    CHECK(near_value(scoring.calc_matrix(0, 1), 0.04, 1e-6));

    const auto alignment = preprocess_alignment({ head_a, head_b }, 2.0f, 1);
    CHECK(alignment.calc_matrix.rows == 2 && alignment.calc_matrix.cols == 3);
    CHECK(alignment.energy_matrix.rows == 2 && alignment.energy_matrix.cols == 3);
}

void test_lyrics_metrics_and_score() {
    using namespace tts_cpp::acestep::lyrics;

    CHECK(token_type_mask({ "hello", "[verse", "]", "world" }) ==
          std::vector<int32_t>({ 1, 0, 0, 1 }));

    const Matrix energy(3, 4, {
        0.8f, 0.2f, 0.0f, 0.0f,
        0.0f, 0.5f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.2f, 0.8f,
    });
    DtwPath path;
    path.token_indices = { 0, 1, 2 };
    path.frame_indices = { 0, 1, 3 };
    const AlignmentMetrics metrics =
        compute_alignment_metrics(energy, path, { 1, 0, 1 }, 0.01, 0.0, 0.5);
    CHECK(near_value(metrics.coverage, 1.0, 1e-12));
    CHECK(near_value(metrics.monotonicity, 1.0, 1e-12));
    CHECK(near_value(metrics.path_confidence, 0.74, 1e-7));
    CHECK(near_value(calculate_lyrics_score(metrics), 0.74, 1e-12));
}

void test_lyrics_timestamps_and_lrc() {
    using namespace tts_cpp::acestep::lyrics;

    const Matrix attention(2, 4, {
        5.0f, 5.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 5.0f, 5.0f,
    });
    const std::vector<TokenTimestamp> aligned =
        token_timestamps(attention, { 1, 2 }, { "Hello", "\n" }, 4.0);
    CHECK(near_value(aligned[0].start, 0.0, 1e-12));
    CHECK(near_value(aligned[0].end, 1.0, 1e-12));
    CHECK(near_value(aligned[1].start, 1.0, 1e-12));
    CHECK(near_value(aligned[1].end, 3.0, 1e-12));

    const std::vector<TokenTimestamp> tokens = {
        { 1, "Hello", 1.23456, 2.0, 0.2 },
        { 2, "\n", 2.0, 2.5, 0.4 },
        { 3, "World", 61.0, 62.0, 0.8 },
    };
    const auto decoder = [](const std::vector<int> & ids) {
        std::string text;
        for (int id : ids) {
            if (id == 1) text += "Hello";
            if (id == 2) text += "\n";
            if (id == 3) text += "World";
        }
        return text;
    };
    const std::vector<SentenceTimestamp> sentences = sentence_timestamps(tokens, decoder);
    CHECK(sentences.size() == 2);
    CHECK(sentences[0].text == "Hello" && sentences[1].text == "World");
    CHECK(near_value(sentences[0].start, 1.235, 1e-12));
    CHECK(near_value(sentences[0].confidence, 0.0, 1e-12));
    CHECK(near_value(sentences[1].confidence, 1.0, 1e-12));
    CHECK(format_lrc(sentences) == "[00:01.24]Hello\n[01:01.00]World");
    CHECK(format_lrc(sentences, true) ==
          "[00:01.24][00:02.50]Hello\n[01:01.00][01:02.00]World");
}

void test_lrc_request_policy() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::RepaintParams;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;
    params.generate_lrc = true;
    params.lyrics = "[verse]\na line to align";
    CHECK(resolve_generate_task(params, task).empty());

    params.lyrics = "[Instrumental]";
    CHECK(resolve_generate_task(params, task).find("requires lyrics") != std::string::npos);

    params.lyrics.clear();
    CHECK(resolve_generate_task(params, task).find("requires lyrics") != std::string::npos);

    params.simple_mode = true;
    params.caption = "a short query";
    CHECK(resolve_generate_task(params, task).empty());

    params.simple_mode = false;
    params.lyrics = "[verse]\na line to align";
    params.edit_plan.push_back(RepaintParams{});
    CHECK(resolve_generate_task(params, task).find("audio edit path") != std::string::npos);
}

void test_format_user_message() {
    using tts_cpp::acestep::lm_format_user_message;

    CHECK(lm_format_user_message("a salsa song", "[verse]\nhello") ==
          "# Caption\na salsa song\n\n# Lyric\n[verse]\nhello");
}

void test_rewrite_query_policy() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::RepaintParams;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask   task;
    params.rewrite_query = true;
    params.caption       = "a salsa song";
    params.lyrics        = "[verse]\nhello";
    CHECK(resolve_generate_task(params, task).empty());

    params.simple_mode = true;
    CHECK(resolve_generate_task(params, task).find("cannot be combined with simple_mode") !=
          std::string::npos);
    params.simple_mode = false;

    params.caption.clear();
    CHECK(resolve_generate_task(params, task).find("requires a caption") != std::string::npos);
    params.caption = "a salsa song";

    params.lyrics.clear();
    CHECK(resolve_generate_task(params, task).find("requires lyrics to format") != std::string::npos);

    params.lyrics = "[Instrumental]";
    CHECK(resolve_generate_task(params, task).find("requires lyric text to preserve") !=
          std::string::npos);
    params.lyrics = "[verse]\nhello";

    params.task_type = tts_cpp::acestep::TASK_COVER_NOFSQ;
    params.source_audio.assign(96, 0.0f);
    CHECK(resolve_generate_task(params, task).find("only task 'text2music'") != std::string::npos);
    params.task_type = tts_cpp::acestep::TASK_TEXT2MUSIC;
    params.source_audio.clear();

    params.audio_codes.assign(4, 1);
    CHECK(resolve_generate_task(params, task).find("pre-supplied audio_codes") != std::string::npos);
    params.audio_codes.clear();

    params.edit_plan.push_back(RepaintParams{});
    CHECK(resolve_generate_task(params, task).find("edit_plan") != std::string::npos);
    params.edit_plan.clear();

    params.lm_phase1 = false;
    CHECK(resolve_generate_task(params, task).find("requires lm_phase1") != std::string::npos);
}

void test_inspire_user_message() {
    using tts_cpp::acestep::lm_inspire_user_message;

    CHECK(lm_inspire_user_message("a short query", "") == "a short query");
    CHECK(lm_inspire_user_message("a short query", "[Instrumental]") ==
          "a short query\n\ninstrumental: true");
    CHECK(lm_inspire_user_message("a short query", "[verse]\nwords") == "a short query");
}

void test_cover_conditioning_switch() {
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::TASK_TEXT2MUSIC;
    using tts_cpp::acestep::needs_cover_conditioning_switch;
    using tts_cpp::acestep::resolve_cover_switch_step;

    GenerateTask task;
    task.type = TASK_COVER_NOFSQ;
    task.audio_cover_strength = 1.0f;
    CHECK(!needs_cover_conditioning_switch(task));
    CHECK(resolve_cover_switch_step(task, 50) == -1);

    task.audio_cover_strength = 0.5f;
    CHECK(needs_cover_conditioning_switch(task));
    CHECK(resolve_cover_switch_step(task, 50) == 25);
    CHECK(resolve_cover_switch_step(task, 8) == 4);

    task.audio_cover_strength = 0.75f;
    CHECK(resolve_cover_switch_step(task, 8) == 6);

    task.audio_cover_strength = 0.0f;
    CHECK(resolve_cover_switch_step(task, 8) == 0);

    task.type = TASK_TEXT2MUSIC;
    task.audio_cover_strength = 0.5f;
    CHECK(!needs_cover_conditioning_switch(task));
    CHECK(resolve_cover_switch_step(task, 50) == -1);
}

void test_generate_task_strengths() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;

    params.audio_cover_strength = 2.0f;
    params.cover_noise_strength = -0.5f;
    CHECK(resolve_generate_task(params, task).empty());
    CHECK(approx(task.audio_cover_strength, 1.0f));
    CHECK(approx(task.cover_noise_strength, 0.0f));

    params.audio_cover_strength = std::numeric_limits<float>::quiet_NaN();
    CHECK(resolve_generate_task(params, task).find("must be finite") != std::string::npos);

    params.audio_cover_strength = std::numeric_limits<float>::infinity();
    CHECK(resolve_generate_task(params, task).find("must be finite") != std::string::npos);

    params.audio_cover_strength = 1.0f;
    params.cover_noise_strength = std::numeric_limits<float>::quiet_NaN();
    CHECK(resolve_generate_task(params, task).find("must be finite") != std::string::npos);

    params.cover_noise_strength = std::numeric_limits<float>::infinity();
    CHECK(resolve_generate_task(params, task).find("must be finite") != std::string::npos);
}

void test_generation_plans() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::GenerationPlan;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::make_generation_plan;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;
    CHECK(resolve_generate_task(params, task).empty());

    GenerationPlan plan = make_generation_plan(params, task);
    CHECK(!plan.encode_source);
    CHECK(!plan.encode_reference);
    CHECK(plan.run_lm);
    CHECK(plan.run_detokenizer);

    params.reference_audio.assign(4, 0.0f);
    CHECK(resolve_generate_task(params, task).empty());
    plan = make_generation_plan(params, task);
    CHECK(plan.encode_reference);
    CHECK(plan.run_lm);

    params.task_type = TASK_COVER_NOFSQ;
    params.source_audio.assign(4, 0.0f);
    params.reference_audio.clear();
    CHECK(resolve_generate_task(params, task).empty());
    plan = make_generation_plan(params, task);
    CHECK(plan.encode_source);
    CHECK(plan.reuse_source_reference);
    CHECK(!plan.run_lm);
    CHECK(!plan.run_detokenizer);
    CHECK(!plan.blend_cover_noise);

    params.reference_audio.assign(4, 0.0f);
    params.cover_noise_strength = 0.5f;
    CHECK(resolve_generate_task(params, task).empty());
    plan = make_generation_plan(params, task);
    CHECK(plan.encode_reference);
    CHECK(!plan.reuse_source_reference);
    CHECK(plan.blend_cover_noise);
}

void test_lego_task_kinds() {
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::TASK_LEGO;
    using tts_cpp::acestep::TASK_TEXT2MUSIC;
    using tts_cpp::acestep::LEGO_TRACK_NAMES;
    using tts_cpp::acestep::is_lego_task;
    using tts_cpp::acestep::is_source_task;
    using tts_cpp::acestep::is_valid_lego_track;

    CHECK(is_lego_task(TASK_LEGO));
    CHECK(!is_lego_task(TASK_TEXT2MUSIC));
    CHECK(!is_lego_task(TASK_COVER_NOFSQ));
    CHECK(is_source_task(TASK_LEGO));
    CHECK(is_source_task(TASK_COVER_NOFSQ));
    CHECK(!is_source_task(TASK_TEXT2MUSIC));

    for (const char * name : LEGO_TRACK_NAMES) {
        CHECK(is_valid_lego_track(name));
    }
    CHECK(!is_valid_lego_track(""));
    CHECK(!is_valid_lego_track("piano"));
    CHECK(!is_valid_lego_track("GUITAR"));
}

void test_lego_task_validation() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::TASK_LEGO;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;

    params.task_type = TASK_LEGO;
    CHECK(resolve_generate_task(params, task).find("requires source_audio") != std::string::npos);

    params.source_audio.assign(3, 0.0f);
    CHECK(resolve_generate_task(params, task).find("source_audio must be interleaved stereo") != std::string::npos);

    params.source_audio.assign(4, 0.0f);
    CHECK(resolve_generate_task(params, task).find("requires a track name") != std::string::npos);

    params.track = "accordion";
    CHECK(resolve_generate_task(params, task).find("unknown lego track") != std::string::npos);

    params.track = "guitar";
    CHECK(resolve_generate_task(params, task).empty());
    CHECK(task.type == TASK_LEGO);
    CHECK(task.track == "guitar");
}

void test_lego_generation_plan() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::GenerationPlan;
    using tts_cpp::acestep::TASK_LEGO;
    using tts_cpp::acestep::make_generation_plan;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;
    params.task_type = TASK_LEGO;
    params.track = "drums";
    params.source_audio.assign(4, 0.0f);
    CHECK(resolve_generate_task(params, task).empty());

    GenerationPlan plan = make_generation_plan(params, task);
    CHECK(plan.encode_source);
    CHECK(!plan.encode_reference);
    CHECK(!plan.reuse_source_reference);
    CHECK(!plan.run_lm);
    CHECK(!plan.run_detokenizer);
    CHECK(!plan.blend_cover_noise);

    params.reference_audio.assign(4, 0.0f);
    CHECK(resolve_generate_task(params, task).empty());
    plan = make_generation_plan(params, task);
    CHECK(plan.encode_reference);
    CHECK(!plan.reuse_source_reference);
}

void test_lego_model_policy() {
    using tts_cpp::acestep::is_sft_model_name;
    using tts_cpp::acestep::lego_model_error;

    CHECK(is_sft_model_name("acestep-v15-sft"));
    CHECK(is_sft_model_name("acestep-v15-xl-sft"));
    CHECK(!is_sft_model_name("acestep-v15-base"));
    CHECK(!is_sft_model_name("acestep-v15-xl-base"));
    CHECK(!is_sft_model_name("acestep-v15-turbo"));
    CHECK(!is_sft_model_name(""));

    CHECK(lego_model_error(false, false).empty());
    CHECK(lego_model_error(true, false).find("requires a base DiT") != std::string::npos);
    CHECK(lego_model_error(true, false).find("turbo") != std::string::npos);
    CHECK(lego_model_error(false, true).find("requires a base DiT") != std::string::npos);
    CHECK(lego_model_error(false, true).find("sft") != std::string::npos);
}

void test_guidance_and_dcw_policy() {
    using tts_cpp::acestep::resolve_dcw_enabled;
    using tts_cpp::acestep::resolve_guidance_scale;

    CHECK(approx(resolve_guidance_scale(0.0f, true), 1.0f));
    CHECK(approx(resolve_guidance_scale(7.0f, true), 1.0f));
    CHECK(approx(resolve_guidance_scale(0.0f, false), 7.0f));
    CHECK(approx(resolve_guidance_scale(3.5f, false), 3.5f));

    CHECK(resolve_dcw_enabled(true, true));
    CHECK(!resolve_dcw_enabled(true, false));
    CHECK(!resolve_dcw_enabled(false, true));
    CHECK(!resolve_dcw_enabled(false, false));
}

// APG guide: golden values hand-derived from the reference apg_forward
// (momentum -0.75, norm_threshold 2.5, projection per channel over T).
void test_apg_guide() {
    using tts_cpp::acestep::dit_apg_guide;

    std::vector<double> momentum(2, 0.0);
    std::vector<float> cond = { 3.0f, 4.0f };
    std::vector<float> uncond = { 2.0f, 2.0f };
    std::vector<float> velocity = cond;
    dit_apg_guide(velocity, uncond, momentum, 7.0f, 2, 1, 1);
    CHECK(approx(velocity[0], 1.08f, 1e-4f));
    CHECK(approx(velocity[1], 5.44f, 1e-4f));

    velocity = cond;
    dit_apg_guide(velocity, uncond, momentum, 7.0f, 2, 1, 1);
    CHECK(approx(velocity[0], 2.52f, 1e-4f));
    CHECK(approx(velocity[1], 4.36f, 1e-4f));

    std::vector<double> fresh_momentum(2, 0.0);
    std::vector<float> parallel_cond = { 10.0f, 0.0f };
    std::vector<float> parallel_uncond = { 6.0f, 0.0f };
    velocity = parallel_cond;
    dit_apg_guide(velocity, parallel_uncond, fresh_momentum, 7.0f, 2, 1, 1);
    CHECK(approx(velocity[0], 10.0f, 1e-4f));
    CHECK(approx(velocity[1], 0.0f, 1e-4f));

    std::vector<double> zero_momentum(2, 0.0);
    std::vector<float> equal = { 1.5f, -0.5f };
    velocity = equal;
    dit_apg_guide(velocity, equal, zero_momentum, 7.0f, 2, 1, 1);
    CHECK(approx(velocity[0], equal[0]));
    CHECK(approx(velocity[1], equal[1]));
}

void test_generation_conditioning() {
    using tts_cpp::acestep::AudioEncoder;
    using tts_cpp::acestep::EncodedAudio;
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::GenerationConditioning;
    using tts_cpp::acestep::GenerationPlan;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::TimbreInput;
    using tts_cpp::acestep::make_generation_plan;
    using tts_cpp::acestep::prepare_generation_conditioning;
    using tts_cpp::acestep::resolve_generate_task;
    using tts_cpp::acestep::resolve_timbre_input;

    GenerateParams params;
    params.task_type = TASK_COVER_NOFSQ;
    params.source_audio = { 1.0f, 2.0f, 3.0f, 4.0f };
    params.reference_audio = { 5.0f, 6.0f };

    GenerateTask task;
    CHECK(resolve_generate_task(params, task).empty());
    const GenerationPlan plan = make_generation_plan(params, task);

    int source_calls = 0;
    int reference_calls = 0;
    const AudioEncoder encode = [&](const std::vector<float> & pcm, const char * stage, EncodedAudio & output) {
        if (std::string(stage) == "source") ++source_calls;
        if (std::string(stage) == "reference") ++reference_calls;
        output.latent = { pcm.front(), pcm.back() };
        output.frames = (int) (pcm.size() / 2);
        return true;
    };

    GenerationConditioning conditioning;
    CHECK(prepare_generation_conditioning(params, plan, encode, conditioning));
    CHECK(source_calls == 1);
    CHECK(reference_calls == 1);
    CHECK(conditioning.source.latent.front() == params.source_audio.front());
    CHECK(conditioning.reference.latent.front() == params.reference_audio.front());

    const std::vector<float> silence = { -1.0f, -1.0f };
    const TimbreInput explicit_reference =
        resolve_timbre_input(plan, conditioning.reference, conditioning.source.latent,
                             conditioning.source.frames, silence);
    CHECK(explicit_reference.data == conditioning.reference.latent.data());
    CHECK(explicit_reference.frames == conditioning.reference.frames);

    params.reference_audio.clear();
    CHECK(resolve_generate_task(params, task).empty());
    const GenerationPlan reuse_plan = make_generation_plan(params, task);
    GenerationConditioning reuse_conditioning;
    CHECK(prepare_generation_conditioning(params, reuse_plan, encode, reuse_conditioning));
    CHECK(source_calls == 2);
    CHECK(reference_calls == 1);
    const TimbreInput reused_source =
        resolve_timbre_input(reuse_plan, reuse_conditioning.reference, reuse_conditioning.source.latent,
                             reuse_conditioning.source.frames, silence);
    CHECK(reused_source.data == reuse_conditioning.source.latent.data());
    CHECK(reused_source.frames == reuse_conditioning.source.frames);
}

void test_cover_noise_blending() {
    using tts_cpp::acestep::CoverNoiseResult;
    using tts_cpp::acestep::apply_cover_noise;

    std::vector<float> noise = { 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f };
    const std::vector<float> source = { 5.0f, 7.0f, 11.0f, 13.0f };
    std::vector<float> schedule = { 1.0f, 0.75f, 0.5f, 0.25f };

    const CoverNoiseResult result = apply_cover_noise(noise, source, 3, 2, 2, 0.5f, schedule);
    CHECK(approx(result.nearest_time, 0.5f));
    CHECK(result.remaining_steps == 2);
    CHECK(schedule.size() == 2);
    CHECK(approx(noise[0], 3.0f));
    CHECK(approx(noise[1], 4.0f));
    CHECK(approx(noise[2], 6.5f));
    CHECK(approx(noise[3], 7.5f));
    CHECK(approx(noise[4], 7.0f));
    CHECK(approx(noise[5], 8.0f));
}

constexpr float TEST_CONSERVATIVE_STRENGTH = 0.9f;
constexpr float TEST_BALANCED_STRENGTH = 0.4f;
constexpr float TEST_BALANCED_HALF_STRENGTH = 0.5f;
constexpr int TEST_BALANCED_BLEND_FRAMES = 15;
constexpr int TEST_BALANCED_HALF_BLEND_FRAMES = 12;
constexpr int TEST_REPAINT_SECONDS = 4;
constexpr int TEST_REPAINT_LATENT_FRAMES = 100;
constexpr int TEST_REPAINT_TRAILING_SAMPLES = 321;
constexpr char TEST_OUTPAINT_ERROR[] = "outpainting";
constexpr char TEST_RANGE_ORDER_ERROR[] = "greater";
constexpr char TEST_SOURCE_CAPTION_ERROR[] = "source_caption";
constexpr char TEST_FLOW_RANGE_ERROR[] = "n_min";
constexpr char TEST_FLOW_AVERAGE_ERROR[] = "n_avg";
constexpr char TEST_FLOW_DCW_ERROR[] = "DCW";
constexpr char TEST_SOURCE_CAPTION[] = "source";
constexpr char TEST_TARGET_CAPTION[] = "target";
constexpr char TEST_MATERIALIZE_EVENT[] = "materialize";
constexpr char TEST_FLOW_EVENT[] = "flow";

void test_repaint_config() {
    using namespace tts_cpp::acestep;

    const RepaintConfig conservative =
        resolve_repaint_config(RepaintMode::Conservative, TEST_CONSERVATIVE_STRENGTH);
    CHECK(approx(conservative.injection_ratio, REPAINT_CONSERVATIVE_INJECTION_RATIO));
    CHECK(conservative.latent_blend_frames == REPAINT_CONSERVATIVE_BLEND_FRAMES);
    CHECK(approx(conservative.waveform_fade_sec, REPAINT_CONSERVATIVE_FADE_SECONDS));
    CHECK(conservative.preserve_waveform);

    const RepaintConfig balanced =
        resolve_repaint_config(RepaintMode::Balanced, TEST_BALANCED_STRENGTH);
    CHECK(approx(balanced.injection_ratio, AUDIO_EDIT_MAX_RATIO - TEST_BALANCED_STRENGTH));
    CHECK(balanced.latent_blend_frames == TEST_BALANCED_BLEND_FRAMES);
    CHECK(approx(balanced.waveform_fade_sec,
                 REPAINT_CONSERVATIVE_FADE_SECONDS *
                     (AUDIO_EDIT_MAX_RATIO - TEST_BALANCED_STRENGTH)));
    CHECK(resolve_repaint_config(RepaintMode::Balanced, TEST_BALANCED_HALF_STRENGTH)
              .latent_blend_frames == TEST_BALANCED_HALF_BLEND_FRAMES);
    CHECK(audio_edit_round_ties_to_even(3.5f) == 4);
    CHECK(audio_edit_round_ties_to_even(4.5f) == 4);

    const RepaintConfig aggressive =
        resolve_repaint_config(RepaintMode::Aggressive, AUDIO_EDIT_MIN_RATIO);
    CHECK(approx(aggressive.injection_ratio, REPAINT_AGGRESSIVE_INJECTION_RATIO));
    CHECK(aggressive.latent_blend_frames == REPAINT_AGGRESSIVE_BLEND_FRAMES);
    CHECK(!aggressive.preserve_waveform);
}

void test_repaint_range() {
    using namespace tts_cpp::acestep;
    const int source_samples = AUDIO_EDIT_SAMPLE_RATE * TEST_REPAINT_SECONDS;
    RepaintRange range;
    CHECK(resolve_repaint_range(AUDIO_EDIT_MIN_RATIO, REPAINT_SOURCE_END_SECONDS,
                                source_samples, TEST_REPAINT_LATENT_FRAMES, range).empty());
    CHECK(range.sample_start == 0);
    CHECK(range.sample_end == source_samples);
    CHECK(range.latent_start == 0);
    CHECK(range.latent_end == TEST_REPAINT_LATENT_FRAMES);
    const int trailing_source_samples = source_samples + TEST_REPAINT_TRAILING_SAMPLES;
    const int trailing_latent_frames = TEST_REPAINT_LATENT_FRAMES + 1;
    CHECK(resolve_repaint_range(AUDIO_EDIT_MIN_RATIO, REPAINT_SOURCE_END_SECONDS,
                                trailing_source_samples, trailing_latent_frames, range).empty());
    CHECK(range.sample_end == trailing_source_samples);
    CHECK(range.latent_end == trailing_latent_frames);
    CHECK(resolve_repaint_range(-0.1f, 1.0f, source_samples, TEST_REPAINT_LATENT_FRAMES, range)
              .find(TEST_OUTPAINT_ERROR) != std::string::npos);
    CHECK(resolve_repaint_range(1.0f, 4.1f, source_samples, TEST_REPAINT_LATENT_FRAMES, range)
              .find(TEST_OUTPAINT_ERROR) != std::string::npos);
    CHECK(resolve_repaint_range(2.0f, 2.0f, source_samples, TEST_REPAINT_LATENT_FRAMES, range)
              .find(TEST_RANGE_ORDER_ERROR) != std::string::npos);
}

void check_all_samples_equal(const std::vector<float> & samples, float expected) {
    for (float sample : samples) CHECK(approx(sample, expected));
}

void test_repaint_mask_injection_blend_and_splice() {
    using namespace tts_cpp::acestep;

    const std::vector<float> mask = make_repaint_mask(5, 1, 4);
    CHECK(mask == std::vector<float>({ 0.0f, 1.0f, 1.0f, 1.0f, 0.0f }));

    std::vector<float> current(10, 10.0f);
    const std::vector<float> clean = { 2, 4, 2, 4, 2, 4, 2, 4, 2, 4 };
    const std::vector<float> noise(10, 8.0f);
    repaint_inject_source(
        current, clean.data(), noise.data(), mask.data(), mask.size(), 0.25f, 2);
    CHECK(approx(current[0], 3.5f));
    CHECK(approx(current[1], 5.0f));
    CHECK(approx(current[2], 10.0f));
    CHECK(approx(current[8], 3.5f));

    std::vector<float> generated(10, 10.0f);
    repaint_blend_latent(
        generated, clean.data(), mask.data(), mask.size(), 0, 2);
    CHECK(approx(generated[0], 2.0f));
    CHECK(approx(generated[1], 4.0f));
    CHECK(approx(generated[2], 10.0f));
    CHECK(approx(generated[8], 2.0f));

    generated.assign(10, 10.0f);
    repaint_blend_latent(
        generated, clean.data(), mask.data(), mask.size(), 1, 2);
    CHECK(generated[0] > clean[0] && generated[0] < 10.0f);
    CHECK(generated[8] > clean[8] && generated[8] < 10.0f);

    std::vector<float> wav(12, 1.0f);
    const std::vector<float> source(12, -1.0f);
    repaint_splice_waveform(wav, source, 2, 4, 0, 2);
    CHECK(approx(wav[0], -1.0f));
    CHECK(approx(wav[2], -1.0f));
    CHECK(approx(wav[4], 1.0f));
    CHECK(approx(wav[8], -1.0f));

    wav.assign(12, 1.0f);
    repaint_splice_waveform(wav, source, 0, 6, 2, 2);
    check_all_samples_equal(wav, 1.0f);
}

void test_flow_edit_validation_and_math() {
    using namespace tts_cpp::acestep;

    FlowEditParams params;
    CHECK(validate_flow_edit_params(params).find(TEST_SOURCE_CAPTION_ERROR) != std::string::npos);
    params.source_caption = TEST_SOURCE_CAPTION;
    params.target_caption = TEST_TARGET_CAPTION;
    CHECK(validate_flow_edit_params(params).empty());
    params.n_min = 0.8f;
    params.n_max = 0.2f;
    CHECK(validate_flow_edit_params(params).find(TEST_FLOW_RANGE_ERROR) != std::string::npos);
    params.n_min = AUDIO_EDIT_MIN_RATIO;
    params.n_max = AUDIO_EDIT_MAX_RATIO;
    params.n_avg = 0;
    CHECK(validate_flow_edit_params(params).find(TEST_FLOW_AVERAGE_ERROR) != std::string::npos);
    params.n_avg = FLOW_EDIT_DEFAULT_AVERAGES;
    params.dcw_enabled = true;
    CHECK(validate_flow_edit_params(params).find(TEST_FLOW_DCW_ERROR) != std::string::npos);

    const std::vector<float> source = { 2.0f, 4.0f };
    const std::vector<float> noise = { 10.0f, 12.0f };
    std::vector<float> noisy_source;
    flow_edit_make_source(source, noise, 0.25f, noisy_source);
    CHECK(approx(noisy_source[0], 4.0f));
    CHECK(approx(noisy_source[1], 6.0f));

    std::vector<float> edit = { 3.0f, 5.0f };
    std::vector<float> noisy_target;
    flow_edit_make_target(edit, noisy_source, source, noisy_target);
    CHECK(approx(noisy_target[0], 5.0f));
    CHECK(approx(noisy_target[1], 7.0f));
    flow_edit_integrate_delta(edit, { 5.0f, 7.0f }, { 1.0f, 2.0f }, -0.1f);
    CHECK(approx(edit[0], 2.6f));
    CHECK(approx(edit[1], 4.5f));
}

void test_audio_edit_factory_and_order() {
    using namespace tts_cpp::acestep;

    RepaintParams repaint;
    FlowEditParams flow;
    flow.source_caption = TEST_SOURCE_CAPTION;
    flow.target_caption = TEST_TARGET_CAPTION;

    std::unique_ptr<AudioEditOperation> repaint_operation =
        make_audio_edit_operation(AudioEditParams(repaint));
    std::unique_ptr<AudioEditOperation> flow_operation =
        make_audio_edit_operation(AudioEditParams(flow));
    CHECK(dynamic_cast<RepaintOperation *>(repaint_operation.get()) != nullptr);
    CHECK(dynamic_cast<FlowEditOperation *>(flow_operation.get()) != nullptr);

    const std::vector<AudioEditParams> plan = { flow, repaint, flow, repaint };
    AudioEditPipeline pipeline = make_audio_edit_pipeline(plan);
    CHECK(pipeline.size() == 4);
    CHECK(std::string(pipeline.at(0).name()) == AUDIO_EDIT_FLOW_STAGE);
    CHECK(std::string(pipeline.at(1).name()) == AUDIO_EDIT_REPAINT_STAGE);

    AudioEditArtifact artifact;
    std::vector<std::string> order;
    AudioEditCapabilities capabilities;
    capabilities.prepare_repaint_source = [&](AudioEditArtifact & value) {
        order.push_back(TEST_MATERIALIZE_EVENT);
        value.pcm_is_current = true;
    };
    capabilities.flow_edit = [&](const FlowEditParams &, AudioEditArtifact & value) {
        order.push_back(TEST_FLOW_EVENT);
        value.pcm_is_current = false;
    };
    capabilities.repaint = [&](const RepaintParams &, AudioEditArtifact & value) {
        order.push_back(AUDIO_EDIT_REPAINT_STAGE);
        value.pcm_is_current = false;
    };
    pipeline.execute(artifact, capabilities);
    const std::vector<std::string> expected = {
        TEST_FLOW_EVENT, TEST_MATERIALIZE_EVENT, AUDIO_EDIT_REPAINT_STAGE,
        TEST_FLOW_EVENT, TEST_MATERIALIZE_EVENT, AUDIO_EDIT_REPAINT_STAGE,
    };
    CHECK(order == expected);
}

void fill_fake_pcm(std::vector<float> & pcm, int frames) {
    for (int frame = 0; frame < frames; ++frame) {
        pcm[(size_t) frame * 2] = (float) frame;
        pcm[(size_t) frame * 2 + 1] = (float) -frame;
    }
}

void fill_fake_latent_frame(std::vector<float> & latent, int frame, float value) {
    using tts_cpp::acestep::VAE_LATENT_CHANNELS;
    for (int channel = 0; channel < VAE_LATENT_CHANNELS; ++channel) {
        latent[(size_t) frame * VAE_LATENT_CHANNELS + channel] = value + (float) channel;
    }
}

int encode_fake_vae(const float * pcm, int frames, std::vector<float> & latent) {
    using tts_cpp::acestep::VAE_ENCODER_UPSAMPLE;
    using tts_cpp::acestep::VAE_LATENT_CHANNELS;

    const int latent_frames = frames / VAE_ENCODER_UPSAMPLE;
    latent.resize((size_t) latent_frames * VAE_LATENT_CHANNELS);
    for (int frame = 0; frame < latent_frames; ++frame) {
        const float value = pcm[(size_t) frame * VAE_ENCODER_UPSAMPLE * 2];
        fill_fake_latent_frame(latent, frame, value);
    }
    return latent_frames;
}

void test_vae_encode_window_boundaries() {
    using tts_cpp::acestep::VAE_AUDIO_CHUNK_FRAMES;
    using tts_cpp::acestep::VAE_AUDIO_OVERLAP_FRAMES;
    using tts_cpp::acestep::make_vae_encode_window;
    using tts_cpp::acestep::vae_encode_window_count;

    CHECK(vae_encode_window_count(VAE_AUDIO_CHUNK_FRAMES) == 1);
    CHECK(vae_encode_window_count(VAE_AUDIO_CHUNK_FRAMES + 1) == 2);

    const auto first = make_vae_encode_window(0, VAE_AUDIO_CHUNK_FRAMES + 1);
    const auto second = make_vae_encode_window(1, VAE_AUDIO_CHUNK_FRAMES + 1);
    CHECK(first.window_start == 0);
    CHECK(second.window_start == second.core_start - VAE_AUDIO_OVERLAP_FRAMES);
    CHECK(first.core_end == second.core_start);
}

void test_vae_encode_window_parity() {
    using tts_cpp::acestep::VAE_AUDIO_CHUNK_FRAMES;
    using tts_cpp::acestep::VAE_AUDIO_STRIDE_FRAMES;
    using tts_cpp::acestep::encode_vae_pcm_bounded;

    const int frames = VAE_AUDIO_CHUNK_FRAMES + VAE_AUDIO_STRIDE_FRAMES;
    std::vector<float> pcm((size_t) frames * 2);
    fill_fake_pcm(pcm, frames);

    std::vector<float> expected;
    const int expected_frames = encode_fake_vae(pcm.data(), frames, expected);
    int actual_frames = 0;
    const std::vector<float> actual =
        encode_vae_pcm_bounded(pcm, frames, encode_fake_vae, &actual_frames);

    CHECK(actual_frames == expected_frames);
    CHECK(actual == expected);
}

template <typename T>
void write_test_value(FILE * file, T value) {
    fwrite(&value, sizeof(T), 1, file);
}

void write_test_wav(FILE * file, uint16_t channels, const std::vector<int16_t> & samples) {
    constexpr uint16_t PCM_FORMAT = 1;
    constexpr uint16_t BITS = 16;
    constexpr uint32_t RATE = 48000;
    constexpr uint32_t FORMAT_SIZE = 16;

    const uint16_t block_align = channels * (BITS / 8);
    const uint32_t byte_rate = RATE * block_align;
    const uint32_t data_size = (uint32_t) (samples.size() * sizeof(int16_t));
    const uint32_t riff_size = 36 + data_size;

    fwrite("RIFF", 1, 4, file);
    write_test_value(file, riff_size);
    fwrite("WAVE", 1, 4, file);
    fwrite("fmt ", 1, 4, file);
    write_test_value(file, FORMAT_SIZE);
    write_test_value(file, PCM_FORMAT);
    write_test_value(file, channels);
    write_test_value(file, RATE);
    write_test_value(file, byte_rate);
    write_test_value(file, block_align);
    write_test_value(file, BITS);
    fwrite("data", 1, 4, file);
    write_test_value(file, data_size);
    fwrite(samples.data(), sizeof(int16_t), samples.size(), file);
}

FILE * open_test_file() {
    FILE * file = tmpfile();
    CHECK(file != nullptr);
    return file;
}

void test_wav_reader_mono_and_stereo() {
    using tts_cpp::acestep::WavReadResult;
    using tts_cpp::acestep::read_pcm16_wav;

    FILE * mono = open_test_file();
    if (!mono) return;
    write_test_wav(mono, 1, { 16384, -16384 });
    const WavReadResult mono_result = read_pcm16_wav(mono);
    fclose(mono);
    CHECK(mono_result.error.empty());
    CHECK(mono_result.frames == 2);
    CHECK(approx(mono_result.pcm[0], 0.5f));
    CHECK(approx(mono_result.pcm[1], 0.5f));

    FILE * stereo = open_test_file();
    if (!stereo) return;
    write_test_wav(stereo, 2, { 16384, -16384 });
    const WavReadResult stereo_result = read_pcm16_wav(stereo);
    fclose(stereo);
    CHECK(stereo_result.error.empty());
    CHECK(stereo_result.frames == 1);
    CHECK(approx(stereo_result.pcm[0], 0.5f));
    CHECK(approx(stereo_result.pcm[1], -0.5f));
}

void test_wav_reader_rejects_multichannel() {
    using tts_cpp::acestep::WavReadResult;
    using tts_cpp::acestep::read_pcm16_wav;

    FILE * file = open_test_file();
    if (!file) return;
    write_test_wav(file, 3, { 1, 2, 3 });
    const WavReadResult result = read_pcm16_wav(file);
    fclose(file);
    CHECK(!result.error.empty());
    CHECK(result.pcm.empty());
}

void test_quantize_policy() {
    using namespace tts_cpp::acestep;

    const QuantVariant * q4km = find_quant_variant("q4_k_m");
    const QuantVariant * q80  = find_quant_variant("Q8_0");
    const QuantVariant * q2k  = find_quant_variant("Q2_K");
    const QuantVariant * q3kl = find_quant_variant("Q3_K_L");
    CHECK(q4km != nullptr && q4km->base == GGML_TYPE_Q4_K);
    CHECK(q80 != nullptr && q2k != nullptr && q3kl != nullptr);
    CHECK(find_quant_variant("Q4_K") == nullptr);

    CHECK(quant_layer_index("model.layers.12.self_attn.v_proj.weight") == 12);
    CHECK(quant_layer_index("model.embed_tokens.weight") == -1);

    // The deny-list: VAE files, 1-D tensors, the text encoder's embedding, and
    // the DiT's quality-critical extras are never quantized.
    CHECK(quant_pick_type("decoder.conv1.weight", 3, "acestep-vae", *q4km, 0) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("model.norm.weight", 1, "acestep-lm", *q4km, 24) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("embed_tokens.weight", 2, "acestep-text-enc", *q4km, 28) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("silence_latent", 2, "acestep-dit", *q4km, 24) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("decoder.scale_shift_table", 2, "acestep-dit", *q4km, 24) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("null_condition_emb", 2, "acestep-dit", *q4km, 24) == GGML_TYPE_COUNT);

    // The LM embedding takes the variant's embed type, everything ordinary the base.
    CHECK(quant_pick_type("model.embed_tokens.weight", 2, "acestep-lm", *q4km, 24) == GGML_TYPE_Q6_K);
    CHECK(quant_pick_type("model.embed_tokens.weight", 2, "acestep-lm", *q80, 24) == GGML_TYPE_Q8_0);
    CHECK(quant_pick_type("model.layers.4.self_attn.q_proj.weight", 2, "acestep-lm", *q4km, 24) ==
          GGML_TYPE_Q4_K);

    // M-variant bump: first n/9, last n/7, and every 3rd layer of v_proj/down_proj.
    CHECK(quant_pick_type("model.layers.0.self_attn.v_proj.weight", 2, "acestep-lm", *q4km, 24) ==
          GGML_TYPE_Q6_K);
    CHECK(quant_pick_type("model.layers.3.mlp.down_proj.weight", 2, "acestep-lm", *q4km, 24) ==
          GGML_TYPE_Q6_K);
    CHECK(quant_pick_type("model.layers.22.self_attn.v_proj.weight", 2, "acestep-lm", *q4km, 24) ==
          GGML_TYPE_Q6_K);
    CHECK(quant_pick_type("model.layers.4.self_attn.v_proj.weight", 2, "acestep-lm", *q4km, 24) ==
          GGML_TYPE_Q4_K);

    // S-variant bump: only the first bump_layer_count layers.
    CHECK(quant_pick_type("model.layers.3.self_attn.v_proj.weight", 2, "acestep-lm", *q2k, 24) ==
          GGML_TYPE_Q4_K);
    CHECK(quant_pick_type("model.layers.4.self_attn.v_proj.weight", 2, "acestep-lm", *q2k, 24) ==
          GGML_TYPE_Q2_K);

    // L-variant bump: o_proj joins the important set, at every layer.
    CHECK(quant_pick_type("model.layers.10.self_attn.o_proj.weight", 2, "acestep-lm", *q3kl, 24) ==
          GGML_TYPE_Q5_K);
    CHECK(quant_pick_type("model.layers.10.self_attn.o_proj.weight", 2, "acestep-lm", *q4km, 24) ==
          GGML_TYPE_Q4_K);

    CHECK(quant_should_promote_f32(1));
    CHECK(!quant_should_promote_f32(2));
}

// MM3's LM (arch "qwen3") uses llama.cpp-style tensor names, and its synth file
// (arch "mm3") bundles a DiT with a condition encoder, RVQ depth decoder, vocoder,
// and timestep Fourier basis that must stay untouched. Neither reuses the HF-style
// v_proj/down_proj/o_proj bump machinery: MM3's own attn_v.weight/ffn_down.weight/
// attn_output.weight names are shared verbatim between the LM and the (denylisted)
// depth decoder, and the DiT's fused attn_qkv/ffn_in/ffn_out layout has no bump
// target to protect, so quantization is flat aside from the embed/output-head
// overrides checked here.
void test_quantize_policy_mm3() {
    using namespace tts_cpp::acestep;

    const QuantVariant * q4km = find_quant_variant("q4_k_m");
    CHECK(q4km != nullptr);

    // LM embedding table and untied output head both take the embed type.
    CHECK(quant_pick_type("token_embd.weight", 2, "qwen3", *q4km, 36) == GGML_TYPE_Q6_K);
    CHECK(quant_pick_type("output.weight", 2, "qwen3", *q4km, 36) == GGML_TYPE_Q6_K);

    // Everything else in the LM is flat base-type: no bump for MM3's naming.
    CHECK(quant_pick_type("blk.0.attn_v.weight", 2, "qwen3", *q4km, 36) == GGML_TYPE_Q4_K);
    CHECK(quant_pick_type("blk.35.ffn_down.weight", 2, "qwen3", *q4km, 36) == GGML_TYPE_Q4_K);
    CHECK(quant_pick_type("blk.0.attn_output.weight", 2, "qwen3", *q4km, 36) == GGML_TYPE_Q4_K);

    // The untied-output rule is scoped to "qwen3" only; it must not leak into ACE-Step's LM.
    CHECK(quant_pick_type("output.weight", 2, "acestep-lm", *q4km, 24) == GGML_TYPE_Q4_K);

    // Synth policy: condition encoder and vocoder never quantize; the RVQ depth
    // decoder follows the requested variant like the DiT, keeping its positional
    // embedding raw and its get_rows code table at Q8_0 (CUDA get_rows has no
    // k-quant path).
    CHECK(quant_pick_type("depth.proj.weight", 2, "mm3", *q4km, 0) == GGML_TYPE_Q4_K);
    CHECK(quant_pick_type("depth.blk.0.attn_v.weight", 2, "mm3", *q4km, 0) == GGML_TYPE_Q4_K);
    CHECK(quant_pick_type("depth.audio_embd.weight", 2, "mm3", *q4km, 0) == GGML_TYPE_Q8_0);
    CHECK(quant_pick_type("depth.audio_embd.weight", 2, "mm3", *find_quant_variant("Q8_0"), 0) ==
          GGML_TYPE_Q8_0);
    CHECK(quant_pick_type("depth.blk.0.input_norm.weight", 1, "mm3", *q4km, 0) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("depth.pos_embd.weight", 2, "mm3", *q4km, 0) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("cond.proj.weight", 3, "mm3", *q4km, 0) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("voc.conv_in.weight", 3, "mm3", *q4km, 0) == GGML_TYPE_COUNT);
    CHECK(quant_pick_type("dit.time_fourier.weight", 2, "mm3", *q4km, 0) == GGML_TYPE_COUNT);

    // The DiT is the one synth component that quantizes, flat, including the
    // attn_output.weight name it shares with the LM and the denylisted depth decoder.
    CHECK(quant_pick_type("dit.blk.0.attn_qkv.weight", 2, "mm3", *q4km, 0) == GGML_TYPE_Q4_K);
    CHECK(quant_pick_type("dit.proj_in.weight", 2, "mm3", *q4km, 0) == GGML_TYPE_Q4_K);
    CHECK(quant_pick_type("dit.blk.0.attn_output.weight", 2, "mm3", *q4km, 0) == GGML_TYPE_Q4_K);
}

// End-to-end guard on the offset/padding planning and the streaming writer: a
// regression there emits a byte-shifted GGUF that only fails at engine load.
// Quantizes a synthetic BF16-era file and reads the result back through the
// gguf loader (which validates offsets and alignment structurally), then
// checks every tensor's type and dequantized values against the source.
void test_quantize_gguf_roundtrip() {
    using namespace tts_cpp::acestep;

    const std::string in_path  = test_temp_dir() + "/qvac-acestep-quantize-in.gguf";
    const std::string out_path = test_temp_dir() + "/qvac-acestep-quantize-out.gguf";

    // 32-wide rows quantize under Q8_0 (block size 32); the 24-wide tensor is
    // unaligned and must be kept as stored; the BF16 1-D norm must be promoted
    // to F32. Q8_0 row size is 34 bytes, so quantized tensors end misaligned
    // and the writer's padding path is exercised between every pair.
    std::vector<float> wide(32 * 4);
    for (size_t i = 0; i < wide.size(); ++i) {
        wide[i] = ((float) i - 64.0f) / 32.0f;
    }
    std::vector<float> narrow(24 * 2);
    for (size_t i = 0; i < narrow.size(); ++i) {
        narrow[i] = ((float) i - 24.0f) / 8.0f;
    }
    std::vector<float> norm(32);
    for (size_t i = 0; i < norm.size(); ++i) {
        norm[i] = 1.0f + (float) i / 100.0f;
    }

    ggml_init_params ip{ 4 * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context *   ctx = ggml_init(ip);
    gguf_context *   gc  = gguf_init_empty();
    gguf_set_val_str(gc, "general.architecture", "acestep-lm");
    gguf_set_val_u32(gc, "acestep-lm.block_count", 1);

    ggml_tensor * embed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 4);
    ggml_set_name(embed, "model.embed_tokens.weight");
    std::memcpy(embed->data, wide.data(), wide.size() * sizeof(float));
    gguf_add_tensor(gc, embed);

    ggml_tensor * vproj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 4);
    ggml_set_name(vproj, "model.layers.0.self_attn.v_proj.weight");
    std::memcpy(vproj->data, wide.data(), wide.size() * sizeof(float));
    gguf_add_tensor(gc, vproj);

    ggml_tensor * ln = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, 32);
    ggml_set_name(ln, "model.layers.0.input_layernorm.weight");
    ggml_fp32_to_bf16_row(norm.data(), (ggml_bf16_t *) ln->data, (int64_t) norm.size());
    gguf_add_tensor(gc, ln);

    std::vector<float> norm_bf16(norm.size());
    ggml_bf16_to_fp32_row((const ggml_bf16_t *) ln->data, norm_bf16.data(), (int64_t) norm.size());

    ggml_tensor * odd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 2);
    ggml_set_name(odd, "model.layers.0.mlp.gate_proj.weight");
    std::memcpy(odd->data, narrow.data(), narrow.size() * sizeof(float));
    gguf_add_tensor(gc, odd);

    CHECK(gguf_write_to_file(gc, in_path.c_str(), /*only_meta=*/false));
    gguf_free(gc);
    ggml_free(ctx);

    const QuantVariant * q80 = find_quant_variant("Q8_0");
    QuantizeStats        stats;
    std::string          error;
    CHECK(quantize_gguf_file(in_path, out_path, *q80, stats, error));
    CHECK(error.empty());
    CHECK(stats.n_tensors == 4);
    CHECK(stats.n_quantized == 2);
    CHECK(stats.n_promoted == 1);

    ggml_context *   out_meta   = nullptr;
    gguf_init_params out_params = { /*no_alloc=*/false, /*ctx=*/&out_meta };
    gguf_context *   out        = gguf_init_from_file(out_path.c_str(), out_params);
    CHECK(out != nullptr);
    if (out) {
        const int64_t ft_idx = gguf_find_key(out, "general.file_type");
        CHECK(ft_idx >= 0 && gguf_get_kv_type(out, ft_idx) == GGUF_TYPE_UINT32);
        CHECK(gguf_get_val_u32(out, ft_idx) == (uint32_t) GGML_FTYPE_MOSTLY_Q8_0);

        const size_t alignment = gguf_get_alignment(out);
        for (int64_t i = 0; i < gguf_get_n_tensors(out); ++i) {
            CHECK(gguf_get_tensor_offset(out, i) % alignment == 0);
        }

        ggml_tensor * q_vproj = ggml_get_tensor(out_meta, "model.layers.0.self_attn.v_proj.weight");
        CHECK(q_vproj && q_vproj->type == GGML_TYPE_Q8_0);
        if (q_vproj) {
            std::vector<float> back(wide.size());
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(q_vproj->data, back.data(),
                                                           (int64_t) back.size());
            for (size_t i = 0; i < back.size(); ++i) {
                CHECK(std::fabs(back[i] - wide[i]) < 0.02f);
            }
        }

        ggml_tensor * q_embed = ggml_get_tensor(out_meta, "model.embed_tokens.weight");
        CHECK(q_embed && q_embed->type == GGML_TYPE_Q8_0);

        ggml_tensor * q_ln = ggml_get_tensor(out_meta, "model.layers.0.input_layernorm.weight");
        CHECK(q_ln && q_ln->type == GGML_TYPE_F32);
        if (q_ln) {
            const float * vals = (const float *) q_ln->data;
            for (size_t i = 0; i < norm_bf16.size(); ++i) {
                CHECK(vals[i] == norm_bf16[i]);
            }
        }

        ggml_tensor * q_odd = ggml_get_tensor(out_meta, "model.layers.0.mlp.gate_proj.weight");
        CHECK(q_odd && q_odd->type == GGML_TYPE_F32);
        if (q_odd) {
            CHECK(std::memcmp(q_odd->data, narrow.data(), narrow.size() * sizeof(float)) == 0);
        }

        gguf_free(out);
        ggml_free(out_meta);
    }

    std::remove(in_path.c_str());
    std::remove(out_path.c_str());
}

// Same end-to-end guard for the MM3 synth layout: the protected condition
// encoder must pass through byte-identical, the depth decoder must land at
// Q8_0, and the DiT takes the variant base type. A regression here is silent
// (the file still loads) and only shows up as degraded audio.
// Each tensor gets its own source values so an expectation built from the
// wrong tensor's bytes cannot pass.
std::vector<float> quant_source_row(size_t count, float offset) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = ((float) i - offset) / 64.0f;
    }
    return values;
}

void test_quantize_gguf_roundtrip_mm3() {
    using namespace tts_cpp::acestep;

    const std::string in_path  = test_temp_dir() + "/qvac-mm3-quantize-in.gguf";
    const std::string out_path = test_temp_dir() + "/qvac-mm3-quantize-out.gguf";

    // 256-wide rows satisfy the k-quant superblock alignment, so only the
    // policy (not the alignment fallback) decides who quantizes.
    const std::vector<float> cond_row  = quant_source_row(256 * 2, 256.0f);
    const std::vector<float> depth_row = quant_source_row(256 * 2, 320.0f);
    const std::vector<float> dit_row   = quant_source_row(256 * 2, 400.0f);

    ggml_init_params ip{ 4 * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context *   ctx = ggml_init(ip);
    gguf_context *   gc  = gguf_init_empty();
    gguf_set_val_str(gc, "general.architecture", "mm3");

    ggml_tensor * cond = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 2);
    ggml_set_name(cond, "cond.proj.weight");
    ggml_fp32_to_fp16_row(cond_row.data(), (ggml_fp16_t *) cond->data, (int64_t) cond_row.size());
    gguf_add_tensor(gc, cond);
    std::vector<ggml_fp16_t> cond_bytes((size_t) cond_row.size());
    std::memcpy(cond_bytes.data(), cond->data, cond_row.size() * sizeof(ggml_fp16_t));

    ggml_tensor * depth = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 2);
    ggml_set_name(depth, "depth.blk.0.attn_v.weight");
    ggml_fp32_to_fp16_row(depth_row.data(), (ggml_fp16_t *) depth->data, (int64_t) depth_row.size());
    gguf_add_tensor(gc, depth);
    std::vector<ggml_fp16_t> depth_bytes((size_t) depth_row.size());
    std::memcpy(depth_bytes.data(), depth->data, depth_row.size() * sizeof(ggml_fp16_t));

    ggml_tensor * dit = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 2);
    ggml_set_name(dit, "dit.blk.0.attn_qkv.weight");
    ggml_fp32_to_fp16_row(dit_row.data(), (ggml_fp16_t *) dit->data, (int64_t) dit_row.size());
    gguf_add_tensor(gc, dit);
    std::vector<ggml_fp16_t> dit_bytes((size_t) dit_row.size());
    std::memcpy(dit_bytes.data(), dit->data, dit_row.size() * sizeof(ggml_fp16_t));

    CHECK(gguf_write_to_file(gc, in_path.c_str(), /*only_meta=*/false));
    gguf_free(gc);
    ggml_free(ctx);

    const QuantVariant * q4km = find_quant_variant("Q4_K_M");
    QuantizeStats        stats;
    std::string          error;
    CHECK(quantize_gguf_file(in_path, out_path, *q4km, stats, error));
    CHECK(error.empty());
    CHECK(stats.n_tensors == 3);
    CHECK(stats.n_quantized == 2);

    ggml_context *   out_meta   = nullptr;
    gguf_init_params out_params = { /*no_alloc=*/false, /*ctx=*/&out_meta };
    gguf_context *   out        = gguf_init_from_file(out_path.c_str(), out_params);
    CHECK(out != nullptr);
    if (out) {
        ggml_tensor * q_cond = ggml_get_tensor(out_meta, "cond.proj.weight");
        CHECK(q_cond && q_cond->type == GGML_TYPE_F16);
        if (q_cond) {
            CHECK(std::memcmp(q_cond->data, cond_bytes.data(),
                              cond_bytes.size() * sizeof(ggml_fp16_t)) == 0);
        }

        ggml_tensor * q_depth = ggml_get_tensor(out_meta, "depth.blk.0.attn_v.weight");
        CHECK(q_depth && q_depth->type == GGML_TYPE_Q4_K);
        if (q_depth) {
            std::vector<float> f32(depth_row.size());
            ggml_fp16_to_fp32_row(depth_bytes.data(), f32.data(), (int64_t) f32.size());
            std::vector<uint8_t> expected(ggml_row_size(GGML_TYPE_Q4_K, 256) * 2);
            ggml_quantize_chunk(GGML_TYPE_Q4_K, f32.data(), expected.data(), 0, 2, 256, nullptr);
            CHECK(std::memcmp(q_depth->data, expected.data(), expected.size()) == 0);
        }

        ggml_tensor * q_dit = ggml_get_tensor(out_meta, "dit.blk.0.attn_qkv.weight");
        CHECK(q_dit && q_dit->type == GGML_TYPE_Q4_K);
        if (q_dit) {
            // The writer must emit exactly ggml's reference quantization of the
            // F32-converted source rows (ggml_quantize_chunk, no imatrix).
            std::vector<float> f32(dit_row.size());
            ggml_fp16_to_fp32_row(dit_bytes.data(), f32.data(), (int64_t) f32.size());
            std::vector<uint8_t> expected(ggml_row_size(GGML_TYPE_Q4_K, 256) * 2);
            ggml_quantize_chunk(GGML_TYPE_Q4_K, f32.data(), expected.data(), 0, 2, 256, nullptr);
            CHECK(std::memcmp(q_dit->data, expected.data(), expected.size()) == 0);

            std::vector<float> back(dit_row.size());
            ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(q_dit->data, back.data(),
                                                           (int64_t) back.size());
            for (size_t i = 0; i < back.size(); ++i) {
                CHECK(std::fabs(back[i] - dit_row[i]) < 0.25f);
            }
        }

        gguf_free(out);
        ggml_free(out_meta);
    }

    std::remove(in_path.c_str());
    std::remove(out_path.c_str());
}

// 12. bpe tokenizer ------------------------------------------------------------
// Weight-free: the tokenizer is hand-built instead of loaded from a GGUF, so
// these lock bpe_encode/bpe_decode/bpe_utf8_codepoint against a tiny vocab
// whose merges are fully traceable by hand.

// Replica of the GPT-2 byte-level table build_byte_encoder() computes at load:
// printable/latin bytes map to their own codepoint, the rest to 256+n in
// ascending byte order (so ' ' -> U+0120).
void build_test_byte2str(std::string byte2str[256]) {
    bool direct[256] = {};
    for (int b = '!'; b <= '~'; ++b) direct[b] = true;
    for (int b = 0xA1; b <= 0xAC; ++b) direct[b] = true;
    for (int b = 0xAE; b <= 0xFF; ++b) direct[b] = true;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        const int cp = direct[b] ? b : 256 + n++;
        std::string out;
        if (cp < 0x80) {
            out += (char) cp;
        } else if (cp < 0x800) {
            out += (char) (0xC0 | (cp >> 6));
            out += (char) (0x80 | (cp & 0x3F));
        } else {
            out += (char) (0xE0 | (cp >> 12));
            out += (char) (0x80 | ((cp >> 6) & 0x3F));
            out += (char) (0x80 | (cp & 0x3F));
        }
        byte2str[b] = out;
    }
}

// Vocab ids: h=0 e=1 l=2 o=3 a=4 b=5 1=6 2=7 he=8 ll=9 hello=10 Ġhello=11.
// The merge chain h+e, l+l, he+ll, hell+o, Ġ+hello reaches "hello" and
// " hello"; "a b" produces a piece deliberately absent from the vocab.
tts_cpp::acestep::BpeTokenizer make_test_bpe_tokenizer() {
    tts_cpp::acestep::BpeTokenizer tok;
    build_test_byte2str(tok.byte2str);
    const std::string g_space = tok.byte2str[(unsigned char) ' '];
    const std::vector<std::string> tokens = {
        "h", "e", "l", "o", "a", "b", "1", "2", "he", "ll", "hello", g_space + "hello",
    };
    for (size_t i = 0; i < tokens.size(); ++i) tok.vocab[tokens[i]] = (int) i;
    tok.n_vocab   = (int) tokens.size();
    tok.id_to_str = tokens;
    const std::vector<std::string> merge_list = {
        "h e", "l l", "he ll", "hell o", g_space + " hello", "a b",
    };
    for (size_t i = 0; i < merge_list.size(); ++i) tok.merges[merge_list[i]] = (int) i;
    return tok;
}

// A cancel armed before generate() must survive into the run that observes it
// and be consumed on exit, so the next run starts clean. Pinned on the scope
// itself: the engine-level path needs model weights.
void test_cancellation_scope_consumes_on_exit() {
    using tts_cpp::acestep::CancellationScope;

    std::atomic<bool> cancelled{ true };
    {
        CancellationScope scope(cancelled);
        CHECK(cancelled.load());
    }
    CHECK(!cancelled.load());

    cancelled.store(false);
    {
        CancellationScope scope(cancelled);
        CHECK(!cancelled.load());
        cancelled.store(true);
    }
    CHECK(!cancelled.load());
}

void test_bpe_utf8_codepoint() {
    using tts_cpp::acestep::bpe_utf8_codepoint;

    int adv = 0;
    CHECK(bpe_utf8_codepoint("A", &adv) == 0x41 && adv == 1);
    CHECK(bpe_utf8_codepoint("~", &adv) == 0x7E && adv == 1);
    CHECK(bpe_utf8_codepoint("\xC3\xA9", &adv) == 0xE9 && adv == 2);      // é
    CHECK(bpe_utf8_codepoint("\xE4\xB8\xAD", &adv) == 0x4E2D && adv == 3);  // 中
    CHECK(bpe_utf8_codepoint("\xF0\x90\x90\x80", &adv) == 0x10400 && adv == 4);

    // Malformed lead bytes fall back to advance-by-1 with the raw byte value.
    const char lone_continuation[] = { (char) 0x80, 0, 0, 0, 0 };
    CHECK(bpe_utf8_codepoint(lone_continuation, &adv) == 0x80 && adv == 1);
    const char invalid_lead[] = { (char) 0xFF, 0, 0, 0, 0 };
    CHECK(bpe_utf8_codepoint(invalid_lead, &adv) == 0xFF && adv == 1);

    // A sequence that runs into the terminator decodes as its raw lead byte
    // rather than consuming the declared width and reading past the end.
    const char truncated_two[] = { (char) 0xC3, 0, 0, 0, 0 };
    CHECK(bpe_utf8_codepoint(truncated_two, &adv) == 0xC3 && adv == 1);
    const char truncated_three[] = { (char) 0xE4, 0, 0, 0, 0 };
    CHECK(bpe_utf8_codepoint(truncated_three, &adv) == 0xE4 && adv == 1);
    const char truncated_three_partial[] = { (char) 0xE4, (char) 0xB8, 0, 0, 0 };
    CHECK(bpe_utf8_codepoint(truncated_three_partial, &adv) == 0xE4 && adv == 1);
    const char truncated_four[] = { (char) 0xF0, (char) 0x90, (char) 0x90, 0, 0 };
    CHECK(bpe_utf8_codepoint(truncated_four, &adv) == 0xF0 && adv == 1);

    // A lead byte at the very end of a buffer must not be read past: this is
    // the case that motivated the clamp.
    const char lead_at_end[] = { (char) 0xF0, 0 };
    CHECK(bpe_utf8_codepoint(lead_at_end, &adv) == 0xF0 && adv == 1);
}

void test_bpe_encode_merges() {
    using tts_cpp::acestep::bpe_encode;

    const auto tok = make_test_bpe_tokenizer();

    // Full merge chain: h+e, l+l, he+ll, hell+o -> "hello".
    CHECK(bpe_encode(tok, "hello") == std::vector<int>({ 10 }));
    // Partial merge: only "h e" applies, the rest stays single symbols.
    CHECK(bpe_encode(tok, "helo") == std::vector<int>({ 8, 2, 3 }));
    // The GPT-2 pre-tokenizer folds the leading space into the word; the
    // byte-level 'Ġ' symbol then merges into a single vocab entry.
    CHECK(bpe_encode(tok, " hello") == std::vector<int>({ 11 }));
    CHECK(bpe_encode(tok, "hello hello") == std::vector<int>({ 10, 11 }));
    // Digits are pre-tokenized one at a time.
    CHECK(bpe_encode(tok, "12") == std::vector<int>({ 6, 7 }));

    CHECK(bpe_encode(tok, "").empty());
    CHECK(bpe_encode(tok, "", /*add_eos=*/true) == std::vector<int>({ tok.eos_id }));
    CHECK(bpe_encode(tok, "hello<|endoftext|>helo") == std::vector<int>({ 10, tok.eos_id, 8, 2, 3 }));
    CHECK(bpe_encode(tok, "<|endoftext|>") == std::vector<int>({ tok.eos_id }));
}

void test_bpe_encode_byte_fallback() {
    using tts_cpp::acestep::bpe_encode;

    const auto tok = make_test_bpe_tokenizer();

    // "a b" merges into "ab", which is not in the vocab: the encoder falls
    // back to the per-byte tokens.
    CHECK(bpe_encode(tok, "ab") == std::vector<int>({ 4, 5 }));
    // Bytes with no single-byte vocab entry are dropped silently.
    CHECK(bpe_encode(tok, "q").empty());
    CHECK(bpe_encode(tok, "abq") == std::vector<int>({ 4, 5 }));
    // A raw non-UTF-8 byte maps to a byte-level symbol outside the vocab whose
    // fallback bytes are unknown too, so it encodes to nothing.
    CHECK(bpe_encode(tok, "\x80").empty());
}

void test_bpe_decode_roundtrip() {
    using tts_cpp::acestep::AUDIO_CODE_BASE;
    using tts_cpp::acestep::TOKEN_IM_END;
    using tts_cpp::acestep::TOKEN_IM_START;
    using tts_cpp::acestep::TOKEN_THINK;
    using tts_cpp::acestep::TOKEN_THINK_END;
    using tts_cpp::acestep::bpe_decode;
    using tts_cpp::acestep::bpe_encode;
    using tts_cpp::acestep::bpe_utf8_codepoint;

    auto tok = make_test_bpe_tokenizer();

    // byte_dec is empty on the hand-built tokenizer, so this exercises the
    // local-fallback byte decoder.
    CHECK(bpe_decode(tok, bpe_encode(tok, "hello")) == "hello");
    CHECK(bpe_decode(tok, bpe_encode(tok, "helo")) == "helo");
    CHECK(bpe_decode(tok, bpe_encode(tok, " hello")) == " hello");
    CHECK(bpe_decode(tok, {}).empty());

    // eos decodes to nothing (outside id_to_str), as do negative and audio ids.
    CHECK(bpe_decode(tok, bpe_encode(tok, "", /*add_eos=*/true)).empty());
    CHECK(bpe_decode(tok, { -1, tok.n_vocab, AUDIO_CODE_BASE, AUDIO_CODE_BASE + 5, 10 }) == "hello");
    CHECK(bpe_decode(tok, { TOKEN_IM_START, 10, TOKEN_IM_END }) == "hello");
    CHECK(bpe_decode(tok, { TOKEN_THINK, 10, TOKEN_THINK_END }) == "<think>hello</think>");

    // Same results through the cached byte_dec branch bpe_load_from_gguf fills.
    for (int b = 0; b < 256; ++b) {
        int adv;
        tok.byte_dec[bpe_utf8_codepoint(tok.byte2str[b].c_str(), &adv)] = (uint8_t) b;
    }
    CHECK(bpe_decode(tok, bpe_encode(tok, " hello")) == " hello");
    CHECK(bpe_decode(tok, { TOKEN_THINK, 10, TOKEN_THINK_END }) == "<think>hello</think>");
}

// 13. quality scoring ----------------------------------------------------------
// Weight-free coverage of the teacher-forced scoring math and target builders
// (quality_score.h); the LM-backed end-to-end path runs in the integration test.

bool quality_near(double a, double b) {
    return std::fabs(a - b) < 1e-9;
}

void test_quality_normalized_pmi() {
    using tts_cpp::acestep::quality_normalized_pmi;

    CHECK(quality_near(quality_normalized_pmi(-1.0, -1.0, 0.1), 0.5));
    CHECK(quality_near(quality_normalized_pmi(-0.9, -1.0, 0.1), 1.0 / (1.0 + std::exp(-1.0))));
    CHECK(quality_near(quality_normalized_pmi(-1.1, -1.0, 0.1), 1.0 - 1.0 / (1.0 + std::exp(-1.0))));
    CHECK(quality_normalized_pmi(0.0, -100.0, 0.1) > 0.999999);
    CHECK(quality_normalized_pmi(-100.0, 0.0, 0.1) < 0.000001);
}

void test_quality_yaml_formatting() {
    using tts_cpp::acestep::quality_yaml_plain_safe;
    using tts_cpp::acestep::quality_yaml_string;

    CHECK(quality_yaml_plain_safe("C major"));
    CHECK(quality_yaml_plain_safe("4/4"));
    CHECK(quality_yaml_plain_safe("d'or"));
    CHECK(!quality_yaml_plain_safe(""));
    CHECK(!quality_yaml_plain_safe("null"));
    CHECK(!quality_yaml_plain_safe("Yes"));
    CHECK(!quality_yaml_plain_safe("120"));
    CHECK(!quality_yaml_plain_safe("-3.5"));
    CHECK(!quality_yaml_plain_safe("key: value"));
    CHECK(!quality_yaml_plain_safe(" padded"));
    CHECK(!quality_yaml_plain_safe("[verse]"));

    CHECK(quality_yaml_string("C major") == "C major");
    CHECK(quality_yaml_string("null") == "'null'");
    CHECK(quality_yaml_string("") == "''");
    CHECK(quality_yaml_string("d''") == "d''");
    CHECK(quality_yaml_string("it's: quoted") == "'it''s: quoted'");
    CHECK(quality_yaml_string("two\nlines") == "'two\n  lines'");
}

void test_quality_targets() {
    using tts_cpp::acestep::quality_caption_target;
    using tts_cpp::acestep::quality_lyrics_target;
    using tts_cpp::acestep::quality_metadata_target;

    CHECK(quality_metadata_target("bpm", 120LL) == "<think>\nbpm: 120\n</think>\n");
    CHECK(quality_metadata_target("keyscale", std::string("C major")) ==
          "<think>\nkeyscale: C major\n</think>\n");
    CHECK(quality_metadata_target("language", std::string("null")) ==
          "<think>\nlanguage: 'null'\n</think>\n");
    CHECK(quality_caption_target("test caption") == "<think>\ncaption: test caption\n</think>\n");
    CHECK(quality_caption_target("") == "<think>\ncaption: ''\n</think>\n");
    CHECK(quality_lyrics_target("[verse]\nhello") == "<think>\n</think>\n# Lyric\n[verse]\nhello\n");

    const std::string long_caption(100, 'x');
    const std::string wrapped = quality_caption_target(long_caption);
    CHECK(wrapped.rfind("<think>\ncaption: ", 0) == 0);
    CHECK(wrapped.find('\n', strlen("<think>\n")) != std::string::npos);
}

void test_quality_encode_target() {
    using tts_cpp::acestep::quality_encode_target;
    using tts_cpp::acestep::TOKEN_THINK;
    using tts_cpp::acestep::TOKEN_THINK_END;

    const tts_cpp::acestep::BpeTokenizer tok = make_test_bpe_tokenizer();
    const std::vector<int> ids = quality_encode_target(tok, "hello<think>hello</think>");
    CHECK(ids == std::vector<int>({ 10, TOKEN_THINK, 10, TOKEN_THINK_END }));
    CHECK(quality_encode_target(tok, "<think></think>") ==
          std::vector<int>({ TOKEN_THINK, TOKEN_THINK_END }));
    CHECK(quality_encode_target(tok, "hello") == std::vector<int>({ 10 }));
}

void test_quality_weighted_global() {
    using tts_cpp::acestep::QualityCondition;
    using tts_cpp::acestep::QualityMetric;
    using tts_cpp::acestep::QualityScoreParams;
    using tts_cpp::acestep::quality_weighted_global;

    std::map<std::string, QualityCondition> conditions;
    conditions["caption"].score  = 0.8;
    conditions["caption"].metric = QualityMetric::PmiNormalized;
    conditions["lyrics"].score   = 0.6;
    conditions["lyrics"].metric  = QualityMetric::PmiNormalized;
    conditions["bpm"].score      = 1.0;
    conditions["keyscale"].score = 0.5;

    QualityScoreParams params;
    double             global = 0.0;
    std::string        report;
    std::string        error;
    CHECK(quality_weighted_global(conditions, params, global, report, error));
    CHECK(quality_near(global, 0.8 * 0.5 + 0.6 * 0.3 + 0.75 * 0.2));
    CHECK(report.find("caption") != std::string::npos);
    CHECK(report.find("Per-condition scores") != std::string::npos);

    conditions.erase("lyrics");
    CHECK(quality_weighted_global(conditions, params, global, report, error));
    CHECK(quality_near(global, (0.8 * 0.5 + 0.75 * 0.2) / 0.7));

    QualityScoreParams zero_weights;
    zero_weights.caption_weight  = 0.0;
    zero_weights.lyrics_weight   = 0.0;
    zero_weights.metadata_weight = 0.0;
    CHECK(!quality_weighted_global(conditions, zero_weights, global, report, error));
    CHECK(!error.empty());
}

void test_quality_score_request_policy() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::RepaintParams;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask   task;
    params.compute_quality_score = true;
    params.caption               = "a caption";
    CHECK(resolve_generate_task(params, task).empty());

    params.task_type = tts_cpp::acestep::TASK_COVER_NOFSQ;
    params.source_audio.assign(96, 0.0f);
    CHECK(resolve_generate_task(params, task).find("LM code path") != std::string::npos);

    params.task_type = tts_cpp::acestep::TASK_LEGO;
    params.track     = "drums";
    CHECK(resolve_generate_task(params, task).find("LM code path") != std::string::npos);

    params.task_type = tts_cpp::acestep::TASK_TEXT2MUSIC;
    params.edit_plan.push_back(RepaintParams{});
    CHECK(resolve_generate_task(params, task).find("audio edit path") != std::string::npos);
}

void add_gate_test_tensor(ggml_context * ctx, gguf_context * weights, const char * name, ggml_type type,
                          int64_t ne0) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, ne0, 2);
    ggml_set_name(t, name);
    std::memset(t->data, 0, ggml_nbytes(t));
    gguf_add_tensor(weights, t);
}

gguf_context * gate_test_weights(ggml_context * ctx, bool with_k_quant) {
    gguf_context * weights = gguf_init_empty();
    add_gate_test_tensor(ctx, weights, "norm.weight", GGML_TYPE_F32, 32);
    add_gate_test_tensor(ctx, weights, "proj.weight", GGML_TYPE_Q8_0, 32);
    if (with_k_quant) add_gate_test_tensor(ctx, weights, "down.weight", GGML_TYPE_Q4_K, 256);
    return weights;
}

void check_hexagon_weight_types() {
    using tts_cpp::acestep::hexagon_runs_weight_type;
    for (const ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
                                  GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4 }) {
        CHECK(hexagon_runs_weight_type(type));
    }
    for (const ggml_type type : { GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                                  GGML_TYPE_Q5_0, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_XS }) {
        CHECK(!hexagon_runs_weight_type(type));
    }
}

void test_hexagon_weight_gate() {
    using tts_cpp::acestep::backend_cpu_init;
    using tts_cpp::acestep::dit_gguf_hexagon_refusal;
    using tts_cpp::acestep::first_tensor_hexagon_cannot_run;
    using tts_cpp::acestep::hexagon_weight_refusal;

    check_hexagon_weight_types();

    ggml_init_params ip{ 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context *   ctx      = ggml_init(ip);
    gguf_context *   runnable = gate_test_weights(ctx, /*with_k_quant=*/false);
    gguf_context *   refused  = gate_test_weights(ctx, /*with_k_quant=*/true);
    CHECK(first_tensor_hexagon_cannot_run(runnable) == -1);
    CHECK(first_tensor_hexagon_cannot_run(refused) == 2);

    const std::string message = hexagon_weight_refusal(refused, 2, "weights.gguf", "acestep-test");
    CHECK(message.rfind("[acestep-test] weights.gguf: down.weight is q4_K; the Hexagon NPU runs ", 0) == 0);
    CHECK(message.find("q8_0") != std::string::npos);
    CHECK(message.find("q4_K,") == std::string::npos);

    ggml_backend_t cpu = backend_cpu_init();
    CHECK(cpu != nullptr);
    if (cpu) {
        CHECK(dit_gguf_hexagon_refusal(refused, "weights.gguf", cpu, "acestep-test").empty());
        ggml_backend_free(cpu);
    }
    gguf_free(runnable);
    gguf_free(refused);
    ggml_free(ctx);
}

struct RegNameOverride {
    explicit RegNameOverride(const char * name) { tts_cpp::acestep::backend_reg_name_override() = name; }
    ~RegNameOverride() { tts_cpp::acestep::backend_reg_name_override() = nullptr; }
};

std::string write_gate_test_gguf() {
    const std::string path = test_temp_dir() + "/qvac-acestep-hexagon-gate-test.gguf";
    ggml_init_params  ip{ 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context *    ctx     = ggml_init(ip);
    gguf_context *    weights = gate_test_weights(ctx, /*with_k_quant=*/true);
    CHECK(gguf_write_to_file(weights, path.c_str(), /*only_meta=*/false));
    gguf_free(weights);
    ggml_free(ctx);
    return path;
}

template <typename Load>
std::string refusal_from(Load && load) {
    try {
        load();
    } catch (const std::runtime_error & e) {
        return e.what();
    }
    return "";
}

void test_hexagon_refusal_paths() {
    using tts_cpp::acestep::backend_cpu_init;
    using tts_cpp::acestep::detok_model_load;
    using tts_cpp::acestep::dit_gguf_open;
    using tts_cpp::acestep::dit_gguf_require_hexagon_weights;
    using tts_cpp::acestep::DitGGUF;
    using tts_cpp::acestep::fit_params;
    using tts_cpp::acestep::FitOptions;
    using tts_cpp::acestep::FitStatus;
    using tts_cpp::acestep::Vae;
    using tts_cpp::acestep::VaeOptions;

    const std::string path    = write_gate_test_gguf();
    const std::string refused = path + ": down.weight is q4_K; the Hexagon NPU runs ";
    ggml_backend_t    cpu     = backend_cpu_init();
    CHECK(cpu != nullptr);
    if (!cpu) return;
    RegNameOverride as_hexagon("HTP");

    DitGGUF g;
    CHECK(dit_gguf_open(g, path));
    const std::string gate = refusal_from([&] { dit_gguf_require_hexagon_weights(g, path, cpu, "acestep-test"); });
    CHECK(gate.rfind("[acestep-test] " + refused, 0) == 0);
    CHECK(g.ctx == nullptr);
    CHECK(g.meta == nullptr);
    CHECK(g.file.data == nullptr);

    const std::string detok = refusal_from([&] { detok_model_load(path, cpu, /*verbose=*/false); });
    CHECK(detok.rfind("[acestep-detok] " + refused, 0) == 0);

    VaeOptions vae_opts;
    vae_opts.backend = "cpu";
    const std::string vae = refusal_from([&] { Vae::load(path, vae_opts); });
    CHECK(vae.rfind("[acestep-vae] " + refused, 0) == 0);

    FitOptions fit;
    fit.backend             = "cpu";
    fit.text_enc_model_path = path;
    fit.lm_model_path       = path;
    fit.dit_model_path      = path;
    fit.vae_model_path      = path;
    fit.duration_seconds    = 8.0f;
    fit.text_tokens         = 32;
    fit.lyric_tokens        = 16;
    const tts_cpp::acestep::FitResult result = fit_params(fit);
    CHECK(result.status == FitStatus::Error);
    CHECK(!result.fits);
    CHECK(result.reason == "measurement-failed");
    CHECK(result.report.rfind("[acestep-fit] " + refused, 0) == 0);

    ggml_backend_free(cpu);
}

}  // namespace

int main() {
    test_schedule();
    test_haar_dcw();
    test_philox();
    test_fsq();
    test_fsq_encode();
    test_understand_prompt();
    test_understand_token_budget();
    test_sampler();
    test_sampler_compact_equivalence();
    test_sampler_r0_edge();
    test_sampler_forced_fast_path();
    test_fsm_forced_token();
    test_sampler_fusion_oracle();
    test_vae_progress();
    test_coreml_sidecar_path();
    test_coreml_plan_windows();
    test_vae_window_core();
    test_backend_device_types();
    test_gpu_fallback_reason();
    test_gpu_tier_policy();
    test_backend_request_matching();
    test_dsp_library_path();
    test_backend_requested_init();
    test_backend_init_first_matching();
    test_lm_backend_request();
    test_fit_pool_sharing();
    test_fit_pools_charge_distinct_lm_device();
    test_fit_pools_shared_budget();
    test_fit_pools_check_every_discrete_device();
    test_weight_placement_on_cpu();
    test_stage_placement();
    test_hexagon_weight_gate();
    test_hexagon_refusal_paths();
    test_placement_env();
    test_vae_backend_request_env();
    test_parallel_rows();
    test_convert_f32_to_f16_rows();
    test_fused_load_fail_closed();
    test_lm_head_chunk_ranges();
    test_lm_split_table_matches_whole_table();
    test_stage_dump_round_trip();
    test_stage_dump_rejects_malformed();
    test_vae_metadata_only_measure();
    test_generate_task_kinds();
    test_generate_task_defaults();
    test_generate_task_audio_layout();
    test_generate_task_errors();
    test_generate_task_strengths();
    test_simple_mode_policy();
    test_simple_mode_prompt_resolvers();
    test_normalize_loudness();
    test_lyrics_matrix_conversion();
    test_lyrics_dtw();
    test_lyrics_preprocessing();
    test_lyrics_metrics_and_score();
    test_lyrics_timestamps_and_lrc();
    test_lrc_request_policy();
    test_inspire_user_message();
    test_format_user_message();
    test_rewrite_query_policy();
    test_cover_conditioning_switch();
    test_generation_plans();
    test_lego_task_kinds();
    test_lego_task_validation();
    test_lego_generation_plan();
    test_lego_model_policy();
    test_guidance_and_dcw_policy();
    test_apg_guide();
    test_generation_conditioning();
    test_cover_noise_blending();
    test_repaint_config();
    test_repaint_range();
    test_repaint_mask_injection_blend_and_splice();
    test_flow_edit_validation_and_math();
    test_audio_edit_factory_and_order();
    test_vae_encode_window_boundaries();
    test_vae_encode_window_parity();
    test_wav_reader_mono_and_stereo();
    test_wav_reader_rejects_multichannel();
    test_quantize_policy();
    test_quantize_policy_mm3();
    test_quantize_gguf_roundtrip();
    test_quantize_gguf_roundtrip_mm3();
    test_cancellation_scope_consumes_on_exit();
    test_bpe_utf8_codepoint();
    test_bpe_encode_merges();
    test_bpe_encode_byte_fallback();
    test_bpe_decode_roundtrip();
    test_quality_normalized_pmi();
    test_quality_yaml_formatting();
    test_quality_targets();
    test_quality_encode_target();
    test_quality_weighted_global();
    test_quality_score_request_policy();

    std::fprintf(stderr, "[test-acestep-units] %d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
