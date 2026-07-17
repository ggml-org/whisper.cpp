#pragma once
// philox.h — Philox4x32-10 PRNG + Box-Muller normal distribution (QVAC-21921).
//
// Matches PyTorch CUDA torch.randn() output (cuRAND Philox4_32_10) so the DiT
// initial noise is reproducible / parity-comparable with acestep.cpp. Ported
// verbatim from acestep.cpp/src/philox.h. Zero deps beyond <cstdint>/<cmath>/
// <cstring>. Kept in the tts_cpp::acestep namespace to avoid symbol clashes.

#include <cmath>
#include <cstdint>
#include <cstring>

namespace tts_cpp::acestep {

static constexpr uint32_t PHILOX_M0 = 0xD2511F53u;
static constexpr uint32_t PHILOX_M1 = 0xCD9E8D57u;
static constexpr uint32_t PHILOX_W0 = 0x9E3779B9u;
static constexpr uint32_t PHILOX_W1 = 0xBB67AE85u;

static constexpr float CURAND_2POW32_INV     = 2.3283064365386963e-10f;  // 1 / 2^32
static constexpr float CURAND_2POW32_INV_2PI = 1.4629180792671596e-09f;  // 2*PI / 2^32

struct Philox4 {
    uint32_t x, y, z, w;
};

static inline void mulhilo32(uint32_t a, uint32_t b, uint32_t * hi, uint32_t * lo) {
    uint64_t prod = (uint64_t) a * (uint64_t) b;
    *lo           = (uint32_t) prod;
    *hi           = (uint32_t) (prod >> 32);
}

static inline Philox4 philox_round(Philox4 ctr, uint32_t k0, uint32_t k1) {
    uint32_t hi0, lo0, hi1, lo1;
    mulhilo32(PHILOX_M0, ctr.x, &hi0, &lo0);
    mulhilo32(PHILOX_M1, ctr.z, &hi1, &lo1);
    return { hi1 ^ ctr.y ^ k0, lo1, hi0 ^ ctr.w ^ k1, lo0 };
}

static inline Philox4 philox4x32_10(Philox4 ctr, uint32_t seed_lo, uint32_t seed_hi) {
    uint32_t k0 = seed_lo;
    uint32_t k1 = seed_hi;
    for (int r = 0; r < 10; r++) {
        ctr = philox_round(ctr, k0, k1);
        if (r < 9) { k0 += PHILOX_W0; k1 += PHILOX_W1; }
    }
    return ctr;
}

static inline void box_muller(uint32_t u0, uint32_t u1, float * n0, float * n1) {
    float u = (float) u0 * CURAND_2POW32_INV + (CURAND_2POW32_INV * 0.5f);
    float v = (float) u1 * CURAND_2POW32_INV_2PI + (CURAND_2POW32_INV_2PI * 0.5f);
    float s = sqrtf(-2.0f * logf(u));
    *n0     = s * sinf(v);
    *n1     = s * cosf(v);
}

static inline void philox_normal4(int64_t seed, int64_t subsequence, int64_t offset, float out[4]) {
    Philox4  ctr = { (uint32_t) (offset), (uint32_t) (offset >> 32), (uint32_t) (subsequence),
                     (uint32_t) (subsequence >> 32) };
    uint32_t slo = (uint32_t) (seed);
    uint32_t shi = (uint32_t) ((uint64_t) seed >> 32);
    Philox4  r   = philox4x32_10(ctr, slo, shi);
    box_muller(r.x, r.y, &out[0], &out[1]);
    box_muller(r.z, r.w, &out[2], &out[3]);
}

static inline float f32_to_bf16_to_f32(float x) {
    uint32_t bits;
    memcpy(&bits, &x, 4);
    bits += 0x7FFF + ((bits >> 16) & 1);  // round-to-nearest-even
    bits &= 0xFFFF0000u;
    float y;
    memcpy(&y, &bits, 4);
    return y;
}

// Fill `out` with n N(0,1) samples matching torch.randn() on CUDA (bf16 dtype).
static inline void philox_randn(int64_t seed, float * out, int n, bool bf16_round = true) {
    for (int k = 0; k < n; k++) {
        float vals[4];
        philox_normal4(seed, k, 0, vals);
        out[k] = bf16_round ? f32_to_bf16_to_f32(vals[0]) : vals[0];
    }
}

} // namespace tts_cpp::acestep
