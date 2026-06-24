/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "../SDL_internal.h"

#include "SDL_audio.h"
#include "SDL_audio_c.h"
#include "SDL_cpuinfo.h"

#ifdef __ARM_NEON
#define HAVE_NEON_INTRINSICS 1
#endif

#ifdef __SSE2__
#define HAVE_SSE2_INTRINSICS
#endif

#if defined(__x86_64__) && defined(HAVE_SSE2_INTRINSICS)
#define NEED_SCALAR_CONVERTER_FALLBACKS 0  /* x86_64 guarantees SSE2. */
#elif defined(__MACOSX__) && defined(HAVE_SSE2_INTRINSICS)
#define NEED_SCALAR_CONVERTER_FALLBACKS 0  /* Mac OS X/Intel guarantees SSE2. */
#elif defined(__ARM_ARCH) && (__ARM_ARCH >= 8) && defined(HAVE_NEON_INTRINSICS)
#define NEED_SCALAR_CONVERTER_FALLBACKS 0 /* ARMv8+ promise NEON. */
#elif defined(__APPLE__) && defined(__ARM_ARCH) && (__ARM_ARCH >= 7) && defined(HAVE_NEON_INTRINSICS)
#define NEED_SCALAR_CONVERTER_FALLBACKS 0 /* All Apple ARMv7 chips promise NEON support. */
#endif

/* Set to zero if platform is guaranteed to use a SIMD codepath here. */
#ifndef NEED_SCALAR_CONVERTER_FALLBACKS
#define NEED_SCALAR_CONVERTER_FALLBACKS 1
#endif

/* Function pointers set to a CPU-specific implementation. */
SDL_AudioFilter SDL_Convert_S8_to_F32 = NULL;
SDL_AudioFilter SDL_Convert_U8_to_F32 = NULL;
SDL_AudioFilter SDL_Convert_S16_to_F32 = NULL;
SDL_AudioFilter SDL_Convert_U16_to_F32 = NULL;
SDL_AudioFilter SDL_Convert_S32_to_F32 = NULL;
SDL_AudioFilter SDL_Convert_F32_to_S8 = NULL;
SDL_AudioFilter SDL_Convert_F32_to_U8 = NULL;
SDL_AudioFilter SDL_Convert_F32_to_S16 = NULL;
SDL_AudioFilter SDL_Convert_F32_to_U16 = NULL;
SDL_AudioFilter SDL_Convert_F32_to_S32 = NULL;

#define DIVBY128     0.0078125f
#define DIVBY32768   0.000030517578125f
#define DIVBY8388607 0.00000011920930376163766f
#define DIVBY2147483648 0.0000000004656612873077392578125f /* 0x1p-31f */

#if NEED_SCALAR_CONVERTER_FALLBACKS

/* This code requires that floats are in the IEEE-754 binary32 format */
SDL_COMPILE_TIME_ASSERT(float_bits, sizeof(float) == sizeof(Uint32));

union float_bits {
    Uint32 u32;
    float f32;
};

/* Create a bit-mask based on the sign-bit. Should optimize to a single arithmetic-shift-right */
#define SIGNMASK(x) (Uint32)(0u - ((Uint32)(x) >> 31))

static void SDLCALL SDL_Convert_S8_to_F32_Scalar(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const int num_samples = cvt->len_cvt;
    const Sint8 *src = (const Sint8 *)cvt->buf;
    float *dst = (float *)cvt->buf;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_S8", "AUDIO_F32");

    for (i = num_samples - 1; i >= 0; --i) {
        /* 1) Construct a float in the range [65536.0, 65538.0)
         * 2) Shift the float range to [-1.0, 1.0) */
        union float_bits x;
        x.u32 = (Uint8)src[i] ^ 0x47800080u;
        dst[i] = x.f32 - 65537.0f;
    }

    cvt->len_cvt *= 4;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_F32SYS);
    }
}

static void SDLCALL SDL_Convert_U8_to_F32_Scalar(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const int num_samples = cvt->len_cvt;
    const Uint8 *src = (const Uint8 *)cvt->buf;
    float *dst = (float *)cvt->buf;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_U8", "AUDIO_F32");

    for (i = num_samples - 1; i >= 0; --i) {
        /* 1) Construct a float in the range [65536.0, 65538.0)
         * 2) Shift the float range to [-1.0, 1.0) */
        union float_bits x;
        x.u32 = src[i] ^ 0x47800000u;
        dst[i] = x.f32 - 65537.0f;
    }

    cvt->len_cvt *= 4;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_F32SYS);
    }
}

static void SDLCALL SDL_Convert_S16_to_F32_Scalar(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const int num_samples = cvt->len_cvt / sizeof(Sint16);
    const Sint16 *src = (const Sint16 *)cvt->buf;
    float *dst = (float *)cvt->buf;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_S16", "AUDIO_F32");

    for (i = num_samples - 1; i >= 0; --i) {
        /* 1) Construct a float in the range [256.0, 258.0)
         * 2) Shift the float range to [-1.0, 1.0) */
        union float_bits x;
        x.u32 = (Uint16)src[i] ^ 0x43808000u;
        dst[i] = x.f32 - 257.0f;
    }

    cvt->len_cvt *= 2;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_F32SYS);
    }
}

static void SDLCALL SDL_Convert_U16_to_F32_Scalar(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const Uint16 *src = ((const Uint16 *)(cvt->buf + cvt->len_cvt)) - 1;
    float *dst = ((float *)(cvt->buf + cvt->len_cvt * 2)) - 1;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_U16", "AUDIO_F32");

    for (i = cvt->len_cvt / sizeof(Uint16); i; --i, --src, --dst) {
        *dst = (((float)*src) * DIVBY32768) - 1.0f;
    }

    cvt->len_cvt *= 2;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_F32SYS);
    }
}

static void SDLCALL SDL_Convert_S32_to_F32_Scalar(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const Sint32 *src = (const Sint32 *)cvt->buf;
    float *dst = (float *)cvt->buf;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_S32", "AUDIO_F32");

    for (i = cvt->len_cvt / sizeof(Sint32); i; --i, ++src, ++dst) {
        *dst = ((float)(*src >> 8)) * DIVBY8388607;
    }

    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_F32SYS);
    }
}

static void SDLCALL SDL_Convert_F32_to_S8_Scalar(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const int num_samples = cvt->len_cvt / sizeof (float);
    const float *src = (const float *)cvt->buf;
    Sint8 *dst = (Sint8 *)cvt->buf;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_F32", "AUDIO_S8");

    for (i = 0; i < num_samples; ++i) {
        /* 1) Shift the float range from [-1.0, 1.0] to [98303.0, 98305.0]
         * 2) Shift the integer range from [0x47BFFF80, 0x47C00080] to [-128, 128]
         * 3) Clamp the value to [-128, 127] */
        union float_bits x;
        Uint32 y, z;
        x.f32 = src[i] + 98304.0f;

        y = x.u32 - 0x47C00000u;
        z = 0x7Fu - (y ^ SIGNMASK(y));
        y = y ^ (z & SIGNMASK(z));

        dst[i] = (Sint8)(y & 0xFF);
    }

    cvt->len_cvt /= 4;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_S8);
    }
}

static void SDLCALL SDL_Convert_F32_to_U8_Scalar(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const int num_samples = cvt->len_cvt / sizeof (float);
    const float *src = (const float *)cvt->buf;
    Uint8 *dst = (Uint8 *)cvt->buf;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_F32", "AUDIO_U8");

    for (i = 0; i < num_samples; ++i) {
        /* 1) Shift the float range from [-1.0, 1.0] to [98303.0, 98305.0]
         * 2) Shift the integer range from [0x47BFFF80, 0x47C00080] to [-128, 128]
         * 3) Clamp the value to [-128, 127]
         * 4) Shift the integer range from [-128, 127] to [0, 255] */
        union float_bits x;
        Uint32 y, z;
        x.f32 = src[i] + 98304.0f;

        y = x.u32 - 0x47C00000u;
        z = 0x7Fu - (y ^ SIGNMASK(y));
        y = (y ^ 0x80u) ^ (z & SIGNMASK(z));

        dst[i] = (Uint8)(y & 0xFF);
    }

    cvt->len_cvt /= 4;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_U8);
    }
}

static void SDLCALL SDL_Convert_F32_to_S16_Scalar(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const int num_samples = cvt->len_cvt / sizeof (float);
    const float *src = (const float *)cvt->buf;
    Sint16 *dst = (Sint16 *)cvt->buf;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_F32", "AUDIO_S16");

    for (i = 0; i < num_samples; ++i) {
        /* 1) Shift the float range from [-1.0, 1.0] to [383.0, 385.0]
         * 2) Shift the integer range from [0x43BF8000, 0x43C08000] to [-32768, 32768]
         * 3) Clamp values outside the [-32768, 32767] range */
        union float_bits x;
        Uint32 y, z;
        x.f32 = src[i] + 384.0f;

        y = x.u32 - 0x43C00000u;
        z = 0x7FFFu - (y ^ SIGNMASK(y));
        y = y ^ (z & SIGNMASK(z));

        dst[i] = (Sint16)(y & 0xFFFF);
    }

    cvt->len_cvt /= 2;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_S16SYS);
    }
}

static void SDLCALL SDL_Convert_F32_to_U16_Scalar(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const float *src = (const float *)cvt->buf;
    Uint16 *dst = (Uint16 *)cvt->buf;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_F32", "AUDIO_U16");

    for (i = cvt->len_cvt / sizeof(float); i; --i, ++src, ++dst) {
        const float sample = *src;
        if (sample >= 1.0f) {
            *dst = 65535;
        } else if (sample <= -1.0f) {
            *dst = 0;
        } else {
            *dst = (Uint16)((sample + 1.0f) * 32767.0f);
        }
    }

    cvt->len_cvt /= 2;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_U16SYS);
    }
}

static void SDLCALL SDL_Convert_F32_to_S32_Scalar(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const int num_samples = cvt->len_cvt / sizeof (float);
    const float *src = (const float *)cvt->buf;
    Sint32 *dst = (Sint32 *)cvt->buf;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_F32", "AUDIO_S32");

    for (i = 0; i < num_samples; ++i) {
        /* 1) Shift the float range from [-1.0, 1.0] to [-2147483648.0, 2147483648.0]
         * 2) Set values outside the [-2147483648.0, 2147483647.0] range to -2147483648.0
         * 3) Convert the float to an integer, and fixup values outside the valid range */
        union float_bits x;
        Uint32 y, z;
        x.f32 = src[i];

        y = x.u32 + 0x0F800000u;
        z = y - 0xCF000000u;
        z &= SIGNMASK(y ^ z);
        x.u32 = y - z;

        dst[i] = (Sint32)x.f32 ^ (Sint32)SIGNMASK(z);
    }

    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_S32SYS);
    }
}
#endif

#ifdef HAVE_SSE2_INTRINSICS
static void SDLCALL SDL_Convert_S8_to_F32_SSE2(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const Sint8 *src = (const Sint8 *)cvt->buf;
    float *dst = (float *)cvt->buf;
    int i = cvt->len_cvt;

    /* 1) Flip the sign bit to convert from S8 to U8 format
     * 2) Construct a float in the range [65536.0, 65538.0)
     * 3) Shift the float range to [-1.0, 1.0)
     * dst[i] = i2f((src[i] ^ 0x80) | 0x47800000) - 65537.0 */
    const __m128i zero = _mm_setzero_si128();
    const __m128i flipper = _mm_set1_epi8(-0x80);
    const __m128i caster = _mm_set1_epi16(0x4780 /* 0x47800000 = f2i(65536.0) */);
    const __m128 offset = _mm_set1_ps(-65537.0);

    LOG_DEBUG_CONVERT("AUDIO_S8", "AUDIO_F32 (using SSE2)");

    while (i >= 16) {
        i -= 16;

        {
        const __m128i bytes = _mm_xor_si128(_mm_loadu_si128((const __m128i *)&src[i]), flipper);

        const __m128i shorts1 = _mm_unpacklo_epi8(bytes, zero);
        const __m128i shorts2 = _mm_unpackhi_epi8(bytes, zero);

        const __m128 floats1 = _mm_add_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(shorts1, caster)), offset);
        const __m128 floats2 = _mm_add_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(shorts1, caster)), offset);
        const __m128 floats3 = _mm_add_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(shorts2, caster)), offset);
        const __m128 floats4 = _mm_add_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(shorts2, caster)), offset);

        _mm_storeu_ps(&dst[i], floats1);
        _mm_storeu_ps(&dst[i + 4], floats2);
        _mm_storeu_ps(&dst[i + 8], floats3);
        _mm_storeu_ps(&dst[i + 12], floats4);
        }
    }

    while (i) {
        --i;
        _mm_store_ss(&dst[i], _mm_add_ss(_mm_castsi128_ps(_mm_cvtsi32_si128((Uint8)src[i] ^ 0x47800080u)), offset));
    }

    cvt->len_cvt *= 4;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_F32SYS);
    }
}

static void SDLCALL SDL_Convert_U8_to_F32_SSE2(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const Sint8 *src = (const Sint8 *)cvt->buf;
    float *dst = (float *)cvt->buf;
    int i = cvt->len_cvt;

    /* 1) Construct a float in the range [65536.0, 65538.0)
     * 2) Shift the float range to [-1.0, 1.0)
     * dst[i] = i2f(src[i] | 0x47800000) - 65537.0 */
    const __m128i zero = _mm_setzero_si128();
    const __m128i caster = _mm_set1_epi16(0x4780 /* 0x47800000 = f2i(65536.0) */);
    const __m128 offset = _mm_set1_ps(-65537.0);

    LOG_DEBUG_CONVERT("AUDIO_U8", "AUDIO_F32 (using SSE2)");

    while (i >= 16) {
        i -= 16;

        {
        const __m128i bytes = _mm_loadu_si128((const __m128i *)&src[i]);

        const __m128i shorts1 = _mm_unpacklo_epi8(bytes, zero);
        const __m128i shorts2 = _mm_unpackhi_epi8(bytes, zero);

        const __m128 floats1 = _mm_add_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(shorts1, caster)), offset);
        const __m128 floats2 = _mm_add_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(shorts1, caster)), offset);
        const __m128 floats3 = _mm_add_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(shorts2, caster)), offset);
        const __m128 floats4 = _mm_add_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(shorts2, caster)), offset);

        _mm_storeu_ps(&dst[i], floats1);
        _mm_storeu_ps(&dst[i + 4], floats2);
        _mm_storeu_ps(&dst[i + 8], floats3);
        _mm_storeu_ps(&dst[i + 12], floats4);
        }
    }

    while (i) {
        --i;
        _mm_store_ss(&dst[i], _mm_add_ss(_mm_castsi128_ps(_mm_cvtsi32_si128((Uint8)src[i] ^ 0x47800000u)), offset));
    }

    cvt->len_cvt *= 4;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_F32SYS);
    }
}

static void SDLCALL SDL_Convert_S16_to_F32_SSE2(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const Sint16 *src = (const Sint16 *)cvt->buf;
    float *dst = (float *)cvt->buf;
    int i = cvt->len_cvt / 2;

    /* 1) Flip the sign bit to convert from S16 to U16 format
     * 2) Construct a float in the range [256.0, 258.0)
     * 3) Shift the float range to [-1.0, 1.0)
     * dst[i] = i2f((src[i] ^ 0x8000) | 0x43800000) - 257.0 */
    const __m128i flipper = _mm_set1_epi16(-0x8000);
    const __m128i caster = _mm_set1_epi16(0x4380 /* 0x43800000 = f2i(256.0) */);
    const __m128 offset = _mm_set1_ps(-257.0f);

    LOG_DEBUG_CONVERT("AUDIO_S16", "AUDIO_F32 (using SSE2)");

    while (i >= 16) {
        i -= 16;

        {
        const __m128i shorts1 = _mm_xor_si128(_mm_loadu_si128((const __m128i *)&src[i]), flipper);
        const __m128i shorts2 = _mm_xor_si128(_mm_loadu_si128((const __m128i *)&src[i + 8]), flipper);

        const __m128 floats1 = _mm_add_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(shorts1, caster)), offset);
        const __m128 floats2 = _mm_add_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(shorts1, caster)), offset);
        const __m128 floats3 = _mm_add_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(shorts2, caster)), offset);
        const __m128 floats4 = _mm_add_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(shorts2, caster)), offset);

        _mm_storeu_ps(&dst[i], floats1);
        _mm_storeu_ps(&dst[i + 4], floats2);
        _mm_storeu_ps(&dst[i + 8], floats3);
        _mm_storeu_ps(&dst[i + 12], floats4);
        }
    }

    while (i) {
        --i;
        _mm_store_ss(&dst[i], _mm_add_ss(_mm_castsi128_ps(_mm_cvtsi32_si128((Uint16)src[i] ^ 0x43808000u)), offset));
    }

    cvt->len_cvt *= 2;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_F32SYS);
    }
}

static void SDLCALL SDL_Convert_U16_to_F32_SSE2(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const Uint16 *src = ((const Uint16 *)(cvt->buf + cvt->len_cvt)) - 1;
    float *dst = ((float *)(cvt->buf + cvt->len_cvt * 2)) - 1;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_U16", "AUDIO_F32 (using SSE2)");

    /* Get dst aligned to 16 bytes (since buffer is growing, we don't have to worry about overreading from src) */
    for (i = cvt->len_cvt / sizeof(Sint16); i && (((size_t)(dst - 7)) & 15); --i, --src, --dst) {
        *dst = (((float)*src) * DIVBY32768) - 1.0f;
    }

    src -= 7;
    dst -= 7; /* adjust to read SSE blocks from the start. */
    SDL_assert(!i || !(((size_t)dst) & 15));

    /* Make sure src is aligned too. */
    if (!(((size_t)src) & 15)) {
        /* Aligned! Do SSE blocks as long as we have 16 bytes available. */
        const __m128 divby32768 = _mm_set1_ps(DIVBY32768);
        const __m128 minus1 = _mm_set1_ps(-1.0f);
        while (i >= 8) {                                               /* 8 * 16-bit */
            const __m128i ints = _mm_load_si128((__m128i const *)src); /* get 8 sint16 into an XMM register. */
            /* treat as int32, shift left to clear every other sint16, then back right with zero-extend. Now sint32. */
            const __m128i a = _mm_srli_epi32(_mm_slli_epi32(ints, 16), 16);
            /* right-shift-sign-extend gets us sint32 with the other set of values. */
            const __m128i b = _mm_srli_epi32(ints, 16);
            /* Interleave these back into the right order, convert to float, multiply, store. */
            _mm_store_ps(dst, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi32(a, b)), divby32768), minus1));
            _mm_store_ps(dst + 4, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi32(a, b)), divby32768), minus1));
            i -= 8;
            src -= 8;
            dst -= 8;
        }
    }

    src += 7;
    dst += 7; /* adjust for any scalar finishing. */

    /* Finish off any leftovers with scalar operations. */
    while (i) {
        *dst = (((float)*src) * DIVBY32768) - 1.0f;
        i--;
        src--;
        dst--;
    }

    cvt->len_cvt *= 2;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_F32SYS);
    }
}

static void SDLCALL SDL_Convert_S32_to_F32_SSE2(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const Sint32 *src = (const Sint32 *)cvt->buf;
    float *dst = (float *)cvt->buf;
    int i = cvt->len_cvt / 4;

    /* dst[i] = f32(src[i]) / f32(0x80000000) */
    const __m128 scaler = _mm_set1_ps(DIVBY2147483648);

    LOG_DEBUG_CONVERT("AUDIO_S32", "AUDIO_F32 (using SSE2)");

    while (i >= 16) {
        i -= 16;

        {
        const __m128i ints1 = _mm_loadu_si128((const __m128i *)&src[i]);
        const __m128i ints2 = _mm_loadu_si128((const __m128i *)&src[i + 4]);
        const __m128i ints3 = _mm_loadu_si128((const __m128i *)&src[i + 8]);
        const __m128i ints4 = _mm_loadu_si128((const __m128i *)&src[i + 12]);

        const __m128 floats1 = _mm_mul_ps(_mm_cvtepi32_ps(ints1), scaler);
        const __m128 floats2 = _mm_mul_ps(_mm_cvtepi32_ps(ints2), scaler);
        const __m128 floats3 = _mm_mul_ps(_mm_cvtepi32_ps(ints3), scaler);
        const __m128 floats4 = _mm_mul_ps(_mm_cvtepi32_ps(ints4), scaler);

        _mm_storeu_ps(&dst[i], floats1);
        _mm_storeu_ps(&dst[i + 4], floats2);
        _mm_storeu_ps(&dst[i + 8], floats3);
        _mm_storeu_ps(&dst[i + 12], floats4);
        }
    }

    while (i) {
        --i;
        _mm_store_ss(&dst[i], _mm_mul_ss(_mm_cvt_si2ss(_mm_setzero_ps(), src[i]), scaler));
    }

    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_F32SYS);
    }
}

static void SDLCALL SDL_Convert_F32_to_S8_SSE2(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const float *src = (const float *)cvt->buf;
    Sint8 *dst = (Sint8 *)cvt->buf;
    int i = cvt->len_cvt / 4;

    /* 1) Shift the float range from [-1.0, 1.0] to [98303.0, 98305.0]
     * 2) Extract the lowest 16 bits and clamp to [-128, 127]
     * Overflow is correctly handled for inputs between roughly [-255.0, 255.0]
     * dst[i] = clamp(i16(f2i(src[i] + 98304.0) & 0xFFFF), -128, 127) */
    const __m128 offset = _mm_set1_ps(98304.0f);
    const __m128i mask = _mm_set1_epi16(0xFF);

    LOG_DEBUG_CONVERT("AUDIO_F32", "AUDIO_S8 (using SSE2)");

    while (i >= 16) {
        const __m128 floats1 = _mm_loadu_ps(&src[0]);
        const __m128 floats2 = _mm_loadu_ps(&src[4]);
        const __m128 floats3 = _mm_loadu_ps(&src[8]);
        const __m128 floats4 = _mm_loadu_ps(&src[12]);

        const __m128i ints1 = _mm_castps_si128(_mm_add_ps(floats1, offset));
        const __m128i ints2 = _mm_castps_si128(_mm_add_ps(floats2, offset));
        const __m128i ints3 = _mm_castps_si128(_mm_add_ps(floats3, offset));
        const __m128i ints4 = _mm_castps_si128(_mm_add_ps(floats4, offset));

        const __m128i shorts1 = _mm_and_si128(_mm_packs_epi16(ints1, ints2), mask);
        const __m128i shorts2 = _mm_and_si128(_mm_packs_epi16(ints3, ints4), mask);

        const __m128i bytes = _mm_packus_epi16(shorts1, shorts2);

        _mm_storeu_si128((__m128i*)dst, bytes);

        i -= 16;
        src += 16;
        dst += 16;
    }

    while (i) {
        const __m128i ints = _mm_castps_si128(_mm_add_ss(_mm_load_ss(src), offset));
        *dst = (Sint8)(_mm_cvtsi128_si32(_mm_packs_epi16(ints, ints)) & 0xFF);

        --i;
        ++src;
        ++dst;
    }

    cvt->len_cvt /= 4;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_S8);
    }
}

static void SDLCALL SDL_Convert_F32_to_U8_SSE2(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const float *src = (const float *)cvt->buf;
    Uint8 *dst = cvt->buf;
    int i = cvt->len_cvt / 4;

    /* 1) Shift the float range from [-1.0, 1.0] to [98304.0, 98306.0]
     * 2) Extract the lowest 16 bits and clamp to [0, 255]
     * Overflow is correctly handled for inputs between roughly [-254.0, 254.0]
     * dst[i] = clamp(i16(f2i(src[i] + 98305.0) & 0xFFFF), 0, 255) */
    const __m128 offset = _mm_set1_ps(98305.0f);
    const __m128i mask = _mm_set1_epi16(0xFF);

    LOG_DEBUG_CONVERT("AUDIO_F32", "AUDIO_U8 (using SSE2)");

    while (i >= 16) {
        const __m128 floats1 = _mm_loadu_ps(&src[0]);
        const __m128 floats2 = _mm_loadu_ps(&src[4]);
        const __m128 floats3 = _mm_loadu_ps(&src[8]);
        const __m128 floats4 = _mm_loadu_ps(&src[12]);

        const __m128i ints1 = _mm_castps_si128(_mm_add_ps(floats1, offset));
        const __m128i ints2 = _mm_castps_si128(_mm_add_ps(floats2, offset));
        const __m128i ints3 = _mm_castps_si128(_mm_add_ps(floats3, offset));
        const __m128i ints4 = _mm_castps_si128(_mm_add_ps(floats4, offset));

        const __m128i shorts1 = _mm_and_si128(_mm_packus_epi16(ints1, ints2), mask);
        const __m128i shorts2 = _mm_and_si128(_mm_packus_epi16(ints3, ints4), mask);

        const __m128i bytes = _mm_packus_epi16(shorts1, shorts2);

        _mm_storeu_si128((__m128i*)dst, bytes);

        i -= 16;
        src += 16;
        dst += 16;
    }

    while (i) {
        const __m128i ints = _mm_castps_si128(_mm_add_ss(_mm_load_ss(src), offset));
        *dst = (Uint8)(_mm_cvtsi128_si32(_mm_packus_epi16(ints, ints)) & 0xFF);

        --i;
        ++src;
        ++dst;
    }

    cvt->len_cvt /= 4;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_U8);
    }
}

static void SDLCALL SDL_Convert_F32_to_S16_SSE2(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const float *src = (const float *)cvt->buf;
    Sint16 *dst = (Sint16 *)cvt->buf;
    int i = cvt->len_cvt / 4;

    /* 1) Shift the float range from [-1.0, 1.0] to [256.0, 258.0]
     * 2) Shift the int range from [0x43800000, 0x43810000] to [-32768,32768]
     * 3) Clamp to range [-32768,32767]
     * Overflow is correctly handled for inputs between roughly [-257.0, +inf)
     * dst[i] = clamp(f2i(src[i] + 257.0) - 0x43808000, -32768, 32767) */
    const __m128 offset = _mm_set1_ps(257.0f);

    LOG_DEBUG_CONVERT("AUDIO_F32", "AUDIO_S16 (using SSE2)");

    while (i >= 16) {
        const __m128 floats1 = _mm_loadu_ps(&src[0]);
        const __m128 floats2 = _mm_loadu_ps(&src[4]);
        const __m128 floats3 = _mm_loadu_ps(&src[8]);
        const __m128 floats4 = _mm_loadu_ps(&src[12]);

        const __m128i ints1 = _mm_sub_epi32(_mm_castps_si128(_mm_add_ps(floats1, offset)), _mm_castps_si128(offset));
        const __m128i ints2 = _mm_sub_epi32(_mm_castps_si128(_mm_add_ps(floats2, offset)), _mm_castps_si128(offset));
        const __m128i ints3 = _mm_sub_epi32(_mm_castps_si128(_mm_add_ps(floats3, offset)), _mm_castps_si128(offset));
        const __m128i ints4 = _mm_sub_epi32(_mm_castps_si128(_mm_add_ps(floats4, offset)), _mm_castps_si128(offset));

        const __m128i shorts1 = _mm_packs_epi32(ints1, ints2);
        const __m128i shorts2 = _mm_packs_epi32(ints3, ints4);

        _mm_storeu_si128((__m128i*)&dst[0], shorts1);
        _mm_storeu_si128((__m128i*)&dst[8], shorts2);

        i -= 16;
        src += 16;
        dst += 16;
    }

    while (i) {
        const __m128i ints = _mm_sub_epi32(_mm_castps_si128(_mm_add_ss(_mm_load_ss(src), offset)), _mm_castps_si128(offset));
        *dst = (Sint16)(_mm_cvtsi128_si32(_mm_packs_epi32(ints, ints)) & 0xFFFF);

        --i;
        ++src;
        ++dst;
    }

    cvt->len_cvt /= 2;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_S16SYS);
    }
}

static void SDLCALL SDL_Convert_F32_to_U16_SSE2(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const float *src = (const float *)cvt->buf;
    Uint16 *dst = (Uint16 *)cvt->buf;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_F32", "AUDIO_U16 (using SSE2)");

    /* Get dst aligned to 16 bytes */
    for (i = cvt->len_cvt / sizeof(float); i && (((size_t)dst) & 15); --i, ++src, ++dst) {
        const float sample = *src;
        if (sample >= 1.0f) {
            *dst = 65535;
        } else if (sample <= -1.0f) {
            *dst = 0;
        } else {
            *dst = (Uint16)((sample + 1.0f) * 32767.0f);
        }
    }

    SDL_assert(!i || !(((size_t)dst) & 15));

    /* Make sure src is aligned too. */
    if (!(((size_t)src) & 15)) {
        /* Aligned! Do SSE blocks as long as we have 16 bytes available. */
        /* This calculates differently than the scalar path because SSE2 can't
           pack int32 data down to unsigned int16. _mm_packs_epi32 does signed
           saturation, so that would corrupt our data. _mm_packus_epi32 exists,
           but not before SSE 4.1. So we convert from float to sint16, packing
           that down with legit signed saturation, and then xor the top bit
           against 1. This results in the correct unsigned 16-bit value, even
           though it looks like dark magic. */
        const __m128 mulby32767 = _mm_set1_ps(32767.0f);
        const __m128i topbit = _mm_set1_epi16(-32768);
        const __m128 one = _mm_set1_ps(1.0f);
        const __m128 negone = _mm_set1_ps(-1.0f);
        __m128i *mmdst = (__m128i *)dst;
        while (i >= 8) {                                                                                                              /* 8 * float32 */
            const __m128i ints1 = _mm_cvtps_epi32(_mm_mul_ps(_mm_min_ps(_mm_max_ps(negone, _mm_load_ps(src)), one), mulby32767));     /* load 4 floats, clamp, convert to sint32 */
            const __m128i ints2 = _mm_cvtps_epi32(_mm_mul_ps(_mm_min_ps(_mm_max_ps(negone, _mm_load_ps(src + 4)), one), mulby32767)); /* load 4 floats, clamp, convert to sint32 */
            _mm_store_si128(mmdst, _mm_xor_si128(_mm_packs_epi32(ints1, ints2), topbit));                                             /* pack to sint16, xor top bit, store out. */
            i -= 8;
            src += 8;
            mmdst++;
        }
        dst = (Uint16 *)mmdst;
    }

    /* Finish off any leftovers with scalar operations. */
    while (i) {
        const float sample = *src;
        if (sample >= 1.0f) {
            *dst = 65535;
        } else if (sample <= -1.0f) {
            *dst = 0;
        } else {
            *dst = (Uint16)((sample + 1.0f) * 32767.0f);
        }
        i--;
        src++;
        dst++;
    }

    cvt->len_cvt /= 2;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_U16SYS);
    }
}

static void SDLCALL SDL_Convert_F32_to_S32_SSE2(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const float *src = (const float *)cvt->buf;
    Sint32 *dst = (Sint32 *)cvt->buf;
    int i = cvt->len_cvt / 4;

    /* 1) Scale the float range from [-1.0, 1.0] to [-2147483648.0, 2147483648.0]
     * 2) Convert to integer (values too small/large become 0x80000000 = -2147483648)
     * 3) Fixup values which were too large (0x80000000 ^ 0xFFFFFFFF = 2147483647)
     * dst[i] = i32(src[i] * 2147483648.0) ^ ((src[i] >= 2147483648.0) ? 0xFFFFFFFF : 0x00000000) */
    const __m128 limit = _mm_set1_ps(2147483648.0f);

    LOG_DEBUG_CONVERT("AUDIO_F32", "AUDIO_S32 (using SSE2)");

    while (i >= 16) {
        const __m128 floats1 = _mm_loadu_ps(&src[0]);
        const __m128 floats2 = _mm_loadu_ps(&src[4]);
        const __m128 floats3 = _mm_loadu_ps(&src[8]);
        const __m128 floats4 = _mm_loadu_ps(&src[12]);

        const __m128 values1 = _mm_mul_ps(floats1, limit);
        const __m128 values2 = _mm_mul_ps(floats2, limit);
        const __m128 values3 = _mm_mul_ps(floats3, limit);
        const __m128 values4 = _mm_mul_ps(floats4, limit);

        const __m128i ints1 = _mm_xor_si128(_mm_cvttps_epi32(values1), _mm_castps_si128(_mm_cmpge_ps(values1, limit)));
        const __m128i ints2 = _mm_xor_si128(_mm_cvttps_epi32(values2), _mm_castps_si128(_mm_cmpge_ps(values2, limit)));
        const __m128i ints3 = _mm_xor_si128(_mm_cvttps_epi32(values3), _mm_castps_si128(_mm_cmpge_ps(values3, limit)));
        const __m128i ints4 = _mm_xor_si128(_mm_cvttps_epi32(values4), _mm_castps_si128(_mm_cmpge_ps(values4, limit)));

        _mm_storeu_si128((__m128i*)&dst[0], ints1);
        _mm_storeu_si128((__m128i*)&dst[4], ints2);
        _mm_storeu_si128((__m128i*)&dst[8], ints3);
        _mm_storeu_si128((__m128i*)&dst[12], ints4);

        i -= 16;
        src += 16;
        dst += 16;
    }

    while (i) {
        const __m128 floats = _mm_load_ss(src);
        const __m128 values = _mm_mul_ss(floats, limit);
        const __m128i ints = _mm_xor_si128(_mm_cvttps_epi32(values), _mm_castps_si128(_mm_cmpge_ss(values, limit)));
        *dst = (Sint32)_mm_cvtsi128_si32(ints);

        --i;
        ++src;
        ++dst;
    }

    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_S32SYS);
    }
}
#endif

#ifdef HAVE_NEON_INTRINSICS
static void SDLCALL SDL_Convert_S8_to_F32_NEON(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const Sint8 *src = ((const Sint8 *)(cvt->buf + cvt->len_cvt)) - 1;
    float *dst = ((float *)(cvt->buf + cvt->len_cvt * 4)) - 1;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_S8", "AUDIO_F32 (using NEON)");

    /* Get dst aligned to 16 bytes (since buffer is growing, we don't have to worry about overreading from src) */
    for (i = cvt->len_cvt; i && (((size_t)(dst - 15)) & 15); --i, --src, --dst) {
        *dst = ((float)*src) * DIVBY128;
    }

    src -= 15;
    dst -= 15; /* adjust to read NEON blocks from the start. */
    SDL_assert(!i || !(((size_t)dst) & 15));

    /* Make sure src is aligned too. */
    if (!(((size_t)src) & 15)) {
        /* Aligned! Do NEON blocks as long as we have 16 bytes available. */
        const int8_t *mmsrc = (const int8_t *)src;
        const float32x4_t divby128 = vdupq_n_f32(DIVBY128);
        while (i >= 16) {                                            /* 16 * 8-bit */
            const int8x16_t bytes = vld1q_s8(mmsrc);                 /* get 16 sint8 into a NEON register. */
            const int16x8_t int16hi = vmovl_s8(vget_high_s8(bytes)); /* convert top 8 bytes to 8 int16 */
            const int16x8_t int16lo = vmovl_s8(vget_low_s8(bytes));  /* convert bottom 8 bytes to 8 int16 */
            /* split int16 to two int32, then convert to float, then multiply to normalize, store. */
            vst1q_f32(dst, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(int16lo))), divby128));
            vst1q_f32(dst + 4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(int16lo))), divby128));
            vst1q_f32(dst + 8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(int16hi))), divby128));
            vst1q_f32(dst + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(int16hi))), divby128));
            i -= 16;
            mmsrc -= 16;
            dst -= 16;
        }

        src = (const Sint8 *)mmsrc;
    }

    src += 15;
    dst += 15; /* adjust for any scalar finishing. */

    /* Finish off any leftovers with scalar operations. */
    while (i) {
        *dst = ((float)*src) * DIVBY128;
        i--;
        src--;
        dst--;
    }

    cvt->len_cvt *= 4;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_F32SYS);
    }
}

static void SDLCALL SDL_Convert_U8_to_F32_NEON(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const Uint8 *src = ((const Uint8 *)(cvt->buf + cvt->len_cvt)) - 1;
    float *dst = ((float *)(cvt->buf + cvt->len_cvt * 4)) - 1;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_U8", "AUDIO_F32 (using NEON)");

    /* Get dst aligned to 16 bytes (since buffer is growing, we don't have to worry about overreading from src) */
    for (i = cvt->len_cvt; i && (((size_t)(dst - 15)) & 15); --i, --src, --dst) {
        *dst = (((float)*src) * DIVBY128) - 1.0f;
    }

    src -= 15;
    dst -= 15; /* adjust to read NEON blocks from the start. */
    SDL_assert(!i || !(((size_t)dst) & 15));

    /* Make sure src is aligned too. */
    if (!(((size_t)src) & 15)) {
        /* Aligned! Do NEON blocks as long as we have 16 bytes available. */
        const uint8_t *mmsrc = (const uint8_t *)src;
        const float32x4_t divby128 = vdupq_n_f32(DIVBY128);
        const float32x4_t negone = vdupq_n_f32(-1.0f);
        while (i >= 16) {                                              /* 16 * 8-bit */
            const uint8x16_t bytes = vld1q_u8(mmsrc);                  /* get 16 uint8 into a NEON register. */
            const uint16x8_t uint16hi = vmovl_u8(vget_high_u8(bytes)); /* convert top 8 bytes to 8 uint16 */
            const uint16x8_t uint16lo = vmovl_u8(vget_low_u8(bytes));  /* convert bottom 8 bytes to 8 uint16 */
            /* split uint16 to two uint32, then convert to float, then multiply to normalize, subtract to adjust for sign, store. */
            vst1q_f32(dst, vmlaq_f32(negone, vcvtq_f32_u32(vmovl_u16(vget_low_u16(uint16lo))), divby128));
            vst1q_f32(dst + 4, vmlaq_f32(negone, vcvtq_f32_u32(vmovl_u16(vget_high_u16(uint16lo))), divby128));
            vst1q_f32(dst + 8, vmlaq_f32(negone, vcvtq_f32_u32(vmovl_u16(vget_low_u16(uint16hi))), divby128));
            vst1q_f32(dst + 12, vmlaq_f32(negone, vcvtq_f32_u32(vmovl_u16(vget_high_u16(uint16hi))), divby128));
            i -= 16;
            mmsrc -= 16;
            dst -= 16;
        }

        src = (const Uint8 *)mmsrc;
    }

    src += 15;
    dst += 15; /* adjust for any scalar finishing. */

    /* Finish off any leftovers with scalar operations. */
    while (i) {
        *dst = (((float)*src) * DIVBY128) - 1.0f;
        i--;
        src--;
        dst--;
    }

    cvt->len_cvt *= 4;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_F32SYS);
    }
}

static void SDLCALL SDL_Convert_S16_to_F32_NEON(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const Sint16 *src = ((const Sint16 *)(cvt->buf + cvt->len_cvt)) - 1;
    float *dst = ((float *)(cvt->buf + cvt->len_cvt * 2)) - 1;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_S16", "AUDIO_F32 (using NEON)");

    /* Get dst aligned to 16 bytes (since buffer is growing, we don't have to worry about overreading from src) */
    for (i = cvt->len_cvt / sizeof(Sint16); i && (((size_t)(dst - 7)) & 15); --i, --src, --dst) {
        *dst = ((float)*src) * DIVBY32768;
    }

    src -= 7;
    dst -= 7; /* adjust to read NEON blocks from the start. */
    SDL_assert(!i || !(((size_t)dst) & 15));

    /* Make sure src is aligned too. */
    if (!(((size_t)src) & 15)) {
        /* Aligned! Do NEON blocks as long as we have 16 bytes available. */
        const float32x4_t divby32768 = vdupq_n_f32(DIVBY32768);
        while (i >= 8) {                                            /* 8 * 16-bit */
            const int16x8_t ints = vld1q_s16((int16_t const *)src); /* get 8 sint16 into a NEON register. */
            /* split int16 to two int32, then convert to float, then multiply to normalize, store. */
            vst1q_f32(dst, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(ints))), divby32768));
            vst1q_f32(dst + 4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(ints))), divby32768));
            i -= 8;
            src -= 8;
            dst -= 8;
        }
    }

    src += 7;
    dst += 7; /* adjust for any scalar finishing. */

    /* Finish off any leftovers with scalar operations. */
    while (i) {
        *dst = ((float)*src) * DIVBY32768;
        i--;
        src--;
        dst--;
    }

    cvt->len_cvt *= 2;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_F32SYS);
    }
}

static void SDLCALL SDL_Convert_U16_to_F32_NEON(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const Uint16 *src = ((const Uint16 *)(cvt->buf + cvt->len_cvt)) - 1;
    float *dst = ((float *)(cvt->buf + cvt->len_cvt * 2)) - 1;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_U16", "AUDIO_F32 (using NEON)");

    /* Get dst aligned to 16 bytes (since buffer is growing, we don't have to worry about overreading from src) */
    for (i = cvt->len_cvt / sizeof(Sint16); i && (((size_t)(dst - 7)) & 15); --i, --src, --dst) {
        *dst = (((float)*src) * DIVBY32768) - 1.0f;
    }

    src -= 7;
    dst -= 7; /* adjust to read NEON blocks from the start. */
    SDL_assert(!i || !(((size_t)dst) & 15));

    /* Make sure src is aligned too. */
    if (!(((size_t)src) & 15)) {
        /* Aligned! Do NEON blocks as long as we have 16 bytes available. */
        const float32x4_t divby32768 = vdupq_n_f32(DIVBY32768);
        const float32x4_t negone = vdupq_n_f32(-1.0f);
        while (i >= 8) {                                               /* 8 * 16-bit */
            const uint16x8_t uints = vld1q_u16((uint16_t const *)src); /* get 8 uint16 into a NEON register. */
            /* split uint16 to two int32, then convert to float, then multiply to normalize, subtract for sign, store. */
            vst1q_f32(dst, vmlaq_f32(negone, vcvtq_f32_u32(vmovl_u16(vget_low_u16(uints))), divby32768));
            vst1q_f32(dst + 4, vmlaq_f32(negone, vcvtq_f32_u32(vmovl_u16(vget_high_u16(uints))), divby32768));
            i -= 8;
            src -= 8;
            dst -= 8;
        }
    }

    src += 7;
    dst += 7; /* adjust for any scalar finishing. */

    /* Finish off any leftovers with scalar operations. */
    while (i) {
        *dst = (((float)*src) * DIVBY32768) - 1.0f;
        i--;
        src--;
        dst--;
    }

    cvt->len_cvt *= 2;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_F32SYS);
    }
}

static void SDLCALL SDL_Convert_S32_to_F32_NEON(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const Sint32 *src = (const Sint32 *)cvt->buf;
    float *dst = (float *)cvt->buf;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_S32", "AUDIO_F32 (using NEON)");

    /* Get dst aligned to 16 bytes */
    for (i = cvt->len_cvt / sizeof(Sint32); i && (((size_t)dst) & 15); --i, ++src, ++dst) {
        *dst = ((float)(*src >> 8)) * DIVBY8388607;
    }

    SDL_assert(!i || !(((size_t)dst) & 15));

    /* Make sure src is aligned too. */
    if (!(((size_t)src) & 15)) {
        /* Aligned! Do NEON blocks as long as we have 16 bytes available. */
        const float32x4_t divby8388607 = vdupq_n_f32(DIVBY8388607);
        const int32_t *mmsrc = (const int32_t *)src;
        while (i >= 4) { /* 4 * sint32 */
            /* shift out lowest bits so int fits in a float32. Small precision loss, but much faster. */
            vst1q_f32(dst, vmulq_f32(vcvtq_f32_s32(vshrq_n_s32(vld1q_s32(mmsrc), 8)), divby8388607));
            i -= 4;
            mmsrc += 4;
            dst += 4;
        }
        src = (const Sint32 *)mmsrc;
    }

    /* Finish off any leftovers with scalar operations. */
    while (i) {
        *dst = ((float)(*src >> 8)) * DIVBY8388607;
        i--;
        src++;
        dst++;
    }

    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_F32SYS);
    }
}

static void SDLCALL SDL_Convert_F32_to_S8_NEON(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const float *src = (const float *)cvt->buf;
    Sint8 *dst = (Sint8 *)cvt->buf;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_F32", "AUDIO_S8 (using NEON)");

    /* Get dst aligned to 16 bytes */
    for (i = cvt->len_cvt / sizeof(float); i && (((size_t)dst) & 15); --i, ++src, ++dst) {
        const float sample = *src;
        if (sample >= 1.0f) {
            *dst = 127;
        } else if (sample <= -1.0f) {
            *dst = -128;
        } else {
            *dst = (Sint8)(sample * 127.0f);
        }
    }

    SDL_assert(!i || !(((size_t)dst) & 15));

    /* Make sure src is aligned too. */
    if (!(((size_t)src) & 15)) {
        /* Aligned! Do NEON blocks as long as we have 16 bytes available. */
        const float32x4_t one = vdupq_n_f32(1.0f);
        const float32x4_t negone = vdupq_n_f32(-1.0f);
        const float32x4_t mulby127 = vdupq_n_f32(127.0f);
        int8_t *mmdst = (int8_t *)dst;
        while (i >= 16) {                                                                                                       /* 16 * float32 */
            const int32x4_t ints1 = vcvtq_s32_f32(vmulq_f32(vminq_f32(vmaxq_f32(negone, vld1q_f32(src)), one), mulby127));      /* load 4 floats, clamp, convert to sint32 */
            const int32x4_t ints2 = vcvtq_s32_f32(vmulq_f32(vminq_f32(vmaxq_f32(negone, vld1q_f32(src + 4)), one), mulby127));  /* load 4 floats, clamp, convert to sint32 */
            const int32x4_t ints3 = vcvtq_s32_f32(vmulq_f32(vminq_f32(vmaxq_f32(negone, vld1q_f32(src + 8)), one), mulby127));  /* load 4 floats, clamp, convert to sint32 */
            const int32x4_t ints4 = vcvtq_s32_f32(vmulq_f32(vminq_f32(vmaxq_f32(negone, vld1q_f32(src + 12)), one), mulby127)); /* load 4 floats, clamp, convert to sint32 */
            const int8x8_t i8lo = vmovn_s16(vcombine_s16(vmovn_s32(ints1), vmovn_s32(ints2)));                                  /* narrow to sint16, combine, narrow to sint8 */
            const int8x8_t i8hi = vmovn_s16(vcombine_s16(vmovn_s32(ints3), vmovn_s32(ints4)));                                  /* narrow to sint16, combine, narrow to sint8 */
            vst1q_s8(mmdst, vcombine_s8(i8lo, i8hi));                                                                           /* combine to int8x16_t, store out */
            i -= 16;
            src += 16;
            mmdst += 16;
        }
        dst = (Sint8 *)mmdst;
    }

    /* Finish off any leftovers with scalar operations. */
    while (i) {
        const float sample = *src;
        if (sample >= 1.0f) {
            *dst = 127;
        } else if (sample <= -1.0f) {
            *dst = -128;
        } else {
            *dst = (Sint8)(sample * 127.0f);
        }
        i--;
        src++;
        dst++;
    }

    cvt->len_cvt /= 4;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_S8);
    }
}

static void SDLCALL SDL_Convert_F32_to_U8_NEON(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const float *src = (const float *)cvt->buf;
    Uint8 *dst = cvt->buf;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_F32", "AUDIO_U8 (using NEON)");

    /* Get dst aligned to 16 bytes */
    for (i = cvt->len_cvt / sizeof(float); i && (((size_t)dst) & 15); --i, ++src, ++dst) {
        const float sample = *src;
        if (sample >= 1.0f) {
            *dst = 255;
        } else if (sample <= -1.0f) {
            *dst = 0;
        } else {
            *dst = (Uint8)((sample + 1.0f) * 127.0f);
        }
    }

    SDL_assert(!i || !(((size_t)dst) & 15));

    /* Make sure src is aligned too. */
    if (!(((size_t)src) & 15)) {
        /* Aligned! Do NEON blocks as long as we have 16 bytes available. */
        const float32x4_t one = vdupq_n_f32(1.0f);
        const float32x4_t negone = vdupq_n_f32(-1.0f);
        const float32x4_t mulby127 = vdupq_n_f32(127.0f);
        uint8_t *mmdst = (uint8_t *)dst;
        while (i >= 16) {                                                                                                                         /* 16 * float32 */
            const uint32x4_t uints1 = vcvtq_u32_f32(vmulq_f32(vaddq_f32(vminq_f32(vmaxq_f32(negone, vld1q_f32(src)), one), one), mulby127));      /* load 4 floats, clamp, convert to uint32 */
            const uint32x4_t uints2 = vcvtq_u32_f32(vmulq_f32(vaddq_f32(vminq_f32(vmaxq_f32(negone, vld1q_f32(src + 4)), one), one), mulby127));  /* load 4 floats, clamp, convert to uint32 */
            const uint32x4_t uints3 = vcvtq_u32_f32(vmulq_f32(vaddq_f32(vminq_f32(vmaxq_f32(negone, vld1q_f32(src + 8)), one), one), mulby127));  /* load 4 floats, clamp, convert to uint32 */
            const uint32x4_t uints4 = vcvtq_u32_f32(vmulq_f32(vaddq_f32(vminq_f32(vmaxq_f32(negone, vld1q_f32(src + 12)), one), one), mulby127)); /* load 4 floats, clamp, convert to uint32 */
            const uint8x8_t ui8lo = vmovn_u16(vcombine_u16(vmovn_u32(uints1), vmovn_u32(uints2)));                                                /* narrow to uint16, combine, narrow to uint8 */
            const uint8x8_t ui8hi = vmovn_u16(vcombine_u16(vmovn_u32(uints3), vmovn_u32(uints4)));                                                /* narrow to uint16, combine, narrow to uint8 */
            vst1q_u8(mmdst, vcombine_u8(ui8lo, ui8hi));                                                                                           /* combine to uint8x16_t, store out */
            i -= 16;
            src += 16;
            mmdst += 16;
        }

        dst = (Uint8 *)mmdst;
    }

    /* Finish off any leftovers with scalar operations. */
    while (i) {
        const float sample = *src;
        if (sample >= 1.0f) {
            *dst = 255;
        } else if (sample <= -1.0f) {
            *dst = 0;
        } else {
            *dst = (Uint8)((sample + 1.0f) * 127.0f);
        }
        i--;
        src++;
        dst++;
    }

    cvt->len_cvt /= 4;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_U8);
    }
}

static void SDLCALL SDL_Convert_F32_to_S16_NEON(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const float *src = (const float *)cvt->buf;
    Sint16 *dst = (Sint16 *)cvt->buf;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_F32", "AUDIO_S16 (using NEON)");

    /* Get dst aligned to 16 bytes */
    for (i = cvt->len_cvt / sizeof(float); i && (((size_t)dst) & 15); --i, ++src, ++dst) {
        const float sample = *src;
        if (sample >= 1.0f) {
            *dst = 32767;
        } else if (sample <= -1.0f) {
            *dst = -32768;
        } else {
            *dst = (Sint16)(sample * 32767.0f);
        }
    }

    SDL_assert(!i || !(((size_t)dst) & 15));

    /* Make sure src is aligned too. */
    if (!(((size_t)src) & 15)) {
        /* Aligned! Do NEON blocks as long as we have 16 bytes available. */
        const float32x4_t one = vdupq_n_f32(1.0f);
        const float32x4_t negone = vdupq_n_f32(-1.0f);
        const float32x4_t mulby32767 = vdupq_n_f32(32767.0f);
        int16_t *mmdst = (int16_t *)dst;
        while (i >= 8) {                                                                                                         /* 8 * float32 */
            const int32x4_t ints1 = vcvtq_s32_f32(vmulq_f32(vminq_f32(vmaxq_f32(negone, vld1q_f32(src)), one), mulby32767));     /* load 4 floats, clamp, convert to sint32 */
            const int32x4_t ints2 = vcvtq_s32_f32(vmulq_f32(vminq_f32(vmaxq_f32(negone, vld1q_f32(src + 4)), one), mulby32767)); /* load 4 floats, clamp, convert to sint32 */
            vst1q_s16(mmdst, vcombine_s16(vmovn_s32(ints1), vmovn_s32(ints2)));                                                  /* narrow to sint16, combine, store out. */
            i -= 8;
            src += 8;
            mmdst += 8;
        }
        dst = (Sint16 *)mmdst;
    }

    /* Finish off any leftovers with scalar operations. */
    while (i) {
        const float sample = *src;
        if (sample >= 1.0f) {
            *dst = 32767;
        } else if (sample <= -1.0f) {
            *dst = -32768;
        } else {
            *dst = (Sint16)(sample * 32767.0f);
        }
        i--;
        src++;
        dst++;
    }

    cvt->len_cvt /= 2;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_S16SYS);
    }
}

static void SDLCALL SDL_Convert_F32_to_U16_NEON(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const float *src = (const float *)cvt->buf;
    Uint16 *dst = (Uint16 *)cvt->buf;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_F32", "AUDIO_U16 (using NEON)");

    /* Get dst aligned to 16 bytes */
    for (i = cvt->len_cvt / sizeof(float); i && (((size_t)dst) & 15); --i, ++src, ++dst) {
        const float sample = *src;
        if (sample >= 1.0f) {
            *dst = 65535;
        } else if (sample <= -1.0f) {
            *dst = 0;
        } else {
            *dst = (Uint16)((sample + 1.0f) * 32767.0f);
        }
    }

    SDL_assert(!i || !(((size_t)dst) & 15));

    /* Make sure src is aligned too. */
    if (!(((size_t)src) & 15)) {
        /* Aligned! Do NEON blocks as long as we have 16 bytes available. */
        const float32x4_t one = vdupq_n_f32(1.0f);
        const float32x4_t negone = vdupq_n_f32(-1.0f);
        const float32x4_t mulby32767 = vdupq_n_f32(32767.0f);
        uint16_t *mmdst = (uint16_t *)dst;
        while (i >= 8) {                                                                                                                           /* 8 * float32 */
            const uint32x4_t uints1 = vcvtq_u32_f32(vmulq_f32(vaddq_f32(vminq_f32(vmaxq_f32(negone, vld1q_f32(src)), one), one), mulby32767));     /* load 4 floats, clamp, convert to uint32 */
            const uint32x4_t uints2 = vcvtq_u32_f32(vmulq_f32(vaddq_f32(vminq_f32(vmaxq_f32(negone, vld1q_f32(src + 4)), one), one), mulby32767)); /* load 4 floats, clamp, convert to uint32 */
            vst1q_u16(mmdst, vcombine_u16(vmovn_u32(uints1), vmovn_u32(uints2)));                                                                  /* narrow to uint16, combine, store out. */
            i -= 8;
            src += 8;
            mmdst += 8;
        }
        dst = (Uint16 *)mmdst;
    }

    /* Finish off any leftovers with scalar operations. */
    while (i) {
        const float sample = *src;
        if (sample >= 1.0f) {
            *dst = 65535;
        } else if (sample <= -1.0f) {
            *dst = 0;
        } else {
            *dst = (Uint16)((sample + 1.0f) * 32767.0f);
        }
        i--;
        src++;
        dst++;
    }

    cvt->len_cvt /= 2;
    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_U16SYS);
    }
}

static void SDLCALL SDL_Convert_F32_to_S32_NEON(SDL_AudioCVT *cvt, SDL_AudioFormat format)
{
    const float *src = (const float *)cvt->buf;
    Sint32 *dst = (Sint32 *)cvt->buf;
    int i;

    LOG_DEBUG_CONVERT("AUDIO_F32", "AUDIO_S32 (using NEON)");

    /* Get dst aligned to 16 bytes */
    for (i = cvt->len_cvt / sizeof(float); i && (((size_t)dst) & 15); --i, ++src, ++dst) {
        const float sample = *src;
        if (sample >= 1.0f) {
            *dst = 2147483647;
        } else if (sample <= -1.0f) {
            *dst = (-2147483647) - 1;
        } else {
            *dst = ((Sint32)(sample * 8388607.0f)) << 8;
        }
    }

    SDL_assert(!i || !(((size_t)dst) & 15));
    SDL_assert(!i || !(((size_t)src) & 15));

    {
        /* Aligned! Do NEON blocks as long as we have 16 bytes available. */
        const float32x4_t one = vdupq_n_f32(1.0f);
        const float32x4_t negone = vdupq_n_f32(-1.0f);
        const float32x4_t mulby8388607 = vdupq_n_f32(8388607.0f);
        int32_t *mmdst = (int32_t *)dst;
        while (i >= 4) { /* 4 * float32 */
            vst1q_s32(mmdst, vshlq_n_s32(vcvtq_s32_f32(vmulq_f32(vminq_f32(vmaxq_f32(negone, vld1q_f32(src)), one), mulby8388607)), 8));
            i -= 4;
            src += 4;
            mmdst += 4;
        }
        dst = (Sint32 *)mmdst;
    }

    /* Finish off any leftovers with scalar operations. */
    while (i) {
        const float sample = *src;
        if (sample >= 1.0f) {
            *dst = 2147483647;
        } else if (sample <= -1.0f) {
            *dst = (-2147483647) - 1;
        } else {
            *dst = ((Sint32)(sample * 8388607.0f)) << 8;
        }
        i--;
        src++;
        dst++;
    }

    if (cvt->filters[++cvt->filter_index]) {
        cvt->filters[cvt->filter_index](cvt, AUDIO_S32SYS);
    }
}
#endif

void SDL_ChooseAudioConverters(void)
{
    static SDL_bool converters_chosen = SDL_FALSE;

    if (converters_chosen) {
        return;
    }

#define SET_CONVERTER_FUNCS(fntype)                           \
    SDL_Convert_S8_to_F32 = SDL_Convert_S8_to_F32_##fntype;   \
    SDL_Convert_U8_to_F32 = SDL_Convert_U8_to_F32_##fntype;   \
    SDL_Convert_S16_to_F32 = SDL_Convert_S16_to_F32_##fntype; \
    SDL_Convert_U16_to_F32 = SDL_Convert_U16_to_F32_##fntype; \
    SDL_Convert_S32_to_F32 = SDL_Convert_S32_to_F32_##fntype; \
    SDL_Convert_F32_to_S8 = SDL_Convert_F32_to_S8_##fntype;   \
    SDL_Convert_F32_to_U8 = SDL_Convert_F32_to_U8_##fntype;   \
    SDL_Convert_F32_to_S16 = SDL_Convert_F32_to_S16_##fntype; \
    SDL_Convert_F32_to_U16 = SDL_Convert_F32_to_U16_##fntype; \
    SDL_Convert_F32_to_S32 = SDL_Convert_F32_to_S32_##fntype; \
    converters_chosen = SDL_TRUE

#ifdef HAVE_SSE2_INTRINSICS
    if (SDL_HasSSE2()) {
        SET_CONVERTER_FUNCS(SSE2);
        return;
    }
#endif

#ifdef HAVE_NEON_INTRINSICS
    if (SDL_HasNEON()) {
        SET_CONVERTER_FUNCS(NEON);
        return;
    }
#endif

#if NEED_SCALAR_CONVERTER_FALLBACKS
    SET_CONVERTER_FUNCS(Scalar);
#endif

#undef SET_CONVERTER_FUNCS

    SDL_assert(converters_chosen == SDL_TRUE);
}

/* vi: set ts=4 sw=4 expandtab: */
