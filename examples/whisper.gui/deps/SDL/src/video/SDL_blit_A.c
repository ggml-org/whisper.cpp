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

#if SDL_HAVE_BLIT_A

#include "SDL_video.h"
#include "SDL_blit.h"

/* Functions to perform alpha blended blitting */

/* N->1 blending with per-surface alpha */
static void BlitNto1SurfaceAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    Uint8 *palmap = info->table;
    SDL_PixelFormat *srcfmt = info->src_fmt;
    SDL_PixelFormat *dstfmt = info->dst_fmt;
    int srcbpp = srcfmt->BytesPerPixel;
    Uint32 Pixel;
    unsigned sR, sG, sB;
    unsigned dR, dG, dB;
    const unsigned A = info->a;

    while (height--) {
        /* *INDENT-OFF* */ /* clang-format off */
        DUFFS_LOOP4(
        {
        DISEMBLE_RGB(src, srcbpp, srcfmt, Pixel, sR, sG, sB);
        dR = dstfmt->palette->colors[*dst].r;
        dG = dstfmt->palette->colors[*dst].g;
        dB = dstfmt->palette->colors[*dst].b;
        ALPHA_BLEND_RGB(sR, sG, sB, A, dR, dG, dB);
        dR &= 0xff;
        dG &= 0xff;
        dB &= 0xff;
        /* Pack RGB into 8bit pixel */
        if ( palmap == NULL ) {
            *dst =((dR>>5)<<(3+2))|((dG>>5)<<(2))|((dB>>6)<<(0));
        } else {
            *dst = palmap[((dR>>5)<<(3+2))|((dG>>5)<<(2))|((dB>>6)<<(0))];
        }
        dst++;
        src += srcbpp;
        },
        width);
        /* *INDENT-ON* */ /* clang-format on */
        src += srcskip;
        dst += dstskip;
    }
}

/* N->1 blending with pixel alpha */
static void BlitNto1PixelAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    Uint8 *palmap = info->table;
    SDL_PixelFormat *srcfmt = info->src_fmt;
    SDL_PixelFormat *dstfmt = info->dst_fmt;
    int srcbpp = srcfmt->BytesPerPixel;
    Uint32 Pixel;
    unsigned sR, sG, sB, sA;
    unsigned dR, dG, dB;

    while (height--) {
        /* *INDENT-OFF* */ /* clang-format off */
        DUFFS_LOOP4(
        {
        DISEMBLE_RGBA(src,srcbpp,srcfmt,Pixel,sR,sG,sB,sA);
        dR = dstfmt->palette->colors[*dst].r;
        dG = dstfmt->palette->colors[*dst].g;
        dB = dstfmt->palette->colors[*dst].b;
        ALPHA_BLEND_RGB(sR, sG, sB, sA, dR, dG, dB);
        dR &= 0xff;
        dG &= 0xff;
        dB &= 0xff;
        /* Pack RGB into 8bit pixel */
        if ( palmap == NULL ) {
            *dst =((dR>>5)<<(3+2))|((dG>>5)<<(2))|((dB>>6)<<(0));
        } else {
            *dst = palmap[((dR>>5)<<(3+2))|((dG>>5)<<(2))|((dB>>6)<<(0))];
        }
        dst++;
        src += srcbpp;
        },
        width);
        /* *INDENT-ON* */ /* clang-format on */
        src += srcskip;
        dst += dstskip;
    }
}

/* colorkeyed N->1 blending with per-surface alpha */
static void BlitNto1SurfaceAlphaKey(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    Uint8 *palmap = info->table;
    SDL_PixelFormat *srcfmt = info->src_fmt;
    SDL_PixelFormat *dstfmt = info->dst_fmt;
    int srcbpp = srcfmt->BytesPerPixel;
    Uint32 ckey = info->colorkey;
    Uint32 Pixel;
    unsigned sR, sG, sB;
    unsigned dR, dG, dB;
    const unsigned A = info->a;

    while (height--) {
        /* *INDENT-OFF* */ /* clang-format off */
        DUFFS_LOOP(
        {
        DISEMBLE_RGB(src, srcbpp, srcfmt, Pixel, sR, sG, sB);
        if ( Pixel != ckey ) {
            dR = dstfmt->palette->colors[*dst].r;
            dG = dstfmt->palette->colors[*dst].g;
            dB = dstfmt->palette->colors[*dst].b;
            ALPHA_BLEND_RGB(sR, sG, sB, A, dR, dG, dB);
            dR &= 0xff;
            dG &= 0xff;
            dB &= 0xff;
            /* Pack RGB into 8bit pixel */
            if ( palmap == NULL ) {
                *dst =((dR>>5)<<(3+2))|((dG>>5)<<(2))|((dB>>6)<<(0));
            } else {
                *dst = palmap[((dR>>5)<<(3+2))|((dG>>5)<<(2))|((dB>>6)<<(0))];
            }
        }
        dst++;
        src += srcbpp;
        },
        width);
        /* *INDENT-ON* */ /* clang-format on */
        src += srcskip;
        dst += dstskip;
    }
}

#ifdef __MMX__

/* fast RGB888->(A)RGB888 blending with surface alpha=128 special case */
static void BlitRGBtoRGBSurfaceAlpha128MMX(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint32 *srcp = (Uint32 *)info->src;
    int srcskip = info->src_skip >> 2;
    Uint32 *dstp = (Uint32 *)info->dst;
    int dstskip = info->dst_skip >> 2;
    Uint32 dalpha = info->dst_fmt->Amask;

    __m64 src1, src2, dst1, dst2, lmask, hmask, dsta;

    hmask = _mm_set_pi32(0x00fefefe, 0x00fefefe); /* alpha128 mask -> hmask */
    lmask = _mm_set_pi32(0x00010101, 0x00010101); /* !alpha128 mask -> lmask */
    dsta = _mm_set_pi32(dalpha, dalpha);          /* dst alpha mask -> dsta */

    while (height--) {
        int n = width;
        if (n & 1) {
            Uint32 s = *srcp++;
            Uint32 d = *dstp;
            *dstp++ = ((((s & 0x00fefefe) + (d & 0x00fefefe)) >> 1) + (s & d & 0x00010101)) | dalpha;
            n--;
        }

        for (n >>= 1; n > 0; --n) {
            dst1 = *(__m64 *)dstp; /* 2 x dst -> dst1(ARGBARGB) */
            dst2 = dst1;           /* 2 x dst -> dst2(ARGBARGB) */

            src1 = *(__m64 *)srcp; /* 2 x src -> src1(ARGBARGB) */
            src2 = src1;           /* 2 x src -> src2(ARGBARGB) */

            dst2 = _mm_and_si64(dst2, hmask); /* dst & mask -> dst2 */
            src2 = _mm_and_si64(src2, hmask); /* src & mask -> src2 */
            src2 = _mm_add_pi32(src2, dst2);  /* dst2 + src2 -> src2 */
            src2 = _mm_srli_pi32(src2, 1);    /* src2 >> 1 -> src2 */

            dst1 = _mm_and_si64(dst1, src1);  /* src & dst -> dst1 */
            dst1 = _mm_and_si64(dst1, lmask); /* dst1 & !mask -> dst1 */
            dst1 = _mm_add_pi32(dst1, src2);  /* src2 + dst1 -> dst1 */
            dst1 = _mm_or_si64(dst1, dsta);   /* dsta(full alpha) | dst1 -> dst1 */

            *(__m64 *)dstp = dst1; /* dst1 -> 2 x dst pixels */
            dstp += 2;
            srcp += 2;
        }

        srcp += srcskip;
        dstp += dstskip;
    }
    _mm_empty();
}

/* fast RGB888->(A)RGB888 blending with surface alpha */
static void BlitRGBtoRGBSurfaceAlphaMMX(SDL_BlitInfo *info)
{
    SDL_PixelFormat *df = info->dst_fmt;
    Uint32 chanmask;
    unsigned alpha = info->a;

    if (alpha == 128 && (df->Rmask | df->Gmask | df->Bmask) == 0x00FFFFFF) {
        /* only call a128 version when R,G,B occupy lower bits */
        BlitRGBtoRGBSurfaceAlpha128MMX(info);
    } else {
        int width = info->dst_w;
        int height = info->dst_h;
        Uint32 *srcp = (Uint32 *)info->src;
        int srcskip = info->src_skip >> 2;
        Uint32 *dstp = (Uint32 *)info->dst;
        int dstskip = info->dst_skip >> 2;
        Uint32 dalpha = df->Amask;
        Uint32 amult;

        __m64 src1, src2, dst1, dst2, mm_alpha, mm_zero, dsta;

        mm_zero = _mm_setzero_si64(); /* 0 -> mm_zero */
        /* form the alpha mult */
        amult = alpha | (alpha << 8);
        amult = amult | (amult << 16);
        chanmask =
            (0xff << df->Rshift) | (0xff << df->Gshift) | (0xff << df->Bshift);
        mm_alpha = _mm_set_pi32(0, amult & chanmask);   /* 0000AAAA -> mm_alpha, minus 1 chan */
        mm_alpha = _mm_unpacklo_pi8(mm_alpha, mm_zero); /* 0A0A0A0A -> mm_alpha, minus 1 chan */
        /* at this point mm_alpha can be 000A0A0A or 0A0A0A00 or another combo */
        dsta = _mm_set_pi32(dalpha, dalpha); /* dst alpha mask -> dsta */

        while (height--) {
            int n = width;
            if (n & 1) {
                /* One Pixel Blend */
                src2 = _mm_cvtsi32_si64(*srcp);         /* src(ARGB) -> src2 (0000ARGB) */
                src2 = _mm_unpacklo_pi8(src2, mm_zero); /* 0A0R0G0B -> src2 */

                dst1 = _mm_cvtsi32_si64(*dstp);         /* dst(ARGB) -> dst1 (0000ARGB) */
                dst1 = _mm_unpacklo_pi8(dst1, mm_zero); /* 0A0R0G0B -> dst1 */

                src2 = _mm_sub_pi16(src2, dst1);       /* src2 - dst2 -> src2 */
                src2 = _mm_mullo_pi16(src2, mm_alpha); /* src2 * alpha -> src2 */
                src2 = _mm_srli_pi16(src2, 8);         /* src2 >> 8 -> src2 */
                dst1 = _mm_add_pi8(src2, dst1);        /* src2 + dst1 -> dst1 */

                dst1 = _mm_packs_pu16(dst1, mm_zero); /* 0000ARGB -> dst1 */
                dst1 = _mm_or_si64(dst1, dsta);       /* dsta | dst1 -> dst1 */
                *dstp = _mm_cvtsi64_si32(dst1);       /* dst1 -> pixel */

                ++srcp;
                ++dstp;

                n--;
            }

            for (n >>= 1; n > 0; --n) {
                /* Two Pixels Blend */
                src1 = *(__m64 *)srcp;                  /* 2 x src -> src1(ARGBARGB) */
                src2 = src1;                            /* 2 x src -> src2(ARGBARGB) */
                src1 = _mm_unpacklo_pi8(src1, mm_zero); /* low - 0A0R0G0B -> src1 */
                src2 = _mm_unpackhi_pi8(src2, mm_zero); /* high - 0A0R0G0B -> src2 */

                dst1 = *(__m64 *)dstp;                  /* 2 x dst -> dst1(ARGBARGB) */
                dst2 = dst1;                            /* 2 x dst -> dst2(ARGBARGB) */
                dst1 = _mm_unpacklo_pi8(dst1, mm_zero); /* low - 0A0R0G0B -> dst1 */
                dst2 = _mm_unpackhi_pi8(dst2, mm_zero); /* high - 0A0R0G0B -> dst2 */

                src1 = _mm_sub_pi16(src1, dst1);       /* src1 - dst1 -> src1 */
                src1 = _mm_mullo_pi16(src1, mm_alpha); /* src1 * alpha -> src1 */
                src1 = _mm_srli_pi16(src1, 8);         /* src1 >> 8 -> src1 */
                dst1 = _mm_add_pi8(src1, dst1);        /* src1 + dst1(dst1) -> dst1 */

                src2 = _mm_sub_pi16(src2, dst2);       /* src2 - dst2 -> src2 */
                src2 = _mm_mullo_pi16(src2, mm_alpha); /* src2 * alpha -> src2 */
                src2 = _mm_srli_pi16(src2, 8);         /* src2 >> 8 -> src2 */
                dst2 = _mm_add_pi8(src2, dst2);        /* src2 + dst2(dst2) -> dst2 */

                dst1 = _mm_packs_pu16(dst1, dst2); /* 0A0R0G0B(res1), 0A0R0G0B(res2) -> dst1(ARGBARGB) */
                dst1 = _mm_or_si64(dst1, dsta);    /* dsta | dst1 -> dst1 */

                *(__m64 *)dstp = dst1; /* dst1 -> 2 x pixel */

                srcp += 2;
                dstp += 2;
            }
            srcp += srcskip;
            dstp += dstskip;
        }
        _mm_empty();
    }
}

/* fast ARGB888->(A)RGB888 blending with pixel alpha */
static void BlitRGBtoRGBPixelAlphaMMX(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint32 *srcp = (Uint32 *)info->src;
    int srcskip = info->src_skip >> 2;
    Uint32 *dstp = (Uint32 *)info->dst;
    int dstskip = info->dst_skip >> 2;
    SDL_PixelFormat *sf = info->src_fmt;
    Uint32 amask = sf->Amask;
    Uint32 ashift = sf->Ashift;
    Uint64 multmask, multmask2;

    __m64 src1, dst1, mm_alpha, mm_zero, mm_alpha2, mm_one_alpha;

    mm_zero = _mm_setzero_si64(); /* 0 -> mm_zero */
    if (amask == 0xFF000000) { /* 1 in the alpha channel -> mm_one_alpha */
        mm_one_alpha = _mm_set_pi16(1, 0, 0, 0);
    } else if (amask == 0x00FF0000) {
        mm_one_alpha = _mm_set_pi16(0, 1, 0, 0);
    } else if (amask == 0x0000FF00) {
        mm_one_alpha = _mm_set_pi16(0, 0, 1, 0);
    } else {
        mm_one_alpha = _mm_set_pi16(0, 0, 0, 1);
    }

    multmask = 0x00FF;
    multmask <<= (ashift * 2);
    multmask2 = 0x00FF00FF00FF00FFULL;

    while (height--) {
        /* *INDENT-OFF* */ /* clang-format off */
        DUFFS_LOOP4({
        Uint32 alpha = *srcp & amask;
        if (alpha == 0) {
            /* do nothing */
        } else if (alpha == amask) {
            *dstp = *srcp;
        } else {
            src1 = _mm_cvtsi32_si64(*srcp); /* src(ARGB) -> src1 (0000ARGB) */
            src1 = _mm_unpacklo_pi8(src1, mm_zero); /* 0A0R0G0B -> src1 */

            dst1 = _mm_cvtsi32_si64(*dstp); /* dst(ARGB) -> dst1 (0000ARGB) */
            dst1 = _mm_unpacklo_pi8(dst1, mm_zero); /* 0A0R0G0B -> dst1 */

            mm_alpha = _mm_cvtsi32_si64(alpha); /* alpha -> mm_alpha (0000000A) */
            mm_alpha = _mm_srli_si64(mm_alpha, ashift); /* mm_alpha >> ashift -> mm_alpha(0000000A) */
            mm_alpha = _mm_unpacklo_pi16(mm_alpha, mm_alpha); /* 00000A0A -> mm_alpha */
            mm_alpha2 = _mm_unpacklo_pi32(mm_alpha, mm_alpha); /* 0A0A0A0A -> mm_alpha2 */
            mm_alpha = _mm_or_si64(mm_alpha2, *(__m64 *) & multmask);    /* 0F0A0A0A -> mm_alpha */
            mm_alpha2 = _mm_xor_si64(mm_alpha2, *(__m64 *) & multmask2);    /* 255 - mm_alpha -> mm_alpha */

            /*
                Alpha blending is:
                    dstRGB = (srcRGB * srcA) + (dstRGB * (1-srcA))
                    dstA = srcA + (dstA * (1-srcA)) *

                Here, 'src1' is:
                    srcRGB * srcA
                    srcA
                And 'dst1' is:
                    dstRGB * (1-srcA)
                    dstA * (1-srcA)
                so that *dstp is 'src1 + dst1'

                src1 is computed using mullo_pi16: (X * mask) >> 8, but is approximate for srcA ((srcA * 255) >> 8).

                need to a 1 to get an exact result: (srcA * 256) >> 8 == srcA
             */
            mm_alpha = _mm_add_pi16(mm_alpha, mm_one_alpha);

            /* blend */
            src1 = _mm_mullo_pi16(src1, mm_alpha);
            src1 = _mm_srli_pi16(src1, 8);
            dst1 = _mm_mullo_pi16(dst1, mm_alpha2);
            dst1 = _mm_srli_pi16(dst1, 8);
            dst1 = _mm_add_pi16(src1, dst1);
            dst1 = _mm_packs_pu16(dst1, mm_zero);

            *dstp = _mm_cvtsi64_si32(dst1); /* dst1 -> pixel */
        }
        ++srcp;
        ++dstp;
        }, width);
        /* *INDENT-ON* */ /* clang-format on */
        srcp += srcskip;
        dstp += dstskip;
    }
    _mm_empty();
}

#endif /* __MMX__ */

#ifdef SDL_ARM_SIMD_BLITTERS
void BlitARGBto565PixelAlphaARMSIMDAsm(int32_t w, int32_t h, uint16_t *dst, int32_t dst_stride, uint32_t *src, int32_t src_stride);

static void BlitARGBto565PixelAlphaARMSIMD(SDL_BlitInfo *info)
{
    int32_t width = info->dst_w;
    int32_t height = info->dst_h;
    uint16_t *dstp = (uint16_t *)info->dst;
    int32_t dststride = width + (info->dst_skip >> 1);
    uint32_t *srcp = (uint32_t *)info->src;
    int32_t srcstride = width + (info->src_skip >> 2);

    BlitARGBto565PixelAlphaARMSIMDAsm(width, height, dstp, dststride, srcp, srcstride);
}

void BlitRGBtoRGBPixelAlphaARMSIMDAsm(int32_t w, int32_t h, uint32_t *dst, int32_t dst_stride, uint32_t *src, int32_t src_stride);

static void BlitRGBtoRGBPixelAlphaARMSIMD(SDL_BlitInfo *info)
{
    int32_t width = info->dst_w;
    int32_t height = info->dst_h;
    uint32_t *dstp = (uint32_t *)info->dst;
    int32_t dststride = width + (info->dst_skip >> 2);
    uint32_t *srcp = (uint32_t *)info->src;
    int32_t srcstride = width + (info->src_skip >> 2);

    BlitRGBtoRGBPixelAlphaARMSIMDAsm(width, height, dstp, dststride, srcp, srcstride);
}
#endif

#ifdef SDL_ARM_NEON_BLITTERS
void BlitARGBto565PixelAlphaARMNEONAsm(int32_t w, int32_t h, uint16_t *dst, int32_t dst_stride, uint32_t *src, int32_t src_stride);

static void BlitARGBto565PixelAlphaARMNEON(SDL_BlitInfo *info)
{
    int32_t width = info->dst_w;
    int32_t height = info->dst_h;
    uint16_t *dstp = (uint16_t *)info->dst;
    int32_t dststride = width + (info->dst_skip >> 1);
    uint32_t *srcp = (uint32_t *)info->src;
    int32_t srcstride = width + (info->src_skip >> 2);

    BlitARGBto565PixelAlphaARMNEONAsm(width, height, dstp, dststride, srcp, srcstride);
}

void BlitRGBtoRGBPixelAlphaARMNEONAsm(int32_t w, int32_t h, uint32_t *dst, int32_t dst_stride, uint32_t *src, int32_t src_stride);

static void BlitRGBtoRGBPixelAlphaARMNEON(SDL_BlitInfo *info)
{
    int32_t width = info->dst_w;
    int32_t height = info->dst_h;
    uint32_t *dstp = (uint32_t *)info->dst;
    int32_t dststride = width + (info->dst_skip >> 2);
    uint32_t *srcp = (uint32_t *)info->src;
    int32_t srcstride = width + (info->src_skip >> 2);

    BlitRGBtoRGBPixelAlphaARMNEONAsm(width, height, dstp, dststride, srcp, srcstride);
}
#endif

/* fast RGB888->(A)RGB888 blending with surface alpha=128 special case */
static void BlitRGBtoRGBSurfaceAlpha128(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint32 *srcp = (Uint32 *)info->src;
    int srcskip = info->src_skip >> 2;
    Uint32 *dstp = (Uint32 *)info->dst;
    int dstskip = info->dst_skip >> 2;

    while (height--) {
        /* *INDENT-OFF* */ /* clang-format off */
        DUFFS_LOOP4({
            Uint32 s = *srcp++;
            Uint32 d = *dstp;
            *dstp++ = ((((s & 0x00fefefe) + (d & 0x00fefefe)) >> 1)
                   + (s & d & 0x00010101)) | 0xff000000;
        }, width);
        /* *INDENT-ON* */ /* clang-format on */
        srcp += srcskip;
        dstp += dstskip;
    }
}

/* fast RGB888->(A)RGB888 blending with surface alpha */
static void BlitRGBtoRGBSurfaceAlpha(SDL_BlitInfo *info)
{
    unsigned alpha = info->a;
    if (alpha == 128) {
        BlitRGBtoRGBSurfaceAlpha128(info);
    } else {
        int width = info->dst_w;
        int height = info->dst_h;
        Uint32 *srcp = (Uint32 *)info->src;
        int srcskip = info->src_skip >> 2;
        Uint32 *dstp = (Uint32 *)info->dst;
        int dstskip = info->dst_skip >> 2;
        Uint32 s;
        Uint32 d;
        Uint32 s1;
        Uint32 d1;

        while (height--) {
            /* *INDENT-OFF* */ /* clang-format off */
            DUFFS_LOOP4({
                s = *srcp;
                d = *dstp;
                s1 = s & 0xff00ff;
                d1 = d & 0xff00ff;
                d1 = (d1 + ((s1 - d1) * alpha >> 8))
                     & 0xff00ff;
                s &= 0xff00;
                d &= 0xff00;
                d = (d + ((s - d) * alpha >> 8)) & 0xff00;
                *dstp = d1 | d | 0xff000000;
                ++srcp;
                ++dstp;
            }, width);
            /* *INDENT-ON* */ /* clang-format on */
            srcp += srcskip;
            dstp += dstskip;
        }
    }
}

/* fast ARGB888->(A)RGB888 blending with pixel alpha */
static void BlitRGBtoRGBPixelAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint32 *srcp = (Uint32 *)info->src;
    int srcskip = info->src_skip >> 2;
    Uint32 *dstp = (Uint32 *)info->dst;
    int dstskip = info->dst_skip >> 2;

    while (height--) {
        /* *INDENT-OFF* */ /* clang-format off */
        DUFFS_LOOP4({
        Uint32 dalpha;
        Uint32 d;
        Uint32 s1;
        Uint32 d1;
        Uint32 s = *srcp;
        Uint32 alpha = s >> 24;
        /* FIXME: Here we special-case opaque alpha since the
           compositioning used (>>8 instead of /255) doesn't handle
           it correctly. Also special-case alpha=0 for speed?
           Benchmark this! */
        if (alpha) {
          if (alpha == SDL_ALPHA_OPAQUE) {
              *dstp = *srcp;
          } else {
            /*
             * take out the middle component (green), and process
             * the other two in parallel. One multiply less.
             */
            d = *dstp;
            dalpha = d >> 24;
            s1 = s & 0xff00ff;
            d1 = d & 0xff00ff;
            d1 = (d1 + ((s1 - d1) * alpha >> 8)) & 0xff00ff;
            s &= 0xff00;
            d &= 0xff00;
            d = (d + ((s - d) * alpha >> 8)) & 0xff00;
            dalpha = alpha + (dalpha * (alpha ^ 0xFF) >> 8);
            *dstp = d1 | d | (dalpha << 24);
          }
        }
        ++srcp;
        ++dstp;
        }, width);
        /* *INDENT-ON* */ /* clang-format on */
        srcp += srcskip;
        dstp += dstskip;
    }
}

/* fast ARGB888->(A)BGR888 blending with pixel alpha */
static void BlitRGBtoBGRPixelAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint32 *srcp = (Uint32 *)info->src;
    int srcskip = info->src_skip >> 2;
    Uint32 *dstp = (Uint32 *)info->dst;
    int dstskip = info->dst_skip >> 2;

    while (height--) {
        /* *INDENT-OFF* */ /* clang-format off */
        DUFFS_LOOP4({
        Uint32 dalpha;
        Uint32 d;
        Uint32 s1;
        Uint32 d1;
        Uint32 s = *srcp;
        Uint32 alpha = s >> 24;
        /* FIXME: Here we special-case opaque alpha since the
           compositioning used (>>8 instead of /255) doesn't handle
           it correctly. Also special-case alpha=0 for speed?
           Benchmark this! */
        if (alpha) {
          /*
           * take out the middle component (green), and process
           * the other two in parallel. One multiply less.
           */
          s1 = s & 0xff00ff;
          s1 = (s1 >> 16) | (s1 << 16);
          s &= 0xff00;

          if (alpha == SDL_ALPHA_OPAQUE) {
              *dstp = 0xff000000 | s | s1;
          } else {
            d = *dstp;
            dalpha = d >> 24;
            d1 = d & 0xff00ff;
            d1 = (d1 + ((s1 - d1) * alpha >> 8)) & 0xff00ff;
            d &= 0xff00;
            d = (d + ((s - d) * alpha >> 8)) & 0xff00;
            dalpha = alpha + (dalpha * (alpha ^ 0xFF) >> 8);
            *dstp = d1 | d | (dalpha << 24);
          }
        }
        ++srcp;
        ++dstp;
        }, width);
        /* *INDENT-ON* */ /* clang-format on */
        srcp += srcskip;
        dstp += dstskip;
    }
}

#ifdef __3dNOW__
/* fast (as in MMX with prefetch) ARGB888->(A)RGB888 blending with pixel alpha */
static void BlitRGBtoRGBPixelAlphaMMX3DNOW(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint32 *srcp = (Uint32 *)info->src;
    int srcskip = info->src_skip >> 2;
    Uint32 *dstp = (Uint32 *)info->dst;
    int dstskip = info->dst_skip >> 2;
    SDL_PixelFormat *sf = info->src_fmt;
    Uint32 amask = sf->Amask;
    Uint32 ashift = sf->Ashift;
    Uint64 multmask, multmask2;

    __m64 src1, dst1, mm_alpha, mm_zero, mm_alpha2, mm_one_alpha;

    mm_zero = _mm_setzero_si64(); /* 0 -> mm_zero */
    if (amask == 0xFF000000) { /* 1 in the alpha channel -> mm_one_alpha */
        mm_one_alpha = _mm_set_pi16(1, 0, 0, 0);
    } else if (amask == 0x00FF0000) {
        mm_one_alpha = _mm_set_pi16(0, 1, 0, 0);
    } else if (amask == 0x0000FF00) {
        mm_one_alpha = _mm_set_pi16(0, 0, 1, 0);
    } else {
        mm_one_alpha = _mm_set_pi16(0, 0, 0, 1);
    }

    multmask = 0x00FF;
    multmask <<= (ashift * 2);
    multmask2 = 0x00FF00FF00FF00FFULL;

    while (height--) {
        /* *INDENT-OFF* */ /* clang-format off */
        DUFFS_LOOP4({
        Uint32 alpha;

        _m_prefetch(srcp + 16);
        _m_prefetch(dstp + 16);

        alpha = *srcp & amask;
        if (alpha == 0) {
            /* do nothing */
        } else if (alpha == amask) {
            *dstp = *srcp;
        } else {
            src1 = _mm_cvtsi32_si64(*srcp); /* src(ARGB) -> src1 (0000ARGB) */
            src1 = _mm_unpacklo_pi8(src1, mm_zero); /* 0A0R0G0B -> src1 */

            dst1 = _mm_cvtsi32_si64(*dstp); /* dst(ARGB) -> dst1 (0000ARGB) */
            dst1 = _mm_unpacklo_pi8(dst1, mm_zero); /* 0A0R0G0B -> dst1 */

            mm_alpha = _mm_cvtsi32_si64(alpha); /* alpha -> mm_alpha (0000000A) */
            mm_alpha = _mm_srli_si64(mm_alpha, ashift); /* mm_alpha >> ashift -> mm_alpha(0000000A) */
            mm_alpha = _mm_unpacklo_pi16(mm_alpha, mm_alpha); /* 00000A0A -> mm_alpha */
            mm_alpha2 = _mm_unpacklo_pi32(mm_alpha, mm_alpha); /* 0A0A0A0A -> mm_alpha2 */
            mm_alpha = _mm_or_si64(mm_alpha2, *(__m64 *) & multmask);    /* 0F0A0A0A -> mm_alpha */
            mm_alpha2 = _mm_xor_si64(mm_alpha2, *(__m64 *) & multmask2);    /* 255 - mm_alpha -> mm_alpha */

            /*
                Alpha blending is:
                    dstRGB = (srcRGB * srcA) + (dstRGB * (1-srcA))
                    dstA = srcA + (dstA * (1-srcA)) *

                Here, 'src1' is:
                    srcRGB * srcA
                    srcA
                And 'dst1' is:
                    dstRGB * (1-srcA)
                    dstA * (1-srcA)
                so that *dstp is 'src1 + dst1'

                src1 is computed using mullo_pi16: (X * mask) >> 8, but is approximate for srcA ((srcA * 255) >> 8).

                need to a 1 to get an exact result: (srcA * 256) >> 8 == srcA
             */
            mm_alpha = _mm_add_pi16(mm_alpha, mm_one_alpha);

            /* blend */
            src1 = _mm_mullo_pi16(src1, mm_alpha);
            src1 = _mm_srli_pi16(src1, 8);
            dst1 = _mm_mullo_pi16(dst1, mm_alpha2);
            dst1 = _mm_srli_pi16(dst1, 8);
            dst1 = _mm_add_pi16(src1, dst1);
            dst1 = _mm_packs_pu16(dst1, mm_zero);

            *dstp = _mm_cvtsi64_si32(dst1); /* dst1 -> pixel */
        }
        ++srcp;
        ++dstp;
        }, width);
        /* *INDENT-ON* */ /* clang-format on */
        srcp += srcskip;
        dstp += dstskip;
    }
    _mm_empty();
}

#endif /* __3dNOW__ */

/* 16bpp special case for per-surface alpha=50%: blend 2 pixels in parallel */

/* blend a single 16 bit pixel at 50% */
#define BLEND16_50(d, s, mask) \
    ((((s & mask) + (d & mask)) >> 1) + (s & d & (~mask & 0xffff)))

/* blend two 16 bit pixels at 50% */
#define BLEND2x16_50(d, s, mask) \
    (((s & (mask | mask << 16)) >> 1) + ((d & (mask | mask << 16)) >> 1) + (s & d & (~(mask | mask << 16))))

static void Blit16to16SurfaceAlpha128(SDL_BlitInfo *info, Uint16 mask)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint16 *srcp = (Uint16 *)info->src;
    int srcskip = info->src_skip >> 1;
    Uint16 *dstp = (Uint16 *)info->dst;
    int dstskip = info->dst_skip >> 1;

    while (height--) {
        if (((uintptr_t)srcp ^ (uintptr_t)dstp) & 2) {
            /*
             * Source and destination not aligned, pipeline it.
             * This is mostly a win for big blits but no loss for
             * small ones
             */
            Uint32 prev_sw;
            int w = width;

            /* handle odd destination */
            if ((uintptr_t)dstp & 2) {
                Uint16 d = *dstp, s = *srcp;
                *dstp = BLEND16_50(d, s, mask);
                dstp++;
                srcp++;
                w--;
            }
            srcp++; /* srcp is now 32-bit aligned */

            /* bootstrap pipeline with first halfword */
            prev_sw = ((Uint32 *)srcp)[-1];

            while (w > 1) {
                Uint32 sw, dw, s;
                sw = *(Uint32 *)srcp;
                dw = *(Uint32 *)dstp;
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
                s = (prev_sw << 16) + (sw >> 16);
#else
                s = (prev_sw >> 16) + (sw << 16);
#endif
                prev_sw = sw;
                *(Uint32 *)dstp = BLEND2x16_50(dw, s, mask);
                dstp += 2;
                srcp += 2;
                w -= 2;
            }

            /* final pixel if any */
            if (w) {
                Uint16 d = *dstp, s;
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
                s = (Uint16)prev_sw;
#else
                s = (Uint16)(prev_sw >> 16);
#endif
                *dstp = BLEND16_50(d, s, mask);
                srcp++;
                dstp++;
            }
            srcp += srcskip - 1;
            dstp += dstskip;
        } else {
            /* source and destination are aligned */
            int w = width;

            /* first odd pixel? */
            if ((uintptr_t)srcp & 2) {
                Uint16 d = *dstp, s = *srcp;
                *dstp = BLEND16_50(d, s, mask);
                srcp++;
                dstp++;
                w--;
            }
            /* srcp and dstp are now 32-bit aligned */

            while (w > 1) {
                Uint32 sw = *(Uint32 *)srcp;
                Uint32 dw = *(Uint32 *)dstp;
                *(Uint32 *)dstp = BLEND2x16_50(dw, sw, mask);
                srcp += 2;
                dstp += 2;
                w -= 2;
            }

            /* last odd pixel? */
            if (w) {
                Uint16 d = *dstp, s = *srcp;
                *dstp = BLEND16_50(d, s, mask);
                srcp++;
                dstp++;
            }
            srcp += srcskip;
            dstp += dstskip;
        }
    }
}

#ifdef __MMX__

/* fast RGB565->RGB565 blending with surface alpha */
static void Blit565to565SurfaceAlphaMMX(SDL_BlitInfo *info)
{
    unsigned alpha = info->a;
    if (alpha == 128) {
        Blit16to16SurfaceAlpha128(info, 0xf7de);
    } else {
        int width = info->dst_w;
        int height = info->dst_h;
        Uint16 *srcp = (Uint16 *)info->src;
        int srcskip = info->src_skip >> 1;
        Uint16 *dstp = (Uint16 *)info->dst;
        int dstskip = info->dst_skip >> 1;
        Uint32 s, d;

#ifdef USE_DUFFS_LOOP
        __m64 src1, dst1, src2, dst2, gmask, bmask, mm_res, mm_alpha;

        alpha &= ~(1 + 2 + 4);             /* cut alpha to get the exact same behaviour */
        mm_alpha = _mm_set_pi32(0, alpha); /* 0000000A -> mm_alpha */
        alpha >>= 3;                       /* downscale alpha to 5 bits */

        mm_alpha = _mm_unpacklo_pi16(mm_alpha, mm_alpha); /* 00000A0A -> mm_alpha */
        mm_alpha = _mm_unpacklo_pi32(mm_alpha, mm_alpha); /* 0A0A0A0A -> mm_alpha */
        /* position alpha to allow for mullo and mulhi on diff channels
           to reduce the number of operations */
        mm_alpha = _mm_slli_si64(mm_alpha, 3);

        /* Setup the 565 color channel masks */
        gmask = _mm_set_pi32(0x07E007E0, 0x07E007E0); /* MASKGREEN -> gmask */
        bmask = _mm_set_pi32(0x001F001F, 0x001F001F); /* MASKBLUE -> bmask */
#endif

        while (height--) {
            /* *INDENT-OFF* */ /* clang-format off */
            DUFFS_LOOP_124(
            {
                s = *srcp++;
                d = *dstp;
                /*
                 * shift out the middle component (green) to
                 * the high 16 bits, and process all three RGB
                 * components at the same time.
                 */
                s = (s | s << 16) & 0x07e0f81f;
                d = (d | d << 16) & 0x07e0f81f;
                d += (s - d) * alpha >> 5;
                d &= 0x07e0f81f;
                *dstp++ = (Uint16)(d | d >> 16);
            },{
                s = *srcp++;
                d = *dstp;
                /*
                 * shift out the middle component (green) to
                 * the high 16 bits, and process all three RGB
                 * components at the same time.
                 */
                s = (s | s << 16) & 0x07e0f81f;
                d = (d | d << 16) & 0x07e0f81f;
                d += (s - d) * alpha >> 5;
                d &= 0x07e0f81f;
                *dstp++ = (Uint16)(d | d >> 16);
                s = *srcp++;
                d = *dstp;
                /*
                 * shift out the middle component (green) to
                 * the high 16 bits, and process all three RGB
                 * components at the same time.
                 */
                s = (s | s << 16) & 0x07e0f81f;
                d = (d | d << 16) & 0x07e0f81f;
                d += (s - d) * alpha >> 5;
                d &= 0x07e0f81f;
                *dstp++ = (Uint16)(d | d >> 16);
            },{
                src1 = *(__m64*)srcp; /* 4 src pixels -> src1 */
                dst1 = *(__m64*)dstp; /* 4 dst pixels -> dst1 */

                /* red */
                src2 = src1;
                src2 = _mm_srli_pi16(src2, 11); /* src2 >> 11 -> src2 [000r 000r 000r 000r] */

                dst2 = dst1;
                dst2 = _mm_srli_pi16(dst2, 11); /* dst2 >> 11 -> dst2 [000r 000r 000r 000r] */

                /* blend */
                src2 = _mm_sub_pi16(src2, dst2);/* src - dst -> src2 */
                src2 = _mm_mullo_pi16(src2, mm_alpha); /* src2 * alpha -> src2 */
                src2 = _mm_srli_pi16(src2, 11); /* src2 >> 11 -> src2 */
                dst2 = _mm_add_pi16(src2, dst2); /* src2 + dst2 -> dst2 */
                dst2 = _mm_slli_pi16(dst2, 11); /* dst2 << 11 -> dst2 */

                mm_res = dst2; /* RED -> mm_res */

                /* green -- process the bits in place */
                src2 = src1;
                src2 = _mm_and_si64(src2, gmask); /* src & MASKGREEN -> src2 */

                dst2 = dst1;
                dst2 = _mm_and_si64(dst2, gmask); /* dst & MASKGREEN -> dst2 */

                /* blend */
                src2 = _mm_sub_pi16(src2, dst2);/* src - dst -> src2 */
                src2 = _mm_mulhi_pi16(src2, mm_alpha); /* src2 * alpha -> src2 */
                src2 = _mm_slli_pi16(src2, 5); /* src2 << 5 -> src2 */
                dst2 = _mm_add_pi16(src2, dst2); /* src2 + dst2 -> dst2 */

                mm_res = _mm_or_si64(mm_res, dst2); /* RED | GREEN -> mm_res */

                /* blue */
                src2 = src1;
                src2 = _mm_and_si64(src2, bmask); /* src & MASKBLUE -> src2[000b 000b 000b 000b] */

                dst2 = dst1;
                dst2 = _mm_and_si64(dst2, bmask); /* dst & MASKBLUE -> dst2[000b 000b 000b 000b] */

                /* blend */
                src2 = _mm_sub_pi16(src2, dst2);/* src - dst -> src2 */
                src2 = _mm_mullo_pi16(src2, mm_alpha); /* src2 * alpha -> src2 */
                src2 = _mm_srli_pi16(src2, 11); /* src2 >> 11 -> src2 */
                dst2 = _mm_add_pi16(src2, dst2); /* src2 + dst2 -> dst2 */
                dst2 = _mm_and_si64(dst2, bmask); /* dst2 & MASKBLUE -> dst2 */

                mm_res = _mm_or_si64(mm_res, dst2); /* RED | GREEN | BLUE -> mm_res */

                *(__m64*)dstp = mm_res; /* mm_res -> 4 dst pixels */

                srcp += 4;
                dstp += 4;
            }, width);
            /* *INDENT-ON* */ /* clang-format on */
            srcp += srcskip;
            dstp += dstskip;
        }
        _mm_empty();
    }
}

/* fast RGB555->RGB555 blending with surface alpha */
static void Blit555to555SurfaceAlphaMMX(SDL_BlitInfo *info)
{
    unsigned alpha = info->a;
    if (alpha == 128) {
        Blit16to16SurfaceAlpha128(info, 0xfbde);
    } else {
        int width = info->dst_w;
        int height = info->dst_h;
        Uint16 *srcp = (Uint16 *)info->src;
        int srcskip = info->src_skip >> 1;
        Uint16 *dstp = (Uint16 *)info->dst;
        int dstskip = info->dst_skip >> 1;
        Uint32 s, d;

#ifdef USE_DUFFS_LOOP
        __m64 src1, dst1, src2, dst2, rmask, gmask, bmask, mm_res, mm_alpha;

        alpha &= ~(1 + 2 + 4);             /* cut alpha to get the exact same behaviour */
        mm_alpha = _mm_set_pi32(0, alpha); /* 0000000A -> mm_alpha */
        alpha >>= 3;                       /* downscale alpha to 5 bits */

        mm_alpha = _mm_unpacklo_pi16(mm_alpha, mm_alpha); /* 00000A0A -> mm_alpha */
        mm_alpha = _mm_unpacklo_pi32(mm_alpha, mm_alpha); /* 0A0A0A0A -> mm_alpha */
        /* position alpha to allow for mullo and mulhi on diff channels
           to reduce the number of operations */
        mm_alpha = _mm_slli_si64(mm_alpha, 3);

        /* Setup the 555 color channel masks */
        rmask = _mm_set_pi32(0x7C007C00, 0x7C007C00); /* MASKRED -> rmask */
        gmask = _mm_set_pi32(0x03E003E0, 0x03E003E0); /* MASKGREEN -> gmask */
        bmask = _mm_set_pi32(0x001F001F, 0x001F001F); /* MASKBLUE -> bmask */
#endif
        while (height--) {
            /* *INDENT-OFF* */ /* clang-format off */
            DUFFS_LOOP_124(
            {
                s = *srcp++;
                d = *dstp;
                /*
                 * shift out the middle component (green) to
                 * the high 16 bits, and process all three RGB
                 * components at the same time.
                 */
                s = (s | s << 16) & 0x03e07c1f;
                d = (d | d << 16) & 0x03e07c1f;
                d += (s - d) * alpha >> 5;
                d &= 0x03e07c1f;
                *dstp++ = (Uint16)(d | d >> 16);
            },{
                s = *srcp++;
                d = *dstp;
                /*
                 * shift out the middle component (green) to
                 * the high 16 bits, and process all three RGB
                 * components at the same time.
                 */
                s = (s | s << 16) & 0x03e07c1f;
                d = (d | d << 16) & 0x03e07c1f;
                d += (s - d) * alpha >> 5;
                d &= 0x03e07c1f;
                *dstp++ = (Uint16)(d | d >> 16);
                    s = *srcp++;
                d = *dstp;
                /*
                 * shift out the middle component (green) to
                 * the high 16 bits, and process all three RGB
                 * components at the same time.
                 */
                s = (s | s << 16) & 0x03e07c1f;
                d = (d | d << 16) & 0x03e07c1f;
                d += (s - d) * alpha >> 5;
                d &= 0x03e07c1f;
                *dstp++ = (Uint16)(d | d >> 16);
            },{
                src1 = *(__m64*)srcp; /* 4 src pixels -> src1 */
                dst1 = *(__m64*)dstp; /* 4 dst pixels -> dst1 */

                /* red -- process the bits in place */
                src2 = src1;
                src2 = _mm_and_si64(src2, rmask); /* src & MASKRED -> src2 */

                dst2 = dst1;
                dst2 = _mm_and_si64(dst2, rmask); /* dst & MASKRED -> dst2 */

                /* blend */
                src2 = _mm_sub_pi16(src2, dst2);/* src - dst -> src2 */
                src2 = _mm_mulhi_pi16(src2, mm_alpha); /* src2 * alpha -> src2 */
                src2 = _mm_slli_pi16(src2, 5); /* src2 << 5 -> src2 */
                dst2 = _mm_add_pi16(src2, dst2); /* src2 + dst2 -> dst2 */
                dst2 = _mm_and_si64(dst2, rmask); /* dst2 & MASKRED -> dst2 */

                mm_res = dst2; /* RED -> mm_res */

                /* green -- process the bits in place */
                src2 = src1;
                src2 = _mm_and_si64(src2, gmask); /* src & MASKGREEN -> src2 */

                dst2 = dst1;
                dst2 = _mm_and_si64(dst2, gmask); /* dst & MASKGREEN -> dst2 */

                /* blend */
                src2 = _mm_sub_pi16(src2, dst2);/* src - dst -> src2 */
                src2 = _mm_mulhi_pi16(src2, mm_alpha); /* src2 * alpha -> src2 */
                src2 = _mm_slli_pi16(src2, 5); /* src2 << 5 -> src2 */
                dst2 = _mm_add_pi16(src2, dst2); /* src2 + dst2 -> dst2 */

                mm_res = _mm_or_si64(mm_res, dst2); /* RED | GREEN -> mm_res */

                /* blue */
                src2 = src1; /* src -> src2 */
                src2 = _mm_and_si64(src2, bmask); /* src & MASKBLUE -> src2[000b 000b 000b 000b] */

                dst2 = dst1; /* dst -> dst2 */
                dst2 = _mm_and_si64(dst2, bmask); /* dst & MASKBLUE -> dst2[000b 000b 000b 000b] */

                /* blend */
                src2 = _mm_sub_pi16(src2, dst2);/* src - dst -> src2 */
                src2 = _mm_mullo_pi16(src2, mm_alpha); /* src2 * alpha -> src2 */
                src2 = _mm_srli_pi16(src2, 11); /* src2 >> 11 -> src2 */
                dst2 = _mm_add_pi16(src2, dst2); /* src2 + dst2 -> dst2 */
                dst2 = _mm_and_si64(dst2, bmask); /* dst2 & MASKBLUE -> dst2 */

                mm_res = _mm_or_si64(mm_res, dst2); /* RED | GREEN | BLUE -> mm_res */

                *(__m64*)dstp = mm_res; /* mm_res -> 4 dst pixels */

                srcp += 4;
                dstp += 4;
            }, width);
            /* *INDENT-ON* */ /* clang-format on */
            srcp += srcskip;
            dstp += dstskip;
        }
        _mm_empty();
    }
}

#endif /* __MMX__ */

/* fast RGB565->RGB565 blending with surface alpha */
static void Blit565to565SurfaceAlpha(SDL_BlitInfo *info)
{
    unsigned alpha = info->a;
    if (alpha == 128) {
        Blit16to16SurfaceAlpha128(info, 0xf7de);
    } else {
        int width = info->dst_w;
        int height = info->dst_h;
        Uint16 *srcp = (Uint16 *)info->src;
        int srcskip = info->src_skip >> 1;
        Uint16 *dstp = (Uint16 *)info->dst;
        int dstskip = info->dst_skip >> 1;
        alpha >>= 3; /* downscale alpha to 5 bits */

        while (height--) {
            /* *INDENT-OFF* */ /* clang-format off */
            DUFFS_LOOP4({
                Uint32 s = *srcp++;
                Uint32 d = *dstp;
                /*
                 * shift out the middle component (green) to
                 * the high 16 bits, and process all three RGB
                 * components at the same time.
                 */
                s = (s | s << 16) & 0x07e0f81f;
                d = (d | d << 16) & 0x07e0f81f;
                d += (s - d) * alpha >> 5;
                d &= 0x07e0f81f;
                *dstp++ = (Uint16)(d | d >> 16);
            }, width);
            /* *INDENT-ON* */ /* clang-format on */
            srcp += srcskip;
            dstp += dstskip;
        }
    }
}

/* fast RGB555->RGB555 blending with surface alpha */
static void Blit555to555SurfaceAlpha(SDL_BlitInfo *info)
{
    unsigned alpha = info->a; /* downscale alpha to 5 bits */
    if (alpha == 128) {
        Blit16to16SurfaceAlpha128(info, 0xfbde);
    } else {
        int width = info->dst_w;
        int height = info->dst_h;
        Uint16 *srcp = (Uint16 *)info->src;
        int srcskip = info->src_skip >> 1;
        Uint16 *dstp = (Uint16 *)info->dst;
        int dstskip = info->dst_skip >> 1;
        alpha >>= 3; /* downscale alpha to 5 bits */

        while (height--) {
            /* *INDENT-OFF* */ /* clang-format off */
            DUFFS_LOOP4({
                Uint32 s = *srcp++;
                Uint32 d = *dstp;
                /*
                 * shift out the middle component (green) to
                 * the high 16 bits, and process all three RGB
                 * components at the same time.
                 */
                s = (s | s << 16) & 0x03e07c1f;
                d = (d | d << 16) & 0x03e07c1f;
                d += (s - d) * alpha >> 5;
                d &= 0x03e07c1f;
                *dstp++ = (Uint16)(d | d >> 16);
            }, width);
            /* *INDENT-ON* */ /* clang-format on */
            srcp += srcskip;
            dstp += dstskip;
        }
    }
}

/* fast ARGB8888->RGB565 blending with pixel alpha */
static void BlitARGBto565PixelAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint32 *srcp = (Uint32 *)info->src;
    int srcskip = info->src_skip >> 2;
    Uint16 *dstp = (Uint16 *)info->dst;
    int dstskip = info->dst_skip >> 1;

    while (height--) {
        /* *INDENT-OFF* */ /* clang-format off */
        DUFFS_LOOP4({
        Uint32 s = *srcp;
        unsigned alpha = s >> 27; /* downscale alpha to 5 bits */
        /* Here we special-case opaque alpha since the
           compositioning used (>>8 instead of /255) doesn't handle
           it correctly. */
        if (alpha) {
          if (alpha == (SDL_ALPHA_OPAQUE >> 3)) {
            *dstp = (Uint16)((s >> 8 & 0xf800) + (s >> 5 & 0x7e0) + (s >> 3  & 0x1f));
          } else {
            Uint32 d = *dstp;
            /*
             * convert source and destination to G0RAB65565
             * and blend all components at the same time
             */
            s = ((s & 0xfc00) << 11) + (s >> 8 & 0xf800) + (s >> 3 & 0x1f);
            d = (d | d << 16) & 0x07e0f81f;
            d += (s - d) * alpha >> 5;
            d &= 0x07e0f81f;
            *dstp = (Uint16)(d | d >> 16);
          }
        }
        srcp++;
        dstp++;
        }, width);
        /* *INDENT-ON* */ /* clang-format on */
        srcp += srcskip;
        dstp += dstskip;
    }
}

/* fast ARGB8888->RGB555 blending with pixel alpha */
static void BlitARGBto555PixelAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint32 *srcp = (Uint32 *)info->src;
    int srcskip = info->src_skip >> 2;
    Uint16 *dstp = (Uint16 *)info->dst;
    int dstskip = info->dst_skip >> 1;

    while (height--) {
        /* *INDENT-OFF* */ /* clang-format off */
        DUFFS_LOOP4({
        unsigned alpha;
        Uint32 s = *srcp;
        alpha = s >> 27; /* downscale alpha to 5 bits */
        /* Here we special-case opaque alpha since the
           compositioning used (>>8 instead of /255) doesn't handle
           it correctly. */
        if (alpha) {
          if (alpha == (SDL_ALPHA_OPAQUE >> 3)) {
            *dstp = (Uint16)((s >> 9 & 0x7c00) + (s >> 6 & 0x3e0) + (s >> 3  & 0x1f));
          } else {
            Uint32 d = *dstp;
            /*
             * convert source and destination to G0RAB55555
             * and blend all components at the same time
             */
            s = ((s & 0xf800) << 10) + (s >> 9 & 0x7c00) + (s >> 3 & 0x1f);
            d = (d | d << 16) & 0x03e07c1f;
            d += (s - d) * alpha >> 5;
            d &= 0x03e07c1f;
            *dstp = (Uint16)(d | d >> 16);
          }
        }
        srcp++;
        dstp++;
        }, width);
        /* *INDENT-ON* */ /* clang-format on */
        srcp += srcskip;
        dstp += dstskip;
    }
}

/* General (slow) N->N blending with per-surface alpha */
static void BlitNtoNSurfaceAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    SDL_PixelFormat *srcfmt = info->src_fmt;
    SDL_PixelFormat *dstfmt = info->dst_fmt;
    int srcbpp = srcfmt->BytesPerPixel;
    int dstbpp = dstfmt->BytesPerPixel;
    Uint32 Pixel;
    unsigned sR, sG, sB;
    unsigned dR, dG, dB, dA;
    const unsigned sA = info->a;

    if (sA) {
        while (height--) {
            /* *INDENT-OFF* */ /* clang-format off */
        DUFFS_LOOP4(
        {
        DISEMBLE_RGB(src, srcbpp, srcfmt, Pixel, sR, sG, sB);
        DISEMBLE_RGBA(dst, dstbpp, dstfmt, Pixel, dR, dG, dB, dA);
        ALPHA_BLEND_RGBA(sR, sG, sB, sA, dR, dG, dB, dA);
        ASSEMBLE_RGBA(dst, dstbpp, dstfmt, dR, dG, dB, dA);
        src += srcbpp;
        dst += dstbpp;
        },
        width);
        /* *INDENT-ON* */ /* clang-format on */
            src += srcskip;
            dst += dstskip;
        }
    }
}

/* General (slow) colorkeyed N->N blending with per-surface alpha */
static void BlitNtoNSurfaceAlphaKey(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    SDL_PixelFormat *srcfmt = info->src_fmt;
    SDL_PixelFormat *dstfmt = info->dst_fmt;
    Uint32 ckey = info->colorkey;
    int srcbpp = srcfmt->BytesPerPixel;
    int dstbpp = dstfmt->BytesPerPixel;
    Uint32 Pixel;
    unsigned sR, sG, sB;
    unsigned dR, dG, dB, dA;
    const unsigned sA = info->a;

    while (height--) {
        /* *INDENT-OFF* */ /* clang-format off */
        DUFFS_LOOP4(
        {
        RETRIEVE_RGB_PIXEL(src, srcbpp, Pixel);
        if (sA && Pixel != ckey) {
            RGB_FROM_PIXEL(Pixel, srcfmt, sR, sG, sB);
            DISEMBLE_RGBA(dst, dstbpp, dstfmt, Pixel, dR, dG, dB, dA);
            ALPHA_BLEND_RGBA(sR, sG, sB, sA, dR, dG, dB, dA);
            ASSEMBLE_RGBA(dst, dstbpp, dstfmt, dR, dG, dB, dA);
        }
        src += srcbpp;
        dst += dstbpp;
        },
        width);
        /* *INDENT-ON* */ /* clang-format on */
        src += srcskip;
        dst += dstskip;
    }
}

/* General (slow) N->N blending with pixel alpha */
static void BlitNtoNPixelAlpha(SDL_BlitInfo *info)
{
    int width = info->dst_w;
    int height = info->dst_h;
    Uint8 *src = info->src;
    int srcskip = info->src_skip;
    Uint8 *dst = info->dst;
    int dstskip = info->dst_skip;
    SDL_PixelFormat *srcfmt = info->src_fmt;
    SDL_PixelFormat *dstfmt = info->dst_fmt;
    int srcbpp;
    int dstbpp;
    Uint32 Pixel;
    unsigned sR, sG, sB, sA;
    unsigned dR, dG, dB, dA;

    /* Set up some basic variables */
    srcbpp = srcfmt->BytesPerPixel;
    dstbpp = dstfmt->BytesPerPixel;

    while (height--) {
        /* *INDENT-OFF* */ /* clang-format off */
        DUFFS_LOOP4(
        {
        DISEMBLE_RGBA(src, srcbpp, srcfmt, Pixel, sR, sG, sB, sA);
        if (sA) {
            DISEMBLE_RGBA(dst, dstbpp, dstfmt, Pixel, dR, dG, dB, dA);
            ALPHA_BLEND_RGBA(sR, sG, sB, sA, dR, dG, dB, dA);
            ASSEMBLE_RGBA(dst, dstbpp, dstfmt, dR, dG, dB, dA);
        }
        src += srcbpp;
        dst += dstbpp;
        },
        width);
        /* *INDENT-ON* */ /* clang-format on */
        src += srcskip;
        dst += dstskip;
    }
}

SDL_BlitFunc SDL_CalculateBlitA(SDL_Surface *surface)
{
    SDL_PixelFormat *sf = surface->format;
    SDL_PixelFormat *df = surface->map->dst->format;

    switch (surface->map->info.flags & ~SDL_COPY_RLE_MASK) {
    case SDL_COPY_BLEND:
        /* Per-pixel alpha blits */
        switch (df->BytesPerPixel) {
        case 1:
            if (df->palette) {
                return BlitNto1PixelAlpha;
            } else {
                /* RGB332 has no palette ! */
                return BlitNtoNPixelAlpha;
            }

        case 2:
#if defined(SDL_ARM_NEON_BLITTERS) || defined(SDL_ARM_SIMD_BLITTERS)
            if (sf->BytesPerPixel == 4 && sf->Amask == 0xff000000 && sf->Gmask == 0xff00 && df->Gmask == 0x7e0 && ((sf->Rmask == 0xff && df->Rmask == 0x1f) || (sf->Bmask == 0xff && df->Bmask == 0x1f))) {
#ifdef SDL_ARM_NEON_BLITTERS
                if (SDL_HasNEON()) {
                    return BlitARGBto565PixelAlphaARMNEON;
                }
#endif
#ifdef SDL_ARM_SIMD_BLITTERS
                if (SDL_HasARMSIMD()) {
                    return BlitARGBto565PixelAlphaARMSIMD;
                }
#endif
            }
#endif
            if (sf->BytesPerPixel == 4 && sf->Amask == 0xff000000 && sf->Gmask == 0xff00 && ((sf->Rmask == 0xff && df->Rmask == 0x1f) || (sf->Bmask == 0xff && df->Bmask == 0x1f))) {
                if (df->Gmask == 0x7e0) {
                    return BlitARGBto565PixelAlpha;
                } else if (df->Gmask == 0x3e0 && !df->Amask) {
                    return BlitARGBto555PixelAlpha;
                }
            }
            return BlitNtoNPixelAlpha;

        case 4:
            if (sf->Rmask == df->Rmask && sf->Gmask == df->Gmask && sf->Bmask == df->Bmask && sf->BytesPerPixel == 4) {
#if defined(__MMX__) || defined(__3dNOW__)
                if (sf->Rshift % 8 == 0 && sf->Gshift % 8 == 0 && sf->Bshift % 8 == 0 && sf->Ashift % 8 == 0 && sf->Aloss == 0) {
#ifdef __3dNOW__
                    if (SDL_Has3DNow()) {
                        return BlitRGBtoRGBPixelAlphaMMX3DNOW;
                    }
#endif
#ifdef __MMX__
                    if (SDL_HasMMX()) {
                        return BlitRGBtoRGBPixelAlphaMMX;
                    }
#endif
                }
#endif /* __MMX__ || __3dNOW__ */
                if (sf->Amask == 0xff000000) {
#ifdef SDL_ARM_NEON_BLITTERS
                    if (SDL_HasNEON()) {
                        return BlitRGBtoRGBPixelAlphaARMNEON;
                    }
#endif
#ifdef SDL_ARM_SIMD_BLITTERS
                    if (SDL_HasARMSIMD()) {
                        return BlitRGBtoRGBPixelAlphaARMSIMD;
                    }
#endif
                    return BlitRGBtoRGBPixelAlpha;
                }
            } else if (sf->Rmask == df->Bmask && sf->Gmask == df->Gmask && sf->Bmask == df->Rmask && sf->BytesPerPixel == 4) {
                if (sf->Amask == 0xff000000) {
                    return BlitRGBtoBGRPixelAlpha;
                }
            }
            return BlitNtoNPixelAlpha;

        case 3:
        default:
            break;
        }
        return BlitNtoNPixelAlpha;

    case SDL_COPY_MODULATE_ALPHA | SDL_COPY_BLEND:
        if (sf->Amask == 0) {
            /* Per-surface alpha blits */
            switch (df->BytesPerPixel) {
            case 1:
                if (df->palette) {
                    return BlitNto1SurfaceAlpha;
                } else {
                    /* RGB332 has no palette ! */
                    return BlitNtoNSurfaceAlpha;
                }

            case 2:
                if (surface->map->identity) {
                    if (df->Gmask == 0x7e0) {
#ifdef __MMX__
                        if (SDL_HasMMX()) {
                            return Blit565to565SurfaceAlphaMMX;
                        } else
#endif
                        {
                            return Blit565to565SurfaceAlpha;
                        }
                    } else if (df->Gmask == 0x3e0) {
#ifdef __MMX__
                        if (SDL_HasMMX()) {
                            return Blit555to555SurfaceAlphaMMX;
                        } else
#endif
                        {
                            return Blit555to555SurfaceAlpha;
                        }
                    }
                }
                return BlitNtoNSurfaceAlpha;

            case 4:
                if (sf->Rmask == df->Rmask && sf->Gmask == df->Gmask && sf->Bmask == df->Bmask && sf->BytesPerPixel == 4) {
#ifdef __MMX__
                    if (sf->Rshift % 8 == 0 && sf->Gshift % 8 == 0 && sf->Bshift % 8 == 0 && SDL_HasMMX()) {
                        return BlitRGBtoRGBSurfaceAlphaMMX;
                    }
#endif
                    if ((sf->Rmask | sf->Gmask | sf->Bmask) == 0xffffff) {
                        return BlitRGBtoRGBSurfaceAlpha;
                    }
                }
                return BlitNtoNSurfaceAlpha;

            case 3:
            default:
                return BlitNtoNSurfaceAlpha;
            }
        }
        break;

    case SDL_COPY_COLORKEY | SDL_COPY_MODULATE_ALPHA | SDL_COPY_BLEND:
        if (sf->Amask == 0) {
            if (df->BytesPerPixel == 1) {

                if (df->palette) {
                    return BlitNto1SurfaceAlphaKey;
                } else {
                    /* RGB332 has no palette ! */
                    return BlitNtoNSurfaceAlphaKey;
                }
            } else {
                return BlitNtoNSurfaceAlphaKey;
            }
        }
        break;
    }

    return NULL;
}

#endif /* SDL_HAVE_BLIT_A */

/* vi: set ts=4 sw=4 expandtab: */
