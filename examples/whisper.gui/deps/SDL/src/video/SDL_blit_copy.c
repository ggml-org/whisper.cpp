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

#include "SDL_video.h"
#include "SDL_blit.h"
#include "SDL_blit_copy.h"

#ifdef __SSE__
/* This assumes 16-byte aligned src and dst */
static SDL_INLINE void SDL_memcpySSE(Uint8 *dst, const Uint8 *src, int len)
{
    int i;

    __m128 values[4];
    for (i = len / 64; i--;) {
        _mm_prefetch((const char *)src, _MM_HINT_NTA);
        values[0] = *(__m128 *)(src + 0);
        values[1] = *(__m128 *)(src + 16);
        values[2] = *(__m128 *)(src + 32);
        values[3] = *(__m128 *)(src + 48);
        _mm_stream_ps((float *)(dst + 0), values[0]);
        _mm_stream_ps((float *)(dst + 16), values[1]);
        _mm_stream_ps((float *)(dst + 32), values[2]);
        _mm_stream_ps((float *)(dst + 48), values[3]);
        src += 64;
        dst += 64;
    }

    if (len & 63) {
        SDL_memcpy(dst, src, len & 63);
    }
}
#endif /* __SSE__ */

#ifdef __MMX__
#ifdef _MSC_VER
#pragma warning(disable : 4799)
#endif
static SDL_INLINE void SDL_memcpyMMX(Uint8 *dst, const Uint8 *src, int len)
{
    int remain = len & 63;
    int i;

    __m64 *d64 = (__m64 *)dst;
    __m64 *s64 = (__m64 *)src;

    for (i = len / 64; i--;) {
        d64[0] = s64[0];
        d64[1] = s64[1];
        d64[2] = s64[2];
        d64[3] = s64[3];
        d64[4] = s64[4];
        d64[5] = s64[5];
        d64[6] = s64[6];
        d64[7] = s64[7];

        d64 += 8;
        s64 += 8;
    }

    if (remain) {
        const int skip = len - remain;
        dst += skip;
        src += skip;
        while (remain--) {
            *dst++ = *src++;
        }
    }
}
#endif /* __MMX__ */

void SDL_BlitCopy(SDL_BlitInfo *info)
{
    SDL_bool overlap;
    Uint8 *src, *dst;
    int w, h;
    int srcskip, dstskip;

    w = info->dst_w * info->dst_fmt->BytesPerPixel;
    h = info->dst_h;
    src = info->src;
    dst = info->dst;
    srcskip = info->src_pitch;
    dstskip = info->dst_pitch;

    /* Properly handle overlapping blits */
    if (src < dst) {
        overlap = (dst < (src + h * srcskip));
    } else {
        overlap = (src < (dst + h * dstskip));
    }
    if (overlap) {
        if (dst < src) {
            while (h--) {
                SDL_memmove(dst, src, w);
                src += srcskip;
                dst += dstskip;
            }
        } else {
            src += ((h - 1) * srcskip);
            dst += ((h - 1) * dstskip);
            while (h--) {
                SDL_memmove(dst, src, w);
                src -= srcskip;
                dst -= dstskip;
            }
        }
        return;
    }

#ifdef __SSE__
    if (SDL_HasSSE() &&
        !((uintptr_t)src & 15) && !(srcskip & 15) &&
        !((uintptr_t)dst & 15) && !(dstskip & 15)) {
        while (h--) {
            SDL_memcpySSE(dst, src, w);
            src += srcskip;
            dst += dstskip;
        }
        return;
    }
#endif

#ifdef __MMX__
    if (SDL_HasMMX() && !(srcskip & 7) && !(dstskip & 7)) {
        while (h--) {
            SDL_memcpyMMX(dst, src, w);
            src += srcskip;
            dst += dstskip;
        }
        _mm_empty();
        return;
    }
#endif

    while (h--) {
        SDL_memcpy(dst, src, w);
        src += srcskip;
        dst += dstskip;
    }
}

/* vi: set ts=4 sw=4 expandtab: */
