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
#include "SDL_blit_slow.h"

#define FORMAT_ALPHA                0
#define FORMAT_NO_ALPHA             -1
#define FORMAT_2101010              1
#define FORMAT_HAS_ALPHA(format)    format == 0
#define FORMAT_HAS_NO_ALPHA(format) format < 0
static int SDL_INLINE detect_format(SDL_PixelFormat *pf)
{
    if (pf->format == SDL_PIXELFORMAT_ARGB2101010) {
        return FORMAT_2101010;
    } else if (pf->Amask) {
        return FORMAT_ALPHA;
    } else {
        return FORMAT_NO_ALPHA;
    }
}

/* The ONE TRUE BLITTER
 * This puppy has to handle all the unoptimized cases - yes, it's slow.
 */
void SDL_Blit_Slow(SDL_BlitInfo *info)
{
    const int flags = info->flags;
    const Uint32 modulateR = info->r;
    const Uint32 modulateG = info->g;
    const Uint32 modulateB = info->b;
    const Uint32 modulateA = info->a;
    Uint32 srcpixel;
    Uint32 srcR, srcG, srcB, srcA;
    Uint32 dstpixel;
    Uint32 dstR, dstG, dstB, dstA;
    Uint64 srcy, srcx;
    Uint64 posy, posx;
    Uint64 incy, incx;
    SDL_PixelFormat *src_fmt = info->src_fmt;
    SDL_PixelFormat *dst_fmt = info->dst_fmt;
    int srcbpp = src_fmt->BytesPerPixel;
    int dstbpp = dst_fmt->BytesPerPixel;
    int srcfmt_val;
    int dstfmt_val;
    Uint32 rgbmask = ~src_fmt->Amask;
    Uint32 ckey = info->colorkey & rgbmask;

    srcfmt_val = detect_format(src_fmt);
    dstfmt_val = detect_format(dst_fmt);

    incy = ((Uint64)info->src_h << 16) / info->dst_h;
    incx = ((Uint64)info->src_w << 16) / info->dst_w;
    posy = incy / 2; /* start at the middle of pixel */

    while (info->dst_h--) {
        Uint8 *src = 0;
        Uint8 *dst = info->dst;
        int n = info->dst_w;
        posx = incx / 2; /* start at the middle of pixel */
        srcy = posy >> 16;
        while (n--) {
            srcx = posx >> 16;
            src = (info->src + (srcy * info->src_pitch) + (srcx * srcbpp));

            if (FORMAT_HAS_ALPHA(srcfmt_val)) {
                DISEMBLE_RGBA(src, srcbpp, src_fmt, srcpixel, srcR, srcG, srcB, srcA);
            } else if (FORMAT_HAS_NO_ALPHA(srcfmt_val)) {
                DISEMBLE_RGB(src, srcbpp, src_fmt, srcpixel, srcR, srcG, srcB);
                srcA = 0xFF;
            } else {
                /* SDL_PIXELFORMAT_ARGB2101010 */
                srcpixel = *((Uint32 *)(src));
                RGBA_FROM_ARGB2101010(srcpixel, srcR, srcG, srcB, srcA);
            }

            if (flags & SDL_COPY_COLORKEY) {
                /* srcpixel isn't set for 24 bpp */
                if (srcbpp == 3) {
                    srcpixel = (srcR << src_fmt->Rshift) |
                               (srcG << src_fmt->Gshift) | (srcB << src_fmt->Bshift);
                }
                if ((srcpixel & rgbmask) == ckey) {
                    posx += incx;
                    dst += dstbpp;
                    continue;
                }
            }
            if ((flags & (SDL_COPY_BLEND | SDL_COPY_ADD | SDL_COPY_MOD | SDL_COPY_MUL))) {
                if (FORMAT_HAS_ALPHA(dstfmt_val)) {
                    DISEMBLE_RGBA(dst, dstbpp, dst_fmt, dstpixel, dstR, dstG, dstB, dstA);
                } else if (FORMAT_HAS_NO_ALPHA(dstfmt_val)) {
                    DISEMBLE_RGB(dst, dstbpp, dst_fmt, dstpixel, dstR, dstG, dstB);
                    dstA = 0xFF;
                } else {
                    /* SDL_PIXELFORMAT_ARGB2101010 */
                    dstpixel = *((Uint32 *) (dst));
                    RGBA_FROM_ARGB2101010(dstpixel, dstR, dstG, dstB, dstA);
                }
            } else {
                /* don't care */
                dstR = dstG = dstB = dstA = 0;
            }

            if (flags & SDL_COPY_MODULATE_COLOR) {
                srcR = (srcR * modulateR) / 255;
                srcG = (srcG * modulateG) / 255;
                srcB = (srcB * modulateB) / 255;
            }
            if (flags & SDL_COPY_MODULATE_ALPHA) {
                srcA = (srcA * modulateA) / 255;
            }
            if (flags & (SDL_COPY_BLEND | SDL_COPY_ADD)) {
                /* This goes away if we ever use premultiplied alpha */
                if (srcA < 255) {
                    srcR = (srcR * srcA) / 255;
                    srcG = (srcG * srcA) / 255;
                    srcB = (srcB * srcA) / 255;
                }
            }
            switch (flags & (SDL_COPY_BLEND | SDL_COPY_ADD | SDL_COPY_MOD | SDL_COPY_MUL)) {
            case 0:
                dstR = srcR;
                dstG = srcG;
                dstB = srcB;
                dstA = srcA;
                break;
            case SDL_COPY_BLEND:
                dstR = srcR + ((255 - srcA) * dstR) / 255;
                dstG = srcG + ((255 - srcA) * dstG) / 255;
                dstB = srcB + ((255 - srcA) * dstB) / 255;
                dstA = srcA + ((255 - srcA) * dstA) / 255;
                break;
            case SDL_COPY_ADD:
                dstR = srcR + dstR;
                if (dstR > 255) {
                    dstR = 255;
                }
                dstG = srcG + dstG;
                if (dstG > 255) {
                    dstG = 255;
                }
                dstB = srcB + dstB;
                if (dstB > 255) {
                    dstB = 255;
                }
                break;
            case SDL_COPY_MOD:
                dstR = (srcR * dstR) / 255;
                dstG = (srcG * dstG) / 255;
                dstB = (srcB * dstB) / 255;
                break;
            case SDL_COPY_MUL:
                dstR = ((srcR * dstR) + (dstR * (255 - srcA))) / 255;
                if (dstR > 255) {
                    dstR = 255;
                }
                dstG = ((srcG * dstG) + (dstG * (255 - srcA))) / 255;
                if (dstG > 255) {
                    dstG = 255;
                }
                dstB = ((srcB * dstB) + (dstB * (255 - srcA))) / 255;
                if (dstB > 255) {
                    dstB = 255;
                }
                break;
            }
            if (FORMAT_HAS_ALPHA(dstfmt_val)) {
                ASSEMBLE_RGBA(dst, dstbpp, dst_fmt, dstR, dstG, dstB, dstA);
            } else if (FORMAT_HAS_NO_ALPHA(dstfmt_val)) {
                ASSEMBLE_RGB(dst, dstbpp, dst_fmt, dstR, dstG, dstB);
            } else {
                /* SDL_PIXELFORMAT_ARGB2101010 */
                Uint32 pixel;
                ARGB2101010_FROM_RGBA(pixel, dstR, dstG, dstB, dstA);
                *(Uint32 *)dst = pixel;
            }
            posx += incx;
            dst += dstbpp;
        }
        posy += incy;
        info->dst += info->dst_pitch;
    }
}

/* vi: set ts=4 sw=4 expandtab: */
