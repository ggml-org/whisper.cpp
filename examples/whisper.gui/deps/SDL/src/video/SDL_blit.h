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

#ifndef SDL_blit_h_
#define SDL_blit_h_

#include "SDL_cpuinfo.h"
#include "SDL_endian.h"
#include "SDL_surface.h"

/* pixman ARM blitters are 32 bit only : */
#if defined(__aarch64__) || defined(_M_ARM64)
#undef SDL_ARM_SIMD_BLITTERS
#undef SDL_ARM_NEON_BLITTERS
#endif

/* Table to do pixel byte expansion */
extern Uint8 *SDL_expand_byte[9];

/* SDL blit copy flags */
#define SDL_COPY_MODULATE_COLOR 0x00000001
#define SDL_COPY_MODULATE_ALPHA 0x00000002
#define SDL_COPY_BLEND          0x00000010
#define SDL_COPY_ADD            0x00000020
#define SDL_COPY_MOD            0x00000040
#define SDL_COPY_MUL            0x00000080
#define SDL_COPY_COLORKEY       0x00000100
#define SDL_COPY_NEAREST        0x00000200
#define SDL_COPY_RLE_DESIRED    0x00001000
#define SDL_COPY_RLE_COLORKEY   0x00002000
#define SDL_COPY_RLE_ALPHAKEY   0x00004000
#define SDL_COPY_RLE_MASK       (SDL_COPY_RLE_DESIRED | SDL_COPY_RLE_COLORKEY | SDL_COPY_RLE_ALPHAKEY)

/* SDL blit CPU flags */
#define SDL_CPU_ANY                0x00000000
#define SDL_CPU_MMX                0x00000001
#define SDL_CPU_3DNOW              0x00000002
#define SDL_CPU_SSE                0x00000004
#define SDL_CPU_SSE2               0x00000008
#define SDL_CPU_ALTIVEC_PREFETCH   0x00000010
#define SDL_CPU_ALTIVEC_NOPREFETCH 0x00000020

typedef struct
{
    Uint8 *src;
    int src_w, src_h;
    int src_pitch;
    int src_skip;
    Uint8 *dst;
    int dst_w, dst_h;
    int dst_pitch;
    int dst_skip;
    SDL_PixelFormat *src_fmt;
    SDL_PixelFormat *dst_fmt;
    Uint8 *table;
    int flags;
    Uint32 colorkey;
    Uint8 r, g, b, a;
} SDL_BlitInfo;

typedef void (*SDL_BlitFunc)(SDL_BlitInfo *info);

typedef struct
{
    Uint32 src_format;
    Uint32 dst_format;
    int flags;
    int cpu;
    SDL_BlitFunc func;
} SDL_BlitFuncEntry;

/* Blit mapping definition */
/* typedef'ed in SDL_surface.h */
struct SDL_BlitMap
{
    SDL_Surface *dst;
    int identity;
    SDL_blit blit;
    void *data;
    SDL_BlitInfo info;

    /* the version count matches the destination; mismatch indicates
       an invalid mapping */
    Uint32 dst_palette_version;
    Uint32 src_palette_version;
};

/* Functions found in SDL_blit.c */
extern int SDL_CalculateBlit(SDL_Surface *surface);

/* Functions found in SDL_blit_*.c */
extern SDL_BlitFunc SDL_CalculateBlit0(SDL_Surface *surface);
extern SDL_BlitFunc SDL_CalculateBlit1(SDL_Surface *surface);
extern SDL_BlitFunc SDL_CalculateBlitN(SDL_Surface *surface);
extern SDL_BlitFunc SDL_CalculateBlitA(SDL_Surface *surface);

/*
 * Useful macros for blitting routines
 */

#if defined(__GNUC__)
#define DECLARE_ALIGNED(t, v, a) t __attribute__((aligned(a))) v
#elif defined(_MSC_VER)
#define DECLARE_ALIGNED(t, v, a) __declspec(align(a)) t v
#else
#define DECLARE_ALIGNED(t, v, a) t v
#endif

/* Load pixel of the specified format from a buffer and get its R-G-B values */
#define RGB_FROM_PIXEL(Pixel, fmt, r, g, b)                                     \
    {                                                                           \
        r = SDL_expand_byte[fmt->Rloss][((Pixel & fmt->Rmask) >> fmt->Rshift)]; \
        g = SDL_expand_byte[fmt->Gloss][((Pixel & fmt->Gmask) >> fmt->Gshift)]; \
        b = SDL_expand_byte[fmt->Bloss][((Pixel & fmt->Bmask) >> fmt->Bshift)]; \
    }
#define RGB_FROM_RGB565(Pixel, r, g, b)                   \
    {                                                     \
        r = SDL_expand_byte[3][((Pixel & 0xF800) >> 11)]; \
        g = SDL_expand_byte[2][((Pixel & 0x07E0) >> 5)];  \
        b = SDL_expand_byte[3][(Pixel & 0x001F)];         \
    }
#define RGB_FROM_RGB555(Pixel, r, g, b)                   \
    {                                                     \
        r = SDL_expand_byte[3][((Pixel & 0x7C00) >> 10)]; \
        g = SDL_expand_byte[3][((Pixel & 0x03E0) >> 5)];  \
        b = SDL_expand_byte[3][(Pixel & 0x001F)];         \
    }
#define RGB_FROM_RGB888(Pixel, r, g, b) \
    {                                   \
        r = ((Pixel & 0xFF0000) >> 16); \
        g = ((Pixel & 0xFF00) >> 8);    \
        b = (Pixel & 0xFF);             \
    }
#define RETRIEVE_RGB_PIXEL(buf, bpp, Pixel)                \
    do {                                                   \
        switch (bpp) {                                     \
        case 1:                                            \
            Pixel = *((Uint8 *)(buf));                     \
            break;                                         \
                                                           \
        case 2:                                            \
            Pixel = *((Uint16 *)(buf));                    \
            break;                                         \
                                                           \
        case 3:                                            \
        {                                                  \
            Uint8 *B = (Uint8 *)(buf);                     \
            if (SDL_BYTEORDER == SDL_LIL_ENDIAN) {         \
                Pixel = B[0] + (B[1] << 8) + (B[2] << 16); \
            } else {                                       \
                Pixel = (B[0] << 16) + (B[1] << 8) + B[2]; \
            }                                              \
        } break;                                           \
                                                           \
        case 4:                                            \
            Pixel = *((Uint32 *)(buf));                    \
            break;                                         \
                                                           \
        default:                                           \
            Pixel = 0; /* stop gcc complaints */           \
            break;                                         \
        }                                                  \
    } while (0)

#define DISEMBLE_RGB(buf, bpp, fmt, Pixel, r, g, b) \
    do {                                            \
        switch (bpp) {                              \
        case 1:                                     \
            Pixel = *((Uint8 *)(buf));              \
            RGB_FROM_PIXEL(Pixel, fmt, r, g, b);    \
            break;                                  \
                                                    \
        case 2:                                     \
            Pixel = *((Uint16 *)(buf));             \
            RGB_FROM_PIXEL(Pixel, fmt, r, g, b);    \
            break;                                  \
                                                    \
        case 3:                                     \
        {                                           \
            Pixel = 0;                              \
            if (SDL_BYTEORDER == SDL_LIL_ENDIAN) {  \
                r = *((buf) + fmt->Rshift / 8);     \
                g = *((buf) + fmt->Gshift / 8);     \
                b = *((buf) + fmt->Bshift / 8);     \
            } else {                                \
                r = *((buf) + 2 - fmt->Rshift / 8); \
                g = *((buf) + 2 - fmt->Gshift / 8); \
                b = *((buf) + 2 - fmt->Bshift / 8); \
            }                                       \
        } break;                                    \
                                                    \
        case 4:                                     \
            Pixel = *((Uint32 *)(buf));             \
            RGB_FROM_PIXEL(Pixel, fmt, r, g, b);    \
            break;                                  \
                                                    \
        default:                                    \
            /* stop gcc complaints */               \
            Pixel = 0;                              \
            r = g = b = 0;                          \
            break;                                  \
        }                                           \
    } while (0)

/* Assemble R-G-B values into a specified pixel format and store them */
#define PIXEL_FROM_RGB(Pixel, fmt, r, g, b)          \
    {                                                \
        Pixel = ((r >> fmt->Rloss) << fmt->Rshift) | \
                ((g >> fmt->Gloss) << fmt->Gshift) | \
                ((b >> fmt->Bloss) << fmt->Bshift) | \
                fmt->Amask;                          \
    }
#define RGB565_FROM_RGB(Pixel, r, g, b)                        \
    {                                                          \
        Pixel = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3); \
    }
#define RGB555_FROM_RGB(Pixel, r, g, b)                        \
    {                                                          \
        Pixel = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3); \
    }
#define RGB888_FROM_RGB(Pixel, r, g, b)   \
    {                                     \
        Pixel = (r << 16) | (g << 8) | b; \
    }
#define ARGB8888_FROM_RGBA(Pixel, r, g, b, a)         \
    {                                                 \
        Pixel = (a << 24) | (r << 16) | (g << 8) | b; \
    }
#define RGBA8888_FROM_RGBA(Pixel, r, g, b, a)         \
    {                                                 \
        Pixel = (r << 24) | (g << 16) | (b << 8) | a; \
    }
#define ABGR8888_FROM_RGBA(Pixel, r, g, b, a)         \
    {                                                 \
        Pixel = (a << 24) | (b << 16) | (g << 8) | r; \
    }
#define BGRA8888_FROM_RGBA(Pixel, r, g, b, a)         \
    {                                                 \
        Pixel = (b << 24) | (g << 16) | (r << 8) | a; \
    }
#define ARGB2101010_FROM_RGBA(Pixel, r, g, b, a)       \
    {                                                  \
        r = r ? ((r << 2) | 0x3) : 0;                  \
        g = g ? ((g << 2) | 0x3) : 0;                  \
        b = b ? ((b << 2) | 0x3) : 0;                  \
        a = (a * 3) / 255;                             \
        Pixel = (a << 30) | (r << 20) | (g << 10) | b; \
    }
#define ASSEMBLE_RGB(buf, bpp, fmt, r, g, b)        \
    {                                               \
        switch (bpp) {                              \
        case 1:                                     \
        {                                           \
            Uint8 _pixel;                           \
                                                    \
            PIXEL_FROM_RGB(_pixel, fmt, r, g, b);   \
            *((Uint8 *)(buf)) = _pixel;             \
        } break;                                    \
                                                    \
        case 2:                                     \
        {                                           \
            Uint16 _pixel;                          \
                                                    \
            PIXEL_FROM_RGB(_pixel, fmt, r, g, b);   \
            *((Uint16 *)(buf)) = _pixel;            \
        } break;                                    \
                                                    \
        case 3:                                     \
        {                                           \
            if (SDL_BYTEORDER == SDL_LIL_ENDIAN) {  \
                *((buf) + fmt->Rshift / 8) = r;     \
                *((buf) + fmt->Gshift / 8) = g;     \
                *((buf) + fmt->Bshift / 8) = b;     \
            } else {                                \
                *((buf) + 2 - fmt->Rshift / 8) = r; \
                *((buf) + 2 - fmt->Gshift / 8) = g; \
                *((buf) + 2 - fmt->Bshift / 8) = b; \
            }                                       \
        } break;                                    \
                                                    \
        case 4:                                     \
        {                                           \
            Uint32 _pixel;                          \
                                                    \
            PIXEL_FROM_RGB(_pixel, fmt, r, g, b);   \
            *((Uint32 *)(buf)) = _pixel;            \
        } break;                                    \
        }                                           \
    }

/* FIXME: Should we rescale alpha into 0..255 here? */
#define RGBA_FROM_PIXEL(Pixel, fmt, r, g, b, a)                                 \
    {                                                                           \
        r = SDL_expand_byte[fmt->Rloss][((Pixel & fmt->Rmask) >> fmt->Rshift)]; \
        g = SDL_expand_byte[fmt->Gloss][((Pixel & fmt->Gmask) >> fmt->Gshift)]; \
        b = SDL_expand_byte[fmt->Bloss][((Pixel & fmt->Bmask) >> fmt->Bshift)]; \
        a = SDL_expand_byte[fmt->Aloss][((Pixel & fmt->Amask) >> fmt->Ashift)]; \
    }
#define RGBA_FROM_8888(Pixel, fmt, r, g, b, a)   \
    {                                            \
        r = (Pixel & fmt->Rmask) >> fmt->Rshift; \
        g = (Pixel & fmt->Gmask) >> fmt->Gshift; \
        b = (Pixel & fmt->Bmask) >> fmt->Bshift; \
        a = (Pixel & fmt->Amask) >> fmt->Ashift; \
    }
#define RGBA_FROM_RGBA8888(Pixel, r, g, b, a) \
    {                                         \
        r = (Pixel >> 24);                    \
        g = ((Pixel >> 16) & 0xFF);           \
        b = ((Pixel >> 8) & 0xFF);            \
        a = (Pixel & 0xFF);                   \
    }
#define RGBA_FROM_ARGB8888(Pixel, r, g, b, a) \
    {                                         \
        r = ((Pixel >> 16) & 0xFF);           \
        g = ((Pixel >> 8) & 0xFF);            \
        b = (Pixel & 0xFF);                   \
        a = (Pixel >> 24);                    \
    }
#define RGBA_FROM_ABGR8888(Pixel, r, g, b, a) \
    {                                         \
        r = (Pixel & 0xFF);                   \
        g = ((Pixel >> 8) & 0xFF);            \
        b = ((Pixel >> 16) & 0xFF);           \
        a = (Pixel >> 24);                    \
    }
#define RGBA_FROM_BGRA8888(Pixel, r, g, b, a) \
    {                                         \
        r = ((Pixel >> 8) & 0xFF);            \
        g = ((Pixel >> 16) & 0xFF);           \
        b = (Pixel >> 24);                    \
        a = (Pixel & 0xFF);                   \
    }
#define RGBA_FROM_ARGB2101010(Pixel, r, g, b, a) \
    {                                            \
        r = ((Pixel >> 22) & 0xFF);              \
        g = ((Pixel >> 12) & 0xFF);              \
        b = ((Pixel >> 2) & 0xFF);               \
        a = SDL_expand_byte[6][(Pixel >> 30)];   \
    }
#define DISEMBLE_RGBA(buf, bpp, fmt, Pixel, r, g, b, a) \
    do {                                                \
        switch (bpp) {                                  \
        case 1:                                         \
            Pixel = *((Uint8 *)(buf));                  \
            RGBA_FROM_PIXEL(Pixel, fmt, r, g, b, a);    \
            break;                                      \
                                                        \
        case 2:                                         \
            Pixel = *((Uint16 *)(buf));                 \
            RGBA_FROM_PIXEL(Pixel, fmt, r, g, b, a);    \
            break;                                      \
                                                        \
        case 3:                                         \
        {                                               \
            Pixel = 0;                                  \
            if (SDL_BYTEORDER == SDL_LIL_ENDIAN) {      \
                r = *((buf) + fmt->Rshift / 8);         \
                g = *((buf) + fmt->Gshift / 8);         \
                b = *((buf) + fmt->Bshift / 8);         \
            } else {                                    \
                r = *((buf) + 2 - fmt->Rshift / 8);     \
                g = *((buf) + 2 - fmt->Gshift / 8);     \
                b = *((buf) + 2 - fmt->Bshift / 8);     \
            }                                           \
            a = 0xFF;                                   \
        } break;                                        \
                                                        \
        case 4:                                         \
            Pixel = *((Uint32 *)(buf));                 \
            RGBA_FROM_PIXEL(Pixel, fmt, r, g, b, a);    \
            break;                                      \
                                                        \
        default:                                        \
            /* stop gcc complaints */                   \
            Pixel = 0;                                  \
            r = g = b = a = 0;                          \
            break;                                      \
        }                                               \
    } while (0)

/* FIXME: this isn't correct, especially for Alpha (maximum != 255) */
#define PIXEL_FROM_RGBA(Pixel, fmt, r, g, b, a)      \
    {                                                \
        Pixel = ((r >> fmt->Rloss) << fmt->Rshift) | \
                ((g >> fmt->Gloss) << fmt->Gshift) | \
                ((b >> fmt->Bloss) << fmt->Bshift) | \
                ((a >> fmt->Aloss) << fmt->Ashift);  \
    }
#define ASSEMBLE_RGBA(buf, bpp, fmt, r, g, b, a)      \
    {                                                 \
        switch (bpp) {                                \
        case 1:                                       \
        {                                             \
            Uint8 _pixel;                             \
                                                      \
            PIXEL_FROM_RGBA(_pixel, fmt, r, g, b, a); \
            *((Uint8 *)(buf)) = _pixel;               \
        } break;                                      \
                                                      \
        case 2:                                       \
        {                                             \
            Uint16 _pixel;                            \
                                                      \
            PIXEL_FROM_RGBA(_pixel, fmt, r, g, b, a); \
            *((Uint16 *)(buf)) = _pixel;              \
        } break;                                      \
                                                      \
        case 3:                                       \
        {                                             \
            if (SDL_BYTEORDER == SDL_LIL_ENDIAN) {    \
                *((buf) + fmt->Rshift / 8) = r;       \
                *((buf) + fmt->Gshift / 8) = g;       \
                *((buf) + fmt->Bshift / 8) = b;       \
            } else {                                  \
                *((buf) + 2 - fmt->Rshift / 8) = r;   \
                *((buf) + 2 - fmt->Gshift / 8) = g;   \
                *((buf) + 2 - fmt->Bshift / 8) = b;   \
            }                                         \
        } break;                                      \
                                                      \
        case 4:                                       \
        {                                             \
            Uint32 _pixel;                            \
                                                      \
            PIXEL_FROM_RGBA(_pixel, fmt, r, g, b, a); \
            *((Uint32 *)(buf)) = _pixel;              \
        } break;                                      \
        }                                             \
    }

/* Blend the RGB values of two pixels with an alpha value */
#define ALPHA_BLEND_RGB(sR, sG, sB, A, dR, dG, dB)            \
    do {                                                      \
        dR = (Uint8)((((int)(sR - dR) * (int)A) / 255) + dR); \
        dG = (Uint8)((((int)(sG - dG) * (int)A) / 255) + dG); \
        dB = (Uint8)((((int)(sB - dB) * (int)A) / 255) + dB); \
    } while (0)

/* Blend the RGBA values of two pixels */
#define ALPHA_BLEND_RGBA(sR, sG, sB, sA, dR, dG, dB, dA)       \
    do {                                                       \
        dR = (Uint8)((((int)(sR - dR) * (int)sA) / 255) + dR); \
        dG = (Uint8)((((int)(sG - dG) * (int)sA) / 255) + dG); \
        dB = (Uint8)((((int)(sB - dB) * (int)sA) / 255) + dB); \
        dA = (Uint8)((int)sA + dA - ((int)sA * dA) / 255);     \
    } while (0)

/* This is a very useful loop for optimizing blitters */
#if defined(_MSC_VER) && (_MSC_VER == 1300)
/* There's a bug in the Visual C++ 7 optimizer when compiling this code */
#else
#define USE_DUFFS_LOOP
#endif
#ifdef USE_DUFFS_LOOP

/* 8-times unrolled loop */
#define DUFFS_LOOP8(pixel_copy_increment, width) \
    {                                            \
        int n = (width + 7) / 8;                 \
        switch (width & 7) {                     \
        case 0:                                  \
            do {                                 \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 7:                              \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 6:                              \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 5:                              \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 4:                              \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 3:                              \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 2:                              \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 1:                              \
                pixel_copy_increment;            \
            } while (--n > 0);                   \
        }                                        \
    }

/* 4-times unrolled loop */
#define DUFFS_LOOP4(pixel_copy_increment, width) \
    {                                            \
        int n = (width + 3) / 4;                 \
        switch (width & 3) {                     \
        case 0:                                  \
            do {                                 \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 3:                              \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 2:                              \
                pixel_copy_increment;            \
                SDL_FALLTHROUGH;                 \
            case 1:                              \
                pixel_copy_increment;            \
            } while (--n > 0);                   \
        }                                        \
    }

/* Use the 8-times version of the loop by default */
#define DUFFS_LOOP(pixel_copy_increment, width) \
    DUFFS_LOOP8(pixel_copy_increment, width)

/* Special version of Duff's device for even more optimization */
#define DUFFS_LOOP_124(pixel_copy_increment1,        \
                       pixel_copy_increment2,        \
                       pixel_copy_increment4, width) \
    {                                                \
        int n = width;                               \
        if (n & 1) {                                 \
            pixel_copy_increment1;                   \
            n -= 1;                                  \
        }                                            \
        if (n & 2) {                                 \
            pixel_copy_increment2;                   \
            n -= 2;                                  \
        }                                            \
        if (n & 4) {                                 \
            pixel_copy_increment4;                   \
            n -= 4;                                  \
        }                                            \
        if (n) {                                     \
            n /= 8;                                  \
            do {                                     \
                pixel_copy_increment4;               \
                pixel_copy_increment4;               \
            } while (--n > 0);                       \
        }                                            \
    }

#else

/* Don't use Duff's device to unroll loops */
#define DUFFS_LOOP(pixel_copy_increment, width) \
    {                                           \
        int n;                                  \
        for (n = width; n > 0; --n) {           \
            pixel_copy_increment;               \
        }                                       \
    }
#define DUFFS_LOOP8(pixel_copy_increment, width) \
    DUFFS_LOOP(pixel_copy_increment, width)
#define DUFFS_LOOP4(pixel_copy_increment, width) \
    DUFFS_LOOP(pixel_copy_increment, width)
#define DUFFS_LOOP_124(pixel_copy_increment1,        \
                       pixel_copy_increment2,        \
                       pixel_copy_increment4, width) \
    DUFFS_LOOP(pixel_copy_increment1, width)

#endif /* USE_DUFFS_LOOP */

/* Prevent Visual C++ 6.0 from printing out stupid warnings */
#if defined(_MSC_VER) && (_MSC_VER >= 600)
#pragma warning(disable : 4550)
#endif

#endif /* SDL_blit_h_ */

/* vi: set ts=4 sw=4 expandtab: */
