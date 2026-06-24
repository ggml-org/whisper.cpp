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

/*
   Code to load and save surfaces in Windows BMP format.

   Why support BMP format?  Well, it's a native format for Windows, and
   most image processing programs can read and write it.  It would be nice
   to be able to have at least one image format that we can natively load
   and save, and since PNG is so complex that it would bloat the library,
   BMP is a good alternative.

   This code currently supports Win32 DIBs in uncompressed 8 and 24 bpp.
*/

#include "SDL_hints.h"
#include "SDL_video.h"
#include "SDL_endian.h"
#include "SDL_pixels_c.h"

#define SAVE_32BIT_BMP

/* Compression encodings for BMP files */
#ifndef BI_RGB
#define BI_RGB       0
#define BI_RLE8      1
#define BI_RLE4      2
#define BI_BITFIELDS 3
#endif

/* Logical color space values for BMP files */
#ifndef LCS_WINDOWS_COLOR_SPACE
/* 0x57696E20 == "Win " */
#define LCS_WINDOWS_COLOR_SPACE 0x57696E20
#endif

static SDL_bool readRlePixels(SDL_Surface *surface, SDL_RWops *src, int isRle8)
{
    /*
    | Sets the surface pixels from src.  A bmp image is upside down.
    */
    int pitch = surface->pitch;
    int height = surface->h;
    Uint8 *start = (Uint8 *)surface->pixels;
    Uint8 *end = start + (height * pitch);
    Uint8 *bits = end - pitch, *spot;
    int ofs = 0;
    Uint8 ch;
    Uint8 needsPad;

#define COPY_PIXEL(x)                \
    spot = &bits[ofs++];             \
    if (spot >= start && spot < end) \
    *spot = (x)

    for (;;) {
        if (!SDL_RWread(src, &ch, 1, 1)) {
            return SDL_TRUE;
        }
        /*
        | encoded mode starts with a run length, and then a byte
        | with two colour indexes to alternate between for the run
        */
        if (ch) {
            Uint8 pixel;
            if (!SDL_RWread(src, &pixel, 1, 1)) {
                return SDL_TRUE;
            }
            if (isRle8) { /* 256-color bitmap, compressed */
                do {
                    COPY_PIXEL(pixel);
                } while (--ch);
            } else { /* 16-color bitmap, compressed */
                Uint8 pixel0 = pixel >> 4;
                Uint8 pixel1 = pixel & 0x0F;
                for (;;) {
                    COPY_PIXEL(pixel0); /* even count, high nibble */
                    if (!--ch) {
                        break;
                    }
                    COPY_PIXEL(pixel1); /* odd count, low nibble */
                    if (!--ch) {
                        break;
                    }
                }
            }
        } else {
            /*
            | A leading zero is an escape; it may signal the end of the bitmap,
            | a cursor move, or some absolute data.
            | zero tag may be absolute mode or an escape
            */
            if (!SDL_RWread(src, &ch, 1, 1)) {
                return SDL_TRUE;
            }
            switch (ch) {
            case 0: /* end of line */
                ofs = 0;
                bits -= pitch; /* go to previous */
                break;
            case 1:               /* end of bitmap */
                return SDL_FALSE; /* success! */
            case 2:               /* delta */
                if (!SDL_RWread(src, &ch, 1, 1)) {
                    return SDL_TRUE;
                }
                ofs += ch;
                if (!SDL_RWread(src, &ch, 1, 1)) {
                    return SDL_TRUE;
                }
                bits -= (ch * pitch);
                break;
            default: /* no compression */
                if (isRle8) {
                    needsPad = (ch & 1);
                    do {
                        Uint8 pixel;
                        if (!SDL_RWread(src, &pixel, 1, 1)) {
                            return SDL_TRUE;
                        }
                        COPY_PIXEL(pixel);
                    } while (--ch);
                } else {
                    needsPad = (((ch + 1) >> 1) & 1); /* (ch+1)>>1: bytes size */
                    for (;;) {
                        Uint8 pixel;
                        if (!SDL_RWread(src, &pixel, 1, 1)) {
                            return SDL_TRUE;
                        }
                        COPY_PIXEL(pixel >> 4);
                        if (!--ch) {
                            break;
                        }
                        COPY_PIXEL(pixel & 0x0F);
                        if (!--ch) {
                            break;
                        }
                    }
                }
                /* pad at even boundary */
                if (needsPad && !SDL_RWread(src, &ch, 1, 1)) {
                    return SDL_TRUE;
                }
                break;
            }
        }
    }
}

static void CorrectAlphaChannel(SDL_Surface *surface)
{
    /* Check to see if there is any alpha channel data */
    SDL_bool hasAlpha = SDL_FALSE;
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
    int alphaChannelOffset = 0;
#else
    int alphaChannelOffset = 3;
#endif
    Uint8 *alpha = ((Uint8 *)surface->pixels) + alphaChannelOffset;
    Uint8 *end = alpha + surface->h * surface->pitch;

    while (alpha < end) {
        if (*alpha != 0) {
            hasAlpha = SDL_TRUE;
            break;
        }
        alpha += 4;
    }

    if (!hasAlpha) {
        alpha = ((Uint8 *)surface->pixels) + alphaChannelOffset;
        while (alpha < end) {
            *alpha = SDL_ALPHA_OPAQUE;
            alpha += 4;
        }
    }
}

SDL_Surface *SDL_LoadBMP_RW(SDL_RWops *src, int freesrc)
{
    SDL_bool was_error;
    Sint64 fp_offset = 0;
    int bmpPitch;
    int i, pad;
    SDL_Surface *surface;
    Uint32 Rmask = 0;
    Uint32 Gmask = 0;
    Uint32 Bmask = 0;
    Uint32 Amask = 0;
    SDL_Palette *palette;
    Uint8 *bits;
    Uint8 *top, *end;
    SDL_bool topDown;
    int ExpandBMP;
    SDL_bool haveRGBMasks = SDL_FALSE;
    SDL_bool haveAlphaMask = SDL_FALSE;
    SDL_bool correctAlpha = SDL_FALSE;

    /* The Win32 BMP file header (14 bytes) */
    char magic[2];
    /* Uint32 bfSize; */
    /* Uint16 bfReserved1; */
    /* Uint16 bfReserved2; */
    Uint32 bfOffBits;

    /* The Win32 BITMAPINFOHEADER struct (40 bytes) */
    Uint32 biSize;
    Sint32 biWidth = 0;
    Sint32 biHeight = 0;
    /* Uint16 biPlanes; */
    Uint16 biBitCount = 0;
    Uint32 biCompression = 0;
    /* Uint32 biSizeImage; */
    /* Sint32 biXPelsPerMeter; */
    /* Sint32 biYPelsPerMeter; */
    Uint32 biClrUsed = 0;
    /* Uint32 biClrImportant; */

    /* Make sure we are passed a valid data source */
    surface = NULL;
    was_error = SDL_FALSE;
    if (!src) {
        SDL_InvalidParamError("src");
        was_error = SDL_TRUE;
        goto done;
    }

    /* Read in the BMP file header */
    fp_offset = SDL_RWtell(src);
    if (fp_offset < 0) {
        was_error = SDL_TRUE;
        goto done;
    }
    SDL_ClearError();
    if (SDL_RWread(src, magic, 1, 2) != 2) {
        SDL_Error(SDL_EFREAD);
        was_error = SDL_TRUE;
        goto done;
    }
    if (SDL_strncmp(magic, "BM", 2) != 0) {
        SDL_SetError("File is not a Windows BMP file");
        was_error = SDL_TRUE;
        goto done;
    }
    /* bfSize      = */ SDL_ReadLE32(src);
    /* bfReserved1 = */ SDL_ReadLE16(src);
    /* bfReserved2 = */ SDL_ReadLE16(src);
    bfOffBits = SDL_ReadLE32(src);

    /* Read the Win32 BITMAPINFOHEADER */
    biSize = SDL_ReadLE32(src);
    if (biSize == 12) { /* really old BITMAPCOREHEADER */
        biWidth = (Uint32)SDL_ReadLE16(src);
        biHeight = (Uint32)SDL_ReadLE16(src);
        /* biPlanes = */ SDL_ReadLE16(src);
        biBitCount = SDL_ReadLE16(src);
        biCompression = BI_RGB;
        /* biSizeImage = 0; */
        /* biXPelsPerMeter = 0; */
        /* biYPelsPerMeter = 0; */
        biClrUsed = 0;
        /* biClrImportant = 0; */
    } else if (biSize >= 40) { /* some version of BITMAPINFOHEADER */
        Uint32 headerSize;
        biWidth = SDL_ReadLE32(src);
        biHeight = SDL_ReadLE32(src);
        /* biPlanes = */ SDL_ReadLE16(src);
        biBitCount = SDL_ReadLE16(src);
        biCompression = SDL_ReadLE32(src);
        /* biSizeImage = */ SDL_ReadLE32(src);
        /* biXPelsPerMeter = */ SDL_ReadLE32(src);
        /* biYPelsPerMeter = */ SDL_ReadLE32(src);
        biClrUsed = SDL_ReadLE32(src);
        /* biClrImportant = */ SDL_ReadLE32(src);

        /* 64 == BITMAPCOREHEADER2, an incompatible OS/2 2.x extension. Skip this stuff for now. */
        if (biSize != 64) {
            /* This is complicated. If compression is BI_BITFIELDS, then
               we have 3 DWORDS that specify the RGB masks. This is either
               stored here in an BITMAPV2INFOHEADER (which only differs in
               that it adds these RGB masks) and biSize >= 52, or we've got
               these masks stored in the exact same place, but strictly
               speaking, this is the bmiColors field in BITMAPINFO immediately
               following the legacy v1 info header, just past biSize. */
            if (biCompression == BI_BITFIELDS) {
                haveRGBMasks = SDL_TRUE;
                Rmask = SDL_ReadLE32(src);
                Gmask = SDL_ReadLE32(src);
                Bmask = SDL_ReadLE32(src);

                /* ...v3 adds an alpha mask. */
                if (biSize >= 56) { /* BITMAPV3INFOHEADER; adds alpha mask */
                    haveAlphaMask = SDL_TRUE;
                    Amask = SDL_ReadLE32(src);
                }
            } else {
                /* the mask fields are ignored for v2+ headers if not BI_BITFIELD. */
                if (biSize >= 52) { /* BITMAPV2INFOHEADER; adds RGB masks */
                    /*Rmask = */ SDL_ReadLE32(src);
                    /*Gmask = */ SDL_ReadLE32(src);
                    /*Bmask = */ SDL_ReadLE32(src);
                }
                if (biSize >= 56) { /* BITMAPV3INFOHEADER; adds alpha mask */
                    /*Amask = */ SDL_ReadLE32(src);
                }
            }

            /* Insert other fields here; Wikipedia and MSDN say we're up to
               v5 of this header, but we ignore those for now (they add gamma,
               color spaces, etc). Ignoring the weird OS/2 2.x format, we
               currently parse up to v3 correctly (hopefully!). */
        }

        /* skip any header bytes we didn't handle... */
        headerSize = (Uint32)(SDL_RWtell(src) - (fp_offset + 14));
        if (biSize > headerSize) {
            SDL_RWseek(src, (biSize - headerSize), RW_SEEK_CUR);
        }
    }
    if (biWidth <= 0 || biHeight == 0) {
        SDL_SetError("BMP file with bad dimensions (%" SDL_PRIs32 "x%" SDL_PRIs32 ")", biWidth, biHeight);
        was_error = SDL_TRUE;
        goto done;
    }
    if (biHeight < 0) {
        topDown = SDL_TRUE;
        biHeight = -biHeight;
    } else {
        topDown = SDL_FALSE;
    }

    /* Check for read error */
    if (SDL_strcmp(SDL_GetError(), "") != 0) {
        was_error = SDL_TRUE;
        goto done;
    }

    /* Expand 1, 2 and 4 bit bitmaps to 8 bits per pixel */
    switch (biBitCount) {
    case 1:
    case 2:
    case 4:
        ExpandBMP = biBitCount;
        biBitCount = 8;
        break;
    case 0:
    case 3:
    case 5:
    case 6:
    case 7:
        SDL_SetError("%d-bpp BMP images are not supported", biBitCount);
        was_error = SDL_TRUE;
        goto done;
    default:
        ExpandBMP = 0;
        break;
    }

    /* RLE4 and RLE8 BMP compression is supported */
    switch (biCompression) {
    case BI_RGB:
        /* If there are no masks, use the defaults */
        SDL_assert(!haveRGBMasks);
        SDL_assert(!haveAlphaMask);
        /* Default values for the BMP format */
        switch (biBitCount) {
        case 15:
        case 16:
            Rmask = 0x7C00;
            Gmask = 0x03E0;
            Bmask = 0x001F;
            break;
        case 24:
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
            Rmask = 0x000000FF;
            Gmask = 0x0000FF00;
            Bmask = 0x00FF0000;
#else
            Rmask = 0x00FF0000;
            Gmask = 0x0000FF00;
            Bmask = 0x000000FF;
#endif
            break;
        case 32:
            /* We don't know if this has alpha channel or not */
            correctAlpha = SDL_TRUE;
            Amask = 0xFF000000;
            Rmask = 0x00FF0000;
            Gmask = 0x0000FF00;
            Bmask = 0x000000FF;
            break;
        default:
            break;
        }
        break;

    case BI_BITFIELDS:
        break; /* we handled this in the info header. */

    default:
        break;
    }

    /* Create a compatible surface, note that the colors are RGB ordered */
    surface =
        SDL_CreateRGBSurface(0, biWidth, biHeight, biBitCount, Rmask, Gmask,
                             Bmask, Amask);
    if (!surface) {
        was_error = SDL_TRUE;
        goto done;
    }

    /* Load the palette, if any */
    palette = (surface->format)->palette;
    if (palette) {
        if (SDL_RWseek(src, fp_offset + 14 + biSize, RW_SEEK_SET) < 0) {
            SDL_Error(SDL_EFSEEK);
            was_error = SDL_TRUE;
            goto done;
        }

        if (biBitCount >= 32) { /* we shift biClrUsed by this value later. */
            SDL_SetError("Unsupported or incorrect biBitCount field");
            was_error = SDL_TRUE;
            goto done;
        }

        if (biClrUsed == 0) {
            biClrUsed = 1 << biBitCount;
        }

        if (biClrUsed > (Uint32)palette->ncolors) {
            biClrUsed = 1 << biBitCount; /* try forcing it? */
            if (biClrUsed > (Uint32)palette->ncolors) {
                SDL_SetError("Unsupported or incorrect biClrUsed field");
                was_error = SDL_TRUE;
                goto done;
            }
        }

        if (biSize == 12) {
            for (i = 0; i < (int)biClrUsed; ++i) {
                SDL_RWread(src, &palette->colors[i].b, 1, 1);
                SDL_RWread(src, &palette->colors[i].g, 1, 1);
                SDL_RWread(src, &palette->colors[i].r, 1, 1);
                palette->colors[i].a = SDL_ALPHA_OPAQUE;
            }
        } else {
            for (i = 0; i < (int)biClrUsed; ++i) {
                SDL_RWread(src, &palette->colors[i].b, 1, 1);
                SDL_RWread(src, &palette->colors[i].g, 1, 1);
                SDL_RWread(src, &palette->colors[i].r, 1, 1);
                SDL_RWread(src, &palette->colors[i].a, 1, 1);

                /* According to Microsoft documentation, the fourth element
                   is reserved and must be zero, so we shouldn't treat it as
                   alpha.
                */
                palette->colors[i].a = SDL_ALPHA_OPAQUE;
            }
        }
        palette->ncolors = biClrUsed;
    }

    /* Read the surface pixels.  Note that the bmp image is upside down */
    if (SDL_RWseek(src, fp_offset + bfOffBits, RW_SEEK_SET) < 0) {
        SDL_Error(SDL_EFSEEK);
        was_error = SDL_TRUE;
        goto done;
    }
    if ((biCompression == BI_RLE4) || (biCompression == BI_RLE8)) {
        was_error = readRlePixels(surface, src, biCompression == BI_RLE8);
        if (was_error) {
            SDL_Error(SDL_EFREAD);
        }
        goto done;
    }
    top = (Uint8 *)surface->pixels;
    end = (Uint8 *)surface->pixels + (surface->h * surface->pitch);
    switch (ExpandBMP) {
    case 1:
        bmpPitch = (biWidth + 7) >> 3;
        pad = (((bmpPitch) % 4) ? (4 - ((bmpPitch) % 4)) : 0);
        break;
    case 2:
        bmpPitch = (biWidth + 3) >> 2;
        pad = (((bmpPitch) % 4) ? (4 - ((bmpPitch) % 4)) : 0);
        break;
    case 4:
        bmpPitch = (biWidth + 1) >> 1;
        pad = (((bmpPitch) % 4) ? (4 - ((bmpPitch) % 4)) : 0);
        break;
    default:
        pad = ((surface->pitch % 4) ? (4 - (surface->pitch % 4)) : 0);
        break;
    }
    if (topDown) {
        bits = top;
    } else {
        bits = end - surface->pitch;
    }
    while (bits >= top && bits < end) {
        switch (ExpandBMP) {
        case 1:
        case 2:
        case 4:
        {
            Uint8 pixel = 0;
            int shift = (8 - ExpandBMP);
            for (i = 0; i < surface->w; ++i) {
                if (i % (8 / ExpandBMP) == 0) {
                    if (!SDL_RWread(src, &pixel, 1, 1)) {
                        SDL_Error(SDL_EFREAD);
                        was_error = SDL_TRUE;
                        goto done;
                    }
                }
                bits[i] = (pixel >> shift);
                if (bits[i] >= biClrUsed) {
                    SDL_SetError("A BMP image contains a pixel with a color out of the palette");
                    was_error = SDL_TRUE;
                    goto done;
                }
                pixel <<= ExpandBMP;
            }
        } break;

        default:
            if (SDL_RWread(src, bits, 1, surface->pitch) != surface->pitch) {
                SDL_Error(SDL_EFREAD);
                was_error = SDL_TRUE;
                goto done;
            }
            if (biBitCount == 8 && palette && biClrUsed < (1u << biBitCount)) {
                for (i = 0; i < surface->w; ++i) {
                    if (bits[i] >= biClrUsed) {
                        SDL_SetError("A BMP image contains a pixel with a color out of the palette");
                        was_error = SDL_TRUE;
                        goto done;
                    }
                }
            }
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
            /* Byte-swap the pixels if needed. Note that the 24bpp
               case has already been taken care of above. */
            switch (biBitCount) {
            case 15:
            case 16:
            {
                Uint16 *pix = (Uint16 *)bits;
                for (i = 0; i < surface->w; i++) {
                    pix[i] = SDL_Swap16(pix[i]);
                }
                break;
            }

            case 32:
            {
                Uint32 *pix = (Uint32 *)bits;
                for (i = 0; i < surface->w; i++) {
                    pix[i] = SDL_Swap32(pix[i]);
                }
                break;
            }
            }
#endif
            break;
        }
        /* Skip padding bytes, ugh */
        if (pad) {
            Uint8 padbyte;
            for (i = 0; i < pad; ++i) {
                SDL_RWread(src, &padbyte, 1, 1);
            }
        }
        if (topDown) {
            bits += surface->pitch;
        } else {
            bits -= surface->pitch;
        }
    }
    if (correctAlpha) {
        CorrectAlphaChannel(surface);
    }
done:
    if (was_error) {
        if (src) {
            SDL_RWseek(src, fp_offset, RW_SEEK_SET);
        }
        SDL_FreeSurface(surface);
        surface = NULL;
    }
    if (freesrc && src) {
        SDL_RWclose(src);
    }
    return surface;
}

int SDL_SaveBMP_RW(SDL_Surface *surface, SDL_RWops *dst, int freedst)
{
    Sint64 fp_offset;
    int i, pad;
    SDL_Surface *intermediate_surface;
    Uint8 *bits;
    SDL_bool save32bit = SDL_FALSE;
    SDL_bool saveLegacyBMP = SDL_FALSE;

    /* The Win32 BMP file header (14 bytes) */
    char magic[2] = { 'B', 'M' };
    Uint32 bfSize;
    Uint16 bfReserved1;
    Uint16 bfReserved2;
    Uint32 bfOffBits;

    /* The Win32 BITMAPINFOHEADER struct (40 bytes) */
    Uint32 biSize;
    Sint32 biWidth;
    Sint32 biHeight;
    Uint16 biPlanes;
    Uint16 biBitCount;
    Uint32 biCompression;
    Uint32 biSizeImage;
    Sint32 biXPelsPerMeter;
    Sint32 biYPelsPerMeter;
    Uint32 biClrUsed;
    Uint32 biClrImportant;

    /* The additional header members from the Win32 BITMAPV4HEADER struct (108 bytes in total) */
    Uint32 bV4RedMask = 0;
    Uint32 bV4GreenMask = 0;
    Uint32 bV4BlueMask = 0;
    Uint32 bV4AlphaMask = 0;
    Uint32 bV4CSType = 0;
    Sint32 bV4Endpoints[3 * 3] = { 0 };
    Uint32 bV4GammaRed = 0;
    Uint32 bV4GammaGreen = 0;
    Uint32 bV4GammaBlue = 0;

    /* Make sure we have somewhere to save */
    intermediate_surface = NULL;
    if (dst) {
#ifdef SAVE_32BIT_BMP
        /* We can save alpha information in a 32-bit BMP */
        if (surface->format->BitsPerPixel >= 8 &&
            (surface->format->Amask != 0 ||
             surface->map->info.flags & SDL_COPY_COLORKEY)) {
            save32bit = SDL_TRUE;
        }
#endif /* SAVE_32BIT_BMP */

        if (surface->format->palette && !save32bit) {
            if (surface->format->BitsPerPixel == 8) {
                intermediate_surface = surface;
            } else {
                SDL_SetError("%d bpp BMP files not supported",
                             surface->format->BitsPerPixel);
            }
        } else if ((surface->format->BitsPerPixel == 24) && !save32bit &&
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                   (surface->format->Rmask == 0x00FF0000) &&
                   (surface->format->Gmask == 0x0000FF00) &&
                   (surface->format->Bmask == 0x000000FF)
#else
                   (surface->format->Rmask == 0x000000FF) &&
                   (surface->format->Gmask == 0x0000FF00) &&
                   (surface->format->Bmask == 0x00FF0000)
#endif
        ) {
            intermediate_surface = surface;
        } else {
            SDL_PixelFormat format;

            /* If the surface has a colorkey or alpha channel we'll save a
               32-bit BMP with alpha channel, otherwise save a 24-bit BMP. */
            if (save32bit) {
                SDL_InitFormat(&format, SDL_PIXELFORMAT_BGRA32);
            } else {
                SDL_InitFormat(&format, SDL_PIXELFORMAT_BGR24);
            }
            intermediate_surface = SDL_ConvertSurface(surface, &format, 0);
            if (!intermediate_surface) {
                SDL_SetError("Couldn't convert image to %d bpp",
                             format.BitsPerPixel);
            }
        }
    } else {
        /* Set no error here because it may overwrite a more useful message from
           SDL_RWFromFile() if SDL_SaveBMP_RW() is called from SDL_SaveBMP(). */
        return -1;
    }

    if (save32bit) {
        saveLegacyBMP = SDL_GetHintBoolean(SDL_HINT_BMP_SAVE_LEGACY_FORMAT, SDL_FALSE);
    }

    if (intermediate_surface && (SDL_LockSurface(intermediate_surface) == 0)) {
        const int bw = intermediate_surface->w * intermediate_surface->format->BytesPerPixel;

        /* Set the BMP file header values */
        bfSize = 0; /* We'll write this when we're done */
        bfReserved1 = 0;
        bfReserved2 = 0;
        bfOffBits = 0; /* We'll write this when we're done */

        /* Write the BMP file header values */
        fp_offset = SDL_RWtell(dst);
        SDL_ClearError();
        SDL_RWwrite(dst, magic, 2, 1);
        SDL_WriteLE32(dst, bfSize);
        SDL_WriteLE16(dst, bfReserved1);
        SDL_WriteLE16(dst, bfReserved2);
        SDL_WriteLE32(dst, bfOffBits);

        /* Set the BMP info values */
        biSize = 40;
        biWidth = intermediate_surface->w;
        biHeight = intermediate_surface->h;
        biPlanes = 1;
        biBitCount = intermediate_surface->format->BitsPerPixel;
        biCompression = BI_RGB;
        biSizeImage = intermediate_surface->h * intermediate_surface->pitch;
        biXPelsPerMeter = 0;
        biYPelsPerMeter = 0;
        if (intermediate_surface->format->palette) {
            biClrUsed = intermediate_surface->format->palette->ncolors;
        } else {
            biClrUsed = 0;
        }
        biClrImportant = 0;

        /* Set the BMP info values for the version 4 header */
        if (save32bit && !saveLegacyBMP) {
            biSize = 108;
            biCompression = BI_BITFIELDS;
            /* The BMP format is always little endian, these masks stay the same */
            bV4RedMask = 0x00ff0000;
            bV4GreenMask = 0x0000ff00;
            bV4BlueMask = 0x000000ff;
            bV4AlphaMask = 0xff000000;
            bV4CSType = LCS_WINDOWS_COLOR_SPACE;
            bV4GammaRed = 0;
            bV4GammaGreen = 0;
            bV4GammaBlue = 0;
        }

        /* Write the BMP info values */
        SDL_WriteLE32(dst, biSize);
        SDL_WriteLE32(dst, biWidth);
        SDL_WriteLE32(dst, biHeight);
        SDL_WriteLE16(dst, biPlanes);
        SDL_WriteLE16(dst, biBitCount);
        SDL_WriteLE32(dst, biCompression);
        SDL_WriteLE32(dst, biSizeImage);
        SDL_WriteLE32(dst, biXPelsPerMeter);
        SDL_WriteLE32(dst, biYPelsPerMeter);
        SDL_WriteLE32(dst, biClrUsed);
        SDL_WriteLE32(dst, biClrImportant);

        /* Write the BMP info values for the version 4 header */
        if (save32bit && !saveLegacyBMP) {
            SDL_WriteLE32(dst, bV4RedMask);
            SDL_WriteLE32(dst, bV4GreenMask);
            SDL_WriteLE32(dst, bV4BlueMask);
            SDL_WriteLE32(dst, bV4AlphaMask);
            SDL_WriteLE32(dst, bV4CSType);
            for (i = 0; i < 3 * 3; i++) {
                SDL_WriteLE32(dst, bV4Endpoints[i]);
            }
            SDL_WriteLE32(dst, bV4GammaRed);
            SDL_WriteLE32(dst, bV4GammaGreen);
            SDL_WriteLE32(dst, bV4GammaBlue);
        }

        /* Write the palette (in BGR color order) */
        if (intermediate_surface->format->palette) {
            SDL_Color *colors;
            int ncolors;

            colors = intermediate_surface->format->palette->colors;
            ncolors = intermediate_surface->format->palette->ncolors;
            for (i = 0; i < ncolors; ++i) {
                SDL_RWwrite(dst, &colors[i].b, 1, 1);
                SDL_RWwrite(dst, &colors[i].g, 1, 1);
                SDL_RWwrite(dst, &colors[i].r, 1, 1);
                SDL_RWwrite(dst, &colors[i].a, 1, 1);
            }
        }

        /* Write the bitmap offset */
        bfOffBits = (Uint32)(SDL_RWtell(dst) - fp_offset);
        if (SDL_RWseek(dst, fp_offset + 10, RW_SEEK_SET) < 0) {
            SDL_Error(SDL_EFSEEK);
        }
        SDL_WriteLE32(dst, bfOffBits);
        if (SDL_RWseek(dst, fp_offset + bfOffBits, RW_SEEK_SET) < 0) {
            SDL_Error(SDL_EFSEEK);
        }

        /* Write the bitmap image upside down */
        bits = (Uint8 *)intermediate_surface->pixels + (intermediate_surface->h * intermediate_surface->pitch);
        pad = ((bw % 4) ? (4 - (bw % 4)) : 0);
        while (bits > (Uint8 *)intermediate_surface->pixels) {
            bits -= intermediate_surface->pitch;
            if (SDL_RWwrite(dst, bits, 1, bw) != bw) {
                SDL_Error(SDL_EFWRITE);
                break;
            }
            if (pad) {
                const Uint8 padbyte = 0;
                for (i = 0; i < pad; ++i) {
                    SDL_RWwrite(dst, &padbyte, 1, 1);
                }
            }
        }

        /* Write the BMP file size */
        bfSize = (Uint32)(SDL_RWtell(dst) - fp_offset);
        if (SDL_RWseek(dst, fp_offset + 2, RW_SEEK_SET) < 0) {
            SDL_Error(SDL_EFSEEK);
        }
        SDL_WriteLE32(dst, bfSize);
        if (SDL_RWseek(dst, fp_offset + bfSize, RW_SEEK_SET) < 0) {
            SDL_Error(SDL_EFSEEK);
        }

        /* Close it up.. */
        SDL_UnlockSurface(intermediate_surface);
        if (intermediate_surface != surface) {
            SDL_FreeSurface(intermediate_surface);
        }
    }

    if (freedst && dst) {
        SDL_RWclose(dst);
    }
    return (SDL_strcmp(SDL_GetError(), "") == 0) ? 0 : -1;
}

/* vi: set ts=4 sw=4 expandtab: */
