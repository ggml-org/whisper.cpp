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
#include "../../SDL_internal.h"

#ifdef SDL_VIDEO_DRIVER_OS2

#include "SDL_os2util.h"

HPOINTER utilCreatePointer(SDL_Surface *surface, ULONG ulHotX, ULONG ulHotY)
{
    HBITMAP             hbm;
    BITMAPINFOHEADER2   bmih;
    BITMAPINFO          bmi;
    HPS                 hps;
    PULONG              pulBitmap;
    PULONG              pulDst, pulSrc, pulDstMask;
    ULONG               ulY, ulX;
    HPOINTER            hptr = NULLHANDLE;

    if (surface->format->format != SDL_PIXELFORMAT_ARGB8888) {
        debug_os2("Image format should be SDL_PIXELFORMAT_ARGB8888");
        return NULLHANDLE;
    }

    pulBitmap = (PULONG) SDL_malloc(surface->h * surface->w * 2 * sizeof(ULONG));
    if (!pulBitmap) {
        SDL_OutOfMemory();
        return NULLHANDLE;
    }

    /* pulDst - last line of surface (image) part of the result bitmap */
    pulDst = &pulBitmap[ (surface->h - 1) * surface->w ];
    /* pulDstMask - last line of mask part of the result bitmap */
    pulDstMask = &pulBitmap[ (2 * surface->h - 1) * surface->w ];
    /* pulSrc - first line of source image */
    pulSrc = (PULONG)surface->pixels;

    for (ulY = 0; ulY < surface->h; ulY++) {
        for (ulX = 0; ulX < surface->w; ulX++) {
            if ((pulSrc[ulX] & 0xFF000000) == 0) {
                pulDst[ulX] = 0;
                pulDstMask[ulX] = 0xFFFFFFFF;
            } else {
                pulDst[ulX] = pulSrc[ulX] & 0xFFFFFF;
                pulDstMask[ulX] = 0;
            }
        }

        /* Set image and mask pointers on one line up */
        pulDst -= surface->w;
        pulDstMask -= surface->w;
        /* Set source image pointer to the next line */
        pulSrc = (PULONG) (((PCHAR)pulSrc) + surface->pitch);
    }

    /* Create system bitmap object. */
    SDL_zero(bmih);
    SDL_zero(bmi);

    bmih.cbFix          = sizeof(BITMAPINFOHEADER2);
    bmih.cx             = surface->w;
    bmih.cy             = 2 * surface->h;
    bmih.cPlanes        = 1;
    bmih.cBitCount      = 32;
    bmih.ulCompression  = BCA_UNCOMP;
    bmih.cbImage        = bmih.cx * bmih.cy * 4;

    bmi.cbFix           = sizeof(BITMAPINFOHEADER);
    bmi.cx              = bmih.cx;
    bmi.cy              = bmih.cy;
    bmi.cPlanes         = 1;
    bmi.cBitCount       = 32;

    hps = WinGetPS(HWND_DESKTOP);
    hbm = GpiCreateBitmap(hps, (PBITMAPINFOHEADER2)&bmih, CBM_INIT,
                          (PBYTE)pulBitmap, (PBITMAPINFO2)&bmi);
    if (hbm == GPI_ERROR) {
        debug_os2("GpiCreateBitmap() failed");
    } else {
        /* Create a system pointer object. */
        hptr = WinCreatePointer(HWND_DESKTOP, hbm, TRUE, ulHotX, ulHotY);
        if (hptr == NULLHANDLE) {
            debug_os2("WinCreatePointer() failed");
        }
    }
    GpiDeleteBitmap(hbm);

    WinReleasePS(hps);
    SDL_free(pulBitmap);

    return hptr;
}

#endif /* SDL_VIDEO_DRIVER_OS2 */

/* vi: set ts=4 sw=4 expandtab: */
