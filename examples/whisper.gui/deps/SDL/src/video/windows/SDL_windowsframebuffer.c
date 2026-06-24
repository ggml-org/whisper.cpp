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

#if defined(SDL_VIDEO_DRIVER_WINDOWS) && !defined(__XBOXONE__) && !defined(__XBOXSERIES__)

#include "SDL_windowsvideo.h"

int WIN_CreateWindowFramebuffer(_THIS, SDL_Window *window, Uint32 *format, void **pixels, int *pitch)
{
    SDL_WindowData *data = (SDL_WindowData *)window->driverdata;
    SDL_bool isstack;
    size_t size;
    LPBITMAPINFO info;
    HBITMAP hbm;
    int w, h;

    SDL_GetWindowSizeInPixels(window, &w, &h);

    /* Free the old framebuffer surface */
    if (data->mdc) {
        DeleteDC(data->mdc);
    }
    if (data->hbm) {
        DeleteObject(data->hbm);
    }

    /* Find out the format of the screen */
    size = sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD);
    info = (LPBITMAPINFO)SDL_small_alloc(Uint8, size, &isstack);
    if (!info) {
        return SDL_OutOfMemory();
    }

    SDL_memset(info, 0, size);
    info->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);

    /* The second call to GetDIBits() fills in the bitfields */
    hbm = CreateCompatibleBitmap(data->hdc, 1, 1);
    GetDIBits(data->hdc, hbm, 0, 0, NULL, info, DIB_RGB_COLORS);
    GetDIBits(data->hdc, hbm, 0, 0, NULL, info, DIB_RGB_COLORS);
    DeleteObject(hbm);

    *format = SDL_PIXELFORMAT_UNKNOWN;
    if (info->bmiHeader.biCompression == BI_BITFIELDS) {
        int bpp;
        Uint32 *masks;

        bpp = info->bmiHeader.biPlanes * info->bmiHeader.biBitCount;
        masks = (Uint32 *)((Uint8 *)info + info->bmiHeader.biSize);
        *format = SDL_MasksToPixelFormatEnum(bpp, masks[0], masks[1], masks[2], 0);
    }
    if (*format == SDL_PIXELFORMAT_UNKNOWN) {
        /* We'll use RGB format for now */
        *format = SDL_PIXELFORMAT_RGB888;

        /* Create a new one */
        SDL_memset(info, 0, size);
        info->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        info->bmiHeader.biPlanes = 1;
        info->bmiHeader.biBitCount = 32;
        info->bmiHeader.biCompression = BI_RGB;
    }

    /* Fill in the size information */
    *pitch = (((w * SDL_BYTESPERPIXEL(*format)) + 3) & ~3);
    info->bmiHeader.biWidth = w;
    info->bmiHeader.biHeight = -h; /* negative for topdown bitmap */
    info->bmiHeader.biSizeImage = (DWORD)h * (*pitch);

    data->mdc = CreateCompatibleDC(data->hdc);
    data->hbm = CreateDIBSection(data->hdc, info, DIB_RGB_COLORS, pixels, NULL, 0);
    SDL_small_free(info, isstack);

    if (!data->hbm) {
        return WIN_SetError("Unable to create DIB");
    }
    SelectObject(data->mdc, data->hbm);

    return 0;
}

int WIN_UpdateWindowFramebuffer(_THIS, SDL_Window *window, const SDL_Rect *rects, int numrects)
{
    SDL_WindowData *data = (SDL_WindowData *)window->driverdata;
    int i;

    for (i = 0; i < numrects; ++i) {
        BitBlt(data->hdc, rects[i].x, rects[i].y, rects[i].w, rects[i].h,
               data->mdc, rects[i].x, rects[i].y, SRCCOPY);
    }
    return 0;
}

void WIN_DestroyWindowFramebuffer(_THIS, SDL_Window *window)
{
    SDL_WindowData *data = (SDL_WindowData *)window->driverdata;

    if (!data) {
        /* The window wasn't fully initialized */
        return;
    }

    if (data->mdc) {
        DeleteDC(data->mdc);
        data->mdc = NULL;
    }
    if (data->hbm) {
        DeleteObject(data->hbm);
        data->hbm = NULL;
    }
}

#endif /* SDL_VIDEO_DRIVER_WINDOWS */

/* vi: set ts=4 sw=4 expandtab: */
