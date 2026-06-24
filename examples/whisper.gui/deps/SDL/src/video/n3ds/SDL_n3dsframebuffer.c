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

#ifdef SDL_VIDEO_DRIVER_N3DS

#include "../SDL_sysvideo.h"
#include "SDL_n3dsframebuffer_c.h"
#include "SDL_n3dsvideo.h"

#define N3DS_SURFACE "_SDL_N3DSSurface"

typedef struct
{
    int width, height;
} Dimensions;

SDL_FORCE_INLINE void FreePreviousWindowFramebuffer(SDL_Window *window);
SDL_FORCE_INLINE SDL_Surface *CreateNewWindowFramebuffer(SDL_Window *window);
SDL_FORCE_INLINE void CopyFramebuffertoN3DS(u32 *dest, const Dimensions dest_dim, const u32 *source, const Dimensions source_dim);
SDL_FORCE_INLINE int GetDestOffset(int x, int y, int dest_width);
SDL_FORCE_INLINE int GetSourceOffset(int x, int y, int source_width);
SDL_FORCE_INLINE void FlushN3DSBuffer(const void *buffer, u32 bufsize, gfxScreen_t screen);

int SDL_N3DS_CreateWindowFramebuffer(_THIS, SDL_Window *window, Uint32 *format, void **pixels, int *pitch)
{
    SDL_Surface *framebuffer;

    FreePreviousWindowFramebuffer(window);
    framebuffer = CreateNewWindowFramebuffer(window);

    if (!framebuffer) {
        return SDL_OutOfMemory();
    }

    SDL_SetWindowData(window, N3DS_SURFACE, framebuffer);
    *format = FRAMEBUFFER_FORMAT;
    *pixels = framebuffer->pixels;
    *pitch = framebuffer->pitch;
    return 0;
}

SDL_FORCE_INLINE void
FreePreviousWindowFramebuffer(SDL_Window *window)
{
    SDL_Surface *surface = (SDL_Surface *)SDL_GetWindowData(window, N3DS_SURFACE);
    SDL_FreeSurface(surface);
}

SDL_FORCE_INLINE SDL_Surface *
CreateNewWindowFramebuffer(SDL_Window *window)
{
    int w, h, bpp;
    Uint32 Rmask, Gmask, Bmask, Amask;
    SDL_PixelFormatEnumToMasks(FRAMEBUFFER_FORMAT, &bpp, &Rmask, &Gmask, &Bmask, &Amask);
    SDL_GetWindowSizeInPixels(window, &w, &h);
    return SDL_CreateRGBSurface(0, w, h, bpp, Rmask, Gmask, Bmask, Amask);
}

int SDL_N3DS_UpdateWindowFramebuffer(_THIS, SDL_Window *window, const SDL_Rect *rects, int numrects)
{
    SDL_WindowData *drv_data = (SDL_WindowData *)window->driverdata;
    SDL_Surface *surface;
    u16 width, height;
    u32 *framebuffer;
    u32 bufsize;

    surface = (SDL_Surface *)SDL_GetWindowData(window, N3DS_SURFACE);
    if (!surface) {
        return SDL_SetError("%s: Unable to get the window surface.", __func__);
    }

    /* Get the N3DS internal framebuffer and its size */
    framebuffer = (u32 *)gfxGetFramebuffer(drv_data->screen, GFX_LEFT, &width, &height);
    bufsize = width * height * 4;

    CopyFramebuffertoN3DS(framebuffer, (Dimensions){ width, height },
                          surface->pixels, (Dimensions){ surface->w, surface->h });
    FlushN3DSBuffer(framebuffer, bufsize, drv_data->screen);

    return 0;
}

SDL_FORCE_INLINE void
CopyFramebuffertoN3DS(u32 *dest, const Dimensions dest_dim, const u32 *source, const Dimensions source_dim)
{
    int rows = SDL_min(dest_dim.width, source_dim.height);
    int cols = SDL_min(dest_dim.height, source_dim.width);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            SDL_memcpy(
                dest + GetDestOffset(x, y, dest_dim.width),
                source + GetSourceOffset(x, y, source_dim.width),
                4);
        }
    }
}

SDL_FORCE_INLINE int
GetDestOffset(int x, int y, int dest_width)
{
    return dest_width - y - 1 + dest_width * x;
}

SDL_FORCE_INLINE int
GetSourceOffset(int x, int y, int source_width)
{
    return x + y * source_width;
}

SDL_FORCE_INLINE void
FlushN3DSBuffer(const void *buffer, u32 bufsize, gfxScreen_t screen)
{
    GSPGPU_FlushDataCache(buffer, bufsize);
    gfxScreenSwapBuffers(screen, false);
}

void SDL_N3DS_DestroyWindowFramebuffer(_THIS, SDL_Window *window)
{
    SDL_Surface *surface;
    surface = (SDL_Surface *)SDL_SetWindowData(window, N3DS_SURFACE, NULL);
    SDL_FreeSurface(surface);
}

#endif /* SDL_VIDEO_DRIVER_N3DS */

/* vi: set sts=4 ts=4 sw=4 expandtab: */
