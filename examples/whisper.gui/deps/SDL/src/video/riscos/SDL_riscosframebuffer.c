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

#ifdef SDL_VIDEO_DRIVER_RISCOS

#include "../SDL_sysvideo.h"
#include "SDL_riscosframebuffer_c.h"
#include "SDL_riscosvideo.h"
#include "SDL_riscoswindow.h"

#include <kernel.h>
#include <swis.h>

int RISCOS_CreateWindowFramebuffer(_THIS, SDL_Window *window, Uint32 *format, void **pixels, int *pitch)
{
    SDL_WindowData *driverdata = (SDL_WindowData *)window->driverdata;
    const char *sprite_name = "display";
    unsigned int sprite_mode;
    _kernel_oserror *error;
    _kernel_swi_regs regs;
    SDL_DisplayMode mode;
    int size;
    int w, h;

    SDL_GetWindowSizeInPixels(window, &w, &h);

    /* Free the old framebuffer surface */
    RISCOS_DestroyWindowFramebuffer(_this, window);

    /* Create a new one */
    SDL_GetCurrentDisplayMode(SDL_GetWindowDisplayIndex(window), &mode);
    if ((SDL_ISPIXELFORMAT_PACKED(mode.format) || SDL_ISPIXELFORMAT_ARRAY(mode.format))) {
        *format = mode.format;
        sprite_mode = (unsigned int)mode.driverdata;
    } else {
        *format = SDL_PIXELFORMAT_BGR888;
        sprite_mode = (1 | (90 << 1) | (90 << 14) | (6 << 27));
    }

    /* Calculate pitch */
    *pitch = (((w * SDL_BYTESPERPIXEL(*format)) + 3) & ~3);

    /* Allocate the sprite area */
    size = sizeof(sprite_area) + sizeof(sprite_header) + ((*pitch) * h);
    driverdata->fb_area = SDL_malloc(size);
    if (!driverdata->fb_area) {
        return SDL_OutOfMemory();
    }

    driverdata->fb_area->size = size;
    driverdata->fb_area->count = 0;
    driverdata->fb_area->start = 16;
    driverdata->fb_area->end = 16;

    /* Create the actual image */
    regs.r[0] = 256 + 15;
    regs.r[1] = (int)driverdata->fb_area;
    regs.r[2] = (int)sprite_name;
    regs.r[3] = 0;
    regs.r[4] = w;
    regs.r[5] = h;
    regs.r[6] = sprite_mode;
    error = _kernel_swi(OS_SpriteOp, &regs, &regs);
    if (error) {
        SDL_free(driverdata->fb_area);
        return SDL_SetError("Unable to create sprite: %s (%i)", error->errmess, error->errnum);
    }

    driverdata->fb_sprite = (sprite_header *)(((Uint8 *)driverdata->fb_area) + driverdata->fb_area->start);
    *pixels = ((Uint8 *)driverdata->fb_sprite) + driverdata->fb_sprite->image_offset;

    return 0;
}

int RISCOS_UpdateWindowFramebuffer(_THIS, SDL_Window *window, const SDL_Rect *rects, int numrects)
{
    SDL_WindowData *driverdata = (SDL_WindowData *)window->driverdata;
    _kernel_swi_regs regs;
    _kernel_oserror *error;

    regs.r[0] = 512 + 52;
    regs.r[1] = (int)driverdata->fb_area;
    regs.r[2] = (int)driverdata->fb_sprite;
    regs.r[3] = 0; /* window->x << 1; */
    regs.r[4] = 0; /* window->y << 1; */
    regs.r[5] = 0x50;
    regs.r[6] = 0;
    regs.r[7] = 0;
    error = _kernel_swi(OS_SpriteOp, &regs, &regs);
    if (error) {
        return SDL_SetError("OS_SpriteOp 52 failed: %s (%i)", error->errmess, error->errnum);
    }

    return 0;
}

void RISCOS_DestroyWindowFramebuffer(_THIS, SDL_Window *window)
{
    SDL_WindowData *driverdata = (SDL_WindowData *)window->driverdata;

    if (driverdata->fb_area) {
        SDL_free(driverdata->fb_area);
        driverdata->fb_area = NULL;
    }
    driverdata->fb_sprite = NULL;
}

#endif /* SDL_VIDEO_DRIVER_RISCOS */

/* vi: set ts=4 sw=4 expandtab: */
