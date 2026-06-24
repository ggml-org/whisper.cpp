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

#ifdef SDL_VIDEO_DRIVER_OFFSCREEN

#include "../SDL_sysvideo.h"
#include "../SDL_egl_c.h"

#include "SDL_offscreenwindow.h"

int OFFSCREEN_CreateWindow(_THIS, SDL_Window *window)
{
    OFFSCREEN_Window *offscreen_window = SDL_calloc(1, sizeof(OFFSCREEN_Window));

    if (!offscreen_window) {
        return SDL_OutOfMemory();
    }

    window->driverdata = offscreen_window;

    if (window->x == SDL_WINDOWPOS_UNDEFINED) {
        window->x = 0;
    }

    if (window->y == SDL_WINDOWPOS_UNDEFINED) {
        window->y = 0;
    }

    offscreen_window->sdl_window = window;

#ifdef SDL_VIDEO_OPENGL_EGL
    if (window->flags & SDL_WINDOW_OPENGL) {

        if (!_this->egl_data) {
            return SDL_SetError("Cannot create an OPENGL window invalid egl_data");
        }

        offscreen_window->egl_surface = SDL_EGL_CreateOffscreenSurface(_this, window->w, window->h);

        if (offscreen_window->egl_surface == EGL_NO_SURFACE) {
            return SDL_SetError("Failed to created an offscreen surface (EGL display: %p)",
                                _this->egl_data->egl_display);
        }
    } else {
        offscreen_window->egl_surface = EGL_NO_SURFACE;
    }
#endif /* SDL_VIDEO_OPENGL_EGL */

    return 0;
}

void OFFSCREEN_DestroyWindow(_THIS, SDL_Window *window)
{
    OFFSCREEN_Window *offscreen_window = window->driverdata;

    if (offscreen_window) {
#ifdef SDL_VIDEO_OPENGL_EGL
        SDL_EGL_DestroySurface(_this, offscreen_window->egl_surface);
#endif
        SDL_free(offscreen_window);
    }

    window->driverdata = NULL;
}

#endif /* SDL_VIDEO_DRIVER_OFFSCREEN */

/* vi: set ts=4 sw=4 expandtab: */
