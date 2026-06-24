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

#ifndef SDL_emscriptenvideo_h_
#define SDL_emscriptenvideo_h_

#include "../SDL_sysvideo.h"
#include "../../events/SDL_touch_c.h"
#include <emscripten/emscripten.h>
#include <emscripten/html5.h>

#ifdef SDL_VIDEO_OPENGL_EGL
#include <EGL/egl.h>
#endif

typedef struct SDL_WindowData
{
#ifdef SDL_VIDEO_OPENGL_EGL
    EGLSurface egl_surface;
#endif
    SDL_Window *window;
    SDL_Surface *surface;

    char *canvas_id;

    float pixel_ratio;

    SDL_bool external_size;

    int requested_fullscreen_mode;
    SDL_bool fullscreen_resize;

    SDL_bool has_pointer_lock;
} SDL_WindowData;

#endif /* SDL_emscriptenvideo_h_ */

/* vi: set ts=4 sw=4 expandtab: */
