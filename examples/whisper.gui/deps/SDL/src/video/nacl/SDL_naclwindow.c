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

#ifdef SDL_VIDEO_DRIVER_NACL

#include "../SDL_sysvideo.h"

#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_keyboard_c.h"
#include "SDL_naclvideo.h"
#include "SDL_naclwindow.h"

int NACL_CreateWindow(_THIS, SDL_Window * window)
{
    SDL_VideoData *driverdata = (SDL_VideoData *) _this->driverdata;

    if (driverdata->window) {
        return SDL_SetError("NaCl only supports one window");
    }
    driverdata->window = window;

    /* Adjust the window data to match the screen */
    window->x = 0;
    window->y = 0;
    window->w = driverdata->w;
    window->h = driverdata->h;

    window->flags &= ~SDL_WINDOW_RESIZABLE;     /* window is NEVER resizeable */
    window->flags |= SDL_WINDOW_FULLSCREEN;     /* window is always fullscreen */
    window->flags &= ~SDL_WINDOW_HIDDEN;
    window->flags |= SDL_WINDOW_SHOWN;          /* only one window on NaCl */
    window->flags |= SDL_WINDOW_INPUT_FOCUS;    /* always has input focus */
    window->flags |= SDL_WINDOW_OPENGL;

    SDL_SetMouseFocus(window);
    SDL_SetKeyboardFocus(window);

    return 0;
}

void NACL_SetWindowTitle(_THIS, SDL_Window * window)
{
    /* TODO */
}

void NACL_DestroyWindow(_THIS, SDL_Window * window)
{
    SDL_VideoData *driverdata = (SDL_VideoData *) _this->driverdata;
    if (window == driverdata->window) {
        driverdata->window = NULL;
    }
}

#endif /* SDL_VIDEO_DRIVER_NACL */

/* vi: set ts=4 sw=4 expandtab: */
