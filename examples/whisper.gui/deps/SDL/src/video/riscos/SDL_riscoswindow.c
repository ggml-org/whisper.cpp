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

#include "SDL_version.h"
#include "SDL_syswm.h"
#include "../SDL_sysvideo.h"
#include "../../events/SDL_mouse_c.h"


#include "SDL_riscosvideo.h"
#include "SDL_riscoswindow.h"

int RISCOS_CreateWindow(_THIS, SDL_Window *window)
{
    SDL_WindowData *driverdata;

    driverdata = (SDL_WindowData *)SDL_calloc(1, sizeof(*driverdata));
    if (!driverdata) {
        return SDL_OutOfMemory();
    }
    driverdata->window = window;

    window->flags |= SDL_WINDOW_FULLSCREEN;

    SDL_SetMouseFocus(window);

    /* All done! */
    window->driverdata = driverdata;
    return 0;
}

void RISCOS_DestroyWindow(_THIS, SDL_Window *window)
{
    SDL_WindowData *driverdata = (SDL_WindowData *)window->driverdata;

    if (!driverdata) {
        return;
    }

    SDL_free(driverdata);
    window->driverdata = NULL;
}

SDL_bool RISCOS_GetWindowWMInfo(_THIS, SDL_Window * window, struct SDL_SysWMinfo *info)
{
    if (info->version.major == SDL_MAJOR_VERSION) {
        info->subsystem = SDL_SYSWM_RISCOS;
        return SDL_TRUE;
    } else {
        SDL_SetError("Application not compiled with SDL %d",
                     SDL_MAJOR_VERSION);
        return SDL_FALSE;
    }
}

#endif /* SDL_VIDEO_DRIVER_RISCOS */

/* vi: set ts=4 sw=4 expandtab: */
