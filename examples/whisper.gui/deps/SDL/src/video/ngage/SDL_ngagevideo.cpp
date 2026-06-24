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

#include <stdlib.h>
#ifdef NULL
#undef NULL
#endif
#include "../../SDL_internal.h"

#ifdef SDL_VIDEO_DRIVER_NGAGE

#ifdef __cplusplus
extern "C" {
#endif

#include "SDL_video.h"
#include "../SDL_sysvideo.h"
#include "../SDL_pixels_c.h"
#include "../../events/SDL_events_c.h"

#ifdef __cplusplus
}
#endif

#include "SDL_ngagevideo.h"
#include "SDL_ngagewindow.h"
#include "SDL_ngageevents_c.h"
#include "SDL_ngageframebuffer_c.h"

#define NGAGEVID_DRIVER_NAME "ngage"

/* Initialization/Query functions */
static int NGAGE_VideoInit(_THIS);
static int NGAGE_SetDisplayMode(_THIS, SDL_VideoDisplay *display, SDL_DisplayMode *mode);
static void NGAGE_VideoQuit(_THIS);

/* NGAGE driver bootstrap functions */

static void NGAGE_DeleteDevice(SDL_VideoDevice *device)
{
    SDL_VideoData *phdata = (SDL_VideoData *)device->driverdata;

    if (phdata) {
        /* Free Epoc resources */

        /* Disable events for me */
        if (phdata->NGAGE_WsEventStatus != KRequestPending) {
            phdata->NGAGE_WsSession.EventReadyCancel();
        }
        if (phdata->NGAGE_RedrawEventStatus != KRequestPending) {
            phdata->NGAGE_WsSession.RedrawReadyCancel();
        }

        free(phdata->NGAGE_DrawDevice);

        if (phdata->NGAGE_WsWindow.WsHandle()) {
            phdata->NGAGE_WsWindow.Close();
        }

        if (phdata->NGAGE_WsWindowGroup.WsHandle()) {
            phdata->NGAGE_WsWindowGroup.Close();
        }

        delete phdata->NGAGE_WindowGc;
        phdata->NGAGE_WindowGc = NULL;

        delete phdata->NGAGE_WsScreen;
        phdata->NGAGE_WsScreen = NULL;

        if (phdata->NGAGE_WsSession.WsHandle()) {
            phdata->NGAGE_WsSession.Close();
        }

        SDL_free(phdata);
        phdata = NULL;
    }

    if (device) {
        SDL_free(device);
        device = NULL;
    }
}

static SDL_VideoDevice *NGAGE_CreateDevice(void)
{
    SDL_VideoDevice *device;
    SDL_VideoData *phdata;

    /* Initialize all variables that we clean on shutdown */
    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        SDL_OutOfMemory();
        return 0;
    }

    /* Initialize internal N-Gage specific data */
    phdata = (SDL_VideoData *)SDL_calloc(1, sizeof(SDL_VideoData));
    if (!phdata) {
        SDL_OutOfMemory();
        SDL_free(device);
        return 0;
    }

    /* General video */
    device->VideoInit = NGAGE_VideoInit;
    device->VideoQuit = NGAGE_VideoQuit;
    device->SetDisplayMode = NGAGE_SetDisplayMode;
    device->PumpEvents = NGAGE_PumpEvents;
    device->CreateWindowFramebuffer = SDL_NGAGE_CreateWindowFramebuffer;
    device->UpdateWindowFramebuffer = SDL_NGAGE_UpdateWindowFramebuffer;
    device->DestroyWindowFramebuffer = SDL_NGAGE_DestroyWindowFramebuffer;
    device->free = NGAGE_DeleteDevice;

    /* "Window" */
    device->CreateSDLWindow = NGAGE_CreateWindow;
    device->DestroyWindow = NGAGE_DestroyWindow;

    /* N-Gage specific data */
    device->driverdata = phdata;

    return device;
}

VideoBootStrap NGAGE_bootstrap = {
    NGAGEVID_DRIVER_NAME, "SDL ngage video driver",
    NGAGE_CreateDevice,
    NULL /* no ShowMessageBox implementation */
};

int NGAGE_VideoInit(_THIS)
{
    SDL_DisplayMode mode;

    /* Use 12-bpp desktop mode */
    mode.format = SDL_PIXELFORMAT_RGB444;
    mode.w = 176;
    mode.h = 208;
    mode.refresh_rate = 0;
    mode.driverdata = NULL;
    if (SDL_AddBasicVideoDisplay(&mode) < 0) {
        return -1;
    }

    SDL_zero(mode);
    SDL_AddDisplayMode(&_this->displays[0], &mode);

    /* We're done! */
    return 0;
}

static int NGAGE_SetDisplayMode(_THIS, SDL_VideoDisplay *display, SDL_DisplayMode *mode)
{
    return 0;
}

void NGAGE_VideoQuit(_THIS)
{
}

#endif /* SDL_VIDEO_DRIVER_NGAGE */

/* vi: set ts=4 sw=4 expandtab: */
