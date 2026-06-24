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

#ifdef SDL_VIDEO_DRIVER_DUMMY

/* Dummy SDL video driver implementation; this is just enough to make an
 *  SDL-based application THINK it's got a working video driver, for
 *  applications that call SDL_Init(SDL_INIT_VIDEO) when they don't need it,
 *  and also for use as a collection of stubs when porting SDL to a new
 *  platform for which you haven't yet written a valid video driver.
 *
 * This is also a great way to determine bottlenecks: if you think that SDL
 *  is a performance problem for a given platform, enable this driver, and
 *  then see if your application runs faster without video overhead.
 *
 * Initial work by Ryan C. Gordon (icculus@icculus.org). A good portion
 *  of this was cut-and-pasted from Stephane Peter's work in the AAlib
 *  SDL video driver.  Renamed to "DUMMY" by Sam Lantinga.
 */

#include "SDL_video.h"
#include "SDL_mouse.h"
#include "../SDL_sysvideo.h"
#include "../SDL_pixels_c.h"
#include "../../events/SDL_events_c.h"

#include "SDL_nullvideo.h"
#include "SDL_nullevents_c.h"
#include "SDL_nullframebuffer_c.h"
#include "SDL_hints.h"

#define DUMMYVID_DRIVER_NAME       "dummy"
#define DUMMYVID_DRIVER_EVDEV_NAME "evdev"

/* Initialization/Query functions */
static int DUMMY_VideoInit(_THIS);
static void DUMMY_VideoQuit(_THIS);

#ifdef SDL_INPUT_LINUXEV
static int evdev = 0;
static void DUMMY_EVDEV_Poll(_THIS);
#endif

/* DUMMY driver bootstrap functions */

static int DUMMY_Available(void)
{
    const char *envr = SDL_GetHint(SDL_HINT_VIDEODRIVER);
    if (envr) {
        if (SDL_strcmp(envr, DUMMYVID_DRIVER_NAME) == 0) {
            return 1;
        }
#ifdef SDL_INPUT_LINUXEV
        if (SDL_strcmp(envr, DUMMYVID_DRIVER_EVDEV_NAME) == 0) {
            evdev = 1;
            return 1;
        }
#endif
    }
    return 0;
}

static void DUMMY_DeleteDevice(SDL_VideoDevice *device)
{
    SDL_free(device);
}

static SDL_VideoDevice *DUMMY_CreateDevice(void)
{
    SDL_VideoDevice *device;

    if (!DUMMY_Available()) {
        return 0;
    }

    /* Initialize all variables that we clean on shutdown */
    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        SDL_OutOfMemory();
        return 0;
    }
    device->is_dummy = SDL_TRUE;

    /* Set the function pointers */
    device->VideoInit = DUMMY_VideoInit;
    device->VideoQuit = DUMMY_VideoQuit;
    device->PumpEvents = DUMMY_PumpEvents;
#ifdef SDL_INPUT_LINUXEV
    if (evdev) {
        device->PumpEvents = DUMMY_EVDEV_Poll;
    }
#endif
    device->CreateWindowFramebuffer = SDL_DUMMY_CreateWindowFramebuffer;
    device->UpdateWindowFramebuffer = SDL_DUMMY_UpdateWindowFramebuffer;
    device->DestroyWindowFramebuffer = SDL_DUMMY_DestroyWindowFramebuffer;

    device->free = DUMMY_DeleteDevice;

    return device;
}

VideoBootStrap DUMMY_bootstrap = {
    DUMMYVID_DRIVER_NAME, "SDL dummy video driver",
    DUMMY_CreateDevice,
    NULL /* no ShowMessageBox implementation */
};

#ifdef SDL_INPUT_LINUXEV
VideoBootStrap DUMMY_evdev_bootstrap = {
    DUMMYVID_DRIVER_EVDEV_NAME, "SDL dummy video driver with evdev",
    DUMMY_CreateDevice,
    NULL /* no ShowMessageBox implementation */
};
void SDL_EVDEV_Init(void);
void SDL_EVDEV_Poll(void);
void SDL_EVDEV_Quit(void);
static void DUMMY_EVDEV_Poll(_THIS)
{
    (void)_this;
    SDL_EVDEV_Poll();
}
#endif

int DUMMY_VideoInit(_THIS)
{
    SDL_DisplayMode mode;

    /* Use a fake 32-bpp desktop mode */
    SDL_zero(mode);
    mode.format = SDL_PIXELFORMAT_RGB888;
    mode.w = 1024;
    mode.h = 768;
    mode.refresh_rate = 60;
    mode.driverdata = NULL;
    if (SDL_AddBasicVideoDisplay(&mode) < 0) {
        return -1;
    }

    SDL_AddDisplayMode(&_this->displays[0], &mode);

#ifdef SDL_INPUT_LINUXEV
    SDL_EVDEV_Init();
#endif

    /* We're done! */
    return 0;
}

void DUMMY_VideoQuit(_THIS)
{
#ifdef SDL_INPUT_LINUXEV
    SDL_EVDEV_Quit();
#endif
}

#endif /* SDL_VIDEO_DRIVER_DUMMY */

/* vi: set ts=4 sw=4 expandtab: */
