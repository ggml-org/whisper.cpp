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

#include "ppapi/c/pp_errors.h"
#include "ppapi/c/pp_instance.h"
#include "ppapi_simple/ps.h"
#include "ppapi_simple/ps_interface.h"
#include "ppapi_simple/ps_event.h"
#include "nacl_io/nacl_io.h"

#include "SDL_naclvideo.h"
#include "SDL_naclwindow.h"
#include "SDL_naclevents_c.h"
#include "SDL_naclopengles.h"
#include "SDL_video.h"
#include "../SDL_sysvideo.h"
#include "../../events/SDL_events_c.h"

#define NACLVID_DRIVER_NAME "nacl"

/* Static init required because NACL_SetScreenResolution
 * may appear even before SDL starts and we want to remember
 * the window width and height
 */
static SDL_VideoData nacl = {0};

void NACL_SetScreenResolution(int width, int height, Uint32 format)
{
    PP_Resource context;

    nacl.w = width;
    nacl.h = height;
    nacl.format = format;

    if (nacl.window) {
        nacl.window->w = width;
        nacl.window->h = height;
        SDL_SendWindowEvent(nacl.window, SDL_WINDOWEVENT_RESIZED, width, height);
    }

    /* FIXME: Check threading issues...otherwise use a hardcoded _this->context across all threads */
    context = (PP_Resource) SDL_GL_GetCurrentContext();
    if (context) {
        PSInterfaceGraphics3D()->ResizeBuffers(context, width, height);
    }

}



/* Initialization/Query functions */
static int NACL_VideoInit(_THIS);
static void NACL_VideoQuit(_THIS);

static int NACL_Available(void) {
    return PSGetInstanceId() != 0;
}

static void NACL_DeleteDevice(SDL_VideoDevice *device) {
    SDL_VideoData *driverdata = (SDL_VideoData*) device->driverdata;
    driverdata->ppb_core->ReleaseResource((PP_Resource) driverdata->ppb_message_loop);
    /* device->driverdata is not freed because it points to static memory */
    SDL_free(device);
}

static int NACL_SetDisplayMode(_THIS, SDL_VideoDisplay * display, SDL_DisplayMode * mode)
{
    return 0;
}

static SDL_VideoDevice *NACL_CreateDevice(void) {
    SDL_VideoDevice *device;

    if (!NACL_Available()) {
        return NULL;
    }

    /* Initialize all variables that we clean on shutdown */
    device = (SDL_VideoDevice *) SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        SDL_OutOfMemory();
        return NULL;
    }
    device->driverdata = &nacl;

    /* Set the function pointers */
    device->VideoInit = NACL_VideoInit;
    device->VideoQuit = NACL_VideoQuit;
    device->PumpEvents = NACL_PumpEvents;

    device->CreateSDLWindow = NACL_CreateWindow;
    device->SetWindowTitle = NACL_SetWindowTitle;
    device->DestroyWindow = NACL_DestroyWindow;

    device->SetDisplayMode = NACL_SetDisplayMode;

    device->free = NACL_DeleteDevice;

    /* GL pointers */
    device->GL_LoadLibrary = NACL_GLES_LoadLibrary;
    device->GL_GetProcAddress = NACL_GLES_GetProcAddress;
    device->GL_UnloadLibrary = NACL_GLES_UnloadLibrary;
    device->GL_CreateContext = NACL_GLES_CreateContext;
    device->GL_MakeCurrent = NACL_GLES_MakeCurrent;
    device->GL_SetSwapInterval = NACL_GLES_SetSwapInterval;
    device->GL_GetSwapInterval = NACL_GLES_GetSwapInterval;
    device->GL_SwapWindow = NACL_GLES_SwapWindow;
    device->GL_DeleteContext = NACL_GLES_DeleteContext;


    return device;
}

VideoBootStrap NACL_bootstrap = {
    NACLVID_DRIVER_NAME, "SDL Native Client Video Driver",
    NACL_CreateDevice,
    NULL /* no ShowMessageBox implementation */
};

int NACL_VideoInit(_THIS) {
    SDL_VideoData *driverdata = (SDL_VideoData *) _this->driverdata;
    SDL_DisplayMode mode;

    SDL_zero(mode);
    mode.format = driverdata->format;
    mode.w = driverdata->w;
    mode.h = driverdata->h;
    mode.refresh_rate = 0;
    mode.driverdata = NULL;
    if (SDL_AddBasicVideoDisplay(&mode) < 0) {
        return -1;
    }

    SDL_AddDisplayMode(&_this->displays[0], &mode);

    PSInterfaceInit();
    driverdata->instance = PSGetInstanceId();
    driverdata->ppb_graphics = PSInterfaceGraphics3D();
    driverdata->ppb_message_loop = PSInterfaceMessageLoop();
    driverdata->ppb_core = PSInterfaceCore();
    driverdata->ppb_fullscreen = PSInterfaceFullscreen();
    driverdata->ppb_instance = PSInterfaceInstance();
    driverdata->ppb_image_data = PSInterfaceImageData();
    driverdata->ppb_view = PSInterfaceView();
    driverdata->ppb_var = PSInterfaceVar();
    driverdata->ppb_input_event = (PPB_InputEvent*) PSGetInterface(PPB_INPUT_EVENT_INTERFACE);
    driverdata->ppb_keyboard_input_event = (PPB_KeyboardInputEvent*) PSGetInterface(PPB_KEYBOARD_INPUT_EVENT_INTERFACE);
    driverdata->ppb_mouse_input_event = (PPB_MouseInputEvent*) PSGetInterface(PPB_MOUSE_INPUT_EVENT_INTERFACE);
    driverdata->ppb_wheel_input_event = (PPB_WheelInputEvent*) PSGetInterface(PPB_WHEEL_INPUT_EVENT_INTERFACE);
    driverdata->ppb_touch_input_event = (PPB_TouchInputEvent*) PSGetInterface(PPB_TOUCH_INPUT_EVENT_INTERFACE);


    driverdata->message_loop = driverdata->ppb_message_loop->Create(driverdata->instance);

    PSEventSetFilter(PSE_ALL);

    /* We're done! */
    return 0;
}

void NACL_VideoQuit(_THIS) {
}

#endif /* SDL_VIDEO_DRIVER_NACL */
/* vi: set ts=4 sw=4 expandtab: */
