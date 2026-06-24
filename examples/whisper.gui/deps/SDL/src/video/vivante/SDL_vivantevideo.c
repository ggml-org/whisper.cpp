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

#ifdef SDL_VIDEO_DRIVER_VIVANTE

/* SDL internals */
#include "../SDL_sysvideo.h"
#include "SDL_version.h"
#include "SDL_syswm.h"
#include "SDL_loadso.h"
#include "SDL_events.h"
#include "../../events/SDL_events_c.h"

#ifdef SDL_INPUT_LINUXEV
#include "../../core/linux/SDL_evdev.h"
#endif

#include "SDL_vivantevideo.h"
#include "SDL_vivanteplatform.h"
#include "SDL_vivanteopengles.h"
#include "SDL_vivantevulkan.h"

static void VIVANTE_Destroy(SDL_VideoDevice *device)
{
    if (device->driverdata) {
        SDL_free(device->driverdata);
        device->driverdata = NULL;
    }
}

static SDL_VideoDevice *VIVANTE_Create()
{
    SDL_VideoDevice *device;
    SDL_VideoData *data;

    /* Initialize SDL_VideoDevice structure */
    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        SDL_OutOfMemory();
        return NULL;
    }

    /* Initialize internal data */
    data = (SDL_VideoData *)SDL_calloc(1, sizeof(SDL_VideoData));
    if (!data) {
        SDL_OutOfMemory();
        SDL_free(device);
        return NULL;
    }

    device->driverdata = data;

    /* Setup amount of available displays */
    device->num_displays = 0;

    /* Set device free function */
    device->free = VIVANTE_Destroy;

    /* Setup all functions which we can handle */
    device->VideoInit = VIVANTE_VideoInit;
    device->VideoQuit = VIVANTE_VideoQuit;
    device->GetDisplayModes = VIVANTE_GetDisplayModes;
    device->SetDisplayMode = VIVANTE_SetDisplayMode;
    device->CreateSDLWindow = VIVANTE_CreateWindow;
    device->SetWindowTitle = VIVANTE_SetWindowTitle;
    device->SetWindowPosition = VIVANTE_SetWindowPosition;
    device->SetWindowSize = VIVANTE_SetWindowSize;
    device->ShowWindow = VIVANTE_ShowWindow;
    device->HideWindow = VIVANTE_HideWindow;
    device->DestroyWindow = VIVANTE_DestroyWindow;
    device->GetWindowWMInfo = VIVANTE_GetWindowWMInfo;

#ifdef SDL_VIDEO_OPENGL_EGL
    device->GL_LoadLibrary = VIVANTE_GLES_LoadLibrary;
    device->GL_GetProcAddress = VIVANTE_GLES_GetProcAddress;
    device->GL_UnloadLibrary = VIVANTE_GLES_UnloadLibrary;
    device->GL_CreateContext = VIVANTE_GLES_CreateContext;
    device->GL_MakeCurrent = VIVANTE_GLES_MakeCurrent;
    device->GL_SetSwapInterval = VIVANTE_GLES_SetSwapInterval;
    device->GL_GetSwapInterval = VIVANTE_GLES_GetSwapInterval;
    device->GL_SwapWindow = VIVANTE_GLES_SwapWindow;
    device->GL_DeleteContext = VIVANTE_GLES_DeleteContext;
#endif

#ifdef SDL_VIDEO_VULKAN
    device->Vulkan_LoadLibrary = VIVANTE_Vulkan_LoadLibrary;
    device->Vulkan_UnloadLibrary = VIVANTE_Vulkan_UnloadLibrary;
    device->Vulkan_GetInstanceExtensions = VIVANTE_Vulkan_GetInstanceExtensions;
    device->Vulkan_CreateSurface = VIVANTE_Vulkan_CreateSurface;
#endif

    device->PumpEvents = VIVANTE_PumpEvents;

    return device;
}

VideoBootStrap VIVANTE_bootstrap = {
    "vivante",
    "Vivante EGL Video Driver",
    VIVANTE_Create,
    NULL /* no ShowMessageBox implementation */
};

/*****************************************************************************/
/* SDL Video and Display initialization/handling functions                   */
/*****************************************************************************/

static int VIVANTE_AddVideoDisplays(_THIS)
{
    SDL_VideoData *videodata = _this->driverdata;
    SDL_VideoDisplay display;
    SDL_DisplayMode current_mode;
    SDL_DisplayData *data;
    int pitch = 0, bpp = 0;
    unsigned long pixels = 0;

    data = (SDL_DisplayData *)SDL_calloc(1, sizeof(SDL_DisplayData));
    if (!data) {
        return SDL_OutOfMemory();
    }

    SDL_zero(current_mode);
#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
    data->native_display = vdkGetDisplay(videodata->vdk_private);

    vdkGetDisplayInfo(data->native_display, &current_mode.w, &current_mode.h, &pixels, &pitch, &bpp);
#else
    data->native_display = videodata->fbGetDisplayByIndex(0);

    videodata->fbGetDisplayInfo(data->native_display, &current_mode.w, &current_mode.h, &pixels, &pitch, &bpp);
#endif /* SDL_VIDEO_DRIVER_VIVANTE_VDK */

    switch (bpp) {
    default: /* Is another format used? */
    case 32:
        current_mode.format = SDL_PIXELFORMAT_ARGB8888;
        break;
    case 16:
        current_mode.format = SDL_PIXELFORMAT_RGB565;
        break;
    }
    /* FIXME: How do we query refresh rate? */
    current_mode.refresh_rate = 60;

    SDL_zero(display);
    display.name = VIVANTE_GetDisplayName(_this);
    display.desktop_mode = current_mode;
    display.current_mode = current_mode;
    display.driverdata = data;
    SDL_AddVideoDisplay(&display, SDL_FALSE);
    return 0;
}

int VIVANTE_VideoInit(_THIS)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;

#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
    videodata->vdk_private = vdkInitialize();
    if (!videodata->vdk_private) {
        return SDL_SetError("vdkInitialize() failed");
    }
#else
    videodata->egl_handle = SDL_LoadObject("libEGL.so.1");
    if (!videodata->egl_handle) {
        videodata->egl_handle = SDL_LoadObject("libEGL.so");
        if (!videodata->egl_handle) {
            return -1;
        }
    }
#define LOAD_FUNC(NAME)                                               \
    videodata->NAME = SDL_LoadFunction(videodata->egl_handle, #NAME); \
    if (!videodata->NAME)                                             \
        return -1;

    LOAD_FUNC(fbGetDisplay);
    LOAD_FUNC(fbGetDisplayByIndex);
    LOAD_FUNC(fbGetDisplayGeometry);
    LOAD_FUNC(fbGetDisplayInfo);
    LOAD_FUNC(fbDestroyDisplay);
    LOAD_FUNC(fbCreateWindow);
    LOAD_FUNC(fbGetWindowGeometry);
    LOAD_FUNC(fbGetWindowInfo);
    LOAD_FUNC(fbDestroyWindow);
#endif

    if (VIVANTE_SetupPlatform(_this) < 0) {
        return -1;
    }

    if (VIVANTE_AddVideoDisplays(_this) < 0) {
        return -1;
    }

    VIVANTE_UpdateDisplayScale(_this);

#ifdef SDL_INPUT_LINUXEV
    if (SDL_EVDEV_Init() < 0) {
        return -1;
    }
#endif

    return 0;
}

void VIVANTE_VideoQuit(_THIS)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;

#ifdef SDL_INPUT_LINUXEV
    SDL_EVDEV_Quit();
#endif

    VIVANTE_CleanupPlatform(_this);

#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
    if (videodata->vdk_private) {
        vdkExit(videodata->vdk_private);
        videodata->vdk_private = NULL;
    }
#else
    if (videodata->egl_handle) {
        SDL_UnloadObject(videodata->egl_handle);
        videodata->egl_handle = NULL;
    }
#endif
}

void VIVANTE_GetDisplayModes(_THIS, SDL_VideoDisplay *display)
{
    /* Only one display mode available, the current one */
    SDL_AddDisplayMode(display, &display->current_mode);
}

int VIVANTE_SetDisplayMode(_THIS, SDL_VideoDisplay *display, SDL_DisplayMode *mode)
{
    return 0;
}

int VIVANTE_CreateWindow(_THIS, SDL_Window *window)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
    SDL_DisplayData *displaydata;
    SDL_WindowData *data;

    displaydata = SDL_GetDisplayDriverData(0);

    /* Allocate window internal data */
    data = (SDL_WindowData *)SDL_calloc(1, sizeof(SDL_WindowData));
    if (!data) {
        return SDL_OutOfMemory();
    }

    /* Setup driver data for this window */
    window->driverdata = data;

#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
    data->native_window = vdkCreateWindow(displaydata->native_display, window->x, window->y, window->w, window->h);
#else
    data->native_window = videodata->fbCreateWindow(displaydata->native_display, window->x, window->y, window->w, window->h);
#endif
    if (!data->native_window) {
        return SDL_SetError("VIVANTE: Can't create native window");
    }

#ifdef SDL_VIDEO_OPENGL_EGL
    if (window->flags & SDL_WINDOW_OPENGL) {
        data->egl_surface = SDL_EGL_CreateSurface(_this, data->native_window);
        if (data->egl_surface == EGL_NO_SURFACE) {
            return SDL_SetError("VIVANTE: Can't create EGL surface");
        }
    } else {
        data->egl_surface = EGL_NO_SURFACE;
    }
#endif

    /* Window has been successfully created */
    return 0;
}

void VIVANTE_DestroyWindow(_THIS, SDL_Window *window)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
    SDL_WindowData *data;

    data = window->driverdata;
    if (data) {
#ifdef SDL_VIDEO_OPENGL_EGL
        if (data->egl_surface != EGL_NO_SURFACE) {
            SDL_EGL_DestroySurface(_this, data->egl_surface);
        }
#endif

        if (data->native_window) {
#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
            vdkDestroyWindow(data->native_window);
#else
            videodata->fbDestroyWindow(data->native_window);
#endif
        }

        SDL_free(data);
    }
    window->driverdata = NULL;
}

void VIVANTE_SetWindowTitle(_THIS, SDL_Window *window)
{
#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
    SDL_WindowData *data = window->driverdata;
    vdkSetWindowTitle(data->native_window, window->title);
#endif
}

void VIVANTE_SetWindowPosition(_THIS, SDL_Window *window)
{
    /* FIXME */
}

void VIVANTE_SetWindowSize(_THIS, SDL_Window *window)
{
    /* FIXME */
}

void VIVANTE_ShowWindow(_THIS, SDL_Window *window)
{
#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
    SDL_WindowData *data = window->driverdata;
    vdkShowWindow(data->native_window);
#endif
    SDL_SetMouseFocus(window);
    SDL_SetKeyboardFocus(window);
}

void VIVANTE_HideWindow(_THIS, SDL_Window *window)
{
#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
    SDL_WindowData *data = window->driverdata;
    vdkHideWindow(data->native_window);
#endif
    SDL_SetMouseFocus(NULL);
    SDL_SetKeyboardFocus(NULL);
}

/*****************************************************************************/
/* SDL Window Manager function                                               */
/*****************************************************************************/
SDL_bool VIVANTE_GetWindowWMInfo(_THIS, SDL_Window *window, struct SDL_SysWMinfo *info)
{
    SDL_WindowData *data = (SDL_WindowData *)window->driverdata;
    SDL_DisplayData *displaydata = SDL_GetDisplayDriverData(0);

    if (info->version.major == SDL_MAJOR_VERSION) {
        info->subsystem = SDL_SYSWM_VIVANTE;
        info->info.vivante.display = displaydata->native_display;
        info->info.vivante.window = data->native_window;
        return SDL_TRUE;
    } else {
        SDL_SetError("Application not compiled with SDL %d",
                     SDL_MAJOR_VERSION);
        return SDL_FALSE;
    }
}

/*****************************************************************************/
/* SDL event functions                                                       */
/*****************************************************************************/
void VIVANTE_PumpEvents(_THIS)
{
#ifdef SDL_INPUT_LINUXEV
    SDL_EVDEV_Poll();
#endif
}

#endif /* SDL_VIDEO_DRIVER_VIVANTE */

/* vi: set ts=4 sw=4 expandtab: */
