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

#ifdef SDL_VIDEO_DRIVER_RPI

/* References
 * http://elinux.org/RPi_VideoCore_APIs
 * https://github.com/raspberrypi/firmware/blob/master/opt/vc/src/hello_pi/hello_triangle/triangle.c
 * http://cgit.freedesktop.org/wayland/weston/tree/src/rpi-renderer.c
 * http://cgit.freedesktop.org/wayland/weston/tree/src/compositor-rpi.c
 */

/* SDL internals */
#include "../SDL_sysvideo.h"
#include "SDL_version.h"
#include "SDL_syswm.h"
#include "SDL_loadso.h"
#include "SDL_events.h"
#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_keyboard_c.h"
#include "SDL_hints.h"

#ifdef SDL_INPUT_LINUXEV
#include "../../core/linux/SDL_evdev.h"
#endif

/* RPI declarations */
#include "SDL_rpivideo.h"
#include "SDL_rpievents_c.h"
#include "SDL_rpiopengles.h"
#include "SDL_rpimouse.h"

static void RPI_Destroy(SDL_VideoDevice *device)
{
    SDL_free(device->driverdata);
    SDL_free(device);
}

static int RPI_GetRefreshRate(void)
{
    TV_DISPLAY_STATE_T tvstate;
    if (vc_tv_get_display_state(&tvstate) == 0) {
        // The width/height parameters are in the same position in the union
        // for HDMI and SDTV
        HDMI_PROPERTY_PARAM_T property;
        property.property = HDMI_PROPERTY_PIXEL_CLOCK_TYPE;
        vc_tv_hdmi_get_property(&property);
        return property.param1 == HDMI_PIXEL_CLOCK_TYPE_NTSC ? tvstate.display.hdmi.frame_rate * (1000.0f / 1001.0f) : tvstate.display.hdmi.frame_rate;
    }
    return 60; /* Failed to get display state, default to 60 */
}

static SDL_VideoDevice *RPI_Create()
{
    SDL_VideoDevice *device;
    SDL_VideoData *phdata;

    /* Initialize SDL_VideoDevice structure */
    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        SDL_OutOfMemory();
        return NULL;
    }

    /* Initialize internal data */
    phdata = (SDL_VideoData *)SDL_calloc(1, sizeof(SDL_VideoData));
    if (!phdata) {
        SDL_OutOfMemory();
        SDL_free(device);
        return NULL;
    }

    device->driverdata = phdata;

    /* Setup amount of available displays */
    device->num_displays = 0;

    /* Set device free function */
    device->free = RPI_Destroy;

    /* Setup all functions which we can handle */
    device->VideoInit = RPI_VideoInit;
    device->VideoQuit = RPI_VideoQuit;
    device->GetDisplayModes = RPI_GetDisplayModes;
    device->SetDisplayMode = RPI_SetDisplayMode;
    device->CreateSDLWindow = RPI_CreateWindow;
    device->CreateSDLWindowFrom = RPI_CreateWindowFrom;
    device->SetWindowTitle = RPI_SetWindowTitle;
    device->SetWindowIcon = RPI_SetWindowIcon;
    device->SetWindowPosition = RPI_SetWindowPosition;
    device->SetWindowSize = RPI_SetWindowSize;
    device->ShowWindow = RPI_ShowWindow;
    device->HideWindow = RPI_HideWindow;
    device->RaiseWindow = RPI_RaiseWindow;
    device->MaximizeWindow = RPI_MaximizeWindow;
    device->MinimizeWindow = RPI_MinimizeWindow;
    device->RestoreWindow = RPI_RestoreWindow;
    device->DestroyWindow = RPI_DestroyWindow;
#if 0
    device->GetWindowWMInfo = RPI_GetWindowWMInfo;
#endif
    device->GL_LoadLibrary = RPI_GLES_LoadLibrary;
    device->GL_GetProcAddress = RPI_GLES_GetProcAddress;
    device->GL_UnloadLibrary = RPI_GLES_UnloadLibrary;
    device->GL_CreateContext = RPI_GLES_CreateContext;
    device->GL_MakeCurrent = RPI_GLES_MakeCurrent;
    device->GL_SetSwapInterval = RPI_GLES_SetSwapInterval;
    device->GL_GetSwapInterval = RPI_GLES_GetSwapInterval;
    device->GL_SwapWindow = RPI_GLES_SwapWindow;
    device->GL_DeleteContext = RPI_GLES_DeleteContext;
    device->GL_DefaultProfileConfig = RPI_GLES_DefaultProfileConfig;

    device->PumpEvents = RPI_PumpEvents;

    return device;
}

VideoBootStrap RPI_bootstrap = {
    "RPI",
    "RPI Video Driver",
    RPI_Create,
    NULL /* no ShowMessageBox implementation */
};

/*****************************************************************************/
/* SDL Video and Display initialization/handling functions                   */
/*****************************************************************************/

static void AddDispManXDisplay(const int display_id)
{
    DISPMANX_MODEINFO_T modeinfo;
    DISPMANX_DISPLAY_HANDLE_T handle;
    SDL_VideoDisplay display;
    SDL_DisplayMode current_mode;
    SDL_DisplayData *data;

    handle = vc_dispmanx_display_open(display_id);
    if (!handle) {
        return; /* this display isn't available */
    }

    if (vc_dispmanx_display_get_info(handle, &modeinfo) < 0) {
        vc_dispmanx_display_close(handle);
        return;
    }

    /* RPI_GetRefreshRate() doesn't distinguish between displays. I'm not sure the hardware distinguishes either */
    SDL_zero(current_mode);
    current_mode.w = modeinfo.width;
    current_mode.h = modeinfo.height;
    current_mode.refresh_rate = RPI_GetRefreshRate();
    /* 32 bpp for default */
    current_mode.format = SDL_PIXELFORMAT_ABGR8888;

    current_mode.driverdata = NULL;

    SDL_zero(display);
    display.desktop_mode = current_mode;
    display.current_mode = current_mode;

    /* Allocate display internal data */
    data = (SDL_DisplayData *)SDL_calloc(1, sizeof(SDL_DisplayData));
    if (!data) {
        vc_dispmanx_display_close(handle);
        return; /* oh well */
    }

    data->dispman_display = handle;

    display.driverdata = data;

    SDL_AddVideoDisplay(&display, SDL_FALSE);
}

int RPI_VideoInit(_THIS)
{
    /* Initialize BCM Host */
    bcm_host_init();

    AddDispManXDisplay(DISPMANX_ID_MAIN_LCD);    /* your default display */
    AddDispManXDisplay(DISPMANX_ID_FORCE_OTHER); /* an "other" display...maybe DSI-connected screen while HDMI is your main */

#ifdef SDL_INPUT_LINUXEV
    if (SDL_EVDEV_Init() < 0) {
        return -1;
    }
#endif

    RPI_InitMouse(_this);

    return 1;
}

void RPI_VideoQuit(_THIS)
{
#ifdef SDL_INPUT_LINUXEV
    SDL_EVDEV_Quit();
#endif
}

void RPI_GetDisplayModes(_THIS, SDL_VideoDisplay *display)
{
    /* Only one display mode available, the current one */
    SDL_AddDisplayMode(display, &display->current_mode);
}

int RPI_SetDisplayMode(_THIS, SDL_VideoDisplay *display, SDL_DisplayMode *mode)
{
    return 0;
}

static void RPI_vsync_callback(DISPMANX_UPDATE_HANDLE_T u, void *data)
{
    SDL_WindowData *wdata = ((SDL_WindowData *)data);

    SDL_LockMutex(wdata->vsync_cond_mutex);
    SDL_CondSignal(wdata->vsync_cond);
    SDL_UnlockMutex(wdata->vsync_cond_mutex);
}

int RPI_CreateWindow(_THIS, SDL_Window *window)
{
    SDL_WindowData *wdata;
    SDL_VideoDisplay *display;
    SDL_DisplayData *displaydata;
    VC_RECT_T dst_rect;
    VC_RECT_T src_rect;
    VC_DISPMANX_ALPHA_T dispman_alpha;
    DISPMANX_UPDATE_HANDLE_T dispman_update;
    uint32_t layer = SDL_RPI_VIDEOLAYER;
    const char *env;

    /* Disable alpha, otherwise the app looks composed with whatever dispman is showing (X11, console,etc) */
    dispman_alpha.flags = DISPMANX_FLAGS_ALPHA_FIXED_ALL_PIXELS;
    dispman_alpha.opacity = 0xFF;
    dispman_alpha.mask = 0;

    /* Allocate window internal data */
    wdata = (SDL_WindowData *)SDL_calloc(1, sizeof(SDL_WindowData));
    if (!wdata) {
        return SDL_OutOfMemory();
    }
    display = SDL_GetDisplayForWindow(window);
    displaydata = (SDL_DisplayData *)display->driverdata;

    /* Windows have one size for now */
    window->w = display->desktop_mode.w;
    window->h = display->desktop_mode.h;

    /* OpenGL ES is the law here, buddy */
    window->flags |= SDL_WINDOW_OPENGL;

    /* Create a dispman element and associate a window to it */
    dst_rect.x = 0;
    dst_rect.y = 0;
    dst_rect.width = window->w;
    dst_rect.height = window->h;

    src_rect.x = 0;
    src_rect.y = 0;
    src_rect.width = window->w << 16;
    src_rect.height = window->h << 16;

    env = SDL_GetHint(SDL_HINT_RPI_VIDEO_LAYER);
    if (env) {
        layer = SDL_atoi(env);
    }

    dispman_update = vc_dispmanx_update_start(0);
    wdata->dispman_window.element = vc_dispmanx_element_add(dispman_update,
                                                            displaydata->dispman_display,
                                                            layer /* layer */,
                                                            &dst_rect,
                                                            0 /*src*/,
                                                            &src_rect,
                                                            DISPMANX_PROTECTION_NONE,
                                                            &dispman_alpha /*alpha*/,
                                                            0 /*clamp*/,
                                                            0 /*transform*/);
    wdata->dispman_window.width = window->w;
    wdata->dispman_window.height = window->h;
    vc_dispmanx_update_submit_sync(dispman_update);

    if (!_this->egl_data) {
        if (SDL_GL_LoadLibrary(NULL) < 0) {
            return -1;
        }
    }
    wdata->egl_surface = SDL_EGL_CreateSurface(_this, (NativeWindowType)&wdata->dispman_window);

    if (wdata->egl_surface == EGL_NO_SURFACE) {
        return SDL_SetError("Could not create GLES window surface");
    }

    /* Start generating vsync callbacks if necesary */
    wdata->double_buffer = SDL_FALSE;
    if (SDL_GetHintBoolean(SDL_HINT_VIDEO_DOUBLE_BUFFER, SDL_FALSE)) {
        wdata->vsync_cond = SDL_CreateCond();
        wdata->vsync_cond_mutex = SDL_CreateMutex();
        wdata->double_buffer = SDL_TRUE;
        vc_dispmanx_vsync_callback(displaydata->dispman_display, RPI_vsync_callback, (void *)wdata);
    }

    /* Setup driver data for this window */
    window->driverdata = wdata;

    /* One window, it always has focus */
    SDL_SetMouseFocus(window);
    SDL_SetKeyboardFocus(window);

    /* Window has been successfully created */
    return 0;
}

void RPI_DestroyWindow(_THIS, SDL_Window *window)
{
    SDL_WindowData *data = (SDL_WindowData *)window->driverdata;
    SDL_VideoDisplay *display = SDL_GetDisplayForWindow(window);
    SDL_DisplayData *displaydata = (SDL_DisplayData *)display->driverdata;

    if (data) {
        if (data->double_buffer) {
            /* Wait for vsync, and then stop vsync callbacks and destroy related stuff, if needed */
            SDL_LockMutex(data->vsync_cond_mutex);
            SDL_CondWait(data->vsync_cond, data->vsync_cond_mutex);
            SDL_UnlockMutex(data->vsync_cond_mutex);

            vc_dispmanx_vsync_callback(displaydata->dispman_display, NULL, NULL);

            SDL_DestroyCond(data->vsync_cond);
            SDL_DestroyMutex(data->vsync_cond_mutex);
        }

#ifdef SDL_VIDEO_OPENGL_EGL
        if (data->egl_surface != EGL_NO_SURFACE) {
            SDL_EGL_DestroySurface(_this, data->egl_surface);
        }
#endif
        SDL_free(data);
        window->driverdata = NULL;
    }
}

int RPI_CreateWindowFrom(_THIS, SDL_Window *window, const void *data)
{
    return -1;
}

void RPI_SetWindowTitle(_THIS, SDL_Window *window)
{
}
void RPI_SetWindowIcon(_THIS, SDL_Window *window, SDL_Surface *icon)
{
}
void RPI_SetWindowPosition(_THIS, SDL_Window *window)
{
}
void RPI_SetWindowSize(_THIS, SDL_Window *window)
{
}
void RPI_ShowWindow(_THIS, SDL_Window *window)
{
}
void RPI_HideWindow(_THIS, SDL_Window *window)
{
}
void RPI_RaiseWindow(_THIS, SDL_Window *window)
{
}
void RPI_MaximizeWindow(_THIS, SDL_Window *window)
{
}
void RPI_MinimizeWindow(_THIS, SDL_Window *window)
{
}
void RPI_RestoreWindow(_THIS, SDL_Window *window)
{
}

/*****************************************************************************/
/* SDL Window Manager function                                               */
/*****************************************************************************/
#if 0
SDL_bool RPI_GetWindowWMInfo(_THIS, SDL_Window * window, struct SDL_SysWMinfo *info)
{
    if (info->version.major <= SDL_MAJOR_VERSION) {
        return SDL_TRUE;
    } else {
        SDL_SetError("application not compiled with SDL %d",
                     SDL_MAJOR_VERSION);
        return SDL_FALSE;
    }

    /* Failed to get window manager information */
    return SDL_FALSE;
}
#endif

#endif /* SDL_VIDEO_DRIVER_RPI */

/* vi: set ts=4 sw=4 expandtab: */
