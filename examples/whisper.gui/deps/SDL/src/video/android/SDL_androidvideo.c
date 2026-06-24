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

#ifdef SDL_VIDEO_DRIVER_ANDROID

/* Android SDL video driver implementation */

#include "SDL_video.h"
#include "SDL_mouse.h"
#include "SDL_hints.h"
#include "../SDL_sysvideo.h"
#include "../SDL_pixels_c.h"
#include "../../events/SDL_events_c.h"
#include "../../events/SDL_windowevents_c.h"

#include "SDL_androidvideo.h"
#include "SDL_androidgl.h"
#include "SDL_androidclipboard.h"
#include "SDL_androidevents.h"
#include "SDL_androidkeyboard.h"
#include "SDL_androidmouse.h"
#include "SDL_androidtouch.h"
#include "SDL_androidwindow.h"
#include "SDL_androidvulkan.h"
#include "SDL_androidmessagebox.h"

#define ANDROID_VID_DRIVER_NAME "Android"

/* Initialization/Query functions */
static int Android_VideoInit(_THIS);
static void Android_VideoQuit(_THIS);
int Android_GetDisplayDPI(_THIS, SDL_VideoDisplay *display, float *ddpi, float *hdpi, float *vdpi);

#include "../SDL_egl_c.h"
#define Android_GLES_GetProcAddress SDL_EGL_GetProcAddress
#define Android_GLES_UnloadLibrary SDL_EGL_UnloadLibrary
#define Android_GLES_SetSwapInterval SDL_EGL_SetSwapInterval
#define Android_GLES_GetSwapInterval SDL_EGL_GetSwapInterval
#define Android_GLES_DeleteContext   SDL_EGL_DeleteContext

/* Android driver bootstrap functions */

/* These are filled in with real values in Android_SetScreenResolution on init (before SDL_main()) */
int Android_SurfaceWidth = 0;
int Android_SurfaceHeight = 0;
static int Android_DeviceWidth = 0;
static int Android_DeviceHeight = 0;
static Uint32 Android_ScreenFormat = SDL_PIXELFORMAT_RGB565; /* Default SurfaceView format, in case this is queried before being filled */
static int Android_ScreenRate = 0;
SDL_sem *Android_PauseSem = NULL;
SDL_sem *Android_ResumeSem = NULL;
SDL_mutex *Android_ActivityMutex = NULL;

static void Android_SuspendScreenSaver(_THIS)
{
    Android_JNI_SuspendScreenSaver(_this->suspend_screensaver);
}

static void Android_DeleteDevice(SDL_VideoDevice *device)
{
    SDL_free(device->driverdata);
    SDL_free(device);
}

static SDL_VideoDevice *Android_CreateDevice(void)
{
    SDL_VideoDevice *device;
    SDL_VideoData *data;
    SDL_bool block_on_pause;

    /* Initialize all variables that we clean on shutdown */
    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        SDL_OutOfMemory();
        return NULL;
    }

    data = (SDL_VideoData *)SDL_calloc(1, sizeof(SDL_VideoData));
    if (!data) {
        SDL_OutOfMemory();
        SDL_free(device);
        return NULL;
    }

    device->driverdata = data;

    /* Set the function pointers */
    device->VideoInit = Android_VideoInit;
    device->VideoQuit = Android_VideoQuit;
    block_on_pause = SDL_GetHintBoolean(SDL_HINT_ANDROID_BLOCK_ON_PAUSE, SDL_TRUE);
    if (block_on_pause) {
        device->PumpEvents = Android_PumpEvents_Blocking;
    } else {
        device->PumpEvents = Android_PumpEvents_NonBlocking;
    }

    device->GetDisplayDPI = Android_GetDisplayDPI;

    device->CreateSDLWindow = Android_CreateWindow;
    device->SetWindowTitle = Android_SetWindowTitle;
    device->SetWindowFullscreen = Android_SetWindowFullscreen;
    device->MinimizeWindow = Android_MinimizeWindow;
    device->SetWindowResizable = Android_SetWindowResizable;
    device->DestroyWindow = Android_DestroyWindow;
    device->GetWindowWMInfo = Android_GetWindowWMInfo;

    device->free = Android_DeleteDevice;

    /* GL pointers */
#ifdef SDL_VIDEO_OPENGL_EGL
    device->GL_LoadLibrary = Android_GLES_LoadLibrary;
    device->GL_GetProcAddress = Android_GLES_GetProcAddress;
    device->GL_UnloadLibrary = Android_GLES_UnloadLibrary;
    device->GL_CreateContext = Android_GLES_CreateContext;
    device->GL_MakeCurrent = Android_GLES_MakeCurrent;
    device->GL_SetSwapInterval = Android_GLES_SetSwapInterval;
    device->GL_GetSwapInterval = Android_GLES_GetSwapInterval;
    device->GL_SwapWindow = Android_GLES_SwapWindow;
    device->GL_DeleteContext = Android_GLES_DeleteContext;
#endif

#ifdef SDL_VIDEO_VULKAN
    device->Vulkan_LoadLibrary = Android_Vulkan_LoadLibrary;
    device->Vulkan_UnloadLibrary = Android_Vulkan_UnloadLibrary;
    device->Vulkan_GetInstanceExtensions = Android_Vulkan_GetInstanceExtensions;
    device->Vulkan_CreateSurface = Android_Vulkan_CreateSurface;
#endif

    /* Screensaver */
    device->SuspendScreenSaver = Android_SuspendScreenSaver;

    /* Text input */
    device->SetTextInputRect = Android_SetTextInputRect;

    /* Screen keyboard */
    device->HasScreenKeyboardSupport = Android_HasScreenKeyboardSupport;
    device->ShowScreenKeyboard = Android_ShowScreenKeyboard;
    device->HideScreenKeyboard = Android_HideScreenKeyboard;
    device->IsScreenKeyboardShown = Android_IsScreenKeyboardShown;

    /* Clipboard */
    device->SetClipboardText = Android_SetClipboardText;
    device->GetClipboardText = Android_GetClipboardText;
    device->HasClipboardText = Android_HasClipboardText;

    return device;
}

VideoBootStrap Android_bootstrap = {
    ANDROID_VID_DRIVER_NAME, "SDL Android video driver",
    Android_CreateDevice,
    Android_ShowMessageBox
};

int Android_VideoInit(_THIS)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
    int display_index;
    SDL_VideoDisplay *display;
    SDL_DisplayMode mode;

    videodata->isPaused = SDL_FALSE;
    videodata->isPausing = SDL_FALSE;
    videodata->pauseAudio = SDL_GetHintBoolean(SDL_HINT_ANDROID_BLOCK_ON_PAUSE_PAUSEAUDIO, SDL_TRUE);

    mode.format = Android_ScreenFormat;
    mode.w = Android_DeviceWidth;
    mode.h = Android_DeviceHeight;
    mode.refresh_rate = Android_ScreenRate;
    mode.driverdata = NULL;

    display_index = SDL_AddBasicVideoDisplay(&mode);
    if (display_index < 0) {
        return -1;
    }
    display = SDL_GetDisplay(display_index);
    display->orientation = Android_JNI_GetDisplayOrientation();

    SDL_AddDisplayMode(&_this->displays[0], &mode);

    Android_InitTouch();

    Android_InitMouse();

    /* We're done! */
    return 0;
}

void Android_VideoQuit(_THIS)
{
    Android_QuitMouse();
    Android_QuitTouch();
}

int Android_GetDisplayDPI(_THIS, SDL_VideoDisplay *display, float *ddpi, float *hdpi, float *vdpi)
{
    return Android_JNI_GetDisplayDPI(ddpi, hdpi, vdpi);
}

void Android_SetScreenResolution(int surfaceWidth, int surfaceHeight, int deviceWidth, int deviceHeight, float rate)
{
    Android_SurfaceWidth = surfaceWidth;
    Android_SurfaceHeight = surfaceHeight;
    Android_DeviceWidth = deviceWidth;
    Android_DeviceHeight = deviceHeight;
    Android_ScreenRate = (int)rate;
}

static Uint32 format_to_pixelFormat(int format)
{
    Uint32 pf;
    if (format == AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM) { /* 1 */
        pf = SDL_PIXELFORMAT_RGBA8888;
    } else if (format == AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM) { /* 2 */
        pf = SDL_PIXELFORMAT_RGBX8888;
    } else if (format == AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM) { /* 3 */
        pf = SDL_PIXELFORMAT_RGB24;
    } else if (format == AHARDWAREBUFFER_FORMAT_R5G6B5_UNORM) { /* 4*/
        pf = SDL_PIXELFORMAT_RGB565;
    } else if (format == 5) {
        pf = SDL_PIXELFORMAT_BGRA8888;
    } else if (format == 6) {
        pf = SDL_PIXELFORMAT_RGBA5551;
    } else if (format == 7) {
        pf = SDL_PIXELFORMAT_RGBA4444;
    } else if (format == 0x115) {
        /* HAL_PIXEL_FORMAT_BGR_565 */
        pf = SDL_PIXELFORMAT_RGB565;
    } else {
        pf = SDL_PIXELFORMAT_UNKNOWN;
    }
    return pf;
}

void Android_SetFormat(int format_wanted, int format_got)
{
    Uint32 pf_wanted;
    Uint32 pf_got;

    pf_wanted = format_to_pixelFormat(format_wanted);
    pf_got = format_to_pixelFormat(format_got);

    Android_ScreenFormat = pf_got;

    SDL_Log("pixel format wanted %s (%d), got %s (%d)",
            SDL_GetPixelFormatName(pf_wanted), format_wanted,
            SDL_GetPixelFormatName(pf_got), format_got);
}

void Android_SendResize(SDL_Window *window)
{
    /*
      Update the resolution of the desktop mode, so that the window
      can be properly resized. The screen resolution change can for
      example happen when the Activity enters or exits immersive mode,
      which can happen after VideoInit().
    */
    SDL_VideoDevice *device = SDL_GetVideoDevice();
    if (device && device->num_displays > 0) {
        SDL_VideoDisplay *display = &device->displays[0];
        display->desktop_mode.format = Android_ScreenFormat;
        display->desktop_mode.w = Android_DeviceWidth;
        display->desktop_mode.h = Android_DeviceHeight;
        display->desktop_mode.refresh_rate = Android_ScreenRate;
    }

    if (window) {
        /* Force the current mode to match the resize otherwise the SDL_WINDOWEVENT_RESTORED event
         * will fall back to the old mode */
        SDL_VideoDisplay *display = SDL_GetDisplayForWindow(window);
        display->display_modes[0].format = Android_ScreenFormat;
        display->display_modes[0].w = Android_DeviceWidth;
        display->display_modes[0].h = Android_DeviceHeight;
        display->display_modes[0].refresh_rate = Android_ScreenRate;
        display->current_mode = display->display_modes[0];

        SDL_SendWindowEvent(window, SDL_WINDOWEVENT_RESIZED, Android_SurfaceWidth, Android_SurfaceHeight);
    }
}

#endif /* SDL_VIDEO_DRIVER_ANDROID */

/* vi: set ts=4 sw=4 expandtab: */
