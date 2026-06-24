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

#include "SDL_syswm.h"
#include "../SDL_sysvideo.h"
#include "../../events/SDL_keyboard_c.h"
#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_windowevents_c.h"
#include "../../core/android/SDL_android.h"

#include "SDL_androidvideo.h"
#include "SDL_androidwindow.h"
#include "SDL_hints.h"

/* Currently only one window */
SDL_Window *Android_Window = NULL;

int Android_CreateWindow(_THIS, SDL_Window *window)
{
    SDL_WindowData *data;
    int retval = 0;

    Android_ActivityMutex_Lock_Running();

    if (Android_Window) {
        retval = SDL_SetError("Android only supports one window");
        goto endfunction;
    }

    /* Set orientation */
    Android_JNI_SetOrientation(window->w, window->h, window->flags & SDL_WINDOW_RESIZABLE, SDL_GetHint(SDL_HINT_ORIENTATIONS));

    /* Adjust the window data to match the screen */
    window->x = 0;
    window->y = 0;
    window->w = Android_SurfaceWidth;
    window->h = Android_SurfaceHeight;

    window->flags &= ~SDL_WINDOW_HIDDEN;
    window->flags |= SDL_WINDOW_SHOWN; /* only one window on Android */

    /* One window, it always has focus */
    SDL_SetMouseFocus(window);
    SDL_SetKeyboardFocus(window);

    data = (SDL_WindowData *)SDL_calloc(1, sizeof(*data));
    if (!data) {
        retval = SDL_OutOfMemory();
        goto endfunction;
    }

    data->native_window = Android_JNI_GetNativeWindow();

    if (!data->native_window) {
        SDL_free(data);
        retval = SDL_SetError("Could not fetch native window");
        goto endfunction;
    }

    /* Do not create EGLSurface for Vulkan window since it will then make the window
       incompatible with vkCreateAndroidSurfaceKHR */
#ifdef SDL_VIDEO_OPENGL_EGL
    if (window->flags & SDL_WINDOW_OPENGL) {
        data->egl_surface = SDL_EGL_CreateSurface(_this, (NativeWindowType)data->native_window);

        if (data->egl_surface == EGL_NO_SURFACE) {
            ANativeWindow_release(data->native_window);
            SDL_free(data);
            retval = -1;
            goto endfunction;
        }
    }
#endif

    window->driverdata = data;
    Android_Window = window;

endfunction:

    SDL_UnlockMutex(Android_ActivityMutex);

    return retval;
}

void Android_SetWindowTitle(_THIS, SDL_Window *window)
{
    Android_JNI_SetActivityTitle(window->title);
}

void Android_SetWindowFullscreen(_THIS, SDL_Window *window, SDL_VideoDisplay *display, SDL_bool fullscreen)
{
    SDL_LockMutex(Android_ActivityMutex);

    if (window == Android_Window) {
        SDL_WindowData *data;
        int old_w, old_h, new_w, new_h;

        /* If the window is being destroyed don't change visible state */
        if (!window->is_destroying) {
            Android_JNI_SetWindowStyle(fullscreen);
        }

        /* Ensure our size matches reality after we've executed the window style change.
         *
         * It is possible that we've set width and height to the full-size display, but on
         * Samsung DeX or Chromebooks or other windowed Android environemtns, our window may
         * still not be the full display size.
         */
        if (!SDL_IsDeXMode() && !SDL_IsChromebook()) {
            goto endfunction;
        }

        data = (SDL_WindowData *)window->driverdata;
        if (!data || !data->native_window) {
            if (data && !data->native_window) {
                SDL_SetError("Missing native window");
            }
            goto endfunction;
        }

        old_w = window->w;
        old_h = window->h;

        new_w = ANativeWindow_getWidth(data->native_window);
        new_h = ANativeWindow_getHeight(data->native_window);

        if (new_w < 0 || new_h < 0) {
            SDL_SetError("ANativeWindow_getWidth/Height() fails");
        }

        if (old_w != new_w || old_h != new_h) {
            SDL_SendWindowEvent(window, SDL_WINDOWEVENT_RESIZED, new_w, new_h);
        }
    }

endfunction:

    SDL_UnlockMutex(Android_ActivityMutex);
}

void Android_MinimizeWindow(_THIS, SDL_Window *window)
{
    Android_JNI_MinizeWindow();
}

void Android_SetWindowResizable(_THIS, SDL_Window *window, SDL_bool resizable)
{
    /* Set orientation */
    Android_JNI_SetOrientation(window->w, window->h, window->flags & SDL_WINDOW_RESIZABLE, SDL_GetHint(SDL_HINT_ORIENTATIONS));
}

void Android_DestroyWindow(_THIS, SDL_Window *window)
{
    SDL_LockMutex(Android_ActivityMutex);

    if (window == Android_Window) {
        Android_Window = NULL;

        if (window->driverdata) {
            SDL_WindowData *data = (SDL_WindowData *)window->driverdata;

#ifdef SDL_VIDEO_OPENGL_EGL
            if (data->egl_surface != EGL_NO_SURFACE) {
                SDL_EGL_DestroySurface(_this, data->egl_surface);
            }
#endif

            if (data->native_window) {
                ANativeWindow_release(data->native_window);
            }
            SDL_free(window->driverdata);
            window->driverdata = NULL;
        }
    }

    SDL_UnlockMutex(Android_ActivityMutex);
}

SDL_bool Android_GetWindowWMInfo(_THIS, SDL_Window *window, SDL_SysWMinfo *info)
{
    SDL_WindowData *data = (SDL_WindowData *)window->driverdata;

    if (info->version.major == SDL_MAJOR_VERSION) {
        info->subsystem = SDL_SYSWM_ANDROID;
        info->info.android.window = data->native_window;

#ifdef SDL_VIDEO_OPENGL_EGL
        info->info.android.surface = data->egl_surface;
#endif

        return SDL_TRUE;
    } else {
        SDL_SetError("Application not compiled with SDL %d",
                     SDL_MAJOR_VERSION);
        return SDL_FALSE;
    }
}

#endif /* SDL_VIDEO_DRIVER_ANDROID */

/* vi: set ts=4 sw=4 expandtab: */
