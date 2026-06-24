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

#if defined(SDL_VIDEO_DRIVER_VITA) && defined(SDL_VIDEO_VITA_PVR)
#include <stdlib.h>
#include <string.h>
#include <psp2/kernel/modulemgr.h>
#include <gpu_es4/psp2_pvr_hint.h>

#include "SDL_error.h"
#include "SDL_log.h"
#include "SDL_vitavideo.h"
#include "../SDL_egl_c.h"
#include "SDL_vitagles_pvr_c.h"

#define MAX_PATH 256 // vita limits are somehow wrong

int VITA_GLES_LoadLibrary(_THIS, const char *path)
{
    PVRSRV_PSP2_APPHINT hint;
    char *override = SDL_getenv("VITA_MODULE_PATH");
    char *skip_init = SDL_getenv("VITA_PVR_SKIP_INIT");
    char *default_path = "app0:module";
    char target_path[MAX_PATH];

    if (!skip_init) { // we don't care about actual value

        if (override) {
            default_path = override;
        }

        sceKernelLoadStartModule("vs0:sys/external/libfios2.suprx", 0, NULL, 0, NULL, NULL);
        sceKernelLoadStartModule("vs0:sys/external/libc.suprx", 0, NULL, 0, NULL, NULL);

        SDL_snprintf(target_path, MAX_PATH, "%s/%s", default_path, "libgpu_es4_ext.suprx");
        sceKernelLoadStartModule(target_path, 0, NULL, 0, NULL, NULL);

        SDL_snprintf(target_path, MAX_PATH, "%s/%s", default_path, "libIMGEGL.suprx");
        sceKernelLoadStartModule(target_path, 0, NULL, 0, NULL, NULL);

        PVRSRVInitializeAppHint(&hint);

        SDL_snprintf(hint.szGLES1, MAX_PATH, "%s/%s", default_path, "libGLESv1_CM.suprx");
        SDL_snprintf(hint.szGLES2, MAX_PATH, "%s/%s", default_path, "libGLESv2.suprx");
        SDL_snprintf(hint.szWindowSystem, MAX_PATH, "%s/%s", default_path, "libpvrPSP2_WSEGL.suprx");

        PVRSRVCreateVirtualAppHint(&hint);
    }

    return SDL_EGL_LoadLibrary(_this, path, (NativeDisplayType)0, 0);
}

SDL_GLContext VITA_GLES_CreateContext(_THIS, SDL_Window *window)
{
    return SDL_EGL_CreateContext(_this, ((SDL_WindowData *)window->driverdata)->egl_surface);
}

int VITA_GLES_MakeCurrent(_THIS, SDL_Window *window, SDL_GLContext context)
{
    if (window && context) {
        return SDL_EGL_MakeCurrent(_this, ((SDL_WindowData *)window->driverdata)->egl_surface, context);
    } else {
        return SDL_EGL_MakeCurrent(_this, NULL, NULL);
    }
}

int VITA_GLES_SwapWindow(_THIS, SDL_Window *window)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
    if (videodata->ime_active) {
        sceImeUpdate();
    }
    return SDL_EGL_SwapBuffers(_this, ((SDL_WindowData *)window->driverdata)->egl_surface);
}

#endif /* SDL_VIDEO_DRIVER_VITA && SDL_VIDEO_VITA_PVR */

/* vi: set ts=4 sw=4 expandtab: */
