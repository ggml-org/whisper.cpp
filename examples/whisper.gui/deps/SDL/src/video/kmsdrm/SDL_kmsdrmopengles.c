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

#ifdef SDL_VIDEO_DRIVER_KMSDRM

#include "SDL_log.h"
#include "SDL_timer.h"

#include "SDL_kmsdrmvideo.h"
#include "SDL_kmsdrmopengles.h"
#include "SDL_kmsdrmdyn.h"
#include <errno.h>

#ifndef EGL_PLATFORM_GBM_MESA
#define EGL_PLATFORM_GBM_MESA 0x31D7
#endif

/* EGL implementation of SDL OpenGL support */

void KMSDRM_GLES_DefaultProfileConfig(_THIS, int *mask, int *major, int *minor)
{
    /* if SDL was _also_ built with the Raspberry Pi driver (so we're
       definitely a Pi device), default to GLES2. */
#ifdef SDL_VIDEO_DRIVER_RPI
    *mask = SDL_GL_CONTEXT_PROFILE_ES;
    *major = 2;
    *minor = 0;
#endif
}

int KMSDRM_GLES_LoadLibrary(_THIS, const char *path)
{
    /* Just pretend you do this here, but don't do it until KMSDRM_CreateWindow(),
       where we do the same library load we would normally do here.
       because this gets called by SDL_CreateWindow() before KMSDR_CreateWindow(),
       so gbm dev isn't yet created when this is called, AND we can't alter the
       call order in SDL_CreateWindow(). */
#if 0
    NativeDisplayType display = (NativeDisplayType)((SDL_VideoData *)_this->driverdata)->gbm_dev;
    return SDL_EGL_LoadLibrary(_this, path, display, EGL_PLATFORM_GBM_MESA);
#endif
    return 0;
}

void KMSDRM_GLES_UnloadLibrary(_THIS)
{
    /* As with KMSDRM_GLES_LoadLibrary(), we define our own "dummy" unloading function
       so we manually unload the library whenever we want. */
}

SDL_EGL_CreateContext_impl(KMSDRM)

    int KMSDRM_GLES_SetSwapInterval(_THIS, int interval)
{

    if (!_this->egl_data) {
        return SDL_SetError("EGL not initialized");
    }

    if (interval == 0 || interval == 1) {
        _this->egl_data->egl_swapinterval = interval;
    } else {
        return SDL_SetError("Only swap intervals of 0 or 1 are supported");
    }

    return 0;
}

int KMSDRM_GLES_SwapWindow(_THIS, SDL_Window *window)
{
    SDL_WindowData *windata = ((SDL_WindowData *)window->driverdata);
    SDL_DisplayData *dispdata = (SDL_DisplayData *)SDL_GetDisplayForWindow(window)->driverdata;
    SDL_VideoData *viddata = ((SDL_VideoData *)_this->driverdata);
    KMSDRM_FBInfo *fb_info;
    int ret = 0;

    /* Always wait for the previous issued flip before issuing a new one,
       even if you do async flips. */
    uint32_t flip_flags = DRM_MODE_PAGE_FLIP_EVENT;

    /* Skip the swap if we've switched away to another VT */
    if (windata->egl_surface == EGL_NO_SURFACE) {
        /* Wait a bit, throttling to ~100 FPS */
        SDL_Delay(10);
        return 0;
    }

    /* Recreate the GBM / EGL surfaces if the display mode has changed */
    if (windata->egl_surface_dirty) {
        KMSDRM_CreateSurfaces(_this, window);
    }

    /* Wait for confirmation that the next front buffer has been flipped, at which
       point the previous front buffer can be released */
    if (!KMSDRM_WaitPageflip(_this, windata)) {
        SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "Wait for previous pageflip failed");
        return 0;
    }

    /* Release the previous front buffer */
    if (windata->bo) {
        KMSDRM_gbm_surface_release_buffer(windata->gs, windata->bo);
        windata->bo = NULL;
    }

    windata->bo = windata->next_bo;

    /* Mark a buffer to become the next front buffer.
       This won't happen until pagelip completes. */
    if (!(_this->egl_data->eglSwapBuffers(_this->egl_data->egl_display,
                                          windata->egl_surface))) {
        SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "eglSwapBuffers failed");
        return 0;
    }

    /* From the GBM surface, get the next BO to become the next front buffer,
       and lock it so it can't be allocated as a back buffer (to prevent EGL
       from drawing into it!) */
    windata->next_bo = KMSDRM_gbm_surface_lock_front_buffer(windata->gs);
    if (!windata->next_bo) {
        SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "Could not lock front buffer on GBM surface");
        return 0;
    }

    /* Get an actual usable fb for the next front buffer. */
    fb_info = KMSDRM_FBFromBO(_this, windata->next_bo);
    if (!fb_info) {
        SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "Could not get a framebuffer");
        return 0;
    }

    if (!windata->bo) {
        /* On the first swap, immediately present the new front buffer. Before
           drmModePageFlip can be used the CRTC has to be configured to use
           the current connector and mode with drmModeSetCrtc */
        ret = KMSDRM_drmModeSetCrtc(viddata->drm_fd,
                                    dispdata->crtc->crtc_id, fb_info->fb_id, 0, 0,
                                    &dispdata->connector->connector_id, 1, &dispdata->mode);

        if (ret) {
            SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "Could not set videomode on CRTC.");
            return 0;
        }
    } else {
        /* On subsequent swaps, queue the new front buffer to be flipped during
           the next vertical blank

           Remember: drmModePageFlip() never blocks, it just issues the flip,
           which will be done during the next vblank, or immediately if
           we pass the DRM_MODE_PAGE_FLIP_ASYNC flag.
           Since calling drmModePageFlip() will return EBUSY if we call it
           without having completed the last issued flip, we must pass the
           DRM_MODE_PAGE_FLIP_ASYNC if we don't block on EGL (egl_swapinterval = 0).
           That makes it flip immediately, without waiting for the next vblank
           to do so, so even if we don't block on EGL, the flip will have completed
           when we get here again. */
        if (_this->egl_data->egl_swapinterval == 0 && viddata->async_pageflip_support) {
            flip_flags |= DRM_MODE_PAGE_FLIP_ASYNC;
        }

        ret = KMSDRM_drmModePageFlip(viddata->drm_fd, dispdata->crtc->crtc_id,
                                     fb_info->fb_id, flip_flags, &windata->waiting_for_flip);

        if (ret == 0) {
            windata->waiting_for_flip = SDL_TRUE;
        } else {
            SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "Could not queue pageflip: %d", ret);
        }

        /* Wait immediately for vsync (as if we only had two buffers).
           Even if we are already doing a WaitPageflip at the beginning of this
           function, this is NOT redundant because here we wait immediately
           after submitting the image to the screen, reducing lag, and if
           we have waited here, there won't be a pending pageflip so the
           WaitPageflip at the beginning of this function will be a no-op.
           Just leave it here and don't worry.
           Run your SDL2 program with "SDL_VIDEO_DOUBLE_BUFFER=1 <program_name>"
           to enable this. */
        if (windata->double_buffer) {
            if (!KMSDRM_WaitPageflip(_this, windata)) {
                SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "Immediate wait for previous pageflip failed");
                return 0;
            }
        }
    }

    return 1;
}

SDL_EGL_MakeCurrent_impl(KMSDRM)

#endif /* SDL_VIDEO_DRIVER_KMSDRM */

    /* vi: set ts=4 sw=4 expandtab: */
