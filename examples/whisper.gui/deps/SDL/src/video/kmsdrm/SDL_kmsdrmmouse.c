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

#include "SDL_kmsdrmvideo.h"
#include "SDL_kmsdrmmouse.h"
#include "SDL_kmsdrmdyn.h"

#include "../../events/SDL_mouse_c.h"
#include "../../events/default_cursor.h"

#include "../SDL_pixels_c.h"

static SDL_Cursor *KMSDRM_CreateDefaultCursor(void);
static SDL_Cursor *KMSDRM_CreateCursor(SDL_Surface *surface, int hot_x, int hot_y);
static int KMSDRM_ShowCursor(SDL_Cursor *cursor);
static void KMSDRM_MoveCursor(SDL_Cursor *cursor);
static void KMSDRM_FreeCursor(SDL_Cursor *cursor);
static void KMSDRM_WarpMouse(SDL_Window *window, int x, int y);
static int KMSDRM_WarpMouseGlobal(int x, int y);

/**************************************************************************************/
/* BEFORE CODING ANYTHING MOUSE/CURSOR RELATED, REMEMBER THIS.                        */
/* How does SDL manage cursors internally? First, mouse =! cursor. The mouse can have */
/* many cursors in mouse->cursors.                                                    */
/* -SDL tells us to create a cursor with KMSDRM_CreateCursor(). It can create many    */
/*  cursosr with this, not only one.                                                  */
/* -SDL stores those cursors in a cursors array, in mouse->cursors.                   */
/* -Whenever it wants (or the programmer wants) takes a cursor from that array        */
/*  and shows it on screen with KMSDRM_ShowCursor().                                  */
/*  KMSDRM_ShowCursor() simply shows or hides the cursor it receives: it does NOT     */
/*  mind if it's mouse->cur_cursor, etc.                                              */
/* -If KMSDRM_ShowCursor() returns succesfully, that cursor becomes mouse->cur_cursor */
/*  and mouse->cursor_shown is 1.                                                     */
/**************************************************************************************/

static SDL_Cursor *KMSDRM_CreateDefaultCursor(void)
{
    return SDL_CreateCursor(default_cdata, default_cmask, DEFAULT_CWIDTH, DEFAULT_CHEIGHT, DEFAULT_CHOTX, DEFAULT_CHOTY);
}

/* Given a display's driverdata, destroy the cursor BO for it.
   To be called from KMSDRM_DestroyWindow(), as that's where we
   destroy the driverdata for the window's display. */
void KMSDRM_DestroyCursorBO(_THIS, SDL_VideoDisplay *display)
{
    SDL_DisplayData *dispdata = (SDL_DisplayData *)display->driverdata;

    /* Destroy the curso GBM BO. */
    if (dispdata->cursor_bo) {
        KMSDRM_gbm_bo_destroy(dispdata->cursor_bo);
        dispdata->cursor_bo = NULL;
        dispdata->cursor_bo_drm_fd = -1;
    }
}

/* Given a display's driverdata, create the cursor BO for it.
   To be called from KMSDRM_CreateWindow(), as that's where we
   build a window and assign a display to it. */
void KMSDRM_CreateCursorBO(SDL_VideoDisplay *display)
{

    SDL_VideoDevice *dev = SDL_GetVideoDevice();
    SDL_VideoData *viddata = ((SDL_VideoData *)dev->driverdata);
    SDL_DisplayData *dispdata = (SDL_DisplayData *)display->driverdata;

    if (!KMSDRM_gbm_device_is_format_supported(viddata->gbm_dev,
                                               GBM_FORMAT_ARGB8888,
                                               GBM_BO_USE_CURSOR | GBM_BO_USE_WRITE)) {
        SDL_SetError("Unsupported pixel format for cursor");
        return;
    }

    if (KMSDRM_drmGetCap(viddata->drm_fd,
                         DRM_CAP_CURSOR_WIDTH, &dispdata->cursor_w) ||
        KMSDRM_drmGetCap(viddata->drm_fd, DRM_CAP_CURSOR_HEIGHT,
                         &dispdata->cursor_h)) {
        SDL_SetError("Could not get the recommended GBM cursor size");
        return;
    }

    if (dispdata->cursor_w == 0 || dispdata->cursor_h == 0) {
        SDL_SetError("Could not get an usable GBM cursor size");
        return;
    }

    dispdata->cursor_bo = KMSDRM_gbm_bo_create(viddata->gbm_dev,
                                               dispdata->cursor_w, dispdata->cursor_h,
                                               GBM_FORMAT_ARGB8888, GBM_BO_USE_CURSOR | GBM_BO_USE_WRITE | GBM_BO_USE_LINEAR);

    if (!dispdata->cursor_bo) {
        SDL_SetError("Could not create GBM cursor BO");
        return;
    }

    dispdata->cursor_bo_drm_fd = viddata->drm_fd;
}

/* Remove a cursor buffer from a display's DRM cursor BO. */
static int KMSDRM_RemoveCursorFromBO(SDL_VideoDisplay *display)
{
    int ret = 0;
    SDL_DisplayData *dispdata = (SDL_DisplayData *)display->driverdata;
    SDL_VideoDevice *video_device = SDL_GetVideoDevice();
    SDL_VideoData *viddata = ((SDL_VideoData *)video_device->driverdata);

    ret = KMSDRM_drmModeSetCursor(viddata->drm_fd,
                                  dispdata->crtc->crtc_id, 0, 0, 0);

    if (ret) {
        ret = SDL_SetError("Could not hide current cursor with drmModeSetCursor().");
    }

    return ret;
}

/* Dump a cursor buffer to a display's DRM cursor BO.  */
static int KMSDRM_DumpCursorToBO(SDL_VideoDisplay *display, SDL_Cursor *cursor)
{
    SDL_DisplayData *dispdata = (SDL_DisplayData *)display->driverdata;
    KMSDRM_CursorData *curdata = (KMSDRM_CursorData *)cursor->driverdata;
    SDL_VideoDevice *video_device = SDL_GetVideoDevice();
    SDL_VideoData *viddata = ((SDL_VideoData *)video_device->driverdata);

    uint32_t bo_handle;
    size_t bo_stride;
    size_t bufsize;
    uint8_t *ready_buffer = NULL;
    uint8_t *src_row;

    int i;
    int ret;

    if (!curdata || !dispdata->cursor_bo) {
        return SDL_SetError("Cursor or display not initialized properly.");
    }

    /* Prepare a buffer we can dump to our GBM BO (different
       size, alpha premultiplication...) */
    bo_stride = KMSDRM_gbm_bo_get_stride(dispdata->cursor_bo);
    bufsize = bo_stride * dispdata->cursor_h;

    ready_buffer = (uint8_t *)SDL_calloc(1, bufsize);

    if (!ready_buffer) {
        ret = SDL_OutOfMemory();
        goto cleanup;
    }

    /* Copy from the cursor buffer to a buffer that we can dump to the GBM BO. */
    for (i = 0; i < curdata->h; i++) {
        src_row = &((uint8_t *)curdata->buffer)[i * curdata->w * 4];
        SDL_memcpy(ready_buffer + (i * bo_stride), src_row, (size_t)4 * curdata->w);
    }

    /* Dump the cursor buffer to our GBM BO. */
    if (KMSDRM_gbm_bo_write(dispdata->cursor_bo, ready_buffer, bufsize)) {
        ret = SDL_SetError("Could not write to GBM cursor BO");
        goto cleanup;
    }

    /* Put the GBM BO buffer on screen using the DRM interface. */
    bo_handle = KMSDRM_gbm_bo_get_handle(dispdata->cursor_bo).u32;
    if (curdata->hot_x == 0 && curdata->hot_y == 0) {
        ret = KMSDRM_drmModeSetCursor(viddata->drm_fd, dispdata->crtc->crtc_id,
                                      bo_handle, dispdata->cursor_w, dispdata->cursor_h);
    } else {
        ret = KMSDRM_drmModeSetCursor2(viddata->drm_fd, dispdata->crtc->crtc_id,
                                       bo_handle, dispdata->cursor_w, dispdata->cursor_h, curdata->hot_x, curdata->hot_y);
    }

    if (ret) {
        ret = SDL_SetError("Failed to set DRM cursor.");
        goto cleanup;
    }

    if (ret) {
        ret = SDL_SetError("Failed to reset cursor position.");
        goto cleanup;
    }

cleanup:

    if (ready_buffer) {
        SDL_free(ready_buffer);
    }
    return ret;
}

/* This is only for freeing the SDL_cursor.*/
static void KMSDRM_FreeCursor(SDL_Cursor *cursor)
{
    KMSDRM_CursorData *curdata;

    /* Even if the cursor is not ours, free it. */
    if (cursor) {
        curdata = (KMSDRM_CursorData *)cursor->driverdata;
        /* Free cursor buffer */
        if (curdata->buffer) {
            SDL_free(curdata->buffer);
            curdata->buffer = NULL;
        }
        /* Free cursor itself */
        if (cursor->driverdata) {
            SDL_free(cursor->driverdata);
        }
        SDL_free(cursor);
    }
}

/* This simply gets the cursor soft-buffer ready.
   We don't copy it to a GBO BO until ShowCursor() because the cusor GBM BO (living
   in dispata) is destroyed and recreated when we recreate windows, etc. */
static SDL_Cursor *KMSDRM_CreateCursor(SDL_Surface *surface, int hot_x, int hot_y)
{
    KMSDRM_CursorData *curdata;
    SDL_Cursor *cursor, *ret;

    curdata = NULL;
    ret = NULL;

    cursor = (SDL_Cursor *)SDL_calloc(1, sizeof(*cursor));
    if (!cursor) {
        SDL_OutOfMemory();
        goto cleanup;
    }
    curdata = (KMSDRM_CursorData *)SDL_calloc(1, sizeof(*curdata));
    if (!curdata) {
        SDL_OutOfMemory();
        goto cleanup;
    }

    /* hox_x and hot_y are the coordinates of the "tip of the cursor" from it's base. */
    curdata->hot_x = hot_x;
    curdata->hot_y = hot_y;
    curdata->w = surface->w;
    curdata->h = surface->h;
    curdata->buffer = NULL;

    /* Configure the cursor buffer info.
       This buffer has the original size of the cursor surface we are given. */
    curdata->buffer_pitch = surface->w;
    curdata->buffer_size = (size_t)surface->w * surface->h * 4;
    curdata->buffer = (uint32_t *)SDL_malloc(curdata->buffer_size);

    if (!curdata->buffer) {
        SDL_OutOfMemory();
        goto cleanup;
    }

    /* All code below assumes ARGB8888 format for the cursor surface,
       like other backends do. Also, the GBM BO pixels have to be
       alpha-premultiplied, but the SDL surface we receive has
       straight-alpha pixels, so we always have to convert. */
    SDL_PremultiplyAlpha(surface->w, surface->h,
                         surface->format->format, surface->pixels, surface->pitch,
                         SDL_PIXELFORMAT_ARGB8888, curdata->buffer, surface->w * 4);

    cursor->driverdata = curdata;

    ret = cursor;

cleanup:
    if (!ret) {
        if (curdata) {
            if (curdata->buffer) {
                SDL_free(curdata->buffer);
            }
            SDL_free(curdata);
        }
        if (cursor) {
            SDL_free(cursor);
        }
    }

    return ret;
}

/* Show the specified cursor, or hide if cursor is NULL or has no focus. */
static int KMSDRM_ShowCursor(SDL_Cursor *cursor)
{
    SDL_VideoDisplay *display;
    SDL_Window *window;
    SDL_Mouse *mouse;

    int num_displays, i;
    int ret = 0;

    /* Get the mouse focused window, if any. */

    mouse = SDL_GetMouse();
    if (!mouse) {
        return SDL_SetError("No mouse.");
    }

    window = mouse->focus;

    if (!window || !cursor) {

        /* If no window is focused by mouse or cursor is NULL,
           since we have no window (no mouse->focus) and hence
           we have no display, we simply hide mouse on all displays.
           This happens on video quit, where we get here after
           the mouse focus has been unset, yet SDL wants to
           restore the system default cursor (makes no sense here). */

        num_displays = SDL_GetNumVideoDisplays();

        /* Iterate on the displays hidding the cursor. */
        for (i = 0; i < num_displays; i++) {
            display = SDL_GetDisplay(i);
            ret = KMSDRM_RemoveCursorFromBO(display);
        }

    } else {

        display = SDL_GetDisplayForWindow(window);

        if (display) {

            if (cursor) {
                /* Dump the cursor to the display DRM cursor BO so it becomes visible
                   on that display. */
                ret = KMSDRM_DumpCursorToBO(display, cursor);

            } else {
                /* Hide the cursor on that display. */
                ret = KMSDRM_RemoveCursorFromBO(display);
            }
        }
    }

    return ret;
}

/* Warp the mouse to (x,y) */
static void KMSDRM_WarpMouse(SDL_Window *window, int x, int y)
{
    /* Only one global/fullscreen window is supported */
    KMSDRM_WarpMouseGlobal(x, y);
}

/* Warp the mouse to (x,y) */
static int KMSDRM_WarpMouseGlobal(int x, int y)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (mouse && mouse->cur_cursor && mouse->focus) {

        SDL_Window *window = mouse->focus;
        SDL_DisplayData *dispdata = (SDL_DisplayData *)SDL_GetDisplayForWindow(window)->driverdata;

        /* Update internal mouse position. */
        SDL_SendMouseMotion(mouse->focus, mouse->mouseID, 0, x, y);

        /* And now update the cursor graphic position on screen. */
        if (dispdata->cursor_bo) {
            int ret = 0;

            ret = KMSDRM_drmModeMoveCursor(dispdata->cursor_bo_drm_fd, dispdata->crtc->crtc_id, x, y);

            if (ret) {
                SDL_SetError("drmModeMoveCursor() failed.");
            }

            return ret;

        } else {
            return SDL_SetError("Cursor not initialized properly.");
        }

    } else {
        return SDL_SetError("No mouse or current cursor.");
    }
}

void KMSDRM_InitMouse(_THIS, SDL_VideoDisplay *display)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_DisplayData *dispdata = (SDL_DisplayData *)display->driverdata;

    mouse->CreateCursor = KMSDRM_CreateCursor;
    mouse->ShowCursor = KMSDRM_ShowCursor;
    mouse->MoveCursor = KMSDRM_MoveCursor;
    mouse->FreeCursor = KMSDRM_FreeCursor;
    mouse->WarpMouse = KMSDRM_WarpMouse;
    mouse->WarpMouseGlobal = KMSDRM_WarpMouseGlobal;

    /* Only create the default cursor for this display if we haven't done so before,
       we don't want several cursors to be created for the same display. */
    if (!dispdata->default_cursor_init) {
        SDL_SetDefaultCursor(KMSDRM_CreateDefaultCursor());
        dispdata->default_cursor_init = SDL_TRUE;
    }
}

void KMSDRM_QuitMouse(_THIS)
{
    /* TODO: ? */
}

/* This is called when a mouse motion event occurs */
static void KMSDRM_MoveCursor(SDL_Cursor *cursor)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    int ret = 0;

    /* We must NOT call SDL_SendMouseMotion() here or we will enter recursivity!
       That's why we move the cursor graphic ONLY. */
    if (mouse && mouse->cur_cursor && mouse->focus) {

        SDL_Window *window = mouse->focus;
        SDL_DisplayData *dispdata = (SDL_DisplayData *)SDL_GetDisplayForWindow(window)->driverdata;

        if (!dispdata->cursor_bo) {
            SDL_SetError("Cursor not initialized properly.");
            return;
        }

        ret = KMSDRM_drmModeMoveCursor(dispdata->cursor_bo_drm_fd, dispdata->crtc->crtc_id, mouse->x, mouse->y);

        if (ret) {
            SDL_SetError("drmModeMoveCursor() failed.");
        }
    }
}

#endif /* SDL_VIDEO_DRIVER_KMSDRM */

/* vi: set ts=4 sw=4 expandtab: */
