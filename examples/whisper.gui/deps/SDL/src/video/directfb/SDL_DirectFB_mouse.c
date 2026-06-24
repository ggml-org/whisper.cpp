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

#ifdef SDL_VIDEO_DRIVER_DIRECTFB


#include "SDL_DirectFB_video.h"
#include "SDL_DirectFB_mouse.h"
#include "SDL_DirectFB_modes.h"
#include "SDL_DirectFB_window.h"

#include "../SDL_sysvideo.h"
#include "../../events/SDL_mouse_c.h"

static SDL_Cursor *DirectFB_CreateDefaultCursor(void);
static SDL_Cursor *DirectFB_CreateCursor(SDL_Surface * surface,
                                         int hot_x, int hot_y);
static int DirectFB_ShowCursor(SDL_Cursor * cursor);
static void DirectFB_FreeCursor(SDL_Cursor * cursor);
static void DirectFB_WarpMouse(SDL_Window * window, int x, int y);

static const char *arrow[] = {
    /* pixels */
    "X                               ",
    "XX                              ",
    "X.X                             ",
    "X..X                            ",
    "X...X                           ",
    "X....X                          ",
    "X.....X                         ",
    "X......X                        ",
    "X.......X                       ",
    "X........X                      ",
    "X.....XXXXX                     ",
    "X..X..X                         ",
    "X.X X..X                        ",
    "XX  X..X                        ",
    "X    X..X                       ",
    "     X..X                       ",
    "      X..X                      ",
    "      X..X                      ",
    "       XX                       ",
    "                                ",
    "                                ",
    "                                ",
    "                                ",
    "                                ",
    "                                ",
    "                                ",
    "                                ",
    "                                ",
    "                                ",
    "                                ",
    "                                ",
    "                                ",
};

static SDL_Cursor *DirectFB_CreateDefaultCursor(void)
{
    SDL_VideoDevice *dev = SDL_GetVideoDevice();

    SDL_DFB_DEVICEDATA(dev);
    DFB_CursorData *curdata;
    DFBSurfaceDescription dsc;
    SDL_Cursor *cursor;
    Uint32 *dest;
    int pitch, i, j;

    SDL_DFB_ALLOC_CLEAR( cursor, sizeof(*cursor));
    SDL_DFB_ALLOC_CLEAR(curdata, sizeof(*curdata));

    dsc.flags =
        DSDESC_WIDTH | DSDESC_HEIGHT | DSDESC_PIXELFORMAT | DSDESC_CAPS;
    dsc.caps = DSCAPS_VIDEOONLY;
    dsc.width = 32;
    dsc.height = 32;
    dsc.pixelformat = DSPF_ARGB;

    SDL_DFB_CHECKERR(devdata->dfb->CreateSurface(devdata->dfb, &dsc,
                                                 &curdata->surf));
    curdata->hotx = 0;
    curdata->hoty = 0;
    cursor->driverdata = curdata;

    SDL_DFB_CHECKERR(curdata->surf->Lock(curdata->surf, DSLF_WRITE,
                                         (void *) &dest, &pitch));

    /* Relies on the fact that this is only called with ARGB surface. */
    for (i = 0; i < 32; i++)
    {
        for (j = 0; j < 32; j++)
        {
            switch (arrow[i][j])
            {
            case ' ': dest[j] = 0x00000000; break;
            case '.': dest[j] = 0xffffffff; break;
            case 'X': dest[j] = 0xff000000; break;
            }
        }
        dest += (pitch >> 2);
    }

    curdata->surf->Unlock(curdata->surf);
    return cursor;
  error:
    return NULL;
}

/* Create a cursor from a surface */
static SDL_Cursor *DirectFB_CreateCursor(SDL_Surface * surface, int hot_x, int hot_y)
{
    SDL_VideoDevice *dev = SDL_GetVideoDevice();

    SDL_DFB_DEVICEDATA(dev);
    DFB_CursorData *curdata;
    DFBSurfaceDescription dsc;
    SDL_Cursor *cursor;
    Uint32 *dest;
    Uint32 *p;
    int pitch, i;

    SDL_assert(surface->format->format == SDL_PIXELFORMAT_ARGB8888);
    SDL_assert(surface->pitch == surface->w * 4);

    SDL_DFB_ALLOC_CLEAR( cursor, sizeof(*cursor));
    SDL_DFB_ALLOC_CLEAR(curdata, sizeof(*curdata));

    dsc.flags =
        DSDESC_WIDTH | DSDESC_HEIGHT | DSDESC_PIXELFORMAT | DSDESC_CAPS;
    dsc.caps = DSCAPS_VIDEOONLY;
    dsc.width = surface->w;
    dsc.height = surface->h;
    dsc.pixelformat = DSPF_ARGB;

    SDL_DFB_CHECKERR(devdata->dfb->CreateSurface(devdata->dfb, &dsc,
                                                 &curdata->surf));
    curdata->hotx = hot_x;
    curdata->hoty = hot_y;
    cursor->driverdata = curdata;

    SDL_DFB_CHECKERR(curdata->surf->Lock(curdata->surf, DSLF_WRITE,
                                         (void *) &dest, &pitch));

    p = surface->pixels;
    for (i = 0; i < surface->h; i++)
        SDL_memcpy((char *) dest + i * pitch,
               (char *) p + i * surface->pitch, 4 * surface->w);

    curdata->surf->Unlock(curdata->surf);
    return cursor;
  error:
    return NULL;
}

/* Show the specified cursor, or hide if cursor is NULL */
static int DirectFB_ShowCursor(SDL_Cursor * cursor)
{
    SDL_DFB_CURSORDATA(cursor);
    SDL_Window *window;

    window = SDL_GetFocusWindow();
    if (!window)
        return -1;
    else {
        SDL_VideoDisplay *display = SDL_GetDisplayForWindow(window);

        if (display) {
            DFB_DisplayData *dispdata =
                (DFB_DisplayData *) display->driverdata;
            DFB_WindowData *windata = (DFB_WindowData *) window->driverdata;

            if (cursor)
                SDL_DFB_CHECKERR(windata->dfbwin->
                                 SetCursorShape(windata->dfbwin,
                                                curdata->surf, curdata->hotx,
                                                curdata->hoty));

            SDL_DFB_CHECKERR(dispdata->layer->
                             SetCooperativeLevel(dispdata->layer,
                                                 DLSCL_ADMINISTRATIVE));
            SDL_DFB_CHECKERR(dispdata->layer->
                             SetCursorOpacity(dispdata->layer,
                                              cursor ? 0xC0 : 0x00));
            SDL_DFB_CHECKERR(dispdata->layer->
                             SetCooperativeLevel(dispdata->layer,
                                                 DLSCL_SHARED));
        }
    }

    return 0;
  error:
    return -1;
}

/* Free a window manager cursor */
static void DirectFB_FreeCursor(SDL_Cursor * cursor)
{
    SDL_DFB_CURSORDATA(cursor);

    SDL_DFB_RELEASE(curdata->surf);
    SDL_DFB_FREE(cursor->driverdata);
    SDL_DFB_FREE(cursor);
}

/* Warp the mouse to (x,y) */
static void DirectFB_WarpMouse(SDL_Window * window, int x, int y)
{
    SDL_VideoDisplay *display = SDL_GetDisplayForWindow(window);
    DFB_DisplayData *dispdata = (DFB_DisplayData *) display->driverdata;
    DFB_WindowData *windata = (DFB_WindowData *) window->driverdata;
    int cx, cy;

    SDL_DFB_CHECKERR(windata->dfbwin->GetPosition(windata->dfbwin, &cx, &cy));
    SDL_DFB_CHECKERR(dispdata->layer->WarpCursor(dispdata->layer,
                                                 cx + x + windata->client.x,
                                                 cy + y + windata->client.y));

  error:
    return;
}

#if USE_MULTI_API

static void DirectFB_MoveCursor(SDL_Cursor * cursor);
static void DirectFB_WarpMouse(SDL_Mouse * mouse, SDL_Window * window,
                               int x, int y);
static void DirectFB_FreeMouse(SDL_Mouse * mouse);

static int id_mask;

static DFBEnumerationResult EnumMice(DFBInputDeviceID device_id, DFBInputDeviceDescription desc, void *callbackdata)
{
    DFB_DeviceData *devdata = callbackdata;

    if ((desc.type & DIDTF_MOUSE) && (device_id & id_mask)) {
        SDL_Mouse mouse;

        SDL_zero(mouse);
        mouse.id = device_id;
        mouse.CreateCursor = DirectFB_CreateCursor;
        mouse.ShowCursor = DirectFB_ShowCursor;
        mouse.MoveCursor = DirectFB_MoveCursor;
        mouse.FreeCursor = DirectFB_FreeCursor;
        mouse.WarpMouse = DirectFB_WarpMouse;
        mouse.FreeMouse = DirectFB_FreeMouse;
        mouse.cursor_shown = 1;

        SDL_AddMouse(&mouse, desc.name, 0, 0, 1);
        devdata->mouse_id[devdata->num_mice++] = device_id;
    }
    return DFENUM_OK;
}

void DirectFB_InitMouse(_THIS)
{
    SDL_DFB_DEVICEDATA(_this);

    devdata->num_mice = 0;
    if (devdata->use_linux_input) {
        /* try non-core devices first */
        id_mask = 0xF0;
        devdata->dfb->EnumInputDevices(devdata->dfb, EnumMice, devdata);
        if (devdata->num_mice == 0) {
            /* try core devices */
            id_mask = 0x0F;
            devdata->dfb->EnumInputDevices(devdata->dfb, EnumMice, devdata);
        }
    }
    if (devdata->num_mice == 0) {
        SDL_Mouse mouse;

        SDL_zero(mouse);
        mouse.CreateCursor = DirectFB_CreateCursor;
        mouse.ShowCursor = DirectFB_ShowCursor;
        mouse.MoveCursor = DirectFB_MoveCursor;
        mouse.FreeCursor = DirectFB_FreeCursor;
        mouse.WarpMouse = DirectFB_WarpMouse;
        mouse.FreeMouse = DirectFB_FreeMouse;
        mouse.cursor_shown = 1;

        SDL_AddMouse(&mouse, "Mouse", 0, 0, 1);
        devdata->num_mice = 1;
    }
}

void DirectFB_QuitMouse(_THIS)
{
    SDL_DFB_DEVICEDATA(_this);

    if (devdata->use_linux_input) {
        SDL_MouseQuit();
    } else {
        SDL_DelMouse(0);
    }
}


/* This is called when a mouse motion event occurs */
static void DirectFB_MoveCursor(SDL_Cursor * cursor)
{

}

/* Warp the mouse to (x,y) */
static void DirectFB_WarpMouse(SDL_Mouse * mouse, SDL_Window * window, int x, int y)
{
    SDL_VideoDisplay *display = SDL_GetDisplayForWindow(window);
    DFB_DisplayData *dispdata = (DFB_DisplayData *) display->driverdata;
    DFB_WindowData *windata = (DFB_WindowData *) window->driverdata;
    DFBResult ret;
    int cx, cy;

    SDL_DFB_CHECKERR(windata->dfbwin->GetPosition(windata->dfbwin, &cx, &cy));
    SDL_DFB_CHECKERR(dispdata->layer->WarpCursor(dispdata->layer,
                                                 cx + x + windata->client.x,
                                                 cy + y + windata->client.y));

  error:
    return;
}

/* Free the mouse when it's time */
static void DirectFB_FreeMouse(SDL_Mouse * mouse)
{
    /* nothing yet */
}

#else /* USE_MULTI_API */

void DirectFB_InitMouse(_THIS)
{
    SDL_DFB_DEVICEDATA(_this);

    SDL_Mouse *mouse = SDL_GetMouse();

    mouse->CreateCursor = DirectFB_CreateCursor;
    mouse->ShowCursor = DirectFB_ShowCursor;
    mouse->WarpMouse = DirectFB_WarpMouse;
    mouse->FreeCursor = DirectFB_FreeCursor;

    SDL_SetDefaultCursor(DirectFB_CreateDefaultCursor());

    devdata->num_mice = 1;
}

void DirectFB_QuitMouse(_THIS)
{
}

#endif

#endif /* SDL_VIDEO_DRIVER_DIRECTFB */

/* vi: set ts=4 sw=4 expandtab: */
