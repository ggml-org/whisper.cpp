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

#ifndef SDL_directfb_window_h_
#define SDL_directfb_window_h_

#include "SDL_DirectFB_video.h"
#include "SDL_DirectFB_WM.h"

#define SDL_DFB_WINDOWDATA(win)  DFB_WindowData *windata = ((win) ? (DFB_WindowData *) ((win)->driverdata) : NULL)

typedef struct _DFB_WindowData DFB_WindowData;
struct _DFB_WindowData
{
    IDirectFBSurface        *window_surface;    /* window surface */
    IDirectFBSurface        *surface;           /* client drawing surface */
    IDirectFBWindow         *dfbwin;
    IDirectFBEventBuffer    *eventbuffer;
    /* SDL_Window                *sdlwin; */
    SDL_Window              *next;
    Uint8                   opacity;
    DFBRectangle            client;
    DFBDimension            size;
    DFBRectangle            restore;

    /* WM extras */
    int                     is_managed;
    int                     wm_needs_redraw;
    IDirectFBSurface        *icon;
    IDirectFBFont           *font;
    DFB_Theme               theme;

    /* WM moving and sizing */
    int                     wm_grab;
    int                     wm_lastx;
    int                     wm_lasty;
};

extern int DirectFB_CreateWindow(_THIS, SDL_Window * window);
extern int DirectFB_CreateWindowFrom(_THIS, SDL_Window * window,
                                     const void *data);
extern void DirectFB_SetWindowTitle(_THIS, SDL_Window * window);
extern void DirectFB_SetWindowIcon(_THIS, SDL_Window * window,
                                   SDL_Surface * icon);

extern void DirectFB_SetWindowPosition(_THIS, SDL_Window * window);
extern void DirectFB_SetWindowSize(_THIS, SDL_Window * window);
extern void DirectFB_ShowWindow(_THIS, SDL_Window * window);
extern void DirectFB_HideWindow(_THIS, SDL_Window * window);
extern void DirectFB_RaiseWindow(_THIS, SDL_Window * window);
extern void DirectFB_MaximizeWindow(_THIS, SDL_Window * window);
extern void DirectFB_MinimizeWindow(_THIS, SDL_Window * window);
extern void DirectFB_RestoreWindow(_THIS, SDL_Window * window);
extern void DirectFB_SetWindowMouseGrab(_THIS, SDL_Window * window, SDL_bool grabbed);
extern void DirectFB_SetWindowKeyboardGrab(_THIS, SDL_Window * window, SDL_bool grabbed);
extern void DirectFB_DestroyWindow(_THIS, SDL_Window * window);
extern SDL_bool DirectFB_GetWindowWMInfo(_THIS, SDL_Window * window,
                                         struct SDL_SysWMinfo *info);

extern void DirectFB_AdjustWindowSurface(SDL_Window * window);
extern int DirectFB_SetWindowOpacity(_THIS, SDL_Window * window, float opacity);

#endif /* SDL_directfb_window_h_ */

/* vi: set ts=4 sw=4 expandtab: */
