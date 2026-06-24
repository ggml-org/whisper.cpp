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

#ifndef SDL_x11modes_h_
#define SDL_x11modes_h_

typedef struct
{
    int screen;
    Visual *visual;
    int depth;
    int scanline_pad;
    int x;
    int y;
    float ddpi;
    float hdpi;
    float vdpi;

    SDL_bool use_xrandr;

#ifdef SDL_VIDEO_DRIVER_X11_XRANDR
    RROutput xrandr_output;
#endif
} SDL_DisplayData;

typedef struct
{
#ifdef SDL_VIDEO_DRIVER_X11_XRANDR
    RRMode xrandr_mode;
#else
    int unused; /* just so struct isn't empty. */
#endif
} SDL_DisplayModeData;

extern int X11_InitModes(_THIS);
extern void X11_GetDisplayModes(_THIS, SDL_VideoDisplay *display);
extern int X11_SetDisplayMode(_THIS, SDL_VideoDisplay *display, SDL_DisplayMode *mode);
extern void X11_QuitModes(_THIS);

/* Some utility functions for working with visuals */
extern int X11_GetVisualInfoFromVisual(Display *display, Visual *visual,
                                       XVisualInfo *vinfo);
extern Uint32 X11_GetPixelFormatFromVisualInfo(Display *display,
                                               XVisualInfo *vinfo);
extern int X11_GetDisplayBounds(_THIS, SDL_VideoDisplay *sdl_display, SDL_Rect *rect);
extern int X11_GetDisplayUsableBounds(_THIS, SDL_VideoDisplay *sdl_display, SDL_Rect *rect);
extern int X11_GetDisplayDPI(_THIS, SDL_VideoDisplay *sdl_display, float *ddpi, float *hdpi, float *vdpi);

#ifdef SDL_VIDEO_DRIVER_X11_XRANDR
extern void X11_HandleXRandREvent(_THIS, const XEvent *xevent);
#endif

#endif /* SDL_x11modes_h_ */

/* vi: set ts=4 sw=4 expandtab: */
