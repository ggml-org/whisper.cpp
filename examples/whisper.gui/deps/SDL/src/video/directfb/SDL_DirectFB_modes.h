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

#ifndef SDL_directfb_modes_h_
#define SDL_directfb_modes_h_

#include <directfb.h>

#include "../SDL_sysvideo.h"

#define SDL_DFB_DISPLAYDATA(win)  DFB_DisplayData *dispdata = ((win) ? (DFB_DisplayData *) SDL_GetDisplayForWindow(window)->driverdata : NULL)

typedef struct _DFB_DisplayData DFB_DisplayData;
struct _DFB_DisplayData
{
    IDirectFBDisplayLayer   *layer;
    DFBSurfacePixelFormat   pixelformat;
    /* FIXME: support for multiple video layer.
     * However, I do not know any card supporting
     * more than one
     */
    DFBDisplayLayerID       vidID;
    IDirectFBDisplayLayer   *vidlayer;

    int                     vidIDinuse;

    int                     cw;
    int                     ch;
};


extern void DirectFB_InitModes(_THIS);
extern void DirectFB_GetDisplayModes(_THIS, SDL_VideoDisplay * display);
extern int DirectFB_SetDisplayMode(_THIS, SDL_VideoDisplay * display, SDL_DisplayMode * mode);
extern void DirectFB_QuitModes(_THIS);

extern void DirectFB_SetContext(_THIS, SDL_Window *window);

#endif /* SDL_directfb_modes_h_ */

/* vi: set ts=4 sw=4 expandtab: */
