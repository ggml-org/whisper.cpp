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

#ifndef SDL_directfb_wm_h_
#define SDL_directfb_wm_h_

#include "SDL_DirectFB_video.h"

typedef struct _DFB_Theme DFB_Theme;
struct _DFB_Theme
{
    int left_size;
    int right_size;
    int top_size;
    int bottom_size;
    DFBColor frame_color;
    int caption_size;
    DFBColor caption_color;
    int font_size;
    DFBColor font_color;
    char *font;
    DFBColor close_color;
    DFBColor max_color;
};

extern void DirectFB_WM_AdjustWindowLayout(SDL_Window * window, int flags, int w, int h);
extern void DirectFB_WM_RedrawLayout(_THIS, SDL_Window * window);

extern int DirectFB_WM_ProcessEvent(_THIS, SDL_Window * window,
                                    DFBWindowEvent * evt);

extern DFBResult DirectFB_WM_GetClientSize(_THIS, SDL_Window * window,
                                           int *cw, int *ch);


#endif /* SDL_directfb_wm_h_ */

/* vi: set ts=4 sw=4 expandtab: */
