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

#ifndef SDL_cocoamodes_h_
#define SDL_cocoamodes_h_

typedef struct
{
    CGDirectDisplayID display;
} SDL_DisplayData;

typedef struct
{
    CFMutableArrayRef modes;
} SDL_DisplayModeData;

extern void Cocoa_InitModes(_THIS);
extern int Cocoa_GetDisplayBounds(_THIS, SDL_VideoDisplay * display, SDL_Rect * rect);
extern int Cocoa_GetDisplayUsableBounds(_THIS, SDL_VideoDisplay * display, SDL_Rect * rect);
extern void Cocoa_GetDisplayModes(_THIS, SDL_VideoDisplay * display);
extern int Cocoa_GetDisplayDPI(_THIS, SDL_VideoDisplay * display, float * ddpi, float * hpdi, float * vdpi);
extern int Cocoa_SetDisplayMode(_THIS, SDL_VideoDisplay * display, SDL_DisplayMode * mode);
extern void Cocoa_QuitModes(_THIS);

#endif /* SDL_cocoamodes_h_ */

/* vi: set ts=4 sw=4 expandtab: */
