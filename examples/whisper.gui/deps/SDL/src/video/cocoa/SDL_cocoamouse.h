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

#ifndef SDL_cocoamouse_h_
#define SDL_cocoamouse_h_

#include "SDL_cocoavideo.h"

extern int Cocoa_InitMouse(_THIS);
extern void Cocoa_HandleMouseEvent(_THIS, NSEvent * event);
extern void Cocoa_HandleMouseWheel(SDL_Window *window, NSEvent * event);
extern void Cocoa_HandleMouseWarp(CGFloat x, CGFloat y);
extern void Cocoa_QuitMouse(_THIS);

typedef struct {
    /* Whether we've seen a cursor warp since the last move event. */
    SDL_bool seenWarp;
    /* What location our last cursor warp was to. */
    CGFloat lastWarpX;
    CGFloat lastWarpY;
    /* What location we last saw the cursor move to. */
    CGFloat lastMoveX;
    CGFloat lastMoveY;
    /* If we just turned on relative mode, and should skip a single mouse motion event. */
    SDL_bool justEnabledRelative;
} SDL_MouseData;

@interface NSCursor (InvisibleCursor)
+ (NSCursor *)invisibleCursor;
@end

#endif /* SDL_cocoamouse_h_ */

/* vi: set ts=4 sw=4 expandtab: */
