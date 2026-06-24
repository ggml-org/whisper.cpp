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

#ifndef SDL_timer_c_h_
#define SDL_timer_c_h_

#include "../SDL_internal.h"

/* Useful functions and variables from SDL_timer.c */
#include "SDL_timer.h"

#define ROUND_RESOLUTION(X) \
    (((X + TIMER_RESOLUTION - 1) / TIMER_RESOLUTION) * TIMER_RESOLUTION)

extern void SDL_TicksInit(void);
extern void SDL_TicksQuit(void);
extern int SDL_TimerInit(void);
extern void SDL_TimerQuit(void);

#endif /* SDL_timer_c_h_ */

/* vi: set ts=4 sw=4 expandtab: */
