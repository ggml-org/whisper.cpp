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

#ifdef SDL_TIMER_VITA

#include "SDL_thread.h"
#include "SDL_timer.h"
#include "SDL_error.h"
#include "../SDL_timer_c.h"
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <psp2/kernel/processmgr.h>

static uint64_t start;
static SDL_bool ticks_started = SDL_FALSE;

void SDL_TicksInit(void)
{
    if (ticks_started) {
        return;
    }
    ticks_started = SDL_TRUE;

    start = sceKernelGetProcessTimeWide();
}

void SDL_TicksQuit(void)
{
    ticks_started = SDL_FALSE;
}

Uint64 SDL_GetTicks64(void)
{
    uint64_t now;

    if (!ticks_started) {
        SDL_TicksInit();
    }

    now = sceKernelGetProcessTimeWide();
    return (Uint64)((now - start) / 1000);
}

Uint64 SDL_GetPerformanceCounter(void)
{
    return sceKernelGetProcessTimeWide();
}

Uint64 SDL_GetPerformanceFrequency(void)
{
    return 1000000;
}

void SDL_Delay(Uint32 ms)
{
    const Uint32 max_delay = 0xffffffffUL / 1000;
    if (ms > max_delay) {
        ms = max_delay;
    }
    sceKernelDelayThreadCB(ms * 1000);
}

#endif /* SDL_TIMER_VITA */

/* vi: set ts=4 sw=4 expandtab: */
