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

#if defined(SDL_TIMER_NGAGE)

#include <e32std.h>
#include <e32hal.h>

#include "SDL_timer.h"

static SDL_bool ticks_started = SDL_FALSE;
static TUint start = 0;
static TInt tickPeriodMilliSeconds;

#ifdef __cplusplus
extern "C" {
#endif

void SDL_TicksInit(void)
{
    if (ticks_started) {
        return;
    }
    ticks_started = SDL_TRUE;
    start = User::TickCount();

    TTimeIntervalMicroSeconds32 period;
    TInt tmp = UserHal::TickPeriod(period);

    (void)tmp; /* Suppress redundant warning. */

    tickPeriodMilliSeconds = period.Int() / 1000;
}

void SDL_TicksQuit(void)
{
    ticks_started = SDL_FALSE;
}

Uint64 SDL_GetTicks64(void)
{
    if (!ticks_started) {
        SDL_TicksInit();
    }

    TUint deltaTics = User::TickCount() - start;

    // Overlaps early, but should do the trick for now.
    return (Uint64)(deltaTics * tickPeriodMilliSeconds);
}

Uint64 SDL_GetPerformanceCounter(void)
{
    return (Uint64)User::TickCount();
}

Uint64 SDL_GetPerformanceFrequency(void)
{
    return 1000000;
}

void SDL_Delay(Uint32 ms)
{
    User::After(TTimeIntervalMicroSeconds32(ms * 1000));
}

#ifdef __cplusplus
}
#endif

#endif /* SDL_TIMER_NGAGE */

/* vi: set ts=4 sw=4 expandtab: */
