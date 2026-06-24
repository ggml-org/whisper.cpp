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

#ifdef SDL_TIMER_UNIX

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>

#include "SDL_timer.h"
#include "SDL_hints.h"
#include "../SDL_timer_c.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

/* The clock_gettime provides monotonous time, so we should use it if
   it's available. The clock_gettime function is behind ifdef
   for __USE_POSIX199309
   Tommi Kyntola (tommi.kyntola@ray.fi) 27/09/2005
*/
/* Reworked monotonic clock to not assume the current system has one
   as not all linux kernels provide a monotonic clock (yeah recent ones
   probably do)
   Also added OS X Monotonic clock support
   Based on work in https://github.com/ThomasHabets/monotonic_clock
 */
#if defined(HAVE_NANOSLEEP) || defined(HAVE_CLOCK_GETTIME)
#include <time.h>
#endif
#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

/* Use CLOCK_MONOTONIC_RAW, if available, which is not subject to adjustment by NTP */
#ifdef HAVE_CLOCK_GETTIME
#ifdef CLOCK_MONOTONIC_RAW
#define SDL_MONOTONIC_CLOCK CLOCK_MONOTONIC_RAW
#else
#define SDL_MONOTONIC_CLOCK CLOCK_MONOTONIC
#endif
#endif

/* The first ticks value of the application */
#ifdef HAVE_CLOCK_GETTIME
static struct timespec start_ts;
#elif defined(__APPLE__)
static uint64_t start_mach;
mach_timebase_info_data_t mach_base_info;
#endif
static SDL_bool has_monotonic_time = SDL_FALSE;
static struct timeval start_tv;
static SDL_bool ticks_started = SDL_FALSE;

void SDL_TicksInit(void)
{
    if (ticks_started) {
        return;
    }
    ticks_started = SDL_TRUE;

    /* Set first ticks value */
#ifdef HAVE_CLOCK_GETTIME
    if (clock_gettime(SDL_MONOTONIC_CLOCK, &start_ts) == 0) {
        has_monotonic_time = SDL_TRUE;
    } else
#elif defined(__APPLE__)
    if (mach_timebase_info(&mach_base_info) == 0) {
        has_monotonic_time = SDL_TRUE;
        start_mach = mach_absolute_time();
    } else
#endif
    {
        gettimeofday(&start_tv, NULL);
    }
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

    if (has_monotonic_time) {
#ifdef HAVE_CLOCK_GETTIME
        struct timespec now;
        clock_gettime(SDL_MONOTONIC_CLOCK, &now);
        return (Uint64)(((Sint64)(now.tv_sec - start_ts.tv_sec) * 1000) + ((now.tv_nsec - start_ts.tv_nsec) / 1000000));
#elif defined(__APPLE__)
        const uint64_t now = mach_absolute_time();
        return (((now - start_mach) * mach_base_info.numer) / mach_base_info.denom) / 1000000;
#else
        SDL_assert(SDL_FALSE);
        return 0;
#endif
    } else {
        struct timeval now;
        gettimeofday(&now, NULL);
        return (Uint64)(((Sint64)(now.tv_sec - start_tv.tv_sec) * 1000) + ((now.tv_usec - start_tv.tv_usec) / 1000));
    }
}

Uint64 SDL_GetPerformanceCounter(void)
{
    Uint64 ticks;
    if (!ticks_started) {
        SDL_TicksInit();
    }

    if (has_monotonic_time) {
#ifdef HAVE_CLOCK_GETTIME
        struct timespec now;

        clock_gettime(SDL_MONOTONIC_CLOCK, &now);
        ticks = now.tv_sec;
        ticks *= 1000000000;
        ticks += now.tv_nsec;
#elif defined(__APPLE__)
        ticks = mach_absolute_time();
#else
        SDL_assert(SDL_FALSE);
        ticks = 0;
#endif
    } else {
        struct timeval now;

        gettimeofday(&now, NULL);
        ticks = now.tv_sec;
        ticks *= 1000000;
        ticks += now.tv_usec;
    }
    return ticks;
}

Uint64 SDL_GetPerformanceFrequency(void)
{
    if (!ticks_started) {
        SDL_TicksInit();
    }

    if (has_monotonic_time) {
#ifdef HAVE_CLOCK_GETTIME
        return 1000000000;
#elif defined(__APPLE__)
        Uint64 freq = mach_base_info.denom;
        freq *= 1000000000;
        freq /= mach_base_info.numer;
        return freq;
#endif
    }

    return 1000000;
}

void SDL_Delay(Uint32 ms)
{
    int was_error;

#ifdef HAVE_NANOSLEEP
    struct timespec elapsed, tv;
#else
    struct timeval tv;
    Uint64 then, now, elapsed;
#endif

#ifdef __EMSCRIPTEN__
    if (emscripten_has_asyncify() && SDL_GetHintBoolean(SDL_HINT_EMSCRIPTEN_ASYNCIFY, SDL_TRUE)) {
        /* pseudo-synchronous pause, used directly or through e.g. SDL_WaitEvent */
        emscripten_sleep(ms);
        return;
    }
#endif

    /* Set the timeout interval */
#ifdef HAVE_NANOSLEEP
    elapsed.tv_sec = ms / 1000;
    elapsed.tv_nsec = (ms % 1000) * 1000000;
#else
    then = SDL_GetTicks64();
#endif
    do {
        errno = 0;

#ifdef HAVE_NANOSLEEP
        tv.tv_sec = elapsed.tv_sec;
        tv.tv_nsec = elapsed.tv_nsec;
        was_error = nanosleep(&tv, &elapsed);
#else
        /* Calculate the time interval left (in case of interrupt) */
        now = SDL_GetTicks64();
        elapsed = (now - then);
        then = now;
        if (elapsed >= ((Uint64)ms)) {
            break;
        }
        ms -= (Uint32)elapsed;
        tv.tv_sec = ms / 1000;
        tv.tv_usec = (ms % 1000) * 1000;

        was_error = select(0, NULL, NULL, NULL, &tv);
#endif /* HAVE_NANOSLEEP */
    } while (was_error && (errno == EINTR));
}

#endif /* SDL_TIMER_UNIX */

/* vi: set ts=4 sw=4 expandtab: */
