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

#ifdef SDL_THREAD_N3DS

/* An implementation of semaphores using libctru's LightSemaphore */

#include <3ds.h>

#include "SDL_thread.h"
#include "SDL_timer.h"

int WaitOnSemaphoreFor(SDL_sem *sem, Uint32 timeout);

struct SDL_semaphore
{
    LightSemaphore semaphore;
};

SDL_sem *SDL_CreateSemaphore(Uint32 initial_value)
{
    SDL_sem *sem;

    if (initial_value > SDL_MAX_SINT16) {
        SDL_SetError("Initial semaphore value too high for this platform");
        return NULL;
    }

    sem = (SDL_sem *)SDL_malloc(sizeof(*sem));
    if (!sem) {
        SDL_OutOfMemory();
        return NULL;
    }

    LightSemaphore_Init(&sem->semaphore, initial_value, SDL_MAX_SINT16);

    return sem;
}

/* WARNING:
   You cannot call this function when another thread is using the semaphore.
*/
void SDL_DestroySemaphore(SDL_sem *sem)
{
    SDL_free(sem);
}

int SDL_SemTryWait(SDL_sem *sem)
{
    if (!sem) {
        return SDL_InvalidParamError("sem");
    }

    if (LightSemaphore_TryAcquire(&sem->semaphore, 1) != 0) {
        /* If we failed, yield to avoid starvation on busy waits */
        svcSleepThread(1);
        return SDL_MUTEX_TIMEDOUT;
    }

    return 0;
}

int SDL_SemWaitTimeout(SDL_sem *sem, Uint32 timeout)
{
    if (!sem) {
        return SDL_InvalidParamError("sem");
    }

    if (timeout == SDL_MUTEX_MAXWAIT) {
        LightSemaphore_Acquire(&sem->semaphore, 1);
        return 0;
    }

    if (LightSemaphore_TryAcquire(&sem->semaphore, 1) != 0) {
        return WaitOnSemaphoreFor(sem, timeout);
    }

    return 0;
}

int WaitOnSemaphoreFor(SDL_sem *sem, Uint32 timeout)
{
    Uint64 stop_time = SDL_GetTicks64() + timeout;
    Uint64 current_time = SDL_GetTicks64();
    while (current_time < stop_time) {
        if (LightSemaphore_TryAcquire(&sem->semaphore, 1) == 0) {
            return 0;
        }
        /* 100 microseconds seems to be the sweet spot */
        svcSleepThread(100000LL);
        current_time = SDL_GetTicks64();
    }

    /* If we failed, yield to avoid starvation on busy waits */
    svcSleepThread(1);
    return SDL_MUTEX_TIMEDOUT;
}

int SDL_SemWait(SDL_sem *sem)
{
    return SDL_SemWaitTimeout(sem, SDL_MUTEX_MAXWAIT);
}

Uint32 SDL_SemValue(SDL_sem *sem)
{
    if (!sem) {
        SDL_InvalidParamError("sem");
        return 0;
    }
    return sem->semaphore.current_count;
}

int SDL_SemPost(SDL_sem *sem)
{
    if (!sem) {
        return SDL_InvalidParamError("sem");
    }
    LightSemaphore_Release(&sem->semaphore, 1);
    return 0;
}

#endif /* SDL_THREAD_N3DS */

/* vi: set sts=4 ts=4 sw=4 expandtab: */
