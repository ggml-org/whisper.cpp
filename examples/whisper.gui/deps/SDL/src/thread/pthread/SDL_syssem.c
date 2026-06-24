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

#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>
#include <time.h>

#include "SDL_thread.h"
#include "SDL_timer.h"

/* Wrapper around POSIX 1003.1b semaphores */

#if defined(__MACOSX__) || defined(__IPHONEOS__)
/* Mac OS X doesn't support sem_getvalue() as of version 10.4 */
#include "../generic/SDL_syssem.c"
#else

struct SDL_semaphore
{
    sem_t sem;
};

/* Create a semaphore, initialized with value */
SDL_sem *SDL_CreateSemaphore(Uint32 initial_value)
{
    SDL_sem *sem = (SDL_sem *)SDL_malloc(sizeof(SDL_sem));
    if (sem) {
        if (sem_init(&sem->sem, 0, initial_value) < 0) {
            SDL_SetError("sem_init() failed");
            SDL_free(sem);
            sem = NULL;
        }
    } else {
        SDL_OutOfMemory();
    }
    return sem;
}

void SDL_DestroySemaphore(SDL_sem *sem)
{
    if (sem) {
        sem_destroy(&sem->sem);
        SDL_free(sem);
    }
}

int SDL_SemTryWait(SDL_sem *sem)
{
    int retval;

    if (!sem) {
        return SDL_InvalidParamError("sem");
    }
    retval = SDL_MUTEX_TIMEDOUT;
    if (sem_trywait(&sem->sem) == 0) {
        retval = 0;
    }
    return retval;
}

int SDL_SemWait(SDL_sem *sem)
{
    int retval;

    if (!sem) {
        return SDL_InvalidParamError("sem");
    }

    do {
        retval = sem_wait(&sem->sem);
    } while (retval < 0 && errno == EINTR);

    if (retval < 0) {
        retval = SDL_SetError("sem_wait() failed");
    }
    return retval;
}

int SDL_SemWaitTimeout(SDL_sem *sem, Uint32 timeout)
{
    int retval = 0;
#ifdef HAVE_SEM_TIMEDWAIT
#ifndef HAVE_CLOCK_GETTIME
    struct timeval now;
#endif
    struct timespec ts_timeout;
#else
    Uint32 end;
#endif

    if (!sem) {
        return SDL_InvalidParamError("sem");
    }

    /* Try the easy cases first */
    if (timeout == 0) {
        return SDL_SemTryWait(sem);
    }
    if (timeout == SDL_MUTEX_MAXWAIT) {
        return SDL_SemWait(sem);
    }

#ifdef HAVE_SEM_TIMEDWAIT
    /* Setup the timeout. sem_timedwait doesn't wait for
     * a lapse of time, but until we reach a certain time.
     * This time is now plus the timeout.
     */
#ifdef HAVE_CLOCK_GETTIME
    clock_gettime(CLOCK_REALTIME, &ts_timeout);

    /* Add our timeout to current time */
    ts_timeout.tv_nsec += (timeout % 1000) * 1000000;
    ts_timeout.tv_sec += timeout / 1000;
#else
    gettimeofday(&now, NULL);

    /* Add our timeout to current time */
    ts_timeout.tv_sec = now.tv_sec + (timeout / 1000);
    ts_timeout.tv_nsec = (now.tv_usec + (timeout % 1000) * 1000) * 1000;
#endif

    /* Wrap the second if needed */
    if (ts_timeout.tv_nsec >= 1000000000) {
        ts_timeout.tv_sec += 1;
        ts_timeout.tv_nsec -= 1000000000;
    }

    /* Wait. */
    do {
        retval = sem_timedwait(&sem->sem, &ts_timeout);
    } while (retval < 0 && errno == EINTR);

    if (retval < 0) {
        if (errno == ETIMEDOUT) {
            retval = SDL_MUTEX_TIMEDOUT;
        } else {
            SDL_SetError("sem_timedwait returned an error: %s", strerror(errno));
        }
    }
#else
    end = SDL_GetTicks() + timeout;
    while ((retval = SDL_SemTryWait(sem)) == SDL_MUTEX_TIMEDOUT) {
        if (SDL_TICKS_PASSED(SDL_GetTicks(), end)) {
            break;
        }
        SDL_Delay(1);
    }
#endif /* HAVE_SEM_TIMEDWAIT */

    return retval;
}

Uint32 SDL_SemValue(SDL_sem *sem)
{
    int ret = 0;

    if (!sem) {
        SDL_InvalidParamError("sem");
        return 0;
    }

    sem_getvalue(&sem->sem, &ret);
    if (ret < 0) {
        ret = 0;
    }
    return (Uint32)ret;
}

int SDL_SemPost(SDL_sem *sem)
{
    int retval;

    if (!sem) {
        return SDL_InvalidParamError("sem");
    }

    retval = sem_post(&sem->sem);
    if (retval < 0) {
        SDL_SetError("sem_post() failed");
    }
    return retval;
}

#endif /* __MACOSX__ */
/* vi: set ts=4 sw=4 expandtab: */
