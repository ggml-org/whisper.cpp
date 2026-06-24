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

#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>

#include "SDL_thread.h"
#include "SDL_sysmutex_c.h"

struct SDL_cond
{
    pthread_cond_t cond;
};

/* Create a condition variable */
SDL_cond *SDL_CreateCond(void)
{
    SDL_cond *cond;

    cond = (SDL_cond *)SDL_malloc(sizeof(SDL_cond));
    if (cond) {
        if (pthread_cond_init(&cond->cond, NULL) != 0) {
            SDL_SetError("pthread_cond_init() failed");
            SDL_free(cond);
            cond = NULL;
        }
    }
    return cond;
}

/* Destroy a condition variable */
void SDL_DestroyCond(SDL_cond *cond)
{
    if (cond) {
        pthread_cond_destroy(&cond->cond);
        SDL_free(cond);
    }
}

/* Restart one of the threads that are waiting on the condition variable */
int SDL_CondSignal(SDL_cond *cond)
{
    int retval;

    if (!cond) {
        return SDL_InvalidParamError("cond");
    }

    retval = 0;
    if (pthread_cond_signal(&cond->cond) != 0) {
        return SDL_SetError("pthread_cond_signal() failed");
    }
    return retval;
}

/* Restart all threads that are waiting on the condition variable */
int SDL_CondBroadcast(SDL_cond *cond)
{
    int retval;

    if (!cond) {
        return SDL_InvalidParamError("cond");
    }

    retval = 0;
    if (pthread_cond_broadcast(&cond->cond) != 0) {
        return SDL_SetError("pthread_cond_broadcast() failed");
    }
    return retval;
}

int SDL_CondWaitTimeout(SDL_cond *cond, SDL_mutex *mutex, Uint32 ms)
{
    int retval;
#ifndef HAVE_CLOCK_GETTIME
    struct timeval delta;
#endif
    struct timespec abstime;

    if (!cond) {
        return SDL_InvalidParamError("cond");
    }

#ifdef HAVE_CLOCK_GETTIME
    clock_gettime(CLOCK_REALTIME, &abstime);

    abstime.tv_nsec += (ms % 1000) * 1000000;
    abstime.tv_sec += ms / 1000;
#else
    gettimeofday(&delta, NULL);

    abstime.tv_sec = delta.tv_sec + (ms / 1000);
    abstime.tv_nsec = (long)(delta.tv_usec + (ms % 1000) * 1000) * 1000;
#endif
    if (abstime.tv_nsec >= 1000000000) {
        abstime.tv_sec += 1;
        abstime.tv_nsec -= 1000000000;
    }

tryagain:
    retval = pthread_cond_timedwait(&cond->cond, &mutex->id, &abstime);
    switch (retval) {
    case EINTR:
        goto tryagain;
        /* break; -Wunreachable-code-break */
    case ETIMEDOUT:
        retval = SDL_MUTEX_TIMEDOUT;
        break;
    case 0:
        break;
    default:
        retval = SDL_SetError("pthread_cond_timedwait() failed");
    }
    return retval;
}

/* Wait on the condition variable, unlocking the provided mutex.
   The mutex must be locked before entering this function!
 */
int SDL_CondWait(SDL_cond *cond, SDL_mutex *mutex)
{
    if (!cond) {
        return SDL_InvalidParamError("cond");
    } else if (pthread_cond_wait(&cond->cond, &mutex->id) != 0) {
        return SDL_SetError("pthread_cond_wait() failed");
    }
    return 0;
}

/* vi: set ts=4 sw=4 expandtab: */
