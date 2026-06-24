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

/* An implementation of semaphores using mutexes and condition variables */

#include "SDL_timer.h"
#include "SDL_thread.h"
#include "SDL_systhread_c.h"

#ifdef SDL_THREADS_DISABLED

SDL_sem *SDL_CreateSemaphore(Uint32 initial_value)
{
    SDL_SetError("SDL not built with thread support");
    return (SDL_sem *)0;
}

void SDL_DestroySemaphore(SDL_sem *sem)
{
}

int SDL_SemTryWait(SDL_sem *sem)
{
    return SDL_SetError("SDL not built with thread support");
}

int SDL_SemWaitTimeout(SDL_sem *sem, Uint32 timeout)
{
    return SDL_SetError("SDL not built with thread support");
}

int SDL_SemWait(SDL_sem *sem)
{
    return SDL_SetError("SDL not built with thread support");
}

Uint32 SDL_SemValue(SDL_sem *sem)
{
    return 0;
}

int SDL_SemPost(SDL_sem *sem)
{
    return SDL_SetError("SDL not built with thread support");
}

#else

struct SDL_semaphore
{
    Uint32 count;
    Uint32 waiters_count;
    SDL_mutex *count_lock;
    SDL_cond *count_nonzero;
};

SDL_sem *SDL_CreateSemaphore(Uint32 initial_value)
{
    SDL_sem *sem;

    sem = (SDL_sem *)SDL_malloc(sizeof(*sem));
    if (!sem) {
        SDL_OutOfMemory();
        return NULL;
    }
    sem->count = initial_value;
    sem->waiters_count = 0;

    sem->count_lock = SDL_CreateMutex();
    sem->count_nonzero = SDL_CreateCond();
    if (!sem->count_lock || !sem->count_nonzero) {
        SDL_DestroySemaphore(sem);
        return NULL;
    }

    return sem;
}

/* WARNING:
   You cannot call this function when another thread is using the semaphore.
*/
void SDL_DestroySemaphore(SDL_sem *sem)
{
    if (sem) {
        sem->count = 0xFFFFFFFF;
        while (sem->waiters_count > 0) {
            SDL_CondSignal(sem->count_nonzero);
            SDL_Delay(10);
        }
        SDL_DestroyCond(sem->count_nonzero);
        if (sem->count_lock) {
            SDL_LockMutex(sem->count_lock);
            SDL_UnlockMutex(sem->count_lock);
            SDL_DestroyMutex(sem->count_lock);
        }
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
    SDL_LockMutex(sem->count_lock);
    if (sem->count > 0) {
        --sem->count;
        retval = 0;
    }
    SDL_UnlockMutex(sem->count_lock);

    return retval;
}

int SDL_SemWaitTimeout(SDL_sem *sem, Uint32 timeout)
{
    int retval;

    if (!sem) {
        return SDL_InvalidParamError("sem");
    }

    /* A timeout of 0 is an easy case */
    if (timeout == 0) {
        return SDL_SemTryWait(sem);
    }

    SDL_LockMutex(sem->count_lock);
    ++sem->waiters_count;
    retval = 0;
    while ((sem->count == 0) && (retval != SDL_MUTEX_TIMEDOUT)) {
        retval = SDL_CondWaitTimeout(sem->count_nonzero,
                                     sem->count_lock, timeout);
    }
    --sem->waiters_count;
    if (retval == 0) {
        --sem->count;
    }
    SDL_UnlockMutex(sem->count_lock);

    return retval;
}

int SDL_SemWait(SDL_sem *sem)
{
    return SDL_SemWaitTimeout(sem, SDL_MUTEX_MAXWAIT);
}

Uint32 SDL_SemValue(SDL_sem *sem)
{
    Uint32 value;

    value = 0;
    if (sem) {
        SDL_LockMutex(sem->count_lock);
        value = sem->count;
        SDL_UnlockMutex(sem->count_lock);
    }
    return value;
}

int SDL_SemPost(SDL_sem *sem)
{
    if (!sem) {
        return SDL_InvalidParamError("sem");
    }

    SDL_LockMutex(sem->count_lock);
    if (sem->waiters_count > 0) {
        SDL_CondSignal(sem->count_nonzero);
    }
    ++sem->count;
    SDL_UnlockMutex(sem->count_lock);

    return 0;
}

#endif /* SDL_THREADS_DISABLED */
/* vi: set ts=4 sw=4 expandtab: */
