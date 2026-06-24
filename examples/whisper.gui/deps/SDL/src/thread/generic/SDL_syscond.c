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

/* An implementation of condition variables using semaphores and mutexes */
/*
   This implementation borrows heavily from the BeOS condition variable
   implementation, written by Christopher Tate and Owen Smith.  Thanks!
 */

#include "SDL_thread.h"

#include "../generic/SDL_syscond_c.h"

/* If two implementations are to be compiled into SDL (the active one
 * will be chosen at runtime), the function names need to be
 * suffixed
 */
#ifndef SDL_THREAD_GENERIC_COND_SUFFIX
#define SDL_CreateCond_generic      SDL_CreateCond
#define SDL_DestroyCond_generic     SDL_DestroyCond
#define SDL_CondSignal_generic      SDL_CondSignal
#define SDL_CondBroadcast_generic   SDL_CondBroadcast
#define SDL_CondWait_generic        SDL_CondWait
#define SDL_CondWaitTimeout_generic SDL_CondWaitTimeout
#endif

typedef struct SDL_cond_generic
{
    SDL_mutex *lock;
    int waiting;
    int signals;
    SDL_sem *wait_sem;
    SDL_sem *wait_done;
} SDL_cond_generic;

/* Create a condition variable */
SDL_cond *SDL_CreateCond_generic(void)
{
    SDL_cond_generic *cond;

    cond = (SDL_cond_generic *)SDL_malloc(sizeof(SDL_cond_generic));
    if (cond) {
        cond->lock = SDL_CreateMutex();
        cond->wait_sem = SDL_CreateSemaphore(0);
        cond->wait_done = SDL_CreateSemaphore(0);
        cond->waiting = cond->signals = 0;
        if (!cond->lock || !cond->wait_sem || !cond->wait_done) {
            SDL_DestroyCond_generic((SDL_cond *)cond);
            cond = NULL;
        }
    } else {
        SDL_OutOfMemory();
    }
    return (SDL_cond *)cond;
}

/* Destroy a condition variable */
void SDL_DestroyCond_generic(SDL_cond *_cond)
{
    SDL_cond_generic *cond = (SDL_cond_generic *)_cond;
    if (cond) {
        if (cond->wait_sem) {
            SDL_DestroySemaphore(cond->wait_sem);
        }
        if (cond->wait_done) {
            SDL_DestroySemaphore(cond->wait_done);
        }
        if (cond->lock) {
            SDL_DestroyMutex(cond->lock);
        }
        SDL_free(cond);
    }
}

/* Restart one of the threads that are waiting on the condition variable */
int SDL_CondSignal_generic(SDL_cond *_cond)
{
    SDL_cond_generic *cond = (SDL_cond_generic *)_cond;
    if (!cond) {
        return SDL_InvalidParamError("cond");
    }

    /* If there are waiting threads not already signalled, then
       signal the condition and wait for the thread to respond.
     */
    SDL_LockMutex(cond->lock);
    if (cond->waiting > cond->signals) {
        ++cond->signals;
        SDL_SemPost(cond->wait_sem);
        SDL_UnlockMutex(cond->lock);
        SDL_SemWait(cond->wait_done);
    } else {
        SDL_UnlockMutex(cond->lock);
    }

    return 0;
}

/* Restart all threads that are waiting on the condition variable */
int SDL_CondBroadcast_generic(SDL_cond *_cond)
{
    SDL_cond_generic *cond = (SDL_cond_generic *)_cond;
    if (!cond) {
        return SDL_InvalidParamError("cond");
    }

    /* If there are waiting threads not already signalled, then
       signal the condition and wait for the thread to respond.
     */
    SDL_LockMutex(cond->lock);
    if (cond->waiting > cond->signals) {
        int i, num_waiting;

        num_waiting = (cond->waiting - cond->signals);
        cond->signals = cond->waiting;
        for (i = 0; i < num_waiting; ++i) {
            SDL_SemPost(cond->wait_sem);
        }
        /* Now all released threads are blocked here, waiting for us.
           Collect them all (and win fabulous prizes!) :-)
         */
        SDL_UnlockMutex(cond->lock);
        for (i = 0; i < num_waiting; ++i) {
            SDL_SemWait(cond->wait_done);
        }
    } else {
        SDL_UnlockMutex(cond->lock);
    }

    return 0;
}

/* Wait on the condition variable for at most 'ms' milliseconds.
   The mutex must be locked before entering this function!
   The mutex is unlocked during the wait, and locked again after the wait.

Typical use:

Thread A:
    SDL_LockMutex(lock);
    while ( ! condition ) {
        SDL_CondWait(cond, lock);
    }
    SDL_UnlockMutex(lock);

Thread B:
    SDL_LockMutex(lock);
    ...
    condition = true;
    ...
    SDL_CondSignal(cond);
    SDL_UnlockMutex(lock);
 */
int SDL_CondWaitTimeout_generic(SDL_cond *_cond, SDL_mutex *mutex, Uint32 ms)
{
    SDL_cond_generic *cond = (SDL_cond_generic *)_cond;
    int retval;

    if (!cond) {
        return SDL_InvalidParamError("cond");
    }

    /* Obtain the protection mutex, and increment the number of waiters.
       This allows the signal mechanism to only perform a signal if there
       are waiting threads.
     */
    SDL_LockMutex(cond->lock);
    ++cond->waiting;
    SDL_UnlockMutex(cond->lock);

    /* Unlock the mutex, as is required by condition variable semantics */
    SDL_UnlockMutex(mutex);

    /* Wait for a signal */
    if (ms == SDL_MUTEX_MAXWAIT) {
        retval = SDL_SemWait(cond->wait_sem);
    } else {
        retval = SDL_SemWaitTimeout(cond->wait_sem, ms);
    }

    /* Let the signaler know we have completed the wait, otherwise
       the signaler can race ahead and get the condition semaphore
       if we are stopped between the mutex unlock and semaphore wait,
       giving a deadlock.  See the following URL for details:
       http://web.archive.org/web/20010914175514/http://www-classic.be.com/aboutbe/benewsletter/volume_III/Issue40.html#Workshop
     */
    SDL_LockMutex(cond->lock);
    if (cond->signals > 0) {
        /* If we timed out, we need to eat a condition signal */
        if (retval > 0) {
            SDL_SemWait(cond->wait_sem);
        }
        /* We always notify the signal thread that we are done */
        SDL_SemPost(cond->wait_done);

        /* Signal handshake complete */
        --cond->signals;
    }
    --cond->waiting;
    SDL_UnlockMutex(cond->lock);

    /* Lock the mutex, as is required by condition variable semantics */
    SDL_LockMutex(mutex);

    return retval;
}

/* Wait on the condition variable forever */
int SDL_CondWait_generic(SDL_cond *cond, SDL_mutex *mutex)
{
    return SDL_CondWaitTimeout_generic(cond, mutex, SDL_MUTEX_MAXWAIT);
}

/* vi: set ts=4 sw=4 expandtab: */
