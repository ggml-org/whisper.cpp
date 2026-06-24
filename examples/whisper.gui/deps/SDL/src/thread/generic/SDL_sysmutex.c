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

/* An implementation of mutexes using semaphores */

#include "SDL_thread.h"
#include "SDL_systhread_c.h"

struct SDL_mutex
{
    int recursive;
    SDL_threadID owner;
    SDL_sem *sem;
};

/* Create a mutex */
SDL_mutex *SDL_CreateMutex(void)
{
    SDL_mutex *mutex;

    /* Allocate mutex memory */
    mutex = (SDL_mutex *)SDL_calloc(1, sizeof(*mutex));

#ifndef SDL_THREADS_DISABLED
    if (mutex) {
        /* Create the mutex semaphore, with initial value 1 */
        mutex->sem = SDL_CreateSemaphore(1);
        mutex->recursive = 0;
        mutex->owner = 0;
        if (!mutex->sem) {
            SDL_free(mutex);
            mutex = NULL;
        }
    } else {
        SDL_OutOfMemory();
    }
#endif /* !SDL_THREADS_DISABLED */

    return mutex;
}

/* Free the mutex */
void SDL_DestroyMutex(SDL_mutex *mutex)
{
    if (mutex) {
        if (mutex->sem) {
            SDL_DestroySemaphore(mutex->sem);
        }
        SDL_free(mutex);
    }
}

/* Lock the mutex */
int SDL_LockMutex(SDL_mutex *mutex) SDL_NO_THREAD_SAFETY_ANALYSIS /* clang doesn't know about NULL mutexes */
{
#ifdef SDL_THREADS_DISABLED
    return 0;
#else
    SDL_threadID this_thread;

    if (mutex == NULL) {
        return 0;
    }

    this_thread = SDL_ThreadID();
    if (mutex->owner == this_thread) {
        ++mutex->recursive;
    } else {
        /* The order of operations is important.
           We set the locking thread id after we obtain the lock
           so unlocks from other threads will fail.
         */
        SDL_SemWait(mutex->sem);
        mutex->owner = this_thread;
        mutex->recursive = 0;
    }

    return 0;
#endif /* SDL_THREADS_DISABLED */
}

/* try Lock the mutex */
int SDL_TryLockMutex(SDL_mutex *mutex)
{
#ifdef SDL_THREADS_DISABLED
    return 0;
#else
    int retval = 0;
    SDL_threadID this_thread;

    if (!mutex) {
        return 0;
    }

    this_thread = SDL_ThreadID();
    if (mutex->owner == this_thread) {
        ++mutex->recursive;
    } else {
        /* The order of operations is important.
         We set the locking thread id after we obtain the lock
         so unlocks from other threads will fail.
         */
        retval = SDL_SemWait(mutex->sem);
        if (retval == 0) {
            mutex->owner = this_thread;
            mutex->recursive = 0;
        }
    }

    return retval;
#endif /* SDL_THREADS_DISABLED */
}

/* Unlock the mutex */
int SDL_UnlockMutex(SDL_mutex *mutex) SDL_NO_THREAD_SAFETY_ANALYSIS /* clang doesn't know about NULL mutexes */
{
#ifdef SDL_THREADS_DISABLED
    return 0;
#else
    if (mutex == NULL) {
        return 0;
    }

    /* If we don't own the mutex, we can't unlock it */
    if (SDL_ThreadID() != mutex->owner) {
        return SDL_SetError("mutex not owned by this thread");
    }

    if (mutex->recursive) {
        --mutex->recursive;
    } else {
        /* The order of operations is important.
           First reset the owner so another thread doesn't lock
           the mutex and set the ownership before we reset it,
           then release the lock semaphore.
         */
        mutex->owner = 0;
        SDL_SemPost(mutex->sem);
    }
    return 0;
#endif /* SDL_THREADS_DISABLED */
}

/* vi: set ts=4 sw=4 expandtab: */
