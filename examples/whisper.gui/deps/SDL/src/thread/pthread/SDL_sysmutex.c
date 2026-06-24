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

#include "SDL_thread.h"

#if !(defined(SDL_THREAD_PTHREAD_RECURSIVE_MUTEX) || \
    defined(SDL_THREAD_PTHREAD_RECURSIVE_MUTEX_NP))
#define FAKE_RECURSIVE_MUTEX
#endif

struct SDL_mutex
{
    pthread_mutex_t id;
#ifdef FAKE_RECURSIVE_MUTEX
    int recursive;
    pthread_t owner;
#endif
};

SDL_mutex *SDL_CreateMutex(void)
{
    SDL_mutex *mutex;
    pthread_mutexattr_t attr;

    /* Allocate the structure */
    mutex = (SDL_mutex *)SDL_calloc(1, sizeof(*mutex));
    if (mutex) {
        pthread_mutexattr_init(&attr);
#ifdef SDL_THREAD_PTHREAD_RECURSIVE_MUTEX
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
#elif defined(SDL_THREAD_PTHREAD_RECURSIVE_MUTEX_NP)
        pthread_mutexattr_setkind_np(&attr, PTHREAD_MUTEX_RECURSIVE_NP);
#else
        /* No extra attributes necessary */
#endif
        if (pthread_mutex_init(&mutex->id, &attr) != 0) {
            SDL_SetError("pthread_mutex_init() failed");
            SDL_free(mutex);
            mutex = NULL;
        }
    } else {
        SDL_OutOfMemory();
    }
    return mutex;
}

void SDL_DestroyMutex(SDL_mutex *mutex)
{
    if (mutex) {
        pthread_mutex_destroy(&mutex->id);
        SDL_free(mutex);
    }
}

/* Lock the mutex */
int SDL_LockMutex(SDL_mutex *mutex) SDL_NO_THREAD_SAFETY_ANALYSIS /* clang doesn't know about NULL mutexes */
{
#ifdef FAKE_RECURSIVE_MUTEX
    pthread_t this_thread;
#endif

    if (mutex == NULL) {
        return 0;
    }

#ifdef FAKE_RECURSIVE_MUTEX
    this_thread = pthread_self();
    if (mutex->owner == this_thread) {
        ++mutex->recursive;
    } else {
        /* The order of operations is important.
           We set the locking thread id after we obtain the lock
           so unlocks from other threads will fail.
         */
        if (pthread_mutex_lock(&mutex->id) == 0) {
            mutex->owner = this_thread;
            mutex->recursive = 0;
        } else {
            return SDL_SetError("pthread_mutex_lock() failed");
        }
    }
#else
    if (pthread_mutex_lock(&mutex->id) != 0) {
        return SDL_SetError("pthread_mutex_lock() failed");
    }
#endif
    return 0;
}

int SDL_TryLockMutex(SDL_mutex *mutex)
{
    int retval;
    int result;
#ifdef FAKE_RECURSIVE_MUTEX
    pthread_t this_thread;
#endif

    if (!mutex) {
        return 0;
    }

    retval = 0;
#ifdef FAKE_RECURSIVE_MUTEX
    this_thread = pthread_self();
    if (mutex->owner == this_thread) {
        ++mutex->recursive;
    } else {
        /* The order of operations is important.
         We set the locking thread id after we obtain the lock
         so unlocks from other threads will fail.
         */
        result = pthread_mutex_trylock(&mutex->id);
        if (result == 0) {
            mutex->owner = this_thread;
            mutex->recursive = 0;
        } else if (result == EBUSY) {
            retval = SDL_MUTEX_TIMEDOUT;
        } else {
            retval = SDL_SetError("pthread_mutex_trylock() failed");
        }
    }
#else
    result = pthread_mutex_trylock(&mutex->id);
    if (result != 0) {
        if (result == EBUSY) {
            retval = SDL_MUTEX_TIMEDOUT;
        } else {
            retval = SDL_SetError("pthread_mutex_trylock() failed");
        }
    }
#endif
    return retval;
}

int SDL_UnlockMutex(SDL_mutex *mutex) SDL_NO_THREAD_SAFETY_ANALYSIS /* clang doesn't know about NULL mutexes */
{
    if (mutex == NULL) {
        return 0;
    }

#ifdef FAKE_RECURSIVE_MUTEX
    /* We can only unlock the mutex if we own it */
    if (pthread_self() == mutex->owner) {
        if (mutex->recursive) {
            --mutex->recursive;
        } else {
            /* The order of operations is important.
               First reset the owner so another thread doesn't lock
               the mutex and set the ownership before we reset it,
               then release the lock semaphore.
             */
            mutex->owner = 0;
            pthread_mutex_unlock(&mutex->id);
        }
    } else {
        return SDL_SetError("mutex not owned by this thread");
    }

#else
    if (pthread_mutex_unlock(&mutex->id) != 0) {
        return SDL_SetError("pthread_mutex_unlock() failed");
    }
#endif /* FAKE_RECURSIVE_MUTEX */

    return 0;
}

/* vi: set ts=4 sw=4 expandtab: */
