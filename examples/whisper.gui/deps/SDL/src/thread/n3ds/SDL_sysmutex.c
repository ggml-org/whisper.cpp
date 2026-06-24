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

/* An implementation of mutexes using libctru's RecursiveLock */

#include "SDL_sysmutex_c.h"

/* Create a mutex */
SDL_mutex *SDL_CreateMutex(void)
{
    SDL_mutex *mutex;

    /* Allocate mutex memory */
    mutex = (SDL_mutex *)SDL_malloc(sizeof(*mutex));
    if (mutex) {
        RecursiveLock_Init(&mutex->lock);
    } else {
        SDL_OutOfMemory();
    }
    return mutex;
}

/* Free the mutex */
void SDL_DestroyMutex(SDL_mutex *mutex)
{
    if (mutex) {
        SDL_free(mutex);
    }
}

/* Lock the mutex */
int SDL_LockMutex(SDL_mutex *mutex) SDL_NO_THREAD_SAFETY_ANALYSIS /* clang doesn't know about NULL mutexes */
{
    if (mutex == NULL) {
        return 0;
    }

    RecursiveLock_Lock(&mutex->lock);

    return 0;
}

/* try Lock the mutex */
int SDL_TryLockMutex(SDL_mutex *mutex)
{
    if (!mutex) {
        return 0;
    }

    return RecursiveLock_TryLock(&mutex->lock);
}

/* Unlock the mutex */
int SDL_UnlockMutex(SDL_mutex *mutex) SDL_NO_THREAD_SAFETY_ANALYSIS /* clang doesn't know about NULL mutexes */
{
    if (mutex == NULL) {
        return 0;
    }

    RecursiveLock_Unlock(&mutex->lock);

    return 0;
}

#endif /* SDL_THREAD_N3DS */

/* vi: set sts=4 ts=4 sw=4 expandtab: */
