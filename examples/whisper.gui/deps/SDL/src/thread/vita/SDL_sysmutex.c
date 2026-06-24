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

#ifdef SDL_THREAD_VITA

#include "SDL_thread.h"
#include "SDL_systhread_c.h"

#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/error.h>

struct SDL_mutex
{
    SceKernelLwMutexWork lock;
};

/* Create a mutex */
SDL_mutex *SDL_CreateMutex(void)
{
    SDL_mutex *mutex = NULL;
    SceInt32 res = 0;

    /* Allocate mutex memory */
    mutex = (SDL_mutex *)SDL_malloc(sizeof(*mutex));
    if (mutex) {

        res = sceKernelCreateLwMutex(
            &mutex->lock,
            "SDL mutex",
            SCE_KERNEL_MUTEX_ATTR_RECURSIVE,
            0,
            NULL);

        if (res < 0) {
            SDL_SetError("Error trying to create mutex: %x", res);
        }
    } else {
        SDL_OutOfMemory();
    }
    return mutex;
}

/* Free the mutex */
void SDL_DestroyMutex(SDL_mutex *mutex)
{
    if (mutex) {
        sceKernelDeleteLwMutex(&mutex->lock);
        SDL_free(mutex);
    }
}

/* Lock the mutex */
int SDL_LockMutex(SDL_mutex *mutex) SDL_NO_THREAD_SAFETY_ANALYSIS /* clang doesn't know about NULL mutexes */
{
#ifdef SDL_THREADS_DISABLED
    return 0;
#else
    SceInt32 res = 0;

    if (!mutex) {
        return 0;
    }

    res = sceKernelLockLwMutex(&mutex->lock, 1, NULL);
    if (res != SCE_KERNEL_OK) {
        return SDL_SetError("Error trying to lock mutex: %x", res);
    }

    return 0;
#endif /* SDL_THREADS_DISABLED */
}

/* Try to lock the mutex */
int SDL_TryLockMutex(SDL_mutex *mutex)
{
#ifdef SDL_THREADS_DISABLED
    return 0;
#else
    SceInt32 res = 0;

    if (!mutex) {
        return 0;
    }

    res = sceKernelTryLockLwMutex(&mutex->lock, 1);
    switch (res) {
    case SCE_KERNEL_OK:
        return 0;
        break;
    case SCE_KERNEL_ERROR_MUTEX_FAILED_TO_OWN:
        return SDL_MUTEX_TIMEDOUT;
        break;
    default:
        return SDL_SetError("Error trying to lock mutex: %x", res);
        break;
    }

    return -1;
#endif /* SDL_THREADS_DISABLED */
}

/* Unlock the mutex */
int SDL_UnlockMutex(SDL_mutex *mutex) SDL_NO_THREAD_SAFETY_ANALYSIS /* clang doesn't know about NULL mutexes */
{
#ifdef SDL_THREADS_DISABLED
    return 0;
#else
    SceInt32 res = 0;

    if (mutex == NULL) {
        return 0;
    }

    res = sceKernelUnlockLwMutex(&mutex->lock, 1);
    if (res != 0) {
        return SDL_SetError("Error trying to unlock mutex: %x", res);
    }

    return 0;
#endif /* SDL_THREADS_DISABLED */
}

#endif /* SDL_THREAD_VITA */

/* vi: set ts=4 sw=4 expandtab: */
