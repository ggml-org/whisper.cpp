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

#ifdef SDL_THREAD_PSP

/* Semaphore functions for the PSP. */

#include <stdio.h>
#include <stdlib.h>

#include "SDL_error.h"
#include "SDL_thread.h"

#include <pspthreadman.h>
#include <pspkerror.h>

struct SDL_semaphore
{
    SceUID semid;
};

/* Create a semaphore */
SDL_sem *SDL_CreateSemaphore(Uint32 initial_value)
{
    SDL_sem *sem;

    sem = (SDL_sem *)SDL_malloc(sizeof(*sem));
    if (sem) {
        /* TODO: Figure out the limit on the maximum value. */
        sem->semid = sceKernelCreateSema("SDL sema", 0, initial_value, 255, NULL);
        if (sem->semid < 0) {
            SDL_SetError("Couldn't create semaphore");
            SDL_free(sem);
            sem = NULL;
        }
    } else {
        SDL_OutOfMemory();
    }

    return sem;
}

/* Free the semaphore */
void SDL_DestroySemaphore(SDL_sem *sem)
{
    if (sem) {
        if (sem->semid > 0) {
            sceKernelDeleteSema(sem->semid);
            sem->semid = 0;
        }

        SDL_free(sem);
    }
}

/* TODO: This routine is a bit overloaded.
 * If the timeout is 0 then just poll the semaphore; if it's SDL_MUTEX_MAXWAIT, pass
 * NULL to sceKernelWaitSema() so that it waits indefinitely; and if the timeout
 * is specified, convert it to microseconds. */
int SDL_SemWaitTimeout(SDL_sem *sem, Uint32 timeout)
{
    Uint32 *pTimeout;
    int res;

    if (!sem) {
        return SDL_InvalidParamError("sem");
    }

    if (timeout == 0) {
        res = sceKernelPollSema(sem->semid, 1);
        if (res < 0) {
            return SDL_MUTEX_TIMEDOUT;
        }
        return 0;
    }

    if (timeout == SDL_MUTEX_MAXWAIT) {
        pTimeout = NULL;
    } else {
        timeout *= 1000; /* Convert to microseconds. */
        pTimeout = &timeout;
    }

    res = sceKernelWaitSema(sem->semid, 1, (SceUInt *)pTimeout);
    switch (res) {
    case SCE_KERNEL_ERROR_OK:
        return 0;
    case SCE_KERNEL_ERROR_WAIT_TIMEOUT:
        return SDL_MUTEX_TIMEDOUT;
    default:
        return SDL_SetError("sceKernelWaitSema() failed");
    }
}

int SDL_SemTryWait(SDL_sem *sem)
{
    return SDL_SemWaitTimeout(sem, 0);
}

int SDL_SemWait(SDL_sem *sem)
{
    return SDL_SemWaitTimeout(sem, SDL_MUTEX_MAXWAIT);
}

/* Returns the current count of the semaphore */
Uint32 SDL_SemValue(SDL_sem *sem)
{
    SceKernelSemaInfo info;

    if (!sem) {
        SDL_InvalidParamError("sem");
        return 0;
    }

    if (sceKernelReferSemaStatus(sem->semid, &info) >= 0) {
        return info.currentCount;
    }

    return 0;
}

int SDL_SemPost(SDL_sem *sem)
{
    int res;

    if (!sem) {
        return SDL_InvalidParamError("sem");
    }

    res = sceKernelSignalSema(sem->semid, 1);
    if (res < 0) {
        return SDL_SetError("sceKernelSignalSema() failed");
    }

    return 0;
}

#endif /* SDL_THREAD_PSP */

/* vim: ts=4 sw=4
 */
