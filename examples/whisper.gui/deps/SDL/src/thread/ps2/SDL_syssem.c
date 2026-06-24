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

#ifdef SDL_THREAD_PS2

/* Semaphore functions for the PS2. */

#include <stdio.h>
#include <stdlib.h>
#include <kernel_util.h>

#include "SDL_error.h"
#include "SDL_thread.h"

#include <kernel.h>

struct SDL_semaphore
{
    s32 semid;
};

/* Create a semaphore */
SDL_sem *SDL_CreateSemaphore(Uint32 initial_value)
{
    SDL_sem *sem;
    ee_sema_t sema;

    sem = (SDL_sem *)SDL_malloc(sizeof(*sem));
    if (sem) {
        /* TODO: Figure out the limit on the maximum value. */
        sema.init_count = initial_value;
        sema.max_count = 255;
        sema.option = 0;
        sem->semid = CreateSema(&sema);

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
            DeleteSema(sem->semid);
            sem->semid = 0;
        }

        SDL_free(sem);
    }
}

int SDL_SemWaitTimeout(SDL_sem *sem, Uint32 timeout)
{
    int ret;
    u64 timeout_usec;
    u64 *timeout_ptr;

    if (!sem) {
        return SDL_InvalidParamError("sem");
    }

    if (timeout == 0) {
        if (PollSema(sem->semid) < 0) {
            return SDL_MUTEX_TIMEDOUT;
        }
        return 0;
    }

    timeout_ptr = NULL;

    if (timeout != SDL_MUTEX_MAXWAIT) {
        timeout_usec = timeout * 1000;
        timeout_ptr = &timeout_usec;
    }

    ret = WaitSemaEx(sem->semid, 1, timeout_ptr);

    if (ret < 0) {
        return SDL_MUTEX_TIMEDOUT;
    }
    return 0; // Wait condition satisfied.
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
    ee_sema_t info;

    if (!sem) {
        SDL_InvalidParamError("sem");
        return 0;
    }

    if (ReferSemaStatus(sem->semid, &info) >= 0) {
        return info.count;
    }

    return 0;
}

int SDL_SemPost(SDL_sem *sem)
{
    int res;

    if (!sem) {
        return SDL_InvalidParamError("sem");
    }

    res = SignalSema(sem->semid);
    if (res < 0) {
        return SDL_SetError("sceKernelSignalSema() failed");
    }

    return 0;
}

#endif /* SDL_THREAD_PS2 */

/* vim: ts=4 sw=4
 */
