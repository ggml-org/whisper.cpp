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

#ifdef SDL_THREAD_OS2

/* An implementation of semaphores for OS/2 */

#include "SDL_thread.h"
#include "../../core/os2/SDL_os2.h"

#define INCL_DOSSEMAPHORES
#define INCL_DOSERRORS
#define INCL_DOSMISC
#include <os2.h>

struct SDL_semaphore {
    HEV     hEv;
    HMTX    hMtx;
    ULONG   cPost;
};


SDL_sem *SDL_CreateSemaphore(Uint32 initial_value)
{
    ULONG ulRC;
    SDL_sem *pSDLSem = SDL_malloc(sizeof(SDL_sem));

    if (!pSDLSem) {
        SDL_OutOfMemory();
        return NULL;
    }

    ulRC = DosCreateEventSem(NULL, &pSDLSem->hEv, DCE_AUTORESET, FALSE);
    if (ulRC != NO_ERROR) {
        debug_os2("DosCreateEventSem(), rc = %u", ulRC);
        SDL_free(pSDLSem);
        return NULL;
    }

    ulRC = DosCreateMutexSem(NULL, &pSDLSem->hMtx, 0, FALSE);
    if (ulRC != NO_ERROR) {
        debug_os2("DosCreateMutexSem(), rc = %u", ulRC);
        DosCloseEventSem(pSDLSem->hEv);
        SDL_free(pSDLSem);
        return NULL;
    }

    pSDLSem->cPost = initial_value;

    return pSDLSem;
}

void SDL_DestroySemaphore(SDL_sem * sem)
{
    if (!sem) return;

    DosCloseMutexSem(sem->hMtx);
    DosCloseEventSem(sem->hEv);
    SDL_free(sem);
}

int SDL_SemWaitTimeout(SDL_sem * sem, Uint32 timeout)
{
    ULONG ulRC;
    ULONG ulStartTime, ulCurTime;
    ULONG ulTimeout;
    ULONG cPost;

    if (!sem)
        return SDL_InvalidParamError("sem");

    if (timeout != SEM_INDEFINITE_WAIT)
        DosQuerySysInfo(QSV_MS_COUNT, QSV_MS_COUNT, &ulStartTime, sizeof(ULONG));

    while (TRUE) {
        ulRC = DosRequestMutexSem(sem->hMtx, SEM_INDEFINITE_WAIT);
        if (ulRC != NO_ERROR)
            return SDL_SetError("DosRequestMutexSem() failed, rc = %u", ulRC);

        cPost = sem->cPost;
        if (sem->cPost != 0)
            sem->cPost--;

        DosReleaseMutexSem(sem->hMtx);

        if (cPost != 0)
            break;

        if (timeout == SEM_INDEFINITE_WAIT)
            ulTimeout = SEM_INDEFINITE_WAIT;
        else {
            DosQuerySysInfo(QSV_MS_COUNT, QSV_MS_COUNT, &ulCurTime, sizeof(ULONG));
            ulTimeout = ulCurTime - ulStartTime;
            if (timeout < ulTimeout)
                return SDL_MUTEX_TIMEDOUT;
            ulTimeout = timeout - ulTimeout;
        }

        ulRC = DosWaitEventSem(sem->hEv, ulTimeout);
        if (ulRC == ERROR_TIMEOUT)
            return SDL_MUTEX_TIMEDOUT;

        if (ulRC != NO_ERROR)
            return SDL_SetError("DosWaitEventSem() failed, rc = %u", ulRC);
    }

    return 0;
}

int SDL_SemTryWait(SDL_sem * sem)
{
    return SDL_SemWaitTimeout(sem, 0);
}

int SDL_SemWait(SDL_sem * sem)
{
    return SDL_SemWaitTimeout(sem, SDL_MUTEX_MAXWAIT);
}

Uint32 SDL_SemValue(SDL_sem * sem)
{
    ULONG ulRC;

    if (!sem) {
        SDL_InvalidParamError("sem");
        return 0;
    }

    ulRC = DosRequestMutexSem(sem->hMtx, SEM_INDEFINITE_WAIT);
    if (ulRC != NO_ERROR)
        return SDL_SetError("DosRequestMutexSem() failed, rc = %u", ulRC);

    ulRC = sem->cPost;
    DosReleaseMutexSem(sem->hMtx);

    return ulRC;
}

int SDL_SemPost(SDL_sem * sem)
{
    ULONG ulRC;

    if (!sem)
        return SDL_InvalidParamError("sem");

    ulRC = DosRequestMutexSem(sem->hMtx, SEM_INDEFINITE_WAIT);
    if (ulRC != NO_ERROR)
        return SDL_SetError("DosRequestMutexSem() failed, rc = %u", ulRC);

    sem->cPost++;

    ulRC = DosPostEventSem(sem->hEv);
    if (ulRC != NO_ERROR && ulRC != ERROR_ALREADY_POSTED) {
        debug_os2("DosPostEventSem() failed, rc = %u", ulRC);
    }

    DosReleaseMutexSem(sem->hMtx);

    return 0;
}

#endif /* SDL_THREAD_OS2 */

/* vi: set ts=4 sw=4 expandtab: */
