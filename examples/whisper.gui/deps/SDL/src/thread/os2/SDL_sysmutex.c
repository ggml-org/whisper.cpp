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

/* An implementation of mutexes for OS/2 */

#include "SDL_thread.h"
#include "SDL_systhread_c.h"
#include "../../core/os2/SDL_os2.h"

#define INCL_DOSSEMAPHORES
#define INCL_DOSERRORS
#include <os2.h>

struct SDL_mutex {
    HMTX  _handle;
};

/* Create a mutex */
SDL_mutex *SDL_CreateMutex(void)
{
    ULONG ulRC;
    HMTX  hMtx;

    ulRC = DosCreateMutexSem(NULL, &hMtx, 0, FALSE);
    if (ulRC != NO_ERROR) {
        debug_os2("DosCreateMutexSem(), rc = %u", ulRC);
        return NULL;
    }

    return (SDL_mutex *)hMtx;
}

/* Free the mutex */
void SDL_DestroyMutex(SDL_mutex * mutex)
{
    HMTX  hMtx = (HMTX)mutex;
    if (hMtx != NULLHANDLE) {
        const ULONG ulRC = DosCloseMutexSem(hMtx);
        if (ulRC != NO_ERROR) {
            debug_os2("DosCloseMutexSem(), rc = %u", ulRC);
        }
    }
}

/* Lock the mutex */
int SDL_LockMutex(SDL_mutex * mutex) SDL_NO_THREAD_SAFETY_ANALYSIS /* clang doesn't know about NULL mutexes */
{
    ULONG ulRC;
    HMTX  hMtx = (HMTX)mutex;

    if (hMtx == NULLHANDLE)
        return 0;

    ulRC = DosRequestMutexSem(hMtx, SEM_INDEFINITE_WAIT);
    if (ulRC != NO_ERROR) {
      debug_os2("DosRequestMutexSem(), rc = %u", ulRC);
      return -1;
    }

    return 0;
}

/* try Lock the mutex */
int SDL_TryLockMutex(SDL_mutex * mutex)
{
    ULONG ulRC;
    HMTX  hMtx = (HMTX)mutex;

    if (hMtx == NULLHANDLE)
        return 0;

    ulRC = DosRequestMutexSem(hMtx, SEM_IMMEDIATE_RETURN);

    if (ulRC == ERROR_TIMEOUT)
        return SDL_MUTEX_TIMEDOUT;

    if (ulRC != NO_ERROR) {
        debug_os2("DosRequestMutexSem(), rc = %u", ulRC);
        return -1;
    }

    return 0;
}

/* Unlock the mutex */
int SDL_UnlockMutex(SDL_mutex * mutex) SDL_NO_THREAD_SAFETY_ANALYSIS /* clang doesn't know about NULL mutexes */
{
    ULONG ulRC;
    HMTX  hMtx = (HMTX)mutex;

    if (hMtx == NULLHANDLE)
        return 0;

    ulRC = DosReleaseMutexSem(hMtx);
    if (ulRC != NO_ERROR)
        return SDL_SetError("DosReleaseMutexSem(), rc = %u", ulRC);

    return 0;
}

#endif /* SDL_THREAD_OS2 */

/* vi: set ts=4 sw=4 expandtab: */
