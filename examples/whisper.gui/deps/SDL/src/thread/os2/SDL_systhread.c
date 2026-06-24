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

/* Thread management routines for SDL */

#include "SDL_thread.h"
#include "../SDL_systhread.h"
#include "../SDL_thread_c.h"
#include "../SDL_systhread.h"
#include "SDL_systls_c.h"
#include "../../core/os2/SDL_os2.h"
#ifndef SDL_PASSED_BEGINTHREAD_ENDTHREAD
#error This source only adjusted for SDL_PASSED_BEGINTHREAD_ENDTHREAD
#endif

#define INCL_DOSPROCESS
#define INCL_DOSERRORS
#include <os2.h>
#include <process.h>


static void RunThread(void *data)
{
    SDL_Thread *thread = (SDL_Thread *) data;
    pfnSDL_CurrentEndThread pfnEndThread = (pfnSDL_CurrentEndThread) thread->endfunc;

    if (ppSDLTLSData)
        *ppSDLTLSData = NULL;

    SDL_RunThread(thread);

    if (pfnEndThread != NULL)
        pfnEndThread();
}

int SDL_SYS_CreateThread(SDL_Thread * thread,
                     pfnSDL_CurrentBeginThread pfnBeginThread,
                     pfnSDL_CurrentEndThread pfnEndThread)
{
    if (thread->stacksize == 0)
        thread->stacksize = 65536;

    if (pfnBeginThread) {
        /* Save the function which we will have to call to clear the RTL of calling app! */
        thread->endfunc = pfnEndThread;
        /* Start the thread using the runtime library of calling app! */
        thread->handle = (SYS_ThreadHandle)
                            pfnBeginThread(RunThread, NULL, thread->stacksize, thread);
    } else {
        thread->endfunc = _endthread;
        thread->handle = (SYS_ThreadHandle)
                            _beginthread(RunThread, NULL, thread->stacksize, thread);
    }

    if (thread->handle == -1)
        return SDL_SetError("Not enough resources to create thread");

    return 0;
}

void SDL_SYS_SetupThread(const char *name)
{
    /* nothing. */
}

SDL_threadID SDL_ThreadID(void)
{
    PTIB  tib;
    PPIB  pib;

    DosGetInfoBlocks(&tib, &pib);
    return tib->tib_ptib2->tib2_ultid;
}

int SDL_SYS_SetThreadPriority(SDL_ThreadPriority priority)
{
    ULONG ulRC;

    ulRC = DosSetPriority(PRTYS_THREAD,
                          (priority < SDL_THREAD_PRIORITY_NORMAL)? PRTYC_IDLETIME :
                           (priority > SDL_THREAD_PRIORITY_NORMAL)? PRTYC_TIMECRITICAL :
                            PRTYC_REGULAR,
                          0, 0);
    if (ulRC != NO_ERROR)
        return SDL_SetError("DosSetPriority() failed, rc = %u", ulRC);

    return 0;
}

void SDL_SYS_WaitThread(SDL_Thread * thread)
{
    ULONG ulRC = DosWaitThread((PTID)&thread->handle, DCWW_WAIT);

    if (ulRC != NO_ERROR) {
        debug_os2("DosWaitThread() failed, rc = %u", ulRC);
    }
}

void SDL_SYS_DetachThread(SDL_Thread * thread)
{
    /* nothing. */
}

#endif /* SDL_THREAD_OS2 */

/* vi: set ts=4 sw=4 expandtab: */
