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

#ifdef SDL_THREAD_NGAGE

/* N-Gage thread management routines for SDL */

#include <e32std.h>

extern "C" {
#undef NULL
#include "SDL_error.h"
#include "SDL_thread.h"
#include "../SDL_systhread.h"
#include "../SDL_thread_c.h"
};

static int object_count;

static int RunThread(TAny *data)
{
    SDL_RunThread((SDL_Thread *)data);
    return 0;
}

static TInt NewThread(const TDesC &aName, TAny *aPtr1, TAny *aPtr2)
{
    return ((RThread *)(aPtr1))->Create(aName, RunThread, KDefaultStackSize, NULL, aPtr2);
}

int CreateUnique(TInt (*aFunc)(const TDesC &aName, TAny *, TAny *), TAny *aPtr1, TAny *aPtr2)
{
    TBuf<16> name;
    TInt status = KErrNone;
    do {
        object_count++;
        name.Format(_L("SDL_%x"), object_count);
        status = aFunc(name, aPtr1, aPtr2);
    } while (status == KErrAlreadyExists);
    return status;
}

int SDL_SYS_CreateThread(SDL_Thread *thread)
{
    RThread rthread;

    TInt status = CreateUnique(NewThread, &rthread, thread);
    if (status != KErrNone) {
        delete (RThread *)thread->handle;
        thread->handle = NULL;
        return SDL_SetError("Not enough resources to create thread");
    }

    rthread.Resume();
    thread->handle = rthread.Handle();
    return 0;
}

void SDL_SYS_SetupThread(const char *name)
{
    return;
}

SDL_threadID SDL_ThreadID(void)
{
    RThread current;
    TThreadId id = current.Id();
    return id;
}

int SDL_SYS_SetThreadPriority(SDL_ThreadPriority priority)
{
    return 0;
}

void SDL_SYS_WaitThread(SDL_Thread *thread)
{
    RThread t;
    t.Open(thread->threadid);
    if (t.ExitReason() == EExitPending) {
        TRequestStatus status;
        t.Logon(status);
        User::WaitForRequest(status);
    }
    t.Close();
}

void SDL_SYS_DetachThread(SDL_Thread *thread)
{
    return;
}

#endif /* SDL_THREAD_NGAGE */

/* vim: ts=4 sw=4
 */
