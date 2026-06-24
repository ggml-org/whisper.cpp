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

extern "C" {
#include "SDL_thread.h"
#include "SDL_systhread_c.h"
}

#include <system_error>

#include "SDL_sysmutex_c.h"
#include <Windows.h>

/* Create a mutex */
extern "C" SDL_mutex *
SDL_CreateMutex(void)
{
    /* Allocate and initialize the mutex */
    try {
        SDL_mutex *mutex = new SDL_mutex;
        return mutex;
    } catch (std::system_error &ex) {
        SDL_SetError("unable to create a C++ mutex: code=%d; %s", ex.code(), ex.what());
        return NULL;
    } catch (std::bad_alloc &) {
        SDL_OutOfMemory();
        return NULL;
    }
}

/* Free the mutex */
extern "C" void
SDL_DestroyMutex(SDL_mutex *mutex)
{
    if (mutex) {
        delete mutex;
    }
}

/* Lock the mutex */
extern "C" int
SDL_LockMutex(SDL_mutex *mutex) SDL_NO_THREAD_SAFETY_ANALYSIS /* clang doesn't know about NULL mutexes */
{
    if (mutex == NULL) {
        return 0;
    }

    try {
        mutex->cpp_mutex.lock();
        return 0;
    } catch (std::system_error &ex) {
        return SDL_SetError("unable to lock a C++ mutex: code=%d; %s", ex.code(), ex.what());
    }
}

/* TryLock the mutex */
int SDL_TryLockMutex(SDL_mutex *mutex)
{
    int retval = 0;

    if (!mutex) {
        return 0;
    }

    if (mutex->cpp_mutex.try_lock() == false) {
        retval = SDL_MUTEX_TIMEDOUT;
    }
    return retval;
}

/* Unlock the mutex */
extern "C" int
SDL_UnlockMutex(SDL_mutex *mutex) SDL_NO_THREAD_SAFETY_ANALYSIS /* clang doesn't know about NULL mutexes */
{
    if (mutex == NULL) {
        return 0;
    }

    mutex->cpp_mutex.unlock();
    return 0;
}

/* vi: set ts=4 sw=4 expandtab: */
