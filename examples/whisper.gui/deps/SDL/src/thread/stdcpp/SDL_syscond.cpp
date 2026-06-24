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
}

#include <chrono>
#include <condition_variable>
#include <ratio>
#include <system_error>

#include "SDL_sysmutex_c.h"

struct SDL_cond
{
    std::condition_variable_any cpp_cond;
};

/* Create a condition variable */
extern "C" SDL_cond *
SDL_CreateCond(void)
{
    /* Allocate and initialize the condition variable */
    try {
        SDL_cond *cond = new SDL_cond;
        return cond;
    } catch (std::system_error &ex) {
        SDL_SetError("unable to create a C++ condition variable: code=%d; %s", ex.code(), ex.what());
        return NULL;
    } catch (std::bad_alloc &) {
        SDL_OutOfMemory();
        return NULL;
    }
}

/* Destroy a condition variable */
extern "C" void
SDL_DestroyCond(SDL_cond *cond)
{
    if (cond) {
        delete cond;
    }
}

/* Restart one of the threads that are waiting on the condition variable */
extern "C" int
SDL_CondSignal(SDL_cond *cond)
{
    if (!cond) {
        return SDL_InvalidParamError("cond");
    }

    cond->cpp_cond.notify_one();
    return 0;
}

/* Restart all threads that are waiting on the condition variable */
extern "C" int
SDL_CondBroadcast(SDL_cond *cond)
{
    if (!cond) {
        return SDL_InvalidParamError("cond");
    }

    cond->cpp_cond.notify_all();
    return 0;
}

/* Wait on the condition variable for at most 'ms' milliseconds.
   The mutex must be locked before entering this function!
   The mutex is unlocked during the wait, and locked again after the wait.

Typical use:

Thread A:
    SDL_LockMutex(lock);
    while ( ! condition ) {
        SDL_CondWait(cond, lock);
    }
    SDL_UnlockMutex(lock);

Thread B:
    SDL_LockMutex(lock);
    ...
    condition = true;
    ...
    SDL_CondSignal(cond);
    SDL_UnlockMutex(lock);
 */
extern "C" int
SDL_CondWaitTimeout(SDL_cond *cond, SDL_mutex *mutex, Uint32 ms)
{
    if (cond == NULL) {
        return SDL_InvalidParamError("cond");
    }

    if (mutex == NULL) {
        return SDL_InvalidParamError("mutex");
    }

    try {
        std::unique_lock<std::recursive_mutex> cpp_lock(mutex->cpp_mutex, std::adopt_lock_t());
        if (ms == SDL_MUTEX_MAXWAIT) {
            cond->cpp_cond.wait(
                cpp_lock);
            cpp_lock.release();
            return 0;
        } else {
            auto wait_result = cond->cpp_cond.wait_for(
                cpp_lock,
                std::chrono::duration<Uint32, std::milli>(ms));
            cpp_lock.release();
            if (wait_result == std::cv_status::timeout) {
                return SDL_MUTEX_TIMEDOUT;
            } else {
                return 0;
            }
        }
    } catch (std::system_error &ex) {
        return SDL_SetError("unable to wait on a C++ condition variable: code=%d; %s", ex.code(), ex.what());
    }
}

/* Wait on the condition variable forever */
extern "C" int
SDL_CondWait(SDL_cond *cond, SDL_mutex *mutex)
{
    return SDL_CondWaitTimeout(cond, mutex, SDL_MUTEX_MAXWAIT);
}

/* vi: set ts=4 sw=4 expandtab: */
