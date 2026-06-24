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

#ifdef SDL_THREAD_WINDOWS

/**
 * Semaphore functions using the Win32 API
 * There are two implementations available based on:
 * - Kernel Semaphores. Available on all OS versions. (kern)
 *   Heavy-weight inter-process kernel objects.
 * - Atomics and WaitOnAddress API. (atom)
 *   Faster due to significantly less context switches.
 *   Requires Windows 8 or newer.
 * which are chosen at runtime.
 */

#include "../../core/windows/SDL_windows.h"

#include "SDL_hints.h"
#include "SDL_thread.h"
#include "SDL_timer.h"

typedef SDL_sem * (*pfnSDL_CreateSemaphore)(Uint32);
typedef void (*pfnSDL_DestroySemaphore)(SDL_sem *);
typedef int (*pfnSDL_SemWaitTimeout)(SDL_sem *, Uint32);
typedef int (*pfnSDL_SemTryWait)(SDL_sem *);
typedef int (*pfnSDL_SemWait)(SDL_sem *);
typedef Uint32 (*pfnSDL_SemValue)(SDL_sem *);
typedef int (*pfnSDL_SemPost)(SDL_sem *);

typedef struct SDL_semaphore_impl_t
{
    pfnSDL_CreateSemaphore Create;
    pfnSDL_DestroySemaphore Destroy;
    pfnSDL_SemWaitTimeout WaitTimeout;
    pfnSDL_SemTryWait TryWait;
    pfnSDL_SemWait Wait;
    pfnSDL_SemValue Value;
    pfnSDL_SemPost Post;
} SDL_sem_impl_t;

/* Implementation will be chosen at runtime based on available Kernel features */
static SDL_sem_impl_t SDL_sem_impl_active = { 0 };

/**
 * Atomic + WaitOnAddress implementation
 */

/* APIs not available on WinPhone 8.1 */
/* https://www.microsoft.com/en-us/download/details.aspx?id=47328 */

#if !SDL_WINAPI_FAMILY_PHONE
#ifdef __WINRT__
/* Functions are guaranteed to be available */
#define pWaitOnAddress       WaitOnAddress
#define pWakeByAddressSingle WakeByAddressSingle
#else
typedef BOOL(WINAPI *pfnWaitOnAddress)(volatile VOID *, PVOID, SIZE_T, DWORD);
typedef VOID(WINAPI *pfnWakeByAddressSingle)(PVOID);

static pfnWaitOnAddress pWaitOnAddress = NULL;
static pfnWakeByAddressSingle pWakeByAddressSingle = NULL;
#endif

typedef struct SDL_semaphore_atom
{
    LONG count;
} SDL_sem_atom;

static SDL_sem *SDL_CreateSemaphore_atom(Uint32 initial_value)
{
    SDL_sem_atom *sem;

    sem = (SDL_sem_atom *)SDL_malloc(sizeof(*sem));
    if (sem) {
        sem->count = initial_value;
    } else {
        SDL_OutOfMemory();
    }
    return (SDL_sem *)sem;
}

static void SDL_DestroySemaphore_atom(SDL_sem *sem)
{
    if (sem) {
        SDL_free(sem);
    }
}

static int SDL_SemTryWait_atom(SDL_sem *_sem)
{
    SDL_sem_atom *sem = (SDL_sem_atom *)_sem;
    LONG count;

    if (!sem) {
        return SDL_InvalidParamError("sem");
    }

    count = sem->count;
    if (count == 0) {
        return SDL_MUTEX_TIMEDOUT;
    }

    if (InterlockedCompareExchange(&sem->count, count - 1, count) == count) {
        return 0;
    }

    return SDL_MUTEX_TIMEDOUT;
}

static int SDL_SemWait_atom(SDL_sem *_sem)
{
    SDL_sem_atom *sem = (SDL_sem_atom *)_sem;
    LONG count;

    if (!sem) {
        return SDL_InvalidParamError("sem");
    }

    for (;;) {
        count = sem->count;
        while (count == 0) {
            if (pWaitOnAddress(&sem->count, &count, sizeof(sem->count), INFINITE) == FALSE) {
                return SDL_SetError("WaitOnAddress() failed");
            }
            count = sem->count;
        }

        if (InterlockedCompareExchange(&sem->count, count - 1, count) == count) {
            return 0;
        }
    }
}

static int SDL_SemWaitTimeout_atom(SDL_sem *_sem, Uint32 timeout)
{
    SDL_sem_atom *sem = (SDL_sem_atom *)_sem;
    LONG count;
    Uint32 now;
    Uint32 deadline;
    DWORD timeout_eff;

    if (timeout == SDL_MUTEX_MAXWAIT) {
        return SDL_SemWait_atom(_sem);
    }

    if (!sem) {
        return SDL_InvalidParamError("sem");
    }

    /**
     * WaitOnAddress is subject to spurious and stolen wakeups so we
     * need to recalculate the effective timeout before every wait
     */
    now = SDL_GetTicks();
    deadline = now + (DWORD)timeout;

    for (;;) {
        count = sem->count;
        /* If no semaphore is available we need to wait */
        while (count == 0) {
            now = SDL_GetTicks();
            if (deadline > now) {
                timeout_eff = deadline - now;
            } else {
                return SDL_MUTEX_TIMEDOUT;
            }
            if (pWaitOnAddress(&sem->count, &count, sizeof(count), timeout_eff) == FALSE) {
                if (GetLastError() == ERROR_TIMEOUT) {
                    return SDL_MUTEX_TIMEDOUT;
                }
                return SDL_SetError("WaitOnAddress() failed");
            }
            count = sem->count;
        }

        /* Actually the semaphore is only consumed if this succeeds */
        /* If it doesn't we need to do everything again */
        if (InterlockedCompareExchange(&sem->count, count - 1, count) == count) {
            return 0;
        }
    }
}

static Uint32 SDL_SemValue_atom(SDL_sem *_sem)
{
    SDL_sem_atom *sem = (SDL_sem_atom *)_sem;

    if (!sem) {
        SDL_InvalidParamError("sem");
        return 0;
    }

    return (Uint32)sem->count;
}

static int SDL_SemPost_atom(SDL_sem *_sem)
{
    SDL_sem_atom *sem = (SDL_sem_atom *)_sem;

    if (!sem) {
        return SDL_InvalidParamError("sem");
    }

    InterlockedIncrement(&sem->count);
    pWakeByAddressSingle(&sem->count);

    return 0;
}

static const SDL_sem_impl_t SDL_sem_impl_atom = {
    &SDL_CreateSemaphore_atom,
    &SDL_DestroySemaphore_atom,
    &SDL_SemWaitTimeout_atom,
    &SDL_SemTryWait_atom,
    &SDL_SemWait_atom,
    &SDL_SemValue_atom,
    &SDL_SemPost_atom,
};
#endif /* !SDL_WINAPI_FAMILY_PHONE */

/**
 * Fallback Semaphore implementation using Kernel Semaphores
 */

typedef struct SDL_semaphore_kern
{
    HANDLE id;
    LONG count;
} SDL_sem_kern;

/* Create a semaphore */
static SDL_sem *SDL_CreateSemaphore_kern(Uint32 initial_value)
{
    SDL_sem_kern *sem;

    /* Allocate sem memory */
    sem = (SDL_sem_kern *)SDL_malloc(sizeof(*sem));
    if (sem) {
        /* Create the semaphore, with max value 32K */
#ifdef __WINRT__
        sem->id = CreateSemaphoreEx(NULL, initial_value, 32 * 1024, NULL, 0, SEMAPHORE_ALL_ACCESS);
#else
        sem->id = CreateSemaphore(NULL, initial_value, 32 * 1024, NULL);
#endif
        sem->count = initial_value;
        if (!sem->id) {
            SDL_SetError("Couldn't create semaphore");
            SDL_free(sem);
            sem = NULL;
        }
    } else {
        SDL_OutOfMemory();
    }
    return (SDL_sem *)sem;
}

/* Free the semaphore */
static void SDL_DestroySemaphore_kern(SDL_sem *_sem)
{
    SDL_sem_kern *sem = (SDL_sem_kern *)_sem;
    if (sem) {
        if (sem->id) {
            CloseHandle(sem->id);
            sem->id = 0;
        }
        SDL_free(sem);
    }
}

static int SDL_SemWaitTimeout_kern(SDL_sem *_sem, Uint32 timeout)
{
    SDL_sem_kern *sem = (SDL_sem_kern *)_sem;
    int retval;
    DWORD dwMilliseconds;

    if (!sem) {
        return SDL_InvalidParamError("sem");
    }

    if (timeout == SDL_MUTEX_MAXWAIT) {
        dwMilliseconds = INFINITE;
    } else {
        dwMilliseconds = (DWORD)timeout;
    }
    switch (WaitForSingleObjectEx(sem->id, dwMilliseconds, FALSE)) {
    case WAIT_OBJECT_0:
        InterlockedDecrement(&sem->count);
        retval = 0;
        break;
    case WAIT_TIMEOUT:
        retval = SDL_MUTEX_TIMEDOUT;
        break;
    default:
        retval = SDL_SetError("WaitForSingleObject() failed");
        break;
    }
    return retval;
}

static int SDL_SemTryWait_kern(SDL_sem *sem)
{
    return SDL_SemWaitTimeout_kern(sem, 0);
}

static int SDL_SemWait_kern(SDL_sem *sem)
{
    return SDL_SemWaitTimeout_kern(sem, SDL_MUTEX_MAXWAIT);
}

/* Returns the current count of the semaphore */
static Uint32 SDL_SemValue_kern(SDL_sem *_sem)
{
    SDL_sem_kern *sem = (SDL_sem_kern *)_sem;
    if (!sem) {
        SDL_InvalidParamError("sem");
        return 0;
    }
    return (Uint32)sem->count;
}

static int SDL_SemPost_kern(SDL_sem *_sem)
{
    SDL_sem_kern *sem = (SDL_sem_kern *)_sem;
    if (!sem) {
        return SDL_InvalidParamError("sem");
    }
    /* Increase the counter in the first place, because
     * after a successful release the semaphore may
     * immediately get destroyed by another thread which
     * is waiting for this semaphore.
     */
    InterlockedIncrement(&sem->count);
    if (ReleaseSemaphore(sem->id, 1, NULL) == FALSE) {
        InterlockedDecrement(&sem->count); /* restore */
        return SDL_SetError("ReleaseSemaphore() failed");
    }
    return 0;
}

static const SDL_sem_impl_t SDL_sem_impl_kern = {
    &SDL_CreateSemaphore_kern,
    &SDL_DestroySemaphore_kern,
    &SDL_SemWaitTimeout_kern,
    &SDL_SemTryWait_kern,
    &SDL_SemWait_kern,
    &SDL_SemValue_kern,
    &SDL_SemPost_kern,
};

/**
 * Runtime selection and redirection
 */

SDL_sem *SDL_CreateSemaphore(Uint32 initial_value)
{
    if (!SDL_sem_impl_active.Create) {
        /* Default to fallback implementation */
        const SDL_sem_impl_t *impl = &SDL_sem_impl_kern;

#if !SDL_WINAPI_FAMILY_PHONE
        if (!SDL_GetHintBoolean(SDL_HINT_WINDOWS_FORCE_SEMAPHORE_KERNEL, SDL_FALSE)) {
#ifdef __WINRT__
            /* Link statically on this platform */
            impl = &SDL_sem_impl_atom;
#else
            /* We already statically link to features from this Api
             * Set (e.g. WaitForSingleObject). Dynamically loading
             * API Sets is not explicitly documented but according to
             * Microsoft our specific use case is legal and correct:
             * https://github.com/microsoft/STL/pull/593#issuecomment-655799859
             */
            HMODULE synch120 = GetModuleHandle(TEXT("api-ms-win-core-synch-l1-2-0.dll"));
            if (synch120) {
                /* Try to load required functions provided by Win 8 or newer */
                pWaitOnAddress = (pfnWaitOnAddress)GetProcAddress(synch120, "WaitOnAddress");
                pWakeByAddressSingle = (pfnWakeByAddressSingle)GetProcAddress(synch120, "WakeByAddressSingle");

                if (pWaitOnAddress && pWakeByAddressSingle) {
                    impl = &SDL_sem_impl_atom;
                }
            }
#endif
        }
#endif

        /* Copy instead of using pointer to save one level of indirection */
        SDL_copyp(&SDL_sem_impl_active, impl);
    }
    return SDL_sem_impl_active.Create(initial_value);
}

void SDL_DestroySemaphore(SDL_sem *sem)
{
    SDL_sem_impl_active.Destroy(sem);
}

int SDL_SemWaitTimeout(SDL_sem *sem, Uint32 timeout)
{
    return SDL_sem_impl_active.WaitTimeout(sem, timeout);
}

int SDL_SemTryWait(SDL_sem *sem)
{
    return SDL_sem_impl_active.TryWait(sem);
}

int SDL_SemWait(SDL_sem *sem)
{
    return SDL_sem_impl_active.Wait(sem);
}

Uint32 SDL_SemValue(SDL_sem *sem)
{
    return SDL_sem_impl_active.Value(sem);
}

int SDL_SemPost(SDL_sem *sem)
{
    return SDL_sem_impl_active.Post(sem);
}

#endif /* SDL_THREAD_WINDOWS */

/* vi: set ts=4 sw=4 expandtab: */
