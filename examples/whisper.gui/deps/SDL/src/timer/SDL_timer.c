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
#include "../SDL_internal.h"

#include "SDL_timer.h"
#include "SDL_timer_c.h"
#include "SDL_atomic.h"
#include "SDL_cpuinfo.h"
#include "../thread/SDL_systhread.h"

/* #define DEBUG_TIMERS */

#if !defined(__EMSCRIPTEN__) || !defined(SDL_THREADS_DISABLED)

typedef struct _SDL_Timer
{
    int timerID;
    SDL_TimerCallback callback;
    void *param;
    Uint32 interval;
    Uint32 scheduled;
    SDL_atomic_t canceled;
    struct _SDL_Timer *next;
} SDL_Timer;

typedef struct _SDL_TimerMap
{
    int timerID;
    SDL_Timer *timer;
    struct _SDL_TimerMap *next;
} SDL_TimerMap;

/* The timers are kept in a sorted list */
typedef struct
{
    /* Data used by the main thread */
    SDL_Thread *thread;
    SDL_atomic_t nextID;
    SDL_TimerMap *timermap;
    SDL_mutex *timermap_lock;

    /* Padding to separate cache lines between threads */
    char cache_pad[SDL_CACHELINE_SIZE];

    /* Data used to communicate with the timer thread */
    SDL_SpinLock lock;
    SDL_sem *sem;
    SDL_Timer *pending;
    SDL_Timer *freelist;
    SDL_atomic_t active;

    /* List of timers - this is only touched by the timer thread */
    SDL_Timer *timers;
} SDL_TimerData;

static SDL_TimerData SDL_timer_data;

/* The idea here is that any thread might add a timer, but a single
 * thread manages the active timer queue, sorted by scheduling time.
 *
 * Timers are removed by simply setting a canceled flag
 */

static void SDL_AddTimerInternal(SDL_TimerData *data, SDL_Timer *timer)
{
    SDL_Timer *prev, *curr;

    prev = NULL;
    for (curr = data->timers; curr; prev = curr, curr = curr->next) {
        if ((Sint32)(timer->scheduled - curr->scheduled) < 0) {
            break;
        }
    }

    /* Insert the timer here! */
    if (prev) {
        prev->next = timer;
    } else {
        data->timers = timer;
    }
    timer->next = curr;
}

static int SDLCALL SDL_TimerThread(void *_data)
{
    SDL_TimerData *data = (SDL_TimerData *)_data;
    SDL_Timer *pending;
    SDL_Timer *current;
    SDL_Timer *freelist_head = NULL;
    SDL_Timer *freelist_tail = NULL;
    Uint32 tick, now, interval, delay;

    /* Threaded timer loop:
     *  1. Queue timers added by other threads
     *  2. Handle any timers that should dispatch this cycle
     *  3. Wait until next dispatch time or new timer arrives
     */
    for (;;) {
        /* Pending and freelist maintenance */
        SDL_AtomicLock(&data->lock);
        {
            /* Get any timers ready to be queued */
            pending = data->pending;
            data->pending = NULL;

            /* Make any unused timer structures available */
            if (freelist_head) {
                freelist_tail->next = data->freelist;
                data->freelist = freelist_head;
            }
        }
        SDL_AtomicUnlock(&data->lock);

        /* Sort the pending timers into our list */
        while (pending) {
            current = pending;
            pending = pending->next;
            SDL_AddTimerInternal(data, current);
        }
        freelist_head = NULL;
        freelist_tail = NULL;

        /* Check to see if we're still running, after maintenance */
        if (!SDL_AtomicGet(&data->active)) {
            break;
        }

        /* Initial delay if there are no timers */
        delay = SDL_MUTEX_MAXWAIT;

        tick = SDL_GetTicks();

        /* Process all the pending timers for this tick */
        while (data->timers) {
            current = data->timers;

            if ((Sint32)(tick - current->scheduled) < 0) {
                /* Scheduled for the future, wait a bit */
                delay = (current->scheduled - tick);
                break;
            }

            /* We're going to do something with this timer */
            data->timers = current->next;

            if (SDL_AtomicGet(&current->canceled)) {
                interval = 0;
            } else {
                interval = current->callback(current->interval, current->param);
            }

            if (interval > 0) {
                /* Reschedule this timer */
                current->interval = interval;
                current->scheduled = tick + interval;
                SDL_AddTimerInternal(data, current);
            } else {
                if (!freelist_head) {
                    freelist_head = current;
                }
                if (freelist_tail) {
                    freelist_tail->next = current;
                }
                freelist_tail = current;

                SDL_AtomicSet(&current->canceled, 1);
            }
        }

        /* Adjust the delay based on processing time */
        now = SDL_GetTicks();
        interval = (now - tick);
        if (interval > delay) {
            delay = 0;
        } else {
            delay -= interval;
        }

        /* Note that each time a timer is added, this will return
           immediately, but we process the timers added all at once.
           That's okay, it just means we run through the loop a few
           extra times.
         */
        SDL_SemWaitTimeout(data->sem, delay);
    }
    return 0;
}

int SDL_TimerInit(void)
{
    SDL_TimerData *data = &SDL_timer_data;

    if (!SDL_AtomicGet(&data->active)) {
        const char *name = "SDLTimer";
        data->timermap_lock = SDL_CreateMutex();
        if (!data->timermap_lock) {
            return -1;
        }

        data->sem = SDL_CreateSemaphore(0);
        if (!data->sem) {
            SDL_DestroyMutex(data->timermap_lock);
            return -1;
        }

        SDL_AtomicSet(&data->active, 1);

        /* Timer threads use a callback into the app, so we can't set a limited stack size here. */
        data->thread = SDL_CreateThreadInternal(SDL_TimerThread, name, 0, data);
        if (!data->thread) {
            SDL_TimerQuit();
            return -1;
        }

        SDL_AtomicSet(&data->nextID, 1);
    }
    return 0;
}

void SDL_TimerQuit(void)
{
    SDL_TimerData *data = &SDL_timer_data;
    SDL_Timer *timer;
    SDL_TimerMap *entry;

    if (SDL_AtomicCAS(&data->active, 1, 0)) { /* active? Move to inactive. */
        /* Shutdown the timer thread */
        if (data->thread) {
            SDL_SemPost(data->sem);
            SDL_WaitThread(data->thread, NULL);
            data->thread = NULL;
        }

        SDL_DestroySemaphore(data->sem);
        data->sem = NULL;

        /* Clean up the timer entries */
        while (data->timers) {
            timer = data->timers;
            data->timers = timer->next;
            SDL_free(timer);
        }
        while (data->freelist) {
            timer = data->freelist;
            data->freelist = timer->next;
            SDL_free(timer);
        }
        while (data->timermap) {
            entry = data->timermap;
            data->timermap = entry->next;
            SDL_free(entry);
        }

        SDL_DestroyMutex(data->timermap_lock);
        data->timermap_lock = NULL;
    }
}

SDL_TimerID SDL_AddTimer(Uint32 interval, SDL_TimerCallback callback, void *param)
{
    SDL_TimerData *data = &SDL_timer_data;
    SDL_Timer *timer;
    SDL_TimerMap *entry;

    SDL_AtomicLock(&data->lock);
    if (!SDL_AtomicGet(&data->active)) {
        if (SDL_TimerInit() < 0) {
            SDL_AtomicUnlock(&data->lock);
            return 0;
        }
    }

    timer = data->freelist;
    if (timer) {
        data->freelist = timer->next;
    }
    SDL_AtomicUnlock(&data->lock);

    if (timer) {
        SDL_RemoveTimer(timer->timerID);
    } else {
        timer = (SDL_Timer *)SDL_malloc(sizeof(*timer));
        if (!timer) {
            SDL_OutOfMemory();
            return 0;
        }
    }
    timer->timerID = SDL_AtomicIncRef(&data->nextID);
    timer->callback = callback;
    timer->param = param;
    timer->interval = interval;
    timer->scheduled = SDL_GetTicks() + interval;
    SDL_AtomicSet(&timer->canceled, 0);

    entry = (SDL_TimerMap *)SDL_malloc(sizeof(*entry));
    if (!entry) {
        SDL_free(timer);
        SDL_OutOfMemory();
        return 0;
    }
    entry->timer = timer;
    entry->timerID = timer->timerID;

    SDL_LockMutex(data->timermap_lock);
    entry->next = data->timermap;
    data->timermap = entry;
    SDL_UnlockMutex(data->timermap_lock);

    /* Add the timer to the pending list for the timer thread */
    SDL_AtomicLock(&data->lock);
    timer->next = data->pending;
    data->pending = timer;
    SDL_AtomicUnlock(&data->lock);

    /* Wake up the timer thread if necessary */
    SDL_SemPost(data->sem);

    return entry->timerID;
}

SDL_bool SDL_RemoveTimer(SDL_TimerID id)
{
    SDL_TimerData *data = &SDL_timer_data;
    SDL_TimerMap *prev, *entry;
    SDL_bool canceled = SDL_FALSE;

    /* Find the timer */
    SDL_LockMutex(data->timermap_lock);
    prev = NULL;
    for (entry = data->timermap; entry; prev = entry, entry = entry->next) {
        if (entry->timerID == id) {
            if (prev) {
                prev->next = entry->next;
            } else {
                data->timermap = entry->next;
            }
            break;
        }
    }
    SDL_UnlockMutex(data->timermap_lock);

    if (entry) {
        if (!SDL_AtomicGet(&entry->timer->canceled)) {
            SDL_AtomicSet(&entry->timer->canceled, 1);
            canceled = SDL_TRUE;
        }
        SDL_free(entry);
    }
    return canceled;
}

#else

#include <emscripten/emscripten.h>
#include <emscripten/eventloop.h>

typedef struct _SDL_TimerMap
{
    int timerID;
    int timeoutID;
    Uint32 interval;
    SDL_TimerCallback callback;
    void *param;
    struct _SDL_TimerMap *next;
} SDL_TimerMap;

typedef struct
{
    int nextID;
    SDL_TimerMap *timermap;
} SDL_TimerData;

static SDL_TimerData SDL_timer_data;

static void SDL_Emscripten_TimerHelper(void *userdata)
{
    SDL_TimerMap *entry = (SDL_TimerMap *)userdata;
    entry->interval = entry->callback(entry->interval, entry->param);
    if (entry->interval > 0) {
        entry->timeoutID = emscripten_set_timeout(&SDL_Emscripten_TimerHelper,
                                                  entry->interval,
                                                  entry);
    }
}

int SDL_TimerInit(void)
{
    return 0;
}

void SDL_TimerQuit(void)
{
    SDL_TimerData *data = &SDL_timer_data;
    SDL_TimerMap *entry;

    while (data->timermap) {
        entry = data->timermap;
        data->timermap = entry->next;
        SDL_free(entry);
    }
}

SDL_TimerID SDL_AddTimer(Uint32 interval, SDL_TimerCallback callback, void *param)
{
    SDL_TimerData *data = &SDL_timer_data;
    SDL_TimerMap *entry;

    entry = (SDL_TimerMap *)SDL_malloc(sizeof(*entry));
    if (!entry) {
        SDL_OutOfMemory();
        return 0;
    }
    entry->timerID = ++data->nextID;
    entry->callback = callback;
    entry->param = param;
    entry->interval = interval;

    entry->timeoutID = emscripten_set_timeout(&SDL_Emscripten_TimerHelper,
                                              entry->interval,
                                              entry);

    entry->next = data->timermap;
    data->timermap = entry;

    return entry->timerID;
}

SDL_bool SDL_RemoveTimer(SDL_TimerID id)
{
    SDL_TimerData *data = &SDL_timer_data;
    SDL_TimerMap *prev, *entry;

    /* Find the timer */
    prev = NULL;
    for (entry = data->timermap; entry; prev = entry, entry = entry->next) {
        if (entry->timerID == id) {
            if (prev) {
                prev->next = entry->next;
            } else {
                data->timermap = entry->next;
            }
            break;
        }
    }

    if (entry) {
        emscripten_clear_timeout(entry->timeoutID);
        SDL_free(entry);

        return SDL_TRUE;
    }
    return SDL_FALSE;
}

#endif

/* This is a legacy support function; SDL_GetTicks() returns a Uint32,
   which wraps back to zero every ~49 days. The newer SDL_GetTicks64()
   doesn't have this problem, so we just wrap that function and clamp to
   the low 32-bits for binary compatibility. */
Uint32 SDL_GetTicks(void)
{
    return (Uint32)(SDL_GetTicks64() & 0xFFFFFFFF);
}

/* vi: set ts=4 sw=4 expandtab: */
