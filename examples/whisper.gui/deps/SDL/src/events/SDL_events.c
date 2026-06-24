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

/* General event handling code for SDL */

#include "SDL.h"
#include "SDL_events.h"
#include "SDL_thread.h"
#include "SDL_events_c.h"
#include "../SDL_hints_c.h"
#include "../timer/SDL_timer_c.h"
#ifndef SDL_JOYSTICK_DISABLED
#include "../joystick/SDL_joystick_c.h"
#endif
#include "../video/SDL_sysvideo.h"
#include "SDL_syswm.h"

/* An arbitrary limit so we don't have unbounded growth */
#define SDL_MAX_QUEUED_EVENTS 65535

/* Determines how often we wake to call SDL_PumpEvents() in SDL_WaitEventTimeout_Device() */
#define PERIODIC_POLL_INTERVAL_MS 3000

typedef struct SDL_EventWatcher
{
    SDL_EventFilter callback;
    void *userdata;
    SDL_bool removed;
} SDL_EventWatcher;

static SDL_mutex *SDL_event_watchers_lock;
static SDL_EventWatcher SDL_EventOK;
static SDL_EventWatcher *SDL_event_watchers = NULL;
static int SDL_event_watchers_count = 0;
static SDL_bool SDL_event_watchers_dispatching = SDL_FALSE;
static SDL_bool SDL_event_watchers_removed = SDL_FALSE;
static SDL_atomic_t SDL_sentinel_pending;

typedef struct
{
    Uint32 bits[8];
} SDL_DisabledEventBlock;

static SDL_DisabledEventBlock *SDL_disabled_events[256];
static Uint32 SDL_userevents = SDL_USEREVENT;

/* Private data -- event queue */
typedef struct _SDL_EventEntry
{
    SDL_Event event;
    SDL_SysWMmsg msg;
    struct _SDL_EventEntry *prev;
    struct _SDL_EventEntry *next;
} SDL_EventEntry;

typedef struct _SDL_SysWMEntry
{
    SDL_SysWMmsg msg;
    struct _SDL_SysWMEntry *next;
} SDL_SysWMEntry;

static struct
{
    SDL_mutex *lock;
    SDL_bool active;
    SDL_atomic_t count;
    int max_events_seen;
    SDL_EventEntry *head;
    SDL_EventEntry *tail;
    SDL_EventEntry *free;
    SDL_SysWMEntry *wmmsg_used;
    SDL_SysWMEntry *wmmsg_free;
} SDL_EventQ = { NULL, SDL_FALSE, { 0 }, 0, NULL, NULL, NULL, NULL, NULL };

#ifndef SDL_JOYSTICK_DISABLED

static SDL_bool SDL_update_joysticks = SDL_TRUE;

static void SDL_CalculateShouldUpdateJoysticks(SDL_bool hint_value)
{
    if (hint_value &&
        (!SDL_disabled_events[SDL_JOYAXISMOTION >> 8] || SDL_JoystickEventState(SDL_QUERY))) {
        SDL_update_joysticks = SDL_TRUE;
    } else {
        SDL_update_joysticks = SDL_FALSE;
    }
}

static void SDLCALL SDL_AutoUpdateJoysticksChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_CalculateShouldUpdateJoysticks(SDL_GetStringBoolean(hint, SDL_TRUE));
}

#endif /* !SDL_JOYSTICK_DISABLED */

#ifndef SDL_SENSOR_DISABLED

static SDL_bool SDL_update_sensors = SDL_TRUE;

static void SDL_CalculateShouldUpdateSensors(SDL_bool hint_value)
{
    if (hint_value &&
        !SDL_disabled_events[SDL_SENSORUPDATE >> 8]) {
        SDL_update_sensors = SDL_TRUE;
    } else {
        SDL_update_sensors = SDL_FALSE;
    }
}

static void SDLCALL SDL_AutoUpdateSensorsChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_CalculateShouldUpdateSensors(SDL_GetStringBoolean(hint, SDL_TRUE));
}

#endif /* !SDL_SENSOR_DISABLED */

static void SDLCALL SDL_PollSentinelChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    (void)SDL_EventState(SDL_POLLSENTINEL, SDL_GetStringBoolean(hint, SDL_TRUE) ? SDL_ENABLE : SDL_DISABLE);
}

/**
 * Verbosity of logged events as defined in SDL_HINT_EVENT_LOGGING:
 *  - 0: (default) no logging
 *  - 1: logging of most events
 *  - 2: as above, plus mouse and finger motion
 *  - 3: as above, plus SDL_SysWMEvents
 */
static int SDL_EventLoggingVerbosity = 0;

static void SDLCALL SDL_EventLoggingChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_EventLoggingVerbosity = (hint && *hint) ? SDL_clamp(SDL_atoi(hint), 0, 3) : 0;
}

static void SDL_LogEvent(const SDL_Event *event)
{
    char name[64];
    char details[128];

    /* sensor/mouse/finger motion are spammy, ignore these if they aren't demanded. */
    if ((SDL_EventLoggingVerbosity < 2) &&
        ((event->type == SDL_MOUSEMOTION) ||
         (event->type == SDL_FINGERMOTION) ||
         (event->type == SDL_CONTROLLERTOUCHPADMOTION) ||
         (event->type == SDL_CONTROLLERSENSORUPDATE) ||
         (event->type == SDL_SENSORUPDATE))) {
        return;
    }

    /* window manager events are even more spammy, and don't provide much useful info. */
    if ((SDL_EventLoggingVerbosity < 3) && (event->type == SDL_SYSWMEVENT)) {
        return;
    }

/* this is to make (void)SDL_snprintf() calls cleaner. */
#define uint unsigned int

    name[0] = '\0';
    details[0] = '\0';

    /* !!! FIXME: This code is kinda ugly, sorry. */

    if ((event->type >= SDL_USEREVENT) && (event->type <= SDL_LASTEVENT)) {
        char plusstr[16];
        SDL_strlcpy(name, "SDL_USEREVENT", sizeof(name));
        if (event->type > SDL_USEREVENT) {
            (void)SDL_snprintf(plusstr, sizeof(plusstr), "+%u", ((uint)event->type) - SDL_USEREVENT);
        } else {
            plusstr[0] = '\0';
        }
        (void)SDL_snprintf(details, sizeof(details), "%s (timestamp=%u windowid=%u code=%d data1=%p data2=%p)",
                           plusstr, (uint)event->user.timestamp, (uint)event->user.windowID,
                           (int)event->user.code, event->user.data1, event->user.data2);
    }

    switch (event->type) {
#define SDL_EVENT_CASE(x) \
    case x:               \
        SDL_strlcpy(name, #x, sizeof(name));
        SDL_EVENT_CASE(SDL_FIRSTEVENT)
        SDL_strlcpy(details, " (THIS IS PROBABLY A BUG!)", sizeof(details));
        break;
        SDL_EVENT_CASE(SDL_QUIT)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u)", (uint)event->quit.timestamp);
        break;
        SDL_EVENT_CASE(SDL_APP_TERMINATING)
        break;
        SDL_EVENT_CASE(SDL_APP_LOWMEMORY)
        break;
        SDL_EVENT_CASE(SDL_APP_WILLENTERBACKGROUND)
        break;
        SDL_EVENT_CASE(SDL_APP_DIDENTERBACKGROUND)
        break;
        SDL_EVENT_CASE(SDL_APP_WILLENTERFOREGROUND)
        break;
        SDL_EVENT_CASE(SDL_APP_DIDENTERFOREGROUND)
        break;
        SDL_EVENT_CASE(SDL_LOCALECHANGED)
        break;
        SDL_EVENT_CASE(SDL_KEYMAPCHANGED)
        break;
        SDL_EVENT_CASE(SDL_CLIPBOARDUPDATE)
        break;
        SDL_EVENT_CASE(SDL_RENDER_TARGETS_RESET)
        break;
        SDL_EVENT_CASE(SDL_RENDER_DEVICE_RESET)
        break;

        SDL_EVENT_CASE(SDL_DISPLAYEVENT)
        {
            char name2[64];
            switch (event->display.event) {
            case SDL_DISPLAYEVENT_NONE:
                SDL_strlcpy(name2, "SDL_DISPLAYEVENT_NONE (THIS IS PROBABLY A BUG!)", sizeof(name2));
                break;
#define SDL_DISPLAYEVENT_CASE(x)               \
    case x:                                    \
        SDL_strlcpy(name2, #x, sizeof(name2)); \
        break
                SDL_DISPLAYEVENT_CASE(SDL_DISPLAYEVENT_ORIENTATION);
                SDL_DISPLAYEVENT_CASE(SDL_DISPLAYEVENT_CONNECTED);
                SDL_DISPLAYEVENT_CASE(SDL_DISPLAYEVENT_DISCONNECTED);
                SDL_DISPLAYEVENT_CASE(SDL_DISPLAYEVENT_MOVED);
#undef SDL_DISPLAYEVENT_CASE
            default:
                SDL_strlcpy(name2, "UNKNOWN (bug? fixme?)", sizeof(name2));
                break;
            }
            (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u display=%u event=%s data1=%d)",
                               (uint)event->display.timestamp, (uint)event->display.display, name2, (int)event->display.data1);
            break;
        }

        SDL_EVENT_CASE(SDL_WINDOWEVENT)
        {
            char name2[64];
            switch (event->window.event) {
            case SDL_WINDOWEVENT_NONE:
                SDL_strlcpy(name2, "SDL_WINDOWEVENT_NONE (THIS IS PROBABLY A BUG!)", sizeof(name2));
                break;
#define SDL_WINDOWEVENT_CASE(x)                \
    case x:                                    \
        SDL_strlcpy(name2, #x, sizeof(name2)); \
        break
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_SHOWN);
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_HIDDEN);
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_EXPOSED);
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_MOVED);
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_RESIZED);
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_SIZE_CHANGED);
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_MINIMIZED);
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_MAXIMIZED);
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_RESTORED);
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_ENTER);
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_LEAVE);
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_FOCUS_GAINED);
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_FOCUS_LOST);
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_CLOSE);
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_TAKE_FOCUS);
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_HIT_TEST);
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_ICCPROF_CHANGED);
                SDL_WINDOWEVENT_CASE(SDL_WINDOWEVENT_DISPLAY_CHANGED);
#undef SDL_WINDOWEVENT_CASE
            default:
                SDL_strlcpy(name2, "UNKNOWN (bug? fixme?)", sizeof(name2));
                break;
            }
            (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u event=%s data1=%d data2=%d)",
                               (uint)event->window.timestamp, (uint)event->window.windowID, name2, (int)event->window.data1, (int)event->window.data2);
            break;
        }

        SDL_EVENT_CASE(SDL_SYSWMEVENT)
        /* !!! FIXME: we don't delve further at the moment. */
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u)", (uint)event->syswm.timestamp);
        break;

#define PRINT_KEY_EVENT(event)                                                                                                   \
    (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u state=%s repeat=%s scancode=%u keycode=%u mod=%u)", \
                       (uint)event->key.timestamp, (uint)event->key.windowID,                                                    \
                       event->key.state == SDL_PRESSED ? "pressed" : "released",                                                 \
                       event->key.repeat ? "true" : "false",                                                                     \
                       (uint)event->key.keysym.scancode,                                                                         \
                       (uint)event->key.keysym.sym,                                                                              \
                       (uint)event->key.keysym.mod)
        SDL_EVENT_CASE(SDL_KEYDOWN)
        PRINT_KEY_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_KEYUP)
        PRINT_KEY_EVENT(event);
        break;
#undef PRINT_KEY_EVENT

        SDL_EVENT_CASE(SDL_TEXTEDITING)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u text='%s' start=%d length=%d)",
                           (uint)event->edit.timestamp, (uint)event->edit.windowID,
                           event->edit.text, (int)event->edit.start, (int)event->edit.length);
        break;

        SDL_EVENT_CASE(SDL_TEXTINPUT)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u text='%s')", (uint)event->text.timestamp, (uint)event->text.windowID, event->text.text);
        break;

        SDL_EVENT_CASE(SDL_MOUSEMOTION)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u which=%u state=%u x=%d y=%d xrel=%d yrel=%d)",
                           (uint)event->motion.timestamp, (uint)event->motion.windowID,
                           (uint)event->motion.which, (uint)event->motion.state,
                           (int)event->motion.x, (int)event->motion.y,
                           (int)event->motion.xrel, (int)event->motion.yrel);
        break;

#define PRINT_MBUTTON_EVENT(event)                                                                                              \
    (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u which=%u button=%u state=%s clicks=%u x=%d y=%d)", \
                       (uint)event->button.timestamp, (uint)event->button.windowID,                                             \
                       (uint)event->button.which, (uint)event->button.button,                                                   \
                       event->button.state == SDL_PRESSED ? "pressed" : "released",                                             \
                       (uint)event->button.clicks, (int)event->button.x, (int)event->button.y)
        SDL_EVENT_CASE(SDL_MOUSEBUTTONDOWN)
        PRINT_MBUTTON_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_MOUSEBUTTONUP)
        PRINT_MBUTTON_EVENT(event);
        break;
#undef PRINT_MBUTTON_EVENT

        SDL_EVENT_CASE(SDL_MOUSEWHEEL)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u which=%u x=%d y=%d preciseX=%f preciseY=%f direction=%s)",
                           (uint)event->wheel.timestamp, (uint)event->wheel.windowID,
                           (uint)event->wheel.which, (int)event->wheel.x, (int)event->wheel.y,
                           event->wheel.preciseX, event->wheel.preciseY,
                           event->wheel.direction == SDL_MOUSEWHEEL_NORMAL ? "normal" : "flipped");
        break;

        SDL_EVENT_CASE(SDL_JOYAXISMOTION)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d axis=%u value=%d)",
                           (uint)event->jaxis.timestamp, (int)event->jaxis.which,
                           (uint)event->jaxis.axis, (int)event->jaxis.value);
        break;

        SDL_EVENT_CASE(SDL_JOYBALLMOTION)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d ball=%u xrel=%d yrel=%d)",
                           (uint)event->jball.timestamp, (int)event->jball.which,
                           (uint)event->jball.ball, (int)event->jball.xrel, (int)event->jball.yrel);
        break;

        SDL_EVENT_CASE(SDL_JOYHATMOTION)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d hat=%u value=%u)",
                           (uint)event->jhat.timestamp, (int)event->jhat.which,
                           (uint)event->jhat.hat, (uint)event->jhat.value);
        break;

#define PRINT_JBUTTON_EVENT(event)                                                              \
    (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d button=%u state=%s)", \
                       (uint)event->jbutton.timestamp, (int)event->jbutton.which,               \
                       (uint)event->jbutton.button, event->jbutton.state == SDL_PRESSED ? "pressed" : "released")
        SDL_EVENT_CASE(SDL_JOYBUTTONDOWN)
        PRINT_JBUTTON_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_JOYBUTTONUP)
        PRINT_JBUTTON_EVENT(event);
        break;
#undef PRINT_JBUTTON_EVENT

#define PRINT_JOYDEV_EVENT(event) (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d)", (uint)event->jdevice.timestamp, (int)event->jdevice.which)
        SDL_EVENT_CASE(SDL_JOYDEVICEADDED)
        PRINT_JOYDEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_JOYDEVICEREMOVED)
        PRINT_JOYDEV_EVENT(event);
        break;
#undef PRINT_JOYDEV_EVENT

        SDL_EVENT_CASE(SDL_CONTROLLERAXISMOTION)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d axis=%u value=%d)",
                           (uint)event->caxis.timestamp, (int)event->caxis.which,
                           (uint)event->caxis.axis, (int)event->caxis.value);
        break;

#define PRINT_CBUTTON_EVENT(event)                                                              \
    (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d button=%u state=%s)", \
                       (uint)event->cbutton.timestamp, (int)event->cbutton.which,               \
                       (uint)event->cbutton.button, event->cbutton.state == SDL_PRESSED ? "pressed" : "released")
        SDL_EVENT_CASE(SDL_CONTROLLERBUTTONDOWN)
        PRINT_CBUTTON_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_CONTROLLERBUTTONUP)
        PRINT_CBUTTON_EVENT(event);
        break;
#undef PRINT_CBUTTON_EVENT

#define PRINT_CONTROLLERDEV_EVENT(event) (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d)", (uint)event->cdevice.timestamp, (int)event->cdevice.which)
        SDL_EVENT_CASE(SDL_CONTROLLERDEVICEADDED)
        PRINT_CONTROLLERDEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_CONTROLLERDEVICEREMOVED)
        PRINT_CONTROLLERDEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_CONTROLLERDEVICEREMAPPED)
        PRINT_CONTROLLERDEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_CONTROLLERSTEAMHANDLEUPDATED)
        PRINT_CONTROLLERDEV_EVENT(event);
        break;
#undef PRINT_CONTROLLERDEV_EVENT

#define PRINT_CTOUCHPAD_EVENT(event)                                                                                     \
    (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d touchpad=%d finger=%d x=%f y=%f pressure=%f)", \
                       (uint)event->ctouchpad.timestamp, (int)event->ctouchpad.which,                                    \
                       (int)event->ctouchpad.touchpad, (int)event->ctouchpad.finger,                                     \
                       event->ctouchpad.x, event->ctouchpad.y, event->ctouchpad.pressure)
        SDL_EVENT_CASE(SDL_CONTROLLERTOUCHPADDOWN)
        PRINT_CTOUCHPAD_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_CONTROLLERTOUCHPADUP)
        PRINT_CTOUCHPAD_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_CONTROLLERTOUCHPADMOTION)
        PRINT_CTOUCHPAD_EVENT(event);
        break;
#undef PRINT_CTOUCHPAD_EVENT

        SDL_EVENT_CASE(SDL_CONTROLLERSENSORUPDATE)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d sensor=%d data[0]=%f data[1]=%f data[2]=%f)",
                           (uint)event->csensor.timestamp, (int)event->csensor.which, (int)event->csensor.sensor,
                           event->csensor.data[0], event->csensor.data[1], event->csensor.data[2]);
        break;

#define PRINT_FINGER_EVENT(event)                                                                                                                      \
    (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u touchid=%" SDL_PRIs64 " fingerid=%" SDL_PRIs64 " x=%f y=%f dx=%f dy=%f pressure=%f)", \
                       (uint)event->tfinger.timestamp, event->tfinger.touchId,                                                                         \
                       event->tfinger.fingerId, event->tfinger.x, event->tfinger.y,                                                                    \
                       event->tfinger.dx, event->tfinger.dy, event->tfinger.pressure)
        SDL_EVENT_CASE(SDL_FINGERDOWN)
        PRINT_FINGER_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_FINGERUP)
        PRINT_FINGER_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_FINGERMOTION)
        PRINT_FINGER_EVENT(event);
        break;
#undef PRINT_FINGER_EVENT

#define PRINT_DOLLAR_EVENT(event)                                                                                                                      \
    (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u touchid=%" SDL_PRIs64 " gestureid=%" SDL_PRIs64 " numfingers=%u error=%f x=%f y=%f)", \
                       (uint)event->dgesture.timestamp, event->dgesture.touchId,                                                                       \
                       event->dgesture.gestureId, (uint)event->dgesture.numFingers,                                                                    \
                       event->dgesture.error, event->dgesture.x, event->dgesture.y)
        SDL_EVENT_CASE(SDL_DOLLARGESTURE)
        PRINT_DOLLAR_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_DOLLARRECORD)
        PRINT_DOLLAR_EVENT(event);
        break;
#undef PRINT_DOLLAR_EVENT

        SDL_EVENT_CASE(SDL_MULTIGESTURE)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u touchid=%" SDL_PRIs64 " dtheta=%f ddist=%f x=%f y=%f numfingers=%u)",
                           (uint)event->mgesture.timestamp, event->mgesture.touchId,
                           event->mgesture.dTheta, event->mgesture.dDist,
                           event->mgesture.x, event->mgesture.y, (uint)event->mgesture.numFingers);
        break;

#define PRINT_DROP_EVENT(event) (void)SDL_snprintf(details, sizeof(details), " (file='%s' timestamp=%u windowid=%u)", event->drop.file, (uint)event->drop.timestamp, (uint)event->drop.windowID)
        SDL_EVENT_CASE(SDL_DROPFILE)
        PRINT_DROP_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_DROPTEXT)
        PRINT_DROP_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_DROPBEGIN)
        PRINT_DROP_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_DROPCOMPLETE)
        PRINT_DROP_EVENT(event);
        break;
#undef PRINT_DROP_EVENT

#define PRINT_AUDIODEV_EVENT(event) (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%u iscapture=%s)", (uint)event->adevice.timestamp, (uint)event->adevice.which, event->adevice.iscapture ? "true" : "false")
        SDL_EVENT_CASE(SDL_AUDIODEVICEADDED)
        PRINT_AUDIODEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_AUDIODEVICEREMOVED)
        PRINT_AUDIODEV_EVENT(event);
        break;
#undef PRINT_AUDIODEV_EVENT

        SDL_EVENT_CASE(SDL_SENSORUPDATE)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d data[0]=%f data[1]=%f data[2]=%f data[3]=%f data[4]=%f data[5]=%f)",
                           (uint)event->sensor.timestamp, (int)event->sensor.which,
                           event->sensor.data[0], event->sensor.data[1], event->sensor.data[2],
                           event->sensor.data[3], event->sensor.data[4], event->sensor.data[5]);
        break;

#undef SDL_EVENT_CASE

    case SDL_POLLSENTINEL:
        /* No logging necessary for this one */
        break;

    default:
        if (!name[0]) {
            SDL_strlcpy(name, "UNKNOWN", sizeof(name));
            (void)SDL_snprintf(details, sizeof(details), " #%u! (Bug? FIXME?)", (uint)event->type);
        }
        break;
    }

    if (name[0]) {
        SDL_Log("SDL EVENT: %s%s", name, details);
    }

#undef uint
}

/* Public functions */

void SDL_StopEventLoop(void)
{
    const char *report = SDL_GetHint("SDL_EVENT_QUEUE_STATISTICS");
    int i;
    SDL_EventEntry *entry;
    SDL_SysWMEntry *wmmsg;

    SDL_LockMutex(SDL_EventQ.lock);

    SDL_EventQ.active = SDL_FALSE;

    if (report && SDL_atoi(report)) {
        SDL_Log("SDL EVENT QUEUE: Maximum events in-flight: %d\n",
                SDL_EventQ.max_events_seen);
    }

    /* Clean out EventQ */
    for (entry = SDL_EventQ.head; entry;) {
        SDL_EventEntry *next = entry->next;
        SDL_free(entry);
        entry = next;
    }
    for (entry = SDL_EventQ.free; entry;) {
        SDL_EventEntry *next = entry->next;
        SDL_free(entry);
        entry = next;
    }
    for (wmmsg = SDL_EventQ.wmmsg_used; wmmsg;) {
        SDL_SysWMEntry *next = wmmsg->next;
        SDL_free(wmmsg);
        wmmsg = next;
    }
    for (wmmsg = SDL_EventQ.wmmsg_free; wmmsg;) {
        SDL_SysWMEntry *next = wmmsg->next;
        SDL_free(wmmsg);
        wmmsg = next;
    }

    SDL_AtomicSet(&SDL_EventQ.count, 0);
    SDL_EventQ.max_events_seen = 0;
    SDL_EventQ.head = NULL;
    SDL_EventQ.tail = NULL;
    SDL_EventQ.free = NULL;
    SDL_EventQ.wmmsg_used = NULL;
    SDL_EventQ.wmmsg_free = NULL;
    SDL_AtomicSet(&SDL_sentinel_pending, 0);

    /* Clear disabled event state */
    for (i = 0; i < SDL_arraysize(SDL_disabled_events); ++i) {
        SDL_free(SDL_disabled_events[i]);
        SDL_disabled_events[i] = NULL;
    }

    if (SDL_event_watchers_lock) {
        SDL_DestroyMutex(SDL_event_watchers_lock);
        SDL_event_watchers_lock = NULL;
    }
    if (SDL_event_watchers) {
        SDL_free(SDL_event_watchers);
        SDL_event_watchers = NULL;
        SDL_event_watchers_count = 0;
    }
    SDL_zero(SDL_EventOK);

    SDL_UnlockMutex(SDL_EventQ.lock);

    if (SDL_EventQ.lock) {
        SDL_DestroyMutex(SDL_EventQ.lock);
        SDL_EventQ.lock = NULL;
    }
}

/* This function (and associated calls) may be called more than once */
int SDL_StartEventLoop(void)
{
    /* We'll leave the event queue alone, since we might have gotten
       some important events at launch (like SDL_DROPFILE)

       FIXME: Does this introduce any other bugs with events at startup?
     */

    /* Create the lock and set ourselves active */
#ifndef SDL_THREADS_DISABLED
    if (!SDL_EventQ.lock) {
        SDL_EventQ.lock = SDL_CreateMutex();
        if (SDL_EventQ.lock == NULL) {
            return -1;
        }
    }
    SDL_LockMutex(SDL_EventQ.lock);

    if (SDL_event_watchers_lock == NULL) {
        SDL_event_watchers_lock = SDL_CreateMutex();
        if (SDL_event_watchers_lock == NULL) {
            SDL_UnlockMutex(SDL_EventQ.lock);
            return -1;
        }
    }
#endif /* !SDL_THREADS_DISABLED */

    /* Process most event types */
    (void)SDL_EventState(SDL_TEXTINPUT, SDL_DISABLE);
    (void)SDL_EventState(SDL_TEXTEDITING, SDL_DISABLE);
    (void)SDL_EventState(SDL_SYSWMEVENT, SDL_DISABLE);
#if 0 /* Leave these events enabled so apps can respond to items being dragged onto them at startup */
    (void)SDL_EventState(SDL_DROPFILE, SDL_DISABLE);
    (void)SDL_EventState(SDL_DROPTEXT, SDL_DISABLE);
#endif

    SDL_EventQ.active = SDL_TRUE;
    SDL_UnlockMutex(SDL_EventQ.lock);
    return 0;
}

/* Add an event to the event queue -- called with the queue locked */
static int SDL_AddEvent(SDL_Event *event)
{
    SDL_EventEntry *entry;
    const int initial_count = SDL_AtomicGet(&SDL_EventQ.count);
    int final_count;

    if (initial_count >= SDL_MAX_QUEUED_EVENTS) {
        SDL_SetError("Event queue is full (%d events)", initial_count);
        return 0;
    }

    if (SDL_EventQ.free == NULL) {
        entry = (SDL_EventEntry *)SDL_malloc(sizeof(*entry));
        if (entry == NULL) {
            return 0;
        }
    } else {
        entry = SDL_EventQ.free;
        SDL_EventQ.free = entry->next;
    }

    if (SDL_EventLoggingVerbosity > 0) {
        SDL_LogEvent(event);
    }

    entry->event = *event;
    if (event->type == SDL_POLLSENTINEL) {
        SDL_AtomicAdd(&SDL_sentinel_pending, 1);
    } else if (event->type == SDL_SYSWMEVENT) {
        entry->msg = *event->syswm.msg;
        entry->event.syswm.msg = &entry->msg;
    }

    if (SDL_EventQ.tail) {
        SDL_EventQ.tail->next = entry;
        entry->prev = SDL_EventQ.tail;
        SDL_EventQ.tail = entry;
        entry->next = NULL;
    } else {
        SDL_assert(!SDL_EventQ.head);
        SDL_EventQ.head = entry;
        SDL_EventQ.tail = entry;
        entry->prev = NULL;
        entry->next = NULL;
    }

    final_count = SDL_AtomicAdd(&SDL_EventQ.count, 1) + 1;
    if (final_count > SDL_EventQ.max_events_seen) {
        SDL_EventQ.max_events_seen = final_count;
    }

    return 1;
}

/* Remove an event from the queue -- called with the queue locked */
static void SDL_CutEvent(SDL_EventEntry *entry)
{
    if (entry->prev) {
        entry->prev->next = entry->next;
    }
    if (entry->next) {
        entry->next->prev = entry->prev;
    }

    if (entry == SDL_EventQ.head) {
        SDL_assert(entry->prev == NULL);
        SDL_EventQ.head = entry->next;
    }
    if (entry == SDL_EventQ.tail) {
        SDL_assert(entry->next == NULL);
        SDL_EventQ.tail = entry->prev;
    }

    if (entry->event.type == SDL_POLLSENTINEL) {
        SDL_AtomicAdd(&SDL_sentinel_pending, -1);
    }

    entry->next = SDL_EventQ.free;
    SDL_EventQ.free = entry;
    SDL_assert(SDL_AtomicGet(&SDL_EventQ.count) > 0);
    SDL_AtomicAdd(&SDL_EventQ.count, -1);
}

static int SDL_SendWakeupEvent(void)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();
    if (_this == NULL || !_this->SendWakeupEvent) {
        return 0;
    }

    SDL_LockMutex(_this->wakeup_lock);
    {
        if (_this->wakeup_window) {
            _this->SendWakeupEvent(_this, _this->wakeup_window);

            /* No more wakeup events needed until we enter a new wait */
            _this->wakeup_window = NULL;
        }
    }
    SDL_UnlockMutex(_this->wakeup_lock);

    return 0;
}

/* Lock the event queue, take a peep at it, and unlock it */
static int SDL_PeepEventsInternal(SDL_Event *events, int numevents, SDL_eventaction action,
                                  Uint32 minType, Uint32 maxType, SDL_bool include_sentinel)
{
    int i, used, sentinels_expected = 0;

    /* Lock the event queue */
    used = 0;

    SDL_LockMutex(SDL_EventQ.lock);
    {
        /* Don't look after we've quit */
        if (!SDL_EventQ.active) {
            /* We get a few spurious events at shutdown, so don't warn then */
            if (action == SDL_GETEVENT) {
                SDL_SetError("The event system has been shut down");
            }
            SDL_UnlockMutex(SDL_EventQ.lock);
            return -1;
        }
        if (action == SDL_ADDEVENT) {
            for (i = 0; i < numevents; ++i) {
                used += SDL_AddEvent(&events[i]);
            }
        } else {
            SDL_EventEntry *entry, *next;
            SDL_SysWMEntry *wmmsg, *wmmsg_next;
            Uint32 type;

            if (action == SDL_GETEVENT) {
                /* Clean out any used wmmsg data
                   FIXME: Do we want to retain the data for some period of time?
                 */
                for (wmmsg = SDL_EventQ.wmmsg_used; wmmsg; wmmsg = wmmsg_next) {
                    wmmsg_next = wmmsg->next;
                    wmmsg->next = SDL_EventQ.wmmsg_free;
                    SDL_EventQ.wmmsg_free = wmmsg;
                }
                SDL_EventQ.wmmsg_used = NULL;
            }

            for (entry = SDL_EventQ.head; entry && (events == NULL || used < numevents); entry = next) {
                next = entry->next;
                type = entry->event.type;
                if (minType <= type && type <= maxType) {
                    if (events) {
                        events[used] = entry->event;
                        if (entry->event.type == SDL_SYSWMEVENT) {
                            /* We need to copy the wmmsg somewhere safe.
                               For now we'll guarantee it's valid at least until
                               the next call to SDL_PeepEvents()
                             */
                            if (SDL_EventQ.wmmsg_free) {
                                wmmsg = SDL_EventQ.wmmsg_free;
                                SDL_EventQ.wmmsg_free = wmmsg->next;
                            } else {
                                wmmsg = (SDL_SysWMEntry *)SDL_malloc(sizeof(*wmmsg));
                            }
                            wmmsg->msg = *entry->event.syswm.msg;
                            wmmsg->next = SDL_EventQ.wmmsg_used;
                            SDL_EventQ.wmmsg_used = wmmsg;
                            events[used].syswm.msg = &wmmsg->msg;
                        }

                        if (action == SDL_GETEVENT) {
                            SDL_CutEvent(entry);
                        }
                    }
                    if (type == SDL_POLLSENTINEL) {
                        /* Special handling for the sentinel event */
                        if (!include_sentinel) {
                            /* Skip it, we don't want to include it */
                            continue;
                        }
                        if (events == NULL || action != SDL_GETEVENT) {
                            ++sentinels_expected;
                        }
                        if (SDL_AtomicGet(&SDL_sentinel_pending) > sentinels_expected) {
                            /* Skip it, there's another one pending */
                            continue;
                        }
                    }
                    ++used;
                }
            }
        }
    }
    SDL_UnlockMutex(SDL_EventQ.lock);

    if (used > 0 && action == SDL_ADDEVENT) {
        SDL_SendWakeupEvent();
    }

    return used;
}
int SDL_PeepEvents(SDL_Event *events, int numevents, SDL_eventaction action,
                   Uint32 minType, Uint32 maxType)
{
    return SDL_PeepEventsInternal(events, numevents, action, minType, maxType, SDL_FALSE);
}

SDL_bool SDL_HasEvent(Uint32 type)
{
    return SDL_PeepEvents(NULL, 0, SDL_PEEKEVENT, type, type) > 0;
}

SDL_bool SDL_HasEvents(Uint32 minType, Uint32 maxType)
{
    return SDL_PeepEvents(NULL, 0, SDL_PEEKEVENT, minType, maxType) > 0;
}

void SDL_FlushEvent(Uint32 type)
{
    SDL_FlushEvents(type, type);
}

void SDL_FlushEvents(Uint32 minType, Uint32 maxType)
{
    SDL_EventEntry *entry, *next;
    Uint32 type;
    /* !!! FIXME: we need to manually SDL_free() the strings in TEXTINPUT and
       drag'n'drop events if we're flushing them without passing them to the
       app, but I don't know if this is the right place to do that. */

    /* Make sure the events are current */
#if 0
    /* Actually, we can't do this since we might be flushing while processing
       a resize event, and calling this might trigger further resize events.
    */
    SDL_PumpEvents();
#endif

    /* Lock the event queue */
    SDL_LockMutex(SDL_EventQ.lock);
    {
        /* Don't look after we've quit */
        if (!SDL_EventQ.active) {
            SDL_UnlockMutex(SDL_EventQ.lock);
            return;
        }
        for (entry = SDL_EventQ.head; entry; entry = next) {
            next = entry->next;
            type = entry->event.type;
            if (minType <= type && type <= maxType) {
                SDL_CutEvent(entry);
            }
        }
    }
    SDL_UnlockMutex(SDL_EventQ.lock);
}

/* Run the system dependent event loops */
static void SDL_PumpEventsInternal(SDL_bool push_sentinel)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();

    /* Release any keys held down from last frame */
    SDL_ReleaseAutoReleaseKeys();

    /* Get events from the video subsystem */
    if (_this) {
        _this->PumpEvents(_this);
    }

#ifndef SDL_JOYSTICK_DISABLED
    /* Check for joystick state change */
    if (SDL_update_joysticks) {
        SDL_JoystickUpdate();
    }
#endif

#ifndef SDL_SENSOR_DISABLED
    /* Check for sensor state change */
    if (SDL_update_sensors) {
        SDL_SensorUpdate();
    }
#endif

    SDL_SendPendingSignalEvents(); /* in case we had a signal handler fire, etc. */

    if (push_sentinel && SDL_GetEventState(SDL_POLLSENTINEL) == SDL_ENABLE) {
        SDL_Event sentinel;

        /* Make sure we don't already have a sentinel in the queue, and add one to the end */
        if (SDL_AtomicGet(&SDL_sentinel_pending) > 0) {
            SDL_PeepEventsInternal(&sentinel, 1, SDL_GETEVENT, SDL_POLLSENTINEL, SDL_POLLSENTINEL, SDL_TRUE);
        }

        SDL_zero(sentinel);
        sentinel.type = SDL_POLLSENTINEL;
        SDL_PushEvent(&sentinel);
    }
}

void SDL_PumpEvents(void)
{
    SDL_PumpEventsInternal(SDL_FALSE);
}

/* Public functions */

int SDL_PollEvent(SDL_Event *event)
{
    return SDL_WaitEventTimeout(event, 0);
}

static SDL_bool SDL_events_need_periodic_poll(void)
{
    SDL_bool need_periodic_poll = SDL_FALSE;

#ifndef SDL_JOYSTICK_DISABLED
    need_periodic_poll =
        SDL_WasInit(SDL_INIT_JOYSTICK) && SDL_update_joysticks;
#endif

#ifndef SDL_SENSOR_DISABLED
    need_periodic_poll = need_periodic_poll ||
                         (SDL_WasInit(SDL_INIT_SENSOR) && SDL_update_sensors);
#endif

    return need_periodic_poll;
}

static int SDL_WaitEventTimeout_Device(_THIS, SDL_Window *wakeup_window, SDL_Event *event, Uint32 start, int timeout)
{
    int loop_timeout = timeout;
    SDL_bool need_periodic_poll = SDL_events_need_periodic_poll();

    for (;;) {
        /* Pump events on entry and each time we wake to ensure:
           a) All pending events are batch processed after waking up from a wait
           b) Waiting can be completely skipped if events are already available to be pumped
           c) Periodic processing that takes place in some platform PumpEvents() functions happens
           d) Signals received in WaitEventTimeout() are turned into SDL events
        */
        int status;
        SDL_PumpEventsInternal(SDL_TRUE);

        SDL_LockMutex(_this->wakeup_lock);
        {
            status = SDL_PeepEvents(event, 1, SDL_GETEVENT, SDL_FIRSTEVENT, SDL_LASTEVENT);
            /* If status == 0 we are going to block so wakeup will be needed. */
            if (status == 0) {
                _this->wakeup_window = wakeup_window;
            } else {
                _this->wakeup_window = NULL;
            }
        }
        SDL_UnlockMutex(_this->wakeup_lock);

        if (status < 0) {
            /* Got an error: return */
            break;
        }
        if (status > 0) {
            /* There is an event, we can return. */
            return 1;
        }
        /* No events found in the queue, call WaitEventTimeout to wait for an event. */
        if (timeout > 0) {
            Uint32 elapsed = SDL_GetTicks() - start;
            if (elapsed >= (Uint32)timeout) {
                /* Set wakeup_window to NULL without holding the lock. */
                _this->wakeup_window = NULL;
                return 0;
            }
            loop_timeout = (int)((Uint32)timeout - elapsed);
        }
        if (need_periodic_poll) {
            if (loop_timeout >= 0) {
                loop_timeout = SDL_min(loop_timeout, PERIODIC_POLL_INTERVAL_MS);
            } else {
                loop_timeout = PERIODIC_POLL_INTERVAL_MS;
            }
        }
        status = _this->WaitEventTimeout(_this, loop_timeout);
        /* Set wakeup_window to NULL without holding the lock. */
        _this->wakeup_window = NULL;
        if (status == 0 && need_periodic_poll && loop_timeout == PERIODIC_POLL_INTERVAL_MS) {
            /* We may have woken up to poll. Try again */
            continue;
        } else if (status <= 0) {
            /* There is either an error or the timeout is elapsed: return */
            return status;
        }
        /* An event was found and pumped into the SDL events queue. Continue the loop
          to let SDL_PeepEvents pick it up .*/
    }
    return 0;
}

static SDL_bool SDL_events_need_polling(void)
{
    SDL_bool need_polling = SDL_FALSE;

#ifndef SDL_JOYSTICK_DISABLED
    need_polling =
        SDL_WasInit(SDL_INIT_JOYSTICK) &&
        SDL_update_joysticks &&
        (SDL_NumJoysticks() > 0);
#endif

#ifndef SDL_SENSOR_DISABLED
    need_polling = need_polling ||
                   (SDL_WasInit(SDL_INIT_SENSOR) && SDL_update_sensors && (SDL_NumSensors() > 0));
#endif

    return need_polling;
}

static SDL_Window *SDL_find_active_window(SDL_VideoDevice *_this)
{
    SDL_Window *window;
    for (window = _this->windows; window; window = window->next) {
        if (!window->is_destroying) {
            return window;
        }
    }
    return NULL;
}

int SDL_WaitEvent(SDL_Event *event)
{
    return SDL_WaitEventTimeout(event, -1);
}

int SDL_WaitEventTimeout(SDL_Event *event, int timeout)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();
    SDL_Window *wakeup_window;
    Uint32 start, expiration;
    SDL_bool include_sentinel = (timeout == 0) ? SDL_TRUE : SDL_FALSE;
    int result;

    /* If there isn't a poll sentinel event pending, pump events and add one */
    if (SDL_AtomicGet(&SDL_sentinel_pending) == 0) {
        SDL_PumpEventsInternal(SDL_TRUE);
    }

    /* First check for existing events */
    result = SDL_PeepEventsInternal(event, 1, SDL_GETEVENT, SDL_FIRSTEVENT, SDL_LASTEVENT, include_sentinel);
    if (result < 0) {
        return 0;
    }
    if (include_sentinel) {
        if (event) {
            if (event->type == SDL_POLLSENTINEL) {
                /* Reached the end of a poll cycle, and not willing to wait */
                return 0;
            }
        } else {
            /* Need to peek the next event to check for sentinel */
            SDL_Event dummy;

            if (SDL_PeepEventsInternal(&dummy, 1, SDL_PEEKEVENT, SDL_FIRSTEVENT, SDL_LASTEVENT, SDL_TRUE) &&
                dummy.type == SDL_POLLSENTINEL) {
                SDL_PeepEventsInternal(&dummy, 1, SDL_GETEVENT, SDL_POLLSENTINEL, SDL_POLLSENTINEL, SDL_TRUE);
                /* Reached the end of a poll cycle, and not willing to wait */
                return 0;
            }
        }
    }
    if (result == 0) {
        if (timeout == 0) {
            /* No events available, and not willing to wait */
            return 0;
        }
    } else {
        /* Has existing events */
        return 1;
    }
    /* We should have completely handled timeout == 0 above */
    SDL_assert(timeout != 0);

    if (timeout > 0) {
        start = SDL_GetTicks();
        expiration = start + timeout;
    } else {
        start = 0;
        expiration = 0;
    }

    if (_this && _this->WaitEventTimeout && _this->SendWakeupEvent && !SDL_events_need_polling()) {
        /* Look if a shown window is available to send the wakeup event. */
        wakeup_window = SDL_find_active_window(_this);
        if (wakeup_window) {
            int status = SDL_WaitEventTimeout_Device(_this, wakeup_window, event, start, timeout);

            /* There may be implementation-defined conditions where the backend cannot
               reliably wait for the next event. If that happens, fall back to polling. */
            if (status >= 0) {
                return status;
            }
        }
    }

    for (;;) {
        SDL_PumpEventsInternal(SDL_TRUE);
        switch (SDL_PeepEvents(event, 1, SDL_GETEVENT, SDL_FIRSTEVENT, SDL_LASTEVENT)) {
        case -1:
            return 0;
        case 0:
            if (timeout > 0 && SDL_TICKS_PASSED(SDL_GetTicks(), expiration)) {
                /* Timeout expired and no events */
                return 0;
            }
            SDL_Delay(1);
            break;
        default:
            /* Has events */
            return 1;
        }
    }
}

int SDL_PushEvent(SDL_Event *event)
{
    event->common.timestamp = SDL_GetTicks();

    if (SDL_EventOK.callback || SDL_event_watchers_count > 0) {
        SDL_LockMutex(SDL_event_watchers_lock);
        {
            if (SDL_EventOK.callback && !SDL_EventOK.callback(SDL_EventOK.userdata, event)) {
                SDL_UnlockMutex(SDL_event_watchers_lock);
                return 0;
            }

            if (SDL_event_watchers_count > 0) {
                /* Make sure we only dispatch the current watcher list */
                int i, event_watchers_count = SDL_event_watchers_count;

                SDL_event_watchers_dispatching = SDL_TRUE;
                for (i = 0; i < event_watchers_count; ++i) {
                    if (!SDL_event_watchers[i].removed) {
                        SDL_event_watchers[i].callback(SDL_event_watchers[i].userdata, event);
                    }
                }
                SDL_event_watchers_dispatching = SDL_FALSE;

                if (SDL_event_watchers_removed) {
                    for (i = SDL_event_watchers_count; i--;) {
                        if (SDL_event_watchers[i].removed) {
                            --SDL_event_watchers_count;
                            if (i < SDL_event_watchers_count) {
                                SDL_memmove(&SDL_event_watchers[i], &SDL_event_watchers[i + 1], (SDL_event_watchers_count - i) * sizeof(SDL_event_watchers[i]));
                            }
                        }
                    }
                    SDL_event_watchers_removed = SDL_FALSE;
                }
            }
        }
        SDL_UnlockMutex(SDL_event_watchers_lock);
    }

    if (SDL_PeepEvents(event, 1, SDL_ADDEVENT, 0, 0) <= 0) {
        return -1;
    }

    SDL_GestureProcessEvent(event);

    return 1;
}

void SDL_SetEventFilter(SDL_EventFilter filter, void *userdata)
{
    SDL_LockMutex(SDL_event_watchers_lock);
    {
        /* Set filter and discard pending events */
        SDL_EventOK.callback = filter;
        SDL_EventOK.userdata = userdata;
        SDL_FlushEvents(SDL_FIRSTEVENT, SDL_LASTEVENT);
    }
    SDL_UnlockMutex(SDL_event_watchers_lock);
}

SDL_bool SDL_GetEventFilter(SDL_EventFilter *filter, void **userdata)
{
    SDL_EventWatcher event_ok;

    SDL_LockMutex(SDL_event_watchers_lock);
    {
        event_ok = SDL_EventOK;
    }
    SDL_UnlockMutex(SDL_event_watchers_lock);

    if (filter) {
        *filter = event_ok.callback;
    }
    if (userdata) {
        *userdata = event_ok.userdata;
    }
    return event_ok.callback ? SDL_TRUE : SDL_FALSE;
}

void SDL_AddEventWatch(SDL_EventFilter filter, void *userdata)
{
    SDL_LockMutex(SDL_event_watchers_lock);
    {
        SDL_EventWatcher *event_watchers;

        event_watchers = SDL_realloc(SDL_event_watchers, (SDL_event_watchers_count + 1) * sizeof(*event_watchers));
        if (event_watchers) {
            SDL_EventWatcher *watcher;

            SDL_event_watchers = event_watchers;
            watcher = &SDL_event_watchers[SDL_event_watchers_count];
            watcher->callback = filter;
            watcher->userdata = userdata;
            watcher->removed = SDL_FALSE;
            ++SDL_event_watchers_count;
        }
    }
    SDL_UnlockMutex(SDL_event_watchers_lock);
}

void SDL_DelEventWatch(SDL_EventFilter filter, void *userdata)
{
    SDL_LockMutex(SDL_event_watchers_lock);
    {
        int i;

        for (i = 0; i < SDL_event_watchers_count; ++i) {
            if (SDL_event_watchers[i].callback == filter && SDL_event_watchers[i].userdata == userdata) {
                if (SDL_event_watchers_dispatching) {
                    SDL_event_watchers[i].removed = SDL_TRUE;
                    SDL_event_watchers_removed = SDL_TRUE;
                } else {
                    --SDL_event_watchers_count;
                    if (i < SDL_event_watchers_count) {
                        SDL_memmove(&SDL_event_watchers[i], &SDL_event_watchers[i + 1], (SDL_event_watchers_count - i) * sizeof(SDL_event_watchers[i]));
                    }
                }
                break;
            }
        }
    }
    SDL_UnlockMutex(SDL_event_watchers_lock);
}

void SDL_FilterEvents(SDL_EventFilter filter, void *userdata)
{
    SDL_LockMutex(SDL_EventQ.lock);
    {
        SDL_EventEntry *entry, *next;
        for (entry = SDL_EventQ.head; entry; entry = next) {
            next = entry->next;
            if (!filter(userdata, &entry->event)) {
                SDL_CutEvent(entry);
            }
        }
    }
    SDL_UnlockMutex(SDL_EventQ.lock);
}

Uint8 SDL_EventState(Uint32 type, int state)
{
    const SDL_bool isde = (state == SDL_DISABLE) || (state == SDL_ENABLE);
    Uint8 current_state;
    Uint8 hi = ((type >> 8) & 0xff);
    Uint8 lo = (type & 0xff);

    if (SDL_disabled_events[hi] &&
        (SDL_disabled_events[hi]->bits[lo / 32] & (1 << (lo & 31)))) {
        current_state = SDL_DISABLE;
    } else {
        current_state = SDL_ENABLE;
    }

    if (isde && state != current_state) {
        if (state == SDL_DISABLE) {
            /* Disable this event type and discard pending events */
            if (!SDL_disabled_events[hi]) {
                SDL_disabled_events[hi] = (SDL_DisabledEventBlock *)SDL_calloc(1, sizeof(SDL_DisabledEventBlock));
            }
            /* Out of memory, nothing we can do... */
            if (SDL_disabled_events[hi]) {
                SDL_disabled_events[hi]->bits[lo / 32] |= (1 << (lo & 31));
                SDL_FlushEvent(type);
            }
        } else { // state == SDL_ENABLE
            SDL_disabled_events[hi]->bits[lo / 32] &= ~(1 << (lo & 31));
        }

#ifndef SDL_JOYSTICK_DISABLED
        SDL_CalculateShouldUpdateJoysticks(SDL_GetHintBoolean(SDL_HINT_AUTO_UPDATE_JOYSTICKS, SDL_TRUE));
#endif
#ifndef SDL_SENSOR_DISABLED
        SDL_CalculateShouldUpdateSensors(SDL_GetHintBoolean(SDL_HINT_AUTO_UPDATE_SENSORS, SDL_TRUE));
#endif
    }

    /* turn off drag'n'drop support if we've disabled the events.
       This might change some UI details at the OS level. */
    if (isde && ((type == SDL_DROPFILE) || (type == SDL_DROPTEXT))) {
        SDL_ToggleDragAndDropSupport();
    }

    return current_state;
}

Uint32 SDL_RegisterEvents(int numevents)
{
    Uint32 event_base;

    if ((numevents > 0) && (SDL_userevents + numevents <= SDL_LASTEVENT)) {
        event_base = SDL_userevents;
        SDL_userevents += numevents;
    } else {
        event_base = (Uint32)-1;
    }
    return event_base;
}

int SDL_SendAppEvent(SDL_EventType eventType)
{
    int posted;

    posted = 0;
    if (SDL_GetEventState(eventType) == SDL_ENABLE) {
        SDL_Event event;
        event.type = eventType;
        posted = (SDL_PushEvent(&event) > 0);
    }
    return posted;
}

int SDL_SendSysWMEvent(SDL_SysWMmsg *message)
{
    int posted;

    posted = 0;
    if (SDL_GetEventState(SDL_SYSWMEVENT) == SDL_ENABLE) {
        SDL_Event event;
        SDL_memset(&event, 0, sizeof(event));
        event.type = SDL_SYSWMEVENT;
        event.syswm.msg = message;
        posted = (SDL_PushEvent(&event) > 0);
    }
    /* Update internal event state */
    return posted;
}

int SDL_SendKeymapChangedEvent(void)
{
    return SDL_SendAppEvent(SDL_KEYMAPCHANGED);
}

int SDL_SendLocaleChangedEvent(void)
{
    return SDL_SendAppEvent(SDL_LOCALECHANGED);
}

int SDL_EventsInit(void)
{
#ifndef SDL_JOYSTICK_DISABLED
    SDL_AddHintCallback(SDL_HINT_AUTO_UPDATE_JOYSTICKS, SDL_AutoUpdateJoysticksChanged, NULL);
#endif
#ifndef SDL_SENSOR_DISABLED
    SDL_AddHintCallback(SDL_HINT_AUTO_UPDATE_SENSORS, SDL_AutoUpdateSensorsChanged, NULL);
#endif
    SDL_AddHintCallback(SDL_HINT_EVENT_LOGGING, SDL_EventLoggingChanged, NULL);
    SDL_AddHintCallback(SDL_HINT_POLL_SENTINEL, SDL_PollSentinelChanged, NULL);
    if (SDL_StartEventLoop() < 0) {
        SDL_DelHintCallback(SDL_HINT_EVENT_LOGGING, SDL_EventLoggingChanged, NULL);
        return -1;
    }

    SDL_QuitInit();

    return 0;
}

void SDL_EventsQuit(void)
{
    SDL_QuitQuit();
    SDL_StopEventLoop();
    SDL_DelHintCallback(SDL_HINT_POLL_SENTINEL, SDL_PollSentinelChanged, NULL);
    SDL_DelHintCallback(SDL_HINT_EVENT_LOGGING, SDL_EventLoggingChanged, NULL);
#ifndef SDL_JOYSTICK_DISABLED
    SDL_DelHintCallback(SDL_HINT_AUTO_UPDATE_JOYSTICKS, SDL_AutoUpdateJoysticksChanged, NULL);
#endif
#ifndef SDL_SENSOR_DISABLED
    SDL_DelHintCallback(SDL_HINT_AUTO_UPDATE_SENSORS, SDL_AutoUpdateSensorsChanged, NULL);
#endif
}

/* vi: set ts=4 sw=4 expandtab: */
