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

#include "SDL.h"
#include "SDL_error.h"
#include "SDL_haptic.h"
#include "../SDL_syshaptic.h"

#ifdef SDL_HAPTIC_XINPUT

#include "SDL_hints.h"
#include "SDL_timer.h"
#include "SDL_windowshaptic_c.h"
#include "SDL_xinputhaptic_c.h"
#include "../../core/windows/SDL_xinput.h"
#include "../../joystick/windows/SDL_windowsjoystick_c.h"
#include "../../thread/SDL_systhread.h"

/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Internal stuff.
 */
static SDL_bool loaded_xinput = SDL_FALSE;

int SDL_XINPUT_HapticInit(void)
{
    if (SDL_GetHintBoolean(SDL_HINT_XINPUT_ENABLED, SDL_TRUE)) {
        loaded_xinput = (WIN_LoadXInputDLL() == 0) ? SDL_TRUE : SDL_FALSE;
    }

    /* If the joystick subsystem is active, it will manage adding XInput haptic devices */
    if (loaded_xinput && !SDL_WasInit(SDL_INIT_JOYSTICK)) {
        DWORD i;
        for (i = 0; i < XUSER_MAX_COUNT; i++) {
            SDL_XINPUT_HapticMaybeAddDevice(i);
        }
    }
    return 0;
}

int SDL_XINPUT_HapticMaybeAddDevice(const DWORD dwUserid)
{
    const Uint8 userid = (Uint8)dwUserid;
    SDL_hapticlist_item *item;
    XINPUT_VIBRATION state;

    if ((!loaded_xinput) || (dwUserid >= XUSER_MAX_COUNT)) {
        return -1;
    }

    /* Make sure we don't already have it */
    for (item = SDL_hapticlist; item; item = item->next) {
        if (item->bXInputHaptic && item->userid == userid) {
            return -1; /* Already added */
        }
    }

    SDL_zero(state);
    if (XINPUTSETSTATE(dwUserid, &state) != ERROR_SUCCESS) {
        return -1; /* no force feedback on this device. */
    }

    item = (SDL_hapticlist_item *)SDL_malloc(sizeof(SDL_hapticlist_item));
    if (!item) {
        return SDL_OutOfMemory();
    }

    SDL_zerop(item);

    /* !!! FIXME: I'm not bothering to query for a real name right now (can we even?) */
    {
        char buf[64];
        (void)SDL_snprintf(buf, sizeof(buf), "XInput Controller #%u", userid + 1);
        item->name = SDL_strdup(buf);
    }

    if (!item->name) {
        SDL_free(item);
        return -1;
    }

    /* Copy the instance over, useful for creating devices. */
    item->bXInputHaptic = SDL_TRUE;
    item->userid = userid;

    return SDL_SYS_AddHapticDevice(item);
}

int SDL_XINPUT_HapticMaybeRemoveDevice(const DWORD dwUserid)
{
    const Uint8 userid = (Uint8)dwUserid;
    SDL_hapticlist_item *item;
    SDL_hapticlist_item *prev = NULL;

    if ((!loaded_xinput) || (dwUserid >= XUSER_MAX_COUNT)) {
        return -1;
    }

    for (item = SDL_hapticlist; item; item = item->next) {
        if (item->bXInputHaptic && item->userid == userid) {
            /* found it, remove it. */
            return SDL_SYS_RemoveHapticDevice(prev, item);
        }
        prev = item;
    }
    return -1;
}

/* !!! FIXME: this is a hack, remove this later. */
/* Since XInput doesn't offer a way to vibrate for X time, we hook into
 *  SDL_PumpEvents() to check if it's time to stop vibrating with some
 *  frequency.
 * In practice, this works for 99% of use cases. But in an ideal world,
 *  we do this in a separate thread so that:
 *    - we aren't bound to when the app chooses to pump the event queue.
 *    - we aren't adding more polling to the event queue
 *    - we can emulate all the haptic effects correctly (start on a delay,
 *      mix multiple effects, etc).
 *
 * Mostly, this is here to get rumbling to work, and all the other features
 *  are absent in the XInput path for now.  :(
 */
static int SDLCALL SDL_RunXInputHaptic(void *arg)
{
    struct haptic_hwdata *hwdata = (struct haptic_hwdata *)arg;

    while (!SDL_AtomicGet(&hwdata->stopThread)) {
        SDL_Delay(50);
        SDL_LockMutex(hwdata->mutex);
        /* If we're currently running and need to stop... */
        if (hwdata->stopTicks) {
            if ((hwdata->stopTicks != SDL_HAPTIC_INFINITY) && SDL_TICKS_PASSED(SDL_GetTicks(), hwdata->stopTicks)) {
                XINPUT_VIBRATION vibration = { 0, 0 };
                hwdata->stopTicks = 0;
                XINPUTSETSTATE(hwdata->userid, &vibration);
            }
        }
        SDL_UnlockMutex(hwdata->mutex);
    }

    return 0;
}

static int SDL_XINPUT_HapticOpenFromUserIndex(SDL_Haptic *haptic, const Uint8 userid)
{
    char threadName[32];
    XINPUT_VIBRATION vibration = { 0, 0 }; /* stop any current vibration */
    XINPUTSETSTATE(userid, &vibration);

    haptic->supported = SDL_HAPTIC_LEFTRIGHT;

    haptic->neffects = 1;
    haptic->nplaying = 1;

    /* Prepare effects memory. */
    haptic->effects = (struct haptic_effect *)
        SDL_malloc(sizeof(struct haptic_effect) * haptic->neffects);
    if (!haptic->effects) {
        return SDL_OutOfMemory();
    }
    /* Clear the memory */
    SDL_memset(haptic->effects, 0,
               sizeof(struct haptic_effect) * haptic->neffects);

    haptic->hwdata = (struct haptic_hwdata *)SDL_malloc(sizeof(*haptic->hwdata));
    if (!haptic->hwdata) {
        SDL_free(haptic->effects);
        haptic->effects = NULL;
        return SDL_OutOfMemory();
    }
    SDL_memset(haptic->hwdata, 0, sizeof(*haptic->hwdata));

    haptic->hwdata->bXInputHaptic = 1;
    haptic->hwdata->userid = userid;

    haptic->hwdata->mutex = SDL_CreateMutex();
    if (!haptic->hwdata->mutex) {
        SDL_free(haptic->effects);
        SDL_free(haptic->hwdata);
        haptic->effects = NULL;
        return SDL_SetError("Couldn't create XInput haptic mutex");
    }

    (void)SDL_snprintf(threadName, sizeof(threadName), "SDLXInputDev%d", userid);
    haptic->hwdata->thread = SDL_CreateThreadInternal(SDL_RunXInputHaptic, threadName, 64 * 1024, haptic->hwdata);

    if (!haptic->hwdata->thread) {
        SDL_DestroyMutex(haptic->hwdata->mutex);
        SDL_free(haptic->effects);
        SDL_free(haptic->hwdata);
        haptic->effects = NULL;
        return SDL_SetError("Couldn't create XInput haptic thread");
    }

    return 0;
}

int SDL_XINPUT_HapticOpen(SDL_Haptic *haptic, SDL_hapticlist_item *item)
{
    return SDL_XINPUT_HapticOpenFromUserIndex(haptic, item->userid);
}

int SDL_XINPUT_JoystickSameHaptic(SDL_Haptic *haptic, SDL_Joystick *joystick)
{
    return haptic->hwdata->userid == joystick->hwdata->userid;
}

int SDL_XINPUT_HapticOpenFromJoystick(SDL_Haptic *haptic, SDL_Joystick *joystick)
{
    SDL_hapticlist_item *item;
    int index = 0;

    /* Since it comes from a joystick we have to try to match it with a haptic device on our haptic list. */
    for (item = SDL_hapticlist; item; item = item->next) {
        if (item->bXInputHaptic && item->userid == joystick->hwdata->userid) {
            haptic->index = index;
            return SDL_XINPUT_HapticOpenFromUserIndex(haptic, joystick->hwdata->userid);
        }
        ++index;
    }

    return SDL_SetError("Couldn't find joystick in haptic device list");
}

void SDL_XINPUT_HapticClose(SDL_Haptic *haptic)
{
    SDL_AtomicSet(&haptic->hwdata->stopThread, 1);
    SDL_WaitThread(haptic->hwdata->thread, NULL);
    SDL_DestroyMutex(haptic->hwdata->mutex);
}

void SDL_XINPUT_HapticQuit(void)
{
    if (loaded_xinput) {
        WIN_UnloadXInputDLL();
        loaded_xinput = SDL_FALSE;
    }
}

int SDL_XINPUT_HapticNewEffect(SDL_Haptic *haptic, struct haptic_effect *effect, SDL_HapticEffect *base)
{
    SDL_assert(base->type == SDL_HAPTIC_LEFTRIGHT); /* should catch this at higher level */
    return SDL_XINPUT_HapticUpdateEffect(haptic, effect, base);
}

int SDL_XINPUT_HapticUpdateEffect(SDL_Haptic *haptic, struct haptic_effect *effect, SDL_HapticEffect *data)
{
    XINPUT_VIBRATION *vib = &effect->hweffect->vibration;
    SDL_assert(data->type == SDL_HAPTIC_LEFTRIGHT);
    /* SDL_HapticEffect has max magnitude of 32767, XInput expects 65535 max, so multiply */
    vib->wLeftMotorSpeed = data->leftright.large_magnitude * 2;
    vib->wRightMotorSpeed = data->leftright.small_magnitude * 2;
    SDL_LockMutex(haptic->hwdata->mutex);
    if (haptic->hwdata->stopTicks) { /* running right now? Update it. */
        XINPUTSETSTATE(haptic->hwdata->userid, vib);
    }
    SDL_UnlockMutex(haptic->hwdata->mutex);
    return 0;
}

int SDL_XINPUT_HapticRunEffect(SDL_Haptic *haptic, struct haptic_effect *effect, Uint32 iterations)
{
    XINPUT_VIBRATION *vib = &effect->hweffect->vibration;
    SDL_assert(effect->effect.type == SDL_HAPTIC_LEFTRIGHT); /* should catch this at higher level */
    SDL_LockMutex(haptic->hwdata->mutex);
    if (effect->effect.leftright.length == SDL_HAPTIC_INFINITY || iterations == SDL_HAPTIC_INFINITY) {
        haptic->hwdata->stopTicks = SDL_HAPTIC_INFINITY;
    } else if ((!effect->effect.leftright.length) || (!iterations)) {
        /* do nothing. Effect runs for zero milliseconds. */
    } else {
        haptic->hwdata->stopTicks = SDL_GetTicks() + (effect->effect.leftright.length * iterations);
        if ((haptic->hwdata->stopTicks == SDL_HAPTIC_INFINITY) || (haptic->hwdata->stopTicks == 0)) {
            haptic->hwdata->stopTicks = 1; /* fix edge cases. */
        }
    }
    SDL_UnlockMutex(haptic->hwdata->mutex);
    return (XINPUTSETSTATE(haptic->hwdata->userid, vib) == ERROR_SUCCESS) ? 0 : -1;
}

int SDL_XINPUT_HapticStopEffect(SDL_Haptic *haptic, struct haptic_effect *effect)
{
    XINPUT_VIBRATION vibration = { 0, 0 };
    SDL_LockMutex(haptic->hwdata->mutex);
    haptic->hwdata->stopTicks = 0;
    SDL_UnlockMutex(haptic->hwdata->mutex);
    return (XINPUTSETSTATE(haptic->hwdata->userid, &vibration) == ERROR_SUCCESS) ? 0 : -1;
}

void SDL_XINPUT_HapticDestroyEffect(SDL_Haptic *haptic, struct haptic_effect *effect)
{
    SDL_XINPUT_HapticStopEffect(haptic, effect);
}

int SDL_XINPUT_HapticGetEffectStatus(SDL_Haptic *haptic, struct haptic_effect *effect)
{
    return SDL_Unsupported();
}

int SDL_XINPUT_HapticSetGain(SDL_Haptic *haptic, int gain)
{
    return SDL_Unsupported();
}

int SDL_XINPUT_HapticSetAutocenter(SDL_Haptic *haptic, int autocenter)
{
    return SDL_Unsupported();
}

int SDL_XINPUT_HapticPause(SDL_Haptic *haptic)
{
    return SDL_Unsupported();
}

int SDL_XINPUT_HapticUnpause(SDL_Haptic *haptic)
{
    return SDL_Unsupported();
}

int SDL_XINPUT_HapticStopAll(SDL_Haptic *haptic)
{
    XINPUT_VIBRATION vibration = { 0, 0 };
    SDL_LockMutex(haptic->hwdata->mutex);
    haptic->hwdata->stopTicks = 0;
    SDL_UnlockMutex(haptic->hwdata->mutex);
    return (XINPUTSETSTATE(haptic->hwdata->userid, &vibration) == ERROR_SUCCESS) ? 0 : -1;
}

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif

#else /* !SDL_HAPTIC_XINPUT */

#include "../../core/windows/SDL_windows.h"

typedef struct SDL_hapticlist_item SDL_hapticlist_item;

int SDL_XINPUT_HapticInit(void)
{
    return 0;
}

int SDL_XINPUT_HapticMaybeAddDevice(const DWORD dwUserid)
{
    return SDL_Unsupported();
}

int SDL_XINPUT_HapticMaybeRemoveDevice(const DWORD dwUserid)
{
    return SDL_Unsupported();
}

int SDL_XINPUT_HapticOpen(SDL_Haptic *haptic, SDL_hapticlist_item *item)
{
    return SDL_Unsupported();
}

int SDL_XINPUT_JoystickSameHaptic(SDL_Haptic *haptic, SDL_Joystick *joystick)
{
    return SDL_Unsupported();
}

int SDL_XINPUT_HapticOpenFromJoystick(SDL_Haptic *haptic, SDL_Joystick *joystick)
{
    return SDL_Unsupported();
}

void SDL_XINPUT_HapticClose(SDL_Haptic *haptic)
{
}

void SDL_XINPUT_HapticQuit(void)
{
}

int SDL_XINPUT_HapticNewEffect(SDL_Haptic *haptic, struct haptic_effect *effect, SDL_HapticEffect *base)
{
    return SDL_Unsupported();
}

int SDL_XINPUT_HapticUpdateEffect(SDL_Haptic *haptic, struct haptic_effect *effect, SDL_HapticEffect *data)
{
    return SDL_Unsupported();
}

int SDL_XINPUT_HapticRunEffect(SDL_Haptic *haptic, struct haptic_effect *effect, Uint32 iterations)
{
    return SDL_Unsupported();
}

int SDL_XINPUT_HapticStopEffect(SDL_Haptic *haptic, struct haptic_effect *effect)
{
    return SDL_Unsupported();
}

void SDL_XINPUT_HapticDestroyEffect(SDL_Haptic *haptic, struct haptic_effect *effect)
{
}

int SDL_XINPUT_HapticGetEffectStatus(SDL_Haptic *haptic, struct haptic_effect *effect)
{
    return SDL_Unsupported();
}

int SDL_XINPUT_HapticSetGain(SDL_Haptic *haptic, int gain)
{
    return SDL_Unsupported();
}

int SDL_XINPUT_HapticSetAutocenter(SDL_Haptic *haptic, int autocenter)
{
    return SDL_Unsupported();
}

int SDL_XINPUT_HapticPause(SDL_Haptic *haptic)
{
    return SDL_Unsupported();
}

int SDL_XINPUT_HapticUnpause(SDL_Haptic *haptic)
{
    return SDL_Unsupported();
}

int SDL_XINPUT_HapticStopAll(SDL_Haptic *haptic)
{
    return SDL_Unsupported();
}

#endif /* SDL_HAPTIC_XINPUT */

/* vi: set ts=4 sw=4 expandtab: */
