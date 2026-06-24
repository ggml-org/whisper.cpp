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

#ifdef SDL_HAPTIC_ANDROID

#include "SDL_timer.h"
#include "SDL_syshaptic_c.h"
#include "../SDL_syshaptic.h"
#include "SDL_haptic.h"
#include "../../core/android/SDL_android.h"
#include "SDL_joystick.h"
#include "../../joystick/SDL_sysjoystick.h"     /* For the real SDL_Joystick */
#include "../../joystick/android/SDL_sysjoystick_c.h"     /* For joystick hwdata */


typedef struct SDL_hapticlist_item
{
    int device_id;
    char *name;
    SDL_Haptic *haptic;
    struct SDL_hapticlist_item *next;
} SDL_hapticlist_item;

static SDL_hapticlist_item *SDL_hapticlist = NULL;
static SDL_hapticlist_item *SDL_hapticlist_tail = NULL;
static int numhaptics = 0;

int SDL_SYS_HapticInit(void)
{
    /* Support for device connect/disconnect is API >= 16 only,
     * so we poll every three seconds
     * Ref: http://developer.android.com/reference/android/hardware/input/InputManager.InputDeviceListener.html
     */
    static Uint32 timeout = 0;
    if (SDL_TICKS_PASSED(SDL_GetTicks(), timeout)) {
        timeout = SDL_GetTicks() + 3000;
        Android_JNI_PollHapticDevices();
    }
    return numhaptics;
}

int SDL_SYS_NumHaptics(void)
{
    return numhaptics;
}

static SDL_hapticlist_item *HapticByOrder(int index)
{
    SDL_hapticlist_item *item = SDL_hapticlist;
    if ((index < 0) || (index >= numhaptics)) {
        return NULL;
    }
    while (index > 0) {
        SDL_assert(item != NULL);
        --index;
        item = item->next;
    }
    return item;
}

static SDL_hapticlist_item *HapticByDevId(int device_id)
{
    SDL_hapticlist_item *item;
    for (item = SDL_hapticlist; item; item = item->next) {
        if (device_id == item->device_id) {
            /*SDL_Log("=+=+=+=+=+= HapticByDevId id [%d]", device_id);*/
            return item;
        }
    }
    return NULL;
}

const char *SDL_SYS_HapticName(int index)
{
    SDL_hapticlist_item *item = HapticByOrder(index);
    if (!item) {
        SDL_SetError("No such device");
        return NULL;
    }
    return item->name;
}

static SDL_hapticlist_item *OpenHaptic(SDL_Haptic *haptic, SDL_hapticlist_item *item)
{
    if (!item) {
        SDL_SetError("No such device");
        return NULL;
    }
    if (item->haptic) {
        SDL_SetError("Haptic already opened");
        return NULL;
    }

    haptic->hwdata = (struct haptic_hwdata *)item;
    item->haptic = haptic;

    haptic->supported = SDL_HAPTIC_LEFTRIGHT;
    haptic->neffects = 1;
    haptic->nplaying = haptic->neffects;
    haptic->effects = (struct haptic_effect *)SDL_malloc(sizeof(struct haptic_effect) * haptic->neffects);
    if (!haptic->effects) {
        SDL_OutOfMemory();
        return NULL;
    }
    SDL_memset(haptic->effects, 0, sizeof(struct haptic_effect) * haptic->neffects);
    return item;
}

static SDL_hapticlist_item *OpenHapticByOrder(SDL_Haptic *haptic, int index)
{
    return OpenHaptic(haptic, HapticByOrder(index));
}

static SDL_hapticlist_item *OpenHapticByDevId(SDL_Haptic *haptic, int device_id)
{
    return OpenHaptic(haptic, HapticByDevId(device_id));
}

int SDL_SYS_HapticOpen(SDL_Haptic *haptic)
{
    return OpenHapticByOrder(haptic, haptic->index) == NULL ? -1 : 0;
}

int SDL_SYS_HapticMouse(void)
{
    return -1;
}

int SDL_SYS_JoystickIsHaptic(SDL_Joystick *joystick)
{
    SDL_hapticlist_item *item;
    item = HapticByDevId(((joystick_hwdata *)joystick->hwdata)->device_id);
    return (item) ? 1 : 0;
}

int SDL_SYS_HapticOpenFromJoystick(SDL_Haptic *haptic, SDL_Joystick *joystick)
{
    return OpenHapticByDevId(haptic, ((joystick_hwdata *)joystick->hwdata)->device_id) == NULL ? -1 : 0;
}

int SDL_SYS_JoystickSameHaptic(SDL_Haptic *haptic, SDL_Joystick *joystick)
{
    return ((SDL_hapticlist_item *)haptic->hwdata)->device_id == ((joystick_hwdata *)joystick->hwdata)->device_id ? 1 : 0;
}

void SDL_SYS_HapticClose(SDL_Haptic *haptic)
{
    ((SDL_hapticlist_item *)haptic->hwdata)->haptic = NULL;
    haptic->hwdata = NULL;
}

void SDL_SYS_HapticQuit(void)
{
/* We don't have any way to scan for joysticks (and their vibrators) at init, so don't wipe the list
 * of joysticks here in case this is a reinit.
 */
#if 0
    SDL_hapticlist_item *item = NULL;
    SDL_hapticlist_item *next = NULL;

    for (item = SDL_hapticlist; item; item = next) {
        next = item->next;
        SDL_free(item);
    }

    SDL_hapticlist = SDL_hapticlist_tail = NULL;
    numhaptics = 0;
    return;
#endif
}

int SDL_SYS_HapticNewEffect(SDL_Haptic *haptic,
                            struct haptic_effect *effect, SDL_HapticEffect *base)
{
    return 0;
}

int SDL_SYS_HapticUpdateEffect(SDL_Haptic *haptic,
                               struct haptic_effect *effect,
                               SDL_HapticEffect *data)
{
    return 0;
}

int SDL_SYS_HapticRunEffect(SDL_Haptic *haptic, struct haptic_effect *effect,
                            Uint32 iterations)
{
    float large = effect->effect.leftright.large_magnitude / 32767.0f;
    float small = effect->effect.leftright.small_magnitude / 32767.0f;

    float total = (large * 0.6f) + (small * 0.4f);

    Android_JNI_HapticRun(((SDL_hapticlist_item *)haptic->hwdata)->device_id, total, effect->effect.leftright.length);
    return 0;
}

int SDL_SYS_HapticStopEffect(SDL_Haptic *haptic, struct haptic_effect *effect)
{
    Android_JNI_HapticStop(((SDL_hapticlist_item *)haptic->hwdata)->device_id);
    return 0;
}

void SDL_SYS_HapticDestroyEffect(SDL_Haptic *haptic, struct haptic_effect *effect)
{
}

int SDL_SYS_HapticGetEffectStatus(SDL_Haptic *haptic, struct haptic_effect *effect)
{
    return 0;
}

int SDL_SYS_HapticSetGain(SDL_Haptic *haptic, int gain)
{
    return 0;
}

int SDL_SYS_HapticSetAutocenter(SDL_Haptic *haptic, int autocenter)
{
    return 0;
}

int SDL_SYS_HapticPause(SDL_Haptic *haptic)
{
    return 0;
}

int SDL_SYS_HapticUnpause(SDL_Haptic *haptic)
{
    return 0;
}

int SDL_SYS_HapticStopAll(SDL_Haptic *haptic)
{
    return 0;
}

int Android_AddHaptic(int device_id, const char *name)
{
    SDL_hapticlist_item *item;
    item = (SDL_hapticlist_item *)SDL_calloc(1, sizeof(SDL_hapticlist_item));
    if (!item) {
        return -1;
    }

    item->device_id = device_id;
    item->name = SDL_strdup(name);
    if (!item->name) {
        SDL_free(item);
        return -1;
    }

    if (!SDL_hapticlist_tail) {
        SDL_hapticlist = SDL_hapticlist_tail = item;
    } else {
        SDL_hapticlist_tail->next = item;
        SDL_hapticlist_tail = item;
    }

    ++numhaptics;
    return numhaptics;
}

int Android_RemoveHaptic(int device_id)
{
    SDL_hapticlist_item *item;
    SDL_hapticlist_item *prev = NULL;

    for (item = SDL_hapticlist; item; item = item->next) {
        /* found it, remove it. */
        if (device_id == item->device_id) {
            const int retval = item->haptic ? item->haptic->index : -1;

            if (prev) {
                prev->next = item->next;
            } else {
                SDL_assert(SDL_hapticlist == item);
                SDL_hapticlist = item->next;
            }
            if (item == SDL_hapticlist_tail) {
                SDL_hapticlist_tail = prev;
            }

            /* Need to decrement the haptic count */
            --numhaptics;
            /* !!! TODO: Send a haptic remove event? */

            SDL_free(item->name);
            SDL_free(item);
            return retval;
        }
        prev = item;
    }
    return -1;
}

#endif /* SDL_HAPTIC_ANDROID */

/* vi: set ts=4 sw=4 expandtab: */
