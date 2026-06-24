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

#if defined(SDL_HAPTIC_DINPUT) || defined(SDL_HAPTIC_XINPUT)

#include "SDL_thread.h"
#include "SDL_mutex.h"
#include "SDL_timer.h"
#include "SDL_hints.h"
#include "SDL_haptic.h"
#include "../SDL_syshaptic.h"
#include "SDL_joystick.h"
#include "../../joystick/SDL_sysjoystick.h"     /* For the real SDL_Joystick */
#include "../../joystick/windows/SDL_windowsjoystick_c.h"      /* For joystick hwdata */
#include "../../joystick/windows/SDL_xinputjoystick_c.h"      /* For xinput rumble */

#include "SDL_windowshaptic_c.h"
#include "SDL_dinputhaptic_c.h"
#include "SDL_xinputhaptic_c.h"

/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Internal stuff.
 */
SDL_hapticlist_item *SDL_hapticlist = NULL;
static SDL_hapticlist_item *SDL_hapticlist_tail = NULL;
static int numhaptics = 0;

/*
 * Initializes the haptic subsystem.
 */
int SDL_SYS_HapticInit(void)
{
    JoyStick_DeviceData *device;

    if (SDL_DINPUT_HapticInit() < 0) {
        return -1;
    }
    if (SDL_XINPUT_HapticInit() < 0) {
        return -1;
    }

    /* The joystick subsystem will usually be initialized before haptics,
     * so the initial HapticMaybeAddDevice() calls from the joystick
     * subsystem will arrive too early to create haptic devices. We will
     * invoke those callbacks again here to pick up any joysticks that
     * were added prior to haptics initialization. */
    for (device = SYS_Joystick; device; device = device->pNext) {
        if (device->bXInputDevice) {
            SDL_XINPUT_HapticMaybeAddDevice(device->XInputUserId);
        } else {
            SDL_DINPUT_HapticMaybeAddDevice(&device->dxdevice);
        }
    }

    return numhaptics;
}

int SDL_SYS_AddHapticDevice(SDL_hapticlist_item *item)
{
    if (!SDL_hapticlist_tail) {
        SDL_hapticlist = SDL_hapticlist_tail = item;
    } else {
        SDL_hapticlist_tail->next = item;
        SDL_hapticlist_tail = item;
    }

    /* Device has been added. */
    ++numhaptics;

    return numhaptics;
}

int SDL_SYS_RemoveHapticDevice(SDL_hapticlist_item *prev, SDL_hapticlist_item *item)
{
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
    --numhaptics;
    /* !!! TODO: Send a haptic remove event? */
    SDL_free(item);
    return retval;
}

int SDL_SYS_NumHaptics(void)
{
    return numhaptics;
}

static SDL_hapticlist_item *HapticByDevIndex(int device_index)
{
    SDL_hapticlist_item *item = SDL_hapticlist;

    if ((device_index < 0) || (device_index >= numhaptics)) {
        return NULL;
    }

    while (device_index > 0) {
        SDL_assert(item != NULL);
        --device_index;
        item = item->next;
    }
    return item;
}

/*
 * Return the name of a haptic device, does not need to be opened.
 */
const char *SDL_SYS_HapticName(int index)
{
    SDL_hapticlist_item *item = HapticByDevIndex(index);
    return item->name;
}

/*
 * Opens a haptic device for usage.
 */
int SDL_SYS_HapticOpen(SDL_Haptic *haptic)
{
    SDL_hapticlist_item *item = HapticByDevIndex(haptic->index);
    if (item->bXInputHaptic) {
        return SDL_XINPUT_HapticOpen(haptic, item);
    } else {
        return SDL_DINPUT_HapticOpen(haptic, item);
    }
}

/*
 * Opens a haptic device from first mouse it finds for usage.
 */
int SDL_SYS_HapticMouse(void)
{
#ifdef SDL_HAPTIC_DINPUT
    SDL_hapticlist_item *item;
    int index = 0;

    /* Grab the first mouse haptic device we find. */
    for (item = SDL_hapticlist; item; item = item->next) {
        if (item->capabilities.dwDevType == DI8DEVCLASS_POINTER) {
            return index;
        }
        ++index;
    }
#endif /* SDL_HAPTIC_DINPUT */
    return -1;
}

/*
 * Checks to see if a joystick has haptic features.
 */
int SDL_SYS_JoystickIsHaptic(SDL_Joystick *joystick)
{
    if (joystick->driver != &SDL_WINDOWS_JoystickDriver) {
        return 0;
    }
#ifdef SDL_HAPTIC_XINPUT
    if (joystick->hwdata->bXInputHaptic) {
        return 1;
    }
#endif
#ifdef SDL_HAPTIC_DINPUT
    if (joystick->hwdata->Capabilities.dwFlags & DIDC_FORCEFEEDBACK) {
        return 1;
    }
#endif
    return 0;
}

/*
 * Checks to see if the haptic device and joystick are in reality the same.
 */
int SDL_SYS_JoystickSameHaptic(SDL_Haptic *haptic, SDL_Joystick *joystick)
{
    if (joystick->driver != &SDL_WINDOWS_JoystickDriver) {
        return 0;
    }
    if (joystick->hwdata->bXInputHaptic != haptic->hwdata->bXInputHaptic) {
        return 0; /* one is XInput, one is not; not the same device. */
    } else if (joystick->hwdata->bXInputHaptic) {
        return SDL_XINPUT_JoystickSameHaptic(haptic, joystick);
    } else {
        return SDL_DINPUT_JoystickSameHaptic(haptic, joystick);
    }
}

/*
 * Opens a SDL_Haptic from a SDL_Joystick.
 */
int SDL_SYS_HapticOpenFromJoystick(SDL_Haptic *haptic, SDL_Joystick *joystick)
{
    SDL_assert(joystick->driver == &SDL_WINDOWS_JoystickDriver);

    if (joystick->hwdata->bXInputDevice) {
        return SDL_XINPUT_HapticOpenFromJoystick(haptic, joystick);
    } else {
        return SDL_DINPUT_HapticOpenFromJoystick(haptic, joystick);
    }
}

/*
 * Closes the haptic device.
 */
void SDL_SYS_HapticClose(SDL_Haptic *haptic)
{
    if (haptic->hwdata) {

        /* Free effects. */
        SDL_free(haptic->effects);
        haptic->effects = NULL;
        haptic->neffects = 0;

        /* Clean up */
        if (haptic->hwdata->bXInputHaptic) {
            SDL_XINPUT_HapticClose(haptic);
        } else {
            SDL_DINPUT_HapticClose(haptic);
        }

        /* Free */
        SDL_free(haptic->hwdata);
        haptic->hwdata = NULL;
    }
}

/*
 * Clean up after system specific haptic stuff
 */
void SDL_SYS_HapticQuit(void)
{
    SDL_hapticlist_item *item;
    SDL_hapticlist_item *next = NULL;
    SDL_Haptic *hapticitem = NULL;

    extern SDL_Haptic *SDL_haptics;
    for (hapticitem = SDL_haptics; hapticitem; hapticitem = hapticitem->next) {
        if ((hapticitem->hwdata->bXInputHaptic) && (hapticitem->hwdata->thread)) {
            /* we _have_ to stop the thread before we free the XInput DLL! */
            SDL_AtomicSet(&hapticitem->hwdata->stopThread, 1);
            SDL_WaitThread(hapticitem->hwdata->thread, NULL);
            hapticitem->hwdata->thread = NULL;
        }
    }

    for (item = SDL_hapticlist; item; item = next) {
        /* Opened and not closed haptics are leaked, this is on purpose.
         * Close your haptic devices after usage. */
        /* !!! FIXME: (...is leaking on purpose a good idea?) - No, of course not. */
        next = item->next;
        SDL_free(item->name);
        SDL_free(item);
    }

    SDL_XINPUT_HapticQuit();
    SDL_DINPUT_HapticQuit();

    numhaptics = 0;
    SDL_hapticlist = NULL;
    SDL_hapticlist_tail = NULL;
}

/*
 * Creates a new haptic effect.
 */
int SDL_SYS_HapticNewEffect(SDL_Haptic *haptic, struct haptic_effect *effect,
                            SDL_HapticEffect *base)
{
    int result;

    /* Alloc the effect. */
    effect->hweffect = (struct haptic_hweffect *)
        SDL_malloc(sizeof(struct haptic_hweffect));
    if (!effect->hweffect) {
        SDL_OutOfMemory();
        return -1;
    }
    SDL_zerop(effect->hweffect);

    if (haptic->hwdata->bXInputHaptic) {
        result = SDL_XINPUT_HapticNewEffect(haptic, effect, base);
    } else {
        result = SDL_DINPUT_HapticNewEffect(haptic, effect, base);
    }
    if (result < 0) {
        SDL_free(effect->hweffect);
        effect->hweffect = NULL;
    }
    return result;
}

/*
 * Updates an effect.
 */
int SDL_SYS_HapticUpdateEffect(SDL_Haptic *haptic,
                               struct haptic_effect *effect,
                               SDL_HapticEffect *data)
{
    if (haptic->hwdata->bXInputHaptic) {
        return SDL_XINPUT_HapticUpdateEffect(haptic, effect, data);
    } else {
        return SDL_DINPUT_HapticUpdateEffect(haptic, effect, data);
    }
}

/*
 * Runs an effect.
 */
int SDL_SYS_HapticRunEffect(SDL_Haptic *haptic, struct haptic_effect *effect,
                            Uint32 iterations)
{
    if (haptic->hwdata->bXInputHaptic) {
        return SDL_XINPUT_HapticRunEffect(haptic, effect, iterations);
    } else {
        return SDL_DINPUT_HapticRunEffect(haptic, effect, iterations);
    }
}

/*
 * Stops an effect.
 */
int SDL_SYS_HapticStopEffect(SDL_Haptic *haptic, struct haptic_effect *effect)
{
    if (haptic->hwdata->bXInputHaptic) {
        return SDL_XINPUT_HapticStopEffect(haptic, effect);
    } else {
        return SDL_DINPUT_HapticStopEffect(haptic, effect);
    }
}

/*
 * Frees the effect.
 */
void SDL_SYS_HapticDestroyEffect(SDL_Haptic *haptic, struct haptic_effect *effect)
{
    if (haptic->hwdata->bXInputHaptic) {
        SDL_XINPUT_HapticDestroyEffect(haptic, effect);
    } else {
        SDL_DINPUT_HapticDestroyEffect(haptic, effect);
    }
    SDL_free(effect->hweffect);
    effect->hweffect = NULL;
}

/*
 * Gets the status of a haptic effect.
 */
int SDL_SYS_HapticGetEffectStatus(SDL_Haptic *haptic,
                                  struct haptic_effect *effect)
{
    if (haptic->hwdata->bXInputHaptic) {
        return SDL_XINPUT_HapticGetEffectStatus(haptic, effect);
    } else {
        return SDL_DINPUT_HapticGetEffectStatus(haptic, effect);
    }
}

/*
 * Sets the gain.
 */
int SDL_SYS_HapticSetGain(SDL_Haptic *haptic, int gain)
{
    if (haptic->hwdata->bXInputHaptic) {
        return SDL_XINPUT_HapticSetGain(haptic, gain);
    } else {
        return SDL_DINPUT_HapticSetGain(haptic, gain);
    }
}

/*
 * Sets the autocentering.
 */
int SDL_SYS_HapticSetAutocenter(SDL_Haptic *haptic, int autocenter)
{
    if (haptic->hwdata->bXInputHaptic) {
        return SDL_XINPUT_HapticSetAutocenter(haptic, autocenter);
    } else {
        return SDL_DINPUT_HapticSetAutocenter(haptic, autocenter);
    }
}

/*
 * Pauses the device.
 */
int SDL_SYS_HapticPause(SDL_Haptic *haptic)
{
    if (haptic->hwdata->bXInputHaptic) {
        return SDL_XINPUT_HapticPause(haptic);
    } else {
        return SDL_DINPUT_HapticPause(haptic);
    }
}

/*
 * Pauses the device.
 */
int SDL_SYS_HapticUnpause(SDL_Haptic *haptic)
{
    if (haptic->hwdata->bXInputHaptic) {
        return SDL_XINPUT_HapticUnpause(haptic);
    } else {
        return SDL_DINPUT_HapticUnpause(haptic);
    }
}

/*
 * Stops all the playing effects on the device.
 */
int SDL_SYS_HapticStopAll(SDL_Haptic *haptic)
{
    if (haptic->hwdata->bXInputHaptic) {
        return SDL_XINPUT_HapticStopAll(haptic);
    } else {
        return SDL_DINPUT_HapticStopAll(haptic);
    }
}

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif

#endif /* SDL_HAPTIC_DINPUT || SDL_HAPTIC_XINPUT */

/* vi: set ts=4 sw=4 expandtab: */
