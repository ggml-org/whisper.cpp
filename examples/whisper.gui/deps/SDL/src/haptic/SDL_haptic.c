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

#include "SDL_syshaptic.h"
#include "SDL_haptic_c.h"
#include "../joystick/SDL_joystick_c.h" /* For SDL_PrivateJoystickValid */

/* Global for SDL_windowshaptic.c */
#if defined(SDL_HAPTIC_DINPUT) || defined(SDL_HAPTIC_XINPUT)
SDL_Haptic *SDL_haptics = NULL;
#else
static SDL_Haptic *SDL_haptics = NULL;
#endif

/*
 * Initializes the Haptic devices.
 */
int SDL_HapticInit(void)
{
    int status;

    status = SDL_SYS_HapticInit();
    if (status >= 0) {
        status = 0;
    }

    return status;
}

/*
 * Checks to see if the haptic device is valid
 */
static int ValidHaptic(SDL_Haptic *haptic)
{
    int valid;
    SDL_Haptic *hapticlist;

    valid = 0;
    if (haptic) {
        hapticlist = SDL_haptics;
        while (hapticlist) {
            if (hapticlist == haptic) {
                valid = 1;
                break;
            }
            hapticlist = hapticlist->next;
        }
    }

    /* Create the error here. */
    if (valid == 0) {
        SDL_SetError("Haptic: Invalid haptic device identifier");
    }

    return valid;
}

/*
 * Returns the number of available devices.
 */
int SDL_NumHaptics(void)
{
    return SDL_SYS_NumHaptics();
}

/*
 * Gets the name of a Haptic device by index.
 */
const char *SDL_HapticName(int device_index)
{
    if ((device_index < 0) || (device_index >= SDL_NumHaptics())) {
        SDL_SetError("Haptic: There are %d haptic devices available",
                     SDL_NumHaptics());
        return NULL;
    }
    return SDL_SYS_HapticName(device_index);
}

/*
 * Opens a Haptic device.
 */
SDL_Haptic *SDL_HapticOpen(int device_index)
{
    SDL_Haptic *haptic;
    SDL_Haptic *hapticlist;

    if ((device_index < 0) || (device_index >= SDL_NumHaptics())) {
        SDL_SetError("Haptic: There are %d haptic devices available",
                     SDL_NumHaptics());
        return NULL;
    }

    hapticlist = SDL_haptics;
    /* If the haptic is already open, return it
     * TODO: Should we create haptic instance IDs like the Joystick API?
     */
    while (hapticlist) {
        if (device_index == hapticlist->index) {
            haptic = hapticlist;
            ++haptic->ref_count;
            return haptic;
        }
        hapticlist = hapticlist->next;
    }

    /* Create the haptic device */
    haptic = (SDL_Haptic *)SDL_malloc(sizeof(*haptic));
    if (!haptic) {
        SDL_OutOfMemory();
        return NULL;
    }

    /* Initialize the haptic device */
    SDL_memset(haptic, 0, sizeof(*haptic));
    haptic->rumble_id = -1;
    haptic->index = device_index;
    if (SDL_SYS_HapticOpen(haptic) < 0) {
        SDL_free(haptic);
        return NULL;
    }

    /* Add haptic to list */
    ++haptic->ref_count;
    /* Link the haptic in the list */
    haptic->next = SDL_haptics;
    SDL_haptics = haptic;

    /* Disable autocenter and set gain to max. */
    if (haptic->supported & SDL_HAPTIC_GAIN) {
        SDL_HapticSetGain(haptic, 100);
    }
    if (haptic->supported & SDL_HAPTIC_AUTOCENTER) {
        SDL_HapticSetAutocenter(haptic, 0);
    }

    return haptic;
}

/*
 * Returns 1 if the device has been opened.
 */
int SDL_HapticOpened(int device_index)
{
    int opened;
    SDL_Haptic *hapticlist;

    /* Make sure it's valid. */
    if ((device_index < 0) || (device_index >= SDL_NumHaptics())) {
        SDL_SetError("Haptic: There are %d haptic devices available",
                     SDL_NumHaptics());
        return 0;
    }

    opened = 0;
    hapticlist = SDL_haptics;
    /* TODO Should this use an instance ID? */
    while (hapticlist) {
        if (hapticlist->index == (Uint8)device_index) {
            opened = 1;
            break;
        }
        hapticlist = hapticlist->next;
    }
    return opened;
}

/*
 * Returns the index to a haptic device.
 */
int SDL_HapticIndex(SDL_Haptic *haptic)
{
    if (!ValidHaptic(haptic)) {
        return -1;
    }

    return haptic->index;
}

/*
 * Returns SDL_TRUE if mouse is haptic, SDL_FALSE if it isn't.
 */
int SDL_MouseIsHaptic(void)
{
    if (SDL_SYS_HapticMouse() < 0) {
        return SDL_FALSE;
    }
    return SDL_TRUE;
}

/*
 * Returns the haptic device if mouse is haptic or NULL elsewise.
 */
SDL_Haptic *SDL_HapticOpenFromMouse(void)
{
    int device_index;

    device_index = SDL_SYS_HapticMouse();

    if (device_index < 0) {
        SDL_SetError("Haptic: Mouse isn't a haptic device.");
        return NULL;
    }

    return SDL_HapticOpen(device_index);
}

/*
 * Returns SDL_TRUE if joystick has haptic features.
 */
int SDL_JoystickIsHaptic(SDL_Joystick *joystick)
{
    int ret;

    SDL_LockJoysticks();
    {
        /* Must be a valid joystick */
        if (!SDL_PrivateJoystickValid(joystick)) {
            SDL_UnlockJoysticks();
            return -1;
        }

        ret = SDL_SYS_JoystickIsHaptic(joystick);
    }
    SDL_UnlockJoysticks();

    if (ret > 0) {
        return SDL_TRUE;
    } else if (ret == 0) {
        return SDL_FALSE;
    }

    return -1;
}

/*
 * Opens a haptic device from a joystick.
 */
SDL_Haptic *SDL_HapticOpenFromJoystick(SDL_Joystick *joystick)
{
    SDL_Haptic *haptic;
    SDL_Haptic *hapticlist;

    /* Make sure there is room. */
    if (SDL_NumHaptics() <= 0) {
        SDL_SetError("Haptic: There are %d haptic devices available",
                     SDL_NumHaptics());
        return NULL;
    }

    SDL_LockJoysticks();
    {
        /* Must be a valid joystick */
        if (!SDL_PrivateJoystickValid(joystick)) {
            SDL_SetError("Haptic: Joystick isn't valid.");
            SDL_UnlockJoysticks();
            return NULL;
        }

        /* Joystick must be haptic */
        if (SDL_SYS_JoystickIsHaptic(joystick) <= 0) {
            SDL_SetError("Haptic: Joystick isn't a haptic device.");
            SDL_UnlockJoysticks();
            return NULL;
        }

        hapticlist = SDL_haptics;
        /* Check to see if joystick's haptic is already open */
        while (hapticlist) {
            if (SDL_SYS_JoystickSameHaptic(hapticlist, joystick)) {
                haptic = hapticlist;
                ++haptic->ref_count;
                SDL_UnlockJoysticks();
                return haptic;
            }
            hapticlist = hapticlist->next;
        }

        /* Create the haptic device */
        haptic = (SDL_Haptic *)SDL_malloc(sizeof(*haptic));
        if (!haptic) {
            SDL_OutOfMemory();
            SDL_UnlockJoysticks();
            return NULL;
        }

        /* Initialize the haptic device */
        SDL_memset(haptic, 0, sizeof(SDL_Haptic));
        haptic->rumble_id = -1;
        if (SDL_SYS_HapticOpenFromJoystick(haptic, joystick) < 0) {
            SDL_SetError("Haptic: SDL_SYS_HapticOpenFromJoystick failed.");
            SDL_free(haptic);
            SDL_UnlockJoysticks();
            return NULL;
        }
    }
    SDL_UnlockJoysticks();

    /* Add haptic to list */
    ++haptic->ref_count;
    /* Link the haptic in the list */
    haptic->next = SDL_haptics;
    SDL_haptics = haptic;

    return haptic;
}

/*
 * Closes a SDL_Haptic device.
 */
void SDL_HapticClose(SDL_Haptic *haptic)
{
    int i;
    SDL_Haptic *hapticlist;
    SDL_Haptic *hapticlistprev;

    /* Must be valid */
    if (!ValidHaptic(haptic)) {
        return;
    }

    /* Check if it's still in use */
    if (--haptic->ref_count > 0) {
        return;
    }

    /* Close it, properly removing effects if needed */
    for (i = 0; i < haptic->neffects; i++) {
        if (haptic->effects[i].hweffect != NULL) {
            SDL_HapticDestroyEffect(haptic, i);
        }
    }
    SDL_SYS_HapticClose(haptic);

    /* Remove from the list */
    hapticlist = SDL_haptics;
    hapticlistprev = NULL;
    while (hapticlist) {
        if (haptic == hapticlist) {
            if (hapticlistprev) {
                /* unlink this entry */
                hapticlistprev->next = hapticlist->next;
            } else {
                SDL_haptics = haptic->next;
            }

            break;
        }
        hapticlistprev = hapticlist;
        hapticlist = hapticlist->next;
    }

    /* Free */
    SDL_free(haptic);
}

/*
 * Cleans up after the subsystem.
 */
void SDL_HapticQuit(void)
{
    while (SDL_haptics) {
        SDL_HapticClose(SDL_haptics);
    }

    SDL_SYS_HapticQuit();
}

/*
 * Returns the number of effects a haptic device has.
 */
int SDL_HapticNumEffects(SDL_Haptic *haptic)
{
    if (!ValidHaptic(haptic)) {
        return -1;
    }

    return haptic->neffects;
}

/*
 * Returns the number of effects a haptic device can play.
 */
int SDL_HapticNumEffectsPlaying(SDL_Haptic *haptic)
{
    if (!ValidHaptic(haptic)) {
        return -1;
    }

    return haptic->nplaying;
}

/*
 * Returns supported effects by the device.
 */
unsigned int SDL_HapticQuery(SDL_Haptic *haptic)
{
    if (!ValidHaptic(haptic)) {
        return 0; /* same as if no effects were supported */
    }

    return haptic->supported;
}

/*
 * Returns the number of axis on the device.
 */
int SDL_HapticNumAxes(SDL_Haptic *haptic)
{
    if (!ValidHaptic(haptic)) {
        return -1;
    }

    return haptic->naxes;
}

/*
 * Checks to see if the device can support the effect.
 */
int SDL_HapticEffectSupported(SDL_Haptic *haptic, SDL_HapticEffect *effect)
{
    if (!ValidHaptic(haptic)) {
        return -1;
    }

    if ((haptic->supported & effect->type) != 0) {
        return SDL_TRUE;
    }
    return SDL_FALSE;
}

/*
 * Creates a new haptic effect.
 */
int SDL_HapticNewEffect(SDL_Haptic *haptic, SDL_HapticEffect *effect)
{
    int i;

    /* Check for device validity. */
    if (!ValidHaptic(haptic)) {
        return -1;
    }

    /* Check to see if effect is supported */
    if (SDL_HapticEffectSupported(haptic, effect) == SDL_FALSE) {
        return SDL_SetError("Haptic: Effect not supported by haptic device.");
    }

    /* See if there's a free slot */
    for (i = 0; i < haptic->neffects; i++) {
        if (haptic->effects[i].hweffect == NULL) {

            /* Now let the backend create the real effect */
            if (SDL_SYS_HapticNewEffect(haptic, &haptic->effects[i], effect) != 0) {
                return -1; /* Backend failed to create effect */
            }

            SDL_memcpy(&haptic->effects[i].effect, effect,
                       sizeof(SDL_HapticEffect));
            return i;
        }
    }

    return SDL_SetError("Haptic: Device has no free space left.");
}

/*
 * Checks to see if an effect is valid.
 */
static int ValidEffect(SDL_Haptic *haptic, int effect)
{
    if ((effect < 0) || (effect >= haptic->neffects)) {
        SDL_SetError("Haptic: Invalid effect identifier.");
        return 0;
    }
    return 1;
}

/*
 * Updates an effect.
 */
int SDL_HapticUpdateEffect(SDL_Haptic *haptic, int effect,
                           SDL_HapticEffect *data)
{
    if (!ValidHaptic(haptic) || !ValidEffect(haptic, effect)) {
        return -1;
    }

    /* Can't change type dynamically. */
    if (data->type != haptic->effects[effect].effect.type) {
        return SDL_SetError("Haptic: Updating effect type is illegal.");
    }

    /* Updates the effect */
    if (SDL_SYS_HapticUpdateEffect(haptic, &haptic->effects[effect], data) <
        0) {
        return -1;
    }

    SDL_memcpy(&haptic->effects[effect].effect, data,
               sizeof(SDL_HapticEffect));
    return 0;
}

/*
 * Runs the haptic effect on the device.
 */
int SDL_HapticRunEffect(SDL_Haptic *haptic, int effect, Uint32 iterations)
{
    if (!ValidHaptic(haptic) || !ValidEffect(haptic, effect)) {
        return -1;
    }

    /* Run the effect */
    if (SDL_SYS_HapticRunEffect(haptic, &haptic->effects[effect], iterations) < 0) {
        return -1;
    }

    return 0;
}

/*
 * Stops the haptic effect on the device.
 */
int SDL_HapticStopEffect(SDL_Haptic *haptic, int effect)
{
    if (!ValidHaptic(haptic) || !ValidEffect(haptic, effect)) {
        return -1;
    }

    /* Stop the effect */
    if (SDL_SYS_HapticStopEffect(haptic, &haptic->effects[effect]) < 0) {
        return -1;
    }

    return 0;
}

/*
 * Gets rid of a haptic effect.
 */
void SDL_HapticDestroyEffect(SDL_Haptic *haptic, int effect)
{
    if (!ValidHaptic(haptic) || !ValidEffect(haptic, effect)) {
        return;
    }

    /* Not allocated */
    if (haptic->effects[effect].hweffect == NULL) {
        return;
    }

    SDL_SYS_HapticDestroyEffect(haptic, &haptic->effects[effect]);
}

/*
 * Gets the status of a haptic effect.
 */
int SDL_HapticGetEffectStatus(SDL_Haptic *haptic, int effect)
{
    if (!ValidHaptic(haptic) || !ValidEffect(haptic, effect)) {
        return -1;
    }

    if (!(haptic->supported & SDL_HAPTIC_STATUS)) {
        return SDL_SetError("Haptic: Device does not support status queries.");
    }

    return SDL_SYS_HapticGetEffectStatus(haptic, &haptic->effects[effect]);
}

/*
 * Sets the global gain of the device.
 */
int SDL_HapticSetGain(SDL_Haptic *haptic, int gain)
{
    const char *env;
    int real_gain, max_gain;

    if (!ValidHaptic(haptic)) {
        return -1;
    }

    if (!(haptic->supported & SDL_HAPTIC_GAIN)) {
        return SDL_SetError("Haptic: Device does not support setting gain.");
    }

    if ((gain < 0) || (gain > 100)) {
        return SDL_SetError("Haptic: Gain must be between 0 and 100.");
    }

    /* We use the envvar to get the maximum gain. */
    env = SDL_getenv("SDL_HAPTIC_GAIN_MAX");
    if (env) {
        max_gain = SDL_atoi(env);

        /* Check for sanity. */
        if (max_gain < 0) {
            max_gain = 0;
        } else if (max_gain > 100) {
            max_gain = 100;
        }

        /* We'll scale it linearly with SDL_HAPTIC_GAIN_MAX */
        real_gain = (gain * max_gain) / 100;
    } else {
        real_gain = gain;
    }

    if (SDL_SYS_HapticSetGain(haptic, real_gain) < 0) {
        return -1;
    }

    return 0;
}

/*
 * Makes the device autocenter, 0 disables.
 */
int SDL_HapticSetAutocenter(SDL_Haptic *haptic, int autocenter)
{
    if (!ValidHaptic(haptic)) {
        return -1;
    }

    if (!(haptic->supported & SDL_HAPTIC_AUTOCENTER)) {
        return SDL_SetError("Haptic: Device does not support setting autocenter.");
    }

    if ((autocenter < 0) || (autocenter > 100)) {
        return SDL_SetError("Haptic: Autocenter must be between 0 and 100.");
    }

    if (SDL_SYS_HapticSetAutocenter(haptic, autocenter) < 0) {
        return -1;
    }

    return 0;
}

/*
 * Pauses the haptic device.
 */
int SDL_HapticPause(SDL_Haptic *haptic)
{
    if (!ValidHaptic(haptic)) {
        return -1;
    }

    if (!(haptic->supported & SDL_HAPTIC_PAUSE)) {
        return SDL_SetError("Haptic: Device does not support setting pausing.");
    }

    return SDL_SYS_HapticPause(haptic);
}

/*
 * Unpauses the haptic device.
 */
int SDL_HapticUnpause(SDL_Haptic *haptic)
{
    if (!ValidHaptic(haptic)) {
        return -1;
    }

    if (!(haptic->supported & SDL_HAPTIC_PAUSE)) {
        return 0; /* Not going to be paused, so we pretend it's unpaused. */
    }

    return SDL_SYS_HapticUnpause(haptic);
}

/*
 * Stops all the currently playing effects.
 */
int SDL_HapticStopAll(SDL_Haptic *haptic)
{
    if (!ValidHaptic(haptic)) {
        return -1;
    }

    return SDL_SYS_HapticStopAll(haptic);
}

/*
 * Checks to see if rumble is supported.
 */
int SDL_HapticRumbleSupported(SDL_Haptic *haptic)
{
    if (!ValidHaptic(haptic)) {
        return -1;
    }

    /* Most things can use SINE, but XInput only has LEFTRIGHT. */
    return (haptic->supported & (SDL_HAPTIC_SINE | SDL_HAPTIC_LEFTRIGHT)) != 0;
}

/*
 * Initializes the haptic device for simple rumble playback.
 */
int SDL_HapticRumbleInit(SDL_Haptic *haptic)
{
    SDL_HapticEffect *efx = &haptic->rumble_effect;

    if (!ValidHaptic(haptic)) {
        return -1;
    }

    /* Already allocated. */
    if (haptic->rumble_id >= 0) {
        return 0;
    }

    SDL_zerop(efx);
    if (haptic->supported & SDL_HAPTIC_SINE) {
        efx->type = SDL_HAPTIC_SINE;
        efx->periodic.direction.type = SDL_HAPTIC_CARTESIAN;
        efx->periodic.period = 1000;
        efx->periodic.magnitude = 0x4000;
        efx->periodic.length = 5000;
        efx->periodic.attack_length = 0;
        efx->periodic.fade_length = 0;
    } else if (haptic->supported & SDL_HAPTIC_LEFTRIGHT) { /* XInput? */
        efx->type = SDL_HAPTIC_LEFTRIGHT;
        efx->leftright.length = 5000;
        efx->leftright.large_magnitude = 0x4000;
        efx->leftright.small_magnitude = 0x4000;
    } else {
        return SDL_SetError("Device doesn't support rumble");
    }

    haptic->rumble_id = SDL_HapticNewEffect(haptic, &haptic->rumble_effect);
    if (haptic->rumble_id >= 0) {
        return 0;
    }
    return -1;
}

/*
 * Runs simple rumble on a haptic device
 */
int SDL_HapticRumblePlay(SDL_Haptic *haptic, float strength, Uint32 length)
{
    SDL_HapticEffect *efx;
    Sint16 magnitude;

    if (!ValidHaptic(haptic)) {
        return -1;
    }

    if (haptic->rumble_id < 0) {
        return SDL_SetError("Haptic: Rumble effect not initialized on haptic device");
    }

    /* Clamp strength. */
    if (strength > 1.0f) {
        strength = 1.0f;
    } else if (strength < 0.0f) {
        strength = 0.0f;
    }
    magnitude = (Sint16)(32767.0f * strength);

    efx = &haptic->rumble_effect;
    if (efx->type == SDL_HAPTIC_SINE) {
        efx->periodic.magnitude = magnitude;
        efx->periodic.length = length;
    } else if (efx->type == SDL_HAPTIC_LEFTRIGHT) {
        efx->leftright.small_magnitude = efx->leftright.large_magnitude = magnitude;
        efx->leftright.length = length;
    } else {
        SDL_assert(0 && "This should have been caught elsewhere");
    }

    if (SDL_HapticUpdateEffect(haptic, haptic->rumble_id, &haptic->rumble_effect) < 0) {
        return -1;
    }

    return SDL_HapticRunEffect(haptic, haptic->rumble_id, 1);
}

/*
 * Stops the simple rumble on a haptic device.
 */
int SDL_HapticRumbleStop(SDL_Haptic *haptic)
{
    if (!ValidHaptic(haptic)) {
        return -1;
    }

    if (haptic->rumble_id < 0) {
        return SDL_SetError("Haptic: Rumble effect not initialized on haptic device");
    }

    return SDL_HapticStopEffect(haptic, haptic->rumble_id);
}

/* vi: set ts=4 sw=4 expandtab: */
