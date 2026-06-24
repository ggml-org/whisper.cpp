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

#ifdef SDL_JOYSTICK_ANDROID

#include <stdio.h>              /* For the definition of NULL */
#include "SDL_error.h"
#include "SDL_events.h"

#include "SDL_joystick.h"
#include "SDL_hints.h"
#include "SDL_timer.h"
#include "SDL_sysjoystick_c.h"
#include "../SDL_joystick_c.h"
#include "../../events/SDL_keyboard_c.h"
#include "../../core/android/SDL_android.h"
#include "../hidapi/SDL_hidapijoystick_c.h"

#include "android/keycodes.h"

/* As of platform android-14, android/keycodes.h is missing these defines */
#ifndef AKEYCODE_BUTTON_1
#define AKEYCODE_BUTTON_1  188
#define AKEYCODE_BUTTON_2  189
#define AKEYCODE_BUTTON_3  190
#define AKEYCODE_BUTTON_4  191
#define AKEYCODE_BUTTON_5  192
#define AKEYCODE_BUTTON_6  193
#define AKEYCODE_BUTTON_7  194
#define AKEYCODE_BUTTON_8  195
#define AKEYCODE_BUTTON_9  196
#define AKEYCODE_BUTTON_10 197
#define AKEYCODE_BUTTON_11 198
#define AKEYCODE_BUTTON_12 199
#define AKEYCODE_BUTTON_13 200
#define AKEYCODE_BUTTON_14 201
#define AKEYCODE_BUTTON_15 202
#define AKEYCODE_BUTTON_16 203
#endif

#define ANDROID_ACCELEROMETER_NAME      "Android Accelerometer"
#define ANDROID_ACCELEROMETER_DEVICE_ID INT_MIN
#define ANDROID_MAX_NBUTTONS            36

static SDL_joylist_item *JoystickByDeviceId(int device_id);

static SDL_joylist_item *SDL_joylist = NULL;
static SDL_joylist_item *SDL_joylist_tail = NULL;
static int numjoysticks = 0;

/* Function to convert Android keyCodes into SDL ones.
 * This code manipulation is done to get a sequential list of codes.
 * FIXME: This is only suited for the case where we use a fixed number of buttons determined by ANDROID_MAX_NBUTTONS
 */
static int keycode_to_SDL(int keycode)
{
    /* FIXME: If this function gets too unwieldy in the future, replace with a lookup table */
    int button = 0;
    switch (keycode) {
    /* Some gamepad buttons (API 9) */
    case AKEYCODE_BUTTON_A:
        button = SDL_CONTROLLER_BUTTON_A;
        break;
    case AKEYCODE_BUTTON_B:
        button = SDL_CONTROLLER_BUTTON_B;
        break;
    case AKEYCODE_BUTTON_X:
        button = SDL_CONTROLLER_BUTTON_X;
        break;
    case AKEYCODE_BUTTON_Y:
        button = SDL_CONTROLLER_BUTTON_Y;
        break;
    case AKEYCODE_BUTTON_L1:
        button = SDL_CONTROLLER_BUTTON_LEFTSHOULDER;
        break;
    case AKEYCODE_BUTTON_R1:
        button = SDL_CONTROLLER_BUTTON_RIGHTSHOULDER;
        break;
    case AKEYCODE_BUTTON_THUMBL:
        button = SDL_CONTROLLER_BUTTON_LEFTSTICK;
        break;
    case AKEYCODE_BUTTON_THUMBR:
        button = SDL_CONTROLLER_BUTTON_RIGHTSTICK;
        break;
    case AKEYCODE_MENU:
    case AKEYCODE_BUTTON_START:
        button = SDL_CONTROLLER_BUTTON_START;
        break;
    case AKEYCODE_BACK:
    case AKEYCODE_BUTTON_SELECT:
        button = SDL_CONTROLLER_BUTTON_BACK;
        break;
    case AKEYCODE_BUTTON_MODE:
        button = SDL_CONTROLLER_BUTTON_GUIDE;
        break;
    case AKEYCODE_BUTTON_L2:
        button = 15;
        break;
    case AKEYCODE_BUTTON_R2:
        button = 16;
        break;
    case AKEYCODE_BUTTON_C:
        button = 17;
        break;
    case AKEYCODE_BUTTON_Z:
        button = 18;
        break;

    /* D-Pad key codes (API 1) */
    case AKEYCODE_DPAD_UP:
        button = SDL_CONTROLLER_BUTTON_DPAD_UP;
        break;
    case AKEYCODE_DPAD_DOWN:
        button = SDL_CONTROLLER_BUTTON_DPAD_DOWN;
        break;
    case AKEYCODE_DPAD_LEFT:
        button = SDL_CONTROLLER_BUTTON_DPAD_LEFT;
        break;
    case AKEYCODE_DPAD_RIGHT:
        button = SDL_CONTROLLER_BUTTON_DPAD_RIGHT;
        break;
    case AKEYCODE_DPAD_CENTER:
        /* This is handled better by applications as the A button */
        /*button = 19;*/
        button = SDL_CONTROLLER_BUTTON_A;
        break;

    /* More gamepad buttons (API 12), these get mapped to 20...35*/
    case AKEYCODE_BUTTON_1:
    case AKEYCODE_BUTTON_2:
    case AKEYCODE_BUTTON_3:
    case AKEYCODE_BUTTON_4:
    case AKEYCODE_BUTTON_5:
    case AKEYCODE_BUTTON_6:
    case AKEYCODE_BUTTON_7:
    case AKEYCODE_BUTTON_8:
    case AKEYCODE_BUTTON_9:
    case AKEYCODE_BUTTON_10:
    case AKEYCODE_BUTTON_11:
    case AKEYCODE_BUTTON_12:
    case AKEYCODE_BUTTON_13:
    case AKEYCODE_BUTTON_14:
    case AKEYCODE_BUTTON_15:
    case AKEYCODE_BUTTON_16:
        button = 20 + (keycode - AKEYCODE_BUTTON_1);
        break;

    default:
        return -1;
        /* break; -Wunreachable-code-break */
    }

    /* This is here in case future generations, probably with six fingers per hand,
     * happily add new cases up above and forget to update the max number of buttons.
     */
    SDL_assert(button < ANDROID_MAX_NBUTTONS);
    return button;
}

static SDL_Scancode button_to_scancode(int button)
{
    switch (button) {
    case SDL_CONTROLLER_BUTTON_A:
        return SDL_SCANCODE_RETURN;
    case SDL_CONTROLLER_BUTTON_B:
        return SDL_SCANCODE_ESCAPE;
    case SDL_CONTROLLER_BUTTON_BACK:
        return SDL_SCANCODE_ESCAPE;
    case SDL_CONTROLLER_BUTTON_START:
        return SDL_SCANCODE_MENU;
    case SDL_CONTROLLER_BUTTON_DPAD_UP:
        return SDL_SCANCODE_UP;
    case SDL_CONTROLLER_BUTTON_DPAD_DOWN:
        return SDL_SCANCODE_DOWN;
    case SDL_CONTROLLER_BUTTON_DPAD_LEFT:
        return SDL_SCANCODE_LEFT;
    case SDL_CONTROLLER_BUTTON_DPAD_RIGHT:
        return SDL_SCANCODE_RIGHT;
    }

    /* Unsupported button */
    return SDL_SCANCODE_UNKNOWN;
}

int Android_OnPadDown(int device_id, int keycode)
{
    SDL_joylist_item *item;
    int button = keycode_to_SDL(keycode);
    if (button >= 0) {
        SDL_LockJoysticks();
        item = JoystickByDeviceId(device_id);
        if (item && item->joystick) {
            SDL_PrivateJoystickButton(item->joystick, button, SDL_PRESSED);
        } else {
            SDL_SendKeyboardKey(SDL_PRESSED, button_to_scancode(button));
        }
        SDL_UnlockJoysticks();
        return 0;
    }

    return -1;
}

int Android_OnPadUp(int device_id, int keycode)
{
    SDL_joylist_item *item;
    int button = keycode_to_SDL(keycode);
    if (button >= 0) {
        SDL_LockJoysticks();
        item = JoystickByDeviceId(device_id);
        if (item && item->joystick) {
            SDL_PrivateJoystickButton(item->joystick, button, SDL_RELEASED);
        } else {
            SDL_SendKeyboardKey(SDL_RELEASED, button_to_scancode(button));
        }
        SDL_UnlockJoysticks();
        return 0;
    }

    return -1;
}

int Android_OnJoy(int device_id, int axis, float value)
{
    /* Android gives joy info normalized as [-1.0, 1.0] or [0.0, 1.0] */
    SDL_joylist_item *item;

    SDL_LockJoysticks();
    item = JoystickByDeviceId(device_id);
    if (item && item->joystick) {
        SDL_PrivateJoystickAxis(item->joystick, axis, (Sint16)(32767. * value));
    }
    SDL_UnlockJoysticks();

    return 0;
}

int Android_OnHat(int device_id, int hat_id, int x, int y)
{
    const int DPAD_UP_MASK = (1 << SDL_CONTROLLER_BUTTON_DPAD_UP);
    const int DPAD_DOWN_MASK = (1 << SDL_CONTROLLER_BUTTON_DPAD_DOWN);
    const int DPAD_LEFT_MASK = (1 << SDL_CONTROLLER_BUTTON_DPAD_LEFT);
    const int DPAD_RIGHT_MASK = (1 << SDL_CONTROLLER_BUTTON_DPAD_RIGHT);

    if (x >= -1 && x <= 1 && y >= -1 && y <= 1) {
        SDL_joylist_item *item;

        SDL_LockJoysticks();
        item = JoystickByDeviceId(device_id);
        if (item && item->joystick) {
            int dpad_state = 0;
            int dpad_delta;
            if (x < 0) {
                dpad_state |= DPAD_LEFT_MASK;
            } else if (x > 0) {
                dpad_state |= DPAD_RIGHT_MASK;
            }
            if (y < 0) {
                dpad_state |= DPAD_UP_MASK;
            } else if (y > 0) {
                dpad_state |= DPAD_DOWN_MASK;
            }

            dpad_delta = (dpad_state ^ item->dpad_state);
            if (dpad_delta) {
                if (dpad_delta & DPAD_UP_MASK) {
                    SDL_PrivateJoystickButton(item->joystick, SDL_CONTROLLER_BUTTON_DPAD_UP, (dpad_state & DPAD_UP_MASK) ? SDL_PRESSED : SDL_RELEASED);
                }
                if (dpad_delta & DPAD_DOWN_MASK) {
                    SDL_PrivateJoystickButton(item->joystick, SDL_CONTROLLER_BUTTON_DPAD_DOWN, (dpad_state & DPAD_DOWN_MASK) ? SDL_PRESSED : SDL_RELEASED);
                }
                if (dpad_delta & DPAD_LEFT_MASK) {
                    SDL_PrivateJoystickButton(item->joystick, SDL_CONTROLLER_BUTTON_DPAD_LEFT, (dpad_state & DPAD_LEFT_MASK) ? SDL_PRESSED : SDL_RELEASED);
                }
                if (dpad_delta & DPAD_RIGHT_MASK) {
                    SDL_PrivateJoystickButton(item->joystick, SDL_CONTROLLER_BUTTON_DPAD_RIGHT, (dpad_state & DPAD_RIGHT_MASK) ? SDL_PRESSED : SDL_RELEASED);
                }
                item->dpad_state = dpad_state;
            }
        }
        SDL_UnlockJoysticks();
        return 0;
    }

    return -1;
}

int Android_AddJoystick(int device_id, const char *name, const char *desc, int vendor_id, int product_id, SDL_bool is_accelerometer, int button_mask, int naxes, int axis_mask, int nhats, int nballs)
{
    SDL_joylist_item *item;
    SDL_JoystickGUID guid;
    int i;
    int result = -1;

    SDL_LockJoysticks();

    if (!SDL_GetHintBoolean(SDL_HINT_TV_REMOTE_AS_JOYSTICK, SDL_TRUE)) {
        /* Ignore devices that aren't actually controllers (e.g. remotes), they'll be handled as keyboard input */
        if (naxes < 2 && nhats < 1) {
            goto done;
        }
    }

    if (JoystickByDeviceId(device_id) != NULL || !name) {
        goto done;
    }

#ifdef SDL_JOYSTICK_HIDAPI
    if (HIDAPI_IsDevicePresent(vendor_id, product_id, 0, name)) {
        /* The HIDAPI driver is taking care of this device */
        goto done;
    }
#endif

#ifdef DEBUG_JOYSTICK
    SDL_Log("Joystick: %s, descriptor %s, vendor = 0x%.4x, product = 0x%.4x, %d axes, %d hats\n", name, desc, vendor_id, product_id, naxes, nhats);
#endif

    if (nhats > 0) {
        /* Hat is translated into DPAD buttons */
        button_mask |= ((1 << SDL_CONTROLLER_BUTTON_DPAD_UP) |
                        (1 << SDL_CONTROLLER_BUTTON_DPAD_DOWN) |
                        (1 << SDL_CONTROLLER_BUTTON_DPAD_LEFT) |
                        (1 << SDL_CONTROLLER_BUTTON_DPAD_RIGHT));
        nhats = 0;
    }

    guid = SDL_CreateJoystickGUID(SDL_HARDWARE_BUS_BLUETOOTH, vendor_id, product_id, 0, NULL, desc, 0, 0);

    /* Update the GUID with capability bits */
    {
        Uint16 *guid16 = (Uint16 *)guid.data;
        guid16[6] = SDL_SwapLE16(button_mask);
        guid16[7] = SDL_SwapLE16(axis_mask);
    }

    item = (SDL_joylist_item *)SDL_malloc(sizeof(SDL_joylist_item));
    if (!item) {
        goto done;
    }

    SDL_zerop(item);
    item->guid = guid;
    item->device_id = device_id;
    item->name = SDL_CreateJoystickName(vendor_id, product_id, NULL, name);
    if (!item->name) {
        SDL_free(item);
        goto done;
    }

    item->is_accelerometer = is_accelerometer;
    if (button_mask == 0xFFFFFFFF) {
        item->nbuttons = ANDROID_MAX_NBUTTONS;
    } else {
        for (i = 0; i < sizeof(button_mask) * 8; ++i) {
            if (button_mask & (1 << i)) {
                item->nbuttons = i + 1;
            }
        }
    }
    item->naxes = naxes;
    item->nhats = nhats;
    item->nballs = nballs;
    item->device_instance = SDL_GetNextJoystickInstanceID();
    if (!SDL_joylist_tail) {
        SDL_joylist = SDL_joylist_tail = item;
    } else {
        SDL_joylist_tail->next = item;
        SDL_joylist_tail = item;
    }

    /* Need to increment the joystick count before we post the event */
    ++numjoysticks;

    SDL_PrivateJoystickAdded(item->device_instance);

    result = numjoysticks;

#ifdef DEBUG_JOYSTICK
    SDL_Log("Added joystick %s with device_id %d", item->name, device_id);
#endif

done:
    SDL_UnlockJoysticks();

    return result;
}

int Android_RemoveJoystick(int device_id)
{
    SDL_joylist_item *item = SDL_joylist;
    SDL_joylist_item *prev = NULL;
    int result = -1;

    SDL_LockJoysticks();

    /* Don't call JoystickByDeviceId here or there'll be an infinite loop! */
    while (item) {
        if (item->device_id == device_id) {
            break;
        }
        prev = item;
        item = item->next;
    }

    if (!item) {
        goto done;
    }

    if (item->joystick) {
        item->joystick->hwdata = NULL;
    }

    if (prev) {
        prev->next = item->next;
    } else {
        SDL_assert(SDL_joylist == item);
        SDL_joylist = item->next;
    }
    if (item == SDL_joylist_tail) {
        SDL_joylist_tail = prev;
    }

    /* Need to decrement the joystick count before we post the event */
    --numjoysticks;

    SDL_PrivateJoystickRemoved(item->device_instance);

    result = numjoysticks;

#ifdef DEBUG_JOYSTICK
    SDL_Log("Removed joystick with device_id %d", device_id);
#endif

    SDL_free(item->name);
    SDL_free(item);

done:
    SDL_UnlockJoysticks();

    return result;
}

static void ANDROID_JoystickDetect(void);

static int ANDROID_JoystickInit(void)
{
    ANDROID_JoystickDetect();

    if (SDL_GetHintBoolean(SDL_HINT_ACCELEROMETER_AS_JOYSTICK, SDL_TRUE)) {
        /* Default behavior, accelerometer as joystick */
        Android_AddJoystick(ANDROID_ACCELEROMETER_DEVICE_ID, ANDROID_ACCELEROMETER_NAME, ANDROID_ACCELEROMETER_NAME, 0, 0, SDL_TRUE, 0, 3, 0x0003, 0, 0);
    }
    return 0;
}

static int ANDROID_JoystickGetCount(void)
{
    return numjoysticks;
}

static void ANDROID_JoystickDetect(void)
{
    /* Support for device connect/disconnect is API >= 16 only,
     * so we poll every three seconds
     * Ref: http://developer.android.com/reference/android/hardware/input/InputManager.InputDeviceListener.html
     */
    static Uint32 timeout = 0;
    if (!timeout || SDL_TICKS_PASSED(SDL_GetTicks(), timeout)) {
        timeout = SDL_GetTicks() + 3000;
        Android_JNI_PollInputDevices();
    }
}

static SDL_joylist_item *JoystickByDevIndex(int device_index)
{
    SDL_joylist_item *item = SDL_joylist;

    if ((device_index < 0) || (device_index >= numjoysticks)) {
        return NULL;
    }

    while (device_index > 0) {
        SDL_assert(item != NULL);
        device_index--;
        item = item->next;
    }

    return item;
}

static SDL_joylist_item *JoystickByDeviceId(int device_id)
{
    SDL_joylist_item *item = SDL_joylist;

    while (item) {
        if (item->device_id == device_id) {
            return item;
        }
        item = item->next;
    }

    /* Joystick not found, try adding it */
    ANDROID_JoystickDetect();

    while (item) {
        if (item->device_id == device_id) {
            return item;
        }
        item = item->next;
    }

    return NULL;
}

static const char *ANDROID_JoystickGetDeviceName(int device_index)
{
    return JoystickByDevIndex(device_index)->name;
}

static const char *ANDROID_JoystickGetDevicePath(int device_index)
{
    return NULL;
}

static int ANDROID_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    return -1;
}

static int ANDROID_JoystickGetDevicePlayerIndex(int device_index)
{
    return -1;
}

static void ANDROID_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
}

static SDL_JoystickGUID ANDROID_JoystickGetDeviceGUID(int device_index)
{
    return JoystickByDevIndex(device_index)->guid;
}

static SDL_JoystickID ANDROID_JoystickGetDeviceInstanceID(int device_index)
{
    return JoystickByDevIndex(device_index)->device_instance;
}

static int ANDROID_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    SDL_joylist_item *item = JoystickByDevIndex(device_index);

    if (!item) {
        return SDL_SetError("No such device");
    }

    if (item->joystick) {
        return SDL_SetError("Joystick already opened");
    }

    joystick->instance_id = item->device_instance;
    joystick->hwdata = (struct joystick_hwdata *)item;
    item->joystick = joystick;
    joystick->nhats = item->nhats;
    joystick->nballs = item->nballs;
    joystick->nbuttons = item->nbuttons;
    joystick->naxes = item->naxes;

    return 0;
}

static int ANDROID_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    return SDL_Unsupported();
}

static int ANDROID_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 ANDROID_JoystickGetCapabilities(SDL_Joystick *joystick)
{
    return 0;
}

static int ANDROID_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static int ANDROID_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static int ANDROID_JoystickSetSensorsEnabled(SDL_Joystick *joystick, SDL_bool enabled)
{
    return SDL_Unsupported();
}

static void ANDROID_JoystickUpdate(SDL_Joystick *joystick)
{
    SDL_joylist_item *item = (SDL_joylist_item *)joystick->hwdata;

    if (!item) {
        return;
    }

    if (item->is_accelerometer) {
        int i;
        Sint16 value;
        float values[3];

        if (Android_JNI_GetAccelerometerValues(values)) {
            for (i = 0; i < 3; i++) {
                if (values[i] > 1.0f) {
                    values[i] = 1.0f;
                } else if (values[i] < -1.0f) {
                    values[i] = -1.0f;
                }

                value = (Sint16)(values[i] * 32767.0f);
                SDL_PrivateJoystickAxis(item->joystick, i, value);
            }
        }
    }
}

static void ANDROID_JoystickClose(SDL_Joystick *joystick)
{
    SDL_joylist_item *item = (SDL_joylist_item *)joystick->hwdata;
    if (item) {
        item->joystick = NULL;
    }
}

static void ANDROID_JoystickQuit(void)
{
/* We don't have any way to scan for joysticks at init, so don't wipe the list
 * of joysticks here in case this is a reinit.
 */
#if 0
    SDL_joylist_item *item = NULL;
    SDL_joylist_item *next = NULL;

    for (item = SDL_joylist; item; item = next) {
        next = item->next;
        SDL_free(item->name);
        SDL_free(item);
    }

    SDL_joylist = SDL_joylist_tail = NULL;

    numjoysticks = 0;
#endif /* 0 */
}

static SDL_bool ANDROID_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    return SDL_FALSE;
}

SDL_JoystickDriver SDL_ANDROID_JoystickDriver = {
    ANDROID_JoystickInit,
    ANDROID_JoystickGetCount,
    ANDROID_JoystickDetect,
    ANDROID_JoystickGetDeviceName,
    ANDROID_JoystickGetDevicePath,
    ANDROID_JoystickGetDeviceSteamVirtualGamepadSlot,
    ANDROID_JoystickGetDevicePlayerIndex,
    ANDROID_JoystickSetDevicePlayerIndex,
    ANDROID_JoystickGetDeviceGUID,
    ANDROID_JoystickGetDeviceInstanceID,
    ANDROID_JoystickOpen,
    ANDROID_JoystickRumble,
    ANDROID_JoystickRumbleTriggers,
    ANDROID_JoystickGetCapabilities,
    ANDROID_JoystickSetLED,
    ANDROID_JoystickSendEffect,
    ANDROID_JoystickSetSensorsEnabled,
    ANDROID_JoystickUpdate,
    ANDROID_JoystickClose,
    ANDROID_JoystickQuit,
    ANDROID_JoystickGetGamepadMapping
};

#endif /* SDL_JOYSTICK_ANDROID */

/* vi: set ts=4 sw=4 expandtab: */
