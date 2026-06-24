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

#ifdef SDL_JOYSTICK_VIRTUAL

/* This is the virtual implementation of the SDL joystick API */

#include "SDL_endian.h"
#include "SDL_virtualjoystick_c.h"
#include "../SDL_sysjoystick.h"
#include "../SDL_joystick_c.h"

static joystick_hwdata *g_VJoys SDL_GUARDED_BY(SDL_joystick_lock) = NULL;

static joystick_hwdata *VIRTUAL_HWDataForIndex(int device_index)
{
    joystick_hwdata *vjoy;

    SDL_AssertJoysticksLocked();

    for (vjoy = g_VJoys; vjoy; vjoy = vjoy->next) {
        if (device_index == 0) {
            break;
        }
        --device_index;
    }
    return vjoy;
}

static void VIRTUAL_FreeHWData(joystick_hwdata *hwdata)
{
    joystick_hwdata *cur;
    joystick_hwdata *prev = NULL;

    SDL_AssertJoysticksLocked();

    if (!hwdata) {
        return;
    }

    /* Remove hwdata from SDL-global list */
    for (cur = g_VJoys; cur; prev = cur, cur = cur->next) {
        if (hwdata == cur) {
            if (prev) {
                prev->next = cur->next;
            } else {
                g_VJoys = cur->next;
            }
            break;
        }
    }

    if (hwdata->joystick) {
        hwdata->joystick->hwdata = NULL;
        hwdata->joystick = NULL;
    }
    if (hwdata->name) {
        SDL_free(hwdata->name);
        hwdata->name = NULL;
    }
    if (hwdata->axes) {
        SDL_free((void *)hwdata->axes);
        hwdata->axes = NULL;
    }
    if (hwdata->buttons) {
        SDL_free((void *)hwdata->buttons);
        hwdata->buttons = NULL;
    }
    if (hwdata->hats) {
        SDL_free(hwdata->hats);
        hwdata->hats = NULL;
    }
    SDL_free(hwdata);
}

int SDL_JoystickAttachVirtualInner(const SDL_VirtualJoystickDesc *desc)
{
    joystick_hwdata *hwdata = NULL;
    int device_index = -1;
    const char *name = NULL;
    int axis_triggerleft = -1;
    int axis_triggerright = -1;

    SDL_AssertJoysticksLocked();

    if (!desc) {
        return SDL_InvalidParamError("desc");
    }
    if (desc->version != SDL_VIRTUAL_JOYSTICK_DESC_VERSION) {
        /* Is this an old version that we can support? */
        return SDL_SetError("Unsupported virtual joystick description version %d", desc->version);
    }

    hwdata = SDL_calloc(1, sizeof(joystick_hwdata));
    if (!hwdata) {
        VIRTUAL_FreeHWData(hwdata);
        return SDL_OutOfMemory();
    }
    SDL_memcpy(&hwdata->desc, desc, sizeof(*desc));

    if (hwdata->desc.name) {
        name = hwdata->desc.name;
    } else {
        switch (hwdata->desc.type) {
        case SDL_JOYSTICK_TYPE_GAMECONTROLLER:
            name = "Virtual Controller";
            break;
        case SDL_JOYSTICK_TYPE_WHEEL:
            name = "Virtual Wheel";
            break;
        case SDL_JOYSTICK_TYPE_ARCADE_STICK:
            name = "Virtual Arcade Stick";
            break;
        case SDL_JOYSTICK_TYPE_FLIGHT_STICK:
            name = "Virtual Flight Stick";
            break;
        case SDL_JOYSTICK_TYPE_DANCE_PAD:
            name = "Virtual Dance Pad";
            break;
        case SDL_JOYSTICK_TYPE_GUITAR:
            name = "Virtual Guitar";
            break;
        case SDL_JOYSTICK_TYPE_DRUM_KIT:
            name = "Virtual Drum Kit";
            break;
        case SDL_JOYSTICK_TYPE_ARCADE_PAD:
            name = "Virtual Arcade Pad";
            break;
        case SDL_JOYSTICK_TYPE_THROTTLE:
            name = "Virtual Throttle";
            break;
        default:
            name = "Virtual Joystick";
            break;
        }
    }
    hwdata->name = SDL_strdup(name);

    if (hwdata->desc.type == SDL_JOYSTICK_TYPE_GAMECONTROLLER) {
        int i, axis;

        if (hwdata->desc.button_mask == 0) {
            for (i = 0; i < hwdata->desc.nbuttons && i < sizeof(hwdata->desc.button_mask) * 8; ++i) {
                hwdata->desc.button_mask |= (1 << i);
            }
        }

        if (hwdata->desc.axis_mask == 0) {
            if (hwdata->desc.naxes >= 2) {
                hwdata->desc.axis_mask |= ((1 << SDL_CONTROLLER_AXIS_LEFTX) | (1 << SDL_CONTROLLER_AXIS_LEFTY));
            }
            if (hwdata->desc.naxes >= 4) {
                hwdata->desc.axis_mask |= ((1 << SDL_CONTROLLER_AXIS_RIGHTX) | (1 << SDL_CONTROLLER_AXIS_RIGHTY));
            }
            if (hwdata->desc.naxes >= 6) {
                hwdata->desc.axis_mask |= ((1 << SDL_CONTROLLER_AXIS_TRIGGERLEFT) | (1 << SDL_CONTROLLER_AXIS_TRIGGERRIGHT));
            }
        }

        /* Find the trigger axes */
        axis = 0;
        for (i = 0; axis < hwdata->desc.naxes && i < SDL_CONTROLLER_AXIS_MAX; ++i) {
            if (hwdata->desc.axis_mask & (1 << i)) {
                if (i == SDL_CONTROLLER_AXIS_TRIGGERLEFT) {
                    axis_triggerleft = axis;
                }
                if (i == SDL_CONTROLLER_AXIS_TRIGGERRIGHT) {
                    axis_triggerright = axis;
                }
                ++axis;
            }
        }
    }

    hwdata->guid = SDL_CreateJoystickGUID(SDL_HARDWARE_BUS_VIRTUAL, hwdata->desc.vendor_id, hwdata->desc.product_id, 0, NULL, name, 'v', (Uint8)hwdata->desc.type);

    /* Allocate fields for different control-types */
    if (hwdata->desc.naxes > 0) {
        hwdata->axes = SDL_calloc(hwdata->desc.naxes, sizeof(Sint16));
        if (!hwdata->axes) {
            VIRTUAL_FreeHWData(hwdata);
            return SDL_OutOfMemory();
        }

        /* Trigger axes are at minimum value at rest */
        if (axis_triggerleft >= 0) {
            hwdata->axes[axis_triggerleft] = SDL_JOYSTICK_AXIS_MIN;
        }
        if (axis_triggerright >= 0) {
            hwdata->axes[axis_triggerright] = SDL_JOYSTICK_AXIS_MIN;
        }
    }
    if (hwdata->desc.nbuttons > 0) {
        hwdata->buttons = SDL_calloc(hwdata->desc.nbuttons, sizeof(Uint8));
        if (!hwdata->buttons) {
            VIRTUAL_FreeHWData(hwdata);
            return SDL_OutOfMemory();
        }
    }
    if (hwdata->desc.nhats > 0) {
        hwdata->hats = SDL_calloc(hwdata->desc.nhats, sizeof(Uint8));
        if (!hwdata->hats) {
            VIRTUAL_FreeHWData(hwdata);
            return SDL_OutOfMemory();
        }
    }

    /* Allocate an instance ID for this device */
    hwdata->instance_id = SDL_GetNextJoystickInstanceID();

    /* Add virtual joystick to SDL-global lists */
    if (g_VJoys) {
        joystick_hwdata *last;

        for (last = g_VJoys; last->next; last = last->next) {
        }
        last->next = hwdata;
    } else {
        g_VJoys = hwdata;
    }
    SDL_PrivateJoystickAdded(hwdata->instance_id);

    /* Return the new virtual-device's index */
    device_index = SDL_JoystickGetDeviceIndexFromInstanceID(hwdata->instance_id);
    return device_index;
}

int SDL_JoystickDetachVirtualInner(int device_index)
{
    SDL_JoystickID instance_id;
    joystick_hwdata *hwdata = VIRTUAL_HWDataForIndex(device_index);
    if (!hwdata) {
        return SDL_SetError("Virtual joystick data not found");
    }
    instance_id = hwdata->instance_id;
    VIRTUAL_FreeHWData(hwdata);
    SDL_PrivateJoystickRemoved(instance_id);
    return 0;
}

int SDL_JoystickSetVirtualAxisInner(SDL_Joystick *joystick, int axis, Sint16 value)
{
    joystick_hwdata *hwdata;

    SDL_LockJoysticks();

    if (!joystick || !joystick->hwdata) {
        SDL_UnlockJoysticks();
        return SDL_SetError("Invalid joystick");
    }

    hwdata = (joystick_hwdata *)joystick->hwdata;
    if (axis < 0 || axis >= hwdata->desc.naxes) {
        SDL_UnlockJoysticks();
        return SDL_SetError("Invalid axis index");
    }

    hwdata->axes[axis] = value;

    SDL_UnlockJoysticks();
    return 0;
}

int SDL_JoystickSetVirtualButtonInner(SDL_Joystick *joystick, int button, Uint8 value)
{
    joystick_hwdata *hwdata;

    SDL_LockJoysticks();

    if (!joystick || !joystick->hwdata) {
        SDL_UnlockJoysticks();
        return SDL_SetError("Invalid joystick");
    }

    hwdata = (joystick_hwdata *)joystick->hwdata;
    if (button < 0 || button >= hwdata->desc.nbuttons) {
        SDL_UnlockJoysticks();
        return SDL_SetError("Invalid button index");
    }

    hwdata->buttons[button] = value;

    SDL_UnlockJoysticks();
    return 0;
}

int SDL_JoystickSetVirtualHatInner(SDL_Joystick *joystick, int hat, Uint8 value)
{
    joystick_hwdata *hwdata;

    SDL_LockJoysticks();

    if (!joystick || !joystick->hwdata) {
        SDL_UnlockJoysticks();
        return SDL_SetError("Invalid joystick");
    }

    hwdata = (joystick_hwdata *)joystick->hwdata;
    if (hat < 0 || hat >= hwdata->desc.nhats) {
        SDL_UnlockJoysticks();
        return SDL_SetError("Invalid hat index");
    }

    hwdata->hats[hat] = value;

    SDL_UnlockJoysticks();
    return 0;
}

static int VIRTUAL_JoystickInit(void)
{
    return 0;
}

static int VIRTUAL_JoystickGetCount(void)
{
    joystick_hwdata *cur;
    int count = 0;

    SDL_AssertJoysticksLocked();

    for (cur = g_VJoys; cur; cur = cur->next) {
        ++count;
    }
    return count;
}

static void VIRTUAL_JoystickDetect(void)
{
}

static const char *VIRTUAL_JoystickGetDeviceName(int device_index)
{
    joystick_hwdata *hwdata = VIRTUAL_HWDataForIndex(device_index);
    if (!hwdata) {
        return NULL;
    }
    return hwdata->name;
}

static const char *VIRTUAL_JoystickGetDevicePath(int device_index)
{
    return NULL;
}

static int VIRTUAL_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    return -1;
}

static int VIRTUAL_JoystickGetDevicePlayerIndex(int device_index)
{
    return -1;
}

static void VIRTUAL_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
    joystick_hwdata *hwdata = VIRTUAL_HWDataForIndex(device_index);

    if (hwdata && hwdata->desc.SetPlayerIndex) {
        hwdata->desc.SetPlayerIndex(hwdata->desc.userdata, player_index);
    }
}

static SDL_JoystickGUID VIRTUAL_JoystickGetDeviceGUID(int device_index)
{
    joystick_hwdata *hwdata = VIRTUAL_HWDataForIndex(device_index);
    if (!hwdata) {
        SDL_JoystickGUID guid;
        SDL_zero(guid);
        return guid;
    }
    return hwdata->guid;
}

static SDL_JoystickID VIRTUAL_JoystickGetDeviceInstanceID(int device_index)
{
    joystick_hwdata *hwdata = VIRTUAL_HWDataForIndex(device_index);
    if (!hwdata) {
        return -1;
    }
    return hwdata->instance_id;
}

static int VIRTUAL_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    joystick_hwdata *hwdata;

    SDL_AssertJoysticksLocked();

    hwdata = VIRTUAL_HWDataForIndex(device_index);
    if (!hwdata) {
        return SDL_SetError("No such device");
    }
    joystick->instance_id = hwdata->instance_id;
    joystick->hwdata = hwdata;
    joystick->naxes = hwdata->desc.naxes;
    joystick->nbuttons = hwdata->desc.nbuttons;
    joystick->nhats = hwdata->desc.nhats;
    hwdata->joystick = joystick;
    return 0;
}

static int VIRTUAL_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    int result;

    SDL_AssertJoysticksLocked();

    if (joystick->hwdata) {
        joystick_hwdata *hwdata = joystick->hwdata;
        if (hwdata->desc.Rumble) {
            result = hwdata->desc.Rumble(hwdata->desc.userdata, low_frequency_rumble, high_frequency_rumble);
        } else {
            result = SDL_Unsupported();
        }
    } else {
        result = SDL_SetError("Rumble failed, device disconnected");
    }

    return result;
}

static int VIRTUAL_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    int result;

    SDL_AssertJoysticksLocked();

    if (joystick->hwdata) {
        joystick_hwdata *hwdata = joystick->hwdata;
        if (hwdata->desc.RumbleTriggers) {
            result = hwdata->desc.RumbleTriggers(hwdata->desc.userdata, left_rumble, right_rumble);
        } else {
            result = SDL_Unsupported();
        }
    } else {
        result = SDL_SetError("Rumble failed, device disconnected");
    }

    return result;
}

static Uint32 VIRTUAL_JoystickGetCapabilities(SDL_Joystick *joystick)
{
    joystick_hwdata *hwdata;
    Uint32 caps = 0;

    SDL_AssertJoysticksLocked();

    hwdata = joystick->hwdata;
    if (hwdata) {
        if (hwdata->desc.Rumble) {
            caps |= SDL_JOYCAP_RUMBLE;
        }
        if (hwdata->desc.RumbleTriggers) {
            caps |= SDL_JOYCAP_RUMBLE_TRIGGERS;
        }
        if (hwdata->desc.SetLED) {
            caps |= SDL_JOYCAP_LED;
        }
    }
    return caps;
}

static int VIRTUAL_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    int result;

    SDL_AssertJoysticksLocked();

    if (joystick->hwdata) {
        joystick_hwdata *hwdata = joystick->hwdata;
        if (hwdata->desc.SetLED) {
            result = hwdata->desc.SetLED(hwdata->desc.userdata, red, green, blue);
        } else {
            result = SDL_Unsupported();
        }
    } else {
        result = SDL_SetError("SetLED failed, device disconnected");
    }

    return result;
}

static int VIRTUAL_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    int result;

    SDL_AssertJoysticksLocked();

    if (joystick->hwdata) {
        joystick_hwdata *hwdata = joystick->hwdata;
        if (hwdata->desc.SendEffect) {
            result = hwdata->desc.SendEffect(hwdata->desc.userdata, data, size);
        } else {
            result = SDL_Unsupported();
        }
    } else {
        result = SDL_SetError("SendEffect failed, device disconnected");
    }

    return result;
}

static int VIRTUAL_JoystickSetSensorsEnabled(SDL_Joystick *joystick, SDL_bool enabled)
{
    return SDL_Unsupported();
}

static void VIRTUAL_JoystickUpdate(SDL_Joystick *joystick)
{
    joystick_hwdata *hwdata;
    int i;

    SDL_AssertJoysticksLocked();

    if (!joystick) {
        return;
    }
    if (!joystick->hwdata) {
        return;
    }

    hwdata = (joystick_hwdata *)joystick->hwdata;

    if (hwdata->desc.Update) {
        hwdata->desc.Update(hwdata->desc.userdata);
    }

    for (i = 0; i < hwdata->desc.naxes; ++i) {
        SDL_PrivateJoystickAxis(joystick, i, hwdata->axes[i]);
    }
    for (i = 0; i < hwdata->desc.nbuttons; ++i) {
        SDL_PrivateJoystickButton(joystick, i, hwdata->buttons[i]);
    }
    for (i = 0; i < hwdata->desc.nhats; ++i) {
        SDL_PrivateJoystickHat(joystick, i, hwdata->hats[i]);
    }
}

static void VIRTUAL_JoystickClose(SDL_Joystick *joystick)
{
    SDL_AssertJoysticksLocked();

    if (joystick->hwdata) {
        joystick_hwdata *hwdata = joystick->hwdata;
        hwdata->joystick = NULL;
        joystick->hwdata = NULL;
    }
}

static void VIRTUAL_JoystickQuit(void)
{
    SDL_AssertJoysticksLocked();

    while (g_VJoys) {
        VIRTUAL_FreeHWData(g_VJoys);
    }
}

static SDL_bool VIRTUAL_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    joystick_hwdata *hwdata = VIRTUAL_HWDataForIndex(device_index);
    int current_button = 0;
    int current_axis = 0;

    if (hwdata->desc.type != SDL_JOYSTICK_TYPE_GAMECONTROLLER) {
        return SDL_FALSE;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_A))) {
        out->a.kind = EMappingKind_Button;
        out->a.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_B))) {
        out->b.kind = EMappingKind_Button;
        out->b.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_X))) {
        out->x.kind = EMappingKind_Button;
        out->x.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_Y))) {
        out->y.kind = EMappingKind_Button;
        out->y.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_BACK))) {
        out->back.kind = EMappingKind_Button;
        out->back.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_GUIDE))) {
        out->guide.kind = EMappingKind_Button;
        out->guide.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_START))) {
        out->start.kind = EMappingKind_Button;
        out->start.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_LEFTSTICK))) {
        out->leftstick.kind = EMappingKind_Button;
        out->leftstick.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_RIGHTSTICK))) {
        out->rightstick.kind = EMappingKind_Button;
        out->rightstick.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_LEFTSHOULDER))) {
        out->leftshoulder.kind = EMappingKind_Button;
        out->leftshoulder.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_RIGHTSHOULDER))) {
        out->rightshoulder.kind = EMappingKind_Button;
        out->rightshoulder.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_DPAD_UP))) {
        out->dpup.kind = EMappingKind_Button;
        out->dpup.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_DPAD_DOWN))) {
        out->dpdown.kind = EMappingKind_Button;
        out->dpdown.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_DPAD_LEFT))) {
        out->dpleft.kind = EMappingKind_Button;
        out->dpleft.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_DPAD_RIGHT))) {
        out->dpright.kind = EMappingKind_Button;
        out->dpright.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_MISC1))) {
        out->misc1.kind = EMappingKind_Button;
        out->misc1.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_PADDLE1))) {
        out->paddle1.kind = EMappingKind_Button;
        out->paddle1.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_PADDLE2))) {
        out->paddle2.kind = EMappingKind_Button;
        out->paddle2.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_PADDLE3))) {
        out->paddle3.kind = EMappingKind_Button;
        out->paddle3.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_CONTROLLER_BUTTON_PADDLE4))) {
        out->paddle4.kind = EMappingKind_Button;
        out->paddle4.target = current_button++;
    }

    if (current_axis < hwdata->desc.naxes && (hwdata->desc.axis_mask & (1 << SDL_CONTROLLER_AXIS_LEFTX))) {
        out->leftx.kind = EMappingKind_Axis;
        out->leftx.target = current_axis++;
    }

    if (current_axis < hwdata->desc.naxes && (hwdata->desc.axis_mask & (1 << SDL_CONTROLLER_AXIS_LEFTY))) {
        out->lefty.kind = EMappingKind_Axis;
        out->lefty.target = current_axis++;
    }

    if (current_axis < hwdata->desc.naxes && (hwdata->desc.axis_mask & (1 << SDL_CONTROLLER_AXIS_RIGHTX))) {
        out->rightx.kind = EMappingKind_Axis;
        out->rightx.target = current_axis++;
    }

    if (current_axis < hwdata->desc.naxes && (hwdata->desc.axis_mask & (1 << SDL_CONTROLLER_AXIS_RIGHTY))) {
        out->righty.kind = EMappingKind_Axis;
        out->righty.target = current_axis++;
    }

    if (current_axis < hwdata->desc.naxes && (hwdata->desc.axis_mask & (1 << SDL_CONTROLLER_AXIS_TRIGGERLEFT))) {
        out->lefttrigger.kind = EMappingKind_Axis;
        out->lefttrigger.target = current_axis++;
    }

    if (current_axis < hwdata->desc.naxes && (hwdata->desc.axis_mask & (1 << SDL_CONTROLLER_AXIS_TRIGGERRIGHT))) {
        out->righttrigger.kind = EMappingKind_Axis;
        out->righttrigger.target = current_axis++;
    }

    return SDL_TRUE;
}

SDL_JoystickDriver SDL_VIRTUAL_JoystickDriver = {
    VIRTUAL_JoystickInit,
    VIRTUAL_JoystickGetCount,
    VIRTUAL_JoystickDetect,
    VIRTUAL_JoystickGetDeviceName,
    VIRTUAL_JoystickGetDevicePath,
    VIRTUAL_JoystickGetDeviceSteamVirtualGamepadSlot,
    VIRTUAL_JoystickGetDevicePlayerIndex,
    VIRTUAL_JoystickSetDevicePlayerIndex,
    VIRTUAL_JoystickGetDeviceGUID,
    VIRTUAL_JoystickGetDeviceInstanceID,
    VIRTUAL_JoystickOpen,
    VIRTUAL_JoystickRumble,
    VIRTUAL_JoystickRumbleTriggers,
    VIRTUAL_JoystickGetCapabilities,
    VIRTUAL_JoystickSetLED,
    VIRTUAL_JoystickSendEffect,
    VIRTUAL_JoystickSetSensorsEnabled,
    VIRTUAL_JoystickUpdate,
    VIRTUAL_JoystickClose,
    VIRTUAL_JoystickQuit,
    VIRTUAL_JoystickGetGamepadMapping
};

#endif /* SDL_JOYSTICK_VIRTUAL */

/* vi: set ts=4 sw=4 expandtab: */
