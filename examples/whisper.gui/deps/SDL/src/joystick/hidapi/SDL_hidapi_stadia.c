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

#ifdef SDL_JOYSTICK_HIDAPI

#include "SDL_events.h"
#include "SDL_joystick.h"
#include "SDL_gamecontroller.h"
#include "../SDL_sysjoystick.h"
#include "SDL_hidapijoystick_c.h"
#include "SDL_hidapi_rumble.h"

#ifdef SDL_JOYSTICK_HIDAPI_STADIA

/* Define this if you want to log all packets from the controller */
/*#define DEBUG_STADIA_PROTOCOL*/

enum
{
    SDL_CONTROLLER_BUTTON_STADIA_SHARE = 15,
    SDL_CONTROLLER_BUTTON_STADIA_GOOGLE_ASSISTANT,
    SDL_CONTROLLER_NUM_STADIA_BUTTONS,
};

typedef struct
{
    SDL_bool rumble_supported;
    Uint8 last_state[USB_PACKET_LENGTH];
} SDL_DriverStadia_Context;

static void HIDAPI_DriverStadia_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_STADIA, callback, userdata);
}

static void HIDAPI_DriverStadia_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_DelHintCallback(SDL_HINT_JOYSTICK_HIDAPI_STADIA, callback, userdata);
}

static SDL_bool HIDAPI_DriverStadia_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_STADIA, SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT));
}

static SDL_bool HIDAPI_DriverStadia_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GameControllerType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    return (type == SDL_CONTROLLER_TYPE_GOOGLE_STADIA) ? SDL_TRUE : SDL_FALSE;
}

static SDL_bool HIDAPI_DriverStadia_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverStadia_Context *ctx;

    ctx = (SDL_DriverStadia_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        SDL_OutOfMemory();
        return SDL_FALSE;
    }
    device->context = ctx;

    /* Check whether rumble is supported */
    {
        Uint8 rumble_packet[] = { 0x05, 0x00, 0x00, 0x00, 0x00 };

        if (SDL_hid_write(device->dev, rumble_packet, sizeof(rumble_packet)) >= 0) {
            ctx->rumble_supported = SDL_TRUE;
        }
    }

    device->type = SDL_CONTROLLER_TYPE_GOOGLE_STADIA;
    HIDAPI_SetDeviceName(device, "Google Stadia Controller");

    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverStadia_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverStadia_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
}

static SDL_bool HIDAPI_DriverStadia_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverStadia_Context *ctx = (SDL_DriverStadia_Context *)device->context;

    SDL_AssertJoysticksLocked();

    SDL_zeroa(ctx->last_state);

    /* Initialize the joystick capabilities */
    joystick->nbuttons = SDL_CONTROLLER_NUM_STADIA_BUTTONS;
    joystick->naxes = SDL_CONTROLLER_AXIS_MAX;
    joystick->epowerlevel = SDL_JOYSTICK_POWER_WIRED;

    return SDL_TRUE;
}

static int HIDAPI_DriverStadia_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    SDL_DriverStadia_Context *ctx = (SDL_DriverStadia_Context *)device->context;

    if (ctx->rumble_supported) {
        Uint8 rumble_packet[] = { 0x05, 0x00, 0x00, 0x00, 0x00 };


        rumble_packet[1] = (low_frequency_rumble & 0xFF);
        rumble_packet[2] = (low_frequency_rumble >> 8);
        rumble_packet[3] = (high_frequency_rumble & 0xFF);
        rumble_packet[4] = (high_frequency_rumble >> 8);

        if (SDL_HIDAPI_SendRumble(device, rumble_packet, sizeof(rumble_packet)) != sizeof(rumble_packet)) {
            return SDL_SetError("Couldn't send rumble packet");
        }
        return 0;
    } else {
        return SDL_Unsupported();
    }
}

static int HIDAPI_DriverStadia_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverStadia_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverStadia_Context *ctx = (SDL_DriverStadia_Context *)device->context;
    Uint32 caps = 0;

    if (ctx->rumble_supported) {
        caps |= SDL_JOYCAP_RUMBLE;
    }
    return caps;
}

static int HIDAPI_DriverStadia_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static int HIDAPI_DriverStadia_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static int HIDAPI_DriverStadia_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, SDL_bool enabled)
{
    return SDL_Unsupported();
}

static void HIDAPI_DriverStadia_HandleStatePacket(SDL_Joystick *joystick, SDL_DriverStadia_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;

    // The format is the same but the original FW will send 10 bytes and January '21 FW update will send 11
    if (size < 10 || data[0] != 0x03) {
        /* We don't know how to handle this report */
        return;
    }

    if (ctx->last_state[1] != data[1]) {
        SDL_bool dpad_up = SDL_FALSE;
        SDL_bool dpad_down = SDL_FALSE;
        SDL_bool dpad_left = SDL_FALSE;
        SDL_bool dpad_right = SDL_FALSE;

        switch (data[1]) {
        case 0:
            dpad_up = SDL_TRUE;
            break;
        case 1:
            dpad_up = SDL_TRUE;
            dpad_right = SDL_TRUE;
            break;
        case 2:
            dpad_right = SDL_TRUE;
            break;
        case 3:
            dpad_right = SDL_TRUE;
            dpad_down = SDL_TRUE;
            break;
        case 4:
            dpad_down = SDL_TRUE;
            break;
        case 5:
            dpad_left = SDL_TRUE;
            dpad_down = SDL_TRUE;
            break;
        case 6:
            dpad_left = SDL_TRUE;
            break;
        case 7:
            dpad_up = SDL_TRUE;
            dpad_left = SDL_TRUE;
            break;
        default:
            break;
        }
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_DOWN, dpad_down);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_UP, dpad_up);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_RIGHT, dpad_right);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_LEFT, dpad_left);
    }

    if (ctx->last_state[2] != data[2]) {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_BACK, (data[2] & 0x40) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_GUIDE, (data[2] & 0x10) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_START, (data[2] & 0x20) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSTICK, (data[2] & 0x80) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_STADIA_SHARE, (data[2] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_STADIA_GOOGLE_ASSISTANT, (data[2] & 0x02) ? SDL_PRESSED : SDL_RELEASED);
    }

    if (ctx->last_state[3] != data[3]) {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_A, (data[3] & 0x40) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_B, (data[3] & 0x20) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_X, (data[3] & 0x10) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_Y, (data[3] & 0x08) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSHOULDER, (data[3] & 0x04) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSHOULDER, (data[3] & 0x02) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSTICK, (data[3] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
    }

#define READ_STICK_AXIS(offset) \
    (data[offset] == 0x80 ? 0 : (Sint16)HIDAPI_RemapVal((float)((int)data[offset] - 0x80), 0x01 - 0x80, 0xff - 0x80, SDL_MIN_SINT16, SDL_MAX_SINT16))
    {
        axis = READ_STICK_AXIS(4);
        SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_LEFTX, axis);
        axis = READ_STICK_AXIS(5);
        SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_LEFTY, axis);
        axis = READ_STICK_AXIS(6);
        SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_RIGHTX, axis);
        axis = READ_STICK_AXIS(7);
        SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_RIGHTY, axis);
    }
#undef READ_STICK_AXIS

#define READ_TRIGGER_AXIS(offset) \
    (Sint16)(((int)data[offset] * 257) - 32768)
    {
        axis = READ_TRIGGER_AXIS(8);
        SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_TRIGGERLEFT, axis);
        axis = READ_TRIGGER_AXIS(9);
        SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_TRIGGERRIGHT, axis);
    }
#undef READ_TRIGGER_AXIS

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static SDL_bool HIDAPI_DriverStadia_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverStadia_Context *ctx = (SDL_DriverStadia_Context *)device->context;
    SDL_Joystick *joystick = NULL;
    Uint8 data[USB_PACKET_LENGTH];
    int size = 0;

    if (device->num_joysticks > 0) {
        joystick = SDL_JoystickFromInstanceID(device->joysticks[0]);
    } else {
        return SDL_FALSE;
    }

    while ((size = SDL_hid_read_timeout(device->dev, data, sizeof(data), 0)) > 0) {
#ifdef DEBUG_STADIA_PROTOCOL
        HIDAPI_DumpPacket("Google Stadia packet: size = %d", data, size);
#endif
        if (!joystick) {
            continue;
        }

        HIDAPI_DriverStadia_HandleStatePacket(joystick, ctx, data, size);
    }

    if (size < 0) {
        /* Read error, device is disconnected */
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    return size >= 0;
}

static void HIDAPI_DriverStadia_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
}

static void HIDAPI_DriverStadia_FreeDevice(SDL_HIDAPI_Device *device)
{
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverStadia = {
    SDL_HINT_JOYSTICK_HIDAPI_STADIA,
    SDL_TRUE,
    HIDAPI_DriverStadia_RegisterHints,
    HIDAPI_DriverStadia_UnregisterHints,
    HIDAPI_DriverStadia_IsEnabled,
    HIDAPI_DriverStadia_IsSupportedDevice,
    HIDAPI_DriverStadia_InitDevice,
    HIDAPI_DriverStadia_GetDevicePlayerIndex,
    HIDAPI_DriverStadia_SetDevicePlayerIndex,
    HIDAPI_DriverStadia_UpdateDevice,
    HIDAPI_DriverStadia_OpenJoystick,
    HIDAPI_DriverStadia_RumbleJoystick,
    HIDAPI_DriverStadia_RumbleJoystickTriggers,
    HIDAPI_DriverStadia_GetJoystickCapabilities,
    HIDAPI_DriverStadia_SetJoystickLED,
    HIDAPI_DriverStadia_SendJoystickEffect,
    HIDAPI_DriverStadia_SetJoystickSensorsEnabled,
    HIDAPI_DriverStadia_CloseJoystick,
    HIDAPI_DriverStadia_FreeDevice,
};

#endif /* SDL_JOYSTICK_HIDAPI_STADIA */

#endif /* SDL_JOYSTICK_HIDAPI */

/* vi: set ts=4 sw=4 expandtab: */
