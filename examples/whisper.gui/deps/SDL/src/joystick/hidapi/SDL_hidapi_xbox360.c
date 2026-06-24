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
#include "SDL_timer.h"
#include "SDL_joystick.h"
#include "SDL_gamecontroller.h"
#include "../../SDL_hints_c.h"
#include "../SDL_sysjoystick.h"
#include "SDL_hidapijoystick_c.h"
#include "SDL_hidapi_rumble.h"

#ifdef SDL_JOYSTICK_HIDAPI_XBOX360

/* Define this if you want to log all packets from the controller */
/*#define DEBUG_XBOX_PROTOCOL*/

typedef struct
{
    SDL_HIDAPI_Device *device;
    SDL_Joystick *joystick;
    int player_index;
    SDL_bool player_lights;
    Uint8 last_state[USB_PACKET_LENGTH];
} SDL_DriverXbox360_Context;

static void HIDAPI_DriverXbox360_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_XBOX, callback, userdata);
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_XBOX_360, callback, userdata);
}

static void HIDAPI_DriverXbox360_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_DelHintCallback(SDL_HINT_JOYSTICK_HIDAPI_XBOX, callback, userdata);
    SDL_DelHintCallback(SDL_HINT_JOYSTICK_HIDAPI_XBOX_360, callback, userdata);
}

static SDL_bool HIDAPI_DriverXbox360_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_XBOX_360,
                              SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_XBOX, SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT)));
}

static SDL_bool HIDAPI_DriverXbox360_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GameControllerType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    const int XB360W_IFACE_PROTOCOL = 129; /* Wireless */

    if (vendor_id == USB_VENDOR_ASTRO && product_id == USB_PRODUCT_ASTRO_C40_XBOX360) {
        /* This is the ASTRO C40 in Xbox 360 mode */
        return SDL_TRUE;
    }
    if (vendor_id == USB_VENDOR_NVIDIA) {
        /* This is the NVIDIA Shield controller which doesn't talk Xbox controller protocol */
        return SDL_FALSE;
    }
    if ((vendor_id == USB_VENDOR_MICROSOFT && (product_id == 0x0291 || product_id == 0x0719)) ||
        (type == SDL_CONTROLLER_TYPE_XBOX360 && interface_protocol == XB360W_IFACE_PROTOCOL)) {
        /* This is the wireless dongle, which talks a different protocol */
        return SDL_FALSE;
    }
    if (interface_number > 0) {
        /* This is the chatpad or other input interface, not the Xbox 360 interface */
        return SDL_FALSE;
    }
#if defined(__MACOSX__) && defined(SDL_JOYSTICK_MFI)
    if (SDL_IsJoystickSteamVirtualGamepad(vendor_id, product_id, version)) {
        /* GCController support doesn't work with the Steam Virtual Gamepad */
        return SDL_TRUE;
    } else {
        /* On macOS you can't write output reports to wired XBox controllers,
           so we'll just use the GCController support instead.
        */
        return SDL_FALSE;
    }
#else
    return (type == SDL_CONTROLLER_TYPE_XBOX360) ? SDL_TRUE : SDL_FALSE;
#endif
}

static SDL_bool SetSlotLED(SDL_hid_device *dev, Uint8 slot, SDL_bool on)
{
    const SDL_bool blink = SDL_FALSE;
    Uint8 mode = on ? ((blink ? 0x02 : 0x06) + slot) : 0;
    Uint8 led_packet[] = { 0x01, 0x03, 0x00 };

    led_packet[2] = mode;
    if (SDL_hid_write(dev, led_packet, sizeof(led_packet)) != sizeof(led_packet)) {
        return SDL_FALSE;
    }
    return SDL_TRUE;
}

static void UpdateSlotLED(SDL_DriverXbox360_Context *ctx)
{
    if (ctx->player_lights && ctx->player_lights >= 0) {
        SetSlotLED(ctx->device->dev, (ctx->player_index % 4), SDL_TRUE);
    } else {
        SetSlotLED(ctx->device->dev, 0, SDL_FALSE);
    }
}

static void SDLCALL SDL_PlayerLEDHintChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_DriverXbox360_Context *ctx = (SDL_DriverXbox360_Context *)userdata;
    SDL_bool player_lights = SDL_GetStringBoolean(hint, SDL_TRUE);

    if (player_lights != ctx->player_lights) {
        ctx->player_lights = player_lights;

        UpdateSlotLED(ctx);
    }
}

static SDL_bool HIDAPI_DriverXbox360_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverXbox360_Context *ctx;

    ctx = (SDL_DriverXbox360_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        SDL_OutOfMemory();
        return SDL_FALSE;
    }
    ctx->device = device;

    device->context = ctx;

    device->type = SDL_CONTROLLER_TYPE_XBOX360;

    if (SDL_IsJoystickSteamVirtualGamepad(device->vendor_id, device->product_id, device->version) &&
        device->product_string && SDL_strncmp(device->product_string, "GamePad-", 8) == 0) {
        int slot = 0;
        SDL_sscanf(device->product_string, "GamePad-%d", &slot);
        device->steam_virtual_gamepad_slot = (slot - 1);
    }

    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverXbox360_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverXbox360_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
    SDL_DriverXbox360_Context *ctx = (SDL_DriverXbox360_Context *)device->context;

    if (!ctx->joystick) {
        return;
    }

    ctx->player_index = player_index;

    UpdateSlotLED(ctx);
}

static SDL_bool HIDAPI_DriverXbox360_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverXbox360_Context *ctx = (SDL_DriverXbox360_Context *)device->context;

    SDL_AssertJoysticksLocked();

    ctx->joystick = joystick;
    SDL_zeroa(ctx->last_state);

    /* Initialize player index (needed for setting LEDs) */
    ctx->player_index = SDL_JoystickGetPlayerIndex(joystick);
    ctx->player_lights = SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_XBOX_360_PLAYER_LED, SDL_TRUE);
    UpdateSlotLED(ctx);

    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_XBOX_360_PLAYER_LED,
                        SDL_PlayerLEDHintChanged, ctx);

    /* Initialize the joystick capabilities */
    joystick->nbuttons = 15;
    joystick->naxes = SDL_CONTROLLER_AXIS_MAX;
    joystick->epowerlevel = SDL_JOYSTICK_POWER_WIRED;

    return SDL_TRUE;
}

static int HIDAPI_DriverXbox360_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    Uint8 rumble_packet[] = { 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

    rumble_packet[3] = (low_frequency_rumble >> 8);
    rumble_packet[4] = (high_frequency_rumble >> 8);

    if (SDL_HIDAPI_SendRumble(device, rumble_packet, sizeof(rumble_packet)) != sizeof(rumble_packet)) {
        return SDL_SetError("Couldn't send rumble packet");
    }
    return 0;
}

static int HIDAPI_DriverXbox360_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverXbox360_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    /* Doesn't have an RGB LED, so don't return SDL_JOYCAP_LED here */
    return SDL_JOYCAP_RUMBLE;
}

static int HIDAPI_DriverXbox360_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static int HIDAPI_DriverXbox360_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static int HIDAPI_DriverXbox360_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, SDL_bool enabled)
{
    return SDL_Unsupported();
}

static void HIDAPI_DriverXbox360_HandleStatePacket(SDL_Joystick *joystick, SDL_DriverXbox360_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;
#ifdef __MACOSX__
    const SDL_bool invert_y_axes = SDL_FALSE;
#else
    const SDL_bool invert_y_axes = SDL_TRUE;
#endif

    if (ctx->last_state[2] != data[2]) {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_UP, (data[2] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_DOWN, (data[2] & 0x02) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_LEFT, (data[2] & 0x04) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_RIGHT, (data[2] & 0x08) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_START, (data[2] & 0x10) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_BACK, (data[2] & 0x20) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSTICK, (data[2] & 0x40) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSTICK, (data[2] & 0x80) ? SDL_PRESSED : SDL_RELEASED);
    }

    if (ctx->last_state[3] != data[3]) {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSHOULDER, (data[3] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSHOULDER, (data[3] & 0x02) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_GUIDE, (data[3] & 0x04) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_A, (data[3] & 0x10) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_B, (data[3] & 0x20) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_X, (data[3] & 0x40) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_Y, (data[3] & 0x80) ? SDL_PRESSED : SDL_RELEASED);
    }

    axis = ((int)data[4] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_TRIGGERLEFT, axis);
    axis = ((int)data[5] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_TRIGGERRIGHT, axis);
    axis = SDL_SwapLE16(*(Sint16 *)(&data[6]));
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_LEFTX, axis);
    axis = SDL_SwapLE16(*(Sint16 *)(&data[8]));
    if (invert_y_axes) {
        axis = ~axis;
    }
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_LEFTY, axis);
    axis = SDL_SwapLE16(*(Sint16 *)(&data[10]));
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_RIGHTX, axis);
    axis = SDL_SwapLE16(*(Sint16 *)(&data[12]));
    if (invert_y_axes) {
        axis = ~axis;
    }
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_RIGHTY, axis);

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static SDL_bool HIDAPI_DriverXbox360_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverXbox360_Context *ctx = (SDL_DriverXbox360_Context *)device->context;
    SDL_Joystick *joystick = NULL;
    Uint8 data[USB_PACKET_LENGTH];
    int size = 0;

    if (device->num_joysticks > 0) {
        joystick = SDL_JoystickFromInstanceID(device->joysticks[0]);
    } else {
        return SDL_FALSE;
    }

    while ((size = SDL_hid_read_timeout(device->dev, data, sizeof(data), 0)) > 0) {
#ifdef DEBUG_XBOX_PROTOCOL
        HIDAPI_DumpPacket("Xbox 360 packet: size = %d", data, size);
#endif
        if (!joystick) {
            continue;
        }

        if (data[0] == 0x00) {
            HIDAPI_DriverXbox360_HandleStatePacket(joystick, ctx, data, size);
        }
    }

    if (size < 0) {
        /* Read error, device is disconnected */
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    return size >= 0;
}

static void HIDAPI_DriverXbox360_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverXbox360_Context *ctx = (SDL_DriverXbox360_Context *)device->context;

    SDL_DelHintCallback(SDL_HINT_JOYSTICK_HIDAPI_XBOX_360_PLAYER_LED,
                        SDL_PlayerLEDHintChanged, ctx);

    ctx->joystick = NULL;
}

static void HIDAPI_DriverXbox360_FreeDevice(SDL_HIDAPI_Device *device)
{
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverXbox360 = {
    SDL_HINT_JOYSTICK_HIDAPI_XBOX_360,
    SDL_TRUE,
    HIDAPI_DriverXbox360_RegisterHints,
    HIDAPI_DriverXbox360_UnregisterHints,
    HIDAPI_DriverXbox360_IsEnabled,
    HIDAPI_DriverXbox360_IsSupportedDevice,
    HIDAPI_DriverXbox360_InitDevice,
    HIDAPI_DriverXbox360_GetDevicePlayerIndex,
    HIDAPI_DriverXbox360_SetDevicePlayerIndex,
    HIDAPI_DriverXbox360_UpdateDevice,
    HIDAPI_DriverXbox360_OpenJoystick,
    HIDAPI_DriverXbox360_RumbleJoystick,
    HIDAPI_DriverXbox360_RumbleJoystickTriggers,
    HIDAPI_DriverXbox360_GetJoystickCapabilities,
    HIDAPI_DriverXbox360_SetJoystickLED,
    HIDAPI_DriverXbox360_SendJoystickEffect,
    HIDAPI_DriverXbox360_SetJoystickSensorsEnabled,
    HIDAPI_DriverXbox360_CloseJoystick,
    HIDAPI_DriverXbox360_FreeDevice,
};

#endif /* SDL_JOYSTICK_HIDAPI_XBOX360 */

#endif /* SDL_JOYSTICK_HIDAPI */

/* vi: set ts=4 sw=4 expandtab: */
