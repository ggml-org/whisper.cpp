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

#include "SDL_hints.h"
#include "SDL_events.h"
#include "SDL_timer.h"
#include "SDL_joystick.h"
#include "SDL_gamecontroller.h"
#include "../../SDL_hints_c.h"
#include "../SDL_sysjoystick.h"
#include "SDL_hidapijoystick_c.h"
#include "SDL_hidapi_rumble.h"

#ifdef SDL_JOYSTICK_HIDAPI_PS3

/* Define this if you want to log all packets from the controller */
/*#define DEBUG_PS3_PROTOCOL*/

#define LOAD16(A, B) (Sint16)((Uint16)(A) | (((Uint16)(B)) << 8))

typedef enum
{
    k_EPS3ReportIdState = 1,
    k_EPS3ReportIdEffects = 1,
} EPS3ReportId;

typedef struct
{
    SDL_HIDAPI_Device *device;
    SDL_Joystick *joystick;
    SDL_bool is_shanwan;
    SDL_bool has_analog_buttons;
    SDL_bool report_sensors;
    SDL_bool effects_updated;
    int player_index;
    Uint8 rumble_left;
    Uint8 rumble_right;
    Uint8 last_state[USB_PACKET_LENGTH];
} SDL_DriverPS3_Context;

static int HIDAPI_DriverPS3_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *effect, int size);

static void HIDAPI_DriverPS3_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_PS3, callback, userdata);
}

static void HIDAPI_DriverPS3_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_DelHintCallback(SDL_HINT_JOYSTICK_HIDAPI_PS3, callback, userdata);
}

static SDL_bool HIDAPI_DriverPS3_IsEnabled(void)
{
    SDL_bool default_value;

#if defined(__MACOSX__)
    /* This works well on macOS */
    default_value = SDL_TRUE;
#elif defined(__WINDOWS__)
    /* You can't initialize the controller with the stock Windows drivers
     * See https://github.com/ViGEm/DsHidMini as an alternative driver
     */
    default_value = SDL_FALSE;
#elif defined(__LINUX__)
    /* Linux drivers do a better job of managing the transition between
     * USB and Bluetooth. There are also some quirks in communicating
     * with PS3 controllers that have been implemented in SDL's hidapi
     * for libusb, but are not possible to support using hidraw if the
     * kernel doesn't already know about them.
     */
    default_value = SDL_FALSE;
#else
    /* Untested, default off */
    default_value = SDL_FALSE;
#endif

    if (default_value) {
        default_value = SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT);
    }
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_PS3, default_value);
}

static SDL_bool HIDAPI_DriverPS3_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GameControllerType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    if (vendor_id == USB_VENDOR_SONY && product_id == USB_PRODUCT_SONY_DS3) {
        return SDL_TRUE;
    }
    if (vendor_id == USB_VENDOR_SHANWAN && product_id == USB_PRODUCT_SHANWAN_DS3) {
        return SDL_TRUE;
    }
    return SDL_FALSE;
}

static int ReadFeatureReport(SDL_hid_device *dev, Uint8 report_id, Uint8 *report, size_t length)
{
    SDL_memset(report, 0, length);
    report[0] = report_id;
    return SDL_hid_get_feature_report(dev, report, length);
}

static int SendFeatureReport(SDL_hid_device *dev, Uint8 *report, size_t length)
{
    return SDL_hid_send_feature_report(dev, report, length);
}

static SDL_bool HIDAPI_DriverPS3_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS3_Context *ctx;
    SDL_bool is_shanwan = SDL_FALSE;

    if (device->vendor_id == USB_VENDOR_SONY &&
        SDL_strncasecmp(device->name, "ShanWan", 7) == 0) {
        is_shanwan = SDL_TRUE;
    }
    if (device->vendor_id == USB_VENDOR_SHANWAN ||
        device->vendor_id == USB_VENDOR_SHANWAN_ALT) {
        is_shanwan = SDL_TRUE;
    }

    ctx = (SDL_DriverPS3_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        SDL_OutOfMemory();
        return SDL_FALSE;
    }
    ctx->device = device;
    ctx->is_shanwan = is_shanwan;
    ctx->has_analog_buttons = SDL_TRUE;

    device->context = ctx;

    /* Set the controller into report mode over Bluetooth */
    {
        Uint8 data[] = { 0xf4, 0x42, 0x03, 0x00, 0x00 };

        SendFeatureReport(device->dev, data, sizeof(data));
    }

    /* Set the controller into report mode over USB */
    {
        Uint8 data[USB_PACKET_LENGTH];

        int size = ReadFeatureReport(device->dev, 0xf2, data, 17);
        if (size < 0) {
            SDL_LogDebug(SDL_LOG_CATEGORY_INPUT,
                         "HIDAPI_DriverPS3_InitDevice(): Couldn't read feature report 0xf2");
            return SDL_FALSE;
        }
#ifdef DEBUG_PS3_PROTOCOL
        HIDAPI_DumpPacket("PS3 0xF2 packet: size = %d", data, size);
#endif
        size = ReadFeatureReport(device->dev, 0xf5, data, 8);
        if (size < 0) {
            SDL_LogDebug(SDL_LOG_CATEGORY_INPUT,
                         "HIDAPI_DriverPS3_InitDevice(): Couldn't read feature report 0xf5");
            return SDL_FALSE;
        }
#ifdef DEBUG_PS3_PROTOCOL
        HIDAPI_DumpPacket("PS3 0xF5 packet: size = %d", data, size);
#endif
        if (!ctx->is_shanwan) {
            /* An output report could cause ShanWan controllers to rumble non-stop */
            SDL_hid_write(device->dev, data, 1);
        }
    }

    device->type = SDL_CONTROLLER_TYPE_PS3;
    HIDAPI_SetDeviceName(device, "PS3 Controller");

    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverPS3_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static int HIDAPI_DriverPS3_UpdateEffects(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    Uint8 effects[] = {
        0x01, 0xff, 0x00, 0xff, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00,
        0xff, 0x27, 0x10, 0x00, 0x32,
        0xff, 0x27, 0x10, 0x00, 0x32,
        0xff, 0x27, 0x10, 0x00, 0x32,
        0xff, 0x27, 0x10, 0x00, 0x32,
        0x00, 0x00, 0x00, 0x00, 0x00
    };

    effects[2] = ctx->rumble_right ? 1 : 0;
    effects[4] = ctx->rumble_left;

    effects[9] = (0x01 << (1 + (ctx->player_index % 4)));

    return HIDAPI_DriverPS3_SendJoystickEffect(device, ctx->joystick, effects, sizeof(effects));
}

static void HIDAPI_DriverPS3_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    if (!ctx) {
        return;
    }

    ctx->player_index = player_index;

    /* This will set the new LED state based on the new player index */
    HIDAPI_DriverPS3_UpdateEffects(device);
}

static SDL_bool HIDAPI_DriverPS3_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    SDL_AssertJoysticksLocked();

    ctx->joystick = joystick;
    ctx->effects_updated = SDL_FALSE;
    ctx->rumble_left = 0;
    ctx->rumble_right = 0;
    SDL_zeroa(ctx->last_state);

    /* Initialize player index (needed for setting LEDs) */
    ctx->player_index = SDL_JoystickGetPlayerIndex(joystick);

    /* Initialize the joystick capabilities */
    joystick->nbuttons = 15;
    joystick->naxes = 6;
    if (ctx->has_analog_buttons) {
        joystick->naxes += 10;
    }
    joystick->epowerlevel = SDL_JOYSTICK_POWER_WIRED;

    SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_ACCEL, 100.0f);

    return SDL_TRUE;
}

static int HIDAPI_DriverPS3_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    ctx->rumble_left = (low_frequency_rumble >> 8);
    ctx->rumble_right = (high_frequency_rumble >> 8);

    return HIDAPI_DriverPS3_UpdateEffects(device);
}

static int HIDAPI_DriverPS3_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverPS3_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    return SDL_JOYCAP_RUMBLE;
}

static int HIDAPI_DriverPS3_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static int HIDAPI_DriverPS3_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *effect, int size)
{
    Uint8 data[49];
    int report_size, offset;

    SDL_zeroa(data);

    data[0] = k_EPS3ReportIdEffects;
    report_size = sizeof(data);
    offset = 1;
    SDL_memcpy(&data[offset], effect, SDL_min((sizeof(data) - offset), (size_t)size));

    if (SDL_HIDAPI_SendRumble(device, data, report_size) != report_size) {
        return SDL_SetError("Couldn't send rumble packet");
    }
    return 0;
}

static int HIDAPI_DriverPS3_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, SDL_bool enabled)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    ctx->report_sensors = enabled;

    return 0;
}

static float HIDAPI_DriverPS3_ScaleAccel(Sint16 value)
{
    /* Accelerometer values are in big endian order */
    value = SDL_SwapBE16(value);
    return ((float)(value - 511) / 113.0f) * SDL_STANDARD_GRAVITY;
}

static void HIDAPI_DriverPS3_HandleMiniStatePacket(SDL_Joystick *joystick, SDL_DriverPS3_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;

    if (ctx->last_state[4] != data[4]) {
        SDL_bool dpad_up = SDL_FALSE;
        SDL_bool dpad_down = SDL_FALSE;
        SDL_bool dpad_left = SDL_FALSE;
        SDL_bool dpad_right = SDL_FALSE;

        switch (data[4] & 0x0f) {
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

        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_Y, (data[4] & 0x10) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_B, (data[4] & 0x20) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_A, (data[4] & 0x40) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_X, (data[4] & 0x80) ? SDL_PRESSED : SDL_RELEASED);
    }

    if (ctx->last_state[5] != data[5]) {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSHOULDER, (data[5] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSHOULDER, (data[5] & 0x02) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_TRIGGERLEFT, (data[5] & 0x04) ? SDL_JOYSTICK_AXIS_MAX : SDL_JOYSTICK_AXIS_MIN);
        SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_TRIGGERRIGHT, (data[5] & 0x08) ? SDL_JOYSTICK_AXIS_MAX : SDL_JOYSTICK_AXIS_MIN);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_BACK, (data[5] & 0x10) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_START, (data[5] & 0x20) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSTICK, (data[5] & 0x40) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSTICK, (data[5] & 0x80) ? SDL_PRESSED : SDL_RELEASED);
    }

    axis = ((int)data[2] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_LEFTX, axis);
    axis = ((int)data[3] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_LEFTY, axis);
    axis = ((int)data[0] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_RIGHTX, axis);
    axis = ((int)data[1] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_RIGHTY, axis);

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static void HIDAPI_DriverPS3_HandleStatePacket(SDL_Joystick *joystick, SDL_DriverPS3_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;

    if (ctx->last_state[2] != data[2]) {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_BACK, (data[2] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSTICK, (data[2] & 0x02) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSTICK, (data[2] & 0x04) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_START, (data[2] & 0x08) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_UP, (data[2] & 0x10) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_RIGHT, (data[2] & 0x20) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_DOWN, (data[2] & 0x40) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_LEFT, (data[2] & 0x80) ? SDL_PRESSED : SDL_RELEASED);
    }

    if (ctx->last_state[3] != data[3]) {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSHOULDER, (data[3] & 0x04) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSHOULDER, (data[3] & 0x08) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_Y, (data[3] & 0x10) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_B, (data[3] & 0x20) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_A, (data[3] & 0x40) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_X, (data[3] & 0x80) ? SDL_PRESSED : SDL_RELEASED);
    }

    if (ctx->last_state[4] != data[4]) {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_GUIDE, (data[4] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
    }

    axis = ((int)data[18] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_TRIGGERLEFT, axis);
    axis = ((int)data[19] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_TRIGGERRIGHT, axis);
    axis = ((int)data[6] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_LEFTX, axis);
    axis = ((int)data[7] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_LEFTY, axis);
    axis = ((int)data[8] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_RIGHTX, axis);
    axis = ((int)data[9] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_RIGHTY, axis);

    /* Buttons are mapped as axes in the order they appear in the button enumeration */
    if (ctx->has_analog_buttons) {
        static int button_axis_offsets[] = {
            24, /* SDL_CONTROLLER_BUTTON_A */
            23, /* SDL_CONTROLLER_BUTTON_B */
            25, /* SDL_CONTROLLER_BUTTON_X */
            22, /* SDL_CONTROLLER_BUTTON_Y */
            0,  /* SDL_CONTROLLER_BUTTON_BACK */
            0,  /* SDL_CONTROLLER_BUTTON_GUIDE */
            0,  /* SDL_CONTROLLER_BUTTON_START */
            0,  /* SDL_CONTROLLER_BUTTON_LEFTSTICK */
            0,  /* SDL_CONTROLLER_BUTTON_RIGHTSTICK */
            20, /* SDL_CONTROLLER_BUTTON_LEFTSHOULDER */
            21, /* SDL_CONTROLLER_BUTTON_RIGHTSHOULDER */
            14, /* SDL_CONTROLLER_BUTTON_DPAD_UP */
            16, /* SDL_CONTROLLER_BUTTON_DPAD_DOWN */
            17, /* SDL_CONTROLLER_BUTTON_DPAD_LEFT */
            15, /* SDL_CONTROLLER_BUTTON_DPAD_RIGHT */
        };
        int i, axis_index = 6;

        for (i = 0; i < SDL_arraysize(button_axis_offsets); ++i) {
            int offset = button_axis_offsets[i];
            if (!offset) {
                /* This button doesn't report as an axis */
                continue;
            }

            axis = ((int)data[offset] * 257) - 32768;
            SDL_PrivateJoystickAxis(joystick, axis_index, axis);
            ++axis_index;
        }
    }

    if (ctx->report_sensors) {
        float sensor_data[3];

        sensor_data[0] = HIDAPI_DriverPS3_ScaleAccel(LOAD16(data[41], data[42]));
        sensor_data[1] = -HIDAPI_DriverPS3_ScaleAccel(LOAD16(data[45], data[46]));
        sensor_data[2] = -HIDAPI_DriverPS3_ScaleAccel(LOAD16(data[43], data[44]));
        SDL_PrivateJoystickSensor(joystick, SDL_SENSOR_ACCEL, 0, sensor_data, SDL_arraysize(sensor_data));
    }

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static SDL_bool HIDAPI_DriverPS3_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;
    SDL_Joystick *joystick = NULL;
    Uint8 data[USB_PACKET_LENGTH];
    int size;

    if (device->num_joysticks > 0) {
        joystick = SDL_JoystickFromInstanceID(device->joysticks[0]);
    } else {
        return SDL_FALSE;
    }

    while ((size = SDL_hid_read_timeout(device->dev, data, sizeof(data), 0)) > 0) {
#ifdef DEBUG_PS3_PROTOCOL
        HIDAPI_DumpPacket("PS3 packet: size = %d", data, size);
#endif
        if (!joystick) {
            continue;
        }

        if (size == 7) {
            /* Seen on a ShanWan PS2 -> PS3 USB converter */
            HIDAPI_DriverPS3_HandleMiniStatePacket(joystick, ctx, data, size);

            /* Wait for the first report to set the LED state after the controller stops blinking */
            if (!ctx->effects_updated) {
                HIDAPI_DriverPS3_UpdateEffects(device);
                ctx->effects_updated = SDL_TRUE;
            }
            continue;
        }

        switch (data[0]) {
        case k_EPS3ReportIdState:
            if (data[1] == 0xFF) {
                /* Invalid data packet, ignore */
                break;
            }
            HIDAPI_DriverPS3_HandleStatePacket(joystick, ctx, data, size);

            /* Wait for the first report to set the LED state after the controller stops blinking */
            if (!ctx->effects_updated) {
                HIDAPI_DriverPS3_UpdateEffects(device);
                ctx->effects_updated = SDL_TRUE;
            }
            break;
        default:
#ifdef DEBUG_JOYSTICK
            SDL_Log("Unknown PS3 packet: 0x%.2x\n", data[0]);
#endif
            break;
        }
    }

    if (size < 0) {
        /* Read error, device is disconnected */
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    return size >= 0;
}

static void HIDAPI_DriverPS3_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    ctx->joystick = NULL;
}

static void HIDAPI_DriverPS3_FreeDevice(SDL_HIDAPI_Device *device)
{
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverPS3 = {
    SDL_HINT_JOYSTICK_HIDAPI_PS3,
    SDL_TRUE,
    HIDAPI_DriverPS3_RegisterHints,
    HIDAPI_DriverPS3_UnregisterHints,
    HIDAPI_DriverPS3_IsEnabled,
    HIDAPI_DriverPS3_IsSupportedDevice,
    HIDAPI_DriverPS3_InitDevice,
    HIDAPI_DriverPS3_GetDevicePlayerIndex,
    HIDAPI_DriverPS3_SetDevicePlayerIndex,
    HIDAPI_DriverPS3_UpdateDevice,
    HIDAPI_DriverPS3_OpenJoystick,
    HIDAPI_DriverPS3_RumbleJoystick,
    HIDAPI_DriverPS3_RumbleJoystickTriggers,
    HIDAPI_DriverPS3_GetJoystickCapabilities,
    HIDAPI_DriverPS3_SetJoystickLED,
    HIDAPI_DriverPS3_SendJoystickEffect,
    HIDAPI_DriverPS3_SetJoystickSensorsEnabled,
    HIDAPI_DriverPS3_CloseJoystick,
    HIDAPI_DriverPS3_FreeDevice,
};

static SDL_bool HIDAPI_DriverPS3ThirdParty_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_PS3,
                              SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI,
                                                 SDL_HIDAPI_DEFAULT));
}

static SDL_bool HIDAPI_DriverPS3ThirdParty_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GameControllerType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    Uint8 data[USB_PACKET_LENGTH];
    int size;

    if (vendor_id == USB_VENDOR_LOGITECH &&
        product_id == USB_PRODUCT_LOGITECH_CHILLSTREAM) {
        return SDL_TRUE;
    }

    if ((type == SDL_CONTROLLER_TYPE_PS3 && vendor_id != USB_VENDOR_SONY) ||
        HIDAPI_SupportsPlaystationDetection(vendor_id, product_id)) {
        if (device && device->dev) {
            size = ReadFeatureReport(device->dev, 0x03, data, sizeof(data));
            if (size == 8 && data[2] == 0x26) {
                /* Supported third party controller */
                return SDL_TRUE;
            } else {
                return SDL_FALSE;
            }
        } else {
            /* Might be supported by this driver, enumerate and find out */
            return SDL_TRUE;
        }
    }
    return SDL_FALSE;
}

static SDL_bool HIDAPI_DriverPS3ThirdParty_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS3_Context *ctx;

    ctx = (SDL_DriverPS3_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        SDL_OutOfMemory();
        return SDL_FALSE;
    }
    ctx->device = device;
    if (device->vendor_id == USB_VENDOR_SWITCH && device->product_id == USB_PRODUCT_SWITCH_RETROBIT_CONTROLLER) {
        ctx->has_analog_buttons = SDL_FALSE;
    } else {
        ctx->has_analog_buttons = SDL_TRUE;
    }

    device->context = ctx;

    device->type = SDL_CONTROLLER_TYPE_PS3;

    if (device->vendor_id == USB_VENDOR_LOGITECH &&
        device->product_id == USB_PRODUCT_LOGITECH_CHILLSTREAM) {
        HIDAPI_SetDeviceName(device, "Logitech ChillStream");
    }

    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverPS3ThirdParty_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverPS3ThirdParty_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
}

static SDL_bool HIDAPI_DriverPS3ThirdParty_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    SDL_AssertJoysticksLocked();

    ctx->joystick = joystick;
    SDL_zeroa(ctx->last_state);

    /* Initialize the joystick capabilities */
    joystick->nbuttons = 15;
    joystick->naxes = 6;
    if (ctx->has_analog_buttons) {
        joystick->naxes += 10;
    }

    if (device->vendor_id == USB_VENDOR_SWITCH && device->product_id == USB_PRODUCT_SWITCH_RETROBIT_CONTROLLER) {
        // This is a wireless controller using a USB dongle
        joystick->epowerlevel = SDL_JOYSTICK_POWER_UNKNOWN;
    } else {
        joystick->epowerlevel = SDL_JOYSTICK_POWER_WIRED;
    }

    return SDL_TRUE;
}

static int HIDAPI_DriverPS3ThirdParty_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    return SDL_Unsupported();
}

static int HIDAPI_DriverPS3ThirdParty_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverPS3ThirdParty_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    return 0;
}

static int HIDAPI_DriverPS3ThirdParty_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static int HIDAPI_DriverPS3ThirdParty_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *effect, int size)
{
    return SDL_Unsupported();
}

static int HIDAPI_DriverPS3ThirdParty_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, SDL_bool enabled)
{
    return SDL_Unsupported();
}

static void HIDAPI_DriverPS3ThirdParty_HandleStatePacket18(SDL_Joystick *joystick, SDL_DriverPS3_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;

    if (ctx->last_state[0] != data[0]) {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_X, (data[0] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_A, (data[0] & 0x02) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_B, (data[0] & 0x04) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_Y, (data[0] & 0x08) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSHOULDER, (data[0] & 0x10) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSHOULDER, (data[0] & 0x20) ? SDL_PRESSED : SDL_RELEASED);
    }

    if (ctx->last_state[1] != data[1]) {
        SDL_bool dpad_up = SDL_FALSE;
        SDL_bool dpad_down = SDL_FALSE;
        SDL_bool dpad_left = SDL_FALSE;
        SDL_bool dpad_right = SDL_FALSE;

        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_BACK, (data[1] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_START, (data[1] & 0x02) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSTICK, (data[1] & 0x04) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSTICK, (data[1] & 0x08) ? SDL_PRESSED : SDL_RELEASED);

        switch (data[1] >> 4) {
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

    axis = ((int)data[16] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_TRIGGERLEFT, axis);
    axis = ((int)data[17] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_TRIGGERRIGHT, axis);
    axis = ((int)data[2] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_LEFTX, axis);
    axis = ((int)data[3] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_LEFTY, axis);
    axis = ((int)data[4] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_RIGHTX, axis);
    axis = ((int)data[5] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_RIGHTY, axis);

    /* Buttons are mapped as axes in the order they appear in the button enumeration */
    if (ctx->has_analog_buttons) {
        static int button_axis_offsets[] = {
            12, /* SDL_GAMEPAD_BUTTON_A */
            11, /* SDL_GAMEPAD_BUTTON_B */
            13, /* SDL_GAMEPAD_BUTTON_X */
            10, /* SDL_GAMEPAD_BUTTON_Y */
            0,  /* SDL_GAMEPAD_BUTTON_BACK */
            0,  /* SDL_GAMEPAD_BUTTON_GUIDE */
            0,  /* SDL_GAMEPAD_BUTTON_START */
            0,  /* SDL_GAMEPAD_BUTTON_LEFT_STICK */
            0,  /* SDL_GAMEPAD_BUTTON_RIGHT_STICK */
            14, /* SDL_GAMEPAD_BUTTON_LEFT_SHOULDER */
            15, /* SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER */
            8,  /* SDL_GAMEPAD_BUTTON_DPAD_UP */
            9,  /* SDL_GAMEPAD_BUTTON_DPAD_DOWN */
            7,  /* SDL_GAMEPAD_BUTTON_DPAD_LEFT */
            6,  /* SDL_GAMEPAD_BUTTON_DPAD_RIGHT */
        };
        int i, axis_index = 6;

        for (i = 0; i < SDL_arraysize(button_axis_offsets); ++i) {
            int offset = button_axis_offsets[i];
            if (!offset) {
                /* This button doesn't report as an axis */
                continue;
            }

            axis = ((int)data[offset] * 257) - 32768;
            SDL_PrivateJoystickAxis(joystick, axis_index, axis);
            ++axis_index;
        }
    }

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static void HIDAPI_DriverPS3ThirdParty_HandleStatePacket19(SDL_Joystick *joystick, SDL_DriverPS3_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;

    if (ctx->last_state[0] != data[0]) {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_X, (data[0] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_A, (data[0] & 0x02) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_B, (data[0] & 0x04) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_Y, (data[0] & 0x08) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSHOULDER, (data[0] & 0x10) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSHOULDER, (data[0] & 0x20) ? SDL_PRESSED : SDL_RELEASED);
    }

    if (ctx->last_state[1] != data[1]) {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_BACK, (data[1] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_START, (data[1] & 0x02) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSTICK, (data[1] & 0x04) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSTICK, (data[1] & 0x08) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_GUIDE, (data[1] & 0x10) ? SDL_PRESSED : SDL_RELEASED);
    }

    if (ctx->device->vendor_id == USB_VENDOR_SAITEK && ctx->device->product_id == USB_PRODUCT_SAITEK_CYBORG_V3) {
        /* Cyborg V.3 Rumble Pad doesn't set the dpad bits as expected, so use the axes instead */
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_DOWN, data[10] ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_UP, data[9] ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_RIGHT, data[7] ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_LEFT, data[8] ? SDL_PRESSED : SDL_RELEASED);
    } else {
        if (ctx->last_state[2] != data[2]) {
            SDL_bool dpad_up = SDL_FALSE;
            SDL_bool dpad_down = SDL_FALSE;
            SDL_bool dpad_left = SDL_FALSE;
            SDL_bool dpad_right = SDL_FALSE;

            switch (data[2] & 0x0f) {
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
    }

    if (data[0] & 0x40) {
        axis = 32767;
    } else {
        axis = ((int)data[17] * 257) - 32768;
    }
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_TRIGGERLEFT, axis);
    if (data[0] & 0x80) {
        axis = 32767;
    } else {
        axis = ((int)data[18] * 257) - 32768;
    }
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_TRIGGERRIGHT, axis);
    axis = ((int)data[3] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_LEFTX, axis);
    axis = ((int)data[4] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_LEFTY, axis);
    axis = ((int)data[5] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_RIGHTX, axis);
    axis = ((int)data[6] * 257) - 32768;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_RIGHTY, axis);

    /* Buttons are mapped as axes in the order they appear in the button enumeration */
    if (ctx->has_analog_buttons) {
        static int button_axis_offsets[] = {
            13, /* SDL_CONTROLLER_BUTTON_A */
            12, /* SDL_CONTROLLER_BUTTON_B */
            14, /* SDL_CONTROLLER_BUTTON_X */
            11, /* SDL_CONTROLLER_BUTTON_Y */
            0,  /* SDL_CONTROLLER_BUTTON_BACK */
            0,  /* SDL_CONTROLLER_BUTTON_GUIDE */
            0,  /* SDL_CONTROLLER_BUTTON_START */
            0,  /* SDL_CONTROLLER_BUTTON_LEFTSTICK */
            0,  /* SDL_CONTROLLER_BUTTON_RIGHTSTICK */
            15, /* SDL_CONTROLLER_BUTTON_LEFTSHOULDER */
            16, /* SDL_CONTROLLER_BUTTON_RIGHTSHOULDER */
            9,  /* SDL_CONTROLLER_BUTTON_DPAD_UP */
            10, /* SDL_CONTROLLER_BUTTON_DPAD_DOWN */
            8,  /* SDL_CONTROLLER_BUTTON_DPAD_LEFT */
            7,  /* SDL_CONTROLLER_BUTTON_DPAD_RIGHT */
        };
        int i, axis_index = 6;

        for (i = 0; i < SDL_arraysize(button_axis_offsets); ++i) {
            int offset = button_axis_offsets[i];
            if (!offset) {
                /* This button doesn't report as an axis */
                continue;
            }

            axis = ((int)data[offset] * 257) - 32768;
            SDL_PrivateJoystickAxis(joystick, axis_index, axis);
            ++axis_index;
        }
    }

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static SDL_bool HIDAPI_DriverPS3ThirdParty_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;
    SDL_Joystick *joystick = NULL;
    Uint8 data[USB_PACKET_LENGTH];
    int size;

    if (device->num_joysticks > 0) {
        joystick = SDL_JoystickFromInstanceID(device->joysticks[0]);
    } else {
        return SDL_FALSE;
    }

    while ((size = SDL_hid_read_timeout(device->dev, data, sizeof(data), 0)) > 0) {
#ifdef DEBUG_PS3_PROTOCOL
        HIDAPI_DumpPacket("PS3 packet: size = %d", data, size);
#endif
        if (!joystick) {
            continue;
        }

        if (size >= 19) {
            HIDAPI_DriverPS3ThirdParty_HandleStatePacket19(joystick, ctx, data, size);
        } else if (size == 18) {
            /* This packet format was seen with the Logitech ChillStream */
            HIDAPI_DriverPS3ThirdParty_HandleStatePacket18(joystick, ctx, data, size);
        } else {
#ifdef DEBUG_JOYSTICK
            SDL_Log("Unknown PS3 packet, size %d\n", size);
#endif
        }
    }

    if (size < 0) {
        /* Read error, device is disconnected */
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    return size >= 0;
}

static void HIDAPI_DriverPS3ThirdParty_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    ctx->joystick = NULL;
}

static void HIDAPI_DriverPS3ThirdParty_FreeDevice(SDL_HIDAPI_Device *device)
{
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverPS3ThirdParty = {
    SDL_HINT_JOYSTICK_HIDAPI_PS3,
    SDL_TRUE,
    HIDAPI_DriverPS3_RegisterHints,
    HIDAPI_DriverPS3_UnregisterHints,
    HIDAPI_DriverPS3ThirdParty_IsEnabled,
    HIDAPI_DriverPS3ThirdParty_IsSupportedDevice,
    HIDAPI_DriverPS3ThirdParty_InitDevice,
    HIDAPI_DriverPS3ThirdParty_GetDevicePlayerIndex,
    HIDAPI_DriverPS3ThirdParty_SetDevicePlayerIndex,
    HIDAPI_DriverPS3ThirdParty_UpdateDevice,
    HIDAPI_DriverPS3ThirdParty_OpenJoystick,
    HIDAPI_DriverPS3ThirdParty_RumbleJoystick,
    HIDAPI_DriverPS3ThirdParty_RumbleJoystickTriggers,
    HIDAPI_DriverPS3ThirdParty_GetJoystickCapabilities,
    HIDAPI_DriverPS3ThirdParty_SetJoystickLED,
    HIDAPI_DriverPS3ThirdParty_SendJoystickEffect,
    HIDAPI_DriverPS3ThirdParty_SetJoystickSensorsEnabled,
    HIDAPI_DriverPS3ThirdParty_CloseJoystick,
    HIDAPI_DriverPS3ThirdParty_FreeDevice,
};

#endif /* SDL_JOYSTICK_HIDAPI_PS3 */

#endif /* SDL_JOYSTICK_HIDAPI */

/* vi: set ts=4 sw=4 expandtab: */
