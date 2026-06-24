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

#ifdef SDL_JOYSTICK_HIDAPI_XBOXONE

/* Define this if you want verbose logging of the init sequence */
/*#define DEBUG_JOYSTICK*/

/* Define this if you want to log all packets from the controller */
/*#define DEBUG_XBOX_PROTOCOL*/

#define CONTROLLER_NEGOTIATION_TIMEOUT_MS   300
#define CONTROLLER_PREPARE_INPUT_TIMEOUT_MS 50

/* Deadzone thresholds */
#define XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE  7849
#define XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE 8689
#define XINPUT_GAMEPAD_TRIGGER_THRESHOLD    -25058 /* Uint8 30 scaled to Sint16 full range */

/* Start controller */
static const Uint8 xboxone_init0[] = {
    0x05, 0x20, 0x03, 0x01, 0x00
};
/* Enable LED */
static const Uint8 xboxone_init1[] = {
    0x0A, 0x20, 0x00, 0x03, 0x00, 0x01, 0x14
};
/* Some PowerA controllers need to actually start the rumble motors */
static const Uint8 xboxone_powera_rumble_init[] = {
    0x09, 0x00, 0x00, 0x09, 0x00, 0x0F, 0x00, 0x00,
    0x1D, 0x1D, 0xFF, 0x00, 0x00
};
/* Setup rumble (not needed for Microsoft controllers, but it doesn't hurt) */
static const Uint8 xboxone_init2[] = {
    0x09, 0x00, 0x00, 0x09, 0x00, 0x0F, 0x00, 0x00,
    0x00, 0x00, 0xFF, 0x00, 0xEB
};
/* This controller passed security check */
static const Uint8 security_passed_packet[] = {
    0x06, 0x20, 0x00, 0x02, 0x01, 0x00
};

/*
 * This specifies the selection of init packets that a gamepad
 * will be sent on init *and* the order in which they will be
 * sent. The correct sequence number will be added when the
 * packet is going to be sent.
 */
typedef struct
{
    Uint16 vendor_id;
    Uint16 product_id;
    Uint16 exclude_vendor_id;
    Uint16 exclude_product_id;
    const Uint8 *data;
    int size;
    const Uint8 response[2];
} SDL_DriverXboxOne_InitPacket;

static const SDL_DriverXboxOne_InitPacket xboxone_init_packets[] = {
    { 0x0000, 0x0000, 0x0000, 0x0000, xboxone_init0, sizeof(xboxone_init0), { 0x00, 0x00 } },
    { 0x0000, 0x0000, 0x0000, 0x0000, xboxone_init1, sizeof(xboxone_init1), { 0x00, 0x00 } },
    /* The PDP Rock Candy and Victrix Gambit controllers don't start sending input until they get this packet */
    { 0x0e6f, 0x0000, 0x0000, 0x0000, security_passed_packet, sizeof(security_passed_packet), { 0x00, 0x00 } },
    { 0x24c6, 0x541a, 0x0000, 0x0000, xboxone_powera_rumble_init, sizeof(xboxone_powera_rumble_init), { 0x00, 0x00 } },
    { 0x24c6, 0x542a, 0x0000, 0x0000, xboxone_powera_rumble_init, sizeof(xboxone_powera_rumble_init), { 0x00, 0x00 } },
    { 0x24c6, 0x543a, 0x0000, 0x0000, xboxone_powera_rumble_init, sizeof(xboxone_powera_rumble_init), { 0x00, 0x00 } },
    { 0x0000, 0x0000, 0x0000, 0x0000, xboxone_init2, sizeof(xboxone_init2), { 0x00, 0x00 } },
};

typedef enum
{
    XBOX_ONE_INIT_STATE_START_NEGOTIATING,
    XBOX_ONE_INIT_STATE_NEGOTIATING,
    XBOX_ONE_INIT_STATE_PREPARE_INPUT,
    XBOX_ONE_INIT_STATE_COMPLETE,
} SDL_XboxOneInitState;

typedef enum
{
    XBOX_ONE_RUMBLE_STATE_IDLE,
    XBOX_ONE_RUMBLE_STATE_QUEUED,
    XBOX_ONE_RUMBLE_STATE_BUSY
} SDL_XboxOneRumbleState;

typedef struct
{
    SDL_HIDAPI_Device *device;
    Uint16 vendor_id;
    Uint16 product_id;
    SDL_bool bluetooth;
    SDL_XboxOneInitState init_state;
    int init_packet;
    Uint32 start_time;
    Uint8 sequence;
    Uint32 send_time;
    SDL_bool has_guide_packet;
    SDL_bool has_color_led;
    SDL_bool has_paddles;
    SDL_bool has_unmapped_state;
    SDL_bool has_trigger_rumble;
    SDL_bool has_share_button;
    Uint8 last_paddle_state;
    Uint8 low_frequency_rumble;
    Uint8 high_frequency_rumble;
    Uint8 left_trigger_rumble;
    Uint8 right_trigger_rumble;
    SDL_XboxOneRumbleState rumble_state;
    Uint32 rumble_time;
    SDL_bool rumble_pending;
    Uint8 last_state[USB_PACKET_LENGTH];
} SDL_DriverXboxOne_Context;

static SDL_bool ControllerHasColorLED(Uint16 vendor_id, Uint16 product_id)
{
    return vendor_id == USB_VENDOR_MICROSOFT && product_id == USB_PRODUCT_XBOX_ONE_ELITE_SERIES_2;
}

static SDL_bool ControllerHasPaddles(Uint16 vendor_id, Uint16 product_id)
{
    return SDL_IsJoystickXboxOneElite(vendor_id, product_id);
}

static SDL_bool ControllerHasTriggerRumble(Uint16 vendor_id, Uint16 product_id)
{
    /* All the Microsoft Xbox One controllers have trigger rumble */
    if (vendor_id == USB_VENDOR_MICROSOFT) {
        return SDL_TRUE;
    }

    /* It turns out other controllers a mixed bag as to whether they support
       trigger rumble or not, and when they do it's often a buzz rather than
       the vibration of the Microsoft trigger rumble, so for now just pretend
       that it is not available.
     */
    return SDL_FALSE;
}

static SDL_bool ControllerHasShareButton(Uint16 vendor_id, Uint16 product_id)
{
    return SDL_IsJoystickXboxSeriesX(vendor_id, product_id);
}

static int GetHomeLEDBrightness(const char *hint)
{
    const int MAX_VALUE = 50;
    int value = 20;

    if (hint && *hint) {
        if (SDL_strchr(hint, '.') != NULL) {
            value = (int)(MAX_VALUE * SDL_atof(hint));
        } else if (!SDL_GetStringBoolean(hint, SDL_TRUE)) {
            value = 0;
        }
    }
    return value;
}

static void SetHomeLED(SDL_DriverXboxOne_Context *ctx, int value)
{
    Uint8 led_packet[] = { 0x0A, 0x20, 0x00, 0x03, 0x00, 0x00, 0x00 };

    if (value > 0) {
        led_packet[5] = 0x01;
        led_packet[6] = (Uint8)value;
    }
    SDL_HIDAPI_SendRumble(ctx->device, led_packet, sizeof(led_packet));
}

static void SDLCALL SDL_HomeLEDHintChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_DriverXboxOne_Context *ctx = (SDL_DriverXboxOne_Context *)userdata;

    if (hint && *hint) {
        SetHomeLED(ctx, GetHomeLEDBrightness(hint));
    }
}

static void SetInitState(SDL_DriverXboxOne_Context *ctx, SDL_XboxOneInitState state)
{
#ifdef DEBUG_JOYSTICK
    SDL_Log("Setting init state %d\n", state);
#endif
    ctx->init_state = state;
}

static void SendAckIfNeeded(SDL_HIDAPI_Device *device, const Uint8 *data, int size)
{
#if defined(__WIN32__) || defined(__WINGDK__)
    /* The Windows driver is taking care of acks */
#else
    if ((data[1] & 0x30) == 0x30) {
        Uint8 ack_packet[] = { 0x01, 0x20, 0x00, 0x09, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

        ack_packet[2] = data[2];
        ack_packet[5] = data[0];
        ack_packet[7] = data[3];

        /* The initial ack needs 0x80 added to the response, for some reason */
        if (data[0] == 0x04 && data[1] == 0xF0) {
            ack_packet[11] = 0x80;
        }

#ifdef DEBUG_XBOX_PROTOCOL
        HIDAPI_DumpPacket("Xbox One sending ACK packet: size = %d", ack_packet, sizeof(ack_packet));
#endif
        if (SDL_HIDAPI_LockRumble() != 0 ||
            SDL_HIDAPI_SendRumbleAndUnlock(device, ack_packet, sizeof(ack_packet)) != sizeof(ack_packet)) {
            SDL_SetError("Couldn't send ack packet");
        }
    }
#endif /* defined(__WIN32__) || defined(__WINGDK__ */
}

#if 0
static SDL_bool SendSerialRequest(SDL_HIDAPI_Device *device, SDL_DriverXboxOne_Context *ctx)
{
    Uint8 serial_packet[] = { 0x1E, 0x30, 0x07, 0x01, 0x04 };

    ctx->send_time = SDL_GetTicks();

    /* Request the serial number
     * Sending this should be done only after the negotiation is complete.
     * It will cancel the announce packet if sent before that, and will be
     * ignored if sent during the negotiation.
     */
    if (SDL_HIDAPI_LockRumble() != 0 ||
        SDL_HIDAPI_SendRumbleAndUnlock(device, serial_packet, sizeof(serial_packet)) != sizeof(serial_packet)) {
        SDL_SetError("Couldn't send serial packet");
        return SDL_FALSE;
    }
    return SDL_TRUE;
}
#endif

static SDL_bool ControllerNeedsNegotiation(SDL_DriverXboxOne_Context *ctx)
{
    if (ctx->vendor_id == USB_VENDOR_PDP && ctx->product_id == 0x0246) {
        /* The PDP Rock Candy (PID 0x0246) doesn't send the announce packet on Linux for some reason */
        return SDL_TRUE;
    }
    return SDL_FALSE;
}

static SDL_bool SendControllerInit(SDL_HIDAPI_Device *device, SDL_DriverXboxOne_Context *ctx)
{
    Uint16 vendor_id = ctx->vendor_id;
    Uint16 product_id = ctx->product_id;
    Uint8 init_packet[USB_PACKET_LENGTH];

    for (; ctx->init_packet < SDL_arraysize(xboxone_init_packets); ++ctx->init_packet) {
        const SDL_DriverXboxOne_InitPacket *packet = &xboxone_init_packets[ctx->init_packet];

        if (packet->vendor_id && (vendor_id != packet->vendor_id)) {
            continue;
        }

        if (packet->product_id && (product_id != packet->product_id)) {
            continue;
        }

        if (packet->exclude_vendor_id && (vendor_id == packet->exclude_vendor_id)) {
            continue;
        }

        if (packet->exclude_product_id && (product_id == packet->exclude_product_id)) {
            continue;
        }

        SDL_memcpy(init_packet, packet->data, packet->size);
        if (init_packet[0] != 0x01) {
            init_packet[2] = ctx->sequence++;
        }
        if (init_packet[0] == 0x0A) {
            /* Get the initial brightness value */
            int brightness = GetHomeLEDBrightness(SDL_GetHint(SDL_HINT_JOYSTICK_HIDAPI_XBOX_ONE_HOME_LED));
            init_packet[5] = (brightness > 0) ? 0x01 : 0x00;
            init_packet[6] = (Uint8)brightness;
        }
#ifdef DEBUG_XBOX_PROTOCOL
        HIDAPI_DumpPacket("Xbox One sending INIT packet: size = %d", init_packet, packet->size);
#endif
        ctx->send_time = SDL_GetTicks();

        if (SDL_HIDAPI_LockRumble() != 0 ||
            SDL_HIDAPI_SendRumbleAndUnlock(device, init_packet, packet->size) != packet->size) {
            SDL_SetError("Couldn't write Xbox One initialization packet");
            return SDL_FALSE;
        }

        if (packet->response[0]) {
            return SDL_TRUE;
        }

        /* Wait to process the rumble packet */
        if (packet->data == xboxone_powera_rumble_init) {
            SDL_Delay(10);
        }
    }

    /* All done with the negotiation, prepare for input! */
    SetInitState(ctx, XBOX_ONE_INIT_STATE_PREPARE_INPUT);

    return SDL_TRUE;
}

static void HIDAPI_DriverXboxOne_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_XBOX, callback, userdata);
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_XBOX_ONE, callback, userdata);
}

static void HIDAPI_DriverXboxOne_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_DelHintCallback(SDL_HINT_JOYSTICK_HIDAPI_XBOX, callback, userdata);
    SDL_DelHintCallback(SDL_HINT_JOYSTICK_HIDAPI_XBOX_ONE, callback, userdata);
}

static SDL_bool HIDAPI_DriverXboxOne_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_XBOX_ONE,
                              SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_XBOX, SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT)));
}

static SDL_bool HIDAPI_DriverXboxOne_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GameControllerType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
#if defined(__MACOSX__) && defined(SDL_JOYSTICK_MFI)
    if (!SDL_IsJoystickBluetoothXboxOne(vendor_id, product_id)) {
        /* On macOS we get a shortened version of the real report and
           you can't write output reports for wired controllers, so
           we'll just use the GCController support instead.
        */
        return SDL_FALSE;
    }
#endif
    return (type == SDL_CONTROLLER_TYPE_XBOXONE) ? SDL_TRUE : SDL_FALSE;
}

static SDL_bool HIDAPI_DriverXboxOne_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverXboxOne_Context *ctx;

    ctx = (SDL_DriverXboxOne_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        SDL_OutOfMemory();
        return SDL_FALSE;
    }
    ctx->device = device;

    device->context = ctx;

    ctx->vendor_id = device->vendor_id;
    ctx->product_id = device->product_id;
    ctx->bluetooth = SDL_IsJoystickBluetoothXboxOne(device->vendor_id, device->product_id);
    ctx->start_time = SDL_GetTicks();
    ctx->sequence = 1;
    ctx->has_color_led = ControllerHasColorLED(ctx->vendor_id, ctx->product_id);
    ctx->has_paddles = ControllerHasPaddles(ctx->vendor_id, ctx->product_id);
    ctx->has_trigger_rumble = ControllerHasTriggerRumble(ctx->vendor_id, ctx->product_id);
    ctx->has_share_button = ControllerHasShareButton(ctx->vendor_id, ctx->product_id);

    /* Assume that the controller is correctly initialized when we start */
    if (ControllerNeedsNegotiation(ctx)) {
        ctx->init_state = XBOX_ONE_INIT_STATE_START_NEGOTIATING;
    } else {
        ctx->init_state = XBOX_ONE_INIT_STATE_COMPLETE;
    }

#ifdef DEBUG_JOYSTICK
    SDL_Log("Controller version: %d (0x%.4x)\n", device->version, device->version);
#endif

    device->type = SDL_CONTROLLER_TYPE_XBOXONE;

    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverXboxOne_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverXboxOne_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
}

static SDL_bool HIDAPI_DriverXboxOne_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverXboxOne_Context *ctx = (SDL_DriverXboxOne_Context *)device->context;

    SDL_AssertJoysticksLocked();

    ctx->low_frequency_rumble = 0;
    ctx->high_frequency_rumble = 0;
    ctx->left_trigger_rumble = 0;
    ctx->right_trigger_rumble = 0;
    ctx->rumble_state = XBOX_ONE_RUMBLE_STATE_IDLE;
    ctx->rumble_time = 0;
    ctx->rumble_pending = SDL_FALSE;
    SDL_zeroa(ctx->last_state);

    /* Initialize the joystick capabilities */
    joystick->nbuttons = 15;
    if (ctx->has_share_button) {
        joystick->nbuttons += 1;
    }
    if (ctx->has_paddles) {
        joystick->nbuttons += 4;
    }
    joystick->naxes = SDL_CONTROLLER_AXIS_MAX;

    if (!ctx->bluetooth) {
        joystick->epowerlevel = SDL_JOYSTICK_POWER_WIRED;
    }

    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_XBOX_ONE_HOME_LED,
                        SDL_HomeLEDHintChanged, ctx);
    return SDL_TRUE;
}

static void HIDAPI_DriverXboxOne_RumbleSent(void *userdata)
{
    SDL_DriverXboxOne_Context *ctx = (SDL_DriverXboxOne_Context *)userdata;
    ctx->rumble_time = SDL_GetTicks();
}

static int HIDAPI_DriverXboxOne_UpdateRumble(SDL_HIDAPI_Device *device)
{
    SDL_DriverXboxOne_Context *ctx = (SDL_DriverXboxOne_Context *)device->context;

    if (ctx->rumble_state == XBOX_ONE_RUMBLE_STATE_QUEUED) {
        if (ctx->rumble_time) {
            ctx->rumble_state = XBOX_ONE_RUMBLE_STATE_BUSY;
        }
    }

    if (ctx->rumble_state == XBOX_ONE_RUMBLE_STATE_BUSY) {
        const Uint32 RUMBLE_BUSY_TIME_MS = ctx->bluetooth ? 50 : 10;
        if (SDL_TICKS_PASSED(SDL_GetTicks(), ctx->rumble_time + RUMBLE_BUSY_TIME_MS)) {
            ctx->rumble_time = 0;
            ctx->rumble_state = XBOX_ONE_RUMBLE_STATE_IDLE;
        }
    }

    if (!ctx->rumble_pending) {
        return 0;
    }

    if (ctx->rumble_state != XBOX_ONE_RUMBLE_STATE_IDLE) {
        return 0;
    }

    /* We're no longer pending, even if we fail to send the rumble below */
    ctx->rumble_pending = SDL_FALSE;

    if (SDL_HIDAPI_LockRumble() != 0) {
        return -1;
    }

    if (ctx->bluetooth) {
        Uint8 rumble_packet[] = { 0x03, 0x0F, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00, 0xEB };

        rumble_packet[2] = ctx->left_trigger_rumble;
        rumble_packet[3] = ctx->right_trigger_rumble;
        rumble_packet[4] = ctx->low_frequency_rumble;
        rumble_packet[5] = ctx->high_frequency_rumble;

        if (SDL_HIDAPI_SendRumbleWithCallbackAndUnlock(device, rumble_packet, sizeof(rumble_packet), HIDAPI_DriverXboxOne_RumbleSent, ctx) != sizeof(rumble_packet)) {
            return SDL_SetError("Couldn't send rumble packet");
        }
    } else {
        Uint8 rumble_packet[] = { 0x09, 0x00, 0x00, 0x09, 0x00, 0x0F, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00, 0xEB };

        rumble_packet[6] = ctx->left_trigger_rumble;
        rumble_packet[7] = ctx->right_trigger_rumble;
        rumble_packet[8] = ctx->low_frequency_rumble;
        rumble_packet[9] = ctx->high_frequency_rumble;

        if (SDL_HIDAPI_SendRumbleWithCallbackAndUnlock(device, rumble_packet, sizeof(rumble_packet), HIDAPI_DriverXboxOne_RumbleSent, ctx) != sizeof(rumble_packet)) {
            return SDL_SetError("Couldn't send rumble packet");
        }
    }

    ctx->rumble_state = XBOX_ONE_RUMBLE_STATE_QUEUED;

    return 0;
}

static int HIDAPI_DriverXboxOne_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    SDL_DriverXboxOne_Context *ctx = (SDL_DriverXboxOne_Context *)device->context;

    /* Magnitude is 1..100 so scale the 16-bit input here */
    ctx->low_frequency_rumble = low_frequency_rumble / 655;
    ctx->high_frequency_rumble = high_frequency_rumble / 655;
    ctx->rumble_pending = SDL_TRUE;

    return HIDAPI_DriverXboxOne_UpdateRumble(device);
}

static int HIDAPI_DriverXboxOne_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    SDL_DriverXboxOne_Context *ctx = (SDL_DriverXboxOne_Context *)device->context;

    if (!ctx->has_trigger_rumble) {
        return SDL_Unsupported();
    }

    /* Magnitude is 1..100 so scale the 16-bit input here */
    ctx->left_trigger_rumble = left_rumble / 655;
    ctx->right_trigger_rumble = right_rumble / 655;
    ctx->rumble_pending = SDL_TRUE;

    return HIDAPI_DriverXboxOne_UpdateRumble(device);
}

static Uint32 HIDAPI_DriverXboxOne_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverXboxOne_Context *ctx = (SDL_DriverXboxOne_Context *)device->context;
    Uint32 result = 0;

    result |= SDL_JOYCAP_RUMBLE;
    if (ctx->has_trigger_rumble) {
        result |= SDL_JOYCAP_RUMBLE_TRIGGERS;
    }

    if (ctx->has_color_led) {
        result |= SDL_JOYCAP_LED;
    }

    return result;
}

static int HIDAPI_DriverXboxOne_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    SDL_DriverXboxOne_Context *ctx = (SDL_DriverXboxOne_Context *)device->context;

    if (ctx->has_color_led) {
        Uint8 led_packet[] = { 0x0E, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00 };

        led_packet[5] = 0x00; /* Whiteness? Sets white intensity when RGB is 0, seems additive */
        led_packet[6] = red;
        led_packet[7] = green;
        led_packet[8] = blue;

        if (SDL_HIDAPI_SendRumble(device, led_packet, sizeof(led_packet)) != sizeof(led_packet)) {
            return SDL_SetError("Couldn't send LED packet");
        }
        return 0;
    } else {
        return SDL_Unsupported();
    }
}

static int HIDAPI_DriverXboxOne_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static int HIDAPI_DriverXboxOne_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, SDL_bool enabled)
{
    return SDL_Unsupported();
}

/*
 * The Xbox One Elite controller with 5.13+ firmware sends the unmapped state in a separate packet.
 * We can use this to send the paddle state when they aren't mapped
 */
static void HIDAPI_DriverXboxOne_HandleUnmappedStatePacket(SDL_Joystick *joystick, SDL_DriverXboxOne_Context *ctx, Uint8 *data, int size)
{
    Uint8 profile;
    int paddle_index;
    int button1_bit;
    int button2_bit;
    int button3_bit;
    int button4_bit;
    SDL_bool paddles_mapped;

    if (size == 21) {
        /* XBox One Elite Series 2 */
        paddle_index = 18;
        button1_bit = 0x01;
        button2_bit = 0x02;
        button3_bit = 0x04;
        button4_bit = 0x08;
        profile = data[19];

        if (profile == 0) {
            paddles_mapped = SDL_FALSE;
        } else if (SDL_memcmp(&data[4], &ctx->last_state[4], 14) == 0) {
            /* We're using a profile, but paddles aren't mapped */
            paddles_mapped = SDL_FALSE;
        } else {
            /* Something is mapped, we can't use the paddles */
            paddles_mapped = SDL_TRUE;
        }

    } else {
        /* Unknown format */
        return;
    }
#ifdef DEBUG_XBOX_PROTOCOL
    SDL_Log(">>> Paddles: %d,%d,%d,%d mapped = %s\n",
            (data[paddle_index] & button1_bit) ? 1 : 0,
            (data[paddle_index] & button2_bit) ? 1 : 0,
            (data[paddle_index] & button3_bit) ? 1 : 0,
            (data[paddle_index] & button4_bit) ? 1 : 0,
            paddles_mapped ? "TRUE" : "FALSE");
#endif

    if (paddles_mapped) {
        /* Respect that the paddles are being used for other controls and don't pass them on to the app */
        data[paddle_index] = 0;
    }

    if (ctx->last_paddle_state != data[paddle_index]) {
        int nButton = SDL_CONTROLLER_BUTTON_MISC1 + ctx->has_share_button; /* Next available button */
        SDL_PrivateJoystickButton(joystick, nButton++, (data[paddle_index] & button1_bit) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, nButton++, (data[paddle_index] & button2_bit) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, nButton++, (data[paddle_index] & button3_bit) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, nButton++, (data[paddle_index] & button4_bit) ? SDL_PRESSED : SDL_RELEASED);
        ctx->last_paddle_state = data[paddle_index];
    }
    ctx->has_unmapped_state = SDL_TRUE;
}

static void HIDAPI_DriverXboxOne_HandleStatePacket(SDL_Joystick *joystick, SDL_DriverXboxOne_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;

    /* Some controllers have larger packets over NDIS, but the real size is in data[3] */
    size = SDL_min(4 + data[3], size);

    /* Enable paddles on the Xbox Elite controller when connected over USB */
    if (ctx->has_paddles && !ctx->has_unmapped_state && size == 50) {
        Uint8 packet[] = { 0x4d, 0x00, 0x00, 0x02, 0x07, 0x00 };

        SDL_HIDAPI_SendRumble(ctx->device, packet, sizeof(packet));
    }

    if (ctx->last_state[4] != data[4]) {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_START, (data[4] & 0x04) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_BACK, (data[4] & 0x08) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_A, (data[4] & 0x10) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_B, (data[4] & 0x20) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_X, (data[4] & 0x40) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_Y, (data[4] & 0x80) ? SDL_PRESSED : SDL_RELEASED);
    }

    if (ctx->last_state[5] != data[5]) {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_UP, (data[5] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_DOWN, (data[5] & 0x02) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_LEFT, (data[5] & 0x04) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_DPAD_RIGHT, (data[5] & 0x08) ? SDL_PRESSED : SDL_RELEASED);
        if (ctx->vendor_id == USB_VENDOR_RAZER && ctx->product_id == USB_PRODUCT_RAZER_ATROX) {
            /* The Razer Atrox has the right and left shoulder bits reversed */
            SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSHOULDER, (data[5] & 0x20) ? SDL_PRESSED : SDL_RELEASED);
            SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSHOULDER, (data[5] & 0x10) ? SDL_PRESSED : SDL_RELEASED);
        } else {
            SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSHOULDER, (data[5] & 0x10) ? SDL_PRESSED : SDL_RELEASED);
            SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSHOULDER, (data[5] & 0x20) ? SDL_PRESSED : SDL_RELEASED);
        }
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSTICK, (data[5] & 0x40) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSTICK, (data[5] & 0x80) ? SDL_PRESSED : SDL_RELEASED);
    }

    if (ctx->has_share_button) {
        /* Xbox Series X firmware version 5.0, report is 36 bytes, share button is in byte 18
         * Xbox Series X firmware version 5.1, report is 44 bytes, share button is in byte 18
         * Xbox Series X firmware version 5.5, report is 48 bytes, share button is in byte 22
         * Victrix Gambit Tournament Controller, report is 50 bytes, share button is in byte 32
         * ThrustMaster eSwap PRO Controller Xbox, report is 64 bytes, share button is in byte 46
         */
        if (size < 48) {
            if (ctx->last_state[18] != data[18]) {
                SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_MISC1, (data[18] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
            }
        } else if (size == 48) {
            if (ctx->last_state[22] != data[22]) {
                SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_MISC1, (data[22] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
            }
        } else if (size == 50) {
            if (ctx->last_state[32] != data[32]) {
                SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_MISC1, (data[32] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
            }
        } else if (size == 64) {
            if (ctx->last_state[46] != data[46]) {
                SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_MISC1, (data[46] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
            }
        }
    }

    /* Xbox One S report is 18 bytes
       Xbox One Elite Series 1 report is 33 bytes, paddles in data[32], mode in data[32] & 0x10, both modes have mapped paddles by default
        Paddle bits:
            P3: 0x01 (A)    P1: 0x02 (B)
            P4: 0x04 (X)    P2: 0x08 (Y)
       Xbox One Elite Series 2 4.x firmware report is 38 bytes, paddles in data[18], mode in data[19], mode 0 has no mapped paddles by default
        Paddle bits:
            P3: 0x04 (A)    P1: 0x01 (B)
            P4: 0x08 (X)    P2: 0x02 (Y)
       Xbox One Elite Series 2 5.x firmware report is 50 bytes, paddles in data[22], mode in data[23], mode 0 has no mapped paddles by default
        Paddle bits:
            P3: 0x04 (A)    P1: 0x01 (B)
            P4: 0x08 (X)    P2: 0x02 (Y)
    */
    if (ctx->has_paddles && !ctx->has_unmapped_state && (size == 33 || size == 38 || size == 50)) {
        int paddle_index;
        int button1_bit;
        int button2_bit;
        int button3_bit;
        int button4_bit;
        SDL_bool paddles_mapped;

        if (size == 33) {
            /* XBox One Elite Series 1 */
            paddle_index = 32;
            button1_bit = 0x02;
            button2_bit = 0x08;
            button3_bit = 0x01;
            button4_bit = 0x04;

            /* The mapped controller state is at offset 4, the raw state is at offset 18, compare them to see if the paddles are mapped */
            paddles_mapped = (SDL_memcmp(&data[4], &data[18], 2) != 0);

        } else if (size == 38) {
            /* XBox One Elite Series 2 */
            paddle_index = 18;
            button1_bit = 0x01;
            button2_bit = 0x02;
            button3_bit = 0x04;
            button4_bit = 0x08;
            paddles_mapped = (data[19] != 0);

        } else /* if (size == 50) */ {
            /* XBox One Elite Series 2 */
            paddle_index = 22;
            button1_bit = 0x01;
            button2_bit = 0x02;
            button3_bit = 0x04;
            button4_bit = 0x08;
            paddles_mapped = (data[23] != 0);
        }
#ifdef DEBUG_XBOX_PROTOCOL
        SDL_Log(">>> Paddles: %d,%d,%d,%d mapped = %s\n",
                (data[paddle_index] & button1_bit) ? 1 : 0,
                (data[paddle_index] & button2_bit) ? 1 : 0,
                (data[paddle_index] & button3_bit) ? 1 : 0,
                (data[paddle_index] & button4_bit) ? 1 : 0,
                paddles_mapped ? "TRUE" : "FALSE");
#endif

        if (paddles_mapped) {
            /* Respect that the paddles are being used for other controls and don't pass them on to the app */
            data[paddle_index] = 0;
        }

        if (ctx->last_paddle_state != data[paddle_index]) {
            int nButton = SDL_CONTROLLER_BUTTON_MISC1 + ctx->has_share_button; /* Next available button */
            SDL_PrivateJoystickButton(joystick, nButton++, (data[paddle_index] & button1_bit) ? SDL_PRESSED : SDL_RELEASED);
            SDL_PrivateJoystickButton(joystick, nButton++, (data[paddle_index] & button2_bit) ? SDL_PRESSED : SDL_RELEASED);
            SDL_PrivateJoystickButton(joystick, nButton++, (data[paddle_index] & button3_bit) ? SDL_PRESSED : SDL_RELEASED);
            SDL_PrivateJoystickButton(joystick, nButton++, (data[paddle_index] & button4_bit) ? SDL_PRESSED : SDL_RELEASED);
            ctx->last_paddle_state = data[paddle_index];
        }
    }

    axis = ((int)SDL_SwapLE16(*(Sint16 *)(&data[6])) * 64) - 32768;
    if (axis == 32704) {
        axis = 32767;
    }
    if (axis == -32768 && size == 30 && (data[22] & 0x80)) {
        axis = 32767;
    }
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_TRIGGERLEFT, axis);

    axis = ((int)SDL_SwapLE16(*(Sint16 *)(&data[8])) * 64) - 32768;
    if (axis == -32768 && size == 30 && (data[22] & 0x40)) {
        axis = 32767;
    }
    if (axis == 32704) {
        axis = 32767;
    }
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_TRIGGERRIGHT, axis);

    axis = SDL_SwapLE16(*(Sint16 *)(&data[10]));
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_LEFTX, axis);
    axis = SDL_SwapLE16(*(Sint16 *)(&data[12]));
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_LEFTY, ~axis);
    axis = SDL_SwapLE16(*(Sint16 *)(&data[14]));
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_RIGHTX, axis);
    axis = SDL_SwapLE16(*(Sint16 *)(&data[16]));
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_RIGHTY, ~axis);

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));

    /* We don't have the unmapped state for this packet */
    ctx->has_unmapped_state = SDL_FALSE;
}

static void HIDAPI_DriverXboxOne_HandleStatusPacket(SDL_Joystick *joystick, SDL_DriverXboxOne_Context *ctx, Uint8 *data, int size)
{
    if (ctx->init_state < XBOX_ONE_INIT_STATE_COMPLETE) {
        SetInitState(ctx, XBOX_ONE_INIT_STATE_COMPLETE);
    }
}

static void HIDAPI_DriverXboxOne_HandleModePacket(SDL_Joystick *joystick, SDL_DriverXboxOne_Context *ctx, const Uint8 *data, int size)
{
    SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_GUIDE, (data[4] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
}

/*
 * Xbox One S with firmware 3.1.1221 uses a 16 byte packet and the GUIDE button in a separate packet
 */
static void HIDAPI_DriverXboxOneBluetooth_HandleButtons16(SDL_Joystick *joystick, SDL_DriverXboxOne_Context *ctx, const Uint8 *data, int size)
{
    if (ctx->last_state[14] != data[14]) {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_A, (data[14] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_B, (data[14] & 0x02) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_X, (data[14] & 0x04) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_Y, (data[14] & 0x08) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSHOULDER, (data[14] & 0x10) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSHOULDER, (data[14] & 0x20) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_BACK, (data[14] & 0x40) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_START, (data[14] & 0x80) ? SDL_PRESSED : SDL_RELEASED);
    }

    if (ctx->last_state[15] != data[15]) {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSTICK, (data[15] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSTICK, (data[15] & 0x02) ? SDL_PRESSED : SDL_RELEASED);
    }
}

/*
 * Xbox One S with firmware 4.8.1923 uses a 17 byte packet with BACK button in byte 16 and the GUIDE button in a separate packet (on Windows), or in byte 15 (on Linux)
 * Xbox One S with firmware 5.x uses a 17 byte packet with BACK and GUIDE buttons in byte 15
 * Xbox One Elite Series 2 with firmware 4.7.1872 uses a 55 byte packet with BACK button in byte 16, paddles starting at byte 33, and the GUIDE button in a separate packet
 * Xbox One Elite Series 2 with firmware 4.8.1908 uses a 33 byte packet with BACK button in byte 16, paddles starting at byte 17, and the GUIDE button in a separate packet
 * Xbox One Elite Series 2 with firmware 5.11.3112 uses a 19 byte packet with BACK and GUIDE buttons in byte 15
 * Xbox Series X with firmware 5.5.2641 uses a 17 byte packet with BACK and GUIDE buttons in byte 15, and SHARE button in byte 17
 */
static void HIDAPI_DriverXboxOneBluetooth_HandleButtons(SDL_Joystick *joystick, SDL_DriverXboxOne_Context *ctx, Uint8 *data, int size)
{
    if (ctx->last_state[14] != data[14]) {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_A, (data[14] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_B, (data[14] & 0x02) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_X, (data[14] & 0x08) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_Y, (data[14] & 0x10) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSHOULDER, (data[14] & 0x40) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSHOULDER, (data[14] & 0x80) ? SDL_PRESSED : SDL_RELEASED);
    }

    if (ctx->last_state[15] != data[15]) {
        if (!ctx->has_guide_packet) {
            SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_GUIDE, (data[15] & 0x10) ? SDL_PRESSED : SDL_RELEASED);
        }
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_START, (data[15] & 0x08) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_LEFTSTICK, (data[15] & 0x20) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_RIGHTSTICK, (data[15] & 0x40) ? SDL_PRESSED : SDL_RELEASED);
    }

    if (ctx->has_share_button) {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_BACK, (data[15] & 0x04) ? SDL_PRESSED : SDL_RELEASED);
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_MISC1, (data[16] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
    } else {
        SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_BACK, ((data[15] & 0x04) || (data[16] & 0x01)) ? SDL_PRESSED : SDL_RELEASED);
    }

    /*
        Paddle bits:
            P3: 0x04 (A)    P1: 0x01 (B)
            P4: 0x08 (X)    P2: 0x02 (Y)
    */
    if (ctx->has_paddles && (size == 20 || size == 39 || size == 55)) {
        int paddle_index;
        int button1_bit;
        int button2_bit;
        int button3_bit;
        int button4_bit;
        SDL_bool paddles_mapped;

        if (size == 55) {
            /* Initial firmware for the Xbox Elite Series 2 controller */
            paddle_index = 33;
            button1_bit = 0x01;
            button2_bit = 0x02;
            button3_bit = 0x04;
            button4_bit = 0x08;
            paddles_mapped = (data[35] != 0);
        } else if (size == 39) {
            /* Updated firmware for the Xbox Elite Series 2 controller */
            paddle_index = 17;
            button1_bit = 0x01;
            button2_bit = 0x02;
            button3_bit = 0x04;
            button4_bit = 0x08;
            paddles_mapped = (data[19] != 0);
        } else /* if (size == 20) */ {
            /* Updated firmware for the Xbox Elite Series 2 controller (5.13+) */
            paddle_index = 19;
            button1_bit = 0x01;
            button2_bit = 0x02;
            button3_bit = 0x04;
            button4_bit = 0x08;
            paddles_mapped = (data[17] != 0);
        }

#ifdef DEBUG_XBOX_PROTOCOL
        SDL_Log(">>> Paddles: %d,%d,%d,%d mapped = %s\n",
                (data[paddle_index] & button1_bit) ? 1 : 0,
                (data[paddle_index] & button2_bit) ? 1 : 0,
                (data[paddle_index] & button3_bit) ? 1 : 0,
                (data[paddle_index] & button4_bit) ? 1 : 0,
                paddles_mapped ? "TRUE" : "FALSE");
#endif

        if (paddles_mapped) {
            /* Respect that the paddles are being used for other controls and don't pass them on to the app */
            data[paddle_index] = 0;
        }

        if (ctx->last_paddle_state != data[paddle_index]) {
            int nButton = SDL_CONTROLLER_BUTTON_MISC1; /* Next available button */
            SDL_PrivateJoystickButton(joystick, nButton++, (data[paddle_index] & button1_bit) ? SDL_PRESSED : SDL_RELEASED);
            SDL_PrivateJoystickButton(joystick, nButton++, (data[paddle_index] & button2_bit) ? SDL_PRESSED : SDL_RELEASED);
            SDL_PrivateJoystickButton(joystick, nButton++, (data[paddle_index] & button3_bit) ? SDL_PRESSED : SDL_RELEASED);
            SDL_PrivateJoystickButton(joystick, nButton++, (data[paddle_index] & button4_bit) ? SDL_PRESSED : SDL_RELEASED);
            ctx->last_paddle_state = data[paddle_index];
        }
    }
}

static void HIDAPI_DriverXboxOneBluetooth_HandleStatePacket(SDL_Joystick *joystick, SDL_DriverXboxOne_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;

    if (size == 16) {
        /* Original Xbox One S, with separate report for guide button */
        HIDAPI_DriverXboxOneBluetooth_HandleButtons16(joystick, ctx, data, size);
    } else if (size > 16) {
        HIDAPI_DriverXboxOneBluetooth_HandleButtons(joystick, ctx, data, size);
    } else {
#ifdef DEBUG_XBOX_PROTOCOL
        SDL_Log("Unknown Bluetooth state packet format\n");
#endif
        return;
    }

    if (ctx->last_state[13] != data[13]) {
        SDL_bool dpad_up = SDL_FALSE;
        SDL_bool dpad_down = SDL_FALSE;
        SDL_bool dpad_left = SDL_FALSE;
        SDL_bool dpad_right = SDL_FALSE;

        switch (data[13]) {
        case 1:
            dpad_up = SDL_TRUE;
            break;
        case 2:
            dpad_up = SDL_TRUE;
            dpad_right = SDL_TRUE;
            break;
        case 3:
            dpad_right = SDL_TRUE;
            break;
        case 4:
            dpad_right = SDL_TRUE;
            dpad_down = SDL_TRUE;
            break;
        case 5:
            dpad_down = SDL_TRUE;
            break;
        case 6:
            dpad_left = SDL_TRUE;
            dpad_down = SDL_TRUE;
            break;
        case 7:
            dpad_left = SDL_TRUE;
            break;
        case 8:
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

    axis = ((int)SDL_SwapLE16(*(Sint16 *)(&data[9])) * 64) - 32768;
    if (axis == 32704) {
        axis = 32767;
    }
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_TRIGGERLEFT, axis);

    axis = ((int)SDL_SwapLE16(*(Sint16 *)(&data[11])) * 64) - 32768;
    if (axis == 32704) {
        axis = 32767;
    }
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_TRIGGERRIGHT, axis);

    axis = (int)SDL_SwapLE16(*(Uint16 *)(&data[1])) - 0x8000;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_LEFTX, axis);
    axis = (int)SDL_SwapLE16(*(Uint16 *)(&data[3])) - 0x8000;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_LEFTY, axis);
    axis = (int)SDL_SwapLE16(*(Uint16 *)(&data[5])) - 0x8000;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_RIGHTX, axis);
    axis = (int)SDL_SwapLE16(*(Uint16 *)(&data[7])) - 0x8000;
    SDL_PrivateJoystickAxis(joystick, SDL_CONTROLLER_AXIS_RIGHTY, axis);

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static void HIDAPI_DriverXboxOneBluetooth_HandleGuidePacket(SDL_Joystick *joystick, SDL_DriverXboxOne_Context *ctx, const Uint8 *data, int size)
{
    ctx->has_guide_packet = SDL_TRUE;
    SDL_PrivateJoystickButton(joystick, SDL_CONTROLLER_BUTTON_GUIDE, (data[1] & 0x01) ? SDL_PRESSED : SDL_RELEASED);
}

static void HIDAPI_DriverXboxOneBluetooth_HandleBatteryPacket(SDL_Joystick *joystick, SDL_DriverXboxOne_Context *ctx, const Uint8 *data, int size)
{
    Uint8 flags = data[1];
    SDL_bool on_usb = (((flags & 0x0C) >> 2) == 0);

    if (on_usb) {
        /* Does this ever happen? */
        SDL_PrivateJoystickBatteryLevel(joystick, SDL_JOYSTICK_POWER_WIRED);
    } else {
        switch ((flags & 0x03)) {
        case 0:
            SDL_PrivateJoystickBatteryLevel(joystick, SDL_JOYSTICK_POWER_LOW);
            break;
        case 1:
            SDL_PrivateJoystickBatteryLevel(joystick, SDL_JOYSTICK_POWER_MEDIUM);
            break;
        default: /* 2, 3 */
            SDL_PrivateJoystickBatteryLevel(joystick, SDL_JOYSTICK_POWER_FULL);
            break;
        }
    }
}

#ifdef SET_SERIAL_AFTER_OPEN
static void HIDAPI_DriverXboxOne_HandleSerialIDPacket(SDL_Joystick *joystick, SDL_DriverXboxOne_Context *ctx, Uint8 *data, int size)
{
    char serial[29];
    int i;

    for (i = 0; i < 14; ++i) {
        SDL_uitoa(data[6 + i], &serial[i * 2], 16);
    }
    serial[i * 2] = '\0';

    if (!joystick->serial || SDL_strcmp(joystick->serial, serial) != 0) {
#ifdef DEBUG_JOYSTICK
        SDL_Log("Setting serial number to %s\n", serial);
#endif
        joystick->serial = SDL_strdup(serial);
    }
}
#endif /* SET_SERIAL_AFTER_OPEN */

static SDL_bool HIDAPI_DriverXboxOne_UpdateInitState(SDL_HIDAPI_Device *device, SDL_DriverXboxOne_Context *ctx)
{
    SDL_XboxOneInitState prev_state;
    do {
        prev_state = ctx->init_state;

        switch (ctx->init_state) {
        case XBOX_ONE_INIT_STATE_START_NEGOTIATING:
#if defined(__WIN32__) || defined(__WINGDK__)
            /* The Windows driver is taking care of negotiation */
            SetInitState(ctx, XBOX_ONE_INIT_STATE_COMPLETE);
#else
            SetInitState(ctx, XBOX_ONE_INIT_STATE_NEGOTIATING);
            ctx->init_packet = 0;
            if (!SendControllerInit(device, ctx)) {
                return SDL_FALSE;
            }
#endif
            break;
        case XBOX_ONE_INIT_STATE_NEGOTIATING:
            if (SDL_TICKS_PASSED(SDL_GetTicks(), ctx->send_time + CONTROLLER_NEGOTIATION_TIMEOUT_MS)) {
                /* We haven't heard anything, let's move on */
#ifdef DEBUG_JOYSTICK
                SDL_Log("Init sequence %d timed out after %u ms\n", ctx->init_packet, (SDL_GetTicks() - ctx->send_time));
#endif
                ++ctx->init_packet;
                if (!SendControllerInit(device, ctx)) {
                    return SDL_FALSE;
                }
            }
            break;
        case XBOX_ONE_INIT_STATE_PREPARE_INPUT:
            if (SDL_TICKS_PASSED(SDL_GetTicks(), ctx->send_time + CONTROLLER_PREPARE_INPUT_TIMEOUT_MS)) {
#ifdef DEBUG_JOYSTICK
                SDL_Log("Prepare input complete after %u ms\n", (SDL_GetTicks() - ctx->send_time));
#endif
                SetInitState(ctx, XBOX_ONE_INIT_STATE_COMPLETE);
            }
            break;
        case XBOX_ONE_INIT_STATE_COMPLETE:
            break;
        }

    } while (ctx->init_state != prev_state);

    return SDL_TRUE;
}

static SDL_bool HIDAPI_DriverXboxOne_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverXboxOne_Context *ctx = (SDL_DriverXboxOne_Context *)device->context;
    SDL_Joystick *joystick = NULL;
    Uint8 data[USB_PACKET_LENGTH];
    int size;

    if (device->num_joysticks > 0) {
        joystick = SDL_JoystickFromInstanceID(device->joysticks[0]);
    } else {
        return SDL_FALSE;
    }

    while ((size = SDL_hid_read_timeout(device->dev, data, sizeof(data), 0)) > 0) {
#ifdef DEBUG_XBOX_PROTOCOL
        HIDAPI_DumpPacket("Xbox One packet: size = %d", data, size);
#endif
        if (ctx->bluetooth) {
            switch (data[0]) {
            case 0x01:
                if (!joystick) {
                    break;
                }
                if (size >= 16) {
                    HIDAPI_DriverXboxOneBluetooth_HandleStatePacket(joystick, ctx, data, size);
                } else {
#ifdef DEBUG_JOYSTICK
                    SDL_Log("Unknown Xbox One Bluetooth packet size: %d\n", size);
#endif
                }
                break;
            case 0x02:
                if (!joystick) {
                    break;
                }
                HIDAPI_DriverXboxOneBluetooth_HandleGuidePacket(joystick, ctx, data, size);
                break;
            case 0x04:
                if (!joystick) {
                    break;
                }
                HIDAPI_DriverXboxOneBluetooth_HandleBatteryPacket(joystick, ctx, data, size);
                break;
            default:
#ifdef DEBUG_JOYSTICK
                SDL_Log("Unknown Xbox One packet: 0x%.2x\n", data[0]);
#endif
                break;
            }
        } else {
            switch (data[0]) {
            case 0x01:
                /* ACK packet */
                /* The data bytes are:
                    0x01 0x20 NN 0x09, where NN is the packet sequence
                    then 0x00
                    then a byte of the sequence being acked
                    then 0x20
                    then 16-bit LE value, the size of the previous packet payload when it's a single packet
                    then 4 bytes of unknown data, often all zero
                 */
                break;
            case 0x02:
                /* Controller is connected and waiting for initialization */
                /* The data bytes are:
                   0x02 0x20 NN 0x1c, where NN is the packet sequence
                   then 6 bytes of wireless MAC address
                   then 2 bytes padding
                   then 16-bit VID
                   then 16-bit PID
                   then 16-bit firmware version quartet AA.BB.CC.DD
                        e.g. 0x05 0x00 0x05 0x00 0x51 0x0a 0x00 0x00
                             is firmware version 5.5.2641.0, and product version 0x0505 = 1285
                   then 8 bytes of unknown data
                */
                if (data[1] == 0x20) {
#ifdef DEBUG_JOYSTICK
                    SDL_Log("Controller announce after %u ms\n", (SDL_GetTicks() - ctx->start_time));
#endif
                    SetInitState(ctx, XBOX_ONE_INIT_STATE_START_NEGOTIATING);
                } else {
                    /* Possibly an announce from a device plugged into the controller */
                }
                break;
            case 0x03:
                /* Controller status update */
                if (!joystick) {
                    /* We actually want to handle this packet any time it arrives */
                    /*break;*/
                }
                HIDAPI_DriverXboxOne_HandleStatusPacket(joystick, ctx, data, size);
                break;
            case 0x04:
                /* Unknown chatty controller information, sent by both sides */
                break;
            case 0x06:
                /* Unknown chatty controller information, sent by both sides */
                break;
            case 0x07:
                if (!joystick) {
                    break;
                }
                HIDAPI_DriverXboxOne_HandleModePacket(joystick, ctx, data, size);
                break;
            case 0x0C:
                if (!joystick) {
                    break;
                }
                HIDAPI_DriverXboxOne_HandleUnmappedStatePacket(joystick, ctx, data, size);
                break;
            case 0x1E:
                /* If the packet starts with this:
                    0x1E 0x30 0x07 0x10 0x04 0x00
                    then the next 14 bytes are the controller serial number
                        e.g. 0x30 0x39 0x37 0x31 0x32 0x33 0x33 0x32 0x33 0x35 0x34 0x30 0x33 0x36
                        is serial number "3039373132333332333534303336"

                   The controller sends that in response to this request:
                    0x1E 0x30 0x07 0x01 0x04
                */
                if (!joystick) {
                    break;
                }
#ifdef SET_SERIAL_AFTER_OPEN
                if (size == 20 && data[3] == 0x10) {
                    HIDAPI_DriverXboxOne_HandleSerialIDPacket(joystick, ctx, data, size);
                }
#endif
                break;
            case 0x20:
                if (ctx->init_state < XBOX_ONE_INIT_STATE_COMPLETE) {
                    SetInitState(ctx, XBOX_ONE_INIT_STATE_COMPLETE);

                    /* Ignore the first input, it may be spurious */
#ifdef DEBUG_JOYSTICK
                    SDL_Log("Controller ignoring spurious input\n");
#endif
                    break;
                }
                if (!joystick) {
                    break;
                }
                HIDAPI_DriverXboxOne_HandleStatePacket(joystick, ctx, data, size);
                break;
            default:
#ifdef DEBUG_JOYSTICK
                SDL_Log("Unknown Xbox One packet: 0x%.2x\n", data[0]);
#endif
                break;
            }

            SendAckIfNeeded(device, data, size);

            if (ctx->init_state == XBOX_ONE_INIT_STATE_NEGOTIATING) {
                const SDL_DriverXboxOne_InitPacket *packet = &xboxone_init_packets[ctx->init_packet];

                if (size >= 4 && data[0] == packet->response[0] && data[1] == packet->response[1]) {
#ifdef DEBUG_JOYSTICK
                    SDL_Log("Init sequence %d got response after %u ms\n", ctx->init_packet, (SDL_GetTicks() - ctx->send_time));
#endif
                    ++ctx->init_packet;
                    SendControllerInit(device, ctx);
                }
            }
        }
    }

    HIDAPI_DriverXboxOne_UpdateInitState(device, ctx);
    HIDAPI_DriverXboxOne_UpdateRumble(device);

    if (size < 0) {
        /* Read error, device is disconnected */
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    return size >= 0;
}

static void HIDAPI_DriverXboxOne_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverXboxOne_Context *ctx = (SDL_DriverXboxOne_Context *)device->context;

    SDL_DelHintCallback(SDL_HINT_JOYSTICK_HIDAPI_XBOX_ONE_HOME_LED,
                        SDL_HomeLEDHintChanged, ctx);
}

static void HIDAPI_DriverXboxOne_FreeDevice(SDL_HIDAPI_Device *device)
{
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverXboxOne = {
    SDL_HINT_JOYSTICK_HIDAPI_XBOX_ONE,
    SDL_TRUE,
    HIDAPI_DriverXboxOne_RegisterHints,
    HIDAPI_DriverXboxOne_UnregisterHints,
    HIDAPI_DriverXboxOne_IsEnabled,
    HIDAPI_DriverXboxOne_IsSupportedDevice,
    HIDAPI_DriverXboxOne_InitDevice,
    HIDAPI_DriverXboxOne_GetDevicePlayerIndex,
    HIDAPI_DriverXboxOne_SetDevicePlayerIndex,
    HIDAPI_DriverXboxOne_UpdateDevice,
    HIDAPI_DriverXboxOne_OpenJoystick,
    HIDAPI_DriverXboxOne_RumbleJoystick,
    HIDAPI_DriverXboxOne_RumbleJoystickTriggers,
    HIDAPI_DriverXboxOne_GetJoystickCapabilities,
    HIDAPI_DriverXboxOne_SetJoystickLED,
    HIDAPI_DriverXboxOne_SendJoystickEffect,
    HIDAPI_DriverXboxOne_SetJoystickSensorsEnabled,
    HIDAPI_DriverXboxOne_CloseJoystick,
    HIDAPI_DriverXboxOne_FreeDevice,
};

#endif /* SDL_JOYSTICK_HIDAPI_XBOXONE */

#endif /* SDL_JOYSTICK_HIDAPI */

/* vi: set ts=4 sw=4 expandtab: */
