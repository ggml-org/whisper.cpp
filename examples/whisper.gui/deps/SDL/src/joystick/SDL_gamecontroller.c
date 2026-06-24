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

/* This is the game controller API for Simple DirectMedia Layer */

#include "SDL_events.h"
#include "SDL_hints.h"
#include "SDL_timer.h"
#include "SDL_sysjoystick.h"
#include "SDL_joystick_c.h"
#include "SDL_steam_virtual_gamepad.h"
#include "SDL_gamecontrollerdb.h"
#include "controller_type.h"
#include "usb_ids.h"
#include "hidapi/SDL_hidapi_nintendo.h"

#ifndef SDL_EVENTS_DISABLED
#include "../events/SDL_events_c.h"
#endif

#if defined(__ANDROID__)
#include "SDL_system.h"
#endif

/* Many controllers turn the center button into an instantaneous button press */
#define SDL_MINIMUM_GUIDE_BUTTON_DELAY_MS 250

#define SDL_CONTROLLER_CRC_FIELD           "crc:"
#define SDL_CONTROLLER_CRC_FIELD_SIZE      4 /* hard-coded for speed */
#define SDL_CONTROLLER_TYPE_FIELD          "type:"
#define SDL_CONTROLLER_TYPE_FIELD_SIZE     SDL_strlen(SDL_CONTROLLER_TYPE_FIELD)
#define SDL_CONTROLLER_PLATFORM_FIELD      "platform:"
#define SDL_CONTROLLER_PLATFORM_FIELD_SIZE SDL_strlen(SDL_CONTROLLER_PLATFORM_FIELD)
#define SDL_CONTROLLER_HINT_FIELD          "hint:"
#define SDL_CONTROLLER_HINT_FIELD_SIZE     SDL_strlen(SDL_CONTROLLER_HINT_FIELD)
#define SDL_CONTROLLER_SDKGE_FIELD         "sdk>=:"
#define SDL_CONTROLLER_SDKGE_FIELD_SIZE    SDL_strlen(SDL_CONTROLLER_SDKGE_FIELD)
#define SDL_CONTROLLER_SDKLE_FIELD         "sdk<=:"
#define SDL_CONTROLLER_SDKLE_FIELD_SIZE    SDL_strlen(SDL_CONTROLLER_SDKLE_FIELD)

/* a list of currently opened game controllers */
static SDL_GameController *SDL_gamecontrollers SDL_GUARDED_BY(SDL_joystick_lock) = NULL;

typedef struct
{
    SDL_GameControllerBindType inputType;
    union
    {
        int button;

        struct
        {
            int axis;
            int axis_min;
            int axis_max;
        } axis;

        struct
        {
            int hat;
            int hat_mask;
        } hat;

    } input;

    SDL_GameControllerBindType outputType;
    union
    {
        SDL_GameControllerButton button;

        struct
        {
            SDL_GameControllerAxis axis;
            int axis_min;
            int axis_max;
        } axis;

    } output;

} SDL_ExtendedGameControllerBind;

/* our hard coded list of mapping support */
typedef enum
{
    SDL_CONTROLLER_MAPPING_PRIORITY_DEFAULT,
    SDL_CONTROLLER_MAPPING_PRIORITY_API,
    SDL_CONTROLLER_MAPPING_PRIORITY_USER,
} SDL_ControllerMappingPriority;

#define _guarded SDL_GUARDED_BY(SDL_joystick_lock)

typedef struct _ControllerMapping_t
{
    SDL_JoystickGUID guid _guarded;
    char *name _guarded;
    char *mapping _guarded;
    SDL_ControllerMappingPriority priority _guarded;
    struct _ControllerMapping_t *next _guarded;
} ControllerMapping_t;

#undef _guarded

static SDL_JoystickGUID s_zeroGUID;
static ControllerMapping_t *s_pSupportedControllers SDL_GUARDED_BY(SDL_joystick_lock) = NULL;
static ControllerMapping_t *s_pDefaultMapping SDL_GUARDED_BY(SDL_joystick_lock) = NULL;
static ControllerMapping_t *s_pXInputMapping SDL_GUARDED_BY(SDL_joystick_lock) = NULL;
static char gamecontroller_magic;

#define _guarded SDL_GUARDED_BY(SDL_joystick_lock)

/* The SDL game controller structure */
struct _SDL_GameController
{
    const void *magic _guarded;

    SDL_Joystick *joystick _guarded; /* underlying joystick device */
    int ref_count _guarded;

    const char *name _guarded;
    SDL_GameControllerType type _guarded;
    ControllerMapping_t *mapping _guarded;
    int num_bindings _guarded;
    SDL_ExtendedGameControllerBind *bindings _guarded;
    SDL_ExtendedGameControllerBind **last_match_axis _guarded;
    Uint8 *last_hat_mask _guarded;
    Uint32 guide_button_down _guarded;

    struct _SDL_GameController *next _guarded; /* pointer to next game controller we have allocated */
};

#undef _guarded

#define CHECK_GAMECONTROLLER_MAGIC(gamecontroller, retval)                   \
    if (!gamecontroller || gamecontroller->magic != &gamecontroller_magic || \
        !SDL_PrivateJoystickValid(gamecontroller->joystick)) {               \
        SDL_InvalidParamError("gamecontroller");                             \
        SDL_UnlockJoysticks();                                               \
        return retval;                                                       \
    }

static SDL_vidpid_list SDL_allowed_controllers = {
    SDL_HINT_GAMECONTROLLER_IGNORE_DEVICES_EXCEPT, 0, 0, NULL,
    NULL, 0, 0, NULL,
    0, NULL,
    SDL_FALSE
};
static SDL_vidpid_list SDL_ignored_controllers = {
    SDL_HINT_GAMECONTROLLER_IGNORE_DEVICES, 0, 0, NULL,
    NULL, 0, 0, NULL,
    0, NULL,
    SDL_FALSE
};

static ControllerMapping_t *SDL_PrivateAddMappingForGUID(SDL_JoystickGUID jGUID, const char *mappingString, SDL_bool *existing, SDL_ControllerMappingPriority priority);
static int SDL_PrivateGameControllerAxis(SDL_GameController *gamecontroller, SDL_GameControllerAxis axis, Sint16 value);
static int SDL_PrivateGameControllerButton(SDL_GameController *gamecontroller, SDL_GameControllerButton button, Uint8 state);

static SDL_bool HasSameOutput(SDL_ExtendedGameControllerBind *a, SDL_ExtendedGameControllerBind *b)
{
    if (a->outputType != b->outputType) {
        return SDL_FALSE;
    }

    if (a->outputType == SDL_CONTROLLER_BINDTYPE_AXIS) {
        return a->output.axis.axis == b->output.axis.axis;
    } else {
        return a->output.button == b->output.button;
    }
}

static void ResetOutput(SDL_GameController *gamecontroller, SDL_ExtendedGameControllerBind *bind)
{
    if (bind->outputType == SDL_CONTROLLER_BINDTYPE_AXIS) {
        SDL_PrivateGameControllerAxis(gamecontroller, bind->output.axis.axis, 0);
    } else {
        SDL_PrivateGameControllerButton(gamecontroller, bind->output.button, SDL_RELEASED);
    }
}

static void HandleJoystickAxis(SDL_GameController *gamecontroller, int axis, int value)
{
    int i;
    SDL_ExtendedGameControllerBind *last_match;
    SDL_ExtendedGameControllerBind *match = NULL;

    SDL_AssertJoysticksLocked();

    last_match = gamecontroller->last_match_axis[axis];
    for (i = 0; i < gamecontroller->num_bindings; ++i) {
        SDL_ExtendedGameControllerBind *binding = &gamecontroller->bindings[i];
        if (binding->inputType == SDL_CONTROLLER_BINDTYPE_AXIS &&
            axis == binding->input.axis.axis) {
            if (binding->input.axis.axis_min < binding->input.axis.axis_max) {
                if (value >= binding->input.axis.axis_min &&
                    value <= binding->input.axis.axis_max) {
                    match = binding;
                    break;
                }
            } else {
                if (value >= binding->input.axis.axis_max &&
                    value <= binding->input.axis.axis_min) {
                    match = binding;
                    break;
                }
            }
        }
    }

    if (last_match && (!match || !HasSameOutput(last_match, match))) {
        /* Clear the last input that this axis generated */
        ResetOutput(gamecontroller, last_match);
    }

    if (match) {
        if (match->outputType == SDL_CONTROLLER_BINDTYPE_AXIS) {
            if (match->input.axis.axis_min != match->output.axis.axis_min || match->input.axis.axis_max != match->output.axis.axis_max) {
                float normalized_value = (float)(value - match->input.axis.axis_min) / (match->input.axis.axis_max - match->input.axis.axis_min);
                value = match->output.axis.axis_min + (int)(normalized_value * (match->output.axis.axis_max - match->output.axis.axis_min));
            }
            SDL_PrivateGameControllerAxis(gamecontroller, match->output.axis.axis, (Sint16)value);
        } else {
            Uint8 state;
            int threshold = match->input.axis.axis_min + (match->input.axis.axis_max - match->input.axis.axis_min) / 2;
            if (match->input.axis.axis_max < match->input.axis.axis_min) {
                state = (value <= threshold) ? SDL_PRESSED : SDL_RELEASED;
            } else {
                state = (value >= threshold) ? SDL_PRESSED : SDL_RELEASED;
            }
            SDL_PrivateGameControllerButton(gamecontroller, match->output.button, state);
        }
    }
    gamecontroller->last_match_axis[axis] = match;
}

static void HandleJoystickButton(SDL_GameController *gamecontroller, int button, Uint8 state)
{
    int i;

    SDL_AssertJoysticksLocked();

    for (i = 0; i < gamecontroller->num_bindings; ++i) {
        SDL_ExtendedGameControllerBind *binding = &gamecontroller->bindings[i];
        if (binding->inputType == SDL_CONTROLLER_BINDTYPE_BUTTON &&
            button == binding->input.button) {
            if (binding->outputType == SDL_CONTROLLER_BINDTYPE_AXIS) {
                int value = state ? binding->output.axis.axis_max : binding->output.axis.axis_min;
                SDL_PrivateGameControllerAxis(gamecontroller, binding->output.axis.axis, (Sint16)value);
            } else {
                SDL_PrivateGameControllerButton(gamecontroller, binding->output.button, state);
            }
            break;
        }
    }
}

static void HandleJoystickHat(SDL_GameController *gamecontroller, int hat, Uint8 value)
{
    int i;
    Uint8 last_mask, changed_mask;

    SDL_AssertJoysticksLocked();

    last_mask = gamecontroller->last_hat_mask[hat];
    changed_mask = (last_mask ^ value);
    for (i = 0; i < gamecontroller->num_bindings; ++i) {
        SDL_ExtendedGameControllerBind *binding = &gamecontroller->bindings[i];
        if (binding->inputType == SDL_CONTROLLER_BINDTYPE_HAT && hat == binding->input.hat.hat) {
            if ((changed_mask & binding->input.hat.hat_mask) != 0) {
                if (value & binding->input.hat.hat_mask) {
                    if (binding->outputType == SDL_CONTROLLER_BINDTYPE_AXIS) {
                        SDL_PrivateGameControllerAxis(gamecontroller, binding->output.axis.axis, (Sint16)binding->output.axis.axis_max);
                    } else {
                        SDL_PrivateGameControllerButton(gamecontroller, binding->output.button, SDL_PRESSED);
                    }
                } else {
                    ResetOutput(gamecontroller, binding);
                }
            }
        }
    }
    gamecontroller->last_hat_mask[hat] = value;
}

/* The joystick layer will _also_ send events to recenter before disconnect,
    but it has to make (sometimes incorrect) guesses at what being "centered"
    is. The game controller layer, however, can set a definite logical idle
    position, so set them all here. If we happened to already be at the
    center thanks to the joystick layer or idle hands, this won't generate
    duplicate events. */
static void RecenterGameController(SDL_GameController *gamecontroller)
{
    SDL_GameControllerButton button;
    SDL_GameControllerAxis axis;

    for (button = (SDL_GameControllerButton)0; button < SDL_CONTROLLER_BUTTON_MAX; button++) {
        if (SDL_GameControllerGetButton(gamecontroller, button)) {
            SDL_PrivateGameControllerButton(gamecontroller, button, SDL_RELEASED);
        }
    }

    for (axis = (SDL_GameControllerAxis)0; axis < SDL_CONTROLLER_AXIS_MAX; axis++) {
        if (SDL_GameControllerGetAxis(gamecontroller, axis) != 0) {
            SDL_PrivateGameControllerAxis(gamecontroller, axis, 0);
        }
    }
}

/*
 * Event filter to fire controller events from joystick ones
 */
static int SDLCALL SDL_GameControllerEventWatcher(void *userdata, SDL_Event *event)
{
    SDL_GameController *controller;

    switch (event->type) {
    case SDL_JOYAXISMOTION:
    {
        SDL_AssertJoysticksLocked();

        for (controller = SDL_gamecontrollers; controller; controller = controller->next) {
            if (controller->joystick->instance_id == event->jaxis.which) {
                HandleJoystickAxis(controller, event->jaxis.axis, event->jaxis.value);
                break;
            }
        }
    } break;
    case SDL_JOYBUTTONDOWN:
    case SDL_JOYBUTTONUP:
    {
        SDL_AssertJoysticksLocked();

        for (controller = SDL_gamecontrollers; controller; controller = controller->next) {
            if (controller->joystick->instance_id == event->jbutton.which) {
                HandleJoystickButton(controller, event->jbutton.button, event->jbutton.state);
                break;
            }
        }
    } break;
    case SDL_JOYHATMOTION:
    {
        SDL_AssertJoysticksLocked();

        for (controller = SDL_gamecontrollers; controller; controller = controller->next) {
            if (controller->joystick->instance_id == event->jhat.which) {
                HandleJoystickHat(controller, event->jhat.hat, event->jhat.value);
                break;
            }
        }
    } break;
    case SDL_JOYDEVICEADDED:
    {
        if (SDL_IsGameController(event->jdevice.which)) {
            SDL_Event deviceevent;
            deviceevent.type = SDL_CONTROLLERDEVICEADDED;
            deviceevent.cdevice.which = event->jdevice.which;
            SDL_PushEvent(&deviceevent);
        }
    } break;
    case SDL_JOYDEVICEREMOVED:
    {
        SDL_AssertJoysticksLocked();

        for (controller = SDL_gamecontrollers; controller; controller = controller->next) {
            if (controller->joystick->instance_id == event->jdevice.which) {
                RecenterGameController(controller);
                break;
            }
        }

        /* We don't know if this was a game controller, so go ahead and send an event */
        {
            SDL_Event deviceevent;

            deviceevent.type = SDL_CONTROLLERDEVICEREMOVED;
            deviceevent.cdevice.which = event->jdevice.which;
            SDL_PushEvent(&deviceevent);
        }
    } break;
    default:
        break;
    }

    return 1;
}

#ifdef __ANDROID__
/*
 * Helper function to guess at a mapping based on the elements reported for this controller
 */
static ControllerMapping_t *SDL_CreateMappingForAndroidController(SDL_JoystickGUID guid)
{
    const int face_button_mask = ((1 << SDL_CONTROLLER_BUTTON_A) |
                                  (1 << SDL_CONTROLLER_BUTTON_B) |
                                  (1 << SDL_CONTROLLER_BUTTON_X) |
                                  (1 << SDL_CONTROLLER_BUTTON_Y));
    SDL_bool existing;
    char mapping_string[1024];
    int button_mask;
    int axis_mask;

    button_mask = SDL_SwapLE16(*(Uint16 *)(&guid.data[sizeof(guid.data) - 4]));
    axis_mask = SDL_SwapLE16(*(Uint16 *)(&guid.data[sizeof(guid.data) - 2]));
    if (!button_mask && !axis_mask) {
        /* Accelerometer, shouldn't have a game controller mapping */
        return NULL;
    }
    if (!(button_mask & face_button_mask)) {
        /* We don't know what buttons or axes are supported, don't make up a mapping */
        return NULL;
    }

    SDL_strlcpy(mapping_string, "none,*,", sizeof(mapping_string));

    if (button_mask & (1 << SDL_CONTROLLER_BUTTON_A)) {
        SDL_strlcat(mapping_string, "a:b0,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_CONTROLLER_BUTTON_B)) {
        SDL_strlcat(mapping_string, "b:b1,", sizeof(mapping_string));
    } else if (button_mask & (1 << SDL_CONTROLLER_BUTTON_BACK)) {
        /* Use the back button as "B" for easy UI navigation with TV remotes */
        SDL_strlcat(mapping_string, "b:b4,", sizeof(mapping_string));
        button_mask &= ~(1 << SDL_CONTROLLER_BUTTON_BACK);
    }
    if (button_mask & (1 << SDL_CONTROLLER_BUTTON_X)) {
        SDL_strlcat(mapping_string, "x:b2,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_CONTROLLER_BUTTON_Y)) {
        SDL_strlcat(mapping_string, "y:b3,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_CONTROLLER_BUTTON_BACK)) {
        SDL_strlcat(mapping_string, "back:b4,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_CONTROLLER_BUTTON_GUIDE)) {
        /* The guide button generally isn't functional (or acts as a home button) on most Android controllers before Android 11 */
        if (SDL_GetAndroidSDKVersion() >= 30 /* Android 11 */) {
            SDL_strlcat(mapping_string, "guide:b5,", sizeof(mapping_string));
        }
    }
    if (button_mask & (1 << SDL_CONTROLLER_BUTTON_START)) {
        SDL_strlcat(mapping_string, "start:b6,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_CONTROLLER_BUTTON_LEFTSTICK)) {
        SDL_strlcat(mapping_string, "leftstick:b7,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_CONTROLLER_BUTTON_RIGHTSTICK)) {
        SDL_strlcat(mapping_string, "rightstick:b8,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_CONTROLLER_BUTTON_LEFTSHOULDER)) {
        SDL_strlcat(mapping_string, "leftshoulder:b9,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_CONTROLLER_BUTTON_RIGHTSHOULDER)) {
        SDL_strlcat(mapping_string, "rightshoulder:b10,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_CONTROLLER_BUTTON_DPAD_UP)) {
        SDL_strlcat(mapping_string, "dpup:b11,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_CONTROLLER_BUTTON_DPAD_DOWN)) {
        SDL_strlcat(mapping_string, "dpdown:b12,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_CONTROLLER_BUTTON_DPAD_LEFT)) {
        SDL_strlcat(mapping_string, "dpleft:b13,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_CONTROLLER_BUTTON_DPAD_RIGHT)) {
        SDL_strlcat(mapping_string, "dpright:b14,", sizeof(mapping_string));
    }
    if (axis_mask & (1 << SDL_CONTROLLER_AXIS_LEFTX)) {
        SDL_strlcat(mapping_string, "leftx:a0,", sizeof(mapping_string));
    }
    if (axis_mask & (1 << SDL_CONTROLLER_AXIS_LEFTY)) {
        SDL_strlcat(mapping_string, "lefty:a1,", sizeof(mapping_string));
    }
    if (axis_mask & (1 << SDL_CONTROLLER_AXIS_RIGHTX)) {
        SDL_strlcat(mapping_string, "rightx:a2,", sizeof(mapping_string));
    }
    if (axis_mask & (1 << SDL_CONTROLLER_AXIS_RIGHTY)) {
        SDL_strlcat(mapping_string, "righty:a3,", sizeof(mapping_string));
    }
    if (axis_mask & (1 << SDL_CONTROLLER_AXIS_TRIGGERLEFT)) {
        SDL_strlcat(mapping_string, "lefttrigger:a4,", sizeof(mapping_string));
    }
    if (axis_mask & (1 << SDL_CONTROLLER_AXIS_TRIGGERRIGHT)) {
        SDL_strlcat(mapping_string, "righttrigger:a5,", sizeof(mapping_string));
    }

    return SDL_PrivateAddMappingForGUID(guid, mapping_string, &existing, SDL_CONTROLLER_MAPPING_PRIORITY_DEFAULT);
}
#endif /* __ANDROID__ */

/*
 * Helper function to guess at a mapping for HIDAPI controllers
 */
static ControllerMapping_t *SDL_CreateMappingForHIDAPIController(SDL_JoystickGUID guid)
{
    SDL_bool existing;
    char mapping_string[1024];
    Uint16 vendor;
    Uint16 product;

    SDL_strlcpy(mapping_string, "none,*,", sizeof(mapping_string));

    SDL_GetJoystickGUIDInfo(guid, &vendor, &product, NULL, NULL);

    if ((vendor == USB_VENDOR_NINTENDO && product == USB_PRODUCT_NINTENDO_GAMECUBE_ADAPTER) ||
        (vendor == USB_VENDOR_DRAGONRISE &&
         (product == USB_PRODUCT_EVORETRO_GAMECUBE_ADAPTER1 ||
          product == USB_PRODUCT_EVORETRO_GAMECUBE_ADAPTER2))) {
        /* GameCube driver has 12 buttons and 6 axes */
        SDL_strlcat(mapping_string, "a:b0,b:b1,dpdown:b6,dpleft:b4,dpright:b5,dpup:b7,lefttrigger:a4,leftx:a0,lefty:a1~,rightshoulder:b9,righttrigger:a5,rightx:a2,righty:a3~,start:b8,x:b2,y:b3,", sizeof(mapping_string));
    } else if (vendor == USB_VENDOR_NINTENDO &&
               (guid.data[15] == k_eSwitchDeviceInfoControllerType_HVCLeft ||
                guid.data[15] == k_eSwitchDeviceInfoControllerType_HVCRight ||
                guid.data[15] == k_eSwitchDeviceInfoControllerType_NESLeft ||
                guid.data[15] == k_eSwitchDeviceInfoControllerType_NESRight ||
                guid.data[15] == k_eSwitchDeviceInfoControllerType_SNES ||
                guid.data[15] == k_eSwitchDeviceInfoControllerType_N64 ||
                guid.data[15] == k_eSwitchDeviceInfoControllerType_SEGA_Genesis ||
                guid.data[15] == k_eWiiExtensionControllerType_None ||
                guid.data[15] == k_eWiiExtensionControllerType_Nunchuk ||
                guid.data[15] == k_eSwitchDeviceInfoControllerType_JoyConLeft ||
                guid.data[15] == k_eSwitchDeviceInfoControllerType_JoyConRight)) {
        switch (guid.data[15]) {
        case k_eSwitchDeviceInfoControllerType_HVCLeft:
            SDL_strlcat(mapping_string, "a:b0,b:b1,back:b4,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,leftshoulder:b9,rightshoulder:b10,start:b6,", sizeof(mapping_string));
            break;
        case k_eSwitchDeviceInfoControllerType_HVCRight:
            SDL_strlcat(mapping_string, "a:b0,b:b1,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,leftshoulder:b9,rightshoulder:b10,", sizeof(mapping_string));
            break;
        case k_eSwitchDeviceInfoControllerType_NESLeft:
        case k_eSwitchDeviceInfoControllerType_NESRight:
            SDL_strlcat(mapping_string, "a:b0,b:b1,back:b4,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,leftshoulder:b9,rightshoulder:b10,start:b6,", sizeof(mapping_string));
            break;
        case k_eSwitchDeviceInfoControllerType_SNES:
            SDL_strlcat(mapping_string, "a:b0,b:b1,back:b4,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,leftshoulder:b9,lefttrigger:a4,rightshoulder:b10,righttrigger:a5,start:b6,x:b2,y:b3,", sizeof(mapping_string));
            break;
        case k_eSwitchDeviceInfoControllerType_N64:
            SDL_strlcat(mapping_string, "a:b0,b:b1,back:b4,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,guide:b5,leftshoulder:b9,leftstick:b7,lefttrigger:a4,leftx:a0,lefty:a1,rightshoulder:b10,righttrigger:a5,start:b6,x:b2,y:b3,misc1:b15,", sizeof(mapping_string));
            break;
        case k_eSwitchDeviceInfoControllerType_SEGA_Genesis:
            SDL_strlcat(mapping_string, "a:b0,b:b1,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,guide:b5,leftshoulder:b9,rightshoulder:b10,righttrigger:a5,start:b6,x:b2,y:b3,misc1:b15,", sizeof(mapping_string));
            break;
        case k_eWiiExtensionControllerType_None:
            SDL_strlcat(mapping_string, "a:b0,b:b1,back:b4,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,guide:b5,start:b6,x:b2,y:b3,", sizeof(mapping_string));
            break;
        case k_eWiiExtensionControllerType_Nunchuk:
        {
            /* FIXME: Should we map this to the left or right side? */
            const SDL_bool map_nunchuck_left_side = SDL_TRUE;

            if (map_nunchuck_left_side) {
                SDL_strlcat(mapping_string, "a:b0,b:b1,back:b4,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,guide:b5,leftshoulder:b9,lefttrigger:a4,leftx:a0,lefty:a1,start:b6,x:b2,y:b3,", sizeof(mapping_string));
            } else {
                SDL_strlcat(mapping_string, "a:b0,b:b1,back:b4,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,guide:b5,rightshoulder:b9,righttrigger:a4,rightx:a0,righty:a1,start:b6,x:b2,y:b3,", sizeof(mapping_string));
            }
        } break;
        default:
            if (SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_VERTICAL_JOY_CONS, SDL_FALSE)) {
                /* Vertical mode */
                if (guid.data[15] == k_eSwitchDeviceInfoControllerType_JoyConLeft) {
                    SDL_strlcat(mapping_string, "back:b4,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,leftshoulder:b9,leftstick:b7,lefttrigger:a4,leftx:a0,lefty:a1,misc1:b15,paddle2:b17,paddle4:b19,", sizeof(mapping_string));
                } else {
                    SDL_strlcat(mapping_string, "a:b0,b:b1,guide:b5,rightshoulder:b10,rightstick:b8,righttrigger:a5,rightx:a2,righty:a3,start:b6,x:b2,y:b3,paddle1:b16,paddle3:b18,", sizeof(mapping_string));
                }
            } else {
                /* Mini gamepad mode */
                if (guid.data[15] == k_eSwitchDeviceInfoControllerType_JoyConLeft) {
                    SDL_strlcat(mapping_string, "a:b0,b:b1,guide:b5,leftshoulder:b9,leftstick:b7,leftx:a0,lefty:a1,rightshoulder:b10,start:b6,x:b2,y:b3,paddle2:b17,paddle4:b19,", sizeof(mapping_string));
                } else {
                    SDL_strlcat(mapping_string, "a:b0,b:b1,guide:b5,leftshoulder:b9,leftstick:b7,leftx:a0,lefty:a1,rightshoulder:b10,start:b6,x:b2,y:b3,paddle1:b16,paddle3:b18,", sizeof(mapping_string));
                }
            }
            break;
        }
    } else {
        /* All other controllers have the standard set of 19 buttons and 6 axes */
        SDL_strlcat(mapping_string, "a:b0,b:b1,back:b4,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,guide:b5,leftshoulder:b9,leftstick:b7,lefttrigger:a4,leftx:a0,lefty:a1,rightshoulder:b10,rightstick:b8,righttrigger:a5,rightx:a2,righty:a3,start:b6,x:b2,y:b3,", sizeof(mapping_string));

        if (SDL_IsJoystickXboxSeriesX(vendor, product)) {
            /* XBox Series X Controllers have a share button under the guide button */
            SDL_strlcat(mapping_string, "misc1:b15,", sizeof(mapping_string));
        } else if (SDL_IsJoystickXboxOneElite(vendor, product)) {
            /* XBox One Elite Controllers have 4 back paddle buttons */
            SDL_strlcat(mapping_string, "paddle1:b15,paddle2:b17,paddle3:b16,paddle4:b18,", sizeof(mapping_string));
        } else if (SDL_IsJoystickSteamController(vendor, product)) {
            /* Steam controllers have 2 back paddle buttons */
            SDL_strlcat(mapping_string, "paddle1:b16,paddle2:b15,", sizeof(mapping_string));
        } else if (SDL_IsJoystickNintendoSwitchJoyConPair(vendor, product)) {
            /* The Nintendo Switch Joy-Con combined controller has a share button and paddles */
            SDL_strlcat(mapping_string, "misc1:b15,paddle1:b16,paddle2:b17,paddle3:b18,paddle4:b19,", sizeof(mapping_string));
        } else {
            switch (SDL_GetJoystickGameControllerTypeFromGUID(guid, NULL)) {
            case SDL_CONTROLLER_TYPE_PS4:
                /* PS4 controllers have an additional touchpad button */
                SDL_strlcat(mapping_string, "touchpad:b15,", sizeof(mapping_string));
                break;
            case SDL_CONTROLLER_TYPE_PS5:
                /* PS5 controllers have a microphone button and an additional touchpad button */
                SDL_strlcat(mapping_string, "touchpad:b15,misc1:b16,", sizeof(mapping_string));
                /* DualSense Edge controllers have paddles */
                if (SDL_IsJoystickDualSenseEdge(vendor, product)) {
                    SDL_strlcat(mapping_string, "paddle1:b20,paddle2:b19,paddle3:b18,paddle4:b17,", sizeof(mapping_string));
                }
                break;
            case SDL_CONTROLLER_TYPE_NINTENDO_SWITCH_PRO:
                /* Nintendo Switch Pro controllers have a screenshot button */
                SDL_strlcat(mapping_string, "misc1:b15,", sizeof(mapping_string));
                break;
            case SDL_CONTROLLER_TYPE_AMAZON_LUNA:
                /* Amazon Luna Controller has a mic button under the guide button */
                SDL_strlcat(mapping_string, "misc1:b15,", sizeof(mapping_string));
                break;
            case SDL_CONTROLLER_TYPE_GOOGLE_STADIA:
                /* The Google Stadia controller has a share button and a Google Assistant button */
                SDL_strlcat(mapping_string, "misc1:b15,", sizeof(mapping_string));
                break;
            case SDL_CONTROLLER_TYPE_NVIDIA_SHIELD:
                /* The NVIDIA SHIELD controller has a share button between back and start buttons */
                SDL_strlcat(mapping_string, "misc1:b15,", sizeof(mapping_string));

                if (product == USB_PRODUCT_NVIDIA_SHIELD_CONTROLLER_V103) {
                    /* The original SHIELD controller has a touchpad as well */
                    SDL_strlcat(mapping_string, "touchpad:b16,", sizeof(mapping_string));
                }
                break;
            default:
                if (vendor == 0 && product == 0) {
                    /* This is a Bluetooth Nintendo Switch Pro controller */
                    SDL_strlcat(mapping_string, "misc1:b15,", sizeof(mapping_string));
                }
                break;
            }
        }
    }

    return SDL_PrivateAddMappingForGUID(guid, mapping_string, &existing, SDL_CONTROLLER_MAPPING_PRIORITY_DEFAULT);
}

/*
 * Helper function to guess at a mapping for RAWINPUT controllers
 */
static ControllerMapping_t *SDL_CreateMappingForRAWINPUTController(SDL_JoystickGUID guid)
{
    SDL_bool existing;
    char mapping_string[1024];

    SDL_strlcpy(mapping_string, "none,*,", sizeof(mapping_string));
    SDL_strlcat(mapping_string, "a:b0,b:b1,x:b2,y:b3,back:b6,guide:b10,start:b7,leftstick:b8,rightstick:b9,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:a4,righttrigger:a5,", sizeof(mapping_string));

    return SDL_PrivateAddMappingForGUID(guid, mapping_string, &existing, SDL_CONTROLLER_MAPPING_PRIORITY_DEFAULT);
}

/*
 * Helper function to guess at a mapping for WGI controllers
 */
static ControllerMapping_t *SDL_CreateMappingForWGIController(SDL_JoystickGUID guid)
{
    SDL_bool existing;
    char mapping_string[1024];

    if (guid.data[15] != SDL_JOYSTICK_TYPE_GAMECONTROLLER) {
        return NULL;
    }

    SDL_strlcpy(mapping_string, "none,*,", sizeof(mapping_string));
    SDL_strlcat(mapping_string, "a:b0,b:b1,x:b2,y:b3,back:b6,start:b7,leftstick:b8,rightstick:b9,leftshoulder:b4,rightshoulder:b5,dpup:b10,dpdown:b12,dpleft:b13,dpright:b11,leftx:a1,lefty:a0~,rightx:a3,righty:a2~,lefttrigger:a4,righttrigger:a5,", sizeof(mapping_string));

    return SDL_PrivateAddMappingForGUID(guid, mapping_string, &existing, SDL_CONTROLLER_MAPPING_PRIORITY_DEFAULT);
}

/*
 * Helper function to scan the mappings database for a controller with the specified GUID
 */
static ControllerMapping_t *SDL_PrivateMatchControllerMappingForGUID(SDL_JoystickGUID guid, SDL_bool match_version, SDL_bool exact_match_crc)
{
    ControllerMapping_t *mapping, *best_match = NULL;
    Uint16 crc = 0;

    SDL_AssertJoysticksLocked();

    SDL_GetJoystickGUIDInfo(guid, NULL, NULL, NULL, &crc);

    /* Clear the CRC from the GUID for matching, the mappings never include it in the GUID */
    SDL_SetJoystickGUIDCRC(&guid, 0);

    if (!match_version) {
        SDL_SetJoystickGUIDVersion(&guid, 0);
    }

    for (mapping = s_pSupportedControllers; mapping; mapping = mapping->next) {
        SDL_JoystickGUID mapping_guid;

        if (SDL_memcmp(&mapping->guid, &s_zeroGUID, sizeof(mapping->guid)) == 0) {
            continue;
        }

        SDL_memcpy(&mapping_guid, &mapping->guid, sizeof(mapping_guid));
        if (!match_version) {
            SDL_SetJoystickGUIDVersion(&mapping_guid, 0);
        }

        if (SDL_memcmp(&guid, &mapping_guid, sizeof(guid)) == 0) {
            const char *crc_string = SDL_strstr(mapping->mapping, SDL_CONTROLLER_CRC_FIELD);
            if (crc_string) {
                Uint16 mapping_crc = (Uint16)SDL_strtol(crc_string + SDL_CONTROLLER_CRC_FIELD_SIZE, NULL, 16);

                if (mapping_crc != crc) {
                    /* This mapping specified a CRC and they don't match */
                    continue;
                }

                /* An exact match, including CRC */
                return mapping;
            } else if (crc && exact_match_crc) {
                return NULL;
            }

            if (!best_match) {
                best_match = mapping;
            }
        }
    }
    return best_match;
}

/*
 * Helper function to scan the mappings database for a controller with the specified GUID
 */
static ControllerMapping_t *SDL_PrivateGetControllerMappingForGUID(SDL_JoystickGUID guid, SDL_bool adding_mapping)
{
    ControllerMapping_t *mapping;

    mapping = SDL_PrivateMatchControllerMappingForGUID(guid, SDL_TRUE, adding_mapping);
    if (mapping) {
        return mapping;
    }

    if (adding_mapping) {
        /* We didn't find an existing mapping */
        return NULL;
    }

    /* Try harder to get the best match, or create a mapping */

    if (SDL_JoystickGUIDUsesVersion(guid)) {
        /* Try again, ignoring the version */
        mapping = SDL_PrivateMatchControllerMappingForGUID(guid, SDL_FALSE, SDL_FALSE);
        if (mapping) {
            return mapping;
        }
    }

#ifdef SDL_JOYSTICK_XINPUT
    if (SDL_IsJoystickXInput(guid)) {
        /* This is an XInput device */
        return s_pXInputMapping;
    }
#endif
    if (SDL_IsJoystickHIDAPI(guid)) {
        mapping = SDL_CreateMappingForHIDAPIController(guid);
    } else if (SDL_IsJoystickRAWINPUT(guid)) {
        mapping = SDL_CreateMappingForRAWINPUTController(guid);
    } else if (SDL_IsJoystickWGI(guid)) {
        mapping = SDL_CreateMappingForWGIController(guid);
    } else if (SDL_IsJoystickVirtual(guid)) {
        /* We'll pick up a robust mapping in VIRTUAL_JoystickGetGamepadMapping */
#ifdef __ANDROID__
    } else {
        mapping = SDL_CreateMappingForAndroidController(guid);
#endif
    }
    return mapping;
}

static const char *map_StringForGameControllerType[] = {
    "unknown",
    "xbox360",
    "xboxone",
    "ps3",
    "ps4",
    "switchpro",
    "virtual",
    "ps5",
    "amazonluna",
    "googlestadia",
    "nvidiashield",
    "joyconleft",
    "joyconright",
    "joyconpair"
};
SDL_COMPILE_TIME_ASSERT(map_StringForGameControllerType, SDL_arraysize(map_StringForGameControllerType) == SDL_CONTROLLER_TYPE_MAX);

/*
 * convert a string to its enum equivalent
 */
SDL_GameControllerType SDL_GetGameControllerTypeFromString(const char *str)
{
    int i;

    if (!str || str[0] == '\0') {
        return SDL_CONTROLLER_TYPE_UNKNOWN;
    }

    if (*str == '+' || *str == '-') {
        ++str;
    }

    for (i = 0; i < SDL_arraysize(map_StringForGameControllerType); ++i) {
        if (SDL_strcasecmp(str, map_StringForGameControllerType[i]) == 0) {
            return (SDL_GameControllerType)i;
        }
    }
    return SDL_CONTROLLER_TYPE_UNKNOWN;
}

static const char *map_StringForControllerAxis[] = {
    "leftx",
    "lefty",
    "rightx",
    "righty",
    "lefttrigger",
    "righttrigger",
    NULL
};

/*
 * convert a string to its enum equivalent
 */
SDL_GameControllerAxis SDL_GameControllerGetAxisFromString(const char *str)
{
    int entry;

    if (str == NULL || str[0] == '\0') {
        return SDL_CONTROLLER_AXIS_INVALID;
    }

    if (*str == '+' || *str == '-') {
        ++str;
    }

    for (entry = 0; map_StringForControllerAxis[entry]; ++entry) {
        if (SDL_strcasecmp(str, map_StringForControllerAxis[entry]) == 0) {
            return (SDL_GameControllerAxis)entry;
        }
    }
    return SDL_CONTROLLER_AXIS_INVALID;
}

/*
 * convert an enum to its string equivalent
 */
const char *SDL_GameControllerGetStringForAxis(SDL_GameControllerAxis axis)
{
    if (axis > SDL_CONTROLLER_AXIS_INVALID && axis < SDL_CONTROLLER_AXIS_MAX) {
        return map_StringForControllerAxis[axis];
    }
    return NULL;
}

static const char *map_StringForControllerButton[] = {
    "a",
    "b",
    "x",
    "y",
    "back",
    "guide",
    "start",
    "leftstick",
    "rightstick",
    "leftshoulder",
    "rightshoulder",
    "dpup",
    "dpdown",
    "dpleft",
    "dpright",
    "misc1",
    "paddle1",
    "paddle2",
    "paddle3",
    "paddle4",
    "touchpad",
    NULL
};

/*
 * convert a string to its enum equivalent
 */
SDL_GameControllerButton SDL_GameControllerGetButtonFromString(const char *str)
{
    int entry;
    if (str == NULL || str[0] == '\0') {
        return SDL_CONTROLLER_BUTTON_INVALID;
    }

    for (entry = 0; map_StringForControllerButton[entry]; ++entry) {
        if (SDL_strcasecmp(str, map_StringForControllerButton[entry]) == 0) {
            return (SDL_GameControllerButton)entry;
        }
    }
    return SDL_CONTROLLER_BUTTON_INVALID;
}

/*
 * convert an enum to its string equivalent
 */
const char *SDL_GameControllerGetStringForButton(SDL_GameControllerButton button)
{
    if (button > SDL_CONTROLLER_BUTTON_INVALID && button < SDL_CONTROLLER_BUTTON_MAX) {
        return map_StringForControllerButton[button];
    }
    return NULL;
}

/*
 * given a controller button name and a joystick name update our mapping structure with it
 */
static void SDL_PrivateGameControllerParseElement(SDL_GameController *gamecontroller, const char *szGameButton, const char *szJoystickButton)
{
    SDL_ExtendedGameControllerBind bind;
    SDL_GameControllerButton button;
    SDL_GameControllerAxis axis;
    SDL_bool invert_input = SDL_FALSE;
    char half_axis_input = 0;
    char half_axis_output = 0;

    SDL_AssertJoysticksLocked();

    if (*szGameButton == '+' || *szGameButton == '-') {
        half_axis_output = *szGameButton++;
    }

    axis = SDL_GameControllerGetAxisFromString(szGameButton);
    button = SDL_GameControllerGetButtonFromString(szGameButton);
    if (axis != SDL_CONTROLLER_AXIS_INVALID) {
        bind.outputType = SDL_CONTROLLER_BINDTYPE_AXIS;
        bind.output.axis.axis = axis;
        if (axis == SDL_CONTROLLER_AXIS_TRIGGERLEFT || axis == SDL_CONTROLLER_AXIS_TRIGGERRIGHT) {
            bind.output.axis.axis_min = 0;
            bind.output.axis.axis_max = SDL_JOYSTICK_AXIS_MAX;
        } else {
            if (half_axis_output == '+') {
                bind.output.axis.axis_min = 0;
                bind.output.axis.axis_max = SDL_JOYSTICK_AXIS_MAX;
            } else if (half_axis_output == '-') {
                bind.output.axis.axis_min = 0;
                bind.output.axis.axis_max = SDL_JOYSTICK_AXIS_MIN;
            } else {
                bind.output.axis.axis_min = SDL_JOYSTICK_AXIS_MIN;
                bind.output.axis.axis_max = SDL_JOYSTICK_AXIS_MAX;
            }
        }
    } else if (button != SDL_CONTROLLER_BUTTON_INVALID) {
        bind.outputType = SDL_CONTROLLER_BINDTYPE_BUTTON;
        bind.output.button = button;
    } else {
        SDL_SetError("Unexpected controller element %s", szGameButton);
        return;
    }

    if (*szJoystickButton == '+' || *szJoystickButton == '-') {
        half_axis_input = *szJoystickButton++;
    }
    if (szJoystickButton[SDL_strlen(szJoystickButton) - 1] == '~') {
        invert_input = SDL_TRUE;
    }

    if (szJoystickButton[0] == 'a' && SDL_isdigit((unsigned char)szJoystickButton[1])) {
        bind.inputType = SDL_CONTROLLER_BINDTYPE_AXIS;
        bind.input.axis.axis = SDL_atoi(&szJoystickButton[1]);
        if (half_axis_input == '+') {
            bind.input.axis.axis_min = 0;
            bind.input.axis.axis_max = SDL_JOYSTICK_AXIS_MAX;
        } else if (half_axis_input == '-') {
            bind.input.axis.axis_min = 0;
            bind.input.axis.axis_max = SDL_JOYSTICK_AXIS_MIN;
        } else {
            bind.input.axis.axis_min = SDL_JOYSTICK_AXIS_MIN;
            bind.input.axis.axis_max = SDL_JOYSTICK_AXIS_MAX;
        }
        if (invert_input) {
            int tmp = bind.input.axis.axis_min;
            bind.input.axis.axis_min = bind.input.axis.axis_max;
            bind.input.axis.axis_max = tmp;
        }
    } else if (szJoystickButton[0] == 'b' && SDL_isdigit((unsigned char)szJoystickButton[1])) {
        bind.inputType = SDL_CONTROLLER_BINDTYPE_BUTTON;
        bind.input.button = SDL_atoi(&szJoystickButton[1]);
    } else if (szJoystickButton[0] == 'h' && SDL_isdigit((unsigned char)szJoystickButton[1]) &&
               szJoystickButton[2] == '.' && SDL_isdigit((unsigned char)szJoystickButton[3])) {
        int hat = SDL_atoi(&szJoystickButton[1]);
        int mask = SDL_atoi(&szJoystickButton[3]);
        bind.inputType = SDL_CONTROLLER_BINDTYPE_HAT;
        bind.input.hat.hat = hat;
        bind.input.hat.hat_mask = mask;
    } else {
        SDL_SetError("Unexpected joystick element: %s", szJoystickButton);
        return;
    }

    ++gamecontroller->num_bindings;
    gamecontroller->bindings = (SDL_ExtendedGameControllerBind *)SDL_realloc(gamecontroller->bindings, gamecontroller->num_bindings * sizeof(*gamecontroller->bindings));
    if (!gamecontroller->bindings) {
        gamecontroller->num_bindings = 0;
        SDL_OutOfMemory();
        return;
    }
    gamecontroller->bindings[gamecontroller->num_bindings - 1] = bind;
}

/*
 * given a controller mapping string update our mapping object
 */
static void SDL_PrivateGameControllerParseControllerConfigString(SDL_GameController *gamecontroller, const char *pchString)
{
    char szGameButton[20];
    char szJoystickButton[20];
    SDL_bool bGameButton = SDL_TRUE;
    int i = 0;
    const char *pchPos = pchString;

    SDL_zeroa(szGameButton);
    SDL_zeroa(szJoystickButton);

    while (pchPos && *pchPos) {
        if (*pchPos == ':') {
            i = 0;
            bGameButton = SDL_FALSE;
        } else if (*pchPos == ' ') {

        } else if (*pchPos == ',') {
            i = 0;
            bGameButton = SDL_TRUE;
            SDL_PrivateGameControllerParseElement(gamecontroller, szGameButton, szJoystickButton);
            SDL_zeroa(szGameButton);
            SDL_zeroa(szJoystickButton);

        } else if (bGameButton) {
            if (i >= sizeof(szGameButton)) {
                szGameButton[sizeof(szGameButton) - 1] = '\0';
                SDL_SetError("Button name too large: %s", szGameButton);
                return;
            }
            szGameButton[i] = *pchPos;
            i++;
        } else {
            if (i >= sizeof(szJoystickButton)) {
                szJoystickButton[sizeof(szJoystickButton) - 1] = '\0';
                SDL_SetError("Joystick button name too large: %s", szJoystickButton);
                return;
            }
            szJoystickButton[i] = *pchPos;
            i++;
        }
        pchPos++;
    }

    /* No more values if the string was terminated by a comma. Don't report an error. */
    if (szGameButton[0] != '\0' || szJoystickButton[0] != '\0') {
        SDL_PrivateGameControllerParseElement(gamecontroller, szGameButton, szJoystickButton);
    }
}

static void SDL_UpdateGameControllerType(SDL_GameController *gamecontroller)
{
    char *type_string, *comma;

    SDL_AssertJoysticksLocked();

    gamecontroller->type = SDL_CONTROLLER_TYPE_UNKNOWN;

    type_string = SDL_strstr(gamecontroller->mapping->mapping, SDL_CONTROLLER_TYPE_FIELD);
    if (type_string) {
        type_string += SDL_CONTROLLER_TYPE_FIELD_SIZE;
        comma = SDL_strchr(type_string, ',');
        if (comma) {
            *comma = '\0';
            gamecontroller->type = SDL_GetGameControllerTypeFromString(type_string);
            *comma = ',';
        } else {
            gamecontroller->type = SDL_GetGameControllerTypeFromString(type_string);
        }
    }
    if (gamecontroller->type == SDL_CONTROLLER_TYPE_UNKNOWN) {
        gamecontroller->type = SDL_GetJoystickGameControllerTypeFromGUID(SDL_JoystickGetGUID(gamecontroller->joystick), SDL_JoystickName(gamecontroller->joystick));
    }
}

/*
 * Make a new button mapping struct
 */
static void SDL_PrivateLoadButtonMapping(SDL_GameController *gamecontroller, ControllerMapping_t *pControllerMapping)
{
    int i;

    SDL_AssertJoysticksLocked();

    gamecontroller->name = pControllerMapping->name;
    gamecontroller->num_bindings = 0;
    gamecontroller->mapping = pControllerMapping;
    if (gamecontroller->joystick->naxes != 0 && gamecontroller->last_match_axis) {
        SDL_memset(gamecontroller->last_match_axis, 0, gamecontroller->joystick->naxes * sizeof(*gamecontroller->last_match_axis));
    }

    SDL_PrivateGameControllerParseControllerConfigString(gamecontroller, pControllerMapping->mapping);

    SDL_UpdateGameControllerType(gamecontroller);

    /* Set the zero point for triggers */
    for (i = 0; i < gamecontroller->num_bindings; ++i) {
        SDL_ExtendedGameControllerBind *binding = &gamecontroller->bindings[i];
        if (binding->inputType == SDL_CONTROLLER_BINDTYPE_AXIS &&
            binding->outputType == SDL_CONTROLLER_BINDTYPE_AXIS &&
            (binding->output.axis.axis == SDL_CONTROLLER_AXIS_TRIGGERLEFT ||
             binding->output.axis.axis == SDL_CONTROLLER_AXIS_TRIGGERRIGHT)) {
            if (binding->input.axis.axis < gamecontroller->joystick->naxes) {
                gamecontroller->joystick->axes[binding->input.axis.axis].value =
                    gamecontroller->joystick->axes[binding->input.axis.axis].zero = (Sint16)binding->input.axis.axis_min;
            }
        }
    }
}

/*
 * grab the guid string from a mapping string
 */
static char *SDL_PrivateGetControllerGUIDFromMappingString(const char *pMapping)
{
    const char *pFirstComma = SDL_strchr(pMapping, ',');
    if (pFirstComma) {
        char *pchGUID = SDL_malloc(pFirstComma - pMapping + 1);
        if (!pchGUID) {
            SDL_OutOfMemory();
            return NULL;
        }
        SDL_memcpy(pchGUID, pMapping, pFirstComma - pMapping);
        pchGUID[pFirstComma - pMapping] = '\0';

        /* Convert old style GUIDs to the new style in 2.0.5 */
#if defined(__WIN32__) || defined(__WINGDK__)
        if (SDL_strlen(pchGUID) == 32 &&
            SDL_memcmp(&pchGUID[20], "504944564944", 12) == 0) {
            SDL_memcpy(&pchGUID[20], "000000000000", 12);
            SDL_memcpy(&pchGUID[16], &pchGUID[4], 4);
            SDL_memcpy(&pchGUID[8], &pchGUID[0], 4);
            SDL_memcpy(&pchGUID[0], "03000000", 8);
        }
#elif defined(__MACOSX__)
        if (SDL_strlen(pchGUID) == 32 &&
            SDL_memcmp(&pchGUID[4], "000000000000", 12) == 0 &&
            SDL_memcmp(&pchGUID[20], "000000000000", 12) == 0) {
            SDL_memcpy(&pchGUID[20], "000000000000", 12);
            SDL_memcpy(&pchGUID[8], &pchGUID[0], 4);
            SDL_memcpy(&pchGUID[0], "03000000", 8);
        }
#endif
        return pchGUID;
    }
    return NULL;
}

/*
 * grab the name string from a mapping string
 */
static char *SDL_PrivateGetControllerNameFromMappingString(const char *pMapping)
{
    const char *pFirstComma, *pSecondComma;
    char *pchName;

    pFirstComma = SDL_strchr(pMapping, ',');
    if (!pFirstComma) {
        return NULL;
    }

    pSecondComma = SDL_strchr(pFirstComma + 1, ',');
    if (!pSecondComma) {
        return NULL;
    }

    pchName = SDL_malloc(pSecondComma - pFirstComma);
    if (!pchName) {
        SDL_OutOfMemory();
        return NULL;
    }
    SDL_memcpy(pchName, pFirstComma + 1, pSecondComma - pFirstComma);
    pchName[pSecondComma - pFirstComma - 1] = 0;
    return pchName;
}

/*
 * grab the button mapping string from a mapping string
 */
static char *SDL_PrivateGetControllerMappingFromMappingString(const char *pMapping)
{
    const char *pFirstComma, *pSecondComma;

    pFirstComma = SDL_strchr(pMapping, ',');
    if (!pFirstComma) {
        return NULL;
    }

    pSecondComma = SDL_strchr(pFirstComma + 1, ',');
    if (!pSecondComma) {
        return NULL;
    }

    return SDL_strdup(pSecondComma + 1); /* mapping is everything after the 3rd comma */
}

/*
 * Helper function to refresh a mapping
 */
static void SDL_PrivateGameControllerRefreshMapping(ControllerMapping_t *pControllerMapping)
{
    SDL_GameController *controller;

    SDL_AssertJoysticksLocked();

    for (controller = SDL_gamecontrollers; controller; controller = controller->next) {
        if (controller->mapping == pControllerMapping) {
            SDL_PrivateLoadButtonMapping(controller, pControllerMapping);

            {
                SDL_Event event;
                event.type = SDL_CONTROLLERDEVICEREMAPPED;
                event.cdevice.which = controller->joystick->instance_id;
                SDL_PushEvent(&event);
            }
        }
    }
}

/*
 * Helper function to add a mapping for a guid
 */
static ControllerMapping_t *SDL_PrivateAddMappingForGUID(SDL_JoystickGUID jGUID, const char *mappingString, SDL_bool *existing, SDL_ControllerMappingPriority priority)
{
    char *pchName;
    char *pchMapping;
    ControllerMapping_t *pControllerMapping;
    Uint16 crc;

    SDL_AssertJoysticksLocked();

    pchName = SDL_PrivateGetControllerNameFromMappingString(mappingString);
    if (!pchName) {
        SDL_SetError("Couldn't parse name from %s", mappingString);
        return NULL;
    }

    pchMapping = SDL_PrivateGetControllerMappingFromMappingString(mappingString);
    if (!pchMapping) {
        SDL_free(pchName);
        SDL_SetError("Couldn't parse %s", mappingString);
        return NULL;
    }

    /* Fix up the GUID and the mapping with the CRC, if needed */
    SDL_GetJoystickGUIDInfo(jGUID, NULL, NULL, NULL, &crc);
    if (crc) {
        /* Make sure the mapping has the CRC */
        char *new_mapping;
        char *crc_end = "";
        char *crc_string = SDL_strstr(pchMapping, SDL_CONTROLLER_CRC_FIELD);
        if (crc_string) {
            crc_end = SDL_strchr(crc_string, ',');
            if (crc_end) {
                ++crc_end;
            } else {
                crc_end = "";
            }
            *crc_string = '\0';
        }

        if (SDL_asprintf(&new_mapping, "%s%s%.4x,%s", pchMapping, SDL_CONTROLLER_CRC_FIELD, crc, crc_end) >= 0) {
            SDL_free(pchMapping);
            pchMapping = new_mapping;
        }
    } else {
        /* Make sure the GUID has the CRC, for matching purposes */
        char *crc_string = SDL_strstr(pchMapping, SDL_CONTROLLER_CRC_FIELD);
        if (crc_string) {
            crc = (Uint16)SDL_strtol(crc_string + SDL_CONTROLLER_CRC_FIELD_SIZE, NULL, 16);
            if (crc) {
                SDL_SetJoystickGUIDCRC(&jGUID, crc);
            }
        }
    }

    pControllerMapping = SDL_PrivateGetControllerMappingForGUID(jGUID, SDL_TRUE);
    if (pControllerMapping) {
        /* Only overwrite the mapping if the priority is the same or higher. */
        if (pControllerMapping->priority <= priority) {
            /* Update existing mapping */
            SDL_free(pControllerMapping->name);
            pControllerMapping->name = pchName;
            SDL_free(pControllerMapping->mapping);
            pControllerMapping->mapping = pchMapping;
            pControllerMapping->priority = priority;
            /* refresh open controllers */
            SDL_PrivateGameControllerRefreshMapping(pControllerMapping);
        } else {
            SDL_free(pchName);
            SDL_free(pchMapping);
        }
        *existing = SDL_TRUE;
    } else {
        pControllerMapping = SDL_malloc(sizeof(*pControllerMapping));
        if (!pControllerMapping) {
            SDL_free(pchName);
            SDL_free(pchMapping);
            SDL_OutOfMemory();
            return NULL;
        }
        /* Clear the CRC, we've already added it to the mapping */
        if (crc) {
            SDL_SetJoystickGUIDCRC(&jGUID, 0);
        }
        pControllerMapping->guid = jGUID;
        pControllerMapping->name = pchName;
        pControllerMapping->mapping = pchMapping;
        pControllerMapping->next = NULL;
        pControllerMapping->priority = priority;

        if (s_pSupportedControllers) {
            /* Add the mapping to the end of the list */
            ControllerMapping_t *pCurrMapping, *pPrevMapping;

            for (pPrevMapping = s_pSupportedControllers, pCurrMapping = pPrevMapping->next;
                 pCurrMapping;
                 pPrevMapping = pCurrMapping, pCurrMapping = pCurrMapping->next) {
                /* continue; */
            }
            pPrevMapping->next = pControllerMapping;
        } else {
            s_pSupportedControllers = pControllerMapping;
        }
        *existing = SDL_FALSE;
    }
    return pControllerMapping;
}

/*
 * Helper function to determine pre-calculated offset to certain joystick mappings
 */
static ControllerMapping_t *SDL_PrivateGetControllerMappingForNameAndGUID(const char *name, SDL_JoystickGUID guid)
{
    ControllerMapping_t *mapping;

    SDL_AssertJoysticksLocked();

    mapping = SDL_PrivateGetControllerMappingForGUID(guid, SDL_FALSE);
#ifdef __LINUX__
    if (!mapping && name) {
        if (SDL_strstr(name, "Xbox 360 Wireless Receiver")) {
            /* The Linux driver xpad.c maps the wireless dpad to buttons */
            SDL_bool existing;
            mapping = SDL_PrivateAddMappingForGUID(guid,
                                                   "none,X360 Wireless Controller,a:b0,b:b1,back:b6,dpdown:b14,dpleft:b11,dpright:b12,dpup:b13,guide:b8,leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,",
                                                   &existing, SDL_CONTROLLER_MAPPING_PRIORITY_DEFAULT);
        }
    }
#endif /* __LINUX__ */

    return mapping;
}

static void SDL_PrivateAppendToMappingString(char *mapping_string,
                                             size_t mapping_string_len,
                                             const char *input_name,
                                             SDL_InputMapping *mapping)
{
    char buffer[16];
    if (mapping->kind == EMappingKind_None) {
        return;
    }

    SDL_strlcat(mapping_string, input_name, mapping_string_len);
    SDL_strlcat(mapping_string, ":", mapping_string_len);
    switch (mapping->kind) {
    case EMappingKind_Button:
        (void)SDL_snprintf(buffer, sizeof(buffer), "b%i", mapping->target);
        break;
    case EMappingKind_Axis:
        (void)SDL_snprintf(buffer, sizeof(buffer), "%sa%i%s",
            mapping->half_axis_positive ? "+" :
            mapping->half_axis_negative ? "-" : "",
            mapping->target,
            mapping->axis_reversed ? "~" : "");
        break;
    case EMappingKind_Hat:
        (void)SDL_snprintf(buffer, sizeof(buffer), "h%i.%i", mapping->target >> 4, mapping->target & 0x0F);
        break;
    default:
        SDL_assert(SDL_FALSE);
    }

    SDL_strlcat(mapping_string, buffer, mapping_string_len);
    SDL_strlcat(mapping_string, ",", mapping_string_len);
}

static ControllerMapping_t *SDL_PrivateGenerateAutomaticControllerMapping(const char *name,
                                                                          SDL_JoystickGUID guid,
                                                                          SDL_GamepadMapping *raw_map)
{
    SDL_bool existing;
    char name_string[128];
    char mapping[1024];

    /* Remove any commas in the name */
    SDL_strlcpy(name_string, name, sizeof(name_string));
    {
        char *spot;
        for (spot = name_string; *spot; ++spot) {
            if (*spot == ',') {
                *spot = ' ';
            }
        }
    }
    (void)SDL_snprintf(mapping, sizeof(mapping), "none,%s,", name_string);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "a", &raw_map->a);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "b", &raw_map->b);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "x", &raw_map->x);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "y", &raw_map->y);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "back", &raw_map->back);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "guide", &raw_map->guide);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "start", &raw_map->start);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "leftstick", &raw_map->leftstick);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "rightstick", &raw_map->rightstick);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "leftshoulder", &raw_map->leftshoulder);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "rightshoulder", &raw_map->rightshoulder);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "dpup", &raw_map->dpup);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "dpdown", &raw_map->dpdown);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "dpleft", &raw_map->dpleft);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "dpright", &raw_map->dpright);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "misc1", &raw_map->misc1);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "paddle1", &raw_map->paddle1);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "paddle2", &raw_map->paddle2);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "paddle3", &raw_map->paddle3);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "paddle4", &raw_map->paddle4);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "leftx", &raw_map->leftx);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "lefty", &raw_map->lefty);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "rightx", &raw_map->rightx);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "righty", &raw_map->righty);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "lefttrigger", &raw_map->lefttrigger);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "righttrigger", &raw_map->righttrigger);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "touchpad", &raw_map->touchpad);

    return SDL_PrivateAddMappingForGUID(guid, mapping, &existing, SDL_CONTROLLER_MAPPING_PRIORITY_DEFAULT);
}

static ControllerMapping_t *SDL_PrivateGetControllerMapping(int device_index)
{
    const char *name;
    SDL_JoystickGUID guid;
    ControllerMapping_t *mapping;

    SDL_AssertJoysticksLocked();

    if ((device_index < 0) || (device_index >= SDL_NumJoysticks())) {
        SDL_SetError("There are %d joysticks available", SDL_NumJoysticks());
        return NULL;
    }

    name = SDL_JoystickNameForIndex(device_index);
    guid = SDL_JoystickGetDeviceGUID(device_index);
    mapping = SDL_PrivateGetControllerMappingForNameAndGUID(name, guid);
    if (!mapping) {
        SDL_GamepadMapping raw_map;

        SDL_zero(raw_map);
        if (SDL_PrivateJoystickGetAutoGamepadMapping(device_index, &raw_map)) {
            mapping = SDL_PrivateGenerateAutomaticControllerMapping(name, guid, &raw_map);
        }
    }

    if (!mapping) {
        mapping = s_pDefaultMapping;
    }
    return mapping;
}

/*
 * Add or update an entry into the Mappings Database
 */
int SDL_GameControllerAddMappingsFromRW(SDL_RWops *rw, int freerw)
{
    const char *platform = SDL_GetPlatform();
    int controllers = 0;
    char *buf, *line, *line_end, *tmp, *comma, line_platform[64];
    size_t db_size, platform_len;

    if (!rw) {
        return SDL_SetError("Invalid RWops");
    }
    db_size = (size_t)SDL_RWsize(rw);

    buf = (char *)SDL_malloc(db_size + 1);
    if (!buf) {
        if (freerw) {
            SDL_RWclose(rw);
        }
        return SDL_SetError("Could not allocate space to read DB into memory");
    }

    if (SDL_RWread(rw, buf, db_size, 1) != 1) {
        if (freerw) {
            SDL_RWclose(rw);
        }
        SDL_free(buf);
        return SDL_SetError("Could not read DB");
    }

    if (freerw) {
        SDL_RWclose(rw);
    }

    buf[db_size] = '\0';
    line = buf;

    while (line < buf + db_size) {
        line_end = SDL_strchr(line, '\n');
        if (line_end) {
            *line_end = '\0';
        } else {
            line_end = buf + db_size;
        }

        /* Extract and verify the platform */
        tmp = SDL_strstr(line, SDL_CONTROLLER_PLATFORM_FIELD);
        if (tmp) {
            tmp += SDL_CONTROLLER_PLATFORM_FIELD_SIZE;
            comma = SDL_strchr(tmp, ',');
            if (comma) {
                platform_len = comma - tmp + 1;
                if (platform_len + 1 < SDL_arraysize(line_platform)) {
                    SDL_strlcpy(line_platform, tmp, platform_len);
                    if (SDL_strncasecmp(line_platform, platform, platform_len) == 0 &&
                        SDL_GameControllerAddMapping(line) > 0) {
                        controllers++;
                    }
                }
            }
        }

        line = line_end + 1;
    }

    SDL_free(buf);
    return controllers;
}

/*
 * Add or update an entry into the Mappings Database with a priority
 */
static int SDL_PrivateGameControllerAddMapping(const char *mappingString, SDL_ControllerMappingPriority priority)
{
    char *pchGUID;
    SDL_JoystickGUID jGUID;
    SDL_bool is_default_mapping = SDL_FALSE;
    SDL_bool is_xinput_mapping = SDL_FALSE;
    SDL_bool existing = SDL_FALSE;
    ControllerMapping_t *pControllerMapping;

    SDL_AssertJoysticksLocked();

    if (!mappingString) {
        return SDL_InvalidParamError("mappingString");
    }

    { /* Extract and verify the hint field */
        const char *tmp;

        tmp = SDL_strstr(mappingString, SDL_CONTROLLER_HINT_FIELD);
        if (tmp) {
            SDL_bool default_value, value, negate;
            int len;
            char hint[128];

            tmp += SDL_CONTROLLER_HINT_FIELD_SIZE;

            if (*tmp == '!') {
                negate = SDL_TRUE;
                ++tmp;
            } else {
                negate = SDL_FALSE;
            }

            len = 0;
            while (*tmp && *tmp != ',' && *tmp != ':' && len < (sizeof(hint) - 1)) {
                hint[len++] = *tmp++;
            }
            hint[len] = '\0';

            if (tmp[0] == ':' && tmp[1] == '=') {
                tmp += 2;
                default_value = SDL_atoi(tmp);
            } else {
                default_value = SDL_FALSE;
            }

            value = SDL_GetHintBoolean(hint, default_value);
            if (negate) {
                value = !value;
            }
            if (!value) {
                return 0;
            }
        }
    }

#ifdef ANDROID
    { /* Extract and verify the SDK version */
        const char *tmp;

        tmp = SDL_strstr(mappingString, SDL_CONTROLLER_SDKGE_FIELD);
        if (tmp) {
            tmp += SDL_CONTROLLER_SDKGE_FIELD_SIZE;
            if (!(SDL_GetAndroidSDKVersion() >= SDL_atoi(tmp))) {
                return SDL_SetError("SDK version %d < minimum version %d", SDL_GetAndroidSDKVersion(), SDL_atoi(tmp));
            }
        }
        tmp = SDL_strstr(mappingString, SDL_CONTROLLER_SDKLE_FIELD);
        if (tmp) {
            tmp += SDL_CONTROLLER_SDKLE_FIELD_SIZE;
            if (!(SDL_GetAndroidSDKVersion() <= SDL_atoi(tmp))) {
                return SDL_SetError("SDK version %d > maximum version %d", SDL_GetAndroidSDKVersion(), SDL_atoi(tmp));
            }
        }
    }
#endif

    pchGUID = SDL_PrivateGetControllerGUIDFromMappingString(mappingString);
    if (!pchGUID) {
        return SDL_SetError("Couldn't parse GUID from %s", mappingString);
    }
    if (!SDL_strcasecmp(pchGUID, "default")) {
        is_default_mapping = SDL_TRUE;
    } else if (!SDL_strcasecmp(pchGUID, "xinput")) {
        is_xinput_mapping = SDL_TRUE;
    }
    jGUID = SDL_JoystickGetGUIDFromString(pchGUID);
    SDL_free(pchGUID);

    pControllerMapping = SDL_PrivateAddMappingForGUID(jGUID, mappingString, &existing, priority);
    if (!pControllerMapping) {
        return -1;
    }

    if (existing) {
        return 0;
    } else {
        if (is_default_mapping) {
            s_pDefaultMapping = pControllerMapping;
        } else if (is_xinput_mapping) {
            s_pXInputMapping = pControllerMapping;
        }
        return 1;
    }
}

/*
 * Add or update an entry into the Mappings Database
 */
int SDL_GameControllerAddMapping(const char *mappingString)
{
    int retval;

    SDL_LockJoysticks();
    {
        retval = SDL_PrivateGameControllerAddMapping(mappingString, SDL_CONTROLLER_MAPPING_PRIORITY_API);
    }
    SDL_UnlockJoysticks();

    return retval;
}

/*
 *  Get the number of mappings installed
 */
int SDL_GameControllerNumMappings(void)
{
    int num_mappings = 0;

    SDL_LockJoysticks();
    {
        ControllerMapping_t *mapping;

        for (mapping = s_pSupportedControllers; mapping; mapping = mapping->next) {
            if (SDL_memcmp(&mapping->guid, &s_zeroGUID, sizeof(mapping->guid)) == 0) {
                continue;
            }
            ++num_mappings;
        }
    }
    SDL_UnlockJoysticks();

    return num_mappings;
}

/*
 * Create a mapping string for a mapping
 */
static char *CreateMappingString(ControllerMapping_t *mapping, SDL_JoystickGUID guid)
{
    char *pMappingString, *pPlatformString;
    char pchGUID[33];
    size_t needed;
    const char *platform = SDL_GetPlatform();

    SDL_AssertJoysticksLocked();

    SDL_JoystickGetGUIDString(guid, pchGUID, sizeof(pchGUID));

    /* allocate enough memory for GUID + ',' + name + ',' + mapping + \0 */
    needed = SDL_strlen(pchGUID) + 1 + SDL_strlen(mapping->name) + 1 + SDL_strlen(mapping->mapping) + 1;

    if (!SDL_strstr(mapping->mapping, SDL_CONTROLLER_PLATFORM_FIELD)) {
        /* add memory for ',' + platform:PLATFORM */
        if (mapping->mapping[SDL_strlen(mapping->mapping) - 1] != ',') {
            needed += 1;
        }
        needed += SDL_CONTROLLER_PLATFORM_FIELD_SIZE + SDL_strlen(platform);
    }

    pMappingString = SDL_malloc(needed);
    if (!pMappingString) {
        SDL_OutOfMemory();
        return NULL;
    }

    (void)SDL_snprintf(pMappingString, needed, "%s,%s,%s", pchGUID, mapping->name, mapping->mapping);

    if (!SDL_strstr(mapping->mapping, SDL_CONTROLLER_PLATFORM_FIELD)) {
        if (mapping->mapping[SDL_strlen(mapping->mapping) - 1] != ',') {
            SDL_strlcat(pMappingString, ",", needed);
        }
        SDL_strlcat(pMappingString, SDL_CONTROLLER_PLATFORM_FIELD, needed);
        SDL_strlcat(pMappingString, platform, needed);
    }

    /* Make sure multiple platform strings haven't made their way into the mapping */
    pPlatformString = SDL_strstr(pMappingString, SDL_CONTROLLER_PLATFORM_FIELD);
    if (pPlatformString) {
        pPlatformString = SDL_strstr(pPlatformString + 1, SDL_CONTROLLER_PLATFORM_FIELD);
        if (pPlatformString) {
            *pPlatformString = '\0';
        }
    }
    return pMappingString;
}

/*
 *  Get the mapping at a particular index.
 */
char *SDL_GameControllerMappingForIndex(int mapping_index)
{
    char *retval = NULL;

    SDL_LockJoysticks();
    {
        ControllerMapping_t *mapping;

        for (mapping = s_pSupportedControllers; mapping; mapping = mapping->next) {
            if (SDL_memcmp(&mapping->guid, &s_zeroGUID, sizeof(mapping->guid)) == 0) {
                continue;
            }
            if (mapping_index == 0) {
                retval = CreateMappingString(mapping, mapping->guid);
                break;
            }
            --mapping_index;
        }
    }
    SDL_UnlockJoysticks();

    if (!retval) {
        SDL_SetError("Mapping not available");
    }
    return retval;
}

/*
 * Get the mapping string for this GUID
 */
char *SDL_GameControllerMappingForGUID(SDL_JoystickGUID guid)
{
    char *retval;

    SDL_LockJoysticks();
    {
        ControllerMapping_t *mapping = SDL_PrivateGetControllerMappingForGUID(guid, SDL_FALSE);
        if (mapping) {
            retval = CreateMappingString(mapping, guid);
        } else {
            SDL_SetError("Mapping not available");
            retval = NULL;
        }
    }
    SDL_UnlockJoysticks();

    return retval;
}

/*
 * Get the mapping string for this device
 */
char *SDL_GameControllerMapping(SDL_GameController *gamecontroller)
{
    char *retval;

    SDL_LockJoysticks();
    {
        CHECK_GAMECONTROLLER_MAGIC(gamecontroller, NULL);

        retval = CreateMappingString(gamecontroller->mapping, gamecontroller->joystick->guid);
    }
    SDL_UnlockJoysticks();

    return retval;
}

static void SDL_GameControllerLoadHints(void)
{
    const char *hint = SDL_GetHint(SDL_HINT_GAMECONTROLLERCONFIG);
    if (hint && hint[0]) {
        size_t nchHints = SDL_strlen(hint);
        char *pUserMappings = SDL_malloc(nchHints + 1);
        char *pTempMappings = pUserMappings;
        SDL_memcpy(pUserMappings, hint, nchHints);
        pUserMappings[nchHints] = '\0';
        while (pUserMappings) {
            char *pchNewLine = NULL;

            pchNewLine = SDL_strchr(pUserMappings, '\n');
            if (pchNewLine) {
                *pchNewLine = '\0';
            }

            SDL_PrivateGameControllerAddMapping(pUserMappings, SDL_CONTROLLER_MAPPING_PRIORITY_USER);

            if (pchNewLine) {
                pUserMappings = pchNewLine + 1;
            } else {
                pUserMappings = NULL;
            }
        }
        SDL_free(pTempMappings);
    }
}

/*
 * Fill the given buffer with the expected controller mapping filepath.
 * Usually this will just be SDL_HINT_GAMECONTROLLERCONFIG_FILE, but for
 * Android, we want to get the internal storage path.
 */
static SDL_bool SDL_GetControllerMappingFilePath(char *path, size_t size)
{
    const char *hint = SDL_GetHint(SDL_HINT_GAMECONTROLLERCONFIG_FILE);
    if (hint && *hint) {
        return SDL_strlcpy(path, hint, size) < size;
    }

#if defined(__ANDROID__)
    return SDL_snprintf(path, size, "%s/controller_map.txt", SDL_AndroidGetInternalStoragePath()) < size;
#else
    return SDL_FALSE;
#endif
}

/*
 * Initialize the game controller system, mostly load our DB of controller config mappings
 */
int SDL_GameControllerInitMappings(void)
{
    char szControllerMapPath[1024];
    int i = 0;
    const char *pMappingString = NULL;

    SDL_AssertJoysticksLocked();

    pMappingString = s_ControllerMappings[i];
    while (pMappingString) {
        SDL_PrivateGameControllerAddMapping(pMappingString, SDL_CONTROLLER_MAPPING_PRIORITY_DEFAULT);

        i++;
        pMappingString = s_ControllerMappings[i];
    }

    if (SDL_GetControllerMappingFilePath(szControllerMapPath, sizeof(szControllerMapPath))) {
        SDL_GameControllerAddMappingsFromFile(szControllerMapPath);
    }

    /* load in any user supplied config */
    SDL_GameControllerLoadHints();

    SDL_LoadVIDPIDList(&SDL_allowed_controllers);
    SDL_LoadVIDPIDList(&SDL_ignored_controllers);

    return 0;
}

int SDL_GameControllerInit(void)
{
    int i;

    /* watch for joy events and fire controller ones if needed */
    SDL_AddEventWatch(SDL_GameControllerEventWatcher, NULL);

    /* Send added events for controllers currently attached */
    for (i = 0; i < SDL_NumJoysticks(); ++i) {
        if (SDL_IsGameController(i)) {
            SDL_Event deviceevent;
            deviceevent.type = SDL_CONTROLLERDEVICEADDED;
            deviceevent.cdevice.which = i;
            SDL_PushEvent(&deviceevent);
        }
    }

    return 0;
}

/*
 * Get the implementation dependent name of a controller
 */
const char *SDL_GameControllerNameForIndex(int joystick_index)
{
    const char *retval = NULL;

    SDL_LockJoysticks();
    {
        ControllerMapping_t *mapping = SDL_PrivateGetControllerMapping(joystick_index);
        if (mapping) {
            if (SDL_strcmp(mapping->name, "*") == 0) {
                retval = SDL_JoystickNameForIndex(joystick_index);
            } else {
                retval = mapping->name;
            }
        }
    }
    SDL_UnlockJoysticks();

    return retval;
}

/*
 * Get the implementation dependent path of a controller
 */
const char *SDL_GameControllerPathForIndex(int joystick_index)
{
    const char *retval = NULL;

    SDL_LockJoysticks();
    {
        ControllerMapping_t *mapping = SDL_PrivateGetControllerMapping(joystick_index);
        if (mapping) {
            retval = SDL_JoystickPathForIndex(joystick_index);
        }
    }
    SDL_UnlockJoysticks();

    return retval;
}

/**
 *  Get the type of a game controller.
 */
SDL_GameControllerType SDL_GameControllerTypeForIndex(int joystick_index)
{
    SDL_JoystickGUID joystick_guid = SDL_JoystickGetDeviceGUID(joystick_index);
    const char *mapping = SDL_GameControllerMappingForGUID(joystick_guid);
    char *type_string, *comma;
    SDL_GameControllerType type;
    if (mapping) {
        type_string = SDL_strstr(mapping, SDL_CONTROLLER_TYPE_FIELD);
        if (type_string) {
            type_string += SDL_CONTROLLER_TYPE_FIELD_SIZE;
            comma = SDL_strchr(type_string, ',');
            if (comma) {
                *comma = '\0';
                type = SDL_GetGameControllerTypeFromString(type_string);
                *comma = ',';
            } else {
                type = SDL_GetGameControllerTypeFromString(type_string);
            }
            return type;
        }
    }
    return SDL_GetJoystickGameControllerTypeFromGUID(joystick_guid, SDL_JoystickNameForIndex(joystick_index));
}

/**
 *  Get the mapping of a game controller.
 *  This can be called before any controllers are opened.
 *  If no mapping can be found, this function returns NULL.
 */
char *SDL_GameControllerMappingForDeviceIndex(int joystick_index)
{
    char *retval = NULL;

    SDL_LockJoysticks();
    {
        ControllerMapping_t *mapping = SDL_PrivateGetControllerMapping(joystick_index);
        if (mapping) {
            SDL_JoystickGUID guid;
            char pchGUID[33];
            size_t needed;
            guid = SDL_JoystickGetDeviceGUID(joystick_index);
            SDL_JoystickGetGUIDString(guid, pchGUID, sizeof(pchGUID));
            /* allocate enough memory for GUID + ',' + name + ',' + mapping + \0 */
            needed = SDL_strlen(pchGUID) + 1 + SDL_strlen(mapping->name) + 1 + SDL_strlen(mapping->mapping) + 1;
            retval = (char *)SDL_malloc(needed);
            if (retval) {
                (void)SDL_snprintf(retval, needed, "%s,%s,%s", pchGUID, mapping->name, mapping->mapping);
            } else {
                SDL_OutOfMemory();
            }
        }
    }
    SDL_UnlockJoysticks();
    return retval;
}

/*
 * Return 1 if the joystick with this name and GUID is a supported controller
 */
SDL_bool SDL_IsGameControllerNameAndGUID(const char *name, SDL_JoystickGUID guid)
{
    SDL_bool retval;

    SDL_LockJoysticks();
    {
        if (s_pDefaultMapping || SDL_PrivateGetControllerMappingForNameAndGUID(name, guid) != NULL) {
            retval = SDL_TRUE;
        } else {
            retval = SDL_FALSE;
        }
    }
    SDL_UnlockJoysticks();

    return retval;
}

/*
 * Return 1 if the joystick at this device index is a supported controller
 */
SDL_bool SDL_IsGameController(int joystick_index)
{
    SDL_bool retval;

    SDL_LockJoysticks();
    {
        if (SDL_PrivateGetControllerMapping(joystick_index) != NULL) {
            retval = SDL_TRUE;
        } else {
            retval = SDL_FALSE;
        }
    }
    SDL_UnlockJoysticks();

    return retval;
}

#if defined(__LINUX__)
static SDL_bool SDL_endswith(const char *string, const char *suffix)
{
    size_t string_length = string ? SDL_strlen(string) : 0;
    size_t suffix_length = suffix ? SDL_strlen(suffix) : 0;

    if (suffix_length > 0 && suffix_length <= string_length) {
        if (SDL_memcmp(string + string_length - suffix_length, suffix, suffix_length) == 0) {
            return SDL_TRUE;
        }
    }
    return SDL_FALSE;
}
#endif

/*
 * Return 1 if the game controller should be ignored by SDL
 */
SDL_bool SDL_ShouldIgnoreGameController(const char *name, SDL_JoystickGUID guid)
{
    Uint16 vendor;
    Uint16 product;
    Uint16 version;

#if defined(__LINUX__)
    if (SDL_endswith(name, " Motion Sensors")) {
        /* Don't treat the PS3 and PS4 motion controls as a separate game controller */
        return SDL_TRUE;
    }
    if (SDL_strncmp(name, "Nintendo ", 9) == 0 && SDL_strstr(name, " IMU") != NULL) {
        /* Don't treat the Nintendo IMU as a separate game controller */
        return SDL_TRUE;
    }
    if (SDL_endswith(name, " Accelerometer") ||
        SDL_endswith(name, " IR") ||
        SDL_endswith(name, " Motion Plus") ||
        SDL_endswith(name, " Nunchuk")) {
        /* Don't treat the Wii extension controls as a separate game controller */
        return SDL_TRUE;
    }
#endif

    if (name && SDL_strcmp(name, "uinput-fpc") == 0) {
        /* The Google Pixel fingerprint sensor reports itself as a joystick */
        return SDL_TRUE;
    }

    SDL_GetJoystickGUIDInfo(guid, &vendor, &product, &version, NULL);

    if (SDL_IsJoystickSteamVirtualGamepad(vendor, product, version)) {
        return !SDL_GetHintBoolean("SDL_GAMECONTROLLER_ALLOW_STEAM_VIRTUAL_GAMEPAD", SDL_FALSE);
    }

    if (SDL_allowed_controllers.num_included_entries > 0) {
        if (SDL_VIDPIDInList(vendor, product, &SDL_allowed_controllers)) {
            return SDL_FALSE;
        }
        return SDL_TRUE;
    } else {
        if (SDL_VIDPIDInList(vendor, product, &SDL_ignored_controllers)) {
            return SDL_TRUE;
        }
        return SDL_FALSE;
    }
}

/*
 * Open a controller for use - the index passed as an argument refers to
 * the N'th controller on the system.  This index is the value which will
 * identify this controller in future controller events.
 *
 * This function returns a controller identifier, or NULL if an error occurred.
 */
SDL_GameController *SDL_GameControllerOpen(int joystick_index)
{
    SDL_JoystickID instance_id;
    SDL_GameController *gamecontroller;
    SDL_GameController *gamecontrollerlist;
    ControllerMapping_t *pSupportedController = NULL;

    SDL_LockJoysticks();

    gamecontrollerlist = SDL_gamecontrollers;
    /* If the controller is already open, return it */
    instance_id = SDL_JoystickGetDeviceInstanceID(joystick_index);
    while (gamecontrollerlist) {
        if (instance_id == gamecontrollerlist->joystick->instance_id) {
            gamecontroller = gamecontrollerlist;
            ++gamecontroller->ref_count;
            SDL_UnlockJoysticks();
            return gamecontroller;
        }
        gamecontrollerlist = gamecontrollerlist->next;
    }

    /* Find a controller mapping */
    pSupportedController = SDL_PrivateGetControllerMapping(joystick_index);
    if (!pSupportedController) {
        SDL_SetError("Couldn't find mapping for device (%d)", joystick_index);
        SDL_UnlockJoysticks();
        return NULL;
    }

    /* Create and initialize the controller */
    gamecontroller = (SDL_GameController *)SDL_calloc(1, sizeof(*gamecontroller));
    if (!gamecontroller) {
        SDL_OutOfMemory();
        SDL_UnlockJoysticks();
        return NULL;
    }
    gamecontroller->magic = &gamecontroller_magic;

    gamecontroller->joystick = SDL_JoystickOpen(joystick_index);
    if (!gamecontroller->joystick) {
        SDL_free(gamecontroller);
        SDL_UnlockJoysticks();
        return NULL;
    }

    if (gamecontroller->joystick->naxes) {
        gamecontroller->last_match_axis = (SDL_ExtendedGameControllerBind **)SDL_calloc(gamecontroller->joystick->naxes, sizeof(*gamecontroller->last_match_axis));
        if (!gamecontroller->last_match_axis) {
            SDL_OutOfMemory();
            SDL_JoystickClose(gamecontroller->joystick);
            SDL_free(gamecontroller);
            SDL_UnlockJoysticks();
            return NULL;
        }
    }
    if (gamecontroller->joystick->nhats) {
        gamecontroller->last_hat_mask = (Uint8 *)SDL_calloc(gamecontroller->joystick->nhats, sizeof(*gamecontroller->last_hat_mask));
        if (!gamecontroller->last_hat_mask) {
            SDL_OutOfMemory();
            SDL_JoystickClose(gamecontroller->joystick);
            SDL_free(gamecontroller->last_match_axis);
            SDL_free(gamecontroller);
            SDL_UnlockJoysticks();
            return NULL;
        }
    }

    SDL_PrivateLoadButtonMapping(gamecontroller, pSupportedController);

    /* Add the controller to list */
    ++gamecontroller->ref_count;
    /* Link the controller in the list */
    gamecontroller->next = SDL_gamecontrollers;
    SDL_gamecontrollers = gamecontroller;

    SDL_UnlockJoysticks();

    return gamecontroller;
}

/*
 * Manually pump for controller updates.
 */
void SDL_GameControllerUpdate(void)
{
    /* Just for API completeness; the joystick API does all the work. */
    SDL_JoystickUpdate();
}

/**
 *  Return whether a game controller has a given axis
 */
SDL_bool SDL_GameControllerHasAxis(SDL_GameController *gamecontroller, SDL_GameControllerAxis axis)
{
    SDL_GameControllerButtonBind bind;

    SDL_LockJoysticks();
    {
        CHECK_GAMECONTROLLER_MAGIC(gamecontroller, SDL_FALSE);

        bind = SDL_GameControllerGetBindForAxis(gamecontroller, axis);
    }
    SDL_UnlockJoysticks();

    return (bind.bindType != SDL_CONTROLLER_BINDTYPE_NONE) ? SDL_TRUE : SDL_FALSE;
}

/*
 * Get the current state of an axis control on a controller
 */
Sint16 SDL_GameControllerGetAxis(SDL_GameController *gamecontroller, SDL_GameControllerAxis axis)
{
    Sint16 retval = 0;

    SDL_LockJoysticks();
    {
        int i;

        CHECK_GAMECONTROLLER_MAGIC(gamecontroller, 0);

        for (i = 0; i < gamecontroller->num_bindings; ++i) {
            SDL_ExtendedGameControllerBind *binding = &gamecontroller->bindings[i];
            if (binding->outputType == SDL_CONTROLLER_BINDTYPE_AXIS && binding->output.axis.axis == axis) {
                int value = 0;
                SDL_bool valid_input_range;
                SDL_bool valid_output_range;

                if (binding->inputType == SDL_CONTROLLER_BINDTYPE_AXIS) {
                    value = SDL_JoystickGetAxis(gamecontroller->joystick, binding->input.axis.axis);
                    if (binding->input.axis.axis_min < binding->input.axis.axis_max) {
                        valid_input_range = (value >= binding->input.axis.axis_min && value <= binding->input.axis.axis_max);
                    } else {
                        valid_input_range = (value >= binding->input.axis.axis_max && value <= binding->input.axis.axis_min);
                    }
                    if (valid_input_range) {
                        if (binding->input.axis.axis_min != binding->output.axis.axis_min || binding->input.axis.axis_max != binding->output.axis.axis_max) {
                            float normalized_value = (float)(value - binding->input.axis.axis_min) / (binding->input.axis.axis_max - binding->input.axis.axis_min);
                            value = binding->output.axis.axis_min + (int)(normalized_value * (binding->output.axis.axis_max - binding->output.axis.axis_min));
                        }
                    } else {
                        value = 0;
                    }
                } else if (binding->inputType == SDL_CONTROLLER_BINDTYPE_BUTTON) {
                    value = SDL_JoystickGetButton(gamecontroller->joystick, binding->input.button);
                    if (value == SDL_PRESSED) {
                        value = binding->output.axis.axis_max;
                    }
                } else if (binding->inputType == SDL_CONTROLLER_BINDTYPE_HAT) {
                    int hat_mask = SDL_JoystickGetHat(gamecontroller->joystick, binding->input.hat.hat);
                    if (hat_mask & binding->input.hat.hat_mask) {
                        value = binding->output.axis.axis_max;
                    }
                }

                if (binding->output.axis.axis_min < binding->output.axis.axis_max) {
                    valid_output_range = (value >= binding->output.axis.axis_min && value <= binding->output.axis.axis_max);
                } else {
                    valid_output_range = (value >= binding->output.axis.axis_max && value <= binding->output.axis.axis_min);
                }
                /* If the value is zero, there might be another binding that makes it non-zero */
                if (value != 0 && valid_output_range) {
                    retval = (Sint16)value;
                    break;
                }
            }
        }
    }
    SDL_UnlockJoysticks();

    return retval;
}

/**
 *  Return whether a game controller has a given button
 */
SDL_bool SDL_GameControllerHasButton(SDL_GameController *gamecontroller, SDL_GameControllerButton button)
{
    SDL_GameControllerButtonBind bind;

    SDL_LockJoysticks();
    {
        CHECK_GAMECONTROLLER_MAGIC(gamecontroller, SDL_FALSE);

        bind = SDL_GameControllerGetBindForButton(gamecontroller, button);
    }
    SDL_UnlockJoysticks();

    return (bind.bindType != SDL_CONTROLLER_BINDTYPE_NONE) ? SDL_TRUE : SDL_FALSE;
}

/*
 * Get the current state of a button on a controller
 */
Uint8 SDL_GameControllerGetButton(SDL_GameController *gamecontroller, SDL_GameControllerButton button)
{
    Uint8 retval = SDL_RELEASED;

    SDL_LockJoysticks();
    {
        int i;

        CHECK_GAMECONTROLLER_MAGIC(gamecontroller, 0);

        for (i = 0; i < gamecontroller->num_bindings; ++i) {
            SDL_ExtendedGameControllerBind *binding = &gamecontroller->bindings[i];
            if (binding->outputType == SDL_CONTROLLER_BINDTYPE_BUTTON && binding->output.button == button) {
                if (binding->inputType == SDL_CONTROLLER_BINDTYPE_AXIS) {
                    SDL_bool valid_input_range;

                    int value = SDL_JoystickGetAxis(gamecontroller->joystick, binding->input.axis.axis);
                    int threshold = binding->input.axis.axis_min + (binding->input.axis.axis_max - binding->input.axis.axis_min) / 2;
                    if (binding->input.axis.axis_min < binding->input.axis.axis_max) {
                        valid_input_range = (value >= binding->input.axis.axis_min && value <= binding->input.axis.axis_max);
                        if (valid_input_range) {
                            retval |= (value >= threshold) ? SDL_PRESSED : SDL_RELEASED;
                        }
                    } else {
                        valid_input_range = (value >= binding->input.axis.axis_max && value <= binding->input.axis.axis_min);
                        if (valid_input_range) {
                            retval |= (value <= threshold) ? SDL_PRESSED : SDL_RELEASED;
                        }
                    }
                } else if (binding->inputType == SDL_CONTROLLER_BINDTYPE_BUTTON) {
                    retval |= SDL_JoystickGetButton(gamecontroller->joystick, binding->input.button);
                } else if (binding->inputType == SDL_CONTROLLER_BINDTYPE_HAT) {
                    int hat_mask = SDL_JoystickGetHat(gamecontroller->joystick, binding->input.hat.hat);
                    retval |= (hat_mask & binding->input.hat.hat_mask) ? SDL_PRESSED : SDL_RELEASED;
                }
            }
        }
    }
    SDL_UnlockJoysticks();

    return retval;
}

/**
 *  Get the number of touchpads on a game controller.
 */
int SDL_GameControllerGetNumTouchpads(SDL_GameController *gamecontroller)
{
    int retval = 0;

    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);
        if (joystick) {
            retval = joystick->ntouchpads;
        }
    }
    SDL_UnlockJoysticks();

    return retval;
}

/**
 *  Get the number of supported simultaneous fingers on a touchpad on a game controller.
 */
int SDL_GameControllerGetNumTouchpadFingers(SDL_GameController *gamecontroller, int touchpad)
{
    int retval = 0;

    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);
        if (joystick) {
            if (touchpad >= 0 && touchpad < joystick->ntouchpads) {
                retval = joystick->touchpads[touchpad].nfingers;
            } else {
                retval = SDL_InvalidParamError("touchpad");
            }
        }
    }
    SDL_UnlockJoysticks();

    return retval;
}

/**
 *  Get the current state of a finger on a touchpad on a game controller.
 */
int SDL_GameControllerGetTouchpadFinger(SDL_GameController *gamecontroller, int touchpad, int finger, Uint8 *state, float *x, float *y, float *pressure)
{
    int retval = -1;

    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);
        if (joystick) {
            if (touchpad >= 0 && touchpad < joystick->ntouchpads) {
                SDL_JoystickTouchpadInfo *touchpad_info = &joystick->touchpads[touchpad];
                if (finger >= 0 && finger < touchpad_info->nfingers) {
                    SDL_JoystickTouchpadFingerInfo *info = &touchpad_info->fingers[finger];

                    if (state) {
                        *state = info->state;
                    }
                    if (x) {
                        *x = info->x;
                    }
                    if (y) {
                        *y = info->y;
                    }
                    if (pressure) {
                        *pressure = info->pressure;
                    }
                    retval = 0;
                } else {
                    retval = SDL_InvalidParamError("finger");
                }
            } else {
                retval = SDL_InvalidParamError("touchpad");
            }
        }
    }
    SDL_UnlockJoysticks();

    return retval;
}

/**
 *  Return whether a game controller has a particular sensor.
 */
SDL_bool SDL_GameControllerHasSensor(SDL_GameController *gamecontroller, SDL_SensorType type)
{
    SDL_bool retval = SDL_FALSE;

    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);
        if (joystick) {
            int i;
            for (i = 0; i < joystick->nsensors; ++i) {
                if (joystick->sensors[i].type == type) {
                    retval = SDL_TRUE;
                    break;
                }
            }
        }
    }
    SDL_UnlockJoysticks();

    return retval;
}

/*
 *  Set whether data reporting for a game controller sensor is enabled
 */
int SDL_GameControllerSetSensorEnabled(SDL_GameController *gamecontroller, SDL_SensorType type, SDL_bool enabled)
{
    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);
        if (joystick) {
            int i;
            for (i = 0; i < joystick->nsensors; ++i) {
                SDL_JoystickSensorInfo *sensor = &joystick->sensors[i];

                if (sensor->type == type) {
                    if (sensor->enabled == enabled) {
                        SDL_UnlockJoysticks();
                        return 0;
                    }

                    if (enabled) {
                        if (joystick->nsensors_enabled == 0) {
                            if (joystick->driver->SetSensorsEnabled(joystick, SDL_TRUE) < 0) {
                                SDL_UnlockJoysticks();
                                return -1;
                            }
                        }
                        ++joystick->nsensors_enabled;
                    } else {
                        if (joystick->nsensors_enabled == 1) {
                            if (joystick->driver->SetSensorsEnabled(joystick, SDL_FALSE) < 0) {
                                SDL_UnlockJoysticks();
                                return -1;
                            }
                        }
                        --joystick->nsensors_enabled;
                    }

                    sensor->enabled = enabled;
                    SDL_UnlockJoysticks();
                    return 0;
                }
            }
        }
    }
    SDL_UnlockJoysticks();

    return SDL_Unsupported();
}

/*
 *  Query whether sensor data reporting is enabled for a game controller
 */
SDL_bool SDL_GameControllerIsSensorEnabled(SDL_GameController *gamecontroller, SDL_SensorType type)
{
    SDL_bool retval = SDL_FALSE;

    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);
        if (joystick) {
            int i;
            for (i = 0; i < joystick->nsensors; ++i) {
                if (joystick->sensors[i].type == type) {
                    retval = joystick->sensors[i].enabled;
                    break;
                }
            }
        }
    }
    SDL_UnlockJoysticks();

    return retval;
}

/*
 *  Get the data rate of a game controller sensor.
 */
float SDL_GameControllerGetSensorDataRate(SDL_GameController *gamecontroller, SDL_SensorType type)
{
    float retval = 0.0f;

    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);
        if (joystick) {
            int i;
            for (i = 0; i < joystick->nsensors; ++i) {
                SDL_JoystickSensorInfo *sensor = &joystick->sensors[i];

                if (sensor->type == type) {
                    retval = sensor->rate;
                    break;
                }
            }
        }
    }
    SDL_UnlockJoysticks();

    return retval;
}

/*
 *  Get the current state of a game controller sensor.
 */
int SDL_GameControllerGetSensorData(SDL_GameController *gamecontroller, SDL_SensorType type, float *data, int num_values)
{
    return SDL_GameControllerGetSensorDataWithTimestamp(gamecontroller, type, NULL, data, num_values);
}

/*
 *  Get the current state of a game controller sensor.
 */
int SDL_GameControllerGetSensorDataWithTimestamp(SDL_GameController *gamecontroller, SDL_SensorType type, Uint64 *timestamp, float *data, int num_values)
{
    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);
        if (joystick) {
            int i;
            for (i = 0; i < joystick->nsensors; ++i) {
                SDL_JoystickSensorInfo *sensor = &joystick->sensors[i];

                if (sensor->type == type) {
                    num_values = SDL_min(num_values, SDL_arraysize(sensor->data));
                    SDL_memcpy(data, sensor->data, num_values * sizeof(*data));
                    if (timestamp) {
                        *timestamp = sensor->timestamp_us;
                    }
                    SDL_UnlockJoysticks();
                    return 0;
                }
            }
        }
    }
    SDL_UnlockJoysticks();

    return SDL_Unsupported();
}

const char *SDL_GameControllerName(SDL_GameController *gamecontroller)
{
    const char *retval = NULL;

    SDL_LockJoysticks();
    {
        CHECK_GAMECONTROLLER_MAGIC(gamecontroller, NULL);

        if (SDL_strcmp(gamecontroller->name, "*") == 0 ||
            gamecontroller->joystick->steam_handle != 0) {
            retval = SDL_JoystickName(gamecontroller->joystick);
        } else {
            retval = gamecontroller->name;
        }
    }
    SDL_UnlockJoysticks();

    return retval;
}

const char *SDL_GameControllerPath(SDL_GameController *gamecontroller)
{
    SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);

    if (!joystick) {
        return NULL;
    }
    return SDL_JoystickPath(joystick);
}

SDL_GameControllerType SDL_GameControllerGetType(SDL_GameController *gamecontroller)
{
    SDL_GameControllerType type = SDL_CONTROLLER_TYPE_UNKNOWN;
    SDL_Joystick *joystick;
    const SDL_SteamVirtualGamepadInfo *info;

    SDL_LockJoysticks();
    {
        CHECK_GAMECONTROLLER_MAGIC(gamecontroller, SDL_CONTROLLER_TYPE_UNKNOWN);

        joystick = gamecontroller->joystick;
        info = SDL_GetJoystickInstanceVirtualGamepadInfo(joystick->instance_id);
        if (info) {
            type = info->type;
        } else if (gamecontroller->type != SDL_CONTROLLER_TYPE_UNKNOWN) {
            type = gamecontroller->type;
        } else {
            type = SDL_GetJoystickGameControllerTypeFromGUID(SDL_JoystickGetGUID(joystick), SDL_JoystickName(joystick));
        }
    }
    SDL_UnlockJoysticks();

    return type;
}

int SDL_GameControllerGetPlayerIndex(SDL_GameController *gamecontroller)
{
    SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);

    if (!joystick) {
        return -1;
    }
    return SDL_JoystickGetPlayerIndex(joystick);
}

/**
 *  Set the player index of an opened game controller
 */
void SDL_GameControllerSetPlayerIndex(SDL_GameController *gamecontroller, int player_index)
{
    SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);

    if (!joystick) {
        return;
    }
    SDL_JoystickSetPlayerIndex(joystick, player_index);
}

Uint16 SDL_GameControllerGetVendor(SDL_GameController *gamecontroller)
{
    SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);

    if (!joystick) {
        return 0;
    }
    return SDL_JoystickGetVendor(joystick);
}

Uint16 SDL_GameControllerGetProduct(SDL_GameController *gamecontroller)
{
    SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);

    if (!joystick) {
        return 0;
    }
    return SDL_JoystickGetProduct(joystick);
}

Uint16 SDL_GameControllerGetProductVersion(SDL_GameController *gamecontroller)
{
    SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);

    if (!joystick) {
        return 0;
    }
    return SDL_JoystickGetProductVersion(joystick);
}

Uint16 SDL_GameControllerGetFirmwareVersion(SDL_GameController *gamecontroller)
{
    SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);

    if (!joystick) {
        return 0;
    }
    return SDL_JoystickGetFirmwareVersion(joystick);
}

const char * SDL_GameControllerGetSerial(SDL_GameController *gamecontroller)
{
    SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);

    if (!joystick) {
        return NULL;
    }
    return SDL_JoystickGetSerial(joystick);
}

Uint64 SDL_GameControllerGetSteamHandle(SDL_GameController *gamecontroller)
{
    Uint64 handle = 0;

    SDL_LockJoysticks();
    {
        CHECK_GAMECONTROLLER_MAGIC(gamecontroller, 0);

        handle = gamecontroller->joystick->steam_handle;
    }
    SDL_UnlockJoysticks();

    return handle;
}

/*
 * Return if the controller in question is currently attached to the system,
 *  \return 0 if not plugged in, 1 if still present.
 */
SDL_bool SDL_GameControllerGetAttached(SDL_GameController *gamecontroller)
{
    SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);

    if (!joystick) {
        return SDL_FALSE;
    }
    return SDL_JoystickGetAttached(joystick);
}

/*
 * Get the joystick for this controller
 */
SDL_Joystick *SDL_GameControllerGetJoystick(SDL_GameController *gamecontroller)
{
    SDL_Joystick *joystick;

    SDL_LockJoysticks();
    {
        CHECK_GAMECONTROLLER_MAGIC(gamecontroller, NULL);

        joystick = gamecontroller->joystick;
    }
    SDL_UnlockJoysticks();

    return joystick;
}

/*
 * Return the SDL_GameController associated with an instance id.
 */
SDL_GameController *SDL_GameControllerFromInstanceID(SDL_JoystickID joyid)
{
    SDL_GameController *gamecontroller;

    SDL_LockJoysticks();
    gamecontroller = SDL_gamecontrollers;
    while (gamecontroller) {
        if (gamecontroller->joystick->instance_id == joyid) {
            SDL_UnlockJoysticks();
            return gamecontroller;
        }
        gamecontroller = gamecontroller->next;
    }
    SDL_UnlockJoysticks();
    return NULL;
}

/**
 * Return the SDL_GameController associated with a player index.
 */
SDL_GameController *SDL_GameControllerFromPlayerIndex(int player_index)
{
    SDL_GameController *retval = NULL;

    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_JoystickFromPlayerIndex(player_index);
        if (joystick) {
            retval = SDL_GameControllerFromInstanceID(joystick->instance_id);
        }
    }
    SDL_UnlockJoysticks();

    return retval;
}

/*
 * Get the SDL joystick layer binding for this controller axis mapping
 */
SDL_GameControllerButtonBind SDL_GameControllerGetBindForAxis(SDL_GameController *gamecontroller, SDL_GameControllerAxis axis)
{
    SDL_GameControllerButtonBind bind;

    SDL_zero(bind);

    SDL_LockJoysticks();
    {
        CHECK_GAMECONTROLLER_MAGIC(gamecontroller, bind);

        if (axis != SDL_CONTROLLER_AXIS_INVALID) {
            int i;
            for (i = 0; i < gamecontroller->num_bindings; ++i) {
                SDL_ExtendedGameControllerBind *binding = &gamecontroller->bindings[i];
                if (binding->outputType == SDL_CONTROLLER_BINDTYPE_AXIS && binding->output.axis.axis == axis) {
                    bind.bindType = binding->inputType;
                    if (binding->inputType == SDL_CONTROLLER_BINDTYPE_AXIS) {
                        /* FIXME: There might be multiple axes bound now that we have axis ranges... */
                        bind.value.axis = binding->input.axis.axis;
                    } else if (binding->inputType == SDL_CONTROLLER_BINDTYPE_BUTTON) {
                        bind.value.button = binding->input.button;
                    } else if (binding->inputType == SDL_CONTROLLER_BINDTYPE_HAT) {
                        bind.value.hat.hat = binding->input.hat.hat;
                        bind.value.hat.hat_mask = binding->input.hat.hat_mask;
                    }
                    break;
                }
            }
        }
    }
    SDL_UnlockJoysticks();

    return bind;
}

/*
 * Get the SDL joystick layer binding for this controller button mapping
 */
SDL_GameControllerButtonBind SDL_GameControllerGetBindForButton(SDL_GameController *gamecontroller, SDL_GameControllerButton button)
{
    SDL_GameControllerButtonBind bind;

    SDL_zero(bind);

    SDL_LockJoysticks();
    {
        CHECK_GAMECONTROLLER_MAGIC(gamecontroller, bind);

        if (button != SDL_CONTROLLER_BUTTON_INVALID) {
            int i;
            for (i = 0; i < gamecontroller->num_bindings; ++i) {
                SDL_ExtendedGameControllerBind *binding = &gamecontroller->bindings[i];
                if (binding->outputType == SDL_CONTROLLER_BINDTYPE_BUTTON && binding->output.button == button) {
                    bind.bindType = binding->inputType;
                    if (binding->inputType == SDL_CONTROLLER_BINDTYPE_AXIS) {
                        bind.value.axis = binding->input.axis.axis;
                    } else if (binding->inputType == SDL_CONTROLLER_BINDTYPE_BUTTON) {
                        bind.value.button = binding->input.button;
                    } else if (binding->inputType == SDL_CONTROLLER_BINDTYPE_HAT) {
                        bind.value.hat.hat = binding->input.hat.hat;
                        bind.value.hat.hat_mask = binding->input.hat.hat_mask;
                    }
                    break;
                }
            }
        }
    }
    SDL_UnlockJoysticks();

    return bind;
}

int SDL_GameControllerRumble(SDL_GameController *gamecontroller, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble, Uint32 duration_ms)
{
    SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);

    if (!joystick) {
        return -1;
    }
    return SDL_JoystickRumble(joystick, low_frequency_rumble, high_frequency_rumble, duration_ms);
}

int SDL_GameControllerRumbleTriggers(SDL_GameController *gamecontroller, Uint16 left_rumble, Uint16 right_rumble, Uint32 duration_ms)
{
    SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);

    if (!joystick) {
        return -1;
    }
    return SDL_JoystickRumbleTriggers(joystick, left_rumble, right_rumble, duration_ms);
}

SDL_bool SDL_GameControllerHasLED(SDL_GameController *gamecontroller)
{
    SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);

    if (!joystick) {
        return SDL_FALSE;
    }
    return SDL_JoystickHasLED(joystick);
}

SDL_bool SDL_GameControllerHasRumble(SDL_GameController *gamecontroller)
{
    SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);

    if (!joystick) {
        return SDL_FALSE;
    }
    return SDL_JoystickHasRumble(joystick);
}

SDL_bool SDL_GameControllerHasRumbleTriggers(SDL_GameController *gamecontroller)
{
    SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);

    if (!joystick) {
        return SDL_FALSE;
    }
    return SDL_JoystickHasRumbleTriggers(joystick);
}

int SDL_GameControllerSetLED(SDL_GameController *gamecontroller, Uint8 red, Uint8 green, Uint8 blue)
{
    SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);

    if (!joystick) {
        return -1;
    }
    return SDL_JoystickSetLED(joystick, red, green, blue);
}

int SDL_GameControllerSendEffect(SDL_GameController *gamecontroller, const void *data, int size)
{
    SDL_Joystick *joystick = SDL_GameControllerGetJoystick(gamecontroller);

    if (!joystick) {
        return -1;
    }
    return SDL_JoystickSendEffect(joystick, data, size);
}

void SDL_GameControllerClose(SDL_GameController *gamecontroller)
{
    SDL_GameController *gamecontrollerlist, *gamecontrollerlistprev;

    SDL_LockJoysticks();

    if (!gamecontroller || gamecontroller->magic != &gamecontroller_magic) {
        SDL_UnlockJoysticks();
        return;
    }

    /* First decrement ref count */
    if (--gamecontroller->ref_count > 0) {
        SDL_UnlockJoysticks();
        return;
    }

    SDL_JoystickClose(gamecontroller->joystick);

    gamecontrollerlist = SDL_gamecontrollers;
    gamecontrollerlistprev = NULL;
    while (gamecontrollerlist) {
        if (gamecontroller == gamecontrollerlist) {
            if (gamecontrollerlistprev) {
                /* unlink this entry */
                gamecontrollerlistprev->next = gamecontrollerlist->next;
            } else {
                SDL_gamecontrollers = gamecontroller->next;
            }
            break;
        }
        gamecontrollerlistprev = gamecontrollerlist;
        gamecontrollerlist = gamecontrollerlist->next;
    }

    gamecontroller->magic = NULL;
    SDL_free(gamecontroller->bindings);
    SDL_free(gamecontroller->last_match_axis);
    SDL_free(gamecontroller->last_hat_mask);
    SDL_free(gamecontroller);

    SDL_UnlockJoysticks();
}

/*
 * Quit the controller subsystem
 */
void SDL_GameControllerQuit(void)
{
    SDL_LockJoysticks();
    while (SDL_gamecontrollers) {
        SDL_gamecontrollers->ref_count = 1;
        SDL_GameControllerClose(SDL_gamecontrollers);
    }
    SDL_UnlockJoysticks();
}

void SDL_GameControllerQuitMappings(void)
{
    ControllerMapping_t *pControllerMap;

    SDL_AssertJoysticksLocked();

    while (s_pSupportedControllers) {
        pControllerMap = s_pSupportedControllers;
        s_pSupportedControllers = s_pSupportedControllers->next;
        SDL_free(pControllerMap->name);
        SDL_free(pControllerMap->mapping);
        SDL_free(pControllerMap);
    }

    SDL_DelEventWatch(SDL_GameControllerEventWatcher, NULL);

    SDL_FreeVIDPIDList(&SDL_allowed_controllers);
    SDL_FreeVIDPIDList(&SDL_ignored_controllers);
}

/*
 * Event filter to transform joystick events into appropriate game controller ones
 */
static int SDL_PrivateGameControllerAxis(SDL_GameController *gamecontroller, SDL_GameControllerAxis axis, Sint16 value)
{
    int posted;

    SDL_AssertJoysticksLocked();

    /* translate the event, if desired */
    posted = 0;
#ifndef SDL_EVENTS_DISABLED
    if (SDL_GetEventState(SDL_CONTROLLERAXISMOTION) == SDL_ENABLE) {
        SDL_Event event;
        event.type = SDL_CONTROLLERAXISMOTION;
        event.caxis.which = gamecontroller->joystick->instance_id;
        event.caxis.axis = axis;
        event.caxis.value = value;
        posted = SDL_PushEvent(&event) == 1;
    }
#endif /* !SDL_EVENTS_DISABLED */
    return posted;
}

/*
 * Event filter to transform joystick events into appropriate game controller ones
 */
static int SDL_PrivateGameControllerButton(SDL_GameController *gamecontroller, SDL_GameControllerButton button, Uint8 state)
{
    int posted;
#ifndef SDL_EVENTS_DISABLED
    SDL_Event event;

    SDL_AssertJoysticksLocked();

    if (button == SDL_CONTROLLER_BUTTON_INVALID) {
        return 0;
    }

    switch (state) {
    case SDL_PRESSED:
        event.type = SDL_CONTROLLERBUTTONDOWN;
        break;
    case SDL_RELEASED:
        event.type = SDL_CONTROLLERBUTTONUP;
        break;
    default:
        /* Invalid state -- bail */
        return 0;
    }
#endif /* !SDL_EVENTS_DISABLED */

    if (button == SDL_CONTROLLER_BUTTON_GUIDE) {
        Uint32 now = SDL_GetTicks();
        if (state == SDL_PRESSED) {
            gamecontroller->guide_button_down = now;

            if (gamecontroller->joystick->delayed_guide_button) {
                /* Skip duplicate press */
                return 0;
            }
        } else {
            if (!SDL_TICKS_PASSED(now, gamecontroller->guide_button_down + SDL_MINIMUM_GUIDE_BUTTON_DELAY_MS)) {
                gamecontroller->joystick->delayed_guide_button = SDL_TRUE;
                return 0;
            }
            gamecontroller->joystick->delayed_guide_button = SDL_FALSE;
        }
    }

    /* translate the event, if desired */
    posted = 0;
#ifndef SDL_EVENTS_DISABLED
    if (SDL_GetEventState(event.type) == SDL_ENABLE) {
        event.cbutton.which = gamecontroller->joystick->instance_id;
        event.cbutton.button = button;
        event.cbutton.state = state;
        posted = SDL_PushEvent(&event) == 1;
    }
#endif /* !SDL_EVENTS_DISABLED */
    return posted;
}

/*
 * Turn off controller events
 */
int SDL_GameControllerEventState(int state)
{
#ifdef SDL_EVENTS_DISABLED
    return SDL_IGNORE;
#else
    const Uint32 event_list[] = {
        SDL_CONTROLLERAXISMOTION,
        SDL_CONTROLLERBUTTONDOWN,
        SDL_CONTROLLERBUTTONUP,
        SDL_CONTROLLERDEVICEADDED,
        SDL_CONTROLLERDEVICEREMOVED,
        SDL_CONTROLLERDEVICEREMAPPED,
        SDL_CONTROLLERTOUCHPADDOWN,
        SDL_CONTROLLERTOUCHPADMOTION,
        SDL_CONTROLLERTOUCHPADUP,
        SDL_CONTROLLERSENSORUPDATE,
    };
    unsigned int i;

    switch (state) {
    case SDL_QUERY:
        state = SDL_IGNORE;
        for (i = 0; i < SDL_arraysize(event_list); ++i) {
            state = SDL_EventState(event_list[i], SDL_QUERY);
            if (state == SDL_ENABLE) {
                break;
            }
        }
        break;
    default:
        for (i = 0; i < SDL_arraysize(event_list); ++i) {
            (void)SDL_EventState(event_list[i], state);
        }
        break;
    }
    return state;
#endif /* SDL_EVENTS_DISABLED */
}

void SDL_GameControllerHandleDelayedGuideButton(SDL_Joystick *joystick)
{
    SDL_GameController *controller;

    SDL_AssertJoysticksLocked();

    for (controller = SDL_gamecontrollers; controller; controller = controller->next) {
        if (controller->joystick == joystick) {
            SDL_PrivateGameControllerButton(controller, SDL_CONTROLLER_BUTTON_GUIDE, SDL_RELEASED);
            break;
        }
    }
}

const char *SDL_GameControllerGetAppleSFSymbolsNameForButton(SDL_GameController *gamecontroller, SDL_GameControllerButton button)
{
#if defined(SDL_JOYSTICK_MFI)
    const char *IOS_GameControllerGetAppleSFSymbolsNameForButton(SDL_GameController * gamecontroller, SDL_GameControllerButton button);
    const char *retval;

    SDL_LockJoysticks();
    {
        CHECK_GAMECONTROLLER_MAGIC(gamecontroller, NULL);

        retval = IOS_GameControllerGetAppleSFSymbolsNameForButton(gamecontroller, button);
    }
    SDL_UnlockJoysticks();

    if (retval && *retval) {
        return retval;
    }
#endif
    return NULL;
}

const char *SDL_GameControllerGetAppleSFSymbolsNameForAxis(SDL_GameController *gamecontroller, SDL_GameControllerAxis axis)
{
#if defined(SDL_JOYSTICK_MFI)
    const char *IOS_GameControllerGetAppleSFSymbolsNameForAxis(SDL_GameController * gamecontroller, SDL_GameControllerAxis axis);
    const char *retval;

    SDL_LockJoysticks();
    {
        CHECK_GAMECONTROLLER_MAGIC(gamecontroller, NULL);

        retval = IOS_GameControllerGetAppleSFSymbolsNameForAxis(gamecontroller, axis);
    }
    SDL_UnlockJoysticks();

    if (retval && *retval) {
        return retval;
    }
#endif
    return NULL;
}

/* vi: set ts=4 sw=4 expandtab: */
