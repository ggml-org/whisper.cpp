/*
  Simple DirectMedia Layer
  Copyright (C) 2025 Sam Lantinga <slouken@libsdl.org>

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
/*
  RAWINPUT Joystick API for better handling XInput-capable devices on Windows.

  XInput is limited to 4 devices.
  Windows.Gaming.Input does not get inputs from XBox One controllers when not in the foreground.
  DirectInput does not get inputs from XBox One controllers when not in the foreground, nor rumble or accurate triggers.
  RawInput does not get rumble or accurate triggers.

  So, combine them as best we can!
*/
#include "../../SDL_internal.h"

#ifdef SDL_JOYSTICK_RAWINPUT

#include "SDL_atomic.h"
#include "SDL_endian.h"
#include "SDL_events.h"
#include "SDL_hints.h"
#include "SDL_timer.h"
#include "../usb_ids.h"
#include "../SDL_sysjoystick.h"
#include "../../core/windows/SDL_windows.h"
#include "../../core/windows/SDL_hid.h"
#include "../hidapi/SDL_hidapijoystick_c.h"

/* SDL_JOYSTICK_RAWINPUT_XINPUT is disabled because using XInput at the same time as
   raw input will turn off the Xbox Series X controller when it is connected via the
   Xbox One Wireless Adapter.
 */
#ifdef HAVE_XINPUT_H
#define SDL_JOYSTICK_RAWINPUT_XINPUT
#endif
#ifdef HAVE_WINDOWS_GAMING_INPUT_H
#define SDL_JOYSTICK_RAWINPUT_WGI
#endif

#ifdef SDL_JOYSTICK_RAWINPUT_XINPUT
#include "../../core/windows/SDL_xinput.h"
#endif

#ifdef SDL_JOYSTICK_RAWINPUT_WGI
#include "../../core/windows/SDL_windows.h"
typedef struct WindowsGamingInputGamepadState WindowsGamingInputGamepadState;
#define GamepadButtons_GUIDE 0x40000000
#define COBJMACROS
#include "windows.gaming.input.h"
#include <roapi.h>
#endif

#if defined(SDL_JOYSTICK_RAWINPUT_XINPUT) || defined(SDL_JOYSTICK_RAWINPUT_WGI)
#define SDL_JOYSTICK_RAWINPUT_MATCHING
#define SDL_JOYSTICK_RAWINPUT_MATCH_AXES
#define SDL_JOYSTICK_RAWINPUT_MATCH_TRIGGERS
#ifdef SDL_JOYSTICK_RAWINPUT_MATCH_TRIGGERS
#define SDL_JOYSTICK_RAWINPUT_MATCH_COUNT 6 // stick + trigger axes
#else
#define SDL_JOYSTICK_RAWINPUT_MATCH_COUNT 4 // stick axes
#endif
#endif

#if 0
#define DEBUG_RAWINPUT
#endif

#ifndef RIDEV_EXINPUTSINK
#define RIDEV_EXINPUTSINK 0x00001000
#define RIDEV_DEVNOTIFY   0x00002000
#endif

#ifndef WM_INPUT_DEVICE_CHANGE
#define WM_INPUT_DEVICE_CHANGE 0x00FE
#endif
#ifndef WM_INPUT
#define WM_INPUT 0x00FF
#endif
#ifndef GIDC_ARRIVAL
#define GIDC_ARRIVAL 1
#define GIDC_REMOVAL 2
#endif

extern void WINDOWS_RAWINPUTEnabledChanged(void);
extern void WINDOWS_JoystickDetect(void);

static SDL_bool SDL_RAWINPUT_inited = SDL_FALSE;
static SDL_bool SDL_RAWINPUT_remote_desktop = SDL_FALSE;
static int SDL_RAWINPUT_numjoysticks = 0;

static void RAWINPUT_JoystickClose(SDL_Joystick *joystick);

typedef struct _SDL_RAWINPUT_Device
{
    SDL_atomic_t refcount;
    char *name;
    char *path;
    Uint16 vendor_id;
    Uint16 product_id;
    Uint16 version;
    SDL_JoystickGUID guid;
    SDL_bool is_xinput;
    SDL_bool is_xboxone;
    int steam_virtual_gamepad_slot;
    PHIDP_PREPARSED_DATA preparsed_data;

    HANDLE hDevice;
    SDL_Joystick *joystick;
    SDL_JoystickID joystick_id;

    struct _SDL_RAWINPUT_Device *next;
} SDL_RAWINPUT_Device;

struct joystick_hwdata
{
    SDL_bool is_xinput;
    SDL_bool is_xboxone;
    PHIDP_PREPARSED_DATA preparsed_data;
    ULONG max_data_length;
    HIDP_DATA *data;
    USHORT *button_indices;
    USHORT *axis_indices;
    USHORT *hat_indices;
    SDL_bool guide_hack;
    SDL_bool trigger_hack;
    USHORT trigger_hack_index;

#ifdef SDL_JOYSTICK_RAWINPUT_MATCHING
    Uint64 match_state; /* Lowest 16 bits for button states, higher 24 for 6 4bit axes */
    Uint32 last_state_packet;
#endif

#ifdef SDL_JOYSTICK_RAWINPUT_XINPUT
    SDL_bool xinput_enabled;
    SDL_bool xinput_correlated;
    Uint8 xinput_correlation_id;
    Uint8 xinput_correlation_count;
    Uint8 xinput_uncorrelate_count;
    Uint8 xinput_slot;
#endif

#ifdef SDL_JOYSTICK_RAWINPUT_WGI
    SDL_bool wgi_correlated;
    Uint8 wgi_correlation_id;
    Uint8 wgi_correlation_count;
    Uint8 wgi_uncorrelate_count;
    WindowsGamingInputGamepadState *wgi_slot;
#endif

    SDL_RAWINPUT_Device *device;
};
typedef struct joystick_hwdata RAWINPUT_DeviceContext;

SDL_RAWINPUT_Device *SDL_RAWINPUT_devices;

static const Uint16 subscribed_devices[] = {
    USB_USAGE_GENERIC_GAMEPAD,
    /* Don't need Joystick for any devices we're handling here (XInput-capable)
    USB_USAGE_GENERIC_JOYSTICK,
    USB_USAGE_GENERIC_MULTIAXISCONTROLLER,
    */
};

#ifdef SDL_JOYSTICK_RAWINPUT_MATCHING

static struct
{
    Uint32 last_state_packet;
    SDL_Joystick *joystick;
    SDL_Joystick *last_joystick;
} guide_button_candidate;

typedef struct WindowsMatchState
{
#ifdef SDL_JOYSTICK_RAWINPUT_MATCH_AXES
    SHORT match_axes[SDL_JOYSTICK_RAWINPUT_MATCH_COUNT];
#endif
#ifdef SDL_JOYSTICK_RAWINPUT_XINPUT
    WORD xinput_buttons;
#endif
#ifdef SDL_JOYSTICK_RAWINPUT_WGI
    Uint32 wgi_buttons;
#endif
    SDL_bool any_data;
} WindowsMatchState;

static void RAWINPUT_FillMatchState(WindowsMatchState *state, Uint64 match_state)
{
#ifdef SDL_JOYSTICK_RAWINPUT_MATCH_AXES
    int ii;
#endif

    SDL_bool any_axes_data = SDL_FALSE;
#ifdef SDL_JOYSTICK_RAWINPUT_MATCH_AXES
    /*  SHORT state->match_axes[4] = {
            (match_state & 0x000F0000) >> 4,
            (match_state & 0x00F00000) >> 8,
            (match_state & 0x0F000000) >> 12,
            (match_state & 0xF0000000) >> 16,
        }; */
    for (ii = 0; ii < 4; ii++) {
        state->match_axes[ii] = (SHORT)((match_state & (0x000F0000ull << (ii * 4))) >> (4 + ii * 4));
        any_axes_data |= ((Uint32)(state->match_axes[ii] + 0x1000) > 0x2000); /* match_state bit is not 0xF, 0x1, or 0x2 */
    }
#endif /* SDL_JOYSTICK_RAWINPUT_MATCH_AXES */
#ifdef SDL_JOYSTICK_RAWINPUT_MATCH_TRIGGERS
    for (; ii < SDL_JOYSTICK_RAWINPUT_MATCH_COUNT; ii++) {
        state->match_axes[ii] = (SHORT)((match_state & (0x000F0000ull << (ii * 4))) >> (4 + ii * 4));
        any_axes_data |= (state->match_axes[ii] != SDL_MIN_SINT16);
    }
#endif /* SDL_JOYSTICK_RAWINPUT_MATCH_TRIGGERS */

    state->any_data = any_axes_data;

#ifdef SDL_JOYSTICK_RAWINPUT_XINPUT
    /* Match axes by checking if the distance between the high 4 bits of axis and the 4 bits from match_state is 1 or less */
#define XInputAxesMatch(gamepad) (                                           \
    (Uint32)(gamepad.sThumbLX - state->match_axes[0] + 0x1000) <= 0x2fff &&  \
    (Uint32)(~gamepad.sThumbLY - state->match_axes[1] + 0x1000) <= 0x2fff && \
    (Uint32)(gamepad.sThumbRX - state->match_axes[2] + 0x1000) <= 0x2fff &&  \
    (Uint32)(~gamepad.sThumbRY - state->match_axes[3] + 0x1000) <= 0x2fff)
    /* Explicit
#define XInputAxesMatch(gamepad) (\
    SDL_abs((Sint8)((gamepad.sThumbLX & 0xF000) >> 8) - ((match_state & 0x000F0000) >> 12)) <= 0x10 && \
    SDL_abs((Sint8)((~gamepad.sThumbLY & 0xF000) >> 8) - ((match_state & 0x00F00000) >> 16)) <= 0x10 && \
    SDL_abs((Sint8)((gamepad.sThumbRX & 0xF000) >> 8) - ((match_state & 0x0F000000) >> 20)) <= 0x10 && \
    SDL_abs((Sint8)((~gamepad.sThumbRY & 0xF000) >> 8) - ((match_state & 0xF0000000) >> 24)) <= 0x10) */

    /* Can only match trigger values if a single trigger has a value. */
#define XInputTriggersMatch(gamepad) (                                                          \
    ((state->match_axes[4] == SDL_MIN_SINT16) && (state->match_axes[5] == SDL_MIN_SINT16)) ||   \
    ((gamepad.bLeftTrigger != 0) && (gamepad.bRightTrigger != 0)) ||                            \
    ((Uint32)((((int)gamepad.bLeftTrigger * 257) - 32768) - state->match_axes[4]) <= 0x2fff) || \
    ((Uint32)((((int)gamepad.bRightTrigger * 257) - 32768) - state->match_axes[5]) <= 0x2fff))

    state->xinput_buttons =
        /* Bitwise map .RLDUWVQTS.KYXBA -> YXBA..WVQTKSRLDU */
        (WORD)(match_state << 12 | (match_state & 0x0780) >> 1 | (match_state & 0x0010) << 1 | (match_state & 0x0040) >> 2 | (match_state & 0x7800) >> 11);
    /*  Explicit
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_A)) ? XINPUT_GAMEPAD_A : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_B)) ? XINPUT_GAMEPAD_B : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_X)) ? XINPUT_GAMEPAD_X : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_Y)) ? XINPUT_GAMEPAD_Y : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_BACK)) ? XINPUT_GAMEPAD_BACK : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_START)) ? XINPUT_GAMEPAD_START : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_LEFTSTICK)) ? XINPUT_GAMEPAD_LEFT_THUMB : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_RIGHTSTICK)) ? XINPUT_GAMEPAD_RIGHT_THUMB: 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_LEFTSHOULDER)) ? XINPUT_GAMEPAD_LEFT_SHOULDER : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_RIGHTSHOULDER)) ? XINPUT_GAMEPAD_RIGHT_SHOULDER : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_DPAD_UP)) ? XINPUT_GAMEPAD_DPAD_UP : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_DPAD_DOWN)) ? XINPUT_GAMEPAD_DPAD_DOWN : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_DPAD_LEFT)) ? XINPUT_GAMEPAD_DPAD_LEFT : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_DPAD_RIGHT)) ? XINPUT_GAMEPAD_DPAD_RIGHT : 0);
    */

    if (state->xinput_buttons) {
        state->any_data = SDL_TRUE;
    }
#endif

#ifdef SDL_JOYSTICK_RAWINPUT_WGI
    /* Match axes by checking if the distance between the high 4 bits of axis and the 4 bits from match_state is 1 or less */
#define WindowsGamingInputAxesMatch(gamepad) (                                                                            \
    (Uint16)(((Sint16)(gamepad.LeftThumbstickX * SDL_MAX_SINT16) & 0xF000) - state->match_axes[0] + 0x1000) <= 0x2fff &&  \
    (Uint16)((~(Sint16)(gamepad.LeftThumbstickY * SDL_MAX_SINT16) & 0xF000) - state->match_axes[1] + 0x1000) <= 0x2fff && \
    (Uint16)(((Sint16)(gamepad.RightThumbstickX * SDL_MAX_SINT16) & 0xF000) - state->match_axes[2] + 0x1000) <= 0x2fff && \
    (Uint16)((~(Sint16)(gamepad.RightThumbstickY * SDL_MAX_SINT16) & 0xF000) - state->match_axes[3] + 0x1000) <= 0x2fff)

#define WindowsGamingInputTriggersMatch(gamepad) (                                                          \
    ((state->match_axes[4] == SDL_MIN_SINT16) && (state->match_axes[5] == SDL_MIN_SINT16)) ||               \
    ((gamepad.LeftTrigger == 0.0f) && (gamepad.RightTrigger == 0.0f)) ||                                    \
    ((Uint16)((((int)(gamepad.LeftTrigger * SDL_MAX_UINT16)) - 32768) - state->match_axes[4]) <= 0x2fff) || \
    ((Uint16)((((int)(gamepad.RightTrigger * SDL_MAX_UINT16)) - 32768) - state->match_axes[5]) <= 0x2fff))

    state->wgi_buttons =
        /* Bitwise map .RLD UWVQ TS.K YXBA -> ..QT WVRL DUYX BAKS */
        /*  RStick/LStick (QT)         RShould/LShould  (WV)                 DPad R/L/D/U                          YXBA                         bac(K)                      (S)tart */
        (match_state & 0x0180) << 5 | (match_state & 0x0600) << 1 | (match_state & 0x7800) >> 5 | (match_state & 0x000F) << 2 | (match_state & 0x0010) >> 3 | (match_state & 0x0040) >> 6;
    /*  Explicit
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_A)) ? GamepadButtons_A : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_B)) ? GamepadButtons_B : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_X)) ? GamepadButtons_X : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_Y)) ? GamepadButtons_Y : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_BACK)) ? GamepadButtons_View : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_START)) ? GamepadButtons_Menu : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_LEFTSTICK)) ? GamepadButtons_LeftThumbstick : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_RIGHTSTICK)) ? GamepadButtons_RightThumbstick: 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_LEFTSHOULDER)) ? GamepadButtons_LeftShoulder: 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_RIGHTSHOULDER)) ? GamepadButtons_RightShoulder: 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_DPAD_UP)) ? GamepadButtons_DPadUp : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_DPAD_DOWN)) ? GamepadButtons_DPadDown : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_DPAD_LEFT)) ? GamepadButtons_DPadLeft : 0) |
        ((match_state & (1<<SDL_CONTROLLER_BUTTON_DPAD_RIGHT)) ? GamepadButtons_DPadRight : 0); */

    if (state->wgi_buttons) {
        state->any_data = SDL_TRUE;
    }
#endif
}

#endif /* SDL_JOYSTICK_RAWINPUT_MATCHING */

#ifdef SDL_JOYSTICK_RAWINPUT_XINPUT

static struct
{
    XINPUT_STATE state;
    XINPUT_BATTERY_INFORMATION_EX battery;
    SDL_bool connected; /* Currently has an active XInput device */
    SDL_bool used;      /* Is currently mapped to an SDL device */
    Uint8 correlation_id;
} xinput_state[XUSER_MAX_COUNT];
static SDL_bool xinput_device_change = SDL_TRUE;
static SDL_bool xinput_state_dirty = SDL_TRUE;

static void RAWINPUT_UpdateXInput(void)
{
    DWORD user_index;
    if (xinput_device_change) {
        for (user_index = 0; user_index < XUSER_MAX_COUNT; user_index++) {
            XINPUT_CAPABILITIES capabilities;
            xinput_state[user_index].connected = (XINPUTGETCAPABILITIES(user_index, XINPUT_FLAG_GAMEPAD, &capabilities) == ERROR_SUCCESS) ? SDL_TRUE : SDL_FALSE;
        }
        xinput_device_change = SDL_FALSE;
        xinput_state_dirty = SDL_TRUE;
    }
    if (xinput_state_dirty) {
        xinput_state_dirty = SDL_FALSE;
        for (user_index = 0; user_index < SDL_arraysize(xinput_state); ++user_index) {
            if (xinput_state[user_index].connected) {
                if (XINPUTGETSTATE(user_index, &xinput_state[user_index].state) != ERROR_SUCCESS) {
                    xinput_state[user_index].connected = SDL_FALSE;
                }
                xinput_state[user_index].battery.BatteryType = BATTERY_TYPE_UNKNOWN;
                if (XINPUTGETBATTERYINFORMATION) {
                    XINPUTGETBATTERYINFORMATION(user_index, BATTERY_DEVTYPE_GAMEPAD, &xinput_state[user_index].battery);
                }
            }
        }
    }
}

static void RAWINPUT_MarkXInputSlotUsed(Uint8 xinput_slot)
{
    if (xinput_slot != XUSER_INDEX_ANY) {
        xinput_state[xinput_slot].used = SDL_TRUE;
    }
}

static void RAWINPUT_MarkXInputSlotFree(Uint8 xinput_slot)
{
    if (xinput_slot != XUSER_INDEX_ANY) {
        xinput_state[xinput_slot].used = SDL_FALSE;
    }
}
static SDL_bool RAWINPUT_MissingXInputSlot(void)
{
    int ii;
    for (ii = 0; ii < SDL_arraysize(xinput_state); ii++) {
        if (xinput_state[ii].connected && !xinput_state[ii].used) {
            return SDL_TRUE;
        }
    }
    return SDL_FALSE;
}

static SDL_bool RAWINPUT_XInputSlotMatches(const WindowsMatchState *state, Uint8 slot_idx)
{
    if (xinput_state[slot_idx].connected) {
        WORD xinput_buttons = xinput_state[slot_idx].state.Gamepad.wButtons;
        if ((xinput_buttons & ~XINPUT_GAMEPAD_GUIDE) == state->xinput_buttons
#ifdef SDL_JOYSTICK_RAWINPUT_MATCH_AXES
            && XInputAxesMatch(xinput_state[slot_idx].state.Gamepad)
#endif
#ifdef SDL_JOYSTICK_RAWINPUT_MATCH_TRIGGERS
            && XInputTriggersMatch(xinput_state[slot_idx].state.Gamepad)
#endif
        ) {
            return SDL_TRUE;
        }
    }
    return SDL_FALSE;
}

static SDL_bool RAWINPUT_GuessXInputSlot(const WindowsMatchState *state, Uint8 *correlation_id, Uint8 *slot_idx)
{
    int user_index;
    int match_count;

    /* If there is only one available slot, let's use that
     * That will be right most of the time, and uncorrelation will fix any bad guesses
     */
    match_count = 0;
    for (user_index = 0; user_index < XUSER_MAX_COUNT; ++user_index) {
        if (xinput_state[user_index].connected && !xinput_state[user_index].used) {
            *slot_idx = user_index;
            ++match_count;
        }
    }
    if (match_count == 1) {
        *correlation_id = ++xinput_state[*slot_idx].correlation_id;
        return SDL_TRUE;
    }

    *slot_idx = 0;

    match_count = 0;
    for (user_index = 0; user_index < XUSER_MAX_COUNT; ++user_index) {
        if (!xinput_state[user_index].used && RAWINPUT_XInputSlotMatches(state, user_index)) {
            ++match_count;
            *slot_idx = (Uint8)user_index;
            /* Incrementing correlation_id for any match, as negative evidence for others being correlated */
            *correlation_id = ++xinput_state[user_index].correlation_id;
        }
    }
    /* Only return a match if we match exactly one, and we have some non-zero data (buttons or axes) that matched.
       Note that we're still invalidating *other* potential correlations if we have more than one match or we have no
       data. */
    if (match_count == 1 && state->any_data) {
        return SDL_TRUE;
    }
    return SDL_FALSE;
}

#endif /* SDL_JOYSTICK_RAWINPUT_XINPUT */

#ifdef SDL_JOYSTICK_RAWINPUT_WGI

typedef struct WindowsGamingInputGamepadState
{
    __x_ABI_CWindows_CGaming_CInput_CIGamepad *gamepad;
    struct __x_ABI_CWindows_CGaming_CInput_CGamepadReading state;
    RAWINPUT_DeviceContext *correlated_context;
    SDL_bool used;      /* Is currently mapped to an SDL device */
    SDL_bool connected; /* Just used during update to track disconnected */
    Uint8 correlation_id;
    struct __x_ABI_CWindows_CGaming_CInput_CGamepadVibration vibration;
} WindowsGamingInputGamepadState;

static struct
{
    WindowsGamingInputGamepadState **per_gamepad;
    int per_gamepad_count;
    SDL_bool initialized;
    SDL_bool dirty;
    SDL_bool need_device_list_update;
    int ref_count;
    __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics *gamepad_statics;
    EventRegistrationToken gamepad_added_token;
    EventRegistrationToken gamepad_removed_token;
} wgi_state;

typedef struct GamepadDelegate
{
    __FIEventHandler_1_Windows__CGaming__CInput__CGamepad iface;
    SDL_atomic_t refcount;
} GamepadDelegate;

static const IID IID_IEventHandler_Gamepad = { 0x8a7639ee, 0x624a, 0x501a, { 0xbb, 0x53, 0x56, 0x2d, 0x1e, 0xc1, 0x1b, 0x52 } };

static HRESULT STDMETHODCALLTYPE IEventHandler_CGamepadVtbl_QueryInterface(__FIEventHandler_1_Windows__CGaming__CInput__CGamepad *This, REFIID riid, void **ppvObject)
{
    if (!ppvObject) {
        return E_INVALIDARG;
    }

    *ppvObject = NULL;
    if (WIN_IsEqualIID(riid, &IID_IUnknown) || WIN_IsEqualIID(riid, &IID_IAgileObject) || WIN_IsEqualIID(riid, &IID_IEventHandler_Gamepad)) {
        *ppvObject = This;
        __FIEventHandler_1_Windows__CGaming__CInput__CGamepad_AddRef(This);
        return S_OK;
    } else if (WIN_IsEqualIID(riid, &IID_IMarshal)) {
        /* This seems complicated. Let's hope it doesn't happen. */
        return E_OUTOFMEMORY;
    } else {
        return E_NOINTERFACE;
    }
}

static ULONG STDMETHODCALLTYPE IEventHandler_CGamepadVtbl_AddRef(__FIEventHandler_1_Windows__CGaming__CInput__CGamepad *This)
{
    GamepadDelegate *self = (GamepadDelegate *)This;
    return SDL_AtomicAdd(&self->refcount, 1) + 1UL;
}

static ULONG STDMETHODCALLTYPE IEventHandler_CGamepadVtbl_Release(__FIEventHandler_1_Windows__CGaming__CInput__CGamepad *This)
{
    GamepadDelegate *self = (GamepadDelegate *)This;
    int rc = SDL_AtomicAdd(&self->refcount, -1) - 1;
    /* Should never free the static delegate objects */
    SDL_assert(rc > 0);
    return rc;
}

static HRESULT STDMETHODCALLTYPE IEventHandler_CGamepadVtbl_InvokeAdded(__FIEventHandler_1_Windows__CGaming__CInput__CGamepad *This, IInspectable *sender, __x_ABI_CWindows_CGaming_CInput_CIGamepad *e)
{
    wgi_state.need_device_list_update = SDL_TRUE;
    return S_OK;
}

static HRESULT STDMETHODCALLTYPE IEventHandler_CGamepadVtbl_InvokeRemoved(__FIEventHandler_1_Windows__CGaming__CInput__CGamepad *This, IInspectable *sender, __x_ABI_CWindows_CGaming_CInput_CIGamepad *e)
{
    wgi_state.need_device_list_update = SDL_TRUE;
    return S_OK;
}

static __FIEventHandler_1_Windows__CGaming__CInput__CGamepadVtbl gamepad_added_vtbl = {
    IEventHandler_CGamepadVtbl_QueryInterface,
    IEventHandler_CGamepadVtbl_AddRef,
    IEventHandler_CGamepadVtbl_Release,
    IEventHandler_CGamepadVtbl_InvokeAdded
};
static GamepadDelegate gamepad_added = {
    { &gamepad_added_vtbl },
    { 1 }
};

static __FIEventHandler_1_Windows__CGaming__CInput__CGamepadVtbl gamepad_removed_vtbl = {
    IEventHandler_CGamepadVtbl_QueryInterface,
    IEventHandler_CGamepadVtbl_AddRef,
    IEventHandler_CGamepadVtbl_Release,
    IEventHandler_CGamepadVtbl_InvokeRemoved
};
static GamepadDelegate gamepad_removed = {
    { &gamepad_removed_vtbl },
    { 1 }
};

static void RAWINPUT_MarkWindowsGamingInputSlotUsed(WindowsGamingInputGamepadState *wgi_slot, RAWINPUT_DeviceContext *ctx)
{
    wgi_slot->used = SDL_TRUE;
    wgi_slot->correlated_context = ctx;
}

static void RAWINPUT_MarkWindowsGamingInputSlotFree(WindowsGamingInputGamepadState *wgi_slot)
{
    wgi_slot->used = SDL_FALSE;
    wgi_slot->correlated_context = NULL;
}

static SDL_bool RAWINPUT_MissingWindowsGamingInputSlot(void)
{
    int ii;
    for (ii = 0; ii < wgi_state.per_gamepad_count; ii++) {
        if (!wgi_state.per_gamepad[ii]->used) {
            return SDL_TRUE;
        }
    }
    return SDL_FALSE;
}

static void RAWINPUT_UpdateWindowsGamingInput(void)
{
    int ii;
    if (!wgi_state.gamepad_statics) {
        return;
    }

    if (!wgi_state.dirty) {
        return;
    }

    wgi_state.dirty = SDL_FALSE;

    if (wgi_state.need_device_list_update) {
        HRESULT hr;
        __FIVectorView_1_Windows__CGaming__CInput__CGamepad *gamepads;
        wgi_state.need_device_list_update = SDL_FALSE;
        for (ii = 0; ii < wgi_state.per_gamepad_count; ii++) {
            wgi_state.per_gamepad[ii]->connected = SDL_FALSE;
        }

        hr = __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics_get_Gamepads(wgi_state.gamepad_statics, &gamepads);
        if (SUCCEEDED(hr)) {
            unsigned int num_gamepads;

            hr = __FIVectorView_1_Windows__CGaming__CInput__CGamepad_get_Size(gamepads, &num_gamepads);
            if (SUCCEEDED(hr)) {
                unsigned int i;
                for (i = 0; i < num_gamepads; ++i) {
                    __x_ABI_CWindows_CGaming_CInput_CIGamepad *gamepad;

                    hr = __FIVectorView_1_Windows__CGaming__CInput__CGamepad_GetAt(gamepads, i, &gamepad);
                    if (SUCCEEDED(hr)) {
                        SDL_bool found = SDL_FALSE;
                        int jj;
                        for (jj = 0; jj < wgi_state.per_gamepad_count; jj++) {
                            if (wgi_state.per_gamepad[jj]->gamepad == gamepad) {
                                found = SDL_TRUE;
                                wgi_state.per_gamepad[jj]->connected = SDL_TRUE;
                                break;
                            }
                        }
                        if (!found) {
                            /* New device, add it */
                            WindowsGamingInputGamepadState *gamepad_state;

                            wgi_state.per_gamepad_count++;
                            wgi_state.per_gamepad = SDL_realloc(wgi_state.per_gamepad, sizeof(wgi_state.per_gamepad[0]) * wgi_state.per_gamepad_count);
                            if (!wgi_state.per_gamepad) {
                                SDL_OutOfMemory();
                                return;
                            }
                            gamepad_state = SDL_calloc(1, sizeof(*gamepad_state));
                            if (!gamepad_state) {
                                SDL_OutOfMemory();
                                return;
                            }
                            wgi_state.per_gamepad[wgi_state.per_gamepad_count - 1] = gamepad_state;
                            gamepad_state->gamepad = gamepad;
                            gamepad_state->connected = SDL_TRUE;
                        } else {
                            /* Already tracked */
                            __x_ABI_CWindows_CGaming_CInput_CIGamepad_Release(gamepad);
                        }
                    }
                }
                for (ii = wgi_state.per_gamepad_count - 1; ii >= 0; ii--) {
                    WindowsGamingInputGamepadState *gamepad_state = wgi_state.per_gamepad[ii];
                    if (!gamepad_state->connected) {
                        /* Device missing, must be disconnected */
                        if (gamepad_state->correlated_context) {
                            gamepad_state->correlated_context->wgi_correlated = SDL_FALSE;
                            gamepad_state->correlated_context->wgi_slot = NULL;
                        }
                        __x_ABI_CWindows_CGaming_CInput_CIGamepad_Release(gamepad_state->gamepad);
                        SDL_free(gamepad_state);
                        wgi_state.per_gamepad[ii] = wgi_state.per_gamepad[wgi_state.per_gamepad_count - 1];
                        --wgi_state.per_gamepad_count;
                    }
                }
            }
            __FIVectorView_1_Windows__CGaming__CInput__CGamepad_Release(gamepads);
        }
    } /* need_device_list_update */

    for (ii = 0; ii < wgi_state.per_gamepad_count; ii++) {
        HRESULT hr = __x_ABI_CWindows_CGaming_CInput_CIGamepad_GetCurrentReading(wgi_state.per_gamepad[ii]->gamepad, &wgi_state.per_gamepad[ii]->state);
        if (!SUCCEEDED(hr)) {
            wgi_state.per_gamepad[ii]->connected = SDL_FALSE; /* Not used by anything, currently */
        }
    }
}
static void RAWINPUT_InitWindowsGamingInput(RAWINPUT_DeviceContext *ctx)
{
    if (!SDL_GetHintBoolean(SDL_HINT_JOYSTICK_WGI, SDL_TRUE)) {
        return;
    }

    wgi_state.ref_count++;
    if (!wgi_state.initialized) {
        static const IID SDL_IID_IGamepadStatics = { 0x8BBCE529, 0xD49C, 0x39E9, { 0x95, 0x60, 0xE4, 0x7D, 0xDE, 0x96, 0xB7, 0xC8 } };
        HRESULT hr;

        if (FAILED(WIN_RoInitialize())) {
            return;
        }
        wgi_state.initialized = SDL_TRUE;
        wgi_state.dirty = SDL_TRUE;

        {
            typedef HRESULT(WINAPI * WindowsCreateStringReference_t)(PCWSTR sourceString, UINT32 length, HSTRING_HEADER * hstringHeader, HSTRING * string);
            typedef HRESULT(WINAPI * RoGetActivationFactory_t)(HSTRING activatableClassId, REFIID iid, void **factory);

#ifdef __WINRT__
            WindowsCreateStringReference_t WindowsCreateStringReferenceFunc = WindowsCreateStringReference;
            RoGetActivationFactory_t RoGetActivationFactoryFunc = RoGetActivationFactory;
#else
            WindowsCreateStringReference_t WindowsCreateStringReferenceFunc = (WindowsCreateStringReference_t)WIN_LoadComBaseFunction("WindowsCreateStringReference");
            RoGetActivationFactory_t RoGetActivationFactoryFunc = (RoGetActivationFactory_t)WIN_LoadComBaseFunction("RoGetActivationFactory");
#endif
            if (WindowsCreateStringReferenceFunc && RoGetActivationFactoryFunc) {
                PCWSTR pNamespace = L"Windows.Gaming.Input.Gamepad";
                HSTRING_HEADER hNamespaceStringHeader;
                HSTRING hNamespaceString;

                hr = WindowsCreateStringReferenceFunc(pNamespace, (UINT32)SDL_wcslen(pNamespace), &hNamespaceStringHeader, &hNamespaceString);
                if (SUCCEEDED(hr)) {
                    RoGetActivationFactoryFunc(hNamespaceString, &SDL_IID_IGamepadStatics, (void **)&wgi_state.gamepad_statics);
                }

                if (wgi_state.gamepad_statics) {
                    wgi_state.need_device_list_update = SDL_TRUE;

                    hr = __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics_add_GamepadAdded(wgi_state.gamepad_statics, &gamepad_added.iface, &wgi_state.gamepad_added_token);
                    if (!SUCCEEDED(hr)) {
                        SDL_SetError("add_GamepadAdded() failed: 0x%lx\n", hr);
                    }

                    hr = __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics_add_GamepadRemoved(wgi_state.gamepad_statics, &gamepad_removed.iface, &wgi_state.gamepad_removed_token);
                    if (!SUCCEEDED(hr)) {
                        SDL_SetError("add_GamepadRemoved() failed: 0x%lx\n", hr);
                    }
                }
            }
        }
    }
}

static SDL_bool RAWINPUT_WindowsGamingInputSlotMatches(const WindowsMatchState *state, WindowsGamingInputGamepadState *slot, SDL_bool xinput_correlated)
{
    Uint32 wgi_buttons = slot->state.Buttons;
    if ((wgi_buttons & 0x3FFF) == state->wgi_buttons
#ifdef SDL_JOYSTICK_RAWINPUT_MATCH_AXES
        && WindowsGamingInputAxesMatch(slot->state)
#endif
#ifdef SDL_JOYSTICK_RAWINPUT_MATCH_TRIGGERS
        // Don't try to match WGI triggers if getting values from XInput
        && (xinput_correlated || WindowsGamingInputTriggersMatch(slot->state))
#endif
    ) {
        return SDL_TRUE;
    }
    return SDL_FALSE;
}

static SDL_bool RAWINPUT_GuessWindowsGamingInputSlot(const WindowsMatchState *state, Uint8 *correlation_id, WindowsGamingInputGamepadState **slot, SDL_bool xinput_correlated)
{
    int match_count, user_index;
    WindowsGamingInputGamepadState *gamepad_state = NULL;

    /* If there is only one available slot, let's use that
     * That will be right most of the time, and uncorrelation will fix any bad guesses
     */
    match_count = 0;
    for (user_index = 0; user_index < wgi_state.per_gamepad_count; ++user_index) {
        gamepad_state = wgi_state.per_gamepad[user_index];
        if (gamepad_state->connected && !gamepad_state->used) {
            *slot = gamepad_state;
            ++match_count;
        }
    }
    if (match_count == 1) {
        *correlation_id = ++gamepad_state->correlation_id;
        return SDL_TRUE;
    }

    match_count = 0;
    for (user_index = 0; user_index < wgi_state.per_gamepad_count; ++user_index) {
        gamepad_state = wgi_state.per_gamepad[user_index];
        if (RAWINPUT_WindowsGamingInputSlotMatches(state, gamepad_state, xinput_correlated)) {
            ++match_count;
            *slot = gamepad_state;
            /* Incrementing correlation_id for any match, as negative evidence for others being correlated */
            *correlation_id = ++gamepad_state->correlation_id;
        }
    }
    /* Only return a match if we match exactly one, and we have some non-zero data (buttons or axes) that matched.
       Note that we're still invalidating *other* potential correlations if we have more than one match or we have no
       data. */
    if (match_count == 1 && state->any_data) {
        return SDL_TRUE;
    }
    return SDL_FALSE;
}

static void RAWINPUT_QuitWindowsGamingInput(RAWINPUT_DeviceContext *ctx)
{
    --wgi_state.ref_count;
    if (!wgi_state.ref_count && wgi_state.initialized) {
        int ii;
        for (ii = 0; ii < wgi_state.per_gamepad_count; ii++) {
            __x_ABI_CWindows_CGaming_CInput_CIGamepad_Release(wgi_state.per_gamepad[ii]->gamepad);
        }
        if (wgi_state.per_gamepad) {
            SDL_free(wgi_state.per_gamepad);
            wgi_state.per_gamepad = NULL;
        }
        wgi_state.per_gamepad_count = 0;
        if (wgi_state.gamepad_statics) {
            __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics_remove_GamepadAdded(wgi_state.gamepad_statics, wgi_state.gamepad_added_token);
            __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics_remove_GamepadRemoved(wgi_state.gamepad_statics, wgi_state.gamepad_removed_token);
            __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics_Release(wgi_state.gamepad_statics);
            wgi_state.gamepad_statics = NULL;
        }
        WIN_RoUninitialize();
        wgi_state.initialized = SDL_FALSE;
    }
}

#endif /* SDL_JOYSTICK_RAWINPUT_WGI */

static SDL_RAWINPUT_Device *RAWINPUT_AcquireDevice(SDL_RAWINPUT_Device *device)
{
    SDL_AtomicIncRef(&device->refcount);
    return device;
}

static void RAWINPUT_ReleaseDevice(SDL_RAWINPUT_Device *device)
{
#ifdef SDL_JOYSTICK_RAWINPUT_XINPUT
    if (device->joystick) {
        RAWINPUT_DeviceContext *ctx = device->joystick->hwdata;

        if (ctx->xinput_enabled && ctx->xinput_correlated) {
            RAWINPUT_MarkXInputSlotFree(ctx->xinput_slot);
            ctx->xinput_correlated = SDL_FALSE;
        }
    }
#endif /* SDL_JOYSTICK_RAWINPUT_XINPUT */

    if (SDL_AtomicDecRef(&device->refcount)) {
        SDL_free(device->preparsed_data);
        SDL_free(device->name);
        SDL_free(device->path);
        SDL_free(device);
    }
}

static SDL_RAWINPUT_Device *RAWINPUT_DeviceFromHandle(HANDLE hDevice)
{
    SDL_RAWINPUT_Device *curr;

    for (curr = SDL_RAWINPUT_devices; curr; curr = curr->next) {
        if (curr->hDevice == hDevice) {
            return curr;
        }
    }
    return NULL;
}

static int GetSteamVirtualGamepadSlot(Uint16 vendor_id, Uint16 product_id, const char *device_path)
{
    int slot = -1;

    // The format for the raw input device path is documented here:
    // https://partner.steamgames.com/doc/features/steam_controller/steam_input_gamepad_emulation_bestpractices
    if (vendor_id == USB_VENDOR_VALVE &&
        product_id == USB_PRODUCT_STEAM_VIRTUAL_GAMEPAD) {
        (void)SDL_sscanf(device_path, "\\\\.\\pipe\\HID#VID_045E&PID_028E&IG_00#%*X&%*X&%*X#%d#%*u", &slot);
    }
    return slot;
}

static void RAWINPUT_AddDevice(HANDLE hDevice)
{
#define CHECK(expression)  \
    {                      \
        if (!(expression)) \
            goto err;      \
    }
    SDL_RAWINPUT_Device *device = NULL;
    SDL_RAWINPUT_Device *curr, *last;
    RID_DEVICE_INFO rdi;
    UINT size;
    char dev_name[MAX_PATH];
    HANDLE hFile = INVALID_HANDLE_VALUE;

    /* Make sure we're not trying to add the same device twice */
    if (RAWINPUT_DeviceFromHandle(hDevice)) {
        return;
    }

    /* Figure out what kind of device it is */
    size = sizeof(rdi);
    CHECK(GetRawInputDeviceInfoA(hDevice, RIDI_DEVICEINFO, &rdi, &size) != (UINT)-1);
    CHECK(rdi.dwType == RIM_TYPEHID);

    /* Get the device "name" (HID Path) */
    size = SDL_arraysize(dev_name);
    CHECK(GetRawInputDeviceInfoA(hDevice, RIDI_DEVICENAME, dev_name, &size) != (UINT)-1);
    /* Only take XInput-capable devices */
    CHECK(SDL_strstr(dev_name, "IG_") != NULL);
#ifdef SDL_JOYSTICK_HIDAPI
    /* Don't take devices handled by HIDAPI */
    CHECK(!HIDAPI_IsDevicePresent((Uint16)rdi.hid.dwVendorId, (Uint16)rdi.hid.dwProductId, (Uint16)rdi.hid.dwVersionNumber, ""));
#endif
    device = (SDL_RAWINPUT_Device *)SDL_calloc(1, sizeof(SDL_RAWINPUT_Device));
    CHECK(device);
    device->hDevice = hDevice;
    device->vendor_id = (Uint16)rdi.hid.dwVendorId;
    device->product_id = (Uint16)rdi.hid.dwProductId;
    device->version = (Uint16)rdi.hid.dwVersionNumber;
    device->is_xinput = SDL_TRUE;
    device->is_xboxone = SDL_IsJoystickXboxOne(device->vendor_id, device->product_id);
    device->steam_virtual_gamepad_slot = GetSteamVirtualGamepadSlot(device->vendor_id, device->product_id, dev_name);

    /* Get HID Top-Level Collection Preparsed Data */
    size = 0;
    CHECK(GetRawInputDeviceInfoA(hDevice, RIDI_PREPARSEDDATA, NULL, &size) != (UINT)-1);
    device->preparsed_data = (PHIDP_PREPARSED_DATA)SDL_calloc(size, sizeof(BYTE));
    CHECK(device->preparsed_data);
    CHECK(GetRawInputDeviceInfoA(hDevice, RIDI_PREPARSEDDATA, device->preparsed_data, &size) != (UINT)-1);

    hFile = CreateFileA(dev_name, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
    CHECK(hFile != INVALID_HANDLE_VALUE);

    {
        char *manufacturer_string = NULL;
        char *product_string = NULL;
        WCHAR string[128];

        if (SDL_HidD_GetManufacturerString(hFile, string, sizeof(string))) {
            manufacturer_string = WIN_StringToUTF8W(string);
        }
        if (SDL_HidD_GetProductString(hFile, string, sizeof(string))) {
            product_string = WIN_StringToUTF8W(string);
        }

        device->name = SDL_CreateJoystickName(device->vendor_id, device->product_id, manufacturer_string, product_string);
        device->guid = SDL_CreateJoystickGUID(SDL_HARDWARE_BUS_USB, device->vendor_id, device->product_id, device->version, manufacturer_string, product_string, 'r', 0);

        if (manufacturer_string) {
            SDL_free(manufacturer_string);
        }
        if (product_string) {
            SDL_free(product_string);
        }
    }

    device->path = SDL_strdup(dev_name);

    CloseHandle(hFile);
    hFile = INVALID_HANDLE_VALUE;

    device->joystick_id = SDL_GetNextJoystickInstanceID();

#ifdef DEBUG_RAWINPUT
    SDL_Log("Adding RAWINPUT device '%s' VID 0x%.4x, PID 0x%.4x, version %d, handle 0x%.8x\n", device->name, device->vendor_id, device->product_id, device->version, device->hDevice);
#endif

    /* Add it to the list */
    RAWINPUT_AcquireDevice(device);
    for (curr = SDL_RAWINPUT_devices, last = NULL; curr; last = curr, curr = curr->next) {
    }
    if (last) {
        last->next = device;
    } else {
        SDL_RAWINPUT_devices = device;
    }

    ++SDL_RAWINPUT_numjoysticks;

    SDL_PrivateJoystickAdded(device->joystick_id);

    return;

err:
    if (hFile != INVALID_HANDLE_VALUE) {
        CloseHandle(hFile);
    }
    if (device) {
        if (device->name) {
            SDL_free(device->name);
        }
        if (device->path) {
            SDL_free(device->path);
        }
        SDL_free(device);
    }
#undef CHECK
}

static void RAWINPUT_DelDevice(SDL_RAWINPUT_Device *device, SDL_bool send_event)
{
    SDL_RAWINPUT_Device *curr, *last;
    for (curr = SDL_RAWINPUT_devices, last = NULL; curr; last = curr, curr = curr->next) {
        if (curr == device) {
            if (last) {
                last->next = curr->next;
            } else {
                SDL_RAWINPUT_devices = curr->next;
            }
            --SDL_RAWINPUT_numjoysticks;

            SDL_PrivateJoystickRemoved(device->joystick_id);

#ifdef DEBUG_RAWINPUT
            SDL_Log("Removing RAWINPUT device '%s' VID 0x%.4x, PID 0x%.4x, version %d, handle %p\n", device->name, device->vendor_id, device->product_id, device->version, device->hDevice);
#endif
            RAWINPUT_ReleaseDevice(device);
            return;
        }
    }
}

static void RAWINPUT_DetectDevices(void)
{
    UINT device_count = 0;

    if ((GetRawInputDeviceList(NULL, &device_count, sizeof(RAWINPUTDEVICELIST)) != -1) && device_count > 0) {
        PRAWINPUTDEVICELIST devices = NULL;
        UINT i;

        devices = (PRAWINPUTDEVICELIST)SDL_malloc(sizeof(RAWINPUTDEVICELIST) * device_count);
        if (devices) {
            device_count = GetRawInputDeviceList(devices, &device_count, sizeof(RAWINPUTDEVICELIST));
            if (device_count != (UINT)-1) {
                for (i = 0; i < device_count; ++i) {
                    RAWINPUT_AddDevice(devices[i].hDevice);
                }
            }
            SDL_free(devices);
        }
    }
}

static void RAWINPUT_RemoveDevices(void)
{
    while (SDL_RAWINPUT_devices) {
        RAWINPUT_DelDevice(SDL_RAWINPUT_devices, SDL_FALSE);
    }
    SDL_assert(SDL_RAWINPUT_numjoysticks == 0);
}

static int RAWINPUT_JoystickInit(void)
{
    SDL_assert(!SDL_RAWINPUT_inited);

    if (!SDL_GetHintBoolean(SDL_HINT_JOYSTICK_RAWINPUT, SDL_TRUE)) {
        return 0;
    }

    if (!WIN_IsWindowsVistaOrGreater()) {
        /* According to bug 6400, this doesn't work on Windows XP */
        return -1;
    }

    if (WIN_LoadHIDDLL() < 0) {
        return -1;
    }

    SDL_RAWINPUT_inited = SDL_TRUE;

    RAWINPUT_DetectDevices();

    return 0;
}

static int RAWINPUT_JoystickGetCount(void)
{
    return SDL_RAWINPUT_numjoysticks;
}

SDL_bool RAWINPUT_IsEnabled(void)
{
    return SDL_RAWINPUT_inited && !SDL_RAWINPUT_remote_desktop;
}

SDL_bool RAWINPUT_IsDevicePresent(Uint16 vendor_id, Uint16 product_id, Uint16 version, const char *name)
{
    SDL_RAWINPUT_Device *device;

    /* If we're being asked about a device, that means another API just detected one, so rescan */
#ifdef SDL_JOYSTICK_RAWINPUT_XINPUT
    xinput_device_change = SDL_TRUE;
#endif

    device = SDL_RAWINPUT_devices;
    while (device) {
        if (vendor_id == device->vendor_id && product_id == device->product_id) {
            return SDL_TRUE;
        }

        /* The Xbox 360 wireless controller shows up as product 0 in WGI.
           Try to match it to a Raw Input device via name or known product ID. */
        if (vendor_id == device->vendor_id && product_id == 0 &&
            ((name && SDL_strstr(device->name, name) != NULL) ||
             (device->vendor_id == USB_VENDOR_MICROSOFT &&
              device->product_id == USB_PRODUCT_XBOX360_XUSB_CONTROLLER))) {
            return SDL_TRUE;
        }

        /* The Xbox One controller shows up as a hardcoded raw input VID/PID */
        if (name && SDL_strcmp(name, "Xbox One Game Controller") == 0 &&
            device->vendor_id == USB_VENDOR_MICROSOFT &&
            device->product_id == USB_PRODUCT_XBOX_ONE_XBOXGIP_CONTROLLER) {
            return SDL_TRUE;
        }

        device = device->next;
    }
    return SDL_FALSE;
}

static void RAWINPUT_PostUpdate(void)
{
#ifdef SDL_JOYSTICK_RAWINPUT_MATCHING
    SDL_bool unmapped_guide_pressed = SDL_FALSE;

#ifdef SDL_JOYSTICK_RAWINPUT_WGI
    if (!wgi_state.dirty) {
        int ii;
        for (ii = 0; ii < wgi_state.per_gamepad_count; ii++) {
            WindowsGamingInputGamepadState *gamepad_state = wgi_state.per_gamepad[ii];
            if (!gamepad_state->used && (gamepad_state->state.Buttons & GamepadButtons_GUIDE)) {
                unmapped_guide_pressed = SDL_TRUE;
                break;
            }
        }
    }
    wgi_state.dirty = SDL_TRUE;
#endif

#ifdef SDL_JOYSTICK_RAWINPUT_XINPUT
    if (!xinput_state_dirty) {
        int ii;
        for (ii = 0; ii < SDL_arraysize(xinput_state); ii++) {
            if (xinput_state[ii].connected && !xinput_state[ii].used && (xinput_state[ii].state.Gamepad.wButtons & XINPUT_GAMEPAD_GUIDE)) {
                unmapped_guide_pressed = SDL_TRUE;
                break;
            }
        }
    }
    xinput_state_dirty = SDL_TRUE;
#endif

    if (unmapped_guide_pressed) {
        if (guide_button_candidate.joystick && !guide_button_candidate.last_joystick) {
            SDL_Joystick *joystick = guide_button_candidate.joystick;
            RAWINPUT_DeviceContext *ctx = joystick->hwdata;
            if (ctx->guide_hack) {
                int guide_button = joystick->nbuttons - 1;

                SDL_PrivateJoystickButton(guide_button_candidate.joystick, guide_button, SDL_PRESSED);
            }
            guide_button_candidate.last_joystick = guide_button_candidate.joystick;
        }
    } else if (guide_button_candidate.last_joystick) {
        SDL_Joystick *joystick = guide_button_candidate.last_joystick;
        RAWINPUT_DeviceContext *ctx = joystick->hwdata;
        if (ctx->guide_hack) {
            int guide_button = joystick->nbuttons - 1;

            SDL_PrivateJoystickButton(joystick, guide_button, SDL_RELEASED);
        }
        guide_button_candidate.last_joystick = NULL;
    }
    guide_button_candidate.joystick = NULL;

#endif /* SDL_JOYSTICK_RAWINPUT_MATCHING */
}

static void RAWINPUT_JoystickDetect(void)
{
    SDL_bool remote_desktop;

    if (!SDL_RAWINPUT_inited) {
        return;
    }

    remote_desktop = GetSystemMetrics(SM_REMOTESESSION) ? SDL_TRUE : SDL_FALSE;
    if (remote_desktop != SDL_RAWINPUT_remote_desktop) {
        SDL_RAWINPUT_remote_desktop = remote_desktop;

        WINDOWS_RAWINPUTEnabledChanged();

        if (remote_desktop) {
            RAWINPUT_RemoveDevices();
            WINDOWS_JoystickDetect();
        } else {
            WINDOWS_JoystickDetect();
            RAWINPUT_DetectDevices();
        }
    }
    RAWINPUT_PostUpdate();
}

static SDL_RAWINPUT_Device *RAWINPUT_GetDeviceByIndex(int device_index)
{
    SDL_RAWINPUT_Device *device = SDL_RAWINPUT_devices;
    while (device) {
        if (device_index == 0) {
            break;
        }
        --device_index;
        device = device->next;
    }
    return device;
}

static const char *RAWINPUT_JoystickGetDeviceName(int device_index)
{
    return RAWINPUT_GetDeviceByIndex(device_index)->name;
}

static const char *RAWINPUT_JoystickGetDevicePath(int device_index)
{
    return RAWINPUT_GetDeviceByIndex(device_index)->path;
}

static int RAWINPUT_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    return RAWINPUT_GetDeviceByIndex(device_index)->steam_virtual_gamepad_slot;
}

static int RAWINPUT_JoystickGetDevicePlayerIndex(int device_index)
{
    return -1;
}

static void RAWINPUT_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
}

static SDL_JoystickGUID RAWINPUT_JoystickGetDeviceGUID(int device_index)
{
    return RAWINPUT_GetDeviceByIndex(device_index)->guid;
}

static SDL_JoystickID RAWINPUT_JoystickGetDeviceInstanceID(int device_index)
{
    return RAWINPUT_GetDeviceByIndex(device_index)->joystick_id;
}

static int SDLCALL RAWINPUT_SortValueCaps(const void *A, const void *B)
{
    HIDP_VALUE_CAPS *capsA = (HIDP_VALUE_CAPS *)A;
    HIDP_VALUE_CAPS *capsB = (HIDP_VALUE_CAPS *)B;

    /* Sort by Usage for single values, or UsageMax for range of values */
    return (int)capsA->NotRange.Usage - capsB->NotRange.Usage;
}

static int RAWINPUT_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    SDL_RAWINPUT_Device *device = RAWINPUT_GetDeviceByIndex(device_index);
    RAWINPUT_DeviceContext *ctx;
    HIDP_CAPS caps;
    HIDP_BUTTON_CAPS *button_caps;
    HIDP_VALUE_CAPS *value_caps;
    ULONG i;

    ctx = (RAWINPUT_DeviceContext *)SDL_calloc(1, sizeof(RAWINPUT_DeviceContext));
    if (!ctx) {
        return SDL_OutOfMemory();
    }
    joystick->hwdata = ctx;

    ctx->device = RAWINPUT_AcquireDevice(device);
    device->joystick = joystick;

    if (device->is_xinput) {
        /* We'll try to get guide button and trigger axes from XInput */
#ifdef SDL_JOYSTICK_RAWINPUT_XINPUT
        xinput_device_change = SDL_TRUE;
        ctx->xinput_enabled = SDL_GetHintBoolean(SDL_HINT_JOYSTICK_RAWINPUT_CORRELATE_XINPUT, SDL_TRUE);
        if (ctx->xinput_enabled && (WIN_LoadXInputDLL() < 0 || !XINPUTGETSTATE)) {
            ctx->xinput_enabled = SDL_FALSE;
        }
        ctx->xinput_slot = XUSER_INDEX_ANY;
#endif
#ifdef SDL_JOYSTICK_RAWINPUT_WGI
        RAWINPUT_InitWindowsGamingInput(ctx);
#endif
    }

    ctx->is_xinput = device->is_xinput;
    ctx->is_xboxone = device->is_xboxone;
#ifdef SDL_JOYSTICK_RAWINPUT_MATCHING
    ctx->match_state = 0x0000008800000000ULL; /* Trigger axes at rest */
#endif
    ctx->preparsed_data = device->preparsed_data;
    ctx->max_data_length = SDL_HidP_MaxDataListLength(HidP_Input, ctx->preparsed_data);
    ctx->data = (HIDP_DATA *)SDL_malloc(ctx->max_data_length * sizeof(*ctx->data));
    if (!ctx->data) {
        RAWINPUT_JoystickClose(joystick);
        return SDL_OutOfMemory();
    }

    if (SDL_HidP_GetCaps(ctx->preparsed_data, &caps) != HIDP_STATUS_SUCCESS) {
        RAWINPUT_JoystickClose(joystick);
        return SDL_SetError("Couldn't get device capabilities");
    }

    button_caps = SDL_stack_alloc(HIDP_BUTTON_CAPS, caps.NumberInputButtonCaps);
    if (SDL_HidP_GetButtonCaps(HidP_Input, button_caps, &caps.NumberInputButtonCaps, ctx->preparsed_data) != HIDP_STATUS_SUCCESS) {
        RAWINPUT_JoystickClose(joystick);
        return SDL_SetError("Couldn't get device button capabilities");
    }

    value_caps = SDL_stack_alloc(HIDP_VALUE_CAPS, caps.NumberInputValueCaps);
    if (SDL_HidP_GetValueCaps(HidP_Input, value_caps, &caps.NumberInputValueCaps, ctx->preparsed_data) != HIDP_STATUS_SUCCESS) {
        RAWINPUT_JoystickClose(joystick);
        SDL_stack_free(button_caps);
        return SDL_SetError("Couldn't get device value capabilities");
    }

    /* Sort the axes by usage, so X comes before Y, etc. */
    SDL_qsort(value_caps, caps.NumberInputValueCaps, sizeof(*value_caps), RAWINPUT_SortValueCaps);

    for (i = 0; i < caps.NumberInputButtonCaps; ++i) {
        HIDP_BUTTON_CAPS *cap = &button_caps[i];

        if (cap->UsagePage == USB_USAGEPAGE_BUTTON) {
            int count;

            if (cap->IsRange) {
                count = 1 + (cap->Range.DataIndexMax - cap->Range.DataIndexMin);
            } else {
                count = 1;
            }

            joystick->nbuttons += count;
        }
    }

    if (joystick->nbuttons > 0) {
        int button_index = 0;

        ctx->button_indices = (USHORT *)SDL_malloc(joystick->nbuttons * sizeof(*ctx->button_indices));
        if (!ctx->button_indices) {
            RAWINPUT_JoystickClose(joystick);
            SDL_stack_free(value_caps);
            SDL_stack_free(button_caps);
            return SDL_OutOfMemory();
        }

        for (i = 0; i < caps.NumberInputButtonCaps; ++i) {
            HIDP_BUTTON_CAPS *cap = &button_caps[i];

            if (cap->UsagePage == USB_USAGEPAGE_BUTTON) {
                if (cap->IsRange) {
                    int j, count = 1 + (cap->Range.DataIndexMax - cap->Range.DataIndexMin);

                    for (j = 0; j < count; ++j) {
                        ctx->button_indices[button_index++] = cap->Range.DataIndexMin + j;
                    }
                } else {
                    ctx->button_indices[button_index++] = cap->NotRange.DataIndex;
                }
            }
        }
    }
    if (ctx->is_xinput && joystick->nbuttons == 10) {
        ctx->guide_hack = SDL_TRUE;
        joystick->nbuttons += 1;
    }

    SDL_stack_free(button_caps);

    for (i = 0; i < caps.NumberInputValueCaps; ++i) {
        HIDP_VALUE_CAPS *cap = &value_caps[i];

        if (cap->IsRange) {
            continue;
        }

        if (ctx->trigger_hack && cap->NotRange.Usage == USB_USAGE_GENERIC_Z) {
            continue;
        }

        if (cap->NotRange.Usage == USB_USAGE_GENERIC_HAT) {
            joystick->nhats += 1;
            continue;
        }

        if (ctx->is_xinput && cap->NotRange.Usage == USB_USAGE_GENERIC_Z) {
            continue;
        }

        joystick->naxes += 1;
    }

    if (joystick->naxes > 0) {
        int axis_index = 0;

        ctx->axis_indices = (USHORT *)SDL_malloc(joystick->naxes * sizeof(*ctx->axis_indices));
        if (!ctx->axis_indices) {
            RAWINPUT_JoystickClose(joystick);
            SDL_stack_free(value_caps);
            return SDL_OutOfMemory();
        }

        for (i = 0; i < caps.NumberInputValueCaps; ++i) {
            HIDP_VALUE_CAPS *cap = &value_caps[i];

            if (cap->IsRange) {
                continue;
            }

            if (cap->NotRange.Usage == USB_USAGE_GENERIC_HAT) {
                continue;
            }

            if (ctx->is_xinput && cap->NotRange.Usage == USB_USAGE_GENERIC_Z) {
                ctx->trigger_hack = SDL_TRUE;
                ctx->trigger_hack_index = cap->NotRange.DataIndex;
                continue;
            }

            ctx->axis_indices[axis_index++] = cap->NotRange.DataIndex;
        }
    }
    if (ctx->trigger_hack) {
        joystick->naxes += 2;
    }

    if (joystick->nhats > 0) {
        int hat_index = 0;

        ctx->hat_indices = (USHORT *)SDL_malloc(joystick->nhats * sizeof(*ctx->hat_indices));
        if (!ctx->hat_indices) {
            RAWINPUT_JoystickClose(joystick);
            SDL_stack_free(value_caps);
            return SDL_OutOfMemory();
        }

        for (i = 0; i < caps.NumberInputValueCaps; ++i) {
            HIDP_VALUE_CAPS *cap = &value_caps[i];

            if (cap->IsRange) {
                continue;
            }

            if (cap->NotRange.Usage != USB_USAGE_GENERIC_HAT) {
                continue;
            }

            ctx->hat_indices[hat_index++] = cap->NotRange.DataIndex;
        }
    }

    SDL_stack_free(value_caps);

    joystick->epowerlevel = SDL_JOYSTICK_POWER_UNKNOWN;

    return 0;
}

static int RAWINPUT_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
#if defined(SDL_JOYSTICK_RAWINPUT_WGI) || defined(SDL_JOYSTICK_RAWINPUT_XINPUT)
    RAWINPUT_DeviceContext *ctx = joystick->hwdata;
#endif
    SDL_bool rumbled = SDL_FALSE;

#ifdef SDL_JOYSTICK_RAWINPUT_XINPUT
    /* Prefer XInput over WGI because it allows rumble in the background */
    if (!rumbled && ctx->xinput_correlated) {
        XINPUT_VIBRATION XVibration;

        if (!XINPUTSETSTATE) {
            return SDL_Unsupported();
        }

        XVibration.wLeftMotorSpeed = low_frequency_rumble;
        XVibration.wRightMotorSpeed = high_frequency_rumble;
        if (XINPUTSETSTATE(ctx->xinput_slot, &XVibration) == ERROR_SUCCESS) {
            rumbled = SDL_TRUE;
        } else {
            return SDL_SetError("XInputSetState() failed");
        }
    }
#endif /* SDL_JOYSTICK_RAWINPUT_XINPUT */

#ifdef SDL_JOYSTICK_RAWINPUT_WGI
    if (!rumbled && ctx->wgi_correlated) {
        WindowsGamingInputGamepadState *gamepad_state = ctx->wgi_slot;
        HRESULT hr;
        gamepad_state->vibration.LeftMotor = (DOUBLE)low_frequency_rumble / SDL_MAX_UINT16;
        gamepad_state->vibration.RightMotor = (DOUBLE)high_frequency_rumble / SDL_MAX_UINT16;
        hr = __x_ABI_CWindows_CGaming_CInput_CIGamepad_put_Vibration(gamepad_state->gamepad, gamepad_state->vibration);
        if (SUCCEEDED(hr)) {
            rumbled = SDL_TRUE;
        }
    }
#endif

    if (!rumbled) {
#if defined(SDL_JOYSTICK_RAWINPUT_WGI) || defined(SDL_JOYSTICK_RAWINPUT_XINPUT)
        return SDL_SetError("Controller isn't correlated yet, try hitting a button first");
#else
        return SDL_Unsupported();
#endif
    }
    return 0;
}

static int RAWINPUT_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
#if defined(SDL_JOYSTICK_RAWINPUT_WGI)
    RAWINPUT_DeviceContext *ctx = joystick->hwdata;

    if (ctx->wgi_correlated) {
        WindowsGamingInputGamepadState *gamepad_state = ctx->wgi_slot;
        HRESULT hr;
        gamepad_state->vibration.LeftTrigger = (DOUBLE)left_rumble / SDL_MAX_UINT16;
        gamepad_state->vibration.RightTrigger = (DOUBLE)right_rumble / SDL_MAX_UINT16;
        hr = __x_ABI_CWindows_CGaming_CInput_CIGamepad_put_Vibration(gamepad_state->gamepad, gamepad_state->vibration);
        if (!SUCCEEDED(hr)) {
            return SDL_SetError("Setting vibration failed: 0x%lx\n", hr);
        }
        return 0;
    } else {
        return SDL_SetError("Controller isn't correlated yet, try hitting a button first");
    }
#else
    return SDL_Unsupported();
#endif
}

static Uint32 RAWINPUT_JoystickGetCapabilities(SDL_Joystick *joystick)
{
    Uint32 result = 0;
#if defined(SDL_JOYSTICK_RAWINPUT_XINPUT) || defined(SDL_JOYSTICK_RAWINPUT_WGI)
    RAWINPUT_DeviceContext *ctx = joystick->hwdata;

#ifdef SDL_JOYSTICK_RAWINPUT_XINPUT
    if (ctx->is_xinput) {
        result |= SDL_JOYCAP_RUMBLE;
    }
#endif
#ifdef SDL_JOYSTICK_RAWINPUT_WGI
    if (ctx->is_xinput) {
        result |= SDL_JOYCAP_RUMBLE;

        if (ctx->is_xboxone) {
            result |= SDL_JOYCAP_RUMBLE_TRIGGERS;
        }
    }
#endif
#endif /**/

    return result;
}

static int RAWINPUT_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static int RAWINPUT_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static int RAWINPUT_JoystickSetSensorsEnabled(SDL_Joystick *joystick, SDL_bool enabled)
{
    return SDL_Unsupported();
}

static HIDP_DATA *GetData(USHORT index, HIDP_DATA *data, ULONG length)
{
    ULONG i;

    /* Check to see if the data is at the expected offset */
    if (index < length && data[index].DataIndex == index) {
        return &data[index];
    }

    /* Loop through the data to find it */
    for (i = 0; i < length; ++i) {
        if (data[i].DataIndex == index) {
            return &data[i];
        }
    }
    return NULL;
}

/* This is the packet format for Xbox 360 and Xbox One controllers on Windows,
   however with this interface there is no rumble support, no guide button,
   and the left and right triggers are tied together as a single axis.

   We use XInput and Windows.Gaming.Input to make up for these shortcomings.
 */
static void RAWINPUT_HandleStatePacket(SDL_Joystick *joystick, Uint8 *data, int size)
{
    RAWINPUT_DeviceContext *ctx = joystick->hwdata;
#ifdef SDL_JOYSTICK_RAWINPUT_MATCHING
    /* Map new buttons and axes into game controller controls */
    static const int button_map[] = {
        SDL_CONTROLLER_BUTTON_A,
        SDL_CONTROLLER_BUTTON_B,
        SDL_CONTROLLER_BUTTON_X,
        SDL_CONTROLLER_BUTTON_Y,
        SDL_CONTROLLER_BUTTON_LEFTSHOULDER,
        SDL_CONTROLLER_BUTTON_RIGHTSHOULDER,
        SDL_CONTROLLER_BUTTON_BACK,
        SDL_CONTROLLER_BUTTON_START,
        SDL_CONTROLLER_BUTTON_LEFTSTICK,
        SDL_CONTROLLER_BUTTON_RIGHTSTICK
    };
#define HAT_MASK ((1 << SDL_CONTROLLER_BUTTON_DPAD_UP) | (1 << SDL_CONTROLLER_BUTTON_DPAD_DOWN) | (1 << SDL_CONTROLLER_BUTTON_DPAD_LEFT) | (1 << SDL_CONTROLLER_BUTTON_DPAD_RIGHT))
    static const int hat_map[] = {
        0,
        (1 << SDL_CONTROLLER_BUTTON_DPAD_UP),
        (1 << SDL_CONTROLLER_BUTTON_DPAD_UP) | (1 << SDL_CONTROLLER_BUTTON_DPAD_RIGHT),
        (1 << SDL_CONTROLLER_BUTTON_DPAD_RIGHT),
        (1 << SDL_CONTROLLER_BUTTON_DPAD_DOWN) | (1 << SDL_CONTROLLER_BUTTON_DPAD_RIGHT),
        (1 << SDL_CONTROLLER_BUTTON_DPAD_DOWN),
        (1 << SDL_CONTROLLER_BUTTON_DPAD_DOWN) | (1 << SDL_CONTROLLER_BUTTON_DPAD_LEFT),
        (1 << SDL_CONTROLLER_BUTTON_DPAD_LEFT),
        (1 << SDL_CONTROLLER_BUTTON_DPAD_UP) | (1 << SDL_CONTROLLER_BUTTON_DPAD_LEFT),
        0,
    };
    Uint64 match_state = ctx->match_state;
    /* Update match_state with button bit, then fall through */
#define SDL_PrivateJoystickButton(joystick, button, state)                  \
    if (button < SDL_arraysize(button_map)) {                               \
        Uint64 button_bit = 1ull << button_map[button];                     \
        match_state = (match_state & ~button_bit) | (button_bit * (state)); \
    }                                                                       \
    SDL_PrivateJoystickButton(joystick, button, state)
#ifdef SDL_JOYSTICK_RAWINPUT_MATCH_AXES
    /* Grab high 4 bits of value, then fall through */
#define AddAxisToMatchState(axis, value)                                                                    \
    {                                                                                                       \
        match_state = (match_state & ~(0xFull << (4 * axis + 16))) | ((value)&0xF000ull) << (4 * axis + 4); \
    }
#define SDL_PrivateJoystickAxis(joystick, axis, value) \
    if (axis < 4)                                      \
        AddAxisToMatchState(axis, value);              \
    SDL_PrivateJoystickAxis(joystick, axis, value)
#endif
#endif /* SDL_JOYSTICK_RAWINPUT_MATCHING */

    ULONG data_length = ctx->max_data_length;
    int i;
    int nbuttons = joystick->nbuttons - (ctx->guide_hack * 1);
    int naxes = joystick->naxes - (ctx->trigger_hack * 2);
    int nhats = joystick->nhats;
    Uint32 button_mask = 0;

    if (SDL_HidP_GetData(HidP_Input, ctx->data, &data_length, ctx->preparsed_data, (PCHAR)data, size) != HIDP_STATUS_SUCCESS) {
        return;
    }

    for (i = 0; i < nbuttons; ++i) {
        HIDP_DATA *item = GetData(ctx->button_indices[i], ctx->data, data_length);
        if (item && item->On) {
            button_mask |= (1 << i);
        }
    }
    for (i = 0; i < nbuttons; ++i) {
        SDL_PrivateJoystickButton(joystick, i, (button_mask & (1 << i)) ? SDL_PRESSED : SDL_RELEASED);
    }

    for (i = 0; i < naxes; ++i) {
        HIDP_DATA *item = GetData(ctx->axis_indices[i], ctx->data, data_length);
        if (item) {
            Sint16 axis = (int)(Uint16)item->RawValue - 0x8000;
            SDL_PrivateJoystickAxis(joystick, i, axis);
        }
    }

    for (i = 0; i < nhats; ++i) {
        HIDP_DATA *item = GetData(ctx->hat_indices[i], ctx->data, data_length);
        if (item) {
            Uint8 hat = SDL_HAT_CENTERED;
            const Uint8 hat_states[] = {
                SDL_HAT_CENTERED,
                SDL_HAT_UP,
                SDL_HAT_UP | SDL_HAT_RIGHT,
                SDL_HAT_RIGHT,
                SDL_HAT_DOWN | SDL_HAT_RIGHT,
                SDL_HAT_DOWN,
                SDL_HAT_DOWN | SDL_HAT_LEFT,
                SDL_HAT_LEFT,
                SDL_HAT_UP | SDL_HAT_LEFT,
                SDL_HAT_CENTERED,
            };
            ULONG state = item->RawValue;

            if (state < SDL_arraysize(hat_states)) {
#ifdef SDL_JOYSTICK_RAWINPUT_MATCHING
                match_state = (match_state & ~HAT_MASK) | hat_map[state];
#endif
                hat = hat_states[state];
            }
            SDL_PrivateJoystickHat(joystick, i, hat);
        }
    }

#ifdef SDL_PrivateJoystickButton
#undef SDL_PrivateJoystickButton
#endif
#ifdef SDL_PrivateJoystickAxis
#undef SDL_PrivateJoystickAxis
#endif

#ifdef SDL_JOYSTICK_RAWINPUT_MATCH_TRIGGERS
#define AddTriggerToMatchState(axis, value)                                          \
    {                                                                                \
        int match_axis = axis + SDL_JOYSTICK_RAWINPUT_MATCH_COUNT - joystick->naxes; \
        AddAxisToMatchState(match_axis, value);                                      \
    }
#endif /* SDL_JOYSTICK_RAWINPUT_MATCH_TRIGGERS */

    if (ctx->trigger_hack) {
        SDL_bool has_trigger_data = SDL_FALSE;
        int left_trigger = joystick->naxes - 2;
        int right_trigger = joystick->naxes - 1;

#ifdef SDL_JOYSTICK_RAWINPUT_XINPUT
        /* Prefer XInput over WindowsGamingInput, it continues to provide data in the background */
        if (!has_trigger_data && ctx->xinput_enabled && ctx->xinput_correlated) {
            has_trigger_data = SDL_TRUE;
        }
#endif /* SDL_JOYSTICK_RAWINPUT_XINPUT */

#ifdef SDL_JOYSTICK_RAWINPUT_WGI
        if (!has_trigger_data && ctx->wgi_correlated) {
            has_trigger_data = SDL_TRUE;
        }
#endif /* SDL_JOYSTICK_RAWINPUT_WGI */

#ifndef SDL_JOYSTICK_RAWINPUT_MATCH_TRIGGERS
        if (!has_trigger_data)
#endif
        {
            HIDP_DATA *item = GetData(ctx->trigger_hack_index, ctx->data, data_length);
            if (item) {
                Sint16 value = (int)(Uint16)item->RawValue - 0x8000;
                Sint16 left_value = (value > 0) ? (value * 2 - 32767) : SDL_MIN_SINT16;
                Sint16 right_value = (value < 0) ? (-value * 2 - 32769) : SDL_MIN_SINT16;

#ifdef SDL_JOYSTICK_RAWINPUT_MATCH_TRIGGERS
                AddTriggerToMatchState(left_trigger, left_value);
                AddTriggerToMatchState(right_trigger, right_value);
                if (!has_trigger_data)
#endif /* SDL_JOYSTICK_RAWINPUT_MATCH_TRIGGERS */
                {
                    SDL_PrivateJoystickAxis(joystick, left_trigger, left_value);
                    SDL_PrivateJoystickAxis(joystick, right_trigger, right_value);
                }
            }
        }
    }

#ifdef AddAxisToMatchState
#undef AddAxisToMatchState
#endif
#ifdef AddTriggerToMatchState
#undef AddTriggerToMatchState
#endif

#ifdef SDL_JOYSTICK_RAWINPUT_MATCHING
    if (ctx->is_xinput) {
        ctx->match_state = match_state;
        ctx->last_state_packet = SDL_GetTicks();
    }
#endif
}

static void RAWINPUT_UpdateOtherAPIs(SDL_Joystick *joystick)
{
#ifdef SDL_JOYSTICK_RAWINPUT_MATCHING
    RAWINPUT_DeviceContext *ctx = joystick->hwdata;
    SDL_bool has_trigger_data = SDL_FALSE;
    SDL_bool correlated = SDL_FALSE;
    WindowsMatchState match_state_xinput;
    int guide_button = joystick->nbuttons - 1;
    int left_trigger = joystick->naxes - 2;
    int right_trigger = joystick->naxes - 1;
#ifdef SDL_JOYSTICK_RAWINPUT_WGI
    SDL_bool xinput_correlated;
#endif

    RAWINPUT_FillMatchState(&match_state_xinput, ctx->match_state);

#ifdef SDL_JOYSTICK_RAWINPUT_WGI
#ifdef SDL_JOYSTICK_RAWINPUT_XINPUT
    xinput_correlated = ctx->xinput_correlated;
#else
    xinput_correlated = SDL_FALSE;
#endif
    /* Parallel logic to WINDOWS_XINPUT below */
    RAWINPUT_UpdateWindowsGamingInput();
    if (ctx->wgi_correlated &&
        !joystick->low_frequency_rumble && !joystick->high_frequency_rumble &&
        !joystick->left_trigger_rumble && !joystick->right_trigger_rumble) {
        /* We have been previously correlated, ensure we are still matching, see comments in XINPUT section */
        if (RAWINPUT_WindowsGamingInputSlotMatches(&match_state_xinput, ctx->wgi_slot, xinput_correlated)) {
            ctx->wgi_uncorrelate_count = 0;
        } else {
            ++ctx->wgi_uncorrelate_count;
            /* Only un-correlate if this is consistent over multiple Update() calls - the timing of polling/event
              pumping can easily cause this to uncorrelate for a frame.  2 seemed reliable in my testing, but
              let's set it to 5 to be safe.  An incorrect un-correlation will simply result in lower precision
              triggers for a frame. */
            if (ctx->wgi_uncorrelate_count >= 5) {
#ifdef DEBUG_RAWINPUT
                SDL_Log("UN-Correlated joystick %d to WindowsGamingInput device #%d\n", joystick->instance_id, ctx->wgi_slot);
#endif
                RAWINPUT_MarkWindowsGamingInputSlotFree(ctx->wgi_slot);
                ctx->wgi_correlated = SDL_FALSE;
                ctx->wgi_correlation_count = 0;
                /* Force release of Guide button, it can't possibly be down on this device now. */
                /* It gets left down if we were actually correlated incorrectly and it was released on the WindowsGamingInput
                  device but we didn't get a state packet. */
                if (ctx->guide_hack) {
                    SDL_PrivateJoystickButton(joystick, guide_button, SDL_RELEASED);
                }
            }
        }
    }
    if (!ctx->wgi_correlated) {
        SDL_bool new_correlation_count = 0;
        if (RAWINPUT_MissingWindowsGamingInputSlot()) {
            Uint8 correlation_id = 0;
            WindowsGamingInputGamepadState *slot_idx = NULL;
            if (RAWINPUT_GuessWindowsGamingInputSlot(&match_state_xinput, &correlation_id, &slot_idx, xinput_correlated)) {
                /* we match exactly one WindowsGamingInput device */
                /* Probably can do without wgi_correlation_count, just check and clear wgi_slot to NULL, unless we need
                   even more frames to be sure. */
                if (ctx->wgi_correlation_count && ctx->wgi_slot == slot_idx) {
                    /* was correlated previously, and still the same device */
                    if (ctx->wgi_correlation_id + 1 == correlation_id) {
                        /* no one else was correlated in the meantime */
                        new_correlation_count = ctx->wgi_correlation_count + 1;
                        if (new_correlation_count == 2) {
                            /* correlation stayed steady and uncontested across multiple frames, guaranteed match */
                            ctx->wgi_correlated = SDL_TRUE;
#ifdef DEBUG_RAWINPUT
                            SDL_Log("Correlated joystick %d to WindowsGamingInput device #%d\n", joystick->instance_id, slot_idx);
#endif
                            correlated = SDL_TRUE;
                            RAWINPUT_MarkWindowsGamingInputSlotUsed(ctx->wgi_slot, ctx);
                            /* If the generalized Guide button was using us, it doesn't need to anymore */
                            if (guide_button_candidate.joystick == joystick) {
                                guide_button_candidate.joystick = NULL;
                            }
                            if (guide_button_candidate.last_joystick == joystick) {
                                guide_button_candidate.last_joystick = NULL;
                            }
                        }
                    } else {
                        /* someone else also possibly correlated to this device, start over */
                        new_correlation_count = 1;
                    }
                } else {
                    /* new possible correlation */
                    new_correlation_count = 1;
                    ctx->wgi_slot = slot_idx;
                }
                ctx->wgi_correlation_id = correlation_id;
            } else {
                /* Match multiple WindowsGamingInput devices, or none (possibly due to no buttons pressed) */
            }
        }
        ctx->wgi_correlation_count = new_correlation_count;
    } else {
        correlated = SDL_TRUE;
    }
#endif /* SDL_JOYSTICK_RAWINPUT_WGI */

#ifdef SDL_JOYSTICK_RAWINPUT_XINPUT
    /* Parallel logic to WINDOWS_GAMING_INPUT above */
    if (ctx->xinput_enabled) {
        RAWINPUT_UpdateXInput();
        if (ctx->xinput_correlated &&
            !joystick->low_frequency_rumble && !joystick->high_frequency_rumble) {
            /* We have been previously correlated, ensure we are still matching */
            /* This is required to deal with two (mostly) un-preventable mis-correlation situations:
              A) Since the HID data stream does not provide an initial state (but polling XInput does), if we open
                 5 controllers (#1-4 XInput mapped, #5 is not), and controller 1 had the A button down (and we don't
                 know), and the user presses A on controller #5, we'll see exactly 1 controller with A down (#5) and
                 exactly 1 XInput device with A down (#1), and incorrectly correlate.  This code will then un-correlate
                 when A is released from either controller #1 or #5.
              B) Since the app may not open all controllers, we could have a similar situation where only controller #5
                 is opened, and the user holds A on controllers #1 and #5 simultaneously - again we see only 1 controller
                 with A down and 1 XInput device with A down, and incorrectly correlate.  This should be very unusual
                 (only when apps do not open all controllers, yet are listening to Guide button presses, yet
                 for some reason want to ignore guide button presses on the un-opened controllers, yet users are
                 pressing buttons on the unopened controllers), and will resolve itself when either button is released
                 and we un-correlate.  We could prevent this by processing the state packets for *all* controllers,
                 even un-opened ones, as that would allow more precise correlation.
            */
            if (RAWINPUT_XInputSlotMatches(&match_state_xinput, ctx->xinput_slot)) {
                ctx->xinput_uncorrelate_count = 0;
            } else {
                ++ctx->xinput_uncorrelate_count;
                /* Only un-correlate if this is consistent over multiple Update() calls - the timing of polling/event
                  pumping can easily cause this to uncorrelate for a frame.  2 seemed reliable in my testing, but
                  let's set it to 5 to be safe.  An incorrect un-correlation will simply result in lower precision
                  triggers for a frame. */
                if (ctx->xinput_uncorrelate_count >= 5) {
#ifdef DEBUG_RAWINPUT
                    SDL_Log("UN-Correlated joystick %d to XInput device #%d\n", joystick->instance_id, ctx->xinput_slot);
#endif
                    RAWINPUT_MarkXInputSlotFree(ctx->xinput_slot);
                    ctx->xinput_correlated = SDL_FALSE;
                    ctx->xinput_correlation_count = 0;
                    /* Force release of Guide button, it can't possibly be down on this device now. */
                    /* It gets left down if we were actually correlated incorrectly and it was released on the XInput
                      device but we didn't get a state packet. */
                    if (ctx->guide_hack) {
                        SDL_PrivateJoystickButton(joystick, guide_button, SDL_RELEASED);
                    }
                }
            }
        }
        if (!ctx->xinput_correlated) {
            Uint8 new_correlation_count = 0;
            if (RAWINPUT_MissingXInputSlot()) {
                Uint8 correlation_id = 0;
                Uint8 slot_idx = 0;
                if (RAWINPUT_GuessXInputSlot(&match_state_xinput, &correlation_id, &slot_idx)) {
                    /* we match exactly one XInput device */
                    /* Probably can do without xinput_correlation_count, just check and clear xinput_slot to ANY, unless
                       we need even more frames to be sure */
                    if (ctx->xinput_correlation_count && ctx->xinput_slot == slot_idx) {
                        /* was correlated previously, and still the same device */
                        if (ctx->xinput_correlation_id + 1 == correlation_id) {
                            /* no one else was correlated in the meantime */
                            new_correlation_count = ctx->xinput_correlation_count + 1;
                            if (new_correlation_count == 2) {
                                /* correlation stayed steady and uncontested across multiple frames, guaranteed match */
                                ctx->xinput_correlated = SDL_TRUE;
#ifdef DEBUG_RAWINPUT
                                SDL_Log("Correlated joystick %d to XInput device #%d\n", joystick->instance_id, slot_idx);
#endif
                                correlated = SDL_TRUE;
                                RAWINPUT_MarkXInputSlotUsed(ctx->xinput_slot);
                                /* If the generalized Guide button was using us, it doesn't need to anymore */
                                if (guide_button_candidate.joystick == joystick) {
                                    guide_button_candidate.joystick = NULL;
                                }
                                if (guide_button_candidate.last_joystick == joystick) {
                                    guide_button_candidate.last_joystick = NULL;
                                }
                            }
                        } else {
                            /* someone else also possibly correlated to this device, start over */
                            new_correlation_count = 1;
                        }
                    } else {
                        /* new possible correlation */
                        new_correlation_count = 1;
                        ctx->xinput_slot = slot_idx;
                    }
                    ctx->xinput_correlation_id = correlation_id;
                } else {
                    /* Match multiple XInput devices, or none (possibly due to no buttons pressed) */
                }
            }
            ctx->xinput_correlation_count = new_correlation_count;
        } else {
            correlated = SDL_TRUE;
        }
    }
#endif /* SDL_JOYSTICK_RAWINPUT_XINPUT */

    /* Poll for trigger data once (not per-state-packet) */
#ifdef SDL_JOYSTICK_RAWINPUT_XINPUT
    /* Prefer XInput over WindowsGamingInput, it continues to provide data in the background */
    if (!has_trigger_data && ctx->xinput_enabled && ctx->xinput_correlated) {
        RAWINPUT_UpdateXInput();
        if (xinput_state[ctx->xinput_slot].connected) {
            XINPUT_BATTERY_INFORMATION_EX *battery_info = &xinput_state[ctx->xinput_slot].battery;

            if (ctx->guide_hack) {
                SDL_PrivateJoystickButton(joystick, guide_button, (xinput_state[ctx->xinput_slot].state.Gamepad.wButtons & XINPUT_GAMEPAD_GUIDE) ? SDL_PRESSED : SDL_RELEASED);
            }
            if (ctx->trigger_hack) {
                SDL_PrivateJoystickAxis(joystick, left_trigger, ((int)xinput_state[ctx->xinput_slot].state.Gamepad.bLeftTrigger * 257) - 32768);
                SDL_PrivateJoystickAxis(joystick, right_trigger, ((int)xinput_state[ctx->xinput_slot].state.Gamepad.bRightTrigger * 257) - 32768);
            }
            has_trigger_data = SDL_TRUE;

            if (battery_info->BatteryType != BATTERY_TYPE_UNKNOWN &&
                battery_info->BatteryType != BATTERY_TYPE_DISCONNECTED) {
                SDL_JoystickPowerLevel ePowerLevel = SDL_JOYSTICK_POWER_UNKNOWN;
                if (battery_info->BatteryType == BATTERY_TYPE_WIRED) {
                    ePowerLevel = SDL_JOYSTICK_POWER_WIRED;
                } else {
                    switch (battery_info->BatteryLevel) {
                    case BATTERY_LEVEL_EMPTY:
                        ePowerLevel = SDL_JOYSTICK_POWER_EMPTY;
                        break;
                    case BATTERY_LEVEL_LOW:
                        ePowerLevel = SDL_JOYSTICK_POWER_LOW;
                        break;
                    case BATTERY_LEVEL_MEDIUM:
                        ePowerLevel = SDL_JOYSTICK_POWER_MEDIUM;
                        break;
                    default:
                    case BATTERY_LEVEL_FULL:
                        ePowerLevel = SDL_JOYSTICK_POWER_FULL;
                        break;
                    }
                }
                SDL_PrivateJoystickBatteryLevel(joystick, ePowerLevel);
            }
        }
    }
#endif /* SDL_JOYSTICK_RAWINPUT_XINPUT */

#ifdef SDL_JOYSTICK_RAWINPUT_WGI
    if (!has_trigger_data && ctx->wgi_correlated) {
        RAWINPUT_UpdateWindowsGamingInput(); /* May detect disconnect / cause uncorrelation */
        if (ctx->wgi_correlated) {           /* Still connected */
            struct __x_ABI_CWindows_CGaming_CInput_CGamepadReading *state = &ctx->wgi_slot->state;

            if (ctx->guide_hack) {
                SDL_PrivateJoystickButton(joystick, guide_button, (state->Buttons & GamepadButtons_GUIDE) ? SDL_PRESSED : SDL_RELEASED);
            }
            if (ctx->trigger_hack) {
                SDL_PrivateJoystickAxis(joystick, left_trigger, ((int)(state->LeftTrigger * SDL_MAX_UINT16)) - 32768);
                SDL_PrivateJoystickAxis(joystick, right_trigger, ((int)(state->RightTrigger * SDL_MAX_UINT16)) - 32768);
            }
            has_trigger_data = SDL_TRUE;
        }
    }
#endif /* SDL_JOYSTICK_RAWINPUT_WGI */

    if (!correlated) {
        if (!guide_button_candidate.joystick ||
            (ctx->last_state_packet && (!guide_button_candidate.last_state_packet ||
                                        SDL_TICKS_PASSED(ctx->last_state_packet, guide_button_candidate.last_state_packet)))) {
            guide_button_candidate.joystick = joystick;
            guide_button_candidate.last_state_packet = ctx->last_state_packet;
        }
    }
#endif /* SDL_JOYSTICK_RAWINPUT_MATCHING */
}

static void RAWINPUT_JoystickUpdate(SDL_Joystick *joystick)
{
    RAWINPUT_UpdateOtherAPIs(joystick);
}

static void RAWINPUT_JoystickClose(SDL_Joystick *joystick)
{
    RAWINPUT_DeviceContext *ctx = joystick->hwdata;

#ifdef SDL_JOYSTICK_RAWINPUT_MATCHING
    if (guide_button_candidate.joystick == joystick) {
        guide_button_candidate.joystick = NULL;
    }
    if (guide_button_candidate.last_joystick == joystick) {
        guide_button_candidate.last_joystick = NULL;
    }
#endif

    if (ctx) {
        SDL_RAWINPUT_Device *device;

#ifdef SDL_JOYSTICK_RAWINPUT_XINPUT
        xinput_device_change = SDL_TRUE;
        if (ctx->xinput_enabled) {
            if (ctx->xinput_correlated) {
                RAWINPUT_MarkXInputSlotFree(ctx->xinput_slot);
            }
            WIN_UnloadXInputDLL();
        }
#endif
#ifdef SDL_JOYSTICK_RAWINPUT_WGI
        RAWINPUT_QuitWindowsGamingInput(ctx);
#endif

        device = ctx->device;
        if (device) {
            SDL_assert(device->joystick == joystick);
            device->joystick = NULL;
            RAWINPUT_ReleaseDevice(device);
        }

        SDL_free(ctx->data);
        SDL_free(ctx->button_indices);
        SDL_free(ctx->axis_indices);
        SDL_free(ctx->hat_indices);
        SDL_free(ctx);
        joystick->hwdata = NULL;
    }
}

int RAWINPUT_RegisterNotifications(HWND hWnd)
{
    int i;
    RAWINPUTDEVICE rid[SDL_arraysize(subscribed_devices)];

    if (!SDL_RAWINPUT_inited) {
        return 0;
    }

    for (i = 0; i < SDL_arraysize(subscribed_devices); i++) {
        rid[i].usUsagePage = USB_USAGEPAGE_GENERIC_DESKTOP;
        rid[i].usUsage = subscribed_devices[i];
        rid[i].dwFlags = RIDEV_DEVNOTIFY | RIDEV_INPUTSINK; /* Receive messages when in background, including device add/remove */
        rid[i].hwndTarget = hWnd;
    }

    if (!RegisterRawInputDevices(rid, SDL_arraysize(rid), sizeof(RAWINPUTDEVICE))) {
        return SDL_SetError("Couldn't register for raw input events");
    }
    return 0;
}

int RAWINPUT_UnregisterNotifications(void)
{
    int i;
    RAWINPUTDEVICE rid[SDL_arraysize(subscribed_devices)];

    if (!SDL_RAWINPUT_inited) {
        return 0;
    }

    for (i = 0; i < SDL_arraysize(subscribed_devices); i++) {
        rid[i].usUsagePage = USB_USAGEPAGE_GENERIC_DESKTOP;
        rid[i].usUsage = subscribed_devices[i];
        rid[i].dwFlags = RIDEV_REMOVE;
        rid[i].hwndTarget = NULL;
    }

    if (!RegisterRawInputDevices(rid, SDL_arraysize(rid), sizeof(RAWINPUTDEVICE))) {
        return SDL_SetError("Couldn't unregister for raw input events");
    }
    return 0;
}

LRESULT CALLBACK
RAWINPUT_WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    LRESULT result = -1;

    if (SDL_RAWINPUT_inited) {
        SDL_LockJoysticks();

        switch (msg) {
        case WM_INPUT_DEVICE_CHANGE:
        {
            HANDLE hDevice = (HANDLE)lParam;
            switch (wParam) {
            case GIDC_ARRIVAL:
                RAWINPUT_AddDevice(hDevice);
                break;
            case GIDC_REMOVAL:
            {
                SDL_RAWINPUT_Device *device;
                device = RAWINPUT_DeviceFromHandle(hDevice);
                if (device) {
                    RAWINPUT_DelDevice(device, SDL_TRUE);
                }
                break;
            }
            default:
                break;
            }
        }
            result = 0;
            break;

        case WM_INPUT:
        {
            Uint8 data[sizeof(RAWINPUTHEADER) + sizeof(RAWHID) + USB_PACKET_LENGTH];
            UINT buffer_size = SDL_arraysize(data);

            if ((int)GetRawInputData((HRAWINPUT)lParam, RID_INPUT, data, &buffer_size, sizeof(RAWINPUTHEADER)) > 0) {
                PRAWINPUT raw_input = (PRAWINPUT)data;
                SDL_RAWINPUT_Device *device = RAWINPUT_DeviceFromHandle(raw_input->header.hDevice);
                if (device) {
                    SDL_Joystick *joystick = device->joystick;
                    if (joystick) {
                        RAWINPUT_HandleStatePacket(joystick, raw_input->data.hid.bRawData, raw_input->data.hid.dwSizeHid);
                    }
                }
            }
        }
            result = 0;
            break;
        }

        SDL_UnlockJoysticks();
    }

    if (result >= 0) {
        return result;
    }
    return CallWindowProc(DefWindowProc, hWnd, msg, wParam, lParam);
}

static void RAWINPUT_JoystickQuit(void)
{
    if (!SDL_RAWINPUT_inited) {
        return;
    }

    RAWINPUT_RemoveDevices();

    WIN_UnloadHIDDLL();

    SDL_RAWINPUT_inited = SDL_FALSE;
}

static SDL_bool RAWINPUT_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    return SDL_FALSE;
}

SDL_JoystickDriver SDL_RAWINPUT_JoystickDriver = {
    RAWINPUT_JoystickInit,
    RAWINPUT_JoystickGetCount,
    RAWINPUT_JoystickDetect,
    RAWINPUT_JoystickGetDeviceName,
    RAWINPUT_JoystickGetDevicePath,
    RAWINPUT_JoystickGetDeviceSteamVirtualGamepadSlot,
    RAWINPUT_JoystickGetDevicePlayerIndex,
    RAWINPUT_JoystickSetDevicePlayerIndex,
    RAWINPUT_JoystickGetDeviceGUID,
    RAWINPUT_JoystickGetDeviceInstanceID,
    RAWINPUT_JoystickOpen,
    RAWINPUT_JoystickRumble,
    RAWINPUT_JoystickRumbleTriggers,
    RAWINPUT_JoystickGetCapabilities,
    RAWINPUT_JoystickSetLED,
    RAWINPUT_JoystickSendEffect,
    RAWINPUT_JoystickSetSensorsEnabled,
    RAWINPUT_JoystickUpdate,
    RAWINPUT_JoystickClose,
    RAWINPUT_JoystickQuit,
    RAWINPUT_JoystickGetGamepadMapping
};

#endif /* SDL_JOYSTICK_RAWINPUT */

/* vi: set ts=4 sw=4 expandtab: */
