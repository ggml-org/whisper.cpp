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

#ifdef SDL_JOYSTICK_WGI

#include "SDL_assert.h"
#include "SDL_atomic.h"
#include "SDL_endian.h"
#include "SDL_events.h"
#include "../SDL_sysjoystick.h"
#include "../hidapi/SDL_hidapijoystick_c.h"
#include "SDL_rawinputjoystick_c.h"

#include "../../core/windows/SDL_windows.h"
#define COBJMACROS
#include "windows.gaming.input.h"
#include <cfgmgr32.h>
#include <objidlbase.h>
#include <roapi.h>

#ifdef ____FIReference_1_INT32_INTERFACE_DEFINED__
/* MinGW-64 uses __FIReference_1_INT32 instead of Microsoft's __FIReference_1_int */
#define __FIReference_1_int           __FIReference_1_INT32
#define __FIReference_1_int_get_Value __FIReference_1_INT32_get_Value
#define __FIReference_1_int_Release   __FIReference_1_INT32_Release
#endif

struct joystick_hwdata
{
    __x_ABI_CWindows_CGaming_CInput_CIRawGameController *controller;
    __x_ABI_CWindows_CGaming_CInput_CIGameController *gamecontroller;
    __x_ABI_CWindows_CGaming_CInput_CIGameControllerBatteryInfo *battery;
    __x_ABI_CWindows_CGaming_CInput_CIGamepad *gamepad;
    __x_ABI_CWindows_CGaming_CInput_CGamepadVibration vibration;
    UINT64 timestamp;
};

typedef struct WindowsGamingInputControllerState
{
    SDL_JoystickID instance_id;
    __x_ABI_CWindows_CGaming_CInput_CIRawGameController *controller;
    char *name;
    SDL_JoystickGUID guid;
    SDL_JoystickType type;
    int naxes;
    int nhats;
    int nbuttons;
    int steam_virtual_gamepad_slot;
} WindowsGamingInputControllerState;

static struct
{
    __x_ABI_CWindows_CGaming_CInput_CIRawGameControllerStatics *statics;
    __x_ABI_CWindows_CGaming_CInput_CIArcadeStickStatics *arcade_stick_statics;
    __x_ABI_CWindows_CGaming_CInput_CIArcadeStickStatics2 *arcade_stick_statics2;
    __x_ABI_CWindows_CGaming_CInput_CIFlightStickStatics *flight_stick_statics;
    __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics *gamepad_statics;
    __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics2 *gamepad_statics2;
    __x_ABI_CWindows_CGaming_CInput_CIRacingWheelStatics *racing_wheel_statics;
    __x_ABI_CWindows_CGaming_CInput_CIRacingWheelStatics2 *racing_wheel_statics2;
    EventRegistrationToken controller_added_token;
    EventRegistrationToken controller_removed_token;
    int controller_count;
    SDL_bool ro_initialized;
    WindowsGamingInputControllerState *controllers;
} wgi;

static const IID IID_IRawGameControllerStatics = { 0xEB8D0792, 0xE95A, 0x4B19, { 0xAF, 0xC7, 0x0A, 0x59, 0xF8, 0xBF, 0x75, 0x9E } };
static const IID IID_IRawGameController = { 0x7CAD6D91, 0xA7E1, 0x4F71, { 0x9A, 0x78, 0x33, 0xE9, 0xC5, 0xDF, 0xEA, 0x62 } };
static const IID IID_IRawGameController2 = { 0x43C0C035, 0xBB73, 0x4756, { 0xA7, 0x87, 0x3E, 0xD6, 0xBE, 0xA6, 0x17, 0xBD } };
static const IID IID_IEventHandler_RawGameController = { 0x00621c22, 0x42e8, 0x529f, { 0x92, 0x70, 0x83, 0x6b, 0x32, 0x93, 0x1d, 0x72 } };
static const IID IID_IGameController = { 0x1BAF6522, 0x5F64, 0x42C5, { 0x82, 0x67, 0xB9, 0xFE, 0x22, 0x15, 0xBF, 0xBD } };
static const IID IID_IGameControllerBatteryInfo = { 0xDCECC681, 0x3963, 0x4DA6, { 0x95, 0x5D, 0x55, 0x3F, 0x3B, 0x6F, 0x61, 0x61 } };
static const IID IID_IArcadeStickStatics = { 0x5C37B8C8, 0x37B1, 0x4AD8, { 0x94, 0x58, 0x20, 0x0F, 0x1A, 0x30, 0x01, 0x8E } };
static const IID IID_IArcadeStickStatics2 = { 0x52B5D744, 0xBB86, 0x445A, { 0xB5, 0x9C, 0x59, 0x6F, 0x0E, 0x2A, 0x49, 0xDF } };
/*static const IID IID_IArcadeStick = { 0xB14A539D, 0xBEFB, 0x4C81, { 0x80, 0x51, 0x15, 0xEC, 0xF3, 0xB1, 0x30, 0x36 } };*/
static const IID IID_IFlightStickStatics = { 0x5514924A, 0xFECC, 0x435E, { 0x83, 0xDC, 0x5C, 0xEC, 0x8A, 0x18, 0xA5, 0x20 } };
/*static const IID IID_IFlightStick = { 0xB4A2C01C, 0xB83B, 0x4459, { 0xA1, 0xA9, 0x97, 0xB0, 0x3C, 0x33, 0xDA, 0x7C } };*/
static const IID IID_IGamepadStatics = { 0x8BBCE529, 0xD49C, 0x39E9, { 0x95, 0x60, 0xE4, 0x7D, 0xDE, 0x96, 0xB7, 0xC8 } };
static const IID IID_IGamepadStatics2 = { 0x42676DC5, 0x0856, 0x47C4, { 0x92, 0x13, 0xB3, 0x95, 0x50, 0x4C, 0x3A, 0x3C } };
/*static const IID IID_IGamepad = { 0xBC7BB43C, 0x0A69, 0x3903, { 0x9E, 0x9D, 0xA5, 0x0F, 0x86, 0xA4, 0x5D, 0xE5 } };*/
static const IID IID_IRacingWheelStatics = { 0x3AC12CD5, 0x581B, 0x4936, { 0x9F, 0x94, 0x69, 0xF1, 0xE6, 0x51, 0x4C, 0x7D } };
static const IID IID_IRacingWheelStatics2 = { 0xE666BCAA, 0xEDFD, 0x4323, { 0xA9, 0xF6, 0x3C, 0x38, 0x40, 0x48, 0xD1, 0xED } };
/*static const IID IID_IRacingWheel = { 0xF546656F, 0xE106, 0x4C82, { 0xA9, 0x0F, 0x55, 0x40, 0x12, 0x90, 0x4B, 0x85 } };*/

typedef HRESULT(WINAPI *WindowsCreateStringReference_t)(PCWSTR sourceString, UINT32 length, HSTRING_HEADER *hstringHeader, HSTRING *string);
typedef HRESULT(WINAPI *RoGetActivationFactory_t)(HSTRING activatableClassId, REFIID iid, void **factory);


extern SDL_bool SDL_XINPUT_Enabled(void);
extern SDL_bool SDL_DINPUT_JoystickPresent(Uint16 vendor, Uint16 product, Uint16 version);


static SDL_bool SDL_IsXInputDevice(Uint16 vendor, Uint16 product)
{
#if defined(SDL_JOYSTICK_XINPUT) || defined(SDL_JOYSTICK_RAWINPUT)
    PRAWINPUTDEVICELIST raw_devices = NULL;
    UINT i, raw_device_count = 0;
    LONG vidpid = MAKELONG(vendor, product);

    /* XInput and RawInput backends will pick up XInput-compatible devices */
    if (!SDL_XINPUT_Enabled()
#ifdef SDL_JOYSTICK_RAWINPUT
        && !RAWINPUT_IsEnabled()
#endif
    ) {
        return SDL_FALSE;
    }

    /* Go through RAWINPUT (WinXP and later) to find HID devices. */
    if ((GetRawInputDeviceList(NULL, &raw_device_count, sizeof(RAWINPUTDEVICELIST)) == -1) || (!raw_device_count)) {
        return SDL_FALSE; /* oh well. */
    }

    raw_devices = (PRAWINPUTDEVICELIST)SDL_malloc(sizeof(RAWINPUTDEVICELIST) * raw_device_count);
    if (!raw_devices) {
        SDL_OutOfMemory();
        return SDL_FALSE;
    }

    raw_device_count = GetRawInputDeviceList(raw_devices, &raw_device_count, sizeof(RAWINPUTDEVICELIST));
    if (raw_device_count == (UINT)-1) {
        SDL_free(raw_devices);
        raw_devices = NULL;
        return SDL_FALSE; /* oh well. */
    }

    for (i = 0; i < raw_device_count; i++) {
        RID_DEVICE_INFO rdi;
        char devName[MAX_PATH];
        UINT rdiSize = sizeof(rdi);
        UINT nameSize = SDL_arraysize(devName);
        DEVINST devNode;
        char devVidPidString[32];
        int j;

        rdi.cbSize = sizeof(rdi);

        if ((raw_devices[i].dwType != RIM_TYPEHID) ||
            (GetRawInputDeviceInfoA(raw_devices[i].hDevice, RIDI_DEVICEINFO, &rdi, &rdiSize) == ((UINT)-1)) ||
            (GetRawInputDeviceInfoA(raw_devices[i].hDevice, RIDI_DEVICENAME, devName, &nameSize) == ((UINT)-1)) ||
            (SDL_strstr(devName, "IG_") == NULL)) {
            /* Skip non-XInput devices */
            continue;
        }

        /* First check for a simple VID/PID match. This will work for Xbox 360 controllers. */
        if (MAKELONG(rdi.hid.dwVendorId, rdi.hid.dwProductId) == vidpid) {
            SDL_free(raw_devices);
            return SDL_TRUE;
        }

        /* For Xbox One controllers, Microsoft doesn't propagate the VID/PID down to the HID stack.
         * We'll have to walk the device tree upwards searching for a match for our VID/PID. */

        /* Make sure the device interface string is something we know how to parse */
        /* Example: \\?\HID#VID_045E&PID_02FF&IG_00#9&2c203035&2&0000#{4d1e55b2-f16f-11cf-88cb-001111000030} */
        if ((SDL_strstr(devName, "\\\\?\\") != devName) || (SDL_strstr(devName, "#{") == NULL)) {
            continue;
        }

        /* Unescape the backslashes in the string and terminate before the GUID portion */
        for (j = 0; devName[j] != '\0'; j++) {
            if (devName[j] == '#') {
                if (devName[j + 1] == '{') {
                    devName[j] = '\0';
                    break;
                } else {
                    devName[j] = '\\';
                }
            }
        }

        /* We'll be left with a string like this: \\?\HID\VID_045E&PID_02FF&IG_00\9&2c203035&2&0000
         * Simply skip the \\?\ prefix and we'll have a properly formed device instance ID */
        if (CM_Locate_DevNodeA(&devNode, &devName[4], CM_LOCATE_DEVNODE_NORMAL) != CR_SUCCESS) {
            continue;
        }

        (void)SDL_snprintf(devVidPidString, sizeof(devVidPidString), "VID_%04X&PID_%04X", vendor, product);

        while (CM_Get_Parent(&devNode, devNode, 0) == CR_SUCCESS) {
            char deviceId[MAX_DEVICE_ID_LEN];

            if ((CM_Get_Device_IDA(devNode, deviceId, SDL_arraysize(deviceId), 0) == CR_SUCCESS) &&
                (SDL_strstr(deviceId, devVidPidString) != NULL)) {
                /* The VID/PID matched a parent device */
                SDL_free(raw_devices);
                return SDL_TRUE;
            }
        }
    }

    SDL_free(raw_devices);
#endif /* SDL_JOYSTICK_XINPUT || SDL_JOYSTICK_RAWINPUT */

    return SDL_FALSE;
}

static void WGI_LoadRawGameControllerStatics(void)
{
    WindowsCreateStringReference_t WindowsCreateStringReferenceFunc = NULL;
    RoGetActivationFactory_t RoGetActivationFactoryFunc = NULL;
    HRESULT hr;

#ifdef __WINRT__
    WindowsCreateStringReferenceFunc = WindowsCreateStringReference;
    RoGetActivationFactoryFunc = RoGetActivationFactory;
#else
    {
        WindowsCreateStringReferenceFunc = (WindowsCreateStringReference_t)WIN_LoadComBaseFunction("WindowsCreateStringReference");
        RoGetActivationFactoryFunc = (RoGetActivationFactory_t)WIN_LoadComBaseFunction("RoGetActivationFactory");
    }
#endif /* __WINRT__ */
    if (WindowsCreateStringReferenceFunc && RoGetActivationFactoryFunc) {
        PCWSTR pNamespace;
        HSTRING_HEADER hNamespaceStringHeader;
        HSTRING hNamespaceString;

        pNamespace = L"Windows.Gaming.Input.RawGameController";
        hr = WindowsCreateStringReferenceFunc(pNamespace, (UINT32)SDL_wcslen(pNamespace), &hNamespaceStringHeader, &hNamespaceString);
        if (SUCCEEDED(hr)) {
            hr = RoGetActivationFactoryFunc(hNamespaceString, &IID_IRawGameControllerStatics, (void **)&wgi.statics);
            if (!SUCCEEDED(hr)) {
                SDL_SetError("Couldn't find IRawGameControllerStatics: 0x%lx", hr);
            }
        }
    }
}

static void WGI_LoadOtherControllerStatics(void)
{
    WindowsCreateStringReference_t WindowsCreateStringReferenceFunc = NULL;
    RoGetActivationFactory_t RoGetActivationFactoryFunc = NULL;
    HRESULT hr;

#ifdef __WINRT__
    WindowsCreateStringReferenceFunc = WindowsCreateStringReference;
    RoGetActivationFactoryFunc = RoGetActivationFactory;
#else
    {
        WindowsCreateStringReferenceFunc = (WindowsCreateStringReference_t)WIN_LoadComBaseFunction("WindowsCreateStringReference");
        RoGetActivationFactoryFunc = (RoGetActivationFactory_t)WIN_LoadComBaseFunction("RoGetActivationFactory");
    }
#endif /* __WINRT__ */
    if (WindowsCreateStringReferenceFunc && RoGetActivationFactoryFunc) {
        PCWSTR pNamespace;
        HSTRING_HEADER hNamespaceStringHeader;
        HSTRING hNamespaceString;

        pNamespace = L"Windows.Gaming.Input.ArcadeStick";
        hr = WindowsCreateStringReferenceFunc(pNamespace, (UINT32)SDL_wcslen(pNamespace), &hNamespaceStringHeader, &hNamespaceString);
        if (SUCCEEDED(hr)) {
            hr = RoGetActivationFactoryFunc(hNamespaceString, &IID_IArcadeStickStatics, (void **)&wgi.arcade_stick_statics);
            if (SUCCEEDED(hr)) {
                __x_ABI_CWindows_CGaming_CInput_CIArcadeStickStatics_QueryInterface(wgi.arcade_stick_statics, &IID_IArcadeStickStatics2, (void **)&wgi.arcade_stick_statics2);
            } else {
                SDL_SetError("Couldn't find IID_IArcadeStickStatics: 0x%lx", hr);
            }
        }

        pNamespace = L"Windows.Gaming.Input.FlightStick";
        hr = WindowsCreateStringReferenceFunc(pNamespace, (UINT32)SDL_wcslen(pNamespace), &hNamespaceStringHeader, &hNamespaceString);
        if (SUCCEEDED(hr)) {
            hr = RoGetActivationFactoryFunc(hNamespaceString, &IID_IFlightStickStatics, (void **)&wgi.flight_stick_statics);
            if (!SUCCEEDED(hr)) {
                SDL_SetError("Couldn't find IID_IFlightStickStatics: 0x%lx", hr);
            }
        }

        pNamespace = L"Windows.Gaming.Input.Gamepad";
        hr = WindowsCreateStringReferenceFunc(pNamespace, (UINT32)SDL_wcslen(pNamespace), &hNamespaceStringHeader, &hNamespaceString);
        if (SUCCEEDED(hr)) {
            hr = RoGetActivationFactoryFunc(hNamespaceString, &IID_IGamepadStatics, (void **)&wgi.gamepad_statics);
            if (SUCCEEDED(hr)) {
                __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics_QueryInterface(wgi.gamepad_statics, &IID_IGamepadStatics2, (void **)&wgi.gamepad_statics2);
            } else {
                SDL_SetError("Couldn't find IGamepadStatics: 0x%lx", hr);
            }
        }

        pNamespace = L"Windows.Gaming.Input.RacingWheel";
        hr = WindowsCreateStringReferenceFunc(pNamespace, (UINT32)SDL_wcslen(pNamespace), &hNamespaceStringHeader, &hNamespaceString);
        if (SUCCEEDED(hr)) {
            hr = RoGetActivationFactoryFunc(hNamespaceString, &IID_IRacingWheelStatics, (void **)&wgi.racing_wheel_statics);
            if (SUCCEEDED(hr)) {
                __x_ABI_CWindows_CGaming_CInput_CIRacingWheelStatics_QueryInterface(wgi.racing_wheel_statics, &IID_IRacingWheelStatics2, (void **)&wgi.racing_wheel_statics2);
            } else {
                SDL_SetError("Couldn't find IRacingWheelStatics: 0x%lx", hr);
            }
        }
    }
}

static SDL_JoystickType GetGameControllerType(__x_ABI_CWindows_CGaming_CInput_CIGameController *gamecontroller)
{
    __x_ABI_CWindows_CGaming_CInput_CIArcadeStick *arcade_stick = NULL;
    __x_ABI_CWindows_CGaming_CInput_CIFlightStick *flight_stick = NULL;
    __x_ABI_CWindows_CGaming_CInput_CIGamepad *gamepad = NULL;
    __x_ABI_CWindows_CGaming_CInput_CIRacingWheel *racing_wheel = NULL;

    /* Wait to initialize these interfaces until we need them.
     * Initializing the gamepad interface will switch Bluetooth PS4 controllers into enhanced mode, breaking DirectInput
     */
    WGI_LoadOtherControllerStatics();

    if (wgi.gamepad_statics2 && SUCCEEDED(__x_ABI_CWindows_CGaming_CInput_CIGamepadStatics2_FromGameController(wgi.gamepad_statics2, gamecontroller, &gamepad)) && gamepad) {
        __x_ABI_CWindows_CGaming_CInput_CIGamepad_Release(gamepad);
        return SDL_JOYSTICK_TYPE_GAMECONTROLLER;
    }

    if (wgi.arcade_stick_statics2 && SUCCEEDED(__x_ABI_CWindows_CGaming_CInput_CIArcadeStickStatics2_FromGameController(wgi.arcade_stick_statics2, gamecontroller, &arcade_stick)) && arcade_stick) {
        __x_ABI_CWindows_CGaming_CInput_CIArcadeStick_Release(arcade_stick);
        return SDL_JOYSTICK_TYPE_ARCADE_STICK;
    }

    if (wgi.flight_stick_statics && SUCCEEDED(__x_ABI_CWindows_CGaming_CInput_CIFlightStickStatics_FromGameController(wgi.flight_stick_statics, gamecontroller, &flight_stick)) && flight_stick) {
        __x_ABI_CWindows_CGaming_CInput_CIFlightStick_Release(flight_stick);
        return SDL_JOYSTICK_TYPE_FLIGHT_STICK;
    }

    if (wgi.racing_wheel_statics2 && SUCCEEDED(__x_ABI_CWindows_CGaming_CInput_CIRacingWheelStatics2_FromGameController(wgi.racing_wheel_statics2, gamecontroller, &racing_wheel)) && racing_wheel) {
        __x_ABI_CWindows_CGaming_CInput_CIRacingWheel_Release(racing_wheel);
        return SDL_JOYSTICK_TYPE_WHEEL;
    }

    return SDL_JOYSTICK_TYPE_UNKNOWN;
}

typedef struct RawGameControllerDelegate
{
    __FIEventHandler_1_Windows__CGaming__CInput__CRawGameController iface;
    SDL_atomic_t refcount;
} RawGameControllerDelegate;

static HRESULT STDMETHODCALLTYPE IEventHandler_CRawGameControllerVtbl_QueryInterface(__FIEventHandler_1_Windows__CGaming__CInput__CRawGameController *This, REFIID riid, void **ppvObject)
{
    if (!ppvObject) {
        return E_INVALIDARG;
    }

    *ppvObject = NULL;
    if (WIN_IsEqualIID(riid, &IID_IUnknown) || WIN_IsEqualIID(riid, &IID_IAgileObject) || WIN_IsEqualIID(riid, &IID_IEventHandler_RawGameController)) {
        *ppvObject = This;
        __FIEventHandler_1_Windows__CGaming__CInput__CRawGameController_AddRef(This);
        return S_OK;
    } else if (WIN_IsEqualIID(riid, &IID_IMarshal)) {
        /* This seems complicated. Let's hope it doesn't happen. */
        return E_OUTOFMEMORY;
    } else {
        return E_NOINTERFACE;
    }
}

static ULONG STDMETHODCALLTYPE IEventHandler_CRawGameControllerVtbl_AddRef(__FIEventHandler_1_Windows__CGaming__CInput__CRawGameController *This)
{
    RawGameControllerDelegate *self = (RawGameControllerDelegate *)This;
    return SDL_AtomicAdd(&self->refcount, 1) + 1UL;
}

static ULONG STDMETHODCALLTYPE IEventHandler_CRawGameControllerVtbl_Release(__FIEventHandler_1_Windows__CGaming__CInput__CRawGameController *This)
{
    RawGameControllerDelegate *self = (RawGameControllerDelegate *)This;
    int rc = SDL_AtomicAdd(&self->refcount, -1) - 1;
    /* Should never free the static delegate objects */
    SDL_assert(rc > 0);
    return rc;
}

static int GetSteamVirtualGamepadSlot(__x_ABI_CWindows_CGaming_CInput_CIRawGameController *controller, Uint16 vendor_id, Uint16 product_id)
{
    int slot = -1;

    if (vendor_id == USB_VENDOR_VALVE &&
        product_id == USB_PRODUCT_STEAM_VIRTUAL_GAMEPAD) {
        __x_ABI_CWindows_CGaming_CInput_CIRawGameController2 *controller2 = NULL;
        HRESULT hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameController_QueryInterface(controller, &IID_IRawGameController2, (void **)&controller2);
        if (SUCCEEDED(hr)) {
            HSTRING hString;
            hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameController2_get_NonRoamableId(controller2, &hString);
            if (SUCCEEDED(hr)) {
                typedef PCWSTR(WINAPI * WindowsGetStringRawBuffer_t)(HSTRING string, UINT32 * length);
                typedef HRESULT(WINAPI * WindowsDeleteString_t)(HSTRING string);

                WindowsGetStringRawBuffer_t WindowsGetStringRawBufferFunc = NULL;
                WindowsDeleteString_t WindowsDeleteStringFunc = NULL;
#ifdef __WINRT__
                WindowsGetStringRawBufferFunc = WindowsGetStringRawBuffer;
                WindowsDeleteStringFunc = WindowsDeleteString;
#else
                {
                    WindowsGetStringRawBufferFunc = (WindowsGetStringRawBuffer_t)WIN_LoadComBaseFunction("WindowsGetStringRawBuffer");
                    WindowsDeleteStringFunc = (WindowsDeleteString_t)WIN_LoadComBaseFunction("WindowsDeleteString");
                }
#endif /* __WINRT__ */
                if (WindowsGetStringRawBufferFunc && WindowsDeleteStringFunc) {
                    PCWSTR string = WindowsGetStringRawBufferFunc(hString, NULL);
                    if (string) {
                        char *id = WIN_StringToUTF8W(string);
                        if (id) {
                            (void)SDL_sscanf(id, "{wgi/nrid/:steam-%*X&%*X&%*X#%d#%*u}", &slot);
                            SDL_free(id);
                        }
                    }
                    WindowsDeleteStringFunc(hString);
                }
            }
            __x_ABI_CWindows_CGaming_CInput_CIRawGameController2_Release(controller2);
        }
    }
    return slot;
}

static HRESULT STDMETHODCALLTYPE IEventHandler_CRawGameControllerVtbl_InvokeAdded(__FIEventHandler_1_Windows__CGaming__CInput__CRawGameController *This, IInspectable *sender, __x_ABI_CWindows_CGaming_CInput_CIRawGameController *e)
{
    HRESULT hr;
    __x_ABI_CWindows_CGaming_CInput_CIRawGameController *controller = NULL;

    SDL_LockJoysticks();

    /* We can get delayed calls to InvokeAdded() after WGI_JoystickQuit() */
    if (SDL_JoysticksQuitting() || !SDL_JoysticksInitialized()) {
        SDL_UnlockJoysticks();
        return S_OK;
    }

    hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameController_QueryInterface(e, &IID_IRawGameController, (void **)&controller);
    if (SUCCEEDED(hr)) {
        char *name = NULL;
        SDL_JoystickGUID guid;
        Uint16 bus = SDL_HARDWARE_BUS_USB;
        Uint16 vendor = 0;
        Uint16 product = 0;
        Uint16 version = 0;
        SDL_JoystickType type = SDL_JOYSTICK_TYPE_UNKNOWN;
        __x_ABI_CWindows_CGaming_CInput_CIRawGameController2 *controller2 = NULL;
        __x_ABI_CWindows_CGaming_CInput_CIGameController *gamecontroller = NULL;
        SDL_bool ignore_joystick = SDL_FALSE;

        __x_ABI_CWindows_CGaming_CInput_CIRawGameController_get_HardwareVendorId(controller, &vendor);
        __x_ABI_CWindows_CGaming_CInput_CIRawGameController_get_HardwareProductId(controller, &product);

        hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameController_QueryInterface(controller, &IID_IRawGameController2, (void **)&controller2);
        if (SUCCEEDED(hr)) {
            typedef PCWSTR(WINAPI * WindowsGetStringRawBuffer_t)(HSTRING string, UINT32 * length);
            typedef HRESULT(WINAPI * WindowsDeleteString_t)(HSTRING string);

            WindowsGetStringRawBuffer_t WindowsGetStringRawBufferFunc = NULL;
            WindowsDeleteString_t WindowsDeleteStringFunc = NULL;
#ifdef __WINRT__
            WindowsGetStringRawBufferFunc = WindowsGetStringRawBuffer;
            WindowsDeleteStringFunc = WindowsDeleteString;
#else
            {
                WindowsGetStringRawBufferFunc = (WindowsGetStringRawBuffer_t)WIN_LoadComBaseFunction("WindowsGetStringRawBuffer");
                WindowsDeleteStringFunc = (WindowsDeleteString_t)WIN_LoadComBaseFunction("WindowsDeleteString");
            }
#endif /* __WINRT__ */
            if (WindowsGetStringRawBufferFunc && WindowsDeleteStringFunc) {
                HSTRING hString;
                hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameController2_get_DisplayName(controller2, &hString);
                if (SUCCEEDED(hr)) {
                    PCWSTR string = WindowsGetStringRawBufferFunc(hString, NULL);
                    if (string) {
                        name = WIN_StringToUTF8W(string);
                    }
                    WindowsDeleteStringFunc(hString);
                }
            }
            __x_ABI_CWindows_CGaming_CInput_CIRawGameController2_Release(controller2);
        }
        if (!name) {
            name = SDL_strdup("");
        }

#ifdef SDL_JOYSTICK_HIDAPI
        if (!ignore_joystick && HIDAPI_IsDevicePresent(vendor, product, version, name)) {
            ignore_joystick = SDL_TRUE;
        }
#endif

#ifdef SDL_JOYSTICK_RAWINPUT
        if (!ignore_joystick && RAWINPUT_IsDevicePresent(vendor, product, version, name)) {
            ignore_joystick = SDL_TRUE;
        }
#endif

        if (!ignore_joystick && SDL_DINPUT_JoystickPresent(vendor, product, version)) {
            ignore_joystick = SDL_TRUE;
        }

        if (!ignore_joystick && SDL_IsXInputDevice(vendor, product)) {
            ignore_joystick = SDL_TRUE;
        }

        if (!ignore_joystick) {
            hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameController_QueryInterface(controller, &IID_IGameController, (void **)&gamecontroller);
            if (SUCCEEDED(hr)) {
                boolean wireless;

                type = GetGameControllerType(gamecontroller);

                hr = __x_ABI_CWindows_CGaming_CInput_CIGameController_get_IsWireless(gamecontroller, &wireless);
                if (SUCCEEDED(hr) && wireless) {
                    bus = SDL_HARDWARE_BUS_BLUETOOTH;
                }

                __x_ABI_CWindows_CGaming_CInput_CIGameController_Release(gamecontroller);
            }

            guid = SDL_CreateJoystickGUID(bus, vendor, product, version, NULL, name, 'w', (Uint8)type);

            if (SDL_ShouldIgnoreJoystick(name, guid)) {
                ignore_joystick = SDL_TRUE;
            }
        }

        if (!ignore_joystick) {
            /* New device, add it */
            WindowsGamingInputControllerState *controllers = SDL_realloc(wgi.controllers, sizeof(wgi.controllers[0]) * (wgi.controller_count + 1));
            if (controllers) {
                WindowsGamingInputControllerState *state = &controllers[wgi.controller_count];
                SDL_JoystickID joystickID = SDL_GetNextJoystickInstanceID();

                SDL_zerop(state);
                state->instance_id = joystickID;
                state->controller = controller;
                state->name = name;
                state->guid = guid;
                state->type = type;
                state->steam_virtual_gamepad_slot = GetSteamVirtualGamepadSlot(controller, vendor, product);

                __x_ABI_CWindows_CGaming_CInput_CIRawGameController_get_ButtonCount(controller, &state->nbuttons);
                __x_ABI_CWindows_CGaming_CInput_CIRawGameController_get_AxisCount(controller, &state->naxes);
                __x_ABI_CWindows_CGaming_CInput_CIRawGameController_get_SwitchCount(controller, &state->nhats);

                __x_ABI_CWindows_CGaming_CInput_CIRawGameController_AddRef(controller);

                ++wgi.controller_count;
                wgi.controllers = controllers;

                SDL_PrivateJoystickAdded(joystickID);
            } else {
                SDL_free(name);
            }
        } else {
            SDL_free(name);
        }

        __x_ABI_CWindows_CGaming_CInput_CIRawGameController_Release(controller);
    }

    SDL_UnlockJoysticks();

    return S_OK;
}

static HRESULT STDMETHODCALLTYPE IEventHandler_CRawGameControllerVtbl_InvokeRemoved(__FIEventHandler_1_Windows__CGaming__CInput__CRawGameController *This, IInspectable *sender, __x_ABI_CWindows_CGaming_CInput_CIRawGameController *e)
{
    HRESULT hr;
    __x_ABI_CWindows_CGaming_CInput_CIRawGameController *controller = NULL;

    SDL_LockJoysticks();

    /* Can we get delayed calls to InvokeRemoved() after WGI_JoystickQuit()? */
    if (!SDL_JoysticksInitialized()) {
        SDL_UnlockJoysticks();
        return S_OK;
    }

    hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameController_QueryInterface(e, &IID_IRawGameController, (void **)&controller);
    if (SUCCEEDED(hr)) {
        int i;

        for (i = 0; i < wgi.controller_count; i++) {
            if (wgi.controllers[i].controller == controller) {
                WindowsGamingInputControllerState *state = &wgi.controllers[i];
                SDL_JoystickID joystickID = state->instance_id;

                __x_ABI_CWindows_CGaming_CInput_CIRawGameController_Release(state->controller);

                SDL_free(state->name);

                --wgi.controller_count;
                if (i < wgi.controller_count) {
                    SDL_memmove(&wgi.controllers[i], &wgi.controllers[i + 1], (wgi.controller_count - i) * sizeof(wgi.controllers[i]));
                }

                SDL_PrivateJoystickRemoved(joystickID);
                break;
            }
        }

        __x_ABI_CWindows_CGaming_CInput_CIRawGameController_Release(controller);
    }

    SDL_UnlockJoysticks();

    return S_OK;
}

static __FIEventHandler_1_Windows__CGaming__CInput__CRawGameControllerVtbl controller_added_vtbl = {
    IEventHandler_CRawGameControllerVtbl_QueryInterface,
    IEventHandler_CRawGameControllerVtbl_AddRef,
    IEventHandler_CRawGameControllerVtbl_Release,
    IEventHandler_CRawGameControllerVtbl_InvokeAdded
};
static RawGameControllerDelegate controller_added = {
    { &controller_added_vtbl },
    { 1 }
};

static __FIEventHandler_1_Windows__CGaming__CInput__CRawGameControllerVtbl controller_removed_vtbl = {
    IEventHandler_CRawGameControllerVtbl_QueryInterface,
    IEventHandler_CRawGameControllerVtbl_AddRef,
    IEventHandler_CRawGameControllerVtbl_Release,
    IEventHandler_CRawGameControllerVtbl_InvokeRemoved
};
static RawGameControllerDelegate controller_removed = {
    { &controller_removed_vtbl },
    { 1 }
};

static int WGI_JoystickInit(void)
{
    HRESULT hr;

    if (!SDL_GetHintBoolean(SDL_HINT_JOYSTICK_WGI, SDL_TRUE)) {
        return 0;
    }

    if (FAILED(WIN_RoInitialize())) {
        return SDL_SetError("RoInitialize() failed");
    }
    wgi.ro_initialized = SDL_TRUE;

#ifndef __WINRT__
    {
        /* There seems to be a bug in Windows where a dependency of WGI can be unloaded from memory prior to WGI itself.
         * This results in Windows_Gaming_Input!GameController::~GameController() invoking an unloaded DLL and crashing.
         * As a workaround, we will keep a reference to the MTA to prevent COM from unloading DLLs later.
         * See https://github.com/libsdl-org/SDL/issues/5552 for more details.
         */
        static PVOID cookie = NULL;
        if (!cookie) {
            typedef HRESULT(WINAPI * CoIncrementMTAUsage_t)(PVOID * pCookie);
            CoIncrementMTAUsage_t CoIncrementMTAUsageFunc = (CoIncrementMTAUsage_t)WIN_LoadComBaseFunction("CoIncrementMTAUsage");
            if (CoIncrementMTAUsageFunc) {
                if (FAILED(CoIncrementMTAUsageFunc(&cookie))) {
                    return SDL_SetError("CoIncrementMTAUsage() failed");
                }
            } else {
                /* CoIncrementMTAUsage() is present since Win8, so we should never make it here. */
                return SDL_SetError("CoIncrementMTAUsage() not found");
            }
        }
    }
#endif

    WGI_LoadRawGameControllerStatics();

    if (wgi.statics) {
        __FIVectorView_1_Windows__CGaming__CInput__CRawGameController *controllers;

        hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameControllerStatics_add_RawGameControllerAdded(wgi.statics, &controller_added.iface, &wgi.controller_added_token);
        if (!SUCCEEDED(hr)) {
            SDL_SetError("add_RawGameControllerAdded() failed: 0x%lx\n", hr);
        }

        hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameControllerStatics_add_RawGameControllerRemoved(wgi.statics, &controller_removed.iface, &wgi.controller_removed_token);
        if (!SUCCEEDED(hr)) {
            SDL_SetError("add_RawGameControllerRemoved() failed: 0x%lx\n", hr);
        }

        hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameControllerStatics_get_RawGameControllers(wgi.statics, &controllers);
        if (SUCCEEDED(hr)) {
            unsigned i, count = 0;

            hr = __FIVectorView_1_Windows__CGaming__CInput__CRawGameController_get_Size(controllers, &count);
            if (SUCCEEDED(hr)) {
                for (i = 0; i < count; ++i) {
                    __x_ABI_CWindows_CGaming_CInput_CIRawGameController *controller = NULL;

                    hr = __FIVectorView_1_Windows__CGaming__CInput__CRawGameController_GetAt(controllers, i, &controller);
                    if (SUCCEEDED(hr) && controller) {
                        IEventHandler_CRawGameControllerVtbl_InvokeAdded(&controller_added.iface, NULL, controller);
                        __x_ABI_CWindows_CGaming_CInput_CIRawGameController_Release(controller);
                    }
                }
            }

            __FIVectorView_1_Windows__CGaming__CInput__CRawGameController_Release(controllers);
        }
    }

    return 0;
}

static int WGI_JoystickGetCount(void)
{
    return wgi.controller_count;
}

static void WGI_JoystickDetect(void)
{
}

static const char *WGI_JoystickGetDeviceName(int device_index)
{
    return wgi.controllers[device_index].name;
}

static const char *WGI_JoystickGetDevicePath(int device_index)
{
    return NULL;
}

static int WGI_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    return wgi.controllers[device_index].steam_virtual_gamepad_slot;
}

static int WGI_JoystickGetDevicePlayerIndex(int device_index)
{
    return -1;
}

static void WGI_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
}

static SDL_JoystickGUID WGI_JoystickGetDeviceGUID(int device_index)
{
    return wgi.controllers[device_index].guid;
}

static SDL_JoystickID WGI_JoystickGetDeviceInstanceID(int device_index)
{
    return wgi.controllers[device_index].instance_id;
}

static int WGI_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    WindowsGamingInputControllerState *state = &wgi.controllers[device_index];
    struct joystick_hwdata *hwdata;
    boolean wireless = SDL_FALSE;

    hwdata = (struct joystick_hwdata *)SDL_calloc(1, sizeof(*hwdata));
    if (!hwdata) {
        return SDL_OutOfMemory();
    }
    joystick->hwdata = hwdata;

    hwdata->controller = state->controller;
    __x_ABI_CWindows_CGaming_CInput_CIRawGameController_AddRef(hwdata->controller);
    __x_ABI_CWindows_CGaming_CInput_CIRawGameController_QueryInterface(hwdata->controller, &IID_IGameController, (void **)&hwdata->gamecontroller);
    __x_ABI_CWindows_CGaming_CInput_CIRawGameController_QueryInterface(hwdata->controller, &IID_IGameControllerBatteryInfo, (void **)&hwdata->battery);

    if (wgi.gamepad_statics2) {
        __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics2_FromGameController(wgi.gamepad_statics2, hwdata->gamecontroller, &hwdata->gamepad);
    }

    if (hwdata->gamecontroller) {
        __x_ABI_CWindows_CGaming_CInput_CIGameController_get_IsWireless(hwdata->gamecontroller, &wireless);
    }

    /* Initialize the joystick capabilities */
    joystick->nbuttons = state->nbuttons;
    joystick->naxes = state->naxes;
    joystick->nhats = state->nhats;
    joystick->epowerlevel = wireless ? SDL_JOYSTICK_POWER_UNKNOWN : SDL_JOYSTICK_POWER_WIRED;

    if (wireless && hwdata->battery) {
        HRESULT hr;
        __x_ABI_CWindows_CDevices_CPower_CIBatteryReport *report;

        hr = __x_ABI_CWindows_CGaming_CInput_CIGameControllerBatteryInfo_TryGetBatteryReport(hwdata->battery, &report);
        if (SUCCEEDED(hr) && report) {
            int full_capacity = 0, curr_capacity = 0;
            __FIReference_1_int *full_capacityP, *curr_capacityP;

            hr = __x_ABI_CWindows_CDevices_CPower_CIBatteryReport_get_FullChargeCapacityInMilliwattHours(report, &full_capacityP);
            if (SUCCEEDED(hr)) {
                __FIReference_1_int_get_Value(full_capacityP, &full_capacity);
                __FIReference_1_int_Release(full_capacityP);
            }

            hr = __x_ABI_CWindows_CDevices_CPower_CIBatteryReport_get_RemainingCapacityInMilliwattHours(report, &curr_capacityP);
            if (SUCCEEDED(hr)) {
                __FIReference_1_int_get_Value(curr_capacityP, &curr_capacity);
                __FIReference_1_int_Release(curr_capacityP);
            }

            if (full_capacity > 0) {
                float ratio = (float)curr_capacity / full_capacity;

                if (ratio <= 0.05f) {
                    joystick->epowerlevel = SDL_JOYSTICK_POWER_EMPTY;
                } else if (ratio <= 0.20f) {
                    joystick->epowerlevel = SDL_JOYSTICK_POWER_LOW;
                } else if (ratio <= 0.70f) {
                    joystick->epowerlevel = SDL_JOYSTICK_POWER_MEDIUM;
                } else {
                    joystick->epowerlevel = SDL_JOYSTICK_POWER_FULL;
                }
            }
            __x_ABI_CWindows_CDevices_CPower_CIBatteryReport_Release(report);
        }
    }
    return 0;
}

static int WGI_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    struct joystick_hwdata *hwdata = joystick->hwdata;

    if (hwdata->gamepad) {
        HRESULT hr;

        hwdata->vibration.LeftMotor = (DOUBLE)low_frequency_rumble / SDL_MAX_UINT16;
        hwdata->vibration.RightMotor = (DOUBLE)high_frequency_rumble / SDL_MAX_UINT16;
        hr = __x_ABI_CWindows_CGaming_CInput_CIGamepad_put_Vibration(hwdata->gamepad, hwdata->vibration);
        if (SUCCEEDED(hr)) {
            return 0;
        } else {
            return SDL_SetError("Setting vibration failed: 0x%lx\n", hr);
        }
    } else {
        return SDL_Unsupported();
    }
}

static int WGI_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    struct joystick_hwdata *hwdata = joystick->hwdata;

    if (hwdata->gamepad) {
        HRESULT hr;

        hwdata->vibration.LeftTrigger = (DOUBLE)left_rumble / SDL_MAX_UINT16;
        hwdata->vibration.RightTrigger = (DOUBLE)right_rumble / SDL_MAX_UINT16;
        hr = __x_ABI_CWindows_CGaming_CInput_CIGamepad_put_Vibration(hwdata->gamepad, hwdata->vibration);
        if (SUCCEEDED(hr)) {
            return 0;
        } else {
            return SDL_SetError("Setting vibration failed: 0x%lx\n", hr);
        }
    } else {
        return SDL_Unsupported();
    }
}

static Uint32 WGI_JoystickGetCapabilities(SDL_Joystick *joystick)
{
    struct joystick_hwdata *hwdata = joystick->hwdata;

    if (hwdata->gamepad) {
        /* FIXME: Can WGI even tell us if trigger rumble is supported? */
        return SDL_JOYCAP_RUMBLE | SDL_JOYCAP_RUMBLE_TRIGGERS;
    } else {
        return 0;
    }
}

static int WGI_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static int WGI_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static int WGI_JoystickSetSensorsEnabled(SDL_Joystick *joystick, SDL_bool enabled)
{
    return SDL_Unsupported();
}

static Uint8 ConvertHatValue(__x_ABI_CWindows_CGaming_CInput_CGameControllerSwitchPosition value)
{
    switch (value) {
    case GameControllerSwitchPosition_Up:
        return SDL_HAT_UP;
    case GameControllerSwitchPosition_UpRight:
        return SDL_HAT_RIGHTUP;
    case GameControllerSwitchPosition_Right:
        return SDL_HAT_RIGHT;
    case GameControllerSwitchPosition_DownRight:
        return SDL_HAT_RIGHTDOWN;
    case GameControllerSwitchPosition_Down:
        return SDL_HAT_DOWN;
    case GameControllerSwitchPosition_DownLeft:
        return SDL_HAT_LEFTDOWN;
    case GameControllerSwitchPosition_Left:
        return SDL_HAT_LEFT;
    case GameControllerSwitchPosition_UpLeft:
        return SDL_HAT_LEFTUP;
    default:
        return SDL_HAT_CENTERED;
    }
}

static void WGI_JoystickUpdate(SDL_Joystick *joystick)
{
    struct joystick_hwdata *hwdata = joystick->hwdata;
    HRESULT hr;
    UINT32 nbuttons = SDL_min(joystick->nbuttons, SDL_MAX_UINT8);
    boolean *buttons = NULL;
    UINT32 nhats = SDL_min(joystick->nhats, SDL_MAX_UINT8);
    __x_ABI_CWindows_CGaming_CInput_CGameControllerSwitchPosition *hats = NULL;
    UINT32 naxes = SDL_min(joystick->naxes, SDL_MAX_UINT8);
    DOUBLE *axes = NULL;
    UINT64 timestamp;

    if (nbuttons > 0) {
        buttons = SDL_stack_alloc(boolean, nbuttons);
    }
    if (nhats > 0) {
        hats = SDL_stack_alloc(__x_ABI_CWindows_CGaming_CInput_CGameControllerSwitchPosition, nhats);
    }
    if (naxes > 0) {
        axes = SDL_stack_alloc(DOUBLE, naxes);
    }

    hr = __x_ABI_CWindows_CGaming_CInput_CIRawGameController_GetCurrentReading(hwdata->controller, nbuttons, buttons, nhats, hats, naxes, axes, &timestamp);
    if (SUCCEEDED(hr) && (!timestamp || timestamp != hwdata->timestamp)) {
        UINT32 i;
        SDL_bool all_zero = SDL_FALSE;

        /* The axes are all zero when the application loses focus */
        if (naxes > 0) {
            all_zero = SDL_TRUE;
            for (i = 0; i < naxes; ++i) {
                if (axes[i] != 0.0f) {
                    all_zero = SDL_FALSE;
                    break;
                }
            }
        }
        if (all_zero) {
            SDL_PrivateJoystickForceRecentering(joystick);
        } else {
            for (i = 0; i < nbuttons; ++i) {
                SDL_PrivateJoystickButton(joystick, (Uint8)i, buttons[i]);
            }
            for (i = 0; i < nhats; ++i) {
                SDL_PrivateJoystickHat(joystick, (Uint8)i, ConvertHatValue(hats[i]));
            }
            for (i = 0; i < naxes; ++i) {
                SDL_PrivateJoystickAxis(joystick, (Uint8)i, (Sint16)((int)(axes[i] * 65535) - 32768));
            }
        }
        hwdata->timestamp = timestamp;
    }

    SDL_stack_free(buttons);
    SDL_stack_free(hats);
    SDL_stack_free(axes);
}

static void WGI_JoystickClose(SDL_Joystick *joystick)
{
    struct joystick_hwdata *hwdata = joystick->hwdata;

    if (hwdata) {
        if (hwdata->controller) {
            __x_ABI_CWindows_CGaming_CInput_CIRawGameController_Release(hwdata->controller);
        }
        if (hwdata->gamecontroller) {
            __x_ABI_CWindows_CGaming_CInput_CIGameController_Release(hwdata->gamecontroller);
        }
        if (hwdata->battery) {
            __x_ABI_CWindows_CGaming_CInput_CIGameControllerBatteryInfo_Release(hwdata->battery);
        }
        if (hwdata->gamepad) {
            __x_ABI_CWindows_CGaming_CInput_CIGamepad_Release(hwdata->gamepad);
        }
        SDL_free(hwdata);
    }
    joystick->hwdata = NULL;
}

static void WGI_JoystickQuit(void)
{
    if (wgi.statics) {
        while (wgi.controller_count > 0) {
            IEventHandler_CRawGameControllerVtbl_InvokeRemoved(&controller_removed.iface, NULL, wgi.controllers[wgi.controller_count - 1].controller);
        }
        if (wgi.controllers) {
            SDL_free(wgi.controllers);
        }

        if (wgi.arcade_stick_statics) {
            __x_ABI_CWindows_CGaming_CInput_CIArcadeStickStatics_Release(wgi.arcade_stick_statics);
        }
        if (wgi.arcade_stick_statics2) {
            __x_ABI_CWindows_CGaming_CInput_CIArcadeStickStatics2_Release(wgi.arcade_stick_statics2);
        }
        if (wgi.flight_stick_statics) {
            __x_ABI_CWindows_CGaming_CInput_CIFlightStickStatics_Release(wgi.flight_stick_statics);
        }
        if (wgi.gamepad_statics) {
            __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics_Release(wgi.gamepad_statics);
        }
        if (wgi.gamepad_statics2) {
            __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics2_Release(wgi.gamepad_statics2);
        }
        if (wgi.racing_wheel_statics) {
            __x_ABI_CWindows_CGaming_CInput_CIRacingWheelStatics_Release(wgi.racing_wheel_statics);
        }
        if (wgi.racing_wheel_statics2) {
            __x_ABI_CWindows_CGaming_CInput_CIRacingWheelStatics2_Release(wgi.racing_wheel_statics2);
        }

        __x_ABI_CWindows_CGaming_CInput_CIRawGameControllerStatics_remove_RawGameControllerAdded(wgi.statics, wgi.controller_added_token);
        __x_ABI_CWindows_CGaming_CInput_CIRawGameControllerStatics_remove_RawGameControllerRemoved(wgi.statics, wgi.controller_removed_token);
        __x_ABI_CWindows_CGaming_CInput_CIRawGameControllerStatics_Release(wgi.statics);
    }

    if (wgi.ro_initialized) {
        WIN_RoUninitialize();
    }

    SDL_zero(wgi);
}

static SDL_bool WGI_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    return SDL_FALSE;
}

SDL_JoystickDriver SDL_WGI_JoystickDriver = {
    WGI_JoystickInit,
    WGI_JoystickGetCount,
    WGI_JoystickDetect,
    WGI_JoystickGetDeviceName,
    WGI_JoystickGetDevicePath,
    WGI_JoystickGetDeviceSteamVirtualGamepadSlot,
    WGI_JoystickGetDevicePlayerIndex,
    WGI_JoystickSetDevicePlayerIndex,
    WGI_JoystickGetDeviceGUID,
    WGI_JoystickGetDeviceInstanceID,
    WGI_JoystickOpen,
    WGI_JoystickRumble,
    WGI_JoystickRumbleTriggers,
    WGI_JoystickGetCapabilities,
    WGI_JoystickSetLED,
    WGI_JoystickSendEffect,
    WGI_JoystickSetSensorsEnabled,
    WGI_JoystickUpdate,
    WGI_JoystickClose,
    WGI_JoystickQuit,
    WGI_JoystickGetGamepadMapping
};

#endif /* SDL_JOYSTICK_WGI */

/* vi: set ts=4 sw=4 expandtab: */
