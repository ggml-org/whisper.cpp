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

#if defined(SDL_JOYSTICK_DINPUT) || defined(SDL_JOYSTICK_XINPUT)

/* DirectInput joystick driver; written by Glenn Maynard, based on Andrei de
 * A. Formiga's WINMM driver.
 *
 * Hats and sliders are completely untested; the app I'm writing this for mostly
 * doesn't use them and I don't own any joysticks with them.
 *
 * We don't bother to use event notification here.  It doesn't seem to work
 * with polled devices, and it's fine to call IDirectInputDevice8_GetDeviceData and
 * let it return 0 events. */

#include "SDL_error.h"
#include "SDL_events.h"
#include "SDL_hints.h"
#include "SDL_timer.h"
#include "SDL_mutex.h"
#include "SDL_joystick.h"
#include "../SDL_sysjoystick.h"
#include "../../thread/SDL_systhread.h"
#include "../../core/windows/SDL_windows.h"
#if !defined(__WINRT__) && !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
#include <dbt.h>
#endif

#define INITGUID /* Only set here, if set twice will cause mingw32 to break. */
#include "SDL_windowsjoystick_c.h"
#include "SDL_dinputjoystick_c.h"
#include "SDL_xinputjoystick_c.h"
#include "SDL_rawinputjoystick_c.h"

#include "../../haptic/windows/SDL_dinputhaptic_c.h" /* For haptic hot plugging */
#include "../../haptic/windows/SDL_xinputhaptic_c.h" /* For haptic hot plugging */

#ifndef DEVICE_NOTIFY_WINDOW_HANDLE
#define DEVICE_NOTIFY_WINDOW_HANDLE 0x00000000
#endif

/* CM_Register_Notification definitions */

#define CR_SUCCESS (0x00000000)

/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

DECLARE_HANDLE(HCMNOTIFICATION);
typedef HCMNOTIFICATION *PHCMNOTIFICATION;

typedef enum _CM_NOTIFY_FILTER_TYPE
{
    CM_NOTIFY_FILTER_TYPE_DEVICEINTERFACE = 0,
    CM_NOTIFY_FILTER_TYPE_DEVICEHANDLE,
    CM_NOTIFY_FILTER_TYPE_DEVICEINSTANCE,
    CM_NOTIFY_FILTER_TYPE_MAX
} CM_NOTIFY_FILTER_TYPE,
    *PCM_NOTIFY_FILTER_TYPE;

typedef struct _CM_NOTIFY_FILTER
{
    DWORD cbSize;
    DWORD Flags;
    CM_NOTIFY_FILTER_TYPE FilterType;
    DWORD Reserved;
    union
    {
        struct
        {
            GUID ClassGuid;
        } DeviceInterface;
        struct
        {
            HANDLE hTarget;
        } DeviceHandle;
        struct
        {
            WCHAR InstanceId[200];
        } DeviceInstance;
    } u;
} CM_NOTIFY_FILTER, *PCM_NOTIFY_FILTER;

typedef enum _CM_NOTIFY_ACTION
{
    CM_NOTIFY_ACTION_DEVICEINTERFACEARRIVAL = 0,
    CM_NOTIFY_ACTION_DEVICEINTERFACEREMOVAL,
    CM_NOTIFY_ACTION_DEVICEQUERYREMOVE,
    CM_NOTIFY_ACTION_DEVICEQUERYREMOVEFAILED,
    CM_NOTIFY_ACTION_DEVICEREMOVEPENDING,
    CM_NOTIFY_ACTION_DEVICEREMOVECOMPLETE,
    CM_NOTIFY_ACTION_DEVICECUSTOMEVENT,
    CM_NOTIFY_ACTION_DEVICEINSTANCEENUMERATED,
    CM_NOTIFY_ACTION_DEVICEINSTANCESTARTED,
    CM_NOTIFY_ACTION_DEVICEINSTANCEREMOVED,
    CM_NOTIFY_ACTION_MAX
} CM_NOTIFY_ACTION,
    *PCM_NOTIFY_ACTION;

typedef struct _CM_NOTIFY_EVENT_DATA
{
    CM_NOTIFY_FILTER_TYPE FilterType;
    DWORD Reserved;
    union
    {
        struct
        {
            GUID ClassGuid;
            WCHAR SymbolicLink[ANYSIZE_ARRAY];
        } DeviceInterface;
        struct
        {
            GUID EventGuid;
            LONG NameOffset;
            DWORD DataSize;
            BYTE Data[ANYSIZE_ARRAY];
        } DeviceHandle;
        struct
        {
            WCHAR InstanceId[ANYSIZE_ARRAY];
        } DeviceInstance;
    } u;
} CM_NOTIFY_EVENT_DATA, *PCM_NOTIFY_EVENT_DATA;

typedef DWORD(CALLBACK *PCM_NOTIFY_CALLBACK)(HCMNOTIFICATION hNotify, PVOID Context, CM_NOTIFY_ACTION Action, PCM_NOTIFY_EVENT_DATA EventData, DWORD EventDataSize);

typedef DWORD(WINAPI *CM_Register_NotificationFunc)(PCM_NOTIFY_FILTER pFilter, PVOID pContext, PCM_NOTIFY_CALLBACK pCallback, PHCMNOTIFICATION pNotifyContext);
typedef DWORD(WINAPI *CM_Unregister_NotificationFunc)(HCMNOTIFICATION NotifyContext);

/* local variables */
static SDL_bool s_bJoystickThread = SDL_FALSE;
static SDL_bool s_bWindowsDeviceChanged = SDL_FALSE;
static SDL_cond *s_condJoystickThread = NULL;
static SDL_mutex *s_mutexJoyStickEnum = NULL;
static SDL_Thread *s_joystickThread = NULL;
static SDL_bool s_bJoystickThreadQuit = SDL_FALSE;
static GUID GUID_DEVINTERFACE_HID = { 0x4D1E55B2L, 0xF16F, 0x11CF, { 0x88, 0xCB, 0x00, 0x11, 0x11, 0x00, 0x00, 0x30 } };

JoyStick_DeviceData *SYS_Joystick; /* array to hold joystick ID values */

#if !defined(__WINRT__) && !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
static HMODULE cfgmgr32_lib_handle;
static CM_Register_NotificationFunc CM_Register_Notification;
static CM_Unregister_NotificationFunc CM_Unregister_Notification;
static HCMNOTIFICATION s_DeviceNotificationFuncHandle;

void WINDOWS_RAWINPUTEnabledChanged(void)
{
    s_bWindowsDeviceChanged = SDL_TRUE;
}

static DWORD CALLBACK SDL_DeviceNotificationFunc(HCMNOTIFICATION hNotify, PVOID context, CM_NOTIFY_ACTION action, PCM_NOTIFY_EVENT_DATA eventData, DWORD event_data_size)
{
    if (action == CM_NOTIFY_ACTION_DEVICEINTERFACEARRIVAL ||
        action == CM_NOTIFY_ACTION_DEVICEINTERFACEREMOVAL) {
        s_bWindowsDeviceChanged = SDL_TRUE;
    }
    return ERROR_SUCCESS;
}

static void SDL_CleanupDeviceNotificationFunc(void)
{
    if (cfgmgr32_lib_handle) {
        if (s_DeviceNotificationFuncHandle != NULL && CM_Unregister_Notification) {
            CM_Unregister_Notification(s_DeviceNotificationFuncHandle);
            s_DeviceNotificationFuncHandle = NULL;
        }

        FreeLibrary(cfgmgr32_lib_handle);
        cfgmgr32_lib_handle = NULL;
    }
}

static SDL_bool SDL_CreateDeviceNotificationFunc(void)
{
    cfgmgr32_lib_handle = LoadLibraryA("cfgmgr32.dll");
    if (cfgmgr32_lib_handle) {
        CM_Register_Notification = (CM_Register_NotificationFunc)GetProcAddress(cfgmgr32_lib_handle, "CM_Register_Notification");
        CM_Unregister_Notification = (CM_Unregister_NotificationFunc)GetProcAddress(cfgmgr32_lib_handle, "CM_Unregister_Notification");
        if (CM_Register_Notification && CM_Unregister_Notification) {
            CM_NOTIFY_FILTER notify_filter;

            SDL_zero(notify_filter);
            notify_filter.cbSize = sizeof(notify_filter);
            notify_filter.FilterType = CM_NOTIFY_FILTER_TYPE_DEVICEINTERFACE;
            notify_filter.u.DeviceInterface.ClassGuid = GUID_DEVINTERFACE_HID;
            if (CM_Register_Notification(&notify_filter, NULL, SDL_DeviceNotificationFunc, &s_DeviceNotificationFuncHandle) == CR_SUCCESS) {
                return SDL_TRUE;
            }
        }
    }

    SDL_CleanupDeviceNotificationFunc();
    return SDL_FALSE;
}

typedef struct
{
    HRESULT coinitialized;
    WNDCLASSEX wincl;
    HWND messageWindow;
    HDEVNOTIFY hNotify;
} SDL_DeviceNotificationData;

#define IDT_SDL_DEVICE_CHANGE_TIMER_1 1200
#define IDT_SDL_DEVICE_CHANGE_TIMER_2 1201

/* windowproc for our joystick detect thread message only window, to detect any USB device addition/removal */
static LRESULT CALLBACK SDL_PrivateJoystickDetectProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
    case WM_DEVICECHANGE:
        switch (wParam) {
        case DBT_DEVICEARRIVAL:
        case DBT_DEVICEREMOVECOMPLETE:
            if (((DEV_BROADCAST_HDR *)lParam)->dbch_devicetype == DBT_DEVTYP_DEVICEINTERFACE) {
                /* notify 300ms and 2 seconds later to ensure all APIs have updated status */
                SetTimer(hwnd, IDT_SDL_DEVICE_CHANGE_TIMER_1, 300, NULL);
                SetTimer(hwnd, IDT_SDL_DEVICE_CHANGE_TIMER_2, 2000, NULL);
            }
            break;
        }
        return 0;
    case WM_TIMER:
        if (wParam == IDT_SDL_DEVICE_CHANGE_TIMER_1 ||
            wParam == IDT_SDL_DEVICE_CHANGE_TIMER_2) {
            KillTimer(hwnd, wParam);
            s_bWindowsDeviceChanged = SDL_TRUE;
            return 0;
        }
        break;
    }

#ifdef SDL_JOYSTICK_RAWINPUT
    return CallWindowProc(RAWINPUT_WindowProc, hwnd, msg, wParam, lParam);
#else
    return CallWindowProc(DefWindowProc, hwnd, msg, wParam, lParam);
#endif
}

static void SDL_CleanupDeviceNotification(SDL_DeviceNotificationData *data)
{
#ifdef SDL_JOYSTICK_RAWINPUT
    RAWINPUT_UnregisterNotifications();
#endif

    if (data->hNotify) {
        UnregisterDeviceNotification(data->hNotify);
    }

    if (data->messageWindow) {
        DestroyWindow(data->messageWindow);
    }

    UnregisterClass(data->wincl.lpszClassName, data->wincl.hInstance);

    if (data->coinitialized == S_OK) {
        WIN_CoUninitialize();
    }
}

static int SDL_CreateDeviceNotification(SDL_DeviceNotificationData *data)
{
    DEV_BROADCAST_DEVICEINTERFACE dbh;

    SDL_zerop(data);

    data->coinitialized = WIN_CoInitialize();

    data->wincl.hInstance = GetModuleHandle(NULL);
    data->wincl.lpszClassName = TEXT("Message");
    data->wincl.lpfnWndProc = SDL_PrivateJoystickDetectProc; /* This function is called by windows */
    data->wincl.cbSize = sizeof(WNDCLASSEX);

    if (!RegisterClassEx(&data->wincl)) {
        WIN_SetError("Failed to create register class for joystick autodetect");
        SDL_CleanupDeviceNotification(data);
        return -1;
    }

    data->messageWindow = CreateWindowEx(0, TEXT("Message"), NULL, 0, 0, 0, 0, 0, HWND_MESSAGE, NULL, NULL, NULL);
    if (!data->messageWindow) {
        WIN_SetError("Failed to create message window for joystick autodetect");
        SDL_CleanupDeviceNotification(data);
        return -1;
    }

    SDL_zero(dbh);
    dbh.dbcc_size = sizeof(dbh);
    dbh.dbcc_devicetype = DBT_DEVTYP_DEVICEINTERFACE;
    dbh.dbcc_classguid = GUID_DEVINTERFACE_HID;

    data->hNotify = RegisterDeviceNotification(data->messageWindow, &dbh, DEVICE_NOTIFY_WINDOW_HANDLE);
    if (!data->hNotify) {
        WIN_SetError("Failed to create notify device for joystick autodetect");
        SDL_CleanupDeviceNotification(data);
        return -1;
    }

#ifdef SDL_JOYSTICK_RAWINPUT
    RAWINPUT_RegisterNotifications(data->messageWindow);
#endif
    return 0;
}

static SDL_bool SDL_WaitForDeviceNotification(SDL_DeviceNotificationData *data, SDL_mutex *mutex)
{
    MSG msg;
    int lastret = 1;

    if (!data->messageWindow) {
        return SDL_FALSE; /* device notifications require a window */
    }

    SDL_UnlockMutex(mutex);
    while (lastret > 0 && s_bWindowsDeviceChanged == SDL_FALSE) {
        lastret = GetMessage(&msg, NULL, 0, 0); /* WM_QUIT causes return value of 0 */
        if (lastret > 0) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }
    SDL_LockMutex(mutex);
    return (lastret != -1) ? SDL_TRUE : SDL_FALSE;
}

#endif /* !defined(__WINRT__) && !defined(__XBOXONE__) && !defined(__XBOXSERIES__) */

#if !defined(__WINRT__)

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
static SDL_DeviceNotificationData s_notification_data;
#endif

/* Function/thread to scan the system for joysticks. */
static int SDLCALL SDL_JoystickThread(void *_data)
{
#ifdef SDL_JOYSTICK_XINPUT
    SDL_bool bOpenedXInputDevices[XUSER_MAX_COUNT];
    SDL_zeroa(bOpenedXInputDevices);
#endif

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
    if (SDL_CreateDeviceNotification(&s_notification_data) < 0) {
        return -1;
    }
#endif

    SDL_LockMutex(s_mutexJoyStickEnum);
    while (s_bJoystickThreadQuit == SDL_FALSE) {
#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
        if (SDL_WaitForDeviceNotification(&s_notification_data, s_mutexJoyStickEnum) == SDL_FALSE) {
#else
        {
#endif
#ifdef SDL_JOYSTICK_XINPUT
            /* WM_DEVICECHANGE not working, poll for new XINPUT controllers */
            SDL_CondWaitTimeout(s_condJoystickThread, s_mutexJoyStickEnum, 1000);
            if (SDL_XINPUT_Enabled()) {
                /* scan for any change in XInput devices */
                Uint8 userId;
                for (userId = 0; userId < XUSER_MAX_COUNT; userId++) {
                    XINPUT_CAPABILITIES capabilities;
                    const DWORD result = XINPUTGETCAPABILITIES(userId, XINPUT_FLAG_GAMEPAD, &capabilities);
                    const SDL_bool available = (result == ERROR_SUCCESS) ? SDL_TRUE : SDL_FALSE;
                    if (bOpenedXInputDevices[userId] != available) {
                        s_bWindowsDeviceChanged = SDL_TRUE;
                        bOpenedXInputDevices[userId] = available;
                    }
                }
            }
#else
            /* WM_DEVICECHANGE not working, no XINPUT, no point in keeping thread alive */
            break;
#endif /* SDL_JOYSTICK_XINPUT */
        }
    }

    SDL_UnlockMutex(s_mutexJoyStickEnum);

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
    SDL_CleanupDeviceNotification(&s_notification_data);
#endif

    return 1;
}

/* spin up the thread to detect hotplug of devices */
static int SDL_StartJoystickThread(void)
{
    s_mutexJoyStickEnum = SDL_CreateMutex();
    if (!s_mutexJoyStickEnum) {
        return -1;
    }

    s_condJoystickThread = SDL_CreateCond();
    if (!s_condJoystickThread) {
        return -1;
    }

    s_bJoystickThreadQuit = SDL_FALSE;
    s_joystickThread = SDL_CreateThreadInternal(SDL_JoystickThread, "SDL_joystick", 64 * 1024, NULL);
    if (!s_joystickThread) {
        return -1;
    }
    return 0;
}

static void SDL_StopJoystickThread(void)
{
    if (!s_joystickThread) {
        return;
    }

    SDL_LockMutex(s_mutexJoyStickEnum);
    s_bJoystickThreadQuit = SDL_TRUE;
    SDL_CondBroadcast(s_condJoystickThread); /* signal the joystick thread to quit */
    SDL_UnlockMutex(s_mutexJoyStickEnum);
    PostThreadMessage(SDL_GetThreadID(s_joystickThread), WM_QUIT, 0, 0);

    /* Unlock joysticks while the joystick thread finishes processing messages */
    SDL_AssertJoysticksLocked();
    SDL_UnlockJoysticks();
    SDL_WaitThread(s_joystickThread, NULL); /* wait for it to bugger off */
    SDL_LockJoysticks();

    SDL_DestroyCond(s_condJoystickThread);
    s_condJoystickThread = NULL;

    SDL_DestroyMutex(s_mutexJoyStickEnum);
    s_mutexJoyStickEnum = NULL;

    s_joystickThread = NULL;
}

#endif /* !defined(__WINRT__) */

void WINDOWS_AddJoystickDevice(JoyStick_DeviceData *device)
{
    device->send_add_event = SDL_TRUE;
    device->nInstanceID = SDL_GetNextJoystickInstanceID();
    device->pNext = SYS_Joystick;
    SYS_Joystick = device;
}

void WINDOWS_JoystickDetect(void);
void WINDOWS_JoystickQuit(void);

/* Function to scan the system for joysticks.
 * Joystick 0 should be the system default joystick.
 * It should return 0, or -1 on an unrecoverable fatal error.
 */
static int WINDOWS_JoystickInit(void)
{
    if (SDL_XINPUT_JoystickInit() < 0) {
        WINDOWS_JoystickQuit();
        return -1;
    }

    if (SDL_DINPUT_JoystickInit() < 0) {
        WINDOWS_JoystickQuit();
        return -1;
    }

    s_bWindowsDeviceChanged = SDL_TRUE; /* force a scan of the system for joysticks this first time */

    WINDOWS_JoystickDetect();

#if !defined(__WINRT__) && !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
    SDL_CreateDeviceNotificationFunc();

    s_bJoystickThread = SDL_GetHintBoolean(SDL_HINT_JOYSTICK_THREAD, SDL_FALSE);
    if (s_bJoystickThread) {
        if (SDL_StartJoystickThread() < 0) {
            return -1;
        }
    } else {
        if (SDL_CreateDeviceNotification(&s_notification_data) < 0) {
            return -1;
        }
    }
#endif

#if defined(__XBOXONE__) || defined(__XBOXSERIES__)
    /* On Xbox, force create the joystick thread for device detection (since other methods don't work */
    s_bJoystickThread = SDL_TRUE;
    if (SDL_StartJoystickThread() < 0) {
        return -1;
    }
#endif
    return 0;
}

/* return the number of joysticks that are connected right now */
static int WINDOWS_JoystickGetCount(void)
{
    int nJoysticks = 0;
    JoyStick_DeviceData *device = SYS_Joystick;
    while (device) {
        nJoysticks++;
        device = device->pNext;
    }

    return nJoysticks;
}

/* detect any new joysticks being inserted into the system */
void WINDOWS_JoystickDetect(void)
{
    JoyStick_DeviceData *pCurList = NULL;

    /* only enum the devices if the joystick thread told us something changed */
    if (!s_bWindowsDeviceChanged) {
        return; /* thread hasn't signaled, nothing to do right now. */
    }

    if (s_mutexJoyStickEnum) {
        SDL_LockMutex(s_mutexJoyStickEnum);
    }

    s_bWindowsDeviceChanged = SDL_FALSE;

    pCurList = SYS_Joystick;
    SYS_Joystick = NULL;

    /* Look for DirectInput joysticks, wheels, head trackers, gamepads, etc.. */
    SDL_DINPUT_JoystickDetect(&pCurList);

    /* Look for XInput devices. Do this last, so they're first in the final list. */
    SDL_XINPUT_JoystickDetect(&pCurList);

    if (s_mutexJoyStickEnum) {
        SDL_UnlockMutex(s_mutexJoyStickEnum);
    }

    while (pCurList) {
        JoyStick_DeviceData *pListNext = NULL;

        if (pCurList->bXInputDevice) {
#ifdef SDL_HAPTIC_XINPUT
            SDL_XINPUT_HapticMaybeRemoveDevice(pCurList->XInputUserId);
#endif
        } else {
#ifdef SDL_HAPTIC_DINPUT
            SDL_DINPUT_HapticMaybeRemoveDevice(&pCurList->dxdevice);
#endif
        }

        SDL_PrivateJoystickRemoved(pCurList->nInstanceID);

        pListNext = pCurList->pNext;
        SDL_free(pCurList->joystickname);
        SDL_free(pCurList);
        pCurList = pListNext;
    }

    for (pCurList = SYS_Joystick; pCurList; pCurList = pCurList->pNext) {
        if (pCurList->send_add_event) {
            if (pCurList->bXInputDevice) {
#ifdef SDL_HAPTIC_XINPUT
                SDL_XINPUT_HapticMaybeAddDevice(pCurList->XInputUserId);
#endif
            } else {
#ifdef SDL_HAPTIC_DINPUT
                SDL_DINPUT_HapticMaybeAddDevice(&pCurList->dxdevice);
#endif
            }

            SDL_PrivateJoystickAdded(pCurList->nInstanceID);

            pCurList->send_add_event = SDL_FALSE;
        }
    }
}

static const char *WINDOWS_JoystickGetDeviceName(int device_index)
{
    JoyStick_DeviceData *device = SYS_Joystick;
    int index;

    for (index = device_index; index > 0; index--) {
        device = device->pNext;
    }

    return device->joystickname;
}

static const char *WINDOWS_JoystickGetDevicePath(int device_index)
{
    JoyStick_DeviceData *device = SYS_Joystick;
    int index;

    for (index = device_index; index > 0; index--) {
        device = device->pNext;
    }

    return device->path;
}

static int WINDOWS_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    JoyStick_DeviceData *device = SYS_Joystick;
    int index;

    for (index = device_index; index > 0; index--) {
        device = device->pNext;
    }

    if (device->bXInputDevice) {
        /* The slot for XInput devices can change as controllers are seated */
        return SDL_XINPUT_GetSteamVirtualGamepadSlot(device->XInputUserId);
    } else {
        return device->steam_virtual_gamepad_slot;
    }
}

static int WINDOWS_JoystickGetDevicePlayerIndex(int device_index)
{
    JoyStick_DeviceData *device = SYS_Joystick;
    int index;

    for (index = device_index; index > 0; index--) {
        device = device->pNext;
    }

    return device->bXInputDevice ? (int)device->XInputUserId : -1;
}

static void WINDOWS_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
}

/* return the stable device guid for this device index */
static SDL_JoystickGUID WINDOWS_JoystickGetDeviceGUID(int device_index)
{
    JoyStick_DeviceData *device = SYS_Joystick;
    int index;

    for (index = device_index; index > 0; index--) {
        device = device->pNext;
    }

    return device->guid;
}

/* Function to perform the mapping between current device instance and this joysticks instance id */
static SDL_JoystickID WINDOWS_JoystickGetDeviceInstanceID(int device_index)
{
    JoyStick_DeviceData *device = SYS_Joystick;
    int index;

    for (index = device_index; index > 0; index--) {
        device = device->pNext;
    }

    return device->nInstanceID;
}

/* Function to open a joystick for use.
   The joystick to open is specified by the device index.
   This should fill the nbuttons and naxes fields of the joystick structure.
   It returns 0, or -1 if there is an error.
 */
static int WINDOWS_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    JoyStick_DeviceData *device = SYS_Joystick;
    int index;

    for (index = device_index; index > 0; index--) {
        device = device->pNext;
    }

    /* allocate memory for system specific hardware data */
    joystick->instance_id = device->nInstanceID;
    joystick->hwdata = (struct joystick_hwdata *)SDL_calloc(1, sizeof(struct joystick_hwdata));
    if (!joystick->hwdata) {
        return SDL_OutOfMemory();
    }
    joystick->hwdata->guid = device->guid;

    if (device->bXInputDevice) {
        return SDL_XINPUT_JoystickOpen(joystick, device);
    } else {
        return SDL_DINPUT_JoystickOpen(joystick, device);
    }
}

static int WINDOWS_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    if (joystick->hwdata->bXInputDevice) {
        return SDL_XINPUT_JoystickRumble(joystick, low_frequency_rumble, high_frequency_rumble);
    } else {
        return SDL_DINPUT_JoystickRumble(joystick, low_frequency_rumble, high_frequency_rumble);
    }
}

static int WINDOWS_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 WINDOWS_JoystickGetCapabilities(SDL_Joystick *joystick)
{
    if (joystick->hwdata->bXInputDevice) {
        return SDL_XINPUT_JoystickGetCapabilities(joystick);
    } else {
        return SDL_DINPUT_JoystickGetCapabilities(joystick);
    }
}

static int WINDOWS_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static int WINDOWS_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static int WINDOWS_JoystickSetSensorsEnabled(SDL_Joystick *joystick, SDL_bool enabled)
{
    return SDL_Unsupported();
}

static void WINDOWS_JoystickUpdate(SDL_Joystick *joystick)
{
    if (!joystick->hwdata) {
        return;
    }

    if (joystick->hwdata->bXInputDevice) {
        SDL_XINPUT_JoystickUpdate(joystick);
    } else {
        SDL_DINPUT_JoystickUpdate(joystick);
    }
}

/* Function to close a joystick after use */
static void WINDOWS_JoystickClose(SDL_Joystick *joystick)
{
    if (joystick->hwdata->bXInputDevice) {
        SDL_XINPUT_JoystickClose(joystick);
    } else {
        SDL_DINPUT_JoystickClose(joystick);
    }

    SDL_free(joystick->hwdata);
}

/* Function to perform any system-specific joystick related cleanup */
void WINDOWS_JoystickQuit(void)
{
    JoyStick_DeviceData *device = SYS_Joystick;

    while (device) {
        JoyStick_DeviceData *device_next = device->pNext;
        SDL_free(device->joystickname);
        SDL_free(device);
        device = device_next;
    }
    SYS_Joystick = NULL;

#if !defined(__WINRT__) && !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
    if (s_bJoystickThread) {
        SDL_StopJoystickThread();
    } else {
        SDL_CleanupDeviceNotification(&s_notification_data);
    }

    SDL_CleanupDeviceNotificationFunc();
#endif

#if defined(__XBOXONE__) || defined(__XBOXSERIES__)
    if (s_bJoystickThread) {
        SDL_StopJoystickThread();
    }
#endif

    SDL_DINPUT_JoystickQuit();
    SDL_XINPUT_JoystickQuit();

    s_bWindowsDeviceChanged = SDL_FALSE;
}

static SDL_bool WINDOWS_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    return SDL_FALSE;
}

SDL_JoystickDriver SDL_WINDOWS_JoystickDriver = {
    WINDOWS_JoystickInit,
    WINDOWS_JoystickGetCount,
    WINDOWS_JoystickDetect,
    WINDOWS_JoystickGetDeviceName,
    WINDOWS_JoystickGetDevicePath,
    WINDOWS_JoystickGetDeviceSteamVirtualGamepadSlot,
    WINDOWS_JoystickGetDevicePlayerIndex,
    WINDOWS_JoystickSetDevicePlayerIndex,
    WINDOWS_JoystickGetDeviceGUID,
    WINDOWS_JoystickGetDeviceInstanceID,
    WINDOWS_JoystickOpen,
    WINDOWS_JoystickRumble,
    WINDOWS_JoystickRumbleTriggers,
    WINDOWS_JoystickGetCapabilities,
    WINDOWS_JoystickSetLED,
    WINDOWS_JoystickSendEffect,
    WINDOWS_JoystickSetSensorsEnabled,
    WINDOWS_JoystickUpdate,
    WINDOWS_JoystickClose,
    WINDOWS_JoystickQuit,
    WINDOWS_JoystickGetGamepadMapping
};

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif

#else

#ifdef SDL_JOYSTICK_RAWINPUT
/* The RAWINPUT driver needs the device notification setup above */
#error SDL_JOYSTICK_RAWINPUT requires SDL_JOYSTICK_DINPUT || defined(SDL_JOYSTICK_XINPUT)
#endif

#endif /* SDL_JOYSTICK_DINPUT || SDL_JOYSTICK_XINPUT */

/* vi: set ts=4 sw=4 expandtab: */
