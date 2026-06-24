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

/* Original hybrid wrapper for Linux by Valve Software. Their original notes:
 *
 * The libusb version doesn't support Bluetooth, but not all Linux
 * distributions allow access to /dev/hidraw*
 *
 * This merges the two, at a small performance cost, until distributions
 * have granted access to /dev/hidraw*
 */
#include "../SDL_internal.h"

#include "SDL_hidapi.h"
#include "SDL_hidapi_c.h"
#include "SDL_loadso.h"
#include "SDL_thread.h"
#include "SDL_timer.h"

#ifndef SDL_HIDAPI_DISABLED

#if defined(__WIN32__) || defined(__WINGDK__)
#include "../core/windows/SDL_windows.h"
#endif

#if defined(__MACOSX__)
#include <AvailabilityMacros.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/hid/IOHIDDevice.h>
#include <IOKit/usb/USBSpec.h>
#include <mach/mach.h>
/* Things named "Master" were renamed to "Main" in macOS 12.0's SDK. */
#if MAC_OS_X_VERSION_MIN_REQUIRED < 120000
#define kIOMainPortDefault kIOMasterPortDefault
#endif
#endif

#include "../core/linux/SDL_udev.h"
#ifdef SDL_USE_LIBUDEV
#include "../core/linux/SDL_sandbox.h"
#include <poll.h>
#endif

#ifdef HAVE_INOTIFY
#include <errno.h> /* errno, strerror */
#include <fcntl.h>
#include <limits.h> /* For the definition of NAME_MAX */
#include <sys/inotify.h>
#include <unistd.h> /* just in case we didn't use that SDL_USE_LIBUDEV block... */
#endif

#if defined(SDL_USE_LIBUDEV) || defined(HAVE_INOTIFY)
#include <unistd.h>
#endif

#if defined(SDL_USE_LIBUDEV)
typedef enum
{
    ENUMERATION_UNSET,
    ENUMERATION_LIBUDEV,
    ENUMERATION_FALLBACK
} LinuxEnumerationMethod;

static LinuxEnumerationMethod linux_enumeration_method = ENUMERATION_UNSET;
#endif

#if defined(HAVE_INOTIFY)
static int inotify_fd = -1;
#endif

#if defined(SDL_USE_LIBUDEV)
static const SDL_UDEV_Symbols *usyms = NULL;
#endif

static struct
{
    SDL_bool m_bInitialized;
    Uint32 m_unDeviceChangeCounter;
    SDL_bool m_bCanGetNotifications;
    Uint32 m_unLastDetect;

#if defined(__WIN32__) || defined(__WINGDK__)
    SDL_threadID m_nThreadID;
    WNDCLASSEXA m_wndClass;
    HWND m_hwndMsg;
    HDEVNOTIFY m_hNotify;
    double m_flLastWin32MessageCheck;
#endif

#if defined(__MACOSX__)
    IONotificationPortRef m_notificationPort;
    mach_port_t m_notificationMach;
#endif

#if defined(SDL_USE_LIBUDEV)
    struct udev *m_pUdev;
    struct udev_monitor *m_pUdevMonitor;
    int m_nUdevFd;
#endif
} SDL_HIDAPI_discovery;

#if defined(__WIN32__) || defined(__WINGDK__)
struct _DEV_BROADCAST_HDR
{
    DWORD dbch_size;
    DWORD dbch_devicetype;
    DWORD dbch_reserved;
};

typedef struct _DEV_BROADCAST_DEVICEINTERFACE_A
{
    DWORD dbcc_size;
    DWORD dbcc_devicetype;
    DWORD dbcc_reserved;
    GUID dbcc_classguid;
    char dbcc_name[1];
} DEV_BROADCAST_DEVICEINTERFACE_A, *PDEV_BROADCAST_DEVICEINTERFACE_A;

typedef struct _DEV_BROADCAST_HDR DEV_BROADCAST_HDR;
#define DBT_DEVICEARRIVAL          0x8000     /* system detected a new device */
#define DBT_DEVICEREMOVECOMPLETE   0x8004     /* device was removed from the system */
#define DBT_DEVTYP_DEVICEINTERFACE 0x00000005 /* device interface class */
#define DBT_DEVNODES_CHANGED       0x0007
#define DBT_CONFIGCHANGED          0x0018
#define DBT_DEVICETYPESPECIFIC     0x8005 /* type specific event */
#define DBT_DEVINSTSTARTED         0x8008 /* device installed and started */

#include <initguid.h>
DEFINE_GUID(GUID_DEVINTERFACE_USB_DEVICE, 0xA5DCBF10L, 0x6530, 0x11D2, 0x90, 0x1F, 0x00, 0xC0, 0x4F, 0xB9, 0x51, 0xED);

static LRESULT CALLBACK ControllerWndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message) {
    case WM_DEVICECHANGE:
        switch (wParam) {
        case DBT_DEVICEARRIVAL:
        case DBT_DEVICEREMOVECOMPLETE:
            if (((DEV_BROADCAST_HDR *)lParam)->dbch_devicetype == DBT_DEVTYP_DEVICEINTERFACE) {
                ++SDL_HIDAPI_discovery.m_unDeviceChangeCounter;
            }
            break;
        }
        return TRUE;
    }

    return DefWindowProc(hwnd, message, wParam, lParam);
}
#endif /* defined(__WIN32__) || defined(__WINGDK__) */

#if defined(__MACOSX__)
static void CallbackIOServiceFunc(void *context, io_iterator_t portIterator)
{
    /* Must drain the iterator, or we won't receive new notifications */
    io_object_t entry;
    while ((entry = IOIteratorNext(portIterator)) != 0) {
        IOObjectRelease(entry);
        ++SDL_HIDAPI_discovery.m_unDeviceChangeCounter;
    }
}
#endif /* __MACOSX__ */

#ifdef HAVE_INOTIFY
#ifdef HAVE_INOTIFY_INIT1
static int SDL_inotify_init1(void)
{
    return inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
}
#else
static int SDL_inotify_init1(void)
{
    int fd = inotify_init();
    if (fd < 0) {
        return -1;
    }
    fcntl(fd, F_SETFL, O_NONBLOCK);
    fcntl(fd, F_SETFD, FD_CLOEXEC);
    return fd;
}
#endif

static int StrHasPrefix(const char *string, const char *prefix)
{
    return SDL_strncmp(string, prefix, SDL_strlen(prefix)) == 0;
}

static int StrIsInteger(const char *string)
{
    const char *p;

    if (*string == '\0') {
        return 0;
    }

    for (p = string; *p != '\0'; p++) {
        if (*p < '0' || *p > '9') {
            return 0;
        }
    }

    return 1;
}
#endif /* HAVE_INOTIFY */

static void HIDAPI_InitializeDiscovery(void)
{
    SDL_HIDAPI_discovery.m_bInitialized = SDL_TRUE;
    SDL_HIDAPI_discovery.m_unDeviceChangeCounter = 1;
    SDL_HIDAPI_discovery.m_bCanGetNotifications = SDL_FALSE;
    SDL_HIDAPI_discovery.m_unLastDetect = 0;

#if defined(__WIN32__) || defined(__WINGDK__)
    SDL_HIDAPI_discovery.m_nThreadID = SDL_ThreadID();

    SDL_zero(SDL_HIDAPI_discovery.m_wndClass);
    SDL_HIDAPI_discovery.m_wndClass.hInstance = GetModuleHandle(NULL);
    SDL_HIDAPI_discovery.m_wndClass.lpszClassName = "SDL_HIDAPI_DEVICE_DETECTION";
    SDL_HIDAPI_discovery.m_wndClass.lpfnWndProc = ControllerWndProc; /* This function is called by windows */
    SDL_HIDAPI_discovery.m_wndClass.cbSize = sizeof(WNDCLASSEX);

    RegisterClassExA(&SDL_HIDAPI_discovery.m_wndClass);
    SDL_HIDAPI_discovery.m_hwndMsg = CreateWindowExA(0, "SDL_HIDAPI_DEVICE_DETECTION", NULL, 0, 0, 0, 0, 0, HWND_MESSAGE, NULL, NULL, NULL);

    {
        DEV_BROADCAST_DEVICEINTERFACE_A devBroadcast;

        SDL_zero(devBroadcast);
        devBroadcast.dbcc_size = sizeof(devBroadcast);
        devBroadcast.dbcc_devicetype = DBT_DEVTYP_DEVICEINTERFACE;
        devBroadcast.dbcc_classguid = GUID_DEVINTERFACE_USB_DEVICE;

        /* DEVICE_NOTIFY_ALL_INTERFACE_CLASSES is important, makes GUID_DEVINTERFACE_USB_DEVICE ignored,
         * but that seems to be necessary to get a notice after each individual usb input device actually
         * installs, rather than just as the composite device is seen.
         */
        SDL_HIDAPI_discovery.m_hNotify = RegisterDeviceNotification(SDL_HIDAPI_discovery.m_hwndMsg, &devBroadcast, DEVICE_NOTIFY_WINDOW_HANDLE | DEVICE_NOTIFY_ALL_INTERFACE_CLASSES);
        SDL_HIDAPI_discovery.m_bCanGetNotifications = (SDL_HIDAPI_discovery.m_hNotify != 0);
    }
#endif /* defined(__WIN32__) || defined(__WINGDK__) */

#if defined(__MACOSX__)
    SDL_HIDAPI_discovery.m_notificationPort = IONotificationPortCreate(kIOMainPortDefault);
    if (SDL_HIDAPI_discovery.m_notificationPort) {
        {
            io_iterator_t portIterator = 0;
            io_object_t entry;
            IOReturn result = IOServiceAddMatchingNotification(
                SDL_HIDAPI_discovery.m_notificationPort,
                kIOFirstMatchNotification,
                IOServiceMatching(kIOHIDDeviceKey),
                CallbackIOServiceFunc, NULL, &portIterator);

            if (result == 0) {
                /* Must drain the existing iterator, or we won't receive new notifications */
                while ((entry = IOIteratorNext(portIterator)) != 0) {
                    IOObjectRelease(entry);
                }
            } else {
                IONotificationPortDestroy(SDL_HIDAPI_discovery.m_notificationPort);
                SDL_HIDAPI_discovery.m_notificationPort = nil;
            }
        }
        {
            io_iterator_t portIterator = 0;
            io_object_t entry;
            IOReturn result = IOServiceAddMatchingNotification(
                SDL_HIDAPI_discovery.m_notificationPort,
                kIOTerminatedNotification,
                IOServiceMatching(kIOHIDDeviceKey),
                CallbackIOServiceFunc, NULL, &portIterator);

            if (result == 0) {
                /* Must drain the existing iterator, or we won't receive new notifications */
                while ((entry = IOIteratorNext(portIterator)) != 0) {
                    IOObjectRelease(entry);
                }
            } else {
                IONotificationPortDestroy(SDL_HIDAPI_discovery.m_notificationPort);
                SDL_HIDAPI_discovery.m_notificationPort = nil;
            }
        }
    }

    SDL_HIDAPI_discovery.m_notificationMach = MACH_PORT_NULL;
    if (SDL_HIDAPI_discovery.m_notificationPort) {
        SDL_HIDAPI_discovery.m_notificationMach = IONotificationPortGetMachPort(SDL_HIDAPI_discovery.m_notificationPort);
    }

    SDL_HIDAPI_discovery.m_bCanGetNotifications = (SDL_HIDAPI_discovery.m_notificationMach != MACH_PORT_NULL);

#endif /* __MACOSX__ */

#if defined(SDL_USE_LIBUDEV)
    if (linux_enumeration_method == ENUMERATION_LIBUDEV) {
        SDL_HIDAPI_discovery.m_pUdev = NULL;
        SDL_HIDAPI_discovery.m_pUdevMonitor = NULL;
        SDL_HIDAPI_discovery.m_nUdevFd = -1;

        usyms = SDL_UDEV_GetUdevSyms();
        if (usyms != NULL) {
            SDL_HIDAPI_discovery.m_pUdev = usyms->udev_new();
            if (SDL_HIDAPI_discovery.m_pUdev != NULL) {
                SDL_HIDAPI_discovery.m_pUdevMonitor = usyms->udev_monitor_new_from_netlink(SDL_HIDAPI_discovery.m_pUdev, "udev");
                if (SDL_HIDAPI_discovery.m_pUdevMonitor != NULL) {
                    usyms->udev_monitor_enable_receiving(SDL_HIDAPI_discovery.m_pUdevMonitor);
                    SDL_HIDAPI_discovery.m_nUdevFd = usyms->udev_monitor_get_fd(SDL_HIDAPI_discovery.m_pUdevMonitor);
                    SDL_HIDAPI_discovery.m_bCanGetNotifications = SDL_TRUE;
                }
            }
        }
    } else
#endif /* SDL_USE_LIBUDEV */
    {
#if defined(HAVE_INOTIFY)
        inotify_fd = SDL_inotify_init1();

        if (inotify_fd < 0) {
            SDL_LogWarn(SDL_LOG_CATEGORY_INPUT,
                        "Unable to initialize inotify, falling back to polling: %s",
                        strerror(errno));
            return;
        }

        /* We need to watch for attribute changes in addition to
         * creation, because when a device is first created, it has
         * permissions that we can't read. When udev chmods it to
         * something that we maybe *can* read, we'll get an
         * IN_ATTRIB event to tell us. */
        if (inotify_add_watch(inotify_fd, "/dev",
                              IN_CREATE | IN_DELETE | IN_MOVE | IN_ATTRIB) < 0) {
            close(inotify_fd);
            inotify_fd = -1;
            SDL_LogWarn(SDL_LOG_CATEGORY_INPUT,
                        "Unable to add inotify watch, falling back to polling: %s",
                        strerror(errno));
            return;
        }

        SDL_HIDAPI_discovery.m_bCanGetNotifications = SDL_TRUE;
#endif /* HAVE_INOTIFY */
    }
}

static void HIDAPI_UpdateDiscovery(void)
{
    if (!SDL_HIDAPI_discovery.m_bInitialized) {
        HIDAPI_InitializeDiscovery();
    }

    if (!SDL_HIDAPI_discovery.m_bCanGetNotifications) {
        const Uint32 SDL_HIDAPI_DETECT_INTERVAL_MS = 3000; /* Update every 3 seconds */
        Uint32 now = SDL_GetTicks();
        if (!SDL_HIDAPI_discovery.m_unLastDetect || SDL_TICKS_PASSED(now, SDL_HIDAPI_discovery.m_unLastDetect + SDL_HIDAPI_DETECT_INTERVAL_MS)) {
            ++SDL_HIDAPI_discovery.m_unDeviceChangeCounter;
            SDL_HIDAPI_discovery.m_unLastDetect = now;
        }
        return;
    }

#if defined(__WIN32__) || defined(__WINGDK__)
#if 0 /* just let the usual SDL_PumpEvents loop dispatch these, fixing bug 4286. --ryan. */
    /* We'll only get messages on the same thread that created the window */
    if (SDL_ThreadID() == SDL_HIDAPI_discovery.m_nThreadID) {
        MSG msg;
        while (PeekMessage(&msg, SDL_HIDAPI_discovery.m_hwndMsg, 0, 0, PM_NOREMOVE)) {
            if (GetMessageA(&msg, SDL_HIDAPI_discovery.m_hwndMsg, 0, 0) != 0) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
        }
    }
#endif
#endif /* defined(__WIN32__) || defined(__WINGDK__) */

#if defined(__MACOSX__)
    if (SDL_HIDAPI_discovery.m_notificationPort) {
        struct
        {
            mach_msg_header_t hdr;
            char payload[4096];
        } msg;
        while (mach_msg(&msg.hdr, MACH_RCV_MSG | MACH_RCV_TIMEOUT, 0, sizeof(msg), SDL_HIDAPI_discovery.m_notificationMach, 0, MACH_PORT_NULL) == KERN_SUCCESS) {
            IODispatchCalloutFromMessage(NULL, &msg.hdr, SDL_HIDAPI_discovery.m_notificationPort);
        }
    }
#endif

#if defined(SDL_USE_LIBUDEV)
    if (linux_enumeration_method == ENUMERATION_LIBUDEV) {
        if (SDL_HIDAPI_discovery.m_nUdevFd >= 0) {
            /* Drain all notification events.
             * We don't expect a lot of device notifications so just
             * do a new discovery on any kind or number of notifications.
             * This could be made more restrictive if necessary.
             */
            for (;;) {
                struct pollfd PollUdev;
                struct udev_device *pUdevDevice;

                PollUdev.fd = SDL_HIDAPI_discovery.m_nUdevFd;
                PollUdev.events = POLLIN;
                if (poll(&PollUdev, 1, 0) != 1) {
                    break;
                }

                pUdevDevice = usyms->udev_monitor_receive_device(SDL_HIDAPI_discovery.m_pUdevMonitor);
                if (pUdevDevice) {
                    const char *action = NULL;
                    action = usyms->udev_device_get_action(pUdevDevice);
                    if (action == NULL || SDL_strcmp(action, "add") == 0 || SDL_strcmp(action, "remove") == 0) {
                        ++SDL_HIDAPI_discovery.m_unDeviceChangeCounter;
                    }
                    usyms->udev_device_unref(pUdevDevice);
                }
            }
        }
    } else
#endif /* SDL_USE_LIBUDEV */
    {
#if defined(HAVE_INOTIFY)
        if (inotify_fd >= 0) {
            union
            {
                struct inotify_event event;
                char storage[4096];
                char enough_for_inotify[sizeof(struct inotify_event) + NAME_MAX + 1];
            } buf;
            ssize_t bytes;
            size_t remain = 0;
            size_t len;

            bytes = read(inotify_fd, &buf, sizeof(buf));

            if (bytes > 0) {
                remain = (size_t)bytes;
            }

            while (remain > 0) {
                if (buf.event.len > 0) {
                    if (StrHasPrefix(buf.event.name, "hidraw") &&
                        StrIsInteger(buf.event.name + SDL_strlen("hidraw"))) {
                        ++SDL_HIDAPI_discovery.m_unDeviceChangeCounter;
                        /* We found an hidraw change. We still continue to
                         * drain the inotify fd to avoid leaving old
                         * notifications in the queue. */
                    }
                }

                len = sizeof(struct inotify_event) + buf.event.len;
                remain -= len;

                if (remain != 0) {
                    SDL_memmove(&buf.storage[0], &buf.storage[len], remain);
                }
            }
        }
#endif /* HAVE_INOTIFY */
    }
}

static void HIDAPI_ShutdownDiscovery(void)
{
    if (!SDL_HIDAPI_discovery.m_bInitialized) {
        return;
    }

#if defined(__WIN32__) || defined(__WINGDK__)
    if (SDL_HIDAPI_discovery.m_hNotify) {
        UnregisterDeviceNotification(SDL_HIDAPI_discovery.m_hNotify);
    }

    if (SDL_HIDAPI_discovery.m_hwndMsg) {
        DestroyWindow(SDL_HIDAPI_discovery.m_hwndMsg);
    }

    UnregisterClassA(SDL_HIDAPI_discovery.m_wndClass.lpszClassName, SDL_HIDAPI_discovery.m_wndClass.hInstance);
#endif

#if defined(__MACOSX__)
    if (SDL_HIDAPI_discovery.m_notificationPort) {
        IONotificationPortDestroy(SDL_HIDAPI_discovery.m_notificationPort);
    }
#endif

#if defined(SDL_USE_LIBUDEV)
    if (linux_enumeration_method == ENUMERATION_LIBUDEV) {
        if (usyms) {
            if (SDL_HIDAPI_discovery.m_pUdevMonitor) {
                usyms->udev_monitor_unref(SDL_HIDAPI_discovery.m_pUdevMonitor);
            }
            if (SDL_HIDAPI_discovery.m_pUdev) {
                usyms->udev_unref(SDL_HIDAPI_discovery.m_pUdev);
            }
            SDL_UDEV_ReleaseUdevSyms();
            usyms = NULL;
        }
    } else
#endif /* SDL_USE_LIBUDEV */
    {
#if defined(HAVE_INOTIFY)
        if (inotify_fd >= 0) {
            close(inotify_fd);
            inotify_fd = -1;
        }
#endif
    }

    SDL_HIDAPI_discovery.m_bInitialized = SDL_FALSE;
}

/* Platform HIDAPI Implementation */

#define hid_device                   PLATFORM_hid_device
#define hid_device_                  PLATFORM_hid_device_
#define hid_init                     PLATFORM_hid_init
#define hid_exit                     PLATFORM_hid_exit
#define hid_enumerate                PLATFORM_hid_enumerate
#define hid_free_enumeration         PLATFORM_hid_free_enumeration
#define hid_open                     PLATFORM_hid_open
#define hid_open_path                PLATFORM_hid_open_path
#define hid_write                    PLATFORM_hid_write
#define hid_read_timeout             PLATFORM_hid_read_timeout
#define hid_read                     PLATFORM_hid_read
#define hid_set_nonblocking          PLATFORM_hid_set_nonblocking
#define hid_send_feature_report      PLATFORM_hid_send_feature_report
#define hid_get_feature_report       PLATFORM_hid_get_feature_report
#define hid_close                    PLATFORM_hid_close
#define hid_get_manufacturer_string  PLATFORM_hid_get_manufacturer_string
#define hid_get_product_string       PLATFORM_hid_get_product_string
#define hid_get_serial_number_string PLATFORM_hid_get_serial_number_string
#define hid_get_indexed_string       PLATFORM_hid_get_indexed_string
#define hid_error                    PLATFORM_hid_error
#define new_hid_device               PLATFORM_new_hid_device
#define free_hid_device              PLATFORM_free_hid_device
#define input_report                 PLATFORM_input_report
#define return_data                  PLATFORM_return_data
#define make_path                    PLATFORM_make_path
#define read_thread                  PLATFORM_read_thread

#undef HIDAPI_H__
#ifdef __LINUX__

#ifdef SDL_USE_LIBUDEV
static const SDL_UDEV_Symbols *udev_ctx = NULL;

#define udev_device_get_sysattr_value                 udev_ctx->udev_device_get_sysattr_value
#define udev_new                                      udev_ctx->udev_new
#define udev_unref                                    udev_ctx->udev_unref
#define udev_device_new_from_devnum                   udev_ctx->udev_device_new_from_devnum
#define udev_device_get_parent_with_subsystem_devtype udev_ctx->udev_device_get_parent_with_subsystem_devtype
#define udev_device_unref                             udev_ctx->udev_device_unref
#define udev_enumerate_new                            udev_ctx->udev_enumerate_new
#define udev_enumerate_add_match_subsystem            udev_ctx->udev_enumerate_add_match_subsystem
#define udev_enumerate_scan_devices                   udev_ctx->udev_enumerate_scan_devices
#define udev_enumerate_get_list_entry                 udev_ctx->udev_enumerate_get_list_entry
#define udev_list_entry_get_name                      udev_ctx->udev_list_entry_get_name
#define udev_device_new_from_syspath                  udev_ctx->udev_device_new_from_syspath
#define udev_device_get_devnode                       udev_ctx->udev_device_get_devnode
#define udev_list_entry_get_next                      udev_ctx->udev_list_entry_get_next
#define udev_enumerate_unref                          udev_ctx->udev_enumerate_unref

#include "linux/hid.c"
#define HAVE_PLATFORM_BACKEND
#endif /* SDL_USE_LIBUDEV */

#elif defined(__MACOSX__)
#include "mac/hid.c"
#define HAVE_PLATFORM_BACKEND
#define udev_ctx              1
#elif defined(__WINDOWS__) || defined(__WINGDK__)
#include "windows/hid.c"
#define HAVE_PLATFORM_BACKEND
#define udev_ctx              1
#elif defined(__ANDROID__)
/* The implementation for Android is in a separate .cpp file */
#include "hidapi/hidapi.h"
#define HAVE_PLATFORM_BACKEND 1
#define udev_ctx              1
#elif defined(__IPHONEOS__) || defined(__TVOS__)
/* The implementation for iOS and tvOS is in a separate .m file */
#include "hidapi/hidapi.h"
#define HAVE_PLATFORM_BACKEND 1
#define udev_ctx              1
#endif

#undef hid_device
#undef hid_device_
#undef hid_init
#undef hid_exit
#undef hid_enumerate
#undef hid_free_enumeration
#undef hid_open
#undef hid_open_path
#undef hid_write
#undef hid_read_timeout
#undef hid_read
#undef hid_set_nonblocking
#undef hid_send_feature_report
#undef hid_get_feature_report
#undef hid_close
#undef hid_get_manufacturer_string
#undef hid_get_product_string
#undef hid_get_serial_number_string
#undef hid_get_indexed_string
#undef hid_error
#undef new_hid_device
#undef free_hid_device
#undef input_report
#undef return_data
#undef make_path
#undef read_thread

#ifdef SDL_JOYSTICK_HIDAPI_STEAMXBOX
#define HAVE_DRIVER_BACKEND 1
#else
#define HAVE_DRIVER_BACKEND 0
#endif

#if HAVE_DRIVER_BACKEND

/* DRIVER HIDAPI Implementation */

#define hid_device                   DRIVER_hid_device
#define hid_device_                  DRIVER_hid_device_
#define hid_init                     DRIVER_hid_init
#define hid_exit                     DRIVER_hid_exit
#define hid_enumerate                DRIVER_hid_enumerate
#define hid_free_enumeration         DRIVER_hid_free_enumeration
#define hid_open                     DRIVER_hid_open
#define hid_open_path                DRIVER_hid_open_path
#define hid_write                    DRIVER_hid_write
#define hid_read_timeout             DRIVER_hid_read_timeout
#define hid_read                     DRIVER_hid_read
#define hid_set_nonblocking          DRIVER_hid_set_nonblocking
#define hid_send_feature_report      DRIVER_hid_send_feature_report
#define hid_get_feature_report       DRIVER_hid_get_feature_report
#define hid_close                    DRIVER_hid_close
#define hid_get_manufacturer_string  DRIVER_hid_get_manufacturer_string
#define hid_get_product_string       DRIVER_hid_get_product_string
#define hid_get_serial_number_string DRIVER_hid_get_serial_number_string
#define hid_get_indexed_string       DRIVER_hid_get_indexed_string
#define hid_error                    DRIVER_hid_error

#ifdef SDL_JOYSTICK_HIDAPI_STEAMXBOX
#undef HIDAPI_H__
#include "steamxbox/hid.c"
#else
#error Need a driver hid.c for this platform!
#endif

#undef hid_device
#undef hid_device_
#undef hid_init
#undef hid_exit
#undef hid_enumerate
#undef hid_free_enumeration
#undef hid_open
#undef hid_open_path
#undef hid_write
#undef hid_read_timeout
#undef hid_read
#undef hid_set_nonblocking
#undef hid_send_feature_report
#undef hid_get_feature_report
#undef hid_close
#undef hid_get_manufacturer_string
#undef hid_get_product_string
#undef hid_get_serial_number_string
#undef hid_get_indexed_string
#undef hid_error

#endif /* HAVE_DRIVER_BACKEND */

#ifdef HAVE_LIBUSB
/* libusb HIDAPI Implementation */

/* Include this now, for our dynamically-loaded libusb context */
#include <libusb.h>

static struct
{
    void *libhandle;

    /* *INDENT-OFF* */ /* clang-format off */
    int (LIBUSB_CALL *init)(libusb_context **ctx);
    void (LIBUSB_CALL *exit)(libusb_context *ctx);
    ssize_t (LIBUSB_CALL *get_device_list)(libusb_context *ctx, libusb_device ***list);
    void (LIBUSB_CALL *free_device_list)(libusb_device **list, int unref_devices);
    int (LIBUSB_CALL *get_device_descriptor)(libusb_device *dev, struct libusb_device_descriptor *desc);
    int (LIBUSB_CALL *get_active_config_descriptor)(libusb_device *dev,    struct libusb_config_descriptor **config);
    int (LIBUSB_CALL *get_config_descriptor)(
        libusb_device *dev,
        uint8_t config_index,
        struct libusb_config_descriptor **config
    );
    void (LIBUSB_CALL *free_config_descriptor)(struct libusb_config_descriptor *config);
    uint8_t (LIBUSB_CALL *get_bus_number)(libusb_device *dev);
    uint8_t (LIBUSB_CALL *get_device_address)(libusb_device *dev);
    int (LIBUSB_CALL *open)(libusb_device *dev, libusb_device_handle **dev_handle);
    void (LIBUSB_CALL *close)(libusb_device_handle *dev_handle);
    int (LIBUSB_CALL *claim_interface)(libusb_device_handle *dev_handle, int interface_number);
    int (LIBUSB_CALL *release_interface)(libusb_device_handle *dev_handle, int interface_number);
    int (LIBUSB_CALL *kernel_driver_active)(libusb_device_handle *dev_handle, int interface_number);
    int (LIBUSB_CALL *detach_kernel_driver)(libusb_device_handle *dev_handle, int interface_number);
    int (LIBUSB_CALL *attach_kernel_driver)(libusb_device_handle *dev_handle, int interface_number);
    int (LIBUSB_CALL *set_interface_alt_setting)(libusb_device_handle *dev, int interface_number, int alternate_setting);
    struct libusb_transfer * (LIBUSB_CALL *alloc_transfer)(int iso_packets);
    int (LIBUSB_CALL *submit_transfer)(struct libusb_transfer *transfer);
    int (LIBUSB_CALL *cancel_transfer)(struct libusb_transfer *transfer);
    void (LIBUSB_CALL *free_transfer)(struct libusb_transfer *transfer);
    int (LIBUSB_CALL *control_transfer)(
        libusb_device_handle *dev_handle,
        uint8_t request_type,
        uint8_t bRequest,
        uint16_t wValue,
        uint16_t wIndex,
        unsigned char *data,
        uint16_t wLength,
        unsigned int timeout
    );
    int (LIBUSB_CALL *interrupt_transfer)(
        libusb_device_handle *dev_handle,
        unsigned char endpoint,
        unsigned char *data,
        int length,
        int *actual_length,
        unsigned int timeout
    );
    int (LIBUSB_CALL *handle_events)(libusb_context *ctx);
    int (LIBUSB_CALL *handle_events_completed)(libusb_context *ctx, int *completed);
    const char * (LIBUSB_CALL *error_name)(int errcode);
/* *INDENT-ON* */ /* clang-format on */

} libusb_ctx;

#define libusb_init                         libusb_ctx.init
#define libusb_exit                         libusb_ctx.exit
#define libusb_get_device_list              libusb_ctx.get_device_list
#define libusb_free_device_list             libusb_ctx.free_device_list
#define libusb_get_device_descriptor        libusb_ctx.get_device_descriptor
#define libusb_get_active_config_descriptor libusb_ctx.get_active_config_descriptor
#define libusb_get_config_descriptor        libusb_ctx.get_config_descriptor
#define libusb_free_config_descriptor       libusb_ctx.free_config_descriptor
#define libusb_get_bus_number               libusb_ctx.get_bus_number
#define libusb_get_device_address           libusb_ctx.get_device_address
#define libusb_open                         libusb_ctx.open
#define libusb_close                        libusb_ctx.close
#define libusb_claim_interface              libusb_ctx.claim_interface
#define libusb_release_interface            libusb_ctx.release_interface
#define libusb_kernel_driver_active         libusb_ctx.kernel_driver_active
#define libusb_detach_kernel_driver         libusb_ctx.detach_kernel_driver
#define libusb_attach_kernel_driver         libusb_ctx.attach_kernel_driver
#define libusb_set_interface_alt_setting    libusb_ctx.set_interface_alt_setting
#define libusb_alloc_transfer               libusb_ctx.alloc_transfer
#define libusb_submit_transfer              libusb_ctx.submit_transfer
#define libusb_cancel_transfer              libusb_ctx.cancel_transfer
#define libusb_free_transfer                libusb_ctx.free_transfer
#define libusb_control_transfer             libusb_ctx.control_transfer
#define libusb_interrupt_transfer           libusb_ctx.interrupt_transfer
#define libusb_handle_events                libusb_ctx.handle_events
#define libusb_handle_events_completed      libusb_ctx.handle_events_completed
#define libusb_error_name                   libusb_ctx.error_name

#define hid_device                   LIBUSB_hid_device
#define hid_device_                  LIBUSB_hid_device_
#define hid_init                     LIBUSB_hid_init
#define hid_exit                     LIBUSB_hid_exit
#define hid_enumerate                LIBUSB_hid_enumerate
#define hid_free_enumeration         LIBUSB_hid_free_enumeration
#define hid_open                     LIBUSB_hid_open
#define hid_open_path                LIBUSB_hid_open_path
#define hid_write                    LIBUSB_hid_write
#define hid_read_timeout             LIBUSB_hid_read_timeout
#define hid_read                     LIBUSB_hid_read
#define hid_set_nonblocking          LIBUSB_hid_set_nonblocking
#define hid_send_feature_report      LIBUSB_hid_send_feature_report
#define hid_get_feature_report       LIBUSB_hid_get_feature_report
#define hid_close                    LIBUSB_hid_close
#define hid_get_manufacturer_string  LIBUSB_hid_get_manufacturer_string
#define hid_get_product_string       LIBUSB_hid_get_product_string
#define hid_get_serial_number_string LIBUSB_hid_get_serial_number_string
#define hid_get_indexed_string       LIBUSB_hid_get_indexed_string
#define hid_error                    LIBUSB_hid_error
#define new_hid_device               LIBUSB_new_hid_device
#define free_hid_device              LIBUSB_free_hid_device
#define input_report                 LIBUSB_input_report
#define return_data                  LIBUSB_return_data
#define make_path                    LIBUSB_make_path
#define read_thread                  LIBUSB_read_thread

#ifndef __FreeBSD__
/* this is awkwardly inlined, so we need to re-implement it here
 * so we can override the libusb_control_transfer call */
static int SDL_libusb_get_string_descriptor(libusb_device_handle *dev,
                                            uint8_t descriptor_index, uint16_t lang_id,
                                            unsigned char *data, int length)
{
    return libusb_control_transfer(dev, LIBUSB_ENDPOINT_IN | 0x0, LIBUSB_REQUEST_GET_DESCRIPTOR, (LIBUSB_DT_STRING << 8) | descriptor_index, lang_id,
                                   data, (uint16_t)length, 1000); /* Endpoint 0 IN */
}
#define libusb_get_string_descriptor SDL_libusb_get_string_descriptor
#endif /* __FreeBSD__ */

#undef HIDAPI_H__
#include "libusb/hid.c"

#undef libusb_init
#undef libusb_exit
#undef libusb_get_device_list
#undef libusb_free_device_list
#undef libusb_get_device_descriptor
#undef libusb_get_active_config_descriptor
#undef libusb_get_config_descriptor
#undef libusb_free_config_descriptor
#undef libusb_get_bus_number
#undef libusb_get_device_address
#undef libusb_open
#undef libusb_close
#undef libusb_claim_interface
#undef libusb_release_interface
#undef libusb_kernel_driver_active
#undef libusb_detach_kernel_driver
#undef libusb_attach_kernel_driver
#undef libusb_set_interface_alt_setting
#undef libusb_alloc_transfer
#undef libusb_submit_transfer
#undef libusb_cancel_transfer
#undef libusb_free_transfer
#undef libusb_control_transfer
#undef libusb_interrupt_transfer
#undef libusb_handle_events
#undef libusb_handle_events_completed
#undef libusb_error_name

#undef hid_device
#undef hid_device_
#undef hid_init
#undef hid_exit
#undef hid_enumerate
#undef hid_free_enumeration
#undef hid_open
#undef hid_open_path
#undef hid_write
#undef hid_read_timeout
#undef hid_read
#undef hid_set_nonblocking
#undef hid_send_feature_report
#undef hid_get_feature_report
#undef hid_close
#undef hid_get_manufacturer_string
#undef hid_get_product_string
#undef hid_get_serial_number_string
#undef hid_get_indexed_string
#undef hid_error
#undef new_hid_device
#undef free_hid_device
#undef input_report
#undef return_data
#undef make_path
#undef read_thread

/* If the platform has any backend other than libusb, try to avoid using
 * libusb as the main backend for devices, since it detaches drivers and
 * therefore makes devices inaccessible to the rest of the OS.
 *
 * We do this by whitelisting devices we know to be accessible _exclusively_
 * via libusb; these are typically devices that look like HIDs but have a
 * quirk that requires direct access to the hardware.
 */
static const struct {
    Uint16 vendor;
    Uint16 product;
} SDL_libusb_whitelist[] = {
    { 0x057e, 0x0337 } /* Nintendo WUP-028, Wii U/Switch GameCube Adapter */
};

static SDL_bool
IsInWhitelist(Uint16 vendor, Uint16 product)
{
    int i;
    for (i = 0; i < SDL_arraysize(SDL_libusb_whitelist); i += 1) {
        if (vendor == SDL_libusb_whitelist[i].vendor &&
            product == SDL_libusb_whitelist[i].product) {
            return SDL_TRUE;
        }
    }
    return SDL_FALSE;
}

#if defined(HAVE_PLATFORM_BACKEND) || HAVE_DRIVER_BACKEND
 #define use_libusb_whitelist_default  SDL_TRUE
#else
 #define use_libusb_whitelist_default SDL_FALSE
#endif /* HAVE_PLATFORM_BACKEND || HAVE_DRIVER_BACKEND */
static SDL_bool use_libusb_whitelist = use_libusb_whitelist_default;

#endif /* HAVE_LIBUSB */

#endif /* !SDL_HIDAPI_DISABLED */

/* Shared HIDAPI Implementation */

struct hidapi_backend
{
    int (*hid_write)(void *device, const unsigned char *data, size_t length);
    int (*hid_read_timeout)(void *device, unsigned char *data, size_t length, int milliseconds);
    int (*hid_read)(void *device, unsigned char *data, size_t length);
    int (*hid_set_nonblocking)(void *device, int nonblock);
    int (*hid_send_feature_report)(void *device, const unsigned char *data, size_t length);
    int (*hid_get_feature_report)(void *device, unsigned char *data, size_t length);
    void (*hid_close)(void *device);
    int (*hid_get_manufacturer_string)(void *device, wchar_t *string, size_t maxlen);
    int (*hid_get_product_string)(void *device, wchar_t *string, size_t maxlen);
    int (*hid_get_serial_number_string)(void *device, wchar_t *string, size_t maxlen);
    int (*hid_get_indexed_string)(void *device, int string_index, wchar_t *string, size_t maxlen);
    const wchar_t *(*hid_error)(void *device);
};

#ifdef HAVE_PLATFORM_BACKEND
static const struct hidapi_backend PLATFORM_Backend = {
    (void *)PLATFORM_hid_write,
    (void *)PLATFORM_hid_read_timeout,
    (void *)PLATFORM_hid_read,
    (void *)PLATFORM_hid_set_nonblocking,
    (void *)PLATFORM_hid_send_feature_report,
    (void *)PLATFORM_hid_get_feature_report,
    (void *)PLATFORM_hid_close,
    (void *)PLATFORM_hid_get_manufacturer_string,
    (void *)PLATFORM_hid_get_product_string,
    (void *)PLATFORM_hid_get_serial_number_string,
    (void *)PLATFORM_hid_get_indexed_string,
    (void *)PLATFORM_hid_error
};
#endif /* HAVE_PLATFORM_BACKEND */

#if HAVE_DRIVER_BACKEND
static const struct hidapi_backend DRIVER_Backend = {
    (void *)DRIVER_hid_write,
    (void *)DRIVER_hid_read_timeout,
    (void *)DRIVER_hid_read,
    (void *)DRIVER_hid_set_nonblocking,
    (void *)DRIVER_hid_send_feature_report,
    (void *)DRIVER_hid_get_feature_report,
    (void *)DRIVER_hid_close,
    (void *)DRIVER_hid_get_manufacturer_string,
    (void *)DRIVER_hid_get_product_string,
    (void *)DRIVER_hid_get_serial_number_string,
    (void *)DRIVER_hid_get_indexed_string,
    (void *)DRIVER_hid_error
};
#endif /* HAVE_DRIVER_BACKEND */

#ifdef HAVE_LIBUSB
static const struct hidapi_backend LIBUSB_Backend = {
    (void *)LIBUSB_hid_write,
    (void *)LIBUSB_hid_read_timeout,
    (void *)LIBUSB_hid_read,
    (void *)LIBUSB_hid_set_nonblocking,
    (void *)LIBUSB_hid_send_feature_report,
    (void *)LIBUSB_hid_get_feature_report,
    (void *)LIBUSB_hid_close,
    (void *)LIBUSB_hid_get_manufacturer_string,
    (void *)LIBUSB_hid_get_product_string,
    (void *)LIBUSB_hid_get_serial_number_string,
    (void *)LIBUSB_hid_get_indexed_string,
    (void *)LIBUSB_hid_error
};
#endif /* HAVE_LIBUSB */

struct SDL_hid_device_
{
    const void *magic;
    void *device;
    const struct hidapi_backend *backend;
};
static char device_magic;

#if defined(HAVE_PLATFORM_BACKEND) || HAVE_DRIVER_BACKEND || defined(HAVE_LIBUSB)

static SDL_hid_device *CreateHIDDeviceWrapper(void *device, const struct hidapi_backend *backend)
{
    SDL_hid_device *wrapper = (SDL_hid_device *)SDL_malloc(sizeof(*wrapper));
    wrapper->magic = &device_magic;
    wrapper->device = device;
    wrapper->backend = backend;
    return wrapper;
}

#endif /* HAVE_PLATFORM_BACKEND || HAVE_DRIVER_BACKEND || HAVE_LIBUSB */

static void DeleteHIDDeviceWrapper(SDL_hid_device *device)
{
    device->magic = NULL;
    SDL_free(device);
}

#define CHECK_DEVICE_MAGIC(device, retval)           \
    if (!device || device->magic != &device_magic) { \
        SDL_SetError("Invalid device");              \
        return retval;                               \
    }

#ifndef SDL_HIDAPI_DISABLED
#if defined(HAVE_PLATFORM_BACKEND) || HAVE_DRIVER_BACKEND || defined(HAVE_LIBUSB)

#define COPY_IF_EXISTS(var)                \
    if (pSrc->var != NULL) {               \
        pDst->var = SDL_strdup(pSrc->var); \
    } else {                               \
        pDst->var = NULL;                  \
    }
#define WCOPY_IF_EXISTS(var)               \
    if (pSrc->var != NULL) {               \
        pDst->var = SDL_wcsdup(pSrc->var); \
    } else {                               \
        pDst->var = NULL;                  \
    }

static void CopyHIDDeviceInfo(struct SDL_hid_device_info *pSrc, struct SDL_hid_device_info *pDst)
{
    COPY_IF_EXISTS(path)
    pDst->vendor_id = pSrc->vendor_id;
    pDst->product_id = pSrc->product_id;
    WCOPY_IF_EXISTS(serial_number)
    pDst->release_number = pSrc->release_number;
    WCOPY_IF_EXISTS(manufacturer_string)
    WCOPY_IF_EXISTS(product_string)
    pDst->usage_page = pSrc->usage_page;
    pDst->usage = pSrc->usage;
    pDst->interface_number = pSrc->interface_number;
    pDst->interface_class = pSrc->interface_class;
    pDst->interface_subclass = pSrc->interface_subclass;
    pDst->interface_protocol = pSrc->interface_protocol;
    pDst->next = NULL;
}

#undef COPY_IF_EXISTS
#undef WCOPY_IF_EXISTS

#endif /* HAVE_PLATFORM_BACKEND || HAVE_DRIVER_BACKEND || HAVE_LIBUSB */
#endif /* !SDL_HIDAPI_DISABLED */

static int SDL_hidapi_refcount = 0;

static void SDL_SetHIDAPIError(const wchar_t *error)
{
    if (error) {
        char *error_utf8 = SDL_iconv_wchar_utf8(error);
        if (error_utf8) {
            SDL_SetError("%s", error_utf8);
            SDL_free(error_utf8);
        }
    }
}

int SDL_hid_init(void)
{
    int attempts = 0, success = 0;

    if (SDL_hidapi_refcount > 0) {
        ++SDL_hidapi_refcount;
        return 0;
    }

#if defined(SDL_USE_LIBUDEV)
    if (SDL_getenv("SDL_HIDAPI_JOYSTICK_DISABLE_UDEV") != NULL) {
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT,
                     "udev disabled by SDL_HIDAPI_JOYSTICK_DISABLE_UDEV");
        linux_enumeration_method = ENUMERATION_FALLBACK;
    } else if (SDL_DetectSandbox() != SDL_SANDBOX_NONE) {
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT,
                     "Container detected, disabling HIDAPI udev integration");
        linux_enumeration_method = ENUMERATION_FALLBACK;
    } else {
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT,
                     "Using udev for HIDAPI joystick device discovery");
        linux_enumeration_method = ENUMERATION_LIBUDEV;
    }
#endif

#ifdef HAVE_LIBUSB
    use_libusb_whitelist = SDL_GetHintBoolean("SDL_HIDAPI_LIBUSB_WHITELIST",
                                              use_libusb_whitelist_default);
    if (SDL_getenv("SDL_HIDAPI_DISABLE_LIBUSB") != NULL) {
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT,
                     "libusb disabled by SDL_HIDAPI_DISABLE_LIBUSB");
        libusb_ctx.libhandle = NULL;
    } else {
        ++attempts;
#ifdef SDL_LIBUSB_DYNAMIC
        libusb_ctx.libhandle = SDL_LoadObject(SDL_LIBUSB_DYNAMIC);
#else
        libusb_ctx.libhandle = (void *)1;
#endif
        if (libusb_ctx.libhandle != NULL) {
            SDL_bool loaded = SDL_TRUE;
#ifdef SDL_LIBUSB_DYNAMIC
#define LOAD_LIBUSB_SYMBOL(func)                                                        \
    if (!(libusb_ctx.func = SDL_LoadFunction(libusb_ctx.libhandle, "libusb_" #func))) { \
        loaded = SDL_FALSE;                                                             \
    }
#else
#define LOAD_LIBUSB_SYMBOL(func) \
    libusb_ctx.func = libusb_##func;
#endif
            LOAD_LIBUSB_SYMBOL(init)
            LOAD_LIBUSB_SYMBOL(exit)
            LOAD_LIBUSB_SYMBOL(get_device_list)
            LOAD_LIBUSB_SYMBOL(free_device_list)
            LOAD_LIBUSB_SYMBOL(get_device_descriptor)
            LOAD_LIBUSB_SYMBOL(get_active_config_descriptor)
            LOAD_LIBUSB_SYMBOL(get_config_descriptor)
            LOAD_LIBUSB_SYMBOL(free_config_descriptor)
            LOAD_LIBUSB_SYMBOL(get_bus_number)
            LOAD_LIBUSB_SYMBOL(get_device_address)
            LOAD_LIBUSB_SYMBOL(open)
            LOAD_LIBUSB_SYMBOL(close)
            LOAD_LIBUSB_SYMBOL(claim_interface)
            LOAD_LIBUSB_SYMBOL(release_interface)
            LOAD_LIBUSB_SYMBOL(kernel_driver_active)
            LOAD_LIBUSB_SYMBOL(detach_kernel_driver)
            LOAD_LIBUSB_SYMBOL(attach_kernel_driver)
            LOAD_LIBUSB_SYMBOL(set_interface_alt_setting)
            LOAD_LIBUSB_SYMBOL(alloc_transfer)
            LOAD_LIBUSB_SYMBOL(submit_transfer)
            LOAD_LIBUSB_SYMBOL(cancel_transfer)
            LOAD_LIBUSB_SYMBOL(free_transfer)
            LOAD_LIBUSB_SYMBOL(control_transfer)
            LOAD_LIBUSB_SYMBOL(interrupt_transfer)
            LOAD_LIBUSB_SYMBOL(handle_events)
            LOAD_LIBUSB_SYMBOL(handle_events_completed)
            LOAD_LIBUSB_SYMBOL(error_name)
#undef LOAD_LIBUSB_SYMBOL

            if (!loaded) {
#ifdef SDL_LIBUSB_DYNAMIC
                SDL_UnloadObject(libusb_ctx.libhandle);
#endif
                libusb_ctx.libhandle = NULL;
                /* SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, SDL_LIBUSB_DYNAMIC " found but could not load function"); */
            } else if (LIBUSB_hid_init() < 0) {
#ifdef SDL_LIBUSB_DYNAMIC
                SDL_UnloadObject(libusb_ctx.libhandle);
#endif
                libusb_ctx.libhandle = NULL;
            } else {
                ++success;
            }
        }
    }
#endif /* HAVE_LIBUSB */

#ifdef HAVE_PLATFORM_BACKEND
    ++attempts;
#ifdef __LINUX__
    udev_ctx = SDL_UDEV_GetUdevSyms();
#endif /* __LINUX __ */
    if (udev_ctx && PLATFORM_hid_init() == 0) {
        ++success;
    }
#endif /* HAVE_PLATFORM_BACKEND */

    if (attempts > 0 && success == 0) {
        return -1;
    }

    ++SDL_hidapi_refcount;
    return 0;
}

int SDL_hid_exit(void)
{
    int result = 0;

    if (SDL_hidapi_refcount == 0) {
        return 0;
    }
    --SDL_hidapi_refcount;
    if (SDL_hidapi_refcount > 0) {
        return 0;
    }
    SDL_hidapi_refcount = 0;

#ifndef SDL_HIDAPI_DISABLED
    HIDAPI_ShutdownDiscovery();
#endif

#ifdef HAVE_PLATFORM_BACKEND
    if (udev_ctx) {
        result |= PLATFORM_hid_exit();
    }
#ifdef __LINUX__
    SDL_UDEV_ReleaseUdevSyms();
#endif /* __LINUX __ */
#endif /* HAVE_PLATFORM_BACKEND */

#ifdef HAVE_LIBUSB
    if (libusb_ctx.libhandle) {
        result |= LIBUSB_hid_exit();
#ifdef SDL_LIBUSB_DYNAMIC
        SDL_UnloadObject(libusb_ctx.libhandle);
#endif
        libusb_ctx.libhandle = NULL;
    }
#endif /* HAVE_LIBUSB */

    return result;
}

Uint32 SDL_hid_device_change_count(void)
{
    Uint32 counter = 0;

#ifndef SDL_HIDAPI_DISABLED
    if (SDL_hidapi_refcount == 0 && SDL_hid_init() != 0) {
        return 0;
    }

    HIDAPI_UpdateDiscovery();

    if (SDL_HIDAPI_discovery.m_unDeviceChangeCounter == 0) {
        /* Counter wrapped! */
        ++SDL_HIDAPI_discovery.m_unDeviceChangeCounter;
    }
    counter = SDL_HIDAPI_discovery.m_unDeviceChangeCounter;

#endif /* !SDL_HIDAPI_DISABLED */

    return counter;
}

struct SDL_hid_device_info *SDL_hid_enumerate(unsigned short vendor_id, unsigned short product_id)
{
#if defined(HAVE_PLATFORM_BACKEND) || HAVE_DRIVER_BACKEND || defined(HAVE_LIBUSB)
#ifdef HAVE_LIBUSB
    struct SDL_hid_device_info *usb_devs = NULL;
    struct SDL_hid_device_info *usb_dev;
#endif
#if HAVE_DRIVER_BACKEND
    struct SDL_hid_device_info *driver_devs = NULL;
    struct SDL_hid_device_info *driver_dev;
#endif
#ifdef HAVE_PLATFORM_BACKEND
    struct SDL_hid_device_info *raw_devs = NULL;
    struct SDL_hid_device_info *raw_dev;
#endif
    struct SDL_hid_device_info *devs = NULL, *last = NULL, *new_dev;

    if (SDL_hidapi_refcount == 0 && SDL_hid_init() != 0) {
        return NULL;
    }

#ifdef HAVE_LIBUSB
    if (libusb_ctx.libhandle) {
        usb_devs = LIBUSB_hid_enumerate(vendor_id, product_id);
#ifdef DEBUG_HIDAPI
        SDL_Log("libusb devices found:");
#endif
        for (usb_dev = usb_devs; usb_dev; usb_dev = usb_dev->next) {
            if (use_libusb_whitelist) {
                if (!IsInWhitelist(usb_dev->vendor_id, usb_dev->product_id)) {
#ifdef DEBUG_HIDAPI
                    SDL_Log("Device was not in libusb whitelist: %ls %ls 0x%.4hx 0x%.4hx",
                            usb_dev->manufacturer_string, usb_dev->product_string,
                            usb_dev->vendor_id, usb_dev->product_id);
#endif /* DEBUG_HIDAPI */
                    continue;
                }
            }
            new_dev = (struct SDL_hid_device_info *)SDL_malloc(sizeof(struct SDL_hid_device_info));
            if (new_dev == NULL) {
                LIBUSB_hid_free_enumeration(usb_devs);
                SDL_hid_free_enumeration(devs);
                SDL_OutOfMemory();
                return NULL;
            }
            CopyHIDDeviceInfo(usb_dev, new_dev);
#ifdef DEBUG_HIDAPI
            SDL_Log(" - %ls %ls 0x%.4hx 0x%.4hx",
                    usb_dev->manufacturer_string, usb_dev->product_string,
                    usb_dev->vendor_id, usb_dev->product_id);
#endif

            if (last != NULL) {
                last->next = new_dev;
            } else {
                devs = new_dev;
            }
            last = new_dev;
        }
    }
#endif /* HAVE_LIBUSB */

#if HAVE_DRIVER_BACKEND
    driver_devs = DRIVER_hid_enumerate(vendor_id, product_id);
    for (driver_dev = driver_devs; driver_dev; driver_dev = driver_dev->next) {
        new_dev = (struct SDL_hid_device_info *)SDL_malloc(sizeof(struct SDL_hid_device_info));
        CopyHIDDeviceInfo(driver_dev, new_dev);

        if (last != NULL) {
            last->next = new_dev;
        } else {
            devs = new_dev;
        }
        last = new_dev;
    }
#endif /* HAVE_DRIVER_BACKEND */

#ifdef HAVE_PLATFORM_BACKEND
    if (udev_ctx) {
        raw_devs = PLATFORM_hid_enumerate(vendor_id, product_id);
#ifdef DEBUG_HIDAPI
        SDL_Log("hidraw devices found:");
#endif
        for (raw_dev = raw_devs; raw_dev; raw_dev = raw_dev->next) {
            SDL_bool bFound = SDL_FALSE;
#ifdef DEBUG_HIDAPI
            SDL_Log(" - %ls %ls 0x%.4hx 0x%.4hx",
                    raw_dev->manufacturer_string, raw_dev->product_string,
                    raw_dev->vendor_id, raw_dev->product_id);
#endif
#ifdef HAVE_LIBUSB
            for (usb_dev = usb_devs; usb_dev; usb_dev = usb_dev->next) {
                if (raw_dev->vendor_id == usb_dev->vendor_id &&
                    raw_dev->product_id == usb_dev->product_id &&
                    (raw_dev->interface_number < 0 || raw_dev->interface_number == usb_dev->interface_number)) {
                    bFound = SDL_TRUE;
                    break;
                }
            }
#endif
#if HAVE_DRIVER_BACKEND
            for (driver_dev = driver_devs; driver_dev; driver_dev = driver_dev->next) {
                if (raw_dev->vendor_id == driver_dev->vendor_id &&
                    raw_dev->product_id == driver_dev->product_id &&
                    (raw_dev->interface_number < 0 || raw_dev->interface_number == driver_dev->interface_number)) {
                    bFound = SDL_TRUE;
                    break;
                }
            }
#endif
            if (!bFound) {
                new_dev = (struct SDL_hid_device_info *)SDL_malloc(sizeof(struct SDL_hid_device_info));
                if (new_dev == NULL) {
#ifdef HAVE_LIBUSB
                    if (libusb_ctx.libhandle) {
                        LIBUSB_hid_free_enumeration(usb_devs);
                    }
#endif
                    PLATFORM_hid_free_enumeration(raw_devs);
                    SDL_hid_free_enumeration(devs);
                    SDL_OutOfMemory();
                    return NULL;
                }
                CopyHIDDeviceInfo(raw_dev, new_dev);
                new_dev->next = NULL;

                if (last != NULL) {
                    last->next = new_dev;
                } else {
                    devs = new_dev;
                }
                last = new_dev;
            }
        }
        PLATFORM_hid_free_enumeration(raw_devs);
    }
#endif /* HAVE_PLATFORM_BACKEND */

#ifdef HAVE_LIBUSB
    if (libusb_ctx.libhandle) {
        LIBUSB_hid_free_enumeration(usb_devs);
    }
#endif
    return devs;

#else
    return NULL;
#endif /* HAVE_PLATFORM_BACKEND || HAVE_DRIVER_BACKEND || HAVE_LIBUSB */
}

void SDL_hid_free_enumeration(struct SDL_hid_device_info *devs)
{
    while (devs) {
        struct SDL_hid_device_info *next = devs->next;
        SDL_free(devs->path);
        SDL_free(devs->serial_number);
        SDL_free(devs->manufacturer_string);
        SDL_free(devs->product_string);
        SDL_free(devs);
        devs = next;
    }
}

SDL_hid_device *SDL_hid_open(unsigned short vendor_id, unsigned short product_id, const wchar_t *serial_number)
{
#if defined(HAVE_PLATFORM_BACKEND) || HAVE_DRIVER_BACKEND || defined(HAVE_LIBUSB)
    void *pDevice = NULL;

    if (SDL_hidapi_refcount == 0 && SDL_hid_init() != 0) {
        return NULL;
    }

#ifdef HAVE_PLATFORM_BACKEND
    if (udev_ctx) {
        pDevice = PLATFORM_hid_open(vendor_id, product_id, serial_number);
        if (pDevice != NULL) {
            return CreateHIDDeviceWrapper(pDevice, &PLATFORM_Backend);
        }
    }
#endif /* HAVE_PLATFORM_BACKEND */

#if HAVE_DRIVER_BACKEND
    pDevice = DRIVER_hid_open(vendor_id, product_id, serial_number);
    if (pDevice != NULL) {
        return CreateHIDDeviceWrapper(pDevice, &DRIVER_Backend);
    }
#endif /* HAVE_DRIVER_BACKEND */

#ifdef HAVE_LIBUSB
    if (libusb_ctx.libhandle != NULL) {
        pDevice = LIBUSB_hid_open(vendor_id, product_id, serial_number);
        if (pDevice != NULL) {
            return CreateHIDDeviceWrapper(pDevice, &LIBUSB_Backend);
        }
    }
#endif /* HAVE_LIBUSB */

#endif /* HAVE_PLATFORM_BACKEND || HAVE_DRIVER_BACKEND || HAVE_LIBUSB */

    return NULL;
}

SDL_hid_device *SDL_hid_open_path(const char *path, int bExclusive /* = false */)
{
#if defined(HAVE_PLATFORM_BACKEND) || HAVE_DRIVER_BACKEND || defined(HAVE_LIBUSB)
    void *pDevice = NULL;

    if (SDL_hidapi_refcount == 0 && SDL_hid_init() != 0) {
        return NULL;
    }

#ifdef HAVE_PLATFORM_BACKEND
    if (udev_ctx) {
        pDevice = PLATFORM_hid_open_path(path, bExclusive);
        if (pDevice != NULL) {
            return CreateHIDDeviceWrapper(pDevice, &PLATFORM_Backend);
        }
    }
#endif /* HAVE_PLATFORM_BACKEND */

#if HAVE_DRIVER_BACKEND
    pDevice = DRIVER_hid_open_path(path, bExclusive);
    if (pDevice != NULL) {
        return CreateHIDDeviceWrapper(pDevice, &DRIVER_Backend);
    }
#endif /* HAVE_DRIVER_BACKEND */

#ifdef HAVE_LIBUSB
    if (libusb_ctx.libhandle != NULL) {
        pDevice = LIBUSB_hid_open_path(path, bExclusive);
        if (pDevice != NULL) {
            return CreateHIDDeviceWrapper(pDevice, &LIBUSB_Backend);
        }
    }
#endif /* HAVE_LIBUSB */

#endif /* HAVE_PLATFORM_BACKEND || HAVE_DRIVER_BACKEND || HAVE_LIBUSB */

    return NULL;
}

int SDL_hid_write(SDL_hid_device *device, const unsigned char *data, size_t length)
{
    int result;

    CHECK_DEVICE_MAGIC(device, -1);

    result = device->backend->hid_write(device->device, data, length);
    if (result < 0) {
        SDL_SetHIDAPIError(device->backend->hid_error(device->device));
    }
    return result;
}

int SDL_hid_read_timeout(SDL_hid_device *device, unsigned char *data, size_t length, int milliseconds)
{
    int result;

    CHECK_DEVICE_MAGIC(device, -1);

    result = device->backend->hid_read_timeout(device->device, data, length, milliseconds);
    if (result < 0) {
        SDL_SetHIDAPIError(device->backend->hid_error(device->device));
    }
    return result;
}

int SDL_hid_read(SDL_hid_device *device, unsigned char *data, size_t length)
{
    int result;

    CHECK_DEVICE_MAGIC(device, -1);

    result = device->backend->hid_read(device->device, data, length);
    if (result < 0) {
        SDL_SetHIDAPIError(device->backend->hid_error(device->device));
    }
    return result;
}

int SDL_hid_set_nonblocking(SDL_hid_device *device, int nonblock)
{
    int result;

    CHECK_DEVICE_MAGIC(device, -1);

    result = device->backend->hid_set_nonblocking(device->device, nonblock);
    if (result < 0) {
        SDL_SetHIDAPIError(device->backend->hid_error(device->device));
    }
    return result;
}

int SDL_hid_send_feature_report(SDL_hid_device *device, const unsigned char *data, size_t length)
{
    int result;

    CHECK_DEVICE_MAGIC(device, -1);

    result = device->backend->hid_send_feature_report(device->device, data, length);
    if (result < 0) {
        SDL_SetHIDAPIError(device->backend->hid_error(device->device));
    }
    return result;
}

int SDL_hid_get_feature_report(SDL_hid_device *device, unsigned char *data, size_t length)
{
    int result;

    CHECK_DEVICE_MAGIC(device, -1);

    result = device->backend->hid_get_feature_report(device->device, data, length);
    if (result < 0) {
        SDL_SetHIDAPIError(device->backend->hid_error(device->device));
    }
    return result;
}

void SDL_hid_close(SDL_hid_device *device)
{
    CHECK_DEVICE_MAGIC(device, );

    device->backend->hid_close(device->device);
    DeleteHIDDeviceWrapper(device);
}

int SDL_hid_get_manufacturer_string(SDL_hid_device *device, wchar_t *string, size_t maxlen)
{
    int result;

    CHECK_DEVICE_MAGIC(device, -1);

    result = device->backend->hid_get_manufacturer_string(device->device, string, maxlen);
    if (result < 0) {
        SDL_SetHIDAPIError(device->backend->hid_error(device->device));
    }
    return result;
}

int SDL_hid_get_product_string(SDL_hid_device *device, wchar_t *string, size_t maxlen)
{
    int result;

    CHECK_DEVICE_MAGIC(device, -1);

    result = device->backend->hid_get_product_string(device->device, string, maxlen);
    if (result < 0) {
        SDL_SetHIDAPIError(device->backend->hid_error(device->device));
    }
    return result;
}

int SDL_hid_get_serial_number_string(SDL_hid_device *device, wchar_t *string, size_t maxlen)
{
    int result;

    CHECK_DEVICE_MAGIC(device, -1);

    result = device->backend->hid_get_serial_number_string(device->device, string, maxlen);
    if (result < 0) {
        SDL_SetHIDAPIError(device->backend->hid_error(device->device));
    }
    return result;
}

int SDL_hid_get_indexed_string(SDL_hid_device *device, int string_index, wchar_t *string, size_t maxlen)
{
    int result;

    CHECK_DEVICE_MAGIC(device, -1);

    result = device->backend->hid_get_indexed_string(device->device, string_index, string, maxlen);
    if (result < 0) {
        SDL_SetHIDAPIError(device->backend->hid_error(device->device));
    }
    return result;
}

void SDL_hid_ble_scan(SDL_bool active)
{
#if !defined(SDL_HIDAPI_DISABLED) && (defined(__IPHONEOS__) || defined(__TVOS__))
    hid_ble_scan(active);
#endif
}

#ifdef HAVE_ENABLE_GAMECUBE_ADAPTORS
/* This is needed to enable input for Nyko and EVORETRO GameCube adaptors */
void SDL_EnableGameCubeAdaptors(void)
{
#ifdef HAVE_LIBUSB
    libusb_context *context = NULL;
    libusb_device **devs = NULL;
    libusb_device_handle *handle = NULL;
    struct libusb_device_descriptor desc;
    ssize_t i, num_devs;
    int kernel_detached = 0;

    if (libusb_ctx.libhandle == NULL) {
        return;
    }

    if (libusb_ctx.init(&context) == 0) {
        num_devs = libusb_ctx.get_device_list(context, &devs);
        for (i = 0; i < num_devs; ++i) {
            if (libusb_ctx.get_device_descriptor(devs[i], &desc) != 0) {
                continue;
            }

            if (desc.idVendor != 0x057e || desc.idProduct != 0x0337) {
                continue;
            }

            if (libusb_ctx.open(devs[i], &handle) != 0) {
                continue;
            }

            if (libusb_ctx.kernel_driver_active(handle, 0)) {
                if (libusb_ctx.detach_kernel_driver(handle, 0) == 0) {
                    kernel_detached = 1;
                }
            }

            if (libusb_ctx.claim_interface(handle, 0) == 0) {
                libusb_ctx.control_transfer(handle, 0x21, 11, 0x0001, 0, NULL, 0, 1000);
                libusb_ctx.release_interface(handle, 0);
            }

            if (kernel_detached) {
                libusb_ctx.attach_kernel_driver(handle, 0);
            }

            libusb_ctx.close(handle);
        }

        libusb_ctx.free_device_list(devs, 1);

        libusb_ctx.exit(context);
    }
#endif /* HAVE_LIBUSB */
}
#endif /* HAVE_ENABLE_GAMECUBE_ADAPTORS */

/* vi: set sts=4 ts=4 sw=4 expandtab: */
