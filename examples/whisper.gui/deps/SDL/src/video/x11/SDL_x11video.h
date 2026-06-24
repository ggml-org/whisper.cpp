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

#ifndef SDL_x11video_h_
#define SDL_x11video_h_

#include "SDL_keycode.h"

#include "../SDL_sysvideo.h"

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>

#ifdef SDL_VIDEO_DRIVER_X11_XCURSOR
#include <X11/Xcursor/Xcursor.h>
#endif
#ifdef SDL_VIDEO_DRIVER_X11_XDBE
#include <X11/extensions/Xdbe.h>
#endif
#ifdef SDL_VIDEO_DRIVER_X11_XINPUT2
#include <X11/extensions/XInput2.h>
#endif
#ifdef SDL_VIDEO_DRIVER_X11_XRANDR
#include <X11/extensions/Xrandr.h>
#endif
#ifdef SDL_VIDEO_DRIVER_X11_XSCRNSAVER
#include <X11/extensions/scrnsaver.h>
#endif
#ifdef SDL_VIDEO_DRIVER_X11_XSHAPE
#include <X11/extensions/shape.h>
#endif

#include "../../core/linux/SDL_dbus.h"
#include "../../core/linux/SDL_ime.h"

#include "SDL_x11dyn.h"

#include "SDL_x11clipboard.h"
#include "SDL_x11events.h"
#include "SDL_x11keyboard.h"
#include "SDL_x11modes.h"
#include "SDL_x11mouse.h"
#include "SDL_x11opengl.h"
#include "SDL_x11window.h"
#include "SDL_x11vulkan.h"

/* Private display data */

typedef struct SDL_VideoData
{
    Display *display;
    Display *request_display;
    char *classname;
    pid_t pid;
    XIM im;
    Uint32 screensaver_activity;
    int numwindows;
    SDL_WindowData **windowlist;
    int windowlistlength;
    XID window_group;
    Window clipboard_window;
#ifdef SDL_VIDEO_DRIVER_X11_XFIXES
    SDL_Window *active_cursor_confined_window;
#endif /* SDL_VIDEO_DRIVER_X11_XFIXES */

    /* This is true for ICCCM2.0-compliant window managers */
    SDL_bool net_wm;

    /* Useful atoms */
    Atom WM_PROTOCOLS;
    Atom WM_DELETE_WINDOW;
    Atom WM_TAKE_FOCUS;
    Atom WM_NAME;
    Atom _NET_WM_STATE;
    Atom _NET_WM_STATE_HIDDEN;
    Atom _NET_WM_STATE_FOCUSED;
    Atom _NET_WM_STATE_MAXIMIZED_VERT;
    Atom _NET_WM_STATE_MAXIMIZED_HORZ;
    Atom _NET_WM_STATE_FULLSCREEN;
    Atom _NET_WM_STATE_ABOVE;
    Atom _NET_WM_STATE_SKIP_TASKBAR;
    Atom _NET_WM_STATE_SKIP_PAGER;
    Atom _NET_WM_ALLOWED_ACTIONS;
    Atom _NET_WM_ACTION_FULLSCREEN;
    Atom _NET_WM_NAME;
    Atom _NET_WM_ICON_NAME;
    Atom _NET_WM_ICON;
    Atom _NET_WM_PING;
    Atom _NET_WM_WINDOW_OPACITY;
    Atom _NET_WM_USER_TIME;
    Atom _NET_ACTIVE_WINDOW;
    Atom _NET_FRAME_EXTENTS;
    Atom _SDL_WAKEUP;
    Atom UTF8_STRING;
    Atom PRIMARY;
    Atom XdndEnter;
    Atom XdndPosition;
    Atom XdndStatus;
    Atom XdndTypeList;
    Atom XdndActionCopy;
    Atom XdndDrop;
    Atom XdndFinished;
    Atom XdndSelection;
    Atom XKLAVIER_STATE;

    SDL_Scancode key_layout[256];
    SDL_bool selection_waiting;

    SDL_bool broken_pointer_grab; /* true if XGrabPointer seems unreliable. */

    Uint32 last_mode_change_deadline;

    SDL_bool global_mouse_changed;
    SDL_Point global_mouse_position;
    Uint32 global_mouse_buttons;

    SDL_XInput2DeviceInfo *mouse_device_info;

    int xrandr_event_base;

#ifdef SDL_VIDEO_DRIVER_X11_HAS_XKBKEYCODETOKEYSYM
    XkbDescPtr xkb;
#endif
    int xkb_event;

    KeyCode filter_code;
    Time filter_time;

#ifdef SDL_VIDEO_VULKAN
    /* Vulkan variables only valid if _this->vulkan_config.loader_handle is not NULL */
    void *vulkan_xlib_xcb_library;
    PFN_XGetXCBConnection vulkan_XGetXCBConnection;
#endif

    /* Used to interact with the on-screen keyboard */
    SDL_bool is_steam_deck;
    SDL_bool steam_keyboard_open;

    SDL_bool is_xwayland;

} SDL_VideoData;

extern SDL_bool X11_UseDirectColorVisuals(void);

#endif /* SDL_x11video_h_ */

/* vi: set ts=4 sw=4 expandtab: */
