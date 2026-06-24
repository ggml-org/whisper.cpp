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

/* Contributed by Thomas Perl <thomas.perl@jollamobile.com> */

#include "../../SDL_internal.h"

#ifdef SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH

#include "SDL_mouse.h"
#include "SDL_keyboard.h"
#include "SDL_waylandtouch.h"
#include "../../events/SDL_touch_c.h"

struct SDL_WaylandTouch
{
    struct qt_touch_extension *touch_extension;
};

/**
 * Qt TouchPointState
 * adapted from qtbase/src/corelib/global/qnamespace.h
 **/
enum QtWaylandTouchPointState
{
    QtWaylandTouchPointPressed = 0x01,
    QtWaylandTouchPointMoved = 0x02,
    /*
    Never sent by the server:
    QtWaylandTouchPointStationary = 0x04,
    */
    QtWaylandTouchPointReleased = 0x08,
};

static void touch_handle_touch(void *data,
                               struct qt_touch_extension *qt_touch_extension,
                               uint32_t time,
                               uint32_t id,
                               uint32_t state,
                               int32_t x,
                               int32_t y,
                               int32_t normalized_x,
                               int32_t normalized_y,
                               int32_t width,
                               int32_t height,
                               uint32_t pressure,
                               int32_t velocity_x,
                               int32_t velocity_y,
                               uint32_t flags,
                               struct wl_array *rawdata)
{
    /**
     * Event is assembled in QtWayland in TouchExtensionGlobal::postTouchEvent
     * (src/compositor/wayland_wrapper/qwltouch.cpp)
     **/

    float FIXED_TO_FLOAT = 1. / 10000.;
    float xf = FIXED_TO_FLOAT * normalized_x;
    float yf = FIXED_TO_FLOAT * normalized_y;

    float PRESSURE_TO_FLOAT = 1. / 255.;
    float pressuref = PRESSURE_TO_FLOAT * pressure;

    uint32_t touchState = state & 0xFFFF;
    /*
    Other fields that are sent by the server (qwltouch.cpp),
    but not used at the moment can be decoded in this way:

    uint32_t sentPointCount = state >> 16;
    uint32_t touchFlags = flags & 0xFFFF;
    uint32_t capabilities = flags >> 16;
    */

    SDL_Window *window = NULL;

    SDL_TouchID deviceId = 1;
    if (SDL_AddTouch(deviceId, SDL_TOUCH_DEVICE_DIRECT, "qt_touch_extension") < 0) {
        SDL_Log("error: can't add touch %s, %d", __FILE__, __LINE__);
    }

    /* FIXME: This should be the window the given wayland surface is associated
     * with, but how do we get the wayland surface? */
    window = SDL_GetMouseFocus();
    if (!window) {
        window = SDL_GetKeyboardFocus();
    }

    switch (touchState) {
    case QtWaylandTouchPointPressed:
    case QtWaylandTouchPointReleased:
        SDL_SendTouch(deviceId, (SDL_FingerID)id, window,
                      (touchState == QtWaylandTouchPointPressed) ? SDL_TRUE : SDL_FALSE,
                      xf, yf, pressuref);
        break;
    case QtWaylandTouchPointMoved:
        SDL_SendTouchMotion(deviceId, (SDL_FingerID)id, window, xf, yf, pressuref);
        break;
    default:
        /* Should not happen */
        break;
    }
}

static void touch_handle_configure(void *data,
                                   struct qt_touch_extension *qt_touch_extension,
                                   uint32_t flags)
{
}

/* wayland-qt-touch-extension.c BEGINS */

static const struct qt_touch_extension_listener touch_listener = {
    touch_handle_touch,
    touch_handle_configure,
};

static const struct wl_interface *qt_touch_extension_types[] = {
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
};

static const struct wl_message qt_touch_extension_requests[] = {
    { "dummy", "", qt_touch_extension_types + 0 },
};

static const struct wl_message qt_touch_extension_events[] = {
    { "touch", "uuuiiiiiiuiiua", qt_touch_extension_types + 0 },
    { "configure", "u", qt_touch_extension_types + 0 },
};

const struct wl_interface qt_touch_extension_interface = {
    "qt_touch_extension",
    1,
    1,
    qt_touch_extension_requests,
    2,
    qt_touch_extension_events,
};

/* wayland-qt-touch-extension.c ENDS */

/* wayland-qt-windowmanager.c BEGINS */
static const struct wl_interface *qt_windowmanager_types[] = {
    NULL,
    NULL,
};

static const struct wl_message qt_windowmanager_requests[] = {
    { "open_url", "us", qt_windowmanager_types + 0 },
};

static const struct wl_message qt_windowmanager_events[] = {
    { "hints", "i", qt_windowmanager_types + 0 },
    { "quit", "", qt_windowmanager_types + 0 },
};

const struct wl_interface qt_windowmanager_interface = {
    "qt_windowmanager",
    1,
    1,
    qt_windowmanager_requests,
    2,
    qt_windowmanager_events,
};
/* wayland-qt-windowmanager.c ENDS */

/* wayland-qt-surface-extension.c BEGINS */
#ifndef SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC
extern const struct wl_interface wl_surface_interface;
#endif

static const struct wl_interface *qt_surface_extension_types[] = {
    NULL,
    NULL,
    &qt_extended_surface_interface,
#ifdef SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC
    /* FIXME: Set this dynamically to (*WAYLAND_wl_surface_interface) ?
     * The value comes from auto generated code and does
     * not appear to actually be used anywhere
     */
    NULL,
#else
    &wl_surface_interface,
#endif
};

static const struct wl_message qt_surface_extension_requests[] = {
    { "get_extended_surface", "no", qt_surface_extension_types + 2 },
};

const struct wl_interface qt_surface_extension_interface = {
    "qt_surface_extension",
    1,
    1,
    qt_surface_extension_requests,
    0,
    NULL,
};

static const struct wl_message qt_extended_surface_requests[] = {
    { "update_generic_property", "sa", qt_surface_extension_types + 0 },
    { "set_content_orientation", "i", qt_surface_extension_types + 0 },
    { "set_window_flags", "i", qt_surface_extension_types + 0 },
};

static const struct wl_message qt_extended_surface_events[] = {
    { "onscreen_visibility", "i", qt_surface_extension_types + 0 },
    { "set_generic_property", "sa", qt_surface_extension_types + 0 },
    { "close", "", qt_surface_extension_types + 0 },
};

const struct wl_interface qt_extended_surface_interface = {
    "qt_extended_surface",
    1,
    3,
    qt_extended_surface_requests,
    3,
    qt_extended_surface_events,
};

/* wayland-qt-surface-extension.c ENDS */

void Wayland_touch_create(SDL_VideoData *data, uint32_t id)
{
    struct SDL_WaylandTouch *touch;

    if (data->touch) {
        Wayland_touch_destroy(data);
    }

    /* !!! FIXME: check for failure, call SDL_OutOfMemory() */
    data->touch = SDL_malloc(sizeof(struct SDL_WaylandTouch));

    touch = data->touch;
    touch->touch_extension = wl_registry_bind(data->registry, id, &qt_touch_extension_interface, 1);
    qt_touch_extension_add_listener(touch->touch_extension, &touch_listener, data);
}

void Wayland_touch_destroy(SDL_VideoData *data)
{
    if (data->touch) {
        struct SDL_WaylandTouch *touch = data->touch;
        if (touch->touch_extension) {
            qt_touch_extension_destroy(touch->touch_extension);
        }

        SDL_free(data->touch);
        data->touch = NULL;
    }
}

#endif /* SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH */
