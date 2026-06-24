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
#include "SDL_stdinc.h"

#ifndef SDL_waylandvideo_h_
#define SDL_waylandvideo_h_

#include <EGL/egl.h>
#include "wayland-util.h"

#include "../SDL_sysvideo.h"
#include "../../core/linux/SDL_dbus.h"
#include "../../core/linux/SDL_ime.h"

struct xkb_context;
struct SDL_WaylandInput;
struct SDL_WaylandTabletManager;

#ifdef SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH
struct SDL_WaylandTouch;
struct qt_surface_extension;
struct qt_windowmanager;
#endif /* SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH */

typedef struct
{
    struct wl_cursor_theme *theme;
    int size;
} SDL_WaylandCursorTheme;

typedef struct SDL_WaylandOutputData SDL_WaylandOutputData;

typedef struct
{
    SDL_bool initializing;
    struct wl_display *display;
    int display_disconnected;
    struct wl_registry *registry;
    struct wl_compositor *compositor;
    struct wl_shm *shm;
    SDL_WaylandCursorTheme *cursor_themes;
    int num_cursor_themes;
    struct wl_pointer *pointer;
    struct
    {
        struct xdg_wm_base *xdg;
#ifdef HAVE_LIBDECOR_H
        struct libdecor *libdecor;
#endif
    } shell;
    struct zwp_relative_pointer_manager_v1 *relative_pointer_manager;
    struct zwp_pointer_constraints_v1 *pointer_constraints;
    struct wl_data_device_manager *data_device_manager;
    struct zwp_primary_selection_device_manager_v1 *primary_selection_device_manager;
    struct zxdg_decoration_manager_v1 *decoration_manager;
    struct zwp_keyboard_shortcuts_inhibit_manager_v1 *key_inhibitor_manager;
    struct zwp_idle_inhibit_manager_v1 *idle_inhibit_manager;
    struct xdg_activation_v1 *activation_manager;
    struct zwp_text_input_manager_v3 *text_input_manager;
    struct zxdg_output_manager_v1 *xdg_output_manager;
    struct wp_viewporter *viewporter;
    struct wp_fractional_scale_manager_v1 *fractional_scale_manager;

    EGLDisplay edpy;
    EGLContext context;
    EGLConfig econf;

    struct xkb_context *xkb_context;
    struct SDL_WaylandInput *input;
    struct SDL_WaylandTabletManager *tablet_manager;
    SDL_WaylandOutputData *output_list;

#ifdef SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH
    struct SDL_WaylandTouch *touch;
    struct qt_surface_extension *surface_extension;
    struct qt_windowmanager *windowmanager;
#endif /* SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH */

    char *classname;

    int relative_mouse_mode;
    SDL_bool egl_transparency_enabled;
} SDL_VideoData;

struct SDL_WaylandOutputData
{
    SDL_VideoData *videodata;
    struct wl_output *output;
    struct zxdg_output_v1 *xdg_output;
    uint32_t registry_id;
    float scale_factor;
    int native_width, native_height;
    int x, y, width, height, refresh, transform;
    SDL_DisplayOrientation orientation;
    int physical_width, physical_height;
    float ddpi, hdpi, vdpi;
    SDL_bool has_logical_position, has_logical_size;
    int index;
    SDL_VideoDisplay placeholder;
    int wl_output_done_count;
    SDL_WaylandOutputData *next;
};

/* Needed here to get wl_surface declaration, fixes GitHub#4594 */
#include "SDL_waylanddyn.h"

extern void SDL_WAYLAND_register_surface(struct wl_surface *surface);
extern void SDL_WAYLAND_register_output(struct wl_output *output);
extern SDL_bool SDL_WAYLAND_own_surface(struct wl_surface *surface);
extern SDL_bool SDL_WAYLAND_own_output(struct wl_output *output);

extern SDL_bool Wayland_LoadLibdecor(SDL_VideoData *data, SDL_bool ignore_xdg);

extern SDL_bool Wayland_VideoReconnect(_THIS);

#endif /* SDL_waylandvideo_h_ */

/* vi: set ts=4 sw=4 expandtab: */
