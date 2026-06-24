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

#ifndef SDL_waylandwindow_h_
#define SDL_waylandwindow_h_

#include "../SDL_sysvideo.h"
#include "SDL_syswm.h"
#include "../../events/SDL_touch_c.h"

#include "SDL_waylandvideo.h"

struct SDL_WaylandInput;

typedef struct
{
    SDL_Window *sdlwindow;
    SDL_VideoData *waylandData;
    struct wl_surface *surface;
    struct wl_callback *gles_swap_frame_callback;
    struct wl_event_queue *gles_swap_frame_event_queue;
    struct wl_surface *gles_swap_frame_surface_wrapper;
    struct wl_callback *surface_damage_frame_callback;

    union
    {
#ifdef HAVE_LIBDECOR_H
        struct
        {
            struct libdecor_frame *frame;
            SDL_bool initial_configure_seen;
        } libdecor;
#endif
        struct
        {
            struct xdg_surface *surface;
            union
            {
                struct xdg_toplevel *toplevel;
                struct
                {
                    struct xdg_popup *popup;
                    struct xdg_positioner *positioner;
                    Uint32 parentID;
                    SDL_Window *child;
                } popup;
            } roleobj;
            SDL_bool initial_configure_seen;
        } xdg;
    } shell_surface;
    enum
    {
        WAYLAND_SURFACE_UNKNOWN = 0,
        WAYLAND_SURFACE_XDG_TOPLEVEL,
        WAYLAND_SURFACE_XDG_POPUP,
        WAYLAND_SURFACE_LIBDECOR
    } shell_surface_type;

    struct wl_egl_window *egl_window;
    struct SDL_WaylandInput *keyboard_device;
#ifdef SDL_VIDEO_OPENGL_EGL
    EGLSurface egl_surface;
#endif
    struct zwp_locked_pointer_v1 *locked_pointer;
    struct zwp_confined_pointer_v1 *confined_pointer;
    struct zxdg_toplevel_decoration_v1 *server_decoration;
    struct zwp_keyboard_shortcuts_inhibitor_v1 *key_inhibitor;
    struct zwp_idle_inhibitor_v1 *idle_inhibitor;
    struct xdg_activation_token_v1 *activation_token;
    struct wp_viewport *draw_viewport;
    struct wp_fractional_scale_v1 *fractional_scale;

    /* floating dimensions for restoring from maximized and fullscreen */
    int floating_width, floating_height;

    SDL_atomic_t swap_interval_ready;

#ifdef SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH
    struct qt_extended_surface *extended_surface;
#endif /* SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH */

    SDL_WaylandOutputData **outputs;
    int num_outputs;

    float scale_factor;
    float pointer_scale_x;
    float pointer_scale_y;
    int drawable_width, drawable_height;
    int fs_output_width, fs_output_height;
    int window_width, window_height;
    int system_min_required_width;
    int system_min_required_height;
    SDL_bool needs_resize_event;
    SDL_bool floating_resize_pending;
    SDL_bool was_floating;
    SDL_bool is_fullscreen;
    SDL_bool in_fullscreen_transition;
    Uint32 fullscreen_flags;
    SDL_bool double_buffer; 
} SDL_WindowData;

extern void Wayland_ShowWindow(_THIS, SDL_Window *window);
extern void Wayland_HideWindow(_THIS, SDL_Window *window);
extern void Wayland_RaiseWindow(_THIS, SDL_Window *window);
extern void Wayland_SetWindowFullscreen(_THIS, SDL_Window *window,
                                        SDL_VideoDisplay *_display,
                                        SDL_bool fullscreen);
extern void Wayland_MaximizeWindow(_THIS, SDL_Window *window);
extern void Wayland_MinimizeWindow(_THIS, SDL_Window *window);
extern void Wayland_SetWindowMouseRect(_THIS, SDL_Window *window);
extern void Wayland_SetWindowMouseGrab(_THIS, SDL_Window *window, SDL_bool grabbed);
extern void Wayland_SetWindowKeyboardGrab(_THIS, SDL_Window *window, SDL_bool grabbed);
extern void Wayland_RestoreWindow(_THIS, SDL_Window *window);
extern void Wayland_SetWindowBordered(_THIS, SDL_Window *window, SDL_bool bordered);
extern void Wayland_SetWindowResizable(_THIS, SDL_Window *window, SDL_bool resizable);
extern int Wayland_CreateWindow(_THIS, SDL_Window *window);
extern void Wayland_SetWindowSize(_THIS, SDL_Window *window);
extern void Wayland_SetWindowMinimumSize(_THIS, SDL_Window *window);
extern void Wayland_SetWindowMaximumSize(_THIS, SDL_Window *window);
extern void Wayland_GetWindowSizeInPixels(_THIS, SDL_Window *window, int *w, int *h);
extern int Wayland_SetWindowModalFor(_THIS, SDL_Window *modal_window, SDL_Window *parent_window);
extern void Wayland_SetWindowTitle(_THIS, SDL_Window *window);
extern void Wayland_DestroyWindow(_THIS, SDL_Window *window);
extern void Wayland_SuspendScreenSaver(_THIS);

extern SDL_bool
Wayland_GetWindowWMInfo(_THIS, SDL_Window * window, SDL_SysWMinfo * info);
extern int Wayland_SetWindowHitTest(SDL_Window *window, SDL_bool enabled);
extern int Wayland_FlashWindow(_THIS, SDL_Window *window, SDL_FlashOperation operation);

extern void Wayland_InitWin(SDL_VideoData *data);
extern void Wayland_QuitWin(SDL_VideoData *data);

#endif /* SDL_waylandwindow_h_ */

/* vi: set ts=4 sw=4 expandtab: */
