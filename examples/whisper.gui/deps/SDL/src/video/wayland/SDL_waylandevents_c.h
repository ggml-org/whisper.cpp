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

#ifndef SDL_waylandevents_h_
#define SDL_waylandevents_h_

#include "SDL_waylandvideo.h"
#include "SDL_waylandwindow.h"
#include "SDL_waylanddatamanager.h"
#include "SDL_waylandkeyboard.h"

enum SDL_WaylandAxisEvent
{
    AXIS_EVENT_CONTINUOUS = 0,
    AXIS_EVENT_DISCRETE,
    AXIS_EVENT_VALUE120
};

struct SDL_WaylandTabletSeat;

struct SDL_WaylandTabletObjectListNode
{
    void *object;
    struct SDL_WaylandTabletObjectListNode *next;
};

struct SDL_WaylandTabletInput
{
    struct SDL_WaylandTabletSeat *seat;

    struct SDL_WaylandTabletObjectListNode *tablets;
    struct SDL_WaylandTabletObjectListNode *tools;
    struct SDL_WaylandTabletObjectListNode *pads;

    SDL_WindowData *tool_focus;
    uint32_t tool_prox_serial;

    /* Last motion location */
    wl_fixed_t sx_w;
    wl_fixed_t sy_w;

    SDL_bool is_down;

    SDL_bool btn_stylus;
    SDL_bool btn_stylus2;
    SDL_bool btn_stylus3;
};

typedef struct
{
    // repeat_rate in range of [1, 1000]
    int32_t repeat_rate;
    int32_t repeat_delay;
    SDL_bool is_initialized;

    SDL_bool is_key_down;
    uint32_t key;
    uint32_t wl_press_time;  // Key press time as reported by the Wayland API
    uint32_t sdl_press_time; // Key press time expressed in SDL ticks
    uint32_t next_repeat_ms;
    uint32_t scancode;
    char text[8];
} SDL_WaylandKeyboardRepeat;

struct SDL_WaylandInput
{
    SDL_VideoData *display;
    struct wl_seat *seat;
    struct wl_pointer *pointer;
    struct wl_touch *touch;
    struct wl_keyboard *keyboard;
    SDL_WaylandDataDevice *data_device;
    SDL_WaylandPrimarySelectionDevice *primary_selection_device;
    SDL_WaylandTextInput *text_input;
    struct zwp_relative_pointer_v1 *relative_pointer;
    SDL_WindowData *pointer_focus;
    SDL_WindowData *keyboard_focus;
    uint32_t pointer_enter_serial;

    /* Last motion location */
    wl_fixed_t sx_w;
    wl_fixed_t sy_w;

    uint32_t buttons_pressed;

    double dx_frac;
    double dy_frac;

    struct
    {
        struct xkb_keymap *keymap;
        struct xkb_state *state;
        struct xkb_compose_table *compose_table;
        struct xkb_compose_state *compose_state;

        /* Keyboard layout "group" */
        uint32_t current_group;

        /* Modifier bitshift values */
        uint32_t idx_shift;
        uint32_t idx_ctrl;
        uint32_t idx_alt;
        uint32_t idx_gui;
        uint32_t idx_num;
        uint32_t idx_caps;
    } xkb;

    /* information about axis events on current frame */
    struct
    {
        enum SDL_WaylandAxisEvent x_axis_type;
        float x;

        enum SDL_WaylandAxisEvent y_axis_type;
        float y;
    } pointer_curr_axis_info;

    SDL_WaylandKeyboardRepeat keyboard_repeat;

    struct SDL_WaylandTabletInput *tablet;

    /* are we forcing relative mouse mode? */
    SDL_bool cursor_visible;
    SDL_bool relative_mode_override;
    SDL_bool warp_emulation_prohibited;
    SDL_bool keyboard_is_virtual;
};

extern void Wayland_PumpEvents(_THIS);
extern void Wayland_SendWakeupEvent(_THIS, SDL_Window *window);
extern int Wayland_WaitEventTimeout(_THIS, int timeout);

extern void Wayland_add_data_device_manager(SDL_VideoData *d, uint32_t id, uint32_t version);
extern void Wayland_add_primary_selection_device_manager(SDL_VideoData *d, uint32_t id, uint32_t version);
extern void Wayland_add_text_input_manager(SDL_VideoData *d, uint32_t id, uint32_t version);

extern void Wayland_display_add_input(SDL_VideoData *d, uint32_t id, uint32_t version);
extern void Wayland_display_destroy_input(SDL_VideoData *d);

extern void Wayland_display_add_pointer_constraints(SDL_VideoData *d, uint32_t id);
extern void Wayland_display_destroy_pointer_constraints(SDL_VideoData *d);

extern int Wayland_input_lock_pointer(struct SDL_WaylandInput *input);
extern int Wayland_input_unlock_pointer(struct SDL_WaylandInput *input);

extern int Wayland_input_confine_pointer(struct SDL_WaylandInput *input, SDL_Window *window);
extern int Wayland_input_unconfine_pointer(struct SDL_WaylandInput *input, SDL_Window *window);

extern void Wayland_display_add_relative_pointer_manager(SDL_VideoData *d, uint32_t id);
extern void Wayland_display_destroy_relative_pointer_manager(SDL_VideoData *d);

extern int Wayland_input_grab_keyboard(SDL_Window *window, struct SDL_WaylandInput *input);
extern int Wayland_input_ungrab_keyboard(SDL_Window *window);

extern void Wayland_input_add_tablet(struct SDL_WaylandInput *input, struct SDL_WaylandTabletManager *tablet_manager);
extern void Wayland_input_destroy_tablet(struct SDL_WaylandInput *input);

#endif /* SDL_waylandevents_h_ */

/* vi: set ts=4 sw=4 expandtab: */
