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

#ifndef SDL_mouse_c_h_
#define SDL_mouse_c_h_

#include "SDL_mouse.h"

typedef Uint32 SDL_MouseID;

struct SDL_Cursor
{
    struct SDL_Cursor *next;
    void *driverdata;
};

typedef struct
{
    SDL_MouseID mouseID;
    Uint32 buttonstate;
} SDL_MouseInputSource;

typedef struct
{
    int last_x, last_y;
    Uint32 last_timestamp;
    Uint8 click_count;
} SDL_MouseClickState;

typedef struct
{
    /* Create a cursor from a surface */
    SDL_Cursor *(*CreateCursor)(SDL_Surface *surface, int hot_x, int hot_y);

    /* Create a system cursor */
    SDL_Cursor *(*CreateSystemCursor)(SDL_SystemCursor id);

    /* Show the specified cursor, or hide if cursor is NULL */
    int (*ShowCursor)(SDL_Cursor *cursor);

    /* This is called when a mouse motion event occurs */
    void (*MoveCursor)(SDL_Cursor *cursor);

    /* Free a window manager cursor */
    void (*FreeCursor)(SDL_Cursor *cursor);

    /* Warp the mouse to (x,y) within a window */
    void (*WarpMouse)(SDL_Window *window, int x, int y);

    /* Warp the mouse to (x,y) in screen space */
    int (*WarpMouseGlobal)(int x, int y);

    /* Set relative mode */
    int (*SetRelativeMouseMode)(SDL_bool enabled);

    /* Set mouse capture */
    int (*CaptureMouse)(SDL_Window *window);

    /* Get absolute mouse coordinates. (x) and (y) are never NULL and set to zero before call. */
    Uint32 (*GetGlobalMouseState)(int *x, int *y);

    /* Data common to all mice */
    SDL_MouseID mouseID;
    SDL_Window *focus;
    int x;
    int y;
    int xdelta;
    int ydelta;
    int last_x, last_y; /* the last reported x and y coordinates */
    float accumulated_wheel_x;
    float accumulated_wheel_y;
    SDL_bool has_position;
    SDL_bool relative_mode;
    SDL_bool relative_mode_warp;
    SDL_bool relative_mode_warp_motion;
    SDL_bool relative_mode_cursor_visible;
    SDL_bool enable_normal_speed_scale;
    float normal_speed_scale;
    SDL_bool enable_relative_speed_scale;
    float relative_speed_scale;
    SDL_bool enable_relative_system_scale;
    int num_system_scale_values;
    float *system_scale_values;
    float scale_accum_x;
    float scale_accum_y;
    Uint32 double_click_time;
    int double_click_radius;
    SDL_bool touch_mouse_events;
    SDL_bool mouse_touch_events;
    SDL_bool was_touch_mouse_events; /* Was a touch-mouse event pending? */
#if defined(__vita__)
    Uint8 vita_touch_mouse_device;
#endif
    SDL_bool auto_capture;
    SDL_bool capture_desired;
    SDL_Window *capture_window;

    /* Data for input source state */
    int num_sources;
    SDL_MouseInputSource *sources;

    /* Data for double-click tracking */
    int num_clickstates;
    SDL_MouseClickState *clickstate;

    SDL_Cursor *cursors;
    SDL_Cursor *def_cursor;
    SDL_Cursor *cur_cursor;
    SDL_bool cursor_shown;

    /* Driver-dependent data. */
    void *driverdata;
} SDL_Mouse;

/* Initialize the mouse subsystem */
extern int SDL_MousePreInit(void);
extern void SDL_MousePostInit(void);

/* Get the mouse state structure */
SDL_Mouse *SDL_GetMouse(void);

/* Set the default mouse cursor */
extern void SDL_SetDefaultCursor(SDL_Cursor *cursor);

/* Set the mouse focus window */
extern void SDL_SetMouseFocus(SDL_Window *window);

/* Update the mouse capture window */
extern int SDL_UpdateMouseCapture(SDL_bool force_release);

/* You can set either a single scale, or a set of {speed, scale} values in sorted order */
extern int SDL_SetMouseSystemScale(int num_values, const float *values);

/* Send a mouse motion event */
extern int SDL_SendMouseMotion(SDL_Window *window, SDL_MouseID mouseID, int relative, int x, int y);

/* Send a mouse button event */
extern int SDL_SendMouseButton(SDL_Window *window, SDL_MouseID mouseID, Uint8 state, Uint8 button);

/* Send a mouse button event with a click count */
extern int SDL_SendMouseButtonClicks(SDL_Window *window, SDL_MouseID mouseID, Uint8 state, Uint8 button, int clicks);

/* Send a mouse wheel event */
extern int SDL_SendMouseWheel(SDL_Window *window, SDL_MouseID mouseID, float x, float y, SDL_MouseWheelDirection direction);

/* Warp the mouse within the window, potentially overriding relative mode */
extern void SDL_PerformWarpMouseInWindow(SDL_Window *window, int x, int y, SDL_bool ignore_relative_mode);

/* TODO RECONNECT: Set mouse state to "zero" */
#if 0
extern void SDL_ResetMouse(void);
#endif /* 0 */

/* Shutdown the mouse subsystem */
extern void SDL_MouseQuit(void);

#endif /* SDL_mouse_c_h_ */

/* vi: set ts=4 sw=4 expandtab: */
