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

/* General mouse handling code for SDL */

#include "SDL_hints.h"
#include "SDL_timer.h"
#include "SDL_events.h"
#include "SDL_events_c.h"
#include "../SDL_hints_c.h"
#include "../video/SDL_sysvideo.h"
#if defined(__WIN32__) || defined(__GDK__)
#include "../core/windows/SDL_windows.h" // For GetDoubleClickTime()
#endif
#if defined(__OS2__)
#define INCL_WIN
#include <os2.h>
#endif

/* #define DEBUG_MOUSE */

/* The mouse state */
static SDL_Mouse SDL_mouse;

/* for mapping mouse events to touch */
static SDL_bool track_mouse_down = SDL_FALSE;

static int SDL_PrivateSendMouseMotion(SDL_Window *window, SDL_MouseID mouseID, int relative, int x, int y);

static void SDLCALL SDL_MouseDoubleClickTimeChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    if (hint && *hint) {
        mouse->double_click_time = SDL_atoi(hint);
    } else {
#if defined(__WIN32__) || defined(__WINGDK__)
        mouse->double_click_time = GetDoubleClickTime();
#elif defined(__OS2__)
        mouse->double_click_time = WinQuerySysValue(HWND_DESKTOP, SV_DBLCLKTIME);
#else
        mouse->double_click_time = 500;
#endif
    }
}

static void SDLCALL SDL_MouseDoubleClickRadiusChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    if (hint && *hint) {
        mouse->double_click_radius = SDL_atoi(hint);
    } else {
        mouse->double_click_radius = 32; /* 32 pixels seems about right for touch interfaces */
    }
}

static void SDLCALL SDL_MouseNormalSpeedScaleChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    if (hint && *hint) {
        mouse->enable_normal_speed_scale = SDL_TRUE;
        mouse->normal_speed_scale = (float)SDL_atof(hint);
    } else {
        mouse->enable_normal_speed_scale = SDL_FALSE;
        mouse->normal_speed_scale = 1.0f;
    }
}

static void SDLCALL SDL_MouseRelativeSpeedScaleChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    if (hint && *hint) {
        mouse->enable_relative_speed_scale = SDL_TRUE;
        mouse->relative_speed_scale = (float)SDL_atof(hint);
    } else {
        mouse->enable_relative_speed_scale = SDL_FALSE;
        mouse->relative_speed_scale = 1.0f;
    }
}

static void SDLCALL SDL_MouseRelativeSystemScaleChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    mouse->enable_relative_system_scale = SDL_GetStringBoolean(hint, SDL_FALSE);
}

static void SDLCALL SDL_TouchMouseEventsChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    mouse->touch_mouse_events = SDL_GetStringBoolean(hint, SDL_TRUE);
}

#if defined(__vita__)
static void SDLCALL SDL_VitaTouchMouseDeviceChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;
    if (hint) {
        switch (*hint) {
        default:
        case '0':
            mouse->vita_touch_mouse_device = 0;
            break;
        case '1':
            mouse->vita_touch_mouse_device = 1;
            break;
        case '2':
            mouse->vita_touch_mouse_device = 2;
            break;
        }
    }
}
#endif

static void SDLCALL SDL_MouseTouchEventsChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;
    SDL_bool default_value;

#if defined(__ANDROID__) || (defined(__IPHONEOS__) && !defined(__TVOS__))
    default_value = SDL_TRUE;
#else
    default_value = SDL_FALSE;
#endif
    mouse->mouse_touch_events = SDL_GetStringBoolean(hint, default_value);

    if (mouse->mouse_touch_events) {
        SDL_AddTouch(SDL_MOUSE_TOUCHID, SDL_TOUCH_DEVICE_DIRECT, "mouse_input");
    }
}

static void SDLCALL SDL_MouseAutoCaptureChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;
    SDL_bool auto_capture = SDL_GetStringBoolean(hint, SDL_TRUE);

    if (auto_capture != mouse->auto_capture) {
        mouse->auto_capture = auto_capture;
        SDL_UpdateMouseCapture(SDL_FALSE);
    }
}

static void SDLCALL SDL_MouseRelativeWarpMotionChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    mouse->relative_mode_warp_motion = SDL_GetStringBoolean(hint, SDL_FALSE);
}

static void SDLCALL SDL_MouseRelativeCursorVisibleChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    mouse->relative_mode_cursor_visible = SDL_GetStringBoolean(hint, SDL_FALSE);
}

/* Public functions */
int SDL_MousePreInit(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    SDL_zerop(mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_DOUBLE_CLICK_TIME,
                        SDL_MouseDoubleClickTimeChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_DOUBLE_CLICK_RADIUS,
                        SDL_MouseDoubleClickRadiusChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_NORMAL_SPEED_SCALE,
                        SDL_MouseNormalSpeedScaleChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_RELATIVE_SPEED_SCALE,
                        SDL_MouseRelativeSpeedScaleChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_RELATIVE_SYSTEM_SCALE,
                        SDL_MouseRelativeSystemScaleChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_TOUCH_MOUSE_EVENTS,
                        SDL_TouchMouseEventsChanged, mouse);

#if defined(__vita__)
    SDL_AddHintCallback(SDL_HINT_VITA_TOUCH_MOUSE_DEVICE,
                        SDL_VitaTouchMouseDeviceChanged, mouse);
#endif

    SDL_AddHintCallback(SDL_HINT_MOUSE_TOUCH_EVENTS,
                        SDL_MouseTouchEventsChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_AUTO_CAPTURE,
                        SDL_MouseAutoCaptureChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_RELATIVE_WARP_MOTION,
                        SDL_MouseRelativeWarpMotionChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_RELATIVE_CURSOR_VISIBLE,
                        SDL_MouseRelativeCursorVisibleChanged, mouse);

    mouse->was_touch_mouse_events = SDL_FALSE; /* no touch to mouse movement event pending */

    mouse->cursor_shown = SDL_TRUE;

    return 0;
}

void SDL_MousePostInit(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    /* Create a dummy mouse cursor for video backends that don't support true cursors,
     * so that mouse grab and focus functionality will work.
     */
    if (!mouse->def_cursor) {
        SDL_Surface *surface = SDL_CreateRGBSurface(0, 1, 1, 32, 0xFF, 0xFF, 0xFF, 0xFF);
        if (surface) {
            SDL_memset(surface->pixels, 0, (size_t)surface->h * surface->pitch);
            SDL_SetDefaultCursor(SDL_CreateColorCursor(surface, 0, 0));
            SDL_FreeSurface(surface);
        }
    }
}

void SDL_SetDefaultCursor(SDL_Cursor *cursor)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (cursor == mouse->def_cursor) {
        return;
    }

    if (mouse->def_cursor) {
        SDL_Cursor *default_cursor = mouse->def_cursor;
        SDL_Cursor *prev, *curr;

        if (mouse->cur_cursor == mouse->def_cursor) {
            mouse->cur_cursor = NULL;
        }
        mouse->def_cursor = NULL;

        for (prev = NULL, curr = mouse->cursors; curr;
             prev = curr, curr = curr->next) {
            if (curr == default_cursor) {
                if (prev) {
                    prev->next = curr->next;
                } else {
                    mouse->cursors = curr->next;
                }

                break;
            }
        }

        if (mouse->FreeCursor && default_cursor->driverdata) {
            mouse->FreeCursor(default_cursor);
        } else {
            SDL_free(default_cursor);
        }
    }

    mouse->def_cursor = cursor;

    if (!mouse->cur_cursor) {
        SDL_SetCursor(cursor);
    }
}

SDL_Mouse *SDL_GetMouse(void)
{
    return &SDL_mouse;
}

static Uint32 GetButtonState(SDL_Mouse *mouse, SDL_bool include_touch)
{
    int i;
    Uint32 buttonstate = 0;

    for (i = 0; i < mouse->num_sources; ++i) {
        if (include_touch || mouse->sources[i].mouseID != SDL_TOUCH_MOUSEID) {
            buttonstate |= mouse->sources[i].buttonstate;
        }
    }
    return buttonstate;
}

SDL_Window *SDL_GetMouseFocus(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    return mouse->focus;
}

/* TODO RECONNECT: Hello from the Wayland video driver!
 * This was once removed from SDL, but it's been added back in comment form
 * because we will need it when Wayland adds compositor reconnect support.
 * If you need this before we do, great! Otherwise, leave this alone, we'll
 * uncomment it at the right time.
 * -flibit
 */
#if 0
void SDL_ResetMouse(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    Uint32 buttonState = GetButtonState(mouse, SDL_FALSE);
    int i;

    for (i = 1; i <= sizeof(buttonState)*8; ++i) {
        if (buttonState & SDL_BUTTON(i)) {
            SDL_SendMouseButton(mouse->focus, mouse->mouseID, SDL_RELEASED, i);
        }
    }
    SDL_assert(GetButtonState(mouse, SDL_FALSE) == 0);
}
#endif /* 0 */

void SDL_SetMouseFocus(SDL_Window *window)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (mouse->focus == window) {
        return;
    }

    /* Actually, this ends up being a bad idea, because most operating
       systems have an implicit grab when you press the mouse button down
       so you can drag things out of the window and then get the mouse up
       when it happens.  So, #if 0...
    */
#if 0
    if (mouse->focus && !window) {
        /* We won't get anymore mouse messages, so reset mouse state */
        SDL_ResetMouse();
    }
#endif

    /* See if the current window has lost focus */
    if (mouse->focus) {
        SDL_SendWindowEvent(mouse->focus, SDL_WINDOWEVENT_LEAVE, 0, 0);
    }

    mouse->focus = window;
    mouse->has_position = SDL_FALSE;

    if (mouse->focus) {
        SDL_SendWindowEvent(mouse->focus, SDL_WINDOWEVENT_ENTER, 0, 0);
    }

    /* Update cursor visibility */
    SDL_SetCursor(NULL);
}

/* Check to see if we need to synthesize focus events */
static SDL_bool SDL_UpdateMouseFocus(SDL_Window *window, int x, int y, Uint32 buttonstate, SDL_bool send_mouse_motion)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_bool inWindow = SDL_TRUE;

    if (window && !(window->flags & SDL_WINDOW_MOUSE_CAPTURE)) {
        int w, h;
        SDL_GetWindowSize(window, &w, &h);
        if (x < 0 || y < 0 || x >= w || y >= h) {
            inWindow = SDL_FALSE;
        }
    }

    if (!inWindow) {
        if (window == mouse->focus) {
#ifdef DEBUG_MOUSE
            SDL_Log("Mouse left window, synthesizing move & focus lost event\n");
#endif
            if (send_mouse_motion) {
                SDL_PrivateSendMouseMotion(window, mouse->mouseID, 0, x, y);
            }
            SDL_SetMouseFocus(NULL);
        }
        return SDL_FALSE;
    }

    if (window != mouse->focus) {
#ifdef DEBUG_MOUSE
        SDL_Log("Mouse entered window, synthesizing focus gain & move event\n");
#endif
        SDL_SetMouseFocus(window);
        if (send_mouse_motion) {
            SDL_PrivateSendMouseMotion(window, mouse->mouseID, 0, x, y);
        }
    }
    return SDL_TRUE;
}

int SDL_SendMouseMotion(SDL_Window *window, SDL_MouseID mouseID, int relative, int x, int y)
{
    if (window && !relative) {
        SDL_Mouse *mouse = SDL_GetMouse();
        if (!SDL_UpdateMouseFocus(window, x, y, GetButtonState(mouse, SDL_TRUE), (mouseID == SDL_TOUCH_MOUSEID) ? SDL_FALSE : SDL_TRUE)) {
            return 0;
        }
    }

    return SDL_PrivateSendMouseMotion(window, mouseID, relative, x, y);
}

static int GetScaledMouseDelta(float scale, int value, float *accum)
{
    if (value && scale != 1.0f) {
        if ((value > 0) != (*accum > 0)) {
            *accum = 0.0f;
        }
        *accum += scale * value;
        if (*accum >= 0.0f) {
            value = (int)SDL_floor(*accum);
        } else {
            value = (int)SDL_ceil(*accum);
        }
        *accum -= value;
    }
    return value;
}

static float CalculateSystemScale(SDL_Mouse *mouse, const int *x, const int *y)
{
    int i;
    int n = mouse->num_system_scale_values;
    float *v = mouse->system_scale_values;
    float speed, coef, scale;

    /* If we're using a single scale value, return that */
    if (n == 1) {
        return v[0];
    }

    speed = SDL_sqrtf((float)(*x * *x) + (*y * *y));
    for (i = 0; i < (n - 2); i += 2) {
        if (speed < v[i + 2]) {
            break;
        }
    }
    if (i == (n - 2)) {
        scale = v[n - 1];
    } else if (speed <= v[i]) {
        scale = v[i + 1];
    } else {
        coef = (speed - v[i]) / (v[i + 2] - v[i]);
        scale = v[i + 1] + (coef * (v[i + 3] - v[i + 1]));
    }
    SDL_Log("speed = %.2f, scale = %.2f\n", speed, scale);
    return scale;
}

/* You can set either a single scale, or a set of {speed, scale} values in ascending order */
int SDL_SetMouseSystemScale(int num_values, const float *values)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    float *v;

    if (num_values == mouse->num_system_scale_values &&
        SDL_memcmp(values, mouse->system_scale_values, num_values * sizeof(*values)) == 0) {
        /* Nothing has changed */
        return 0;
    }

    if (num_values < 1) {
        return SDL_SetError("You must have at least one scale value");
    }

    if (num_values > 1) {
        /* Validate the values */
        int i;

        if (num_values < 4 || (num_values % 2) != 0) {
            return SDL_SetError("You must pass a set of {speed, scale} values");
        }

        for (i = 0; i < (num_values - 2); i += 2) {
            if (values[i] >= values[i + 2]) {
                return SDL_SetError("Speed values must be in ascending order");
            }
        }
    }

    v = (float *)SDL_realloc(mouse->system_scale_values, num_values * sizeof(*values));
    if (!v) {
        return SDL_OutOfMemory();
    }
    SDL_memcpy(v, values, num_values * sizeof(*values));

    mouse->num_system_scale_values = num_values;
    mouse->system_scale_values = v;
    return 0;
}

static void GetScaledMouseDeltas(SDL_Mouse *mouse, int *x, int *y)
{
    if (mouse->relative_mode) {
        if (mouse->enable_relative_speed_scale) {
            *x = GetScaledMouseDelta(mouse->relative_speed_scale, *x, &mouse->scale_accum_x);
            *y = GetScaledMouseDelta(mouse->relative_speed_scale, *y, &mouse->scale_accum_y);
        } else if (mouse->enable_relative_system_scale && mouse->num_system_scale_values > 0) {
            float relative_system_scale = CalculateSystemScale(mouse, x, y);
            *x = GetScaledMouseDelta(relative_system_scale, *x, &mouse->scale_accum_x);
            *y = GetScaledMouseDelta(relative_system_scale, *y, &mouse->scale_accum_y);
        }
    } else {
        if (mouse->enable_normal_speed_scale) {
            *x = GetScaledMouseDelta(mouse->normal_speed_scale, *x, &mouse->scale_accum_x);
            *y = GetScaledMouseDelta(mouse->normal_speed_scale, *y, &mouse->scale_accum_y);
        }
    }
}

static int SDL_PrivateSendMouseMotion(SDL_Window *window, SDL_MouseID mouseID, int relative, int x, int y)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    int posted;
    int xrel = 0;
    int yrel = 0;

    /* SDL_HINT_MOUSE_TOUCH_EVENTS: controlling whether mouse events should generate synthetic touch events */
    if (mouse->mouse_touch_events) {
        if (mouseID != SDL_TOUCH_MOUSEID && !relative && track_mouse_down) {
            if (window) {
                float fx = (float)x / (float)window->w;
                float fy = (float)y / (float)window->h;
                SDL_SendTouchMotion(SDL_MOUSE_TOUCHID, 0, window, fx, fy, 1.0f);
            }
        }
    }

    /* SDL_HINT_TOUCH_MOUSE_EVENTS: if not set, discard synthetic mouse events coming from platform layer */
    if (mouse->touch_mouse_events == 0) {
        if (mouseID == SDL_TOUCH_MOUSEID) {
            return 0;
        }
    }

    if (mouseID != SDL_TOUCH_MOUSEID && mouse->relative_mode_warp) {
        int center_x = 0, center_y = 0;
        SDL_GetWindowSize(window, &center_x, &center_y);
        center_x /= 2;
        center_y /= 2;
        if (x == center_x && y == center_y) {
            mouse->last_x = center_x;
            mouse->last_y = center_y;
            if (!mouse->relative_mode_warp_motion) {
                return 0;
            }
        } else {
            if (window && (window->flags & SDL_WINDOW_INPUT_FOCUS)) {
                if (mouse->WarpMouse) {
                    mouse->WarpMouse(window, center_x, center_y);
                } else {
                    SDL_PrivateSendMouseMotion(window, mouseID, 0, center_x, center_y);
                }
            }
        }
    }

    if (relative) {
        GetScaledMouseDeltas(mouse, &x, &y);
        xrel = x;
        yrel = y;
        x = (mouse->last_x + xrel);
        y = (mouse->last_y + yrel);
    } else if (mouse->has_position) {
        xrel = x - mouse->last_x;
        yrel = y - mouse->last_y;
    }

    /* Ignore relative motion when first positioning the mouse */
    if (!mouse->has_position) {
        mouse->x = x;
        mouse->y = y;
        mouse->has_position = SDL_TRUE;
    } else if (!xrel && !yrel) { /* Drop events that don't change state */
#ifdef DEBUG_MOUSE
        SDL_Log("Mouse event didn't change state - dropped!\n");
#endif
        return 0;
    }

    /* Ignore relative motion positioning the first touch */
    if (mouseID == SDL_TOUCH_MOUSEID && !GetButtonState(mouse, SDL_TRUE)) {
        xrel = 0;
        yrel = 0;
    }

    /* Update internal mouse coordinates */
    if (!mouse->relative_mode) {
        mouse->x = x;
        mouse->y = y;
    } else {
        mouse->x += xrel;
        mouse->y += yrel;
    }

    /* make sure that the pointers find themselves inside the windows,
       unless we have the mouse captured. */
    if (window && !(window->flags & SDL_WINDOW_MOUSE_CAPTURE)) {
        int x_min = 0, x_max = 0;
        int y_min = 0, y_max = 0;
        const SDL_Rect *confine = SDL_GetWindowMouseRect(window);

        SDL_GetWindowSize(window, &x_max, &y_max);
        --x_max;
        --y_max;

        if (confine) {
            SDL_Rect window_rect;
            SDL_Rect mouse_rect;

            window_rect.x = 0;
            window_rect.y = 0;
            window_rect.w = x_max + 1;
            window_rect.h = y_max + 1;
            if (SDL_IntersectRect(confine, &window_rect, &mouse_rect)) {
                x_min = mouse_rect.x;
                y_min = mouse_rect.y;
                x_max = x_min + mouse_rect.w - 1;
                y_max = y_min + mouse_rect.h - 1;
            }
        }

        if (mouse->x > x_max) {
            mouse->x = x_max;
        }
        if (mouse->x < x_min) {
            mouse->x = x_min;
        }

        if (mouse->y > y_max) {
            mouse->y = y_max;
        }
        if (mouse->y < y_min) {
            mouse->y = y_min;
        }
    }

    mouse->xdelta += xrel;
    mouse->ydelta += yrel;

    /* Move the mouse cursor, if needed */
    if (mouse->cursor_shown && !mouse->relative_mode &&
        mouse->MoveCursor && mouse->cur_cursor) {
        mouse->MoveCursor(mouse->cur_cursor);
    }

    /* Post the event, if desired */
    posted = 0;
    if (SDL_GetEventState(SDL_MOUSEMOTION) == SDL_ENABLE) {
        SDL_Event event;
        event.motion.type = SDL_MOUSEMOTION;
        event.motion.windowID = mouse->focus ? mouse->focus->id : 0;
        event.motion.which = mouseID;
        /* Set us pending (or clear during a normal mouse movement event) as having triggered */
        mouse->was_touch_mouse_events = (mouseID == SDL_TOUCH_MOUSEID) ? SDL_TRUE : SDL_FALSE;
        event.motion.state = GetButtonState(mouse, SDL_TRUE);
        event.motion.x = mouse->x;
        event.motion.y = mouse->y;
        event.motion.xrel = xrel;
        event.motion.yrel = yrel;
        posted = (SDL_PushEvent(&event) > 0);
    }
    if (relative) {
        mouse->last_x = mouse->x;
        mouse->last_y = mouse->y;
    } else {
        /* Use unclamped values if we're getting events outside the window */
        mouse->last_x = x;
        mouse->last_y = y;
    }
    return posted;
}

static SDL_MouseInputSource *GetMouseInputSource(SDL_Mouse *mouse, SDL_MouseID mouseID)
{
    SDL_MouseInputSource *source, *sources;
    int i;

    for (i = 0; i < mouse->num_sources; ++i) {
        source = &mouse->sources[i];
        if (source->mouseID == mouseID) {
            return source;
        }
    }

    sources = (SDL_MouseInputSource *)SDL_realloc(mouse->sources, (mouse->num_sources + 1) * sizeof(*mouse->sources));
    if (sources) {
        mouse->sources = sources;
        ++mouse->num_sources;
        source = &sources[mouse->num_sources - 1];
        source->mouseID = mouseID;
        source->buttonstate = 0;
        return source;
    }
    return NULL;
}

static SDL_MouseClickState *GetMouseClickState(SDL_Mouse *mouse, Uint8 button)
{
    if (button >= mouse->num_clickstates) {
        int i, count = button + 1;
        SDL_MouseClickState *clickstate = (SDL_MouseClickState *)SDL_realloc(mouse->clickstate, count * sizeof(*mouse->clickstate));
        if (!clickstate) {
            return NULL;
        }
        mouse->clickstate = clickstate;

        for (i = mouse->num_clickstates; i < count; ++i) {
            SDL_zero(mouse->clickstate[i]);
        }
        mouse->num_clickstates = count;
    }
    return &mouse->clickstate[button];
}

static int SDL_PrivateSendMouseButton(SDL_Window *window, SDL_MouseID mouseID, Uint8 state, Uint8 button, int clicks)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    int posted;
    Uint32 type;
    Uint32 buttonstate;
    SDL_MouseInputSource *source;

    source = GetMouseInputSource(mouse, mouseID);
    if (!source) {
        return 0;
    }
    buttonstate = source->buttonstate;

    /* SDL_HINT_MOUSE_TOUCH_EVENTS: controlling whether mouse events should generate synthetic touch events */
    if (mouse->mouse_touch_events) {
        if (mouseID != SDL_TOUCH_MOUSEID && button == SDL_BUTTON_LEFT) {
            if (state == SDL_PRESSED) {
                track_mouse_down = SDL_TRUE;
            } else {
                track_mouse_down = SDL_FALSE;
            }
            if (window) {
                float fx = (float)mouse->x / (float)window->w;
                float fy = (float)mouse->y / (float)window->h;
                SDL_SendTouch(SDL_MOUSE_TOUCHID, 0, window, track_mouse_down, fx, fy, 1.0f);
            }
        }
    }

    /* SDL_HINT_TOUCH_MOUSE_EVENTS: if not set, discard synthetic mouse events coming from platform layer */
    if (mouse->touch_mouse_events == 0) {
        if (mouseID == SDL_TOUCH_MOUSEID) {
            return 0;
        }
    }

    /* Figure out which event to perform */
    switch (state) {
    case SDL_PRESSED:
        type = SDL_MOUSEBUTTONDOWN;
        buttonstate |= SDL_BUTTON(button);
        break;
    case SDL_RELEASED:
        type = SDL_MOUSEBUTTONUP;
        buttonstate &= ~SDL_BUTTON(button);
        break;
    default:
        /* Invalid state -- bail */
        return 0;
    }

    /* We do this after calculating buttonstate so button presses gain focus */
    if (window && state == SDL_PRESSED) {
        SDL_UpdateMouseFocus(window, mouse->x, mouse->y, buttonstate, SDL_TRUE);
    }

    if (buttonstate == source->buttonstate) {
        /* Ignore this event, no state change */
        return 0;
    }
    source->buttonstate = buttonstate;

    if (clicks < 0) {
        SDL_MouseClickState *clickstate = GetMouseClickState(mouse, button);
        if (clickstate) {
            if (state == SDL_PRESSED) {
                Uint32 now = SDL_GetTicks();

                if (SDL_TICKS_PASSED(now, clickstate->last_timestamp + mouse->double_click_time) ||
                    SDL_abs(mouse->x - clickstate->last_x) > mouse->double_click_radius ||
                    SDL_abs(mouse->y - clickstate->last_y) > mouse->double_click_radius) {
                    clickstate->click_count = 0;
                }
                clickstate->last_timestamp = now;
                clickstate->last_x = mouse->x;
                clickstate->last_y = mouse->y;
                if (clickstate->click_count < 255) {
                    ++clickstate->click_count;
                }
            }
            clicks = clickstate->click_count;
        } else {
            clicks = 1;
        }
    }

    /* Post the event, if desired */
    posted = 0;
    if (SDL_GetEventState(type) == SDL_ENABLE) {
        SDL_Event event;
        event.type = type;
        event.button.windowID = mouse->focus ? mouse->focus->id : 0;
        event.button.which = mouseID;
        event.button.state = state;
        event.button.button = button;
        event.button.clicks = (Uint8)SDL_min(clicks, 255);
        event.button.x = mouse->x;
        event.button.y = mouse->y;
        posted = (SDL_PushEvent(&event) > 0);
    }

    /* We do this after dispatching event so button releases can lose focus */
    if (window && state == SDL_RELEASED) {
        SDL_UpdateMouseFocus(window, mouse->x, mouse->y, buttonstate, SDL_TRUE);
    }

    /* Automatically capture the mouse while buttons are pressed */
    if (mouse->auto_capture) {
        SDL_UpdateMouseCapture(SDL_FALSE);
    }

    return posted;
}

int SDL_SendMouseButtonClicks(SDL_Window *window, SDL_MouseID mouseID, Uint8 state, Uint8 button, int clicks)
{
    clicks = SDL_max(clicks, 0);
    return SDL_PrivateSendMouseButton(window, mouseID, state, button, clicks);
}

int SDL_SendMouseButton(SDL_Window *window, SDL_MouseID mouseID, Uint8 state, Uint8 button)
{
    return SDL_PrivateSendMouseButton(window, mouseID, state, button, -1);
}

int SDL_SendMouseWheel(SDL_Window *window, SDL_MouseID mouseID, float x, float y, SDL_MouseWheelDirection direction)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    int posted;
    int integral_x, integral_y;

    if (window) {
        SDL_SetMouseFocus(window);
    }

    if (x == 0.0f && y == 0.0f) {
        return 0;
    }

    if (x > 0.0f) {
        if (mouse->accumulated_wheel_x < 0.0f) {
            mouse->accumulated_wheel_x = 0.0f;
        }
    } else if (x < 0.0f) {
        if (mouse->accumulated_wheel_x > 0.0f) {
            mouse->accumulated_wheel_x = 0.0f;
        }
    }
    mouse->accumulated_wheel_x += x;
    if (mouse->accumulated_wheel_x > 0.0f) {
        integral_x = (int)SDL_floor(mouse->accumulated_wheel_x);
    } else if (mouse->accumulated_wheel_x < 0.0f) {
        integral_x = (int)SDL_ceil(mouse->accumulated_wheel_x);
    } else {
        integral_x = 0;
    }
    mouse->accumulated_wheel_x -= integral_x;

    if (y > 0.0f) {
        if (mouse->accumulated_wheel_y < 0.0f) {
            mouse->accumulated_wheel_y = 0.0f;
        }
    } else if (y < 0.0f) {
        if (mouse->accumulated_wheel_y > 0.0f) {
            mouse->accumulated_wheel_y = 0.0f;
        }
    }
    mouse->accumulated_wheel_y += y;
    if (mouse->accumulated_wheel_y > 0.0f) {
        integral_y = (int)SDL_floor(mouse->accumulated_wheel_y);
    } else if (mouse->accumulated_wheel_y < 0.0f) {
        integral_y = (int)SDL_ceil(mouse->accumulated_wheel_y);
    } else {
        integral_y = 0;
    }
    mouse->accumulated_wheel_y -= integral_y;

    /* Post the event, if desired */
    posted = 0;
    if (SDL_GetEventState(SDL_MOUSEWHEEL) == SDL_ENABLE) {
        SDL_Event event;
        event.type = SDL_MOUSEWHEEL;
        event.wheel.windowID = mouse->focus ? mouse->focus->id : 0;
        event.wheel.which = mouseID;
        event.wheel.x = integral_x;
        event.wheel.y = integral_y;
        event.wheel.preciseX = x;
        event.wheel.preciseY = y;
        event.wheel.direction = (Uint32)direction;
        event.wheel.mouseX = mouse->x;
        event.wheel.mouseY = mouse->y;
        posted = (SDL_PushEvent(&event) > 0);
    }
    return posted;
}

void SDL_MouseQuit(void)
{
    SDL_Cursor *cursor, *next;
    SDL_Mouse *mouse = SDL_GetMouse();

    if (mouse->CaptureMouse) {
        SDL_CaptureMouse(SDL_FALSE);
        SDL_UpdateMouseCapture(SDL_TRUE);
    }
    SDL_SetRelativeMouseMode(SDL_FALSE);
    SDL_ShowCursor(1);

    cursor = mouse->cursors;
    while (cursor) {
        next = cursor->next;
        SDL_FreeCursor(cursor);
        cursor = next;
    }
    mouse->cursors = NULL;
    mouse->cur_cursor = NULL;

    if (mouse->def_cursor && mouse->FreeCursor) {
        mouse->FreeCursor(mouse->def_cursor);
        mouse->def_cursor = NULL;
    }

    if (mouse->sources) {
        SDL_free(mouse->sources);
        mouse->sources = NULL;
    }
    mouse->num_sources = 0;

    if (mouse->clickstate) {
        SDL_free(mouse->clickstate);
        mouse->clickstate = NULL;
    }
    mouse->num_clickstates = 0;

    if (mouse->system_scale_values) {
        SDL_free(mouse->system_scale_values);
        mouse->system_scale_values = NULL;
    }
    mouse->num_system_scale_values = 0;

    SDL_DelHintCallback(SDL_HINT_MOUSE_DOUBLE_CLICK_TIME,
                        SDL_MouseDoubleClickTimeChanged, mouse);

    SDL_DelHintCallback(SDL_HINT_MOUSE_DOUBLE_CLICK_RADIUS,
                        SDL_MouseDoubleClickRadiusChanged, mouse);

    SDL_DelHintCallback(SDL_HINT_MOUSE_NORMAL_SPEED_SCALE,
                        SDL_MouseNormalSpeedScaleChanged, mouse);

    SDL_DelHintCallback(SDL_HINT_MOUSE_RELATIVE_SPEED_SCALE,
                        SDL_MouseRelativeSpeedScaleChanged, mouse);

    SDL_DelHintCallback(SDL_HINT_MOUSE_RELATIVE_SYSTEM_SCALE,
                        SDL_MouseRelativeSystemScaleChanged, mouse);

    SDL_DelHintCallback(SDL_HINT_TOUCH_MOUSE_EVENTS,
                        SDL_TouchMouseEventsChanged, mouse);

    SDL_DelHintCallback(SDL_HINT_MOUSE_TOUCH_EVENTS,
                        SDL_MouseTouchEventsChanged, mouse);

    SDL_DelHintCallback(SDL_HINT_MOUSE_AUTO_CAPTURE,
                        SDL_MouseAutoCaptureChanged, mouse);

    SDL_DelHintCallback(SDL_HINT_MOUSE_RELATIVE_WARP_MOTION,
                        SDL_MouseRelativeWarpMotionChanged, mouse);

    SDL_DelHintCallback(SDL_HINT_MOUSE_RELATIVE_CURSOR_VISIBLE,
                        SDL_MouseRelativeCursorVisibleChanged, mouse);
}

Uint32 SDL_GetMouseState(int *x, int *y)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (x) {
        *x = mouse->x;
    }
    if (y) {
        *y = mouse->y;
    }
    return GetButtonState(mouse, SDL_TRUE);
}

Uint32 SDL_GetRelativeMouseState(int *x, int *y)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (x) {
        *x = mouse->xdelta;
    }
    if (y) {
        *y = mouse->ydelta;
    }
    mouse->xdelta = 0;
    mouse->ydelta = 0;
    return GetButtonState(mouse, SDL_TRUE);
}

Uint32 SDL_GetGlobalMouseState(int *x, int *y)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (mouse->GetGlobalMouseState) {
        int tmpx, tmpy;

        /* make sure these are never NULL for the backend implementations... */
        if (!x) {
            x = &tmpx;
        }
        if (!y) {
            y = &tmpy;
        }

        *x = *y = 0;

        return mouse->GetGlobalMouseState(x, y);
    } else {
        return SDL_GetMouseState(x, y);
    }
}

void SDL_PerformWarpMouseInWindow(SDL_Window *window, int x, int y, SDL_bool ignore_relative_mode)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (!window) {
        window = mouse->focus;
    }

    if (!window) {
        return;
    }

    if ((window->flags & SDL_WINDOW_MINIMIZED) == SDL_WINDOW_MINIMIZED) {
        return;
    }

    /* Ignore the previous position when we warp */
    mouse->last_x = x;
    mouse->last_y = y;
    mouse->has_position = SDL_FALSE;

    if (mouse->relative_mode && !ignore_relative_mode) {
        /* 2.0.22 made warping in relative mode actually functional, which
         * surprised many applications that weren't expecting the additional
         * mouse motion.
         *
         * So for now, warping in relative mode adjusts the absolution position
         * but doesn't generate motion events, unless SDL_HINT_MOUSE_RELATIVE_WARP_MOTION is set.
         */
        if (!mouse->relative_mode_warp_motion) {
            mouse->x = x;
            mouse->y = y;
            mouse->has_position = SDL_TRUE;
            return;
        }
    }

    if (mouse->WarpMouse &&
        (!mouse->relative_mode || mouse->relative_mode_warp)) {
        mouse->WarpMouse(window, x, y);
    } else {
        SDL_PrivateSendMouseMotion(window, mouse->mouseID, 0, x, y);
    }
}

void SDL_WarpMouseInWindow(SDL_Window *window, int x, int y)
{
    SDL_PerformWarpMouseInWindow(window, x, y, SDL_FALSE);
}

int SDL_WarpMouseGlobal(int x, int y)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (mouse->WarpMouseGlobal) {
        return mouse->WarpMouseGlobal(x, y);
    }

    return SDL_Unsupported();
}

static SDL_bool ShouldUseRelativeModeWarp(SDL_Mouse *mouse)
{
    if (!mouse->WarpMouse) {
        /* Need this functionality for relative mode warp implementation */
        return SDL_FALSE;
    }

    return SDL_GetHintBoolean(SDL_HINT_MOUSE_RELATIVE_MODE_WARP, SDL_FALSE);
}

int SDL_SetRelativeMouseMode(SDL_bool enabled)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_Window *focusWindow = SDL_GetKeyboardFocus();

    if (enabled == mouse->relative_mode) {
        return 0;
    }

    /* Set the relative mode */
    if (!enabled && mouse->relative_mode_warp) {
        mouse->relative_mode_warp = SDL_FALSE;
    } else if (enabled && ShouldUseRelativeModeWarp(mouse)) {
        mouse->relative_mode_warp = SDL_TRUE;
    } else if (!mouse->SetRelativeMouseMode || mouse->SetRelativeMouseMode(enabled) < 0) {
        if (enabled) {
            /* Fall back to warp mode if native relative mode failed */
            if (!mouse->WarpMouse) {
                return SDL_SetError("No relative mode implementation available");
            }
            mouse->relative_mode_warp = SDL_TRUE;
        }
    }
    mouse->relative_mode = enabled;
    mouse->scale_accum_x = 0.0f;
    mouse->scale_accum_y = 0.0f;

    if (enabled) {
        /* Update cursor visibility before we potentially warp the mouse */
        SDL_SetCursor(NULL);
    }

    if (enabled && focusWindow) {
        SDL_SetMouseFocus(focusWindow);

        if (mouse->relative_mode_warp) {
            SDL_PerformWarpMouseInWindow(focusWindow, focusWindow->w / 2, focusWindow->h / 2, SDL_TRUE);
        }
    }

    if (focusWindow) {
        SDL_UpdateWindowGrab(focusWindow);

        /* Put the cursor back to where the application expects it */
        if (!enabled) {
            SDL_PerformWarpMouseInWindow(focusWindow, mouse->x, mouse->y, SDL_TRUE);
        }

        SDL_UpdateMouseCapture(SDL_FALSE);
    }

    if (!enabled) {
        /* Update cursor visibility after we restore the mouse position */
        SDL_SetCursor(NULL);
    }

    /* Flush pending mouse motion - ideally we would pump events, but that's not always safe */
    SDL_FlushEvent(SDL_MOUSEMOTION);

    return 0;
}

SDL_bool SDL_GetRelativeMouseMode(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    return mouse->relative_mode;
}

int SDL_UpdateMouseCapture(SDL_bool force_release)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_Window *capture_window = NULL;

    if (!mouse->CaptureMouse) {
        return 0;
    }

    if (!force_release) {
        if (SDL_GetMessageBoxCount() == 0 &&
            (mouse->capture_desired || (mouse->auto_capture && GetButtonState(mouse, SDL_FALSE) != 0))) {
            if (!mouse->relative_mode) {
                capture_window = SDL_GetKeyboardFocus();
            }
        }
    }

    if (capture_window != mouse->capture_window) {
        /* We can get here recursively on Windows, so make sure we complete
         * all of the window state operations before we change the capture state
         * (e.g. https://github.com/libsdl-org/SDL/pull/5608)
         */
        SDL_Window *previous_capture = mouse->capture_window;

        if (previous_capture) {
            previous_capture->flags &= ~SDL_WINDOW_MOUSE_CAPTURE;
        }

        if (capture_window) {
            capture_window->flags |= SDL_WINDOW_MOUSE_CAPTURE;
        }

        mouse->capture_window = capture_window;

        if (mouse->CaptureMouse(capture_window) < 0) {
            /* CaptureMouse() will have set an error, just restore the state */
            if (previous_capture) {
                previous_capture->flags |= SDL_WINDOW_MOUSE_CAPTURE;
            }
            if (capture_window) {
                capture_window->flags &= ~SDL_WINDOW_MOUSE_CAPTURE;
            }
            mouse->capture_window = previous_capture;

            return -1;
        }
    }
    return 0;
}

int SDL_CaptureMouse(SDL_bool enabled)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (!mouse->CaptureMouse) {
        return SDL_Unsupported();
    }

#if defined(__WIN32__) || defined(__WINGDK__)
    /* Windows mouse capture is tied to the current thread, and must be called
     * from the thread that created the window being captured. Since we update
     * the mouse capture state from the event processing, any application state
     * changes must be processed on that thread as well.
     */
    if (!SDL_OnVideoThread()) {
        return SDL_SetError("SDL_CaptureMouse() must be called on the main thread");
    }
#endif /* defined(__WIN32__) || defined(__WINGDK__) */

    if (enabled && SDL_GetKeyboardFocus() == NULL) {
        return SDL_SetError("No window has focus");
    }
    mouse->capture_desired = enabled;

    return SDL_UpdateMouseCapture(SDL_FALSE);
}

SDL_Cursor *SDL_CreateCursor(const Uint8 *data, const Uint8 *mask,
                 int w, int h, int hot_x, int hot_y)
{
    SDL_Surface *surface;
    SDL_Cursor *cursor;
    int x, y;
    Uint32 *pixel;
    Uint8 datab = 0, maskb = 0;
    const Uint32 black = 0xFF000000;
    const Uint32 white = 0xFFFFFFFF;
    const Uint32 transparent = 0x00000000;

    /* Make sure the width is a multiple of 8 */
    w = ((w + 7) & ~7);

    /* Create the surface from a bitmap */
    surface = SDL_CreateRGBSurface(0, w, h, 32,
                                   0x00FF0000,
                                   0x0000FF00,
                                   0x000000FF,
                                   0xFF000000);
    if (!surface) {
        return NULL;
    }
    for (y = 0; y < h; ++y) {
        pixel = (Uint32 *)((Uint8 *)surface->pixels + y * surface->pitch);
        for (x = 0; x < w; ++x) {
            if ((x % 8) == 0) {
                datab = *data++;
                maskb = *mask++;
            }
            if (maskb & 0x80) {
                *pixel++ = (datab & 0x80) ? black : white;
            } else {
                *pixel++ = (datab & 0x80) ? black : transparent;
            }
            datab <<= 1;
            maskb <<= 1;
        }
    }

    cursor = SDL_CreateColorCursor(surface, hot_x, hot_y);

    SDL_FreeSurface(surface);

    return cursor;
}

SDL_Cursor *SDL_CreateColorCursor(SDL_Surface *surface, int hot_x, int hot_y)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_Surface *temp = NULL;
    SDL_Cursor *cursor;

    if (!surface) {
        SDL_InvalidParamError("surface");
        return NULL;
    }

    /* Sanity check the hot spot */
    if ((hot_x < 0) || (hot_y < 0) ||
        (hot_x >= surface->w) || (hot_y >= surface->h)) {
        SDL_SetError("Cursor hot spot doesn't lie within cursor");
        return NULL;
    }

    if (surface->format->format != SDL_PIXELFORMAT_ARGB8888) {
        temp = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_ARGB8888, 0);
        if (!temp) {
            return NULL;
        }
        surface = temp;
    }

    if (mouse->CreateCursor) {
        cursor = mouse->CreateCursor(surface, hot_x, hot_y);
    } else {
        cursor = SDL_calloc(1, sizeof(*cursor));
        if (!cursor) {
            SDL_OutOfMemory();
        }
    }

    if (cursor) {
        cursor->next = mouse->cursors;
        mouse->cursors = cursor;
    }

    SDL_FreeSurface(temp);

    return cursor;
}

SDL_Cursor *SDL_CreateSystemCursor(SDL_SystemCursor id)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_Cursor *cursor;

    if (!mouse->CreateSystemCursor) {
        SDL_SetError("CreateSystemCursor is not currently supported");
        return NULL;
    }

    cursor = mouse->CreateSystemCursor(id);
    if (cursor) {
        cursor->next = mouse->cursors;
        mouse->cursors = cursor;
    }

    return cursor;
}

/* SDL_SetCursor(NULL) can be used to force the cursor redraw,
   if this is desired for any reason.  This is used when setting
   the video mode and when the SDL window gains the mouse focus.
 */
void SDL_SetCursor(SDL_Cursor *cursor)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    /* Return immediately if setting the cursor to the currently set one (fixes #7151) */
    if (cursor == mouse->cur_cursor) {
        return;
    }

    /* Set the new cursor */
    if (cursor) {
        /* Make sure the cursor is still valid for this mouse */
        if (cursor != mouse->def_cursor) {
            SDL_Cursor *found;
            for (found = mouse->cursors; found; found = found->next) {
                if (found == cursor) {
                    break;
                }
            }
            if (!found) {
                SDL_SetError("Cursor not associated with the current mouse");
                return;
            }
        }
        mouse->cur_cursor = cursor;
    } else {
        if (mouse->focus) {
            cursor = mouse->cur_cursor;
        } else {
            cursor = mouse->def_cursor;
        }
    }

    if (cursor && mouse->cursor_shown && (!mouse->relative_mode || mouse->relative_mode_cursor_visible)) {
        if (mouse->ShowCursor) {
            mouse->ShowCursor(cursor);
        }
    } else {
        if (mouse->ShowCursor) {
            mouse->ShowCursor(NULL);
        }
    }
}

SDL_Cursor *SDL_GetCursor(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (!mouse) {
        return NULL;
    }
    return mouse->cur_cursor;
}

SDL_Cursor *SDL_GetDefaultCursor(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (!mouse) {
        return NULL;
    }
    return mouse->def_cursor;
}

void SDL_FreeCursor(SDL_Cursor *cursor)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_Cursor *curr, *prev;

    if (!cursor) {
        return;
    }

    if (cursor == mouse->def_cursor) {
        return;
    }
    if (cursor == mouse->cur_cursor) {
        SDL_SetCursor(mouse->def_cursor);
    }

    for (prev = NULL, curr = mouse->cursors; curr;
         prev = curr, curr = curr->next) {
        if (curr == cursor) {
            if (prev) {
                prev->next = curr->next;
            } else {
                mouse->cursors = curr->next;
            }

            if (mouse->FreeCursor) {
                mouse->FreeCursor(curr);
            } else {
                SDL_free(curr);
            }
            return;
        }
    }
}

int SDL_ShowCursor(int toggle)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_bool shown;

    if (!mouse) {
        return 0;
    }

    shown = mouse->cursor_shown;
    if (toggle >= 0) {
        if (toggle) {
            mouse->cursor_shown = SDL_TRUE;
        } else {
            mouse->cursor_shown = SDL_FALSE;
        }
        if (mouse->cursor_shown != shown) {
            SDL_SetCursor(NULL);
        }
    }
    return shown;
}

/* vi: set ts=4 sw=4 expandtab: */
