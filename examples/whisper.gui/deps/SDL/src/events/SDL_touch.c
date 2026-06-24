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

/* General touch handling code for SDL */

#include "SDL_events.h"
#include "SDL_events_c.h"
#include "../video/SDL_sysvideo.h"

static int SDL_num_touch = 0;
static SDL_Touch **SDL_touchDevices = NULL;

/* for mapping touch events to mice */

#define SYNTHESIZE_TOUCH_TO_MOUSE 1

#if SYNTHESIZE_TOUCH_TO_MOUSE
static SDL_bool finger_touching = SDL_FALSE;
static SDL_FingerID track_fingerid;
static SDL_TouchID track_touchid;
#endif

/* Public functions */
int SDL_TouchInit(void)
{
    return 0;
}

int SDL_GetNumTouchDevices(void)
{
    return SDL_num_touch;
}

SDL_TouchID SDL_GetTouchDevice(int index)
{
    if (index < 0 || index >= SDL_num_touch) {
        SDL_SetError("Unknown touch device index %d", index);
        return 0;
    }
    return SDL_touchDevices[index]->id;
}

const char *SDL_GetTouchName(int index)
{
    if (index < 0 || index >= SDL_num_touch) {
        SDL_SetError("Unknown touch device");
        return NULL;
    }
    return SDL_touchDevices[index]->name;
}

static int SDL_GetTouchIndex(SDL_TouchID id)
{
    int index;
    SDL_Touch *touch;

    for (index = 0; index < SDL_num_touch; ++index) {
        touch = SDL_touchDevices[index];
        if (touch->id == id) {
            return index;
        }
    }
    return -1;
}

SDL_Touch *SDL_GetTouch(SDL_TouchID id)
{
    int index = SDL_GetTouchIndex(id);
    if (index < 0 || index >= SDL_num_touch) {
        if (SDL_GetVideoDevice()->ResetTouch != NULL) {
            SDL_SetError("Unknown touch id %d, resetting", (int)id);
            (SDL_GetVideoDevice()->ResetTouch)(SDL_GetVideoDevice());
        } else {
            SDL_SetError("Unknown touch device id %d, cannot reset", (int)id);
        }
        return NULL;
    }
    return SDL_touchDevices[index];
}

SDL_TouchDeviceType SDL_GetTouchDeviceType(SDL_TouchID id)
{
    SDL_Touch *touch = SDL_GetTouch(id);
    if (touch) {
        return touch->type;
    }
    return SDL_TOUCH_DEVICE_INVALID;
}

static int SDL_GetFingerIndex(const SDL_Touch *touch, SDL_FingerID fingerid)
{
    int index;
    for (index = 0; index < touch->num_fingers; ++index) {
        if (touch->fingers[index]->id == fingerid) {
            return index;
        }
    }
    return -1;
}

static SDL_Finger *SDL_GetFinger(const SDL_Touch *touch, SDL_FingerID id)
{
    int index = SDL_GetFingerIndex(touch, id);
    if (index < 0 || index >= touch->num_fingers) {
        return NULL;
    }
    return touch->fingers[index];
}

int SDL_GetNumTouchFingers(SDL_TouchID touchID)
{
    SDL_Touch *touch = SDL_GetTouch(touchID);
    if (touch) {
        return touch->num_fingers;
    }
    return 0;
}

SDL_Finger *SDL_GetTouchFinger(SDL_TouchID touchID, int index)
{
    SDL_Touch *touch = SDL_GetTouch(touchID);
    if (!touch) {
        return NULL;
    }
    if (index < 0 || index >= touch->num_fingers) {
        SDL_SetError("Unknown touch finger");
        return NULL;
    }
    return touch->fingers[index];
}

int SDL_AddTouch(SDL_TouchID touchID, SDL_TouchDeviceType type, const char *name)
{
    SDL_Touch **touchDevices;
    int index;

    index = SDL_GetTouchIndex(touchID);
    if (index >= 0) {
        return index;
    }

    /* Add the touch to the list of touch */
    touchDevices = (SDL_Touch **)SDL_realloc(SDL_touchDevices,
                                             (SDL_num_touch + 1) * sizeof(*touchDevices));
    if (!touchDevices) {
        return SDL_OutOfMemory();
    }

    SDL_touchDevices = touchDevices;
    index = SDL_num_touch;

    SDL_touchDevices[index] = (SDL_Touch *)SDL_malloc(sizeof(*SDL_touchDevices[index]));
    if (!SDL_touchDevices[index]) {
        return SDL_OutOfMemory();
    }

    /* Added touch to list */
    ++SDL_num_touch;

    /* we're setting the touch properties */
    SDL_touchDevices[index]->id = touchID;
    SDL_touchDevices[index]->type = type;
    SDL_touchDevices[index]->num_fingers = 0;
    SDL_touchDevices[index]->max_fingers = 0;
    SDL_touchDevices[index]->fingers = NULL;
    SDL_touchDevices[index]->name = SDL_strdup(name ? name : "");

    /* Record this touch device for gestures */
    /* We could do this on the fly in the gesture code if we wanted */
    SDL_GestureAddTouch(touchID);

    return index;
}

static int SDL_AddFinger(SDL_Touch *touch, SDL_FingerID fingerid, float x, float y, float pressure)
{
    SDL_Finger *finger;

    if (touch->num_fingers == touch->max_fingers) {
        SDL_Finger **new_fingers;
        new_fingers = (SDL_Finger **)SDL_realloc(touch->fingers, (touch->max_fingers + 1) * sizeof(*touch->fingers));
        if (!new_fingers) {
            return SDL_OutOfMemory();
        }
        touch->fingers = new_fingers;
        touch->fingers[touch->max_fingers] = (SDL_Finger *)SDL_malloc(sizeof(*finger));
        if (!touch->fingers[touch->max_fingers]) {
            return SDL_OutOfMemory();
        }
        touch->max_fingers++;
    }

    finger = touch->fingers[touch->num_fingers++];
    finger->id = fingerid;
    finger->x = x;
    finger->y = y;
    finger->pressure = pressure;
    return 0;
}

static int SDL_DelFinger(SDL_Touch *touch, SDL_FingerID fingerid)
{
    SDL_Finger *temp;

    int index = SDL_GetFingerIndex(touch, fingerid);
    if (index < 0) {
        return -1;
    }

    touch->num_fingers--;
    temp = touch->fingers[index];
    touch->fingers[index] = touch->fingers[touch->num_fingers];
    touch->fingers[touch->num_fingers] = temp;
    return 0;
}

int SDL_SendTouch(SDL_TouchID id, SDL_FingerID fingerid, SDL_Window *window,
                  SDL_bool down, float x, float y, float pressure)
{
    int posted;
    SDL_Finger *finger;
    SDL_Mouse *mouse;

    SDL_Touch *touch = SDL_GetTouch(id);
    if (!touch) {
        return -1;
    }

    mouse = SDL_GetMouse();

#if SYNTHESIZE_TOUCH_TO_MOUSE
    /* SDL_HINT_TOUCH_MOUSE_EVENTS: controlling whether touch events should generate synthetic mouse events */
    /* SDL_HINT_VITA_TOUCH_MOUSE_DEVICE: controlling which touchpad should generate synthetic mouse events, PSVita-only */
    {
#if defined(__vita__)
        if (mouse->touch_mouse_events && ((mouse->vita_touch_mouse_device == id) || (mouse->vita_touch_mouse_device == 2))) {
#else
        if (mouse->touch_mouse_events) {
#endif
            /* FIXME: maybe we should only restrict to a few SDL_TouchDeviceType */
            if (id != SDL_MOUSE_TOUCHID) {
                if (window) {
                    if (down) {
                        if (finger_touching == SDL_FALSE) {
                            int pos_x = (int)(x * (float)window->w);
                            int pos_y = (int)(y * (float)window->h);
                            if (pos_x < 0) {
                                pos_x = 0;
                            }
                            if (pos_x > window->w - 1) {
                                pos_x = window->w - 1;
                            }
                            if (pos_y < 0) {
                                pos_y = 0;
                            }
                            if (pos_y > window->h - 1) {
                                pos_y = window->h - 1;
                            }
                            SDL_SendMouseMotion(window, SDL_TOUCH_MOUSEID, 0, pos_x, pos_y);
                            SDL_SendMouseButton(window, SDL_TOUCH_MOUSEID, SDL_PRESSED, SDL_BUTTON_LEFT);
                        }
                    } else {
                        if (finger_touching == SDL_TRUE && track_touchid == id && track_fingerid == fingerid) {
                            SDL_SendMouseButton(window, SDL_TOUCH_MOUSEID, SDL_RELEASED, SDL_BUTTON_LEFT);
                        }
                    }
                }
                if (down) {
                    if (finger_touching == SDL_FALSE) {
                        finger_touching = SDL_TRUE;
                        track_touchid = id;
                        track_fingerid = fingerid;
                    }
                } else {
                    if (finger_touching == SDL_TRUE && track_touchid == id && track_fingerid == fingerid) {
                        finger_touching = SDL_FALSE;
                    }
                }
            }
        }
    }
#endif

    /* SDL_HINT_MOUSE_TOUCH_EVENTS: if not set, discard synthetic touch events coming from platform layer */
    if (mouse->mouse_touch_events == 0) {
        if (id == SDL_MOUSE_TOUCHID) {
            return 0;
        }
    }

    finger = SDL_GetFinger(touch, fingerid);
    if (down) {
        if (finger) {
            /* This finger is already down.
               Assume the finger-up for the previous touch was lost, and send it. */
            SDL_SendTouch(id, fingerid, window, SDL_FALSE, x, y, pressure);
        }

        if (SDL_AddFinger(touch, fingerid, x, y, pressure) < 0) {
            return 0;
        }

        posted = 0;
        if (SDL_GetEventState(SDL_FINGERDOWN) == SDL_ENABLE) {
            SDL_Event event;
            event.tfinger.type = SDL_FINGERDOWN;
            event.tfinger.touchId = id;
            event.tfinger.fingerId = fingerid;
            event.tfinger.x = x;
            event.tfinger.y = y;
            event.tfinger.dx = 0;
            event.tfinger.dy = 0;
            event.tfinger.pressure = pressure;
            event.tfinger.windowID = window ? SDL_GetWindowID(window) : 0;
            posted = (SDL_PushEvent(&event) > 0);
        }
    } else {
        if (!finger) {
            /* This finger is already up */
            return 0;
        }

        posted = 0;
        if (SDL_GetEventState(SDL_FINGERUP) == SDL_ENABLE) {
            SDL_Event event;
            event.tfinger.type = SDL_FINGERUP;
            event.tfinger.touchId = id;
            event.tfinger.fingerId = fingerid;
            /* I don't trust the coordinates passed on fingerUp */
            event.tfinger.x = finger->x;
            event.tfinger.y = finger->y;
            event.tfinger.dx = 0;
            event.tfinger.dy = 0;
            event.tfinger.pressure = pressure;
            event.tfinger.windowID = window ? SDL_GetWindowID(window) : 0;
            posted = (SDL_PushEvent(&event) > 0);
        }

        SDL_DelFinger(touch, fingerid);
    }
    return posted;
}

int SDL_SendTouchMotion(SDL_TouchID id, SDL_FingerID fingerid, SDL_Window *window,
                        float x, float y, float pressure)
{
    SDL_Touch *touch;
    SDL_Finger *finger;
    SDL_Mouse *mouse;
    int posted;
    float xrel, yrel, prel;

    touch = SDL_GetTouch(id);
    if (!touch) {
        return -1;
    }

    mouse = SDL_GetMouse();

#if SYNTHESIZE_TOUCH_TO_MOUSE
    /* SDL_HINT_TOUCH_MOUSE_EVENTS: controlling whether touch events should generate synthetic mouse events */
    {
        if (mouse->touch_mouse_events) {
            if (id != SDL_MOUSE_TOUCHID) {
                if (window) {
                    if (finger_touching == SDL_TRUE && track_touchid == id && track_fingerid == fingerid) {
                        int pos_x = (int)(x * (float)window->w);
                        int pos_y = (int)(y * (float)window->h);
                        if (pos_x < 0) {
                            pos_x = 0;
                        }
                        if (pos_x > window->w - 1) {
                            pos_x = window->w - 1;
                        }
                        if (pos_y < 0) {
                            pos_y = 0;
                        }
                        if (pos_y > window->h - 1) {
                            pos_y = window->h - 1;
                        }
                        SDL_SendMouseMotion(window, SDL_TOUCH_MOUSEID, 0, pos_x, pos_y);
                    }
                }
            }
        }
    }
#endif

    /* SDL_HINT_MOUSE_TOUCH_EVENTS: if not set, discard synthetic touch events coming from platform layer */
    if (mouse->mouse_touch_events == 0) {
        if (id == SDL_MOUSE_TOUCHID) {
            return 0;
        }
    }

    finger = SDL_GetFinger(touch, fingerid);
    if (!finger) {
        return SDL_SendTouch(id, fingerid, window, SDL_TRUE, x, y, pressure);
    }

    xrel = x - finger->x;
    yrel = y - finger->y;
    prel = pressure - finger->pressure;

    /* Drop events that don't change state */
    if (xrel == 0.0f && yrel == 0.0f && prel == 0.0f) {
#if 0
        printf("Touch event didn't change state - dropped!\n");
#endif
        return 0;
    }

    /* Update internal touch coordinates */
    finger->x = x;
    finger->y = y;
    finger->pressure = pressure;

    /* Post the event, if desired */
    posted = 0;
    if (SDL_GetEventState(SDL_FINGERMOTION) == SDL_ENABLE) {
        SDL_Event event;
        event.tfinger.type = SDL_FINGERMOTION;
        event.tfinger.touchId = id;
        event.tfinger.fingerId = fingerid;
        event.tfinger.x = x;
        event.tfinger.y = y;
        event.tfinger.dx = xrel;
        event.tfinger.dy = yrel;
        event.tfinger.pressure = pressure;
        event.tfinger.windowID = window ? SDL_GetWindowID(window) : 0;
        posted = (SDL_PushEvent(&event) > 0);
    }
    return posted;
}

void SDL_DelTouch(SDL_TouchID id)
{
    int i, index;
    SDL_Touch *touch;

    if (SDL_num_touch == 0) {
        /* We've already cleaned up, we won't find this device */
        return;
    }

    index = SDL_GetTouchIndex(id);
    touch = SDL_GetTouch(id);
    if (!touch) {
        return;
    }

    for (i = 0; i < touch->max_fingers; ++i) {
        SDL_free(touch->fingers[i]);
    }
    SDL_free(touch->fingers);
    SDL_free(touch->name);
    SDL_free(touch);

    SDL_num_touch--;
    SDL_touchDevices[index] = SDL_touchDevices[SDL_num_touch];

    /* Delete this touch device for gestures */
    SDL_GestureDelTouch(id);
}

void SDL_TouchQuit(void)
{
    int i;

    for (i = SDL_num_touch; i--;) {
        SDL_DelTouch(SDL_touchDevices[i]->id);
    }
    SDL_assert(SDL_num_touch == 0);

    SDL_free(SDL_touchDevices);
    SDL_touchDevices = NULL;
    SDL_GestureQuit();
}

/* vi: set ts=4 sw=4 expandtab: */
