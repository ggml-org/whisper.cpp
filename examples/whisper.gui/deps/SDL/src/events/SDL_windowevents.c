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

/* Window event handling code for SDL */

#include "SDL_events.h"
#include "SDL_events_c.h"
#include "SDL_mouse_c.h"
#include "SDL_hints.h"

typedef struct RemovePendingSizeChangedAndResizedEvents_Data
{
    const SDL_Event *new_event;
    SDL_bool saw_resized;
} RemovePendingSizeChangedAndResizedEvents_Data;

static int SDLCALL RemovePendingSizeChangedAndResizedEvents(void *_userdata, SDL_Event *event)
{
    RemovePendingSizeChangedAndResizedEvents_Data *userdata = (RemovePendingSizeChangedAndResizedEvents_Data *)_userdata;
    const SDL_Event *new_event = userdata->new_event;

    if (event->type == SDL_WINDOWEVENT &&
        (event->window.event == SDL_WINDOWEVENT_SIZE_CHANGED ||
         event->window.event == SDL_WINDOWEVENT_RESIZED) &&
        event->window.windowID == new_event->window.windowID) {

        if (event->window.event == SDL_WINDOWEVENT_RESIZED) {
            userdata->saw_resized = SDL_TRUE;
        }

        /* We're about to post a new size event, drop the old one */
        return 0;
    }
    return 1;
}

static int SDLCALL RemovePendingMoveEvents(void *userdata, SDL_Event *event)
{
    SDL_Event *new_event = (SDL_Event *)userdata;

    if (event->type == SDL_WINDOWEVENT &&
        event->window.event == SDL_WINDOWEVENT_MOVED &&
        event->window.windowID == new_event->window.windowID) {
        /* We're about to post a new move event, drop the old one */
        return 0;
    }
    return 1;
}

static int SDLCALL RemovePendingExposedEvents(void *userdata, SDL_Event *event)
{
    SDL_Event *new_event = (SDL_Event *)userdata;

    if (event->type == SDL_WINDOWEVENT &&
        event->window.event == SDL_WINDOWEVENT_EXPOSED &&
        event->window.windowID == new_event->window.windowID) {
        /* We're about to post a new exposed event, drop the old one */
        return 0;
    }
    return 1;
}

int SDL_SendWindowEvent(SDL_Window *window, Uint8 windowevent, int data1,
                        int data2)
{
    int posted;

    if (!window) {
        return 0;
    }
    switch (windowevent) {
    case SDL_WINDOWEVENT_SHOWN:
        if (window->flags & SDL_WINDOW_SHOWN) {
            return 0;
        }
        window->flags &= ~(SDL_WINDOW_HIDDEN | SDL_WINDOW_MINIMIZED);
        window->flags |= SDL_WINDOW_SHOWN;
        SDL_OnWindowShown(window);
        break;
    case SDL_WINDOWEVENT_HIDDEN:
        if (!(window->flags & SDL_WINDOW_SHOWN)) {
            return 0;
        }
        window->flags &= ~SDL_WINDOW_SHOWN;
        window->flags |= SDL_WINDOW_HIDDEN;
        SDL_OnWindowHidden(window);
        break;
    case SDL_WINDOWEVENT_MOVED:
        if (SDL_WINDOWPOS_ISUNDEFINED(data1) ||
            SDL_WINDOWPOS_ISUNDEFINED(data2)) {
            return 0;
        }
        if (!(window->flags & SDL_WINDOW_FULLSCREEN)) {
            window->windowed.x = data1;
            window->windowed.y = data2;
        }
        if (data1 == window->x && data2 == window->y) {
            return 0;
        }
        window->x = data1;
        window->y = data2;
        SDL_OnWindowMoved(window);
        break;
    case SDL_WINDOWEVENT_RESIZED:
        if (!(window->flags & SDL_WINDOW_FULLSCREEN)) {
            window->windowed.w = data1;
            window->windowed.h = data2;
        }
        if (data1 == window->w && data2 == window->h) {
            return 0;
        }
        window->w = data1;
        window->h = data2;
        SDL_OnWindowResized(window);
        break;
    case SDL_WINDOWEVENT_MINIMIZED:
        if (window->flags & SDL_WINDOW_MINIMIZED) {
            return 0;
        }
        window->flags &= ~SDL_WINDOW_MAXIMIZED;
        window->flags |= SDL_WINDOW_MINIMIZED;
        SDL_OnWindowMinimized(window);
        break;
    case SDL_WINDOWEVENT_MAXIMIZED:
        if (window->flags & SDL_WINDOW_MAXIMIZED) {
            return 0;
        }
        window->flags &= ~SDL_WINDOW_MINIMIZED;
        window->flags |= SDL_WINDOW_MAXIMIZED;
        break;
    case SDL_WINDOWEVENT_RESTORED:
        if (!(window->flags & (SDL_WINDOW_MINIMIZED | SDL_WINDOW_MAXIMIZED))) {
            return 0;
        }
        window->flags &= ~(SDL_WINDOW_MINIMIZED | SDL_WINDOW_MAXIMIZED);
        SDL_OnWindowRestored(window);
        break;
    case SDL_WINDOWEVENT_ENTER:
        if (window->flags & SDL_WINDOW_MOUSE_FOCUS) {
            return 0;
        }
        window->flags |= SDL_WINDOW_MOUSE_FOCUS;
        SDL_OnWindowEnter(window);
        break;
    case SDL_WINDOWEVENT_LEAVE:
        if (!(window->flags & SDL_WINDOW_MOUSE_FOCUS)) {
            return 0;
        }
        window->flags &= ~SDL_WINDOW_MOUSE_FOCUS;
        SDL_OnWindowLeave(window);
        break;
    case SDL_WINDOWEVENT_FOCUS_GAINED:
        if (window->flags & SDL_WINDOW_INPUT_FOCUS) {
            return 0;
        }
        window->flags |= SDL_WINDOW_INPUT_FOCUS;
        SDL_OnWindowFocusGained(window);
        break;
    case SDL_WINDOWEVENT_FOCUS_LOST:
        if (!(window->flags & SDL_WINDOW_INPUT_FOCUS)) {
            return 0;
        }
        window->flags &= ~SDL_WINDOW_INPUT_FOCUS;
        SDL_OnWindowFocusLost(window);
        break;
    }

    /* Post the event, if desired */
    posted = 0;
    if (SDL_GetEventState(SDL_WINDOWEVENT) == SDL_ENABLE) {
        SDL_Event event;
        event.type = SDL_WINDOWEVENT;
        event.window.event = windowevent;
        event.window.data1 = data1;
        event.window.data2 = data2;
        event.window.windowID = window->id;

        /* Fixes queue overflow with resize events that aren't processed */
        if (windowevent == SDL_WINDOWEVENT_SIZE_CHANGED) {
            /* !!! FIXME: in SDL3, let's make RESIZED/SIZE_CHANGED into one event with a flag to distinguish between them, and remove all this tapdancing. */
            RemovePendingSizeChangedAndResizedEvents_Data userdata;
            userdata.new_event = &event;
            userdata.saw_resized = SDL_FALSE;
            SDL_FilterEvents(RemovePendingSizeChangedAndResizedEvents, &userdata);
            if (userdata.saw_resized) { /* if there was a pending resize, make sure one at the new dimensions remains. */
                event.window.event = SDL_WINDOWEVENT_RESIZED;
                if (SDL_PushEvent(&event) <= 0) {
                    return 0; /* oh well. */
                }
                event.window.event = SDL_WINDOWEVENT_SIZE_CHANGED; /* then push the actual event next. */
            }
        }
        if (windowevent == SDL_WINDOWEVENT_MOVED) {
            SDL_FilterEvents(RemovePendingMoveEvents, &event);
        }
        if (windowevent == SDL_WINDOWEVENT_EXPOSED) {
            SDL_FilterEvents(RemovePendingExposedEvents, &event);
        }
        posted = (SDL_PushEvent(&event) > 0);
    }

    if (windowevent == SDL_WINDOWEVENT_CLOSE) {
        if (!window->prev && !window->next) {
            if (SDL_GetHintBoolean(SDL_HINT_QUIT_ON_LAST_WINDOW_CLOSE, SDL_TRUE)) {
                SDL_SendQuit(); /* This is the last window in the list so send the SDL_QUIT event */
            }
        }
    }

    return posted;
}

/* vi: set ts=4 sw=4 expandtab: */
