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
#include "SDL_events.h"
#include <sys/time.h>
#include <dev/wscons/wsconsio.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdlib.h>
#include <fcntl.h>

#include "../../events/SDL_mouse_c.h"

typedef struct SDL_WSCONS_mouse_input_data
{
    int fd;
} SDL_WSCONS_mouse_input_data;

SDL_WSCONS_mouse_input_data *SDL_WSCONS_Init_Mouse()
{
#ifdef WSMOUSEIO_SETVERSION
    int version = WSMOUSE_EVENT_VERSION;
#endif
    SDL_WSCONS_mouse_input_data *mouseInputData = SDL_calloc(1, sizeof(SDL_WSCONS_mouse_input_data));

    if (!mouseInputData) {
        return NULL;
    }
    mouseInputData->fd = open("/dev/wsmouse", O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (mouseInputData->fd == -1) {
        free(mouseInputData);
        return NULL;
    }
#ifdef WSMOUSEIO_SETMODE
    ioctl(mouseInputData->fd, WSMOUSEIO_SETMODE, WSMOUSE_COMPAT);
#endif
#ifdef WSMOUSEIO_SETVERSION
    ioctl(mouseInputData->fd, WSMOUSEIO_SETVERSION, &version);
#endif
    return mouseInputData;
}

void updateMouse(SDL_WSCONS_mouse_input_data *inputData)
{
    struct wscons_event events[64];
    int type;
    int n, i;
    SDL_Mouse *mouse = SDL_GetMouse();

    if ((n = read(inputData->fd, events, sizeof(events))) > 0) {
        n /= sizeof(struct wscons_event);
        for (i = 0; i < n; i++) {
            type = events[i].type;
            switch (type) {
            case WSCONS_EVENT_MOUSE_DOWN:
            {
                switch (events[i].value) {
                case 0: /* Left Mouse Button. */
                    SDL_SendMouseButton(mouse->focus, mouse->mouseID, SDL_PRESSED, SDL_BUTTON_LEFT);
                    break;
                case 1: /* Middle Mouse Button. */
                    SDL_SendMouseButton(mouse->focus, mouse->mouseID, SDL_PRESSED, SDL_BUTTON_MIDDLE);
                    break;
                case 2: /* Right Mouse Button. */
                    SDL_SendMouseButton(mouse->focus, mouse->mouseID, SDL_PRESSED, SDL_BUTTON_RIGHT);
                    break;
                }
            } break;
            case WSCONS_EVENT_MOUSE_UP:
            {
                switch (events[i].value) {
                case 0: /* Left Mouse Button. */
                    SDL_SendMouseButton(mouse->focus, mouse->mouseID, SDL_RELEASED, SDL_BUTTON_LEFT);
                    break;
                case 1: /* Middle Mouse Button. */
                    SDL_SendMouseButton(mouse->focus, mouse->mouseID, SDL_RELEASED, SDL_BUTTON_MIDDLE);
                    break;
                case 2: /* Right Mouse Button. */
                    SDL_SendMouseButton(mouse->focus, mouse->mouseID, SDL_RELEASED, SDL_BUTTON_RIGHT);
                    break;
                }
            } break;
            case WSCONS_EVENT_MOUSE_DELTA_X:
            {
                SDL_SendMouseMotion(mouse->focus, mouse->mouseID, 1, events[i].value, 0);
                break;
            }
            case WSCONS_EVENT_MOUSE_DELTA_Y:
            {
                SDL_SendMouseMotion(mouse->focus, mouse->mouseID, 1, 0, -events[i].value);
                break;
            }
            case WSCONS_EVENT_MOUSE_DELTA_W:
            {
                SDL_SendMouseWheel(mouse->focus, mouse->mouseID, events[i].value, 0, SDL_MOUSEWHEEL_NORMAL);
                break;
            }
            case WSCONS_EVENT_MOUSE_DELTA_Z:
            {
                SDL_SendMouseWheel(mouse->focus, mouse->mouseID, 0, -events[i].value, SDL_MOUSEWHEEL_NORMAL);
                break;
            }
            }
        }
    }
}

void SDL_WSCONS_Quit_Mouse(SDL_WSCONS_mouse_input_data *inputData)
{
    if (!inputData) {
        return;
    }
    close(inputData->fd);
    free(inputData);
}
