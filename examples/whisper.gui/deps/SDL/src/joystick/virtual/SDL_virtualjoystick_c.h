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

#ifndef SDL_VIRTUALJOYSTICK_C_H
#define SDL_VIRTUALJOYSTICK_C_H

#ifdef SDL_JOYSTICK_VIRTUAL

#include "SDL_joystick.h"

/**
 * Data for a virtual, software-only joystick.
 */
typedef struct joystick_hwdata
{
    SDL_JoystickType type;
    SDL_bool attached;
    char *name;
    SDL_JoystickGUID guid;
    SDL_VirtualJoystickDesc desc;
    Sint16 *axes;
    Uint8 *buttons;
    Uint8 *hats;
    SDL_JoystickID instance_id;
    SDL_Joystick *joystick;

    struct joystick_hwdata *next;
} joystick_hwdata;

int SDL_JoystickAttachVirtualInner(const SDL_VirtualJoystickDesc *desc);
int SDL_JoystickDetachVirtualInner(int device_index);

int SDL_JoystickSetVirtualAxisInner(SDL_Joystick *joystick, int axis, Sint16 value);
int SDL_JoystickSetVirtualButtonInner(SDL_Joystick *joystick, int button, Uint8 value);
int SDL_JoystickSetVirtualHatInner(SDL_Joystick *joystick, int hat, Uint8 value);

#endif /* SDL_JOYSTICK_VIRTUAL */

#endif /* SDL_VIRTUALJOYSTICK_C_H */

/* vi: set ts=4 sw=4 expandtab: */
