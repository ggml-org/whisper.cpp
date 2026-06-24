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

#ifndef SDL_JOYSTICK_IOS_H
#define SDL_JOYSTICK_IOS_H

#include "SDL_stdinc.h"
#include "../SDL_sysjoystick.h"

#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>

@class GCController;

typedef struct joystick_hwdata
{
    SDL_bool accelerometer;

    GCController __unsafe_unretained *controller;
    void *rumble;
    int pause_button_index;
    Uint32 pause_button_pressed;

    char *name;
    SDL_Joystick *joystick;
    SDL_JoystickID instance_id;
    SDL_JoystickGUID guid;

    int naxes;
    int nbuttons;
    int nhats;
    Uint32 button_mask;
    SDL_bool is_xbox;
    SDL_bool is_ps4;
    SDL_bool is_ps5;
    SDL_bool is_switch_pro;
    SDL_bool is_switch_joycon_pair;
    SDL_bool is_switch_joyconL;
    SDL_bool is_switch_joyconR;
    SDL_bool is_stadia;
    SDL_bool is_backbone_one;
    int is_siri_remote;

    NSArray __unsafe_unretained *axes;
    NSArray __unsafe_unretained *buttons;

    SDL_bool has_dualshock_touchpad;
    SDL_bool has_xbox_paddles;
    SDL_bool has_xbox_share_button;

    struct joystick_hwdata *next;
} joystick_hwdata;

typedef joystick_hwdata SDL_JoystickDeviceItem;

#endif /* SDL_JOYSTICK_IOS_H */

/* vi: set ts=4 sw=4 expandtab: */
