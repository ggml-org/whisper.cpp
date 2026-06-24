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

#include "../SDL_sysjoystick.h"
#include "../SDL_joystick_c.h"
#include "SDL_steamcontroller.h"

void SDL_InitSteamControllers(SteamControllerConnectedCallback_t connectedCallback,
                              SteamControllerDisconnectedCallback_t disconnectedCallback)
{
}

void SDL_GetSteamControllerInputs(int *nbuttons, int *naxes, int *nhats)
{
    *nbuttons = 0;
    *naxes = 0;
    *nhats = 0;
}

void SDL_UpdateSteamControllers(void)
{
}

void SDL_UpdateSteamController(SDL_Joystick *joystick)
{
}

void SDL_QuitSteamControllers(void)
{
}

/* vi: set ts=4 sw=4 expandtab: */
