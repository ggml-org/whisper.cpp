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

#ifndef SDL_POWER_DISABLED
#if SDL_POWER_WINRT

#include "SDL_power.h"

extern "C"
SDL_bool SDL_GetPowerInfo_WinRT(SDL_PowerState * state, int *seconds, int *percent)
{
    /* TODO, WinRT: Battery info is available on at least one WinRT platform (Windows Phone 8).  Implement SDL_GetPowerInfo_WinRT as appropriate. */
    /* Notes:
         - the Win32 function, GetSystemPowerStatus, is not available for use on WinRT
         - Windows Phone 8 has a 'Battery' class, which is documented as available for C++
             - More info on WP8's Battery class can be found at http://msdn.microsoft.com/library/windowsphone/develop/jj207231
    */
    return SDL_FALSE;
}

#endif /* SDL_POWER_WINRT */
#endif /* SDL_POWER_DISABLED */

/* vi: set ts=4 sw=4 expandtab: */
