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

#if !defined(SDL_POWER_DISABLED) && defined(SDL_POWER_N3DS)

#include <3ds.h>

#include "SDL_error.h"
#include "SDL_power.h"

SDL_FORCE_INLINE SDL_PowerState GetPowerState(void);
SDL_FORCE_INLINE int ReadStateFromPTMU(bool *is_plugged, u8 *is_charging);
SDL_FORCE_INLINE int GetBatteryPercentage(void);

#define BATTERY_PERCENT_REG      0xB
#define BATTERY_PERCENT_REG_SIZE 2

SDL_bool SDL_GetPowerInfo_N3DS(SDL_PowerState *state, int *seconds, int *percent)
{
    *state = GetPowerState();
    *percent = GetBatteryPercentage();
    *seconds = -1; /* libctru doesn't provide a way to estimate battery life */

    return SDL_TRUE;
}

static SDL_PowerState GetPowerState(void)
{
    bool is_plugged;
    u8 is_charging;

    if (ReadStateFromPTMU(&is_plugged, &is_charging) < 0) {
        return SDL_POWERSTATE_UNKNOWN;
    }

    if (is_charging) {
        return SDL_POWERSTATE_CHARGING;
    }

    if (is_plugged) {
        return SDL_POWERSTATE_CHARGED;
    }

    return SDL_POWERSTATE_ON_BATTERY;
}

static int ReadStateFromPTMU(bool *is_plugged, u8 *is_charging)
{
    if (R_FAILED(ptmuInit())) {
        return SDL_SetError("Failed to initialise PTMU service");
    }

    if (R_FAILED(PTMU_GetAdapterState(is_plugged))) {
        ptmuExit();
        return SDL_SetError("Failed to read adapter state");
    }

    if (R_FAILED(PTMU_GetBatteryChargeState(is_charging))) {
        ptmuExit();
        return SDL_SetError("Failed to read battery charge state");
    }

    ptmuExit();
    return 0;
}

SDL_FORCE_INLINE int
GetBatteryPercentage(void)
{
    u8 data[BATTERY_PERCENT_REG_SIZE];

    if (R_FAILED(mcuHwcInit())) {
        return SDL_SetError("Failed to initialise mcuHwc service");
    }

    if (R_FAILED(MCUHWC_ReadRegister(BATTERY_PERCENT_REG, data, BATTERY_PERCENT_REG_SIZE))) {
        mcuHwcExit();
        return SDL_SetError("Failed to read battery register");
    }

    mcuHwcExit();

    return (int)SDL_round(data[0] + data[1] / 256.0);
}

#endif /* !SDL_POWER_DISABLED && SDL_POWER_N3DS */

/* vi: set sts=4 ts=4 sw=4 expandtab: */
