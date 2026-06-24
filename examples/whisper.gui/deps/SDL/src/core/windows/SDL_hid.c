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

#ifndef __WINRT__

#include "SDL_hid.h"

HidD_GetString_t SDL_HidD_GetManufacturerString;
HidD_GetString_t SDL_HidD_GetProductString;
HidP_GetCaps_t SDL_HidP_GetCaps;
HidP_GetButtonCaps_t SDL_HidP_GetButtonCaps;
HidP_GetValueCaps_t SDL_HidP_GetValueCaps;
HidP_MaxDataListLength_t SDL_HidP_MaxDataListLength;
HidP_GetData_t SDL_HidP_GetData;

static HMODULE s_pHIDDLL = 0;
static int s_HIDDLLRefCount = 0;

int WIN_LoadHIDDLL(void)
{
    if (s_pHIDDLL) {
        SDL_assert(s_HIDDLLRefCount > 0);
        s_HIDDLLRefCount++;
        return 0; /* already loaded */
    }

    s_pHIDDLL = LoadLibrary(TEXT("hid.dll"));
    if (!s_pHIDDLL) {
        return -1;
    }

    SDL_assert(s_HIDDLLRefCount == 0);
    s_HIDDLLRefCount = 1;

    SDL_HidD_GetManufacturerString = (HidD_GetString_t)GetProcAddress(s_pHIDDLL, "HidD_GetManufacturerString");
    SDL_HidD_GetProductString = (HidD_GetString_t)GetProcAddress(s_pHIDDLL, "HidD_GetProductString");
    SDL_HidP_GetCaps = (HidP_GetCaps_t)GetProcAddress(s_pHIDDLL, "HidP_GetCaps");
    SDL_HidP_GetButtonCaps = (HidP_GetButtonCaps_t)GetProcAddress(s_pHIDDLL, "HidP_GetButtonCaps");
    SDL_HidP_GetValueCaps = (HidP_GetValueCaps_t)GetProcAddress(s_pHIDDLL, "HidP_GetValueCaps");
    SDL_HidP_MaxDataListLength = (HidP_MaxDataListLength_t)GetProcAddress(s_pHIDDLL, "HidP_MaxDataListLength");
    SDL_HidP_GetData = (HidP_GetData_t)GetProcAddress(s_pHIDDLL, "HidP_GetData");
    if (!SDL_HidD_GetManufacturerString || !SDL_HidD_GetProductString ||
        !SDL_HidP_GetCaps || !SDL_HidP_GetButtonCaps ||
        !SDL_HidP_GetValueCaps || !SDL_HidP_MaxDataListLength || !SDL_HidP_GetData) {
        WIN_UnloadHIDDLL();
        return -1;
    }

    return 0;
}

void WIN_UnloadHIDDLL(void)
{
    if (s_pHIDDLL) {
        SDL_assert(s_HIDDLLRefCount > 0);
        if (--s_HIDDLLRefCount == 0) {
            FreeLibrary(s_pHIDDLL);
            s_pHIDDLL = NULL;
        }
    } else {
        SDL_assert(s_HIDDLLRefCount == 0);
    }
}

#endif /* !__WINRT__ */

/* vi: set ts=4 sw=4 expandtab: */
