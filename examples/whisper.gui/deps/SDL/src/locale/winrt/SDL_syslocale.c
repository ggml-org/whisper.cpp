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

#include <Windows.h>

#include "../../SDL_internal.h"
#include "../SDL_syslocale.h"

/*using namespace Windows::Graphics::Display;*/
#include <wchar.h>

void SDL_SYS_GetPreferredLocales(char *buf, size_t buflen)
{
    WCHAR wbuffer[128] = L"";
    int ret = 0;

    /* !!! FIXME: do we not have GetUserPreferredUILanguages on WinPhone or UWP? */
#if SDL_WINAPI_FAMILY_PHONE
    ret = GetLocaleInfoEx(LOCALE_NAME_SYSTEM_DEFAULT, LOCALE_SNAME, wbuffer, SDL_arraysize(wbuffer));
#else
    ret = GetSystemDefaultLocaleName(wbuffer, SDL_arraysize(wbuffer));
#endif

    if (ret > 0) {
        /* Need to convert LPWSTR to LPSTR, that is wide char to char. */
        int i;

        if (((size_t)ret) >= (buflen - 1)) {
            ret = (int)(buflen - 1);
        }
        for (i = 0; i < ret; i++) {
            buf[i] = (char)wbuffer[i]; /* assume this was ASCII anyhow. */
        }
    }
}

/* vi: set ts=4 sw=4 expandtab: */
