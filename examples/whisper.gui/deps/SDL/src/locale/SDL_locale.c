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
#include "SDL_syslocale.h"
#include "SDL_hints.h"

static SDL_Locale *build_locales_from_csv_string(char *csv)
{
    size_t num_locales = 1; /* at least one */
    size_t slen;
    size_t alloclen;
    char *ptr;
    SDL_Locale *loc;
    SDL_Locale *retval;

    if (!csv || !csv[0]) {
        return NULL; /* nothing to report */
    }

    for (ptr = csv; *ptr; ptr++) {
        if (*ptr == ',') {
            num_locales++;
        }
    }

    num_locales++; /* one more for terminator */

    slen = ((size_t)(ptr - csv)) + 1; /* SDL_strlen(csv) + 1 */
    alloclen = slen + (num_locales * sizeof(SDL_Locale));

    loc = retval = (SDL_Locale *)SDL_calloc(1, alloclen);
    if (!retval) {
        SDL_OutOfMemory();
        return NULL; /* oh well */
    }
    ptr = (char *)(retval + num_locales);
    SDL_strlcpy(ptr, csv, slen);

    while (SDL_TRUE) { /* parse out the string */
        while (*ptr == ' ') {
            ptr++; /* skip whitespace. */
        }

        if (*ptr == '\0') {
            break;
        }
        loc->language = ptr++;
        while (SDL_TRUE) {
            const char ch = *ptr;
            if (ch == '_') {
                *(ptr++) = '\0';
                loc->country = ptr;
            } else if (ch == ' ') {
                *(ptr++) = '\0'; /* trim ending whitespace and keep going. */
            } else if (ch == ',') {
                *(ptr++) = '\0';
                loc++;
                break;
            } else if (ch == '\0') {
                loc++;
                break;
            } else {
                ptr++; /* just keep going, still a valid string */
            }
        }
    }

    return retval;
}

SDL_Locale *SDL_GetPreferredLocales(void)
{
    char locbuf[128]; /* enough for 21 "xx_YY," language strings. */
    const char *hint = SDL_GetHint(SDL_HINT_PREFERRED_LOCALES);
    if (hint) {
        SDL_strlcpy(locbuf, hint, sizeof(locbuf));
    } else {
        SDL_zeroa(locbuf);
        SDL_SYS_GetPreferredLocales(locbuf, sizeof(locbuf));
    }
    return build_locales_from_csv_string(locbuf);
}

/* vi: set ts=4 sw=4 expandtab: */
