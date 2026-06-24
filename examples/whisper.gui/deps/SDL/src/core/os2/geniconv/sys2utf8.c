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

#include "geniconv.h"

#ifndef GENICONV_STANDALONE
#include "../../../SDL_internal.h"
#else
#include <stdlib.h>
#define SDL_malloc malloc
#define SDL_realloc realloc
#define SDL_free free
#endif

int StrUTF8(int to_utf8, char *dst, int c_dst, char *src, int c_src)
{
    size_t  rc;
    char   *dststart = dst;
    iconv_t cd;
    char   *tocp, *fromcp;
    int     err = 0;

    if (c_dst < 4) {
        return -1;
    }

    if (to_utf8) {
        tocp   = "UTF-8";
        fromcp = "";
    } else {
        tocp   = "";
        fromcp = "UTF-8";
    }

    cd = iconv_open(tocp, fromcp);
    if (cd == (iconv_t)-1) {
        return -1;
    }

    while (c_src > 0) {
        rc = iconv(cd, &src, (size_t *)&c_src, &dst, (size_t *)&c_dst);
        if (rc == (size_t)-1) {
            if (errno == EILSEQ) {
                /* Try to skip invalid character */
                src++;
                c_src--;
                continue;
            }

            err = 1;
            break;
        }
    }

    iconv_close(cd);

    /* Write trailing ZERO (1 byte for UTF-8, 2 bytes for the system cp) */
    if (to_utf8) {
        if (c_dst < 1) {
            dst--;
            err = 1;    /* The destination buffer overflow */
        }
        *dst = '\0';
    } else {
        if (c_dst < 2) {
            dst -= (c_dst == 0) ? 2 : 1;
            err = 1;    /* The destination buffer overflow */
        }
        *((short *)dst) = '\0';
    }

    return (err) ? -1 : (dst - dststart);
}

char *StrUTF8New(int to_utf8, char *str, int c_str)
{
    int   c_newstr = (((c_str > 4) ? c_str : 4) + 1) * 2;
    char *  newstr = (char *) SDL_malloc(c_newstr);

    if (!newstr) {
        return NULL;
    }

    c_newstr = StrUTF8(to_utf8, newstr, c_newstr, str, c_str);
    if (c_newstr != -1) {
        str = (char *) SDL_realloc(newstr, c_newstr + ((to_utf8) ? 1 : sizeof(short)));
        if (str) {
            return str;
        }
    }

    SDL_free(newstr);
    return NULL;
}

void StrUTF8Free(char *str)
{
    SDL_free(str);
}

/* vi: set ts=4 sw=4 expandtab: */
