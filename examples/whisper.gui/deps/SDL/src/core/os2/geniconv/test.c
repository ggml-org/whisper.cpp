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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "geniconv.h"

int main(void)
{
    char    acBuf[128];
    char   *inbuf = "\xf4\xc5\xd3\xd4\x20\x2d\x20\xd0\xd2\xcf\xd7\xc5\xd2\xcb\xc1"; /* KOI8-R encoding of "Тест - проверка" */
    size_t  inbytesleft = strlen(inbuf);
    char   *outbuf = acBuf;
    size_t  outbytesleft = sizeof(acBuf);
    iconv_t ic;

    /* KOI8 -> system cp */
    ic = iconv_open("", "KOI8-R");
    if (ic == (iconv_t)(-1)) {
        puts("iconv_open() fail");
        return 1;
    }

    iconv(ic, &inbuf, &inbytesleft, &outbuf, &outbytesleft);
    printf("KOI8-R to system cp: %s\n", acBuf);

    iconv_close(ic);

    /* System cp -> UTF-8 -> system cp: */

    /* System cp -> UTF-8 by StrUTF8New() */
    inbuf = StrUTF8New(1, acBuf, strlen(acBuf));

    /* UTF-8 -> system cp. by StrUTF8() */
    if (StrUTF8(0, acBuf, sizeof(acBuf), inbuf, strlen(inbuf)) == -1) {
        puts("StrUTF8() failed");
    } else {
        printf("system cp. -> UTF-8 -> system cp.: %s\n", acBuf);
    }

    free(inbuf);

    /* Unload used DLL */
    iconv_clean();

    puts("Done.");
    return 0;
}

/* vi: set ts=4 sw=4 expandtab: */
