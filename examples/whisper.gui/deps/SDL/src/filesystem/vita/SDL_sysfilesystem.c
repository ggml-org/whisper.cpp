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

#ifdef SDL_FILESYSTEM_VITA

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* System dependent filesystem routines                                */

#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <psp2/io/stat.h>
#include <sys/types.h>
#include <limits.h>
#include <fcntl.h>

#include "SDL_error.h"
#include "SDL_stdinc.h"
#include "SDL_filesystem.h"
#include "SDL_rwops.h"

char *SDL_GetBasePath(void)
{
    const char *basepath = "app0:/";
    char *retval = SDL_strdup(basepath);
    return retval;
}

char *SDL_GetPrefPath(const char *org, const char *app)
{
    const char *envr = "ux0:/data/";
    char *retval = NULL;
    char *ptr = NULL;
    size_t len = 0;

    if (!app) {
        SDL_InvalidParamError("app");
        return NULL;
    }
    if (!org) {
        org = "";
    }

    len = SDL_strlen(envr);

    len += SDL_strlen(org) + SDL_strlen(app) + 3;
    retval = (char *)SDL_malloc(len);
    if (!retval) {
        SDL_OutOfMemory();
        return NULL;
    }

    if (*org) {
        SDL_snprintf(retval, len, "%s%s/%s/", envr, org, app);
    } else {
        SDL_snprintf(retval, len, "%s%s/", envr, app);
    }

    for (ptr = retval + 1; *ptr; ptr++) {
        if (*ptr == '/') {
            *ptr = '\0';
            sceIoMkdir(retval, 0777);
            *ptr = '/';
        }
    }
    sceIoMkdir(retval, 0777);

    return retval;
}

#endif /* SDL_FILESYSTEM_VITA */

/* vi: set ts=4 sw=4 expandtab: */
