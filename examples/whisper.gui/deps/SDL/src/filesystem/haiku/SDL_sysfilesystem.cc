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

#ifdef SDL_FILESYSTEM_HAIKU

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* System dependent filesystem routines                                */

#include <kernel/image.h>
#include <storage/Directory.h>
#include <storage/Entry.h>
#include <storage/Path.h>

#include "SDL_error.h"
#include "SDL_stdinc.h"
#include "SDL_filesystem.h"

char *SDL_GetBasePath(void)
{
    image_info info;
    int32 cookie = 0;

    while (get_next_image_info(0, &cookie, &info) == B_OK) {
        if (info.type == B_APP_IMAGE) {
            break;
        }
    }

    BEntry entry(info.name, true);
    BPath path;
    status_t rc = entry.GetPath(&path);  /* (path) now has binary's path. */
    SDL_assert(rc == B_OK);
    rc = path.GetParent(&path); /* chop filename, keep directory. */
    SDL_assert(rc == B_OK);
    const char *str = path.Path();
    SDL_assert(str != NULL);

    const size_t len = SDL_strlen(str);
    char *retval = (char *) SDL_malloc(len + 2);
    if (!retval) {
        SDL_OutOfMemory();
        return NULL;
    }

    SDL_memcpy(retval, str, len);
    retval[len] = '/';
    retval[len+1] = '\0';
    return retval;
}


char *SDL_GetPrefPath(const char *org, const char *app)
{
    // !!! FIXME: is there a better way to do this?
    const char *home = SDL_getenv("HOME");
    const char *append = "/config/settings/";
    size_t len = SDL_strlen(home);

    if (!app) {
        SDL_InvalidParamError("app");
        return NULL;
    }
    if (!org) {
        org = "";
    }

    if (!len || (home[len - 1] == '/')) {
        ++append; // home empty or ends with separator, skip the one from append
    }
    len += SDL_strlen(append) + SDL_strlen(org) + SDL_strlen(app) + 3;
    char *retval = (char *) SDL_malloc(len);
    if (!retval) {
        SDL_OutOfMemory();
    } else {
        if (*org) {
            SDL_snprintf(retval, len, "%s%s%s/%s/", home, append, org, app);
        } else {
            SDL_snprintf(retval, len, "%s%s%s/", home, append, app);
        }
        create_directory(retval, 0700);  // Haiku api: creates missing dirs
    }

    return retval;
}

#endif /* SDL_FILESYSTEM_HAIKU */

/* vi: set ts=4 sw=4 expandtab: */
