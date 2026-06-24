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

#ifdef SDL_FILESYSTEM_N3DS

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* System dependent filesystem routines                                */

#include <3ds.h>
#include <dirent.h>
#include <errno.h>

#include "SDL_error.h"
#include "SDL_filesystem.h"

SDL_FORCE_INLINE char *MakePrefPath(const char *app);
SDL_FORCE_INLINE int CreatePrefPathDir(const char *pref);

char *SDL_GetBasePath(void)
{
    char *base_path = SDL_strdup("romfs:/");
    return base_path;
}

char *SDL_GetPrefPath(const char *org, const char *app)
{
    char *pref_path = NULL;
    if (!app) {
        SDL_InvalidParamError("app");
        return NULL;
    }

    pref_path = MakePrefPath(app);
    if (!pref_path) {
        return NULL;
    }

    if (CreatePrefPathDir(pref_path) < 0) {
        SDL_free(pref_path);
        return NULL;
    }

    return pref_path;
}

SDL_FORCE_INLINE char *
MakePrefPath(const char *app)
{
    char *pref_path;
    if (SDL_asprintf(&pref_path, "sdmc:/3ds/%s/", app) < 0) {
        SDL_OutOfMemory();
        return NULL;
    }
    return pref_path;
}

SDL_FORCE_INLINE int
CreatePrefPathDir(const char *pref)
{
    int result = mkdir(pref, 0666);

    if (result == -1 && errno != EEXIST) {
        return SDL_SetError("Failed to create '%s' (%s)", pref, strerror(errno));
    }
    return 0;
}

#endif /* SDL_FILESYSTEM_N3DS */

/* vi: set sts=4 ts=4 sw=4 expandtab: */
