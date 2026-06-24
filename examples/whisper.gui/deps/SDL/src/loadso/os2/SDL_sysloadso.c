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

#ifdef SDL_LOADSO_OS2

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* System dependent library loading routines                           */

#include "SDL_loadso.h"
#include "../../core/os2/SDL_os2.h"

#define INCL_DOSMODULEMGR
#define INCL_DOSERRORS
#include <os2.h>

void *SDL_LoadObject(const char *sofile)
{
    ULONG   ulRC;
    HMODULE hModule;
    CHAR    acError[256];
    PSZ     pszModName;

    if (!sofile) {
        SDL_InvalidParamError("sofile");
        return NULL;
    }

    pszModName = OS2_UTF8ToSys(sofile);
    ulRC = DosLoadModule(acError, sizeof(acError), pszModName, &hModule);

    if (ulRC != NO_ERROR && !SDL_strrchr(pszModName, '\\') && !SDL_strrchr(pszModName, '/')) {
        /* strip .dll extension and retry only if name has no path. */
        size_t len = SDL_strlen(pszModName);
        if (len > 4 && SDL_strcasecmp(&pszModName[len - 4], ".dll") == 0) {
            pszModName[len - 4] = '\0';
            ulRC = DosLoadModule(acError, sizeof(acError), pszModName, &hModule);
        }
    }
    if (ulRC != NO_ERROR) {
        SDL_SetError("Failed loading %s: %s (E%u)", sofile, acError, ulRC);
        hModule = NULLHANDLE;
    }
    SDL_free(pszModName);

    return (void *)hModule;
}

void *SDL_LoadFunction(void *handle, const char *name)
{
    ULONG   ulRC;
    PFN     pFN;

    ulRC = DosQueryProcAddr((HMODULE)handle, 0, name, &pFN);
    if (ulRC != NO_ERROR) {
        /* retry with an underscore prepended, e.g. for gcc-built dlls. */
        SDL_bool isstack;
        size_t len = SDL_strlen(name) + 1;
        char *_name = SDL_small_alloc(char, len + 1, &isstack);
        _name[0] = '_';
        SDL_memcpy(&_name[1], name, len);
        ulRC = DosQueryProcAddr((HMODULE)handle, 0, _name, &pFN);
        SDL_small_free(_name, isstack);
    }
    if (ulRC != NO_ERROR) {
        SDL_SetError("Failed loading procedure %s (E%u)", name, ulRC);
        return NULL;
    }

    return (void *)pFN;
}

void SDL_UnloadObject(void *handle)
{
    if (handle) {
        DosFreeModule((HMODULE)handle);
    }
}

#endif /* SDL_LOADSO_OS2 */

/* vi: set ts=4 sw=4 expandtab: */
