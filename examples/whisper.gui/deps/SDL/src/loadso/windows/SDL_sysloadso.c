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

#ifdef SDL_LOADSO_WINDOWS

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* System dependent library loading routines                           */

#include "../../core/windows/SDL_windows.h"

#include "SDL_loadso.h"

void *SDL_LoadObject(const char *sofile)
{
    void *handle;
    LPTSTR tstr;

    if (!sofile) {
        SDL_InvalidParamError("sofile");
        return NULL;
    }
    tstr = WIN_UTF8ToString(sofile);
#ifdef __WINRT__
    /* WinRT only publicly supports LoadPackagedLibrary() for loading .dll
       files.  LoadLibrary() is a private API, and not available for apps
       (that can be published to MS' Windows Store.)
    */
    handle = (void *)LoadPackagedLibrary(tstr, 0);
#else
    handle = (void *)LoadLibrary(tstr);
#endif
    SDL_free(tstr);

    /* Generate an error message if all loads failed */
    if (!handle) {
        char errbuf[512];
        SDL_strlcpy(errbuf, "Failed loading ", SDL_arraysize(errbuf));
        SDL_strlcat(errbuf, sofile, SDL_arraysize(errbuf));
        WIN_SetError(errbuf);
    }
    return handle;
}

void *SDL_LoadFunction(void *handle, const char *name)
{
    void *symbol = (void *)GetProcAddress((HMODULE)handle, name);
    if (!symbol) {
        char errbuf[512];
        SDL_strlcpy(errbuf, "Failed loading ", SDL_arraysize(errbuf));
        SDL_strlcat(errbuf, name, SDL_arraysize(errbuf));
        WIN_SetError(errbuf);
    }
    return symbol;
}

void SDL_UnloadObject(void *handle)
{
    if (handle) {
        FreeLibrary((HMODULE)handle);
    }
}

#endif /* SDL_LOADSO_WINDOWS */

/* vi: set ts=4 sw=4 expandtab: */
