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

#ifdef SDL_VIDEO_DRIVER_DIRECTFB

#include "SDL_DirectFB_video.h"
#include "SDL_DirectFB_dyn.h"

#ifdef SDL_VIDEO_DRIVER_DIRECTFB_DYNAMIC
#include "SDL_name.h"
#include "SDL_loadso.h"

#define DFB_SYM(ret, name, args, al, func) ret (*name) args;
static struct _SDL_DirectFB_Symbols
{
    DFB_SYMS
    const unsigned int *directfb_major_version;
    const unsigned int *directfb_minor_version;
    const unsigned int *directfb_micro_version;
} SDL_DirectFB_Symbols;
#undef DFB_SYM

#define DFB_SYM(ret, name, args, al, func) ret name args { func SDL_DirectFB_Symbols.name al  ; }
DFB_SYMS
#undef DFB_SYM

static void *handle = NULL;

int SDL_DirectFB_LoadLibrary(void)
{
    int retval = 0;

    if (handle == NULL) {
        handle = SDL_LoadObject(SDL_VIDEO_DRIVER_DIRECTFB_DYNAMIC);
        if (handle != NULL) {
            retval = 1;
#define DFB_SYM(ret, name, args, al, func) if (!(SDL_DirectFB_Symbols.name = SDL_LoadFunction(handle, # name))) retval = 0;
            DFB_SYMS
#undef DFB_SYM
            if (!
                    (SDL_DirectFB_Symbols.directfb_major_version =
                     SDL_LoadFunction(handle, "directfb_major_version")))
                retval = 0;
            if (!
                (SDL_DirectFB_Symbols.directfb_minor_version =
                 SDL_LoadFunction(handle, "directfb_minor_version")))
                retval = 0;
            if (!
                (SDL_DirectFB_Symbols.directfb_micro_version =
                 SDL_LoadFunction(handle, "directfb_micro_version")))
                retval = 0;
        }
    }
    if (retval) {
        const char *stemp = DirectFBCheckVersion(DIRECTFB_MAJOR_VERSION,
                                                 DIRECTFB_MINOR_VERSION,
                                                 DIRECTFB_MICRO_VERSION);
        /* Version Check */
        if (stemp != NULL) {
            fprintf(stderr,
                    "DirectFB Lib: Version mismatch. Compiled: %d.%d.%d Library %d.%d.%d\n",
                    DIRECTFB_MAJOR_VERSION, DIRECTFB_MINOR_VERSION,
                    DIRECTFB_MICRO_VERSION,
                    *SDL_DirectFB_Symbols.directfb_major_version,
                    *SDL_DirectFB_Symbols.directfb_minor_version,
                    *SDL_DirectFB_Symbols.directfb_micro_version);
            retval = 0;
        }
    }
    if (!retval)
        SDL_DirectFB_UnLoadLibrary();
    return retval;
}

void SDL_DirectFB_UnLoadLibrary(void)
{
    if (handle) {
        SDL_UnloadObject(handle);
        handle = NULL;
    }
}

#else

int SDL_DirectFB_LoadLibrary(void)
{
    return 1;
}

void SDL_DirectFB_UnLoadLibrary(void)
{
}

#endif

#endif /* SDL_VIDEO_DRIVER_DIRECTFB */
