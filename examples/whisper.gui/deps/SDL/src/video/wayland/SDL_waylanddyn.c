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

#ifdef SDL_VIDEO_DRIVER_WAYLAND

#define DEBUG_DYNAMIC_WAYLAND 0

#include "SDL_waylanddyn.h"

#ifdef SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC

#include "SDL_name.h"
#include "SDL_loadso.h"

typedef struct
{
    void *lib;
    const char *libname;
} waylanddynlib;

static waylanddynlib waylandlibs[] = {
    { NULL, SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC },
#ifdef SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_EGL
    { NULL, SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_EGL },
#endif
#ifdef SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_CURSOR
    { NULL, SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_CURSOR },
#endif
#ifdef SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_XKBCOMMON
    { NULL, SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_XKBCOMMON },
#endif
#ifdef SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_LIBDECOR
    { NULL, SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_LIBDECOR },
#endif
    { NULL, NULL }
};

static void *WAYLAND_GetSym(const char *fnname, int *pHasModule, SDL_bool required)
{
    void *fn = NULL;
    waylanddynlib *dynlib;
    for (dynlib = waylandlibs; dynlib->libname; dynlib++) {
        if (dynlib->lib) {
            fn = SDL_LoadFunction(dynlib->lib, fnname);
            if (fn) {
                break;
            }
        }
    }

#if DEBUG_DYNAMIC_WAYLAND
    if (fn) {
        SDL_Log("WAYLAND: Found '%s' in %s (%p)\n", fnname, dynlib->libname, fn);
    } else {
        SDL_Log("WAYLAND: Symbol '%s' NOT FOUND!\n", fnname);
    }
#endif

    if (!fn && required) {
        *pHasModule = 0; /* kill this module. */
    }

    return fn;
}

#else

#include <wayland-egl.h>

#endif /* SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC */

/* Define all the function pointers and wrappers... */
#define SDL_WAYLAND_MODULE(modname)         int SDL_WAYLAND_HAVE_##modname = 0;
#define SDL_WAYLAND_SYM(rc, fn, params)     SDL_DYNWAYLANDFN_##fn WAYLAND_##fn = NULL;
#define SDL_WAYLAND_SYM_OPT(rc, fn, params) SDL_DYNWAYLANDFN_##fn WAYLAND_##fn = NULL;
#define SDL_WAYLAND_INTERFACE(iface)        const struct wl_interface *WAYLAND_##iface = NULL;
#include "SDL_waylandsym.h"

static int wayland_load_refcount = 0;

void SDL_WAYLAND_UnloadSymbols(void)
{
    /* Don't actually unload if more than one module is using the libs... */
    if (wayland_load_refcount > 0) {
        if (--wayland_load_refcount == 0) {
#ifdef SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC
            int i;
#endif

            /* set all the function pointers to NULL. */
#define SDL_WAYLAND_MODULE(modname)         SDL_WAYLAND_HAVE_##modname = 0;
#define SDL_WAYLAND_SYM(rc, fn, params)     WAYLAND_##fn = NULL;
#define SDL_WAYLAND_SYM_OPT(rc, fn, params) WAYLAND_##fn = NULL;
#define SDL_WAYLAND_INTERFACE(iface)        WAYLAND_##iface = NULL;
#include "SDL_waylandsym.h"

#ifdef SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC
            for (i = 0; i < SDL_TABLESIZE(waylandlibs); i++) {
                if (waylandlibs[i].lib) {
                    SDL_UnloadObject(waylandlibs[i].lib);
                    waylandlibs[i].lib = NULL;
                }
            }
#endif
        }
    }
}

/* returns non-zero if all needed symbols were loaded. */
int SDL_WAYLAND_LoadSymbols(void)
{
    int rc = 1; /* always succeed if not using Dynamic WAYLAND stuff. */

    /* deal with multiple modules (dga, wayland, etc) needing these symbols... */
    if (wayland_load_refcount++ == 0) {
#ifdef SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC
        int i;
        int *thismod = NULL;
        for (i = 0; i < SDL_TABLESIZE(waylandlibs); i++) {
            if (waylandlibs[i].libname) {
                waylandlibs[i].lib = SDL_LoadObject(waylandlibs[i].libname);
            }
        }

#define SDL_WAYLAND_MODULE(modname) SDL_WAYLAND_HAVE_##modname = 1; /* default yes */
#include "SDL_waylandsym.h"

#define SDL_WAYLAND_MODULE(modname)         thismod = &SDL_WAYLAND_HAVE_##modname;
#define SDL_WAYLAND_SYM(rc, fn, params)     WAYLAND_##fn = (SDL_DYNWAYLANDFN_##fn)WAYLAND_GetSym(#fn, thismod, SDL_TRUE);
#define SDL_WAYLAND_SYM_OPT(rc, fn, params) WAYLAND_##fn = (SDL_DYNWAYLANDFN_##fn)WAYLAND_GetSym(#fn, thismod, SDL_FALSE);
#define SDL_WAYLAND_INTERFACE(iface)        WAYLAND_##iface = (struct wl_interface *)WAYLAND_GetSym(#iface, thismod, SDL_TRUE);
#include "SDL_waylandsym.h"

        if (SDL_WAYLAND_HAVE_WAYLAND_CLIENT) {
            /* all required symbols loaded. */
            SDL_ClearError();
        } else {
            /* in case something got loaded... */
            SDL_WAYLAND_UnloadSymbols();
            rc = 0;
        }

#else /* no dynamic WAYLAND */

#define SDL_WAYLAND_MODULE(modname)         SDL_WAYLAND_HAVE_##modname = 1; /* default yes */
#define SDL_WAYLAND_SYM(rc, fn, params)     WAYLAND_##fn = fn;
#define SDL_WAYLAND_SYM_OPT(rc, fn, params) WAYLAND_##fn = fn;
#define SDL_WAYLAND_INTERFACE(iface)        WAYLAND_##iface = &iface;
#include "SDL_waylandsym.h"

#endif
    }

    return rc;
}

#endif /* SDL_VIDEO_DRIVER_WAYLAND */

/* vi: set ts=4 sw=4 expandtab: */
