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

#ifndef SDL_x11dyn_h_
#define SDL_x11dyn_h_

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>

#ifdef SDL_VIDEO_DRIVER_X11_HAS_XKBKEYCODETOKEYSYM
#include <X11/XKBlib.h>
#endif

/* Apparently some X11 systems can't include this multiple times... */
#ifndef SDL_INCLUDED_XLIBINT_H
#define SDL_INCLUDED_XLIBINT_H 1
#include <X11/Xlibint.h>
#endif

#include <X11/Xproto.h>
#include <X11/extensions/Xext.h>

#ifndef NO_SHARED_MEMORY
#include <sys/ipc.h>
#include <sys/shm.h>
#include <X11/extensions/XShm.h>
#endif

#ifdef SDL_VIDEO_DRIVER_X11_XCURSOR
#include <X11/Xcursor/Xcursor.h>
#endif
#ifdef SDL_VIDEO_DRIVER_X11_XDBE
#include <X11/extensions/Xdbe.h>
#endif
#if defined(SDL_VIDEO_DRIVER_X11_XINPUT2) || defined(SDL_VIDEO_DRIVER_X11_XFIXES)
#include <X11/extensions/XInput2.h>
#endif
#ifdef SDL_VIDEO_DRIVER_X11_XFIXES
#include <X11/extensions/Xfixes.h>
#endif
#ifdef SDL_VIDEO_DRIVER_X11_XRANDR
#include <X11/extensions/Xrandr.h>
#endif
#ifdef SDL_VIDEO_DRIVER_X11_XSCRNSAVER
#include <X11/extensions/scrnsaver.h>
#endif
#ifdef SDL_VIDEO_DRIVER_X11_XSHAPE
#include <X11/extensions/shape.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* evil function signatures... */
typedef Bool (*SDL_X11_XESetWireToEventRetType)(Display *, XEvent *, xEvent *);
typedef int (*SDL_X11_XSynchronizeRetType)(Display *);
typedef Status (*SDL_X11_XESetEventToWireRetType)(Display *, XEvent *, xEvent *);

int SDL_X11_LoadSymbols(void);
void SDL_X11_UnloadSymbols(void);

/* Declare all the function pointers and wrappers... */
#define SDL_X11_SYM(rc, fn, params, args, ret) \
    typedef rc(*SDL_DYNX11FN_##fn) params;     \
    extern SDL_DYNX11FN_##fn X11_##fn;
#include "SDL_x11sym.h"

/* Annoying varargs entry point... */
#ifdef X_HAVE_UTF8_STRING
typedef XIC (*SDL_DYNX11FN_XCreateIC)(XIM, ...);
typedef char *(*SDL_DYNX11FN_XGetICValues)(XIC, ...);
extern SDL_DYNX11FN_XCreateIC X11_XCreateIC;
extern SDL_DYNX11FN_XGetICValues X11_XGetICValues;
#endif

/* These SDL_X11_HAVE_* flags are here whether you have dynamic X11 or not. */
#define SDL_X11_MODULE(modname) extern int SDL_X11_HAVE_##modname;
#include "SDL_x11sym.h"

#ifdef __cplusplus
}
#endif

#endif /* !defined SDL_x11dyn_h_ */
/* vi: set ts=4 sw=4 expandtab: */
