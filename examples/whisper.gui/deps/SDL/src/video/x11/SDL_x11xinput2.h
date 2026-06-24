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

#ifndef SDL_x11xinput2_h_
#define SDL_x11xinput2_h_

#ifndef SDL_VIDEO_DRIVER_X11_SUPPORTS_GENERIC_EVENTS
/* Define XGenericEventCookie as forward declaration when
 *xinput2 is not available in order to compile */
struct XGenericEventCookie;
typedef struct XGenericEventCookie XGenericEventCookie;
#endif

extern void X11_InitXinput2(_THIS);
extern void X11_InitXinput2Multitouch(_THIS);
extern int X11_HandleXinput2Event(SDL_VideoData *videodata, XGenericEventCookie *cookie);
extern int X11_Xinput2IsInitialized(void);
extern int X11_Xinput2IsMultitouchSupported(void);
extern void X11_Xinput2SelectTouch(_THIS, SDL_Window *window);
extern void X11_Xinput2GrabTouch(_THIS, SDL_Window *window);
extern void X11_Xinput2UngrabTouch(_THIS, SDL_Window *window);

#endif /* SDL_x11xinput2_h_ */

/* vi: set ts=4 sw=4 expandtab: */
