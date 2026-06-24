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

#ifndef SDL_x11keyboard_h_
#define SDL_x11keyboard_h_

extern int X11_InitKeyboard(_THIS);
extern void X11_UpdateKeymap(_THIS, SDL_bool send_event);
extern void X11_QuitKeyboard(_THIS);
extern void X11_StartTextInput(_THIS);
extern void X11_StopTextInput(_THIS);
extern void X11_SetTextInputRect(_THIS, const SDL_Rect *rect);
extern SDL_bool X11_HasScreenKeyboardSupport(_THIS);
extern void X11_ShowScreenKeyboard(_THIS, SDL_Window *window);
extern void X11_HideScreenKeyboard(_THIS, SDL_Window *window);
extern SDL_bool X11_IsScreenKeyboardShown(_THIS, SDL_Window *window);
extern KeySym X11_KeyCodeToSym(_THIS, KeyCode, unsigned char group);

#endif /* SDL_x11keyboard_h_ */

/* vi: set ts=4 sw=4 expandtab: */
