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

#ifndef SDL_waylandkeyboard_h_
#define SDL_waylandkeyboard_h_

typedef struct SDL_WaylandTextInput
{
    struct zwp_text_input_v3 *text_input;
    SDL_Rect cursor_rect;
    SDL_bool has_preedit;
    SDL_bool is_enabled;
} SDL_WaylandTextInput;

extern int Wayland_InitKeyboard(_THIS);
extern void Wayland_QuitKeyboard(_THIS);
extern void Wayland_StartTextInput(_THIS);
extern void Wayland_StopTextInput(_THIS);
extern void Wayland_SetTextInputRect(_THIS, const SDL_Rect *rect);
extern SDL_bool Wayland_HasScreenKeyboardSupport(_THIS);

#endif /* SDL_waylandkeyboard_h_ */

/* vi: set ts=4 sw=4 expandtab: */
