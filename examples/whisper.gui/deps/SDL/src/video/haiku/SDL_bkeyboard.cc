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

#ifdef SDL_VIDEO_DRIVER_HAIKU

#include <SupportDefs.h>
#include <support/UTF8.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "SDL_events.h"
#include "SDL_keycode.h"

#include "SDL_bkeyboard.h"


#define KEYMAP_SIZE 128


static SDL_Scancode keymap[KEYMAP_SIZE];
static int8 keystate[KEYMAP_SIZE];

void HAIKU_InitOSKeymap(void) {
        for ( uint i = 0; i < SDL_TABLESIZE(keymap); ++i ) {
            keymap[i] = SDL_SCANCODE_UNKNOWN;
        }

        for ( uint i = 0; i < KEYMAP_SIZE; ++i ) {
            keystate[i] = SDL_RELEASED;
        }

        keymap[0x01]        = SDL_GetScancodeFromKey(SDLK_ESCAPE);
        keymap[B_F1_KEY]    = SDL_GetScancodeFromKey(SDLK_F1);
        keymap[B_F2_KEY]    = SDL_GetScancodeFromKey(SDLK_F2);
        keymap[B_F3_KEY]    = SDL_GetScancodeFromKey(SDLK_F3);
        keymap[B_F4_KEY]    = SDL_GetScancodeFromKey(SDLK_F4);
        keymap[B_F5_KEY]    = SDL_GetScancodeFromKey(SDLK_F5);
        keymap[B_F6_KEY]    = SDL_GetScancodeFromKey(SDLK_F6);
        keymap[B_F7_KEY]    = SDL_GetScancodeFromKey(SDLK_F7);
        keymap[B_F8_KEY]    = SDL_GetScancodeFromKey(SDLK_F8);
        keymap[B_F9_KEY]    = SDL_GetScancodeFromKey(SDLK_F9);
        keymap[B_F10_KEY]    = SDL_GetScancodeFromKey(SDLK_F10);
        keymap[B_F11_KEY]    = SDL_GetScancodeFromKey(SDLK_F11);
        keymap[B_F12_KEY]    = SDL_GetScancodeFromKey(SDLK_F12);
        keymap[B_PRINT_KEY]    = SDL_GetScancodeFromKey(SDLK_PRINTSCREEN);
        keymap[B_SCROLL_KEY]    = SDL_GetScancodeFromKey(SDLK_SCROLLLOCK);
        keymap[B_PAUSE_KEY]    = SDL_GetScancodeFromKey(SDLK_PAUSE);
        keymap[0x11]        = SDL_GetScancodeFromKey(SDLK_BACKQUOTE);
        keymap[0x12]        = SDL_GetScancodeFromKey(SDLK_1);
        keymap[0x13]        = SDL_GetScancodeFromKey(SDLK_2);
        keymap[0x14]        = SDL_GetScancodeFromKey(SDLK_3);
        keymap[0x15]        = SDL_GetScancodeFromKey(SDLK_4);
        keymap[0x16]        = SDL_GetScancodeFromKey(SDLK_5);
        keymap[0x17]        = SDL_GetScancodeFromKey(SDLK_6);
        keymap[0x18]        = SDL_GetScancodeFromKey(SDLK_7);
        keymap[0x19]        = SDL_GetScancodeFromKey(SDLK_8);
        keymap[0x1a]        = SDL_GetScancodeFromKey(SDLK_9);
        keymap[0x1b]        = SDL_GetScancodeFromKey(SDLK_0);
        keymap[0x1c]        = SDL_GetScancodeFromKey(SDLK_MINUS);
        keymap[0x1d]        = SDL_GetScancodeFromKey(SDLK_EQUALS);
        keymap[0x1e]        = SDL_GetScancodeFromKey(SDLK_BACKSPACE);
        keymap[0x1f]        = SDL_GetScancodeFromKey(SDLK_INSERT);
        keymap[0x20]        = SDL_GetScancodeFromKey(SDLK_HOME);
        keymap[0x21]        = SDL_GetScancodeFromKey(SDLK_PAGEUP);
        keymap[0x22]        = SDL_GetScancodeFromKey(SDLK_NUMLOCKCLEAR);
        keymap[0x23]        = SDL_GetScancodeFromKey(SDLK_KP_DIVIDE);
        keymap[0x24]        = SDL_GetScancodeFromKey(SDLK_KP_MULTIPLY);
        keymap[0x25]        = SDL_GetScancodeFromKey(SDLK_KP_MINUS);
        keymap[0x26]        = SDL_GetScancodeFromKey(SDLK_TAB);
        keymap[0x27]        = SDL_GetScancodeFromKey(SDLK_q);
        keymap[0x28]        = SDL_GetScancodeFromKey(SDLK_w);
        keymap[0x29]        = SDL_GetScancodeFromKey(SDLK_e);
        keymap[0x2a]        = SDL_GetScancodeFromKey(SDLK_r);
        keymap[0x2b]        = SDL_GetScancodeFromKey(SDLK_t);
        keymap[0x2c]        = SDL_GetScancodeFromKey(SDLK_y);
        keymap[0x2d]        = SDL_GetScancodeFromKey(SDLK_u);
        keymap[0x2e]        = SDL_GetScancodeFromKey(SDLK_i);
        keymap[0x2f]        = SDL_GetScancodeFromKey(SDLK_o);
        keymap[0x30]        = SDL_GetScancodeFromKey(SDLK_p);
        keymap[0x31]        = SDL_GetScancodeFromKey(SDLK_LEFTBRACKET);
        keymap[0x32]        = SDL_GetScancodeFromKey(SDLK_RIGHTBRACKET);
        keymap[0x33]        = SDL_GetScancodeFromKey(SDLK_BACKSLASH);
        keymap[0x34]        = SDL_GetScancodeFromKey(SDLK_DELETE);
        keymap[0x35]        = SDL_GetScancodeFromKey(SDLK_END);
        keymap[0x36]        = SDL_GetScancodeFromKey(SDLK_PAGEDOWN);
        keymap[0x37]        = SDL_GetScancodeFromKey(SDLK_KP_7);
        keymap[0x38]        = SDL_GetScancodeFromKey(SDLK_KP_8);
        keymap[0x39]        = SDL_GetScancodeFromKey(SDLK_KP_9);
        keymap[0x3a]        = SDL_GetScancodeFromKey(SDLK_KP_PLUS);
        keymap[0x3b]        = SDL_GetScancodeFromKey(SDLK_CAPSLOCK);
        keymap[0x3c]        = SDL_GetScancodeFromKey(SDLK_a);
        keymap[0x3d]        = SDL_GetScancodeFromKey(SDLK_s);
        keymap[0x3e]        = SDL_GetScancodeFromKey(SDLK_d);
        keymap[0x3f]        = SDL_GetScancodeFromKey(SDLK_f);
        keymap[0x40]        = SDL_GetScancodeFromKey(SDLK_g);
        keymap[0x41]        = SDL_GetScancodeFromKey(SDLK_h);
        keymap[0x42]        = SDL_GetScancodeFromKey(SDLK_j);
        keymap[0x43]        = SDL_GetScancodeFromKey(SDLK_k);
        keymap[0x44]        = SDL_GetScancodeFromKey(SDLK_l);
        keymap[0x45]        = SDL_GetScancodeFromKey(SDLK_SEMICOLON);
        keymap[0x46]        = SDL_GetScancodeFromKey(SDLK_QUOTE);
        keymap[0x47]        = SDL_GetScancodeFromKey(SDLK_RETURN);
        keymap[0x48]        = SDL_GetScancodeFromKey(SDLK_KP_4);
        keymap[0x49]        = SDL_GetScancodeFromKey(SDLK_KP_5);
        keymap[0x4a]        = SDL_GetScancodeFromKey(SDLK_KP_6);
        keymap[0x4b]        = SDL_GetScancodeFromKey(SDLK_LSHIFT);
        keymap[0x4c]        = SDL_GetScancodeFromKey(SDLK_z);
        keymap[0x4d]        = SDL_GetScancodeFromKey(SDLK_x);
        keymap[0x4e]        = SDL_GetScancodeFromKey(SDLK_c);
        keymap[0x4f]        = SDL_GetScancodeFromKey(SDLK_v);
        keymap[0x50]        = SDL_GetScancodeFromKey(SDLK_b);
        keymap[0x51]        = SDL_GetScancodeFromKey(SDLK_n);
        keymap[0x52]        = SDL_GetScancodeFromKey(SDLK_m);
        keymap[0x53]        = SDL_GetScancodeFromKey(SDLK_COMMA);
        keymap[0x54]        = SDL_GetScancodeFromKey(SDLK_PERIOD);
        keymap[0x55]        = SDL_GetScancodeFromKey(SDLK_SLASH);
        keymap[0x56]        = SDL_GetScancodeFromKey(SDLK_RSHIFT);
        keymap[0x57]        = SDL_GetScancodeFromKey(SDLK_UP);
        keymap[0x58]        = SDL_GetScancodeFromKey(SDLK_KP_1);
        keymap[0x59]        = SDL_GetScancodeFromKey(SDLK_KP_2);
        keymap[0x5a]        = SDL_GetScancodeFromKey(SDLK_KP_3);
        keymap[0x5b]        = SDL_GetScancodeFromKey(SDLK_KP_ENTER);
        keymap[0x5c]        = SDL_GetScancodeFromKey(SDLK_LCTRL);
        keymap[0x5d]        = SDL_GetScancodeFromKey(SDLK_LALT);
        keymap[0x5e]        = SDL_GetScancodeFromKey(SDLK_SPACE);
        keymap[0x5f]        = SDL_GetScancodeFromKey(SDLK_RALT);
        keymap[0x60]        = SDL_GetScancodeFromKey(SDLK_RCTRL);
        keymap[0x61]        = SDL_GetScancodeFromKey(SDLK_LEFT);
        keymap[0x62]        = SDL_GetScancodeFromKey(SDLK_DOWN);
        keymap[0x63]        = SDL_GetScancodeFromKey(SDLK_RIGHT);
        keymap[0x64]        = SDL_GetScancodeFromKey(SDLK_KP_0);
        keymap[0x65]        = SDL_GetScancodeFromKey(SDLK_KP_PERIOD);
        keymap[0x66]        = SDL_GetScancodeFromKey(SDLK_LGUI);
        keymap[0x67]        = SDL_GetScancodeFromKey(SDLK_RGUI);
        keymap[0x68]        = SDL_GetScancodeFromKey(SDLK_MENU);
        keymap[0x69]        = SDL_GetScancodeFromKey(SDLK_2); /* SDLK_EURO */
        keymap[0x6a]        = SDL_GetScancodeFromKey(SDLK_KP_EQUALS);
        keymap[0x6b]        = SDL_GetScancodeFromKey(SDLK_POWER);
}

SDL_Scancode HAIKU_GetScancodeFromBeKey(int32 bkey) {
    if (bkey > 0 && bkey < (int32)SDL_TABLESIZE(keymap)) {
        return keymap[bkey];
    } else {
        return SDL_SCANCODE_UNKNOWN;
    }
}

int8 HAIKU_GetKeyState(int32 bkey) {
    if (bkey > 0 && bkey < KEYMAP_SIZE) {
        return keystate[bkey];
    } else {
        return SDL_RELEASED;
    }
}

void HAIKU_SetKeyState(int32 bkey, int8 state) {
    if (bkey > 0 && bkey < KEYMAP_SIZE) {
        keystate[bkey] = state;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* SDL_VIDEO_DRIVER_HAIKU */

/* vi: set ts=4 sw=4 expandtab: */
