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
#include "../SDL_internal.h"

#ifndef SDL_keyboard_c_h_
#define SDL_keyboard_c_h_

#include "SDL_keycode.h"
#include "SDL_events.h"

/* Initialize the keyboard subsystem */
extern int SDL_KeyboardInit(void);

/* Get the default keymap */
extern void SDL_GetDefaultKeymap(SDL_Keycode *keymap);

/* Get the default key code for a scancode */
extern SDL_Keycode SDL_GetDefaultKeyFromScancode(SDL_Scancode scancode);

/* Set the mapping of scancode to key codes */
extern void SDL_SetKeymap(int start, const SDL_Keycode *keys, int length, SDL_bool send_event);

/* Set a platform-dependent key name, overriding the default platform-agnostic
   name. Encoded as UTF-8. The string is not copied, thus the pointer given to
   this function must stay valid forever (or at least until the call to
   VideoQuit()). */
extern void SDL_SetScancodeName(SDL_Scancode scancode, const char *name);

/* Set the keyboard focus window */
extern void SDL_SetKeyboardFocus(SDL_Window *window);

/* Send a character from an on-screen keyboard as scancode and modifier key events,
   currently assuming ASCII characters on a US keyboard layout
 */
extern int SDL_SendKeyboardUnicodeKey(Uint32 ch);

/* Send a key from a virtual key source, like an on-screen keyboard */
extern int SDL_SendVirtualKeyboardKey(Uint8 state, SDL_Scancode scancode);

/* Send a keyboard key event */
extern int SDL_SendKeyboardKey(Uint8 state, SDL_Scancode scancode);
extern int SDL_SendKeyboardKeyAutoRelease(SDL_Scancode scancode);

/* This is for platforms that don't know the keymap but can report scancode and keycode directly.
   Most platforms should prefer to optionally call SDL_SetKeymap and then use SDL_SendKeyboardKey. */
extern int SDL_SendKeyboardKeyAndKeycode(Uint8 state, SDL_Scancode scancode, SDL_Keycode keycode);

/* Release all the autorelease keys */
extern void SDL_ReleaseAutoReleaseKeys(void);

/* Return true if any hardware key is pressed */
extern SDL_bool SDL_HardwareKeyboardKeyPressed(void);

/* Send keyboard text input */
extern int SDL_SendKeyboardText(const char *text);

/* Send editing text for selected range from start to end */
extern int SDL_SendEditingText(const char *text, int start, int length);

/* Shutdown the keyboard subsystem */
extern void SDL_KeyboardQuit(void);

/* Convert to UTF-8 */
extern char *SDL_UCS4ToUTF8(Uint32 ch, char *dst);

/* Toggle on or off pieces of the keyboard mod state. */
extern void SDL_ToggleModState(const SDL_Keymod modstate, const SDL_bool toggle);

#endif /* SDL_keyboard_c_h_ */

/* vi: set ts=4 sw=4 expandtab: */
