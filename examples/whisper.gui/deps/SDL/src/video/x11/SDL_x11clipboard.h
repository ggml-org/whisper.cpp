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

#ifndef SDL_x11clipboard_h_
#define SDL_x11clipboard_h_

enum ESDLX11ClipboardMimeType
{
    SDL_X11_CLIPBOARD_MIME_TYPE_STRING,
    SDL_X11_CLIPBOARD_MIME_TYPE_TEXT_PLAIN,
#ifdef X_HAVE_UTF8_STRING
    SDL_X11_CLIPBOARD_MIME_TYPE_TEXT_PLAIN_UTF8,
#endif
    SDL_X11_CLIPBOARD_MIME_TYPE_TEXT,
    SDL_X11_CLIPBOARD_MIME_TYPE_MAX
};

extern int X11_SetClipboardText(_THIS, const char *text);
extern char *X11_GetClipboardText(_THIS);
extern SDL_bool X11_HasClipboardText(_THIS);
extern int X11_SetPrimarySelectionText(_THIS, const char *text);
extern char *X11_GetPrimarySelectionText(_THIS);
extern SDL_bool X11_HasPrimarySelectionText(_THIS);
extern Atom X11_GetSDLCutBufferClipboardType(Display *display, enum ESDLX11ClipboardMimeType mime_type, Atom selection_type);
extern Atom X11_GetSDLCutBufferClipboardExternalFormat(Display *display, enum ESDLX11ClipboardMimeType mime_type);
extern Atom X11_GetSDLCutBufferClipboardInternalFormat(Display *display, enum ESDLX11ClipboardMimeType mime_type);

#endif /* SDL_x11clipboard_h_ */

/* vi: set ts=4 sw=4 expandtab: */
