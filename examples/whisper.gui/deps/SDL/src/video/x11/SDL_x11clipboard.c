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

#ifdef SDL_VIDEO_DRIVER_X11

#include <limits.h> /* For INT_MAX */

#include "SDL_events.h"
#include "SDL_x11video.h"
#include "SDL_timer.h"
#include "SDL_x11clipboard.h"

/* Get any application owned window handle for clipboard association */
static Window GetWindow(_THIS)
{
    SDL_VideoData *data = (SDL_VideoData *)_this->driverdata;

    /* We create an unmapped window that exists just to manage the clipboard,
       since X11 selection data is tied to a specific window and dies with it.
       We create the window on demand, so apps that don't use the clipboard
       don't have to keep an unnecessary resource around. */
    if (data->clipboard_window == None) {
        Display *dpy = data->display;
        Window parent = RootWindow(dpy, DefaultScreen(dpy));
        XSetWindowAttributes xattr;
        data->clipboard_window = X11_XCreateWindow(dpy, parent, -10, -10, 1, 1, 0,
                                                   CopyFromParent, InputOnly,
                                                   CopyFromParent, 0, &xattr);
        X11_XFlush(data->display);
    }

    return data->clipboard_window;
}

/* We use our own cut-buffer for intermediate storage instead of
   XA_CUT_BUFFER0 because their use isn't really defined for holding UTF8. */
Atom X11_GetSDLCutBufferClipboardType(Display *display, enum ESDLX11ClipboardMimeType mime_type,
                                      Atom selection_type)
{
    switch (mime_type) {
    case SDL_X11_CLIPBOARD_MIME_TYPE_STRING:
    case SDL_X11_CLIPBOARD_MIME_TYPE_TEXT_PLAIN:
#ifdef X_HAVE_UTF8_STRING
    case SDL_X11_CLIPBOARD_MIME_TYPE_TEXT_PLAIN_UTF8:
#endif
    case SDL_X11_CLIPBOARD_MIME_TYPE_TEXT:
        return X11_XInternAtom(display, selection_type == XA_PRIMARY ? "SDL_CUTBUFFER_PRIMARY_SELECTION" : "SDL_CUTBUFFER",
                               False);
    default:
        SDL_SetError("Can't find mime_type.");
        return XA_STRING;
    }
}

Atom X11_GetSDLCutBufferClipboardExternalFormat(Display *display, enum ESDLX11ClipboardMimeType mime_type)
{
    switch (mime_type) {
    case SDL_X11_CLIPBOARD_MIME_TYPE_STRING:
/* If you don't support UTF-8, you might use XA_STRING here */
#ifdef X_HAVE_UTF8_STRING
        return X11_XInternAtom(display, "UTF8_STRING", False);
#else
        return XA_STRING;
#endif
    case SDL_X11_CLIPBOARD_MIME_TYPE_TEXT_PLAIN:
        return X11_XInternAtom(display, "text/plain", False);
#ifdef X_HAVE_UTF8_STRING
    case SDL_X11_CLIPBOARD_MIME_TYPE_TEXT_PLAIN_UTF8:
        return X11_XInternAtom(display, "text/plain;charset=utf-8", False);
#endif
    case SDL_X11_CLIPBOARD_MIME_TYPE_TEXT:
        return X11_XInternAtom(display, "TEXT", False);
    default:
        SDL_SetError("Can't find mime_type.");
        return XA_STRING;
    }
}
Atom X11_GetSDLCutBufferClipboardInternalFormat(Display *display, enum ESDLX11ClipboardMimeType mime_type)
{
    switch (mime_type) {
    case SDL_X11_CLIPBOARD_MIME_TYPE_STRING:
    case SDL_X11_CLIPBOARD_MIME_TYPE_TEXT_PLAIN:
#ifdef X_HAVE_UTF8_STRING
    case SDL_X11_CLIPBOARD_MIME_TYPE_TEXT_PLAIN_UTF8:
#endif
    case SDL_X11_CLIPBOARD_MIME_TYPE_TEXT:
/* If you don't support UTF-8, you might use XA_STRING here */
#ifdef X_HAVE_UTF8_STRING
        return X11_XInternAtom(display, "UTF8_STRING", False);
#else
        return XA_STRING;
#endif
    default:
        SDL_SetError("Can't find mime_type.");
        return XA_STRING;
    }
}

static int SetSelectionText(_THIS, const char *text, Atom selection_type)
{
    Display *display = ((SDL_VideoData *)_this->driverdata)->display;
    Window window;

    /* Get the SDL window that will own the selection */
    window = GetWindow(_this);
    if (window == None) {
        return SDL_SetError("Couldn't find a window to own the selection");
    }

    /* Save the selection on the root window */
    X11_XChangeProperty(display, DefaultRootWindow(display),
                        X11_GetSDLCutBufferClipboardType(display, SDL_X11_CLIPBOARD_MIME_TYPE_STRING, selection_type),
                        X11_GetSDLCutBufferClipboardInternalFormat(display, SDL_X11_CLIPBOARD_MIME_TYPE_STRING), 8, PropModeReplace,
                        (const unsigned char *)text, SDL_strlen(text));

    X11_XSetSelectionOwner(display, selection_type, window, CurrentTime);
    return 0;
}

static char *GetSlectionText(_THIS, Atom selection_type)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
    Display *display = videodata->display;
    Atom format;
    Window window;
    Window owner;
    Atom selection;
    Atom seln_type;
    int seln_format;
    unsigned long nbytes;
    unsigned long overflow;
    unsigned char *src;
    char *text;
    Uint32 waitStart;
    Uint32 waitElapsed;

    text = NULL;

    /* Get the window that holds the selection */
    window = GetWindow(_this);
    format = X11_GetSDLCutBufferClipboardInternalFormat(display, SDL_X11_CLIPBOARD_MIME_TYPE_STRING);
    owner = X11_XGetSelectionOwner(display, selection_type);
    if (owner == None) {
        /* Fall back to ancient X10 cut-buffers which do not support UTF8 strings*/
        owner = DefaultRootWindow(display);
        selection = XA_CUT_BUFFER0;
        format = XA_STRING;
    } else if (owner == window) {
        owner = DefaultRootWindow(display);
        selection = X11_GetSDLCutBufferClipboardType(display, SDL_X11_CLIPBOARD_MIME_TYPE_STRING, selection_type);
    } else {
        /* Request that the selection owner copy the data to our window */
        owner = window;
        selection = X11_XInternAtom(display, "SDL_SELECTION", False);
        X11_XConvertSelection(display, selection_type, format, selection, owner,
                              CurrentTime);

        /* When using synergy on Linux and when data has been put in the clipboard
           on the remote (Windows anyway) machine then selection_waiting may never
           be set to False. Time out after a while. */
        waitStart = SDL_GetTicks();
        videodata->selection_waiting = SDL_TRUE;
        while (videodata->selection_waiting) {
            SDL_PumpEvents();
            waitElapsed = SDL_GetTicks() - waitStart;
            /* Wait one second for a selection response. */
            if (waitElapsed > 1000) {
                videodata->selection_waiting = SDL_FALSE;
                SDL_SetError("Selection timeout");
                /* We need to set the selection text so that next time we won't
                   timeout, otherwise we will hang on every call to this function. */
                SetSelectionText(_this, "", selection_type);
                return SDL_strdup("");
            }
        }
    }

    if (X11_XGetWindowProperty(display, owner, selection, 0, INT_MAX / 4, False,
                               format, &seln_type, &seln_format, &nbytes, &overflow, &src) == Success) {
        if (seln_type == format) {
            text = (char *)SDL_malloc(nbytes + 1);
            if (text) {
                SDL_memcpy(text, src, nbytes);
                text[nbytes] = '\0';
            }
        }
        X11_XFree(src);
    }

    if (!text) {
        text = SDL_strdup("");
    }

    return text;
}

int X11_SetClipboardText(_THIS, const char *text)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
    Atom XA_CLIPBOARD = X11_XInternAtom(videodata->display, "CLIPBOARD", 0);
    if (XA_CLIPBOARD == None) {
        return SDL_SetError("Couldn't access X clipboard");
    }
    return SetSelectionText(_this, text, XA_CLIPBOARD);
}

int X11_SetPrimarySelectionText(_THIS, const char *text)
{
    return SetSelectionText(_this, text, XA_PRIMARY);
}

char *X11_GetClipboardText(_THIS)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
    Atom XA_CLIPBOARD = X11_XInternAtom(videodata->display, "CLIPBOARD", 0);
    if (XA_CLIPBOARD == None) {
        SDL_SetError("Couldn't access X clipboard");
        return SDL_strdup("");
    }
    return GetSlectionText(_this, XA_CLIPBOARD);
}

char *X11_GetPrimarySelectionText(_THIS)
{
    return GetSlectionText(_this, XA_PRIMARY);
}

SDL_bool X11_HasClipboardText(_THIS)
{
    SDL_bool result = SDL_FALSE;
    char *text = X11_GetClipboardText(_this);
    if (text) {
        result = text[0] != '\0' ? SDL_TRUE : SDL_FALSE;
        SDL_free(text);
    }
    return result;
}

SDL_bool X11_HasPrimarySelectionText(_THIS)
{
    SDL_bool result = SDL_FALSE;
    char *text = X11_GetPrimarySelectionText(_this);
    if (text) {
        result = text[0] != '\0' ? SDL_TRUE : SDL_FALSE;
        SDL_free(text);
    }
    return result;
}

#endif /* SDL_VIDEO_DRIVER_X11 */

/* vi: set ts=4 sw=4 expandtab: */
