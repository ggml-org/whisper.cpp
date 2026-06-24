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
#include "../SDL_sysvideo.h"

#include "SDL_BWin.h"
#include <new>

#include "SDL_syswm.h"

/* Define a path to window's BWIN data */
#ifdef __cplusplus
extern "C" {
#endif

static SDL_INLINE SDL_BWin *_ToBeWin(SDL_Window *window) {
    return (SDL_BWin *)(window->driverdata);
}

static SDL_INLINE SDL_BLooper *_GetBeLooper() {
    return SDL_Looper;
}

static int _InitWindow(_THIS, SDL_Window *window) {
    uint32 flags = 0;
    window_look look = B_TITLED_WINDOW_LOOK;

    BRect bounds(
        window->x,
        window->y,
        window->x + window->w - 1,    //BeWindows have an off-by-one px w/h thing
        window->y + window->h - 1
    );
    
    if (window->flags & SDL_WINDOW_FULLSCREEN) {
        /* TODO: Add support for this flag */
        printf(__FILE__": %d!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",__LINE__);
    }
    if (window->flags & SDL_WINDOW_OPENGL) {
        /* TODO: Add support for this flag */
    }
    if (!(window->flags & SDL_WINDOW_RESIZABLE)) {
        flags |= B_NOT_RESIZABLE | B_NOT_ZOOMABLE;
    }
    if (window->flags & SDL_WINDOW_BORDERLESS) {
        look = B_NO_BORDER_WINDOW_LOOK;
    }

    SDL_BWin *bwin = new(std::nothrow) SDL_BWin(bounds, look, flags);
    if (!bwin) {
        return -1;
    }

    window->driverdata = bwin;
    int32 winID = _GetBeLooper()->GetID(window);
    bwin->SetID(winID);

    return 0;
}

int HAIKU_CreateWindow(_THIS, SDL_Window *window) {
    if (_InitWindow(_this, window) < 0) {
        return -1;
    }
    
    /* Start window loop */
    _ToBeWin(window)->Show();
    return 0;
}

int HAIKU_CreateWindowFrom(_THIS, SDL_Window * window, const void *data) {

    SDL_BWin *otherBWin = (SDL_BWin*)data;
    if (!otherBWin->LockLooper()) {
        return -1;
    }
    
    /* Create the new window and initialize its members */
    window->x = (int)otherBWin->Frame().left;
    window->y = (int)otherBWin->Frame().top;
    window->w = (int)otherBWin->Frame().Width();
    window->h = (int)otherBWin->Frame().Height();
    
    /* Set SDL flags */
    if (!(otherBWin->Flags() & B_NOT_RESIZABLE)) {
        window->flags |= SDL_WINDOW_RESIZABLE;
    }
    
    /* If we are out of memory, return the error code */
    if (_InitWindow(_this, window) < 0) {
        return -1;
    }
    
    /* TODO: Add any other SDL-supported window attributes here */
    _ToBeWin(window)->SetTitle(otherBWin->Title());
    
    /* Start window loop and unlock the other window */
    _ToBeWin(window)->Show();
    
    otherBWin->UnlockLooper();
    return 0;
}

void HAIKU_SetWindowTitle(_THIS, SDL_Window * window) {
    BMessage msg(BWIN_SET_TITLE);
    msg.AddString("window-title", window->title);
    _ToBeWin(window)->PostMessage(&msg);
}

void HAIKU_SetWindowIcon(_THIS, SDL_Window * window, SDL_Surface * icon) {
    /* FIXME: Icons not supported by Haiku */
}

void HAIKU_SetWindowPosition(_THIS, SDL_Window * window) {
    BMessage msg(BWIN_MOVE_WINDOW);
    msg.AddInt32("window-x", window->x);
    msg.AddInt32("window-y", window->y);
    _ToBeWin(window)->PostMessage(&msg);
}

void HAIKU_SetWindowSize(_THIS, SDL_Window * window) {
    BMessage msg(BWIN_RESIZE_WINDOW);
    msg.AddInt32("window-w", window->w - 1);
    msg.AddInt32("window-h", window->h - 1);
    _ToBeWin(window)->PostMessage(&msg);
}

void HAIKU_SetWindowBordered(_THIS, SDL_Window * window, SDL_bool bordered) {
    BMessage msg(BWIN_SET_BORDERED);
    msg.AddBool("window-border", bordered != SDL_FALSE);
    _ToBeWin(window)->PostMessage(&msg);
}

void HAIKU_SetWindowResizable(_THIS, SDL_Window * window, SDL_bool resizable) {
    BMessage msg(BWIN_SET_RESIZABLE);
    msg.AddBool("window-resizable", resizable != SDL_FALSE);
    _ToBeWin(window)->PostMessage(&msg);
}

void HAIKU_ShowWindow(_THIS, SDL_Window * window) {
    BMessage msg(BWIN_SHOW_WINDOW);
    _ToBeWin(window)->PostMessage(&msg);
}

void HAIKU_HideWindow(_THIS, SDL_Window * window) {
    BMessage msg(BWIN_HIDE_WINDOW);
    _ToBeWin(window)->PostMessage(&msg);
}

void HAIKU_RaiseWindow(_THIS, SDL_Window * window) {
    BMessage msg(BWIN_SHOW_WINDOW);    /* Activate this window and move to front */
    _ToBeWin(window)->PostMessage(&msg);
}

void HAIKU_MaximizeWindow(_THIS, SDL_Window * window) {
    BMessage msg(BWIN_MAXIMIZE_WINDOW);
    _ToBeWin(window)->PostMessage(&msg);
}

void HAIKU_MinimizeWindow(_THIS, SDL_Window * window) {
    BMessage msg(BWIN_MINIMIZE_WINDOW);
    _ToBeWin(window)->PostMessage(&msg);
}

void HAIKU_RestoreWindow(_THIS, SDL_Window * window) {
    BMessage msg(BWIN_RESTORE_WINDOW);
    _ToBeWin(window)->PostMessage(&msg);
}

void HAIKU_SetWindowFullscreen(_THIS, SDL_Window * window,
        SDL_VideoDisplay * display, SDL_bool fullscreen) {
    /* Haiku tracks all video display information */
    BMessage msg(BWIN_FULLSCREEN);
    msg.AddBool("fullscreen", fullscreen);
    _ToBeWin(window)->PostMessage(&msg);
    
}

int HAIKU_SetWindowGammaRamp(_THIS, SDL_Window * window, const Uint16 * ramp) {
    /* FIXME: Not Haiku supported */
    return -1;
}

int HAIKU_GetWindowGammaRamp(_THIS, SDL_Window * window, Uint16 * ramp) {
    /* FIXME: Not Haiku supported */
    return -1;
}


void HAIKU_SetWindowMinimumSize(_THIS, SDL_Window * window) {
    BMessage msg(BWIN_MINIMUM_SIZE_WINDOW);
    msg.AddInt32("window-w", window->w -1);
    msg.AddInt32("window-h", window->h -1);
    _ToBeWin(window)->PostMessage(&msg);
}

void HAIKU_SetWindowMouseGrab(_THIS, SDL_Window * window, SDL_bool grabbed) {
    /* TODO: Implement this! */
}

void HAIKU_DestroyWindow(_THIS, SDL_Window * window) {
    _ToBeWin(window)->LockLooper();    /* This MUST be locked */
    _GetBeLooper()->ClearID(_ToBeWin(window));
    _ToBeWin(window)->Quit();
    window->driverdata = NULL;
}

SDL_bool HAIKU_GetWindowWMInfo(_THIS, SDL_Window * window,
                                    struct SDL_SysWMinfo *info) {
    /* FIXME: What is the point of this? What information should be included? */
	if (info->version.major == SDL_MAJOR_VERSION) {
	    info->subsystem = SDL_SYSWM_HAIKU;
	    return SDL_TRUE;
	} else {
	    SDL_SetError("Application not compiled with SDL %d",
	                 SDL_MAJOR_VERSION);
	    return SDL_FALSE;
	}
}




 
#ifdef __cplusplus
}
#endif

#endif /* SDL_VIDEO_DRIVER_HAIKU */

/* vi: set ts=4 sw=4 expandtab: */
