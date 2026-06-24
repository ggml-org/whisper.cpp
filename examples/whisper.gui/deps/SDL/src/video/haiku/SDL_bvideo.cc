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
#include "../../main/haiku/SDL_BApp.h"

#ifdef SDL_VIDEO_DRIVER_HAIKU

#include "SDL_BWin.h"
#include <Url.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "SDL_bkeyboard.h"
#include "SDL_bwindow.h"
#include "SDL_bclipboard.h"
#include "SDL_bvideo.h"
#include "SDL_bopengl.h"
#include "SDL_bmodes.h"
#include "SDL_bframebuffer.h"
#include "SDL_bevents.h"
#include "SDL_bmessagebox.h"

static SDL_INLINE SDL_BWin *_ToBeWin(SDL_Window *window) {
    return (SDL_BWin *)(window->driverdata);
}

/* FIXME: Undefined functions */
//    #define HAIKU_PumpEvents NULL
    #define HAIKU_StartTextInput NULL
    #define HAIKU_StopTextInput NULL
    #define HAIKU_SetTextInputRect NULL

//    #define HAIKU_DeleteDevice NULL

/* End undefined functions */

static SDL_VideoDevice * HAIKU_CreateDevice(void)
{
    SDL_VideoDevice *device;
    /*SDL_VideoData *data;*/

    /* Initialize all variables that we clean on shutdown */
    device = (SDL_VideoDevice *) SDL_calloc(1, sizeof(SDL_VideoDevice));

    device->driverdata = NULL; /* FIXME: Is this the cause of some of the
                                  SDL_Quit() errors? */

/* TODO: Figure out if any initialization needs to go here */

    /* Set the function pointers */
    device->VideoInit = HAIKU_VideoInit;
    device->VideoQuit = HAIKU_VideoQuit;
    device->GetDisplayBounds = HAIKU_GetDisplayBounds;
    device->GetDisplayModes = HAIKU_GetDisplayModes;
    device->SetDisplayMode = HAIKU_SetDisplayMode;
    device->PumpEvents = HAIKU_PumpEvents;

    device->CreateSDLWindow = HAIKU_CreateWindow;
    device->CreateSDLWindowFrom = HAIKU_CreateWindowFrom;
    device->SetWindowTitle = HAIKU_SetWindowTitle;
    device->SetWindowIcon = HAIKU_SetWindowIcon;
    device->SetWindowPosition = HAIKU_SetWindowPosition;
    device->SetWindowSize = HAIKU_SetWindowSize;
    device->ShowWindow = HAIKU_ShowWindow;
    device->HideWindow = HAIKU_HideWindow;
    device->RaiseWindow = HAIKU_RaiseWindow;
    device->MaximizeWindow = HAIKU_MaximizeWindow;
    device->MinimizeWindow = HAIKU_MinimizeWindow;
    device->RestoreWindow = HAIKU_RestoreWindow;
    device->SetWindowBordered = HAIKU_SetWindowBordered;
    device->SetWindowResizable = HAIKU_SetWindowResizable;
    device->SetWindowFullscreen = HAIKU_SetWindowFullscreen;
    device->SetWindowGammaRamp = HAIKU_SetWindowGammaRamp;
    device->GetWindowGammaRamp = HAIKU_GetWindowGammaRamp;
    device->SetWindowMouseGrab = HAIKU_SetWindowMouseGrab;
    device->SetWindowMinimumSize = HAIKU_SetWindowMinimumSize;
    device->DestroyWindow = HAIKU_DestroyWindow;
    device->GetWindowWMInfo = HAIKU_GetWindowWMInfo;
    device->CreateWindowFramebuffer = HAIKU_CreateWindowFramebuffer;
    device->UpdateWindowFramebuffer = HAIKU_UpdateWindowFramebuffer;
    device->DestroyWindowFramebuffer = HAIKU_DestroyWindowFramebuffer;
    
    device->shape_driver.CreateShaper = NULL;
    device->shape_driver.SetWindowShape = NULL;
    device->shape_driver.ResizeWindowShape = NULL;

#ifdef SDL_VIDEO_OPENGL
    device->GL_LoadLibrary = HAIKU_GL_LoadLibrary;
    device->GL_GetProcAddress = HAIKU_GL_GetProcAddress;
    device->GL_UnloadLibrary = HAIKU_GL_UnloadLibrary;
    device->GL_CreateContext = HAIKU_GL_CreateContext;
    device->GL_MakeCurrent = HAIKU_GL_MakeCurrent;
    device->GL_SetSwapInterval = HAIKU_GL_SetSwapInterval;
    device->GL_GetSwapInterval = HAIKU_GL_GetSwapInterval;
    device->GL_SwapWindow = HAIKU_GL_SwapWindow;
    device->GL_DeleteContext = HAIKU_GL_DeleteContext;
#endif

    device->StartTextInput = HAIKU_StartTextInput;
    device->StopTextInput = HAIKU_StopTextInput;
    device->SetTextInputRect = HAIKU_SetTextInputRect;

    device->SetClipboardText = HAIKU_SetClipboardText;
    device->GetClipboardText = HAIKU_GetClipboardText;
    device->HasClipboardText = HAIKU_HasClipboardText;

    device->free = HAIKU_DeleteDevice;

    return device;
}

VideoBootStrap HAIKU_bootstrap = {
    "haiku", "Haiku graphics",
    HAIKU_CreateDevice,
    HAIKU_ShowMessageBox
};

void HAIKU_DeleteDevice(SDL_VideoDevice * device)
{
    SDL_free(device->driverdata);
    SDL_free(device);
}

static SDL_Cursor * HAIKU_CreateSystemCursor(SDL_SystemCursor id)
{
    SDL_Cursor *cursor;
    BCursorID cursorId = B_CURSOR_ID_SYSTEM_DEFAULT;

    switch(id)
    {
    default:
        SDL_assert(0);
        return NULL;
    case SDL_SYSTEM_CURSOR_ARROW:     cursorId = B_CURSOR_ID_SYSTEM_DEFAULT; break;
    case SDL_SYSTEM_CURSOR_IBEAM:     cursorId = B_CURSOR_ID_I_BEAM; break;
    case SDL_SYSTEM_CURSOR_WAIT:      cursorId = B_CURSOR_ID_PROGRESS; break;
    case SDL_SYSTEM_CURSOR_CROSSHAIR: cursorId = B_CURSOR_ID_CROSS_HAIR; break;
    case SDL_SYSTEM_CURSOR_WAITARROW: cursorId = B_CURSOR_ID_PROGRESS; break;
    case SDL_SYSTEM_CURSOR_SIZENWSE:  cursorId = B_CURSOR_ID_RESIZE_NORTH_WEST_SOUTH_EAST; break;
    case SDL_SYSTEM_CURSOR_SIZENESW:  cursorId = B_CURSOR_ID_RESIZE_NORTH_EAST_SOUTH_WEST; break;
    case SDL_SYSTEM_CURSOR_SIZEWE:    cursorId = B_CURSOR_ID_RESIZE_EAST_WEST; break;
    case SDL_SYSTEM_CURSOR_SIZENS:    cursorId = B_CURSOR_ID_RESIZE_NORTH_SOUTH; break;
    case SDL_SYSTEM_CURSOR_SIZEALL:   cursorId = B_CURSOR_ID_MOVE; break;
    case SDL_SYSTEM_CURSOR_NO:        cursorId = B_CURSOR_ID_NOT_ALLOWED; break;
    case SDL_SYSTEM_CURSOR_HAND:      cursorId = B_CURSOR_ID_FOLLOW_LINK; break;
    }

    cursor = (SDL_Cursor *) SDL_calloc(1, sizeof(*cursor));
    if (cursor) {
        cursor->driverdata = (void *)new BCursor(cursorId);
    } else {
        SDL_OutOfMemory();
    }

    return cursor;
}

static SDL_Cursor * HAIKU_CreateDefaultCursor()
{
    return HAIKU_CreateSystemCursor(SDL_SYSTEM_CURSOR_ARROW);
}

static void HAIKU_FreeCursor(SDL_Cursor * cursor)
{
    if (cursor->driverdata) {
        delete (BCursor*) cursor->driverdata;
    }
    SDL_free(cursor);
}

static SDL_Cursor * HAIKU_CreateCursor(SDL_Surface * surface, int hot_x, int hot_y)
{
    SDL_Cursor *cursor;
    SDL_Surface *converted;

    converted = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_ARGB8888, 0);
    if (!converted) {
        return NULL;
    }

	BBitmap *cursorBitmap = new BBitmap(BRect(0, 0, surface->w - 1, surface->h - 1), B_RGBA32);
	cursorBitmap->SetBits(converted->pixels, converted->h * converted->pitch, 0, B_RGBA32);
    SDL_FreeSurface(converted);

    cursor = (SDL_Cursor *) SDL_calloc(1, sizeof(*cursor));
    if (cursor) {
        cursor->driverdata = (void *)new BCursor(cursorBitmap, BPoint(hot_x, hot_y));
    } else {
        return NULL;
    }

    return cursor;
}

static int HAIKU_ShowCursor(SDL_Cursor *cursor)
{
	SDL_Mouse *mouse = SDL_GetMouse();

	if (!mouse) {
		return 0;
	}

	if (cursor) {
		BCursor *hCursor = (BCursor*)cursor->driverdata;
		be_app->SetCursor(hCursor);
	} else {
		BCursor *hCursor = new BCursor(B_CURSOR_ID_NO_CURSOR);
		be_app->SetCursor(hCursor);
		delete hCursor;
	}

	return 0;
}

static int HAIKU_SetRelativeMouseMode(SDL_bool enabled)
{
    SDL_Window *window = SDL_GetMouseFocus();
    if (!window) {
      return 0;
    }

	SDL_BWin *bewin = _ToBeWin(window);
	BGLView *_SDL_GLView = bewin->GetGLView();

	bewin->Lock();
	if (enabled)
		_SDL_GLView->SetEventMask(B_POINTER_EVENTS, B_NO_POINTER_HISTORY);
	else
		_SDL_GLView->SetEventMask(0, 0);
	bewin->Unlock();

    return 0;
}

static void HAIKU_MouseInit(_THIS)
{
	SDL_Mouse *mouse = SDL_GetMouse();
	if (!mouse) {
		return;
	}
	mouse->CreateCursor = HAIKU_CreateCursor;
	mouse->CreateSystemCursor = HAIKU_CreateSystemCursor;
	mouse->ShowCursor = HAIKU_ShowCursor;
	mouse->FreeCursor = HAIKU_FreeCursor;
	mouse->SetRelativeMouseMode = HAIKU_SetRelativeMouseMode;

	SDL_SetDefaultCursor(HAIKU_CreateDefaultCursor());
}

int HAIKU_VideoInit(_THIS)
{
    /* Initialize the Be Application for appserver interaction */
    if (SDL_InitBeApp() < 0) {
        return -1;
    }
    
    /* Initialize video modes */
    HAIKU_InitModes(_this);

    /* Init the keymap */
    HAIKU_InitOSKeymap();

    HAIKU_MouseInit(_this);

#ifdef SDL_VIDEO_OPENGL
        /* testgl application doesn't load library, just tries to load symbols */
        /* is it correct? if so we have to load library here */
    HAIKU_GL_LoadLibrary(_this, NULL);
#endif

    /* We're done! */
    return 0;
}

void HAIKU_VideoQuit(_THIS)
{

    HAIKU_QuitModes(_this);

    SDL_QuitBeApp();
}

// just sticking this function in here so it's in a C++ source file.
extern "C" { int HAIKU_OpenURL(const char *url); }
int HAIKU_OpenURL(const char *url)
{
    BUrl burl(url);
    const status_t rc = burl.OpenWithPreferredApplication(false);
    return (rc == B_NO_ERROR) ? 0 : SDL_SetError("URL open failed (err=%d)", (int)rc);
}

#ifdef __cplusplus
}
#endif

#endif /* SDL_VIDEO_DRIVER_HAIKU */

/* vi: set ts=4 sw=4 expandtab: */
