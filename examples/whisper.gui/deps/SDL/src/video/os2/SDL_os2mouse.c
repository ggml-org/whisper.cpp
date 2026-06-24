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

#ifdef SDL_VIDEO_DRIVER_OS2

#include "SDL_os2video.h"
#include "../../events/SDL_mouse_c.h"
#include "SDL_os2util.h"

HPOINTER    hptrCursor = NULLHANDLE;

static SDL_Cursor* OS2_CreateCursor(SDL_Surface *surface, int hot_x, int hot_y)
{
    ULONG       ulMaxW = WinQuerySysValue(HWND_DESKTOP, SV_CXPOINTER);
    ULONG       ulMaxH = WinQuerySysValue(HWND_DESKTOP, SV_CYPOINTER);
    HPOINTER    hptr;
    SDL_Cursor* pSDLCursor;

    if (surface->w > ulMaxW || surface->h > ulMaxH) {
        debug_os2("Given image size is %u x %u, maximum allowed size is %u x %u",
                  surface->w, surface->h, ulMaxW, ulMaxH);
        return NULL;
    }

    hptr = utilCreatePointer(surface, hot_x, ulMaxH - hot_y - 1);
    if (hptr == NULLHANDLE)
        return NULL;

    pSDLCursor = SDL_calloc(1, sizeof(SDL_Cursor));
    if (!pSDLCursor) {
        WinDestroyPointer(hptr);
        SDL_OutOfMemory();
        return NULL;
    }

    pSDLCursor->driverdata = (void *)hptr;
    return pSDLCursor;
}

static SDL_Cursor* OS2_CreateSystemCursor(SDL_SystemCursor id)
{
    SDL_Cursor* pSDLCursor;
    LONG        lSysId;
    HPOINTER    hptr;

    switch (id) {
    case SDL_SYSTEM_CURSOR_ARROW:     lSysId = SPTR_ARROW;    break;
    case SDL_SYSTEM_CURSOR_IBEAM:     lSysId = SPTR_TEXT;     break;
    case SDL_SYSTEM_CURSOR_WAIT:      lSysId = SPTR_WAIT;     break;
    case SDL_SYSTEM_CURSOR_CROSSHAIR: lSysId = SPTR_MOVE;     break;
    case SDL_SYSTEM_CURSOR_WAITARROW: lSysId = SPTR_WAIT;     break;
    case SDL_SYSTEM_CURSOR_SIZENWSE:  lSysId = SPTR_SIZENWSE; break;
    case SDL_SYSTEM_CURSOR_SIZENESW:  lSysId = SPTR_SIZENESW; break;
    case SDL_SYSTEM_CURSOR_SIZEWE:    lSysId = SPTR_SIZEWE;   break;
    case SDL_SYSTEM_CURSOR_SIZENS:    lSysId = SPTR_SIZENS;   break;
    case SDL_SYSTEM_CURSOR_SIZEALL:   lSysId = SPTR_MOVE;     break;
    case SDL_SYSTEM_CURSOR_NO:        lSysId = SPTR_ILLEGAL;  break;
    case SDL_SYSTEM_CURSOR_HAND:      lSysId = SPTR_ARROW;    break;
    default:
        debug_os2("Unknown cursor id: %u", id);
        return NULL;
    }

    /* On eCS SPTR_WAIT for last paramether fCopy=TRUE/FALSE gives different
     * "wait" icons. -=8( ) */
    hptr = WinQuerySysPointer(HWND_DESKTOP, lSysId,
                              id == SDL_SYSTEM_CURSOR_WAIT);
    if (hptr == NULLHANDLE) {
        debug_os2("Cannot load OS/2 system pointer %u for SDL cursor id %u",
                  lSysId, id);
        return NULL;
    }

    pSDLCursor = SDL_calloc(1, sizeof(SDL_Cursor));
    if (!pSDLCursor) {
        WinDestroyPointer(hptr);
        SDL_OutOfMemory();
        return NULL;
    }

    pSDLCursor->driverdata = (void *)hptr;
    return pSDLCursor;
}

static void OS2_FreeCursor(SDL_Cursor *cursor)
{
    HPOINTER    hptr = (HPOINTER)cursor->driverdata;

    WinDestroyPointer(hptr);
    SDL_free(cursor);
}

static int OS2_ShowCursor(SDL_Cursor *cursor)
{
    hptrCursor = (cursor)? (HPOINTER)cursor->driverdata : NULLHANDLE;
    return ((SDL_GetMouseFocus() == NULL) ||
             WinSetPointer(HWND_DESKTOP, hptrCursor))? 0 : -1;
}

static void OS2_WarpMouse(SDL_Window * window, int x, int y)
{
    WINDATA    *pWinData = (WINDATA *)window->driverdata;
    POINTL      pointl;

    pointl.x = x;
    pointl.y = window->h - y - 1;
    WinMapWindowPoints(pWinData->hwnd, HWND_DESKTOP, &pointl, 1);
    WinSetPointerPos(HWND_DESKTOP, pointl.x, pointl.y);
    SDL_SendMouseMotion(window, SDL_GetMouse()->mouseID, 0, x, y);
}

static int OS2_WarpMouseGlobal(int x, int y)
{
    WinSetPointerPos(HWND_DESKTOP, x,
                     WinQuerySysValue(HWND_DESKTOP, SV_CYSCREEN) - y);
    return 0;
}

static int OS2_CaptureMouse(SDL_Window *window)
{
    return WinSetCapture(HWND_DESKTOP, (!window)? NULLHANDLE :
                                         ((WINDATA *)window->driverdata)->hwnd)? 0 : -1;
}

static Uint32 OS2_GetGlobalMouseState(int *x, int *y)
{
    POINTL  pointl;
    ULONG   ulRes;

    WinQueryPointerPos(HWND_DESKTOP, &pointl);
    *x = pointl.x;
    *y = WinQuerySysValue(HWND_DESKTOP, SV_CYSCREEN) - pointl.y - 1;

    ulRes = (WinGetKeyState(HWND_DESKTOP, VK_BUTTON1) & 0x8000)? SDL_BUTTON_LMASK : 0;
    if (WinGetKeyState(HWND_DESKTOP, VK_BUTTON2) & 0x8000)
        ulRes |= SDL_BUTTON_RMASK;
    if (WinGetKeyState(HWND_DESKTOP, VK_BUTTON3) & 0x8000)
        ulRes |= SDL_BUTTON_MMASK;

    return ulRes;
}


void OS2_InitMouse(_THIS, ULONG hab)
{
    SDL_Mouse   *pSDLMouse = SDL_GetMouse();

    pSDLMouse->CreateCursor         = OS2_CreateCursor;
    pSDLMouse->CreateSystemCursor   = OS2_CreateSystemCursor;
    pSDLMouse->ShowCursor           = OS2_ShowCursor;
    pSDLMouse->FreeCursor           = OS2_FreeCursor;
    pSDLMouse->WarpMouse            = OS2_WarpMouse;
    pSDLMouse->WarpMouseGlobal      = OS2_WarpMouseGlobal;
    pSDLMouse->CaptureMouse         = OS2_CaptureMouse;
    pSDLMouse->GetGlobalMouseState  = OS2_GetGlobalMouseState;

    SDL_SetDefaultCursor(OS2_CreateSystemCursor(SDL_SYSTEM_CURSOR_ARROW));
    if (hptrCursor == NULLHANDLE)
        hptrCursor = WinQuerySysPointer(HWND_DESKTOP, SPTR_ARROW, TRUE);
}

void OS2_QuitMouse(_THIS)
{
    SDL_Mouse   *pSDLMouse = SDL_GetMouse();

    if (pSDLMouse->def_cursor) {
        SDL_free(pSDLMouse->def_cursor);
        pSDLMouse->def_cursor = NULL;
        pSDLMouse->cur_cursor = NULL;
    }
}

#endif /* SDL_VIDEO_DRIVER_OS2 */

/* vi: set ts=4 sw=4 expandtab: */
