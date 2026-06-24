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

#ifdef SDL_VIDEO_DRIVER_NGAGE

#include "../SDL_sysvideo.h"

#include "SDL_ngagewindow.h"

const TUint32 WindowClientHandle = 9210;

void DisableKeyBlocking(_THIS);
void ConstructWindowL(_THIS);

int NGAGE_CreateWindow(_THIS, SDL_Window *window)
{
    NGAGE_Window *ngage_window = (NGAGE_Window *)SDL_calloc(1, sizeof(NGAGE_Window));

    if (!ngage_window) {
        return SDL_OutOfMemory();
    }

    window->driverdata = ngage_window;

    if (window->x == SDL_WINDOWPOS_UNDEFINED) {
        window->x = 0;
    }

    if (window->y == SDL_WINDOWPOS_UNDEFINED) {
        window->y = 0;
    }

    ngage_window->sdl_window = window;

    ConstructWindowL(_this);

    return 0;
}

void NGAGE_DestroyWindow(_THIS, SDL_Window *window)
{
    NGAGE_Window *ngage_window = (NGAGE_Window *)window->driverdata;

    if (ngage_window) {
        SDL_free(ngage_window);
    }

    window->driverdata = NULL;
}

/*****************************************************************************/
/* Internal                                                                  */
/*****************************************************************************/

void DisableKeyBlocking(_THIS)
{
    SDL_VideoData *phdata = (SDL_VideoData *)_this->driverdata;
    TRawEvent event;

    event.Set((TRawEvent::TType) /*EDisableKeyBlock*/ 51);
    phdata->NGAGE_WsSession.SimulateRawEvent(event);
}

void ConstructWindowL(_THIS)
{
    SDL_VideoData *phdata = (SDL_VideoData *)_this->driverdata;
    TInt error;

    error = phdata->NGAGE_WsSession.Connect();
    User::LeaveIfError(error);
    phdata->NGAGE_WsScreen = new (ELeave) CWsScreenDevice(phdata->NGAGE_WsSession);
    User::LeaveIfError(phdata->NGAGE_WsScreen->Construct());
    User::LeaveIfError(phdata->NGAGE_WsScreen->CreateContext(phdata->NGAGE_WindowGc));

    phdata->NGAGE_WsWindowGroup = RWindowGroup(phdata->NGAGE_WsSession);
    User::LeaveIfError(phdata->NGAGE_WsWindowGroup.Construct(WindowClientHandle));
    phdata->NGAGE_WsWindowGroup.SetOrdinalPosition(0);

    RProcess thisProcess;
    TParse exeName;
    exeName.Set(thisProcess.FileName(), NULL, NULL);
    TBuf<32> winGroupName;
    winGroupName.Append(0);
    winGroupName.Append(0);
    winGroupName.Append(0); // UID
    winGroupName.Append(0);
    winGroupName.Append(exeName.Name()); // Caption
    winGroupName.Append(0);
    winGroupName.Append(0); // DOC name
    phdata->NGAGE_WsWindowGroup.SetName(winGroupName);

    phdata->NGAGE_WsWindow = RWindow(phdata->NGAGE_WsSession);
    User::LeaveIfError(phdata->NGAGE_WsWindow.Construct(phdata->NGAGE_WsWindowGroup, WindowClientHandle - 1));
    phdata->NGAGE_WsWindow.SetBackgroundColor(KRgbWhite);
    phdata->NGAGE_WsWindow.Activate();
    phdata->NGAGE_WsWindow.SetSize(phdata->NGAGE_WsScreen->SizeInPixels());
    phdata->NGAGE_WsWindow.SetVisible(ETrue);

    phdata->NGAGE_WsWindowGroupID = phdata->NGAGE_WsWindowGroup.Identifier();
    phdata->NGAGE_IsWindowFocused = EFalse;

    DisableKeyBlocking(_this);
}

#endif /* SDL_VIDEO_DRIVER_NGAGE */

/* vi: set ts=4 sw=4 expandtab: */
