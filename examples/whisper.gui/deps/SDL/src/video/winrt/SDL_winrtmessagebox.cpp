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

#ifdef SDL_VIDEO_DRIVER_WINRT

extern "C" {
#include "SDL_messagebox.h"
#include "../../core/windows/SDL_windows.h"
}

#include "SDL_winrtevents_c.h"

#include <windows.ui.popups.h>
using namespace Platform;
using namespace Windows::Foundation;
using namespace Windows::UI::Popups;

static String ^ WINRT_UTF8ToPlatformString(const char *str) {
    wchar_t *wstr = WIN_UTF8ToString(str);
    String ^ rtstr = ref new String(wstr);
    SDL_free(wstr);
    return rtstr;
}

extern "C" int WINRT_ShowMessageBox(const SDL_MessageBoxData *messageboxdata, int *buttonid)
{
#if SDL_WINAPI_FAMILY_PHONE && (NTDDI_VERSION == NTDDI_WIN8)
    /* Sadly, Windows Phone 8 doesn't include the MessageDialog class that
     * Windows 8.x/RT does, even though MSDN's reference documentation for
     * Windows Phone 8 mentions it.
     *
     * The .NET runtime on Windows Phone 8 does, however, include a
     * MessageBox class.  Perhaps this could be called, somehow?
     */
    return SDL_SetError("SDL_messagebox support is not available for Windows Phone 8.0");
#else
    SDL_VideoDevice *_this = SDL_GetVideoDevice();

#if SDL_WINAPI_FAMILY_PHONE
    const int maxbuttons = 2;
    const char *platform = "Windows Phone 8.1+";
#else
    const int maxbuttons = 3;
    const char *platform = "Windows 8.x";
#endif

    if (messageboxdata->numbuttons > maxbuttons) {
        return SDL_SetError("WinRT's MessageDialog only supports %d buttons, at most, on %s. %d were requested.",
                            maxbuttons, platform, messageboxdata->numbuttons);
    }

    /* Build a MessageDialog object and its buttons */
    MessageDialog ^ dialog = ref new MessageDialog(WINRT_UTF8ToPlatformString(messageboxdata->message));
    dialog->Title = WINRT_UTF8ToPlatformString(messageboxdata->title);
    for (int i = 0; i < messageboxdata->numbuttons; ++i) {
        const SDL_MessageBoxButtonData *sdlButton;
        if (messageboxdata->flags & SDL_MESSAGEBOX_BUTTONS_RIGHT_TO_LEFT) {
            sdlButton = &messageboxdata->buttons[messageboxdata->numbuttons - 1 - i];
        } else {
            sdlButton = &messageboxdata->buttons[i];
        }
        UICommand ^ button = ref new UICommand(WINRT_UTF8ToPlatformString(sdlButton->text));
        button->Id = IntPtr((int)((size_t)(sdlButton - messageboxdata->buttons)));
        dialog->Commands->Append(button);
        if (sdlButton->flags & SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT) {
            dialog->CancelCommandIndex = i;
        }
        if (sdlButton->flags & SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT) {
            dialog->DefaultCommandIndex = i;
        }
    }

    /* Display the MessageDialog, then wait for it to be closed */
    /* TODO, WinRT: Find a way to redraw MessageDialog instances if a GPU device-reset occurs during the following event-loop */
    auto operation = dialog->ShowAsync();
    while (operation->Status == Windows::Foundation::AsyncStatus::Started) {
        WINRT_PumpEvents(_this);
    }

    /* Retrieve results from the MessageDialog and process them accordingly */
    if (operation->Status != Windows::Foundation::AsyncStatus::Completed) {
        return SDL_SetError("An unknown error occurred in displaying the WinRT MessageDialog");
    }
    if (buttonid) {
        IntPtr results = safe_cast<IntPtr>(operation->GetResults()->Id);
        int clicked_index = results.ToInt32();
        *buttonid = messageboxdata->buttons[clicked_index].buttonid;
    }
    return 0;
#endif /* if SDL_WINAPI_FAMILY_PHONE / else */
}

#endif /* SDL_VIDEO_DRIVER_WINRT */

/* vi: set ts=4 sw=4 expandtab: */
