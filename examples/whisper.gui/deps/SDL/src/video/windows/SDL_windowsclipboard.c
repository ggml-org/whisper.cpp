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

#if defined(SDL_VIDEO_DRIVER_WINDOWS) && !defined(__XBOXONE__) && !defined(__XBOXSERIES__)

#include "SDL_windowsvideo.h"
#include "SDL_windowswindow.h"
#include "SDL_timer.h"
#include "../../events/SDL_clipboardevents_c.h"

#ifdef UNICODE
#define TEXT_FORMAT CF_UNICODETEXT
#else
#define TEXT_FORMAT CF_TEXT
#endif

/* Get any application owned window handle for clipboard association */
static HWND GetWindowHandle(_THIS)
{
    SDL_Window *window;

    window = _this->windows;
    if (window) {
        return ((SDL_WindowData *)window->driverdata)->hwnd;
    }
    return NULL;
}

int WIN_SetClipboardText(_THIS, const char *text)
{
    SDL_VideoData *data = (SDL_VideoData *)_this->driverdata;
    int result = 0;

    if (OpenClipboard(GetWindowHandle(_this))) {
        HANDLE hMem;
        LPTSTR tstr;
        SIZE_T i, size;

        /* Convert the text from UTF-8 to Windows Unicode */
        tstr = WIN_UTF8ToString(text);
        if (!tstr) {
            return -1;
        }

        /* Find out the size of the data */
        for (size = 0, i = 0; tstr[i]; ++i, ++size) {
            if (tstr[i] == '\n' && (i == 0 || tstr[i - 1] != '\r')) {
                /* We're going to insert a carriage return */
                ++size;
            }
        }
        size = (size + 1) * sizeof(*tstr);

        /* Save the data to the clipboard */
        hMem = GlobalAlloc(GMEM_MOVEABLE, size);
        if (hMem) {
            LPTSTR dst = (LPTSTR)GlobalLock(hMem);
            if (dst) {
                /* Copy the text over, adding carriage returns as necessary */
                for (i = 0; tstr[i]; ++i) {
                    if (tstr[i] == '\n' && (i == 0 || tstr[i - 1] != '\r')) {
                        *dst++ = '\r';
                    }
                    *dst++ = tstr[i];
                }
                *dst = 0;
                GlobalUnlock(hMem);
            }

            EmptyClipboard();
            if (!SetClipboardData(TEXT_FORMAT, hMem)) {
                result = WIN_SetError("Couldn't set clipboard data");
            }
            data->clipboard_count = GetClipboardSequenceNumber();
        }
        SDL_free(tstr);

        CloseClipboard();
    } else {
        result = WIN_SetError("Couldn't open clipboard");
    }
    return result;
}

char *WIN_GetClipboardText(_THIS)
{
    char *text = NULL;

    if (IsClipboardFormatAvailable(TEXT_FORMAT)) {
        /* Retry to open the clipboard in case another application has it open */
        const int MAX_ATTEMPTS = 3;
        int attempt;

        for (attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
            if (OpenClipboard(GetWindowHandle(_this))) {
                HANDLE hMem;
                LPTSTR tstr;

                hMem = GetClipboardData(TEXT_FORMAT);
                if (hMem) {
                    tstr = (LPTSTR)GlobalLock(hMem);
                    text = WIN_StringToUTF8(tstr);
                    GlobalUnlock(hMem);
                } else {
                    WIN_SetError("Couldn't get clipboard data");
                }
                CloseClipboard();
                break;
            }
            SDL_Delay(10);
        }
    }
    if (!text) {
        text = SDL_strdup("");
    }
    return text;
}

SDL_bool WIN_HasClipboardText(_THIS)
{
    SDL_bool result = SDL_FALSE;
    char *text = WIN_GetClipboardText(_this);
    if (text) {
        result = text[0] != '\0' ? SDL_TRUE : SDL_FALSE;
        SDL_free(text);
    }
    return result;
}

void WIN_CheckClipboardUpdate(struct SDL_VideoData *data)
{
    const DWORD count = GetClipboardSequenceNumber();
    if (count != data->clipboard_count) {
        if (data->clipboard_count) {
            SDL_SendClipboardUpdate();
        }
        data->clipboard_count = count;
    }
}

#endif /* SDL_VIDEO_DRIVER_WINDOWS */

/* vi: set ts=4 sw=4 expandtab: */
