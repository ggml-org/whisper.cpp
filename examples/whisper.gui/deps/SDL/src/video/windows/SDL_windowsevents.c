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

#ifdef SDL_VIDEO_DRIVER_WINDOWS

#include "SDL_windowsvideo.h"
#include "SDL_windowsshape.h"
#include "SDL_system.h"
#include "SDL_syswm.h"
#include "SDL_timer.h"
#include "SDL_vkeys.h"
#include "SDL_hints.h"
#include "SDL_main.h"
#include "../../events/SDL_events_c.h"
#include "../../events/SDL_touch_c.h"
#include "../../events/scancodes_windows.h"
#include "SDL_hints.h"
#include "SDL_log.h"

/* Dropfile support */
#include <shellapi.h>

/* For GET_X_LPARAM, GET_Y_LPARAM. */
#include <windowsx.h>

/* For WM_TABLET_QUERYSYSTEMGESTURESTATUS et. al. */
#ifdef HAVE_TPCSHRD_H
#include <tpcshrd.h>
#endif /* HAVE_TPCSHRD_H */

/* #define WMMSG_DEBUG */
#ifdef WMMSG_DEBUG
#include <stdio.h>
#include "wmmsg.h"
#endif

#ifdef __GDK__
#include "../../core/gdk/SDL_gdk.h"
#endif

/* #define HIGHDPI_DEBUG */

/* Masks for processing the windows KEYDOWN and KEYUP messages */
#define REPEATED_KEYMASK (1 << 30)
#define EXTENDED_KEYMASK (1 << 24)

#define VK_ENTER 10 /* Keypad Enter ... no VKEY defined? */
#ifndef VK_OEM_NEC_EQUAL
#define VK_OEM_NEC_EQUAL 0x92
#endif

/* Make sure XBUTTON stuff is defined that isn't in older Platform SDKs... */
#ifndef WM_XBUTTONDOWN
#define WM_XBUTTONDOWN 0x020B
#endif
#ifndef WM_XBUTTONUP
#define WM_XBUTTONUP 0x020C
#endif
#ifndef GET_XBUTTON_WPARAM
#define GET_XBUTTON_WPARAM(w) (HIWORD(w))
#endif
#ifndef WM_INPUT
#define WM_INPUT 0x00ff
#endif
#ifndef WM_TOUCH
#define WM_TOUCH 0x0240
#endif
#ifndef WM_MOUSEHWHEEL
#define WM_MOUSEHWHEEL 0x020E
#endif
#ifndef WM_POINTERUPDATE
#define WM_POINTERUPDATE 0x0245
#endif
#ifndef WM_UNICHAR
#define WM_UNICHAR 0x0109
#endif
#ifndef WM_DPICHANGED
#define WM_DPICHANGED 0x02E0
#endif
#ifndef WM_GETDPISCALEDSIZE
#define WM_GETDPISCALEDSIZE 0x02E4
#endif
#ifndef TOUCHEVENTF_PEN
#define TOUCHEVENTF_PEN 0x0040
#endif

#ifndef IS_HIGH_SURROGATE
#define IS_HIGH_SURROGATE(x) (((x) >= 0xd800) && ((x) <= 0xdbff))
#endif
#ifndef IS_LOW_SURROGATE
#define IS_LOW_SURROGATE(x) (((x) >= 0xdc00) && ((x) <= 0xdfff))
#endif
#ifndef IS_SURROGATE_PAIR
#define IS_SURROGATE_PAIR(h, l) (IS_HIGH_SURROGATE(h) && IS_LOW_SURROGATE(l))
#endif

#ifndef USER_TIMER_MINIMUM
#define USER_TIMER_MINIMUM 0x0000000A
#endif

static SDL_Scancode VKeytoScancodeFallback(WPARAM vkey)
{
    switch (vkey) {
    case VK_LEFT:
        return SDL_SCANCODE_LEFT;
    case VK_UP:
        return SDL_SCANCODE_UP;
    case VK_RIGHT:
        return SDL_SCANCODE_RIGHT;
    case VK_DOWN:
        return SDL_SCANCODE_DOWN;
    case VK_CONTROL:
        return SDL_SCANCODE_LCTRL;
    case VK_V:
        return SDL_SCANCODE_V;

    default:
        return SDL_SCANCODE_UNKNOWN;
    }
}

static SDL_Scancode VKeytoScancode(WPARAM vkey)
{
    switch (vkey) {
    case VK_BACK:
        return SDL_SCANCODE_BACKSPACE;
    case VK_CAPITAL:
        return SDL_SCANCODE_CAPSLOCK;

    case VK_MODECHANGE:
        return SDL_SCANCODE_MODE;
    case VK_SELECT:
        return SDL_SCANCODE_SELECT;
    case VK_EXECUTE:
        return SDL_SCANCODE_EXECUTE;
    case VK_HELP:
        return SDL_SCANCODE_HELP;
    case VK_PAUSE:
        return SDL_SCANCODE_PAUSE;
    case VK_NUMLOCK:
        return SDL_SCANCODE_NUMLOCKCLEAR;

    case VK_F13:
        return SDL_SCANCODE_F13;
    case VK_F14:
        return SDL_SCANCODE_F14;
    case VK_F15:
        return SDL_SCANCODE_F15;
    case VK_F16:
        return SDL_SCANCODE_F16;
    case VK_F17:
        return SDL_SCANCODE_F17;
    case VK_F18:
        return SDL_SCANCODE_F18;
    case VK_F19:
        return SDL_SCANCODE_F19;
    case VK_F20:
        return SDL_SCANCODE_F20;
    case VK_F21:
        return SDL_SCANCODE_F21;
    case VK_F22:
        return SDL_SCANCODE_F22;
    case VK_F23:
        return SDL_SCANCODE_F23;
    case VK_F24:
        return SDL_SCANCODE_F24;

    case VK_OEM_NEC_EQUAL:
        return SDL_SCANCODE_KP_EQUALS;
    case VK_BROWSER_BACK:
        return SDL_SCANCODE_AC_BACK;
    case VK_BROWSER_FORWARD:
        return SDL_SCANCODE_AC_FORWARD;
    case VK_BROWSER_REFRESH:
        return SDL_SCANCODE_AC_REFRESH;
    case VK_BROWSER_STOP:
        return SDL_SCANCODE_AC_STOP;
    case VK_BROWSER_SEARCH:
        return SDL_SCANCODE_AC_SEARCH;
    case VK_BROWSER_FAVORITES:
        return SDL_SCANCODE_AC_BOOKMARKS;
    case VK_BROWSER_HOME:
        return SDL_SCANCODE_AC_HOME;
    case VK_VOLUME_MUTE:
        return SDL_SCANCODE_MUTE;
    case VK_VOLUME_DOWN:
        return SDL_SCANCODE_VOLUMEDOWN;
    case VK_VOLUME_UP:
        return SDL_SCANCODE_VOLUMEUP;

    case VK_MEDIA_NEXT_TRACK:
        return SDL_SCANCODE_AUDIONEXT;
    case VK_MEDIA_PREV_TRACK:
        return SDL_SCANCODE_AUDIOPREV;
    case VK_MEDIA_STOP:
        return SDL_SCANCODE_AUDIOSTOP;
    case VK_MEDIA_PLAY_PAUSE:
        return SDL_SCANCODE_AUDIOPLAY;
    case VK_LAUNCH_MAIL:
        return SDL_SCANCODE_MAIL;
    case VK_LAUNCH_MEDIA_SELECT:
        return SDL_SCANCODE_MEDIASELECT;

    case VK_OEM_102:
        return SDL_SCANCODE_NONUSBACKSLASH;

    case VK_ATTN:
        return SDL_SCANCODE_SYSREQ;
    case VK_CRSEL:
        return SDL_SCANCODE_CRSEL;
    case VK_EXSEL:
        return SDL_SCANCODE_EXSEL;
    case VK_OEM_CLEAR:
        return SDL_SCANCODE_CLEAR;

    case VK_LAUNCH_APP1:
        return SDL_SCANCODE_APP1;
    case VK_LAUNCH_APP2:
        return SDL_SCANCODE_APP2;

    default:
        return SDL_SCANCODE_UNKNOWN;
    }
}

static SDL_Scancode WindowsScanCodeToSDLScanCode(LPARAM lParam, WPARAM wParam)
{
    SDL_Scancode code;
    int nScanCode = (lParam >> 16) & 0xFF;
    SDL_bool bIsExtended = (lParam & (1 << 24)) != 0;

    code = VKeytoScancode(wParam);

    if (code == SDL_SCANCODE_UNKNOWN && nScanCode <= 127) {
        code = windows_scancode_table[nScanCode];

        if (bIsExtended) {
            switch (code) {
            case SDL_SCANCODE_RETURN:
                code = SDL_SCANCODE_KP_ENTER;
                break;
            case SDL_SCANCODE_LALT:
                code = SDL_SCANCODE_RALT;
                break;
            case SDL_SCANCODE_LCTRL:
                code = SDL_SCANCODE_RCTRL;
                break;
            case SDL_SCANCODE_SLASH:
                code = SDL_SCANCODE_KP_DIVIDE;
                break;
            case SDL_SCANCODE_CAPSLOCK:
                code = SDL_SCANCODE_KP_PLUS;
                break;
            default:
                break;
            }
        } else {
            switch (code) {
            case SDL_SCANCODE_HOME:
                code = SDL_SCANCODE_KP_7;
                break;
            case SDL_SCANCODE_UP:
                code = SDL_SCANCODE_KP_8;
                break;
            case SDL_SCANCODE_PAGEUP:
                code = SDL_SCANCODE_KP_9;
                break;
            case SDL_SCANCODE_LEFT:
                code = SDL_SCANCODE_KP_4;
                break;
            case SDL_SCANCODE_RIGHT:
                code = SDL_SCANCODE_KP_6;
                break;
            case SDL_SCANCODE_END:
                code = SDL_SCANCODE_KP_1;
                break;
            case SDL_SCANCODE_DOWN:
                code = SDL_SCANCODE_KP_2;
                break;
            case SDL_SCANCODE_PAGEDOWN:
                code = SDL_SCANCODE_KP_3;
                break;
            case SDL_SCANCODE_INSERT:
                code = SDL_SCANCODE_KP_0;
                break;
            case SDL_SCANCODE_DELETE:
                code = SDL_SCANCODE_KP_PERIOD;
                break;
            case SDL_SCANCODE_PRINTSCREEN:
                code = SDL_SCANCODE_KP_MULTIPLY;
                break;
            default:
                break;
            }
        }
    }

    /* The on-screen keyboard can generate VK_LEFT and VK_RIGHT events without a scancode
     * value set, however we cannot simply map these in VKeytoScancode() or we will be
     * incorrectly handling the arrow keys on the number pad when NumLock is disabled
     * (which also generate VK_LEFT, VK_RIGHT, etc in that scenario). Instead, we'll only
     * map them if none of the above special number pad mappings applied. */
    if (code == SDL_SCANCODE_UNKNOWN) {
        code = VKeytoScancodeFallback(wParam);
    }

    return code;
}

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
static SDL_bool WIN_ShouldIgnoreFocusClick(void)
{
    return !SDL_GetHintBoolean(SDL_HINT_MOUSE_FOCUS_CLICKTHROUGH, SDL_FALSE);
}

static void WIN_CheckWParamMouseButton(SDL_bool bwParamMousePressed, Uint32 mouseFlags, SDL_bool bSwapButtons, SDL_WindowData *data, Uint8 button, SDL_MouseID mouseID)
{
    if (bSwapButtons) {
        if (button == SDL_BUTTON_LEFT) {
            button = SDL_BUTTON_RIGHT;
        } else if (button == SDL_BUTTON_RIGHT) {
            button = SDL_BUTTON_LEFT;
        }
    }

    if (data->focus_click_pending & SDL_BUTTON(button)) {
        /* Ignore the button click for activation */
        if (!bwParamMousePressed) {
            data->focus_click_pending &= ~SDL_BUTTON(button);
            WIN_UpdateClipCursor(data->window);
        }
        if (WIN_ShouldIgnoreFocusClick()) {
            return;
        }
    }

    if (bwParamMousePressed && !(mouseFlags & SDL_BUTTON(button))) {
        SDL_SendMouseButton(data->window, mouseID, SDL_PRESSED, button);
    } else if (!bwParamMousePressed && (mouseFlags & SDL_BUTTON(button))) {
        SDL_SendMouseButton(data->window, mouseID, SDL_RELEASED, button);
    }
}

/*
 * Some windows systems fail to send a WM_LBUTTONDOWN sometimes, but each mouse move contains the current button state also
 *  so this function reconciles our view of the world with the current buttons reported by windows
 */
static void WIN_CheckWParamMouseButtons(WPARAM wParam, SDL_WindowData *data, SDL_MouseID mouseID)
{
    if (wParam != data->mouse_button_flags) {
        Uint32 mouseFlags = SDL_GetMouseState(NULL, NULL);

        /* WM_LBUTTONDOWN and friends handle button swapping for us. No need to check SM_SWAPBUTTON here.  */
        WIN_CheckWParamMouseButton((wParam & MK_LBUTTON), mouseFlags, SDL_FALSE, data, SDL_BUTTON_LEFT, mouseID);
        WIN_CheckWParamMouseButton((wParam & MK_MBUTTON), mouseFlags, SDL_FALSE, data, SDL_BUTTON_MIDDLE, mouseID);
        WIN_CheckWParamMouseButton((wParam & MK_RBUTTON), mouseFlags, SDL_FALSE, data, SDL_BUTTON_RIGHT, mouseID);
        WIN_CheckWParamMouseButton((wParam & MK_XBUTTON1), mouseFlags, SDL_FALSE, data, SDL_BUTTON_X1, mouseID);
        WIN_CheckWParamMouseButton((wParam & MK_XBUTTON2), mouseFlags, SDL_FALSE, data, SDL_BUTTON_X2, mouseID);

        data->mouse_button_flags = wParam;
    }
}

static void WIN_CheckRawMouseButtons(HANDLE hDevice, ULONG rawButtons, SDL_WindowData *data, SDL_MouseID mouseID)
{
    // Add a flag to distinguish raw mouse buttons from wParam above
    rawButtons |= 0x8000000;

    if (rawButtons != data->mouse_button_flags) {
        Uint32 mouseFlags = SDL_GetMouseState(NULL, NULL);
        SDL_bool swapButtons = GetSystemMetrics(SM_SWAPBUTTON) != 0;
        if (swapButtons && hDevice == NULL) {
            /* Touchpad, already has buttons swapped */
            swapButtons = SDL_FALSE;
        }
        if (rawButtons & RI_MOUSE_BUTTON_1_DOWN) {
            WIN_CheckWParamMouseButton((rawButtons & RI_MOUSE_BUTTON_1_DOWN), mouseFlags, swapButtons, data, SDL_BUTTON_LEFT, mouseID);
        }
        if (rawButtons & RI_MOUSE_BUTTON_1_UP) {
            WIN_CheckWParamMouseButton(!(rawButtons & RI_MOUSE_BUTTON_1_UP), mouseFlags, swapButtons, data, SDL_BUTTON_LEFT, mouseID);
        }
        if (rawButtons & RI_MOUSE_BUTTON_2_DOWN) {
            WIN_CheckWParamMouseButton((rawButtons & RI_MOUSE_BUTTON_2_DOWN), mouseFlags, swapButtons, data, SDL_BUTTON_RIGHT, mouseID);
        }
        if (rawButtons & RI_MOUSE_BUTTON_2_UP) {
            WIN_CheckWParamMouseButton(!(rawButtons & RI_MOUSE_BUTTON_2_UP), mouseFlags, swapButtons, data, SDL_BUTTON_RIGHT, mouseID);
        }
        if (rawButtons & RI_MOUSE_BUTTON_3_DOWN) {
            WIN_CheckWParamMouseButton((rawButtons & RI_MOUSE_BUTTON_3_DOWN), mouseFlags, swapButtons, data, SDL_BUTTON_MIDDLE, mouseID);
        }
        if (rawButtons & RI_MOUSE_BUTTON_3_UP) {
            WIN_CheckWParamMouseButton(!(rawButtons & RI_MOUSE_BUTTON_3_UP), mouseFlags, swapButtons, data, SDL_BUTTON_MIDDLE, mouseID);
        }
        if (rawButtons & RI_MOUSE_BUTTON_4_DOWN) {
            WIN_CheckWParamMouseButton((rawButtons & RI_MOUSE_BUTTON_4_DOWN), mouseFlags, swapButtons, data, SDL_BUTTON_X1, mouseID);
        }
        if (rawButtons & RI_MOUSE_BUTTON_4_UP) {
            WIN_CheckWParamMouseButton(!(rawButtons & RI_MOUSE_BUTTON_4_UP), mouseFlags, swapButtons, data, SDL_BUTTON_X1, mouseID);
        }
        if (rawButtons & RI_MOUSE_BUTTON_5_DOWN) {
            WIN_CheckWParamMouseButton((rawButtons & RI_MOUSE_BUTTON_5_DOWN), mouseFlags, swapButtons, data, SDL_BUTTON_X2, mouseID);
        }
        if (rawButtons & RI_MOUSE_BUTTON_5_UP) {
            WIN_CheckWParamMouseButton(!(rawButtons & RI_MOUSE_BUTTON_5_UP), mouseFlags, swapButtons, data, SDL_BUTTON_X2, mouseID);
        }
        data->mouse_button_flags = rawButtons;
    }
}

static void WIN_CheckAsyncMouseRelease(SDL_WindowData *data)
{
    Uint32 mouseFlags;
    SHORT keyState;
    SDL_bool swapButtons;

    /* mouse buttons may have changed state here, we need to resync them,
       but we will get a WM_MOUSEMOVE right away which will fix things up if in non raw mode also
    */
    mouseFlags = SDL_GetMouseState(NULL, NULL);
    swapButtons = GetSystemMetrics(SM_SWAPBUTTON) != 0;

    keyState = GetAsyncKeyState(VK_LBUTTON);
    if (!(keyState & 0x8000)) {
        WIN_CheckWParamMouseButton(SDL_FALSE, mouseFlags, swapButtons, data, SDL_BUTTON_LEFT, 0);
    }
    keyState = GetAsyncKeyState(VK_RBUTTON);
    if (!(keyState & 0x8000)) {
        WIN_CheckWParamMouseButton(SDL_FALSE, mouseFlags, swapButtons, data, SDL_BUTTON_RIGHT, 0);
    }
    keyState = GetAsyncKeyState(VK_MBUTTON);
    if (!(keyState & 0x8000)) {
        WIN_CheckWParamMouseButton(SDL_FALSE, mouseFlags, swapButtons, data, SDL_BUTTON_MIDDLE, 0);
    }
    keyState = GetAsyncKeyState(VK_XBUTTON1);
    if (!(keyState & 0x8000)) {
        WIN_CheckWParamMouseButton(SDL_FALSE, mouseFlags, swapButtons, data, SDL_BUTTON_X1, 0);
    }
    keyState = GetAsyncKeyState(VK_XBUTTON2);
    if (!(keyState & 0x8000)) {
        WIN_CheckWParamMouseButton(SDL_FALSE, mouseFlags, swapButtons, data, SDL_BUTTON_X2, 0);
    }
    data->mouse_button_flags = (WPARAM)-1;
}

static void WIN_UpdateFocus(SDL_Window *window, SDL_bool expect_focus)
{
    SDL_WindowData *data = (SDL_WindowData *)window->driverdata;
    HWND hwnd = data->hwnd;
    SDL_bool had_focus = (SDL_GetKeyboardFocus() == window) ? SDL_TRUE : SDL_FALSE;
    SDL_bool has_focus = (GetForegroundWindow() == hwnd) ? SDL_TRUE : SDL_FALSE;

    if (had_focus == has_focus || has_focus != expect_focus) {
        return;
    }

    if (has_focus) {
        POINT cursorPos;

        SDL_bool swapButtons = GetSystemMetrics(SM_SWAPBUTTON) != 0;
        if (GetAsyncKeyState(VK_LBUTTON)) {
            data->focus_click_pending |= !swapButtons ? SDL_BUTTON_LMASK : SDL_BUTTON_RMASK;
        }
        if (GetAsyncKeyState(VK_RBUTTON)) {
            data->focus_click_pending |= !swapButtons ? SDL_BUTTON_RMASK : SDL_BUTTON_LMASK;
        }
        if (GetAsyncKeyState(VK_MBUTTON)) {
            data->focus_click_pending |= SDL_BUTTON_MMASK;
        }
        if (GetAsyncKeyState(VK_XBUTTON1)) {
            data->focus_click_pending |= SDL_BUTTON_X1MASK;
        }
        if (GetAsyncKeyState(VK_XBUTTON2)) {
            data->focus_click_pending |= SDL_BUTTON_X2MASK;
        }

        SDL_SetKeyboardFocus(window);

        /* In relative mode we are guaranteed to have mouse focus if we have keyboard focus */
        if (!SDL_GetMouse()->relative_mode) {
            SDL_Point point;
            GetCursorPos(&cursorPos);
            ScreenToClient(hwnd, &cursorPos);
            point.x = cursorPos.x;
            point.y = cursorPos.y;
            WIN_ClientPointToSDL(data->window, &point.x, &point.y);
            SDL_SendMouseMotion(window, 0, 0, point.x, point.y);
        }

        WIN_CheckAsyncMouseRelease(data);
        WIN_UpdateClipCursor(window);

        /*
         * FIXME: Update keyboard state
         */
        WIN_CheckClipboardUpdate(data->videodata);

        SDL_ToggleModState(KMOD_CAPS, (GetKeyState(VK_CAPITAL) & 0x0001) ? SDL_TRUE : SDL_FALSE);
        SDL_ToggleModState(KMOD_NUM, (GetKeyState(VK_NUMLOCK) & 0x0001) ? SDL_TRUE : SDL_FALSE);
        SDL_ToggleModState(KMOD_SCROLL, (GetKeyState(VK_SCROLL) & 0x0001) ? SDL_TRUE : SDL_FALSE);

        WIN_UpdateWindowICCProfile(data->window, SDL_TRUE);
    } else {
        RECT rect;

        data->in_window_deactivation = SDL_TRUE;

        SDL_SetKeyboardFocus(NULL);
        /* In relative mode we are guaranteed to not have mouse focus if we don't have keyboard focus */
        if (SDL_GetMouse()->relative_mode) {
            SDL_SetMouseFocus(NULL);
        }
        WIN_ResetDeadKeys();

        if (GetClipCursor(&rect) && SDL_memcmp(&rect, &data->cursor_clipped_rect, sizeof(rect)) == 0) {
            ClipCursor(NULL);
            SDL_zero(data->cursor_clipped_rect);
        }

        data->in_window_deactivation = SDL_FALSE;
    }
}
#endif /*!defined(__XBOXONE__) && !defined(__XBOXSERIES__)*/

static BOOL WIN_ConvertUTF32toUTF8(UINT32 codepoint, char *text)
{
    if (codepoint <= 0x7F) {
        text[0] = (char)codepoint;
        text[1] = '\0';
    } else if (codepoint <= 0x7FF) {
        text[0] = 0xC0 | (char)((codepoint >> 6) & 0x1F);
        text[1] = 0x80 | (char)(codepoint & 0x3F);
        text[2] = '\0';
    } else if (codepoint <= 0xFFFF) {
        text[0] = 0xE0 | (char)((codepoint >> 12) & 0x0F);
        text[1] = 0x80 | (char)((codepoint >> 6) & 0x3F);
        text[2] = 0x80 | (char)(codepoint & 0x3F);
        text[3] = '\0';
    } else if (codepoint <= 0x10FFFF) {
        text[0] = 0xF0 | (char)((codepoint >> 18) & 0x0F);
        text[1] = 0x80 | (char)((codepoint >> 12) & 0x3F);
        text[2] = 0x80 | (char)((codepoint >> 6) & 0x3F);
        text[3] = 0x80 | (char)(codepoint & 0x3F);
        text[4] = '\0';
    } else {
        return SDL_FALSE;
    }
    return SDL_TRUE;
}

static BOOL WIN_ConvertUTF16toUTF8(UINT32 high_surrogate, UINT32 low_surrogate, char *text)
{
    const UINT32 SURROGATE_OFFSET = 0x10000 - (0xD800 << 10) - 0xDC00;
    const UINT32 codepoint = (high_surrogate << 10) + low_surrogate + SURROGATE_OFFSET;
    return WIN_ConvertUTF32toUTF8(codepoint, text);
}

static SDL_bool ShouldGenerateWindowCloseOnAltF4(void)
{
    return !SDL_GetHintBoolean(SDL_HINT_WINDOWS_NO_CLOSE_ON_ALT_F4, SDL_FALSE);
}

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
/* We want to generate mouse events from mouse and pen, and touch events from touchscreens */
#define MI_WP_SIGNATURE      0xFF515700
#define MI_WP_SIGNATURE_MASK 0xFFFFFF00
#define IsTouchEvent(dw)     ((dw)&MI_WP_SIGNATURE_MASK) == MI_WP_SIGNATURE

typedef enum
{
    SDL_MOUSE_EVENT_SOURCE_UNKNOWN,
    SDL_MOUSE_EVENT_SOURCE_MOUSE,
    SDL_MOUSE_EVENT_SOURCE_TOUCH,
    SDL_MOUSE_EVENT_SOURCE_PEN,
} SDL_MOUSE_EVENT_SOURCE;

static SDL_MOUSE_EVENT_SOURCE GetMouseMessageSource(void)
{
    LPARAM extrainfo = GetMessageExtraInfo();
    /* Mouse data (ignoring synthetic mouse events generated for touchscreens) */
    /* Versions below Vista will set the low 7 bits to the Mouse ID and don't use bit 7:
       Check bits 8-32 for the signature (which will indicate a Tablet PC Pen or Touch Device).
       Only check bit 7 when Vista and up(Cleared=Pen, Set=Touch(which we need to filter out)),
       when the signature is set. The Mouse ID will be zero for an actual mouse. */
    if (IsTouchEvent(extrainfo)) {
        if (extrainfo & 0x80) {
            return SDL_MOUSE_EVENT_SOURCE_TOUCH;
        } else {
            return SDL_MOUSE_EVENT_SOURCE_PEN;
        }
    }
    return SDL_MOUSE_EVENT_SOURCE_MOUSE;
}
#endif /*!defined(__XBOXONE__) && !defined(__XBOXSERIES__)*/

static SDL_WindowData *WIN_GetWindowDataFromHWND(HWND hwnd)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();
    SDL_Window *window;

    if (_this) {
        for (window = _this->windows; window; window = window->next) {
            SDL_WindowData *data = (SDL_WindowData *)window->driverdata;
            if (data && data->hwnd == hwnd) {
                return data;
            }
        }
    }
    return NULL;
}

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
LRESULT CALLBACK
WIN_KeyboardHookProc(int nCode, WPARAM wParam, LPARAM lParam)
{
    KBDLLHOOKSTRUCT *hookData = (KBDLLHOOKSTRUCT *)lParam;
    SDL_VideoData *data = SDL_GetVideoDevice()->driverdata;
    SDL_Scancode scanCode;

    if (nCode < 0 || nCode != HC_ACTION) {
        return CallNextHookEx(NULL, nCode, wParam, lParam);
    }
    if (hookData->scanCode == 0x21d) {
	    // Skip fake LCtrl when RAlt is pressed
	    return 1;
    }

    switch (hookData->vkCode) {
    case VK_LWIN:
        scanCode = SDL_SCANCODE_LGUI;
        break;
    case VK_RWIN:
        scanCode = SDL_SCANCODE_RGUI;
        break;
    case VK_LMENU:
        scanCode = SDL_SCANCODE_LALT;
        break;
    case VK_RMENU:
        scanCode = SDL_SCANCODE_RALT;
        break;
    case VK_LCONTROL:
        scanCode = SDL_SCANCODE_LCTRL;
        break;
    case VK_RCONTROL:
        scanCode = SDL_SCANCODE_RCTRL;
        break;

    /* These are required to intercept Alt+Tab and Alt+Esc on Windows 7 */
    case VK_TAB:
        scanCode = SDL_SCANCODE_TAB;
        break;
    case VK_ESCAPE:
        scanCode = SDL_SCANCODE_ESCAPE;
        break;

    default:
        return CallNextHookEx(NULL, nCode, wParam, lParam);
    }

    if (wParam == WM_KEYDOWN || wParam == WM_SYSKEYDOWN) {
        SDL_SendKeyboardKey(SDL_PRESSED, scanCode);
    } else {
        SDL_SendKeyboardKey(SDL_RELEASED, scanCode);

        /* If the key was down prior to our hook being installed, allow the
           key up message to pass normally the first time. This ensures other
           windows have a consistent view of the key state, and avoids keys
           being stuck down in those windows if they are down when the grab
           happens and raised while grabbed. */
        if (hookData->vkCode <= 0xFF && data->pre_hook_key_state[hookData->vkCode]) {
            data->pre_hook_key_state[hookData->vkCode] = 0;
            return CallNextHookEx(NULL, nCode, wParam, lParam);
        }
    }

    return 1;
}

#endif /*!defined(__XBOXONE__) && !defined(__XBOXSERIES__)*/


// Return 1 if spruious LCtrl is pressed
// LCtrl is sent when RAltGR is pressed
int skip_bad_lcrtl(WPARAM wParam, LPARAM lParam)
{
    MSG next_msg;
    DWORD msg_time;
    if (wParam != VK_CONTROL) {
        return 0;
    }
    // Is this an extended key (i.e. right key)?
    if (lParam & 0x01000000)
                return 0;

    // Here is a trick: "Alt Gr" sends LCTRL, then RALT. We only
    // want the RALT message, so we try to see if the next message
    // is a RALT message. In that case, this is a false LCTRL!
    msg_time = GetMessageTime();
    if (PeekMessage(&next_msg, NULL, 0, 0, PM_NOREMOVE)) {
        if (next_msg.message == WM_KEYDOWN ||
            next_msg.message == WM_SYSKEYDOWN) {
            if (next_msg.wParam == VK_MENU &&
                (next_msg.lParam & 0x01000000) &&
                next_msg.time == msg_time) {
                // Next message is a RALT down message, which
                // means that this is NOT a proper LCTRL message!
                return 1;
            }
        }
    }
    return 0;
}

LRESULT CALLBACK
WIN_WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    static SDL_bool s_ModalMoveResizeLoop;
    SDL_WindowData *data;
    LRESULT returnCode = -1;

    /* Send a SDL_SYSWMEVENT if the application wants them */
    if (SDL_GetEventState(SDL_SYSWMEVENT) == SDL_ENABLE) {
        SDL_SysWMmsg wmmsg;

        SDL_VERSION(&wmmsg.version);
        wmmsg.subsystem = SDL_SYSWM_WINDOWS;
        wmmsg.msg.win.hwnd = hwnd;
        wmmsg.msg.win.msg = msg;
        wmmsg.msg.win.wParam = wParam;
        wmmsg.msg.win.lParam = lParam;
        SDL_SendSysWMEvent(&wmmsg);
    }

    /* Get the window data for the window */
    data = WIN_GetWindowDataFromHWND(hwnd);
#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
    if (!data) {
        /* Fallback */
        data = (SDL_WindowData *)GetProp(hwnd, TEXT("SDL_WindowData"));
    }
#endif
    if (!data) {
        return CallWindowProc(DefWindowProc, hwnd, msg, wParam, lParam);
    }

#ifdef WMMSG_DEBUG
    {
        char message[1024];
        if (msg > MAX_WMMSG) {
            SDL_snprintf(message, sizeof(message), "Received windows message: %p UNKNOWN (%d) -- 0x%p, 0x%p\n", hwnd, msg, wParam, lParam);
        } else {
            SDL_snprintf(message, sizeof(message), "Received windows message: %p %s -- 0x%p, 0x%p\n", hwnd, wmtab[msg], wParam, lParam);
        }
        OutputDebugStringA(message);
    }
#endif /* WMMSG_DEBUG */

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
    if (IME_HandleMessage(hwnd, msg, wParam, &lParam, data->videodata)) {
        return 0;
    }
#endif

    switch (msg) {

    case WM_SHOWWINDOW:
    {
        if (wParam) {
            SDL_SendWindowEvent(data->window, SDL_WINDOWEVENT_SHOWN, 0, 0);
        } else {
            SDL_SendWindowEvent(data->window, SDL_WINDOWEVENT_HIDDEN, 0, 0);
        }
    } break;

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
    case WM_NCACTIVATE:
    {
        /* Don't immediately clip the cursor in case we're clicking minimize/maximize buttons */
        data->skip_update_clipcursor = SDL_TRUE;

        /* Update the focus here, since it's possible to get WM_ACTIVATE and WM_SETFOCUS without
           actually being the foreground window, but this appears to get called in all cases where
           the global foreground window changes to and from this window. */
        WIN_UpdateFocus(data->window, !!wParam);
    } break;

    case WM_ACTIVATE:
    {
        /* Update the focus in case we changed focus to a child window and then away from the application */
        WIN_UpdateFocus(data->window, !!LOWORD(wParam));
    } break;

    case WM_SETFOCUS:
    {
        /* Update the focus in case it's changing between top-level windows in the same application */
        WIN_UpdateFocus(data->window, SDL_TRUE);
    } break;

    case WM_KILLFOCUS:
    case WM_ENTERIDLE:
    {
        /* Update the focus in case it's changing between top-level windows in the same application */
        WIN_UpdateFocus(data->window, SDL_FALSE);
    } break;

    case WM_POINTERUPDATE:
    {
        data->last_pointer_update = lParam;
        break;
    }

    case WM_MOUSEMOVE:
    {
        SDL_Mouse *mouse = SDL_GetMouse();

        if (!data->mouse_tracked) {
            TRACKMOUSEEVENT trackMouseEvent;

            trackMouseEvent.cbSize = sizeof(TRACKMOUSEEVENT);
            trackMouseEvent.dwFlags = TME_LEAVE;
            trackMouseEvent.hwndTrack = data->hwnd;

            if (TrackMouseEvent(&trackMouseEvent)) {
                data->mouse_tracked = SDL_TRUE;
            }
        }

        if (!mouse->relative_mode || mouse->relative_mode_warp) {
            /* Only generate mouse events for real mouse */
            if (GetMouseMessageSource() != SDL_MOUSE_EVENT_SOURCE_TOUCH &&
                lParam != data->last_pointer_update) {
                int x = GET_X_LPARAM(lParam);
                int y = GET_Y_LPARAM(lParam);

                WIN_ClientPointToSDL(data->window, &x, &y);

                SDL_SendMouseMotion(data->window, 0, 0, x, y);
            }
        }
    } break;

    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
    case WM_MBUTTONUP:
    case WM_XBUTTONUP:
    case WM_LBUTTONDOWN:
    case WM_LBUTTONDBLCLK:
    case WM_RBUTTONDOWN:
    case WM_RBUTTONDBLCLK:
    case WM_MBUTTONDOWN:
    case WM_MBUTTONDBLCLK:
    case WM_XBUTTONDOWN:
    case WM_XBUTTONDBLCLK:
    {
        SDL_Mouse *mouse = SDL_GetMouse();
        if (!mouse->relative_mode || mouse->relative_mode_warp) {
            if (GetMouseMessageSource() != SDL_MOUSE_EVENT_SOURCE_TOUCH &&
                lParam != data->last_pointer_update) {
                WIN_CheckWParamMouseButtons(wParam, data, 0);
            }
        }
    } break;

    case WM_INPUT:
    {
        SDL_Mouse *mouse = SDL_GetMouse();
        HRAWINPUT hRawInput = (HRAWINPUT)lParam;
        RAWINPUT inp;
        UINT size = sizeof(inp);

        /* We only use raw mouse input in relative mode */
        if (!mouse->relative_mode || mouse->relative_mode_warp) {
            break;
        }

        /* Relative mouse motion is delivered to the window with keyboard focus */
        if (data->window != SDL_GetKeyboardFocus()) {
            break;
        }

        GetRawInputData(hRawInput, RID_INPUT, &inp, &size, sizeof(RAWINPUTHEADER));

        /* Mouse data (ignoring synthetic mouse events generated for touchscreens) */
        if (inp.header.dwType == RIM_TYPEMOUSE) {
            SDL_MouseID mouseID;
            RAWMOUSE *rawmouse;
            if (SDL_GetNumTouchDevices() > 0 &&
                (GetMouseMessageSource() == SDL_MOUSE_EVENT_SOURCE_TOUCH || (GetMessageExtraInfo() & 0x80) == 0x80)) {
                break;
            }
            /* We do all of our mouse state checking against mouse ID 0
             * We would only use the actual hDevice if we were tracking
             * all mouse motion independently, and never using mouse ID 0.
             */
            mouseID = 0; /* (SDL_MouseID)(uintptr_t)inp.header.hDevice; */
            rawmouse = &inp.data.mouse;

            if ((rawmouse->usFlags & 0x01) == MOUSE_MOVE_RELATIVE) {
                SDL_SendMouseMotion(data->window, mouseID, 1, (int)rawmouse->lLastX, (int)rawmouse->lLastY);
            } else if (rawmouse->lLastX || rawmouse->lLastY) {
                /* This is absolute motion, either using a tablet or mouse over RDP

                   Notes on how RDP appears to work, as of Windows 10 2004:
                    - SetCursorPos() calls are cached, with multiple calls coalesced into a single call that's sent to the RDP client. If the last call to SetCursorPos() has the same value as the last one that was sent to the client, it appears to be ignored and not sent. This means that we need to jitter the SetCursorPos() position slightly in order for the recentering to work correctly.
                    - User mouse motion is coalesced with SetCursorPos(), so the WM_INPUT positions we see will not necessarily match the positon we requested with SetCursorPos().
                    - SetCursorPos() outside of the bounds of the focus window appears not to do anything.
                    - SetCursorPos() while the cursor is NULL doesn't do anything

                   We handle this by creating a safe area within the application window, and when the mouse leaves that safe area, we warp back to the opposite side. Any single motion > 50% of the safe area is assumed to be a warp and ignored.
                */
                SDL_bool remote_desktop = GetSystemMetrics(SM_REMOTESESSION) ? SDL_TRUE : SDL_FALSE;
                SDL_bool virtual_desktop = (rawmouse->usFlags & MOUSE_VIRTUAL_DESKTOP) ? SDL_TRUE : SDL_FALSE;
                SDL_bool normalized_coordinates = !(rawmouse->usFlags & 0x40) ? SDL_TRUE : SDL_FALSE;
                int w = GetSystemMetrics(virtual_desktop ? SM_CXVIRTUALSCREEN : SM_CXSCREEN);
                int h = GetSystemMetrics(virtual_desktop ? SM_CYVIRTUALSCREEN : SM_CYSCREEN);
                int x = normalized_coordinates ? (int)(((float)rawmouse->lLastX / 65535.0f) * w) : (int)rawmouse->lLastX;
                int y = normalized_coordinates ? (int)(((float)rawmouse->lLastY / 65535.0f) * h) : (int)rawmouse->lLastY;
                int relX, relY;

                /* Calculate relative motion */
                if (data->last_raw_mouse_position.x == 0 && data->last_raw_mouse_position.y == 0) {
                    data->last_raw_mouse_position.x = x;
                    data->last_raw_mouse_position.y = y;
                }
                relX = x - data->last_raw_mouse_position.x;
                relY = y - data->last_raw_mouse_position.y;

                if (remote_desktop) {
                    if (!data->in_title_click && !data->focus_click_pending) {
                        static int wobble;
                        float floatX = (float)x / w;
                        float floatY = (float)y / h;

                        /* See if the mouse is at the edge of the screen, or in the RDP title bar area */
                        if (floatX <= 0.01f || floatX >= 0.99f || floatY <= 0.01f || floatY >= 0.99f || y < 32) {
                            /* Wobble the cursor position so it's not ignored if the last warp didn't have any effect */
                            RECT rect = data->cursor_clipped_rect;
                            int warpX = rect.left + ((rect.right - rect.left) / 2) + wobble;
                            int warpY = rect.top + ((rect.bottom - rect.top) / 2);

                            WIN_SetCursorPos(warpX, warpY);

                            ++wobble;
                            if (wobble > 1) {
                                wobble = -1;
                            }
                        } else {
                            /* Send relative motion if we didn't warp last frame (had good position data)
                               We also sometimes get large deltas due to coalesced mouse motion and warping,
                               so ignore those.
                             */
                            const int MAX_RELATIVE_MOTION = (h / 6);
                            if (SDL_abs(relX) < MAX_RELATIVE_MOTION &&
                                SDL_abs(relY) < MAX_RELATIVE_MOTION) {
                                SDL_SendMouseMotion(data->window, mouseID, 1, relX, relY);
                            }
                        }
                    }
                } else {
                    const int MAXIMUM_TABLET_RELATIVE_MOTION = 32;
                    if (SDL_abs(relX) > MAXIMUM_TABLET_RELATIVE_MOTION ||
                        SDL_abs(relY) > MAXIMUM_TABLET_RELATIVE_MOTION) {
                        /* Ignore this motion, probably a pen lift and drop */
                    } else {
                        SDL_SendMouseMotion(data->window, mouseID, 1, relX, relY);
                    }
                }

                data->last_raw_mouse_position.x = x;
                data->last_raw_mouse_position.y = y;
            }
            WIN_CheckRawMouseButtons(inp.header.hDevice, rawmouse->usButtonFlags, data, mouseID);
        }
    } break;

    case WM_MOUSEWHEEL:
    case WM_MOUSEHWHEEL:
    {
        short amount = GET_WHEEL_DELTA_WPARAM(wParam);
        float fAmount = (float)amount / WHEEL_DELTA;
        if (msg == WM_MOUSEWHEEL) {
            SDL_SendMouseWheel(data->window, 0, 0.0f, fAmount, SDL_MOUSEWHEEL_NORMAL);
        } else {
            SDL_SendMouseWheel(data->window, 0, fAmount, 0.0f, SDL_MOUSEWHEEL_NORMAL);
        }
    } break;

    case WM_MOUSELEAVE:
        if (!(data->window->flags & SDL_WINDOW_MOUSE_CAPTURE)) {
            if (SDL_GetMouseFocus() == data->window && !SDL_GetMouse()->relative_mode && !IsIconic(hwnd)) {
                SDL_Mouse *mouse;
                POINT cursorPos;
                SDL_Point point;
                GetCursorPos(&cursorPos);
                ScreenToClient(hwnd, &cursorPos);
                point.x = cursorPos.x;
                point.y = cursorPos.y;
                WIN_ClientPointToSDL(data->window, &point.x, &point.y);
                mouse = SDL_GetMouse();
                if (!mouse->was_touch_mouse_events) { /* we're not a touch handler causing a mouse leave? */
                    SDL_SendMouseMotion(data->window, 0, 0, point.x, point.y);
                } else {                                       /* touch handling? */
                    mouse->was_touch_mouse_events = SDL_FALSE; /* not anymore */
                    if (mouse->touch_mouse_events) {           /* convert touch to mouse events */
                        SDL_SendMouseMotion(data->window, SDL_TOUCH_MOUSEID, 0, point.x, point.y);
                    } else { /* normal handling */
                        SDL_SendMouseMotion(data->window, 0, 0, point.x, point.y);
                    }
                }
            }

            if (!SDL_GetMouse()->relative_mode) {
                /* When WM_MOUSELEAVE is fired we can be assured that the cursor has left the window */
                SDL_SetMouseFocus(NULL);
            }
        }

        /* Once we get WM_MOUSELEAVE we're guaranteed that the window is no longer tracked */
        data->mouse_tracked = SDL_FALSE;

        returnCode = 0;
        break;
#endif /*!defined(__XBOXONE__) && !defined(__XBOXSERIES__)*/

    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
    {
        SDL_Scancode code = WindowsScanCodeToSDLScanCode(lParam, wParam);
        const Uint8 *keyboardState = SDL_GetKeyboardState(NULL);

        if (skip_bad_lcrtl(wParam, lParam)) {
            returnCode = 0;
            break;
        }

        /* Detect relevant keyboard shortcuts */
        if (keyboardState[SDL_SCANCODE_LALT] == SDL_PRESSED || keyboardState[SDL_SCANCODE_RALT] == SDL_PRESSED) {
            /* ALT+F4: Close window */
            if (code == SDL_SCANCODE_F4 && ShouldGenerateWindowCloseOnAltF4()) {
                SDL_SendWindowEvent(data->window, SDL_WINDOWEVENT_CLOSE, 0, 0);
            }
        }

        if (code != SDL_SCANCODE_UNKNOWN) {
            SDL_SendKeyboardKey(SDL_PRESSED, code);
        }
    }

        returnCode = 0;
        break;

    case WM_SYSKEYUP:
    case WM_KEYUP:
    {
        SDL_Scancode code = WindowsScanCodeToSDLScanCode(lParam, wParam);
        const Uint8 *keyboardState = SDL_GetKeyboardState(NULL);

        if (skip_bad_lcrtl(wParam, lParam)) {
            returnCode = 0;
            break;
        }

        if (code != SDL_SCANCODE_UNKNOWN) {
            if (code == SDL_SCANCODE_PRINTSCREEN &&
                keyboardState[code] == SDL_RELEASED) {
                SDL_SendKeyboardKey(SDL_PRESSED, code);
            }
            SDL_SendKeyboardKey(SDL_RELEASED, code);
        }
    }
        returnCode = 0;
        break;

    case WM_UNICHAR:
        if (wParam == UNICODE_NOCHAR) {
            returnCode = 1;
        } else {
            char text[5];
            if (WIN_ConvertUTF32toUTF8((UINT32)wParam, text)) {
                SDL_SendKeyboardText(text);
            }
            returnCode = 0;
        }
        break;

    case WM_CHAR:
        /* When a user enters a Unicode code point defined in the Basic Multilingual Plane, Windows sends a WM_CHAR
           message with the code point encoded as UTF-16. When a user enters a Unicode code point from a Supplementary
           Plane, Windows sends the code point in two separate WM_CHAR messages: The first message includes the UTF-16
           High Surrogate and the second the UTF-16 Low Surrogate. The High and Low Surrogates cannot be individually
           converted to valid UTF-8, therefore, we must save the High Surrogate from the first WM_CHAR message and
           concatenate it with the Low Surrogate from the second WM_CHAR message. At that point, we have a valid
           UTF-16 surrogate pair ready to re-encode as UTF-8. */
        if (IS_HIGH_SURROGATE(wParam)) {
            data->high_surrogate = (WCHAR)wParam;
        } else if (IS_SURROGATE_PAIR(data->high_surrogate, wParam)) {
            /* The code point is in a Supplementary Plane.
               Here wParam is the Low Surrogate. */
            char text[5];
            if (WIN_ConvertUTF16toUTF8((UINT32)data->high_surrogate, (UINT32)wParam, text)) {
                SDL_SendKeyboardText(text);
            }
            data->high_surrogate = 0;
        } else {
            /* The code point is in the Basic Multilingual Plane.
               It's numerically equal to UTF-32. */
            char text[5];
            if (WIN_ConvertUTF32toUTF8((UINT32)wParam, text)) {
                SDL_SendKeyboardText(text);
            }
        }
        returnCode = 0;
        break;

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
#ifdef WM_INPUTLANGCHANGE
    case WM_INPUTLANGCHANGE:
    {
        WIN_UpdateKeymap(SDL_TRUE);
    }
        returnCode = 1;
        break;
#endif /* WM_INPUTLANGCHANGE */

    case WM_NCLBUTTONDOWN:
    {
        data->in_title_click = SDL_TRUE;
    } break;

    case WM_CAPTURECHANGED:
    {
        data->in_title_click = SDL_FALSE;

        /* The mouse may have been released during a modal loop */
        WIN_CheckAsyncMouseRelease(data);
    } break;

#ifdef WM_GETMINMAXINFO
    case WM_GETMINMAXINFO:
    {
        MINMAXINFO *info;
        RECT size;
        int x, y;
        int w, h;
        int min_w, min_h;
        int max_w, max_h;
        BOOL constrain_max_size;

        if (SDL_IsShapedWindow(data->window)) {
            Win32_ResizeWindowShape(data->window);
        }

        /* If this is an expected size change, allow it */
        if (data->expected_resize) {
            break;
        }

        /* Get the current position of our window */
        GetWindowRect(hwnd, &size);
        x = size.left;
        y = size.top;

        /* Calculate current size of our window */
        SDL_GetWindowSize(data->window, &w, &h);
        SDL_GetWindowMinimumSize(data->window, &min_w, &min_h);
        SDL_GetWindowMaximumSize(data->window, &max_w, &max_h);

        /* Convert w, h, min_w, min_h, max_w, max_h from dpi-scaled points to pixels,
           treating them as coordinates within the client area. */
        WIN_ClientPointFromSDL(data->window, &w, &h);
        WIN_ClientPointFromSDL(data->window, &min_w, &min_h);
        WIN_ClientPointFromSDL(data->window, &max_w, &max_h);

        /* Store in min_w and min_h difference between current size and minimal
           size so we don't need to call AdjustWindowRectEx twice */
        min_w -= w;
        min_h -= h;
        if (max_w && max_h) {
            max_w -= w;
            max_h -= h;
            constrain_max_size = TRUE;
        } else {
            constrain_max_size = FALSE;
        }

        if (!(SDL_GetWindowFlags(data->window) & SDL_WINDOW_BORDERLESS)) {
            LONG style = GetWindowLong(hwnd, GWL_STYLE);
            /* DJM - according to the docs for GetMenu(), the
               return value is undefined if hwnd is a child window.
               Apparently it's too difficult for MS to check
               inside their function, so I have to do it here.
             */
            BOOL menu = (style & WS_CHILDWINDOW) ? FALSE : (GetMenu(hwnd) != NULL);
            UINT dpi;

            dpi = 96;
            size.top = 0;
            size.left = 0;
            size.bottom = h;
            size.right = w;

            if (WIN_IsPerMonitorV2DPIAware(SDL_GetVideoDevice())) {
                dpi = data->videodata->GetDpiForWindow(hwnd);
                data->videodata->AdjustWindowRectExForDpi(&size, style, menu, 0, dpi);
            } else {
                AdjustWindowRectEx(&size, style, menu, 0);
            }
            w = size.right - size.left;
            h = size.bottom - size.top;
#ifdef HIGHDPI_DEBUG
            SDL_Log("WM_GETMINMAXINFO: max window size: %dx%d using dpi: %u", w, h, dpi);
#endif
        }

        /* Fix our size to the current size */
        info = (MINMAXINFO *)lParam;
        if (SDL_GetWindowFlags(data->window) & SDL_WINDOW_RESIZABLE) {
            if (SDL_GetWindowFlags(data->window) & SDL_WINDOW_BORDERLESS) {
                int screenW = GetSystemMetrics(SM_CXSCREEN);
                int screenH = GetSystemMetrics(SM_CYSCREEN);
                info->ptMaxSize.x = SDL_max(w, screenW);
                info->ptMaxSize.y = SDL_max(h, screenH);
                info->ptMaxPosition.x = SDL_min(0, ((screenW - w) / 2));
                info->ptMaxPosition.y = SDL_min(0, ((screenH - h) / 2));
            }
            info->ptMinTrackSize.x = (LONG)w + min_w;
            info->ptMinTrackSize.y = (LONG)h + min_h;
            if (constrain_max_size) {
                info->ptMaxTrackSize.x = (LONG)w + max_w;
                info->ptMaxTrackSize.y = (LONG)h + max_h;
            }
        } else {
            info->ptMaxSize.x = w;
            info->ptMaxSize.y = h;
            info->ptMaxPosition.x = x;
            info->ptMaxPosition.y = y;
            info->ptMinTrackSize.x = w;
            info->ptMinTrackSize.y = h;
            info->ptMaxTrackSize.x = w;
            info->ptMaxTrackSize.y = h;
        }
    }
        returnCode = 0;
        break;
#endif /* WM_GETMINMAXINFO */

    case WM_WINDOWPOSCHANGING:

        if (data->expected_resize) {
            returnCode = 0;
        }
        break;

    case WM_WINDOWPOSCHANGED:
    {
        RECT rect;
        int x, y;
        int w, h;
        int display_index = data->window->display_index;

        if (data->initializing || data->in_border_change) {
            break;
        }

        /* When the window is minimized it's resized to the dock icon size, ignore this */
        if (IsIconic(hwnd)) {
            break;
        }

        if (!GetClientRect(hwnd, &rect) || WIN_IsRectEmpty(&rect)) {
            break;
        }
        ClientToScreen(hwnd, (LPPOINT)&rect);
        ClientToScreen(hwnd, (LPPOINT)&rect + 1);

        WIN_UpdateClipCursor(data->window);

        x = rect.left;
        y = rect.top;
        WIN_ScreenPointToSDL(&x, &y);

        SDL_SendWindowEvent(data->window, SDL_WINDOWEVENT_MOVED, x, y);

        /* Convert client area width/height from pixels to dpi-scaled points */
        w = rect.right - rect.left;
        h = rect.bottom - rect.top;
        WIN_ClientPointToSDL(data->window, &w, &h);

        SDL_SendWindowEvent(data->window, SDL_WINDOWEVENT_RESIZED, w, h);

#ifdef HIGHDPI_DEBUG
        SDL_Log("WM_WINDOWPOSCHANGED: Windows client rect (pixels): (%d, %d) (%d x %d)\tSDL client rect (points): (%d, %d) (%d x %d) cached dpi %d, windows reported dpi %d",
                rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top,
                x, y, w, h, data->scaling_dpi, data->videodata->GetDpiForWindow ? data->videodata->GetDpiForWindow(data->hwnd) : 0);
#endif

        /* Forces a WM_PAINT event */
        InvalidateRect(hwnd, NULL, FALSE);

        if (data->window->display_index != display_index) {
            /* Display changed, check ICC profile */
            WIN_UpdateWindowICCProfile(data->window, SDL_TRUE);
        }
    } break;

    case WM_ENTERSIZEMOVE:
    case WM_ENTERMENULOOP:
    {
        SetTimer(hwnd, (UINT_PTR)&s_ModalMoveResizeLoop, USER_TIMER_MINIMUM, NULL);
    } break;

    case WM_TIMER:
    {
        if (wParam == (UINT_PTR)&s_ModalMoveResizeLoop) {
            SDL_OnWindowLiveResizeUpdate(data->window);
            return 0;
        }
    } break;

    case WM_EXITSIZEMOVE:
    case WM_EXITMENULOOP:
    {
        KillTimer(hwnd, (UINT_PTR)&s_ModalMoveResizeLoop);
    } break;

    case WM_SIZE:
    {
        switch (wParam) {
        case SIZE_MAXIMIZED:
            SDL_SendWindowEvent(data->window,
                                SDL_WINDOWEVENT_RESTORED, 0, 0);
            SDL_SendWindowEvent(data->window,
                                SDL_WINDOWEVENT_MAXIMIZED, 0, 0);
            break;
        case SIZE_MINIMIZED:
            SDL_SendWindowEvent(data->window,
                                SDL_WINDOWEVENT_MINIMIZED, 0, 0);
            break;
        case SIZE_RESTORED:
            SDL_SendWindowEvent(data->window,
                                SDL_WINDOWEVENT_RESTORED, 0, 0);
            break;
        default:
            break;
        }
    } break;

    case WM_SETCURSOR:
    {
        Uint16 hittest;

        hittest = LOWORD(lParam);
        if (hittest == HTCLIENT) {
            SetCursor(SDL_cursor);
            returnCode = TRUE;
        } else if (!g_WindowFrameUsableWhileCursorHidden && !SDL_cursor) {
            SetCursor(NULL);
            returnCode = TRUE;
        }
    } break;

        /* We were occluded, refresh our display */
    case WM_PAINT:
    {
        RECT rect;
        if (GetUpdateRect(hwnd, &rect, FALSE)) {
            ValidateRect(hwnd, NULL);
            SDL_SendWindowEvent(data->window, SDL_WINDOWEVENT_EXPOSED, 0, 0);
        }
    }
        returnCode = 0;
        break;

        /* We'll do our own drawing, prevent flicker */
    case WM_ERASEBKGND:
        if (!data->videodata->cleared) {
            RECT client_rect;
            HBRUSH brush;
            data->videodata->cleared = SDL_TRUE;
            GetClientRect(hwnd, &client_rect);
            brush = CreateSolidBrush(0);
            FillRect(GetDC(hwnd), &client_rect, brush);
            DeleteObject(brush);
        }
        return 1;

    case WM_SYSCOMMAND:
    {
        if (!g_WindowsEnableMenuMnemonics) {
            if ((wParam & 0xFFF0) == SC_KEYMENU) {
                return 0;
            }
        }

#if defined(SC_SCREENSAVE) || defined(SC_MONITORPOWER)
        /* Don't start the screensaver or blank the monitor in fullscreen apps */
        if ((wParam & 0xFFF0) == SC_SCREENSAVE ||
            (wParam & 0xFFF0) == SC_MONITORPOWER) {
            if (SDL_GetVideoDevice()->suspend_screensaver) {
                return 0;
            }
        }
#endif /* System has screensaver support */
    } break;

    case WM_CLOSE:
    {
        SDL_SendWindowEvent(data->window, SDL_WINDOWEVENT_CLOSE, 0, 0);
    }
        returnCode = 0;
        break;

    case WM_TOUCH:
        if (data->videodata->GetTouchInputInfo && data->videodata->CloseTouchInputHandle) {
            UINT i, num_inputs = LOWORD(wParam);
            SDL_bool isstack;
            PTOUCHINPUT inputs = SDL_small_alloc(TOUCHINPUT, num_inputs, &isstack);
            if (data->videodata->GetTouchInputInfo((HTOUCHINPUT)lParam, num_inputs, inputs, sizeof(TOUCHINPUT))) {
                RECT rect;
                float x, y;

                if (!GetClientRect(hwnd, &rect) || WIN_IsRectEmpty(&rect)) {
                    if (inputs) {
                        SDL_small_free(inputs, isstack);
                    }
                    break;
                }
                ClientToScreen(hwnd, (LPPOINT)&rect);
                ClientToScreen(hwnd, (LPPOINT)&rect + 1);
                rect.top *= 100;
                rect.left *= 100;
                rect.bottom *= 100;
                rect.right *= 100;

                for (i = 0; i < num_inputs; ++i) {
                    PTOUCHINPUT input = &inputs[i];

                    const SDL_TouchID touchId = (SDL_TouchID)((size_t)input->hSource);

                    /* TODO: Can we use GetRawInputDeviceInfo and HID info to
                       determine if this is a direct or indirect touch device?
                     */
                    if (SDL_AddTouch(touchId, SDL_TOUCH_DEVICE_DIRECT, (input->dwFlags & TOUCHEVENTF_PEN) == TOUCHEVENTF_PEN ? "pen" : "touch") < 0) {
                        continue;
                    }

                    /* Get the normalized coordinates for the window */
                    x = (float)(input->x - rect.left) / (rect.right - rect.left);
                    y = (float)(input->y - rect.top) / (rect.bottom - rect.top);

                    if (input->dwFlags & TOUCHEVENTF_DOWN) {
                        SDL_SendTouch(touchId, input->dwID, data->window, SDL_TRUE, x, y, 1.0f);
                    }
                    if (input->dwFlags & TOUCHEVENTF_MOVE) {
                        SDL_SendTouchMotion(touchId, input->dwID, data->window, x, y, 1.0f);
                    }
                    if (input->dwFlags & TOUCHEVENTF_UP) {
                        SDL_SendTouch(touchId, input->dwID, data->window, SDL_FALSE, x, y, 1.0f);
                    }
                }
            }
            SDL_small_free(inputs, isstack);

            data->videodata->CloseTouchInputHandle((HTOUCHINPUT)lParam);
            return 0;
        }
        break;

#ifdef HAVE_TPCSHRD_H

    case WM_TABLET_QUERYSYSTEMGESTURESTATUS:
        /* See https://msdn.microsoft.com/en-us/library/windows/desktop/bb969148(v=vs.85).aspx .
         * If we're handling our own touches, we don't want any gestures.
         * Not all of these settings are documented.
         * The use of the undocumented ones was suggested by https://github.com/bjarkeck/GCGJ/blob/master/Monogame/Windows/WinFormsGameForm.cs . */
        return TABLET_DISABLE_PRESSANDHOLD | TABLET_DISABLE_PENTAPFEEDBACK | TABLET_DISABLE_PENBARRELFEEDBACK | TABLET_DISABLE_TOUCHUIFORCEON | TABLET_DISABLE_TOUCHUIFORCEOFF | TABLET_DISABLE_TOUCHSWITCH | TABLET_DISABLE_FLICKS | TABLET_DISABLE_SMOOTHSCROLLING | TABLET_DISABLE_FLICKFALLBACKKEYS; /*  disables press and hold (right-click) gesture */
                                                                                                                                                                                                                                                                                                         /*  disables UI feedback on pen up (waves) */
                                                                                                                                                                                                                                                                                                         /*  disables UI feedback on pen button down (circle) */
                                                                                                                                                                                                                                                                                                         /*  disables pen flicks (back, forward, drag down, drag up) */

#endif /* HAVE_TPCSHRD_H */

    case WM_DROPFILES:
    {
        UINT i;
        HDROP drop = (HDROP)wParam;
        UINT count = DragQueryFile(drop, 0xFFFFFFFF, NULL, 0);
        for (i = 0; i < count; ++i) {
            SDL_bool isstack;
            UINT size = DragQueryFile(drop, i, NULL, 0) + 1;
            LPTSTR buffer = SDL_small_alloc(TCHAR, size, &isstack);
            if (buffer) {
                if (DragQueryFile(drop, i, buffer, size)) {
                    char *file = WIN_StringToUTF8(buffer);
                    SDL_SendDropFile(data->window, file);
                    SDL_free(file);
                }
                SDL_small_free(buffer, isstack);
            }
        }
        SDL_SendDropComplete(data->window);
        DragFinish(drop);
        return 0;
    } break;

    case WM_DISPLAYCHANGE:
    {
        // Reacquire displays if any were added or removed
        WIN_RefreshDisplays(SDL_GetVideoDevice());
    } break;

    case WM_NCCALCSIZE:
    {
        Uint32 window_flags = SDL_GetWindowFlags(data->window);
        if (wParam == TRUE && (window_flags & SDL_WINDOW_BORDERLESS) && !(window_flags & SDL_WINDOW_FULLSCREEN)) {
            /* When borderless, need to tell windows that the size of the non-client area is 0 */
            if (!(window_flags & SDL_WINDOW_RESIZABLE)) {
                int w, h;
                NCCALCSIZE_PARAMS *params = (NCCALCSIZE_PARAMS *)lParam;
                w = data->window->windowed.w;
                h = data->window->windowed.h;
                WIN_ClientPointFromSDL(data->window, &w, &h);
                params->rgrc[0].right = params->rgrc[0].left + w;
                params->rgrc[0].bottom = params->rgrc[0].top + h;
            }
            return 0;
        }
    } break;

    case WM_NCHITTEST:
    {
        SDL_Window *window = data->window;
        if (window->hit_test) {
            POINT winpoint;
            winpoint.x = GET_X_LPARAM(lParam);
            winpoint.y = GET_Y_LPARAM(lParam);
            if (ScreenToClient(hwnd, &winpoint)) {
                SDL_Point point;
                SDL_HitTestResult rc;
                point.x = winpoint.x;
                point.y = winpoint.y;
                WIN_ClientPointToSDL(data->window, &point.x, &point.y);
                rc = window->hit_test(window, &point, window->hit_test_data);
                switch (rc) {
#define POST_HIT_TEST(ret)                                                 \
    {                                                                      \
        SDL_SendWindowEvent(data->window, SDL_WINDOWEVENT_HIT_TEST, 0, 0); \
        return ret;                                                        \
    }
                case SDL_HITTEST_DRAGGABLE:
                    POST_HIT_TEST(HTCAPTION);
                case SDL_HITTEST_RESIZE_TOPLEFT:
                    POST_HIT_TEST(HTTOPLEFT);
                case SDL_HITTEST_RESIZE_TOP:
                    POST_HIT_TEST(HTTOP);
                case SDL_HITTEST_RESIZE_TOPRIGHT:
                    POST_HIT_TEST(HTTOPRIGHT);
                case SDL_HITTEST_RESIZE_RIGHT:
                    POST_HIT_TEST(HTRIGHT);
                case SDL_HITTEST_RESIZE_BOTTOMRIGHT:
                    POST_HIT_TEST(HTBOTTOMRIGHT);
                case SDL_HITTEST_RESIZE_BOTTOM:
                    POST_HIT_TEST(HTBOTTOM);
                case SDL_HITTEST_RESIZE_BOTTOMLEFT:
                    POST_HIT_TEST(HTBOTTOMLEFT);
                case SDL_HITTEST_RESIZE_LEFT:
                    POST_HIT_TEST(HTLEFT);
#undef POST_HIT_TEST
                case SDL_HITTEST_NORMAL:
                    return HTCLIENT;
                }
            }
            /* If we didn't return, this will call DefWindowProc below. */
        }
    } break;

    case WM_GETDPISCALEDSIZE:
        /* Windows 10 Creators Update+ */
        /* Documented as only being sent to windows that are per-monitor V2 DPI aware.

           Experimentation shows it's only sent during interactive dragging, not in response to
           SetWindowPos. */
        if (data->videodata->GetDpiForWindow && data->videodata->AdjustWindowRectExForDpi) {
            /* Windows expects applications to scale their window rects linearly
               when dragging between monitors with different DPI's.
               e.g. a 100x100 window dragged to a 200% scaled monitor
               becomes 200x200.

               For SDL, we instead want the client size to scale linearly.
               This is not the same as the window rect scaling linearly,
               because Windows doesn't scale the non-client area (titlebar etc.)
               linearly. So, we need to handle this message to request custom
               scaling. */

            const int nextDPI = (int)wParam;
            const int prevDPI = (int)data->videodata->GetDpiForWindow(hwnd);
            SIZE *sizeInOut = (SIZE *)lParam;

            int frame_w, frame_h;
            int query_client_w_win, query_client_h_win;

            const DWORD style = GetWindowLong(hwnd, GWL_STYLE);
            const BOOL menu = (style & WS_CHILDWINDOW) ? FALSE : (GetMenu(hwnd) != NULL);

#ifdef HIGHDPI_DEBUG
            SDL_Log("WM_GETDPISCALEDSIZE: current DPI: %d potential DPI: %d input size: (%dx%d)",
                    prevDPI, nextDPI, sizeInOut->cx, sizeInOut->cy);
#endif

            /* Subtract the window frame size that would have been used at prevDPI */
            {
                RECT rect = { 0 };

                if (!(data->window->flags & SDL_WINDOW_BORDERLESS)) {
                    data->videodata->AdjustWindowRectExForDpi(&rect, style, menu, 0, prevDPI);
                }

                frame_w = -rect.left + rect.right;
                frame_h = -rect.top + rect.bottom;

                query_client_w_win = sizeInOut->cx - frame_w;
                query_client_h_win = sizeInOut->cy - frame_h;
            }

            /* Convert to new dpi if we are using scaling.
             * Otherwise leave as pixels.
             */
            if (data->videodata->dpi_scaling_enabled) {
                query_client_w_win = MulDiv(query_client_w_win, nextDPI, prevDPI);
                query_client_h_win = MulDiv(query_client_h_win, nextDPI, prevDPI);
            }

            /* Add the window frame size that would be used at nextDPI */
            {
                RECT rect = { 0 };
                rect.right = query_client_w_win;
                rect.bottom = query_client_h_win;

                if (!(data->window->flags & SDL_WINDOW_BORDERLESS)) {
                    data->videodata->AdjustWindowRectExForDpi(&rect, style, menu, 0, nextDPI);
                }

                /* This is supposed to control the suggested rect param of WM_DPICHANGED */
                sizeInOut->cx = rect.right - rect.left;
                sizeInOut->cy = rect.bottom - rect.top;
            }

#ifdef HIGHDPI_DEBUG
            SDL_Log("WM_GETDPISCALEDSIZE: output size: (%dx%d)", sizeInOut->cx, sizeInOut->cy);
#endif
            return TRUE;
        }
        break;

    case WM_DPICHANGED:
        /* Windows 8.1+ */
        {
            const int newDPI = HIWORD(wParam);
            RECT *const suggestedRect = (RECT *)lParam;
            int w, h;

#ifdef HIGHDPI_DEBUG
            SDL_Log("WM_DPICHANGED: to %d\tsuggested rect: (%d, %d), (%dx%d)\n", newDPI,
                    suggestedRect->left, suggestedRect->top, suggestedRect->right - suggestedRect->left, suggestedRect->bottom - suggestedRect->top);
#endif

            if (data->expected_resize) {
                /* This DPI change is coming from an explicit SetWindowPos call within SDL.
                   Assume all call sites are calculating the DPI-aware frame correctly, so
                   we don't need to do any further adjustment. */

                if (data->videodata->dpi_scaling_enabled) {
                    /* Update the cached DPI value for this window */
                    data->scaling_dpi = newDPI;

                    /* Send a SDL_WINDOWEVENT_SIZE_CHANGED saying that the client size (in dpi-scaled points) is unchanged.
                       Renderers need to get this to know that the framebuffer size changed.

                       We clear the window size to force the event to be delivered, but what we really
                       want for SDL3 is a new event to notify that the DPI changed and then watch for
                       that in the renderer directly.
                     */
                    data->window->w = 0;
                    data->window->h = 0;
                    SDL_SendWindowEvent(data->window, SDL_WINDOWEVENT_SIZE_CHANGED, data->window->w, data->window->h);
                }

#ifdef HIGHDPI_DEBUG
                SDL_Log("WM_DPICHANGED: Doing nothing, assuming window is already sized correctly");
#endif
                return 0;
            }

            /* Interactive user-initiated resizing/movement */

            if (WIN_IsPerMonitorV2DPIAware(SDL_GetVideoDevice())) {
                /* WM_GETDPISCALEDSIZE should have been called prior, so we can trust the given
                   suggestedRect. */
                w = suggestedRect->right - suggestedRect->left;
                h = suggestedRect->bottom - suggestedRect->top;

#ifdef HIGHDPI_DEBUG
                SDL_Log("WM_DPICHANGED: using suggestedRect");
#endif
            } else {
                /* permonitor and earlier DPI awareness: calculate the new frame w/h such that
                   the client area size is maintained. */
                const DWORD style = GetWindowLong(hwnd, GWL_STYLE);
                const BOOL menu = (style & WS_CHILDWINDOW) ? FALSE : (GetMenu(hwnd) != NULL);

                RECT rect = { 0 };
                rect.right = data->window->w;
                rect.bottom = data->window->h;

                if (data->videodata->dpi_scaling_enabled) {
                    /* scale client size to from points to the new DPI */
                    rect.right = MulDiv(rect.right, newDPI, 96);
                    rect.bottom = MulDiv(rect.bottom, newDPI, 96);
                }

                if (!(data->window->flags & SDL_WINDOW_BORDERLESS)) {
                    AdjustWindowRectEx(&rect, style, menu, 0);
                }

                w = rect.right - rect.left;
                h = rect.bottom - rect.top;
            }

#ifdef HIGHDPI_DEBUG
            SDL_Log("WM_DPICHANGED: current SDL window size: (%dx%d)\tcalling SetWindowPos: (%d, %d), (%dx%d)\n",
                    data->window->w, data->window->h,
                    suggestedRect->left, suggestedRect->top, w, h);
#endif

            data->expected_resize = SDL_TRUE;
            SetWindowPos(hwnd,
                         NULL,
                         suggestedRect->left,
                         suggestedRect->top,
                         w,
                         h,
                         SWP_NOZORDER | SWP_NOACTIVATE);
            data->expected_resize = SDL_FALSE;

            if (data->videodata->dpi_scaling_enabled) {
                /* Update the cached DPI value for this window */
                data->scaling_dpi = newDPI;

                /* Send a SDL_WINDOWEVENT_SIZE_CHANGED saying that the client size (in dpi-scaled points) is unchanged.
                   Renderers need to get this to know that the framebuffer size changed. */
                SDL_SendWindowEvent(data->window, SDL_WINDOWEVENT_SIZE_CHANGED, data->window->w, data->window->h);
            }

            return 0;
        }
        break;

    case WM_SETTINGCHANGE:
        if (wParam == SPI_SETMOUSE || wParam == SPI_SETMOUSESPEED) {
            WIN_UpdateMouseSystemScale();
        }
        break;

#endif /*!defined(__XBOXONE__) && !defined(__XBOXSERIES__)*/
    }

    /* If there's a window proc, assume it's going to handle messages */
    if (data->wndproc) {
        return CallWindowProc(data->wndproc, hwnd, msg, wParam, lParam);
    } else if (returnCode >= 0) {
        return returnCode;
    } else {
        return CallWindowProc(DefWindowProc, hwnd, msg, wParam, lParam);
    }
}

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
static void WIN_UpdateClipCursorForWindows(void)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();
    SDL_Window *window;
    Uint32 now = SDL_GetTicks();
    const Uint32 CLIPCURSOR_UPDATE_INTERVAL_MS = 3000;

    if (_this) {
        for (window = _this->windows; window; window = window->next) {
            SDL_WindowData *data = (SDL_WindowData *)window->driverdata;
            if (data) {
                if (data->skip_update_clipcursor) {
                    data->skip_update_clipcursor = SDL_FALSE;
                    WIN_UpdateClipCursor(window);
                } else if ((now - data->last_updated_clipcursor) >= CLIPCURSOR_UPDATE_INTERVAL_MS) {
                    WIN_UpdateClipCursor(window);
                }
            }
        }
    }
}

static void WIN_UpdateMouseCapture(void)
{
    SDL_Window *focusWindow = SDL_GetKeyboardFocus();

    if (focusWindow && (focusWindow->flags & SDL_WINDOW_MOUSE_CAPTURE)) {
        SDL_WindowData *data = (SDL_WindowData *)focusWindow->driverdata;

        if (!data->mouse_tracked) {
            POINT cursorPos;

            if (GetCursorPos(&cursorPos) && ScreenToClient(data->hwnd, &cursorPos)) {
                SDL_bool swapButtons = GetSystemMetrics(SM_SWAPBUTTON) != 0;
                SDL_MouseID mouseID = SDL_GetMouse()->mouseID;
                SDL_Point point;
                point.x = cursorPos.x;
                point.y = cursorPos.y;
                WIN_ClientPointToSDL(data->window, &point.x, &point.y);

                SDL_SendMouseMotion(data->window, mouseID, 0, point.x, point.y);
                SDL_SendMouseButton(data->window, mouseID, GetAsyncKeyState(VK_LBUTTON) & 0x8000 ? SDL_PRESSED : SDL_RELEASED, !swapButtons ? SDL_BUTTON_LEFT : SDL_BUTTON_RIGHT);
                SDL_SendMouseButton(data->window, mouseID, GetAsyncKeyState(VK_RBUTTON) & 0x8000 ? SDL_PRESSED : SDL_RELEASED, !swapButtons ? SDL_BUTTON_RIGHT : SDL_BUTTON_LEFT);
                SDL_SendMouseButton(data->window, mouseID, GetAsyncKeyState(VK_MBUTTON) & 0x8000 ? SDL_PRESSED : SDL_RELEASED, SDL_BUTTON_MIDDLE);
                SDL_SendMouseButton(data->window, mouseID, GetAsyncKeyState(VK_XBUTTON1) & 0x8000 ? SDL_PRESSED : SDL_RELEASED, SDL_BUTTON_X1);
                SDL_SendMouseButton(data->window, mouseID, GetAsyncKeyState(VK_XBUTTON2) & 0x8000 ? SDL_PRESSED : SDL_RELEASED, SDL_BUTTON_X2);
            }
        }
    }
}
#endif /*!defined(__XBOXONE__) && !defined(__XBOXSERIES__)*/

/* A message hook called before TranslateMessage() */
static SDL_WindowsMessageHook g_WindowsMessageHook = NULL;
static void *g_WindowsMessageHookData = NULL;

void SDL_SetWindowsMessageHook(SDL_WindowsMessageHook callback, void *userdata)
{
    g_WindowsMessageHook = callback;
    g_WindowsMessageHookData = userdata;
}

int WIN_WaitEventTimeout(_THIS, int timeout)
{
    if (g_WindowsEnableMessageLoop) {
#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
        DWORD dwMilliseconds, ret;
        dwMilliseconds = timeout < 0 ? INFINITE : (DWORD)timeout;
        ret = MsgWaitForMultipleObjects(0, NULL, FALSE, dwMilliseconds, QS_ALLINPUT);
        if (ret == WAIT_OBJECT_0) {
            return 1;
        } else {
            return 0;
        }
#else
        /* MsgWaitForMultipleObjects is desktop-only. */
        MSG msg;
        BOOL message_result;
        UINT_PTR timer_id = 0;
        if (timeout > 0) {
            timer_id = SetTimer(NULL, 0, timeout, NULL);
            message_result = GetMessage(&msg, 0, 0, 0);
            KillTimer(NULL, timer_id);
        } else if (timeout == 0) {
            message_result = PeekMessage(&msg, NULL, 0, 0, PM_REMOVE);
        } else {
            message_result = GetMessage(&msg, 0, 0, 0);
        }
        if (message_result) {
            if (msg.message == WM_TIMER && !msg.hwnd && msg.wParam == timer_id) {
                return 0;
            }
            if (g_WindowsMessageHook) {
                g_WindowsMessageHook(g_WindowsMessageHookData, msg.hwnd, msg.message, msg.wParam, msg.lParam);
            }
            /* Always translate the message in case it's a non-SDL window (e.g. with Qt integration) */
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            return 1;
        } else {
            return 0;
        }
#endif /*!defined(__XBOXONE__) && !defined(__XBOXSERIES__)*/
    } else {
        /* Fail the wait so the caller falls back to polling */
        return -1;
    }
}

void WIN_SendWakeupEvent(_THIS, SDL_Window *window)
{
    SDL_WindowData *data = (SDL_WindowData *)window->driverdata;
    PostMessage(data->hwnd, data->videodata->_SDL_WAKEUP, 0, 0);
}

void WIN_PumpEvents(_THIS)
{
    MSG msg;
    DWORD end_ticks = GetTickCount() + 1;
    int new_messages = 0;
#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
    const Uint8 *keystate;
    SDL_Window *focusWindow;
#endif

    if (g_WindowsEnableMessageLoop) {
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (g_WindowsMessageHook) {
                g_WindowsMessageHook(g_WindowsMessageHookData, msg.hwnd, msg.message, msg.wParam, msg.lParam);
            }

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
            /* Don't dispatch any mouse motion queued prior to or including the last mouse warp */
            if (msg.message == WM_MOUSEMOVE && SDL_last_warp_time) {
                if (!SDL_TICKS_PASSED(msg.time, (SDL_last_warp_time + 1))) {
                    continue;
                }

                /* This mouse message happened after the warp */
                SDL_last_warp_time = 0;
            }
#endif /*!defined(__XBOXONE__) && !defined(__XBOXSERIES__)*/

            /* Always translate the message in case it's a non-SDL window (e.g. with Qt integration) */
            TranslateMessage(&msg);
            DispatchMessage(&msg);

            /* Make sure we don't busy loop here forever if there are lots of events coming in */
            if (SDL_TICKS_PASSED(msg.time, end_ticks)) {
                /* We might get a few new messages generated by the Steam overlay or other application hooks
                   In this case those messages will be processed before any pending input, so we want to continue after those messages.
                   (thanks to Peter Deayton for his investigation here)
                 */
                const int MAX_NEW_MESSAGES = 3;
                ++new_messages;
                if (new_messages > MAX_NEW_MESSAGES) {
                    break;
                }
            }
        }
    }

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
    /* Windows loses a shift KEYUP event when you have both pressed at once and let go of one.
       You won't get a KEYUP until both are released, and that keyup will only be for the second
       key you released. Take heroic measures and check the keystate as of the last handled event,
       and if we think a key is pressed when Windows doesn't, unstick it in SDL's state. */
    keystate = SDL_GetKeyboardState(NULL);
    if ((keystate[SDL_SCANCODE_LSHIFT] == SDL_PRESSED) && !(GetKeyState(VK_LSHIFT) & 0x8000)) {
        SDL_SendKeyboardKey(SDL_RELEASED, SDL_SCANCODE_LSHIFT);
    }
    if ((keystate[SDL_SCANCODE_RSHIFT] == SDL_PRESSED) && !(GetKeyState(VK_RSHIFT) & 0x8000)) {
        SDL_SendKeyboardKey(SDL_RELEASED, SDL_SCANCODE_RSHIFT);
    }

    /* The Windows key state gets lost when using Windows+Space or Windows+G shortcuts and
       not grabbing the keyboard. Note: If we *are* grabbing the keyboard, GetKeyState()
       will return inaccurate results for VK_LWIN and VK_RWIN but we don't need it anyway. */
    focusWindow = SDL_GetKeyboardFocus();
    if (!focusWindow || !(focusWindow->flags & SDL_WINDOW_KEYBOARD_GRABBED)) {
        if ((keystate[SDL_SCANCODE_LGUI] == SDL_PRESSED) && !(GetKeyState(VK_LWIN) & 0x8000)) {
            SDL_SendKeyboardKey(SDL_RELEASED, SDL_SCANCODE_LGUI);
        }
        if ((keystate[SDL_SCANCODE_RGUI] == SDL_PRESSED) && !(GetKeyState(VK_RWIN) & 0x8000)) {
            SDL_SendKeyboardKey(SDL_RELEASED, SDL_SCANCODE_RGUI);
        }
    }

    /* Update the clipping rect in case someone else has stolen it */
    WIN_UpdateClipCursorForWindows();

    /* Update mouse capture */
    WIN_UpdateMouseCapture();
#endif /*!defined(__XBOXONE__) && !defined(__XBOXSERIES__)*/

#ifdef __GDK__
    GDK_DispatchTaskQueue();
#endif
}

static int app_registered = 0;
LPTSTR SDL_Appname = NULL;
Uint32 SDL_Appstyle = 0;
HINSTANCE SDL_Instance = NULL;

static void WIN_CleanRegisterApp(WNDCLASSEX wcex)
{
#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
    if (wcex.hIcon) {
        DestroyIcon(wcex.hIcon);
    }
    if (wcex.hIconSm) {
        DestroyIcon(wcex.hIconSm);
    }
#endif
    SDL_free(SDL_Appname);
    SDL_Appname = NULL;
}

static BOOL CALLBACK WIN_ResourceNameCallback(HMODULE hModule, LPCTSTR lpType, LPTSTR lpName, LONG_PTR lParam)
{
    WNDCLASSEX *wcex = (WNDCLASSEX *)lParam;

    (void)lpType; /* We already know that the resource type is RT_GROUP_ICON. */

    /* We leave hIconSm as NULL as it will allow Windows to automatically
       choose the appropriate small icon size to suit the current DPI. */
    wcex->hIcon = LoadIcon(hModule, lpName);

    /* Do not bother enumerating any more. */
    return FALSE;
}

/* Register the class for this application */
int SDL_RegisterApp(const char *name, Uint32 style, void *hInst)
{
    WNDCLASSEX wcex;
#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
    const char *hint;
#endif

    /* Only do this once... */
    if (app_registered) {
        ++app_registered;
        return 0;
    }
    SDL_assert(!SDL_Appname);
    if (!name) {
        name = "SDL_app";
#if defined(CS_BYTEALIGNCLIENT) || defined(CS_OWNDC)
        style = (CS_BYTEALIGNCLIENT | CS_OWNDC);
#endif
    }
    SDL_Appname = WIN_UTF8ToString(name);
    SDL_Appstyle = style;
    SDL_Instance = hInst ? hInst : GetModuleHandle(NULL);

    /* Register the application class */
    wcex.cbSize = sizeof(WNDCLASSEX);
    wcex.hCursor = NULL;
    wcex.hIcon = NULL;
    wcex.hIconSm = NULL;
    wcex.lpszMenuName = NULL;
    wcex.lpszClassName = SDL_Appname;
    wcex.style = SDL_Appstyle;
    wcex.hbrBackground = NULL;
    wcex.lpfnWndProc = WIN_WindowProc;
    wcex.hInstance = SDL_Instance;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;

#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
    hint = SDL_GetHint(SDL_HINT_WINDOWS_INTRESOURCE_ICON);
    if (hint && *hint) {
        wcex.hIcon = LoadIcon(SDL_Instance, MAKEINTRESOURCE(SDL_atoi(hint)));

        hint = SDL_GetHint(SDL_HINT_WINDOWS_INTRESOURCE_ICON_SMALL);
        if (hint && *hint) {
            wcex.hIconSm = LoadIcon(SDL_Instance, MAKEINTRESOURCE(SDL_atoi(hint)));
        }
    } else {
        /* Use the first icon as a default icon, like in the Explorer. */
        EnumResourceNames(SDL_Instance, RT_GROUP_ICON, WIN_ResourceNameCallback, (LONG_PTR)&wcex);
    }
#endif /*!defined(__XBOXONE__) && !defined(__XBOXSERIES__)*/

    if (!RegisterClassEx(&wcex)) {
        WIN_CleanRegisterApp(wcex);
        return SDL_SetError("Couldn't register application class");
    }

    app_registered = 1;
    return 0;
}

/* Unregisters the windowclass registered in SDL_RegisterApp above. */
void SDL_UnregisterApp(void)
{
    WNDCLASSEX wcex;

    /* SDL_RegisterApp might not have been called before */
    if (!app_registered) {
        return;
    }
    --app_registered;
    if (app_registered == 0) {
        /* Ensure the icons are initialized. */
        wcex.hIcon = NULL;
        wcex.hIconSm = NULL;
        /* Check for any registered window classes. */
#if !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
        if (GetClassInfoEx(SDL_Instance, SDL_Appname, &wcex)) {
            UnregisterClass(SDL_Appname, SDL_Instance);
        }
#endif
        WIN_CleanRegisterApp(wcex);
    }
}

#endif /* SDL_VIDEO_DRIVER_WINDOWS */

/* vi: set ts=4 sw=4 expandtab: */
