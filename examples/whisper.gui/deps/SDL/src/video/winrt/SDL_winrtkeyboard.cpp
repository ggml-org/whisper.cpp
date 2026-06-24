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

/* Windows-specific includes */
#include <Windows.h>
#include <agile.h>

/* SDL-specific includes */
#include "SDL.h"
#include "SDL_winrtevents_c.h"

extern "C" {
#include "../../events/scancodes_windows.h"
#include "../../events/SDL_keyboard_c.h"
}

static SDL_Scancode WinRT_Official_Keycodes[] = {
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.None -- 0 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.LeftButton -- 1 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.RightButton -- 2 */
    SDL_SCANCODE_CANCEL,       /* VirtualKey.Cancel -- 3 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.MiddleButton -- 4 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.XButton1 -- 5 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.XButton2 -- 6 */
    SDL_SCANCODE_UNKNOWN,      /* -- 7 */
    SDL_SCANCODE_BACKSPACE,    /* VirtualKey.Back -- 8 */
    SDL_SCANCODE_TAB,          /* VirtualKey.Tab -- 9 */
    SDL_SCANCODE_UNKNOWN,      /* -- 10 */
    SDL_SCANCODE_UNKNOWN,      /* -- 11 */
    SDL_SCANCODE_CLEAR,        /* VirtualKey.Clear -- 12 */
    SDL_SCANCODE_RETURN,       /* VirtualKey.Enter -- 13 */
    SDL_SCANCODE_UNKNOWN,      /* -- 14 */
    SDL_SCANCODE_UNKNOWN,      /* -- 15 */
    SDL_SCANCODE_LSHIFT,       /* VirtualKey.Shift -- 16 */
    SDL_SCANCODE_LCTRL,        /* VirtualKey.Control -- 17 */
    SDL_SCANCODE_MENU,         /* VirtualKey.Menu -- 18 */
    SDL_SCANCODE_PAUSE,        /* VirtualKey.Pause -- 19 */
    SDL_SCANCODE_CAPSLOCK,     /* VirtualKey.CapitalLock -- 20 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.Kana or VirtualKey.Hangul -- 21 */
    SDL_SCANCODE_UNKNOWN,      /* -- 22 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.Junja -- 23 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.Final -- 24 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.Hanja or VirtualKey.Kanji -- 25 */
    SDL_SCANCODE_UNKNOWN,      /* -- 26 */
    SDL_SCANCODE_ESCAPE,       /* VirtualKey.Escape -- 27 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.Convert -- 28 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.NonConvert -- 29 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.Accept -- 30 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.ModeChange -- 31  (maybe SDL_SCANCODE_MODE ?) */
    SDL_SCANCODE_SPACE,        /* VirtualKey.Space -- 32 */
    SDL_SCANCODE_PAGEUP,       /* VirtualKey.PageUp -- 33 */
    SDL_SCANCODE_PAGEDOWN,     /* VirtualKey.PageDown -- 34 */
    SDL_SCANCODE_END,          /* VirtualKey.End -- 35 */
    SDL_SCANCODE_HOME,         /* VirtualKey.Home -- 36 */
    SDL_SCANCODE_LEFT,         /* VirtualKey.Left -- 37 */
    SDL_SCANCODE_UP,           /* VirtualKey.Up -- 38 */
    SDL_SCANCODE_RIGHT,        /* VirtualKey.Right -- 39 */
    SDL_SCANCODE_DOWN,         /* VirtualKey.Down -- 40 */
    SDL_SCANCODE_SELECT,       /* VirtualKey.Select -- 41 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.Print -- 42  (maybe SDL_SCANCODE_PRINTSCREEN ?) */
    SDL_SCANCODE_EXECUTE,      /* VirtualKey.Execute -- 43 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.Snapshot -- 44 */
    SDL_SCANCODE_INSERT,       /* VirtualKey.Insert -- 45 */
    SDL_SCANCODE_DELETE,       /* VirtualKey.Delete -- 46 */
    SDL_SCANCODE_HELP,         /* VirtualKey.Help -- 47 */
    SDL_SCANCODE_0,            /* VirtualKey.Number0 -- 48 */
    SDL_SCANCODE_1,            /* VirtualKey.Number1 -- 49 */
    SDL_SCANCODE_2,            /* VirtualKey.Number2 -- 50 */
    SDL_SCANCODE_3,            /* VirtualKey.Number3 -- 51 */
    SDL_SCANCODE_4,            /* VirtualKey.Number4 -- 52 */
    SDL_SCANCODE_5,            /* VirtualKey.Number5 -- 53 */
    SDL_SCANCODE_6,            /* VirtualKey.Number6 -- 54 */
    SDL_SCANCODE_7,            /* VirtualKey.Number7 -- 55 */
    SDL_SCANCODE_8,            /* VirtualKey.Number8 -- 56 */
    SDL_SCANCODE_9,            /* VirtualKey.Number9 -- 57 */
    SDL_SCANCODE_UNKNOWN,      /* -- 58 */
    SDL_SCANCODE_UNKNOWN,      /* -- 59 */
    SDL_SCANCODE_UNKNOWN,      /* -- 60 */
    SDL_SCANCODE_UNKNOWN,      /* -- 61 */
    SDL_SCANCODE_UNKNOWN,      /* -- 62 */
    SDL_SCANCODE_UNKNOWN,      /* -- 63 */
    SDL_SCANCODE_UNKNOWN,      /* -- 64 */
    SDL_SCANCODE_A,            /* VirtualKey.A -- 65 */
    SDL_SCANCODE_B,            /* VirtualKey.B -- 66 */
    SDL_SCANCODE_C,            /* VirtualKey.C -- 67 */
    SDL_SCANCODE_D,            /* VirtualKey.D -- 68 */
    SDL_SCANCODE_E,            /* VirtualKey.E -- 69 */
    SDL_SCANCODE_F,            /* VirtualKey.F -- 70 */
    SDL_SCANCODE_G,            /* VirtualKey.G -- 71 */
    SDL_SCANCODE_H,            /* VirtualKey.H -- 72 */
    SDL_SCANCODE_I,            /* VirtualKey.I -- 73 */
    SDL_SCANCODE_J,            /* VirtualKey.J -- 74 */
    SDL_SCANCODE_K,            /* VirtualKey.K -- 75 */
    SDL_SCANCODE_L,            /* VirtualKey.L -- 76 */
    SDL_SCANCODE_M,            /* VirtualKey.M -- 77 */
    SDL_SCANCODE_N,            /* VirtualKey.N -- 78 */
    SDL_SCANCODE_O,            /* VirtualKey.O -- 79 */
    SDL_SCANCODE_P,            /* VirtualKey.P -- 80 */
    SDL_SCANCODE_Q,            /* VirtualKey.Q -- 81 */
    SDL_SCANCODE_R,            /* VirtualKey.R -- 82 */
    SDL_SCANCODE_S,            /* VirtualKey.S -- 83 */
    SDL_SCANCODE_T,            /* VirtualKey.T -- 84 */
    SDL_SCANCODE_U,            /* VirtualKey.U -- 85 */
    SDL_SCANCODE_V,            /* VirtualKey.V -- 86 */
    SDL_SCANCODE_W,            /* VirtualKey.W -- 87 */
    SDL_SCANCODE_X,            /* VirtualKey.X -- 88 */
    SDL_SCANCODE_Y,            /* VirtualKey.Y -- 89 */
    SDL_SCANCODE_Z,            /* VirtualKey.Z -- 90 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.LeftWindows -- 91  (maybe SDL_SCANCODE_APPLICATION or SDL_SCANCODE_LGUI ?) */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.RightWindows -- 92  (maybe SDL_SCANCODE_APPLICATION or SDL_SCANCODE_RGUI ?) */
    SDL_SCANCODE_APPLICATION,  /* VirtualKey.Application -- 93 */
    SDL_SCANCODE_UNKNOWN,      /* -- 94 */
    SDL_SCANCODE_SLEEP,        /* VirtualKey.Sleep -- 95 */
    SDL_SCANCODE_KP_0,         /* VirtualKey.NumberPad0 -- 96 */
    SDL_SCANCODE_KP_1,         /* VirtualKey.NumberPad1 -- 97 */
    SDL_SCANCODE_KP_2,         /* VirtualKey.NumberPad2 -- 98 */
    SDL_SCANCODE_KP_3,         /* VirtualKey.NumberPad3 -- 99 */
    SDL_SCANCODE_KP_4,         /* VirtualKey.NumberPad4 -- 100 */
    SDL_SCANCODE_KP_5,         /* VirtualKey.NumberPad5 -- 101 */
    SDL_SCANCODE_KP_6,         /* VirtualKey.NumberPad6 -- 102 */
    SDL_SCANCODE_KP_7,         /* VirtualKey.NumberPad7 -- 103 */
    SDL_SCANCODE_KP_8,         /* VirtualKey.NumberPad8 -- 104 */
    SDL_SCANCODE_KP_9,         /* VirtualKey.NumberPad9 -- 105 */
    SDL_SCANCODE_KP_MULTIPLY,  /* VirtualKey.Multiply -- 106 */
    SDL_SCANCODE_KP_PLUS,      /* VirtualKey.Add -- 107 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.Separator -- 108 */
    SDL_SCANCODE_KP_MINUS,     /* VirtualKey.Subtract -- 109 */
    SDL_SCANCODE_UNKNOWN,      /* VirtualKey.Decimal -- 110  (maybe SDL_SCANCODE_DECIMALSEPARATOR, SDL_SCANCODE_KP_DECIMAL, or SDL_SCANCODE_KP_PERIOD ?) */
    SDL_SCANCODE_KP_DIVIDE,    /* VirtualKey.Divide -- 111 */
    SDL_SCANCODE_F1,           /* VirtualKey.F1 -- 112 */
    SDL_SCANCODE_F2,           /* VirtualKey.F2 -- 113 */
    SDL_SCANCODE_F3,           /* VirtualKey.F3 -- 114 */
    SDL_SCANCODE_F4,           /* VirtualKey.F4 -- 115 */
    SDL_SCANCODE_F5,           /* VirtualKey.F5 -- 116 */
    SDL_SCANCODE_F6,           /* VirtualKey.F6 -- 117 */
    SDL_SCANCODE_F7,           /* VirtualKey.F7 -- 118 */
    SDL_SCANCODE_F8,           /* VirtualKey.F8 -- 119 */
    SDL_SCANCODE_F9,           /* VirtualKey.F9 -- 120 */
    SDL_SCANCODE_F10,          /* VirtualKey.F10 -- 121 */
    SDL_SCANCODE_F11,          /* VirtualKey.F11 -- 122 */
    SDL_SCANCODE_F12,          /* VirtualKey.F12 -- 123 */
    SDL_SCANCODE_F13,          /* VirtualKey.F13 -- 124 */
    SDL_SCANCODE_F14,          /* VirtualKey.F14 -- 125 */
    SDL_SCANCODE_F15,          /* VirtualKey.F15 -- 126 */
    SDL_SCANCODE_F16,          /* VirtualKey.F16 -- 127 */
    SDL_SCANCODE_F17,          /* VirtualKey.F17 -- 128 */
    SDL_SCANCODE_F18,          /* VirtualKey.F18 -- 129 */
    SDL_SCANCODE_F19,          /* VirtualKey.F19 -- 130 */
    SDL_SCANCODE_F20,          /* VirtualKey.F20 -- 131 */
    SDL_SCANCODE_F21,          /* VirtualKey.F21 -- 132 */
    SDL_SCANCODE_F22,          /* VirtualKey.F22 -- 133 */
    SDL_SCANCODE_F23,          /* VirtualKey.F23 -- 134 */
    SDL_SCANCODE_F24,          /* VirtualKey.F24 -- 135 */
    SDL_SCANCODE_UNKNOWN,      /* -- 136 */
    SDL_SCANCODE_UNKNOWN,      /* -- 137 */
    SDL_SCANCODE_UNKNOWN,      /* -- 138 */
    SDL_SCANCODE_UNKNOWN,      /* -- 139 */
    SDL_SCANCODE_UNKNOWN,      /* -- 140 */
    SDL_SCANCODE_UNKNOWN,      /* -- 141 */
    SDL_SCANCODE_UNKNOWN,      /* -- 142 */
    SDL_SCANCODE_UNKNOWN,      /* -- 143 */
    SDL_SCANCODE_NUMLOCKCLEAR, /* VirtualKey.NumberKeyLock -- 144 */
    SDL_SCANCODE_SCROLLLOCK,   /* VirtualKey.Scroll -- 145 */
    SDL_SCANCODE_UNKNOWN,      /* -- 146 */
    SDL_SCANCODE_UNKNOWN,      /* -- 147 */
    SDL_SCANCODE_UNKNOWN,      /* -- 148 */
    SDL_SCANCODE_UNKNOWN,      /* -- 149 */
    SDL_SCANCODE_UNKNOWN,      /* -- 150 */
    SDL_SCANCODE_UNKNOWN,      /* -- 151 */
    SDL_SCANCODE_UNKNOWN,      /* -- 152 */
    SDL_SCANCODE_UNKNOWN,      /* -- 153 */
    SDL_SCANCODE_UNKNOWN,      /* -- 154 */
    SDL_SCANCODE_UNKNOWN,      /* -- 155 */
    SDL_SCANCODE_UNKNOWN,      /* -- 156 */
    SDL_SCANCODE_UNKNOWN,      /* -- 157 */
    SDL_SCANCODE_UNKNOWN,      /* -- 158 */
    SDL_SCANCODE_UNKNOWN,      /* -- 159 */
    SDL_SCANCODE_LSHIFT,       /* VirtualKey.LeftShift -- 160 */
    SDL_SCANCODE_RSHIFT,       /* VirtualKey.RightShift -- 161 */
    SDL_SCANCODE_LCTRL,        /* VirtualKey.LeftControl -- 162 */
    SDL_SCANCODE_RCTRL,        /* VirtualKey.RightControl -- 163 */
    SDL_SCANCODE_MENU,         /* VirtualKey.LeftMenu -- 164 */
    SDL_SCANCODE_MENU,         /* VirtualKey.RightMenu -- 165 */
    SDL_SCANCODE_AC_BACK,      /* VirtualKey.GoBack -- 166 : The go back key. */
    SDL_SCANCODE_AC_FORWARD,   /* VirtualKey.GoForward -- 167 : The go forward key. */
    SDL_SCANCODE_AC_REFRESH,   /* VirtualKey.Refresh -- 168 : The refresh key. */
    SDL_SCANCODE_AC_STOP,      /* VirtualKey.Stop -- 169 : The stop key. */
    SDL_SCANCODE_AC_SEARCH,    /* VirtualKey.Search -- 170 : The search key. */
    SDL_SCANCODE_AC_BOOKMARKS, /* VirtualKey.Favorites -- 171 : The favorites key. */
    SDL_SCANCODE_AC_HOME       /* VirtualKey.GoHome -- 172 : The go home key. */
};

/* Attempt to translate a keycode that isn't listed in WinRT's VirtualKey enum.
 */
static SDL_Scancode WINRT_TranslateUnofficialKeycode(int keycode)
{
    switch (keycode) {
    case 173:
        return SDL_SCANCODE_MUTE; /* VK_VOLUME_MUTE */
    case 174:
        return SDL_SCANCODE_VOLUMEDOWN; /* VK_VOLUME_DOWN */
    case 175:
        return SDL_SCANCODE_VOLUMEUP; /* VK_VOLUME_UP */
    case 176:
        return SDL_SCANCODE_AUDIONEXT; /* VK_MEDIA_NEXT_TRACK */
    case 177:
        return SDL_SCANCODE_AUDIOPREV; /* VK_MEDIA_PREV_TRACK */
    // case 178: return ;                       /* VK_MEDIA_STOP */
    case 179:
        return SDL_SCANCODE_AUDIOPLAY; /* VK_MEDIA_PLAY_PAUSE */
    case 180:
        return SDL_SCANCODE_MAIL; /* VK_LAUNCH_MAIL */
    case 181:
        return SDL_SCANCODE_MEDIASELECT; /* VK_LAUNCH_MEDIA_SELECT */
    // case 182: return ;                       /* VK_LAUNCH_APP1 */
    case 183:
        return SDL_SCANCODE_CALCULATOR; /* VK_LAUNCH_APP2 */
    // case 184: return ;                       /* ... reserved ... */
    // case 185: return ;                       /* ... reserved ... */
    case 186:
        return SDL_SCANCODE_SEMICOLON; /* VK_OEM_1, ';:' key on US standard keyboards */
    case 187:
        return SDL_SCANCODE_EQUALS; /* VK_OEM_PLUS */
    case 188:
        return SDL_SCANCODE_COMMA; /* VK_OEM_COMMA */
    case 189:
        return SDL_SCANCODE_MINUS; /* VK_OEM_MINUS */
    case 190:
        return SDL_SCANCODE_PERIOD; /* VK_OEM_PERIOD */
    case 191:
        return SDL_SCANCODE_SLASH; /* VK_OEM_2, '/?' key on US standard keyboards */
    case 192:
        return SDL_SCANCODE_GRAVE; /* VK_OEM_3, '`~' key on US standard keyboards */
    // ?
    // ... reserved or unassigned ...
    // ?
    case 219:
        return SDL_SCANCODE_LEFTBRACKET; /* VK_OEM_4, '[{' key on US standard keyboards */
    case 220:
        return SDL_SCANCODE_BACKSLASH; /* VK_OEM_5, '\|' key on US standard keyboards */
    case 221:
        return SDL_SCANCODE_RIGHTBRACKET; /* VK_OEM_6, ']}' key on US standard keyboards */
    case 222:
        return SDL_SCANCODE_APOSTROPHE; /* VK_OEM_7, 'single/double quote' on US standard keyboards */
    default:
        break;
    }
    return SDL_SCANCODE_UNKNOWN;
}

static SDL_Scancode WINRT_TranslateKeycode(int keycode, unsigned int nativeScancode)
{
    // TODO, WinRT: try filling out the WinRT keycode table as much as possible, using the Win32 table for interpretation hints

    SDL_Scancode scancode = SDL_SCANCODE_UNKNOWN;

    /* HACK ALERT: At least one VirtualKey constant (Shift) with a left/right
     * designation might not get reported with its correct handedness, however
     * its hardware scan code can fill in the gaps.  If this is detected,
     * use the hardware scan code to try telling if the left, or the right
     * side's key was used.
     *
     * If Microsoft ever allows MapVirtualKey or MapVirtualKeyEx to be used
     * in WinRT apps, or something similar to these (it doesn't appear to be,
     * at least not for Windows [Phone] 8/8.1, as of Oct 24, 2014), then this
     * hack might become deprecated, or obsolete.
     */
    if (nativeScancode < SDL_arraysize(windows_scancode_table)) {
        switch (keycode) {
        case 16: // VirtualKey.Shift
            switch (windows_scancode_table[nativeScancode]) {
            case SDL_SCANCODE_LSHIFT:
            case SDL_SCANCODE_RSHIFT:
                return windows_scancode_table[nativeScancode];
            }
            break;

            // Add others, as necessary.
            //
            // Unfortunately, this hack doesn't seem to work in determining
            // handedness with Control keys.

        default:
            break;
        }
    }

    /* Try to get a documented, WinRT, 'VirtualKey' next (as documented at
       http://msdn.microsoft.com/en-us/library/windows/apps/windows.system.virtualkey.aspx ).
       If that fails, fall back to a Win32 virtual key.
       If that fails, attempt to fall back to a scancode-derived key.
    */
    if (keycode < SDL_arraysize(WinRT_Official_Keycodes)) {
        scancode = WinRT_Official_Keycodes[keycode];
    }
    if (scancode == SDL_SCANCODE_UNKNOWN) {
        scancode = WINRT_TranslateUnofficialKeycode(keycode);
    }
    if (scancode == SDL_SCANCODE_UNKNOWN) {
        if (nativeScancode < SDL_arraysize(windows_scancode_table)) {
            scancode = windows_scancode_table[nativeScancode];
        }
    }
    /*
    if (scancode == SDL_SCANCODE_UNKNOWN) {
        SDL_Log("WinRT TranslateKeycode, unknown keycode=%d\n", (int)keycode);
    }
    */
    return scancode;
}

void WINRT_ProcessKeyDownEvent(Windows::UI::Core::KeyEventArgs ^ args)
{
    SDL_Scancode sdlScancode = WINRT_TranslateKeycode((int)args->VirtualKey, args->KeyStatus.ScanCode);
#if 0
    SDL_Keycode keycode = SDL_GetKeyFromScancode(sdlScancode);
    SDL_Log("key down, handled=%s, ext?=%s, released?=%s, menu key down?=%s, "
            "repeat count=%d, native scan code=0x%x, was down?=%s, vkey=%d, "
            "sdl scan code=%d (%s), sdl key code=%d (%s)\n",
        (args->Handled ? "1" : "0"),
        (args->KeyStatus.IsExtendedKey ? "1" : "0"),
        (args->KeyStatus.IsKeyReleased ? "1" : "0"),
        (args->KeyStatus.IsMenuKeyDown ? "1" : "0"),
        args->KeyStatus.RepeatCount,
        args->KeyStatus.ScanCode,
        (args->KeyStatus.WasKeyDown ? "1" : "0"),
        args->VirtualKey,
        sdlScancode,
        SDL_GetScancodeName(sdlScancode),
        keycode,
        SDL_GetKeyName(keycode));
    //args->Handled = true;
#endif
    SDL_SendKeyboardKey(SDL_PRESSED, sdlScancode);
}

void WINRT_ProcessKeyUpEvent(Windows::UI::Core::KeyEventArgs ^ args)
{
    SDL_Scancode sdlScancode = WINRT_TranslateKeycode((int)args->VirtualKey, args->KeyStatus.ScanCode);
#if 0
    SDL_Keycode keycode = SDL_GetKeyFromScancode(sdlScancode);
    SDL_Log("key up, handled=%s, ext?=%s, released?=%s, menu key down?=%s, "
            "repeat count=%d, native scan code=0x%x, was down?=%s, vkey=%d, "
            "sdl scan code=%d (%s), sdl key code=%d (%s)\n",
        (args->Handled ? "1" : "0"),
        (args->KeyStatus.IsExtendedKey ? "1" : "0"),
        (args->KeyStatus.IsKeyReleased ? "1" : "0"),
        (args->KeyStatus.IsMenuKeyDown ? "1" : "0"),
        args->KeyStatus.RepeatCount,
        args->KeyStatus.ScanCode,
        (args->KeyStatus.WasKeyDown ? "1" : "0"),
        args->VirtualKey,
        sdlScancode,
        SDL_GetScancodeName(sdlScancode),
        keycode,
        SDL_GetKeyName(keycode));
    //args->Handled = true;
#endif
    SDL_SendKeyboardKey(SDL_RELEASED, sdlScancode);
}

void WINRT_ProcessCharacterReceivedEvent(Windows::UI::Core::CharacterReceivedEventArgs ^ args)
{
    wchar_t src_ucs2[2];
    char dest_utf8[16];
    int result;

    /* Setup src */
    src_ucs2[0] = args->KeyCode;
    src_ucs2[1] = L'\0';

    /* Convert the text, then send an SDL_TEXTINPUT event. */
    result = WideCharToMultiByte(CP_UTF8, 0, (LPCWSTR)&src_ucs2, -1, (LPSTR)dest_utf8, sizeof(dest_utf8), NULL, NULL);
    if (result > 0) {
        SDL_SendKeyboardText(dest_utf8);
    }
}

#if NTDDI_VERSION >= NTDDI_WIN10

static bool WINRT_InputPaneVisible = false;

void WINTRT_OnInputPaneShowing(Windows::UI::ViewManagement::InputPane ^ sender, Windows::UI::ViewManagement::InputPaneVisibilityEventArgs ^ args)
{
    WINRT_InputPaneVisible = true;
}

void WINTRT_OnInputPaneHiding(Windows::UI::ViewManagement::InputPane ^ sender, Windows::UI::ViewManagement::InputPaneVisibilityEventArgs ^ args)
{
    WINRT_InputPaneVisible = false;
}

void WINTRT_InitialiseInputPaneEvents(_THIS)
{
    using namespace Windows::UI::ViewManagement;
    InputPane ^ inputPane = InputPane::GetForCurrentView();
    if (inputPane) {
        inputPane->Showing += ref new Windows::Foundation::TypedEventHandler<Windows::UI::ViewManagement::InputPane ^,
                                                                             Windows::UI::ViewManagement::InputPaneVisibilityEventArgs ^>(&WINTRT_OnInputPaneShowing);
        inputPane->Hiding += ref new Windows::Foundation::TypedEventHandler<Windows::UI::ViewManagement::InputPane ^,
                                                                            Windows::UI::ViewManagement::InputPaneVisibilityEventArgs ^>(&WINTRT_OnInputPaneHiding);
    }
}

SDL_bool WINRT_HasScreenKeyboardSupport(_THIS)
{
    return SDL_TRUE;
}

void WINRT_ShowScreenKeyboard(_THIS, SDL_Window *window)
{
    using namespace Windows::UI::ViewManagement;
    InputPane ^ inputPane = InputPane::GetForCurrentView();
    if (inputPane) {
        inputPane->TryShow();
    }
}

void WINRT_HideScreenKeyboard(_THIS, SDL_Window *window)
{
    using namespace Windows::UI::ViewManagement;
    InputPane ^ inputPane = InputPane::GetForCurrentView();
    if (inputPane) {
        inputPane->TryHide();
    }
}

SDL_bool WINRT_IsScreenKeyboardShown(_THIS, SDL_Window *window)
{
    using namespace Windows::UI::ViewManagement;
    InputPane ^ inputPane = InputPane::GetForCurrentView();
    if (inputPane) {
        switch (SDL_WinRTGetDeviceFamily()) {
        case SDL_WINRT_DEVICEFAMILY_XBOX:
            // Documentation recommends using inputPane->Visible
            // https://learn.microsoft.com/en-us/uwp/api/windows.ui.viewmanagement.inputpane.visible?view=winrt-22621
            // This does not seem to work on latest UWP/Xbox.
            // Workaround: Listen to Showing/Hiding events
            if (WINRT_InputPaneVisible) {
                return SDL_TRUE;
            }
            break;
        default:
            // OccludedRect is recommend on universal apps per docs
            // https://learn.microsoft.com/en-us/uwp/api/windows.ui.viewmanagement.inputpane.visible?view=winrt-22621
            Windows::Foundation::Rect rect = inputPane->OccludedRect;
            if (rect.Width > 0 && rect.Height > 0) {
                return SDL_TRUE;
            }
            break;
        }
    }
    return SDL_FALSE;
}

#endif // NTDDI_VERSION >= ...

#endif // SDL_VIDEO_DRIVER_WINRT
