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
#include "SDL_hints.h"

#include "../../events/SDL_keyboard_c.h"
#include "../../events/scancodes_windows.h"

#include <imm.h>
#include <oleauto.h>

#ifndef SDL_DISABLE_WINDOWS_IME
static void IME_Init(SDL_VideoData *videodata, HWND hwnd);
static void IME_Enable(SDL_VideoData *videodata, HWND hwnd);
static void IME_Disable(SDL_VideoData *videodata, HWND hwnd);
static void IME_Quit(SDL_VideoData *videodata);
static SDL_bool IME_IsTextInputShown(SDL_VideoData *videodata);
#endif /* !SDL_DISABLE_WINDOWS_IME */

#ifndef MAPVK_VK_TO_VSC
#define MAPVK_VK_TO_VSC 0
#endif
#ifndef MAPVK_VSC_TO_VK
#define MAPVK_VSC_TO_VK 1
#endif
#ifndef MAPVK_VK_TO_CHAR
#define MAPVK_VK_TO_CHAR 2
#endif

/* Alphabetic scancodes for PC keyboards */
void WIN_InitKeyboard(_THIS)
{
#ifndef SDL_DISABLE_WINDOWS_IME
    SDL_VideoData *data = (SDL_VideoData *)_this->driverdata;

    data->ime_com_initialized = SDL_FALSE;
    data->ime_threadmgr = 0;
    data->ime_initialized = SDL_FALSE;
    data->ime_enabled = SDL_FALSE;
    data->ime_available = SDL_FALSE;
    data->ime_hwnd_main = 0;
    data->ime_hwnd_current = 0;
    data->ime_himc = 0;
    data->ime_composition_length = 32 * sizeof(WCHAR);
    data->ime_composition = (WCHAR *)SDL_malloc(data->ime_composition_length + sizeof(WCHAR));
    data->ime_composition[0] = 0;
    data->ime_readingstring[0] = 0;
    data->ime_cursor = 0;

    data->ime_candlist = SDL_FALSE;
    data->ime_candidates = NULL;
    data->ime_candcount = 0;
    data->ime_candref = 0;
    data->ime_candsel = 0;
    data->ime_candpgsize = 0;
    data->ime_candlistindexbase = 0;
    data->ime_candvertical = SDL_TRUE;

    data->ime_dirty = SDL_FALSE;
    SDL_memset(&data->ime_rect, 0, sizeof(data->ime_rect));
    SDL_memset(&data->ime_candlistrect, 0, sizeof(data->ime_candlistrect));
    data->ime_winwidth = 0;
    data->ime_winheight = 0;

    data->ime_hkl = 0;
    data->ime_himm32 = 0;
    data->GetReadingString = 0;
    data->ShowReadingWindow = 0;
    data->ImmLockIMC = 0;
    data->ImmUnlockIMC = 0;
    data->ImmLockIMCC = 0;
    data->ImmUnlockIMCC = 0;
    data->ime_uiless = SDL_FALSE;
    data->ime_threadmgrex = 0;
    data->ime_uielemsinkcookie = TF_INVALID_COOKIE;
    data->ime_alpnsinkcookie = TF_INVALID_COOKIE;
    data->ime_openmodesinkcookie = TF_INVALID_COOKIE;
    data->ime_convmodesinkcookie = TF_INVALID_COOKIE;
    data->ime_uielemsink = 0;
    data->ime_ippasink = 0;
#endif /* !SDL_DISABLE_WINDOWS_IME */

    WIN_UpdateKeymap(SDL_FALSE);

    SDL_SetScancodeName(SDL_SCANCODE_APPLICATION, "Menu");
    SDL_SetScancodeName(SDL_SCANCODE_LGUI, "Left Windows");
    SDL_SetScancodeName(SDL_SCANCODE_RGUI, "Right Windows");

    /* Are system caps/num/scroll lock active? Set our state to match. */
    SDL_ToggleModState(KMOD_CAPS, (GetKeyState(VK_CAPITAL) & 0x0001) ? SDL_TRUE : SDL_FALSE);
    SDL_ToggleModState(KMOD_NUM, (GetKeyState(VK_NUMLOCK) & 0x0001) ? SDL_TRUE : SDL_FALSE);
    SDL_ToggleModState(KMOD_SCROLL, (GetKeyState(VK_SCROLL) & 0x0001) ? SDL_TRUE : SDL_FALSE);
}

void WIN_UpdateKeymap(SDL_bool send_event)
{
    int i;
    SDL_Scancode scancode;
    SDL_Keycode keymap[SDL_NUM_SCANCODES];

    SDL_GetDefaultKeymap(keymap);

    for (i = 0; i < SDL_arraysize(windows_scancode_table); i++) {
        int vk;
        /* Make sure this scancode is a valid character scancode */
        scancode = windows_scancode_table[i];
        if (scancode == SDL_SCANCODE_UNKNOWN) {
            continue;
        }

        /* If this key is one of the non-mappable keys, ignore it */
        /* Uncomment the second part re-enable the behavior of not mapping the "`"(grave) key to the users actual keyboard layout */
        if ((keymap[scancode] & SDLK_SCANCODE_MASK) /*|| scancode == SDL_SCANCODE_GRAVE*/) {
            continue;
        }

        vk = MapVirtualKey(i, MAPVK_VSC_TO_VK);
        if (vk) {
            int ch = (MapVirtualKey(vk, MAPVK_VK_TO_CHAR) & 0x7FFF);
            if (ch) {
                if (ch >= 'A' && ch <= 'Z') {
                    keymap[scancode] = SDLK_a + (ch - 'A');
                } else {
                    keymap[scancode] = ch;
                }
            }
        }
    }

    SDL_SetKeymap(0, keymap, SDL_NUM_SCANCODES, send_event);
}

void WIN_QuitKeyboard(_THIS)
{
    SDL_VideoData *data = (SDL_VideoData *)_this->driverdata;

#ifndef SDL_DISABLE_WINDOWS_IME
    IME_Quit(data);

    if (data->ime_composition) {
        SDL_free(data->ime_composition);
        data->ime_composition = NULL;
    }
#endif /* !SDL_DISABLE_WINDOWS_IME */
}

void WIN_ResetDeadKeys(void)
{
    /*
    if a deadkey has been typed, but not the next character (which the deadkey might modify),
    this tries to undo the effect pressing the deadkey.
    see: http://archives.miloush.net/michkap/archive/2006/09/10/748775.html
    */
    BYTE keyboardState[256];
    WCHAR buffer[16];
    int keycode, scancode, result, i;

    GetKeyboardState(keyboardState);

    keycode = VK_SPACE;
    scancode = MapVirtualKey(keycode, MAPVK_VK_TO_VSC);
    if (scancode == 0) {
        /* the keyboard doesn't have this key */
        return;
    }

    for (i = 0; i < 5; i++) {
        result = ToUnicode(keycode, scancode, keyboardState, (LPWSTR)buffer, 16, 0);
        if (result > 0) {
            /* success */
            return;
        }
    }
}

void WIN_StartTextInput(_THIS)
{
#ifndef SDL_DISABLE_WINDOWS_IME
    SDL_Window *window;
#endif

    WIN_ResetDeadKeys();

#ifndef SDL_DISABLE_WINDOWS_IME
    window = SDL_GetKeyboardFocus();
    if (window) {
        HWND hwnd = ((SDL_WindowData *)window->driverdata)->hwnd;
        SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
        SDL_GetWindowSize(window, &videodata->ime_winwidth, &videodata->ime_winheight);
        IME_Init(videodata, hwnd);
        IME_Enable(videodata, hwnd);
    }
#endif /* !SDL_DISABLE_WINDOWS_IME */
}

void WIN_StopTextInput(_THIS)
{
#ifndef SDL_DISABLE_WINDOWS_IME
    SDL_Window *window;
#endif

    WIN_ResetDeadKeys();

#ifndef SDL_DISABLE_WINDOWS_IME
    window = SDL_GetKeyboardFocus();
    if (window) {
        HWND hwnd = ((SDL_WindowData *)window->driverdata)->hwnd;
        SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
        IME_Init(videodata, hwnd);
        IME_Disable(videodata, hwnd);
    }
#endif /* !SDL_DISABLE_WINDOWS_IME */
}

void WIN_SetTextInputRect(_THIS, const SDL_Rect *rect)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
    HIMC himc = 0;

    if (!rect) {
        SDL_InvalidParamError("rect");
        return;
    }

#ifndef SDL_DISABLE_WINDOWS_IME
    videodata->ime_rect = *rect;

    himc = ImmGetContext(videodata->ime_hwnd_current);
    if (himc) {
        COMPOSITIONFORM cof;
        CANDIDATEFORM caf;

        cof.dwStyle = CFS_RECT;
        cof.ptCurrentPos.x = videodata->ime_rect.x;
        cof.ptCurrentPos.y = videodata->ime_rect.y;
        cof.rcArea.left = videodata->ime_rect.x;
        cof.rcArea.right = (LONG)videodata->ime_rect.x + videodata->ime_rect.w;
        cof.rcArea.top = videodata->ime_rect.y;
        cof.rcArea.bottom = (LONG)videodata->ime_rect.y + videodata->ime_rect.h;
        ImmSetCompositionWindow(himc, &cof);

        caf.dwIndex = 0;
        caf.dwStyle = CFS_EXCLUDE;
        caf.ptCurrentPos.x = videodata->ime_rect.x;
        caf.ptCurrentPos.y = videodata->ime_rect.y;
        caf.rcArea.left = videodata->ime_rect.x;
        caf.rcArea.right = (LONG)videodata->ime_rect.x + videodata->ime_rect.w;
        caf.rcArea.top = videodata->ime_rect.y;
        caf.rcArea.bottom = (LONG)videodata->ime_rect.y + videodata->ime_rect.h;
        ImmSetCandidateWindow(himc, &caf);

        ImmReleaseContext(videodata->ime_hwnd_current, himc);
    }
#endif /* !SDL_DISABLE_WINDOWS_IME */
}

#ifdef SDL_DISABLE_WINDOWS_IME

void WIN_ClearComposition(_THIS)
{
}

SDL_bool WIN_IsTextInputShown(_THIS)
{
    return SDL_FALSE;
}

SDL_bool IME_HandleMessage(HWND hwnd, UINT msg, WPARAM wParam, LPARAM *lParam, SDL_VideoData *videodata)
{
    return SDL_FALSE;
}

void IME_Present(SDL_VideoData *videodata)
{
}

#else

#ifdef SDL_msctf_h_
#define USE_INIT_GUID
#elif defined(__GNUC__)
#define USE_INIT_GUID
#endif
#ifdef USE_INIT_GUID
#undef DEFINE_GUID
#define DEFINE_GUID(n, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) static const GUID n = { l, w1, w2, { b1, b2, b3, b4, b5, b6, b7, b8 } }
DEFINE_GUID(IID_ITfInputProcessorProfileActivationSink, 0x71C6E74E, 0x0F28, 0x11D8, 0xA8, 0x2A, 0x00, 0x06, 0x5B, 0x84, 0x43, 0x5C);
DEFINE_GUID(IID_ITfUIElementSink, 0xEA1EA136, 0x19DF, 0x11D7, 0xA6, 0xD2, 0x00, 0x06, 0x5B, 0x84, 0x43, 0x5C);
DEFINE_GUID(GUID_TFCAT_TIP_KEYBOARD, 0x34745C63, 0xB2F0, 0x4784, 0x8B, 0x67, 0x5E, 0x12, 0xC8, 0x70, 0x1A, 0x31);
DEFINE_GUID(IID_ITfSource, 0x4EA48A35, 0x60AE, 0x446F, 0x8F, 0xD6, 0xE6, 0xA8, 0xD8, 0x24, 0x59, 0xF7);
DEFINE_GUID(IID_ITfUIElementMgr, 0xEA1EA135, 0x19DF, 0x11D7, 0xA6, 0xD2, 0x00, 0x06, 0x5B, 0x84, 0x43, 0x5C);
DEFINE_GUID(IID_ITfCandidateListUIElement, 0xEA1EA138, 0x19DF, 0x11D7, 0xA6, 0xD2, 0x00, 0x06, 0x5B, 0x84, 0x43, 0x5C);
DEFINE_GUID(IID_ITfReadingInformationUIElement, 0xEA1EA139, 0x19DF, 0x11D7, 0xA6, 0xD2, 0x00, 0x06, 0x5B, 0x84, 0x43, 0x5C);
DEFINE_GUID(IID_ITfThreadMgr, 0xAA80E801, 0x2021, 0x11D2, 0x93, 0xE0, 0x00, 0x60, 0xB0, 0x67, 0xB8, 0x6E);
DEFINE_GUID(CLSID_TF_ThreadMgr, 0x529A9E6B, 0x6587, 0x4F23, 0xAB, 0x9E, 0x9C, 0x7D, 0x68, 0x3E, 0x3C, 0x50);
DEFINE_GUID(IID_ITfThreadMgrEx, 0x3E90ADE3, 0x7594, 0x4CB0, 0xBB, 0x58, 0x69, 0x62, 0x8F, 0x5F, 0x45, 0x8C);
#endif

#define LANG_CHT MAKELANGID(LANG_CHINESE, SUBLANG_CHINESE_TRADITIONAL)
#define LANG_CHS MAKELANGID(LANG_CHINESE, SUBLANG_CHINESE_SIMPLIFIED)

#define MAKEIMEVERSION(major, minor) ((DWORD)(((BYTE)(major) << 24) | ((BYTE)(minor) << 16)))
#define IMEID_VER(id)                ((id)&0xffff0000)
#define IMEID_LANG(id)               ((id)&0x0000ffff)

#define CHT_HKL_DAYI          ((HKL)(UINT_PTR)0xE0060404)
#define CHT_HKL_NEW_PHONETIC  ((HKL)(UINT_PTR)0xE0080404)
#define CHT_HKL_NEW_CHANG_JIE ((HKL)(UINT_PTR)0xE0090404)
#define CHT_HKL_NEW_QUICK     ((HKL)(UINT_PTR)0xE00A0404)
#define CHT_HKL_HK_CANTONESE  ((HKL)(UINT_PTR)0xE00B0404)
#define CHT_IMEFILENAME1      "TINTLGNT.IME"
#define CHT_IMEFILENAME2      "CINTLGNT.IME"
#define CHT_IMEFILENAME3      "MSTCIPHA.IME"
#define IMEID_CHT_VER42       (LANG_CHT | MAKEIMEVERSION(4, 2))
#define IMEID_CHT_VER43       (LANG_CHT | MAKEIMEVERSION(4, 3))
#define IMEID_CHT_VER44       (LANG_CHT | MAKEIMEVERSION(4, 4))
#define IMEID_CHT_VER50       (LANG_CHT | MAKEIMEVERSION(5, 0))
#define IMEID_CHT_VER51       (LANG_CHT | MAKEIMEVERSION(5, 1))
#define IMEID_CHT_VER52       (LANG_CHT | MAKEIMEVERSION(5, 2))
#define IMEID_CHT_VER60       (LANG_CHT | MAKEIMEVERSION(6, 0))
#define IMEID_CHT_VER_VISTA   (LANG_CHT | MAKEIMEVERSION(7, 0))

#define CHS_HKL          ((HKL)(UINT_PTR)0xE00E0804)
#define CHS_IMEFILENAME1 "PINTLGNT.IME"
#define CHS_IMEFILENAME2 "MSSCIPYA.IME"
#define IMEID_CHS_VER41  (LANG_CHS | MAKEIMEVERSION(4, 1))
#define IMEID_CHS_VER42  (LANG_CHS | MAKEIMEVERSION(4, 2))
#define IMEID_CHS_VER53  (LANG_CHS | MAKEIMEVERSION(5, 3))

#define LANG()         LOWORD((videodata->ime_hkl))
#define PRIMLANG()     ((WORD)PRIMARYLANGID(LANG()))
#define SUBLANG()      SUBLANGID(LANG())

static void IME_UpdateInputLocale(SDL_VideoData *videodata);
static int IME_ShowCandidateList(SDL_VideoData *videodata);
static void IME_ClearComposition(SDL_VideoData *videodata);
static void IME_SetWindow(SDL_VideoData *videodata, HWND hwnd);
static void IME_SetupAPI(SDL_VideoData *videodata);
static DWORD IME_GetId(SDL_VideoData *videodata, UINT uIndex);
static void IME_SendEditingEvent(SDL_VideoData *videodata);
static void IME_DestroyTextures(SDL_VideoData *videodata);

static SDL_bool UILess_SetupSinks(SDL_VideoData *videodata);
static void UILess_ReleaseSinks(SDL_VideoData *videodata);
static void UILess_EnableUIUpdates(SDL_VideoData *videodata);
static void UILess_DisableUIUpdates(SDL_VideoData *videodata);

static SDL_bool WIN_ShouldShowNativeUI(void)
{
    return SDL_GetHintBoolean(SDL_HINT_IME_SHOW_UI, SDL_FALSE);
}

static void IME_Init(SDL_VideoData *videodata, HWND hwnd)
{
    HRESULT hResult = S_OK;

    if (videodata->ime_initialized) {
        return;
    }

    videodata->ime_hwnd_main = hwnd;
    if (SUCCEEDED(WIN_CoInitialize())) {
        videodata->ime_com_initialized = SDL_TRUE;
        hResult = CoCreateInstance(&CLSID_TF_ThreadMgr, NULL, CLSCTX_INPROC_SERVER, &IID_ITfThreadMgr, (LPVOID *)&videodata->ime_threadmgr);
        if (hResult != S_OK) {
            videodata->ime_available = SDL_FALSE;
            SDL_SetError("CoCreateInstance() failed, HRESULT is %08X", (unsigned int)hResult);
            return;
        }
    }
    videodata->ime_initialized = SDL_TRUE;
    videodata->ime_himm32 = SDL_LoadObject("imm32.dll");
    if (!videodata->ime_himm32) {
        videodata->ime_available = SDL_FALSE;
        SDL_ClearError();
        return;
    }
    /* *INDENT-OFF* */ /* clang-format off */
    videodata->ImmLockIMC = (LPINPUTCONTEXT2 (WINAPI *)(HIMC))SDL_LoadFunction(videodata->ime_himm32, "ImmLockIMC");
    videodata->ImmUnlockIMC = (BOOL (WINAPI *)(HIMC))SDL_LoadFunction(videodata->ime_himm32, "ImmUnlockIMC");
    videodata->ImmLockIMCC = (LPVOID (WINAPI *)(HIMCC))SDL_LoadFunction(videodata->ime_himm32, "ImmLockIMCC");
    videodata->ImmUnlockIMCC = (BOOL (WINAPI *)(HIMCC))SDL_LoadFunction(videodata->ime_himm32, "ImmUnlockIMCC");
    /* *INDENT-ON* */ /* clang-format on */

    IME_SetWindow(videodata, hwnd);
    videodata->ime_himc = ImmGetContext(hwnd);
    ImmReleaseContext(hwnd, videodata->ime_himc);
    if (!videodata->ime_himc) {
        videodata->ime_available = SDL_FALSE;
        IME_Disable(videodata, hwnd);
        return;
    }
    videodata->ime_available = SDL_TRUE;
    IME_UpdateInputLocale(videodata);
    IME_SetupAPI(videodata);
    if (WIN_ShouldShowNativeUI()) {
        videodata->ime_uiless = SDL_FALSE;
    } else {
        videodata->ime_uiless = UILess_SetupSinks(videodata);
    }
    IME_UpdateInputLocale(videodata);
    IME_Disable(videodata, hwnd);
}

static void IME_Enable(SDL_VideoData *videodata, HWND hwnd)
{
    if (!videodata->ime_initialized || !videodata->ime_hwnd_current) {
        return;
    }

    if (!videodata->ime_available) {
        IME_Disable(videodata, hwnd);
        return;
    }
    if (videodata->ime_hwnd_current == videodata->ime_hwnd_main) {
        ImmAssociateContext(videodata->ime_hwnd_current, videodata->ime_himc);
    }

    videodata->ime_enabled = SDL_TRUE;
    IME_UpdateInputLocale(videodata);
    UILess_EnableUIUpdates(videodata);
}

static void IME_Disable(SDL_VideoData *videodata, HWND hwnd)
{
    if (!videodata->ime_initialized || !videodata->ime_hwnd_current) {
        return;
    }

    IME_ClearComposition(videodata);
    if (videodata->ime_hwnd_current == videodata->ime_hwnd_main) {
        ImmAssociateContext(videodata->ime_hwnd_current, (HIMC)0);
    }

    videodata->ime_enabled = SDL_FALSE;
    UILess_DisableUIUpdates(videodata);
}

static void IME_Quit(SDL_VideoData *videodata)
{
    if (!videodata->ime_initialized) {
        return;
    }

    UILess_ReleaseSinks(videodata);
    if (videodata->ime_hwnd_main) {
        ImmAssociateContext(videodata->ime_hwnd_main, videodata->ime_himc);
    }

    videodata->ime_hwnd_main = 0;
    videodata->ime_himc = 0;
    if (videodata->ime_himm32) {
        SDL_UnloadObject(videodata->ime_himm32);
        videodata->ime_himm32 = 0;
    }
    if (videodata->ime_threadmgr) {
        videodata->ime_threadmgr->lpVtbl->Release(videodata->ime_threadmgr);
        videodata->ime_threadmgr = 0;
    }
    if (videodata->ime_com_initialized) {
        WIN_CoUninitialize();
        videodata->ime_com_initialized = SDL_FALSE;
    }
    IME_DestroyTextures(videodata);
    videodata->ime_initialized = SDL_FALSE;
}

static void IME_GetReadingString(SDL_VideoData *videodata, HWND hwnd)
{
    DWORD id = 0;
    HIMC himc = 0;
    WCHAR buffer[16];
    WCHAR *s = buffer;
    DWORD len = 0;
    INT err = 0;
    BOOL vertical = FALSE;
    UINT maxuilen = 0;

    if (videodata->ime_uiless) {
        return;
    }

    videodata->ime_readingstring[0] = 0;

    id = IME_GetId(videodata, 0);
    if (!id) {
        return;
    }

    himc = ImmGetContext(hwnd);
    if (!himc) {
        return;
    }

    if (videodata->GetReadingString) {
        len = videodata->GetReadingString(himc, 0, 0, &err, &vertical, &maxuilen);
        if (len) {
            if (len > SDL_arraysize(buffer)) {
                len = SDL_arraysize(buffer);
            }

            len = videodata->GetReadingString(himc, len, s, &err, &vertical, &maxuilen);
        }
        SDL_wcslcpy(videodata->ime_readingstring, s, len);
    } else {
        LPINPUTCONTEXT2 lpimc = videodata->ImmLockIMC(himc);
        LPBYTE p = 0;
        s = 0;
        switch (id) {
        case IMEID_CHT_VER42:
        case IMEID_CHT_VER43:
        case IMEID_CHT_VER44:
            p = *(LPBYTE *)((LPBYTE)videodata->ImmLockIMCC(lpimc->hPrivate) + 24);
            if (!p) {
                break;
            }

            len = *(DWORD *)(p + 7 * 4 + 32 * 4);
            s = (WCHAR *)(p + 56);
            break;
        case IMEID_CHT_VER51:
        case IMEID_CHT_VER52:
        case IMEID_CHS_VER53:
            p = *(LPBYTE *)((LPBYTE)videodata->ImmLockIMCC(lpimc->hPrivate) + 4);
            if (!p) {
                break;
            }

            p = *(LPBYTE *)(p + 1 * 4 + 5 * 4);
            if (!p) {
                break;
            }

            len = *(DWORD *)(p + 1 * 4 + (16 * 2 + 2 * 4) + 5 * 4 + 16 * 2);
            s = (WCHAR *)(p + 1 * 4 + (16 * 2 + 2 * 4) + 5 * 4);
            break;
        case IMEID_CHS_VER41:
        {
            int offset = (IME_GetId(videodata, 1) >= 0x00000002) ? 8 : 7;
            p = *(LPBYTE *)((LPBYTE)videodata->ImmLockIMCC(lpimc->hPrivate) + offset * 4);
            if (!p) {
                break;
            }

            len = *(DWORD *)(p + 7 * 4 + 16 * 2 * 4);
            s = (WCHAR *)(p + 6 * 4 + 16 * 2 * 1);
        } break;
        case IMEID_CHS_VER42:
            p = *(LPBYTE *)((LPBYTE)videodata->ImmLockIMCC(lpimc->hPrivate) + 1 * 4 + 1 * 4 + 6 * 4);
            if (!p) {
                break;
            }

            len = *(DWORD *)(p + 1 * 4 + (16 * 2 + 2 * 4) + 5 * 4 + 16 * 2);
            s = (WCHAR *)(p + 1 * 4 + (16 * 2 + 2 * 4) + 5 * 4);
            break;
        }
        if (s) {
            size_t size = SDL_min((size_t)(len + 1), SDL_arraysize(videodata->ime_readingstring));
            SDL_wcslcpy(videodata->ime_readingstring, s, size);
        }

        videodata->ImmUnlockIMCC(lpimc->hPrivate);
        videodata->ImmUnlockIMC(himc);
    }
    ImmReleaseContext(hwnd, himc);
    IME_SendEditingEvent(videodata);
}

static void IME_InputLangChanged(SDL_VideoData *videodata)
{
    UINT lang = PRIMLANG();
    IME_UpdateInputLocale(videodata);
    if (!videodata->ime_uiless) {
        videodata->ime_candlistindexbase = (videodata->ime_hkl == CHT_HKL_DAYI) ? 0 : 1;
    }

    IME_SetupAPI(videodata);
    if (lang != PRIMLANG()) {
        IME_ClearComposition(videodata);
    }
}

static DWORD IME_GetId(SDL_VideoData *videodata, UINT uIndex)
{
    static HKL hklprev = 0;
    static DWORD dwRet[2] = { 0 };
    DWORD dwVerSize = 0;
    DWORD dwVerHandle = 0;
    LPVOID lpVerBuffer = 0;
    LPVOID lpVerData = 0;
    UINT cbVerData = 0;
    char szTemp[256];
    HKL hkl = 0;
    DWORD dwLang = 0;
    SDL_assert(uIndex < sizeof(dwRet) / sizeof(dwRet[0]));

    hkl = videodata->ime_hkl;
    if (hklprev == hkl) {
        return dwRet[uIndex];
    }
    hklprev = hkl;

    SDL_assert(uIndex == 0);
    dwLang = ((DWORD_PTR)hkl & 0xffff);
    if (videodata->ime_uiless && dwLang == LANG_CHT) {
        dwRet[0] = IMEID_CHT_VER_VISTA;
        dwRet[1] = 0;
        return dwRet[0];
    }
    if (hkl != CHT_HKL_NEW_PHONETIC && hkl != CHT_HKL_NEW_CHANG_JIE && hkl != CHT_HKL_NEW_QUICK && hkl != CHT_HKL_HK_CANTONESE && hkl != CHS_HKL) {
        dwRet[0] = dwRet[1] = 0;
        return dwRet[0];
    }
    if (!ImmGetIMEFileNameA(hkl, szTemp, sizeof(szTemp) - 1)) {
        dwRet[0] = dwRet[1] = 0;
        return dwRet[0];
    }
    if (!videodata->GetReadingString) {
#define LCID_INVARIANT MAKELCID(MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US), SORT_DEFAULT)
        if (CompareStringA(LCID_INVARIANT, NORM_IGNORECASE, szTemp, -1, CHT_IMEFILENAME1, -1) != 2 && CompareStringA(LCID_INVARIANT, NORM_IGNORECASE, szTemp, -1, CHT_IMEFILENAME2, -1) != 2 && CompareStringA(LCID_INVARIANT, NORM_IGNORECASE, szTemp, -1, CHT_IMEFILENAME3, -1) != 2 && CompareStringA(LCID_INVARIANT, NORM_IGNORECASE, szTemp, -1, CHS_IMEFILENAME1, -1) != 2 && CompareStringA(LCID_INVARIANT, NORM_IGNORECASE, szTemp, -1, CHS_IMEFILENAME2, -1) != 2) {
            dwRet[0] = dwRet[1] = 0;
            return dwRet[0];
        }
#undef LCID_INVARIANT
        dwVerSize = GetFileVersionInfoSizeA(szTemp, &dwVerHandle);
        if (dwVerSize) {
            lpVerBuffer = SDL_malloc(dwVerSize);
            if (lpVerBuffer) {
                if (GetFileVersionInfoA(szTemp, dwVerHandle, dwVerSize, lpVerBuffer)) {
                    if (VerQueryValueA(lpVerBuffer, "\\", &lpVerData, &cbVerData)) {
#define pVerFixedInfo ((VS_FIXEDFILEINFO FAR *)lpVerData)
                        DWORD dwVer = pVerFixedInfo->dwFileVersionMS;
                        dwVer = (dwVer & 0x00ff0000) << 8 | (dwVer & 0x000000ff) << 16;
                        if ((videodata->GetReadingString) ||
                            ((dwLang == LANG_CHT) && (dwVer == MAKEIMEVERSION(4, 2) ||
                                                      dwVer == MAKEIMEVERSION(4, 3) ||
                                                      dwVer == MAKEIMEVERSION(4, 4) ||
                                                      dwVer == MAKEIMEVERSION(5, 0) ||
                                                      dwVer == MAKEIMEVERSION(5, 1) ||
                                                      dwVer == MAKEIMEVERSION(5, 2) ||
                                                      dwVer == MAKEIMEVERSION(6, 0))) ||
                            ((dwLang == LANG_CHS) && (dwVer == MAKEIMEVERSION(4, 1) ||
                                                      dwVer == MAKEIMEVERSION(4, 2) ||
                                                      dwVer == MAKEIMEVERSION(5, 3)))) {
                            dwRet[0] = dwVer | dwLang;
                            dwRet[1] = pVerFixedInfo->dwFileVersionLS;
                            SDL_free(lpVerBuffer);
                            return dwRet[0];
                        }
#undef pVerFixedInfo
                    }
                }
            }
            SDL_free(lpVerBuffer);
        }
    }
    dwRet[0] = dwRet[1] = 0;
    return dwRet[0];
}

static void IME_SetupAPI(SDL_VideoData *videodata)
{
    char ime_file[MAX_PATH + 1];
    void *hime = 0;
    HKL hkl = 0;
    videodata->GetReadingString = 0;
    videodata->ShowReadingWindow = 0;
    if (videodata->ime_uiless) {
        return;
    }

    hkl = videodata->ime_hkl;
    if (!ImmGetIMEFileNameA(hkl, ime_file, sizeof(ime_file) - 1)) {
        return;
    }

    hime = SDL_LoadObject(ime_file);
    if (!hime) {
        return;
    }

    /* *INDENT-OFF* */ /* clang-format off */
    videodata->GetReadingString = (UINT (WINAPI *)(HIMC, UINT, LPWSTR, PINT, BOOL*, PUINT))
        SDL_LoadFunction(hime, "GetReadingString");
    videodata->ShowReadingWindow = (BOOL (WINAPI *)(HIMC, BOOL))
        SDL_LoadFunction(hime, "ShowReadingWindow");
    /* *INDENT-ON* */ /* clang-format on */

    if (videodata->ShowReadingWindow) {
        HIMC himc = ImmGetContext(videodata->ime_hwnd_current);
        if (himc) {
            videodata->ShowReadingWindow(himc, FALSE);
            ImmReleaseContext(videodata->ime_hwnd_current, himc);
        }
    }
}

static void IME_SetWindow(SDL_VideoData *videodata, HWND hwnd)
{
    videodata->ime_hwnd_current = hwnd;
    if (videodata->ime_threadmgr) {
        struct ITfDocumentMgr *document_mgr = 0;
        if (SUCCEEDED(videodata->ime_threadmgr->lpVtbl->AssociateFocus(videodata->ime_threadmgr, hwnd, NULL, &document_mgr))) {
            if (document_mgr) {
                document_mgr->lpVtbl->Release(document_mgr);
            }
        }
    }
}

static void IME_UpdateInputLocale(SDL_VideoData *videodata)
{
    HKL hklnext = GetKeyboardLayout(0);

    if (hklnext == videodata->ime_hkl) {
        return;
    }

    videodata->ime_hkl = hklnext;
    videodata->ime_candvertical = (PRIMLANG() == LANG_KOREAN || LANG() == LANG_CHS) ? SDL_FALSE : SDL_TRUE;
}

static void IME_ClearComposition(SDL_VideoData *videodata)
{
    HIMC himc = 0;
    if (!videodata->ime_initialized) {
        return;
    }

    himc = ImmGetContext(videodata->ime_hwnd_current);
    if (!himc) {
        return;
    }

    ImmNotifyIME(himc, NI_COMPOSITIONSTR, CPS_CANCEL, 0);
    if (videodata->ime_uiless) {
        ImmSetCompositionString(himc, SCS_SETSTR, TEXT(""), sizeof(TCHAR), TEXT(""), sizeof(TCHAR));
    }

    ImmNotifyIME(himc, NI_CLOSECANDIDATE, 0, 0);
    ImmReleaseContext(videodata->ime_hwnd_current, himc);
    SDL_SendEditingText("", 0, 0);
}

static SDL_bool IME_IsTextInputShown(SDL_VideoData *videodata)
{
    if (!videodata->ime_initialized || !videodata->ime_available || !videodata->ime_enabled) {
        return SDL_FALSE;
    }

    return videodata->ime_uicontext != 0 ? SDL_TRUE : SDL_FALSE;
}

static void IME_GetCompositionString(SDL_VideoData *videodata, HIMC himc, DWORD string)
{
    LONG length;
    DWORD dwLang = ((DWORD_PTR)videodata->ime_hkl & 0xffff);

    length = ImmGetCompositionStringW(himc, string, NULL, 0);
    if (length > 0 && videodata->ime_composition_length < length) {
        if (videodata->ime_composition) {
            SDL_free(videodata->ime_composition);
        }

        videodata->ime_composition = (WCHAR *)SDL_malloc(length + sizeof(WCHAR));
        videodata->ime_composition_length = length;
    }

    length = ImmGetCompositionStringW(
        himc,
        string,
        videodata->ime_composition,
        videodata->ime_composition_length);

    if (length < 0) {
        length = 0;
    }

    length /= sizeof(WCHAR);
    videodata->ime_cursor = LOWORD(ImmGetCompositionStringW(himc, GCS_CURSORPOS, 0, 0));
    if ((dwLang == LANG_CHT || dwLang == LANG_CHS) &&
        videodata->ime_cursor > 0 &&
        videodata->ime_cursor < (int)(videodata->ime_composition_length / sizeof(WCHAR)) &&
        (videodata->ime_composition[0] == 0x3000 || videodata->ime_composition[0] == 0x0020)) {
        // Traditional Chinese IMEs add a placeholder U+3000
        // Simplified Chinese IMEs seem to add a placeholder U+0020 sometimes
        int i;
        for (i = videodata->ime_cursor + 1; i < length; ++i) {
            videodata->ime_composition[i - 1] = videodata->ime_composition[i];
        }

        --length;
    }

    videodata->ime_composition[length] = 0;

    // Get the correct caret position if we've selected a candidate from the candidate window
    if (videodata->ime_cursor == 0 && length > 0) {
        Sint32 start = 0;
        Sint32 end = 0;

        length = ImmGetCompositionStringW(himc, GCS_COMPATTR, NULL, 0);
        if (length > 0) {
            Uint8 *attributes = (Uint8 *)SDL_malloc(length + sizeof(WCHAR));
            ImmGetCompositionString(himc, GCS_COMPATTR, attributes, length);

            for (start = 0; start < length; ++start) {
                if (attributes[start] == ATTR_TARGET_CONVERTED || attributes[start] == ATTR_TARGET_NOTCONVERTED) {
                    break;
                }
            }

            for (end = start; end < length; ++end) {
                if (attributes[end] != ATTR_TARGET_CONVERTED && attributes[end] != ATTR_TARGET_NOTCONVERTED) {
                    break;
                }
            }

            if (start == length) {
                start = 0;
                end = length;
            }

            SDL_free(attributes);
        }

        videodata->ime_cursor = end;
    }
}

static void IME_SendInputEvent(SDL_VideoData *videodata)
{
    char *s = 0;
    s = WIN_StringToUTF8W(videodata->ime_composition);
    SDL_SendKeyboardText(s);
    SDL_free(s);

    videodata->ime_composition[0] = 0;
    videodata->ime_readingstring[0] = 0;
    videodata->ime_cursor = 0;
}

static void IME_SendEditingEvent(SDL_VideoData *videodata)
{
    char *s = NULL;
    WCHAR *buffer = NULL;
    size_t size = videodata->ime_composition_length;
    if (videodata->ime_readingstring[0]) {
        size_t len = SDL_min(SDL_wcslen(videodata->ime_composition), (size_t)videodata->ime_cursor);

        size += sizeof(videodata->ime_readingstring);
        buffer = (WCHAR *)SDL_malloc(size + sizeof(WCHAR));
        buffer[0] = 0;

        SDL_wcslcpy(buffer, videodata->ime_composition, len + 1);
        SDL_wcslcat(buffer, videodata->ime_readingstring, size);
        SDL_wcslcat(buffer, &videodata->ime_composition[len], size);
    } else {
        buffer = (WCHAR *)SDL_malloc(size + sizeof(WCHAR));
        buffer[0] = 0;
        SDL_wcslcpy(buffer, videodata->ime_composition, size);
    }

    s = WIN_StringToUTF8W(buffer);
    SDL_SendEditingText(s, videodata->ime_cursor + (int)SDL_wcslen(videodata->ime_readingstring), 0);
    SDL_free(s);
    SDL_free(buffer);
}

static void IME_AddCandidate(SDL_VideoData *videodata, UINT i, LPCWSTR candidate)
{
    LPWSTR dst = &videodata->ime_candidates[i * MAX_CANDLENGTH];
    LPWSTR end = &dst[MAX_CANDLENGTH - 1];
    SDL_COMPILE_TIME_ASSERT(IME_CANDIDATE_INDEXING_REQUIRES, MAX_CANDLIST == 10);
    *dst++ = (WCHAR)(TEXT('0') + ((i + videodata->ime_candlistindexbase) % 10));
    if (videodata->ime_candvertical) {
        *dst++ = TEXT(' ');
    }

    while (*candidate && dst < end) {
        *dst++ = *candidate++;
    }

    *dst = (WCHAR)'\0';
}

static void IME_GetCandidateList(HWND hwnd, SDL_VideoData *videodata)
{
    HIMC himc;
    DWORD size;
    LPCANDIDATELIST cand_list;

    if (IME_ShowCandidateList(videodata) < 0) {
        return;
    }
    himc = ImmGetContext(hwnd);
    if (!himc) {
        return;
    }
    size = ImmGetCandidateListW(himc, 0, 0, 0);
    if (size != 0) {
        cand_list = (LPCANDIDATELIST)SDL_malloc(size);
        if (cand_list != NULL) {
            size = ImmGetCandidateListW(himc, 0, cand_list, size);
            if (size != 0) {
                UINT i, j;
                UINT page_start = 0;
                videodata->ime_candsel = cand_list->dwSelection;
                videodata->ime_candcount = cand_list->dwCount;

                if (LANG() == LANG_CHS && IME_GetId(videodata, 0)) {
                    const UINT maxcandchar = 18;
                    size_t cchars = 0;

                    for (i = 0; i < videodata->ime_candcount; ++i) {
                        size_t len = SDL_wcslen((LPWSTR)((DWORD_PTR)cand_list + cand_list->dwOffset[i])) + 1;
                        if (len + cchars > maxcandchar) {
                            if (i > cand_list->dwSelection) {
                                break;
                            }

                            page_start = i;
                            cchars = len;
                        } else {
                            cchars += len;
                        }
                    }
                    videodata->ime_candpgsize = i - page_start;
                } else {
                    videodata->ime_candpgsize = SDL_min(cand_list->dwPageSize == 0 ? MAX_CANDLIST : cand_list->dwPageSize, MAX_CANDLIST);
                    page_start = (cand_list->dwSelection / videodata->ime_candpgsize) * videodata->ime_candpgsize;
                }
                for (i = page_start, j = 0; (DWORD)i < cand_list->dwCount && j < videodata->ime_candpgsize; i++, j++) {
                    LPCWSTR candidate = (LPCWSTR)((DWORD_PTR)cand_list + cand_list->dwOffset[i]);
                    IME_AddCandidate(videodata, j, candidate);
                }
                // TODO: why was this necessary? check ime_candvertical instead? PRIMLANG() never equals LANG_CHT !
                // if (PRIMLANG() == LANG_KOREAN || (PRIMLANG() == LANG_CHT && !IME_GetId(videodata, 0)))
                //    videodata->ime_candsel = -1;
            }
            SDL_free(cand_list);
        }
    }
    ImmReleaseContext(hwnd, himc);
}

static int IME_ShowCandidateList(SDL_VideoData *videodata)
{
    void *candidates;

    videodata->ime_candcount = 0;
    candidates = SDL_realloc(videodata->ime_candidates, MAX_CANDSIZE);
    if (candidates) {
        videodata->ime_candidates = (WCHAR *)candidates;
    }

    if (!videodata->ime_candidates) {
        return -1;
    }

    SDL_memset(videodata->ime_candidates, 0, MAX_CANDSIZE);

    videodata->ime_dirty = SDL_TRUE;
    videodata->ime_candlist = SDL_TRUE;
    IME_DestroyTextures(videodata);
    IME_SendEditingEvent(videodata);
    return 0;
}

static void IME_HideCandidateList(SDL_VideoData *videodata)
{
    videodata->ime_dirty = SDL_FALSE;
    videodata->ime_candlist = SDL_FALSE;
    IME_DestroyTextures(videodata);
    IME_SendEditingEvent(videodata);
}

SDL_bool IME_HandleMessage(HWND hwnd, UINT msg, WPARAM wParam, LPARAM *lParam, SDL_VideoData *videodata)
{
    SDL_bool trap = SDL_FALSE;
    HIMC himc = 0;
    if (!videodata->ime_initialized || !videodata->ime_available || !videodata->ime_enabled) {
        return SDL_FALSE;
    }

    switch (msg) {
    case WM_KEYDOWN:
        if (wParam == VK_PROCESSKEY) {
            videodata->ime_uicontext = 1;
            trap = SDL_TRUE;
        } else {
            videodata->ime_uicontext = 0;
        }
        break;
    case WM_INPUTLANGCHANGE:
        IME_InputLangChanged(videodata);
        break;
    case WM_IME_SETCONTEXT:
        if (videodata->ime_uiless) {
            *lParam = 0;
        }
        break;
    case WM_IME_STARTCOMPOSITION:
        videodata->ime_suppress_endcomposition_event = SDL_FALSE;
        trap = SDL_TRUE;
        break;
    case WM_IME_COMPOSITION:
        trap = SDL_TRUE;
        himc = ImmGetContext(hwnd);
        if (*lParam & GCS_RESULTSTR) {
            videodata->ime_suppress_endcomposition_event = SDL_TRUE;
            IME_GetCompositionString(videodata, himc, GCS_RESULTSTR);
            SDL_SendEditingText("", 0, 0);
            IME_SendInputEvent(videodata);
        }
        if (*lParam & GCS_COMPSTR) {
            if (!videodata->ime_uiless) {
                videodata->ime_readingstring[0] = 0;
            }

            IME_GetCompositionString(videodata, himc, GCS_COMPSTR);
            IME_SendEditingEvent(videodata);
        }
        ImmReleaseContext(hwnd, himc);
        break;
    case WM_IME_ENDCOMPOSITION:
        videodata->ime_uicontext = 0;
        videodata->ime_composition[0] = 0;
        videodata->ime_readingstring[0] = 0;
        videodata->ime_cursor = 0;
        if (videodata->ime_suppress_endcomposition_event == SDL_FALSE) {
            SDL_SendEditingText("", 0, 0);
        }
        videodata->ime_suppress_endcomposition_event = SDL_FALSE;
        break;
    case WM_IME_NOTIFY:
        switch (wParam) {
        case IMN_SETCONVERSIONMODE:
        case IMN_SETOPENSTATUS:
            IME_UpdateInputLocale(videodata);
            break;
        case IMN_OPENCANDIDATE:
        case IMN_CHANGECANDIDATE:
            if (videodata->ime_uiless) {
                break;
            }

            trap = SDL_TRUE;
            videodata->ime_uicontext = 1;
            IME_GetCandidateList(hwnd, videodata);
            break;
        case IMN_CLOSECANDIDATE:
            trap = SDL_TRUE;
            videodata->ime_uicontext = 0;
            IME_HideCandidateList(videodata);
            break;
        case IMN_PRIVATE:
        {
            DWORD dwId = IME_GetId(videodata, 0);
            IME_GetReadingString(videodata, hwnd);
            switch (dwId) {
            case IMEID_CHT_VER42:
            case IMEID_CHT_VER43:
            case IMEID_CHT_VER44:
            case IMEID_CHS_VER41:
            case IMEID_CHS_VER42:
                if (*lParam == 1 || *lParam == 2) {
                    trap = SDL_TRUE;
                }

                break;
            case IMEID_CHT_VER50:
            case IMEID_CHT_VER51:
            case IMEID_CHT_VER52:
            case IMEID_CHT_VER60:
            case IMEID_CHS_VER53:
                if (*lParam == 16 || *lParam == 17 || *lParam == 26 || *lParam == 27 || *lParam == 28) {
                    trap = SDL_TRUE;
                }
                break;
            }
        } break;
        default:
            trap = SDL_TRUE;
            break;
        }
        break;
    }
    return trap;
}

static void IME_CloseCandidateList(SDL_VideoData *videodata)
{
    IME_HideCandidateList(videodata);
    videodata->ime_candcount = 0;
    SDL_free(videodata->ime_candidates);
    videodata->ime_candidates = NULL;
}

static void UILess_GetCandidateList(SDL_VideoData *videodata, ITfCandidateListUIElement *pcandlist)
{
    UINT selection = 0;
    UINT count = 0;
    UINT page = 0;
    UINT pgcount = 0;
    DWORD pgstart = 0;
    DWORD pgsize = 0;
    UINT i, j;

    if (IME_ShowCandidateList(videodata) < 0) {
        return;
    }

    pcandlist->lpVtbl->GetSelection(pcandlist, &selection);
    pcandlist->lpVtbl->GetCount(pcandlist, &count);
    pcandlist->lpVtbl->GetCurrentPage(pcandlist, &page);

    videodata->ime_candsel = selection;
    videodata->ime_candcount = count;

    pcandlist->lpVtbl->GetPageIndex(pcandlist, 0, 0, &pgcount);
    if (pgcount > 0) {
        UINT *idxlist = SDL_malloc(sizeof(UINT) * pgcount);
        if (idxlist) {
            pcandlist->lpVtbl->GetPageIndex(pcandlist, idxlist, pgcount, &pgcount);
            pgstart = idxlist[page];
            if (page < pgcount - 1) {
                pgsize = SDL_min(count, idxlist[page + 1]) - pgstart;
            } else {
                pgsize = count - pgstart;
            }

            SDL_free(idxlist);
        }
    }
    videodata->ime_candpgsize = SDL_min(pgsize, MAX_CANDLIST);
    videodata->ime_candsel = videodata->ime_candsel - pgstart;
    for (i = pgstart, j = 0; (DWORD)i < count && j < videodata->ime_candpgsize; i++, j++) {
        BSTR bstr;
        if (SUCCEEDED(pcandlist->lpVtbl->GetString(pcandlist, i, &bstr))) {
            if (bstr) {
                IME_AddCandidate(videodata, j, bstr);
                SysFreeString(bstr);
            }
        }
    }
    // TODO: why was this necessary? check ime_candvertical instead?
    // if (PRIMLANG() == LANG_KOREAN)
    //    videodata->ime_candsel = -1;
}

STDMETHODIMP_(ULONG)
TSFSink_AddRef(TSFSink *sink)
{
    return ++sink->refcount;
}

STDMETHODIMP_(ULONG)
TSFSink_Release(TSFSink *sink)
{
    --sink->refcount;
    if (sink->refcount == 0) {
        SDL_free(sink);
        return 0;
    }
    return sink->refcount;
}

STDMETHODIMP UIElementSink_QueryInterface(TSFSink *sink, REFIID riid, PVOID *ppv)
{
    if (!ppv) {
        return E_INVALIDARG;
    }

    *ppv = 0;
    if (WIN_IsEqualIID(riid, &IID_IUnknown)) {
        *ppv = (IUnknown *)sink;
    } else if (WIN_IsEqualIID(riid, &IID_ITfUIElementSink)) {
        *ppv = (ITfUIElementSink *)sink;
    }

    if (*ppv) {
        TSFSink_AddRef(sink);
        return S_OK;
    }
    return E_NOINTERFACE;
}

ITfUIElement *UILess_GetUIElement(SDL_VideoData *videodata, DWORD dwUIElementId)
{
    ITfUIElementMgr *puiem = 0;
    ITfUIElement *pelem = 0;
    ITfThreadMgrEx *threadmgrex = videodata->ime_threadmgrex;

    if (SUCCEEDED(threadmgrex->lpVtbl->QueryInterface(threadmgrex, &IID_ITfUIElementMgr, (LPVOID *)&puiem))) {
        puiem->lpVtbl->GetUIElement(puiem, dwUIElementId, &pelem);
        puiem->lpVtbl->Release(puiem);
    }
    return pelem;
}

STDMETHODIMP UIElementSink_BeginUIElement(TSFSink *sink, DWORD dwUIElementId, BOOL *pbShow)
{
    ITfUIElement *element = UILess_GetUIElement((SDL_VideoData *)sink->data, dwUIElementId);
    ITfReadingInformationUIElement *preading = 0;
    ITfCandidateListUIElement *pcandlist = 0;
    SDL_VideoData *videodata = (SDL_VideoData *)sink->data;
    if (!element) {
        return E_INVALIDARG;
    }

    *pbShow = FALSE;
    if (SUCCEEDED(element->lpVtbl->QueryInterface(element, &IID_ITfReadingInformationUIElement, (LPVOID *)&preading))) {
        BSTR bstr;
        if (SUCCEEDED(preading->lpVtbl->GetString(preading, &bstr)) && bstr) {
            SysFreeString(bstr);
        }
        preading->lpVtbl->Release(preading);
    } else if (SUCCEEDED(element->lpVtbl->QueryInterface(element, &IID_ITfCandidateListUIElement, (LPVOID *)&pcandlist))) {
        videodata->ime_candref++;
        UILess_GetCandidateList(videodata, pcandlist);
        pcandlist->lpVtbl->Release(pcandlist);
    }
    return S_OK;
}

STDMETHODIMP UIElementSink_UpdateUIElement(TSFSink *sink, DWORD dwUIElementId)
{
    ITfUIElement *element = UILess_GetUIElement((SDL_VideoData *)sink->data, dwUIElementId);
    ITfReadingInformationUIElement *preading = 0;
    ITfCandidateListUIElement *pcandlist = 0;
    SDL_VideoData *videodata = (SDL_VideoData *)sink->data;
    if (!element) {
        return E_INVALIDARG;
    }

    if (SUCCEEDED(element->lpVtbl->QueryInterface(element, &IID_ITfReadingInformationUIElement, (LPVOID *)&preading))) {
        BSTR bstr;
        if (SUCCEEDED(preading->lpVtbl->GetString(preading, &bstr)) && bstr) {
            WCHAR *s = (WCHAR *)bstr;
            SDL_wcslcpy(videodata->ime_readingstring, s, SDL_arraysize(videodata->ime_readingstring));
            IME_SendEditingEvent(videodata);
            SysFreeString(bstr);
        }
        preading->lpVtbl->Release(preading);
    } else if (SUCCEEDED(element->lpVtbl->QueryInterface(element, &IID_ITfCandidateListUIElement, (LPVOID *)&pcandlist))) {
        UILess_GetCandidateList(videodata, pcandlist);
        pcandlist->lpVtbl->Release(pcandlist);
    }
    return S_OK;
}

STDMETHODIMP UIElementSink_EndUIElement(TSFSink *sink, DWORD dwUIElementId)
{
    ITfUIElement *element = UILess_GetUIElement((SDL_VideoData *)sink->data, dwUIElementId);
    ITfReadingInformationUIElement *preading = 0;
    ITfCandidateListUIElement *pcandlist = 0;
    SDL_VideoData *videodata = (SDL_VideoData *)sink->data;
    if (!element) {
        return E_INVALIDARG;
    }

    if (SUCCEEDED(element->lpVtbl->QueryInterface(element, &IID_ITfReadingInformationUIElement, (LPVOID *)&preading))) {
        videodata->ime_readingstring[0] = 0;
        IME_SendEditingEvent(videodata);
        preading->lpVtbl->Release(preading);
    }
    if (SUCCEEDED(element->lpVtbl->QueryInterface(element, &IID_ITfCandidateListUIElement, (LPVOID *)&pcandlist))) {
        videodata->ime_candref--;
        if (videodata->ime_candref == 0) {
            IME_CloseCandidateList(videodata);
        }

        pcandlist->lpVtbl->Release(pcandlist);
    }
    return S_OK;
}

STDMETHODIMP IPPASink_QueryInterface(TSFSink *sink, REFIID riid, PVOID *ppv)
{
    if (!ppv) {
        return E_INVALIDARG;
    }

    *ppv = 0;
    if (WIN_IsEqualIID(riid, &IID_IUnknown)) {
        *ppv = (IUnknown *)sink;
    } else if (WIN_IsEqualIID(riid, &IID_ITfInputProcessorProfileActivationSink)) {
        *ppv = (ITfInputProcessorProfileActivationSink *)sink;
    }

    if (*ppv) {
        TSFSink_AddRef(sink);
        return S_OK;
    }
    return E_NOINTERFACE;
}

STDMETHODIMP IPPASink_OnActivated(TSFSink *sink, DWORD dwProfileType, LANGID langid, REFCLSID clsid, REFGUID catid, REFGUID guidProfile, HKL hkl, DWORD dwFlags)
{
    static const GUID SDL_TF_PROFILE_DAYI = { 0x037B2C25, 0x480C, 0x4D7F, { 0xB0, 0x27, 0xD6, 0xCA, 0x6B, 0x69, 0x78, 0x8A } };
    SDL_VideoData *videodata = (SDL_VideoData *)sink->data;
    videodata->ime_candlistindexbase = WIN_IsEqualGUID(&SDL_TF_PROFILE_DAYI, guidProfile) ? 0 : 1;
    if (WIN_IsEqualIID(catid, &GUID_TFCAT_TIP_KEYBOARD) && (dwFlags & TF_IPSINK_FLAG_ACTIVE)) {
        IME_InputLangChanged((SDL_VideoData *)sink->data);
    }

    IME_HideCandidateList(videodata);
    return S_OK;
}

static void *vtUIElementSink[] = {
    (void *)(UIElementSink_QueryInterface),
    (void *)(TSFSink_AddRef),
    (void *)(TSFSink_Release),
    (void *)(UIElementSink_BeginUIElement),
    (void *)(UIElementSink_UpdateUIElement),
    (void *)(UIElementSink_EndUIElement)
};

static void *vtIPPASink[] = {
    (void *)(IPPASink_QueryInterface),
    (void *)(TSFSink_AddRef),
    (void *)(TSFSink_Release),
    (void *)(IPPASink_OnActivated)
};

static void UILess_EnableUIUpdates(SDL_VideoData *videodata)
{
    ITfSource *source = 0;
    if (!videodata->ime_threadmgrex || videodata->ime_uielemsinkcookie != TF_INVALID_COOKIE) {
        return;
    }

    if (SUCCEEDED(videodata->ime_threadmgrex->lpVtbl->QueryInterface(videodata->ime_threadmgrex, &IID_ITfSource, (LPVOID *)&source))) {
        source->lpVtbl->AdviseSink(source, &IID_ITfUIElementSink, (IUnknown *)videodata->ime_uielemsink, &videodata->ime_uielemsinkcookie);
        source->lpVtbl->Release(source);
    }
}

static void UILess_DisableUIUpdates(SDL_VideoData *videodata)
{
    ITfSource *source = 0;
    if (!videodata->ime_threadmgrex || videodata->ime_uielemsinkcookie == TF_INVALID_COOKIE) {
        return;
    }

    if (SUCCEEDED(videodata->ime_threadmgrex->lpVtbl->QueryInterface(videodata->ime_threadmgrex, &IID_ITfSource, (LPVOID *)&source))) {
        source->lpVtbl->UnadviseSink(source, videodata->ime_uielemsinkcookie);
        videodata->ime_uielemsinkcookie = TF_INVALID_COOKIE;
        source->lpVtbl->Release(source);
    }
}

static SDL_bool UILess_SetupSinks(SDL_VideoData *videodata)
{
    TfClientId clientid = 0;
    SDL_bool result = SDL_FALSE;
    ITfSource *source = 0;
    if (FAILED(CoCreateInstance(&CLSID_TF_ThreadMgr, NULL, CLSCTX_INPROC_SERVER, &IID_ITfThreadMgrEx, (LPVOID *)&videodata->ime_threadmgrex))) {
        return SDL_FALSE;
    }

    if (FAILED(videodata->ime_threadmgrex->lpVtbl->ActivateEx(videodata->ime_threadmgrex, &clientid, TF_TMAE_UIELEMENTENABLEDONLY))) {
        return SDL_FALSE;
    }

    videodata->ime_uielemsink = SDL_malloc(sizeof(TSFSink));
    videodata->ime_ippasink = SDL_malloc(sizeof(TSFSink));

    videodata->ime_uielemsink->lpVtbl = vtUIElementSink;
    videodata->ime_uielemsink->refcount = 1;
    videodata->ime_uielemsink->data = videodata;

    videodata->ime_ippasink->lpVtbl = vtIPPASink;
    videodata->ime_ippasink->refcount = 1;
    videodata->ime_ippasink->data = videodata;

    if (SUCCEEDED(videodata->ime_threadmgrex->lpVtbl->QueryInterface(videodata->ime_threadmgrex, &IID_ITfSource, (LPVOID *)&source))) {
        if (SUCCEEDED(source->lpVtbl->AdviseSink(source, &IID_ITfUIElementSink, (IUnknown *)videodata->ime_uielemsink, &videodata->ime_uielemsinkcookie))) {
            if (SUCCEEDED(source->lpVtbl->AdviseSink(source, &IID_ITfInputProcessorProfileActivationSink, (IUnknown *)videodata->ime_ippasink, &videodata->ime_alpnsinkcookie))) {
                result = SDL_TRUE;
            }
        }
        source->lpVtbl->Release(source);
    }
    return result;
}

#define SAFE_RELEASE(p)                \
    {                                  \
        if (p) {                       \
            (p)->lpVtbl->Release((p)); \
            (p) = 0;                   \
        }                              \
    }

static void UILess_ReleaseSinks(SDL_VideoData *videodata)
{
    ITfSource *source = 0;
    if (videodata->ime_threadmgrex && SUCCEEDED(videodata->ime_threadmgrex->lpVtbl->QueryInterface(videodata->ime_threadmgrex, &IID_ITfSource, (LPVOID *)&source))) {
        source->lpVtbl->UnadviseSink(source, videodata->ime_uielemsinkcookie);
        source->lpVtbl->UnadviseSink(source, videodata->ime_alpnsinkcookie);
        SAFE_RELEASE(source);
        videodata->ime_threadmgrex->lpVtbl->Deactivate(videodata->ime_threadmgrex);
        SAFE_RELEASE(videodata->ime_threadmgrex);
        TSFSink_Release(videodata->ime_uielemsink);
        videodata->ime_uielemsink = 0;
        TSFSink_Release(videodata->ime_ippasink);
        videodata->ime_ippasink = 0;
    }
}

static void *StartDrawToBitmap(HDC hdc, HBITMAP *hhbm, int width, int height)
{
    BITMAPINFO info;
    BITMAPINFOHEADER *infoHeader = &info.bmiHeader;
    BYTE *bits = NULL;
    if (hhbm) {
        SDL_zero(info);
        infoHeader->biSize = sizeof(BITMAPINFOHEADER);
        infoHeader->biWidth = width;
        infoHeader->biHeight = (LONG)-1 * SDL_abs(height);
        infoHeader->biPlanes = 1;
        infoHeader->biBitCount = 32;
        infoHeader->biCompression = BI_RGB;
        *hhbm = CreateDIBSection(hdc, &info, DIB_RGB_COLORS, (void **)&bits, 0, 0);
        if (*hhbm) {
            SelectObject(hdc, *hhbm);
        }
    }
    return bits;
}

static void StopDrawToBitmap(HDC hdc, HBITMAP *hhbm)
{
    if (hhbm && *hhbm) {
        DeleteObject(*hhbm);
        *hhbm = NULL;
    }
}

/* This draws only within the specified area and fills the entire region. */
static void DrawRect(HDC hdc, int left, int top, int right, int bottom, int pensize)
{
    /* The case of no pen (PenSize = 0) is automatically taken care of. */
    const int penadjust = (int)SDL_floor(pensize / 2.0f - 0.5f);
    left += pensize / 2;
    top += pensize / 2;
    right -= penadjust;
    bottom -= penadjust;
    Rectangle(hdc, left, top, right, bottom);
}

static void IME_DestroyTextures(SDL_VideoData *videodata)
{
}

#define SDL_swap(a, b) \
    {                  \
        int c = (a);   \
        (a) = (b);     \
        (b) = c;       \
    }

static void IME_PositionCandidateList(SDL_VideoData *videodata, SIZE size)
{
    int left, top, right, bottom;
    SDL_bool ok = SDL_FALSE;
    int winw = videodata->ime_winwidth;
    int winh = videodata->ime_winheight;

    /* Bottom */
    left = videodata->ime_rect.x;
    top = videodata->ime_rect.y + videodata->ime_rect.h;
    right = left + size.cx;
    bottom = top + size.cy;
    if (right >= winw) {
        left -= right - winw;
        right = winw;
    }
    if (bottom < winh) {
        ok = SDL_TRUE;
    }

    /* Top */
    if (!ok) {
        left = videodata->ime_rect.x;
        top = videodata->ime_rect.y - size.cy;
        right = left + size.cx;
        bottom = videodata->ime_rect.y;
        if (right >= winw) {
            left -= right - winw;
            right = winw;
        }
        if (top >= 0) {
            ok = SDL_TRUE;
        }
    }

    /* Right */
    if (!ok) {
        left = videodata->ime_rect.x + size.cx;
        top = 0;
        right = left + size.cx;
        bottom = size.cy;
        if (right < winw) {
            ok = SDL_TRUE;
        }
    }

    /* Left */
    if (!ok) {
        left = videodata->ime_rect.x - size.cx;
        top = 0;
        right = videodata->ime_rect.x;
        bottom = size.cy;
        if (right >= 0) {
            ok = SDL_TRUE;
        }
    }

    /* Window too small, show at (0,0) */
    if (!ok) {
        left = 0;
        top = 0;
        right = size.cx;
        bottom = size.cy;
    }

    videodata->ime_candlistrect.x = left;
    videodata->ime_candlistrect.y = top;
    videodata->ime_candlistrect.w = right - left;
    videodata->ime_candlistrect.h = bottom - top;
}

static void IME_RenderCandidateList(SDL_VideoData *videodata, HDC hdc)
{
    int i, j;
    SIZE size = { 0 };
    SIZE candsizes[MAX_CANDLIST];
    SIZE maxcandsize = { 0 };
    HBITMAP hbm = NULL;
    int candcount = SDL_min(SDL_min(MAX_CANDLIST, videodata->ime_candcount), videodata->ime_candpgsize);
    SDL_bool vertical = videodata->ime_candvertical;

    const int listborder = 1;
    const int listpadding = 0;
    const int listbordercolor = RGB(0xB4, 0xC7, 0xAA);
    const int listfillcolor = RGB(255, 255, 255);

    const int candborder = 1;
    const int candpadding = 0;
    const int candmargin = 1;
    const COLORREF candbordercolor = RGB(255, 255, 255);
    const COLORREF candfillcolor = RGB(255, 255, 255);
    const COLORREF candtextcolor = RGB(0, 0, 0);
    const COLORREF selbordercolor = RGB(0x84, 0xAC, 0xDD);
    const COLORREF selfillcolor = RGB(0xD2, 0xE6, 0xFF);
    const COLORREF seltextcolor = RGB(0, 0, 0);
    const int horzcandspacing = 5;

    HPEN listpen = listborder != 0 ? CreatePen(PS_SOLID, listborder, listbordercolor) : (HPEN)GetStockObject(NULL_PEN);
    HBRUSH listbrush = CreateSolidBrush(listfillcolor);
    HPEN candpen = candborder != 0 ? CreatePen(PS_SOLID, candborder, candbordercolor) : (HPEN)GetStockObject(NULL_PEN);
    HBRUSH candbrush = CreateSolidBrush(candfillcolor);
    HPEN selpen = candborder != 0 ? CreatePen(PS_DOT, candborder, selbordercolor) : (HPEN)GetStockObject(NULL_PEN);
    HBRUSH selbrush = CreateSolidBrush(selfillcolor);
    HFONT font = CreateFont((int)(1 + videodata->ime_rect.h * 0.75f), 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_CHARACTER_PRECIS, CLIP_DEFAULT_PRECIS, PROOF_QUALITY, VARIABLE_PITCH | FF_SWISS, TEXT("Microsoft Sans Serif"));

    SetBkMode(hdc, TRANSPARENT);
    SelectObject(hdc, font);

    for (i = 0; i < candcount; ++i) {
        const WCHAR *s = &videodata->ime_candidates[i * MAX_CANDLENGTH];
        if (!*s) {
            candcount = i;
            break;
        }

        GetTextExtentPoint32W(hdc, s, (int)SDL_wcslen(s), &candsizes[i]);
        maxcandsize.cx = SDL_max(maxcandsize.cx, candsizes[i].cx);
        maxcandsize.cy = SDL_max(maxcandsize.cy, candsizes[i].cy);
    }
    if (vertical) {
        size.cx =
            (listborder * 2) +
            (listpadding * 2) +
            (candmargin * 2) +
            (candborder * 2) +
            (candpadding * 2) +
            (maxcandsize.cx);
        size.cy =
            (listborder * 2) +
            (listpadding * 2) +
            ((candcount + 1) * candmargin) +
            (candcount * candborder * 2) +
            (candcount * candpadding * 2) +
            (candcount * maxcandsize.cy);
    } else {
        size.cx =
            (LONG)(listborder * 2) +
            (listpadding * 2) +
            ((candcount + 1) * candmargin) +
            (candcount * candborder * 2) +
            (candcount * candpadding * 2) +
            ((candcount - 1) * horzcandspacing);

        for (i = 0; i < candcount; ++i) {
            size.cx += candsizes[i].cx;
        }

        size.cy =
            (listborder * 2) +
            (listpadding * 2) +
            (candmargin * 2) +
            (candborder * 2) +
            (candpadding * 2) +
            (maxcandsize.cy);
    }

    StartDrawToBitmap(hdc, &hbm, size.cx, size.cy);

    SelectObject(hdc, listpen);
    SelectObject(hdc, listbrush);
    DrawRect(hdc, 0, 0, size.cx, size.cy, listborder);

    SelectObject(hdc, candpen);
    SelectObject(hdc, candbrush);
    SetTextColor(hdc, candtextcolor);
    SetBkMode(hdc, TRANSPARENT);

    for (i = 0; i < candcount; ++i) {
        const WCHAR *s = &videodata->ime_candidates[i * MAX_CANDLENGTH];
        int left, top, right, bottom;

        if (vertical) {
            left = listborder + listpadding + candmargin;
            top = listborder + listpadding + (i * candborder * 2) + (i * candpadding * 2) + ((i + 1) * candmargin) + (i * maxcandsize.cy);
            right = size.cx - listborder - listpadding - candmargin;
            bottom = top + maxcandsize.cy + (candpadding * 2) + (candborder * 2);
        } else {
            left = listborder + listpadding + (i * candborder * 2) + (i * candpadding * 2) + ((i + 1) * candmargin) + (i * horzcandspacing);

            for (j = 0; j < i; ++j) {
                left += candsizes[j].cx;
            }

            top = listborder + listpadding + candmargin;
            right = left + candsizes[i].cx + (candpadding * 2) + (candborder * 2);
            bottom = size.cy - listborder - listpadding - candmargin;
        }

        if (i == videodata->ime_candsel) {
            SelectObject(hdc, selpen);
            SelectObject(hdc, selbrush);
            SetTextColor(hdc, seltextcolor);
        } else {
            SelectObject(hdc, candpen);
            SelectObject(hdc, candbrush);
            SetTextColor(hdc, candtextcolor);
        }

        DrawRect(hdc, left, top, right, bottom, candborder);
        ExtTextOutW(hdc, left + candborder + candpadding, top + candborder + candpadding, 0, NULL, s, (int)SDL_wcslen(s), NULL);
    }
    StopDrawToBitmap(hdc, &hbm);

    DeleteObject(listpen);
    DeleteObject(listbrush);
    DeleteObject(candpen);
    DeleteObject(candbrush);
    DeleteObject(selpen);
    DeleteObject(selbrush);
    DeleteObject(font);

    IME_PositionCandidateList(videodata, size);
}

static void IME_Render(SDL_VideoData *videodata)
{
    HDC hdc = CreateCompatibleDC(NULL);

    if (videodata->ime_candlist) {
        IME_RenderCandidateList(videodata, hdc);
    }

    DeleteDC(hdc);

    videodata->ime_dirty = SDL_FALSE;
}

void IME_Present(SDL_VideoData *videodata)
{
    if (videodata->ime_dirty) {
        IME_Render(videodata);
    }

    /* FIXME: Need to show the IME bitmap */
}

SDL_bool WIN_IsTextInputShown(_THIS)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
    return IME_IsTextInputShown(videodata);
}

void WIN_ClearComposition(_THIS)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
    IME_ClearComposition(videodata);
}

#endif /* SDL_DISABLE_WINDOWS_IME */

#endif /* SDL_VIDEO_DRIVER_WINDOWS */

/* vi: set ts=4 sw=4 expandtab: */
