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

#if defined(__WIN32__) || defined(__WINRT__) || defined(__GDK__)

#include "SDL_windows.h"
#include "SDL_error.h"
#include "SDL_system.h"

#include <objbase.h> /* for CoInitialize/CoUninitialize (Win32 only) */
#if defined(HAVE_ROAPI_H)
#include <roapi.h> /* For RoInitialize/RoUninitialize (Win32 only) */
#else
typedef enum RO_INIT_TYPE
{
    RO_INIT_SINGLETHREADED = 0,
    RO_INIT_MULTITHREADED = 1
} RO_INIT_TYPE;
#endif

#ifndef _WIN32_WINNT_VISTA
#define _WIN32_WINNT_VISTA 0x0600
#endif
#ifndef _WIN32_WINNT_WIN7
#define _WIN32_WINNT_WIN7 0x0601
#endif
#ifndef _WIN32_WINNT_WIN8
#define _WIN32_WINNT_WIN8 0x0602
#endif

#ifndef LOAD_LIBRARY_SEARCH_SYSTEM32
#define LOAD_LIBRARY_SEARCH_SYSTEM32 0x00000800
#endif

/* Sets an error message based on an HRESULT */
int WIN_SetErrorFromHRESULT(const char *prefix, HRESULT hr)
{
    TCHAR buffer[1024];
    char *message;
    TCHAR *p = buffer;
    DWORD c = FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, NULL, hr, 0,
                            buffer, SDL_arraysize(buffer), NULL);
    buffer[c] = 0;
    /* kill CR/LF that FormatMessage() sticks at the end */
    while (*p) {
        if (*p == '\r') {
            *p = 0;
            break;
        }
        ++p;
    }
    message = WIN_StringToUTF8(buffer);
    SDL_SetError("%s%s%s", prefix ? prefix : "", prefix ? ": " : "", message);
    SDL_free(message);
    return -1;
}

/* Sets an error message based on GetLastError() */
int WIN_SetError(const char *prefix)
{
    return WIN_SetErrorFromHRESULT(prefix, GetLastError());
}

HRESULT
WIN_CoInitialize(void)
{
    /* SDL handles any threading model, so initialize with the default, which
       is compatible with OLE and if that doesn't work, try multi-threaded mode.

       If you need multi-threaded mode, call CoInitializeEx() before SDL_Init()
    */
#ifdef __WINRT__
    /* DLudwig: On WinRT, it is assumed that COM was initialized in main().
       CoInitializeEx is available (not CoInitialize though), however
       on WinRT, main() is typically declared with the [MTAThread]
       attribute, which, AFAIK, should initialize COM.
    */
    return S_OK;
#elif defined(__XBOXONE__) || defined(__XBOXSERIES__)
    /* On Xbox, there's no need to call CoInitializeEx (and it's not implemented) */
    return S_OK;
#else
    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (hr == RPC_E_CHANGED_MODE) {
        hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    }

    /* S_FALSE means success, but someone else already initialized. */
    /* You still need to call CoUninitialize in this case! */
    if (hr == S_FALSE) {
        return S_OK;
    }

    return hr;
#endif
}

void WIN_CoUninitialize(void)
{
#ifndef __WINRT__
    CoUninitialize();
#endif
}

#ifndef __WINRT__
void *WIN_LoadComBaseFunction(const char *name)
{
    static SDL_bool s_bLoaded;
    static HMODULE s_hComBase;

    if (!s_bLoaded) {
        s_hComBase = LoadLibraryEx(TEXT("combase.dll"), NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
        s_bLoaded = SDL_TRUE;
    }
    if (s_hComBase) {
        return GetProcAddress(s_hComBase, name);
    } else {
        return NULL;
    }
}
#endif

HRESULT
WIN_RoInitialize(void)
{
#ifdef __WINRT__
    return S_OK;
#else
    typedef HRESULT(WINAPI * RoInitialize_t)(RO_INIT_TYPE initType);
    RoInitialize_t RoInitializeFunc = (RoInitialize_t)WIN_LoadComBaseFunction("RoInitialize");
    if (RoInitializeFunc) {
        /* RO_INIT_SINGLETHREADED is equivalent to COINIT_APARTMENTTHREADED */
        HRESULT hr = RoInitializeFunc(RO_INIT_SINGLETHREADED);
        if (hr == RPC_E_CHANGED_MODE) {
            hr = RoInitializeFunc(RO_INIT_MULTITHREADED);
        }

        /* S_FALSE means success, but someone else already initialized. */
        /* You still need to call RoUninitialize in this case! */
        if (hr == S_FALSE) {
            return S_OK;
        }

        return hr;
    } else {
        return E_NOINTERFACE;
    }
#endif
}

void WIN_RoUninitialize(void)
{
#ifndef __WINRT__
    typedef void(WINAPI * RoUninitialize_t)(void);
    RoUninitialize_t RoUninitializeFunc = (RoUninitialize_t)WIN_LoadComBaseFunction("RoUninitialize");
    if (RoUninitializeFunc) {
        RoUninitializeFunc();
    }
#endif
}

#if !defined(__WINRT__) && !defined(__XBOXONE__) && !defined(__XBOXSERIES__)
static BOOL IsWindowsVersionOrGreater(WORD wMajorVersion, WORD wMinorVersion, WORD wServicePackMajor)
{
    OSVERSIONINFOEXW osvi;
    DWORDLONG const dwlConditionMask = VerSetConditionMask(
        VerSetConditionMask(
            VerSetConditionMask(
                0, VER_MAJORVERSION, VER_GREATER_EQUAL),
            VER_MINORVERSION, VER_GREATER_EQUAL),
        VER_SERVICEPACKMAJOR, VER_GREATER_EQUAL);

    SDL_zero(osvi);
    osvi.dwOSVersionInfoSize = sizeof(osvi);
    osvi.dwMajorVersion = wMajorVersion;
    osvi.dwMinorVersion = wMinorVersion;
    osvi.wServicePackMajor = wServicePackMajor;

    return VerifyVersionInfoW(&osvi, VER_MAJORVERSION | VER_MINORVERSION | VER_SERVICEPACKMAJOR, dwlConditionMask) != FALSE;
}
#endif

BOOL WIN_IsWindowsVistaOrGreater(void)
{
#if defined(__WINRT__) || defined(__XBOXONE__) || defined(__XBOXSERIES__)
    return TRUE;
#else
    return IsWindowsVersionOrGreater(HIBYTE(_WIN32_WINNT_VISTA), LOBYTE(_WIN32_WINNT_VISTA), 0);
#endif
}

BOOL WIN_IsWindows7OrGreater(void)
{
#if defined(__WINRT__) || defined(__XBOXONE__) || defined(__XBOXSERIES__)
    return TRUE;
#else
    return IsWindowsVersionOrGreater(HIBYTE(_WIN32_WINNT_WIN7), LOBYTE(_WIN32_WINNT_WIN7), 0);
#endif
}

BOOL WIN_IsWindows8OrGreater(void)
{
#if defined(__WINRT__) || defined(__XBOXONE__) || defined(__XBOXSERIES__)
    return TRUE;
#else
    return IsWindowsVersionOrGreater(HIBYTE(_WIN32_WINNT_WIN8), LOBYTE(_WIN32_WINNT_WIN8), 0);
#endif
}

/*
WAVExxxCAPS gives you 31 bytes for the device name, and just truncates if it's
longer. However, since WinXP, you can use the WAVExxxCAPS2 structure, which
will give you a name GUID. The full name is in the Windows Registry under
that GUID, located here: HKLM\System\CurrentControlSet\Control\MediaCategories

Note that drivers can report GUID_NULL for the name GUID, in which case,
Windows makes a best effort to fill in those 31 bytes in the usual place.
This info summarized from MSDN:

http://web.archive.org/web/20131027093034/http://msdn.microsoft.com/en-us/library/windows/hardware/ff536382(v=vs.85).aspx

Always look this up in the registry if possible, because the strings are
different! At least on Win10, I see "Yeti Stereo Microphone" in the
Registry, and a unhelpful "Microphone(Yeti Stereo Microph" in winmm. Sigh.

(Also, DirectSound shouldn't be limited to 32 chars, but its device enum
has the same problem.)

WASAPI doesn't need this. This is just for DirectSound/WinMM.
*/
char *WIN_LookupAudioDeviceName(const WCHAR *name, const GUID *guid)
{
#if defined(__WINRT__) || defined(__XBOXONE__) || defined(__XBOXSERIES__)
    return WIN_StringToUTF8(name); /* No registry access on WinRT/UWP and Xbox, go with what we've got. */
#else
    static const GUID nullguid = { 0 };
    const unsigned char *ptr;
    char keystr[128];
    WCHAR *strw = NULL;
    SDL_bool rc;
    HKEY hkey;
    DWORD len = 0;
    char *retval = NULL;

    if (WIN_IsEqualGUID(guid, &nullguid)) {
        return WIN_StringToUTF8(name); /* No GUID, go with what we've got. */
    }

    ptr = (const unsigned char *)guid;
    (void)SDL_snprintf(keystr, sizeof(keystr),
                       "System\\CurrentControlSet\\Control\\MediaCategories\\{%02X%02X%02X%02X-%02X%02X-%02X%02X-%02X%02X-%02X%02X%02X%02X%02X%02X}",
                       ptr[3], ptr[2], ptr[1], ptr[0], ptr[5], ptr[4], ptr[7], ptr[6],
                       ptr[8], ptr[9], ptr[10], ptr[11], ptr[12], ptr[13], ptr[14], ptr[15]);

    strw = WIN_UTF8ToString(keystr);
    rc = (RegOpenKeyExW(HKEY_LOCAL_MACHINE, strw, 0, KEY_QUERY_VALUE, &hkey) == ERROR_SUCCESS);
    SDL_free(strw);
    if (!rc) {
        return WIN_StringToUTF8(name); /* oh well. */
    }

    rc = (RegQueryValueExW(hkey, L"Name", NULL, NULL, NULL, &len) == ERROR_SUCCESS);
    if (!rc) {
        RegCloseKey(hkey);
        return WIN_StringToUTF8(name); /* oh well. */
    }

    strw = (WCHAR *)SDL_malloc(len + sizeof(WCHAR));
    if (!strw) {
        RegCloseKey(hkey);
        return WIN_StringToUTF8(name); /* oh well. */
    }

    rc = (RegQueryValueExW(hkey, L"Name", NULL, NULL, (LPBYTE)strw, &len) == ERROR_SUCCESS);
    RegCloseKey(hkey);
    if (!rc) {
        SDL_free(strw);
        return WIN_StringToUTF8(name); /* oh well. */
    }

    strw[len / 2] = 0; /* make sure it's null-terminated. */

    retval = WIN_StringToUTF8(strw);
    SDL_free(strw);
    return retval ? retval : WIN_StringToUTF8(name);
#endif /* if __WINRT__ / else */
}

BOOL WIN_IsEqualGUID(const GUID *a, const GUID *b)
{
    return SDL_memcmp(a, b, sizeof(*a)) == 0;
}

BOOL WIN_IsEqualIID(REFIID a, REFIID b)
{
    return SDL_memcmp(a, b, sizeof(*a)) == 0;
}

void WIN_RECTToRect(const RECT *winrect, SDL_Rect *sdlrect)
{
    sdlrect->x = winrect->left;
    sdlrect->w = (winrect->right - winrect->left) + 1;
    sdlrect->y = winrect->top;
    sdlrect->h = (winrect->bottom - winrect->top) + 1;
}

void WIN_RectToRECT(const SDL_Rect *sdlrect, RECT *winrect)
{
    winrect->left = sdlrect->x;
    winrect->right = sdlrect->x + sdlrect->w - 1;
    winrect->top = sdlrect->y;
    winrect->bottom = sdlrect->y + sdlrect->h - 1;
}

BOOL WIN_IsRectEmpty(const RECT *rect)
{
    /* Calculating this manually because UWP and Xbox do not support Win32 IsRectEmpty. */
    return (rect->right <= rect->left) || (rect->bottom <= rect->top);
}
#endif /* defined(__WIN32__) || defined(__WINRT__) || defined(__GDK__) */

/*
 * Public APIs
 */
#if !defined(SDL_VIDEO_DRIVER_WINDOWS)

#if defined(__WIN32__) || defined(__GDK__)
int SDL_RegisterApp(const char *name, Uint32 style, void *hInst)
{
    (void)name;
    (void)style;
    (void)hInst;
    return 0;
}

void SDL_UnregisterApp(void)
{
}

void SDL_SetWindowsMessageHook(SDL_WindowsMessageHook callback, void *userdata)
{
}
#endif /* __WIN32__ || __GDK__ */

#if defined(__WIN32__) || defined(__WINGDK__)
int SDL_Direct3D9GetAdapterIndex(int displayIndex)
{
    (void)displayIndex;
    return 0; /* D3DADAPTER_DEFAULT */
}

SDL_bool SDL_DXGIGetOutputInfo(int displayIndex, int *adapterIndex, int *outputIndex)
{
    (void)displayIndex;
    if (adapterIndex) {
        *adapterIndex = -1;
    }
    if (outputIndex) {
        *outputIndex = -1;
    }
    return SDL_FALSE;
}
#endif /* __WIN32__ || __WINGDK__ */

#endif /* !SDL_VIDEO_DRIVER_WINDOWS */

/* vi: set ts=4 sw=4 expandtab: */
