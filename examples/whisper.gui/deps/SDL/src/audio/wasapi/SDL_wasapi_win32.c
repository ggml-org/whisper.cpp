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

/* This is code that Windows uses to talk to WASAPI-related system APIs.
   This is for non-WinRT desktop apps. The C++/CX implementation of these
   functions, exclusive to WinRT, are in SDL_wasapi_winrt.cpp.
   The code in SDL_wasapi.c is used by both standard Windows and WinRT builds
   to deal with audio and calls into these functions. */

#if defined(SDL_AUDIO_DRIVER_WASAPI) && !defined(__WINRT__)

#include "../../core/windows/SDL_windows.h"
#include "../../core/windows/SDL_immdevice.h"
#include "SDL_audio.h"
#include "SDL_timer.h"
#include "../SDL_audio_c.h"
#include "../SDL_sysaudio.h"

#include <audioclient.h>

#include "SDL_wasapi.h"

/* handle to Avrt.dll--Vista and later!--for flagging the callback thread as "Pro Audio" (low latency). */
static HMODULE libavrt = NULL;
typedef HANDLE(WINAPI *pfnAvSetMmThreadCharacteristicsW)(LPCWSTR, LPDWORD);
typedef BOOL(WINAPI *pfnAvRevertMmThreadCharacteristics)(HANDLE);
static pfnAvSetMmThreadCharacteristicsW pAvSetMmThreadCharacteristicsW = NULL;
static pfnAvRevertMmThreadCharacteristics pAvRevertMmThreadCharacteristics = NULL;

/* Some GUIDs we need to know without linking to libraries that aren't available before Vista. */
static const IID SDL_IID_IAudioClient = { 0x1cb9ad4c, 0xdbfa, 0x4c32, { 0xb1, 0x78, 0xc2, 0xf5, 0x68, 0xa7, 0x03, 0xb2 } };

int WASAPI_PlatformInit(void)
{
    if (SDL_IMMDevice_Init() < 0) {
        return -1; /* This is set by SDL_IMMDevice_Init */
    }

    libavrt = LoadLibrary(TEXT("avrt.dll")); /* this library is available in Vista and later. No WinXP, so have to LoadLibrary to use it for now! */
    if (libavrt) {
        pAvSetMmThreadCharacteristicsW = (pfnAvSetMmThreadCharacteristicsW)GetProcAddress(libavrt, "AvSetMmThreadCharacteristicsW");
        pAvRevertMmThreadCharacteristics = (pfnAvRevertMmThreadCharacteristics)GetProcAddress(libavrt, "AvRevertMmThreadCharacteristics");
    }

    return 0;
}

void WASAPI_PlatformDeinit(void)
{
    if (libavrt) {
        FreeLibrary(libavrt);
        libavrt = NULL;
    }

    pAvSetMmThreadCharacteristicsW = NULL;
    pAvRevertMmThreadCharacteristics = NULL;

    SDL_IMMDevice_Quit();
}

void WASAPI_PlatformThreadInit(_THIS)
{
    /* this thread uses COM. */
    if (SUCCEEDED(WIN_CoInitialize())) { /* can't report errors, hope it worked! */
        this->hidden->coinitialized = SDL_TRUE;
    }

    /* Set this thread to very high "Pro Audio" priority. */
    if (pAvSetMmThreadCharacteristicsW) {
        DWORD idx = 0;
        this->hidden->task = pAvSetMmThreadCharacteristicsW(L"Pro Audio", &idx);
    }
}

void WASAPI_PlatformThreadDeinit(_THIS)
{
    /* Set this thread back to normal priority. */
    if (this->hidden->task && pAvRevertMmThreadCharacteristics) {
        pAvRevertMmThreadCharacteristics(this->hidden->task);
        this->hidden->task = NULL;
    }

    if (this->hidden->coinitialized) {
        WIN_CoUninitialize();
        this->hidden->coinitialized = SDL_FALSE;
    }
}

int WASAPI_ActivateDevice(_THIS, const SDL_bool isrecovery)
{
    IMMDevice *device = NULL;
    HRESULT ret;

    if (SDL_IMMDevice_Get(this->hidden->devid, &device, this->iscapture) < 0) {
        this->hidden->client = NULL;
        return -1; /* This is already set by SDL_IMMDevice_Get */
    }

    /* this is not async in standard win32, yay! */
    ret = IMMDevice_Activate(device, &SDL_IID_IAudioClient, CLSCTX_ALL, NULL, (void **)&this->hidden->client);
    IMMDevice_Release(device);

    if (FAILED(ret)) {
        SDL_assert(this->hidden->client == NULL);
        return WIN_SetErrorFromHRESULT("WASAPI can't activate audio endpoint", ret);
    }

    SDL_assert(this->hidden->client != NULL);
    if (WASAPI_PrepDevice(this, isrecovery) == -1) { /* not async, fire it right away. */
        return -1;
    }

    return 0; /* good to go. */
}

void WASAPI_EnumerateEndpoints(void)
{
    SDL_IMMDevice_EnumerateEndpoints(SDL_FALSE);
}

int WASAPI_GetDefaultAudioInfo(char **name, SDL_AudioSpec *spec, int iscapture)
{
    return SDL_IMMDevice_GetDefaultAudioInfo(name, spec, iscapture);
}

void WASAPI_PlatformDeleteActivationHandler(void *handler)
{
    /* not asynchronous. */
    SDL_assert(!"This function should have only been called on WinRT.");
}

#endif /* SDL_AUDIO_DRIVER_WASAPI && !defined(__WINRT__) */

/* vi: set ts=4 sw=4 expandtab: */
