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

#ifndef SDL_wasapi_h_
#define SDL_wasapi_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "../SDL_sysaudio.h"

/* Hidden "this" pointer for the audio functions */
#ifdef __cplusplus
#define _THIS SDL_AudioDevice *_this
#else
#define _THIS SDL_AudioDevice *this
#endif

struct SDL_PrivateAudioData
{
    SDL_atomic_t refcount;
    WCHAR *devid;
    WAVEFORMATEX *waveformat;
    IAudioClient *client;
    IAudioRenderClient *render;
    IAudioCaptureClient *capture;
    SDL_AudioStream *capturestream;
    HANDLE event;
    HANDLE task;
    SDL_threadID open_threadid;
    SDL_bool coinitialized;
    int framesize;
    int default_device_generation;
    SDL_bool device_lost;
    void *activation_handler;
    SDL_atomic_t just_activated;
};

/* win32 and winrt implementations call into these. */
int WASAPI_PrepDevice(_THIS, const SDL_bool updatestream);
void WASAPI_RefDevice(_THIS);
void WASAPI_UnrefDevice(_THIS);

/* These are functions that are implemented differently for Windows vs WinRT. */
int WASAPI_PlatformInit(void);
void WASAPI_PlatformDeinit(void);
void WASAPI_EnumerateEndpoints(void);
int WASAPI_GetDefaultAudioInfo(char **name, SDL_AudioSpec *spec, int iscapture);
int WASAPI_ActivateDevice(_THIS, const SDL_bool isrecovery);
void WASAPI_PlatformThreadInit(_THIS);
void WASAPI_PlatformThreadDeinit(_THIS);
void WASAPI_PlatformDeleteActivationHandler(void *handler);

#ifdef __cplusplus
}
#endif

#endif /* SDL_wasapi_h_ */

/* vi: set ts=4 sw=4 expandtab: */
