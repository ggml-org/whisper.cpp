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

#ifndef SDL_IMMDEVICE_H
#define SDL_IMMDEVICE_H

#include "SDL_atomic.h"
#include "SDL_audio.h"

#define COBJMACROS
#include <mmdeviceapi.h>
#include <mmreg.h>

int SDL_IMMDevice_Init(void);
void SDL_IMMDevice_Quit(void);
int SDL_IMMDevice_Get(LPCWSTR devid, IMMDevice **device, SDL_bool iscapture);
void SDL_IMMDevice_EnumerateEndpoints(SDL_bool useguid);
int SDL_IMMDevice_GetDefaultAudioInfo(char **name, SDL_AudioSpec *spec, int iscapture);

SDL_AudioFormat WaveFormatToSDLFormat(WAVEFORMATEX *waveformat);

/* these increment as default devices change. Opened default devices pick up changes in their threads. */
extern SDL_atomic_t SDL_IMMDevice_DefaultPlaybackGeneration;
extern SDL_atomic_t SDL_IMMDevice_DefaultCaptureGeneration;

#endif /* SDL_IMMDEVICE_H */
