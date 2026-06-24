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

#ifndef _SDL_openslesaudio_h
#define _SDL_openslesaudio_h

#include "../SDL_sysaudio.h"

/* Hidden "this" pointer for the audio functions */
#define _THIS SDL_AudioDevice *this

#define NUM_BUFFERS 2 /* -- Don't lower this! */

struct SDL_PrivateAudioData
{
    Uint8 *mixbuff;
    int next_buffer;
    Uint8 *pmixbuff[NUM_BUFFERS];
    SDL_sem *playsem;
};

void openslES_ResumeDevices(void);
void openslES_PauseDevices(void);

#endif /* _SDL_openslesaudio_h */

/* vi: set ts=4 sw=4 expandtab: */
