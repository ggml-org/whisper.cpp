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

#ifndef _SDL_n3dsaudio_h_
#define _SDL_n3dsaudio_h_

#include <3ds.h>

/* Hidden "this" pointer for the audio functions */
#define _THIS SDL_AudioDevice *this

#define NUM_BUFFERS 3 /* -- Minimum 2! */

struct SDL_PrivateAudioData
{
    /* Speaker data */
    Uint8 *mixbuf;
    Uint32 mixlen;
    Uint32 format;
    Uint32 samplerate;
    Uint32 channels;
    Uint8 bytePerSample;
    Uint32 isSigned;
    Uint32 nextbuf;
    ndspWaveBuf waveBuf[NUM_BUFFERS];
    LightLock lock;
    CondVar cv;
    SDL_bool isCancelled;
};

#endif /* _SDL_n3dsaudio_h_ */
/* vi: set sts=4 ts=4 sw=4 expandtab: */
