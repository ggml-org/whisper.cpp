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

#ifndef SDL_os2mm_h_
#define SDL_os2mm_h_

#include "../SDL_sysaudio.h"

#define  INCL_OS2MM
#define  INCL_PM
#define  INCL_DOS
#define  INCL_DOSERRORS
#include <os2.h>
#include <os2me.h>

/* Hidden "this" pointer for the audio functions */
#define _THIS   SDL_AudioDevice *_this

#define NUM_BUFFERS 3

typedef struct SDL_PrivateAudioData
{
    USHORT              usDeviceId;
    BYTE                _pad[2];
    MCI_MIXSETUP_PARMS  stMCIMixSetup;
    HEV                 hevBuf;
    PMCI_MIX_BUFFER     pFillBuffer;
    PMCI_MIX_BUFFER     pDrainBuffer;
    ULONG               ulState;
    ULONG               cMixBuffers;
    MCI_MIX_BUFFER      aMixBuffers[NUM_BUFFERS];
} SDL_PrivateAudioData;

#endif /* SDL_os2mm_h_ */

/* vi: set ts=4 sw=4 expandtab: */
