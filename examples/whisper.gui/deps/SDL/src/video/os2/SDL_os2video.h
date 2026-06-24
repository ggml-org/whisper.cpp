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

#ifndef SDL_os2video_h_
#define SDL_os2video_h_

#include "../SDL_sysvideo.h"
#include "../../core/os2/SDL_os2.h"

#define INCL_DOS
#define INCL_DOSERRORS
#define INCL_DOSPROCESS
#define INCL_WIN
#define INCL_GPI
#define INCL_OS2MM
#define INCL_DOSMEMMGR
#include <os2.h>

#include "SDL_os2mouse.h"
#include "SDL_os2output.h"

typedef struct SDL_VideoData {
    HAB             hab;
    HMQ             hmq;
    OS2VIDEOOUTPUT *pOutput; /* Video output routines */
} SDL_VideoData;

typedef struct _WINDATA {
    SDL_Window     *window;
    OS2VIDEOOUTPUT *pOutput; /* Video output routines */
    HWND            hwndFrame;
    HWND            hwnd;
    PFNWP           fnUserWndProc;
    PFNWP           fnWndFrameProc;

    PVODATA         pVOData; /* Video output data */

    HRGN            hrgnShape;
    HPOINTER        hptrIcon;
    RECTL           rectlBeforeFS;

    LONG            lSkipWMSize;
    LONG            lSkipWMMove;
    LONG            lSkipWMMouseMove;
    LONG            lSkipWMVRNEnabled;
    LONG            lSkipWMAdjustFramePos;
} WINDATA;

typedef struct _DISPLAYDATA {
    ULONG           ulDPIHor;
    ULONG           ulDPIVer;
    ULONG           ulDPIDiag;
} DISPLAYDATA;

typedef struct _MODEDATA {
    ULONG           ulDepth;
    ULONG           fccColorEncoding;
    ULONG           ulScanLineBytes;
} MODEDATA;

#endif /* SDL_os2video_h_ */

/* vi: set ts=4 sw=4 expandtab: */
