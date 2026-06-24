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
#ifndef SDL_os2output_
#define SDL_os2output_

#include "../../core/os2/SDL_os2.h"

typedef struct _VODATA *PVODATA;

typedef struct _VIDEOOUTPUTINFO {
    ULONG     ulBPP;
    ULONG     fccColorEncoding;
    ULONG     ulScanLineSize;
    ULONG     ulHorizResolution;
    ULONG     ulVertResolution;
} VIDEOOUTPUTINFO;

typedef struct _OS2VIDEOOUTPUT {
    BOOL (*QueryInfo)(VIDEOOUTPUTINFO *pInfo);
    PVODATA (*Open)();
    VOID (*Close)(PVODATA pVOData);

    BOOL (*SetVisibleRegion)(PVODATA pVOData, HWND hwnd,
                             SDL_DisplayMode *pSDLDisplayMode, HRGN hrgnShape,
                             BOOL fVisible);

    PVOID (*VideoBufAlloc)(PVODATA pVOData, ULONG ulWidth, ULONG ulHeight,
                           ULONG ulBPP, ULONG fccColorEncoding,
                           PULONG pulScanLineSize);

    VOID (*VideoBufFree)(PVODATA pVOData);
    BOOL (*Update)(PVODATA pVOData, HWND hwnd, SDL_Rect *pSDLRects,
                   ULONG cSDLRects);
} OS2VIDEOOUTPUT;

extern OS2VIDEOOUTPUT voDive;
extern OS2VIDEOOUTPUT voVMan;

#endif /* SDL_os2output_ */

/* vi: set ts=4 sw=4 expandtab: */
