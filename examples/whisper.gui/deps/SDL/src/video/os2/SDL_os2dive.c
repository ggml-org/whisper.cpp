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
#include "../SDL_sysvideo.h"
#define INCL_WIN
#define INCL_GPI
#include <os2.h>
#define  _MEERROR_H_
#include <mmioos2.h>
#include <os2me.h>
#define INCL_MM_OS2
#include <dive.h>
#include <fourcc.h>
#include "SDL_os2output.h"

typedef struct _VODATA {
  HDIVE    hDive;
  PVOID    pBuffer;
  ULONG    ulDIVEBufNum;
  FOURCC   fccColorEncoding;
  ULONG    ulWidth;
  ULONG    ulHeight;
  BOOL     fBlitterReady;
} VODATA;

static BOOL voQueryInfo(VIDEOOUTPUTINFO *pInfo);
static PVODATA voOpen(void);
static VOID voClose(PVODATA pVOData);
static BOOL voSetVisibleRegion(PVODATA pVOData, HWND hwnd,
                               SDL_DisplayMode *pSDLDisplayMode,
                               HRGN hrgnShape, BOOL fVisible);
static PVOID voVideoBufAlloc(PVODATA pVOData, ULONG ulWidth, ULONG ulHeight,
                             ULONG ulBPP, ULONG fccColorEncoding,
                             PULONG pulScanLineSize);
static VOID voVideoBufFree(PVODATA pVOData);
static BOOL voUpdate(PVODATA pVOData, HWND hwnd, SDL_Rect *pSDLRects,
                     ULONG cSDLRects);

OS2VIDEOOUTPUT voDive = {
    voQueryInfo,
    voOpen,
    voClose,
    voSetVisibleRegion,
    voVideoBufAlloc,
    voVideoBufFree,
    voUpdate
};


static BOOL voQueryInfo(VIDEOOUTPUTINFO *pInfo)
{
    DIVE_CAPS sDiveCaps;
    FOURCC fccFormats[100];

    /* Query information about display hardware from DIVE. */
    SDL_zeroa(fccFormats);
    SDL_zero(sDiveCaps);
    sDiveCaps.pFormatData    = fccFormats;
    sDiveCaps.ulFormatLength = 100;
    sDiveCaps.ulStructLen    = sizeof(DIVE_CAPS);

    if (DiveQueryCaps(&sDiveCaps, DIVE_BUFFER_SCREEN)) {
        debug_os2("DiveQueryCaps() failed.");
        return FALSE;
    }

    if (sDiveCaps.ulDepth < 8) {
        debug_os2("Not enough screen colors to run DIVE. "
                  "Must be at least 256 colors.");
        return FALSE;
    }

    pInfo->ulBPP             = sDiveCaps.ulDepth;
    pInfo->fccColorEncoding  = sDiveCaps.fccColorEncoding;
    pInfo->ulScanLineSize    = sDiveCaps.ulScanLineBytes;
    pInfo->ulHorizResolution = sDiveCaps.ulHorizontalResolution;
    pInfo->ulVertResolution  = sDiveCaps.ulVerticalResolution;

    return TRUE;
}

PVODATA voOpen(void)
{
    PVODATA pVOData = SDL_calloc(1, sizeof(VODATA));

    if (!pVOData) {
        SDL_OutOfMemory();
        return NULL;
    }

    if (DiveOpen(&pVOData->hDive, FALSE, NULL) != DIVE_SUCCESS) {
        SDL_free(pVOData);
        SDL_SetError("DIVE: A display engine instance open failed");
        return NULL;
    }

    return pVOData;
}

static VOID voClose(PVODATA pVOData)
{
    voVideoBufFree(pVOData);
    DiveClose(pVOData->hDive);
    SDL_free(pVOData);
}

static BOOL voSetVisibleRegion(PVODATA pVOData, HWND hwnd,
                               SDL_DisplayMode *pSDLDisplayMode,
                               HRGN hrgnShape, BOOL fVisible)
{
    HPS     hps;
    HRGN    hrgn;
    RGNRECT rgnCtl;
    PRECTL  prectl = NULL;
    ULONG   ulRC;

    if (!fVisible) {
        if (pVOData->fBlitterReady) {
            pVOData->fBlitterReady = FALSE;
            DiveSetupBlitter(pVOData->hDive, 0);
            debug_os2("DIVE blitter is tuned off");
        }
        return TRUE;
    }

    /* Query visible rectangles */
    hps = WinGetPS(hwnd);
    hrgn = GpiCreateRegion(hps, 0, NULL);
    if (hrgn == NULLHANDLE) {
        WinReleasePS(hps);
        SDL_SetError("GpiCreateRegion() failed");
    } else {
        WinQueryVisibleRegion(hwnd, hrgn);
        if (hrgnShape != NULLHANDLE)
            GpiCombineRegion(hps, hrgn, hrgn, hrgnShape, CRGN_AND);

        rgnCtl.ircStart     = 1;
        rgnCtl.crc          = 0;
        rgnCtl.ulDirection  = 1;
        GpiQueryRegionRects(hps, hrgn, NULL, &rgnCtl, NULL);
        if (rgnCtl.crcReturned != 0) {
            prectl = SDL_malloc(rgnCtl.crcReturned * sizeof(RECTL));
            if (prectl != NULL) {
                rgnCtl.ircStart     = 1;
                rgnCtl.crc          = rgnCtl.crcReturned;
                rgnCtl.ulDirection  = 1;
                GpiQueryRegionRects(hps, hrgn, NULL, &rgnCtl, prectl);
            } else {
                SDL_OutOfMemory();
            }
        }
        GpiDestroyRegion(hps, hrgn);
        WinReleasePS(hps);

        if (prectl != NULL) {
            /* Setup DIVE blitter. */
            SETUP_BLITTER   sSetupBlitter;
            SWP             swp;
            POINTL          pointl = { 0,0 };

            WinQueryWindowPos(hwnd, &swp);
            WinMapWindowPoints(hwnd, HWND_DESKTOP, &pointl, 1);

            sSetupBlitter.ulStructLen       = sizeof(SETUP_BLITTER);
            sSetupBlitter.fccSrcColorFormat = pVOData->fccColorEncoding;
            sSetupBlitter.fInvert           = FALSE;
            sSetupBlitter.ulSrcWidth        = pVOData->ulWidth;
            sSetupBlitter.ulSrcHeight       = pVOData->ulHeight;
            sSetupBlitter.ulSrcPosX         = 0;
            sSetupBlitter.ulSrcPosY         = 0;
            sSetupBlitter.ulDitherType      = 0;
            sSetupBlitter.fccDstColorFormat = FOURCC_SCRN;
            sSetupBlitter.ulDstWidth        = swp.cx;
            sSetupBlitter.ulDstHeight       = swp.cy;
            sSetupBlitter.lDstPosX          = 0;
            sSetupBlitter.lDstPosY          = 0;
            sSetupBlitter.lScreenPosX       = pointl.x;
            sSetupBlitter.lScreenPosY       = pointl.y;

            sSetupBlitter.ulNumDstRects     = rgnCtl.crcReturned;
            sSetupBlitter.pVisDstRects      = prectl;

            ulRC = DiveSetupBlitter(pVOData->hDive, &sSetupBlitter);
            SDL_free(prectl);

            if (ulRC == DIVE_SUCCESS) {
                pVOData->fBlitterReady = TRUE;
                WinInvalidateRect(hwnd, NULL, TRUE);
                debug_os2("DIVE blitter is ready now.");
                return TRUE;
            }

            SDL_SetError("DiveSetupBlitter(), rc = 0x%X", ulRC);
        } /* if (prectl != NULL) */
    } /* if (hrgn == NULLHANDLE) else */

    pVOData->fBlitterReady = FALSE;
    DiveSetupBlitter(pVOData->hDive, 0);
    return FALSE;
}

static PVOID voVideoBufAlloc(PVODATA pVOData, ULONG ulWidth, ULONG ulHeight,
                             ULONG ulBPP, FOURCC fccColorEncoding,
                             PULONG pulScanLineSize)
{
    ULONG   ulRC;
    ULONG   ulScanLineSize = ulWidth * (ulBPP >> 3);

    /* Destroy previous buffer. */
    voVideoBufFree(pVOData);

    if (ulWidth == 0 || ulHeight == 0 || ulBPP == 0)
        return NULL;

    /* Bytes per line. */
    ulScanLineSize  = (ulScanLineSize + 3) & ~3; /* 4-byte aligning */
    *pulScanLineSize = ulScanLineSize;

    ulRC = DosAllocMem(&pVOData->pBuffer,
                       (ulHeight * ulScanLineSize) + sizeof(ULONG),
                       PAG_COMMIT | PAG_EXECUTE | PAG_READ | PAG_WRITE);
    if (ulRC != NO_ERROR) {
        debug_os2("DosAllocMem(), rc = %u", ulRC);
        return NULL;
    }

    ulRC = DiveAllocImageBuffer(pVOData->hDive, &pVOData->ulDIVEBufNum,
                                fccColorEncoding, ulWidth, ulHeight,
                                ulScanLineSize, pVOData->pBuffer);
    if (ulRC != DIVE_SUCCESS) {
        debug_os2("DiveAllocImageBuffer(), rc = 0x%X", ulRC);
        DosFreeMem(pVOData->pBuffer);
        pVOData->pBuffer = NULL;
        pVOData->ulDIVEBufNum = 0;
        return NULL;
    }

    pVOData->fccColorEncoding = fccColorEncoding;
    pVOData->ulWidth = ulWidth;
    pVOData->ulHeight = ulHeight;

    debug_os2("buffer: 0x%P, DIVE buffer number: %u",
              pVOData->pBuffer, pVOData->ulDIVEBufNum);

    return pVOData->pBuffer;
}

static VOID voVideoBufFree(PVODATA pVOData)
{
    ULONG   ulRC;

    if (pVOData->ulDIVEBufNum != 0) {
        ulRC = DiveFreeImageBuffer(pVOData->hDive, pVOData->ulDIVEBufNum);
        if (ulRC != DIVE_SUCCESS) {
            debug_os2("DiveFreeImageBuffer(,%u), rc = %u", pVOData->ulDIVEBufNum, ulRC);
        } else {
            debug_os2("DIVE buffer %u destroyed", pVOData->ulDIVEBufNum);
        }
        pVOData->ulDIVEBufNum = 0;
    }

    if (pVOData->pBuffer) {
        ulRC = DosFreeMem(pVOData->pBuffer);
        if (ulRC != NO_ERROR) {
            debug_os2("DosFreeMem(), rc = %u", ulRC);
        }
        pVOData->pBuffer = NULL;
    }
}

static BOOL voUpdate(PVODATA pVOData, HWND hwnd, SDL_Rect *pSDLRects,
                     ULONG cSDLRects)
{
    ULONG   ulRC;

    if (!pVOData->fBlitterReady || (pVOData->ulDIVEBufNum == 0)) {
        debug_os2("DIVE blitter is not ready");
        return FALSE;
    }

    if (pSDLRects) {
        PBYTE   pbLineMask;

        pbLineMask = SDL_stack_alloc(BYTE, pVOData->ulHeight);
        if (!pbLineMask) {
            debug_os2("Not enough stack size");
            return FALSE;
        }
        SDL_memset(pbLineMask, 0, pVOData->ulHeight);

        for ( ; ((LONG)cSDLRects) > 0; cSDLRects--, pSDLRects++) {
            SDL_memset(&pbLineMask[pSDLRects->y], 1, pSDLRects->h);
        }

        ulRC = DiveBlitImageLines(pVOData->hDive, pVOData->ulDIVEBufNum,
                                  DIVE_BUFFER_SCREEN, pbLineMask);
        SDL_stack_free(pbLineMask);

        if (ulRC != DIVE_SUCCESS) {
            debug_os2("DiveBlitImageLines(), rc = 0x%X", ulRC);
        }
    } else {
        ulRC = DiveBlitImage(pVOData->hDive, pVOData->ulDIVEBufNum,
                             DIVE_BUFFER_SCREEN);
        if (ulRC != DIVE_SUCCESS) {
            debug_os2("DiveBlitImage(), rc = 0x%X", ulRC);
        }
    }

    return ulRC == DIVE_SUCCESS;
}

/* vi: set ts=4 sw=4 expandtab: */
