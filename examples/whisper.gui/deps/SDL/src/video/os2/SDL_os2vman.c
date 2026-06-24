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

#define INCL_DOSERRORS
#define INCL_DOSPROCESS
#define INCL_DOSMODULEMGR
#define INCL_WIN
#define INCL_GPI
#define INCL_GPIBITMAPS /* GPI bit map functions */
#include <os2.h>
#include "SDL_os2output.h"
#include "SDL_os2video.h"

#include "SDL_gradd.h"

typedef struct _VODATA {
  PVOID    pBuffer;
  HRGN     hrgnVisible;
  ULONG    ulBPP;
  ULONG    ulScanLineSize;
  ULONG    ulWidth;
  ULONG    ulHeight;
  ULONG    ulScreenHeight;
  ULONG    ulScreenBytesPerLine;
  RECTL    rectlWin;

  PRECTL   pRectl;
  ULONG    cRectl;
  PBLTRECT pBltRect;
  ULONG    cBltRect;
} VODATA;

static BOOL voQueryInfo(VIDEOOUTPUTINFO *pInfo);
static PVODATA voOpen();
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

OS2VIDEOOUTPUT voVMan = {
    voQueryInfo,
    voOpen,
    voClose,
    voSetVisibleRegion,
    voVideoBufAlloc,
    voVideoBufFree,
    voUpdate
};


static HMODULE  hmodVMan = NULLHANDLE;
static FNVMIENTRY *pfnVMIEntry = NULL;
static ULONG        ulVRAMAddress = 0;

static VOID APIENTRY ExitVMan(VOID)
{
    if (ulVRAMAddress != 0 && hmodVMan != NULLHANDLE) {
        pfnVMIEntry(0, VMI_CMD_TERMPROC, NULL, NULL);
        DosFreeModule(hmodVMan);
    }

    DosExitList(EXLST_EXIT, (PFNEXITLIST)NULL);
}

static BOOL _vmanInit(void)
{
    ULONG       ulRC;
    CHAR        acBuf[256];
    INITPROCOUT stInitProcOut;

    if (hmodVMan != NULLHANDLE) /* already initialized */
        return TRUE;

    /* Load vman.dll */
    ulRC = DosLoadModule(acBuf, sizeof(acBuf), "VMAN", &hmodVMan);
    if (ulRC != NO_ERROR) {
        debug_os2("Could not load VMAN.DLL, rc = %u : %s", ulRC, acBuf);
        hmodVMan = NULLHANDLE;
        return FALSE;
    }

    /* Get VMIEntry */
    ulRC = DosQueryProcAddr(hmodVMan, 0L, "VMIEntry", (PFN *)&pfnVMIEntry);
    if (ulRC != NO_ERROR) {
        debug_os2("Could not query address of VMIEntry from VMAN.DLL (Err: %lu)", ulRC);
        DosFreeModule(hmodVMan);
        hmodVMan = NULLHANDLE;
        return FALSE;
    }

    /* VMAN initialization */
    stInitProcOut.ulLength = sizeof(stInitProcOut);
    ulRC = pfnVMIEntry(0, VMI_CMD_INITPROC, NULL, &stInitProcOut);
    if (ulRC != RC_SUCCESS) {
        debug_os2("Could not initialize VMAN for this process");
        pfnVMIEntry = NULL;
        DosFreeModule(hmodVMan);
        hmodVMan = NULLHANDLE;
        return FALSE;
    }

    /* Store video memory virtual address */
    ulVRAMAddress = stInitProcOut.ulVRAMVirt;
    /* We use exit list for VMI_CMD_TERMPROC */
    if (DosExitList(EXLST_ADD | 0x00001000, (PFNEXITLIST)ExitVMan) != NO_ERROR) {
        debug_os2("DosExitList() failed");
    }

    return TRUE;
}

static PRECTL _getRectlArray(PVODATA pVOData, ULONG cRects)
{
    PRECTL  pRectl;

    if (pVOData->cRectl >= cRects)
        return pVOData->pRectl;

    pRectl = SDL_realloc(pVOData->pRectl, cRects * sizeof(RECTL));
    if (!pRectl)
        return NULL;

    pVOData->pRectl = pRectl;
    pVOData->cRectl = cRects;
    return pRectl;
}

static PBLTRECT _getBltRectArray(PVODATA pVOData, ULONG cRects)
{
    PBLTRECT    pBltRect;

    if (pVOData->cBltRect >= cRects)
        return pVOData->pBltRect;

    pBltRect = SDL_realloc(pVOData->pBltRect, cRects * sizeof(BLTRECT));
    if (!pBltRect)
        return NULL;

    pVOData->pBltRect = pBltRect;
    pVOData->cBltRect = cRects;
    return pBltRect;
}


static BOOL voQueryInfo(VIDEOOUTPUTINFO *pInfo)
{
    ULONG       ulRC;
    GDDMODEINFO sCurModeInfo;

    if (!_vmanInit())
        return FALSE;

    /* Query current (desktop) mode */
    ulRC = pfnVMIEntry(0, VMI_CMD_QUERYCURRENTMODE, NULL, &sCurModeInfo);
    if (ulRC != RC_SUCCESS) {
        debug_os2("Could not query desktop video mode.");
        return FALSE;
    }

    pInfo->ulBPP             = sCurModeInfo.ulBpp;
    pInfo->ulHorizResolution = sCurModeInfo.ulHorizResolution;
    pInfo->ulVertResolution  = sCurModeInfo.ulVertResolution;
    pInfo->ulScanLineSize    = sCurModeInfo.ulScanLineSize;
    pInfo->fccColorEncoding  = sCurModeInfo.fccColorEncoding;

    return TRUE;
}

static PVODATA voOpen(void)
{
    PVODATA pVOData;

    if (!_vmanInit())
        return NULL;

    pVOData = SDL_calloc(1, sizeof(VODATA));
    if (!pVOData) {
        SDL_OutOfMemory();
        return NULL;
    }

    return pVOData;
}

static VOID voClose(PVODATA pVOData)
{
    if (pVOData->pRectl)
        SDL_free(pVOData->pRectl);

    if (pVOData->pBltRect)
        SDL_free(pVOData->pBltRect);

    voVideoBufFree(pVOData);
}

static BOOL voSetVisibleRegion(PVODATA pVOData, HWND hwnd,
                               SDL_DisplayMode *pSDLDisplayMode,
                               HRGN hrgnShape, BOOL fVisible)
{
    HPS   hps;
    BOOL  fSuccess = FALSE;

    hps = WinGetPS(hwnd);

    if (pVOData->hrgnVisible != NULLHANDLE) {
        GpiDestroyRegion(hps, pVOData->hrgnVisible);
        pVOData->hrgnVisible = NULLHANDLE;
    }

    if (fVisible) {
        /* Query visible rectangles */
        pVOData->hrgnVisible = GpiCreateRegion(hps, 0, NULL);
        if (pVOData->hrgnVisible == NULLHANDLE) {
            SDL_SetError("GpiCreateRegion() failed");
        } else {
            if (WinQueryVisibleRegion(hwnd, pVOData->hrgnVisible) == RGN_ERROR) {
                GpiDestroyRegion(hps, pVOData->hrgnVisible);
                pVOData->hrgnVisible = NULLHANDLE;
            } else {
                if (hrgnShape != NULLHANDLE)
                    GpiCombineRegion(hps, pVOData->hrgnVisible, pVOData->hrgnVisible,
                                     hrgnShape, CRGN_AND);
                fSuccess = TRUE;
            }
        }

        WinQueryWindowRect(hwnd, &pVOData->rectlWin);
        WinMapWindowPoints(hwnd, HWND_DESKTOP, (PPOINTL)&pVOData->rectlWin, 2);

        if (pSDLDisplayMode) {
            pVOData->ulScreenHeight = pSDLDisplayMode->h;
            pVOData->ulScreenBytesPerLine =
                     ((MODEDATA *)pSDLDisplayMode->driverdata)->ulScanLineBytes;
        }
    }

    WinReleasePS(hps);

    return fSuccess;
}

static PVOID voVideoBufAlloc(PVODATA pVOData, ULONG ulWidth, ULONG ulHeight,
                             ULONG ulBPP, ULONG fccColorEncoding,
                             PULONG pulScanLineSize)
{
    ULONG ulRC;
    ULONG ulScanLineSize = ulWidth * (ulBPP >> 3);

    /* Destroy previous buffer */
    voVideoBufFree(pVOData);

    if (ulWidth == 0 || ulHeight == 0 || ulBPP == 0)
        return NULL;

    /* Bytes per line */
    ulScanLineSize  = (ulScanLineSize + 3) & ~3; /* 4-byte aligning */
    *pulScanLineSize = ulScanLineSize;

    ulRC = DosAllocMem(&pVOData->pBuffer,
                       (ulHeight * ulScanLineSize) + sizeof(ULONG),
                       PAG_COMMIT | PAG_EXECUTE | PAG_READ | PAG_WRITE);
    if (ulRC != NO_ERROR) {
        debug_os2("DosAllocMem(), rc = %u", ulRC);
        return NULL;
    }

    pVOData->ulBPP          = ulBPP;
    pVOData->ulScanLineSize = ulScanLineSize;
    pVOData->ulWidth        = ulWidth;
    pVOData->ulHeight       = ulHeight;

    return pVOData->pBuffer;
}

static VOID voVideoBufFree(PVODATA pVOData)
{
    ULONG ulRC;

    if (!pVOData->pBuffer)
        return;

    ulRC = DosFreeMem(pVOData->pBuffer);
    if (ulRC != NO_ERROR) {
        debug_os2("DosFreeMem(), rc = %u", ulRC);
    } else {
        pVOData->pBuffer = NULL;
    }
}

static BOOL voUpdate(PVODATA pVOData, HWND hwnd, SDL_Rect *pSDLRects,
                     ULONG cSDLRects)
{
    PRECTL      prectlDst, prectlScan;
    HPS         hps;
    HRGN        hrgnUpdate;
    RGNRECT     rgnCtl;
    SDL_Rect    stSDLRectDef;
    BMAPINFO    bmiSrc;
    BMAPINFO    bmiDst;
    PPOINTL     pptlSrcOrg;
    PBLTRECT    pbrDst;
    HWREQIN     sHWReqIn;
    BITBLTINFO  sBitbltInfo;
    ULONG       ulIdx;

    if (!pVOData->pBuffer)
        return FALSE;

    if (pVOData->hrgnVisible == NULLHANDLE)
        return TRUE;

    bmiSrc.ulLength = sizeof(BMAPINFO);
    bmiSrc.ulType = BMAP_MEMORY;
    bmiSrc.ulWidth = pVOData->ulWidth;
    bmiSrc.ulHeight = pVOData->ulHeight;
    bmiSrc.ulBpp = pVOData->ulBPP;
    bmiSrc.ulBytesPerLine = pVOData->ulScanLineSize;
    bmiSrc.pBits = (PBYTE)pVOData->pBuffer;

    bmiDst.ulLength = sizeof(BMAPINFO);
    bmiDst.ulType = BMAP_VRAM;
    bmiDst.pBits = (PBYTE)ulVRAMAddress;
    bmiDst.ulWidth = bmiSrc.ulWidth;
    bmiDst.ulHeight = bmiSrc.ulHeight;
    bmiDst.ulBpp = bmiSrc.ulBpp;
    bmiDst.ulBytesPerLine = pVOData->ulScreenBytesPerLine;

    /* List of update rectangles. This is the intersection of requested
     * rectangles and visible rectangles.  */
    if (cSDLRects == 0) {
        /* Full update requested */
        stSDLRectDef.x = 0;
        stSDLRectDef.y = 0;
        stSDLRectDef.w = bmiSrc.ulWidth;
        stSDLRectDef.h = bmiSrc.ulHeight;
        pSDLRects = &stSDLRectDef;
        cSDLRects = 1;
    }

    /* Make list of destination rectangles (prectlDst) list from the source
     * list (prectl).  */
    prectlDst = _getRectlArray(pVOData, cSDLRects);
    if (!prectlDst) {
        debug_os2("Not enough memory");
        return FALSE;
    }
    prectlScan = prectlDst;
    for (ulIdx = 0; ulIdx < cSDLRects; ulIdx++, pSDLRects++, prectlScan++) {
        prectlScan->xLeft   = pSDLRects->x;
        prectlScan->yTop    = pVOData->ulHeight - pSDLRects->y;
        prectlScan->xRight  = prectlScan->xLeft + pSDLRects->w;
        prectlScan->yBottom = prectlScan->yTop - pSDLRects->h;
    }

    hps = WinGetPS(hwnd);
    if (hps == NULLHANDLE)
        return FALSE;

    /* Make destination region to update */
    hrgnUpdate = GpiCreateRegion(hps, cSDLRects, prectlDst);
    /* "AND" on visible and destination regions, result is region to update */
    GpiCombineRegion(hps, hrgnUpdate, hrgnUpdate, pVOData->hrgnVisible, CRGN_AND);

    /* Get rectangles of the region to update */
    rgnCtl.ircStart     = 1;
    rgnCtl.crc          = 0;
    rgnCtl.ulDirection  = 1;
    rgnCtl.crcReturned  = 0;
    GpiQueryRegionRects(hps, hrgnUpdate, NULL, &rgnCtl, NULL);
    if (rgnCtl.crcReturned == 0) {
        GpiDestroyRegion(hps, hrgnUpdate);
        WinReleasePS(hps);
        return TRUE;
    }
    /* We don't need prectlDst, use it again to store update regions */
    prectlDst = _getRectlArray(pVOData, rgnCtl.crcReturned);
    if (!prectlDst) {
        debug_os2("Not enough memory");
        GpiDestroyRegion(hps, hrgnUpdate);
        WinReleasePS(hps);
        return FALSE;
    }
    rgnCtl.ircStart     = 1;
    rgnCtl.crc          = rgnCtl.crcReturned;
    rgnCtl.ulDirection  = 1;
    GpiQueryRegionRects(hps, hrgnUpdate, NULL, &rgnCtl, prectlDst);
    GpiDestroyRegion(hps, hrgnUpdate);
    WinReleasePS(hps);
    cSDLRects = rgnCtl.crcReturned;

    /* Now cRect/prectlDst is a list of regions in window (update && visible) */

    /* Make lists for blitting from update regions */
    pbrDst = _getBltRectArray(pVOData, cSDLRects);
    if (!pbrDst) {
        debug_os2("Not enough memory");
        return FALSE;
    }

    prectlScan = prectlDst;
    pptlSrcOrg = (PPOINTL)prectlDst; /* Yes, this memory block will be used again */
    for (ulIdx = 0; ulIdx < cSDLRects; ulIdx++, prectlScan++, pptlSrcOrg++) {
        pbrDst[ulIdx].ulXOrg = pVOData->rectlWin.xLeft + prectlScan->xLeft;
        pbrDst[ulIdx].ulYOrg = pVOData->ulScreenHeight -
                              (pVOData->rectlWin.yBottom + prectlScan->yTop);
        pbrDst[ulIdx].ulXExt = prectlScan->xRight - prectlScan->xLeft;
        pbrDst[ulIdx].ulYExt = prectlScan->yTop - prectlScan->yBottom;
        pptlSrcOrg->x = prectlScan->xLeft;
        pptlSrcOrg->y = bmiSrc.ulHeight - prectlScan->yTop;
    }
    pptlSrcOrg = (PPOINTL)prectlDst;

    /* Request HW */
    sHWReqIn.ulLength = sizeof(HWREQIN);
    sHWReqIn.ulFlags = REQUEST_HW;
    sHWReqIn.cScrChangeRects = 1;
    sHWReqIn.arectlScreen = &pVOData->rectlWin;
    if (pfnVMIEntry(0, VMI_CMD_REQUESTHW, &sHWReqIn, NULL) != RC_SUCCESS) {
        debug_os2("pfnVMIEntry(,VMI_CMD_REQUESTHW,,) failed");
        sHWReqIn.cScrChangeRects = 0; /* for fail signal only */
    } else {
        RECTL rclSrcBounds;

        rclSrcBounds.xLeft = 0;
        rclSrcBounds.yBottom = 0;
        rclSrcBounds.xRight = bmiSrc.ulWidth;
        rclSrcBounds.yTop = bmiSrc.ulHeight;

        SDL_zero(sBitbltInfo);
        sBitbltInfo.ulLength = sizeof(BITBLTINFO);
        sBitbltInfo.ulBltFlags = BF_DEFAULT_STATE | BF_ROP_INCL_SRC | BF_PAT_HOLLOW;
        sBitbltInfo.cBlits = cSDLRects;
        sBitbltInfo.ulROP = ROP_SRCCOPY;
        sBitbltInfo.pSrcBmapInfo = &bmiSrc;
        sBitbltInfo.pDstBmapInfo = &bmiDst;
        sBitbltInfo.prclSrcBounds = &rclSrcBounds;
        sBitbltInfo.prclDstBounds = &pVOData->rectlWin;
        sBitbltInfo.aptlSrcOrg = pptlSrcOrg;
        sBitbltInfo.abrDst = pbrDst;

        /* Screen update */
        if (pfnVMIEntry(0, VMI_CMD_BITBLT, &sBitbltInfo, NULL) != RC_SUCCESS) {
            debug_os2("pfnVMIEntry(,VMI_CMD_BITBLT,,) failed");
            sHWReqIn.cScrChangeRects = 0; /* for fail signal only */
        }

        /* Release HW */
        sHWReqIn.ulFlags = 0;
        if (pfnVMIEntry(0, VMI_CMD_REQUESTHW, &sHWReqIn, NULL) != RC_SUCCESS) {
          debug_os2("pfnVMIEntry(,VMI_CMD_REQUESTHW,,) failed");
        }
    }

    return sHWReqIn.cScrChangeRects != 0;
}

/* vi: set ts=4 sw=4 expandtab: */
