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

#ifdef SDL_VIDEO_DRIVER_OS2

#include "SDL_video.h"
#include "SDL_mouse.h"
#include "../SDL_pixels_c.h"
#include "../SDL_shape_internals.h"
#include "../../events/SDL_events_c.h"
#include "SDL_os2video.h"
#include "SDL_syswm.h"
#include "SDL_os2util.h"
#include "SDL_os2messagebox.h"

#define __MEERROR_H__
#define  _MEERROR_H_
#include <mmioos2.h>
#include <fourcc.h>
#ifndef FOURCC_R666
#define FOURCC_R666 mmioFOURCC('R','6','6','6')
#endif

#define WIN_CLIENT_CLASS        "SDL2"
#define OS2DRIVER_NAME_DIVE     "DIVE"
#define OS2DRIVER_NAME_VMAN     "VMAN"


static const SDL_Scancode aSDLScancode[] = {
         /*   0                       1                           2                           3                           4                        5                                                       6                           7 */
         /*   8                       9                           A                           B                           C                        D                                                       E                           F */
         SDL_SCANCODE_UNKNOWN,        SDL_SCANCODE_ESCAPE,        SDL_SCANCODE_1,             SDL_SCANCODE_2,             SDL_SCANCODE_3,          SDL_SCANCODE_4,             SDL_SCANCODE_5,             SDL_SCANCODE_6,          /* 0 */
         SDL_SCANCODE_7,              SDL_SCANCODE_8,             SDL_SCANCODE_9,             SDL_SCANCODE_0,             SDL_SCANCODE_MINUS,      SDL_SCANCODE_EQUALS,        SDL_SCANCODE_BACKSPACE,     SDL_SCANCODE_TAB,        /* 0 */

         SDL_SCANCODE_Q,              SDL_SCANCODE_W,             SDL_SCANCODE_E,             SDL_SCANCODE_R,             SDL_SCANCODE_T,          SDL_SCANCODE_Y,             SDL_SCANCODE_U,             SDL_SCANCODE_I,          /* 1 */
         SDL_SCANCODE_O,              SDL_SCANCODE_P,             SDL_SCANCODE_LEFTBRACKET,   SDL_SCANCODE_RIGHTBRACKET,  SDL_SCANCODE_RETURN,     SDL_SCANCODE_LCTRL,         SDL_SCANCODE_A,             SDL_SCANCODE_S,          /* 1 */

         SDL_SCANCODE_D,              SDL_SCANCODE_F,             SDL_SCANCODE_G,             SDL_SCANCODE_H,             SDL_SCANCODE_J,          SDL_SCANCODE_K,             SDL_SCANCODE_L,             SDL_SCANCODE_SEMICOLON,  /* 2 */
         SDL_SCANCODE_APOSTROPHE,     SDL_SCANCODE_GRAVE,         SDL_SCANCODE_LSHIFT,        SDL_SCANCODE_BACKSLASH,     SDL_SCANCODE_Z,          SDL_SCANCODE_X,             SDL_SCANCODE_C,             SDL_SCANCODE_V,          /* 2 */

         SDL_SCANCODE_B,              SDL_SCANCODE_N,             SDL_SCANCODE_M,             SDL_SCANCODE_COMMA,         SDL_SCANCODE_PERIOD,     SDL_SCANCODE_SLASH,         SDL_SCANCODE_RSHIFT,  /*55*/SDL_SCANCODE_KP_MULTIPLY,/* 3 */
         SDL_SCANCODE_LALT,           SDL_SCANCODE_SPACE,         SDL_SCANCODE_CAPSLOCK,      SDL_SCANCODE_F1,            SDL_SCANCODE_F2,         SDL_SCANCODE_F3,            SDL_SCANCODE_F4,            SDL_SCANCODE_F5,         /* 3 */

         SDL_SCANCODE_F6,             SDL_SCANCODE_F7,            SDL_SCANCODE_F8,            SDL_SCANCODE_F9,            SDL_SCANCODE_F10,        SDL_SCANCODE_NUMLOCKCLEAR,  SDL_SCANCODE_SCROLLLOCK,    SDL_SCANCODE_KP_7,       /* 4 */
 /*72*/  SDL_SCANCODE_KP_8,     /*73*/SDL_SCANCODE_KP_9,          SDL_SCANCODE_KP_MINUS,/*75*/SDL_SCANCODE_KP_4,    /*76*/SDL_SCANCODE_KP_5, /*77*/SDL_SCANCODE_KP_6,    /*78*/SDL_SCANCODE_KP_PLUS, /*79*/SDL_SCANCODE_KP_1,       /* 4 */

 /*80*/  SDL_SCANCODE_KP_2,     /*81*/SDL_SCANCODE_KP_3,          SDL_SCANCODE_KP_0,    /*83*/SDL_SCANCODE_KP_PERIOD,     SDL_SCANCODE_UNKNOWN,    SDL_SCANCODE_UNKNOWN,       SDL_SCANCODE_NONUSBACKSLASH,SDL_SCANCODE_F11,        /* 5 */
 /*88*/  SDL_SCANCODE_F12,      /*89*/SDL_SCANCODE_PAUSE,   /*90*/SDL_SCANCODE_KP_ENTER,/*91*/SDL_SCANCODE_RCTRL,   /*92*/SDL_SCANCODE_KP_DIVIDE,  SDL_SCANCODE_APPLICATION,   SDL_SCANCODE_RALT,    /*95*/SDL_SCANCODE_UNKNOWN,    /* 5 */

 /*96*/  SDL_SCANCODE_HOME,     /*97*/SDL_SCANCODE_UP,      /*98*/SDL_SCANCODE_PAGEUP,        SDL_SCANCODE_LEFT,   /*100*/SDL_SCANCODE_RIGHT,      SDL_SCANCODE_END,    /*102*/SDL_SCANCODE_DOWN,   /*103*/SDL_SCANCODE_PAGEDOWN,   /* 6 */
/*104*/  SDL_SCANCODE_F17,     /*105*/SDL_SCANCODE_DELETE,        SDL_SCANCODE_F19,           SDL_SCANCODE_UNKNOWN,       SDL_SCANCODE_UNKNOWN,    SDL_SCANCODE_UNKNOWN,/*110*/SDL_SCANCODE_UNKNOWN,/*111*/SDL_SCANCODE_UNKNOWN,    /* 6 */

/*112*/  SDL_SCANCODE_INTERNATIONAL2, SDL_SCANCODE_UNKNOWN,       SDL_SCANCODE_UNKNOWN,       SDL_SCANCODE_INTERNATIONAL1,SDL_SCANCODE_UNKNOWN,    SDL_SCANCODE_UNKNOWN,       SDL_SCANCODE_UNKNOWN,       SDL_SCANCODE_UNKNOWN,    /* 7 */
/*120*/  SDL_SCANCODE_UNKNOWN,        SDL_SCANCODE_INTERNATIONAL4,SDL_SCANCODE_UNKNOWN,       SDL_SCANCODE_INTERNATIONAL5,SDL_SCANCODE_APPLICATION,SDL_SCANCODE_INTERNATIONAL3,SDL_SCANCODE_LGUI,          SDL_SCANCODE_RGUI        /* 7 */
};

/*  Utilites.
 *  ---------
 */
static BOOL _getSDLPixelFormatData(SDL_PixelFormat *pSDLPixelFormat,
                                   ULONG ulBPP, ULONG fccColorEncoding)
{
    ULONG   ulRshift, ulGshift, ulBshift;
    ULONG   ulRmask, ulGmask, ulBmask;
    ULONG   ulRloss, ulGloss, ulBloss;

    pSDLPixelFormat->BitsPerPixel = ulBPP;
    pSDLPixelFormat->BytesPerPixel = (pSDLPixelFormat->BitsPerPixel + 7) / 8;

    switch (fccColorEncoding) {
    case FOURCC_LUT8:
        ulRshift = 0; ulGshift = 0; ulBshift = 0;
        ulRmask = 0; ulGmask = 0; ulBmask = 0;
        ulRloss = 8; ulGloss = 8; ulBloss = 8;
        break;

    case FOURCC_R555:
        ulRshift = 10; ulGshift = 5; ulBshift = 0;
        ulRmask = 0x7C00; ulGmask = 0x03E0; ulBmask = 0x001F;
        ulRloss = 3; ulGloss = 3; ulBloss = 3;
        break;

    case FOURCC_R565:
        ulRshift = 11; ulGshift = 5; ulBshift = 0;
        ulRmask = 0xF800; ulGmask = 0x07E0; ulBmask = 0x001F;
        ulRloss = 3; ulGloss = 2; ulBloss = 3;
        break;

    case FOURCC_R664:
        ulRshift = 10; ulGshift = 4; ulBshift = 0;
        ulRmask = 0xFC00; ulGmask = 0x03F0; ulBmask = 0x000F;
        ulRloss = 2; ulGloss = 4; ulBloss = 3;
        break;

    case FOURCC_R666:
        ulRshift = 12; ulGshift = 6; ulBshift = 0;
        ulRmask = 0x03F000; ulGmask = 0x000FC0; ulBmask = 0x00003F;
        ulRloss = 2; ulGloss = 2; ulBloss = 2;
        break;

    case FOURCC_RGB3:
    case FOURCC_RGB4:
        ulRshift = 0; ulGshift = 8; ulBshift = 16;
        ulRmask = 0x0000FF; ulGmask = 0x00FF00; ulBmask = 0xFF0000;
        ulRloss = 0x00; ulGloss = 0x00; ulBloss = 0x00;
        break;

    case FOURCC_BGR3:
    case FOURCC_BGR4:
        ulRshift = 16; ulGshift = 8; ulBshift = 0;
        ulRmask = 0xFF0000; ulGmask = 0x00FF00; ulBmask = 0x0000FF;
        ulRloss = 0; ulGloss = 0; ulBloss = 0;
        break;

    default:
/*      printf("Unknown color encoding: %.4s\n", fccColorEncoding);*/
        SDL_memset(pSDLPixelFormat, 0, sizeof(SDL_PixelFormat));
        return FALSE;
    }

    pSDLPixelFormat->Rshift = ulRshift;
    pSDLPixelFormat->Gshift = ulGshift;
    pSDLPixelFormat->Bshift = ulBshift;
    pSDLPixelFormat->Rmask  = ulRmask;
    pSDLPixelFormat->Gmask  = ulGmask;
    pSDLPixelFormat->Bmask  = ulBmask;
    pSDLPixelFormat->Rloss  = ulRloss;
    pSDLPixelFormat->Gloss  = ulGloss;
    pSDLPixelFormat->Bloss  = ulBloss;

    pSDLPixelFormat->Ashift = 0x00;
    pSDLPixelFormat->Amask  = 0x00;
    pSDLPixelFormat->Aloss  = 0x00;

    return TRUE;
}

static Uint32 _getSDLPixelFormat(ULONG ulBPP, FOURCC fccColorEncoding)
{
    SDL_PixelFormat stSDLPixelFormat;
    Uint32          uiResult = SDL_PIXELFORMAT_UNKNOWN;

    if (_getSDLPixelFormatData(&stSDLPixelFormat, ulBPP, fccColorEncoding))
        uiResult = SDL_MasksToPixelFormatEnum(ulBPP, stSDLPixelFormat.Rmask,
                                              stSDLPixelFormat.Gmask,
                                              stSDLPixelFormat.Bmask, 0);

    return uiResult;
}

static SDL_DisplayMode *_getDisplayModeForSDLWindow(SDL_Window *window)
{
    SDL_VideoDisplay *pSDLDisplay = SDL_GetDisplayForWindow(window);

    if (!pSDLDisplay) {
        debug_os2("No display for the window");
        return FALSE;
    }

    return &pSDLDisplay->current_mode;
}

static VOID _mouseCheck(WINDATA *pWinData)
{
    SDL_Mouse *pSDLMouse = SDL_GetMouse();

    if ((pSDLMouse->relative_mode || (pWinData->window->flags & SDL_WINDOW_MOUSE_GRABBED) != 0) &&
        ((pWinData->window->flags & SDL_WINDOW_INPUT_FOCUS) != 0)) {
        /* We will make a real capture in _wmMouseButton() */
    } else {
        WinSetCapture(HWND_DESKTOP, NULLHANDLE);
    }
}


/*  PM window procedure.
 *  --------------------
 */
static int OS2_ResizeWindowShape(SDL_Window *window);

static VOID _setVisibleRegion(WINDATA *pWinData, BOOL fVisible)
{
    SDL_VideoDisplay *pSDLDisplay;

    if (!pWinData->pVOData)
        return;

     pSDLDisplay = (fVisible)? SDL_GetDisplayForWindow(pWinData->window) : NULL;
     pWinData->pOutput->SetVisibleRegion(pWinData->pVOData, pWinData->hwnd,
                                         (!pSDLDisplay) ?
                                         NULL : &pSDLDisplay->current_mode,
                                         pWinData->hrgnShape, fVisible);
}

static VOID _wmPaint(WINDATA *pWinData, HWND hwnd)
{
    if (!pWinData->pVOData ||
        !pWinData->pOutput->Update(pWinData->pVOData, hwnd, NULL, 0)) {
        RECTL   rectl;
        HPS     hps;

        hps = WinBeginPaint(hwnd, 0, &rectl);
        WinFillRect(hps, &rectl, CLR_BLACK);
        WinEndPaint(hps);
    }
}

static VOID _wmMouseMove(WINDATA *pWinData, SHORT lX, SHORT lY)
{
    SDL_Mouse *pSDLMouse = SDL_GetMouse();
    POINTL  pointl;
    BOOL    fWinActive = (pWinData->window->flags & SDL_WINDOW_INPUT_FOCUS) != 0;

    if (!pSDLMouse->relative_mode || pSDLMouse->relative_mode_warp) {
        if (!pSDLMouse->relative_mode && fWinActive &&
            ((pWinData->window->flags & SDL_WINDOW_MOUSE_GRABBED) != 0) &&
            (WinQueryCapture(HWND_DESKTOP) == pWinData->hwnd)) {

            pointl.x = lX;
            pointl.y = lY;

            if (lX < 0)
                lX = 0;
            else if (lX >= pWinData->window->w)
                lX = pWinData->window->w - 1;

            if (lY < 0)
                lY = 0;
            else if (lY >= pWinData->window->h)
                lY = pWinData->window->h - 1;

            if (lX != pointl.x || lY != pointl.x) {
                pointl.x = lX;
                pointl.y = lY;
                WinMapWindowPoints(pWinData->hwnd, HWND_DESKTOP, &pointl, 1);
                pWinData->lSkipWMMouseMove++;
                WinSetPointerPos(HWND_DESKTOP, pointl.x, pointl.y);
            }
        }

        SDL_SendMouseMotion(pWinData->window, 0, 0, lX,
                            pWinData->window->h - lY - 1);
        return;
    }

    if (fWinActive) {
        pointl.x = pWinData->window->w / 2;
        pointl.y = pWinData->window->h / 2;
        WinMapWindowPoints(pWinData->hwnd, HWND_DESKTOP, &pointl, 1);

        SDL_SendMouseMotion(pWinData->window, 0, 1,
                            lX - pointl.x, pointl.y - lY);

        pWinData->lSkipWMMouseMove++;
        WinSetPointerPos(HWND_DESKTOP, pointl.x, pointl.y);
    }
}

static VOID _wmMouseButton(WINDATA *pWinData, ULONG ulButton, BOOL fDown)
{
    static ULONG  aBtnGROP2SDL[3] = { SDL_BUTTON_LEFT, SDL_BUTTON_RIGHT,
                                      SDL_BUTTON_MIDDLE };
    SDL_Mouse *pSDLMouse = SDL_GetMouse();

    if ((pSDLMouse->relative_mode || ((pWinData->window->flags & SDL_WINDOW_MOUSE_GRABBED) != 0)) &&
        ((pWinData->window->flags & SDL_WINDOW_INPUT_FOCUS) != 0) &&
        (WinQueryCapture(HWND_DESKTOP) != pWinData->hwnd)) {
        /* Mouse should be captured. */
        if (pSDLMouse->relative_mode && !pSDLMouse->relative_mode_warp) {
            POINTL  pointl;

            pointl.x = pWinData->window->w / 2;
            pointl.y = pWinData->window->h / 2;
            WinMapWindowPoints(pWinData->hwnd, HWND_DESKTOP, &pointl, 1);
            pWinData->lSkipWMMouseMove++;
            WinSetPointerPos(HWND_DESKTOP, pointl.x, pointl.y);
        }

        WinSetCapture(HWND_DESKTOP, pWinData->hwnd);
    }

    SDL_SendMouseButton(pWinData->window, 0,
                        (fDown)? SDL_PRESSED : SDL_RELEASED,
                        aBtnGROP2SDL[ulButton]);
}

static VOID _wmChar(WINDATA *pWinData, MPARAM mp1, MPARAM mp2)
{
    ULONG   ulFlags = SHORT1FROMMP(mp1);      /* WM_CHAR flags         */
    ULONG   ulVirtualKey = SHORT2FROMMP(mp2); /* Virtual key code VK_* */
    ULONG   ulCharCode = SHORT1FROMMP(mp2);   /* Character code        */
    ULONG   ulScanCode = CHAR4FROMMP(mp1);    /* Scan code             */

    if (((ulFlags & (KC_VIRTUALKEY | KC_KEYUP | KC_ALT)) == (KC_VIRTUALKEY | KC_ALT)) &&
        (ulVirtualKey == VK_F4)) {
        SDL_SendWindowEvent(pWinData->window, SDL_WINDOWEVENT_CLOSE, 0, 0);
    }

    if ((ulFlags & KC_SCANCODE) != 0) {
        SDL_SendKeyboardKey(((ulFlags & KC_KEYUP) == 0)? SDL_PRESSED : SDL_RELEASED, aSDLScancode[ulScanCode]);
    }

    if ((ulFlags & KC_CHAR) != 0) {
#if defined(HAVE_ICONV) && defined(HAVE_ICONV_H)
        char *utf8 = SDL_iconv_string("UTF-8", "", (char *)&ulCharCode, 1);
        SDL_SendKeyboardText((utf8 && *utf8) ? utf8 : (char *)&ulCharCode);
        SDL_free(utf8);
#else
        char utf8[4];
        int rc = StrUTF8(1, utf8, sizeof(utf8), (char *)&ulCharCode, 1);
        SDL_SendKeyboardText((rc > 0) ? utf8 : (char *) &ulCharCode);
#endif
    }
}

static VOID _wmMove(WINDATA *pWinData)
{
    SDL_DisplayMode *pSDLDisplayMode = _getDisplayModeForSDLWindow(pWinData->window);
    POINTL  pointl = { 0,0 };
    RECTL   rectl;

    WinQueryWindowRect(pWinData->hwnd, &rectl);
    WinMapWindowPoints(pWinData->hwnd, HWND_DESKTOP, (PPOINTL)&rectl, 2);

    WinMapWindowPoints(pWinData->hwnd, HWND_DESKTOP, &pointl, 1);
    SDL_SendWindowEvent(pWinData->window, SDL_WINDOWEVENT_MOVED, rectl.xLeft,
                       pSDLDisplayMode->h - rectl.yTop);
}

static MRESULT _wmDragOver(WINDATA *pWinData, PDRAGINFO pDragInfo)
{
    ULONG       ulIdx;
    PDRAGITEM   pDragItem;
    USHORT      usDrag   = DOR_NEVERDROP;
    USHORT      usDragOp = DO_UNKNOWN;

    if (!DrgAccessDraginfo(pDragInfo))
        return MRFROM2SHORT(DOR_NEVERDROP, DO_UNKNOWN);

    for (ulIdx = 0; ulIdx < pDragInfo->cditem; ulIdx++) {
        pDragItem = DrgQueryDragitemPtr(pDragInfo, ulIdx);

        /* We accept WPS files only. */
        if (!DrgVerifyRMF(pDragItem, "DRM_OS2FILE", NULL)) {
            usDrag   = DOR_NEVERDROP;
            usDragOp = DO_UNKNOWN;
            break;
        }

        if (pDragInfo->usOperation == DO_DEFAULT &&
            (pDragItem->fsSupportedOps & DO_COPYABLE) != 0) {
            usDrag   = DOR_DROP;
            usDragOp = DO_COPY;
        } else
        if (pDragInfo->usOperation == DO_LINK &&
            (pDragItem->fsSupportedOps & DO_LINKABLE) != 0) {
            usDrag   = DOR_DROP;
            usDragOp = DO_LINK;
        } else {
            usDrag   = DOR_NODROPOP;
            usDragOp = DO_UNKNOWN;
            break;
        }
    }

    /* Update window (The DIVE surface spoiled while dragging) */
    WinInvalidateRect(pWinData->hwnd, NULL, FALSE);
    WinUpdateWindow(pWinData->hwnd);

    DrgFreeDraginfo(pDragInfo);
    return MPFROM2SHORT(usDrag, usDragOp);
}

static MRESULT _wmDrop(WINDATA *pWinData, PDRAGINFO pDragInfo)
{
    ULONG       ulIdx;
    PDRAGITEM   pDragItem;
    CHAR        acFName[CCHMAXPATH];
    PCHAR       pcFName;

    if (!DrgAccessDraginfo(pDragInfo))
        return MRFROM2SHORT(DOR_NEVERDROP, 0);

    for (ulIdx = 0; ulIdx < pDragInfo->cditem; ulIdx++) {
        pDragItem = DrgQueryDragitemPtr(pDragInfo, ulIdx);

        if (DrgVerifyRMF(pDragItem, "DRM_OS2FILE", NULL) &&
            pDragItem->hstrContainerName != NULLHANDLE &&
            pDragItem->hstrSourceName != NULLHANDLE) {
            /* Get file name from the item. */
            DrgQueryStrName(pDragItem->hstrContainerName, sizeof(acFName), acFName);
            pcFName = SDL_strchr(acFName, '\0');
            DrgQueryStrName(pDragItem->hstrSourceName,
                            sizeof(acFName) - (pcFName - acFName), pcFName);

            /* Send to SDL full file name converted to UTF-8. */
            pcFName = OS2_SysToUTF8(acFName);
            SDL_SendDropFile(pWinData->window, pcFName);
            SDL_free(pcFName);

            /* Notify a source that a drag operation is complete. */
            if (pDragItem->hwndItem)
                DrgSendTransferMsg(pDragItem->hwndItem, DM_ENDCONVERSATION,
                                   (MPARAM)pDragItem->ulItemID,
                                   (MPARAM)DMFL_TARGETSUCCESSFUL);
        }
    }

    DrgDeleteDraginfoStrHandles(pDragInfo);
    DrgFreeDraginfo(pDragInfo);

    SDL_SendDropComplete(pWinData->window);

    return (MRESULT)FALSE;
}

static MRESULT EXPENTRY wndFrameProc(HWND hwnd, ULONG msg, MPARAM mp1, MPARAM mp2)
{
    HWND    hwndClient = WinQueryWindow(hwnd, QW_BOTTOM);
    WINDATA * pWinData = (WINDATA *)WinQueryWindowULong(hwndClient, 0);

    if (!pWinData)
        return WinDefWindowProc(hwnd, msg, mp1, mp2);

    /* Send a SDL_SYSWMEVENT if the application wants them */
    if (SDL_GetEventState(SDL_SYSWMEVENT) == SDL_ENABLE) {
        SDL_SysWMmsg wmmsg;

        SDL_VERSION(&wmmsg.version);
        wmmsg.subsystem = SDL_SYSWM_OS2;
        wmmsg.msg.os2.fFrame = TRUE;
        wmmsg.msg.os2.hwnd = hwnd;
        wmmsg.msg.os2.msg = msg;
        wmmsg.msg.os2.mp1 = mp1;
        wmmsg.msg.os2.mp2 = mp2;
        SDL_SendSysWMEvent(&wmmsg);
    }

    switch (msg) {
    case WM_MINMAXFRAME:
        if ((((PSWP)mp1)->fl & SWP_RESTORE) != 0) {
            pWinData->lSkipWMMove += 2;
            SDL_SendWindowEvent(pWinData->window, SDL_WINDOWEVENT_RESTORED, 0, 0);
        }
        if ((((PSWP)mp1)->fl & SWP_MINIMIZE) != 0) {
            pWinData->lSkipWMSize++;
            pWinData->lSkipWMMove += 2;
            SDL_SendWindowEvent(pWinData->window, SDL_WINDOWEVENT_MINIMIZED, 0, 0);
        }
        if ((((PSWP)mp1)->fl & SWP_MAXIMIZE) != 0) {
            SDL_SendWindowEvent(pWinData->window, SDL_WINDOWEVENT_MAXIMIZED, 0, 0);
        }
        break;

    case WM_ADJUSTFRAMEPOS:
        if (pWinData->lSkipWMAdjustFramePos > 0) {
            pWinData->lSkipWMAdjustFramePos++;
            break;
        }
        if ((pWinData->window->flags & SDL_WINDOW_FULLSCREEN) != 0 &&
            (((PSWP)mp1)->fl & SWP_RESTORE) != 0) {
            /* Keep fullscreen window size on restore. */
            RECTL rectl;

            rectl.xLeft = 0;
            rectl.yBottom = 0;
            rectl.xRight = WinQuerySysValue(HWND_DESKTOP, SV_CXSCREEN);
            rectl.yTop = WinQuerySysValue(HWND_DESKTOP, SV_CYSCREEN);
            WinCalcFrameRect(hwnd, &rectl, FALSE);
            ((PSWP)mp1)->x = rectl.xLeft;
            ((PSWP)mp1)->y = rectl.yBottom;
            ((PSWP)mp1)->cx = rectl.xRight - rectl.xLeft;
            ((PSWP)mp1)->cy = rectl.yTop - rectl.yBottom;
        }
        if ((((PSWP)mp1)->fl & (SWP_SIZE | SWP_MINIMIZE)) == SWP_SIZE) {
            if ((pWinData->window->flags & SDL_WINDOW_FULLSCREEN) != 0) {
                /* SDL_WINDOW_FULLSCREEN_DESKTOP have flag SDL_WINDOW_FULLSCREEN... */
                if (SDL_IsShapedWindow(pWinData->window))
                    OS2_ResizeWindowShape(pWinData->window);
                break;
            }
            if ((SDL_GetWindowFlags(pWinData->window) & SDL_WINDOW_RESIZABLE) != 0) {
                RECTL   rectl;
                int     iMinW, iMinH, iMaxW, iMaxH;
                int     iWinW, iWinH;

                rectl.xLeft = 0;
                rectl.yBottom = 0;
                SDL_GetWindowSize(pWinData->window,
                                  (int *)&rectl.xRight, (int *)&rectl.yTop);
                iWinW = rectl.xRight;
                iWinH = rectl.yTop;

                SDL_GetWindowMinimumSize(pWinData->window, &iMinW, &iMinH);
                SDL_GetWindowMaximumSize(pWinData->window, &iMaxW, &iMaxH);

                if (iWinW < iMinW)
                    rectl.xRight = iMinW;
                else if (iMaxW != 0 && iWinW > iMaxW)
                    rectl.xRight = iMaxW;

                if (iWinH < iMinH)
                    rectl.yTop = iMinW;
                else if (iMaxH != 0 && iWinH > iMaxH)
                    rectl.yTop = iMaxH;

                if (rectl.xRight == iWinW && rectl.yTop == iWinH) {
                    if (SDL_IsShapedWindow(pWinData->window))
                        OS2_ResizeWindowShape(pWinData->window);
                    break;
                }

                WinCalcFrameRect(hwnd, &rectl, FALSE);
                ((PSWP)mp1)->cx = rectl.xRight - rectl.xLeft;
                ((PSWP)mp1)->cy = rectl.yTop - rectl.yBottom;
            }
        }
        break;
    }

    return pWinData->fnWndFrameProc(hwnd, msg, mp1, mp2);
}

static MRESULT EXPENTRY wndProc(HWND hwnd, ULONG msg, MPARAM mp1, MPARAM mp2)
{
    WINDATA *pWinData = (WINDATA *)WinQueryWindowULong(hwnd, 0);

    if (!pWinData)
        return WinDefWindowProc(hwnd, msg, mp1, mp2);

    /* Send a SDL_SYSWMEVENT if the application wants them */
    if (SDL_GetEventState(SDL_SYSWMEVENT) == SDL_ENABLE) {
        SDL_SysWMmsg wmmsg;

        SDL_VERSION(&wmmsg.version);
        wmmsg.subsystem = SDL_SYSWM_OS2;
        wmmsg.msg.os2.fFrame = FALSE;
        wmmsg.msg.os2.hwnd = hwnd;
        wmmsg.msg.os2.msg = msg;
        wmmsg.msg.os2.mp1 = mp1;
        wmmsg.msg.os2.mp2 = mp2;
        SDL_SendSysWMEvent(&wmmsg);
    }

    switch (msg) {
    case WM_CLOSE:
    case WM_QUIT:
        SDL_SendWindowEvent(pWinData->window, SDL_WINDOWEVENT_CLOSE, 0, 0);
        if (!pWinData->fnUserWndProc)
            return (MRESULT)FALSE;
        break;

    case WM_PAINT:
        _wmPaint(pWinData, hwnd);
        break;

    case WM_SHOW:
        SDL_SendWindowEvent(pWinData->window, (SHORT1FROMMP(mp1) == 0)?
                                               SDL_WINDOWEVENT_HIDDEN :
                                               SDL_WINDOWEVENT_SHOWN   ,
                            0, 0);
        break;

    case WM_UPDATEFRAME:
        /* Return TRUE - no further action for the frame control window procedure */
        return (MRESULT)TRUE;

    case WM_ACTIVATE:
        if ((BOOL)mp1) {
            POINTL  pointl;

            if (SDL_GetKeyboardFocus() != pWinData->window)
                SDL_SetKeyboardFocus(pWinData->window);

            WinQueryPointerPos(HWND_DESKTOP, &pointl);
            WinMapWindowPoints(HWND_DESKTOP, pWinData->hwnd, &pointl, 1);
            SDL_SendMouseMotion(pWinData->window, 0, 0,
                                pointl.x, pWinData->window->h - pointl.y - 1);
        } else {
            if (SDL_GetKeyboardFocus() == pWinData->window)
                SDL_SetKeyboardFocus(NULL);

            WinSetCapture(HWND_DESKTOP,  NULLHANDLE);
        }
        break;

    case WM_MOUSEMOVE:
        WinSetPointer(HWND_DESKTOP, hptrCursor);

        if (pWinData->lSkipWMMouseMove > 0)
            pWinData->lSkipWMMouseMove--;
        else {
            _wmMouseMove(pWinData, SHORT1FROMMP(mp1), SHORT2FROMMP(mp1));
        }
        return (MRESULT)FALSE;

    case WM_BUTTON1DOWN:
    case WM_BUTTON1DBLCLK:
        _wmMouseButton(pWinData, 0, TRUE);
        break;

    case WM_BUTTON1UP:
        _wmMouseButton(pWinData, 0, FALSE);
        break;

    case WM_BUTTON2DOWN:
    case WM_BUTTON2DBLCLK:
        _wmMouseButton(pWinData, 1, TRUE);
        break;

    case WM_BUTTON2UP:
        _wmMouseButton(pWinData, 1, FALSE);
        break;

    case WM_BUTTON3DOWN:
    case WM_BUTTON3DBLCLK:
        _wmMouseButton(pWinData, 2, TRUE);
        break;

    case WM_BUTTON3UP:
        _wmMouseButton(pWinData, 2, FALSE);
        break;

    case WM_TRANSLATEACCEL:
        /* ALT and acceleration keys not allowed (must be processed in WM_CHAR) */
        if (!mp1 || ((PQMSG)mp1)->msg != WM_CHAR)
            break;
        return (MRESULT)FALSE;

    case WM_CHAR:
        _wmChar(pWinData, mp1, mp2);
        break;

    case WM_SIZE:
        if (pWinData->lSkipWMSize > 0)
            pWinData->lSkipWMSize--;
        else {
            if ((pWinData->window->flags & SDL_WINDOW_FULLSCREEN) == 0) {
                SDL_SendWindowEvent(pWinData->window, SDL_WINDOWEVENT_RESIZED,
                                    SHORT1FROMMP(mp2), SHORT2FROMMP(mp2));
            } else {
                pWinData->lSkipWMVRNEnabled++;
            }
        }
        break;

    case WM_MOVE:
        if (pWinData->lSkipWMMove > 0)
            pWinData->lSkipWMMove--;
        else if ((pWinData->window->flags & SDL_WINDOW_FULLSCREEN) == 0) {
            _wmMove(pWinData);
        }
        break;

    case WM_VRNENABLED:
        if (pWinData->lSkipWMVRNEnabled > 0)
            pWinData->lSkipWMVRNEnabled--;
        else {
            _setVisibleRegion(pWinData, TRUE);
        }
        return (MRESULT)TRUE;

    case WM_VRNDISABLED:
        _setVisibleRegion(pWinData, FALSE);
        return (MRESULT)TRUE;

    case DM_DRAGOVER:
        return _wmDragOver(pWinData, (PDRAGINFO)PVOIDFROMMP(mp1));

    case DM_DROP:
        return _wmDrop(pWinData, (PDRAGINFO)PVOIDFROMMP(mp1));
    }

    return (pWinData->fnUserWndProc)?
            pWinData->fnUserWndProc(hwnd, msg, mp1, mp2) :
            WinDefWindowProc(hwnd, msg, mp1, mp2);
}


/*  SDL routines.
 *  ------------
 */

static void OS2_PumpEvents(_THIS)
{
    SDL_VideoData *pVData = (SDL_VideoData *)_this->driverdata;
    QMSG  qmsg;

    if (WinPeekMsg(pVData->hab, &qmsg, NULLHANDLE, 0, 0, PM_REMOVE))
        WinDispatchMsg(pVData->hab, &qmsg);
}

static WINDATA *_setupWindow(_THIS, SDL_Window *window, HWND hwndFrame,
                             HWND hwnd)
{
    SDL_VideoData *pVData = (SDL_VideoData *)_this->driverdata;
    WINDATA       *pWinData = SDL_calloc(1, sizeof(WINDATA));

    if (!pWinData) {
        SDL_OutOfMemory();
        return NULL;
     }
    pWinData->hwnd = hwnd;
    pWinData->hwndFrame = hwndFrame;
    pWinData->window = window;
    window->driverdata = pWinData;

    WinSetWindowULong(hwnd, 0, (ULONG)pWinData);
    pWinData->fnWndFrameProc = WinSubclassWindow(hwndFrame, wndFrameProc);

    pWinData->pOutput = pVData->pOutput;
    pWinData->pVOData = pVData->pOutput->Open();

    WinSetVisibleRegionNotify(hwnd, TRUE);

    return pWinData;
}

static int OS2_CreateWindow(_THIS, SDL_Window *window)
{
    RECTL            rectl;
    HWND             hwndFrame, hwnd;
    SDL_DisplayMode *pSDLDisplayMode = _getDisplayModeForSDLWindow(window);
    ULONG            ulFrameFlags = FCF_TASKLIST  | FCF_TITLEBAR | FCF_SYSMENU |
                                    FCF_MINBUTTON | FCF_SHELLPOSITION;
    ULONG            ulSWPFlags   = SWP_SIZE | SWP_SHOW | SWP_ZORDER | SWP_ACTIVATE;
    WINDATA         *pWinData;

    if (!pSDLDisplayMode)
        return -1;

    /* Create a PM window */
    if ((window->flags & SDL_WINDOW_RESIZABLE) != 0)
        ulFrameFlags |= FCF_SIZEBORDER | FCF_DLGBORDER | FCF_MAXBUTTON;
    else if ((window->flags & SDL_WINDOW_BORDERLESS) == 0)
        ulFrameFlags |= FCF_DLGBORDER;

    if ((window->flags & SDL_WINDOW_MAXIMIZED) != 0)
        ulSWPFlags |= SWP_MAXIMIZE;
    else if ((window->flags & SDL_WINDOW_MINIMIZED) != 0)
        ulSWPFlags |= SWP_MINIMIZE;

    hwndFrame = WinCreateStdWindow(HWND_DESKTOP, 0, &ulFrameFlags,
                                   WIN_CLIENT_CLASS, "SDL2", 0, 0, 0, &hwnd);
    if (hwndFrame == NULLHANDLE)
        return SDL_SetError("Couldn't create window");

    /* Setup window data and frame window procedure */
    pWinData = _setupWindow(_this, window, hwndFrame, hwnd);
    if (!pWinData) {
        WinDestroyWindow(hwndFrame);
        return -1;
    }

    /* Show window */
    rectl.xLeft   = 0;
    rectl.yBottom = 0;
    rectl.xRight  = window->w;
    rectl.yTop    = window->h;
    WinCalcFrameRect(hwndFrame, &rectl, FALSE);
    pWinData->lSkipWMSize++;
    pWinData->lSkipWMMove++;
    WinSetWindowPos(hwndFrame, HWND_TOP, rectl.xLeft, rectl.yBottom,
                    rectl.xRight - rectl.xLeft, rectl.yTop - rectl.yBottom,
                    ulSWPFlags);

    rectl.xLeft   = 0;
    rectl.yBottom = 0;
    WinMapWindowPoints(hwnd, HWND_DESKTOP, (PPOINTL)&rectl, 1);
    window->x = rectl.xLeft;
    window->y = pSDLDisplayMode->h - (rectl.yBottom + window->h);

    window->flags |= SDL_WINDOW_SHOWN;

    return 0;
}

static int OS2_CreateWindowFrom(_THIS, SDL_Window *window, const void *data)
{
    SDL_VideoData   *pVData = (SDL_VideoData *)_this->driverdata;
    CHAR             acBuf[256];
    CLASSINFO        stCI;
    HWND             hwndUser = (HWND)data;
    HWND             hwndFrame, hwnd;
    ULONG            cbText;
    PSZ              pszText;
    WINDATA         *pWinData;
    SDL_DisplayMode *pSDLDisplayMode = _getDisplayModeForSDLWindow(window);
    SWP              swp;
    POINTL           pointl;

    debug_os2("Enter");
    if (!pSDLDisplayMode)
        return -1;

    /* User can accept client OR frame window handle.
     * Get client and frame window handles. */
    WinQueryClassName(hwndUser, sizeof(acBuf), acBuf);
    if (!WinQueryClassInfo(pVData->hab, acBuf, &stCI))
        return SDL_SetError("Cannot get user window class information");

    if ((stCI.flClassStyle & CS_FRAME) == 0) {
        /* Client window handle is specified */
        hwndFrame = WinQueryWindow(hwndUser, QW_PARENT);
        if (hwndFrame == NULLHANDLE)
            return SDL_SetError("Cannot get parent window handle");

        if ((ULONG)WinSendMsg(hwndFrame, WM_QUERYFRAMEINFO, 0, 0) == 0)
            return SDL_SetError("Parent window is not a frame window");

        hwnd = hwndUser;
    } else {
        /* Frame window handle is specified */
        hwnd = WinWindowFromID(hwndUser, FID_CLIENT);
        if (hwnd == NULLHANDLE)
            return SDL_SetError("Cannot get client window handle");

        hwndFrame = hwndUser;

        WinQueryClassName(hwnd, sizeof(acBuf), acBuf);
        if (!WinQueryClassInfo(pVData->hab, acBuf, &stCI))
            return SDL_SetError("Cannot get client window class information");
    }

    /* Check window's reserved storage */
    if (stCI.cbWindowData < sizeof(ULONG))
        return SDL_SetError("Reserved storage of window must be at least %u bytes", sizeof(ULONG));

    /* Set SDL-window title */
    cbText = WinQueryWindowTextLength(hwndFrame);
    pszText = SDL_stack_alloc(CHAR, cbText + 1);

    if (pszText != NULL)
        cbText = (pszText != NULL)? WinQueryWindowText(hwndFrame, cbText, pszText) : 0;

    if (cbText != 0)
        window->title = OS2_SysToUTF8(pszText);

    if (pszText != NULL) {
        SDL_stack_free(pszText);
    }

    /* Set SDL-window flags */
    window->flags &= ~(SDL_WINDOW_SHOWN     | SDL_WINDOW_BORDERLESS |
                       SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED  |
                       SDL_WINDOW_MINIMIZED | SDL_WINDOW_INPUT_FOCUS);

    if (WinIsWindowVisible(hwnd))
        window->flags |= SDL_WINDOW_SHOWN;

    WinSendMsg(hwndFrame, WM_QUERYBORDERSIZE, MPFROMP(&pointl), 0);
    if (pointl.y == WinQuerySysValue(HWND_DESKTOP, SV_CYSIZEBORDER))
        window->flags |= SDL_WINDOW_RESIZABLE;
    else if (pointl.y <= WinQuerySysValue(HWND_DESKTOP, SV_CYBORDER))
        window->flags |= SDL_WINDOW_BORDERLESS;

    WinQueryWindowPos(hwndFrame, &swp);

    if ((swp.fl & SWP_MAXIMIZE) != 0)
        window->flags |= SDL_WINDOW_MAXIMIZED;
    if ((swp.fl & SWP_MINIMIZE) != 0)
        window->flags |= SDL_WINDOW_MINIMIZED;

    pointl.x = 0;
    pointl.y = 0;
    WinMapWindowPoints(hwnd, HWND_DESKTOP, &pointl, 1);
    window->x = pointl.x;
    window->y = pSDLDisplayMode->h - (pointl.y + swp.cy);

    WinQueryWindowPos(hwnd, &swp);
    window->w = swp.cx;
    window->h = swp.cy;

    /* Setup window data and frame window procedure */
    pWinData = _setupWindow(_this, window, hwndFrame, hwnd);
    if (!pWinData) {
        SDL_free(window->title);
        window->title = NULL;
        return -1;
    }
    pWinData->fnUserWndProc = WinSubclassWindow(hwnd, wndProc);

    if (WinQueryActiveWindow(HWND_DESKTOP) == hwndFrame)
        SDL_SetKeyboardFocus(window);

    return 0;
}

static void OS2_DestroyWindow(_THIS, SDL_Window * window)
{
    SDL_VideoData *pVData = (SDL_VideoData *)_this->driverdata;
    WINDATA       *pWinData = (WINDATA *)window->driverdata;

    debug_os2("Enter");
    if (!pWinData)
        return;

    if (pWinData->hrgnShape != NULLHANDLE) {
        HPS hps = WinGetPS(pWinData->hwnd);
        GpiDestroyRegion(hps, pWinData->hrgnShape);
        WinReleasePS(hps);
    }

    if (window->shaper) {
        SDL_free(window->shaper);
        window->shaper = NULL;
    }

    if (!pWinData->fnUserWndProc) {
        /* Window was created by SDL (OS2_CreateWindow()),
         * not by user (OS2_CreateWindowFrom()) */
        WinDestroyWindow(pWinData->hwndFrame);
    } else {
        WinSetWindowULong(pWinData->hwnd, 0, 0);
    }

    if ((pVData) && (pWinData->pVOData)) {
        pVData->pOutput->Close(pWinData->pVOData);
        pWinData->pVOData = NULL;
    }

    if (pWinData->hptrIcon != NULLHANDLE) {
        WinDestroyPointer(pWinData->hptrIcon);
        pWinData->hptrIcon = NULLHANDLE;
    }

    SDL_free(pWinData);
    window->driverdata = NULL;
}

static void OS2_SetWindowTitle(_THIS, SDL_Window *window)
{
    PSZ pszTitle = (!window->title)? NULL : OS2_UTF8ToSys(window->title);

    WinSetWindowText(((WINDATA *)window->driverdata)->hwndFrame, pszTitle);
    SDL_free(pszTitle);
}

static void OS2_SetWindowIcon(_THIS, SDL_Window *window, SDL_Surface *icon)
{
    WINDATA  *pWinData = (WINDATA *)window->driverdata;
    HPOINTER  hptr = utilCreatePointer(icon, 0, 0);

    if (hptr == NULLHANDLE)
        return;

    /* Destroy old icon */
    if (pWinData->hptrIcon != NULLHANDLE)
        WinDestroyPointer(pWinData->hptrIcon);

    /* Set new window icon */
    pWinData->hptrIcon = hptr;
    if (!WinSendMsg(pWinData->hwndFrame, WM_SETICON, MPFROMLONG(hptr), 0)) {
        debug_os2("Cannot set icon for the window");
    }
}

static void OS2_SetWindowPosition(_THIS, SDL_Window *window)
{
    WINDATA         *pWinData = (WINDATA *)window->driverdata;
    RECTL            rectl;
    ULONG            ulFlags;
    SDL_DisplayMode *pSDLDisplayMode = _getDisplayModeForSDLWindow(window);

    debug_os2("Enter");
    if (!pSDLDisplayMode)
        return;

    rectl.xLeft = 0;
    rectl.yBottom = 0;
    rectl.xRight = window->w;
    rectl.yTop = window->h;
    WinCalcFrameRect(pWinData->hwndFrame, &rectl, FALSE);

    if (SDL_ShouldAllowTopmost() &&
        (window->flags & (SDL_WINDOW_FULLSCREEN|SDL_WINDOW_INPUT_FOCUS)) ==
                         (SDL_WINDOW_FULLSCREEN|SDL_WINDOW_INPUT_FOCUS) )
        ulFlags = SWP_ZORDER | SWP_MOVE | SWP_SIZE;
    else
        ulFlags = SWP_MOVE | SWP_SIZE;

    pWinData->lSkipWMSize++;
    pWinData->lSkipWMMove++;
    WinSetWindowPos(pWinData->hwndFrame, HWND_TOP,
                    window->x + rectl.xLeft,
                    (pSDLDisplayMode->h - window->y) - window->h + rectl.yBottom,
                    rectl.xRight - rectl.xLeft, rectl.yTop - rectl.yBottom,
                    ulFlags);
}

static void OS2_SetWindowSize(_THIS, SDL_Window *window)
{
    debug_os2("Enter");
    OS2_SetWindowPosition(_this, window);
}

static void OS2_ShowWindow(_THIS, SDL_Window *window)
{
    WINDATA *pWinData = (WINDATA *)window->driverdata;

    debug_os2("Enter");
    WinShowWindow(pWinData->hwndFrame, TRUE);
}

static void OS2_HideWindow(_THIS, SDL_Window *window)
{
    WINDATA *pWinData = (WINDATA *)window->driverdata;

    debug_os2("Enter");
    WinShowWindow(pWinData->hwndFrame, FALSE);
}

static void OS2_RaiseWindow(_THIS, SDL_Window *window)
{
    debug_os2("Enter");
    OS2_SetWindowPosition(_this, window);
}

static void OS2_MaximizeWindow(_THIS, SDL_Window *window)
{
    WINDATA *pWinData = (WINDATA *)window->driverdata;

    debug_os2("Enter");
    WinSetWindowPos(pWinData->hwndFrame, HWND_TOP, 0, 0, 0, 0, SWP_MAXIMIZE);
}

static void OS2_MinimizeWindow(_THIS, SDL_Window *window)
{
    WINDATA *pWinData = (WINDATA *)window->driverdata;

    debug_os2("Enter");
    WinSetWindowPos(pWinData->hwndFrame, HWND_TOP, 0, 0, 0, 0, SWP_MINIMIZE | SWP_DEACTIVATE);
}

static void OS2_RestoreWindow(_THIS, SDL_Window *window)
{
    WINDATA *pWinData = (WINDATA *)window->driverdata;

    debug_os2("Enter");
    WinSetWindowPos(pWinData->hwndFrame, HWND_TOP, 0, 0, 0, 0, SWP_RESTORE);
}

static void OS2_SetWindowBordered(_THIS, SDL_Window * window,
                                  SDL_bool bordered)
{
    WINDATA *pWinData = (WINDATA *)window->driverdata;
    ULONG    ulStyle = WinQueryWindowULong(pWinData->hwndFrame, QWL_STYLE);
    RECTL    rectl;

    debug_os2("Enter");

    /* New frame sytle */
    if (bordered)
        ulStyle |= ((window->flags & SDL_WINDOW_RESIZABLE) != 0) ? FS_SIZEBORDER : FS_DLGBORDER;
    else
        ulStyle &= ~(FS_SIZEBORDER | FS_BORDER | FS_DLGBORDER);

    /* Save client window position */
    WinQueryWindowRect(pWinData->hwnd, &rectl);
    WinMapWindowPoints(pWinData->hwnd, HWND_DESKTOP, (PPOINTL)&rectl, 2);

    /* Change the frame */
    WinSetWindowULong(pWinData->hwndFrame, QWL_STYLE, ulStyle);
    WinSendMsg(pWinData->hwndFrame, WM_UPDATEFRAME, MPFROMLONG(FCF_BORDER), 0);

    /* Restore client window position */
    WinCalcFrameRect(pWinData->hwndFrame, &rectl, FALSE);
    pWinData->lSkipWMMove++;
    WinSetWindowPos(pWinData->hwndFrame, HWND_TOP, rectl.xLeft, rectl.yBottom,
                    rectl.xRight - rectl.xLeft,
                    rectl.yTop - rectl.yBottom,
                    SWP_SIZE | SWP_MOVE | SWP_NOADJUST);
}

static void OS2_SetWindowFullscreen(_THIS, SDL_Window *window,
                                    SDL_VideoDisplay *display,
                                    SDL_bool fullscreen)
{
    RECTL            rectl;
    ULONG            ulFlags;
    WINDATA         *pWinData = (WINDATA *)window->driverdata;
    SDL_DisplayMode *pSDLDisplayMode = &display->current_mode;

    debug_os2("Enter, fullscreen: %u", fullscreen);

    if (!pSDLDisplayMode)
        return;

    if (SDL_ShouldAllowTopmost() &&
        (window->flags & (SDL_WINDOW_FULLSCREEN|SDL_WINDOW_INPUT_FOCUS)) == (SDL_WINDOW_FULLSCREEN|SDL_WINDOW_INPUT_FOCUS))
        ulFlags = SWP_SIZE | SWP_MOVE | SWP_ZORDER | SWP_NOADJUST;
    else
        ulFlags = SWP_SIZE | SWP_MOVE | SWP_NOADJUST;

    if (fullscreen) {
        rectl.xLeft = 0;
        rectl.yBottom = 0;
        rectl.xRight = pSDLDisplayMode->w;
        rectl.yTop = pSDLDisplayMode->h;
        /* We need send the restore command now to allow WinCalcFrameRect() */
        WinSetWindowPos(pWinData->hwndFrame, HWND_TOP, 0, 0, 0, 0, SWP_RESTORE);
    } else {
        pWinData->lSkipWMMove++;
        rectl.xLeft = window->windowed.x;
        rectl.yTop = pSDLDisplayMode->h - window->windowed.y;
        rectl.xRight = rectl.xLeft + window->windowed.w;
        rectl.yBottom = rectl.yTop - window->windowed.h;
    }

    if (!WinCalcFrameRect(pWinData->hwndFrame, &rectl, FALSE)) {
        debug_os2("WinCalcFrameRect() failed");
    }
    else if (!WinSetWindowPos(pWinData->hwndFrame, HWND_TOP,
                              rectl.xLeft, rectl.yBottom,
                              rectl.xRight - rectl.xLeft, rectl.yTop - rectl.yBottom,
                              ulFlags)) {
        debug_os2("WinSetWindowPos() failed");
    }
}

static SDL_bool OS2_GetWindowWMInfo(_THIS, SDL_Window * window,
                                    struct SDL_SysWMinfo *info)
{
    WINDATA *pWinData = (WINDATA *)window->driverdata;

    if (info->version.major <= SDL_MAJOR_VERSION) {
        info->subsystem = SDL_SYSWM_OS2;
        info->info.os2.hwnd = pWinData->hwnd;
        info->info.os2.hwndFrame = pWinData->hwndFrame;
        return SDL_TRUE;
    }

    SDL_SetError("Application not compiled with SDL %u",
                 SDL_MAJOR_VERSION);
    return SDL_FALSE;
}

static void OS2_OnWindowEnter(_THIS, SDL_Window * window)
{
}

static int OS2_SetWindowHitTest(SDL_Window *window, SDL_bool enabled)
{
  debug_os2("Enter");
  return 0;
}

static void OS2_SetWindowMouseGrab(_THIS, SDL_Window *window, SDL_bool grabbed)
{
    WINDATA *pWinData = (WINDATA *)window->driverdata;

    debug_os2("Enter, %u", grabbed);
    _mouseCheck(pWinData);
}


/* Shaper
 */
typedef struct _SHAPERECTS {
  PRECTL     pRects;
  ULONG      cRects;
  ULONG      ulWinHeight;
} SHAPERECTS;

static void _combineRectRegions(SDL_ShapeTree *node, void *closure)
{
    SHAPERECTS *pShapeRects = (SHAPERECTS *)closure;
    PRECTL      pRect;

    /* Expand rectangles list */
    if ((pShapeRects->cRects & 0x0F) == 0) {
        pRect = SDL_realloc(pShapeRects->pRects, (pShapeRects->cRects + 0x10) * sizeof(RECTL));
        if (!pRect)
            return;
        pShapeRects->pRects = pRect;
    }

    /* Add a new rectangle */
    pRect = &pShapeRects->pRects[pShapeRects->cRects];
    pShapeRects->cRects++;
    /* Fill rectangle data */
    pRect->xLeft = node->data.shape.x;
    pRect->yTop = pShapeRects->ulWinHeight - node->data.shape.y;
    pRect->xRight += node->data.shape.w;
    pRect->yBottom = pRect->yTop - node->data.shape.h;
}

static SDL_WindowShaper* OS2_CreateShaper(SDL_Window * window)
{
    SDL_WindowShaper* pSDLShaper = SDL_malloc(sizeof(SDL_WindowShaper));

    debug_os2("Enter");
    pSDLShaper->window = window;
    pSDLShaper->mode.mode = ShapeModeDefault;
    pSDLShaper->mode.parameters.binarizationCutoff = 1;
    pSDLShaper->userx = 0;
    pSDLShaper->usery = 0;
    pSDLShaper->driverdata = (SDL_ShapeTree *)NULL;
    window->shaper = pSDLShaper;

    if (OS2_ResizeWindowShape(window) != 0) {
        window->shaper = NULL;
        SDL_free(pSDLShaper);
        return NULL;
    }

    return pSDLShaper;
}

static int OS2_SetWindowShape(SDL_WindowShaper *shaper, SDL_Surface *shape,
                              SDL_WindowShapeMode *shape_mode)
{
    SDL_ShapeTree *pShapeTree;
    WINDATA       *pWinData;
    SHAPERECTS     stShapeRects;
    HPS            hps;

    debug_os2("Enter");
    if (!shaper || !shape ||
        (shape->format->Amask == 0 && shape_mode->mode != ShapeModeColorKey) ||
        shape->w != shaper->window->w || shape->h != shaper->window->h) {
        return SDL_INVALID_SHAPE_ARGUMENT;
    }

    if (shaper->driverdata)
        SDL_FreeShapeTree((SDL_ShapeTree **)&shaper->driverdata);

    pShapeTree = SDL_CalculateShapeTree(*shape_mode, shape);
    shaper->driverdata = pShapeTree;

    SDL_zero(stShapeRects);
    stShapeRects.ulWinHeight = shaper->window->h;
    SDL_TraverseShapeTree(pShapeTree, &_combineRectRegions, &stShapeRects);

    pWinData = (WINDATA *)shaper->window->driverdata;
    hps = WinGetPS(pWinData->hwnd);

    if (pWinData->hrgnShape != NULLHANDLE)
        GpiDestroyRegion(hps, pWinData->hrgnShape);

    pWinData->hrgnShape = (!stShapeRects.pRects) ? NULLHANDLE :
                                GpiCreateRegion(hps, stShapeRects.cRects, stShapeRects.pRects);

    WinReleasePS(hps);
    SDL_free(stShapeRects.pRects);
    WinSendMsg(pWinData->hwnd, WM_VRNENABLED, 0, 0);

    return 0;
}

static int OS2_ResizeWindowShape(SDL_Window *window)
{
    debug_os2("Enter");
    if (!window)
        return -1;

    if (window->x != -1000) {
        if (window->shaper->driverdata)
            SDL_FreeShapeTree((SDL_ShapeTree **)window->shaper->driverdata);

        if (window->shaper->hasshape == SDL_TRUE) {
            window->shaper->userx = window->x;
            window->shaper->usery = window->y;
            SDL_SetWindowPosition(window, -1000, -1000);
        }
    }

    return 0;
}


/* Frame buffer
 */
static void OS2_DestroyWindowFramebuffer(_THIS, SDL_Window *window)
{
    WINDATA *pWinData = (WINDATA *)window->driverdata;

    debug_os2("Enter");
    if (pWinData && pWinData->pVOData)
        pWinData->pOutput->VideoBufFree(pWinData->pVOData);
}

static int OS2_CreateWindowFramebuffer(_THIS, SDL_Window *window,
                                       Uint32 *format, void **pixels,
                                       int *pitch)
{
    WINDATA          *pWinData = (WINDATA *)window->driverdata;
    SDL_VideoDisplay *pSDLDisplay = SDL_GetDisplayForWindow(window);
    SDL_DisplayMode  *pSDLDisplayMode;
    MODEDATA         *pModeData;
    ULONG             ulWidth, ulHeight;

    debug_os2("Enter");
    if (!pSDLDisplay) {
        debug_os2("No display for the window");
        return -1;
    }

    pSDLDisplayMode = &pSDLDisplay->current_mode;
    pModeData = (MODEDATA *)pSDLDisplayMode->driverdata;
    if (!pModeData)
        return SDL_SetError("No mode data for the display");

    SDL_GetWindowSize(window, (int *)&ulWidth, (int *)&ulHeight);
    debug_os2("Window size: %u x %u", ulWidth, ulHeight);

    *pixels = pWinData->pOutput->VideoBufAlloc(
                        pWinData->pVOData, ulWidth, ulHeight, pModeData->ulDepth,
                        pModeData->fccColorEncoding, (PULONG)pitch);
    if (!*pixels)
        return -1;

    *format = pSDLDisplayMode->format;
    debug_os2("Pitch: %u, frame buffer: 0x%X.", *pitch, *pixels);
    WinSendMsg(pWinData->hwnd, WM_VRNENABLED, 0, 0);

    return 0;
}

static int OS2_UpdateWindowFramebuffer(_THIS, SDL_Window * window,
                                       const SDL_Rect *rects, int numrects)
{
    WINDATA *pWinData = (WINDATA *)window->driverdata;

    return pWinData->pOutput->Update(pWinData->pVOData, pWinData->hwnd,
                                     (SDL_Rect *)rects, (ULONG)numrects)
           ? 0 : -1;
}


/* Clipboard
 */
static int OS2_SetClipboardText(_THIS, const char *text)
{
    SDL_VideoData *pVData = (SDL_VideoData *)_this->driverdata;
    PSZ   pszClipboard;
    PSZ   pszText = (!text)? NULL : OS2_UTF8ToSys(text);
    ULONG cbText;
    ULONG ulRC;
    BOOL  fSuccess;

    debug_os2("Enter");
    if (!pszText)
        return -1;
    cbText = SDL_strlen(pszText) + 1;

    ulRC = DosAllocSharedMem((PPVOID)&pszClipboard, 0, cbText,
                              PAG_COMMIT | PAG_READ | PAG_WRITE |
                              OBJ_GIVEABLE | OBJ_GETTABLE | OBJ_TILE);
    if (ulRC != NO_ERROR) {
        debug_os2("DosAllocSharedMem() failed, rc = %u", ulRC);
        SDL_free(pszText);
        return -1;
    }

    SDL_memcpy(pszClipboard, pszText, cbText);
    SDL_free(pszText);

    if (!WinOpenClipbrd(pVData->hab)) {
        debug_os2("WinOpenClipbrd() failed");
        fSuccess = FALSE;
    } else {
        WinEmptyClipbrd(pVData->hab);
        fSuccess = WinSetClipbrdData(pVData->hab, (ULONG)pszClipboard, CF_TEXT, CFI_POINTER);
        if (!fSuccess) {
            debug_os2("WinOpenClipbrd() failed");
        }
        WinCloseClipbrd(pVData->hab);
    }

    if (!fSuccess) {
        DosFreeMem(pszClipboard);
        return -1;
    }
    return 0;
}

static char *OS2_GetClipboardText(_THIS)
{
    SDL_VideoData *pVData = (SDL_VideoData *)_this->driverdata;
    PSZ pszClipboard = NULL;

    if (!WinOpenClipbrd(pVData->hab)) {
        debug_os2("WinOpenClipbrd() failed");
    } else {
        pszClipboard = (PSZ)WinQueryClipbrdData(pVData->hab, CF_TEXT);
        if (pszClipboard != NULL)
            pszClipboard = OS2_SysToUTF8(pszClipboard);
        WinCloseClipbrd(pVData->hab);
    }

    return (!pszClipboard) ? SDL_strdup("") : pszClipboard;
}

static SDL_bool OS2_HasClipboardText(_THIS)
{
    SDL_VideoData *pVData = (SDL_VideoData *)_this->driverdata;
    PSZ pszClipboard;
    SDL_bool  result;

    if (!WinOpenClipbrd(pVData->hab)) {
        debug_os2("WinOpenClipbrd() failed");
        return SDL_FALSE;
    }

    pszClipboard = (PSZ)WinQueryClipbrdData(pVData->hab, CF_TEXT);
    result = (pszClipboard && *pszClipboard) ? SDL_TRUE : SDL_FALSE;
    WinCloseClipbrd(pVData->hab);

    return result;
}


static int OS2_VideoInit(_THIS)
{
    SDL_VideoData *pVData;
    PTIB  tib;
    PPIB  pib;

    /* Create SDL video driver private data */
    pVData = SDL_calloc(1, sizeof(SDL_VideoData));
    if (!pVData)
        return SDL_OutOfMemory();

    /* Change process type code for use Win* API from VIO session */
    DosGetInfoBlocks(&tib, &pib);
    if (pib->pib_ultype == 2 || pib->pib_ultype == 0) {
        /* VIO windowable or fullscreen protect-mode session */
        pib->pib_ultype = 3; /* Presentation Manager protect-mode session */
    }

    /* PM initialization */
    pVData->hab = WinInitialize(0);
    pVData->hmq = WinCreateMsgQueue(pVData->hab, 0);
    if (pVData->hmq == NULLHANDLE) {
        SDL_free(pVData);
        return SDL_SetError("Message queue cannot be created.");
    }

    if (!WinRegisterClass(pVData->hab, WIN_CLIENT_CLASS, wndProc,
                          CS_SIZEREDRAW | CS_MOVENOTIFY | CS_SYNCPAINT,
                          sizeof(SDL_VideoData*))) {
        SDL_free(pVData);
        return SDL_SetError("Window class not successfully registered.");
    }

    if (SDL_strcasecmp(_this->name, OS2DRIVER_NAME_VMAN) == 0)
        pVData->pOutput = &voVMan;
    else
        pVData->pOutput = &voDive;

    _this->driverdata = pVData;

    /* Add display */
    {
        SDL_VideoDisplay    stSDLDisplay;
        SDL_DisplayMode     stSDLDisplayMode;
        DISPLAYDATA        *pDisplayData;
        MODEDATA           *pModeData;
        VIDEOOUTPUTINFO     stVOInfo;

        if (!pVData->pOutput->QueryInfo(&stVOInfo)) {
            SDL_free(pVData);
            return SDL_SetError("Video mode query failed.");
        }

        SDL_zero(stSDLDisplay); SDL_zero(stSDLDisplayMode);

        stSDLDisplayMode.format = _getSDLPixelFormat(stVOInfo.ulBPP,
                                                     stVOInfo.fccColorEncoding);
        stSDLDisplayMode.w = stVOInfo.ulHorizResolution;
        stSDLDisplayMode.h = stVOInfo.ulVertResolution;
        stSDLDisplayMode.refresh_rate = 0;
        stSDLDisplayMode.driverdata = NULL;

        pModeData = SDL_malloc(sizeof(MODEDATA));
        if (pModeData) {
            pModeData->ulDepth = stVOInfo.ulBPP;
            pModeData->fccColorEncoding = stVOInfo.fccColorEncoding;
            pModeData->ulScanLineBytes = stVOInfo.ulScanLineSize;
            stSDLDisplayMode.driverdata = pModeData;
        }

        stSDLDisplay.name = "Primary";
        stSDLDisplay.desktop_mode = stSDLDisplayMode;
        stSDLDisplay.current_mode = stSDLDisplayMode;
        stSDLDisplay.driverdata = NULL;
        stSDLDisplay.num_display_modes = 0;

        pDisplayData = SDL_malloc(sizeof(DISPLAYDATA));
        if (pDisplayData) {
            HPS hps = WinGetPS(HWND_DESKTOP);
            HDC hdc = GpiQueryDevice(hps);

            /* May be we can use CAPS_HORIZONTAL_RESOLUTION and
             * CAPS_VERTICAL_RESOLUTION - pels per meter?  */
            DevQueryCaps(hdc, CAPS_HORIZONTAL_FONT_RES, 1,
                          (PLONG)&pDisplayData->ulDPIHor);
            DevQueryCaps(hdc, CAPS_VERTICAL_FONT_RES, 1,
                          (PLONG)&pDisplayData->ulDPIVer);
            WinReleasePS(hps);

            pDisplayData->ulDPIDiag = SDL_ComputeDiagonalDPI(
                  stVOInfo.ulHorizResolution, stVOInfo.ulVertResolution,
                  (float)stVOInfo.ulHorizResolution / pDisplayData->ulDPIHor,
                  (float)stVOInfo.ulVertResolution / pDisplayData->ulDPIVer);

            stSDLDisplayMode.driverdata = pDisplayData;
        }

        SDL_AddVideoDisplay(&stSDLDisplay, SDL_FALSE);
    }

    OS2_InitMouse(_this, pVData->hab);

    return 0;
}

static void OS2_VideoQuit(_THIS)
{
    SDL_VideoData *pVData = (SDL_VideoData *)_this->driverdata;

    OS2_QuitMouse(_this);

    WinDestroyMsgQueue(pVData->hmq);
    WinTerminate(pVData->hab);

    /* our caller SDL_VideoQuit() already frees display_modes, driverdata, etc. */
}

static int OS2_GetDisplayBounds(_THIS, SDL_VideoDisplay *display,
                                SDL_Rect *rect)
{
    debug_os2("Enter");

    rect->x = 0;
    rect->y = 0;
    rect->w = display->desktop_mode.w;
    rect->h = display->desktop_mode.h;

    return 0;
}

static int OS2_GetDisplayDPI(_THIS, SDL_VideoDisplay *display, float *ddpi,
                             float *hdpi, float *vdpi)
{
    DISPLAYDATA *pDisplayData = (DISPLAYDATA *)display->driverdata;

    debug_os2("Enter");
    if (!pDisplayData)
        return -1;

    if (ddpi)
        *hdpi = pDisplayData->ulDPIDiag;
    if (hdpi)
        *hdpi = pDisplayData->ulDPIHor;
    if (vdpi)
        *vdpi = pDisplayData->ulDPIVer;

    return 0;
}

static void OS2_GetDisplayModes(_THIS, SDL_VideoDisplay *display)
{
    SDL_DisplayMode mode;

    debug_os2("Enter");
    SDL_copyp(&mode, &display->current_mode);
    mode.driverdata = (MODEDATA *) SDL_malloc(sizeof(MODEDATA));
    if (!mode.driverdata) return; /* yikes.. */
    SDL_memcpy(mode.driverdata, display->current_mode.driverdata, sizeof(MODEDATA));
    SDL_AddDisplayMode(display, &mode);
}

static int OS2_SetDisplayMode(_THIS, SDL_VideoDisplay *display,
                              SDL_DisplayMode *mode)
{
    debug_os2("Enter");
    return -1;
}


static void OS2_DeleteDevice(SDL_VideoDevice *device)
{
    SDL_free(device);
}

static SDL_VideoDevice *OS2_CreateDevice(void)
{
    SDL_VideoDevice *device;

    /* Initialize all variables that we clean on shutdown */
    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        SDL_OutOfMemory();
        return NULL;
    }

    /* Set the function pointers */
    device->VideoInit = OS2_VideoInit;
    device->VideoQuit = OS2_VideoQuit;
    device->GetDisplayBounds = OS2_GetDisplayBounds;
    device->GetDisplayDPI = OS2_GetDisplayDPI;
    device->GetDisplayModes = OS2_GetDisplayModes;
    device->SetDisplayMode = OS2_SetDisplayMode;
    device->PumpEvents = OS2_PumpEvents;
    device->CreateSDLWindow = OS2_CreateWindow;
    device->CreateSDLWindowFrom = OS2_CreateWindowFrom;
    device->DestroyWindow = OS2_DestroyWindow;
    device->SetWindowTitle = OS2_SetWindowTitle;
    device->SetWindowIcon = OS2_SetWindowIcon;
    device->SetWindowPosition = OS2_SetWindowPosition;
    device->SetWindowSize = OS2_SetWindowSize;
    device->ShowWindow = OS2_ShowWindow;
    device->HideWindow = OS2_HideWindow;
    device->RaiseWindow = OS2_RaiseWindow;
    device->MaximizeWindow = OS2_MaximizeWindow;
    device->MinimizeWindow = OS2_MinimizeWindow;
    device->RestoreWindow = OS2_RestoreWindow;
    device->SetWindowBordered = OS2_SetWindowBordered;
    device->SetWindowFullscreen = OS2_SetWindowFullscreen;
    device->GetWindowWMInfo = OS2_GetWindowWMInfo;
    device->OnWindowEnter = OS2_OnWindowEnter;
    device->SetWindowHitTest = OS2_SetWindowHitTest;
    device->SetWindowMouseGrab = OS2_SetWindowMouseGrab;
    device->CreateWindowFramebuffer = OS2_CreateWindowFramebuffer;
    device->UpdateWindowFramebuffer = OS2_UpdateWindowFramebuffer;
    device->DestroyWindowFramebuffer = OS2_DestroyWindowFramebuffer;

    device->SetClipboardText = OS2_SetClipboardText;
    device->GetClipboardText = OS2_GetClipboardText;
    device->HasClipboardText = OS2_HasClipboardText;

    device->shape_driver.CreateShaper = OS2_CreateShaper;
    device->shape_driver.SetWindowShape = OS2_SetWindowShape;
    device->shape_driver.ResizeWindowShape = OS2_ResizeWindowShape;

    device->free = OS2_DeleteDevice;

    return device;
}

static SDL_VideoDevice *OS2DIVE_CreateDevice(void)
{
    VIDEOOUTPUTINFO stVOInfo;
    if (!voDive.QueryInfo(&stVOInfo)) {
        return NULL;
    }
    return OS2_CreateDevice();
}

static SDL_VideoDevice *OS2VMAN_CreateDevice(void)
{
    VIDEOOUTPUTINFO stVOInfo;
    if (!voVMan.QueryInfo(&stVOInfo)) {
          return NULL;
    }
    return OS2_CreateDevice();
}


/* DIVE and VMAN bootstraps both call the same OS2_CreateDevice() function.
 * Video output system will be selected in OS2_VideoInit() by driver name.  */
VideoBootStrap OS2DIVE_bootstrap =
{
    OS2DRIVER_NAME_DIVE, "OS/2 video driver",
    OS2DIVE_CreateDevice,
    OS2_ShowMessageBox
};
VideoBootStrap OS2VMAN_bootstrap =
{
    OS2DRIVER_NAME_VMAN, "OS/2 video driver",
    OS2VMAN_CreateDevice,
    OS2_ShowMessageBox
};

#endif /* SDL_VIDEO_DRIVER_OS2 */

/* vi: set ts=4 sw=4 expandtab: */
