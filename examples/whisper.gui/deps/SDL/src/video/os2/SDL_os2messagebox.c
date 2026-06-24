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

/* Display a OS/2 message box */

#include "SDL.h"
#include "../../core/os2/SDL_os2.h"
#include "SDL_os2video.h"
#define INCL_WIN
#include <os2.h>

#define IDD_TEXT_MESSAGE    1001
#define IDD_BITMAP          1002
#define IDD_PB_FIRST        1003

typedef struct _MSGBOXDLGDATA {
    USHORT      cb;
    HWND        hwndUnder;
} MSGBOXDLGDATA;

static VOID _wmInitDlg(HWND hwnd, MSGBOXDLGDATA *pDlgData)
{
    HPS     hps = WinGetPS(hwnd);
    POINTL  aptText[TXTBOX_COUNT];
    HENUM   hEnum;
    HWND    hWndNext;
    CHAR    acBuf[256];
    ULONG   cbBuf;
    ULONG   cButtons = 0;
    ULONG   ulButtonsCY = 0;
    ULONG   ulButtonsCX = 0;
    RECTL   rectl;
    ULONG   ulX;
    ULONG   ulIdx;
    struct _BUTTON {
      HWND  hwnd;   /* Button window handle. */
      ULONG ulCX;   /* Button width in dialog coordinates. */
    } aButtons[32];
    RECTL   rectlItem;
    HAB     hab = WinQueryAnchorBlock(hwnd);

    /* --- Align the buttons to the right/bottom. --- */

    /* Collect window handles of all buttons in dialog. */
    hEnum = WinBeginEnumWindows(hwnd);

    while ((hWndNext = WinGetNextWindow(hEnum)) != NULLHANDLE) {
        if (WinQueryClassName(hWndNext, sizeof(acBuf), acBuf) == 0) {
            continue;
        }
        if (SDL_strcmp(acBuf, "#3") == 0) { /* Class name of button. */
            if (cButtons < sizeof(aButtons) / sizeof(struct _BUTTON)) {
                aButtons[cButtons].hwnd = hWndNext;
                cButtons++;
            }
        }
    }
    WinEndEnumWindows(hEnum);

    /* Query size of text for each button, get width of each button, total
     * buttons width (ulButtonsCX) and max. height (ulButtonsCX) in _dialog
     * coordinates_. */
    hps = WinGetPS(hwnd);

    for(ulIdx = 0; ulIdx < cButtons; ulIdx++) {
        /* Query size of text in window coordinates. */
        cbBuf = WinQueryWindowText(aButtons[ulIdx].hwnd, sizeof(acBuf), acBuf);
        GpiQueryTextBox(hps, cbBuf, acBuf, TXTBOX_COUNT, aptText);
        aptText[TXTBOX_TOPRIGHT].x -= aptText[TXTBOX_BOTTOMLEFT].x;
        aptText[TXTBOX_TOPRIGHT].y -= aptText[TXTBOX_BOTTOMLEFT].y;
        /* Convert text size to dialog coordinates. */
        WinMapDlgPoints(hwnd, &aptText[TXTBOX_TOPRIGHT], 1, FALSE);
        /* Add vertical and horizontal space for button's frame (dialog coord.). */
        if (aptText[TXTBOX_TOPRIGHT].x < 30) {          /* Minimal button width. */
            aptText[TXTBOX_TOPRIGHT].x = 30;
        } else {
            aptText[TXTBOX_TOPRIGHT].x += 4;
        }
        aptText[TXTBOX_TOPRIGHT].y += 3;

        aButtons[ulIdx].ulCX = aptText[TXTBOX_TOPRIGHT].x; /* Store button width   */
        ulButtonsCX += aptText[TXTBOX_TOPRIGHT].x + 2;     /* Add total btn. width */
        /* Get max. height for buttons. */
        if (ulButtonsCY < aptText[TXTBOX_TOPRIGHT].y)
            ulButtonsCY = aptText[TXTBOX_TOPRIGHT].y + 1;
    }

    WinReleasePS(hps);

    /* Expand horizontal size of the window to fit all buttons and move window
     * to the center of parent window. */

    /* Convert total width of buttons to window coordinates. */
    aptText[0].x = ulButtonsCX + 4;
    WinMapDlgPoints(hwnd, &aptText[0], 1, TRUE);
    /* Check width of the window and expand as needed. */
    WinQueryWindowRect(hwnd, &rectlItem);
    if (rectlItem.xRight <= aptText[0].x)
        rectlItem.xRight = aptText[0].x;

    /* Move window rectangle to the center of owner window. */
    WinQueryWindowRect(pDlgData->hwndUnder, &rectl);
    /* Left-bottom point of centered dialog on owner window. */
    rectl.xLeft = (rectl.xRight - rectlItem.xRight) / 2;
    rectl.yBottom = (rectl.yTop - rectlItem.yTop) / 2;
    /* Map left-bottom point to desktop. */
    WinMapWindowPoints(pDlgData->hwndUnder, HWND_DESKTOP, (PPOINTL)&rectl, 1);
    WinOffsetRect(hab, &rectlItem, rectl.xLeft, rectl.yBottom);

    /* Set new rectangle for the window. */
    WinSetWindowPos(hwnd, HWND_TOP, rectlItem.xLeft, rectlItem.yBottom,
                    rectlItem.xRight - rectlItem.xLeft,
                    rectlItem.yTop - rectlItem.yBottom,
                    SWP_SIZE | SWP_MOVE);

    /* Set buttons positions. */

    /* Get horizontal position for the first button. */
    WinMapDlgPoints(hwnd, (PPOINTL)&rectlItem, 2, FALSE);       /* Win size to dlg coord. */
    ulX = rectlItem.xRight - rectlItem.xLeft - ulButtonsCX - 2; /* First button position. */

    /* Set positions and sizes for all buttons. */
    for (ulIdx = 0; ulIdx < cButtons; ulIdx++) {
        /* Get poisition and size for the button in dialog coordinates. */
        aptText[0].x = ulX;
        aptText[0].y = 2;
        aptText[1].x = aButtons[ulIdx].ulCX;
        aptText[1].y = ulButtonsCY;
        /* Convert to window coordinates. */
        WinMapDlgPoints(hwnd, aptText, 2, TRUE);

        WinSetWindowPos(aButtons[ulIdx].hwnd, HWND_TOP,
                        aptText[0].x, aptText[0].y, aptText[1].x, aptText[1].y,
                        SWP_MOVE | SWP_SIZE);

        /* Offset horizontal position for the next button. */
        ulX += aButtons[ulIdx].ulCX + 2;
    }

    /* Set right bound of the text to right bound of the last button and
     * bottom bound of the text just above the buttons. */

    aptText[2].x = 25;              /* Left bound of text in dlg coordinates.  */
    aptText[2].y = ulButtonsCY + 3; /* Bottom bound of the text in dlg coords. */
    WinMapDlgPoints(hwnd, &aptText[2], 1, TRUE); /* Convert ^^^ to win. coords */
    hWndNext = WinWindowFromID(hwnd, IDD_TEXT_MESSAGE);
    WinQueryWindowRect(hWndNext, &rectlItem);
    rectlItem.xLeft = aptText[2].x;
    rectlItem.yBottom = aptText[2].y;
    /* Right bound of the text equals right bound of the last button. */
    rectlItem.xRight = aptText[0].x + aptText[1].x;
    WinSetWindowPos(hWndNext, HWND_TOP, rectlItem.xLeft, rectlItem.yBottom,
                    rectlItem.xRight - rectlItem.xLeft,
                    rectlItem.yTop - rectlItem.yBottom,
                    SWP_MOVE | SWP_SIZE);
}

static MRESULT EXPENTRY DynDlgProc(HWND hwnd, USHORT message, MPARAM mp1, MPARAM mp2)
{
    switch (message) {
    case WM_INITDLG:
        _wmInitDlg(hwnd, (MSGBOXDLGDATA*)mp2);
        break;

    case WM_COMMAND:
        switch (SHORT1FROMMP(mp1)) {
        case DID_OK:
            WinDismissDlg(hwnd, FALSE);
            break;
        default:
            break;
        }

    default:
        return(WinDefDlgProc(hwnd, message, mp1, mp2));
    }

    return FALSE;
}

static HWND _makeDlg(const SDL_MessageBoxData *messageboxdata)
{
    SDL_MessageBoxButtonData*
        pSDLBtnData =  (SDL_MessageBoxButtonData *)messageboxdata->buttons;
    ULONG               cSDLBtnData = messageboxdata->numbuttons;

    PSZ                 pszTitle = OS2_UTF8ToSys(messageboxdata->title);
    ULONG               cbTitle  = (!pszTitle)? 1 : (SDL_strlen(pszTitle) + 1);
    PSZ                 pszText  = OS2_UTF8ToSys(messageboxdata->message);
    ULONG               cbText   = (!pszText)? 1 : (SDL_strlen(pszText) + 1);

    PDLGTEMPLATE        pTemplate;
    ULONG               cbTemplate;
    ULONG               ulIdx;
    PCHAR               pcDlgData;
    PDLGTITEM           pDlgItem;
    PSZ                 pszBtnText;
    ULONG               cbBtnText;
    HWND                hwnd;

    const SDL_MessageBoxColor* pSDLColors = (!messageboxdata->colorScheme)?
                                       NULL : messageboxdata->colorScheme->colors;
    const SDL_MessageBoxColor* pSDLColor;

    MSGBOXDLGDATA       stDlgData;

    /* Build a dialog tamplate in memory */

    /* Size of template */
    cbTemplate = sizeof(DLGTEMPLATE) + ((2 + cSDLBtnData) * sizeof(DLGTITEM)) +
                 sizeof(ULONG) +  /* First item data - frame control data. */
                 cbTitle       +  /* First item data - frame title + ZERO. */
                 cbText        +  /* Second item data - ststic text + ZERO.*/
                 3;               /* Third item data - system icon Id.     */

    /* Button items datas - text for buttons. */
    for (ulIdx = 0; ulIdx < cSDLBtnData; ulIdx++) {
        pszBtnText = (PSZ)pSDLBtnData[ulIdx].text;
        cbTemplate += (!pszBtnText)? 1 : (SDL_strlen(pszBtnText) + 1);
    }

    /* Presentation parameter space. */
    if (pSDLColors) {
        cbTemplate += 26 /* PP for frame.       */ +
                      26 /* PP for static text. */ +
                     (48 * cSDLBtnData); /* PP for buttons. */
    }

    /* Allocate memory for the dialog template. */
    pTemplate = (PDLGTEMPLATE) SDL_malloc(cbTemplate);
    /* Pointer on data for dialog items in allocated memory. */
    pcDlgData = &((PCHAR)pTemplate)[sizeof(DLGTEMPLATE) +
                                    ((2 + cSDLBtnData) * sizeof(DLGTITEM))];

    /* Header info */
    pTemplate->cbTemplate = cbTemplate; /* size of dialog template to pass to WinCreateDlg() */
    pTemplate->type = 0;                /* Currently always 0. */
    pTemplate->codepage = 0;
    pTemplate->offadlgti = 14;          /* Offset to array of DLGTITEMs. */
    pTemplate->fsTemplateStatus = 0;    /* Reserved field?  */

    /* Index in array of dlg items of item to get focus,          */
    /* if 0 then focus goes to first control that can have focus. */
    pTemplate->iItemFocus = 0;
    pTemplate->coffPresParams = 0;

    /* First item info - frame */
    pDlgItem = pTemplate->adlgti;
    pDlgItem->fsItemStatus = 0;  /* Reserved? */
    /* Number of dialog item child windows owned by this item. */
    pDlgItem->cChildren = 2 + cSDLBtnData; /* Ststic text + buttons. */
    /* Length of class name, if 0 then offClassname contains a WC_ value. */
    pDlgItem->cchClassName = 0;
    pDlgItem->offClassName = (USHORT)WC_FRAME;
    /* Length of text. */
    pDlgItem->cchText = cbTitle;
    pDlgItem->offText = pcDlgData - (PCHAR)pTemplate; /* Offset to title text.  */
    /* Copy text for the title into the dialog template. */
    if (pszTitle != NULL) {
        SDL_memcpy(pcDlgData, pszTitle, cbTitle);
    } else {
        *pcDlgData = '\0';
    }
    pcDlgData += pDlgItem->cchText;

    pDlgItem->flStyle = WS_GROUP | WS_VISIBLE | WS_CLIPSIBLINGS |
                        FS_DLGBORDER | WS_SAVEBITS;
    pDlgItem->x  = 100;
    pDlgItem->y  = 100;
    pDlgItem->cx = 175;
    pDlgItem->cy = 65;
    pDlgItem->id = DID_OK; /* An ID value? */
    if (!pSDLColors)
        pDlgItem->offPresParams = 0;
    else {
        /* Presentation parameter for the frame - dialog colors. */
        pDlgItem->offPresParams = pcDlgData - (PCHAR)pTemplate;
        ((PPRESPARAMS)pcDlgData)->cb = 22;
        pcDlgData += 4;
        ((PPARAM)pcDlgData)->id = PP_FOREGROUNDCOLOR;
        ((PPARAM)pcDlgData)->cb = 3;
        ((PPARAM)pcDlgData)->ab[0] = pSDLColors[SDL_MESSAGEBOX_COLOR_TEXT].b;
        ((PPARAM)pcDlgData)->ab[1] = pSDLColors[SDL_MESSAGEBOX_COLOR_TEXT].g;
        ((PPARAM)pcDlgData)->ab[2] = pSDLColors[SDL_MESSAGEBOX_COLOR_TEXT].r;
        pcDlgData += 11;
        ((PPARAM)pcDlgData)->id = PP_BACKGROUNDCOLOR;
        ((PPARAM)pcDlgData)->cb = 3;
        ((PPARAM)pcDlgData)->ab[0] = pSDLColors[SDL_MESSAGEBOX_COLOR_BACKGROUND].b;
        ((PPARAM)pcDlgData)->ab[1] = pSDLColors[SDL_MESSAGEBOX_COLOR_BACKGROUND].g;
        ((PPARAM)pcDlgData)->ab[2] = pSDLColors[SDL_MESSAGEBOX_COLOR_BACKGROUND].r;
        pcDlgData += 11;
    }

    /* Offset to ctl data. */
    pDlgItem->offCtlData = pcDlgData - (PCHAR)pTemplate;
    /* Put CtlData for the dialog in here */
    *((PULONG)pcDlgData) = FCF_TITLEBAR | FCF_SYSMENU;
    pcDlgData += sizeof(ULONG);

    /* Second item info - static text (message). */
    pDlgItem++;
    pDlgItem->fsItemStatus = 0;
    /* No children since its a control, it could have child control */
    /* (ex. a group box).                                           */
    pDlgItem->cChildren = 0;
    /* Length of class name, 0 - offClassname contains a WC_ constant. */
    pDlgItem->cchClassName = 0;
    pDlgItem->offClassName = (USHORT)WC_STATIC;

    pDlgItem->cchText = cbText;
    pDlgItem->offText = pcDlgData - (PCHAR)pTemplate;   /* Offset to the text. */
    /* Copy message text into the dialog template. */
    if (pszText != NULL) {
        SDL_memcpy(pcDlgData, pszText, cbText);
    } else {
        *pcDlgData = '\0';
    }
    pcDlgData += pDlgItem->cchText;

    pDlgItem->flStyle = SS_TEXT | DT_TOP | DT_LEFT | DT_WORDBREAK | WS_VISIBLE;
    /* It will be really set in _wmInitDlg(). */
    pDlgItem->x = 25;
    pDlgItem->y = 13;
    pDlgItem->cx = 147;
    pDlgItem->cy = 62;  /* It will be used. */

    pDlgItem->id = IDD_TEXT_MESSAGE;      /* an ID value */
    if (!pSDLColors)
        pDlgItem->offPresParams = 0;
    else {
        /* Presentation parameter for the static text - dialog colors. */
        pDlgItem->offPresParams = pcDlgData - (PCHAR)pTemplate;
        ((PPRESPARAMS)pcDlgData)->cb = 22;
        pcDlgData += 4;
        ((PPARAM)pcDlgData)->id = PP_FOREGROUNDCOLOR;
        ((PPARAM)pcDlgData)->cb = 3;
        ((PPARAM)pcDlgData)->ab[0] = pSDLColors[SDL_MESSAGEBOX_COLOR_TEXT].b;
        ((PPARAM)pcDlgData)->ab[1] = pSDLColors[SDL_MESSAGEBOX_COLOR_TEXT].g;
        ((PPARAM)pcDlgData)->ab[2] = pSDLColors[SDL_MESSAGEBOX_COLOR_TEXT].r;
        pcDlgData += 11;
        ((PPARAM)pcDlgData)->id = PP_BACKGROUNDCOLOR;
        ((PPARAM)pcDlgData)->cb = 3;
        ((PPARAM)pcDlgData)->ab[0] = pSDLColors[SDL_MESSAGEBOX_COLOR_BACKGROUND].b;
        ((PPARAM)pcDlgData)->ab[1] = pSDLColors[SDL_MESSAGEBOX_COLOR_BACKGROUND].g;
        ((PPARAM)pcDlgData)->ab[2] = pSDLColors[SDL_MESSAGEBOX_COLOR_BACKGROUND].r;
        pcDlgData += 11;
    }
    pDlgItem->offCtlData = 0;

    /* Third item info - static bitmap. */
    pDlgItem++;
    pDlgItem->fsItemStatus = 0;
    pDlgItem->cChildren = 0;
    pDlgItem->cchClassName = 0;
    pDlgItem->offClassName = (USHORT)WC_STATIC;

    pDlgItem->cchText = 3; /* 0xFF, low byte of the icon Id, high byte of icon Id. */
    pDlgItem->offText = pcDlgData - (PCHAR)pTemplate;   /* Offset to the Id. */
    /* Write system icon ID into dialog template. */
    *((PBYTE)pcDlgData) = 0xFF; /* First byte is 0xFF, next 2 are system pointer Id. */
    pcDlgData++;
    *((PUSHORT)pcDlgData) = ((messageboxdata->flags & SDL_MESSAGEBOX_ERROR) != 0)?
                              SPTR_ICONERROR :
                                ((messageboxdata->flags & SDL_MESSAGEBOX_WARNING) != 0)?
                                  SPTR_ICONWARNING : SPTR_ICONINFORMATION;
    pcDlgData += 2;

    pDlgItem->flStyle = SS_SYSICON | WS_VISIBLE;

    pDlgItem->x = 4;
    pDlgItem->y = 45; /* It will be really set in _wmInitDlg(). */
    pDlgItem->cx = 0;
    pDlgItem->cy = 0;

    pDlgItem->id = IDD_BITMAP;
    pDlgItem->offPresParams = 0;
    pDlgItem->offCtlData = 0;

    /* Next items - buttons. */
    for (ulIdx = 0; ulIdx < cSDLBtnData; ulIdx++) {
        pDlgItem++;

        pDlgItem->fsItemStatus = 0;
        pDlgItem->cChildren = 0;     /* No children. */
        pDlgItem->cchClassName = 0;  /* 0 - offClassname is WC_ constant. */
        pDlgItem->offClassName = (USHORT)WC_BUTTON;

        pszBtnText = OS2_UTF8ToSys(pSDLBtnData[ulIdx].text);
        cbBtnText = (!pszBtnText)? 1 : (SDL_strlen(pszBtnText) + 1);
        pDlgItem->cchText = cbBtnText;
        pDlgItem->offText = pcDlgData - (PCHAR)pTemplate; /* Offset to the text. */
        /* Copy text for the button into the dialog template. */
        if (pszBtnText != NULL) {
            SDL_memcpy(pcDlgData, pszBtnText, cbBtnText);
        } else {
            *pcDlgData = '\0';
        }
        pcDlgData += pDlgItem->cchText;
        SDL_free(pszBtnText);

        pDlgItem->flStyle = BS_PUSHBUTTON | WS_TABSTOP | WS_VISIBLE;
        if (pSDLBtnData[ulIdx].flags == SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT) {
            pDlgItem->flStyle |= BS_DEFAULT;
            pTemplate->iItemFocus = ulIdx + 3; /* +3 - frame, static text and icon. */
            pSDLColor = &pSDLColors[SDL_MESSAGEBOX_COLOR_BUTTON_SELECTED];
        } else {
            pSDLColor = &pSDLColors[SDL_MESSAGEBOX_COLOR_TEXT];
        }

        /* It will be really set in _wmInitDlg() */
        pDlgItem->x = 10;
        pDlgItem->y = 10;
        pDlgItem->cx = 70;
        pDlgItem->cy = 15;

        pDlgItem->id = IDD_PB_FIRST + ulIdx;  /* an ID value */
        if (!pSDLColors)
            pDlgItem->offPresParams = 0;
        else {
            /* Presentation parameter for the button - dialog colors. */
            pDlgItem->offPresParams = pcDlgData - (PCHAR)pTemplate;
            ((PPRESPARAMS)pcDlgData)->cb = 44;
            pcDlgData += 4;
            ((PPARAM)pcDlgData)->id = PP_FOREGROUNDCOLOR;
            ((PPARAM)pcDlgData)->cb = 3;
            ((PPARAM)pcDlgData)->ab[0] = pSDLColor->b;
            ((PPARAM)pcDlgData)->ab[1] = pSDLColor->g;
            ((PPARAM)pcDlgData)->ab[2] = pSDLColor->r;
            pcDlgData += 11;
            ((PPARAM)pcDlgData)->id = PP_BACKGROUNDCOLOR;
            ((PPARAM)pcDlgData)->cb = 3;
            ((PPARAM)pcDlgData)->ab[0] = pSDLColors[SDL_MESSAGEBOX_COLOR_BUTTON_BACKGROUND].b;
            ((PPARAM)pcDlgData)->ab[1] = pSDLColors[SDL_MESSAGEBOX_COLOR_BUTTON_BACKGROUND].g;
            ((PPARAM)pcDlgData)->ab[2] = pSDLColors[SDL_MESSAGEBOX_COLOR_BUTTON_BACKGROUND].r;
            pcDlgData += 11;
            ((PPARAM)pcDlgData)->id = PP_BORDERLIGHTCOLOR;
            ((PPARAM)pcDlgData)->cb = 3;
            ((PPARAM)pcDlgData)->ab[0] = pSDLColors[SDL_MESSAGEBOX_COLOR_BUTTON_BORDER].b;
            ((PPARAM)pcDlgData)->ab[1] = pSDLColors[SDL_MESSAGEBOX_COLOR_BUTTON_BORDER].g;
            ((PPARAM)pcDlgData)->ab[2] = pSDLColors[SDL_MESSAGEBOX_COLOR_BUTTON_BORDER].r;
            pcDlgData += 11;
            ((PPARAM)pcDlgData)->id = PP_BORDERDARKCOLOR;
            ((PPARAM)pcDlgData)->cb = 3;
            ((PPARAM)pcDlgData)->ab[0] = pSDLColors[SDL_MESSAGEBOX_COLOR_BUTTON_BORDER].b;
            ((PPARAM)pcDlgData)->ab[1] = pSDLColors[SDL_MESSAGEBOX_COLOR_BUTTON_BORDER].g;
            ((PPARAM)pcDlgData)->ab[2] = pSDLColors[SDL_MESSAGEBOX_COLOR_BUTTON_BORDER].r;
            pcDlgData += 11;
        }
        pDlgItem->offCtlData = 0;
    }
    /* Check, end of templ. data: &((PCHAR)pTemplate)[cbTemplate] == pcDlgData */

    /* Create the dialog from template. */
    stDlgData.cb = sizeof(MSGBOXDLGDATA);
    stDlgData.hwndUnder = (messageboxdata->window && messageboxdata->window->driverdata)?
                            ((WINDATA *)messageboxdata->window->driverdata)->hwnd : HWND_DESKTOP;

    hwnd = WinCreateDlg(HWND_DESKTOP, /* Parent is desktop. */
                        stDlgData.hwndUnder,
                        (PFNWP)DynDlgProc, pTemplate, &stDlgData);
    SDL_free(pTemplate);
    SDL_free(pszTitle);
    SDL_free(pszText);

    return hwnd;
}


int OS2_ShowMessageBox(const SDL_MessageBoxData *messageboxdata, int *buttonid)
{
    HWND    hwnd;
    ULONG   ulRC;
    SDL_MessageBoxButtonData
            *pSDLBtnData = (SDL_MessageBoxButtonData *)messageboxdata->buttons;
    ULONG   cSDLBtnData = messageboxdata->numbuttons;
    BOOL    fVideoInitialized = SDL_WasInit(SDL_INIT_VIDEO);
    HAB     hab;
    HMQ     hmq;
    BOOL    fSuccess = FALSE;

    if (!fVideoInitialized) {
        PTIB    tib;
        PPIB    pib;

        DosGetInfoBlocks(&tib, &pib);
        if (pib->pib_ultype == 2 || pib->pib_ultype == 0) {
            /* VIO windowable or fullscreen protect-mode session */
            pib->pib_ultype = 3; /* Presentation Manager protect-mode session */
        }

        hab = WinInitialize(0);
        if (hab == NULLHANDLE) {
            debug_os2("WinInitialize() failed");
            return -1;
        }
        hmq = WinCreateMsgQueue(hab, 0);
        if (hmq == NULLHANDLE) {
            debug_os2("WinCreateMsgQueue() failed");
            return -1;
        }
    }

    /* Create dynamic dialog. */
    hwnd = _makeDlg(messageboxdata);
    /* Show dialog and obtain button Id. */
    ulRC = WinProcessDlg(hwnd);
    /* Destroy dialog, */
    WinDestroyWindow(hwnd);

    if (ulRC == DID_CANCEL) {
        /* Window closed by ESC, Alt+F4 or system menu. */
        ULONG   ulIdx;

        for (ulIdx = 0; ulIdx < cSDLBtnData; ulIdx++, pSDLBtnData++) {
            if (pSDLBtnData->flags == SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT) {
                *buttonid = pSDLBtnData->buttonid;
                fSuccess = TRUE;
                break;
            }
        }
    } else {
        /* Button pressed. */
        ulRC -= IDD_PB_FIRST;
        if (ulRC < cSDLBtnData) {
            *buttonid = pSDLBtnData[ulRC].buttonid;
            fSuccess = TRUE;
        }
    }

    if (!fVideoInitialized) {
        WinDestroyMsgQueue(hmq);
        WinTerminate(hab);
    }

    return (fSuccess)? 0 : -1;
}

#endif /* SDL_VIDEO_DRIVER_OS2 */

/* vi: set ts=4 sw=4 expandtab: */
