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

#ifdef SDL_VIDEO_DRIVER_NGAGE

#include <SDL.h>

#include "../SDL_sysvideo.h"
#include "SDL_ngagevideo.h"
#include "SDL_ngageframebuffer_c.h"

#define NGAGE_SURFACE "NGAGE_FrameBuffer"

/* For 12 bit screen HW. Table for fast conversion from 8 bit to 12 bit
 *
 * TUint16 is enough, but using TUint32 so we can use better instruction
 * selection on ARMI.
 */
static TUint32 NGAGE_HWPalette_256_to_Screen[256];

int GetBpp(TDisplayMode displaymode);
void DirectUpdate(_THIS, int numrects, SDL_Rect *rects);
void DrawBackground(_THIS);
void DirectDraw(_THIS, int numrects, SDL_Rect *rects, TUint16 *screenBuffer);
void RedrawWindowL(_THIS);

int SDL_NGAGE_CreateWindowFramebuffer(_THIS, SDL_Window *window, Uint32 *format, void **pixels, int *pitch)
{
    SDL_VideoData *phdata = (SDL_VideoData *)_this->driverdata;
    SDL_Surface *surface;
    const Uint32 surface_format = SDL_PIXELFORMAT_RGB444;
    int w, h;

    /* Free the old framebuffer surface */
    SDL_NGAGE_DestroyWindowFramebuffer(_this, window);

    /* Create a new one */
    SDL_GetWindowSizeInPixels(window, &w, &h);
    surface = SDL_CreateRGBSurfaceWithFormat(0, w, h, 0, surface_format);
    if (!surface) {
        return -1;
    }

    /* Save the info and return! */
    SDL_SetWindowData(window, NGAGE_SURFACE, surface);
    *format = surface_format;
    *pixels = surface->pixels;
    *pitch = surface->pitch;

    /* Initialise Epoc frame buffer */

    TDisplayMode displayMode = phdata->NGAGE_WsScreen->DisplayMode();

    TScreenInfoV01 screenInfo;
    TPckg<TScreenInfoV01> sInfo(screenInfo);
    UserSvr::ScreenInfo(sInfo);

    phdata->NGAGE_ScreenSize = screenInfo.iScreenSize;
    phdata->NGAGE_DisplayMode = displayMode;
    phdata->NGAGE_HasFrameBuffer = screenInfo.iScreenAddressValid;
    phdata->NGAGE_FrameBuffer = phdata->NGAGE_HasFrameBuffer ? (TUint8 *)screenInfo.iScreenAddress : NULL;
    phdata->NGAGE_BytesPerPixel = ((GetBpp(displayMode) - 1) / 8) + 1;

    phdata->NGAGE_BytesPerScanLine = screenInfo.iScreenSize.iWidth * phdata->NGAGE_BytesPerPixel;
    phdata->NGAGE_BytesPerScreen = phdata->NGAGE_BytesPerScanLine * phdata->NGAGE_ScreenSize.iHeight;

    SDL_Log("Screen width        %d", screenInfo.iScreenSize.iWidth);
    SDL_Log("Screen height       %d", screenInfo.iScreenSize.iHeight);
    SDL_Log("Screen dmode        %d", displayMode);
    SDL_Log("Screen valid        %d", screenInfo.iScreenAddressValid);

    SDL_Log("Bytes per pixel     %d", phdata->NGAGE_BytesPerPixel);
    SDL_Log("Bytes per scan line %d", phdata->NGAGE_BytesPerScanLine);
    SDL_Log("Bytes per screen    %d", phdata->NGAGE_BytesPerScreen);

    /* It seems that in SA1100 machines for 8bpp displays there is a 512
     * palette table at the beginning of the frame buffer.
     *
     * In 12 bpp machines the table has 16 entries.
     */
    if (phdata->NGAGE_HasFrameBuffer && GetBpp(displayMode) == 8) {
        phdata->NGAGE_FrameBuffer += 512;
    } else {
        phdata->NGAGE_FrameBuffer += 32;
    }
#if 0
    if (phdata->NGAGE_HasFrameBuffer && GetBpp(displayMode) == 12) {
        phdata->NGAGE_FrameBuffer += 16 * 2;
    }
    if (phdata->NGAGE_HasFrameBuffer && GetBpp(displayMode) == 16) {
        phdata->NGAGE_FrameBuffer += 16 * 2;
    }
#endif

    // Get draw device for updating the screen
    TScreenInfoV01 screenInfo2;

    NGAGE_Runtime::GetScreenInfo(screenInfo2);

    TRAPD(status, phdata->NGAGE_DrawDevice = CFbsDrawDevice::NewScreenDeviceL(screenInfo2, displayMode));
    User::LeaveIfError(status);

    /* Activate events for me */
    phdata->NGAGE_WsEventStatus = KRequestPending;
    phdata->NGAGE_WsSession.EventReady(&phdata->NGAGE_WsEventStatus);

    SDL_Log("SDL:WsEventStatus");
    User::WaitForRequest(phdata->NGAGE_WsEventStatus);

    phdata->NGAGE_RedrawEventStatus = KRequestPending;
    phdata->NGAGE_WsSession.RedrawReady(&phdata->NGAGE_RedrawEventStatus);

    SDL_Log("SDL:RedrawEventStatus");
    User::WaitForRequest(phdata->NGAGE_RedrawEventStatus);

    phdata->NGAGE_WsWindow.PointerFilter(EPointerFilterDrag, 0);

    phdata->NGAGE_ScreenOffset = TPoint(0, 0);

    SDL_Log("SDL:DrawBackground");
    DrawBackground(_this); // Clear screen

    return 0;
}

int SDL_NGAGE_UpdateWindowFramebuffer(_THIS, SDL_Window *window, const SDL_Rect *rects, int numrects)
{
    static int frame_number;
    SDL_Surface *surface;

    surface = (SDL_Surface *)SDL_GetWindowData(window, NGAGE_SURFACE);
    if (!surface) {
        return SDL_SetError("Couldn't find ngage surface for window");
    }

    /* Send the data to the display */
    if (SDL_getenv("SDL_VIDEO_NGAGE_SAVE_FRAMES")) {
        char file[128];
        SDL_snprintf(file, sizeof(file), "SDL_window%d-%8.8d.bmp",
                     (int)SDL_GetWindowID(window), ++frame_number);
        SDL_SaveBMP(surface, file);
    }

    DirectUpdate(_this, numrects, (SDL_Rect *)rects);

    return 0;
}

void SDL_NGAGE_DestroyWindowFramebuffer(_THIS, SDL_Window *window)
{
    SDL_Surface *surface;

    surface = (SDL_Surface *)SDL_SetWindowData(window, NGAGE_SURFACE, NULL);
    SDL_FreeSurface(surface);
}

/*****************************************************************************/
/* Runtime                                                                   */
/*****************************************************************************/

#include <e32svr.h>
#include <hal_data.h>
#include <hal.h>

EXPORT_C void NGAGE_Runtime::GetScreenInfo(TScreenInfoV01 &screenInfo2)
{
    TPckg<TScreenInfoV01> sInfo2(screenInfo2);
    UserSvr::ScreenInfo(sInfo2);
}

/*****************************************************************************/
/* Internal                                                                  */
/*****************************************************************************/

int GetBpp(TDisplayMode displaymode)
{
    return TDisplayModeUtils::NumDisplayModeBitsPerPixel(displaymode);
}

void DrawBackground(_THIS)
{
    SDL_VideoData *phdata = (SDL_VideoData *)_this->driverdata;
    /* Draw background */
    TUint16 *screenBuffer = (TUint16 *)phdata->NGAGE_FrameBuffer;
    /* Draw black background */
    Mem::FillZ(screenBuffer, phdata->NGAGE_BytesPerScreen);
}

void DirectDraw(_THIS, int numrects, SDL_Rect *rects, TUint16 *screenBuffer)
{
    SDL_VideoData *phdata = (SDL_VideoData *)_this->driverdata;
    SDL_Surface *screen = (SDL_Surface *)SDL_GetWindowData(_this->windows, NGAGE_SURFACE);

    TInt i;

    TDisplayMode displayMode = phdata->NGAGE_DisplayMode;
    const TInt sourceNumBytesPerPixel = ((GetBpp(displayMode) - 1) / 8) + 1;

    const TPoint fixedOffset = phdata->NGAGE_ScreenOffset;
    const TInt screenW = screen->w;
    const TInt screenH = screen->h;
    const TInt sourceScanlineLength = screenW;
    const TInt targetScanlineLength = phdata->NGAGE_ScreenSize.iWidth;

    /* Render the rectangles in the list */

    for (i = 0; i < numrects; ++i) {
        const SDL_Rect &currentRect = rects[i];
        SDL_Rect rect2;
        rect2.x = currentRect.x;
        rect2.y = currentRect.y;
        rect2.w = currentRect.w;
        rect2.h = currentRect.h;

        if (rect2.w <= 0 || rect2.h <= 0) /* Sanity check */ {
            continue;
        }

        /* All variables are measured in pixels */

        /* Check rects validity, i.e. upper and lower bounds */
        TInt maxX = Min(screenW - 1, rect2.x + rect2.w - 1);
        TInt maxY = Min(screenH - 1, rect2.y + rect2.h - 1);
        if (maxX < 0 || maxY < 0) /* sanity check */ {
            continue;
        }
        /* Clip from bottom */

        maxY = Min(maxY, phdata->NGAGE_ScreenSize.iHeight - 1);
        /* TODO: Clip from the right side */

        const TInt sourceRectWidth = maxX - rect2.x + 1;
        const TInt sourceRectWidthInBytes = sourceRectWidth * sourceNumBytesPerPixel;
        const TInt sourceRectHeight = maxY - rect2.y + 1;
        const TInt sourceStartOffset = rect2.x + rect2.y * sourceScanlineLength;
        const TUint skipValue = 1; /* 1 = No skip */

        TInt targetStartOffset = fixedOffset.iX + rect2.x + (fixedOffset.iY + rect2.y) * targetScanlineLength;

        switch (screen->format->BitsPerPixel) {
        case 12:
        {
            TUint16 *bitmapLine = (TUint16 *)screen->pixels + sourceStartOffset;
            TUint16 *screenMemory = screenBuffer + targetStartOffset;

            if (skipValue == 1) {
                for (TInt y = 0; y < sourceRectHeight; y++) {
                    Mem::Copy(screenMemory, bitmapLine, sourceRectWidthInBytes);
                    bitmapLine += sourceScanlineLength;
                    screenMemory += targetScanlineLength;
                }
            } else {
                for (TInt y = 0; y < sourceRectHeight; y++) {
                    TUint16 *bitmapPos = bitmapLine;             /* 2 bytes per pixel */
                    TUint16 *screenMemoryLinePos = screenMemory; /* 2 bytes per pixel */
                    for (TInt x = 0; x < sourceRectWidth; x++) {
                        __ASSERT_DEBUG(screenMemory < (screenBuffer + phdata->NGAGE_ScreenSize.iWidth * phdata->NGAGE_ScreenSize.iHeight), User::Panic(_L("SDL"), KErrCorrupt));
                        __ASSERT_DEBUG(screenMemory >= screenBuffer, User::Panic(_L("SDL"), KErrCorrupt));
                        __ASSERT_DEBUG(bitmapLine < ((TUint16 *)screen->pixels + (screen->w * screen->h)), User::Panic(_L("SDL"), KErrCorrupt));
                        __ASSERT_DEBUG(bitmapLine >= (TUint16 *)screen->pixels, User::Panic(_L("SDL"), KErrCorrupt));

                        *screenMemoryLinePos++ = *bitmapPos;
                        bitmapPos += skipValue;
                    }
                    bitmapLine += sourceScanlineLength;
                    screenMemory += targetScanlineLength;
                }
            }
        } break;
        // 256 color paletted mode: 8 bpp --> 12 bpp
        default:
        {
            if (phdata->NGAGE_BytesPerPixel <= 2) {
                TUint8 *bitmapLine = (TUint8 *)screen->pixels + sourceStartOffset;
                TUint16 *screenMemory = screenBuffer + targetStartOffset;

                for (TInt y = 0; y < sourceRectHeight; y++) {
                    TUint8 *bitmapPos = bitmapLine;              /* 1 byte per pixel */
                    TUint16 *screenMemoryLinePos = screenMemory; /* 2 bytes per pixel */
                    /* Convert each pixel from 256 palette to 4k color values */
                    for (TInt x = 0; x < sourceRectWidth; x++) {
                        __ASSERT_DEBUG(screenMemoryLinePos < (screenBuffer + (phdata->NGAGE_ScreenSize.iWidth * phdata->NGAGE_ScreenSize.iHeight)), User::Panic(_L("SDL"), KErrCorrupt));
                        __ASSERT_DEBUG(screenMemoryLinePos >= screenBuffer, User::Panic(_L("SDL"), KErrCorrupt));
                        __ASSERT_DEBUG(bitmapPos < ((TUint8 *)screen->pixels + (screen->w * screen->h)), User::Panic(_L("SDL"), KErrCorrupt));
                        __ASSERT_DEBUG(bitmapPos >= (TUint8 *)screen->pixels, User::Panic(_L("SDL"), KErrCorrupt));
                        *screenMemoryLinePos++ = NGAGE_HWPalette_256_to_Screen[*bitmapPos++];
                    }
                    bitmapLine += sourceScanlineLength;
                    screenMemory += targetScanlineLength;
                }
            } else {
                TUint8 *bitmapLine = (TUint8 *)screen->pixels + sourceStartOffset;
                TUint32 *screenMemory = reinterpret_cast<TUint32 *>(screenBuffer + targetStartOffset);
                for (TInt y = 0; y < sourceRectHeight; y++) {
                    TUint8 *bitmapPos = bitmapLine;              /* 1 byte per pixel */
                    TUint32 *screenMemoryLinePos = screenMemory; /* 2 bytes per pixel */
                    /* Convert each pixel from 256 palette to 4k color values */
                    for (TInt x = 0; x < sourceRectWidth; x++) {
                        __ASSERT_DEBUG(screenMemoryLinePos < (reinterpret_cast<TUint32 *>(screenBuffer) + (phdata->NGAGE_ScreenSize.iWidth * phdata->NGAGE_ScreenSize.iHeight)), User::Panic(_L("SDL"), KErrCorrupt));
                        __ASSERT_DEBUG(screenMemoryLinePos >= reinterpret_cast<TUint32 *>(screenBuffer), User::Panic(_L("SDL"), KErrCorrupt));
                        __ASSERT_DEBUG(bitmapPos < ((TUint8 *)screen->pixels + (screen->w * screen->h)), User::Panic(_L("SDL"), KErrCorrupt));
                        __ASSERT_DEBUG(bitmapPos >= (TUint8 *)screen->pixels, User::Panic(_L("SDL"), KErrCorrupt));
                        *screenMemoryLinePos++ = NGAGE_HWPalette_256_to_Screen[*bitmapPos++];
                    }
                    bitmapLine += sourceScanlineLength;
                    screenMemory += targetScanlineLength;
                }
            }
        }
        }
    }
}

void DirectUpdate(_THIS, int numrects, SDL_Rect *rects)
{
    SDL_VideoData *phdata = (SDL_VideoData *)_this->driverdata;

    if (!phdata->NGAGE_IsWindowFocused) {
        SDL_PauseAudio(1);
        SDL_Delay(1000);
        return;
    }

    SDL_PauseAudio(0);

    TUint16 *screenBuffer = (TUint16 *)phdata->NGAGE_FrameBuffer;

#if 0
    if (phdata->NGAGE_ScreenOrientation == CFbsBitGc::EGraphicsOrientationRotated270) {
        // ...
    } else
#endif
    {
        DirectDraw(_this, numrects, rects, screenBuffer);
    }

    for (int i = 0; i < numrects; ++i) {
        TInt aAx = rects[i].x;
        TInt aAy = rects[i].y;
        TInt aBx = rects[i].w;
        TInt aBy = rects[i].h;
        TRect rect2 = TRect(aAx, aAy, aBx, aBy);

        phdata->NGAGE_DrawDevice->UpdateRegion(rect2); /* Should we update rects parameter area only? */
        phdata->NGAGE_DrawDevice->Update();
    }
}

void RedrawWindowL(_THIS)
{
    SDL_VideoData *phdata = (SDL_VideoData *)_this->driverdata;
    SDL_Surface *screen = (SDL_Surface *)SDL_GetWindowData(_this->windows, NGAGE_SURFACE);

    int w = screen->w;
    int h = screen->h;
    if (phdata->NGAGE_ScreenOrientation == CFbsBitGc::EGraphicsOrientationRotated270) {
        w = screen->h;
        h = screen->w;
    }
    if ((w < phdata->NGAGE_ScreenSize.iWidth) || (h < phdata->NGAGE_ScreenSize.iHeight)) {
        DrawBackground(_this);
    }

    /* Tell the system that something has been drawn */
    TRect rect = TRect(phdata->NGAGE_WsWindow.Size());
    phdata->NGAGE_WsWindow.Invalidate(rect);

    /* Draw current buffer */
    SDL_Rect fullScreen;
    fullScreen.x = 0;
    fullScreen.y = 0;
    fullScreen.w = screen->w;
    fullScreen.h = screen->h;
    DirectUpdate(_this, 1, &fullScreen);
}

#endif /* SDL_VIDEO_DRIVER_NGAGE */

/* vi: set ts=4 sw=4 expandtab: */
