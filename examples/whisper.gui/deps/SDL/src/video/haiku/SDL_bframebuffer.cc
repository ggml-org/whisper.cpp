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

#ifdef SDL_VIDEO_DRIVER_HAIKU

#include "SDL_bframebuffer.h"

#include <AppKit.h>
#include <InterfaceKit.h>
#include "SDL_bmodes.h"
#include "SDL_BWin.h"

#include "../../main/haiku/SDL_BApp.h"

#ifdef __cplusplus
extern "C" {
#endif

static SDL_INLINE SDL_BWin *_ToBeWin(SDL_Window *window) {
    return (SDL_BWin *)(window->driverdata);
}

static SDL_INLINE SDL_BLooper *_GetBeLooper() {
    return SDL_Looper;
}

int HAIKU_CreateWindowFramebuffer(_THIS, SDL_Window * window,
                                       Uint32 * format,
                                       void ** pixels, int *pitch) {
    SDL_BWin *bwin = _ToBeWin(window);
    BScreen bscreen;
    if (!bscreen.IsValid()) {
        return -1;
    }

    /* Make sure we have exclusive access to frame buffer data */
    bwin->LockBuffer();

    bwin->CreateView();

    /* format */
    display_mode bmode;
    bscreen.GetMode(&bmode);
    *format = HAIKU_ColorSpaceToSDLPxFormat(bmode.space);

    /* Create the new bitmap object */
    BBitmap *bitmap = bwin->GetBitmap();

    if (bitmap) {
        delete bitmap;
    }
    bitmap = new BBitmap(bwin->Bounds(), (color_space)bmode.space,
            false,    /* Views not accepted */
            true);    /* Contiguous memory required */

    if (bitmap->InitCheck() != B_OK) {
        delete bitmap;
        return SDL_SetError("Could not initialize back buffer!");
    }


    bwin->SetBitmap(bitmap);

    /* Set the pixel pointer */
    *pixels = bitmap->Bits();

    /* pitch = width of window, in bytes */
    *pitch = bitmap->BytesPerRow();

    bwin->UnlockBuffer();
    return 0;
}



int HAIKU_UpdateWindowFramebuffer(_THIS, SDL_Window * window,
                                      const SDL_Rect * rects, int numrects) {
    if (!window) {
        return 0;
    }

    SDL_BWin *bwin = _ToBeWin(window);

    bwin->PostMessage(BWIN_UPDATE_FRAMEBUFFER);

    return 0;
}

void HAIKU_DestroyWindowFramebuffer(_THIS, SDL_Window * window) {
    SDL_BWin *bwin = _ToBeWin(window);

    bwin->LockBuffer();

    /* Free and clear the window buffer */
    BBitmap *bitmap = bwin->GetBitmap();
    delete bitmap;
    bwin->SetBitmap(NULL);

    bwin->RemoveView();

    bwin->UnlockBuffer();
}

#ifdef __cplusplus
}
#endif

#endif /* SDL_VIDEO_DRIVER_HAIKU */

/* vi: set ts=4 sw=4 expandtab: */
