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
#ifndef SDL_uikitwindow_h_
#define SDL_uikitwindow_h_

#include "../SDL_sysvideo.h"
#import "SDL_uikitvideo.h"
#import "SDL_uikitview.h"
#import "SDL_uikitviewcontroller.h"

extern int UIKit_CreateWindow(_THIS, SDL_Window * window);
extern void UIKit_SetWindowTitle(_THIS, SDL_Window * window);
extern void UIKit_ShowWindow(_THIS, SDL_Window * window);
extern void UIKit_HideWindow(_THIS, SDL_Window * window);
extern void UIKit_RaiseWindow(_THIS, SDL_Window * window);
extern void UIKit_SetWindowBordered(_THIS, SDL_Window * window, SDL_bool bordered);
extern void UIKit_SetWindowFullscreen(_THIS, SDL_Window * window, SDL_VideoDisplay * display, SDL_bool fullscreen);
extern void UIKit_SetWindowMouseGrab(_THIS, SDL_Window * window, SDL_bool grabbed);
extern void UIKit_UpdatePointerLock(_THIS, SDL_Window * window);
extern void UIKit_DestroyWindow(_THIS, SDL_Window * window);
extern void UIKit_GetWindowSizeInPixels(_THIS, SDL_Window * window, int *w, int *h);
extern SDL_bool UIKit_GetWindowWMInfo(_THIS, SDL_Window * window,
                                      struct SDL_SysWMinfo * info);

extern NSUInteger UIKit_GetSupportedOrientations(SDL_Window * window);

@class UIWindow;

@interface SDL_WindowData : NSObject

@property (nonatomic, strong) UIWindow *uiwindow;
@property (nonatomic, strong) SDL_uikitviewcontroller *viewcontroller;

/* Array of SDL_uikitviews owned by this window. */
@property (nonatomic, copy) NSMutableArray *views;

@end

#endif /* SDL_uikitwindow_h_ */

/* vi: set ts=4 sw=4 expandtab: */
