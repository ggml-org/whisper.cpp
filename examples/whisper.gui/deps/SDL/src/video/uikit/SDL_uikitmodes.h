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

#ifndef SDL_uikitmodes_h_
#define SDL_uikitmodes_h_

#include "SDL_uikitvideo.h"

@interface SDL_DisplayData : NSObject

- (instancetype)initWithScreen:(UIScreen*)screen;

@property (nonatomic, strong) UIScreen *uiscreen;
@property (nonatomic) float screenDPI;

@end

@interface SDL_DisplayModeData : NSObject

@property (nonatomic, strong) UIScreenMode *uiscreenmode;

@end

extern SDL_bool UIKit_IsDisplayLandscape(UIScreen *uiscreen);

extern int UIKit_InitModes(_THIS);
extern int UIKit_AddDisplay(UIScreen *uiscreen, SDL_bool send_event);
extern void UIKit_DelDisplay(UIScreen *uiscreen);
extern void UIKit_GetDisplayModes(_THIS, SDL_VideoDisplay * display);
extern int UIKit_GetDisplayDPI(_THIS, SDL_VideoDisplay * display, float * ddpi, float * hdpi, float * vdpi);
extern int UIKit_SetDisplayMode(_THIS, SDL_VideoDisplay * display, SDL_DisplayMode * mode);
extern void UIKit_QuitModes(_THIS);
extern int UIKit_GetDisplayUsableBounds(_THIS, SDL_VideoDisplay * display, SDL_Rect * rect);

#endif /* SDL_uikitmodes_h_ */

/* vi: set ts=4 sw=4 expandtab: */
