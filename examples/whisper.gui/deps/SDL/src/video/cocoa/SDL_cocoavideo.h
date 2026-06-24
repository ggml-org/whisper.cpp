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

#ifndef SDL_cocoavideo_h_
#define SDL_cocoavideo_h_

#include "SDL_opengl.h"

#include <ApplicationServices/ApplicationServices.h>
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <Cocoa/Cocoa.h>

#include "SDL_keycode.h"
#include "../SDL_sysvideo.h"

#include "SDL_cocoaclipboard.h"
#include "SDL_cocoaevents.h"
#include "SDL_cocoakeyboard.h"
#include "SDL_cocoamodes.h"
#include "SDL_cocoamouse.h"
#include "SDL_cocoaopengl.h"
#include "SDL_cocoawindow.h"

#ifndef MAC_OS_X_VERSION_10_12
#define DECLARE_EVENT(name) static const NSEventType NSEventType##name = NS##name
DECLARE_EVENT(LeftMouseDown);
DECLARE_EVENT(LeftMouseUp);
DECLARE_EVENT(RightMouseDown);
DECLARE_EVENT(RightMouseUp);
DECLARE_EVENT(OtherMouseDown);
DECLARE_EVENT(OtherMouseUp);
DECLARE_EVENT(MouseMoved);
DECLARE_EVENT(LeftMouseDragged);
DECLARE_EVENT(RightMouseDragged);
DECLARE_EVENT(OtherMouseDragged);
DECLARE_EVENT(ScrollWheel);
DECLARE_EVENT(KeyDown);
DECLARE_EVENT(KeyUp);
DECLARE_EVENT(FlagsChanged);
#undef DECLARE_EVENT

static const NSEventMask NSEventMaskAny = NSAnyEventMask;

#define DECLARE_MODIFIER_FLAG(name) static const NSUInteger NSEventModifierFlag##name = NS##name##KeyMask
DECLARE_MODIFIER_FLAG(Shift);
DECLARE_MODIFIER_FLAG(Control);
DECLARE_MODIFIER_FLAG(Command);
DECLARE_MODIFIER_FLAG(NumericPad);
DECLARE_MODIFIER_FLAG(Help);
DECLARE_MODIFIER_FLAG(Function);
#undef DECLARE_MODIFIER_FLAG
static const NSUInteger NSEventModifierFlagCapsLock = NSAlphaShiftKeyMask;
static const NSUInteger NSEventModifierFlagOption = NSAlternateKeyMask;

#define DECLARE_WINDOW_MASK(name) static const unsigned int NSWindowStyleMask##name = NS##name##WindowMask
DECLARE_WINDOW_MASK(Borderless);
DECLARE_WINDOW_MASK(Titled);
DECLARE_WINDOW_MASK(Closable);
DECLARE_WINDOW_MASK(Miniaturizable);
DECLARE_WINDOW_MASK(Resizable);
DECLARE_WINDOW_MASK(TexturedBackground);
DECLARE_WINDOW_MASK(UnifiedTitleAndToolbar);
DECLARE_WINDOW_MASK(FullScreen);
/*DECLARE_WINDOW_MASK(FullSizeContentView);*/ /* Not used, fails compile on older SDKs */
static const unsigned int NSWindowStyleMaskUtilityWindow = NSUtilityWindowMask;
static const unsigned int NSWindowStyleMaskDocModalWindow = NSDocModalWindowMask;
static const unsigned int NSWindowStyleMaskHUDWindow = NSHUDWindowMask;
#undef DECLARE_WINDOW_MASK

#define DECLARE_ALERT_STYLE(name) static const NSUInteger NSAlertStyle##name = NS##name##AlertStyle
DECLARE_ALERT_STYLE(Warning);
DECLARE_ALERT_STYLE(Informational);
DECLARE_ALERT_STYLE(Critical);
#undef DECLARE_ALERT_STYLE
#endif

/* Private display data */

@class SDLTranslatorResponder;

@interface SDL_VideoData : NSObject
    @property (nonatomic) int allow_spaces;
    @property (nonatomic) int trackpad_is_touch_only;
    @property (nonatomic) unsigned int modifierFlags;
    @property (nonatomic) void *key_layout;
    @property (nonatomic) SDLTranslatorResponder *fieldEdit;
    @property (nonatomic) NSInteger clipboard_count;
    @property (nonatomic) IOPMAssertionID screensaver_assertion;
    @property (nonatomic) SDL_mutex *swaplock;
@end

/* Utility functions */
extern NSImage * Cocoa_CreateImage(SDL_Surface * surface);

/* Fix build with the 10.11 SDK */
#if MAC_OS_X_VERSION_MAX_ALLOWED < 101200
#define NSEventSubtypeMouseEvent NSMouseEventSubtype
#endif

#endif /* SDL_cocoavideo_h_ */

/* vi: set ts=4 sw=4 expandtab: */
