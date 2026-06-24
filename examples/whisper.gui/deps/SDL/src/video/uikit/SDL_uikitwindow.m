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

#ifdef SDL_VIDEO_DRIVER_UIKIT

#include "SDL_hints.h"
#include "SDL_mouse.h"
#include "SDL_system.h"
#include "SDL_syswm.h"
#include "SDL_video.h"
#include "../SDL_sysvideo.h"
#include "../SDL_pixels_c.h"
#include "../../events/SDL_events_c.h"

#include "SDL_uikitvideo.h"
#include "SDL_uikitevents.h"
#include "SDL_uikitmodes.h"
#include "SDL_uikitwindow.h"
#import "SDL_uikitappdelegate.h"

#import "SDL_uikitview.h"
#import "SDL_uikitopenglview.h"

#include <Foundation/Foundation.h>

@implementation SDL_WindowData

@synthesize uiwindow;
@synthesize viewcontroller;
@synthesize views;

- (instancetype)init
{
    if ((self = [super init])) {
        views = [NSMutableArray new];
    }

    return self;
}

@end

@interface SDL_uikitwindow : UIWindow

- (void)layoutSubviews;

@end

@implementation SDL_uikitwindow

- (void)layoutSubviews
{
    /* Workaround to fix window orientation issues in iOS 8. */
    /* As of July 1 2019, I haven't been able to reproduce any orientation
     * issues with this disabled on iOS 12. The issue this is meant to fix might
     * only happen on iOS 8, or it might have been fixed another way with other
     * code... This code prevents split view (iOS 9+) from working on iPads, so
     * we want to avoid using it if possible. */
    if (!UIKit_IsSystemVersionAtLeast(9.0)) {
        self.frame = self.screen.bounds;
    }
    [super layoutSubviews];
}

@end


static int SetupWindowData(_THIS, SDL_Window *window, UIWindow *uiwindow, SDL_bool created)
{
    SDL_VideoDisplay *display = SDL_GetDisplayForWindow(window);
    SDL_DisplayData *displaydata = (__bridge SDL_DisplayData *) display->driverdata;
    SDL_uikitview *view;

    CGRect frame = UIKit_ComputeViewFrame(window, displaydata.uiscreen);
    int width  = (int) frame.size.width;
    int height = (int) frame.size.height;

    SDL_WindowData *data = [[SDL_WindowData alloc] init];
    if (!data) {
        return SDL_OutOfMemory();
    }

    window->driverdata = (void *) CFBridgingRetain(data);

    data.uiwindow = uiwindow;

    /* only one window on iOS, always shown */
    window->flags &= ~SDL_WINDOW_HIDDEN;

    if (displaydata.uiscreen != [UIScreen mainScreen]) {
        window->flags &= ~SDL_WINDOW_RESIZABLE;  /* window is NEVER resizable */
        window->flags &= ~SDL_WINDOW_INPUT_FOCUS;  /* never has input focus */
        window->flags |= SDL_WINDOW_BORDERLESS;  /* never has a status bar. */
    }

#if !TARGET_OS_TV
    if (displaydata.uiscreen == [UIScreen mainScreen]) {
        NSUInteger orients = UIKit_GetSupportedOrientations(window);
        BOOL supportsLandscape = (orients & UIInterfaceOrientationMaskLandscape) != 0;
        BOOL supportsPortrait = (orients & (UIInterfaceOrientationMaskPortrait|UIInterfaceOrientationMaskPortraitUpsideDown)) != 0;

        /* Make sure the width/height are oriented correctly */
        if ((width > height && !supportsLandscape) || (height > width && !supportsPortrait)) {
            int temp = width;
            width = height;
            height = temp;
        }
    }
#endif /* !TARGET_OS_TV */

#if 0 /* Don't set the x/y position, it's already placed on a display */
    window->x = 0;
    window->y = 0;
#endif
    window->w = width;
    window->h = height;

    /* The View Controller will handle rotating the view when the device
     * orientation changes. This will trigger resize events, if appropriate. */
    data.viewcontroller = [[SDL_uikitviewcontroller alloc] initWithSDLWindow:window];

    /* The window will initially contain a generic view so resizes, touch events,
     * etc. can be handled without an active OpenGL view/context. */
    view = [[SDL_uikitview alloc] initWithFrame:frame];

    /* Sets this view as the controller's view, and adds the view to the window
     * heirarchy. */
    [view setSDLWindow:window];

    return 0;
}

int UIKit_CreateWindow(_THIS, SDL_Window *window)
{
    @autoreleasepool {
        SDL_VideoDisplay *display = SDL_GetDisplayForWindow(window);
        SDL_DisplayData *data = (__bridge SDL_DisplayData *) display->driverdata;
        SDL_Window *other;
        UIWindow *uiwindow;
#if !TARGET_OS_TV
        const CGSize origsize = data.uiscreen.currentMode.size;
#endif

        /* We currently only handle a single window per display on iOS */
        for (other = _this->windows; other; other = other->next) {
            if (other != window && SDL_GetDisplayForWindow(other) == display) {
                return SDL_SetError("Only one window allowed per display.");
            }
        }

        /* If monitor has a resolution of 0x0 (hasn't been explicitly set by the
         * user, so it's in standby), try to force the display to a resolution
         * that most closely matches the desired window size. */
#if !TARGET_OS_TV
        if ((origsize.width == 0.0f) && (origsize.height == 0.0f)) {
            int i;
            const SDL_DisplayMode *bestmode = NULL;
            if (display->num_display_modes == 0) {
                _this->GetDisplayModes(_this, display);
            }

            for (i = display->num_display_modes; i >= 0; i--) {
                const SDL_DisplayMode *mode = &display->display_modes[i];
                if ((mode->w >= window->w) && (mode->h >= window->h)) {
                    bestmode = mode;
                }
            }

            if (bestmode) {
                SDL_DisplayModeData *modedata = (__bridge SDL_DisplayModeData *)bestmode->driverdata;
                [data.uiscreen setCurrentMode:modedata.uiscreenmode];

                /* desktop_mode doesn't change here (the higher level will
                 * use it to set all the screens back to their defaults
                 * upon window destruction, SDL_Quit(), etc. */
                display->current_mode = *bestmode;
            }
        }

        if (data.uiscreen == [UIScreen mainScreen]) {
            if (window->flags & (SDL_WINDOW_FULLSCREEN|SDL_WINDOW_BORDERLESS)) {
                [UIApplication sharedApplication].statusBarHidden = YES;
            } else {
                [UIApplication sharedApplication].statusBarHidden = NO;
            }
        }
#endif /* !TARGET_OS_TV */

        /* ignore the size user requested, and make a fullscreen window */
        /* !!! FIXME: can we have a smaller view? */
        uiwindow = [[SDL_uikitwindow alloc] initWithFrame:data.uiscreen.bounds];

        /* put the window on an external display if appropriate. */
        if (data.uiscreen != [UIScreen mainScreen]) {
            [uiwindow setScreen:data.uiscreen];
        }

        if (SetupWindowData(_this, window, uiwindow, SDL_TRUE) < 0) {
            return -1;
        }
    }

    return 1;
}

void UIKit_SetWindowTitle(_THIS, SDL_Window * window)
{
    @autoreleasepool {
        SDL_WindowData *data = (__bridge SDL_WindowData *) window->driverdata;
        data.viewcontroller.title = @(window->title);
    }
}

void UIKit_ShowWindow(_THIS, SDL_Window * window)
{
    @autoreleasepool {
        SDL_VideoDisplay *display;
        SDL_DisplayData *displaydata;
        SDL_WindowData *data = (__bridge SDL_WindowData *) window->driverdata;
        [data.uiwindow makeKeyAndVisible];

        /* Make this window the current mouse focus for touch input */
        display = SDL_GetDisplayForWindow(window);
        displaydata = (__bridge SDL_DisplayData *) display->driverdata;
        if (displaydata.uiscreen == [UIScreen mainScreen]) {
            SDL_SetMouseFocus(window);
            SDL_SetKeyboardFocus(window);
        }
    }
}

void UIKit_HideWindow(_THIS, SDL_Window * window)
{
    @autoreleasepool {
        SDL_WindowData *data = (__bridge SDL_WindowData *) window->driverdata;
        data.uiwindow.hidden = YES;
    }
}

void UIKit_RaiseWindow(_THIS, SDL_Window * window)
{
    /* We don't currently offer a concept of "raising" the SDL window, since
     * we only allow one per display, in the iOS fashion.
     * However, we use this entry point to rebind the context to the view
     * during OnWindowRestored processing. */
    _this->GL_MakeCurrent(_this, _this->current_glwin, _this->current_glctx);
}

static void UIKit_UpdateWindowBorder(_THIS, SDL_Window * window)
{
    SDL_WindowData *data = (__bridge SDL_WindowData *) window->driverdata;
    SDL_uikitviewcontroller *viewcontroller = data.viewcontroller;

#if !TARGET_OS_TV
    if (data.uiwindow.screen == [UIScreen mainScreen]) {
        if (window->flags & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_BORDERLESS)) {
            [UIApplication sharedApplication].statusBarHidden = YES;
        } else {
            [UIApplication sharedApplication].statusBarHidden = NO;
        }

        [viewcontroller setNeedsStatusBarAppearanceUpdate];
    }

    /* Update the view's frame to account for the status bar change. */
    viewcontroller.view.frame = UIKit_ComputeViewFrame(window, data.uiwindow.screen);
#endif /* !TARGET_OS_TV */

#ifdef SDL_IPHONE_KEYBOARD
    /* Make sure the view is offset correctly when the keyboard is visible. */
    [viewcontroller updateKeyboard];
#endif

    [viewcontroller.view setNeedsLayout];
    [viewcontroller.view layoutIfNeeded];
}

void UIKit_SetWindowBordered(_THIS, SDL_Window * window, SDL_bool bordered)
{
    @autoreleasepool {
        UIKit_UpdateWindowBorder(_this, window);
    }
}

void UIKit_SetWindowFullscreen(_THIS, SDL_Window * window, SDL_VideoDisplay * display, SDL_bool fullscreen)
{
    @autoreleasepool {
        UIKit_UpdateWindowBorder(_this, window);
    }
}

void UIKit_SetWindowMouseGrab(_THIS, SDL_Window * window, SDL_bool grabbed)
{
    /* There really isn't a concept of window grab or cursor confinement on iOS */
}

void UIKit_UpdatePointerLock(_THIS, SDL_Window * window)
{
#if !TARGET_OS_TV
#if defined(__IPHONE_14_0) && __IPHONE_OS_VERSION_MAX_ALLOWED >= __IPHONE_14_0
    @autoreleasepool {
        SDL_WindowData *data = (__bridge SDL_WindowData *) window->driverdata;
        SDL_uikitviewcontroller *viewcontroller = data.viewcontroller;
        if (@available(iOS 14.0, *)) {
            [viewcontroller setNeedsUpdateOfPrefersPointerLocked];
        }
    }
#endif /* defined(__IPHONE_14_0) && __IPHONE_OS_VERSION_MAX_ALLOWED >= __IPHONE_14_0 */
#endif /* !TARGET_OS_TV */
}

void UIKit_DestroyWindow(_THIS, SDL_Window * window)
{
    @autoreleasepool {
        if (window->driverdata != NULL) {
            SDL_WindowData *data = (__bridge SDL_WindowData *)window->driverdata;
            NSArray *views = nil;

            [data.viewcontroller stopAnimation];

            /* Detach all views from this window. We use a copy of the array
             * because setSDLWindow will remove the object from the original
             * array, which would be undesirable if we were iterating over it. */
            views = [data.views copy];
            for (SDL_uikitview *view in views) {
                [view setSDLWindow:NULL];
            }

            /* iOS may still hold a reference to the window after we release it.
             * We want to make sure the SDL view controller isn't accessed in
             * that case, because it would contain an invalid pointer to the old
             * SDL window. */
            data.uiwindow.rootViewController = nil;
            data.uiwindow.hidden = YES;

            CFRelease(window->driverdata);
            window->driverdata = NULL;
        }
    }
}

void UIKit_GetWindowSizeInPixels(_THIS, SDL_Window * window, int *w, int *h)
{ @autoreleasepool
{
    SDL_WindowData *windata = (__bridge SDL_WindowData *) window->driverdata;
    UIView *view = windata.viewcontroller.view;
    CGSize size = view.bounds.size;
    CGFloat scale = 1.0;

    if (window->flags & SDL_WINDOW_ALLOW_HIGHDPI) {
        scale = windata.uiwindow.screen.nativeScale;
    }

    /* Integer truncation of fractional values matches SDL_uikitmetalview and
     * SDL_uikitopenglview. */
    *w = size.width * scale;
    *h = size.height * scale;
}}

SDL_bool UIKit_GetWindowWMInfo(_THIS, SDL_Window * window, SDL_SysWMinfo * info)
{
    @autoreleasepool {
        SDL_WindowData *data = (__bridge SDL_WindowData *) window->driverdata;

        if (info->version.major <= SDL_MAJOR_VERSION) {
            int versionnum = SDL_VERSIONNUM(info->version.major, info->version.minor, info->version.patch);

            info->subsystem = SDL_SYSWM_UIKIT;
            info->info.uikit.window = data.uiwindow;

            /* These struct members were added in SDL 2.0.4. */
            if (versionnum >= SDL_VERSIONNUM(2,0,4)) {
#if defined(SDL_VIDEO_OPENGL_ES) || defined(SDL_VIDEO_OPENGL_ES2)
                if ([data.viewcontroller.view isKindOfClass:[SDL_uikitopenglview class]]) {
                    SDL_uikitopenglview *glview = (SDL_uikitopenglview *)data.viewcontroller.view;
                    info->info.uikit.framebuffer = glview.drawableFramebuffer;
                    info->info.uikit.colorbuffer = glview.drawableRenderbuffer;
                    info->info.uikit.resolveFramebuffer = glview.msaaResolveFramebuffer;
                } else {
#else
                {
#endif
                    info->info.uikit.framebuffer = 0;
                    info->info.uikit.colorbuffer = 0;
                    info->info.uikit.resolveFramebuffer = 0;
                }
            }

            return SDL_TRUE;
        } else {
            SDL_SetError("Application not compiled with SDL %d",
                         SDL_MAJOR_VERSION);
            return SDL_FALSE;
        }
    }
}

#if !TARGET_OS_TV
NSUInteger UIKit_GetSupportedOrientations(SDL_Window * window)
{
    const char *hint = SDL_GetHint(SDL_HINT_ORIENTATIONS);
    NSUInteger validOrientations = UIInterfaceOrientationMaskAll;
    NSUInteger orientationMask = 0;

    @autoreleasepool {
        SDL_WindowData *data = (__bridge SDL_WindowData *) window->driverdata;
        UIApplication *app = [UIApplication sharedApplication];

        /* Get all possible valid orientations. If the app delegate doesn't tell
         * us, we get the orientations from Info.plist via UIApplication. */
        if ([app.delegate respondsToSelector:@selector(application:supportedInterfaceOrientationsForWindow:)]) {
            validOrientations = [app.delegate application:app supportedInterfaceOrientationsForWindow:data.uiwindow];
        } else {
            validOrientations = [app supportedInterfaceOrientationsForWindow:data.uiwindow];
        }

        if (hint != NULL) {
            NSArray *orientations = [@(hint) componentsSeparatedByString:@" "];

            if ([orientations containsObject:@"LandscapeLeft"]) {
                orientationMask |= UIInterfaceOrientationMaskLandscapeLeft;
            }
            if ([orientations containsObject:@"LandscapeRight"]) {
                orientationMask |= UIInterfaceOrientationMaskLandscapeRight;
            }
            if ([orientations containsObject:@"Portrait"]) {
                orientationMask |= UIInterfaceOrientationMaskPortrait;
            }
            if ([orientations containsObject:@"PortraitUpsideDown"]) {
                orientationMask |= UIInterfaceOrientationMaskPortraitUpsideDown;
            }
        }

        if (orientationMask == 0 && (window->flags & SDL_WINDOW_RESIZABLE)) {
            /* any orientation is okay. */
            orientationMask = UIInterfaceOrientationMaskAll;
        }

        if (orientationMask == 0) {
            if (window->windowed.w >= window->windowed.h) {
                orientationMask |= UIInterfaceOrientationMaskLandscape;
            }
            if (window->windowed.h >= window->windowed.w) {
                orientationMask |= (UIInterfaceOrientationMaskPortrait | UIInterfaceOrientationMaskPortraitUpsideDown);
            }
        }

        /* Don't allow upside-down orientation on phones, so answering calls is in the natural orientation */
        if ([UIDevice currentDevice].userInterfaceIdiom == UIUserInterfaceIdiomPhone) {
            orientationMask &= ~UIInterfaceOrientationMaskPortraitUpsideDown;
        }

        /* If none of the specified orientations are actually supported by the
         * app, we'll revert to what the app supports. An exception would be
         * thrown by the system otherwise. */
        if ((validOrientations & orientationMask) == 0) {
            orientationMask = validOrientations;
        }
    }

    return orientationMask;
}
#endif /* !TARGET_OS_TV */

int SDL_iPhoneSetAnimationCallback(SDL_Window * window, int interval, void (*callback)(void*), void *callbackParam)
{
    if (!window || !window->driverdata) {
        return SDL_SetError("Invalid window");
    }

    @autoreleasepool {
        SDL_WindowData *data = (__bridge SDL_WindowData *)window->driverdata;
        [data.viewcontroller setAnimationCallback:interval
                                         callback:callback
                                    callbackParam:callbackParam];
    }

    return 0;
}

#endif /* SDL_VIDEO_DRIVER_UIKIT */

/* vi: set ts=4 sw=4 expandtab: */
