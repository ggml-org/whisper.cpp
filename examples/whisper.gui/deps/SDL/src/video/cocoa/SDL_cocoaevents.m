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

#ifdef SDL_VIDEO_DRIVER_COCOA

#include "SDL_timer.h"

#include "SDL_cocoavideo.h"
#include "../../events/SDL_events_c.h"
#include "SDL_hints.h"

/* This define was added in the 10.9 SDK. */
#ifndef kIOPMAssertPreventUserIdleDisplaySleep
#define kIOPMAssertPreventUserIdleDisplaySleep kIOPMAssertionTypePreventUserIdleDisplaySleep
#endif
#ifndef NSAppKitVersionNumber10_8
#define NSAppKitVersionNumber10_8 1187
#endif
#ifndef MAC_OS_X_VERSION_10_12
#define NSEventTypeApplicationDefined NSApplicationDefined
#endif

static SDL_Window *FindSDLWindowForNSWindow(NSWindow *win)
{
    SDL_Window *sdlwindow = NULL;
    SDL_VideoDevice *device = SDL_GetVideoDevice();
    if (device && device->windows) {
        for (sdlwindow = device->windows; sdlwindow; sdlwindow = sdlwindow->next) {
            NSWindow *nswindow = ((__bridge SDL_WindowData *) sdlwindow->driverdata).nswindow;
            if (win == nswindow) {
                return sdlwindow;
            }
        }
    }

    return sdlwindow;
}

@interface SDLApplication : NSApplication

- (void)terminate:(id)sender;
- (void)sendEvent:(NSEvent *)theEvent;

+ (void)registerUserDefaults;

@end

@implementation SDLApplication

// Override terminate to handle Quit and System Shutdown smoothly.
- (void)terminate:(id)sender
{
    SDL_SendQuit();
}

static SDL_bool s_bShouldHandleEventsInSDLApplication = SDL_FALSE;

static void Cocoa_DispatchEvent(NSEvent *theEvent)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();

    switch ([theEvent type]) {
        case NSEventTypeLeftMouseDown:
        case NSEventTypeOtherMouseDown:
        case NSEventTypeRightMouseDown:
        case NSEventTypeLeftMouseUp:
        case NSEventTypeOtherMouseUp:
        case NSEventTypeRightMouseUp:
        case NSEventTypeLeftMouseDragged:
        case NSEventTypeRightMouseDragged:
        case NSEventTypeOtherMouseDragged: /* usually middle mouse dragged */
        case NSEventTypeMouseMoved:
        case NSEventTypeScrollWheel:
            Cocoa_HandleMouseEvent(_this, theEvent);
            break;
        case NSEventTypeKeyDown:
        case NSEventTypeKeyUp:
        case NSEventTypeFlagsChanged:
            Cocoa_HandleKeyEvent(_this, theEvent);
            break;
        default:
            break;
    }
}

// Dispatch events here so that we can handle events caught by
// nextEventMatchingMask in SDL, as well as events caught by other
// processes (such as CEF) that are passed down to NSApp.
- (void)sendEvent:(NSEvent *)theEvent
{
    if (s_bShouldHandleEventsInSDLApplication) {
        Cocoa_DispatchEvent(theEvent);
    }

    [super sendEvent:theEvent];
}

+ (void)registerUserDefaults
{
    NSDictionary *appDefaults = [[NSDictionary alloc] initWithObjectsAndKeys:
                                 [NSNumber numberWithBool:NO], @"AppleMomentumScrollSupported",
                                 [NSNumber numberWithBool:NO], @"ApplePressAndHoldEnabled",
                                 [NSNumber numberWithBool:YES], @"ApplePersistenceIgnoreState",
                                 nil];
    [[NSUserDefaults standardUserDefaults] registerDefaults:appDefaults];
}

@end // SDLApplication

/* setAppleMenu disappeared from the headers in 10.4 */
@interface NSApplication(NSAppleMenu)
- (void)setAppleMenu:(NSMenu *)menu;
@end

@interface SDLAppDelegate : NSObject <NSApplicationDelegate> {
@public
    BOOL seenFirstActivate;
}

- (id)init;
- (void)localeDidChange:(NSNotification *)notification;
- (BOOL)applicationSupportsSecureRestorableState:(NSApplication *)app;
@end

@implementation SDLAppDelegate : NSObject
- (id)init
{
    self = [super init];
    if (self) {
        NSNotificationCenter *center = [NSNotificationCenter defaultCenter];

        seenFirstActivate = NO;

        [center addObserver:self
                   selector:@selector(windowWillClose:)
                       name:NSWindowWillCloseNotification
                     object:nil];

        [center addObserver:self
                   selector:@selector(focusSomeWindow:)
                       name:NSApplicationDidBecomeActiveNotification
                     object:nil];

        [center addObserver:self
                   selector:@selector(localeDidChange:)
                       name:NSCurrentLocaleDidChangeNotification
                     object:nil];
    }

    return self;
}

- (void)dealloc
{
    NSNotificationCenter *center = [NSNotificationCenter defaultCenter];

    [center removeObserver:self name:NSWindowWillCloseNotification object:nil];
    [center removeObserver:self name:NSApplicationDidBecomeActiveNotification object:nil];
    [center removeObserver:self name:NSCurrentLocaleDidChangeNotification object:nil];

    /* Remove our URL event handler only if we set it */
    if ([NSApp delegate] == self) {
        [[NSAppleEventManager sharedAppleEventManager]
            removeEventHandlerForEventClass:kInternetEventClass
                                 andEventID:kAEGetURL];
    }
}

- (void)windowWillClose:(NSNotification *)notification;
{
    NSWindow *win = (NSWindow*)[notification object];

    if (![win isKeyWindow]) {
        return;
    }

    /* Don't do anything if this was not an SDL window that was closed */
    if (FindSDLWindowForNSWindow(win) == NULL) {
        return;
    }

    /* HACK: Make the next window in the z-order key when the key window is
     * closed. The custom event loop and/or windowing code we have seems to
     * prevent the normal behavior: https://bugzilla.libsdl.org/show_bug.cgi?id=1825
     */

    /* +[NSApp orderedWindows] never includes the 'About' window, but we still
     * want to try its list first since the behavior in other apps is to only
     * make the 'About' window key if no other windows are on-screen.
     */
    for (NSWindow *window in [NSApp orderedWindows]) {
        if (window != win && [window canBecomeKeyWindow]) {
            if (![window isOnActiveSpace]) {
                continue;
            }
            [window makeKeyAndOrderFront:self];
            return;
        }
    }

    /* If a window wasn't found above, iterate through all visible windows in
     * the active Space in z-order (including the 'About' window, if it's shown)
     * and make the first one key.
     */
    for (NSNumber *num in [NSWindow windowNumbersWithOptions:0]) {
        NSWindow *window = [NSApp windowWithWindowNumber:[num integerValue]];
        if (window && window != win && [window canBecomeKeyWindow]) {
            [window makeKeyAndOrderFront:self];
            return;
        }
    }
}

- (void)focusSomeWindow:(NSNotification *)aNotification
{
    SDL_VideoDevice *device;
    /* HACK: Ignore the first call. The application gets a
     * applicationDidBecomeActive: a little bit after the first window is
     * created, and if we don't ignore it, a window that has been created with
     * SDL_WINDOW_MINIMIZED will ~immediately be restored.
     */
    if (!seenFirstActivate) {
        seenFirstActivate = YES;
        return;
    }

    /* Don't do anything if the application already has a key window
     * that is not an SDL window.
     */
    if ([NSApp keyWindow] && FindSDLWindowForNSWindow([NSApp keyWindow]) == NULL) {
        return;
    }

    device = SDL_GetVideoDevice();
    if (device && device->windows) {
        SDL_Window *window = device->windows;
        int i;
        for (i = 0; i < device->num_displays; ++i) {
            SDL_Window *fullscreen_window = device->displays[i].fullscreen_window;
            if (fullscreen_window) {
                if (fullscreen_window->flags & SDL_WINDOW_MINIMIZED) {
                    SDL_RestoreWindow(fullscreen_window);
                }
                return;
            }
        }

        if (window->flags & SDL_WINDOW_MINIMIZED) {
            SDL_RestoreWindow(window);
        } else {
            SDL_RaiseWindow(window);
        }
    }
}

- (void)localeDidChange:(NSNotification *)notification;
{
    SDL_SendLocaleChangedEvent();
}

- (BOOL)application:(NSApplication *)theApplication openFile:(NSString *)filename
{
    return (BOOL)SDL_SendDropFile(NULL, [filename UTF8String]) && SDL_SendDropComplete(NULL);
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification
{
    /* The menu bar of SDL apps which don't have the typical .app bundle
     * structure fails to work the first time a window is created (until it's
     * de-focused and re-focused), if this call is in Cocoa_RegisterApp instead
     * of here. https://bugzilla.libsdl.org/show_bug.cgi?id=3051
     */
    if (!SDL_GetHintBoolean(SDL_HINT_MAC_BACKGROUND_APP, SDL_FALSE)) {
        /* Get more aggressive for Catalina: activate the Dock first so we definitely reset all activation state. */
        for (NSRunningApplication *i in [NSRunningApplication runningApplicationsWithBundleIdentifier:@"com.apple.dock"]) {
            [i activateWithOptions:NSApplicationActivateIgnoringOtherApps];
            break;
        }
        SDL_Delay(300);  /* !!! FIXME: this isn't right. */
        [NSApp activateIgnoringOtherApps:YES];
    }

    /* If we call this before NSApp activation, macOS might print a complaint
     * about ApplePersistenceIgnoreState. */
    [SDLApplication registerUserDefaults];
}

- (void)handleURLEvent:(NSAppleEventDescriptor *)event withReplyEvent:(NSAppleEventDescriptor *)replyEvent
{
    NSString* path = [[event paramDescriptorForKeyword:keyDirectObject] stringValue];
    SDL_SendDropFile(NULL, [path UTF8String]);
    SDL_SendDropComplete(NULL);
}

- (BOOL)applicationSupportsSecureRestorableState:(NSApplication *)app
{
    // This just tells Cocoa that we didn't do any custom save state magic for the app,
    // so the system is safe to use NSSecureCoding internally, instead of using unencrypted
    // save states for backwards compatibility. If we don't return YES here, we'll get a
    // warning on the console at startup:
    //
    // ```
    // WARNING: Secure coding is not enabled for restorable state! Enable secure coding by implementing NSApplicationDelegate.applicationSupportsSecureRestorableState: and returning YES.
    // ```
    //
    // More-detailed explanation:
    // https://stackoverflow.com/questions/77283578/sonoma-and-nsapplicationdelegate-applicationsupportssecurerestorablestate/77320845#77320845
    return YES;
}

@end

static SDLAppDelegate *appDelegate = nil;

static NSString *GetApplicationName(void)
{
    NSString *appName;

    /* Determine the application name */
    appName = [[NSBundle mainBundle] objectForInfoDictionaryKey:@"CFBundleDisplayName"];
    if (!appName) {
        appName = [[NSBundle mainBundle] objectForInfoDictionaryKey:@"CFBundleName"];
    }

    if (![appName length]) {
        appName = [[NSProcessInfo processInfo] processName];
    }

    return appName;
}

static bool LoadMainMenuNibIfAvailable(void)
{
#if MAC_OS_X_VERSION_MAX_ALLOWED >= 1080
    NSDictionary *infoDict;
    NSString *mainNibFileName;
    bool success = false;

    if (floor(NSAppKitVersionNumber) < NSAppKitVersionNumber10_8) {
        return false;
    }
    infoDict = [[NSBundle mainBundle] infoDictionary];
    if (infoDict) {
        mainNibFileName = [infoDict valueForKey:@"NSMainNibFile"];

        if (mainNibFileName) {
            success = [[NSBundle mainBundle] loadNibNamed:mainNibFileName owner:[NSApplication sharedApplication] topLevelObjects:nil];
        }
    }

    return success;
#else
    return false;
#endif
}

static void CreateApplicationMenus(void)
{
    NSString *appName;
    NSString *title;
    NSMenu *appleMenu;
    NSMenu *serviceMenu;
    NSMenu *windowMenu;
    NSMenuItem *menuItem;
    NSMenu *mainMenu;

    if (NSApp == nil) {
        return;
    }

    mainMenu = [[NSMenu alloc] init];

    /* Create the main menu bar */
    [NSApp setMainMenu:mainMenu];

    /* Create the application menu */
    appName = GetApplicationName();
    appleMenu = [[NSMenu alloc] initWithTitle:@""];

    /* Add menu items */
    title = [@"About " stringByAppendingString:appName];
    [appleMenu addItemWithTitle:title action:@selector(orderFrontStandardAboutPanel:) keyEquivalent:@""];

    [appleMenu addItem:[NSMenuItem separatorItem]];

    [appleMenu addItemWithTitle:@"Preferences…" action:nil keyEquivalent:@","];

    [appleMenu addItem:[NSMenuItem separatorItem]];

    serviceMenu = [[NSMenu alloc] initWithTitle:@""];
    menuItem = [appleMenu addItemWithTitle:@"Services" action:nil keyEquivalent:@""];
    [menuItem setSubmenu:serviceMenu];

    [NSApp setServicesMenu:serviceMenu];

    [appleMenu addItem:[NSMenuItem separatorItem]];

    title = [@"Hide " stringByAppendingString:appName];
    [appleMenu addItemWithTitle:title action:@selector(hide:) keyEquivalent:@"h"];

    menuItem = (NSMenuItem *)[appleMenu addItemWithTitle:@"Hide Others" action:@selector(hideOtherApplications:) keyEquivalent:@"h"];
    [menuItem setKeyEquivalentModifierMask:(NSEventModifierFlagOption|NSEventModifierFlagCommand)];

    [appleMenu addItemWithTitle:@"Show All" action:@selector(unhideAllApplications:) keyEquivalent:@""];

    [appleMenu addItem:[NSMenuItem separatorItem]];

    title = [@"Quit " stringByAppendingString:appName];
    [appleMenu addItemWithTitle:title action:@selector(terminate:) keyEquivalent:@"q"];

    /* Put menu into the menubar */
    menuItem = [[NSMenuItem alloc] initWithTitle:@"" action:nil keyEquivalent:@""];
    [menuItem setSubmenu:appleMenu];
    [[NSApp mainMenu] addItem:menuItem];

    /* Tell the application object that this is now the application menu */
    [NSApp setAppleMenu:appleMenu];

    /* Create the window menu */
    windowMenu = [[NSMenu alloc] initWithTitle:@"Window"];

    /* Add menu items */
    [windowMenu addItemWithTitle:@"Close" action:@selector(performClose:) keyEquivalent:@"w"];

    [windowMenu addItemWithTitle:@"Minimize" action:@selector(performMiniaturize:) keyEquivalent:@"m"];

    [windowMenu addItemWithTitle:@"Zoom" action:@selector(performZoom:) keyEquivalent:@""];

    /* Add the fullscreen toggle menu option. */
    /* Cocoa should update the title to Enter or Exit Full Screen automatically.
     * But if not, then just fallback to Toggle Full Screen.
     */
    menuItem = [[NSMenuItem alloc] initWithTitle:@"Toggle Full Screen" action:@selector(toggleFullScreen:) keyEquivalent:@"f"];
    [menuItem setKeyEquivalentModifierMask:NSEventModifierFlagControl | NSEventModifierFlagCommand];
    [windowMenu addItem:menuItem];

    /* Put menu into the menubar */
    menuItem = [[NSMenuItem alloc] initWithTitle:@"Window" action:nil keyEquivalent:@""];
    [menuItem setSubmenu:windowMenu];
    [[NSApp mainMenu] addItem:menuItem];

    /* Tell the application object that this is now the window menu */
    [NSApp setWindowsMenu:windowMenu];
}

void Cocoa_RegisterApp(void)
{ @autoreleasepool
{
    /* This can get called more than once! Be careful what you initialize! */

    if (NSApp == nil) {
        [SDLApplication sharedApplication];
        SDL_assert(NSApp != nil);

        s_bShouldHandleEventsInSDLApplication = SDL_TRUE;

        if (!SDL_GetHintBoolean(SDL_HINT_MAC_BACKGROUND_APP, SDL_FALSE)) {
            [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        }

        /* If there aren't already menus in place, look to see if there's
         * a nib we should use. If not, then manually create the basic
         * menus we meed.
         */
        if ([NSApp mainMenu] == nil) {
            bool nibLoaded;

            nibLoaded = LoadMainMenuNibIfAvailable();
            if (!nibLoaded) {
                CreateApplicationMenus();
            }
        }
        [NSApp finishLaunching];
        if ([NSApp delegate]) {
            /* The SDL app delegate calls this in didFinishLaunching if it's
             * attached to the NSApp, otherwise we need to call it manually.
             */
            [SDLApplication registerUserDefaults];
        }
    }
    if (NSApp && !appDelegate) {
        appDelegate = [[SDLAppDelegate alloc] init];

        /* If someone else has an app delegate, it means we can't turn a
         * termination into SDL_Quit, and we can't handle application:openFile:
         */
        if (![NSApp delegate]) {
            /* Only register the URL event handler if we are being set as the
             * app delegate to avoid replacing any existing event handler.
             */
            [[NSAppleEventManager sharedAppleEventManager]
                setEventHandler:appDelegate
                    andSelector:@selector(handleURLEvent:withReplyEvent:)
                  forEventClass:kInternetEventClass
                     andEventID:kAEGetURL];

            [(NSApplication *)NSApp setDelegate:appDelegate];
        } else {
            appDelegate->seenFirstActivate = YES;
        }
    }
}}

int Cocoa_PumpEventsUntilDate(_THIS, NSDate *expiration, bool accumulate)
{
    for ( ; ; ) {
        NSEvent *event = [NSApp nextEventMatchingMask:NSEventMaskAny untilDate:expiration inMode:NSDefaultRunLoopMode dequeue:YES ];
        if ( event == nil ) {
            return 0;
        }

        if (!s_bShouldHandleEventsInSDLApplication) {
            Cocoa_DispatchEvent(event);
        }

        // Pass events down to SDLApplication to be handled in sendEvent:
        [NSApp sendEvent:event];
        if ( !accumulate) {
            break;
        }
    }
    return 1;
}

int Cocoa_WaitEventTimeout(_THIS, int timeout)
{ @autoreleasepool
{
    if (timeout > 0) {
        NSDate *limitDate = [NSDate dateWithTimeIntervalSinceNow: (double) timeout / 1000.0];
        return Cocoa_PumpEventsUntilDate(_this, limitDate, false);
    } else if (timeout == 0) {
        return Cocoa_PumpEventsUntilDate(_this, [NSDate distantPast], false);
    } else {
        while (Cocoa_PumpEventsUntilDate(_this, [NSDate distantFuture], false) == 0) {
        }
    }
    return 1;
}}

void Cocoa_PumpEvents(_THIS)
{ @autoreleasepool
{
    Cocoa_PumpEventsUntilDate(_this, [NSDate distantPast], true);
}}

void Cocoa_SendWakeupEvent(_THIS, SDL_Window *window)
{ @autoreleasepool
{
    NSEvent* event = [NSEvent otherEventWithType: NSEventTypeApplicationDefined
                                    location: NSMakePoint(0,0)
                               modifierFlags: 0
                                   timestamp: 0.0
                                windowNumber: ((__bridge SDL_WindowData *) window->driverdata).window_number
                                     context: nil
                                     subtype: 0
                                       data1: 0
                                       data2: 0];

    [NSApp postEvent: event atStart: YES];
}}

void Cocoa_SuspendScreenSaver(_THIS)
{ @autoreleasepool
{
    SDL_VideoData *data = (__bridge SDL_VideoData *)_this->driverdata;

    if (data.screensaver_assertion) {
        IOPMAssertionRelease(data.screensaver_assertion);
        data.screensaver_assertion = kIOPMNullAssertionID;
    }

    if (_this->suspend_screensaver) {
        /* FIXME: this should ideally describe the real reason why the game
         * called SDL_DisableScreenSaver. Note that the name is only meant to be
         * seen by OS X power users. there's an additional optional human-readable
         * (localized) reason parameter which we don't set.
         */
        IOPMAssertionID assertion = kIOPMNullAssertionID;
        NSString *name = [GetApplicationName() stringByAppendingString:@" using SDL_DisableScreenSaver"];
        IOPMAssertionCreateWithDescription(kIOPMAssertPreventUserIdleDisplaySleep,
                                           (__bridge CFStringRef) name,
                                           NULL, NULL, NULL, 0, NULL,
                                           &assertion);
        data.screensaver_assertion = assertion;
    }
}}

#endif /* SDL_VIDEO_DRIVER_COCOA */

/* vi: set ts=4 sw=4 expandtab: */
