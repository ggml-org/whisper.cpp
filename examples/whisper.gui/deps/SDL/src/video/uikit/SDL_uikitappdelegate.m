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

#include "../SDL_sysvideo.h"
#include "SDL_hints.h"
#include "SDL_system.h"
#include "SDL_main.h"

#import "SDL_uikitappdelegate.h"
#import "SDL_uikitmodes.h"
#import "SDL_uikitwindow.h"

#include "../../events/SDL_events_c.h"

#if !TARGET_OS_TV
#include <AvailabilityVersions.h>

# ifndef __IPHONE_13_0
# define __IPHONE_13_0 130000
# endif
#endif

#ifdef main
#undef main
#endif

static SDL_main_func forward_main;
static int forward_argc;
static char **forward_argv;
static int exit_status;

int SDL_UIKitRunApp(int argc, char *argv[], SDL_main_func mainFunction)
{
    int i;

    /* store arguments */
    forward_main = mainFunction;
    forward_argc = argc;
    forward_argv = (char **)malloc((argc+1) * sizeof(char *));
    for (i = 0; i < argc; i++) {
        forward_argv[i] = malloc( (strlen(argv[i])+1) * sizeof(char));
        strcpy(forward_argv[i], argv[i]);
    }
    forward_argv[i] = NULL;

    /* Give over control to run loop, SDLUIKitDelegate will handle most things from here */
    @autoreleasepool {
        UIApplicationMain(argc, argv, nil, [SDLUIKitDelegate getAppDelegateClassName]);
    }

    /* free the memory we used to hold copies of argc and argv */
    for (i = 0; i < forward_argc; i++) {
        free(forward_argv[i]);
    }
    free(forward_argv);

    return exit_status;
}

static void SDLCALL
SDL_IdleTimerDisabledChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    BOOL disable = (hint && *hint != '0');
    [UIApplication sharedApplication].idleTimerDisabled = disable;
}

#if !TARGET_OS_TV
/* Load a launch image using the old UILaunchImageFile-era naming rules. */
static UIImage *SDL_LoadLaunchImageNamed(NSString *name, int screenh)
{
    UIInterfaceOrientation curorient = [UIApplication sharedApplication].statusBarOrientation;
    UIUserInterfaceIdiom idiom = [UIDevice currentDevice].userInterfaceIdiom;
    UIImage *image = nil;

    if (idiom == UIUserInterfaceIdiomPhone && screenh == 568) {
        /* The image name for the iPhone 5 uses its height as a suffix. */
        image = [UIImage imageNamed:[NSString stringWithFormat:@"%@-568h", name]];
    } else if (idiom == UIUserInterfaceIdiomPad) {
        /* iPad apps can launch in any orientation. */
        if (UIInterfaceOrientationIsLandscape(curorient)) {
            if (curorient == UIInterfaceOrientationLandscapeLeft) {
                image = [UIImage imageNamed:[NSString stringWithFormat:@"%@-LandscapeLeft", name]];
            } else {
                image = [UIImage imageNamed:[NSString stringWithFormat:@"%@-LandscapeRight", name]];
            }
            if (!image) {
                image = [UIImage imageNamed:[NSString stringWithFormat:@"%@-Landscape", name]];
            }
        } else {
            if (curorient == UIInterfaceOrientationPortraitUpsideDown) {
                image = [UIImage imageNamed:[NSString stringWithFormat:@"%@-PortraitUpsideDown", name]];
            }
            if (!image) {
                image = [UIImage imageNamed:[NSString stringWithFormat:@"%@-Portrait", name]];
            }
        }
    }

    if (!image) {
        image = [UIImage imageNamed:name];
    }

    return image;
}

@interface SDLLaunchStoryboardViewController : UIViewController
@property (nonatomic, strong) UIViewController *storyboardViewController;
- (instancetype)initWithStoryboardViewController:(UIViewController *)storyboardViewController;
@end

@implementation SDLLaunchStoryboardViewController

- (instancetype)initWithStoryboardViewController:(UIViewController *)storyboardViewController {
    self = [super init];
    self.storyboardViewController = storyboardViewController;
    return self;
}

- (void)viewDidLoad {
    [super viewDidLoad];

    [self addChildViewController:self.storyboardViewController];
    [self.view addSubview:self.storyboardViewController.view];
    self.storyboardViewController.view.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    self.storyboardViewController.view.frame = self.view.bounds;
    [self.storyboardViewController didMoveToParentViewController:self];

    UIApplication.sharedApplication.statusBarHidden = self.prefersStatusBarHidden;
    UIApplication.sharedApplication.statusBarStyle = self.preferredStatusBarStyle;
}

- (BOOL)prefersStatusBarHidden {
    return [[NSBundle.mainBundle objectForInfoDictionaryKey:@"UIStatusBarHidden"] boolValue];
}

- (UIStatusBarStyle)preferredStatusBarStyle {
    NSString *statusBarStyle = [NSBundle.mainBundle objectForInfoDictionaryKey:@"UIStatusBarStyle"];
    if ([statusBarStyle isEqualToString:@"UIStatusBarStyleLightContent"]) {
        return UIStatusBarStyleLightContent;
    }
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= __IPHONE_13_0
    if (@available(iOS 13.0, *)) {
        if ([statusBarStyle isEqualToString:@"UIStatusBarStyleDarkContent"]) {
            return UIStatusBarStyleDarkContent;
        }
    }
#endif
    return UIStatusBarStyleDefault;
}

@end
#endif /* !TARGET_OS_TV */

@interface SDLLaunchScreenController ()

#if !TARGET_OS_TV
- (NSUInteger)supportedInterfaceOrientations;
#endif

@end

@implementation SDLLaunchScreenController

- (instancetype)init
{
    return [self initWithNibName:nil bundle:[NSBundle mainBundle]];
}

- (instancetype)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil
{
    NSString *screenname;
    NSBundle *bundle;
    if (!(self = [super initWithNibName:nil bundle:nil])) {
        return nil;
    }

    screenname = nibNameOrNil;
    bundle = nibBundleOrNil;

    /* A launch screen may not exist. Fall back to launch images in that case. */
    if (screenname) {
        @try {
            self.view = [bundle loadNibNamed:screenname owner:self options:nil][0];
        }
        @catch (NSException *exception) {
            /* If a launch screen name is specified but it fails to load, iOS
             * displays a blank screen rather than falling back to an image. */
            return nil;
        }
    }

    if (!self.view) {
        NSArray *launchimages = [bundle objectForInfoDictionaryKey:@"UILaunchImages"];
        NSString *imagename = nil;
        UIImage *image = nil;

        int screenw = (int)([UIScreen mainScreen].bounds.size.width + 0.5);
        int screenh = (int)([UIScreen mainScreen].bounds.size.height + 0.5);

#if !TARGET_OS_TV
        UIInterfaceOrientation curorient = [UIApplication sharedApplication].statusBarOrientation;

        /* We always want portrait-oriented size, to match UILaunchImageSize. */
        if (screenw > screenh) {
            int width = screenw;
            screenw = screenh;
            screenh = width;
        }
#endif

        /* Xcode 5 introduced a dictionary of launch images in Info.plist. */
        if (launchimages) {
            for (NSDictionary *dict in launchimages) {
#if !TARGET_OS_TV
                UIInterfaceOrientationMask orientmask;
                NSString *orientstring;
#endif
                NSString *minversion = dict[@"UILaunchImageMinimumOSVersion"];
                NSString *sizestring = dict[@"UILaunchImageSize"];

                /* Ignore this image if the current version is too low. */
                if (minversion && !UIKit_IsSystemVersionAtLeast(minversion.doubleValue)) {
                    continue;
                }

                /* Ignore this image if the size doesn't match. */
                if (sizestring) {
                    CGSize size = CGSizeFromString(sizestring);
                    if ((int)(size.width + 0.5) != screenw || (int)(size.height + 0.5) != screenh) {
                        continue;
                    }
                }

#if !TARGET_OS_TV
                orientmask = UIInterfaceOrientationMaskPortrait | UIInterfaceOrientationMaskPortraitUpsideDown;
                orientstring = dict[@"UILaunchImageOrientation"];

                if (orientstring) {
                    if ([orientstring isEqualToString:@"PortraitUpsideDown"]) {
                        orientmask = UIInterfaceOrientationMaskPortraitUpsideDown;
                    } else if ([orientstring isEqualToString:@"Landscape"]) {
                        orientmask = UIInterfaceOrientationMaskLandscape;
                    } else if ([orientstring isEqualToString:@"LandscapeLeft"]) {
                        orientmask = UIInterfaceOrientationMaskLandscapeLeft;
                    } else if ([orientstring isEqualToString:@"LandscapeRight"]) {
                        orientmask = UIInterfaceOrientationMaskLandscapeRight;
                    }
                }

                /* Ignore this image if the orientation doesn't match. */
                if ((orientmask & (1 << curorient)) == 0) {
                    continue;
                }
#endif

                imagename = dict[@"UILaunchImageName"];
            }

            if (imagename) {
                image = [UIImage imageNamed:imagename];
            }
        }
#if !TARGET_OS_TV
        else {
            imagename = [bundle objectForInfoDictionaryKey:@"UILaunchImageFile"];

            if (imagename) {
                image = SDL_LoadLaunchImageNamed(imagename, screenh);
            }

            if (!image) {
                image = SDL_LoadLaunchImageNamed(@"Default", screenh);
            }
        }
#endif

        if (image) {
            UIImageView *view = [[UIImageView alloc] initWithFrame:[UIScreen mainScreen].bounds];
            UIImageOrientation imageorient = UIImageOrientationUp;

#if !TARGET_OS_TV
            /* Bugs observed / workaround tested in iOS 8.3. */
            if (UIInterfaceOrientationIsLandscape(curorient)) {
                if (image.size.width < image.size.height) {
                    /* On iOS 8, portrait launch images displayed in forced-
                     * landscape mode (e.g. a standard Default.png on an iPhone
                     * when Info.plist only supports landscape orientations) need
                     * to be rotated to display in the expected orientation. */
                    if (curorient == UIInterfaceOrientationLandscapeLeft) {
                        imageorient = UIImageOrientationRight;
                    } else if (curorient == UIInterfaceOrientationLandscapeRight) {
                        imageorient = UIImageOrientationLeft;
                    }
                }
            }
#endif

            /* Create the properly oriented image. */
            view.image = [[UIImage alloc] initWithCGImage:image.CGImage scale:image.scale orientation:imageorient];

            self.view = view;
        }
    }

    return self;
}

- (void)loadView
{
    /* Do nothing. */
}

#if !TARGET_OS_TV
- (BOOL)shouldAutorotate
{
    /* If YES, the launch image will be incorrectly rotated in some cases. */
    return NO;
}

- (NSUInteger)supportedInterfaceOrientations
{
    /* We keep the supported orientations unrestricted to avoid the case where
     * there are no common orientations between the ones set in Info.plist and
     * the ones set here (it will cause an exception in that case.) */
    return UIInterfaceOrientationMaskAll;
}
#endif /* !TARGET_OS_TV */

@end

@implementation SDLUIKitDelegate {
    UIWindow *launchWindow;
}

/* convenience method */
+ (id)sharedAppDelegate
{
    /* the delegate is set in UIApplicationMain(), which is guaranteed to be
     * called before this method */
    return [UIApplication sharedApplication].delegate;
}

+ (NSString *)getAppDelegateClassName
{
    /* subclassing notice: when you subclass this appdelegate, make sure to add
     * a category to override this method and return the actual name of the
     * delegate */
    return @"SDLUIKitDelegate";
}

- (void)hideLaunchScreen
{
    UIWindow *window = launchWindow;

    if (!window || window.hidden) {
        return;
    }

    launchWindow = nil;

    /* Do a nice animated fade-out (roughly matches the real launch behavior.) */
    [UIView animateWithDuration:0.2 animations:^{
        window.alpha = 0.0;
    } completion:^(BOOL finished) {
        window.hidden = YES;
        UIKit_ForceUpdateHomeIndicator(); /* Wait for launch screen to hide so settings are applied to the actual view controller. */
    }];
}

- (void)postFinishLaunch
{
    /* Hide the launch screen the next time the run loop is run. SDL apps will
     * have a chance to load resources while the launch screen is still up. */
    [self performSelector:@selector(hideLaunchScreen) withObject:nil afterDelay:0.0];

    /* run the user's application, passing argc and argv */
    SDL_iPhoneSetEventPump(SDL_TRUE);
    exit_status = forward_main(forward_argc, forward_argv);
    SDL_iPhoneSetEventPump(SDL_FALSE);

    if (launchWindow) {
        launchWindow.hidden = YES;
        launchWindow = nil;
    }

    /* exit, passing the return status from the user's application */
    /* We don't actually exit to support applications that do setup in their
     * main function and then allow the Cocoa event loop to run. */
    /* exit(exit_status); */
}

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
    NSBundle *bundle = [NSBundle mainBundle];

#ifdef SDL_IPHONE_LAUNCHSCREEN
    /* The normal launch screen is displayed until didFinishLaunching returns,
     * but SDL_main is called after that happens and there may be a noticeable
     * delay between the start of SDL_main and when the first real frame is
     * displayed (e.g. if resources are loaded before SDL_GL_SwapWindow is
     * called), so we show the launch screen programmatically until the first
     * time events are pumped. */
    UIViewController *vc = nil;
    NSString *screenname = nil;

    /* tvOS only uses a plain launch image. */
#if !TARGET_OS_TV
    screenname = [bundle objectForInfoDictionaryKey:@"UILaunchStoryboardName"];

    if (screenname) {
        @try {
            /* The launch storyboard is actually a nib in some older versions of
             * Xcode. We'll try to load it as a storyboard first, as it's more
             * modern. */
            UIStoryboard *storyboard = [UIStoryboard storyboardWithName:screenname bundle:bundle];
            __auto_type storyboardVc = [storyboard instantiateInitialViewController];
            vc = [[SDLLaunchStoryboardViewController alloc] initWithStoryboardViewController:storyboardVc];
        }
        @catch (NSException *exception) {
            /* Do nothing (there's more code to execute below). */
        }
    }
#endif

    if (vc == nil) {
        vc = [[SDLLaunchScreenController alloc] initWithNibName:screenname bundle:bundle];
    }

    if (vc.view) {
        launchWindow = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];

        /* We don't want the launch window immediately hidden when a real SDL
         * window is shown - we fade it out ourselves when we're ready. */
        launchWindow.windowLevel = UIWindowLevelNormal + 1.0;

        /* Show the window but don't make it key. Events should always go to
         * other windows when possible. */
        launchWindow.hidden = NO;

        launchWindow.rootViewController = vc;
    }
#endif

    /* Set working directory to resource path */
    [[NSFileManager defaultManager] changeCurrentDirectoryPath:[bundle resourcePath]];

    /* register a callback for the idletimer hint */
    SDL_AddHintCallback(SDL_HINT_IDLE_TIMER_DISABLED,
                        SDL_IdleTimerDisabledChanged, NULL);

    SDL_SetMainReady();
    [self performSelector:@selector(postFinishLaunch) withObject:nil afterDelay:0.0];

    return YES;
}

- (UIWindow *)window
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();
    if (_this) {
        SDL_Window *window = NULL;
        for (window = _this->windows; window != NULL; window = window->next) {
            SDL_WindowData *data = (__bridge SDL_WindowData *) window->driverdata;
            if (data != nil) {
                return data.uiwindow;
            }
        }
    }
    return nil;
}

- (void)setWindow:(UIWindow *)window
{
    /* Do nothing. */
}

- (void)sendDropFileForURL:(NSURL *)url
{
    NSURL *fileURL = url.filePathURL;
    if (fileURL != nil) {
        SDL_SendDropFile(NULL, fileURL.path.UTF8String);
    } else {
        SDL_SendDropFile(NULL, url.absoluteString.UTF8String);
    }
    SDL_SendDropComplete(NULL);
}

#if TARGET_OS_TV || (defined(__IPHONE_9_0) && __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_9_0)

- (BOOL)application:(UIApplication *)app openURL:(NSURL *)url options:(NSDictionary<UIApplicationOpenURLOptionsKey,id> *)options
{
    /* TODO: Handle options */
    [self sendDropFileForURL:url];
    return YES;
}

#else

- (BOOL)application:(UIApplication *)application openURL:(NSURL *)url sourceApplication:(NSString *)sourceApplication annotation:(id)annotation
{
    [self sendDropFileForURL:url];
    return YES;
}

#endif

@end

#endif /* SDL_VIDEO_DRIVER_UIKIT */

/* vi: set ts=4 sw=4 expandtab: */
