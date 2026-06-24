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

#include "SDL_cocoavideo.h"

/* We need this for IODisplayCreateInfoDictionary and kIODisplayOnlyPreferredName */
#include <IOKit/graphics/IOGraphicsLib.h>

/* We need this for CVDisplayLinkGetNominalOutputVideoRefreshPeriod */
#include <CoreVideo/CVBase.h>
#include <CoreVideo/CVDisplayLink.h>

/* This gets us MAC_OS_X_VERSION_MIN_REQUIRED... */
#include <AvailabilityMacros.h>

#ifndef MAC_OS_X_VERSION_10_13
#define NSAppKitVersionNumber10_12 1504
#endif
#if (IOGRAPHICSTYPES_REV < 40)
#define kDisplayModeNativeFlag 0x02000000
#endif


static int CG_SetError(const char *prefix, CGDisplayErr result)
{
    const char *error;

    switch (result) {
    case kCGErrorFailure:
        error = "kCGErrorFailure";
        break;
    case kCGErrorIllegalArgument:
        error = "kCGErrorIllegalArgument";
        break;
    case kCGErrorInvalidConnection:
        error = "kCGErrorInvalidConnection";
        break;
    case kCGErrorInvalidContext:
        error = "kCGErrorInvalidContext";
        break;
    case kCGErrorCannotComplete:
        error = "kCGErrorCannotComplete";
        break;
    case kCGErrorNotImplemented:
        error = "kCGErrorNotImplemented";
        break;
    case kCGErrorRangeCheck:
        error = "kCGErrorRangeCheck";
        break;
    case kCGErrorTypeCheck:
        error = "kCGErrorTypeCheck";
        break;
    case kCGErrorInvalidOperation:
        error = "kCGErrorInvalidOperation";
        break;
    case kCGErrorNoneAvailable:
        error = "kCGErrorNoneAvailable";
        break;
    default:
        error = "Unknown Error";
        break;
    }
    return SDL_SetError("%s: %s", prefix, error);
}

static int GetDisplayModeRefreshRate(CGDisplayModeRef vidmode, CVDisplayLinkRef link)
{
    int refreshRate = (int) (CGDisplayModeGetRefreshRate(vidmode) + 0.5);

    /* CGDisplayModeGetRefreshRate can return 0 (eg for built-in displays). */
    if (refreshRate == 0 && link != NULL) {
        CVTime time = CVDisplayLinkGetNominalOutputVideoRefreshPeriod(link);
        if ((time.flags & kCVTimeIsIndefinite) == 0 && time.timeValue != 0) {
            refreshRate = (int) ((time.timeScale / (double) time.timeValue) + 0.5);
        }
    }

    return refreshRate;
}

static SDL_bool HasValidDisplayModeFlags(CGDisplayModeRef vidmode)
{
    uint32_t ioflags = CGDisplayModeGetIOFlags(vidmode);

    /* Filter out modes which have flags that we don't want. */
    if (ioflags & (kDisplayModeNeverShowFlag | kDisplayModeNotGraphicsQualityFlag)) {
        return SDL_FALSE;
    }

    /* Filter out modes which don't have flags that we want. */
    if (!(ioflags & kDisplayModeValidFlag) || !(ioflags & kDisplayModeSafeFlag)) {
        return SDL_FALSE;
    }

    return SDL_TRUE;
}

static Uint32 GetDisplayModePixelFormat(CGDisplayModeRef vidmode)
{
    /* This API is deprecated in 10.11 with no good replacement (as of 10.15). */
    CFStringRef fmt = CGDisplayModeCopyPixelEncoding(vidmode);
    Uint32 pixelformat = SDL_PIXELFORMAT_UNKNOWN;

    if (CFStringCompare(fmt, CFSTR(IO32BitDirectPixels),
                        kCFCompareCaseInsensitive) == kCFCompareEqualTo) {
        pixelformat = SDL_PIXELFORMAT_ARGB8888;
    } else if (CFStringCompare(fmt, CFSTR(IO16BitDirectPixels),
                        kCFCompareCaseInsensitive) == kCFCompareEqualTo) {
        pixelformat = SDL_PIXELFORMAT_ARGB1555;
    } else if (CFStringCompare(fmt, CFSTR(kIO30BitDirectPixels),
                        kCFCompareCaseInsensitive) == kCFCompareEqualTo) {
        pixelformat = SDL_PIXELFORMAT_ARGB2101010;
    } else {
        /* ignore 8-bit and such for now. */
    }

    CFRelease(fmt);

    return pixelformat;
}

static SDL_bool GetDisplayMode(_THIS, CGDisplayModeRef vidmode, SDL_bool vidmodeCurrent, CFArrayRef modelist, CVDisplayLinkRef link, SDL_DisplayMode *mode)
{
    SDL_DisplayModeData *data;
    bool usableForGUI = CGDisplayModeIsUsableForDesktopGUI(vidmode);
    int width = (int) CGDisplayModeGetWidth(vidmode);
    int height = (int) CGDisplayModeGetHeight(vidmode);
    uint32_t ioflags = CGDisplayModeGetIOFlags(vidmode);
    int refreshrate = GetDisplayModeRefreshRate(vidmode, link);
    Uint32 format = GetDisplayModePixelFormat(vidmode);
    bool interlaced = (ioflags & kDisplayModeInterlacedFlag) != 0;
    CFMutableArrayRef modes;

    if (format == SDL_PIXELFORMAT_UNKNOWN) {
        return SDL_FALSE;
    }

    /* Don't fail the current mode based on flags because this could prevent Cocoa_InitModes from
     * succeeding if the current mode lacks certain flags (esp kDisplayModeSafeFlag). */
    if (!vidmodeCurrent && !HasValidDisplayModeFlags(vidmode)) {
        return SDL_FALSE;
    }

    modes = CFArrayCreateMutable(NULL, 0, &kCFTypeArrayCallBacks);
    CFArrayAppendValue(modes, vidmode);

    /* If a list of possible diplay modes is passed in, use it to filter out
     * modes that have duplicate sizes. We don't just rely on SDL's higher level
     * duplicate filtering because this code can choose what properties are
     * prefered, and it can add CGDisplayModes to the DisplayModeData's list of
     * modes to try (see comment below for why that's necessary).
     * CGDisplayModeGetPixelWidth and friends are only available in 10.8+. */
#ifdef MAC_OS_X_VERSION_10_8
    if (modelist != NULL && floor(NSAppKitVersionNumber) > NSAppKitVersionNumber10_7) {
        int pixelW = (int) CGDisplayModeGetPixelWidth(vidmode);
        int pixelH = (int) CGDisplayModeGetPixelHeight(vidmode);

        CFIndex modescount = CFArrayGetCount(modelist);
        int  i;

        for (i = 0; i < modescount; i++) {
            int otherW, otherH, otherpixelW, otherpixelH, otherrefresh;
            Uint32 otherformat;
            bool otherGUI;
            CGDisplayModeRef othermode = (CGDisplayModeRef) CFArrayGetValueAtIndex(modelist, i);
            uint32_t otherioflags = CGDisplayModeGetIOFlags(othermode);

            if (CFEqual(vidmode, othermode)) {
                continue;
            }

            if (!HasValidDisplayModeFlags(othermode)) {
                continue;
            }

            otherW = (int) CGDisplayModeGetWidth(othermode);
            otherH = (int) CGDisplayModeGetHeight(othermode);
            otherpixelW = (int) CGDisplayModeGetPixelWidth(othermode);
            otherpixelH = (int) CGDisplayModeGetPixelHeight(othermode);
            otherrefresh = GetDisplayModeRefreshRate(othermode, link);
            otherformat = GetDisplayModePixelFormat(othermode);
            otherGUI = CGDisplayModeIsUsableForDesktopGUI(othermode);

            /* Ignore this mode if it's low-dpi (@1x) and we have a high-dpi
             * mode in the list with the same size in points.
             */
            if (width == pixelW && height == pixelH
                && width == otherW && height == otherH
                && refreshrate == otherrefresh && format == otherformat
                && (otherpixelW != otherW || otherpixelH != otherH)) {
                CFRelease(modes);
                return SDL_FALSE;
            }

            /* Ignore this mode if it's interlaced and there's a non-interlaced
             * mode in the list with the same properties.
             */
            if (interlaced && ((otherioflags & kDisplayModeInterlacedFlag) == 0)
                && width == otherW && height == otherH && pixelW == otherpixelW
                && pixelH == otherpixelH && refreshrate == otherrefresh
                && format == otherformat && usableForGUI == otherGUI) {
                CFRelease(modes);
                return SDL_FALSE;
            }

            /* Ignore this mode if it's not usable for desktop UI and its
             * properties are equal to another GUI-capable mode in the list.
             */
            if (width == otherW && height == otherH && pixelW == otherpixelW
                && pixelH == otherpixelH && !usableForGUI && otherGUI
                && refreshrate == otherrefresh && format == otherformat) {
                CFRelease(modes);
                return SDL_FALSE;
            }

            /* If multiple modes have the exact same properties, they'll all
             * go in the list of modes to try when SetDisplayMode is called.
             * This is needed because kCGDisplayShowDuplicateLowResolutionModes
             * (which is used to expose highdpi display modes) can make the
             * list of modes contain duplicates (according to their properties
             * obtained via public APIs) which don't work with SetDisplayMode.
             * Those duplicate non-functional modes *do* have different pixel
             * formats according to their internal data structure viewed with
             * NSLog, but currently no public API can detect that.
             * https://bugzilla.libsdl.org/show_bug.cgi?id=4822
             *
             * As of macOS 10.15.0, those duplicates have the exact same
             * properties via public APIs in every way (even their IO flags and
             * CGDisplayModeGetIODisplayModeID is the same), so we could test
             * those for equality here too, but I'm intentionally not doing that
             * in case there are duplicate modes with different IO flags or IO
             * display mode IDs in the future. In that case I think it's better
             * to try them all in SetDisplayMode than to risk one of them being
             * correct but it being filtered out by SDL_AddDisplayMode as being
             * a duplicate.
             */
            if (width == otherW && height == otherH && pixelW == otherpixelW
                && pixelH == otherpixelH && usableForGUI == otherGUI
                && refreshrate == otherrefresh && format == otherformat) {
                CFArrayAppendValue(modes, othermode);
            }
        }
    }
#endif

    data = (SDL_DisplayModeData *) SDL_malloc(sizeof(*data));
    if (!data) {
        CFRelease(modes);
        return SDL_FALSE;
    }
    data->modes = modes;
    mode->format = format;
    mode->w = width;
    mode->h = height;
    mode->refresh_rate = refreshrate;
    mode->driverdata = data;
    return SDL_TRUE;
}

static const char *Cocoa_GetDisplayName(CGDirectDisplayID displayID)
{
    /* This API is deprecated in 10.9 with no good replacement (as of 10.15). */
    io_service_t servicePort = CGDisplayIOServicePort(displayID);
    CFDictionaryRef deviceInfo = IODisplayCreateInfoDictionary(servicePort, kIODisplayOnlyPreferredName);
    NSDictionary *localizedNames = [(__bridge NSDictionary *)deviceInfo objectForKey:[NSString stringWithUTF8String:kDisplayProductName]];
    const char* displayName = NULL;

    if ([localizedNames count] > 0) {
        displayName = SDL_strdup([[localizedNames objectForKey:[[localizedNames allKeys] objectAtIndex:0]] UTF8String]);
    }
    CFRelease(deviceInfo);
    return displayName;
}

void Cocoa_InitModes(_THIS)
{ @autoreleasepool
{
    CGDisplayErr result;
    CGDirectDisplayID *displays;
    CGDisplayCount numDisplays;
    SDL_bool isstack;
    int pass, i;

    result = CGGetOnlineDisplayList(0, NULL, &numDisplays);
    if (result != kCGErrorSuccess) {
        CG_SetError("CGGetOnlineDisplayList()", result);
        return;
    }
    displays = SDL_small_alloc(CGDirectDisplayID, numDisplays, &isstack);
    result = CGGetOnlineDisplayList(numDisplays, displays, &numDisplays);
    if (result != kCGErrorSuccess) {
        CG_SetError("CGGetOnlineDisplayList()", result);
        SDL_small_free(displays, isstack);
        return;
    }

    /* Pick up the primary display in the first pass, then get the rest */
    for (pass = 0; pass < 2; ++pass) {
        for (i = 0; i < numDisplays; ++i) {
            SDL_VideoDisplay display;
            SDL_DisplayData *displaydata;
            SDL_DisplayMode mode;
            CGDisplayModeRef moderef = NULL;
            CVDisplayLinkRef link = NULL;

            if (pass == 0) {
                if (!CGDisplayIsMain(displays[i])) {
                    continue;
                }
            } else {
                if (CGDisplayIsMain(displays[i])) {
                    continue;
                }
            }

            if (CGDisplayMirrorsDisplay(displays[i]) != kCGNullDirectDisplay) {
                continue;
            }

            moderef = CGDisplayCopyDisplayMode(displays[i]);

            if (!moderef) {
                continue;
            }

            displaydata = (SDL_DisplayData *) SDL_malloc(sizeof(*displaydata));
            if (!displaydata) {
                CGDisplayModeRelease(moderef);
                continue;
            }
            displaydata->display = displays[i];

            CVDisplayLinkCreateWithCGDisplay(displays[i], &link);

            SDL_zero(display);
            /* this returns a stddup'ed string */
            display.name = (char *)Cocoa_GetDisplayName(displays[i]);
            if (!GetDisplayMode(_this, moderef, SDL_TRUE, NULL, link, &mode)) {
                CVDisplayLinkRelease(link);
                CGDisplayModeRelease(moderef);
                SDL_free(display.name);
                SDL_free(displaydata);
                continue;
            }

            CVDisplayLinkRelease(link);
            CGDisplayModeRelease(moderef);

            display.desktop_mode = mode;
            display.current_mode = mode;
            display.driverdata = displaydata;
            SDL_AddVideoDisplay(&display, SDL_FALSE);
            SDL_free(display.name);
        }
    }
    SDL_small_free(displays, isstack);
}}

int Cocoa_GetDisplayBounds(_THIS, SDL_VideoDisplay * display, SDL_Rect * rect)
{
    SDL_DisplayData *displaydata = (SDL_DisplayData *) display->driverdata;
    CGRect cgrect;

    cgrect = CGDisplayBounds(displaydata->display);
    rect->x = (int)cgrect.origin.x;
    rect->y = (int)cgrect.origin.y;
    rect->w = (int)cgrect.size.width;
    rect->h = (int)cgrect.size.height;
    return 0;
}

int Cocoa_GetDisplayUsableBounds(_THIS, SDL_VideoDisplay * display, SDL_Rect * rect)
{
    SDL_DisplayData *displaydata = (SDL_DisplayData *) display->driverdata;
    const CGDirectDisplayID cgdisplay = displaydata->display;
    NSArray *screens = [NSScreen screens];
    NSScreen *screen = nil;

    /* !!! FIXME: maybe track the NSScreen in SDL_DisplayData? */
    for (NSScreen *i in screens) {
        const CGDirectDisplayID thisDisplay = (CGDirectDisplayID) [[[i deviceDescription] objectForKey:@"NSScreenNumber"] unsignedIntValue];
        if (thisDisplay == cgdisplay) {
            screen = i;
            break;
        }
    }

    SDL_assert(screen != nil);  /* didn't find it?! */
    if (screen == nil) {
        return -1;
    }

    {
        const NSRect frame = [screen visibleFrame];
        rect->x = (int)frame.origin.x;
        rect->y = (int)(CGDisplayPixelsHigh(kCGDirectMainDisplay) - frame.origin.y - frame.size.height);
        rect->w = (int)frame.size.width;
        rect->h = (int)frame.size.height;
    }

    return 0;
}

int Cocoa_GetDisplayDPI(_THIS, SDL_VideoDisplay * display, float * ddpi, float * hdpi, float * vdpi)
{ @autoreleasepool
{
    const float MM_IN_INCH = 25.4f;

    SDL_DisplayData *data = (SDL_DisplayData *) display->driverdata;

    /* we need the backingScaleFactor for Retina displays, which is only exposed through NSScreen, not CGDisplay, afaik, so find our screen... */
    CGFloat scaleFactor = 1.0f;
    NSArray *screens = [NSScreen screens];
    NSSize displayNativeSize;
    displayNativeSize.width = (int) CGDisplayPixelsWide(data->display);
    displayNativeSize.height = (int) CGDisplayPixelsHigh(data->display);

    for (NSScreen *screen in screens) {
        const CGDirectDisplayID dpyid = (const CGDirectDisplayID ) [[[screen deviceDescription] objectForKey:@"NSScreenNumber"] unsignedIntValue];
        if (dpyid == data->display) {
#ifdef MAC_OS_X_VERSION_10_8
            /* Neither CGDisplayScreenSize(description's NSScreenNumber) nor [NSScreen backingScaleFactor] can calculate the correct dpi in macOS. E.g. backingScaleFactor is always 2 in all display modes for rMBP 16" */
            if (floor(NSAppKitVersionNumber) > NSAppKitVersionNumber10_7) {
                CFStringRef dmKeys[1] = { kCGDisplayShowDuplicateLowResolutionModes };
                CFBooleanRef dmValues[1] = { kCFBooleanTrue };
                CFDictionaryRef dmOptions = CFDictionaryCreate(kCFAllocatorDefault, (const void**) dmKeys, (const void**) dmValues, 1, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks );
                CFArrayRef allDisplayModes = CGDisplayCopyAllDisplayModes(dpyid, dmOptions);
                CFIndex n = CFArrayGetCount(allDisplayModes);
                for(CFIndex i = 0; i < n; ++i) {
                    CGDisplayModeRef m = (CGDisplayModeRef)CFArrayGetValueAtIndex(allDisplayModes, i);
                    CGFloat width = CGDisplayModeGetPixelWidth(m);
                    CGFloat height = CGDisplayModeGetPixelHeight(m);
                    CGFloat HiDPIWidth = CGDisplayModeGetWidth(m);

                    //Only check 1x mode
                    if(width == HiDPIWidth) {
                        if (CGDisplayModeGetIOFlags(m) & kDisplayModeNativeFlag) {
                            displayNativeSize.width = width;
                            displayNativeSize.height = height;
                            break;
                        }

                        //Get the largest size even if kDisplayModeNativeFlag is not present e.g. iMac 27-Inch with 5K Retina
                        if(width > displayNativeSize.width) {
                            displayNativeSize.width = width;
                            displayNativeSize.height = height;
                        }
                    }
                }
                CFRelease(allDisplayModes);
                CFRelease(dmOptions);
            } else
#endif
            {
                // fallback for 10.7
                scaleFactor = [screen backingScaleFactor];
                displayNativeSize.width = displayNativeSize.width * scaleFactor;
                displayNativeSize.height = displayNativeSize.height * scaleFactor;
                break;
            }
        }
    }

    {
        const CGSize displaySize = CGDisplayScreenSize(data->display);
        const int pixelWidth =  displayNativeSize.width;
        const int pixelHeight = displayNativeSize.height;

        if (ddpi) {
            *ddpi = (SDL_ComputeDiagonalDPI(pixelWidth, pixelHeight, displaySize.width / MM_IN_INCH, displaySize.height / MM_IN_INCH));
        }
        if (hdpi) {
            *hdpi = (pixelWidth * MM_IN_INCH / displaySize.width);
        }
        if (vdpi) {
            *vdpi = (pixelHeight * MM_IN_INCH / displaySize.height);
        }
    }
    return 0;
}}

void Cocoa_GetDisplayModes(_THIS, SDL_VideoDisplay * display)
{
    SDL_DisplayData *data = (SDL_DisplayData *) display->driverdata;
    CVDisplayLinkRef link = NULL;
    CGDisplayModeRef desktopmoderef;
    SDL_DisplayMode desktopmode;
    CFArrayRef modes;
    CFDictionaryRef dict = NULL;

    CVDisplayLinkCreateWithCGDisplay(data->display, &link);

    desktopmoderef = CGDisplayCopyDisplayMode(data->display);

    /* CopyAllDisplayModes won't always contain the desktop display mode (if
     * NULL is passed in) - for example on a retina 15" MBP, System Preferences
     * allows choosing 1920x1200 but it's not in the list. AddDisplayMode makes
     * sure there are no duplicates so it's safe to always add the desktop mode
     * even in cases where it is in the CopyAllDisplayModes list.
     */
    if (desktopmoderef && GetDisplayMode(_this, desktopmoderef, SDL_TRUE, NULL, link, &desktopmode)) {
        if (!SDL_AddDisplayMode(display, &desktopmode)) {
            CFRelease(((SDL_DisplayModeData*)desktopmode.driverdata)->modes);
            SDL_free(desktopmode.driverdata);
        }
    }

    CGDisplayModeRelease(desktopmoderef);

    /* By default, CGDisplayCopyAllDisplayModes will only get a subset of the
     * system's available modes. For example on a 15" 2016 MBP, users can
     * choose 1920x1080@2x in System Preferences but it won't show up here,
     * unless we specify the option below.
     * The display modes returned by CGDisplayCopyAllDisplayModes are also not
     * high dpi-capable unless this option is set.
     * macOS 10.15 also seems to have a bug where entering, exiting, and
     * re-entering exclusive fullscreen with a low dpi display mode can cause
     * the content of the screen to move up, which this setting avoids:
     * https://bugzilla.libsdl.org/show_bug.cgi?id=4822
     */
#ifdef MAC_OS_X_VERSION_10_8
    if (floor(NSAppKitVersionNumber) > NSAppKitVersionNumber10_7) {
        const CFStringRef dictkeys[] = {kCGDisplayShowDuplicateLowResolutionModes};
        const CFBooleanRef dictvalues[] = {kCFBooleanTrue};
        dict = CFDictionaryCreate(NULL,
                                  (const void **)dictkeys,
                                  (const void **)dictvalues,
                                  1,
                                  &kCFCopyStringDictionaryKeyCallBacks,
                                  &kCFTypeDictionaryValueCallBacks);
    }
#endif

    modes = CGDisplayCopyAllDisplayModes(data->display, dict);

    if (dict) {
        CFRelease(dict);
    }

    if (modes) {
        CFIndex i;
        const CFIndex count = CFArrayGetCount(modes);

        for (i = 0; i < count; i++) {
            CGDisplayModeRef moderef = (CGDisplayModeRef) CFArrayGetValueAtIndex(modes, i);
            SDL_DisplayMode mode;

            if (GetDisplayMode(_this, moderef, SDL_FALSE, modes, link, &mode)) {
                if (!SDL_AddDisplayMode(display, &mode)) {
                    CFRelease(((SDL_DisplayModeData*)mode.driverdata)->modes);
                    SDL_free(mode.driverdata);
                }
            }
        }

        CFRelease(modes);
    }

    CVDisplayLinkRelease(link);
}

static CGError SetDisplayModeForDisplay(CGDirectDisplayID display, SDL_DisplayModeData *data)
{
    /* SDL_DisplayModeData can contain multiple CGDisplayModes to try (with
     * identical properties), some of which might not work. See GetDisplayMode.
     */
    CGError result = kCGErrorFailure;
    for (CFIndex i = 0; i < CFArrayGetCount(data->modes); i++) {
        CGDisplayModeRef moderef = (CGDisplayModeRef)CFArrayGetValueAtIndex(data->modes, i);
        result = CGDisplaySetDisplayMode(display, moderef, NULL);
        if (result == kCGErrorSuccess) {
            /* If this mode works, try it first next time. */
            CFArrayExchangeValuesAtIndices(data->modes, i, 0);
            break;
        }
    }
    return result;
}

int Cocoa_SetDisplayMode(_THIS, SDL_VideoDisplay * display, SDL_DisplayMode * mode)
{
    SDL_DisplayData *displaydata = (SDL_DisplayData *) display->driverdata;
    SDL_DisplayModeData *data = (SDL_DisplayModeData *) mode->driverdata;
    CGDisplayFadeReservationToken fade_token = kCGDisplayFadeReservationInvalidToken;
    CGError result;

    /* Fade to black to hide resolution-switching flicker */
    if (CGAcquireDisplayFadeReservation(5, &fade_token) == kCGErrorSuccess) {
        CGDisplayFade(fade_token, 0.3, kCGDisplayBlendNormal, kCGDisplayBlendSolidColor, 0.0, 0.0, 0.0, TRUE);
    }

    if (data == display->desktop_mode.driverdata) {
        /* Restoring desktop mode */
        SetDisplayModeForDisplay(displaydata->display, data);

        if (CGDisplayIsMain(displaydata->display)) {
            CGReleaseAllDisplays();
        } else {
            CGDisplayRelease(displaydata->display);
        }
    } else {
        /* Put up the blanking window (a window above all other windows) */
        if (CGDisplayIsMain(displaydata->display)) {
            /* If we don't capture all displays, Cocoa tries to rearrange windows... *sigh* */
            result = CGCaptureAllDisplays();
        } else {
            result = CGDisplayCapture(displaydata->display);
        }
        if (result != kCGErrorSuccess) {
            CG_SetError("CGDisplayCapture()", result);
            goto ERR_NO_CAPTURE;
        }

        /* Do the physical switch */
        result =  SetDisplayModeForDisplay(displaydata->display, data);
        if (result != kCGErrorSuccess) {
            CG_SetError("CGDisplaySwitchToMode()", result);
            goto ERR_NO_SWITCH;
        }
    }

    /* Fade in again (asynchronously) */
    if (fade_token != kCGDisplayFadeReservationInvalidToken) {
        CGDisplayFade(fade_token, 0.5, kCGDisplayBlendSolidColor, kCGDisplayBlendNormal, 0.0, 0.0, 0.0, FALSE);
        CGReleaseDisplayFadeReservation(fade_token);
    }

    return 0;

    /* Since the blanking window covers *all* windows (even force quit) correct recovery is crucial */
ERR_NO_SWITCH:
    if (CGDisplayIsMain(displaydata->display)) {
        CGReleaseAllDisplays();
    } else {
        CGDisplayRelease(displaydata->display);
    }
ERR_NO_CAPTURE:
    if (fade_token != kCGDisplayFadeReservationInvalidToken) {
        CGDisplayFade (fade_token, 0.5, kCGDisplayBlendSolidColor, kCGDisplayBlendNormal, 0.0, 0.0, 0.0, FALSE);
        CGReleaseDisplayFadeReservation(fade_token);
    }
    return -1;
}

void Cocoa_QuitModes(_THIS)
{
    int i, j;

    for (i = 0; i < _this->num_displays; ++i) {
        SDL_VideoDisplay *display = &_this->displays[i];
        SDL_DisplayModeData *mode;

        if (display->current_mode.driverdata != display->desktop_mode.driverdata) {
            Cocoa_SetDisplayMode(_this, display, &display->desktop_mode);
        }

        mode = (SDL_DisplayModeData *) display->desktop_mode.driverdata;
        CFRelease(mode->modes);

        for (j = 0; j < display->num_display_modes; j++) {
            mode = (SDL_DisplayModeData*) display->display_modes[j].driverdata;
            CFRelease(mode->modes);
        }
    }
}

#endif /* SDL_VIDEO_DRIVER_COCOA */

/* vi: set ts=4 sw=4 expandtab: */
