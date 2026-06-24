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

/* This is the iOS implementation of the SDL joystick API */
#include "SDL_events.h"
#include "SDL_joystick.h"
#include "SDL_hints.h"
#include "SDL_stdinc.h"
#include "SDL_timer.h"
#include "../SDL_sysjoystick.h"
#include "../SDL_joystick_c.h"
#include "../hidapi/SDL_hidapijoystick_c.h"
#include "../usb_ids.h"

#include "SDL_mfijoystick_c.h"

#ifndef SDL_EVENTS_DISABLED
#include "../../events/SDL_events_c.h"
#endif

#if TARGET_OS_IOS
#define SDL_JOYSTICK_iOS_ACCELEROMETER
#import <CoreMotion/CoreMotion.h>
#endif

#if defined(__MACOSX__)
#include <IOKit/hid/IOHIDManager.h>
#include <AppKit/NSApplication.h>
#ifndef NSAppKitVersionNumber10_15
#define NSAppKitVersionNumber10_15 1894
#endif
#endif /* __MACOSX__ */

#ifdef SDL_JOYSTICK_MFI
#import <GameController/GameController.h>

static id connectObserver = nil;
static id disconnectObserver = nil;

#include <Availability.h>
#include <objc/message.h>

#ifndef __IPHONE_OS_VERSION_MAX_ALLOWED
#define __IPHONE_OS_VERSION_MAX_ALLOWED 0
#endif

#ifndef __APPLETV_OS_VERSION_MAX_ALLOWED
#define __APPLETV_OS_VERSION_MAX_ALLOWED 0
#endif

#ifndef __MAC_OS_VERSION_MAX_ALLOWED
#define __MAC_OS_VERSION_MAX_ALLOWED 0
#endif

/* remove compilation warnings for strict builds by defining these selectors, even though
 * they are only ever used indirectly through objc_msgSend
 */
@interface GCController (SDL)
#if defined(__MACOSX__) && (__MAC_OS_X_VERSION_MAX_ALLOWED <= 101600)
+ (BOOL)supportsHIDDevice:(IOHIDDeviceRef)device;
#endif
#if !((__IPHONE_OS_VERSION_MAX_ALLOWED >= 130000) || (__APPLETV_OS_VERSION_MAX_ALLOWED >= 130000) || (__MAC_OS_VERSION_MAX_ALLOWED >= 1500000))
@property(nonatomic, readonly) NSString *productCategory;
#endif
#if !((__IPHONE_OS_VERSION_MAX_ALLOWED >= 140500) || (__APPLETV_OS_VERSION_MAX_ALLOWED >= 140500) || (__MAC_OS_X_VERSION_MAX_ALLOWED >= 110300))
@property(class, nonatomic, readwrite) BOOL shouldMonitorBackgroundEvents;
#endif
@end
@interface GCExtendedGamepad (SDL)
#if !((__IPHONE_OS_VERSION_MAX_ALLOWED >= 121000) || (__APPLETV_OS_VERSION_MAX_ALLOWED >= 121000) || (__MAC_OS_VERSION_MAX_ALLOWED >= 1401000))
@property(nonatomic, readonly, nullable) GCControllerButtonInput *leftThumbstickButton;
@property(nonatomic, readonly, nullable) GCControllerButtonInput *rightThumbstickButton;
#endif
#if !((__IPHONE_OS_VERSION_MAX_ALLOWED >= 130000) || (__APPLETV_OS_VERSION_MAX_ALLOWED >= 130000) || (__MAC_OS_VERSION_MAX_ALLOWED >= 1500000))
@property(nonatomic, readonly) GCControllerButtonInput *buttonMenu;
@property(nonatomic, readonly, nullable) GCControllerButtonInput *buttonOptions;
#endif
#if !((__IPHONE_OS_VERSION_MAX_ALLOWED >= 140000) || (__APPLETV_OS_VERSION_MAX_ALLOWED >= 140000) || (__MAC_OS_VERSION_MAX_ALLOWED > 1500000))
@property(nonatomic, readonly, nullable) GCControllerButtonInput *buttonHome;
#endif
@end
@interface GCMicroGamepad (SDL)
#if !((__IPHONE_OS_VERSION_MAX_ALLOWED >= 130000) || (__APPLETV_OS_VERSION_MAX_ALLOWED >= 130000) || (__MAC_OS_VERSION_MAX_ALLOWED >= 1500000))
@property(nonatomic, readonly) GCControllerButtonInput *buttonMenu;
#endif
@end

#if (__IPHONE_OS_VERSION_MAX_ALLOWED >= 140000) || (__APPLETV_OS_VERSION_MAX_ALLOWED >= 140000) || (__MAC_OS_VERSION_MAX_ALLOWED > 1500000) || (__MAC_OS_X_VERSION_MAX_ALLOWED > 101600)
#define ENABLE_MFI_BATTERY
#define ENABLE_MFI_RUMBLE
#define ENABLE_MFI_LIGHT
#define ENABLE_MFI_SENSORS
#define ENABLE_MFI_SYSTEM_GESTURE_STATE
#define ENABLE_PHYSICAL_INPUT_PROFILE
#endif

#ifdef ENABLE_MFI_RUMBLE
#import <CoreHaptics/CoreHaptics.h>
#endif

#endif /* SDL_JOYSTICK_MFI */

#ifdef SDL_JOYSTICK_iOS_ACCELEROMETER
static const char *accelerometerName = "iOS Accelerometer";
static CMMotionManager *motionManager = nil;
#endif /* SDL_JOYSTICK_iOS_ACCELEROMETER */

static SDL_JoystickDeviceItem *deviceList = NULL;

static int numjoysticks = 0;
int SDL_AppleTVRemoteOpenedAsJoystick = 0;

static SDL_JoystickDeviceItem *GetDeviceForIndex(int device_index)
{
    SDL_JoystickDeviceItem *device = deviceList;
    int i = 0;

    while (i < device_index) {
        if (device == NULL) {
            return NULL;
        }
        device = device->next;
        i++;
    }

    return device;
}

#ifdef SDL_JOYSTICK_MFI
static BOOL IsControllerPS4(GCController *controller)
{
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
        if ([controller.productCategory isEqualToString:@"DualShock 4"]) {
            return TRUE;
        }
    } else {
        if ([controller.vendorName containsString:@"DUALSHOCK"]) {
            return TRUE;
        }
    }
    return FALSE;
}
static BOOL IsControllerPS5(GCController *controller)
{
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
        if ([controller.productCategory isEqualToString:@"DualSense"]) {
            return TRUE;
        }
    } else {
        if ([controller.vendorName containsString:@"DualSense"]) {
            return TRUE;
        }
    }
    return FALSE;
}
static BOOL IsControllerXbox(GCController *controller)
{
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
        if ([controller.productCategory isEqualToString:@"Xbox One"]) {
            return TRUE;
        }
    } else {
        if ([controller.vendorName containsString:@"Xbox"]) {
            return TRUE;
        }
    }
    return FALSE;
}
static BOOL IsControllerSwitchPro(GCController *controller)
{
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
        if ([controller.productCategory isEqualToString:@"Switch Pro Controller"]) {
            return TRUE;
        }
    }
    return FALSE;
}
static BOOL IsControllerSwitchJoyConL(GCController *controller)
{
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
        if ([controller.productCategory isEqualToString:@"Nintendo Switch Joy-Con (L)"]) {
            return TRUE;
        }
    }
    return FALSE;
}
static BOOL IsControllerSwitchJoyConR(GCController *controller)
{
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
        if ([controller.productCategory isEqualToString:@"Nintendo Switch Joy-Con (R)"]) {
            return TRUE;
        }
    }
    return FALSE;
}
static BOOL IsControllerSwitchJoyConPair(GCController *controller)
{
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
        if ([controller.productCategory isEqualToString:@"Nintendo Switch Joy-Con (L/R)"]) {
            return TRUE;
        }
    }
    return FALSE;
}
static BOOL IsControllerStadia(GCController *controller)
{
    if ([controller.vendorName hasPrefix:@"Stadia"]) {
        return TRUE;
    }
    return FALSE;
}
static BOOL IsControllerBackboneOne(GCController *controller)
{
    if ([controller.vendorName hasPrefix:@"Backbone One"]) {
        return TRUE;
    }
    return FALSE;
}
static void CheckControllerSiriRemote(GCController *controller, int *is_siri_remote)
{
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
        if ([controller.productCategory hasPrefix:@"Siri Remote"]) {
            *is_siri_remote = 1;
            SDL_sscanf(controller.productCategory.UTF8String, "Siri Remote (%i%*s Generation)", is_siri_remote);
            return;
        }
    }
    *is_siri_remote = 0;
}

#ifdef ENABLE_PHYSICAL_INPUT_PROFILE
static BOOL ElementAlreadyHandled(SDL_JoystickDeviceItem *device, NSString *element, NSDictionary<NSString *, GCControllerElement *> *elements)
{
    if ([element isEqualToString:@"Left Thumbstick Left"] ||
        [element isEqualToString:@"Left Thumbstick Right"]) {
        if (elements[@"Left Thumbstick X Axis"]) {
            return TRUE;
        }
    }
    if ([element isEqualToString:@"Left Thumbstick Up"] ||
        [element isEqualToString:@"Left Thumbstick Down"]) {
        if (elements[@"Left Thumbstick Y Axis"]) {
            return TRUE;
        }
    }
    if ([element isEqualToString:@"Right Thumbstick Left"] ||
        [element isEqualToString:@"Right Thumbstick Right"]) {
        if (elements[@"Right Thumbstick X Axis"]) {
            return TRUE;
        }
    }
    if ([element isEqualToString:@"Right Thumbstick Up"] ||
        [element isEqualToString:@"Right Thumbstick Down"]) {
        if (elements[@"Right Thumbstick Y Axis"]) {
            return TRUE;
        }
    }
    if (device->is_siri_remote) {
        if ([element isEqualToString:@"Direction Pad Left"] ||
            [element isEqualToString:@"Direction Pad Right"]) {
            if (elements[@"Direction Pad X Axis"]) {
                return TRUE;
            }
        }
        if ([element isEqualToString:@"Direction Pad Up"] ||
            [element isEqualToString:@"Direction Pad Down"]) {
            if (elements[@"Direction Pad Y Axis"]) {
                return TRUE;
            }
        }
    } else {
        if ([element isEqualToString:@"Direction Pad X Axis"]) {
            if (elements[@"Direction Pad Left"] &&
                elements[@"Direction Pad Right"]) {
                return TRUE;
            }
        }
        if ([element isEqualToString:@"Direction Pad Y Axis"]) {
            if (elements[@"Direction Pad Up"] &&
                elements[@"Direction Pad Down"]) {
                return TRUE;
            }
        }
    }
    if ([element isEqualToString:@"Cardinal Direction Pad X Axis"]) {
        if (elements[@"Cardinal Direction Pad Left"] &&
            elements[@"Cardinal Direction Pad Right"]) {
            return TRUE;
        }
    }
    if ([element isEqualToString:@"Cardinal Direction Pad Y Axis"]) {
        if (elements[@"Cardinal Direction Pad Up"] &&
            elements[@"Cardinal Direction Pad Down"]) {
            return TRUE;
        }
    }
    if ([element isEqualToString:@"Touchpad 1 X Axis"] ||
        [element isEqualToString:@"Touchpad 1 Y Axis"] ||
        [element isEqualToString:@"Touchpad 1 Left"] ||
        [element isEqualToString:@"Touchpad 1 Right"] ||
        [element isEqualToString:@"Touchpad 1 Up"] ||
        [element isEqualToString:@"Touchpad 1 Down"] ||
        [element isEqualToString:@"Touchpad 2 X Axis"] ||
        [element isEqualToString:@"Touchpad 2 Y Axis"] ||
        [element isEqualToString:@"Touchpad 2 Left"] ||
        [element isEqualToString:@"Touchpad 2 Right"] ||
        [element isEqualToString:@"Touchpad 2 Up"] ||
        [element isEqualToString:@"Touchpad 2 Down"]) {
        /* The touchpad is handled separately */
        return TRUE;
    }
    if ([element isEqualToString:@"Button Home"]) {
        if (device->is_switch_joycon_pair) {
            /* The Nintendo Switch JoyCon home button doesn't ever show as being held down */
            return TRUE;
        }
#if TARGET_OS_TV
        /* The OS uses the home button, it's not available to apps */
        return TRUE;
#endif
    }
    if ([element isEqualToString:@"Button Share"]) {
        if (device->is_backbone_one) {
            /* The Backbone app uses share button */
            return TRUE;
        }
    }
    return FALSE;
}
#endif /* ENABLE_PHYSICAL_INPUT_PROFILE */

static BOOL IOS_AddMFIJoystickDevice(SDL_JoystickDeviceItem *device, GCController *controller)
{
    Uint16 vendor = 0;
    Uint16 product = 0;
    Uint8 subtype = 0;
    Uint16 signature = 0;
    const char *name = NULL;

    if (@available(macOS 11.3, iOS 14.5, tvOS 14.5, *)) {
        if (!GCController.shouldMonitorBackgroundEvents) {
            GCController.shouldMonitorBackgroundEvents = YES;
        }
    }

    /* Explicitly retain the controller because SDL_JoystickDeviceItem is a
     * struct, and ARC doesn't work with structs. */
    device->controller = (__bridge GCController *)CFBridgingRetain(controller);

    if (controller.vendorName) {
        name = controller.vendorName.UTF8String;
    }

    if (!name) {
        name = "MFi Gamepad";
    }

    device->name = SDL_CreateJoystickName(0, 0, NULL, name);

#ifdef DEBUG_CONTROLLER_PROFILE
    NSLog(@"Product name: %@\n", controller.vendorName);
    NSLog(@"Product category: %@\n", controller.productCategory);
    NSLog(@"Elements available:\n");
#ifdef ENABLE_PHYSICAL_INPUT_PROFILE
    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        NSDictionary<NSString *, GCControllerElement *> *elements = controller.physicalInputProfile.elements;
        for (id key in controller.physicalInputProfile.buttons) {
            NSLog(@"\tButton: %@ (%s)\n", key, elements[key].analog ? "analog" : "digital");
        }
        for (id key in controller.physicalInputProfile.axes) {
            NSLog(@"\tAxis: %@\n", key);
        }
        for (id key in controller.physicalInputProfile.dpads) {
            NSLog(@"\tHat: %@\n", key);
        }
    }
#endif
#endif

    device->is_xbox = IsControllerXbox(controller);
    device->is_ps4 = IsControllerPS4(controller);
    device->is_ps5 = IsControllerPS5(controller);
    device->is_switch_pro = IsControllerSwitchPro(controller);
    device->is_switch_joycon_pair = IsControllerSwitchJoyConPair(controller);
    device->is_stadia = IsControllerStadia(controller);
    device->is_backbone_one = IsControllerBackboneOne(controller);
    device->is_switch_joyconL = IsControllerSwitchJoyConL(controller);
    device->is_switch_joyconR = IsControllerSwitchJoyConR(controller);
#ifdef SDL_JOYSTICK_HIDAPI
    if ((device->is_xbox && (HIDAPI_IsDeviceTypePresent(SDL_CONTROLLER_TYPE_XBOXONE) ||
                             HIDAPI_IsDeviceTypePresent(SDL_CONTROLLER_TYPE_XBOX360))) ||
        (device->is_ps4 && HIDAPI_IsDeviceTypePresent(SDL_CONTROLLER_TYPE_PS4)) ||
        (device->is_ps5 && HIDAPI_IsDeviceTypePresent(SDL_CONTROLLER_TYPE_PS5)) ||
        (device->is_switch_pro && HIDAPI_IsDeviceTypePresent(SDL_CONTROLLER_TYPE_NINTENDO_SWITCH_PRO)) ||
        (device->is_switch_joycon_pair && HIDAPI_IsDevicePresent(USB_VENDOR_NINTENDO, USB_PRODUCT_NINTENDO_SWITCH_JOYCON_PAIR, 0, "")) ||
        (device->is_stadia && HIDAPI_IsDevicePresent(USB_VENDOR_GOOGLE, USB_PRODUCT_GOOGLE_STADIA_CONTROLLER, 0, "")) ||
        (device->is_switch_joyconL && HIDAPI_IsDevicePresent(USB_VENDOR_NINTENDO, USB_PRODUCT_NINTENDO_SWITCH_JOYCON_LEFT, 0, "")) ||
        (device->is_switch_joyconR && HIDAPI_IsDevicePresent(USB_VENDOR_NINTENDO, USB_PRODUCT_NINTENDO_SWITCH_JOYCON_RIGHT, 0, ""))) {
        /* The HIDAPI driver is taking care of this device */
        return FALSE;
    }
#endif
    if (device->is_xbox && SDL_strncmp(name, "GamePad-", 8) == 0) {
        /* This is a Steam Virtual Gamepad, which isn't supported by GCController */
        return FALSE;
    }
    CheckControllerSiriRemote(controller, &device->is_siri_remote);

    if (device->is_siri_remote && !SDL_GetHintBoolean(SDL_HINT_TV_REMOTE_AS_JOYSTICK, SDL_TRUE)) {
        /* Ignore remotes, they'll be handled as keyboard input */
        return SDL_FALSE;
    }

#ifdef ENABLE_PHYSICAL_INPUT_PROFILE
    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        if (controller.physicalInputProfile.buttons[GCInputDualShockTouchpadButton] != nil) {
            device->has_dualshock_touchpad = TRUE;
        }
        if (controller.physicalInputProfile.buttons[GCInputXboxPaddleOne] != nil) {
            device->has_xbox_paddles = TRUE;
        }
        if (controller.physicalInputProfile.buttons[@"Button Share"] != nil) {
            device->has_xbox_share_button = TRUE;
        }
    }
#endif /* ENABLE_PHYSICAL_INPUT_PROFILE */

    if (device->is_backbone_one) {
        vendor = USB_VENDOR_BACKBONE;
        if (device->is_ps5) {
            product = USB_PRODUCT_BACKBONE_ONE_IOS_PS5;
        } else {
            product = USB_PRODUCT_BACKBONE_ONE_IOS;
        }
    } else if (device->is_xbox) {
        vendor = USB_VENDOR_MICROSOFT;
        if (device->has_xbox_paddles) {
            /* Assume Xbox One Elite Series 2 Controller unless/until GCController flows VID/PID */
            product = USB_PRODUCT_XBOX_ONE_ELITE_SERIES_2_BLUETOOTH;
        } else if (device->has_xbox_share_button) {
            /* Assume Xbox Series X Controller unless/until GCController flows VID/PID */
            product = USB_PRODUCT_XBOX_SERIES_X_BLE;
        } else {
            /* Assume Xbox One S Bluetooth Controller unless/until GCController flows VID/PID */
            product = USB_PRODUCT_XBOX_ONE_S_REV1_BLUETOOTH;
        }
    } else if (device->is_ps4) {
        /* Assume DS4 Slim unless/until GCController flows VID/PID */
        vendor = USB_VENDOR_SONY;
        product = USB_PRODUCT_SONY_DS4_SLIM;
        if (device->has_dualshock_touchpad) {
            subtype = 1;
        }
    } else if (device->is_ps5) {
        vendor = USB_VENDOR_SONY;
        product = USB_PRODUCT_SONY_DS5;
    } else if (device->is_switch_pro) {
        vendor = USB_VENDOR_NINTENDO;
        product = USB_PRODUCT_NINTENDO_SWITCH_PRO;
    } else if (device->is_switch_joycon_pair) {
        vendor = USB_VENDOR_NINTENDO;
        product = USB_PRODUCT_NINTENDO_SWITCH_JOYCON_PAIR;
    } else if (device->is_switch_joyconL) {
        vendor = USB_VENDOR_NINTENDO;
        product = USB_PRODUCT_NINTENDO_SWITCH_JOYCON_LEFT;
    } else if (device->is_switch_joyconR) {
        vendor = USB_VENDOR_NINTENDO;
        product = USB_PRODUCT_NINTENDO_SWITCH_JOYCON_RIGHT;
#ifdef ENABLE_PHYSICAL_INPUT_PROFILE
    } else if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        vendor = USB_VENDOR_APPLE;
        product = 4;
        subtype = 4;
#endif
    } else if (controller.extendedGamepad) {
        vendor = USB_VENDOR_APPLE;
        product = 1;
        subtype = 1;
    } else if (controller.gamepad) {
        vendor = USB_VENDOR_APPLE;
        product = 2;
        subtype = 2;
#if TARGET_OS_TV
    } else if (controller.microGamepad) {
        vendor = USB_VENDOR_APPLE;
        product = 3;
        subtype = 3;
#endif
    } else {
        /* We don't know how to get input events from this device */
        return SDL_FALSE;
    }

#ifdef ENABLE_PHYSICAL_INPUT_PROFILE
    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        NSDictionary<NSString *, GCControllerElement *> *elements = controller.physicalInputProfile.elements;

        /* Provide both axes and analog buttons as SDL axes */
        NSArray *axes = [[[elements allKeys] sortedArrayUsingSelector:@selector(caseInsensitiveCompare:)]
                                         filteredArrayUsingPredicate:[NSPredicate predicateWithBlock:^BOOL(id object, NSDictionary *bindings) {
            GCControllerElement *element;

            if (ElementAlreadyHandled(device, (NSString *)object, elements)) {
                return NO;
            }

            element = elements[object];
            if (element.analog) {
                if ([element isKindOfClass:[GCControllerAxisInput class]] ||
                    [element isKindOfClass:[GCControllerButtonInput class]]) {
                    return YES;
                }
            }
            return NO;
        }]];
        NSArray *buttons = [[[elements allKeys] sortedArrayUsingSelector:@selector(caseInsensitiveCompare:)]
                                            filteredArrayUsingPredicate:[NSPredicate predicateWithBlock:^BOOL(id object, NSDictionary *bindings) {
            GCControllerElement *element;

            if (ElementAlreadyHandled(device, (NSString *)object, elements)) {
                return NO;
            }

            element = elements[object];
            if ([element isKindOfClass:[GCControllerButtonInput class]]) {
                return YES;
            }
            return NO;
        }]];
        /* Explicitly retain the arrays because SDL_JoystickDeviceItem is a
         * struct, and ARC doesn't work with structs. */
        device->naxes = (int)axes.count;
        device->axes = (__bridge NSArray *)CFBridgingRetain(axes);
        device->nbuttons = (int)buttons.count;
        device->buttons = (__bridge NSArray *)CFBridgingRetain(buttons);
        subtype = 4;

#ifdef DEBUG_CONTROLLER_PROFILE
        NSLog(@"Elements used:\n", controller.vendorName);
        for (id key in device->buttons) {
            NSLog(@"\tButton: %@ (%s)\n", key, elements[key].analog ? "analog" : "digital");
        }
        for (id key in device->axes) {
            NSLog(@"\tAxis: %@\n", key);
        }
#endif /* DEBUG_CONTROLLER_PROFILE */

#if TARGET_OS_TV
        /* tvOS turns the menu button into a system gesture, so we grab it here instead */
        if (elements[GCInputButtonMenu] && !elements[@"Button Home"]) {
            device->pause_button_index = [device->buttons indexOfObject:GCInputButtonMenu];
        }
#endif
    } else
#endif
    if (controller.extendedGamepad) {
        GCExtendedGamepad *gamepad = controller.extendedGamepad;
        int nbuttons = 0;
        BOOL has_direct_menu = FALSE;

        /* These buttons are part of the original MFi spec */
        device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_A);
        device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_B);
        device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_X);
        device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_Y);
        device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_LEFTSHOULDER);
        device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_RIGHTSHOULDER);
        nbuttons += 6;

        /* These buttons are available on some newer controllers */
        if (@available(macOS 10.14.1, iOS 12.1, tvOS 12.1, *)) {
            if (gamepad.leftThumbstickButton) {
                device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_LEFTSTICK);
                ++nbuttons;
            }
            if (gamepad.rightThumbstickButton) {
                device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_RIGHTSTICK);
                ++nbuttons;
            }
        }
        if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
            if (gamepad.buttonOptions) {
                device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_BACK);
                ++nbuttons;
            }
        }
        device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_START);
        ++nbuttons;

        if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
            if (gamepad.buttonMenu) {
                has_direct_menu = TRUE;
            }
        }
#if TARGET_OS_TV
        /* The single menu button isn't very reliable, at least as of tvOS 16.1 */
        if ((device->button_mask & (1 << SDL_CONTROLLER_BUTTON_BACK)) == 0) {
            has_direct_menu = FALSE;
        }
#endif
        if (!has_direct_menu) {
            device->pause_button_index = (nbuttons - 1);
        }

        device->naxes = 6; /* 2 thumbsticks and 2 triggers */
        device->nhats = 1; /* d-pad */
        device->nbuttons = nbuttons;

    } else if (controller.gamepad) {
        int nbuttons = 0;

        /* These buttons are part of the original MFi spec */
        device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_A);
        device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_B);
        device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_X);
        device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_Y);
        device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_LEFTSHOULDER);
        device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_RIGHTSHOULDER);
        device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_START);
        nbuttons += 7;
        device->pause_button_index = (nbuttons - 1);

        device->naxes = 0; /* no traditional analog inputs */
        device->nhats = 1; /* d-pad */
        device->nbuttons = nbuttons;
    }
#if TARGET_OS_TV
    else if (controller.microGamepad) {
        int nbuttons = 0;

        device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_A);
        device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_X); /* Button X on microGamepad */
        device->button_mask |= (1 << SDL_CONTROLLER_BUTTON_B);
        nbuttons += 3;
        device->pause_button_index = (nbuttons - 1);

        device->naxes = 2; /* treat the touch surface as two axes */
        device->nhats = 0; /* apparently the touch surface-as-dpad is buggy */
        device->nbuttons = nbuttons;

        controller.microGamepad.allowsRotation = SDL_GetHintBoolean(SDL_HINT_APPLE_TV_REMOTE_ALLOW_ROTATION, SDL_FALSE);
    }
#endif
    else {
        /* We don't know how to get input events from this device */
        return SDL_FALSE;
    }

    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        signature = 0;
        signature = SDL_crc16(signature, device->name, SDL_strlen(device->name));
        for (id key in device->axes) {
            const char *string = ((NSString *)key).UTF8String;
            signature = SDL_crc16(signature, string, SDL_strlen(string));
        }
        for (id key in device->buttons) {
            const char *string = ((NSString *)key).UTF8String;
            signature = SDL_crc16(signature, string, SDL_strlen(string));
        }
    } else {
        signature = device->button_mask;
    }
    device->guid = SDL_CreateJoystickGUID(SDL_HARDWARE_BUS_BLUETOOTH, vendor, product, signature, NULL, name, 'm', subtype);

    if (SDL_ShouldIgnoreJoystick(name, device->guid)) {
        return SDL_FALSE;
    }

    /* This will be set when the first button press of the controller is
     * detected. */
    controller.playerIndex = -1;
    return TRUE;
}
#endif /* SDL_JOYSTICK_MFI */

#if defined(SDL_JOYSTICK_iOS_ACCELEROMETER) || defined(SDL_JOYSTICK_MFI)
static void IOS_AddJoystickDevice(GCController *controller, SDL_bool accelerometer)
{
    SDL_JoystickDeviceItem *device = deviceList;

    while (device != NULL) {
        if (device->controller == controller) {
            return;
        }
        device = device->next;
    }

    device = (SDL_JoystickDeviceItem *)SDL_calloc(1, sizeof(SDL_JoystickDeviceItem));
    if (device == NULL) {
        return;
    }

    device->accelerometer = accelerometer;
    device->instance_id = SDL_GetNextJoystickInstanceID();
    device->pause_button_index = -1;

    if (accelerometer) {
#ifdef SDL_JOYSTICK_iOS_ACCELEROMETER
        device->name = SDL_strdup(accelerometerName);
        device->naxes = 3; /* Device acceleration in the x, y, and z axes. */
        device->nhats = 0;
        device->nbuttons = 0;

        /* Use the accelerometer name as a GUID. */
        SDL_memcpy(&device->guid.data, device->name, SDL_min(sizeof(SDL_JoystickGUID), SDL_strlen(device->name)));
#else
        SDL_free(device);
        return;
#endif /* SDL_JOYSTICK_iOS_ACCELEROMETER */
    } else if (controller) {
#ifdef SDL_JOYSTICK_MFI
        if (!IOS_AddMFIJoystickDevice(device, controller)) {
            SDL_free(device->name);
            SDL_free(device);
            return;
        }
#else
        SDL_free(device);
        return;
#endif /* SDL_JOYSTICK_MFI */
    }

    if (deviceList == NULL) {
        deviceList = device;
    } else {
        SDL_JoystickDeviceItem *lastdevice = deviceList;
        while (lastdevice->next != NULL) {
            lastdevice = lastdevice->next;
        }
        lastdevice->next = device;
    }

    ++numjoysticks;

    SDL_PrivateJoystickAdded(device->instance_id);
}
#endif /* SDL_JOYSTICK_iOS_ACCELEROMETER || SDL_JOYSTICK_MFI */

static SDL_JoystickDeviceItem *IOS_RemoveJoystickDevice(SDL_JoystickDeviceItem *device)
{
    SDL_JoystickDeviceItem *prev = NULL;
    SDL_JoystickDeviceItem *next = NULL;
    SDL_JoystickDeviceItem *item = deviceList;

    if (device == NULL) {
        return NULL;
    }

    next = device->next;

    while (item != NULL) {
        if (item == device) {
            break;
        }
        prev = item;
        item = item->next;
    }

    /* Unlink the device item from the device list. */
    if (prev) {
        prev->next = device->next;
    } else if (device == deviceList) {
        deviceList = device->next;
    }

    if (device->joystick) {
        device->joystick->hwdata = NULL;
    }

#ifdef SDL_JOYSTICK_MFI
    @autoreleasepool {
        /* These were explicitly retained in the struct, so they should be explicitly released before freeing the struct. */
        if (device->controller) {
            GCController *controller = CFBridgingRelease((__bridge CFTypeRef)(device->controller));
            controller.controllerPausedHandler = nil;
            device->controller = nil;
        }
        if (device->axes) {
            CFRelease((__bridge CFTypeRef)device->axes);
            device->axes = nil;
        }
        if (device->buttons) {
            CFRelease((__bridge CFTypeRef)device->buttons);
            device->buttons = nil;
        }
    }
#endif /* SDL_JOYSTICK_MFI */

    --numjoysticks;

    SDL_PrivateJoystickRemoved(device->instance_id);

    SDL_free(device->name);
    SDL_free(device);

    return next;
}

#if TARGET_OS_TV
static void SDLCALL SDL_AppleTVRemoteRotationHintChanged(void *udata, const char *name, const char *oldValue, const char *newValue)
{
    BOOL allowRotation = newValue != NULL && *newValue != '0';

    @autoreleasepool {
        for (GCController *controller in [GCController controllers]) {
            if (controller.microGamepad) {
                controller.microGamepad.allowsRotation = allowRotation;
            }
        }
    }
}
#endif /* TARGET_OS_TV */

static int IOS_JoystickInit(void)
{
    if (!SDL_GetHintBoolean(SDL_HINT_JOYSTICK_MFI, SDL_TRUE)) {
        return 0;
    }

#if defined(__MACOSX__)
#if _SDL_HAS_BUILTIN(__builtin_available)
    if (@available(macOS 10.16, *)) {
        /* Continue with initialization on macOS 11+ */
    } else {
        return 0;
    }
#else
    /* No @available, must be an older macOS version */
    return 0;
#endif
#endif

    @autoreleasepool {
#ifdef SDL_JOYSTICK_MFI
        NSNotificationCenter *center;
#endif
#ifdef SDL_JOYSTICK_iOS_ACCELEROMETER
        if (SDL_GetHintBoolean(SDL_HINT_ACCELEROMETER_AS_JOYSTICK, SDL_TRUE)) {
            /* Default behavior, accelerometer as joystick */
            IOS_AddJoystickDevice(nil, SDL_TRUE);
        }
#endif

#ifdef SDL_JOYSTICK_MFI
        /* GameController.framework was added in iOS 7. */
        if (![GCController class]) {
            return 0;
        }

        /* For whatever reason, this always returns an empty array on
         macOS 11.0.1 */
        for (GCController *controller in [GCController controllers]) {
            IOS_AddJoystickDevice(controller, SDL_FALSE);
        }

#if TARGET_OS_TV
        SDL_AddHintCallback(SDL_HINT_APPLE_TV_REMOTE_ALLOW_ROTATION,
                            SDL_AppleTVRemoteRotationHintChanged, NULL);
#endif /* TARGET_OS_TV */

        center = [NSNotificationCenter defaultCenter];

        connectObserver = [center addObserverForName:GCControllerDidConnectNotification
                                              object:nil
                                               queue:nil
                                          usingBlock:^(NSNotification *note) {
                                            GCController *controller = note.object;
                                            SDL_LockJoysticks();
                                            IOS_AddJoystickDevice(controller, SDL_FALSE);
                                            SDL_UnlockJoysticks();
                                          }];

        disconnectObserver = [center addObserverForName:GCControllerDidDisconnectNotification
                                                 object:nil
                                                  queue:nil
                                             usingBlock:^(NSNotification *note) {
                                               GCController *controller = note.object;
                                               SDL_JoystickDeviceItem *device;
                                               SDL_LockJoysticks();
                                               for (device = deviceList; device != NULL; device = device->next) {
                                                   if (device->controller == controller) {
                                                       IOS_RemoveJoystickDevice(device);
                                                       break;
                                                   }
                                               }
                                               SDL_UnlockJoysticks();
                                             }];
#endif /* SDL_JOYSTICK_MFI */
    }

    return 0;
}

static int IOS_JoystickGetCount(void)
{
    return numjoysticks;
}

static void IOS_JoystickDetect(void)
{
}

static const char *IOS_JoystickGetDeviceName(int device_index)
{
    SDL_JoystickDeviceItem *device = GetDeviceForIndex(device_index);
    return device ? device->name : "Unknown";
}

static const char *IOS_JoystickGetDevicePath(int device_index)
{
    return NULL;
}

static int IOS_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    return -1;
}

static int IOS_JoystickGetDevicePlayerIndex(int device_index)
{
#ifdef SDL_JOYSTICK_MFI
    SDL_JoystickDeviceItem *device = GetDeviceForIndex(device_index);
    if (device && device->controller) {
        return (int)device->controller.playerIndex;
    }
#endif
    return -1;
}

static void IOS_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
#ifdef SDL_JOYSTICK_MFI
    SDL_JoystickDeviceItem *device = GetDeviceForIndex(device_index);
    if (device && device->controller) {
        device->controller.playerIndex = player_index;
    }
#endif
}

static SDL_JoystickGUID IOS_JoystickGetDeviceGUID(int device_index)
{
    SDL_JoystickDeviceItem *device = GetDeviceForIndex(device_index);
    SDL_JoystickGUID guid;
    if (device) {
        guid = device->guid;
    } else {
        SDL_zero(guid);
    }
    return guid;
}

static SDL_JoystickID IOS_JoystickGetDeviceInstanceID(int device_index)
{
    SDL_JoystickDeviceItem *device = GetDeviceForIndex(device_index);
    return device ? device->instance_id : -1;
}

static int IOS_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    SDL_JoystickDeviceItem *device = GetDeviceForIndex(device_index);
    if (device == NULL) {
        return SDL_SetError("Could not open Joystick: no hardware device for the specified index");
    }

    joystick->hwdata = device;
    joystick->instance_id = device->instance_id;

    joystick->naxes = device->naxes;
    joystick->nhats = device->nhats;
    joystick->nbuttons = device->nbuttons;
    joystick->nballs = 0;

    if (device->has_dualshock_touchpad) {
        SDL_PrivateJoystickAddTouchpad(joystick, 2);
    }

    device->joystick = joystick;

    @autoreleasepool {
        if (device->accelerometer) {
#ifdef SDL_JOYSTICK_iOS_ACCELEROMETER
            if (motionManager == nil) {
                motionManager = [[CMMotionManager alloc] init];
            }

            /* Shorter times between updates can significantly increase CPU usage. */
            motionManager.accelerometerUpdateInterval = 0.1;
            [motionManager startAccelerometerUpdates];
#endif
        } else {
#ifdef SDL_JOYSTICK_MFI
            if (device->pause_button_index >= 0) {
                GCController *controller = device->controller;
                controller.controllerPausedHandler = ^(GCController *c) {
                  if (joystick->hwdata) {
                      joystick->hwdata->pause_button_pressed = SDL_GetTicks();
                  }
                };
            }

#ifdef ENABLE_MFI_SENSORS
            if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
                GCController *controller = joystick->hwdata->controller;
                GCMotion *motion = controller.motion;
                if (motion && motion.hasRotationRate) {
                    SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_GYRO, 0.0f);
                }
                if (motion && motion.hasGravityAndUserAcceleration) {
                    SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_ACCEL, 0.0f);
                }
            }
#endif /* ENABLE_MFI_SENSORS */

#ifdef ENABLE_MFI_SYSTEM_GESTURE_STATE
            if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
                GCController *controller = joystick->hwdata->controller;
                for (id key in controller.physicalInputProfile.buttons) {
                    GCControllerButtonInput *button = controller.physicalInputProfile.buttons[key];
                    if ([button isBoundToSystemGesture]) {
                        button.preferredSystemGestureState = GCSystemGestureStateDisabled;
                    }
                }
            }
#endif /* ENABLE_MFI_SYSTEM_GESTURE_STATE */

#endif /* SDL_JOYSTICK_MFI */
        }
    }
    if (device->is_siri_remote) {
        ++SDL_AppleTVRemoteOpenedAsJoystick;
    }

    return 0;
}

static void IOS_AccelerometerUpdate(SDL_Joystick *joystick)
{
#ifdef SDL_JOYSTICK_iOS_ACCELEROMETER
    const float maxgforce = SDL_IPHONE_MAX_GFORCE;
    const SInt16 maxsint16 = 0x7FFF;
    CMAcceleration accel;

    @autoreleasepool {
        if (!motionManager.isAccelerometerActive) {
            return;
        }

        accel = motionManager.accelerometerData.acceleration;
    }

    /*
     Convert accelerometer data from floating point to Sint16, which is what
     the joystick system expects.

     To do the conversion, the data is first clamped onto the interval
     [-SDL_IPHONE_MAX_G_FORCE, SDL_IPHONE_MAX_G_FORCE], then the data is multiplied
     by MAX_SINT16 so that it is mapped to the full range of an Sint16.

     You can customize the clamped range of this function by modifying the
     SDL_IPHONE_MAX_GFORCE macro in SDL_config_iphoneos.h.

     Once converted to Sint16, the accelerometer data no longer has coherent
     units. You can convert the data back to units of g-force by multiplying
     it in your application's code by SDL_IPHONE_MAX_GFORCE / 0x7FFF.
     */

    /* clamp the data */
    accel.x = SDL_clamp(accel.x, -maxgforce, maxgforce);
    accel.y = SDL_clamp(accel.y, -maxgforce, maxgforce);
    accel.z = SDL_clamp(accel.z, -maxgforce, maxgforce);

    /* pass in data mapped to range of SInt16 */
    SDL_PrivateJoystickAxis(joystick, 0, (accel.x / maxgforce) * maxsint16);
    SDL_PrivateJoystickAxis(joystick, 1, -(accel.y / maxgforce) * maxsint16);
    SDL_PrivateJoystickAxis(joystick, 2, (accel.z / maxgforce) * maxsint16);
#endif /* SDL_JOYSTICK_iOS_ACCELEROMETER */
}

#ifdef SDL_JOYSTICK_MFI
static Uint8 IOS_MFIJoystickHatStateForDPad(GCControllerDirectionPad *dpad)
{
    Uint8 hat = 0;

    if (dpad.up.isPressed) {
        hat |= SDL_HAT_UP;
    } else if (dpad.down.isPressed) {
        hat |= SDL_HAT_DOWN;
    }

    if (dpad.left.isPressed) {
        hat |= SDL_HAT_LEFT;
    } else if (dpad.right.isPressed) {
        hat |= SDL_HAT_RIGHT;
    }

    if (hat == 0) {
        return SDL_HAT_CENTERED;
    }

    return hat;
}
#endif

static void IOS_MFIJoystickUpdate(SDL_Joystick *joystick)
{
#ifdef SDL_JOYSTICK_MFI
    @autoreleasepool {
        SDL_JoystickDeviceItem *device = joystick->hwdata;
        GCController *controller = device->controller;
        Uint8 hatstate = SDL_HAT_CENTERED;
        int i;

#if defined(DEBUG_CONTROLLER_STATE) && defined(ENABLE_PHYSICAL_INPUT_PROFILE)
        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            if (controller.physicalInputProfile) {
                for (id key in controller.physicalInputProfile.buttons) {
                    GCControllerButtonInput *button = controller.physicalInputProfile.buttons[key];
                    if (button.isPressed)
                        NSLog(@"Button %@ = %s\n", key, button.isPressed ? "pressed" : "released");
                }
                for (id key in controller.physicalInputProfile.axes) {
                    GCControllerAxisInput *axis = controller.physicalInputProfile.axes[key];
                    if (axis.value != 0.0f)
                        NSLog(@"Axis %@ = %g\n", key, axis.value);
                }
                for (id key in controller.physicalInputProfile.dpads) {
                    GCControllerDirectionPad *dpad = controller.physicalInputProfile.dpads[key];
                    if (dpad.up.isPressed || dpad.down.isPressed || dpad.left.isPressed || dpad.right.isPressed) {
                        NSLog(@"Hat %@ =%s%s%s%s\n", key,
                            dpad.up.isPressed ? " UP" : "",
                            dpad.down.isPressed ? " DOWN" : "",
                            dpad.left.isPressed ? " LEFT" : "",
                            dpad.right.isPressed ? " RIGHT" : "");
                    }
                }
            }
        }
#endif /* DEBUG_CONTROLLER_STATE */

#ifdef ENABLE_PHYSICAL_INPUT_PROFILE
        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            NSDictionary<NSString *, GCControllerElement *> *elements = controller.physicalInputProfile.elements;
            NSDictionary<NSString *, GCControllerButtonInput *> *buttons = controller.physicalInputProfile.buttons;
            int axis = 0;
            int button = 0;

            for (id key in device->axes) {
                Sint16 value;
                GCControllerElement *element = elements[key];
                if ([element isKindOfClass:[GCControllerAxisInput class]]) {
                    value = (Sint16)([(GCControllerAxisInput *)element value] * 32767);
                } else {
                    value = (Sint16)([(GCControllerButtonInput *)element value] * 32767);
                }
                SDL_PrivateJoystickAxis(joystick, axis++, value);
            }

            for (id key in device->buttons) {
                Uint8 value;
                if (button == device->pause_button_index) {
                    value = (device->pause_button_pressed > 0);
                } else {
                    value = buttons[key].isPressed;
                }
                SDL_PrivateJoystickButton(joystick, button++, value);
            }
        } else
#endif
        if (controller.extendedGamepad) {
            SDL_bool isstack;
            GCExtendedGamepad *gamepad = controller.extendedGamepad;

            /* Axis order matches the XInput Windows mappings. */
            Sint16 axes[] = {
                (Sint16)(gamepad.leftThumbstick.xAxis.value * 32767),
                (Sint16)(gamepad.leftThumbstick.yAxis.value * -32767),
                (Sint16)((gamepad.leftTrigger.value * 65535) - 32768),
                (Sint16)(gamepad.rightThumbstick.xAxis.value * 32767),
                (Sint16)(gamepad.rightThumbstick.yAxis.value * -32767),
                (Sint16)((gamepad.rightTrigger.value * 65535) - 32768),
            };

            /* Button order matches the XInput Windows mappings. */
            Uint8 *buttons = SDL_small_alloc(Uint8, joystick->nbuttons, &isstack);
            int button_count = 0;

            if (buttons == NULL) {
                SDL_OutOfMemory();
                return;
            }

            /* These buttons are part of the original MFi spec */
            buttons[button_count++] = gamepad.buttonA.isPressed;
            buttons[button_count++] = gamepad.buttonB.isPressed;
            buttons[button_count++] = gamepad.buttonX.isPressed;
            buttons[button_count++] = gamepad.buttonY.isPressed;
            buttons[button_count++] = gamepad.leftShoulder.isPressed;
            buttons[button_count++] = gamepad.rightShoulder.isPressed;

            /* These buttons are available on some newer controllers */
            if (@available(macOS 10.14.1, iOS 12.1, tvOS 12.1, *)) {
                if (device->button_mask & (1 << SDL_CONTROLLER_BUTTON_LEFTSTICK)) {
                    buttons[button_count++] = gamepad.leftThumbstickButton.isPressed;
                }
                if (device->button_mask & (1 << SDL_CONTROLLER_BUTTON_RIGHTSTICK)) {
                    buttons[button_count++] = gamepad.rightThumbstickButton.isPressed;
                }
            }
            if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
                if (device->button_mask & (1 << SDL_CONTROLLER_BUTTON_BACK)) {
                    buttons[button_count++] = gamepad.buttonOptions.isPressed;
                }
            }
            if (device->button_mask & (1 << SDL_CONTROLLER_BUTTON_START)) {
                if (device->pause_button_index >= 0) {
                    /* Guaranteed if buttonMenu is not supported on this OS */
                    buttons[button_count++] = (device->pause_button_pressed > 0);
                } else {
                    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
                        buttons[button_count++] = gamepad.buttonMenu.isPressed;
                    }
                }
            }

            hatstate = IOS_MFIJoystickHatStateForDPad(gamepad.dpad);

            for (i = 0; i < SDL_arraysize(axes); i++) {
                SDL_PrivateJoystickAxis(joystick, i, axes[i]);
            }

            for (i = 0; i < button_count; i++) {
                SDL_PrivateJoystickButton(joystick, i, buttons[i]);
            }

            SDL_small_free(buttons, isstack);
        } else if (controller.gamepad) {
            SDL_bool isstack;
            GCGamepad *gamepad = controller.gamepad;

            /* Button order matches the XInput Windows mappings. */
            Uint8 *buttons = SDL_small_alloc(Uint8, joystick->nbuttons, &isstack);
            int button_count = 0;

            if (buttons == NULL) {
                SDL_OutOfMemory();
                return;
            }

            buttons[button_count++] = gamepad.buttonA.isPressed;
            buttons[button_count++] = gamepad.buttonB.isPressed;
            buttons[button_count++] = gamepad.buttonX.isPressed;
            buttons[button_count++] = gamepad.buttonY.isPressed;
            buttons[button_count++] = gamepad.leftShoulder.isPressed;
            buttons[button_count++] = gamepad.rightShoulder.isPressed;
            buttons[button_count++] = (device->pause_button_pressed > 0);

            hatstate = IOS_MFIJoystickHatStateForDPad(gamepad.dpad);

            for (i = 0; i < button_count; i++) {
                SDL_PrivateJoystickButton(joystick, i, buttons[i]);
            }

            SDL_small_free(buttons, isstack);
        }
#if TARGET_OS_TV
        else if (controller.microGamepad) {
            Uint8 buttons[joystick->nbuttons];
            int button_count = 0;
            GCMicroGamepad *gamepad = controller.microGamepad;

            Sint16 axes[] = {
                (Sint16)(gamepad.dpad.xAxis.value * 32767),
                (Sint16)(gamepad.dpad.yAxis.value * -32767),
            };

            for (i = 0; i < SDL_arraysize(axes); i++) {
                SDL_PrivateJoystickAxis(joystick, i, axes[i]);
            }

            buttons[button_count++] = gamepad.buttonA.isPressed;
            buttons[button_count++] = gamepad.buttonX.isPressed;
            buttons[button_count++] = (device->pause_button_pressed > 0);

            for (i = 0; i < button_count; i++) {
                SDL_PrivateJoystickButton(joystick, i, buttons[i]);
            }
        }
#endif /* TARGET_OS_TV */

        if (joystick->nhats > 0) {
            SDL_PrivateJoystickHat(joystick, 0, hatstate);
        }

        if (device->pause_button_pressed) {
            /* The pause callback is instantaneous, so we extend the duration to allow "holding down" by pressing it repeatedly */
            const int PAUSE_BUTTON_PRESS_DURATION_MS = 250;
            if (SDL_TICKS_PASSED(SDL_GetTicks(), device->pause_button_pressed + PAUSE_BUTTON_PRESS_DURATION_MS)) {
                device->pause_button_pressed = 0;
            }
        }

#ifdef ENABLE_PHYSICAL_INPUT_PROFILE
        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            if (device->has_dualshock_touchpad) {
                GCControllerDirectionPad *dpad;

                dpad = controller.physicalInputProfile.dpads[GCInputDualShockTouchpadOne];
                if (dpad.xAxis.value || dpad.yAxis.value) {
                    SDL_PrivateJoystickTouchpad(joystick, 0, 0, SDL_PRESSED, (1.0f + dpad.xAxis.value) * 0.5f, 1.0f - (1.0f + dpad.yAxis.value) * 0.5f, 1.0f);
                } else {
                    SDL_PrivateJoystickTouchpad(joystick, 0, 0, SDL_RELEASED, 0.0f, 0.0f, 1.0f);
                }

                dpad = controller.physicalInputProfile.dpads[GCInputDualShockTouchpadTwo];
                if (dpad.xAxis.value || dpad.yAxis.value) {
                    SDL_PrivateJoystickTouchpad(joystick, 0, 1, SDL_PRESSED, (1.0f + dpad.xAxis.value) * 0.5f, 1.0f - (1.0f + dpad.yAxis.value) * 0.5f, 1.0f);
                } else {
                    SDL_PrivateJoystickTouchpad(joystick, 0, 1, SDL_RELEASED, 0.0f, 0.0f, 1.0f);
                }
            }
        }
#endif /* ENABLE_PHYSICAL_INPUT_PROFILE */

#ifdef ENABLE_MFI_SENSORS
            if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
                GCMotion *motion = controller.motion;
                if (motion && motion.sensorsActive) {
                    float data[3];

                    if (motion.hasRotationRate) {
                        GCRotationRate rate = motion.rotationRate;
                        data[0] = rate.x;
                        data[1] = rate.z;
                        data[2] = -rate.y;
                        SDL_PrivateJoystickSensor(joystick, SDL_SENSOR_GYRO, 0, data, 3);
                    }
                    if (motion.hasGravityAndUserAcceleration) {
                        GCAcceleration accel = motion.acceleration;
                        data[0] = -accel.x * SDL_STANDARD_GRAVITY;
                        data[1] = -accel.y * SDL_STANDARD_GRAVITY;
                        data[2] = -accel.z * SDL_STANDARD_GRAVITY;
                        SDL_PrivateJoystickSensor(joystick, SDL_SENSOR_ACCEL, 0, data, 3);
                    }
                }
            }
#endif /* ENABLE_MFI_SENSORS */

#ifdef ENABLE_MFI_BATTERY
        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            GCDeviceBattery *battery = controller.battery;
            if (battery) {
                SDL_JoystickPowerLevel ePowerLevel = SDL_JOYSTICK_POWER_UNKNOWN;

                switch (battery.batteryState) {
                case GCDeviceBatteryStateDischarging:
                {
                    float power_level = battery.batteryLevel;
                    if (power_level <= 0.05f) {
                        ePowerLevel = SDL_JOYSTICK_POWER_EMPTY;
                    } else if (power_level <= 0.20f) {
                        ePowerLevel = SDL_JOYSTICK_POWER_LOW;
                    } else if (power_level <= 0.70f) {
                        ePowerLevel = SDL_JOYSTICK_POWER_MEDIUM;
                    } else {
                        ePowerLevel = SDL_JOYSTICK_POWER_FULL;
                    }
                } break;
                case GCDeviceBatteryStateCharging:
                    ePowerLevel = SDL_JOYSTICK_POWER_WIRED;
                    break;
                case GCDeviceBatteryStateFull:
                    ePowerLevel = SDL_JOYSTICK_POWER_FULL;
                    break;
                default:
                    break;
                }

                SDL_PrivateJoystickBatteryLevel(joystick, ePowerLevel);
            }
        }
#endif /* ENABLE_MFI_BATTERY */
    }
#endif /* SDL_JOYSTICK_MFI */
}

#ifdef ENABLE_MFI_RUMBLE

@interface SDL_RumbleMotor : NSObject
@property(nonatomic, strong) CHHapticEngine *engine API_AVAILABLE(macos(10.16), ios(13.0), tvos(14.0));
@property(nonatomic, strong) id<CHHapticPatternPlayer> player API_AVAILABLE(macos(10.16), ios(13.0), tvos(14.0));
@property bool active;
@end

@implementation SDL_RumbleMotor
{
}

- (void)cleanup
{
    @autoreleasepool {
        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            if (self.player != nil) {
                [self.player cancelAndReturnError:nil];
                self.player = nil;
            }
            if (self.engine != nil) {
                [self.engine stopWithCompletionHandler:nil];
                self.engine = nil;
            }
        }
    }
}

- (int)setIntensity:(float)intensity
{
    @autoreleasepool {
        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            NSError *error = nil;
            CHHapticDynamicParameter *param;

            if (self.engine == nil) {
                return SDL_SetError("Haptics engine was stopped");
            }

            if (intensity == 0.0f) {
                if (self.player && self.active) {
                    [self.player stopAtTime:0 error:&error];
                }
                self.active = false;
                return 0;
            }

            if (self.player == nil) {
                CHHapticEventParameter *event_param = [[CHHapticEventParameter alloc] initWithParameterID:CHHapticEventParameterIDHapticIntensity value:1.0f];
                CHHapticEvent *event = [[CHHapticEvent alloc] initWithEventType:CHHapticEventTypeHapticContinuous parameters:[NSArray arrayWithObjects:event_param, nil] relativeTime:0 duration:GCHapticDurationInfinite];
                CHHapticPattern *pattern = [[CHHapticPattern alloc] initWithEvents:[NSArray arrayWithObject:event] parameters:[[NSArray alloc] init] error:&error];
                if (error != nil) {
                    return SDL_SetError("Couldn't create haptic pattern: %s", [error.localizedDescription UTF8String]);
                }

                self.player = [self.engine createPlayerWithPattern:pattern error:&error];
                if (error != nil) {
                    return SDL_SetError("Couldn't create haptic player: %s", [error.localizedDescription UTF8String]);
                }
                self.active = false;
            }

            param = [[CHHapticDynamicParameter alloc] initWithParameterID:CHHapticDynamicParameterIDHapticIntensityControl value:intensity relativeTime:0];
            [self.player sendParameters:[NSArray arrayWithObject:param] atTime:0 error:&error];
            if (error != nil) {
                return SDL_SetError("Couldn't update haptic player: %s", [error.localizedDescription UTF8String]);
            }

            if (!self.active) {
                [self.player startAtTime:0 error:&error];
                self.active = true;
            }
        }

        return 0;
    }
}

- (id)initWithController:(GCController *)controller locality:(GCHapticsLocality)locality API_AVAILABLE(macos(10.16), ios(14.0), tvos(14.0))
{
    @autoreleasepool {
        NSError *error;
        __weak __typeof(self) weakSelf;
        self = [super init];
        weakSelf = self;

        self.engine = [controller.haptics createEngineWithLocality:locality];
        if (self.engine == nil) {
            SDL_SetError("Couldn't create haptics engine");
            return nil;
        }

        [self.engine startAndReturnError:&error];
        if (error != nil) {
            SDL_SetError("Couldn't start haptics engine");
            return nil;
        }

        self.engine.stoppedHandler = ^(CHHapticEngineStoppedReason stoppedReason) {
          SDL_RumbleMotor *_this = weakSelf;
          if (_this == nil) {
              return;
          }

          _this.player = nil;
          _this.engine = nil;
        };
        self.engine.resetHandler = ^{
          SDL_RumbleMotor *_this = weakSelf;
          if (_this == nil) {
              return;
          }

          _this.player = nil;
          [_this.engine startAndReturnError:nil];
        };

        return self;
    }
}

@end

@interface SDL_RumbleContext : NSObject
@property(nonatomic, strong) SDL_RumbleMotor *lowFrequencyMotor;
@property(nonatomic, strong) SDL_RumbleMotor *highFrequencyMotor;
@property(nonatomic, strong) SDL_RumbleMotor *leftTriggerMotor;
@property(nonatomic, strong) SDL_RumbleMotor *rightTriggerMotor;
@end

@implementation SDL_RumbleContext
{
}

- (id)initWithLowFrequencyMotor:(SDL_RumbleMotor *)low_frequency_motor
             HighFrequencyMotor:(SDL_RumbleMotor *)high_frequency_motor
               LeftTriggerMotor:(SDL_RumbleMotor *)left_trigger_motor
              RightTriggerMotor:(SDL_RumbleMotor *)right_trigger_motor
{
    self = [super init];
    self.lowFrequencyMotor = low_frequency_motor;
    self.highFrequencyMotor = high_frequency_motor;
    self.leftTriggerMotor = left_trigger_motor;
    self.rightTriggerMotor = right_trigger_motor;
    return self;
}

- (int)rumbleWithLowFrequency:(Uint16)low_frequency_rumble andHighFrequency:(Uint16)high_frequency_rumble
{
    int result = 0;

    result += [self.lowFrequencyMotor setIntensity:((float)low_frequency_rumble / 65535.0f)];
    result += [self.highFrequencyMotor setIntensity:((float)high_frequency_rumble / 65535.0f)];
    return ((result < 0) ? -1 : 0);
}

- (int)rumbleLeftTrigger:(Uint16)left_rumble andRightTrigger:(Uint16)right_rumble
{
    int result = 0;

    if (self.leftTriggerMotor && self.rightTriggerMotor) {
        result += [self.leftTriggerMotor setIntensity:((float)left_rumble / 65535.0f)];
        result += [self.rightTriggerMotor setIntensity:((float)right_rumble / 65535.0f)];
    } else {
        result = SDL_Unsupported();
    }
    return ((result < 0) ? -1 : 0);
}

- (void)cleanup
{
    [self.lowFrequencyMotor cleanup];
    [self.highFrequencyMotor cleanup];
}

@end

static SDL_RumbleContext *IOS_JoystickInitRumble(GCController *controller)
{
    @autoreleasepool {
        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            SDL_RumbleMotor *low_frequency_motor = [[SDL_RumbleMotor alloc] initWithController:controller locality:GCHapticsLocalityLeftHandle];
            SDL_RumbleMotor *high_frequency_motor = [[SDL_RumbleMotor alloc] initWithController:controller locality:GCHapticsLocalityRightHandle];
            SDL_RumbleMotor *left_trigger_motor = [[SDL_RumbleMotor alloc] initWithController:controller locality:GCHapticsLocalityLeftTrigger];
            SDL_RumbleMotor *right_trigger_motor = [[SDL_RumbleMotor alloc] initWithController:controller locality:GCHapticsLocalityRightTrigger];
            if (low_frequency_motor && high_frequency_motor) {
                return [[SDL_RumbleContext alloc] initWithLowFrequencyMotor:low_frequency_motor
                                                         HighFrequencyMotor:high_frequency_motor
                                                           LeftTriggerMotor:left_trigger_motor
                                                          RightTriggerMotor:right_trigger_motor];
            }
        }
    }
    return nil;
}

#endif /* ENABLE_MFI_RUMBLE */

static int IOS_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
#ifdef ENABLE_MFI_RUMBLE
    SDL_JoystickDeviceItem *device = joystick->hwdata;

    if (device == NULL) {
        return SDL_SetError("Controller is no longer connected");
    }

    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        if (!device->rumble && device->controller && device->controller.haptics) {
            SDL_RumbleContext *rumble = IOS_JoystickInitRumble(device->controller);
            if (rumble) {
                device->rumble = (void *)CFBridgingRetain(rumble);
            }
        }
    }

    if (device->rumble) {
        SDL_RumbleContext *rumble = (__bridge SDL_RumbleContext *)device->rumble;
        return [rumble rumbleWithLowFrequency:low_frequency_rumble andHighFrequency:high_frequency_rumble];
    } else {
        return SDL_Unsupported();
    }
#else
    return SDL_Unsupported();
#endif
}

static int IOS_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
#ifdef ENABLE_MFI_RUMBLE
    SDL_JoystickDeviceItem *device = joystick->hwdata;

    if (device == NULL) {
        return SDL_SetError("Controller is no longer connected");
    }

    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        if (!device->rumble && device->controller && device->controller.haptics) {
            SDL_RumbleContext *rumble = IOS_JoystickInitRumble(device->controller);
            if (rumble) {
                device->rumble = (void *)CFBridgingRetain(rumble);
            }
        }
    }

    if (device->rumble) {
        SDL_RumbleContext *rumble = (__bridge SDL_RumbleContext *)device->rumble;
        return [rumble rumbleLeftTrigger:left_rumble andRightTrigger:right_rumble];
    } else {
        return SDL_Unsupported();
    }
#else
    return SDL_Unsupported();
#endif
}

static Uint32 IOS_JoystickGetCapabilities(SDL_Joystick *joystick)
{
    Uint32 result = 0;

#if defined(ENABLE_MFI_LIGHT) || defined(ENABLE_MFI_RUMBLE)
    @autoreleasepool {
        SDL_JoystickDeviceItem *device = joystick->hwdata;

        if (device == NULL) {
            return 0;
        }

        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            GCController *controller = device->controller;
#ifdef ENABLE_MFI_LIGHT
            if (controller.light) {
                result |= SDL_JOYCAP_LED;
            }
#endif

#ifdef ENABLE_MFI_RUMBLE
            if (controller.haptics) {
                for (GCHapticsLocality locality in controller.haptics.supportedLocalities) {
                    if ([locality isEqualToString:GCHapticsLocalityHandles]) {
                        result |= SDL_JOYCAP_RUMBLE;
                    } else if ([locality isEqualToString:GCHapticsLocalityTriggers]) {
                        result |= SDL_JOYCAP_RUMBLE_TRIGGERS;
                    }
                }
            }
#endif
        }
    }
#endif /* ENABLE_MFI_LIGHT || ENABLE_MFI_RUMBLE */

    return result;
}

static int IOS_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
#ifdef ENABLE_MFI_LIGHT
    @autoreleasepool {
        SDL_JoystickDeviceItem *device = joystick->hwdata;

        if (device == NULL) {
            return SDL_SetError("Controller is no longer connected");
        }

        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            GCController *controller = device->controller;
            GCDeviceLight *light = controller.light;
            if (light) {
                light.color = [[GCColor alloc] initWithRed:(float)red / 255.0f
                                                     green:(float)green / 255.0f
                                                      blue:(float)blue / 255.0f];
                return 0;
            }
        }
    }
#endif /* ENABLE_MFI_LIGHT */

    return SDL_Unsupported();
}

static int IOS_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static int IOS_JoystickSetSensorsEnabled(SDL_Joystick *joystick, SDL_bool enabled)
{
#ifdef ENABLE_MFI_SENSORS
    @autoreleasepool {
        SDL_JoystickDeviceItem *device = joystick->hwdata;

        if (device == NULL) {
            return SDL_SetError("Controller is no longer connected");
        }

        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            GCController *controller = device->controller;
            GCMotion *motion = controller.motion;
            if (motion) {
                motion.sensorsActive = enabled ? YES : NO;
                return 0;
            }
        }
    }
#endif /* ENABLE_MFI_SENSORS */

    return SDL_Unsupported();
}

static void IOS_JoystickUpdate(SDL_Joystick *joystick)
{
    SDL_JoystickDeviceItem *device = joystick->hwdata;

    if (device == NULL) {
        return;
    }

    if (device->accelerometer) {
        IOS_AccelerometerUpdate(joystick);
    } else if (device->controller) {
        IOS_MFIJoystickUpdate(joystick);
    }
}

static void IOS_JoystickClose(SDL_Joystick *joystick)
{
    SDL_JoystickDeviceItem *device = joystick->hwdata;

    if (device == NULL) {
        return;
    }

    device->joystick = NULL;

    @autoreleasepool {
#ifdef ENABLE_MFI_RUMBLE
        if (device->rumble) {
            SDL_RumbleContext *rumble = (__bridge SDL_RumbleContext *)device->rumble;

            [rumble cleanup];
            CFRelease(device->rumble);
            device->rumble = NULL;
        }
#endif /* ENABLE_MFI_RUMBLE */

        if (device->accelerometer) {
#ifdef SDL_JOYSTICK_iOS_ACCELEROMETER
            [motionManager stopAccelerometerUpdates];
#endif
        } else if (device->controller) {
#ifdef SDL_JOYSTICK_MFI
            GCController *controller = device->controller;
            controller.controllerPausedHandler = nil;
            controller.playerIndex = -1;

#ifdef ENABLE_MFI_SYSTEM_GESTURE_STATE
            if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
                for (id key in controller.physicalInputProfile.buttons) {
                    GCControllerButtonInput *button = controller.physicalInputProfile.buttons[key];
                    if ([button isBoundToSystemGesture]) {
                        button.preferredSystemGestureState = GCSystemGestureStateEnabled;
                    }
                }
            }
#endif /* ENABLE_MFI_SYSTEM_GESTURE_STATE */

#endif /* SDL_JOYSTICK_MFI */
        }
    }
    if (device->is_siri_remote) {
        --SDL_AppleTVRemoteOpenedAsJoystick;
    }
}

static void IOS_JoystickQuit(void)
{
    @autoreleasepool {
#ifdef SDL_JOYSTICK_MFI
        NSNotificationCenter *center = [NSNotificationCenter defaultCenter];

        if (connectObserver) {
            [center removeObserver:connectObserver name:GCControllerDidConnectNotification object:nil];
            connectObserver = nil;
        }

        if (disconnectObserver) {
            [center removeObserver:disconnectObserver name:GCControllerDidDisconnectNotification object:nil];
            disconnectObserver = nil;
        }

#if TARGET_OS_TV
        SDL_DelHintCallback(SDL_HINT_APPLE_TV_REMOTE_ALLOW_ROTATION,
                            SDL_AppleTVRemoteRotationHintChanged, NULL);
#endif /* TARGET_OS_TV */
#endif /* SDL_JOYSTICK_MFI */

        while (deviceList != NULL) {
            IOS_RemoveJoystickDevice(deviceList);
        }

#ifdef SDL_JOYSTICK_iOS_ACCELEROMETER
        motionManager = nil;
#endif
    }

    numjoysticks = 0;
}

static SDL_bool IOS_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
#ifdef ENABLE_PHYSICAL_INPUT_PROFILE
    SDL_JoystickDeviceItem *device = GetDeviceForIndex(device_index);
    if (device == NULL) {
        return SDL_FALSE;
    }
    if (device->accelerometer) {
        return SDL_FALSE;
    }

    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        int axis = 0;
        int button = 0;
        for (id key in device->axes) {
            if ([(NSString *)key isEqualToString:@"Left Thumbstick X Axis"] ||
                [(NSString *)key isEqualToString:@"Direction Pad X Axis"]) {
                out->leftx.kind = EMappingKind_Axis;
                out->leftx.target = axis;
            } else if ([(NSString *)key isEqualToString:@"Left Thumbstick Y Axis"] ||
                       [(NSString *)key isEqualToString:@"Direction Pad Y Axis"]) {
                out->lefty.kind = EMappingKind_Axis;
                out->lefty.target = axis;
                out->lefty.axis_reversed = SDL_TRUE;
            } else if ([(NSString *)key isEqualToString:@"Right Thumbstick X Axis"]) {
                out->rightx.kind = EMappingKind_Axis;
                out->rightx.target = axis;
            } else if ([(NSString *)key isEqualToString:@"Right Thumbstick Y Axis"]) {
                out->righty.kind = EMappingKind_Axis;
                out->righty.target = axis;
                out->righty.axis_reversed = SDL_TRUE;
            } else if ([(NSString *)key isEqualToString:GCInputLeftTrigger]) {
                out->lefttrigger.kind = EMappingKind_Axis;
                out->lefttrigger.target = axis;
                out->lefttrigger.half_axis_positive = SDL_TRUE;
            } else if ([(NSString *)key isEqualToString:GCInputRightTrigger]) {
                out->righttrigger.kind = EMappingKind_Axis;
                out->righttrigger.target = axis;
                out->righttrigger.half_axis_positive = SDL_TRUE;
            }
            ++axis;
        }

        for (id key in device->buttons) {
            SDL_InputMapping *mapping = NULL;

            if ([(NSString *)key isEqualToString:GCInputButtonA]) {
                if (device->is_siri_remote > 1) {
                    /* GCInputButtonA is triggered for any D-Pad press, ignore it in favor of "Button Center" */
                } else {
                    mapping = &out->a;
                }
            } else if ([(NSString *)key isEqualToString:GCInputButtonB]) {
                if (device->is_switch_joyconL || device->is_switch_joyconR) {
                    mapping = &out->x;
                } else {
                    mapping = &out->b;
                }
            } else if ([(NSString *)key isEqualToString:GCInputButtonX]) {
                if (device->is_switch_joyconL || device->is_switch_joyconR) {
                    mapping = &out->b;
                } else {
                    mapping = &out->x;
                }
            } else if ([(NSString *)key isEqualToString:GCInputButtonY]) {
                mapping = &out->y;
            } else if ([(NSString *)key isEqualToString:@"Direction Pad Left"]) {
                mapping = &out->dpleft;
            } else if ([(NSString *)key isEqualToString:@"Direction Pad Right"]) {
                mapping = &out->dpright;
            } else if ([(NSString *)key isEqualToString:@"Direction Pad Up"]) {
                mapping = &out->dpup;
            } else if ([(NSString *)key isEqualToString:@"Direction Pad Down"]) {
                mapping = &out->dpdown;
            } else if ([(NSString *)key isEqualToString:@"Cardinal Direction Pad Left"]) {
                mapping = &out->dpleft;
            } else if ([(NSString *)key isEqualToString:@"Cardinal Direction Pad Right"]) {
                mapping = &out->dpright;
            } else if ([(NSString *)key isEqualToString:@"Cardinal Direction Pad Up"]) {
                mapping = &out->dpup;
            } else if ([(NSString *)key isEqualToString:@"Cardinal Direction Pad Down"]) {
                mapping = &out->dpdown;
            } else if ([(NSString *)key isEqualToString:GCInputLeftShoulder]) {
                mapping = &out->leftshoulder;
            } else if ([(NSString *)key isEqualToString:GCInputRightShoulder]) {
                mapping = &out->rightshoulder;
            } else if ([(NSString *)key isEqualToString:GCInputLeftThumbstickButton]) {
                mapping = &out->leftstick;
            } else if ([(NSString *)key isEqualToString:GCInputRightThumbstickButton]) {
                mapping = &out->rightstick;
            } else if ([(NSString *)key isEqualToString:@"Button Home"]) {
                mapping = &out->guide;
            } else if ([(NSString *)key isEqualToString:GCInputButtonMenu]) {
                if (device->is_siri_remote) {
                    mapping = &out->b;
                } else {
                    mapping = &out->start;
                }
            } else if ([(NSString *)key isEqualToString:GCInputButtonOptions]) {
                mapping = &out->back;
            } else if ([(NSString *)key isEqualToString:@"Button Share"]) {
                mapping = &out->misc1;
            } else if ([(NSString *)key isEqualToString:GCInputXboxPaddleOne]) {
                mapping = &out->paddle1;
            } else if ([(NSString *)key isEqualToString:GCInputXboxPaddleTwo]) {
                mapping = &out->paddle3;
            } else if ([(NSString *)key isEqualToString:GCInputXboxPaddleThree]) {
                mapping = &out->paddle2;
            } else if ([(NSString *)key isEqualToString:GCInputXboxPaddleFour]) {
                mapping = &out->paddle4;
            } else if ([(NSString *)key isEqualToString:GCInputLeftTrigger]) {
                mapping = &out->lefttrigger;
            } else if ([(NSString *)key isEqualToString:GCInputRightTrigger]) {
                mapping = &out->righttrigger;
            } else if ([(NSString *)key isEqualToString:GCInputDualShockTouchpadButton]) {
                mapping = &out->touchpad;
            } else if ([(NSString *)key isEqualToString:@"Button Center"]) {
                mapping = &out->a;
            }
            if (mapping && mapping->kind == EMappingKind_None) {
                mapping->kind = EMappingKind_Button;
                mapping->target = button;
            }
            ++button;
        }

        return SDL_TRUE;
    }
#endif /* ENABLE_PHYSICAL_INPUT_PROFILE */

    return SDL_FALSE;
}

#if defined(SDL_JOYSTICK_MFI) && defined(__MACOSX__)
SDL_bool IOS_SupportedHIDDevice(IOHIDDeviceRef device)
{
    if (!SDL_GetHintBoolean(SDL_HINT_JOYSTICK_MFI, SDL_TRUE)) {
        return SDL_FALSE;
    }

    if (@available(macOS 10.16, *)) {
        const int MAX_ATTEMPTS = 3;
        for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
            if ([GCController supportsHIDDevice:device]) {
                return SDL_TRUE;
            }

            /* The framework may not have seen the device yet */
            SDL_Delay(10);
        }
    }
    return SDL_FALSE;
}
#endif

#if defined(SDL_JOYSTICK_MFI) && defined(ENABLE_PHYSICAL_INPUT_PROFILE)
/* NOLINTNEXTLINE(readability-non-const-parameter): getCString takes a non-const char* */
static void GetAppleSFSymbolsNameForElement(GCControllerElement *element, char *name)
{
    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        if (element) {
            [element.sfSymbolsName getCString:name maxLength:255 encoding:NSASCIIStringEncoding];
        }
    }
}

static GCControllerDirectionPad *GetDirectionalPadForController(GCController *controller)
{
    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        return controller.physicalInputProfile.dpads[GCInputDirectionPad];
    }

    if (controller.extendedGamepad) {
        return controller.extendedGamepad.dpad;
    }

    if (controller.gamepad) {
        return controller.gamepad.dpad;
    }

    if (controller.microGamepad) {
        return controller.microGamepad.dpad;
    }

    return nil;
}
#endif /* SDL_JOYSTICK_MFI && ENABLE_PHYSICAL_INPUT_PROFILE */

static char elementName[256];

const char *IOS_GameControllerGetAppleSFSymbolsNameForButton(SDL_GameController *gamecontroller, SDL_GameControllerButton button)
{
    elementName[0] = '\0';
#if defined(SDL_JOYSTICK_MFI) && defined(ENABLE_PHYSICAL_INPUT_PROFILE)
    if (gamecontroller && SDL_GameControllerGetJoystick(gamecontroller)->driver == &SDL_IOS_JoystickDriver) {
        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            GCController *controller = SDL_GameControllerGetJoystick(gamecontroller)->hwdata->controller;
            if ([controller respondsToSelector:@selector(physicalInputProfile)]) {
                NSDictionary<NSString *, GCControllerElement *> *elements = controller.physicalInputProfile.elements;
                switch (button) {
                case SDL_CONTROLLER_BUTTON_A:
                    GetAppleSFSymbolsNameForElement(elements[GCInputButtonA], elementName);
                    break;
                case SDL_CONTROLLER_BUTTON_B:
                    GetAppleSFSymbolsNameForElement(elements[GCInputButtonB], elementName);
                    break;
                case SDL_CONTROLLER_BUTTON_X:
                    GetAppleSFSymbolsNameForElement(elements[GCInputButtonX], elementName);
                    break;
                case SDL_CONTROLLER_BUTTON_Y:
                    GetAppleSFSymbolsNameForElement(elements[GCInputButtonY], elementName);
                    break;
                case SDL_CONTROLLER_BUTTON_BACK:
                    GetAppleSFSymbolsNameForElement(elements[GCInputButtonOptions], elementName);
                    break;
                case SDL_CONTROLLER_BUTTON_GUIDE:
                    GetAppleSFSymbolsNameForElement(elements[GCInputButtonHome], elementName);
                    break;
                case SDL_CONTROLLER_BUTTON_START:
                    GetAppleSFSymbolsNameForElement(elements[GCInputButtonMenu], elementName);
                    break;
                case SDL_CONTROLLER_BUTTON_LEFTSTICK:
                    GetAppleSFSymbolsNameForElement(elements[GCInputLeftThumbstickButton], elementName);
                    break;
                case SDL_CONTROLLER_BUTTON_RIGHTSTICK:
                    GetAppleSFSymbolsNameForElement(elements[GCInputRightThumbstickButton], elementName);
                    break;
                case SDL_CONTROLLER_BUTTON_LEFTSHOULDER:
                    GetAppleSFSymbolsNameForElement(elements[GCInputLeftShoulder], elementName);
                    break;
                case SDL_CONTROLLER_BUTTON_RIGHTSHOULDER:
                    GetAppleSFSymbolsNameForElement(elements[GCInputRightShoulder], elementName);
                    break;
                case SDL_CONTROLLER_BUTTON_DPAD_UP:
                {
                    GCControllerDirectionPad *dpad = GetDirectionalPadForController(controller);
                    if (dpad) {
                        GetAppleSFSymbolsNameForElement(dpad.up, elementName);
                        if (SDL_strlen(elementName) == 0) {
                            SDL_strlcpy(elementName, "dpad.up.fill", sizeof(elementName));
                        }
                    }
                    break;
                }
                case SDL_CONTROLLER_BUTTON_DPAD_DOWN:
                {
                    GCControllerDirectionPad *dpad = GetDirectionalPadForController(controller);
                    if (dpad) {
                        GetAppleSFSymbolsNameForElement(dpad.down, elementName);
                        if (SDL_strlen(elementName) == 0) {
                            SDL_strlcpy(elementName, "dpad.down.fill", sizeof(elementName));
                        }
                    }
                    break;
                }
                case SDL_CONTROLLER_BUTTON_DPAD_LEFT:
                {
                    GCControllerDirectionPad *dpad = GetDirectionalPadForController(controller);
                    if (dpad) {
                        GetAppleSFSymbolsNameForElement(dpad.left, elementName);
                        if (SDL_strlen(elementName) == 0) {
                            SDL_strlcpy(elementName, "dpad.left.fill", sizeof(elementName));
                        }
                    }
                    break;
                }
                case SDL_CONTROLLER_BUTTON_DPAD_RIGHT:
                {
                    GCControllerDirectionPad *dpad = GetDirectionalPadForController(controller);
                    if (dpad) {
                        GetAppleSFSymbolsNameForElement(dpad.right, elementName);
                        if (SDL_strlen(elementName) == 0) {
                            SDL_strlcpy(elementName, "dpad.right.fill", sizeof(elementName));
                        }
                    }
                    break;
                }
                case SDL_CONTROLLER_BUTTON_MISC1:
                    GetAppleSFSymbolsNameForElement(elements[GCInputDualShockTouchpadButton], elementName);
                    break;
                case SDL_CONTROLLER_BUTTON_PADDLE1:
                    GetAppleSFSymbolsNameForElement(elements[GCInputXboxPaddleOne], elementName);
                    break;
                case SDL_CONTROLLER_BUTTON_PADDLE2:
                    GetAppleSFSymbolsNameForElement(elements[GCInputXboxPaddleTwo], elementName);
                    break;
                case SDL_CONTROLLER_BUTTON_PADDLE3:
                    GetAppleSFSymbolsNameForElement(elements[GCInputXboxPaddleThree], elementName);
                    break;
                case SDL_CONTROLLER_BUTTON_PADDLE4:
                    GetAppleSFSymbolsNameForElement(elements[GCInputXboxPaddleFour], elementName);
                    break;
                case SDL_CONTROLLER_BUTTON_TOUCHPAD:
                    GetAppleSFSymbolsNameForElement(elements[GCInputDualShockTouchpadButton], elementName);
                    break;
                default:
                    break;
                }
            }
        }
    }
#endif
    return elementName;
}

const char *IOS_GameControllerGetAppleSFSymbolsNameForAxis(SDL_GameController *gamecontroller, SDL_GameControllerAxis axis)
{
    elementName[0] = '\0';
#if defined(SDL_JOYSTICK_MFI) && defined(ENABLE_PHYSICAL_INPUT_PROFILE)
    if (gamecontroller && SDL_GameControllerGetJoystick(gamecontroller)->driver == &SDL_IOS_JoystickDriver) {
        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            GCController *controller = SDL_GameControllerGetJoystick(gamecontroller)->hwdata->controller;
            if ([controller respondsToSelector:@selector(physicalInputProfile)]) {
                NSDictionary<NSString *, GCControllerElement *> *elements = controller.physicalInputProfile.elements;
                switch (axis) {
                case SDL_CONTROLLER_AXIS_LEFTX:
                    GetAppleSFSymbolsNameForElement(elements[GCInputLeftThumbstick], elementName);
                    break;
                case SDL_CONTROLLER_AXIS_LEFTY:
                    GetAppleSFSymbolsNameForElement(elements[GCInputLeftThumbstick], elementName);
                    break;
                case SDL_CONTROLLER_AXIS_RIGHTX:
                    GetAppleSFSymbolsNameForElement(elements[GCInputRightThumbstick], elementName);
                    break;
                case SDL_CONTROLLER_AXIS_RIGHTY:
                    GetAppleSFSymbolsNameForElement(elements[GCInputRightThumbstick], elementName);
                    break;
                case SDL_CONTROLLER_AXIS_TRIGGERLEFT:
                    GetAppleSFSymbolsNameForElement(elements[GCInputLeftTrigger], elementName);
                    break;
                case SDL_CONTROLLER_AXIS_TRIGGERRIGHT:
                    GetAppleSFSymbolsNameForElement(elements[GCInputRightTrigger], elementName);
                    break;
                default:
                    break;
                }
            }
        }
    }
#endif
    return *elementName ? elementName : NULL;
}

SDL_JoystickDriver SDL_IOS_JoystickDriver = {
    IOS_JoystickInit,
    IOS_JoystickGetCount,
    IOS_JoystickDetect,
    IOS_JoystickGetDeviceName,
    IOS_JoystickGetDevicePath,
    IOS_JoystickGetDeviceSteamVirtualGamepadSlot,
    IOS_JoystickGetDevicePlayerIndex,
    IOS_JoystickSetDevicePlayerIndex,
    IOS_JoystickGetDeviceGUID,
    IOS_JoystickGetDeviceInstanceID,
    IOS_JoystickOpen,
    IOS_JoystickRumble,
    IOS_JoystickRumbleTriggers,
    IOS_JoystickGetCapabilities,
    IOS_JoystickSetLED,
    IOS_JoystickSendEffect,
    IOS_JoystickSetSensorsEnabled,
    IOS_JoystickUpdate,
    IOS_JoystickClose,
    IOS_JoystickQuit,
    IOS_JoystickGetGamepadMapping
};

/* vi: set ts=4 sw=4 expandtab: */
