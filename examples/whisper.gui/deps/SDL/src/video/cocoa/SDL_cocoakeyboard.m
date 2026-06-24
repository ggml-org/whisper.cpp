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

#include "../../events/SDL_events_c.h"
#include "../../events/SDL_keyboard_c.h"
#include "../../events/scancodes_darwin.h"

#include <Carbon/Carbon.h>

/*#define DEBUG_IME NSLog */
#define DEBUG_IME(...)

@interface SDLTranslatorResponder : NSView <NSTextInputClient> {
    NSString *_markedText;
    NSRange   _markedRange;
    NSRange   _selectedRange;
    SDL_Rect  _inputRect;
}
- (void)doCommandBySelector:(SEL)myselector;
- (void)setInputRect:(const SDL_Rect *)rect;
@end

@implementation SDLTranslatorResponder

- (void)setInputRect:(const SDL_Rect *)rect
{
    _inputRect = *rect;
}

- (void)insertText:(id)aString replacementRange:(NSRange)replacementRange
{
    /* TODO: Make use of replacementRange? */

    const char *str;

    DEBUG_IME(@"insertText: %@", aString);

    /* Could be NSString or NSAttributedString, so we have
     * to test and convert it before return as SDL event */
    if ([aString isKindOfClass: [NSAttributedString class]]) {
        str = [[aString string] UTF8String];
    } else {
        str = [aString UTF8String];
    }

    /* We're likely sending the composed text, so we reset the IME status. */
    if ([self hasMarkedText]) {
        [self unmarkText];
    }

    SDL_SendKeyboardText(str);
}

- (void)doCommandBySelector:(SEL)myselector
{
    /* No need to do anything since we are not using Cocoa
       selectors to handle special keys, instead we use SDL
       key events to do the same job.
    */
}

- (BOOL)hasMarkedText
{
    return _markedText != nil;
}

- (NSRange)markedRange
{
    return _markedRange;
}

- (NSRange)selectedRange
{
    return _selectedRange;
}

- (void)setMarkedText:(id)aString selectedRange:(NSRange)selectedRange replacementRange:(NSRange)replacementRange
{
    if ([aString isKindOfClass:[NSAttributedString class]]) {
        aString = [aString string];
    }

    if ([aString length] == 0) {
        [self unmarkText];
        return;
    }

    if (_markedText != aString) {
        _markedText = aString;
    }

    _selectedRange = selectedRange;
    _markedRange = NSMakeRange(0, [aString length]);

    SDL_SendEditingText([aString UTF8String],
                        (int) selectedRange.location, (int) selectedRange.length);

    DEBUG_IME(@"setMarkedText: %@, (%d, %d)", _markedText,
          selectedRange.location, selectedRange.length);
}

- (void)unmarkText
{
    _markedText = nil;

    SDL_SendEditingText("", 0, 0);
}

- (NSRect)firstRectForCharacterRange:(NSRange)aRange actualRange:(NSRangePointer)actualRange
{
    NSWindow *window = [self window];
    NSRect contentRect = [window contentRectForFrameRect:[window frame]];
    float windowHeight = contentRect.size.height;
    NSRect rect = NSMakeRect(_inputRect.x, windowHeight - _inputRect.y - _inputRect.h,
                             _inputRect.w, _inputRect.h);

    if (actualRange) {
        *actualRange = aRange;
    }

    DEBUG_IME(@"firstRectForCharacterRange: (%d, %d): windowHeight = %g, rect = %@",
            aRange.location, aRange.length, windowHeight,
            NSStringFromRect(rect));

    rect = [window convertRectToScreen:rect];

    return rect;
}

- (NSAttributedString *)attributedSubstringForProposedRange:(NSRange)aRange actualRange:(NSRangePointer)actualRange
{
    DEBUG_IME(@"attributedSubstringFromRange: (%d, %d)", aRange.location, aRange.length);
    return nil;
}

- (NSInteger)conversationIdentifier
{
    return (NSInteger) self;
}

/* This method returns the index for character that is
 * nearest to thePoint.  thPoint is in screen coordinate system.
 */
- (NSUInteger)characterIndexForPoint:(NSPoint)thePoint
{
    DEBUG_IME(@"characterIndexForPoint: (%g, %g)", thePoint.x, thePoint.y);
    return 0;
}

/* This method is the key to attribute extension.
 * We could add new attributes through this method.
 * NSInputServer examines the return value of this
 * method & constructs appropriate attributed string.
 */
- (NSArray *)validAttributesForMarkedText
{
    return [NSArray array];
}

@end

static bool IsModifierKeyPressed(unsigned int flags,
                                 unsigned int target_mask,
                                 unsigned int other_mask,
                                 unsigned int either_mask)
{
    bool target_pressed = (flags & target_mask) != 0;
    bool other_pressed = (flags & other_mask) != 0;
    bool either_pressed = (flags & either_mask) != 0;

    if (either_pressed != (target_pressed || other_pressed))
        return either_pressed;

    return target_pressed;
}

static void HandleModifiers(_THIS, SDL_Scancode code, unsigned int modifierFlags)
{
    bool pressed = false;

    if (code == SDL_SCANCODE_LSHIFT) {
        pressed = IsModifierKeyPressed(modifierFlags, NX_DEVICELSHIFTKEYMASK,
                                       NX_DEVICERSHIFTKEYMASK, NX_SHIFTMASK);
    } else if (code == SDL_SCANCODE_LCTRL) {
        pressed = IsModifierKeyPressed(modifierFlags, NX_DEVICELCTLKEYMASK,
                                       NX_DEVICERCTLKEYMASK, NX_CONTROLMASK);
    } else if (code == SDL_SCANCODE_LALT) {
        pressed = IsModifierKeyPressed(modifierFlags, NX_DEVICELALTKEYMASK,
                                       NX_DEVICERALTKEYMASK, NX_ALTERNATEMASK);
    } else if (code == SDL_SCANCODE_LGUI) {
        pressed = IsModifierKeyPressed(modifierFlags, NX_DEVICELCMDKEYMASK,
                                       NX_DEVICERCMDKEYMASK, NX_COMMANDMASK);
    } else if (code == SDL_SCANCODE_RSHIFT) {
        pressed = IsModifierKeyPressed(modifierFlags, NX_DEVICERSHIFTKEYMASK,
                                       NX_DEVICELSHIFTKEYMASK, NX_SHIFTMASK);
    } else if (code == SDL_SCANCODE_RCTRL) {
        pressed = IsModifierKeyPressed(modifierFlags, NX_DEVICERCTLKEYMASK,
                                       NX_DEVICELCTLKEYMASK, NX_CONTROLMASK);
    } else if (code == SDL_SCANCODE_RALT) {
        pressed = IsModifierKeyPressed(modifierFlags, NX_DEVICERALTKEYMASK,
                                       NX_DEVICELALTKEYMASK, NX_ALTERNATEMASK);
    } else if (code == SDL_SCANCODE_RGUI) {
        pressed = IsModifierKeyPressed(modifierFlags, NX_DEVICERCMDKEYMASK,
                                       NX_DEVICELCMDKEYMASK, NX_COMMANDMASK);
    } else {
        return;
    }

    if (pressed) {
        SDL_SendKeyboardKey(SDL_PRESSED, code);
    } else {
        SDL_SendKeyboardKey(SDL_RELEASED, code);
    }
}

static void UpdateKeymap(SDL_VideoData *data, SDL_bool send_event)
{
    TISInputSourceRef key_layout;
    const void *chr_data;
    int i;
    SDL_Scancode scancode;
    SDL_Keycode keymap[SDL_NUM_SCANCODES];
    CFDataRef uchrDataRef;

    /* See if the keymap needs to be updated */
    key_layout = TISCopyCurrentKeyboardLayoutInputSource();
    if (key_layout == data.key_layout) {
        return;
    }
    data.key_layout = key_layout;

    SDL_GetDefaultKeymap(keymap);

    /* Try Unicode data first */
    uchrDataRef = TISGetInputSourceProperty(key_layout, kTISPropertyUnicodeKeyLayoutData);
    if (uchrDataRef) {
        chr_data = CFDataGetBytePtr(uchrDataRef);
    } else {
        goto cleanup;
    }

    if (chr_data) {
        UInt32 keyboard_type = LMGetKbdType();
        OSStatus err;

        for (i = 0; i < SDL_arraysize(darwin_scancode_table); i++) {
            UniChar s[8];
            UniCharCount len;
            UInt32 dead_key_state;

            /* Make sure this scancode is a valid character scancode */
            scancode = darwin_scancode_table[i];
            if (scancode == SDL_SCANCODE_UNKNOWN ||
                (keymap[scancode] & SDLK_SCANCODE_MASK)) {
                continue;
            }

            /*
             * Swap the scancode for these two wrongly translated keys
             * UCKeyTranslate() function does not do its job properly for ISO layout keyboards, where the key '@',
             * which is located in the top left corner of the keyboard right under the Escape key, and the additional
             * key '<', which is on the right of the Shift key, are inverted
            */
            if ((scancode == SDL_SCANCODE_NONUSBACKSLASH || scancode == SDL_SCANCODE_GRAVE) && KBGetLayoutType(LMGetKbdType()) == kKeyboardISO) {
                /* see comments in scancodes_darwin.h */
                scancode = (SDL_Scancode)((SDL_SCANCODE_NONUSBACKSLASH + SDL_SCANCODE_GRAVE) - scancode);
            }

            dead_key_state = 0;
            err = UCKeyTranslate ((UCKeyboardLayout *) chr_data,
                                  i, kUCKeyActionDown,
                                  0, keyboard_type,
                                  kUCKeyTranslateNoDeadKeysMask,
                                  &dead_key_state, 8, &len, s);
            if (err != noErr) {
                continue;
            }

            if (len > 0 && s[0] != 0x10) {
                keymap[scancode] = s[0];
            }
        }
        SDL_SetKeymap(0, keymap, SDL_NUM_SCANCODES, send_event);
        return;
    }

cleanup:
    CFRelease(key_layout);
}

void Cocoa_InitKeyboard(_THIS)
{
    SDL_VideoData *data = (__bridge SDL_VideoData *) _this->driverdata;

    UpdateKeymap(data, SDL_FALSE);

    /* Set our own names for the platform-dependent but layout-independent keys */
    /* This key is NumLock on the MacBook keyboard. :) */
    /*SDL_SetScancodeName(SDL_SCANCODE_NUMLOCKCLEAR, "Clear");*/
    SDL_SetScancodeName(SDL_SCANCODE_LALT, "Left Option");
    SDL_SetScancodeName(SDL_SCANCODE_LGUI, "Left Command");
    SDL_SetScancodeName(SDL_SCANCODE_RALT, "Right Option");
    SDL_SetScancodeName(SDL_SCANCODE_RGUI, "Right Command");

    data.modifierFlags = (unsigned int)[NSEvent modifierFlags];
    SDL_ToggleModState(KMOD_CAPS, (data.modifierFlags & NSEventModifierFlagCapsLock) ? SDL_TRUE : SDL_FALSE);
}

void Cocoa_StartTextInput(_THIS)
{ @autoreleasepool
{
    NSView *parentView;
    SDL_VideoData *data = (__bridge SDL_VideoData *) _this->driverdata;
    SDL_Window *window = SDL_GetKeyboardFocus();
    NSWindow *nswindow = nil;
    if (window) {
        nswindow = ((__bridge SDL_WindowData*)window->driverdata).nswindow;
    }

    parentView = [nswindow contentView];

    /* We only keep one field editor per process, since only the front most
     * window can receive text input events, so it make no sense to keep more
     * than one copy. When we switched to another window and requesting for
     * text input, simply remove the field editor from its superview then add
     * it to the front most window's content view */
    if (!data.fieldEdit) {
        data.fieldEdit =
            [[SDLTranslatorResponder alloc] initWithFrame: NSMakeRect(0.0, 0.0, 0.0, 0.0)];
    }

    if (![[data.fieldEdit superview] isEqual:parentView]) {
        /* DEBUG_IME(@"add fieldEdit to window contentView"); */
        [data.fieldEdit removeFromSuperview];
        [parentView addSubview: data.fieldEdit];
        [nswindow makeFirstResponder: data.fieldEdit];
    }
}}

void Cocoa_StopTextInput(_THIS)
{ @autoreleasepool
{
    SDL_VideoData *data = (__bridge SDL_VideoData *) _this->driverdata;

    if (data && data.fieldEdit) {
        [data.fieldEdit removeFromSuperview];
        data.fieldEdit = nil;
    }
}}

void Cocoa_SetTextInputRect(_THIS, const SDL_Rect *rect)
{
    SDL_VideoData *data = (__bridge SDL_VideoData *) _this->driverdata;

    if (!rect) {
        SDL_InvalidParamError("rect");
        return;
    }

    [data.fieldEdit setInputRect:rect];
}

void Cocoa_HandleKeyEvent(_THIS, NSEvent *event)
{
    unsigned short scancode;
    SDL_Scancode code;
    SDL_VideoData *data = _this ? ((__bridge SDL_VideoData *) _this->driverdata) : nil;
    if (!data) {
        return;  /* can happen when returning from fullscreen Space on shutdown */
    }

    scancode = [event keyCode];
#if 0
    const char *text;
#endif

    if ((scancode == 10 || scancode == 50) && KBGetLayoutType(LMGetKbdType()) == kKeyboardISO) {
        /* see comments in scancodes_darwin.h */
        scancode = 60 - scancode;
    }

    if (scancode < SDL_arraysize(darwin_scancode_table)) {
        code = darwin_scancode_table[scancode];
    } else {
        /* Hmm, does this ever happen?  If so, need to extend the keymap... */
        code = SDL_SCANCODE_UNKNOWN;
    }

    switch ([event type]) {
    case NSEventTypeKeyDown:
        if (![event isARepeat]) {
            /* See if we need to rebuild the keyboard layout */
            UpdateKeymap(data, SDL_TRUE);
        }

        SDL_SendKeyboardKey(SDL_PRESSED, code);
#ifdef DEBUG_SCANCODES
        if (code == SDL_SCANCODE_UNKNOWN) {
            SDL_Log("The key you just pressed is not recognized by SDL. To help get this fixed, report this to the SDL forums/mailing list <https://discourse.libsdl.org/> or to Christian Walther <cwalther@gmx.ch>. Mac virtual key code is %d.\n", scancode);
        }
#endif
        if (SDL_EventState(SDL_TEXTINPUT, SDL_QUERY)) {
            /* FIXME CW 2007-08-16: only send those events to the field editor for which we actually want text events, not e.g. esc or function keys. Arrow keys in particular seem to produce crashes sometimes. */
            [data.fieldEdit interpretKeyEvents:[NSArray arrayWithObject:event]];
#if 0
            text = [[event characters] UTF8String];
            if(text && *text) {
                SDL_SendKeyboardText(text);
                [data->fieldEdit setString:@""];
            }
#endif
        }
        break;
    case NSEventTypeKeyUp:
        SDL_SendKeyboardKey(SDL_RELEASED, code);
        break;
    case NSEventTypeFlagsChanged: {
        // see if the new modifierFlags mean any existing keys should be pressed/released...
        const unsigned int modflags = (unsigned int)[event modifierFlags];
        HandleModifiers(_this, SDL_SCANCODE_LSHIFT, modflags);
        HandleModifiers(_this, SDL_SCANCODE_LCTRL, modflags);
        HandleModifiers(_this, SDL_SCANCODE_LALT, modflags);
        HandleModifiers(_this, SDL_SCANCODE_LGUI, modflags);
        HandleModifiers(_this, SDL_SCANCODE_RSHIFT, modflags);
        HandleModifiers(_this, SDL_SCANCODE_RCTRL, modflags);
        HandleModifiers(_this, SDL_SCANCODE_RALT, modflags);
        HandleModifiers(_this, SDL_SCANCODE_RGUI, modflags);
        break;
    }
    default: /* just to avoid compiler warnings */
        break;
    }
}

void Cocoa_QuitKeyboard(_THIS)
{
}

typedef int CGSConnection;
typedef enum {
    CGSGlobalHotKeyEnable = 0,
    CGSGlobalHotKeyDisable = 1,
} CGSGlobalHotKeyOperatingMode;

extern CGSConnection _CGSDefaultConnection(void);
extern CGError CGSSetGlobalHotKeyOperatingMode(CGSConnection connection, CGSGlobalHotKeyOperatingMode mode);

void Cocoa_SetWindowKeyboardGrab(_THIS, SDL_Window * window, SDL_bool grabbed)
{
#ifdef SDL_MAC_NO_SANDBOX
    CGSSetGlobalHotKeyOperatingMode(_CGSDefaultConnection(), grabbed ? CGSGlobalHotKeyDisable : CGSGlobalHotKeyEnable);
#endif
}

#endif /* SDL_VIDEO_DRIVER_COCOA */

/* vi: set ts=4 sw=4 expandtab: */
