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

#ifdef SDL_VIDEO_DRIVER_X11

#include "SDL_hints.h"
#include "SDL_misc.h"
#include "SDL_x11video.h"

#include "../../events/SDL_keyboard_c.h"
#include "../../events/SDL_scancode_tables_c.h"

#include <X11/keysym.h>
#include <X11/XKBlib.h>

#include "../../events/imKStoUCS.h"
#include "../../events/SDL_keysym_to_scancode_c.h"

#ifdef X_HAVE_UTF8_STRING
#include <locale.h>
#endif

static SDL_ScancodeTable scancode_set[] = {
    SDL_SCANCODE_TABLE_DARWIN,
    SDL_SCANCODE_TABLE_XFREE86_1,
    SDL_SCANCODE_TABLE_XFREE86_2,
    SDL_SCANCODE_TABLE_XVNC,
};

static SDL_bool X11_ScancodeIsRemappable(SDL_Scancode scancode)
{
    /*
     * XKB remappings can assign different keysyms for these scancodes, but
     * as these keys are in fixed positions, the scancodes themselves shouldn't
     * be switched. Mark them as not being remappable.
     */
    switch (scancode) {
    case SDL_SCANCODE_ESCAPE:
    case SDL_SCANCODE_CAPSLOCK:
    case SDL_SCANCODE_NUMLOCKCLEAR:
    case SDL_SCANCODE_LSHIFT:
    case SDL_SCANCODE_RSHIFT:
    case SDL_SCANCODE_LCTRL:
    case SDL_SCANCODE_RCTRL:
    case SDL_SCANCODE_LALT:
    case SDL_SCANCODE_RALT:
    case SDL_SCANCODE_LGUI:
    case SDL_SCANCODE_RGUI:
        return SDL_FALSE;
    default:
        return SDL_TRUE;
    }
}

/* This function only correctly maps letters and numbers for keyboards in US QWERTY layout */
static SDL_Scancode X11_KeyCodeToSDLScancode(_THIS, KeyCode keycode)
{
    const KeySym keysym = X11_KeyCodeToSym(_this, keycode, 0);

    if (keysym == NoSymbol) {
        return SDL_SCANCODE_UNKNOWN;
    }

    return SDL_GetScancodeFromKeySym(keysym, keycode);
}

static Uint32 X11_KeyCodeToUcs4(_THIS, KeyCode keycode, unsigned char group)
{
    KeySym keysym = X11_KeyCodeToSym(_this, keycode, group);

    if (keysym == NoSymbol) {
        return 0;
    }

    return SDL_KeySymToUcs4(keysym);
}

KeySym
X11_KeyCodeToSym(_THIS, KeyCode keycode, unsigned char group)
{
    SDL_VideoData *data = (SDL_VideoData *)_this->driverdata;
    KeySym keysym;

#ifdef SDL_VIDEO_DRIVER_X11_HAS_XKBKEYCODETOKEYSYM
    if (data->xkb) {
        int num_groups = XkbKeyNumGroups(data->xkb, keycode);
        unsigned char info = XkbKeyGroupInfo(data->xkb, keycode);

        if (num_groups && group >= num_groups) {

            int action = XkbOutOfRangeGroupAction(info);

            if (action == XkbRedirectIntoRange) {
                group = XkbOutOfRangeGroupNumber(info);
                if (group >= num_groups) {
                    group = 0;
                }
            } else if (action == XkbClampIntoRange) {
                group = num_groups - 1;
            } else {
                group %= num_groups;
            }
        }
        keysym = X11_XkbKeycodeToKeysym(data->display, keycode, group, 0);
    } else {
        keysym = X11_XKeycodeToKeysym(data->display, keycode, 0);
    }
#else
    keysym = X11_XKeycodeToKeysym(data->display, keycode, 0);
#endif

    return keysym;
}

int X11_InitKeyboard(_THIS)
{
    SDL_VideoData *data = (SDL_VideoData *)_this->driverdata;
    int i = 0;
    int j = 0;
    int min_keycode, max_keycode;
    struct
    {
        SDL_Scancode scancode;
        KeySym keysym;
        int value;
    } fingerprint[] = {
        { SDL_SCANCODE_HOME, XK_Home, 0 },
        { SDL_SCANCODE_PAGEUP, XK_Prior, 0 },
        { SDL_SCANCODE_UP, XK_Up, 0 },
        { SDL_SCANCODE_LEFT, XK_Left, 0 },
        { SDL_SCANCODE_DELETE, XK_Delete, 0 },
        { SDL_SCANCODE_KP_ENTER, XK_KP_Enter, 0 },
    };
    int best_distance;
    int best_index;
    int distance;
    Bool xkb_repeat = 0;

#ifdef SDL_VIDEO_DRIVER_X11_HAS_XKBKEYCODETOKEYSYM
    {
        int xkb_major = XkbMajorVersion;
        int xkb_minor = XkbMinorVersion;

        if (X11_XkbQueryExtension(data->display, NULL, &data->xkb_event, NULL, &xkb_major, &xkb_minor)) {
            data->xkb = X11_XkbGetMap(data->display, XkbAllClientInfoMask, XkbUseCoreKbd);
        }

        /* This will remove KeyRelease events for held keys */
        X11_XkbSetDetectableAutoRepeat(data->display, True, &xkb_repeat);
    }
#endif

    /* Open a connection to the X input manager */
#ifdef X_HAVE_UTF8_STRING
    if (SDL_X11_HAVE_UTF8) {
        /* Set the locale, and call XSetLocaleModifiers before XOpenIM so that
           Compose keys will work correctly. */
        char *prev_locale = setlocale(LC_ALL, NULL);
        char *prev_xmods = X11_XSetLocaleModifiers(NULL);
        const char *new_xmods = "";
        const char *env_xmods = SDL_getenv("XMODIFIERS");
        SDL_bool has_dbus_ime_support = SDL_FALSE;

        if (prev_locale) {
            prev_locale = SDL_strdup(prev_locale);
        }

        if (prev_xmods) {
            prev_xmods = SDL_strdup(prev_xmods);
        }

        /* IBus resends some key events that were filtered by XFilterEvents
           when it is used via XIM which causes issues. Prevent this by forcing
           @im=none if XMODIFIERS contains @im=ibus. IBus can still be used via
           the DBus implementation, which also has support for pre-editing. */
        if (env_xmods && SDL_strstr(env_xmods, "@im=ibus") != NULL) {
            has_dbus_ime_support = SDL_TRUE;
        }
        if (env_xmods && SDL_strstr(env_xmods, "@im=fcitx") != NULL) {
            has_dbus_ime_support = SDL_TRUE;
        }
        if (has_dbus_ime_support || !xkb_repeat) {
            new_xmods = "@im=none";
        }

        (void)setlocale(LC_ALL, "");
        X11_XSetLocaleModifiers(new_xmods);

        data->im = X11_XOpenIM(data->display, NULL, data->classname, data->classname);

        /* Reset the locale + X locale modifiers back to how they were,
           locale first because the X locale modifiers depend on it. */
        (void)setlocale(LC_ALL, prev_locale);
        X11_XSetLocaleModifiers(prev_xmods);

        if (prev_locale) {
            SDL_free(prev_locale);
        }

        if (prev_xmods) {
            SDL_free(prev_xmods);
        }
    }
#endif
    /* Try to determine which scancodes are being used based on fingerprint */
    best_distance = SDL_arraysize(fingerprint) + 1;
    best_index = -1;
    X11_XDisplayKeycodes(data->display, &min_keycode, &max_keycode);
    for (i = 0; i < SDL_arraysize(fingerprint); ++i) {
        fingerprint[i].value = X11_XKeysymToKeycode(data->display, fingerprint[i].keysym) - min_keycode;
    }
    for (i = 0; i < SDL_arraysize(scancode_set); ++i) {
        int table_size;
        const SDL_Scancode *table = SDL_GetScancodeTable(scancode_set[i], &table_size);

        distance = 0;
        for (j = 0; j < SDL_arraysize(fingerprint); ++j) {
            if (fingerprint[j].value < 0 || fingerprint[j].value >= table_size) {
                distance += 1;
            } else if (table[fingerprint[j].value] != fingerprint[j].scancode) {
                distance += 1;
            }
        }
        if (distance < best_distance) {
            best_distance = distance;
            best_index = i;
        }
    }
    if (best_index < 0 || best_distance > 2) {
        /* This is likely to be SDL_SCANCODE_TABLE_XFREE86_2 with remapped keys, double check a rarely remapped value */
        int fingerprint_value = X11_XKeysymToKeycode(data->display, 0x1008FF5B /* XF86Documents */) - min_keycode;
        if (fingerprint_value == 235) {
            for (i = 0; i < SDL_arraysize(scancode_set); ++i) {
                if (scancode_set[i] == SDL_SCANCODE_TABLE_XFREE86_2) {
                    best_index = i;
                    best_distance = 0;
                    break;
                }
            }
        }
    }
    if (best_index >= 0 && best_distance <= 2) {
        SDL_Keycode default_keymap[SDL_NUM_SCANCODES];
        int table_size;
        const SDL_Scancode *table = SDL_GetScancodeTable(scancode_set[best_index], &table_size);

#ifdef DEBUG_KEYBOARD
        SDL_Log("Using scancode set %d, min_keycode = %d, max_keycode = %d, table_size = %d\n", best_index, min_keycode, max_keycode, table_size);
#endif
        /* This should never happen, but just in case... */
        if (table_size > (SDL_arraysize(data->key_layout) - min_keycode)) {
            table_size = (SDL_arraysize(data->key_layout) - min_keycode);
        }
        SDL_memcpy(&data->key_layout[min_keycode], table, sizeof(SDL_Scancode) * table_size);

        /* Scancodes represent physical locations on the keyboard, unaffected by keyboard mapping.
           However, there are a number of extended scancodes that have no standard location, so use
           the X11 mapping for all non-character keys.
         */
        SDL_GetDefaultKeymap(default_keymap);

        for (i = min_keycode; i <= max_keycode; ++i) {
            SDL_Scancode scancode = X11_KeyCodeToSDLScancode(_this, i);
#ifdef DEBUG_KEYBOARD
            {
                KeySym sym;
                sym = X11_KeyCodeToSym(_this, (KeyCode)i, 0);
                SDL_Log("code = %d, sym = 0x%X (%s) ", i - min_keycode,
                        (unsigned int)sym, sym == NoSymbol ? "NoSymbol" : X11_XKeysymToString(sym));
            }
#endif
            if (scancode == data->key_layout[i]) {
                continue;
            }
            if (default_keymap[scancode] >= SDLK_SCANCODE_MASK && X11_ScancodeIsRemappable(scancode)) {
                /* Not a character key and the scancode is safe to remap */
#ifdef DEBUG_KEYBOARD
                SDL_Log("Changing scancode, was %d (%s), now %d (%s)\n", data->key_layout[i], SDL_GetScancodeName(data->key_layout[i]), scancode, SDL_GetScancodeName(scancode));
#endif
                data->key_layout[i] = scancode;
            }
        }
    } else {
#ifdef DEBUG_SCANCODES
        SDL_Log("Keyboard layout unknown, please report the following to the SDL forums/mailing list (https://discourse.libsdl.org/):\n");
#endif

        /* Determine key_layout - only works on US QWERTY layout */
        for (i = min_keycode; i <= max_keycode; ++i) {
            SDL_Scancode scancode = X11_KeyCodeToSDLScancode(_this, i);
#ifdef DEBUG_SCANCODES
            {
                KeySym sym;
                sym = X11_KeyCodeToSym(_this, (KeyCode)i, 0);
                SDL_Log("code = %d, sym = 0x%X (%s) ", i - min_keycode,
                        (unsigned int)sym, sym == NoSymbol ? "NoSymbol" : X11_XKeysymToString(sym));
            }
            if (scancode == SDL_SCANCODE_UNKNOWN) {
                SDL_Log("scancode not found\n");
            } else {
                SDL_Log("scancode = %d (%s)\n", scancode, SDL_GetScancodeName(scancode));
            }
#endif
            data->key_layout[i] = scancode;
        }
    }

    X11_UpdateKeymap(_this, SDL_FALSE);

    SDL_SetScancodeName(SDL_SCANCODE_APPLICATION, "Menu");

#ifdef SDL_USE_IME
    SDL_IME_Init();
#endif

    X11_ReconcileKeyboardState(_this);

    return 0;
}

void X11_UpdateKeymap(_THIS, SDL_bool send_event)
{
    SDL_VideoData *data = (SDL_VideoData *)_this->driverdata;
    int i;
    SDL_Scancode scancode;
    SDL_Keycode keymap[SDL_NUM_SCANCODES];
    unsigned char group = 0;

    SDL_GetDefaultKeymap(keymap);

#ifdef SDL_VIDEO_DRIVER_X11_HAS_XKBKEYCODETOKEYSYM
    if (data->xkb) {
        XkbStateRec state;
        X11_XkbGetUpdatedMap(data->display, XkbAllClientInfoMask, data->xkb);

        if (X11_XkbGetState(data->display, XkbUseCoreKbd, &state) == Success) {
            group = state.group;
        }
    }
#endif

    for (i = 0; i < SDL_arraysize(data->key_layout); i++) {
        Uint32 key;

        /* Make sure this is a valid scancode */
        scancode = data->key_layout[i];
        if (scancode == SDL_SCANCODE_UNKNOWN) {
            continue;
        }

        /* See if there is a UCS keycode for this scancode */
        key = X11_KeyCodeToUcs4(_this, (KeyCode)i, group);
        if (key) {
            keymap[scancode] = key;
        } else {
            SDL_Scancode keyScancode = X11_KeyCodeToSDLScancode(_this, (KeyCode)i);

            switch (keyScancode) {
            case SDL_SCANCODE_RETURN:
                keymap[scancode] = SDLK_RETURN;
                break;
            case SDL_SCANCODE_ESCAPE:
                keymap[scancode] = SDLK_ESCAPE;
                break;
            case SDL_SCANCODE_BACKSPACE:
                keymap[scancode] = SDLK_BACKSPACE;
                break;
            case SDL_SCANCODE_TAB:
                keymap[scancode] = SDLK_TAB;
                break;
            case SDL_SCANCODE_DELETE:
                keymap[scancode] = SDLK_DELETE;
                break;
            default:
                keymap[scancode] = SDL_SCANCODE_TO_KEYCODE(keyScancode);
                break;
            }
        }
    }
    SDL_SetKeymap(0, keymap, SDL_NUM_SCANCODES, send_event);
}

void X11_QuitKeyboard(_THIS)
{
    SDL_VideoData *data = (SDL_VideoData *)_this->driverdata;

#ifdef SDL_VIDEO_DRIVER_X11_HAS_XKBKEYCODETOKEYSYM
    if (data->xkb) {
        X11_XkbFreeKeyboard(data->xkb, 0, True);
        data->xkb = NULL;
    }
#endif

#ifdef SDL_USE_IME
    SDL_IME_Quit();
#endif
}

static void X11_ResetXIM(_THIS)
{
#ifdef X_HAVE_UTF8_STRING
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
    int i;

    if (videodata && videodata->windowlist) {
        for (i = 0; i < videodata->numwindows; ++i) {
            SDL_WindowData *data = videodata->windowlist[i];
            if (data && data->ic) {
                /* Clear any partially entered dead keys */
                char *contents = X11_Xutf8ResetIC(data->ic);
                if (contents) {
                    X11_XFree(contents);
                }
            }
        }
    }
#endif
}

void X11_StartTextInput(_THIS)
{
    X11_ResetXIM(_this);
}

void X11_StopTextInput(_THIS)
{
    X11_ResetXIM(_this);
#ifdef SDL_USE_IME
    SDL_IME_Reset();
#endif
}

void X11_SetTextInputRect(_THIS, const SDL_Rect *rect)
{
    if (!rect) {
        SDL_InvalidParamError("rect");
        return;
    }

#ifdef SDL_USE_IME
    SDL_IME_UpdateTextRect(rect);
#endif
}

SDL_bool X11_HasScreenKeyboardSupport(_THIS)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
    return videodata->is_steam_deck;
}

void X11_ShowScreenKeyboard(_THIS, SDL_Window *window)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;

    if (videodata->is_steam_deck) {
        /* For more documentation of the URL parameters, see:
         * https://partner.steamgames.com/doc/api/ISteamUtils#ShowFloatingGamepadTextInput
         */
        char deeplink[128];
        (void)SDL_snprintf(deeplink, sizeof(deeplink),
                           "steam://open/keyboard?XPosition=0&YPosition=0&Width=0&Height=0&Mode=%d",
                           SDL_GetHintBoolean(SDL_HINT_RETURN_KEY_HIDES_IME, SDL_FALSE) ? 0 : 1);
        SDL_OpenURL(deeplink);
        videodata->steam_keyboard_open = SDL_TRUE;
    }
}

void X11_HideScreenKeyboard(_THIS, SDL_Window *window)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;

    if (videodata->is_steam_deck) {
        SDL_OpenURL("steam://close/keyboard");
        videodata->steam_keyboard_open = SDL_FALSE;
    }
}

SDL_bool X11_IsScreenKeyboardShown(_THIS, SDL_Window *window)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;

    return videodata->steam_keyboard_open;
}

#endif /* SDL_VIDEO_DRIVER_X11 */

/* vi: set ts=4 sw=4 expandtab: */
