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

#ifdef SDL_VIDEO_DRIVER_NACL

#include "SDL.h"
#include "../../events/SDL_events_c.h"
#include "../SDL_sysvideo.h"
#include "SDL_naclevents_c.h"
#include "SDL_naclvideo.h"
#include "ppapi_simple/ps_event.h"

/* Ref: https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent */

static SDL_Scancode NACL_Keycodes[] = {
    SDL_SCANCODE_UNKNOWN,               /* 0 */
    SDL_SCANCODE_UNKNOWN,               /* 1 */
    SDL_SCANCODE_UNKNOWN,               /* 2 */
    SDL_SCANCODE_CANCEL,                /* DOM_VK_CANCEL 3 */
    SDL_SCANCODE_UNKNOWN,               /* 4 */
    SDL_SCANCODE_UNKNOWN,               /* 5 */
    SDL_SCANCODE_HELP,                  /* DOM_VK_HELP 6 */
    SDL_SCANCODE_UNKNOWN,               /* 7 */
    SDL_SCANCODE_BACKSPACE,             /* DOM_VK_BACK_SPACE 8 */
    SDL_SCANCODE_TAB,                   /* DOM_VK_TAB 9 */
    SDL_SCANCODE_UNKNOWN,               /* 10 */
    SDL_SCANCODE_UNKNOWN,               /* 11 */
    SDL_SCANCODE_CLEAR,                 /* DOM_VK_CLEAR 12 */
    SDL_SCANCODE_RETURN,                /* DOM_VK_RETURN 13 */
    SDL_SCANCODE_RETURN,                /* DOM_VK_ENTER 14 */
    SDL_SCANCODE_UNKNOWN,               /* 15 */
    SDL_SCANCODE_LSHIFT,                /* DOM_VK_SHIFT 16 */
    SDL_SCANCODE_LCTRL,                 /* DOM_VK_CONTROL 17 */
    SDL_SCANCODE_LALT,                  /* DOM_VK_ALT 18 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_PAUSE 19 */
    SDL_SCANCODE_CAPSLOCK,              /* DOM_VK_CAPS_LOCK 20 */
    SDL_SCANCODE_LANG1,                 /* DOM_VK_KANA  DOM_VK_HANGUL 21 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_EISU 22 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_JUNJA 23 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_FINAL 24 */
    SDL_SCANCODE_LANG2,                 /* DOM_VK_HANJA  DOM_VK_KANJI 25 */
    SDL_SCANCODE_UNKNOWN,               /* 26 */
    SDL_SCANCODE_ESCAPE,                /* DOM_VK_ESCAPE 27 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_CONVERT 28 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_NONCONVERT 29 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_ACCEPT 30 */
    SDL_SCANCODE_MODE,                  /* DOM_VK_MODECHANGE 31 */
    SDL_SCANCODE_SPACE,                 /* DOM_VK_SPACE 32 */
    SDL_SCANCODE_PAGEUP,                /* DOM_VK_PAGE_UP 33 */
    SDL_SCANCODE_PAGEDOWN,              /* DOM_VK_PAGE_DOWN 34 */
    SDL_SCANCODE_END,                   /* DOM_VK_END 35 */
    SDL_SCANCODE_HOME,                  /* DOM_VK_HOME 36 */
    SDL_SCANCODE_LEFT,                  /* DOM_VK_LEFT 37 */
    SDL_SCANCODE_UP,                    /* DOM_VK_UP 38 */
    SDL_SCANCODE_RIGHT,                 /* DOM_VK_RIGHT 39 */
    SDL_SCANCODE_DOWN,                  /* DOM_VK_DOWN 40 */
    SDL_SCANCODE_SELECT,                /* DOM_VK_SELECT 41 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_PRINT 42 */
    SDL_SCANCODE_EXECUTE,               /* DOM_VK_EXECUTE 43 */
    SDL_SCANCODE_PRINTSCREEN,           /* DOM_VK_PRINTSCREEN 44 */
    SDL_SCANCODE_INSERT,                /* DOM_VK_INSERT 45 */
    SDL_SCANCODE_DELETE,                /* DOM_VK_DELETE 46 */
    SDL_SCANCODE_UNKNOWN,               /* 47 */
    SDL_SCANCODE_0,                     /* DOM_VK_0 48 */
    SDL_SCANCODE_1,                     /* DOM_VK_1 49 */
    SDL_SCANCODE_2,                     /* DOM_VK_2 50 */
    SDL_SCANCODE_3,                     /* DOM_VK_3 51 */
    SDL_SCANCODE_4,                     /* DOM_VK_4 52 */
    SDL_SCANCODE_5,                     /* DOM_VK_5 53 */
    SDL_SCANCODE_6,                     /* DOM_VK_6 54 */
    SDL_SCANCODE_7,                     /* DOM_VK_7 55 */
    SDL_SCANCODE_8,                     /* DOM_VK_8 56 */
    SDL_SCANCODE_9,                     /* DOM_VK_9 57 */
    SDL_SCANCODE_KP_COLON,              /* DOM_VK_COLON 58 */
    SDL_SCANCODE_SEMICOLON,             /* DOM_VK_SEMICOLON 59 */
    SDL_SCANCODE_KP_LESS,               /* DOM_VK_LESS_THAN 60 */
    SDL_SCANCODE_EQUALS,                /* DOM_VK_EQUALS 61 */
    SDL_SCANCODE_KP_GREATER,            /* DOM_VK_GREATER_THAN 62 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_QUESTION_MARK 63 */
    SDL_SCANCODE_KP_AT,                 /* DOM_VK_AT 64 */
    SDL_SCANCODE_A,                     /* DOM_VK_A 65 */
    SDL_SCANCODE_B,                     /* DOM_VK_B 66 */
    SDL_SCANCODE_C,                     /* DOM_VK_C 67 */
    SDL_SCANCODE_D,                     /* DOM_VK_D 68 */
    SDL_SCANCODE_E,                     /* DOM_VK_E 69 */
    SDL_SCANCODE_F,                     /* DOM_VK_F 70 */
    SDL_SCANCODE_G,                     /* DOM_VK_G 71 */
    SDL_SCANCODE_H,                     /* DOM_VK_H 72 */
    SDL_SCANCODE_I,                     /* DOM_VK_I 73 */
    SDL_SCANCODE_J,                     /* DOM_VK_J 74 */
    SDL_SCANCODE_K,                     /* DOM_VK_K 75 */
    SDL_SCANCODE_L,                     /* DOM_VK_L 76 */
    SDL_SCANCODE_M,                     /* DOM_VK_M 77 */
    SDL_SCANCODE_N,                     /* DOM_VK_N 78 */
    SDL_SCANCODE_O,                     /* DOM_VK_O 79 */
    SDL_SCANCODE_P,                     /* DOM_VK_P 80 */
    SDL_SCANCODE_Q,                     /* DOM_VK_Q 81 */
    SDL_SCANCODE_R,                     /* DOM_VK_R 82 */
    SDL_SCANCODE_S,                     /* DOM_VK_S 83 */
    SDL_SCANCODE_T,                     /* DOM_VK_T 84 */
    SDL_SCANCODE_U,                     /* DOM_VK_U 85 */
    SDL_SCANCODE_V,                     /* DOM_VK_V 86 */
    SDL_SCANCODE_W,                     /* DOM_VK_W 87 */
    SDL_SCANCODE_X,                     /* DOM_VK_X 88 */
    SDL_SCANCODE_Y,                     /* DOM_VK_Y 89 */
    SDL_SCANCODE_Z,                     /* DOM_VK_Z 90 */
    SDL_SCANCODE_LGUI,                  /* DOM_VK_WIN 91 */
    SDL_SCANCODE_UNKNOWN,               /* 92 */
    SDL_SCANCODE_APPLICATION,           /* DOM_VK_CONTEXT_MENU 93 */
    SDL_SCANCODE_UNKNOWN,               /* 94 */
    SDL_SCANCODE_SLEEP,                 /* DOM_VK_SLEEP 95 */
    SDL_SCANCODE_KP_0,                  /* DOM_VK_NUMPAD0 96 */
    SDL_SCANCODE_KP_1,                  /* DOM_VK_NUMPAD1 97 */
    SDL_SCANCODE_KP_2,                  /* DOM_VK_NUMPAD2 98 */
    SDL_SCANCODE_KP_3,                  /* DOM_VK_NUMPAD3 99 */
    SDL_SCANCODE_KP_4,                  /* DOM_VK_NUMPAD4 100 */
    SDL_SCANCODE_KP_5,                  /* DOM_VK_NUMPAD5 101 */
    SDL_SCANCODE_KP_6,                  /* DOM_VK_NUMPAD6 102 */
    SDL_SCANCODE_KP_7,                  /* DOM_VK_NUMPAD7 103 */
    SDL_SCANCODE_KP_8,                  /* DOM_VK_NUMPAD8 104 */
    SDL_SCANCODE_KP_9,                  /* DOM_VK_NUMPAD9 105 */
    SDL_SCANCODE_KP_MULTIPLY,           /* DOM_VK_MULTIPLY 106 */
    SDL_SCANCODE_KP_PLUS,               /* DOM_VK_ADD 107 */
    SDL_SCANCODE_KP_COMMA,              /* DOM_VK_SEPARATOR 108 */
    SDL_SCANCODE_KP_MINUS,              /* DOM_VK_SUBTRACT 109 */
    SDL_SCANCODE_KP_PERIOD,             /* DOM_VK_DECIMAL 110 */
    SDL_SCANCODE_KP_DIVIDE,             /* DOM_VK_DIVIDE 111 */
    SDL_SCANCODE_F1,                    /* DOM_VK_F1 112 */
    SDL_SCANCODE_F2,                    /* DOM_VK_F2 113 */
    SDL_SCANCODE_F3,                    /* DOM_VK_F3 114 */
    SDL_SCANCODE_F4,                    /* DOM_VK_F4 115 */
    SDL_SCANCODE_F5,                    /* DOM_VK_F5 116 */
    SDL_SCANCODE_F6,                    /* DOM_VK_F6 117 */
    SDL_SCANCODE_F7,                    /* DOM_VK_F7 118 */
    SDL_SCANCODE_F8,                    /* DOM_VK_F8 119 */
    SDL_SCANCODE_F9,                    /* DOM_VK_F9 120 */
    SDL_SCANCODE_F10,                   /* DOM_VK_F10 121 */
    SDL_SCANCODE_F11,                   /* DOM_VK_F11 122 */
    SDL_SCANCODE_F12,                   /* DOM_VK_F12 123 */
    SDL_SCANCODE_F13,                   /* DOM_VK_F13 124 */
    SDL_SCANCODE_F14,                   /* DOM_VK_F14 125 */
    SDL_SCANCODE_F15,                   /* DOM_VK_F15 126 */
    SDL_SCANCODE_F16,                   /* DOM_VK_F16 127 */
    SDL_SCANCODE_F17,                   /* DOM_VK_F17 128 */
    SDL_SCANCODE_F18,                   /* DOM_VK_F18 129 */
    SDL_SCANCODE_F19,                   /* DOM_VK_F19 130 */
    SDL_SCANCODE_F20,                   /* DOM_VK_F20 131 */
    SDL_SCANCODE_F21,                   /* DOM_VK_F21 132 */
    SDL_SCANCODE_F22,                   /* DOM_VK_F22 133 */
    SDL_SCANCODE_F23,                   /* DOM_VK_F23 134 */
    SDL_SCANCODE_F24,                   /* DOM_VK_F24 135 */
    SDL_SCANCODE_UNKNOWN,               /* 136 */
    SDL_SCANCODE_UNKNOWN,               /* 137 */
    SDL_SCANCODE_UNKNOWN,               /* 138 */
    SDL_SCANCODE_UNKNOWN,               /* 139 */
    SDL_SCANCODE_UNKNOWN,               /* 140 */
    SDL_SCANCODE_UNKNOWN,               /* 141 */
    SDL_SCANCODE_UNKNOWN,               /* 142 */
    SDL_SCANCODE_UNKNOWN,               /* 143 */
    SDL_SCANCODE_NUMLOCKCLEAR,          /* DOM_VK_NUM_LOCK 144 */
    SDL_SCANCODE_SCROLLLOCK,            /* DOM_VK_SCROLL_LOCK 145 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_FJ_JISHO 146 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_FJ_MASSHOU 147 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_FJ_TOUROKU 148 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_FJ_LOYA 149 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_FJ_ROYA 150 */
    SDL_SCANCODE_UNKNOWN,               /* 151 */
    SDL_SCANCODE_UNKNOWN,               /* 152 */
    SDL_SCANCODE_UNKNOWN,               /* 153 */
    SDL_SCANCODE_UNKNOWN,               /* 154 */
    SDL_SCANCODE_UNKNOWN,               /* 155 */
    SDL_SCANCODE_UNKNOWN,               /* 156 */
    SDL_SCANCODE_UNKNOWN,               /* 157 */
    SDL_SCANCODE_UNKNOWN,               /* 158 */
    SDL_SCANCODE_UNKNOWN,               /* 159 */
    SDL_SCANCODE_GRAVE,                 /* DOM_VK_CIRCUMFLEX 160 */
    SDL_SCANCODE_KP_EXCLAM,             /* DOM_VK_EXCLAMATION 161 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_DOUBLE_QUOTE 162 */
    SDL_SCANCODE_KP_HASH,               /* DOM_VK_HASH 163 */
    SDL_SCANCODE_CURRENCYUNIT,          /* DOM_VK_DOLLAR 164 */
    SDL_SCANCODE_KP_PERCENT,            /* DOM_VK_PERCENT 165 */
    SDL_SCANCODE_KP_AMPERSAND,          /* DOM_VK_AMPERSAND 166 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_UNDERSCORE 167 */
    SDL_SCANCODE_KP_LEFTPAREN,          /* DOM_VK_OPEN_PAREN 168 */
    SDL_SCANCODE_KP_RIGHTPAREN,         /* DOM_VK_CLOSE_PAREN 169 */
    SDL_SCANCODE_KP_MULTIPLY,           /* DOM_VK_ASTERISK 170 */
    SDL_SCANCODE_KP_PLUS,               /* DOM_VK_PLUS 171 */
    SDL_SCANCODE_KP_PLUS,               /* DOM_VK_PIPE 172 */
    SDL_SCANCODE_MINUS,                 /* DOM_VK_HYPHEN_MINUS 173 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_OPEN_CURLY_BRACKET 174 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_CLOSE_CURLY_BRACKET 175 */
    SDL_SCANCODE_NONUSBACKSLASH,        /* DOM_VK_TILDE 176 */
    SDL_SCANCODE_UNKNOWN,               /* 177 */
    SDL_SCANCODE_UNKNOWN,               /* 178 */
    SDL_SCANCODE_UNKNOWN,               /* 179 */
    SDL_SCANCODE_UNKNOWN,               /* 180 */
    SDL_SCANCODE_MUTE,                  /* DOM_VK_VOLUME_MUTE 181 */
    SDL_SCANCODE_VOLUMEDOWN,            /* DOM_VK_VOLUME_DOWN 182 */
    SDL_SCANCODE_VOLUMEUP,              /* DOM_VK_VOLUME_UP 183 */
    SDL_SCANCODE_UNKNOWN,               /* 184 */
    SDL_SCANCODE_UNKNOWN,               /* 185 */
    SDL_SCANCODE_UNKNOWN,               /* 186 */
    SDL_SCANCODE_UNKNOWN,               /* 187 */
    SDL_SCANCODE_COMMA,                 /* DOM_VK_COMMA 188 */
    SDL_SCANCODE_UNKNOWN,               /* 189 */
    SDL_SCANCODE_PERIOD,                /* DOM_VK_PERIOD 190 */
    SDL_SCANCODE_SLASH,                 /* DOM_VK_SLASH 191 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_BACK_QUOTE 192 */
    SDL_SCANCODE_UNKNOWN,               /* 193 */
    SDL_SCANCODE_UNKNOWN,               /* 194 */
    SDL_SCANCODE_UNKNOWN,               /* 195 */
    SDL_SCANCODE_UNKNOWN,               /* 196 */
    SDL_SCANCODE_UNKNOWN,               /* 197 */
    SDL_SCANCODE_UNKNOWN,               /* 198 */
    SDL_SCANCODE_UNKNOWN,               /* 199 */
    SDL_SCANCODE_UNKNOWN,               /* 200 */
    SDL_SCANCODE_UNKNOWN,               /* 201 */
    SDL_SCANCODE_UNKNOWN,               /* 202 */
    SDL_SCANCODE_UNKNOWN,               /* 203 */
    SDL_SCANCODE_UNKNOWN,               /* 204 */
    SDL_SCANCODE_UNKNOWN,               /* 205 */
    SDL_SCANCODE_UNKNOWN,               /* 206 */
    SDL_SCANCODE_UNKNOWN,               /* 207 */
    SDL_SCANCODE_UNKNOWN,               /* 208 */
    SDL_SCANCODE_UNKNOWN,               /* 209 */
    SDL_SCANCODE_UNKNOWN,               /* 210 */
    SDL_SCANCODE_UNKNOWN,               /* 211 */
    SDL_SCANCODE_UNKNOWN,               /* 212 */
    SDL_SCANCODE_UNKNOWN,               /* 213 */
    SDL_SCANCODE_UNKNOWN,               /* 214 */
    SDL_SCANCODE_UNKNOWN,               /* 215 */
    SDL_SCANCODE_UNKNOWN,               /* 216 */
    SDL_SCANCODE_UNKNOWN,               /* 217 */
    SDL_SCANCODE_UNKNOWN,               /* 218 */
    SDL_SCANCODE_LEFTBRACKET,           /* DOM_VK_OPEN_BRACKET 219 */
    SDL_SCANCODE_BACKSLASH,             /* DOM_VK_BACK_SLASH 220 */
    SDL_SCANCODE_RIGHTBRACKET,          /* DOM_VK_CLOSE_BRACKET 221 */
    SDL_SCANCODE_APOSTROPHE,            /* DOM_VK_QUOTE 222 */
    SDL_SCANCODE_UNKNOWN,               /* 223 */
    SDL_SCANCODE_RGUI,                  /* DOM_VK_META 224 */
    SDL_SCANCODE_RALT,                  /* DOM_VK_ALTGR 225 */
    SDL_SCANCODE_UNKNOWN,               /* 226 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_ICO_HELP 227 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_ICO_00 228 */
    SDL_SCANCODE_UNKNOWN,               /* 229 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_ICO_CLEAR 230 */
    SDL_SCANCODE_UNKNOWN,               /* 231 */
    SDL_SCANCODE_UNKNOWN,               /* 232 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_RESET 233 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_JUMP 234 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_PA1 235 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_PA2 236 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_PA3 237 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_WSCTRL 238 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_CUSEL 239 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_ATTN 240 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_FINISH 241 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_COPY 242 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_AUTO 243 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_ENLW 244 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_BACKTAB 245 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_ATTN 246 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_CRSEL 247 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_EXSEL 248 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_EREOF 249 */
    SDL_SCANCODE_AUDIOPLAY,             /* DOM_VK_PLAY 250 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_ZOOM 251 */
    SDL_SCANCODE_UNKNOWN,               /* 252 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_PA1 253 */
    SDL_SCANCODE_UNKNOWN,               /* DOM_VK_WIN_OEM_CLEAR 254 */
    SDL_SCANCODE_UNKNOWN,               /* 255 */
};

static Uint8 SDL_NACL_translate_mouse_button(int32_t button) {
    switch (button) {
        case PP_INPUTEVENT_MOUSEBUTTON_LEFT:
            return SDL_BUTTON_LEFT;
        case PP_INPUTEVENT_MOUSEBUTTON_MIDDLE:
            return SDL_BUTTON_MIDDLE;
        case PP_INPUTEVENT_MOUSEBUTTON_RIGHT:
            return SDL_BUTTON_RIGHT;

        case PP_INPUTEVENT_MOUSEBUTTON_NONE:
        default:
            return 0;
    }
}

static SDL_Scancode SDL_NACL_translate_keycode(int keycode)
{
    SDL_Scancode scancode = SDL_SCANCODE_UNKNOWN;

    if (keycode < SDL_arraysize(NACL_Keycodes)) {
        scancode = NACL_Keycodes[keycode];
    }
#ifdef DEBUG_SCANCODES
    if (scancode == SDL_SCANCODE_UNKNOWN) {
        SDL_Log("The key you just pressed is not recognized by SDL. To help get this fixed, please report this to the SDL forums/mailing list <https://discourse.libsdl.org/> NACL KeyCode %d", keycode);
    }
#endif
    return scancode;
}

void NACL_PumpEvents(_THIS) {
  PSEvent* ps_event;
  PP_Resource event;
  PP_InputEvent_Type type;
  PP_InputEvent_Modifier modifiers;
  struct PP_Rect rect;
  struct PP_FloatPoint fp;
  struct PP_Point location;
  struct PP_Var var;
  const char *str;
  char text[SDL_TEXTINPUTEVENT_TEXT_SIZE];
  Uint32 str_len;
  SDL_VideoData *driverdata = (SDL_VideoData *) _this->driverdata;
  SDL_Mouse *mouse = SDL_GetMouse();

  if (driverdata->window) {
    while ((ps_event = PSEventTryAcquire()) != NULL) {
        event = ps_event->as_resource;
        switch(ps_event->type) {
            /* From DidChangeView, contains a view resource */
            case PSE_INSTANCE_DIDCHANGEVIEW:
                driverdata->ppb_view->GetRect(event, &rect);
                NACL_SetScreenResolution(rect.size.width, rect.size.height, SDL_PIXELFORMAT_UNKNOWN);
                // FIXME: Rebuild context? See life.c UpdateContext
                break;

            /* From HandleInputEvent, contains an input resource. */
            case PSE_INSTANCE_HANDLEINPUT:
                type = driverdata->ppb_input_event->GetType(event);
                modifiers = driverdata->ppb_input_event->GetModifiers(event);
                switch(type) {
                    case PP_INPUTEVENT_TYPE_MOUSEDOWN:
                        SDL_SendMouseButton(mouse->focus, mouse->mouseID, SDL_PRESSED, SDL_NACL_translate_mouse_button(driverdata->ppb_mouse_input_event->GetButton(event)));
                        break;
                    case PP_INPUTEVENT_TYPE_MOUSEUP:
                        SDL_SendMouseButton(mouse->focus, mouse->mouseID, SDL_RELEASED, SDL_NACL_translate_mouse_button(driverdata->ppb_mouse_input_event->GetButton(event)));
                        break;
                    case PP_INPUTEVENT_TYPE_WHEEL:
                        /* FIXME: GetTicks provides high resolution scroll events */
                        fp = driverdata->ppb_wheel_input_event->GetDelta(event);
                        SDL_SendMouseWheel(mouse->focus, mouse->mouseID, fp.x, fp.y, SDL_MOUSEWHEEL_NORMAL);
                        break;

                    case PP_INPUTEVENT_TYPE_MOUSEENTER:
                    case PP_INPUTEVENT_TYPE_MOUSELEAVE:
                        /* FIXME: Mouse Focus */
                        break;


                    case PP_INPUTEVENT_TYPE_MOUSEMOVE:
                        location = driverdata->ppb_mouse_input_event->GetPosition(event);
                        SDL_SendMouseMotion(mouse->focus, mouse->mouseID, SDL_FALSE, location.x, location.y);
                        break;

                    case PP_INPUTEVENT_TYPE_TOUCHSTART:
                    case PP_INPUTEVENT_TYPE_TOUCHMOVE:
                    case PP_INPUTEVENT_TYPE_TOUCHEND:
                    case PP_INPUTEVENT_TYPE_TOUCHCANCEL:
                        /* FIXME: Touch events */
                        break;

                    case PP_INPUTEVENT_TYPE_KEYDOWN:
                        SDL_SendKeyboardKey(SDL_PRESSED, SDL_NACL_translate_keycode(driverdata->ppb_keyboard_input_event->GetKeyCode(event)));
                        break;

                    case PP_INPUTEVENT_TYPE_KEYUP:
                        SDL_SendKeyboardKey(SDL_RELEASED, SDL_NACL_translate_keycode(driverdata->ppb_keyboard_input_event->GetKeyCode(event)));
                        break;

                    case PP_INPUTEVENT_TYPE_CHAR:
                        var = driverdata->ppb_keyboard_input_event->GetCharacterText(event);
                        str = driverdata->ppb_var->VarToUtf8(var, &str_len);
                        /* str is not null terminated! */
                        if ( str_len >= SDL_arraysize(text) ) {
                            str_len = SDL_arraysize(text) - 1;
                        }
                        SDL_strlcpy(text, str, str_len );
                        text[str_len] = '\0';

                        SDL_SendKeyboardText(text);
                        /* FIXME: Do we have to handle ref counting? driverdata->ppb_var->Release(var);*/
                        break;

                    default:
                        break;
                }
                break;


            /* From HandleMessage, contains a PP_Var. */
            case PSE_INSTANCE_HANDLEMESSAGE:
                break;

            /* From DidChangeFocus, contains a PP_Bool with the current focus state. */
            case PSE_INSTANCE_DIDCHANGEFOCUS:
                break;

            /* When the 3D context is lost, no resource. */
            case PSE_GRAPHICS3D_GRAPHICS3DCONTEXTLOST:
                break;

            /* When the mouse lock is lost. */
            case PSE_MOUSELOCK_MOUSELOCKLOST:
                break;

            default:
                break;
        }

        PSEventRelease(ps_event);
    }
  }
}

#endif /* SDL_VIDEO_DRIVER_NACL */

/* vi: set ts=4 sw=4 expandtab: */
