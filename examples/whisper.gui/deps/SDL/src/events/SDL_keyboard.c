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
#include "../SDL_internal.h"

/* General keyboard handling code for SDL */

#include "SDL_hints.h"
#include "SDL_timer.h"
#include "SDL_events.h"
#include "SDL_events_c.h"
#include "../video/SDL_sysvideo.h"
#include "scancodes_ascii.h"

/* #define DEBUG_KEYBOARD */

/* Global keyboard information */

#define KEYBOARD_HARDWARE    0x01
#define KEYBOARD_VIRTUAL     0x02
#define KEYBOARD_AUTORELEASE 0x04

typedef struct SDL_Keyboard SDL_Keyboard;

struct SDL_Keyboard
{
    /* Data common to all keyboards */
    SDL_Window *focus;
    Uint16 modstate;
    Uint8 keysource[SDL_NUM_SCANCODES];
    Uint8 keystate[SDL_NUM_SCANCODES];
    SDL_Keycode keymap[SDL_NUM_SCANCODES];
    SDL_bool autorelease_pending;
    Uint32 hardware_timestamp;
};

static SDL_Keyboard SDL_keyboard;

static const SDL_Keycode SDL_default_keymap[SDL_NUM_SCANCODES] = {
    /* 0 */ 0,
    /* 1 */ 0,
    /* 2 */ 0,
    /* 3 */ 0,
    /* 4 */ 'a',
    /* 5 */ 'b',
    /* 6 */ 'c',
    /* 7 */ 'd',
    /* 8 */ 'e',
    /* 9 */ 'f',
    /* 10 */ 'g',
    /* 11 */ 'h',
    /* 12 */ 'i',
    /* 13 */ 'j',
    /* 14 */ 'k',
    /* 15 */ 'l',
    /* 16 */ 'm',
    /* 17 */ 'n',
    /* 18 */ 'o',
    /* 19 */ 'p',
    /* 20 */ 'q',
    /* 21 */ 'r',
    /* 22 */ 's',
    /* 23 */ 't',
    /* 24 */ 'u',
    /* 25 */ 'v',
    /* 26 */ 'w',
    /* 27 */ 'x',
    /* 28 */ 'y',
    /* 29 */ 'z',
    /* 30 */ '1',
    /* 31 */ '2',
    /* 32 */ '3',
    /* 33 */ '4',
    /* 34 */ '5',
    /* 35 */ '6',
    /* 36 */ '7',
    /* 37 */ '8',
    /* 38 */ '9',
    /* 39 */ '0',
    /* 40 */ SDLK_RETURN,
    /* 41 */ SDLK_ESCAPE,
    /* 42 */ SDLK_BACKSPACE,
    /* 43 */ SDLK_TAB,
    /* 44 */ SDLK_SPACE,
    /* 45 */ '-',
    /* 46 */ '=',
    /* 47 */ '[',
    /* 48 */ ']',
    /* 49 */ '\\',
    /* 50 */ '#',
    /* 51 */ ';',
    /* 52 */ '\'',
    /* 53 */ '`',
    /* 54 */ ',',
    /* 55 */ '.',
    /* 56 */ '/',
    /* 57 */ SDLK_CAPSLOCK,
    /* 58 */ SDLK_F1,
    /* 59 */ SDLK_F2,
    /* 60 */ SDLK_F3,
    /* 61 */ SDLK_F4,
    /* 62 */ SDLK_F5,
    /* 63 */ SDLK_F6,
    /* 64 */ SDLK_F7,
    /* 65 */ SDLK_F8,
    /* 66 */ SDLK_F9,
    /* 67 */ SDLK_F10,
    /* 68 */ SDLK_F11,
    /* 69 */ SDLK_F12,
    /* 70 */ SDLK_PRINTSCREEN,
    /* 71 */ SDLK_SCROLLLOCK,
    /* 72 */ SDLK_PAUSE,
    /* 73 */ SDLK_INSERT,
    /* 74 */ SDLK_HOME,
    /* 75 */ SDLK_PAGEUP,
    /* 76 */ SDLK_DELETE,
    /* 77 */ SDLK_END,
    /* 78 */ SDLK_PAGEDOWN,
    /* 79 */ SDLK_RIGHT,
    /* 80 */ SDLK_LEFT,
    /* 81 */ SDLK_DOWN,
    /* 82 */ SDLK_UP,
    /* 83 */ SDLK_NUMLOCKCLEAR,
    /* 84 */ SDLK_KP_DIVIDE,
    /* 85 */ SDLK_KP_MULTIPLY,
    /* 86 */ SDLK_KP_MINUS,
    /* 87 */ SDLK_KP_PLUS,
    /* 88 */ SDLK_KP_ENTER,
    /* 89 */ SDLK_KP_1,
    /* 90 */ SDLK_KP_2,
    /* 91 */ SDLK_KP_3,
    /* 92 */ SDLK_KP_4,
    /* 93 */ SDLK_KP_5,
    /* 94 */ SDLK_KP_6,
    /* 95 */ SDLK_KP_7,
    /* 96 */ SDLK_KP_8,
    /* 97 */ SDLK_KP_9,
    /* 98 */ SDLK_KP_0,
    /* 99 */ SDLK_KP_PERIOD,
    /* 100 */ 0,
    /* 101 */ SDLK_APPLICATION,
    /* 102 */ SDLK_POWER,
    /* 103 */ SDLK_KP_EQUALS,
    /* 104 */ SDLK_F13,
    /* 105 */ SDLK_F14,
    /* 106 */ SDLK_F15,
    /* 107 */ SDLK_F16,
    /* 108 */ SDLK_F17,
    /* 109 */ SDLK_F18,
    /* 110 */ SDLK_F19,
    /* 111 */ SDLK_F20,
    /* 112 */ SDLK_F21,
    /* 113 */ SDLK_F22,
    /* 114 */ SDLK_F23,
    /* 115 */ SDLK_F24,
    /* 116 */ SDLK_EXECUTE,
    /* 117 */ SDLK_HELP,
    /* 118 */ SDLK_MENU,
    /* 119 */ SDLK_SELECT,
    /* 120 */ SDLK_STOP,
    /* 121 */ SDLK_AGAIN,
    /* 122 */ SDLK_UNDO,
    /* 123 */ SDLK_CUT,
    /* 124 */ SDLK_COPY,
    /* 125 */ SDLK_PASTE,
    /* 126 */ SDLK_FIND,
    /* 127 */ SDLK_MUTE,
    /* 128 */ SDLK_VOLUMEUP,
    /* 129 */ SDLK_VOLUMEDOWN,
    /* 130 */ 0,
    /* 131 */ 0,
    /* 132 */ 0,
    /* 133 */ SDLK_KP_COMMA,
    /* 134 */ SDLK_KP_EQUALSAS400,
    /* 135 */ 0,
    /* 136 */ 0,
    /* 137 */ 0,
    /* 138 */ 0,
    /* 139 */ 0,
    /* 140 */ 0,
    /* 141 */ 0,
    /* 142 */ 0,
    /* 143 */ 0,
    /* 144 */ 0,
    /* 145 */ 0,
    /* 146 */ 0,
    /* 147 */ 0,
    /* 148 */ 0,
    /* 149 */ 0,
    /* 150 */ 0,
    /* 151 */ 0,
    /* 152 */ 0,
    /* 153 */ SDLK_ALTERASE,
    /* 154 */ SDLK_SYSREQ,
    /* 155 */ SDLK_CANCEL,
    /* 156 */ SDLK_CLEAR,
    /* 157 */ SDLK_PRIOR,
    /* 158 */ SDLK_RETURN2,
    /* 159 */ SDLK_SEPARATOR,
    /* 160 */ SDLK_OUT,
    /* 161 */ SDLK_OPER,
    /* 162 */ SDLK_CLEARAGAIN,
    /* 163 */ SDLK_CRSEL,
    /* 164 */ SDLK_EXSEL,
    /* 165 */ 0,
    /* 166 */ 0,
    /* 167 */ 0,
    /* 168 */ 0,
    /* 169 */ 0,
    /* 170 */ 0,
    /* 171 */ 0,
    /* 172 */ 0,
    /* 173 */ 0,
    /* 174 */ 0,
    /* 175 */ 0,
    /* 176 */ SDLK_KP_00,
    /* 177 */ SDLK_KP_000,
    /* 178 */ SDLK_THOUSANDSSEPARATOR,
    /* 179 */ SDLK_DECIMALSEPARATOR,
    /* 180 */ SDLK_CURRENCYUNIT,
    /* 181 */ SDLK_CURRENCYSUBUNIT,
    /* 182 */ SDLK_KP_LEFTPAREN,
    /* 183 */ SDLK_KP_RIGHTPAREN,
    /* 184 */ SDLK_KP_LEFTBRACE,
    /* 185 */ SDLK_KP_RIGHTBRACE,
    /* 186 */ SDLK_KP_TAB,
    /* 187 */ SDLK_KP_BACKSPACE,
    /* 188 */ SDLK_KP_A,
    /* 189 */ SDLK_KP_B,
    /* 190 */ SDLK_KP_C,
    /* 191 */ SDLK_KP_D,
    /* 192 */ SDLK_KP_E,
    /* 193 */ SDLK_KP_F,
    /* 194 */ SDLK_KP_XOR,
    /* 195 */ SDLK_KP_POWER,
    /* 196 */ SDLK_KP_PERCENT,
    /* 197 */ SDLK_KP_LESS,
    /* 198 */ SDLK_KP_GREATER,
    /* 199 */ SDLK_KP_AMPERSAND,
    /* 200 */ SDLK_KP_DBLAMPERSAND,
    /* 201 */ SDLK_KP_VERTICALBAR,
    /* 202 */ SDLK_KP_DBLVERTICALBAR,
    /* 203 */ SDLK_KP_COLON,
    /* 204 */ SDLK_KP_HASH,
    /* 205 */ SDLK_KP_SPACE,
    /* 206 */ SDLK_KP_AT,
    /* 207 */ SDLK_KP_EXCLAM,
    /* 208 */ SDLK_KP_MEMSTORE,
    /* 209 */ SDLK_KP_MEMRECALL,
    /* 210 */ SDLK_KP_MEMCLEAR,
    /* 211 */ SDLK_KP_MEMADD,
    /* 212 */ SDLK_KP_MEMSUBTRACT,
    /* 213 */ SDLK_KP_MEMMULTIPLY,
    /* 214 */ SDLK_KP_MEMDIVIDE,
    /* 215 */ SDLK_KP_PLUSMINUS,
    /* 216 */ SDLK_KP_CLEAR,
    /* 217 */ SDLK_KP_CLEARENTRY,
    /* 218 */ SDLK_KP_BINARY,
    /* 219 */ SDLK_KP_OCTAL,
    /* 220 */ SDLK_KP_DECIMAL,
    /* 221 */ SDLK_KP_HEXADECIMAL,
    /* 222 */ 0,
    /* 223 */ 0,
    /* 224 */ SDLK_LCTRL,
    /* 225 */ SDLK_LSHIFT,
    /* 226 */ SDLK_LALT,
    /* 227 */ SDLK_LGUI,
    /* 228 */ SDLK_RCTRL,
    /* 229 */ SDLK_RSHIFT,
    /* 230 */ SDLK_RALT,
    /* 231 */ SDLK_RGUI,
    /* 232 */ 0,
    /* 233 */ 0,
    /* 234 */ 0,
    /* 235 */ 0,
    /* 236 */ 0,
    /* 237 */ 0,
    /* 238 */ 0,
    /* 239 */ 0,
    /* 240 */ 0,
    /* 241 */ 0,
    /* 242 */ 0,
    /* 243 */ 0,
    /* 244 */ 0,
    /* 245 */ 0,
    /* 246 */ 0,
    /* 247 */ 0,
    /* 248 */ 0,
    /* 249 */ 0,
    /* 250 */ 0,
    /* 251 */ 0,
    /* 252 */ 0,
    /* 253 */ 0,
    /* 254 */ 0,
    /* 255 */ 0,
    /* 256 */ 0,
    /* 257 */ SDLK_MODE,
    /* 258 */ SDLK_AUDIONEXT,
    /* 259 */ SDLK_AUDIOPREV,
    /* 260 */ SDLK_AUDIOSTOP,
    /* 261 */ SDLK_AUDIOPLAY,
    /* 262 */ SDLK_AUDIOMUTE,
    /* 263 */ SDLK_MEDIASELECT,
    /* 264 */ SDLK_WWW,
    /* 265 */ SDLK_MAIL,
    /* 266 */ SDLK_CALCULATOR,
    /* 267 */ SDLK_COMPUTER,
    /* 268 */ SDLK_AC_SEARCH,
    /* 269 */ SDLK_AC_HOME,
    /* 270 */ SDLK_AC_BACK,
    /* 271 */ SDLK_AC_FORWARD,
    /* 272 */ SDLK_AC_STOP,
    /* 273 */ SDLK_AC_REFRESH,
    /* 274 */ SDLK_AC_BOOKMARKS,
    /* 275 */ SDLK_BRIGHTNESSDOWN,
    /* 276 */ SDLK_BRIGHTNESSUP,
    /* 277 */ SDLK_DISPLAYSWITCH,
    /* 278 */ SDLK_KBDILLUMTOGGLE,
    /* 279 */ SDLK_KBDILLUMDOWN,
    /* 280 */ SDLK_KBDILLUMUP,
    /* 281 */ SDLK_EJECT,
    /* 282 */ SDLK_SLEEP,
    /* 283 */ SDLK_APP1,
    /* 284 */ SDLK_APP2,
    /* 285 */ SDLK_AUDIOREWIND,
    /* 286 */ SDLK_AUDIOFASTFORWARD,
    /* 287 */ SDLK_SOFTLEFT,
    /* 288 */ SDLK_SOFTRIGHT,
    /* 289 */ SDLK_CALL,
    /* 290 */ SDLK_ENDCALL,
};

static const char *SDL_scancode_names[SDL_NUM_SCANCODES] = {
    /* 0 */ NULL,
    /* 1 */ NULL,
    /* 2 */ NULL,
    /* 3 */ NULL,
    /* 4 */ "A",
    /* 5 */ "B",
    /* 6 */ "C",
    /* 7 */ "D",
    /* 8 */ "E",
    /* 9 */ "F",
    /* 10 */ "G",
    /* 11 */ "H",
    /* 12 */ "I",
    /* 13 */ "J",
    /* 14 */ "K",
    /* 15 */ "L",
    /* 16 */ "M",
    /* 17 */ "N",
    /* 18 */ "O",
    /* 19 */ "P",
    /* 20 */ "Q",
    /* 21 */ "R",
    /* 22 */ "S",
    /* 23 */ "T",
    /* 24 */ "U",
    /* 25 */ "V",
    /* 26 */ "W",
    /* 27 */ "X",
    /* 28 */ "Y",
    /* 29 */ "Z",
    /* 30 */ "1",
    /* 31 */ "2",
    /* 32 */ "3",
    /* 33 */ "4",
    /* 34 */ "5",
    /* 35 */ "6",
    /* 36 */ "7",
    /* 37 */ "8",
    /* 38 */ "9",
    /* 39 */ "0",
    /* 40 */ "Return",
    /* 41 */ "Escape",
    /* 42 */ "Backspace",
    /* 43 */ "Tab",
    /* 44 */ "Space",
    /* 45 */ "-",
    /* 46 */ "=",
    /* 47 */ "[",
    /* 48 */ "]",
    /* 49 */ "\\",
    /* 50 */ "#",
    /* 51 */ ";",
    /* 52 */ "'",
    /* 53 */ "`",
    /* 54 */ ",",
    /* 55 */ ".",
    /* 56 */ "/",
    /* 57 */ "CapsLock",
    /* 58 */ "F1",
    /* 59 */ "F2",
    /* 60 */ "F3",
    /* 61 */ "F4",
    /* 62 */ "F5",
    /* 63 */ "F6",
    /* 64 */ "F7",
    /* 65 */ "F8",
    /* 66 */ "F9",
    /* 67 */ "F10",
    /* 68 */ "F11",
    /* 69 */ "F12",
    /* 70 */ "PrintScreen",
    /* 71 */ "ScrollLock",
    /* 72 */ "Pause",
    /* 73 */ "Insert",
    /* 74 */ "Home",
    /* 75 */ "PageUp",
    /* 76 */ "Delete",
    /* 77 */ "End",
    /* 78 */ "PageDown",
    /* 79 */ "Right",
    /* 80 */ "Left",
    /* 81 */ "Down",
    /* 82 */ "Up",
    /* 83 */ "Numlock",
    /* 84 */ "Keypad /",
    /* 85 */ "Keypad *",
    /* 86 */ "Keypad -",
    /* 87 */ "Keypad +",
    /* 88 */ "Keypad Enter",
    /* 89 */ "Keypad 1",
    /* 90 */ "Keypad 2",
    /* 91 */ "Keypad 3",
    /* 92 */ "Keypad 4",
    /* 93 */ "Keypad 5",
    /* 94 */ "Keypad 6",
    /* 95 */ "Keypad 7",
    /* 96 */ "Keypad 8",
    /* 97 */ "Keypad 9",
    /* 98 */ "Keypad 0",
    /* 99 */ "Keypad .",
    /* 100 */ NULL,
    /* 101 */ "Application",
    /* 102 */ "Power",
    /* 103 */ "Keypad =",
    /* 104 */ "F13",
    /* 105 */ "F14",
    /* 106 */ "F15",
    /* 107 */ "F16",
    /* 108 */ "F17",
    /* 109 */ "F18",
    /* 110 */ "F19",
    /* 111 */ "F20",
    /* 112 */ "F21",
    /* 113 */ "F22",
    /* 114 */ "F23",
    /* 115 */ "F24",
    /* 116 */ "Execute",
    /* 117 */ "Help",
    /* 118 */ "Menu",
    /* 119 */ "Select",
    /* 120 */ "Stop",
    /* 121 */ "Again",
    /* 122 */ "Undo",
    /* 123 */ "Cut",
    /* 124 */ "Copy",
    /* 125 */ "Paste",
    /* 126 */ "Find",
    /* 127 */ "Mute",
    /* 128 */ "VolumeUp",
    /* 129 */ "VolumeDown",
    /* 130 */ NULL,
    /* 131 */ NULL,
    /* 132 */ NULL,
    /* 133 */ "Keypad ,",
    /* 134 */ "Keypad = (AS400)",
    /* 135 */ NULL,
    /* 136 */ NULL,
    /* 137 */ NULL,
    /* 138 */ NULL,
    /* 139 */ NULL,
    /* 140 */ NULL,
    /* 141 */ NULL,
    /* 142 */ NULL,
    /* 143 */ NULL,
    /* 144 */ NULL,
    /* 145 */ NULL,
    /* 146 */ NULL,
    /* 147 */ NULL,
    /* 148 */ NULL,
    /* 149 */ NULL,
    /* 150 */ NULL,
    /* 151 */ NULL,
    /* 152 */ NULL,
    /* 153 */ "AltErase",
    /* 154 */ "SysReq",
    /* 155 */ "Cancel",
    /* 156 */ "Clear",
    /* 157 */ "Prior",
    /* 158 */ "Return",
    /* 159 */ "Separator",
    /* 160 */ "Out",
    /* 161 */ "Oper",
    /* 162 */ "Clear / Again",
    /* 163 */ "CrSel",
    /* 164 */ "ExSel",
    /* 165 */ NULL,
    /* 166 */ NULL,
    /* 167 */ NULL,
    /* 168 */ NULL,
    /* 169 */ NULL,
    /* 170 */ NULL,
    /* 171 */ NULL,
    /* 172 */ NULL,
    /* 173 */ NULL,
    /* 174 */ NULL,
    /* 175 */ NULL,
    /* 176 */ "Keypad 00",
    /* 177 */ "Keypad 000",
    /* 178 */ "ThousandsSeparator",
    /* 179 */ "DecimalSeparator",
    /* 180 */ "CurrencyUnit",
    /* 181 */ "CurrencySubUnit",
    /* 182 */ "Keypad (",
    /* 183 */ "Keypad )",
    /* 184 */ "Keypad {",
    /* 185 */ "Keypad }",
    /* 186 */ "Keypad Tab",
    /* 187 */ "Keypad Backspace",
    /* 188 */ "Keypad A",
    /* 189 */ "Keypad B",
    /* 190 */ "Keypad C",
    /* 191 */ "Keypad D",
    /* 192 */ "Keypad E",
    /* 193 */ "Keypad F",
    /* 194 */ "Keypad XOR",
    /* 195 */ "Keypad ^",
    /* 196 */ "Keypad %",
    /* 197 */ "Keypad <",
    /* 198 */ "Keypad >",
    /* 199 */ "Keypad &",
    /* 200 */ "Keypad &&",
    /* 201 */ "Keypad |",
    /* 202 */ "Keypad ||",
    /* 203 */ "Keypad :",
    /* 204 */ "Keypad #",
    /* 205 */ "Keypad Space",
    /* 206 */ "Keypad @",
    /* 207 */ "Keypad !",
    /* 208 */ "Keypad MemStore",
    /* 209 */ "Keypad MemRecall",
    /* 210 */ "Keypad MemClear",
    /* 211 */ "Keypad MemAdd",
    /* 212 */ "Keypad MemSubtract",
    /* 213 */ "Keypad MemMultiply",
    /* 214 */ "Keypad MemDivide",
    /* 215 */ "Keypad +/-",
    /* 216 */ "Keypad Clear",
    /* 217 */ "Keypad ClearEntry",
    /* 218 */ "Keypad Binary",
    /* 219 */ "Keypad Octal",
    /* 220 */ "Keypad Decimal",
    /* 221 */ "Keypad Hexadecimal",
    /* 222 */ NULL,
    /* 223 */ NULL,
    /* 224 */ "Left Ctrl",
    /* 225 */ "Left Shift",
    /* 226 */ "Left Alt",
    /* 227 */ "Left GUI",
    /* 228 */ "Right Ctrl",
    /* 229 */ "Right Shift",
    /* 230 */ "Right Alt",
    /* 231 */ "Right GUI",
    /* 232 */ NULL,
    /* 233 */ NULL,
    /* 234 */ NULL,
    /* 235 */ NULL,
    /* 236 */ NULL,
    /* 237 */ NULL,
    /* 238 */ NULL,
    /* 239 */ NULL,
    /* 240 */ NULL,
    /* 241 */ NULL,
    /* 242 */ NULL,
    /* 243 */ NULL,
    /* 244 */ NULL,
    /* 245 */ NULL,
    /* 246 */ NULL,
    /* 247 */ NULL,
    /* 248 */ NULL,
    /* 249 */ NULL,
    /* 250 */ NULL,
    /* 251 */ NULL,
    /* 252 */ NULL,
    /* 253 */ NULL,
    /* 254 */ NULL,
    /* 255 */ NULL,
    /* 256 */ NULL,
    /* 257 */ "ModeSwitch",
    /* 258 */ "AudioNext",
    /* 259 */ "AudioPrev",
    /* 260 */ "AudioStop",
    /* 261 */ "AudioPlay",
    /* 262 */ "AudioMute",
    /* 263 */ "MediaSelect",
    /* 264 */ "WWW",
    /* 265 */ "Mail",
    /* 266 */ "Calculator",
    /* 267 */ "Computer",
    /* 268 */ "AC Search",
    /* 269 */ "AC Home",
    /* 270 */ "AC Back",
    /* 271 */ "AC Forward",
    /* 272 */ "AC Stop",
    /* 273 */ "AC Refresh",
    /* 274 */ "AC Bookmarks",
    /* 275 */ "BrightnessDown",
    /* 276 */ "BrightnessUp",
    /* 277 */ "DisplaySwitch",
    /* 278 */ "KBDIllumToggle",
    /* 279 */ "KBDIllumDown",
    /* 280 */ "KBDIllumUp",
    /* 281 */ "Eject",
    /* 282 */ "Sleep",
    /* 283 */ "App1",
    /* 284 */ "App2",
    /* 285 */ "AudioRewind",
    /* 286 */ "AudioFastForward",
    /* 287 */ "SoftLeft",
    /* 288 */ "SoftRight",
    /* 289 */ "Call",
    /* 290 */ "EndCall",
};

/* Taken from SDL_iconv() */
char *SDL_UCS4ToUTF8(Uint32 ch, char *dst)
{
    Uint8 *p = (Uint8 *)dst;
    if (ch <= 0x7F) {
        *p = (Uint8)ch;
        ++dst;
    } else if (ch <= 0x7FF) {
        p[0] = 0xC0 | (Uint8)((ch >> 6) & 0x1F);
        p[1] = 0x80 | (Uint8)(ch & 0x3F);
        dst += 2;
    } else if (ch <= 0xFFFF) {
        p[0] = 0xE0 | (Uint8)((ch >> 12) & 0x0F);
        p[1] = 0x80 | (Uint8)((ch >> 6) & 0x3F);
        p[2] = 0x80 | (Uint8)(ch & 0x3F);
        dst += 3;
    } else {
        p[0] = 0xF0 | (Uint8)((ch >> 18) & 0x07);
        p[1] = 0x80 | (Uint8)((ch >> 12) & 0x3F);
        p[2] = 0x80 | (Uint8)((ch >> 6) & 0x3F);
        p[3] = 0x80 | (Uint8)(ch & 0x3F);
        dst += 4;
    }
    return dst;
}

/* Public functions */
int SDL_KeyboardInit(void)
{
    /* Set the default keymap */
    SDL_SetKeymap(0, SDL_default_keymap, SDL_NUM_SCANCODES, SDL_FALSE);
    return 0;
}

void SDL_ResetKeyboard(void)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;
    SDL_Scancode scancode;

#ifdef DEBUG_KEYBOARD
    printf("Resetting keyboard\n");
#endif
    for (scancode = (SDL_Scancode)0; scancode < SDL_NUM_SCANCODES; ++scancode) {
        if (keyboard->keystate[scancode] == SDL_PRESSED) {
            SDL_SendKeyboardKey(SDL_RELEASED, scancode);
        }
    }
}

void SDL_GetDefaultKeymap(SDL_Keycode *keymap)
{
    SDL_memcpy(keymap, SDL_default_keymap, sizeof(SDL_default_keymap));
}

void SDL_SetKeymap(int start, const SDL_Keycode *keys, int length, SDL_bool send_event)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;
    SDL_Scancode scancode;
    SDL_Keycode normalized_keymap[SDL_NUM_SCANCODES];
    SDL_bool is_azerty = SDL_FALSE;

    if (start < 0 || start + length > SDL_NUM_SCANCODES) {
        return;
    }

    if (start > 0) {
        SDL_memcpy(&normalized_keymap[0], &keyboard->keymap[0], sizeof(*keys) * start);
    }

    SDL_memcpy(&normalized_keymap[start], keys, sizeof(*keys) * length);

    if (start + length < SDL_NUM_SCANCODES) {
        int offset = start + length;
        SDL_memcpy(&normalized_keymap[offset], &keyboard->keymap[offset], sizeof(*keys) * (SDL_NUM_SCANCODES - offset));
    }

    /* On AZERTY layouts the number keys are technically symbols, but users (and games)
     * always think of them and view them in UI as number keys, so remap them here.
     */
    if (normalized_keymap[SDL_SCANCODE_0] < SDLK_0 || normalized_keymap[SDL_SCANCODE_0] > SDLK_9) {
        is_azerty = SDL_TRUE;
        for (scancode = SDL_SCANCODE_1; scancode <= SDL_SCANCODE_9; ++scancode) {
            if (normalized_keymap[scancode] >= SDLK_0 && normalized_keymap[scancode] <= SDLK_9) {
                /* There's a number on this row, it's not AZERTY */
                is_azerty = SDL_FALSE;
                break;
            }
        }
    }
    if (is_azerty) {
        normalized_keymap[SDL_SCANCODE_0] = SDLK_0;
        for (scancode = SDL_SCANCODE_1; scancode <= SDL_SCANCODE_9; ++scancode) {
            normalized_keymap[scancode] = SDLK_1 + (scancode - SDL_SCANCODE_1);
        }
    }

    /* If the mapping didn't really change, we're done here */
    if (!SDL_memcmp(&keyboard->keymap[start], &normalized_keymap[start], sizeof(*keys) * length)) {
        return;
    }

    SDL_memcpy(&keyboard->keymap[start], &normalized_keymap[start], sizeof(*keys) * length);

    if (send_event) {
        SDL_SendKeymapChangedEvent();
    }
}

void SDL_SetScancodeName(SDL_Scancode scancode, const char *name)
{
    if (scancode >= SDL_NUM_SCANCODES) {
        return;
    }
    SDL_scancode_names[scancode] = name;
}

SDL_Window *SDL_GetKeyboardFocus(void)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;

    return keyboard->focus;
}

void SDL_SetKeyboardFocus(SDL_Window *window)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;

    if (keyboard->focus && !window) {
        /* We won't get anymore keyboard messages, so reset keyboard state */
        SDL_ResetKeyboard();
    }

    /* See if the current window has lost focus */
    if (keyboard->focus && keyboard->focus != window) {

        /* new window shouldn't think it has mouse captured. */
        SDL_assert(window == NULL || !(window->flags & SDL_WINDOW_MOUSE_CAPTURE));

        /* old window must lose an existing mouse capture. */
        if (keyboard->focus->flags & SDL_WINDOW_MOUSE_CAPTURE) {
            SDL_CaptureMouse(SDL_FALSE); /* drop the capture. */
            SDL_UpdateMouseCapture(SDL_TRUE);
            SDL_assert(!(keyboard->focus->flags & SDL_WINDOW_MOUSE_CAPTURE));
        }

        SDL_SendWindowEvent(keyboard->focus, SDL_WINDOWEVENT_FOCUS_LOST,
                            0, 0);

        /* Ensures IME compositions are committed */
        if (SDL_EventState(SDL_TEXTINPUT, SDL_QUERY)) {
            SDL_VideoDevice *video = SDL_GetVideoDevice();
            if (video && video->StopTextInput) {
                video->StopTextInput(video);
            }
        }
    }

    keyboard->focus = window;

    if (keyboard->focus) {
        SDL_SendWindowEvent(keyboard->focus, SDL_WINDOWEVENT_FOCUS_GAINED,
                            0, 0);

        if (SDL_EventState(SDL_TEXTINPUT, SDL_QUERY)) {
            SDL_VideoDevice *video = SDL_GetVideoDevice();
            if (video && video->StartTextInput) {
                video->StartTextInput(video);
            }
        }
    }
}

static int SDL_SendKeyboardKeyInternal(Uint8 source, Uint8 state, SDL_Scancode scancode, SDL_Keycode keycode)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;
    int posted;
    SDL_Keymod modifier;
    Uint32 type;
    Uint8 repeat = SDL_FALSE;

    if (scancode == SDL_SCANCODE_UNKNOWN || scancode >= SDL_NUM_SCANCODES) {
        return 0;
    }

#ifdef DEBUG_KEYBOARD
    printf("The '%s' key has been %s\n", SDL_GetScancodeName(scancode),
           state == SDL_PRESSED ? "pressed" : "released");
#endif

    /* Figure out what type of event this is */
    switch (state) {
    case SDL_PRESSED:
        type = SDL_KEYDOWN;
        break;
    case SDL_RELEASED:
        type = SDL_KEYUP;
        break;
    default:
        /* Invalid state -- bail */
        return 0;
    }

    /* Drop events that don't change state */
    if (state) {
        if (keyboard->keystate[scancode]) {
            if (!(keyboard->keysource[scancode] & source)) {
                keyboard->keysource[scancode] |= source;
                return 0;
            }
            repeat = SDL_TRUE;
        }
        keyboard->keysource[scancode] |= source;
    } else {
        if (!keyboard->keystate[scancode]) {
            return 0;
        }
        keyboard->keysource[scancode] = 0;
    }

    /* Update internal keyboard state */
    keyboard->keystate[scancode] = state;

    if (keycode == SDLK_UNKNOWN) {
        keycode = keyboard->keymap[scancode];
    }

    if (source == KEYBOARD_HARDWARE) {
        keyboard->hardware_timestamp = SDL_GetTicks();
    } else if (source == KEYBOARD_AUTORELEASE) {
        keyboard->autorelease_pending = SDL_TRUE;
    }

    /* Update modifiers state if applicable */
    switch (keycode) {
    case SDLK_LCTRL:
        modifier = KMOD_LCTRL;
        break;
    case SDLK_RCTRL:
        modifier = KMOD_RCTRL;
        break;
    case SDLK_LSHIFT:
        modifier = KMOD_LSHIFT;
        break;
    case SDLK_RSHIFT:
        modifier = KMOD_RSHIFT;
        break;
    case SDLK_LALT:
        modifier = KMOD_LALT;
        break;
    case SDLK_RALT:
        modifier = KMOD_RALT;
        break;
    case SDLK_LGUI:
        modifier = KMOD_LGUI;
        break;
    case SDLK_RGUI:
        modifier = KMOD_RGUI;
        break;
    case SDLK_MODE:
        modifier = KMOD_MODE;
        break;
    default:
        modifier = KMOD_NONE;
        break;
    }
    if (SDL_KEYDOWN == type) {
        switch (keycode) {
        case SDLK_NUMLOCKCLEAR:
            keyboard->modstate ^= KMOD_NUM;
            break;
        case SDLK_CAPSLOCK:
            keyboard->modstate ^= KMOD_CAPS;
            break;
        case SDLK_SCROLLLOCK:
            keyboard->modstate ^= KMOD_SCROLL;
            break;
        default:
            keyboard->modstate |= modifier;
            break;
        }
    } else {
        keyboard->modstate &= ~modifier;
    }

    /* Post the event, if desired */
    posted = 0;
    if (SDL_GetEventState(type) == SDL_ENABLE) {
        SDL_Event event;
        event.key.type = type;
        event.key.state = state;
        event.key.repeat = repeat;
        event.key.keysym.scancode = scancode;
        event.key.keysym.sym = keycode;
        event.key.keysym.mod = keyboard->modstate;
        event.key.windowID = keyboard->focus ? keyboard->focus->id : 0;
        posted = (SDL_PushEvent(&event) > 0);
    }

    /* If the keyboard is grabbed and the grabbed window is in full-screen,
       minimize the window when we receive Alt+Tab, unless the application
       has explicitly opted out of this behavior. */
    if (keycode == SDLK_TAB &&
        state == SDL_PRESSED &&
        (keyboard->modstate & KMOD_ALT) &&
        keyboard->focus &&
        (keyboard->focus->flags & SDL_WINDOW_KEYBOARD_GRABBED) &&
        (keyboard->focus->flags & SDL_WINDOW_FULLSCREEN) &&
        SDL_GetHintBoolean(SDL_HINT_ALLOW_ALT_TAB_WHILE_GRABBED, SDL_TRUE)) {
        /* We will temporarily forfeit our grab by minimizing our window,
           allowing the user to escape the application */
        SDL_MinimizeWindow(keyboard->focus);
    }

    return posted;
}

int SDL_SendKeyboardUnicodeKey(Uint32 ch)
{
    SDL_Scancode code = SDL_SCANCODE_UNKNOWN;
    uint16_t mod = 0;

    if (ch < SDL_arraysize(SDL_ASCIIKeyInfoTable)) {
        code = SDL_ASCIIKeyInfoTable[ch].code;
        mod = SDL_ASCIIKeyInfoTable[ch].mod;
    }

    if (mod & KMOD_SHIFT) {
        /* If the character uses shift, press shift down */
        SDL_SendKeyboardKeyInternal(KEYBOARD_VIRTUAL, SDL_PRESSED, SDL_SCANCODE_LSHIFT, SDLK_UNKNOWN);
    }

    /* Send a keydown and keyup for the character */
    SDL_SendKeyboardKeyInternal(KEYBOARD_VIRTUAL, SDL_PRESSED, code, SDLK_UNKNOWN);
    SDL_SendKeyboardKeyInternal(KEYBOARD_VIRTUAL, SDL_RELEASED, code, SDLK_UNKNOWN);

    if (mod & KMOD_SHIFT) {
        /* If the character uses shift, release shift */
        SDL_SendKeyboardKeyInternal(KEYBOARD_VIRTUAL, SDL_RELEASED, SDL_SCANCODE_LSHIFT, SDLK_UNKNOWN);
    }
    return 0;
}

int SDL_SendVirtualKeyboardKey(Uint8 state, SDL_Scancode scancode)
{
    return SDL_SendKeyboardKeyInternal(KEYBOARD_VIRTUAL, state, scancode, SDLK_UNKNOWN);
}

int SDL_SendKeyboardKey(Uint8 state, SDL_Scancode scancode)
{
    return SDL_SendKeyboardKeyInternal(KEYBOARD_HARDWARE, state, scancode, SDLK_UNKNOWN);
}

int SDL_SendKeyboardKeyAndKeycode(Uint8 state, SDL_Scancode scancode, SDL_Keycode keycode)
{
    return SDL_SendKeyboardKeyInternal(KEYBOARD_HARDWARE, state, scancode, keycode);
}

int SDL_SendKeyboardKeyAutoRelease(SDL_Scancode scancode)
{
    return SDL_SendKeyboardKeyInternal(KEYBOARD_AUTORELEASE, SDL_PRESSED, scancode, SDLK_UNKNOWN);
}

void SDL_ReleaseAutoReleaseKeys(void)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;
    SDL_Scancode scancode;

    if (keyboard->autorelease_pending) {
        for (scancode = SDL_SCANCODE_UNKNOWN; scancode < SDL_NUM_SCANCODES; ++scancode) {
            if (keyboard->keysource[scancode] == KEYBOARD_AUTORELEASE) {
                SDL_SendKeyboardKeyInternal(KEYBOARD_AUTORELEASE, SDL_RELEASED, scancode, SDLK_UNKNOWN);
            }
        }
        keyboard->autorelease_pending = SDL_FALSE;
    }

    if (keyboard->hardware_timestamp) {
        /* Keep hardware keyboard "active" for 250 ms */
        if (SDL_TICKS_PASSED(SDL_GetTicks(), keyboard->hardware_timestamp + 250)) {
            keyboard->hardware_timestamp = 0;
        }
    }
}

SDL_bool SDL_HardwareKeyboardKeyPressed(void)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;
    SDL_Scancode scancode;

    for (scancode = SDL_SCANCODE_UNKNOWN; scancode < SDL_NUM_SCANCODES; ++scancode) {
        if (keyboard->keysource[scancode] & KEYBOARD_HARDWARE) {
            return SDL_TRUE;
        }
    }

    return keyboard->hardware_timestamp ? SDL_TRUE : SDL_FALSE;
}

int SDL_SendKeyboardText(const char *text)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;
    int posted;

    /* Don't post text events for unprintable characters */
    if ((unsigned char)*text < ' ' || *text == 127) {
        return 0;
    }

    /* Post the event, if desired */
    posted = 0;
    if (SDL_GetEventState(SDL_TEXTINPUT) == SDL_ENABLE) {
        SDL_Event event;
        size_t pos = 0, advance, length = SDL_strlen(text);

        event.text.type = SDL_TEXTINPUT;
        event.text.windowID = keyboard->focus ? keyboard->focus->id : 0;
        while (pos < length) {
            advance = SDL_utf8strlcpy(event.text.text, text + pos, SDL_arraysize(event.text.text));
            if (!advance) {
                break;
            }
            pos += advance;
            posted |= (SDL_PushEvent(&event) > 0);
        }
    }
    return posted;
}

int SDL_SendEditingText(const char *text, int start, int length)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;
    int posted;

    /* Post the event, if desired */
    posted = 0;
    if (SDL_GetEventState(SDL_TEXTEDITING) == SDL_ENABLE) {
        SDL_Event event;

        if (SDL_GetHintBoolean(SDL_HINT_IME_SUPPORT_EXTENDED_TEXT, SDL_FALSE) &&
            SDL_strlen(text) >= SDL_arraysize(event.text.text)) {
            event.editExt.type = SDL_TEXTEDITING_EXT;
            event.editExt.windowID = keyboard->focus ? keyboard->focus->id : 0;
            event.editExt.text = text ? SDL_strdup(text) : NULL;
            event.editExt.start = start;
            event.editExt.length = length;
        } else {
            event.edit.type = SDL_TEXTEDITING;
            event.edit.windowID = keyboard->focus ? keyboard->focus->id : 0;
            event.edit.start = start;
            event.edit.length = length;
            SDL_utf8strlcpy(event.edit.text, text, SDL_arraysize(event.edit.text));
        }

        posted = (SDL_PushEvent(&event) > 0);
    }
    return posted;
}

void SDL_KeyboardQuit(void)
{
}

const Uint8 *SDL_GetKeyboardState(int *numkeys)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;

    if (numkeys != (int *)0) {
        *numkeys = SDL_NUM_SCANCODES;
    }
    return keyboard->keystate;
}

SDL_Keymod SDL_GetModState(void)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;

    return (SDL_Keymod)keyboard->modstate;
}

void SDL_SetModState(SDL_Keymod modstate)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;

    keyboard->modstate = modstate;
}

/* Note that SDL_ToggleModState() is not a public API. SDL_SetModState() is. */
void SDL_ToggleModState(const SDL_Keymod modstate, const SDL_bool toggle)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;
    if (toggle) {
        keyboard->modstate |= modstate;
    } else {
        keyboard->modstate &= ~modstate;
    }
}

SDL_Keycode SDL_GetKeyFromScancode(SDL_Scancode scancode)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;

    if (((int)scancode) < SDL_SCANCODE_UNKNOWN || scancode >= SDL_NUM_SCANCODES) {
        SDL_InvalidParamError("scancode");
        return 0;
    }

    return keyboard->keymap[scancode];
}

SDL_Keycode SDL_GetDefaultKeyFromScancode(SDL_Scancode scancode)
{
    if (((int)scancode) < SDL_SCANCODE_UNKNOWN || scancode >= SDL_NUM_SCANCODES) {
        SDL_InvalidParamError("scancode");
        return 0;
    }

    return SDL_default_keymap[scancode];
}

SDL_Scancode SDL_GetScancodeFromKey(SDL_Keycode key)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;
    SDL_Scancode scancode;

    for (scancode = SDL_SCANCODE_UNKNOWN; scancode < SDL_NUM_SCANCODES;
         ++scancode) {
        if (keyboard->keymap[scancode] == key) {
            return scancode;
        }
    }
    return SDL_SCANCODE_UNKNOWN;
}

const char *SDL_GetScancodeName(SDL_Scancode scancode)
{
    const char *name;
    if (((int)scancode) < SDL_SCANCODE_UNKNOWN || scancode >= SDL_NUM_SCANCODES) {
        SDL_InvalidParamError("scancode");
        return "";
    }

    name = SDL_scancode_names[scancode];
    if (name) {
        return name;
    }

    return "";
}

SDL_Scancode SDL_GetScancodeFromName(const char *name)
{
    int i;

    if (!name || !*name) {
        SDL_InvalidParamError("name");
        return SDL_SCANCODE_UNKNOWN;
    }

    for (i = 0; i < SDL_arraysize(SDL_scancode_names); ++i) {
        if (!SDL_scancode_names[i]) {
            continue;
        }
        if (SDL_strcasecmp(name, SDL_scancode_names[i]) == 0) {
            return (SDL_Scancode)i;
        }
    }

    SDL_InvalidParamError("name");
    return SDL_SCANCODE_UNKNOWN;
}

const char *SDL_GetKeyName(SDL_Keycode key)
{
    static char name[8];
    char *end;

    if (key & SDLK_SCANCODE_MASK) {
        return SDL_GetScancodeName((SDL_Scancode)(key & ~SDLK_SCANCODE_MASK));
    }

    switch (key) {
    case SDLK_RETURN:
        return SDL_GetScancodeName(SDL_SCANCODE_RETURN);
    case SDLK_ESCAPE:
        return SDL_GetScancodeName(SDL_SCANCODE_ESCAPE);
    case SDLK_BACKSPACE:
        return SDL_GetScancodeName(SDL_SCANCODE_BACKSPACE);
    case SDLK_TAB:
        return SDL_GetScancodeName(SDL_SCANCODE_TAB);
    case SDLK_SPACE:
        return SDL_GetScancodeName(SDL_SCANCODE_SPACE);
    case SDLK_DELETE:
        return SDL_GetScancodeName(SDL_SCANCODE_DELETE);
    default:
        /* Unaccented letter keys on latin keyboards are normally
           labeled in upper case (and probably on others like Greek or
           Cyrillic too, so if you happen to know for sure, please
           adapt this). */
        if (key >= 'a' && key <= 'z') {
            key -= 32;
        }

        end = SDL_UCS4ToUTF8((Uint32)key, name);
        *end = '\0';
        return name;
    }
}

SDL_Keycode SDL_GetKeyFromName(const char *name)
{
    SDL_Keycode key;

    /* Check input */
    if (!name) {
        return SDLK_UNKNOWN;
    }

    /* If it's a single UTF-8 character, then that's the keycode itself */
    key = *(const unsigned char *)name;
    if (key >= 0xF0) {
        if (SDL_strlen(name) == 4) {
            int i = 0;
            key = (Uint16)(name[i] & 0x07) << 18;
            key |= (Uint16)(name[++i] & 0x3F) << 12;
            key |= (Uint16)(name[++i] & 0x3F) << 6;
            key |= (Uint16)(name[++i] & 0x3F);
            return key;
        }
        return SDLK_UNKNOWN;
    } else if (key >= 0xE0) {
        if (SDL_strlen(name) == 3) {
            int i = 0;
            key = (Uint16)(name[i] & 0x0F) << 12;
            key |= (Uint16)(name[++i] & 0x3F) << 6;
            key |= (Uint16)(name[++i] & 0x3F);
            return key;
        }
        return SDLK_UNKNOWN;
    } else if (key >= 0xC0) {
        if (SDL_strlen(name) == 2) {
            int i = 0;
            key = (Uint16)(name[i] & 0x1F) << 6;
            key |= (Uint16)(name[++i] & 0x3F);
            return key;
        }
        return SDLK_UNKNOWN;
    } else {
        if (SDL_strlen(name) == 1) {
            if (key >= 'A' && key <= 'Z') {
                key += 32;
            }
            return key;
        }

        /* Get the scancode for this name, and the associated keycode */
        return SDL_default_keymap[SDL_GetScancodeFromName(name)];
    }
}

/* vi: set ts=4 sw=4 expandtab: */
