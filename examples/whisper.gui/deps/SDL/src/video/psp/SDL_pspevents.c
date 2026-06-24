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

#ifdef SDL_VIDEO_DRIVER_PSP

/* Being a null driver, there's no event stream. We just define stubs for
   most of the API. */

#include "SDL.h"
#include "../../events/SDL_events_c.h"
#include "../../events/SDL_keyboard_c.h"
#include "../SDL_sysvideo.h"
#include "SDL_pspvideo.h"
#include "SDL_pspevents_c.h"
#include "SDL_keyboard.h"
#include "../../thread/SDL_systhread.h"
#include <psphprm.h>
#include <pspthreadman.h>

#ifdef PSPIRKEYB
#include <pspirkeyb.h>
#include <pspirkeyb_rawkeys.h>

#define IRKBD_CONFIG_FILE NULL /* this will take ms0:/seplugins/pspirkeyb.ini */

static int irkbd_ready = 0;
static SDL_Keycode keymap[256];
#endif

static enum PspHprmKeys hprm = 0;
static SDL_sem *event_sem = NULL;
static SDL_Thread *thread = NULL;
static int running = 0;
static struct
{
    enum PspHprmKeys id;
    SDL_Keycode sym;
} keymap_psp[] = {
    { PSP_HPRM_PLAYPAUSE, SDLK_F10 },
    { PSP_HPRM_FORWARD, SDLK_F11 },
    { PSP_HPRM_BACK, SDLK_F12 },
    { PSP_HPRM_VOL_UP, SDLK_F13 },
    { PSP_HPRM_VOL_DOWN, SDLK_F14 },
    { PSP_HPRM_HOLD, SDLK_F15 }
};

int EventUpdate(void *data)
{
    while (running) {
        SDL_SemWait(event_sem);
        sceHprmPeekCurrentKey((u32 *)&hprm);
        SDL_SemPost(event_sem);
        /* Delay 1/60th of a second */
        sceKernelDelayThread(1000000 / 60);
    }
    return 0;
}

void PSP_PumpEvents(_THIS)
{
    int i;
    enum PspHprmKeys keys;
    enum PspHprmKeys changed;
    static enum PspHprmKeys old_keys = 0;

    SDL_SemWait(event_sem);
    keys = hprm;
    SDL_SemPost(event_sem);

    /* HPRM Keyboard */
    changed = old_keys ^ keys;
    old_keys = keys;
    if (changed) {
        for (i = 0; i < sizeof(keymap_psp) / sizeof(keymap_psp[0]); i++) {
            if (changed & keymap_psp[i].id) {
                SDL_SendKeyboardKey((keys & keymap_psp[i].id) ? SDL_PRESSED : SDL_RELEASED, SDL_GetScancodeFromKey(keymap_psp[i].sym));
            }
        }
    }

#ifdef PSPIRKEYB
    if (irkbd_ready) {
        unsigned char buffer[255];
        int i, length, count;
        SIrKeybScanCodeData *scanData;

        if (pspIrKeybReadinput(buffer, &length) >= 0) {
            if ((length % sizeof(SIrKeybScanCodeData)) == 0) {
                count = length / sizeof(SIrKeybScanCodeData);
                for (i = 0; i < count; i++) {
                    unsigned char raw, pressed;
                    scanData = (SIrKeybScanCodeData *)buffer + i;
                    raw = scanData->raw;
                    pressed = scanData->pressed;
                    sym.scancode = raw;
                    sym.sym = keymap[raw];
                    /* not tested */
                    /* SDL_PrivateKeyboard(pressed?SDL_PRESSED:SDL_RELEASED, &sym); */
                    SDL_SendKeyboardKey((keys & keymap_psp[i].id) ? SDL_PRESSED : SDL_RELEASED, SDL_GetScancodeFromKey(keymap[raw]));
                }
            }
        }
    }
#endif
    sceKernelDelayThread(0);

    return;
}

void PSP_InitOSKeymap(_THIS)
{
#ifdef PSPIRKEYB
    int i;
    for (i = 0; i < SDL_TABLESIZE(keymap); ++i) {
        keymap[i] = SDLK_UNKNOWN;
    }

    keymap[KEY_ESC] = SDLK_ESCAPE;

    keymap[KEY_F1] = SDLK_F1;
    keymap[KEY_F2] = SDLK_F2;
    keymap[KEY_F3] = SDLK_F3;
    keymap[KEY_F4] = SDLK_F4;
    keymap[KEY_F5] = SDLK_F5;
    keymap[KEY_F6] = SDLK_F6;
    keymap[KEY_F7] = SDLK_F7;
    keymap[KEY_F8] = SDLK_F8;
    keymap[KEY_F9] = SDLK_F9;
    keymap[KEY_F10] = SDLK_F10;
    keymap[KEY_F11] = SDLK_F11;
    keymap[KEY_F12] = SDLK_F12;
    keymap[KEY_F13] = SDLK_PRINT;
    keymap[KEY_F14] = SDLK_PAUSE;

    keymap[KEY_GRAVE] = SDLK_BACKQUOTE;
    keymap[KEY_1] = SDLK_1;
    keymap[KEY_2] = SDLK_2;
    keymap[KEY_3] = SDLK_3;
    keymap[KEY_4] = SDLK_4;
    keymap[KEY_5] = SDLK_5;
    keymap[KEY_6] = SDLK_6;
    keymap[KEY_7] = SDLK_7;
    keymap[KEY_8] = SDLK_8;
    keymap[KEY_9] = SDLK_9;
    keymap[KEY_0] = SDLK_0;
    keymap[KEY_MINUS] = SDLK_MINUS;
    keymap[KEY_EQUAL] = SDLK_EQUALS;
    keymap[KEY_BACKSPACE] = SDLK_BACKSPACE;

    keymap[KEY_TAB] = SDLK_TAB;
    keymap[KEY_Q] = SDLK_q;
    keymap[KEY_W] = SDLK_w;
    keymap[KEY_E] = SDLK_e;
    keymap[KEY_R] = SDLK_r;
    keymap[KEY_T] = SDLK_t;
    keymap[KEY_Y] = SDLK_y;
    keymap[KEY_U] = SDLK_u;
    keymap[KEY_I] = SDLK_i;
    keymap[KEY_O] = SDLK_o;
    keymap[KEY_P] = SDLK_p;
    keymap[KEY_LEFTBRACE] = SDLK_LEFTBRACKET;
    keymap[KEY_RIGHTBRACE] = SDLK_RIGHTBRACKET;
    keymap[KEY_ENTER] = SDLK_RETURN;

    keymap[KEY_CAPSLOCK] = SDLK_CAPSLOCK;
    keymap[KEY_A] = SDLK_a;
    keymap[KEY_S] = SDLK_s;
    keymap[KEY_D] = SDLK_d;
    keymap[KEY_F] = SDLK_f;
    keymap[KEY_G] = SDLK_g;
    keymap[KEY_H] = SDLK_h;
    keymap[KEY_J] = SDLK_j;
    keymap[KEY_K] = SDLK_k;
    keymap[KEY_L] = SDLK_l;
    keymap[KEY_SEMICOLON] = SDLK_SEMICOLON;
    keymap[KEY_APOSTROPHE] = SDLK_QUOTE;
    keymap[KEY_BACKSLASH] = SDLK_BACKSLASH;

    keymap[KEY_Z] = SDLK_z;
    keymap[KEY_X] = SDLK_x;
    keymap[KEY_C] = SDLK_c;
    keymap[KEY_V] = SDLK_v;
    keymap[KEY_B] = SDLK_b;
    keymap[KEY_N] = SDLK_n;
    keymap[KEY_M] = SDLK_m;
    keymap[KEY_COMMA] = SDLK_COMMA;
    keymap[KEY_DOT] = SDLK_PERIOD;
    keymap[KEY_SLASH] = SDLK_SLASH;

    keymap[KEY_SPACE] = SDLK_SPACE;

    keymap[KEY_UP] = SDLK_UP;
    keymap[KEY_DOWN] = SDLK_DOWN;
    keymap[KEY_LEFT] = SDLK_LEFT;
    keymap[KEY_RIGHT] = SDLK_RIGHT;

    keymap[KEY_HOME] = SDLK_HOME;
    keymap[KEY_END] = SDLK_END;
    keymap[KEY_INSERT] = SDLK_INSERT;
    keymap[KEY_DELETE] = SDLK_DELETE;

    keymap[KEY_NUMLOCK] = SDLK_NUMLOCK;
    keymap[KEY_LEFTMETA] = SDLK_LSUPER;

    keymap[KEY_KPSLASH] = SDLK_KP_DIVIDE;
    keymap[KEY_KPASTERISK] = SDLK_KP_MULTIPLY;
    keymap[KEY_KPMINUS] = SDLK_KP_MINUS;
    keymap[KEY_KPPLUS] = SDLK_KP_PLUS;
    keymap[KEY_KPDOT] = SDLK_KP_PERIOD;
    keymap[KEY_KPEQUAL] = SDLK_KP_EQUALS;

    keymap[KEY_LEFTCTRL] = SDLK_LCTRL;
    keymap[KEY_RIGHTCTRL] = SDLK_RCTRL;
    keymap[KEY_LEFTALT] = SDLK_LALT;
    keymap[KEY_RIGHTALT] = SDLK_RALT;
    keymap[KEY_LEFTSHIFT] = SDLK_LSHIFT;
    keymap[KEY_RIGHTSHIFT] = SDLK_RSHIFT;
#endif
}

int PSP_EventInit(_THIS)
{
#ifdef PSPIRKEYB
    int outputmode = PSP_IRKBD_OUTPUT_MODE_SCANCODE;
    int ret = pspIrKeybInit(IRKBD_CONFIG_FILE, 0);
    if (ret == PSP_IRKBD_RESULT_OK) {
        pspIrKeybOutputMode(outputmode);
        irkbd_ready = 1;
    } else {
        irkbd_ready = 0;
    }
#endif
    /* Start thread to read data */
    if ((event_sem = SDL_CreateSemaphore(1)) == NULL) {
        return SDL_SetError("Can't create input semaphore");
    }
    running = 1;
    if ((thread = SDL_CreateThreadInternal(EventUpdate, "PSPInputThread", 4096, NULL)) == NULL) {
        return SDL_SetError("Can't create input thread");
    }
    return 0;
}

void PSP_EventQuit(_THIS)
{
    running = 0;
    SDL_WaitThread(thread, NULL);
    SDL_DestroySemaphore(event_sem);
#ifdef PSPIRKEYB
    if (irkbd_ready) {
        pspIrKeybFinish();
        irkbd_ready = 0;
    }
#endif
}

/* end of SDL_pspevents.c ... */

#endif /* SDL_VIDEO_DRIVER_PSP */

/* vi: set ts=4 sw=4 expandtab: */
