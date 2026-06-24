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

#include "SDL_ime.h"
#include "SDL_ibus.h"
#include "SDL_fcitx.h"

typedef SDL_bool (*_SDL_IME_Init)(void);
typedef void (*_SDL_IME_Quit)(void);
typedef void (*_SDL_IME_SetFocus)(SDL_bool);
typedef void (*_SDL_IME_Reset)(void);
typedef SDL_bool (*_SDL_IME_ProcessKeyEvent)(Uint32, Uint32, Uint8 state);
typedef void (*_SDL_IME_UpdateTextRect)(const SDL_Rect *);
typedef void (*_SDL_IME_PumpEvents)(void);

static _SDL_IME_Init SDL_IME_Init_Real = NULL;
static _SDL_IME_Quit SDL_IME_Quit_Real = NULL;
static _SDL_IME_SetFocus SDL_IME_SetFocus_Real = NULL;
static _SDL_IME_Reset SDL_IME_Reset_Real = NULL;
static _SDL_IME_ProcessKeyEvent SDL_IME_ProcessKeyEvent_Real = NULL;
static _SDL_IME_UpdateTextRect SDL_IME_UpdateTextRect_Real = NULL;
static _SDL_IME_PumpEvents SDL_IME_PumpEvents_Real = NULL;

static void InitIME(void)
{
    static SDL_bool inited = SDL_FALSE;
#ifdef HAVE_FCITX
    const char *im_module = SDL_getenv("SDL_IM_MODULE");
    const char *xmodifiers = SDL_getenv("XMODIFIERS");
#endif

    if (inited == SDL_TRUE) {
        return;
    }

    inited = SDL_TRUE;

    /* See if fcitx IME support is being requested */
#ifdef HAVE_FCITX
    if (!SDL_IME_Init_Real &&
        ((im_module && SDL_strcmp(im_module, "fcitx") == 0) ||
         (!im_module && xmodifiers && SDL_strstr(xmodifiers, "@im=fcitx") != NULL))) {
        SDL_IME_Init_Real = SDL_Fcitx_Init;
        SDL_IME_Quit_Real = SDL_Fcitx_Quit;
        SDL_IME_SetFocus_Real = SDL_Fcitx_SetFocus;
        SDL_IME_Reset_Real = SDL_Fcitx_Reset;
        SDL_IME_ProcessKeyEvent_Real = SDL_Fcitx_ProcessKeyEvent;
        SDL_IME_UpdateTextRect_Real = SDL_Fcitx_UpdateTextRect;
        SDL_IME_PumpEvents_Real = SDL_Fcitx_PumpEvents;
    }
#endif /* HAVE_FCITX */

    /* default to IBus */
#ifdef HAVE_IBUS_IBUS_H
    if (!SDL_IME_Init_Real) {
        SDL_IME_Init_Real = SDL_IBus_Init;
        SDL_IME_Quit_Real = SDL_IBus_Quit;
        SDL_IME_SetFocus_Real = SDL_IBus_SetFocus;
        SDL_IME_Reset_Real = SDL_IBus_Reset;
        SDL_IME_ProcessKeyEvent_Real = SDL_IBus_ProcessKeyEvent;
        SDL_IME_UpdateTextRect_Real = SDL_IBus_UpdateTextRect;
        SDL_IME_PumpEvents_Real = SDL_IBus_PumpEvents;
    }
#endif /* HAVE_IBUS_IBUS_H */
}

SDL_bool SDL_IME_Init(void)
{
    InitIME();

    if (SDL_IME_Init_Real) {
        if (SDL_IME_Init_Real()) {
            return SDL_TRUE;
        }

        /* uhoh, the IME implementation's init failed! Disable IME support. */
        SDL_IME_Init_Real = NULL;
        SDL_IME_Quit_Real = NULL;
        SDL_IME_SetFocus_Real = NULL;
        SDL_IME_Reset_Real = NULL;
        SDL_IME_ProcessKeyEvent_Real = NULL;
        SDL_IME_UpdateTextRect_Real = NULL;
        SDL_IME_PumpEvents_Real = NULL;
    }

    return SDL_FALSE;
}

void SDL_IME_Quit(void)
{
    if (SDL_IME_Quit_Real) {
        SDL_IME_Quit_Real();
    }
}

void SDL_IME_SetFocus(SDL_bool focused)
{
    if (SDL_IME_SetFocus_Real) {
        SDL_IME_SetFocus_Real(focused);
    }
}

void SDL_IME_Reset(void)
{
    if (SDL_IME_Reset_Real) {
        SDL_IME_Reset_Real();
    }
}

SDL_bool SDL_IME_ProcessKeyEvent(Uint32 keysym, Uint32 keycode, Uint8 state)
{
    if (SDL_IME_ProcessKeyEvent_Real) {
        return SDL_IME_ProcessKeyEvent_Real(keysym, keycode, state);
    }

    return SDL_FALSE;
}

void SDL_IME_UpdateTextRect(const SDL_Rect *rect)
{
    if (SDL_IME_UpdateTextRect_Real) {
        SDL_IME_UpdateTextRect_Real(rect);
    }
}

void SDL_IME_PumpEvents(void)
{
    if (SDL_IME_PumpEvents_Real) {
        SDL_IME_PumpEvents_Real();
    }
}

/* vi: set ts=4 sw=4 expandtab: */
