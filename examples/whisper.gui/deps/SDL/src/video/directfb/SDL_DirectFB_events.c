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

#ifdef SDL_VIDEO_DRIVER_DIRECTFB

/* Handle the event stream, converting DirectFB input events into SDL events */

#include "SDL_DirectFB_video.h"
#include "SDL_DirectFB_window.h"
#include "SDL_DirectFB_modes.h"

#include "SDL_syswm.h"

#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_keyboard_c.h"
#include "../../events/SDL_windowevents_c.h"
#include "../../events/SDL_events_c.h"
#include "../../events/SDL_scancode_tables_c.h"

#include "SDL_DirectFB_events.h"

#if USE_MULTI_API
#define SDL_SendMouseMotion_ex(w, id, relative, x, y, p) SDL_SendMouseMotion(w, id, relative, x, y, p)
#define SDL_SendMouseButton_ex(w, id, state, button) SDL_SendMouseButton(w, id, state, button)
#define SDL_SendKeyboardKey_ex(id, state, scancode) SDL_SendKeyboardKey(id, state, scancode)
#define SDL_SendKeyboardText_ex(id, text) SDL_SendKeyboardText(id, text)
#else
#define SDL_SendMouseMotion_ex(w, id, relative, x, y, p) SDL_SendMouseMotion(w, id, relative, x, y)
#define SDL_SendMouseButton_ex(w, id, state, button) SDL_SendMouseButton(w, id, state, button)
#define SDL_SendKeyboardKey_ex(id, state, scancode) SDL_SendKeyboardKey(state, scancode)
#define SDL_SendKeyboardText_ex(id, text) SDL_SendKeyboardText(text)
#endif

typedef struct _cb_data cb_data;
struct _cb_data
{
    DFB_DeviceData *devdata;
    int sys_ids;
    int sys_kbd;
};

/* The translation tables from a DirectFB keycode to a SDL keysym */
static SDL_Scancode oskeymap[256];


static SDL_Keysym *DirectFB_TranslateKey(_THIS, DFBWindowEvent * evt,
                                         SDL_Keysym * keysym, Uint32 *unicode);
static SDL_Keysym *DirectFB_TranslateKeyInputEvent(_THIS, DFBInputEvent * evt,
                                                   SDL_Keysym * keysym, Uint32 *unicode);

static void DirectFB_InitOSKeymap(_THIS, SDL_Scancode * keypmap, int numkeys);
static int DirectFB_TranslateButton(DFBInputDeviceButtonIdentifier button);

static void UnicodeToUtf8( Uint16 w , char *utf8buf)
{
        unsigned char *utf8s = (unsigned char *) utf8buf;

    if ( w < 0x0080 ) {
        utf8s[0] = ( unsigned char ) w;
        utf8s[1] = 0;
    }
    else if ( w < 0x0800 ) {
        utf8s[0] = 0xc0 | (( w ) >> 6 );
        utf8s[1] = 0x80 | (( w ) & 0x3f );
        utf8s[2] = 0;
    }
    else {
        utf8s[0] = 0xe0 | (( w ) >> 12 );
        utf8s[1] = 0x80 | (( ( w ) >> 6 ) & 0x3f );
        utf8s[2] = 0x80 | (( w ) & 0x3f );
        utf8s[3] = 0;
    }
}

static void FocusAllMice(_THIS, SDL_Window *window)
{
#if USE_MULTI_API
    SDL_DFB_DEVICEDATA(_this);
    int index;

    for (index = 0; index < devdata->num_mice; index++)
        SDL_SetMouseFocus(devdata->mouse_id[index], id);
#else
    SDL_SetMouseFocus(window);
#endif
}


static void FocusAllKeyboards(_THIS, SDL_Window *window)
{
#if USE_MULTI_API
    SDL_DFB_DEVICEDATA(_this);
    int index;

    for (index = 0; index < devdata->num_keyboard; index++)
        SDL_SetKeyboardFocus(index, id);
#else
    SDL_SetKeyboardFocus(window);
#endif
}

static void MotionAllMice(_THIS, int x, int y)
{
#if USE_MULTI_API
    SDL_DFB_DEVICEDATA(_this);
    int index;

    for (index = 0; index < devdata->num_mice; index++) {
        SDL_Mouse *mouse = SDL_GetMouse(index);
        mouse->x = mouse->last_x = x;
        mouse->y = mouse->last_y = y;
        /* SDL_SendMouseMotion(devdata->mouse_id[index], 0, x, y, 0); */
    }
#endif
}

static int KbdIndex(_THIS, int id)
{
    SDL_DFB_DEVICEDATA(_this);
    int index;

    for (index = 0; index < devdata->num_keyboard; index++) {
        if (devdata->keyboard[index].id == id)
            return index;
    }
    return -1;
}

static int ClientXY(DFB_WindowData * p, int *x, int *y)
{
    int cx, cy;

    cx = *x;
    cy = *y;

    cx -= p->client.x;
    cy -= p->client.y;

    if (cx < 0 || cy < 0)
        return 0;
    if (cx >= p->client.w || cy >= p->client.h)
        return 0;
    *x = cx;
    *y = cy;
    return 1;
}

static void ProcessWindowEvent(_THIS, SDL_Window *sdlwin, DFBWindowEvent * evt)
{
    SDL_DFB_DEVICEDATA(_this);
    SDL_DFB_WINDOWDATA(sdlwin);
    SDL_Keysym keysym;
    Uint32 unicode;
    char text[SDL_TEXTINPUTEVENT_TEXT_SIZE];

    if (evt->clazz == DFEC_WINDOW) {
        switch (evt->type) {
        case DWET_BUTTONDOWN:
            if (ClientXY(windata, &evt->x, &evt->y)) {
                if (!devdata->use_linux_input) {
                    SDL_SendMouseMotion_ex(sdlwin, devdata->mouse_id[0], 0, evt->x,
                                        evt->y, 0);
                    SDL_SendMouseButton_ex(sdlwin, devdata->mouse_id[0],
                                        SDL_PRESSED,
                                        DirectFB_TranslateButton
                                        (evt->button));
                } else {
                    MotionAllMice(_this, evt->x, evt->y);
                }
            }
            break;
        case DWET_BUTTONUP:
            if (ClientXY(windata, &evt->x, &evt->y)) {
                if (!devdata->use_linux_input) {
                    SDL_SendMouseMotion_ex(sdlwin, devdata->mouse_id[0], 0, evt->x,
                                        evt->y, 0);
                    SDL_SendMouseButton_ex(sdlwin, devdata->mouse_id[0],
                                        SDL_RELEASED,
                                        DirectFB_TranslateButton
                                        (evt->button));
                } else {
                    MotionAllMice(_this, evt->x, evt->y);
                }
            }
            break;
        case DWET_MOTION:
            if (ClientXY(windata, &evt->x, &evt->y)) {
                if (!devdata->use_linux_input) {
                    if (!(sdlwin->flags & SDL_WINDOW_MOUSE_GRABBED))
                        SDL_SendMouseMotion_ex(sdlwin, devdata->mouse_id[0], 0,
                                            evt->x, evt->y, 0);
                } else {
                    /* relative movements are not exact!
                     * This code should limit the number of events sent.
                     * However it kills MAME axis recognition ... */
                    static int cnt = 0;
                    if (1 && ++cnt > 20) {
                        MotionAllMice(_this, evt->x, evt->y);
                        cnt = 0;
                    }
                }
                if (!(sdlwin->flags & SDL_WINDOW_MOUSE_FOCUS))
                    SDL_SendWindowEvent(sdlwin, SDL_WINDOWEVENT_ENTER, 0,
                                        0);
            }
            break;
        case DWET_KEYDOWN:
            if (!devdata->use_linux_input) {
                DirectFB_TranslateKey(_this, evt, &keysym, &unicode);
                /* printf("Scancode %d  %d %d\n", keysym.scancode, evt->key_code, evt->key_id); */
                SDL_SendKeyboardKey_ex(0, SDL_PRESSED, keysym.scancode);
                if (SDL_EventState(SDL_TEXTINPUT, SDL_QUERY)) {
                    SDL_zeroa(text);
                    UnicodeToUtf8(unicode, text);
                    if (*text) {
                        SDL_SendKeyboardText_ex(0, text);
                    }
                }
            }
            break;
        case DWET_KEYUP:
            if (!devdata->use_linux_input) {
                DirectFB_TranslateKey(_this, evt, &keysym, &unicode);
                SDL_SendKeyboardKey_ex(0, SDL_RELEASED, keysym.scancode);
            }
            break;
        case DWET_POSITION:
            if (ClientXY(windata, &evt->x, &evt->y)) {
                SDL_SendWindowEvent(sdlwin, SDL_WINDOWEVENT_MOVED,
                                    evt->x, evt->y);
            }
            break;
        case DWET_POSITION_SIZE:
            if (ClientXY(windata, &evt->x, &evt->y)) {
                SDL_SendWindowEvent(sdlwin, SDL_WINDOWEVENT_MOVED,
                                    evt->x, evt->y);
            }
            SDL_FALLTHROUGH;
        case DWET_SIZE:
            /* FIXME: what about < 0 */
            evt->w -= (windata->theme.right_size + windata->theme.left_size);
            evt->h -=
                (windata->theme.top_size + windata->theme.bottom_size +
                 windata->theme.caption_size);
            SDL_SendWindowEvent(sdlwin, SDL_WINDOWEVENT_RESIZED,
                                evt->w, evt->h);
            break;
        case DWET_CLOSE:
            SDL_SendWindowEvent(sdlwin, SDL_WINDOWEVENT_CLOSE, 0, 0);
            break;
        case DWET_GOTFOCUS:
            DirectFB_SetContext(_this, sdlwin);
            FocusAllKeyboards(_this, sdlwin);
            SDL_SendWindowEvent(sdlwin, SDL_WINDOWEVENT_FOCUS_GAINED,
                                0, 0);
            break;
        case DWET_LOSTFOCUS:
            SDL_SendWindowEvent(sdlwin, SDL_WINDOWEVENT_FOCUS_LOST, 0, 0);
            FocusAllKeyboards(_this, 0);
            break;
        case DWET_ENTER:
            /* SDL_DirectFB_ReshowCursor(_this, 0); */
            FocusAllMice(_this, sdlwin);
            /* FIXME: when do we really enter ? */
            if (ClientXY(windata, &evt->x, &evt->y))
                MotionAllMice(_this, evt->x, evt->y);
            SDL_SendWindowEvent(sdlwin, SDL_WINDOWEVENT_ENTER, 0, 0);
            break;
        case DWET_LEAVE:
            SDL_SendWindowEvent(sdlwin, SDL_WINDOWEVENT_LEAVE, 0, 0);
            FocusAllMice(_this, 0);
            /* SDL_DirectFB_ReshowCursor(_this, 1); */
            break;
        default:
            ;
        }
    } else
        printf("Event Clazz %d\n", evt->clazz);
}

static void ProcessInputEvent(_THIS, DFBInputEvent * ievt)
{
    SDL_DFB_DEVICEDATA(_this);
    SDL_Keysym keysym;
    int kbd_idx;
    Uint32 unicode;
    char text[SDL_TEXTINPUTEVENT_TEXT_SIZE];
    SDL_Window* grabbed_window = SDL_GetGrabbedWindow();

    if (!devdata->use_linux_input) {
        if (ievt->type == DIET_AXISMOTION) {
            if ((grabbed_window) && (ievt->flags & DIEF_AXISREL)) {
                if (ievt->axis == DIAI_X)
                    SDL_SendMouseMotion_ex(grabbed_window, ievt->device_id, 1,
                                        ievt->axisrel, 0, 0);
                else if (ievt->axis == DIAI_Y)
                    SDL_SendMouseMotion_ex(grabbed_window, ievt->device_id, 1, 0,
                                        ievt->axisrel, 0);
            }
        }
    } else {
        static int last_x, last_y;

        switch (ievt->type) {
        case DIET_AXISMOTION:
            if (ievt->flags & DIEF_AXISABS) {
                if (ievt->axis == DIAI_X)
                    last_x = ievt->axisabs;
                else if (ievt->axis == DIAI_Y)
                    last_y = ievt->axisabs;
                if (!(ievt->flags & DIEF_FOLLOW)) {
#if USE_MULTI_API
                    SDL_Mouse *mouse = SDL_GetMouse(ievt->device_id);
                    SDL_Window *window = SDL_GetWindowFromID(mouse->focus);
#else
                    SDL_Window *window = grabbed_window;
#endif
                    if (window) {
                        DFB_WindowData *windata =
                            (DFB_WindowData *) window->driverdata;
                        int x, y;

                        windata->dfbwin->GetPosition(windata->dfbwin, &x, &y);
                        SDL_SendMouseMotion_ex(window, ievt->device_id, 0,
                                            last_x - (x +
                                                      windata->client.x),
                                            last_y - (y +
                                                      windata->client.y), 0);
                    } else {
                        SDL_SendMouseMotion_ex(window, ievt->device_id, 0, last_x,
                                            last_y, 0);
                    }
                }
            } else if (ievt->flags & DIEF_AXISREL) {
                if (ievt->axis == DIAI_X)
                    SDL_SendMouseMotion_ex(grabbed_window, ievt->device_id, 1,
                                        ievt->axisrel, 0, 0);
                else if (ievt->axis == DIAI_Y)
                    SDL_SendMouseMotion_ex(grabbed_window, ievt->device_id, 1, 0,
                                        ievt->axisrel, 0);
            }
            break;
        case DIET_KEYPRESS:
            kbd_idx = KbdIndex(_this, ievt->device_id);
            DirectFB_TranslateKeyInputEvent(_this, ievt, &keysym, &unicode);
            /* printf("Scancode %d  %d %d\n", keysym.scancode, evt->key_code, evt->key_id); */
            SDL_SendKeyboardKey_ex(kbd_idx, SDL_PRESSED, keysym.scancode);
            if (SDL_EventState(SDL_TEXTINPUT, SDL_QUERY)) {
                SDL_zeroa(text);
                UnicodeToUtf8(unicode, text);
                if (*text) {
                    SDL_SendKeyboardText_ex(kbd_idx, text);
                }
            }
            break;
        case DIET_KEYRELEASE:
            kbd_idx = KbdIndex(_this, ievt->device_id);
            DirectFB_TranslateKeyInputEvent(_this, ievt, &keysym, &unicode);
            SDL_SendKeyboardKey_ex(kbd_idx, SDL_RELEASED, keysym.scancode);
            break;
        case DIET_BUTTONPRESS:
            if (ievt->buttons & DIBM_LEFT)
                SDL_SendMouseButton_ex(grabbed_window, ievt->device_id, SDL_PRESSED, 1);
            if (ievt->buttons & DIBM_MIDDLE)
                SDL_SendMouseButton_ex(grabbed_window, ievt->device_id, SDL_PRESSED, 2);
            if (ievt->buttons & DIBM_RIGHT)
                SDL_SendMouseButton_ex(grabbed_window, ievt->device_id, SDL_PRESSED, 3);
            break;
        case DIET_BUTTONRELEASE:
            if (!(ievt->buttons & DIBM_LEFT))
                SDL_SendMouseButton_ex(grabbed_window, ievt->device_id, SDL_RELEASED, 1);
            if (!(ievt->buttons & DIBM_MIDDLE))
                SDL_SendMouseButton_ex(grabbed_window, ievt->device_id, SDL_RELEASED, 2);
            if (!(ievt->buttons & DIBM_RIGHT))
                SDL_SendMouseButton_ex(grabbed_window, ievt->device_id, SDL_RELEASED, 3);
            break;
        default:
            break;              /* please gcc */
        }
    }
}

void DirectFB_PumpEventsWindow(_THIS)
{
    SDL_DFB_DEVICEDATA(_this);
    DFBInputEvent ievt;
    SDL_Window *w;

    for (w = devdata->firstwin; w; w = w->next) {
        SDL_DFB_WINDOWDATA(w);
        DFBWindowEvent evt;

        while (windata->eventbuffer->GetEvent(windata->eventbuffer,
                                        DFB_EVENT(&evt)) == DFB_OK) {
            if (!DirectFB_WM_ProcessEvent(_this, w, &evt)) {
                /* Send a SDL_SYSWMEVENT if the application wants them */
                if (SDL_GetEventState(SDL_SYSWMEVENT) == SDL_ENABLE) {
                    SDL_SysWMmsg wmmsg;
                    SDL_VERSION(&wmmsg.version);
                    wmmsg.subsystem = SDL_SYSWM_DIRECTFB;
                    wmmsg.msg.dfb.event.window = evt;
                    SDL_SendSysWMEvent(&wmmsg);
                }
                ProcessWindowEvent(_this, w, &evt);
            }
        }
    }

    /* Now get relative events in case we need them */
    while (devdata->events->GetEvent(devdata->events,
                                     DFB_EVENT(&ievt)) == DFB_OK) {

        if (SDL_GetEventState(SDL_SYSWMEVENT) == SDL_ENABLE) {
            SDL_SysWMmsg wmmsg;
            SDL_VERSION(&wmmsg.version);
            wmmsg.subsystem = SDL_SYSWM_DIRECTFB;
            wmmsg.msg.dfb.event.input = ievt;
            SDL_SendSysWMEvent(&wmmsg);
        }
        ProcessInputEvent(_this, &ievt);
    }
}

void DirectFB_InitOSKeymap(_THIS, SDL_Scancode * keymap, int numkeys)
{
    int i;

    /* Initialize the DirectFB key translation table */
    for (i = 0; i < numkeys; ++i)
        keymap[i] = SDL_SCANCODE_UNKNOWN;

    keymap[DIKI_A - DIKI_UNKNOWN] = SDL_SCANCODE_A;
    keymap[DIKI_B - DIKI_UNKNOWN] = SDL_SCANCODE_B;
    keymap[DIKI_C - DIKI_UNKNOWN] = SDL_SCANCODE_C;
    keymap[DIKI_D - DIKI_UNKNOWN] = SDL_SCANCODE_D;
    keymap[DIKI_E - DIKI_UNKNOWN] = SDL_SCANCODE_E;
    keymap[DIKI_F - DIKI_UNKNOWN] = SDL_SCANCODE_F;
    keymap[DIKI_G - DIKI_UNKNOWN] = SDL_SCANCODE_G;
    keymap[DIKI_H - DIKI_UNKNOWN] = SDL_SCANCODE_H;
    keymap[DIKI_I - DIKI_UNKNOWN] = SDL_SCANCODE_I;
    keymap[DIKI_J - DIKI_UNKNOWN] = SDL_SCANCODE_J;
    keymap[DIKI_K - DIKI_UNKNOWN] = SDL_SCANCODE_K;
    keymap[DIKI_L - DIKI_UNKNOWN] = SDL_SCANCODE_L;
    keymap[DIKI_M - DIKI_UNKNOWN] = SDL_SCANCODE_M;
    keymap[DIKI_N - DIKI_UNKNOWN] = SDL_SCANCODE_N;
    keymap[DIKI_O - DIKI_UNKNOWN] = SDL_SCANCODE_O;
    keymap[DIKI_P - DIKI_UNKNOWN] = SDL_SCANCODE_P;
    keymap[DIKI_Q - DIKI_UNKNOWN] = SDL_SCANCODE_Q;
    keymap[DIKI_R - DIKI_UNKNOWN] = SDL_SCANCODE_R;
    keymap[DIKI_S - DIKI_UNKNOWN] = SDL_SCANCODE_S;
    keymap[DIKI_T - DIKI_UNKNOWN] = SDL_SCANCODE_T;
    keymap[DIKI_U - DIKI_UNKNOWN] = SDL_SCANCODE_U;
    keymap[DIKI_V - DIKI_UNKNOWN] = SDL_SCANCODE_V;
    keymap[DIKI_W - DIKI_UNKNOWN] = SDL_SCANCODE_W;
    keymap[DIKI_X - DIKI_UNKNOWN] = SDL_SCANCODE_X;
    keymap[DIKI_Y - DIKI_UNKNOWN] = SDL_SCANCODE_Y;
    keymap[DIKI_Z - DIKI_UNKNOWN] = SDL_SCANCODE_Z;

    keymap[DIKI_0 - DIKI_UNKNOWN] = SDL_SCANCODE_0;
    keymap[DIKI_1 - DIKI_UNKNOWN] = SDL_SCANCODE_1;
    keymap[DIKI_2 - DIKI_UNKNOWN] = SDL_SCANCODE_2;
    keymap[DIKI_3 - DIKI_UNKNOWN] = SDL_SCANCODE_3;
    keymap[DIKI_4 - DIKI_UNKNOWN] = SDL_SCANCODE_4;
    keymap[DIKI_5 - DIKI_UNKNOWN] = SDL_SCANCODE_5;
    keymap[DIKI_6 - DIKI_UNKNOWN] = SDL_SCANCODE_6;
    keymap[DIKI_7 - DIKI_UNKNOWN] = SDL_SCANCODE_7;
    keymap[DIKI_8 - DIKI_UNKNOWN] = SDL_SCANCODE_8;
    keymap[DIKI_9 - DIKI_UNKNOWN] = SDL_SCANCODE_9;

    keymap[DIKI_F1 - DIKI_UNKNOWN] = SDL_SCANCODE_F1;
    keymap[DIKI_F2 - DIKI_UNKNOWN] = SDL_SCANCODE_F2;
    keymap[DIKI_F3 - DIKI_UNKNOWN] = SDL_SCANCODE_F3;
    keymap[DIKI_F4 - DIKI_UNKNOWN] = SDL_SCANCODE_F4;
    keymap[DIKI_F5 - DIKI_UNKNOWN] = SDL_SCANCODE_F5;
    keymap[DIKI_F6 - DIKI_UNKNOWN] = SDL_SCANCODE_F6;
    keymap[DIKI_F7 - DIKI_UNKNOWN] = SDL_SCANCODE_F7;
    keymap[DIKI_F8 - DIKI_UNKNOWN] = SDL_SCANCODE_F8;
    keymap[DIKI_F9 - DIKI_UNKNOWN] = SDL_SCANCODE_F9;
    keymap[DIKI_F10 - DIKI_UNKNOWN] = SDL_SCANCODE_F10;
    keymap[DIKI_F11 - DIKI_UNKNOWN] = SDL_SCANCODE_F11;
    keymap[DIKI_F12 - DIKI_UNKNOWN] = SDL_SCANCODE_F12;

    keymap[DIKI_ESCAPE - DIKI_UNKNOWN] = SDL_SCANCODE_ESCAPE;
    keymap[DIKI_LEFT - DIKI_UNKNOWN] = SDL_SCANCODE_LEFT;
    keymap[DIKI_RIGHT - DIKI_UNKNOWN] = SDL_SCANCODE_RIGHT;
    keymap[DIKI_UP - DIKI_UNKNOWN] = SDL_SCANCODE_UP;
    keymap[DIKI_DOWN - DIKI_UNKNOWN] = SDL_SCANCODE_DOWN;
    keymap[DIKI_CONTROL_L - DIKI_UNKNOWN] = SDL_SCANCODE_LCTRL;
    keymap[DIKI_CONTROL_R - DIKI_UNKNOWN] = SDL_SCANCODE_RCTRL;
    keymap[DIKI_SHIFT_L - DIKI_UNKNOWN] = SDL_SCANCODE_LSHIFT;
    keymap[DIKI_SHIFT_R - DIKI_UNKNOWN] = SDL_SCANCODE_RSHIFT;
    keymap[DIKI_ALT_L - DIKI_UNKNOWN] = SDL_SCANCODE_LALT;
    keymap[DIKI_ALT_R - DIKI_UNKNOWN] = SDL_SCANCODE_RALT;
    keymap[DIKI_META_L - DIKI_UNKNOWN] = SDL_SCANCODE_LGUI;
    keymap[DIKI_META_R - DIKI_UNKNOWN] = SDL_SCANCODE_RGUI;
    keymap[DIKI_SUPER_L - DIKI_UNKNOWN] = SDL_SCANCODE_APPLICATION;
    keymap[DIKI_SUPER_R - DIKI_UNKNOWN] = SDL_SCANCODE_APPLICATION;
    /* FIXME:Do we read hyper keys ?
     * keymap[DIKI_HYPER_L - DIKI_UNKNOWN] = SDL_SCANCODE_APPLICATION;
     * keymap[DIKI_HYPER_R - DIKI_UNKNOWN] = SDL_SCANCODE_APPLICATION;
     */
    keymap[DIKI_TAB - DIKI_UNKNOWN] = SDL_SCANCODE_TAB;
    keymap[DIKI_ENTER - DIKI_UNKNOWN] = SDL_SCANCODE_RETURN;
    keymap[DIKI_SPACE - DIKI_UNKNOWN] = SDL_SCANCODE_SPACE;
    keymap[DIKI_BACKSPACE - DIKI_UNKNOWN] = SDL_SCANCODE_BACKSPACE;
    keymap[DIKI_INSERT - DIKI_UNKNOWN] = SDL_SCANCODE_INSERT;
    keymap[DIKI_DELETE - DIKI_UNKNOWN] = SDL_SCANCODE_DELETE;
    keymap[DIKI_HOME - DIKI_UNKNOWN] = SDL_SCANCODE_HOME;
    keymap[DIKI_END - DIKI_UNKNOWN] = SDL_SCANCODE_END;
    keymap[DIKI_PAGE_UP - DIKI_UNKNOWN] = SDL_SCANCODE_PAGEUP;
    keymap[DIKI_PAGE_DOWN - DIKI_UNKNOWN] = SDL_SCANCODE_PAGEDOWN;
    keymap[DIKI_CAPS_LOCK - DIKI_UNKNOWN] = SDL_SCANCODE_CAPSLOCK;
    keymap[DIKI_NUM_LOCK - DIKI_UNKNOWN] = SDL_SCANCODE_NUMLOCKCLEAR;
    keymap[DIKI_SCROLL_LOCK - DIKI_UNKNOWN] = SDL_SCANCODE_SCROLLLOCK;
    keymap[DIKI_PRINT - DIKI_UNKNOWN] = SDL_SCANCODE_PRINTSCREEN;
    keymap[DIKI_PAUSE - DIKI_UNKNOWN] = SDL_SCANCODE_PAUSE;

    keymap[DIKI_KP_EQUAL - DIKI_UNKNOWN] = SDL_SCANCODE_KP_EQUALS;
    keymap[DIKI_KP_DECIMAL - DIKI_UNKNOWN] = SDL_SCANCODE_KP_PERIOD;
    keymap[DIKI_KP_0 - DIKI_UNKNOWN] = SDL_SCANCODE_KP_0;
    keymap[DIKI_KP_1 - DIKI_UNKNOWN] = SDL_SCANCODE_KP_1;
    keymap[DIKI_KP_2 - DIKI_UNKNOWN] = SDL_SCANCODE_KP_2;
    keymap[DIKI_KP_3 - DIKI_UNKNOWN] = SDL_SCANCODE_KP_3;
    keymap[DIKI_KP_4 - DIKI_UNKNOWN] = SDL_SCANCODE_KP_4;
    keymap[DIKI_KP_5 - DIKI_UNKNOWN] = SDL_SCANCODE_KP_5;
    keymap[DIKI_KP_6 - DIKI_UNKNOWN] = SDL_SCANCODE_KP_6;
    keymap[DIKI_KP_7 - DIKI_UNKNOWN] = SDL_SCANCODE_KP_7;
    keymap[DIKI_KP_8 - DIKI_UNKNOWN] = SDL_SCANCODE_KP_8;
    keymap[DIKI_KP_9 - DIKI_UNKNOWN] = SDL_SCANCODE_KP_9;
    keymap[DIKI_KP_DIV - DIKI_UNKNOWN] = SDL_SCANCODE_KP_DIVIDE;
    keymap[DIKI_KP_MULT - DIKI_UNKNOWN] = SDL_SCANCODE_KP_MULTIPLY;
    keymap[DIKI_KP_MINUS - DIKI_UNKNOWN] = SDL_SCANCODE_KP_MINUS;
    keymap[DIKI_KP_PLUS - DIKI_UNKNOWN] = SDL_SCANCODE_KP_PLUS;
    keymap[DIKI_KP_ENTER - DIKI_UNKNOWN] = SDL_SCANCODE_KP_ENTER;

    keymap[DIKI_QUOTE_LEFT - DIKI_UNKNOWN] = SDL_SCANCODE_GRAVE;        /*  TLDE  */
    keymap[DIKI_MINUS_SIGN - DIKI_UNKNOWN] = SDL_SCANCODE_MINUS;        /*  AE11  */
    keymap[DIKI_EQUALS_SIGN - DIKI_UNKNOWN] = SDL_SCANCODE_EQUALS;      /*  AE12  */
    keymap[DIKI_BRACKET_LEFT - DIKI_UNKNOWN] = SDL_SCANCODE_RIGHTBRACKET;       /*  AD11  */
    keymap[DIKI_BRACKET_RIGHT - DIKI_UNKNOWN] = SDL_SCANCODE_LEFTBRACKET;       /*  AD12  */
    keymap[DIKI_BACKSLASH - DIKI_UNKNOWN] = SDL_SCANCODE_BACKSLASH;     /*  BKSL  */
    keymap[DIKI_SEMICOLON - DIKI_UNKNOWN] = SDL_SCANCODE_SEMICOLON;     /*  AC10  */
    keymap[DIKI_QUOTE_RIGHT - DIKI_UNKNOWN] = SDL_SCANCODE_APOSTROPHE;  /*  AC11  */
    keymap[DIKI_COMMA - DIKI_UNKNOWN] = SDL_SCANCODE_COMMA;     /*  AB08  */
    keymap[DIKI_PERIOD - DIKI_UNKNOWN] = SDL_SCANCODE_PERIOD;   /*  AB09  */
    keymap[DIKI_SLASH - DIKI_UNKNOWN] = SDL_SCANCODE_SLASH;     /*  AB10  */
    keymap[DIKI_LESS_SIGN - DIKI_UNKNOWN] = SDL_SCANCODE_NONUSBACKSLASH;        /*  103rd  */

}

static SDL_Keysym *DirectFB_TranslateKey(_THIS, DFBWindowEvent * evt, SDL_Keysym * keysym, Uint32 *unicode)
{
    SDL_DFB_DEVICEDATA(_this);
    int kbd_idx = 0; /* Window events lag the device source KbdIndex(_this, evt->device_id); */
    DFB_KeyboardData *kbd = &devdata->keyboard[kbd_idx];

    keysym->scancode = SDL_SCANCODE_UNKNOWN;

    if (kbd->map && evt->key_code >= kbd->map_adjust &&
        evt->key_code < kbd->map_size + kbd->map_adjust)
        keysym->scancode = kbd->map[evt->key_code - kbd->map_adjust];

    if (keysym->scancode == SDL_SCANCODE_UNKNOWN ||
        devdata->keyboard[kbd_idx].is_generic) {
        if (evt->key_id - DIKI_UNKNOWN < SDL_arraysize(oskeymap))
            keysym->scancode = oskeymap[evt->key_id - DIKI_UNKNOWN];
        else
            keysym->scancode = SDL_SCANCODE_UNKNOWN;
    }

    *unicode =
        (DFB_KEY_TYPE(evt->key_symbol) == DIKT_UNICODE) ? evt->key_symbol : 0;
    if (*unicode == 0 &&
        (evt->key_symbol > 0 && evt->key_symbol < 255))
        *unicode = evt->key_symbol;

    return keysym;
}

static SDL_Keysym *DirectFB_TranslateKeyInputEvent(_THIS, DFBInputEvent * evt,
                                SDL_Keysym * keysym, Uint32 *unicode)
{
    SDL_DFB_DEVICEDATA(_this);
    int kbd_idx = KbdIndex(_this, evt->device_id);
    DFB_KeyboardData *kbd = &devdata->keyboard[kbd_idx];

    keysym->scancode = SDL_SCANCODE_UNKNOWN;

    if (kbd->map && evt->key_code >= kbd->map_adjust &&
        evt->key_code < kbd->map_size + kbd->map_adjust)
        keysym->scancode = kbd->map[evt->key_code - kbd->map_adjust];

    if (keysym->scancode == SDL_SCANCODE_UNKNOWN || devdata->keyboard[kbd_idx].is_generic) {
        if (evt->key_id - DIKI_UNKNOWN < SDL_arraysize(oskeymap))
            keysym->scancode = oskeymap[evt->key_id - DIKI_UNKNOWN];
        else
            keysym->scancode = SDL_SCANCODE_UNKNOWN;
    }

    *unicode =
        (DFB_KEY_TYPE(evt->key_symbol) == DIKT_UNICODE) ? evt->key_symbol : 0;
    if (*unicode == 0 &&
        (evt->key_symbol > 0 && evt->key_symbol < 255))
        *unicode = evt->key_symbol;

    return keysym;
}

static int DirectFB_TranslateButton(DFBInputDeviceButtonIdentifier button)
{
    switch (button) {
    case DIBI_LEFT:
        return 1;
    case DIBI_MIDDLE:
        return 2;
    case DIBI_RIGHT:
        return 3;
    default:
        return 0;
    }
}

static DFBEnumerationResult EnumKeyboards(DFBInputDeviceID device_id, DFBInputDeviceDescription desc, void *callbackdata)
{
    cb_data *cb = callbackdata;
    DFB_DeviceData *devdata = cb->devdata;
#if USE_MULTI_API
    SDL_Keyboard keyboard;
#endif

    if (!cb->sys_kbd) {
        if (cb->sys_ids) {
            if (device_id >= 0x10)
                return DFENUM_OK;
        } else {
            if (device_id < 0x10)
                return DFENUM_OK;
        }
    } else {
        if (device_id != DIDID_KEYBOARD)
            return DFENUM_OK;
    }

    if ((desc.caps & DIDTF_KEYBOARD)) {
#if USE_MULTI_API
        SDL_zero(keyboard);
        SDL_AddKeyboard(&keyboard, devdata->num_keyboard);
#endif
        devdata->keyboard[devdata->num_keyboard].id = device_id;
        devdata->keyboard[devdata->num_keyboard].is_generic = 0;
        if (!SDL_strncmp("X11", desc.name, 3))
        {
            devdata->keyboard[devdata->num_keyboard].map = SDL_GetScancodeTable(SDL_SCANCODE_TABLE_XFREE86_2, &devdata->keyboard[devdata->num_keyboard].map_size);
            devdata->keyboard[devdata->num_keyboard].map_adjust = 8;
        } else {
            devdata->keyboard[devdata->num_keyboard].map = SDL_GetScancodeTable(SDL_SCANCODE_TABLE_LINUX, &devdata->keyboard[devdata->num_keyboard].map_size);
            devdata->keyboard[devdata->num_keyboard].map_adjust = 0;
        }

        SDL_DFB_LOG("Keyboard %d - %s\n", device_id, desc.name);

        devdata->num_keyboard++;

        if (cb->sys_kbd)
            return DFENUM_CANCEL;
    }
    return DFENUM_OK;
}

void DirectFB_InitKeyboard(_THIS)
{
    SDL_DFB_DEVICEDATA(_this);
    cb_data cb;

    DirectFB_InitOSKeymap(_this, &oskeymap[0], SDL_arraysize(oskeymap));

    devdata->num_keyboard = 0;
    cb.devdata = devdata;

    if (devdata->use_linux_input) {
        cb.sys_kbd = 0;
        cb.sys_ids = 0;
        SDL_DFB_CHECK(devdata->dfb->
                      EnumInputDevices(devdata->dfb, EnumKeyboards, &cb));
        if (devdata->num_keyboard == 0) {
            cb.sys_ids = 1;
            SDL_DFB_CHECK(devdata->dfb->EnumInputDevices(devdata->dfb,
                                                         EnumKeyboards,
                                                         &cb));
        }
    } else {
        cb.sys_kbd = 1;
        SDL_DFB_CHECK(devdata->dfb->EnumInputDevices(devdata->dfb,
                                                     EnumKeyboards,
                                                     &cb));
    }
}

void DirectFB_QuitKeyboard(_THIS)
{
    /* SDL_DFB_DEVICEDATA(_this); */
}

#endif /* SDL_VIDEO_DRIVER_DIRECTFB */
