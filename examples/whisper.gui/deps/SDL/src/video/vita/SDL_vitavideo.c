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

#ifdef SDL_VIDEO_DRIVER_VITA

/* SDL internals */
#include "../SDL_sysvideo.h"
#include "SDL_version.h"
#include "SDL_syswm.h"
#include "SDL_loadso.h"
#include "SDL_events.h"
#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_keyboard_c.h"

/* VITA declarations */
#include <psp2/kernel/processmgr.h>
#include "SDL_vitavideo.h"
#include "SDL_vitatouch.h"
#include "SDL_vitakeyboard.h"
#include "SDL_vitamouse_c.h"
#include "SDL_vitaframebuffer.h"
#include "SDL_vitamessagebox.h"

#if defined(SDL_VIDEO_VITA_PIB)
#include "SDL_vitagles_c.h"
#elif defined(SDL_VIDEO_VITA_PVR)
#include "SDL_vitagles_pvr_c.h"
#if defined(SDL_VIDEO_VITA_PVR_OGL)
#include "SDL_vitagl_pvr_c.h"
#endif
  #define VITA_GLES_GetProcAddress SDL_EGL_GetProcAddress
  #define VITA_GLES_UnloadLibrary SDL_EGL_UnloadLibrary
  #define VITA_GLES_SetSwapInterval SDL_EGL_SetSwapInterval
  #define VITA_GLES_GetSwapInterval SDL_EGL_GetSwapInterval
  #define VITA_GLES_DeleteContext SDL_EGL_DeleteContext
#endif

SDL_Window *Vita_Window;

static void VITA_Destroy(SDL_VideoDevice *device)
{
    /*    SDL_VideoData *phdata = (SDL_VideoData *) device->driverdata; */

    SDL_free(device->driverdata);
    SDL_free(device);
    //    if (device->driverdata != NULL) {
    //        device->driverdata = NULL;
    //    }
}

static SDL_VideoDevice *VITA_Create()
{
    SDL_VideoDevice *device;
    SDL_VideoData *phdata;
#ifdef SDL_VIDEO_VITA_PIB
    SDL_GLDriverData *gldata;
#endif
    /* Initialize SDL_VideoDevice structure */
    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        SDL_OutOfMemory();
        return NULL;
    }

    /* Initialize internal VITA specific data */
    phdata = (SDL_VideoData *)SDL_calloc(1, sizeof(SDL_VideoData));
    if (!phdata) {
        SDL_OutOfMemory();
        SDL_free(device);
        return NULL;
    }
#ifdef SDL_VIDEO_VITA_PIB

    gldata = (SDL_GLDriverData *)SDL_calloc(1, sizeof(SDL_GLDriverData));
    if (!gldata) {
        SDL_OutOfMemory();
        SDL_free(device);
        SDL_free(phdata);
        return NULL;
    }
    device->gl_data = gldata;
    phdata->egl_initialized = SDL_TRUE;
#endif
    phdata->ime_active = SDL_FALSE;

    device->driverdata = phdata;

    /* Setup amount of available displays and current display */
    device->num_displays = 0;

    /* Set device free function */
    device->free = VITA_Destroy;

    /* Setup all functions which we can handle */
    device->VideoInit = VITA_VideoInit;
    device->VideoQuit = VITA_VideoQuit;
    device->GetDisplayModes = VITA_GetDisplayModes;
    device->SetDisplayMode = VITA_SetDisplayMode;
    device->CreateSDLWindow = VITA_CreateWindow;
    device->CreateSDLWindowFrom = VITA_CreateWindowFrom;
    device->SetWindowTitle = VITA_SetWindowTitle;
    device->SetWindowIcon = VITA_SetWindowIcon;
    device->SetWindowPosition = VITA_SetWindowPosition;
    device->SetWindowSize = VITA_SetWindowSize;
    device->ShowWindow = VITA_ShowWindow;
    device->HideWindow = VITA_HideWindow;
    device->RaiseWindow = VITA_RaiseWindow;
    device->MaximizeWindow = VITA_MaximizeWindow;
    device->MinimizeWindow = VITA_MinimizeWindow;
    device->RestoreWindow = VITA_RestoreWindow;
    device->SetWindowMouseGrab = VITA_SetWindowGrab;
    device->SetWindowKeyboardGrab = VITA_SetWindowGrab;
    device->DestroyWindow = VITA_DestroyWindow;
    device->GetWindowWMInfo = VITA_GetWindowWMInfo;

    /*
        // Disabled, causes issues on high-framerate updates. SDL still emulates this.
        device->CreateWindowFramebuffer = VITA_CreateWindowFramebuffer;
        device->UpdateWindowFramebuffer = VITA_UpdateWindowFramebuffer;
        device->DestroyWindowFramebuffer = VITA_DestroyWindowFramebuffer;
    */

#if defined(SDL_VIDEO_VITA_PIB) || defined(SDL_VIDEO_VITA_PVR)
#if defined(SDL_VIDEO_VITA_PVR_OGL)
    if (SDL_getenv("VITA_PVR_OGL") != NULL) {
        device->GL_LoadLibrary = VITA_GL_LoadLibrary;
        device->GL_CreateContext = VITA_GL_CreateContext;
        device->GL_GetProcAddress = VITA_GL_GetProcAddress;
    } else {
#endif
        device->GL_LoadLibrary = VITA_GLES_LoadLibrary;
        device->GL_CreateContext = VITA_GLES_CreateContext;
        device->GL_GetProcAddress = VITA_GLES_GetProcAddress;
#if defined(SDL_VIDEO_VITA_PVR_OGL)
    }
#endif

    device->GL_UnloadLibrary = VITA_GLES_UnloadLibrary;
    device->GL_MakeCurrent = VITA_GLES_MakeCurrent;
    device->GL_SetSwapInterval = VITA_GLES_SetSwapInterval;
    device->GL_GetSwapInterval = VITA_GLES_GetSwapInterval;
    device->GL_SwapWindow = VITA_GLES_SwapWindow;
    device->GL_DeleteContext = VITA_GLES_DeleteContext;
#endif

    device->HasScreenKeyboardSupport = VITA_HasScreenKeyboardSupport;
    device->ShowScreenKeyboard = VITA_ShowScreenKeyboard;
    device->HideScreenKeyboard = VITA_HideScreenKeyboard;
    device->IsScreenKeyboardShown = VITA_IsScreenKeyboardShown;

    device->PumpEvents = VITA_PumpEvents;

    return device;
}

VideoBootStrap VITA_bootstrap = {
    "VITA",
    "VITA Video Driver",
    VITA_Create,
    VITA_ShowMessageBox
};

/*****************************************************************************/
/* SDL Video and Display initialization/handling functions                   */
/*****************************************************************************/
int VITA_VideoInit(_THIS)
{
    SDL_VideoDisplay display;
    SDL_DisplayMode current_mode;
#if defined(SDL_VIDEO_VITA_PVR)
    char *res = SDL_getenv("VITA_RESOLUTION");
#endif
    SDL_zero(current_mode);

#if defined(SDL_VIDEO_VITA_PVR)
    if (res) {
        /* 1088i for PSTV (Or Sharpscale) */
        if (!SDL_strncmp(res, "1080", 4)) {
            current_mode.w = 1920;
            current_mode.h = 1088;
        }
        /* 725p for PSTV (Or Sharpscale) */
        else if (!SDL_strncmp(res, "720", 3)) {
            current_mode.w = 1280;
            current_mode.h = 725;
        }
    }
    /* 544p */
    else {
#endif
        current_mode.w = 960;
        current_mode.h = 544;
#if defined(SDL_VIDEO_VITA_PVR)
    }
#endif

    current_mode.refresh_rate = 60;
    /* 32 bpp for default */
    current_mode.format = SDL_PIXELFORMAT_ABGR8888;

    current_mode.driverdata = NULL;

    SDL_zero(display);
    display.desktop_mode = current_mode;
    display.current_mode = current_mode;
    display.driverdata = NULL;

    SDL_AddVideoDisplay(&display, SDL_FALSE);
    VITA_InitTouch();
    VITA_InitKeyboard();
    VITA_InitMouse();

    return 1;
}

void VITA_VideoQuit(_THIS)
{
    VITA_QuitTouch();
}

void VITA_GetDisplayModes(_THIS, SDL_VideoDisplay *display)
{
    SDL_AddDisplayMode(display, &display->current_mode);
}

int VITA_SetDisplayMode(_THIS, SDL_VideoDisplay *display, SDL_DisplayMode *mode)
{
    return 0;
}

int VITA_CreateWindow(_THIS, SDL_Window *window)
{
    SDL_WindowData *wdata;
#if defined(SDL_VIDEO_VITA_PVR)
    Psp2NativeWindow win;
    int temp_major = 2;
    int temp_minor = 1;
    int temp_profile = 0;
#endif

    /* Allocate window internal data */
    wdata = (SDL_WindowData *)SDL_calloc(1, sizeof(SDL_WindowData));
    if (!wdata) {
        return SDL_OutOfMemory();
    }

    /* Setup driver data for this window */
    window->driverdata = wdata;

    // Vita can only have one window
    if (Vita_Window) {
        return SDL_SetError("Only one window supported");
    }

    Vita_Window = window;

#if defined(SDL_VIDEO_VITA_PVR)
    win.type = PSP2_DRAWABLE_TYPE_WINDOW;
    win.numFlipBuffers = 2;
    win.flipChainThrdAffinity = 0x20000;

    /* 1088i for PSTV (Or Sharpscale) */
    if (window->w == 1920) {
        win.windowSize = PSP2_WINDOW_1920X1088;
    }
    /* 725p for PSTV (Or Sharpscale) */
    else if (window->w == 1280) {
        win.windowSize = PSP2_WINDOW_1280X725;
    }
    /* 544p */
    else {
        win.windowSize = PSP2_WINDOW_960X544;
    }
    if (window->flags & SDL_WINDOW_OPENGL) {
        if (SDL_getenv("VITA_PVR_OGL") != NULL) {
            /* Set version to 2.1 and PROFILE to ES */
            temp_major = _this->gl_config.major_version;
            temp_minor = _this->gl_config.minor_version;
            temp_profile = _this->gl_config.profile_mask;

            _this->gl_config.major_version = 2;
            _this->gl_config.minor_version = 1;
            _this->gl_config.profile_mask = SDL_GL_CONTEXT_PROFILE_ES;
        }
        wdata->egl_surface = SDL_EGL_CreateSurface(_this, &win);
        if (wdata->egl_surface == EGL_NO_SURFACE) {
            return SDL_SetError("Could not create GLES window surface");
        }
        if (SDL_getenv("VITA_PVR_OGL") != NULL) {
            /* Revert */
            _this->gl_config.major_version = temp_major;
            _this->gl_config.minor_version = temp_minor;
            _this->gl_config.profile_mask = temp_profile;
        }
    }
#endif

    // fix input, we need to find a better way
    SDL_SetKeyboardFocus(window);

    /* Window has been successfully created */
    return 0;
}

int VITA_CreateWindowFrom(_THIS, SDL_Window *window, const void *data)
{
    return -1;
}

void VITA_SetWindowTitle(_THIS, SDL_Window *window)
{
}
void VITA_SetWindowIcon(_THIS, SDL_Window *window, SDL_Surface *icon)
{
}
void VITA_SetWindowPosition(_THIS, SDL_Window *window)
{
}
void VITA_SetWindowSize(_THIS, SDL_Window *window)
{
}
void VITA_ShowWindow(_THIS, SDL_Window *window)
{
}
void VITA_HideWindow(_THIS, SDL_Window *window)
{
}
void VITA_RaiseWindow(_THIS, SDL_Window *window)
{
}
void VITA_MaximizeWindow(_THIS, SDL_Window *window)
{
}
void VITA_MinimizeWindow(_THIS, SDL_Window *window)
{
}
void VITA_RestoreWindow(_THIS, SDL_Window *window)
{
}
void VITA_SetWindowGrab(_THIS, SDL_Window *window, SDL_bool grabbed)
{
}

void VITA_DestroyWindow(_THIS, SDL_Window *window)
{
    //    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
    SDL_WindowData *data;

    data = window->driverdata;
    if (data) {
        // TODO: should we destroy egl context? No one sane should recreate ogl window as non-ogl
        SDL_free(data);
    }

    window->driverdata = NULL;
    Vita_Window = NULL;
}

/*****************************************************************************/
/* SDL Window Manager function                                               */
/*****************************************************************************/
SDL_bool VITA_GetWindowWMInfo(_THIS, SDL_Window * window, struct SDL_SysWMinfo *info)
{
    if (info->version.major <= SDL_MAJOR_VERSION) {
        return SDL_TRUE;
    } else {
        SDL_SetError("application not compiled with SDL %d\n",
                     SDL_MAJOR_VERSION);
        return SDL_FALSE;
    }

    /* Failed to get window manager information */
    return SDL_FALSE;
}

SDL_bool VITA_HasScreenKeyboardSupport(_THIS)
{
    return SDL_TRUE;
}

#if !defined(SCE_IME_LANGUAGE_ENGLISH_US)
#define SCE_IME_LANGUAGE_ENGLISH_US SCE_IME_LANGUAGE_ENGLISH
#endif

static void utf16_to_utf8(const uint16_t *src, uint8_t *dst)
{
    int i;
    for (i = 0; src[i]; i++) {
        if (!(src[i] & 0xFF80)) {
            *(dst++) = src[i] & 0xFF;
        } else if (!(src[i] & 0xF800)) {
            *(dst++) = ((src[i] >> 6) & 0xFF) | 0xC0;
            *(dst++) = (src[i] & 0x3F) | 0x80;
        } else if ((src[i] & 0xFC00) == 0xD800 && (src[i + 1] & 0xFC00) == 0xDC00) {
            *(dst++) = (((src[i] + 64) >> 8) & 0x3) | 0xF0;
            *(dst++) = (((src[i] >> 2) + 16) & 0x3F) | 0x80;
            *(dst++) = ((src[i] >> 4) & 0x30) | 0x80 | ((src[i + 1] << 2) & 0xF);
            *(dst++) = (src[i + 1] & 0x3F) | 0x80;
            i += 1;
        } else {
            *(dst++) = ((src[i] >> 12) & 0xF) | 0xE0;
            *(dst++) = ((src[i] >> 6) & 0x3F) | 0x80;
            *(dst++) = (src[i] & 0x3F) | 0x80;
        }
    }

    *dst = '\0';
}

#if defined(SDL_VIDEO_VITA_PVR)
SceWChar16 libime_out[SCE_IME_MAX_PREEDIT_LENGTH + SCE_IME_MAX_TEXT_LENGTH + 1];
char libime_initval[8] = { 1 };
SceImeCaret caret_rev;

void VITA_ImeEventHandler(void *arg, const SceImeEventData *e)
{
    SDL_VideoData *videodata = (SDL_VideoData *)arg;
    SDL_Scancode scancode;
    uint8_t utf8_buffer[SCE_IME_MAX_TEXT_LENGTH];
    switch (e->id) {
    case SCE_IME_EVENT_UPDATE_TEXT:
        if (e->param.text.caretIndex == 0) {
            SDL_SendKeyboardKeyAutoRelease(SDL_SCANCODE_BACKSPACE);
            sceImeSetText((SceWChar16 *)libime_initval, 4);
        } else {
            scancode = SDL_GetScancodeFromKey(*(SceWChar16 *)&libime_out[1]);
            if (scancode == SDL_SCANCODE_SPACE) {
                SDL_SendKeyboardKeyAutoRelease(SDL_SCANCODE_SPACE);
            } else {
                utf16_to_utf8((SceWChar16 *)&libime_out[1], utf8_buffer);
                SDL_SendKeyboardText((const char *)utf8_buffer);
            }
            SDL_memset(&caret_rev, 0, sizeof(SceImeCaret));
            SDL_memset(libime_out, 0, ((SCE_IME_MAX_PREEDIT_LENGTH + SCE_IME_MAX_TEXT_LENGTH + 1) * sizeof(SceWChar16)));
            caret_rev.index = 1;
            sceImeSetCaret(&caret_rev);
            sceImeSetText((SceWChar16 *)libime_initval, 4);
        }
        break;
    case SCE_IME_EVENT_PRESS_ENTER:
        SDL_SendKeyboardKeyAutoRelease(SDL_SCANCODE_RETURN);
    case SCE_IME_EVENT_PRESS_CLOSE:
        sceImeClose();
        videodata->ime_active = SDL_FALSE;
        break;
    }
}
#endif

void VITA_ShowScreenKeyboard(_THIS, SDL_Window *window)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
    SceInt32 res;

#if defined(SDL_VIDEO_VITA_PVR)

    SceUInt32 libime_work[SCE_IME_WORK_BUFFER_SIZE / sizeof(SceInt32)];
    SceImeParam param;

    sceImeParamInit(&param);

    SDL_memset(libime_out, 0, ((SCE_IME_MAX_PREEDIT_LENGTH + SCE_IME_MAX_TEXT_LENGTH + 1) * sizeof(SceWChar16)));

    param.supportedLanguages = SCE_IME_LANGUAGE_ENGLISH_US;
    param.languagesForced = SCE_FALSE;
    param.type = SCE_IME_TYPE_DEFAULT;
    param.option = SCE_IME_OPTION_NO_ASSISTANCE;
    param.inputTextBuffer = libime_out;
    param.maxTextLength = SCE_IME_MAX_TEXT_LENGTH;
    param.handler = VITA_ImeEventHandler;
    param.filter = NULL;
    param.initialText = (SceWChar16 *)libime_initval;
    param.arg = videodata;
    param.work = libime_work;

    res = sceImeOpen(&param);
    if (res < 0) {
        SDL_SetError("Failed to init IME");
        return;
    }

#else
    SceWChar16 *title = u"";
    SceWChar16 *text = u"";

    SceImeDialogParam param;
    sceImeDialogParamInit(&param);

    param.supportedLanguages = 0;
    param.languagesForced = SCE_FALSE;
    param.type = SCE_IME_TYPE_DEFAULT;
    param.option = 0;
    param.textBoxMode = SCE_IME_DIALOG_TEXTBOX_MODE_WITH_CLEAR;
    param.maxTextLength = SCE_IME_DIALOG_MAX_TEXT_LENGTH;

    param.title = title;
    param.initialText = text;
    param.inputTextBuffer = videodata->ime_buffer;

    res = sceImeDialogInit(&param);
    if (res < 0) {
        SDL_SetError("Failed to init IME dialog");
        return;
    }

#endif

    videodata->ime_active = SDL_TRUE;
}

void VITA_HideScreenKeyboard(_THIS, SDL_Window *window)
{
#if !defined(SDL_VIDEO_VITA_PVR)
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;

    SceCommonDialogStatus dialogStatus = sceImeDialogGetStatus();

    switch (dialogStatus) {
    default:
    case SCE_COMMON_DIALOG_STATUS_NONE:
    case SCE_COMMON_DIALOG_STATUS_RUNNING:
        break;
    case SCE_COMMON_DIALOG_STATUS_FINISHED:
        sceImeDialogTerm();
        break;
    }

    videodata->ime_active = SDL_FALSE;
#endif
}

SDL_bool VITA_IsScreenKeyboardShown(_THIS, SDL_Window *window)
{
#if defined(SDL_VIDEO_VITA_PVR)
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
    return videodata->ime_active;
#else
    SceCommonDialogStatus dialogStatus = sceImeDialogGetStatus();
    return dialogStatus == SCE_COMMON_DIALOG_STATUS_RUNNING;
#endif
}

void VITA_PumpEvents(_THIS)
{
#if !defined(SDL_VIDEO_VITA_PVR)
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
#endif

    if (_this->suspend_screensaver) {
        // cancel all idle timers to prevent vita going to sleep
        sceKernelPowerTick(SCE_KERNEL_POWER_TICK_DEFAULT);
    }

    VITA_PollTouch();
    VITA_PollKeyboard();
    VITA_PollMouse();

#if !defined(SDL_VIDEO_VITA_PVR)
    if (videodata->ime_active == SDL_TRUE) {
        // update IME status. Terminate, if finished
        SceCommonDialogStatus dialogStatus = sceImeDialogGetStatus();
        if (dialogStatus == SCE_COMMON_DIALOG_STATUS_FINISHED) {
            uint8_t utf8_buffer[SCE_IME_DIALOG_MAX_TEXT_LENGTH];

            SceImeDialogResult result;
            SDL_memset(&result, 0, sizeof(SceImeDialogResult));
            sceImeDialogGetResult(&result);

            // Convert UTF16 to UTF8
            utf16_to_utf8(videodata->ime_buffer, utf8_buffer);

            // Send SDL event
            SDL_SendKeyboardText((const char *)utf8_buffer);

            // Send enter key only on enter
            if (result.button == SCE_IME_DIALOG_BUTTON_ENTER) {
                SDL_SendKeyboardKeyAutoRelease(SDL_SCANCODE_RETURN);
            }

            sceImeDialogTerm();

            videodata->ime_active = SDL_FALSE;
        }
    }
#endif
}

#endif /* SDL_VIDEO_DRIVER_VITA */

/* vi: set ts=4 sw=4 expandtab: */
