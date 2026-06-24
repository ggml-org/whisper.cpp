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

#ifdef SDL_VIDEO_DRIVER_ANDROID

#include "SDL_androidevents.h"
#include "SDL_events.h"
#include "SDL_hints.h"
#include "SDL_androidkeyboard.h"
#include "SDL_androidwindow.h"
#include "../SDL_sysvideo.h"
#include "../../events/SDL_events_c.h"

/* Can't include sysaudio "../../audio/android/SDL_androidaudio.h"
 * because of THIS redefinition */

#if !defined(SDL_AUDIO_DISABLED) && defined(SDL_AUDIO_DRIVER_ANDROID)
extern void ANDROIDAUDIO_ResumeDevices(void);
extern void ANDROIDAUDIO_PauseDevices(void);
#else
static void ANDROIDAUDIO_ResumeDevices(void) {}
static void ANDROIDAUDIO_PauseDevices(void) {}
#endif

#if !defined(SDL_AUDIO_DISABLED) && defined(SDL_AUDIO_DRIVER_OPENSLES)
extern void openslES_ResumeDevices(void);
extern void openslES_PauseDevices(void);
#else
static void openslES_ResumeDevices(void)
{
}
static void openslES_PauseDevices(void) {}
#endif

#if !defined(SDL_AUDIO_DISABLED) && defined(SDL_AUDIO_DRIVER_AAUDIO)
extern void aaudio_ResumeDevices(void);
extern void aaudio_PauseDevices(void);
SDL_bool aaudio_DetectBrokenPlayState(void);
#else
static void aaudio_ResumeDevices(void)
{
}
static void aaudio_PauseDevices(void) {}
static SDL_bool aaudio_DetectBrokenPlayState(void) { return SDL_FALSE; }
#endif

/* Number of 'type' events in the event queue */
static int SDL_NumberOfEvents(Uint32 type)
{
    return SDL_PeepEvents(NULL, 0, SDL_PEEKEVENT, type, type);
}

#ifdef SDL_VIDEO_OPENGL_EGL
static void android_egl_context_restore(SDL_Window *window)
{
    if (window) {
        SDL_Event event;
        SDL_WindowData *data = (SDL_WindowData *)window->driverdata;
        SDL_GL_MakeCurrent(window, NULL);
        if (SDL_GL_MakeCurrent(window, (SDL_GLContext)data->egl_context) < 0) {
            /* The context is no longer valid, create a new one */
            data->egl_context = (EGLContext)SDL_GL_CreateContext(window);
            SDL_GL_MakeCurrent(window, (SDL_GLContext)data->egl_context);
            event.type = SDL_RENDER_DEVICE_RESET;
            SDL_PushEvent(&event);
        }
        data->backup_done = 0;
    }
}

static void android_egl_context_backup(SDL_Window *window)
{
    if (window) {
        /* Keep a copy of the EGL Context so we can try to restore it when we resume */
        SDL_WindowData *data = (SDL_WindowData *)window->driverdata;
        data->egl_context = SDL_GL_GetCurrentContext();
        /* We need to do this so the EGLSurface can be freed */
        SDL_GL_MakeCurrent(window, NULL);
        data->backup_done = 1;
    }
}
#endif

/*
 * Android_ResumeSem and Android_PauseSem are signaled from Java_org_libsdl_app_SDLActivity_nativePause and Java_org_libsdl_app_SDLActivity_nativeResume
 * When the pause semaphore is signaled, if Android_PumpEvents_Blocking is used, the event loop will block until the resume signal is emitted.
 *
 * No polling necessary
 */

void Android_PumpEvents_Blocking(_THIS)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;

    if (videodata->isPaused) {
        SDL_bool isContextExternal = SDL_IsVideoContextExternal();

#ifdef SDL_VIDEO_OPENGL_EGL
        /* Make sure this is the last thing we do before pausing */
        if (!isContextExternal) {
            SDL_LockMutex(Android_ActivityMutex);
            android_egl_context_backup(Android_Window);
            SDL_UnlockMutex(Android_ActivityMutex);
        }
#endif

        ANDROIDAUDIO_PauseDevices();
        openslES_PauseDevices();
        aaudio_PauseDevices();

        if (SDL_SemWait(Android_ResumeSem) == 0) {

            videodata->isPaused = 0;

            /* Android_ResumeSem was signaled */
            SDL_SendAppEvent(SDL_APP_WILLENTERFOREGROUND);
            SDL_SendAppEvent(SDL_APP_DIDENTERFOREGROUND);
            SDL_SendWindowEvent(Android_Window, SDL_WINDOWEVENT_RESTORED, 0, 0);

            ANDROIDAUDIO_ResumeDevices();
            openslES_ResumeDevices();
            aaudio_ResumeDevices();

            /* Restore the GL Context from here, as this operation is thread dependent */
#ifdef SDL_VIDEO_OPENGL_EGL
            if (!isContextExternal && !SDL_HasEvent(SDL_QUIT)) {
                SDL_LockMutex(Android_ActivityMutex);
                android_egl_context_restore(Android_Window);
                SDL_UnlockMutex(Android_ActivityMutex);
            }
#endif

            /* Make sure SW Keyboard is restored when an app becomes foreground */
            Android_RestoreScreenKeyboardOnResume(_this, Android_Window);
        }
    } else {
        if (videodata->isPausing || SDL_SemTryWait(Android_PauseSem) == 0) {

            /* Android_PauseSem was signaled */
            if (videodata->isPausing == 0) {
                SDL_SendWindowEvent(Android_Window, SDL_WINDOWEVENT_MINIMIZED, 0, 0);
                SDL_SendAppEvent(SDL_APP_WILLENTERBACKGROUND);
                SDL_SendAppEvent(SDL_APP_DIDENTERBACKGROUND);
            }

            /* We've been signaled to pause (potentially several times), but before we block ourselves,
             * we need to make sure that the very last event (of the first pause sequence, if several)
             * has reached the app */
            if (SDL_NumberOfEvents(SDL_APP_DIDENTERBACKGROUND) > SDL_SemValue(Android_PauseSem)) {
                videodata->isPausing = 1;
            } else {
                videodata->isPausing = 0;
                videodata->isPaused = 1;
            }
        }
    }

    if (aaudio_DetectBrokenPlayState()) {
        aaudio_PauseDevices();
        aaudio_ResumeDevices();
    }
}

void Android_PumpEvents_NonBlocking(_THIS)
{
    SDL_VideoData *videodata = (SDL_VideoData *)_this->driverdata;
    static int backup_context = 0;

    if (videodata->isPaused) {

        SDL_bool isContextExternal = SDL_IsVideoContextExternal();
        if (backup_context) {

#ifdef SDL_VIDEO_OPENGL_EGL
            if (!isContextExternal) {
                SDL_LockMutex(Android_ActivityMutex);
                android_egl_context_backup(Android_Window);
                SDL_UnlockMutex(Android_ActivityMutex);
            }
#endif

            if (videodata->pauseAudio) {
                ANDROIDAUDIO_PauseDevices();
                openslES_PauseDevices();
                aaudio_PauseDevices();
            }

            backup_context = 0;
        }

        if (SDL_SemTryWait(Android_ResumeSem) == 0) {

            videodata->isPaused = 0;

            /* Android_ResumeSem was signaled */
            SDL_SendAppEvent(SDL_APP_WILLENTERFOREGROUND);
            SDL_SendAppEvent(SDL_APP_DIDENTERFOREGROUND);
            SDL_SendWindowEvent(Android_Window, SDL_WINDOWEVENT_RESTORED, 0, 0);

            if (videodata->pauseAudio) {
                ANDROIDAUDIO_ResumeDevices();
                openslES_ResumeDevices();
                aaudio_ResumeDevices();
            }

#ifdef SDL_VIDEO_OPENGL_EGL
            /* Restore the GL Context from here, as this operation is thread dependent */
            if (!isContextExternal && !SDL_HasEvent(SDL_QUIT)) {
                SDL_LockMutex(Android_ActivityMutex);
                android_egl_context_restore(Android_Window);
                SDL_UnlockMutex(Android_ActivityMutex);
            }
#endif

            /* Make sure SW Keyboard is restored when an app becomes foreground */
            Android_RestoreScreenKeyboardOnResume(_this, Android_Window);
        }
    } else {
        if (videodata->isPausing || SDL_SemTryWait(Android_PauseSem) == 0) {

            /* Android_PauseSem was signaled */
            if (videodata->isPausing == 0) {
                SDL_SendWindowEvent(Android_Window, SDL_WINDOWEVENT_MINIMIZED, 0, 0);
                SDL_SendAppEvent(SDL_APP_WILLENTERBACKGROUND);
                SDL_SendAppEvent(SDL_APP_DIDENTERBACKGROUND);
            }

            /* We've been signaled to pause (potentially several times), but before we block ourselves,
             * we need to make sure that the very last event (of the first pause sequence, if several)
             * has reached the app */
            if (SDL_NumberOfEvents(SDL_APP_DIDENTERBACKGROUND) > SDL_SemValue(Android_PauseSem)) {
                videodata->isPausing = 1;
            } else {
                videodata->isPausing = 0;
                videodata->isPaused = 1;
                backup_context = 1;
            }
        }
    }

    if (aaudio_DetectBrokenPlayState()) {
        aaudio_PauseDevices();
        aaudio_ResumeDevices();
    }
}

#endif /* SDL_VIDEO_DRIVER_ANDROID */

/* vi: set ts=4 sw=4 expandtab: */
