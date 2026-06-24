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
#include "SDL_system.h"

/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
/* *INDENT-OFF* */
extern "C" {
/* *INDENT-ON* */
#endif

#include <EGL/eglplatform.h>
#include <android/native_window_jni.h>

#include "SDL_audio.h"
#include "SDL_rect.h"
#include "SDL_video.h"

/* Interface from the SDL library into the Android Java activity */
extern void Android_JNI_SetActivityTitle(const char *title);
extern void Android_JNI_SetWindowStyle(SDL_bool fullscreen);
extern void Android_JNI_SetOrientation(int w, int h, int resizable, const char *hint);
extern void Android_JNI_MinizeWindow(void);
extern SDL_bool Android_JNI_ShouldMinimizeOnFocusLoss(void);

extern SDL_bool Android_JNI_GetAccelerometerValues(float values[3]);
extern void Android_JNI_ShowScreenKeyboard(SDL_Rect *inputRect);
extern void Android_JNI_HideScreenKeyboard(void);
extern SDL_bool Android_JNI_IsScreenKeyboardShown(void);
extern ANativeWindow *Android_JNI_GetNativeWindow(void);

extern SDL_DisplayOrientation Android_JNI_GetDisplayOrientation(void);
extern int Android_JNI_GetDisplayDPI(float *ddpi, float *xdpi, float *ydpi);

/* Audio support */
extern void Android_DetectDevices(void);
extern int Android_JNI_OpenAudioDevice(int iscapture, int device_id, SDL_AudioSpec *spec);
extern void *Android_JNI_GetAudioBuffer(void);
extern void Android_JNI_WriteAudioBuffer(void);
extern int Android_JNI_CaptureAudioBuffer(void *buffer, int buflen);
extern void Android_JNI_FlushCapturedAudio(void);
extern void Android_JNI_CloseAudioDevice(const int iscapture);
extern void Android_JNI_AudioSetThreadPriority(int iscapture, int device_id);

/* Detecting device type */
extern SDL_bool Android_IsDeXMode(void);
extern SDL_bool Android_IsChromebook(void);

#include "SDL_rwops.h"

int Android_JNI_FileOpen(SDL_RWops* ctx, const char* fileName, const char* mode);
Sint64 Android_JNI_FileSize(SDL_RWops* ctx);
Sint64 Android_JNI_FileSeek(SDL_RWops* ctx, Sint64 offset, int whence);
size_t Android_JNI_FileRead(SDL_RWops* ctx, void* buffer, size_t size, size_t maxnum);
size_t Android_JNI_FileWrite(SDL_RWops* ctx, const void* buffer, size_t size, size_t num);
int Android_JNI_FileClose(SDL_RWops* ctx);

/* Environment support */
void Android_JNI_GetManifestEnvironmentVariables(void);

/* Clipboard support */
int Android_JNI_SetClipboardText(const char *text);
char *Android_JNI_GetClipboardText(void);
SDL_bool Android_JNI_HasClipboardText(void);

/* Power support */
int Android_JNI_GetPowerInfo(int *plugged, int *charged, int *battery, int *seconds, int *percent);

/* Joystick support */
void Android_JNI_PollInputDevices(void);

/* Haptic support */
void Android_JNI_PollHapticDevices(void);
void Android_JNI_HapticRun(int device_id, float intensity, int length);
void Android_JNI_HapticStop(int device_id);

/* Video */
void Android_JNI_SuspendScreenSaver(SDL_bool suspend);

/* Touch support */
void Android_JNI_InitTouch(void);

/* Threads */
#include <jni.h>
JNIEnv *Android_JNI_GetEnv(void);
int Android_JNI_SetupThread(void);

/* Locale */
int Android_JNI_GetLocale(char *buf, size_t buflen);

/* Generic messages */
int Android_JNI_SendMessage(int command, int param);

/* Init */
JNIEXPORT void JNICALL SDL_Android_Init(JNIEnv *mEnv, jclass cls);

/* MessageBox */
#include "SDL_messagebox.h"
int Android_JNI_ShowMessageBox(const SDL_MessageBoxData *messageboxdata, int *buttonid);

/* Cursor support */
int Android_JNI_CreateCustomCursor(SDL_Surface *surface, int hot_x, int hot_y);
void Android_JNI_DestroyCustomCursor(int cursorID);
SDL_bool Android_JNI_SetCustomCursor(int cursorID);
SDL_bool Android_JNI_SetSystemCursor(int cursorID);

/* Relative mouse support */
SDL_bool Android_JNI_SupportsRelativeMouse(void);
SDL_bool Android_JNI_SetRelativeMouseEnabled(SDL_bool enabled);

/* Request permission */
SDL_bool Android_JNI_RequestPermission(const char *permission);

/* Show toast notification */
int Android_JNI_ShowToast(const char *message, int duration, int gravity, int xOffset, int yOffset);

int Android_JNI_OpenURL(const char *url);

int SDL_GetAndroidSDKVersion(void);

SDL_bool SDL_IsAndroidTablet(void);
SDL_bool SDL_IsAndroidTV(void);
SDL_bool SDL_IsChromebook(void);
SDL_bool SDL_IsDeXMode(void);

void Android_ActivityMutex_Lock(void);
void Android_ActivityMutex_Unlock(void);
void Android_ActivityMutex_Lock_Running(void);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
/* *INDENT-OFF* */
}
/* *INDENT-ON* */
#endif

/* vi: set ts=4 sw=4 expandtab: */
