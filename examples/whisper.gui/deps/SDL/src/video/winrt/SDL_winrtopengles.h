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
#include "SDL_config.h"

#ifndef SDL_winrtopengles_h_
#define SDL_winrtopengles_h_

#if defined(SDL_VIDEO_DRIVER_WINRT) && defined(SDL_VIDEO_OPENGL_EGL)

#include "../SDL_sysvideo.h"
#include "../SDL_egl_c.h"

/* OpenGLES functions */
#define WINRT_GLES_GetAttribute SDL_EGL_GetAttribute
#define WINRT_GLES_GetProcAddress SDL_EGL_GetProcAddress
#define WINRT_GLES_SetSwapInterval SDL_EGL_SetSwapInterval
#define WINRT_GLES_GetSwapInterval SDL_EGL_GetSwapInterval
#define WINRT_GLES_DeleteContext   SDL_EGL_DeleteContext

extern int WINRT_GLES_LoadLibrary(_THIS, const char *path);
extern void WINRT_GLES_UnloadLibrary(_THIS);
extern SDL_GLContext WINRT_GLES_CreateContext(_THIS, SDL_Window *window);
extern int WINRT_GLES_SwapWindow(_THIS, SDL_Window *window);
extern int WINRT_GLES_MakeCurrent(_THIS, SDL_Window *window, SDL_GLContext context);

#ifdef __cplusplus

/* Typedefs for ANGLE/WinRT's C++-based native-display and native-window types,
 * which are used when calling eglGetDisplay and eglCreateWindowSurface.
 */
typedef Microsoft::WRL::ComPtr<IUnknown> WINRT_EGLNativeWindowType_Old;

/* Function pointer typedefs for 'old' ANGLE/WinRT's functions, which may
 * require that C++ objects be passed in:
 */
typedef EGLDisplay(EGLAPIENTRY *eglGetDisplay_Old_Function)(WINRT_EGLNativeWindowType_Old);
typedef EGLSurface(EGLAPIENTRY *eglCreateWindowSurface_Old_Function)(EGLDisplay, EGLConfig, WINRT_EGLNativeWindowType_Old, const EGLint *);
typedef HRESULT(EGLAPIENTRY *CreateWinrtEglWindow_Old_Function)(Microsoft::WRL::ComPtr<IUnknown>, int, IUnknown **result);

#endif /* __cplusplus */

/* Function pointer typedefs for 'new' ANGLE/WinRT functions, which, unlike
 * the old functions, do not require C++ support and work with plain C.
 */
typedef EGLDisplay(EGLAPIENTRY *eglGetPlatformDisplayEXT_Function)(EGLenum, void *, const EGLint *);

#endif /* SDL_VIDEO_DRIVER_WINRT && SDL_VIDEO_OPENGL_EGL */

#endif /* SDL_winrtopengles_h_ */

/* vi: set ts=4 sw=4 expandtab: */
