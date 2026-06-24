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

#ifndef SDL_vivantevideo_h_
#define SDL_vivantevideo_h_

#include "../../SDL_internal.h"
#include "../SDL_sysvideo.h"

#include "SDL_egl.h"

#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
#include <gc_vdk.h>
#else
#include <EGL/egl.h>
#endif

typedef struct SDL_VideoData
{
#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
    vdkPrivate vdk_private;
#else
    void *egl_handle; /* EGL shared library handle */
    EGLNativeDisplayType(EGLAPIENTRY *fbGetDisplay)(void *context);
    EGLNativeDisplayType(EGLAPIENTRY *fbGetDisplayByIndex)(int DisplayIndex);
    void(EGLAPIENTRY *fbGetDisplayGeometry)(EGLNativeDisplayType Display, int *Width, int *Height);
    void(EGLAPIENTRY *fbGetDisplayInfo)(EGLNativeDisplayType Display, int *Width, int *Height, unsigned long *Physical, int *Stride, int *BitsPerPixel);
    void(EGLAPIENTRY *fbDestroyDisplay)(EGLNativeDisplayType Display);
    EGLNativeWindowType(EGLAPIENTRY *fbCreateWindow)(EGLNativeDisplayType Display, int X, int Y, int Width, int Height);
    void(EGLAPIENTRY *fbGetWindowGeometry)(EGLNativeWindowType Window, int *X, int *Y, int *Width, int *Height);
    void(EGLAPIENTRY *fbGetWindowInfo)(EGLNativeWindowType Window, int *X, int *Y, int *Width, int *Height, int *BitsPerPixel, unsigned int *Offset);
    void(EGLAPIENTRY *fbDestroyWindow)(EGLNativeWindowType Window);
#endif
} SDL_VideoData;

typedef struct SDL_DisplayData
{
    EGLNativeDisplayType native_display;
} SDL_DisplayData;

typedef struct SDL_WindowData
{
    EGLNativeWindowType native_window;
    EGLSurface egl_surface;
} SDL_WindowData;

/****************************************************************************/
/* SDL_VideoDevice functions declaration                                    */
/****************************************************************************/

/* Display and window functions */
int VIVANTE_VideoInit(_THIS);
void VIVANTE_VideoQuit(_THIS);
void VIVANTE_GetDisplayModes(_THIS, SDL_VideoDisplay *display);
int VIVANTE_SetDisplayMode(_THIS, SDL_VideoDisplay *display, SDL_DisplayMode *mode);
int VIVANTE_CreateWindow(_THIS, SDL_Window *window);
void VIVANTE_SetWindowTitle(_THIS, SDL_Window *window);
void VIVANTE_SetWindowPosition(_THIS, SDL_Window *window);
void VIVANTE_SetWindowSize(_THIS, SDL_Window *window);
void VIVANTE_ShowWindow(_THIS, SDL_Window *window);
void VIVANTE_HideWindow(_THIS, SDL_Window *window);
void VIVANTE_DestroyWindow(_THIS, SDL_Window *window);

/* Window manager function */
SDL_bool VIVANTE_GetWindowWMInfo(_THIS, SDL_Window * window,
                             struct SDL_SysWMinfo *info);

/* Event functions */
void VIVANTE_PumpEvents(_THIS);

#endif /* SDL_vivantevideo_h_ */

/* vi: set ts=4 sw=4 expandtab: */
