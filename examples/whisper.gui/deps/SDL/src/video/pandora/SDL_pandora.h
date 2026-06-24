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

#ifndef __SDL_PANDORA_H__
#define __SDL_PANDORA_H__

#include <GLES/egl.h>

#include "../../SDL_internal.h"
#include "../SDL_sysvideo.h"

typedef struct SDL_VideoData
{
    SDL_bool egl_initialized;   /* OpenGL ES device initialization status */
    EGLDisplay egl_display;     /* OpenGL ES display connection           */
    uint32_t egl_refcount;      /* OpenGL ES reference count              */
    uint32_t swapinterval;      /* OpenGL ES default swap interval        */

} SDL_VideoData;


typedef struct SDL_DisplayData
{

} SDL_DisplayData;


typedef struct SDL_WindowData
{
    SDL_bool uses_gles;         /* if true window must support OpenGL ES */

    EGLConfig gles_configs[32];
    EGLint gles_config;         /* OpenGL ES configuration index      */
    EGLContext gles_context;    /* OpenGL ES context                  */
    EGLint gles_attributes[256];        /* OpenGL ES attributes for context   */
    EGLSurface gles_surface;    /* OpenGL ES target rendering surface */

} SDL_WindowData;


/****************************************************************************/
/* SDL_VideoDevice functions declaration                                    */
/****************************************************************************/

/* Display and window functions */
int PND_videoinit(_THIS);
void PND_videoquit(_THIS);
void PND_getdisplaymodes(_THIS, SDL_VideoDisplay * display);
int PND_setdisplaymode(_THIS, SDL_VideoDisplay * display, SDL_DisplayMode * mode);
int PND_createwindow(_THIS, SDL_Window * window);
int PND_createwindowfrom(_THIS, SDL_Window * window, const void *data);
void PND_setwindowtitle(_THIS, SDL_Window * window);
void PND_setwindowicon(_THIS, SDL_Window * window, SDL_Surface * icon);
void PND_setwindowposition(_THIS, SDL_Window * window);
void PND_setwindowsize(_THIS, SDL_Window * window);
void PND_showwindow(_THIS, SDL_Window * window);
void PND_hidewindow(_THIS, SDL_Window * window);
void PND_raisewindow(_THIS, SDL_Window * window);
void PND_maximizewindow(_THIS, SDL_Window * window);
void PND_minimizewindow(_THIS, SDL_Window * window);
void PND_restorewindow(_THIS, SDL_Window * window);
void PND_destroywindow(_THIS, SDL_Window * window);

/* Window manager function */
SDL_bool PND_getwindowwminfo(_THIS, SDL_Window * window,
                             struct SDL_SysWMinfo *info);

/* OpenGL/OpenGL ES functions */
int PND_gl_loadlibrary(_THIS, const char *path);
void *PND_gl_getprocaddres(_THIS, const char *proc);
void PND_gl_unloadlibrary(_THIS);
SDL_GLContext PND_gl_createcontext(_THIS, SDL_Window * window);
int PND_gl_makecurrent(_THIS, SDL_Window * window, SDL_GLContext context);
int PND_gl_setswapinterval(_THIS, int interval);
int PND_gl_getswapinterval(_THIS);
int PND_gl_swapwindow(_THIS, SDL_Window * window);
void PND_gl_deletecontext(_THIS, SDL_GLContext context);


#endif /* __SDL_PANDORA_H__ */

/* vi: set ts=4 sw=4 expandtab: */
