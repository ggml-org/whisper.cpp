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

#ifndef SDL_x11opengl_h_
#define SDL_x11opengl_h_

#ifdef SDL_VIDEO_OPENGL_GLX
#include "SDL_opengl.h"
#include <GL/glx.h>

typedef enum SDL_GLSwapIntervalTearBehavior
{
    SDL_SWAPINTERVALTEAR_UNTESTED,
    SDL_SWAPINTERVALTEAR_UNKNOWN,
    SDL_SWAPINTERVALTEAR_MESA,
    SDL_SWAPINTERVALTEAR_NVIDIA
} SDL_GLSwapIntervalTearBehavior;

struct SDL_GLDriverData
{
    int errorBase, eventBase;

    SDL_bool HAS_GLX_EXT_visual_rating;
    SDL_bool HAS_GLX_EXT_visual_info;
    SDL_bool HAS_GLX_EXT_swap_control_tear;
    SDL_bool HAS_GLX_ARB_context_flush_control;
    SDL_bool HAS_GLX_ARB_create_context_robustness;
    SDL_bool HAS_GLX_ARB_create_context_no_error;

    /* Max version of OpenGL ES context that can be created if the
       implementation supports GLX_EXT_create_context_es2_profile.
       major = minor = 0 when unsupported.
     */
    struct
    {
        int major;
        int minor;
    } es_profile_max_supported_version;

    SDL_GLSwapIntervalTearBehavior swap_interval_tear_behavior;

    Bool (*glXQueryExtension)(Display *, int *, int *);
    void *(*glXGetProcAddress)(const GLubyte *);
    XVisualInfo *(*glXChooseVisual)(Display *, int, int *);
    GLXContext (*glXCreateContext)(Display *, XVisualInfo *, GLXContext, Bool);
    GLXContext (*glXCreateContextAttribsARB)(Display *, GLXFBConfig, GLXContext, Bool, const int *);
    GLXFBConfig *(*glXChooseFBConfig)(Display *, int, const int *, int *);
    XVisualInfo *(*glXGetVisualFromFBConfig)(Display *, GLXFBConfig);
    void (*glXDestroyContext)(Display *, GLXContext);
    Bool (*glXMakeCurrent)(Display *, GLXDrawable, GLXContext);
    void (*glXSwapBuffers)(Display *, GLXDrawable);
    void (*glXQueryDrawable)(Display *, GLXDrawable, int, unsigned int *);
    void (*glXSwapIntervalEXT)(Display *, GLXDrawable, int);
    int (*glXSwapIntervalSGI)(int);
    int (*glXSwapIntervalMESA)(int);
    int (*glXGetSwapIntervalMESA)(void);
};

/* OpenGL functions */
extern int X11_GL_LoadLibrary(_THIS, const char *path);
extern void *X11_GL_GetProcAddress(_THIS, const char *proc);
extern void X11_GL_UnloadLibrary(_THIS);
extern SDL_bool X11_GL_UseEGL(_THIS);
extern XVisualInfo *X11_GL_GetVisual(_THIS, Display *display, int screen);
extern SDL_GLContext X11_GL_CreateContext(_THIS, SDL_Window *window);
extern int X11_GL_MakeCurrent(_THIS, SDL_Window *window,
                              SDL_GLContext context);
extern int X11_GL_SetSwapInterval(_THIS, int interval);
extern int X11_GL_GetSwapInterval(_THIS);
extern int X11_GL_SwapWindow(_THIS, SDL_Window *window);
extern void X11_GL_DeleteContext(_THIS, SDL_GLContext context);

#endif /* SDL_VIDEO_OPENGL_GLX */

#endif /* SDL_x11opengl_h_ */

/* vi: set ts=4 sw=4 expandtab: */
