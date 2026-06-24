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


#ifndef SDL_directfb_opengl_h_
#define SDL_directfb_opengl_h_

#include "SDL_DirectFB_video.h"

#ifdef SDL_DIRECTFB_OPENGL

#include "SDL_opengl.h"

typedef struct _DirectFB_GLContext DirectFB_GLContext;
struct _DirectFB_GLContext
{
    IDirectFBGL         *context;
    DirectFB_GLContext  *next;

    SDL_Window          *sdl_window;
    int                 is_locked;
};

/* OpenGL functions */
extern int DirectFB_GL_Initialize(_THIS);
extern void DirectFB_GL_Shutdown(_THIS);

extern int DirectFB_GL_LoadLibrary(_THIS, const char *path);
extern void *DirectFB_GL_GetProcAddress(_THIS, const char *proc);
extern SDL_GLContext DirectFB_GL_CreateContext(_THIS, SDL_Window * window);
extern int DirectFB_GL_MakeCurrent(_THIS, SDL_Window * window,
                                   SDL_GLContext context);
extern int DirectFB_GL_SetSwapInterval(_THIS, int interval);
extern int DirectFB_GL_GetSwapInterval(_THIS);
extern int DirectFB_GL_SwapWindow(_THIS, SDL_Window * window);
extern void DirectFB_GL_DeleteContext(_THIS, SDL_GLContext context);

extern void DirectFB_GL_FreeWindowContexts(_THIS, SDL_Window * window);
extern void DirectFB_GL_ReAllocWindowContexts(_THIS, SDL_Window * window);
extern void DirectFB_GL_DestroyWindowContexts(_THIS, SDL_Window * window);

#endif /* SDL_DIRECTFB_OPENGL */

#endif /* SDL_directfb_opengl_h_ */

/* vi: set ts=4 sw=4 expandtab: */
