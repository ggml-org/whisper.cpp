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

#ifndef SDL_BOPENGL_H
#define SDL_BOPENGL_H

#if defined(SDL_VIDEO_DRIVER_HAIKU) && defined(SDL_VIDEO_OPENGL)

#ifdef __cplusplus
extern "C" {
#endif

#include "../SDL_sysvideo.h"

extern int HAIKU_GL_LoadLibrary(_THIS, const char *path);      /* FIXME */
extern void *HAIKU_GL_GetProcAddress(_THIS, const char *proc); /* FIXME */
extern void HAIKU_GL_UnloadLibrary(_THIS);                     /* TODO */
extern int HAIKU_GL_MakeCurrent(_THIS, SDL_Window *window,
                                SDL_GLContext context);
extern int HAIKU_GL_SetSwapInterval(_THIS, int interval); /* TODO */
extern int HAIKU_GL_GetSwapInterval(_THIS);               /* TODO */
extern int HAIKU_GL_SwapWindow(_THIS, SDL_Window *window);
extern SDL_GLContext HAIKU_GL_CreateContext(_THIS, SDL_Window *window);
extern void HAIKU_GL_DeleteContext(_THIS, SDL_GLContext context);

extern void HAIKU_GL_RebootContexts(_THIS);

#ifdef __cplusplus
}
#endif

#endif /* SDL_VIDEO_DRIVER_HAIKU && SDL_VIDEO_OPENGL */

#endif

/* vi: set ts=4 sw=4 expandtab: */
