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
#ifndef SDL_uikitopengles_
#define SDL_uikitopengles_

#if defined(SDL_VIDEO_OPENGL_ES) || defined(SDL_VIDEO_OPENGL_ES2)

#include "../SDL_sysvideo.h"

extern int UIKit_GL_MakeCurrent(_THIS, SDL_Window * window,
                                SDL_GLContext context);
extern void UIKit_GL_GetDrawableSize(_THIS, SDL_Window * window,
                                     int * w, int * h);
extern int UIKit_GL_SwapWindow(_THIS, SDL_Window * window);
extern SDL_GLContext UIKit_GL_CreateContext(_THIS, SDL_Window * window);
extern void UIKit_GL_DeleteContext(_THIS, SDL_GLContext context);
extern void *UIKit_GL_GetProcAddress(_THIS, const char *proc);
extern int UIKit_GL_LoadLibrary(_THIS, const char *path);

extern void UIKit_GL_RestoreCurrentContext(void);

#endif // SDL_VIDEO_OPENGL_ES || defined(SDL_VIDEO_OPENGL_ES2)

#endif /* SDL_uikitopengles_ */

/* vi: set ts=4 sw=4 expandtab: */
