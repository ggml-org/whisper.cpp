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

#ifndef SDL_androidvideo_h_
#define SDL_androidvideo_h_

#include "SDL_mutex.h"
#include "SDL_rect.h"
#include "../SDL_sysvideo.h"

/* Called by the JNI layer when the screen changes size or format */
extern void Android_SetScreenResolution(int surfaceWidth, int surfaceHeight, int deviceWidth, int deviceHeight, float rate);
extern void Android_SetFormat(int format_wanted, int format_got);
extern void Android_SendResize(SDL_Window *window);

/* Private display data */

typedef struct SDL_VideoData
{
    SDL_Rect textRect;
    int isPaused;
    int isPausing;
    int pauseAudio;
} SDL_VideoData;

extern int Android_SurfaceWidth;
extern int Android_SurfaceHeight;
extern SDL_sem *Android_PauseSem, *Android_ResumeSem;
extern SDL_mutex *Android_ActivityMutex;

#endif /* SDL_androidvideo_h_ */

/* vi: set ts=4 sw=4 expandtab: */
