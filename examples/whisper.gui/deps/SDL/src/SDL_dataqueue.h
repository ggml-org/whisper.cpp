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
#ifndef SDL_dataqueue_h_
#define SDL_dataqueue_h_

/* this is not (currently) a public API. But maybe it should be! */

struct SDL_DataQueue;
typedef struct SDL_DataQueue SDL_DataQueue;

SDL_DataQueue *SDL_NewDataQueue(const size_t packetlen, const size_t initialslack);
void SDL_FreeDataQueue(SDL_DataQueue *queue);
void SDL_ClearDataQueue(SDL_DataQueue *queue, const size_t slack);
int SDL_WriteToDataQueue(SDL_DataQueue *queue, const void *data, const size_t len);
size_t SDL_ReadFromDataQueue(SDL_DataQueue *queue, void *buf, const size_t len);
size_t SDL_PeekIntoDataQueue(SDL_DataQueue *queue, void *buf, const size_t len);
size_t SDL_CountDataQueue(SDL_DataQueue *queue);
SDL_mutex *SDL_GetDataQueueMutex(SDL_DataQueue *queue);  /* don't destroy this, obviously. */

#endif /* SDL_dataqueue_h_ */

/* vi: set ts=4 sw=4 expandtab: */
