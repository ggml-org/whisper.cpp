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

#ifndef SDL_naclvideo_h_
#define SDL_naclvideo_h_

#include "../SDL_sysvideo.h"
#include "ppapi_simple/ps_interface.h"
#include "ppapi/c/pp_input_event.h"


/* Hidden "this" pointer for the video functions */
#define _THIS  SDL_VideoDevice *_this


/* Private display data */

typedef struct SDL_VideoData {
  Uint32 format;
  int w, h;
  SDL_Window *window;

  const PPB_Graphics3D *ppb_graphics;
  const PPB_MessageLoop *ppb_message_loop;
  const PPB_Core *ppb_core;
  const PPB_Fullscreen *ppb_fullscreen;
  const PPB_Instance *ppb_instance;
  const PPB_ImageData *ppb_image_data;
  const PPB_View *ppb_view;
  const PPB_Var *ppb_var;
  const PPB_InputEvent *ppb_input_event;
  const PPB_KeyboardInputEvent *ppb_keyboard_input_event;
  const PPB_MouseInputEvent *ppb_mouse_input_event;
  const PPB_WheelInputEvent *ppb_wheel_input_event;
  const PPB_TouchInputEvent *ppb_touch_input_event;
      
  PP_Resource message_loop;
  PP_Instance instance;
  
  /* FIXME: Check threading issues...otherwise use a hardcoded _this->context across all threads */
  /* PP_Resource context; */

} SDL_VideoData;

extern void NACL_SetScreenResolution(int width, int height, Uint32 format);


#endif /* SDL_naclvideo_h_ */
