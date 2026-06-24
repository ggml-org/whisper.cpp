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

#ifndef SDL_pipewire_h_
#define SDL_pipewire_h_

#include "../SDL_sysaudio.h"
#include <pipewire/pipewire.h>

/* Hidden "this" pointer for the audio functions */
#define _THIS SDL_AudioDevice *this

struct SDL_PrivateAudioData
{
    struct pw_thread_loop *loop;
    struct pw_stream *stream;
    struct pw_context *context;
    struct SDL_DataQueue *buffer;

    size_t input_buffer_packet_size;
    Sint32 stride; /* Bytes-per-frame */
    int stream_init_status;
};

#endif /* SDL_pipewire_h_ */

/* vi: set ts=4 sw=4 expandtab: */
