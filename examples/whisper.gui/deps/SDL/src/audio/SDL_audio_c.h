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

#ifndef SDL_audio_c_h_
#define SDL_audio_c_h_

#include "../SDL_internal.h"

#ifndef DEBUG_CONVERT
#define DEBUG_CONVERT 0
#endif

#if DEBUG_CONVERT
#define LOG_DEBUG_CONVERT(from, to) SDL_Log("SDL_AUDIO_CONVERT: Converting %s to %s.\n", from, to);
#else
#define LOG_DEBUG_CONVERT(from, to)
#endif

/* Functions and variables exported from SDL_audio.c for SDL_sysaudio.c */

#ifdef HAVE_LIBSAMPLERATE_H
#include "samplerate.h"
extern SDL_bool SRC_available;
extern int SRC_converter;
extern SRC_STATE *(*SRC_src_new)(int converter_type, int channels, int *error);
extern int (*SRC_src_process)(SRC_STATE *state, SRC_DATA *data);
extern int (*SRC_src_reset)(SRC_STATE *state);
extern SRC_STATE *(*SRC_src_delete)(SRC_STATE *state);
extern const char *(*SRC_src_strerror)(int error);
extern int (*SRC_src_simple)(SRC_DATA *data, int converter_type, int channels);
#endif

/* Functions to get a list of "close" audio formats */
extern SDL_AudioFormat SDL_FirstAudioFormat(SDL_AudioFormat format);
extern SDL_AudioFormat SDL_NextAudioFormat(void);

/* Function to calculate the size and silence for a SDL_AudioSpec */
extern Uint8 SDL_SilenceValueForFormat(const SDL_AudioFormat format);
extern void SDL_CalculateAudioSpec(SDL_AudioSpec *spec);

/* Choose the audio filter functions below */
extern void SDL_ChooseAudioConverters(void);

/* These pointers get set during SDL_ChooseAudioConverters() to various SIMD implementations. */
extern SDL_AudioFilter SDL_Convert_S8_to_F32;
extern SDL_AudioFilter SDL_Convert_U8_to_F32;
extern SDL_AudioFilter SDL_Convert_S16_to_F32;
extern SDL_AudioFilter SDL_Convert_U16_to_F32;
extern SDL_AudioFilter SDL_Convert_S32_to_F32;
extern SDL_AudioFilter SDL_Convert_F32_to_S8;
extern SDL_AudioFilter SDL_Convert_F32_to_U8;
extern SDL_AudioFilter SDL_Convert_F32_to_S16;
extern SDL_AudioFilter SDL_Convert_F32_to_U16;
extern SDL_AudioFilter SDL_Convert_F32_to_S32;

#endif /* SDL_audio_c_h_ */

/* vi: set ts=4 sw=4 expandtab: */
