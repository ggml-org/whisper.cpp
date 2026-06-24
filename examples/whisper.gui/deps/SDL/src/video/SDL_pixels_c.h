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

#ifndef SDL_pixels_c_h_
#define SDL_pixels_c_h_

#include "../SDL_internal.h"

/* Useful functions and variables from SDL_pixel.c */

#include "SDL_blit.h"

/* Pixel format functions */
extern int SDL_InitFormat(SDL_PixelFormat *format, Uint32 pixel_format);

/* Blit mapping functions */
extern SDL_BlitMap *SDL_AllocBlitMap(void);
extern void SDL_InvalidateMap(SDL_BlitMap *map);
extern int SDL_MapSurface(SDL_Surface *src, SDL_Surface *dst);
extern void SDL_FreeBlitMap(SDL_BlitMap *map);

extern void SDL_InvalidateAllBlitMap(SDL_Surface *surface);

/* Miscellaneous functions */
extern void SDL_DitherColors(SDL_Color *colors, int bpp);
extern Uint8 SDL_FindColor(SDL_Palette *pal, Uint8 r, Uint8 g, Uint8 b, Uint8 a);
extern void SDL_DetectPalette(SDL_Palette *pal, SDL_bool *is_opaque, SDL_bool *has_alpha_channel);

#endif /* SDL_pixels_c_h_ */

/* vi: set ts=4 sw=4 expandtab: */
