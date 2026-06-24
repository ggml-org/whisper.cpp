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

#ifndef math_libm_h_
#define math_libm_h_

#include "../SDL_internal.h"

/* Math routines from uClibc: http://www.uclibc.org */

double SDL_uclibc_atan(double x);
double SDL_uclibc_atan2(double y, double x);    
double SDL_uclibc_copysign(double x, double y);       
double SDL_uclibc_cos(double x);         
double SDL_uclibc_exp(double x);
double SDL_uclibc_fabs(double x);        
double SDL_uclibc_floor(double x);
double SDL_uclibc_fmod(double x, double y);
double SDL_uclibc_log(double x);
double SDL_uclibc_log10(double x);
double SDL_uclibc_pow(double x, double y);    
double SDL_uclibc_scalbn(double x, int n);
double SDL_uclibc_sin(double x);
double SDL_uclibc_sqrt(double x);
double SDL_uclibc_tan(double x);

#endif /* math_libm_h_ */

/* vi: set ts=4 sw=4 expandtab: */
