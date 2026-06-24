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

#if defined(__clang_analyzer__) && !defined(SDL_DISABLE_ANALYZE_MACROS)
#define SDL_DISABLE_ANALYZE_MACROS 1
#endif

#include "../SDL_internal.h"

/* This file contains portable stdlib functions for SDL */

#include "SDL_stdinc.h"
#include "../libm/math_libm.h"

double
SDL_atan(double x)
{
#if defined(HAVE_ATAN)
    return atan(x);
#else
    return SDL_uclibc_atan(x);
#endif
}

float SDL_atanf(float x)
{
#if defined(HAVE_ATANF)
    return atanf(x);
#else
    return (float)SDL_atan((double)x);
#endif
}

double
SDL_atan2(double y, double x)
{
#if defined(HAVE_ATAN2)
    return atan2(y, x);
#else
    return SDL_uclibc_atan2(y, x);
#endif
}

float SDL_atan2f(float y, float x)
{
#if defined(HAVE_ATAN2F)
    return atan2f(y, x);
#else
    return (float)SDL_atan2((double)y, (double)x);
#endif
}

double
SDL_acos(double val)
{
#if defined(HAVE_ACOS)
    return acos(val);
#else
    double result;
    if (val == -1.0) {
        result = M_PI;
    } else {
        result = SDL_atan(SDL_sqrt(1.0 - val * val) / val);
        if (result < 0.0)
        {
            result += M_PI;
        }
    }
    return result;
#endif
}

float SDL_acosf(float val)
{
#if defined(HAVE_ACOSF)
    return acosf(val);
#else
    return (float)SDL_acos((double)val);
#endif
}

double
SDL_asin(double val)
{
#if defined(HAVE_ASIN)
    return asin(val);
#else
    double result;
    if (val == -1.0) {
        result = -(M_PI / 2.0);
    } else {
        result = (M_PI / 2.0) - SDL_acos(val);
    }
    return result;
#endif
}

float SDL_asinf(float val)
{
#if defined(HAVE_ASINF)
    return asinf(val);
#else
    return (float)SDL_asin((double)val);
#endif
}

double
SDL_ceil(double x)
{
#if defined(HAVE_CEIL)
    return ceil(x);
#else
    double integer = SDL_floor(x);
    double fraction = x - integer;
    if (fraction > 0.0) {
        integer += 1.0;
    }
    return integer;
#endif /* HAVE_CEIL */
}

float SDL_ceilf(float x)
{
#if defined(HAVE_CEILF)
    return ceilf(x);
#else
    return (float)SDL_ceil((double)x);
#endif
}

double
SDL_copysign(double x, double y)
{
#if defined(HAVE_COPYSIGN)
    return copysign(x, y);
#elif defined(HAVE__COPYSIGN)
    return _copysign(x, y);
#elif defined(__WATCOMC__) && defined(__386__)
    /* this is nasty as hell, but it works.. */
    unsigned int *xi = (unsigned int *)&x,
                 *yi = (unsigned int *)&y;
    xi[1] = (yi[1] & 0x80000000) | (xi[1] & 0x7fffffff);
    return x;
#else
    return SDL_uclibc_copysign(x, y);
#endif /* HAVE_COPYSIGN */
}

float SDL_copysignf(float x, float y)
{
#if defined(HAVE_COPYSIGNF)
    return copysignf(x, y);
#else
    return (float)SDL_copysign((double)x, (double)y);
#endif
}

double
SDL_cos(double x)
{
#if defined(HAVE_COS)
    return cos(x);
#else
    return SDL_uclibc_cos(x);
#endif
}

float SDL_cosf(float x)
{
#if defined(HAVE_COSF)
    return cosf(x);
#else
    return (float)SDL_cos((double)x);
#endif
}

double
SDL_exp(double x)
{
#if defined(HAVE_EXP)
    return exp(x);
#else
    return SDL_uclibc_exp(x);
#endif
}

float SDL_expf(float x)
{
#if defined(HAVE_EXPF)
    return expf(x);
#else
    return (float)SDL_exp((double)x);
#endif
}

double
SDL_fabs(double x)
{
#if defined(HAVE_FABS)
    return fabs(x);
#else
    return SDL_uclibc_fabs(x);
#endif
}

float SDL_fabsf(float x)
{
#if defined(HAVE_FABSF)
    return fabsf(x);
#else
    return (float)SDL_fabs((double)x);
#endif
}

double
SDL_floor(double x)
{
#if defined(HAVE_FLOOR)
    return floor(x);
#else
    return SDL_uclibc_floor(x);
#endif
}

float SDL_floorf(float x)
{
#if defined(HAVE_FLOORF)
    return floorf(x);
#else
    return (float)SDL_floor((double)x);
#endif
}

double
SDL_trunc(double x)
{
#if defined(HAVE_TRUNC)
    return trunc(x);
#else
    if (x >= 0.0f) {
        return SDL_floor(x);
    } else {
        return SDL_ceil(x);
    }
#endif
}

float SDL_truncf(float x)
{
#if defined(HAVE_TRUNCF)
    return truncf(x);
#else
    return (float)SDL_trunc((double)x);
#endif
}

double
SDL_fmod(double x, double y)
{
#if defined(HAVE_FMOD)
    return fmod(x, y);
#else
    return SDL_uclibc_fmod(x, y);
#endif
}

float SDL_fmodf(float x, float y)
{
#if defined(HAVE_FMODF)
    return fmodf(x, y);
#else
    return (float)SDL_fmod((double)x, (double)y);
#endif
}

double
SDL_log(double x)
{
#if defined(HAVE_LOG)
    return log(x);
#else
    return SDL_uclibc_log(x);
#endif
}

float SDL_logf(float x)
{
#if defined(HAVE_LOGF)
    return logf(x);
#else
    return (float)SDL_log((double)x);
#endif
}

double
SDL_log10(double x)
{
#if defined(HAVE_LOG10)
    return log10(x);
#else
    return SDL_uclibc_log10(x);
#endif
}

float SDL_log10f(float x)
{
#if defined(HAVE_LOG10F)
    return log10f(x);
#else
    return (float)SDL_log10((double)x);
#endif
}

double
SDL_pow(double x, double y)
{
#if defined(HAVE_POW)
    return pow(x, y);
#else
    return SDL_uclibc_pow(x, y);
#endif
}

float SDL_powf(float x, float y)
{
#if defined(HAVE_POWF)
    return powf(x, y);
#else
    return (float)SDL_pow((double)x, (double)y);
#endif
}

double
SDL_round(double arg)
{
#if defined HAVE_ROUND
    return round(arg);
#else
    if (arg >= 0.0) {
        return SDL_floor(arg + 0.5);
    } else {
        return SDL_ceil(arg - 0.5);
    }
#endif
}

float SDL_roundf(float arg)
{
#if defined HAVE_ROUNDF
    return roundf(arg);
#else
    return (float)SDL_round((double)arg);
#endif
}

long SDL_lround(double arg)
{
#if defined HAVE_LROUND
    return lround(arg);
#else
    return (long)SDL_round(arg);
#endif
}

long SDL_lroundf(float arg)
{
#if defined HAVE_LROUNDF
    return lroundf(arg);
#else
    return (long)SDL_round((double)arg);
#endif
}

double
SDL_scalbn(double x, int n)
{
#if defined(HAVE_SCALBN)
    return scalbn(x, n);
#elif defined(HAVE__SCALB)
    return _scalb(x, n);
#elif defined(HAVE_LIBC) && defined(HAVE_FLOAT_H) && (FLT_RADIX == 2)
    /* from scalbn(3): If FLT_RADIX equals 2 (which is
     * usual), then scalbn() is equivalent to ldexp(3). */
    return ldexp(x, n);
#else
    return SDL_uclibc_scalbn(x, n);
#endif
}

float SDL_scalbnf(float x, int n)
{
#if defined(HAVE_SCALBNF)
    return scalbnf(x, n);
#else
    return (float)SDL_scalbn((double)x, n);
#endif
}

double
SDL_sin(double x)
{
#if defined(HAVE_SIN)
    return sin(x);
#else
    return SDL_uclibc_sin(x);
#endif
}

float SDL_sinf(float x)
{
#if defined(HAVE_SINF)
    return sinf(x);
#else
    return (float)SDL_sin((double)x);
#endif
}

double
SDL_sqrt(double x)
{
#if defined(HAVE_SQRT)
    return sqrt(x);
#else
    return SDL_uclibc_sqrt(x);
#endif
}

float SDL_sqrtf(float x)
{
#if defined(HAVE_SQRTF)
    return sqrtf(x);
#else
    return (float)SDL_sqrt((double)x);
#endif
}

double
SDL_tan(double x)
{
#if defined(HAVE_TAN)
    return tan(x);
#else
    return SDL_uclibc_tan(x);
#endif
}

float SDL_tanf(float x)
{
#if defined(HAVE_TANF)
    return tanf(x);
#else
    return (float)SDL_tan((double)x);
#endif
}

int SDL_abs(int x)
{
#if defined(HAVE_ABS)
    return abs(x);
#else
    return (x < 0) ? -x : x;
#endif
}

#if defined(HAVE_CTYPE_H)
int SDL_isalpha(int x)
{
    return isalpha(x);
}
int SDL_isalnum(int x) { return isalnum(x); }
int SDL_isdigit(int x) { return isdigit(x); }
int SDL_isxdigit(int x) { return isxdigit(x); }
int SDL_ispunct(int x) { return ispunct(x); }
int SDL_isspace(int x) { return isspace(x); }
int SDL_isupper(int x) { return isupper(x); }
int SDL_islower(int x) { return islower(x); }
int SDL_isprint(int x) { return isprint(x); }
int SDL_isgraph(int x) { return isgraph(x); }
int SDL_iscntrl(int x) { return iscntrl(x); }
int SDL_toupper(int x) { return toupper(x); }
int SDL_tolower(int x) { return tolower(x); }
#else
int SDL_isalpha(int x)
{
    return (SDL_isupper(x)) || (SDL_islower(x));
}
int SDL_isalnum(int x) { return (SDL_isalpha(x)) || (SDL_isdigit(x)); }
int SDL_isdigit(int x) { return ((x) >= '0') && ((x) <= '9'); }
int SDL_isxdigit(int x) { return (((x) >= 'A') && ((x) <= 'F')) || (((x) >= 'a') && ((x) <= 'f')) || (SDL_isdigit(x)); }
int SDL_ispunct(int x) { return (SDL_isgraph(x)) && (!SDL_isalnum(x)); }
int SDL_isspace(int x) { return ((x) == ' ') || ((x) == '\t') || ((x) == '\r') || ((x) == '\n') || ((x) == '\f') || ((x) == '\v'); }
int SDL_isupper(int x) { return ((x) >= 'A') && ((x) <= 'Z'); }
int SDL_islower(int x) { return ((x) >= 'a') && ((x) <= 'z'); }
int SDL_isprint(int x) { return ((x) >= ' ') && ((x) < '\x7f'); }
int SDL_isgraph(int x) { return (SDL_isprint(x)) && ((x) != ' '); }
int SDL_iscntrl(int x) { return (((x) >= '\0') && ((x) <= '\x1f')) || ((x) == '\x7f'); }
int SDL_toupper(int x) { return ((x) >= 'a') && ((x) <= 'z') ? ('A' + ((x) - 'a')) : (x); }
int SDL_tolower(int x) { return ((x) >= 'A') && ((x) <= 'Z') ? ('a' + ((x) - 'A')) : (x); }
#endif

/* This file contains a portable memcpy manipulation function for SDL */

void *SDL_memcpy(SDL_OUT_BYTECAP(len) void *dst, SDL_IN_BYTECAP(len) const void *src, size_t len)
{
#ifdef __GNUC__
    /* Presumably this is well tuned for speed.
       On my machine this is twice as fast as the C code below.
     */
    return __builtin_memcpy(dst, src, len);
#elif defined(HAVE_MEMCPY)
    return memcpy(dst, src, len);
#elif defined(HAVE_BCOPY)
    bcopy(src, dst, len);
    return dst;
#else
    /* GCC 4.9.0 with -O3 will generate movaps instructions with the loop
       using Uint32* pointers, so we need to make sure the pointers are
       aligned before we loop using them.
     */
    if (((uintptr_t)src & 0x3) || ((uintptr_t)dst & 0x3)) {
        /* Do an unaligned byte copy */
        Uint8 *srcp1 = (Uint8 *)src;
        Uint8 *dstp1 = (Uint8 *)dst;

        while (len--) {
            *dstp1++ = *srcp1++;
        }
    } else {
        size_t left = (len % 4);
        Uint32 *srcp4, *dstp4;
        Uint8 *srcp1, *dstp1;

        srcp4 = (Uint32 *)src;
        dstp4 = (Uint32 *)dst;
        len /= 4;
        while (len--) {
            *dstp4++ = *srcp4++;
        }

        srcp1 = (Uint8 *)srcp4;
        dstp1 = (Uint8 *)dstp4;
        switch (left) {
        case 3:
            *dstp1++ = *srcp1++;
        case 2:
            *dstp1++ = *srcp1++;
        case 1:
            *dstp1++ = *srcp1++;
        }
    }
    return dst;
#endif /* __GNUC__ */
}

void *SDL_memset(SDL_OUT_BYTECAP(len) void *dst, int c, size_t len)
{
#if defined(HAVE_MEMSET)
    return memset(dst, c, len);
#else
    size_t left;
    Uint32 *dstp4;
    Uint8 *dstp1 = (Uint8 *)dst;
    Uint8 value1;
    Uint32 value4;

    /* The value used in memset() is a byte, passed as an int */
    c &= 0xff;

    /* The destination pointer needs to be aligned on a 4-byte boundary to
     * execute a 32-bit set. Set first bytes manually if needed until it is
     * aligned. */
    value1 = (Uint8)c;
    while ((uintptr_t)dstp1 & 0x3) {
        if (len--) {
            *dstp1++ = value1;
        } else {
            return dst;
        }
    }

    value4 = ((Uint32)c | ((Uint32)c << 8) | ((Uint32)c << 16) | ((Uint32)c << 24));
    dstp4 = (Uint32 *)dstp1;
    left = (len % 4);
    len /= 4;
    while (len--) {
        *dstp4++ = value4;
    }

    dstp1 = (Uint8 *)dstp4;
    switch (left) {
    case 3:
        *dstp1++ = value1;
    case 2:
        *dstp1++ = value1;
    case 1:
        *dstp1++ = value1;
    }

    return dst;
#endif /* HAVE_MEMSET */
}

#if defined(HAVE_CTYPE_H) && defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
int SDL_isblank(int x)
{
    return isblank(x);
}
#else
int SDL_isblank(int x)
{
    return ((x) == ' ') || ((x) == '\t');
}
#endif

/* vi: set ts=4 sw=4 expandtab: */
