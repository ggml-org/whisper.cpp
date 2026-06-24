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

#ifndef SDL_config_macosx_h_
#define SDL_config_macosx_h_
#define SDL_config_h_

#include "SDL_platform.h"

/* This gets us MAC_OS_X_VERSION_MIN_REQUIRED... */
#include <AvailabilityMacros.h>

/* This is a set of defines to configure the SDL features */

#ifdef __LP64__
    #define SIZEOF_VOIDP 8
#else
    #define SIZEOF_VOIDP 4
#endif

/* Useful headers */
#define STDC_HEADERS    1
#define HAVE_ALLOCA_H       1
#define HAVE_CTYPE_H    1
#define HAVE_FLOAT_H    1
#define HAVE_INTTYPES_H 1
#define HAVE_LIMITS_H   1
#define HAVE_MATH_H 1
#define HAVE_SIGNAL_H   1
#define HAVE_STDINT_H   1
#define HAVE_STDIO_H    1
#define HAVE_STRING_H   1
#define HAVE_SYS_TYPES_H    1
#define HAVE_LIBUNWIND_H    1

/* C library functions */
#define HAVE_DLOPEN 1
#define HAVE_MALLOC 1
#define HAVE_CALLOC 1
#define HAVE_REALLOC    1
#define HAVE_FREE   1
#define HAVE_ALLOCA 1
#define HAVE_GETENV 1
#define HAVE_SETENV 1
#define HAVE_PUTENV 1
#define HAVE_UNSETENV   1
#define HAVE_QSORT  1
#define HAVE_BSEARCH 1
#define HAVE_ABS    1
#define HAVE_BCOPY  1
#define HAVE_MEMSET 1
#define HAVE_MEMCPY 1
#define HAVE_MEMMOVE    1
#define HAVE_MEMCMP 1
#define HAVE_STRLEN 1
#define HAVE_STRLCPY    1
#define HAVE_STRLCAT    1
#define HAVE_STRCHR 1
#define HAVE_STRRCHR    1
#define HAVE_STRSTR 1
#define HAVE_STRTOK_R 1
#define HAVE_STRTOL 1
#define HAVE_STRTOUL    1
#define HAVE_STRTOLL    1
#define HAVE_STRTOULL   1
#define HAVE_STRTOD 1
#define HAVE_ATOI   1
#define HAVE_ATOF   1
#define HAVE_STRCMP 1
#define HAVE_STRNCMP    1
#define HAVE_STRCASECMP 1
#define HAVE_STRNCASECMP 1
#define HAVE_STRCASESTR 1
#define HAVE_VSSCANF 1
#define HAVE_VSNPRINTF  1
#define HAVE_M_PI   1
#define HAVE_ACOS   1
#define HAVE_ACOSF  1
#define HAVE_ASIN   1
#define HAVE_ASINF  1
#define HAVE_ATAN   1
#define HAVE_ATANF  1
#define HAVE_ATAN2  1
#define HAVE_ATAN2F 1
#define HAVE_CEIL   1
#define HAVE_CEILF  1
#define HAVE_COPYSIGN   1
#define HAVE_COPYSIGNF  1
#define HAVE_COS    1
#define HAVE_COSF   1
#define HAVE_EXP    1
#define HAVE_EXPF   1
#define HAVE_FABS   1
#define HAVE_FABSF  1
#define HAVE_FLOOR  1
#define HAVE_FLOORF 1
#define HAVE_FMOD   1
#define HAVE_FMODF  1
#define HAVE_LOG    1
#define HAVE_LOGF   1
#define HAVE_LOG10  1
#define HAVE_LOG10F 1
#define HAVE_LROUND 1
#define HAVE_LROUNDF 1
#define HAVE_POW    1
#define HAVE_POWF   1
#define HAVE_ROUND  1
#define HAVE_ROUNDF 1
#define HAVE_SCALBN 1
#define HAVE_SCALBNF    1
#define HAVE_SIN    1
#define HAVE_SINF   1
#define HAVE_SQRT   1
#define HAVE_SQRTF  1
#define HAVE_TAN    1
#define HAVE_TANF   1
#define HAVE_TRUNC    1
#define HAVE_TRUNCF   1
#define HAVE_SIGACTION  1
#define HAVE_SETJMP 1
#define HAVE_NANOSLEEP  1
#define HAVE_SYSCONF    1
#define HAVE_SYSCTLBYNAME 1

#if defined(__has_include) && (defined(__i386__) || defined(__x86_64))
# if __has_include(<immintrin.h>)
#   define HAVE_IMMINTRIN_H 1
# endif
#endif

#if (MAC_OS_X_VERSION_MAX_ALLOWED >= 1070)
#define HAVE_O_CLOEXEC 1
#endif

#define HAVE_GCC_ATOMICS 1

/* Enable various audio drivers */
#define SDL_AUDIO_DRIVER_COREAUDIO  1
#define SDL_AUDIO_DRIVER_DISK   1
#define SDL_AUDIO_DRIVER_DUMMY  1

/* Enable various input drivers */
#define SDL_JOYSTICK_HIDAPI 1
#define SDL_JOYSTICK_IOKIT  1
#define SDL_JOYSTICK_VIRTUAL    1
#define SDL_HAPTIC_IOKIT    1

/* The MFI controller support requires ARC Objective C runtime */
#if MAC_OS_X_VERSION_MIN_REQUIRED >= 1080 && !defined(__i386__)
#define SDL_JOYSTICK_MFI 1
#endif

/* Enable the dummy sensor driver */
#define SDL_SENSOR_DUMMY  1

/* Enable various shared object loading systems */
#define SDL_LOADSO_DLOPEN   1

/* Enable various threading systems */
#define SDL_THREAD_PTHREAD  1
#define SDL_THREAD_PTHREAD_RECURSIVE_MUTEX  1

/* Enable various timer systems */
#define SDL_TIMER_UNIX  1

/* Enable various video drivers */
#define SDL_VIDEO_DRIVER_COCOA  1
#define SDL_VIDEO_DRIVER_DUMMY  1
#undef SDL_VIDEO_DRIVER_X11
#define SDL_VIDEO_DRIVER_X11_DYNAMIC "/opt/X11/lib/libX11.6.dylib"
#define SDL_VIDEO_DRIVER_X11_DYNAMIC_XEXT "/opt/X11/lib/libXext.6.dylib"
#define SDL_VIDEO_DRIVER_X11_DYNAMIC_XINPUT2 "/opt/X11/lib/libXi.6.dylib"
#define SDL_VIDEO_DRIVER_X11_DYNAMIC_XRANDR "/opt/X11/lib/libXrandr.2.dylib"
#define SDL_VIDEO_DRIVER_X11_DYNAMIC_XSS "/opt/X11/lib/libXss.1.dylib"
#define SDL_VIDEO_DRIVER_X11_XDBE 1
#define SDL_VIDEO_DRIVER_X11_XRANDR 1
#define SDL_VIDEO_DRIVER_X11_XSCRNSAVER 1
#define SDL_VIDEO_DRIVER_X11_XSHAPE 1
#define SDL_VIDEO_DRIVER_X11_HAS_XKBKEYCODETOKEYSYM 1

#ifdef MAC_OS_X_VERSION_10_8
/*
 * No matter the versions targeted, this is the 10.8 or later SDK, so you have
 *  to use the external Xquartz, which is a more modern Xlib. Previous SDKs
 *  used an older Xlib.
 */
#define SDL_VIDEO_DRIVER_X11_XINPUT2 1
#define SDL_VIDEO_DRIVER_X11_SUPPORTS_GENERIC_EVENTS 1
#endif

#ifndef SDL_VIDEO_RENDER_OGL
#define SDL_VIDEO_RENDER_OGL    1
#endif

#ifndef SDL_VIDEO_RENDER_OGL_ES2
#define SDL_VIDEO_RENDER_OGL_ES2 1
#endif

/* Metal only supported on 64-bit architectures with 10.11+ */
#if TARGET_RT_64_BIT && (MAC_OS_X_VERSION_MAX_ALLOWED >= 101100)
#define SDL_PLATFORM_SUPPORTS_METAL    1
#else
#define SDL_PLATFORM_SUPPORTS_METAL    0
#endif

#ifndef SDL_VIDEO_RENDER_METAL
#if SDL_PLATFORM_SUPPORTS_METAL
#define SDL_VIDEO_RENDER_METAL    1
#else
#define SDL_VIDEO_RENDER_METAL    0
#endif
#endif

/* Enable OpenGL support */
#ifndef SDL_VIDEO_OPENGL
#define SDL_VIDEO_OPENGL    1
#endif
#ifndef SDL_VIDEO_OPENGL_ES2
#define SDL_VIDEO_OPENGL_ES2    1
#endif
#ifndef SDL_VIDEO_OPENGL_EGL
#define SDL_VIDEO_OPENGL_EGL    1
#endif
#ifndef SDL_VIDEO_OPENGL_CGL
#define SDL_VIDEO_OPENGL_CGL    1
#endif
#ifndef SDL_VIDEO_OPENGL_GLX
#define SDL_VIDEO_OPENGL_GLX    1
#endif

/* Enable Vulkan and Metal support */
#ifndef SDL_VIDEO_VULKAN
#if SDL_PLATFORM_SUPPORTS_METAL
#define SDL_VIDEO_VULKAN 1
#else
#define SDL_VIDEO_VULKAN 0
#endif
#endif

#ifndef SDL_VIDEO_METAL
#if SDL_PLATFORM_SUPPORTS_METAL
#define SDL_VIDEO_METAL 1
#else
#define SDL_VIDEO_METAL 0
#endif
#endif

/* Enable system power support */
#define SDL_POWER_MACOSX 1

/* enable filesystem support */
#define SDL_FILESYSTEM_COCOA   1

/* Enable assembly routines */
#ifdef __ppc__
#define SDL_ALTIVEC_BLITTERS    1
#endif

#endif /* SDL_config_macosx_h_ */
