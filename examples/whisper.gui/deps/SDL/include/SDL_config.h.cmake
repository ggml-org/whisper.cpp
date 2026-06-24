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

#ifndef SDL_config_h_
#define SDL_config_h_

/**
 *  \file SDL_config.h.in
 *
 *  This is a set of defines to configure the SDL features
 */

/* General platform specific identifiers */
#include "SDL_platform.h"

/* C language features */
#cmakedefine const @HAVE_CONST@
#cmakedefine inline @HAVE_INLINE@
#cmakedefine volatile @HAVE_VOLATILE@

/* C datatypes */
/* Define SIZEOF_VOIDP for 64/32 architectures */
#if defined(__LP64__) || defined(_LP64) || defined(_WIN64)
#define SIZEOF_VOIDP 8
#else
#define SIZEOF_VOIDP 4
#endif

#cmakedefine HAVE_GCC_ATOMICS @HAVE_GCC_ATOMICS@
#cmakedefine HAVE_GCC_SYNC_LOCK_TEST_AND_SET @HAVE_GCC_SYNC_LOCK_TEST_AND_SET@

/* Comment this if you want to build without any C library requirements */
#cmakedefine HAVE_LIBC 1
#ifdef HAVE_LIBC

/* Useful headers */
#cmakedefine STDC_HEADERS 1
#cmakedefine HAVE_ALLOCA_H 1
#cmakedefine HAVE_CTYPE_H 1
#cmakedefine HAVE_FLOAT_H 1
#cmakedefine HAVE_ICONV_H 1
#cmakedefine HAVE_INTTYPES_H 1
#cmakedefine HAVE_LIMITS_H 1
#cmakedefine HAVE_MALLOC_H 1
#cmakedefine HAVE_MATH_H 1
#cmakedefine HAVE_MEMORY_H 1
#cmakedefine HAVE_SIGNAL_H 1
#cmakedefine HAVE_STDARG_H 1
#cmakedefine HAVE_STDDEF_H 1
#cmakedefine HAVE_STDINT_H 1
#cmakedefine HAVE_STDIO_H 1
#cmakedefine HAVE_STDLIB_H 1
#cmakedefine HAVE_STRINGS_H 1
#cmakedefine HAVE_STRING_H 1
#cmakedefine HAVE_SYS_TYPES_H 1
#cmakedefine HAVE_WCHAR_H 1
#cmakedefine HAVE_LINUX_INPUT_H 1
#cmakedefine HAVE_PTHREAD_NP_H 1
#cmakedefine HAVE_LIBUNWIND_H 1

/* C library functions */
#cmakedefine HAVE_DLOPEN 1
#cmakedefine HAVE_MALLOC 1
#cmakedefine HAVE_CALLOC 1
#cmakedefine HAVE_REALLOC 1
#cmakedefine HAVE_FREE 1
#cmakedefine HAVE_ALLOCA 1
#ifndef __WIN32__ /* Don't use C runtime versions of these on Windows */
#cmakedefine HAVE_GETENV 1
#cmakedefine HAVE_SETENV 1
#cmakedefine HAVE_PUTENV 1
#cmakedefine HAVE_UNSETENV 1
#endif
#cmakedefine HAVE_QSORT 1
#cmakedefine HAVE_BSEARCH 1
#cmakedefine HAVE_ABS 1
#cmakedefine HAVE_BCOPY 1
#cmakedefine HAVE_MEMSET 1
#cmakedefine HAVE_MEMCPY 1
#cmakedefine HAVE_MEMMOVE 1
#cmakedefine HAVE_MEMCMP 1
#cmakedefine HAVE_WCSLEN 1
#cmakedefine HAVE_WCSLCPY 1
#cmakedefine HAVE_WCSLCAT 1
#cmakedefine HAVE__WCSDUP 1
#cmakedefine HAVE_WCSDUP 1
#cmakedefine HAVE_WCSSTR 1
#cmakedefine HAVE_WCSCMP 1
#cmakedefine HAVE_WCSNCMP 1
#cmakedefine HAVE_WCSCASECMP 1
#cmakedefine HAVE__WCSICMP 1
#cmakedefine HAVE_WCSNCASECMP 1
#cmakedefine HAVE__WCSNICMP 1
#cmakedefine HAVE_STRLEN 1
#cmakedefine HAVE_STRLCPY 1
#cmakedefine HAVE_STRLCAT 1
#cmakedefine HAVE__STRREV 1
#cmakedefine HAVE__STRUPR 1
#cmakedefine HAVE__STRLWR 1
#cmakedefine HAVE_INDEX 1
#cmakedefine HAVE_RINDEX 1
#cmakedefine HAVE_STRCHR 1
#cmakedefine HAVE_STRRCHR 1
#cmakedefine HAVE_STRSTR 1
#cmakedefine HAVE_STRTOK_R 1
#cmakedefine HAVE_ITOA 1
#cmakedefine HAVE__LTOA 1
#cmakedefine HAVE__UITOA 1
#cmakedefine HAVE__ULTOA 1
#cmakedefine HAVE_STRTOL 1
#cmakedefine HAVE_STRTOUL 1
#cmakedefine HAVE__I64TOA 1
#cmakedefine HAVE__UI64TOA 1
#cmakedefine HAVE_STRTOLL 1
#cmakedefine HAVE_STRTOULL 1
#cmakedefine HAVE_STRTOD 1
#cmakedefine HAVE_ATOI 1
#cmakedefine HAVE_ATOF 1
#cmakedefine HAVE_STRCMP 1
#cmakedefine HAVE_STRNCMP 1
#cmakedefine HAVE__STRICMP 1
#cmakedefine HAVE_STRCASECMP 1
#cmakedefine HAVE__STRNICMP 1
#cmakedefine HAVE_STRNCASECMP 1
#cmakedefine HAVE_STRCASESTR 1
#cmakedefine HAVE_SSCANF 1
#cmakedefine HAVE_VSSCANF 1
#cmakedefine HAVE_VSNPRINTF 1
#cmakedefine HAVE_M_PI 1
#cmakedefine HAVE_ACOS 1
#cmakedefine HAVE_ACOSF 1
#cmakedefine HAVE_ASIN 1
#cmakedefine HAVE_ASINF 1
#cmakedefine HAVE_ATAN 1
#cmakedefine HAVE_ATANF 1
#cmakedefine HAVE_ATAN2 1
#cmakedefine HAVE_ATAN2F 1
#cmakedefine HAVE_CEIL 1
#cmakedefine HAVE_CEILF 1
#cmakedefine HAVE_COPYSIGN 1
#cmakedefine HAVE_COPYSIGNF 1
#cmakedefine HAVE_COS 1
#cmakedefine HAVE_COSF 1
#cmakedefine HAVE_EXP 1
#cmakedefine HAVE_EXPF 1
#cmakedefine HAVE_FABS 1
#cmakedefine HAVE_FABSF 1
#cmakedefine HAVE_FLOOR 1
#cmakedefine HAVE_FLOORF 1
#cmakedefine HAVE_FMOD 1
#cmakedefine HAVE_FMODF 1
#cmakedefine HAVE_LOG 1
#cmakedefine HAVE_LOGF 1
#cmakedefine HAVE_LOG10 1
#cmakedefine HAVE_LOG10F 1
#cmakedefine HAVE_LROUND 1
#cmakedefine HAVE_LROUNDF 1
#cmakedefine HAVE_POW 1
#cmakedefine HAVE_POWF 1
#cmakedefine HAVE_ROUND 1
#cmakedefine HAVE_ROUNDF 1
#cmakedefine HAVE_SCALBN 1
#cmakedefine HAVE_SCALBNF 1
#cmakedefine HAVE_SIN 1
#cmakedefine HAVE_SINF 1
#cmakedefine HAVE_SQRT 1
#cmakedefine HAVE_SQRTF 1
#cmakedefine HAVE_TAN 1
#cmakedefine HAVE_TANF 1
#cmakedefine HAVE_TRUNC 1
#cmakedefine HAVE_TRUNCF 1
#cmakedefine HAVE_FOPEN64 1
#cmakedefine HAVE_FSEEKO 1
#cmakedefine HAVE_FSEEKO64 1
#cmakedefine HAVE_MEMFD_CREATE 1
#cmakedefine HAVE_POSIX_FALLOCATE 1
#cmakedefine HAVE_SIGACTION 1
#cmakedefine HAVE_SA_SIGACTION 1
#cmakedefine HAVE_SETJMP 1
#cmakedefine HAVE_NANOSLEEP 1
#cmakedefine HAVE_SYSCONF 1
#cmakedefine HAVE_SYSCTLBYNAME 1
#cmakedefine HAVE_CLOCK_GETTIME 1
#cmakedefine HAVE_GETPAGESIZE 1
#cmakedefine HAVE_MPROTECT 1
#cmakedefine HAVE_ICONV 1
#cmakedefine SDL_USE_LIBICONV 1
#cmakedefine HAVE_PTHREAD_SETNAME_NP 1
#cmakedefine HAVE_PTHREAD_SET_NAME_NP 1
#cmakedefine HAVE_SEM_TIMEDWAIT 1
#cmakedefine HAVE_GETAUXVAL 1
#cmakedefine HAVE_ELF_AUX_INFO 1
#cmakedefine HAVE_POLL 1
#cmakedefine HAVE__EXIT 1

#else
#cmakedefine HAVE_STDARG_H 1
#cmakedefine HAVE_STDDEF_H 1
#cmakedefine HAVE_STDINT_H 1
#cmakedefine HAVE_FLOAT_H 1
#endif /* HAVE_LIBC */

#cmakedefine HAVE_ALTIVEC_H 1
#cmakedefine HAVE_DBUS_DBUS_H 1
#cmakedefine HAVE_FCITX 1
#cmakedefine HAVE_IBUS_IBUS_H 1
#cmakedefine HAVE_SYS_INOTIFY_H 1
#cmakedefine HAVE_INOTIFY_INIT 1
#cmakedefine HAVE_INOTIFY_INIT1 1
#cmakedefine HAVE_INOTIFY 1
#cmakedefine HAVE_LIBUSB 1
#cmakedefine HAVE_O_CLOEXEC 1

/* Apple platforms might be building universal binaries, where Intel builds
   can use immintrin.h but other architectures can't. */
#ifdef __APPLE__
#  if defined(__has_include) && (defined(__i386__) || defined(__x86_64))
#    if __has_include(<immintrin.h>)
#       define HAVE_IMMINTRIN_H 1
#    endif
#  endif
#else  /* non-Apple platforms can use the normal CMake check for this. */
#cmakedefine HAVE_IMMINTRIN_H 1
#endif

#cmakedefine HAVE_LIBUDEV_H 1
#cmakedefine HAVE_LIBSAMPLERATE_H 1
#cmakedefine HAVE_LIBDECOR_H 1

#cmakedefine HAVE_D3D_H @HAVE_D3D_H@
#cmakedefine HAVE_D3D11_H @HAVE_D3D11_H@
#cmakedefine HAVE_D3D12_H @HAVE_D3D12_H@
#cmakedefine HAVE_DDRAW_H @HAVE_DDRAW_H@
#cmakedefine HAVE_DSOUND_H @HAVE_DSOUND_H@
#cmakedefine HAVE_DINPUT_H @HAVE_DINPUT_H@
#cmakedefine HAVE_XINPUT_H @HAVE_XINPUT_H@
#cmakedefine HAVE_WINDOWS_GAMING_INPUT_H @HAVE_WINDOWS_GAMING_INPUT_H@
#cmakedefine HAVE_DXGI_H @HAVE_DXGI_H@

#cmakedefine HAVE_MMDEVICEAPI_H @HAVE_MMDEVICEAPI_H@
#cmakedefine HAVE_AUDIOCLIENT_H @HAVE_AUDIOCLIENT_H@
#cmakedefine HAVE_TPCSHRD_H @HAVE_TPCSHRD_H@
#cmakedefine HAVE_SENSORSAPI_H @HAVE_SENSORSAPI_H@
#cmakedefine HAVE_ROAPI_H @HAVE_ROAPI_H@
#cmakedefine HAVE_SHELLSCALINGAPI_H @HAVE_SHELLSCALINGAPI_H@

#cmakedefine USE_POSIX_SPAWN @USE_POSIX_SPAWN@

/* SDL internal assertion support */
#if @SDL_DEFAULT_ASSERT_LEVEL_CONFIGURED@
#cmakedefine SDL_DEFAULT_ASSERT_LEVEL @SDL_DEFAULT_ASSERT_LEVEL@
#endif

/* Allow disabling of core subsystems */
#cmakedefine SDL_ATOMIC_DISABLED @SDL_ATOMIC_DISABLED@
#cmakedefine SDL_AUDIO_DISABLED @SDL_AUDIO_DISABLED@
#cmakedefine SDL_CPUINFO_DISABLED @SDL_CPUINFO_DISABLED@
#cmakedefine SDL_EVENTS_DISABLED @SDL_EVENTS_DISABLED@
#cmakedefine SDL_FILE_DISABLED @SDL_FILE_DISABLED@
#cmakedefine SDL_JOYSTICK_DISABLED @SDL_JOYSTICK_DISABLED@
#cmakedefine SDL_HAPTIC_DISABLED @SDL_HAPTIC_DISABLED@
#cmakedefine SDL_HIDAPI_DISABLED @SDL_HIDAPI_DISABLED@
#cmakedefine SDL_SENSOR_DISABLED @SDL_SENSOR_DISABLED@
#cmakedefine SDL_LOADSO_DISABLED @SDL_LOADSO_DISABLED@
#cmakedefine SDL_RENDER_DISABLED @SDL_RENDER_DISABLED@
#cmakedefine SDL_THREADS_DISABLED @SDL_THREADS_DISABLED@
#cmakedefine SDL_TIMERS_DISABLED @SDL_TIMERS_DISABLED@
#cmakedefine SDL_VIDEO_DISABLED @SDL_VIDEO_DISABLED@
#cmakedefine SDL_POWER_DISABLED @SDL_POWER_DISABLED@
#cmakedefine SDL_FILESYSTEM_DISABLED @SDL_FILESYSTEM_DISABLED@
#cmakedefine SDL_LOCALE_DISABLED @SDL_LOCALE_DISABLED@
#cmakedefine SDL_MISC_DISABLED @SDL_MISC_DISABLED@

/* Enable various audio drivers */
#cmakedefine SDL_AUDIO_DRIVER_ALSA @SDL_AUDIO_DRIVER_ALSA@
#cmakedefine SDL_AUDIO_DRIVER_ALSA_DYNAMIC @SDL_AUDIO_DRIVER_ALSA_DYNAMIC@
#cmakedefine SDL_AUDIO_DRIVER_ANDROID @SDL_AUDIO_DRIVER_ANDROID@
#cmakedefine SDL_AUDIO_DRIVER_OPENSLES @SDL_AUDIO_DRIVER_OPENSLES@
#cmakedefine SDL_AUDIO_DRIVER_AAUDIO @SDL_AUDIO_DRIVER_AAUDIO@
#cmakedefine SDL_AUDIO_DRIVER_ARTS @SDL_AUDIO_DRIVER_ARTS@
#cmakedefine SDL_AUDIO_DRIVER_ARTS_DYNAMIC @SDL_AUDIO_DRIVER_ARTS_DYNAMIC@
#cmakedefine SDL_AUDIO_DRIVER_COREAUDIO @SDL_AUDIO_DRIVER_COREAUDIO@
#cmakedefine SDL_AUDIO_DRIVER_DISK @SDL_AUDIO_DRIVER_DISK@
#cmakedefine SDL_AUDIO_DRIVER_DSOUND @SDL_AUDIO_DRIVER_DSOUND@
#cmakedefine SDL_AUDIO_DRIVER_DUMMY @SDL_AUDIO_DRIVER_DUMMY@
#cmakedefine SDL_AUDIO_DRIVER_EMSCRIPTEN @SDL_AUDIO_DRIVER_EMSCRIPTEN@
#cmakedefine SDL_AUDIO_DRIVER_ESD @SDL_AUDIO_DRIVER_ESD@
#cmakedefine SDL_AUDIO_DRIVER_ESD_DYNAMIC @SDL_AUDIO_DRIVER_ESD_DYNAMIC@
#cmakedefine SDL_AUDIO_DRIVER_FUSIONSOUND @SDL_AUDIO_DRIVER_FUSIONSOUND@
#cmakedefine SDL_AUDIO_DRIVER_FUSIONSOUND_DYNAMIC @SDL_AUDIO_DRIVER_FUSIONSOUND_DYNAMIC@
#cmakedefine SDL_AUDIO_DRIVER_HAIKU @SDL_AUDIO_DRIVER_HAIKU@
#cmakedefine SDL_AUDIO_DRIVER_JACK @SDL_AUDIO_DRIVER_JACK@
#cmakedefine SDL_AUDIO_DRIVER_JACK_DYNAMIC @SDL_AUDIO_DRIVER_JACK_DYNAMIC@
#cmakedefine SDL_AUDIO_DRIVER_NAS @SDL_AUDIO_DRIVER_NAS@
#cmakedefine SDL_AUDIO_DRIVER_NAS_DYNAMIC @SDL_AUDIO_DRIVER_NAS_DYNAMIC@
#cmakedefine SDL_AUDIO_DRIVER_NETBSD @SDL_AUDIO_DRIVER_NETBSD@
#cmakedefine SDL_AUDIO_DRIVER_OSS @SDL_AUDIO_DRIVER_OSS@
#cmakedefine SDL_AUDIO_DRIVER_PAUDIO @SDL_AUDIO_DRIVER_PAUDIO@
#cmakedefine SDL_AUDIO_DRIVER_PIPEWIRE @SDL_AUDIO_DRIVER_PIPEWIRE@
#cmakedefine SDL_AUDIO_DRIVER_PIPEWIRE_DYNAMIC @SDL_AUDIO_DRIVER_PIPEWIRE_DYNAMIC@
#cmakedefine SDL_AUDIO_DRIVER_PULSEAUDIO @SDL_AUDIO_DRIVER_PULSEAUDIO@
#cmakedefine SDL_AUDIO_DRIVER_PULSEAUDIO_DYNAMIC @SDL_AUDIO_DRIVER_PULSEAUDIO_DYNAMIC@
#cmakedefine SDL_AUDIO_DRIVER_QSA @SDL_AUDIO_DRIVER_QSA@
#cmakedefine SDL_AUDIO_DRIVER_SNDIO @SDL_AUDIO_DRIVER_SNDIO@
#cmakedefine SDL_AUDIO_DRIVER_SNDIO_DYNAMIC @SDL_AUDIO_DRIVER_SNDIO_DYNAMIC@
#cmakedefine SDL_AUDIO_DRIVER_SUNAUDIO @SDL_AUDIO_DRIVER_SUNAUDIO@
#cmakedefine SDL_AUDIO_DRIVER_WASAPI @SDL_AUDIO_DRIVER_WASAPI@
#cmakedefine SDL_AUDIO_DRIVER_WINMM @SDL_AUDIO_DRIVER_WINMM@
#cmakedefine SDL_AUDIO_DRIVER_OS2 @SDL_AUDIO_DRIVER_OS2@
#cmakedefine SDL_AUDIO_DRIVER_VITA @SDL_AUDIO_DRIVER_VITA@
#cmakedefine SDL_AUDIO_DRIVER_PSP @SDL_AUDIO_DRIVER_PSP@
#cmakedefine SDL_AUDIO_DRIVER_PS2 @SDL_AUDIO_DRIVER_PS2@
#cmakedefine SDL_AUDIO_DRIVER_N3DS @SDL_AUDIO_DRIVER_N3DS@

/* Enable various input drivers */
#cmakedefine SDL_INPUT_LINUXEV @SDL_INPUT_LINUXEV@
#cmakedefine SDL_INPUT_LINUXKD @SDL_INPUT_LINUXKD@
#cmakedefine SDL_INPUT_FBSDKBIO @SDL_INPUT_FBSDKBIO@
#cmakedefine SDL_INPUT_WSCONS @SDL_INPUT_WSCONS@
#cmakedefine SDL_JOYSTICK_ANDROID @SDL_JOYSTICK_ANDROID@
#cmakedefine SDL_JOYSTICK_HAIKU @SDL_JOYSTICK_HAIKU@
#cmakedefine SDL_JOYSTICK_WGI @SDL_JOYSTICK_WGI@
#cmakedefine SDL_JOYSTICK_DINPUT @SDL_JOYSTICK_DINPUT@
#cmakedefine SDL_JOYSTICK_XINPUT @SDL_JOYSTICK_XINPUT@
#cmakedefine SDL_JOYSTICK_DUMMY @SDL_JOYSTICK_DUMMY@
#cmakedefine SDL_JOYSTICK_IOKIT @SDL_JOYSTICK_IOKIT@
#cmakedefine SDL_JOYSTICK_MFI @SDL_JOYSTICK_MFI@
#cmakedefine SDL_JOYSTICK_LINUX @SDL_JOYSTICK_LINUX@
#cmakedefine SDL_JOYSTICK_OS2 @SDL_JOYSTICK_OS2@
#cmakedefine SDL_JOYSTICK_USBHID @SDL_JOYSTICK_USBHID@
#cmakedefine SDL_HAVE_MACHINE_JOYSTICK_H @SDL_HAVE_MACHINE_JOYSTICK_H@
#cmakedefine SDL_JOYSTICK_HIDAPI @SDL_JOYSTICK_HIDAPI@
#cmakedefine SDL_JOYSTICK_RAWINPUT @SDL_JOYSTICK_RAWINPUT@
#cmakedefine SDL_JOYSTICK_EMSCRIPTEN @SDL_JOYSTICK_EMSCRIPTEN@
#cmakedefine SDL_JOYSTICK_VIRTUAL @SDL_JOYSTICK_VIRTUAL@
#cmakedefine SDL_JOYSTICK_VITA @SDL_JOYSTICK_VITA@
#cmakedefine SDL_JOYSTICK_PSP @SDL_JOYSTICK_PSP@
#cmakedefine SDL_JOYSTICK_PS2 @SDL_JOYSTICK_PS2@
#cmakedefine SDL_JOYSTICK_N3DS @SDL_JOYSTICK_N3DS@
#cmakedefine SDL_HAPTIC_DUMMY @SDL_HAPTIC_DUMMY@
#cmakedefine SDL_HAPTIC_LINUX @SDL_HAPTIC_LINUX@
#cmakedefine SDL_HAPTIC_IOKIT @SDL_HAPTIC_IOKIT@
#cmakedefine SDL_HAPTIC_DINPUT @SDL_HAPTIC_DINPUT@
#cmakedefine SDL_HAPTIC_XINPUT @SDL_HAPTIC_XINPUT@
#cmakedefine SDL_HAPTIC_ANDROID @SDL_HAPTIC_ANDROID@
#cmakedefine SDL_LIBUSB_DYNAMIC @SDL_LIBUSB_DYNAMIC@
#cmakedefine SDL_UDEV_DYNAMIC @SDL_UDEV_DYNAMIC@

/* Enable various sensor drivers */
#cmakedefine SDL_SENSOR_ANDROID @SDL_SENSOR_ANDROID@
#cmakedefine SDL_SENSOR_COREMOTION @SDL_SENSOR_COREMOTION@
#cmakedefine SDL_SENSOR_WINDOWS @SDL_SENSOR_WINDOWS@
#cmakedefine SDL_SENSOR_DUMMY @SDL_SENSOR_DUMMY@
#cmakedefine SDL_SENSOR_VITA @SDL_SENSOR_VITA@
#cmakedefine SDL_SENSOR_N3DS @SDL_SENSOR_N3DS@

/* Enable various shared object loading systems */
#cmakedefine SDL_LOADSO_DLOPEN @SDL_LOADSO_DLOPEN@
#cmakedefine SDL_LOADSO_DUMMY @SDL_LOADSO_DUMMY@
#cmakedefine SDL_LOADSO_LDG @SDL_LOADSO_LDG@
#cmakedefine SDL_LOADSO_WINDOWS @SDL_LOADSO_WINDOWS@
#cmakedefine SDL_LOADSO_OS2 @SDL_LOADSO_OS2@

/* Enable various threading systems */
#cmakedefine SDL_THREAD_GENERIC_COND_SUFFIX @SDL_THREAD_GENERIC_COND_SUFFIX@
#cmakedefine SDL_THREAD_PTHREAD @SDL_THREAD_PTHREAD@
#cmakedefine SDL_THREAD_PTHREAD_RECURSIVE_MUTEX @SDL_THREAD_PTHREAD_RECURSIVE_MUTEX@
#cmakedefine SDL_THREAD_PTHREAD_RECURSIVE_MUTEX_NP @SDL_THREAD_PTHREAD_RECURSIVE_MUTEX_NP@
#cmakedefine SDL_THREAD_WINDOWS @SDL_THREAD_WINDOWS@
#cmakedefine SDL_THREAD_OS2 @SDL_THREAD_OS2@
#cmakedefine SDL_THREAD_VITA @SDL_THREAD_VITA@
#cmakedefine SDL_THREAD_PSP @SDL_THREAD_PSP@
#cmakedefine SDL_THREAD_PS2 @SDL_THREAD_PS2@
#cmakedefine SDL_THREAD_N3DS @SDL_THREAD_N3DS@

/* Enable various timer systems */
#cmakedefine SDL_TIMER_HAIKU @SDL_TIMER_HAIKU@
#cmakedefine SDL_TIMER_DUMMY @SDL_TIMER_DUMMY@
#cmakedefine SDL_TIMER_UNIX @SDL_TIMER_UNIX@
#cmakedefine SDL_TIMER_WINDOWS @SDL_TIMER_WINDOWS@
#cmakedefine SDL_TIMER_OS2 @SDL_TIMER_OS2@
#cmakedefine SDL_TIMER_VITA @SDL_TIMER_VITA@
#cmakedefine SDL_TIMER_PSP @SDL_TIMER_PSP@
#cmakedefine SDL_TIMER_PS2 @SDL_TIMER_PS2@
#cmakedefine SDL_TIMER_N3DS @SDL_TIMER_N3DS@

/* Enable various video drivers */
#cmakedefine SDL_VIDEO_DRIVER_ANDROID @SDL_VIDEO_DRIVER_ANDROID@
#cmakedefine SDL_VIDEO_DRIVER_EMSCRIPTEN @SDL_VIDEO_DRIVER_EMSCRIPTEN@
#cmakedefine SDL_VIDEO_DRIVER_HAIKU @SDL_VIDEO_DRIVER_HAIKU@
#cmakedefine SDL_VIDEO_DRIVER_COCOA @SDL_VIDEO_DRIVER_COCOA@
#cmakedefine SDL_VIDEO_DRIVER_UIKIT @SDL_VIDEO_DRIVER_UIKIT@
#cmakedefine SDL_VIDEO_DRIVER_DIRECTFB @SDL_VIDEO_DRIVER_DIRECTFB@
#cmakedefine SDL_VIDEO_DRIVER_DIRECTFB_DYNAMIC @SDL_VIDEO_DRIVER_DIRECTFB_DYNAMIC@
#cmakedefine SDL_VIDEO_DRIVER_DUMMY @SDL_VIDEO_DRIVER_DUMMY@
#cmakedefine SDL_VIDEO_DRIVER_OFFSCREEN @SDL_VIDEO_DRIVER_OFFSCREEN@
#cmakedefine SDL_VIDEO_DRIVER_WINDOWS @SDL_VIDEO_DRIVER_WINDOWS@
#cmakedefine SDL_VIDEO_DRIVER_WINRT @SDL_VIDEO_DRIVER_WINRT@
#cmakedefine SDL_VIDEO_DRIVER_WAYLAND @SDL_VIDEO_DRIVER_WAYLAND@
#cmakedefine SDL_VIDEO_DRIVER_RPI @SDL_VIDEO_DRIVER_RPI@
#cmakedefine SDL_VIDEO_DRIVER_VIVANTE @SDL_VIDEO_DRIVER_VIVANTE@
#cmakedefine SDL_VIDEO_DRIVER_VIVANTE_VDK @SDL_VIDEO_DRIVER_VIVANTE_VDK@
#cmakedefine SDL_VIDEO_DRIVER_OS2 @SDL_VIDEO_DRIVER_OS2@
#cmakedefine SDL_VIDEO_DRIVER_QNX @SDL_VIDEO_DRIVER_QNX@
#cmakedefine SDL_VIDEO_DRIVER_RISCOS @SDL_VIDEO_DRIVER_RISCOS@
#cmakedefine SDL_VIDEO_DRIVER_PSP @SDL_VIDEO_DRIVER_PSP@
#cmakedefine SDL_VIDEO_DRIVER_PS2 @SDL_VIDEO_DRIVER_PS2@

#cmakedefine SDL_VIDEO_DRIVER_KMSDRM @SDL_VIDEO_DRIVER_KMSDRM@
#cmakedefine SDL_VIDEO_DRIVER_KMSDRM_DYNAMIC @SDL_VIDEO_DRIVER_KMSDRM_DYNAMIC@
#cmakedefine SDL_VIDEO_DRIVER_KMSDRM_DYNAMIC_GBM @SDL_VIDEO_DRIVER_KMSDRM_DYNAMIC_GBM@

#cmakedefine SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH @SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH@
#cmakedefine SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC @SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC@
#cmakedefine SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_EGL @SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_EGL@
#cmakedefine SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_CURSOR @SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_CURSOR@
#cmakedefine SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_XKBCOMMON @SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_XKBCOMMON@
#cmakedefine SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_LIBDECOR @SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_LIBDECOR@

#cmakedefine SDL_VIDEO_DRIVER_X11 @SDL_VIDEO_DRIVER_X11@
#cmakedefine SDL_VIDEO_DRIVER_X11_DYNAMIC @SDL_VIDEO_DRIVER_X11_DYNAMIC@
#cmakedefine SDL_VIDEO_DRIVER_X11_DYNAMIC_XEXT @SDL_VIDEO_DRIVER_X11_DYNAMIC_XEXT@
#cmakedefine SDL_VIDEO_DRIVER_X11_DYNAMIC_XCURSOR @SDL_VIDEO_DRIVER_X11_DYNAMIC_XCURSOR@
#cmakedefine SDL_VIDEO_DRIVER_X11_DYNAMIC_XINPUT2 @SDL_VIDEO_DRIVER_X11_DYNAMIC_XINPUT2@
#cmakedefine SDL_VIDEO_DRIVER_X11_DYNAMIC_XFIXES @SDL_VIDEO_DRIVER_X11_DYNAMIC_XFIXES@
#cmakedefine SDL_VIDEO_DRIVER_X11_DYNAMIC_XRANDR @SDL_VIDEO_DRIVER_X11_DYNAMIC_XRANDR@
#cmakedefine SDL_VIDEO_DRIVER_X11_DYNAMIC_XSS @SDL_VIDEO_DRIVER_X11_DYNAMIC_XSS@
#cmakedefine SDL_VIDEO_DRIVER_X11_XCURSOR @SDL_VIDEO_DRIVER_X11_XCURSOR@
#cmakedefine SDL_VIDEO_DRIVER_X11_XDBE @SDL_VIDEO_DRIVER_X11_XDBE@
#cmakedefine SDL_VIDEO_DRIVER_X11_XINPUT2 @SDL_VIDEO_DRIVER_X11_XINPUT2@
#cmakedefine SDL_VIDEO_DRIVER_X11_XINPUT2_SUPPORTS_MULTITOUCH @SDL_VIDEO_DRIVER_X11_XINPUT2_SUPPORTS_MULTITOUCH@
#cmakedefine SDL_VIDEO_DRIVER_X11_XFIXES @SDL_VIDEO_DRIVER_X11_XFIXES@
#cmakedefine SDL_VIDEO_DRIVER_X11_XRANDR @SDL_VIDEO_DRIVER_X11_XRANDR@
#cmakedefine SDL_VIDEO_DRIVER_X11_XSCRNSAVER @SDL_VIDEO_DRIVER_X11_XSCRNSAVER@
#cmakedefine SDL_VIDEO_DRIVER_X11_XSHAPE @SDL_VIDEO_DRIVER_X11_XSHAPE@
#cmakedefine SDL_VIDEO_DRIVER_X11_SUPPORTS_GENERIC_EVENTS @SDL_VIDEO_DRIVER_X11_SUPPORTS_GENERIC_EVENTS@
#cmakedefine SDL_VIDEO_DRIVER_X11_HAS_XKBKEYCODETOKEYSYM @SDL_VIDEO_DRIVER_X11_HAS_XKBKEYCODETOKEYSYM@
#cmakedefine SDL_VIDEO_DRIVER_VITA @SDL_VIDEO_DRIVER_VITA@
#cmakedefine SDL_VIDEO_DRIVER_N3DS @SDL_VIDEO_DRIVER_N3DS@

#cmakedefine SDL_VIDEO_RENDER_D3D @SDL_VIDEO_RENDER_D3D@
#cmakedefine SDL_VIDEO_RENDER_D3D11 @SDL_VIDEO_RENDER_D3D11@
#cmakedefine SDL_VIDEO_RENDER_D3D12 @SDL_VIDEO_RENDER_D3D12@
#cmakedefine SDL_VIDEO_RENDER_OGL @SDL_VIDEO_RENDER_OGL@
#cmakedefine SDL_VIDEO_RENDER_OGL_ES @SDL_VIDEO_RENDER_OGL_ES@
#cmakedefine SDL_VIDEO_RENDER_OGL_ES2 @SDL_VIDEO_RENDER_OGL_ES2@
#cmakedefine SDL_VIDEO_RENDER_DIRECTFB @SDL_VIDEO_RENDER_DIRECTFB@
#cmakedefine SDL_VIDEO_RENDER_METAL @SDL_VIDEO_RENDER_METAL@
#cmakedefine SDL_VIDEO_RENDER_VITA_GXM @SDL_VIDEO_RENDER_VITA_GXM@
#cmakedefine SDL_VIDEO_RENDER_PS2 @SDL_VIDEO_RENDER_PS2@
#cmakedefine SDL_VIDEO_RENDER_PSP @SDL_VIDEO_RENDER_PSP@

/* Enable OpenGL support */
#cmakedefine SDL_VIDEO_OPENGL @SDL_VIDEO_OPENGL@
#cmakedefine SDL_VIDEO_OPENGL_ES @SDL_VIDEO_OPENGL_ES@
#cmakedefine SDL_VIDEO_OPENGL_ES2 @SDL_VIDEO_OPENGL_ES2@
#cmakedefine SDL_VIDEO_OPENGL_BGL @SDL_VIDEO_OPENGL_BGL@
#cmakedefine SDL_VIDEO_OPENGL_CGL @SDL_VIDEO_OPENGL_CGL@
#cmakedefine SDL_VIDEO_OPENGL_GLX @SDL_VIDEO_OPENGL_GLX@
#cmakedefine SDL_VIDEO_OPENGL_WGL @SDL_VIDEO_OPENGL_WGL@
#cmakedefine SDL_VIDEO_OPENGL_EGL @SDL_VIDEO_OPENGL_EGL@
#cmakedefine SDL_VIDEO_OPENGL_OSMESA @SDL_VIDEO_OPENGL_OSMESA@
#cmakedefine SDL_VIDEO_OPENGL_OSMESA_DYNAMIC @SDL_VIDEO_OPENGL_OSMESA_DYNAMIC@

/* Enable Vulkan support */
#cmakedefine SDL_VIDEO_VULKAN @SDL_VIDEO_VULKAN@

/* Enable Metal support */
#cmakedefine SDL_VIDEO_METAL @SDL_VIDEO_METAL@

/* Enable system power support */
#cmakedefine SDL_POWER_ANDROID @SDL_POWER_ANDROID@
#cmakedefine SDL_POWER_LINUX @SDL_POWER_LINUX@
#cmakedefine SDL_POWER_WINDOWS @SDL_POWER_WINDOWS@
#cmakedefine SDL_POWER_WINRT @SDL_POWER_WINRT@
#cmakedefine SDL_POWER_MACOSX @SDL_POWER_MACOSX@
#cmakedefine SDL_POWER_UIKIT @SDL_POWER_UIKIT@
#cmakedefine SDL_POWER_HAIKU @SDL_POWER_HAIKU@
#cmakedefine SDL_POWER_EMSCRIPTEN @SDL_POWER_EMSCRIPTEN@
#cmakedefine SDL_POWER_HARDWIRED @SDL_POWER_HARDWIRED@
#cmakedefine SDL_POWER_VITA @SDL_POWER_VITA@
#cmakedefine SDL_POWER_PSP @SDL_POWER_PSP@
#cmakedefine SDL_POWER_N3DS @SDL_POWER_N3DS@

/* Enable system filesystem support */
#cmakedefine SDL_FILESYSTEM_ANDROID @SDL_FILESYSTEM_ANDROID@
#cmakedefine SDL_FILESYSTEM_HAIKU @SDL_FILESYSTEM_HAIKU@
#cmakedefine SDL_FILESYSTEM_COCOA @SDL_FILESYSTEM_COCOA@
#cmakedefine SDL_FILESYSTEM_DUMMY @SDL_FILESYSTEM_DUMMY@
#cmakedefine SDL_FILESYSTEM_RISCOS @SDL_FILESYSTEM_RISCOS@
#cmakedefine SDL_FILESYSTEM_UNIX @SDL_FILESYSTEM_UNIX@
#cmakedefine SDL_FILESYSTEM_WINDOWS @SDL_FILESYSTEM_WINDOWS@
#cmakedefine SDL_FILESYSTEM_EMSCRIPTEN @SDL_FILESYSTEM_EMSCRIPTEN@
#cmakedefine SDL_FILESYSTEM_OS2 @SDL_FILESYSTEM_OS2@
#cmakedefine SDL_FILESYSTEM_VITA @SDL_FILESYSTEM_VITA@
#cmakedefine SDL_FILESYSTEM_PSP @SDL_FILESYSTEM_PSP@
#cmakedefine SDL_FILESYSTEM_PS2 @SDL_FILESYSTEM_PS2@
#cmakedefine SDL_FILESYSTEM_N3DS @SDL_FILESYSTEM_N3DS@

/* Enable misc subsystem */
#cmakedefine SDL_MISC_DUMMY @SDL_MISC_DUMMY@

/* Enable locale subsystem */
#cmakedefine SDL_LOCALE_DUMMY @SDL_LOCALE_DUMMY@

/* Enable assembly routines */
#cmakedefine SDL_ALTIVEC_BLITTERS @SDL_ALTIVEC_BLITTERS@
#cmakedefine SDL_ARM_SIMD_BLITTERS @SDL_ARM_SIMD_BLITTERS@
#cmakedefine SDL_ARM_NEON_BLITTERS @SDL_ARM_NEON_BLITTERS@

/* Whether SDL_DYNAMIC_API needs dlopen */
#cmakedefine DYNAPI_NEEDS_DLOPEN @DYNAPI_NEEDS_DLOPEN@

/* Enable dynamic libsamplerate support */
#cmakedefine SDL_LIBSAMPLERATE_DYNAMIC @SDL_LIBSAMPLERATE_DYNAMIC@

/* Enable ime support */
#cmakedefine SDL_USE_IME @SDL_USE_IME@

/* Platform specific definitions */
#cmakedefine SDL_IPHONE_KEYBOARD @SDL_IPHONE_KEYBOARD@
#cmakedefine SDL_IPHONE_LAUNCHSCREEN @SDL_IPHONE_LAUNCHSCREEN@

#cmakedefine SDL_VIDEO_VITA_PIB @SDL_VIDEO_VITA_PIB@
#cmakedefine SDL_VIDEO_VITA_PVR @SDL_VIDEO_VITA_PVR@
#cmakedefine SDL_VIDEO_VITA_PVR_OGL @SDL_VIDEO_VITA_PVR_OGL@

#cmakedefine SDL_HAVE_LIBDECOR_GET_MIN_MAX @SDL_HAVE_LIBDECOR_GET_MIN_MAX@

#if !defined(HAVE_STDINT_H) && !defined(_STDINT_H_)
/* Most everything except Visual Studio 2008 and earlier has stdint.h now */
#if defined(_MSC_VER) && (_MSC_VER < 1600)
typedef signed __int8 int8_t;
typedef unsigned __int8 uint8_t;
typedef signed __int16 int16_t;
typedef unsigned __int16 uint16_t;
typedef signed __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef signed __int64 int64_t;
typedef unsigned __int64 uint64_t;
#ifndef _UINTPTR_T_DEFINED
#ifdef  _WIN64
typedef unsigned __int64 uintptr_t;
#else
typedef unsigned int uintptr_t;
#endif
#define _UINTPTR_T_DEFINED
#endif
#endif /* Visual Studio 2008 */
#endif /* !_STDINT_H_ && !HAVE_STDINT_H */

#endif /* SDL_config_h_ */
