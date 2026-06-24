dnl Process this file with autoconf to produce a configure script.
AC_PREREQ([2.65])
AC_INIT
AC_CONFIG_SRCDIR([src/SDL.c])
AC_CONFIG_HEADERS([include/SDL_config.h])
AC_CONFIG_AUX_DIR([build-scripts])
AC_CONFIG_MACRO_DIR([acinclude])

dnl Save the CFLAGS to see whether they were passed in or generated
orig_CFLAGS="$CFLAGS"

dnl Set various version strings - taken gratefully from the GTk sources
# See docs/release_checklist.md
SDL_MAJOR_VERSION=2
SDL_MINOR_VERSION=30
SDL_MICRO_VERSION=11
SDL_VERSION=$SDL_MAJOR_VERSION.$SDL_MINOR_VERSION.$SDL_MICRO_VERSION

SDL_BINARY_AGE=`expr $SDL_MINOR_VERSION \* 100 + $SDL_MICRO_VERSION`
AS_CASE(["$SDL_MINOR_VERSION"],
  [*@<:@02468@:>@],
    dnl Stable branch, 2.24.1 -> libSDL2-2.0.so.0.2400.1
    [SDL_INTERFACE_AGE="$SDL_MICRO_VERSION"],
  [*],
    dnl Development branch, 2.23.1 -> libSDL2-2.0.so.0.2301.0
    [SDL_INTERFACE_AGE=0])

AC_SUBST(SDL_MAJOR_VERSION)
AC_SUBST(SDL_MINOR_VERSION)
AC_SUBST(SDL_MICRO_VERSION)
AC_SUBST(SDL_INTERFACE_AGE)
AC_SUBST(SDL_BINARY_AGE)
AC_SUBST(SDL_VERSION)

# libtool versioning
LT_INIT([win32-dll])
LT_LANG([Windows Resource])

# For historical reasons, the library name redundantly includes the major
# version twice: libSDL2-2.0.so.0.
# TODO: in SDL 3, stop using -release, which will simplify it to libSDL3.so.0
LT_RELEASE=2.0
# Increment this if there is an incompatible change - but if that happens,
# we should rename the library from SDL2 to SDL3, at which point this would
# reset to 0 anyway.
LT_MAJOR=0
LT_AGE=`expr $SDL_BINARY_AGE - $SDL_INTERFACE_AGE`
LT_CURRENT=`expr $LT_MAJOR + $LT_AGE`
LT_REVISION=$SDL_INTERFACE_AGE
m4_pattern_allow([^LT_])

AC_SUBST(LT_RELEASE)
AC_SUBST(LT_CURRENT)
AC_SUBST(LT_REVISION)
AC_SUBST(LT_AGE)

dnl Detect the canonical build and host environments
dnl AC_CANONICAL_HOST

dnl Check for tools
AC_PROG_AWK
AC_PROG_CC
AC_PROG_CXX
AC_PROG_EGREP
AC_PROG_FGREP
AC_PROG_INSTALL
AC_PROG_MAKE_SET
PKG_PROG_PKG_CONFIG

if [ test -z "$AWK" ]; then
    AC_MSG_ERROR([*** awk not found, aborting])
fi

AC_CHECK_PROGS([SORT], [gsort sort], [false])
AS_IF([! "$SORT" -V </dev/null >/dev/null], [AC_MSG_WARN([sort(1) that supports the -V option is required to find dynamic libraries])])

dnl 64-bit file offsets if possible unless --disable-largefile is specified
AC_SYS_LARGEFILE

dnl Make sure that srcdir is a full pathname
case "$host" in
    *-*-mingw*)
        # Except on msys, where make can't handle full pathnames (bug 1972)
        ;;
    *)
        srcdir=`cd $srcdir && pwd`
        ;;
esac

dnl Set up the compiler and linker flags
INCLUDE="-I$srcdir/include"

dnl Don't use our khronos headers on QNX.
case "$host" in
    *-*-nto-qnx*)
        ;;
    *)
        INCLUDE="$INCLUDE -idirafter $srcdir/src/video/khronos"
        ;;
esac

dnl use CXX for linker on Haiku
case "$host" in
    *-*-haiku*)
        LINKER='$(CXX)'
        LIBTOOLLINKERTAG='CXX'
        ;;
    *)
        LINKER='$(CC)'
        LIBTOOLLINKERTAG='CC'
        ;;
esac
AC_SUBST(LINKER)
AC_SUBST(LIBTOOLLINKERTAG)

if test x$srcdir != x.; then
    INCLUDE="-Iinclude $INCLUDE"
elif test -d .git; then
    AC_MSG_ERROR([
*** When building from a git clone you should configure and build in a
    separate directory so you don't clobber SDL_config.h, SDL_revision.h
])
fi
BASE_CFLAGS=""
BASE_LDFLAGS=""
case "$host" in
    *-*-cygwin*)
        # We build SDL on cygwin without the UNIX emulation layer
        save_CFLAGS="$CFLAGS"
        have_no_cygwin=no
        AC_MSG_CHECKING(for GCC -mno-cygwin option)
        CFLAGS="$save_CFLAGS -mno-cygwin"

        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],[])], [have_no_cygwin=yes],[])
        AC_MSG_RESULT($have_no_cygwin)
        CFLAGS="$save_CFLAGS"

        if test x$have_no_cygwin = xyes; then
            BASE_CFLAGS="-mno-cygwin"
            BASE_LDFLAGS="-mno-cygwin"
        fi
        BASE_CFLAGS="$BASE_CFLAGS -I/usr/include/mingw"
        ;;
esac
# Uncomment the following line if you want to force SDL and applications
# built with it to be compiled for a particular architecture.
#AX_GCC_ARCHFLAG([no], [BASE_CFLAGS="$BASE_CFLAGS $ax_cv_gcc_archflag]")
BUILD_CFLAGS="$CFLAGS $CPPFLAGS -DUSING_GENERATED_CONFIG_H"
# The default optimization for SDL is -O3 (Bug #31)
if test "x$orig_CFLAGS" = x; then
    BUILD_CFLAGS=`echo $BUILD_CFLAGS | sed 's/-O2/-O3/'`
fi
EXTRA_CFLAGS="$INCLUDE $BASE_CFLAGS"
BUILD_LDFLAGS="$LDFLAGS"
EXTRA_LDFLAGS="$BASE_LDFLAGS"
## These are common directories to find software packages
#for path in /usr/freeware /usr/pkg /usr/X11R6 /usr/local; do
#    if test -d $path/include; then
#        EXTRA_CFLAGS="$EXTRA_CFLAGS -I$path/include"
#    fi
#    if test -d $path/lib; then
#        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -L$path/lib"
#    fi
#done
SDL_CFLAGS="$BASE_CFLAGS"
SDL_LIBS="-lSDL2"
if test "x$BASE_LDFLAGS" != x; then
    SDL_LIBS="$SDL_LIBS $BASE_LDFLAGS"
fi
if test "x$EXTRA_CFLAGS" != x; then
    CPPFLAGS="$CPPFLAGS $EXTRA_CFLAGS"
    CFLAGS="$CFLAGS $EXTRA_CFLAGS"
fi
if test "x$EXTRA_LDFLAGS" != x; then
    LDFLAGS="$LDFLAGS $EXTRA_LDFLAGS"
fi

dnl set this to use on systems that use lib64 instead of lib
base_libdir=`echo \${libdir} | sed 's/.*\/\(.*\)/\1/; q'`

dnl Function to find a library in the compiler search path
find_lib()
{
    gcc_bin_path=[`$CC -print-search-dirs 2>/dev/null | $FGREP programs: | sed 's/[^=]*=\(.*\)/\1/' | sed 's/:/ /g'`]
    gcc_lib_path=[`$CC -print-search-dirs 2>/dev/null | $FGREP libraries: | sed 's/[^=]*=\(.*\)/\1/' | sed 's/:/ /g'`]
    env_lib_path=[`echo $LIBS $LDFLAGS $* | sed 's/-L[ ]*//g'`]
    if test "$cross_compiling" = yes; then
        host_lib_path=""
    else
        host_lib_path="/usr/$base_libdir /usr/local/$base_libdir"
    fi
    for path in $env_lib_path $gcc_bin_path $gcc_lib_path $host_lib_path; do
        lib=[`ls -- $path/$1 2>/dev/null | sed 's,.*/,,' | "$SORT" -V -r | $AWK 'BEGIN{FS="."}{ print NF, $0 }' | "$SORT" -n -s | sed 's,[0-9]* ,,' | head -1`]
        if test x$lib != x; then
            echo $lib
            return
        fi
    done
}

dnl Check for compiler characteristics
AC_C_CONST
AC_C_INLINE
AC_C_VOLATILE

dnl See whether we want assertions for debugging/sanity checking SDL itself.
AC_ARG_ENABLE(assertions,
[AS_HELP_STRING([--enable-assertions],
               [Enable internal sanity checks (auto/disabled/release/enabled/paranoid) [default=auto]])],
              , enable_assertions=auto)
case "$enable_assertions" in
    auto) # Use optimization settings to determine assertion level
        ;;
    disabled)
        AC_DEFINE(SDL_DEFAULT_ASSERT_LEVEL, 0, [ ])
        ;;
    release)
        AC_DEFINE(SDL_DEFAULT_ASSERT_LEVEL, 1, [ ])
        ;;
    enabled)
        AC_DEFINE(SDL_DEFAULT_ASSERT_LEVEL, 2, [ ])
        ;;
    paranoid)
        AC_DEFINE(SDL_DEFAULT_ASSERT_LEVEL, 3, [ ])
        ;;
    *)
        AC_MSG_ERROR([*** unknown assertion level. stop.])
        ;;
esac

dnl For use in static assertions
EXTRA_CFLAGS="$EXTRA_CFLAGS -DSDL_BUILD_MAJOR_VERSION=$SDL_MAJOR_VERSION -DSDL_BUILD_MINOR_VERSION=$SDL_MINOR_VERSION -DSDL_BUILD_MICRO_VERSION=$SDL_MICRO_VERSION"

dnl See whether we can use gcc style dependency tracking
AC_ARG_ENABLE(dependency-tracking,
[AS_HELP_STRING([--enable-dependency-tracking],
               [Use gcc -MMD -MT dependency tracking [default=yes]])],
              , enable_dependency_tracking=yes)
if test x$enable_dependency_tracking = xyes; then
    have_gcc_mmd_mt=no
    AC_MSG_CHECKING(for GCC -MMD -MT option)
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
    #if !defined(__GNUC__) || __GNUC__ < 3
    #error Dependency tracking requires GCC 3.0 or newer
    #endif
    ]],[])], [have_gcc_mmd_mt=yes],[])
    AC_MSG_RESULT($have_gcc_mmd_mt)

    if test x$have_gcc_mmd_mt = xyes; then
        DEPENDENCY_TRACKING_OPTIONS="-MMD -MT \$@"
    fi
fi

AC_MSG_CHECKING(for linker option --no-undefined)
have_no_undefined=no
case "$host" in
dnl Skip this on platforms where it is just simply busted.
    *-*-openbsd*)
        ;;
    *)
        save_LDFLAGS="$LDFLAGS"
        LDFLAGS="$LDFLAGS -Wl,--no-undefined"
        AC_LINK_IFELSE([AC_LANG_PROGRAM([],[])], [
        have_no_undefined=yes
        BUILD_LDFLAGS="$BUILD_LDFLAGS -Wl,--no-undefined"
        ],[])
        LDFLAGS="$save_LDFLAGS"
        ;;
esac
AC_MSG_RESULT($have_no_undefined)

AC_MSG_CHECKING(for linker option --dynamicbase)
have_dynamicbase=no
case "$host" in
    *)
        save_LDFLAGS="$LDFLAGS"
        LDFLAGS="$LDFLAGS -Wl,--dynamicbase"
        AC_LINK_IFELSE([AC_LANG_PROGRAM([],[])], [
        have_dynamicbase=yes
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,--dynamicbase"
        ],[])
        LDFLAGS="$save_LDFLAGS"
        ;;
esac
AC_MSG_RESULT($have_dynamicbase)

AC_MSG_CHECKING(for linker option --nxcompat)
have_nxcompat=no
case "$host" in
    *)
        save_LDFLAGS="$LDFLAGS"
        LDFLAGS="$LDFLAGS -Wl,--nxcompat"
        AC_LINK_IFELSE([AC_LANG_PROGRAM([],[])], [
        have_nxcompat=yes
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,--nxcompat"
        ],[])
        LDFLAGS="$save_LDFLAGS"
        ;;
esac
AC_MSG_RESULT($have_nxcompat)

AC_MSG_CHECKING(for linker option --high-entropy-va)
have_high_entropy_va=no
case "$host" in
    *)
        save_LDFLAGS="$LDFLAGS"
        LDFLAGS="$LDFLAGS -Wl,--high-entropy-va"
        AC_LINK_IFELSE([AC_LANG_PROGRAM([],[])], [
        have_high_entropy_va=yes
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,--high-entropy-va"
        ],[])
        LDFLAGS="$save_LDFLAGS"
        ;;
esac
AC_MSG_RESULT($have_high_entropy_va)

dnl See whether we are allowed to use the system C library
AC_ARG_ENABLE(libc,
[AS_HELP_STRING([--enable-libc], [Use the system C library [default=yes]])],
              , enable_libc=yes)

dnl See whether we are allowed to use system iconv
enable_system_iconv_default=yes
case "$host" in
  *-*-cygwin*|*-*-mingw*|*-*-darwin*|*-ios-*)
    enable_system_iconv_default=no
    ;;
esac
AC_ARG_ENABLE(system-iconv,
[AS_HELP_STRING([--enable-system-iconv], [Use iconv() from system-installed libraries [default=no for windows, yes for others]])],
              , enable_system_iconv=$enable_system_iconv_default)
dnl See whether we prefer libiconv over libc
AC_ARG_ENABLE(libiconv,
[AS_HELP_STRING([--enable-libiconv], [Prefer iconv() from libiconv, if available, over libc version [default=no]])],
              , enable_libiconv=no)

if test x$enable_libc = xyes; then
    AC_DEFINE(HAVE_LIBC, 1, [ ])

dnl Check for C library headers
dnl AC_CHECK_INCLUDES_DEFAULT is an autoconf-2.7x thing where AC_HEADER_STDC is deprecated.
    m4_ifdef([AC_CHECK_INCLUDES_DEFAULT], [AC_CHECK_INCLUDES_DEFAULT], [AC_HEADER_STDC])
    AC_CHECK_HEADERS(sys/types.h stdio.h stdlib.h stddef.h stdarg.h malloc.h memory.h string.h strings.h wchar.h inttypes.h stdint.h limits.h ctype.h math.h float.h iconv.h signal.h)

dnl Check for typedefs, structures, etc.
    AC_TYPE_SIZE_T

dnl Check for defines
    AC_CHECK_DEFINE(M_PI, math.h)

    AC_FUNC_ALLOCA

dnl Checks for library functions.
    AC_CHECK_FUNC(mprotect,
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
          #include <sys/types.h>
          #include <sys/mman.h>
        ]],[])], [
        AC_DEFINE(HAVE_MPROTECT, 1, [ ])
        ],[]),
    )
    AC_CHECK_FUNCS(malloc calloc realloc free getenv setenv putenv unsetenv bsearch qsort abs bcopy memset memcmp memcpy memmove wcslen wcslcpy wcslcat _wcsdup wcsdup wcsstr wcscmp wcsncmp wcscasecmp _wcsicmp wcsncasecmp _wcsnicmp strlen strlcpy strlcat _strrev _strupr _strlwr index rindex strchr strrchr strstr strtok_r itoa _ltoa _uitoa _ultoa strtod strtol strtoul _i64toa _ui64toa strtoll strtoull atoi atof strcmp strncmp _stricmp strcasecmp _strnicmp strncasecmp strcasestr vsscanf vsnprintf fopen64 fseeko fseeko64 sigaction setjmp nanosleep sysconf sysctlbyname getauxval elf_aux_info poll memfd_create posix_fallocate _Exit)

    AC_CHECK_LIB(m, pow, [LIBS="$LIBS -lm"; EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lm"])
    AC_CHECK_FUNCS(acos acosf asin asinf atan atanf atan2 atan2f ceil ceilf copysign copysignf cos cosf exp expf fabs fabsf floor floorf trunc truncf fmod fmodf log logf log10 log10f lround lroundf pow powf round roundf scalbn scalbnf sin sinf sqrt sqrtf tan tanf)

    if test x$enable_system_iconv = xyes; then
        AC_MSG_CHECKING(for iconv in libc)
        AC_LINK_IFELSE([AC_LANG_PROGRAM([[
         #define LIBICONV_PLUG 1 /* in case libiconv header is in include path */
         #include <stddef.h>
         #include <iconv.h>
        ]],[[
         iconv_open(NULL,NULL);
        ]])], [have_libc_iconv=yes],[have_libc_iconv=no])
        AC_MSG_RESULT($have_libc_iconv)

        AC_MSG_CHECKING(for iconv in libiconv)
        save_LIBS="$LIBS"
        LIBS="$LIBS -liconv"
        AC_LINK_IFELSE([AC_LANG_PROGRAM([[
         #include <stddef.h>
         #include <iconv.h>
        ]],[[
         iconv_open(NULL,NULL);
        ]])], [have_libiconv=yes],[have_libiconv=no])
        LIBS="$save_LIBS"
        AC_MSG_RESULT($have_libiconv)

        if test x$have_libc_iconv = xyes || test x$have_libiconv = xyes; then
           AC_DEFINE(HAVE_ICONV,1,[ ])

           if test x$have_libiconv = xyes; then
              if test x$have_libc_iconv != xyes; then
                 use_libiconv=yes
              elif test x$enable_libiconv = xyes; then
                 use_libiconv=yes
              fi
           fi
           if test x$use_libiconv = xyes; then
              AC_DEFINE(SDL_USE_LIBICONV,1,[ ])
              EXTRA_LDFLAGS="$EXTRA_LDFLAGS -liconv"
              echo "-- Using iconv from libiconv"
           else
              echo "-- Using iconv from libc"
           fi
        fi
    fi

    AC_CHECK_MEMBER(struct sigaction.sa_sigaction,[AC_DEFINE([HAVE_SA_SIGACTION], 1, [ ])], ,[#include <signal.h>])

dnl Check for additional non-standard headers
    AC_CHECK_HEADERS(libunwind.h)
fi

dnl AC_CHECK_SIZEOF(void*)

dnl See whether we can use gcc atomic operations on this architecture
AC_ARG_ENABLE(gcc-atomics,
[AS_HELP_STRING([--enable-gcc-atomics],
               [Use gcc builtin atomics [default=yes]])],
              , enable_gcc_atomics=yes)
if test x$enable_gcc_atomics = xyes; then
    have_gcc_atomics=no
    AC_MSG_CHECKING(for GCC builtin atomic operations)
    AC_LINK_IFELSE([AC_LANG_PROGRAM([],[[
    int a;
    void *x, *y, *z;
    __sync_lock_test_and_set(&a, 4);
    __sync_lock_test_and_set(&x, y);
    __sync_fetch_and_add(&a, 1);
    __sync_bool_compare_and_swap(&a, 5, 10);
    __sync_bool_compare_and_swap(&x, y, z);
    ]])],
     [have_gcc_atomics=yes],[])
    AC_MSG_RESULT($have_gcc_atomics)

    if test x$have_gcc_atomics = xyes; then
        AC_DEFINE(HAVE_GCC_ATOMICS, 1, [ ])
    else
        # See if we have the minimum operation needed for GCC atomics
        AC_LINK_IFELSE([AC_LANG_PROGRAM([],[[
        int a;
        __sync_lock_test_and_set(&a, 1);
        __sync_lock_release(&a);
        ]])], [have_gcc_sync_lock_test_and_set=yes],[])
        if test x$have_gcc_sync_lock_test_and_set = xyes; then
            AC_DEFINE(HAVE_GCC_SYNC_LOCK_TEST_AND_SET, 1, [ ])
        fi
    fi
fi

# Standard C sources
SOURCES="$SOURCES $srcdir/src/*.c"
SOURCES="$SOURCES $srcdir/src/atomic/*.c"
SOURCES="$SOURCES $srcdir/src/audio/*.c"
SOURCES="$SOURCES $srcdir/src/cpuinfo/*.c"
SOURCES="$SOURCES $srcdir/src/dynapi/*.c"
SOURCES="$SOURCES $srcdir/src/events/*.c"
SOURCES="$SOURCES $srcdir/src/file/*.c"
SOURCES="$SOURCES $srcdir/src/haptic/*.c"
SOURCES="$SOURCES $srcdir/src/hidapi/*.c"
SOURCES="$SOURCES $srcdir/src/joystick/*.c"
SOURCES="$SOURCES $srcdir/src/libm/*.c"
SOURCES="$SOURCES $srcdir/src/misc/*.c"
SOURCES="$SOURCES $srcdir/src/power/*.c"
#SOURCES="$SOURCES $srcdir/src/filesystem/*.c"
SOURCES="$SOURCES $srcdir/src/render/*.c"
SOURCES="$SOURCES $srcdir/src/render/*/*.c"
SOURCES="$SOURCES $srcdir/src/sensor/*.c"
SOURCES="$SOURCES $srcdir/src/stdlib/*.c"
SOURCES="$SOURCES $srcdir/src/thread/*.c"
SOURCES="$SOURCES $srcdir/src/timer/*.c"
SOURCES="$SOURCES $srcdir/src/video/*.c"
SOURCES="$SOURCES $srcdir/src/video/yuv2rgb/*.c"
SOURCES="$SOURCES $srcdir/src/locale/*.c"

dnl Enable/disable various subsystems of the SDL library

case "$host" in
    *-*-emscripten*)
        default_atomic=no
        ;;
    *)
        default_atomic=yes
        ;;
esac

AC_ARG_ENABLE(atomic,
[AS_HELP_STRING([--enable-atomic], [Enable the atomic operations subsystem [default=yes]])],
              , enable_atomic=$default_atomic)
if test x$enable_atomic != xyes; then
    AC_DEFINE(SDL_ATOMIC_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} atomic"
fi
AC_ARG_ENABLE(audio,
[AS_HELP_STRING([--enable-audio], [Enable the audio subsystem [default=yes]])],
              , enable_audio=yes)
if test x$enable_audio != xyes; then
    AC_DEFINE(SDL_AUDIO_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} audio"
fi
AC_ARG_ENABLE(video,
[AS_HELP_STRING([--enable-video], [Enable the video subsystem [default=yes]])],
              , enable_video=yes)
if test x$enable_video != xyes; then
    AC_DEFINE(SDL_VIDEO_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} video"
fi
AC_ARG_ENABLE(render,
[AS_HELP_STRING([--enable-render], [Enable the render subsystem [default=yes]])],
              , enable_render=yes)
if test x$enable_render != xyes; then
    AC_DEFINE(SDL_RENDER_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} render"
fi
AC_ARG_ENABLE(events,
[AS_HELP_STRING([--enable-events], [Enable the events subsystem [default=yes]])],
              , enable_events=yes)
if test x$enable_events != xyes; then
    AC_DEFINE(SDL_EVENTS_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} events"
fi
AC_ARG_ENABLE(joystick,
[AS_HELP_STRING([--enable-joystick], [Enable the joystick subsystem [default=yes]])],
              , enable_joystick=yes)
if test x$enable_joystick != xyes; then
    AC_DEFINE(SDL_JOYSTICK_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} joystick"
fi
AC_ARG_ENABLE(haptic,
[AS_HELP_STRING([--enable-haptic], [Enable the haptic (force feedback) subsystem [default=yes]])],
              , enable_haptic=yes)
if test x$enable_haptic != xyes; then
    AC_DEFINE(SDL_HAPTIC_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} haptic"
fi
AC_ARG_ENABLE(hidapi,
[AS_HELP_STRING([--enable-hidapi], [Enable the HIDAPI subsystem [default=yes]])],
              , enable_hidapi=yes)
if test x$enable_hidapi != xyes; then
    AC_DEFINE(SDL_HIDAPI_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} hidapi"
fi
AC_ARG_ENABLE(sensor,
[AS_HELP_STRING([--enable-sensor], [Enable the sensor subsystem [default=yes]])],
              , enable_sensor=yes)
if test x$enable_sensor != xyes; then
    AC_DEFINE(SDL_SENSOR_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} sensor"
fi
AC_ARG_ENABLE(power,
[AS_HELP_STRING([--enable-power], [Enable the power subsystem [default=yes]])],
              , enable_power=yes)
if test x$enable_power != xyes; then
    AC_DEFINE(SDL_POWER_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} power"
fi
AC_ARG_ENABLE(filesystem,
[AS_HELP_STRING([--enable-filesystem], [Enable the filesystem subsystem [default=yes]])],
              , enable_filesystem=yes)
if test x$enable_filesystem != xyes; then
    AC_DEFINE(SDL_FILESYSTEM_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} filesystem"
fi
# Many subsystems depend on threads, so leave them enabled by default
#AC_ARG_ENABLE(threads,
#[AS_HELP_STRING([--enable-threads], [Enable the threading subsystem [default=yes]])],
#              , enable_threads=yes)
enable_threads=yes
if test x$enable_threads != xyes; then
    AC_DEFINE(SDL_THREADS_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} threads"
fi
AC_ARG_ENABLE(timers,
[AS_HELP_STRING([--enable-timers], [Enable the timer subsystem [default=yes]])],
              , enable_timers=yes)
if test x$enable_timers != xyes; then
    AC_DEFINE(SDL_TIMERS_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} timers"
fi
AC_ARG_ENABLE(file,
[AS_HELP_STRING([--enable-file], [Enable the file subsystem [default=yes]])],
              , enable_file=yes)
if test x$enable_file != xyes; then
    AC_DEFINE(SDL_FILE_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} file"
fi
AC_ARG_ENABLE(misc,
[AS_HELP_STRING([--enable-misc], [Enable the misc subsystem [default=yes]])],
              , enable_misc=yes)
if test x$enable_misc != xyes; then
    AC_DEFINE(SDL_MISC_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} misc"
fi
AC_ARG_ENABLE(locale,
[AS_HELP_STRING([--enable-locale], [Enable the locale subsystem [default=yes]])],
              , enable_locale=yes)
if test x$enable_locale != xyes; then
    AC_DEFINE(SDL_LOCALE_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} locale"
fi
AC_ARG_ENABLE(loadso,
[AS_HELP_STRING([--enable-loadso], [Enable the shared object loading subsystem [default=yes]])],
              , enable_loadso=yes)
if test x$enable_loadso != xyes; then
    AC_DEFINE(SDL_LOADSO_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} loadso"
fi
AC_ARG_ENABLE(cpuinfo,
[AS_HELP_STRING([--enable-cpuinfo], [Enable the cpuinfo subsystem [default=yes]])],
              , enable_cpuinfo=yes)
if test x$enable_cpuinfo != xyes; then
    AC_DEFINE(SDL_CPUINFO_DISABLED, 1, [ ])
else
    SUMMARY_modules="${SUMMARY_modules} cpuinfo"
fi
AC_ARG_ENABLE(assembly,
[AS_HELP_STRING([--enable-assembly], [Enable assembly routines [default=yes]])],
              , enable_assembly=yes)
if test x$enable_assembly = xyes; then
    SUMMARY_modules="${SUMMARY_modules} assembly"

    # Make sure that we don't generate floating point code that would
    # cause illegal instruction exceptions on older processors
    case "$host" in
        *-*-darwin*)
            # Don't need to worry about Apple hardware, it's all SSE capable
            default_ssemath=yes
            ;;
        *64-*-*)
            # x86 64-bit architectures all have SSE instructions
            default_ssemath=yes
            ;;
        *)
            default_ssemath=no
            ;;
    esac
    AC_ARG_ENABLE(ssemath,
[AS_HELP_STRING([--enable-ssemath], [Allow GCC to use SSE floating point math [default=maybe]])],
                  , enable_ssemath=$default_ssemath)
    if test x$enable_ssemath = xno; then
        if test x$have_gcc_sse = xyes -o x$have_gcc_sse2 = xyes -o x$have_gcc_sse3 = xyes; then
            EXTRA_CFLAGS="$EXTRA_CFLAGS -mfpmath=387"
        fi
    fi

dnl Check for various instruction support
    AC_ARG_ENABLE(mmx,
[AS_HELP_STRING([--enable-mmx], [use MMX assembly routines [default=yes]])],
                  , enable_mmx=yes)
    if test x$enable_mmx = xyes; then
        save_CFLAGS="$CFLAGS"
        have_gcc_mmx=no
        AC_MSG_CHECKING(for GCC -mmmx option)
        mmx_CFLAGS="-mmmx"
        CFLAGS="$save_CFLAGS $mmx_CFLAGS"

        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
        #ifdef __MINGW32__
        #include <_mingw.h>
        #ifdef __MINGW64_VERSION_MAJOR
        #include <intrin.h>
        #else
        #include <mmintrin.h>
        #endif
        #else
        #include <mmintrin.h>
        #endif
        #ifndef __MMX__
        #error Assembler CPP flag not enabled
        #endif
        ]],[])], [have_gcc_mmx=yes],[])
        AC_MSG_RESULT($have_gcc_mmx)
        CFLAGS="$save_CFLAGS"

        if test x$have_gcc_mmx = xyes; then
            EXTRA_CFLAGS="$EXTRA_CFLAGS $mmx_CFLAGS"
            SUMMARY_math="${SUMMARY_math} mmx"
        fi
    fi

    AC_ARG_ENABLE(3dnow,
[AS_HELP_STRING([--enable-3dnow], [use 3DNow! assembly routines [default=no]])],
                  , enable_3dnow=no)
    if test x$enable_3dnow = xyes; then
        save_CFLAGS="$CFLAGS"
        have_gcc_3dnow=no
        AC_MSG_CHECKING(for GCC -m3dnow option)
        amd3dnow_CFLAGS="-m3dnow"
        CFLAGS="$save_CFLAGS $amd3dnow_CFLAGS"

        AC_LINK_IFELSE([AC_LANG_PROGRAM([[
        #include <mm3dnow.h>
        #ifndef __3dNOW__
        #error Assembler CPP flag not enabled
        #endif
        ]], [[
        void *p = 0;
        _m_prefetch(p);
        ]])],
         [have_gcc_3dnow=yes],[])
        AC_MSG_RESULT($have_gcc_3dnow)
        CFLAGS="$save_CFLAGS"

        if test x$have_gcc_3dnow = xyes; then
            EXTRA_CFLAGS="$EXTRA_CFLAGS $amd3dnow_CFLAGS"
            SUMMARY_math="${SUMMARY_math} 3dnow"
        fi
    fi

    AC_ARG_ENABLE(sse,
[AS_HELP_STRING([--enable-sse], [use SSE assembly routines [default=yes]])],
                  , enable_sse=yes)
    if test x$enable_sse = xyes; then
        save_CFLAGS="$CFLAGS"
        have_gcc_sse=no
        AC_MSG_CHECKING(for GCC -msse option)
        sse_CFLAGS="-msse"
        CFLAGS="$save_CFLAGS $sse_CFLAGS"

        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
        #ifdef __MINGW32__
        #include <_mingw.h>
        #ifdef __MINGW64_VERSION_MAJOR
        #include <intrin.h>
        #else
        #include <xmmintrin.h>
        #endif
        #else
        #include <xmmintrin.h>
        #endif
        #ifndef __SSE__
        #error Assembler CPP flag not enabled
        #endif
        ]],[])], [have_gcc_sse=yes],[])
        AC_MSG_RESULT($have_gcc_sse)
        CFLAGS="$save_CFLAGS"

        if test x$have_gcc_sse = xyes; then
            EXTRA_CFLAGS="$EXTRA_CFLAGS $sse_CFLAGS"
            SUMMARY_math="${SUMMARY_math} sse"
        fi
    fi

    AC_ARG_ENABLE(sse2,
[AS_HELP_STRING([--enable-sse2], [use SSE2 assembly routines [default=maybe]])],
                  , enable_sse2=$default_ssemath)
    if test x$enable_sse2 = xyes; then
        save_CFLAGS="$CFLAGS"
        have_gcc_sse2=no
        AC_MSG_CHECKING(for GCC -msse2 option)
        sse2_CFLAGS="-msse2"
        CFLAGS="$save_CFLAGS $sse2_CFLAGS"

        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
        #ifdef __MINGW32__
        #include <_mingw.h>
        #ifdef __MINGW64_VERSION_MAJOR
        #include <intrin.h>
        #else
        #include <emmintrin.h>
        #endif
        #else
        #include <emmintrin.h>
        #endif
        #ifndef __SSE2__
        #error Assembler CPP flag not enabled
        #endif
        ]],[])], [have_gcc_sse2=yes],[])
        AC_MSG_RESULT($have_gcc_sse2)
        CFLAGS="$save_CFLAGS"

        if test x$have_gcc_sse2 = xyes; then
            EXTRA_CFLAGS="$EXTRA_CFLAGS $sse2_CFLAGS"
            SUMMARY_math="${SUMMARY_math} sse2"
        fi
    fi

    AC_ARG_ENABLE(sse3,
[AS_HELP_STRING([--enable-sse3], [use SSE3 assembly routines [default=maybe]])],
                  , enable_sse3=$default_ssemath)
    if test x$enable_sse3 = xyes; then
        save_CFLAGS="$CFLAGS"
        have_gcc_sse3=no
        AC_MSG_CHECKING(for GCC -msse3 option)
        sse3_CFLAGS="-msse3"
        CFLAGS="$save_CFLAGS $sse3_CFLAGS"

        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
        #ifdef __MINGW32__
        #include <_mingw.h>
        #ifdef __MINGW64_VERSION_MAJOR
        #include <intrin.h>
        #else
        #include <pmmintrin.h>
        #endif
        #else
        #include <pmmintrin.h>
        #endif
        #ifndef __SSE2__
        #error Assembler CPP flag not enabled
        #endif
        ]],[])], [have_gcc_sse3=yes],[])
        AC_MSG_RESULT($have_gcc_sse3)
        CFLAGS="$save_CFLAGS"

        if test x$have_gcc_sse3 = xyes; then
            EXTRA_CFLAGS="$EXTRA_CFLAGS $sse3_CFLAGS"
            SUMMARY_math="${SUMMARY_math} sse3"
        fi
    fi

    AC_MSG_CHECKING(for immintrin.h)
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <immintrin.h>]])],
       [have_immintrin_h_hdr=yes],[have_immintrin_h_hdr=no])
    AC_MSG_RESULT($have_immintrin_h_hdr)
    if test x$have_immintrin_h_hdr = xyes; then
        AC_DEFINE(HAVE_IMMINTRIN_H, 1, [ ])
    fi

    AC_ARG_ENABLE(altivec,
[AS_HELP_STRING([--enable-altivec], [use Altivec assembly routines [default=yes]])],
                  , enable_altivec=yes)
    if test x$enable_altivec = xyes; then
        save_CFLAGS="$CFLAGS"
        have_gcc_altivec=no
        have_altivec_h_hdr=no
        altivec_CFLAGS="-maltivec"
        CFLAGS="$save_CFLAGS $altivec_CFLAGS"

        AC_MSG_CHECKING(for Altivec with GCC altivec.h and -maltivec option)
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
        #include <altivec.h>
        vector unsigned int vzero() {
            return vec_splat_u32(0);
        }
        ]],[])], [
        have_gcc_altivec=yes
        have_altivec_h_hdr=yes
        ],[])
        AC_MSG_RESULT($have_gcc_altivec)

        if test x$have_gcc_altivec = xno; then
            AC_MSG_CHECKING(for Altivec with GCC -maltivec option)
            AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
            vector unsigned int vzero() {
                return vec_splat_u32(0);
            }
            ]],[])], [have_gcc_altivec=yes],[])
            AC_MSG_RESULT($have_gcc_altivec)
        fi

        if test x$have_gcc_altivec = xno; then
            AC_MSG_CHECKING(for Altivec with GCC altivec.h and -faltivec option)
            altivec_CFLAGS="-faltivec"
            CFLAGS="$save_CFLAGS $altivec_CFLAGS"
            AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
            #include <altivec.h>
            vector unsigned int vzero() {
                return vec_splat_u32(0);
            }
            ]],[])], [
            have_gcc_altivec=yes
            have_altivec_h_hdr=yes
            ],[])
            AC_MSG_RESULT($have_gcc_altivec)
        fi

        if test x$have_gcc_altivec = xno; then
            AC_MSG_CHECKING(for Altivec with GCC -faltivec option)
            AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
            vector unsigned int vzero() {
                return vec_splat_u32(0);
            }
            ]],[])], [have_gcc_altivec=yes],[])
            AC_MSG_RESULT($have_gcc_altivec)
        fi
        CFLAGS="$save_CFLAGS"

        if test x$have_gcc_altivec = xyes; then
            AC_DEFINE(SDL_ALTIVEC_BLITTERS, 1, [ ])
            if test x$have_altivec_h_hdr = xyes; then
              AC_DEFINE(HAVE_ALTIVEC_H, 1, [ ])
            fi
            EXTRA_CFLAGS="$EXTRA_CFLAGS $altivec_CFLAGS"
            SUMMARY_math="${SUMMARY_math} altivec"
        fi
    fi
fi

AC_ARG_ENABLE(lsx,
[AS_HELP_STRING([--enable-lsx], [use LSX assembly routines [default=yes]])],
                  , enable_lsx=yes)
if test x$enable_lsx = xyes; then
    save_CFLAGS="$CFLAGS"
    have_gcc_lsx=no
    AC_MSG_CHECKING(for GCC -mlsx option)
    lsx_CFLAGS="-mlsx"
    CFLAGS="$save_CFLAGS $lsx_CFLAGS"

    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
    #ifndef __loongarch_sx
    #error Assembler CPP flag not enabled
    #endif
    ]], [])], [have_gcc_lsx=yes], [])
    AC_MSG_RESULT($have_gcc_lsx)
    CFLAGS="$save_CFLAGS"

    if test x$have_gcc_lsx = xyes; then
       EXTRA_CFLAGS="$EXTRA_CFLAGS $lsx_CFLAGS"
       SUMMARY_math="${SUMMARY_math} lsx"
    fi
fi

AC_MSG_CHECKING(for lsxintrin.h)
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <lsxintrin.h>]])],
   [have_lsxintrin_h_hdr=yes],[have_lsxintrin_h_hdr=no])
AC_MSG_RESULT($have_lsxintrin_h_hdr)
if test x$have_lsxintrin_h_hdr = xyes; then
   AC_DEFINE(HAVE_LSXINTRIN_H, 1, [ ])
fi

dnl See if the OSS audio interface is supported
CheckOSS()
{
    AC_ARG_ENABLE(oss,
[AS_HELP_STRING([--enable-oss], [support the OSS audio API [default=maybe]])],
                  , enable_oss=maybe)
    if test x$enable_oss = xmaybe; then
        enable_oss=yes
    fi

    if test x$enable_audio = xyes -a x$enable_oss = xyes; then
        AC_MSG_CHECKING(for OSS audio support)
        have_oss=no
        if test x$have_oss != xyes; then
            AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
              #include <sys/soundcard.h>
            ]], [[
              int arg = SNDCTL_DSP_SETFRAGMENT;
            ]])], [have_oss=yes],[])
        fi
        AC_MSG_RESULT($have_oss)
        if test x$have_oss = xyes; then
            SUMMARY_audio="${SUMMARY_audio} oss"
            AC_DEFINE(SDL_AUDIO_DRIVER_OSS, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/audio/dsp/*.c"
            have_audio=yes

            # We may need to link with ossaudio emulation library
            case "$host" in
                *-*-netbsd*)
                    EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lossaudio";;
            esac
        fi
    fi
}

dnl See if the ALSA audio interface is supported
CheckALSA()
{
    AC_ARG_ENABLE(alsa,
[AS_HELP_STRING([--enable-alsa], [support the ALSA audio API [default=yes]])],
                  , enable_alsa=yes)
    if test x$enable_audio = xyes -a x$enable_alsa = xyes; then
        AM_PATH_ALSA(1.0.11, have_alsa=yes, have_alsa=no)
        # Restore all flags from before the ALSA detection runs
        CFLAGS="$alsa_save_CFLAGS"
        LDFLAGS="$alsa_save_LDFLAGS"
        LIBS="$alsa_save_LIBS"
        if test x$have_alsa = xyes; then
            AC_ARG_ENABLE(alsa-shared,
[AS_HELP_STRING([--enable-alsa-shared], [dynamically load ALSA audio support [default=yes]])],
                          , enable_alsa_shared=yes)
            alsa_lib=[`find_lib "libasound.so.*" "$ALSA_LIBS" | sed 's/.*\/\(.*\)/\1/; q'`]

            AC_DEFINE(SDL_AUDIO_DRIVER_ALSA, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/audio/alsa/*.c"
            EXTRA_CFLAGS="$EXTRA_CFLAGS $ALSA_CFLAGS"
            if test x$have_loadso != xyes && \
               test x$enable_alsa_shared = xyes; then
                AC_MSG_WARN([You must have SDL_LoadObject() support for dynamic ALSA loading])
            fi
            if test x$have_loadso = xyes && \
               test x$enable_alsa_shared = xyes && test x$alsa_lib != x; then
                echo "-- dynamic libasound -> $alsa_lib"
                AC_DEFINE_UNQUOTED(SDL_AUDIO_DRIVER_ALSA_DYNAMIC, "$alsa_lib", [ ])
                SUMMARY_audio="${SUMMARY_audio} alsa(dynamic)"
            else
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS $ALSA_LIBS"
                SUMMARY_audio="${SUMMARY_audio} alsa"
            fi
            have_audio=yes
        fi
    fi
}

dnl Find JACK Audio
CheckJACK()
{
    AC_ARG_ENABLE(jack,
[AS_HELP_STRING([--enable-jack], [use JACK audio [default=yes]])],
                  , enable_jack=yes)
    if test x$enable_audio = xyes -a x$enable_jack = xyes; then
        PKG_CHECK_MODULES([JACK], [jack >= 0.125], audio_jack=yes, audio_jack=no)

        if test x$audio_jack = xyes; then
            AC_ARG_ENABLE(jack-shared,
[AS_HELP_STRING([--enable-jack-shared], [dynamically load JACK audio support [default=yes]])],
                          , enable_jack_shared=yes)
            jack_lib=[`find_lib "libjack.so.*" "$JACK_LIBS" | sed 's/.*\/\(.*\)/\1/; q'`]

            AC_DEFINE(SDL_AUDIO_DRIVER_JACK, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/audio/jack/*.c"
            EXTRA_CFLAGS="$EXTRA_CFLAGS $JACK_CFLAGS"
            if test x$have_loadso != xyes && \
               test x$enable_jack_shared = xyes; then
                AC_MSG_WARN([You must have SDL_LoadObject() support for dynamic JACK audio loading])
            fi
            if test x$have_loadso = xyes && \
               test x$enable_jack_shared = xyes && test x$jack_lib != x; then
                echo "-- dynamic libjack -> $jack_lib"
                AC_DEFINE_UNQUOTED(SDL_AUDIO_DRIVER_JACK_DYNAMIC, "$jack_lib", [ ])
                SUMMARY_audio="${SUMMARY_audio} jack(dynamic)"

                case "$host" in
                    # On Solaris, jack must be linked deferred explicitly
                    # to prevent undefined symbol failures.
                    *-*-solaris*)
                        JACK_LIBS=`echo $JACK_LIBS | sed 's/\-l/-Wl,-l/g'`
                        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-zdeferred $JACK_LIBS -Wl,-znodeferred"
                esac
            else
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS $JACK_LIBS"
                SUMMARY_audio="${SUMMARY_audio} jack"
            fi
            have_audio=yes
        fi
    fi
}

dnl Find the ESD includes and libraries
CheckESD()
{
    AC_ARG_ENABLE(esd,
[AS_HELP_STRING([--enable-esd], [support the Enlightened Sound Daemon [default=yes]])],
                  , enable_esd=yes)
    if test x$enable_audio = xyes -a x$enable_esd = xyes; then
        PKG_CHECK_MODULES([ESD], [esound >= 0.2.8], have_esd=yes, have_esd=no)
        if test x$have_esd = xno; then
           AM_PATH_ESD(0.2.8, have_esd=yes, have_esd=no)
        fi
        if test x$have_esd = xyes; then
            AC_ARG_ENABLE(esd-shared,
[AS_HELP_STRING([--enable-esd-shared], [dynamically load ESD audio support [default=yes]])],
                          , enable_esd_shared=yes)
            esd_lib=[`find_lib "libesd.so.*" "$ESD_LIBS" | sed 's/.*\/\(.*\)/\1/; q'`]

            AC_DEFINE(SDL_AUDIO_DRIVER_ESD, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/audio/esd/*.c"
            EXTRA_CFLAGS="$EXTRA_CFLAGS $ESD_CFLAGS"
            if test x$have_loadso != xyes && \
               test x$enable_esd_shared = xyes; then
                AC_MSG_WARN([You must have SDL_LoadObject() support for dynamic ESD loading])
            fi
            if test x$have_loadso = xyes && \
               test x$enable_esd_shared = xyes && test x$esd_lib != x; then
                echo "-- dynamic libesd -> $esd_lib"
                AC_DEFINE_UNQUOTED(SDL_AUDIO_DRIVER_ESD_DYNAMIC, "$esd_lib", [ ])
                SUMMARY_audio="${SUMMARY_audio} esd(dynamic)"
            else
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS $ESD_LIBS"
                SUMMARY_audio="${SUMMARY_audio} esd"
            fi
            have_audio=yes
        fi
    fi
}

dnl Find Pipewire
CheckPipewire()
{
    AC_ARG_ENABLE(pipewire,
[AS_HELP_STRING([--enable-pipewire], [use Pipewire audio [default=yes]])],
                  , enable_pipewire=yes)
    if test x$enable_audio = xyes -a x$enable_pipewire = xyes; then
        PKG_CHECK_MODULES([PIPEWIRE], [libpipewire-0.3 >= 0.3.20], audio_pipewire=yes, audio_pipewire=no)

        if test x$audio_pipewire = xyes; then
            AC_ARG_ENABLE(pipewire-shared,
[AS_HELP_STRING([--enable-pipewire-shared], [dynamically load Pipewire support [default=yes]])],
                          , enable_pipewire_shared=yes)
            pipewire_lib=[`find_lib "libpipewire-0.3.so.*" "$PIPEWIRE_LIBS" | sed 's/.*\/\(.*\)/\1/; q'`]

            AC_DEFINE(SDL_AUDIO_DRIVER_PIPEWIRE, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/audio/pipewire/*.c"
            EXTRA_CFLAGS="$EXTRA_CFLAGS $PIPEWIRE_CFLAGS"
            if test x$have_loadso != xyes && \
               test x$enable_pipewire_shared = xyes; then
                AC_MSG_WARN([You must have SDL_LoadObject() support for dynamic Pipewire loading])
            fi
            if test x$have_loadso = xyes && \
               test x$enable_pipewire_shared = xyes && test x$pipewire_lib != x; then
                echo "-- dynamic libpipewire-0.3 -> $pipewire_lib"
                AC_DEFINE_UNQUOTED(SDL_AUDIO_DRIVER_PIPEWIRE_DYNAMIC, "$pipewire_lib", [ ])
                SUMMARY_audio="${SUMMARY_audio} pipewire(dynamic)"
            else
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS $PIPEWIRE_LIBS"
                SUMMARY_audio="${SUMMARY_audio} pipewire"
            fi
            have_audio=yes
        fi
    fi
}

dnl Find PulseAudio
CheckPulseAudio()
{
    AC_ARG_ENABLE(pulseaudio,
[AS_HELP_STRING([--enable-pulseaudio], [use PulseAudio [default=yes]])],
                  , enable_pulseaudio=yes)
    if test x$enable_audio = xyes -a x$enable_pulseaudio = xyes; then
        PKG_CHECK_MODULES([PULSEAUDIO], [libpulse >= 0.9.15], audio_pulseaudio=yes, audio_pulseaudio=no)

        if test x$audio_pulseaudio = xyes; then
            AC_ARG_ENABLE(pulseaudio-shared,
[AS_HELP_STRING([--enable-pulseaudio-shared], [dynamically load PulseAudio support [default=yes]])],
                          , enable_pulseaudio_shared=yes)
            pulseaudio_lib=[`find_lib "libpulse.so.*" "$PULSEAUDIO_LIBS" | sed 's/.*\/\(.*\)/\1/; q'`]

            AC_DEFINE(SDL_AUDIO_DRIVER_PULSEAUDIO, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/audio/pulseaudio/*.c"
            EXTRA_CFLAGS="$EXTRA_CFLAGS $PULSEAUDIO_CFLAGS"
            if test x$have_loadso != xyes && \
               test x$enable_pulseaudio_shared = xyes; then
                AC_MSG_WARN([You must have SDL_LoadObject() support for dynamic PulseAudio loading])
            fi
            if test x$have_loadso = xyes && \
               test x$enable_pulseaudio_shared = xyes && test x$pulseaudio_lib != x; then
                echo "-- dynamic libpulse -> $pulseaudio_lib"
                AC_DEFINE_UNQUOTED(SDL_AUDIO_DRIVER_PULSEAUDIO_DYNAMIC, "$pulseaudio_lib", [ ])
                SUMMARY_audio="${SUMMARY_audio} pulse(dynamic)"

                case "$host" in
                    # On Solaris, pulseaudio must be linked deferred explicitly
                    # to prevent undefined symbol failures.
                    *-*-solaris*)
                        PULSEAUDIO_LIBS=`echo $PULSEAUDIO_LIBS | sed 's/\-l/-Wl,-l/g'`
                        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-zdeferred $PULSEAUDIO_LIBS -Wl,-znodeferred"
                esac
            else
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS $PULSEAUDIO_LIBS"
                SUMMARY_audio="${SUMMARY_audio} pulse"
            fi
            have_audio=yes
        fi
    fi
}

CheckARTSC()
{
    AC_ARG_ENABLE(arts,
[AS_HELP_STRING([--enable-arts], [support the Analog Real Time Synthesizer [default=yes]])],
                  , enable_arts=yes)
    if test x$enable_audio = xyes -a x$enable_arts = xyes; then
        AC_PATH_PROG(ARTSCONFIG, artsc-config)
        if test x$ARTSCONFIG = x -o x$ARTSCONFIG = x'"$ARTSCONFIG"'; then
            : # arts isn't installed
        else
            ARTS_CFLAGS=`$ARTSCONFIG --cflags`
            ARTS_LIBS=`$ARTSCONFIG --libs`
            AC_MSG_CHECKING(for aRts development environment)
            audio_arts=no
            save_CFLAGS="$CFLAGS"
            CFLAGS="$CFLAGS $ARTS_CFLAGS"
            AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
             #include <artsc.h>
            ]], [[
             arts_stream_t stream;
            ]])], [audio_arts=yes],[])
            CFLAGS="$save_CFLAGS"
            AC_MSG_RESULT($audio_arts)
            if test x$audio_arts = xyes; then
                AC_ARG_ENABLE(arts-shared,
[AS_HELP_STRING([--enable-arts-shared], [dynamically load aRts audio support [default=yes]])],
                              , enable_arts_shared=yes)
                arts_lib=[`find_lib "libartsc.so.*" "$ARTS_LIBS" | sed 's/.*\/\(.*\)/\1/; q'`]

                AC_DEFINE(SDL_AUDIO_DRIVER_ARTS, 1, [ ])
                SOURCES="$SOURCES $srcdir/src/audio/arts/*.c"
                EXTRA_CFLAGS="$EXTRA_CFLAGS $ARTS_CFLAGS"
                if test x$have_loadso != xyes && \
                   test x$enable_arts_shared = xyes; then
                    AC_MSG_WARN([You must have SDL_LoadObject() support for dynamic ARTS loading])
                fi
                if test x$have_loadso = xyes && \
                   test x$enable_arts_shared = xyes && test x$arts_lib != x; then
                    echo "-- dynamic libartsc -> $arts_lib"
                    AC_DEFINE_UNQUOTED(SDL_AUDIO_DRIVER_ARTS_DYNAMIC, "$arts_lib", [ ])
                    SUMMARY_audio="${SUMMARY_audio} arts(dynamic)"
                else
                    EXTRA_LDFLAGS="$EXTRA_LDFLAGS $ARTS_LIBS"
                    SUMMARY_audio="${SUMMARY_audio} arts"
                fi
                have_audio=yes
            fi
        fi
    fi
}

dnl See if the NAS audio interface is supported
CheckNAS()
{
    AC_ARG_ENABLE(nas,
[AS_HELP_STRING([--enable-nas], [support the NAS audio API [default=yes]])],
                  , enable_nas=yes)
    if test x$enable_audio = xyes -a x$enable_nas = xyes; then
        AC_CHECK_HEADER(audio/audiolib.h, have_nas_hdr=yes)
        AC_CHECK_LIB(audio, AuOpenServer, have_nas_lib=yes)

        AC_MSG_CHECKING(for NAS audio support)
        have_nas=no

        if test x$have_nas_hdr = xyes -a x$have_nas_lib = xyes; then
            have_nas=yes
            NAS_LIBS="-laudio"

        elif test -r /usr/X11R6/include/audio/audiolib.h; then
            have_nas=yes
            NAS_CFLAGS="-I/usr/X11R6/include/"
            NAS_LIBS="-L/usr/X11R6/lib -laudio -lXt"

        fi

        AC_MSG_RESULT($have_nas)

        if test x$have_nas = xyes; then
            AC_ARG_ENABLE(nas-shared,
[AS_HELP_STRING([--enable-nas-shared], [dynamically load NAS audio support [default=yes]])],
                          , enable_nas_shared=yes)
            nas_lib=[`find_lib "libaudio.so.*" "$NAS_LIBS" | sed 's/.*\/\(.*\)/\1/; q'`]

            if test x$have_loadso != xyes && \
               test x$enable_nas_shared = xyes; then
                AC_MSG_WARN([You must have SDL_LoadObject() support for dynamic NAS loading])
            fi
            if test x$have_loadso = xyes && \
               test x$enable_nas_shared = xyes && test x$nas_lib != x; then
                echo "-- dynamic libaudio -> $nas_lib"
                AC_DEFINE_UNQUOTED(SDL_AUDIO_DRIVER_NAS_DYNAMIC, "$nas_lib", [ ])
                SUMMARY_audio="${SUMMARY_audio} nas(dynamic)"
            else
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS $NAS_LIBS"
                SUMMARY_audio="${SUMMARY_audio} nas"
            fi

            AC_DEFINE(SDL_AUDIO_DRIVER_NAS, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/audio/nas/*.c"
            EXTRA_CFLAGS="$EXTRA_CFLAGS $NAS_CFLAGS"
            have_audio=yes
        fi
    fi
}

dnl See if the sndio audio interface is supported
CheckSNDIO()
{
    AC_ARG_ENABLE(sndio,
[AS_HELP_STRING([--enable-sndio], [support the sndio audio API [default=yes]])],
                  , enable_sndio=yes)
    if test x$enable_audio = xyes -a x$enable_sndio = xyes; then
        PKG_CHECK_MODULES([SNDIO], [sndio], audio_sndio=yes, audio_sndio=no)

        if test x$audio_sndio = xyes; then
            AC_ARG_ENABLE(sndio-shared,
[AS_HELP_STRING([--enable-sndio-shared], [dynamically load sndio audio support [default=yes]])],
                          , enable_sndio_shared=yes)
            sndio_lib=[`find_lib "libsndio.so.*" "$SNDIO_LIBS" | sed 's/.*\/\(.*\)/\1/; q'`]

            if test x$have_loadso != xyes && \
               test x$enable_sndio_shared = xyes; then
                AC_MSG_WARN([You must have SDL_LoadObject() support for dynamic sndio loading])
            fi
            if test x$have_loadso = xyes && \
               test x$enable_sndio_shared = xyes && test x$sndio_lib != x; then
                echo "-- dynamic libsndio -> $sndio_lib"
                AC_DEFINE_UNQUOTED(SDL_AUDIO_DRIVER_SNDIO_DYNAMIC, "$sndio_lib", [ ])
                SUMMARY_audio="${SUMMARY_audio} sndio(dynamic)"
            else
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS $SNDIO_LIBS"
                SUMMARY_audio="${SUMMARY_audio} sndio"
            fi

            AC_DEFINE(SDL_AUDIO_DRIVER_SNDIO, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/audio/sndio/*.c"
            EXTRA_CFLAGS="$EXTRA_CFLAGS $SNDIO_CFLAGS"
            have_audio=yes
        fi
    fi
}

dnl Find FusionSound
CheckFusionSound()
{
    AC_ARG_ENABLE(fusionsound,
[AS_HELP_STRING([--enable-fusionsound], [use FusionSound audio driver [default=no]])],
                  , enable_fusionsound=no)
    if test x$enable_audio = xyes -a x$enable_fusionsound = xyes; then
        PKG_CHECK_MODULES([FUSIONSOUND], [fusionsound >= 1.1.1], fusionsound=yes, fusionsound=no)

        if test x$fusionsound = xyes; then
            AC_DEFINE(SDL_AUDIO_DRIVER_FUSIONSOUND, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/audio/fusionsound/*.c"
            EXTRA_CFLAGS="$EXTRA_CFLAGS $FUSIONSOUND_CFLAGS"

            AC_ARG_ENABLE(fusionsound-shared,
[AS_HELP_STRING([--enable-fusionsound-shared], [dynamically load fusionsound audio support [default=yes]])],
                          , enable_fusionsound_shared=yes)
            fusionsound_shared=no
            AC_MSG_CHECKING(for FusionSound dynamic loading support)
            if test x$have_loadso != xyes && \
               test x$enable_fusionsound_shared = xyes; then
                AC_MSG_WARN([You must have SDL_LoadObject() support for dynamic fusionsound loading])
            fi
            if test x$have_loadso = xyes && \
               test x$enable_fusionsound_shared = xyes; then
                AC_DEFINE_UNQUOTED(SDL_AUDIO_DRIVER_FUSIONSOUND_DYNAMIC, "libfusionsound.so", [ ])
                fusionsound_shared=yes
                SUMMARY_audio="${SUMMARY_audio} fusionsound(dynamic)"
            else
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS $FUSIONSOUND_LIBS"
                SUMMARY_audio="${SUMMARY_audio} fusionsound"
            fi
            AC_MSG_RESULT($fusionsound_shared)

            have_audio=yes
        fi
    fi
}

dnl rcg07142001 See if the user wants the disk writer audio driver...
CheckDiskAudio()
{
    AC_ARG_ENABLE(diskaudio,
[AS_HELP_STRING([--enable-diskaudio], [support the disk writer audio driver [default=yes]])],
                  , enable_diskaudio=yes)
    if test x$enable_audio = xyes -a x$enable_diskaudio = xyes; then
        AC_DEFINE(SDL_AUDIO_DRIVER_DISK, 1, [ ])
        SOURCES="$SOURCES $srcdir/src/audio/disk/*.c"
        SUMMARY_audio="${SUMMARY_audio} disk"
        have_audio=yes
    fi
}

dnl rcg03142006 See if the user wants the dummy audio driver...
CheckDummyAudio()
{
    AC_ARG_ENABLE(dummyaudio,
[AS_HELP_STRING([--enable-dummyaudio], [support the dummy audio driver [default=yes]])],
                  , enable_dummyaudio=yes)
    if test x$enable_audio = xyes -a x$enable_dummyaudio = xyes; then
        AC_DEFINE(SDL_AUDIO_DRIVER_DUMMY, 1, [ ])
        SOURCES="$SOURCES $srcdir/src/audio/dummy/*.c"
        SUMMARY_audio="${SUMMARY_audio} dummy"
        have_audio=yes
    fi
}

dnl See if libsamplerate is available
CheckLibSampleRate()
{
    AC_ARG_ENABLE(libsamplerate,
[AS_HELP_STRING([--enable-libsamplerate], [use libsamplerate for audio rate conversion [default=yes]])],
                  , enable_libsamplerate=yes)
    if test x$enable_libsamplerate = xyes; then
        AC_CHECK_HEADER(samplerate.h,
                        have_samplerate_h_hdr=yes,
                        have_samplerate_h_hdr=no)
        if test x$have_samplerate_h_hdr = xyes; then
            AC_DEFINE(HAVE_LIBSAMPLERATE_H, 1, [ ])

            AC_ARG_ENABLE(libsamplerate-shared,
[AS_HELP_STRING([--enable-libsamplerate-shared], [dynamically load libsamplerate [default=yes]])],
                          , enable_libsamplerate_shared=yes)

            samplerate_lib=[`find_lib "libsamplerate.so.*" "" | sed 's/.*\/\(.*\)/\1/; q'`]

            if test x$have_loadso != xyes && \
               test x$enable_libsamplerate_shared = xyes; then
                AC_MSG_WARN([You must have SDL_LoadObject() support for dynamic libsamplerate loading])
            fi
            if test x$have_loadso = xyes && \
               test x$enable_libsamplerate_shared = xyes && test x$samplerate_lib != x; then
                echo "-- dynamic libsamplerate -> $samplerate_lib"
                AC_DEFINE_UNQUOTED(SDL_LIBSAMPLERATE_DYNAMIC, "$samplerate_lib", [ ])
            else
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lsamplerate"
            fi
        fi
    fi
}

dnl Check for ARM instruction support using gas syntax
CheckARM()
{
    AC_ARG_ENABLE(arm-simd,
[AS_HELP_STRING([--enable-arm-simd], [use SIMD assembly blitters on ARM [default=no]])],
                  enable_arm_simd=$enableval, enable_arm_simd=no)
    if test x$enable_video = xyes -a x$enable_assembly = xyes -a x$enable_arm_simd = xyes; then
        save_CFLAGS="$CFLAGS"
        have_arm_simd=no
        CFLAGS="-x assembler-with-cpp $CFLAGS"

        AC_MSG_CHECKING(for ARM SIMD)
        AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
        .text
        .arch armv6
        .object_arch armv4
        .arm
        .altmacro
        #ifndef __ARM_EABI__
        #error EABI is required (to be sure that calling conventions are compatible)
        #endif
        pld [r0]
        uqadd8 r0, r0, r0
        ]])], have_arm_simd=yes)
        AC_MSG_RESULT($have_arm_simd)

        CFLAGS="$save_CFLAGS"

        if test x$have_arm_simd = xyes; then
            AC_DEFINE(SDL_ARM_SIMD_BLITTERS, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/video/arm/pixman-arm-simd*.S"
            WARN_ABOUT_ARM_SIMD_ASM_MIT="yes"
        fi
    fi
}

dnl Check for ARM NEON instruction support using gas syntax
CheckNEON()
{
    AC_ARG_ENABLE(arm-neon,
[AS_HELP_STRING([--enable-arm-neon], [use NEON assembly blitters on ARM [default=no]])],
                  enable_arm_neon=$enableval, enable_arm_neon=no)
    if test x$enable_video = xyes -a x$enable_assembly = xyes -a x$enable_arm_neon = xyes; then
        save_CFLAGS="$CFLAGS"
        have_arm_neon=no
        CFLAGS="-x assembler-with-cpp $CFLAGS"

        AC_MSG_CHECKING(for ARM NEON)
        AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
        .text
        .fpu neon
        .arch armv7a
        .object_arch armv4
        .eabi_attribute 10, 0
        .arm
        .altmacro
        #ifndef __ARM_EABI__
        #error EABI is required (to be sure that calling conventions are compatible)
        #endif
        pld [r0]
        vmovn.u16 d0, q0
        ]])], have_arm_neon=yes)
        AC_MSG_RESULT($have_arm_neon)
        CFLAGS="$save_CFLAGS"

        if test x$have_arm_neon = xyes; then
            AC_DEFINE(SDL_ARM_NEON_BLITTERS, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/video/arm/pixman-arm-neon*.S"
            WARN_ABOUT_ARM_NEON_ASM_MIT="yes"
        fi
    fi
}

dnl See if clang's -fobjc-arc supported.
dnl  Reference: https://github.com/libsdl-org/SDL/pull/5632
CheckObjectiveCARC()
{
    AC_MSG_CHECKING(for clang -fobjc-arc option)
    have_clang_objc_arc=no

    save_CFLAGS="$CFLAGS"
    CFLAGS="$save_CFLAGS -fobjc-arc"
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
    int x = 0;
    ]],[])], [have_clang_objc_arc=yes],[])
    AC_MSG_RESULT($have_clang_objc_arc)
    CFLAGS="$save_CFLAGS"

    if test x$have_clang_objc_arc = xyes; then
        EXTRA_CFLAGS="$EXTRA_CFLAGS -fobjc-arc"
    fi
}

dnl See if GCC's -gdwarf-4 is supported
dnl See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=101377 for why this is needed on Windows
CheckGDwarf4()
{
    AC_MSG_CHECKING(for GCC -gdwarf-4 option)
    have_gcc_gdwarf4=no

    save_CFLAGS="$CFLAGS"
    CFLAGS="$save_CFLAGS -gdwarf-4"
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
    int x = 0;
    ]],[])], [have_gcc_gdwarf4=yes],[])
    AC_MSG_RESULT($have_gcc_gdwarf4)
    CFLAGS="$save_CFLAGS"

    if test x$have_gcc_gdwarf4 = xyes; then
        EXTRA_CFLAGS="$EXTRA_CFLAGS -gdwarf-4"
    fi
}

dnl See if GCC's -fvisibility=hidden is supported (gcc4 and later, usually).
dnl  Details of this flag are here: http://gcc.gnu.org/wiki/Visibility
CheckVisibilityHidden()
{
    AC_MSG_CHECKING(for GCC -fvisibility=hidden option)
    have_gcc_fvisibility=no

    visibility_CFLAGS="-fvisibility=hidden"
    save_CFLAGS="$CFLAGS"
    CFLAGS="$save_CFLAGS $visibility_CFLAGS -Werror"
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
    #if !defined(__GNUC__) || __GNUC__ < 4
    #error SDL only uses visibility attributes in GCC 4 or newer
    #endif
    ]],[])], [have_gcc_fvisibility=yes],[])
    AC_MSG_RESULT($have_gcc_fvisibility)
    CFLAGS="$save_CFLAGS"

    if test x$have_gcc_fvisibility = xyes; then
        EXTRA_CFLAGS="$EXTRA_CFLAGS $visibility_CFLAGS"
    fi
}

dnl See if GCC's -fno-strict-aliasingis supported.
dnl  Reference: https://bugzilla.libsdl.org/show_bug.cgi?id=4254
CheckNoStrictAliasing()
{
    AC_MSG_CHECKING(for GCC -fno-strict-aliasing option)
    have_gcc_no_strict_aliasing=no

    save_CFLAGS="$CFLAGS"
    CFLAGS="$save_CFLAGS -fno-strict-aliasing"
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
    int x = 0;
    ]],[])], [have_gcc_no_strict_aliasing=yes],[])
    AC_MSG_RESULT($have_gcc_no_strict_aliasing)
    CFLAGS="$save_CFLAGS"

    if test x$have_gcc_no_strict_aliasing = xyes; then
        EXTRA_CFLAGS="$EXTRA_CFLAGS -fno-strict-aliasing"
    fi
}

dnl See if GCC's -Werror is supported.
CheckWerror()
{
    AC_ARG_ENABLE(werror,
[AS_HELP_STRING([--enable-werror], [treat warnings as errors [default=no]])],
                  enable_werror=$enableval, enable_werror=no)
    if test x$enable_werror = xyes; then
        AC_MSG_CHECKING(for GCC -Werror option)
        have_gcc_werror=no

        save_CFLAGS="$CFLAGS"
        CFLAGS="$save_CFLAGS -Werror"
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
        int x = 0;
        ]],[])], [have_gcc_werror=yes],[])
        AC_MSG_RESULT($have_gcc_werror)
        CFLAGS="$save_CFLAGS"

        if test x$have_gcc_werror = xyes; then
            EXTRA_CFLAGS="$EXTRA_CFLAGS -Werror"
        fi
    fi
}

dnl See if GCC's -Wno-error=deprecated-declarations is supported.
CheckNoErrorDeprecatedDeclarationsWerror()
{
    AC_MSG_CHECKING(for GCC -Wno-error=deprecated-declarations option)
    have_gcc_no_werror_deprecated_declarations=no

    save_CFLAGS="$CFLAGS"
    CFLAGS="$save_CFLAGS -Wno-error=deprecated-declarations"
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
    int x = 0;
    ]],[])], [have_gcc_no_werror_deprecated_declarations=yes],[])
    AC_MSG_RESULT($have_gcc_werror)
    CFLAGS="$save_CFLAGS"

    if test x$have_gcc_no_werror_deprecated_declarations = xyes; then
        EXTRA_CFLAGS="$EXTRA_CFLAGS -Wno-error=deprecated-declarations"
    fi
}

dnl See if GCC's -Wdeclaration-after-statement is supported.
dnl  This lets us catch things that would fail on a C89 compiler when using
dnl  a modern GCC.
CheckDeclarationAfterStatement()
{
    AC_MSG_CHECKING(for GCC -Wdeclaration-after-statement option)
    have_gcc_declaration_after_statement=no

    save_CFLAGS="$CFLAGS"
    CFLAGS="$save_CFLAGS -Wdeclaration-after-statement -Werror=declaration-after-statement"
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
    int x = 0;
    ]],[])], [have_gcc_declaration_after_statement=yes],[])
    AC_MSG_RESULT($have_gcc_declaration_after_statement)
    CFLAGS="$save_CFLAGS"

    if test x$have_gcc_declaration_after_statement = xyes; then
        EXTRA_CFLAGS="$EXTRA_CFLAGS -Wdeclaration-after-statement -Werror=declaration-after-statement"
    fi
}

dnl See if GCC's -Wall is supported.
CheckWarnAll()
{
    AC_MSG_CHECKING(for GCC -Wall option)
    have_gcc_Wall=no

    save_CFLAGS="$CFLAGS"
    CFLAGS="$save_CFLAGS -Wall"
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
    int x = 0;
    ]],[])], [have_gcc_Wall=yes],[])
    AC_MSG_RESULT($have_gcc_Wall)
    CFLAGS="$save_CFLAGS"

    if test x$have_gcc_Wall = xyes; then
        EXTRA_CFLAGS="$EXTRA_CFLAGS -Wall"

dnl Haiku headers use multicharacter constants all over the place. Ignore these warnings when using -Wall.
        AC_MSG_CHECKING(for necessary GCC -Wno-multichar option)
        need_gcc_Wno_multichar=no
        case "$host" in
            *-*-haiku*)
                need_gcc_Wno_multichar=yes
                ;;
        esac
        AC_MSG_RESULT($need_gcc_Wno_multichar)
        if test x$need_gcc_Wno_multichar = xyes; then
            EXTRA_CFLAGS="$EXTRA_CFLAGS -Wno-multichar"
        fi
    fi
}

dnl See if GCC's -Wunused-local-typedefs is supported and disable it
dnl  because it triggers on gcc 4.8.4 for compile time asserts inside
dnl  of functions.
CheckUnusedLocalTypedefs()
{
    AC_MSG_CHECKING(for GCC -Wunused-local-typedefs option)
    have_gcc_unused_local_typedefs=no

    save_CFLAGS="$CFLAGS"
    CFLAGS="$save_CFLAGS -Wunused-local-typedefs"
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
    int x = 0;
    ]],[])], [have_gcc_unused_local_typedefs=yes],[])
    AC_MSG_RESULT($have_gcc_unused_local_typedefs)
    CFLAGS="$save_CFLAGS"

    if test x$have_gcc_unused_local_typedefs = xyes; then
        EXTRA_CFLAGS="$EXTRA_CFLAGS -Wno-unused-local-typedefs"
    fi
}

dnl Check for Wayland
CheckWayland()
{
    AC_ARG_ENABLE(video-wayland,
[AS_HELP_STRING([--enable-video-wayland], [use Wayland video driver [default=yes]])],
                  ,enable_video_wayland=yes)

    AC_ARG_ENABLE(video-wayland-qt-touch,
[AS_HELP_STRING([--enable-video-wayland-qt-touch], [QtWayland server support for Wayland video driver [default=yes]])],
                  ,enable_video_wayland_qt_touch=yes)

    if test x$enable_video = xyes -a x$enable_video_wayland = xyes; then
        AC_MSG_CHECKING(for Wayland support)
        video_wayland=no
        if  test x$video_opengl_egl = xyes && \
            test x$video_opengles_v2 = xyes; then
            if $PKG_CONFIG --exists 'wayland-client >= 1.18' wayland-scanner wayland-egl wayland-cursor egl 'xkbcommon >= 0.5.0'; then
                WAYLAND_CFLAGS=`$PKG_CONFIG --cflags wayland-client wayland-egl wayland-cursor xkbcommon`
                WAYLAND_LIBS=`$PKG_CONFIG --libs wayland-client wayland-egl wayland-cursor xkbcommon`
                WAYLAND_SCANNER=`$PKG_CONFIG --variable=wayland_scanner wayland-scanner`
                AS_IF([$PKG_CONFIG --exists 'wayland-scanner >= 1.15'],
                      [WAYLAND_SCANNER_CODE_MODE=private-code],
                      [WAYLAND_SCANNER_CODE_MODE=code])
                video_wayland=yes
            fi
        fi
        AC_MSG_RESULT($video_wayland)

        if test x$video_wayland = xyes; then
            AC_DEFINE(SDL_VIDEO_DRIVER_WAYLAND, 1, [ ])
            if test x$enable_video_wayland_qt_touch = xyes; then
                AC_DEFINE(SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH, 1, [ ])
            fi

            WAYLAND_SOURCES="$srcdir/src/video/wayland/*.c"
            SOURCES="$SOURCES $WAYLAND_SOURCES"
            EXTRA_CFLAGS="$EXTRA_CFLAGS $WAYLAND_CFLAGS -I\$(gen)"
            AC_ARG_ENABLE(wayland-shared,
[AS_HELP_STRING([--enable-wayland-shared], [dynamically load Wayland support [default=maybe]])],
                          , enable_wayland_shared=maybe)

dnl FIXME: Do BSD and OS X need special cases?
            case "$host" in
                *)
                    wayland_client_lib=[`find_lib "libwayland-client.so.*" "$WAYLAND_LIBS" | sed 's/.*\/\(.*\)/\1/; q'`]
                    wayland_egl_lib=[`find_lib "libwayland-egl.so.*" "$WAYLAND_LIBS" | sed 's/.*\/\(.*\)/\1/; q'`]
                    if test x$wayland_egl_lib = x; then
                        # This works in Ubuntu 13.10, maybe others
                        wayland_egl_lib=[`find_lib "mesa-egl/libwayland-egl.so.*" "$WAYLAND_LIBS" | sed 's/.*\/\(.*\)/\1/; q'`]
                    fi
                    wayland_cursor_lib=[`find_lib "libwayland-cursor.so.*" "$WAYLAND_LIBS" | sed 's/.*\/\(.*\)/\1/; q'`]
                    xkbcommon_lib=[`find_lib "libxkbcommon.so.*" "$WAYLAND_LIBS" | sed 's/.*\/\(.*\)/\1/; q'`]
                    ;;
            esac

            if test x$enable_wayland_shared = xmaybe; then
                enable_wayland_shared=yes
            fi
            if test x$have_loadso != xyes && \
               test x$enable_wayland_shared = xyes; then
                AC_MSG_WARN([You must have SDL_LoadObject() support for dynamic Wayland loading])
                enable_wayland_shared=no
            fi
            if test x$have_loadso = xyes && \
               test x$enable_wayland_shared = xyes && \
               test x$wayland_client_lib != x && \
               test x$wayland_egl_lib != x && \
               test x$wayland_cursor_lib != x && \
               test x$xkbcommon_lib != x; then
                echo "-- dynamic libwayland-client -> $wayland_client_lib"
                echo "-- dynamic libwayland-egl -> $wayland_egl_lib"
                echo "-- dynamic libwayland-cursor -> $wayland_cursor_lib"
                echo "-- dynamic libxkbcommon -> $xkbcommon_lib"
                AC_DEFINE_UNQUOTED(SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC, "$wayland_client_lib", [ ])
                AC_DEFINE_UNQUOTED(SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_EGL, "$wayland_egl_lib", [ ])
                AC_DEFINE_UNQUOTED(SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_CURSOR, "$wayland_cursor_lib", [ ])
                AC_DEFINE_UNQUOTED(SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_XKBCOMMON, "$xkbcommon_lib", [ ])
                SUMMARY_video="${SUMMARY_video} wayland(dynamic)"
            else
                enable_wayland_shared=no
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS $WAYLAND_LIBS"
                SUMMARY_video="${SUMMARY_video} wayland"
            fi
            have_video=yes

dnl See if libdecor is available
            AC_ARG_ENABLE(libdecor,
[AS_HELP_STRING([--enable-libdecor], [use libdecor for Wayland client-side decorations [default=yes]])],, enable_libdecor=yes)
            if test x$enable_libdecor = xyes; then
                PKG_CHECK_MODULES([DECOR], [libdecor-0], video_libdecor=yes, video_libdecor=no)
                if test x$video_libdecor = xyes; then
                    EXTRA_CFLAGS="$EXTRA_CFLAGS $DECOR_CFLAGS"
                    AC_DEFINE(HAVE_LIBDECOR_H, 1, [ ])

                    AC_ARG_ENABLE(libdecor-shared,
[AS_HELP_STRING([--enable-libdecor-shared], [dynamically load libdecor [default=yes]])],, enable_libdecor_shared=yes)

                    decor_lib=[`find_lib "libdecor-0.so.*" "$DECOR_LIBS" | sed 's/.*\/\(.*\)/\1/; q'`]

                    if test x$enable_wayland_shared != xyes; then
                       enable_libdecor_shared=no
                    fi
                    if test x$have_loadso != xyes && \
                       test x$enable_libdecor_shared = xyes; then
                        AC_MSG_WARN([You must have SDL_LoadObject() support for dynamic libdecor loading])
                    fi
                    if test x$have_loadso = xyes && \
                       test x$enable_libdecor_shared = xyes && test x$decor_lib != x; then
                        echo "-- dynamic libdecor -> $decor_lib"
                        AC_DEFINE_UNQUOTED(SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_LIBDECOR, "$decor_lib", [ ])
                    else
                        EXTRA_LDFLAGS="$EXTRA_LDFLAGS $DECOR_LIBS"
                    fi

                    saved_cflags=$CFLAGS
                    CFLAGS="$CFLAGS $DECOR_CFLAGS"
                    AC_CHECK_DECLS([libdecor_frame_get_min_content_size, libdecor_frame_get_max_content_size], [libdecor_get_min_max=yes], [ ], [[#include <libdecor.h>]])
                    if test x$libdecor_get_min_max = xyes; then
                        AC_DEFINE(SDL_HAVE_LIBDECOR_GET_MIN_MAX, 1, [ ])
                    fi
                    CFLAGS="$saved_cflags"
                fi
            fi
        fi
    fi
}

dnl Check for Native Client stuff
CheckNativeClient()
{
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
          #if !defined(__native_client__)
          #error "NO NACL"
          #endif
        ]],[])], [
        AC_DEFINE(SDL_VIDEO_DRIVER_NACL, 1, [ ])
        AC_DEFINE(SDL_AUDIO_DRIVER_NACL, 1, [ ])
        AC_DEFINE(HAVE_POW, 1, [ ])
        AC_DEFINE(HAVE_OPENGLES2, 1, [ ])
        AC_DEFINE(SDL_VIDEO_OPENGL_ES2, 1, [ ])
        AC_DEFINE(SDL_VIDEO_RENDER_OGL_ES2, 1, [ ])

        SDL_LIBS="-lppapi_simple -lppapi_gles2 $SDL_LIBS"

        SDLMAIN_SOURCES="$srcdir/src/main/nacl/*.c"
        SOURCES="$SOURCES $srcdir/src/audio/nacl/*.c"
        SUMMARY_audio="${SUMMARY_audio} nacl"
        have_audio=yes
        SOURCES="$SOURCES $srcdir/src/video/nacl/*.c"
        SUMMARY_video="${SUMMARY_video} nacl opengles2"
        have_video=yes
        ],[])
}

CheckRPI()
{
    AC_ARG_ENABLE(video-rpi,
[AS_HELP_STRING([--enable-video-rpi], [use Raspberry Pi 2/3 video driver [default=yes]])],
                  , enable_video_rpi=yes)
    if test x$enable_video = xyes -a x$enable_video_rpi = xyes; then
        PKG_CHECK_MODULES([RPI], [bcm_host brcmegl], video_rpi=yes, video_rpi=no)

        if test x$video_rpi = xno; then
            if test x$ARCH = xnetbsd; then
                RPI_CFLAGS="-I/usr/pkg/include -I/usr/pkg/include/interface/vcos/pthreads -I/usr/pkg/include/interface/vmcs_host/linux"
                RPI_LIBS="-Wl,-R/usr/pkg/lib -L/usr/pkg/lib -lbcm_host"
            else
                RPI_CFLAGS="-I/opt/vc/include -I/opt/vc/include/interface/vcos/pthreads -I/opt/vc/include/interface/vmcs_host/linux"
                RPI_LIBS="-Wl,-rpath,/opt/vc/lib -L/opt/vc/lib -lbcm_host"
            fi
        fi

        # Save the original compiler flags and libraries
        ac_save_cflags="$CFLAGS"; ac_save_libs="$LIBS"

        # Add the Raspberry Pi compiler flags and libraries
        CFLAGS="$CFLAGS $RPI_CFLAGS"; LIBS="$LIBS $RPI_LIBS"

        AC_MSG_CHECKING(for Raspberry Pi 2/3)
        have_video_rpi=no
        AC_LINK_IFELSE([AC_LANG_PROGRAM([[
          #include <bcm_host.h>
          #include <EGL/eglplatform.h>
        ]], [[
          EGL_DISPMANX_WINDOW_T window;
          bcm_host_init();
        ]])], [have_video_rpi=yes],[])
        AC_MSG_RESULT($have_video_rpi)

        # Restore the compiler flags and libraries
        CFLAGS="$ac_save_cflags"; LIBS="$ac_save_libs"

        if test x$have_video_rpi = xyes; then
            CFLAGS="$CFLAGS $RPI_CFLAGS"
            SDL_CFLAGS="$SDL_CFLAGS $RPI_CFLAGS"
            EXTRA_CFLAGS="$EXTRA_CFLAGS $RPI_CFLAGS"
            EXTRA_LDFLAGS="$EXTRA_LDFLAGS $RPI_LIBS"
            SOURCES="$SOURCES $srcdir/src/video/raspberry/*.c"
            AC_DEFINE(SDL_VIDEO_DRIVER_RPI, 1, [ ])
            SUMMARY_video="${SUMMARY_video} rpi"
            have_video=yes
        fi
    fi
}

dnl Find the X11 include and library directories
CheckX11()
{
    AC_ARG_ENABLE(video-x11,
[AS_HELP_STRING([--enable-video-x11], [use X11 video driver [default=maybe]])],
                  ,[
    enable_video_x11=yes
    case "$host" in
        *-*-darwin*|*-ios-*)
            enable_video_x11=no
            ;;
    esac])

    if test x$enable_video = xyes -a x$enable_video_x11 = xyes; then
        case "$host" in
            *-*-darwin*)
                # This isn't necessary for X11, but fixes GLX detection
                if test "x$x_includes" = xNONE && \
                   test "x$x_libraries" = xNONE && \
                   test -d /opt/X11/include && \
                   test -d /opt/X11/lib; then
                    x_includes="/opt/X11/include"
                    x_libraries="/opt/X11/lib"
                fi
                ;;
        esac
        AC_PATH_X
        AC_PATH_XTRA
        if test x$have_x = xyes; then
            AC_ARG_ENABLE(x11-shared,
[AS_HELP_STRING([--enable-x11-shared], [dynamically load X11 support [default=maybe]])],
                          , enable_x11_shared=maybe)

            case "$host" in
                *-*-darwin*)
                    # Apple now puts this in /opt/X11
                    x11_lib='/opt/X11/lib/libX11.6.dylib'
                    x11ext_lib='/opt/X11/lib/libXext.6.dylib'
                    xcursor_lib='/opt/X11/lib/libXcursor.1.dylib'
                    xinput_lib='/opt/X11/lib/libXi.6.dylib'
                    xfixes_lib='/opt/X11/lib/libXfixes.3.dylib'
                    xrandr_lib='/opt/X11/lib/libXrandr.2.dylib'
                    xrender_lib='/opt/X11/lib/libXrender.1.dylib'
                    xss_lib='/opt/X11/lib/libXss.1.dylib'
                    ;;
                *-*-openbsd*)
                    x11_lib='libX11.so'
                    x11ext_lib='libXext.so'
                    xcursor_lib='libXcursor.so'
                    xinput_lib='libXi.so'
                    xfixes_lib='libXfixes.so'
                    xrandr_lib='libXrandr.so'
                    xrender_lib='libXrender.so'
                    xss_lib='libXss.so'
                    ;;
                *)
                    x11_lib=[`find_lib "libX11.so.*" "$X_LIBS -L/usr/X11/$base_libdir -L/usr/X11R6/$base_libdir" | sed 's/.*\/\(.*\)/\1/; q'`]
                    x11ext_lib=[`find_lib "libXext.so.*" "$X_LIBS -L/usr/X11/$base_libdir -L/usr/X11R6/$base_libdir" | sed 's/.*\/\(.*\)/\1/; q'`]
                    xcursor_lib=[`find_lib "libXcursor.so.*" "$X_LIBS -L/usr/X11/$base_libdir -L/usr/X11R6/$base_libdir" | sed 's/.*\/\(.*\)/\1/; q'`]
                    xinput_lib=[`find_lib "libXi.so.*" "$X_LIBS -L/usr/X11/$base_libdir -L/usr/X11R6/$base_libdir" | sed 's/.*\/\(.*\)/\1/; q'`]
                    xfixes_lib=[`find_lib "libXfixes.so.*" "$X_LIBS -L/usr/X11/$base_libdir -L/usr/X11R6/$base_libdir" | sed 's/.*\/\(.*\)/\1/; q'`]
                    xrandr_lib=[`find_lib "libXrandr.so.*" "$X_LIBS -L/usr/X11/$base_libdir -L/usr/X11R6/$base_libdir" | sed 's/.*\/\(.*\)/\1/; q'`]
                    xrender_lib=[`find_lib "libXrender.so.*" "$X_LIBS -L/usr/X11/$base_libdir -L/usr/X11R6/$base_libdir" | sed 's/.*\/\(.*\)/\1/; q'`]
                    xss_lib=[`find_lib "libXss.so.*" "$X_LIBS -L/usr/X11/$base_libdir -L/usr/X11R6/$base_libdir" | sed 's/.*\/\(.*\)/\1/; q'`]
                    ;;
            esac

            if test x$ac_cv_func_shmat != xyes; then
                X_CFLAGS="$X_CFLAGS -DNO_SHARED_MEMORY"
            fi
            CFLAGS="$CFLAGS $X_CFLAGS"
            LDFLAGS="$LDFLAGS $X_LIBS"

            AC_CHECK_HEADER(X11/extensions/Xext.h,
                            have_xext_h_hdr=yes,
                            have_xext_h_hdr=no,
                            [#include <X11/Xlib.h>
                             #include <X11/Xproto.h>
                            ])
            if test x$have_xext_h_hdr != xyes; then
               AC_MSG_ERROR([
*** Missing Xext.h, maybe you need to install the libxext-dev package?
               ])
            fi

            AC_DEFINE(SDL_VIDEO_DRIVER_X11, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/video/x11/*.c"
            EXTRA_CFLAGS="$EXTRA_CFLAGS $X_CFLAGS"

            # Needed so SDL applications can include SDL_syswm.h
            SDL_CFLAGS="$SDL_CFLAGS $X_CFLAGS"

            if test x$enable_x11_shared = xmaybe; then
                enable_x11_shared=yes
            fi
            if test x$have_loadso != xyes && \
               test x$enable_x11_shared = xyes; then
                AC_MSG_WARN([You must have SDL_LoadObject() support for dynamic X11 loading])
                enable_x11_shared=no
            fi
            if test x$have_loadso = xyes && \
               test x$enable_x11_shared = xyes && test x$x11_lib != x && test x$x11ext_lib != x; then
                echo "-- dynamic libX11 -> $x11_lib"
                echo "-- dynamic libX11ext -> $x11ext_lib"
                AC_DEFINE_UNQUOTED(SDL_VIDEO_DRIVER_X11_DYNAMIC, "$x11_lib", [ ])
                AC_DEFINE_UNQUOTED(SDL_VIDEO_DRIVER_X11_DYNAMIC_XEXT, "$x11ext_lib", [ ])
                SUMMARY_video="${SUMMARY_video} x11(dynamic)"
            else
                enable_x11_shared=no
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS $X_LIBS -lX11 -lXext"
                SUMMARY_video="${SUMMARY_video} x11"
            fi
            have_video=yes

            AC_MSG_CHECKING([for XGenericEvent])
            have_XGenericEvent=no
            AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                #include <X11/Xlib.h>
            ]], [[
Display *display;
XEvent event;
XGenericEventCookie *cookie = &event.xcookie;
XNextEvent(display, &event);
XGetEventData(display, cookie);
XFreeEventData(display, cookie);
            ]])], [
                have_XGenericEvent=yes
                AC_DEFINE([SDL_VIDEO_DRIVER_X11_SUPPORTS_GENERIC_EVENTS], 1, [ ])
            ],[])
            AC_MSG_RESULT($have_XGenericEvent)

            AC_CHECK_LIB(X11, XkbKeycodeToKeysym, AC_DEFINE(SDL_VIDEO_DRIVER_X11_HAS_XKBKEYCODETOKEYSYM, 1, [Have XkbKeycodeToKeysym]))

            AC_ARG_ENABLE(video-x11-xcursor,
[AS_HELP_STRING([--enable-video-x11-xcursor], [enable X11 Xcursor support [default=yes]])],
                            , enable_video_x11_xcursor=yes)
            if test x$enable_video_x11_xcursor = xyes; then
                definitely_enable_video_x11_xcursor=no
                AC_CHECK_HEADER(X11/Xcursor/Xcursor.h,
                                have_xcursor_h_hdr=yes,
                                have_xcursor_h_hdr=no,
                                [#include <X11/Xlib.h>
                                ])
                if test x$have_xcursor_h_hdr = xyes; then
                    if test x$enable_x11_shared = xyes && test x$xcursor_lib != x ; then
                        echo "-- dynamic libXcursor -> $xcursor_lib"
                        AC_DEFINE_UNQUOTED(SDL_VIDEO_DRIVER_X11_DYNAMIC_XCURSOR, "$xcursor_lib", [ ])
                        definitely_enable_video_x11_xcursor=yes
                    else
                        AC_CHECK_LIB(Xcursor, XcursorImageCreate, have_xcursor_lib=yes)
                        if test x$have_xcursor_lib = xyes ; then
                            EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lXcursor"
                            definitely_enable_video_x11_xcursor=yes
                        fi
                    fi
                fi
            fi
            if test x$definitely_enable_video_x11_xcursor = xyes; then
                AC_DEFINE(SDL_VIDEO_DRIVER_X11_XCURSOR, 1, [ ])
                SUMMARY_video_x11="${SUMMARY_video_x11} xcursor"
            fi
            AC_ARG_ENABLE(video-x11-xdbe,
[AS_HELP_STRING([--enable-video-x11-xdbe], [enable X11 Xdbe support [default=yes]])],
                            , enable_video_x11_xdbe=yes)
            if test x$enable_video_x11_xdbe = xyes; then
                AC_CHECK_HEADER(X11/extensions/Xdbe.h,
                                have_dbe_h_hdr=yes,
                                have_dbe_h_hdr=no,
                                [#include <X11/Xlib.h>
                                ])
                if test x$have_dbe_h_hdr = xyes; then
                    AC_DEFINE(SDL_VIDEO_DRIVER_X11_XDBE, 1, [ ])
                    SUMMARY_video_x11="${SUMMARY_video_x11} xdbe"
                fi
            fi
            AC_ARG_ENABLE(video-x11-xinput,
[AS_HELP_STRING([--enable-video-x11-xinput], [enable X11 XInput extension for manymouse, tablets, etc [default=yes]])],
                            , enable_video_x11_xinput=yes)
            if test x$enable_video_x11_xinput = xyes; then
                definitely_enable_video_x11_xinput=no
                AC_CHECK_HEADER(X11/extensions/XInput2.h,
                                have_xinput_h_hdr=yes,
                                have_xinput_h_hdr=no,
                                [#include <X11/Xlib.h>
                                ])
                if test x$have_xinput_h_hdr = xyes; then
                    if test x$enable_x11_shared = xyes && test x$xinput_lib != x ; then
                        echo "-- dynamic libXi -> $xinput_lib"
                        AC_DEFINE_UNQUOTED(SDL_VIDEO_DRIVER_X11_DYNAMIC_XINPUT2, "$xinput_lib", [ ])
                        definitely_enable_video_x11_xinput=yes
                    else
                        AC_CHECK_LIB(Xi, XOpenDevice, have_xinput_lib=yes)
                        if test x$have_xinput_lib = xyes ; then
                            EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lXi"
                            definitely_enable_video_x11_xinput=yes
                        fi
                    fi
                fi
            fi
            if test x$definitely_enable_video_x11_xinput = xyes; then
                SUMMARY_video_x11="${SUMMARY_video_x11} xinput2"
                AC_DEFINE(SDL_VIDEO_DRIVER_X11_XINPUT2, 1, [ ])
                AC_MSG_CHECKING(for xinput2 multitouch)
                have_xinput2_multitouch=no
                AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                    #include <X11/Xlib.h>
                    #include <X11/Xproto.h>
                    #include <X11/extensions/XInput2.h>
                    ]], [[
int event_type = XI_TouchBegin;
XITouchClassInfo *t;
                    ]])], [
                    have_xinput2_multitouch=yes
                    AC_DEFINE([SDL_VIDEO_DRIVER_X11_XINPUT2_SUPPORTS_MULTITOUCH], 1, [ ])
                    SUMMARY_video_x11="${SUMMARY_video_x11} xinput2_multitouch"
                    ],[])
                AC_MSG_RESULT($have_xinput2_multitouch)
            fi
            AC_ARG_ENABLE(video-x11-xfixes,
[AS_HELP_STRING([--enable-video-x11-xfixes], [enable X11 Xfixes support [default=yes]])],
                            , enable_video_x11_xfixes=yes)
            if test x$enable_video_x11_xfixes = xyes; then
                definitely_enable_video_x11_xfixes=no
                # check along with XInput2.h because we use Xfixes with XIBarrierReleasePointer
                AC_MSG_CHECKING(for X11/extensions/Xfixes.h)
                AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                    #include <X11/Xlib.h>
                    #include <X11/Xproto.h>
                    #include <X11/extensions/XInput2.h>
                    #include <X11/extensions/Xfixes.h>]],
                    [BarrierEventID b;])],
                    [have_xfixes_h_hdr=yes],
                    [have_xfixes_h_hdr=no])
                AC_MSG_RESULT($have_xfixes_h_hdr)
                if test x$have_xfixes_h_hdr = xyes; then
                    if test x$enable_x11_shared = xyes && test x$xfixes_lib != x ; then
                        echo "-- dynamic libXfixes -> $xfixes_lib"
                        AC_DEFINE_UNQUOTED(SDL_VIDEO_DRIVER_X11_DYNAMIC_XFIXES, "$xfixes_lib", [ ])
                        definitely_enable_video_x11_xfixes=yes
                    else
                        AC_CHECK_LIB(Xfixes, XFixesCreatePointerBarrier, have_xfixes_lib=yes)
                        if test x$have_xfixes_lib = xyes ; then
                            EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lXfixes"
                            definitely_enable_video_x11_xfixes=yes
                        fi
                    fi
                fi
            fi
            if test x$definitely_enable_video_x11_xfixes = xyes; then
                AC_DEFINE(SDL_VIDEO_DRIVER_X11_XFIXES, 1, [ ])
                SUMMARY_video_x11="${SUMMARY_video_x11} xfixes"
            fi
            AC_ARG_ENABLE(video-x11-xrandr,
[AS_HELP_STRING([--enable-video-x11-xrandr], [enable X11 Xrandr extension for fullscreen [default=yes]])],
                            , enable_video_x11_xrandr=yes)
            if test x$enable_video_x11_xrandr = xyes; then
dnl             XRRScreenResources is only present in Xrandr >= 1.2, we use that as a test.
                definitely_enable_video_x11_xrandr=no
                have_xrandr_h_hdr=no
                AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                #include <X11/Xlib.h>
                #include <X11/extensions/Xrandr.h>
                ]], [[
                XRRScreenResources *res = NULL;
                ]])], [have_xrandr_h_hdr=yes],[])
                if test x$have_xrandr_h_hdr = xyes; then
                    if test x$enable_x11_shared = xyes && test x$xrandr_lib != x ; then
                        echo "-- dynamic libXrandr -> $xrandr_lib"
                        AC_DEFINE_UNQUOTED(SDL_VIDEO_DRIVER_X11_DYNAMIC_XRANDR, "$xrandr_lib", [ ])
                        definitely_enable_video_x11_xrandr=yes
                    else
                        AC_CHECK_LIB(Xrandr, XRRQueryExtension, have_xrandr_lib=yes)
                        if test x$have_xrandr_lib = xyes ; then
                            EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lXrandr"
                            definitely_enable_video_x11_xrandr=yes
                        fi
                    fi
                fi
            fi
            if test x$definitely_enable_video_x11_xrandr = xyes; then
                AC_DEFINE(SDL_VIDEO_DRIVER_X11_XRANDR, 1, [ ])
                SUMMARY_video_x11="${SUMMARY_video_x11} xrandr"
            fi
            AC_ARG_ENABLE(video-x11-scrnsaver,
[AS_HELP_STRING([--enable-video-x11-scrnsaver], [enable X11 screensaver extension [default=yes]])],
                            , enable_video_x11_scrnsaver=yes)
            if test x$enable_video_x11_scrnsaver = xyes; then
                AC_CHECK_HEADER(X11/extensions/scrnsaver.h,
                                have_scrnsaver_h_hdr=yes,
                                have_scrnsaver_h_hdr=no,
                                [#include <X11/Xlib.h>
                                ])
                if test x$have_scrnsaver_h_hdr = xyes; then
                    if test x$enable_x11_shared = xyes && test x$xss_lib != x ; then
                        echo "-- dynamic libXss -> $xss_lib"
                        AC_DEFINE_UNQUOTED(SDL_VIDEO_DRIVER_X11_DYNAMIC_XSS, "$xss_lib", [ ])
                        definitely_enable_video_x11_scrnsaver=yes
                    else
                        AC_CHECK_LIB(Xss, XScreenSaverSuspend, have_xss_lib=yes)
                        if test x$have_xss_lib = xyes ; then
                            EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lXss"
                            definitely_enable_video_x11_scrnsaver=yes
                        fi
                    fi
                fi
            fi
            if test x$definitely_enable_video_x11_scrnsaver = xyes; then
                AC_DEFINE(SDL_VIDEO_DRIVER_X11_XSCRNSAVER, 1, [ ])
                SUMMARY_video_x11="${SUMMARY_video_x11} xscrnsaver"
            fi
            AC_ARG_ENABLE(video-x11-xshape,
[AS_HELP_STRING([--enable-video-x11-xshape], [enable X11 XShape support [default=yes]])],
                            , enable_video_x11_xshape=yes)
            if test x$enable_video_x11_xshape = xyes; then
                AC_CHECK_HEADER(X11/extensions/shape.h,
                                have_shape_h_hdr=yes,
                                have_shape_h_hdr=no,
                                [#include <X11/Xlib.h>
                                ])
                if test x$have_shape_h_hdr = xyes; then
                    AC_DEFINE(SDL_VIDEO_DRIVER_X11_XSHAPE, 1, [ ])
                    SUMMARY_video_x11="${SUMMARY_video_x11} xshape"
                fi
            fi
        fi
    fi
    if test x$have_x != xyes; then
        # Prevent Mesa from including X11 headers
        EXTRA_CFLAGS="$EXTRA_CFLAGS -DMESA_EGL_NO_X11_HEADERS -DEGL_NO_X11"
    fi
}

dnl Set up the Vivante video driver if enabled
CheckVivanteVideo()
{
    AC_ARG_ENABLE(video-vivante,
[AS_HELP_STRING([--enable-video-vivante], [use Vivante EGL video driver [default=yes]])],
                  , enable_video_vivante=yes)
    if test x$enable_video = xyes -a x$enable_video_vivante = xyes; then
        AC_MSG_CHECKING(for Vivante VDK API)
        have_vivante_vdk=no
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
          #define LINUX
          #define EGL_API_FB
          #include <gc_vdk.h>
        ]],[])], [have_vivante_vdk=yes],[])
        AC_MSG_RESULT($have_vivante_vdk)

        AC_MSG_CHECKING(for Vivante FB API)
        have_vivante_egl=no
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
          #define LINUX
          #define EGL_API_FB
          #include <EGL/eglvivante.h>
        ]],[])], [have_vivante_egl=yes],[])
        AC_MSG_RESULT($have_vivante_egl)

        if test x$have_vivante_vdk = xyes -o x$have_vivante_egl = xyes; then
            AC_DEFINE(SDL_VIDEO_DRIVER_VIVANTE, 1, [ ])
            EXTRA_CFLAGS="$EXTRA_CFLAGS -DLINUX -DEGL_API_FB"
            if test x$have_vivante_vdk = xyes; then
                AC_DEFINE(SDL_VIDEO_DRIVER_VIVANTE_VDK, 1, [ ])
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lVDK"
            fi
            SOURCES="$SOURCES $srcdir/src/video/vivante/*.c"
            SUMMARY_video="${SUMMARY_video} vivante"
            have_video=yes
        fi
    fi
}

dnl Set up the Haiku video driver if enabled
CheckHaikuVideo()
{
    if test x$enable_video = xyes; then
        AC_DEFINE(SDL_VIDEO_DRIVER_HAIKU, 1, [ ])
        SOURCES="$SOURCES $srcdir/src/video/haiku/*.cc"
        have_video=yes
        SUMMARY_video="${SUMMARY_video} haiku"
    fi
}

dnl Set up the Cocoa video driver for Mac OS X (but not Darwin)
CheckCOCOA()
{
    AC_ARG_ENABLE(video-cocoa,
[AS_HELP_STRING([--enable-video-cocoa], [use Cocoa video driver [default=yes]])],
                  , enable_video_cocoa=yes)
    if test x$enable_video = xyes -a x$enable_video_cocoa = xyes; then
        save_CFLAGS="$CFLAGS"
dnl     Work around that we don't have Objective-C support in autoconf
        CFLAGS="$CFLAGS -x objective-c"
        AC_MSG_CHECKING(for Cocoa framework)
        have_cocoa=no
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
          #import <Cocoa/Cocoa.h>
        ]],[])], [have_cocoa=yes],[])
        AC_MSG_RESULT($have_cocoa)
        CFLAGS="$save_CFLAGS"
        if test x$have_cocoa = xyes; then
            AC_DEFINE(SDL_VIDEO_DRIVER_COCOA, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/video/cocoa/*.m"
            SUMMARY_video="${SUMMARY_video} cocoa"
            have_video=yes
        fi
    fi
}

CheckMETAL()
{
    AC_ARG_ENABLE(video-metal,
[AS_HELP_STRING([--enable-video-metal], [include Metal support [default=yes]])],
              , enable_video_metal=yes)
    AC_ARG_ENABLE(render-metal,
[AS_HELP_STRING([--enable-render-metal], [enable the Metal render driver [default=yes]])],
                                , enable_render_metal=yes)
    if test x$enable_video = xyes -a x$enable_video_metal = xyes; then
        save_CFLAGS="$CFLAGS"
dnl     Work around that we don't have Objective-C support in autoconf
        CFLAGS="$CFLAGS -x objective-c"
        AC_MSG_CHECKING(for Metal framework)
        have_metal=no
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
          #import <Cocoa/Cocoa.h>
          #import <Metal/Metal.h>
          #import <QuartzCore/CAMetalLayer.h>

          #if TARGET_CPU_X86
          #error Metal doesn't work on this configuration
          #endif
        ]],[])], [have_metal=yes],[])
        CFLAGS="$save_CFLAGS"
        AC_MSG_RESULT($have_metal)
        if test x$have_metal = xyes; then
            AC_DEFINE(SDL_VIDEO_METAL, 1, [ ])
            if test x$enable_render = xyes -a x$enable_render_metal = xyes; then
                AC_DEFINE(SDL_VIDEO_RENDER_METAL, 1, [ ])
                SOURCES="$SOURCES $srcdir/src/render/metal/*.m"
            fi
            SUMMARY_video="${SUMMARY_video} metal"
        else
            enable_video_metal=no
            enable_render_metal=no
        fi
    fi
}

dnl Find DirectFB
CheckDirectFB()
{
    AC_ARG_ENABLE(video-directfb,
[AS_HELP_STRING([--enable-video-directfb], [use DirectFB video driver [default=no]])],
                  , enable_video_directfb=no)
    if test x$enable_video = xyes -a x$enable_video_directfb = xyes; then
        PKG_CHECK_MODULES([DIRECTFB], [directfb >= 1.0.0], video_directfb=yes, video_directfb=no)

        if test x$video_directfb = xyes; then
            # SuSE 11.1 installs directfb-config without directfb-devel
            save_CPPFLAGS="$CPPFLAGS"
            CPPFLAGS="$CPPFLAGS $DIRECTFB_CFLAGS"
            AC_CHECK_HEADER(directfb.h, have_directfb_hdr=yes, have_directfb_hdr=no)
            CPPFLAGS="$save_CPPFLAGS"
            video_directfb=$have_directfb_hdr
        fi

        if test x$video_directfb = xyes; then
            AC_ARG_ENABLE(directfb-shared,
[AS_HELP_STRING([--enable-directfb-shared], [dynamically load directfb support [default=yes]])],
                              , enable_directfb_shared=yes)

            AC_DEFINE(SDL_VIDEO_DRIVER_DIRECTFB, 1, [ ])
            AC_DEFINE(SDL_VIDEO_RENDER_DIRECTFB, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/video/directfb/*.c"
            EXTRA_CFLAGS="$EXTRA_CFLAGS $DIRECTFB_CFLAGS"

            directfb_shared=no
            directfb_lib=[`find_lib "libdirectfb*.so.*" "$DIRECTFB_LIBS" | sed 's/.*\/\(.*\)/\1/; q'`]
            if test x$have_loadso != xyes && \
               test x$enable_directfb_shared = xyes; then
                AC_MSG_WARN([You must have SDL_LoadObject() support for dynamic directfb loading])
            fi
            if test x$have_loadso = xyes && \
               test x$enable_directfb_shared = xyes && test x$directfb_lib != x; then
                directfb_shared=yes
                echo "-- dynamic libdirectfb -> $directfb_lib"
                AC_DEFINE_UNQUOTED(SDL_VIDEO_DRIVER_DIRECTFB_DYNAMIC, "$directfb_lib", [ ])
                SUMMARY_video="${SUMMARY_video} directfb(dynamic)"
            else
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS $DIRECTFB_LIBS"
                SUMMARY_video="${SUMMARY_video} directfb"
            fi
            SDL_CFLAGS="$SDL_CFLAGS $DIRECTFB_CFLAGS"
            have_video=yes
        fi
    fi
}

dnl Find KMSDRM
CheckKMSDRM()
{
    AC_ARG_ENABLE(video-kmsdrm,
[AS_HELP_STRING([--enable-video-kmsdrm], [use KMSDRM video driver [default=yes]])],
                  , enable_video_kmsdrm=yes)

    if  test x$enable_video = xyes && \
        test x$enable_video_kmsdrm = xyes && \
        test x$video_opengl_egl = xyes; then

        video_kmsdrm=no

        PKG_CHECK_MODULES([LIBDRM], [libdrm >= 1.4.82], libdrm_avail=yes, libdrm_avail=no)
        PKG_CHECK_MODULES([LIBGBM], [gbm >= 11.1.0], libgbm_avail=yes, libgbm_avail=no)

        if test x$libdrm_avail = xyes -a x$libgbm_avail = xyes; then
            video_kmsdrm=yes
        fi

        if test x$video_kmsdrm = xyes; then
            AC_ARG_ENABLE(kmsdrm-shared,
[AS_HELP_STRING([--enable-kmsdrm-shared], [dynamically load kmsdrm support [default=yes]])],
                , enable_kmsdrm_shared=yes)

            AC_DEFINE(SDL_VIDEO_DRIVER_KMSDRM, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/video/kmsdrm/*.c"
            EXTRA_CFLAGS="$EXTRA_CFLAGS $LIBDRM_CFLAGS $LIBGBM_CFLAGS"

            AC_MSG_CHECKING(for kmsdrm dynamic loading support)
            kmsdrm_shared=no
            drm_lib=[`find_lib "libdrm.so.*" "$LIBDRM_LIBS"`]
            gbm_lib=[`find_lib "libgbm.so.*" "$LIBGBM_LIBS"`]
            if test x$have_loadso != xyes && \
               test x$enable_kmsdrm_shared = xyes; then
                AC_MSG_WARN([You must have SDL_LoadObject() support for dynamic kmsdrm loading])
            fi
            if test x$have_loadso = xyes && \
               test x$enable_kmsdrm_shared = xyes && test x$drm_lib != x && test x$gbm_lib != x; then
                kmsdrm_shared=yes
                AC_DEFINE_UNQUOTED(SDL_VIDEO_DRIVER_KMSDRM_DYNAMIC, "$drm_lib", [ ])
                AC_DEFINE_UNQUOTED(SDL_VIDEO_DRIVER_KMSDRM_DYNAMIC_GBM, "$gbm_lib", [ ])
                AC_DEFINE_UNQUOTED(HAVE_KMSDRM_SHARED, "TRUE", [ ])
                SUMMARY_video="${SUMMARY_video} kmsdrm(dynamic)"
            else
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS $LIBDRM_LIBS $LIBGBM_LIBS"
                SUMMARY_video="${SUMMARY_video} kmsdrm"
            fi
            AC_MSG_RESULT($kmsdrm_shared)
            if test x$kmsdrm_shared = xyes; then
                echo "-- dynamic libdrm -> $drm_lib"
                echo "-- dynamic libgbm -> $gbm_lib"
            fi
            have_video=yes
        fi
    fi
}

dnl rcg04172001 Set up the Null video driver.
CheckDummyVideo()
{
    AC_ARG_ENABLE(video-dummy,
[AS_HELP_STRING([--enable-video-dummy], [use dummy video driver [default=yes]])],
                  , enable_video_dummy=yes)
    if test x$enable_video_dummy = xyes; then
        AC_DEFINE(SDL_VIDEO_DRIVER_DUMMY, 1, [ ])
        SOURCES="$SOURCES $srcdir/src/video/dummy/*.c"
        have_video=yes
        SUMMARY_video="${SUMMARY_video} dummy"
    fi
}

CheckOffscreenVideo()
{
    AC_ARG_ENABLE(video-offscreen,
[AS_HELP_STRING([--enable-video-offscreen], [use offscreen video driver [default=yes]])],
                  , enable_video_offscreen=yes)
    if test x$enable_video_offscreen = xyes; then
        AC_DEFINE(SDL_VIDEO_DRIVER_OFFSCREEN, 1, [ ])
        SOURCES="$SOURCES $srcdir/src/video/offscreen/*.c"
        have_video=yes
        SUMMARY_video="${SUMMARY_video} offscreen"
    fi
}

dnl Set up the QNX video driver if enabled
CheckQNXVideo()
{
    if test x$enable_video = xyes; then
        AC_DEFINE(SDL_VIDEO_DRIVER_QNX, 1, [ ])
        SOURCES="$SOURCES $srcdir/src/video/qnx/*.c"
        have_video=yes
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lscreen -lEGL -lGLESv2"
        SUMMARY_video="${SUMMARY_video} qnx"
    fi
}

dnl Set up the QNX audio driver if enabled
CheckQNXAudio()
{
    if test x$enable_audio = xyes; then
        AC_DEFINE(SDL_AUDIO_DRIVER_QSA, 1, [ ])
        SOURCES="$SOURCES $srcdir/src/audio/qsa/*.c"
        have_audio=yes
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lasound"
        SUMMARY_audio="${SUMMARY_audio} qsa"
    fi
}

dnl Check to see if OpenGL support is desired
AC_ARG_ENABLE(video-opengl,
[AS_HELP_STRING([--enable-video-opengl], [include OpenGL support [default=yes]])],
              , enable_video_opengl=yes)

dnl Find GLX
CheckGLX()
{
    if test x$enable_video = xyes -a x$enable_video_opengl = xyes; then
        AC_MSG_CHECKING(for GLX support)
        video_opengl_glx=no
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
         #include <GL/glx.h>
        ]],[])], [video_opengl_glx=yes],[])
        AC_MSG_RESULT($video_opengl_glx)
        if test x$video_opengl_glx = xyes; then
            AC_DEFINE(SDL_VIDEO_OPENGL, 1, [ ])
            AC_DEFINE(SDL_VIDEO_OPENGL_GLX, 1, [ ])
        fi
    fi
}

dnl Check to see if OpenGL ES support is desired
AC_ARG_ENABLE(video-opengles,
[AS_HELP_STRING([--enable-video-opengles], [include OpenGL ES support [default=yes]])],
              , enable_video_opengles=yes)
AC_ARG_ENABLE(video-opengles1,
[AS_HELP_STRING([--enable-video-opengles1], [include OpenGL ES 1.1 support [default=yes]])],
              , enable_video_opengles1=yes)
AC_ARG_ENABLE(video-opengles2,
[AS_HELP_STRING([--enable-video-opengles2], [include OpenGL ES 2.0 support [default=yes]])],
              , enable_video_opengles2=yes)

dnl Find EGL
CheckEGL()
{
    if test x$enable_video = xyes -a x$enable_video_opengl = xyes || test x$enable_video = xyes -a x$enable_video_opengles = xyes; then
        AC_MSG_CHECKING(for EGL support)
        video_opengl_egl=no
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
          #define LINUX
          #define EGL_API_FB
          #define MESA_EGL_NO_X11_HEADERS
          #define EGL_NO_X11
          #include <EGL/egl.h>
          #include <EGL/eglext.h>
        ]],[])], [video_opengl_egl=yes],[])
        AC_MSG_RESULT($video_opengl_egl)
        if test x$video_opengl_egl = xyes; then
            AC_DEFINE(SDL_VIDEO_OPENGL_EGL, 1, [ ])
        fi
    fi
}

dnl Find OpenGL
CheckOpenGL()
{
    if test x$enable_video = xyes -a x$enable_video_opengl = xyes; then
        AC_MSG_CHECKING(for OpenGL headers)
        video_opengl=no
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
         #include <GL/gl.h>
         #include <GL/glext.h>
        ]],[])], [video_opengl=yes],[])
        AC_MSG_RESULT($video_opengl)
        if test x$video_opengl = xyes; then 
            AC_DEFINE(SDL_VIDEO_OPENGL, 1, [ ])
            AC_DEFINE(SDL_VIDEO_RENDER_OGL, 1, [ ])
            SUMMARY_video="${SUMMARY_video} opengl"
        fi
    fi
}

dnl Find OpenGL ES
CheckOpenGLES()
{
    if test x$enable_video = xyes -a x$enable_video_opengles = xyes; then
        if test x$enable_video_opengles1 = xyes; then
            AC_MSG_CHECKING(for OpenGL ES v1 headers)
            video_opengles_v1=no
            AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
             #include <GLES/gl.h>
             #include <GLES/glext.h>
            ]],[])], [video_opengles_v1=yes],[])
            AC_MSG_RESULT($video_opengles_v1)
            if test x$video_opengles_v1 = xyes; then
                AC_DEFINE(SDL_VIDEO_OPENGL_ES, 1, [ ])
                AC_DEFINE(SDL_VIDEO_RENDER_OGL_ES, 1, [ ])
                SUMMARY_video="${SUMMARY_video} opengl_es1"
            fi
        fi

        if test x$enable_video_opengles2 = xyes; then
            AC_MSG_CHECKING(for OpenGL ES v2 headers)
            video_opengles_v2=no
            AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
             #include <GLES2/gl2.h>
             #include <GLES2/gl2ext.h>
            ]],[])], [video_opengles_v2=yes],[])
            AC_MSG_RESULT($video_opengles_v2)
            if test x$video_opengles_v2 = xyes; then
                AC_DEFINE(SDL_VIDEO_OPENGL_ES2, 1, [ ])
                AC_DEFINE(SDL_VIDEO_RENDER_OGL_ES2, 1, [ ])
                SUMMARY_video="${SUMMARY_video} opengl_es2"
            fi
        fi
    fi
}

dnl Check for Windows OpenGL
CheckWINDOWSGL()
{
    if test x$enable_video = xyes -a x$enable_video_opengl = xyes; then
        AC_DEFINE(SDL_VIDEO_OPENGL, 1, [ ])
        AC_DEFINE(SDL_VIDEO_OPENGL_WGL, 1, [ ])
        AC_DEFINE(SDL_VIDEO_RENDER_OGL, 1, [ ])
        SUMMARY_video="${SUMMARY_video} opengl"
    fi
}

dnl Check for Windows OpenGL
CheckWINDOWSGLES()
{
    if test x$enable_video = xyes -a x$enable_video_opengles = xyes; then

        AC_MSG_CHECKING(for EGL support)
        video_opengl_egl=no
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
         #include <EGL/egl.h>
        ]],[])], [video_opengl_egl=yes],[])
        AC_MSG_RESULT($video_opengl_egl)
        if test x$video_opengl_egl = xyes; then
            AC_DEFINE(SDL_VIDEO_OPENGL, 1, [ ])
            AC_DEFINE(SDL_VIDEO_OPENGL_EGL, 1, [ ])
            SUMMARY_video="${SUMMARY_video} opengl_es1"
        fi

        AC_MSG_CHECKING(for OpenGL ES v2 headers)
        video_opengles_v2=no
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
         #include <GLES2/gl2.h>
         #include <GLES2/gl2ext.h>
        ]],[])], [video_opengles_v2=yes],[])
        AC_MSG_RESULT($video_opengles_v2)
        if test x$video_opengles_v2 = xyes; then
            AC_DEFINE(SDL_VIDEO_OPENGL, 1, [ ])
            AC_DEFINE(SDL_VIDEO_OPENGL_ES2, 1, [ ])
            AC_DEFINE(SDL_VIDEO_RENDER_OGL_ES2, 1, [ ])
            SUMMARY_video="${SUMMARY_video} opengl_es2"
        fi
    fi
}

dnl Check for Haiku OpenGL
CheckHaikuGL()
{
    if test x$enable_video = xyes -a x$enable_video_opengl = xyes; then
        AC_DEFINE(SDL_VIDEO_OPENGL, 1, [ ])
        AC_DEFINE(SDL_VIDEO_OPENGL_HAIKU, 1, [ ])
        AC_DEFINE(SDL_VIDEO_RENDER_OGL, 1, [ ])
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lGL"
        SUMMARY_video="${SUMMARY_video} opengl"
    fi
}

dnl Check for MacOS OpenGL
CheckMacGL()
{
    if test x$enable_video = xyes -a x$enable_video_opengl = xyes; then
        AC_DEFINE(SDL_VIDEO_OPENGL, 1, [ ])
        AC_DEFINE(SDL_VIDEO_OPENGL_CGL, 1, [ ])
        AC_DEFINE(SDL_VIDEO_RENDER_OGL, 1, [ ])
        SUMMARY_video="${SUMMARY_video} opengl"
    fi
}

dnl Check for MacOS OpenGLES
CheckMacGLES()
{
    if test x$enable_video = xyes -a x$enable_video_opengles = xyes; then
        video_opengl_egl=yes
        AC_DEFINE(SDL_VIDEO_OPENGL_EGL, 1, [ ])
        video_opengles_v2=yes
        AC_DEFINE(SDL_VIDEO_OPENGL_ES2, 1, [ ])
        AC_DEFINE(SDL_VIDEO_RENDER_OGL_ES2, 1, [ ])
        SUMMARY_video="${SUMMARY_video} opengl_es2"
    fi
}

CheckEmscriptenGLES()
{
    if test x$enable_video = xyes -a x$enable_video_opengles = xyes; then
        AC_MSG_CHECKING(for EGL support)
        video_opengl_egl=no
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
         #include <EGL/egl.h>
        ]],[])], [video_opengl_egl=yes],[])
        AC_MSG_RESULT($video_opengl_egl)
        if test x$video_opengl_egl = xyes; then
            AC_DEFINE(SDL_VIDEO_OPENGL_EGL, 1, [ ])
        fi

        AC_MSG_CHECKING(for OpenGL ES v2 headers)
        video_opengles_v2=no
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
         #include <GLES2/gl2.h>
         #include <GLES2/gl2ext.h>
        ]],[])], [video_opengles_v2=yes],[])
        AC_MSG_RESULT($video_opengles_v2)
        if test x$video_opengles_v2 = xyes; then
            AC_DEFINE(SDL_VIDEO_OPENGL_ES2, 1, [ ])
            AC_DEFINE(SDL_VIDEO_RENDER_OGL_ES2, 1, [ ])
            SUMMARY_video="${SUMMARY_video} opengl_es2"
        fi
    fi
}

dnl Check to see if Vulkan support is desired
AC_ARG_ENABLE(video-vulkan,
[AS_HELP_STRING([--enable-video-vulkan], [include Vulkan support [default=yes]])],
              , enable_video_vulkan=yes)

dnl Find Vulkan Header
CheckVulkan()
{
    if test x$enable_video = xyes -a x$enable_video_vulkan = xyes; then
        case "$host" in
            *-*-android*)
                AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                  #if defined(__ARM_ARCH) && __ARM_ARCH < 7
                  #error Vulkan doesn't work on this configuration
                  #endif
                ]],[])], [],[enable_video_vulkan=no])
                ;;
            *-*-darwin*)
                save_CFLAGS="$CFLAGS"
dnl             Work around that we don't have Objective-C support in autoconf
                CFLAGS="$CFLAGS -x objective-c"
                AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                  #include <Cocoa/Cocoa.h>
                  #include <Metal/Metal.h>
                  #include <QuartzCore/CAMetalLayer.h>

                  #if TARGET_CPU_X86
                  #error Vulkan doesn't work on this configuration
                  #endif
                ]],[])], [],[enable_video_vulkan=no])
                CFLAGS="$save_CFLAGS"
                ;;
            *)
                ;;
        esac
        if test x$enable_video_vulkan = xno; then
            # For reasons I am totally unable to see, I get an undefined macro error if
            # I put this in the AC_TRY_COMPILE.
            AC_MSG_WARN([Vulkan does not work on this configuration.])
        fi
    fi
    if test x$have_loadso != xyes; then
        AC_MSG_WARN([Vulkan support is available, but disabled because there's no loadso.])
        enable_video_vulkan=no
    fi
    if test x$enable_video_vulkan = xyes; then
        AC_DEFINE(SDL_VIDEO_VULKAN, 1, [ ])
        SUMMARY_video="${SUMMARY_video} vulkan"
    fi
}

dnl See if we can use the new unified event interface in Linux 2.4
CheckInputEvents()
{
dnl Check for Linux 2.4 unified input event interface support
        AC_CHECK_HEADERS(linux/input.h)

        AC_MSG_CHECKING(for Linux 2.4 unified input interface)
        use_input_events=no
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
          #include <linux/input.h>
        ]], [[
          #ifndef EVIOCGNAME
          #error EVIOCGNAME() ioctl not available
          #endif
        ]])], [use_input_events=yes],[])
        AC_MSG_RESULT($use_input_events)
        if test x$use_input_events = xyes; then
            AC_DEFINE(SDL_INPUT_LINUXEV, 1, [ ])
            SUMMARY_input="${SUMMARY_input} linuxev"
        fi
}

dnl See if we can use the kernel kd.h header
CheckInputKD()
{
    AC_MSG_CHECKING(for Linux kd.h)
    use_input_kd=no
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
      #include <linux/kd.h>
      #include <linux/keyboard.h>
      #include <sys/ioctl.h>
    ]], [[
        struct kbentry kbe;
        kbe.kb_table = KG_CTRL;
        ioctl(0, KDGKBENT, &kbe);
    ]])], [use_input_kd=yes],[])
    AC_MSG_RESULT($use_input_kd)
    if test x$use_input_kd = xyes; then
        AC_DEFINE(SDL_INPUT_LINUXKD, 1, [ ])
        SUMMARY_input="${SUMMARY_input} linuxkd"
    fi
}

dnl See if we can use the FreeBSD kernel kbio.h header
CheckInputKBIO()
{
    AC_MSG_CHECKING(for FreeBSD kbio.h)
    use_input_kbio=no
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
      #include <sys/kbio.h>
      #include <sys/ioctl.h>
    ]], [[
        accentmap_t accTable;
        ioctl(0, KDENABIO, 1);
    ]])], [use_input_kbio=yes],[])
    AC_MSG_RESULT($use_input_kbio)
    if test x$use_input_kbio = xyes; then
        AC_DEFINE(SDL_INPUT_FBSDKBIO, 1, [ ])
        SUMMARY_input="${SUMMARY_input} fbsdkbio"
    fi
}

dnl See if we can use the wscons input driver
CheckInputWSCONS()
{
    AC_MSG_CHECKING(for OpenBSD wscons)
    use_input_wscons=no
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
     #include <sys/time.h>
     #include <dev/wscons/wsconsio.h>
     #include <dev/wscons/wsksymdef.h>
     #include <dev/wscons/wsksymvar.h>
     #include <sys/ioctl.h>
    ]], [[
     struct wskbd_map_data data;
     ioctl(0, WSKBDIO_GETMAP, &data);
    ]])], [use_input_wscons=yes],[])
    AC_MSG_RESULT($use_input_wscons)
    if test x$use_input_wscons = xyes; then
        AC_DEFINE(SDL_INPUT_WSCONS, 1, [ ])
        SUMMARY_input="${SUMMARY_input} wscons"
    fi
}

dnl See if the platform offers libudev for device enumeration and hotplugging.
CheckLibUDev()
{
    AC_ARG_ENABLE(libudev,
[AS_HELP_STRING([--enable-libudev], [enable libudev support [default=yes]])],
                        , enable_libudev=yes)
    if test x$enable_libudev = xyes; then
        AC_CHECK_HEADER(libudev.h,
                        have_libudev_h_hdr=yes,
                        have_libudev_h_hdr=no)
        if test x$have_libudev_h_hdr = xyes; then
            AC_DEFINE(HAVE_LIBUDEV_H, 1, [ ])

            udev_lib=[`find_lib "libudev.so.*" "" | sed 's/.*\/\(.*\)/\1/; q'`]
            if test x$udev_lib != x; then
                echo "-- dynamic udev -> $udev_lib"
                AC_DEFINE_UNQUOTED(SDL_UDEV_DYNAMIC, "$udev_lib", [ ])
            fi
        fi
    fi
}

dnl See if the platform offers libdbus for various IPC techniques.
CheckDBus()
{
    AC_ARG_ENABLE(dbus,
[AS_HELP_STRING([--enable-dbus], [enable D-Bus support [default=yes]])],
                        , enable_dbus=yes)
    if test x$enable_dbus = xyes; then
        PKG_CHECK_MODULES([DBUS], [dbus-1], have_dbus=yes, have_dbus=no)
        save_CPPFLAGS="$CPPFLAGS"
        CPPFLAGS="$save_CPPFLAGS $DBUS_CFLAGS"
        AC_CHECK_HEADER(dbus/dbus.h,
                        have_dbus_dbus_h_hdr=yes,
                        have_dbus_dbus_h_hdr=no)
        CPPFLAGS="$save_CPPFLAGS"
        if test x$have_dbus_dbus_h_hdr = xyes; then
            AC_DEFINE(HAVE_DBUS_DBUS_H, 1, [ ])
            EXTRA_CFLAGS="$EXTRA_CFLAGS $DBUS_CFLAGS"
            SOURCES="$SOURCES $srcdir/src/core/linux/SDL_dbus.c"
        fi
    fi
}

dnl See if the platform wanna IME support.
CheckIME()
{
    AC_ARG_ENABLE(ime,
[AS_HELP_STRING([--enable-ime], [enable IME support [default=yes]])],
                  , enable_ime=yes)
    if test x$enable_ime = xyes; then
        AC_DEFINE(SDL_USE_IME, 1, [ ])
        SOURCES="$SOURCES $srcdir/src/core/linux/SDL_ime.c"
    fi
}

dnl Check inotify presense
CheckInotify()
{
    save_LIBS="$LIBS"
    case "$host" in
     *-*-freebsd*|*-*dragonfly*) LIBS="$LIBS -linotify"
      ;;
    esac
    AC_CHECK_HEADERS(sys/inotify.h, [have_inotify_inotify_h_hdr=yes])
    AC_CHECK_FUNCS(inotify_init, [have_inotify=yes])
    AC_CHECK_FUNCS(inotify_init1)
    if test x$have_inotify_inotify_h_hdr = xyes -a x$have_inotify = xyes; then
        AC_DEFINE(HAVE_INOTIFY, 1, [ ])
        case "$host" in
            *-*-freebsd*|*-*-dragonfly*)
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS -linotify"
                ;;
        esac
    fi
    LIBS="$save_LIBS"
}

dnl See if the platform has libibus IME support.
CheckIBus()
{
    AC_ARG_ENABLE(ibus,
[AS_HELP_STRING([--enable-ibus], [enable IBus support [default=yes]])],
                  , enable_ibus=yes)
    if test x$enable_ibus = xyes; then
        PKG_CHECK_MODULES([IBUS], [ibus-1.0], have_ibus=yes, have_ibus=no)
        save_CPPFLAGS="$CPPFLAGS"
        CPPFLAGS="$save_CPPFLAGS $IBUS_CFLAGS"
        AC_CHECK_HEADER(ibus-1.0/ibus.h,
                        have_ibus_ibus_h_hdr=yes,
                        have_ibus_ibus_h_hdr=no)
        CPPFLAGS="$save_CPPFLAGS"
        if test x$have_ibus_ibus_h_hdr = xyes; then
            if test x$enable_ime != xyes; then
                AC_MSG_WARN([IME support is required for IBus.])
                have_ibus_ibus_h_hdr=no
            elif test x$enable_dbus != xyes; then
                AC_MSG_WARN([DBus support is required for IBus.])
                have_ibus_ibus_h_hdr=no
            elif test x$have_inotify_inotify_h_hdr != xyes; then
                AC_MSG_WARN([INotify support is required for IBus.])
                have_ibus_ibus_h_hdr=no
            else
                AC_DEFINE(HAVE_IBUS_IBUS_H, 1, [ ])
                EXTRA_CFLAGS="$EXTRA_CFLAGS $IBUS_CFLAGS"
                SOURCES="$SOURCES $srcdir/src/core/linux/SDL_ibus.c"
            fi
        fi
    fi
}

dnl See if the platform has fcitx IME support.
CheckFcitx()
{
    AC_ARG_ENABLE(fcitx,
[AS_HELP_STRING([--enable-fcitx], [enable fcitx support [default=yes]])],
                  , enable_fcitx=yes)
    if test x$enable_fcitx = xyes; then
        AC_MSG_CHECKING(for fcitx support)
        have_fcitx=no
        if test x$enable_ime != xyes; then
            AC_MSG_WARN([IME support is required for fcitx.])
        elif test x$have_dbus_dbus_h_hdr != xyes; then
            AC_MSG_WARN([DBus support is required for fcitx.])
        else
            have_fcitx=yes
            AC_DEFINE(HAVE_FCITX, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/core/linux/SDL_fcitx.c"
        fi
        AC_MSG_RESULT($have_fcitx)
    fi
}

dnl Check to see if GameController framework support is desired
CheckJoystickMFI()
{
    AC_ARG_ENABLE(joystick-mfi,
[AS_HELP_STRING([--enable-joystick-mfi], [include macOS MFI joystick support [default=yes]])],
              , enable_joystick_mfi=yes)

    if test x$enable_joystick_mfi = xyes; then
        save_CFLAGS="$CFLAGS"
        save_LDFLAGS="$LDFLAGS"
dnl     Work around that we don't have Objective-C support in autoconf
        CFLAGS="$CFLAGS -x objective-c -fobjc-weak"
        LDFLAGS="$LDFLAGS -Wl,-weak_framework,CoreHaptics -Wl,-weak_framework,GameController"
        AC_MSG_CHECKING(for GameController framework)
        enable_joystick_mfi=no
        AC_LINK_IFELSE([AC_LANG_PROGRAM([[
          #include <AvailabilityMacros.h>
          #include <TargetConditionals.h>
          #import <GameController/GameController.h>
        ]], [[
          #if MAC_OS_X_VERSION_MIN_REQUIRED < 1080
          #error GameController framework doesn't work on this configuration
          #endif
          #if TARGET_CPU_X86
          #error GameController framework doesn't work on this configuration
          #endif
        ]])], [enable_joystick_mfi=yes],[])
        CFLAGS="$save_CFLAGS"
        LDFLAGS="$save_LDFLAGS"

        AC_MSG_RESULT($enable_joystick_mfi)
        if test x$enable_joystick_mfi = xyes; then
            AC_DEFINE(SDL_JOYSTICK_MFI, 1, [ ])
            EXTRA_CFLAGS="$EXTRA_CFLAGS -fobjc-weak -Wno-unused-command-line-argument"
            EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-weak_framework,CoreHaptics -Wl,-weak_framework,GameController"
        fi
    fi
}

dnl See what type of thread model to use on Linux and Solaris
CheckPTHREAD()
{
dnl Check for pthread support

dnl Emscripten pthreads work, but you need to have a non-pthread fallback build
dnl  for systems without support. It's not currently enough to not use
dnl  pthread functions in a pthread-build; it won't start up on unsupported
dnl  browsers. As such, you have to explicitly enable it on Emscripten builds
dnl  for the time being. This default with change to ON once this becomes
dnl  commonly supported in browsers or the Emscripten teams makes a single
dnl  binary work everywhere.

    case "$host" in
        *-*-emscripten*)
            enable_pthreads_default=no
            ;;
        *)
            enable_pthreads_default=yes
            ;;
    esac

    AC_ARG_ENABLE(pthreads,
[AS_HELP_STRING([--enable-pthreads], [use POSIX threads for multi-threading [default=maybe]])],
                  , enable_pthreads=maybe)
dnl This is used on Linux for glibc binary compatibility (Doh!)
    AC_ARG_ENABLE(pthread-sem,
[AS_HELP_STRING([--enable-pthread-sem], [use pthread semaphores [default=maybe]])],
                  , enable_pthread_sem=maybe)

    if test x$enable_pthreads = xmaybe; then
        enable_pthreads=$enable_pthreads_default
    fi
    if test x$enable_pthread_sem = xmaybe; then
        enable_pthread_sem=$enable_pthreads
    fi

    case "$host" in
         *-*-android*)
            pthread_cflags="-D_REENTRANT -D_THREAD_SAFE"
            pthread_lib=""
            ;;
        *-*-linux*|*-*-uclinux*)
            pthread_cflags="-D_REENTRANT"
            pthread_lib="-lpthread"
            ;;
        *-*-bsdi*)
            pthread_cflags="-D_REENTRANT -D_THREAD_SAFE"
            pthread_lib=""
            ;;
        *-*-darwin*)
            pthread_cflags="-D_THREAD_SAFE"
# causes Carbon.p complaints?
#            pthread_cflags="-D_REENTRANT -D_THREAD_SAFE"
            ;;
        *-*-freebsd*|*-*-dragonfly*)
            pthread_cflags="-D_REENTRANT -D_THREAD_SAFE"
            pthread_lib="-pthread"
            ;;
        *-*-netbsd*)
            pthread_cflags="-D_REENTRANT -D_THREAD_SAFE"
            pthread_lib="-lpthread"
            ;;
        *-*-openbsd*)
            pthread_cflags="-D_REENTRANT"
            pthread_lib="-lpthread"
            ;;
        *-*-solaris2.9)
            # From Solaris 9+, posix4's preferred name is rt.
            pthread_cflags="-D_REENTRANT"
            pthread_lib="-lpthread -lrt"
            ;;
        *-*-solaris2.10)
            # Solaris 10+ merged pthread into libc.
            pthread_cflags="-D_REENTRANT"
            pthread_lib="-lrt"
            ;;
        *-*-solaris*)
            # Solaris 11+ merged rt into libc.
            pthread_cflags="-D_REENTRANT"
            pthread_lib=""
            ;;
        *-*-sysv5*)
            pthread_cflags="-D_REENTRANT -Kthread"
            pthread_lib=""
            ;;
        *-*-aix*)
            pthread_cflags="-D_REENTRANT -mthreads"
            pthread_lib="-lpthread"
            ;;
        *-*-hpux11*)
            pthread_cflags="-D_REENTRANT"
            pthread_lib="-L/usr/lib -lpthread"
            ;;
        *-*-haiku*)
            pthread_cflags="-D_REENTRANT"
            pthread_lib=""
            ;;
        *-*-nto*)
            pthread_cflags="-D_REENTRANT"
            pthread_lib=""
            ;;
        *-*-emscripten*)
            pthread_cflags="-D_REENTRANT -pthread"
            pthread_lib="-pthread"
            ;;
        *)
            pthread_cflags="-D_REENTRANT"
            pthread_lib="-lpthread"
            ;;
    esac
    if test x$enable_threads = xyes -a x$enable_pthreads = xyes; then
        # Save the original compiler flags and libraries
        ac_save_cflags="$CFLAGS"; ac_save_libs="$LIBS"
        # Add the pthread compiler flags and libraries
        CFLAGS="$CFLAGS $pthread_cflags"; LIBS="$LIBS $pthread_lib"
        # Check to see if we have pthread support on this system
        AC_MSG_CHECKING(for pthreads)
        use_pthreads=no
        AC_LINK_IFELSE([AC_LANG_PROGRAM([[
         #include <pthread.h>
        ]], [[
         pthread_attr_t type;
         pthread_attr_init(&type);
        ]])], [use_pthreads=yes],[])
        AC_MSG_RESULT($use_pthreads)
        # Restore the compiler flags and libraries
        CFLAGS="$ac_save_cflags"; LIBS="$ac_save_libs"

        # Do futher testing if we have pthread support...
        if test x$use_pthreads = xyes; then
            AC_DEFINE(SDL_THREAD_PTHREAD, 1, [ ])
            EXTRA_CFLAGS="$EXTRA_CFLAGS $pthread_cflags"
            EXTRA_LDFLAGS="$EXTRA_LDFLAGS $pthread_lib"
            SDL_CFLAGS="$SDL_CFLAGS $pthread_cflags"

            # Save the original compiler flags and libraries
            ac_save_cflags="$CFLAGS"; ac_save_libs="$LIBS"
            # Add the pthread compiler flags and libraries
            CFLAGS="$CFLAGS $pthread_cflags"; LIBS="$LIBS $pthread_lib"

            # Check to see if recursive mutexes are available
            AC_MSG_CHECKING(for recursive mutexes)
            has_recursive_mutexes=no
            if test x$has_recursive_mutexes = xno; then
                AC_LINK_IFELSE([AC_LANG_PROGRAM([[
                  #define _GNU_SOURCE 1
                  #include <pthread.h>
                ]], [[
                  pthread_mutexattr_t attr;
                  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
                ]])], [
                has_recursive_mutexes=yes
                AC_DEFINE(SDL_THREAD_PTHREAD_RECURSIVE_MUTEX, 1, [ ])
                ],[])
            fi
            if test x$has_recursive_mutexes = xno; then
                AC_LINK_IFELSE([AC_LANG_PROGRAM([[
                  #define _GNU_SOURCE 1
                  #include <pthread.h>
                ]], [[
                  pthread_mutexattr_t attr;
                  pthread_mutexattr_setkind_np(&attr, PTHREAD_MUTEX_RECURSIVE_NP);
                ]])],[
                has_recursive_mutexes=yes
                AC_DEFINE(SDL_THREAD_PTHREAD_RECURSIVE_MUTEX_NP, 1, [ ])
                ],[])
            fi
            AC_MSG_RESULT($has_recursive_mutexes)

            # Check to see if pthread semaphore support is missing
            if test x$enable_pthread_sem = xyes; then
                AC_MSG_CHECKING(for pthread semaphores)
                have_pthread_sem=no
                AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                  #include <pthread.h>
                  #include <semaphore.h>
                ]],[])], [have_pthread_sem=yes],[])
                AC_MSG_RESULT($have_pthread_sem)
            fi
            if test x$have_pthread_sem = xyes; then
                AC_MSG_CHECKING(for sem_timedwait)
                have_sem_timedwait=no
                AC_LINK_IFELSE([AC_LANG_PROGRAM([[
                  #include <pthread.h>
                  #include <semaphore.h>
                ]], [[
                  sem_timedwait(NULL, NULL);
                ]])], [
                have_sem_timedwait=yes
                AC_DEFINE([HAVE_SEM_TIMEDWAIT], 1, [ ])
                ],[])
                AC_MSG_RESULT($have_sem_timedwait)
            fi

            AC_CHECK_HEADER(pthread_np.h, have_pthread_np_h=yes, have_pthread_np_h=no, [ #include <pthread.h> ])
            if test x$have_pthread_np_h = xyes; then
                AC_DEFINE(HAVE_PTHREAD_NP_H, 1, [ ])
            fi

            # Check to see if pthread naming is available
            AC_MSG_CHECKING(for pthread_setname_np)
            AC_TRY_LINK_FUNC(pthread_setname_np, [
              has_pthread_setname_np=yes
              AC_DEFINE(HAVE_PTHREAD_SETNAME_NP, 1, [ ])
            ],[
              has_pthread_setname_np=no
            ])
            AC_MSG_RESULT($has_pthread_setname_np)

            AC_MSG_CHECKING(for pthread_set_name_np)
            AC_TRY_LINK_FUNC(pthread_set_name_np, [
              has_pthread_set_name_np=yes
              AC_DEFINE(HAVE_PTHREAD_SET_NAME_NP, 1, [ ])
            ],[
              has_pthread_set_name_np=no
            ])
            AC_MSG_RESULT($has_pthread_set_name_np)

            # Restore the compiler flags and libraries
            CFLAGS="$ac_save_cflags"; LIBS="$ac_save_libs"

            # Basic thread creation functions
            SOURCES="$SOURCES $srcdir/src/thread/pthread/SDL_systhread.c"

            # Semaphores
            # We can fake these with mutexes and condition variables if necessary
            if test x$have_pthread_sem = xyes; then
                SOURCES="$SOURCES $srcdir/src/thread/pthread/SDL_syssem.c"
            else
                SOURCES="$SOURCES $srcdir/src/thread/generic/SDL_syssem.c"
            fi

            # Mutexes
            # We can fake these with semaphores if necessary
            SOURCES="$SOURCES $srcdir/src/thread/pthread/SDL_sysmutex.c"

            # Condition variables
            # We can fake these with semaphores and mutexes if necessary
            SOURCES="$SOURCES $srcdir/src/thread/pthread/SDL_syscond.c"

            # Thread local storage
            SOURCES="$SOURCES $srcdir/src/thread/pthread/SDL_systls.c"

            have_threads=yes
        fi
    fi
}

dnl Determine whether the compiler can produce Windows executables
CheckWINDOWS()
{
    AC_MSG_CHECKING(Windows compiler)
    have_win32_gcc=no
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
     #include <windows.h>
    ]],[])], [have_win32_gcc=yes],[])
    AC_MSG_RESULT($have_win32_gcc)
    if test x$have_win32_gcc != xyes; then
       AC_MSG_ERROR([
*** Your compiler ($CC) does not produce Windows executables!
       ])
    fi

    AC_MSG_CHECKING(Windows CE)
    have_wince=no
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#if !defined(_WIN32_WCE) && !defined(__MINGW32CE__)
#error This is not Windows CE
#endif
    ]],[])], [
    have_wince=yes
       AC_MSG_ERROR([
*** Sorry, Windows CE is no longer supported.
       ])
    ],[])
    AC_MSG_RESULT($have_wince)

    # headers needed elsewhere
    AC_CHECK_HEADER(tpcshrd.h,have_tpcshrd_h=yes)
    if test x$have_tpcshrd_h = xyes; then
        AC_DEFINE(HAVE_TPCSHRD_H, 1, [ ])
    fi
    AC_CHECK_HEADER(roapi.h,have_roapi_h=yes)
    if test x$have_roapi_h = xyes; then
        AC_DEFINE(HAVE_ROAPI_H, 1, [ ])
    fi
    AC_CHECK_HEADER(shellscalingapi.h,have_shellscalingapi_h=yes)
    if test x$shellscalingapi_h = xyes; then
        AC_DEFINE(HAVE_SHELLSCALINGAPI_H, 1, [ ])
    fi
}

dnl Determine whether the compiler can produce OS/2 executables
CheckOS2()
{
    AC_MSG_CHECKING(OS/2 compiler)
    have_os2_gcc=no
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <os2.h>]],
     [])],[have_os2_gcc=yes],[])
    AC_MSG_RESULT($have_os2_gcc)
    if test x$have_os2_gcc != xyes; then
       AC_MSG_ERROR([
*** Your compiler ($CC) does not produce OS/2 executables!
       ])
    fi
}

dnl Find the DirectX includes and libraries
CheckDIRECTX()
{
    AC_ARG_ENABLE(directx,
[AS_HELP_STRING([--enable-directx], [use DirectX for Windows audio/video [default=yes]])],
                                 , enable_directx=yes)
    if test x$enable_directx = xyes; then
        AC_CHECK_HEADER(d3d9.h, have_d3d=yes)
        AC_CHECK_HEADER(d3d11_1.h, have_d3d11=yes)
        AC_MSG_CHECKING(for compatible d3d12 headers)
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#include <d3d12.h>
#include <d3d12sdklayers.h>
ID3D12Device1 *device;
            ]])], [have_d3d12=yes],[have_d3d12=no])
        AC_MSG_RESULT($have_d3d12)

        AC_CHECK_HEADER(ddraw.h, have_ddraw=yes)
        AC_CHECK_HEADER(dsound.h, have_dsound=yes)
        AC_CHECK_HEADER(dinput.h, have_dinput=yes)
        AC_CHECK_HEADER(dxgi.h, have_dxgi=yes)

        if test x$have_ddraw = xyes; then
            AC_DEFINE(HAVE_DDRAW_H, 1, [ ])
        fi
        if test x$have_dinput = xyes; then
            AC_DEFINE(HAVE_DINPUT_H, 1, [ ])
        fi
        if test x$have_dsound = xyes; then
            AC_DEFINE(HAVE_DSOUND_H, 1, [ ])
        fi
        if test x$have_dxgi = xyes; then
            AC_DEFINE(HAVE_DXGI_H, 1, [ ])
        fi

        # FIXME: latest Cygwin finds dinput headers, but we die on other win32 headers.
        # FIXME:  ...so force it off for now.
        case "$host" in
            *-*-cygwin*)
            have_dinput=false
            ;;
        esac
    fi

    AC_ARG_ENABLE(xinput,
[AS_HELP_STRING([--enable-xinput], [use Xinput for Windows [default=yes]])],
                                , enable_xinput=yes)
    if test x$enable_xinput = xyes; then
        AC_CHECK_HEADER(xinput.h, have_xinput=yes)
        if test x$have_xinput = xyes; then
            AC_DEFINE(HAVE_XINPUT_H, 1, [ ])
        fi
    fi

    AC_MSG_CHECKING(for windows.gaming.input.h)
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#define COBJMACROS
#include <windows.gaming.input.h>
__x_ABI_CWindows_CGaming_CInput_CIGamepadStatics2 *s2;
        ]],[])], [have_wgi=yes],[have_wgi=no])
    AC_MSG_RESULT($have_wgi)

    if test x$have_wgi = xyes; then
        AC_DEFINE(HAVE_WINDOWS_GAMING_INPUT_H, 1, [ ])
    fi

    AC_CHECK_HEADER(mmdeviceapi.h, have_wasapi=yes)
    if test x$have_wasapi = xyes; then
        AC_DEFINE(HAVE_MMDEVICEAPI_H, 1, [ ])
    fi
    AC_CHECK_HEADER(audioclient.h,,have_wasapi=no)
    if test x$have_wasapi = xyes; then
        AC_DEFINE(HAVE_AUDIOCLIENT_H, 1, [ ])
    fi

    AC_ARG_ENABLE(wasapi,
[AS_HELP_STRING([--enable-wasapi], [use the Windows WASAPI audio driver [default=yes]])],
                                , enable_wasapi=yes)
}

dnl Check for the dlfcn.h interface for dynamically loading objects
dnl NOTE: CheckDLOPEN is called only for relevant platforms
CheckDLOPEN()
{
    AC_DEFINE(DYNAPI_NEEDS_DLOPEN, 1, [ ])

    AC_CHECK_HEADER(dlfcn.h,have_dlfcn_h=yes,have_dlfcn_h=no)

    have_dlopen=no
    AC_CHECK_LIB(c, dlopen, have_dlopen=yes,
       AC_CHECK_LIB(dl, dlopen, [have_dlopen=yes; EXTRA_LDFLAGS="$EXTRA_LDFLAGS -ldl"]))
    AC_MSG_CHECKING(for dlopen)
    AC_MSG_RESULT($have_dlopen)

    if test x$have_dlfcn_h = xyes -a x$have_dlopen = xyes; then
       AC_DEFINE(HAVE_DLOPEN, 1, [ ])
       if test x$enable_loadso = xyes; then
          AC_DEFINE(SDL_LOADSO_DLOPEN, 1, [ ])
          SOURCES="$SOURCES $srcdir/src/loadso/dlopen/*.c"
          have_loadso=yes
       fi
    fi
}

CheckO_CLOEXEC()
{
    AC_MSG_CHECKING(for O_CLOEXEC)
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#include <fcntl.h>
int flag = O_CLOEXEC;
        ]],[])], [have_o_cloexec=yes],[have_o_cloexec=no])
    AC_MSG_RESULT($have_o_cloexec)
    if test $have_o_cloexec = yes; then
       AC_DEFINE(HAVE_O_CLOEXEC, 1, [ ])
    fi
}

dnl Check for the usbhid(3) library on *BSD
CheckUSBHID()
{
    case "$host" in
        *-*-*bsd*|*-*-dragonfly*)
            if test x$enable_joystick = xyes; then
                AC_CHECK_LIB(usbhid, hid_init, have_libusbhid=yes)
                if test x$have_libusbhid = xyes; then
                    AC_CHECK_HEADER(usbhid.h, [USB_CFLAGS="-DHAVE_USBHID_H"])
                    AC_CHECK_HEADER(libusbhid.h, [USB_CFLAGS="-DHAVE_LIBUSBHID_H"])
                    USB_LIBS="$USB_LIBS -lusbhid"
                else
                    AC_CHECK_HEADER(usb.h, [USB_CFLAGS="-DHAVE_USB_H"])
                    AC_CHECK_HEADER(libusb.h, [USB_CFLAGS="-DHAVE_LIBUSB_H"])
                    AC_CHECK_LIB(usb, hid_init, [USB_LIBS="$USB_LIBS -lusb"])
                fi

                save_CFLAGS="$CFLAGS"
                CFLAGS="$CFLAGS $USB_CFLAGS"

                AC_MSG_CHECKING(for usbhid)
                have_usbhid=no
                AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                  #include <sys/types.h>
                  #if defined(HAVE_USB_H)
                  #include <usb.h>
                  #endif
                  #ifdef __DragonFly__
                  # include <bus/u4b/usb.h>
                  # include <bus/u4b/usbhid.h>
                  #else
                  # include <dev/usb/usb.h>
                  # include <dev/usb/usbhid.h>
                  #endif
                  #if defined(HAVE_USBHID_H)
                  #include <usbhid.h>
                  #elif defined(HAVE_LIBUSB_H)
                  #include <libusb.h>
                  #elif defined(HAVE_LIBUSBHID_H)
                  #include <libusbhid.h>
                  #endif
                ]], [[
                  struct report_desc *repdesc;
                  struct usb_ctl_report *repbuf;
                  hid_kind_t hidkind;
                ]])], [have_usbhid=yes],[])
                AC_MSG_RESULT($have_usbhid)

                if test x$have_usbhid = xyes; then
                    AC_MSG_CHECKING(for ucr_data member of usb_ctl_report)
                    have_usbhid_ucr_data=no
                    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                      #include <sys/types.h>
                      #if defined(HAVE_USB_H)
                      #include <usb.h>
                      #endif
                      #ifdef __DragonFly__
                      # include <bus/u4b/usb.h>
                      # include <bus/u4b/usbhid.h>
                      #else
                      # include <dev/usb/usb.h>
                      # include <dev/usb/usbhid.h>
                      #endif
                      #if defined(HAVE_USBHID_H)
                      #include <usbhid.h>
                      #elif defined(HAVE_LIBUSB_H)
                      #include <libusb.h>
                      #elif defined(HAVE_LIBUSBHID_H)
                      #include <libusbhid.h>
                      #endif
                    ]], [[
                      struct usb_ctl_report buf;
                      if (buf.ucr_data) { }
                    ]])], [have_usbhid_ucr_data=yes],[])
                    if test x$have_usbhid_ucr_data = xyes; then
                        USB_CFLAGS="$USB_CFLAGS -DUSBHID_UCR_DATA"
                    fi
                    AC_MSG_RESULT($have_usbhid_ucr_data)

                    AC_MSG_CHECKING(for new usbhid API)
                    have_usbhid_new=no
                    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                      #include <sys/types.h>
                      #if defined(HAVE_USB_H)
                      #include <usb.h>
                      #endif
                      #ifdef __DragonFly__
                      #include <bus/u4b/usb.h>
                      #include <bus/u4b/usbhid.h>
                      #else
                      #include <dev/usb/usb.h>
                      #include <dev/usb/usbhid.h>
                      #endif
                      #if defined(HAVE_USBHID_H)
                      #include <usbhid.h>
                      #elif defined(HAVE_LIBUSB_H)
                      #include <libusb.h>
                      #elif defined(HAVE_LIBUSBHID_H)
                      #include <libusbhid.h>
                      #endif
                    ]], [[
                      report_desc_t d;
                      hid_start_parse(d, 1, 1);
                    ]])], [have_usbhid_new=yes],[])
                    if test x$have_usbhid_new = xyes; then
                        USB_CFLAGS="$USB_CFLAGS -DUSBHID_NEW"
                    fi
                    AC_MSG_RESULT($have_usbhid_new)

                    AC_MSG_CHECKING(for struct joystick in machine/joystick.h)
                    have_machine_joystick=no
                    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                      #include <machine/joystick.h>
                    ]], [[
                      struct joystick t;
                    ]])], [have_machine_joystick=yes],[])
                    if test x$have_machine_joystick = xyes; then
                        AC_DEFINE(SDL_HAVE_MACHINE_JOYSTICK_H, 1, [ ])
                    fi
                    AC_MSG_RESULT($have_machine_joystick)

                    AC_DEFINE(SDL_JOYSTICK_USBHID, 1, [ ])
                    SOURCES="$SOURCES $srcdir/src/joystick/bsd/*.c"
                    EXTRA_CFLAGS="$EXTRA_CFLAGS $USB_CFLAGS"
                    EXTRA_LDFLAGS="$EXTRA_LDFLAGS $USB_LIBS"
                    have_joystick=yes
                fi

                CFLAGS="$save_CFLAGS"
            fi
        ;;
    esac
}

dnl Check for HIDAPI joystick drivers
CheckHIDAPI()
{
    AC_ARG_ENABLE(hidapi-joystick,
[AS_HELP_STRING([--enable-hidapi-joystick], [use HIDAPI for low level joystick drivers [default=yes]])],
                  , enable_hidapi_joystick=yes)
    AC_ARG_ENABLE(hidapi-libusb,
[AS_HELP_STRING([--enable-hidapi-libusb], [use libusb for low level joystick drivers [default=maybe]])],
                  , enable_hidapi_libusb=maybe)

    if test x$enable_hidapi = xyes; then
        case "$host" in
            # libusb does not support iOS
            *-ios-* )
                enable_hidapi_libusb=no
                ;;
            # On the other hand, *BSD specifically uses libusb only
            *-*-*bsd* )
                enable_hidapi_libusb=yes
                require_hidapi_libusb=yes
                ;;
            *-*-os2* )
                enable_hidapi_libusb=yes
                ;;
        esac

        hidapi_support=yes
        if test x$enable_hidapi_libusb = xyes; then
            PKG_CHECK_MODULES([LIBUSB], [libusb-1.0], have_libusb=yes, have_libusb=no)
            save_CPPFLAGS="$CPPFLAGS"
            CPPFLAGS="$save_CPPFLAGS $LIBUSB_CFLAGS"
            AC_CHECK_HEADER(libusb.h, have_libusb_h=yes, have_libusb_h=no)
            CPPFLAGS="$save_CPPFLAGS"
            if test x$have_libusb_h = xno && test x$require_hidapi_libusb = xyes; then
                hidapi_support=no
            fi
        fi

        if test x$hidapi_support = xyes; then
            if test x$have_libusb_h = xyes; then
                AC_DEFINE(HAVE_LIBUSB)
                EXTRA_CFLAGS="$EXTRA_CFLAGS $LIBUSB_CFLAGS"
                if test x$require_hidapi_libusb = xyes; then
                    EXTRA_LDFLAGS="$EXTRA_LDFLAGS $LIBUSB_LIBS"
                else
                    if test x$have_loadso != xyes; then
                        AC_MSG_WARN([You must have SDL_LoadObject() support for dynamic libusb loading])
                    fi
                    # libusb is loaded dynamically, so don't add it to LDFLAGS
                    libusb_lib=""
                    case "$host" in
                        *-*-darwin* )
                            libusb_lib="libusb-1.0.0.dylib"
                            ;;
                        *-*-cygwin* | *-*-mingw* )
                            libusb_lib="libusb-1.0.dll"
                            ;;
                        *-*-os2* )
                            libusb_lib="usb100.dll"
                            ;;
                    esac
                    if test x$libusb_lib = x; then
                        libusb_lib=[`find_lib "libusb-1.0.so.*" "" | sed 's/.*\/\(.*\)/\1/; q'`]
                    fi
                    AC_DEFINE_UNQUOTED(SDL_LIBUSB_DYNAMIC, "$libusb_lib", [ ])
                fi
            fi
        fi

        AC_MSG_CHECKING(for hidapi joystick support)
        AC_MSG_RESULT($hidapi_support)
    fi

    if test x$enable_joystick = xyes -a x$hidapi_support = xyes -a x$enable_hidapi_joystick = xyes; then
        AC_DEFINE(SDL_JOYSTICK_HIDAPI, 1, [ ])
        SOURCES="$SOURCES $srcdir/src/joystick/hidapi/*.c"
    fi
}

dnl Check for clock_gettime()
CheckClockGettime()
{
    AC_ARG_ENABLE(clock_gettime,
[AS_HELP_STRING([--enable-clock_gettime], [use clock_gettime() instead of gettimeofday() on UNIX [default=yes]])],
                  , enable_clock_gettime=yes)
    if test x$enable_clock_gettime = xyes; then
        AC_CHECK_LIB(c, clock_gettime, have_clock_gettime=yes)
        if test x$have_clock_gettime = xyes; then
           AC_DEFINE(HAVE_CLOCK_GETTIME, 1, [ ])
        else
            AC_CHECK_LIB(rt, clock_gettime, have_clock_gettime=yes)
            if test x$have_clock_gettime = xyes; then
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lrt"
                AC_DEFINE(HAVE_CLOCK_GETTIME, 1, [ ])
            fi
        fi
    fi
}

dnl Check for a valid linux/version.h
CheckLinuxVersion()
{
    AC_CHECK_HEADER(linux/version.h, have_linux_version_h=yes)
    if test x$have_linux_version_h = xyes; then
        EXTRA_CFLAGS="$EXTRA_CFLAGS -DHAVE_LINUX_VERSION_H"
    fi
}

dnl Check if we want to use RPATH
CheckRPATH()
{
    AC_ARG_ENABLE(rpath,
[AS_HELP_STRING([--enable-rpath], [use an rpath when linking SDL [default=yes]])],
                  , enable_rpath=yes)
}

dnl Check if we want to use custom signals to fake iOS/Android's backgrounding
dnl  events. These could be useful if you're building a custom embedded
dnl  environment, etc, but most people don't need this.
CheckEventSignals()
{
    AC_ARG_ENABLE(backgrounding-signal,
[AS_HELP_STRING([--enable-backgrounding-signal], [number to use for magic backgrounding signal or 'no' [default=no]])],
                  , enable_backgrounding_signal=no)
    if test x$enable_backgrounding_signal != xno; then
        EXTRA_CFLAGS="$EXTRA_CFLAGS -DSDL_BACKGROUNDING_SIGNAL=$enable_backgrounding_signal"
    fi

    AC_ARG_ENABLE(foregrounding-signal,
[AS_HELP_STRING([--enable-foregrounding-signal], [number to use for magic foregrounding signal or 'no' [default=no]])],
                  , enable_foregrounding_signal=no)
    if test x$enable_foregrounding_signal != xno; then
        EXTRA_CFLAGS="$EXTRA_CFLAGS -DSDL_FOREGROUNDING_SIGNAL=$enable_foregrounding_signal"
    fi
}

dnl Set up the Virtual joystick driver.
CheckVirtualJoystick()
{
    AC_ARG_ENABLE(joystick-virtual,
[AS_HELP_STRING([--enable-joystick-virtual], [enable virtual joystick APIs [default=yes]])],
                  , enable_joystick_virtual=yes)
    if test x$enable_joystick = xyes -a x$enable_joystick_virtual = xyes; then
        AC_DEFINE(SDL_JOYSTICK_VIRTUAL, 1, [ ])
        SOURCES="$SOURCES $srcdir/src/joystick/virtual/*.c"
        have_joystick_virtual=yes
    fi
}

dnl Do this on all platforms, before everything else (other things might want to override it).
CheckWarnAll
CheckUnusedLocalTypedefs
CheckNoStrictAliasing

dnl Do this for every platform, but for some it doesn't mean anything, but better to catch it here anyhow.
CheckEventSignals

have_locale=no

dnl Set up the configuration based on the host platform!
case "$host" in
    *-*-linux*|*-*-uclinux*|*-*-gnu*|*-*-k*bsd*-gnu|*-*-bsdi*|*-*-freebsd*|*-*-dragonfly*|*-*-netbsd*|*-*-openbsd*|*-*-sysv5*|*-*-solaris*|*-*-hpux*|*-*-aix*|*-*-minix*|*-*-nto*)
        case "$host" in
            *-*-android*)
                # Android
                ARCH=android
                ANDROID_CFLAGS="-DGL_GLEXT_PROTOTYPES"
                CFLAGS="$CFLAGS $ANDROID_CFLAGS"
                SDL_CFLAGS="$SDL_CFLAGS $ANDROID_CFLAGS"
                EXTRA_CFLAGS="$EXTRA_CFLAGS $ANDROID_CFLAGS"
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS -ldl -lGLESv1_CM -lGLESv2 -llog -landroid"
                SDLMAIN_SOURCES="$srcdir/src/main/android/*.c"
                if test x$enable_video = xyes; then
                    SOURCES="$SOURCES $srcdir/src/core/android/*.c $srcdir/src/video/android/*.c"
                    AC_DEFINE(SDL_VIDEO_DRIVER_ANDROID, 1, [ ])
                    SUMMARY_video="${SUMMARY_video} android"
                    have_video=yes
                fi
                ;;
            *-*-linux*)         ARCH=linux ;;
            *-*-uclinux*)       ARCH=linux ;;
            *-*-kfreebsd*-gnu)  ARCH=kfreebsd-gnu ;;
            *-*-knetbsd*-gnu)   ARCH=knetbsd-gnu ;;
            *-*-kopenbsd*-gnu)  ARCH=kopenbsd-gnu ;;
            *-*-gnu*)           ARCH=gnu ;; # must be last of the gnu variants
            *-*-bsdi*)          ARCH=bsdi ;;
            *-*-freebsd*)       ARCH=freebsd ;;
            *-*-dragonfly*)     ARCH=freebsd ;;
            *-*-netbsd*)        ARCH=netbsd ;;
            *-*-openbsd*)       ARCH=openbsd ;;
            *-*-sysv5*)         ARCH=sysv5 ;;
            *-*-solaris*)       ARCH=solaris ;;
            *-*-hpux*)          ARCH=hpux ;;
            *-*-aix*)           ARCH=aix ;;
            *-*-minix*)         ARCH=minix ;;
            *-*-nto*)           ARCH=nto
                CheckQNXVideo
                ;;
        esac
        CheckVisibilityHidden
        CheckWerror
        CheckDeclarationAfterStatement
        CheckDummyVideo
        CheckOffscreenVideo
        CheckDiskAudio
        CheckDummyAudio
        CheckDLOPEN
        CheckARM
        CheckNEON
        CheckO_CLOEXEC
        CheckOSS
        CheckALSA
        CheckPipewire
        CheckPulseAudio
        CheckJACK
        CheckARTSC
        CheckESD
        CheckNAS
        CheckSNDIO
        CheckFusionSound
        CheckLibSampleRate
        # Need to check for Raspberry PI first and add platform specific compiler flags, otherwise the test for GLES fails!
        CheckRPI
        CheckX11
        CheckDirectFB
        # Need to check for EGL first because KMSDRM and Wayland depends on it.
        CheckEGL
        CheckKMSDRM
        CheckGLX
        CheckOpenGL
        CheckOpenGLES
        CheckVulkan
        CheckWayland
        CheckInputEvents
        CheckLibUDev
        CheckDBus
        CheckIME
        CheckInotify
        CheckIBus
        CheckFcitx
        case $ARCH in
          linux)
              CheckInputKD
          ;;
          freebsd)
              CheckInputKBIO
          ;;
          openbsd|netbsd)
              CheckInputWSCONS
          ;;
        esac
        CheckUSBHID
        CheckHIDAPI
        CheckPTHREAD
        CheckClockGettime
        CheckLinuxVersion
        CheckRPATH
        CheckVivanteVideo

        # Set up files for the misc library
        if test x$enable_misc = xyes; then
            SOURCES="$SOURCES $srcdir/src/misc/unix/*.c"
            have_misc=yes
        fi
        # Set up files for the locale library
        if test x$enable_locale = xyes; then
            SOURCES="$SOURCES $srcdir/src/locale/unix/*.c"
            have_locale=yes
        fi
        # Set up files for the audio library
        if test x$enable_audio = xyes; then
          case $ARCH in
            sysv5|solaris|hpux)
                AC_DEFINE(SDL_AUDIO_DRIVER_SUNAUDIO, 1, [ ])
                SOURCES="$SOURCES $srcdir/src/audio/sun/*.c"
                SUMMARY_audio="${SUMMARY_audio} sun"
                have_audio=yes
            ;;
            netbsd)  # Don't use this on OpenBSD, it's busted.
                AC_DEFINE(SDL_AUDIO_DRIVER_NETBSD, 1, [ ])
                SOURCES="$SOURCES $srcdir/src/audio/netbsd/*.c"
                SUMMARY_audio="${SUMMARY_audio} netbsd"
                have_audio=yes
            ;;
            aix)
                AC_DEFINE(SDL_AUDIO_DRIVER_PAUDIO, 1, [ ])
                SOURCES="$SOURCES $srcdir/src/audio/paudio/*.c"
                SUMMARY_audio="${SUMMARY_audio} paudio"
                have_audio=yes
            ;;
            android)
                AC_DEFINE(SDL_AUDIO_DRIVER_ANDROID, 1, [ ])
                SOURCES="$SOURCES $srcdir/src/audio/android/*.c"
                SUMMARY_audio="${SUMMARY_audio} android"

                AC_DEFINE(SDL_AUDIO_DRIVER_OPENSLES, 1, [ ])
                SOURCES="$SOURCES $srcdir/src/audio/openslES/*.c"
                SUMMARY_audio="${SUMMARY_audio} openslES"

                AC_DEFINE(SDL_AUDIO_DRIVER_AAUDIO, 1, [ ])
                SOURCES="$SOURCES $srcdir/src/audio/aaudio/*.c"
                SUMMARY_audio="${SUMMARY_audio} aaudio"

                have_audio=yes
            ;;
            nto)
                CheckQNXAudio
            ;;
          esac
        fi
        # Set up files for the joystick library
        if test x$enable_joystick = xyes; then
          case $ARCH in
            linux)
                if test "x$ac_cv_header_linux_input_h" = xyes; then
                    AC_DEFINE(SDL_JOYSTICK_LINUX, 1, [ ])
                    SOURCES="$SOURCES $srcdir/src/joystick/linux/*.c"
                    SOURCES="$SOURCES $srcdir/src/joystick/steam/*.c"
                    have_joystick=yes
                fi
            ;;
            freebsd)
                if test x$use_input_events = xyes -a x$ac_cv_header_linux_input_h = xyes; then
                    AC_DEFINE(SDL_JOYSTICK_LINUX, 1, [ ])
                    SOURCES="$SOURCES $srcdir/src/joystick/linux/*.c"
                    SOURCES="$SOURCES $srcdir/src/joystick/steam/*.c"
                    have_joystick=yes
                fi
            ;;
            android)
                AC_DEFINE(SDL_JOYSTICK_ANDROID, 1, [ ])
                SOURCES="$SOURCES $srcdir/src/joystick/android/*.c"
                SOURCES="$SOURCES $srcdir/src/joystick/steam/*.c"
                have_joystick=yes
            ;;
          esac
        fi
        # Set up files for the haptic library
        if test x$enable_haptic = xyes; then
          case $ARCH in
            linux|freebsd)
                if test x$use_input_events = xyes; then
                    AC_DEFINE(SDL_HAPTIC_LINUX, 1, [ ])
                    SOURCES="$SOURCES $srcdir/src/haptic/linux/*.c"
                    have_haptic=yes
                fi
            ;;
            android)
                AC_DEFINE(SDL_HAPTIC_ANDROID, 1, [ ])
                SOURCES="$SOURCES $srcdir/src/haptic/android/*.c"
                have_haptic=yes
            ;;
          esac
        fi
        # Set up files for the sensor library
        if test x$enable_sensor = xyes; then
          case $ARCH in
            android)
                AC_DEFINE(SDL_SENSOR_ANDROID, 1, [ ])
                SOURCES="$SOURCES $srcdir/src/sensor/android/*.c"
                have_sensor=yes
            ;;
          esac
        fi
        # Set up files for the power library
        if test x$enable_power = xyes; then
             case $ARCH in
               linux)
                   AC_DEFINE(SDL_POWER_LINUX, 1, [ ])
                   SOURCES="$SOURCES $srcdir/src/power/linux/*.c"
                   have_power=yes
               ;;
               android)
                   AC_DEFINE(SDL_POWER_ANDROID, 1, [ ])
                   SOURCES="$SOURCES $srcdir/src/power/android/*.c"
                   have_power=yes
               ;;
             esac
        fi
        # Set up files for the filesystem library
        if test x$enable_filesystem = xyes; then
             case $ARCH in
               android)
                   AC_DEFINE(SDL_FILESYSTEM_ANDROID, 1, [ ])
                   SOURCES="$SOURCES $srcdir/src/filesystem/android/*.c"
                   have_filesystem=yes
               ;;
               *)
                   AC_DEFINE(SDL_FILESYSTEM_UNIX, 1, [ ])
                   SOURCES="$SOURCES $srcdir/src/filesystem/unix/*.c"
                   have_filesystem=yes
               ;;
             esac
        fi
        # Set up files for the timer library
        if test x$enable_timers = xyes; then
            AC_DEFINE(SDL_TIMER_UNIX, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/timer/unix/*.c"
            have_timers=yes
        fi
        # Set up files for udev hotplugging support
        if test x$enable_libudev = xyes && test x$have_libudev_h_hdr = xyes; then
            SOURCES="$SOURCES $srcdir/src/core/linux/SDL_udev.c"
        fi
        # Set up files for evdev input
        if test x$use_input_events = xyes; then
            SOURCES="$SOURCES $srcdir/src/core/linux/SDL_evdev.c"
            SOURCES="$SOURCES $srcdir/src/core/linux/SDL_evdev_kbd.c"
            SOURCES="$SOURCES $srcdir/src/core/freebsd/SDL_evdev_kbd_freebsd.c"
        fi
        # Set up files for wscons input
        if test x$use_input_wscons = xyes; then
            SOURCES="$SOURCES $srcdir/src/core/openbsd/SDL_wscons_kbd.c"
            SOURCES="$SOURCES $srcdir/src/core/openbsd/SDL_wscons_mouse.c"
        fi
        # Set up other core UNIX files
        SOURCES="$SOURCES $srcdir/src/core/linux/SDL_evdev_capabilities.c"
        SOURCES="$SOURCES $srcdir/src/core/linux/SDL_threadprio.c"
        SOURCES="$SOURCES $srcdir/src/core/linux/SDL_sandbox.c"
        SOURCES="$SOURCES $srcdir/src/core/unix/*.c"
        ;;
    *-*-cygwin* | *-*-mingw*)
        ARCH=win32
        if test "$build" != "$host"; then # cross-compiling
            # Default cross-compile location
            ac_default_prefix=/usr/local/cross-tools/$host
        else
            # Look for the location of the tools and install there
            if test "$BUILD_PREFIX" != ""; then
                ac_default_prefix=$BUILD_PREFIX
            fi
        fi
        if test x$enable_loadso = xyes; then
            have_loadso=yes
        fi
        CheckGDwarf4
        CheckWerror
        CheckDeclarationAfterStatement
        CheckDummyVideo
        CheckOffscreenVideo
        CheckDiskAudio
        CheckDummyAudio
        CheckWINDOWS
        CheckWINDOWSGL
        CheckWINDOWSGLES
        CheckVulkan
        CheckDIRECTX
        CheckHIDAPI

        # Set up the core platform files
        SOURCES="$SOURCES $srcdir/src/core/windows/*.c"
        # Set up files for the misc library
        if test x$enable_misc = xyes; then
            SOURCES="$SOURCES $srcdir/src/misc/windows/*.c"
            have_misc=yes
        fi
        # Use the Windows locale APIs.
        if test x$enable_locale = xyes; then
            SOURCES="$SOURCES $srcdir/src/locale/windows/*.c"
            have_locale=yes
        fi
        # Set up files for the video library
        if test x$enable_video = xyes; then
            AC_DEFINE(SDL_VIDEO_DRIVER_WINDOWS, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/video/windows/*.c"
            have_video=yes
            AC_ARG_ENABLE(render-d3d,
[AS_HELP_STRING([--enable-render-d3d], [enable the Direct3D render driver [default=yes]])],
                                , enable_render_d3d=yes)
            if test x$enable_render_d3d = xyes -a x$have_d3d = xyes; then
                AC_DEFINE(SDL_VIDEO_RENDER_D3D, 1, [ ])
                SUMMARY_video="${SUMMARY_video} d3d9"
            fi
            if test x$enable_render_d3d = xyes -a x$have_d3d11 = xyes; then
                AC_DEFINE(SDL_VIDEO_RENDER_D3D11, 1, [ ])
                SUMMARY_video="${SUMMARY_video} d3d11"
            fi
            if test x$enable_render_d3d = xyes -a x$have_d3d12 = xyes; then
                AC_DEFINE(SDL_VIDEO_RENDER_D3D12, 1, [ ])
                SUMMARY_video="${SUMMARY_video} d3d12"
            fi
        fi
        # Set up files for the audio library
        if test x$enable_audio = xyes; then
            AC_DEFINE(SDL_AUDIO_DRIVER_WINMM, 1, [ ])
            SUMMARY_audio="${SUMMARY_audio} winmm"
            SOURCES="$SOURCES $srcdir/src/audio/winmm/*.c"
            if test x$have_dsound = xyes; then
                AC_DEFINE(SDL_AUDIO_DRIVER_DSOUND, 1, [ ])
                SUMMARY_audio="${SUMMARY_audio} directsound"
                SOURCES="$SOURCES $srcdir/src/audio/directsound/*.c"
            fi
            if test x$have_wasapi = xyes -a x$enable_wasapi = xyes; then
                AC_DEFINE(SDL_AUDIO_DRIVER_WASAPI, 1, [ ])
                SUMMARY_audio="${SUMMARY_audio} wasapi"
                SOURCES="$SOURCES $srcdir/src/audio/wasapi/*.c"
            fi
            have_audio=yes
        fi
        # Set up files for the joystick library
        if test x$enable_joystick = xyes; then
            AC_DEFINE(SDL_JOYSTICK_RAWINPUT, 1, [ ])
            if test x$have_dinput = xyes -o x$have_xinput = xyes -o x$have_wgi = xyes; then
                if test x$have_xinput = xyes; then
                    AC_DEFINE(SDL_JOYSTICK_XINPUT, 1, [ ])
                fi
                if test x$have_wgi = xyes; then
                    AC_DEFINE(SDL_JOYSTICK_WGI, 1, [ ])
                fi
                if test x$have_dinput = xyes; then
                    AC_DEFINE(SDL_JOYSTICK_DINPUT, 1, [ ])
                    EXTRA_LDFLAGS="$EXTRA_LDFLAGS -ldinput8 -ldxguid -ldxerr8"
                fi
            fi
            SOURCES="$SOURCES $srcdir/src/joystick/windows/*.c"
            have_joystick=yes
        fi
        if test x$enable_haptic = xyes; then
            if test x$have_dinput = xyes -o x$have_xinput = xyes; then
                if test x$have_xinput = xyes; then
                    AC_DEFINE(SDL_HAPTIC_XINPUT, 1, [ ])
                fi
                if test x$have_dinput = xyes; then
                    AC_DEFINE(SDL_HAPTIC_DINPUT, 1, [ ])
                fi
                SOURCES="$SOURCES $srcdir/src/haptic/windows/*.c"
                have_haptic=yes
            fi
        fi
        # Set up files for the sensor library
        AC_CHECK_HEADER(sensorsapi.h,have_winsensors=yes,have_winsensors=no)
        if test x$have_winsensors = xyes; then
            AC_DEFINE(HAVE_SENSORSAPI_H, 1, [ ])
        fi
        if test x$enable_sensor = xyes -a x$have_winsensors = xyes; then
            AC_DEFINE(SDL_SENSOR_WINDOWS, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/sensor/windows/*.c"
            have_sensor=yes
        fi
        # Set up files for the power library
        if test x$enable_power = xyes; then
            AC_DEFINE(SDL_POWER_WINDOWS, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/power/windows/SDL_syspower.c"
            have_power=yes
        fi
        # Set up files for the filesystem library
        if test x$enable_filesystem = xyes; then
            AC_DEFINE(SDL_FILESYSTEM_WINDOWS, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/filesystem/windows/SDL_sysfilesystem.c"
            have_filesystem=yes
        fi
        # Set up files for the thread library
        if test x$enable_threads = xyes; then
            AC_DEFINE(SDL_THREAD_GENERIC_COND_SUFFIX, 1, [ ])
            AC_DEFINE(SDL_THREAD_WINDOWS, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/thread/windows/*.c"
            SOURCES="$SOURCES $srcdir/src/thread/generic/SDL_syscond.c"
            have_threads=yes
        fi
        # Set up files for the timer library
        if test x$enable_timers = xyes; then
            AC_DEFINE(SDL_TIMER_WINDOWS, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/timer/windows/*.c"
            have_timers=yes
        fi
        # Set up files for the shared object loading library
        if test x$enable_loadso = xyes; then
            AC_DEFINE(SDL_LOADSO_WINDOWS, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/loadso/windows/*.c"
        fi
        # Set up the system libraries we need
        if test -f /lib/w32api/libuuid.a; then
            LIBUUID=/lib/w32api/libuuid.a
        else
            LIBUUID=-luuid
        fi
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -luser32 -lgdi32 -lwinmm -limm32 -lole32 -loleaut32 -lshell32 -lsetupapi -lversion $LIBUUID"
        BUILD_LDFLAGS="$BUILD_LDFLAGS -Wc,-static-libgcc"
        # The Windows platform requires special setup
        VERSION_SOURCES="$srcdir/src/main/windows/*.rc"
        SDLMAIN_SOURCES="$srcdir/src/main/windows/*.c"
        SDL_CFLAGS="$SDL_CFLAGS -Dmain=SDL_main"
        SDL_LIBS="-lSDL2main $SDL_LIBS -mwindows"

        # Check to see if this is a mingw or cygwin build
        have_mingw32=
        AC_CHECK_LIB(mingw32, main, [have_mingw32=yes])
        if test x$have_mingw32 = xyes; then
            SDL_LIBS="-lmingw32 $SDL_LIBS"
        else
            SDL_LIBS="-lcygwin $SDL_LIBS"
        fi
        ;;
dnl BeOS support removed after SDL 2.0.1. Haiku still works.  --ryan.
    *-*-beos*)
        AC_MSG_ERROR([
*** BeOS support has been removed as of SDL 2.0.2.
        ])
        ;;
    *-*-haiku*)
        ARCH=haiku
        ac_default_prefix=/boot/system
        CheckDummyVideo
        CheckOffscreenVideo
        CheckDiskAudio
        CheckDummyAudio
        CheckDLOPEN
        CheckO_CLOEXEC
        CheckHaikuVideo
        CheckHaikuGL
        CheckPTHREAD

        # Set up files for the audio library
        if test x$enable_audio = xyes; then
            AC_DEFINE(SDL_AUDIO_DRIVER_HAIKU, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/audio/haiku/*.cc"
            SUMMARY_audio="${SUMMARY_audio} haiku"
            have_audio=yes
        fi
        # Set up files for the joystick library
        if test x$enable_joystick = xyes; then
            AC_DEFINE(SDL_JOYSTICK_HAIKU, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/joystick/haiku/*.cc"
            have_joystick=yes
        fi
        # Set up files for the timer library
        if test x$enable_timers = xyes; then
            AC_DEFINE(SDL_TIMER_HAIKU, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/timer/haiku/*.c"
            have_timers=yes
        fi
        # Set up files for the system power library
        if test x$enable_power = xyes; then
            AC_DEFINE(SDL_POWER_HAIKU, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/power/haiku/*.c"
            have_power=yes
        fi
        # Set up files for the system filesystem library
        if test x$enable_filesystem = xyes; then
            AC_DEFINE(SDL_FILESYSTEM_HAIKU, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/filesystem/haiku/*.cc"
            have_filesystem=yes
        fi
        # Set up files for the misc library
        if test x$enable_misc = xyes; then
            SOURCES="$SOURCES $srcdir/src/misc/haiku/*.cc"
            have_misc=yes
        fi
        # Set up files for the locale library
        if test x$enable_locale = xyes; then
            SOURCES="$SOURCES $srcdir/src/locale/haiku/*.cc"
            have_locale=yes
        fi
        # The Haiku platform requires special setup.
        SOURCES="$srcdir/src/main/haiku/*.cc $SOURCES"
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lroot -lbe -lmedia -lgame -ldevice -ltextencoding"
        ;;
    *-ios-*)
        ARCH=ios

        CheckVisibilityHidden
        CheckWerror
        CheckNoErrorDeprecatedDeclarationsWerror
        CheckDeclarationAfterStatement
        CheckDummyVideo
        CheckOffscreenVideo
        CheckDiskAudio
        CheckDummyAudio
        CheckDLOPEN
        CheckO_CLOEXEC
        CheckMETAL
        CheckVulkan
        CheckPTHREAD

        # Set up files for the misc library
        if test x$enable_misc = xyes; then
            SOURCES="$SOURCES $srcdir/src/misc/ios/*.m"
            have_misc=yes
        fi
        # Set up files for the locale library
        if test x$enable_locale = xyes; then
            SOURCES="$SOURCES $srcdir/src/locale/macosx/*.m"
            have_locale=yes
        fi
        # Set up files for the audio library
        if test x$enable_audio = xyes; then
            AC_DEFINE(SDL_AUDIO_DRIVER_COREAUDIO, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/audio/coreaudio/*.m"
            SUMMARY_audio="${SUMMARY_audio} coreaudio"
            have_audio=yes
        fi
        # Set up files for the joystick library
        if test x$enable_joystick = xyes; then
            AC_DEFINE(SDL_JOYSTICK_MFI, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/joystick/iphoneos/*.m"
            SOURCES="$SOURCES $srcdir/src/joystick/steam/*.c"
            have_joystick=yes
        else
            # Need this code for accelerometer as joystick support
            SOURCES="$SOURCES $srcdir/src/joystick/iphoneos/*.m"
        fi
        # Set up files for the haptic library
        #if test x$enable_haptic = xyes; then
        #    SOURCES="$SOURCES $srcdir/src/haptic/darwin/*.c"
        #    have_haptic=yes
        #    EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,ForceFeedback"
        #fi
        # Set up files for the sensor library
        if test x$enable_sensor = xyes; then
            AC_DEFINE(SDL_SENSOR_COREMOTION, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/sensor/coremotion/*.m"
            have_sensor=yes
        fi
        # Set up files for the power library
        if test x$enable_power = xyes; then
            AC_DEFINE(SDL_POWER_UIKIT, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/power/uikit/*.m"
            have_power=yes
        fi
        # Set up files for the filesystem library
        if test x$enable_filesystem = xyes; then
            SOURCES="$SOURCES $srcdir/src/filesystem/cocoa/*.m"
            have_filesystem=yes
        fi
        # Set up additional files for the file library
        if test x$enable_file = xyes; then
            AC_DEFINE(SDL_FILESYSTEM_COCOA, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/file/cocoa/*.m"
        fi
        # Set up files for the timer library
        if test x$enable_timers = xyes; then
            AC_DEFINE(SDL_TIMER_UNIX, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/timer/unix/*.c"
            have_timers=yes
        fi
        # Set up other core UNIX files
        SOURCES="$SOURCES $srcdir/src/core/unix/*.c"
        # The iOS platform requires special setup.
        AC_DEFINE(SDL_VIDEO_DRIVER_UIKIT, 1, [ ])
        AC_DEFINE(SDL_VIDEO_OPENGL_ES2, 1, [ ])
        AC_DEFINE(SDL_VIDEO_OPENGL_ES, 1, [ ])
        AC_DEFINE(SDL_VIDEO_RENDER_OGL_ES, 1, [ ])
        AC_DEFINE(SDL_VIDEO_RENDER_OGL_ES2, 1, [ ])
        SOURCES="$SOURCES $srcdir/src/video/uikit/*.m"
        SUMMARY_video="${SUMMARY_video} uikit"
        have_video=yes
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lm"
        if test x$enable_system_iconv = xyes; then
            EXTRA_LDFLAGS="$EXTRA_LDFLAGS -liconv"
        fi
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lobjc"
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,AVFoundation"
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,AudioToolbox"
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,CoreAudio"
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,CoreGraphics"
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,CoreMotion"
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,Foundation"
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,GameController"
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,OpenGLES"
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,QuartzCore"
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,UIKit"

        if test x$enable_video_metal = xyes -o x$enable_video_vulkan = xyes; then
            EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,Metal"
        fi
        ;;
    *-*-darwin* )
        # This could be either full "Mac OS X", or plain "Darwin" which is
        # just the OS X kernel sans upper layers like Carbon and Cocoa.
        # Next line is broken, and a few files below require Mac OS X (full)
        ARCH=macosx

        # Mac OS X builds with both the Carbon and OSX APIs at the moment
        EXTRA_CFLAGS="$EXTRA_CFLAGS -DTARGET_API_MAC_CARBON"
        EXTRA_CFLAGS="$EXTRA_CFLAGS -DTARGET_API_MAC_OSX"

        CheckObjectiveCARC
        CheckVisibilityHidden
        CheckWerror
        CheckNoErrorDeprecatedDeclarationsWerror
        CheckDeclarationAfterStatement
        CheckDummyVideo
        CheckOffscreenVideo
        CheckDiskAudio
        CheckDummyAudio
        CheckDLOPEN
        CheckO_CLOEXEC
        CheckCOCOA
        CheckMETAL
        CheckX11
        CheckMacGL
        CheckMacGLES
        CheckGLX
        CheckOpenGL
        CheckVulkan
        CheckPTHREAD
        CheckHIDAPI

        # Set up files for the misc library
        if test x$enable_misc = xyes; then
            SOURCES="$SOURCES $srcdir/src/misc/macosx/*.m"
            have_misc=yes
        fi
        # Set up files for the locale library
        if test x$enable_locale = xyes; then
            SOURCES="$SOURCES $srcdir/src/locale/macosx/*.m"
            have_locale=yes
        fi
        # Set up files for the audio library
        if test x$enable_audio = xyes; then
            AC_DEFINE(SDL_AUDIO_DRIVER_COREAUDIO, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/audio/coreaudio/*.m"
            EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,CoreAudio -Wl,-framework,AudioToolbox"
            SUMMARY_audio="${SUMMARY_audio} coreaudio"
            have_audio=yes
        fi
        # Set up files for the joystick library
        if test x$enable_joystick = xyes; then
            AC_DEFINE(SDL_JOYSTICK_IOKIT, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/joystick/darwin/*.c"
            SOURCES="$SOURCES $srcdir/src/joystick/iphoneos/*.m"
            CheckJoystickMFI
            have_joystick=yes
        fi
        # Set up files for the haptic library
        if test x$enable_haptic = xyes; then
            AC_DEFINE(SDL_HAPTIC_IOKIT, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/haptic/darwin/*.c"
            EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,ForceFeedback"
            have_haptic=yes
        fi
        # Set up files for the power library
        if test x$enable_power = xyes; then
            AC_DEFINE(SDL_POWER_MACOSX, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/power/macosx/*.c"
            have_power=yes
        fi
        # Set up files for the filesystem library
        if test x$enable_filesystem = xyes; then
            AC_DEFINE(SDL_FILESYSTEM_COCOA, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/filesystem/cocoa/*.m"
            have_filesystem=yes
        fi
        # Set up files for the timer library
        if test x$enable_timers = xyes; then
            AC_DEFINE(SDL_TIMER_UNIX, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/timer/unix/*.c"
            have_timers=yes
        fi
        # Set up additional files for the file library
        if test x$enable_file = xyes; then
            SOURCES="$SOURCES $srcdir/src/file/cocoa/*.m"
        fi
        # Set up other core UNIX files
        SOURCES="$SOURCES $srcdir/src/core/unix/*.c"
        # The Mac OS X platform requires special setup.
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lobjc"
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,CoreVideo"
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,Cocoa"
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,Carbon"
        EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-framework,IOKit"

        if test x$enable_video_metal = xyes -o x$enable_video_vulkan = xyes; then
            EXTRA_LDFLAGS="$EXTRA_LDFLAGS -Wl,-weak_framework,QuartzCore -Wl,-weak_framework,Metal"
        fi
        ;;
    *-nacl|*-pnacl)
        ARCH=nacl
        CheckNativeClient
        CheckDummyAudio
        CheckDummyVideo
        CheckOffscreenVideo
        CheckInputEvents
        CheckPTHREAD
        CheckO_CLOEXEC

        # Set up files for the timer library
        if test x$enable_timers = xyes; then
            AC_DEFINE(SDL_TIMER_UNIX, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/timer/unix/*.c"
            have_timers=yes
        fi

        if test x$enable_filesystem = xyes; then
            AC_DEFINE(SDL_FILESYSTEM_NACL, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/filesystem/nacl/*.c"
            have_filesystem=yes
        fi
        ;;
    *-*-emscripten* )
        if test x$enable_video = xyes; then
            AC_DEFINE(SDL_VIDEO_DRIVER_EMSCRIPTEN, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/video/emscripten/*.c"
            have_video=yes
            SUMMARY_video="${SUMMARY_video} emscripten"
        fi

        if test x$enable_audio = xyes; then
            AC_DEFINE(SDL_AUDIO_DRIVER_EMSCRIPTEN, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/audio/emscripten/*.c"
            have_audio=yes
            SUMMARY_audio="${SUMMARY_audio} emscripten"
        fi

        CheckVisibilityHidden
        CheckWerror
        CheckDeclarationAfterStatement
        CheckDummyVideo
        CheckOffscreenVideo
        CheckDiskAudio
        CheckDummyAudio
        CheckPTHREAD
        CheckDLOPEN
        CheckClockGettime
        CheckEmscriptenGLES

        # Set up files for the misc library
        if test x$enable_misc = xyes; then
            SOURCES="$SOURCES $srcdir/src/misc/emscripten/*.c"
            have_misc=yes
        fi

         # Set up files for the power library
        if test x$enable_power = xyes; then
            AC_DEFINE(SDL_POWER_EMSCRIPTEN, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/power/emscripten/*.c"
            have_power=yes
        fi

        # Set up files for the joystick library
        if test x$enable_joystick = xyes; then
            AC_DEFINE(SDL_JOYSTICK_EMSCRIPTEN, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/joystick/emscripten/*.c"
            have_joystick=yes
        fi

        # Set up files for the filesystem library
        if test x$enable_filesystem = xyes; then
            AC_DEFINE(SDL_FILESYSTEM_EMSCRIPTEN, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/filesystem/emscripten/*.c"
            have_filesystem=yes
        fi
        # Set up files for the timer library
        if test x$enable_timers = xyes; then
            AC_DEFINE(SDL_TIMER_UNIX, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/timer/unix/*.c"
            have_timers=yes
        fi
        # Set up files for the locale library
        if test x$enable_locale = xyes; then
            SOURCES="$SOURCES $srcdir/src/locale/emscripten/*.c"
            have_locale=yes
        fi
        ;;
    *-*-riscos*)
        ARCH=riscos
        CheckVisibilityHidden
        CheckWerror
        CheckDeclarationAfterStatement
        CheckDummyVideo
        CheckOffscreenVideo
        CheckDiskAudio
        CheckDummyAudio
        CheckDLOPEN
        CheckO_CLOEXEC
        CheckOSS
        CheckPTHREAD
        CheckClockGettime

        # Set up files for the misc library
        if test x$enable_misc = xyes; then
            SOURCES="$SOURCES $srcdir/src/misc/riscos/*.c"
            have_misc=yes
        fi
        # Set up files for the video library
        if test x$enable_video = xyes; then
            AC_DEFINE(SDL_VIDEO_DRIVER_RISCOS, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/video/riscos/*.c"
            have_video=yes
            SUMMARY_video="${SUMMARY_video} riscos"
        fi
        # Set up files for the filesystem library
        if test x$enable_filesystem = xyes; then
            AC_DEFINE(SDL_FILESYSTEM_RISCOS, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/filesystem/riscos/*.c"
            have_filesystem=yes
        fi
        # Set up files for the timer library
        if test x$enable_timers = xyes; then
            AC_DEFINE(SDL_TIMER_UNIX, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/timer/unix/*.c"
            have_timers=yes
        fi
        ;;
    *-*-os2*)
        ARCH=os2
        if test "$build" != "$host"; then # cross-compiling
            # Default cross-compile location
            ac_default_prefix=/@unixroot/usr/local/cross-tools/$host
        else
            # Look for the location of the tools and install there
            if test "$BUILD_PREFIX" != ""; then
                ac_default_prefix=$BUILD_PREFIX
            fi
        fi
        enable_static=no # disable static builds
        EXTRA_CFLAGS="$EXTRA_CFLAGS -DBUILD_SDL -DOS2EMX_PLAIN_CHAR"
        CheckOS2
        CheckWerror
        CheckDeclarationAfterStatement
        CheckDummyVideo
        CheckDiskAudio
        CheckDummyAudio
        CheckHIDAPI

        # Set up the core platform files
        SOURCES="$SOURCES $srcdir/src/core/os2/*.c"
        if test x$enable_system_iconv = xyes; then
            if test x$ac_cv_func_iconv != xyes -o x$ac_cv_header_iconv_h != xyes; then
                SOURCES="$SOURCES $srcdir/src/core/os2/geniconv/*.c"
            fi
        fi
        # Use the Unix locale APIs.
        if test x$enable_locale = xyes; then
            SOURCES="$SOURCES $srcdir/src/locale/unix/*.c"
            have_locale=yes
        fi
        # Set up files for the video library
        if test x$enable_video = xyes; then
            AC_DEFINE(SDL_VIDEO_DRIVER_OS2, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/video/os2/*.c"
            have_video=yes
            SUMMARY_video="${SUMMARY_video} OS/2"
        fi
        # Set up files for the audio library
        if test x$enable_audio = xyes; then
            AC_DEFINE(SDL_AUDIO_DRIVER_OS2, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/audio/os2/*.c"
            have_audio=yes
            SUMMARY_audio="${SUMMARY_audio} OS/2"
            EXTRA_LDFLAGS="$EXTRA_LDFLAGS -lmmpm2"
        fi
        # Set up files for the thread library
        if test x$enable_threads = xyes; then
            AC_DEFINE(SDL_THREAD_OS2, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/thread/os2/*.c"
            SOURCES="$SOURCES $srcdir/src/thread/generic/SDL_syscond.c"
            have_threads=yes
        fi
        # Set up files for the timer library
        if test x$enable_timers = xyes; then
            AC_DEFINE(SDL_TIMER_OS2, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/timer/os2/*.c"
            have_timers=yes
        fi
        # Set up files for the shared object loading library
        if test x$enable_loadso = xyes; then
            AC_DEFINE(SDL_LOADSO_OS2, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/loadso/os2/*.c"
            have_loadso=yes
        fi
        # Set up files for the filesystem library
        if test x$enable_filesystem = xyes; then
            AC_DEFINE(SDL_FILESYSTEM_OS2, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/filesystem/os2/*.c"
            have_filesystem=yes
        fi
        # Set up files for the joystick library
        if test x$enable_joystick = xyes; then
            AC_DEFINE(SDL_JOYSTICK_OS2, 1, [ ])
            SOURCES="$SOURCES $srcdir/src/joystick/os2/*.c"
            have_joystick=yes
        fi
        ;;
    *)
        AC_MSG_ERROR([
*** Unsupported host:  Please add to configure.ac
        ])
        ;;
esac

dnl Permit use of virtual joystick APIs on any platform (subject to configure options)
CheckVirtualJoystick

# Check whether to install sdl2-config
AC_MSG_CHECKING(whether to install sdl2-config)
AC_ARG_ENABLE([sdl2-config],
    [AS_HELP_STRING([--enable-sdl2-config],[Install sdl2-config [default=yes]])],
    [case "${enableval}" in
        yes) enable_sdl2_config="TRUE" ;;
        no) enable_sdl2_config="FALSE" ;;
        *) AC_MSG_ERROR([bad value '${enableval}' for --enable-sdl2-config]) ;;
    esac], [enable_sdl2_config="TRUE"])
if test "$enable_sdl2_config" = "TRUE"; then
    AC_MSG_RESULT(yes)
else
    AC_MSG_RESULT(no)
fi
AC_SUBST([INSTALL_SDL2_CONFIG], [$enable_sdl2_config])

AC_ARG_ENABLE([vendor-info],
    [AS_HELP_STRING([--enable-vendor-info=STRING], [Add vendor info to SDL_REVISION])],
    [enable_vendor_info="$enableval"], [enable_vendor_info=])
AS_IF([test "$enable_vendor_info" = no], [enable_vendor_info=])
AC_SUBST([SDL_VENDOR_INFO], [$enable_vendor_info])

# Verify that we have all the platform specific files we need

if test x$have_audio != xyes; then
    if test x$enable_audio = xyes; then
        AC_DEFINE(SDL_AUDIO_DRIVER_DUMMY, 1, [ ])
    fi
    SOURCES="$SOURCES $srcdir/src/audio/dummy/*.c"
fi
if test x$have_video != xyes; then
    if test x$enable_video = xyes; then
        AC_DEFINE(SDL_VIDEO_DRIVER_DUMMY, 1, [ ])
    fi
    SOURCES="$SOURCES $srcdir/src/video/dummy/*.c"
fi
if test x$have_misc != xyes; then
    if test x$enable_misc = xyes; then
        AC_DEFINE(SDL_MISC_DUMMY, 1, [ ])
    fi
    SOURCES="$SOURCES $srcdir/src/misc/dummy/*.c"
fi
if test x$have_locale != xyes; then
    if test x$enable_locale = xyes; then
        AC_DEFINE(SDL_LOCALE_DUMMY, 1, [ ])
    fi
    SOURCES="$SOURCES $srcdir/src/locale/dummy/*.c"
fi
if test x$have_joystick != xyes; then
    if test x$enable_joystick = xyes; then
        AC_DEFINE(SDL_JOYSTICK_DUMMY, 1, [ ])
    fi
    SOURCES="$SOURCES $srcdir/src/joystick/dummy/*.c"
fi
if test x$have_haptic != xyes; then
    if test x$enable_haptic = xyes; then
        AC_DEFINE(SDL_HAPTIC_DUMMY, 1, [ ])
    fi
    SOURCES="$SOURCES $srcdir/src/haptic/dummy/*.c"
fi
if test x$have_sensor != xyes; then
    if test x$enable_sensor = xyes; then
        AC_DEFINE(SDL_SENSOR_DUMMY, 1, [ ])
    fi
    SOURCES="$SOURCES $srcdir/src/sensor/dummy/*.c"
fi
if test x$have_threads != xyes; then
    if test x$enable_threads = xyes; then
        AC_DEFINE(SDL_THREADS_DISABLED, 1, [ ])
    fi
    SOURCES="$SOURCES $srcdir/src/thread/generic/*.c"
fi
if test x$have_timers != xyes; then
    if test x$enable_timers = xyes; then
        AC_DEFINE(SDL_TIMER_DUMMY, 1, [ ])
    fi
    SOURCES="$SOURCES $srcdir/src/timer/dummy/*.c"
fi
if test x$have_filesystem != xyes; then
    if test x$enable_filesystem = xyes; then
        AC_DEFINE(SDL_FILESYSTEM_DUMMY, 1, [ ])
    fi
    SOURCES="$SOURCES $srcdir/src/filesystem/dummy/*.c"
fi
if test x$have_loadso != xyes; then
    if test x$enable_loadso = xyes; then
        AC_DEFINE(SDL_LOADSO_DUMMY, 1, [ ])
    fi
    SOURCES="$SOURCES $srcdir/src/loadso/dummy/*.c"
fi
if test x$SDLMAIN_SOURCES = x; then
    SDLMAIN_SOURCES="$srcdir/src/main/dummy/*.c"
fi
SDLTEST_SOURCES="$srcdir/src/test/*.c"

if test x$video_wayland = xyes; then
    WAYLAND_PROTOCOLS=`cd $srcdir/wayland-protocols ; for p in *.xml ; do printf '%s' "\$p" |sed 's,\\.xml\$, ,g' ; done`
    WAYLAND_PROTOCOLS_SOURCES=`for p in $WAYLAND_PROTOCOLS ; do printf '%s' "\\$(gen)/\$p-protocol.c " ; done`
    WAYLAND_PROTOCOLS_HEADERS=`for p in $WAYLAND_PROTOCOLS ; do printf '%s' "\\$(gen)/\$p-client-protocol.h " ; done`
    GEN_SOURCES="$GEN_SOURCES $WAYLAND_PROTOCOLS_SOURCES"
    GEN_HEADERS="$GEN_HEADERS $WAYLAND_PROTOCOLS_HEADERS"

    WAYLAND_PROTOCOLS_DEPENDS=`for p in $WAYLAND_PROTOCOLS ; do\
        echo ;\
        printf '%s\n' "\\$(gen)/\$p-client-protocol.h: \\$(srcdir)/wayland-protocols/\$p.xml" ;\
        printf '%s\n' "	@\\$(SHELL) \\$(auxdir)/mkinstalldirs \\$(gen)" ;\
        printf '%s\n' "	\\$(RUN_CMD_GEN)\\$(WAYLAND_SCANNER) client-header \\$< \\$@" ;\
        echo ;\
        printf '%s\n' "\\$(gen)/\$p-protocol.c: \\$(srcdir)/wayland-protocols/\$p.xml" ;\
        printf '%s\n' "	@\\$(SHELL) \\$(auxdir)/mkinstalldirs \\$(gen)" ;\
        printf '%s\n' "	\\$(RUN_CMD_GEN)\\$(WAYLAND_SCANNER) \\$(WAYLAND_SCANNER_CODE_MODE) \\$< \\$@" ;\
        echo ;\
        printf '%s\n' "\\$(objects)/\$p-protocol.lo: \\$(gen)/\$p-protocol.c \\$(gen)/\$p-client-protocol.h" ;\
        printf '%s\n' "	@\\$(SHELL) \\$(auxdir)/mkinstalldirs \\$(objects)" ;\
        printf '%s\n' "	\\$(RUN_CMD_CC)\\$(LIBTOOL) --tag=CC --mode=compile \\$(CC) \\$(CFLAGS) \\$(EXTRA_CFLAGS) $DEPENDENCY_TRACKING_OPTIONS -c \\$< -o \\$@" ;\
        done ;\
        echo ;\
        for s in $WAYLAND_SOURCES ; do printf '%s' "\$s:" ; printf ' \$(gen)/%s-client-protocol.h' $WAYLAND_PROTOCOLS ; echo ; done ; echo`
fi

OBJECTS=`echo $SOURCES`
DEPENDS=`echo $SOURCES | tr ' ' '\n'`
for EXT in asm cc m c S; do
    OBJECTS=`echo "$OBJECTS" | sed 's,[[^ ]]*/\([[^ ]]*\)\.'$EXT',$(objects)/\1.lo,g'`
    DEPENDS=`echo "$DEPENDS" | sed "s,^\\([[^ ]]*\\)/\\([[^ ]]*\\)\\.$EXT\\$,\\\\
\\$(objects)/\\2.lo: \\1/\\2.$EXT \\$(objects)/.created\\\\
	\\$(RUN_CMD_CC)\\$(LIBTOOL) --tag=CC --mode=compile \\$(CC) \\$(CFLAGS) \\$(EXTRA_CFLAGS) $DEPENDENCY_TRACKING_OPTIONS -c \\$< -o \\$@,g"`
done

GEN_OBJECTS=`echo "$GEN_SOURCES" | sed 's,[[^ ]]*/\([[^ ]]*\)\.c,$(objects)/\1.lo,g'`

VERSION_OBJECTS=`echo $VERSION_SOURCES`
VERSION_DEPENDS=`echo $VERSION_SOURCES`
VERSION_OBJECTS=`echo "$VERSION_OBJECTS" | sed 's,[[^ ]]*/\([[^ ]]*\)\.rc,$(objects)/\1.lo,g'`
VERSION_DEPENDS=`echo "$VERSION_DEPENDS" | sed "s,\\([[^ ]]*\\)/\\([[^ ]]*\\)\\.rc,\\\\
\\$(objects)/\\2.lo: \\1/\\2.rc \\$(objects)/.created\\\\
	\\$(RUN_CMD_RC)\\$(LIBTOOL) --mode=compile --tag=RC \\$(RC) -i \\$< -o \\$@,g"`

SDLMAIN_OBJECTS=`echo $SDLMAIN_SOURCES`
SDLMAIN_DEPENDS=`echo $SDLMAIN_SOURCES`
SDLMAIN_OBJECTS=`echo "$SDLMAIN_OBJECTS" | sed 's,[[^ ]]*/\([[^ ]]*\)\.c,$(objects)/\1.lo,g'`
SDLMAIN_DEPENDS=`echo "$SDLMAIN_DEPENDS" | sed "s,\\([[^ ]]*\\)/\\([[^ ]]*\\)\\.c,\\\\
\\$(objects)/\\2.lo: \\1/\\2.c \\$(objects)/.created\\\\
	\\$(RUN_CMD_CC)\\$(LIBTOOL) --tag=CC --mode=compile \\$(CC) \\$(CFLAGS) \\$(EXTRA_CFLAGS) $DEPENDENCY_TRACKING_OPTIONS -c \\$< -o \\$@,g"`

SDLTEST_OBJECTS=`echo $SDLTEST_SOURCES`
SDLTEST_DEPENDS=`echo $SDLTEST_SOURCES`
SDLTEST_OBJECTS=`echo "$SDLTEST_OBJECTS" | sed 's,[[^ ]]*/\([[^ ]]*\)\.c,$(objects)/\1.lo,g'`
SDLTEST_DEPENDS=`echo "$SDLTEST_DEPENDS" | sed "s,\\([[^ ]]*\\)/\\([[^ ]]*\\)\\.c,\\\\
\\$(objects)/\\2.lo: \\1/\\2.c \\$(objects)/.created\\\\
	\\$(RUN_CMD_CC)\\$(LIBTOOL) --tag=CC --mode=compile \\$(CC) \\$(CFLAGS) \\$(EXTRA_CFLAGS) $DEPENDENCY_TRACKING_OPTIONS -c \\$< -o \\$@,g"`

# Set runtime shared library paths as needed

if test "x$enable_rpath" = "xyes" -a "x$enable_shared" = "xyes"; then
  if test $ARCH = bsdi -o $ARCH = freebsd -o $ARCH = linux -o $ARCH = netbsd; then
    SDL_RLD_FLAGS="-Wl,-rpath,\${libdir}"

    AC_MSG_CHECKING(for linker option --enable-new-dtags)
    have_enable_new_dtags=no
    save_LDFLAGS="$LDFLAGS"
    LDFLAGS="$LDFLAGS -Wl,--enable-new-dtags"
    AC_LINK_IFELSE([AC_LANG_PROGRAM([],[])], [
    have_enable_new_dtags=yes
    SDL_RLD_FLAGS="$SDL_RLD_FLAGS -Wl,--enable-new-dtags"
    ],[])
    LDFLAGS="$save_LDFLAGS"
    AC_MSG_RESULT($have_enable_new_dtags)
  fi
  if test $ARCH = solaris; then
    SDL_RLD_FLAGS="-R\${libdir}"
  fi
else
  SDL_RLD_FLAGS=""
fi

SDL_STATIC_LIBS="$EXTRA_LDFLAGS"

dnl Calculate the location of the prefix, relative to the cmake folder
dnl Calculate the location of the prefix, relative to bindir
pkg_cmakedir='$libdir/cmake/SDL2'
AX_COMPUTE_RELATIVE_PATHS([pkg_cmakedir:prefix:cmake_prefix_relpath bindir:prefix:bin_prefix_relpath])
AC_SUBST([cmake_prefix_relpath])
AC_SUBST([bin_prefix_relpath])

dnl Expand the cflags and libraries needed by apps using SDL
AC_SUBST(SDL_CFLAGS)
AC_SUBST(SDL_LIBS)
AC_SUBST(SDL_STATIC_LIBS)
AC_SUBST(SDL_RLD_FLAGS)
PKGCONFIG_DEPENDS=""
AC_SUBST(PKGCONFIG_DEPENDS)
if test x$enable_shared = xyes; then
    PKGCONFIG_LIBS_PRIV="
Libs.private:"
    ENABLE_SHARED_TRUE=
    ENABLE_SHARED_FALSE="#"
else
    PKGCONFIG_LIBS_PRIV=
    ENABLE_SHARED_TRUE="#"
    ENABLE_SHARED_FALSE=
fi
if test x$enable_static = xyes; then
    ENABLE_STATIC_TRUE=
    ENABLE_STATIC_FALSE="#"
else
    ENABLE_STATIC_TRUE="#"
    ENABLE_STATIC_FALSE=
fi
AC_SUBST(PKGCONFIG_LIBS_PRIV)
AC_SUBST(ENABLE_SHARED_TRUE)
AC_SUBST(ENABLE_SHARED_FALSE)
AC_SUBST(ENABLE_STATIC_TRUE)
AC_SUBST(ENABLE_STATIC_FALSE)

dnl Expand the sources and objects needed to build the library
AC_SUBST(ac_aux_dir)
AC_SUBST(INCLUDE)
AC_SUBST(OBJECTS)
AC_SUBST(GEN_HEADERS)
AC_SUBST(GEN_OBJECTS)
AC_SUBST(VERSION_OBJECTS)
AC_SUBST(SDLMAIN_OBJECTS)
AC_SUBST(SDLTEST_OBJECTS)
AC_SUBST(BUILD_CFLAGS)
AC_SUBST(EXTRA_CFLAGS)
AC_SUBST(BUILD_LDFLAGS)
AC_SUBST(EXTRA_LDFLAGS)
AC_SUBST(WAYLAND_SCANNER)
AC_SUBST(WAYLAND_SCANNER_CODE_MODE)

cat >Makefile.rules <<__EOF__

# Build rules for objects
-include \$(OBJECTS:.lo=.d)

# Special dependency for SDL.c, since it depends on SDL_revision.h
$srcdir/src/SDL.c: update-revision
$DEPENDS
$VERSION_DEPENDS
$SDLMAIN_DEPENDS
$SDLTEST_DEPENDS
$WAYLAND_PROTOCOLS_DEPENDS
__EOF__

AC_CONFIG_FILES([
    Makefile:Makefile.in:Makefile.rules sdl2-config sdl2-config.cmake sdl2-config-version.cmake SDL2.spec sdl2.pc
])
AC_CONFIG_COMMANDS([sdl2_config],[chmod a+x sdl2-config])

SUMMARY="SDL2 Configure Summary:\n"
if test x$enable_shared = xyes; then
    SUMMARY="${SUMMARY}Building Shared Libraries\n"
fi
if test x$enable_static = xyes; then
    SUMMARY="${SUMMARY}Building Static Libraries\n"
fi
SUMMARY="${SUMMARY}Enabled modules :${SUMMARY_modules}\n"
SUMMARY="${SUMMARY}Assembly Math   :${SUMMARY_math}\n" 
SUMMARY="${SUMMARY}Audio drivers   :${SUMMARY_audio}\n"
SUMMARY="${SUMMARY}Video drivers   :${SUMMARY_video}\n"
if test x$have_x = xyes; then
    SUMMARY="${SUMMARY}X11 libraries   :${SUMMARY_video_x11}\n"
fi
SUMMARY="${SUMMARY}Input drivers   :${SUMMARY_input}\n"
if test x$have_joystick_virtual = xyes; then
    SUMMARY="${SUMMARY}Enable virtual joystick APIs : YES\n"
else
    SUMMARY="${SUMMARY}Enable virtual joystick APIs : NO\n"
fi
if test x$have_samplerate_h_hdr = xyes; then
    SUMMARY="${SUMMARY}Using libsamplerate : YES\n"
else
    SUMMARY="${SUMMARY}Using libsamplerate : NO\n"
fi
if test x$have_libudev_h_hdr = xyes; then
    SUMMARY="${SUMMARY}Using libudev       : YES\n"
else
    SUMMARY="${SUMMARY}Using libudev       : NO\n"
fi
if test x$have_dbus_dbus_h_hdr = xyes; then
    SUMMARY="${SUMMARY}Using dbus          : YES\n"
else
    SUMMARY="${SUMMARY}Using dbus          : NO\n"
fi
if test x$enable_ime = xyes; then
    SUMMARY="${SUMMARY}Using ime           : YES\n"
else
    SUMMARY="${SUMMARY}Using ime           : NO\n"
fi
if test x$have_ibus_ibus_h_hdr = xyes; then
    SUMMARY="${SUMMARY}Using ibus          : YES\n"
else
    SUMMARY="${SUMMARY}Using ibus          : NO\n"
fi
if test x$have_fcitx = xyes; then
    SUMMARY="${SUMMARY}Using fcitx         : YES\n"
else
    SUMMARY="${SUMMARY}Using fcitx         : NO\n"
fi

if test x$WARN_ABOUT_ARM_SIMD_ASM_MIT = xyes; then
    SUMMARY="${SUMMARY}\nSDL is being built with ARM SIMD optimizations, which\n"
    SUMMARY="${SUMMARY}uses code licensed under the MIT license. If this is a\n"
    SUMMARY="${SUMMARY}problem, please disable that code by rerunning the\n"
    SUMMARY="${SUMMARY}configure script with:\n"
    SUMMARY="${SUMMARY}\n    --disable-arm-simd\n"
fi
if test x$WARN_ABOUT_ARM_NEON_ASM_MIT = xyes; then
    SUMMARY="${SUMMARY}\nSDL is being built with ARM NEON optimizations, which\n"
    SUMMARY="${SUMMARY}uses code licensed under the MIT license. If this is a\n"
    SUMMARY="${SUMMARY}problem, please disable that code by rerunning the\n"
    SUMMARY="${SUMMARY}configure script with:\n"
    SUMMARY="${SUMMARY}\n    --disable-arm-neon\n"
fi

AC_CONFIG_COMMANDS([summary], [printf "$SUMMARY"], [SUMMARY="$SUMMARY"])

AC_OUTPUT
