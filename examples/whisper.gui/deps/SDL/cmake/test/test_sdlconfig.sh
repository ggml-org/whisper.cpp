#!/bin/sh

if test "x$CC" = "x"; then
    CC=cc
fi

machine="$($CC -dumpmachine)"
case "$machine" in
    *mingw* )
        EXEPREFIX=""
        EXESUFFIX=".exe"
        ;;
    *android* )
        EXEPREFIX="lib"
        EXESUFFIX=".so"
        LDFLAGS="$LDFLAGS -shared"
        ;;
    * )
        EXEPREFIX=""
        EXESUFFIX=""
        ;;
esac

set -e

# Get the canonical path of the folder containing this script
testdir=$(cd -P -- "$(dirname -- "$0")" && printf '%s\n' "$(pwd -P)")
SDL_CFLAGS="$( sdl2-config --cflags )"
SDL_LDFLAGS="$( sdl2-config --libs )"
SDL_STATIC_LDFLAGS="$( sdl2-config --static-libs )"

compile_cmd="$CC -c "$testdir/main_gui.c" -o main_gui_sdlconfig.c.o $CFLAGS $SDL_CFLAGS"
link_cmd="$CC main_gui_sdlconfig.c.o -o ${EXEPREFIX}main_gui_sdlconfig${EXESUFFIX} $SDL_LDFLAGS $LDFLAGS"
static_link_cmd="$CC main_gui_sdlconfig.c.o -o ${EXEPREFIX}main_gui_sdlconfig_static${EXESUFFIX} $SDL_STATIC_LDFLAGS $LDFLAGS"

echo "-- CC:                 $CC"
echo "-- CFLAGS:             $CFLAGS"
echo "-- LDFLAGS:            $LDFLAGS"
echo "-- SDL_CFLAGS:         $SDL_CFLAGS"
echo "-- SDL_LDFLAGS:        $SDL_LDFLAGS"
echo "-- SDL_STATIC_LDFLAGS: $SDL_STATIC_LDFLAGS"

echo "-- COMPILE:       $compile_cmd"
echo "-- LINK:          $link_cmd"
echo "-- STATIC_LINK:   $static_link_cmd"

set -x

$compile_cmd
$link_cmd
$static_link_cmd
