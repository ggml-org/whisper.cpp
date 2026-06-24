#!/bin/bash

if [ -z "$SDKDIR" ]; then
    SDKDIR="/emsdk"
fi

ENVSCRIPT="$SDKDIR/emsdk_env.sh"
if [ ! -f "$ENVSCRIPT" ]; then
   echo "ERROR: This script expects the Emscripten SDK to be in '$SDKDIR'." 1>&2
   echo "ERROR: Set the \$SDKDIR environment variable to override this." 1>&2
   exit 1
fi

TARBALL="$1"
if [ -z $1 ]; then
    TARBALL=sdl-emscripten.tar.xz
fi

cd `dirname "$0"`
cd ..
SDLBASE=`pwd`

if [ -z "$MAKE" ]; then
    OSTYPE=`uname -s`
    if [ "$OSTYPE" == "Linux" ]; then
        NCPU=`cat /proc/cpuinfo |grep vendor_id |wc -l`
        let NCPU=$NCPU+1
    elif [ "$OSTYPE" = "Darwin" ]; then
        NCPU=`sysctl -n hw.ncpu`
    elif [ "$OSTYPE" = "SunOS" ]; then
        NCPU=`/usr/sbin/psrinfo |wc -l |sed -e 's/^ *//g;s/ *$//g'`
    else
        NCPU=1
    fi

    if [ -z "$NCPU" ]; then
        NCPU=1
    elif [ "$NCPU" = "0" ]; then
        NCPU=1
    fi

    MAKE="make -j$NCPU"
fi

echo "\$MAKE is '$MAKE'"

echo "Setting up Emscripten SDK environment..."
source "$ENVSCRIPT"

echo "Setting up..."
set -x
cd "$SDLBASE"
rm -rf buildbot
mkdir buildbot
pushd buildbot

echo "Configuring..."
emconfigure ../configure --host=wasm32-unknown-emscripten --disable-assembly --disable-threads --disable-cpuinfo CFLAGS="-s USE_SDL=0 -O2 -Wno-warn-absolute-paths -Wdeclaration-after-statement -Werror=declaration-after-statement" --prefix="$PWD/emscripten-sdl2-installed" || exit $?

echo "Building..."
emmake $MAKE || exit $?

echo "Moving things around..."
emmake $MAKE install || exit $?

# Fix up a few things to a real install path
perl -w -pi -e "s#$PWD/emscripten-sdl2-installed#/usr/local#g;" ./emscripten-sdl2-installed/lib/libSDL2.la ./emscripten-sdl2-installed/lib/pkgconfig/sdl2.pc ./emscripten-sdl2-installed/bin/sdl2-config
mkdir -p ./usr
mv ./emscripten-sdl2-installed ./usr/local
tar -cJvvf $TARBALL usr
popd

exit 0

# end of emscripten-buildbot.sh ...

