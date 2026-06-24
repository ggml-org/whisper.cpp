#!/bin/bash
if [ -z "$1" ] && [ -z "$NACL_SDK_ROOT" ]; then
    echo "Usage: ./naclbuild ~/nacl/pepper_35"
    echo "This will build SDL for Native Client, and testgles2.c as a demo"
    echo "You can set env vars CC, AR, LD and RANLIB to override the default PNaCl toolchain used"
    echo "You can set env var SOURCES to select a different source file than testgles2.c"
    exit 1
fi

if [ -n "$1" ]; then
    NACL_SDK_ROOT="$1"
fi

CC=""

if [ -n "$2" ]; then
    CC="$2"
fi

echo "Using SDK at $NACL_SDK_ROOT"

export NACL_SDK_ROOT="$NACL_SDK_ROOT"
export CFLAGS="$CFLAGS -I$NACL_SDK_ROOT/include -I$NACL_SDK_ROOT/include/pnacl"

NCPUS="1"
case "$OSTYPE" in
    darwin*)
        NCPU=`sysctl -n hw.ncpu`
        ;; 
    linux*)
        if [ -n `which nproc` ]; then
            NCPUS=`nproc`
        fi  
        ;;
  *);;
esac

CURDIR=`pwd -P`
SDLPATH="$( cd "$(dirname "$0")/.." ; pwd -P )"
BUILDPATH="$SDLPATH/build/nacl"
TESTBUILDPATH="$BUILDPATH/test"
SDL2_STATIC="$BUILDPATH/build/.libs/libSDL2.a"
mkdir -p $BUILDPATH
mkdir -p $TESTBUILDPATH

if [ -z "$CC" ]; then
    export CC="$NACL_SDK_ROOT/toolchain/linux_pnacl/bin/pnacl-clang"
fi
if [ -z "$AR" ]; then
    export AR="$NACL_SDK_ROOT/toolchain/linux_pnacl/bin/pnacl-ar"
fi
if [ -z "$LD" ]; then
    export LD="$NACL_SDK_ROOT/toolchain/linux_pnacl/bin/pnacl-ar"
fi
if [ -z "$RANLIB" ]; then
    export RANLIB="$NACL_SDK_ROOT/toolchain/linux_pnacl/bin/pnacl-ranlib"
fi

if [ -z "$SOURCES" ]; then
    export SOURCES="$SDLPATH/test/testgles2.c"
fi

if [ ! -f "$CC" ]; then
    echo "Could not find compiler at $CC"
    exit 1
fi




cd $BUILDPATH
$SDLPATH/configure --host=pnacl --prefix $TESTBUILDPATH
make -j$NCPUS CFLAGS="$CFLAGS -I./include"
make install

if [ ! -f "$SDL2_STATIC" ]; then
    echo "Build failed! $SDL2_STATIC"
    exit 1
fi

echo "Building test"
cp -f $SDLPATH/test/nacl/* $TESTBUILDPATH
# Some tests need these resource files
cp -f $SDLPATH/test/*.bmp $TESTBUILDPATH
cp -f $SDLPATH/test/*.wav $TESTBUILDPATH
cp -f $SDL2_STATIC $TESTBUILDPATH

# Copy user sources
_SOURCES=($SOURCES)
for src in "${_SOURCES[@]}"
do
    cp $src $TESTBUILDPATH
done
export SOURCES="$SOURCES"

cd $TESTBUILDPATH
make -j$NCPUS CONFIG="Release" CFLAGS="$CFLAGS -I$TESTBUILDPATH/include/SDL2 -I$SDLPATH/include"
make -j$NCPUS CONFIG="Debug" CFLAGS="$CFLAGS -I$TESTBUILDPATH/include/SDL2 -I$SDLPATH/include"

echo
echo "Run the test with: "
echo "cd $TESTBUILDPATH;python -m SimpleHTTPServer"
echo "Then visit http://localhost:8000 with Chrome"

cd $CURDIR
