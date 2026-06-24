#!/bin/bash

# This is the script buildbot.libsdl.org uses to cross-compile SDL2 from
#  amd64 Linux to NaCl.

# PLEASE NOTE that we have reports that SDL built with pepper_49 (current
#  stable release as of November 10th, 2016) is broken. Please retest
#  when something newer becomes stable and then decide if this was SDL's
#  bug or NaCl's bug.  --ryan.
export NACL_SDK_ROOT="/nacl_sdk/pepper_47"

TARBALL="$1"
if [ -z $1 ]; then
    TARBALL=sdl-nacl.tar.xz
fi

OSTYPE=`uname -s`
if [ "$OSTYPE" != "Linux" ]; then
    # !!! FIXME
    echo "This only works on x86 or x64-64 Linux at the moment." 1>&2
    exit 1
fi

if [ "x$MAKE" == "x" ]; then
    NCPU=`cat /proc/cpuinfo |grep vendor_id |wc -l`
    let NCPU=$NCPU+1
    MAKE="make -j$NCPU"
fi

BUILDBOTDIR="nacl-buildbot"
PARENTDIR="$PWD"

set -e
set -x
rm -f $TARBALL
rm -rf $BUILDBOTDIR
mkdir -p $BUILDBOTDIR
pushd $BUILDBOTDIR

# !!! FIXME: ccache?
export CC="$NACL_SDK_ROOT/toolchain/linux_pnacl/bin/pnacl-clang"
export CFLAGS="$CFLAGS -I$NACL_SDK_ROOT/include -I$NACL_SDK_ROOT/include/pnacl"
export AR="$NACL_SDK_ROOT/toolchain/linux_pnacl/bin/pnacl-ar"
export LD="$NACL_SDK_ROOT/toolchain/linux_pnacl/bin/pnacl-ar"
export RANLIB="$NACL_SDK_ROOT/toolchain/linux_pnacl/bin/pnacl-ranlib"

../configure --host=pnacl --prefix=$PWD/nacl-sdl2-installed
$MAKE
$MAKE install
# Fix up a few things to a real install path
perl -w -pi -e "s#$PWD/nacl-sdl2-installed#/usr/local#g;" ./nacl-sdl2-installed/lib/libSDL2.la ./nacl-sdl2-installed/lib/pkgconfig/sdl2.pc ./nacl-sdl2-installed/bin/sdl2-config
mkdir -p ./usr
mv ./nacl-sdl2-installed ./usr/local

popd
tar -cJvvf $TARBALL -C $BUILDBOTDIR usr
rm -rf $BUILDBOTDIR

set +x
echo "All done. Final installable is in $TARBALL ...";

