#!/bin/bash
#
# Build the Android libraries without needing a project
# (AndroidManifest.xml, jni/{Application,Android}.mk, etc.)
#
# Usage: androidbuildlibs.sh [arg for ndk-build ...]"
#
# Useful NDK arguments:
#
#  NDK_DEBUG=1          - build debug version
#  NDK_LIBS_OUT=<dest>  - specify alternate destination for installable
#                         modules.
#
# Note that SDLmain is not an installable module (.so) so libSDLmain.a
# can be found in $obj/local/<abi> along with the unstripped libSDL.so.
#


# Android.mk is in srcdir
srcdir=`dirname $0`/..
srcdir=`cd $srcdir && pwd`
cd $srcdir


#
# Create the build directories
#

build=build
buildandroid=$build/android
obj=
lib=
ndk_args=

# Allow an external caller to specify locations.
for arg in $*; do
    if [ "${arg:0:8}" == "NDK_OUT=" ]; then
        obj=${arg#NDK_OUT=}
    elif [ "${arg:0:13}" == "NDK_LIBS_OUT=" ]; then
        lib=${arg#NDK_LIBS_OUT=}
    else
        ndk_args="$ndk_args $arg"
    fi
done

if [ -z $obj ]; then
    obj=$buildandroid/obj
fi
if [ -z $lib ]; then
    lib=$buildandroid/lib
fi

for dir in $build $buildandroid $obj $lib; do
    if test -d $dir; then
        :
    else
        mkdir $dir || exit 1
    fi
done


# APP_* variables set in the environment here will not be seen by the
# ndk-build makefile segments that use them, e.g., default-application.mk.
# For consistency, pass all values on the command line.
ndk-build \
    NDK_PROJECT_PATH=null \
    NDK_OUT=$obj \
    NDK_LIBS_OUT=$lib \
    APP_BUILD_SCRIPT=Android.mk \
    APP_ABI="armeabi-v7a arm64-v8a x86 x86_64" \
    APP_PLATFORM=android-16 \
    APP_MODULES="SDL2 SDL2_main" \
    $ndk_args
