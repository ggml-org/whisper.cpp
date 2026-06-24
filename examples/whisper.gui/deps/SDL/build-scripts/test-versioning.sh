#!/bin/sh
# Copyright 2022 Collabora Ltd.
# SPDX-License-Identifier: Zlib

set -eu

cd `dirname $0`/..

ref_major=$(sed -ne 's/^#define SDL_MAJOR_VERSION  *//p' include/SDL_version.h)
ref_minor=$(sed -ne 's/^#define SDL_MINOR_VERSION  *//p' include/SDL_version.h)
ref_micro=$(sed -ne 's/^#define SDL_PATCHLEVEL  *//p' include/SDL_version.h)
ref_version="${ref_major}.${ref_minor}.${ref_micro}"

tests=0
failed=0

ok () {
    tests=$(( tests + 1 ))
    echo "ok - $*"
}

not_ok () {
    tests=$(( tests + 1 ))
    echo "not ok - $*"
    failed=1
}

major=$(sed -ne 's/^SDL_MAJOR_VERSION=//p' configure.ac)
minor=$(sed -ne 's/^SDL_MINOR_VERSION=//p' configure.ac)
micro=$(sed -ne 's/^SDL_MICRO_VERSION=//p' configure.ac)
version="${major}.${minor}.${micro}"

if [ "$ref_version" = "$version" ]; then
    ok "configure.ac $version"
else
    not_ok "configure.ac $version disagrees with SDL_version.h $ref_version"
fi

major=$(sed -ne 's/^SDL_MAJOR_VERSION=//p' configure)
minor=$(sed -ne 's/^SDL_MINOR_VERSION=//p' configure)
micro=$(sed -ne 's/^SDL_MICRO_VERSION=//p' configure)
version="${major}.${minor}.${micro}"

if [ "$ref_version" = "$version" ]; then
    ok "configure $version"
else
    not_ok "configure $version disagrees with SDL_version.h $ref_version"
fi

major=$(sed -ne 's/^set(SDL_MAJOR_VERSION \([0-9]*\))$/\1/p' CMakeLists.txt)
minor=$(sed -ne 's/^set(SDL_MINOR_VERSION \([0-9]*\))$/\1/p' CMakeLists.txt)
micro=$(sed -ne 's/^set(SDL_MICRO_VERSION \([0-9]*\))$/\1/p' CMakeLists.txt)
version="${major}.${minor}.${micro}"

if [ "$ref_version" = "$version" ]; then
    ok "CMakeLists.txt $version"
else
    not_ok "CMakeLists.txt $version disagrees with SDL_version.h $ref_version"
fi

major=$(sed -ne 's/.*SDL_MAJOR_VERSION = \([0-9]*\);/\1/p' android-project/app/src/main/java/org/libsdl/app/SDLActivity.java)
minor=$(sed -ne 's/.*SDL_MINOR_VERSION = \([0-9]*\);/\1/p' android-project/app/src/main/java/org/libsdl/app/SDLActivity.java)
micro=$(sed -ne 's/.*SDL_MICRO_VERSION = \([0-9]*\);/\1/p' android-project/app/src/main/java/org/libsdl/app/SDLActivity.java)
version="${major}.${minor}.${micro}"

if [ "$ref_version" = "$version" ]; then
    ok "SDLActivity.java $version"
else
    not_ok "android-project/app/src/main/java/org/libsdl/app/SDLActivity.java $version disagrees with SDL_version.h $ref_version"
fi

major=$(sed -ne 's/^MAJOR_VERSION *= *//p' Makefile.os2)
minor=$(sed -ne 's/^MINOR_VERSION *= *//p' Makefile.os2)
micro=$(sed -ne 's/^MICRO_VERSION *= *//p' Makefile.os2)
version="${major}.${minor}.${micro}"

if [ "$ref_version" = "$version" ]; then
    ok "Makefile.os2 $version"
else
    not_ok "Makefile.os2 $version disagrees with SDL_version.h $ref_version"
fi

major=$(sed -ne 's/^MAJOR_VERSION *= *//p' Makefile.w32)
minor=$(sed -ne 's/^MINOR_VERSION *= *//p' Makefile.w32)
micro=$(sed -ne 's/^MICRO_VERSION *= *//p' Makefile.w32)
version="${major}.${minor}.${micro}"

if [ "$ref_version" = "$version" ]; then
    ok "Makefile.w32 $version"
else
    not_ok "Makefile.w32 $version disagrees with SDL_version.h $ref_version"
fi

tuple=$(sed -ne 's/^ *FILEVERSION *//p' src/main/windows/version.rc | tr -d '\r')
ref_tuple="${ref_major},${ref_minor},${ref_micro},0"

if [ "$ref_tuple" = "$tuple" ]; then
    ok "version.rc FILEVERSION $tuple"
else
    not_ok "version.rc FILEVERSION $tuple disagrees with SDL_version.h $ref_tuple"
fi

tuple=$(sed -ne 's/^ *PRODUCTVERSION *//p' src/main/windows/version.rc | tr -d '\r')

if [ "$ref_tuple" = "$tuple" ]; then
    ok "version.rc PRODUCTVERSION $tuple"
else
    not_ok "version.rc PRODUCTVERSION $tuple disagrees with SDL_version.h $ref_tuple"
fi

tuple=$(sed -Ene 's/^ *VALUE "FileVersion", "([0-9, ]*)\\0"\r?$/\1/p' src/main/windows/version.rc | tr -d '\r')
ref_tuple="${ref_major}, ${ref_minor}, ${ref_micro}, 0"

if [ "$ref_tuple" = "$tuple" ]; then
    ok "version.rc FileVersion $tuple"
else
    not_ok "version.rc FileVersion $tuple disagrees with SDL_version.h $ref_tuple"
fi

tuple=$(sed -Ene 's/^ *VALUE "ProductVersion", "([0-9, ]*)\\0"\r?$/\1/p' src/main/windows/version.rc | tr -d '\r')

if [ "$ref_tuple" = "$tuple" ]; then
    ok "version.rc ProductVersion $tuple"
else
    not_ok "version.rc ProductVersion $tuple disagrees with SDL_version.h $ref_tuple"
fi

version=$(sed -Ene '/CFBundleShortVersionString/,+1 s/.*<string>(.*)<\/string>.*/\1/p' Xcode/SDL/Info-Framework.plist)

if [ "$ref_version" = "$version" ]; then
    ok "Info-Framework.plist CFBundleShortVersionString $version"
else
    not_ok "Info-Framework.plist CFBundleShortVersionString $version disagrees with SDL_version.h $ref_version"
fi

version=$(sed -Ene '/CFBundleVersion/,+1 s/.*<string>(.*)<\/string>.*/\1/p' Xcode/SDL/Info-Framework.plist)

if [ "$ref_version" = "$version" ]; then
    ok "Info-Framework.plist CFBundleVersion $version"
else
    not_ok "Info-Framework.plist CFBundleVersion $version disagrees with SDL_version.h $ref_version"
fi

version=$(sed -Ene 's/Title SDL (.*)/\1/p' Xcode/SDL/pkg-support/SDL.info)

if [ "$ref_version" = "$version" ]; then
    ok "SDL.info Title $version"
else
    not_ok "SDL.info Title $version disagrees with SDL_version.h $ref_version"
fi

marketing=$(sed -Ene 's/.*MARKETING_VERSION = (.*);/\1/p' Xcode/SDL/SDL.xcodeproj/project.pbxproj)

ref="$ref_version
$ref_version"

if [ "$ref" = "$marketing" ]; then
    ok "project.pbxproj MARKETING_VERSION is consistent"
else
    not_ok "project.pbxproj MARKETING_VERSION is inconsistent, expected $ref, got $marketing"
fi

# For simplicity this assumes we'll never break ABI before SDL 3.
dylib_compat=$(sed -Ene 's/.*DYLIB_COMPATIBILITY_VERSION = (.*);$/\1/p' Xcode/SDL/SDL.xcodeproj/project.pbxproj)

case "$ref_minor" in
    (*[02468])
        major="$(( ref_minor * 100 + 1 ))"
        minor="0"
        ;;
    (*)
        major="$(( ref_minor * 100 + ref_micro + 1 ))"
        minor="0"
        ;;
esac

ref="${major}.${minor}.0
${major}.${minor}.0
${major}.${minor}.0
${major}.${minor}.0"

if [ "$ref" = "$dylib_compat" ]; then
    ok "project.pbxproj DYLIB_COMPATIBILITY_VERSION is consistent"
else
    not_ok "project.pbxproj DYLIB_COMPATIBILITY_VERSION is inconsistent, expected $ref, got $dylib_compat"
fi

dylib_cur=$(sed -Ene 's/.*DYLIB_CURRENT_VERSION = (.*);$/\1/p' Xcode/SDL/SDL.xcodeproj/project.pbxproj)

case "$ref_minor" in
    (*[02468])
        major="$(( ref_minor * 100 + 1 ))"
        minor="$ref_micro"
        ;;
    (*)
        major="$(( ref_minor * 100 + ref_micro + 1 ))"
        minor="0"
        ;;
esac

ref="${major}.${minor}.0
${major}.${minor}.0
${major}.${minor}.0
${major}.${minor}.0"

if [ "$ref" = "$dylib_cur" ]; then
    ok "project.pbxproj DYLIB_CURRENT_VERSION is consistent"
else
    not_ok "project.pbxproj DYLIB_CURRENT_VERSION is inconsistent, expected $ref, got $dylib_cur"
fi

echo "1..$tests"
exit "$failed"
