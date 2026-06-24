#!/bin/bash

set -e

if ! [ "x$ANDROID_NDK_HOME" != "x" -a -d "$ANDROID_NDK_HOME" ]; then
    echo "ANDROID_NDK_HOME environment variable is not set"
    exit 1
fi

if ! [ "x$ANDROID_HOME" != "x" -a -d "$ANDROID_HOME" ]; then
    echo "ANDROID_HOME environment variable is not set"
    exit 1
fi

if [ "x$ANDROID_API" = "x" ]; then
    ANDROID_API="$(ls "$ANDROID_HOME/platforms" | grep -E "^android-[0-9]+$" | sed 's/android-//' | sort -n -r | head -1)"
    if [ "x$ANDROID_API" = "x" ]; then
        echo "No Android platform found in $ANDROID_HOME/platforms"
        exit 1
    fi
else
    if ! [ -d "$ANDROID_HOME/platforms/android-$ANDROID_API" ]; then
        echo "Android api version $ANDROID_API is not available ($ANDROID_HOME/platforms/android-$ANDROID_API does not exist)" >2
        exit 1
    fi
fi

android_platformdir="$ANDROID_HOME/platforms/android-$ANDROID_API"

echo "Building for android api version $ANDROID_API"
echo "android_platformdir=$android_platformdir"

scriptdir=$(cd -P -- "$(dirname -- "$0")" && printf '%s\n' "$(pwd -P)")
sdl_root=$(cd -P -- "$(dirname -- "$0")/.." && printf '%s\n' "$(pwd -P)")

build_root="${sdl_root}/build-android-prefab"

android_abis="armeabi-v7a arm64-v8a x86 x86_64"
android_api=19
android_ndk=21
android_stl="c++_shared"

sdl_major=$(sed -ne 's/^#define SDL_MAJOR_VERSION  *//p' "${sdl_root}/include/SDL_version.h")
sdl_minor=$(sed -ne 's/^#define SDL_MINOR_VERSION  *//p' "${sdl_root}/include/SDL_version.h")
sdl_patch=$(sed -ne 's/^#define SDL_PATCHLEVEL  *//p' "${sdl_root}/include/SDL_version.h")
sdl_version="${sdl_major}.${sdl_minor}.${sdl_patch}"
echo "Building Android prefab package for SDL version $sdl_version"

prefabhome="${build_root}/prefab-${sdl_version}"
rm -rf "$prefabhome"
mkdir -p "${prefabhome}"

build_cmake_projects() {
    for android_abi in $android_abis; do
        echo "Configuring CMake project for $android_abi"
        cmake -S "$sdl_root" -B "${build_root}/build_${android_abi}" \
            -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
            -DANDROID_PLATFORM=${android_platform} \
            -DANDROID_ABI=${android_abi} \
            -DSDL_SHARED=ON \
            -DSDL_STATIC=ON \
            -DSDL_STATIC_PIC=ON \
            -DSDL_TEST=ON \
            -DSDL2_DISABLE_SDL2MAIN=OFF \
            -DSDL2_DISABLE_INSTALL=OFF \
            -DCMAKE_INSTALL_PREFIX="${build_root}/build_${android_abi}/prefix" \
            -DCMAKE_INSTALL_INCLUDEDIR=include \
            -DCMAKE_INSTALL_LIBDIR=lib \
            -DCMAKE_BUILD_TYPE=Release \
            -GNinja

        rm -rf "${build_root}/build_${android_abi}/prefix"

        echo "Building CMake project for $android_abi"
        cmake --build "${build_root}/build_${android_abi}"

        echo "Installing CMake project for $android_abi"
        cmake --install "${build_root}/build_${android_abi}"
    done
}

classes_sources_jar_path="${prefabhome}/classes-sources.jar"
classes_jar_path="${prefabhome}/classes.jar"
compile_java() {
    classes_sources_root="${prefabhome}/classes-sources"

    rm -rf "${classes_sources_root}"
    mkdir -p "${classes_sources_root}/META-INF"

    echo "Copying LICENSE.txt to java build folder"
    cp "$sdl_root/LICENSE.txt" "${classes_sources_root}/META-INF"

    echo "Copy JAVA sources to java build folder"
    cp -r "$sdl_root/android-project/app/src/main/java/org" "${classes_sources_root}"

    java_sourceslist_path="${prefabhome}/java_sources.txt"
    pushd "${classes_sources_root}"
        echo "Collecting sources for classes-sources.jar"
        find "." -name "*.java" >"${java_sourceslist_path}"
        find "META-INF" -name "*" >>"${java_sourceslist_path}"

        echo "Creating classes-sources.jar"
        jar -cf "${classes_sources_jar_path}" "@${java_sourceslist_path}"
    popd

    classes_root="${prefabhome}/classes"
    mkdir -p "${classes_root}/META-INF"
    cp "$sdl_root/LICENSE.txt" "${classes_root}/META-INF"
    java_sourceslist_path="${prefabhome}/java_sources.txt"

    echo "Collecting sources for classes.jar"
    find "$sdl_root/android-project/app/src/main/java" -name "*.java" >"${java_sourceslist_path}"

    echo "Compiling classes"
    javac -encoding utf-8 -classpath "$android_platformdir/android.jar" -d "${classes_root}" "@${java_sourceslist_path}"

    java_classeslist_path="${prefabhome}/java_classes.txt"
    pushd "${classes_root}"
        find "." -name "*.class" >"${java_classeslist_path}"
        find "META-INF" -name "*" >>"${java_classeslist_path}"
        echo "Creating classes.jar"
        jar -cf "${classes_jar_path}" "@${java_classeslist_path}"
    popd
}

pom_filename="SDL${sdl_major}-${sdl_version}.pom"
pom_filepath="${prefabhome}/${pom_filename}"
create_pom_xml() {
    echo "Creating ${pom_filename}"
    cat >"${pom_filepath}" <<EOF
<project>
  <modelVersion>4.0.0</modelVersion>
  <groupId>org.libsdl.android</groupId>
  <artifactId>SDL${sdl_major}</artifactId>
  <version>${sdl_version}</version>
  <packaging>aar</packaging>
  <name>SDL${sdl_major}</name>
  <description>The AAR for SDL${sdl_major}</description>
  <url>https://libsdl.org/</url>
  <licenses>
    <license>
      <name>zlib License</name>
      <url>https://github.com/libsdl-org/SDL/blob/main/LICENSE.txt</url>
      <distribution>repo</distribution>
    </license>
  </licenses>
  <scm>
    <connection>scm:git:https://github.com/libsdl-org/SDL</connection>
    <url>https://github.com/libsdl-org/SDL</url>
  </scm>
</project>
EOF
}

create_aar_androidmanifest() {
    echo "Creating AndroidManifest.xml"
    cat >"${aar_root}/AndroidManifest.xml" <<EOF
<manifest
    xmlns:android="http://schemas.android.com/apk/res/android"
    package="org.libsdl.android" android:versionCode="1"
    android:versionName="1.0">
	<uses-sdk android:minSdkVersion="16"
              android:targetSdkVersion="29"/>
</manifest>
EOF
}

echo "Creating AAR root directory"
aar_root="${prefabhome}/SDL${sdl_major}-${sdl_version}"
mkdir -p "${aar_root}"

aar_metainfdir_path=${aar_root}/META-INF
mkdir -p "${aar_metainfdir_path}"
cp "${sdl_root}/LICENSE.txt" "${aar_metainfdir_path}"

prefabworkdir="${aar_root}/prefab"
mkdir -p "${prefabworkdir}"

cat >"${prefabworkdir}/prefab.json" <<EOF
{
  "schema_version": 2,
  "name": "SDL$sdl_major",
  "version": "$sdl_version",
  "dependencies": []
}
EOF

modulesworkdir="${prefabworkdir}/modules"
mkdir -p "${modulesworkdir}"

create_shared_sdl_module() {
    echo "Creating SDL${sdl_major} prefab module"
    for android_abi in $android_abis; do
        sdl_moduleworkdir="${modulesworkdir}/SDL${sdl_major}"
        mkdir -p "${sdl_moduleworkdir}"

        abi_build_prefix="${build_root}/build_${android_abi}/prefix"

        cat >"${sdl_moduleworkdir}/module.json" <<EOF
{
  "export_libraries": [],
  "library_name": "libSDL${sdl_major}"
}
EOF
        mkdir -p "${sdl_moduleworkdir}/include"
        cp -r "${abi_build_prefix}/include/SDL${sdl_major}/"* "${sdl_moduleworkdir}/include/"
        rm "${sdl_moduleworkdir}/include/SDL_config.h"
        cp "$sdl_root/include/SDL_config.h" "$sdl_root/include/SDL_config_android.h" "${sdl_moduleworkdir}/include/"

        abi_sdllibdir="${sdl_moduleworkdir}/libs/android.${android_abi}"
        mkdir -p "${abi_sdllibdir}"
        cat >"${abi_sdllibdir}/abi.json" <<EOF
{
  "abi": "${android_abi}",
  "api": ${android_api},
  "ndk": ${android_ndk},
  "stl": "${android_stl}",
  "static": false
}
EOF
        cp "${abi_build_prefix}/lib/libSDL${sdl_major}.so" "${abi_sdllibdir}"
    done
}

create_static_sdl_module() {
    echo "Creating SDL${sdl_major}-static prefab module"
    for android_abi in $android_abis; do
        sdl_moduleworkdir="${modulesworkdir}/SDL${sdl_major}-static"
        mkdir -p "${sdl_moduleworkdir}"

        abi_build_prefix="${build_root}/build_${android_abi}/prefix"

        cat >"${sdl_moduleworkdir}/module.json" <<EOF
{
  "export_libraries": ["-ldl", "-lGLESv1_CM", "-lGLESv2", "-llog", "-landroid", "-lOpenSLES"]
  "library_name": "libSDL${sdl_major}"
}
EOF
        mkdir -p "${sdl_moduleworkdir}/include"
        cp -r "${abi_build_prefix}/include/SDL${sdl_major}/"* "${sdl_moduleworkdir}/include"
        rm "${sdl_moduleworkdir}/include/SDL_config.h"
        cp "$sdl_root/include/SDL_config.h" "$sdl_root/include/SDL_config_android.h" "${sdl_moduleworkdir}/include/"

        abi_sdllibdir="${sdl_moduleworkdir}/libs/android.${android_abi}"
        mkdir -p "${abi_sdllibdir}"
        cat >"${abi_sdllibdir}/abi.json" <<EOF
{
  "abi": "${android_abi}",
  "api": ${android_api},
  "ndk": ${android_ndk},
  "stl": "${android_stl}",
  "static": true
}
EOF
        cp "${abi_build_prefix}/lib/libSDL${sdl_major}.a" "${abi_sdllibdir}"
    done
}

create_sdlmain_module() {
    echo "Creating SDL${sdl_major}main prefab module"
    for android_abi in $android_abis; do
        sdl_moduleworkdir="${modulesworkdir}/SDL${sdl_major}main"
        mkdir -p "${sdl_moduleworkdir}"

        abi_build_prefix="${build_root}/build_${android_abi}/prefix"

        cat >"${sdl_moduleworkdir}/module.json" <<EOF
{
  "export_libraries": [],
  "library_name": "libSDL${sdl_major}main"
}
EOF
        abi_sdllibdir="${sdl_moduleworkdir}/libs/android.${android_abi}"
        mkdir -p "${abi_sdllibdir}"
        cat >"${abi_sdllibdir}/abi.json" <<EOF
{
  "abi": "${android_abi}",
  "api": ${android_api},
  "ndk": ${android_ndk},
  "stl": "${android_stl}",
  "static": true
}
EOF
        cp "${abi_build_prefix}/lib/libSDL${sdl_major}main.a" "${abi_sdllibdir}"
    done
}

create_sdltest_module() {
    echo "Creating SDL${sdl_major}test prefab module"
    for android_abi in $android_abis; do
        sdl_moduleworkdir="${modulesworkdir}/SDL${sdl_major}test"
        mkdir -p "${sdl_moduleworkdir}"

        abi_build_prefix="${build_root}/build_${android_abi}/prefix"

        cat >"${sdl_moduleworkdir}/module.json" <<EOF
{
  "export_libraries": [],
  "library_name": "libSDL${sdl_major}_test"
}
EOF
        abi_sdllibdir="${sdl_moduleworkdir}/libs/android.${android_abi}"
        mkdir -p "${abi_sdllibdir}"
        cat >"${abi_sdllibdir}/abi.json" <<EOF
{
  "abi": "${android_abi}",
  "api": ${android_api},
  "ndk": ${android_ndk},
  "stl": "${android_stl}",
  "static": true
}
EOF
        cp "${abi_build_prefix}/lib/libSDL${sdl_major}_test.a" "${abi_sdllibdir}"
    done
}

build_cmake_projects

compile_java

create_pom_xml

create_aar_androidmanifest

create_shared_sdl_module

create_static_sdl_module

create_sdlmain_module

create_sdltest_module

pushd "${aar_root}"
    aar_filename="SDL${sdl_major}-${sdl_version}.aar"
    cp "${classes_jar_path}" ./classes.jar
    cp "${classes_sources_jar_path}" ./classes-sources.jar
    zip -r "${aar_filename}" AndroidManifest.xml classes.jar classes-sources.jar prefab META-INF
    zip -Tv "${aar_filename}" 2>/dev/null ;
    mv "${aar_filename}" "${prefabhome}"
popd

maven_filename="SDL${sdl_major}-${sdl_version}.zip"

pushd "${prefabhome}"
    zip_filename="SDL${sdl_major}-${sdl_version}.zip"
    zip "${maven_filename}" "${aar_filename}" "${pom_filename}" 2>/dev/null;
    zip -Tv "${zip_filename}" 2>/dev/null;
popd

echo "Prefab zip is ready at ${prefabhome}/${aar_filename}"
echo "Maven archive is ready at ${prefabhome}/${zip_filename}"
