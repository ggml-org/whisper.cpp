#!/bin/bash
#
# Synchronize ggml changes to whisper.cpp
#
# Usage:
#
#   $ cd /path/to/whisper.cpp
#   $ ./scripts/sync-ggml-am.sh -skip hash0,hash1,hash2...
#

set -e

sd=$(dirname $0)
cd $sd/../

SRC_WHISPER=$(pwd)
SRC_GGML=$(cd ../ggml; pwd)

if [ ! -d $SRC_GGML ]; then
    echo "ggml not found at $SRC_GGML"
    exit 1
fi

lc=$(cat $SRC_WHISPER/scripts/sync-ggml.last)
echo "Syncing ggml changes since commit $lc"

to_skip=""
if [ "$1" == "-skip" ]; then
    to_skip=$2
fi

cd $SRC_GGML

git log --oneline $lc..HEAD
git log --oneline $lc..HEAD --reverse | grep -v "(whisper/[0-9]*)" | cut -d' ' -f1 > $SRC_WHISPER/ggml-commits

if [ ! -s $SRC_WHISPER/ggml-commits ]; then
    rm -v $SRC_WHISPER/ggml-commits
    echo "No new commits"
    exit 0
fi

if [ -f $SRC_WHISPER/ggml-src.patch ]; then
    rm -v $SRC_WHISPER/ggml-src.patch
fi

while read c; do
    if [ -n "$to_skip" ]; then
        if [[ $to_skip == *"$c"* ]]; then
            echo "Skipping $c"
            continue
        fi
    fi

    git format-patch -k $c~1..$c --stdout -- \
        CMakeLists.txt \
        src/CMakeLists.txt \
        cmake/BuildTypes.cmake \
        cmake/GitVars.cmake \
        cmake/common.cmake \
        cmake/ggml-config.cmake.in \
        src/ggml-cpu/cmake/FindSIMD.cmake \
        src/ggml*.h \
        src/ggml* \
        include/ggml*.h \
        include/gguf*.h \
        examples/common.h \
        examples/common.cpp \
        examples/common-ggml.h \
        examples/common-ggml.cpp \
        LICENSE \
        scripts/gen-authors.sh \
        >> $SRC_WHISPER/ggml-src.patch
done < $SRC_WHISPER/ggml-commits

rm -v $SRC_WHISPER/ggml-commits

# delete files if empty
if [ ! -s $SRC_WHISPER/ggml-src.patch ]; then
    rm -v $SRC_WHISPER/ggml-src.patch
fi

cd $SRC_WHISPER

if [ -f $SRC_WHISPER/ggml-src.patch ]; then
    # replace PR numbers
    #
    # Subject: some text (#1234)
    # Subject: some text (ggml/1234)
    cat ggml-src.patch | sed -e 's/^Subject: \(.*\) (#\([0-9]*\))/Subject: \1 (ggml\/\2)/' > ggml-src.patch.tmp
    mv ggml-src.patch.tmp ggml-src.patch

    cat ggml-src.patch | sed -e 's/^\(.*\) (#\([0-9]*\))$/\1 (ggml\/\2)/' > ggml-src.patch.tmp
    mv ggml-src.patch.tmp ggml-src.patch

    # replace filenames:
    #
    # CMakelists.txt       -> ggml/CMakeLists.txt
    # src/CMakeLists.txt   -> ggml/src/CMakeLists.txt
    #
    # cmake/BuildTypes.cmake            -> ggml/cmake/BuildTypes.cmake
    # cmake/GitVars.cmake               -> ggml/cmake/GitVars.cmake
    # cmake/common.cmake                -> ggml/cmake/common.cmake
    # cmake/ggml-config.cmake.in        -> ggml/cmake/ggml-config.cmake.in
    # src/ggml-cpu/cmake/FindSIMD.cmake -> ggml/src/ggml-cpu/cmake/FindSIMD.cmake
    #
    # src/ggml* -> ggml/src/ggml*.c
    #
    # include/ggml*.h -> ggml/include/ggml*.h
    # include/gguf*.h -> ggml/include/gguf*.h
    #
    # examples/common.h        -> examples/common.h
    # examples/common.cpp      -> examples/common.cpp
    # examples/common-ggml.h   -> examples/common-ggml.h
    # examples/common-ggml.cpp -> examples/common-ggml.cpp
    #
    # LICENSE                     -> LICENSE
    # ggml/scripts/gen-authors.sh -> scripts/gen-authors.sh

    cat ggml-src.patch | sed -E \
        -e 's/(^[[:space:]]| [ab]\/)CMakeLists.txt/\1ggml\/CMakeLists.txt/g' \
        -e 's/(^[[:space:]]| [ab]\/)src\/CMakeLists.txt/\1ggml\/src\/CMakeLists.txt/g' \
        -e 's/(^[[:space:]]| [ab]\/)cmake\/BuildTypes.cmake/\1ggml\/cmake\/BuildTypes.cmake/g' \
        -e 's/(^[[:space:]]| [ab]\/)cmake\/GitVars.cmake/\1ggml\/cmake\/GitVars.cmake/g' \
        -e 's/(^[[:space:]]| [ab]\/)cmake\/common.cmake/\1ggml\/cmake\/common.cmake/g' \
        -e 's/(^[[:space:]]| [ab]\/)cmake\/ggml-config.cmake.in/\1ggml\/cmake\/ggml-config.cmake.in/g' \
        -e 's/(^[[:space:]]| [ab]\/)src\/ggml-cpu\/cmake\/FindSIMD.cmake/\1ggml\/src\/ggml-cpu\/cmake\/FindSIMD.cmake/g' \
        -e 's/([[:space:]]| [ab]\/)src\/ggml(.*)/\1ggml\/src\/ggml\2/g' \
        -e 's/(^[[:space:]]| [ab]\/)include\/ggml(.*)\.h/\1ggml\/include\/ggml\2.h/g' \
        -e 's/(^[[:space:]]| [ab]\/)include\/gguf(.*)\.h/\1ggml\/include\/gguf\2.h/g' \
        -e 's/(^[[:space:]]| [ab]\/)examples\/common\.h/\1examples\/common.h/g' \
        -e 's/(^[[:space:]]| [ab]\/)examples\/common\.cpp/\1examples\/common.cpp/g' \
        -e 's/(^[[:space:]]| [ab]\/)examples\/common-ggml\.h/\1examples\/common-ggml.h/g' \
        -e 's/(^[[:space:]]| [ab]\/)examples\/common-ggml\.cpp/\1examples\/common-ggml.cpp/g' \
        -e 's/(^[[:space:]]| [ab]\/)LICENSE/\1LICENSE/g' \
        -e 's/(^[[:space:]]| [ab]\/)scripts\/gen-authors\.sh/\1scripts\/gen-authors.sh/g' \
        > ggml-src.patch.tmp
    mv ggml-src.patch.tmp ggml-src.patch

    git am ggml-src.patch

    rm -v $SRC_WHISPER/ggml-src.patch
fi

# update last commit
cd $SRC_GGML
git log -1 --format=%H > $SRC_WHISPER/scripts/sync-ggml.last

echo "Done"

exit 0
