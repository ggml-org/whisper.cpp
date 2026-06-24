#!/bin/bash

# This is a script used by some Buildbot buildslaves to push the project
#  through Clang's static analyzer and prepare the output to be uploaded
#  back to the buildmaster. You might find it useful too.

# Install Clang (you already have it on Mac OS X, apt-get install clang
#  on Ubuntu, etc), and make sure scan-build is in your $PATH.

FINALDIR="$1"

set -x
set -e

cd `dirname "$0"`
cd ..

rm -rf checker-buildbot analysis
if [ ! -z "$FINALDIR" ]; then
    rm -rf "$FINALDIR"
fi

mkdir checker-buildbot
cd checker-buildbot

# We turn off deprecated declarations, because we don't care about these warnings during static analysis.
# The -Wno-liblto is new since our checker-279 upgrade, I think; checker otherwise warns "libLTO.dylib relative to clang installed dir not found"

# You might want to do this for CMake-backed builds instead...
scan-build -o analysis cmake -G Ninja -Wno-dev -DSDL_STATIC=OFF -DCMAKE_BUILD_TYPE=Debug -DSDL_ASSERTIONS=enabled -DCMAKE_C_FLAGS="-Wno-deprecated-declarations" -DCMAKE_SHARED_LINKER_FLAGS="-Wno-liblto" ..

# ...or run configure without the scan-build wrapper...
#CC="$CHECKERDIR/libexec/ccc-analyzer" CFLAGS="-O0 -Wno-deprecated-declarations" LDFLAGS="-Wno-liblto" ../configure --enable-assertions=enabled

rm -rf analysis
scan-build -o analysis ninja

if [ `ls -A analysis |wc -l` == 0 ] ; then
    mkdir analysis/zarro
    echo '<html><head><title>Zarro boogs</title></head><body>Static analysis: no issues to report.</body></html>' >analysis/zarro/index.html
fi

mv analysis/* ../analysis
rmdir analysis   # Make sure this is empty.
cd ..
chmod -R a+r analysis
chmod -R go-w analysis
find analysis -type d -exec chmod a+x {} \;
if [ -x /usr/bin/xattr ]; then find analysis -exec /usr/bin/xattr -d com.apple.quarantine {} \; 2>/dev/null ; fi

if [ ! -z "$FINALDIR" ]; then
    mv analysis "$FINALDIR"
else
    FINALDIR=analysis
fi

rm -rf checker-buildbot

echo "Done. Final output is in '$FINALDIR' ..."

# end of checker-buildbot.sh ...

