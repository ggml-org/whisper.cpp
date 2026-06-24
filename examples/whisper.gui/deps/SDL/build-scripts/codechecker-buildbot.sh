#!/bin/bash

# This is a script used by some Buildbot build workers to push the project
#  through Clang's static analyzer and prepare the output to be uploaded
#  back to the buildmaster. You might find it useful too.

# Install Clang (you already have it on macOS, apt-get install clang
#  on Ubuntu, etc), install CMake, and pip3 install codechecker.

FINALDIR="$1"

set -x
set -e

cd `dirname "$0"`
cd ..

rm -rf codechecker-buildbot
if [ ! -z "$FINALDIR" ]; then
    rm -rf "$FINALDIR"
fi

mkdir codechecker-buildbot
cd codechecker-buildbot

# We turn off deprecated declarations, because we don't care about these warnings during static analysis.
cmake -Wno-dev -DSDL_STATIC=OFF -DCMAKE_BUILD_TYPE=Debug -DSDL_ASSERTIONS=enabled -DCMAKE_C_FLAGS="-Wno-deprecated-declarations" -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..

# CMake on macOS adds "-arch arm64" or whatever is appropriate, but this confuses CodeChecker, so strip it out.
perl -w -pi -e 's/\-arch\s+.*?\s+//g;' compile_commands.json

rm -rf ../analysis
CodeChecker analyze compile_commands.json -o ./reports

# "parse" returns 2 if there was a static analysis issue to report, but this
#  does not signify an error in the parsing (that would be error code 1). Turn
#  off the abort-on-error flag.
set +e
CodeChecker parse ./reports -e html -o ../analysis
set -e

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

rm -rf codechecker-buildbot

echo "Done. Final output is in '$FINALDIR' ..."

# end of codechecker-buildbot.sh ...

