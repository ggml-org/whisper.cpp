#!/bin/sh
#
# Print the current source revision, if available

SDL_ROOT=$(dirname $0)/..
cd $SDL_ROOT

if [ -e ./VERSION.txt ]; then
    cat ./VERSION.txt
    exit 0
fi

major=$(sed -ne 's/^#define SDL_MAJOR_VERSION  *//p' include/SDL_version.h)
minor=$(sed -ne 's/^#define SDL_MINOR_VERSION  *//p' include/SDL_version.h)
micro=$(sed -ne 's/^#define SDL_PATCHLEVEL  *//p' include/SDL_version.h)
version="${major}.${minor}.${micro}"

if [ -x "$(command -v git)" ]; then
    rev="$(git describe --tags --long 2>/dev/null)"
    if [ -n "$rev" ]; then
        # e.g. release-2.24.0-542-g96361fc47
        # or release-2.24.1-5-g36b987dab
        # or prerelease-2.23.2-0-gcb46e1b3f
        echo "$rev"
        exit 0
    fi

    rev="$(git describe --always --tags --long 2>/dev/null)"
    if [ -n "$rev" ]; then
        # Just a truncated sha1, e.g. 96361fc47.
        # Turn it into e.g. 2.25.0-g96361fc47
        echo "${version}-g${rev}"
        exit 0
    fi
fi

if [ -x "$(command -v p4)" ]; then
    rev="$(p4 changes -m1 ./...\#have 2>/dev/null| awk '{print $2}')"
    if [ $? = 0 ]; then
        # e.g. 2.25.0-p7511446
        echo "${version}-p${rev}"
        exit 0
    fi
fi

# best we can do
echo "${version}-no-vcs"
exit 0
