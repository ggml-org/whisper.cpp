#!/bin/sh
#
# libtool assumes that the compiler can handle the -fPIC flag
# This isn't always true (for example, nasm can't handle it)
command=""
while [ $# -gt 0 ]; do
    case "$1" in
        -?PIC)
            # Ignore -fPIC and -DPIC options
            ;;
        -fno-common)
            # Ignore -fPIC and -DPIC options
            ;;
        *)
            command="$command $1"
            ;;
    esac
    shift
done
echo $command
exec $command
