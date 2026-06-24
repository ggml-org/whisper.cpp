#!/bin/sh

set -e

echo "Generating build information using autoconf"
echo "This may take a while ..."

srcdir=`dirname $0`
test -z "$srcdir" && srcdir=.
cd "$srcdir"

# Regenerate configuration files
cat acinclude/* >aclocal.m4

"${AUTOCONF:-autoconf}"
rm aclocal.m4
rm -rf autom4te.cache

(cd test; sh autogen.sh)

echo "Now you are ready to run ./configure"
