#!/bin/sh

if [ "$SED" = "" ]; then
    if type gsed >/dev/null; then
        SED=gsed
    else
        SED=sed
    fi
fi

find . -type f \
| grep -v \.git                                 \
| while read file; do                           \
    LC_ALL=C $SED -b -i "s/\(.*Copyright.*\)[0-9]\{4\}\( *Sam Lantinga\)/\1`date +%Y`\2/" "$file"; \
done
