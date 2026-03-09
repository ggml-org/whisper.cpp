#!/bin/bash
#
# Resolve the latest ROCm nightly tarball URL for a given GPU target and platform.
#
# Usage:
#   source ci/resolve-rocm-version.sh <platform> <gfx_target> <rocm_version>
#
# Arguments:
#   platform      - "linux" or "windows"
#   gfx_target    - GPU target (gfx1151, gfx1150, gfx110X, gfx120X)
#   rocm_version  - Specific version (e.g. 7.11.0a20251205) or "latest"
#
# Outputs (exported):
#   ROCM_RESOLVED_VERSION - The resolved version string
#   ROCM_TARBALL_URL      - The full S3 URL to download

platform="$1"
gfx_target="$2"
rocm_version="$3"

if [ -z "$platform" ] || [ -z "$gfx_target" ] || [ -z "$rocm_version" ]; then
    echo "Usage: source ci/resolve-rocm-version.sh <platform> <gfx_target> <rocm_version>"
    return 1 2>/dev/null || exit 1
fi

# Map GPU target to S3 naming convention
# Group targets (gfx110X, gfx120X) use "-all" suffix; individual targets have no suffix
s3_target="$gfx_target"
if [ "$gfx_target" = "gfx110X" ] || [ "$gfx_target" = "gfx120X" ]; then
    s3_target="${gfx_target}-all"
fi

dist_prefix="therock-dist-${platform}-${s3_target}"

if [ "$rocm_version" = "latest" ]; then
    echo "Auto-detecting latest ROCm version for ${platform}/${gfx_target}..."
    s3_response=$(curl -s "https://therock-nightly-tarball.s3.amazonaws.com/?prefix=${dist_prefix}-7")

    files=$(echo "$s3_response" | tr '<' '\n' | sed -n 's/^Key>\([^<]*\)/\1/p' | grep "${dist_prefix}-")

    latest_file=""
    latest_major=0
    latest_minor=0
    latest_patch=0
    latest_rc=0
    latest_is_alpha=false

    while IFS= read -r file; do
        if [[ "$file" =~ ${dist_prefix}-.*?([0-9]+\.[0-9]+\.[0-9]+(a|rc)[0-9]+)\.tar\.gz ]]; then
            version="${BASH_REMATCH[1]}"
            major=$(echo "$version" | cut -d. -f1)
            minor=$(echo "$version" | cut -d. -f2)
            patch=$(echo "$version" | cut -d. -f3 | sed 's/\(a\|rc\).*//')
            rc=$(echo "$version" | sed 's/.*\(a\|rc\)//')
            is_alpha=false
            if [[ "$version" =~ a ]]; then is_alpha=true; fi

            is_newer=false
            if [ "$major" -gt "$latest_major" ]; then is_newer=true;
            elif [ "$major" -eq "$latest_major" ] && [ "$minor" -gt "$latest_minor" ]; then is_newer=true;
            elif [ "$major" -eq "$latest_major" ] && [ "$minor" -eq "$latest_minor" ] && [ "$patch" -gt "$latest_patch" ]; then is_newer=true;
            elif [ "$major" -eq "$latest_major" ] && [ "$minor" -eq "$latest_minor" ] && [ "$patch" -eq "$latest_patch" ]; then
                if [ "$is_alpha" = true ] && [ "$latest_is_alpha" = false ]; then is_newer=true;
                elif [ "$is_alpha" = "$latest_is_alpha" ] && [ "$rc" -gt "$latest_rc" ]; then is_newer=true;
                fi
            fi

            if [ "$is_newer" = true ]; then
                latest_file="$file"
                latest_major="$major"
                latest_minor="$minor"
                latest_patch="$patch"
                latest_rc="$rc"
                latest_is_alpha="$is_alpha"
            fi
        fi
    done <<< "$files"

    echo "Found latest file: $latest_file"

    if [[ "$latest_file" =~ ${dist_prefix}-.*?([0-9]+\.[0-9]+\.[0-9]+(a|rc)[0-9]+)\.tar\.gz ]]; then
        rocm_version="${BASH_REMATCH[1]}"
        echo "Detected latest ROCm version: $rocm_version"
    else
        echo "Failed to extract ROCm version from latest file: $latest_file"
        return 1 2>/dev/null || exit 1
    fi

    export ROCM_TARBALL_URL="https://therock-nightly-tarball.s3.amazonaws.com/$latest_file"
else
    export ROCM_TARBALL_URL="https://therock-nightly-tarball.s3.amazonaws.com/${dist_prefix}-${rocm_version}.tar.gz"
fi

export ROCM_RESOLVED_VERSION="$rocm_version"
echo "ROCm URL: $ROCM_TARBALL_URL"
