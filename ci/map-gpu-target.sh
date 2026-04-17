#!/bin/bash
#
# Map a GFX target shorthand to specific GPU architectures for CMake.
#
# Usage:
#   source ci/map-gpu-target.sh <gfx_target>
#
# Arguments:
#   gfx_target - GPU target (gfx1151, gfx1150, gfx110X, gfx120X, or specific)
#
# Outputs (exported):
#   MAPPED_GPU_TARGET - Semicolon-separated list of GPU architectures

gfx_target="$1"

if [ -z "$gfx_target" ]; then
    echo "Usage: source ci/map-gpu-target.sh <gfx_target>"
    return 1 2>/dev/null || exit 1
fi

case "$gfx_target" in
    gfx110X)  MAPPED_GPU_TARGET="gfx1100;gfx1101;gfx1102" ;;
    gfx120X)  MAPPED_GPU_TARGET="gfx1200;gfx1201" ;;
    *)        MAPPED_GPU_TARGET="$gfx_target" ;;
esac

export MAPPED_GPU_TARGET
echo "Mapped GPU target: $gfx_target -> $MAPPED_GPU_TARGET"
