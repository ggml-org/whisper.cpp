#!/bin/bash
#
# Resolve the ROCm tarball URL for a given platform and version.
#
# Uses AMD's official repo tarball distribution:
#   https://repo.amd.com/rocm/tarball/therock-dist-{platform}-{gfx_target}-{version}.tar.gz
#
# Usage:
#   source ci/resolve-rocm-version.sh <platform> <gfx_target> <rocm_version>
#
# Arguments:
#   platform      - "linux" or "windows"
#   gfx_target    - GPU target (defaults to gfx1151 if not specified or is a group target)
#   rocm_version  - Specific version (e.g. 7.12.0, 7.2.1) - required, no "latest" auto-detection
#
# Outputs (exported):
#   ROCM_RESOLVED_VERSION - The resolved version string
#   ROCM_TARBALL_URL      - The full URL to download

platform="$1"
gfx_target="$2"
rocm_version="$3"

if [ -z "$platform" ] || [ -z "$gfx_target" ] || [ -z "$rocm_version" ]; then
    echo "Usage: source ci/resolve-rocm-version.sh <platform> <gfx_target> <rocm_version>"
    return 1 2>/dev/null || exit 1
fi

# Validate that a specific version was provided (no "latest" auto-detection)
if [ "$rocm_version" = "latest" ]; then
    echo "ERROR: 'latest' auto-detection is not supported."
    echo "Please specify a concrete ROCm version (e.g., 7.12.0, 7.2.1)."
    echo "Available versions: https://repo.amd.com/rocm/tarball/"
    return 1 2>/dev/null || exit 1
fi

# Validate version format (should be X.Y.Z or X.Y.ZaNNNNNNNN pattern)
if ! echo "$rocm_version" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+'; then
    echo "ERROR: Invalid ROCm version format: '$rocm_version'"
    echo "Expected format: X.Y.Z (e.g., 7.12.0) or X.Y.ZaNNNNNNNN (e.g., 7.11.0a20251205)"
    return 1 2>/dev/null || exit 1
fi

# Map our GFX target shorthand to the exact tarball name AMD publishes.
# Each GPU family has its own tarball with the right prebuilt kernel libraries.
# Using the wrong tarball (e.g. gfx1151 for gfx110X) gives you gfx1151 rocBLAS/hipBLASLT
# kernels in a gfx1100 package — builds succeed but fail silently on user hardware.
case "$gfx_target" in
    gfx110X)  tarball_target="gfx110X" ;;   # RDNA3 dGPU: RX 7900/7800/7700/7600
    gfx120X)  tarball_target="gfx120X" ;;   # RDNA4 dGPU: RX 9070/9060
    gfx1150)  tarball_target="gfx1150" ;;   # RDNA3.5 APU: Strix Point
    gfx1151)  tarball_target="gfx1151" ;;   # RDNA3.5 APU: Strix Halo
    gfx1100)  tarball_target="gfx1100" ;;   # RDNA3 dGPU specific
    gfx1101)  tarball_target="gfx1101" ;;
    gfx1200)  tarball_target="gfx1200" ;;
    gfx1201)  tarball_target="gfx1201" ;;
    *)        tarball_target="$gfx_target" ;;
esac

# Construct the AMD official repo URL
ROCM_TARBALL_URL="https://repo.amd.com/rocm/tarball/therock-dist-${platform}-${tarball_target}-${rocm_version}.tar.gz"

export ROCM_RESOLVED_VERSION="$rocm_version"
echo "ROCm version: $ROCM_RESOLVED_VERSION"
echo "ROCm URL: $ROCM_TARBALL_URL"
