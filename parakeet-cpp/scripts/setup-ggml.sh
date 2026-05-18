#!/usr/bin/env bash
# Clone qvac-ext-ggml into ./ggml on the branch this repo is pinned against.
# Idempotent: safe to re-run.
#
# Update GGML_URL / GGML_BRANCH here whenever the pin is bumped; this file
# is the single source of truth for which ggml fork parakeet.cpp builds
# against.
#
# qvac-ext-ggml's `speech` branch carries the equivalents of the patches
# that used to live under patches/ggml-*.patch (backend-reg filename
# prefix, opencl non-Adreno support, opencl program binary cache). The
# script therefore does not apply local patches anymore.

set -euo pipefail

GGML_URL="https://github.com/tetherto/qvac-ext-ggml.git"
GGML_BRANCH="speech"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "parakeet.cpp: setting up ggml from ${GGML_URL} (branch: ${GGML_BRANCH})"

if [ -e ggml ] && [ ! -d ggml/.git ]; then
    if [ -L ggml ]; then
        echo "  -> ggml is a symlink to '$(readlink ggml)'; leaving it alone"
        echo "     (delete the symlink and re-run this script to clone fresh)"
        exit 0
    fi
    echo "  ERROR: ./ggml exists but is not a git checkout." >&2
    echo "         Remove it and re-run this script." >&2
    exit 1
fi

if [ ! -d ggml/.git ]; then
    echo "  -> cloning ${GGML_URL} (branch ${GGML_BRANCH})"
    git clone --branch "$GGML_BRANCH" "$GGML_URL" ggml
fi

cd ggml

# Make sure the local checkout actually has the requested branch fetched
# (e.g. when the user previously cloned with a narrow refspec).
if ! git rev-parse --verify --quiet "refs/heads/${GGML_BRANCH}" >/dev/null; then
    echo "  -> fetching ${GGML_BRANCH}"
    git fetch origin "${GGML_BRANCH}:${GGML_BRANCH}"
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
if [ "$CURRENT_BRANCH" != "$GGML_BRANCH" ]; then
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "  -> resetting ggml worktree to pristine before switching branches"
        git checkout -- .
    fi
    echo "  -> checking out ${GGML_BRANCH}"
    git checkout "$GGML_BRANCH"
fi

echo "  -> ok, on ${GGML_BRANCH} at $(git rev-parse --short=8 HEAD)"

echo
echo "ggml is ready. Next:"
echo "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release"
echo "    cmake --build build -j\$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
