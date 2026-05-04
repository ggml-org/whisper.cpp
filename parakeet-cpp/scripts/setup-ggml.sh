#!/usr/bin/env bash
# Clone ggml into ./ggml at the commit this repo is pinned against, and
# apply every patch under patches/ in lexicographic order.  Idempotent:
# safe to re-run.
#
# Update GGML_COMMIT here whenever the pin is bumped; this file is the
# single source of truth for which upstream ggml parakeet.cpp builds
# against.
#
# Patches we ship today:
#   patches/ggml-backend-reg-filename-prefix.patch
#       Teaches ggml_backend_load_best() to honour a compile-time
#       GGML_BACKEND_DL_PROJECT_PREFIX macro so renaming the bundled
#       backend .so/.dll files (PARAKEET_GGML_LIB_PREFIX=ON, the default,
#       emits libspeech-ggml-*.so) does not break runtime backend
#       discovery under GGML_BACKEND_DL=ON. No-op when the macro is
#       undefined.
#   patches/ggml-opencl-allow-non-adreno.patch
#       Lets the ggml-opencl backend run on non-Adreno/Intel GPUs
#       (NVIDIA, AMD, Apple) so the build can be parity-tested on
#       commodity desktop hardware. Real Adreno deployments build with
#       the patch applied as a no-op (Adreno path is unchanged).
#   patches/ggml-opencl-program-binary-cache.patch
#       Persistent OpenCL kernel binary cache via clCreateProgramWithBinary +
#       CL_PROGRAM_BINARIES. Removes seconds of cold-start shader compile on
#       every Adreno / Mesa / Mali / iGPU launch by serialising compiled
#       kernels under $GGML_OPENCL_CACHE_DIR (or XDG/HOME fallback).
#       See patches/README.md for the full rationale.

set -euo pipefail

GGML_COMMIT="58c38058"
GGML_URL="https://github.com/ggml-org/ggml.git"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "parakeet.cpp: setting up ggml at pinned commit ${GGML_COMMIT}"

if [ ! -d ggml/.git ]; then
    echo "  -> cloning ${GGML_URL}"
    git clone "$GGML_URL" ggml
fi

# Find every patch under patches/ matching ggml-*.patch, sorted.
shopt -s nullglob
PATCHES=( "$REPO_ROOT"/patches/ggml-*.patch )
shopt -u nullglob

cd ggml

CURRENT="$(git rev-parse --short=8 HEAD 2>/dev/null || echo '')"
NEED_CHECKOUT="0"
if [ "$CURRENT" != "$GGML_COMMIT" ]; then
    NEED_CHECKOUT="1"
fi

if [ "$NEED_CHECKOUT" = "1" ]; then
    git checkout -- . 2>/dev/null || true
    git checkout "$GGML_COMMIT"
    echo "  -> ok, at $(git rev-parse --short=8 HEAD)"
fi

# Apply patches.  We always reset to the pinned commit before applying so
# this is fully idempotent: re-running the script never stacks patches on
# top of patches.  We bail loudly on a real failure (CRLF in working
# tree, conflict, ...) instead of silently linking against unpatched ggml.
if [ ${#PATCHES[@]} -gt 0 ]; then
    if [ "$NEED_CHECKOUT" = "0" ]; then
        # Same commit as last run, but patches may already be applied;
        # reset to pristine before re-applying.
        if ! git diff --quiet || ! git diff --cached --quiet; then
            echo "  -> resetting ggml worktree to pristine ${GGML_COMMIT}"
            git checkout -- .
        fi
    fi
    for patch in "${PATCHES[@]}"; do
        name="$(basename "$patch")"
        # Detect whether the patch has already been applied (idempotent
        # re-run of the script). `git apply --reverse --check` succeeds
        # iff every hunk reverses cleanly, which only happens when the
        # patch is currently applied to the working tree.
        if git apply --reverse --check "$patch" 2>/dev/null; then
            echo "  -> $name: already applied, skipping"
            continue
        fi

        # Strip CR line endings from the patch on the fly. Windows checkouts
        # with `core.autocrlf=true` (git's default on Windows) leave the
        # patch as CRLF in the working tree even though it is LF in the
        # index, and `git apply` then refuses with a context-mismatch
        # error.  This converts on read instead of mutating the file.
        sanitized="$(mktemp)"
        # shellcheck disable=SC2064
        trap "rm -f '$sanitized'" EXIT
        tr -d '\r' < "$patch" > "$sanitized"

        echo "  -> applying $name"
        if ! git apply --check "$sanitized" 2>/tmp/setup-ggml-apply.err; then
            echo "    ERROR: patch '$name' does not apply against ggml@${GGML_COMMIT}." >&2
            sed 's/^/    /' /tmp/setup-ggml-apply.err >&2
            echo "    Aborting so the build does not silently link unpatched ggml." >&2
            rm -f /tmp/setup-ggml-apply.err
            exit 1
        fi
        rm -f /tmp/setup-ggml-apply.err
        git apply "$sanitized"
    done
fi

echo
echo "ggml is ready. Next:"
echo "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release"
echo "    cmake --build build -j\$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
