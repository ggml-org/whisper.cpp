#!/usr/bin/env bash
# Clone ggml into ./ggml at the commit this repo is pinned against, and
# apply every patch under patches/ in lexicographic order.  Idempotent:
# safe to re-run; local edits under ./ggml are discarded before patches
# are re-applied.
#
# Update GGML_COMMIT here whenever the pin is bumped; this file is the
# single source of truth for which upstream ggml chatterbox.cpp builds
# against.  See patches/README.md for what each patch does.

set -euo pipefail

GGML_COMMIT="58c38058"
GGML_URL="https://github.com/ggml-org/ggml.git"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "chatterbox.cpp: setting up ggml at pinned commit ${GGML_COMMIT}"

if [ ! -d ggml/.git ]; then
    echo "  → cloning ${GGML_URL}"
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
    echo "  → ok, at $(git rev-parse --short=8 HEAD)"
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
            echo "  → resetting ggml worktree to pristine ${GGML_COMMIT}"
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
            echo "  → $name: already applied, skipping"
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

        echo "  → applying $name"
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
echo "ggml is ready.  Next:"
echo "  Metal:   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON"
echo "  OpenCL:  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENCL=ON"
echo "  Vulkan:  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON"
echo "  cmake --build build -j\$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
