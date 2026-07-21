#!/usr/bin/env bash
#
# setup-ggml.sh — clone the qvac-ext-ggml@speech branch into tts-cpp/ggml/
#
# The bundled-ggml dev build path for tts-cpp out of this in-tree subtree.
# Replaces the vcpkg-port consumption when you want a fast iteration loop
# without going through vcpkg installs.
#
# Pinned to the head of the `speech` branch (a tetherto/qvac-ext-ggml fork
# of ggml-org/ggml carrying all QVAC infrastructure patches + the
# Supertonic 2 fused custom op family pre-applied as commits — no
# patches/ directory needed at this layer).
#
# Usage:
#   bash tts-cpp/scripts/setup-ggml.sh
#   cmake -S tts-cpp -B tts-cpp/build -DTTS_CPP_USE_SYSTEM_GGML=OFF
#   cmake --build tts-cpp/build -j
#
# To update to a newer pin: bump GGML_REF below and re-run.  The script
# is idempotent — re-running checks out the right ref into the existing
# tts-cpp/ggml/ clone without re-cloning.

set -euo pipefail

GGML_REPO_URL="https://github.com/tetherto/qvac-ext-ggml.git"
GGML_REF="60a172e48f699bd0a00575ef911feed9473b2187"   # merge of qvac-ext-ggml#8 (speech HEAD)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTS_CPP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
GGML_DIR="${TTS_CPP_DIR}/ggml"

if [ -d "${GGML_DIR}/.git" ]; then
    echo "setup-ggml: existing clone at ${GGML_DIR} — fetching + checking out pin ${GGML_REF:0:10}"
    git -C "${GGML_DIR}" fetch --depth 1 origin "${GGML_REF}"
    git -C "${GGML_DIR}" checkout --detach "${GGML_REF}"
else
    echo "setup-ggml: cloning qvac-ext-ggml @ ${GGML_REF:0:10} into ${GGML_DIR}"
    rm -rf "${GGML_DIR}"
    git clone --depth 1 --no-tags "${GGML_REPO_URL}" "${GGML_DIR}"
    git -C "${GGML_DIR}" fetch --depth 1 origin "${GGML_REF}"
    git -C "${GGML_DIR}" checkout --detach "${GGML_REF}"
fi

echo "setup-ggml: tts-cpp/ggml/ ready at $(git -C "${GGML_DIR}" rev-parse --short HEAD)"
echo "setup-ggml: next: cmake -S tts-cpp -B tts-cpp/build -DTTS_CPP_USE_SYSTEM_GGML=OFF"
