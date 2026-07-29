#!/usr/bin/env bash
#
# Package-consumption regression test for the installed qvac-parakeet
# artifact contract.
#
# Guards the class of breakage that a namespace rename introduces and that
# no in-tree ctest can see, because it only manifests in an *install tree*:
#
#   1. the install tree carries the qvac-parakeet artifacts, and carries
#      NO bare-`parakeet` artifact that would collide with upstream
#      whisper.cpp's own parakeet in a shared prefix (the vcpkg
#      file-ownership conflict this namespace exists to avoid);
#   2. `find_package(qvac-parakeet CONFIG)` + `qvac::parakeet` compiles,
#      links and runs — on a static-only install, so a dead bin/ or
#      BIN_DIR reference in the config would fail here;
#   3. the same via pkg-config, including `--static` so Libs.private
#      (OpenMP, Apple frameworks) is exercised: these PRIVATE deps are
#      not inside the static archive, so an incomplete .pc fails to link.
#
# Usage:
#   scripts/test-package-consumption.sh --ggml-prefix <dir> [options]
#
#   --ggml-prefix <dir>   CMAKE_PREFIX_PATH entry holding the system ggml
#                         install (required)
#   --work-dir <dir>      scratch dir (default: mktemp -d)
#   --coreml              configure the engine with PARAKEET_COREML=ON
#                         (Apple only; exercises the framework entries in
#                         Libs.private)
#   --keep                keep the scratch dir on success
#
# Exits non-zero with a diagnostic on the first failing assertion.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

GGML_PREFIX=""
WORK_DIR=""
COREML=OFF
KEEP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --ggml-prefix) GGML_PREFIX="$2"; shift 2 ;;
        --work-dir)    WORK_DIR="$2";    shift 2 ;;
        --coreml)      COREML=ON;        shift ;;
        --keep)        KEEP=1;           shift ;;
        -h|--help)     sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "error: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

if [ -z "${GGML_PREFIX}" ]; then
    echo "error: --ggml-prefix is required (path to a system ggml install)" >&2
    exit 2
fi
if [ ! -d "${GGML_PREFIX}" ]; then
    echo "error: --ggml-prefix '${GGML_PREFIX}' is not a directory" >&2
    exit 2
fi

if [ -z "${WORK_DIR}" ]; then
    WORK_DIR="$(mktemp -d)"
fi
mkdir -p "${WORK_DIR}"

BUILD_DIR="${WORK_DIR}/build"
PREFIX="${WORK_DIR}/install"

fail() { echo "FAIL: $*" >&2; exit 1; }
ok()   { echo "  ok: $*"; }

# Extra -D flags for the engine configure, whitespace-separated. Exists so
# a developer on a host where CMake does not auto-find OpenMP (e.g. macOS
# + brew libomp) can reproduce the Linux CI configuration, where OpenMP is
# found and therefore appears in the exported link interface and in
# Libs.private:
#   PARAKEET_PKGTEST_CMAKE_ARGS="-DPARAKEET_OPENMP=ON \
#     -DPARAKEET_OPENMP_USER_OVERRIDE=ON -DOpenMP_ROOT=/opt/homebrew/opt/libomp"
read -r -a _pk_extra_args <<< "${PARAKEET_PKGTEST_CMAKE_ARGS:-}"

echo "==> configuring + installing the engine (static, system ggml, COREML=${COREML})"
cmake -S "${ENGINE_DIR}" -B "${BUILD_DIR}" \
    ${_pk_extra_args[@]+"${_pk_extra_args[@]}"} \
    -DCMAKE_BUILD_TYPE=Release \
    -DPARAKEET_USE_SYSTEM_GGML=ON \
    -DPARAKEET_BUILD_LIBRARY=ON \
    -DPARAKEET_BUILD_EXECUTABLES=OFF \
    -DPARAKEET_BUILD_TESTS=OFF \
    -DPARAKEET_BUILD_EXAMPLES=OFF \
    -DPARAKEET_INSTALL=ON \
    -DPARAKEET_COREML="${COREML}" \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_PREFIX_PATH="${GGML_PREFIX}" \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" > "${WORK_DIR}/configure.log" 2>&1 \
    || { tail -30 "${WORK_DIR}/configure.log"; fail "engine configure"; }
cmake --build "${BUILD_DIR}" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" \
    > "${WORK_DIR}/build.log" 2>&1 \
    || { tail -30 "${WORK_DIR}/build.log"; fail "engine build"; }
cmake --install "${BUILD_DIR}" > "${WORK_DIR}/install.log" 2>&1 \
    || { tail -30 "${WORK_DIR}/install.log"; fail "engine install"; }

# ----------------------------------------------------------------------
echo "==> 1/3 install-tree artifact audit"

# Expected artifacts. The library filename is platform-dependent
# (libqvac-parakeet.a / .dylib / .so, qvac-parakeet.lib), so glob for any
# shape rather than hardcoding one per runner.
shopt -s nullglob
_pk_libs=( "${PREFIX}"/lib/libqvac-parakeet.* "${PREFIX}"/lib/qvac-parakeet.lib )
shopt -u nullglob
[ "${#_pk_libs[@]}" -gt 0 ] || fail "no qvac-parakeet library found under ${PREFIX}/lib"
ok "qvac-parakeet library present (${_pk_libs[0]##*/})"

for expected in \
    "lib/cmake/qvac-parakeet/qvac-parakeet-config.cmake" \
    "lib/cmake/qvac-parakeet/qvac-parakeet-config-version.cmake" \
    "lib/cmake/qvac-parakeet/qvac-parakeet-targets.cmake" \
    "lib/pkgconfig/qvac-parakeet.pc" \
    "include/parakeet/parakeet.h" ; do
    [ -f "${PREFIX}/${expected}" ] || fail "missing expected artifact: ${expected}"
done
ok "cmake package, pkg-config file and headers present"

# The collision guard: none of upstream whisper.cpp's parakeet artifact
# paths may be produced by this engine, or the two ports cannot share a
# vcpkg prefix. Keep this list in sync with upstream's install rules
# (third_party/whisper.cpp/CMakeLists.txt: lib/libparakeet.*,
# lib/cmake/parakeet/, parakeet.pc, include/parakeet.h).
while IFS= read -r stray; do
    [ -n "${stray}" ] && fail "install tree contains a bare-parakeet artifact that collides with upstream whisper.cpp: ${stray#${PREFIX}/}"
done < <(
    { ls -d "${PREFIX}"/lib/libparakeet.* \
            "${PREFIX}"/lib/parakeet.lib \
            "${PREFIX}"/lib/cmake/parakeet \
            "${PREFIX}"/lib/pkgconfig/parakeet.pc \
            "${PREFIX}"/include/parakeet.h \
            "${PREFIX}"/share/parakeet-cpp 2>/dev/null || true; }
)
ok "no bare-parakeet artifacts (no collision with upstream whisper.cpp)"

# ----------------------------------------------------------------------
echo "==> 2/3 CMake consumer: find_package(qvac-parakeet) + qvac::parakeet"
CM_BUILD="${WORK_DIR}/consumer-cmake"
# The extra args are forwarded to the consumer too: when the engine build
# links OpenMP, the installed package config does find_dependency(OpenMP)
# (the PRIVATE link is exported as $<LINK_ONLY:OpenMP::OpenMP_CXX> for a
# static build), so any toolchain hint needed to locate OpenMP -- e.g.
# -DOpenMP_ROOT on macOS + brew libomp -- has to reach the consumer's
# configure as well. Engine-only -DPARAKEET_* flags are simply unused
# there, which CMake reports as a non-fatal warning.
cmake -S "${ENGINE_DIR}/test/consumer" -B "${CM_BUILD}" \
    ${_pk_extra_args[@]+"${_pk_extra_args[@]}"} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${PREFIX};${GGML_PREFIX}" \
    > "${WORK_DIR}/consumer-cmake-configure.log" 2>&1 \
    || { tail -30 "${WORK_DIR}/consumer-cmake-configure.log"; fail "CMake consumer configure"; }
cmake --build "${CM_BUILD}" > "${WORK_DIR}/consumer-cmake-build.log" 2>&1 \
    || { tail -30 "${WORK_DIR}/consumer-cmake-build.log"; fail "CMake consumer build/link"; }
ok "configured and linked"

# Shared-ggml installs need the loader to find libggml at run time.
export LD_LIBRARY_PATH="${GGML_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="${GGML_PREFIX}/lib:${DYLD_LIBRARY_PATH:-}"
"${CM_BUILD}/consumer" > "${WORK_DIR}/consumer-cmake-run.log" 2>&1 \
    || { cat "${WORK_DIR}/consumer-cmake-run.log"; fail "CMake consumer run"; }
grep -q "qvac-parakeet consumer OK" "${WORK_DIR}/consumer-cmake-run.log" \
    || fail "CMake consumer produced unexpected output"
ok "ran successfully"

# ----------------------------------------------------------------------
echo "==> 3/3 pkg-config consumer (static: exercises Libs.private)"
if ! command -v pkg-config >/dev/null 2>&1; then
    echo "  skip: pkg-config not installed on this runner"
else
    export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${GGML_PREFIX}/lib/pkgconfig:${GGML_PREFIX}/share/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
    pkg-config --exists --print-errors qvac-parakeet || fail "pkg-config cannot resolve qvac-parakeet"
    ok "pkg-config resolves the module"

    # No hand-added include/lib paths: the .pc must be self-sufficient
    # (it advertises ggml's includedir and libdir itself).
    PC_CFLAGS="$(pkg-config --cflags qvac-parakeet)"
    PC_LIBS="$(pkg-config --static --libs qvac-parakeet)"
    echo "  cflags: ${PC_CFLAGS}"
    echo "  libs:   ${PC_LIBS}"

    # shellcheck disable=SC2086
    ${CXX:-c++} -std=c++17 "${ENGINE_DIR}/test/consumer/main.cpp" \
        ${PC_CFLAGS} ${PC_LIBS} \
        -o "${WORK_DIR}/consumer-pc" > "${WORK_DIR}/consumer-pc-build.log" 2>&1 \
        || { tail -30 "${WORK_DIR}/consumer-pc-build.log"; fail "pkg-config consumer compile/link"; }
    ok "compiled and linked with pkg-config flags only"

    "${WORK_DIR}/consumer-pc" > "${WORK_DIR}/consumer-pc-run.log" 2>&1 \
        || { cat "${WORK_DIR}/consumer-pc-run.log"; fail "pkg-config consumer run"; }
    grep -q "qvac-parakeet consumer OK" "${WORK_DIR}/consumer-pc-run.log" \
        || fail "pkg-config consumer produced unexpected output"
    ok "ran successfully"
fi

echo
echo "PASS: qvac-parakeet package-consumption contract holds"
if [ "${KEEP}" -eq 0 ] && [ -z "${WORK_DIR_PRESET:-}" ]; then
    rm -rf "${WORK_DIR}"
else
    echo "(scratch kept at ${WORK_DIR})"
fi
