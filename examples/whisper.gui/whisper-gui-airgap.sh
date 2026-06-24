#!/usr/bin/env bash
#
# Two-stage air-gapped bundler/installer for the Linux whisper-gui.
#
#   On a CONNECTED machine, from the repo root:
#       examples/whisper.gui/whisper-gui-airgap.sh stage [BUNDLE_DIR]
#   then copy BUNDLE_DIR to the OFFLINE machine and run:
#       <BUNDLE_DIR>/whisper-gui-airgap.sh install [BUNDLE_DIR]
#
# Stage downloads: the Python wheels (sherpa-onnx, numpy), the whisper +
# diarization models, the OS dev packages (best-effort, dnf/apt), and a copy of
# the repo (which already vendors SDL2 + Dear ImGui). Install builds whisper-gui,
# sets up an isolated venv for the diarization helper, and writes a launcher.
#
# Python wheels are matched to the STAGING machine's Python; stage on a box whose
# Python minor version matches the target (or use the same distro release).
#
set -euo pipefail

MODE="${1:-}"
BUNDLE="${2:-$PWD/whisper-gui-bundle}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
mkdir -p "$BUNDLE" 2>/dev/null || true
BUNDLE="$(cd "$BUNDLE" && pwd)"

GH=https://github.com/k2-fsa/sherpa-onnx/releases/download
EMB="$GH/speaker-recongition-models/nemo_en_titanet_small.onnx"
SEG="$GH/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
# whisper model to bundle (override: WHISPER_MODEL_NAME=large-v3 ./... stage)
WHISPER_MODEL_NAME="${WHISPER_MODEL_NAME:-large-v3-turbo}"
WHISPER_MODEL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${WHISPER_MODEL_NAME}.bin"

PKGS_DNF="gcc-c++ cmake mesa-libGL-devel libX11-devel libXext-devel libxkbcommon-devel wayland-devel python3 python3-pip"
PKGS_APT="g++ cmake libgl1-mesa-dev libx11-dev libxext-dev libxkbcommon-dev libwayland-dev python3 python3-pip python3-venv"

say()  { printf '\033[36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!!  %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mXX  %s\033[0m\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
pm()   { if have dnf; then echo dnf; elif have apt-get; then echo apt; else echo none; fi; }

# --------------------------------------------------------------------- stage --
do_stage() {
    have curl   || die "curl is required for staging"
    have python3 || die "python3 is required for staging"
    mkdir -p "$BUNDLE"/{wheels,models,pkgs,repo}
    say "staging into $BUNDLE"

    say "downloading Python wheels (sherpa-onnx, numpy)"
    python3 -m pip download sherpa-onnx numpy -d "$BUNDLE/wheels" \
        || die "pip download failed"

    say "downloading models (~190 MB)"
    [ -f "$BUNDLE/models/ggml-${WHISPER_MODEL_NAME}.bin" ] || curl -fL -o "$BUNDLE/models/ggml-${WHISPER_MODEL_NAME}.bin" "$WHISPER_MODEL"
    [ -f "$BUNDLE/models/nemo_en_titanet_small.onnx" ] || curl -fL -o "$BUNDLE/models/nemo_en_titanet_small.onnx" "$EMB"
    [ -f "$BUNDLE/models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" ] || \
        curl -fL -o "$BUNDLE/models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" "$SEG"

    # OS dev packages - best effort, distro specific (the fragile part)
    if [ "${SKIP_PKGS:-0}" = 1 ]; then
        warn "SKIP_PKGS=1 - not staging system packages (target must already have the build deps)"
    else
        case "$(pm)" in
            dnf)
                say "staging RPMs via 'dnf download --resolve' (incl. gcc-c++ + cmake)"
                # 'dnf download' comes from dnf-plugins-core (dnf4) / dnf5-plugins (dnf5)
                if ! dnf download --help >/dev/null 2>&1; then
                    sudo dnf install -y dnf-plugins-core >/dev/null 2>&1 \
                        || sudo dnf install -y dnf5-plugins >/dev/null 2>&1 || true
                fi
                dnf download --resolve --alldeps --destdir "$BUNDLE/pkgs" $PKGS_DNF \
                    || warn "dnf download failed - run 'sudo dnf install dnf-plugins-core' (or 'dnf5-plugins'), then re-stage; or SKIP_PKGS=1." ;;
            apt)
                say "staging .debs via apt-get (incl. g++ + cmake)"
                ( cd "$BUNDLE/pkgs" && apt-get download $(apt-cache depends --recurse --no-recommends --no-suggests \
                    --no-conflicts --no-breaks --no-replaces --no-enhances -i $PKGS_APT 2>/dev/null | grep '^\w' | sort -u) ) \
                    || warn "apt download incomplete. Stage .debs manually into $BUNDLE/pkgs, or set SKIP_PKGS=1." ;;
            *)  warn "no dnf/apt found - skipping system packages" ;;
        esac
        # tell the user NOW (not after they carry the bundle) whether it worked
        n=$(ls "$BUNDLE"/pkgs/*.rpm "$BUNDLE"/pkgs/*.deb 2>/dev/null | wc -l)
        if [ "$n" -eq 0 ]; then
            warn "no OS packages were staged - the offline target will need gcc-c++/g++ + cmake by other means."
        else
            say "staged $n OS package file(s) (compiler + build deps) into pkgs/"
        fi
    fi

    say "copying repo into bundle (excluding build dirs / .git)"
    if have rsync; then
        rsync -a --exclude build --exclude 'build-*' --exclude .git \
              --exclude "$(basename "$BUNDLE")" "$REPO_ROOT"/ "$BUNDLE/repo"/
    else
        ( cd "$REPO_ROOT" && tar --exclude=build --exclude='build-*' --exclude=.git \
              --exclude="$(basename "$BUNDLE")" -cf - . ) | tar -xf - -C "$BUNDLE/repo"
    fi

    cp "${BASH_SOURCE[0]}" "$BUNDLE/whisper-gui-airgap.sh"
    chmod +x "$BUNDLE/whisper-gui-airgap.sh"

    say "STAGE COMPLETE -> $BUNDLE"
    echo "Copy that folder to the offline machine, then run:"
    echo "    $BUNDLE/whisper-gui-airgap.sh install $BUNDLE"
}

# ------------------------------------------------------------------- install --
do_install() {
    [ -d "$BUNDLE/repo" ] || die "bundle '$BUNDLE' has no repo/ - run 'stage' first"
    local repo="$BUNDLE/repo"

    # 1. OS dev packages (best effort)
    if ls "$BUNDLE"/pkgs/*.rpm  >/dev/null 2>&1; then
        say "installing RPMs"; sudo dnf install -y --disablerepo='*' "$BUNDLE"/pkgs/*.rpm || warn "rpm install had issues"
    elif ls "$BUNDLE"/pkgs/*.deb >/dev/null 2>&1; then
        say "installing .debs"; sudo apt-get install -y "$BUNDLE"/pkgs/*.deb || { sudo dpkg -i "$BUNDLE"/pkgs/*.deb || true; }
    else
        warn "no staged packages - assuming the build deps are already installed"
    fi
    have cmake || die "cmake not found - install it (or stage its package) and re-run"
    if ! { have gcc || have g++ || have cc || have clang; }; then
        die "no C++ compiler found. Install gcc-c++ / g++ (or stage it on a connected machine via 'stage'), then re-run."
    fi

    # 2. isolated venv for the diarization helper (avoids PEP 668 / system pollution)
    say "creating venv + installing sherpa-onnx, numpy (offline)"
    python3 -m venv "$repo/.venv" || die "python3 -m venv failed - install python3-venv"
    local vpy="$repo/.venv/bin/python"
    if "$vpy" -m pip install --no-index --find-links "$BUNDLE/wheels" sherpa-onnx numpy; then
        :  # installed from the bundle, fully offline
    else
        local tag; tag="$("$vpy" -c 'import sys;print("cp%d%d"%sys.version_info[:2])')"
        warn "no bundled wheel matches this machine's Python ($tag). Bundle has: $(ls "$BUNDLE/wheels" 2>/dev/null | tr '\n' ' ')"
        warn "(binary wheels are Python-version-specific - stage on a machine whose Python matches the target)"
        if "$vpy" -m pip install sherpa-onnx numpy; then   # online fallback if a network is up
            warn "installed sherpa-onnx/numpy from the internet - this run was NOT fully offline."
        else
            die "could not install sherpa-onnx/numpy offline or online. For a true air-gapped build, re-stage on a CONNECTED machine running the SAME Python version as this one ($tag)."
        fi
    fi
    "$vpy" -c "import sherpa_onnx, numpy; print('python deps OK', sherpa_onnx.__version__)"

    # 3. models into repo/models (+ extract segmentation)
    mkdir -p "$repo/models"
    cp -f "$BUNDLE"/models/ggml-*.bin "$BUNDLE/models/nemo_en_titanet_small.onnx" "$repo/models/"
    [ -f "$repo/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx" ] || \
        tar -xf "$BUNDLE/models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" -C "$repo/models"

    # 4. build whisper-gui (vendored SDL2 + ImGui, no network)
    say "building whisper-gui (compiles SDL2 from source the first time)"
    cmake -S "$repo" -B "$repo/build" -DWHISPER_GUI=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$repo/build" --target whisper-gui -j"$(nproc)"

    # 5. launcher that puts the venv's python on PATH (so the GUI's "python3"
    #    finds sherpa-onnx) and runs from the repo root (so models/ resolve)
    cat > "$repo/run-whisper-gui.sh" <<'LAUNCH'
#!/usr/bin/env bash
cd "$(dirname "$(readlink -f "$0")")"
export PATH="$PWD/.venv/bin:$PATH"
exec ./build/bin/whisper-gui "$@"
LAUNCH
    chmod +x "$repo/run-whisper-gui.sh"

    say "INSTALL COMPLETE."
    echo "Run it with:"
    echo "    $repo/run-whisper-gui.sh"
    echo "Then: Browse a .wav -> Diarize = Accurate (sherpa-onnx) -> Speakers = N -> Transcribe."
}

case "$MODE" in
    stage)   do_stage ;;
    install) do_install ;;
    *) die "usage: $0 {stage|install} [BUNDLE_DIR]" ;;
esac
