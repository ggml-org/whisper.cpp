#!/usr/bin/env bash
# install.sh — Install dependencies and set up whisper.linux
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_NAME="${1:-base}"
MODEL_FILE="ggml-${MODEL_NAME}.bin"
CONFIG_DIR="$HOME/.config/whisper-linux"

echo "=== whisper.linux installer ==="
echo ""

# --- 1. Install system dependencies ---
echo "[1/4] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pyqt5 \
    xdotool \
    xclip \
    wl-clipboard \
    ydotool \
    alsa-utils \
    build-essential \
    cmake

# Install Vulkan dev deps for GPU acceleration (optional, non-fatal)
sudo apt-get install -y -qq libvulkan-dev glslc 2>/dev/null || true

echo "  ✓ System dependencies installed"

# --- 2. Build whisper-cli ---
echo "[2/4] Building whisper-cli..."
BUILD_DIR="$REPO_ROOT/build"
mkdir -p "$BUILD_DIR"

# Enable Vulkan GPU acceleration if available
CMAKE_EXTRA=""
if pkg-config --exists vulkan 2>/dev/null || [ -f /usr/include/vulkan/vulkan.h ]; then
    CMAKE_EXTRA="-DGGML_VULKAN=ON"
    echo "  → Vulkan detected, building with GPU support"
else
    echo "  → Vulkan not found, building CPU-only"
fi

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release $CMAKE_EXTRA
cmake --build "$BUILD_DIR" -j "$(nproc)"

WHISPER_CLI="$BUILD_DIR/bin/whisper-cli"
if [ ! -f "$WHISPER_CLI" ]; then
    # Fallback: some builds put it directly in build/
    WHISPER_CLI="$BUILD_DIR/whisper-cli"
fi

if [ ! -x "$WHISPER_CLI" ]; then
    echo "  ✗ Failed to build whisper-cli"
    exit 1
fi
echo "  ✓ whisper-cli built: $WHISPER_CLI"

# --- 3. Download model ---
echo "[3/4] Downloading model ($MODEL_NAME)..."
MODEL_DIR="$REPO_ROOT/models"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

if [ -f "$MODEL_PATH" ]; then
    echo "  ✓ Model already exists: $MODEL_PATH"
else
    bash "$REPO_ROOT/models/download-ggml-model.sh" "$MODEL_NAME"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "  ✗ Model download failed"
        exit 1
    fi
    echo "  ✓ Model downloaded: $MODEL_PATH"
fi

# --- 4. Create config ---
echo "[4/4] Creating config..."
mkdir -p "$CONFIG_DIR"
CONFIG_FILE="$CONFIG_DIR/config.ini"

if [ -f "$CONFIG_FILE" ]; then
    echo "  → Config exists: $CONFIG_FILE"
    # Update model and whisper_cli paths to match current build
    if grep -q "^model\s*=" "$CONFIG_FILE"; then
        sed -i "s|^model\s*=.*|model = $MODEL_PATH|" "$CONFIG_FILE"
    fi
    if grep -q "^whisper_cli\s*=" "$CONFIG_FILE"; then
        sed -i "s|^whisper_cli\s*=.*|whisper_cli = $WHISPER_CLI|" "$CONFIG_FILE"
    fi
    echo "  ✓ Config updated (model=$MODEL_PATH)"
else
    cat > "$CONFIG_FILE" <<EOF
[whisper-linux]
whisper_cli = $WHISPER_CLI
model = $MODEL_PATH
language = ru
threads = 4
display_server =
use_clipboard_fallback = False
notification = True
EOF
    echo "  ✓ Config created: $CONFIG_FILE"
fi

# --- Optional: uinput access for ydotool (Wayland text injection) ---
if [ -c /dev/uinput ]; then
    echo "[+] Setting up /dev/uinput access for ydotool..."
    # Persistent udev rule
    UINPUT_RULE="/etc/udev/rules.d/99-uinput-whisper.rules"
    if [ ! -f "$UINPUT_RULE" ]; then
        echo "KERNEL==\"uinput\", MODE=\"0660\", GROUP=\"$(id -gn)\"" | sudo tee "$UINPUT_RULE" > /dev/null
        sudo udevadm control --reload-rules
        sudo udevadm trigger /dev/uinput
    fi
    # Immediate fix for current session
    sudo chmod 0660 /dev/uinput
    sudo chown "root:$(id -gn)" /dev/uinput
    echo "  ✓ /dev/uinput access configured"
fi

# --- Optional: install desktop file for autostart ---
AUTOSTART_DIR="$HOME/.config/autostart"
DESKTOP_FILE="$SCRIPT_DIR/whisper-linux.desktop"
if [ -f "$DESKTOP_FILE" ]; then
    mkdir -p "$AUTOSTART_DIR"
    # Update Exec path in desktop file
    sed "s|Exec=.*|Exec=python3 -m app|" \
        "$DESKTOP_FILE" > "$AUTOSTART_DIR/whisper-linux.desktop"
    echo "  ✓ Autostart desktop file installed"
fi

# --- Set up keyboard shortcut (GNOME) ---
LAUNCHER="$SCRIPT_DIR/whisper-linux"
TOGGLE_CMD="$LAUNCHER --toggle"
echo ""
echo "=== Installation complete ==="
echo ""
echo "Usage:"
echo "  $LAUNCHER              # Start (tray icon)"
echo "  $LAUNCHER --toggle     # Toggle recording"
echo "  $LAUNCHER --debug      # Start with debug logging"
echo "  pkill -f whisper-linux  # Stop"
echo ""
echo "=== Keyboard shortcut setup ==="
echo "  GNOME: Settings → Keyboard → Custom Shortcuts → Add:"
echo "    Name:     whisper-linux"
echo "    Command:  $TOGGLE_CMD"
echo "    Shortcut: Super+V (or any key you like)"
echo ""
echo "  Or run this command to set it up automatically:"
echo "    gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings \"['/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/whisper-linux/']\""
echo "    gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/whisper-linux/ name 'whisper-linux'"
echo "    gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/whisper-linux/ command '$TOGGLE_CMD'"
echo "    gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/whisper-linux/ binding 'F8'"
