"""Text injection (xdotool/ydotool/clipboard) for whisper.linux."""

import os
import subprocess
import time

from .config import Config, log


class TextInjector:
    """Injects text at cursor position using xdotool, ydotool, or clipboard."""

    # X11 keysym → evdev name (for ydotool which uses evdev names)
    _YDOTOOL_KEY_MAP = {
        "Return": "Enter",
        "Escape": "Esc",
    }

    def __init__(self, config: Config):
        self.config = config

    def inject(self, text: str):
        if not text:
            return
        ds = self.config.display_server
        if ds == "wayland":
            self._inject_wayland(text)
        else:
            self._inject_x11(text)

    def _inject_x11(self, text: str):
        if self.config.use_clipboard_fallback or not text.isascii():
            self._inject_clipboard_x11(text)
            return
        try:
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--", text],
                check=True, timeout=10,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            log.warning("xdotool type failed, falling back to clipboard")
            self._inject_clipboard_x11(text)

    def _inject_clipboard_x11(self, text: str):
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode("utf-8"), check=True, timeout=5,
        )
        time.sleep(0.1)
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
            check=True, timeout=5,
        )

    def _inject_wayland(self, text: str):
        try:
            subprocess.run(["wtype", "--", text], check=True, timeout=10)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        try:
            subprocess.run(
                ["wl-copy", "--", text], check=True, timeout=5,
            )
            subprocess.run(
                ["wl-copy", "--primary", "--", text], check=True, timeout=5,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            log.warning("wl-copy failed, falling back to xclip (XWayland only)")
            self._inject_x11(text)
            return

        paste_keys = self.config.paste_keys
        log.info("Clipboard set via wl-copy, sending %s", paste_keys)
        time.sleep(0.3)

        try:
            subprocess.run(
                ["ydotool", "key", "--delay", "100", paste_keys],
                check=True, timeout=5, capture_output=True,
            )
            log.info("Paste sent via ydotool (%s)", paste_keys)
            return
        except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError) as e:
            log.debug("ydotool failed: %s", e)

        try:
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers", paste_keys],
                check=True, timeout=5,
            )
            log.info("Paste sent via xdotool (%s)", paste_keys)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        log.warning("Text copied to clipboard (Ctrl+V to paste). "
                    "For auto-paste: sudo chmod 0660 /dev/uinput && "
                    "sudo chown root:$USER /dev/uinput")

    def send_key(self, key: str):
        """Send a key press (e.g. 'Return', 'Tab', 'ctrl+BackSpace')."""
        ds = self.config.display_server
        if ds == "wayland":
            self._send_key_wayland(key)
        else:
            self._send_key_x11(key)

    def _send_key_x11(self, key: str):
        try:
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers", key],
                check=True, timeout=5,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log.warning("xdotool key '%s' failed: %s", key, e)

    def _send_key_wayland(self, key: str):
        # wtype supports XKB key names natively on Wayland
        try:
            if "+" in key:
                parts = key.split("+")
                cmd = ["wtype"]
                for mod in parts[:-1]:
                    cmd.extend(["-M", mod.lower()])
                cmd.extend(["-k", parts[-1]])
            else:
                cmd = ["wtype", "-k", key]
            subprocess.run(cmd, check=True, timeout=5)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        # ydotool uses evdev key names (Enter, not Return; Esc, not Escape)
        try:
            if "+" in key:
                parts = key.split("+")
                evdev_parts = [self._YDOTOOL_KEY_MAP.get(p, p) for p in parts]
                evdev_key = "+".join(evdev_parts)
            else:
                evdev_key = self._YDOTOOL_KEY_MAP.get(key, key)
            subprocess.run(
                ["ydotool", "key", "--delay", "50", evdev_key],
                check=True, timeout=5, capture_output=True,
            )
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        # xdotool via XWayland
        try:
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers", key],
                check=True, timeout=5,
            )
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        log.warning("Failed to send key '%s' — no working key sender", key)
