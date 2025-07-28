#!/usr/bin/env python3
"""
siri_remote_whisper_socket.py – Streams Siri-remote audio to whisper-stream-socket

Compared with the original FIFO / stdin based version this client talks to the
new examples/stream_socket/whisper-stream-socket binary via a Unix domain
socket (default: /tmp/whisper_stream.sock).  This frees the terminal’s
stdin/stdout for logging and lets us receive incremental JSON transcripts over
the same full-duplex connection that carries the raw PCM.

Usage example (same PacketLogger pipe as before):

    sudo PacketLogger … convert -s -f nhdr | \
        ./siri_remote_whisper_socket.py C08N1RGN2330 |
        jq -r .text

One TCP/Unix socket connection is opened per voice session: the client writes
16-kHz 16-bit little-endian mono PCM and simultaneously reads newline-delimited
JSON messages of the form

    {"type": "partial", "text": "…"}
    {"type": "final",   "text": "…"}

All diagnostic output from the C++ server goes to its stderr so your pipeline
remains clean.
"""

from __future__ import annotations

import io
import os
import platform
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional


# ── Auto-locate Homebrew libopus on macOS (before importing opuslib) ────────────
if platform.system() == "Darwin" and "OPUS_LIB" not in os.environ:
    try:
        brew_prefix = subprocess.check_output(["brew", "--prefix", "opus"], text=True).strip()
        dylib = Path(brew_prefix, "lib/libopus.dylib")
        if dylib.exists():
            os.environ["OPUS_LIB"] = str(dylib)
    except Exception:
        pass

try:
    import opuslib  # type: ignore
except ImportError:
    sys.stderr.write("✖︎ opuslib not installed –  pip install opuslib\n")
    sys.exit(1)


SAMPLE_RATE = 16_000
CHANNELS = 1

# Top-2-bit mask for Opus header heuristic
HEADER_MASK = 0xC0

SOCKET_PATH = "/tmp/whisper_stream.sock"  # must match C++ server default


# ── Helpers ────────────────────────────────────────────────────────────────────


class Opus16kDecoder:
    """Very thin wrapper around opuslib.Decoder returning 16-bit PCM bytes."""

    def __init__(self):
        # returns int16 PCM in little endian already
        self._dec = opuslib.Decoder(SAMPLE_RATE, CHANNELS)

    def decode(self, packet: bytes) -> bytes:
        return self._dec.decode(packet, 640)  # 40 ms @16 kHz mono → 640 samples


def looks_like_header(tokens: list[str]) -> bool:
    return (
        len(tokens) >= 2
        and (int(tokens[0], 16) & HEADER_MASK) == 0x40
        and tokens[1] == "20"
    )


def is_voice_start(tail: str) -> bool:
    return tail.endswith("1B 39 00 20 00")


def is_voice_end(tail: str) -> bool:
    return tail.endswith("1B 39 00 00 00") or tail.endswith("1B 39 00 00")


# Voice-related packets always contain 1B 39 (start/stop) or 1B 35 (audio)
def is_voice_packet(tokens: list[str]) -> bool:
    return any(tokens[i] == "1B" and tokens[i + 1] in ("39", "35") for i in range(len(tokens) - 1))


# ── Whisper socket client ──────────────────────────────────────────────────────


class WhisperSocketSession:
    """Opens one Unix-domain socket, streams PCM and prints transcripts."""

    def __init__(self, path: str = SOCKET_PATH):
        self.path = path
        self.sock: Optional[socket.socket] = None
        self._read_thread: Optional[threading.Thread] = None
        self._closed = threading.Event()

    # ── Lifecycle ------------------------------------------------------------

    def start(self):
        if self.sock is not None:
            self.close()

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.connect(self.path)
        except FileNotFoundError:
            sys.stderr.write(f"✖︎ Whisper socket {self.path} not found – is whisper-stream-socket running?\n")
            raise

        # Reader thread for JSON-lines
        self._read_thread = threading.Thread(target=self._reader, daemon=True)
        self._read_thread.start()

    def write(self, pcm_s16: bytes):
        if self.sock is None:
            return  # not in session
        try:
            self.sock.sendall(pcm_s16)
        except (BrokenPipeError, ConnectionResetError):
            # Server died mid-session – abandon silently.
            self.close()

    def close(self):
        if self.sock is None:
            return

        try:
            # Closing write-side first lets the server detect EOF and send final JSON.
            self.sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass

        # Wait for reader to finish (with timeout so we never block forever).
        if self._read_thread and self._read_thread.is_alive():
            self._closed.wait(timeout=2.0)

        try:
            self.sock.close()
        finally:
            self.sock = None

    # ── Background reader ----------------------------------------------------

    def _reader(self):
        assert self.sock is not None
        f = self.sock.makefile("r", encoding="utf-8", newline="\n", buffering=1)
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    import json

                    msg = json.loads(line)
                    kind = msg.get("type", "?")
                    text = msg.get("text", "")
                    prefix = "[FINAL]" if kind == "final" else "[PART ]"
                    print(f"{prefix} {text}")
                except json.JSONDecodeError:
                    print("WHISPER:", line)
        finally:
            self._closed.set()


# ── Main packet-decoder loop (unchanged except session handling) ─────────────


hex_tail_pat = re.compile(r"([0-9A-F]{2}(?:\s+[0-9A-F]{2})+)$")


def main():
    # Optional MAC or device name filter argument
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    mac_filt, name_filt = "", ""
    if filt:
        if re.fullmatch(r"[0-9A-F]{2}([:\-]?[0-9A-F]{2}){5}", filt, re.I):
            mac_filt = filt.replace("-", ":").lower()
        else:
            name_filt = filt.lower()
        print(f"Filtering on: {filt}")

    opus_dec = Opus16kDecoder()
    whisper = WhisperSocketSession()

    collecting = False
    frame_frag: List[str] = []
    header_streak = 0  # consecutive Opus headers while not collecting

    def send_frame(hex_str: str):
        """Decode single hex-encoded Opus packet and stream to Whisper."""
        data = bytes.fromhex(hex_str.replace(" ", ""))
        if len(data) <= data[0]:
            return
        try:
            whisper.write(opus_dec.decode(data[1 : 1 + data[0]]))
        except opuslib.OpusError as e:
            print(e, file=sys.stderr)

    def flush_leftover():
        nonlocal frame_frag
        if frame_frag:
            send_frame(" ".join(frame_frag))
            frame_frag.clear()

    for raw in sys.stdin:
        line = raw.rstrip()
        if (
            (mac_filt and mac_filt not in line.lower())
            or (name_filt and name_filt not in line.lower())
            or " RECV " not in line
        ):
            continue

        m = hex_tail_pat.search(line)
        if not m:
            continue
        hex_tail = m.group(1)
        tokens = hex_tail.split()

        if is_voice_start(hex_tail):
            if collecting:
                flush_leftover()
                whisper.close()
            print("Voice started …")
            try:
                whisper.start()
            except FileNotFoundError:
                break  # fatal, no server
            collecting = True
            frame_frag.clear()
            continue

        if is_voice_end(hex_tail):
            if not collecting:
                continue  # stray end
            print("Voice ended …")
            flush_leftover()
            whisper.close()
            collecting = False
            continue

        if looks_like_header(tokens) and is_voice_packet(tokens):
            has_b8 = "B8" in tokens  # Opus packet marker

            if not collecting:
                if not has_b8:
                    header_streak = 0
                    continue

                header_streak += 1
                if header_streak < 3:
                    continue
                # auto-start after 3 consecutive headers
                print("Voice auto-started (3 B8 headers)")
                try:
                    whisper.start()
                except FileNotFoundError:
                    break
                collecting = True
                frame_frag.clear()
            else:
                header_streak = 0

            if not has_b8:
                frame_frag.extend(tokens)
                continue

            try:
                b8 = tokens.index("B8")
            except ValueError:
                b8 = -1
            if b8 > 0:
                if frame_frag:
                    send_frame(" ".join(frame_frag))
                frame_frag = tokens[b8 - 1 :]
            else:
                frame_frag.extend(tokens)

        elif collecting and len(tokens) > 3:
            # Append non-header packets belonging to current frame
            frame_frag.extend(tokens[3:])

    # Clean termination on ctrl-C / EOF
    whisper.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
