#!/usr/bin/env python3
"""Live microphone streaming with Parakeet-tdt-0.6b-v3.

Captures audio from the mic (pw-record or arecord), runs a sliding-window
transcription, and prints text progressively as new audio arrives.
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
SR = 16000
DEFAULT_WINDOW_S = 8.0
DEFAULT_SHIFT_S = 2.5


def _norm(w: str) -> str:
    return "".join(ch for ch in w.lower() if ch.isalnum())


def _dedup_overlap(prev_text: str, curr_text: str) -> str:
    prev_words = prev_text.split()
    curr_words = curr_text.split()
    prev_norm = [_norm(w) for w in prev_words]
    curr_norm = [_norm(w) for w in curr_words]
    max_n = min(len(prev_norm), len(curr_norm))
    for n in range(max_n, 0, -1):
        if prev_norm[-n:] == curr_norm[:n]:
            return " ".join(curr_words[n:])
    return curr_text


def _find_recorder():
    for p in ("pw-record", "/usr/bin/pw-record", "/usr/local/bin/pw-record"):
        if shutil.which(p) if not os.path.isabs(p) else os.path.isfile(p):
            return p
    return shutil.which("arecord")


def _build_record_cmd():
    rec = _find_recorder()
    if not rec:
        raise RuntimeError("Neither pw-record nor arecord found")
    if "pw-record" in rec:
        return [rec, "--format", "s16", "--rate", str(SR), "--channels", "1", "-"]
    return [rec, "-f", "S16_LE", "-r", str(SR), "-c", "1", "-t", "raw"]


def load_model():
    import nemo.collections.asr as nemo_asr
    from nemo.utils import logging as nemo_logging
    import torch
    import logging

    nemo_logging.set_verbosity(nemo_logging.ERROR)
    logging.getLogger("nemo_logger").setLevel(logging.ERROR)
    print(f"Loading {MODEL_NAME} ...", flush=True)
    asr = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
    asr = asr.cuda() if torch.cuda.is_available() else asr
    nemo_logging.set_verbosity(nemo_logging.ERROR)
    return asr


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_S,
                        help=f"sliding window length in seconds (default: {DEFAULT_WINDOW_S})")
    parser.add_argument("--shift", type=float, default=DEFAULT_SHIFT_S,
                        help=f"window shift / step in seconds (default: {DEFAULT_SHIFT_S})")
    args = parser.parse_args()

    if args.shift <= 0 or args.window <= 0:
        sys.exit("window and shift must be positive")
    if args.shift > args.window:
        sys.exit("shift must be <= window")

    import soundfile as sf

    asr = load_model()
    cmd = _build_record_cmd()

    win_samples = int(args.window * SR)
    shift_samples = int(args.shift * SR)
    shift_bytes = shift_samples * 2  # int16 mono

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", prefix="parakeet-mic-", delete=False)
    tmp_wav = tmp.name
    tmp.close()

    print(f"Recorder: {cmd[0]}", flush=True)
    print(f"Window: {args.window:.1f}s, shift: {args.shift:.1f}s "
          f"(overlap: {args.window - args.shift:.1f}s)", flush=True)
    print("--- speak into mic; Ctrl-C to stop ---\n", flush=True)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _shutdown(*_):
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)
        print("\n--- stopped ---", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    buffer = np.zeros(0, dtype=np.int16)
    prev_text = ""

    try:
        while True:
            new_bytes = proc.stdout.read(shift_bytes)
            if not new_bytes:
                break
            new_samples = np.frombuffer(new_bytes, dtype=np.int16)
            buffer = np.concatenate([buffer, new_samples])
            if len(buffer) > win_samples:
                buffer = buffer[-win_samples:]
            if len(buffer) < shift_samples:
                continue

            sf.write(tmp_wav, buffer, SR)
            t0 = time.monotonic()
            out = asr.transcribe([tmp_wav], timestamps=False, verbose=False)
            elapsed = time.monotonic() - t0
            current = out[0].text.strip()
            if not current:
                continue

            new_part = _dedup_overlap(prev_text, current) if prev_text else current
            if new_part.strip():
                print(new_part, end=" ", flush=True)
            prev_text = current
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
