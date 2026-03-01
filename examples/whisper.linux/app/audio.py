"""Audio recording, streaming, and VAD for whisper.linux."""

import array
import math
import os
import signal
import struct
import subprocess
import tempfile
import threading
import time
from typing import Optional

from .config import Config, log


# ---------------------------------------------------------------------------
# AudioRecorder
# ---------------------------------------------------------------------------

class AudioRecorder:
    """Records audio to a temporary WAV file."""

    def __init__(self, config: Config):
        self._config = config
        self._proc: Optional[subprocess.Popen] = None
        self._wav_path: Optional[str] = None

    @property
    def wav_path(self) -> Optional[str]:
        return self._wav_path

    @property
    def is_recording(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _build_record_cmd(self, path: str) -> list:
        audio_device = self._config.audio_device
        if audio_device.startswith("pw") or audio_device == "auto":
            pw = self._find_pw_record()
            if pw:
                return [pw, "--format", "s16", "--rate", "16000", "--channels", "1", path]
        if audio_device and audio_device not in ("auto", "default"):
            return ["arecord", "-D", audio_device, "-f", "S16_LE", "-r", "16000",
                    "-c", "1", "-t", "wav", path]
        pw = self._find_pw_record()
        if pw:
            return [pw, "--format", "s16", "--rate", "16000", "--channels", "1", path]
        return ["arecord", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "wav", path]

    @staticmethod
    def _find_pw_record():
        for p in ("/usr/bin/pw-record", "/usr/local/bin/pw-record"):
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
        return None

    def start(self) -> str:
        if self.is_recording:
            raise RuntimeError("Already recording")
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="whisper-linux-")
        os.close(fd)
        self._wav_path = path
        cmd = self._build_record_cmd(path)
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.2)
        if self._proc.poll() is not None:
            stderr = self._proc.stderr.read().decode(errors="replace").strip()
            self._proc = None
            os.unlink(path)
            self._wav_path = None
            raise RuntimeError(f"Recording failed to start ({cmd[0]}): {stderr}")
        log.info("Recording started → %s (pid %d, cmd=%s)", path, self._proc.pid, cmd[0])
        return path

    def stop(self) -> Optional[str]:
        if self._proc is None:
            return None
        if self._proc.poll() is None:
            self._proc.send_signal(signal.SIGINT)
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        log.info("Recording stopped → %s", self._wav_path)
        self._proc = None
        return self._wav_path

    def cleanup(self):
        self.stop()
        if self._wav_path and os.path.exists(self._wav_path):
            os.unlink(self._wav_path)
            self._wav_path = None


# ---------------------------------------------------------------------------
# _write_wav
# ---------------------------------------------------------------------------

def _write_wav(path: str, pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, bits: int = 16):
    """Write raw PCM s16le data to a WAV file with RIFF header."""
    data_size = len(pcm_data)
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm_data)


# ---------------------------------------------------------------------------
# SimpleVAD
# ---------------------------------------------------------------------------

class SimpleVAD:
    """Energy-based Voice Activity Detection on raw s16le PCM frames."""

    FRAME_MS = 30
    TRAILING_SILENCE_MS = 300

    def __init__(self, config: Config, on_speech_end=None, on_speech_start=None):
        self._threshold = config.vad_threshold
        self._min_speech_ms = config.min_speech_ms
        self._max_speech_s = config.max_speech_s
        self._sample_rate = 16000
        self._frame_bytes = self._sample_rate * 2 * self.FRAME_MS // 1000
        self._silence_frames = self.TRAILING_SILENCE_MS // self.FRAME_MS
        self.on_speech_end = on_speech_end
        self.on_speech_start = on_speech_start

        self._buffer = bytearray()
        self._speech_buffer = bytearray()
        self._in_speech = False
        self._silent_count = 0

    def reset(self):
        self._buffer.clear()
        self._speech_buffer.clear()
        self._in_speech = False
        self._silent_count = 0

    @staticmethod
    def _rms(frame_bytes: bytes) -> float:
        if len(frame_bytes) < 2:
            return 0.0
        samples = array.array("h")
        samples.frombytes(frame_bytes[:len(frame_bytes) - len(frame_bytes) % 2])
        if not samples:
            return 0.0
        return math.sqrt(sum(s * s for s in samples) / len(samples))

    def feed(self, chunk: bytes):
        self._buffer.extend(chunk)
        while len(self._buffer) >= self._frame_bytes:
            frame = bytes(self._buffer[:self._frame_bytes])
            del self._buffer[:self._frame_bytes]
            rms = self._rms(frame)

            if rms >= self._threshold:
                if not self._in_speech:
                    self._in_speech = True
                    self._silent_count = 0
                    log.debug("VAD: speech start (RMS=%.0f)", rms)
                    if self.on_speech_start:
                        self.on_speech_start()
                self._speech_buffer.extend(frame)
                self._silent_count = 0
            elif self._in_speech:
                self._speech_buffer.extend(frame)
                self._silent_count += 1
                if self._silent_count >= self._silence_frames:
                    self._emit_speech()

            max_bytes = int(self._max_speech_s * self._sample_rate * 2)
            if self._in_speech and len(self._speech_buffer) >= max_bytes:
                self._emit_speech()

    def _emit_speech(self):
        pcm = bytes(self._speech_buffer)
        duration_ms = len(pcm) * 1000 // (self._sample_rate * 2)
        self._speech_buffer.clear()
        self._in_speech = False
        self._silent_count = 0
        if duration_ms >= self._min_speech_ms:
            log.debug("VAD: speech end (%dms)", duration_ms)
            if self.on_speech_end:
                self.on_speech_end(pcm)
        else:
            log.debug("VAD: speech too short (%dms), discarded", duration_ms)


# ---------------------------------------------------------------------------
# AudioStream
# ---------------------------------------------------------------------------

class AudioStream:
    """Streams raw PCM from pw-record to a callback in a reader thread."""

    SAMPLE_RATE = 16000
    BYTES_PER_SEC = 32000
    CHUNK_MS = 100
    CHUNK_BYTES = BYTES_PER_SEC * CHUNK_MS // 1000

    def __init__(self, config: Config):
        self._config = config
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._on_data = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running and self._proc is not None and self._proc.poll() is None

    def _build_stream_cmd(self) -> list:
        audio_device = self._config.audio_device
        pw = AudioRecorder._find_pw_record()
        if pw:
            return [pw, "--format", "s16", "--rate", "16000", "--channels", "1", "-"]
        cmd = ["arecord", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"]
        if audio_device and audio_device not in ("auto", "default"):
            cmd.extend(["-D", audio_device])
        return cmd

    def start(self, on_data):
        if self._running:
            raise RuntimeError("AudioStream already running")
        self._on_data = on_data
        self._running = True
        cmd = self._build_stream_cmd()
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.2)
        if self._proc.poll() is not None:
            stderr = self._proc.stderr.read().decode(errors="replace").strip()
            self._running = False
            self._proc = None
            raise RuntimeError(f"AudioStream failed to start ({cmd[0]}): {stderr}")
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        log.info("AudioStream started (pid %d, cmd=%s)", self._proc.pid, cmd[0])

    def stop(self):
        self._running = False
        if self._proc and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGINT)
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        self._proc = None
        log.info("AudioStream stopped")

    def _reader_loop(self):
        try:
            while self._running and self._proc and self._proc.poll() is None:
                data = self._proc.stdout.read(self.CHUNK_BYTES)
                if not data:
                    break
                if self._on_data and self._running:
                    self._on_data(data)
        except Exception as e:
            log.error("AudioStream reader error: %s", e)
        finally:
            self._running = False
