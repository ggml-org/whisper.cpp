#!/usr/bin/env python3
"""
Local Voice Assistant
Flow: Microphone → Whisper large-v3 → Ollama → Piper TTS → afplay

# TODO (future upgrades):
#   - streaming transcription (faster first-token latency)
#   - VAD (voice activity detection, e.g. silero-vad) to auto-stop recording
#   - interruptible TTS (stop playback when user starts speaking)
#   - real-time streaming ASR + streaming LLM output
#   - memory / chat history (pass conversation turns to Ollama)
#   - wake word support (e.g. openWakeWord) so no Enter key is needed
"""

import os
import sys
import wave
import platform
import subprocess
from typing import Optional


def ensure_project_venv() -> None:
    """Re-run this script with the local project virtualenv when available."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(project_dir, ".venv")
    venv_python = os.path.join(project_dir, ".venv", "bin", "python")

    if not os.path.exists(venv_python):
        return
    if os.path.realpath(sys.prefix) == os.path.realpath(venv_dir):
        return

    os.execv(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]])


def require_dependency(module_name: str, package_name: Optional[str] = None):
    """Import a dependency or exit with an actionable environment hint."""
    try:
        return __import__(module_name, fromlist=["*"])
    except ModuleNotFoundError:
        package = package_name or module_name
        project_dir = os.path.dirname(os.path.abspath(__file__))
        venv_python = os.path.join(project_dir, ".venv", "bin", "python")
        print(f"Missing Python package: {package}")
        print("This usually means main.py is running outside voice_assistant/.venv.")
        print(f"Install with: {venv_python} -m pip install -r {os.path.join(project_dir, 'requirements.txt')}")
        print(f"Run with:     {venv_python} {os.path.abspath(__file__)}")
        sys.exit(1)


ensure_project_venv()

requests = require_dependency("requests")
np = require_dependency("numpy")
sd = require_dependency("sounddevice")
wavfile = require_dependency("scipy.io.wavfile")
WhisperModel = require_dependency("faster_whisper").WhisperModel

# ─── Configuration ────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000      # Whisper requires 16 kHz mono
RECORD_SECONDS = 6
CHANNELS = 1

WHISPER_MODEL_SIZE = "large-v3"
WHISPER_LANGUAGE = None  # None = auto-detect (handles Chinese + English)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

# Download a Piper .onnx model and point this path to it (see README).
# Example: ~/piper_models/en_US-lessac-medium.onnx
PIPER_MODEL = os.path.expanduser("~/piper_models/en_US-lessac-medium.onnx")

RECORDING_PATH = "/tmp/va_recording.wav"
TTS_OUTPUT_PATH = "/tmp/va_response.wav"

# ─── Device Detection ─────────────────────────────────────────────────────────

def get_whisper_device():
    """
    Apple Silicon → CPU int8 (fast, no CUDA needed).
    CUDA available  → CUDA float16.
    Fallback        → CPU int8.
    """
    if platform.machine() in ("arm64", "aarch64"):
        print("  [Apple Silicon detected] Using CPU int8 compute.")
        return "cpu", "int8"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"

# ─── Stage 1: Record from Microphone ─────────────────────────────────────────

def record_audio(path: str, seconds: int = RECORD_SECONDS) -> None:
    """Capture audio from the default mic, save as 16-bit PCM WAV."""
    print(f"  Recording {seconds}s — speak now...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()  # block until the buffer is full
    wavfile.write(path, SAMPLE_RATE, audio)
    print("  Recording saved.")

# ─── Stage 2: Speech-to-Text via faster-whisper ──────────────────────────────

def transcribe(path: str, model: WhisperModel) -> str:
    """Return the recognized text from a WAV file."""
    segments, info = model.transcribe(
        path,
        language=WHISPER_LANGUAGE,
        beam_size=5,
        vad_filter=True,   # skip silent/noise-only segments
    )
    detected = info.language if WHISPER_LANGUAGE is None else WHISPER_LANGUAGE
    print(f"  Detected language: {detected} (probability {info.language_probability:.2f})")
    return " ".join(seg.text.strip() for seg in segments).strip()

# ─── Stage 3: Query Local LLM via Ollama ─────────────────────────────────────

def query_ollama(prompt: str) -> str:
    """POST prompt to Ollama, return plain-text response (non-streaming)."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "[Error: Cannot reach Ollama. Is it running? Try: ollama serve]"
    except requests.exceptions.Timeout:
        return "[Error: Ollama request timed out after 120 s.]"
    except Exception as exc:
        return f"[Error: {exc}]"

# ─── Stage 4: Text-to-Speech via Piper ───────────────────────────────────────

def synthesize_speech(text: str, output_path: str) -> bool:
    """
    Convert text to WAV using piper-tts.
    Returns True on success, False if TTS is unavailable or fails.
    """
    if not os.path.exists(PIPER_MODEL):
        print(f"  [TTS skipped] Piper model not found: {PIPER_MODEL}")
        print("  Download a model — see README for instructions.")
        return False

    try:
        from piper import PiperVoice  # type: ignore
    except ImportError:
        print("  [TTS skipped] piper-tts not installed (pip install piper-tts).")
        return False

    try:
        voice = PiperVoice.load(PIPER_MODEL)
        with wave.open(output_path, "w") as wav_file:
            voice.synthesize_wav(text, wav_file)
        return True
    except Exception as exc:
        print(f"  [TTS error] {exc}")
        return False

# ─── Stage 5: Play Audio on macOS ────────────────────────────────────────────

def play_audio(path: str) -> None:
    """Play a WAV file with afplay (macOS built-in)."""
    if not os.path.exists(path):
        print("  [Playback skipped] Audio file not found.")
        return
    subprocess.run(["afplay", path], check=False)

# ─── Main Loop ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 52)
    print("  Local Voice Assistant")
    print(f"  STT: Whisper {WHISPER_MODEL_SIZE}  |  LLM: {OLLAMA_MODEL}")
    print("  TTS: Piper  |  Playback: afplay (macOS)")
    print("=" * 52)
    print("Press Ctrl+C at any time to quit.\n")

    # Load Whisper once — expensive on first call (downloads model if needed)
    device, compute_type = get_whisper_device()
    print(f"Loading Whisper model '{WHISPER_MODEL_SIZE}' [{device}/{compute_type}]...")
    whisper = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
    print("Whisper ready.\n")

    while True:
        # ── Wait for user ──────────────────────────────────────────────────
        try:
            input("Press Enter to start recording...")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)
        print()

        # ── Stage 1: Record ───────────────────────────────────────────────
        try:
            record_audio(RECORDING_PATH)
        except Exception as exc:
            print(f"  [Recording error] {exc}\n")
            continue

        # ── Stage 2: Transcribe ───────────────────────────────────────────
        print("  Transcribing...")
        try:
            text = transcribe(RECORDING_PATH, whisper)
        except Exception as exc:
            print(f"  [Transcription error] {exc}\n")
            continue

        if not text:
            print("  [Nothing recognized — try speaking more clearly]\n")
            continue

        print(f"\nYou: {text}\n")

        # ── Stage 3: LLM ──────────────────────────────────────────────────
        print("  Querying Ollama...")
        response = query_ollama(text)
        print(f"Assistant: {response}\n")

        # ── Stage 4: TTS ──────────────────────────────────────────────────
        print("  Synthesizing speech...")
        tts_ok = synthesize_speech(response, TTS_OUTPUT_PATH)

        # ── Stage 5: Play ─────────────────────────────────────────────────
        if tts_ok:
            print("  Playing response...")
            play_audio(TTS_OUTPUT_PATH)

        print()


if __name__ == "__main__":
    main()
