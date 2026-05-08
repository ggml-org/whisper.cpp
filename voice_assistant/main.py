#!/usr/bin/env python3
"""
Local Voice Assistant — Streaming Mode + Memory
Flow: Microphone (VAD) → Whisper large-v3 → Ollama chat (stream) → Piper (PCM) → sounddevice

# TODO (future upgrades):
#   - streaming transcription (faster first-token latency)
#   - interruptible TTS (detect mic energy, kill player thread mid-sentence)
#   - real-time streaming ASR + streaming LLM output simultaneously
#   - wake word support (e.g. openWakeWord) so no Enter key is needed
"""

import os
import sys
import io
import json
import re
import time
import wave
import platform
import threading
import queue
from typing import Optional, Iterator


def ensure_project_venv() -> None:
    """Re-run this script with the local project virtualenv when available."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir    = os.path.join(project_dir, ".venv")
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
        package     = package_name or module_name
        project_dir = os.path.dirname(os.path.abspath(__file__))
        venv_python = os.path.join(project_dir, ".venv", "bin", "python")
        print(f"Missing Python package: {package}")
        print("This usually means main.py is running outside voice_assistant/.venv.")
        print(f"Install with: {venv_python} -m pip install -r {os.path.join(project_dir, 'requirements.txt')}")
        print(f"Run with:     {venv_python} {os.path.abspath(__file__)}")
        sys.exit(1)


ensure_project_venv()

requests     = require_dependency("requests")
np           = require_dependency("numpy")
sd           = require_dependency("sounddevice")
wavfile      = require_dependency("scipy.io.wavfile")
WhisperModel = require_dependency("faster_whisper").WhisperModel

# ─── Configuration ────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
CHANNELS    = 1

WHISPER_MODEL_SIZE = "distil-large-v3"
# "zh" handles Chinese-primary + English code-switching and avoids hallucinations.
# Change to "en" for English-only, or None for auto-detect (not recommended: prone
# to the "Chinese character, Chinese character..." repetition loop on Chinese audio).
WHISPER_LANGUAGE   = "en"

# Ollama chat API (maintains multi-turn context server-side via messages list)
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL    = "gemma4"
SYSTEM_PROMPT   = (
    "You are a senior technical interviewer at a Tier-1 tech company. Your tone is professional and slightly skeptical. Provide a 'Bridge Sentence' whenever I am stuck. When giving feedback, use a 'Scaffold' approach: tell me the 3 key concepts I should have used, then let me try again with those concepts."
    "Keep responses concise and natural for speech. "
    "Avoid markdown, bullet points, or code blocks."
)
MAX_HISTORY_TURNS = 10  # sliding window: keep last N user+assistant pairs

PIPER_MODEL      = os.path.expanduser("~/piper_models/en_US-lessac-medium.onnx")
RECORDING_PATH   = "/tmp/va_recording.wav"
TRANSCRIPT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversation.txt")

# VAD: stop recording after SILENCE_DURATION seconds of RMS < SILENCE_THRESHOLD
MAX_RECORD_SECONDS = 30
SILENCE_THRESHOLD  = 300   # RMS amplitude (0–32767); raise if mic is noisy
SILENCE_DURATION   = 3.0   # seconds of quiet to trigger stop
MIN_SPEECH_SECONDS = 0.5   # ignore VAD until at least this much audio is captured

# TTS speed: 1.0 = normal, 1.2 = ~20% slower, 1.4 = noticeably slower
TTS_LENGTH_SCALE = 1.2

# TTS force-flush: don't wait forever for a sentence boundary in long clauses
MAX_TTS_CHARS = 80

_SENTENCE_END = re.compile(r'(?<=[.!?。！？])\s+|(?<=[.!?。！？])$')

# ─── Conversation history ─────────────────────────────────────────────────────

_chat_history: list = []  # [{role: "user"|"assistant", content: str}, ...]


def _trim_history() -> None:
    """Evict oldest turns so history never exceeds MAX_HISTORY_TURNS exchanges."""
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(_chat_history) > max_msgs:
        del _chat_history[: len(_chat_history) - max_msgs]


def _append_transcript(user_text: str, assistant_text: str) -> None:
    """Append one exchange to the transcript file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]\n")
        f.write(f"You: {user_text}\n\n")
        f.write(f"Assistant: {assistant_text}\n")
        f.write("-" * 60 + "\n\n")

# ─── Device Detection ─────────────────────────────────────────────────────────

def get_whisper_device():
    """Apple Silicon → CPU int8. CUDA available → CUDA float16. Fallback → CPU int8."""
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

# ─── Stage 1: Record from Microphone with VAD ────────────────────────────────

def record_with_vad(path: str) -> None:
    """
    Record mic audio in 100 ms chunks. Stop early once SILENCE_DURATION seconds
    of consecutive silence (RMS < SILENCE_THRESHOLD) is detected after the
    minimum speech window. Hard cap at MAX_RECORD_SECONDS.

    This replaces a fixed-length recording and typically saves 2–4 s of dead air
    per turn, which directly reduces total round-trip latency.
    """
    chunk_frames       = int(SAMPLE_RATE * 0.1)          # 100 ms
    silence_chunks_max = int(SILENCE_DURATION / 0.1)
    min_chunks         = int(MIN_SPEECH_SECONDS / 0.1)
    max_chunks         = int(MAX_RECORD_SECONDS / 0.1)

    print(f"  Recording (VAD, max {MAX_RECORD_SECONDS}s) — speak now...")
    recorded      = []
    silence_count = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16") as stream:
        for i in range(max_chunks):
            chunk, _ = stream.read(chunk_frames)
            recorded.append(chunk.copy())
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            if i >= min_chunks:
                if rms < SILENCE_THRESHOLD:
                    silence_count += 1
                    if silence_count >= silence_chunks_max:
                        break
                else:
                    silence_count = 0

    audio    = np.concatenate(recorded, axis=0)
    duration = len(audio) / SAMPLE_RATE
    wavfile.write(path, SAMPLE_RATE, audio)
    print(f"  Stopped after {duration:.1f}s.")

# ─── Stage 2: Speech-to-Text via faster-whisper ──────────────────────────────

def transcribe(path: str, model) -> str:
    """Return the recognized text from a WAV file."""
    segments, info = model.transcribe(
        path,
        language=WHISPER_LANGUAGE,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,  # breaks "Chinese character..." repetition loops
    )
    detected = info.language if WHISPER_LANGUAGE is None else WHISPER_LANGUAGE
    print(f"  Language: {detected} ({info.language_probability:.2f})")
    return " ".join(seg.text.strip() for seg in segments).strip()

# ─── Stage 3: Stream tokens from Ollama (chat API + history) ─────────────────

def _ollama_tokens(user_text: str) -> Iterator[str]:
    """
    Append user_text to history, stream raw tokens from /api/chat, yield each token.
    Appends the completed reply to history when the stream ends.
    Sentence-buffering for TTS is handled by the caller (main loop), so this
    function stays focused on network I/O and history management.
    """
    _chat_history.append({"role": "user", "content": user_text})
    _trim_history()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + _chat_history
    payload  = {"model": OLLAMA_MODEL, "messages": messages, "stream": True}

    full_response = ""
    try:
        with requests.post(OLLAMA_CHAT_URL, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data   = json.loads(line)
                token  = data.get("message", {}).get("content", "")
                full_response += token
                yield token
                if data.get("done"):
                    break

        if full_response.strip():
            _chat_history.append({"role": "assistant", "content": full_response.strip()})

    except requests.exceptions.ConnectionError:
        yield "[Error: Cannot reach Ollama — is it running? Try: ollama serve]"
    except requests.exceptions.Timeout:
        yield "[Error: Ollama timed out after 120 s.]"
    except Exception as exc:
        yield f"[Error: {exc}]"

# ─── Stage 4+5: Piper TTS → sounddevice ──────────────────────────────────────

def load_piper_voice():
    """Load Piper voice model. Returns (voice, sample_rate) or (None, None)."""
    if not os.path.exists(PIPER_MODEL):
        print(f"  [TTS skipped] Model not found: {PIPER_MODEL}")
        print("  See README for Piper model download instructions.")
        return None, None
    try:
        from piper import PiperVoice  # type: ignore
        voice = PiperVoice.load(PIPER_MODEL)
        voice.config.length_scale = TTS_LENGTH_SCALE
        return voice, voice.config.sample_rate
    except ImportError:
        print("  [TTS skipped] piper-tts not installed (pip install piper-tts).")
        return None, None
    except Exception as exc:
        print(f"  [TTS load error] {exc}")
        return None, None


def _synthesize_to_pcm(voice, text: str):
    """Synthesize one sentence to int16 PCM via synthesize_wav → BytesIO → ndarray."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    buf.seek(0)
    with wave.open(buf, "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
    return np.frombuffer(frames, dtype=np.int16)


def _clean_for_tts(text: str) -> str:
    """Strip markdown symbols so Piper doesn't read them aloud."""
    text = re.sub(r'```[\s\S]*?```', '', text)          # fenced code blocks
    text = re.sub(r'`([^`]*)`', r'\1', text)            # inline code
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # headers
    text = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', text) # bold/italic
    text = re.sub(r'_{1,3}([^_\n]+)_{1,3}', r'\1', text)   # underscore bold/italic
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)  # bullet points
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # numbered lists
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)      # blockquotes
    return text.strip()


def _tts_player_worker(voice, sample_rate: int, q: queue.Queue) -> None:
    """
    Background thread: pull sentences, synthesize PCM, write to one persistent
    sounddevice OutputStream. One open stream avoids inter-sentence gaps.
    Exits on None poison pill; OutputStream.__exit__ drains the buffer.
    """
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16") as stream:
        while True:
            sentence = q.get()
            if sentence is None:
                break
            try:
                clean = _clean_for_tts(sentence)
                if clean:
                    audio = _synthesize_to_pcm(voice, clean)
                    stream.write(audio)
            except Exception as exc:
                print(f"\n  [TTS error] {exc}")

# ─── Main Loop ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 54)
    print("  Local Voice Assistant  (Streaming + Memory)")
    print(f"  STT : Whisper {WHISPER_MODEL_SIZE}")
    print(f"  LLM : {OLLAMA_MODEL}  (chat, history={MAX_HISTORY_TURNS} turns)")
    print("  TTS : Piper → sounddevice")
    print("  VAD : auto-stop on silence")
    print("=" * 54)
    print("Press Ctrl+C at any time to quit.\n")

    device, compute_type = get_whisper_device()
    print(f"Loading Whisper model '{WHISPER_MODEL_SIZE}' [{device}/{compute_type}]...")
    whisper = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
    print("Whisper ready.")

    voice, tts_sample_rate = load_piper_voice()
    if voice:
        print(f"Piper voice ready ({tts_sample_rate} Hz).")
    print()

    while True:
        # ── Auto-start after a brief pause ────────────────────────────────
        try:
            print("Listening in 1 s … (Ctrl+C to quit)")
            time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)
        print()

        # ── Stage 1: Record (VAD) ─────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            record_with_vad(RECORDING_PATH)
        except Exception as exc:
            print(f"  [Recording error] {exc}\n")
            continue
        t1 = time.perf_counter()

        # ── Stage 2: Transcribe ───────────────────────────────────────────
        print("  Transcribing...")
        try:
            text = transcribe(RECORDING_PATH, whisper)
        except Exception as exc:
            print(f"  [Transcription error] {exc}\n")
            continue
        t2 = time.perf_counter()
        print(f"  [record {t1-t0:.1f}s | transcribe {t2-t1:.1f}s]")

        if not text:
            print("  [Nothing recognized — try speaking more clearly]\n")
            continue
        print(f"\nYou: {text}\n")

        # ── Stage 3+4+5: Stream LLM → TTS → play (pipelined) ─────────────
        # Print every token immediately for visible streaming, while batching
        # complete sentences into the TTS queue for natural speech prosody.
        print("Assistant: ", end="", flush=True)
        t3 = time.perf_counter()

        tts_buf = ""
        if voice:
            tts_queue = queue.Queue()
            player = threading.Thread(
                target=_tts_player_worker,
                args=(voice, tts_sample_rate, tts_queue),
                daemon=True,
            )
            player.start()

            for token in _ollama_tokens(text):
                print(token, end="", flush=True)   # visible token-by-token streaming
                tts_buf += token

                # Flush complete sentences to TTS
                while True:
                    m = _SENTENCE_END.search(tts_buf)
                    if not m:
                        break
                    sentence = tts_buf[: m.start() + 1].strip()
                    tts_buf  = tts_buf[m.end():]
                    if sentence:
                        tts_queue.put(sentence)

                # Force-flush long clauses (handles Chinese without sentence-final punctuation)
                if len(tts_buf) >= MAX_TTS_CHARS:
                    for sep in (" ", "，", ",", "、"):
                        pos = tts_buf.rfind(sep, 0, MAX_TTS_CHARS)
                        if pos > MAX_TTS_CHARS // 3:
                            tts_queue.put(tts_buf[:pos].strip())
                            tts_buf = tts_buf[pos + len(sep):]
                            break
                    else:
                        tts_queue.put(tts_buf.strip())
                        tts_buf = ""

            if tts_buf.strip():
                tts_queue.put(tts_buf.strip())
            tts_queue.put(None)
            player.join()
        else:
            for token in _ollama_tokens(text):
                print(token, end="", flush=True)

        t4 = time.perf_counter()

        if _chat_history and _chat_history[-1]["role"] == "assistant":
            _append_transcript(text, _chat_history[-1]["content"])
        turns = len(_chat_history) // 2
        print(f"\n  [LLM+TTS {t4-t3:.1f}s | history: {turns} turn(s)]\n")


if __name__ == "__main__":
    main()
