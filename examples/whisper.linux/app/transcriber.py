"""Transcriber and WakeWordDetector for whisper.linux."""

import difflib
import os
import re
import subprocess

from .config import Config, log


# ---------------------------------------------------------------------------
# Hallucination filter — whisper generates these from training data on silence
# ---------------------------------------------------------------------------

# Known hallucination phrases (training data leaks)
_HALLUCINATION_PATTERNS = [
    r"субтитр",
    r"редактор\s+субтитр",
    r"перевод\w*\s+субтитр",
    r"подредактир",
    r"продолжение\s+следует",
    r"спасибо\s+за\s+(просмотр|подписк|внимание)",
    r"подписывайтесь",
    r"не\s+забудьте\s+подписаться",
    r"ставьте\s+лайк",
    r"с\s+вами\s+был[аио]?\s",
    r"а[пп]+ортизатор",
    r"добро\s+пожаловать",
    r"до\s+новых\s+встреч",
    r"до\s+свидания.*друзья",
    r"subtitles?\s+(by|made|edited|created)",
    r"thanks?\s+for\s+watching",
    r"subscribe\s+(to|and)",
    r"please\s+(like|subscribe)",
    r"don'?t\s+forget\s+to\s+subscribe",
    r"copyright\s+\d{4}",
]

_HALLUCINATION_RE = re.compile(
    "|".join(_HALLUCINATION_PATTERNS), re.IGNORECASE,
)

# Speech rate limits — normal speech is ~2-3 words/sec, ~12-15 chars/sec.
# Hallucinations on short audio produce way more text than physically possible.
_MAX_WORDS_PER_SEC = 5      # generous upper bound
_MAX_CHARS_PER_SEC = 25      # generous upper bound


def _is_hallucination(text: str, duration_s: float = 0) -> bool:
    """Return True if text looks like a whisper hallucination.

    Uses two layers:
    1. Pattern matching — known hallucination phrases.
    2. Speech rate check — if audio duration is known, rejects text that
       is impossibly long for the given duration (e.g. 5 words from 0.5s).
    """
    # Layer 1: known patterns
    if _HALLUCINATION_RE.search(text):
        return True

    # Layer 2: speech rate sanity check (only when duration is known)
    if duration_s > 0:
        words = text.split()
        word_count = len(words)
        char_count = len(text.replace(" ", ""))

        max_words = max(2, duration_s * _MAX_WORDS_PER_SEC)
        max_chars = max(10, duration_s * _MAX_CHARS_PER_SEC)

        if word_count > max_words:
            log.debug("Hallucination (words): %d words in %.1fs (max %.0f)",
                       word_count, duration_s, max_words)
            return True
        if char_count > max_chars:
            log.debug("Hallucination (chars): %d chars in %.1fs (max %.0f)",
                       char_count, duration_s, max_chars)
            return True

    return False


# ---------------------------------------------------------------------------
# WakeWordDetector
# ---------------------------------------------------------------------------

class WakeWordDetector:
    """Checks transcription text for the wake word using exact + fuzzy matching."""

    FUZZY_THRESHOLD = 0.7

    def __init__(self, wake_word: str):
        self._wake_word = wake_word.lower().strip()
        self._variants = self._build_variants(self._wake_word)

    @staticmethod
    def _build_variants(word):
        variants = {word}
        for suffix in ("а", "я", "ша", "жа"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                variants.add(word[:-len(suffix)])
        return variants

    def contains_wake_word(self, text: str) -> bool:
        text_lower = text.lower().strip()
        if self._wake_word in text_lower:
            return True
        for word in text_lower.split():
            word = word.strip(".,!?;:-\"'()[]")
            if word and self._is_fuzzy_match(word):
                return True
        return False

    def strip_wake_word(self, text: str) -> str:
        import re
        pattern = re.compile(re.escape(self._wake_word), re.IGNORECASE)
        if pattern.search(text):
            result = pattern.sub("", text)
            result = re.sub(r"\s+", " ", result).strip()
            return result.strip(".,!?;:- ")
        words = text.split()
        kept = []
        for w in words:
            clean = w.strip(".,!?;:-\"'()[]").lower()
            if clean and self._is_fuzzy_match(clean):
                continue
            kept.append(w)
        result = " ".join(kept).strip()
        return result.strip(".,!?;:- ")

    def _is_fuzzy_match(self, word: str) -> bool:
        ratio = difflib.SequenceMatcher(None, self._wake_word, word).ratio()
        if ratio >= self.FUZZY_THRESHOLD:
            return True
        for variant in self._variants:
            ratio = difflib.SequenceMatcher(None, variant, word).ratio()
            if ratio >= self.FUZZY_THRESHOLD:
                return True
        return False


# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------

PARAKEET_PREFIX = "parakeet:"
PARAKEET_VENV_PYTHON = (
    "/mnt/82A23910A2390A65/Trade/EducationAndHack/VOICE/whisper.cpp/"
    "examples/whisper.youtube/.venv-parakeet/bin/python"
)


class Transcriber:
    """Runs whisper-cli or Parakeet (NeMo) and returns transcribed text."""

    def __init__(self, config: Config):
        self.config = config
        self._parakeet_proc = None
        self._parakeet_model = None

    def transcribe(self, wav_path: str, model: str = None,
                   duration_s: float = 0) -> str:
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        target = model or self.config.model
        if target.startswith(PARAKEET_PREFIX):
            return self._transcribe_parakeet(wav_path, target, duration_s)
        return self._transcribe_whisper(wav_path, target, duration_s)

    def _transcribe_whisper(self, wav_path, model, duration_s):
        cmd = [
            self.config.whisper_cli,
            "-m", model,
            "-f", wav_path,
            "-nt",
            "-np",
            "-t", str(self.config.threads),
            "-l", self.config.language,
            "-dev", str(self.config.gpu_device),
        ]
        log.info("Transcribing whisper (%.1fs): %s", duration_s, " ".join(cmd))

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )

        if result.returncode != 0:
            log.error("whisper-cli stderr: %s", result.stderr)
            raise RuntimeError(f"whisper-cli failed (rc={result.returncode}): {result.stderr[:200]}")

        text = result.stdout.strip()
        text = text.replace("[BLANK_AUDIO]", "").strip()
        if text and _is_hallucination(text, duration_s):
            log.info("Hallucination filtered (%.1fs): %r", duration_s, text[:100])
            return ""
        log.info("Transcription: %r", text[:100])
        return text

    _PARAKEET_WORKER_SCRIPT = (
        "import sys, json, os\n"
        "os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')\n"
        "import nemo.collections.asr as nemo_asr\n"
        "import torch\n"
        "model_name = sys.argv[1]\n"
        "asr = nemo_asr.models.ASRModel.from_pretrained(model_name)\n"
        "if torch.cuda.is_available():\n"
        "    asr = asr.cuda()\n"
        "asr.eval()\n"
        "sys.stdout.write('__PARAKEET_READY__\\n')\n"
        "sys.stdout.flush()\n"
        "for line in sys.stdin:\n"
        "    wav = line.strip()\n"
        "    if not wav: continue\n"
        "    try:\n"
        "        out = asr.transcribe([wav], timestamps=False, verbose=False)\n"
        "        text = out[0].text or ''\n"
        "        sys.stdout.write('__PARAKEET_RESULT__' + json.dumps({'text': text}) + '\\n')\n"
        "    except Exception as e:\n"
        "        sys.stdout.write('__PARAKEET_RESULT__' + json.dumps({'error': str(e)}) + '\\n')\n"
        "    sys.stdout.flush()\n"
    )

    def _ensure_parakeet_worker(self, model_name):
        """Spawn persistent NeMo worker if not running or model changed."""
        if (self._parakeet_proc is not None
                and self._parakeet_proc.poll() is None
                and self._parakeet_model == model_name):
            return

        if self._parakeet_proc is not None and self._parakeet_proc.poll() is None:
            log.info("Stopping parakeet worker (model change)")
            self._parakeet_proc.terminate()
            try:
                self._parakeet_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._parakeet_proc.kill()

        log.info("Starting parakeet worker for model: %s", model_name)
        self._parakeet_proc = subprocess.Popen(
            [PARAKEET_VENV_PYTHON, "-c", self._PARAKEET_WORKER_SCRIPT, model_name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._parakeet_model = model_name

        # wait for ready signal
        while True:
            line = self._parakeet_proc.stdout.readline()
            if not line:
                err = self._parakeet_proc.stderr.read()
                raise RuntimeError(f"parakeet worker died at startup: {err[-500:]}")
            if line.strip() == "__PARAKEET_READY__":
                log.info("Parakeet worker ready")
                return

    def _transcribe_parakeet(self, wav_path, model, duration_s):
        model_name = model[len(PARAKEET_PREFIX):]
        log.info("Transcribing parakeet (%.1fs): model=%s file=%s",
                 duration_s, model_name, wav_path)

        self._ensure_parakeet_worker(model_name)
        self._parakeet_proc.stdin.write(wav_path + "\n")
        self._parakeet_proc.stdin.flush()

        line = self._parakeet_proc.stdout.readline()
        if not line:
            err = self._parakeet_proc.stderr.read()
            raise RuntimeError(f"parakeet worker died: {err[-500:]}")

        import json
        if not line.startswith("__PARAKEET_RESULT__"):
            raise RuntimeError(f"unexpected parakeet output: {line[:200]}")
        data = json.loads(line[len("__PARAKEET_RESULT__"):])
        if "error" in data:
            raise RuntimeError(f"parakeet error: {data['error']}")
        text = data["text"].strip()

        if text and _is_hallucination(text, duration_s):
            log.info("Hallucination filtered (%.1fs): %r", duration_s, text[:100])
            return ""
        log.info("Transcription: %r", text[:100])
        return text

    def shutdown(self):
        """Cleanly stop parakeet worker if running."""
        if self._parakeet_proc is not None and self._parakeet_proc.poll() is None:
            log.info("Shutting down parakeet worker")
            self._parakeet_proc.terminate()
            try:
                self._parakeet_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._parakeet_proc.kill()
