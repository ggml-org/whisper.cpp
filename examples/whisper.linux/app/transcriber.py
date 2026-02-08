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

class Transcriber:
    """Runs whisper-cli and returns transcribed text."""

    def __init__(self, config: Config):
        self.config = config

    def transcribe(self, wav_path: str, model: str = None,
                   duration_s: float = 0) -> str:
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        cmd = [
            self.config.whisper_cli,
            "-m", model or self.config.model,
            "-f", wav_path,
            "-nt",
            "-np",
            "-t", str(self.config.threads),
            "-l", self.config.language,
            "-dev", str(self.config.gpu_device),
        ]
        log.info("Transcribing (%.1fs): %s", duration_s, " ".join(cmd))

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
