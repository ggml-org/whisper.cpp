"""Voice command processing for whisper.linux.

Detects command words in transcribed text (e.g. "Enter", "Backspace")
and executes corresponding key presses instead of typing them literally.
"""

import difflib
from typing import Callable, Optional

from .config import DEFAULT_VOICE_COMMANDS, log


class VoiceCommands:
    """Processes transcribed text for voice commands (Enter, Backspace, etc.)."""

    DEFAULT_COMMANDS = DEFAULT_VOICE_COMMANDS
    FUZZY_THRESHOLD = 0.75

    def __init__(self, commands: dict = None):
        self._commands = commands if commands is not None else dict(self.DEFAULT_COMMANDS)

    def process(self, text: str, inject_fn: Callable, send_key_fn: Callable) -> bool:
        """Process text, calling inject_fn for text and send_key_fn for key presses.

        For "backspace", removes the previous word from the buffer. If the buffer
        is empty, sends Ctrl+BackSpace to delete the word in the editor.

        Returns True if any commands were found.
        """
        words = text.split()
        if not words:
            return False

        buffer = []
        had_commands = False

        for word in words:
            clean = word.strip(".,!?;:-\"'()[]").lower()
            action = self._match_command(clean)
            if action:
                had_commands = True
                if action == "backspace":
                    if buffer:
                        removed = buffer.pop()
                        log.info("Voice command: backspace (removed '%s' from buffer)", removed)
                    else:
                        send_key_fn("ctrl+BackSpace")
                        log.info("Voice command: backspace (Ctrl+BackSpace sent)")
                else:
                    # Flush text buffer first, then send key
                    if buffer:
                        inject_fn(" ".join(buffer))
                        buffer.clear()
                    key = action[4:]  # strip "key:" prefix
                    send_key_fn(key)
                    log.info("Voice command: %s -> %s", clean, key)
            else:
                buffer.append(word)

        # Flush remaining text
        if buffer:
            inject_fn(" ".join(buffer))

        return had_commands

    def _match_command(self, word: str) -> Optional[str]:
        """Match a word to a command (exact, then fuzzy)."""
        if not word:
            return None
        # Exact match
        if word in self._commands:
            return self._commands[word]
        # Fuzzy match — pick best above threshold
        best_ratio = 0.0
        best_action = None
        for cmd_word, action in self._commands.items():
            ratio = difflib.SequenceMatcher(None, cmd_word, word).ratio()
            if ratio >= self.FUZZY_THRESHOLD and ratio > best_ratio:
                best_ratio = ratio
                best_action = action
        return best_action
