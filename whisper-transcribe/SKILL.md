---
name: whisper
description: >
  Speech-to-text transcription using whisper.cpp. Use this skill whenever the user
  wants to transcribe audio, convert speech to text, extract text from audio/video
  files, or mentions whisper, STT, or speech recognition. Also use when the user
  has audio files (.wav, .mp3, .ogg, .flac, .m4a, .mp4, .webm) they want to
  process into text.
---

# Whisper Speech-to-Text Skill

Transcribe audio files using whisper.cpp. Two backends: HTTP server (Docker) and local CLI.

## Step 1 — Detect Backend (run ONCE per session)

**IMPORTANT: Only run detection once.** If you already determined the backend earlier in this conversation, or if the caller told you which backend to use, skip directly to Step 2.

Run this single command:

```bash
curl -s -o /dev/null -w "http=%{http_code}" "${WHISPER_CPP_URL:-http://localhost:8080}" --max-time 3 2>/dev/null; command -v whisper-cli >/dev/null 2>&1 && test -f "$WHISPER_CPP_MODEL" 2>/dev/null && echo " cli=ready" || echo " cli=no"
```

**Pick the first that matches:**

1. Output contains `http=200` → use **HTTP mode** (Step 3)
2. Output contains `cli=ready` → use **CLI mode** (Step 4)
3. Neither → read `references/local-setup.md` and `references/docker-setup.md`, help the user set up one backend

Remember which mode you chose. Do NOT re-run this check for subsequent files.

**Orchestration:** If you are delegating transcription to subagents, pass the detected backend to them (e.g. "Use HTTP mode at `<url>`" or "Use CLI mode") so they skip detection entirely.

## Step 2 — Find Audio Files

If the user did not specify exact files, find them:

```bash
find /path/to/search -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.ogg" -o -name "*.m4a" -o -name "*.mp4" -o -name "*.webm" \) 2>/dev/null
```

Use the actual directory the user mentioned. Present the list and confirm before transcribing.

## Step 3 — Transcribe via HTTP or CLI

If using HTTP mode refer to Step 3: HTTP Mode.
If using CLI mode refer to Step 3: CLI Mode.

### HTTP Mode
**Convert non-WAV files first** (the HTTP server only accepts WAV):

```bash
ffmpeg -i INPUT -ar 16000 -ac 1 -c:a pcm_s16le /tmp/whisper_input.wav
```

**Transcribe:**

```bash
curl -s -F "file=@/absolute/path/to/audio.wav" -F "response_format=text" "${WHISPER_CPP_URL:-http://localhost:8080}/inference"
```

The response body is the plain transcription text. Process files one at a time — the server serializes requests.

### CLI Mode

**The CLI natively supports: wav, mp3, flac, ogg.** No conversion needed for these.

For other formats (m4a, mp4, webm), convert first:

```bash
ffmpeg -i INPUT -ar 16000 -ac 1 -c:a pcm_s16le /tmp/whisper_input.wav
```

**Transcribe:**

```bash
whisper-cli -m "$WHISPER_CPP_MODEL" -f /absolute/path/to/audio.wav -np
```

Flags: `-m` model path, `-f` input file (absolute path), `-np` suppresses debug output (critical — always include).

Add `--no-timestamps` if the user wants plain text without `[HH:MM:SS.mmm --> HH:MM:SS.mmm]` prefixes.

## Step 4 — Present Output

For **every** file, show the complete transcription text. Never abbreviate, summarize, or write "(same as above)" — even if multiple files produce identical text. If timestamps are present, keep them formatted readably.

## Errors

| Error | Fix |
| ------- | ----- |
| curl returns non-200 or connection refused | Server not running — read `references/docker-setup.md` |
| `whisper-cli: command not found` | Not built — read `references/local-setup.md` |
| `failed to open model` | Model missing — run: `cd whisper.cpp && sh models/download-ggml-model.sh base.en` |
| `ffmpeg: command not found` | Install: `sudo apt install ffmpeg` (Linux) or `brew install ffmpeg` (macOS) |

## Tips

- For advanced flags (diarization, VAD, language detection, translate), run `whisper-cli -h`
- Use `--language auto` for non-English audio with `large-v3` model
- Use `--translate` to translate non-English audio to English
