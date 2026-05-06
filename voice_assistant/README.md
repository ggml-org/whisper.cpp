# Local Voice Assistant

A fully local, offline voice assistant pipeline for macOS.

```
Microphone → Whisper large-v3 (STT) → Ollama qwen2.5:7b (LLM) → Piper (TTS) → afplay
```

Supports **Chinese and English** with automatic language detection.

---

## Requirements

- macOS (Apple Silicon or Intel)
- Python 3.10+
- [Ollama](https://ollama.com)
- Piper TTS voice model (downloaded separately)

---

## 1 — Python Setup

```bash
cd voice_assistant

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Apple Silicon note:** `faster-whisper` uses CPU int8 by default on arm64,
> which is fast and memory-efficient with no extra setup needed.

---

## 2 — Install & Start Ollama

```bash
# Install via the official installer
brew install ollama

# Or download from https://ollama.com/download

# Start the Ollama server (keep this running in a separate terminal)
ollama serve
```

### Pull the default model

```bash
ollama pull qwen2.5:7b
```

You can use any other model by editing `OLLAMA_MODEL` in `main.py`.

---

## 3 — Piper TTS Setup

Piper needs a voice model file (`.onnx`) downloaded separately.

### Step 1 — Create the models directory

```bash
mkdir -p ~/piper_models
```

### Step 2 — Download a voice model

Choose a voice from the [Piper voice list](https://github.com/rhasspy/piper/blob/master/VOICES.md).

**English (US) — recommended:**
```bash
cd ~/piper_models

# Download model + config
curl -LO https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en-us-lessac-medium.tar.gz
tar -xzf voice-en-us-lessac-medium.tar.gz

# The .onnx file will be at ~/piper_models/en_US-lessac-medium.onnx
```

**Chinese (Mandarin) — alternative:**
```bash
cd ~/piper_models
curl -LO https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-zh-cn-huayan-medium.tar.gz
tar -xzf voice-zh-cn-huayan-medium.tar.gz
```

### Step 3 — Update the model path in `main.py`

```python
# main.py line ~35
PIPER_MODEL = os.path.expanduser("~/piper_models/en_US-lessac-medium.onnx")
```

Change the filename to match whichever model you downloaded.

---

## 4 — Run the Assistant

Make sure the Ollama server is running (`ollama serve`), then:

```bash
source .venv/bin/activate   # if not already active
python main.py
```

**Interaction:**

1. Press **Enter** to start recording
2. Speak for up to 6 seconds
3. The assistant transcribes, queries the LLM, and speaks the response
4. Press **Ctrl+C** to quit

---

## 5 — Configuration

All options are at the top of `main.py`:

| Variable | Default | Description |
|---|---|---|
| `RECORD_SECONDS` | `6` | Max recording length per turn |
| `WHISPER_MODEL_SIZE` | `large-v3` | Any faster-whisper model size |
| `WHISPER_LANGUAGE` | `None` | `None` = auto, or `"zh"` / `"en"` |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Any model pulled in Ollama |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `PIPER_MODEL` | `~/piper_models/en_US-lessac-medium.onnx` | Piper voice model path |

---

## 6 — Troubleshooting

**No audio input / `sounddevice` error:**
```bash
# List available audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

**Ollama not reachable:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags
# If not, start it:
ollama serve
```

**TTS skipped (model not found):**
Check that `PIPER_MODEL` in `main.py` points to an existing `.onnx` file.

**Whisper downloading the model on first run:**
`faster-whisper` downloads `large-v3` (~3 GB) on first use and caches it in
`~/.cache/huggingface/hub`. Subsequent runs are instant.

---

## 7 — Future Improvements

See the TODO comments at the top of `main.py`:

- **Streaming transcription** — lower first-word latency
- **VAD (Voice Activity Detection)** — auto-stop recording when silence is detected
- **Interruptible TTS** — stop playback when the user starts speaking
- **Real-time streaming** — live ASR + streaming LLM output
- **Chat history / memory** — pass prior turns to Ollama for context
- **Wake word** — trigger recording with a keyword instead of Enter
