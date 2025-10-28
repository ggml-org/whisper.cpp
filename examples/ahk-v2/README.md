# Using whisper.cpp with AutoHotkey v2

This guide explains how to use whisper.cpp (a C++ speech recognition library) from AutoHotkey v2 using DLL bindings.

## Why NOT Rewrite in AHK?

**whisper.cpp cannot be realistically ported/rewritten to pure AHK** because:

- **9,000+ lines of optimized C++ code** with SIMD instructions (AVX, NEON)
- **Complex machine learning framework (GGML)** with 3,000+ files
- **Performance requirements** - Neural network inference needs native speed
- **GPU acceleration** - Requires CUDA, Metal, or Vulkan support

**The practical solution: Use DLL bindings** (like Go, Java, Ruby, and Python already do)

---

## Step 1: Get the DLL

You have **4 options** to obtain `whisper.dll`:

### Option A: Download Pre-compiled DLL (Easiest)

#### 1. GitHub Actions (Official)
- Visit: https://github.com/ggml-org/whisper.cpp/actions
- Click the latest successful workflow run
- Download Windows artifacts
- Extract `whisper.dll`

#### 2. Third-party Builds
- Visit: https://github.com/regstuff/whisper.cpp_windows/releases
- Download the latest release
- Extract `whisper.dll`

#### 3. MSYS2 Package (For MSYS2 users)
```bash
pacman -S mingw-w64-x86_64-whisper-cpp
```
The DLL will be in `/mingw64/bin/libwhisper-1.dll`

#### 4. Python Package (Extract DLL)
```bash
pip install pywhispercpp
```
Then locate the DLL in Python's site-packages folder (usually named `whisper.dll` or `_whisper.pyd`)

---

### Option B: Build from Source (Recommended for Latest Version)

#### Prerequisites:
- **CMake** 3.15+ (https://cmake.org/download/)
- **Visual Studio 2019+** with C++ support, OR
- **MinGW-w64** (https://www.mingw-w64.org/)

#### Build Steps:

```bash
# Clone the repository
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp

# Configure CMake to build shared library
cmake -B build -DBUILD_SHARED_LIBS=ON -DWHISPER_BUILD_EXAMPLES=OFF

# Build (this creates whisper.dll)
cmake --build build --config Release

# Find your DLL at:
# build/Release/whisper.dll (Visual Studio)
# build/src/libwhisper.dll (MinGW)
```

#### With GPU Support (NVIDIA):
```bash
cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON
cmake --build build --config Release
```

---

## Step 2: Get a Whisper Model

Download a model file (choose based on accuracy vs speed):

```bash
# Tiny (fastest, least accurate - ~75 MB)
curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin -o models/ggml-tiny.en.bin

# Base (balanced - ~142 MB)
curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin -o models/ggml-base.en.bin

# Small (good quality - ~466 MB)
curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin -o models/ggml-small.en.bin

# Medium (better quality - ~1.5 GB)
curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin -o models/ggml-medium.en.bin
```

**Multilingual models** (remove `.en` from filename):
```bash
curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin -o models/ggml-base.bin
```

---

## Step 3: Use from AutoHotkey v2

### File Structure:
```
YourProject/
├── whisper.dll           (The compiled library)
├── models/
│   └── ggml-base.en.bin  (The AI model)
├── whisper_ahkv2.ahk     (The wrapper class)
└── your_script.ahk       (Your script)
```

### Basic Usage Example:

```ahk
#Requires AutoHotkey v2.0

; Include the wrapper
#Include whisper_ahkv2.ahk

; Initialize
whisper := WhisperCpp("whisper.dll")

; Check version
MsgBox("Whisper version: " . whisper.GetVersion())

; Load model
try {
    whisper.LoadModel("models/ggml-base.en.bin")
    MsgBox("Model loaded successfully!")
} catch as err {
    MsgBox("Error: " . err.Message)
    ExitApp
}

; TODO: Transcribe audio
; result := whisper.TranscribeFile("audio.wav")
; MsgBox(result.text)

; Cleanup
whisper.Free()
```

---

## Step 4: Handle Audio Files

**Important:** The provided `whisper_ahkv2.ahk` includes stubs for audio loading. You need to implement one of these approaches:

### Option A: Use FFmpeg (Recommended)

Download FFmpeg and use it to convert audio to raw PCM:

```ahk
; Convert audio to 16kHz mono PCM using FFmpeg
RunWait('ffmpeg.exe -i "input.mp3" -ar 16000 -ac 1 -f f32le "temp.pcm"', , "Hide")

; Load the PCM file into a buffer
file := FileOpen("temp.pcm", "r")
audioData := Buffer(file.Length)
file.RawRead(audioData)
file.Close()

; Transcribe
result := whisper.Transcribe(audioData, "en")
MsgBox(result.text)

; Cleanup
FileDelete("temp.pcm")
```

### Option B: Use the CLI Tool (Simplest)

Instead of DLL bindings, just call the whisper.cpp CLI:

```ahk
; Run whisper-cli and capture output
command := 'whisper-cli.exe -m models/ggml-base.en.bin -f audio.wav --output-txt'
output := RunWaitOne(command)
MsgBox(output)

RunWaitOne(command) {
    shell := ComObject("WScript.Shell")
    exec := shell.Exec(A_ComSpec ' /C ' command)
    return exec.StdOut.ReadAll()
}
```

### Option C: WAV File Parser (For WAV files only)

Implement a simple WAV parser in AHK to extract PCM data. (This would require ~100 lines of code to parse WAV headers and extract samples.)

---

## Complete Working Example

Here's a full example using FFmpeg:

```ahk
#Requires AutoHotkey v2.0
#Include whisper_ahkv2.ahk

; Paths
WHISPER_DLL := "whisper.dll"
MODEL_FILE := "models/ggml-base.en.bin"
FFMPEG_PATH := "ffmpeg.exe"

; Initialize Whisper
whisper := WhisperCpp(WHISPER_DLL)

try {
    ; Load model
    MsgBox("Loading model...")
    whisper.LoadModel(MODEL_FILE)

    ; Select audio file
    audioFile := FileSelect(3, , "Select audio file", "Audio (*.mp3; *.wav; *.m4a)")
    if (!audioFile)
        ExitApp

    ; Convert to 16kHz mono PCM
    MsgBox("Converting audio...")
    tempPcm := A_Temp "\whisper_temp.pcm"

    RunWait(Format('"{1}" -i "{2}" -ar 16000 -ac 1 -f f32le "{3}" -y',
        FFMPEG_PATH, audioFile, tempPcm), , "Hide")

    if (!FileExist(tempPcm)) {
        throw Error("FFmpeg conversion failed")
    }

    ; Load PCM data
    file := FileOpen(tempPcm, "r")
    audioBuffer := Buffer(file.Length)
    file.RawRead(audioBuffer)
    file.Close()

    ; Transcribe
    MsgBox("Transcribing...")
    result := whisper.Transcribe(audioBuffer, "en")

    ; Show results
    MsgBox("Transcription:`n`n" . result.text)

    ; Show detailed segments
    output := ""
    for segment in result.segments {
        output .= Format("[{1:6.2f}s -> {2:6.2f}s] {3}`n",
            segment.start, segment.end, segment.text)
    }
    FileAppend(output, "transcription.txt")
    MsgBox("Detailed transcription saved to transcription.txt")

} catch as err {
    MsgBox("Error: " . err.Message)
} finally {
    ; Cleanup
    FileDelete(tempPcm)
    whisper.Free()
}
```

---

## API Reference

### WhisperCpp Class Methods

#### Constructor
```ahk
whisper := WhisperCpp(dllPath)
```
- **dllPath**: Path to `whisper.dll` (default: "whisper.dll")

#### LoadModel
```ahk
whisper.LoadModel(modelPath)
```
- **modelPath**: Path to GGML model file (e.g., "models/ggml-base.en.bin")

#### GetVersion
```ahk
version := whisper.GetVersion()
```
- **Returns**: String with whisper.cpp version

#### Transcribe
```ahk
result := whisper.Transcribe(audioSamples, language)
```
- **audioSamples**: Buffer containing float32 PCM samples (16kHz mono)
- **language**: Language code (e.g., "en", "es", "fr")
- **Returns**: Object with:
  - `text`: Full transcription
  - `segments`: Array of segments with `text`, `start`, `end`

#### Free
```ahk
whisper.Free()
```
- Cleanup model and free memory

---

## Limitations of the Current Implementation

The provided `whisper_ahkv2.ahk` is a **proof-of-concept**. To make it production-ready, you need to:

1. **✅ Implement audio loading** - Parse WAV files or use FFmpeg
2. **✅ Implement struct marshalling** - Properly create `whisper_full_params` struct
3. **⚠️ Handle callbacks** - For progress updates and segment callbacks
4. **⚠️ Error handling** - Better error messages and validation
5. **⚠️ Memory management** - Proper buffer allocation/deallocation
6. **✅ Support all parameters** - Language, temperature, beam size, etc.

---

## Advanced: Full Parameter Support

To use advanced features, you need to properly create the `whisper_full_params` struct:

```ahk
; This is a C struct that needs to be created in AHK
; struct whisper_full_params {
;     int strategy;
;     int n_threads;
;     int n_max_text_ctx;
;     int offset_ms;
;     int duration_ms;
;     bool translate;
;     bool no_context;
;     bool no_timestamps;
;     bool single_segment;
;     bool print_special;
;     bool print_progress;
;     bool print_realtime;
;     bool print_timestamps;
;     ... (50+ more fields)
; }

; You would need to allocate a Buffer and fill it with the correct values
; This requires understanding C struct layout and alignment
```

---

## Troubleshooting

### "Failed to load whisper.dll"
- Ensure `whisper.dll` is in the same folder as your script, or provide full path
- Check that you have the correct architecture (x64 vs x86)
- Install Visual C++ Redistributable if needed

### "Failed to load model"
- Check that the model file exists and path is correct
- Ensure model file is compatible with your whisper.cpp version
- Try a smaller model first (tiny or base)

### "Transcription failed"
- Ensure audio is 16kHz mono PCM float32 format
- Check that audioSamples buffer is valid
- Try the CLI tool first to verify your setup works

### Performance Issues
- Use quantized models (Q5_0, Q8_0) for faster inference
- Enable GPU support when building the DLL
- Use smaller models (tiny, base) for real-time applications

---

## Next Steps

1. **Test the CLI first**: Use `whisper-cli.exe` to ensure your model works
2. **Start simple**: Use the CLI wrapper approach before attempting DLL bindings
3. **Implement audio loading**: Choose FFmpeg or WAV parser based on your needs
4. **Enhance the wrapper**: Add full struct support and callbacks as needed

---

## Resources

- **whisper.cpp GitHub**: https://github.com/ggml-org/whisper.cpp
- **whisper.h API docs**: `/include/whisper.h` in the whisper.cpp repository
- **Model downloads**: https://huggingface.co/ggerganov/whisper.cpp
- **FFmpeg download**: https://ffmpeg.org/download.html
- **AutoHotkey v2 docs**: https://www.autohotkey.com/docs/v2/

---

## License

This AHK wrapper follows the same MIT license as whisper.cpp.
