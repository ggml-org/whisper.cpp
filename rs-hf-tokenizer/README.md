# HuggingFace Tokenizer for Android Intent Classifier

This Rust-based tokenizer provides proper BERT WordPiece tokenization for your Android intent classifier, replacing the basic word-level tokenizer that was reducing accuracy.

## Overview

The previous tokenizer implementation was doing simple word-level tokenization, which doesn't match how BERT models were trained. BERT models expect **WordPiece tokenization** which:

- Splits words into subword tokens (e.g., "playing" → ["play", "##ing"])
- Handles out-of-vocabulary words better
- Maintains the exact tokenization strategy used during model training

This Rust implementation uses the HuggingFace `tokenizers` library to provide identical tokenization to what your model expects.

## Setup Instructions

### 1. Install Prerequisites

**Rust:**
```powershell
# Install Rust if not already installed
winget install Rustlang.Rust.MSVC

# Add Android targets
rustup target add aarch64-linux-android
rustup target add armv7-linux-androideabi 
rustup target add i686-linux-android
rustup target add x86_64-linux-android
```

**Android NDK:**
- Install Android NDK through Android Studio or standalone
- Note the NDK path (e.g., `C:/Users/YourUsername/AppData/Local/Android/Sdk/ndk/26.1.10909125`)

### 2. Configure Build Environment

1. Edit `.cargo/config.toml` and replace `PATH_TO_NDK` with your actual NDK path
2. Make sure the NDK version in the paths matches your installed version

### 3. Build the Native Libraries

```powershell
cd rs-hf-tokenizer
.\build_android.bat
```

This will generate `.so` files for all Android architectures:
- `target/aarch64-linux-android/release/libhftokenizer.so` (arm64-v8a)
- `target/armv7-linux-androideabi/release/libhftokenizer.so` (armeabi-v7a)
- `target/i686-linux-android/release/libhftokenizer.so` (x86)
- `target/x86_64-linux-android/release/libhftokenizer.so` (x86_64)

### 4. Integrate with Android Project

1. Create the jniLibs directory structure in your Android app:
   ```
   app/src/main/jniLibs/
   ├── arm64-v8a/libhftokenizer.so
   ├── armeabi-v7a/libhftokenizer.so
   ├── x86/libhftokenizer.so
   └── x86_64/libhftokenizer.so
   ```

2. Copy the compiled `.so` files to the appropriate directories:
   ```powershell
   # Example commands (adjust paths as needed)
   copy target\aarch64-linux-android\release\libhftokenizer.so ..\whisper.cpp-master\examples\whisper.android\app\src\main\jniLibs\arm64-v8a\
   copy target\armv7-linux-androideabi\release\libhftokenizer.so ..\whisper.cpp-master\examples\whisper.android\app\src\main\jniLibs\armeabi-v7a\
   copy target\i686-linux-android\release\libhftokenizer.so ..\whisper.cpp-master\examples\whisper.android\app\src\main\jniLibs\x86\
   copy target\x86_64-linux-android\release\libhftokenizer.so ..\whisper.cpp-master\examples\whisper.android\app\src\main\jniLibs\x86_64\
   ```

### 5. Android Project Configuration

The `IntentClassifier.kt` and `HFTokenizer.kt` files have already been updated to use the new tokenizer. No additional changes needed in your Android code.

## Key Improvements

### Before (Basic Word Tokenizer)
- Simple word splitting on whitespace
- No subword tokenization
- Mismatched tokenization vs. model training
- Lower accuracy due to vocabulary misalignment

### After (HuggingFace BERT Tokenizer)
- Proper WordPiece tokenization
- Exact match with model training tokenization
- Better handling of out-of-vocabulary words
- Significantly improved accuracy

## Expected Results

With proper tokenization, you should see:
- **Higher intent classification accuracy**
- **Better handling of unseen words and phrases**
- **More consistent predictions**
- **Improved confidence scores**

## Troubleshooting

### Build Issues
- Ensure NDK paths are correct in `.cargo/config.toml`
- Check that Rust Android targets are installed: `rustup target list | grep android`
- Verify NDK version compatibility

### Runtime Issues
- Ensure all `.so` files are in the correct jniLibs directories
- Check Android logs for native library loading errors
- Verify `tokenizer.json` file exists in assets/tokenizer/

### Testing
Test with the same phrases that were previously misclassified to verify the improvement.

## Architecture

```
Text Input 
    ↓
Rust HF Tokenizer (WordPiece)
    ↓
TensorFlow Lite Model (BERT + Classification Head)
    ↓
Intent Classification Output
```

The Rust tokenizer now provides the exact same tokenization that your BERT model was trained with, ensuring optimal accuracy.