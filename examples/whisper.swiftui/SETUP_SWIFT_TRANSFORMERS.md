# Swift Transformers Setup Guide

## Installation Instructions

### Option 1: Swift Package Manager (Recommended)

1. **Open Xcode project**: Open `whisper.swiftui.xcodeproj` in Xcode
2. **Add Package Dependencies**:
   - Go to `File` → `Add Package Dependencies...`
   - Add the following URLs:
     - `https://github.com/huggingface/swift-transformers`
     - `https://github.com/tensorflow/swift-apis` (for TensorFlow support)

3. **Configure Package Dependencies**:
   - Select `swift-transformers` for the `Transformers` framework
   - Select appropriate version (latest stable)

### Option 2: Manual Package.swift Integration

If using the Package.swift approach:

```bash
# From the whisper.swiftui directory
swift package resolve
swift package update
```

## Required Files Structure

Ensure your bundle includes these files:

```
Resources/
├── intent_classifier.tflite    # TensorFlow Lite model
├── label_encoder.json         # Intent label mappings
└── tokenizer/                 # BERT tokenizer files
    ├── tokenizer.json        # Main tokenizer configuration
    ├── tokenizer_config.json # Tokenizer metadata
    └── vocab.txt             # Vocabulary file
```

## Usage

The updated `IntentClassifier` now uses proper BERT tokenization:

1. **Proper WordPiece tokenization** with special tokens ([CLS], [SEP])
2. **Attention masking** for variable-length sequences
3. **Automatic padding/truncation** to 256 tokens
4. **Full compatibility** with Hugging Face BERT models

## Troubleshooting

If you encounter initialization errors:

1. **Check bundle resources**: Verify all files are included in the Xcode project target
2. **Verify tokenizer files**: Ensure the tokenizer directory contains all required files
3. **Check dependencies**: Ensure Swift Transformers is properly linked
4. **Review logs**: Check the console for detailed error messages

## Key Improvements

- ✅ **Proper BERT tokenization** using Swift Transformers
- ✅ **WordPiece tokenization** with correct special tokens
- ✅ **Attention masking** for better model performance
- ✅ **Padding and truncation** handling
- ✅ **Error handling** and debugging support