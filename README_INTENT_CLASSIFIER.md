# Whisper.cpp with Fine-tuned BERT Intent Classifier

This repository contains a modified version of [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with an integrated fine-tuned BERT intent classifier for Android applications.

## 🚀 Features

- **Complete End-to-End Intent Classification**: Single TensorFlow Lite model (no separate encoder needed)
- **Fine-tuned BERT Model**: Custom trained on TensorFlow 2.19.0 with dynamic int8 quantization
- **Android Ready**: Optimized for mobile deployment with TFLite
- **13 Intent Classes**: Supports LogEvent, MediaAction, OpenApp, PhoneAction, QueryPoint, QueryTrend, SetGoal, SetThreshold, StartActivity, StopActivity, TimerStopwatch, ToggleFeature, WeatherQuery

## 📱 Android Integration

### Model Files
- `intent_classifier.tflite` - Complete end-to-end intent classifier (int8 quantized)
- `model_metadata.json` - Intent mappings and model configuration
- `tokenizer/` - BERT tokenizer files (vocab.txt, tokenizer.json, etc.)

### Architecture
```
Text Input → BERT Tokenizer → Complete TFLite Model → Intent Classification
```

### Key Features
- **Input**: Raw text strings
- **Tokenization**: BERT-style with [CLS], [SEP], [PAD], [UNK] tokens
- **Max Sequence Length**: 256 tokens
- **Model Type**: Complete end-to-end (includes both encoder and classification head)
- **Quantization**: Dynamic int8 for optimal mobile performance

## 🔧 Implementation Details

### Model Inputs
- **Input 0**: `input_ids` - Tokenized text as int32 array [1, 256]
- **Input 1**: `attention_mask` - Attention mask as int32 array [1, 256]

### Model Output
- **Output**: Intent logits as float32 array [1, 13]

### IntentClassifier.kt
The Android implementation handles:
- BERT tokenization with proper special tokens
- TensorFlow Lite inference with multiple inputs
- Softmax application for probability scores
- Intent mapping to human-readable labels

## 📊 Performance

- **Model Size**: Optimized with int8 quantization
- **Inference Speed**: Single model call (faster than two-stage approaches)
- **Memory Footprint**: Reduced compared to separate encoder + classifier
- **Accuracy**: Maintains high accuracy with quantization

## 🛠️ Usage

### Android Integration
1. Copy model files to `/assets/` directory
2. Use `IntentClassifier.kt` for inference
3. Initialize with `initialize()` method
4. Classify with `classifyIntent(text: String)`

### Example
```kotlin
val classifier = IntentClassifier(context)
classifier.initialize()

val result = classifier.classifyIntent("Set my daily step goal to 10000")
// result.intent = "SetGoal"
// result.confidence = 0.95
```

## 📝 Supported Intent Classes

1. **LogEvent** - Logging activities/events
2. **MediaAction** - Media control (volume, play, pause)
3. **OpenApp** - Opening applications
4. **PhoneAction** - Phone-related actions (calls, etc.)
5. **QueryPoint** - Point-in-time queries (current status)
6. **QueryTrend** - Trend/historical queries
7. **SetGoal** - Setting goals/targets
8. **SetThreshold** - Setting thresholds/limits
9. **StartActivity** - Starting activities/workouts
10. **StopActivity** - Stopping activities
11. **TimerStopwatch** - Timer/stopwatch operations
12. **ToggleFeature** - Enabling/disabling features
13. **WeatherQuery** - Weather-related queries

## 🎯 Model Training

- **Base Model**: BERT (fine-tuned)
- **Framework**: TensorFlow 2.19.0
- **Quantization**: Dynamic int8 for mobile optimization
- **Training**: Custom dataset with 13 intent classes
- **Tokenizer**: BERT tokenizer with 30,522 vocabulary size

## 📄 Files Changed

### Key Android Files
- `examples/whisper.android/app/src/main/java/com/whispercppdemo/intent/IntentClassifier.kt`
- `examples/whisper.android/app/src/main/assets/intent_classifier.tflite`
- `examples/whisper.android/app/src/main/assets/model_metadata.json`
- `examples/whisper.android/app/src/main/assets/tokenizer/`

### Backup Files
Original files are backed up in `/assets/backup/` for reference.

## 🔍 Testing

Test examples:
```
"decrease volume" → MediaAction
"what's my heart rate" → QueryPoint  
"enable DND" → ToggleFeature
"set daily step goal to 10000" → SetGoal
```

## 🚀 Deployment

Ready for production Android deployment with:
- Optimized model size
- Fast inference
- Low memory usage  
- High accuracy intent classification

## 📋 Requirements

- Android API Level 21+
- TensorFlow Lite for Android
- ~30MB additional storage for model files

---

*Based on whisper.cpp by Georgi Gerganov with custom intent classification integration.*