# Quick Setup Guide - iOS Intent Classification

## 🚀 Quick Start (5 Minutes)

### Step 1: Add TensorFlow Lite to Your Project

In Xcode, go to **File → Add Package Dependencies** and add:
```
https://github.com/tensorflow/tensorflow
```

### Step 2: Copy Model Files

Copy these files to your Xcode project bundle:
- `intent_classifier.tflite`
- `lightweight_sentence_encoder.tflite` 
- `model_metadata.json`
- `vocabulary.json`

### Step 3: Add Swift Files

Copy these 4 Swift files to your project:
- `IntentModels.swift`
- `IntentClassifier.swift`
- `SlotExtractor.swift`
- `IntentTestView.swift`

### Step 4: Update Your App

Add to your main view:

```swift
TabView {
    // Your existing content
    
    IntentTestView()
        .tabItem {
            Image(systemName: "brain.head.profile")
            Text("Intent Test")
        }
}
```

### Step 5: Test

Run your app and test with commands like:
- "How many steps did I take today?"
- "Set my daily step goal to 10000"
- "Turn on do not disturb"

## 🎯 Expected Results

Each command will return:
- **Intent**: The classified intent type
- **Confidence**: Classification confidence (0-100%)
- **Slots**: Extracted parameters with values
- **Slot Confidence**: Slot extraction confidence

Example:
```
Input: "How many steps did I take today?"
→ Intent: QueryPoint (95%)
→ Slots: {metric: "steps", time_ref: "today"} (100%)
```

That's it! You now have a complete NLU system in your iOS app! 🎉