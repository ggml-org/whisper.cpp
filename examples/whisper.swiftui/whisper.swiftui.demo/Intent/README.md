# iOS Intent Classification & Slot Extraction Implementation

This directory contains a complete iOS implementation of the Intent Classification and Slot Extraction system using SwiftUI and TensorFlow Lite, matching the Android implementation functionality.

## 🏗️ Architecture Overview

### Core Components

1. **IntentModels.swift** - Data structures and error types
2. **IntentClassifier.swift** - Main TensorFlow Lite inference engine
3. **SlotExtractor.swift** - Comprehensive slot extraction system
4. **IntentTestView.swift** - SwiftUI interface for testing

### Key Features

- ✅ **Intent Classification**: 13 intent types with TensorFlow Lite
- ✅ **Slot Extraction**: 14+ slot types with pattern matching
- ✅ **SwiftUI Interface**: Modern iOS UI with comprehensive testing
- ✅ **Real-time Processing**: Async/await for smooth performance
- ✅ **Comprehensive Logging**: Detailed logging for debugging

## 📱 Requirements

### System Requirements
- iOS 15.0+
- Xcode 13.0+
- Swift 5.5+

### Dependencies
- TensorFlow Lite Swift (2.12.0 recommended for compatibility)
- SwiftUI
- Foundation

### Required Assets
Copy these files from the Android implementation to your iOS app bundle:

```
Assets/
├── intent_classifier.tflite           (21KB)
├── lightweight_sentence_encoder.tflite (657KB)
├── model_metadata.json
└── vocabulary.json
```

## 🚀 Installation & Setup

### 1. Add TensorFlow Lite Dependency

Add to your Xcode project using Swift Package Manager:

```
https://github.com/tensorflow/tensorflow
```

Or in `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/tensorflow/tensorflow", from: "2.12.0")
]
```

### 2. Copy Model Files

1. Copy the TensorFlow Lite models from the Android implementation:
   ```bash
   cp examples/whisper.android/app/src/main/assets/*.tflite ./
   cp examples/whisper.android/app/src/main/assets/*.json ./
   ```

2. Add these files to your Xcode project bundle

### 3. Add Intent Classification Files

Copy all Swift files from the `Intent/` directory to your Xcode project:

```
Intent/
├── IntentModels.swift
├── IntentClassifier.swift
├── SlotExtractor.swift
└── IntentTestView.swift
```

### 4. Update Your Main View

Update your app's main view to include the IntentTestView:

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

## 💻 Usage Examples

### Basic Intent Classification

```swift
import SwiftUI

struct MyView: View {
    @StateObject private var intentClassifier = IntentClassifier()
    
    var body: some View {
        VStack {
            Button("Test Intent") {
                Task {
                    let result = await intentClassifier.classifyIntent("How many steps did I take today?")
                    print("Intent: \(result?.intent ?? "Unknown")")
                    print("Slots: \(result?.slots ?? [:])")
                }
            }
        }
        .task {
            await intentClassifier.initialize()
        }
    }
}
```

### Advanced Usage with Slot Extraction

```swift
func processVoiceCommand(_ text: String) async {
    guard let result = await intentClassifier.classifyIntent(text) else {
        print("Classification failed")
        return
    }
    
    print("🎯 Intent: \(result.intent)")
    print("🎯 Confidence: \(Int(result.confidence * 100))%")
    
    if !result.slots.isEmpty {
        print("🏷️ Extracted Slots:")
        for (key, value) in result.slots {
            print("  - \(key): \(value)")
        }
        print("🏷️ Slot Confidence: \(Int(result.slotConfidence * 100))%")
    }
    
    // Process based on intent and slots
    switch result.intent {
    case "QueryPoint":
        if let metric = result.slots["metric"] as? String,
           let timeRef = result.slots["time_ref"] as? String {
            await queryHealthData(metric: metric, timeRef: timeRef)
        }
    case "SetGoal":
        if let metric = result.slots["metric"] as? String,
           let target = result.slots["target"] as? Int {
            await setGoal(metric: metric, target: target)
        }
    default:
        print("Unhandled intent: \(result.intent)")
    }
}
```

## 🏷️ Supported Intents & Slots

### Intent Types (13 Total)
- **QueryPoint** - Query health metrics
- **SetGoal** - Set health/fitness goals
- **SetThreshold** - Set health thresholds
- **TimerStopwatch** - Timer/stopwatch operations
- **ToggleFeature** - Toggle device features
- **LogEvent** - Log health events
- **StartActivity** - Start fitness activities
- **StopActivity** - Stop fitness activities
- **OpenApp** - Open applications
- **PhoneAction** - Phone operations
- **MediaAction** - Media controls
- **WeatherQuery** - Weather information
- **QueryTrend** - Query health trends

### Slot Types (14+ Total)
- **metric** - Health metrics (steps, heart rate, etc.)
- **time_ref** - Time references (today, yesterday, etc.)
- **unit** - Units (bpm, kg, km, etc.)
- **qualifier** - Qualifiers (minimum, maximum, average)
- **threshold** - Numeric thresholds
- **target** - Goal targets
- **value** - Numeric values
- **feature** - Device features
- **state** - States (on/off, increase/decrease)
- **action** - Actions (set, start, stop, etc.)
- **tool** - Tools (timer, alarm, stopwatch)
- **activity_type** - Activity types
- **app** - Applications
- **contact** - Contacts
- **location** - Locations
- **attribute** - Attributes
- **type** - Types
- **period** - Time periods
- **event_type** - Event types

## 🧪 Testing Examples

### Voice Command Examples with Expected Results

```swift
// Query Examples
"How many steps did I take today?"
// → Intent: QueryPoint, Slots: {metric: "steps", time_ref: "today"}

"What's my average heart rate yesterday?"
// → Intent: QueryPoint, Slots: {metric: "heart rate", qualifier: "average", time_ref: "yesterday"}

// Goal Setting Examples
"Set my daily step goal to 10000"
// → Intent: SetGoal, Slots: {metric: "steps", target: 10000, unit: "count"}

// Feature Control Examples
"Turn on do not disturb"
// → Intent: ToggleFeature, Slots: {feature: "do not disturb", state: "on"}

// Timer Examples
"Set a timer for 15 minutes"
// → Intent: TimerStopwatch, Slots: {tool: "timer", action: "set", value: 15}
```

### Unit Testing

```swift
import XCTest

class IntentClassifierTests: XCTestCase {
    var classifier: IntentClassifier!
    
    override func setUp() async throws {
        classifier = IntentClassifier()
        await classifier.initialize()
    }
    
    func testStepsQuery() async throws {
        let result = await classifier.classifyIntent("How many steps did I take today?")
        
        XCTAssertEqual(result?.intent, "QueryPoint")
        XCTAssertEqual(result?.slots["metric"] as? String, "steps")
        XCTAssertEqual(result?.slots["time_ref"] as? String, "today")
    }
    
    func testGoalSetting() async throws {
        let result = await classifier.classifyIntent("Set my daily step goal to 10000")
        
        XCTAssertEqual(result?.intent, "SetGoal")
        XCTAssertEqual(result?.slots["metric"] as? String, "steps")
        XCTAssertEqual(result?.slots["target"] as? Int, 10000)
    }
}
```

## 🔧 Performance Optimization

### Memory Management
- Models are loaded once during initialization
- Automatic cleanup in deinit
- Efficient tensor operations

### Async Processing
- All heavy operations use async/await
- Non-blocking UI updates
- Background processing support

### Error Handling
- Comprehensive error types
- Graceful degradation
- Detailed logging

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Fails**
   ```
   Solution: Ensure TFLite files are added to app bundle
   Check: Build Phases → Copy Bundle Resources
   ```

2. **Vocabulary Not Found**
   ```
   Solution: Verify vocabulary.json is in bundle
   Check: Bundle.main.path(forResource: "vocabulary", ofType: "json")
   ```

3. **TensorFlow Lite Import Error**
   ```
   Solution: Add TensorFlow Lite Swift package
   Check: Package Dependencies in Xcode
   ```

4. **Poor Accuracy**
   ```
   Solution: Ensure models match Android versions
   Check: Model compatibility and TF Lite version
   ```

### Debug Logging

Enable detailed logging in `IntentClassifier.swift`:

```swift
private static let logger = Logger(subsystem: "com.yourapp.intent", category: "IntentClassifier")
```

View logs in Console.app or Xcode debug console.

## 🔄 Migration from Android

### Key Differences

| Android | iOS | Notes |
|---------|-----|-------|
| `Interpreter` | `Interpreter` | Same TensorFlow Lite API |
| `Log.d()` | `Logger.info()` | Use os.log framework |
| `Regex.find()` | `String.range(of:options:)` | Swift regex patterns |
| `JSONObject` | `JSONSerialization` | Native JSON parsing |
| `Map<String, Any>` | `[String: Any]` | Swift dictionaries |

### Model Compatibility
- ✅ Same TensorFlow Lite models
- ✅ Same vocabulary and metadata
- ✅ Same embedding dimensions
- ✅ Same intent mapping

## 📋 Development Checklist

- [ ] Add TensorFlow Lite dependency
- [ ] Copy model files to bundle
- [ ] Add Intent Swift files
- [ ] Update main view with TabView
- [ ] Test basic classification
- [ ] Test slot extraction
- [ ] Add error handling
- [ ] Implement unit tests
- [ ] Optimize performance
- [ ] Add documentation

## 🎯 Features Parity with Android

| Feature | Android | iOS | Status |
|---------|---------|-----|--------|
| Intent Classification | ✅ | ✅ | Complete |
| Slot Extraction | ✅ | ✅ | Complete |
| TensorFlow Lite | ✅ | ✅ | Complete |
| 13 Intent Types | ✅ | ✅ | Complete |
| 14+ Slot Types | ✅ | ✅ | Complete |
| Pattern Matching | ✅ | ✅ | Complete |
| Contextual Inference | ✅ | ✅ | Complete |
| Real-time Processing | ✅ | ✅ | Complete |
| Comprehensive UI | ✅ | ✅ | Complete |
| Detailed Logging | ✅ | ✅ | Complete |
| Error Handling | ✅ | ✅ | Complete |

## 🚀 Next Steps

1. **Voice Integration** - Connect with speech recognition
2. **Core ML Conversion** - Convert to Core ML for better iOS integration
3. **Watch App** - Extend to Apple Watch
4. **Shortcuts Integration** - Add Siri Shortcuts support
5. **Health Kit Integration** - Connect with iOS Health data
6. **Performance Profiling** - Optimize for battery and memory

The iOS implementation now provides complete feature parity with the Android version, offering a comprehensive Natural Language Understanding system for iOS applications! 🎉