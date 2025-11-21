import Foundation

// MARK: - Data Models

struct IntentResult {
    let intent: String
    let confidence: Float
    let allProbabilities: [String: Float]
    let slots: [String: Any]
    let slotConfidence: Float
    
    init(intent: String, confidence: Float, allProbabilities: [String: Float], slots: [String: Any] = [:], slotConfidence: Float = 0.0) {
        self.intent = intent
        self.confidence = confidence
        self.allProbabilities = allProbabilities
        self.slots = slots
        self.slotConfidence = slotConfidence
    }
}

struct SlotExtractionResult {
    let slots: [String: Any]
    let confidence: Float
}

// MARK: - Intent Classification Error Types

enum IntentClassificationError: Error {
    case modelNotLoaded
    case tokenizerNotLoaded
    case metadataNotLoaded
    case invalidInput
    case predictionFailed
    
    var localizedDescription: String {
        switch self {
        case .modelNotLoaded:
            return "Failed to load intent_classifier.tflite"
        case .tokenizerNotLoaded:
            return "Failed to load tokenizer/tokenizer.json"
        case .metadataNotLoaded:
            return "Failed to load label_encoder.json"
        case .invalidInput:
            return "Invalid input text"
        case .predictionFailed:
            return "Failed to run model prediction"
        }
    }
}