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
    case vocabularyNotLoaded
    case metadataNotLoaded
    case invalidInput
    case predictionFailed
    
    var localizedDescription: String {
        switch self {
        case .modelNotLoaded:
            return "TensorFlow Lite model not loaded"
        case .vocabularyNotLoaded:
            return "Vocabulary not loaded"
        case .metadataNotLoaded:
            return "Model metadata not loaded"
        case .invalidInput:
            return "Invalid input text"
        case .predictionFailed:
            return "Intent prediction failed"
        }
    }
}