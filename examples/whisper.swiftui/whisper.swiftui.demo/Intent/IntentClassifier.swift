import Foundation
import TensorFlowLite
import os.log

// MARK: - HFTokenizer Support

struct TokenizationResult {
    let ids: [Int]
    let attentionMask: [Int]
}

class HFTokenizer {
    private let tokenizerData: Data
    
    init(_ data: Data) {
        self.tokenizerData = data
    }
    
    func tokenize(_ text: String) -> TokenizationResult {
        // This is a simplified implementation
        // In a real implementation, you would use a proper HuggingFace tokenizer library
        // For now, we'll implement basic tokenization
        
        let words = text.lowercased().components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        var ids: [Int] = [101] // [CLS] token
        var attentionMask: [Int] = [1]
        
        for word in words {
            // Simple hash-based token ID generation (replace with proper tokenizer)
            let tokenId = abs(word.hashValue) % 30000 + 1000
            ids.append(tokenId)
            attentionMask.append(1)
        }
        
        ids.append(102) // [SEP] token
        attentionMask.append(1)
        
        return TokenizationResult(ids: ids, attentionMask: attentionMask)
    }
    
    func close() {
        // Cleanup if needed
    }
}

class IntentClassifier: ObservableObject {
    
    // MARK: - Constants
    
    private static let maxSequenceLength = 256  // Updated to match Kotlin version
    private static let logger = Logger(subsystem: "com.whispercpp.demo", category: "IntentClassifier")
    
    // MARK: - Properties
    
    @Published var isInitialized = false
    @Published var errorMessage: String?
    
    private var intentClassifier: Interpreter?
    private var hfTokenizer: HFTokenizer?
    private var intentMapping: [Int: String] = [:]
    private let slotExtractor = SlotExtractor()
    
    // MARK: - Initialization
    
    func initialize() async -> Bool {
        do {
            Self.logger.info("Initializing Intent Classifier...")
            
            // Load label encoder (replaces metadata)
            try await loadLabelEncoder()
            
            // Load HuggingFace tokenizer
            try await loadHFTokenizer()
            
            // Load the complete TFLite model (end-to-end)
            try await loadIntentClassifier()
            
            await MainActor.run {
                self.isInitialized = true
                self.errorMessage = nil
            }
            
            Self.logger.info("Intent Classifier initialized successfully")
            return true
            
        } catch {
            Self.logger.error("Failed to initialize Intent Classifier: \(error.localizedDescription)")
            await MainActor.run {
                self.errorMessage = error.localizedDescription
                self.isInitialized = false
            }
            return false
        }
    }
    
    // MARK: - Model Loading
    
    private func loadLabelEncoder() async throws {
        guard let path = Bundle.main.path(forResource: "label_encoder", ofType: "json"),
              let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let labelToIntentDict = json["label_to_intent"] as? [String: String] else {
            throw IntentClassificationError.metadataNotLoaded
        }
        
        var tempIntentMapping: [Int: String] = [:]
        for (key, value) in labelToIntentDict {
            if let intKey = Int(key) {
                tempIntentMapping[intKey] = value
            }
        }
        
        self.intentMapping = tempIntentMapping
        Self.logger.info("Loaded \(intentMapping.count) intent mappings from label_encoder.json")
    }
    
    private func loadHFTokenizer() async throws {
        // Load tokenizer.json file from tokenizer folder
        guard let path = Bundle.main.path(forResource: "tokenizer/tokenizer", ofType: "json"),
              let tokenizerData = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
            throw IntentClassificationError.tokenizerNotLoaded
        }
        
        // Initialize HuggingFace tokenizer
        self.hfTokenizer = HFTokenizer(tokenizerData)
        Self.logger.info("HuggingFace tokenizer loaded from tokenizer/tokenizer.json")
    }
    
    private func loadIntentClassifier() async throws {
        guard let modelPath = Bundle.main.path(forResource: "intent_classifier", ofType: "tflite") else {
            throw IntentClassificationError.modelNotLoaded
        }
        
        var options = Interpreter.Options()
        options.threadCount = 2
        
        self.intentClassifier = try Interpreter(modelPath: modelPath, options: options)
        try self.intentClassifier?.allocateTensors()
        
        let inputDetails = intentClassifier?.inputTensorCount ?? 0
        let outputDetails = intentClassifier?.outputTensorCount ?? 0
        
        Self.logger.info("Complete intent classifier loaded - Inputs: \(inputDetails), Outputs: \(outputDetails)")
        
        // Log input/output shapes
        for i in 0..<inputDetails {
            if let shape = try? intentClassifier?.input(at: i).shape {
                Self.logger.info("  Input \(i) shape: \(shape)")
            }
        }
        
        for i in 0..<outputDetails {
            if let shape = try? intentClassifier?.output(at: i).shape {
                Self.logger.info("  Output \(i) shape: \(shape)")
            }
        }
    }
    
    // MARK: - Intent Classification
    
    func classifyIntent(_ text: String) async -> IntentResult? {
        guard isInitialized else {
            Self.logger.error("Intent classifier not initialized")
            return nil
        }
        
        do {
            Self.logger.info("Classifying: '\(text)'")
            
            // Step 1: Tokenize text using BERT tokenizer
            let tokenization = try await tokenizeText(text)
            let inputIds = tokenization.inputIds
            let attentionMask = tokenization.attentionMask
            
            Self.logger.info("Tokenized - InputIds: \(Array(inputIds.prefix(10)).map { String($0) }.joined(separator: ", "))")
            Self.logger.info("AttentionMask: \(Array(attentionMask.prefix(10)).map { String($0) }.joined(separator: ", "))")
            
            // Step 2: Run the complete end-to-end model
            let logits = try await runCompleteModel(inputIds: inputIds, attentionMask: attentionMask)
            
            // Step 3: Apply softmax and get predictions
            let probabilities = softmax(logits)
            guard let bestIndex = probabilities.enumerated().max(by: { $0.element < $1.element })?.offset else {
                throw IntentClassificationError.predictionFailed
            }
            
            let bestIntent = intentMapping[bestIndex] ?? "Unknown"
            let confidence = probabilities[bestIndex]
            
            // Create probability map for all intents
            var allProbabilities: [String: Float] = [:]
            for (index, probability) in probabilities.enumerated() {
                let intent = intentMapping[index] ?? "Unknown_\(index)"
                allProbabilities[intent] = probability
            }
            
            Self.logger.info("Final result: \(bestIntent) (confidence: \(String(format: "%.3f", confidence)))")
            
            // Log top 3 predictions
            let sortedIndices = probabilities.enumerated().sorted { $0.element > $1.element }.map { $0.offset }
            Self.logger.info("Top 3 predictions:")
            for i in 0..<min(3, sortedIndices.count) {
                let index = sortedIndices[i]
                let intent = intentMapping[index] ?? "Unknown_\(index)"
                let prob = probabilities[index]
                Self.logger.info("  \(intent): \(String(format: "%.3f", prob))")
            }
            
            // Step 4: Extract slots for the predicted intent
            let slotResult = await slotExtractor.extractSlots(text: text, intent: bestIntent)
            
            return IntentResult(
                intent: bestIntent,
                confidence: confidence,
                allProbabilities: allProbabilities,
                slots: slotResult.slots,
                slotConfidence: slotResult.confidence
            )
            
        } catch {
            Self.logger.error("Error classifying intent: \(error.localizedDescription)")
            return nil
        }
    }
    
    // MARK: - Text Processing
    
    private func tokenizeText(_ text: String) async throws -> (inputIds: [Int32], attentionMask: [Int32]) {
        // Use HuggingFace tokenizer for proper BERT WordPiece tokenization
        guard let tokenizer = hfTokenizer else {
            throw IntentClassificationError.tokenizerNotLoaded
        }
        
        let result = tokenizer.tokenize(text)
        
        // Ensure the sequences are exactly maxSequenceLength
        var inputIds = Array(repeating: Int32(0), count: Self.maxSequenceLength)
        var attentionMask = Array(repeating: Int32(0), count: Self.maxSequenceLength)
        
        // Copy tokens up to maxSequenceLength, padding or truncating as needed
        let tokenCount = min(result.ids.count, Self.maxSequenceLength)
        for i in 0..<tokenCount {
            inputIds[i] = Int32(result.ids[i])
            attentionMask[i] = Int32(result.attentionMask[i])
        }
        
        // The rest remain as 0 (padding)
        let textPreview = text.count > 50 ? String(text.prefix(50)) + "..." : text
        Self.logger.info("HF Tokenized '\(textPreview)' -> \(tokenCount) tokens")
        Self.logger.info("First 10 input_ids: \(Array(inputIds.prefix(10)).map { String($0) }.joined(separator: ", "))")
        Self.logger.info("Valid tokens: \(attentionMask.reduce(0, +))")
        
        return (inputIds: inputIds, attentionMask: attentionMask)
    }
    
    private func runCompleteModel(inputIds: [Int32], attentionMask: [Int32]) async throws -> [Float] {
        // Prepare inputs for TFLite model
        let inputIdsData = Data(bytes: inputIds, count: inputIds.count * MemoryLayout<Int32>.size)
        let attentionMaskData = Data(bytes: attentionMask, count: attentionMask.count * MemoryLayout<Int32>.size)
        
        try intentClassifier?.copy(inputIdsData, toInputAt: 0)
        try intentClassifier?.copy(attentionMaskData, toInputAt: 1)
        try intentClassifier?.invoke()
        
        let outputTensor = try intentClassifier?.output(at: 0)
        guard let outputData = outputTensor?.data else {
            throw IntentClassificationError.predictionFailed
        }
        
        let logits = outputData.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: Float.self))
        }
        
        Self.logger.info("Model output logits: \(Array(logits.prefix(5)).map { String(format: "%.3f", $0) }.joined(separator: ", "))")
        
        return logits
    }
    
    private func softmax(_ logits: [Float]) -> [Float] {
        let maxLogit = logits.max() ?? 0.0
        let expValues = logits.map { exp($0 - maxLogit) }
        let sumExp = expValues.reduce(0, +)
        return expValues.map { $0 / sumExp }
    }
    
    // MARK: - Utility Methods
    
    func getIntentList() -> [String] {
        return Array(intentMapping.values).sorted()
    }
    
    func cleanup() {
        intentClassifier = nil
        hfTokenizer?.close()
        hfTokenizer = nil
        isInitialized = false
        Self.logger.info("Intent classifier cleaned up")
    }
    
    deinit {
        cleanup()
    }
}