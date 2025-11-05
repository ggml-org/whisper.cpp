import Foundation
import TensorFlowLite
import os.log

class IntentClassifier: ObservableObject {
    
    // MARK: - Constants
    
    private static let maxSequenceLength = 32
    private static let embeddingDim = 384
    private static let logger = Logger(subsystem: "com.whispercpp.demo", category: "IntentClassifier")
    
    // MARK: - Properties
    
    @Published var isInitialized = false
    @Published var errorMessage: String?
    
    private var sentenceEncoder: Interpreter?
    private var intentClassifier: Interpreter?
    private var vocabulary: [String: Int] = [:]
    private var intentMapping: [Int: String] = [:]
    private let slotExtractor = SlotExtractor()
    
    // MARK: - Initialization
    
    func initialize() async -> Bool {
        do {
            Self.logger.info("🔄 Initializing Intent Classifier...")
            
            // Load metadata
            try await loadMetadata()
            
            // Load vocabulary
            try await loadVocabulary()
            
            // Load TensorFlow Lite models
            try await loadSentenceEncoder()
            try await loadIntentClassifier()
            
            await MainActor.run {
                self.isInitialized = true
                self.errorMessage = nil
            }
            
            Self.logger.info("✅ Intent Classifier initialized successfully")
            return true
            
        } catch {
            Self.logger.error("❌ Failed to initialize Intent Classifier: \(error.localizedDescription)")
            await MainActor.run {
                self.errorMessage = error.localizedDescription
                self.isInitialized = false
            }
            return false
        }
    }
    
    // MARK: - Model Loading
    
    private func loadMetadata() async throws {
        guard let path = Bundle.main.path(forResource: "model_metadata", ofType: "json"),
              let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let intentMappingDict = json["intent_mapping"] as? [String: String] else {
            throw IntentClassificationError.metadataNotLoaded
        }
        
        var tempIntentMapping: [Int: String] = [:]
        for (key, value) in intentMappingDict {
            if let intKey = Int(key) {
                tempIntentMapping[intKey] = value
            }
        }
        
        self.intentMapping = tempIntentMapping
        Self.logger.info("📋 Loaded \(intentMapping.count) intent mappings")
    }
    
    private func loadVocabulary() async throws {
        guard let path = Bundle.main.path(forResource: "vocabulary", ofType: "json"),
              let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            throw IntentClassificationError.vocabularyNotLoaded
        }
        
        self.vocabulary = json
        Self.logger.info("📝 Loaded vocabulary with \(vocabulary.count) words")
    }
    
    private func loadSentenceEncoder() async throws {
        guard let modelPath = Bundle.main.path(forResource: "lightweight_sentence_encoder", ofType: "tflite") else {
            throw IntentClassificationError.modelNotLoaded
        }
        
        var options = Interpreter.Options()
        options.threadCount = 2
        
        self.sentenceEncoder = try Interpreter(modelPath: modelPath, options: options)
        try self.sentenceEncoder?.allocateTensors()
        
        let inputShape = try sentenceEncoder?.input(at: 0).shape
        let outputShape = try sentenceEncoder?.output(at: 0).shape
        
        Self.logger.info("🔤 Sentence encoder loaded - Input: \(String(describing: inputShape)), Output: \(String(describing: outputShape))")
    }
    
    private func loadIntentClassifier() async throws {
        guard let modelPath = Bundle.main.path(forResource: "intent_classifier", ofType: "tflite") else {
            throw IntentClassificationError.modelNotLoaded
        }
        
        var options = Interpreter.Options()
        options.threadCount = 2
        
        self.intentClassifier = try Interpreter(modelPath: modelPath, options: options)
        try self.intentClassifier?.allocateTensors()
        
        let inputShape = try intentClassifier?.input(at: 0).shape
        let outputShape = try intentClassifier?.output(at: 0).shape
        
        Self.logger.info("🎯 Intent classifier loaded - Input: \(String(describing: inputShape)), Output: \(String(describing: outputShape))")
    }
    
    // MARK: - Intent Classification
    
    func classifyIntent(_ text: String) async -> IntentResult? {
        guard isInitialized else {
            Self.logger.error("❌ Intent classifier not initialized")
            return nil
        }
        
        do {
            Self.logger.info("🔍 Classifying: '\(text)'")
            
            // Step 1: Convert text to embeddings
            let embeddings = try await textToEmbeddings(text)
            Self.logger.info("📊 Generated embeddings: \(embeddings.count) dimensions")
            
            // Check if embeddings are meaningful
            let nonZeroCount = embeddings.filter { $0 != 0 }.count
            Self.logger.info("🔍 Embedding analysis: nonZeroCount=\(nonZeroCount)")
            
            // Step 2: Classify intent using embeddings
            let probabilities = try await classifyWithEmbeddings(embeddings)
            
            // Step 3: Get best prediction
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
            
            Self.logger.info("✅ Final result: \(bestIntent) (confidence: \(String(format: "%.3f", confidence)))")
            
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
            Self.logger.error("❌ Error classifying intent: \(error.localizedDescription)")
            return nil
        }
    }
    
    // MARK: - Text Processing
    
    private func textToEmbeddings(_ text: String) async throws -> [Float] {
        // Preprocess text
        let processedText = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let words = processedText.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        
        Self.logger.info("🔤 Original text: '\(text)'")
        Self.logger.info("🔤 Processed text: '\(processedText)'")
        Self.logger.info("🔤 Words: \(words.joined(separator: ", "))")
        
        // Convert to token sequence
        var tokenSequence = Array(repeating: Float(0), count: Self.maxSequenceLength)
        
        var knownWords = 0
        for (index, word) in words.enumerated() {
            guard index < Self.maxSequenceLength else { break }
            
            let tokenId = vocabulary[word] ?? 0
            tokenSequence[index] = Float(tokenId)
            if tokenId != 0 { knownWords += 1 }
            Self.logger.debug("🔤 Word '\(word)' -> token \(tokenId)")
        }
        
        let knownPercentage = words.isEmpty ? 0 : (knownWords * 100 / words.count)
        Self.logger.info("🔤 Known words: \(knownWords)/\(words.count) (\(knownPercentage)%)")
        
        // Run sentence encoder
        let inputData = Data(bytes: &tokenSequence, count: tokenSequence.count * MemoryLayout<Float>.size)
        try sentenceEncoder?.copy(inputData, toInputAt: 0)
        try sentenceEncoder?.invoke()
        
        let outputTensor = try sentenceEncoder?.output(at: 0)
        guard let outputData = outputTensor?.data else {
            throw IntentClassificationError.predictionFailed
        }
        
        let embeddings = outputData.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: Float.self))
        }
        
        // Log embedding statistics
        let mean = embeddings.reduce(0, +) / Float(embeddings.count)
        let variance = embeddings.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(embeddings.count)
        let std = sqrt(variance)
        let minVal = embeddings.min() ?? 0
        let maxVal = embeddings.max() ?? 0
        
        Self.logger.info("📊 Embedding stats: mean=\(String(format: "%.6f", mean)), std=\(String(format: "%.6f", std)), range=[\(String(format: "%.3f", minVal)), \(String(format: "%.3f", maxVal))]")
        
        return embeddings
    }
    
    private func classifyWithEmbeddings(_ embeddings: [Float]) async throws -> [Float] {
        var embeddingsCopy = embeddings
        let inputData = Data(bytes: &embeddingsCopy, count: embeddingsCopy.count * MemoryLayout<Float>.size)
        
        try intentClassifier?.copy(inputData, toInputAt: 0)
        try intentClassifier?.invoke()
        
        let outputTensor = try intentClassifier?.output(at: 0)
        guard let outputData = outputTensor?.data else {
            throw IntentClassificationError.predictionFailed
        }
        
        let probabilities = outputData.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: Float.self))
        }
        
        // Log detailed probability information
        Self.logger.info("🎯 Raw predictions:")
        for (index, prob) in probabilities.enumerated() {
            let intent = intentMapping[index] ?? "Unknown_\(index)"
            Self.logger.debug("  \(index). \(intent): \(String(format: "%.3f", prob))")
        }
        
        // Find top 3 predictions
        let sortedIndices = probabilities.enumerated().sorted { $0.element > $1.element }.map { $0.offset }
        Self.logger.info("🏆 Top 3 predictions:")
        for i in 0..<min(3, sortedIndices.count) {
            let index = sortedIndices[i]
            let intent = intentMapping[index] ?? "Unknown_\(index)"
            let prob = probabilities[index]
            Self.logger.info("  \(intent): \(String(format: "%.3f", prob))")
        }
        
        return probabilities
    }
    
    // MARK: - Utility Methods
    
    func getIntentList() -> [String] {
        return Array(intentMapping.values).sorted()
    }
    
    func cleanup() {
        sentenceEncoder = nil
        intentClassifier = nil
        isInitialized = false
        Self.logger.info("🔒 Intent classifier cleaned up")
    }
    
    deinit {
        cleanup()
    }
}