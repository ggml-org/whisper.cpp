import Foundation
import TensorFlowLite
import os.log
import Transformers

// MARK: - BERT Tokenizer Support

struct TokenizationResult {
    let ids: [Int]
    let attentionMask: [Int]
}

class BERTTokenizer {
    private var tokenizer: any Tokenizer
    private let maxLength: Int
    
    init(tokenizerPath: String, maxLength: Int = 256) async throws {
        self.maxLength = maxLength
        // Load BERT tokenizer using swift-transformers API
        self.tokenizer = try await AutoTokenizer.from(pretrained: tokenizerPath)
    }
    
    func tokenize(_ text: String) async throws -> TokenizationResult {
        // Use swift-transformers tokenizer.encode method to get proper BERT tokens
        let tokens = try tokenizer.encode(text: text)
        
        // Handle BERT tokenization: [CLS] + tokens + [SEP] + padding
        var inputIds = [Int]()
        
        // Add [CLS] token if not already present
        if tokens.first != 101 {  // 101 is [CLS] token ID for BERT
            inputIds.append(101)
        }
        
        // Add the encoded tokens
        inputIds.append(contentsOf: tokens)
        
        // Add [SEP] token if not already present  
        if tokens.last != 102 {  // 102 is [SEP] token ID for BERT
            inputIds.append(102)
        }
        
        // Truncate or pad to maxLength
        if inputIds.count > maxLength {
            inputIds = Array(inputIds.prefix(maxLength - 1)) + [102]  // Ensure [SEP] at end
        }
        
        // Create attention mask (1 for real tokens, 0 for padding)
        var attentionMask = Array(repeating: 1, count: inputIds.count)
        
        // Pad with zeros if needed
        while inputIds.count < maxLength {
            inputIds.append(0)      // [PAD] token
            attentionMask.append(0) // No attention for padding
        }
        
        return TokenizationResult(
            ids: inputIds,
            attentionMask: attentionMask
        )
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
    private var bertTokenizer: BERTTokenizer?
    private var intentMapping: [Int: String] = []
    private let slotExtractor = SlotExtractor()
    
    // MARK: - Initialization
    
    func initialize() async -> Bool {
        do {
            Self.logger.info("Initializing Intent Classifier...")
            
            // Debug bundle info
            Self.logger.info("Bundle main path: \(Bundle.main.bundlePath)")
            Self.logger.info("Bundle resource path: \(Bundle.main.resourcePath ?? "nil")")
            
            // List all bundle resources for debugging
            if let resourcePath = Bundle.main.resourcePath {
                do {
                    let resourceContents = try FileManager.default.contentsOfDirectory(atPath: resourcePath)
                    Self.logger.info("Bundle resources: \(resourceContents)")
                } catch {
                    Self.logger.error("Could not list bundle resources: \(error)")
                }
            }
            
            // Load label encoder (replaces metadata)
            do {
                try await loadLabelEncoder()
                Self.logger.info("✅ Label encoder loaded successfully")
            } catch {
                Self.logger.error("❌ Failed to load label encoder: \(error)")
                throw error
            }
            
            // Load BERT tokenizer
            do {
                try await loadBERTTokenizer()
                Self.logger.info("✅ BERT tokenizer loaded successfully")
            } catch {
                Self.logger.error("❌ Failed to load BERT tokenizer: \(error)")
                throw error
            }
            
            // Load the complete TFLite model (end-to-end)
            do {
                try await loadIntentClassifier()
                Self.logger.info("✅ TensorFlow Lite model loaded successfully")
            } catch {
                Self.logger.error("❌ Failed to load TFLite model: \(error)")
                throw error
            }
            
            await MainActor.run {
                self.isInitialized = true
                self.errorMessage = nil
            }
            
            Self.logger.info("✅ Intent Classifier initialized successfully")
            return true
            
        } catch {
            let errorMsg = "Intent Classifier initialization failed: \(error.localizedDescription)"
            Self.logger.error("\(errorMsg)")
            await MainActor.run {
                self.errorMessage = errorMsg
                self.isInitialized = false
            }
            return false
        }
    }
    
    // MARK: - Model Loading
    
    private func loadLabelEncoder() async throws {
        Self.logger.info("Loading label encoder...")
        
        // Try multiple bundle resource loading approaches
        var url: URL?
        
        // Approach 1: Standard bundle resource
        url = Bundle.main.url(forResource: "label_encoder", withExtension: "json")
        
        // Approach 2: Try in Resources subdirectory
        if url == nil {
            url = Bundle.main.url(forResource: "label_encoder", withExtension: "json", subdirectory: "Resources")
        }
        
        // Approach 3: Try direct path if bundle is not working
        if url == nil {
            if let resourcePath = Bundle.main.resourcePath {
                let directPath = "\(resourcePath)/label_encoder.json"
                if FileManager.default.fileExists(atPath: directPath) {
                    url = URL(fileURLWithPath: directPath)
                }
            }
        }
        
        guard let finalUrl = url else {
            Self.logger.error("Could not find label_encoder.json in bundle")
            Self.logger.error("Bundle path: \(Bundle.main.bundlePath)")
            Self.logger.error("Resource path: \(Bundle.main.resourcePath ?? "nil")")
            throw IntentClassificationError.metadataNotLoaded
        }
        
        Self.logger.info("Found label_encoder.json at: \(finalUrl.path)")
        
        do {
            let data = try Data(contentsOf: finalUrl)
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            
            // Handle both label_to_intent and intent_to_label like Python test code
            guard let labelToIntentDict = json?["label_to_intent"] as? [String: String] else {
                Self.logger.error("label_to_intent dictionary not found in JSON")
                throw IntentClassificationError.metadataNotLoaded
            }
            
            // Also verify intent_to_label exists (for completeness)
            if let intentToLabelDict = json?["intent_to_label"] as? [String: Int] {
                Self.logger.info("Found intent_to_label mapping with \(intentToLabelDict.count) entries")
            }
            
            // Get classes array for validation
            if let classes = json?["classes"] as? [String] {
                Self.logger.info("Total classes: \(classes.count)")
                Self.logger.info("Classes: \(classes.joined(separator: ", "))")
            }
            
            var tempIntentMapping: [Int: String] = [:]
            for (key, value) in labelToIntentDict {
                if let intKey = Int(key) {
                    tempIntentMapping[intKey] = value
                }
            }
            
            self.intentMapping = tempIntentMapping
            Self.logger.info("Loaded \(intentMapping.count) intent mappings from label_encoder.json")
            
        } catch {
            Self.logger.error("Error loading label_encoder.json: \(error.localizedDescription)")
            throw IntentClassificationError.metadataNotLoaded
        }
    }
    
    private func loadBERTTokenizer() async throws {
        Self.logger.info("Loading BERT tokenizer...")
        
        // Try to find tokenizer directory in bundle
        var tokenizerPath: String?
        
        // Approach 1: Check Resources/tokenizer subdirectory
        if let resourcePath = Bundle.main.resourcePath {
            let directPath = "\(resourcePath)/tokenizer"
            if FileManager.default.fileExists(atPath: directPath) {
                tokenizerPath = directPath
            }
        }
        
        // Approach 2: Check if tokenizer subdirectory exists directly
        if tokenizerPath == nil {
            if let bundlePath = Bundle.main.url(forResource: "tokenizer", withExtension: nil, subdirectory: "Resources") {
                tokenizerPath = bundlePath.path
            }
        }
        
        // Approach 3: Try tokenizer subdirectory directly
        if tokenizerPath == nil {
            if let bundlePath = Bundle.main.url(forResource: "tokenizer", withExtension: nil) {
                tokenizerPath = bundlePath.path
            }
        }
        
        guard let finalTokenizerPath = tokenizerPath else {
            Self.logger.error("Could not find tokenizer directory in bundle")
            Self.logger.error("Bundle path: \(Bundle.main.bundlePath)")
            Self.logger.error("Resource path: \(Bundle.main.resourcePath ?? "nil")")
            throw IntentClassificationError.tokenizerNotLoaded
        }
        
        Self.logger.info("Found tokenizer directory at: \(finalTokenizerPath)")
        
        do {
            // Initialize BERT tokenizer exactly like Python: AutoTokenizer.from_pretrained('./tokenizer')
            self.bertTokenizer = try await BERTTokenizer(
                tokenizerPath: finalTokenizerPath,
                maxLength: Self.maxSequenceLength
            )
            Self.logger.info("✓ BERT tokenizer loaded from \(finalTokenizerPath)")
            Self.logger.info("  Max sequence length: \(Self.maxSequenceLength)")
            
        } catch {
            Self.logger.error("Error loading BERT tokenizer: \(error.localizedDescription)")
            throw IntentClassificationError.tokenizerNotLoaded
        }
    }
    
    private func loadIntentClassifier() async throws {
        Self.logger.info("Loading TensorFlow Lite model...")
        
        // Try multiple bundle resource loading approaches
        var url: URL?
        
        // Approach 1: Standard bundle resource
        url = Bundle.main.url(forResource: "intent_classifier", withExtension: "tflite")
        
        // Approach 2: Try in Resources subdirectory
        if url == nil {
            url = Bundle.main.url(forResource: "intent_classifier", withExtension: "tflite", subdirectory: "Resources")
        }
        
        // Approach 3: Try direct path if bundle is not working
        if url == nil {
            if let resourcePath = Bundle.main.resourcePath {
                let directPath = "\(resourcePath)/intent_classifier.tflite"
                if FileManager.default.fileExists(atPath: directPath) {
                    url = URL(fileURLWithPath: directPath)
                }
            }
        }
        
        guard let finalUrl = url else {
            Self.logger.error("Could not find intent_classifier.tflite in bundle")
            Self.logger.error("Bundle path: \(Bundle.main.bundlePath)")
            Self.logger.error("Resource path: \(Bundle.main.resourcePath ?? "nil")")
            throw IntentClassificationError.modelNotLoaded
        }
        
        Self.logger.info("Found intent_classifier.tflite at: \(finalUrl.path)")
        
        do {
            var options = Interpreter.Options()
            options.threadCount = 2
            
            self.intentClassifier = try Interpreter(modelPath: finalUrl.path, options: options)
            try self.intentClassifier?.allocateTensors()
            
            let inputDetails = intentClassifier?.inputTensorCount ?? 0
            let outputDetails = intentClassifier?.outputTensorCount ?? 0
            
            Self.logger.info("✓ Complete TFLite model loaded successfully!")
            
            // Log input/output details like Python test code
            for i in 0..<inputDetails {
                if let shape = try? intentClassifier?.input(at: i).shape {
                    Self.logger.info("  Input \(i): Shape: \(shape)")
                }
            }
            
            for i in 0..<outputDetails {
                if let shape = try? intentClassifier?.output(at: i).shape {
                    Self.logger.info("  Output \(i): Shape: \(shape)")
                }
            }
            
            Self.logger.info("\nThis is a single end-to-end model: Text → Intent")
            
        } catch {
            Self.logger.error("Error loading TensorFlow Lite model: \(error.localizedDescription)")
            throw IntentClassificationError.modelNotLoaded
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
            
            Self.logger.info("Prediction: \(bestIntent) (label: \(bestIndex))")
            Self.logger.info("Confidence: \(String(format: "%.4f", confidence))")
            
            // Create probability map for all intents
            var allProbabilities: [String: Float] = [:]
            for (index, probability) in probabilities.enumerated() {
                let intent = intentMapping[index] ?? "Unknown_\(index)"
                allProbabilities[intent] = probability
            }
            
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
        // Use BERT tokenizer for proper WordPiece tokenization
        guard let tokenizer = bertTokenizer else {
            throw IntentClassificationError.tokenizerNotLoaded
        }
        
        let result = try await tokenizer.tokenize(text)
        
        // Convert to Int32 arrays for TensorFlow Lite
        let inputIds = result.ids.map { Int32($0) }
        let attentionMask = result.attentionMask.map { Int32($0) }
        
        // Ensure arrays are exactly maxSequenceLength
        var paddedInputIds = Array(repeating: Int32(0), count: Self.maxSequenceLength)
        var paddedAttentionMask = Array(repeating: Int32(0), count: Self.maxSequenceLength)
        
        let copyCount = min(inputIds.count, Self.maxSequenceLength)
        for i in 0..<copyCount {
            paddedInputIds[i] = inputIds[i]
            paddedAttentionMask[i] = attentionMask[i]
        }
        
        let textPreview = text.count > 50 ? String(text.prefix(50)) + "..." : text
        Self.logger.info("Tokenized '\(textPreview)' -> \(copyCount) tokens (seq_len=\(Self.maxSequenceLength))")
        Self.logger.info("Input IDs: [\(Array(paddedInputIds.prefix(5)).map { String($0) }.joined(separator: ", "))...]")
        Self.logger.info("Attention: [\(Array(paddedAttentionMask.prefix(5)).map { String($0) }.joined(separator: ", "))...] (sum=\(paddedAttentionMask.reduce(0, +)))")
        
        return (inputIds: paddedInputIds, attentionMask: paddedAttentionMask)
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
        
        Self.logger.info("Model logits: [\(Array(logits.prefix(3)).map { String(format: "%.3f", $0) }.joined(separator: ", "))...] (\(logits.count) classes)")
        
        return logits
    }
    
    private func softmax(_ logits: [Float]) -> [Float] {
        // Implement exactly like Python: np.exp(logits) / np.sum(np.exp(logits))
        let expValues = logits.map { exp($0) }
        let sumExp = expValues.reduce(0, +)
        return expValues.map { $0 / sumExp }
    }
    
    // MARK: - Utility Methods
    
    func getIntentList() -> [String] {
        return Array(intentMapping.values).sorted()
    }
    
    func cleanup() {
        intentClassifier = nil
        bertTokenizer?.close()
        bertTokenizer = nil
        isInitialized = false
        Self.logger.info("Intent classifier cleaned up")
    }
    
    deinit {
        cleanup()
    }
}