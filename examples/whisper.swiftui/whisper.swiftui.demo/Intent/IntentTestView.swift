import SwiftUI

struct IntentTestView: View {
    @StateObject private var intentClassifier = IntentClassifier()
    @State private var inputText = ""
    @State private var result: IntentResult?
    @State private var isLoading = false
    
    private let exampleCommands = [
        "How many steps did I take today?",
        "What's my average heart rate yesterday?",
        "Set my daily step goal to 10000",
        "Set a threshold for heart rate above 100 bpm",
        "Set a timer for 15 minutes",
        "Turn on do not disturb",
        "Start a running workout",
        "Call mom",
        "What's the weather in London?",
        "How far did I walk last week?"
    ]
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Header
                    headerSection
                    
                    // Initialization Status
                    if !intentClassifier.isInitialized {
                        initializationSection
                    }
                    
                    // Input Section
                    if intentClassifier.isInitialized {
                        inputSection
                    }
                    
                    // Example Commands
                    if intentClassifier.isInitialized {
                        exampleCommandsSection
                    }
                    
                    // Results Section
                    if let result = result {
                        resultsSection(result: result)
                    }
                    
                    // Available Intents
                    if intentClassifier.isInitialized {
                        availableIntentsSection
                    }
                }
                .padding()
            }
            .navigationTitle("Intent Test")
            .navigationBarTitleDisplayMode(.inline)
        }
        .task {
            if !intentClassifier.isInitialized {
                await intentClassifier.initialize()
            }
        }
    }
    
    // MARK: - Header Section
    
    private var headerSection: some View {
        VStack(spacing: 8) {
            Text("🎯 Intent Classification Test")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("Test the TensorFlow Lite intent classifier with your voice commands")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .background(Color.blue.opacity(0.1))
        .cornerRadius(12)
    }
    
    // MARK: - Initialization Section
    
    private var initializationSection: some View {
        VStack(spacing: 12) {
            if let errorMessage = intentClassifier.errorMessage {
                Label("Initialization Failed", systemImage: "exclamationmark.triangle")
                    .foregroundColor(.red)
                    .font(.headline)
                
                Text(errorMessage)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                
                Button("Retry") {
                    Task {
                        await intentClassifier.initialize()
                    }
                }
                .buttonStyle(.borderedProminent)
                
            } else {
                ProgressView()
                    .scaleEffect(1.2)
                
                Text("Initializing Intent Classifier...")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color.orange.opacity(0.1))
        .cornerRadius(12)
    }
    
    // MARK: - Input Section
    
    private var inputSection: some View {
        VStack(spacing: 12) {
            Text("💬 Test Input")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            HStack {
                TextField("Enter your command...", text: $inputText)
                    .textFieldStyle(.roundedBorder)
                
                Button(action: testCommand) {
                    if isLoading {
                        ProgressView()
                            .scaleEffect(0.8)
                    } else {
                        Image(systemName: "paperplane.fill")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(inputText.isEmpty || isLoading)
            }
            
            Button("Clear") {
                inputText = ""
                result = nil
            }
            .buttonStyle(.bordered)
            .frame(maxWidth: .infinity, alignment: .trailing)
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(12)
    }
    
    // MARK: - Example Commands Section
    
    private var exampleCommandsSection: some View {
        VStack(spacing: 12) {
            Text("📝 Example Commands")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            LazyVGrid(columns: [GridItem(.flexible())], spacing: 8) {
                ForEach(exampleCommands, id: \.self) { command in
                    Button(action: { tryExample(command) }) {
                        Text(command)
                            .font(.caption)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .frame(maxWidth: .infinity)
                            .background(Color.blue.opacity(0.1))
                            .cornerRadius(8)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .padding()
        .background(Color.green.opacity(0.1))
        .cornerRadius(12)
    }
    
    // MARK: - Results Section
    
    private func resultsSection(result: IntentResult) -> some View {
        VStack(spacing: 16) {
            Text("🎯 Classification Result")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            // Main Result
            VStack(spacing: 8) {
                HStack {
                    Text(result.intent)
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.green)
                    
                    Spacer()
                    
                    Text("Confidence: \(Int(result.confidence * 100))%")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            }
            
            // Slot Extraction Results
            if !result.slots.isEmpty {
                Divider()
                
                VStack(spacing: 12) {
                    HStack {
                        Text("🏷️ Extracted Slots:")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        
                        Spacer()
                        
                        Text("Slot Confidence: \(Int(result.slotConfidence * 100))%")
                            .font(.caption)
                            .foregroundColor(.blue)
                    }
                    
                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                        ForEach(Array(result.slots.keys.sorted()), id: \.self) { key in
                            HStack {
                                Text(key)
                                    .font(.caption)
                                    .fontWeight(.medium)
                                    .foregroundColor(.blue)
                                
                                Spacer()
                                
                                Text("\(result.slots[key] as? String ?? String(describing: result.slots[key] ?? ""))")
                                    .font(.caption)
                                    .fontWeight(.bold)
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.blue.opacity(0.1))
                            .cornerRadius(6)
                        }
                    }
                }
            }
            
            Divider()
            
            // All Probabilities
            VStack(spacing: 8) {
                Text("All Intent Probabilities:")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .frame(maxWidth: .infinity, alignment: .leading)
                
                ScrollView {
                    LazyVStack(spacing: 4) {
                        ForEach(result.allProbabilities.sorted { $0.value > $1.value }, id: \.key) { intent, probability in
                            HStack {
                                Text(intent)
                                    .font(.caption)
                                    .fontWeight(intent == result.intent ? .bold : .regular)
                                
                                Spacer()
                                
                                Text("\(Int(probability * 100))%")
                                    .font(.caption)
                                    .fontWeight(intent == result.intent ? .bold : .regular)
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 2)
                            .background(intent == result.intent ? Color.green.opacity(0.2) : Color.clear)
                            .cornerRadius(4)
                        }
                    }
                }
                .frame(maxHeight: 200)
            }
        }
        .padding()
        .background(Color.green.opacity(0.1))
        .cornerRadius(12)
    }
    
    // MARK: - Available Intents Section
    
    private var availableIntentsSection: some View {
        VStack(spacing: 12) {
            Text("🏷️ Available Intents (\(intentClassifier.getIntentList().count))")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 4) {
                    ForEach(intentClassifier.getIntentList(), id: \.self) { intent in
                        Text("• \(intent)")
                            .font(.caption)
                            .padding(.vertical, 1)
                    }
                }
            }
            .frame(maxHeight: 150)
        }
        .padding()
        .background(Color.purple.opacity(0.1))
        .cornerRadius(12)
    }
    
    // MARK: - Actions
    
    private func testCommand() {
        guard !inputText.isEmpty else { return }
        
        isLoading = true
        result = nil
        
        Task {
            let classificationResult = await intentClassifier.classifyIntent(inputText)
            
            await MainActor.run {
                self.result = classificationResult
                self.isLoading = false
            }
        }
    }
    
    private func tryExample(_ command: String) {
        inputText = command
        testCommand()
    }
}

// MARK: - Preview

struct IntentTestView_Previews: PreviewProvider {
    static var previews: some View {
        IntentTestView()
    }
}