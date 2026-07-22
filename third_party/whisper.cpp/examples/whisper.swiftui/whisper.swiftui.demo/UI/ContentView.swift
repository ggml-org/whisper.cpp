import SwiftUI
import AVFoundation
import Foundation

struct ContentView: View {
    @StateObject var whisperState = WhisperState()
    
    var body: some View {
        NavigationStack {
            VStack {
                HStack {
                    Button("Transcribe", action: {
                        Task {
                            await whisperState.transcribeSample()
                        }
                    })
                    .buttonStyle(.bordered)
                    .disabled(!whisperState.canTranscribe)
                    
                    Button(whisperState.isRecording ? "Stop recording" : "Start recording", action: {
                        Task {
                            await whisperState.toggleRecord()
                        }
                    })
                    .buttonStyle(.bordered)
                    .disabled(!whisperState.canTranscribe)
                }
                
                ScrollView {
                    Text(verbatim: whisperState.messageLog)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .font(.footnote)
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(10)

                HStack {
                    Button("Clear Logs", action: {
                        whisperState.messageLog = ""
                    })
                    .font(.footnote)
                    .buttonStyle(.bordered)

                    Button("Copy Logs", action: {
                        UIPasteboard.general.string = whisperState.messageLog
                    })
                    .font(.footnote)
                    .buttonStyle(.bordered)

                    Button("Bench", action: {
                        Task {
                            await whisperState.benchCurrentModel()
                        }
                    })
                    .font(.footnote)
                    .buttonStyle(.bordered)
                    .disabled(!whisperState.canTranscribe)

                    Button("Bench All", action: {
                        Task {
                            await whisperState.bench(models: ModelsView.getDownloadedModels())
                        }
                    })
                    .font(.footnote)
                    .buttonStyle(.bordered)
                    .disabled(!whisperState.canTranscribe)
                }

                NavigationLink(destination: ModelsView(whisperState: whisperState)) {
                    Text("View Models")
                }
                .font(.footnote)
                .padding()
            }
            .navigationTitle("Whisper SwiftUI Demo")
            .padding()
        }
    }

    struct ModelsView: View {
        @ObservedObject var whisperState: WhisperState
        @Environment(\.dismiss) var dismiss
        
        private static let models: [Model] = [
            Model(name: "tiny", info: "(F16, 75 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin", filename: "tiny.bin"),
            Model(name: "tiny-q5_1", info: "(31 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny-q5_1.bin", filename: "tiny-q5_1.bin"),
            Model(name: "tiny-q8_0", info: "(42 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny-q8_0.bin", filename: "tiny-q8_0.bin"),
            Model(name: "tiny.en", info: "(F16, 75 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin", filename: "tiny.en.bin"),
            Model(name: "tiny.en-q5_1", info: "(31 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin", filename: "tiny.en-q5_1.bin"),
            Model(name: "tiny.en-q8_0", info: "(42 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q8_0.bin", filename: "tiny.en-q8_0.bin"),
            Model(name: "base", info: "(F16, 142 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin", filename: "base.bin"),
            Model(name: "base-q5_1", info: "(57 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin", filename: "base-q5_1.bin"),
            Model(name: "base-q8_0", info: "(78 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q8_0.bin", filename: "base-q8_0.bin"),
            Model(name: "base.en", info: "(F16, 142 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin", filename: "base.en.bin"),
            Model(name: "base.en-q5_1", info: "(57 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q5_1.bin", filename: "base.en-q5_1.bin"),
            Model(name: "base.en-q8_0", info: "(78 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q8_0.bin", filename: "base.en-q8_0.bin"),
            Model(name: "small", info: "(F16, 466 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin", filename: "small.bin"),
            Model(name: "small-q5_1", info: "(181 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin", filename: "small-q5_1.bin"),
            Model(name: "small-q8_0", info: "(252 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q8_0.bin", filename: "small-q8_0.bin"),
            Model(name: "small.en", info: "(F16, 466 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin", filename: "small.en.bin"),
            Model(name: "small.en-q5_1", info: "(181 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en-q5_1.bin", filename: "small.en-q5_1.bin"),
            Model(name: "small.en-q8_0", info: "(252 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en-q8_0.bin", filename: "small.en-q8_0.bin"),
            Model(name: "medium", info: "(F16, 1.5 GiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin", filename: "medium.bin"),
            Model(name: "medium-q5_0", info: "(514 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin", filename: "medium-q5_0.bin"),
            Model(name: "medium-q8_0", info: "(785 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q8_0.bin", filename: "medium-q8_0.bin"),
            Model(name: "medium.en", info: "(F16, 1.5 GiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin", filename: "medium.en.bin"),
            Model(name: "medium.en-q5_0", info: "(514 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en-q5_0.bin", filename: "medium.en-q5_0.bin"),
            Model(name: "medium.en-q8_0", info: "(785 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en-q8_0.bin", filename: "medium.en-q8_0.bin"),
            Model(name: "large-v1", info: "(F16, 2.9 GiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large.bin", filename: "large.bin"),
            Model(name: "large-v2", info: "(F16, 2.9 GiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin", filename: "large-v2.bin"),
            Model(name: "large-v2-q5_0", info: "(1.1 GiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2-q5_0.bin", filename: "large-v2-q5_0.bin"),
            Model(name: "large-v2-q8_0", info: "(1.5 GiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2-q8_0.bin", filename: "large-v2-q8_0.bin"),
            Model(name: "large-v3", info: "(F16, 2.9 GiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin", filename: "large-v3.bin"),
            Model(name: "large-v3-q5_0", info: "(1.1 GiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin", filename: "large-v3-q5_0.bin"),
            Model(name: "large-v3-turbo", info: "(F16, 1.5 GiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin", filename: "large-v3-turbo.bin"),
            Model(name: "large-v3-turbo-q5_0", info: "(547 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin", filename: "large-v3-turbo-q5_0.bin"),
            Model(name: "large-v3-turbo-q8_0", info: "(834 MiB)", url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin", filename: "large-v3-turbo-q8_0.bin"),
        ]

        static func getDownloadedModels() -> [Model] {
            // Filter models that have been downloaded
            return models.filter {
                FileManager.default.fileExists(atPath: $0.fileURL.path())
            }
        }

        func loadModel(model: Model) {
            Task {
                dismiss()
                whisperState.loadModel(path: model.fileURL)
            }
        }

        var body: some View {
            List {
                Section(header: Text("Models")) {
                    ForEach(ModelsView.models) { model in
                        DownloadButton(model: model)
                            .onLoad(perform: loadModel)
                    }
                }
            }
            .listStyle(GroupedListStyle())
            .navigationBarTitle("Models", displayMode: .inline).toolbar {}
        }
    }
}

//struct ContentView_Previews: PreviewProvider {
//    static var previews: some View {
//        ContentView()
//    }
//}
