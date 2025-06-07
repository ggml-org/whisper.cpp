import SwiftUI
import AppKit

enum OverlayState {
    case idle, listening, transcribing(String), done(String)
}

struct OverlayView: View {
    @State private var state: OverlayState = .idle
    @State private var recorder = Recorder()
    @State private var modelURL: URL? = nil
    @State private var manager = ModelManager()
    @State private var task: Task<Void, Never>? = nil

    var body: some View {
        VStack {
            switch state {
            case .idle:
                Image(systemName: "mic")
                    .onTapGesture { toggleListening() }
            case .listening:
                ProgressView("Listening…")
                    .onAppear { startRecording() }
            case .transcribing(let text):
                ProgressView(text)
            case .done(let text):
                Text(text)
            }
        }
        .frame(width: 320, height: 200)
        .background(Material.thick)
        .cornerRadius(12)
        .onReceive(NotificationCenter.default.publisher(for: .toggleOverlay)) { _ in
            toggleListening()
        }
    }

    private func toggleListening() {
        switch state {
        case .idle: state = .listening
        case .listening: stopRecording()
        default: break
        }
    }

    private func startRecording() {
        task = Task {
            do {
                try await manager.ensureModel()
                let file = try FileManager.default
                    .temporaryDirectory.appending(path: "record.wav")
                try await recorder.startRecording(toOutputFile: file, delegate: nil)
            } catch {}
        }
    }

    private func stopRecording() {
        task?.cancel()
        Task {
            recorder.stopRecording()
            if let url = recorder.currentFile {
                state = .transcribing("Transcribing…")
                let ctx = try? WhisperContext.createContext(path: manager.modelPath().path())
                if let data = try? decodeWaveFile(url) {
                    ctx?.fullTranscribe(samples: data, language: "")
                    let text = ctx?.getTranscription() ?? ""
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(text, forType: .string)
                    state = .done(text)
                }
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                state = .idle
            }
        }
    }
}

extension Notification.Name {
    static let toggleOverlay = Notification.Name("ToggleOverlay")
}
