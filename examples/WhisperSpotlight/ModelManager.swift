import Foundation

struct ModelManager {
    private let fileManager = FileManager.default
    private let modelFile = "ggml-large-v3-turbo.bin"
    private let url = URL(string: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin")!

    func modelPath() -> URL {
        let app = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appending(path: "WhisperSpotlight")
        try? fileManager.createDirectory(at: app, withIntermediateDirectories: true)
        return app.appending(path: modelFile)
    }

    func ensureModel(progress: ((Double) -> Void)? = nil) async throws {
        let path = modelPath()
        if fileManager.fileExists(atPath: path.path) { return }
        try await downloadModel(to: path, progress: progress)
    }

    private func downloadModel(to path: URL, progress: ((Double) -> Void)?) async throws {
        let request = URLRequest(url: url)
        for _ in 0..<3 {
            do {
                let (temp, response) = try await URLSession.shared.download(for: request, delegate: ProgressDelegate(progress))
                guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else { throw URLError(.badServerResponse) }
                try fileManager.moveItem(at: temp, to: path)
                let attr = try fileManager.attributesOfItem(atPath: path.path)
                if let size = attr[.size] as? NSNumber, size.intValue > 1_400_000_000 { return }
            } catch {
                try? fileManager.removeItem(at: path)
                continue
            }
        }
        throw URLError(.cannotCreateFile)
    }
}

private class ProgressDelegate: NSObject, URLSessionTaskDelegate {
    let callback: ((Double) -> Void)?
    init(_ cb: ((Double) -> Void)?) { self.callback = cb }
    func urlSession(_ session: URLSession, task: URLSessionTask, didSendBodyData bytesSent: Int64, totalBytesSent: Int64, totalBytesExpectedToSend: Int64) {
        if totalBytesExpectedToSend > 0 {
            callback?(Double(totalBytesSent)/Double(totalBytesExpectedToSend))
        }
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        if totalBytesExpectedToWrite > 0 {
            callback?(Double(totalBytesWritten)/Double(totalBytesExpectedToWrite))
        }
    }
}
