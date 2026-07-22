import SwiftUI

struct DownloadButton: View {
    private var model: Model

    @State private var status: String

    @State private var downloadTask: URLSessionDownloadTask?
    @State private var progress = 0.0
    @State private var observation: NSKeyValueObservation?

    private var onLoad: ((_ model: Model) -> Void)?

    init(model: Model) {
        self.model = model
        status = model.fileExists() ? "downloaded" : "download"
    }

    func onLoad(perform action: @escaping (_ model: Model) -> Void) -> DownloadButton {
        var button = self
        button.onLoad = action
        return button
    }

    private func download() {
        status = "downloading"
        print("Downloading model \(model.name) from \(model.url)")
        guard let url = URL(string: model.url) else { return }

        downloadTask = URLSession.shared.downloadTask(with: url) { temporaryURL, response, error in
            if let error = error {
                print("Error: \(error.localizedDescription)")
                return
            }

            guard let response = response as? HTTPURLResponse, (200...299).contains(response.statusCode) else {
                print("Server error!")
                return
            }

            do {
                if let temporaryURL = temporaryURL {
                    try FileManager.default.copyItem(at: temporaryURL, to: model.fileURL)
                    print("Writing to \(model.filename) completed")
                    status = "downloaded"
                }
            } catch let err {
                print("Error: \(err.localizedDescription)")
            }
        }

        observation = downloadTask?.progress.observe(\.fractionCompleted) { progress, _ in
            self.progress = progress.fractionCompleted
        }

        downloadTask?.resume()
    }

    var body: some View {
        VStack {
            Button(action: {
                if (status == "download") {
                    download()
                } else if (status == "downloading") {
                    downloadTask?.cancel()
                    status = "download"
                } else if (status == "downloaded") {
                    if !model.fileExists() {
                        download()
                    }
                    onLoad?(model)
                }
            }) {
                let title = "\(model.name) \(model.info)"
                if (status == "download") {
                    Text("Download \(title)")
                } else if (status == "downloading") {
                    Text("\(title) (Downloading \(Int(progress * 100))%)")
                } else if (status == "downloaded") {
                    Text("Load \(title)")
                } else {
                    Text("Unknown status")
                }
            }.swipeActions {
                if (status == "downloaded") {
                    Button("Delete") {
                        do {
                            try FileManager.default.removeItem(at: model.fileURL)
                        } catch {
                            print("Error deleting file: \(error)")
                        }
                        status = "download"
                    }
                    .tint(.red)
                }
            }
        }
        .onDisappear() {
            downloadTask?.cancel()
        }
    }
}
