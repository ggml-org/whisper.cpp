import Foundation
import AVFoundation

actor Recorder {
    private var recorder: AVAudioRecorder?
    private(set) var currentFile: URL?

    func startRecording(toOutputFile url: URL, delegate: AVAudioRecorderDelegate?) throws {
        currentFile = url
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16000.0,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ]
        let rec = try AVAudioRecorder(url: url, settings: settings)
        rec.delegate = delegate
        guard rec.record() else { throw NSError(domain: "rec", code: 1) }
        recorder = rec
    }

    func stopRecording() {
        recorder?.stop()
        recorder = nil
    }
}
