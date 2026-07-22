import Foundation
import AVFoundation

actor Recorder {
    private var recorder: AVAudioRecorder?
    
    enum RecorderError: Error {
        case couldNotStartRecording
    }
    
    func startRecording(toOutputFile url: URL, delegate: AVAudioRecorderDelegate?) throws {
        let recordSettings: [String : Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16000.0,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ]
#if !os(macOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default)
#endif
        let recorder = try AVAudioRecorder(url: url, settings: recordSettings)
        recorder.delegate = delegate
        if recorder.record() == false {
            print("Could not start recording")
            throw RecorderError.couldNotStartRecording
        }
        self.recorder = recorder
    }
    
    func stopRecording() {
        recorder?.stop()
        recorder = nil
    }
}
