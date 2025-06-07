import XCTest
@testable import WhisperSpotlight

final class RecorderTests: XCTestCase {
    func testWavHeader() async throws {
        let recorder = Recorder()
        let url = FileManager.default.temporaryDirectory.appending(path: "test.wav")
        try await recorder.startRecording(toOutputFile: url, delegate: nil)
        recorder.stopRecording()
        let data = try Data(contentsOf: url)
        XCTAssertEqual(String(data: data.prefix(4), encoding: .ascii), "RIFF")
    }
}
