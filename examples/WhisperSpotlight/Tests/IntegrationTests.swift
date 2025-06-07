import XCTest
@testable import WhisperSpotlight

final class IntegrationTests: XCTestCase {
    func testClipboardWrite() throws {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString("hello", forType: .string)
        XCTAssertEqual(NSPasteboard.general.string(forType: .string), "hello")
    }
}
