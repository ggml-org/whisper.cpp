import XCTest
@testable import WhisperSpotlight

final class ModelManagerTests: XCTestCase {
    func testModelPathCreation() throws {
        let manager = ModelManager()
        let path = manager.modelPath()
        XCTAssertTrue(path.path.contains("WhisperSpotlight"))
    }
}
