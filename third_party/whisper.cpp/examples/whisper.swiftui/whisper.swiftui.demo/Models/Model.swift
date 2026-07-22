import Foundation

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var info: String
    var url: String

    var filename: String
    var fileURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(filename)
    }

    func fileExists() -> Bool {
        FileManager.default.fileExists(atPath: fileURL.path)
    }
}
