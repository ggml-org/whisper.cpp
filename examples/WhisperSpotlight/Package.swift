// swift-tools-version:5.7
import PackageDescription

let package = Package(
    name: "WhisperSpotlight",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "WhisperSpotlight", targets: ["WhisperSpotlight"])
    ],
    dependencies: [],
    targets: [
        .target(name: "WhisperSpotlight", dependencies: [], path: "", exclude: ["Tests"]),
        .testTarget(name: "WhisperSpotlightTests", dependencies: ["WhisperSpotlight"], path: "Tests")
    ]
)
