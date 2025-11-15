// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "WhisperSwiftUI",
    platforms: [
        .iOS(.v15),
        .macOS(.v12)
    ],
    products: [
        .library(
            name: "WhisperSwiftUI",
            targets: ["WhisperSwiftUI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.0"),
        .package(url: "https://github.com/tensorflow/tensorflow", from: "2.19.0")
    ],
    targets: [
        .target(
            name: "WhisperSwiftUI",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "TensorFlowLiteC", package: "tensorflow")
            ]),
        .testTarget(
            name: "WhisperSwiftUITests",
            dependencies: ["WhisperSwiftUI"]),
    ]
)