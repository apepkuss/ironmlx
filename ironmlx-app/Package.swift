// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "IronMLXApp",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "ironmlx-app", targets: ["IronMLXApp"])
    ],
    targets: [
        .executableTarget(
            name: "IronMLXApp",
            dependencies: ["IronMLXAppCore"]
        ),
        .target(
            name: "IronMLXAppCore",
            resources: [
                .copy("Resources/dashboard2.html"),
                .copy("Resources/logo.png"),
                .copy("Resources/menubar-icon.png"),
                .copy("Resources/menubar-icon@2x.png"),
                .copy("Resources/sidebar-logo@2x.png"),
            ]
        ),
        .testTarget(
            name: "IronMLXAppCoreTests",
            dependencies: ["IronMLXAppCore"]
        ),
    ]
)
