// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "IronMLXApp",
    platforms: [
        .macOS("26.2")
    ],
    products: [
        .executable(name: "ironmlx-app", targets: ["IronMLXApp"]),
        .executable(name: "ironmlx-model-migrate", targets: ["IronMLXModelMigrate"]),
    ],
    targets: [
        .executableTarget(
            name: "IronMLXApp",
            dependencies: ["IronMLXAppCore"]
        ),
        .executableTarget(
            name: "IronMLXModelMigrate",
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
            dependencies: ["IronMLXAppCore"],
            resources: [
                .copy("Fixtures/backend_crash_helper.py"),
            ]
        ),
    ]
)
