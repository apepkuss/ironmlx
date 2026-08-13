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
    dependencies: [
        .package(
            url: "https://github.com/sparkle-project/Sparkle.git",
            exact: "2.9.5"
        ),
        .package(
            url: "https://github.com/weichsel/ZIPFoundation.git",
            exact: "0.9.20"
        ),
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
            dependencies: [
                .product(name: "Sparkle", package: "Sparkle"),
                .product(name: "ZIPFoundation", package: "ZIPFoundation"),
            ],
            resources: [
                .copy("Resources/dashboard2.html"),
                .copy("Resources/hermes-agent-logo.svg"),
                .copy("Resources/logo.png"),
                .copy("Resources/menubar-icon.png"),
                .copy("Resources/menubar-icon@2x.png"),
                .copy("Resources/oh-my-pi-logo.svg"),
                .copy("Resources/sidebar-logo@2x.png"),
            ]
        ),
        .testTarget(
            name: "IronMLXAppCoreTests",
            dependencies: [
                "IronMLXAppCore",
                .product(name: "ZIPFoundation", package: "ZIPFoundation"),
            ],
            resources: [
                .copy("Fixtures/backend_crash_helper.py"),
            ]
        ),
    ]
)
