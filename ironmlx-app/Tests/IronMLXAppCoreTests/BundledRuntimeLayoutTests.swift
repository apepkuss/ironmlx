import Foundation
import Testing

@testable import IronMLXAppCore

@Test func bundledRuntimeLayoutResolvesOnlyTheDeclaredBundleFiles() throws {
    let bundleURL = try makeCompleteTestBundle()
    defer { try? FileManager.default.removeItem(at: bundleURL.deletingLastPathComponent()) }

    let layout = try BundledRuntimeLayout.resolve(bundleURL: bundleURL)

    #expect(layout.backendURL.path.hasSuffix("IronMLX.app/Contents/Helpers/ironmlx"))
    #expect(layout.ironBenchURL.path.hasSuffix("IronMLX.app/Contents/Helpers/iron-bench"))
    #expect(layout.metallibURL.path.hasSuffix("IronMLX.app/Contents/Resources/mlx.metallib"))
    #expect(layout.dashboardURL.path.hasSuffix("IronMLX.app/Contents/Resources/dashboard2.html"))
}

@Test func bundledRuntimeLayoutFailsWhenARequiredResourceIsMissing() throws {
    let bundleURL = try makeCompleteTestBundle()
    defer { try? FileManager.default.removeItem(at: bundleURL.deletingLastPathComponent()) }
    try FileManager.default.removeItem(
        at: bundleURL.appendingPathComponent("Contents/Resources/mlx.metallib")
    )

    #expect(throws: BundledRuntimeLayoutError.self) {
        try BundledRuntimeLayout.resolve(bundleURL: bundleURL)
    }
}

@Test func bundledRuntimeFailuresExposeAStableDashboardLocalizationCode() {
    let error = BundledRuntimeLayoutError.missingFile(
        "/Applications/IronMLX.app/Contents/Helpers/ironmlx"
    )

    #expect(DashboardErrorClassifier.code(for: error) == "bundled_runtime_invalid")
}

@Test func bundledRuntimeLayoutRejectsHelperSymlinks() throws {
    let bundleURL = try makeCompleteTestBundle()
    defer { try? FileManager.default.removeItem(at: bundleURL.deletingLastPathComponent()) }
    let helperURL = bundleURL.appendingPathComponent("Contents/Helpers/ironmlx")
    try FileManager.default.removeItem(at: helperURL)
    try FileManager.default.createSymbolicLink(at: helperURL, withDestinationURL: URL(fileURLWithPath: "/bin/echo"))

    #expect(throws: BundledRuntimeLayoutError.self) {
        try BundledRuntimeLayout.resolve(bundleURL: bundleURL)
    }
}

@Test func bundledChildProcessEnvironmentRemovesMlxAndDyldOverrides() {
    let environment = [
        "PATH": "/usr/bin:/bin",
        "HF_TOKEN": "secret",
        "MLX_DIR": "/tmp/mlx",
        "MLX_ROOT": "/tmp/mlx",
        "MLX_METAL_PATH": "/tmp/mlx.metallib",
        "MLX_UNKNOWN_OVERRIDE": "/tmp/unknown",
        "DYLD_LIBRARY_PATH": "/tmp/lib",
        "DYLD_INSERT_LIBRARIES": "/tmp/injected.dylib",
    ]

    #expect(BundledChildProcessEnvironment.sanitized(environment) == [
        "PATH": "/usr/bin:/bin",
        "HF_TOKEN": "secret",
    ])
}

@Test func backendLaunchConfigurationPinsTheBundledMetallibBeforeServe() {
    let config = BackendLaunchConfiguration(
        executableURL: URL(fileURLWithPath: "/Applications/IronMLX.app/Contents/Helpers/ironmlx"),
        metallibURL: URL(fileURLWithPath: "/Applications/IronMLX.app/Contents/Resources/mlx.metallib"),
        host: "127.0.0.1",
        port: 9068
    )

    #expect(config.arguments.prefix(3) == [
        "--mlx-metallib",
        "/Applications/IronMLX.app/Contents/Resources/mlx.metallib",
        "serve",
    ])
}

@Test func backendBinaryResolverHasOnlyBundleInternalCandidates() {
    let expected = BundledRuntimeLayout.expected()

    #expect(BackendBinaryResolver.resolve() == expected.backendURL)
    #expect(BackendBinaryResolver.resolveIronBenchBinary() == expected.ironBenchURL)
    #expect(BackendBinaryResolver.resolveMetallib() == expected.metallibURL)
}

private func makeCompleteTestBundle() throws -> URL {
    let rootURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let bundleURL = rootURL.appendingPathComponent("IronMLX.app", isDirectory: true)
    let helpersURL = bundleURL.appendingPathComponent("Contents/Helpers", isDirectory: true)
    let resourcesURL = bundleURL.appendingPathComponent("Contents/Resources", isDirectory: true)
    try FileManager.default.createDirectory(at: helpersURL, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: resourcesURL, withIntermediateDirectories: true)

    for helper in ["ironmlx", "iron-bench"] {
        let helperURL = helpersURL.appendingPathComponent(helper)
        try Data("#!/bin/sh\nexit 0\n".utf8).write(to: helperURL)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: helperURL.path)
    }
    for resource in [
        "mlx.metallib",
        "dashboard2.html",
        "AppIcon.icns",
        "menubar-icon.png",
        "menubar-icon@2x.png",
        "logo.png",
        "sidebar-logo@2x.png",
    ] {
        try Data(resource.utf8).write(to: resourcesURL.appendingPathComponent(resource))
    }
    return bundleURL
}
