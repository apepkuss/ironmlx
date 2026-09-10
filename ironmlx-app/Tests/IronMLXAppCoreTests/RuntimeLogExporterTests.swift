import AppKit
import Foundation
import Testing
@testable import IronMLXAppCore

@Test @MainActor func runtimeLogExportWritesBothSourcesExactly() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    for source in ["ironmlx-app", "ironmlx-server"] {
        let url = root.appendingPathComponent(source + ".log")
        try Data("previous".utf8).write(to: url)
        let content = "中文日志\n\u{001B}[32mINFO\n"
        let exporter = RuntimeLogExporter { filename in
            #expect(filename.hasPrefix(source + "-"))
            #expect(filename.hasSuffix(".log"))
            return url
        }
        #expect(await exporter.export(.init(source: source, content: content)) == "exported")
        #expect(try Data(contentsOf: url) == Data(content.utf8))
    }
}

@Test @MainActor func runtimeLogExportCancellationAndWriteFailure() async throws {
    let request = RuntimeLogExportRequest(source: "ironmlx-app", content: "test\n")
    let cancelled = RuntimeLogExporter { _ in nil }
    #expect(await cancelled.export(request) == "cancelled")
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let failed = RuntimeLogExporter { _ in root.appendingPathComponent("absent/file.log") }
    #expect(await failed.export(request) == "failed")
    #expect(!FileManager.default.fileExists(atPath: root.path))
}

@Test @MainActor func runtimeLogExportDoesNotOpenDuplicatePanels() async {
    let request = RuntimeLogExportRequest(source: "ironmlx-server", content: "test")
    var pending: CheckedContinuation<URL?, Never>?
    let exporter = RuntimeLogExporter { _ in
        await withCheckedContinuation { pending = $0 }
    }
    let first = Task { await exporter.export(request) }
    while pending == nil { await Task.yield() }
    #expect(await exporter.export(request) == "busy")
    pending?.resume(returning: nil)
    #expect(await first.value == "cancelled")
}

@Test @MainActor func runtimeLogExportRejectsUnknownSource() throws {
    #expect(DashboardBridge.handlerNames.contains("exportRuntimeLog"))
    #expect(throws: (any Error).self) {
        try RuntimeLogExportRequest.decode(#"{"source":"../../other","content":"text"}"#)
    }
}
