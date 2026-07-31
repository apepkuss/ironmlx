import Foundation
import Testing

@testable import IronMLXAppCore

@Test func logStoreRotatesOversizedLogBeforeAppendingSessionHeader() throws {
    let root = try temporaryDirectory().appendingPathComponent("logs", isDirectory: true)
    let store = IronMLXLogStore(rootURL: root, maxFileSizeBytes: 10, retainedRotatedFiles: 2)
    let appLog = store.url(for: .app)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try "0123456789012345\n".write(to: appLog, atomically: true, encoding: .utf8)

    try store.prepareLog(.app, sessionHeader: "===== test app session =====")

    let current = try String(contentsOf: appLog, encoding: .utf8)
    let rotated = try String(contentsOf: appLog.appendingPathExtension("1"), encoding: .utf8)
    #expect(current.contains("test app session"))
    #expect(rotated.contains("0123456789012345"))
}

@Test func logStoreRetainsConfiguredNumberOfRotatedFiles() throws {
    let root = try temporaryDirectory().appendingPathComponent("logs", isDirectory: true)
    let store = IronMLXLogStore(rootURL: root, maxFileSizeBytes: 4, retainedRotatedFiles: 2)
    let backendLog = store.url(for: .backend)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try "old-2\n".write(to: backendLog.appendingPathExtension("2"), atomically: true, encoding: .utf8)
    try "old-1\n".write(to: backendLog.appendingPathExtension("1"), atomically: true, encoding: .utf8)
    try "current\n".write(to: backendLog, atomically: true, encoding: .utf8)

    try store.prepareLog(.backend, sessionHeader: "===== backend session =====")

    #expect(FileManager.default.fileExists(atPath: backendLog.path))
    #expect(FileManager.default.fileExists(atPath: backendLog.appendingPathExtension("1").path))
    #expect(FileManager.default.fileExists(atPath: backendLog.appendingPathExtension("2").path))
    #expect(!FileManager.default.fileExists(atPath: backendLog.appendingPathExtension("3").path))
    let newestRotated = try String(contentsOf: backendLog.appendingPathExtension("1"), encoding: .utf8)
    #expect(newestRotated.contains("current"))
}

@Test func logStoreAppendsLinesWithoutReplacingExistingLog() throws {
    let root = try temporaryDirectory().appendingPathComponent("logs", isDirectory: true)
    let store = IronMLXLogStore(rootURL: root, maxFileSizeBytes: 1024, retainedRotatedFiles: 2)
    let appLog = store.url(for: .app)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try "previous\n".write(to: appLog, atomically: true, encoding: .utf8)

    try store.appendLine("next", to: .app)

    let text = try String(contentsOf: appLog, encoding: .utf8)
    #expect(text.contains("previous"))
    #expect(text.contains("next"))
}

@Test func logStoreTailHonorsByteAndLineBoundsAcrossUTF8Boundary() throws {
    let root = try temporaryDirectory().appendingPathComponent("logs", isDirectory: true)
    let store = IronMLXLogStore(rootURL: root)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let lines = (0..<300).map { "\($0)-后台日志-🙂" }.joined(separator: "\n")
    try lines.write(to: store.url(for: .backend), atomically: true, encoding: .utf8)

    let tail = store.tailText(from: .backend, maxLines: 20, maxBytes: 512)
    let tailLines = tail.split(separator: "\n")

    #expect(tail.utf8.count <= 512)
    #expect(tailLines.count <= 20)
    #expect(tail.contains("299-后台日志-🙂"))
}
