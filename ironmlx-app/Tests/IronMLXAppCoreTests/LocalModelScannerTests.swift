import Foundation
import Testing

@testable import IronMLXAppCore

@Test func localModelScannerRejectsIncompleteShardedSnapshot() throws {
    let root = try temporaryDirectory()
    let snapshot = root
        .appendingPathComponent("models", isDirectory: true)
        .appendingPathComponent("models--mlx-community--Sharded-4bit", isDirectory: true)
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent("main", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: snapshot.appendingPathComponent("config.json"))
    try Data("""
    {"weight_map":{"layer.0":"model-00001-of-00002.safetensors","layer.1":"model-00002-of-00002.safetensors"}}
    """.utf8).write(to: snapshot.appendingPathComponent("model.safetensors.index.json"))
    try Data("partial".utf8).write(to: snapshot.appendingPathComponent("model-00001-of-00002.safetensors"))

    let scanner = LocalModelScanner(rootURL: root)

    #expect(scanner.scan().isEmpty)

    try Data("complete".utf8).write(to: snapshot.appendingPathComponent("model-00002-of-00002.safetensors"))

    #expect(scanner.scan().map(\.repoID) == ["mlx-community/Sharded-4bit"])
}

@Test func localModelScannerReadsContextWindowFromConfig() throws {
    let root = try temporaryDirectory()
    let snapshot = root
        .appendingPathComponent("models", isDirectory: true)
        .appendingPathComponent("models--mlx-community--LongContext-4bit", isDirectory: true)
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent("main", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try Data("""
    {
      "model_type": "qwen3_5",
      "text_config": {
        "max_position_embeddings": 262144
      }
    }
    """.utf8).write(to: snapshot.appendingPathComponent("config.json"))
    try Data("weights".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))

    let scanner = LocalModelScanner(rootURL: root)

    #expect(scanner.scan().first?.maxPositionEmbeddings == 262144)
}

func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("ironmlx-app-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
