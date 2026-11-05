import Foundation
import Testing

@testable import IronMLXAppCore

@Test func componentSelectionUsesEachIndexAndExcludesAlternativeMainWeights() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let metadata = [
        "config.json": #"{"model_type":"llama","mtp_file":"optiq/mtp.safetensors","optiq_vision":{"sidecar":"vision.safetensors"}}"#,
        "model.safetensors.index.json": #"{"weight_map":{"main":"main.safetensors"}}"#,
        "encoder/config.json": "{}",
        "encoder/model.safetensors.index.json": #"{"weight_map":{"encoder":"part.safetensors"}}"#,
        "vocoder/config.json": "{}",
    ]
    for (path, contents) in metadata {
        let file = root.appendingPathComponent(path)
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(contents.utf8).write(to: file)
    }
    let inventory = Set(metadata.keys).union([
        "main.safetensors", "model.safetensors", "unused.safetensors", "encoder/part.safetensors",
        "vocoder/decoder.safetensors", "vision.safetensors", "optiq/mtp.safetensors",
    ])
    let selection = try ModelWeightSelection(directory: root, inventory: inventory)
    #expect(selection.requiredPaths == ["main.safetensors", "encoder/part.safetensors", "vocoder/decoder.safetensors", "vision.safetensors"])
    #expect(selection.optionalPaths == ["optiq/mtp.safetensors"])
    #expect(!selection.paths.contains("model.safetensors"))
    #expect(!selection.paths.contains("unused.safetensors"))
}

@Test(arguments: ["traversal", "mixed", "empty", "missing"])
func componentSelectionRejectsUnsafeOrIncompleteNestedIndex(fault: String) throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let component = root.appendingPathComponent("encoder")
    try FileManager.default.createDirectory(at: component, withIntermediateDirectories: true)
    let index: String
    switch fault {
    case "traversal": index = #"{"weight_map":{"tensor":"../model.safetensors"}}"#
    case "mixed": index = #"{"weight_map":{"tensor":7}}"#
    case "empty": index = #"{"weight_map":{}}"#
    default: index = #"{"weight_map":{"tensor":"missing.safetensors"}}"#
    }
    try Data(index.utf8).write(to: component.appendingPathComponent("model.safetensors.index.json"))
    #expect(throws: ModelSnapshotVerificationError.self) {
        try ModelWeightSelection(directory: root, inventory: ["model.safetensors", "encoder/model.safetensors.index.json"])
    }
}

@Test func componentPipelineWithoutCanonicalRootWeightsRetainsAllWeightFiles() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let paths: Set<String> = ["diffusion_pytorch_model.safetensors", "vae/model.safetensors"]
    let selection = try ModelWeightSelection(directory: root, inventory: paths)
    #expect(selection.paths == paths)
}
