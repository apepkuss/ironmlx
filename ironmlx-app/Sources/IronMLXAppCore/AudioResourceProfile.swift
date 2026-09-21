import CryptoKit
import Foundation
import ZIPFoundation

enum AudioResourceError: Error, LocalizedError {
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case let .invalid(detail): "Audio resources: \(detail)"
        }
    }
}

/// Fixed conversion instructions generated from the independently verified
/// safetensors recipe. No pickle parser, model code, Python or PyTorch is run.
struct AudioResourceProfile: Sendable {
    struct File: Codable, Sendable {
        var path: String
        var bytes: Int64
        var sha256: String
        var role: String?

        func verify(at url: URL) throws {
            try Task.checkCancellation()
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
            guard values.isRegularFile == true, values.isSymbolicLink != true,
                  Int64(values.fileSize ?? -1) == bytes else {
                throw AudioResourceError.invalid("missing or incorrect size: \(path)")
            }
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            var digest = SHA256()
            while let block = try handle.read(upToCount: 1_048_576), !block.isEmpty {
                try Task.checkCancellation()
                digest.update(data: block)
            }
            guard digest.finalize().map({ String(format: "%02x", $0) }).joined() == sha256 else {
                throw AudioResourceError.invalid("SHA-256 mismatch: \(path)")
            }
        }
    }

    struct Source: Decodable, Sendable {
        var repository: String
        var revision: String
        var files: [File]
    }

    struct TextResource: Decodable, Sendable {
        struct Artifact: Decodable, Sendable {
            var url: URL
            var size: Int64
        }
        var name: String
        var artifact: Artifact
        var sha256: String
        var members: [File]
    }

    struct Component: Decodable, Sendable {
        struct Block: Decodable, Sendable {
            var role: String
            var archive: String
            var member: String
            var bytes: Int64
        }
        var path: String
        var header: Data
        var blocks: [Block]
    }

    let source: Source
    let auxiliary: [Source]
    let text: [TextResource]
    let inputs: [File]
    let files: [File]
    let outputs: [File]
    let components: [Component]
    let resourceLock: Data
    let manifest: Data

    init(data: Data? = nil) throws {
        let profileData: Data
        if let data { profileData = data }
        else {
            guard let url = IronMLXAppResourceResolver.url(forResource: "indextts25-preparation", withExtension: "json") else {
                throw AudioResourceError.invalid("bundled preparation profile is missing")
            }
            profileData = try Data(contentsOf: url)
        }
        guard let object = try JSONSerialization.jsonObject(with: profileData) as? [String: Any],
              object["version"] as? Int == 1,
              let sources = object["sources"] as? [String: Any],
              let derived = object["manifest"] as? [String: Any],
              let outputSpecs = derived["components"] as? [String: [String: Any]] else {
            throw AudioResourceError.invalid("invalid bundled profile")
        }
        func encode(_ object: Any?) throws -> Data {
            guard let object else { throw AudioResourceError.invalid("missing profile field") }
            return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        }
        func decode<T: Decodable>(_ type: T.Type, _ object: Any?) throws -> T {
            try JSONDecoder().decode(type, from: encode(object))
        }
        source = try decode(Source.self, sources["source"])
        auxiliary = try decode([Source].self, sources["auxiliary_sources"])
        text = try decode([TextResource].self, sources["text_resources"])
        inputs = try decode([File].self, derived["inputs"])
        files = try decode([File].self, derived["files"])
        outputs = try outputSpecs.sorted(by: { $0.key < $1.key }).map { name, spec in
            var spec = spec
            spec["path"] = name
            return try decode(File.self, spec)
        }
        components = try decode([Component].self, object["components"])
        resourceLock = try encode(sources)
        manifest = try encode(derived)
    }

    static func path(_ relative: String, in directory: URL) throws -> URL {
        try ModelSnapshotVerifier.safeFileURL(path: relative, beneath: directory)
    }

    /// Caller supplies a private staging directory and publishes only after full verification.
    func convert(roots: [String: URL], destination: URL) throws {
        for input in inputs {
            guard let role = input.role, let root = roots[role] else {
                throw AudioResourceError.invalid("missing conversion source")
            }
            try input.verify(at: Self.path(input.path, in: root))
        }
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        var archives: [URL: Archive] = [:]
        for component in components {
            let target = try Self.path(component.path, in: destination)
            guard FileManager.default.createFile(atPath: target.path, contents: component.header) else {
                throw AudioResourceError.invalid("cannot create \(component.path)")
            }
            let output = try FileHandle(forWritingTo: target)
            defer { try? output.close() }
            try output.seekToEnd()
            for block in component.blocks {
                try Task.checkCancellation()
                guard let root = roots[block.role] else {
                    throw AudioResourceError.invalid("missing block source")
                }
                let url = try Self.path(block.archive, in: root)
                let archive: Archive
                if let cached = archives[url] { archive = cached }
                else {
                    archive = try Archive(url: url, accessMode: .read)
                    archives[url] = archive
                }
                guard let entry = archive[block.member], entry.type == .file,
                      entry.uncompressedSize == block.bytes else {
                    throw AudioResourceError.invalid("invalid storage: \(block.member)")
                }
                _ = try archive.extract(entry) { data in
                    try Task.checkCancellation()
                    try output.write(contentsOf: data)
                }
            }
            try output.synchronize()
        }
        for file in files {
            guard let input = inputs.first(where: { $0.sha256 == file.sha256 && $0.bytes == file.bytes }),
                  let role = input.role, let root = roots[role] else {
                throw AudioResourceError.invalid("missing notice or configuration: \(file.path)")
            }
            let target = try Self.path(file.path, in: destination)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: Self.path(input.path, in: root), to: target)
        }
        try manifest.write(to: destination.appendingPathComponent("manifest.json"), options: .atomic)
        try verifyDerived(at: destination)
    }

    func verifyDerived(at directory: URL) throws {
        for file in outputs + files {
            try file.verify(at: Self.path(file.path, in: directory))
        }
        guard try Data(contentsOf: directory.appendingPathComponent("manifest.json")) == manifest else {
            throw AudioResourceError.invalid("derived manifest differs from bundled profile")
        }
    }
}
