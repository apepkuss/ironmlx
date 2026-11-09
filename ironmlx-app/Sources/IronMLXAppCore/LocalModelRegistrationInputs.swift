import Darwin
import Foundation

/// The identity of one file or directory as `stat` sees it: through a
/// symbolic link (size, modification and status-change times, inode of the
/// target) and of the link itself. A replacement or a retargeted link changes
/// the inode; any write, even in place with the same size and a restored
/// modification time, changes the status-change time.
struct LocalModelFileStamp: Equatable, Sendable {
    var path: String
    var exists: Bool
    var size: Int64 = 0
    var modified: Int64 = 0
    var statusChanged: Int64 = 0
    var inode: UInt64 = 0
    var device: Int32 = 0
    var linkModified: Int64 = 0
    var linkStatusChanged: Int64 = 0
    var linkInode: UInt64 = 0

    init(path: String) {
        self.path = path
        var target = stat()
        var link = stat()
        let linkExists = lstat(path, &link) == 0
        exists = stat(path, &target) == 0
        if exists {
            size = Int64(target.st_size)
            modified = Self.nanoseconds(target.st_mtimespec)
            statusChanged = Self.nanoseconds(target.st_ctimespec)
            inode = UInt64(target.st_ino)
            device = target.st_dev
        }
        if linkExists {
            linkModified = Self.nanoseconds(link.st_mtimespec)
            linkStatusChanged = Self.nanoseconds(link.st_ctimespec)
            linkInode = UInt64(link.st_ino)
        }
    }

    /// `url` and everything beneath it, hidden files included, in path order.
    static func tree(at url: URL) -> [LocalModelFileStamp] {
        let root = url.path
        var stamps = [LocalModelFileStamp(path: root)]
        guard stamps[0].exists,
              let enumerator = FileManager.default.enumerator(atPath: root)
        else {
            return stamps
        }
        var relativePaths: [String] = []
        while let relative = enumerator.nextObject() as? String {
            relativePaths.append(relative)
        }
        for relative in relativePaths.sorted() {
            stamps.append(LocalModelFileStamp(path: root + "/" + relative))
        }
        return stamps
    }

    private static func nanoseconds(_ time: timespec) -> Int64 {
        Int64(time.tv_sec) * 1_000_000_000 + Int64(time.tv_nsec)
    }
}

/// Everything a model's registration is computed from. While these stay the
/// same, so does the registration.
struct LocalModelRegistrationInputs: Equatable, Sendable {
    var modelsRoot: String
    var parametersFile: String
    /// The model as the caller scanned it: readiness, capabilities, type,
    /// pinned state, MTP candidates, size.
    var model: LocalModel
    var parameters: ModelParameters?
    var isDefault: Bool
    /// Only when the max cache cap reads it (KV cache, no saved cap).
    var activeKvOffloadEnabled: Bool?
    /// The MTP draft the registration looks up, when MTP is on.
    var mtpModelID: String?
    /// The model's repositories and snapshot files, the MTP draft's, and the
    /// speech resources for TTS models.
    var files: [LocalModelFileStamp]

    static func read(
        for model: LocalModel,
        defaultModel: String?,
        scanner: LocalModelScanner,
        parameterStore: ModelParameterStore,
        activeKvOffloadEnabled: Bool
    ) -> Self {
        let parameters = parameterStore.parameters(for: model.id)
        let capabilities = model.capabilities
        let mtpModelID = expectedMtpModelID(for: model, parameters: parameters)
        var files = scanner.registrationFileStamps(for: model.id)
        if let mtpModelID {
            files += scanner.registrationFileStamps(for: mtpModelID)
        }
        if model.type == "tts" {
            files += scanner.audioResourceFileStamps()
        }
        return Self(
            modelsRoot: scanner.rootURL.path,
            parametersFile: parameterStore.url.path,
            model: model,
            parameters: parameters,
            isDefault: model.id == defaultModel,
            activeKvOffloadEnabled: capabilities?.supportsKvCache != false && parameters?.maxCacheCap == nil
                ? activeKvOffloadEnabled
                : nil,
            mtpModelID: mtpModelID,
            files: files
        )
    }

    /// The draft `ModelMtpRuntimeResolver` selects for a registration: the
    /// saved one, else the first compatible candidate. `nil` when MTP is off.
    static func expectedMtpModelID(for model: LocalModel, parameters: ModelParameters?) -> String? {
        guard model.capabilities?.supportsMtp != false, parameters?.mtpEnabled == true else {
            return nil
        }
        let saved = parameters?.mtpModelID?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return saved.isEmpty ? model.mtp?.candidates.first?.id : saved
    }
}
