import Foundation

public struct ColdCacheCapacity: Codable, Equatable {
    public var minGB: Int
    public var maxGB: Int
    public var defaultGB: Int
    public var reserveGB: Int
    public var freeGB: Int?

    public init(minGB: Int, maxGB: Int, defaultGB: Int, reserveGB: Int, freeGB: Int?) {
        self.minGB = minGB
        self.maxGB = maxGB
        self.defaultGB = defaultGB
        self.reserveGB = reserveGB
        self.freeGB = freeGB
    }

    enum CodingKeys: String, CodingKey {
        case minGB = "min_gb"
        case maxGB = "max_gb"
        case defaultGB = "default_gb"
        case reserveGB = "reserve_gb"
        case freeGB = "free_gb"
    }
}

public enum ColdCacheCapacityPolicy {
    public static let minGB = 1
    public static let hardMaxGB = 100
    public static let reserveGB = 10

    public static func capacity(
        forDirectoryPath path: String?,
        fileManager: FileManager = .default
    ) -> ColdCacheCapacity {
        let availableBytes = availableCapacityBytes(forDirectoryPath: path, fileManager: fileManager)
        return ColdCacheCapacity(
            minGB: minGB,
            maxGB: maximumGigabytes(availableBytes: availableBytes),
            defaultGB: BackendLaunchOptions.defaultColdCacheLimitGB,
            reserveGB: reserveGB,
            freeGB: freeGigabytes(availableBytes: availableBytes)
        )
    }

    public static func maximumGigabytes(availableBytes: Int?) -> Int {
        guard let availableBytes, availableBytes > 0 else {
            return hardMaxGB
        }

        let gib = BackendLaunchOptions.bytesPerGigabyte
        let hardMaxBytes = hardMaxGB * gib
        let halfAvailableBytes = availableBytes / 2
        let reserveProtectedBytes = max(0, availableBytes - reserveGB * gib)
        let candidateBytes = min(hardMaxBytes, halfAvailableBytes, reserveProtectedBytes)
        return max(minGB, candidateBytes / gib)
    }

    public static func availableCapacityBytes(
        forDirectoryPath path: String?,
        fileManager: FileManager = .default
    ) -> Int? {
        let volumeURL = existingVolumeURL(forDirectoryPath: path, fileManager: fileManager)
        guard let values = try? volumeURL.resourceValues(forKeys: [
            .volumeAvailableCapacityForImportantUsageKey,
            .volumeAvailableCapacityKey,
        ]) else {
            return nil
        }
        if let capacity = values.volumeAvailableCapacityForImportantUsage {
            return Int(clamping: capacity)
        }
        return values.volumeAvailableCapacity
    }

    private static func freeGigabytes(availableBytes: Int?) -> Int? {
        guard let availableBytes, availableBytes >= 0 else {
            return nil
        }
        return availableBytes / BackendLaunchOptions.bytesPerGigabyte
    }

    private static func existingVolumeURL(
        forDirectoryPath path: String?,
        fileManager: FileManager
    ) -> URL {
        let rawPath = path?.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedPath = rawPath?.isEmpty == false
            ? rawPath!
            : BackendLaunchOptions.defaultPagedPrefixCacheDirectory
        let expandedPath = (normalizedPath as NSString).expandingTildeInPath
        var url = URL(fileURLWithPath: expandedPath, isDirectory: true)
        var isDirectory = ObjCBool(false)

        while !fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) {
            let parent = url.deletingLastPathComponent()
            if parent.path == url.path {
                return URL(fileURLWithPath: "/", isDirectory: true)
            }
            url = parent
        }
        return url
    }
}
