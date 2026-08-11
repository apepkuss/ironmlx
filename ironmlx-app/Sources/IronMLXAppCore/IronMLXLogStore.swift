import Darwin
import Foundation

public enum IronMLXLogFile: String, Sendable {
    case app = "app.log"
    case backend = "backend.log"
}

public struct DiagnosticLogTail: Sendable, Equatable {
    public var text: String
    public var truncated: Bool
    public var status: String

    public init(text: String, truncated: Bool, status: String) {
        self.text = text
        self.truncated = truncated
        self.status = status
    }
}

public struct IronMLXLogStore: Sendable {
    public static let defaultMaxFileSizeBytes = 20 * 1024 * 1024
    public static let defaultRetainedRotatedFiles = 5

    public let rootURL: URL
    public let maxFileSizeBytes: UInt64
    public let retainedRotatedFiles: Int

    public init(
        rootURL: URL = Self.defaultRootURL(),
        maxFileSizeBytes: UInt64 = UInt64(Self.defaultMaxFileSizeBytes),
        retainedRotatedFiles: Int = Self.defaultRetainedRotatedFiles
    ) {
        self.rootURL = rootURL
        self.maxFileSizeBytes = maxFileSizeBytes
        self.retainedRotatedFiles = max(0, retainedRotatedFiles)
    }

    public static func defaultRootURL() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ironmlx", isDirectory: true)
            .appendingPathComponent("logs", isDirectory: true)
    }

    public func url(for file: IronMLXLogFile) -> URL {
        rootURL.appendingPathComponent(file.rawValue)
    }

    public func prepareLog(_ file: IronMLXLogFile, sessionHeader: String? = nil) throws {
        try ensureLogDirectory()
        try rotateIfNeeded(file)
        try createLogIfNeeded(file)
        if let sessionHeader {
            try appendLine(sessionHeader, to: file)
        }
    }

    public func openFileForAppend(_ file: IronMLXLogFile) throws -> FileHandle {
        try ensureLogDirectory()
        try createLogIfNeeded(file)
        let handle = try FileHandle(forWritingTo: url(for: file))
        try handle.seekToEnd()
        return handle
    }

    public func appendLine(_ line: String, to file: IronMLXLogFile) throws {
        let handle = try openFileForAppend(file)
        defer {
            try? handle.close()
        }
        let text = line.hasSuffix("\n") ? line : line + "\n"
        if let data = text.data(using: .utf8) {
            try handle.write(contentsOf: data)
        }
    }

    public func tailText(
        from file: IronMLXLogFile,
        maxLines: Int = 500,
        maxBytes: Int = 65_536
    ) -> String {
        guard maxLines > 0, maxBytes > 0,
              let handle = try? FileHandle(forReadingFrom: url(for: file))
        else {
            return ""
        }
        defer {
            try? handle.close()
        }
        let size = (try? handle.seekToEnd()) ?? 0
        let boundedBytes = UInt64(maxBytes)
        let offset = size > boundedBytes ? size - boundedBytes : 0
        try? handle.seek(toOffset: offset)
        var data = (try? handle.readToEnd()) ?? Data()
        if offset > 0 {
            while let first = data.first, first & 0b1100_0000 == 0b1000_0000 {
                data.removeFirst()
            }
        }
        guard var text = String(data: data, encoding: .utf8) else {
            return ""
        }
        if offset > 0, let newline = text.firstIndex(of: "\n") {
            text.removeSubrange(...newline)
        }
        let lines = text.split(separator: "\n").suffix(maxLines).map(String.init)
        return lines.joined(separator: "\n")
    }

    public func diagnosticTail(
        from file: IronMLXLogFile,
        maxLines: Int,
        maxBytes: Int
    ) -> DiagnosticLogTail {
        guard maxLines > 0, maxBytes > 0 else {
            return DiagnosticLogTail(text: "", truncated: false, status: "invalid_limit")
        }
        let path = url(for: file).path
        let descriptor = Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
        guard descriptor >= 0 else {
            return DiagnosticLogTail(
                text: "",
                truncated: false,
                status: errno == ENOENT ? "missing" : "unreadable"
            )
        }
        defer { Darwin.close(descriptor) }

        var metadata = stat()
        guard fstat(descriptor, &metadata) == 0, (metadata.st_mode & S_IFMT) == S_IFREG else {
            return DiagnosticLogTail(text: "", truncated: false, status: "not_regular_file")
        }
        let fileSize = max(0, Int64(metadata.st_size))
        let byteLimit = Int64(maxBytes)
        let offset = max(0, fileSize - byteLimit)
        guard lseek(descriptor, off_t(offset), SEEK_SET) >= 0 else {
            return DiagnosticLogTail(text: "", truncated: false, status: "unreadable")
        }

        var buffer = [UInt8](repeating: 0, count: min(maxBytes, Int(fileSize - offset)))
        let bufferCount = buffer.count
        var received = 0
        while received < bufferCount {
            let remaining = bufferCount - received
            let count = buffer.withUnsafeMutableBytes { rawBuffer in
                Darwin.read(
                    descriptor,
                    rawBuffer.baseAddress!.advanced(by: received),
                    remaining
                )
            }
            if count < 0, errno == EINTR { continue }
            guard count > 0 else { break }
            received += count
        }
        buffer.removeSubrange(received..<bufferCount)
        var text = String(decoding: buffer, as: UTF8.self)
        if offset > 0, let newline = text.firstIndex(of: "\n") {
            text.removeSubrange(...newline)
        }
        let allLines = text.split(separator: "\n", omittingEmptySubsequences: false)
        let lineTruncated = allLines.count > maxLines
        text = allLines.suffix(maxLines).joined(separator: "\n")
        return DiagnosticLogTail(
            text: text,
            truncated: offset > 0 || lineTruncated,
            status: "available"
        )
    }

    private func ensureLogDirectory() throws {
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
    }

    private func createLogIfNeeded(_ file: IronMLXLogFile) throws {
        let fileURL = url(for: file)
        if !FileManager.default.fileExists(atPath: fileURL.path) {
            FileManager.default.createFile(atPath: fileURL.path, contents: nil)
        }
    }

    private func rotateIfNeeded(_ file: IronMLXLogFile) throws {
        let fileURL = url(for: file)
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            return
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
        let size = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
        guard size > maxFileSizeBytes else {
            return
        }
        guard retainedRotatedFiles > 0 else {
            try FileManager.default.removeItem(at: fileURL)
            return
        }

        let oldestURL = fileURL.appendingPathExtension(String(retainedRotatedFiles))
        if FileManager.default.fileExists(atPath: oldestURL.path) {
            try FileManager.default.removeItem(at: oldestURL)
        }

        if retainedRotatedFiles > 1 {
            for index in stride(from: retainedRotatedFiles - 1, through: 1, by: -1) {
                let source = fileURL.appendingPathExtension(String(index))
                guard FileManager.default.fileExists(atPath: source.path) else {
                    continue
                }
                let destination = fileURL.appendingPathExtension(String(index + 1))
                if FileManager.default.fileExists(atPath: destination.path) {
                    try FileManager.default.removeItem(at: destination)
                }
                try FileManager.default.moveItem(at: source, to: destination)
            }
        }

        try FileManager.default.moveItem(at: fileURL, to: fileURL.appendingPathExtension("1"))
    }
}

public enum IronMLXAppLogger {
    private static let lock = NSLock()
    private static let store = IronMLXLogStore()

    public static func startSession(date: Date = Date()) {
        let header = "===== IronMLX App started at \(timestamp(date)) ====="
        lock.withLock {
            try? store.prepareLog(.app, sessionHeader: header)
        }
    }

    public static func info(_ message: String) {
        log(level: "INFO", message)
    }

    public static func warning(_ message: String) {
        log(level: "WARN", message)
    }

    public static func error(_ message: String) {
        log(level: "ERROR", message)
    }

    public static func backendSessionHeader(command: String, date: Date = Date()) -> String {
        "===== IronMLX Backend started at \(timestamp(date)) =====\ncommand: \(command)"
    }

    private static func log(level: String, _ message: String) {
        let line = "\(timestamp(Date())) \(level) ironmlx-app: \(message)"
        lock.withLock {
            try? store.prepareLog(.app)
            try? store.appendLine(line, to: .app)
        }
        NSLog("%@", line)
    }

    private static func timestamp(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }
}
