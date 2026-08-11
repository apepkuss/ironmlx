import AppKit
import Darwin
import Foundation

public enum DiagnosticExportStatus: String, Codable, Sendable {
    case exported
    case cancelled
    case failed
    case busy
}

public struct DiagnosticExportResult: Codable, Sendable {
    public var status: DiagnosticExportStatus
    public var errorCode: String?

    enum CodingKeys: String, CodingKey {
        case status
        case errorCode = "error_code"
    }
}

public struct DiagnosticArchivePublisher: Sendable {
    public init() {}

    public func publish(_ data: Data, to destination: URL, maximumBytes: Int) throws {
        guard destination.isFileURL, data.count <= maximumBytes else {
            throw DiagnosticBundleError.archiveCapacityExceeded
        }
        let directory = destination.deletingLastPathComponent()
        let temporary = directory.appendingPathComponent(
            ".\(destination.lastPathComponent).\(UUID().uuidString).tmp",
            isDirectory: false
        )
        var descriptor = Darwin.open(
            temporary.path,
            O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
            S_IRUSR | S_IWUSR
        )
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        var shouldRemove = true
        defer {
            if descriptor >= 0 { Darwin.close(descriptor) }
            if shouldRemove { try? FileManager.default.removeItem(at: temporary) }
        }
        try data.withUnsafeBytes { buffer in
            guard let base = buffer.baseAddress else { return }
            var offset = 0
            while offset < buffer.count {
                if Task.isCancelled { throw DiagnosticBundleError.cancelled }
                let written = Darwin.write(descriptor, base.advanced(by: offset), buffer.count - offset)
                if written < 0, errno == EINTR { continue }
                guard written > 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
                offset += written
            }
        }
        guard fsync(descriptor) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        if Task.isCancelled { throw DiagnosticBundleError.cancelled }
        guard Darwin.close(descriptor) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        descriptor = -1
        guard rename(temporary.path, destination.path) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        shouldRemove = false
    }
}

@MainActor
public final class DiagnosticExportCoordinator {
    public typealias Completion = @MainActor (DiagnosticExportResult) -> Void
    public typealias DestinationChooser = @MainActor () async -> URL?

    private let service: DiagnosticBundleService
    private let publisher: DiagnosticArchivePublisher
    private let maximumArchiveBytes: Int
    private weak var window: NSWindow?
    private var task: Task<Void, Never>?
    private var exportID: UUID?
    private var activeCompletion: Completion?
    private let destinationChooser: DestinationChooser

    public init(
        window: NSWindow?,
        service: DiagnosticBundleService = DiagnosticBundleService(),
        publisher: DiagnosticArchivePublisher = DiagnosticArchivePublisher(),
        maximumArchiveBytes: Int = DiagnosticBundleLimits().maximumArchiveBytes,
        destinationChooser: DestinationChooser? = nil
    ) {
        self.window = window
        self.service = service
        self.publisher = publisher
        self.maximumArchiveBytes = maximumArchiveBytes
        self.destinationChooser = destinationChooser ?? { [weak window] in
            await Self.chooseDestination(window: window)
        }
    }

    public var isExporting: Bool { task != nil }

    public func export(config: AppConfig, backendRunning: Bool, completion: @escaping Completion) {
        guard task == nil else {
            completion(DiagnosticExportResult(status: .busy, errorCode: "export_in_progress"))
            return
        }
        let id = UUID()
        exportID = id
        activeCompletion = completion
        task = Task { [weak self] in
            guard let self else { return }
            let result: DiagnosticExportResult
            do {
                let artifact = try await service.collect(config: config, backendRunning: backendRunning)
                try Task.checkCancellation()
                guard let destination = await destinationChooser() else {
                    result = DiagnosticExportResult(status: .cancelled, errorCode: nil)
                    finish(result, id: id, completion: completion)
                    return
                }
                let publisher = publisher
                let maximumArchiveBytes = maximumArchiveBytes
                let publishTask = Task.detached(priority: .utility) {
                    try publisher.publish(
                        artifact.archiveData,
                        to: destination,
                        maximumBytes: maximumArchiveBytes
                    )
                }
                try await withTaskCancellationHandler {
                    try await publishTask.value
                } onCancel: {
                    publishTask.cancel()
                }
                result = DiagnosticExportResult(status: .exported, errorCode: nil)
            } catch is CancellationError {
                result = DiagnosticExportResult(status: .cancelled, errorCode: nil)
            } catch DiagnosticBundleError.cancelled {
                result = DiagnosticExportResult(status: .cancelled, errorCode: nil)
            } catch {
                IronMLXAppLogger.error("Diagnostic export failed: \(DiagnosticPrivacy.sanitizedText(error.localizedDescription, maximumBytes: 1_024))")
                result = DiagnosticExportResult(status: .failed, errorCode: "diagnostic_export_failed")
            }
            finish(result, id: id, completion: completion)
        }
    }

    public func cancel() {
        guard task != nil else { return }
        task?.cancel()
        task = nil
        exportID = nil
        let completion = activeCompletion
        activeCompletion = nil
        completion?(DiagnosticExportResult(status: .cancelled, errorCode: nil))
    }

    private func finish(_ result: DiagnosticExportResult, id: UUID, completion: Completion) {
        guard exportID == id else { return }
        task = nil
        exportID = nil
        activeCompletion = nil
        completion(result)
    }

    private static func chooseDestination(window: NSWindow?) async -> URL? {
        let panel = NSSavePanel()
        let timestamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        panel.nameFieldStringValue = "ironmlx-diagnostics-\(timestamp).zip"
        panel.allowedContentTypes = [.zip]
        panel.canCreateDirectories = true
        return await withCheckedContinuation { continuation in
            let completion: (NSApplication.ModalResponse) -> Void = { response in
                continuation.resume(returning: response == .OK ? panel.url : nil)
            }
            if let window {
                panel.beginSheetModal(for: window, completionHandler: completion)
            } else {
                panel.begin(completionHandler: completion)
            }
        }
    }
}
