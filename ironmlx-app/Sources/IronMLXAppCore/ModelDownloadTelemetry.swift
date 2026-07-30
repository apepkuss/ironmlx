import Foundation

actor ModelDownloadTelemetryTracker {
    typealias Logger = @Sendable (String) -> Void

    private struct FileProgress {
        var expectedBytes: Int64
        var resumedBytes: Int64
        var availableBytes: Int64

        var transferredBytes: Int64 {
            max(0, availableBytes - resumedBytes)
        }
    }

    private let provider: ModelRepositoryProvider
    private let repoID: String
    private let startedAt: Date
    private let startedMonotonic: TimeInterval
    private let monotonicNow: @Sendable () -> TimeInterval
    private let logger: Logger
    private var files: [String: FileProgress] = [:]
    private var expectedBytes: Int64 = 0
    private var expectedBytesIsExplicit = false
    private var activeNetworkStartedAt: TimeInterval?
    private var accumulatedNetworkSeconds: TimeInterval = 0
    private var verificationStartedAt: TimeInterval?
    private var verificationFinishedAt: TimeInterval?
    private var publicationStartedAt: TimeInterval?
    private var publicationFinishedAt: TimeInterval?
    private var lastRateSampleAt: TimeInterval?
    private var lastRateSampleBytes: Int64 = 0
    private var lastProgressLogAt: TimeInterval?
    private var ewmaBytesPerSecond: Double = 0
    private var hasEWMASample = false
    private var finished = false

    init(
        provider: ModelRepositoryProvider,
        repoID: String,
        startedAt: Date = Date(),
        monotonicNow: @escaping @Sendable () -> TimeInterval = {
            ProcessInfo.processInfo.systemUptime
        },
        logger: @escaping Logger
    ) {
        self.provider = provider
        self.repoID = repoID
        self.startedAt = startedAt
        self.monotonicNow = monotonicNow
        self.logger = logger
        startedMonotonic = monotonicNow()
    }

    func start() {
        emit(
            event: "started",
            fields: [
                "started_at": Self.iso8601(startedAt)
            ]
        )
    }

    func beginNetwork() {
        guard activeNetworkStartedAt == nil else {
            return
        }
        let now = monotonicNow()
        activeNetworkStartedAt = now
        lastRateSampleAt = now
        lastRateSampleBytes = transferredBytes
        lastProgressLogAt = now
    }

    func setExpectedBytes(_ bytes: Int64) {
        expectedBytes = max(0, bytes)
        expectedBytesIsExplicit = true
    }

    func registerFile(path: String, expectedBytes: Int64, resumedBytes: Int64) {
        let expected = max(0, expectedBytes)
        let resumed = min(expected, max(0, resumedBytes))
        files[path] = FileProgress(
            expectedBytes: expected,
            resumedBytes: resumed,
            availableBytes: resumed
        )
        if !expectedBytesIsExplicit {
            self.expectedBytes = files.values.reduce(0) { $0 + $1.expectedBytes }
        }
    }

    func recordProgress(path: String, availableBytes: Int64) {
        guard var file = files[path] else {
            return
        }
        let available = min(file.expectedBytes, max(0, availableBytes))
        if available < file.resumedBytes {
            file.resumedBytes = 0
            file.availableBytes = available
        } else {
            file.availableBytes = max(file.availableBytes, available)
        }
        files[path] = file

        let now = monotonicNow()
        updateEWMA(now: now)
        if let lastProgressLogAt,
            now - lastProgressLogAt >= 10
        {
            emitProgress(now: now, currentFile: path)
            self.lastProgressLogAt = now
        }
    }

    func endNetwork() {
        guard let activeNetworkStartedAt else {
            return
        }
        let now = monotonicNow()
        updateEWMA(now: now)
        accumulatedNetworkSeconds += max(0, now - activeNetworkStartedAt)
        self.activeNetworkStartedAt = nil
        lastRateSampleAt = nil
    }

    func beginVerification() {
        verificationStartedAt = verificationStartedAt ?? monotonicNow()
    }

    func endVerification() {
        verificationFinishedAt = verificationFinishedAt ?? monotonicNow()
    }

    func beginPublication() {
        publicationStartedAt = publicationStartedAt ?? monotonicNow()
    }

    func endPublication() {
        publicationFinishedAt = publicationFinishedAt ?? monotonicNow()
    }

    func finish(outcome: String, errorCode: String? = nil) {
        guard !finished else {
            return
        }
        finished = true
        let now = monotonicNow()
        if let activeNetworkStartedAt {
            updateEWMA(now: now)
            accumulatedNetworkSeconds += max(0, now - activeNetworkStartedAt)
            self.activeNetworkStartedAt = nil
        }
        if verificationStartedAt != nil, verificationFinishedAt == nil {
            verificationFinishedAt = now
        }
        if publicationStartedAt != nil, publicationFinishedAt == nil {
            publicationFinishedAt = now
        }

        let networkSeconds = accumulatedNetworkSeconds
        var fields: [String: Any] = [
            "outcome": outcome,
            "started_at": Self.iso8601(startedAt),
            "elapsed_seconds": max(0, now - startedMonotonic),
            "expected_bytes": expectedBytes,
            "resumed_bytes": resumedBytes,
            "transferred_bytes": transferredBytes,
            "average_bytes_per_second": networkSeconds > 0
                ? Double(transferredBytes) / networkSeconds
                : 0,
            "ewma_bytes_per_second": hasEWMASample ? ewmaBytesPerSecond : 0,
            "network_seconds": networkSeconds,
            "verification_seconds": phaseDuration(
                start: verificationStartedAt,
                finish: verificationFinishedAt
            ),
            "publication_seconds": phaseDuration(
                start: publicationStartedAt,
                finish: publicationFinishedAt
            ),
        ]
        if let errorCode {
            fields["error_code"] = errorCode
        }
        emit(event: "finished", fields: fields)
    }

    private var resumedBytes: Int64 {
        files.values.reduce(0) { $0 + $1.resumedBytes }
    }

    private var transferredBytes: Int64 {
        files.values.reduce(0) { $0 + $1.transferredBytes }
    }

    private func updateEWMA(now: TimeInterval) {
        guard let lastRateSampleAt else {
            self.lastRateSampleAt = now
            lastRateSampleBytes = transferredBytes
            return
        }
        let elapsed = now - lastRateSampleAt
        guard elapsed >= 0.25 else {
            return
        }
        let currentBytes = transferredBytes
        let delta = max(0, currentBytes - lastRateSampleBytes)
        let instantaneous = Double(delta) / elapsed
        if hasEWMASample {
            let alpha = 1 - exp(-elapsed / 5)
            ewmaBytesPerSecond += alpha * (instantaneous - ewmaBytesPerSecond)
        } else {
            ewmaBytesPerSecond = instantaneous
            hasEWMASample = true
        }
        self.lastRateSampleAt = now
        lastRateSampleBytes = currentBytes
    }

    private func emitProgress(now: TimeInterval, currentFile: String) {
        let activeSeconds = activeNetworkStartedAt.map { max(0, now - $0) } ?? 0
        let networkSeconds = accumulatedNetworkSeconds + activeSeconds
        emit(
            event: "progress",
            fields: [
                "current_file": currentFile,
                "expected_bytes": expectedBytes,
                "resumed_bytes": resumedBytes,
                "transferred_bytes": transferredBytes,
                "average_bytes_per_second": networkSeconds > 0
                    ? Double(transferredBytes) / networkSeconds
                    : 0,
                "ewma_bytes_per_second": hasEWMASample ? ewmaBytesPerSecond : 0,
                "network_seconds": networkSeconds,
            ]
        )
    }

    private func phaseDuration(
        start: TimeInterval?,
        finish: TimeInterval?
    ) -> TimeInterval {
        guard let start, let finish else {
            return 0
        }
        return max(0, finish - start)
    }

    private func emit(event: String, fields: [String: Any]) {
        var object = fields
        object["event"] = event
        object["provider"] = provider.rawValue
        object["repo_id"] = repoID
        guard JSONSerialization.isValidJSONObject(object),
            let data = try? JSONSerialization.data(
                withJSONObject: object,
                options: [.sortedKeys]
            ),
            let json = String(data: data, encoding: .utf8)
        else {
            return
        }
        logger("model_download_telemetry \(json)")
    }

    private static func iso8601(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }
}
