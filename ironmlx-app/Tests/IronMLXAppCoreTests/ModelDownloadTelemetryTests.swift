import Foundation
import Testing

@testable import IronMLXAppCore

@Test func modelDownloadTelemetryReportsResumeSpeedAndPhaseDurations() async throws {
    let time = TelemetryTestTime(monotonic: 100)
    let logs = TelemetryTestLogs()
    let tracker = ModelDownloadTelemetryTracker(
        provider: .huggingFace,
        repoID: "org/model",
        startedAt: Date(timeIntervalSince1970: 1_700_000_000),
        monotonicNow: { time.current() },
        logger: { logs.append($0) }
    )

    await tracker.start()
    await tracker.beginNetwork()
    await tracker.setExpectedBytes(10)
    await tracker.registerFile(path: "model.safetensors", expectedBytes: 10, resumedBytes: 4)
    time.advance(by: 2)
    await tracker.recordProgress(path: "model.safetensors", availableBytes: 6)
    time.advance(by: 10)
    await tracker.recordProgress(path: "model.safetensors", availableBytes: 10)
    await tracker.endNetwork()
    await tracker.beginVerification()
    time.advance(by: 3)
    await tracker.endVerification()
    await tracker.beginPublication()
    time.advance(by: 0.5)
    await tracker.endPublication()
    await tracker.finish(outcome: "completed")

    let events = try logs.events()
    #expect(events.map { $0["event"] as? String } == ["started", "progress", "finished"])
    let finished = try #require(events.last)
    #expect(finished["provider"] as? String == "huggingface")
    #expect(finished["repo_id"] as? String == "org/model")
    #expect(finished["outcome"] as? String == "completed")
    #expect((finished["expected_bytes"] as? NSNumber)?.int64Value == 10)
    #expect((finished["resumed_bytes"] as? NSNumber)?.int64Value == 4)
    #expect((finished["transferred_bytes"] as? NSNumber)?.int64Value == 6)
    #expect((finished["network_seconds"] as? NSNumber)?.doubleValue == 12)
    #expect((finished["verification_seconds"] as? NSNumber)?.doubleValue == 3)
    #expect((finished["publication_seconds"] as? NSNumber)?.doubleValue == 0.5)
    #expect((finished["average_bytes_per_second"] as? NSNumber)?.doubleValue == 0.5)
    #expect(((finished["ewma_bytes_per_second"] as? NSNumber)?.doubleValue ?? 0) > 0)
    #expect(finished["started_at"] as? String == "2023-11-14T22:13:20.000Z")
}

@Test func modelDownloadTelemetryDoesNotCountDiscardedPartialAsEffectiveResume() async throws {
    let time = TelemetryTestTime(monotonic: 10)
    let logs = TelemetryTestLogs()
    let tracker = ModelDownloadTelemetryTracker(
        provider: .huggingFace,
        repoID: "org/model",
        monotonicNow: { time.current() },
        logger: { logs.append($0) }
    )

    await tracker.start()
    await tracker.beginNetwork()
    await tracker.setExpectedBytes(10)
    await tracker.registerFile(path: "model.safetensors", expectedBytes: 10, resumedBytes: 4)
    await tracker.recordProgress(path: "model.safetensors", availableBytes: 0)
    time.advance(by: 2)
    await tracker.recordProgress(path: "model.safetensors", availableBytes: 10)
    await tracker.endNetwork()
    await tracker.finish(outcome: "completed")

    let finished = try #require(try logs.events().last)
    #expect((finished["resumed_bytes"] as? NSNumber)?.int64Value == 0)
    #expect((finished["transferred_bytes"] as? NSNumber)?.int64Value == 10)
}

@Test func modelDownloadTelemetryExcludesPreflightPauseFromNetworkDuration() async throws {
    let time = TelemetryTestTime(monotonic: 10)
    let logs = TelemetryTestLogs()
    let tracker = ModelDownloadTelemetryTracker(
        provider: .huggingFace,
        repoID: "org/model",
        monotonicNow: { time.current() },
        logger: { logs.append($0) }
    )

    await tracker.start()
    await tracker.beginNetwork()
    time.advance(by: 2)
    await tracker.endNetwork()
    time.advance(by: 20)
    await tracker.beginNetwork()
    time.advance(by: 3)
    await tracker.endNetwork()
    await tracker.finish(outcome: "completed")

    let finished = try #require(try logs.events().last)
    #expect((finished["network_seconds"] as? NSNumber)?.doubleValue == 5)
    #expect((finished["elapsed_seconds"] as? NSNumber)?.doubleValue == 25)
}

private final class TelemetryTestTime: @unchecked Sendable {
    private let lock = NSLock()
    private var monotonic: TimeInterval

    init(monotonic: TimeInterval) {
        self.monotonic = monotonic
    }

    func current() -> TimeInterval {
        lock.withLock { monotonic }
    }

    func advance(by interval: TimeInterval) {
        lock.withLock {
            monotonic += interval
        }
    }
}

private final class TelemetryTestLogs: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []

    func append(_ line: String) {
        lock.withLock {
            lines.append(line)
        }
    }

    func events() throws -> [[String: Any]] {
        try lock.withLock {
            try lines.map { line in
                let prefix = "model_download_telemetry "
                let json = try #require(
                    line.hasPrefix(prefix) ? String(line.dropFirst(prefix.count)) : nil)
                let object = try JSONSerialization.jsonObject(with: Data(json.utf8))
                return try #require(object as? [String: Any])
            }
        }
    }
}
