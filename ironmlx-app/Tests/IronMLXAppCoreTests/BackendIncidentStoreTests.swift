import Foundation
import Testing

@testable import IronMLXAppCore

@Test @MainActor
func incidentStoreRedactsSecretsAndRetainsOnlyNewestRecords() throws {
    let root = URL(fileURLWithPath: "/private/tmp", isDirectory: true)
        .appendingPathComponent("ironmlx-incident-tests-\(UUID().uuidString)")
    let store = BackendIncidentStore(
        url: root.appendingPathComponent("incidents.json"),
        retainedIncidents: 20
    )
    let snapshot = BackendRecoverySnapshot(config: AppConfig(), models: [])

    for index in 0..<25 {
        let termination = BackendProcessTermination(
            occurredAt: Date(timeIntervalSince1970: TimeInterval(index)),
            launchID: UUID(),
            generation: UInt64(index),
            pid: Int32(index),
            terminationStatus: 9,
            terminationReason: "uncaught_signal",
            stopIntent: .unexpected,
            logTail: """
            Authorization: Bearer secret-\(index)
            api_key=value-\(index)
            {"token":"json-secret-\(index)"}
            """
        )
        try store.upsert(BackendIncidentRecord(termination: termination, snapshot: snapshot))
    }

    let records = store.records()
    #expect(records.count == 20)
    #expect(records.first?.generation == 5)
    #expect(records.last?.generation == 24)
    #expect(records.last?.logTail.contains("secret-24") == false)
    #expect(records.last?.logTail.contains("value-24") == false)
    #expect(records.last?.logTail.contains("json-secret-24") == false)
    #expect(records.last?.logTail.contains("<redacted>") == true)
}

@Test @MainActor
func incidentStoreFiltersDetailsAndClearWithoutTouchingOtherFiles() throws {
    let root = URL(fileURLWithPath: "/private/tmp", isDirectory: true)
        .appendingPathComponent("ironmlx-incident-query-tests-\(UUID().uuidString)")
    let historyURL = root.appendingPathComponent("incidents.json")
    let unrelatedLogURL = root.appendingPathComponent("backend.log")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try Data("runtime log".utf8).write(to: unrelatedLogURL)
    let store = BackendIncidentStore(url: historyURL)

    var recovered = incidentRecord(
        at: Date(timeIntervalSince1970: 100),
        model: "mlx-community/Recovered-4bit"
    )
    recovered.recoveredModels = recovered.modelsBeforeCrash
    recovered.recoveryAttempt = 1
    recovered.recoveryResult = "recovered"
    recovered.updatedAt = Date(timeIntervalSince1970: 101)
    try store.upsert(recovered)

    var failed = incidentRecord(
        at: Date(timeIntervalSince1970: 200),
        model: "mlx-community/Failed-4bit"
    )
    failed.failedModels = failed.modelsBeforeCrash
    failed.recoveryAttempt = 1
    failed.recoveryResult = "failed"
    failed.failures = [
        BackendModelRecoveryFailure(
            model: failed.modelsBeforeCrash[0],
            stage: .verification,
            code: "model_snapshot_corrupt",
            message: "checksum mismatch",
            retryable: false,
            action: .redownloadModel
        ),
    ]
    failed.updatedAt = Date(timeIntervalSince1970: 201)
    try store.upsert(failed)

    let matches = store.records(
        matching: BackendIncidentQuery(
            status: .recoveryFailed,
            model: "failed",
            reason: .modelSnapshotInvalid,
            from: Date(timeIntervalSince1970: 150),
            to: Date(timeIntervalSince1970: 250),
            limit: 10
        )
    )
    #expect(matches.map(\.id) == [failed.id])
    #expect(store.detail(id: failed.id)?.recoveryStatus == .recoveryFailed)
    #expect(store.listPayload(matching: BackendIncidentQuery()).total == 2)

    try store.clear()

    #expect(store.records().isEmpty)
    #expect(try String(contentsOf: unrelatedLogURL, encoding: .utf8) == "runtime log")
}

@Test @MainActor
func incidentStoreRedactsPromptRequestCredentialsAndHomePathBeforeExport() throws {
    let root = URL(fileURLWithPath: "/private/tmp", isDirectory: true)
        .appendingPathComponent("ironmlx-incident-export-tests-\(UUID().uuidString)")
    let store = BackendIncidentStore(
        url: root.appendingPathComponent("incidents.json"),
        maximumStoreBytes: 65_536,
        maximumExportBytes: 65_536
    )
    let secret = "secret-credential-value"
    let prompt = "private user prompt"
    let requestBody = "full private request"
    let termination = BackendProcessTermination(
        occurredAt: Date(),
        launchID: UUID(),
        generation: 1,
        pid: 42,
        terminationStatus: 9,
        terminationReason: "uncaught_signal",
        stopIntent: .unexpected,
        logTail: """
        Authorization: Bearer \(secret)
        {"prompt":"\(prompt)","token":"\(secret)"}
        request body: \(requestBody)
        model path: \(NSHomeDirectory())/models/private
        """
    )
    try store.upsert(
        BackendIncidentRecord(
            termination: termination,
            snapshot: BackendRecoverySnapshot(
                config: AppConfig(
                    defaultModel: "\(NSHomeDirectory())/models/private",
                    loadedModels: ["\(NSHomeDirectory())/models/private"]
                ),
                models: [
                    BackendRecoveryModel(
                        id: "\(NSHomeDirectory())/models/private",
                        modelDir: "\(NSHomeDirectory())/models/private",
                        isDefault: true,
                        pinned: false,
                        maxCacheCap: nil,
                        mtpModelDir: nil,
                        mtpDraftTokens: nil,
                        promptLookup: nil,
                        samplingDefaults: .empty
                    ),
                ]
            )
        )
    )

    let persisted = try String(contentsOf: store.url, encoding: .utf8)
    let export = try store.exportData(matching: BackendIncidentQuery())
    let exported = String(decoding: export, as: UTF8.self)

    for value in [persisted, exported] {
        #expect(!value.contains(secret))
        #expect(!value.contains(prompt))
        #expect(!value.contains(requestBody))
        #expect(!value.contains(NSHomeDirectory()))
        #expect(value.contains("<redacted>"))
    }
    #expect(export.count <= store.maximumExportBytes)
}

@Test @MainActor
func incidentStoreBoundsExportAndIgnoresOversizedOrCorruptFiles() throws {
    let root = URL(fileURLWithPath: "/private/tmp", isDirectory: true)
        .appendingPathComponent("ironmlx-incident-capacity-tests-\(UUID().uuidString)")
    let url = root.appendingPathComponent("incidents.json")
    let store = BackendIncidentStore(
        url: url,
        retainedIncidents: 20,
        maximumStoreBytes: 16_384,
        maximumExportBytes: 4_096
    )
    for index in 0..<20 {
        var record = incidentRecord(
            at: Date(timeIntervalSince1970: TimeInterval(index)),
            model: "mlx-community/Model-\(index)-4bit"
        )
        record.recoveryResult = "failed"
        record.error = String(repeating: "diagnostic-\(index)-", count: 80)
        try store.upsert(record)
    }

    let export = try store.exportData(matching: BackendIncidentQuery(limit: 20))
    let payload = try JSONDecoder.ironMLXIncident.decode(
        BackendIncidentExportPayload.self,
        from: export
    )
    #expect(export.count <= store.maximumExportBytes)
    #expect(payload.truncated)
    #expect(payload.incidents.count < 20)

    try Data(String(repeating: "x", count: store.maximumStoreBytes + 1).utf8)
        .write(to: url, options: .atomic)
    #expect(store.records().isEmpty)
    try Data("not-json".utf8).write(to: url, options: .atomic)
    #expect(store.records().isEmpty)
}

@Test @MainActor
func incidentStoreReadsRecordsWrittenBeforeSchemaFieldsWereAdded() throws {
    let root = URL(fileURLWithPath: "/private/tmp", isDirectory: true)
        .appendingPathComponent("ironmlx-incident-schema-tests-\(UUID().uuidString)")
    let url = root.appendingPathComponent("incidents.json")
    let store = BackendIncidentStore(url: url)
    var record = incidentRecord(at: Date(timeIntervalSince1970: 100), model: "legacy/model")
    record.failures = [
        BackendModelRecoveryFailure(
            model: "legacy/model",
            stage: .loading,
            code: "model_file_missing",
            message: "missing",
            retryable: false,
            action: .redownloadModel
        ),
    ]
    let encoded = try JSONEncoder.ironMLXIncident.encode([record])
    var array = try #require(JSONSerialization.jsonObject(with: encoded) as? [[String: Any]])
    array[0].removeValue(forKey: "schemaVersion")
    array[0].removeValue(forKey: "incidentType")
    array[0].removeValue(forKey: "updatedAt")
    array[0].removeValue(forKey: "recoverySteps")
    if var failures = array[0]["failures"] as? [[String: Any]] {
        failures[0].removeValue(forKey: "reason")
        array[0]["failures"] = failures
    }
    let legacyData = try JSONSerialization.data(withJSONObject: array)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try legacyData.write(to: url)

    let loaded = try #require(store.records().first)
    #expect(loaded.schemaVersion == BackendIncidentRecord.currentSchemaVersion)
    #expect(loaded.incidentType == .unexpectedBackendExit)
    #expect(loaded.updatedAt == loaded.occurredAt)
    #expect(loaded.recoverySteps.isEmpty)
    #expect(loaded.failures.first?.reason == .modelFilesMissing)
}

private func incidentRecord(at date: Date, model: String) -> BackendIncidentRecord {
    BackendIncidentRecord(
        termination: BackendProcessTermination(
            occurredAt: date,
            launchID: UUID(),
            generation: 1,
            pid: 42,
            terminationStatus: 9,
            terminationReason: "uncaught_signal",
            stopIntent: .unexpected,
            logTail: "tail"
        ),
        snapshot: BackendRecoverySnapshot(
            config: AppConfig(defaultModel: model, loadedModels: [model]),
            models: [
                BackendRecoveryModel(
                    id: model,
                    modelDir: nil,
                    isDefault: true,
                    pinned: false,
                    maxCacheCap: nil,
                    mtpModelDir: nil,
                    mtpDraftTokens: nil,
                    promptLookup: nil,
                    samplingDefaults: .empty
                ),
            ]
        )
    )
}
