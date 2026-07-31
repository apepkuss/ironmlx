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
