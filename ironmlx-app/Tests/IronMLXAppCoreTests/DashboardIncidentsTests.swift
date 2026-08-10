import Foundation
import Testing

@testable import IronMLXAppCore

@Test @MainActor
func dashboardIncidentQueryParsesAllSupportedFilters() throws {
    let query = try DashboardBridge.incidentQuery(
        from: "/admin/api/incidents?status=recovery_failed&model=Tiny%204bit&reason=model_snapshot_invalid&from=2026-08-01T00:00:00Z&to=2026-08-10T00:00:00.000Z&limit=7"
    )

    #expect(query.status == .recoveryFailed)
    #expect(query.model == "Tiny 4bit")
    #expect(query.reason == .modelSnapshotInvalid)
    #expect(query.from != nil)
    #expect(query.to != nil)
    #expect(query.limit == 7)
}

@Test @MainActor
func dashboardIncidentQueryRejectsUnknownStructuredFilters() {
    #expect(throws: (any Error).self) {
        try DashboardBridge.incidentQuery(
            from: "/admin/api/incidents?status=whatever"
        )
    }
    #expect(throws: (any Error).self) {
        try DashboardBridge.incidentQuery(
            from: "/admin/api/incidents?reason=raw-log-guess"
        )
    }
}

@Test @MainActor
func dashboardIncidentPayloadUsesStableVersionedJSONKeys() throws {
    let root = URL(fileURLWithPath: "/private/tmp", isDirectory: true)
        .appendingPathComponent("ironmlx-dashboard-incident-api-\(UUID().uuidString)")
    let store = BackendIncidentStore(url: root.appendingPathComponent("incidents.json"))
    let record = BackendIncidentRecord(
        termination: BackendProcessTermination(
            occurredAt: Date(timeIntervalSince1970: 100),
            launchID: UUID(),
            generation: 2,
            pid: 42,
            terminationStatus: 9,
            terminationReason: "uncaught_signal",
            stopIntent: .unexpected,
            logTail: "tail"
        ),
        snapshot: BackendRecoverySnapshot(config: AppConfig(), models: [])
    )
    try store.upsert(record)

    let listData = try JSONEncoder.ironMLXIncident.encode(
        store.listPayload(matching: BackendIncidentQuery())
    )
    let listJSON = try #require(JSONSerialization.jsonObject(with: listData) as? [String: Any])
    let incidents = try #require(listJSON["incidents"] as? [[String: Any]])
    let summary = try #require(incidents.first)
    #expect(listJSON["schema_version"] as? Int == BackendIncidentRecord.currentSchemaVersion)
    #expect(listJSON["retention_limit"] as? Int == store.retainedIncidents)
    #expect(summary["incident_type"] as? String == "unexpected_backend_exit")
    #expect(summary["recovery_status"] as? String == "recovery_pending")
    #expect(summary["occurred_at"] as? String != nil)

    let detail = try #require(store.detail(id: record.id))
    let detailData = try JSONEncoder.ironMLXIncident.encode(
        BackendIncidentDetailPayload(incident: detail)
    )
    let detailJSON = try #require(JSONSerialization.jsonObject(with: detailData) as? [String: Any])
    let incident = try #require(detailJSON["incident"] as? [String: Any])
    #expect(incident["launch_id"] as? String == record.launchID.uuidString)
    #expect(incident["log_tail"] as? String == "tail")
}

@Test func dashboardLogsPageIncludesIncidentHistoryWithoutReplacingRuntimeLogs() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(html.contains(#"id="log-tab-runtime""#))
    #expect(html.contains(#"id="log-tab-incidents""#))
    #expect(html.contains(#"id="incident-unread-badge""#))
    #expect(html.contains("function switchLogTab(tab)"))
    #expect(html.contains("function renderIncidentHistory(payload)"))
    #expect(html.contains("function renderIncidentDetail(incident)"))
    #expect(html.contains("/admin/api/incidents/export"))
    #expect(html.contains("/admin/api/incidents/clear"))
    #expect(html.contains("switchLogTab('runtime');"))
    #expect(html.contains("function onServerCrash(phase, detail)"))
    #expect(html.contains("showServerBanner"))
    #expect(html.contains("function exportCurrentLog()"))
}

@Test func dashboardIncidentHistoryIsLocalizedForEverySupportedLanguage() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    for key in [
        "runtime_logs:",
        "incident_history:",
        "incident_status_stopped:",
        "recovery_reason_breaker:",
        "incident_clear_confirm:",
        "incident_export_failed:",
    ] {
        #expect(html.components(separatedBy: key).count - 1 == 5, "missing locale for \(key)")
    }
}
