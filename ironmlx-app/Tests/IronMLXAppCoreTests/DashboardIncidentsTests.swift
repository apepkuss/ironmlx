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
func dashboardSeparatesFullDiagnosticsFromFilteredIncidentExport() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(DashboardBridge.handlerNames.contains("exportDiagnosticBundle"))
    #expect(html.contains(#"id="diagnostic-export-btn""#))
    #expect(html.contains("messageHandlers.exportDiagnosticBundle"))
    #expect(html.contains("handler.postMessage('export')"))
    #expect(html.contains("const path = incidentFilterPath('/admin/api/incidents/export')"))
    #expect(html.contains("if (path) apiFetch(path)"))
    #expect(!html.contains("exportDiagnosticBundle(incidentFilterPath"))
    #expect(html.contains(#"aria-describedby="diagnostic-export-tooltip""#))
    #expect(html.contains(#"aria-describedby="incident-export-tooltip""#))
    #expect(html.contains(".diagnostic-export-help:focus-within"))
    #expect(html.contains("diagnosticExportBusy"))
    #expect(html.contains("button.disabled = diagnosticExportBusy"))
    #expect(html.contains(#"incident_export: "Export Incident Records""#))
    #expect(html.contains(#"incident_export: "导出故障记录""#))
    #expect(html.contains(#"incident_export: "匯出故障記錄""#))
    #expect(html.contains(#"incident_export: "障害記録をエクスポート""#))
    #expect(html.contains(#"incident_export: "장애 기록 내보내기""#))
    #expect(!html.contains(#"incident_export: "导出故障信息""#))

    for key in [
        "diagnostic_export:", "diagnostic_export_collecting:",
        "diagnostic_export_tooltip:", "diagnostic_export_accessible_label:",
        "diagnostic_exported:", "diagnostic_export_cancelled:",
        "incident_export_tooltip:", "incident_export_accessible_label:",
    ] {
        #expect(html.components(separatedBy: key).count - 1 == 5, "missing locale for \(key)")
    }
}

@Test
func dashboardIncidentTimesUseExplicit24HourControlsAndFormatting() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(!html.contains(#"type="datetime-local" id="incident-filter-from""#))
    #expect(!html.contains(#"type="datetime-local" id="incident-filter-to""#))
    for id in [
        "incident-filter-from-date",
        "incident-filter-from-time",
        "incident-filter-to-date",
        "incident-filter-to-time",
    ] {
        #expect(html.contains(#"id="\#(id)""#), "missing 24-hour filter control \(id)")
    }
    #expect(html.contains(#"class="incident-datetime-control""#))
    #expect(html.contains(#"class="incident-time-input""#))
    #expect(html.contains(#"placeholder="HH:mm""#))
    #expect(html.contains("normalizeIncidentTimeInput(input)"))
    #expect(html.contains("hour > 23 || minute < 0 || minute > 59"))
    #expect(html.contains("initializeIncidentDateTimeFilter('from', '00:00')"))
    #expect(html.contains("initializeIncidentDateTimeFilter('to', '23:59')"))
    #expect(html.components(separatedBy: "synchronizeTimeAvailability();").count - 1 == 1)
    #expect(html.contains("const fromValue = incidentFilterDateTimeValue('from')"))
    #expect(html.contains("const toValue = incidentFilterDateTimeValue('to')"))
    #expect(html.contains("pad(date.getHours()) + ':'"))
    #expect(!html.contains("hourCycle: 'h23'"))

    for key in [
        "incident_from_date_accessible_label:",
        "incident_from_time_accessible_label:",
        "incident_to_date_accessible_label:",
        "incident_to_time_accessible_label:",
        "incident_time_invalid:",
    ] {
        #expect(html.components(separatedBy: key).count - 1 == 5, "missing locale for \(key)")
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
