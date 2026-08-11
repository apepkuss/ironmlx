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
    #expect(html.components(separatedBy: #"id="diagnostic-export-btn""#).count - 1 == 1)
    #expect(html.contains("messageHandlers.exportDiagnosticBundle"))
    #expect(html.contains("handler.postMessage('export')"))
    #expect(html.contains("function incidentExportPathForQuery(path)"))
    #expect(html.contains("params.delete('limit')"))
    #expect(html.contains("appliedIncidentExportPath"))
    #expect(html.contains("pendingIncidentQueries"))
    #expect(html.contains("incidentFilterChangeGeneration !== appliedIncidentFilterGeneration"))
    #expect(!html.contains("incidentFilterPath('/admin/api/incidents/export')"))
    #expect(!html.contains("exportDiagnosticBundle(incidentFilterPath"))
    #expect(html.contains(#"aria-describedby="diagnostic-export-tooltip""#))
    #expect(html.contains(#"aria-describedby="incident-export-tooltip""#))
    #expect(html.contains(".diagnostic-export-help:focus-within"))
    #expect(html.contains("diagnosticExportBusy"))
    #expect(html.contains("button.disabled = diagnosticExportBusy"))
    #expect(html.contains(#"class="incident-history-toolbar" role="toolbar""#))
    #expect(html.contains(#"class="incident-history-export-actions""#))
    #expect(html.contains(#"class="card-header incident-filter-header""#))
    #expect(html.contains(#"class="incident-filter-actions""#))
    #expect(html.contains("flex-wrap: wrap"))

    let toolbarStyleStart = try #require(html.range(of: ".incident-history-toolbar {"))
    let filterActionsStyleStart = try #require(
        html.range(of: ".incident-filter-actions {", range: toolbarStyleStart.upperBound ..< html.endIndex)
    )
    let toolbarStyles = String(html[toolbarStyleStart.lowerBound ..< filterActionsStyleStart.lowerBound])
    #expect(toolbarStyles.contains("gap: 16px"))
    #expect(toolbarStyles.contains(".incident-history-export-actions"))
    #expect(toolbarStyles.contains("gap: 8px"))
    #expect(toolbarStyles.contains(".incident-history-toolbar button"))
    #expect(toolbarStyles.contains("height: 34px"))
    #expect(toolbarStyles.contains("align-items: center"))
    #expect(toolbarStyles.contains(".incident-history-toolbar #incident-clear-btn"))
    #expect(toolbarStyles.contains("border: 1px solid transparent"))
    #expect(toolbarStyles.contains("background-clip: padding-box"))

    let queryIndex = try #require(html.range(of: #"id="incident-refresh-btn""#))
    let incidentExportIndex = try #require(html.range(of: #"id="incident-export-btn""#))
    let diagnosticExportIndex = try #require(html.range(of: #"id="diagnostic-export-btn""#))
    let clearHistoryIndex = try #require(html.range(of: #"id="incident-clear-btn""#))
    #expect(incidentExportIndex.lowerBound < diagnosticExportIndex.lowerBound)
    #expect(diagnosticExportIndex.lowerBound < clearHistoryIndex.lowerBound)
    #expect(clearHistoryIndex.lowerBound < queryIndex.lowerBound)
    #expect(html.components(separatedBy: #"id="incident-refresh-btn""#).count - 1 == 1)
    #expect(html.contains(#"data-i18n="incident_query">Query</button>"#))
    #expect(html.contains(#"incident_export: "Export Incident Records""#))
    #expect(html.contains(#"incident_export: "导出故障记录""#))
    #expect(html.contains(#"incident_export: "匯出故障記錄""#))
    #expect(html.contains(#"incident_export: "障害記録をエクスポート""#))
    #expect(html.contains(#"incident_export: "장애 기록 내보내기""#))
    #expect(!html.contains(#"incident_export: "导出故障信息""#))

    for key in [
        "diagnostic_export:", "diagnostic_export_collecting:",
        "diagnostic_export_tooltip:", "diagnostic_export_accessible_label:",
        "diagnostic_exported:",
        "incident_export_tooltip:", "incident_export_accessible_label:",
        "incident_history_actions_accessible_label:",
        "incident_export_all_title:", "incident_export_all_message:",
        "incident_export_unapplied_message:", "incident_export_all_confirm:",
        "incident_export_current_title:", "incident_export_current_message:",
        "incident_export_current_confirm:", "incident_return_to_filters:",
        "incident_clear_title:",
    ] {
        #expect(html.components(separatedBy: key).count - 1 == 5, "missing locale for \(key)")
    }

    #expect(html.contains("showConfirmDialog(message"))
    #expect(html.contains("showConfirmDialog(\n      dict.incident_clear_confirm"))
    #expect(html.contains("此操作将永久删除所有故障历史记录，删除后无法恢复。运行日志不会受到影响。"))
    #expect(!html.contains("diagnostic_export_cancelled"))
    #expect(html.contains("result.status !== 'cancelled' && result.status !== 'busy'"))
    #expect(html.contains("danger: true"))
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
    #expect(html.contains(#"id="incident-filter-from-date" required"#))
    #expect(html.contains(#"id="incident-filter-to-date" required"#))
    #expect(html.contains(#"class="incident-datetime-control""#))
    #expect(html.contains(#"class="incident-time-input""#))
    #expect(html.contains(#"placeholder="HH:mm""#))
    #expect(html.contains(#"id="incident-filter-from-time""#))
    #expect(html.contains(#"maxlength="5" value="00:00""#))
    #expect(html.contains(#"id="incident-filter-to-time""#))
    #expect(html.contains(#"maxlength="5" value="23:59""#))
    #expect(!html.contains(#"data-i18n-aria-label="incident_from_time_accessible_label" disabled"#))
    #expect(!html.contains(#"data-i18n-aria-label="incident_to_time_accessible_label" disabled"#))
    #expect(html.contains("normalizeIncidentTimeInput(input)"))
    #expect(html.contains("hour > 23 || minute < 0 || minute > 59"))
    #expect(html.contains("initializeIncidentDateTimeFilter('from', '00:00')"))
    #expect(html.contains("initializeIncidentDateTimeFilter('to', '23:59')"))
    #expect(html.contains("function initializeIncidentDateDefaults(payload)"))
    #expect(html.contains("payload.oldest_retained_occurred_at || null"))
    #expect(html.contains("if (incidentDateDefaultsInitialized) return"))
    #expect(html.contains("if (!incidentDateFilterTouched.from)"))
    #expect(html.contains("if (!incidentDateFilterTouched.to)"))
    #expect(html.contains("initializeIncidentDateDefaults(payload)"))
    #expect(html.contains("if (!incidentDateDefaultsInitialized) return ''"))
    #expect(html.contains("dateInput.reportValidity()"))
    #expect(html.contains("const fromValue = incidentFilterDateTimeValue('from')"))
    #expect(html.contains("const toValue = incidentFilterDateTimeValue('to')"))
    #expect(html.contains("const isEnd = kind === 'to'"))
    #expect(html.contains("isEnd ? 59 : 0, isEnd ? 999 : 0"))
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
    #expect(listJSON["oldest_retained_occurred_at"] as? String != nil)
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
        "incident_query:",
        "incident_status_stopped:",
        "recovery_reason_breaker:",
        "incident_clear_confirm:",
        "incident_export_failed:",
    ] {
        #expect(html.components(separatedBy: key).count - 1 == 5, "missing locale for \(key)")
    }
}
