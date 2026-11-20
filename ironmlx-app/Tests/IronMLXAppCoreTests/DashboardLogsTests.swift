import Foundation
import Testing

@testable import IronMLXAppCore

@MainActor
@Test func dashboardBridgeRegistersDashboardLogHandler() {
    #expect(DashboardBridge.handlerNames.contains("dashboardLog"))
}

@Test func dashboardLogsPageWiresAppAndBackendLogControls() throws {
    let html = try String(
        contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
        encoding: .utf8
    )

    #expect(html.contains(#"id="log-source-seg""#))
    #expect(html.contains(#"data-value="server""#))
    #expect(html.contains(#"data-value="app""#))
    #expect(html.contains(#"id="log-level-chips""#))
    #expect(html.contains(#"id="log-pause-btn""#))
    #expect(html.contains(#"id="log-export-btn""#))
    #expect(html.contains("function exportCurrentLog()"))
    #expect(html.contains("if (page === 'logs')"))
    #expect(html.contains("initLogs();"))
    #expect(html.contains("stopLogAutoRefresh();"))
    #expect(html.contains("function stripLogAnsi("))
    #expect(!html.contains("ironmlx-server</option>"))
    #expect(!html.contains("levelAliases"))
    #expect(!html.contains("2026-03-20 15:29:31"))
}
