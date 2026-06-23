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

    #expect(html.contains(#"id="log-file-select""#))
    #expect(html.contains(#"<option value="server">ironmlx-server</option>"#))
    #expect(html.contains(#"<option value="app">ironmlx-app</option>"#))
    #expect(html.contains(#"id="log-refresh-btn""#))
    #expect(html.contains(#"id="log-export-link""#))
    #expect(html.contains("function exportCurrentLog()"))
    #expect(html.contains("logLinesInput.addEventListener('input', applyLogFilters)"))
    #expect(html.contains("if (page === 'logs')"))
    #expect(html.contains("initLogs();"))
    #expect(html.contains("stopLogAutoRefresh();"))
    #expect(html.contains("levelAliases"))
    #expect(html.contains("'WARNING': ['WARNING', 'WARN']"))
}
