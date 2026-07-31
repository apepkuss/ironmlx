import Foundation
import JavaScriptCore
import Testing

@testable import IronMLXAppCore

@Suite("Dashboard JavaScript syntax")
struct DashboardJavaScriptSyntaxTests {
    @Test("dashboard script compiles")
    func dashboardScriptCompiles() throws {
        let html = try String(
            contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
            encoding: .utf8
        )
        let script = try #require(extractFirstScript(from: html))
        let context = JSContext()!

        context.setObject(script, forKeyedSubscript: "dashboardScript" as NSString)
        let result = context.evaluateScript(
            """
            try {
              new Function(dashboardScript);
              ({ ok: true, error: null });
            } catch (error) {
              ({ ok: false, error: String(error && error.message ? error.message : error) });
            }
            """
        )

        let ok = result?.forProperty("ok")?.toBool() ?? false
        let error = result?.forProperty("error")?.toString() ?? "unknown JavaScript syntax error"
        #expect(ok, "dashboard2.html JavaScript syntax error: \(error)")
    }

    private func extractFirstScript(from html: String) -> String? {
        guard let startRange = html.range(of: "<script>"),
              let endRange = html[startRange.upperBound...].range(of: "</script>")
        else {
            return nil
        }
        return String(html[startRange.upperBound..<endRange.lowerBound])
    }
}
