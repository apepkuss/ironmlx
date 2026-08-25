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

    @Test("all dashboard locale dictionaries expose the same keys")
    func dashboardLocaleDictionariesHaveMatchingKeys() throws {
        let html = try String(
            contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
            encoding: .utf8
        )
        let markers = [
            "    en: {",
            "    \"zh-Hans\": {",
            "    \"zh-Hant\": {",
            "    ja: {",
            "    ko: {",
            "  const LANG_MAP =",
        ]
        let names = ["en", "zh-Hans", "zh-Hant", "ja", "ko"]
        let keyPattern = try NSRegularExpression(
            pattern: #"(?:^\s+|,\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\""#,
            options: .anchorsMatchLines
        )
        var localeKeys: [String: Set<String>] = [:]

        for index in names.indices {
            let start = try #require(html.range(of: markers[index]))
            let end = try #require(
                html.range(of: markers[index + 1], range: start.upperBound..<html.endIndex)
            )
            let segment = String(html[start.lowerBound..<end.lowerBound])
            let range = NSRange(segment.startIndex..<segment.endIndex, in: segment)
            let keys = keyPattern.matches(in: segment, range: range).compactMap { match -> String? in
                guard let keyRange = Range(match.range(at: 1), in: segment) else {
                    return nil
                }
                return String(segment[keyRange])
            }
            localeKeys[names[index]] = Set(keys)
        }

        let englishKeys = try #require(localeKeys["en"])
        for name in names.dropFirst() {
            #expect(localeKeys[name] == englishKeys, "\(name) locale keys differ from English")
        }

        var referencedKeys = Set<String>()
        for pattern in [
            #"data-i18n(?:-placeholder|-aria-label|-title)?="([^"]+)""#,
            #"\bt\(\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]"#,
        ] {
            let referencePattern = try NSRegularExpression(pattern: pattern)
            let range = NSRange(html.startIndex..<html.endIndex, in: html)
            for match in referencePattern.matches(in: html, range: range) {
                guard let keyRange = Range(match.range(at: 1), in: html) else {
                    continue
                }
                referencedKeys.insert(String(html[keyRange]))
            }
        }
        #expect(
            referencedKeys.isSubset(of: englishKeys),
            "Dashboard references missing locale keys: \(referencedKeys.subtracting(englishKeys).sorted())"
        )
    }

    @Test("dashboard never renders opaque backend errors directly")
    func dashboardDoesNotRenderOpaqueErrorsDirectly() throws {
        let html = try String(
            contentsOfFile: "Sources/IronMLXAppCore/Resources/dashboard2.html",
            encoding: .utf8
        )
        let bridge = try String(
            contentsOfFile: "Sources/IronMLXAppCore/DashboardBridge.swift",
            encoding: .utf8
        )

        for forbidden in [
            "showToast(result.error ||",
            "showToast(status.error ||",
            "replace('{detail}', status.error ||",
            "replace('{detail}', schedulerProfilePreview.error)",
            ": (result.error || dict.",
        ] {
            #expect(!html.contains(forbidden), "Opaque error sink remains: \(forbidden)")
        }
        #expect(!bridge.contains("sendJavaScript(\"showToast("))
        #expect(bridge.contains("code: result.errorCode ?? \"operation_failed\""))
        #expect(html.contains("localizeErrorResult(result, fallbackKey)"))
        #expect(html.components(separatedBy: "err_bundled_runtime_invalid:").count - 1 == 5)
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
